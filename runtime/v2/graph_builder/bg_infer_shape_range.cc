/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_infer_shape_range.h"
#include "securec.h"
#include "common/checker.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "exe_graph/lowering/value_holder.h"
#include "storage_format.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "bg_compatible_utils.h"
#include "framework/common/types.h"
#include "register/node_converter_registry.h"
#include "bg_model_desc.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
namespace bg {
namespace {
constexpr size_t kMaxShapeIndex = 2U;
bool NeedInferShapeRangeCompatible(const std::string &type, const gert::OpImplSpaceRegistryV2Ptr &space_registry) {
  if (space_registry == nullptr) {
    return true;
  }
  auto op_funcs = space_registry->GetOpImpl(type.c_str());
  if ((op_funcs == nullptr) || (op_funcs->infer_shape_range == nullptr)) {
    return true;
  }
  return false;
}
}
ShapeRangeInferenceResult::ShapeRangeInferenceResult() : status_(), all_outputs_(0U), compute_node_output_num_() {}

ShapeRangeInferenceResult::ShapeRangeInferenceResult(size_t outputs_num, vector<ValueHolderPtr> &out_holder) :
    status_(), all_outputs_(out_holder), compute_node_output_num_(outputs_num) {
}

std::vector<ValueHolderPtr> ShapeRangeInferenceResult::GetAllMaxShapes() {
  std::vector<ValueHolderPtr> max_shapes;
  max_shapes.insert(max_shapes.cbegin(), all_outputs_.cbegin() + kMaxShapeIndex * compute_node_output_num_,
                    all_outputs_.cend());
  return max_shapes;
}

std::vector<ValueHolderPtr> ShapeRangeInferenceResult::GetAllShapeRanges() {
  std::vector<ValueHolderPtr> shape_ranges;
  shape_ranges.insert(shape_ranges.cbegin(), all_outputs_.cbegin(), all_outputs_.cbegin() + compute_node_output_num_);
  return shape_ranges;
}

std::vector<ValueHolderPtr> ShapeRangeInferenceResult::GetAllMinShapes() {
  std::vector<ValueHolderPtr> min_shapes;
  min_shapes.insert(min_shapes.cbegin(), all_outputs_.cbegin() + compute_node_output_num_,
                    all_outputs_.cbegin() + kMaxShapeIndex * compute_node_output_num_);
  return min_shapes;
}

bool ShapeRangeInferenceResult::IsSuccess() const {
  return status_.IsSuccess();
}

void ShapeRangeInferenceResult::SetErrorStatus() {
  status_ = HyperStatus::ErrorStatus("error");
}

ShapeRangeInferenceResult ShapeRangeInferenceResult::ErrorResult() {
  ShapeRangeInferenceResult result;
  result.SetErrorStatus();
  return result;
}

/*
 *                            InferShapeRange
 *                            /              \
 * all-tensor_ranges and shape-ranges      FindInferShapeRangeFunc
 *                                             /           \
 *                                          node-type   space_registry
 */
std::vector<ValueHolderPtr> BuildInferShapeRangeGraph(const ge::NodePtr &node,
                                                      const std::vector<ValueHolderPtr> &input_ranges,
                                                      const ValueHolderPtr &space_registry) {
  std::string type;
  if (ge::GetOriginalType(node, type) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get original type from %s(%s).", node->GetName().c_str(), node->GetType().c_str());
    return {};
  }
  GE_ASSERT_NOTNULL(space_registry);
  auto node_type = ValueHolder::CreateConst(type.c_str(), type.size() + 1, true);
  auto infer_func = ValueHolder::CreateSingleDataOutput("FindInferShapeRangeFunc",{node_type, space_registry});
  auto inputs = input_ranges;
  inputs.emplace_back(infer_func);
  /*
   * 3n个输出，0~n-1 ShapeRange, n~2n-1 min shape，2n~3n-1 max shape,n是node输出个数
   */
  return ValueHolder::CreateDataOutput("InferShapeRange", inputs, node->GetAllOutDataAnchorsSize() * 3U);
}

/*
 *        CompatibleInferShapeRange
 *            /       \     \______________________
 *           /         |                           |
 * all-shapes   FindCompatibleInferShapeRangeFunc    operator
 *                      |
 *                   node-type
 */
std::vector<ValueHolderPtr> BuildCompatibleInferShapeRangeGraph(const ge::NodePtr &node,
                                                                const std::vector<ValueHolderPtr> &input_ranges) {
  std::string type;
  if (ge::GetOriginalType(node, type) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get original type from %s(%s).", node->GetName().c_str(), node->GetType().c_str());
    return {};
  }
  auto node_type = ValueHolder::CreateConst(type.c_str(), type.size() + 1, true);
  auto infer_func = ValueHolder::CreateSingleDataOutput("FindCompatibleInferShapeRangeFunc", {node_type});

  auto op_buffer_vec = CompatibleUtils::BuildOpDescBufferConst(node);
  auto op = ValueHolder::CreateSingleDataOutput("CreateOpFromBuffer", op_buffer_vec);

  std::vector<ValueHolderPtr> inputs;
  inputs.emplace_back(op);
  inputs.emplace_back(infer_func);
  inputs.insert(inputs.cend(), input_ranges.cbegin(), input_ranges.cend());
  /*
   * 3n个输出，0~n-1 ShapeRange, n~2n-1 min shape，2n~3n-1 max shape,n是node输出个数
   */
  return ValueHolder::CreateDataOutput("CompatibleInferShapeRange", inputs, node->GetAllOutDataAnchorsSize() * 3U);
}

ShapeRangeInferenceResult InferShapeRange(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                          const LoweringGlobalData &global_data) {
  if (node == nullptr) {
    return ShapeRangeInferenceResult::ErrorResult();
  }
  auto real_inputs_size = node->GetInDataNodesAndAnchors().size();
  if (input_shapes.size() != real_inputs_size) {
    GELOGE(ge::FAILED, "Failed to generate InferShapeRange node for node %s, the input shape ranges size %zu,"
                       " node input size %zu", node->GetName().c_str(), input_shapes.size(), real_inputs_size);
    return ShapeRangeInferenceResult::ErrorResult();
  }

  auto ranges = ValueHolder::CreateDataOutput("CreateTensorRangesAndShapeRanges", input_shapes,
                                              input_shapes.size());
  // To compatible with old version infer_shape_range_fun, build different exe graph for infer_shape_range
  std::string type;
  if (ge::GetOriginalType(node, type) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get original type from %s(%s).", node->GetName().c_str(), node->GetType().c_str());
    return {};
  }
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  std::vector<ValueHolderPtr> infer_shape_range_ret;
  if (NeedInferShapeRangeCompatible(type, global_data.GetSpaceRegistryV2(static_cast<gert::OppImplVersionTag>(
                                              node->GetOpDesc()->GetOppImplVersion())))) {
    GELOGD("Node %s type %s not support v2 infer_shape_range. Turns to v1 infer_shape_range.", node->GetNamePtr(),
           type.c_str());
    infer_shape_range_ret = BuildCompatibleInferShapeRangeGraph(node, ranges);
  } else {
    auto opp_impl_version = node->GetOpDesc()->GetOppImplVersion();
    infer_shape_range_ret = BuildInferShapeRangeGraph(node, ranges,bg::GetSpaceRegistry(global_data, opp_impl_version));
  }
  return ShapeRangeInferenceResult(node->GetAllOutDataAnchorsSize(), infer_shape_range_ret);
}

std::vector<ValueHolderPtr> InferMaxShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                          const LoweringGlobalData &global_data) {
  if (node == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "node is null");
    return {};
  }
  auto result = InferShapeRange(node, input_shapes, global_data);
  if (!result.IsSuccess()) {
    GELOGE(ge::GRAPH_FAILED, "infer shape range failed. node: %s(%s)",
           node->GetName().c_str(), node->GetType().c_str());
    return {};
  }
  return result.GetAllMaxShapes();
}
}  // namespace bg
}  // namespace gert
