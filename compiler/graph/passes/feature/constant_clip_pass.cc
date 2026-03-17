/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/constant_clip_pass.h"
#include <map>
#include <string>
#include <climits>
#include <cfloat>
#include <cmath>
#include "graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "common/util/mem_utils.h"
#include "graph/utils/type_utils.h"

namespace {
  const std::string Enable_Clip = "1";
  constexpr ge::float32_t FP16_MAX_VALUE = 65504;
  constexpr ge::float32_t FP16_MIN_VALUE = -65504;
  constexpr ge::float32_t BF16_MAX_VALUE = 3.38E38f;
  constexpr ge::float32_t BF16_MIN_VALUE = -3.38E38f;
  constexpr ge::float32_t HIF8_MAX_VALUE = std::pow(2, 15);
  constexpr ge::float32_t E5M2_MAX_VALUE = 1.75 * std::pow(2, 15);
  constexpr ge::float32_t E4M3FN_MAX_VALUE = 1.75 * std::pow(2, 8);
  bool IsFloatDt(const ge::DataType &dt) {
    return (dt == ge::DT_FLOAT) || (dt == ge::DT_FLOAT16) || (dt == ge::DT_DOUBLE) || (dt == ge::DT_BF16) ||
           (dt == ge::DT_HIFLOAT8) || (dt == ge::DT_FLOAT8_E5M2) || (dt == ge::DT_FLOAT8_E4M3FN);
  }

  ge::float32_t GetMax(const ge::DataType dt) {
    static std::map<ge::DataType, ge::float32_t> dt_max_map = {{ge::DT_FLOAT16, FP16_MAX_VALUE},
        {ge::DT_BF16, BF16_MAX_VALUE}, {ge::DT_FLOAT, FLT_MAX}, {ge::DT_HIFLOAT8, HIF8_MAX_VALUE},
        {ge::DT_FLOAT8_E5M2, E5M2_MAX_VALUE}, {ge::DT_FLOAT8_E4M3FN, E4M3FN_MAX_VALUE},
    };
    return dt_max_map[dt];
  }

  ge::float32_t GetMin(const ge::DataType dt) {
    static std::map<ge::DataType, ge::float32_t> dt_min_map = {{ge::DT_FLOAT16, FP16_MIN_VALUE},
        {ge::DT_BF16, BF16_MIN_VALUE}, {ge::DT_FLOAT, -FLT_MAX}, {ge::DT_HIFLOAT8, -HIF8_MAX_VALUE},
        {ge::DT_FLOAT8_E5M2, -E5M2_MAX_VALUE}, {ge::DT_FLOAT8_E4M3FN, -E4M3FN_MAX_VALUE},
    };
    return dt_min_map[dt];
  }
}

namespace ge {
Status ConstantClipPass::Run(NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  if (node->GetType() != CAST) {
    return SUCCESS;
  }
  std::string is_weight_clip;
  bool need_weight_clip = (GetContext().GetOption("ge.is_weight_clip", is_weight_clip) == SUCCESS) &&
                          (is_weight_clip == Enable_Clip);
  if (!need_weight_clip) {
    return SUCCESS;
  }
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto in_tensor_desc = op_desc->MutableInputDesc(0U);
  GE_ASSERT_NOTNULL(in_tensor_desc);
  const DataType in_dt = in_tensor_desc->GetDataType();
  auto out_tensor_desc = op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(out_tensor_desc);
  const DataType out_dt = out_tensor_desc->GetDataType();
  GELOGD("enter ConstantClipPass cast %s, in_dt %s, out_dt %s", node->GetName().c_str(),
      TypeUtils::DataTypeToSerialString(in_dt).c_str(), TypeUtils::DataTypeToSerialString(out_dt).c_str());
  // in_dt can only be float or double
  if (!IsFloatDt(in_dt) || !IsFloatDt(out_dt) || (GetSizeByDataType(in_dt) <= GetSizeByDataType(out_dt))) {
    return SUCCESS;
  }
  auto input_nodes = OpDescUtils::GetConstInputs(*node);
  if (input_nodes.empty()) {
    return SUCCESS;
  }
  GELOGD("find cast op %s, src_dt %s, dst_dt %s, link const input %s", node->GetName().c_str(),
      TypeUtils::DataTypeToSerialString(in_dt).c_str(), TypeUtils::DataTypeToSerialString(out_dt).c_str(),
      input_nodes[0]->GetName().c_str());
  auto weight_inputs = OpDescUtils::GetWeights(*(input_nodes[0]));
  if (!weight_inputs.empty()) {
    bool is_infinite = false;
    GE_ASSERT_SUCCESS(CheckWeightInfinite(weight_inputs[0], out_dt, is_infinite));
    if (!is_infinite) {
      return SUCCESS;
    }
    GELOGW("find cast op %s, src_dt %s, dst_dt %s need to be cliped, it may be lower accuracy",
        node->GetName().c_str(), TypeUtils::DataTypeToSerialString(in_dt).c_str(),
        TypeUtils::DataTypeToSerialString(out_dt).c_str());
    GE_ASSERT_SUCCESS(InsertClipByValueBetweenCastAndConst(node));
  }
  return SUCCESS;
}

Status ConstantClipPass::CheckWeightInfinite(const ConstGeTensorPtr &const_tensor,
    const DataType &dst_dt, bool &is_infinite) const {
  is_infinite = false;
  const DataType const_dt = const_tensor->GetTensorDesc().GetDataType();
  const auto min_value = GetMin(dst_dt);
  const auto max_value = GetMax(dst_dt);
  if (const_dt == DT_FLOAT) {
    const float32_t *const shape_data = reinterpret_cast<const float32_t *>(const_tensor->GetData().GetData());
    GE_ASSERT_NOTNULL(shape_data);
    const size_t dims_num = const_tensor->GetData().size() / sizeof(float32_t);
    for (size_t i = 0UL; i < dims_num; ++i) {
      if ((shape_data[i] > max_value) || (shape_data[i] < min_value)) {
        is_infinite = true;
        GELOGI("find float32_t value %f infinite", shape_data[i]);
        return SUCCESS;
      }
    }
  } else if (const_dt == DT_DOUBLE) {
    const float64_t *const shape_data = reinterpret_cast<const float64_t *>(const_tensor->GetData().GetData());
    GE_ASSERT_NOTNULL(shape_data);
    const size_t dims_num = const_tensor->GetData().size() / sizeof(float64_t);
    for (size_t i = 0UL; i < dims_num; ++i) {
      if ((shape_data[i] > static_cast<float64_t>(max_value)) || (shape_data[i] < static_cast<float64_t>(min_value))) {
        is_infinite = true;
        GELOGI("find float64_t value %lf infinite", shape_data[i]);
        return SUCCESS;
      }
    }
  }
  return SUCCESS;
}

Status ConstantClipPass::InsertClipByValueBetweenCastAndConst(const NodePtr &node) {
  auto in_data_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(in_data_anchor);
  auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_data_anchor);
  auto src_node = out_data_anchor->GetOwnerNode();
  NodePtr clip_node = CreateClipNode(node);
  GE_ASSERT_NOTNULL(clip_node);
  GE_ASSERT_SUCCESS(GraphUtils::InsertNodeBetweenDataAnchors(out_data_anchor, in_data_anchor, clip_node));
  GELOGI("insert clip node %s between %s and %s successfully",
      clip_node->GetName().c_str(), src_node->GetName().c_str(), node->GetName().c_str());
  return SUCCESS;
}

NodePtr ConstantClipPass::CreateClipMinMaxNode(const NodePtr &node, const bool &is_min) const {
  const auto src_tensor_desc = node->GetOpDesc()->MutableInputDesc(0);
  const auto dst_tensor_desc = node->GetOpDesc()->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(src_tensor_desc);
  GE_ASSERT_NOTNULL(dst_tensor_desc);
  const DataType src_dt = src_tensor_desc->GetDataType();
  const DataType dst_dt = dst_tensor_desc->GetDataType();
  GeShape shape;
  GeTensorDesc const_desc(shape, FORMAT_ND, src_dt);
  GeTensorPtr const_tensor = nullptr;
  if (src_dt == DT_FLOAT) {
    if (is_min) {
      float32_t min_val = GetMin(dst_dt);
      const_tensor = MakeShared<GeTensor>(const_desc, reinterpret_cast<uint8_t *>(&min_val), sizeof(float32_t));
    } else {
      float32_t max_val = GetMax(dst_dt);
      const_tensor = MakeShared<GeTensor>(const_desc, reinterpret_cast<uint8_t *>(&max_val), sizeof(float32_t));
    }
  } else if (src_dt == DT_DOUBLE) {
    if (is_min) {
      float64_t min_val = static_cast<float64_t>(GetMin(dst_dt));
      const_tensor = MakeShared<GeTensor>(const_desc, reinterpret_cast<uint8_t *>(&min_val), sizeof(float64_t));
    } else {
      float64_t max_val = static_cast<float64_t>(GetMax(dst_dt));
      const_tensor = MakeShared<GeTensor>(const_desc, reinterpret_cast<uint8_t *>(&max_val), sizeof(float64_t));
    }
  }
  GE_ASSERT_NOTNULL(const_tensor);
  OpDescPtr const_op = OpDescUtils::CreateConstOp(const_tensor);
  GE_ASSERT_NOTNULL(const_op);
  auto graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph);
  return graph->AddNode(const_op);
}

NodePtr ConstantClipPass::CreateClipNode(const NodePtr &node) {
  // construct min and max const op
  auto const_min_node = CreateClipMinMaxNode(node, true);
  GE_ASSERT_NOTNULL(const_min_node);
  auto const_max_node = CreateClipMinMaxNode(node, false);
  GE_ASSERT_NOTNULL(const_max_node);

  // construct clipbyvalue 3 inputs 1 output
  std::string name = node->GetName() + "_const_clip";
  auto clip_desc = MakeShared<OpDesc>(name, "ClipByValue");
  GE_ASSERT_NOTNULL(clip_desc);
  GE_ASSERT_GRAPH_SUCCESS(clip_desc->AddInputDesc(*node->GetOpDesc()->MutableInputDesc(0)));
  GE_ASSERT_GRAPH_SUCCESS(clip_desc->AddInputDesc(const_min_node->GetOpDesc()->GetOutputDesc(0)));
  GE_ASSERT_GRAPH_SUCCESS(clip_desc->AddInputDesc(const_max_node->GetOpDesc()->GetOutputDesc(0)));
  GE_ASSERT_GRAPH_SUCCESS(clip_desc->AddOutputDesc(*node->GetOpDesc()->MutableInputDesc(0)));
  auto graph = node->GetOwnerComputeGraph();
  auto clip_node = graph->AddNode(clip_desc);
  GE_ASSERT_NOTNULL(clip_node);

  // link min and max to clipbyvalue
  auto min_out_anchor = const_min_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(min_out_anchor);
  auto max_out_anchor = const_max_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(max_out_anchor);
  auto min_in_anchor = clip_node->GetInDataAnchor(1);
  GE_ASSERT_NOTNULL(min_in_anchor);
  auto max_in_anchor = clip_node->GetInDataAnchor(2);
  GE_ASSERT_NOTNULL(max_in_anchor);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(min_out_anchor, min_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(max_out_anchor, max_in_anchor));

  // add all inserted node to repass for constant folding again
  AddRePassNode(const_min_node);
  AddRePassNode(const_max_node);
  AddRePassNode(clip_node);
  GELOGI("create clip node %s, const_min_node %s, const_max_node %s successfully",
      clip_node->GetName().c_str(), const_min_node->GetName().c_str(), const_max_node->GetName().c_str());
  return clip_node;
}

REG_PASS_OPTION("ConstantClipPass").LEVELS(OoLevel::kO3);
}  // namespace ge
