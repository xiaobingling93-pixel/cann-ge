/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inputs_converter.h"
#include <cinttypes>
#include <sstream>
#include <securec.h>
#include "common/checker.h"
#include "common/const_place_holder_utils/const_place_holder_utils.h"
#include "graph/node.h"
#include "graph/def_types.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/value_holder.h"
#include "register/node_converter_registry.h"
#include "kernel/tensor_attr.h"
#include "graph_builder/converter_checker.h"
#include "kernel/common_kernel_impl/build_tensor.h"
#include "graph/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "formats/utils/formats_trans_utils.h"
#include "graph_builder/value_holder_generator.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "common/host_resource_center/host_resource_center.h"
#include "common/host_resource_center/weight_manager.h"
#include "engine/node_converter_utils.h"
#include "utils/utils.h"

namespace gert {
using namespace bg;
constexpr const char *kDataNodeCounter = "DataNodeCounter";
namespace {
template <typename T>
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}
int64_t GetTensorSize(const ge::GeTensor *tensor) {
  auto data_type = tensor->GetTensorDesc().GetDataType();
  if (data_type == ge::DT_STRING) {
    return static_cast<int64_t>(tensor->GetData().GetSize());
  }
  auto shape_size = tensor->GetTensorDesc().GetShape().GetShapeSize();
  if ((tensor->GetTensorDesc().GetShape().IsScalar()) && (tensor->GetData().GetSize() != 0U)) {
    shape_size = 1;
  }
  return ge::GetSizeInBytes(shape_size, data_type);
}
const ge::GeTensor *GetWeightFromResourceCenter(const ge::NodePtr &node, gert::LoweringGlobalData *const global_data) {
  auto host_resource_center = reinterpret_cast<ge::HostResourceCenter *>(global_data->GetHostResourceCenter());
  GE_ASSERT_NOTNULL(host_resource_center);
  auto weight_manager =
      dynamic_cast<const ge::WeightManager *>(host_resource_center->GetHostResourceMgr(ge::HostResourceType::kWeight));
  GE_ASSERT_NOTNULL(weight_manager);
  auto weight_resource = dynamic_cast<const ge::WeightResource *>(weight_manager->GetResource(node->GetOpDesc(), 0));
  GE_ASSERT_NOTNULL(weight_resource);
  return weight_resource->GetWeight();
}
} // namespace
void VecToSmallVec(const std::vector<int64_t> &src, Shape &dst) {
  dst.SetDimNum(0);
  for (const auto &src_data : src) {
    dst.AppendDim(src_data);
  }
}

std::unique_ptr<Tensor> CreateExecuteTensorFromCompute(const ge::GeTensor *tensor) {
  StorageShape exe_shape;
  VecToSmallVec(tensor->GetTensorDesc().GetOriginShape().GetDims(), exe_shape.MutableOriginShape());
  VecToSmallVec(tensor->GetTensorDesc().GetShape().GetDims(), exe_shape.MutableStorageShape());
  // todo pading in storage format
  StorageFormat exe_format(tensor->GetTensorDesc().GetOriginFormat(), tensor->GetTensorDesc().GetFormat(),
                           ExpandDimsType());
  auto exe_tensor =
      ge::MakeUnique<Tensor>(exe_shape, exe_format, kOnHost, tensor->GetTensorDesc().GetDataType(), nullptr);
  if (exe_tensor == nullptr) {
    return nullptr;
  }

  auto tensor_size = GetTensorSize(tensor);
  if (tensor_size == 0) {
    return exe_tensor;
  }
  exe_tensor->MutableTensorData().SetAddr(tensor->GetData().data(), nullptr);
  exe_tensor->MutableTensorData().SetSize(tensor_size);
  return exe_tensor;
}

/**
 *      FreeMemory
 *         |
 *     SplitTensor
 *         |
 *       Const
 * @param node
 * @param lower_input
 * @return
 */
LowerResult LoweringConstNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  if (lower_input.global_data == nullptr) {
    return {HyperStatus::ErrorStatus("Failed to lowering const node, the global data is nullptr"), {}, {}, {}};
  }
  auto ge_tensor_ptr = GetWeightFromResourceCenter(node, lower_input.global_data);
  GE_ASSERT_NOTNULL(ge_tensor_ptr);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  auto outputs = FrameSelector::OnInitRoot([&ge_tensor_ptr, &op_desc]() -> std::vector<ValueHolderPtr> {
    auto exe_tensor = CreateExecuteTensorFromCompute(ge_tensor_ptr);
    auto const_holder = ValueHolder::CreateConst(exe_tensor.get(), sizeof(Tensor));
    const int64_t logical_stream_id = op_desc->GetStreamId();
    auto stream_id_holder = bg::ValueHolder::CreateConst(&logical_stream_id, sizeof(logical_stream_id));
    auto split_outputs = DevMemValueHolder::CreateDataOutput(kernel::kSplitConstTensor, {const_holder, stream_id_holder},
                                                             static_cast<size_t>(kernel::SplitTensorOutputs::kNum), 0);
    if (split_outputs.size() != static_cast<size_t>(kernel::SplitTensorOutputs::kNum)) {
      GELOGE(ge::FAILED, "Failed to create SplitTensor");
      return {};
    }
    auto address = split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)];
    if (address != nullptr) {
      address->SetPlacement(kOnHost);
    }
    std::vector<ValueHolderPtr> ret(split_outputs.begin(), split_outputs.end());
    return ret;
  });
  CONVERTER_CHECK_HOLDERS_ALL_OK(outputs, static_cast<size_t>(kernel::SplitTensorOutputs::kNum));

  // const不返回ordered holders，因为const的lowering结果会产生于Init图上，不依赖于任何人；
  // 而在计算图上，const本身是可能被用来传递控制关系的
  // 由如上两条可以知道：const的lowering是无法传递控制关系的，
  // 因此const的lowering就不返回ordered_holders了，由框架自动传递所有的输入控制，作为const的输出控制
  auto output_addr = std::dynamic_pointer_cast<bg::DevMemValueHolder>(
      outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)]);
  return {
      HyperStatus::Success(), {}, {outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]}, {output_addr}};
}

ge::Status GetConstPlaceHolderAttr(const ge::OpDescPtr &op_desc, void* &data_addr,
                                   int64_t &data_length, StorageShape &out_shape) {
  uint8_t *addr = nullptr;
  GE_ASSERT_GRAPH_SUCCESS(GetConstPlaceHolderAddr(op_desc, addr));
  data_addr = static_cast<void *>(addr);
  vector<int64_t > shape;
  GE_ASSERT_TRUE(ge::AttrUtils::GetListInt(op_desc, "origin_shape", shape));
  vector<int64_t > storage_shape;
  GE_ASSERT_TRUE(ge::AttrUtils::GetListInt(op_desc, "storage_shape", storage_shape));
  for (const auto &dim : storage_shape) {
      (void)out_shape.MutableStorageShape().AppendDim(dim);
  }
  for (const auto &dim : shape) {
      (void)out_shape.MutableOriginShape().AppendDim(dim);
  }
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, "size", data_length));
  GELOGI("[Lowering] op %s, addr ptr is %p, shape is [%s], storage_shape is [%s].", op_desc->GetNamePtr(),
         data_addr, ge::formats::JoinToString(shape).c_str(), ge::formats::JoinToString(storage_shape).c_str());
  return ge::SUCCESS;
}

LowerResult LoweringConstPlaceHolderNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LOWER_REQUIRE_NOTNULL(lower_input.global_data);
  LOWER_REQUIRE_NOTNULL(node);
  const ge::OpDescPtr &op_desc = node->GetOpDesc();
  LOWER_REQUIRE_NOTNULL(op_desc);
  StorageShape out_shape;
  void *data_addr = nullptr;
  int64_t data_length = 0L;
  LOWER_REQUIRE_SUCCESS(GetConstPlaceHolderAttr(op_desc, data_addr, data_length, out_shape),
                        "Node [%s] failed to get info from attr", node->GetNamePtr());
  gert::GertTensorData tensorData(data_addr, static_cast<size_t>(data_length),
                                  gert::TensorPlacement::kOnDeviceHbm, op_desc->GetStreamId());
  auto const_place_holder_outputs = FrameSelector::OnInitRoot([&tensorData, &op_desc,
                                                               &out_shape]() -> std::vector<ValueHolderPtr> {
      const auto shape_holder = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), out_shape);
      auto const_place_holder = DevMemValueHolder::CreateConst(&tensorData,
                                                               sizeof(tensorData), op_desc->GetStreamId());
      return {shape_holder, const_place_holder};
  });

  CONVERTER_CHECK_HOLDERS_ALL_OK(const_place_holder_outputs, 2U);
  auto address = std::dynamic_pointer_cast<bg::DevMemValueHolder>(const_place_holder_outputs[1U]);
  LOWER_REQUIRE_NOTNULL(address, "Node [%s] failed in lowering, DevMemValueHolder address is nullptr",
                        node->GetNamePtr());
  return {HyperStatus::Success(), {}, {const_place_holder_outputs[0U]}, {address}};
}

LowerResult LoweringDataNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto global_data = lower_input.global_data;
  if (global_data == nullptr) {
    return {HyperStatus::ErrorStatus("Failed to lowering data node, the global data is nullptr"), {}, {}, {}};
  }
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  int32_t index;
  if (!ge::AttrUtils::GetInt(op_desc, "index", index)) {
    return CreateErrorLowerResult("Failed to get index from data %s", node->GetName().c_str());
  }

  size_t data_size = global_data->GetValueHoldersSize(kDataNodeCounter);
  if ((node->GetType() == "AippData") && (index != static_cast<int32_t>(data_size))) {
    index = static_cast<int32_t>(data_size);
  }
  GELOGD("Lowering data node name %s, index %d, data size %zu", node->GetNamePtr(), index, data_size);
  global_data->SetValueHolders(kDataNodeCounter, nullptr);

  auto feed_tensors = GetOrCreateInputFeeds(global_data, node->GetOwnerComputeGraph());
  GE_ASSERT_TRUE(index < static_cast<int32_t>(feed_tensors.size()),
      "Data index:%d should less than feed tensor size: %zu.", index, feed_tensors.size());
  const auto feed_tensor = feed_tensors[index];
  GE_ASSERT_NOTNULL(feed_tensor);

  bool frozen_input = false;
  (void)ge::AttrUtils::GetBool(op_desc, "frozen_input", frozen_input);
  if (frozen_input) {
    return LoweringConstPlaceHolderNode(node, lower_input);
  }
  const int64_t logic_stream_id = op_desc->GetStreamId();

  AllocatorDesc allocator_desc = {static_cast<TensorPlacement>(feed_tensor->GetPlacement()), AllocatorUsage::kAllocNodeWorkspace};
  auto outputs = FrameSelector::OnInitRoot([&lower_input, &allocator_desc]() -> std::vector<ValueHolderPtr> {
    auto init_allocator = lower_input.global_data->GetOrCreateAllocator(allocator_desc);
    return {init_allocator};
  });
  auto allocator = outputs[0];
  auto split_outputs = DevMemValueHolder::CreateDataOutput(kernel::kSplitDataTensor, {feed_tensor, allocator},
                                                           static_cast<size_t>(kernel::SplitTensorOutputs::kNum),
                                                           logic_stream_id);
  CONVERTER_CHECK_HOLDERS_ALL_OK(split_outputs, static_cast<size_t>(kernel::SplitTensorOutputs::kNum));
  auto address = split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)];
  // unknown placement when lowering data node
  if (IsInputPlacementOnDeviceHbm()) {
    address->SetPlacement(kOnDeviceHbm);
  } else {
    address->SetPlacement(kTensorPlacementEnd);
  }
  return {HyperStatus::Success(),
          {},
          {split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]},
          {split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)]}};
}

LowerResult LoweringUnfedDataNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  static_cast<void>(lower_input);
  static_cast<void>(node);
  auto outputs = DevMemValueHolder::CreateDataOutput(
      "UnfedData", {}, static_cast<size_t>(kernel::SplitTensorOutputs::kNum), node->GetOpDescBarePtr()->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(outputs);
  return {HyperStatus::Success(),
          {},
          {outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]},
          {outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)]}};
}

LowerResult LoweringRefDataNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  // refdata in subgraph
  if (node->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr() != nullptr) {
    auto builder = []() -> std::vector<bg::ValueHolderPtr> { return {}; };
    auto origin_refdata = lower_input.global_data->GetOrCreateUniqueValueHolder(node->GetName(), builder);
    if (origin_refdata.size() < 2U) {
      return CreateErrorLowerResult("Failed to get refdata addr from global data %s.", node->GetNamePtr());
    }
    auto return_output_addr = std::dynamic_pointer_cast<bg::DevMemValueHolder>(
        origin_refdata[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)]);
    return {HyperStatus::Success(),
            {},
            {origin_refdata[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]},
            {return_output_addr}};
  }
  auto lowering_result = LoweringDataNode(node, lower_input);
  if (lowering_result.result.IsSuccess()) {
    // 把refdata的shape和地址保存到global data中，为分配ref node的输出内存做准备
    auto out_shape = lowering_result.out_shapes[0];
    auto address = lowering_result.out_addrs[0];
    auto builder = [&out_shape, &address]() -> std::vector<bg::ValueHolderPtr> { return {out_shape, address}; };
    lower_input.global_data->GetOrCreateUniqueValueHolder(node->GetName(), builder);
  }
  return lowering_result;
}

ValueHolderPtr GetInputDataShape(const ge::NodePtr &input_node, const LowerInput &lower_input) {
  const auto op_desc = input_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto *const_lower_result = op_desc->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
  if (const_lower_result == nullptr) {
    GELOGE(ge::FAILED, "Failed to get LowerResult of node %s.", input_node->GetName().c_str());
    return nullptr;
  }
  auto *lower_result = const_cast<PlacedLoweringResult *>(const_lower_result);
  const auto result = lower_result->GetOutputResult(*lower_input.global_data, 0UL, {-1, op_desc->GetStreamId()}, false);
  if (result == nullptr) {
    GELOGE(ge::FAILED, "Failed to get LowerResult of node %s.", input_node->GetName().c_str());
    return nullptr;
  }
  return result->shape;
}

ge::Status GetMultiBatchInputShapeAndIndex(const ge::NodePtr &node,
                                           const LowerInput &lower_input,
                                           std::vector<std::vector<int32_t>> &shape_index_vec,
                                           std::vector<ValueHolderPtr> &get_cur_shape_input) {
  const auto graph = node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  std::vector<int32_t> unknown_shape_data_index;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "_dynamic_batch_unknown_data_index", unknown_shape_data_index);
  GE_ASSERT_TRUE(!unknown_shape_data_index.empty());
  GE_ASSERT_TRUE(node->GetInControlNodesSize() == unknown_shape_data_index.size());
  shape_index_vec.resize(unknown_shape_data_index.size());
  get_cur_shape_input.resize(unknown_shape_data_index.size());
  for (const auto &in_ctrl_node : node->GetInControlNodes()) {
    GE_ASSERT_TRUE(in_ctrl_node->GetType() == ge::DATA);
    int32_t data_index = -1;
    (void)ge::AttrUtils::GetInt(in_ctrl_node->GetOpDescBarePtr(), ge::ATTR_NAME_INDEX, data_index);
    size_t index = 0UL;
    for (; index < unknown_shape_data_index.size(); index++) {
      if (unknown_shape_data_index[index] == data_index) {
        break;
      }
    }
    GE_ASSERT_TRUE(index != unknown_shape_data_index.size());
    std::vector<int32_t> unknown_dim_index;
    (void)ge::AttrUtils::GetListInt(in_ctrl_node->GetOpDesc(), "_dynamic_batch_unknown_dim_index", unknown_dim_index);
    GE_ASSERT_TRUE(!unknown_dim_index.empty());
    shape_index_vec[index] = unknown_dim_index;
    auto data_input_shape = GetInputDataShape(in_ctrl_node, lower_input);
    GE_ASSERT_NOTNULL(data_input_shape);
    get_cur_shape_input[index] = data_input_shape;
  }
  return ge::SUCCESS;
}

DevMemValueHolderPtr CreateShapeDataAddr(const ge::NodePtr &node, const LowerInput &lower_input,
                                         std::vector<ValueHolderPtr> &input_value_holder) {
  std::vector<std::vector<int32_t>> shape_index_vec;
  std::vector<ValueHolderPtr> get_cur_shape_input;
  if (GetMultiBatchInputShapeAndIndex(node, lower_input, shape_index_vec, get_cur_shape_input) != ge::SUCCESS) {
    return nullptr;
  }
  auto index_const = CreateConstVecHolder(shape_index_vec);
  GE_ASSERT_NOTNULL(index_const);
  input_value_holder.emplace_back(index_const);
  input_value_holder.insert(input_value_holder.cend(),
      get_cur_shape_input.cbegin(), get_cur_shape_input.cend());
  const auto node_desc = node->GetOpDescBarePtr();
  if (node_desc == nullptr) {
    return nullptr;
  }
  auto cur_shape_value_holder = bg::DevMemValueHolder::CreateSingleDataOutput("GetCurDynamicShape",
      input_value_holder, node_desc->GetStreamId());
  GE_ASSERT_NOTNULL(cur_shape_value_holder);
  cur_shape_value_holder->SetPlacement(kOnHost);
  return cur_shape_value_holder;
}

ValueHolderPtr CreateShapeDataShape(const ge::NodePtr &node) {
    const auto &output_desc = node->GetOpDescBarePtr()->GetOutputDescPtr(0UL);
    return NodeConverterUtils::CreateOutputShape(output_desc);
}

LowerResult LoweringMultiBatchShapeDataNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto get_cur_shape_shape = CreateShapeDataShape(node);
  LOWER_REQUIRE_NOTNULL(get_cur_shape_shape);
  std::vector<ValueHolderPtr> input_value_holder{get_cur_shape_shape};
  auto get_cur_shape_addr = CreateShapeDataAddr(node, lower_input, input_value_holder);
  LOWER_REQUIRE_NOTNULL(get_cur_shape_addr);
  return {HyperStatus::Success(), {}, {get_cur_shape_shape}, {get_cur_shape_addr}};
}

REGISTER_NODE_CONVERTER("Const", LoweringConstNode);
REGISTER_NODE_CONVERTER("ConstPlaceHolder", LoweringConstPlaceHolderNode);
REGISTER_NODE_CONVERTER("Data", LoweringDataNode);
REGISTER_NODE_CONVERTER("RefData", LoweringRefDataNode);
REGISTER_NODE_CONVERTER("AippData", LoweringDataNode);
REGISTER_NODE_CONVERTER("UnfedData", LoweringUnfedDataNode);
REGISTER_NODE_CONVERTER("multi_batch_shape_data", LoweringMultiBatchShapeDataNode);
}  // namespace gert
