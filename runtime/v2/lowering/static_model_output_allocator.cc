/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "static_model_output_allocator.h"
#include "common/ge_inner_attrs.h"
#include "common/checker.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_types.h"
#include "graph_builder/value_holder_generator.h"
#include "graph_builder/bg_memory.h"
#include "engine/node_converter_utils.h"
#include "graph_builder/bg_variable.h"
#include "graph/utils/math_util.h"
#include "graph/utils/op_type_utils.h"

namespace gert {
namespace {
constexpr uint64_t kAlignBytes = 32U;
using NodeTypeToMemoryBaseType = std::unordered_map<std::string, kernel::MemoryBaseType>;
// todo : 支持不完整
// 1 If 条件语句的两个子图，一个是ariable内存，一个是feature map内存
NodeTypeToMemoryBaseType kNodeType2MemoryBaseTypeMap = {
    {ge::CONSTANT, kernel::MemoryBaseType::kMemoryBaseTypeWeight}
    // todo 当前子图内FileConstant直连netouput支持还有问题，后续整改
};

ge::Status GetDataNodes(const ge::ComputeGraph &graph, std::map<int64_t, int32_t> &data_address_2_index) {
  std::map<int32_t, int64_t> ordered_data_nodes;
  for (const auto &node : graph.GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      int32_t index = -1;
      const auto op_desc = node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      if (!ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, index)) {
        GELOGW("Data %s seems not subgraph input, maybe %s is a static compiled root graph?", node->GetNamePtr(),
               graph.GetName().c_str());
        GE_ASSERT(ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_INDEX, index));
      }
      const auto logic_address = op_desc->GetOutputOffset();
      GE_ASSERT_TRUE(!logic_address.empty(), "data node[%s] output offsets is empty",
                     node->GetNamePtr(), logic_address.size());
      GE_ASSERT(ordered_data_nodes.emplace(index, logic_address[0U]).second, "Duplicated data index %d on graph %s",
                index, graph.GetName().c_str());
    }
  }

  // reindex
  int32_t data_index = 0;
  for (const auto &it : ordered_data_nodes) {
    (void)data_address_2_index.emplace(it.second, data_index++);
  }

  return ge::SUCCESS;
}

ge::Status GetVarNodes(const ge::ComputeGraph &graph, std::map<int64_t, ge::NodePtr> &address_2_node) {
  for (const auto &node : graph.GetAllNodes()) {
    if ((node->GetType() == ge::CONSTANTOP) || (node->GetType() == ge::VARIABLE)) {
      const auto op_desc = node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      const auto logic_address = op_desc->GetOutputOffset();
      GE_ASSERT_TRUE(!logic_address.empty(), "node[%s] output offsets is empty",
                     node->GetNamePtr(), logic_address.size());
      address_2_node.emplace(logic_address[0], node);
    }
  }
  return ge::SUCCESS;
}

ge::Status CalcTensorSize(const ge::ConstGeTensorDescPtr &ge_tensor_desc, uint64_t &ret_tensor_size) {
  int64_t tensor_size;
  GE_ASSERT_SUCCESS(ge::TensorUtils::CalcTensorMemSize(ge_tensor_desc->GetShape(), ge_tensor_desc->GetFormat(),
                                                       ge_tensor_desc->GetDataType(), tensor_size));
  // 不可能溢出，因为tensor_size最大值也只有int64的最大值
  ret_tensor_size = ge::RoundUp(static_cast<uint64_t>(tensor_size), kAlignBytes) + kAlignBytes;
  return ge::SUCCESS;
}

ge::Status ConstructParam(const ge::InDataAnchor *in_data_anchor,
                          ParseParam &param) {
  GE_CHECK_NOTNULL(in_data_anchor);
  const auto &out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(out_data_anchor);
  param.src_node = out_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(param.src_node);
  GE_CHECK_NOTNULL(param.src_node->GetOpDescBarePtr());

  param.input_index = in_data_anchor->GetIdx();
  param.ge_tensor_desc_ptr = param.op_desc->GetInputDescPtr(param.input_index);

  const auto input_offset = param.op_desc->GetInputOffset();
  GE_ASSERT_TRUE(static_cast<size_t>(param.input_index) < input_offset.size(),
                 "node: %s input_index: %d, input_offset size: %zu",
                 param.op_desc->GetNamePtr(), param.input_index, input_offset.size());
  param.input_address = input_offset[param.input_index];

  // 连接const的逻辑地址获取和其他的不一样
  const ge::vector_bit_t &v_is_input_const = param.op_desc->GetIsInputConst();
  if ((static_cast<size_t>(param.input_index) < v_is_input_const.size()) && (v_is_input_const[param.input_index])) {
    param.input_address = std::numeric_limits<int64_t>::max();
  }
  return ge::SUCCESS;
}
}  // namespace

StaticModelOutputAllocator::StaticModelOutputAllocator(const bg::ValueHolderPtr &davinci_model_holder,
                                                       const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                                       const bg::ValueHolderPtr &update_workspaces_holder)
    : davinci_model_holder_(davinci_model_holder),
      input_addrs_(input_addrs),
      update_workspaces_holder_(update_workspaces_holder) {}

StaticModelOutputAllocator::StaticModelOutputAllocator(const bg::ValueHolderPtr &davinci_model_holder,
                                                       const std::vector<bg::DevMemValueHolderPtr> &input_addrs)
    : davinci_model_holder_(davinci_model_holder), input_addrs_(input_addrs), update_workspaces_holder_(nullptr) {}

StaticModelOutputAllocator::~StaticModelOutputAllocator() = default;

ge::Status StaticModelOutputAllocator::ParseReuseInputs(ParseParam &param) {
  const auto &iter = param.data_address_2_index.find(param.input_address);
  if (iter != param.data_address_2_index.cend()) {
    OutputReuseInfo reuse_inputs_info{};
    reuse_inputs_info.is_reuse = true;
    reuse_inputs_info.reuse_type = OutputReuseType::kReuseInput;
    reuse_inputs_info.reuse_index = iter->second;
    GELOGD("output[%u] reuses input[%d]", param.input_index, iter->second);
    param.output_reuse_infos.emplace_back(std::move(reuse_inputs_info));
  }
  return ge::SUCCESS;
}

// output reference inner node
ge::Status StaticModelOutputAllocator::ParseRefOutputs(ParseParam &param) {
  const auto &iter = kNodeType2MemoryBaseTypeMap.find(param.src_node->GetType());
  int64_t data_offset = 0;
  if (iter != kNodeType2MemoryBaseTypeMap.cend()) {
    OutputReuseInfo ref_output_info{};
    const auto &tensor_desc = param.op_desc->MutableInputDesc(param.input_index);
    GE_ASSERT_GRAPH_SUCCESS(ge::TensorUtils::GetDataOffset(*tensor_desc, data_offset));
    uint64_t tensor_size;
    GE_ASSERT_SUCCESS(CalcTensorSize(param.ge_tensor_desc_ptr, tensor_size));
    ref_output_info.is_reuse = true;
    ref_output_info.reuse_type = OutputReuseType::kRefOutput;
    ref_output_info.mem_base_type_offset.offset = data_offset;
    ref_output_info.mem_base_type_offset.base_type = iter->second;
    ref_output_info.mem_base_type_offset.size = tensor_size;
    param.output_reuse_infos.emplace_back(std::move(ref_output_info));
    GELOGD("Static graph [%s] output[%d] ref to node [%s], base_memory_type:%u, data_offset:%ld, size:%lu",
           param.src_node->GetOwnerComputeGraphBarePtr()->GetName().c_str(), param.input_index,
           param.src_node->GetNamePtr(), static_cast<uint32_t>(iter->second), data_offset, tensor_size);
  }
  return ge::SUCCESS;
}

// output reference variable
ge::Status StaticModelOutputAllocator::ParseRefVariable(ParseParam &param) {
  const auto &iter = param.var_address_2_nodes.find(param.input_address);
  if (iter != param.var_address_2_nodes.end()) {
    OutputReuseInfo ref_output_info{};
    const ge::NodePtr &src_node = iter->second;
    ref_output_info.is_reuse = true;
    ref_output_info.reuse_type = OutputReuseType::kRefVariable;
    ref_output_info.var_name = src_node->GetName();
    param.output_reuse_infos.emplace_back(std::move(ref_output_info));
    GELOGD("Static graph [%s] output[%d] ref to node [%s], base_memory_type:variable, var_name:%s",
           src_node->GetOwnerComputeGraphBarePtr()->GetName().c_str(), param.input_index, src_node->GetNamePtr(),
           src_node->GetNamePtr());
  }
  return ge::SUCCESS;
}

ge::graphStatus StaticModelOutputAllocator::ParseReuseOutputs(ParseParam &param) {
  if (param.input_address != std::numeric_limits<int64_t>::max()) {
    const auto iter_and_inserted = param.offset_to_index_map.emplace(param.input_address, param.input_index);
    if (!iter_and_inserted.second) {
      OutputReuseInfo reuse_output_info{};
      reuse_output_info.is_reuse = true;
      reuse_output_info.reuse_type = OutputReuseType::kReuseOutput;
      reuse_output_info.reuse_index = iter_and_inserted.first->second;
      param.output_reuse_infos.emplace_back(std::move(reuse_output_info));
      GELOGD("[%s] output [%d] reuse output [%d], data_offset: %ld.",
             param.op_desc->GetNamePtr(), param.input_index, iter_and_inserted.first->second, param.input_address);
    }
  }
  return ge::SUCCESS;
}

ge::Status StaticModelOutputAllocator::ParseModelOutputReuseInfo(ParseParam &param) {
  std::vector<ParseFunc> funcs {
      StaticModelOutputAllocator::ParseReuseOutputs,  // 静态子图的某个输出可能复用静态子图的其他输出
      StaticModelOutputAllocator::ParseRefOutputs,    // 静态子图的某个输出可能复用const输出
      StaticModelOutputAllocator::ParseReuseInputs,   // 静态子图的某个输出可能复用静态子图某个输入
      StaticModelOutputAllocator::ParseRefVariable};  // 静态子图的某个输出可能复用变量输出
  const auto origin_size = param.output_reuse_infos.size();
  for (const auto &func : funcs) {
    GE_ASSERT_SUCCESS(func(param));
    if (param.output_reuse_infos.size() > origin_size) {
      break;
    }
  }
  if (param.output_reuse_infos.size() == origin_size) {
    OutputReuseInfo no_reuse_info{};
    no_reuse_info.is_reuse = false;
    param.output_reuse_infos.emplace_back(std::move(no_reuse_info));
  }
  param.output_reuse_infos.back().ge_tensor_desc_ptr = param.ge_tensor_desc_ptr;
  return ge::SUCCESS;
}

ge::Status StaticModelOutputAllocator::GenerateOutputsReuseInfos(const ge::ComputeGraphPtr &graph,
                                                                 std::vector<OutputReuseInfo> &output_reuse_infos) {
  const auto &net_output_node = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGI("[%s] Subgraph do not got net output", graph->GetName().c_str());
    return ge::SUCCESS;
  }
  const auto net_output_desc = net_output_node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL(net_output_desc);

  std::map<int64_t, int32_t> offset_to_index_map;
  ParseParam param(net_output_desc, output_reuse_infos, offset_to_index_map);
  GE_ASSERT_SUCCESS(GetDataNodes(*graph, param.data_address_2_index));
  GE_ASSERT_SUCCESS(GetVarNodes(*graph, param.var_address_2_nodes));
  for (const auto in_data_anchor : net_output_node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_SUCCESS(ConstructParam(in_data_anchor, param));
    GE_ASSERT_SUCCESS(ParseModelOutputReuseInfo(param));
  }
  return ge::SUCCESS;
}

LowerResult StaticModelOutputAllocator::AllocAllOutputs(const std::vector<OutputReuseInfo> &output_reuse_infos,
                                                        LoweringGlobalData &global_data) const {
  auto ref_output_addr_holders = AllocAllOutputsForRefOutputType(output_reuse_infos);
  auto ref_var_addr_holders = AllocAllOutputsForRefVariableType(output_reuse_infos, global_data);
  auto malloced_output_addr_holders = AllocAllOutputsForNoReuse(output_reuse_infos, global_data);
  if (!malloced_output_addr_holders.empty()) {
    LOWER_REQUIRE_VALID_HOLDER(malloced_output_addr_holders, "alloc output memory failed");
  }
  std::vector<bg::DevMemValueHolderPtr> all_output_holders;
  std::vector<ge::ConstGeTensorDescPtr> all_output_tensor_descs;
  size_t malloced_index = 0U;
  size_t ref_output_index = 0U;
  size_t ref_var_index = 0U;
  for (const auto& output_info : output_reuse_infos) {
    all_output_tensor_descs.emplace_back(output_info.ge_tensor_desc_ptr);
    if (!output_info.is_reuse) {
      LOWER_REQUIRE(malloced_index < malloced_output_addr_holders.size(),
                    "index:%zu, malloced_output_addr_holders size:%zu",
                    malloced_index, malloced_output_addr_holders.size());
      all_output_holders.emplace_back(malloced_output_addr_holders[malloced_index++]);
      continue;
    }

    switch (output_info.reuse_type) {
      case OutputReuseType::kRefOutput:
        LOWER_REQUIRE(ref_output_index < ref_output_addr_holders.size(),
                      "index:%zu, ref_output_addr_holders size:%zu",
                      ref_output_index, ref_output_addr_holders.size());
        all_output_holders.emplace_back(ref_output_addr_holders[ref_output_index++]);
        break;
      case OutputReuseType::kRefVariable:
        LOWER_REQUIRE(ref_var_index < ref_var_addr_holders.size(),
                      "index:%zu, ref_var_addr_holders size:%zu",
                      ref_var_index, ref_var_addr_holders.size());
        all_output_holders.emplace_back(ref_var_addr_holders[ref_var_index++]);
        break;
      case OutputReuseType::kReuseInput:
        LOWER_REQUIRE(static_cast<size_t>(output_info.reuse_index) < input_addrs_.size(),
                      "index:%d, input_addrs size:%zu",
                      output_info.reuse_index, input_addrs_.size());
        all_output_holders.emplace_back(input_addrs_[output_info.reuse_index]);
        break;
      case OutputReuseType::kReuseOutput:
        LOWER_REQUIRE(static_cast<size_t>(output_info.reuse_index) < all_output_holders.size(),
                      "index:%d, all_output_holders size:%zu",
                      output_info.reuse_index, all_output_holders.size());
        all_output_holders.emplace_back(all_output_holders[output_info.reuse_index]);
        break;
      case OutputReuseType::kNoReuse:
      case OutputReuseType::kEnd:
        break;
    }
  }
  auto shape_holders = NodeConverterUtils::CreateOutputShapes(all_output_tensor_descs);
  std::vector<bg::ValueHolderPtr> order_holder(malloced_output_addr_holders.cbegin(),
                                               malloced_output_addr_holders.cend());
  return {HyperStatus::Success(), {order_holder}, std::move(shape_holders), std::move(all_output_holders)};
}

std::vector<bg::DevMemValueHolderPtr> StaticModelOutputAllocator::AllocAllOutputsForRefOutputType(
    const std::vector<OutputReuseInfo> &output_reuse_infos) const {
  std::vector<kernel::MemoryBaseTypeOffset> mem_base_types_offsets;
  for (const auto& outputs_info : output_reuse_infos) {
    if (!outputs_info.is_reuse) {
      continue;
    }
    if (outputs_info.reuse_type == OutputReuseType::kRefOutput) {
      mem_base_types_offsets.emplace_back(outputs_info.mem_base_type_offset);
    }
  }
  if (mem_base_types_offsets.empty()) {
    GELOGI("ref outputs is empty");
    return {};
  }

  auto ref_output_addr_holders = GetRefOutputsAddress(mem_base_types_offsets);
  if (ref_output_addr_holders.size() != mem_base_types_offsets.size()) {
    GELOGE(ge::FAILED, "size not match! ref_output_addr_holders size is %zu, mem_base_types_offsets size is %zu",
           ref_output_addr_holders.size(), mem_base_types_offsets.size());
    return {};
  }

  // getting ref output address must be after updating workspace memory
  if (update_workspaces_holder_ != nullptr) {
    for (auto &ref_holder : ref_output_addr_holders) {
      (void)bg::ValueHolder::AddDependency(update_workspaces_holder_, ref_holder);
    }
  }
  return ref_output_addr_holders;
}

std::vector<bg::DevMemValueHolderPtr> StaticModelOutputAllocator::AllocAllOutputsForRefVariableType(
    const std::vector<OutputReuseInfo> &output_reuse_infos, LoweringGlobalData &global_data) const {
  std::vector<bg::DevMemValueHolderPtr> outputs;
  for (const auto &outputs_info : output_reuse_infos) {
    if (!outputs_info.is_reuse) {
      continue;
    }
    if (outputs_info.reuse_type == OutputReuseType::kRefVariable) {
      // static graph's parent node must be in main stream
      outputs.emplace_back(bg::GetVariableAddr(outputs_info.var_name, global_data, bg::kMainStream));
    }
  }
  return outputs;
}

std::vector<bg::DevMemValueHolderPtr> StaticModelOutputAllocator::GetRefOutputsAddress(
    const std::vector<kernel::MemoryBaseTypeOffset> &mem_base_types_offsets) const {
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(davinci_model_holder_);
  int64_t logic_stream_id = 0;
  auto stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
  GE_ASSERT_TRUE(IsValidHolder(stream_id_holder));
  inputs.emplace_back(stream_id_holder);
  for (const auto mem_base_type_offset : mem_base_types_offsets) {
    inputs.emplace_back(bg::ValueHolder::CreateConst(&mem_base_type_offset, sizeof(mem_base_type_offset)));
  }
  // static graph's parent node must be in stream 0
  return bg::DevMemValueHolder::CreateDataOutput("DavinciModelGetRunAddress", inputs, mem_base_types_offsets.size(),
                                                 bg::kMainStream);
}

std::vector<bg::DevMemValueHolderPtr> StaticModelOutputAllocator::AllocAllOutputsForNoReuse(
    const std::vector<OutputReuseInfo> &output_reuse_infos, LoweringGlobalData &global_data) {
  const auto size_holders = GetNoReuseOutputsSize(output_reuse_infos);
  auto alloc_mem_holders = bg::AllocMemoriesWithoutGuarder(kOnDeviceHbm, size_holders, global_data, bg::kMainStream);
  if (alloc_mem_holders.size() != size_holders.size()) {
    GELOGE(ge::FAILED, "size not match! alloc_mem_holders size is %zu, size_holders size is %zu",
           alloc_mem_holders.size(), size_holders.size());
    return {};
  }
  return alloc_mem_holders;
}

std::vector<bg::ValueHolderPtr> StaticModelOutputAllocator::GetNoReuseOutputsSize(
    const std::vector<OutputReuseInfo> &output_reuse_infos) {
  std::vector<bg::ValueHolderPtr> size_holders;
  for (const auto& output_info : output_reuse_infos) {
    if (output_info.is_reuse) {
      continue;
    }
    uint64_t tensor_size;
    const auto ret = CalcTensorSize(output_info.ge_tensor_desc_ptr, tensor_size);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "calc tensor size by shape failed");
      return {};
    }
    size_holders.emplace_back(bg::ValueHolder::CreateConst(&tensor_size, sizeof(tensor_size)));
  }
  return size_holders;
}
} // namespace gert
