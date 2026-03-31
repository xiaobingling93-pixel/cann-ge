/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/bg_kernel_context_extend.h"

#include "framework/common/debug/ge_log.h"
#include "common/checker.h"

#include "exe_graph/lowering/bg_ir_attrs.h"
#include "exe_graph/runtime/context_extend.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/math_util.h"

namespace gert {
namespace bg {
namespace {
ge::graphStatus InitIOInstanceInfo(const ge::NodePtr &node, ComputeNodeInfo &compute_node_info) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto in_ir_index_to_instance_index_pair_map
    = ge::OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(node->GetOpDesc());
  if (in_ir_index_to_instance_index_pair_map.empty()) {
    GELOGI("node [%s(%s)] ir_index_to_instance_index_pair_map is empty",
           node->GetNamePtr(), node->GetTypePtr());
    return ge::GRAPH_SUCCESS;
  }
  const auto &ir_inputs = op_desc->GetIrInputs();
  size_t input_index = 0;
  for (size_t i = 0; i < ir_inputs.size(); ++i) {
    auto ins_info = compute_node_info.MutableInputInstanceInfo(i);
    GE_ASSERT_NOTNULL(ins_info);
    size_t instance_num = in_ir_index_to_instance_index_pair_map[i].second;
    compute_node_info.MutableInputInstanceInfo(i)->SetInstantiationNum(instance_num);
    compute_node_info.MutableInputInstanceInfo(i)->SetInstanceStart(input_index);
    input_index += instance_num;
  }

  auto out_ir_index_to_instance_index_pair_map
      = ge::OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(node->GetOpDesc());
  if (out_ir_index_to_instance_index_pair_map.empty()) {
    GELOGI("node [%s(%s)] output ir_index_to_instance_index_pair_map is empty",
           node->GetNamePtr(), node->GetTypePtr());
    return ge::GRAPH_SUCCESS;
  }
  const auto &ir_outputs = op_desc->GetIrOutputs();
  size_t output_index = 0;
  for (size_t i = 0; i < ir_outputs.size(); ++i) {
    auto ins_info = compute_node_info.MutableOutputInstanceInfo(i);
    GE_ASSERT_NOTNULL(ins_info);
    size_t instance_num = out_ir_index_to_instance_index_pair_map[i].second;
    compute_node_info.MutableOutputInstanceInfo(i)->SetInstantiationNum(instance_num);
    compute_node_info.MutableOutputInstanceInfo(i)->SetInstanceStart(output_index);
    output_index += instance_num;
  }
  return ge::GRAPH_SUCCESS;
}

void SetCompileTimeTd(const ge::ConstGeTensorDescPtr &desc, CompileTimeTensorDesc &td) {
  td.SetDataType(desc->GetDataType());
  td.SetOriginFormat(desc->GetOriginFormat());
  td.SetStorageFormat(desc->GetFormat());
  int64_t reshape_type_mask = 0;
  if (ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask)) {
    td.SetExpandDimsType(ExpandDimsType(reshape_type_mask));
  }
  bool is_null_output = false;
  bool ret = ge::AttrUtils::GetBool(desc, ge::ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
  if (ret && is_null_output) {
    GELOGI("op name: %s set exist false", desc->GetName().c_str());
    td.SetExist(false);
  }
}

ge::graphStatus GetConnectedEdgeIndexesToAnchorIndexMap(const ge::NodePtr &node,
    std::map<size_t, size_t> &connected_edge_indexes_to_anchor_index) {
  size_t compute_node_index = 0U;
  for (const auto anchor : node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(anchor);
    if (anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    GE_ASSERT_NOTNULL(anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr());
    connected_edge_indexes_to_anchor_index[compute_node_index] = static_cast<size_t>(anchor->GetIdx());
    ++compute_node_index;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitCompileTimeTD(const ge::NodePtr &node, ComputeNodeInfo &compute_node_info) {
  std::map<size_t, size_t> connected_edge_indexes_to_anchor_index;
  const auto ret = GetConnectedEdgeIndexesToAnchorIndexMap(node, connected_edge_indexes_to_anchor_index);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ret, "get connected edge indexes to anchor index map failed. node:%s(%s)",
           node->GetName().c_str(), node->GetType().c_str());
    return ret;
  }
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(connected_edge_indexes_to_anchor_index.size() == compute_node_info.GetInputsNum());
  for (size_t i = 0; i < compute_node_info.GetInputsNum(); ++i) {
    GE_ASSERT_TRUE(i < op_desc->GetAllInputsSize());
    const auto &desc_need_check = op_desc->GetInputDesc(connected_edge_indexes_to_anchor_index[i]);
    if (desc_need_check.IsValid() != ge::GRAPH_SUCCESS) {
      continue;
    }
    const auto &desc = op_desc->GetInputDescPtr(connected_edge_indexes_to_anchor_index[i]);
    GE_ASSERT_NOTNULL(desc);
    auto td = compute_node_info.MutableInputTdInfo(i);
    GE_ASSERT_NOTNULL(td);
    SetCompileTimeTd(desc, *td);
  }

  for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto &desc = op_desc->GetOutputDescPtr(i);
    GE_ASSERT_NOTNULL(desc);
    auto td = compute_node_info.MutableOutputTdInfo(i);
    GE_ASSERT_NOTNULL(td);
    SetCompileTimeTd(desc, *td);
  }
  return ge::SUCCESS;
}
bool GetPrivateAttrsList(const ge::NodePtr &node, const gert::OpImplRegisterV2::PrivateAttrList &private_attrs,
                         std::vector<ge::AnyValue> &runtime_attrs_list) {
  const auto op = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op);
  const auto &all_attrs = op->GetAllAttrs();
  for (auto &private_attr : private_attrs) {
    auto &private_attr_name = private_attr.first;
    auto iter = all_attrs.find(private_attr_name.GetString());
    if (iter == all_attrs.end()) {
      if (!private_attr.second.IsEmpty()) {
        runtime_attrs_list.push_back(private_attr.second);
        continue;
      }
      GELOGE(ge::FAILED, "Can not find the private attr %s from node %s",
             private_attr_name.GetString(), node->GetName().c_str());
      return false;
    }
    runtime_attrs_list.push_back(iter->second);
  }
  return true;
}
std::unique_ptr<uint8_t[]> CreateComputeNodeInfoImpl(const std::unique_ptr<uint8_t[]> &attr_buf,
                                                     const size_t attr_size,
                                                     const ge::NodePtr &node,
                                                     BufferPool &buffer_pool,
                                                     size_t &total_size) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const size_t ir_input_num = op_desc->GetIrInputs().size();
  const size_t ir_output_num = op_desc->GetIrOutputs().size();
  const size_t input_num = node->GetInDataNodesAndAnchors().size();
  const uint32_t output_num = node->GetAllOutDataAnchorsSize();
  GELOGD("node: %s(%s), ir_input_num:%zu, ir_output_num:%zu, input_num:%zu, output_num:%u.", node->GetNamePtr(),
         node->GetTypePtr(), ir_input_num, ir_output_num, input_num, output_num);
  GE_ASSERT_SUCCESS(ComputeNodeInfo::CalcSize(ir_input_num, ir_output_num, input_num, output_num, total_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, attr_size, total_size));
  auto compute_node_info_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(compute_node_info_holder, "Create compute node info holder failed");

  auto node_name = buffer_pool.AddStr(node->GetNamePtr());
  auto node_type = buffer_pool.AddStr(node->GetTypePtr());
  auto compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder.get());
  compute_node_info->Init(ir_input_num, ir_output_num, input_num, output_num, attr_size,
                          ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_name)),
                          ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_type)));

  auto ret = InitIOInstanceInfo(node, *compute_node_info);
  GE_ASSERT_SUCCESS(ret, "Init input instance info for node:%s failed.", node->GetNamePtr());

  ret = InitCompileTimeTD(node, *compute_node_info);
  GE_ASSERT_SUCCESS(ret, "Init compile time tensor desc for node:%s failed.", node->GetNamePtr());

  auto attr = compute_node_info->MutableAttrs();
  const auto offset = ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr) - compute_node_info_holder.get();
  if (static_cast<size_t>(offset) > total_size) {
    GELOGE(
        ge::FAILED,
        "Failed to create kernel context extend info, the offset of attr %zu beyond the total size of ExtendInfo %zu",
        offset, attr_size);
    return nullptr;
  }
  const auto outputs_ins_info_size = compute_node_info->GetIrOutputsNum() * sizeof(AnchorInstanceInfo);
  ret = ge::GeMemcpy(ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr), (total_size - offset - outputs_ins_info_size),
                     attr_buf.get(), attr_size);
  GE_ASSERT_SUCCESS(ret, "memcpy_s failed, copy size is %zu, dst size is %zu", attr_size,
                    (total_size - offset - outputs_ins_info_size));
  GELOGI("Node %s, compute_node_info attr_size %zu, outputs_ins_info_size:%zu, offset:%zu, total_size:%zu.",
         node->GetNamePtr(), attr_size, outputs_ins_info_size, offset, total_size);
  return compute_node_info_holder;
}
}  // namespace

std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool, size_t &total_size) {
  size_t attr_size;
  const auto attr_buf = CreateAttrBuffer(node, attr_size);
  GE_ASSERT_NOTNULL(attr_buf, "Create attr buffer for node: %s failed", node->GetNamePtr());
  return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}

std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node,
                                                 BufferPool &buffer_pool,
                                                 const gert::OpImplRegisterV2::PrivateAttrList &private_attrs,
                                                 size_t &total_size) {
  std::vector<ge::AnyValue> runtime_attrs_list;
  GE_ASSERT_TRUE(GetPrivateAttrsList(node, private_attrs, runtime_attrs_list));
  size_t attr_size;
  const auto attr_buf = CreateAttrBuffer(node, runtime_attrs_list, attr_size);
  GE_ASSERT_NOTNULL(attr_buf, "Create attr buffer for node: %s failed", node->GetNamePtr());
  return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}
std::unique_ptr<uint8_t[]> CreateComputeNodeInfoWithoutIrAttr(const ge::NodePtr &node, BufferPool &buffer_pool,
    const gert::OpImplRegisterV2::PrivateAttrList &private_attrs, size_t &total_size) {
  std::vector<ge::AnyValue> runtime_attrs_list;
  GE_ASSERT_TRUE(GetPrivateAttrsList(node, private_attrs, runtime_attrs_list));
  size_t attr_size;
  const auto attr_buf = CreateAttrBufferWithoutIr(node, runtime_attrs_list, attr_size);
  GE_ASSERT_NOTNULL(attr_buf, "Create attr buffer without ir for node: %s failed", node->GetNamePtr());
  return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool) {
  size_t total_size;
  return CreateComputeNodeInfo(node, buffer_pool, total_size);
}
}  // namespace bg
}  // namespace gert
