/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "common/plugin/ge_make_unique_util.h"
#include "common/omg_util/omg_util.h"
#include "common/math/math_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "base/err_msg.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"

namespace {
using namespace ge;
const int32_t kIdentityAnchorIndex = 0;
const size_t kSerialStringVecSize = 4U;

const int32_t kCaseReadOnly = 0;
const int32_t kCaseScopeWriteable = 2;
const int32_t kCaseWriteable = 3;
const int32_t kCaseInvalidRWType = 5;
// attr _input_mutable = true means node will modify its input in runtime
const char *const kModifyInput = "_input_mutable";

// rw type of input.
enum class InputRWType {
  kReadOnly,        // Normal op input only read
  kWriteable,       // Op like Assign/ApplyMomentum
  kScopeWriteable,  // Op like hcom_allreduce/while, it will modify input ,but not expect take effect on pre ouput
  kInvalidRWType
};
// rw type of output
enum class OutputRWType {
  kReadOnlyConst, // 1.const output
  kReadOnly,   // 1.not ref output but has several peer output
  kSoftRead,   // not ref output but only has one output node
  kWriteable,  // ref output. Like Assign/ApplyMomentum
  kInvalidRWType
};

// input and output rw_type of one node. key is anchor_idx, value is rw_type
struct NodeInputOutputRWType {
  std::unordered_map<uint32_t, InputRWType> input_rw_type_map;
  std::unordered_map<uint32_t, OutputRWType> output_rw_type_map;
};
// input and output rw_type of node in current graph
thread_local std::unordered_map<std::string, NodeInputOutputRWType> node_rwtype_map_;
thread_local std::unordered_map<std::string, std::unordered_map<int32_t, NodePtr>> subgraph_inputs_;
thread_local std::unordered_map<std::string, NodePtr> subgraph_netoutput_;
thread_local std::unordered_map<std::string, std::unordered_map<uint32_t, uint32_t>> refs_input_2_output_;
thread_local std::unordered_map<std::string, std::unordered_map<uint32_t, uint32_t>> refs_output_2_input_;

///
/// @brief Convert input rw_type enum to string. For log print.
/// @param rw_type
/// @return rw_type_name
///
static std::string InputRWTypeToSerialString(InputRWType rw_type) {
  const static char *names[kSerialStringVecSize] = {"ReadOnly", "Writeable", "ScopeWriteable", "InvalidRWType"};
  GE_ASSERT_TRUE(static_cast<size_t>(rw_type) < kSerialStringVecSize);
  return names[static_cast<int32_t>(rw_type)];
}

///
/// @brief Convert output rw_type enum to string. For log print.
/// @param rw_type
/// @return rw_type_name
///
static std::string OutputRWTypeToSerialString(OutputRWType rw_type) {
  const static char *names[kSerialStringVecSize] = {"ReadOnly", "SoftRead", "Writeable", "InvalidRWType"};
  GE_ASSERT_TRUE(static_cast<size_t>(rw_type) < kSerialStringVecSize);
  return names[static_cast<int32_t>(rw_type)];
}

Status MarkRefRelations(const ComputeGraphPtr &compute_graph) {
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto input_name_index = op_desc->GetAllInputName();
    bool is_ref = false;
    for (const auto &name_index : input_name_index) {
      const int32_t out_index = op_desc->GetOutputIndexByName(name_index.first);
      if (out_index != -1) {
        refs_output_2_input_[op_desc->GetName()].emplace(static_cast<uint32_t>(out_index), name_index.second);
        refs_input_2_output_[op_desc->GetName()].emplace(name_index.second, static_cast<uint32_t>(out_index));
        is_ref = true;
      }
    }
    if (is_ref) {
      AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
      GELOGI("Node %s is reference node, set attribute %s to be true.", node->GetName().c_str(),
             ATTR_NAME_REFERENCE.c_str());
    }
  }
  return SUCCESS;
}

OutputRWType GetSingleNodeOutputRWTypeByIndex(const NodePtr &node, uint32_t index) {
  const auto &op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return OutputRWType::kInvalidRWType;
  }
  const auto &op_type = op_desc->GetType();
  if (OpTypeUtils::IsVarLikeNode(op_type)) {
    return OutputRWType::kWriteable;
  }
  // 当前torchair没有引入refdata，图上会出现data节点连ref，正式方案是torchair将可被写的data
  // 切换为refdata，当前判断为临时方案
  if (OpTypeUtils::IsDataNode(op_type) && !AttrUtils::HasAttr(op_desc, ATTR_NAME_PARENT_NODE_INDEX)) {
    return OutputRWType::kWriteable;
  }
  // 子图中的data，需要从parent node的输入来判断out rw type
  if (OpTypeUtils::IsDataNode(op_type) && AttrUtils::HasAttr(op_desc, ATTR_NAME_PARENT_NODE_INDEX)) {
    const auto pin_node_and_out_anchor = NodeUtils::GetParentInputAndAnchor(node);
    if (pin_node_and_out_anchor.first == nullptr || pin_node_and_out_anchor.second == nullptr) {
      return OutputRWType::kInvalidRWType;
    }
    return GetSingleNodeOutputRWTypeByIndex(pin_node_and_out_anchor.first, pin_node_and_out_anchor.second->GetIdx());
  }
  // check if it is ref output
  const auto iter = refs_output_2_input_.find(op_desc->GetName());
  if (iter != refs_output_2_input_.end()) {
    if (iter->second.find(index) != iter->second.end()) {
      return OutputRWType::kWriteable;
    }
  }
  // check if it is ref switch
  std::string type;
  if ((op_type == FRAMEWORK_OP_TYPE) && AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type) &&
      (type == REFSWITCH)) {
    return OutputRWType::kWriteable;
  }

  if ((op_type == CONSTANT) || (op_type == CONSTANTOP) || (op_type == CONSTPLACEHOLDER)) {
    return OutputRWType::kReadOnlyConst;
  }
  auto out_data_anchor = node->GetOutDataAnchor(index);
  if (out_data_anchor == nullptr) {
    return OutputRWType::kInvalidRWType;
  }
  if (out_data_anchor->GetPeerInDataNodesSize() > 1U) {
    return OutputRWType::kReadOnly;
  } else {
    return OutputRWType::kSoftRead;
  }
}

///
/// @brief Get input rw_type of one node with sub graph. It will return rw_type after solve conflict scene.
/// @param rw_type_set
/// @return
///
InputRWType GetInputRwTypeInConflict(const std::set<int32_t> &rw_type_set) {
  InputRWType conflict_map[][3] = {
      {InputRWType::kReadOnly, InputRWType::kWriteable, InputRWType::kScopeWriteable},
      {InputRWType::kWriteable, InputRWType::kWriteable, InputRWType::kInvalidRWType},
      {InputRWType::kScopeWriteable, InputRWType::kInvalidRWType, InputRWType::kScopeWriteable},
  };
  InputRWType ret = InputRWType::kReadOnly;
  for (auto rw : rw_type_set) {
    if (static_cast<InputRWType>(rw) == InputRWType::kInvalidRWType || ret == InputRWType::kInvalidRWType) {
      return InputRWType::kInvalidRWType;
    }
    ret = conflict_map[rw][static_cast<int32_t>(ret)];
  }
  return ret;
}

bool IsSubgraphInputNode(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) || (node->GetType() != DATA) ||
      (node->GetOwnerComputeGraph() == nullptr) || (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
    return false;
  }
  return true;
}

bool IsSubgraphOutputNode(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) || (node->GetType() != NETOUTPUT) ||
      (node->GetOwnerComputeGraph()->GetParentNode() == nullptr)) {
    return false;
  }
  return true;
}

GeTensorDesc GetCleanTensorDesc(const NodePtr &node, int32_t out_anchor_idx) {
  auto data_desc = node->GetOpDesc()->GetOutputDesc(out_anchor_idx);
  auto temp = TensorAdapter::GeTensorDesc2TensorDesc(data_desc);
  return TensorAdapter::TensorDesc2GeTensorDesc(temp);
}

OpDescPtr CreateIdentityOpDesc(const NodePtr &src_node, int32_t out_anchor_idx) {
  if (src_node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param src_node is invalid, which has no opdesc");
    GELOGE(GRAPH_PARAM_INVALID, "[Get][OpDesc] failed, Param src_node opdesc is nullptr.");
    return nullptr;
  }
  static std::atomic_long identity_num(0);
  auto next_num = identity_num.fetch_add(1);
  // 1. create new identity op desc
  std::string identity_name = src_node->GetName() + "_" + IDENTITY + std::to_string(next_num);
  OpDescBuilder op_desc_builder(identity_name, IDENTITY);
  auto data_desc = src_node->GetOpDesc()->GetOutputDesc(out_anchor_idx);
  ge::TensorUtils::SetReuseInput(data_desc, false);
  auto desc = GetCleanTensorDesc(src_node, out_anchor_idx);
  auto identity_op_desc = op_desc_builder.AddInput("x", desc).AddOutput("y", desc).Build();
  std::string batch_label;
  if ((AttrUtils::GetStr(src_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label)) && (!batch_label.empty())) {
    (void)AttrUtils::SetStr(identity_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
  }
  GELOGI("Insert new Identity node %s.", identity_name.c_str());
  return identity_op_desc;
}

void GetSubgraphOutputNodes(const NodePtr &node, std::vector<std::string> &subgraph_names,
                            std::vector<NodePtr> &output_node_vec, bool use_cache) {
  if (use_cache) {
    for (const auto &subgraph_name : subgraph_names) {
      output_node_vec.emplace_back(subgraph_netoutput_[subgraph_name]);
    }
  } else {
    output_node_vec = NodeUtils::GetSubgraphOutputNodes(*node.get());
  }
}

OutputRWType GetOutputRWTypeByIndex(const NodePtr &node, uint32_t index, bool use_cache = false) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return OutputRWType::kInvalidRWType;
  }
  if (kWhileOpTypes.count(op_desc->GetType()) > 0U) {
    return OutputRWType::kSoftRead;
  }
  std::vector<std::string> subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    // single node without sub graph
    return GetSingleNodeOutputRWTypeByIndex(node, index);
  }
  // node with sub graph
  std::vector<NodePtr> output_node_vec;
  GetSubgraphOutputNodes(node, subgraph_names, output_node_vec, use_cache);

  auto output_rw_type = OutputRWType::kSoftRead;
  if ((output_node_vec.size() == 1U) && (output_node_vec.at(0U) != nullptr)) {
    // find rw type from map.
    std::unordered_map<std::string, NodeInputOutputRWType>::const_iterator iter =
        node_rwtype_map_.find(output_node_vec.at(0)->GetName());
    if (iter == node_rwtype_map_.cend()) {
      GELOGW("Can not find rw type of node %s from map.It could take some effect on following preprocess.",
             output_node_vec.at(0)->GetName().c_str());
      return OutputRWType::kInvalidRWType;
    }
    std::unordered_map<uint32_t, OutputRWType>::const_iterator index_2_output_rw_type =
        iter->second.output_rw_type_map.find(index);
    if (index_2_output_rw_type == iter->second.output_rw_type_map.cend()) {
      GELOGW("Can not find rw type of node %s from map.It could take some effect on following preprocess.",
             output_node_vec.at(0)->GetName().c_str());
      return OutputRWType::kInvalidRWType;
    }
    output_rw_type = index_2_output_rw_type->second;
  }
  if (output_rw_type == OutputRWType::kWriteable) {
    return output_rw_type;
  }
  // check peer input
  auto out_data_anchor = node->GetOutDataAnchor(index);
  if (out_data_anchor == nullptr) {
    return OutputRWType::kInvalidRWType;
  }
  if (out_data_anchor->GetPeerInDataNodesSize() > 1U) {
    return OutputRWType::kReadOnly;
  } else {
    return output_rw_type;
  }
}

InputRWType GetSingleNodeInputRWTypeByIndex(const NodePtr &node, uint32_t index) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return InputRWType::kInvalidRWType;
  }
  bool is_mutable_input = false;
  (void)AttrUtils::GetBool(op_desc, kModifyInput, is_mutable_input);
  if (is_mutable_input) {
    return InputRWType::kScopeWriteable;
  }
  // SGAT插入的phonyconcat没有遵循IR定义，强行将concat的输入输出names指定为一致的，用于表达输入输出共内存。
  // 实际上该算子不是写算子。且该算子输入要求内存连续，这里需要单独判断，否则按照写算子插入identity后会导致
  // 内存分配时校验内存连续失败
  if (op_desc->GetType() == PHONYCONCAT || op_desc->GetType() == PHONYSPLIT) {
    return InputRWType::kReadOnly;
  }
  // check if it is ref input
  const auto iter = refs_input_2_output_.find(op_desc->GetName());
  if (iter != refs_input_2_output_.end()) {
    if (iter->second.find(index) != iter->second.end()) {
      return InputRWType::kWriteable;
    }
  }
  // check if it is ref switch
  const auto &op_type = op_desc->GetType();
  std::string type;
  if ((index == 0U) && (op_type == FRAMEWORK_OP_TYPE) &&
      (AttrUtils::GetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type)) && (type == REFSWITCH)) {
    return InputRWType::kWriteable;
  }

  return InputRWType::kReadOnly;
}

InputRWType GetInputRWTypeByIndex(const NodePtr &node, uint32_t index, bool use_cache = false) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return InputRWType::kInvalidRWType;
  }
  if (kWhileOpTypes.count(op_desc->GetType()) > 0U) {
    return InputRWType::kScopeWriteable;
  }
  auto input_desc = op_desc->MutableInputDesc(index);
  int64_t special_input_size = 0;
  // When input has attribute ATTR_NAME_SPECIAL_INPUT_SIZE, the input will be allocated with twice memory. So the input
  // rw type need to be modified to scope writeable for memory isolation.
  if (input_desc != nullptr && AttrUtils::GetInt(input_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, special_input_size) &&
      (special_input_size > 0)) {
    return InputRWType::kScopeWriteable;
  }
  std::vector<std::string> subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    // single node without sub graph
    return GetSingleNodeInputRWTypeByIndex(node, index);
  }
  std::vector<NodePtr> data_node_vec;
  if (use_cache) {
    for (const auto &subgraph_name : subgraph_names) {
      const auto iter = subgraph_inputs_[subgraph_name].find(static_cast<int32_t>(index));
      if (iter != subgraph_inputs_[subgraph_name].cend()) {
        data_node_vec.push_back(iter->second);
      }
    }
  } else {
    data_node_vec = NodeUtils::GetSubgraphDataNodesByIndex(*node.get(), index);
  }
  // get all input data node in subgraph
  std::set<int32_t> anchor_rw_type_set;
  for (const auto &data_node : data_node_vec) {
    if (data_node == nullptr) {
      continue;
    }
    // Data only has 1 out data anchor. Here just take first out data anchor. And index 0 is valid.
    auto out_data_anchor = data_node->GetOutDataAnchor(0);
    if (out_data_anchor == nullptr) {
      continue;
    }
    auto data_op_desc = data_node->GetOpDesc();
    if (data_op_desc == nullptr) {
      continue;
    }
    // find rw type from map.
    std::unordered_map<std::string, NodeInputOutputRWType>::const_iterator iter =
        node_rwtype_map_.find(data_op_desc->GetName());
    if (iter == node_rwtype_map_.cend()) {
      GELOGW("Can not find rw type of node %s from map.It could take some effect on following preprocess.",
             data_op_desc->GetName().c_str());
      return InputRWType::kInvalidRWType;
    }
    std::unordered_map<uint32_t, InputRWType>::const_iterator input_rw_type =
        iter->second.input_rw_type_map.find(out_data_anchor->GetIdx());
    if (input_rw_type == iter->second.input_rw_type_map.cend()) {
      GELOGW("Can not find rw type of node %s from map.It could take some effect on following preprocess.",
             data_op_desc->GetName().c_str());
      return InputRWType::kInvalidRWType;
    }
    anchor_rw_type_set.emplace(static_cast<int32_t>(input_rw_type->second));
  }
  return GetInputRwTypeInConflict(anchor_rw_type_set);
}

Status IsOutputRwConfilctAmongSubGraph(const NodePtr &parent_node, uint32_t parent_index, bool &is_conflict) {
  auto op_desc = parent_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto all_sub_graph_name = op_desc->GetSubgraphInstanceNames();
  auto root_graph = GraphUtils::FindRootGraph(parent_node->GetOwnerComputeGraph());
  for (const auto &sub_graph_name : all_sub_graph_name) {
    auto sub_graph = root_graph->GetSubgraph(sub_graph_name);
    GE_ASSERT_NOTNULL(sub_graph);
    auto netoutput_node = sub_graph->GetOrUpdateNetOutputNode();
    GE_ASSERT_NOTNULL(netoutput_node);
    auto in_anchor = netoutput_node->GetInDataAnchor(parent_index);
    GE_ASSERT_NOTNULL(in_anchor);
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    auto peer_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_node);
    uint32_t peer_index = static_cast<uint32_t>(peer_out_anchor->GetIdx());
    auto peer_rw_type = GetOutputRWTypeByIndex(peer_node, peer_index);
    if (peer_rw_type == OutputRWType::kReadOnly) {
      GELOGD("SubGrpah[%s] with output parent_index %u has ReadOnly OutputRWType", sub_graph_name.c_str(),
             parent_index);
      is_conflict = true;
      return ge::SUCCESS;
    }
  }
  return ge::SUCCESS;
}

bool JudgeOptimizableByParentNode(const NodePtr &parent_node, uint32_t parent_index) {
  auto output_anchor = parent_node->GetOutDataAnchor(parent_index);
  if ((parent_node->GetType() == PARTITIONEDCALL) || (output_anchor == nullptr)) {
    return true;
  }
  // 此PASS需要扩大到所有除While之外的所有控制节点类型，如果某一个子图中某一个输出是可写，其他任一子图对应的相同index输出是可读，
  // 则认为是“读写冲突”，需要后续；流程中插入identity节点，避免“读写冲突”
  if (parent_node->GetOpDesc()->GetSubgraphInstanceNames().size() > 1U) {
    GELOGD("JudgeOptimizableByParentNode: Check node %s[%s] with output RwConfilct among subGraph ",
           parent_node->GetName().c_str(), parent_node->GetType().c_str());
    bool is_conflict = false;
    if (IsOutputRwConfilctAmongSubGraph(parent_node, parent_index, is_conflict) == SUCCESS) {
      if (is_conflict) {
        // 冲突时需要返回false，由后续流程插入identity
        GELOGD("RWConflict with parent_node[%s] with output parent_index %u ", parent_node->GetName().c_str(),
               parent_index);
        return false;
      }
    }
  }

  for (const auto &in_anchor : output_anchor->GetPeerInDataAnchors()) {
    if (in_anchor == nullptr) {
      continue;
    }
    if (!in_anchor->GetOwnerNode()->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
      return false;
    }

    const auto input_type = GetSingleNodeInputRWTypeByIndex(in_anchor->GetOwnerNode(), in_anchor->GetIdx());
    if (input_type != InputRWType::kReadOnly) {
      return false;
    }
  }
  return true;
}

Status MarkRWTypeForSubgraphInput(const NodePtr &node) {
  std::set<int32_t> anchor_rw_type_set;
  // calc all input_rw_type of peer output , as input_rw_type of DATA. Index 0 is valid.
  auto anchor_2_node_vec = NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, 0);
  for (const auto &anchor_2_node_pair : anchor_2_node_vec) {
    GE_CHECK_NOTNULL(anchor_2_node_pair.second);
    GE_CHECK_NOTNULL(anchor_2_node_pair.first);
    auto input_rw_type = GetInputRWTypeByIndex(anchor_2_node_pair.second, anchor_2_node_pair.first->GetIdx());
    GELOGD("Input rw type of Node %s %dth input anchor is %s", anchor_2_node_pair.second->GetName().c_str(),
           anchor_2_node_pair.first->GetIdx(), InputRWTypeToSerialString(input_rw_type).c_str());
    anchor_rw_type_set.emplace(static_cast<int32_t>(input_rw_type));
  }
  auto anchor_rw_type = GetInputRwTypeInConflict(anchor_rw_type_set);
  GELOGD("Input rw type of Node %s is %s", node->GetName().c_str(), InputRWTypeToSerialString(anchor_rw_type).c_str());
  std::unordered_map<uint32_t, InputRWType> input_rw_type_map{std::make_pair(0U, anchor_rw_type)};
  NodeInputOutputRWType data_rw_type{input_rw_type_map, {}};
  node_rwtype_map_.emplace(std::make_pair(node->GetName(), data_rw_type));
  return SUCCESS;
}

Status MarkRWTypeForSubgraphOutput(const ComputeGraphPtr &sub_graph, const NodePtr &node) {
  // calc all output_rw_type of peer input , as output_rw_type of DATA
  std::unordered_map<uint32_t, OutputRWType> output_rw_type_map;
  auto parent_node = sub_graph->GetParentNode();
  GE_ASSERT_NOTNULL(parent_node);
  GE_ASSERT_NOTNULL(parent_node->GetOpDesc());
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    auto index = static_cast<uint32_t>(in_data_anchor->GetIdx());
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    GE_ASSERT_TRUE(index <= node->GetOpDesc()->GetAllInputsSize());
    uint32_t parent_idx = 0U;
    GE_ASSERT_TRUE(
        AttrUtils::GetInt(node->GetOpDesc()->MutableInputDesc(index), ATTR_NAME_PARENT_NODE_INDEX, parent_idx));
    auto pre_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(pre_out_anchor);
    auto pre_node = pre_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(pre_node);

    auto pre_output_rw_type = GetOutputRWTypeByIndex(pre_node, pre_out_anchor->GetIdx());
    GELOGD("Output rw type of Node %s %dth output anchor is %s", pre_node->GetName().c_str(), pre_out_anchor->GetIdx(),
           OutputRWTypeToSerialString(pre_output_rw_type).c_str());

    if ((pre_output_rw_type == OutputRWType::kWriteable) &&
        (!JudgeOptimizableByParentNode(parent_node, parent_idx))) {
      // insert identity
      auto identity_op = CreateIdentityOpDesc(pre_node, pre_out_anchor->GetIdx());
      GE_CHECK_NOTNULL(identity_op);
      NodePtr identity_node = GraphUtils::InsertNodeAfter(pre_out_anchor, {in_data_anchor}, identity_op);
      if (identity_node == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Insert Identity node %s(%s) between %s(%s) -> %s(%s) failed.",
                          identity_op->GetName().c_str(), identity_op->GetType().c_str(),
                          pre_node->GetName().c_str(), pre_node->GetType().c_str(), node->GetName().c_str(),
                          node->GetType().c_str());
        GELOGE(FAILED, "[Insert][IdentityNode] %s(%s) between %s(%s) -> %s(%s) failed.",
               identity_op->GetName().c_str(), identity_op->GetType().c_str(), pre_node->GetName().c_str(),
               pre_node->GetType().c_str(), node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
      GELOGI("InsertNode %s between %s:%d and %s:%d successfully.", identity_node->GetName().c_str(),
             pre_node->GetName().c_str(), pre_out_anchor->GetIdx(),
             node->GetName().c_str(), in_data_anchor->GetIdx());
      pre_output_rw_type = OutputRWType::kSoftRead;
    }
    output_rw_type_map.emplace(std::make_pair(in_data_anchor->GetIdx(), pre_output_rw_type));
  }
  NodeInputOutputRWType output_rw_type{{}, output_rw_type_map};
  node_rwtype_map_.emplace(std::make_pair(node->GetName(), output_rw_type));
  return SUCCESS;
}

Status MarkRWTypeForSubgraph(const ComputeGraphPtr &sub_graph) {
  GE_CHECK_NOTNULL(sub_graph);
  const auto &sub_graph_name = sub_graph->GetName();
  // the name of the subgraph should be unique
  if (subgraph_inputs_.count(sub_graph_name) > 0U || subgraph_netoutput_.count(sub_graph_name) > 0U) {
    GELOGE(FAILED, "Subgraph name %s is not uniqe", sub_graph_name.c_str());
    return FAILED;
  }
  for (const auto &node : sub_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetType() == DATA) {
      GE_CHK_STATUS_RET(MarkRWTypeForSubgraphInput(node), "Data node %s mark failed in subgraph %s",
                        node->GetName().c_str(), sub_graph_name.c_str());
      int32_t parent_node_index = INT32_MAX;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
        GELOGE(FAILED, "Input data %s has no attr [%s]!", node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
        return FAILED;
      }
      if (!(subgraph_inputs_[sub_graph_name].insert({parent_node_index, node}).second)) {
        GELOGE(FAILED, "Input data %s has same attr [%s] value[%d] with other data!", node->GetName().c_str(),
               ATTR_NAME_PARENT_NODE_INDEX.c_str(), parent_node_index);
        return FAILED;
      }
    }

    if (node->GetType() == NETOUTPUT) {
      GE_CHK_STATUS_RET(MarkRWTypeForSubgraphOutput(sub_graph, node), "Netoutput node %s mark failed in subgraph %s",
                        node->GetName().c_str(), sub_graph_name.c_str());
      subgraph_netoutput_[sub_graph_name] = node;
    }
  }
  return SUCCESS;
}
///
/// @brief Reverse traversal all subgraph and mark rw_type for Data/Netoutput.
/// @param sub_graph_vecgs
///
Status MarkRWTypeForAllSubgraph(const std::vector<ComputeGraphPtr> &sub_graph_vec) {
  // the name of the subgraph should be unique
  for (auto iter = sub_graph_vec.rbegin(); iter != sub_graph_vec.rend(); ++iter) {
    GE_CHECK_NOTNULL(*iter);
    auto parent_node = (*iter)->GetParentNode();
    if (parent_node == nullptr) {
      GELOGD("Current sub graph has no parent node. Ignore it.");
      continue;
    }
    if (kWhileOpTypes.count(parent_node->GetType()) > 0U) {
      continue;
    }
    auto ret = MarkRWTypeForSubgraph(*iter);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

///
/// @brief Check identity is near subgraph.
///    Eg. As output of Data node in subgraph
///        or as input of Netoutput of subgraph
///        or as input of one node with subgraph
///        or as output of one node with subgraph
/// @param node
/// @return is_near_subgraph
///
bool CheckIdentityIsNearSubgraph(const NodePtr &node) {
  for (const auto &in_node : node->GetInDataNodes()) {
    if (in_node == nullptr) {
      continue;
    }
    auto in_node_opdesc = in_node->GetOpDesc();
    if (in_node_opdesc == nullptr) {
      continue;
    }
    // near entrance of subgraph
    if (IsSubgraphInputNode(in_node)) {
      return true;
    }
    // near subgraph
    if (!in_node_opdesc->GetSubgraphInstanceNames().empty()) {
      return true;
    }
  }

  for (const auto &out_node : node->GetOutDataNodes()) {
    if (out_node == nullptr) {
      continue;
    }
    auto out_node_opdesc = out_node->GetOpDesc();
    if (out_node_opdesc == nullptr) {
      continue;
    }
    // near output of subgraph
    if (IsSubgraphOutputNode(out_node)) {
      return true;
    }
    // near subgraph
    if (!out_node_opdesc->GetSubgraphInstanceNames().empty()) {
      return true;
    }
  }
  return false;
}
enum class ConflictResult { DO_NOTHING, WRONG_GRAPH, INSERT_IDENTITY };
vector<std::vector<ConflictResult>> output_2_input_rwtype = {
    {ConflictResult::DO_NOTHING, ConflictResult::INSERT_IDENTITY, ConflictResult::INSERT_IDENTITY},
    {ConflictResult::DO_NOTHING, ConflictResult::DO_NOTHING, ConflictResult::INSERT_IDENTITY},
    {ConflictResult::DO_NOTHING, ConflictResult::DO_NOTHING, ConflictResult::DO_NOTHING},
    {ConflictResult::DO_NOTHING, ConflictResult::DO_NOTHING, ConflictResult::INSERT_IDENTITY}};
ConflictResult GetConflictResultBetweenNode(const OutputRWType output_rw_type, const InputRWType input_rw_type) {
  if (output_rw_type == OutputRWType::kInvalidRWType || input_rw_type == InputRWType::kInvalidRWType) {
    return ConflictResult::WRONG_GRAPH;
  }
  auto n = static_cast<int32_t>(output_rw_type);
  auto m = static_cast<int32_t>(input_rw_type);

  // no need to check index or container, because container and index is all defined.
  return output_2_input_rwtype[n][m];
}

///
/// @brief Keep identity_node which near subgraph or has multi output
/// @param node
/// @return
///
Status RemoveNoUseIdentity(const NodePtr &node) {
  if (node->GetInDataNodes().empty() || node->GetOutDataNodesSize() > 1U) {
    return SUCCESS;
  }
  if (node->GetOutDataNodesSize() == 1U && node->GetOutDataNodes().at(0)->GetType() == STREAMMERGE) {
    return SUCCESS;
  }
  if (CheckIdentityIsNearSubgraph(node)) {
    return SUCCESS;
  }
  if (NodeUtils::IsIdentityUsefulForRWControl(node)) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(node->GetInDataAnchor(kIdentityAnchorIndex));
  auto pre_out_anchor = node->GetInDataAnchor(kIdentityAnchorIndex)->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(pre_out_anchor);
  auto pre_node = pre_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(pre_node);
  auto pre_output_rw_type = GetOutputRWTypeByIndex(pre_node, pre_out_anchor->GetIdx(), true);
  auto anchor_2_outnode_vec = NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, kIdentityAnchorIndex);
  ConflictResult conflict_result = ConflictResult::WRONG_GRAPH;
  if (!anchor_2_outnode_vec.empty()) {
    auto anchor_2_outnode = anchor_2_outnode_vec.at(0);
    GE_CHECK_NOTNULL(anchor_2_outnode.second);
    GE_CHECK_NOTNULL(anchor_2_outnode.first);
    auto peer_input_rw_type = GetInputRWTypeByIndex(anchor_2_outnode.second, anchor_2_outnode.first->GetIdx(), true);
    GELOGD("Pre Node %s %dth output rw type is %s, peer node %s %dth input rw type is %s.", pre_node->GetName().c_str(),
           pre_out_anchor->GetIdx(), OutputRWTypeToSerialString(pre_output_rw_type).c_str(),
           anchor_2_outnode.second->GetName().c_str(), anchor_2_outnode.first->GetIdx(),
           InputRWTypeToSerialString(peer_input_rw_type).c_str());
    conflict_result = GetConflictResultBetweenNode(pre_output_rw_type, peer_input_rw_type);
  } else {
    // identity node has no out data node, it can be removed
    conflict_result = ConflictResult::DO_NOTHING;
  }
  if (conflict_result != ConflictResult::DO_NOTHING) {
    return SUCCESS;
  }
  GELOGI("No need insert Identity. Node %s need to remove.", node->GetName().c_str());
  auto ret = GraphUtils::IsolateNode(node, {0});
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Isolate Node:%s failed", node->GetName().c_str());
    GELOGE(ret, "[Isolate][Node] %s failed.", node->GetName().c_str());
    return ret;
  }
  ret = GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Call RemoveNodeWithoutRelink failed, node:%s", node->GetName().c_str());
    GELOGE(ret, "[Call][RemoveNodeWithoutRelink] failed for node %s.", node->GetName().c_str());
    return ret;
  }
  GELOGI("Pre node is %s and %dth out rw type is %s. Isolate and remove Identity node %s.", pre_node->GetName().c_str(),
         pre_out_anchor->GetIdx(), OutputRWTypeToSerialString(pre_output_rw_type).c_str(), node->GetName().c_str());
  return SUCCESS;
}

Status SplitIdentityAlongAnchor(const OutDataAnchorPtr &out_data_anchor, const InDataAnchorPtr &peer_in_data_anchor,
                                const OutDataAnchorPtr &pre_out_data_anchor, NodePtr &pre_node) {
  // 1.check peer in node RW type.
  GE_CHECK_NOTNULL(pre_node);
  GE_CHECK_NOTNULL(peer_in_data_anchor);
  auto peer_in_data_node = peer_in_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_in_data_node);
  auto input_rw_type = GetInputRWTypeByIndex(peer_in_data_node, peer_in_data_anchor->GetIdx(), true);
  GE_CHECK_NOTNULL(out_data_anchor);
  GE_CHECK_NOTNULL(pre_out_data_anchor);
  auto old_identity = out_data_anchor->GetOwnerNode();
  if (input_rw_type == InputRWType::kScopeWriteable || input_rw_type == InputRWType::kWriteable) {
    auto new_identity_op = CreateIdentityOpDesc(pre_node, pre_out_data_anchor->GetIdx());
    GE_CHECK_NOTNULL(new_identity_op);
    GE_ASSERT_NOTNULL(GraphUtils::InsertNodeBefore(peer_in_data_anchor, new_identity_op,
        kIdentityAnchorIndex, kIdentityAnchorIndex));
    GELOGI("Node %s intput rw type is %s. Insert Identity between %s and %s.", peer_in_data_node->GetName().c_str(),
           InputRWTypeToSerialString(input_rw_type).c_str(), pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
           peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
  } else {
    (void)out_data_anchor->Unlink(peer_in_data_anchor);
    // copy control edge to pre and peer node
    if (GraphUtils::CopyInCtrlEdges(old_identity, peer_in_data_node) != SUCCESS ||
        GraphUtils::CopyOutCtrlEdges(old_identity, pre_node) != SUCCESS) {
      GELOGW("Fail to copy control edge from node %s.", old_identity->GetName().c_str());
      return FAILED;
    }
    // link identity pre node to next node directly
    if (GraphUtils::AddEdge(pre_out_data_anchor, peer_in_data_anchor) != SUCCESS) {
      GELOGW("Fail to link data edge from node %s to %s.", pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_data_anchor->GetOwnerNode()->GetName().c_str());
      return FAILED;
    }
    GELOGI("Node %s input rw type is %s, link data edge from Identity input node %s to out node %s directly.",
           peer_in_data_node->GetName().c_str(), InputRWTypeToSerialString(input_rw_type).c_str(),
           pre_node->GetName().c_str(), peer_in_data_node->GetName().c_str());
  }
  return SUCCESS;
}

Status SplitIdentity(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto out_data_anchor = node->GetOutDataAnchor(kIdentityAnchorIndex);
  GE_CHECK_NOTNULL(out_data_anchor);
  if (out_data_anchor->GetPeerInDataNodesSize() <= 1U) {
    return SUCCESS;
  }
  // get pre node and next node of identity
  GE_CHECK_NOTNULL(node->GetInDataAnchor(kIdentityAnchorIndex));
  auto pre_out_data_anchor = node->GetInDataAnchor(kIdentityAnchorIndex)->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(pre_out_data_anchor);
  auto pre_node = pre_out_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(pre_node);
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    Status ret = SplitIdentityAlongAnchor(out_data_anchor, peer_in_data_anchor, pre_out_data_anchor, pre_node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][SplitIdentityAlongAnchor] failed, ret:%d, node:%s, pre_node:%s.", ret,
             node->GetName().c_str(), pre_node->GetName().c_str());
      return ret;
    }
  }
  // 2.isolate Identity node with no data output
  if (node->GetOutDataNodesSize() == 0) {
    Status ret = GraphUtils::IsolateNode(node, {});
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "IsolateNode %s failed, ret:%u", node->GetName().c_str(), ret);
      GELOGE(FAILED, "[Isolate][Node] %s failed, ret:%u", node->GetName().c_str(), ret);
      return FAILED;
    }
    ret = GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call RemoveNodeWithoutRelink failed, node:%s", node->GetName().c_str());
      GELOGE(FAILED, "[Call][RemoveNodeWithoutRelink] IsolateAndDelete identity node %s failed.",
             node->GetName().c_str());
      return FAILED;
    }
    GELOGI("IsolateAndDelete identity node %s.", node->GetName().c_str());
  }
  return SUCCESS;
}

bool IsLastLinkCanSkipIdentity(const NodePtr &node, InputRWType rw_type, NodePtr &last_node) {
  if (last_node == nullptr) {

    last_node = node;
  }
  // if in anchor in different node, can't skip
  if (last_node != node) {
    return false;
  }
  // if in anchor rw type is not scope write, can't skip
  if (rw_type != InputRWType::kScopeWriteable) {
    return false;
  }
  return true;
}

bool NeedCheckSkipIdentiy(OutputRWType type, const NodePtr &out_node, const OutDataAnchorPtr &out_anchor) {
  if (type == OutputRWType::kReadOnly) {
    return true;
  }

  if (type != OutputRWType::kWriteable) {
    return false;
  }

  // 可写内存跳过插入identity: 从当前节点向前查找，如果该内存仅在一条路径上使用，则无需插入identity
  // 1. output存在多个边，return false(如果是const节点，在前面的处理中，已经插入identity算子，
  // 在查找到identity时，会命中返回条件，不会查找到cosnt节点)
  // 2. 输入节点
  // 2.1 子图输入：return false，不向外查找
  // 2.2 根图：return true
  // 3. 获取Output type
  // 3.1 非ref节点：返回true
  // 3.2 ref节点：继续向前查找
  NodePtr temp = out_node;
  OutDataAnchorPtr temp_out_anchor = out_anchor;
  while (true) {
    // 判断能否跳过插入identity
    if (temp_out_anchor->GetPeerInDataAnchors().size() > 1) {
      break;
    }
    if (OpTypeUtils::IsDataNode(temp->GetType())) {
      return !AttrUtils::HasAttr(temp->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX);
    }
    // 找到ref的output2input map
    const auto iter = refs_output_2_input_.find(temp->GetOpDesc()->GetName());
    if (iter == refs_output_2_input_.end()) { 
      return true;
    }
    // 找到对应的输入input anchor
    auto input_idx = iter->second.find(temp_out_anchor->GetIdx());
    if (input_idx == iter->second.end()) {
      return true;
    }
    auto temp_in_anchor = temp->GetInDataAnchor(input_idx->second);

    // 找到对应的output node 和index
    temp_out_anchor = temp_in_anchor->GetPeerOutAnchor();
    if (temp_out_anchor == nullptr) {
      break;
    }
    temp = temp_out_anchor->GetOwnerNode();
  }

  return false;
}

Status CreateIdentityAndInsertBefore(const NodePtr &dst, const InDataAnchorPtr &dst_anchor, const NodePtr &src,
                                     const OutDataAnchorPtr &src_anchor) {
  auto identity_op = CreateIdentityOpDesc(src, src_anchor->GetIdx());
  GE_CHECK_NOTNULL(identity_op);
  GE_ASSERT_NOTNULL(GraphUtils::InsertNodeBefore(dst_anchor, identity_op, kIdentityAnchorIndex, kIdentityAnchorIndex));
  GELOGI("Insert Identity %s between %s:%d and %s:%d to handle memory conflict.", identity_op->GetName().c_str(),
         src->GetName().c_str(), src_anchor->GetIdx(), dst->GetName().c_str(), dst_anchor->GetIdx());
  return SUCCESS;
}

Status InsertIdentityAsNeeded(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (node->GetOutDataNodesSize() == 0U) {
    return SUCCESS;
  }
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    NodePtr linked_node = nullptr;
    auto output_rw_type = GetOutputRWTypeByIndex(node, out_data_anchor->GetIdx(), true);
    // if a node not const, and output anchor connect to all scopewrite anchor in one node,
    // the last edge skip insert identity
    auto peer_anchors = out_data_anchor->GetPeerInDataAnchors();
    bool skip_insert_identity = NeedCheckSkipIdentiy(output_rw_type, node, out_data_anchor);
    for (auto it = peer_anchors.begin(); it < peer_anchors.end(); it++) {
      auto &peer_in_data_anchor = *it;
      GE_CHECK_NOTNULL(peer_in_data_anchor);
      auto peer_in_node = peer_in_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_in_node);
      bool need_skip_rw_confilct = false;
      (void)AttrUtils::GetBool(peer_in_node->GetOpDesc(), "_skip_rw_conflict", need_skip_rw_confilct);
      if (need_skip_rw_confilct) {
        GELOGI("Node [%s] [%s] has checked rw conflict on hccl_memcpy_pass, skip here", peer_in_node->GetNamePtr(),
               peer_in_node->GetTypePtr());
        continue;
      }
      auto input_rw_type = GetInputRWTypeByIndex(peer_in_node, peer_in_data_anchor->GetIdx(), true);
      GELOGD("Node %s:%d output rw type is %s, Node %s:%d input rw type is %s", node->GetName().c_str(),
             out_data_anchor->GetIdx(), OutputRWTypeToSerialString(output_rw_type).c_str(),
             peer_in_node->GetName().c_str(), peer_in_data_anchor->GetIdx(),
             InputRWTypeToSerialString(input_rw_type).c_str());
      auto conflict_result = GetConflictResultBetweenNode(output_rw_type, input_rw_type);
      if (skip_insert_identity) {
        skip_insert_identity = IsLastLinkCanSkipIdentity(peer_in_node, input_rw_type, linked_node);
      }
      switch (conflict_result) {
        case ConflictResult::DO_NOTHING:
        case ConflictResult::WRONG_GRAPH:
          GELOGD("No need insert Identity.");
          skip_insert_identity = false;  // have other rw type connection, can't skip
          continue;
        case ConflictResult::INSERT_IDENTITY: {
          if (it == peer_anchors.end() - 1 && skip_insert_identity) {
            continue;
          }
          GE_ASSERT_SUCCESS(CreateIdentityAndInsertBefore(peer_in_node, peer_in_data_anchor, node, out_data_anchor));
          continue;
        }
        default:
          break;
      }
    }
  }
  return SUCCESS;
}

void ReInit() {
  node_rwtype_map_.clear();
  subgraph_inputs_.clear();
  subgraph_netoutput_.clear();
  refs_input_2_output_.clear();
  refs_output_2_input_.clear();
}
}  // namespace

namespace ge {
Status GraphOptimize::CheckRWConflict(ComputeGraphPtr &compute_graph, bool &has_conflict) const {
  ReInit();
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHK_STATUS(MarkRefRelations(compute_graph), "Mark ref relations failed for %s", compute_graph->GetName().c_str());
  const auto &sub_graph_vec = compute_graph->GetAllSubgraphs();
  if (sub_graph_vec.empty()) {
    GELOGD("No sub graph here. Ignore memory conflict handle.");
    return SUCCESS;
  }
  // 1.loop all subgraph, mark rw type from inside to outside
  Status ret = MarkRWTypeForAllSubgraph(sub_graph_vec);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][MarkRWTypeForAllSubgraph] failed for %s.", compute_graph->GetName().c_str());
    return ret;
  }
  has_conflict = false;
  for (const auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (node->GetOutDataNodesSize() == 0) {
      return SUCCESS;
    }
    if (kWhileOpTypes.count(node->GetType()) > 0U) {
      return SUCCESS;
    }
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_data_anchor);
      auto output_rw_type = GetOutputRWTypeByIndex(node, out_data_anchor->GetIdx(), true);
      for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(peer_in_data_anchor);
        auto peer_in_node = peer_in_data_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(peer_in_node);
        if (kWhileOpTypes.count(peer_in_node->GetType()) > 0U) {
          return SUCCESS;
        }
        auto input_rw_type = GetInputRWTypeByIndex(peer_in_node, peer_in_data_anchor->GetIdx(), true);
        auto conflict_result = GetConflictResultBetweenNode(output_rw_type, input_rw_type);
        switch (conflict_result) {
          case ConflictResult::DO_NOTHING:
            GELOGD("No rw conflict.");
            continue;
          case ConflictResult::WRONG_GRAPH:
            has_conflict = true;
            GELOGI("Node %s output rw type is %s, next node %s input_rw_type is %s.It is wrong graph.",
                   node->GetName().c_str(), OutputRWTypeToSerialString(output_rw_type).c_str(),
                   peer_in_node->GetName().c_str(), InputRWTypeToSerialString(input_rw_type).c_str());
            return SUCCESS;
          case ConflictResult::INSERT_IDENTITY:
            GELOGD("There is rw conflict. It will handle later.");
            continue;
          default:
            break;
        }
      }
    }
  }
  return SUCCESS;
}

Status GraphOptimize::HandleMemoryRWConflict(ComputeGraphPtr &compute_graph) const {
  ReInit();
  GE_CHECK_NOTNULL(compute_graph);
  GE_DUMP(compute_graph, "BeforeHandleMemConflict");
  GE_CHK_STATUS(MarkRefRelations(compute_graph), "Mark ref relations failed for %s", compute_graph->GetName().c_str());
  const auto &sub_graph_vec = compute_graph->GetAllSubgraphs();

  // 1.loop all subgraph, mark rw type from inside to outside
  Status ret = MarkRWTypeForAllSubgraph(sub_graph_vec);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][MarkRWTypeForAllSubgraph] failed for %s.", compute_graph->GetName().c_str());
    return ret;
  }
  // 2.loop all node, including node in subgraph and handle memory rw conflict
  for (auto &node : compute_graph->GetAllNodes()) {
    // ignore while subgraph node
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOwnerComputeGraph());
    const auto parent_node = node->GetOwnerComputeGraph()->GetParentNode();
    if ((parent_node != nullptr) && (kWhileOpTypes.count(parent_node->GetType()) > 0)) {
      continue;
    }
    // ignore data / netoutput of subgraph
    if (IsSubgraphInputNode(node) || IsSubgraphOutputNode(node)) {
      continue;
    }

    bool identity_reserved = false;
    AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CANNOT_BE_DELETED, identity_reserved);
    if (identity_reserved) {
      GELOGD("Identity [%s] need to be reserved", node->GetName().c_str());
      continue;
    }
    if (node->GetType() == IDENTITY || node->GetType() == READVARIABLEOP) {
      // split identity
      ret = SplitIdentity(node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Split][Identity] %s failed.", node->GetName().c_str());
        return ret;
      }
      // remove no use identity
      ret = RemoveNoUseIdentity(node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Remove][Identity] %s failed.", node->GetName().c_str());
        return ret;
      }
    }
    // insert Identity
    ret = InsertIdentityAsNeeded(node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Insert][Identity] %s failed.", node->GetName().c_str());
      return ret;
    }
  }
  GE_DUMP(compute_graph, "AfterHandleMemConflict");
  return SUCCESS;
}
}  // namespace ge
