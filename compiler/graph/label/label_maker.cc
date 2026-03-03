/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/label/label_maker.h"

#include <set>
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/constant_utils.h"

namespace ge {
namespace {
bool IsCalcType(const std::string &node_type) {
  return !OpTypeUtils::IsVarLikeNode(node_type) && !OpTypeUtils::IsDataNode(node_type) &&
         (node_type != CONSTANTOP) && (node_type != CONSTANT);
}
}  // namespace
/**
 * @ingroup ge
 * @brief Link node to graph head.
 * @param [in] graph: graph for add node.
 * @param [in] node: Node add to graph head.
 * @return: void
 */
void LabelMaker::LinkToGraphHead(const ComputeGraphPtr &graph, const NodePtr &node) const {
  for (auto &n : graph->GetDirectNode()) {
    if (!IsCalcType(n->GetType())) {
      continue;
    }

    const auto nodes = n->GetInDataNodes();
    if (nodes.empty()) {
      continue;
    }

    bool is_head_node = true;
    for (auto &in_node : nodes) {
      if (IsCalcType(in_node->GetType())) {
        is_head_node = false;
        break;
      }
    }

    if (!is_head_node) {
      continue;
    }

    if (GraphUtils::AddEdge(node->GetOutControlAnchor(), n->GetInControlAnchor()) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge from %s to %s in graph:%s fail", node->GetName().c_str(),
                        n->GetName().c_str(), graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] from %s to %s failed.", node->GetName().c_str(), n->GetName().c_str());
    }
  }
}

/**
 * @ingroup ge
 * @brief Link node to graph tail.
 * @param [in] graph: graph for add node.
 * @param [in] node: Node add to graph tail.
 * @return: void
 */
void LabelMaker::LinkToGraphTail(const ComputeGraphPtr &graph, const NodePtr &node) const {
  auto tail = graph->FindFirstNodeMatchType(NETOUTPUT);
  while (tail != nullptr) {
    auto nodes = tail->GetOutControlNodes();
    if (!nodes.empty()) {
      tail = nodes.at(0);
      continue;
    }

    if (GraphUtils::AddEdge(tail->GetOutControlAnchor(), node->GetInControlAnchor()) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge from %s to %s in graph:%s fail", tail->GetName().c_str(),
                        node->GetName().c_str(), graph->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] from %s to %s failed.", tail->GetName().c_str(), node->GetName().c_str());
    }
    return;
  }
}

/**
 * @ingroup ge
 * @brief Add StreamActive node at graph front.
 * @param [in] graph: graph for add node.
 * @param [in] name: stream active node name.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddStreamActive(const ComputeGraphPtr &graph, const std::string &name) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  const auto &node_list = graph->GetDirectNode();
  if (node_list.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Check param graph:%s has no node", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] LabelSet: Graph %s node is empty.", graph->GetName().c_str());
    return nullptr;
  }

  OpDescPtr op_desc = MakeShared<OpDesc>(name, STREAMACTIVE);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  const std::vector<std::string> &active_label_list = GetActiveLabelList(graph);
  GELOGI("StreamActive: Create node %s, active label list size: %zu.", op_desc->GetName().c_str(),
         active_label_list.size());
  std::vector<uint32_t> active_streams;
  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_SWITCH_BRANCH_NODE_LABEL, op_desc->GetName());
  (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_streams);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_SUBGRAPH_FIRST_ACTIVE, true);
  if (!active_label_list.empty()) {
    (void)AttrUtils::SetListStr(op_desc, ATTR_NAME_ACTIVE_LABEL_LIST, active_label_list);
  }
  NodePtr stream_active = graph->AddNodeFront(op_desc);
  GE_CHECK_NOTNULL_EXEC(stream_active, return nullptr);

  LinkToGraphHead(graph, stream_active);
  return stream_active;
}

std::vector<std::string> LabelMaker::GetActiveLabelList(const ComputeGraphPtr &graph) const {
  std::set<std::string> activated_label_list;
  std::set<std::string> stream_label_list;

  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      std::string stream_label;
      if (AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label)) {
        stream_label_list.emplace(stream_label);
      }
      std::vector<std::string> tmp_activated_label_list;
      if (AttrUtils::GetListStr(op_desc, ATTR_NAME_ACTIVE_LABEL_LIST, tmp_activated_label_list)) {
        activated_label_list.insert(tmp_activated_label_list.cbegin(), tmp_activated_label_list.cend());
      }
    }
  }
  GELOGD("Existed stream label size: %zu, active label list size: %zu in graph[%s].", stream_label_list.size(),
         activated_label_list.size(), graph->GetName().c_str());

  std::vector<std::string> need_active_label_list;
  if (activated_label_list.empty()) {
    need_active_label_list.insert(need_active_label_list.cbegin(), stream_label_list.cbegin(),
                                  stream_label_list.cend());
  } else {
    for (const auto &stream_label : stream_label_list) {
      if (activated_label_list.count(stream_label) == 0) {
        need_active_label_list.emplace_back(stream_label);
      }
    }
  }
  return need_active_label_list;
}

/**
 * @ingroup ge
 * @brief Add LabelSet node at graph front.
 * @param [in] graph: graph for add node.
 * @param [in] name: label set node name.
 * @param [in] index: label id for set.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelSetEnter(const ComputeGraphPtr &graph, const std::string &name, uint32_t index,
                                     const NodePtr &stream_active) const {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);
  GE_CHECK_NOTNULL_EXEC(stream_active, return nullptr);

  const auto &node_list = graph->GetDirectNode();
  if (node_list.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Check param graph:%s has no node", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] LabelSet: Graph %s node is empty.", graph->GetName().c_str());
    return nullptr;
  }

  OpDescPtr op_desc = MakeShared<OpDesc>(name, LABELSET);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  GELOGI("LabelSet: Create node %s.", op_desc->GetName().c_str());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, index);
  NodePtr label_set = graph->AddNodeFront(op_desc);
  GE_CHECK_NOTNULL_EXEC(label_set, return nullptr);

  if (GraphUtils::AddEdge(label_set->GetOutControlAnchor(), stream_active->GetInControlAnchor()) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge from %s to %s in graph:%s fail", label_set->GetName().c_str(),
                      stream_active->GetName().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] from %s to %s failed.", label_set->GetName().c_str(),
           stream_active->GetName().c_str());
    return nullptr;
  }

  return label_set;
}

/**
 * @ingroup ge
 * @brief Add LabelSet node at graph back.
 * @param [in] graph: graph for add node.
 * @param [in] name: label set node name.
 * @param [in] index: label id for set.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelSetLeave(const ComputeGraphPtr &graph, const std::string &name, uint32_t index) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  OpDescPtr op_desc = MakeShared<OpDesc>(name, LABELSET);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  GELOGI("LabelSet: Create node %s.", op_desc->GetName().c_str());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, index);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_SUBGRAPH_END_NODE, true);
  NodePtr label_set = graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(label_set, return nullptr);

  // Link control edge to graph tail.
  LinkToGraphTail(graph, label_set);
  return label_set;
}

/**
 * @ingroup ge
 * @brief Add LabelGoto node at graph front.
 * @param [in] graph: graph for add node.
 * @param [in] name: label goto node name.
 * @param [in] index: label id for goto.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelGotoEnter(const ComputeGraphPtr &graph, const std::string &name, uint32_t index) const {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  const auto &node_list = graph->GetDirectNode();
  auto it = node_list.begin();
  if (it == node_list.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Check param graph:%s has no node", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] LabelGoto: Graph %s node is empty.", graph->GetName().c_str());
    return nullptr;
  }

  OpDescPtr op_desc = MakeShared<OpDesc>(name, LABELGOTOEX);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  GELOGI("LabelGoto: Create node %s.", op_desc->GetName().c_str());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, index);
  NodePtr label_goto = graph->AddNodeFront(op_desc);
  if (label_goto == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s fail",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] to graph %s failed.", graph->GetName().c_str());
    return nullptr;
  }

  return label_goto;
}

/**
 * @ingroup ge
 * @brief Add LabelGoto node at graph back.
 * @param [in] graph: graph for add node.
 * @param [in] name: label goto node name.
 * @param [in] index: label id for goto.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelGotoLeave(const ComputeGraphPtr &graph, const std::string &name, uint32_t index) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  OpDescPtr op_desc = MakeShared<OpDesc>(name, LABELGOTOEX);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  GELOGI("LabelGoto: Create node %s.", op_desc->GetName().c_str());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, index);
  NodePtr label_goto = graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(label_goto, return nullptr);

  // Link control edge to graph tail.
  LinkToGraphTail(graph, label_goto);
  return label_goto;
}

/**
 * @ingroup ge
 * @brief Add LabelSwitch node at graph front.
 * @param [in] graph: graph for add node.
 * @param [in] name: label switch node name.
 * @param [in] desc: label index data desc.
 * @param [in] labels: label id for switch.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelSwitchEnter(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                                        const std::vector<uint32_t> &labels) const {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  const auto &node_list = graph->GetDirectNode();
  auto it = node_list.begin();
  if (it == node_list.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Check param graph:%s has no node", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] LabelSwitchByIndex: Graph %s node is empty.", graph->GetName().c_str());
    return nullptr;
  }

  OpDescPtr op_desc = MakeShared<OpDesc>(name, LABELSWITCHBYINDEX);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  GELOGI("LabelSwitchByIndex: Create node %s.", op_desc->GetName().c_str());
  if (op_desc->AddInputDesc(desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add input desc into node:%s(%s) in graph:%s fail",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] failed.");
    return nullptr;
  }

  if (!AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, labels)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_LABEL_SWITCH_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s failed.", ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return nullptr;
  }

  NodePtr label_switch = graph->AddNodeFront(op_desc);
  if (label_switch == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s ahead fail",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] to graph %s failed.", graph->GetName().c_str());
    return nullptr;
  }

  return label_switch;
}

/**
 * @ingroup ge
 * @brief Add LabelSwitch node at graph back.
 * @param [in] graph: graph for add node.
 * @param [in] name: label switch node name.
 * @param [in] desc: label index data desc.
 * @param [in] labels: label id for switch.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelSwitchLeave(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                                        const std::vector<uint32_t> &labels) const {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  OpDescPtr op_desc = MakeShared<OpDesc>(name, LABELSWITCHBYINDEX);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);
  (void)AttrUtils::SetBool(op_desc, ATTR_NAME_RTS_LABEL_NODE, true);

  GELOGI("LabelSwitchByIndex: Create node %s.", op_desc->GetName().c_str());
  if (op_desc->AddInputDesc(desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add input desc into node:%s(%s) in graph:%s fail",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] into node:%s(%s) in graph:%s fail",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  if (!AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, labels)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_LABEL_SWITCH_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s failed.", ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return nullptr;
  }

  NodePtr label_switch = graph->AddNode(op_desc);
  GE_CHECK_NOTNULL_EXEC(label_switch, return nullptr);

  // Link control edge to graph tail.
  LinkToGraphTail(graph, label_switch);
  return label_switch;
}

/**
 * @ingroup ge
 * @brief Add Data node at graph front for switch input.
 * @param [in] graph: graph for add node.
 * @param [in] name: label switch node name.
 * @param [in] desc: label index data desc.
 * @param [in] sw_node: switch node for add input.
 * @param [in] parent_index: index for parent node.
 * @return: NodePtr for success / nullptr for fail
 */
NodePtr LabelMaker::AddLabelSwitchIndex(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                                        const NodePtr &sw_node, uint32_t parent_index) const {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  OpDescPtr op_desc = MakeShared<OpDesc>(name, DATA);
  GE_CHECK_NOTNULL_EXEC(op_desc, return nullptr);

  GELOGI("Data: Create node %s.", op_desc->GetName().c_str());
  if (op_desc->AddInputDesc(desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add input desc into node:%s(%s) in graph:%s fail",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] into node:%s(%s) in graph:%s fail",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }
  if (op_desc->AddOutputDesc(desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add output desc into node:%s(%s) in graph:%s fail",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] into node:%s(%s) in graph:%s fail",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
    return nullptr;
  }

  if (!AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return nullptr;
  }
  NodePtr op_data = graph->AddNodeFront(op_desc);
  GE_CHECK_NOTNULL_EXEC(op_data, return nullptr);
  GE_CHECK_NOTNULL_EXEC(graph->AddInputNode(op_data), return nullptr);  // take as input node for memory assign.

  // Link control edge to graph head.
  if (GraphUtils::AddEdge(op_data->GetOutDataAnchor(0), sw_node->GetInDataAnchor(0)) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge from %s to %s in graph:%s fail", op_data->GetName().c_str(),
                      sw_node->GetName().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] to %s failed.", op_data->GetName().c_str());
    return nullptr;
  }

  return op_data;
}
}  // namespace ge
