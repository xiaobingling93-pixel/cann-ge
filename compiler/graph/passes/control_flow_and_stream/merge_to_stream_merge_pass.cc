/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/control_flow_and_stream/merge_to_stream_merge_pass.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/plugin/ge_make_unique_util.h"
#include "ge/ge_api_types.h"
#include "common/omg_util/omg_util.h"

namespace ge {
Status MergeToStreamMergePass::Run(ComputeGraphPtr graph) {
  GELOGD("MergeToStreamMergePass Enter");

  bypass_nodes_.clear();
  for (const auto &node : graph->GetDirectNode()) {
    std::string type;
    GE_CHK_STATUS_RET(GetOriginalType(node, type),
                      "[Get][OriginalType] of node in graph:%s failed.", graph->GetName().c_str());
    if ((type != MERGE) && (type != REFMERGE)) {
      continue;
    }

    OpDescPtr merge_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(merge_op_desc);
    if (merge_op_desc->HasAttr(ATTR_INSERT_BY_MBATCH)) {
      GE_CHK_STATUS_RET(AddActiveNodes(graph, node), "Merge add active node failed.");
      auto status = SetStreamLabel(node, node->GetName());
      if (status != ge::SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Set stream_label:%s to op:%s(%s) failed",
                          node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
               node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        return status;
      }
    } else {
      GE_CHK_STATUS_RET(ReplaceMergeNode(graph, node),
                        "[Replace][MergeNode] %s in graph:%s failed.", node->GetName().c_str(),
                        graph->GetName().c_str());
    }
  }

  for (const auto &node : bypass_nodes_) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveNodeWithoutRelink(graph, node) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                                       node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
                     return FAILED,
                     "[Remove][Node] %s(%s) without relink in graph:%s failed",
                     node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
  }

  GELOGD("MergeToStreamMergePass Leave");
  return SUCCESS;
}

///
/// @brief Replace Merge Op
/// @param [in] graph
/// @param [in] merge_node
/// @return Status
///
Status MergeToStreamMergePass::ReplaceMergeNode(const ComputeGraphPtr &graph, const NodePtr &merge_node) const {
  OpDescPtr merge_op_desc = merge_node->GetOpDesc();
  GE_CHECK_NOTNULL(merge_op_desc);
  ge::OpDescUtilsEx::SetType(merge_op_desc, STREAMMERGE);

  return AddActiveNodes(graph, merge_node);
}

///
/// @brief Add StreamActive Op before StreamMerge/Merge
/// @param [in] graph
/// @param [in] node
/// @return Status
///
Status MergeToStreamMergePass::AddActiveNodes(const ComputeGraphPtr &graph, const NodePtr &node) const {
  GE_CHK_BOOL_EXEC(node != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
                   return FAILED, "[Check][Param] Param of pre node is nullptr.");
  int64_t group_index = -1;
  (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index);
  for (const InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    const std::string &type = in_node->GetType();
    // For WhileLoop, no need to add active nodes here, since which have been added in NextIterationPass.
    GE_IF_BOOL_EXEC((type == ENTER) || (type == REFENTER) || (type == NEXTITERATION) || (type == REFNEXTITERATION),
                    continue);
    NodePtr active_node = CreateActiveNode(graph, in_node);
    GE_CHK_BOOL_EXEC(active_node != nullptr, return FAILED,
                     "[Create][StreamActiveNode] failed, in_node:%s.", in_node->GetName().c_str());
    GE_CHK_STATUS(GraphUtils::AddEdge(active_node->GetOutControlAnchor(), node->GetInControlAnchor()),
                  "[Add][CtrlEdge] between %s and %s failed.",
                  active_node->GetName().c_str(), node->GetName().c_str());
    if (SetActiveLabelList(active_node, { node->GetName() }) != SUCCESS) {
      GELOGE(FAILED, "[Set][ActiveLabelList] for node %s failed.", active_node->GetName().c_str());
      return FAILED;
    }
    SetControlFlowGroup(active_node, group_index);
  }

  return SUCCESS;
}

///
/// @brief Create Active Op
/// @param [in] graph
/// @param [in] node
/// @return ge::NodePtr
///
NodePtr MergeToStreamMergePass::CreateActiveNode(const ComputeGraphPtr &graph, const NodePtr &node) const {
  const std::string &node_name = node->GetName() + "_" + STREAMACTIVE;
  GELOGI("Create StreamActive op:%s.", node_name.c_str());
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, STREAMACTIVE);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed, name:%s, type:%s.", node_name.c_str(), STREAMACTIVE);
    GELOGE(FAILED, "[New][OpDesc] failed, name:%s, type:%s.", node_name.c_str(), STREAMACTIVE);
    return nullptr;
  }

  NodePtr active_node = graph->InsertNode(node, op_desc);
  GE_CHK_BOOL_EXEC(active_node != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s failed",
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
                   return nullptr,
                   "[Add][Node] %s(%s) to graph:%s failed",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph->GetName().c_str());
  GE_IF_BOOL_EXEC(GraphUtils::AddEdge(node->GetOutControlAnchor(), active_node->GetInControlAnchor()) != SUCCESS,
                  REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                                    node->GetName().c_str(), node->GetType().c_str(),
                                    active_node->GetName().c_str(), active_node->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                         node->GetName().c_str(), node->GetType().c_str(),
                         active_node->GetName().c_str(), active_node->GetType().c_str());
                  return nullptr);
  GE_IF_BOOL_EXEC(SetSwitchBranchNodeLabel(active_node, node_name) != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "[Set][SwitchBranchNodeLabel] failed, node:%s, label:%s",
                         active_node->GetName().c_str(), node_name.c_str());
                  return nullptr);

  return active_node;
}

REG_PASS_OPTION("MergeToStreamMergePass").LEVELS(OoLevel::kO0);
}  // namespace ge
