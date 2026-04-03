/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/save_pass.h"

#include <string>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
const char *const kSave = "Save";
const char *const kVar = "Variable";
const char *const kVarIsSave = "save_checkpoint";
const char *const kVarAttrVarIsSave = "_var_is_save";
}  // namespace

Status SavePass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  std::vector<NodePtr> front_nodes;
  std::vector<uint8_t> out_index;
  std::vector<NodePtr> del_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kSave) {
      for (auto &in : node->GetAllInDataAnchors()) {
        auto out_anchor = in->GetPeerOutAnchor();
        if (out_anchor != nullptr) {
          ge::NodePtr peer_node = out_anchor->GetOwnerNode();
          if (peer_node->GetType() == kVar) {
            front_nodes.emplace_back(peer_node);
            out_index.emplace_back(out_anchor->GetIdx());
            ge::OpDescPtr op_desc = peer_node->GetOpDesc();
            GE_CHECK_NOTNULL(op_desc);
            GE_IF_BOOL_EXEC(!ge::AttrUtils::SetStr(op_desc, kVarAttrVarIsSave, kVarIsSave),
                            REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", kVarAttrVarIsSave,
                                              op_desc->GetName().c_str(), op_desc->GetType().c_str());
                            GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", kVarAttrVarIsSave,
                                   op_desc->GetName().c_str(), op_desc->GetType().c_str());
                            return INTERNAL_ERROR);
          }
        }
      }
      del_nodes.emplace_back(node);
    }
  }
  // add output nodes for save
  std::vector<std::pair<NodePtr, int32_t>> out_nodes_info{};
  for (size_t i = 0; i < front_nodes.size(); i++) {
    GELOGI("graph add out_node %s:%d", front_nodes[i]->GetName().c_str(), out_index[i]);
    out_nodes_info.emplace_back(std::pair<NodePtr, int32_t>(front_nodes[i], out_index[i]));
  }
  graph->AppendGraphOutNodesInfo(out_nodes_info);
  GE_IF_BOOL_EXEC(front_nodes.size() != 0U, GE_ASSERT_SUCCESS(graph->CreateOrUpdateNetoutput(true)););

  // delete save node
  for (auto &node_ptr : del_nodes) {
    auto ret = graph->RemoveNode(node_ptr);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) from graph:%s failed",
                        node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), graph->GetName().c_str());
      GELOGE(ret, "[Remove][Node] %s(%s) from graph:%s failed",
             node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), graph->GetName().c_str());
      return ret;
    }

    // update Target list
    std::vector<NodePtr> graph_target = graph->GetGraphTargetNodesInfo();
    auto iter = find(graph_target.begin(), graph_target.end(), node_ptr);
    if (iter != graph_target.end()) {
      GELOGI("Current node %s is as Target, remove it from target vector.", node_ptr->GetName().c_str());
      graph_target.erase(iter);
      graph->SetGraphTargetNodesInfo(graph_target);
    }
  }

  return SUCCESS;
}

REG_PASS_OPTION("SavePass").LEVELS(OoLevel::kO0);
}  // namespace ge
