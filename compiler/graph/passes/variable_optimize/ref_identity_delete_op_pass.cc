/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/variable_optimize/ref_identity_delete_op_pass.h"
#include <map>
#include <stack>
#include "common/op/transop_util.h"

namespace ge {
Status RefIdentityDeleteOpPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != REFIDENTITY) {
      continue;
    }
    int32_t input_index = 0;
    CHECK_FALSE_EXEC(GetRefNode(node, input_index) != nullptr,
                     REPORT_INNER_ERR_MSG("E19999", "Get Ref node of node:%s(%s) failed",
                                       node->GetName().c_str(), node->GetType().c_str());
                     GELOGE(FAILED, "[Get][RefNode] of node:%s(%s) failed",
                            node->GetName().c_str(), node->GetType().c_str());
                     return FAILED);
    CHECK_FALSE_EXEC(DealNoOutputRef(node, graph) == SUCCESS,
                     GELOGE(FAILED, "[Deal][NoOutputRef] for node:%s failed, index:%d",
                            node->GetName().c_str(), input_index);
                     return FAILED);
  }
  return SUCCESS;
}

NodePtr RefIdentityDeleteOpPass::GetRefNode(const NodePtr &node, int32_t &input_index) const {
  OutDataAnchorPtr out_anchor = node->GetOutDataAnchor(0);
  CHECK_FALSE_EXEC(out_anchor != nullptr, return nullptr);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    auto peer_node = peer_in_anchor->GetOwnerNode();
    const auto &peer_op_desc = peer_node->GetOpDesc();
    CHECK_FALSE_EXEC(peer_op_desc != nullptr, return nullptr);
    const auto &peer_input_desc = peer_op_desc->GetInputDescPtr(static_cast<uint32_t>(peer_in_anchor->GetIdx()));
    CHECK_FALSE_EXEC(peer_input_desc != nullptr, return nullptr);
    if (!peer_input_desc->GetRefPortIndex().empty()) {
      input_index = peer_in_anchor->GetIdx();
      return peer_node;
    }
  }
  return nullptr;
}

Status RefIdentityDeleteOpPass::DealNoOutputRef(const NodePtr &ref_identity, const ComputeGraphPtr &graph) const {
  // remove ref identity
  if (GraphUtils::IsolateNode(ref_identity, {0}) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Isolate op:%s(%s) failed",
                      ref_identity->GetName().c_str(), ref_identity->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Isolate][Node] %s, type:%s failed", ref_identity->GetName().c_str(),
           ref_identity->GetType().c_str());
    return FAILED;
  }
  if (GraphUtils::RemoveNodeWithoutRelink(graph, ref_identity) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      ref_identity->GetName().c_str(), ref_identity->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] %s, type:%s without relink in graph:%s failed",
           ref_identity->GetName().c_str(), ref_identity->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  GELOGI("Successfully removed node[%s] from graph[%s]", ref_identity->GetName().c_str(), graph->GetName().c_str());

  return SUCCESS;
}

REG_PASS_OPTION("RefIdentityDeleteOpPass").LEVELS(OoLevel::kO0);
}  // namespace ge
