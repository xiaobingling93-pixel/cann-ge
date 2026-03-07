/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/multi_batch/multi_batch_pass.h"

#include <stack>
#include <unordered_set>
#include "common/plugin/ge_make_unique_util.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "formats/utils/formats_trans_utils.h"

namespace ge {
namespace {
const std::unordered_set<std::string> kGenMaskNodes = {"DropOutGenMask", "DropOutGenMaskV3"};
}
Status MultiBatchPass::Run(ComputeGraphPtr graph) {
  if (graph->GetParentGraph() != nullptr) {
    GELOGI("Subgraph %s skip the MultiBatchPass.", graph->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("MultiBatchPass Enter");
  for (const NodePtr &node : graph->GetDirectNode()) {
    if (node->GetType() == CASE) {
      const auto &func_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(func_desc);
      if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
        GELOGD("Graph: %s not multi-batch, case node: %s", graph->GetName().c_str(), node->GetName().c_str());
        return SUCCESS;
      }
      GE_CHK_STATUS_RET(SetCaseLabel(graph, node),
                        "[Set][CaseLabel] for node:%s(%s) in graph:%s failed",
                        node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    }
  }
  return PreparingForGenMaskParallel(graph);
}

///
/// @ingroup ge
/// @brief Set batch label for Case mode.
/// @param [in] const ComputeGraphPtr &graph: Root/Case graph.
/// @param [in] const NodePtr &case_node: Case Node.
/// @return 0: SUCCESS / others: FAILED
///
Status MultiBatchPass::SetCaseLabel(const ComputeGraphPtr &graph, const NodePtr &case_node) const {
  const auto &func_desc = case_node->GetOpDesc();
  const auto &dynamic_branch_names = func_desc->GetSubgraphInstanceNames();
  for (size_t i = 0; i < dynamic_branch_names.size(); ++i) {
    const auto &subgraph = graph->GetSubgraph(dynamic_branch_names[i]);
    GE_CHECK_NOTNULL(subgraph);

    const std::string batch_label = "Batch_" + std::to_string(i);
    for (const auto &node : subgraph->GetAllNodes()) {
      (void)AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
    }
  }

  return SUCCESS;
}

void MultiBatchPass::GetAllGenMaskNodes(const ComputeGraphPtr &graph,
    std::unordered_map<ComputeGraphPtr, std::vector<NodePtr>> &subgraph_to_gen_mask_nodes) {
  for (const auto &sub_graph : graph->GetAllSubgraphs()) {
    for (const auto &node : sub_graph->GetDirectNode()) {
      std::string op_type;
      (void)GetOriginalType(node, op_type);
      if (kGenMaskNodes.count(op_type) > 0UL) {
        subgraph_to_gen_mask_nodes[sub_graph].emplace_back(node);
      }
    }
  }
}

Status MultiBatchPass::TryToReplaceConstInput(const ComputeGraphPtr &graph,
                                              const std::vector<NodePtr> &gen_mask_nodes) {
  for (const auto &gen_mask_node : gen_mask_nodes) {
    std::string batch_label;
    (void)AttrUtils::GetStr(gen_mask_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
    if (batch_label.empty()) {
      continue;
    }
    for (const auto &in_anchor : gen_mask_node->GetAllInDataAnchors()) {
      const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(peer_out_anchor);
      const auto &in_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(in_node);
      if ((in_node->GetInControlNodes().empty()) ||
          ((in_node->GetType() != CONSTANTOP) && (in_node->GetType() != CONSTANT))) {
        continue;
      }
      GeTensorPtr weight = nullptr;
      const bool get_weight = AttrUtils::MutableTensor(in_node->GetOpDesc(), ATTR_NAME_WEIGHTS, weight);
      if (!get_weight) {
        GELOGE(INTERNAL_ERROR, "Failed to get weight from node:%s, type:%s",
               in_node->GetName().c_str(), in_node->GetType().c_str());
        return INTERNAL_ERROR;
      }
      const auto &const_desc = OpDescUtils::CreateConstOp(weight);
      GE_CHECK_NOTNULL(const_desc);
      const_desc->SetName(batch_label + "_" + const_desc->GetName());
      const auto &const_node = graph->AddNodeFront(const_desc);
      GE_CHECK_NOTNULL(const_node);
      const auto ret = GraphUtils::ReplaceEdgeSrc(peer_out_anchor, in_anchor, const_node->GetOutDataAnchor(0));
      if (ret != GRAPH_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "Failed to replace edge, src node:%s, new src node:%s, dst node:%s, dst input idx:%d.",
               in_node->GetName().c_str(), const_node->GetName().c_str(),
               gen_mask_node->GetName().c_str(), in_anchor->GetIdx());
        return INTERNAL_ERROR;
      }
      GELOGI("Replace edge, src node:%s, new src node:%s, dst node:%s, dst input idx:%d.",
             in_node->GetName().c_str(), const_node->GetName().c_str(),
             gen_mask_node->GetName().c_str(), in_anchor->GetIdx());
    }
  }
  return SUCCESS;
}

Status MultiBatchPass::PreparingForGenMaskParallel(const ComputeGraphPtr &graph) {
  std::unordered_map<ComputeGraphPtr, std::vector<NodePtr>> subgraph_to_gen_mask_nodes;
  GetAllGenMaskNodes(graph, subgraph_to_gen_mask_nodes);
  for (const auto &item : subgraph_to_gen_mask_nodes) {
    if (TryToReplaceConstInput(item.first, item.second) != SUCCESS) {
      GELOGE(FAILED, "[Replace][ConstInput] Failed to replace const input, graph:%s.",
             item.first->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("MultiBatchPass").LEVELS(OoLevel::kO1);
}  // namespace ge
