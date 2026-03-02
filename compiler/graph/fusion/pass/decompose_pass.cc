/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/fusion/pass/decompose_pass.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "ge/fusion/subgraph_boundary.h"
#include "ge/fusion/graph_rewriter.h"
#include "graph/fusion/fusion_utils.h"

namespace ge {
namespace fusion {
namespace {
std::unique_ptr<SubgraphBoundary> BuildSubgraphBoundary(const NodePtr &node) {
  std::vector<SubgraphInput> subgraph_inputs;
  std::map<OutDataAnchorPtr, size_t> out_anchor_2_idx;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto peer_out_anchor =  in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    NodeIo node_input = {NodeAdapter::Node2GNode(node), in_anchor->GetIdx()};
    const auto iter = out_anchor_2_idx.find(peer_out_anchor);
    if (iter == out_anchor_2_idx.cend()) {
      auto peer_out_node = peer_out_anchor->GetOwnerNode();
      SubgraphInput subgraph_input({node_input});
      subgraph_inputs.emplace_back(subgraph_input);
      out_anchor_2_idx.emplace(peer_out_anchor, subgraph_inputs.size() - 1);
    } else {
      auto idx = iter->second;
      GE_ASSERT_TRUE(idx < subgraph_inputs.size());
      subgraph_inputs[idx].AddInput(node_input);
    }
  }
  std::vector<SubgraphOutput> subgraph_outputs;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(out_anchor);
    NodeIo node_output = {NodeAdapter::Node2GNode(node), out_anchor->GetIdx()};
    subgraph_outputs.emplace_back(node_output);
  }
  return ge::MakeUnique<SubgraphBoundary>(subgraph_inputs, subgraph_outputs);
}

bool IsMatchAnyOfType(const std::string &op_type,  const std::vector<AscendString> &matched_op_types) {
  return std::any_of(
      matched_op_types.cbegin(), matched_op_types.cend(),
      [&op_type](const AscendString &matched_op_type) { return (op_type == matched_op_type.GetString()); });
}

std::vector<GNode> MatchFromAllNodes(const ComputeGraphPtr &root_graph, const std::vector<AscendString> &op_types) {
  std::vector<GNode> matched_nodes;
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (IsMatchAnyOfType(node->GetType(), op_types)) {
      matched_nodes.emplace_back(NodeAdapter::Node2GNode(node));
    }
  }
  return matched_nodes;
}
} // namespace
DecomposePass::DecomposePass(const vector<AscendString> &op_types) : op_types_(op_types) {}

Status fusion::DecomposePass::Run(GraphPtr &graph, CustomPassContext &pass_context) {
  (void) pass_context;
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
  const auto matched_nodes = MatchFromAllNodes(compute_graph, op_types_);
  if (matched_nodes.empty()) {
    GELOGD("[MATCH] not find any op with match_type");
    return NOT_CHANGED;
  }
  bool is_changed = false;
  const uint32_t graph_id = compute_graph->GetGraphID();
  std::string pass_name = pass_context.GetPassName().GetString();
  // 记录匹配节点的match_times与effect_times
  std::map<AscendString, std::pair<int32_t, int32_t>> fusion_info_map;
  for (auto op_type : op_types_) {
    fusion_info_map.emplace(std::piecewise_construct,
      std::forward_as_tuple(op_type),
      std::forward_as_tuple(0,0));
  }
  for (const auto &g_node : matched_nodes) {
    auto node = NodeAdapter::GNode2Node(g_node);
    AscendString g_node_type;
    GE_ASSERT_GRAPH_SUCCESS(g_node.GetType(g_node_type));
    auto it_fusion_info = fusion_info_map.find(g_node_type);
    GE_ASSERT_TRUE(it_fusion_info != fusion_info_map.end());
    it_fusion_info->second.first++;
    if (!MeetRequirements(g_node)) {
      GELOGD("Node [%s][%s] is not match requires", node->GetNamePtr(), node->GetTypePtr());
      continue;
    }

    auto boundary = BuildSubgraphBoundary(node);
    GE_ASSERT_NOTNULL(boundary, "Failed to build boundary for node [%s][%s]", node->GetNamePtr(), node->GetTypePtr());
    auto replacement = Replacement(g_node);
    GE_ASSERT_NOTNULL(replacement, "Got null replacement graph");
    (void)FusionUtils::MarkPassNameOnReplacementNodes(replacement, boundary, pass_name);
    GE_ASSERT_SUCCESS(SubgraphRewriter::Replace(*boundary, *replacement), "Failed to replace node [%s][%s] with replacement graph",
                      node->GetNamePtr(), node->GetTypePtr());
    if (!is_changed) {
      is_changed = true;
    }
    GELOGI("Replace node[%s][%s] with %s", node->GetNamePtr(), node->GetTypePtr(), FusionUtils::ToString(replacement).c_str());
    it_fusion_info->second.second++; // effect_times ++
  }
  for (auto &pair : fusion_info_map) {
    FusionUtils::RecordFusionStatistic(compute_graph->GetSessionID(), to_string(graph_id),pass_name,
      pair.second.first, pair.second.second);
    GELOGD("GraphId[%d], GraphFusionPass[%s]: pattern=%s, matched_times=%d, effected_times=%d", graph_id, pass_name.c_str(),
      pair.first.GetString(), pair.second.first, pair.second.second);
  }
  return is_changed ? SUCCESS : NOT_CHANGED;
}

bool DecomposePass::MeetRequirements(const GNode &matched_node) {
  (void)matched_node;
  return true;
}
} // namespace fusion
}  // namespace ge
