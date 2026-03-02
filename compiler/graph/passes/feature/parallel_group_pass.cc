/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/parallel_group_pass.h"
#include <queue>
#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
const int32_t kMaxRecursionDepth = 10;
const int64_t kLoopType = 1;

auto with_subgraph_node_filter = [](const Node &node) {
  const auto &subgraph_name = node.GetOpDesc()->GetSubgraphInstanceNames();
  return !subgraph_name.empty();
};
}

Status ParallelGroupPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (graph->GetParentGraph() != nullptr) {
    return SUCCESS;
  }

  GELOGD("ParallelGroupPass running");
  if (graph->TopologicalSorting() != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[TopoSort][Graph]Graph:%s topological sort failed.", graph->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Graph:%s topological sort failed when ParallelGroupPass run.",
                      graph->GetName().c_str());
    return FAILED;
  }

  std::unordered_set<std::string> parallel_groups;
  if (ProcessGraphGroupNodes(graph, parallel_groups) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Process][Graph]Process group nodes of graph %s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (graph->TopologicalSorting() != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[TopoSort][Graph]Graph:%s topological sort failed.", graph->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Graph:%s topological sort failed when ParallelGroupPass run.",
                      graph->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}
Status ParallelGroupPass::ProcessOnGraph(
    const ComputeGraphPtr &graph,
    std::map<ge::NodePtr, std::unordered_set<std::string>> &pnodes_2_parallel_groups) const {
  auto candidates = graph->GetDirectNode();
  std::unordered_set<std::string> parallel_groups;
  std::map<std::string, std::vector<NodePtr>> group_nodes;
  for (const auto &node : candidates) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    std::string group_name;
    if (AttrUtils::GetStr(op_desc, ATTR_NAME_PARALLEL_GROUP, group_name)) {
      group_nodes[group_name].push_back(node);
      parallel_groups.insert(group_name);
      GELOGD("Find group node:%s, group_name:%s", node->GetName().c_str(), group_name.c_str());
    }

    const auto &subgraph_name = op_desc->GetSubgraphInstanceNames();
    if (subgraph_name.empty()) {
      continue;
    }
    auto iter = pnodes_2_parallel_groups.find(node);
    if (iter != pnodes_2_parallel_groups.end()) {
      for (const auto &sub_parallel_group : iter->second) {
        parallel_groups.insert(sub_parallel_group);
        group_nodes[sub_parallel_group].emplace_back(node);
      }
    }
  }

  std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> node_2_switch_merge;
  if (ProcessGroupNodeInSwitch(graph, node_2_switch_merge) != SUCCESS) {
    GELOGE(FAILED, "[Process][Node]Process group node in switch failed, graph:%s.", graph->GetName().c_str());
    return FAILED;
  }

  for (const auto &itr : group_nodes) {
    const auto &nodes = itr.second;
    if (nodes.empty()) {
      continue;
    }
    NodePtr pre_node = nodes[0];
    NodePtr cur_node = nullptr;
    for (std::size_t i = 1; i < nodes.size(); i++) {
      cur_node = nodes[i];
      GELOGD("Original add ctrl anchor for node:%s->%s", pre_node->GetName().c_str(), cur_node->GetName().c_str());
      if (ReplaceWithSwitchAndMerge(pre_node, cur_node, node_2_switch_merge) != SUCCESS) {
        GELOGE(FAILED, "[Replace][Node]Replace switch and merges for nodes: %s and %s failed.",
               pre_node->GetName().c_str(), cur_node->GetName().c_str());
        return FAILED;
      }
      pre_node = cur_node;
    }
  }
  // get parent node
  pnodes_2_parallel_groups[graph->GetParentNode()] = parallel_groups;
  return SUCCESS;
}

Status ParallelGroupPass::ProcessGraphGroupNodes(ComputeGraphPtr graph,
                                                 std::unordered_set<std::string> &parallel_groups) const {
  // get all node with subgraph
  auto pnodes_with_subgraph = graph->GetNodes(false, with_subgraph_node_filter, nullptr);
  std::map<ge::NodePtr, std::unordered_set<std::string>> pnodes_2_parallel_groups;
  for (auto iter = pnodes_with_subgraph.end() -1; iter != pnodes_with_subgraph.begin() - 1; --iter) {
    const auto &pnode = *iter;
    const auto &subgraph_name = pnode->GetOpDesc()->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_name.rbegin(); name_iter != subgraph_name.rend(); ++name_iter) {
      const auto &sub_graph = graph->GetSubgraph(*name_iter);
      GE_CHECK_NOTNULL(sub_graph);
      GE_ASSERT_SUCCESS(ProcessOnGraph(sub_graph, pnodes_2_parallel_groups));
    }
  }
  GE_ASSERT_SUCCESS(ProcessOnGraph(graph, pnodes_2_parallel_groups));
  parallel_groups = pnodes_2_parallel_groups[nullptr];
  return SUCCESS;
}

Status ParallelGroupPass::AddCtrlEdge(NodePtr pre_node, NodePtr cur_node) const {
  if (pre_node == cur_node) {
    GELOGD("Pre_node and cur_node are same, no need add anchor");
    return SUCCESS;
  }
  if (cur_node->GetType() == DATA) {
    return SUCCESS;
  }
  auto in_nodes = cur_node->GetInAllNodes();
  for (const auto &node :  in_nodes) {
    if (pre_node == node) {
      GELOGD("Node:%s and %s already linked", pre_node->GetName().c_str(),
             cur_node->GetName().c_str());
      return SUCCESS;
    }
  }
  GELOGD("Finally add ctrl anchor for node:%s->%s", pre_node->GetName().c_str(), cur_node->GetName().c_str());
  return GraphUtils::AddEdge(pre_node->GetOutControlAnchor(), cur_node->GetInControlAnchor());
}

Status ParallelGroupPass::ProcessGroupNodeInSwitch(ComputeGraphPtr graph,
    std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge) const {

  std::string type;
  auto direct_nodes = graph->GetDirectNode();
  for (const auto &node : direct_nodes) {
    type = node->GetType();
    if (type != STREAMSWITCH) {
      continue;
    }

    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (IsBigSmallLoopStreamSwitch(node->GetOpDesc()) || IsWhileStreamSwitch(node->GetOpDesc())) {
      continue;
    }

    std::vector<NodePtr> merge_nodes;
    std::set<NodePtr> group_nodes;
    std::set<std::string> stream_labels;

    FindGroupNodeAndMerge(node, group_nodes, merge_nodes, stream_labels);

    if (merge_nodes.empty() || (!group_nodes.empty() && stream_labels.size() > 1)) {
      GELOGE(FAILED, "[Process][Node]Cannot find merge node or exist switch nestification, switch node:%s,"
             "merge_vec size:%zu, stream_labels size:%zu, graph:%s.", node->GetName().c_str(),
             merge_nodes.size(), stream_labels.size(), graph->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Cannot find merge node or exist switch nest, switch node:%s,"
                         "merge_vec size: %zu, stream_labels size: %zu, graph:%s.", node->GetName().c_str(),
                         merge_nodes.size(), stream_labels.size(), graph->GetName().c_str());
      return FAILED;
    }

    std::sort(merge_nodes.begin(), merge_nodes.end(), [] (NodePtr a, NodePtr b) -> bool {
      if ((a->GetOpDesc() == nullptr) || (b->GetOpDesc() == nullptr)) {
        return false;
      }
      return (a->GetOpDesc()->GetId() < b->GetOpDesc()->GetId());
    });

    NodePtr cast_node = NodeUtils::GetInDataNodeByIndex(*node, 0);
    GE_CHECK_NOTNULL(cast_node);
    if (MappingNodeToSwitchAndMerge(group_nodes, merge_nodes, cast_node, node, node_2_switch_merge) != SUCCESS) {
      GELOGE(FAILED, "[Mapping][Node]Mapping node to switch and merge failed, graph:%s.", graph->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

void ParallelGroupPass::FindGroupNodeAndMerge(NodePtr stream_switch_node, std::set<NodePtr> &group_nodes,
    std::vector<NodePtr> &merge_nodes, std::set<std::string> &stream_labels) const {
  std::string type;
  std::deque<NodePtr> candidates;
  std::set<NodePtr> visited;

  candidates.push_back(stream_switch_node);
  while (!candidates.empty()) {
    NodePtr tmp_node = candidates.front();
    candidates.pop_front();
    for (const auto &out_node : tmp_node->GetOutAllNodes()) {
      type = out_node->GetType();
      if (type == STREAMMERGE) {
        merge_nodes.emplace_back(out_node);
        continue;
      }
      const auto &op = out_node->GetOpDesc();
      if (op != nullptr && op->HasAttr(ATTR_NAME_PARALLEL_GROUP)) {
        group_nodes.emplace(out_node);
      }
      if (visited.count(out_node) > 0) {
        continue;
      }
      candidates.push_back(out_node);
      visited.insert(out_node);
      std::string stream_label;
      if (ge::AttrUtils::GetStr(out_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
        stream_labels.insert(stream_label);
      }
    }
  }
}

Status ParallelGroupPass::MappingNodeToSwitchAndMerge(const std::set<NodePtr> &group_nodes,
    const std::vector<NodePtr> &merge_nodes, const NodePtr &cast_node, const NodePtr &switch_node,
    std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge) const {
  for (const auto &group_node : group_nodes) {
    auto itr = node_2_switch_merge.find(group_node);
    if (itr != node_2_switch_merge.end()) {
      auto &tmp = itr->second;
      auto &switch_set = tmp.first;
      const auto &merge_node = tmp.second;
      GELOGD("Find group node: %s in switch %s and merge %s.",
             group_node->GetName().c_str(), switch_node->GetName().c_str(), merge_node->GetName().c_str());
      if (merge_node != merge_nodes.back()) {
        GELOGE(FAILED, "[Mapping][Node]Has two different merge nodes: %s and %s, graph's structure is invalid",
               merge_node->GetName().c_str(), merge_nodes.back()->GetName().c_str());
        REPORT_INNER_ERR_MSG("E19999", "Has two different merge nodes: %s and %s, graph's structure is invalid",
                           merge_node->GetName().c_str(), merge_nodes.back()->GetName().c_str());
        return FAILED;
      }
      switch_set.insert(cast_node);
    } else {
      node_2_switch_merge.emplace(group_node, std::make_pair(std::set<NodePtr>{cast_node}, merge_nodes.back()));
    }
  }
  return SUCCESS;
}

Status ParallelGroupPass::ReplaceWithSwitchAndMerge(NodePtr pre_node, NodePtr cur_node,
    const std::map<NodePtr, std::pair<std::set<NodePtr>, NodePtr>> &node_2_switch_merge) const {
  auto pre_itr = node_2_switch_merge.find(pre_node);
  auto cur_itr = node_2_switch_merge.find(cur_node);
  if (pre_itr != node_2_switch_merge.end()) {
    if (cur_itr != node_2_switch_merge.end()) {
      const auto &pre_set = pre_itr->second.first;
      const auto &cur_set = cur_itr->second.first;
      if (!HasSameSwitch(pre_set, cur_set)) {
        pre_node = pre_itr->second.second;
        for (const auto &switch_node : cur_itr->second.first) {
          if (AddCtrlEdge(pre_node, switch_node) != SUCCESS) {
            GELOGE(FAILED, "[AddEdge][Node]Add edge for nodes: %s->%s failed.",
                   pre_node->GetName().c_str(), switch_node->GetName().c_str());
            REPORT_INNER_ERR_MSG("E19999", "[AddEdge][Node]Add edge for nodes: %s->%s failed.",
                              pre_node->GetName().c_str(), switch_node->GetName().c_str());
            return FAILED;
          }
        }
      }
      return SUCCESS;
    } else {
      pre_node = pre_itr->second.second;
      return AddCtrlEdge(pre_node, cur_node);
    }
  } else {
    if (cur_itr != node_2_switch_merge.end()) {
      GE_CHECK_NOTNULL(pre_node->GetOpDesc());
      for (const auto &switch_node : cur_itr->second.first) {
        GE_CHECK_NOTNULL(switch_node->GetOpDesc());
        int64_t pre_id = pre_node->GetOpDesc()->GetId();
        int64_t switch_id = switch_node->GetOpDesc()->GetId();
        NodePtr first_node = pre_node;
        NodePtr second_node = switch_node;
        if (pre_id > switch_id && IsIndirectConnect(switch_node, pre_node)) {
          // avoid ring, merge->pre_node
          first_node = cur_itr->second.second;
          second_node = pre_node;
        }
        if (AddCtrlEdge(first_node, second_node) != SUCCESS) {
          GELOGE(FAILED, "[AddEdge][Node]Add edge for nodes: %s->%s failed.",
                 first_node->GetName().c_str(), second_node->GetName().c_str());
          REPORT_INNER_ERR_MSG("E19999", "[AddEdge][Node]Add edge for nodes: %s->%s failed.",
                            first_node->GetName().c_str(), second_node->GetName().c_str());
          return FAILED;
        }
      }
    } else {
      return AddCtrlEdge(pre_node, cur_node);
    }
  }
  return SUCCESS;
}

bool ParallelGroupPass::HasSameSwitch(const std::set<NodePtr> &switch_set1,
                                      const std::set<NodePtr> &switch_set2) const {
  for (const auto &node1 : switch_set1) {
    const auto itr = switch_set2.find(node1);
    if (itr != switch_set2.end()) {
      return true;
    }
  }
  return false;
}

bool ParallelGroupPass::IsBigSmallLoopStreamSwitch(OpDescPtr switch_op_desc) const {
  return !AttrUtils::HasAttr(switch_op_desc, ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG);
}

bool ParallelGroupPass::IsWhileStreamSwitch(OpDescPtr switch_op_desc) const {
  int64_t stream_switch_type = -1;
  return (AttrUtils::GetInt(switch_op_desc, ATTR_NAME_STREAM_SWITCH_TYPE, stream_switch_type) &&
    stream_switch_type == kLoopType);
}

bool ParallelGroupPass::IsIndirectConnect(const NodePtr &node_a, const NodePtr &node_b) const {
  if (node_a == nullptr || node_b == nullptr || node_b->GetOpDesc() == nullptr) {
    GELOGW("node_a or node_b is nullptr.");
    return false;
  }
  int64_t end_id = node_b->GetOpDesc()->GetId();
  std::queue<NodePtr> nodes;
  nodes.push(node_a);
  while (!nodes.empty()) {
    NodePtr tmp_node = nodes.front();
    nodes.pop();
    if ((tmp_node == nullptr) || (tmp_node->GetOpDesc() == nullptr) || (tmp_node->GetOpDesc()->GetId() > end_id)) {
      continue;
    }
    if (tmp_node == node_b) {
      return true;
    }
    for (const auto &out_node : tmp_node->GetOutAllNodes()) {
      nodes.push(out_node);
    }
  }
  return false;
}

REG_PASS_OPTION("ParallelGroupPass").LEVELS(OoLevel::kO0);
} // namespace ge
