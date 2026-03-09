/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_lint.h"
#include "framework/common/debug/ge_log.h"
#include "common/omg_util/omg_util.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/utils/graph_utils.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/range_vistor.h"

namespace ge {
namespace {
bool IsNodeWithSubGraph(const OpDesc *const op_desc) {
  const auto &subgraph_instance_names = op_desc->GetSubgraphInstanceNames();
  return std::any_of(subgraph_instance_names.cbegin(), subgraph_instance_names.cend(),
                     [](const std::string &instance_name) { return !instance_name.empty(); });
}

bool IsNodeTypeEqual(const char_t *const node_type, const char_t *const target_type) {
  return (strcmp(node_type, target_type) == 0);
}

graphStatus MarkSingleNode(const NodePtr &node, GraphLint::NodeInputRWDesc &input_rw_desc) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &input_name_index = op_desc->GetAllInputName();
  input_rw_desc.Init(input_name_index.size());

  std::string node_type;
  GE_ASSERT_GRAPH_SUCCESS(GetOriginalType(node, node_type));
  if (IsNodeTypeEqual(node_type.c_str(), CMO)) {
    for (size_t i = 0u; i < input_name_index.size(); ++i) {
      GE_ASSERT_GRAPH_SUCCESS(input_rw_desc.SetInputRwType(i, GraphLint::RWType::kCanIgnore));
    }
  } else {
    for (const auto &name_index : input_name_index) {
      const int32_t out_index = op_desc->GetOutputIndexByName(name_index.first);
      if (out_index != -1) {
        GE_ASSERT_GRAPH_SUCCESS(input_rw_desc.SetInputRwType(name_index.second, GraphLint::RWType::kWritable));
      } else {
        GE_ASSERT_GRAPH_SUCCESS(input_rw_desc.SetInputRwType(name_index.second, GraphLint::RWType::kReadOnly));
      }
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(input_rw_desc.SetIsMarked(), "Node %s input rw type is not all setted.", node->GetTypePtr());
  return GRAPH_SUCCESS;
}

graphStatus RefreshParentNodeIfNeed(const NodePtr &node, std::vector<GraphLint::NodeInputRWDesc> &nodes_2_rw_descs) {
  const auto parent_node = node->GetOwnerComputeGraphBarePtr()->GetParentNode();
  if (parent_node == nullptr) {
    return GRAPH_SUCCESS;
  }
  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    const auto peer_in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_in_node);
    if (!IsNodeTypeEqual(peer_in_node->GetTypePtr(), DATA)) {
      continue;
    }

    auto current_input_rw_type = GraphLint::RWType::kInvalid;
    GE_ASSERT_GRAPH_SUCCESS(
        nodes_2_rw_descs[node->GetOpDescBarePtr()->GetId()].GetInputRwType(in_anchor->GetIdx(), current_input_rw_type));
    GE_ASSERT_TRUE(current_input_rw_type != GraphLint::RWType::kInvalid);

    auto &parent_node_rw_desc = nodes_2_rw_descs[parent_node->GetOpDescBarePtr()->GetId()];
    if (!parent_node_rw_desc.IsInit()) {
      parent_node_rw_desc.Init(parent_node->GetAllInDataAnchorsSize(), GraphLint::RWType::kReadOnly);
    }
    int32_t parent_node_input_index = INT32_MAX;
    GE_ASSERT_TRUE(AttrUtils::GetInt(peer_in_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_input_index));
    GE_ASSERT_TRUE(parent_node_input_index != INT32_MAX);

    GE_ASSERT_GRAPH_SUCCESS(parent_node_rw_desc.SetInputRwType(parent_node_input_index, current_input_rw_type),
                            "Failed to mark wrapper node %s rw_type, input index %d", parent_node->GetNamePtr(),
                            parent_node_input_index);
  }
  return GRAPH_SUCCESS;
}

graphStatus MarkAllNodesInputRwType(ComputeGraph::Vistor<NodePtr> &all_nodes,
                                    std::vector<GraphLint::NodeInputRWDesc> &nodes_2_rw_descs) {
  for (auto iter = all_nodes.rbegin(); iter != all_nodes.rend(); ++iter) {
    const auto &node = *iter;
    const auto op_desc_ptr = node->GetOpDescBarePtr();
    const auto topo_id = op_desc_ptr->GetId();
    if (IsNodeWithSubGraph(op_desc_ptr)) {
      GE_ASSERT_GRAPH_SUCCESS(nodes_2_rw_descs[topo_id].SetIsMarked(), "Failed to mark node %s, topo id %ld.",
                              op_desc_ptr->GetNamePtr(), topo_id);
    } else {
      GE_ASSERT_GRAPH_SUCCESS(MarkSingleNode(node, nodes_2_rw_descs[topo_id]));
      GE_ASSERT_GRAPH_SUCCESS(RefreshParentNodeIfNeed(node, nodes_2_rw_descs));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus InitializeConnectionMatrix(
    const ComputeGraphPtr &root_graph, const std::vector<ComputeGraphPtr> &all_subgraphs,
    std::unordered_map<const ComputeGraph *, std::unique_ptr<ConnectionMatrix>> &matrixes) {
  auto connect_matrix_ptr = ComGraphMakeUnique<ConnectionMatrix>(root_graph);
  GE_ASSERT_NOTNULL(connect_matrix_ptr);
  GE_ASSERT_GRAPH_SUCCESS(connect_matrix_ptr->Generate(root_graph));
  matrixes[root_graph.get()] = std::move(connect_matrix_ptr);
  for (const auto &subgraph : all_subgraphs) {
    auto sub_connect_matrix_ptr = ComGraphMakeUnique<ConnectionMatrix>(subgraph);
    GE_ASSERT_NOTNULL(sub_connect_matrix_ptr);
    GE_ASSERT_GRAPH_SUCCESS(sub_connect_matrix_ptr->Generate(subgraph));
    matrixes[subgraph.get()] = std::move(sub_connect_matrix_ptr);
  }
  return GRAPH_SUCCESS;
}

graphStatus CollectReadWriteNodesByInAnchors(const std::vector<GraphLint::NodeInputRWDesc> &all_nodes_input_descs,
                                             const std::vector<InDataAnchor *> &in_data_anchors,
                                             std::unordered_set<Node *> &readonly_nodes,
                                             std::unordered_set<Node *> &writeable_nodes) {
  for (const auto &peer_in_anchor : in_data_anchors) {
    const auto peer_in_node = peer_in_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_in_node);

    const auto &peer_in_rw_desc = all_nodes_input_descs[peer_in_node->GetOpDescBarePtr()->GetId()];
    GE_ASSERT_TRUE(peer_in_rw_desc.IsMarked());
    auto peer_in_rw_type = GraphLint::RWType::kInvalid;
    GE_ASSERT_GRAPH_SUCCESS(peer_in_rw_desc.GetInputRwType(peer_in_anchor->GetIdx(), peer_in_rw_type));
    GE_ASSERT_TRUE(peer_in_rw_type != GraphLint::RWType::kInvalid);

    if (peer_in_rw_type == GraphLint::RWType::kWritable) {
      writeable_nodes.emplace(peer_in_node);
    } else if (peer_in_rw_type == GraphLint::RWType::kReadOnly) {
      readonly_nodes.emplace(peer_in_node);
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus GraphLint::Initialize(const ComputeGraphPtr &root_graph) {
  GE_TRACE_START(GraphLintInit);
  if (root_graph->GetParentNodeBarePtr() != nullptr) {
    GELOGW("Only support verify on root graph. Current graph %s is subgraph.", root_graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  GE_TRACE_START(GraphLintInitTopoSorting);
  GE_ASSERT_GRAPH_SUCCESS(root_graph->TopologicalSorting());
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphLintInitTopoSorting, "GraphLintInitTopoSorting");

  // mark all rw type
  GE_TRACE_START(GraphLintMark);
  auto all_nodes = root_graph->GetAllNodes();
  nodes_2_rw_descs_.resize(all_nodes.size());
  GE_ASSERT_GRAPH_SUCCESS(MarkAllNodesInputRwType(all_nodes, nodes_2_rw_descs_));
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphLintMark, "GraphLintMark");

  // init matrix
  GE_TRACE_START(GraphLintGenMatrix);
  const auto &all_subgraphs = root_graph->GetAllSubgraphs();
  graph_2_connection_matrixes_.reserve(all_subgraphs.size() + 1u);
  GE_ASSERT_GRAPH_SUCCESS(InitializeConnectionMatrix(root_graph, all_subgraphs, graph_2_connection_matrixes_));
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphLintGenMatrix, "GraphLintGenMatrix");

  GE_COMPILE_TRACE_TIMESTAMP_END(GraphLintInit, "GraphLintInit");
  return GRAPH_SUCCESS;
}

graphStatus GraphLint::Verify(const ComputeGraphPtr &root_graph) {
  GE_ASSERT_NOTNULL(root_graph);
  GE_ASSERT_GRAPH_SUCCESS(Initialize(root_graph));

  GE_TRACE_START(GraphLintVerify);
  GE_WARN_ASSERT_GRAPH_SUCCESS(VerifyRwConflictPerGraph(root_graph), "There is read&write conflict among graph[%s].",
                               root_graph->GetName().c_str());
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    GE_WARN_ASSERT_GRAPH_SUCCESS(VerifyRwConflictPerGraph(subgraph), "There is read&write conflict among graph[%s].",
                                 subgraph->GetName().c_str());
  }
  GE_COMPILE_TRACE_TIMESTAMP_END(GraphLintVerify, "GraphLintVerify");
  return GRAPH_SUCCESS;
}

graphStatus GraphLint::VerifyRwConflictPerGraph(const ComputeGraphPtr &graph) const {
  for (const auto &node : graph->GetDirectNode()) {
    const auto &out_data_anchors = node->GetAllOutDataAnchors();
    for (const auto &out_data_anchor : out_data_anchors) {
      GE_WARN_ASSERT_GRAPH_SUCCESS(VerifyRwConflictPerOutAnchor(out_data_anchor),
                                   "Verify read&write conflict along out anchor[%d] of node [%s][%s] failed.",
                                   out_data_anchor->GetIdx(), node->GetNamePtr(), node->GetTypePtr());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus GraphLint::VerifyRwConflictPerOutAnchor(const OutDataAnchorPtr &out_anchor) const {
  const auto &peer_in_data_anchors = out_anchor->GetPeerInDataAnchorsPtr();
  if (peer_in_data_anchors.size() < 2U) {
    return GRAPH_SUCCESS;
  }
  std::unordered_set<Node *> writeable_nodes;
  std::unordered_set<Node *> readonly_nodes;
  GE_WARN_ASSERT_GRAPH_SUCCESS(
      CollectReadWriteNodesByInAnchors(nodes_2_rw_descs_, peer_in_data_anchors, readonly_nodes, writeable_nodes));

  const auto current_graph = out_anchor->GetOwnerNodeBarePtr()->GetOwnerComputeGraphBarePtr();
  const auto &current_graph_matrix = graph_2_connection_matrixes_.at(current_graph);
  for (auto &write_node : writeable_nodes) {
    for (auto &other_write_node : writeable_nodes) {
      if (write_node == other_write_node) {
        continue;
      }
      const bool is_connect =
          current_graph_matrix->IsConnected(write_node->shared_from_this(), other_write_node->shared_from_this()) ||
          current_graph_matrix->IsConnected(other_write_node->shared_from_this(), write_node->shared_from_this());
      if (!is_connect) {
        GELOGW("There is no control relation between write node[%s] and other write node[%s].",
               write_node->GetNamePtr(), other_write_node->GetNamePtr());
        REPORT_INNER_ERR_MSG("W18888",
                           "There is no control relation between write node[%s] and other write node[%s]. Please check "
                           "graph make it valid, sometimes it may cause problem (precision problem).",
                           write_node->GetNamePtr(), other_write_node->GetNamePtr());
        return GRAPH_FAILED;
      }
    }
    for (auto &other_read_node : readonly_nodes) {
      const bool is_connect =
          current_graph_matrix->IsConnected(write_node->shared_from_this(), other_read_node->shared_from_this()) ||
          current_graph_matrix->IsConnected(other_read_node->shared_from_this(), write_node->shared_from_this());
      if (!is_connect) {
        GELOGW("There is no control relation between write node[%s] and read node[%s].", write_node->GetNamePtr(),
               other_read_node->GetNamePtr());
        REPORT_INNER_ERR_MSG("W18888",
                           "There is no control relation between write node[%s] and read node[%s]. Please check "
                           "graph make it valid, sometimes it may cause problem (precision problem).",
                           write_node->GetNamePtr(), other_read_node->GetNamePtr());
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge