/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <string>
#include <set>
#include <queue>
#include "ge_common/ge_api_types.h"
#include "asc_graph_pass.h"
#include "ascir_ops.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "utils/autofuse_utils.h"
#include "graph/utils/graph_utils.h"
#include "post_process/post_process_util.h"

namespace ge {
namespace {
struct NodePtrCompare {
  bool operator()(const NodePtr &a, const NodePtr &b) const {
    const auto op_desc_a = a->GetOpDesc();
    const auto op_desc_b = b->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc_a);
    GE_ASSERT_NOTNULL(op_desc_b);
    return op_desc_a->GetId() < op_desc_b->GetId();  // 升序排序
  }
};
struct CandidateNodes {
  std::set<NodePtr, NodePtrCompare> sorted_nodes;
  std::unordered_set<NodePtr> candidate_nodes_set;
  void InsertCandidate(const NodePtr &node) {
    if (candidate_nodes_set.insert(node).second) {
      GELOGD("Node %s is candidate of CSE.", node->GetNamePtr());
      sorted_nodes.insert(node);
    }
  }
};

Status GetCseKeyOffset(const NodePtr &node, std::stringstream &ss) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(load_attr);
  Expression  load_offset;
  if (ge::GRAPH_SUCCESS == (load_attr->GetOffset(load_offset))) {
    ss << "-offset-" << load_offset.Serialize().get();
  } else {
    GELOGW("CseKey offset from node(%s %s) is null", node->GetName().c_str(), node->GetType().c_str());
  }
  return SUCCESS;
}

Status GetCseKeyGatherAxis(const NodePtr &node, std::stringstream &ss) {
  int64_t axis = std::numeric_limits<int64_t>::max();
  GE_ASSERT_SUCCESS(asc_adapt::GetGatherAxis(node, axis));
  ss << "-gather-axis-" << axis;
  return SUCCESS;
}


Status GetCseKeyScalarValue(const NodePtr &node, std::stringstream &ss) {
  const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  const auto scalar_attr = dynamic_cast<ascir_op::Scalar::AscScalarIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(scalar_attr);
  std::string value;
  if (scalar_attr->GetValue(value) != SUCCESS) {
    value = "";
  }
  ss << "-scalar-value-" << value;
  return SUCCESS;
}

Status GetCseKeySplitIndex(const NodePtr &node, std::stringstream &ss) {
  const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  const auto split_attr = dynamic_cast<ascir_op::Split::AscSplitIrAttrDef *>(attr->ir_attr.get());
  GE_ASSERT_NOTNULL(split_attr);
  int64_t value;
  if (split_attr->GetIndex(value) != SUCCESS) {
    value = 0;
  }
  ss << "-split-index-" << to_string(value);
  return SUCCESS;
}

Status GetCseKey(const NodePtr &node, std::string &key) {
  std::stringstream ss;
  ss << node->GetType();
  const auto asc_node_op_desc = node->GetOpDesc();
  if (node->GetType() != kOutputType) {
    auto asc_node_attr = asc_node_op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(asc_node_attr);
    ss << "-" << asc_node_attr->sched.loop_axis << "-";
  }

  ss << "-data-inputs-";
  for (auto &in_anchor : node->GetAllInDataAnchors()) {
    auto src_anchor = in_anchor->GetPeerOutAnchor();
    if (src_anchor == nullptr) {
      ss << in_anchor->GetIdx() << "-null-";
    } else {
      ss << in_anchor->GetIdx() << "-" << src_anchor->GetOwnerNode()->GetName() << "-" << src_anchor->GetIdx() << "-";
    }
  }

  ss << "-tensor-outputs-";
  for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
    GE_ASSERT_NOTNULL(output_desc);
    auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(output_desc_tensor_attr);
    ss << "-" << AutofuseUtils::VectorToStr(output_desc_tensor_attr->axis) << "-";
    ss << "-" << AutofuseUtils::VectorToStr(output_desc_tensor_attr->repeats) << "-";
    ss << "-" << AutofuseUtils::VectorToStr(output_desc_tensor_attr->strides) << "-";
    ss << "-" << TypeUtils::DataTypeToSerialString(output_desc_tensor_attr->dtype) << "-";
  }

  if (node->GetType() == kLoadType) {
    GE_ASSERT_SUCCESS(GetCseKeyOffset(node, ss));
  } else if (node->GetType() == kGatherType) {
    GE_ASSERT_SUCCESS(GetCseKeyGatherAxis(node, ss));
  } else if (node->GetType() == kScalarType) {
    GE_ASSERT_SUCCESS(GetCseKeyScalarValue(node, ss));
  } else if (node->GetType() == kSplitType) {
    GE_ASSERT_SUCCESS(GetCseKeySplitIndex(node, ss));
  }

  GELOGD("Generating partial CSE key(%s) for node(%s)", ss.str().c_str(), node->GetName().c_str());
  ss << "-control-inputs-";
  std::set<std::string> control_in_node_names;
  for (auto &src_node : node->GetInControlNodes()) {
    control_in_node_names.insert(src_node->GetName());
  }
  for (auto &name : control_in_node_names) {
    ss << name << "-";
  }

  ss << "attrs-" << AttrUtils::GetAllAttrsStr(node->GetOpDesc());
  key = ss.str();
  return SUCCESS;
}

Status CollectPeerCandidateNodesWithType(
    const OutDataAnchor *const out_data_anchor, const std::unordered_map<NodePtr, size_t> &nodes_2_topo_idx,
    std::map<std::string, std::map<size_t, NodePtr>> &node_types_to_ordered_nodes) {
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    const auto &peer_in_node = peer_in_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    const auto &node_type = peer_in_node->GetType();
    // GE构图不会存在写相同一块内存的多个store节点，即store节点都不是相同节点，所以不考虑cse消除store节点
    // 后端不支持store节点输出多引用，所以不考虑cse消除output节点
    if (node_type == kOutputType || node_type == kStoreType) {
      continue;
    }
    const auto iter = nodes_2_topo_idx.find(peer_in_node);
    GE_ASSERT_TRUE(iter != nodes_2_topo_idx.cend());
    node_types_to_ordered_nodes[node_type].emplace(iter->second, peer_in_node);
  }
  return SUCCESS;
}

Status CollectCandidate(const NodePtr &node, const std::unordered_map<NodePtr, size_t> &nodes_2_topo_idx,
                        CandidateNodes &candidate_nodes) {
  // output has same type sibling
  std::map<std::string, std::map<size_t, NodePtr>> node_type_2_nodes;
  for (const auto out_data_anchor : node->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(out_data_anchor);
    if (out_data_anchor->GetPeerInDataNodesSize() < 2U) {
      continue;
    }
    GE_ASSERT_SUCCESS(CollectPeerCandidateNodesWithType(out_data_anchor, nodes_2_topo_idx, node_type_2_nodes));

    for (const auto &type_2_nodes : node_type_2_nodes) {
      if (type_2_nodes.second.size() < 2U) {
        continue;
      }
      // 因为node_type_2_nodes有序(只是当前可消除的节点是有序，后续新产生的节点加入到candidate_nodes需要重新排序后才是有序的)
      GELOGD("Node %s output %d has %zu sibling of type %s.", node->GetNamePtr(), out_data_anchor->GetIdx(),
             type_2_nodes.second.size(), type_2_nodes.first.c_str());
      for (const auto &candidate : type_2_nodes.second) {
        candidate_nodes.InsertCandidate(candidate.second);
      }
    }
    node_type_2_nodes.clear();
  }
  return GRAPH_SUCCESS;
}

Status NodeElimination(const AscGraph &graph, const NodePtr &node_to_replace, const NodePtr &node_to_elimination) {
  std::vector<int32_t> output_map(node_to_elimination->GetAllOutDataAnchorsSize());
  for (size_t i = 0U; i < node_to_elimination->GetAllOutDataAnchorsSize(); ++i) {
    output_map[i] = i;
  }

  GE_ASSERT_TRUE(GraphUtils::ReplaceNodeAnchors(node_to_replace, node_to_elimination, {}, output_map) == GRAPH_SUCCESS,
                 "Replace node:%s(%s)'s anchor by node:%s(%s) failed", node_to_elimination->GetName().c_str(),
                 node_to_elimination->GetType().c_str(), node_to_replace->GetName().c_str(),
                 node_to_replace->GetType().c_str());

  NodeUtils::UnlinkAll(*node_to_elimination);
  GE_ASSERT_TRUE(
      GraphUtils::RemoveNodeWithoutRelink(AscGraphUtils::GetComputeGraph(graph), node_to_elimination) == GRAPH_SUCCESS,
      "Remove node:%s(%s) without relink in graph:%s failed", node_to_elimination->GetName().c_str(),
      node_to_elimination->GetType().c_str(), graph.GetName().c_str());
  GELOGI("Remove node %s by the CSE process, replace it with node %s", node_to_elimination->GetName().c_str(),
         node_to_replace->GetName().c_str());
  return GRAPH_SUCCESS;
}

Status CandidateNodesElimination(const AscGraph &graph, std::unordered_map<NodePtr, size_t> nodes_2_topo_idx,
                                 CandidateNodes &candidate_nodes,
                                 std::unordered_set<NodePtr> &removed_output_nodes) {
  std::unordered_map<std::string, NodePtr> keys_to_node;
  bool has_removed_node = false;
  while (!candidate_nodes.sorted_nodes.empty()) {
    const auto node = *(candidate_nodes.sorted_nodes.begin());
    candidate_nodes.sorted_nodes.erase(node);
    std::string key;
    GELOGD("Generating CSE key for node %s", node->GetName().c_str());
    GE_ASSERT_SUCCESS(GetCseKey(node, key));
    GELOGD("Generated CSE key for node %s", node->GetName().c_str());
    GE_ASSERT_TRUE(!key.empty(), "node %s", node->GetName().c_str());
    auto iter = keys_to_node.find(key);
    if (iter == keys_to_node.cend()) {
      keys_to_node[key] = node;
      continue;
    }

    if (node->GetAllOutDataAnchorsSize() != iter->second->GetAllOutDataAnchorsSize()) {
      GELOGW("The node %s and %s have the same CSE key, but different output anchor count, skip to fusion them",
             iter->second->GetName().c_str(), node->GetName().c_str());
      continue;
    }

    GE_ASSERT_SUCCESS(NodeElimination(graph, iter->second, node));

    has_removed_node = true;
    if (node->GetType() == kOutputType) {
      removed_output_nodes.insert(node);
    }
    GE_ASSERT_SUCCESS(CollectCandidate(iter->second, nodes_2_topo_idx, candidate_nodes));
  }
  if (has_removed_node) {
    GELOGI("CSE Pass changed graph %s", graph.GetName().c_str());
  }
  return SUCCESS;
}

Status AddToCandidateListIfScalar(const NodePtr &node, CandidateNodes &candidate_nodes) {
  if (node->GetType() != kScalarType) {
    return SUCCESS;
  }
  candidate_nodes.InsertCandidate(node);
  return SUCCESS;
}

Status SubgraphCommonSubexpressionElimination(AscGraph &graph, [[maybe_unused]] const NodePtr &asc_node) {
  // here mark nodes to its topo idx, to make sure CSE optimize follow origin node seq
  std::unordered_map<NodePtr, size_t> nodes_2_topo_idx;
  CandidateNodes candidate_nodes;
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    nodes_2_topo_idx.emplace(node, op_desc->GetId());
    //  做 CSE 优化前先把相同value的scalar收集到候选节点，便于消除scalar后继节点拥有共同输入做cse
    GE_ASSERT_SUCCESS(AddToCandidateListIfScalar(node, candidate_nodes));
  }

  // collect node may match CSE
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    GE_ASSERT_SUCCESS(CollectCandidate(node, nodes_2_topo_idx, candidate_nodes));
  }

  std::unordered_set<NodePtr> removed_output_nodes;
  GE_ASSERT_SUCCESS(CandidateNodesElimination(graph, nodes_2_topo_idx, candidate_nodes, removed_output_nodes));

  GE_ASSERT_SUCCESS(asc_adapt::UpdateSubgraphOutputAttr(removed_output_nodes, asc_node));
  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(graph));
  return SUCCESS;
}

Status CommonSubexpressionElimination(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(asc_adapt::ProcessAscBackendNodes(ge_or_fused_asc_backend_graph,
                                                      SubgraphCommonSubexpressionElimination, "cse_pass"));
  return SUCCESS;
}
}  // namespace

Status CsePass::Run(const ComputeGraphPtr &graph) const {
  // CSE PASS
  GE_ASSERT_SUCCESS(CommonSubexpressionElimination(graph));
  GELOGI("Graph %s completed CSE optimization successfully.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
