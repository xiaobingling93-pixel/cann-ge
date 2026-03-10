/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_ir_lowerer.h"
#include "lowerings.h"
#include "liftings.h"
#include "backend/backend_spec.h"
#include "utils/auto_fuse_config.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "utils/autofuse_attrs.h"
#include "op_helper/lower_concat_helper.h"
#include "op_helper/lower_split_helper.h"
#include "common/scope_tracing_recorder.h"

namespace ge {
using namespace autofuse;
namespace {
bool IsNodeShouldPrune(const NodePtr &node) {
  if (!node->GetOutNodes().empty()) {
    return false;
  }
  if (OpTypeUtils::IsDataNode(node->GetType())) {
    GELOGI("Skip prune unused data %s", node->GetName().c_str());
    return false;
  }
  return true;
}

bool IsAscBackendOpNode(const NodePtr &node) {
  return (node->GetType() == kAscBackend || node->GetType() == kAscBackendNoKernelOp);
}

void GetOriginNamesAndTypes(OpDescPtr &op_desc, std::vector<std::string> &original_names,
                            std::vector<std::string> &original_types) {
  std::vector<std::string> origin_op_names;
  std::vector<std::string> origin_op_types;
  bool is_has_attr = ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, origin_op_names);
  ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_op_types);
  if (!is_has_attr) {
    original_names.push_back(op_desc->GetName());
    original_types.push_back(op_desc->GetType());
  } else {
    if (!origin_op_names.empty()) {
      for (auto &node_name : origin_op_names) {
        if (!node_name.empty()) {
          original_names.push_back(node_name);
        }
      }
    }
    if (!origin_op_types.empty()) {
      for (auto &node_type : origin_op_types) {
        if (!node_type.empty()) {
          original_types.push_back(node_type);
        }
      }
    }
  }
}

graphStatus SetDataDumpAttrForAscBackend(NodePtr &node) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  std::vector<const ge::Node *> original_nodes;
  std::set<const ge::Node *> original_nodes_set;
  for (const ge::Node *ge_node : fuse_attrs->GetOriginNodes()) {
    GE_ASSERT_NOTNULL(ge_node);
    original_nodes_set.insert(ge_node);
  }
  for (auto ge_node : original_nodes_set) {
    original_nodes.emplace_back(ge_node);
  }
  std::vector<std::string> original_names;
  std::vector<std::string> original_types;
  for (auto &ge_node : original_nodes) {
    ge::OpDescPtr op_desc = ge_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    (void)GetOriginNamesAndTypes(op_desc, original_names, original_types);
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, original_types);
  return GRAPH_SUCCESS;
}

graphStatus RecordControlEdgeRelation(std::map<std::string, std::vector<NodePtr>> &node_to_const,
                                      const std::string &node_name, const NodePtr &control_node) {
  auto it = node_to_const.find(node_name);
  if (it == node_to_const.end()) {
    node_to_const.insert(std::pair<std::string, std::vector<NodePtr>>(node_name, {control_node}));
  } else {
    it->second.emplace_back(control_node);
  }
  return GRAPH_SUCCESS;
}

graphStatus ClearControlEdgeRelation(std::map<std::string, std::vector<NodePtr>> &node_to_const,
                                     const std::string &node_name) {
  auto it = node_to_const.find(node_name);
  if (it != node_to_const.end()) {
    node_to_const.erase(node_name);
  }
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus AscIrLowerer::Lowering(const ComputeGraphPtr &graph) {
  TRACING_PERF_SCOPE(TracingModule::kModelCompile, "Lowering", graph->GetName());
  auto nodes = graph->GetAllNodes();
  if (std::any_of(nodes.begin(), nodes.end(), [](const NodePtr &node) {
        return IsAscBackendOpNode(node);
      })) {
    GELOGI("Skip lowering for graph %s as it has been lowered", graph->GetName().c_str());
    do_lowered_ = false;
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_GRAPH_SUCCESS(ProcessControlEdge(graph));
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_CHECK_NOTNULL(backend_spec);
  LoweringConfig config;
  config.max_loop_ops = AutoFuseConfig::LoweringConfig().max_fused_loop_ops;
  config.max_loop_loads = backend_spec->max_load_num;
  GELOGD("Max load num in one Ascbackend is %u.", config.max_loop_loads);
  config.max_buffer_readers = AutoFuseConfig::LoweringConfig().max_buffer_readers;
  GE_ASSERT_GRAPH_SUCCESS(LoweringManager::LoweringGraph(graph, config));
  GE_ASSERT_GRAPH_SUCCESS(LoweringManager::FusedLoopToAscBackendOp(graph, kAscBackendFuseConfig, counter_));
  auto graphs = graph->GetAllSubgraphs();
  if (std::find(graphs.begin(), graphs.end(), graph) == graphs.end()) {
    graphs.insert(graphs.begin(), graph);
  }
  for (const auto &subgraph : graphs) {
    GE_ASSERT_GRAPH_SUCCESS(RemoveDirectNodeUnusedEdges(subgraph));
  }
  do_lowered_ = true;
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::Lifting(const ComputeGraphPtr &graph) {
  TRACING_PERF_SCOPE(TracingModule::kModelCompile, "Lifting", graph->GetName());
  if (!do_lowered_) {
    GELOGI("Skip lifting for graph %s as it is not lowered this time", graph->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  if (ge::AutoFuseConfig::LoweringConfig().experimental_disable_lifting) {
    GELOGI("Skip lifting for graph %s as params disable_lifting is true", graph->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_GRAPH_SUCCESS(LowerConcatHelper::LiftingPoorPerfFusedAscBackendOps(graph));
  GE_ASSERT_GRAPH_SUCCESS(LiftingManager::LiftingGraph(graph));
  GE_ASSERT_GRAPH_SUCCESS(DfxForAscBackendOp(graph));
  GE_ASSERT_SUCCESS(PruneUnusedNodesAfterLifting(graph), "Failed to prune unused nodes in graph %s",
                    graph->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(RecoverInitControlEdge(graph));
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::RemoveDirectNodeUnusedEdges(const ComputeGraphPtr &graph) {
  std::deque<NodePtr> used_deque;
  for (auto &node : graph->GetDirectNode()) {
    bool force_skip_prune = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_SKIP_PRUNE_OPTIMIZE, force_skip_prune);
    if (node->GetType() == ge::NETOUTPUT || force_skip_prune) {
      GELOGD("Add node %s to used start nodes", node->GetName().c_str());
      used_deque.push_back(node);
    }
  }
  std::set<const ge::Node *> used_nodes;
  while (!used_deque.empty()) {
    NodePtr node = used_deque.front();
    used_deque.pop_front();
    if (!used_nodes.insert(node.get()).second) {
      continue;
    }
    for (const auto &in_node : node->GetInAllNodes()) {
      GELOGD("Add node %s to used nodes", in_node->GetName().c_str());
      used_deque.push_back(in_node);
    }
  }
  for (const auto &node : graph->GetDirectNode()) {
    if (!IsAscBackendOpNode(node)) {
      continue;
    }
    const auto fuse_attr = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    GE_ASSERT_NOTNULL(fuse_attr);
    const auto &buffers = fuse_attr->GetOriginOutputBuffers();
    std::for_each(buffers.begin(), buffers.end(),
                  [this](const OutDataAnchor *buffer) { replaced_nodes_.insert(buffer->GetOwnerNode()); });
    std::vector<std::pair<NodePtr, InDataAnchorPtr>> unused_node_and_anchors;
    for (auto &node_and_anchor : node->GetOutDataNodesAndAnchors()) {
      auto &dst_node = node_and_anchor.first;
      if (IsAscBackendOpNode(dst_node) || used_nodes.count(dst_node.get()) > 0U) {
        continue;
      }
      unused_node_and_anchors.emplace_back(node_and_anchor);
    }
    for (auto &node_and_anchor : unused_node_and_anchors) {
      auto anchor = node_and_anchor.second;
      GELOGI("Remove unused edge %s->%s before backend fuse", loop::BufferName(anchor->GetPeerOutAnchor()).c_str(),
             loop::BufferName(anchor).c_str());
      GE_WARN_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(anchor->GetPeerOutAnchor(), anchor));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::PruneUnusedNodesAfterLifting(const ge::ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);
  std::set<NodePtr> seen_nodes;
  std::deque<NodePtr> nodes_to_remove;
  for (const NodePtr &node : replaced_nodes_) {
    if (node->GetOutNodes().empty()) {
      nodes_to_remove.push_back(node);
    }
  }
  for (const NodePtr &node : graph->GetAllNodes()) {
    if (IsAscBackendOpNode(node) && node->GetOutNodes().empty()) {
      nodes_to_remove.push_back(node);
    }
  }
  std::set<NodePtr> removed;
  while (!nodes_to_remove.empty()) {
    NodePtr node = nodes_to_remove.front();
    nodes_to_remove.pop_front();
    if (!removed.insert(node).second) {
      continue;
    }
    std::vector<NodePtr> input_nodes;
    for (auto &in_node : node->GetInAllNodes()) {
      input_nodes.emplace_back(in_node);
    }
    GELOGD("Remove unused node %s after lifting", node->GetName().c_str());
    (void)NodeUtils::RemoveSubgraphsOnNode(node);
    (void)graph->RemoveNode(node);
    for (auto &in_node : input_nodes) {
      if (IsNodeShouldPrune(in_node)) {
        nodes_to_remove.push_back(in_node);
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::DfxForAscBackendOp(const ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);
  const std::string aiv_cnt_key = "_op_vectorcore_num";
  for (auto &node : graph->GetAllNodes()) {
    if (!IsAscBackendOpNode(node) && (node->GetType() != "FusedAscBackend")) {
      continue;
    }
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    if (fuse_attrs == nullptr) {
      continue;
    }
    // reconstruct the original computer graph for each ascbackend node
    GE_ASSERT_GRAPH_SUCCESS(LoweringManager::GetFusedOriginComputeGraph(*fuse_attrs, node));
    // set data dump attr for ascbackend node
    GE_ASSERT_SUCCESS(SetDataDumpAttrForAscBackend(node));

    int32_t vector_core_num = GetInterAttrs(fuse_attrs).vector_core_num;
    GE_CHECK_GE(vector_core_num, 0);
    if (vector_core_num > 0){
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), aiv_cnt_key, std::to_string(vector_core_num));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus CutCtrEdgeIfNeed(const NodePtr &control_node, const NodePtr &node,
                             const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                             std::map<std::string, std::vector<NodePtr>> &node_to_control_const_node) {
  if (OpTypeUtils::IsConstNode(control_node->GetType())) {
    bool is_from_constant_folding = false;
    (void)ge::AttrUtils::GetBool(control_node->GetOpDesc(), "_is_from_constant_folding", is_from_constant_folding);
    if (is_from_constant_folding) {
      GELOGD("node:%s has control edge from/to const/constant nodes:%s, which is generated by constant folding.",
             node->GetName().c_str(), control_node->GetName().c_str());
      GE_CHK_GRAPH_STATUS_RET(ge::GraphUtils::RemoveEdge(src, dst), "[Remove][ControlEdge] between %s and %s failed",
                              control_node->GetName().c_str(), node->GetName().c_str());
      GE_ASSERT_GRAPH_SUCCESS(
          RecordControlEdgeRelation(node_to_control_const_node, node->GetName(), control_node));
    } else {
      GELOGD("node:%s has control edge from/to const/constant nodes:%s,"
             "but is not generated by constant folding, skip process.",
             node->GetName().c_str(), control_node->GetName().c_str());
    }
  } else {
    GELOGD("node:%s has control edge from/to non const/constant nodes:%s, skip process.",
           node->GetName().c_str(), control_node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::ProcessControlEdge(const ComputeGraphPtr &graph) {
  /*
   * if node has input/output control edge, which is from/to const/constant node.
   * and const/constant node is from constant folding, consider removing control edge for more fuse chance.
   */
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetOutControlNodes().empty() && node->GetInControlNodes().empty()) {
      continue;
    }
    auto type = node->GetType();
    if (OpTypeUtils::IsAutofuseNode(type) || OpTypeUtils::IsGraphInputNode(type) ||
        OpTypeUtils::IsGraphOutputNode(type)) {
      continue;
    }
    if (node->GetOutDataNodes().size() == 0U) {
      continue;
    }
    (void) LoweringManager::Lowering(node);
    const auto node_out_anchor = node->GetOutDataAnchor(0);
    const auto node_kernel_box = loop::GetKernelBox(node_out_anchor);
    const auto &fuse_type = FuseTypeToString(node_kernel_box.Type());
    if (fuse_type == "extern") {
      continue;
    }

    for (auto &in_control_node : node->GetInControlNodes()) {
      GE_CHECK_NOTNULL(in_control_node);
      GE_ASSERT_GRAPH_SUCCESS(CutCtrEdgeIfNeed(in_control_node, node, in_control_node->GetOutControlAnchor(),
                                               node->GetInControlAnchor(), node_in_control_to_const_));
    }
    for (auto &out_control_node : node->GetOutControlNodes()) {
      GE_CHECK_NOTNULL(out_control_node);
      GE_ASSERT_GRAPH_SUCCESS(CutCtrEdgeIfNeed(out_control_node, node, node->GetOutControlAnchor(),
                                               out_control_node->GetInControlAnchor(), node_out_control_to_const_));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::RecoverInitControlEdge(const ComputeGraphPtr &graph) {
  for (auto &node : graph->GetAllNodes()) {
    auto type = node->GetType();
    if (OpTypeUtils::IsAutofuseNode(type) || OpTypeUtils::IsGraphInputNode(type) ||
        OpTypeUtils::IsGraphOutputNode(type)) {
      continue;
    }
    auto it = node_in_control_to_const_.find(node->GetName());
    if (it != node_in_control_to_const_.end()) {
      for (const auto &const_node : it->second) {
        GE_CHK_GRAPH_STATUS_RET(ge::GraphUtils::AddEdge(const_node->GetOutControlAnchor(),
                                                        node->GetInControlAnchor()),
                                "[Add][ControlEdge] between %s and %s failed",
                                const_node->GetName().c_str(), node->GetName().c_str());
        GELOGI("Success to recover control edge from node %s to node %s", const_node->GetName().c_str(),
               node->GetName().c_str());
      }
    }
    it = node_out_control_to_const_.find(node->GetName());
    if (it != node_out_control_to_const_.end()) {
      for (const auto &const_node : it->second) {
        GE_CHK_GRAPH_STATUS_RET(ge::GraphUtils::AddEdge(node->GetOutControlAnchor(),
                                                        const_node->GetInControlAnchor()),
                                "[Add][ControlEdge] between %s and %s failed",
                                node->GetName().c_str(), const_node->GetName().c_str());
        GELOGI("Success to recover control edge from node %s to node %s", node->GetName().c_str(),
               const_node->GetName().c_str());
      }
    }
    ClearControlEdgeRelation(node_in_control_to_const_, node->GetName());
    ClearControlEdgeRelation(node_out_control_to_const_, node->GetName());
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge