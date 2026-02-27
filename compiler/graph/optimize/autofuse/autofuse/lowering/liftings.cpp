/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "liftings.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "asc_lowerer/asc_overrides.h"
#include "asc_lowerer/loop_common.h"
#include "lowerings.h"

#include "ascir_ops_utils.h"
#include "backend/backend_spec.h"
#include "op_helper/lower_split_helper.h"

namespace ge {
constexpr size_t kMinComputeNodes = 2U;
constexpr size_t kMinOneNodeInData = 64U;
constexpr size_t kMatmulMinInputNum = 2U;
constexpr size_t kNumOne = 1U;
const char *const kMatmulSubgraph = "matmul_subgraph";
const char *const kMMV3Type = "MatMulV3";
const char *const kBMMV3Type = "BatchMatMulV3";

graphStatus CreateMMSubgraphAttr(const NodePtr &node, vector<const Node *> &compute_ops,
                                 size_t &cube_real_inputs) {
  const auto &sub_graph = ComGraphMakeShared<ComputeGraph>(kMatmulSubgraph + node->GetName());
  GE_ASSERT_NOTNULL(sub_graph);
  for (auto *org_node : compute_ops) {
    if ((org_node->GetType() == "MatMulV3") || (org_node->GetType() == "BatchMatMulV3")) {
      const auto &op_desc = GraphUtils::CopyOpDesc(org_node->GetOpDesc(), nullptr);
      GE_ASSERT_NOTNULL(op_desc);
      op_desc->SetName(org_node->GetName());
      auto mm_node = sub_graph->AddNode(op_desc);
      GE_ASSERT_NOTNULL(mm_node);
      bool is_a_b_same_input = cube_real_inputs > node->GetAllInDataAnchors().size();
      for (auto i = 0U; i < cube_real_inputs; i++) {
        // a矩阵、b矩阵同输入存在ascgraph的matmul有两个输入，Ascackend只有一个输入，需多加一个输入再生成kernel函数
        auto i_node = is_a_b_same_input ? (i == 0U ? 0U : i - 1U) : i;
        const auto &src_anchor = node->GetInDataAnchor(i_node);  // cube垂直向后融合可以保证cube输入在前
        GE_ASSERT_NOTNULL(src_anchor);
        auto peer_anchor = src_anchor->GetPeerOutAnchor();
        GE_ASSERT_NOTNULL(peer_anchor);
        auto peer_node = peer_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(peer_node);
        const auto &op_desc = GraphUtils::CopyOpDesc(peer_node->GetOpDesc(), nullptr);
        GE_ASSERT_NOTNULL(op_desc);
        op_desc->SetName(peer_node->GetName());
        auto mm_peer_node = sub_graph->AddNode(op_desc);
        GE_ASSERT_NOTNULL(mm_peer_node);
        const auto &peer_node_out_anchor = mm_peer_node->GetOutDataAnchor(peer_anchor->GetIdx());
        GE_ASSERT_NOTNULL(peer_node_out_anchor);
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(peer_node_out_anchor, mm_node->GetInDataAnchor(i)));
      }
      break;
    }
  }

  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kMatmulSubgraph, sub_graph));
  return GRAPH_SUCCESS;
}

bool IsCubeSkipLifting(const NodePtr &node, const size_t min_compute_nodes,
                       const AutoFuseAttrs *fuse_attrs, bool is_fuse_from_lowering) {
  auto origin_nodes = fuse_attrs->GetOriginNodes();
  vector<const Node *> compute_ops = AutofuseUtils::GetComputeOps(origin_nodes); // GetComputeOps里面融合reshape等节点不会统计成compute节点
  if ((compute_ops.size() < min_compute_nodes) && is_fuse_from_lowering) { // 需要is_fuse_from_lowering标记判断是否经过canfuse融合
    return false;
  }

  size_t cube_real_inputs = kMatmulMinInputNum;
  const auto asc_graph = fuse_attrs->GetAscGraph();
  GE_ASSERT_NOTNULL(asc_graph);
  for (const auto &asc_node : asc_graph->GetAllNodes()) {
    if (!AutofuseUtils::IsCubeNodeType(asc_node)) {
      continue;
    }
    cube_real_inputs = asc_node->GetInNodes().size(); // ascgraph里面的matmul节点，即使输入是某个节点输出的多引用，也有至少两个输入
  }
  GE_ASSERT_SUCCESS(CreateMMSubgraphAttr(node, compute_ops, cube_real_inputs));

  GELOGI("Skip lifting node: %s, cube_real_inputs %zu", node->GetName().c_str(), cube_real_inputs);
  return true;
}

bool IsSingleTransposeShouldSkipLifting(const NodePtr &node) {
  const auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  const auto asc_graph = fuse_attrs->GetAscGraph();
  GE_ASSERT_NOTNULL(asc_graph);
  for (const auto &asc_node : asc_graph->GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Transpose>(asc_node)) {
      const auto input_size = asc_node->inputs[0].attr.axis.size();
      GE_ASSERT_TRUE(input_size > 0, "input_size %d out of range", input_size);
      const auto &input_tail_axis = asc_node->inputs[0].attr.axis[input_size - 1];
      const auto &output_tail_axis = asc_node->outputs[0].attr.axis[input_size - 1];
      const auto repeat = asc_node->inputs[0].attr.repeats[input_size - 1];
      int64_t dim = -1;
      GE_ASSERT_TRUE(repeat.GetHint(dim), "Failed to get int value, expr = %s", ge::SymbolicUtils::ToString(repeat).c_str());
      const auto data_type_size = GetSizeByDataType(asc_node->inputs[0].attr.dtype);
      GE_ASSERT_TRUE(data_type_size > 0, "data_type_size must greater than 0", ge::SymbolicUtils::ToString(repeat).c_str());
      constexpr int64_t limited_tail_size = 512U;
      const auto limited_size = limited_tail_size / data_type_size;
      // 目前仅非尾轴转置且大尾轴场景跳过Lifting
      if ((input_tail_axis == output_tail_axis) && (dim >= limited_size)) {
        return true;
      }
    }
  }
  return false;
}

bool IsSpecificConditionSkipLifting(const NodePtr &node) {
  if (IsSingleTransposeShouldSkipLifting(node)) {
    return true;
  }
  // 可新增其他特殊场景
  return false;
}

bool IsSkipLifting(const NodePtr &node, size_t min_compute_nodes) {
  auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  bool disable_lifting = false;
  if (AttrUtils::GetBool(node->GetOpDesc(), "_disable_lifting", disable_lifting) && disable_lifting) {
    GELOGI("Skip lifting node: %s, as it has disable lifting flag", node->GetName().c_str());
    return true;
  }
  // step1: cube type
  if (fuse_attrs->HasFuseType(loop::FuseType::kCube)) {
    return IsCubeSkipLifting(node, min_compute_nodes, fuse_attrs, GetInterAttrs(fuse_attrs).is_fuse_from_lowering);
  }
  // step2: fused from can_fuse
  auto &inner_fuse_attr = GetInterAttrs(fuse_attrs);
  if (inner_fuse_attr.split_global_id == kNonSplitGlobalId && !inner_fuse_attr.is_fuse_from_lowering) {
    GELOGI("Skip lifting node: %s, as it is fused from can_fuse, and is not split fuse type.", node->GetName().c_str());
    return true;
  }
  // step3: split type
  if (fuse_attrs->HasFuseType(loop::FuseType::kSplit)) {
    bool need_lifting = false;
    LowerSplitHelper split_helper(node);
    split_helper.NeedLifting(need_lifting);
    return !need_lifting;
  }
  auto origin_nodes = fuse_attrs->GetOriginNodes();
  if ((origin_nodes.size() > kNumOne) && (fuse_attrs->HasFuseType(loop::FuseType::kSliceSplit))) {
    GELOGI("Skip lifting node: %s, as slice fuse other node, origin node size is %zu",
           node->GetName().c_str(), origin_nodes.size());
    return true;
  }
  // step4: compute node num
  vector<const Node*> compute_nodes = AutofuseUtils::GetComputeOps(origin_nodes);
  if (compute_nodes.size() >= min_compute_nodes) {
    GELOGD("Skip lifting node：%s, as num fused nodes num %zu >= %zu",
           node->GetName().c_str(), compute_nodes.size(), min_compute_nodes);
    return true;
  }

  if (fuse_attrs->GetOriginOutputBuffers().size() > kNumOne) {
    GELOGD("Skip lifting node: %s, as num origin output anchors %zu > 1", node->GetName().c_str(),
           fuse_attrs->GetOriginOutputBuffers().size());
    return true;
  }

  if (!fuse_attrs->GetOptimizedInputBuffers().empty()) {
    auto optimized_input_buffers = fuse_attrs->GetOptimizedInputBuffers();
    for (auto optimized_input_buffer : optimized_input_buffers) {
      GE_ASSERT_NOTNULL(optimized_input_buffer);
      GE_ASSERT_NOTNULL(optimized_input_buffer->GetOwnerNode());
      auto optimized_input_node = optimized_input_buffer->GetOwnerNode();
      if (!OpTypeUtils::IsConstNode(optimized_input_node->GetType())) {
        GELOGD("Skip lifting node: %s, as it optimize buffer loads %s", node->GetName().c_str(),
               loop::BufferName(*fuse_attrs->GetOptimizedInputBuffers().begin()).c_str());
        return true;
      }
    }
  }

  auto min_one_node_in_data = kMinOneNodeInData;
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  if (backend_spec != nullptr) {
    min_one_node_in_data = backend_spec->concat_max_input_num + 1;
  }
  if ((origin_nodes.size() == kNumOne) && (origin_nodes.at(0) != nullptr) &&
      (origin_nodes.at(0)->GetAllInDataAnchorsSize() >= min_one_node_in_data)) {
    GELOGI("Skip lifting node: %s, as it Only one node But Origin Input Size %u", node->GetName().c_str(),
           fuse_attrs->GetOriginNodes().at(0)->GetAllInDataAnchorsSize());
    return true;
  }

  // AscIR只包含Transpose类型节点时跳过lifting
  if ((origin_nodes.size() == kNumOne) && (IsSpecificConditionSkipLifting(node))) {
    GELOGI("Skip lifting node: %s, as the origin node is "
           "Non-tail axis Transpose with Tail axis greater than or equal to 512B",
           node->GetName().c_str());
    return true;
  }
  return false;
}

graphStatus LiftingAscBackendOp(const NodePtr &node) {
  const auto fuse_attr = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_WARN_ASSERT(fuse_attr != nullptr, "Node %s has no AutoFuseAttrs", node->GetName().c_str());
  std::vector<const ge::Node *> origin_nodes = fuse_attr->GetOriginNodes();

  const std::map<size_t, std::set<const ge::InDataAnchor *>> &concrete_edges = fuse_attr->GetConcreteEdges();
  for (auto &edges : concrete_edges) {
    auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(edges.first));
    GE_ASSERT_NOTNULL(in_anchor, "Node %s has no input anchor %zu", node->GetName().c_str(), edges.first);
    auto src = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src, "Node %s input %zu has no peer out anchor", node->GetName().c_str(), edges.first);
    for (auto &dst : edges.second) {
      GE_ASSERT_NOTNULL(dst);
      if (!dst->IsLinkedWith(src)) {
        GELOGI("Lifting recover edge %s->%s", loop::BufferName(src).c_str(), loop::BufferName(dst).c_str());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src, const_cast<ge::InDataAnchor *>(dst)->shared_from_this()));
      }
    }
  }
  auto origin_index = 0U;
  for (auto asc_output: node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(asc_output);
    GE_ASSERT_TRUE(fuse_attr->GetOriginOutputBuffers().size() > origin_index);
    const auto origin_output = fuse_attr->GetOriginOutputBuffers()[origin_index++];
    for (auto &peer : asc_output->GetPeerAnchors()) {
      GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(asc_output).c_str(), loop::BufferName(peer).c_str(),
             loop::BufferName(origin_output).c_str());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(asc_output, peer->shared_from_this()));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(origin_output->shared_from_this(), peer));
    }

    auto origin_control = origin_output->GetOwnerNode()->GetOutControlAnchor();
    auto asc_control = asc_output->GetOwnerNode()->GetOutControlAnchor();
    for (auto &peer : asc_control->GetPeerAnchors()) {
      GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(asc_control).c_str(), loop::BufferName(peer).c_str(),
             loop::BufferName(origin_control).c_str());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(asc_control, peer));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(origin_control, peer));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LiftingAscBackendOps(const std::vector<NodePtr> &nodes) {
  for (auto &node : nodes) {
    GE_ASSERT_GRAPH_SUCCESS(LiftingAscBackendOp(node));
  }
  return GRAPH_SUCCESS;
}

graphStatus LiftingMultiOutNode(const NodePtr &node, const NodePtr &origin_node, AutoFuseAttrs *fuse_attrs,
                                std::map<NodePtr, std::vector<NodePtr>> &node_maybe_lifting_outputs) {
  auto &maybe_lifting = node_maybe_lifting_outputs[origin_node];
  maybe_lifting.push_back(node);
  GELOGD("Maybe lifting %s of node %s", node->GetNamePtr(), origin_node->GetNamePtr());
  size_t num_of_out_anchors = 0U;
  for (const auto &lifting_node : maybe_lifting) {
    num_of_out_anchors += lifting_node->GetAllOutDataAnchorsSize();
  }
  if (num_of_out_anchors == origin_node->GetAllOutDataAnchorsSize()) {
    GELOGI("Lift AscBackend nodes %s, node list is %s, as: Num fused nodes %zu < %zu.",
           loop::StrJoin(maybe_lifting, [](const NodePtr &n) { return n->GetName(); }).c_str(),
           loop::StrJoin(fuse_attrs->GetOriginNodes(), [](const Node *n) {
             return n->GetType() + "(" + n->GetName() + ")";
           }).c_str(), fuse_attrs->GetOriginNodes().size(), kMinComputeNodes);
    GE_ASSERT_GRAPH_SUCCESS(LiftingAscBackendOps(maybe_lifting));
    maybe_lifting.clear();
  }
  return GRAPH_SUCCESS;
}

graphStatus LiftingManager::LiftingGraph(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  std::map<NodePtr, std::vector<NodePtr>> node_maybe_lifting_outputs;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() != kAscBackend && node->GetType() != kAscBackendNoKernelOp) {
      continue;
    }
    auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    if (fuse_attrs == nullptr) {
      GELOGD("Skip lifting node: %s, as it has no auto fuse attrs", node->GetName().c_str());
      continue;
    }
    auto &origin_output = fuse_attrs->GetOriginOutputBuffers()[0];
    GE_ASSERT_NOTNULL(origin_output);
    const auto &origin_node = origin_output->GetOwnerNode();
    GE_ASSERT_NOTNULL(origin_node);

    if (IsSkipLifting(node, kMinComputeNodes)) {
      GELOGD("Skip lifting node: %s(%s), as it need skip lifting process.",
             node->GetName().c_str(), node->GetType().c_str());
      continue;
    }

    if (origin_node->GetAllOutDataAnchorsSize() > kNumOne) {
      GE_ASSERT_GRAPH_SUCCESS(LiftingMultiOutNode(node, origin_node, fuse_attrs, node_maybe_lifting_outputs));
      continue;
    }

    GELOGI("Lift AscBackend node %s, node list is %s, as: Num fused nodes %zu < %zu.", node->GetName().c_str(),
           loop::StrJoin(fuse_attrs->GetOriginNodes(), [](const Node *n) {
             return n->GetType() + "(" + n->GetName() + ")";
           }).c_str(), fuse_attrs->GetOriginNodes().size(), kMinComputeNodes);
    GE_ASSERT_GRAPH_SUCCESS(LiftingAscBackendOp(node));
  }
  return GRAPH_SUCCESS;
}
}
