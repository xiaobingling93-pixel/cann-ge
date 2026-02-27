/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice_forward_fusion_pass.h"
#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "pattern_fusion_utils.h"

namespace ge {
namespace {
using namespace pattern_fusion;
const std::unordered_set<std::string> kSliceOpTypes = {"Slice", "SliceD", "StridedSlice", "StridedSliceD"};

// 检查节点的所有输入是否来自同一个节点，且所有输出去往同一个节点
bool HasSingleInputAndOutput(const NodePtr &node) {
  const auto &in_nodes = node->GetInDataNodes();
  const auto &out_nodes = node->GetOutDataNodes();

  if (in_nodes.empty() || out_nodes.empty()) {
    return false;
  }

  // 检查所有输入是否来自同一个节点
  NodePtr unique_input = in_nodes.at(0);
  for (const auto &in: in_nodes) {
    if (in != unique_input) {
      return false;
    }
  }

  // 检查所有输出是否去往同一个节点
  NodePtr unique_output = out_nodes.at(0);
  for (const auto &out: out_nodes) {
    if (out != unique_output) {
      return false;
    }
  }

  return true;
}

std::vector<NodePtr> CollectElementwiseChain(const NodePtr &slice_node) {
  std::vector<NodePtr> chain;
  const auto &slice_inputs = slice_node->GetInDataNodes();
  if (slice_inputs.empty()) {
    return chain;
  }
  NodePtr current = slice_inputs.at(0);
  while (current != nullptr && IsElementwise(current) && HasSingleInputAndOutput(current)) {
    // dtype变化时停止收集，避免slice不支持dtype或引入精度问题
    const auto input_dtype = current->GetOpDesc()->GetInputDesc(0).GetDataType();
    const auto output_dtype = current->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (input_dtype != output_dtype) {
      break;
    }
    chain.push_back(current);

    const auto &inputs = current->GetInDataNodes();
    if (inputs.empty()) {
      break;
    }
    current = inputs.at(0);
  }
  return chain;
}

graphStatus RemoveEdgesFromSource(const NodePtr &dst, const NodePtr &src) {
  for (const auto &in_anchor: dst->GetAllInDataAnchors()) {
    auto peer_out = in_anchor->GetPeerOutAnchor();
    if (peer_out != nullptr && peer_out->GetOwnerNode() == src) {
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(peer_out, in_anchor), "Failed to remove edge %s -> %s",
                              src->GetNamePtr(), dst->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ConnectAllInputsToSource(const NodePtr &dst, const OutDataAnchorPtr &src_out, const NodePtr &src) {
  for (const auto &in_anchor: dst->GetAllInDataAnchors()) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(src_out, in_anchor),
                            "Failed to connect %s -> %s", src->GetNamePtr(), dst->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}

graphStatus DoSliceForward(const NodePtr &slice_node, const std::vector<NodePtr> &elem_chain) {
  const NodePtr &elem_near_slice = elem_chain.front(); // elem_chain[0]
  const NodePtr &elem_far_from_slice = elem_chain.back(); // elem_chain[N-1]
  const auto &far_inputs = elem_far_from_slice->GetInDataNodes();
  GE_ASSERT_TRUE(!far_inputs.empty(), "Element %s has no input nodes", elem_far_from_slice->GetNamePtr());
  const NodePtr &chain_input = far_inputs.at(0);
  GE_CHECK_NOTNULL(chain_input);

  const auto &near_inputs = elem_near_slice->GetInDataNodes();
  GE_ASSERT_TRUE(!near_inputs.empty(), "Element %s has no input nodes", elem_near_slice->GetNamePtr());
  const NodePtr &near_input = near_inputs.at(0);
  GE_CHECK_NOTNULL(near_input);

  const OutDataAnchorPtr &chain_input_out = elem_far_from_slice->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(chain_input_out);

  // 断开：chain_input -> elem_far_from_slice 的所有边（处理多输入情况）
  GE_ASSERT_GRAPH_SUCCESS(RemoveEdgesFromSource(elem_far_from_slice, chain_input));
  // 断开：elem_near_slice -> slice
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(elem_near_slice->GetOutDataAnchor(0), slice_node->GetInDataAnchor(0)),
                          "Failed to disconnect %s -> %s", elem_near_slice->GetNamePtr(),
                          slice_node->GetNamePtr());

  // 重连：chain_input -> slice -> elem_far_from_slice -> ... -> elem_near_slice -> [原slice的输出]
  // 1. slice 的原输出改为 elem_near_slice 的输出
  for (const auto &peer_in: slice_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    GE_CHK_GRAPH_STATUS_RET(
      GraphUtils::ReplaceEdgeSrc(slice_node->GetOutDataAnchor(0), peer_in,
        elem_near_slice->GetOutDataAnchor(0)),
      "Failed to replace output edge of slice %s", slice_node->GetNamePtr());
  }

  // 2. chain_input -> slice
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(chain_input_out, slice_node->GetInDataAnchor(0)),
                          "Failed to connect %s -> %s", chain_input->GetNamePtr(),
                          slice_node->GetNamePtr());

  // 3. slice -> elem_far_from_slice
  GE_CHK_GRAPH_STATUS_RET(ConnectAllInputsToSource(elem_far_from_slice, slice_node->GetOutDataAnchor(0), slice_node),
                          "Failed to connect %s -> %s", slice_node->GetNamePtr(),
                          elem_far_from_slice->GetNamePtr());

  return GRAPH_SUCCESS;
}

graphStatus UpdateElementwiseShapes(const std::vector<NodePtr> &elem_chain,
                                    const GeShape &slice_output_shape,
                                    const gert::SymbolShape &slice_symbol_shape) {
  for (const auto &elem: elem_chain) {
    GE_CHK_STATUS_RET(SetNodeShape(elem, slice_output_shape, slice_output_shape, slice_symbol_shape),
                      "Failed to update shape for node %s", elem->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}
} // namespace

graphStatus SliceForwardFusionPass::Run(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node: graph->GetDirectNode()) {
    if (kSliceOpTypes.find(node->GetType()) == kSliceOpTypes.end()) {
      continue;
    }

    auto elem_chain = CollectElementwiseChain(node);
    if (elem_chain.empty()) {
      continue;
    }

    GELOGD("SliceForwardFusionPass: hoist %s past elementwise chain (size=%zu), first_elem=%s, last_elem=%s",
           node->GetNamePtr(), elem_chain.size(),
           elem_chain.front()->GetNamePtr(), elem_chain.back()->GetNamePtr());

    const auto &slice_output_shape = node->GetOpDesc()->GetOutputDesc(0).GetShape();
    const auto &slice_symbol_shape = GetNodeSymbolShape(node);

    GE_CHK_GRAPH_STATUS_RET(DoSliceForward(node, elem_chain),
                            "Failed to do slice forward for node %s", node->GetNamePtr());
    GE_CHK_GRAPH_STATUS_RET(UpdateElementwiseShapes(elem_chain, slice_output_shape, slice_symbol_shape),
                            "Failed to update elementwise shapes for slice %s", node->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}
} // namespace ge
