/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "redundant_control_edge_remove_pass.h"
#include <algorithm>
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
constexpr auto kOpTypeConst = "Const";
constexpr auto kOpTypeConstant = "Constant";

bool IsConstNode(const NodePtr &node) {
  return node->GetType() == kOpTypeConst || node->GetType() == kOpTypeConstant;
}

bool HasIncomingControlEdges(const NodePtr &node) {
  return node->GetInControlNodesSize() > 0U;
}

// 检查单个输出节点是否以 ctrl_src 为数据输入
bool IsCtrlSrcDataInputTo(const NodePtr &ctrl_src, const NodePtr &output_node) {
  const auto &input_nodes = output_node->GetInDataNodes();
  return std::find(input_nodes.begin(), input_nodes.end(), ctrl_src) != input_nodes.end();
}

// 条件：const_node 的所有数据输出节点的数据输入都全部包含 const_node 的控制边输入
// 只有满足这个条件，才能删除 const_node 的所有控制边
bool CanRemoveAllControlEdges(const NodePtr &const_node) {
  const auto &ctrl_srcs = const_node->GetInControlNodes();
  if (ctrl_srcs.empty()) {
    return false;
  }
  // 检查每个数据输出节点是否都以所有控制边输入源为数据输入
  for (const auto &data_out_node: const_node->GetOutDataNodes()) {
    for (const auto &ctrl_src: ctrl_srcs) {
      if (!IsCtrlSrcDataInputTo(ctrl_src, data_out_node)) {
        GELOGD("Cannot remove control edges: data output node %s does not have ctrl_src %s as data input",
               data_out_node->GetNamePtr(), ctrl_src->GetNamePtr());
        return false;
      }
    }
  }

  return true;
}

// 移除单个 Const 节点的所有冗余控制边
graphStatus RemoveRedundantControlEdges(const NodePtr &const_node, bool &changed) {
  if (!CanRemoveAllControlEdges(const_node)) {
    return GRAPH_SUCCESS; // 不满足条件不算错误
  }

  const auto &out_nodes = const_node->GetOutDataNodes();
  std::vector<NodePtr> ctrl_srcs_to_remove;
  for (const auto &ctrl_src_node: const_node->GetInControlNodes()) {
    ctrl_srcs_to_remove.push_back(ctrl_src_node);
  }
  // 删除所有控制边
  for (const auto &ctrl_src_node: ctrl_srcs_to_remove) {
    auto ret = GraphUtils::RemoveEdge(ctrl_src_node->GetOutControlAnchor(),
                                      const_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Failed to remove control edge from %s to %s",
             ctrl_src_node->GetNamePtr(), const_node->GetNamePtr());
      return ret;
    }
    GELOGI("Removed redundant control edge: %s -ctrl-> %s (all %zu outputs contain all ctrl inputs)",
           ctrl_src_node->GetNamePtr(), const_node->GetNamePtr(), out_nodes.size());
  }
  changed = true;
  return GRAPH_SUCCESS;
}
} // namespace

graphStatus RedundantControlEdgeRemovePass::Run(const ComputeGraphPtr &graph, bool &changed) const {
  GE_ASSERT_NOTNULL(graph);

  for (const auto &node: graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (!IsConstNode(node) || !HasIncomingControlEdges(node) ||
        node->GetOutDataNodesSize() == 0UL) {
      continue;
    }
    GE_ASSERT_SUCCESS(RemoveRedundantControlEdges(node, changed), "Failed to remove control edges for node: %s",
                      node->GetNamePtr());
  }

  return GRAPH_SUCCESS;
}
} // namespace ge
