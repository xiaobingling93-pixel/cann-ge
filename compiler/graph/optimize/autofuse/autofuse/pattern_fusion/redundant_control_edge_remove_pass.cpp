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

// 条件：const_node 的所有数据输出节点都必须以 ctrl_src 为数据输入
// 这样可以确保删除 ctrl_src -ctrl-> const_node 后，执行顺序仍然被保持
bool CanRemoveControlEdge(const NodePtr &ctrl_src, const NodePtr &const_node) {
  const auto &out_nodes = const_node->GetOutAllNodes();
  if (out_nodes.empty()) {
    return false;
  }

  // 检查所有输出节点是否都以 ctrl_src 为数据输入
  for (const auto &output_node: out_nodes) {
    if (!IsCtrlSrcDataInputTo(ctrl_src, output_node)) {
      GELOGD("Cannot remove control edge: output node %s is not a data input of ctrl_src %s",
             output_node->GetNamePtr(), ctrl_src->GetNamePtr());
      return false;
    }
  }

  return true;
}

// 移除单个 Const 节点的冗余控制边
graphStatus RemoveRedundantControlEdges(const NodePtr &const_node) {
  const auto &out_nodes = const_node->GetOutDataNodes();
  // 收集需要删除的控制边
  std::vector<NodePtr> ctrl_srcs_to_remove;
  for (const auto &ctrl_src_node: const_node->GetInControlNodes()) {
    if (CanRemoveControlEdge(ctrl_src_node, const_node)) {
      ctrl_srcs_to_remove.push_back(ctrl_src_node);
    }
  }

  // 执行删除
  for (const auto &ctrl_src_node: ctrl_srcs_to_remove) {
    auto ret = GraphUtils::RemoveEdge(ctrl_src_node->GetOutControlAnchor(),
                                      const_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Failed to remove control edge from %s to %s",
             ctrl_src_node->GetNamePtr(), const_node->GetNamePtr());
      return ret;
    }
    GELOGI("Removed redundant control edge: %s -ctrl-> %s (controls all %zu outputs)",
           ctrl_src_node->GetNamePtr(), const_node->GetNamePtr(), out_nodes.size());
  }

  return GRAPH_SUCCESS;
}
} // namespace

graphStatus RedundantControlEdgeRemovePass::Run(const ComputeGraphPtr &graph, bool &changed) const {
  GE_ASSERT_NOTNULL(graph);

  uint32_t processed_count = 0U;
  for (const auto &node: graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (!IsConstNode(node) || !HasIncomingControlEdges(node) ||
        node->GetOutDataNodesSize() == 0UL) {
      continue;
    }

    auto ret = RemoveRedundantControlEdges(node);
    if (ret != GRAPH_SUCCESS) {
      GELOGD("Failed to remove control edges for node: %s", node->GetNamePtr());
      continue;
    }
    processed_count++;
  }

  if (processed_count > 0U) {
    GELOGI("Removed redundant control edges for %u const nodes", processed_count);
    changed = true;
  }

  return GRAPH_SUCCESS;
}
} // namespace ge
