/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/merge_unknown_shape_n_pass.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"

namespace {
const std::string kDefaultNodeSuffix = "_merged";
const size_t kDefaultSize = 2U;
}  // namespace

namespace ge {
Status MergeUnknownShapeNPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != SHAPEN) {
      continue;
    }
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string attr_opdesc_name;
    if (AttrUtils::GetStr(op_desc, ATTR_NAME_SPLIT_SHAPEN_ORIGIN_NAME, attr_opdesc_name)) {
      node_merge_map_[attr_opdesc_name].emplace_back(node);
    }
  }
  auto ret = MergeShapeN(graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Merge][ShapeN] graph:%s failed",
           graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status MergeUnknownShapeNPass::MergeShapeN(const ComputeGraphPtr &graph) {
  for (const auto &it : node_merge_map_) {
    if (it.second.size() < kDefaultSize) {
      continue;
    }
    size_t input_index = 0U;
    size_t output_index = 0U;
    OpDescPtr op_desc_ptr = MakeShared<OpDesc>(it.first + kDefaultNodeSuffix, SHAPEN);
    GE_CHECK_NOTNULL(op_desc_ptr);
    for (size_t i = 0U; i < it.second.size(); ++i) {
      auto node = it.second.at(i);
      OpDescPtr opdesc_old_node = node->GetOpDesc();
      GE_CHECK_NOTNULL(opdesc_old_node);
      for (size_t j = 0U; j < opdesc_old_node->GetAllInputsSize(); ++j) {
        op_desc_ptr->AddInputDesc(opdesc_old_node->GetInputDesc(j));
        op_desc_ptr->AddOutputDesc(opdesc_old_node->GetOutputDesc(j));
      }
    }
    const auto new_node_vec = graph->FuseNodeKeepTopo(it.second, {op_desc_ptr});
    GE_ASSERT_TRUE(!new_node_vec.empty());
    NodePtr new_node = new_node_vec.front();
    GE_CHECK_NOTNULL(new_node);
    for (const auto &node : it.second) {
      auto ret = ReplaceMergeNodeAnchors(node, new_node, input_index, output_index);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Replace node:%s(%s) by node:%s(%s) failed",
                          node->GetName().c_str(), node->GetType().c_str(),
                          new_node->GetName().c_str(), new_node->GetType().c_str());
        GELOGE(FAILED, "[Replace][Node] %s(%s) by node:%s(%s) failed",
               node->GetName().c_str(), node->GetType().c_str(),
               new_node->GetName().c_str(), new_node->GetType().c_str());
        return FAILED;
      }
      if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call RemoveNodeWithoutRelink failed, node:%s", node->GetName().c_str());
        GELOGE(FAILED, "[Call][RemoveNodeWithoutRelink] failed for node %s.", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status MergeUnknownShapeNPass::ReplaceMergeNodeAnchors(const NodePtr &node, const NodePtr &new_node,
                                                       size_t &input_index, size_t &output_index) {
  if (new_node == nullptr || node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "[Check][Param] Parameter is nullptr");
    GELOGE(FAILED, "[Check][Param] Parameter is nullptr");
    return FAILED;
  }
  GELOGI("Replace node:%s(%s) by node:%s(%s)", node->GetName().c_str(), node->GetType().c_str(),
         new_node->GetName().c_str(), new_node->GetType().c_str());
  for (size_t i = 0U; i < node->GetAllOutDataAnchors().size(); ++i) {
    const auto out_anchor = node->GetAllOutDataAnchors().at(i);
    for (const InDataAnchorPtr &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GraphUtils::ReplaceEdgeSrc(out_anchor, peer_in_anchor, new_node->GetOutDataAnchor(output_index));
    }
    ++output_index;
  }
  for (size_t j = 0U; j < node->GetAllInDataAnchors().size(); ++j) {
    const auto in_anchor = node->GetAllInDataAnchors().at(j);
    const OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GraphUtils::ReplaceEdgeDst(peer_out_anchor, in_anchor, new_node->GetInDataAnchor(input_index));
    ++input_index;
  }
  auto ret = ReplaceControlAnchors(new_node, node);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Replace][ControlAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", node->GetName().c_str(), node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status MergeUnknownShapeNPass::ReplaceControlAnchors(const NodePtr &new_node, const NodePtr &old_node) {
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(new_node->GetInControlAnchor());
  GE_CHECK_NOTNULL(old_node);
  GE_CHECK_NOTNULL(old_node->GetInControlAnchor());
  const auto peer_out_anchors = old_node->GetInControlAnchor()->GetPeerOutControlAnchors();
  const auto new_in_control_anchor = new_node->GetInControlAnchor();
  const auto exists_out_anchors = new_in_control_anchor->GetPeerOutControlAnchors();
  const auto exists_out_anchors_set = std::set<AnchorPtr>(exists_out_anchors.begin(), exists_out_anchors.end());
  for (const auto &peer_out_anchor : peer_out_anchors) {
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (exists_out_anchors_set.count(peer_out_anchor) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::ReplaceEdgeDst(peer_out_anchor,
                                                old_node->GetInControlAnchor(), new_in_control_anchor);
    GE_ASSERT_SUCCESS(ret, "Add edge from %s to %s failed, ret:%u", peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                      new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
  }
  const auto old_out_control_anchor = old_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(old_out_control_anchor);
  const auto peer_in_anchors = old_out_control_anchor->GetPeerInControlAnchors();
  const auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  auto exists_in_anchors = new_out_control_anchor->GetPeerInControlAnchors();
  const auto exists_in_anchors_set = std::set<AnchorPtr>(exists_in_anchors.begin(), exists_in_anchors.end());
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (exists_in_anchors_set.count(peer_in_anchor) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::ReplaceEdgeSrc(old_out_control_anchor, peer_in_anchor, new_out_control_anchor);
    GE_ASSERT_SUCCESS(ret, "AddEdge from %s to %s failed, ret:%u",
                      new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                      peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
  }
  return SUCCESS;
}

REG_PASS_OPTION("MergeUnknownShapeNPass").LEVELS(OoLevel::kO3);
}  // namespace ge
