/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inner_tensor_move_add_pass.h"

#include "node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_util.h"
#include "graph/operator_factory.h"
#include "graph/passes/pass_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
const std::string kInnerTensorMoveAttr = "_inner_tensor_move";
// 这些算子需要被后续的ref算子修改输出
bool IsChangedByRefNode(const NodePtr &node) {
  const auto &type = NodeUtils::GetNodeType(node);
  return (type == REFSWITCH) || (type == REFMERGE) || (type == READVARIABLEOP);
}

bool IsNoNeedInsertTensorMove(const NodePtr &node) {
  NodePtr real_node = node;
  if (OpTypeUtils::IsDataNode(real_node->GetType())) {
    const auto in_node = NodeUtils::GetParentInput(real_node);
    if (in_node != nullptr) {
      real_node = in_node;
    }
  }
  return OpTypeUtils::IsVarLikeNode(real_node->GetType()) || IsChangedByRefNode(real_node);
}
}
Status InnerTensorMoveAddPass::Run(ComputeGraphPtr graph) {
  GE_ASSERT_NOTNULL(graph);
  if (connectivity_ == nullptr) {
    connectivity_ = ComGraphMakeUnique<ConnectionMatrix>(graph);
    if (connectivity_ == nullptr) {
      GELOGW("Make shared failed");
      return FAILED;
    }

    GE_ASSERT_SUCCESS(connectivity_->Generate(graph), "Cannot generate connection matrix for graph %s.",
                      graph->GetName().c_str());
  }
  GE_ASSERT_SUCCESS(PassUtils::UpdateRefAttr(graph));
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    bool is_ref = false;
    (void)AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
    if (!is_ref) {
      continue;
    }
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      int32_t reuse_in_index = -1;
      if (!GraphUtils::IsRefFromInput(out_data_anchor, reuse_in_index)) {
        continue;
      }
      auto in_data_anchor = node->GetInDataAnchor(reuse_in_index);
      GE_ASSERT_NOTNULL(in_data_anchor);
      auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_data_anchor);

      auto ref_input_node = peer_out_data_anchor->GetOwnerNode();
      if ((ref_input_node->GetType() == TENSORMOVE) && (peer_out_data_anchor->GetPeerInDataNodesSize() == 1U)) {
        GELOGD("ref node %s input is TensorMove %s, this Tensormove has only one output, skip insert inner Tensormove",
               node->GetNamePtr(), ref_input_node->GetNamePtr());
        continue;
      }

      if (IsNoNeedInsertTensorMove(ref_input_node)) {
        GELOGD("ref node %s input is %s, skip insert inner Tensormove", node->GetNamePtr(), ref_input_node->GetNamePtr());
        continue;
      }
      std::vector<InDataAnchorPtr> target_in_data_anchors; // 记录ref_input_node多引用时需要重新连边的输入inanchor
      for (const auto &peer_in_data_anchor : peer_out_data_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_NOTNULL(peer_in_data_anchor);
        if ((peer_in_data_anchor == in_data_anchor) || (node == peer_in_data_anchor->GetOwnerNode())) {
          continue;
        }
        if (connectivity_->IsConnected(node, peer_in_data_anchor->GetOwnerNode())) {
          target_in_data_anchors.emplace_back(peer_in_data_anchor);
        }
      }
      // 创建TensorMove并断边、重新连边
      auto tensor_move = AddTensorMove(node->GetOwnerComputeGraph(), peer_out_data_anchor, in_data_anchor);
      GE_ASSERT_NOTNULL(tensor_move);
      // ref_input_node多引用断边重新从TensorMove连边过去
      for (const auto &peer_in_data_anchor : target_in_data_anchors) {
        GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(peer_out_data_anchor, peer_in_data_anchor),
                            "RemoveEdge from %s to %s failed", ref_input_node->GetNamePtr(),
                            peer_in_data_anchor->GetOwnerNode()->GetNamePtr());
        GE_ASSERT_SUCCESS(GraphUtils::AddEdge(tensor_move->GetOutDataAnchor(0), peer_in_data_anchor));
        GELOGD("Add edge from %s to %s", tensor_move->GetNamePtr(),
               peer_in_data_anchor->GetOwnerNode()->GetNamePtr());
      }
    }
  }
  return SUCCESS;
}

NodePtr InnerTensorMoveAddPass::AddTensorMove(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_anchor,
                                         const InDataAnchorPtr &dst_anchor) {
  static size_t tensor_move_count = 0;
  std::string name = "inner_tensormove_" + std::to_string(tensor_move_count++);
  const auto tensor_move_op = OperatorFactory::CreateOperator(name.c_str(), TENSORMOVE);
  tensor_move_op.BreakConnect();
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(tensor_move_op);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(AttrUtils::SetBool(op_desc, kInnerTensorMoveAttr, true));
  auto tensor_move_node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(tensor_move_node);
  GE_ASSERT_SUCCESS(GraphUtils::InsertNodeAfter(src_anchor, {dst_anchor}, tensor_move_node, 0, 0),
                    "Insert node %s between %s and %s failed", tensor_move_node->GetNamePtr(),
                    src_anchor->GetOwnerNode()->GetNamePtr(), dst_anchor->GetOwnerNode()->GetNamePtr());
  connectivity_->Update(graph, {tensor_move_node});
  GELOGI("Add tensor_move_node %s between src_node %s and dst_node %s", tensor_move_node->GetName().c_str(),
         src_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetOwnerNode()->GetName().c_str());
  return tensor_move_node;
}

REG_PASS_OPTION("InnerTensorMoveAddPass").LEVELS(OoLevel::kO0);
}  // namespace ge
