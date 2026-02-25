/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "inner_tensor_move_delete_pass.h"
#include <checker.h>
#include "graph/ge_context.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_util.h"
#include "graph/passes/pass_utils.h"
#include "common/ge_common/ge_types.h"

namespace ge {
namespace {
const std::string kInnerTensorMoveAttr = "_inner_tensor_move";
bool HasTensorMemoryScope(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (const auto &tensor_desc : op_desc->GetAllOutputsDescPtr()) {
    if (AttrUtils::HasAttr(tensor_desc, ATTR_NAME_TENSOR_MEMORY_SCOPE)) {
      return true;
    }
  }
  return false;
}
// nano形态下，是根据编译出来的kernel决定基地址来自于ioa/workspace/weight，所以连着输入的算子不能轻易删除，后续会有正式方案解决.
// 临时方案:根据ATTR_NAME_TENSOR_MEMORY_SCOPE属性判断nano形态，并且判断是否连着输入data，如果连着就不删除
bool IsCannotDelete(const NodePtr &tensor_move) {
  bool has_tensor_memory_scope = HasTensorMemoryScope(tensor_move);
  for (const auto &node : tensor_move->GetOutDataNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (HasTensorMemoryScope(node)) {
      has_tensor_memory_scope = true;
      break;
    }
  }
  if (!has_tensor_memory_scope) {
    return false;
  }
  for (const auto &node: tensor_move->GetInDataNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      return true;
    }
  }
  return false;
}

bool IsInnerTensorMove(const NodePtr &node) {
  if (node->GetType() != TENSORMOVE) {
    return false;
  }
  bool is_inner_tensor_move = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), kInnerTensorMoveAttr, is_inner_tensor_move);
  return is_inner_tensor_move;
}

bool IsRefNode(const NodePtr &node) {
  bool is_ref = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);
  return is_ref;
}

NodePtr FindRefNode(const NodePtr &node) {
  NodePtr ref_node = nullptr;
  for (const auto &out_node : node->GetOutDataNodes()) {
    bool is_ref = IsRefNode(out_node);
    if (is_ref) {
      ref_node = out_node;
      break;
    }
  }
  return ref_node;
}

NodePtr FindRealOutNode(const NodePtr &node) {
  // 如果是内置TensorMove则需要继续向后找到其对应的ref node，这个才是实际的输出
  if (IsInnerTensorMove(node)) {
    return FindRefNode(node);
  }
  return node;
}

bool IsStableRDFS() {
  std::string topo_sorting_mode_str;
  if ((GetContext().GetOption(OPTION_TOPOSORTING_MODE, topo_sorting_mode_str) == GRAPH_SUCCESS) &&
      (!topo_sorting_mode_str.empty())) {
    const int32_t base = 10;
    auto topo_sorting_mode = static_cast<TopoSortingMode>(std::strtol(topo_sorting_mode_str.c_str(), nullptr, base));
    return topo_sorting_mode == TopoSortingMode::kStableRDFS;
  }
  return false;
}
}

Status InnerTensorMoveDeletePass::IsolateAndDeleteTensorMoveNode(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  if (!IsInnerTensorMove(node)) {
    GELOGE(FAILED, "node %s is not inner tensor move, cannot delete", node->GetNamePtr());
    return FAILED;
  }
  GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(node, {0}));
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  GE_ASSERT_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(graph, node));
  return SUCCESS;
}

Status InnerTensorMoveDeletePass::DeleteInnerTensorMove(const NodePtr &node) {
  GE_ASSERT_TRUE(node->GetInDataNodesSize() == 1U);
  auto tensor_move_peer_out_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(tensor_move_peer_out_anchor);
  auto input_node = tensor_move_peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(input_node);
  // 场景一：TensorMove输入节点是const之类不可改写的类型，则不可以删除
  if (OpTypeUtils::IsConstNode(input_node->GetType())) {
    GELOGI("TensorMove %s 's input %s is const, cannot remove it", node->GetNamePtr(), input_node->GetNamePtr());
    return SUCCESS;
  }

  // 如果topo排序方式是稳定拓扑序，则可以删除
  if (IsStableRDFS()) {
    GELOGI(
        "TensorMove %s 's input has multi output, and it self has one output, topo sort mode is StableRDFS, can be deleted",
        node->GetNamePtr());
    return IsolateAndDeleteTensorMoveNode(node);
  }

  // 如果inner tensormove后面连的不是ref算子，则可以删除
  bool is_all_out_not_ref = true;
  for (const auto &out_node : node->GetOutDataNodes()) {
    is_all_out_not_ref = (is_all_out_not_ref && !IsRefNode(out_node));
  }
  if (is_all_out_not_ref) {
    GELOGI("TensorMove %s 's output is not ref, can be deleted", node->GetNamePtr());
    return IsolateAndDeleteTensorMoveNode(node);
  }

  if (tensor_move_peer_out_anchor->GetPeerInDataNodesSize() == 1U) {
    // 场景二：TensorMove输入节点A是单引用，且TensorMove输出是单引用，则可以删除
    if (node->GetOutDataNodesSize() == 1U) {
      GELOGI("TensorMove %s 's input has only one output, and it self has only one output, can be deleted",
             node->GetNamePtr());
      return IsolateAndDeleteTensorMoveNode(node);
    }
    // 场景三：TensorMove输入节点A是单引用，TensorMove是多引用，TensorMove除Ref之外的其他所有输出都受Ref控制则可以删除
    auto ref_node = FindRefNode(node);
    GE_ASSERT_NOTNULL(ref_node, "TensorMove %s has no ref output node", node->GetNamePtr());
    bool is_other_all_out_depend_ref = true;
    for (const auto &out_node : node->GetOutDataNodes()) {
      if (out_node == ref_node) {
        continue;
      }
      auto real_out_node = FindRealOutNode(out_node);
      is_other_all_out_depend_ref = (is_other_all_out_depend_ref && connectivity_->IsConnected(
                                         ref_node, real_out_node));
    }
    if (is_other_all_out_depend_ref) {
      GELOGI(
          "TensorMove %s 's input has only one output, and it self has multi output, all out node except ref depend ref, can be deleted",
          node->GetNamePtr());
      return IsolateAndDeleteTensorMoveNode(node);
    }
  } else {
    if (node->GetOutDataNodesSize() == 1U) {
      // 场景四：TensorMove输入节点A是多引用，且TensorMove输出是单引用，如果A节点除TensorMove之外的其他所有输出都指向了ref，则可以删除
      auto ref_node = FindRefNode(node);
      GE_ASSERT_NOTNULL(ref_node);
      bool is_ref_depend_other_all_node = true;
      for (const auto &out_node : input_node->GetOutDataNodes()) {
        if (out_node == node) {
          continue;
        }
        auto real_out_node = FindRealOutNode(out_node);
        is_ref_depend_other_all_node = (is_ref_depend_other_all_node &&
                                        connectivity_->IsConnected(real_out_node, ref_node));
      }
      if (is_ref_depend_other_all_node) {
        GELOGI(
            "TensorMove %s 's input has multi output, and it self has one output, ref depend other all A's out node, can be deleted",
            node->GetNamePtr());
        return IsolateAndDeleteTensorMoveNode(node);
      }
    } else {
      // 场景五：TensorMove输入节点A是多引用，且TensorMove输出是多引用，且TensorMove的其他输出节点都依赖ref，且A节点除TensorMove之外的其他输出都指向ref，则可以删除
      auto ref_node = FindRefNode(node);
      GE_ASSERT_NOTNULL(ref_node);
      bool can_delete = true;
      // TensorMove的其他输出节点都依赖ref
      for (const auto &out_node : node->GetOutDataNodes()) {
        if (out_node == ref_node) {
          continue;
        }
        auto real_out_node = FindRealOutNode(out_node);
        can_delete = (can_delete && connectivity_->IsConnected(ref_node, real_out_node));
      }

      // A节点除TensorMove之外的其他输出都指向ref
      for (const auto &out_node : input_node->GetOutDataNodes()) {
        if (out_node == node) {
          continue;
        }
        auto real_out_node = FindRealOutNode(out_node);
        can_delete = (can_delete && connectivity_->IsConnected(real_out_node, ref_node));
      }
      if (can_delete) {
        GELOGI(
            "TensorMove %s 's input has multi output, and it self has multi output, ref depend other all A's out node, other all TensorMove's out node depend ref, can be deleted",
            node->GetNamePtr());
        return IsolateAndDeleteTensorMoveNode(node);
      }
    }
  }
  GELOGI("TensorMove %s cannot be deleted", node->GetNamePtr());
  return SUCCESS;
}

Status InnerTensorMoveDeletePass::Run(ComputeGraphPtr graph) {
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
    if (IsInnerTensorMove(node)) {
      if (IsCannotDelete(node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(DeleteInnerTensorMove(node), "Inner TensorMove %s delete failed", node->GetNamePtr());
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("InnerTensorMoveDeletePass").LEVELS(OoLevel::kO0);
} // namespace ge