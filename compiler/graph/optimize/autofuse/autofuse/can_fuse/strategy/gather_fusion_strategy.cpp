/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_fusion_strategy.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/auto_fuse_config.h"
#include "utils/not_fuse_reason_code.h"
#include "backend/backend_spec.h"

namespace ge {

bool GatherFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_CHECK_NOTNULL(backend_spec);
  // 1.gather不能前融合，只处理垂直融合; gather不能和gather融合，不区分垂直融合还是水平融合
  if (attr2->HasFuseType(loop::FuseType::kGather)) {
    if (!IsForwardFuseGather(node1, node2) &&
      CheckGatherWithView(node1, node2, backend_spec->gather_spec.enable_gather_broadcast_fusion)) {
      return true;
    }
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Gather can not fuse forward]",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kGatherCanNotFuseForward));
    return false;
  }

  // 2.gather和reduce融合,需要根据平台信息判断
  if (attr1->HasFuseType(loop::FuseType::kGather) && attr2->HasFuseType(loop::FuseType::kReduction)) {
    return CheckGatherReduceFuse(node1, node2, backend_spec->gather_spec.enable_reduce_gather_fusion,
                                 backend_spec->gather_spec.enable_gather_broadcast_fusion);
  }

  // 3.gather和concat融合,需要根据平台信息判断
  if (attr1->HasFuseType(loop::FuseType::kGather) && attr2->HasFuseType(loop::FuseType::kConcat)) {
    return CheckGatherConcatFuse(node1, node2, backend_spec->gather_spec.enable_gather_concat_fusion,
                                 backend_spec->gather_spec.enable_gather_broadcast_fusion);
  }

  // 4.gather只能和elementwise垂直融合
  uint64_t supported_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));
  if (attr1->HasFuseType(loop::FuseType::kGather) && (attr2->GetAllFuseType() == supported_type)) {
    return CheckGatherElemwiseFuse(node1, node2, backend_spec->gather_spec.enable_gather_broadcast_fusion);
  }

  GELOGI(
      "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Gather can only vertically fuse "
      "with elementwise]", node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
      ge::NotFuseReasonCode(ge::NotFuseReason::kGatherCanOnlyVerticallyFuseWithElementwise));
  return false;
}

bool GatherFusionStrategy::IsForwardFuseGather(const NodePtr &node1, const NodePtr &node2) {
  if (node2->GetType() != kFusedAscBackendType) {
    return true;
  }

  // gather与concat融合时, FusedAscBackend可能fuse type包含gather，但node1不一定连着node2中的Gather
  for (const auto &node_and_anchor : node1->GetOutDataNodesAndAnchors()) {
    if (node_and_anchor.first == node2) {
      int32_t in_anchor_idx;
      auto node = BackendUtils::GetFusedAscBackendInputNode(node2, node_and_anchor.second->GetIdx(), in_anchor_idx);
      const auto attr = BackendUtils::GetNodeAutoFuseAttr(node);
      GE_ASSERT_NOTNULL(attr);
      if (attr->HasFuseType(loop::FuseType::kGather)) {
        return true;
      }
    }
  }
  return false;
}

// gather 向下垂直融合，设置优先级
FusionPriority GatherFusionStrategy::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr);
  FusionPriority fusion_priority = FusionPriority::DEFAULT;
  // 首轮融合才要处理，只有AscBackend场景
  if (attr->GetFuseType() == loop::FuseType::kGather && BackendUtils::IsVertical(node1, node2)) {
    fusion_priority = FusionPriority::HIGH;
    GELOGD("node1 %s(Gather) --> node2 %s(*) priority:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
           fusion_priority);
  }
  return fusion_priority;
}

// gather 调度轴个数不同不能融合，不区分垂直融合还是水平融合
bool GatherFusionStrategy::CheckGatherFuseAxis(const NodePtr &node1, const NodePtr &node2) {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  const auto graph_attr1 = graph1->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr1);
  const auto graph_attr2 = graph2->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr2);
  // node2暂不支持broadcast
  if (graph_attr1->axis.size() != graph_attr2->axis.size()) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In gather fusion occasion, node1 axis "
        "size(%zu) not equal to node2 axis size(%zu)]", node1->GetNamePtr(), node1->GetType().c_str(),
        node2->GetNamePtr(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kGatherNodeAxisSizeNotEqual),
        graph_attr1->axis.size(), graph_attr2->axis.size());
    return false;
  }
  return true;
}

bool GatherFusionStrategy::CheckGatherConcatFuse(const NodePtr &node1, const NodePtr &node2,
                                                 const bool enable_gather_concat, const bool enable_gather_broadcast) {
  if (!BackendUtils::IsVertical(node1, node2) || !enable_gather_concat) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Gather can not fuse with Concat]",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kGatherCanNotFuseWithConcat));
    return false;
  }

  if (!CheckGatherWithView(node1, node2, enable_gather_broadcast)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In gather fusion occasion, node1 is "
        "Gather, node2 is Concat and node2 input has view op]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kGatherNextNodeIsConcatAndInputHasViewOp));
    return false;
  }
  return true;
}

bool GatherFusionStrategy::CheckGatherReduceFuse(const NodePtr &node1, const NodePtr &node2,
                                                 const bool enable_gather_reduce, const bool enable_gather_broadcast) {
  if (!enable_gather_reduce) {
    GELOGI(
      "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Gather can not fuse with Reduce]",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kGatherCanNotFuseWithReduce));
    return false;
  }
  if (!CheckGatherFuseAxis(node1, node2)) {
    return false;
  }
  if (!CheckGatherWithView(node1, node2, enable_gather_broadcast)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In gather fusion occasion, node1 is "
        "Gather, node2 is Reduce and node2 input has view op]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kGatherNextNodeIsReduceAndInputHasViewOp));
    return false;
  }
  return true;
}

bool GatherFusionStrategy::CheckGatherElemwiseFuse(const NodePtr &node1, const NodePtr &node2,
                                                 const bool enable_gather_broadcast) {
  if (!BackendUtils::IsVertical(node1, node2)) {
    return false;
  }
  if (!CheckGatherFuseAxis(node1, node2)) {
    return false;
  }
  if (!CheckGatherWithView(node1, node2, enable_gather_broadcast)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In gather fusion occasion, node1 is "
        "Gather, node2 is Pointwise and node2 input has view op]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kGatherNextNodeIsPointWiseAndInputHasViewOp));
    return false;
  }
  return true;
}

// gather 直连broadcast时不能融合
bool GatherFusionStrategy::CheckGatherWithView(const NodePtr &node1, const NodePtr &node2,
                                               const bool enable_gather_broadcast) {
  // node2中不包含View类算子
  if (BackendUtils::IsNodeAllInputsAreSimplestLoad(node2)) {
    return true;
  }

  if (!enable_gather_broadcast) {
    return false;
  }

  // node2与gather直连路径中不包含View类算子
  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));

  // 检查是否存在既有水平融合又有垂直融合的复杂场景
  // 既存在垂直连接（node1->node2），又存在水平共享输入
  bool has_both_horizontal_and_vertical =
      !node_fuse_info.GetNode1ToNode2LinkMap().empty() &&
      !node_fuse_info.GetSameInputMap().empty();
  if (has_both_horizontal_and_vertical) {
    // 在这种复杂场景下，检查 node2（elem）是否有非SimplestLoad的输入（包括broadcast等view操作）
    // 如果有，拒绝融合以防止broadcast被错误地拷贝到Gather
    if (!BackendUtils::IsNodeAllInputsAreSimplestLoad(node2)) {
      GELOGI("Gather %s and elem %s have both horizontal and vertical fusion, but elem has view operations "
             "(such as broadcast) in its input path. Reject fusion to prevent broadcast from being "
             "incorrectly copied to Gather.",
             node1->GetNamePtr(), node2->GetNamePtr());
      return false;
    }
  }

  for (const auto &subgraph_link : node_fuse_info.GetNode1ToNode2LinkMap()) {
    std::vector<ViewOpAttrInfo> attr_infos;
    if (!BackendUtils::CurNodeInputIsSimplestLoad(node2, subgraph_link.second, attr_infos)) {
      return false;
    }
  }
  return true;
}

REGISTER_FUSION_STRATEGY(GatherFusionStrategy, loop::FuseType::kGather);
}  // namespace ge
