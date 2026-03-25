/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/not_fuse_reason_code.h"
#include "slice_split_fusion_strategy.h"
#include "backend/backend_spec.h"
#include "base/base_types.h"

namespace ge {
namespace {
// load承载的view信息分类
enum class LoadViewType : uint32_t{
  ViewLoadSlice = 0U,       // load承载slice信息
  ViewLoadBroadcast = 1U,   // load承载broadcast信息
  ViewLoadTranspose = 2U,   // load承载transpose信息
  NonViewLoad = 3U          // load不承载view信息
};

// 用于判断attr_infos中是否包含指定的load类型
bool AnyOfAttrInfosContainSpecifiedLoadTypes(const NodePtr &node, const std::vector<ViewOpAttrInfo> &attr_infos,
                                             const std::set<LoadViewType> &has_load_types) {
  // 预先检查需要检查的类型是否存在
  bool has_slice = has_load_types.count(LoadViewType::ViewLoadSlice);
  bool has_broadcast = has_load_types.count(LoadViewType::ViewLoadBroadcast);
  bool has_transpose = has_load_types.count(LoadViewType::ViewLoadTranspose);
  bool has_non_view_load = has_load_types.count(LoadViewType::NonViewLoad);

  for (const auto &attr_info : attr_infos) {
    // 检查slice
    if (has_slice && attr_info.slice_info.two_slice_node_flag) {
      GELOGI("node %s(%s) has view type slice", node->GetName().c_str(), node->GetType().c_str());
      return true;
    }
    // 检查broadcast
    if (has_broadcast && !attr_info.broadcast_info.empty()) {
      GELOGI("node %s(%s) has view type broadcast", node->GetName().c_str(), node->GetType().c_str());
      return true;
    }
    // 检查transpose
    if (has_transpose && !attr_info.transpose_info.empty()) {
      GELOGI("node %s(%s) has view type transpose", node->GetName().c_str(), node->GetType().c_str());
      return true;
    }
    // 检查非view类的load：调用CurNodeInputIsSimplestLoad、PreNodeInputIsSimplestLoad获得attr_infos时，
    // 会为每个load生成一个attr_info，如果某个attr_info没有view类op信息，则是非view类的load产生的
    if (has_non_view_load) {
      bool is_view_op = (!attr_info.broadcast_info.empty()) ||
                        (attr_info.slice_info.two_slice_node_flag) ||
                        !attr_info.transpose_info.empty();
      if (!is_view_op) {
        GELOGI("node %s(%s) has non-view type load", node->GetName().c_str(), node->GetType().c_str());
        return true;
      }
    }
  }
  return false;
}

/**
 * 该函数用于获取node1和node2之间的连边关系，并判断连边路径上前后节点的Load是否满足
 * 特定的view配对关系，例如前序slice类load接后序broadcast类load。
 * @param node1 第一个节点的指针（输入参数），表示CanFuse垂直关系的前序Ascend节点。
 * @param node2 第二个节点的指针（输入参数），表示CanFuse垂直关系的后序Ascend节点。
 * @param prev_load_types node1需要检查的Load类型。
 * @param post_load_types node2需要检查的Load类型。
 * @return 返回判断结果。
 */
bool CheckIfSubGraphLinksHaveSpecifiedLoadTypePairs(const NodePtr &node1, const NodePtr &node2,
                                                    const std::set<LoadViewType> &prev_load_types,
                                                    const std::set<LoadViewType> &post_load_types) {
  // node1与node2直连路径中不包含View类算子
  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  for (const auto &subgraph_link : node_fuse_info.GetNode1ToNode2LinkMap()) {
    std::vector<ViewOpAttrInfo> attr_infos;
    (void) BackendUtils::CurNodeInputIsSimplestLoad(node2, subgraph_link.second, attr_infos);
    std::vector<ViewOpAttrInfo> pre_node_attr_infos;
    (void) BackendUtils::PreNodeInputIsSimplestLoad(node2, subgraph_link.second, pre_node_attr_infos);
    // 判断subgraph_link的Node1侧是否含有prev_load_types
    if (!AnyOfAttrInfosContainSpecifiedLoadTypes(node1, pre_node_attr_infos, prev_load_types)) {
      continue;
    }
    // 若存在prev_load_types，则判断subgraph_link的Node2侧的load是否是post_load_types
    if (AnyOfAttrInfosContainSpecifiedLoadTypes(node2, attr_infos, post_load_types)) {
      return true;
    }
    GELOGD("node %s(%s) input index(%d) is simplest load.", node2->GetName().c_str(),
           node2->GetType().c_str(), subgraph_link.second);
  }
  return false;
}

bool CheckIfSliceEndDimIsOne(const NodePtr &node1) {
  GELOGI("node1 name is %s, type is %s.", node1->GetName().c_str(), node1->GetType().c_str());
  auto buffer = node1->GetOutDataAnchor(0).get();
  GE_ASSERT_NOTNULL(buffer);
  GE_ASSERT_NOTNULL(buffer->GetOwnerNode());
  GE_ASSERT_NOTNULL(buffer->GetOwnerNode()->GetOpDesc());
  const auto desc = buffer->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(buffer->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  const auto sym_attr = desc->GetAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(sym_attr);
  std::vector<Expression> output_dims = sym_attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  GE_ASSERT_TRUE(!output_dims.empty(), "node [%s] output end dim is 0.", node1->GetName().c_str());
  auto index = output_dims.size() - 1UL;
  GELOGI("slice op output end dim is %s.", output_dims[index].Serialize().get());
  if (BackendUtils::IsEqOne(output_dims[index])) {
    return true;
  }
  return false;
}

bool CheckIfSliceAscBackendNodeContainsBroadcast(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  if (node->GetType() != kAscBackendType) {
    GELOGD("node %s(%s) is not asc backend type, does not have asc subgraph.", node->GetName().c_str(),
           node->GetType().c_str());
    return false;
  }
  const auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  if (!autofuse_attr->HasFuseType(loop::FuseType::kSliceSplit)) {
    GELOGD("node %s(%s) does not have fuse type slice.", node->GetName().c_str(), node->GetType().c_str());
    return false;
  }
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  const auto subgraph_input_nodes = graph->GetInputNodes();
  for (auto &data_node: subgraph_input_nodes) {
    const auto data_out_anchor = data_node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(data_out_anchor);
    for (const auto &node_peer_anchor : data_out_anchor->GetPeerInDataAnchors()) {
      const auto load_node = node_peer_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(load_node);
      ViewOpAttrInfo attr_info;
      GE_ASSERT_SUCCESS(BackendUtils::FusedBackSteppingViewOp(data_node, load_node, attr_info, true));
      NodePtr found_node = nullptr;
      GE_ASSERT_SUCCESS(BackendUtils::GetViewOpNextNodeByLoad(load_node, found_node));
      if (found_node != nullptr) {
        GELOGD("in the ascgraph of slice node %s(%s), data op %s and load op %s are followed by broadcast op %s(%s).",
               node->GetNamePtr(), node->GetType().c_str(), data_node->GetName().c_str(), load_node->GetName().c_str(),
               found_node->GetName().c_str(), found_node->GetType().c_str());
        return true;
      }
      if (!attr_info.broadcast_info.empty()) {
        GELOGD("in the ascgraph of slice node %s(%s), load op %s contains broadcast info.", node->GetNamePtr(),
               node->GetType().c_str(), load_node->GetName().c_str());
        return true;
      }
    }
    GELOGD("in the ascgraph of slice node %s(%s), data op %s does not contain broadcast info, and does not followed by "
           "any broadcast op.", node->GetName().c_str(), node->GetType().c_str(), data_node->GetName().c_str());
  }
  GELOGD("slice node %s(%s) does not contain broadcast.", node->GetName().c_str(), node->GetType().c_str());
  return false;
}

bool CheckIfSliceNodeContainsBroadcast(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("check if node %s(%s) has fuse type slice and ascgraph of this node contains broadcast.",
         node->GetName().c_str(), node->GetType().c_str());
  if (node->GetType() == kFusedAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
    for (auto &sub_node : graph->GetAllNodes()) {
      GE_ASSERT_NOTNULL(sub_node);
      if (CheckIfSliceAscBackendNodeContainsBroadcast(sub_node)) {
        GELOGD("sub-node %s(%s) in the fused subgraph of node %s(%s) has fuse type slice, and ascgraph of sub-node %s "
               "contains broadcast.", sub_node->GetName().c_str(), sub_node->GetType().c_str(), node->GetName().c_str(),
               node->GetType().c_str(), sub_node->GetName().c_str());
        return true;
      }
    }
    GELOGD("fused subgraph of node %s(%s) does not contain broadcast.", node->GetName().c_str(), node->GetType().c_str());
    return false;
  }
  return CheckIfSliceAscBackendNodeContainsBroadcast(node);
}
}

bool SliceSplitFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  // 同一个AscBackend内同时含slice与broadcast时，如果CanFuse进行了融合，会产生反推导致错误
  if ((attr1->HasFuseType(loop::FuseType::kSliceSplit) && CheckIfSliceNodeContainsBroadcast(node1)) ||
      (attr2->HasFuseType(loop::FuseType::kSliceSplit) && CheckIfSliceNodeContainsBroadcast(node2))) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][This fusion introduces slice node with view op]",
           node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kFusedSliceHasViewOp));
    return false;
  }
  bool is_vertical = BackendUtils::IsVertical(node1, node2);
  // slice不做水平融合，但是既有水平又有垂直连接的情况可以继续处理，slice与非slice类节点不发生横向融合，由框架判断处理
  if (!is_vertical && attr1->HasFuseType(loop::FuseType::kSliceSplit) && attr2->HasFuseType(loop::FuseType::kSliceSplit)) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][horizontal slice/split without vertical link]",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kSliceHasOnlyHorizontalLink));
    return false;
  }
  if (is_vertical) {
    // slice不与非slice类节点做前向融合，sub_graph_links后序实际不连接slice的时候不拦截
    if (attr2->HasFuseType(loop::FuseType::kSliceSplit)) {
      std::set prev_load_types = {LoadViewType::NonViewLoad, LoadViewType::ViewLoadBroadcast, LoadViewType::ViewLoadTranspose};
      std::set post_load_types = {LoadViewType::ViewLoadSlice};
      if (CheckIfSubGraphLinksHaveSpecifiedLoadTypePairs(node1, node2, prev_load_types, post_load_types)) {
        GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In slice/split fusion occasion, "
               "node2 %s(%s) connected input contains slice/split]", node1->GetNamePtr(), node1->GetType().c_str(),
               node2->GetName().c_str(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kNodeInputHasSplit),
               node2->GetNamePtr(), node2->GetType().c_str());
        return false;
      }
    }
    const auto backend_spec = optimize::BackendSpec::GetInstance();
    GE_ASSERT_NOTNULL(backend_spec);
    if (!backend_spec->slice_split_spec.slice_fuse_with_end_dim_1 && (attr1->GetFuseType() == loop::FuseType::kSliceSplit) &&
      !attr2->HasFuseType(loop::FuseType::kSliceSplit) && CheckIfSliceEndDimIsOne(node1)) {
      return false;
    }
    // 判断sub_graph_links前序的load节点是否含有slice类型，不是slice load，则后续有view也不影响融合判断；
    std::set prev_load_types = {LoadViewType::ViewLoadSlice};
    std::set post_load_types = {LoadViewType::ViewLoadBroadcast, LoadViewType::ViewLoadTranspose};
    if (attr1->HasFuseType(loop::FuseType::kSliceSplit) &&
        CheckIfSubGraphLinksHaveSpecifiedLoadTypePairs(node1, node2, prev_load_types, post_load_types)) {
      GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][This fusion introduces slice node with view op]",
             node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
             ge::NotFuseReasonCode(ge::NotFuseReason::kFusedSliceHasViewOp));
      return false;
    }
  }
  return true;
}

REGISTER_FUSION_STRATEGY(SliceSplitFusionStrategy, loop::FuseType::kSliceSplit);
}
