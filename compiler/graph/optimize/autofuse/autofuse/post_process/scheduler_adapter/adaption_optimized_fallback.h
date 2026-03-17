/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_OPTIMIZED_FALLBACK_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_OPTIMIZED_FALLBACK_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "post_process/post_process_util.h"
#include "adaption_fallback_load.h"

namespace ge {
namespace asc_adapt {
// 根据axis在找到做过broadcast后的repeat作为当前broadcast节点的repeat，适配transpose并存场景
inline Status UpdateToAfterBroadcastRepeats(const std::vector<int64_t> &axis1, std::vector<ge::Expression> &repeats1,
                                     const std::vector<int64_t> &axis2, const std::vector<ge::Expression> &repeats2) {
  std::unordered_map<int64_t, ge::Expression> axis2_to_repeats2;
  for (size_t i = 0; i < axis2.size(); ++i) {
    axis2_to_repeats2[axis2[i]] = repeats2[i];
  }

  for (size_t i = 0; i < axis1.size(); ++i) {
    auto it = axis2_to_repeats2.find(axis1[i]);
    if (it != axis2_to_repeats2.end()) {
      // 找到匹配的axis，用repeats2的值覆盖repeats1的值
      repeats1[i] = it->second;
    }
  }
  return SUCCESS;
}

inline Status InsertBroadcastBeforeNode(AscGraph &asc_graph, const NodePtr &node, const TensorAttrInfo &temp_cur_attr,
                                        const TensorAttrInfo &temp_peer_out_attr,
                                        const std::pair<int32_t, ViewOpAttrInfo> &input_index_to_view_info) {
  const auto &broadcast_info = input_index_to_view_info.second.broadcast_info;
  auto input_index = input_index_to_view_info.first;
  GELOGI("node %s(%s) need to add broadcast node before input(%d) with axis id(%s) in graph %s.",
         node->GetName().c_str(), node->GetType().c_str(), input_index,
         AutofuseUtils::VectorToStr(broadcast_info).c_str(), asc_graph.GetName().c_str());

  GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, temp_peer_out_attr.topo_id, broadcast_info.size()));
  auto b_topo_id = temp_peer_out_attr.topo_id + broadcast_info.size();
  const auto b_dtype = temp_peer_out_attr.dtype;
  const std::vector<int64_t> b_sched_axis = temp_peer_out_attr.sched_axis;
  const std::vector<int64_t> b_axis = temp_peer_out_attr.axis;
  std::vector<ge::Expression> b_repeats = temp_peer_out_attr.repeats;
  GE_ASSERT_SUCCESS(UpdateToAfterBroadcastRepeats(b_axis, b_repeats, temp_cur_attr.axis, temp_cur_attr.repeats));
  std::vector<ge::Expression> b_strides = temp_peer_out_attr.strides;
  GE_ASSERT_SUCCESS(BackendUtils::UpdateContinueStrides(b_repeats, b_strides));

  auto connect_node = static_cast<NodePtr>(node);
  for (auto index = 0U; index < broadcast_info.size(); index++) {
    const auto b_node = CreateBroadcastNode(asc_graph, node, broadcast_info, index);
    GE_ASSERT_NOTNULL(b_node);
    GE_ASSERT_SUCCESS(ConnectNodeBeforeNode(connect_node, b_node, input_index));
    GE_ASSERT_SUCCESS(UpdateBroadcastNodeAttrs(b_node, b_axis, b_repeats, b_strides, broadcast_info[index]));
    GE_ASSERT_SUCCESS(UpdateBroadcastNodeSchedInfo(b_node, b_sched_axis));
    GE_ASSERT_SUCCESS(UpdateNodeTopoInfo(b_node, b_topo_id));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(b_node, DT_FLOAT, b_dtype));  // 默认类型是DT_FLOAT
    b_topo_id--;
    connect_node = b_node;
    input_index = 0;
  }
  return SUCCESS;
}

inline Status UpdateReduceNodeRepeats(const NodePtr &asc_node, TensorAttrInfo &temp_cur_attr) {
  auto reduce_attrs = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  if (reduce_attrs == nullptr) {
    return SUCCESS;
  }

  const auto &original_axis = reduce_attrs->GetReduceOriginalAxis();
  const auto &original_repeats = reduce_attrs->GetReduceOriginalRepeats();

  if (original_axis.empty() || original_repeats.empty()) {
    return SUCCESS;
  }

  std::unordered_map<int64_t, size_t> original_axis_to_index;
  for (size_t i = 0U; i < original_axis.size(); ++i) {
    original_axis_to_index[original_axis[i]] = i;
  }

  for (size_t i = 0U; i < temp_cur_attr.axis.size(); ++i) {
    auto it = original_axis_to_index.find(temp_cur_attr.axis[i]);
    if (it != original_axis_to_index.end()) {
      size_t original_index = it->second;
      temp_cur_attr.repeats[i] = original_repeats[original_index];
    }
  }

  return SUCCESS;
}

inline Status GetBroadcastInfoForNode(const NodePtr &cur_node, TensorAttrInfo &temp_cur_attr,
                                      const NodePtr &peer_out_node, ViewOpAttrInfo &attr_info,
                                      const NodePtr &asc_node) {
  attr_info.broadcast_info.clear();

  if (IsReduceNode(cur_node)) {
    GE_ASSERT_SUCCESS(UpdateReduceNodeRepeats(asc_node, temp_cur_attr));
  }

  TensorAttrInfo temp_peer_out_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeTensorAttrInfo(peer_out_node, temp_peer_out_attr));
  GE_ASSERT_SUCCESS(BackendUtils::FusedBackSteppingViewOpBroadcast(temp_cur_attr, temp_peer_out_attr, attr_info));

  return SUCCESS;
}

inline Status OptimizedBackSteppingPro(AscGraph &asc_graph, const NodePtr &cur_node, const NodePtr &asc_node) {
  TensorAttrInfo temp_cur_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeTensorAttrInfo(cur_node, temp_cur_attr));
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(cur_node, peer_out_nodes));
  int32_t index = 0;
  for (const auto &peer_out_node : peer_out_nodes) {
    TensorAttrInfo temp_peer_out_attr;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeTensorAttrInfo(peer_out_node, temp_peer_out_attr));
    ViewOpAttrInfo attr_info;
    GE_ASSERT_SUCCESS(GetBroadcastInfoForNode(cur_node, temp_cur_attr, peer_out_node, attr_info, asc_node));

    if (!attr_info.broadcast_info.empty()) {
      std::pair<int32_t, ViewOpAttrInfo> input_index_to_view_info(index, attr_info);
      GELOGD(
          "optimizes back step, graph %s, peer out node:%s(%s) repeats:%s strides:%s axis:%s, sched axis:%s."
          "cur node:%s(%s) repeats:%s strides:%s axis:%s, sched axis:%s.",
          asc_graph.GetName().c_str(), peer_out_node->GetNamePtr(), peer_out_node->GetType().c_str(),
          AutofuseUtils::VectorToStr(temp_peer_out_attr.repeats).c_str(),
          AutofuseUtils::VectorToStr(temp_peer_out_attr.strides).c_str(),
          AutofuseUtils::VectorToStr(temp_peer_out_attr.axis).c_str(),
          AutofuseUtils::VectorToStr(temp_peer_out_attr.sched_axis).c_str(), cur_node->GetNamePtr(),
          cur_node->GetType().c_str(), AutofuseUtils::VectorToStr(temp_cur_attr.repeats).c_str(),
          AutofuseUtils::VectorToStr(temp_cur_attr.strides).c_str(),
          AutofuseUtils::VectorToStr(temp_cur_attr.axis).c_str(),
          AutofuseUtils::VectorToStr(temp_cur_attr.sched_axis).c_str());
      GE_ASSERT_SUCCESS(
          InsertBroadcastBeforeNode(asc_graph, cur_node, temp_cur_attr, temp_peer_out_attr, input_index_to_view_info));
    }
    index++;
  }

  return SUCCESS;
}

inline Status OptimizedFallbackPro(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  if (autofuse_attr->HasFuseType(loop::FuseType::kConcat)) {
    GELOGI("graph %s fuse type is concat, don't fallback.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(AscGraphUtils::GetComputeGraph(asc_graph), asc_node));
  for (const auto &node : asc_graph.GetAllNodes()) {
    // 1、前面不需要插broadcast的节点类型
    if (node->GetType() == kDataType || node->GetType() == kScalarType || node->GetType() == kLoadType ||
        node->GetType() == kGatherType || node->GetType() == kOutputType || AutofuseUtils::IsCubeNodeType(node)) {
      continue;
    }
    // 2、找到前序节点repeat是1，当前节点repeat不是1的节点（依赖前面broadcast删除；依赖canfuse融合前的补轴）
    // 2.1、考虑当前节点多输入，要根据输入index来区分是哪一个输入需要插入broadcast
    // 3、在当前节点前面插入1个或多个broadcast
    // 4、依赖cse消除输出多引用插入的多与broadcast，后续的scalar反推可以删除了
    GE_ASSERT_SUCCESS(OptimizedBackSteppingPro(asc_graph, node, asc_node));
  }
  // 5、 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

inline Status OptimizedFallback(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, OptimizedFallbackPro, "optimized_fallback"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_OPTIMIZED_FALLBACK_H
