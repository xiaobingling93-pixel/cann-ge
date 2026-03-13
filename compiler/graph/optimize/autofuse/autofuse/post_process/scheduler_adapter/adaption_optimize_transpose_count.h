/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_OPTIMIZE_TRANSPOSE_COUNT_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_OPTIMIZE_TRANSPOSE_COUNT_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "post_process/post_process_util.h"
#include "adaption_complete_node_attrs.h"

namespace ge {
namespace asc_adapt {
inline Status DelAllBackSteppingViewNodes(AscGraph &asc_graph) {
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (CheckNodeType(node, kTransposeType) || CheckNodeType(node, kBroadcastType)) {
      GE_ASSERT_SUCCESS(asc_adapt::DelNode(asc_graph, node));
    }
  }
  return SUCCESS;
}

inline Status GetGraphAxis(AscGraph &asc_graph, std::vector<int64_t> &graph_axis) {
  TensorAttrInfo graph_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, graph_attr));
  graph_axis = graph_attr.axis;
  GELOGI("graph axis: %s in graph %s.", AutofuseUtils::VectorToStr(graph_axis).c_str(),
         asc_graph.GetName().c_str());
  return SUCCESS;
}

inline Status FindLoopAxesForLoadStore(AscGraph &asc_graph, std::vector<std::vector<int64_t>> &all_loop_axes) {
  for (const auto &node : asc_graph.GetAllNodes()) {
    // tf和torch的 transpose反推流程归一，在canfuse前做完融合前的反推
    if (node->GetType() == kLoadType || node->GetType() == kStoreType || node->GetType() == kGatherType) {
      const auto node_opdesc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(node_opdesc);
      const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
      GE_ASSERT_NOTNULL(output_tensor_desc);
      auto output_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_attr);
      all_loop_axes.push_back(output_attr->axis);
    }
  }
  return SUCCESS;
}

inline Status DeduplicateVector(const std::vector<std::vector<int64_t>> &in_vector,
                                std::vector<std::vector<int64_t>> &out_vector) {
  std::unordered_set<std::string> unique_set;  // 用于存储唯一标识
  for (const auto &tmp_vector : in_vector) {
    std::string key = AutofuseUtils::VectorToStr(tmp_vector);  // 将当前 tmp_vector 转换为字符串
    if (unique_set.find(key) == unique_set.end()) {
      unique_set.insert(key);
      out_vector.push_back(tmp_vector);
    }
  }
  return SUCCESS;
}

inline Status FindUniqueLoopAxesForLoadStore(AscGraph &asc_graph, std::vector<std::vector<int64_t>> &unique_loop_axes) {
  std::vector<std::vector<int64_t>> all_loop_axes;
  GE_ASSERT_SUCCESS(FindLoopAxesForLoadStore(asc_graph, all_loop_axes));
  GE_ASSERT_SUCCESS(DeduplicateVector(all_loop_axes, unique_loop_axes));
  for (size_t i = 0; i < unique_loop_axes.size(); ++i) {
    GELOGI("find unique loop axes %u: %s, in graph %s.", i, AutofuseUtils::VectorToStr(unique_loop_axes[i]).c_str(),
           asc_graph.GetName().c_str());
  }
  return SUCCESS;
}

inline Status CountExpectedTransposesForLoad(const NodePtr &load_node,
                                             const std::vector<std::vector<int64_t>> &unique_loop_axes,
                                             std::vector<std::pair<int64_t, int64_t>> &load_store_transpose_cnt) {
  const auto load_node_opdesc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_node_opdesc);
  const auto load_output_tensor_desc = load_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(load_output_tensor_desc);
  auto load_output_attr = load_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(load_output_attr);
  for (size_t i = 0; i < unique_loop_axes.size(); ++i) {
    int64_t swap_count = 0;
    std::vector<std::pair<int64_t, int64_t>> swaps = {};
    GE_ASSERT_SUCCESS(BackendUtils::MinSwapCount(load_output_attr->axis, unique_loop_axes[i], swap_count, swaps));
    // 后端支持一个transpose交换多个轴，因此可以把同一个load上的多对轴transpose变换统计为1个
    load_store_transpose_cnt[i].first += swap_count > 0 ? 1 : 0;
    GELOGD("node %s %s, axis: %s, unique loop axes: %s, load transpose cnt: %" PRId64 ", store transpose cnt: %" PRId64
           ".",
           load_node->GetName().c_str(), load_node->GetType().c_str(),
           AutofuseUtils::VectorToStr(load_output_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(unique_loop_axes[i]).c_str(), load_store_transpose_cnt[i].first,
           load_store_transpose_cnt[i].second);
  }
  return SUCCESS;
}

inline Status CountExpectedTransposesForStore(const NodePtr &store_node,
                                              const std::vector<std::vector<int64_t>> &unique_loop_axes,
                                              std::vector<std::pair<int64_t, int64_t>> &load_store_transpose_cnt) {
  const auto store_node_opdesc = store_node->GetOpDesc();
  GE_ASSERT_NOTNULL(store_node_opdesc);
  const auto store_output_tensor_desc = store_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(store_output_tensor_desc);
  auto store_output_attr = store_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(store_output_attr);
  for (size_t i = 0; i < unique_loop_axes.size(); ++i) {
    int64_t swap_count = 0;
    std::vector<std::pair<int64_t, int64_t>> swaps = {};
    // store统计的是把前面load变成unique_loop_axes后，再变回store的axis需要插入的transpose数量
    GE_ASSERT_SUCCESS(BackendUtils::MinSwapCount(unique_loop_axes[i], store_output_attr->axis, swap_count, swaps));
    // 后端支持一个transpose交换多个轴，因此可以把同一个load上的多对轴transpose变换统计为1个
    load_store_transpose_cnt[i].second += swap_count > 0 ? 1 : 0;
    GELOGD("node %s %s axis: %s, unique loop axes: %s, load transpose cnt: %" PRId64 ", store transpose cnt: %" PRId64
           ".",
           store_node->GetName().c_str(), store_node->GetType().c_str(),
           AutofuseUtils::VectorToStr(store_output_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(unique_loop_axes[i]).c_str(), load_store_transpose_cnt[i].first,
           load_store_transpose_cnt[i].second);
  }
  return SUCCESS;
}

inline Status CountExpectedTransposesForLoopAxis(AscGraph &asc_graph,
                                                 const std::vector<std::vector<int64_t>> &unique_loop_axes,
                                                 std::vector<std::pair<int64_t, int64_t>> &load_store_transpose_cnt) {
  for (const auto &node : asc_graph.GetAllNodes()) {
    if ((node->GetType() == kLoadType) || (node->GetType() == kGatherType)) {
      // 不需要考虑load前面的data节点输出多引用，因为反推都是在load节点后面加view节点，多以引用也是在各自的引用分支加view节点
      NodePtr peer_out_node;
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_out_node, 0));
      if (peer_out_node->GetType() != kDataType) {
        // 注意前面非data的中间load不需要做统计（相当于transpose前移或者后移）
        GELOGD("node %s %s, peer out node is %s %s, not data, in graph %s, don't count transpose number.",
               node->GetName().c_str(), node->GetType().c_str(), peer_out_node->GetName().c_str(),
               peer_out_node->GetType().c_str(), asc_graph.GetName().c_str());
        continue;
      }
      GE_ASSERT_SUCCESS(CountExpectedTransposesForLoad(node, unique_loop_axes, load_store_transpose_cnt));
    } else if (node->GetType() == kStoreType) {
      GE_ASSERT_SUCCESS(CountExpectedTransposesForStore(node, unique_loop_axes, load_store_transpose_cnt));
    }
  }
  return SUCCESS;
}

inline Status FindOptimalTransposeCount(const std::vector<std::pair<int64_t, int64_t>> &load_store_transpose_cnt, size_t &min_index) {
  GE_ASSERT_TRUE(!load_store_transpose_cnt.empty());
  int64_t min_sum = std::numeric_limits<int64_t>::max();
  int64_t min_first = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < load_store_transpose_cnt.size(); ++i) {
    const auto &transpose_cnt = load_store_transpose_cnt[i];
    int64_t current_sum = transpose_cnt.first + transpose_cnt.second;
    if (current_sum < min_sum) {
      min_sum = current_sum;
      min_first = transpose_cnt.first;
      min_index = i;
    } else if (current_sum == min_sum) {
      if (transpose_cnt.first < min_first) {
        min_first = transpose_cnt.first;
        min_index = i;
      }
    }
  }
  return SUCCESS;
}

inline Status SelectOptimalLoopAxisByTransposeCount(
    const std::vector<std::vector<int64_t>> &unique_loop_axes,
    const std::vector<std::pair<int64_t, int64_t>> &load_store_transpose_cnt, std::vector<int64_t> &optimal_loop_axes,
    size_t &optimal_loop_axis_index) {
  GE_ASSERT_SUCCESS(FindOptimalTransposeCount(load_store_transpose_cnt, optimal_loop_axis_index));
  optimal_loop_axes = unique_loop_axes[optimal_loop_axis_index];
  GELOGI("slect optimal loop axes : %s.", AutofuseUtils::VectorToStr(optimal_loop_axes).c_str());
  return SUCCESS;
}

inline bool IsInUpdateLoopAxisBlackList(const NodePtr &node) {
  if (CheckNodeType(node, kDataType) || CheckNodeType(node, kStoreType) || CheckNodeType(node, kOutputType) ||
      CheckNodeType(node, kLoadType) || CheckNodeType(node, kScalarType) || CheckNodeType(node, kGatherType)) {
    GELOGD("node %s(%s) no need update loop axis.", node->GetNamePtr(), node->GetType().c_str());
    return true;
  }
  return false;
}

inline Status UpdateAscGraphWithAppliedLoopAxis(AscGraph &asc_graph, std::vector<int64_t> &graph_axis,
                                                const std::vector<int64_t> &optimal_loop_axes) {
  // 更新graph
  const auto graph_attr = AscGraphUtils::GetComputeGraph(asc_graph)->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  int64_t swap_count = 0;
  std::vector<std::pair<int64_t, int64_t>> transpose_info;
  GE_ASSERT_SUCCESS(BackendUtils::MinSwapCount(graph_axis, optimal_loop_axes, swap_count, transpose_info));
  GE_ASSERT_SUCCESS(BackendUtils::SwapGraphAxis(transpose_info, graph_attr->axis));

  // 更新node
  for (const auto &node : asc_graph.GetAllNodes()) {
    // data、scalar、load、store、output节点不需要刷轴，其他节点（这些节点轴可能不同）刷成和最优轴一致
    if (IsInUpdateLoopAxisBlackList(node)) {
      continue;
    }
    const auto node_opdesc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(node_opdesc);
    auto node_attr = node_opdesc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    for (size_t i = 0; i < node_opdesc->GetAllOutputsDescSize(); ++i) {
      const auto output_tensor_desc = node_opdesc->MutableOutputDesc(i);
      GE_ASSERT_NOTNULL(output_tensor_desc);
      auto output_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_attr);
      // 只有非空轴信息节点才在此处刷轴，其他节点在后续的补轴过程中刷轴，strides在补轴流程刷成连续
      if (output_attr->axis.size() != 0U) {
        GELOGD("node %s(%s) update loop axis from %s to %s.", node->GetNamePtr(), node->GetType().c_str(),
               AutofuseUtils::VectorToStr(output_attr->axis).c_str(),
               AutofuseUtils::VectorToStr(optimal_loop_axes).c_str());
        int64_t output_attr_swap_count = 0;
        std::vector<std::pair<int64_t, int64_t>> output_attr_transpose_info;
        GE_ASSERT_SUCCESS(BackendUtils::MinSwapCount(output_attr->axis, optimal_loop_axes, output_attr_swap_count,
                                                     output_attr_transpose_info));
        auto sched_axis = node_attr->sched.axis; // ApplySwaps接口会更新sched axis，此处不希望更新
        GE_ASSERT_SUCCESS(BackendUtils::ApplySwaps(output_attr->axis, output_attr->repeats, output_attr->strides,
                                                   sched_axis, output_attr_transpose_info));
        GELOGI("applied transpose info, node %s %s sched_axis:%s, out tensor axis:%s repeats:%s stride:%s.",
               node->GetName().c_str(), node->GetType().c_str(),
               AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str(),
               AutofuseUtils::VectorToStr(output_attr->axis).c_str(),
               AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
               AutofuseUtils::VectorToStr(output_attr->strides).c_str());
      }
    }
  }
  // 预留接口给slices场景刷新load轴做特殊处理
  return SUCCESS;
}

inline Status OptimizeTransposeCountPro(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  if (BackendUtils::IsCubeAscNode(asc_node)) {
    GELOGI("graph %s fuse type is cube, don't need to optimize transpose count.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  // 1、删除原本的transpose节点,便于对比轴id获取最优循环轴;删除原本的broadcast节点，在反推后重新插入，可以不用因为transpose删除而刷broadcast的轴信息
  GE_ASSERT_SUCCESS(DelAllBackSteppingViewNodes(asc_graph));
  std::vector<int64_t> graph_axis;
  GE_ASSERT_SUCCESS(GetGraphAxis(asc_graph, graph_axis));

  // 2、找出所有load和store可能循环轴，去重
  std::vector<std::vector<int64_t>> unique_loop_axes;
  GE_ASSERT_SUCCESS(FindUniqueLoopAxesForLoadStore(asc_graph, unique_loop_axes));
  if (unique_loop_axes.size() <= 1U) {
    GELOGI("only one or zero unique loop axes in graph %s, dont't need to optimize transpose count.",
           asc_graph.GetName().c_str());
    return SUCCESS;
  }

  // 3、根据某个循环轴为基准，应用到loat和store，统计load和store预期反推出的transpose个数
  // 初始化load和store的transpose计数器，pair的first是load的transpose计数器，second是store的transpose计数器
  std::vector<std::pair<int64_t, int64_t>> load_store_transpose_cnt(unique_loop_axes.size(), {0, 0});
  GE_ASSERT_SUCCESS(CountExpectedTransposesForLoopAxis(asc_graph, unique_loop_axes, load_store_transpose_cnt));

  // 4、使用transpose都在load上的轴，避免transpose在store上，否则会推出多个transpose
  std::vector<int64_t> optimal_loop_axes = unique_loop_axes[0];
  for (auto index = 0U; index < load_store_transpose_cnt.size(); index++) {
    if (load_store_transpose_cnt[index].second == 0U) {
      optimal_loop_axes = unique_loop_axes[index];
      break;
    }
  }
  //  找出transpose总个数最少的循环轴，如果总个数相等，找出load预期反推出的transpose个数少的循环轴，应用到load和store
  //  （当前先注释，后端支持后放开）GE_ASSERT_SUCCESS(SelectOptimalLoopAxisByTransposeCount(unique_loop_axes,
  //  （当前先注释，后端支持后放开）load_store_transpose_cnt, optimal_loop_axes, optimal_loop_axis_index));

  // 5、根据应用后的循环轴，刷新ascgraph图
  GE_ASSERT_SUCCESS(UpdateAscGraphWithAppliedLoopAxis(asc_graph, graph_axis, optimal_loop_axes));
  return SUCCESS;
}

inline Status OptimizeTransposeCount(const ComputeGraphPtr &graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(graph, OptimizeTransposeCountPro, "optimize_transpose_count"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_OPTIMIZE_TRANSPOSE_COUNT_H
