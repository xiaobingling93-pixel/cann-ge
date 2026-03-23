/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend_utils.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_tensor.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "common/util/mem_utils.h"
#include "mmpa/mmpa_api.h"
#include "utils/autofuse_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/autofuse_attrs.h"
#include "post_process/post_process_util.h"
#include "ascir_ops.h"
#include "post_process/scheduler_adapter/torch_adaption_fallback_load.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"
#include "asc_graph_axis_mapping.h"
#include "can_fuse/graph_manager.h"
#include "utils/not_fuse_reason_code.h"

namespace ge {
using namespace autofuse;
namespace {
template <typename Container, typename ValueType>
size_t FindIndex(const Container &container, const ValueType &value) {
  auto it = std::find(container.begin(), container.end(), value);
  if (it == container.end()) {
    return static_cast<size_t>(-1);
  }
  return std::distance(container.begin(), it);
}

bool IsLoadConnectedToSplit(const NodePtr &load_node) {
  const auto load_out_anchor = load_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(load_out_anchor);
  const auto load_peer_in_anchors = load_out_anchor->GetPeerInDataAnchors();
  for (auto &load_peer_in_anchor: load_peer_in_anchors) {
    GE_ASSERT_NOTNULL(load_peer_in_anchor);
    const auto load_peer_in_node = load_peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(load_peer_in_node);
    if (load_peer_in_node->GetType() == kSplitType) {
      GELOGD("load node %s is followed by split node %s.", load_node->GetName().c_str(), load_peer_in_node->GetName().c_str());
      return true;
    }
  }
  return false;
}

Status SwapSchedAxis(const std::vector<std::pair<int64_t, int64_t>> &transpose_info, std::vector<int64_t> &sched_axis) {
  if (sched_axis.empty()) {
    return SUCCESS;
  }
  for (const auto &info : transpose_info) {
    auto index1 = FindIndex(sched_axis, info.first);
    if (index1 == static_cast<size_t>(-1)) {
      return FAILED;
    }

    auto index2 = FindIndex(sched_axis, info.second);
    if (index2 == static_cast<size_t>(-1)) {
      return FAILED;
    }
    std::swap(sched_axis[index1], sched_axis[index2]);
  }
  return SUCCESS;
}

Status SwapTensorInfo(const std::vector<std::pair<int64_t, int64_t>> &transpose_info, std::vector<int64_t> &axis,
                      std::vector<ge::Expression> &data) {
  if (axis.empty()) {
    return SUCCESS;
  }
  for (const auto &info : transpose_info) {
    auto index1 = FindIndex(axis, info.first);
    if (index1 == static_cast<size_t>(-1)) {
      return FAILED;
    }

    auto index2 = FindIndex(axis, info.second);
    if (index2 == static_cast<size_t>(-1)) {
      return FAILED;
    }
    std::swap(axis[index1], axis[index2]);
    std::swap(data[index1], data[index2]);
  }
  return SUCCESS;
}

bool CheckValidTranspose(const ComputeGraphPtr &graph, const std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  const auto &axis = graph_attr->axis;

  auto find_axis = [&axis](const int64_t id) -> bool {
    auto it = std::find_if(axis.begin(), axis.end(), [id](const AxisPtr &axis_ptr) { return axis_ptr->id == id; });
    return it != axis.end();
  };

  for (const auto &info : transpose_info) {
    if (!find_axis(info.first)) {
      return false;
    }

    if (!find_axis(info.second)) {
      return false;
    }
  }

  return true;
}

Status GetCurAscLoadNode(const ComputeGraphPtr &graph, const int32_t index, NodePtr &load_node) {
  const auto subgraph_input_nodes = graph->GetInputNodes();
  GE_ASSERT_TRUE(subgraph_input_nodes.size() > static_cast<size_t>(index));
  const auto data_node = subgraph_input_nodes.at(index);
  GE_ASSERT_NOTNULL(data_node);
  GELOGD("get cur node %s(%s) index=%d.", data_node->GetNamePtr(), data_node->GetType().c_str(), index);
  load_node = BackendUtils::GetDataNextNode(data_node);
  GE_ASSERT_NOTNULL(load_node);
  return SUCCESS;
}

bool CanFuseByStrategyPro(const NodePtr &node1, const NodePtr &node2, uint32_t &max_fusion_node_input_size) {
  const auto fuse_type = BackendUtils::GetAllFuseType(node1, node2);
  uint64_t max_fusion_nodes_size = 0U;
  for (const auto fusion_strategy : FusionStrategyRegistry::Instance().Get(fuse_type)) {
    if (fusion_strategy != nullptr) {
      if (!fusion_strategy->CanFuse(node1, node2)) {
        return false;
      }
      const uint32_t strategy_max_fusion_node_input_size = fusion_strategy->GetMaxFusionNodeInputSize(node1, node2);
      if (strategy_max_fusion_node_input_size > max_fusion_node_input_size) {
        max_fusion_node_input_size = strategy_max_fusion_node_input_size;
      }
      const uint64_t strategy_max_fusion_nodes_size = fusion_strategy->GetMaxFusionNodesSize(node1, node2);
      if (strategy_max_fusion_nodes_size > max_fusion_nodes_size) {
        max_fusion_nodes_size = strategy_max_fusion_nodes_size;
      }
    }
  }

  auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  if (attr1->GetVectorCoreNum() != attr2->GetVectorCoreNum()) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][It has different vector core num scope, "
           "node1 vector core num is %d, node2 vector core num is %d]", node1->GetNamePtr(), node1->GetType().c_str(),
           node2->GetNamePtr(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kVectorCoreNumNotEqual),
           attr1->GetVectorCoreNum(), attr2->GetVectorCoreNum());
    return false;
  }
  // 融合后节点数超过总数限制不做融合
  if ((attr1->GetFusionNodesSize() + attr2->GetFusionNodesSize()) > max_fusion_nodes_size) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 size(%" PRIu64 ") plus node2 size(%" PRIu64 ")"
        " exceeds max fusion nodes size(%" PRIu64 ")]", node1->GetNamePtr(), node1->GetType().c_str(),
        node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kMaxFusionNodesSizeExceedThreshold), attr1->GetFusionNodesSize(),
        attr2->GetFusionNodesSize(), max_fusion_nodes_size);
    return false;
  }
  return true;
}

bool IsSplitComplete(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  // 复用缓存结果
  if (attr->GetSplitComplete()) {
    GELOGD("split node %s is compelete", node->GetName().c_str());
    return true;
  }
  GE_ASSERT_TRUE(attr->GetFuseType() == loop::FuseType::kSplit);
  GE_ASSERT_TRUE(!attr->GetOriginNodes().empty());
  auto origin_node = attr->GetOriginNodes()[0];
  GE_ASSERT_NOTNULL(origin_node);
  auto origin_output_num = origin_node->GetAllOutDataAnchorsSize();
  auto fused_split_num = 0U;
  auto asc_graph = BackendUtils::GetNodeFusedAscGraph(node);
  GE_ASSERT_NOTNULL(asc_graph);
  for (auto ascir_node: asc_graph->GetAllNodes()) {
    if (ascir_node->GetType() == kSplitType) {
      fused_split_num++;
    }
    if (kPureSplitIncludedAscirNodeTypes.count(ascir_node->GetType()) == 0U) {
      GELOGD("split AscBackend node %s has ascir node %s(%s), has fused with other types of AscBackend",
             node->GetName().c_str(), ascir_node->GetType().c_str(), ascir_node->GetName().c_str());
      return false;
    }
  }
  GELOGD("split node %s, original node %s, original output num %zu, fused split node %zu", node->GetName().c_str(),
         origin_node->GetName().c_str(), origin_output_num, fused_split_num);
  if (fused_split_num == origin_output_num) {
    // 缓存原split节点融合完整的标志
    attr->SetSplitComplete();
    return true;
  }
  return false;
}

bool IsSplitLowFusionRatio(const NodePtr &node, uint32_t &max_fusion_node_input_size) {
  GE_ASSERT_NOTNULL(node);
  const auto attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  GE_ASSERT_TRUE(attr->GetFuseType() == loop::FuseType::kSplit);
  if (attr->GetSplitFusionRatioRequirementState() == SplitFusionRatioRequirementState::SATISFIED) {
    // 复用缓存的判断结果避免重复判断
    GELOGD("node %s has high fuse ratio", node->GetName().c_str());
    return false;
  }
  GELOGD("node %s is calculating fuse ratio", node->GetName().c_str());
  uint32_t can_fuse = 0;
  GE_ASSERT_TRUE(!attr->GetOriginNodes().empty());
  auto origin_node = attr->GetOriginNodes()[0];
  GE_ASSERT_NOTNULL(origin_node);
  uint32_t total = origin_node->GetAllOutDataAnchorsSize();
  for (auto anchor: node->GetAllOutDataAnchors()) {
    for (auto peer_anchor: anchor->GetPeerInDataAnchors()) {
      auto peer_node = peer_anchor->GetOwnerNode();
      if (!BackendUtils::IsBackendFuseNode(peer_node)) {
        continue;
      }
      if (CanFuseByStrategyPro(node, peer_node, max_fusion_node_input_size)) {
        can_fuse++;
        break;
      }
    }
  }
  GE_ASSERT_TRUE(total > 0U);
  const float32_t ratio = static_cast<float32_t>(can_fuse) / static_cast<float32_t>(total);
  GELOGD("node %s, total number of output: %zu, number of can-fuse output: %zu, fuse ratio %f, threshold: %f", node->GetName().c_str(), total,
         can_fuse, ratio, kSplitLowFusionRatioThreshold);
  if (ratio > kSplitLowFusionRatioThreshold) {
    GELOGD("node %s has high fuse ratio, can fuse", node->GetName().c_str());
    // 缓存判断结果避免重复判断
    attr->SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState::SATISFIED);
    return false;
  }
  GELOGD("node %s has low fuse ratio, can not fuse", node->GetName().c_str());
  // 缓存判断结果避免重复判断
  attr->SetSplitLowFusionRatioRequirementState(SplitFusionRatioRequirementState::NOT_SATISFIED);
  return true;
}
}  // namespace

bool BackendUtils::IsEq(const Expression &e1, const Expression &e2) {
  return SymbolicUtils::StaticCheckEq(e1, e2) == TriBool::kTrue;
}

bool BackendUtils::IsEqOne(const Expression &e1) {
  return SymbolicUtils::StaticCheckEq(e1, kSymbolOne) == TriBool::kTrue;
}

bool BackendUtils::IsEqZero(const Expression &e1) {
  return SymbolicUtils::StaticCheckEq(e1, kSymbolZero) == TriBool::kTrue;
}

Status BackendUtils::UpdateContinueStrides(const std::vector<ge::Expression> &repeats, std::vector<ge::Expression> &strides) {
  auto size = repeats.size();
  if (size > 0U) {
    strides[size - 1U] = kSymbolOne;
    for (auto index = 0U; index < size - 1U; index++) {
      auto sub_index = size - 1U - index;
      strides[sub_index - 1U] = repeats[sub_index] * strides[sub_index];
    }
    for (auto index = 0U; index < size - 1U; index++) {
      if (BackendUtils::IsEqOne(repeats[index])) {
        strides[index] = kSymbolZero;
      }
    }
  }
  return SUCCESS;
}

Status BackendUtils::SwapGraphAxis(const std::vector<std::pair<int64_t, int64_t>> &transpose_info,
                                   std::vector<AxisPtr> &axis) {
  if (axis.empty()) {
    return SUCCESS;
  }
  for (const auto &info : transpose_info) {
    auto find_index = [&axis](const auto &value) -> size_t {
      auto it =
          std::find_if(axis.begin(), axis.end(), [value](const AxisPtr &axis_ptr) { return axis_ptr->id == value; });
      return (it == axis.end()) ? (static_cast<size_t>(-1)) : (std::distance(axis.begin(), it));
    };

    auto index1 = find_index(info.first);
    GE_ASSERT_TRUE(index1 != static_cast<size_t>(-1));

    auto index2 = find_index(info.second);
    GE_ASSERT_TRUE(index2 != static_cast<size_t>(-1));
    std::swap(axis[index1], axis[index2]);
  }
  return SUCCESS;
}

// 用data和load的输出tensor信息判断是否是单纯不包括view操作的load
bool BackendUtils::IsSimplestLoad(const NodePtr &node, const NodePtr &load_node, const NodePtr &data_node,
                                  std::vector<ViewOpAttrInfo> &attr_infos, const bool is_condition_with_node_type) {
  const auto attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  if (attr->HasFuseType(loop::FuseType::kConcat)){
    GELOGD("node %s(Concat) is simplest op.", node->GetNamePtr());
    return true;
  }
  // 如果load后有broadcast说明是有view算子的(transpose刷新过轴，不当作view来处理)
  if (is_condition_with_node_type) {
    NodePtr finded_node = nullptr;
    GE_ASSERT_SUCCESS(BackendUtils::GetViewOpNextNodeByLoad(load_node, finded_node));
    if (finded_node != nullptr) {
      GELOGD("data node %s(%s) and load node %s(%s) are view op, have broadcast op.", data_node->GetNamePtr(),
             data_node->GetType().c_str(), load_node->GetNamePtr(), load_node->GetType().c_str());
      return false;
    }
  }

  // 通过load反推出来，如果有broadcast或者slice轴信息，说明是view算子
  ViewOpAttrInfo attr_info;
  GE_ASSERT_SUCCESS(FusedBackSteppingViewOp(data_node, load_node, attr_info, true));
  attr_infos.push_back(attr_info);
  if ((!attr_info.broadcast_info.empty()) || (attr_info.slice_info.two_slice_node_flag) || !attr_info.transpose_info.empty()) {
    GELOGD("data node %s(%s) and load node %s(%s) have broadcast or transpose op.", data_node->GetNamePtr(),
           data_node->GetType().c_str(), load_node->GetNamePtr(), load_node->GetType().c_str());
    return false;
  }

  GELOGD("data node %s(%s) and load node %s(%s) are simplest op, no view.", data_node->GetNamePtr(),
         data_node->GetType().c_str(), load_node->GetNamePtr(), load_node->GetType().c_str());
  return true;
}

// Broadcast 反推处理函数
Status BackendUtils::BackSteppingViewOpBroadcast(TensorAttrInfo &temp_data_attr,
                                                 TensorAttrInfo &temp_load_attr, ViewOpAttrInfo &attr_info) {
  // 1.反推出 broadcast_info
  auto &broadcast_info = attr_info.broadcast_info;
  auto &data_repeats = temp_data_attr.repeats;
  auto &data_strides = temp_data_attr.strides;
  const auto &data_axis = temp_data_attr.axis;
  const auto &load_repeats = temp_load_attr.repeats;

  for (auto index = 0U; index < data_repeats.size(); index++) {
    if (BackendUtils::IsEqOne(data_repeats[index]) && (BackendUtils::IsEqZero(data_strides[index]))) {
      const auto axis_id = data_axis[index];
      if (BackendUtils::IsEqOne(load_repeats[index])) {
        continue;
      }
      broadcast_info.push_back(axis_id);
    }
  }
  GELOGD("back stepping view op broadcast axis id:%s.",
         AutofuseUtils::VectorToStr(attr_info.broadcast_info).c_str());
  return SUCCESS;
}

Status BackendUtils::FusedBackSteppingViewOpTranspose(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                                      ViewOpAttrInfo &attr_info) {
  // 使用graph轴和load轴计算转置信息
  auto &transpose_info = attr_info.transpose_info;
  const auto &graph_axis = temp_graph_attr.axis;
  const auto &load_axis = temp_load_attr.axis;

  // 计算将load轴变为graph轴所需的最小交换次数，并记录每次交换的axis id
  int64_t swap_count = 0;
  GE_ASSERT_SUCCESS(MinSwapCount(graph_axis, load_axis, swap_count, transpose_info));
  GELOGD("Fused back stepping view op transpose axis id: %s.",
         AutofuseUtils::VectorPairToStr(attr_info.transpose_info).c_str());
  return SUCCESS;
}

Status BackendUtils::FusedBackSteppingViewOpBroadcast(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                                      ViewOpAttrInfo &attr_info) {
  // 1.反推出 broadcast_info
  auto &broadcast_info = attr_info.broadcast_info;
  const auto &load_axis = temp_load_attr.axis;
  const auto &load_repeats = temp_load_attr.repeats;

  GE_ASSERT_TRUE(temp_graph_attr.axis.size() <= load_repeats.size());
  for (auto index = 0U; index < load_repeats.size(); index++) {
    if (!BackendUtils::IsEqOne(load_repeats[index])) { // slice的load可能出现repeat是1，stride是512，同时跟graph非1的size相比是需要broadcast的
      continue;
    }
    const auto axis_id = load_axis[index];
    for (auto graph_index = 0U; graph_index < temp_graph_attr.axis.size(); graph_index++) {
      if ((temp_graph_attr.axis[graph_index] == axis_id) && (!BackendUtils::IsEqOne(temp_graph_attr.repeats[graph_index]))) {
        broadcast_info.push_back(axis_id);
      }
    }
  }
  if (!broadcast_info.empty()) {
    GELOGD("fused back stepping view op broadcast axis id:%s.",
           AutofuseUtils::VectorToStr(attr_info.broadcast_info).c_str());
  }
  return SUCCESS;
}

// 循环分解检测法 计算将 in_axis 变为 out_axis 所需的最小交换次数，并记录每次交换的axis id
Status BackendUtils::MinSwapCount(const std::vector<int64_t> &in_axis, const std::vector<int64_t> &out_axis,
                                  int64_t &swap_count, std::vector<std::pair<int64_t, int64_t>> &swaps) {
  if (in_axis.size() != out_axis.size()) {
    return SUCCESS;
  }
  // 检查 in_axis 和 out_axis 的元素集合是否一致,顺序不必一致
  std::unordered_set<int> in_set(in_axis.begin(), in_axis.end());
  std::unordered_set<int> out_set(out_axis.begin(), out_axis.end());
  GE_ASSERT_TRUE(in_set == out_set);

  std::unordered_map<int64_t, size_t> out_index_map;
  for (size_t i = 0U; i < out_axis.size(); ++i) {
    out_index_map[out_axis[i]] = i;
  }
  std::vector<size_t> in_sorted_indices(in_axis.size());
  for (size_t i = 0U; i < in_axis.size(); ++i) {
    in_sorted_indices[i] = out_index_map[in_axis[i]];
  }

  // 计算最小交换次数，并记录每次交换的axis id
  std::vector<bool> visited(in_axis.size(), false);
  for (size_t i = 0U; i < in_sorted_indices.size(); ++i) {
    // 如果当前元素已经在正确的位置，或者已经被访问过，跳过
    if (visited[i] || in_sorted_indices[i] == i) {
      continue;
    }
    size_t cycle_size = 0U;
    size_t j = i;
    while (!visited[j]) {
      visited[j] = true;
      j = in_sorted_indices[j];
      cycle_size++;
    }
    // 每个循环需要的交换次数为 cycle_size - 1
    GE_ASSERT_TRUE(cycle_size > 0U);
    swap_count += (cycle_size - 1U);

    j = i;
    while (cycle_size > 1U) {
      size_t next_j = in_sorted_indices[j];
      swaps.emplace_back(in_axis[j], in_axis[next_j]);
      j = next_j;
      cycle_size--;
    }
  }
  return SUCCESS;
}

Status BackendUtils::ApplySwap(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                               std::vector<ge::Expression> &strides, std::vector<int64_t> &sched_axis,
                               const std::pair<int64_t, int64_t> &swap) {
  int64_t axis1 = swap.first;
  int64_t axis2 = swap.second;
  auto it1 = std::find(axis.begin(), axis.end(), axis1);
  auto it2 = std::find(axis.begin(), axis.end(), axis2);
  if (it1 != axis.end() && it2 != axis.end()) {
    std::swap(*it1, *it2);
    size_t index1 = std::distance(axis.begin(), it1);
    size_t index2 = std::distance(axis.begin(), it2);
    std::swap(repeats[index1], repeats[index2]);
    std::swap(strides[index1], strides[index2]);
    std::swap(sched_axis[index1], sched_axis[index2]);
  } else {
    return FAILED;
  }
  return SUCCESS;
}

Status BackendUtils::ApplySwaps(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                                std::vector<ge::Expression> &strides, std::vector<int64_t> &sched_axis,
                                const std::vector<std::pair<int64_t, int64_t>> &swaps) {
  // 遍历 swaps，依次交换 axis 和 repeats 中的元素
  for (const auto &swap : swaps) {
    GE_ASSERT_SUCCESS(ApplySwap(axis, repeats, strides, sched_axis, swap));
  }
  return SUCCESS;
}

// 根据 swaps 处理 data_axis 和 repeats，获取交换后的 data_axis 和 repeats
Status BackendUtils::ApplySwaps(TensorAttrInfo &temp_data_attr, const std::vector<std::pair<int64_t, int64_t>> &swaps) {
  GE_ASSERT_SUCCESS(ApplySwaps(temp_data_attr.axis, temp_data_attr.repeats, temp_data_attr.strides,
                               temp_data_attr.sched_axis, swaps));
  // strides需要根据repeats重新计算,然后再根据repeats为1的,把对应的strides刷为0,当前在补轴流程做
  return SUCCESS;
}

// Transpose 反推处理函数
Status BackendUtils::PostProBackSteppingViewOpTranspose(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                                        ViewOpAttrInfo &attr_info, const std::string &cur_node_type) {
  auto &transpose_info = attr_info.transpose_info;
  const auto &graph_axis = temp_graph_attr.axis;
  auto &cur_axis = temp_load_attr.axis;
  // 反推出 transpose_info 并 根据反推结果还原临时 attr_info ，为下一次反推处理准备attr_info
  int64_t swap_count = 0;
  if ((cur_node_type == kLoadType) || (cur_node_type == kGatherType)) {
    GE_ASSERT_SUCCESS(MinSwapCount(cur_axis, graph_axis, swap_count, transpose_info));
  } else {  // if is store
    GE_ASSERT_SUCCESS(MinSwapCount(graph_axis, cur_axis, swap_count, transpose_info));
  }
  GELOGD("post process back stepping view op transpose axis id:%s.",
         AutofuseUtils::VectorPairToStr(attr_info.transpose_info).c_str());
  return SUCCESS;
}

// 从无序的数组中获取交换轴信息（最少次数交换）
Status BackendUtils:: MinSwapsToSortDesc(const vector<int64_t> &arr, std::vector<std::pair<int64_t, int64_t>> &swaps) {
  vector<int64_t> non_zero_values;
  vector<int64_t> non_zero_indices;
  // 存在0的时候
  for (size_t i = 0U; i < arr.size(); ++i) {
    if (arr[i] != 0) {
      non_zero_values.push_back(arr[i]);
      non_zero_indices.push_back(static_cast<int64_t>(i));
    }
  }

  // 对非零元素进行降序排序，记录交换的索引对
  for (size_t i = 0U; i < non_zero_values.size(); ++i) {
    size_t max_index = i;
    for (size_t j = i + 1U; j < non_zero_values.size(); ++j) {
      if (non_zero_values[j] > non_zero_values[max_index]) {
        max_index = j;
      }
    }
    if (max_index != i) {
      // 记录交换的原始索引
      swaps.emplace_back(non_zero_indices[i], non_zero_indices[max_index]);
      // 交换非零值
      std::swap(non_zero_values[i], non_zero_values[max_index]);
    }
  }
  return SUCCESS;
}

// Ge流程的Transpose 反推处理函数
Status BackendUtils::BackSteppingViewOpTranspose(const TensorAttrInfo &temp_data_attr, ViewOpAttrInfo &attr_info) {
  auto &transpose_info = attr_info.transpose_info;
  auto &data_strides = temp_data_attr.strides;
  GELOGD("before apply, temp data attr axis:%s repeats:%s strides:%s.",
         AutofuseUtils::VectorToStr(temp_data_attr.axis).c_str(),
         AutofuseUtils::VectorToStr(temp_data_attr.repeats).c_str(),
         AutofuseUtils::VectorToStr(temp_data_attr.strides).c_str());
  vector<int64_t> data_strides_hint(data_strides.size(), 0);
  for (size_t i = 0U; i < data_strides.size(); ++i) {
    data_strides[i].GetHint(data_strides_hint[i]);
  }
  // 获取交换轴信息
  GE_ASSERT_SUCCESS(MinSwapsToSortDesc(data_strides_hint, transpose_info));
  GELOGD("Ge back stepping view op transpose axis id:%s.",
         AutofuseUtils::VectorPairToStr(attr_info.transpose_info).c_str());
  return SUCCESS;
}

// Slice 融合过程中的反推处理函数
Status SliceGetNodeOffset(const NodePtr &load_node, Expression &load_offset, bool &slice_op_flag) {
  GE_ASSERT_NOTNULL(load_node->GetOpDesc());
  const auto &attr = load_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr);
  auto load_attr = attr->ir_attr->DownCastTo<ascir_op::Load::AscLoadIrAttrDef>();
  GE_ASSERT_NOTNULL(load_attr);
  if (load_attr->GetOffset(load_offset) != SUCCESS) {
    slice_op_flag = false;
    return SUCCESS;
  }
  slice_op_flag = true;
  return SUCCESS;
}

AscTensorAttr *SliceGetDataPeerOutNodeAttr(const NodePtr &node2, const NodePtr &data_node) {
  const auto data_op_desc = data_node->GetOpDesc();
  auto data_node_attr = data_op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(data_node_attr);
  int32_t ir_idx = -1;
  GE_ASSERT_NOTNULL(data_node_attr->ir_attr);
  (void)data_node_attr->ir_attr->GetAttrValue("index", ir_idx);
  GE_ASSERT_TRUE(ir_idx >= 0, "Get invalid ir_index from node :[%s].", data_op_desc->GetNamePtr());

  auto in_data_anchor = node2->GetInDataAnchor(ir_idx);
  GE_ASSERT_NOTNULL(in_data_anchor);
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  auto peer_out_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_out_node);

  ComputeGraphPtr origin_graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_out_node, origin_graph));
  GELOGI("AscBackendGraphFuse: before fuse origin graph: %s, node: %s(%s)).",
         origin_graph->GetName().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
  OpDescPtr cur_op_desc = nullptr;
  for (const auto &inner_node : origin_graph->GetAllNodes()) {
    GELOGI("inner_node->GetType(): %s.", inner_node->GetType().c_str());
    if ((inner_node->GetType() == kOutputType) || (inner_node->GetType() == kNetOutputType)) {
      int64_t index = -1;
      const auto op_desc_temp = inner_node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc_temp);
      auto node_attr_temp = op_desc_temp->GetAttrsGroup<AscNodeAttr>();
      GE_ASSERT_NOTNULL(node_attr_temp);
      GE_ASSERT_NOTNULL(data_node_attr->ir_attr);
      (void)data_node_attr->ir_attr->GetAttrValue("index", index);
      GE_ASSERT_TRUE(index >= 0, "Get invalid ir_index from node :[%s].", op_desc_temp->GetNamePtr());
      if (index == ir_idx) {
        cur_op_desc = inner_node->GetOpDesc();
        break;
      }
    }
  }
  GE_ASSERT_NOTNULL(cur_op_desc);
  const auto cur_output_desc = cur_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(cur_output_desc);
  auto cur_output_attr = cur_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(cur_output_attr); 
  return cur_output_attr;
}

Status SliceCalcOffsetAndStridePerAxisPreNode(TensorAttrInfo &temp_load_attr, ViewOpAttrInfo &attr_info,
                                              Expression &offset) {
  auto &slice_info = attr_info.slice_info;
  auto &load_strides = temp_load_attr.strides;

  // 计算对应每根轴的原始stride
  for (const auto &load_stride : load_strides) {
      slice_info.pre_load_strides.push_back(load_stride);
  }
  GELOGD("slice info pre load strides: %s.", AutofuseUtils::VectorToStr(slice_info.pre_load_strides).c_str());
  slice_info.pre_load_offsets.push_back(offset);
  return SUCCESS;
}

Status SliceCalcOffsetAndStridePerAxisCurNode(TensorAttrInfo &temp_load_attr, TensorAttrInfo &temp_pre_out_data_attr,
                                              ViewOpAttrInfo &attr_info, Expression &offset) {
  auto &slice_info = attr_info.slice_info;
  auto &peer_output_data_repeats = temp_pre_out_data_attr.repeats;
  auto &peer_output_data_strides = temp_pre_out_data_attr.strides;
  auto &load_strides = temp_load_attr.strides;
  Expression strides_temp;

  // 计算对应每根轴的原始stride
  for (auto index = 0U; index < peer_output_data_repeats.size(); index++) {
      strides_temp = BackendUtils::IsEqOne(peer_output_data_repeats[index]) ? kSymbolZero
                   : (ge::sym::Floor(load_strides[index] / peer_output_data_strides[index])).Simplify();
      slice_info.load_strides.push_back(strides_temp);
  }
  GELOGD("slice info load strides: %s.", AutofuseUtils::VectorToStr(slice_info.load_strides).c_str());

  Expression offset_remain = offset;
  Expression offset_temp;
  for (auto index = 0U; index < peer_output_data_repeats.size(); index++) {
    offset_temp =  BackendUtils::IsEqOne(peer_output_data_repeats[index]) ? kSymbolZero
                   : (ge::sym::Floor(offset_remain / peer_output_data_strides[index])).Simplify();
    offset_remain = (offset_remain - offset_temp * peer_output_data_strides[index]).Simplify();
    slice_info.load_offsets.push_back(offset_temp);
  }
  GELOGD("slice_info.load_offsets: %s.", AutofuseUtils::VectorToStr(slice_info.load_offsets).c_str());
  return SUCCESS;
}

Status SlicePreLoadDataNodePro(TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr,
                               ViewOpAttrInfo &attr_info, Expression &offset) {
  auto &slice_info = attr_info.slice_info;
  auto &data_repeats = temp_data_attr.repeats;
  auto &data_strides = temp_data_attr.strides;
  const auto &load_rstrides = temp_load_attr.strides;

  for (auto index = 0U; index < data_repeats.size(); index++) {
    slice_info.pre_data_repeat_save.push_back(data_repeats[index]);
    slice_info.pre_data_strides_save.push_back(data_strides[index]);
    slice_info.pre_load_strides_save.push_back(load_rstrides[index]);
  }
  GE_ASSERT_SUCCESS(SliceCalcOffsetAndStridePerAxisPreNode(temp_load_attr, attr_info, offset));
  return SUCCESS;
}

Status ApplyViewOpFinalInfoSlice(const TensorAttrInfo &temp_data_attr, ViewOpAttrInfo &attr_info) {
  auto &slice_info = attr_info.slice_info;
  auto &data_repeats = temp_data_attr.repeats;
  Expression offset_orign = slice_info.pre_load_offsets[0];
  Expression strides_temp;
  for (auto index = 0U; index < data_repeats.size(); index++) {
    strides_temp = ((slice_info.load_strides[index] * slice_info.pre_load_strides[index])).Simplify();
    slice_info.backend_strides.push_back(strides_temp);
    offset_orign = (offset_orign + slice_info.load_offsets[index] * slice_info.pre_load_strides_save[index]).Simplify();
  }
  slice_info.offset_backend = offset_orign;
  return SUCCESS;
}

std::vector<Expression> ContiguousStrides(const std::vector<Expression> &dims) {
  if (dims.empty()) {
    return {};
  }
  std::vector<Expression> strides(dims.size(), kSymbolOne);
  for (size_t i = dims.size() - 1U; i > 0U; --i) {
    strides[i - 1] = strides[i] * dims[i];
  }
  for (size_t i = 0U; i < dims.size(); ++i) {
    if (BackendUtils::IsEqOne(dims[i]) || BackendUtils::IsEqZero(dims[i])) {
      strides[i] = kSymbolZero;  // 保证dim size为0或1时，stride必定为0，简化stride判定规则，
    }
    strides[i] = strides[i].Simplify();
  }
  return strides;
}

bool IsNeedCheckLoadSame(const NodePtr &node1, const NodePtr &node2,
                         std::vector<ViewOpAttrInfo> &attr_infos1, std::vector<ViewOpAttrInfo> &attr_infos2) {
    /* 如果都不为0，需要判断这两个load上是不是都是slice，如果不是，也不进行任何操作 */
    bool attr_infos1_all_is_slice = true;
    bool attr_infos2_all_is_slice = true;
    for (auto index1 = 0U; index1 < attr_infos1.size(); index1++) {
      if (attr_infos1[index1].slice_info.two_slice_node_flag != true) {
        attr_infos1_all_is_slice = false;
        break;
      }
    }

    for (auto index2 = 0U; index2 < attr_infos2.size(); index2++) {
      if (attr_infos2[index2].slice_info.two_slice_node_flag != true) {
        attr_infos2_all_is_slice = false;
        break;
      }
    }
    GELOGD("attr size: node1 %s(size is %d). node2 %s(size is %d).", node1->GetNamePtr(), attr_infos1.size(), node2->GetNamePtr(), attr_infos2.size());
    GELOGD("slice flag : node1 %s(flag is %d). node2 %s(flag is %d).", node1->GetNamePtr(), attr_infos1_all_is_slice, node2->GetNamePtr(), attr_infos2_all_is_slice);
    if ((((attr_infos1_all_is_slice == true) && (attr_infos1.size() == 1U))
         && ((attr_infos2_all_is_slice == true) && (attr_infos2.size() == 1U)))) {
      return true;
    } 
    return false;
}

bool BackendUtils::SliceHasSameLoad(const NodePtr &node1, const NodePtr &node2,
                                    std::vector<std::pair<int32_t, int32_t>> &same_input_map_) {
  bool slice_has_nosame_load_flag = false;
  for (const auto &same_input : same_input_map_) {
    std::vector<ViewOpAttrInfo> attr_infos1, attr_infos2;
    BackendUtils::CurNodeInputIsSimplestLoad(node1, same_input.first, attr_infos1);
    BackendUtils::CurNodeInputIsSimplestLoad(node2, same_input.second, attr_infos2);
    if ((attr_infos1.size() == 0U) || (attr_infos2.size() == 0U)) {
      continue;
    }
    if (IsNeedCheckLoadSame(node1, node2, attr_infos1, attr_infos2) != true) {
      break;
    }

    for (auto index = 0U; index < attr_infos1.size(); index++) {
      auto &node1_slice_info = attr_infos1[index].slice_info;
      auto &node2_slice_info = attr_infos2[index].slice_info;
      
      GE_ASSERT_TRUE((node1_slice_info.pre_load_offsets.size() != 0) && (node2_slice_info.pre_load_offsets.size() != 0));
      if (node1_slice_info.pre_load_offsets[0] != node2_slice_info.pre_load_offsets[0]) {
        GELOGD("two slice offset is not equal: node1 %s(%s). node2 %s(%s).", 
                node1->GetNamePtr(), node1_slice_info.pre_load_offsets[0].Serialize().get(),
                node2->GetNamePtr(), node2_slice_info.pre_load_offsets[0].Serialize().get()); 
        slice_has_nosame_load_flag = true;
        break;
      }
      bool not_equal_flag = false;
      for (auto dim_index = 0U; dim_index < node1_slice_info.pre_load_strides_save.size(); dim_index++) {
        if ((node1_slice_info.pre_load_strides_save[dim_index] != node2_slice_info.pre_load_strides_save[dim_index]) || (node1_slice_info.pre_data_repeat_save[dim_index] != node2_slice_info.pre_data_repeat_save[dim_index])) {
          GELOGD("two slice stride or repeat not equal: node1 %s, node2 %s, stride or repeat index is %u.", 
                 node1->GetNamePtr(), node2->GetNamePtr(), dim_index);
          not_equal_flag = true;
          break;
        }
      }
      if (not_equal_flag == true) {
        slice_has_nosame_load_flag = true;
        break;         
      }
    }
  }
  return slice_has_nosame_load_flag;
}

bool IsSameStride(std::vector<ge::Expression> &strides, std::vector<ge::Expression> &compare_strides) {
  for (size_t index = 0U; index < compare_strides.size(); index++) {
    if (strides[index] != compare_strides[index]) {
      GELOGD("pre node and cur node are both slice.");
      return true;
    }
  }
  return false;
}

bool IsOffsetZero(const NodePtr &load_node) {
  bool node_slice_op_flag = false;
  Expression pre_load_offset;
  GE_ASSERT_SUCCESS(SliceGetNodeOffset(load_node, pre_load_offset, node_slice_op_flag));
  // 使用直接比较而不是 BackendUtils::IsEqZero，原因：
  // 1. IsEqZero 内部调用 StaticCheckEq -> Simplify() -> SymEngine::expand()
  // 2. SplitV 等算子的 offset 表达式可能是复杂的 SymEngine 表达式
  // 3. 某些情况下 pre_load_offset 的内部 sym_expr_ 可能为空指针
  // 4. SymEngine::expand() 在遇到空指针时会调用虚函数导致段错误
  if ((node_slice_op_flag == true) && (pre_load_offset != kSymbolZero)) {
    return true;
  }
  return false;
}

bool IsSlice(const NodePtr &load_node,
             TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr, ViewOpAttrInfo &attr_info,
             bool is_fuse) {
  auto &load_repeats = temp_load_attr.repeats;
  auto &data_repeats = temp_data_attr.repeats;
  auto load_strides = temp_load_attr.strides;
  auto data_strides = temp_data_attr.strides;
  GELOGD("temp load attr repeats:%s. temp data attr repeats:%s. load strides:%s. data strides:%s.", 
         AutofuseUtils::VectorToStr(load_repeats).c_str(), AutofuseUtils::VectorToStr(data_repeats).c_str(),
         AutofuseUtils::VectorToStr(load_strides).c_str(), AutofuseUtils::VectorToStr(data_strides).c_str());
  if((!attr_info.broadcast_info.empty()) || (!attr_info.transpose_info.empty())) {
    return false;
  }
  std::vector<ge::Expression> strides = ContiguousStrides(data_repeats);
  std::vector<ge::Expression> compare_strides = data_strides;
  if (is_fuse) {
    strides = ContiguousStrides(load_repeats);
    compare_strides = load_strides;
  }
  GELOGD("contiguous strides:%s.", AutofuseUtils::VectorToStr(strides).c_str());
  auto stride_same_flag = IsSameStride(strides, compare_strides);
  auto offset_zero_flag = IsOffsetZero(load_node);

  return ((stride_same_flag == true) || (offset_zero_flag == true));
}

Status BackendUtils::BackSteppingViewOpSlice(TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr,
                                             const NodePtr &load_node, ViewOpAttrInfo &attr_info, bool is_fuse,
                                             const bool is_merge_check) {
  bool pre_node_slice_op_flag = false;
  Expression pre_load_offset;
  attr_info.slice_info.two_slice_node_flag = false;
  GELOGD("before apply, temp data attr axis:%s repeats:%s strides:%s.",
         AutofuseUtils::VectorToStr(temp_data_attr.axis).c_str(),
         AutofuseUtils::VectorToStr(temp_data_attr.repeats).c_str(),
         AutofuseUtils::VectorToStr(temp_data_attr.strides).c_str());

  pre_node_slice_op_flag = IsSlice(load_node, temp_data_attr, temp_load_attr, attr_info, is_fuse);
  if (!pre_node_slice_op_flag) {
    return SUCCESS;
  }

  attr_info.slice_info.two_slice_node_flag = true;
  if (is_fuse) {
    GE_ASSERT_SUCCESS(SliceGetNodeOffset(load_node, pre_load_offset, pre_node_slice_op_flag));
    GELOGD("pre load offset is:%s. load node name is:%s.", pre_load_offset.Serialize().get(),
          load_node->GetName().c_str());
    GE_ASSERT_SUCCESS(SlicePreLoadDataNodePro(temp_data_attr, temp_load_attr, attr_info, pre_load_offset));
  }

  if (is_merge_check) {
    GELOGD("Is merge check, not change node attr.");
    return SUCCESS;
  }
  temp_load_attr.strides = temp_data_attr.strides;
  auto op_desc = load_node->GetOpDesc();
  AscNodeAttr *node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);
  auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  for (size_t i = 0U; i < temp_data_attr.strides.size(); i++) {
    if (!BackendUtils::IsEqZero(output_attr->strides[i])) {
      output_attr->strides[i] = temp_data_attr.strides[i];
    }
  }
  GELOGD("after change stride, output_attr attr axis:%s repeats:%s strides:%s.",
         AutofuseUtils::VectorToStr(output_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(output_attr->strides).c_str());
  return SUCCESS;
}

// Broadcast apply处理函数
Status BackendUtils::FusedApplyViewOpBroadcast(const AscNodeAttr *node_attr, AscTensorAttr *output_attr,
                                               ViewOpAttrInfo &attr_info) {
  auto &broadcast_info = attr_info.broadcast_info;
  for (const auto &broadcast_axis : broadcast_info) {
    auto it = std::find(node_attr->sched.axis.begin(), node_attr->sched.axis.end(), broadcast_axis);
    GE_ASSERT_TRUE(it != node_attr->sched.axis.end());
    int32_t index = std::distance(node_attr->sched.axis.begin(), it);
    output_attr->repeats[index] = kSymbolOne;
    output_attr->strides[index] = kSymbolZero;
  }
  if (!attr_info.broadcast_info.empty()) {
    output_attr->strides = ContiguousStrides(output_attr->repeats);  // slice和broadcast不共存于同一节点，可刷load strides
  }
  return SUCCESS;
}

Status BackendUtils::ApplySwaps(const NodePtr &node, const std::vector<std::pair<int64_t, int64_t>> &swaps) {
  // 仅处理Tensor的axis、repeats、strides，不处理sched axis
  auto op_desc = node->GetOpDesc();
  auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  auto &axis = output_attr->axis;
  auto &repeats = output_attr->repeats;
  auto &strides = output_attr->strides;
  for (const auto &swap : swaps) {
    int64_t axis1 = swap.first;
    int64_t axis2 = swap.second;
    auto it1 = std::find(axis.begin(), axis.end(), axis1);
    auto it2 = std::find(axis.begin(), axis.end(), axis2);
    GE_ASSERT_TRUE(it1 != axis.end() && it2 != axis.end());
    std::swap(*it1, *it2);
    size_t index1 = std::distance(axis.begin(), it1);
    size_t index2 = std::distance(axis.begin(), it2);
    std::swap(repeats[index1], repeats[index2]);
    std::swap(strides[index1], strides[index2]);
  }
  output_attr->strides = ContiguousStrides(output_attr->repeats);
  GELOGD("after Apply Transpose, node name:(%s) output_attr axis:%s repeats:%s strides:%s.",
         node->GetName().c_str(),
         AutofuseUtils::VectorToStr(axis).c_str(),
         AutofuseUtils::VectorToStr(repeats).c_str(),
         AutofuseUtils::VectorToStr(strides).c_str());
  return SUCCESS;
}

// Transpose apply处理函数
Status BackendUtils::FusedApplyViewOpTranspose(const NodePtr &data_node, const NodePtr &load_node,
                                               ViewOpAttrInfo &attr_info) {
  auto transpose_info = attr_info.transpose_info;
  if (!transpose_info.empty()) {
    GE_ASSERT_SUCCESS(ApplySwaps(data_node, transpose_info));
    GE_ASSERT_SUCCESS(ApplySwaps(load_node, transpose_info));
  }
  return SUCCESS;
}

Status GetNodeAttr(const NodePtr &data_node, const NodePtr &load_node, TensorAttrInfo &temp_data_attr,
                   TensorAttrInfo &temp_load_attr) {
  const auto data_op_desc = data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(data_op_desc);
  const auto data_output_desc = data_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(data_output_desc);
  const auto data_output_attr = data_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(data_output_attr);
  GE_ASSERT_TRUE(data_output_attr->axis.size() == data_output_attr->repeats.size());
  AscNodeAttr *data_node_attr = data_op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(data_node_attr);

  const auto load_op_desc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_op_desc);
  const auto load_output_desc = load_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(load_output_desc);
  const auto load_output_attr = load_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(load_output_attr);
  GE_ASSERT_TRUE(load_output_attr->axis.size() == load_output_attr->repeats.size());
  AscNodeAttr *load_node_attr = load_op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(load_node_attr);

  temp_data_attr.sched_axis = data_node_attr->sched.axis;
  temp_data_attr.axis = data_output_attr->axis;
  temp_data_attr.repeats = data_output_attr->repeats;
  temp_data_attr.strides = data_output_attr->strides;
  temp_load_attr.sched_axis = load_node_attr->sched.axis;
  temp_load_attr.axis = load_output_attr->axis;
  temp_load_attr.repeats = load_output_attr->repeats;
  temp_load_attr.strides = load_output_attr->strides;

  GELOGD(
      "before back step, data node %s(%s), data_output_attr repeats:%s strides:%s axis:%s, sched axis:%s."
      "load node %s(%s), load_output_attr repeats:%s strides:%s axis:%s, sched axis:%s.",
      data_node->GetNamePtr(), data_node->GetType().c_str(),
      AutofuseUtils::VectorToStr(data_output_attr->repeats).c_str(),
      AutofuseUtils::VectorToStr(data_output_attr->strides).c_str(),
      AutofuseUtils::VectorToStr(data_output_attr->axis).c_str(),
      AutofuseUtils::VectorToStr(data_node_attr->sched.axis).c_str(), load_node->GetNamePtr(),
      load_node->GetType().c_str(), AutofuseUtils::VectorToStr(load_output_attr->repeats).c_str(),
      AutofuseUtils::VectorToStr(load_output_attr->strides).c_str(),
      AutofuseUtils::VectorToStr(load_output_attr->axis).c_str(),
      AutofuseUtils::VectorToStr(load_node_attr->sched.axis).c_str());
  return SUCCESS;
}

// slice的load apply view的处理
Status ApplyViewOpSliceLoad(const NodePtr &load_node, ViewOpAttrInfo &attr_info) {
  auto op_desc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  AscNodeAttr *node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);
  auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  GE_ASSERT_TRUE(node_attr->sched.axis.size() == output_attr->repeats.size());

  auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(node_attr->ir_attr.get());
  GE_ASSERT_NOTNULL(load_attr);
  Expression offset_origin;
  (void)load_attr->GetOffset(offset_origin);
  GELOGD("slice load node apply view op before, node %s(%s) output repeats:%s strides:%s offset:%s.",
         load_node->GetNamePtr(), load_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(output_attr->strides).c_str(), offset_origin.Serialize().get());

  auto &slice_info = attr_info.slice_info;
  for (auto index = 0U; index < slice_info.backend_strides.size(); index++) {
    output_attr->strides[index] = slice_info.backend_strides[index];
  }

  GE_ASSERT_SUCCESS(load_attr->SetOffset(slice_info.offset_backend));
  GELOGD("slice load node apply view op after, node %s(%s) output repeats:%s strides:%s offset:%s.",
         load_node->GetNamePtr(), load_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(output_attr->strides).c_str(), slice_info.offset_backend.Serialize().get());
  return SUCCESS;
}

// 融合过程中Slice apply处理函数,只用load做推导
Status BackendUtils::FusedApplyViewOpSlice(AscTensorAttr *output_attr, const NodePtr &data_node,
                                           const NodePtr &load_node, ViewOpAttrInfo &attr_info, const NodePtr &node2) {
  bool curr_node_slice_op_flag = false;
  Expression curr_load_offset;
  if (!attr_info.slice_info.two_slice_node_flag) {
    return SUCCESS;
  }
  // 将原始的 axis、repeats、strides 保存到临时变量, 每一次反推后修改temp_data_attr后，把temp_data_attr给下一次反推用
  TensorAttrInfo temp_data_attr;
  TensorAttrInfo temp_load_attr;
  GE_ASSERT_SUCCESS(GetNodeAttr(data_node, load_node, temp_data_attr, temp_load_attr));
  curr_node_slice_op_flag = IsSlice(load_node, temp_data_attr, temp_load_attr, attr_info, true);
  if (!curr_node_slice_op_flag) {
    return SUCCESS;
  }

  TensorAttrInfo temp_pre_out_data_attr;
  AscTensorAttr *pre_output_attr = SliceGetDataPeerOutNodeAttr(node2, data_node);
  GE_ASSERT_NOTNULL(pre_output_attr);
  temp_pre_out_data_attr.axis = pre_output_attr->axis;
  temp_pre_out_data_attr.repeats = pre_output_attr->repeats;
  temp_pre_out_data_attr.strides = pre_output_attr->strides;

  GE_ASSERT_SUCCESS(SliceGetNodeOffset(load_node, curr_load_offset, curr_node_slice_op_flag));
  GE_ASSERT_SUCCESS(SliceCalcOffsetAndStridePerAxisCurNode(temp_load_attr, temp_pre_out_data_attr, attr_info,
                                                           curr_load_offset));
  // 计算最终需要更新的offset以及stride
  GE_ASSERT_SUCCESS(ApplyViewOpFinalInfoSlice(temp_data_attr, attr_info));
  // 需要将pre_data的节点信息更新到data节点的输出属性上
  auto &slice_info = attr_info.slice_info;
  for (auto index = 0U; index < slice_info.pre_data_repeat_save.size(); index++) {
    output_attr->repeats[index] = slice_info.pre_data_repeat_save[index];
    output_attr->strides[index] = slice_info.pre_data_strides_save[index];
  }

  // 将前面反推出来的offset以及stride更新到load节点上信息更新到当前节点的信息上
  GE_ASSERT_SUCCESS(ApplyViewOpSliceLoad(load_node, attr_info));
  return SUCCESS;
}

// GE流程反推，只需要反推出broadcast和tranpose
Status BackendUtils::BackSteppingViewOpPro(TensorAttrInfo &temp_data_attr, TensorAttrInfo &temp_load_attr,
                                           ViewOpAttrInfo &attr_info, bool just_broadcast) {
  GE_ASSERT_SUCCESS(BackSteppingViewOpBroadcast(temp_data_attr, temp_load_attr, attr_info));
  if (!just_broadcast) {
    GE_ASSERT_SUCCESS(BackSteppingViewOpTranspose(temp_data_attr, attr_info));
  }
  return SUCCESS;
}

Status BackendUtils::FusedApplyViewOpPro(const AscNodeAttr *node_attr, AscTensorAttr *output_attr, const NodePtr &data_node,
                                         const NodePtr &load_node, ViewOpAttrInfo &attr_info, const NodePtr &node2) {
  GE_ASSERT_SUCCESS(FusedApplyViewOpTranspose(data_node, load_node, attr_info));
  GE_ASSERT_SUCCESS(FusedApplyViewOpBroadcast(node_attr, output_attr, attr_info));
  GE_ASSERT_SUCCESS(FusedApplyViewOpSlice(output_attr, data_node, load_node, attr_info, node2));
  return SUCCESS;
}

// 融合后的还原
Status BackendUtils::PostProBackSteppingViewOpPro(TensorAttrInfo &temp_graph_attr, TensorAttrInfo &temp_load_attr,
                                                  ViewOpAttrInfo &attr_info, bool is_back_broadcast,
                                                  const std::string &cur_node_type) {
  if (is_back_broadcast) {
    GE_ASSERT_SUCCESS(FusedBackSteppingViewOpBroadcast(temp_graph_attr, temp_load_attr, attr_info));
  }
  return SUCCESS; // 后端只支持一个ascgraph里面有一个transpose,后处理transpose反推流程暂时不走了，走transpose后移流程来插入transpose节点
  GE_ASSERT_SUCCESS(PostProBackSteppingViewOpTranspose(temp_graph_attr, temp_load_attr, attr_info, cur_node_type));
  return SUCCESS;
}

Status BackendUtils::PostProBackSteppingViewOp(AscGraph &asc_graph, const NodePtr &cur_node, ViewOpAttrInfo &attr_info,
                                               bool is_back_broadcast) {
  const auto cur_op_desc = cur_node->GetOpDesc();
  const auto cur_output_desc = cur_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(cur_output_desc);
  const auto cur_output_attr = cur_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(cur_output_attr);
  // 考虑到后续增加transpose反推，data和load的axis可能不一致，删除data和load的axis一致校验
  GE_ASSERT_TRUE(cur_output_attr->axis.size() == cur_output_attr->repeats.size());
  AscNodeAttr *cur_node_attr = cur_op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(cur_node_attr);

  // 将原始的 axis保存到临时变量, 每一次反推后修改temp_data_attr后，把temp_data_attr给下一次反推用
  TensorAttrInfo temp_graph_attr;
  TensorAttrInfo temp_cur_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, temp_graph_attr));
  // 调用对应的处理函数前,需要把load或者store的轴补齐
  cur_node_attr->sched.axis = temp_graph_attr.axis;
  GE_ASSERT_SUCCESS(
      asc_adapt::UpdateTensorAttrsIfNotEmpty(cur_node, temp_graph_attr.axis, temp_graph_attr.repeats, cur_output_attr));

  temp_cur_attr.sched_axis = cur_node_attr->sched.axis;
  temp_cur_attr.axis = cur_output_attr->axis;
  temp_cur_attr.repeats = cur_output_attr->repeats;
  temp_cur_attr.strides = cur_output_attr->strides;
  GELOGD(
      "post process before back step, graph %s, axis:%s."
      "node %s(%s), output_attr repeats:%s strides:%s axis:%s, sched axis:%s.",
      asc_graph.GetName().c_str(),
      AutofuseUtils::VectorToStr(temp_graph_attr.axis).c_str(), cur_node->GetNamePtr(),
      cur_node->GetType().c_str(), AutofuseUtils::VectorToStr(cur_output_attr->repeats).c_str(),
      AutofuseUtils::VectorToStr(cur_output_attr->strides).c_str(),
      AutofuseUtils::VectorToStr(cur_output_attr->axis).c_str(),
      AutofuseUtils::VectorToStr(cur_node_attr->sched.axis).c_str());

  auto cur_node_type = cur_node->GetType();
  GE_ASSERT_SUCCESS(PostProBackSteppingViewOpPro(temp_graph_attr, temp_cur_attr, attr_info, is_back_broadcast, cur_node_type));
  return SUCCESS;
}

Status BackendUtils::UpdateBroadcastInfoToLoad(AscGraph &asc_graph) {
  // 融合过程中的反推是需要删除broadcast，并且把broadcast信息加在load上
  std::vector<NodePtr> del_nodes;
  for (const auto &node : asc_graph.GetAllNodes()) {
    if ((node->GetType() != kLoadType) && (node->GetType() != kGatherType)) {
      continue;
    }
    NodePtr cur_node = node;
    NodePtr pre_node = node;
    const auto load_op_desc = node->GetOpDesc();
    const auto load_output_desc = load_op_desc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(load_output_desc);
    const auto load_output_attr = load_output_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(load_output_attr);
    GE_ASSERT_TRUE(load_output_attr->axis.size() == load_output_attr->repeats.size());
    AscNodeAttr *load_node_attr = load_op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(load_node_attr);

    while ((cur_node->GetType() == kLoadType) || (cur_node->GetType() == kBroadcastType) ||
           (cur_node->GetType() == kTransposeType) || ((cur_node->GetType() == kGatherType))) {
      auto out_anchor = cur_node->GetOutDataAnchor(0);
      GE_ASSERT_NOTNULL(out_anchor);
      auto out_size = out_anchor->GetPeerInDataAnchors().size();
      GE_ASSERT_TRUE(out_size > 0U);
      auto out_anchor_peer = out_anchor->GetPeerInDataAnchors().at(0);
      GE_ASSERT_NOTNULL(out_anchor_peer);
      const auto next_node = out_anchor_peer->GetOwnerNode();
      GE_ASSERT_NOTNULL(next_node);
      if (cur_node->GetType() == kBroadcastType) {
        del_nodes.push_back(cur_node);
        ViewOpAttrInfo attr_info;
        GE_ASSERT_SUCCESS(BackendUtils::BackSteppingViewOp(pre_node, cur_node, attr_info, true));
        // 把broadcast在load上应用
        for (const auto &broadcast_axis : attr_info.broadcast_info) {
          auto it = std::find(load_node_attr->sched.axis.begin(), load_node_attr->sched.axis.end(), broadcast_axis);
          GE_ASSERT_TRUE(it != load_node_attr->sched.axis.end());
          int32_t index = std::distance(load_node_attr->sched.axis.begin(), it);
          load_output_attr->repeats[index] = kSymbolOne;
          load_output_attr->strides[index] = kSymbolZero;
        }
        if (!attr_info.broadcast_info.empty()) {
          load_output_attr->strides =
              ContiguousStrides(load_output_attr->repeats);  // slice和broadcast不共存于同一节点，可刷load strides
        }
      }
      pre_node = cur_node;
      cur_node = next_node;
    }
  }
  for (const auto &node : del_nodes) {
    GE_ASSERT_SUCCESS(asc_adapt::DelNode(asc_graph, node));
  }
  return SUCCESS;
}

// GE流程的反推
Status BackendUtils::BackSteppingViewOp(const NodePtr &data_node, const NodePtr &load_node, ViewOpAttrInfo &attr_info,
                                        bool just_broadcast) {
  TensorAttrInfo temp_data_attr;
  TensorAttrInfo temp_load_attr;
  GE_ASSERT_SUCCESS(GetNodeAttr(data_node, load_node, temp_data_attr, temp_load_attr));

  // 调用对应的处理函数
  GE_ASSERT_SUCCESS(BackSteppingViewOpPro(temp_data_attr, temp_load_attr, attr_info, just_broadcast));
  GE_ASSERT_SUCCESS(BackSteppingViewOpSlice(temp_data_attr, temp_load_attr, load_node, attr_info, false, false));
  return SUCCESS;
}

Status BackendUtils::FusedBackSteppingViewOp(const NodePtr &data_node, const NodePtr &load_node, ViewOpAttrInfo &attr_info,
                                             const bool is_merge_check) {
  TensorAttrInfo temp_data_attr;
  TensorAttrInfo temp_load_attr;
  GE_ASSERT_SUCCESS(GetNodeAttr(data_node, load_node, temp_data_attr, temp_load_attr));

  const auto compute_graph = load_node->GetOwnerComputeGraph();
  auto get_graph_attr = [](const std::vector<AxisPtr> &axis_info_list, std::vector<int64_t> &axis,
                           std::vector<ge::Expression> &repeats) -> Status {
    for (auto &axis_info : axis_info_list) {
      axis.push_back(axis_info->id);
      repeats.push_back(axis_info->size);
    }
    return SUCCESS;
  };
  auto graph_attr = compute_graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  TensorAttrInfo temp_graph_attr;
  GE_ASSERT_SUCCESS(get_graph_attr(graph_attr->axis, temp_graph_attr.axis, temp_graph_attr.repeats));

  // 调用对应的处理函数
  GELOGD("temp_data_attr.axis: %s.", AutofuseUtils::VectorToStr(temp_data_attr.axis).c_str());
  GELOGD("temp_graph_attr.axis: %s.", AutofuseUtils::VectorToStr(temp_graph_attr.axis).c_str());
  GE_ASSERT_SUCCESS(FusedBackSteppingViewOpTranspose(temp_graph_attr, temp_load_attr, attr_info));
  GE_ASSERT_SUCCESS(FusedBackSteppingViewOpBroadcast(temp_graph_attr, temp_load_attr, attr_info));
  GE_ASSERT_SUCCESS(BackSteppingViewOpSlice(temp_data_attr, temp_load_attr, load_node, attr_info, true, is_merge_check));
  return SUCCESS;
}

Status BackendUtils::FusedApplyViewOp(const NodePtr &data_node, const NodePtr &load_node,
                                      ViewOpAttrInfo &attr_info, const NodePtr &node2) {
  auto op_desc = load_node->GetOpDesc();
  AscNodeAttr *node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);
  auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  GE_ASSERT_TRUE(node_attr->sched.axis.size() == output_attr->repeats.size());
  GELOGD("fused apply view op before, node %s(%s) output repeats:%s strides:%s.", load_node->GetNamePtr(),
         load_node->GetType().c_str(), AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(output_attr->strides).c_str());

  GE_ASSERT_SUCCESS(FusedApplyViewOpPro(node_attr, output_attr, data_node, load_node, attr_info, node2));
  GELOGD("fused apply view op after, node %s(%s) output repeats:%s strides:%s.", load_node->GetNamePtr(),
         load_node->GetType().c_str(), AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(output_attr->strides).c_str());

  return SUCCESS;
}

// 把一个node的输出Tensor信息转移到其他节点
Status BackendUtils::TransferOutputTensorToOtherNode(const NodePtr &src_node, const NodePtr &node, 
                                                     const bool is_vertical, const std::vector<int64_t> &broadcast_info) {
  auto src_op_desc = src_node->GetOpDesc();
  GE_ASSERT_NOTNULL(src_op_desc);
  auto src_output_desc = src_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(src_output_desc);
  auto src_output_attr = src_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(src_output_attr);
  AscNodeAttr *src_node_attr = src_op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(src_node_attr);
  auto op_desc = node->GetOpDesc();
  auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  AscNodeAttr *node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);
  output_attr->repeats = src_output_attr->repeats;
  output_attr->strides = src_output_attr->strides;
  output_attr->axis = src_output_attr->axis;
  node_attr->sched.axis = src_node_attr->sched.axis;
  for (const auto &axis_id : broadcast_info) { // 此处目前只有水平融合会走
    GE_ASSERT_TRUE(static_cast<size_t>(axis_id) < output_attr->strides.size(), "%lu VS %zu.", axis_id,
                   output_attr->strides.size());
    output_attr->strides[axis_id] = kSymbolZero;
  }
  GELOGD(
      "transfer node %s(%s) output repeats:%s strides:%s axis:%s sched axis:%s to node %s(%s) output repeats:%s "
      "strides:%s axis:%s  sched axis:%s.",
      src_node->GetNamePtr(), src_node->GetType().c_str(), AutofuseUtils::VectorToStr(src_output_attr->repeats).c_str(),
      AutofuseUtils::VectorToStr(src_output_attr->strides).c_str(),
      AutofuseUtils::VectorToStr(src_output_attr->axis).c_str(),
      AutofuseUtils::VectorToStr(src_node_attr->sched.axis).c_str(), node->GetNamePtr(), node->GetType().c_str(),
      AutofuseUtils::VectorToStr(output_attr->repeats).c_str(),
      AutofuseUtils::VectorToStr(output_attr->strides).c_str(), AutofuseUtils::VectorToStr(output_attr->axis).c_str(),
      AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str());
  if (is_vertical && (((src_node->GetType() == kLoadType) || (src_node->GetType() == kGatherType)) &&
       ((node->GetType() == kLoadType) || (node->GetType() == kGatherType)))) {
    auto node_attr_ptr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(node_attr->ir_attr.get());
    GE_ASSERT_NOTNULL(node_attr_ptr);
    auto src_node_attr_ptr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(src_node_attr->ir_attr.get());
    GE_ASSERT_NOTNULL(src_node_attr_ptr);
    Expression offset;
    GELOGD("src node offset: %s.", offset.Serialize().get());
    if (src_node_attr_ptr->GetOffset(offset) == SUCCESS) {
      node_attr_ptr->SetOffset(offset);
    }
  }
  return SUCCESS;
}

Status BackendUtils::FindPrefaceLoadNodes(const OutDataAnchorPtr &o_anchor, std::vector<NodePtr> &nodes) {
  GE_ASSERT_NOTNULL(o_anchor);
  auto node = o_anchor->GetOwnerNode();
  if (((node->GetType() == kLoadType) || (node->GetType() == kGatherType)) &&
      (std::find(nodes.begin(), nodes.end(), node) == nodes.end())) {
    nodes.push_back(node);
    GELOGD("find ascgraph pre load node %s(%s).", node->GetNamePtr(), node->GetType().c_str());
    return SUCCESS;
  }

  for (size_t i = 0U; i < node->GetInDataNodesSize(); i++) {
    const auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    GE_ASSERT_NOTNULL(in_anchor);
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    GE_ASSERT_SUCCESS(FindPrefaceLoadNodes(peer_out_anchor, nodes));
  }
  return SUCCESS;
}

Status BackendUtils::ProcessViewOps(const NodePtr &node1, const NodePtr &node2, const NodePtr &store_peer_node,
                                    const NodePtr &data_node, const NodePtr &load_node) {
  std::vector<NodePtr> load_nodes;
  NodeAttrPair backup_node_attr_and_tensor_attr;
  GE_ASSERT_SUCCESS(BackupNodeAscTensorAttrAndAscNodeAttr(load_node, backup_node_attr_and_tensor_attr));
  GE_ASSERT_SUCCESS(FindPrefaceLoadNodes(store_peer_node->GetOutDataAnchor(0), load_nodes));
  for (const auto &pre_load_node : load_nodes) {
    const auto &load_in_anchor = pre_load_node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(load_in_anchor);
    const auto node_out_anchor = load_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(node_out_anchor);
    const auto pre_data_node = node_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(pre_data_node);
    // 用备份的tensor_attr和node_attr恢复/还原load_node对应的attr
    GE_ASSERT_SUCCESS(RecoverNodeAscTensorAttrAndAscNodeAttr(load_node, backup_node_attr_and_tensor_attr));
    std::vector<ViewOpAttrInfo> attr_infos;
    if (IsLoadConnectedToSplit(pre_load_node)) {
      GELOGD("data node: %s(%s), pre graph is split type, do not transfer data output tensor to pre data node.",
             data_node->GetNamePtr(), data_node->GetType().c_str());
      continue;
    }
    if (IsSimplestLoad(node1, pre_load_node, pre_data_node, attr_infos)) {
      // 如果前序的load是不包含view操作，直接更新data output tensor
      GELOGD(
          "data node: %s(%s) is view, pre graph not reduction, and pre data node: %s(%s) is no view, transfer "
          "data output tensor to pre data node.",
          data_node->GetNamePtr(), data_node->GetType().c_str(), pre_data_node->GetNamePtr(),
          pre_data_node->GetType().c_str());
      GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(data_node, pre_data_node));
      GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(load_node, pre_load_node));
    } else {
      // 如果前序load包含view操作，需要先反推出view具体操作，然后应用到当前data上
      GELOGD(
          "data node: %s(%s) is view, pre graph not reduction, and pre data node: %s(%s) is view, "
          "back step cur view and apply pre view.",
          data_node->GetNamePtr(), data_node->GetType().c_str(), pre_data_node->GetNamePtr(),
          pre_data_node->GetType().c_str());
      ViewOpAttrInfo attr_info;
      GE_ASSERT_SUCCESS(FusedBackSteppingViewOp(pre_data_node, pre_load_node, attr_info, false));
      GE_ASSERT_SUCCESS(FusedApplyViewOp(data_node, load_node, attr_info, node2));
      GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(data_node, pre_data_node));
      GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(load_node, pre_load_node));
    }
  }
  return SUCCESS;
}

Status BackendUtils::DeleteInvalidLoad(const NodePtr &node1, const NodePtr &node2, const NodePtr &store_peer_node,
                                       const OutDataAnchorPtr &in_anchor_peer, const NodePtr &data_node,
                                       std::vector<NodePtr> &del_output_and_store_nodes, bool is_reduction) {
  const auto size = store_peer_node->GetAllOutDataAnchorsSize();
  GE_ASSERT_TRUE(size > 0U);
  std::vector<int32_t> node_output_map(size, -1);
  GE_ASSERT_TRUE(static_cast<size_t>(in_anchor_peer->GetIdx()) < node_output_map.size(), "size %d VS size %zu.",
                 in_anchor_peer->GetIdx(), node_output_map.size());
  node_output_map[in_anchor_peer->GetIdx()] = 0;

  bool is_replaced = false;
  std::vector<ViewOpAttrInfo> attr_infos;
  for (const auto &load_in_anchor : in_anchor_peer->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(load_in_anchor);
    auto load_node = load_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(load_node);
    if ((load_node->GetType() != kLoadType) && (load_node->GetType() != kGatherType)) {
      continue;
    }
    // 如果load包含了view操作，我们不能删除load节点;cube不需要去merge load
    if (!IsSimplestLoad(node2, load_node, data_node, attr_infos) && !IsCubeAscNode(node1)) {
      if (is_reduction) {
        GELOGD(
            "data node: %s(%s) is view and pre graph have view node, transfer data output tensor to store peer node.",
            data_node->GetNamePtr(), data_node->GetType().c_str());
      } else {
        // 如果load有多个只需用一个去刷新前序load，存在多个load场景说明合并过，view操作一定都是相同的
        if (!is_replaced) {
          is_replaced = true;
          // 如果是非reduction场景的话需要把这个data信息更新到前序子图的data上
          GE_ASSERT_SUCCESS(ProcessViewOps(node1, node2, store_peer_node, data_node, load_node));
        }
      }
    }
    GELOGI("delete load node %s(%s), input map {}, output map %s.", load_node->GetNamePtr(),
           load_node->GetType().c_str(), AutofuseUtils::VectorToStr(node_output_map).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(store_peer_node, load_node, {}, node_output_map));
    del_output_and_store_nodes.push_back(load_node);
  }
  return SUCCESS;
}

Status BackendUtils::CreateSubGraphInput(const ComputeGraphPtr &sub_graph, const NodePtr &node, uint32_t in_nums,
                                         const std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes,
                                         bool need_inherit_pre_node_tensor) {
  for (auto input_idx = 0U; input_idx < in_nums; input_idx++) {
    OpDescBuilder op_desc_builder("data_" + std::to_string(AutofuseUtils::GenUniqueNumber()), kDataType);
    op_desc_builder.AddInput("x");
    op_desc_builder.AddOutput("y");
    const auto &op_desc = op_desc_builder.Build();
    const auto &data_node = sub_graph->AddNode(op_desc);
    GE_ASSERT_NOTNULL(data_node);
    GE_ASSERT_SUCCESS(data_node->SetOwnerComputeGraph(sub_graph));

    // fused中data需要继承前序node的输出tensor，但是序列化的时候不需要继承该信息
    if (need_inherit_pre_node_tensor) {
      GE_ASSERT_TRUE(static_cast<size_t>(input_idx) < pre_nodes.size(), "size %u VS size %zu.",
                     input_idx, pre_nodes.size());
      auto pre_node_op_desc = pre_nodes[input_idx].first->GetOpDescBarePtr();
      auto pre_node_desc = pre_node_op_desc->MutableOutputDesc(pre_nodes[input_idx].second);
      GE_ASSERT_NOTNULL(pre_node_desc);
      auto pre_node_attr = pre_node_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(pre_node_attr);
      auto data_desc = op_desc->MutableOutputDesc(0);
      GE_ASSERT_NOTNULL(data_desc);
      auto data_attr = data_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(data_attr);
      *data_attr = *pre_node_attr;
    }
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(input_idx)));
  }
  GELOGD("node %s(%s) in num(%u), create subgraph input success.", node->GetNamePtr(), node->GetType().c_str(), in_nums);
  return SUCCESS;
}

NodePtr BackendUtils::CreateNetOutput(const ComputeGraphPtr &sub_graph, uint32_t out_nums) {
  OpDescBuilder op_desc_builder("netoutput_" + std::to_string(AutofuseUtils::GenUniqueNumber()), kNetOutputType);
  for (auto output_idx = 0U; output_idx < out_nums; output_idx++) {
    op_desc_builder.AddInput("x_" + std::to_string(output_idx));
    op_desc_builder.AddOutput("y_" + std::to_string(output_idx));
  }
  const auto &op_desc = op_desc_builder.Build();
  const auto &net_output_node = sub_graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(net_output_node);
  GE_ASSERT_SUCCESS(net_output_node->SetOwnerComputeGraph(sub_graph));
  for (auto output_idx = 0U; output_idx < out_nums; output_idx++) {
    auto input_desc = op_desc->MutableInputDesc(output_idx);
    GE_ASSERT_NOTNULL(input_desc);
    auto input_attr = input_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(input_attr);
    auto output_desc = op_desc->MutableOutputDesc(output_idx);
    GE_ASSERT_NOTNULL(output_desc);
    auto output_attr = output_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(output_attr);
  }
  return net_output_node;
}

Status BackendUtils::CreateSubGraphOutput(const ComputeGraphPtr &sub_graph, const NodePtr &node, uint32_t out_nums,
                                          const std::vector<uint32_t> &node_output_index) {
  const auto net_output_node = CreateNetOutput(sub_graph, out_nums);
  GE_ASSERT_NOTNULL(net_output_node);
  for (auto output_idx = 0U; output_idx < out_nums; output_idx++) {
    GE_ASSERT_TRUE(static_cast<size_t>(output_idx) < node_output_index.size(), "size %zu VS size %zu.", output_idx,
                   node_output_index.size());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(node->GetOutDataAnchor(node_output_index[output_idx]),
                                                net_output_node->GetInDataAnchor(output_idx)));
    auto op_desc = net_output_node->GetOpDescBarePtr();
    auto input_desc = op_desc->MutableInputDesc(output_idx);
    GE_ASSERT_NOTNULL(input_desc);
    auto input_attr = input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(input_attr);
    auto node_op_desc = node->GetOpDescBarePtr();
    auto node_output_desc = node_op_desc->MutableOutputDesc(node_output_index[output_idx]);
    GE_ASSERT_NOTNULL(node_output_desc);
    auto node_attr = node_output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    *input_attr = *node_attr;
  }
  GELOGD("node %s(%s) out num(%u), create subgraph output success.", node->GetNamePtr(), node->GetType().c_str(),
         out_nums);
  return SUCCESS;
}

bool BackendUtils::IsOutputNode(const NodePtr &node) {
  if (node->GetType() == kOutputType) {
    return true;
  }
  return false;
}

bool BackendUtils::IsInputNode(const NodePtr &node) {
  if (node->GetType() == kDataType) {
    return true;
  }
  return false;
}

bool BackendUtils::IsNetOutputNode(const NodePtr &node) {
  if (node->GetType() == kNetOutputType) {
    return true;
  }
  return false;
}

bool BackendUtils::IsAscBackendNoKernelNode(const NodePtr &node) {
  if (node->GetType() == kAscBackendNoKernelType) {
    return true;
  }
  return false;
}

bool BackendUtils::IsCanFuseBlackList(const NodePtr &node) {
  if (IsNetOutputNode(node) || IsInputNode(node) || IsOutputNode(node) || IsAscBackendNoKernelNode(node)) {
    GELOGI("node %s(%s) no need fuse.", node->GetNamePtr(), node->GetType().c_str());
    return true;
  }
  return false;
}

AutoFuseAttrs *BackendUtils::GetNodeAutoFuseAttr(const NodePtr &node) {
  auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
  return attr;
}

bool BackendUtils::IsBackendFuseNode(const NodePtr &node) {
  auto attr = GetNodeAutoFuseAttr(node);
  if (attr == nullptr) {
    GELOGI("node %s(%s) auto fuse attr is null, no need fuse.", node->GetNamePtr(), node->GetType().c_str());
    return false;
  }
  auto fuse_asc_graph = attr->GetAscGraph();
  auto fuse_compute_graph = attr->GetFuseComputeGraph();
  if (fuse_asc_graph == nullptr && fuse_compute_graph == nullptr) {
    GELOGD("node %s(%s) fused graph attr null, no need fuse.", node->GetNamePtr(), node->GetType().c_str());
    return false;
  }
  return true;
}

const std::shared_ptr<AscGraph> BackendUtils::GetNodeFusedAscGraph(const NodePtr &node) {
  auto attr = GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  return attr->GetAscGraph();
}

ComputeGraphPtr BackendUtils::GetNodeFusedComputeGraph(const NodePtr &node) {
  auto attr = GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  return attr->GetFuseComputeGraph();
}

Status BackendUtils::AddInputOutputNodesForAscGraph(const ComputeGraphPtr &asc_graph) {
  auto index = 0U;
  if (!asc_graph->GetInputNodes().empty() && !asc_graph->GetOutputNodes().empty()) {
    return SUCCESS;
  }
  for (const auto &asc_node : asc_graph->GetAllNodes()) {
    if (IsInputNode(asc_node)) {
      GE_ASSERT_NOTNULL(asc_graph->AddInputNode(asc_node));
    } else if (IsOutputNode(asc_node)) {
      GE_ASSERT_NOTNULL(asc_graph->AddOutputNodeByIndex(asc_node, index++));
    }
  }
  return SUCCESS;
}

Status BackendUtils::GetNodeFusedGraph(const NodePtr &node, ComputeGraphPtr &fused_graph) {
  if (node->GetType() == kFusedAscBackendType) {
    auto graph = GetNodeFusedComputeGraph(node);
    fused_graph = graph;
  } else if (node->GetType() == kAscBackendType) {
    auto asc_graph = GetNodeFusedAscGraph(node);
    GE_ASSERT_NOTNULL(asc_graph);
    fused_graph = AscGraphUtils::GetComputeGraph(*asc_graph);
    GE_ASSERT_NOTNULL(fused_graph);
    GE_ASSERT_SUCCESS(AddInputOutputNodesForAscGraph(fused_graph));
  } else {
    return FAILED;
  }
  GE_ASSERT_NOTNULL(fused_graph);
  return SUCCESS;
}

void BackendUtils::GetOutputMap(const std::vector<std::pair<int32_t, int32_t>> &link_map,
                                const uint32_t node_out_node_size, std::vector<int32_t> &node_output_map) {
  std::vector<std::pair<int32_t, int32_t>> temp_link_map = link_map;
  for (auto i = 0U; i < node_out_node_size; i++) {
    if (temp_link_map.empty()){
      node_output_map.push_back(static_cast<int32_t>(i));
      continue;
    }
    auto it = std::find_if(temp_link_map.begin(), temp_link_map.end(), [i](const std::pair<int32_t, int32_t> &pair) {
      return static_cast<uint32_t>(pair.first) == i;
    });
    if (it != temp_link_map.end()) {
      temp_link_map.erase(it);
      continue;
    }

    node_output_map.push_back(static_cast<int32_t>(i));
  }
}

// 更新子图的fused_subgraph_outputs属性，保证融合后的节点输出和ascgraph的输出节点对应，简化对子图输出的管理
Status BackendUtils::UpdateSubgraphOutputAttr(const ComputeGraphPtr &subgraph, const NodePtr &node) {
  if (node->GetType() == kFusedAscBackendType) {
    GELOGD("node %s(%s) reset fused subgraph output attr.", node->GetNamePtr(), node->GetType().c_str());
    GE_ASSERT_SUCCESS(ResetFusedSubgraphOutputsAttr(node));
    return SUCCESS;
  }
  if (node->GetType() != kAscBackendType) {
    GELOGD("node %s(%s) no need update subgraph output attr.", node->GetNamePtr(), node->GetType().c_str());
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(AddInputOutputNodesForAscGraph(subgraph));
  auto autofuse_attr = GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  auto &outputs = GetInterAttrs(autofuse_attr).fused_subgraph_outputs;
  auto subgraph_outputs = subgraph->GetOutputNodes();
  if (outputs.empty()) {
    const auto node_out_data_size = node->GetAllOutDataAnchorsSize();
    GE_ASSERT_TRUE(static_cast<size_t>(node_out_data_size) <= subgraph_outputs.size(), "size %zu VS size %zu.",
                   static_cast<size_t>(node_out_data_size), subgraph_outputs.size());
    for (auto node_output = 0U; node_output < node_out_data_size; node_output++) {
      const auto node_out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(node_output));
      GE_ASSERT_NOTNULL(node_out_anchor);
      const auto &node_peer_anchors = node_out_anchor->GetPeerInDataAnchors();
      for (size_t i = 0U; i < node_peer_anchors.size(); i++) {
        outputs.push_back(subgraph_outputs.at(node_output));
      }
      if (node_peer_anchors.empty()) {
        outputs.push_back(subgraph_outputs.at(node_output));
      }
    }
  } else {
    GE_ASSERT_SUCCESS(ResetFusedSubgraphOutputsAttr(node));
  }
  return SUCCESS;
}

inline Status UpdateFusedAscBackendNetOutputReally(const ComputeGraphPtr &fused_compute_graph,
                                                   const std::vector<int32_t> &in_data_anchor_indexes) {
  const auto netoutput = fused_compute_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netoutput);
  // 创建新的NetOutput, 把需要保留的输入anchor_index保留下来
  const auto net_output_node = BackendUtils::CreateNetOutput(fused_compute_graph, in_data_anchor_indexes.size());
  GE_ASSERT_NOTNULL(net_output_node);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(net_output_node, netoutput, in_data_anchor_indexes, {}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(fused_compute_graph, netoutput));
  NodeUtils::UnlinkAll(*netoutput);
  return SUCCESS;
}

Status BackendUtils::UpdateFusedAscBackendNetOutput(const NodePtr &node) {
  const auto *attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  auto &fused_compute_graph = attr->GetFuseComputeGraph();
  GE_ASSERT_NOTNULL(fused_compute_graph);
  const auto netoutput = fused_compute_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netoutput);
  std::vector<InDataAnchorPtr> in_data_anchors;
  for (const auto &netoutput_in_anchor : netoutput->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(netoutput_in_anchor);
    in_data_anchors.push_back(netoutput_in_anchor);
  }
  size_t in_anchor_size = in_data_anchors.size();
  // Eliminate Duplicates
  asc_adapt::RemoveDuplicatesConditional(in_data_anchors, [](const InDataAnchorPtr &a, const InDataAnchorPtr &b) {
    return a->GetPeerOutAnchor() == b->GetPeerOutAnchor();
  });

  if (in_data_anchors.size() == in_anchor_size) {
    // 去重后数量无变化，无需更新netoutput
    GELOGD("node %s(%s) no need update NetOutput and in anchor size:%zu.", node->GetNamePtr(), node->GetType().c_str(),
           in_anchor_size);
    return SUCCESS;
  }
  std::vector<int32_t> in_data_anchor_indexes;
  for (const auto &in_data_anchor : in_data_anchors) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    in_data_anchor_indexes.push_back(in_data_anchor->GetIdx());
  }
  GELOGD("node %s(%s) UpdateFusedAscBackendNetOutput in_data_anchor_indexes:%s.", node->GetNamePtr(),
         node->GetType().c_str(), AutofuseUtils::VectorToStr(in_data_anchor_indexes).c_str());
  UpdateFusedAscBackendNetOutputReally(fused_compute_graph, in_data_anchor_indexes);
  return SUCCESS;
}

Status BackendUtils::UpdateSubGraphOutput(std::vector<ge::NodePtr> &outputs,
                                          const std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map) {
  std::set<size_t, std::greater<size_t>> positions_to_remove;
  for (const auto &link : subgraph_link_map) {
    GELOGD("need delete subgraph output node index=%u.", link.first);
    positions_to_remove.emplace(link.first);
  }
  for (const size_t pos : positions_to_remove) {
    GE_ASSERT_TRUE(static_cast<size_t>(pos) < outputs.size(), "size %zu VS size %zu.", pos, outputs.size());
    outputs.erase(outputs.begin() + pos);
  }
  return SUCCESS;
}

Status BackendUtils::CreateNewNodeInputDescAttr(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                                                const std::vector<int32_t> &node1_input_map,
                                                const std::vector<int32_t> &node2_input_map) {
  auto op_desc = new_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto input_idx = 0U;
  for (input_idx = 0U; input_idx < node1_input_map.size(); input_idx++) {
    auto input_desc = op_desc->MutableInputDesc(input_idx);
    GE_ASSERT_NOTNULL(input_desc);
    auto input_attr = input_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(input_attr);

    auto node1_op_desc = node1->GetOpDescBarePtr();
    const auto node1_input_desc = node1_op_desc->MutableInputDesc(node1_input_map[input_idx]);
    GE_ASSERT_NOTNULL(node1_input_desc);
    auto node1_attr = node1_input_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node1_attr);

    input_desc->SetOriginShape(node1_input_desc->GetOriginShape());
    input_desc->SetShape(node1_input_desc->GetShape());
    input_desc->SetDataType(node1_input_desc->GetDataType());
    input_desc->SetOriginDataType(node1_input_desc->GetOriginDataType());
    input_desc->SetFormat(node1_input_desc->GetFormat());
    input_desc->SetOriginFormat(node1_input_desc->GetOriginFormat());
    *input_attr = *node1_attr;
  }
  for (; input_idx < node2_input_map.size(); input_idx++) {
    auto input_desc = op_desc->MutableInputDesc(input_idx);
    GE_ASSERT_NOTNULL(input_desc);
    auto input_attr = input_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(input_attr);

    GE_ASSERT_NOTNULL(node2);
    auto node2_op_desc = node2->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(node2_op_desc);
    const auto node2_input_desc = node2_op_desc->MutableInputDesc(node2_input_map[input_idx]);
    GE_ASSERT_NOTNULL(node2_input_desc);
    auto node2_attr = node2_input_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node2_attr);

    input_desc->SetOriginShape(node2_input_desc->GetOriginShape());
    input_desc->SetShape(node2_input_desc->GetShape());
    input_desc->SetDataType(node2_input_desc->GetDataType());
    input_desc->SetOriginDataType(node2_input_desc->GetOriginDataType());
    input_desc->SetFormat(node2_input_desc->GetFormat());
    input_desc->SetOriginFormat(node2_input_desc->GetOriginFormat());
    *input_attr = *node2_attr;
  }
  return SUCCESS;
}

Status BackendUtils::CreateNewNodeOutputDescAttr(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                                                 const std::vector<int32_t> &node1_output_map,
                                                 const std::vector<int32_t> &node2_output_map) {
  auto op_desc = new_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto output_idx = 0U;
  for (output_idx = 0U; output_idx < node1_output_map.size(); output_idx++) {
    auto output_desc = op_desc->MutableOutputDesc(output_idx);
    GE_ASSERT_NOTNULL(output_desc);
    auto output_attr = output_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(output_attr);

    auto node1_op_desc = node1->GetOpDescBarePtr();
    auto node1_output_desc = node1_op_desc->MutableOutputDesc(node1_output_map[output_idx]);
    GE_ASSERT_NOTNULL(node1_output_desc);
    auto node1_attr = node1_output_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node1_attr);

    output_desc->SetOriginShape(node1_output_desc->GetOriginShape());
    output_desc->SetShape(node1_output_desc->GetShape());
    output_desc->SetDataType(node1_output_desc->GetDataType());
    output_desc->SetOriginDataType(node1_output_desc->GetOriginDataType());
    output_desc->SetFormat(node1_output_desc->GetFormat());
    output_desc->SetOriginFormat(node1_output_desc->GetOriginFormat());
    *output_attr = *node1_attr;
  }
  for (; output_idx < node2_output_map.size(); output_idx++) {
    auto output_desc = op_desc->MutableOutputDesc(output_idx);
    GE_ASSERT_NOTNULL(output_desc);
    auto output_attr = output_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(output_attr);

    GE_ASSERT_NOTNULL(node2);
    auto node2_op_desc = node2->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(node2_op_desc);
    auto node2_output_desc = node2_op_desc->MutableOutputDesc(node2_output_map[output_idx]);
    GE_ASSERT_NOTNULL(node2_output_desc);
    auto node2_attr = node2_output_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(node2_attr);

    output_desc->SetOriginShape(node2_output_desc->GetOriginShape());
    output_desc->SetShape(node2_output_desc->GetShape());
    output_desc->SetDataType(node2_output_desc->GetDataType());
    output_desc->SetOriginDataType(node2_output_desc->GetOriginDataType());
    output_desc->SetFormat(node2_output_desc->GetFormat());
    output_desc->SetOriginFormat(node2_output_desc->GetOriginFormat());
    *output_attr = *node2_attr;
  }
  return SUCCESS;
}

Status BackendUtils::TryRemoveNodesCtrEdges(const NodePtr &node1, const NodePtr &node2) {
  auto node1_out_anchor = node1->GetOutControlAnchor();
  GE_ASSERT_NOTNULL(node1_out_anchor);
  for (const auto &in_anchor : node1_out_anchor->GetPeerInControlAnchors()) {
    auto peer_node = in_anchor->GetOwnerNode();
    if (peer_node == node2) {
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(node1_out_anchor, in_anchor));
    }
  }
  return SUCCESS;
}

NodePtr BackendUtils::GetDataNextNode(const NodePtr &data) {
  auto data_out_anchor = data->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(data_out_anchor);
  auto data_out_size = data_out_anchor->GetPeerInDataAnchors().size();
  GE_ASSERT_TRUE(data_out_size > 0U);
  auto data_out_anchor_peer = data_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(data_out_anchor_peer);
  return data_out_anchor_peer->GetOwnerNode();
}

Status BackendUtils::DumpAscGraph(const NodePtr &node) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }
  if (node->GetType() == kAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(GetNodeFusedGraph(node, graph));
    GELOGD("\n  ascgraph name:(%s), node name:(%s), node type:(%s)", graph->GetName().c_str(), node->GetName().c_str(),
           node->GetType().c_str());
    for (const auto &asc_node : graph->GetAllNodes()) {
      auto asc_node_op_desc = asc_node->GetOpDesc();
      GE_ASSERT_NOTNULL(asc_node_op_desc);
      AscNodeAttr *asc_node_attr = asc_node_op_desc->GetAttrsGroup<AscNodeAttr>();
      GE_ASSERT_NOTNULL(asc_node_attr);
      GELOGD("\n    node name:%s(%s)\n     sched axis(%s)", asc_node->GetName().c_str(), asc_node->GetType().c_str(),
             AutofuseUtils::VectorToStr(asc_node_attr->sched.axis).c_str());
      for (auto index = 0U; index < asc_node_op_desc->GetAllOutputsDescSize(); index++) {
        auto asc_node_output_desc = asc_node_op_desc->MutableOutputDesc(index);
        GE_ASSERT_NOTNULL(asc_node_output_desc);
        auto asc_node_tensor_attr = asc_node_output_desc->GetAttrsGroup<AscTensorAttr>();
        GE_ASSERT_NOTNULL(asc_node_tensor_attr);
        GELOGD("\n        tensor[%u]: repeats(%s), strides(%s), axis(%s).", index,
               AutofuseUtils::VectorToStr(asc_node_tensor_attr->repeats).c_str(),
               AutofuseUtils::VectorToStr(asc_node_tensor_attr->strides).c_str(),
               AutofuseUtils::VectorToStr(asc_node_tensor_attr->axis).c_str());
      }
    }

    auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(graph_attr);
    for (auto &axis_info : graph_attr->axis) {
      GELOGD("\n  axis info: axis name(%s), axis id(%ld), axis size(%s).", axis_info->name.c_str(), axis_info->id,
             std::string(axis_info->size.Str().get()).c_str());
    }
  } else if (node->GetType() == kFusedAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(GetNodeFusedGraph(node, graph));
    GELOGD("\nfusedascgraph name:(%s)", graph->GetName().c_str());
    for (const auto &asc_node : graph->GetAllNodes()) {
      GE_ASSERT_SUCCESS(DumpAscGraph(asc_node));
    }
  } else {
    // nothing
  }
  return SUCCESS;
}

Status BackendUtils::GetAscGraphAxisGroup(const NodePtr &node, optimize::autoschedule::AxisGroup &axes_group,
                                          const AxisPairSet &axis_map) {
  auto attr = GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  if (GetInterAttrs(attr).axis_group.IsEmpty()) {
    if ((node->GetType() != kAscBackendType)) {
      GELOGD("node(%s) fusedascbackend node no axis group info.", node->GetNamePtr());
      return FAILED;
    }
    auto asc_graph = GetNodeFusedAscGraph(node);
    GE_ASSERT_NOTNULL(asc_graph);
    GE_ASSERT_SUCCESS(GenAscGraphAxisGroup(*asc_graph, axes_group));
    GetInterAttrs(attr).axis_group = axes_group;
  } else {
    axes_group = GetInterAttrs(attr).axis_group;
  }
  GELOGD("node:%s(%s) origin group info:%s.", node->GetNamePtr(), node->GetType().c_str(),
         axes_group.ToString().c_str());
  if (ConvertAxis(axis_map, axes_group.x_group, true) != SUCCESS) {
    GELOGD("node(%s) convert xgroup axis failed.", node->GetNamePtr());
    return FAILED;
  }
  if (ConvertAxis(axis_map, axes_group.y_group, true) != SUCCESS) {
    GELOGD("node(%s) convert ygroup axis failed.", node->GetNamePtr());
    return FAILED;
  }
  if (ConvertAxis(axis_map, axes_group.r_group, true) != SUCCESS) {
    GELOGD("node(%s) convert rgroup axis failed.", node->GetNamePtr());
    return FAILED;
  }
  if (ConvertAxis(axis_map, axes_group.n_group, true) != SUCCESS) {
    GELOGD("node(%s) convert ngroup axis failed.", node->GetNamePtr());
    return FAILED;
  }
  GELOGD("node:%s(%s) convert group info:%s.", node->GetNamePtr(), node->GetType().c_str(),
         axes_group.ToString().c_str());
  return SUCCESS;
}

void BackendUtils::MergeAxesOrder(optimize::autoschedule::AxisGroup &group1,
                                  optimize::autoschedule::AxisGroup &group2) {
  std::set<size_t> merged_set;
  merged_set.insert(group1.axes_order.begin(), group1.axes_order.end());
  merged_set.insert(group2.axes_order.begin(), group2.axes_order.end());

  bool is_group1_subset = (group2.axes_order.size() == merged_set.size());
  bool is_group2_subset = (group1.axes_order.size() == merged_set.size());
  if (is_group1_subset && is_group2_subset) {
    // 如果两者互为子集，说明它们完全相同，无需合并
    // no op
  } else if (is_group1_subset) {
    // 如果 group1 是 group2 的子集，将 group2 的差异元素追加到 group1.axes_order 中
    std::vector<size_t> difference;
    std::set_difference(group2.axes_order.begin(), group2.axes_order.end(), group1.axes_order.begin(),
                        group1.axes_order.end(), std::back_inserter(difference));
    group1.axes_order.insert(group1.axes_order.end(), difference.begin(), difference.end());
    GELOGD("complete group1 axes_order size:%zu, group2 axes_order size:%zu.", group1.axes_order.size(),
           group2.axes_order.size());
  } else if (is_group2_subset) {
    // 如果 group2 是 group1 的子集，将 group1 的差异元素追加到 group2.axes_order 中
    std::vector<size_t> difference;
    std::set_difference(group1.axes_order.begin(), group1.axes_order.end(), group2.axes_order.begin(),
                        group2.axes_order.end(), std::back_inserter(difference));
    group2.axes_order.insert(group2.axes_order.end(), difference.begin(), difference.end());
    GELOGD("complete group1 axes_order size:%zu, group2 axes_order size:%zu.", group1.axes_order.size(),
           group2.axes_order.size());
  } else {
    // 如果两者都不是对方的子集，无需合并
    GELOGD("axes_order not map, no need merge.");
  }
}

bool BackendUtils::IsCanMergeAxisGroup(optimize::autoschedule::AxisGroup &group1,
                                       optimize::autoschedule::AxisGroup &group2,
                                       optimize::autoschedule::AxisGroup &merged_axes_group) {
  GELOGD("origin group1 info:%s, group2 info:%s.", group1.ToString().c_str(), group2.ToString().c_str());

  // 两个group可能轴个数不同，比如reduce+elemwise垂直融合
  auto collect_elements = [](const std::vector<int64_t> &vec1, const std::vector<int64_t> &vec2,
                             const std::vector<int64_t> &vec3, const std::vector<int64_t> &vec4) {
    std::set<int64_t> result;
    result.insert(vec1.begin(), vec1.end());
    result.insert(vec2.begin(), vec2.end());
    result.insert(vec3.begin(), vec3.end());
    result.insert(vec4.begin(), vec4.end());
    return result;
  };
  auto group1_all_elements = collect_elements(group1.x_group, group1.y_group, group1.r_group, group1.n_group);
  auto group2_all_elements = collect_elements(group2.x_group, group2.y_group, group2.r_group, group2.n_group);

  std::set<int64_t> merged_set;
  merged_set.insert(group1_all_elements.begin(), group1_all_elements.end());
  merged_set.insert(group2_all_elements.begin(), group2_all_elements.end());

  bool is_group1_subset = (group2_all_elements.size() == merged_set.size());
  bool is_group2_subset = (group1_all_elements.size() == merged_set.size());

  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  auto fwk_type = config.fwk_type;
  bool is_ge_call = false;
  if (fwk_type == AutoFuseFwkType::kGe) {
    is_ge_call = true;
  }

  if (is_group1_subset && is_group2_subset) {
    // 如果两者互为子集，说明它们完全相同，无需合并
    // no op
  } else if (is_group1_subset) {
    // 如果 group1 是 group2 的子集，将 group2 的差异元素追加到 group1.y_group 中
    std::vector<int64_t> difference;
    std::set_difference(group2_all_elements.begin(), group2_all_elements.end(), group1_all_elements.begin(),
                        group1_all_elements.end(), std::back_inserter(difference));
    group1.y_group.insert(group1.y_group.end(), difference.begin(), difference.end());
    GELOGD("complete group1 info:%s, group2 info:%s.", group1.ToString().c_str(), group2.ToString().c_str());
  } else if (is_group2_subset) {
    // 如果 group2 是 group1 的子集，将 group1 的差异元素追加到 group2.y_group 中
    std::vector<int64_t> difference;
    std::set_difference(group1_all_elements.begin(), group1_all_elements.end(), group2_all_elements.begin(),
                        group2_all_elements.end(), std::back_inserter(difference));
    group2.y_group.insert(group2.y_group.end(), difference.begin(), difference.end());
    GELOGD("complete group1 info:%s, group2 info:%s.", group1.ToString().c_str(), group2.ToString().c_str());
  } else {
    // 如果两者都不是对方的子集，无需合并
    auto ret = CanMergeAxisGroup(group1, group2, merged_axes_group, is_ge_call);
    if (ret != SUCCESS) {
      GELOGD("axis group not map, can merge failed.");
    }
    return ret;
  }

  // 补齐axes_order
  MergeAxesOrder(group1, group2);
  auto ret = CanMergeAxisGroup(group1, group2, merged_axes_group, is_ge_call);
  GELOGD("merged group info:%s.", merged_axes_group.ToString().c_str());
  return ret;
}

bool BackendUtils::CheckAxisSubsetRelation(const std::vector<int64_t> &axis1, const std::vector<int64_t> &axis2) {
  size_t i = 0U;
  size_t j = 0U;
  while (i < axis1.size() && j < axis2.size()) {
    if (axis1[i] == axis2[j]) {
      ++i;
    }
    ++j;
  }
  if (i == axis1.size()) {
    return true;  // axis1 是 axis2 的循序子集
  }
  i = 0U;
  j = 0U;
  while (i < axis2.size() && j < axis1.size()) {
    if (axis2[i] == axis1[j]) {
      ++i;
    }
    ++j;
  }
  if (i == axis2.size()) {
    return true;  // axis2 是 axis1 的循序子集
  }

  return false;
}

bool BackendUtils::ViewOpMerge(const NodePtr &data_node1, const NodePtr &data_node2, const NodePtr &load_node1,
                               const NodePtr &load_node2, bool is_simplest_load1, bool is_simplest_load2) {
  if (is_simplest_load1 && !is_simplest_load2) {
    GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(data_node2, data_node1, false));
    const auto data1_out_anchor = data_node1->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(data1_out_anchor);
    for (const auto &node_peer_anchor : data1_out_anchor->GetPeerInDataAnchors()) {
      GE_ASSERT_NOTNULL(node_peer_anchor);
      const auto load_node = node_peer_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(load_node);
      ViewOpAttrInfo attr_info;
      GE_ASSERT_SUCCESS(FusedBackSteppingViewOp(data_node2, load_node2, attr_info, false));
      GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(load_node2, load_node, false, attr_info.broadcast_info));
    }
    GELOGD("data node: %s(%s) is no view and data node: %s(%s) is view, data node can fuse.", data_node1->GetNamePtr(),
           data_node1->GetType().c_str(), data_node2->GetNamePtr(), data_node2->GetType().c_str());
    return true;
  } else if (!is_simplest_load1 && is_simplest_load2) {
    GELOGD("data node: %s(%s) is view and data node: %s(%s) is no view, data node can fuse.", data_node1->GetNamePtr(),
           data_node1->GetType().c_str(), data_node2->GetNamePtr(), data_node2->GetType().c_str());
    GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(load_node1, load_node2, false));
    return true;
  }
  return false;
}

bool BackendUtils::TryAscDataNodeMerge(const NodePtr &node1, const NodePtr &node2, const NodePtr &data_node1,
                                       const NodePtr &data_node2) {
  auto asc_node_op_desc1 = data_node1->GetOpDesc();
  GE_ASSERT_NOTNULL(asc_node_op_desc1);
  auto asc_node_output_desc1 = asc_node_op_desc1->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(asc_node_output_desc1);
  auto asc_node_output_tensor_attr1 = asc_node_output_desc1->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(asc_node_output_tensor_attr1);
  const auto load_node1 = GetDataNextNode(data_node1);
  GE_ASSERT_NOTNULL(load_node1);

  auto asc_node_op_desc2 = data_node2->GetOpDesc();
  GE_ASSERT_NOTNULL(asc_node_op_desc2);
  auto asc_node_output_desc2 = asc_node_op_desc2->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(asc_node_output_desc2);
  auto asc_node_output_tensor_attr2 = asc_node_output_desc2->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(asc_node_output_tensor_attr2);
  const auto load_node2 = GetDataNextNode(data_node2);
  GE_ASSERT_NOTNULL(load_node2);

  std::vector<ViewOpAttrInfo> attr_infos1;
  std::vector<ViewOpAttrInfo> attr_infos2;
  bool is_simplest_load1 = IsSimplestLoad(node1, load_node1, data_node1, attr_infos1);
  bool is_simplest_load2 = IsSimplestLoad(node2, load_node2, data_node2, attr_infos2);
  if (BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2)) {
    GELOGD("node1 and node2 broadcast infos are same, data node can fuse.");
    return true;
  }

  if (ViewOpMerge(data_node1, data_node2, load_node1, load_node2, is_simplest_load1, is_simplest_load2)) {
    return true;
  }
  // 非view算子具备子集关系的可以merge
  if (BackendUtils::CheckAxisSubsetRelation(asc_node_output_tensor_attr1->axis, asc_node_output_tensor_attr2->axis)) {
    const auto data1_out_anchor = data_node1->GetOutDataAnchor(0);
    for (const auto &node_peer_anchor : data1_out_anchor->GetPeerInDataAnchors()) {
      GE_ASSERT_NOTNULL(node_peer_anchor);
      const auto load_node = node_peer_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(load_node);
      GE_ASSERT_SUCCESS(TransferOutputTensorToOtherNode(load_node2, load_node, false));
    }
    GELOGD("data node: %s(%s) and data node: %s(%s) is subset relation, data node can fuse.", data_node1->GetNamePtr(),
           data_node1->GetType().c_str(), data_node2->GetNamePtr(), data_node2->GetType().c_str());
    return true;
  }
  GELOGD("data node: %s(%s) and data node: %s(%s) mismatch, data node can't fuse.", data_node1->GetNamePtr(),
         data_node1->GetType().c_str(), data_node2->GetNamePtr(), data_node2->GetType().c_str());
  return false;
}

Status BackendUtils::ConvertAxis(const AxisPairSet &node_map, std::vector<int64_t> &base_line_axis, bool need_flash) {
  if (node_map.empty()) {
    return SUCCESS;
  }
  for (auto i = 0U; i < base_line_axis.size(); i++) {
    auto axis_id = base_line_axis[i];
    auto it = std::find_if(node_map.begin(), node_map.end(),
                           [axis_id](const std::pair<int64_t, int64_t> &pair) { return pair.first == axis_id; });
    if (it != node_map.end()) {
      if (need_flash) {
        base_line_axis[i] = it->second;
      }
    } else {
      GELOGW("subgraph can't find axis=%ld map info.", axis_id);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status BackendUtils::ConvertAxis(const AxisPairSet &node_map, int64_t &axis_id, bool need_flash) {
  if (node_map.empty()) {
    return SUCCESS;
  }
  std::vector<int64_t> base_line_axis;
  base_line_axis.push_back(axis_id);
  if (ConvertAxis(node_map, base_line_axis, need_flash) != SUCCESS) {
    return FAILED;
  }
  axis_id = base_line_axis.back();
  return SUCCESS;
}

std::string BackendUtils::AscAxisToStr(const std::vector<AxisPtr> &axis) {
  std::ostringstream oss;
  oss << "[";
  auto i = 0U;
  for (const auto &axis_info : axis) {
    oss << "(axis name:" << axis_info->name.c_str() << ", axis id:" << axis_info->id << ", axis size:"
        << std::string(axis_info->size.Str().get()).c_str() << ")";
    if (i < axis.size() - 1U) {
      oss << ", ";
    }
    i++;
  }
  oss << "]";
  return oss.str();
}


NodePtr BackendUtils::GetFusedAscBackendInputNode(const NodePtr &node, const int32_t index,
                                                  int32_t &in_anchor_idx) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  const auto subgraph_input_nodes = graph->GetInputNodes();
  GE_ASSERT_TRUE(subgraph_input_nodes.size() > static_cast<size_t>(index));
  auto data_node = subgraph_input_nodes.at(index);
  GE_ASSERT_NOTNULL(data_node);
  const auto data_out_anchor = data_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(data_out_anchor);
  const auto node_in_anchor = data_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(node_in_anchor);
  in_anchor_idx = node_in_anchor->GetIdx();
  return node_in_anchor->GetOwnerNode();
}

//                     __________________
//                    | FusedAscBackend  |
//                    |     Data   Data  |
//                    |      |      |    |
//                    |      A      B    |
//         Node1 ---  |       \    /     |
//         /  \       |     NetOutput    |
//      Node2 Node3   |__________________|
// Node1是AscBackend时直接取Node1的属性，Node1是FusedAscBackend时，获取子图内部和Node2对应的节点属性
// 如：Node2获取到的是节点A的属性，Node3获取到的是节点B的属性
AutoFuseAttrs *BackendUtils::GetAscGraphOutputAttr(const NodePtr &node1, const NodePtr &node2) {
  auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  // 获取子图内部和node2的对应的节点
  if (node1->GetType() == kFusedAscBackendType) {
    int32_t anchor_index = -1;
    for (const auto node1_out_anchor : node1->GetAllOutDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(node1_out_anchor);
      for (const auto &node1_peer_anchor : node1_out_anchor->GetPeerInDataAnchorsPtr()) {
        if (node2 == node1_peer_anchor->GetOwnerNode()) {
          anchor_index = node1_out_anchor->GetIdx();
          GELOGI("node1:%s out:%d --> node2:%s.", node1->GetNamePtr(), anchor_index, node2->GetNamePtr());
          break;
        }
      }
      if (anchor_index != -1) {
        break;
      }
    }

    if (anchor_index != -1) {
      ComputeGraphPtr graph;
      GetNodeFusedGraph(node1, graph);
      GE_ASSERT_NOTNULL(graph);
      const auto netoutput = graph->GetOrUpdateNetOutputNode();
      GE_ASSERT_NOTNULL(netoutput);
      auto netoutput_in_anchor = netoutput->GetInDataAnchor(anchor_index);
      GE_ASSERT_NOTNULL(netoutput_in_anchor);
      const auto &netoutput_peer_out_anchor = netoutput_in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(netoutput_peer_out_anchor);
      auto asc_node = netoutput_peer_out_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(asc_node);
      attr1 = BackendUtils::GetNodeAutoFuseAttr(asc_node);
      GE_ASSERT_NOTNULL(attr1);
      GELOGI("Get %s's %p attr %p.", asc_node->GetNamePtr(), asc_node.get(), attr1);
    }
  }
  return attr1;
}

bool BackendUtils::CanFuseByStrategy(const NodePtr &node1, const NodePtr &node2, uint32_t &max_fusion_node_input_size) {
  if (!CanFuseByStrategyPro(node1, node2, max_fusion_node_input_size)) {
    return false;
  }
  auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  if ((attr1->GetFuseType() == loop::FuseType::kSplit) && IsSplitComplete(node1) &&
      IsSplitLowFusionRatio(node1, max_fusion_node_input_size)) {
      // Split类型不支持低融合比例, 当前只支持split后向融合
      GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 or node2 has low fuse ratio]",
             node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
             ge::NotFuseReasonCode(ge::NotFuseReason::kSplitLowFuseRatio));
      return false;
  }
  return true;
}

bool BackendUtils::CanMergeLoopByStrategy(const NodePtr &node1, const NodePtr &node2) {
  const auto fuse_type = BackendUtils::GetAllFuseType(node1, node2);
  for (const auto fusion_strategy : FusionStrategyRegistry::Instance().Get(fuse_type)) {
    if (fusion_strategy != nullptr) {
      if (!fusion_strategy->CanMergeLoop(node1, node2)) {
        return false;
      }
    }
  }
  return true;
}

uint64_t BackendUtils::GetAllFuseType(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  const auto fuse_type1 = attr1->GetAllFuseType();
  const auto fuse_type2 = attr2->GetAllFuseType();
  const auto fuse_type = fuse_type1 | fuse_type2;
  GELOGD("node1:%s and node2:%s fuse_type:%u.", node1->GetNamePtr(), node2->GetNamePtr(), fuse_type);
  return fuse_type;
}

bool BackendUtils::IsVertical(const NodePtr &node1, const NodePtr &node2) {
  for (const auto node1_out_anchor : node1->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(node1_out_anchor);
    for (const auto &node1_peer_anchor : node1_out_anchor->GetPeerInDataAnchorsPtr()) {
      if (node2 == node1_peer_anchor->GetOwnerNode()) {
        return true;
      }
    }
  }
  return false;
}

Status BackendUtils::ResetFusedSubgraphOutputsAttr(const NodePtr &node) {
  // 如果ascbackend和fusedascbackend融合，ascbackend原来依赖输出一个引用，fusedascbackend链接的这个输入可能是多个引用，这个时候如果需要
  // ascbackend被融合进fusedascbackend，fused_subgraph_outputs就不对了，需要重置fused_subgraph_outputs属性
  const auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  auto output_nodes = GetInterAttrs(autofuse_attr).fused_subgraph_outputs;
  asc_adapt::RemoveDuplicates(output_nodes);  // output_nodes去重
  const auto node_out_data_size = node->GetAllOutDataAnchorsSize();
  std::vector<ge::NodePtr> new_outputs;
  GE_ASSERT_TRUE(static_cast<size_t>(node_out_data_size) <= output_nodes.size(), "size %zu VS size %zu",
                 static_cast<size_t>(node_out_data_size), output_nodes.size());
  for (auto node_output = 0U; node_output < node_out_data_size; node_output++) {
    const auto node_out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(node_output));
    GE_ASSERT_NOTNULL(node_out_anchor);
    const auto &node_peer_anchors = node_out_anchor->GetPeerInDataAnchors();
    for (size_t i = 0U; i < node_peer_anchors.size(); i++) {
      new_outputs.push_back(output_nodes.at(node_output));
    }
    if (node_peer_anchors.empty()) {
      new_outputs.push_back(output_nodes.at(node_output));
    }
  }
  auto old_output_nodes = GetInterAttrs(autofuse_attr).fused_subgraph_outputs;
  if ((node->GetType() == kFusedAscBackendType) && (new_outputs.size() < old_output_nodes.size())) {
    GELOGD("node %s(%s) old_output_nodes:%s, output_nodes(remove dup):%s, new_outputs:%s.", node->GetNamePtr(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(old_output_nodes).c_str(),
           AutofuseUtils::VectorToStr(output_nodes).c_str(), AutofuseUtils::VectorToStr(new_outputs).c_str());
    /**
     * FusedAscBackend节点外部输出的接收节点水平融合导致实际需要输出节点减少，需要更新FusedAscBackend自身netoutput的输入与图外输出对应
     */
    std::vector<int32_t> in_data_anchor_indexes;

    // 比较new_outputs和原始output_nodes 找到相同的index
    size_t offset = 0U;
    for (const auto &output : new_outputs) {
      while ((offset < old_output_nodes.size()) && (output != old_output_nodes[offset])) {
        offset++;
      }
      if (offset < old_output_nodes.size()) {
        in_data_anchor_indexes.emplace_back(offset);
        offset++;
      }
    }
    GELOGD("node %s(%s) UpdateFusedAscBackendNetOutput in_data_anchor_indexes:%s.", node->GetNamePtr(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(in_data_anchor_indexes).c_str());
    UpdateFusedAscBackendNetOutputReally(autofuse_attr->GetFuseComputeGraph(), in_data_anchor_indexes);
  }
  GetInterAttrs(autofuse_attr).fused_subgraph_outputs = new_outputs;
  GELOGI("reset node %s(%s) fused subgraph outputs attr success, new fused subgraph outputs size: %zu.",
         node->GetName().c_str(), node->GetType().c_str(), new_outputs.size());
  return SUCCESS;
}

Status BackendUtils::GetPreNodeAndAnchor(const NodePtr &node, const int32_t index, NodePtr &peer_node,
                                         InDataAnchorPtr &in_anchor) {
  GE_ASSERT_TRUE(node->GetAllInDataAnchorsSize() > static_cast<size_t>(index));
  in_anchor = node->GetAllInDataAnchors().at(index);
  GE_ASSERT_NOTNULL(in_anchor);
  const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  peer_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_node);
  GELOGD("get node %s(%s) pre node %s(%s).", node->GetNamePtr(), node->GetType().c_str(), peer_node->GetNamePtr(),
         peer_node->GetType().c_str());
  return SUCCESS;
}

Status BackendUtils::GetSubgraphOutputIndex(const NodePtr &parent_node, const InDataAnchorPtr &peer_in_anchor,
                                            uint32_t &anchor_index, uint32_t &node_index) {
  anchor_index = 0U;
  node_index = 0U;
  for (auto node_output = 0U; node_output < parent_node->GetAllOutDataAnchorsSize(); node_output++) {
    const auto node_out_anchor = parent_node->GetOutDataAnchor(static_cast<int32_t>(node_output));
    GE_ASSERT_NOTNULL(node_out_anchor);
    for (const auto &node_peer_anchor : node_out_anchor->GetPeerInDataAnchors()) {
      if (node_peer_anchor == peer_in_anchor) {
        return SUCCESS;
      }
      node_index++;
    }
    if (node_out_anchor->GetPeerInDataAnchors().empty()) {
      node_index++;
    }
    anchor_index++;
  }
  return FAILED;
}

Status BackendUtils::GetPreAscNode(const NodePtr &parent_node, const InDataAnchorPtr &peer_in_anchor,
                                   NodePtr &asc_node) {
  auto anchor_index = 0U;
  auto node_index = 0U;
  GE_ASSERT_SUCCESS(GetSubgraphOutputIndex(parent_node, peer_in_anchor, anchor_index, node_index));
  auto autofuse_attr = GetNodeAutoFuseAttr(parent_node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  GELOGI("node: %s, output num=%zu, node_index=%u.", parent_node->GetName().c_str(),
         GetInterAttrs(autofuse_attr).fused_subgraph_outputs.size(), node_index);
  const auto &subgraph_output_nodes = GetInterAttrs(autofuse_attr).fused_subgraph_outputs;
  GE_ASSERT_TRUE(subgraph_output_nodes.size() > node_index);
  auto output_node = subgraph_output_nodes.at(node_index);
  auto asc_node_in_anchor = output_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(asc_node_in_anchor);
  auto in_anchor_peer = asc_node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(in_anchor_peer);
  asc_node = in_anchor_peer->GetOwnerNode();
  GELOGD("get pre asc node %s(%s) node_index=%u.", asc_node->GetNamePtr(), asc_node->GetType().c_str(), node_index);
  return SUCCESS;
}

Status BackendUtils::GetPreAscNodeAttrs(const NodePtr &parent_node, const InDataAnchorPtr &peer_in_anchor,
                                        std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats) {
  NodePtr pre_asc_node;
  GE_ASSERT_SUCCESS(BackendUtils::GetPreAscNode(parent_node, peer_in_anchor, pre_asc_node));
  GE_ASSERT_NOTNULL(pre_asc_node);
  auto pre_asc_node_op_desc = pre_asc_node->GetOpDesc();
  GE_ASSERT_NOTNULL(pre_asc_node_op_desc);
  auto pre_asc_node_output_desc = pre_asc_node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(pre_asc_node_output_desc);
  auto pre_asc_node_output_tensor_attr = pre_asc_node_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(pre_asc_node_output_tensor_attr);
  repeats = pre_asc_node_output_tensor_attr->repeats;
  axis = pre_asc_node_output_tensor_attr->axis;
  GE_ASSERT_TRUE(axis.size() == repeats.size());
  return SUCCESS;
}

Status BackendUtils::GetPreStoreNode(const NodePtr &node, const int32_t index, NodePtr &store_node) {
  NodePtr peer_node;
  InDataAnchorPtr in_anchor;
  GE_ASSERT_SUCCESS(GetPreNodeAndAnchor(node, index, peer_node, in_anchor));
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_node, graph));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph, peer_node));
  GE_ASSERT_SUCCESS(BackendUtils::GetPreAscNode(peer_node, in_anchor, store_node));
  GE_ASSERT_NOTNULL(store_node);
  return SUCCESS;
}

// 通过两个调度轴的对比找出transpose轴信息
Status BackendUtils::GetTransposeInfo(NodePtr &node, std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  std::vector<int64_t> graph_axis;
  const auto compute_graph = node->GetOwnerComputeGraph();
  auto get_graph_attr = [](const std::vector<AxisPtr> &axis_info_list, std::vector<int64_t> &axis) -> Status {
    for (auto &axis_info : axis_info_list) {
      axis.push_back(axis_info->id);
    }
    return SUCCESS;
  };
  auto graph_attr = compute_graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  GE_ASSERT_SUCCESS(get_graph_attr(graph_attr->axis, graph_axis));

  auto node_op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_output_desc = node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(node_output_desc);
  auto node_tensor_attr = node_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(node_tensor_attr);
  GELOGD("node name:%s(%s) sched axis(%s), graph axis(%s).", node->GetName().c_str(), node->GetType().c_str(),
         AutofuseUtils::VectorToStr(node_tensor_attr->axis).c_str(), AutofuseUtils::VectorToStr(graph_axis).c_str());

  int64_t swap_count = 0;
  GE_ASSERT_SUCCESS(MinSwapCount(graph_axis, node_tensor_attr->axis, swap_count, transpose_info));
  return SUCCESS;
}

Status BackendUtils::GetViewOpNextNodeByLoad(const NodePtr &load_node, NodePtr &finded_node) {
  NodePtr cur_node = load_node;
  while ((cur_node->GetType() == kLoadType) || (cur_node->GetType() == kBroadcastType) ||
         (cur_node->GetType() == kGatherType)) {
    auto cur_out_anchor = cur_node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(cur_out_anchor);
    auto out_size = cur_out_anchor->GetPeerInDataAnchors().size();
    GE_ASSERT_TRUE(out_size > 0U);
    auto cur_out_anchor_peer = cur_out_anchor->GetPeerInDataAnchors().at(0);
    GE_ASSERT_NOTNULL(cur_out_anchor_peer);
    const auto next_node = cur_out_anchor_peer->GetOwnerNode();
    GE_ASSERT_NOTNULL(next_node);

    if ((cur_node->GetType() == kBroadcastType)) {
      finded_node = cur_node;
      GELOGD("finded node name:%s(%s).", cur_node->GetName().c_str(), cur_node->GetType().c_str());
      break;
    }
    cur_node = next_node;
  }
  return SUCCESS;
}

Status BackendUtils::ApplyTransposeOp(const ComputeGraphPtr &graph,
                                      const std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  for (auto &asc_node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(asc_node);
    auto asc_node_op_desc = asc_node->GetOpDesc();
    GE_ASSERT_NOTNULL(asc_node_op_desc);
    AscNodeAttr *asc_node_attr = asc_node_op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(asc_node_attr);
    GE_ASSERT_SUCCESS(SwapSchedAxis(transpose_info, asc_node_attr->sched.axis));
    if ((asc_node->GetType() == kLoadType) || (asc_node->GetType() == kStoreType) ||
        (asc_node->GetType() == kOutputType) || (asc_node->GetType() == kDataType) ||
        (asc_node->GetType() == kGatherType)) {
      continue;
    }
    for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_desc_tensor_attr);
      GE_ASSERT_TRUE(output_desc_tensor_attr->axis.size() == output_desc_tensor_attr->repeats.size());
      GE_ASSERT_TRUE(output_desc_tensor_attr->axis.size() == output_desc_tensor_attr->strides.size());
      GELOGD("swap before tensor: repeats(%s), strides(%s), axis(%s).",
             AutofuseUtils::VectorToStr(output_desc_tensor_attr->repeats).c_str(),
             AutofuseUtils::VectorToStr(output_desc_tensor_attr->strides).c_str(),
             AutofuseUtils::VectorToStr(output_desc_tensor_attr->axis).c_str());
      auto tmp_axis1 = output_desc_tensor_attr->axis;
      GE_ASSERT_SUCCESS(
          SwapTensorInfo(transpose_info, tmp_axis1, output_desc_tensor_attr->repeats));
      auto tmp_axis2 = output_desc_tensor_attr->axis;
      GE_ASSERT_SUCCESS(
          SwapTensorInfo(transpose_info, tmp_axis2, output_desc_tensor_attr->strides));
      GE_ASSERT_SUCCESS(UpdateContinueStrides(output_desc_tensor_attr->repeats, output_desc_tensor_attr->strides));
      GE_ASSERT_SUCCESS(SwapSchedAxis(transpose_info, output_desc_tensor_attr->axis));
      GELOGD("swap after tensor: repeats(%s), strides(%s), axis(%s).",
             AutofuseUtils::VectorToStr(output_desc_tensor_attr->repeats).c_str(),
             AutofuseUtils::VectorToStr(output_desc_tensor_attr->strides).c_str(),
             AutofuseUtils::VectorToStr(output_desc_tensor_attr->axis).c_str());
    }
  }
  auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  for (auto &axis_info : graph_attr->axis) {
    GELOGD("swap before axis info: axis name(%s), axis id(%ld), axis size(%s).", axis_info->name.c_str(), axis_info->id,
           std::string(axis_info->size.Str().get()).c_str());
  }
  GE_ASSERT_SUCCESS(SwapGraphAxis(transpose_info, graph_attr->axis));
  for (auto &axis_info : graph_attr->axis) {
    GELOGD("swap after axis info: axis name(%s), axis id(%ld), axis size(%s).", axis_info->name.c_str(), axis_info->id,
           std::string(axis_info->size.Str().get()).c_str());
  }
  return SUCCESS;
}

Status BackendUtils::RemoveTransposeOp(const ComputeGraphPtr &graph) {
  std::vector<NodePtr> del_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == kTransposeType) {
      del_nodes.push_back(node);
    }
  }
  for (const auto &node : del_nodes) {
    const auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    const auto src_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_anchor);
    GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::RemoveEdge(src_anchor, in_anchor));
    const auto out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(out_anchor);
    for (const auto &dst_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::RemoveEdge(out_anchor, dst_anchor));
      GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(src_anchor, dst_anchor));
    }

    GELOGD("Remove node %s %s from asc_graph:%s.", node->GetName().c_str(), node->GetType().c_str(),
           graph->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node), "[Remove][JustNode] failed, graph:%s, node:%s.",
                            graph->GetName().c_str(), node->GetName().c_str());
    NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

Status BackendUtils::UpdateBroadcastBeforeMerge(const NodePtr &node1, const NodePtr &node2) {
  auto asc_graph1 = BackendUtils::GetNodeFusedAscGraph(node1);
  GE_ASSERT_NOTNULL(asc_graph1);
  GE_ASSERT_SUCCESS(BackendUtils::UpdateBroadcastInfoToLoad(*asc_graph1));
  BackendUtils::DumpAscGraph(node1);
  auto asc_graph2 = BackendUtils::GetNodeFusedAscGraph(node2);
  GE_ASSERT_NOTNULL(asc_graph2);
  GE_ASSERT_SUCCESS(BackendUtils::UpdateBroadcastInfoToLoad(*asc_graph2));
  BackendUtils::DumpAscGraph(node2);
  return SUCCESS;
}

Status BackendUtils::UpdateTransposeBeforeMerge(const NodePtr &node2, const ComputeGraphPtr &graph1,
                                                const ComputeGraphPtr &graph2, const NodeFuseInfo &fuse_info) {
  std::vector<std::pair<int64_t, int64_t>> transpose_info;
  NodePtr load_node;
  NodePtr store_node;
  if (!fuse_info.GetNode1ToNode2LinkMap().empty()) {
    for (const auto &subgraph_link : fuse_info.GetNode1ToNode2LinkMap()) {
      // 先保证前序node的store和循环轴一致
      GE_ASSERT_SUCCESS(GetPreStoreNode(node2, subgraph_link.second, store_node));
      GE_ASSERT_SUCCESS(GetTransposeInfo(store_node, transpose_info));
      if (!transpose_info.empty()) {
        GE_ASSERT_SUCCESS(ApplyTransposeOp(graph1, transpose_info));
      }
      transpose_info.clear();
      // 把后续的transpose更新到前序
      GE_ASSERT_SUCCESS(GetCurAscLoadNode(graph2, subgraph_link.second, load_node));
      GE_ASSERT_SUCCESS(GetTransposeInfo(load_node, transpose_info));
      GELOGD("graph %s get transpose info %s.", graph2->GetName().c_str(),
             AutofuseUtils::VectorPairToStr(transpose_info).c_str());
      if (transpose_info.empty()) {
        GE_ASSERT_SUCCESS(RemoveTransposeOp(graph1));
        GE_ASSERT_SUCCESS(RemoveTransposeOp(graph2));
        return SUCCESS;
      }
      // 校验transpose能否在需要变化的graph上找到对应的轴
      if (!CheckValidTranspose(graph1, transpose_info)) {
        GELOGD("graph %s can't find transpose axis by graph %s.", graph1->GetName().c_str(), graph2->GetName().c_str());
        return FAILED;
      }
      GE_ASSERT_SUCCESS(ApplyTransposeOp(graph2, transpose_info));
      GE_ASSERT_SUCCESS(RemoveTransposeOp(graph1));
      GE_ASSERT_SUCCESS(RemoveTransposeOp(graph2));
      // 只做一次变换
      break;
    }
  } else {
    // 水平融合是两个图都需要变换
    for (const auto &same_input : fuse_info.GetSameInputMap()) {
      auto process_transpose = [&load_node, &transpose_info](const ComputeGraphPtr &graph,
                                                             const int32_t input_index) -> Status {
        transpose_info.clear();
        GE_ASSERT_SUCCESS(GetCurAscLoadNode(graph, input_index, load_node));
        GE_ASSERT_SUCCESS(GetTransposeInfo(load_node, transpose_info));
        GELOGD("graph %s get transpose info %s.", graph->GetName().c_str(),
               AutofuseUtils::VectorPairToStr(transpose_info).c_str());
        if (!transpose_info.empty()) {
          GE_ASSERT_SUCCESS(ApplyTransposeOp(graph, transpose_info));
          GE_ASSERT_SUCCESS(RemoveTransposeOp(graph));
        }
        return SUCCESS;
      };
      GE_ASSERT_SUCCESS(process_transpose(graph1, same_input.first));
      GE_ASSERT_SUCCESS(process_transpose(graph2, same_input.second));
      // 只做一次变换
      break;
    }
  }
  return SUCCESS;
}

Status CompleteNodeAttrsBeforeMerge(const NodePtr &node1, const NodePtr &node2) {
  auto asc_graph1 = BackendUtils::GetNodeFusedAscGraph(node1);
  GE_ASSERT_NOTNULL(asc_graph1);
  GE_ASSERT_SUCCESS(asc_adapt::CompleteNodeAttrsOnAscGraph(*asc_graph1, node1));
  auto asc_graph2 = BackendUtils::GetNodeFusedAscGraph(node2);
  GE_ASSERT_NOTNULL(asc_graph2);
  GE_ASSERT_SUCCESS(asc_adapt::CompleteNodeAttrsOnAscGraph(*asc_graph2, node2));
  return SUCCESS;
}

// 根据graph的链接关系变换ascgraph循环轴，如果有垂直融合通过变换前序graph的循环轴，load不变，其他都变换；
// 水平融合变换两个graph
Status BackendUtils::TuningSubgraphBeforeMerge(const NodePtr &node1, const NodePtr &node2,
                                               const ComputeGraphPtr &graph1, const ComputeGraphPtr &graph2,
                                               const NodeFuseInfo &fuse_info) {
  // 为了给未参与过融合的graph的load后的空轴计算结点，能正确按照graph补轴，参与过融合的graph缺的轴repeat和stride就补1和0
  GE_ASSERT_SUCCESS(CompleteNodeAttrsBeforeMerge(node1, node2));

  // 为了broadcast水平和垂直合并，融合前把broadcast消除
  GE_ASSERT_SUCCESS(UpdateBroadcastBeforeMerge(node1, node2));

  GE_ASSERT_SUCCESS(UpdateTransposeBeforeMerge(node2, graph1, graph2, fuse_info));

  return SUCCESS;
}

Status BackendUtils::ProcessAscgraphAfterMerge(const NodePtr &new_node) {
  auto asc_graph = BackendUtils::GetNodeFusedAscGraph(new_node);
  GE_ASSERT_NOTNULL(asc_graph);
  GE_ASSERT_SUCCESS(asc_adapt::FallbackPro(*asc_graph, new_node));
  GE_ASSERT_SUCCESS(asc_adapt::CompleteNodeAttrsOnAscGraph(*asc_graph, new_node));
  GELOGD("dump attr after merged node:%s(%s) asc subgraph info:", new_node->GetNamePtr(),
         new_node->GetType().c_str());
  BackendUtils::DumpAscGraph(new_node);
  return SUCCESS;
}

Status BackendUtils::GetNodeTensorAttrInfo(const NodePtr &node, TensorAttrInfo &tensor_attr) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);
  const auto &output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  const auto &output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  tensor_attr.topo_id = op_desc->GetId();
  tensor_attr.sched_axis = node_attr->sched.axis;
  tensor_attr.axis = output_attr->axis;
  tensor_attr.repeats = output_attr->repeats;
  tensor_attr.strides = output_attr->strides;
  tensor_attr.dtype = output_attr->dtype;
  return SUCCESS;
}

Status BackendUtils::GetGraphAttrInfo(const AscGraph &asc_graph, TensorAttrInfo &current_node_attr) {
  const auto graph_attr = AscGraphUtils::GetComputeGraph(asc_graph)->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  for (const auto &axis_info : graph_attr->axis) {
    current_node_attr.axis.push_back(axis_info->id);
    current_node_attr.repeats.push_back(axis_info->size);
    current_node_attr.sched_axis.push_back(axis_info->id);
  }
  GE_ASSERT_TRUE(current_node_attr.axis.size() == current_node_attr.repeats.size());
  ge::Expression tmpe_stride = kSymbolOne;
  for (size_t i = current_node_attr.axis.size(); i > 0U; --i) {
    if (BackendUtils::IsEqOne(current_node_attr.repeats[i - 1U])) {
      current_node_attr.strides.insert(current_node_attr.strides.begin(), kSymbolZero);
    } else {
      current_node_attr.strides.insert(current_node_attr.strides.begin(), tmpe_stride);
      tmpe_stride = current_node_attr.repeats[i - 1U] * tmpe_stride;
    }
  }
  GELOGI("asc_graph %s, get tensor info: axis:%s, repeats:%s, strides:%s, sched_axis:%s.", asc_graph.GetName().c_str(),
         AutofuseUtils::VectorToStr(current_node_attr.axis).c_str(),
         AutofuseUtils::VectorToStr(current_node_attr.repeats).c_str(),
         AutofuseUtils::VectorToStr(current_node_attr.strides).c_str(),
         AutofuseUtils::VectorToStr(current_node_attr.sched_axis).c_str());
  return SUCCESS;
}

bool BackendUtils::IsSameAttrInfoInVector(std::vector<ViewOpAttrInfo> &attr_infos) {
  for (auto &attr_info : attr_infos) {
    std::sort(attr_info.broadcast_info.begin(), attr_info.broadcast_info.end());
  }
  if (attr_infos.size() <= 1U) {
    return true;
  }
  // 判断各个vector是否都一样
  const std::vector<int64_t> broadcast_info = attr_infos[0].broadcast_info;
  for (size_t i = 1U; i < attr_infos.size(); ++i) {
    if (attr_infos[i].broadcast_info != broadcast_info) {
      return false;
    }
  }
  return true;
}

bool BackendUtils::IsSameBroadCastInfo(std::vector<ViewOpAttrInfo> &attr_infos1,
                                       std::vector<ViewOpAttrInfo> &attr_infos2) {
  // 1.attr_infos1和attr_infos2中broadcast info都为空，返回true
  if (attr_infos1.empty() && attr_infos2.empty()) {
    return true;
  }
  // 2.attr_infos1和attr_infos2其中一个为空，返回false
  if ((attr_infos1.empty() && !attr_infos2.empty()) || (!attr_infos1.empty() && attr_infos2.empty())) {
    return false;
  }
  // 3.attr_infos1中各元素中的broadcast info相同，不同就返回false
  if (!BackendUtils::IsSameAttrInfoInVector(attr_infos1)) {
    return false;
  }
  // 4.attr_infos2中各元素中的broadcast info相同，不同就返回false
  if (!BackendUtils::IsSameAttrInfoInVector(attr_infos2)) {
    return false;
  }
  // 5.attr_infos1和attr_infos2都不为空，对比attr_infos1和attr_infos2中broadcast info是否相同，如果不同返回false，否则返回true
  const std::vector<int64_t> broadcast_info1 = attr_infos1[0].broadcast_info;
  const std::vector<int64_t> broadcast_info2 = attr_infos2[0].broadcast_info;
  return broadcast_info1 == broadcast_info2;
}

bool BackendUtils::IsNodeAllInputsAreSimplestLoad(const NodePtr &node) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  const auto subgraph_input_nodes = graph->GetInputNodes();
  if (node->GetType() == kAscBackendType) {
    for (auto index = 0U; index < subgraph_input_nodes.size(); ++index) {
      GE_ASSERT_TRUE(subgraph_input_nodes.size() > index);
      const auto data_node = subgraph_input_nodes.at(index);
      GE_ASSERT_NOTNULL(data_node);
      const auto data_out_anchor = data_node->GetOutDataAnchor(0);
      GE_ASSERT_NOTNULL(data_out_anchor);
      for (const auto &node_peer_anchor : data_out_anchor->GetPeerInDataAnchors()) {
        const auto load_node = node_peer_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(load_node);
        std::vector<ViewOpAttrInfo> attr_infos;
        if (!BackendUtils::IsSimplestLoad(node, load_node, data_node, attr_infos)) {
          GELOGD("node %s(%s) ascgraph data node %s(%s) have view op.", node->GetNamePtr(), node->GetType().c_str(),
                 data_node->GetNamePtr(), data_node->GetType().c_str());
          return false;
        }
      }
    }
    return true;
  } else if (node->GetType() == kFusedAscBackendType) {
    int32_t in_anchor_idx;
    for (auto index = 0U; index < subgraph_input_nodes.size(); ++index) {
      const auto asc_node = BackendUtils::GetFusedAscBackendInputNode(node, static_cast<int32_t>(index), in_anchor_idx);
      GE_ASSERT_NOTNULL(asc_node);
      if (!IsNodeAllInputsAreSimplestLoad(asc_node)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool BackendUtils::CurNodeInputIsSimplestLoad(const NodePtr &node, const int32_t index,
                                              std::vector<ViewOpAttrInfo> &attr_infos,
                                              bool is_condition_with_node_type) {
  if (node->GetType() == kAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
    const auto subgraph_input_nodes = graph->GetInputNodes();
    GE_ASSERT_TRUE(subgraph_input_nodes.size() > static_cast<size_t>(index));
    const auto data_node = subgraph_input_nodes.at(index);
    GE_ASSERT_NOTNULL(data_node);
    const auto data_out_anchor = data_node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(data_out_anchor);
    for (const auto &node_peer_anchor : data_out_anchor->GetPeerInDataAnchors()) {
      const auto load_node = node_peer_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(load_node);
      if (!BackendUtils::IsSimplestLoad(node, load_node, data_node, attr_infos, is_condition_with_node_type)) {
        GELOGD("cur node %s(%s) ascgraph data node %s(%s) have view op.", node->GetNamePtr(), node->GetType().c_str(),
               data_node->GetNamePtr(), data_node->GetType().c_str());
        return false;
      }
    }
    return true;
  } else if (node->GetType() == kFusedAscBackendType) {
    int32_t in_anchor_idx;
    const auto asc_node = BackendUtils::GetFusedAscBackendInputNode(node, index, in_anchor_idx);
    GE_ASSERT_NOTNULL(asc_node);
    return CurNodeInputIsSimplestLoad(asc_node, in_anchor_idx, attr_infos);
  }
  return false;
}

bool BackendUtils::AscNodeInputIsSimplestLoad(const NodePtr &peer_node, const InDataAnchorPtr &in_anchor,
                                              std::vector<ViewOpAttrInfo> &attr_infos) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_node, graph));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph, peer_node));
  NodePtr asc_store_node;
  GE_ASSERT_SUCCESS(BackendUtils::GetPreAscNode(peer_node, in_anchor, asc_store_node));
  std::vector<NodePtr> load_nodes;
  auto store_in_anchor = asc_store_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(store_in_anchor);
  auto in_anchor_peer = store_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(in_anchor_peer);
  auto store_peer_node = in_anchor_peer->GetOwnerNode();
  GE_ASSERT_SUCCESS(BackendUtils::FindPrefaceLoadNodes(store_peer_node->GetOutDataAnchor(0), load_nodes));
  for (const auto &pre_load_node : load_nodes) {
    const auto &load_in_anchor = pre_load_node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(load_in_anchor);
    const auto node_out_anchor = load_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(node_out_anchor);
    const auto pre_data_node = node_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(pre_data_node);
    if (!IsSimplestLoad(peer_node, pre_load_node, pre_data_node, attr_infos)) {
      GELOGD("cur node %s(%s) ascgraph date node %s(%s) have view op.", peer_node->GetNamePtr(),
                peer_node->GetType().c_str(), pre_data_node->GetNamePtr(), pre_data_node->GetType().c_str());
      return false;
    }
  }
  return true;
}

Status BackendUtils::GetPreAscBackendNodeAndAnchor(const NodePtr &node, const NodePtr &peer_node,
                                                   const InDataAnchorPtr &fused_in_anchor, NodePtr &asc_node,
                                                   InDataAnchorPtr &netoutput_in_anchor) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_node, graph));
  auto anchor_index = 0U;
  auto node_index = 0U;
  GE_ASSERT_SUCCESS(BackendUtils::GetSubgraphOutputIndex(peer_node, fused_in_anchor, anchor_index, node_index));
  const auto netoutput = graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netoutput);
  netoutput_in_anchor = netoutput->GetInDataAnchor(static_cast<int32_t>(anchor_index));
  GE_ASSERT_NOTNULL(netoutput_in_anchor);
  const auto &netoutput_peer_out_anchor = netoutput_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(netoutput_peer_out_anchor);
  asc_node = netoutput_peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(asc_node);
  GELOGD("get node %s(%s) pre ascbackend node %s(%s).", node->GetNamePtr(), node->GetType().c_str(),
            asc_node->GetNamePtr(), asc_node->GetType().c_str());
  return SUCCESS;
}

bool BackendUtils::PreNodeInputIsSimplestLoad(const NodePtr &node, const int32_t index,
                                              std::vector<ViewOpAttrInfo> &attr_infos) {
  NodePtr peer_node;
  InDataAnchorPtr in_anchor;
  GE_ASSERT_SUCCESS(BackendUtils::GetPreNodeAndAnchor(node, index, peer_node, in_anchor));
  if (peer_node->GetType() == kAscBackendType) {
    return BackendUtils::AscNodeInputIsSimplestLoad(peer_node, in_anchor, attr_infos);
  } else if (peer_node->GetType() == kFusedAscBackendType) {
    NodePtr asc_node;
    InDataAnchorPtr netoutput_in_anchor;
    GE_ASSERT_SUCCESS(BackendUtils::GetPreAscBackendNodeAndAnchor(node, peer_node, in_anchor, asc_node, netoutput_in_anchor));
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(asc_node, graph));
    GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph, asc_node));
    return BackendUtils::AscNodeInputIsSimplestLoad(asc_node, netoutput_in_anchor, attr_infos);
  }
  return false;
}

Status BackendUtils::BackupNodeAscTensorAttrAndAscNodeAttr(const NodePtr &origin_node,
                                                           NodeAttrPair &backup_node_attr_and_tensor_attr) {
  auto origin_op_desc = origin_node->GetOpDesc();
  GE_ASSERT_NOTNULL(origin_op_desc);
  auto origin_output_desc = origin_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(origin_output_desc);
  AscTensorAttr *origin_output_attr = origin_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(origin_output_attr);
  AscNodeAttr *origin_node_attr = origin_op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(origin_node_attr);

  std::unique_ptr<AscTensorAttr> asc_tensor_attr = std::make_unique<AscTensorAttr>();
  GE_ASSERT_NOTNULL(asc_tensor_attr);
  asc_tensor_attr->repeats = origin_output_attr->repeats;
  asc_tensor_attr->strides = origin_output_attr->strides;
  asc_tensor_attr->axis = origin_output_attr->axis;

  std::unique_ptr<AscNodeAttr> asc_node_attr = std::make_unique<AscNodeAttr>();
  GE_ASSERT_NOTNULL(asc_node_attr);
  asc_node_attr->sched.axis = origin_node_attr->sched.axis;

  backup_node_attr_and_tensor_attr.first = std::move(asc_node_attr);
  backup_node_attr_and_tensor_attr.second = std::move(asc_tensor_attr);

  return SUCCESS;
}

Status BackendUtils::RecoverNodeAscTensorAttrAndAscNodeAttr(const NodePtr &load_node,
                                                            NodeAttrPair &backup_node_attr_and_tensor_attr) {
  auto op_desc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  AscTensorAttr *output_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  AscNodeAttr *node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);

  output_attr->repeats = backup_node_attr_and_tensor_attr.second->repeats;
  output_attr->strides = backup_node_attr_and_tensor_attr.second->strides;
  output_attr->axis = backup_node_attr_and_tensor_attr.second->axis;
  node_attr->sched.axis = backup_node_attr_and_tensor_attr.first->sched.axis;
  return SUCCESS;
}

bool BackendUtils::CheckSameSchedAxis(const NodePtr &node1, const NodePtr &node2, const AxisPairSet &node1_map,
                                      const AxisPairSet &node2_map, const NodeFuseInfo &node_fuse_info) {
  const auto fuse_type = BackendUtils::GetAllFuseType(node1, node2);
  for (const auto fusion_strategy : FusionStrategyRegistry::Instance().Get(fuse_type)) {
    if (fusion_strategy != nullptr) {
      if (!fusion_strategy->CheckSameSchedAxis(node1, node2, node1_map, node2_map, node_fuse_info)) {
        return false;
      }
    }
  }
  return true;
}

Status BackendUtils::GetScheduleAxisInfo(const NodePtr &node, int32_t input_index, const AxisPairSet &node_map,
                                         AxisIdAndSizePair &axis_id_and_size_pair_list) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  auto b_node = node;
  if (node->GetType() == kFusedAscBackendType) {
    GE_ASSERT_TRUE(static_cast<size_t>(input_index) < graph->GetInputNodes().size(),
                   "input index %zu >= graph input size %zu", static_cast<size_t>(input_index),
                   graph->GetInputNodes().size());
    const auto data_node = graph->GetInputNodes().at(input_index);
    GE_ASSERT_NOTNULL(data_node);
    b_node = BackendUtils::GetDataNextNode(data_node);
    GE_ASSERT_NOTNULL(b_node);
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(b_node, graph));
  }

  const auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  std::pair<int64_t, ge::Expression> axis_id_and_size_pair;
  for (auto &axis_info : graph_attr->axis) {
    int64_t axis_id = axis_info->id;
    if (BackendUtils::ConvertAxis(node_map, axis_id, true) != SUCCESS) {
      return FAILED;
    }
    axis_id_and_size_pair.first = axis_id;
    axis_id_and_size_pair.second = axis_info->size;
    axis_id_and_size_pair_list.push_back(axis_id_and_size_pair);
    GELOGD("  \nget graph axis info: axis name(%s), axis id(%ld), axis size(%s), graph name(%s).",
           axis_info->name.c_str(), axis_id, std::string(axis_info->size.Str().get()).c_str(),
           graph->GetName().c_str());
  }
  return SUCCESS;
}

bool BackendUtils::IsHorizontal(const NodePtr &node1, const NodePtr &node2) {
  for (auto node1_input = 0U; node1_input < node1->GetAllInDataAnchorsSize(); node1_input++) {
    for (auto node2_input = 0U; node2_input < node2->GetAllInDataAnchorsSize(); node2_input++) {
      const auto node1_in_anchor = node1->GetInDataAnchor(static_cast<int32_t>(node1_input));
      GE_ASSERT_NOTNULL(node1_in_anchor);
      const auto node1_peer_out_anchor = node1_in_anchor->GetPeerOutAnchor();
      if (node1_peer_out_anchor == nullptr) {
        continue;
      }
      const auto node2_in_anchor = node2->GetInDataAnchor(static_cast<int32_t>(node2_input));
      GE_ASSERT_NOTNULL(node2_in_anchor);
      const auto node2_peer_out_anchor = node2_in_anchor->GetPeerOutAnchor();
      if (node2_peer_out_anchor == nullptr) {
        continue;
      }
      if (node1_peer_out_anchor == node2_peer_out_anchor) {
        return true;
      }
    }
  }
  return false;
}

Status BackendUtils::DumpGraph(const std::string &graph_name, const std::string &path, const std::string &suffix) {
  auto &manager = FusionGraphManager::GetInstance();
  GE_ASSERT_SUCCESS(manager.DumpGraph(graph_name, path, suffix));
  return SUCCESS;
}

Status BackendUtils::AddMergeGraphMap(const std::string &new_node, const std::string &node1, const std::string &node2,
                                      const std::string &merged_graph) {
  auto &manager = FusionGraphManager::GetInstance();
  // 添加融合关系
  GE_ASSERT_SUCCESS(manager.AddMergeGraphMap(new_node, node1, node2));
  GE_ASSERT_SUCCESS(manager.AddMergeGraphMap(merged_graph, node1, node2));
  return SUCCESS;
}

Status BackendUtils::AddSubGraphMergeGraphMap(const std::string &new_node, const std::string &node1,
                                              const std::string &node2) {
  auto &manager = FusionGraphManager::GetInstance();
  // 添加融合关系
  GE_ASSERT_SUCCESS(manager.AddMergeGraphMap(new_node, node1, node2));
  return SUCCESS;
}

Status BackendUtils::CacheGraph(const NodePtr &node) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  auto &manager = FusionGraphManager::GetInstance();
  GE_ASSERT_SUCCESS(manager.CacheGraph(node->GetName(), graph));
  return SUCCESS;
}

Status BackendUtils::CacheGraph(const std::string &graph_name, const ComputeGraphPtr &graph) {
  auto &manager = FusionGraphManager::GetInstance();
  // 缓存图
  GE_ASSERT_SUCCESS(manager.CacheGraph(graph_name, graph));
  return SUCCESS;
}

Status BackendUtils::CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2,
                                           const std::string &origin_graph_name) {
  auto &manager = FusionGraphManager::GetInstance();
  // FusedAscBackend子图融合时缓存正在融合图的名字
  manager.CacheCurrentGraphName(graph_name1, graph_name2, origin_graph_name);
  return SUCCESS;
}

Status BackendUtils::CacheCurrentGraphName(const std::string &graph_name1, const std::string &graph_name2) {
  auto &manager = FusionGraphManager::GetInstance();
  // AscBackend图融合时缓存正在融合图的名字
  manager.CacheCurrentGraphName(graph_name1, graph_name2);
  return SUCCESS;
}

Status BackendUtils::DumpCurrentGraphAndSubgraphs() {
  auto &manager = FusionGraphManager::GetInstance();
  // dump CacheCurrentGraphName缓存的对应图
  GE_ASSERT_SUCCESS(manager.DumpCurrentGraphAndSubgraphs(kCanFuseDir));
  return SUCCESS;
}

Status BackendUtils::DumpGraphAndSubgraphs(const std::vector<std::string> &target_graphs, const std::string &path) {
  auto &manager = FusionGraphManager::GetInstance();
  // dump图
  GE_ASSERT_SUCCESS(manager.DumpGraphAndSubgraphs(target_graphs, path));
  return SUCCESS;
}

bool BackendUtils::IsCubeAscNode(const NodePtr &asc_node) {
  const auto &op_desc = asc_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  if (attr->HasFuseType(loop::FuseType::kCube)) {
    return true;
  }
  return false;
}

size_t BackendUtils::GetComputeNodeNumInAscgraph(const NodePtr &node) {
  const auto asc_graph = GetNodeFusedAscGraph(node);
  GE_ASSERT_NOTNULL(asc_graph);
  size_t compute_node_num = 0UL;
  for (const auto &sub_node : asc_graph->GetAllNodes()) {
    const auto node_type = sub_node->GetType();
    if (node_type == kDataType || node_type == kLoadType || node_type == kStoreType || node_type == kOutputType) {
      continue;
    }
    compute_node_num += 1UL;
  }
  return compute_node_num;
}

bool BackendUtils::IsAllInputFromSameNode(const NodePtr &node) {
  const auto in_anchor_size = node->GetAllInDataAnchorsSize();
  const auto target_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  GE_ASSERT_NOTNULL(target_node);
  for (auto i = 1U; i < in_anchor_size; ++i) {
    const auto parent_node = node->GetInDataAnchor(i)->GetPeerOutAnchor()->GetOwnerNode();
    if (parent_node != target_node) {
      GELOGD("node %s(%s) inputs(0, %d) come from different nodes", node->GetNamePtr(), node->GetType().c_str(), i);
      return false;
    }
  }
  GELOGD("node %s(%s) inputs come from same node", node->GetNamePtr(), node->GetType().c_str());
  return true;
}

bool BackendUtils::HasScalarInAscgraph(const NodePtr &node) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  for (const auto &sub_node : graph->GetAllNodes()) {
    if (sub_node->GetType() == kScalarType) {
      GELOGD("node %s(%s) ascgraph has scalar node", node->GetNamePtr(), node->GetType().c_str());
      return true;
    }
  }
  return false;
}

bool BackendUtils::HasTypesInAscgraph(const NodePtr &node, const std::vector<std::string> &target_types) {
  if (!target_types.empty()) {
    std::unordered_set<std::string> target_type_set(target_types.begin(), target_types.end());
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
    for (const auto &sub_node : graph->GetAllNodes()) {
      const std::string &node_type = sub_node->GetType();
      if (target_type_set.count(node_type) > 0) {
        GELOGD("node %s(%s) ascgraph has target type node: %s", node->GetNamePtr(), node->GetType().c_str(),
               node_type.c_str());
        return true;
      }
    }
  }
  return false;
}

bool BackendUtils::OnlyHasTypesInAscgraph(const NodePtr &node, const std::vector<std::string> &target_types) {
  if (!target_types.empty()) {
    std::vector<std::string> all_types = {kDataType, kLoadType, kStoreType, kOutputType};
    all_types.insert(all_types.end(), target_types.begin(), target_types.end());
    std::unordered_set<std::string> target_type_set(all_types.begin(), all_types.end());

    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
    for (const auto &sub_node : graph->GetAllNodes()) {
      const std::string &node_type = sub_node->GetType();
      if (target_type_set.count(node_type) == 0) {
        GELOGD("node %s(%s) ascgraph has unexpected type: %s", node->GetNamePtr(), node->GetType().c_str(), node_type.c_str());
        return false;
      }
    }
  }
  return true;
}

void BackendUtils::SetReduceOriginalAxisInfo(AutofuseInnerAttrs &attr_new, const AutofuseInnerAttrs &attr1,
                                             const AutofuseInnerAttrs &attr2) {
  // 目前业界暂不支持融合reduce和reduce，即融合的两个节点只有一个是reduce，后期如果要融合两个reduce则需要根据reduce节点名字保存对应的原始轴信息
  if (!attr1.reduce_original_axis.empty()) {
    attr_new.reduce_original_axis = attr1.reduce_original_axis;
  } else {
    attr_new.reduce_original_axis = attr2.reduce_original_axis;
  }
  if (!attr1.reduce_original_repeats.empty()) {
    attr_new.reduce_original_repeats = attr1.reduce_original_repeats;
  } else {
    attr_new.reduce_original_repeats = attr2.reduce_original_repeats;
  }
}

Status GetNodeTransposeInfo(const NodePtr &node, const TensorAttrInfo &temp_graph_attr,
                                     std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  const auto cur_op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(cur_op_desc);
  const auto cur_output_desc = cur_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(cur_output_desc);
  const auto cur_output_attr = cur_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(cur_output_attr);
  int64_t swap = 0;
  if ((node->GetType() == kLoadType) || (node->GetType() == kGatherType)) {
    GE_ASSERT_SUCCESS(BackendUtils::MinSwapCount(cur_output_attr->axis, temp_graph_attr.axis, swap, transpose_info));
  } else {  // if is store
    GE_ASSERT_SUCCESS(BackendUtils::MinSwapCount(temp_graph_attr.axis, cur_output_attr->axis, swap, transpose_info));
  }
  return SUCCESS;
}

bool IsFallBackTransposeNode(const NodePtr &node) {
  return (node->GetType() == kLoadType) || (node->GetType() == kStoreType) || (node->GetType() == kGatherType);
}

Status BackendUtils::GetTransposeInfos(
    AscGraph &asc_graph, bool &has_only_one_transpose,
    std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> &fallback_node_to_transpose_info) {
  TensorAttrInfo temp_graph_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, temp_graph_attr));
  int64_t swap_count_tal = 0;
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (!IsFallBackTransposeNode(node)) {
      continue;
    }
    std::vector<std::pair<int64_t, int64_t>> transpose_info;
    GE_ASSERT_SUCCESS(GetNodeTransposeInfo(node, temp_graph_attr, transpose_info));
    if (transpose_info.empty()) {
      continue;
    }
    fallback_node_to_transpose_info.emplace(node, transpose_info);
    GELOGI("node %s(%s) need to add transpose node with axis id %s in graph %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorPairToStr(transpose_info).c_str(),
           asc_graph.GetName().c_str());
    swap_count_tal += 1;
  }
  has_only_one_transpose = (swap_count_tal == 1);
  return SUCCESS;
}
}  // namespace ge
