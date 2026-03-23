/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_TRANSPOSE_BACKWARD_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_TRANSPOSE_BACKWARD_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "post_process/post_process_util.h"
#include "adaption_fallback_load.h"

namespace ge {
namespace asc_adapt {
inline Status InsertTransposeToNode(AscGraph &asc_graph, const NodePtr &node,
                                       std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  ViewOpAttrInfo attr_info;
  attr_info.transpose_info = transpose_info;
  GE_ASSERT_SUCCESS(InsertViewOpNodes(asc_graph, node, attr_info));
  return SUCCESS;
}

inline Status InsertTranspose(
    AscGraph &asc_graph,
    std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> &fallback_node_to_transpose_info) {
  for (auto it = fallback_node_to_transpose_info.begin(); it != fallback_node_to_transpose_info.end(); ++it) {
    ViewOpAttrInfo attr_info;
    attr_info.transpose_info = it->second;
    GE_ASSERT_SUCCESS(InsertViewOpNodes(asc_graph, it->first, attr_info));
  }
  return SUCCESS;
}

inline Status ApplySwapTensorInfo(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                                  std::vector<ge::Expression> &strides, bool need_update_repeats,
                                  const std::pair<int64_t, int64_t> &swap) {
  int64_t axis1 = swap.first;
  int64_t axis2 = swap.second;
  auto it1 = std::find(axis.begin(), axis.end(), axis1);
  auto it2 = std::find(axis.begin(), axis.end(), axis2);
  if ((it1 != axis.end()) && (it2 != axis.end())) {
    std::swap(*it1, *it2);
    if (!need_update_repeats) {
      return SUCCESS;
    }

    size_t index1 = std::distance(axis.begin(), it1);
    size_t index2 = std::distance(axis.begin(), it2);
    GE_ASSERT_TRUE(axis.size() == repeats.size());
    GE_ASSERT_TRUE(axis.size() == strides.size());
    std::swap(repeats[index1], repeats[index2]);
    strides.clear();
    ge::Expression temp_stride = kSymbolOne;
    for (size_t i = axis.size(); i > 0U; --i) {
      if (BackendUtils::IsEqOne(repeats[i - 1U])) {
        strides.insert(strides.begin(), kSymbolZero);
      } else {
        strides.insert(strides.begin(), temp_stride);
        temp_stride = repeats[i - 1U] * temp_stride;
      }
    }
  } else {
    return FAILED;
  }
  return SUCCESS;
}

inline Status ApplySwapTensorInfo(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                                  std::vector<ge::Expression> &strides, bool need_update_repeats,
                                  const std::vector<std::pair<int64_t, int64_t>> &swaps) {
  for (auto it = swaps.rbegin(); it != swaps.rend(); ++it) {
    GE_ASSERT_SUCCESS(ApplySwapTensorInfo(axis, repeats, strides, need_update_repeats, *it));
  }
  return SUCCESS;
}

inline Status ApplySwapTensorInfoInOrder(std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats,
                                         std::vector<ge::Expression> &strides, bool need_update_repeats,
                                         const std::vector<std::pair<int64_t, int64_t>> &swaps) {
  for (auto it = swaps.begin(); it != swaps.end(); ++it) {
    GE_ASSERT_SUCCESS(ApplySwapTensorInfo(axis, repeats, strides, need_update_repeats, *it));
  }
  return SUCCESS;
}

inline Status UpdateOneNodeTensorInfo(const NodePtr &node, std::vector<std::pair<int64_t, int64_t>> &transpose_info,
                                      std::unordered_set<NodePtr> &updated_tensor_info_nodes) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);

    GELOGI("node %s(%s) before apply swap tensor info axis %s repeats %s strides %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
    GE_ASSERT_SUCCESS(
        ApplySwapTensorInfo(tensor_attr->axis, tensor_attr->repeats, tensor_attr->strides, true, transpose_info));
    GELOGI("node %s(%s) after apply swap tensor info axis %s repeats %s strides %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
  }
  updated_tensor_info_nodes.insert(node);
  return SUCCESS;
}

inline Status UpdateTensorInfoAfterInsertTranspose(const NodePtr &node,
                                                   std::vector<std::pair<int64_t, int64_t>> &transpose_info,
                                                   std::unordered_set<NodePtr> &updated_tensor_info_nodes,
                                                   bool is_only_update_cur_node) {
  // 已遍历过返回
  if (updated_tensor_info_nodes.find(node) != updated_tensor_info_nodes.end()) {
    return SUCCESS;
  }
  // 递归到头返回
  if ((node->GetType() == kLoadType) || (node->GetType() == kDataType) || (node->GetType() == kGatherType)) {
    return SUCCESS;
  }
  // 前面是load的cast的轴信息跟load是一样的，所以也作为端点,返回
  if (node->GetType() == kCastType) {
    NodePtr peer_out_node;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_out_node, 0));
    if ((peer_out_node->GetType() == kLoadType) || (peer_out_node->GetType() == kGatherType)) {
      return SUCCESS;
    }
  }
  if (node->GetType() == kScalarType) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const auto output_tensor_desc = op_desc->MutableOutputDesc(0U);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);
    GELOGI("node %s(%s) before apply swap tensor info axis %s repeats %s strides %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
    GE_ASSERT_SUCCESS(
        ApplySwapTensorInfo(tensor_attr->axis, tensor_attr->repeats, tensor_attr->strides, false, transpose_info));
    updated_tensor_info_nodes.insert(node);
    GELOGI("node %s(%s) after apply swap tensor info axis %s repeats %s strides %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
    return SUCCESS;
  }
  // 非store节点需要交换轴
  if (node->GetType() != kStoreType) {
    GE_ASSERT_SUCCESS(UpdateOneNodeTensorInfo(node, transpose_info, updated_tensor_info_nodes));
  }
  // 继续往前递归
  if (!is_only_update_cur_node) {
    for (size_t i = 0U; i < node->GetInDataNodesSize(); i++) {
      const auto in_anchor = node->GetInDataAnchor(i);
      GE_ASSERT_NOTNULL(in_anchor);
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      auto peer_out_node = peer_out_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(peer_out_node);
      GE_ASSERT_SUCCESS(UpdateTensorInfoAfterInsertTranspose(peer_out_node, transpose_info, updated_tensor_info_nodes,
                                                             is_only_update_cur_node));
    }
  }
  return SUCCESS;
}

inline Status UpdatePreOutNodeTensorInfo(const NodePtr &node, std::vector<std::pair<int64_t, int64_t>> &transpose_info,
                                         std::unordered_set<NodePtr> &updated_tensor_info_nodes,
                                         bool is_only_update_cur_node) {
  std::vector<NodePtr> peer_out_nodes;
  // 获取所有前驱节点
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(node, peer_out_nodes));
  for (const auto &peer_out_node : peer_out_nodes) {
    GE_ASSERT_SUCCESS(UpdateTensorInfoAfterInsertTranspose(peer_out_node, transpose_info, updated_tensor_info_nodes,
                                                           is_only_update_cur_node));
  }
  return SUCCESS;
}

// 检查节点是否终端节点
inline bool IsTerminalNode(const NodePtr &node) {
  return (node->GetType() == kDataType) || (node->GetType() == kLoadType) || (node->GetType() == kScalarType) ||
         (node->GetType() == kGatherType);
}

// 检查输入节点是否都有transpose
inline Status CheckAllInputsHaveTranspose(
    AscGraph &asc_graph, const NodePtr &node,
    const std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> &fallback_node_to_transpose_info,
    bool &is_all_transpose_load, std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  std::vector<NodePtr> load_nodes;

  for (size_t i = 0U; i < node->GetInDataNodesSize(); i++) {
    const auto in_anchor = node->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_anchor);
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    const auto peer_out_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_out_node);

    GE_ASSERT_SUCCESS(BackendUtils::FindPrefaceLoadNodes(peer_out_anchor, load_nodes));
    if (load_nodes.empty()) {
      is_all_transpose_load = false;
      GELOGI("peer out node %s(%s) has no peer out loads: graph %s node %s(%s).",
             peer_out_node->GetName().c_str(), peer_out_node->GetType().c_str(), asc_graph.GetName().c_str(),
             node->GetName().c_str(), node->GetType().c_str());
    }
    for (auto &load_node : load_nodes) {
      auto it = fallback_node_to_transpose_info.find(load_node);
      if (it == fallback_node_to_transpose_info.end()) {
        is_all_transpose_load = false;
        GELOGI("peer out node %s(%s) has no transpose peer out loads: graph %s node %s(%s).",
               peer_out_node->GetName().c_str(), peer_out_node->GetType().c_str(), asc_graph.GetName().c_str(),
               node->GetName().c_str(), node->GetType().c_str());
        break;
      } else {
        transpose_info = it->second;
      }
    }
  }
  return SUCCESS;
}

// 1、向上找到第一个多输入节点，2、无多输入节点则找到最后一个输出多引用节点
inline Status FindSomePreNode(const NodePtr &cur_node, NodePtr &first_pre_mul_inputs_node,
                              NodePtr &last_pre_outputs_mul_reference_node, std::unordered_set<NodePtr> &found_nodes,
                              bool &if_found) {
  NodePtr pre_node = cur_node;
  while (!IsTerminalNode(pre_node)) {
    std::vector<NodePtr> peer_out_nodes;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(pre_node, peer_out_nodes));
    std::vector<NodePtr> peer_in_nodes;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(pre_node, peer_in_nodes, 0));
    if (peer_out_nodes.size() > 1U) {
      first_pre_mul_inputs_node = pre_node;
      return SUCCESS;
    }
    if (peer_in_nodes.size() > 1U) {
      last_pre_outputs_mul_reference_node = pre_node;
    }
    if (peer_out_nodes.size() < 1U) {
      return SUCCESS;
    } else {
      pre_node = peer_out_nodes[0];
      if (found_nodes.find(pre_node) != found_nodes.end()) {
        if_found = true;
        return SUCCESS;
      }
      found_nodes.insert(pre_node);
    }
  }
  return SUCCESS;
}

// 1、从当前节点向上找到第一个transpose支持的dtype的节点后面插入transpose
inline Status FindDtypeSupportedNodeForward(const NodePtr &cur_node, NodePtr &dtype_supported_node,
                                            std::vector<NodePtr> &nodes_need_update_tensor) {
  NodePtr pre_node = cur_node;
  while (!IsTerminalNode(pre_node)) {
    AscTensorAttr *output_attr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(pre_node, output_attr));
    if (asc_adapt::CheckTransposeDtype(output_attr->dtype)) {
      dtype_supported_node = pre_node;
      return SUCCESS;
    }
    if (!IsSingleInAndOutNode(pre_node)) {
      GELOGI("node %s %s has multiple peer out/in nodes, can't find dtype supported place to insert transpose.",
             pre_node->GetName().c_str(), pre_node->GetType().c_str());
      return SUCCESS;
    } else {
      std::vector<NodePtr> peer_out_nodes;
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(pre_node, peer_out_nodes));
      nodes_need_update_tensor.push_back(pre_node);
      pre_node = peer_out_nodes[0];
    }
  }
  return SUCCESS;
}

inline Status InsertTransposePro(AscGraph &asc_graph, const NodePtr &node_need_transpose,
                                 std::vector<std::pair<int64_t, int64_t>> &transpose_info,
                                 std::unordered_set<NodePtr> &updated_tensor_info_nodes,
                                 std::unordered_set<NodePtr> &found_nodes) {
  GELOGI("graph %s has transpose(%s) backward to node %s %s.", asc_graph.GetName().c_str(),
         AutofuseUtils::VectorPairToStr(transpose_info).c_str(), node_need_transpose->GetName().c_str(),
         node_need_transpose->GetType().c_str());
  // 1.1、刷新了当前node(此处仅刷新当前node)后再插入transpose，否则transpose轴序不对
  GE_ASSERT_SUCCESS(
      UpdateTensorInfoAfterInsertTranspose(node_need_transpose, transpose_info, updated_tensor_info_nodes, true));
  // 1.2、插入transpose
  GE_ASSERT_SUCCESS(InsertTransposeToNode(asc_graph, node_need_transpose, transpose_info));
  // 1.3、标记transpose为查找过的节点，避免输出多引用场景重复插
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(node_need_transpose, peer_in_nodes, 0));
  GE_ASSERT_TRUE(!peer_in_nodes.empty());
  found_nodes.insert(peer_in_nodes[0]);
  // 1.4、把当前节点前面的节点都（data、load、cast、scalar、store除外）递归重新刷轴
  GE_ASSERT_SUCCESS(UpdatePreOutNodeTensorInfo(node_need_transpose, transpose_info, updated_tensor_info_nodes, false));
  return SUCCESS;
}

inline Status FindAndInsertTranspose(
    AscGraph &asc_graph, const NodePtr &node,
    std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> &fallback_node_to_transpose_info,
    std::unordered_set<NodePtr> &updated_tensor_info_nodes, std::unordered_set<NodePtr> &found_nodes) {
  // 第一步：检查节点是否已处理或是终端节点
  if ((found_nodes.find(node) != found_nodes.end()) || IsTerminalNode(node)) {
    return SUCCESS;
  }
  found_nodes.insert(node);

  // 第二步：检查当前节点的所有输入（load节点）是否都有transpose
  std::vector<std::pair<int64_t, int64_t>> transpose_info;
  bool is_all_transpose_load = true;
  GE_ASSERT_SUCCESS(CheckAllInputsHaveTranspose(asc_graph, node, fallback_node_to_transpose_info, is_all_transpose_load,
                                                transpose_info));

  // 第三步：根据检查结果决定处理路径
  if (is_all_transpose_load) {  // 当前节点前序load全为transpose，做插入transpose处理
    GE_ASSERT_TRUE(!transpose_info.empty());
    auto it = fallback_node_to_transpose_info.find(node);
    // 1、此输入有store不带transpose，在store前面插transpose；或者此输入非store节点，在此节点后插入transpose
    if (it == fallback_node_to_transpose_info.end()) {
      NodePtr first_pre_mul_inputs_node = nullptr;
      NodePtr last_pre_outputs_mul_reference_node = nullptr;
      bool if_found = false;
      GE_ASSERT_SUCCESS(
          FindSomePreNode(node, first_pre_mul_inputs_node, last_pre_outputs_mul_reference_node, found_nodes, if_found));
      if (first_pre_mul_inputs_node != nullptr) {
        GE_ASSERT_SUCCESS(InsertTransposePro(asc_graph, first_pre_mul_inputs_node, transpose_info,
                                             updated_tensor_info_nodes, found_nodes));
      } else if (last_pre_outputs_mul_reference_node != nullptr) {
        GE_ASSERT_SUCCESS(InsertTransposePro(asc_graph, last_pre_outputs_mul_reference_node, transpose_info,
                                             updated_tensor_info_nodes, found_nodes));
      } else if (if_found) {  // transpose已插入过
        return SUCCESS;
      } else {
        GE_ASSERT_SUCCESS(InsertTransposePro(asc_graph, node, transpose_info, updated_tensor_info_nodes, found_nodes));
      }
    } else if (node->GetType() == kStoreType) {
      // 2、此输入有store带transpose，则对消不插入transpose
      GELOGI("graph %s has transpose(%s) before node %s(%s), don't backward.", asc_graph.GetName().c_str(),
             AutofuseUtils::VectorPairToStr(transpose_info).c_str(), node->GetName().c_str(), node->GetType().c_str());
      // 2.1、把当前节点前面的节点都（data、load、cast、scalar、store除外）递归，重新刷轴
      GE_ASSERT_SUCCESS(UpdatePreOutNodeTensorInfo(node, transpose_info, updated_tensor_info_nodes, false));
    }
  } else {  // 3、当前节点前序load非全为transpose，遍历当前节点的前一级输入节点做递归判断
    for (size_t i = 0U; i < node->GetInDataNodesSize(); i++) {
      const auto in_anchor = node->GetInDataAnchor(i);
      GE_ASSERT_NOTNULL(in_anchor);
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      const auto peer_out_node = peer_out_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(peer_out_node);
      GE_ASSERT_SUCCESS(FindAndInsertTranspose(asc_graph, peer_out_node, fallback_node_to_transpose_info,
                                               updated_tensor_info_nodes, found_nodes));
    }
  }
  return SUCCESS;
}

inline Status TransposeBackwardPro(
    AscGraph &asc_graph,
    std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> &fallback_node_to_transpose_info) {
  std::unordered_set<NodePtr> updated_tensor_info_nodes;
  std::unordered_set<NodePtr> found_nodes;
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (node->GetType() == kStoreType) {
      GELOGI("graph %s node %s(%s) start to do transpose backward process.", asc_graph.GetName().c_str(),
             node->GetName().c_str(), node->GetType().c_str());
      GE_ASSERT_SUCCESS(FindAndInsertTranspose(asc_graph, node, fallback_node_to_transpose_info,
                                               updated_tensor_info_nodes, found_nodes));
    }
  }
  return SUCCESS;
}

inline Status TransposePlaceUpdateForDtype(AscGraph &asc_graph) {
  std::vector<std::pair<int64_t, int64_t>> transpose_info;
  std::vector<NodePtr> nodes_need_update_tensor;
  NodePtr dtype_supported_node = nullptr;
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (node->GetType() != kTransposeType) {
      continue;
    }
    // 当前transpose的dtype是支持的直接返回
    AscTensorAttr *output_attr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node, output_attr));
    if (asc_adapt::CheckTransposeDtype(output_attr->dtype)) {
      GELOGI("node %s %s is with supported dtype in graph.", node->GetName().c_str(), node->GetType().c_str(),
             asc_graph.GetName().c_str());
      return SUCCESS;
    }
    // 往前找transpose的dtype是支持的节点后面插
    GE_ASSERT_SUCCESS(FindDtypeSupportedNodeForward(node, dtype_supported_node, nodes_need_update_tensor));
    if (dtype_supported_node != nullptr) {
      AscTensorAttr *dtype_supported_output_attr;
      int64_t swap_count = 0;
      GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(dtype_supported_node, dtype_supported_output_attr));
      AscTensorAttr *output_attr;
      GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node, output_attr));
      GE_ASSERT_SUCCESS(
          BackendUtils::MinSwapCount(dtype_supported_output_attr->axis, output_attr->axis, swap_count, transpose_info));
      GE_ASSERT_SUCCESS(asc_adapt::DelNode(asc_graph, node));
      break;
    }
    nodes_need_update_tensor.clear();
  }
  if (dtype_supported_node == nullptr) {
    GELOGI("graph %s can not find a place to insert transpose with supported dtype.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  // 找到后插入transpose，并更新原transpose和找到的位置之间的节点tensor信息
  GE_ASSERT_SUCCESS(InsertTransposeToNode(asc_graph, dtype_supported_node, transpose_info));
  for (auto &node_need_update_tensor : nodes_need_update_tensor) {
    AscTensorAttr *output_attr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node_need_update_tensor, output_attr));
    GE_ASSERT_SUCCESS(ApplySwapTensorInfoInOrder(output_attr->axis, output_attr->repeats, output_attr->strides, true,
                                                 transpose_info));
  }
  return SUCCESS;
}

inline Status TransposeBackward(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  if (BackendUtils::IsCubeAscNode(asc_node)) {
    GELOGI("graph %s fuse type is cube, don't transpose backward.", asc_graph.GetName().c_str());
    return SUCCESS;
  }

  bool has_only_one_transpose = false;
  std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> fallback_node_to_transpose_info;
  GE_ASSERT_SUCCESS(
      BackendUtils::GetTransposeInfos(asc_graph, has_only_one_transpose, fallback_node_to_transpose_info));
  if (fallback_node_to_transpose_info.empty()) {
    GELOGI("graph %s fuse type has no transpose, don't transpose backward.", asc_graph.GetName().c_str());
    return SUCCESS;
  } else if (has_only_one_transpose) {  // 只有一个transpose，不需要后移
    GELOGI("graph %s has only one transpose, don't transpose backward.", asc_graph.GetName().c_str());
    GE_ASSERT_SUCCESS(InsertTranspose(asc_graph, fallback_node_to_transpose_info));
    // transpose有些dtype不支持，如果当前的transpose不支持例如uint64_t，则往前或者往后找dtype不是uint64_t的位置
    GE_ASSERT_SUCCESS(TransposePlaceUpdateForDtype(asc_graph));
  } else {
    GE_ASSERT_SUCCESS(TransposeBackwardPro(asc_graph, fallback_node_to_transpose_info));
    // transpose有些dtype不支持，如果当前的transpose不支持例如uint64_t，则往前或者往后找dtype不是uint64_t的位置
    GE_ASSERT_SUCCESS(TransposePlaceUpdateForDtype(asc_graph));
  }

  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

inline Status TransposeBackwardForCodegen(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  // TransposeBackward临时流程，因为后端目前只支持一个ascgraph里面有1个transpose节点，后续删除,删除此流程时需要把反推transpose接口打开
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, TransposeBackward, "transpose_backward"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_TRANSPOSE_BACKWARD_H
