/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_COMPLETE_NODE_ATTRS_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_COMPLETE_NODE_ATTRS_H
#include "common/checker.h"
#include "post_process/post_process_util.h"

namespace ge {
namespace asc_adapt {
inline Status UpdateStridesByReapeats(const std::vector<ge::Expression> &repeats,
                                      std::vector<ge::Expression> &strides) {
  strides.clear();
  ge::Expression temp_stride = kSymbolOne;
  for (size_t i = repeats.size(); i > 0U; --i) {
    if (BackendUtils::IsEqOne(repeats[i - 1U])) {
      strides.insert(strides.begin(), kSymbolZero);
    } else {
      strides.insert(strides.begin(), temp_stride);
      temp_stride = repeats[i - 1U] * temp_stride;
    }
  }
  return SUCCESS;
}

inline Status UpdateTensorAttrsIfEmpty(const NodePtr &node, AscTensorAttr *tensor_attr,
                                       const std::vector<int64_t> &axis, const std::vector<Expression> &repeats) {
  GELOGD("before complete attrs: node %s(%s), axis size is 0.", node->GetName().c_str(), node->GetType().c_str());
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(node, peer_out_nodes));
  GE_ASSERT_TRUE(!peer_out_nodes.empty());
  NodePtr &peer_out_node = peer_out_nodes[0];
  for (const auto &node : peer_out_nodes) {
    if (node->GetType() != kScalarType) {
      peer_out_node = node;
      break; // Scalar没有tensor信息，获取第一个非scalar给后续节点补轴
    }
  }
  // 1、前面是load场景有transpose后不能根据load补轴，得根据graph来补轴；2、前面是scalar场景，需要根据graph来补轴;
  if ((peer_out_node->GetType() == kScalarType) || (peer_out_node->GetType() == kLoadType) ||
      (peer_out_node->GetType() == kGatherType)) {
    // 前驱节点都是scalar，从图上获取tensor信息
    GELOGD("peer_out_nodes are all scalar or has load, get tensor info from graph.");
    tensor_attr->axis = axis;
    tensor_attr->repeats = repeats;
    GE_ASSERT_SUCCESS(UpdateStridesByReapeats(repeats, tensor_attr->strides));
  } else {
    // 获取前驱节点的输出描述符
    GeTensorDescPtr peer_output_tensor_desc;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(peer_out_node, peer_output_tensor_desc));
    const auto peer_output_attr = peer_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(peer_output_attr);
    tensor_attr->axis = peer_output_attr->axis;
    tensor_attr->repeats = peer_output_attr->repeats;
    tensor_attr->strides = peer_output_attr->strides;
  }
  GELOGD("after complete attrs: node %s(%s) ,axis:%s, repeats:%s stride:%s.", node->GetName().c_str(),
         node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
  return SUCCESS;
}

inline Status UpdateTensorAttrsIfNotEmpty(const NodePtr &node, const std::vector<int64_t> &axis,
                                          const std::vector<Expression> &repeats, AscTensorAttr *tensor_attr) {
  (void)repeats;
  GELOGD("before update attrs: node %s(%s), axis:%s, repeats:%s stride:%s.", node->GetName().c_str(),
         node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
  // 校验node轴id的size比graph轴id的size小或相等，因为有transpose没法保证是graph轴id的有序子集
  GE_ASSERT_TRUE(tensor_attr->axis.size() <= axis.size());
  // 1. 从graph补齐轴id,repeat和stride设置为1和0，保存indexs
  // 2. 刷新所有stride为低维的stride*repeat
  // 3. 刷新indexs对应维度的stride为0
  std::vector<size_t> axis_idx_to_complete;
  for (size_t i = 0U; i < axis.size(); i++) {
    // 考虑到后面有transpose会更改axis位置，不能直接对比同一个index上的axis是否一致来判断是否要补轴，要直接对比axis
    auto it = std::find(tensor_attr->axis.begin(), tensor_attr->axis.end(), axis[i]);
    if (it == tensor_attr->axis.end()) {
      // 给 tensor 补齐一个轴
      tensor_attr->axis.insert(tensor_attr->axis.begin() + i, axis[i]);
      // 先更新 repeats
      tensor_attr->repeats.insert(tensor_attr->repeats.begin() + i, kSymbolOne);
      tensor_attr->strides.insert(tensor_attr->strides.begin() + i, kSymbolZero);
    }
    if (BackendUtils::IsEqOne(tensor_attr->repeats[i]) && BackendUtils::IsEqZero(tensor_attr->strides[i])) {
      axis_idx_to_complete.push_back(i);
    }
  }
  // slice 的 stride 不连续,不能刷strides
  if ((node->GetType() == kLoadType) || (node->GetType() == kGatherType)) {
    GELOGD("after update attrs: node %s(%s), axis:%s, repeats:%s, stride:%s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
    return SUCCESS;
  }

  const auto axis_size = tensor_attr->strides.size();
  GE_ASSERT_TRUE(axis_size > 0U);
  tensor_attr->strides[axis_size - 1U] = kSymbolOne;
  for (size_t i = axis_size - 1U; i > 0U; i--) {
    tensor_attr->strides[i - 1U] = tensor_attr->strides[i] * tensor_attr->repeats[i];
  }
  for (const auto i : axis_idx_to_complete) {
    tensor_attr->strides[i] = kSymbolZero;
  }
  GELOGD("after update attrs: node %s(%s), axis:%s, repeats:%s, stride:%s.", node->GetName().c_str(),
         node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
  return SUCCESS;
}

inline Status UpdateTensorAttrs(const NodePtr &node, const std::vector<int64_t> &axis,
                                const std::vector<Expression> &repeats) {
  // 获取node属性
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);

  // 更新output tensor desc属性
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);
    // Scalar的tensor轴信息为空，填充repeat和stride为1和0
    if (node->GetType() == kScalarType) {
      tensor_attr->axis = axis;
      tensor_attr->repeats.clear();
      tensor_attr->strides.clear();
      for (size_t j = 0U; j < axis.size(); ++j) {
        tensor_attr->repeats.push_back(kSymbolOne);
        tensor_attr->strides.push_back(kSymbolZero);
      }
      continue;
    }
    if ((tensor_attr->axis.empty()) && (node->GetType() != kDataType)) {
      // 当前节点的轴信息为空，则需要根据前一个节点的轴信息进行补齐，data、load、broadcast、reduce等节点轴信息不为空
      GE_ASSERT_SUCCESS(UpdateTensorAttrsIfEmpty(node, tensor_attr, axis, repeats));
      continue;
    }
    // 当前节点的轴信息不为空，则根据graph的轴信息进行补齐，data、load、broadcast、reduce等节点轴信息不为空
    GE_ASSERT_SUCCESS(UpdateTensorAttrsIfNotEmpty(node, axis, repeats, tensor_attr));
  }
  return SUCCESS;
}

inline Status CompleteNodeAttrsOnAscGraph(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  TensorAttrInfo graph_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, graph_attr));
  GELOGI("max sched axis %s in graph %s.", AutofuseUtils::VectorToStr(graph_attr.axis).c_str(),
         asc_graph.GetName().c_str());

  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetDirectNode()) {
    // torch data 没有任何轴或者node属性信息，直接跳过
    if (IsTorchDataType(node)) {
      GELOGI("torch node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (IsCubeRelatedAscNode(node)) {
      GELOGI("cube related node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (IsGatherData(node)) {
    // data 节点后接 gather，不走标准补轴，直接跳过,在无效轴删除前刷新一把
      GELOGI("gather node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
    } else if (BackendUtils::IsOutputNode(node)) {
    // output是由sched axis的，但是没有tensor轴，所以得把sched axis刷成output前一个节点的sched axis，不刷新tensor轴
      NodePtr peer_out_node;
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_out_node, 0));
      const auto &peer_out_op_desc = peer_out_node->GetOpDesc();
      GE_ASSERT_NOTNULL(peer_out_op_desc);
      auto peer_out_node_attr = peer_out_op_desc->GetAttrsGroup<AscNodeAttr>();
      GE_ASSERT_NOTNULL(peer_out_node_attr);
    } else {
      // 以graph轴刷新tensor轴，并获取tensor轴给node属性中的sched轴赋值
      GE_ASSERT_SUCCESS(UpdateTensorAttrs(node, graph_attr.axis, graph_attr.repeats));
    }
    // 以graph轴刷新sched轴，同一个ascgraph里面的节点调度轴都跟graph轴是一样的
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    GELOGI("node %s(%s) before complete sched axis %s to %s in graph %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str(),
           AutofuseUtils::VectorToStr(graph_attr.axis).c_str(), asc_graph.GetName().c_str());
    node_attr->sched.axis = graph_attr.axis;
    GELOGI("node %s(%s) after complete sched axis %s to %s in graph %s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str(),
           AutofuseUtils::VectorToStr(graph_attr.axis).c_str(), asc_graph.GetName().c_str());
  }
  return SUCCESS;
}

inline Status UpdateInvalidIndices(const NodePtr &node, std::vector<int64_t> &graph_invalid_axis_id) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    if (graph_invalid_axis_id.empty()) {
      return SUCCESS;
    }
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);
    for (auto it = graph_invalid_axis_id.begin(); it != graph_invalid_axis_id.end();) {
      auto axis_id = *it;
      auto axis_it = std::find(tensor_attr->axis.begin(), tensor_attr->axis.end(), axis_id);
      if (axis_it == tensor_attr->axis.end()) {
        ++it;
        continue;
      }
      auto idx = static_cast<int64_t>(std::distance(tensor_attr->axis.begin(), axis_it));
      if (BackendUtils::IsEqOne(tensor_attr->repeats[idx]) && BackendUtils::IsEqZero(tensor_attr->strides[idx])) {
        ++it;
      } else {
        GELOGI("node %s(%s) output(%u), axis idx %" PRId64 " is valid, del idx from graph invalid axis indices.",
               node->GetName().c_str(), node->GetType().c_str(), i, idx);
        // 有node的输出tensor的repeat和stride不为1和0，说明该轴是有效的，从graph_invalid_axis_id中删除
        it = graph_invalid_axis_id.erase(it);
      }
    }
  }
  return SUCCESS;
}

inline Status GetInvalidAxis(AscGraph &asc_graph, std::vector<int64_t> &graph_invalid_axis_id) {
  const auto graph_attr = AscGraphUtils::GetComputeGraph(asc_graph)->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  auto size_t = graph_attr->axis.size();
  if (size_t <= 1U) {
    GELOGI("graph axis size <= 1, no need to del invalid axis, in graph %s.", asc_graph.GetName().c_str());
    return SUCCESS;
  }
  for (auto idx = 0U; idx < size_t; idx++) {
    const auto &axis_info = graph_attr->axis[idx];
    if (BackendUtils::IsEqOne(axis_info->size)) {
      // 有transpose节点，transpose节点可能改变该轴的index，所以不能直接用index，需要用axis id
      graph_invalid_axis_id.push_back(axis_info->id);
    }
  }
  GELOGI("before update with node tensor info, graph invalid axis id %s in graph %s.",
         (AutofuseUtils::VectorToStr(graph_invalid_axis_id)).c_str(), asc_graph.GetName().c_str());

  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetDirectNode()) {
    if (graph_invalid_axis_id.empty()) {
      return SUCCESS;
    }
    // torch data 没有任何轴或者node属性信息，直接跳过
    if (BackendUtils::IsOutputNode(node) || (node->GetType() == kScalarType) || (node->GetType() == kDataType)) {
      continue;
    }
    GE_ASSERT_SUCCESS(UpdateInvalidIndices(node, graph_invalid_axis_id));
  }
  GELOGI("after update with node tensor info, graph invalid axis id %s in graph %s.",
         (AutofuseUtils::VectorToStr(graph_invalid_axis_id)).c_str(), asc_graph.GetName().c_str());
  return SUCCESS;
}

inline Status FlashContinueNodeAxis(std::vector<int64_t> &axis, const std::vector<int64_t> &graph_invalid_axis_id) {
  for (const auto &deleted_axis_id : graph_invalid_axis_id) {
    for (auto &axis_id : axis) {
      if (axis_id > deleted_axis_id) {
        GELOGD("node axis id flash from %ld to %ld.", axis_id, axis_id - 1);
        (axis_id)--;
      }
    }
  }
  return SUCCESS;
}

inline Status FlashContinueGraphAxis(std::vector<AxisPtr> &axis, const std::vector<int64_t> &graph_invalid_axis_id) {
  for (const auto &deleted_axis_id : graph_invalid_axis_id) {
    for (auto axis_info : axis) {
      if (axis_info->id > deleted_axis_id) {
        GELOGD("graph axis id flash from %ld to %ld.", axis_info->id, axis_info->id - 1);
        (axis_info->id)--;
      }
    }
  }
  return SUCCESS;
}

inline Status RemoveGraphInvalidAxis(AscGraph &asc_graph, const std::vector<int64_t> &graph_invalid_axis_id) {
  const auto graph_attr = AscGraphUtils::GetComputeGraph(asc_graph)->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  for (auto it = graph_attr->axis.begin(); it != graph_attr->axis.end();) {
    const auto axis = *it;
    if (std::find(graph_invalid_axis_id.begin(), graph_invalid_axis_id.end(), axis->id) !=
        graph_invalid_axis_id.end()) {
      GELOGD("graph %s axis id %ld is del.", asc_graph.GetName().c_str(), axis->id);
      it = graph_attr->axis.erase(it);  // erase 返回下一个有效迭代器
    } else {
      ++it;  // 如果未删除，继续下一个元素
    }
  }
  // 后端有连续轴约束，需要把删除后的剩余的轴从0开始重新变连续（但是轴的相对位置不变）
  GE_ASSERT_SUCCESS(FlashContinueGraphAxis(graph_attr->axis, graph_invalid_axis_id));
  return SUCCESS;
}

inline Status RemoveNodeInvalidAxis(const NodePtr &node, const std::vector<int64_t> &graph_invalid_axis_id) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);

  for (auto it = graph_invalid_axis_id.begin(); it != graph_invalid_axis_id.end(); ++it) {
    auto axis_id = *it;
    // 删除 node_attr->sched.axis 中的 axis_id
    auto sched_axis_it = std::find(node_attr->sched.axis.begin(), node_attr->sched.axis.end(), axis_id);
    if (sched_axis_it == node_attr->sched.axis.end()) {
      continue; // 找不到可认为是已经删除
    }
    GELOGD("node %s(%s) sched_axis id %ld is del.", node->GetName().c_str(), node->GetType().c_str(), axis_id);
    node_attr->sched.axis.erase(sched_axis_it);  // 直接通过迭代器删除
    if ((BackendUtils::IsOutputNode(node)) || IsGatherData(node)) {
      continue;
    }

    for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
      GE_ASSERT_NOTNULL(output_tensor_desc);
      auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(tensor_attr);
      auto axis_it = std::find(tensor_attr->axis.begin(), tensor_attr->axis.end(), axis_id);
      if (axis_it == tensor_attr->axis.end()) {
        continue; // 找不到可认为是已经删除
      }
      // 每一次删除使用的idx都是当前循环里查找出来的，所以不会删除错误
      auto axis_idx = static_cast<int64_t>(std::distance(tensor_attr->axis.begin(), axis_it));
      GELOGD("node %s(%s) output(%u), axis/repeats/strides on axis id %ld is del.", node->GetName().c_str(),
             node->GetType().c_str(), i, axis_id);
      tensor_attr->axis.erase(tensor_attr->axis.begin() + axis_idx);
      // kScalarType补轴只补了axis，没补repeats和strides，因此scalar没有repeats和strides，不需要删除，后续考虑给scalar的repeat和stride补1和0
      if ((!tensor_attr->repeats.empty()) && (!tensor_attr->strides.empty())) {
        tensor_attr->repeats.erase(tensor_attr->repeats.begin() + axis_idx);
        tensor_attr->strides.erase(tensor_attr->strides.begin() + axis_idx);
      }
    }
  }
  // 后端有连续轴约束，需要把删除后的剩余的轴从0开始重新变连续（但是轴的相对位置不变）
  GE_ASSERT_SUCCESS(FlashContinueNodeAxis(node_attr->sched.axis, graph_invalid_axis_id));
  if (BackendUtils::IsOutputNode(node)) {
    return SUCCESS;
  }
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);
    GE_ASSERT_SUCCESS(FlashContinueNodeAxis(tensor_attr->axis, graph_invalid_axis_id));
  }
  return SUCCESS;
}

inline Status RemoveInvalidAxis(AscGraph &asc_graph, const std::vector<int64_t> &graph_invalid_axis_id) {
  if (graph_invalid_axis_id.empty()) {
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(RemoveGraphInvalidAxis(asc_graph, graph_invalid_axis_id));
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetDirectNode()) {
    // torch data 没有任何轴或者node属性信息，直接跳过； gather data在前面特殊处理了
    if (IsTorchDataType(node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(RemoveNodeInvalidAxis(node, graph_invalid_axis_id));
  }
  return SUCCESS;
}

inline Status UpdateGatherDataAxis(const AscGraph &asc_graph, const NodePtr &node,
                                   std::vector<NodePtr> &gather_data2_nodes) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(node, peer_out_nodes));
  GE_ASSERT_TRUE(peer_out_nodes.size() == 2U);  // gather只有两个输入，data1和data2轴不一样
  AscTensorAttr *gather_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node, gather_output_attr));
  AscTensorAttr *data1_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(peer_out_nodes[0], data1_output_attr));
  AscTensorAttr *data2_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(peer_out_nodes[1], data2_output_attr));
  gather_data2_nodes.push_back(peer_out_nodes[1]);
  GELOGI("node %s(%s) before update gather data1 axis %s in graph %s.", peer_out_nodes[0]->GetName().c_str(),
         peer_out_nodes[0]->GetType().c_str(), AutofuseUtils::VectorToStr(data1_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());
  GELOGI("node %s(%s) before update gather data2 axis %s in graph %s.", peer_out_nodes[1]->GetName().c_str(),
         peer_out_nodes[1]->GetType().c_str(), AutofuseUtils::VectorToStr(data2_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());

  int64_t gather_axis_index = std::numeric_limits<int64_t>::max();
  GE_ASSERT_SUCCESS(asc_adapt::GetGatherAxis(node, gather_axis_index));
  auto axis_index = static_cast<size_t>(gather_axis_index);
  const auto &gather_axis = gather_output_attr->axis;
  auto &data1_axis = data1_output_attr->axis;
  auto &data2_axis = data2_output_attr->axis;
  auto data2_axis_size = data2_axis.size();
  auto gather_axis_size = gather_axis.size();
  GE_ASSERT_TRUE(axis_index + data2_axis_size <= gather_axis_size);
  data1_axis.clear();
  data2_axis.clear();
  for (size_t i = 0U; i < gather_axis_size; ++i) {
    if (axis_index == i) {
      data1_axis.push_back(gather_axis[i]);
      data2_axis.push_back(gather_axis[i]);
    } else if ((i > axis_index) && (i < axis_index + data2_axis_size)) {
      data2_axis.push_back(gather_axis[i]);
    } else {
      data1_axis.push_back(gather_axis[i]);
    }
  }
  GELOGI("node %s(%s) after update gather data1 axis %s in graph %s.", peer_out_nodes[0]->GetName().c_str(),
         peer_out_nodes[0]->GetType().c_str(), AutofuseUtils::VectorToStr(data1_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());
  GELOGI("node %s(%s) after update gather data2 axis %s in graph %s.", peer_out_nodes[1]->GetName().c_str(),
         peer_out_nodes[1]->GetType().c_str(), AutofuseUtils::VectorToStr(data2_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());
  return SUCCESS;
}

inline Status FindAndUpdateGatherData(const AscGraph &asc_graph, std::vector<NodePtr> &gather_data2_nodes) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetDirectNode()) {
    if (node->GetType() == kGatherType) {
      // gather的data1和data2需要特殊补轴处理
      GE_ASSERT_SUCCESS(UpdateGatherDataAxis(asc_graph, node, gather_data2_nodes));
    }
  }
  return SUCCESS;
}

inline Status RemoveGatherInvalidAxisIndex(const AscGraph &asc_graph, const NodePtr &node,
                                           const std::vector<int64_t> &graph_invalid_axis_id) {
  AscTensorAttr *gather_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node, gather_output_attr));
  const auto &gather_axis = gather_output_attr->axis;
  auto gather_axis_size = gather_axis.size();
  int64_t gather_axis_index = std::numeric_limits<int64_t>::max();
  GE_ASSERT_SUCCESS(asc_adapt::GetGatherAxis(node, gather_axis_index));
  GE_ASSERT_TRUE(static_cast<size_t>(gather_axis_index) < gather_axis_size);

  int64_t threshold = gather_axis[gather_axis_index];
  int64_t cnt = std::count_if(graph_invalid_axis_id.begin(), graph_invalid_axis_id.end(),
                              [threshold](int64_t val) { return val < threshold; });
  GE_ASSERT_TRUE(gather_axis_index >= cnt);
  GE_ASSERT_SUCCESS(asc_adapt::SetGatherAxis(node, gather_axis_index - cnt));
  GELOGI("update gather(%s %s) axis index from %" PRId64 " to %" PRId64 " in graph %s.", node->GetName().c_str(),
         node->GetType().c_str(), gather_axis_index, gather_axis_index - cnt, asc_graph.GetName().c_str());
  return SUCCESS;
}

inline Status RemoveGatherDataInvalidAxis(const AscGraph &asc_graph, const NodePtr &node,
                                          const std::vector<int64_t> &graph_invalid_axis_id) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(node, peer_out_nodes));
  GE_ASSERT_TRUE(peer_out_nodes.size() == 2U);  // gather只有两个输入，data1和data2轴不一样
  AscTensorAttr *gather_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node, gather_output_attr));
  AscTensorAttr *data1_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(peer_out_nodes[0], data1_output_attr));
  AscTensorAttr *data2_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(peer_out_nodes[1], data2_output_attr));
  GELOGI("node %s(%s) before remove gather data1 axis %s in graph %s.", peer_out_nodes[0]->GetName().c_str(),
         peer_out_nodes[0]->GetType().c_str(), AutofuseUtils::VectorToStr(data1_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());
  GELOGI("node %s(%s) before remove gather data2 axis %s in graph %s.", peer_out_nodes[1]->GetName().c_str(),
         peer_out_nodes[1]->GetType().c_str(), AutofuseUtils::VectorToStr(data2_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());

  int64_t gather_axis_index = std::numeric_limits<int64_t>::max();
  GE_ASSERT_SUCCESS(asc_adapt::GetGatherAxis(node, gather_axis_index));
  auto gather_replace_axis = gather_output_attr->axis[gather_axis_index];

  for (auto it = graph_invalid_axis_id.begin(); it != graph_invalid_axis_id.end(); ++it) {
    auto axis_id = *it;
    auto axis_it1 = std::find(data1_output_attr->axis.begin(), data1_output_attr->axis.end(), axis_id);
    if (axis_it1 != data1_output_attr->axis.end()) {
      if (axis_id != gather_replace_axis) {
        auto axis_idx = static_cast<int64_t>(std::distance(data1_output_attr->axis.begin(), axis_it1));
        data1_output_attr->axis.erase(data1_output_attr->axis.begin() + axis_idx);
        data1_output_attr->repeats.erase(data1_output_attr->repeats.begin() + axis_idx);
        data1_output_attr->strides.erase(data1_output_attr->strides.begin() + axis_idx);
      }
    }
    auto axis_it2 = std::find(data2_output_attr->axis.begin(), data2_output_attr->axis.end(), axis_id);
    if (axis_it2 != data2_output_attr->axis.end()) {
      auto axis_idx = static_cast<int64_t>(std::distance(data2_output_attr->axis.begin(), axis_it2));
      data2_output_attr->axis.erase(data2_output_attr->axis.begin() + axis_idx);
      data2_output_attr->repeats.erase(data2_output_attr->repeats.begin() + axis_idx);
      data2_output_attr->strides.erase(data2_output_attr->strides.begin() + axis_idx);
    }
  }
  GELOGI("node %s(%s) after remove gather data1 axis %s in graph %s.", peer_out_nodes[0]->GetName().c_str(),
         peer_out_nodes[0]->GetType().c_str(), AutofuseUtils::VectorToStr(data1_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());
  GELOGI("node %s(%s) after remove gather data2 axis %s in graph %s.", peer_out_nodes[1]->GetName().c_str(),
         peer_out_nodes[1]->GetType().c_str(), AutofuseUtils::VectorToStr(data2_output_attr->axis).c_str(),
         asc_graph.GetName().c_str());
  return SUCCESS;
}

inline Status RemoveGatherInvalidAxis(const AscGraph &asc_graph, const std::vector<int64_t> &graph_invalid_axis_id) {
  if (graph_invalid_axis_id.empty()) {
    return SUCCESS;
  }
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetDirectNode()) {
    if (node->GetType() == kGatherType) {
      GE_ASSERT_SUCCESS(RemoveGatherDataInvalidAxis(asc_graph, node, graph_invalid_axis_id));
      GE_ASSERT_SUCCESS(RemoveGatherInvalidAxisIndex(asc_graph, node, graph_invalid_axis_id));
    }
  }
  return SUCCESS;
}

inline Status UpdateInvalidAxis(const AscGraph &asc_graph, std::vector<int64_t> &graph_invalid_axis_id,
                                std::vector<NodePtr> &gather_data2_nodes) {
  // 如果gather data2全是无效轴，需要保留data2的第一根轴
  for (const auto &data2_node : gather_data2_nodes) {
    if (graph_invalid_axis_id.empty()) {
      return SUCCESS;
    }
    AscTensorAttr *data2_output_attr = nullptr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(data2_node, data2_output_attr));
    auto &axis = data2_output_attr->axis;
    GE_ASSERT_TRUE(!axis.empty());
    bool allFound = std::all_of(axis.begin(), axis.end(), [&graph_invalid_axis_id](int64_t elem) {
      return std::find(graph_invalid_axis_id.begin(), graph_invalid_axis_id.end(), elem) != graph_invalid_axis_id.end();
    });
    if (allFound) {
      auto it = std::find(graph_invalid_axis_id.begin(), graph_invalid_axis_id.end(), axis[0]);
      if (it != graph_invalid_axis_id.end()) {
        graph_invalid_axis_id.erase(it);
        GELOGI("all data2 axis is invalid, id %" PRId64 " is kept in graph %s.", axis[0],
               asc_graph.GetName().c_str());
      }
    }
  }

  // 如果graph全是无效轴，需要保留graph第一根轴
  const auto graph_attr = AscGraphUtils::GetComputeGraph(asc_graph)->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  auto size_t = graph_attr->axis.size();
  if (size_t == graph_invalid_axis_id.size()) {
    GELOGI("all graph axis is invalid, id %" PRId64 " is kept in graph %s.", *(graph_invalid_axis_id.begin()),
           asc_graph.GetName().c_str());
    graph_invalid_axis_id.erase(graph_invalid_axis_id.begin());
  }
  return SUCCESS;
}

inline Status GetAndRemoveInvalidAxis(AscGraph &asc_graph, const NodePtr &asc_node, bool is_fused) {
  if (is_fused) {
    auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
    GE_ASSERT_NOTNULL(autofuse_attr);
    // 0.1、处理FusedAscBackend里面的非concat的AscBackend
    if (autofuse_attr->HasFuseType(loop::FuseType::kConcat)) {
      GELOGI("graph %s fuse type is concat, don't RemoveInvalidAxis.", asc_graph.GetName().c_str());
      return SUCCESS;
    }
    // 0.2、无轴交换的场景，在schedule做无效轴删除；有轴交换场景在后处理做无效轴删除
    bool has_only_one_transpose = false;
    std::unordered_map<NodePtr, std::vector<std::pair<int64_t, int64_t>>> fallback_node_to_transpose_info;
    GE_ASSERT_SUCCESS(
        BackendUtils::GetTransposeInfos(asc_graph, has_only_one_transpose, fallback_node_to_transpose_info));
    if (fallback_node_to_transpose_info.empty()) {
      GELOGI("graph %s fuse type has no transpose, don't RemoveInvalidAxis.", asc_graph.GetName().c_str());
      return SUCCESS;
    }
  }
  // 1、无效轴处理流程前需要把gather的data1和data2特殊补轴处理,并找出data2节点
  std::vector<NodePtr> gather_data2_nodes;
  GE_ASSERT_SUCCESS(FindAndUpdateGatherData(asc_graph, gather_data2_nodes));
  // 2、获取无效轴
  std::vector<int64_t> graph_invalid_axis_id;
  GE_ASSERT_SUCCESS(GetInvalidAxis(asc_graph, graph_invalid_axis_id));
  // 3、gather data2如果全是无效轴需要保留一根轴
  GE_ASSERT_SUCCESS(UpdateInvalidAxis(asc_graph, graph_invalid_axis_id, gather_data2_nodes));
  // 4、根据需要删除的无效轴处理gather的替换轴index和gather data
  GE_ASSERT_SUCCESS(RemoveGatherInvalidAxis(asc_graph, graph_invalid_axis_id));
  // 5、删除和刷新无效轴（使用 std::greater<int64_t> 作为比较器，实现从大到小的排序，删除轴后，剩余轴进行刷轴要从大到小开始刷）
  std::sort(graph_invalid_axis_id.begin(), graph_invalid_axis_id.end(), std::greater<int64_t>());
  GE_ASSERT_SUCCESS(RemoveInvalidAxis(asc_graph, graph_invalid_axis_id));
  return SUCCESS;
}

inline Status RemoveInvalidAxisOnAscGraph(const ComputeGraphPtr &graph, bool is_fused) {
  for (const auto &node : graph->GetDirectNode()) {
    if (!BackendUtils::IsBackendFuseNode(node)) {
      continue;
    }
    std::string proc_name= "remove_invalid_axis";
    if (node->GetType() == kAscBackendType) { // FusedAscBackend里面目前只有concat相关，FusedAscBackend在scheduler展开后做无效轴删除
      GELOGI("before remove invalid axis, AscBackend node(%s), type:%s.", node->GetName().c_str(),
             node->GetType().c_str());
      const auto &op_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      const auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
      GE_ASSERT_NOTNULL(attr);
      GE_ASSERT_NOTNULL(attr->GetAscGraph());
      const auto fused_graph = AscGraphUtils::GetComputeGraph(*(attr->GetAscGraph()));
      GE_ASSERT_SUCCESS(CacheGraphBeforePostProcess(node, proc_name, fused_graph));
      auto ret = GetAndRemoveInvalidAxis(*(attr->GetAscGraph()), node, is_fused);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "AscBackend node(%s %s), post process(%s) failed, start to dump cache graphs;",
               node->GetName().c_str(), node->GetType().c_str(), proc_name.c_str());
        GE_ASSERT_SUCCESS(DumpCacheGraphForExceptionPostProcess(node, proc_name));
        return ret;
      }
      GELOGD("after remove invalid axis, dump node:%s(%s) asc graph info(with tensor attr info):", node->GetNamePtr(),
             node->GetType().c_str());
      BackendUtils::DumpAscGraph(node);
    } else if (node->GetType() == kFusedAscBackendType) {  // FusedAscBackend无效轴删除解决输出多引用给两个reshape，分别在不同的位置加size为1的轴，再后融合concat场景，会反推出transpose，需要删除无效轴
      GELOGI("FusedAscbackend node: %s(%s) start to run the process(%s).", node->GetName().c_str(),
             node->GetType().c_str(), proc_name.c_str());
      GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
      const auto attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
      GE_ASSERT_NOTNULL(attr);
      GE_ASSERT_NOTNULL(attr->GetFuseComputeGraph());
      auto ret = RemoveInvalidAxisOnAscGraph(attr->GetFuseComputeGraph(), true);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "FusedAscBackend node: %s(%s), post process(%s) failed, start to dump cache graphs;",
               node->GetName().c_str(), node->GetType().c_str(), proc_name.c_str());
        GE_ASSERT_SUCCESS(DumpFusedCacheGraphForExceptionPostProcess((attr->GetFuseComputeGraph())->GetName()));
        return ret;
      }
    }
  }
  return SUCCESS;
}

inline Status CompleteNodeAttrsOnAscGraphForSched(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(
      ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, CompleteNodeAttrsOnAscGraph, "complete_attrs"));
  // 1、后端处理不了 repeat为1，stride为0 的无效轴，需要去掉
  // 2、无效轴判断条件：1）非FusedBackendNode，FusedBackendNode里面有concat，concat的轴repeat为1，stride为0是有效的
  //                 2）graph轴id 例如 id_n 对应的 repeat为1，stride为0
  //                 3）全图node的id_n对应的repeat均为1，stride均为0，此id_n才是无效轴
  //                 4）满足条件的id_n对应的graph轴、调度轴、tensor轴的id_n需要删除
  GE_ASSERT_SUCCESS(RemoveInvalidAxisOnAscGraph(ge_or_fused_asc_backend_graph, false));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_COMPLETE_NODE_ATTRS_H
