/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_backward_pass.h"
#include "ge_common/ge_api_error_codes.h"
#include "post_process/post_process_util.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"

namespace ge {
namespace {
std::vector<std::string> view_op_type = {kTransposeType, kBroadcastType, kSliceType, kSplitType, kConcatType, kGatherType, "Sum",
                                         "Mean",         "Max",          "Min",      "Prod",      "Any",       "All"};
Status GetSingleNextNode(NodePtr &node, NodePtr &peer_in_node) {
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(node, peer_in_nodes, 0));

  // 认为每个节点只有一个输出节点
  if (peer_in_nodes.size() != 1U) {
    GELOGI("node:%s(%s) has %zu peer out nodes", node->GetName().c_str(), node->GetType().c_str(),
           peer_in_nodes.size());
    return FAILED;
  }
  peer_in_node = peer_in_nodes.at(0);
  return SUCCESS;
}

bool IsNextViewOp(const NodePtr &next_node) {
  // 通过判断当前节点的类型进行判断
  std::string type = next_node->GetType();
  return std::find(view_op_type.begin(), view_op_type.end(), type) != view_op_type.end();
}

bool IsDtypeNotSupportOp(const NodePtr &next_node, DataType &output_dtype) {
  std::vector<DataType> input_dtypes;
  std::vector<DataType> expect_output_dtypes;
  GeTensorDescPtr output_tensor_desc;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(next_node, output_tensor_desc));
  output_dtype = output_tensor_desc->GetDataType();
  expect_output_dtypes.push_back(output_dtype);
  input_dtypes.push_back(output_dtype);
  return (next_node->GetType() == kCastType) &&
         (AutofuseUtils::CallAscirCommonInferDtype(kBroadcastType, input_dtypes, expect_output_dtypes) != SUCCESS);
}

Status ReverseCollectBrcNodes(const NodePtr &node, vector<NodePtr> &bro_nodes) {
  NodePtr cur_node = node;
  while ((cur_node->GetType() == kBroadcastType) && asc_adapt::IsSingleInAndOutNode(cur_node)) {
    bro_nodes.push_back(cur_node);
    // 获取前驱节点
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(cur_node, cur_node, 0));
  }
  return SUCCESS;
}

Status GetBroAxisFromNode(const NodePtr &bro_node, int64_t &bro_axis) {
  bro_axis = -1;
  NodePtr pre_bro_node;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(bro_node, pre_bro_node, 0));
  AscTensorAttr *pre_bro_output_attr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(pre_bro_node, pre_bro_output_attr));
  auto pre_bro_repeats = pre_bro_output_attr->repeats;
  auto pre_bro_strides = pre_bro_output_attr->strides;
  auto pre_bro_axis = pre_bro_output_attr->axis;

  AscTensorAttr *bro_output_attr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(bro_node, bro_output_attr));
  auto bro_repeats = bro_output_attr->repeats;
  auto bro_strides = bro_output_attr->strides;
  auto bro_attr_axis = bro_output_attr->axis;
  GE_ASSERT_TRUE(pre_bro_repeats.size() == pre_bro_strides.size());
  GE_ASSERT_TRUE(bro_attr_axis.size() == pre_bro_axis.size());
  for (auto index = 0U; index < bro_attr_axis.size(); index++) {
    if ((BackendUtils::IsEqOne(pre_bro_repeats[index])) && BackendUtils::IsEqZero(pre_bro_strides[index])) {
      if (!BackendUtils::IsEqOne(bro_repeats[index])) {
        bro_axis = bro_attr_axis[index];
        return SUCCESS;
      }
    }
  }
  GELOGW(
      "GetBroAxisFromNode failed bro_node %s(%s), axis:%s, repeats:%s stride:%s. pre_bro_node %s(%s), axis:%s, "
      "repeats:%s stride:%s",
      bro_node->GetName().c_str(), bro_node->GetType().c_str(), AutofuseUtils::VectorToStr(bro_attr_axis).c_str(),
      AutofuseUtils::VectorToStr(bro_repeats).c_str(), AutofuseUtils::VectorToStr(bro_strides).c_str(),
      pre_bro_node->GetName().c_str(), pre_bro_node->GetType().c_str(),
      AutofuseUtils::VectorToStr(pre_bro_axis).c_str(), AutofuseUtils::VectorToStr(pre_bro_repeats).c_str(),
      AutofuseUtils::VectorToStr(pre_bro_strides).c_str());
  return FAILED;
}

Status GetBroAxises(const vector<NodePtr> &bro_nodes, vector<int64_t> &bro_axis_idx) {
  for (const auto &bro_node : bro_nodes) {
    int64_t bro_axis;
    GE_ASSERT_SUCCESS(GetBroAxisFromNode(bro_node, bro_axis));
    if (bro_axis >= 0) {
      bro_axis_idx.push_back(bro_axis);
    }
  }
  return SUCCESS;
}

Status GetBroAxisesIndex(vector<size_t> &bro_axis_idx, const vector<ge::Expression> &pre_bro_repeats,
                         const vector<ge::Expression> &pre_bro_strides,
                         const vector<ge::Expression> &last_bro_repeats) {
  GE_ASSERT_TRUE(pre_bro_repeats.size() == pre_bro_strides.size());
  GE_ASSERT_TRUE(pre_bro_repeats.size() == last_bro_repeats.size());
  for (auto index = 0U; index < pre_bro_repeats.size(); index++) {
    if (BackendUtils::IsEqOne(pre_bro_repeats[index]) && BackendUtils::IsEqZero(pre_bro_strides[index])) {
      if (BackendUtils::IsEqOne(last_bro_repeats[index])) {
        continue;
      }
      bro_axis_idx.push_back(index);
    }
  }
  return SUCCESS;
}

bool IsSameBroNodes(const vector<NodePtr> &bro_nodes1, const vector<NodePtr> &bro_nodes2) {
  // 判断两个Brc列表进行的Brc动作整体是否一致
  if (bro_nodes1.size() != bro_nodes2.size()) {
    return false;
  }
  // 比较两个bro链前后是否相同
  std::vector<int64_t> bro_axis_idx1;  // 存储Broadcast轴信息
  std::vector<int64_t> bro_axis_idx2;
  GetBroAxises(bro_nodes1, bro_axis_idx1);
  GetBroAxises(bro_nodes2, bro_axis_idx2);
  return bro_axis_idx1 == bro_axis_idx2;
}

Status RemoveAndRelinkNodeEdge(InDataAnchorPtr &bro_in_anchor, OutDataAnchorPtr &bro_out_anchor) {
  GE_ASSERT_NOTNULL(bro_in_anchor);
  GE_ASSERT_NOTNULL(bro_out_anchor);
  GE_ASSERT_TRUE(!bro_out_anchor->GetPeerInDataAnchors().empty());
  auto before_bro_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  auto after_bro_in_anchor = bro_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(before_bro_out_anchor);
  GE_ASSERT_NOTNULL(after_bro_in_anchor);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(before_bro_out_anchor, bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(bro_out_anchor, after_bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(before_bro_out_anchor, after_bro_in_anchor));
  return SUCCESS;
}

Status RemoveBroadcastOneByOne(vector<NodePtr> &bro_nodes, AscGraph &graph) {
  for (auto &node : bro_nodes) {
    auto bro_in_anchor = node->GetInDataAnchor(0);
    auto bro_out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_SUCCESS(RemoveAndRelinkNodeEdge(bro_in_anchor, bro_out_anchor));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(AscGraphUtils::GetComputeGraph(graph), node));
    NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

Status RemoveBroadcasts(vector<NodePtr> &bro_nodes, AscGraph &graph) {
  // 删除当前的Brc列表
  auto bro_in_anchor = bro_nodes.front()->GetInDataAnchor(0);
  auto bro_out_anchor = bro_nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_SUCCESS(RemoveAndRelinkNodeEdge(bro_in_anchor, bro_out_anchor));
  for (auto &node : bro_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(AscGraphUtils::GetComputeGraph(graph), node));
    NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

std::set<int64_t> FindSubSet(vector<int64_t> &bro_axis_idx1, vector<int64_t> &bro_axis_idx2) {
  std::set<int64_t> common_elements;
  if (bro_axis_idx1.empty() || bro_axis_idx2.empty()) {
    return common_elements;
  }
  // 对两个向量进行降序排序
  std::sort(bro_axis_idx1.begin(), bro_axis_idx1.end(), std::greater<int64_t>());
  std::sort(bro_axis_idx2.begin(), bro_axis_idx2.end(), std::greater<int64_t>());
  auto it1 = bro_axis_idx1.begin();
  auto it2 = bro_axis_idx2.begin();
  while (it1 != bro_axis_idx1.end() && it2 != bro_axis_idx2.end()) {
    if (*it1 == *it2) {
      common_elements.insert(*it1);
      ++it1;
      ++it2;
    } else if (*it1 > *it2) {
      ++it1;
    } else {
      ++it2;
    }
  }
  return common_elements;
}

bool CollectSameBrcAxis(NodePtr &cur_node, NodePtr &next_node, std::vector<std::vector<NodePtr>> &bro_nodes_list,
                        std::set<int64_t> &common_axes, std::vector<NodePtr> &origin_bro_nodes) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(next_node, peer_out_nodes));
  // 判定存在可部分后移的Brc
  for (const auto &node : peer_out_nodes) {
    if ((cur_node != nullptr) && (node == cur_node)) {
      // 当前分支还未后移，Brc节点没有链接到多输入节点，相关信息已经在函数外面做了维护
      continue;
    }
    if (node->GetType() != kBroadcastType) {
      // 不存在Brc节点不用判断
      bro_nodes_list.clear();
      common_axes.clear();
      return false;
    }
    if (origin_bro_nodes.empty()) {
      // 先收集一个基础的比较标准
      ReverseCollectBrcNodes(node, origin_bro_nodes);
      std::reverse(origin_bro_nodes.begin(), origin_bro_nodes.end());
      bro_nodes_list.push_back(origin_bro_nodes);
      continue;
    }
    // 存在Brc，需要比较Brc链是否子集关系
    std::vector<NodePtr> temp_bro_nodes;
    ReverseCollectBrcNodes(node, temp_bro_nodes);
    std::reverse(temp_bro_nodes.begin(), temp_bro_nodes.end());
    bro_nodes_list.push_back(temp_bro_nodes);

    // 比较两个bro链前后是否包含关系
    vector<int64_t> bro_axis_idx1;
    vector<int64_t> bro_axis_idx2;
    if (common_axes.empty()) {
      GetBroAxises(origin_bro_nodes, bro_axis_idx1);
    } else {
      bro_axis_idx1.assign(common_axes.begin(), common_axes.end());
    }
    GetBroAxises(temp_bro_nodes, bro_axis_idx2);
    std::set<int64_t> temp_common_axes = FindSubSet(bro_axis_idx1, bro_axis_idx2);

    if (temp_common_axes.empty()) {
      // 没有公共子集，证明无需后移处理
      bro_nodes_list.clear();
      common_axes.clear();
      return false;
    }
    common_axes = temp_common_axes;
    origin_bro_nodes =
        origin_bro_nodes.size() < temp_bro_nodes.size() ? origin_bro_nodes : temp_bro_nodes;  // bro_nodes保持最小子集
  }
  return true;
}

bool IsMulInputsCanBackward(NodePtr &cur_node, NodePtr &next_node, vector<NodePtr> &bro_nodes, AscGraph &graph,
                            std::set<NodePtr> &mul_input_nodes) {
  auto in_data_anchor_size = next_node->GetAllInDataAnchorsSize();
  if (in_data_anchor_size == 1U) {
    // 单输入场景
    return false;
  }

  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(next_node, peer_out_nodes));
  std::vector<std::vector<NodePtr>> remove_bro_nodes_list;
  for (const auto &node : peer_out_nodes) {
    if (node == cur_node) {
      // 当前节点不比较
      continue;
    }
    if (node->GetType() != kBroadcastType) {
      // 不存在Brc节点不用判断
      return false;
    }

    // 存在Brc，需要比较Brc链是否一致
    std::vector<NodePtr> temp_bro_nodes;
    ReverseCollectBrcNodes(node, temp_bro_nodes);
    std::reverse(temp_bro_nodes.begin(), temp_bro_nodes.end());
    if (!IsSameBroNodes(bro_nodes, temp_bro_nodes)) {
      // 记录多输入节点，用于后续部分Brc场景继续后移
      std::vector<std::vector<NodePtr>> bro_nodes_list;
      std::set<int64_t> common_axes;
      std::vector<NodePtr> origin_bro_nodes;
      origin_bro_nodes.assign(bro_nodes.begin(), bro_nodes.end());
      bro_nodes_list.push_back(origin_bro_nodes);
      if (CollectSameBrcAxis(cur_node, next_node, bro_nodes_list, common_axes, origin_bro_nodes)) {
        mul_input_nodes.insert(next_node);
      }
      return false;
    }
    remove_bro_nodes_list.push_back(temp_bro_nodes);
  }

  // 循环完成证明需要删除其他分支上的bro列表
  for (std::vector<NodePtr> &remove_nodes : remove_bro_nodes_list) {
    RemoveBroadcasts(remove_nodes, graph);
  }
  return true;
}

bool CanBackward(NodePtr &cur_node, NodePtr &next_node, vector<NodePtr> &bro_nodes, AscGraph &graph,
                 std::set<NodePtr> &mul_input_nodes) {
  if (next_node->GetType() == kStoreType) {
    // 当前节点是Store节点时不在后移
    return false;
  }
  if (IsNextViewOp(next_node)) {
    // 下一节点是View类算子时不在后移
    return false;
  }
  if (!asc_adapt::IsSingleOutNode(next_node)) {
    // 下一节点多输出或者输出多引用时不在后移
    return false;
  }
  DataType output_dtype;
  if (IsDtypeNotSupportOp(next_node, output_dtype)) {
    // Broadcast不支持的dtype不进行后移
    GELOGI("Node %s(%s) can not backward with dtype(%s)", next_node->GetName().c_str(), next_node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_dtype).c_str());
    return false;
  }

  if (!asc_adapt::IsSingleInNode(next_node) &&
      !IsMulInputsCanBackward(cur_node, next_node, bro_nodes, graph, mul_input_nodes)) {
    // 多输入场景若且不存在同样的Brc节点能支持后移
    return false;
  }
  return true;
}

Status CollectBroNodes(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &nodes,
                       std::vector<int64_t> &topo_list) {
  if (asc_adapt::IsSingleInAndOutNode(cur_node) && (cur_node->GetType() == kBroadcastType)) {
    nodes.push_back(cur_node);
    const auto &op_desc = cur_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    topo_list.push_back(op_desc->GetId());
  }
  while (asc_adapt::IsSingleInAndOutNode(next_node) && (next_node->GetType() == kBroadcastType)) {
    nodes.push_back(next_node);
    const auto &op_desc = next_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    topo_list.push_back(op_desc->GetId());
    cur_node = next_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
  }
  return SUCCESS;
}

Status CollectCmpNodes(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &nodes,
                       std::vector<int64_t> &topo_list, std::vector<NodePtr> &bro_nodes, AscGraph &graph,
                       std::set<NodePtr> &mul_input_nodes) {
  while (CanBackward(cur_node, next_node, bro_nodes, graph, mul_input_nodes)) {
    nodes.push_back(next_node);
    const auto &op_desc = next_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    topo_list.push_back(op_desc->GetId());
    cur_node = next_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
  }
  return SUCCESS;
}

/**
 * 节点调序逻辑：
 * step1：断开Broadcast之前节点和Broadcast边，断开Broadcast节点及之后Compute节点边，断开可后移Compute节点和不可后移节点之间的边
 * step2：链接Broadcast之前节点和可后移Compute节点，链接可后移Compute节点和Broadcast节点，链接Broadcast节点和不可后移节点
 * *
 * @param compute_nodes 可以支持Broadcast后移的计算节点列表
 * @param bro_nodes broadcast节点列表
 * @return SUCCESS/FAILED
 */
Status ReorderBroadcasts(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &bro_nodes) {
  auto bro_in_anchor = bro_nodes.front()->GetInDataAnchor(0);
  auto bro_out_anchor = bro_nodes.back()->GetOutDataAnchor(0);
  auto comp_out_anchor = compute_nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(bro_in_anchor);
  GE_ASSERT_NOTNULL(bro_out_anchor);
  GE_ASSERT_NOTNULL(comp_out_anchor);
  GE_ASSERT_TRUE(!comp_out_anchor->GetPeerInDataAnchors().empty());
  GE_ASSERT_TRUE(!bro_out_anchor->GetPeerInDataAnchors().empty());
  auto before_bro_out_anchor = bro_in_anchor->GetPeerOutAnchor();
  auto after_comp_in_anchor = comp_out_anchor->GetPeerInDataAnchors().at(0);	
  auto comp_in_anchor = bro_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(before_bro_out_anchor);
  GE_ASSERT_NOTNULL(after_comp_in_anchor);
  GE_ASSERT_NOTNULL(comp_in_anchor);

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(before_bro_out_anchor, bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(bro_out_anchor, comp_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(comp_out_anchor, after_comp_in_anchor));

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(before_bro_out_anchor, comp_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(comp_out_anchor, bro_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(bro_out_anchor, after_comp_in_anchor));
  return SUCCESS;
}

/**
 * topo id更新逻辑：先记录当前从第一个Broadcast节点到后移节点之间的topo序列表
 * 例如：Load（id0)->Broadcast（id1）->Broadcast（id2）->Abs（id3）->Exp（id4）->Store(id5)
 * step1 调序变更位置:
 *      Load（id0)->Abs（id3）->Exp（id4）->Broadcast（id1）->Broadcast（id2）->Store(id5)
 * step2 按顺序先更新可后移计算节点再更新Broadcast节点的topo id：
 *      Load（id0)->Abs（id1）->Exp（id2）->Broadcast（id3）->Broadcast（id4）->Store(id5)
 * @param compute_nodes 可以支持Broadcast后移的计算节点列表
 * @param bro_nodes broadcast节点列表
 * @param topo_list 之前记录的topo序列表
 * @return 如果处理成功返回SUCCESS；否则返回相应的错误码
 */
Status UpdateTopoId(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &bro_nodes,
                    const std::vector<int64_t> &topo_list) {
  size_t index = 0;
  for (const auto &comp_node : compute_nodes) {
    const auto &op_desc = comp_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    GE_ASSERT_TRUE(index < topo_list.size(), "index:%zu is over topo list size:%zu", index, topo_list.size());
    op_desc->SetId(topo_list.at(index));
    index++;
  }
  for (const auto &bro_node : bro_nodes) {
    const auto &op_desc = bro_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    GE_ASSERT_TRUE(index < topo_list.size(), "index:%zu is over topo list size:%zu", index, topo_list.size());
    op_desc->SetId(topo_list.at(index));
    index++;
  }
  return SUCCESS;
}

Status UpdateComputeNodesAscTensorAttr(std::vector<NodePtr> &bro_nodes, std::vector<NodePtr> &compute_nodes,
                                       const NodePtr &pre_bro_node) {
  AscTensorAttr *pre_bro_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(pre_bro_node, pre_bro_output_attr));

  AscTensorAttr *last_bro_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(bro_nodes.back(), last_bro_output_attr));

  std::vector<size_t> bro_axis_idx;  // 存储Broadcast轴信息
  auto pre_bro_axis = pre_bro_output_attr->axis;
  auto pre_bro_repeats = pre_bro_output_attr->repeats;
  auto pre_bro_strides = pre_bro_output_attr->strides;
  auto last_bro_repeats = last_bro_output_attr->repeats;
  GE_ASSERT_SUCCESS(GetBroAxisesIndex(bro_axis_idx, pre_bro_repeats, pre_bro_strides, last_bro_repeats));
  GELOGD("broadcast axis id:%s.", AutofuseUtils::VectorToStr(bro_axis_idx).c_str());
  for (const auto &compute_node : compute_nodes) {
    // 更新输出描述
    AscTensorAttr *compute_output_attr = nullptr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(compute_node, compute_output_attr));
    compute_output_attr->axis = pre_bro_axis;        // axis使用Broadcast前驱节点
    compute_output_attr->repeats = pre_bro_repeats;  // repeats使用Broadcast前驱节点
    compute_output_attr->strides = pre_bro_strides;  // Scalar场景补repeats=1，strides=0 当前补属性已补过可以直接使用
    GE_ASSERT_TRUE(compute_output_attr->strides.size() > 0U);
    auto it = std::find(bro_axis_idx.begin(), bro_axis_idx.end(), pre_bro_strides.size() - 1U);
    if (it != bro_axis_idx.end()) {
      compute_output_attr->strides[compute_output_attr->strides.size() - 1U] = kSymbolZero;
    }
    if (pre_bro_node->GetType() != kScalarType) {
      // strides需要重新计算，不可使用前驱节点（Slice场景load不连续，计算节点应为连续)
      GE_ASSERT_SUCCESS(asc_adapt::UpdateStridesByReapeats(compute_output_attr->repeats, compute_output_attr->strides));
    }
  }
  return SUCCESS;
}

Status UpdateBroadcastNodesDataType(std::vector<NodePtr> &bro_nodes, const NodePtr &last_comp_node) {
  const auto last_comp_opdesc = last_comp_node->GetOpDesc();
  GE_ASSERT_NOTNULL(last_comp_opdesc);
  const auto comp_output_tensor_desc = last_comp_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(comp_output_tensor_desc);
  auto last_comp_dtype = comp_output_tensor_desc->GetDataType();
  for (const auto &bro_node : bro_nodes) {
    // 更新每个Broadcast节点的DataType
    const auto bro_opdesc = bro_node->GetOpDesc();
    GE_ASSERT_NOTNULL(bro_opdesc);
    const auto bro_output_tensor_desc = bro_opdesc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(bro_output_tensor_desc);
    bro_output_tensor_desc->SetDataType(last_comp_dtype);
  }
  return SUCCESS;
}

/**
 * 实现后移Broadcast逻辑
 *
 * @param compute_nodes 可以移到Broadcast节点前的计算节点列表
 * @param bro_nodes 可以后移的Broadcast节点列表
 * @param topo_list 收集移动前的顺序 Broadcast节点列+计算节点列的topo id列表
 * @param pre_bro_node Broadcast节点列的前驱节点，用于移动后计算节点的属性更新
 * @return 如果处理成功返回SUCCESS；否则返回相应的错误码
 */
Status BroadcastBackwardReally(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &bro_nodes,
                               const std::vector<int64_t> &topo_list, const NodePtr &pre_bro_node) {
  // Broadcast节点调序
  GE_ASSERT_SUCCESS(ReorderBroadcasts(compute_nodes, bro_nodes));

  // 更新后移后的topo id
  GE_ASSERT_SUCCESS(UpdateTopoId(compute_nodes, bro_nodes, topo_list));

  // 更新可移动计算节点的输出shape描述tensor
  GE_ASSERT_SUCCESS(UpdateComputeNodesAscTensorAttr(bro_nodes, compute_nodes, pre_bro_node));

  // 更新Broadcast节点的data_type(计算节点中存在Cast移动后需要更新Broadcast保持精度)
  GE_ASSERT_SUCCESS(UpdateBroadcastNodesDataType(bro_nodes, compute_nodes.back()));
  return SUCCESS;
}

/**
 * 找到图上所有brc的前驱节点，作为后移判断开始的起点
 */
Status CollectBackwardStartNodes(const AscGraph &graph, std::vector<NodePtr> &pre_brc_nodes) {
  for (const auto &node : graph.GetAllNodes()) {
    NodePtr cur_node = node;
    while ((cur_node->GetType() == kBroadcastType) && asc_adapt::IsSingleOutNode(cur_node)) {
      // 循环经过单输出单引用的Brc节点链
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(cur_node, cur_node, 0));
    }
    // 当前节点输出为UbScalar时，broadcast节点不做后移
    AscTensorAttr *output_tensor_attr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(cur_node, output_tensor_attr));
    bool is_ub_scalar = AutofuseUtils::IsUbScalar(output_tensor_attr->repeats);
    if ((cur_node != node) && (cur_node->GetType() != kScalarType) && !is_ub_scalar) {
      // 至少存在单输出的Brc节点就把前驱节点当作输出(Scalar和UbScalar无法判断后续计算节点是否支持暂时不做后移)
      pre_brc_nodes.push_back(cur_node);
    }
  }
  return SUCCESS;
}

/**
 * 找到满足判断后移条件的节点起点
 */
Status CollectBackwardSatisfyStartNodes(const NodePtr &node, vector<NodePtr> &peer_in_nodes) {
  // 获取节点所有输出
  auto output_size = node->GetAllOutDataAnchorsSize();
  for (uint32_t idx = 0U; idx < output_size; ++idx) {
    std::vector<NodePtr> temp_nodes;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerInNodes(node, temp_nodes, idx));
    if (!temp_nodes.empty() && (temp_nodes.front()->GetType() == kBroadcastType)) {
      // 找到对应的存在Brc的分支
      peer_in_nodes.insert(peer_in_nodes.end(), temp_nodes.begin(), temp_nodes.end());
    }
  }
  return SUCCESS;
}

Status RemoveBroadcasts(AscGraph &graph, std::vector<std::vector<NodePtr>> &bro_nodes_list,
                        const std::set<int64_t> &common_axises) {
  // 根据bro axis信息删除其中的broadcast节点
  for (std::vector<NodePtr> &bro_nodes : bro_nodes_list) {
    // 这里不能全部删除，得保留不可后移的部分
    std::vector<NodePtr> remove_nodes;
    for (auto it = bro_nodes.begin(); it != bro_nodes.end();) {
      int64_t bro_axis;
      GE_ASSERT_SUCCESS(GetBroAxisFromNode(*it, bro_axis));
      if ((bro_axis != -1) && common_axises.count(bro_axis) != 0) {
        remove_nodes.push_back(*it);
        it = bro_nodes.erase(it);  // 原brc列表存储剩余需要更新属性的
      } else {
        ++it;
      }
    }
    GE_ASSERT_SUCCESS(RemoveBroadcastOneByOne(remove_nodes, graph));
  }
  return SUCCESS;
}

Status GetBackwardBrcNodes(const std::vector<NodePtr> &origin_bro_nodes,
                           std::vector<NodePtr> &origin_need_move_bro_nodes, const std::set<int64_t> &common_axises) {
  for (auto &bro_node : origin_bro_nodes) {
    int64_t bro_axis;
    GE_ASSERT_SUCCESS(GetBroAxisFromNode(bro_node, bro_axis));
    if ((bro_axis != -1) && common_axises.count(bro_axis) != 0) {
      origin_need_move_bro_nodes.push_back(bro_node);
    }
  }
  return SUCCESS;
}

Status CreateAndUpdateBroadcastNode(AscGraph &asc_graph, const NodePtr &node, NodePtr &connect_node,
                                    asc_adapt::TensorInfo &tensor_info) {
  const std::vector<int64_t> &broadcast_info = tensor_info.broadcast_info;
  GE_ASSERT_TRUE(broadcast_info.size() > 0U);
  for (auto index = 0U; index < broadcast_info.size(); index++) {
    const auto b_node = asc_adapt::CreateBroadcastNode(asc_graph, node, broadcast_info, index);
    GE_ASSERT_NOTNULL(b_node);

    int32_t anchor_idx = 0;
    GE_ASSERT_TRUE(node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() == 1U);
    anchor_idx = node->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(b_node, connect_node, {anchor_idx}, {}));
    GE_ASSERT_GRAPH_SUCCESS(
        GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), connect_node->GetInDataAnchor(anchor_idx)));

    GE_ASSERT_SUCCESS(asc_adapt::UpdateBroadcastNodeAttrs(b_node, tensor_info.axis, tensor_info.repeats,
                                                          tensor_info.strides, broadcast_info[index]));
    GE_ASSERT_SUCCESS(asc_adapt::UpdateBroadcastNodeSchedInfo(b_node, tensor_info.sched_axis));
    GE_ASSERT_SUCCESS(asc_adapt::UpdateNodeTopoInfo(b_node, tensor_info.current_topo_id));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(b_node, DT_FLOAT, tensor_info.dtype));
    tensor_info.current_topo_id--;
    connect_node = b_node;
  }
  return SUCCESS;
}

Status InsertBroadcastNode(NodePtr &pre_bro_node, AscGraph &graph, std::set<int64_t> &broadcast_axis) {
  std::vector<int64_t> broadcast_info(broadcast_axis.begin(), broadcast_axis.end());
  asc_adapt::TensorInfo tensor_info;
  GE_ASSERT_SUCCESS(asc_adapt::GetTensorInfo(pre_bro_node, tensor_info));
  const auto pre_bro_node_opdesc = pre_bro_node->GetOpDesc();
  GE_ASSERT_NOTNULL(pre_bro_node_opdesc);
  GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(graph, pre_bro_node, broadcast_info.size()));
  tensor_info.current_topo_id = pre_bro_node_opdesc->GetId() + broadcast_info.size();
  tensor_info.broadcast_info = broadcast_info;
  NodePtr connect_node;
  GE_ASSERT_SUCCESS(GetSingleNextNode(pre_bro_node, connect_node));
  GE_ASSERT_SUCCESS(CreateAndUpdateBroadcastNode(graph, pre_bro_node, connect_node, tensor_info));
  return SUCCESS;
}

Status UpdateOutputTensor(std::vector<NodePtr> &nodes, std::set<int64_t> &common_axises) {
  for (auto &node : nodes) {
    AscTensorAttr *compute_output_attr = nullptr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(node, compute_output_attr));
    auto &repeats = compute_output_attr->repeats;
    auto &strides = compute_output_attr->strides;
    auto &attr_axis = compute_output_attr->axis;
    for (auto common_axis : common_axises) {
      auto it = std::find(attr_axis.begin(), attr_axis.end(), common_axis);
      GE_ASSERT_TRUE(it != attr_axis.end());
      auto index = std::distance(attr_axis.begin(), it);
      repeats[index] = kSymbolOne;
      strides[index] = kSymbolZero;
    }
    GE_ASSERT_SUCCESS(asc_adapt::UpdateTensorAttrsIfNotEmpty(node, attr_axis, repeats, compute_output_attr));
  }
  return SUCCESS;
}

/**
 * 1.找到对应可后移的Broadcast axis轴信息集合
 * 2.计算出后移到的位置，记录后移要更新的计算节点，记录下一次判定需要的多输入节点
 * 3.删除各个分支上对应的Broadcast节点
 * 4.插入Broadcast节点（包含更新Broadcast节点的outputTensor和datatype),更新插入Broadcast后的相关节点相应的topoId
 * 5.更新计算节点outputTensor(直接刷新对应轴的strides、repeats同时刷新Bro列表 可以不考虑Brc之间的位置关系)
 * 6.继续迭代更新下一轮可后移节点
 */
Status JudgePartBackward(std::set<NodePtr> &mul_input_nodes, bool &is_changed, AscGraph &graph) {
  std::set<NodePtr> next_mul_input_nodes;
  for (auto mul_input_node : mul_input_nodes) {
    std::vector<std::vector<NodePtr>> bro_nodes_list;  // 记录多输入，每个输入的全量broadcast链
    std::vector<NodePtr> origin_bro_nodes;             // 记录多输入的来源链中全量的broadcast
    std::vector<NodePtr> origin_need_move_bro_nodes;   // 记录多输入的来源链中需要后移的broadcast
    std::vector<NodePtr> compute_nodes;                // 记录多输入的来源链中需要后移的计算节点
    std::set<int64_t> common_axises;                   // 存储可后移的Brc轴
    std::vector<int64_t> topo_list;                    // 复用方法保持结构，实际无需使用
    NodePtr peer_out_node;                             // 复用方法保持结构，实际无需使用
    if (!CollectSameBrcAxis(peer_out_node, mul_input_node, bro_nodes_list, common_axises, origin_bro_nodes)) {
      continue;
    }
    if (bro_nodes_list.empty() || origin_bro_nodes.size() > 1U) {
      // 没有可后移的公共Brc,当前仅针对多输入中只有单个Brc场景后移动
      GELOGI("bro_nodes_list size %zu, and same broadcast axis axis size: %zu.", bro_nodes_list.size(),
             origin_bro_nodes.size());
      continue;
    }

    // 收集所有可后移的Broadcast节点以及广播轴信息
    GE_ASSERT_SUCCESS(GetBackwardBrcNodes(origin_bro_nodes, origin_need_move_bro_nodes, common_axises));

    // 收集所有可后移的计算节点
    auto cur_node = mul_input_node;
    auto next_node = mul_input_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
    compute_nodes.push_back(mul_input_node);
    GE_ASSERT_SUCCESS(
        CollectCmpNodes(cur_node, next_node, compute_nodes, topo_list, origin_bro_nodes, graph, next_mul_input_nodes));
    GELOGD("node:%s(%s), common_axes:%s, compute_nodes:%s, origin_bro_nodes:%s", mul_input_node->GetName().c_str(),
           mul_input_node->GetType().c_str(), AutofuseUtils::SetToStr(common_axises).c_str(),
           AutofuseUtils::VectorToStr(compute_nodes).c_str(), AutofuseUtils::VectorToStr(origin_bro_nodes).c_str());

    is_changed = true;
    // 删除所有broadcast节点
    GE_ASSERT_SUCCESS(RemoveBroadcasts(graph, bro_nodes_list, common_axises));

    // 插入Broadcast节点
    GE_ASSERT_SUCCESS(InsertBroadcastNode(compute_nodes.back(), graph, common_axises));

    // 更新brc列表和计算节点的outputTensor包含bro_nodes_list和compute_nodes
    std::vector<NodePtr> merged_nodes;
    for (const auto &row : bro_nodes_list) {
      merged_nodes.insert(merged_nodes.end(), row.begin(), row.end());
    }
    merged_nodes.insert(merged_nodes.end(), compute_nodes.begin(), compute_nodes.end());
    GE_ASSERT_SUCCESS(UpdateOutputTensor(merged_nodes, common_axises));
  }
  // 迭代下一次
  if (!next_mul_input_nodes.empty()) {
    return JudgePartBackward(next_mul_input_nodes, is_changed, graph);
  }
  return SUCCESS;
}

/**
 * Broadcast节点位置后移，目的是性能优化，减少算子计算成本：
 * 1.找到Broadcast节点链列表，如果后续节点是Store证明已经是循环外提后无需操作
 * 2.通过判断后续节点是是单输入单输出且单引用且非Store节点记录Broadcast节点列表和可后移的节点列表
 * 3.通过anchor间断边加边将Broadcast节点进行后移
 * 4.递补更新后移后的topo id及更新计算节点的Tensor信息
 */
Status BroadcastBackward(AscGraph &graph, [[maybe_unused]] const NodePtr &asc_node) {
  if (BackendUtils::IsCubeAscNode(asc_node)) {
    GELOGI("graph %s fuse type is cube, don't backward broadcast.", graph.GetName().c_str());
    return SUCCESS;
  }
  bool is_changed = false;            // 记录是否产生变化，用于决定最后是否调用整图排序
  std::set<NodePtr> mul_input_nodes;  // 记录多输入节点，用于再次判断能否后移部分Brc节点
  std::vector<NodePtr> start_nodes;
  GE_ASSERT_SUCCESS(CollectBackwardStartNodes(graph, start_nodes));  // 收集所有Brc前驱节点
  asc_adapt::RemoveDuplicates(start_nodes);                          // 去重
  for (const auto &start_node : start_nodes) {
    // 获取该节点全部的后驱输出节点
    std::vector<NodePtr> peer_in_nodes;  // 获取输入节点的所有输出分支，此处列表存储Load节点列表（Scalar无Load节点）
    GE_ASSERT_SUCCESS(
        CollectBackwardSatisfyStartNodes(start_node, peer_in_nodes));  // 收集真正存在Brc节点路径的起始节点
    for (const auto &peer_in_node : peer_in_nodes) {
      // 注意Scalar无Load节点
      NodePtr cur_node = start_node;
      NodePtr next_node = peer_in_node;

      std::vector<int64_t> topo_list;      // 记录原始的Broadcast+可后移计算节点的topo id 列表
      std::vector<NodePtr> bro_nodes;      // 记录Broadcast节点列表
      std::vector<NodePtr> compute_nodes;  // 记录Broadcast可后移的计算节点列表
      NodePtr pre_bro_node = start_node;   // 记录后移前Broadcast的前驱节点，用于属性更新
      // 收集 Broadcast 节点, 当前多引用的Broadcast节点不处理
      GE_ASSERT_SUCCESS(CollectBroNodes(cur_node, next_node, bro_nodes, topo_list));

      // 收集可后移的计算节点
      GE_ASSERT_SUCCESS(
          CollectCmpNodes(cur_node, next_node, compute_nodes, topo_list, bro_nodes, graph, mul_input_nodes));

      GELOGI("Processing node %s(%s) with %zu broadcast nodes and %zu compute nodes.", peer_in_node->GetName().c_str(),
             peer_in_node->GetType().c_str(), bro_nodes.size(), compute_nodes.size());

      // 当存在broadcast节点且后续存在可后移节点时 进行调序
      if (bro_nodes.empty() || compute_nodes.empty()) {
        continue;
      }
      is_changed = true;
      GE_ASSERT_SUCCESS(BroadcastBackwardReally(compute_nodes, bro_nodes, topo_list, pre_bro_node));
    }
  }

  if (!mul_input_nodes.empty()) {
    // 针对Brc多输入场景，需要再判断是否存在部分Brc后移可能
    GE_ASSERT_SUCCESS(JudgePartBackward(mul_input_nodes, is_changed, graph));
  }
  if (is_changed) {
    asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(graph));
  }
  return SUCCESS;
}
}  // namespace

Status BroadcastBackwardPass::Run(const ComputeGraphPtr &graph) {
  GE_ASSERT_SUCCESS(asc_adapt::ProcessAscBackendNodes(graph, BroadcastBackward, "broadcast_backward_pass"));
  GELOGI("Graph %s completed BroadcastBackward successfully.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
