/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <regex>

#include "asc_backend_fusion_decider.h"
#include "common/checker.h"
#include "fusion_decider_registry.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/autofuse_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "utils/auto_fuse_config.h"
#include "utils/not_fuse_reason_code.h"

namespace ge {
using namespace autofuse;
// 子图融合流程缓存dump图和缓存当前正在融合的节点名字
Status CacheGraphBeforeSubGraphMerge(const NodePtr &node1, const NodePtr &node2, const ComputeGraphPtr &origin_graph) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  // 缓存dump图，此处缓存的origin_graph是FusedAscBackend的merged_graph
  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node1));
  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node2));
  GE_ASSERT_SUCCESS(BackendUtils::CacheCurrentGraphName(node1->GetName(), node2->GetName(), origin_graph->GetName()));
  return SUCCESS;
}

// 子图融合流程缓存dump图和缓存当前正在融合的节点名字
Status CacheGraphAfterSubGraphMerge(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                                    const ComputeGraphPtr &merged_graph) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  // 后处理的子图dump后需要dump融合过程，如果没有在融合后缓存new_node下一次融合就没法通过node1或者node2来缓存这一次new_node
  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(new_node->GetName(), merged_graph));
  // 缓存dump图映射
  GE_ASSERT_SUCCESS(BackendUtils::AddSubGraphMergeGraphMap(
      new_node->GetName(), node1->GetName(),
      node2->GetName()));  // 不能直接使用graph1来获取name，中间流程会更改graph1的名字
  return SUCCESS;
}

// 缓存dump图和缓存当前正在融合的节点名字
Status CacheGraphBeforeMerge(const NodePtr &node1, const NodePtr &node2) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node1));
  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node2));
  GE_ASSERT_SUCCESS(BackendUtils::CacheCurrentGraphName(node1->GetName(), node2->GetName()));
  return SUCCESS;
}

// 缓存dump图和缓存当前正在融合的节点名字
Status CacheGraphAfterMerge(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                            const ComputeGraphPtr &merged_graph) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  // 后处理的子图dump后需要dump融合过程，如果没有在融合后缓存new_node下一次融合就没法通过node1或者node2来缓存这一次new_node
  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(new_node->GetName(), merged_graph));
  // FusedAscBackend的子图dump后需要dump FusedAscBackend的融合过程
  GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(merged_graph->GetName(), merged_graph));
  // 缓存dump图映射,如果是FusedAscBackend，在子图融合的时候需要找到融合获得FusedAscBackend的过程dump图
  GE_ASSERT_SUCCESS(BackendUtils::AddMergeGraphMap(
      new_node->GetName(), node1->GetName(), node2->GetName(),
      merged_graph->GetName()));  // 不能直接使用node1的子图graph1来获取name，中间流程会更改graph1的名字
  return SUCCESS;
}

bool AscBackendSubGraphFusionDecider::CanFuse(const NodePtr &node1, const NodePtr &node2) const {
  uint32_t max_fusion_node_input_size = AutoFuseConfig::Config().GetFusionStrategySolver().max_input_nums_after_fuse;
  if (!BackendUtils::CanFuseByStrategy(node1, node2, max_fusion_node_input_size)) {
    return false;
  }

  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  GE_ASSERT_TRUE((node1->GetType() == kAscBackendType) && (node2->GetType() == kAscBackendType));
  GE_ASSERT_SUCCESS(BackendUtils::ResetFusedSubgraphOutputsAttr(node1));
  GE_ASSERT_SUCCESS(BackendUtils::ResetFusedSubgraphOutputsAttr(node2));
  // 尝试做轴映射，如果无法映射融合失败
  AscGraphAxisMapping graph_axis_map;
  NodeFuseInfo node_fuse_info(false);
  if (node_fuse_info.UpdateNodeFuseInfo(node1, node2) != SUCCESS) {
    return false;
  }
  if (UnifySubgraphAxis(node1, node2, node_fuse_info, graph_axis_map, false) != SUCCESS) {
    GELOGD(
        "AscBackendGraphFuse: check fusedAscBackend graph fuse, node %s(%s) and node %s(%s) can't unify subgraph axis, "
        "can fuse false",
        node1->GetName().c_str(), node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str());
    return false;
  }
  if (CanMergeAscGraph(graph1, graph2, node1, node2) != SUCCESS) {
    GELOGD(
        "AscBackendGraphFuse: node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Node1 and node2's "
        "ascgraph can not merge]", node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
        node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kCanNotMergeAscGraph));
    return false;
  }
  // 判断group merge是否成功
  optimize::autoschedule::AxisGroup axes_group1;
  if (BackendUtils::GetAscGraphAxisGroup(node1, axes_group1, graph_axis_map.GetNode1AxisMap()) != SUCCESS) {
    GELOGD("get node %s(%s) axis group failed, can fuse false", node1->GetNamePtr(), node1->GetType().c_str());
    return false;
  }
  optimize::autoschedule::AxisGroup axes_group2;
  if (BackendUtils::GetAscGraphAxisGroup(node2, axes_group2, graph_axis_map.GetNode2AxisMap()) != SUCCESS) {
    GELOGD("get node %s(%s) axis group failed, can fuse false", node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }
  optimize::autoschedule::AxisGroup merged_axes_group;
  if (!BackendUtils::IsCanMergeAxisGroup(axes_group1, axes_group2, merged_axes_group)) {
    GELOGD("node %s(%s) and node %s(%s) axis group merge failed, can fuse false", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }
  GELOGI("AscBackendGraphFuse: check fusedAscBackend graph fuse, node: %s(%s) and %s(%s) can loop merge.",
         node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
  return true;
}

NodePtr AscBackendSubGraphFusionDecider::Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  const auto &origin_graph = node1->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(origin_graph);
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  GELOGI("AscBackendGraphFuse: before fuse fusedAscBackend graph: %s, fuse node: %s(%s) and %s(%s).",
         origin_graph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
         node2->GetType().c_str());
  // 子图融合流程异常时dump图的缓存融合前dump图流程
  GE_ASSERT_SUCCESS(CacheGraphBeforeSubGraphMerge(node1, node2, origin_graph));

  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph1, node1));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph2, node2));
  AscGraphAxisMapping graph_axis_map(false);
  GE_ASSERT_SUCCESS(UnifySubgraphAxis(node1, node2, node_fuse_info, graph_axis_map));
  GE_ASSERT_SUCCESS(BackendUtils::TuningSubgraphBeforeMerge(node1, node2, graph1, graph2, node_fuse_info));
  const auto merged_graph = MergeAscGraphByLoop(graph1, graph2, node1, node2, node_fuse_info);
  GE_ASSERT_NOTNULL(merged_graph);
  auto new_node = FuseNode(node1, node2, merged_graph, node_fuse_info, counter);
  GE_ASSERT_NOTNULL(new_node);
  GELOGI("AscBackendGraphFuse: merged asc graph: %s, parent node: %s(%s).", merged_graph->GetName().c_str(),
         new_node->GetNamePtr(), new_node->GetType().c_str());

  GELOGI("AscBackendGraphFuse: after fuse fusedAscBackend graph: %s, fuse node: %s(%s) and %s(%s).",
         origin_graph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
         node2->GetType().c_str());
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(merged_graph, new_node));
  GE_ASSERT_SUCCESS(UpdateSubgraphAxisAttr(new_node, node1, node2));
  GE_ASSERT_SUCCESS(UpdateAxisGroupInfo(node1, node2, new_node, graph_axis_map));
  // 子图融合流程异常时dump图的缓存融合后dump图流程
  GE_ASSERT_SUCCESS(CacheGraphAfterSubGraphMerge(new_node, node1, node2, merged_graph));

  // 融合后做反推和补轴，下次融合时具备完整信息
  GE_ASSERT_SUCCESS(BackendUtils::ProcessAscgraphAfterMerge(new_node));
  return new_node;
}

Status AscBackendFusionDecider::UpdateSubgraphAxisAttr(const NodePtr &new_node, const NodePtr &node1,
                                                       const NodePtr &node2) const {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  ComputeGraphPtr new_graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(new_node, new_graph));

  auto graph_attr1 = graph1->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr1);
  auto graph_attr2 = graph2->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr2);
  auto new_graph_attr = new_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(new_graph_attr);
  GELOGD("node1: %s(%s) origin axis info: %s, node2: %s(%s) origin axis info: %s.", node1->GetNamePtr(),
         node1->GetType().c_str(), BackendUtils::AscAxisToStr(graph_attr1->axis).c_str(), node2->GetNamePtr(),
         node2->GetType().c_str(), BackendUtils::AscAxisToStr(graph_attr2->axis).c_str());

  // 处理reduce场景，如果后续是reduce后的graph，那需要用前序的轴size
  auto new_graph_attr2 = graph_attr2;
  for (auto index1 = 0U; index1 < new_graph_attr2->axis.size(); index1++) {
    if (!BackendUtils::IsEqOne(new_graph_attr2->axis[index1]->size)) {
      continue;
    }
    for (auto index2 = 0U; index2 < graph_attr1->axis.size(); index2++) {
      if (graph_attr1->axis[index2]->id != new_graph_attr2->axis[index1]->id) {
        continue;
      }
      new_graph_attr2->axis[index1]->size = graph_attr1->axis[index2]->size;
    }
  }
  // 如果轴个数相同按照后序graph更新，否则用轴个数最多的子图作为新图的属性
  if (graph_attr1->axis.size() == graph_attr2->axis.size()) {
    new_graph_attr->axis = new_graph_attr2->axis;
  } else {
    if (graph_attr1->axis.size() > graph_attr2->axis.size()) {
      new_graph_attr->axis = graph_attr1->axis;
    } else {
      new_graph_attr->axis = new_graph_attr2->axis;
    }
  }
  GELOGD("node: %s(%s) subgraph axis update success, merged axis info: %s.", new_node->GetNamePtr(),
         new_node->GetType().c_str(), BackendUtils::AscAxisToStr(new_graph_attr->axis).c_str());
  return SUCCESS;
}

Status AscBackendFusionDecider::CanMergeAscGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                                                 const NodePtr &node1, const NodePtr &node2) const {
  (void)subgraph1;
  (void)subgraph2;
  // todo concat或者其他模版类型处理
  if (!BackendUtils::CanMergeLoopByStrategy(node1, node2)) {
    return FAILED;
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::UpdateAscGraph(const ComputeGraphPtr &subgraph,
                                               const ComputeGraph::Vistor<NodePtr> &subgraph1_nodes,
                                               const ComputeGraph::Vistor<NodePtr> &subgraph2_nodes,
                                               const NodeFuseInfo &node_fuse_info) const {
  auto update_exec_order = [](const ComputeGraph::Vistor<NodePtr> &nodes, int64_t offset) -> Status {
    for (const auto &node : nodes) {
      const auto &op_desc = node->GetOpDesc();
      op_desc->SetId(op_desc->GetId() + offset);
    }
    return SUCCESS;
  };

  auto get_max_exec_order = [](const ComputeGraph::Vistor<NodePtr> &nodes, int64_t &max_topo_id) -> Status {
    max_topo_id = 0;
    for (const auto &node : nodes) {
      const auto &op_desc = node->GetOpDesc();
      if (op_desc->GetId() > max_topo_id) {
        max_topo_id = op_desc->GetId();
      }
    }
    return SUCCESS;
  };

  // 垂直融合使用前序graph的最大topo id更新后序graph，水平融合需要根据topo排序id刷新
  if (!node_fuse_info.GetNode1ToNode2LinkMap().empty()) {
    int64_t max_topo_id;
    GE_ASSERT_SUCCESS(get_max_exec_order(subgraph1_nodes, max_topo_id));
    max_topo_id++;
    GE_ASSERT_SUCCESS(update_exec_order(subgraph2_nodes, max_topo_id));
  } else {
    // 水平融合先做topo默认排序, 和后端排序统一
    GE_ASSERT_GRAPH_SUCCESS(subgraph->TopologicalSorting());
  }
  return SUCCESS;
}

ComputeGraphPtr AscBackendFusionDecider::MergeAscGraphByLoop(const ComputeGraphPtr &subgraph1,
                                                             const ComputeGraphPtr &subgraph2, const NodePtr &node1,
                                                             const NodePtr &node2, NodeFuseInfo &node_fuse_info) const {
  const auto subgraph1_input_nodes = subgraph1->GetInputNodes();
  const auto subgraph2_input_nodes = subgraph2->GetInputNodes();
  auto autofuse_attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(autofuse_attr1);
  auto &subgraph1_output_nodes = GetInterAttrs(autofuse_attr1).fused_subgraph_outputs;
  auto subgraph1_is_reduction = GetInterAttrs(autofuse_attr1).IsReduction();
  GELOGI("subgraph: %s, output num=%zu, is_reduction=%d.", subgraph1->GetName().c_str(),
         GetInterAttrs(autofuse_attr1).fused_subgraph_outputs.size(), subgraph1_is_reduction);

  auto subgraph1_nodes = subgraph1->GetAllNodes();
  auto subgraph2_nodes = subgraph2->GetAllNodes();

  // 把子图2上的input node加入到子图1中，为了保序
  for (const auto &subgraph2_input_node : subgraph2_input_nodes) {
    GE_ASSERT_NOTNULL(subgraph1->AddInputNode(subgraph2_input_node));
  }

  for (const auto &node : subgraph2_nodes) {
    // Input node已经单独加过了
    if (BackendUtils::IsInputNode(node)) {
      continue;
    }
    GE_ASSERT_NOTNULL(subgraph1->AddNode(node));
  }
  std::vector<NodePtr> del_data_nodes;
  std::vector<NodePtr> del_output_and_store_nodes;
  // 获取两个节点相同的输入节点信息, 把子图2上的data合入到子图1上
  if (node_fuse_info.CanDoHorizontalMapping()) {
    for (const auto &same_input : node_fuse_info.GetSameInputMap()) {
      GE_ASSERT_TRUE(static_cast<size_t>(same_input.first) < subgraph1_input_nodes.size(), "size %zu VS size %zu",
                     static_cast<size_t>(same_input.first), subgraph1_input_nodes.size());
      GE_ASSERT_TRUE(static_cast<size_t>(same_input.second) < subgraph2_input_nodes.size(), "size %zu VS size %zu",
                     static_cast<size_t>(same_input.second), subgraph2_input_nodes.size());
      const auto data1 = subgraph1_input_nodes.at(same_input.first);
      const auto data2 = subgraph2_input_nodes.at(same_input.second);
      // 完全相同的data或者只有一个是view才能合并，包括repeat和strides也要相同，如果不同需要回退接口融合的map信息，保证每个data都有正确的输入
      if (BackendUtils::TryAscDataNodeMerge(node1, node2, data1, data2)) {
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(data1, data2, {}, {0}));
        del_data_nodes.push_back(data2);
      } else {
        // 如果不能合并，需要回退映射关系，后面生成的融合节点的输入anchor才能正确
        GELOGW("data nodes can't merge, need rollback node2 input map info.");
        node_fuse_info.RollbackNode2InputMap(same_input.second);
      }
    }
  }
  // 处理node1的输出链接node2
  GE_ASSERT_GRAPH_SUCCESS(LinkAscSubGraphNode(node1, node2, subgraph1_output_nodes, subgraph2_input_nodes,
                                              node_fuse_info, del_data_nodes, del_output_and_store_nodes,
                                              subgraph1_is_reduction));

  // 更新新子图输出
  GE_ASSERT_SUCCESS(
      BackendUtils::UpdateSubGraphOutput(subgraph1_output_nodes, node_fuse_info.GetNode1ToNode2LinkMap()));
  // 在这之前需要使用原来owner信息
  for (const auto &node : subgraph2_nodes) {
    GE_ASSERT_SUCCESS(node->SetOwnerComputeGraph(subgraph1));
  }
  // 删除失效node
  // slice和split算子在能同时进行水平融合和垂直融合时只进行垂直融合，需要对节点去重，避免重复删除
  std::set<NodePtr> del_data_nodes_set(del_data_nodes.begin(), del_data_nodes.end());
  for (const auto &data_node : del_data_nodes_set) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(subgraph1, data_node));
    NodeUtils::UnlinkAll(*data_node);
  }
  // slice和split算子在能同时进行水平融合和垂直融合时只进行垂直融合，需要对节点去重，避免重复删除
  std::set<NodePtr> del_output_and_store_nodes_set(del_output_and_store_nodes.begin(), del_output_and_store_nodes.end());
  for (const auto &invalid_node : del_output_and_store_nodes_set) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(subgraph1, invalid_node));
    NodeUtils::UnlinkAll(*invalid_node);
  }
  // 刷新新ascgraph的执行序
  GE_ASSERT_SUCCESS(UpdateAscGraph(subgraph1, subgraph1_nodes, subgraph2_nodes, node_fuse_info));
  return subgraph1;
}

Status AscBackendFusionDecider::UpdateNewNodeAttr(const OpDescPtr op, const NodePtr &node1, const NodePtr &node2,
                                                  const NodeFuseInfo &node_fuse_info) const {
  auto attr = GetOrCreateAutoFuseAttrs(op);
  GE_ASSERT_NOTNULL(attr);
  auto autofuse_attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(autofuse_attr1);
  auto autofuse_attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(autofuse_attr2);
  GetInterAttrs(attr).origin_nodes = GetInterAttrs(autofuse_attr1).origin_nodes;
  GetInterAttrs(attr).origin_nodes.insert(GetInterAttrs(attr).origin_nodes.end(),
                                          GetInterAttrs(autofuse_attr2).origin_nodes.begin(),
                                          GetInterAttrs(autofuse_attr2).origin_nodes.end());
  GetInterAttrs(attr).vector_core_num = GetInterAttrs(autofuse_attr1).vector_core_num;

  auto merge_output_buffer = [](const std::vector<ge::OutDataAnchor *> &vec, const std::vector<int32_t> &indices,
                                std::vector<ge::OutDataAnchor *> &result) -> Status {
    if (vec.empty()) {
      return SUCCESS;
    }
    for (const auto &index : indices) {
      if (index == -1) {
        continue;
      }
      GE_ASSERT_TRUE(static_cast<size_t>(index) < vec.size());
      result.push_back(vec[index]);
    }
    return SUCCESS;
  };
  GE_ASSERT_SUCCESS(merge_output_buffer(GetInterAttrs(autofuse_attr1).output_buffers,
                                        node_fuse_info.GetNode1OutputMap(), GetInterAttrs(attr).output_buffers));
  GE_ASSERT_SUCCESS(merge_output_buffer(GetInterAttrs(autofuse_attr2).output_buffers,
                                        node_fuse_info.GetNode2OutputMap(), GetInterAttrs(attr).output_buffers));

  auto &subgraph1_output_nodes = GetInterAttrs(autofuse_attr1).fused_subgraph_outputs;
  auto &subgraph2_output_nodes = GetInterAttrs(autofuse_attr2).fused_subgraph_outputs;
  GetInterAttrs(attr).fused_subgraph_outputs = subgraph1_output_nodes;
  GetInterAttrs(attr).fused_subgraph_outputs.insert(GetInterAttrs(attr).fused_subgraph_outputs.end(),
                                                    subgraph2_output_nodes.begin(), subgraph2_output_nodes.end());

  auto fuse_type =
      MergeFuseType(GetInterAttrs(autofuse_attr1).fuse_type, GetInterAttrs(autofuse_attr2).fuse_type);
  attr->SetAscGraph(BackendUtils::GetNodeFusedAscGraph(node1), autofuse_attr1->GetFuseType());
  GetInterAttrs(attr).fuse_type = fuse_type;

  return SUCCESS;
}

std::string ExtractNameFromAscBackend(const std::string &input) {
  /**
   * lowering命名规则：autofuse_<pattern_name>_<unique_number>_[<ordered_node_type>...]
   * can_fuse命名规则：autofuse_<unique_number>_[<ordered_node_type>...]/autofuse_fused_<unique_number>_[<ordered_node_type>...]
   */
  std::regex pattern(R"(autofuse(_fused)?(?:_[^_]+)?_([0-9]+)_(.*))");
  std::smatch match;
  const size_t pos = 3U;  // ordered_node_type匹配捕获组的位置

  if (std::regex_search(input, match, pattern)) {
    // 提取 ordered_node_type
    return match.size() > pos ? match[pos].str() : "";
  }
  return "";
}

std::string CreateFuseNodeName(const NodePtr &node1, const NodePtr &node2, const std::string &node_type,
                               const CounterPtr &counter) {
  if (counter == nullptr) {
    return "fused_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  // pattern_name暂时不命名，后续定位需要补充
  std::stringstream ss;
  std::string ordered_node_type;

  // 判断topo id 融合命名时将id小的排在前面
  const auto &op_desc1 = node1->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc1);
  auto node1_id = op_desc1->GetId();
  const auto &op_desc2 = node2->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc2);
  auto node2_id = op_desc2->GetId();
  auto node1_ordered_type = ExtractNameFromAscBackend(node1->GetName());
  auto node2_ordered_type = ExtractNameFromAscBackend(node2->GetName());
  if (node1_id < node2_id) {
    ordered_node_type = node1_ordered_type + (node1_ordered_type.empty() ? "" : "_") + node2_ordered_type;
  } else {
    ordered_node_type = node2_ordered_type + (node2_ordered_type.empty() ? "" : "_") + node1_ordered_type;
  }
  if (node_type == kAscBackendType) {
    ss << "autofuse_" << counter->NextId() << "_" << ordered_node_type;
  } else if (node_type == kFusedAscBackendType) {
    ss << "autofuse_fused_" << counter->NextId() << "_" << ordered_node_type;
  }
  std::string node_name = ss.str();
  if (node_name.size() > AutoFuseConfig::FusionStrategySolverConfig().max_op_name_len) {
    // 判断超长截断
    node_name = node_name.substr(0, AutoFuseConfig::FusionStrategySolverConfig().max_op_name_len);
  }
  return node_name;
}

NodePtr AscBackendFusionDecider::FuseNode(NodePtr node1, NodePtr node2, const ComputeGraphPtr merged_graph,
                                          const NodeFuseInfo &node_fuse_info, const CounterPtr &counter) const {
  auto graph = node1->GetOwnerComputeGraph();
  GE_ASSERT_EQ(graph, node2->GetOwnerComputeGraph());

  // 创建一个融合node
  std::string node_type = (merged_graph == nullptr) ? kFusedAscBackendType : kAscBackendType;
  auto node_name = CreateFuseNodeName(node1, node2, node_type, counter);
  auto asc_op = OperatorFactory::CreateOperator(node_name.c_str(), node_type.c_str());
  asc_op.BreakConnect();
  asc_op.DynamicInputRegister("inputs", node_fuse_info.GetNode2InputMap().size());
  asc_op.DynamicOutputRegister("outputs", node_fuse_info.GetNode2OutputMap().size());
  auto asc_desc = OpDescUtils::GetOpDescFromOperator(asc_op);
  GE_ASSERT_NOTNULL(asc_desc);
  GE_ASSERT_SUCCESS(AutofuseUtils::AddOperatorPrototypeAttrs(asc_desc));
  if (merged_graph != nullptr) {
    merged_graph->SetName(node_name);
  }
  GE_ASSERT_SUCCESS(UpdateNewNodeAttr(asc_desc, node1, node2, node_fuse_info));
  auto new_node = graph->AddNode(asc_desc);
  GE_ASSERT_NOTNULL(new_node);
  GE_ASSERT_SUCCESS(new_node->SetOwnerComputeGraph(graph));

  GELOGI("replace data anchors from node %s(%s) to node %s(%s), input map %s, output map %s.", node1->GetNamePtr(),
         node1->GetType().c_str(), new_node->GetNamePtr(), new_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode1InputMap()).c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode1OutputMap()).c_str());
  GELOGI("replace data anchors from node %s(%s) to node %s(%s), input map %s, output map %s.", node2->GetNamePtr(),
         node2->GetType().c_str(), new_node->GetNamePtr(), new_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode2InputMap()).c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode2OutputMap()).c_str());
  // 为新节点的每个输入输出描述创建属性组
  GE_ASSERT_SUCCESS(BackendUtils::CreateNewNodeInputDescAttr(new_node, node1, node2, node_fuse_info.GetNode1InputMap(),
                                                             node_fuse_info.GetNode2InputMap()));
  GE_ASSERT_SUCCESS(BackendUtils::CreateNewNodeOutputDescAttr(
      new_node, node1, node2, node_fuse_info.GetNode1OutputMap(), node_fuse_info.GetNode2OutputMap()));

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, node1, node_fuse_info.GetNode1InputMap(),
                                                             node_fuse_info.GetNode1OutputMap()));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, node2, node_fuse_info.GetNode2InputMap(),
                                                             node_fuse_info.GetNode2OutputMap()));
  GE_ASSERT_SUCCESS(UpdateNewNodeAndGENodeOutputMappingRelation(new_node, node1, node2, node_fuse_info));
  GE_ASSERT_SUCCESS(UpdateNewNodeAndGENodeInputMappingRelation(new_node, node1, node2, node_fuse_info));
  GE_ASSERT_SUCCESS(SetReduceFusedElementwiseNodeNum(new_node, node1, node2));
  GE_ASSERT_SUCCESS(SetSplitNodeGlobalId(new_node, node1, node2));
  GE_ASSERT_SUCCESS(SetSplitNodeConcreteEdges(new_node, node1, node2));
  GE_ASSERT_SUCCESS(BackendUtils::TryRemoveNodesCtrEdges(node1, node2));
  GE_ASSERT_SUCCESS(BackendUtils::TryRemoveNodesCtrEdges(node2, node1));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveInCtrlEdges(node1, new_node));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveInCtrlEdges(node2, new_node));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveOutCtrlEdges(node1, new_node));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveOutCtrlEdges(node2, new_node));
  GELOGI("node %s(%s) and node %s(%s) fuse success.", node1->GetNamePtr(), node1->GetType().c_str(),
         node2->GetNamePtr(), node2->GetType().c_str());
  // 原图上清理掉融合后的节点
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNodes(graph, {node1, node2}));
  NodeUtils::UnlinkAll(*node1);
  NodeUtils::UnlinkAll(*node2);
  return new_node;
}

ComputeGraphPtr AscBackendFusionDecider::CreateAscBackendNodeSubGraph(
    const NodePtr &node, uint32_t in_nums, uint32_t out_nums, const std::vector<uint32_t> &node_output_index,
    const std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes) const {
  const std::string graph_name = "FusedAscBackendNode_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber());
  ComputeGraphPtr sub_graph;
  GE_ASSERT_SUCCESS(AutofuseUtils::CreateComputeGraphWithGraphID(node, graph_name, sub_graph));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveNodeToGraph(node, *sub_graph));
  GE_ASSERT_SUCCESS(BackendUtils::CreateSubGraphInput(sub_graph, node, in_nums, pre_nodes));
  GE_ASSERT_SUCCESS(BackendUtils::CreateSubGraphOutput(sub_graph, node, out_nums, node_output_index));
  return sub_graph;
}

Status AscBackendFusionDecider::LinkDataNode(InDataAnchorPtr &in_anchor, const NodePtr &data_node,
                                             bool need_del_edge) const {
  auto in_anchor_peer = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(in_anchor_peer);
  if (need_del_edge) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(in_anchor_peer, in_anchor));
  }
  auto data_out_anchor = data_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(data_out_anchor);
  for (const auto &data_out_anchor_peer : data_out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(data_out_anchor_peer);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceEdgeSrc(data_out_anchor, data_out_anchor_peer, in_anchor_peer));
  }

  // 添加控制边
  auto out_anchor = in_anchor_peer->GetOwnerNode()->GetOutControlAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &data_out_anchor_peer : data_out_anchor->GetPeerInDataAnchors()) {
    auto in_anchor = data_out_anchor_peer->GetOwnerNode()->GetInControlAnchor();
    GE_ASSERT_NOTNULL(in_anchor);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(out_anchor, in_anchor));
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::LinkAscSubGraphNode(const NodePtr &node1, const NodePtr &node2,
    std::vector<ge::NodePtr> &outputs, const ComputeGraph::Vistor<NodePtr> &inputs, const NodeFuseInfo &node_fuse_info,
    std::vector<NodePtr> &del_data_nodes, std::vector<NodePtr> &del_output_and_store_nodes, bool is_reduction) const {
  for (const auto &subgraph_link : node_fuse_info.GetNode1ToNode2LinkMap()) {
    // 把data的输出anchor和netoutput的peer out anchor replace
    GE_ASSERT_TRUE(static_cast<size_t>(subgraph_link.first) < outputs.size(), "size %zu VS size %zu",
                   static_cast<size_t>(subgraph_link.first), outputs.size());
    GE_ASSERT_TRUE(static_cast<size_t>(subgraph_link.second) < inputs.size(), "size %zu VS size %zu",
                   static_cast<size_t>(subgraph_link.second), inputs.size());
    auto output_node = outputs.at(subgraph_link.first);
    GE_ASSERT_NOTNULL(output_node);
    auto output_in_anchor = output_node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(output_in_anchor);
    auto output_in_anchor_peer = output_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(output_in_anchor_peer);
    auto store_node = output_in_anchor_peer->GetOwnerNode();
    GE_ASSERT_NOTNULL(store_node);
    auto data_node = inputs.at(subgraph_link.second);
    GE_ASSERT_NOTNULL(data_node);
    auto store_in_anchor = store_node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(store_in_anchor);
    auto in_anchor_peer = store_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(in_anchor_peer);
    auto store_peer_node = in_anchor_peer->GetOwnerNode();
    GE_ASSERT_NOTNULL(store_peer_node);
    GE_ASSERT_SUCCESS(LinkDataNode(store_in_anchor, data_node, node_fuse_info.IsSingleReference(subgraph_link.first)));
    del_data_nodes.push_back(data_node);
    // 单输出才需要删除store
    if (node_fuse_info.IsSingleReference(subgraph_link.first)) {
      del_output_and_store_nodes.push_back(store_node);
      del_output_and_store_nodes.push_back(output_node);
    }
    // 删除无效的load节点
    GE_ASSERT_SUCCESS(BackendUtils::DeleteInvalidLoad(node1, node2, store_peer_node, in_anchor_peer, data_node,
                                                      del_output_and_store_nodes, is_reduction));
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::LinkSubGraphNode(const NodePtr &subgraph_netoutput,
                                                 const ComputeGraph::Vistor<NodePtr> &inputs,
                                                 const std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map,
                                                 std::vector<NodePtr> &del_data_nodes) const {
  for (const auto &subgraph_link : subgraph_link_map) {
    // 把data的输出anchor和netoutput的peer out anchor replace
    auto netoutput_in_anchor = subgraph_netoutput->GetInDataAnchor(subgraph_link.first);
    GE_ASSERT_NOTNULL(netoutput_in_anchor);
    GE_ASSERT_TRUE(static_cast<size_t>(subgraph_link.second) < inputs.size(), "size %zu VS size %zu",
                   static_cast<size_t>(subgraph_link.second), inputs.size());
    auto data_node = inputs.at(subgraph_link.second);
    GE_ASSERT_NOTNULL(data_node);
    GE_ASSERT_SUCCESS(LinkDataNode(netoutput_in_anchor, data_node));
    del_data_nodes.push_back(data_node);
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::MergeSubGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                                              const NodeFuseInfo &node_fuse_info) const {
  const auto subgraph1_input_nodes = subgraph1->GetInputNodes();
  GE_ASSERT_EQ(subgraph1_input_nodes.size(), static_cast<size_t>(node_fuse_info.GetNode1InDataSize()));
  const auto subgraph2_input_nodes = subgraph2->GetInputNodes();
  GE_ASSERT_EQ(subgraph2_input_nodes.size(), static_cast<size_t>(node_fuse_info.GetNode2InDataSize()));

  // 把子图2上的input node加入到子图1中，为了保序
  for (const auto &subgraph2_input_node : subgraph2_input_nodes) {
    GE_ASSERT_NOTNULL(subgraph1->AddInputNode(subgraph2_input_node));
  }

  auto subgraph1_netoutput = subgraph1->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(subgraph1_netoutput);
  auto subgraph2_netoutput = subgraph2->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(subgraph2_netoutput);
  for (const auto &node : subgraph2->GetAllNodes()) {
    GE_ASSERT_SUCCESS(node->SetOwnerComputeGraph(subgraph1));
    // 输入已经单独加过了
    if (BackendUtils::IsInputNode(node)) {
      continue;
    }
    GE_ASSERT_NOTNULL(subgraph1->AddNode(node));
  }
  std::vector<NodePtr> del_data_nodes;
  if (node_fuse_info.CanDoHorizontalMapping()) {
    // 获取两个节点相同的输入节点信息, 把子图2上的data合入到子图1上
    for (const auto &same_input : node_fuse_info.GetSameInputMap()) {
      GE_ASSERT_TRUE(static_cast<size_t>(same_input.first) < subgraph1_input_nodes.size(), "size %zu VS size %zu",
                     static_cast<size_t>(same_input.first), subgraph1_input_nodes.size());
      GE_ASSERT_TRUE(static_cast<size_t>(same_input.second) < subgraph2_input_nodes.size(), "size %zu VS size %zu",
                     static_cast<size_t>(same_input.second), subgraph2_input_nodes.size());
      const auto data1 = subgraph1_input_nodes.at(same_input.first);
      const auto data2 = subgraph2_input_nodes.at(same_input.second);
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(data1, data2, {}, {0}));
      del_data_nodes.push_back(data2);
    }
  }
  // 处理node1的输出链接node2
  GE_ASSERT_GRAPH_SUCCESS(LinkSubGraphNode(subgraph1_netoutput, subgraph2_input_nodes,
                                           node_fuse_info.GetNode1ToNode2LinkMap(), del_data_nodes));

  GE_ASSERT_TRUE(static_cast<size_t>(node_fuse_info.GetNode1OutNodeSize() + node_fuse_info.GetNode2OutNodeSize()) >=
                 (node_fuse_info.GetNode1ToNode2LinkMap().size()));
  const auto new_output_input_num =
      static_cast<size_t>(node_fuse_info.GetNode1OutNodeSize() + node_fuse_info.GetNode2OutNodeSize()) -
      (node_fuse_info.GetNode1ToNode2LinkMap().size());
  // 给merge后的子图创建netoutput, 把原来两个netoutput replace过来
  const auto net_output_node = BackendUtils::CreateNetOutput(subgraph1, new_output_input_num);
  GE_ASSERT_NOTNULL(net_output_node);
  std::vector<int32_t> node1_output_map;
  BackendUtils::GetOutputMap(node_fuse_info.GetNode1ToNode2LinkMap(), node_fuse_info.GetNode1OutNodeSize(),
                             node1_output_map);
  std::vector<int32_t> node2_output_map(node1_output_map.size(), -1);
  BackendUtils::GetOutputMap({}, node_fuse_info.GetNode2OutNodeSize(), node2_output_map);

  GELOGI("replace data anchors from node %s(%s) to node %s(%s), input map %s, output map {}.",
         subgraph1_netoutput->GetNamePtr(), subgraph1_netoutput->GetType().c_str(), net_output_node->GetNamePtr(),
         net_output_node->GetType().c_str(), AutofuseUtils::VectorToStr(node1_output_map).c_str());
  GELOGI("replace data anchors from node %s(%s) to node %s(%s), input map %s, output map {}.",
         subgraph2_netoutput->GetNamePtr(), subgraph2_netoutput->GetType().c_str(), net_output_node->GetNamePtr(),
         net_output_node->GetType().c_str(), AutofuseUtils::VectorToStr(node2_output_map).c_str());
  // 为新节点的每个输出描述创建属性组
  GE_ASSERT_SUCCESS(BackendUtils::CreateNewNodeOutputDescAttr(net_output_node, subgraph1_netoutput, subgraph2_netoutput,
                                                              node1_output_map, node2_output_map));

  GE_ASSERT_GRAPH_SUCCESS(
      GraphUtils::ReplaceNodeDataAnchors(net_output_node, subgraph1_netoutput, node1_output_map, {}));
  GE_ASSERT_GRAPH_SUCCESS(
      GraphUtils::ReplaceNodeDataAnchors(net_output_node, subgraph2_netoutput, node2_output_map, {}));

  // 删除失效node
  // slice和split算子在能同时进行水平融合和垂直融合时只进行垂直融合，需要对节点去重，避免重复删除
  std::set<NodePtr> del_data_nodes_set(del_data_nodes.begin(), del_data_nodes.end());
  for (const auto &data_node : del_data_nodes_set) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(subgraph1, data_node));
    NodeUtils::UnlinkAll(*data_node);
  }
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(subgraph1, subgraph1_netoutput));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(subgraph1, subgraph2_netoutput));
  NodeUtils::UnlinkAll(*subgraph2_netoutput);
  NodeUtils::UnlinkAll(*subgraph2_netoutput);
  return SUCCESS;
}

Status AscBackendFusionDecider::GetFusePossibilityLinkNodes(
    const NodePtr &subgraph_netoutput, const ComputeGraph::Vistor<NodePtr> &inputs,
    const std::pair<int32_t, int32_t> &link_map, std::set<std::pair<NodePtr, NodePtr>> &link_ascnode_map) const {
  const auto netoutput_in_anchor = subgraph_netoutput->GetInDataAnchor(link_map.first);
  GE_ASSERT_NOTNULL(netoutput_in_anchor);
  GE_ASSERT_TRUE(static_cast<size_t>(link_map.second) < inputs.size(), "size %zu VS size %zu",
                 static_cast<size_t>(link_map.second), inputs.size());
  const auto &data_node = inputs.at(link_map.second);
  GE_ASSERT_NOTNULL(data_node);
  auto in_anchor_peer = netoutput_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(in_anchor_peer);

  const auto data_out_anchor = data_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(data_out_anchor);
  const auto data_out_size = data_out_anchor->GetPeerInDataAnchors().size();
  for (size_t i = 0U; i < data_out_size; i++) {
    auto data_out_anchor_peer = data_out_anchor->GetPeerInDataAnchors().at(i);
    GE_ASSERT_NOTNULL(data_out_anchor_peer);
    link_ascnode_map.insert(std::make_pair(in_anchor_peer->GetOwnerNode(), data_out_anchor_peer->GetOwnerNode()));
  }
  return SUCCESS;
}

// fusedAscBackend node融合后会产生新的AscBackend node直接的链接关系，寻找AscBackend node间可能循环合并的节点对
Status AscBackendFusionDecider::GetAllFusePossibilityNodes(const ComputeGraphPtr &subgraph1,
                                                           const ComputeGraphPtr &subgraph2,
                                                           std::set<std::pair<NodePtr, NodePtr>> &fuse_possibility_map,
                                                           const NodeFuseInfo &node_fuse_info) const {
  const auto subgraph1_input_nodes = subgraph1->GetInputNodes();
  GE_ASSERT_EQ(subgraph1_input_nodes.size(), static_cast<size_t>(node_fuse_info.GetNode1InDataSize()));
  const auto subgraph2_input_nodes = subgraph2->GetInputNodes();
  GE_ASSERT_EQ(subgraph2_input_nodes.size(), static_cast<size_t>(node_fuse_info.GetNode2InDataSize()));
  const auto subgraph1_netoutput = subgraph1->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(subgraph1_netoutput);
  auto subgraph2_netoutput = subgraph2->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(subgraph2_netoutput);

  if(node_fuse_info.CanDoHorizontalMapping()) {
    for (const auto &same_input : node_fuse_info.GetSameInputMap()) {
      GE_ASSERT_TRUE(static_cast<size_t>(same_input.first) < subgraph1_input_nodes.size(), "size %zu VS size %zu",
                     static_cast<size_t>(same_input.first), subgraph1_input_nodes.size());
      GE_ASSERT_TRUE(static_cast<size_t>(same_input.second) < subgraph2_input_nodes.size(), "size %zu VS size %zu",
                     static_cast<size_t>(same_input.second), subgraph2_input_nodes.size());
      auto data1 = subgraph1_input_nodes.at(same_input.first);
      auto data2 = subgraph2_input_nodes.at(same_input.second);
      auto data1_out_anchor = data1->GetOutDataAnchor(0);
      GE_ASSERT_NOTNULL(data1_out_anchor);
      auto data1_out_size = data1_out_anchor->GetPeerInDataAnchors().size();
      auto data2_out_anchor = data2->GetOutDataAnchor(0);
      GE_ASSERT_NOTNULL(data2_out_anchor);
      auto data2_out_size = data2_out_anchor->GetPeerInDataAnchors().size();

      for (size_t i = 0U; i < data1_out_size; ++i) {
        auto data1_out_anchor_peer = data1_out_anchor->GetPeerInDataAnchors().at(i);
        GE_ASSERT_NOTNULL(data1_out_anchor_peer);
        for (size_t j = 0U; j < data2_out_size; ++j) {
          auto data2_out_anchor_peer = data2_out_anchor->GetPeerInDataAnchors().at(j);
          GE_ASSERT_NOTNULL(data2_out_anchor_peer);
          fuse_possibility_map.insert(
              std::make_pair(data1_out_anchor_peer->GetOwnerNode(), data2_out_anchor_peer->GetOwnerNode()));
        }
      }
    }
  }

  for (const auto &node1_to_node2_link : node_fuse_info.GetNode1ToNode2LinkMap())
    GE_ASSERT_SUCCESS(GetFusePossibilityLinkNodes(subgraph1_netoutput, subgraph2_input_nodes, node1_to_node2_link,
                                                  fuse_possibility_map));
  return SUCCESS;
}

Status AscBackendFusionDecider::AddIndexForFusedAscBackendNode(const ComputeGraphPtr &merged_subgraph) const {
  // FusedAscBackend子图中的data添加对应index属性
  int64_t index = -1;
  auto input_nodes = merged_subgraph->GetInputNodes();
  for (const auto &input_node : input_nodes) {
    GE_ASSERT_TRUE(AttrUtils::SetInt(input_node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, ++index),
                   "[Set][Attr] %s on subgraph node:%s(%s) failed.", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                   input_node->GetNamePtr(), input_node->GetType().c_str());
  }
  return SUCCESS;
}

ComputeGraphPtr AscBackendFusionDecider::MergeGraphToFusedAscBackendNode(const ComputeGraphPtr &subgraph1,
                                                                         const ComputeGraphPtr &subgraph2,
                                                                         const NodePtr &node1, const NodePtr &node2,
                                                                         const NodePtr &fused_node,
                                                                         const NodeFuseInfo &node_fuse_info) const {
  auto merged_subgraph = subgraph1;
  auto to_be_merged_subgraph = subgraph2;

  // 如果一个node是AscBackend type, 先创建一个子图，再做合并，合并逻辑归一
  if (node1->GetType() == kAscBackendType) {
    merged_subgraph =
        CreateAscBackendNodeSubGraph(node1, node_fuse_info.GetNode1InDataSize(), node_fuse_info.GetNode1OutNodeSize(),
                                     node_fuse_info.GetNode1OutputIndex(), node_fuse_info.GetNode1PreNodes());
  }
  if (node2->GetType() == kAscBackendType) {
    to_be_merged_subgraph =
        CreateAscBackendNodeSubGraph(node2, node_fuse_info.GetNode2InDataSize(), node_fuse_info.GetNode2OutNodeSize(),
                                     node_fuse_info.GetNode2OutputIndex(), node_fuse_info.GetNode2PreNodes());
  }
  GE_ASSERT_NOTNULL(merged_subgraph);
  GE_ASSERT_NOTNULL(to_be_merged_subgraph);
  GELOGI("AscBackendGraphFuse: before merge AscBackend node graph1: %s, parent node: %s(%s).",
         merged_subgraph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str());
  GELOGI("AscBackendGraphFuse: before merge AscBackend node graph2: %s, parent node: %s(%s).",
         to_be_merged_subgraph->GetName().c_str(), node2->GetNamePtr(), node2->GetType().c_str());

  // 获取两个FuseAscBackendNode字图里面的AscBackendNode再次融合的可能,在merge前做，merge后状态会改变
  std::set<std::pair<NodePtr, NodePtr>> fuse_possibility_map;
  GE_ASSERT_SUCCESS(GetPossibleFuseNodePairFromSubgraph(fuse_possibility_map, merged_subgraph));
  GE_ASSERT_SUCCESS(GetPossibleFuseNodePairFromSubgraph(fuse_possibility_map, to_be_merged_subgraph));
  GE_ASSERT_SUCCESS(
      GetAllFusePossibilityNodes(merged_subgraph, to_be_merged_subgraph, fuse_possibility_map, node_fuse_info));
  for (const auto &node_pair : fuse_possibility_map) {
    GELOGI("node %s(%s) and node %s(%s) possible fuse again.", node_pair.first->GetNamePtr(),
           node_pair.first->GetType().c_str(), node_pair.second->GetNamePtr(), node_pair.second->GetType().c_str());
  }
  GE_ASSERT_SUCCESS(MergeSubGraph(merged_subgraph, to_be_merged_subgraph, node_fuse_info));
  GELOGI("AscBackendGraphFuse: merged AscBackend node graph: %s, parent node: %s(%s).",
         merged_subgraph->GetName().c_str(), fused_node->GetNamePtr(), fused_node->GetType().c_str());

  // FusedAscBackend子图中的data添加对应index属性
  GE_ASSERT_SUCCESS(AddIndexForFusedAscBackendNode(merged_subgraph));

  // 准备再次融合数据
  if (!fuse_possibility_map.empty()) {
    auto attr = GetOrCreateAutoFuseAttrs(merged_subgraph);
    GE_ASSERT_NOTNULL(attr);
    if (GetInterAttrs(attr).decider == nullptr) {
      GetInterAttrs(attr).decider = std::move(ComGraphMakeUnique<AscBackendSubGraphFusionDecider>());
    }
    GetInterAttrs(attr).possible_fusion_nodes.insert(fuse_possibility_map.begin(), fuse_possibility_map.end());
  }
  return merged_subgraph;
}

Status AscBackendFusionDecider::UnifySubgraphAxis(const NodePtr &node1, const NodePtr &node2,
                                                  const NodeFuseInfo &fuse_info, AscGraphAxisMapping &graph_axis_map,
                                                  bool need_flash) const {
  if (graph_axis_map.CreateSubGraphAxisMapInfo(node1, node2, fuse_info) != SUCCESS) {
    GELOGD_IF(graph_axis_map.IsOpenLog(), "node %s(%s) and node %s(%s) can't align axis info, can fuse false.",
              node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return FAILED;
  }
  // ascgraph中的轴映射为一套轴，can fuse的时候不真正映射
  if (graph_axis_map.FlushSubGraphAxisInfo(node1, graph_axis_map.GetNode1AxisMap(), need_flash) != SUCCESS) {
    GELOGD_IF(graph_axis_map.IsOpenLog(), "node %s(%s) have some axis no map info in ascgraph, can fuse false",
              node1->GetNamePtr(), node1->GetType().c_str());
    return FAILED;
  }
  if (graph_axis_map.FlushSubGraphAxisInfo(node2, graph_axis_map.GetNode2AxisMap(), need_flash) != SUCCESS) {
    GELOGD_IF(graph_axis_map.IsOpenLog(), "node %s(%s) have some axis no map info in ascgraph, can fuse false",
              node2->GetNamePtr(), node2->GetType().c_str());
    return FAILED;
  }
  // axis group轴映射为一套轴
  if (need_flash) {
    GE_ASSERT_SUCCESS(UpdateFusedAscGraphAxisGroup(node1, graph_axis_map.GetNode1AxisMap()));
    GE_ASSERT_SUCCESS(UpdateFusedAscGraphAxisGroup(node2, graph_axis_map.GetNode2AxisMap()));
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::UpdateFusedAscGraphAxisGroup(const NodePtr &node, const AxisPairSet &axis_map) const {
  if (node->GetType() == kFusedAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
    for (auto &sub_node : graph->GetAllNodes()) {
      if (sub_node->GetType() == kAscBackendType) {
        optimize::autoschedule::AxisGroup axes_group;
        GE_ASSERT_SUCCESS(BackendUtils::GetAscGraphAxisGroup(node, axes_group, axis_map));
        auto attr = BackendUtils::GetNodeAutoFuseAttr(node);
        GE_ASSERT_NOTNULL(attr);
        GetInterAttrs(attr).axis_group = axes_group;
      }
    }
  }
  return SUCCESS;
}

// 把融合后的axis group信息更新到融合节点属性上，融合节点再次融合的时候直接使用group
Status AscBackendFusionDecider::UpdateAxisGroupInfo(const NodePtr &node1, const NodePtr &node2, const NodePtr &new_node,
                                                    const AscGraphAxisMapping &graph_axis_map) const {
  optimize::autoschedule::AxisGroup axes_group1;
  GE_ASSERT_SUCCESS(BackendUtils::GetAscGraphAxisGroup(node1, axes_group1, graph_axis_map.GetNode1AxisMap()));
  auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GetInterAttrs(attr1).axis_group = axes_group1;
  optimize::autoschedule::AxisGroup axes_group2;
  GE_ASSERT_SUCCESS(BackendUtils::GetAscGraphAxisGroup(node2, axes_group2, graph_axis_map.GetNode2AxisMap()));
  auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GetInterAttrs(attr2).axis_group = axes_group2;
  optimize::autoschedule::AxisGroup merged_axes_group;
  GE_ASSERT_TRUE(BackendUtils::IsCanMergeAxisGroup(axes_group1, axes_group2, merged_axes_group));
  auto attr = BackendUtils::GetNodeAutoFuseAttr(new_node);
  GE_ASSERT_NOTNULL(attr);
  GetInterAttrs(attr).axis_group = merged_axes_group;
  GELOGD("node:%s(%s) asc subgraph axis group update success.", new_node->GetNamePtr(), new_node->GetType().c_str());
  return SUCCESS;
}

NodePtr AscBackendFusionDecider::Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  ComputeGraphPtr merged_graph;
  const auto &origin_graph = node1->GetOwnerComputeGraph();
  GELOGI("AscBackendGraphFuse: before fuse origin graph: %s, fuse node: %s(%s) and %s(%s).",
         origin_graph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
         node2->GetType().c_str());
  // 异常时dump图的缓存融合前dump图流程
  GE_ASSERT_SUCCESS(CacheGraphBeforeMerge(node1, node2));

  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph1, node1));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph2, node2));
  // 真正的轴映射，刷新两个子图的轴为一套轴
  AscGraphAxisMapping graph_axis_map(false);
  GE_ASSERT_SUCCESS(UnifySubgraphAxis(node1, node2, node_fuse_info, graph_axis_map));
  // AscBackend类型应该尝试做循环合并
  if ((node1->GetType() == kAscBackendType) && (node2->GetType() == kAscBackendType)) {
    GELOGI("node %s(%s) and node %s(%s) are AscBackendType, try to loop merge.", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    if (CanMergeAscGraph(graph1, graph2, node1, node2) == SUCCESS) {
      GE_ASSERT_SUCCESS(BackendUtils::TuningSubgraphBeforeMerge(node1, node2, graph1, graph2, node_fuse_info));
      merged_graph = MergeAscGraphByLoop(graph1, graph2, node1, node2, node_fuse_info);
    }
  }
  const auto new_node = FuseNode(node1, node2, merged_graph, node_fuse_info, counter);
  GE_ASSERT_NOTNULL(new_node);
  // 如果不能循环合并，创建一个FusedAscBackend node子图
  if (merged_graph == nullptr) {
    merged_graph = MergeGraphToFusedAscBackendNode(graph1, graph2, node1, node2, new_node, node_fuse_info);
    GE_ASSERT_NOTNULL(merged_graph, "Failed to merge graph for node %s and node %s.", node1->GetNamePtr(),
                      node2->GetNamePtr());
    GELOGI("node %s(%s) and node %s(%s) fuse to FusedAscbackend node (%s) subgraph.", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(), new_node->GetNamePtr());
    auto attr = BackendUtils::GetNodeAutoFuseAttr(new_node);
    GE_ASSERT_NOTNULL(attr);
    attr->SetFuseComputeGraph(merged_graph);

    // 更新新子图输出
    auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(new_node);
    GE_ASSERT_NOTNULL(autofuse_attr);
    auto &subgraph_output_nodes = GetInterAttrs(autofuse_attr).fused_subgraph_outputs;
    GE_ASSERT_SUCCESS(
        BackendUtils::UpdateSubGraphOutput(subgraph_output_nodes, node_fuse_info.GetNode1ToNode2LinkMap()));
  } else {
    GELOGI("AscBackendGraphFuse: merged asc graph: %s, parent node: %s(%s).", merged_graph->GetName().c_str(),
           new_node->GetNamePtr(), new_node->GetType().c_str());
  }
  GELOGI("AscBackendGraphFuse: after fuse origin graph: %s, fuse node: %s(%s) and %s(%s) to %s(%s).",
         origin_graph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
         node2->GetType().c_str(), new_node->GetNamePtr(), new_node->GetType().c_str());
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(merged_graph, new_node));
  GE_ASSERT_SUCCESS(UpdateSubgraphAxisAttr(new_node, node1, node2));
  GE_ASSERT_SUCCESS(UpdateAxisGroupInfo(node1, node2, new_node, graph_axis_map));
  GELOGD("dump merged node:%s(%s) asc subgraph info:", new_node->GetNamePtr(), new_node->GetType().c_str());
  BackendUtils::DumpAscGraph(new_node);
  // 异常时dump图的缓存融合后dump图流程
  GE_ASSERT_SUCCESS(CacheGraphAfterMerge(new_node, node1, node2, merged_graph));

  // 融合后做反推和补轴，下次融合时具备完整信息
  if (new_node->GetType() == kAscBackendType) {
    GE_ASSERT_SUCCESS(BackendUtils::ProcessAscgraphAfterMerge(new_node));
  }
  return new_node;
}

bool AscBackendFusionDecider::CanFuse(const NodePtr &node1, const NodePtr &node2) const {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  thread_local static std::set<std::string> has_been_dumped_nodes;  // 存放已经dump过的node，如果node已经dump过了，就不再重复dump这个node的图
  GELOGI("AscBackendGraphFuse:can fuse check before, node: %s(%s) and node: %s(%s).", node1->GetNamePtr(),
         node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
  // 黑名单的node不可融合
  if (BackendUtils::IsCanFuseBlackList(node1) || BackendUtils::IsCanFuseBlackList(node2)) {
    return false;
  }
  // 具备autofuse属性node才能融合
  if (!BackendUtils::IsBackendFuseNode(node1) || !BackendUtils::IsBackendFuseNode(node2)) {
    return false;
  }

  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  uint32_t max_fusion_node_input_size = config.max_input_nums_after_fuse;
  if (!BackendUtils::CanFuseByStrategy(node1, node2, max_fusion_node_input_size)) {
    return false;
  }

  if (BackendUtils::GetNodeFusedGraph(node1, graph1) != SUCCESS) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1(%s) can't get subgraph]",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kGetSubgraphFailed), node1->GetNamePtr());
    return false;
  }
  if (BackendUtils::GetNodeFusedGraph(node2, graph2) != SUCCESS) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node2(%s) can't get subgraph]",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
           ge::NotFuseReasonCode(ge::NotFuseReason::kGetSubgraphFailed), node2->GetNamePtr());
    return false;
  }
  if (BackendUtils::UpdateSubgraphOutputAttr(graph1, node1) != SUCCESS) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1(%s) can't update subgraph output attr]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kUpdateSubgraphOutputAttrFailed), node1->GetNamePtr());
    return false;
  }
  if (BackendUtils::UpdateSubgraphOutputAttr(graph2, node2) != SUCCESS) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node2(%s) can't update subgraph output attr]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kUpdateSubgraphOutputAttrFailed), node2->GetNamePtr());
    return false;
  }

  if (has_been_dumped_nodes.find(node1->GetName()) == has_been_dumped_nodes.end()) {
    GELOGD("dump node:%s(%s) asc subgraph info:", node1->GetNamePtr(), node1->GetType().c_str());
    BackendUtils::DumpAscGraph(node1);
    has_been_dumped_nodes.insert(node1->GetName());
  }

  if (has_been_dumped_nodes.find(node2->GetName()) == has_been_dumped_nodes.end()) {
    GELOGD("dump node:%s(%s) asc subgraph info:", node2->GetNamePtr(), node2->GetType().c_str());
    BackendUtils::DumpAscGraph(node2);
    has_been_dumped_nodes.insert(node2->GetName());
  }
  // 尝试做轴映射，如果无法映射融合失败
  AscGraphAxisMapping graph_axis_map;
  NodeFuseInfo node_fuse_info(false);
  if (node_fuse_info.UpdateNodeFuseInfo(node1, node2) != SUCCESS) {
    return false;
  }
  if (UnifySubgraphAxis(node1, node2, node_fuse_info, graph_axis_map, false) != SUCCESS) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 can't unify subgraph axis]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kUnifySubgraphAxisFailed));
    return false;
  }

  // concat水平融合优化
  if (!BackendUtils::CheckSameSchedAxis(node1, node2, graph_axis_map.GetNode1AxisMap(),
                                        graph_axis_map.GetNode2AxisMap(), node_fuse_info)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In concat fusion occasion, node1 and node2's "
        "schedule axis not equal]", node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), 
        node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kConcatNodeSchedAxisNotEqual));
    return false;
  }

  // 当融合后节点的最大输入个数超过阈值就不融合
  if (node_fuse_info.GetNode2InputMap().size() > max_fusion_node_input_size) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][fused node input data nums(%zu) exceed "
        "threshold(%u) after fuse]", node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
        node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kInputNumsExceedThreshold),
        node_fuse_info.GetNode2InputMap().size(), max_fusion_node_input_size);
    return false;
  }

  // 判断group merge是否成功
  optimize::autoschedule::AxisGroup axes_group1;
  if (BackendUtils::GetAscGraphAxisGroup(node1, axes_group1, graph_axis_map.GetNode1AxisMap()) != SUCCESS) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1(%s) can't get ascgraph axis group]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kGetAscgraphAxisGroupFailed), node1->GetNamePtr());
    return false;
  }
  optimize::autoschedule::AxisGroup axes_group2;
  if (BackendUtils::GetAscGraphAxisGroup(node2, axes_group2, graph_axis_map.GetNode2AxisMap()) != SUCCESS) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node2(%s) can't get ascgraph axis group]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kGetAscgraphAxisGroupFailed), node2->GetNamePtr());
    return false;
  }
  optimize::autoschedule::AxisGroup merged_axes_group;
  if (!BackendUtils::IsCanMergeAxisGroup(axes_group1, axes_group2, merged_axes_group)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 can't merge axis group]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kMergeAxisGroupFailed));
    return false;
  }
  GELOGI("AscBackendGraphFuse:can fuse check end, node: %s(%s) and node: %s(%s) can fuse.", node1->GetNamePtr(),
         node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
  return true;
}

FusionPriority AscBackendFusionDecider::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  const auto fuse_type = BackendUtils::GetAllFuseType(node1, node2);
  for (const auto fusion_strategy : FusionStrategyRegistry::Instance().Get(fuse_type)) {
    if (fusion_strategy != nullptr) {
      const auto priority = fusion_strategy->GetFusionPairPriority(node1, node2);
      if (priority != FusionPriority::DEFAULT) {
        return priority;
      }
    }
  }
  return FusionPriority::DEFAULT;
}

Status AscBackendFusionDecider::GetPossibleFuseNodePairFromSubgraph(
    std::set<std::pair<NodePtr, NodePtr>> &fuse_possibility_map, const ComputeGraphPtr &subgraph) {
  const auto attr = GetOrCreateAutoFuseAttrs(subgraph);
  GE_ASSERT_NOTNULL(attr);
  fuse_possibility_map.insert(GetInterAttrs(attr).possible_fusion_nodes.begin(),
                              GetInterAttrs(attr).possible_fusion_nodes.end());
  return SUCCESS;
}

Status AscBackendFusionDecider::UpdateNewNodeAndGENodeOutputMappingRelation(const NodePtr &new_node,
                                                                            const NodePtr &node1, const NodePtr &node2,
                                                                            const NodeFuseInfo &node_fuse_info) {
  std::vector<std::pair<std::string, int32_t>> new_output_names;
  // node1
  GE_ASSERT_SUCCESS(AscBackendFusionDecider::GetNodeOriginInfo(node1, new_node->GetAllOutDataAnchorsSize(),
                                                               node_fuse_info.GetNode1OutputMap(),
                                                               new_output_names, true));
  // node2
  GE_ASSERT_SUCCESS(AscBackendFusionDecider::GetNodeOriginInfo(node2, new_node->GetAllOutDataAnchorsSize(),
                                                               node_fuse_info.GetNode2OutputMap(),
                                                               new_output_names, true));
  GetInterAttrs(GetOrCreateAutoFuseAttrs(new_node->GetOpDesc())).origin_output_names_ = new_output_names;
  return SUCCESS;
}

Status AscBackendFusionDecider::UpdateNewNodeAndGENodeInputMappingRelation(const NodePtr &new_node,
                                                                           const NodePtr &node1, const NodePtr &node2,
                                                                           const NodeFuseInfo &node_fuse_info) {
  std::vector<std::pair<std::string, int32_t>> new_input_names;
  // node1
  GE_ASSERT_SUCCESS(AscBackendFusionDecider::GetNodeOriginInfo(node1, new_node->GetAllInDataAnchorsSize(),
                                                               node_fuse_info.GetNode1InputMap(),
                                                               new_input_names, false));
  // node2
  GE_ASSERT_SUCCESS(AscBackendFusionDecider::GetNodeOriginInfo(node2, new_node->GetAllInDataAnchorsSize(),
                                                               node_fuse_info.GetNode2InputMap(),
                                                               new_input_names, false));
  GetInterAttrs(GetOrCreateAutoFuseAttrs(new_node->GetOpDesc())).origin_input_names_ = new_input_names;
  return SUCCESS;
}

Status AscBackendFusionDecider::GetNodeOriginInfo(const NodePtr &node,
                                                  const uint32_t &new_node_in_or_out_data_anchor_size,
                                                  const vector<int32_t> &node_input_or_output_map,
                                                  std::vector<std::pair<std::string, int32_t>> &new_output_names,
                                                  bool is_output) {
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  auto node_origin_info = GetInterAttrs(attr).origin_input_names_;
  if (is_output) {
    node_origin_info = GetInterAttrs(attr).origin_output_names_;
  }
  for (uint32_t i = 0U; i < new_node_in_or_out_data_anchor_size; ++i) {
    if (i >= node_input_or_output_map.size()) {
      break;
    }
    const auto old_index = node_input_or_output_map.at(i);
    if (old_index < 0 || (static_cast<size_t>(old_index) >= node_origin_info.size())) {
      continue;
    }
    auto origin_output_name = node_origin_info.at(old_index);
    new_output_names.emplace_back(origin_output_name.first, origin_output_name.second);
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::SetReduceFusedElementwiseNodeNum(const NodePtr &new_node, const NodePtr &node1,
                                                                 const NodePtr &node2) {
  const auto attr = BackendUtils::GetNodeAutoFuseAttr(new_node);
  GE_ASSERT_NOTNULL(attr);
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  uint64_t supported_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));
  if (attr1->HasFuseType(loop::FuseType::kReduction) && (attr2->GetAllFuseType() == supported_type)) {
    const auto new_reduce_node_fused_elementwise_num =
        attr1->GetReduceFusedElementwiseNodeNum() + BackendUtils::GetComputeNodeNumInAscgraph(node2);
    attr->SetReduceFusedElementwiseNodeNum(new_reduce_node_fused_elementwise_num);
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::SetSplitNodeGlobalId(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2) {
  const auto attr = BackendUtils::GetNodeAutoFuseAttr(new_node);
  GE_ASSERT_NOTNULL(attr);
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  if (attr1->HasFuseType(loop::FuseType::kSplit)) {
    attr->SetSplitGlobalId(attr1->GetSplitGlobalId());
  }
  if (attr2->HasFuseType(loop::FuseType::kSplit)) {
    auto gid = attr2->GetSplitGlobalId();
    GE_ASSERT_TRUE(!attr1->HasFuseType(loop::FuseType::kSplit) || attr1->GetSplitGlobalId() == gid,
                   "fused nodes %s(%s) and %s(%s) have incompatible split global ids.",
                   node1->GetType().c_str(), node1->GetName().c_str(),
                   node2->GetType().c_str(), node2->GetName().c_str());
    attr->SetSplitGlobalId(gid);
  }
  return SUCCESS;
}

Status AscBackendFusionDecider::SetSplitNodeConcreteEdges(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2) {
  const auto attr = BackendUtils::GetNodeAutoFuseAttr(new_node);
  GE_ASSERT_NOTNULL(attr);
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  if ((attr1->GetFuseType() == loop::FuseType::kSplit) && (attr2->GetFuseType() == loop::FuseType::kSplit)) {
    const auto concrete_edges1 = attr1->GetConcreteEdges();
    for (const auto &edges : concrete_edges1) {
      for (const auto &in_data_anchor : edges.second) {
        attr->AddConcreteEdges(edges.first, in_data_anchor);
      }
    }
    const auto concrete_edges2 = attr2->GetConcreteEdges();
    for (const auto &edges : concrete_edges2) {
      for (const auto &in_data_anchor : edges.second) {
        attr->AddConcreteEdges(edges.first, in_data_anchor);
      }
    }
  }

  return SUCCESS;
}

REGISTER_FUSION_DECIDER(AscBackendFusionDecider, AutoFuseFwkType::kGe);
}  // namespace ge
