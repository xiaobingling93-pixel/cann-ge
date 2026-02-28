/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_graph_axis_mapping.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"
#include "utils/autofuse_utils.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "backend_utils.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"

namespace ge {
Status NodeFuseInfo::GetSubgraphSameInputIndex(const NodePtr &node1, const NodePtr &node2,
                                               std::vector<std::pair<int32_t, int32_t>> &same_input_map) const {
  const auto node1_in_data_size = node1->GetAllInDataAnchorsSize();
  const auto node2_in_data_size = node2->GetAllInDataAnchorsSize();

  for (auto node1_input = 0U; node1_input < node1_in_data_size; node1_input++) {
    for (auto node2_input = 0U; node2_input < node2_in_data_size; node2_input++) {
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
        same_input_map.emplace_back(node1_input, node2_input);
      }
    }
  }
  GELOGI_IF(open_log_, "before removing duplicates, node %s(%s) and node %s(%s), same input map: %s.",
            node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            AutofuseUtils::VectorPairToStr(same_input_map).c_str());
  /*
   *              data
   *            / /  \ \
   *           / /    \ \
   *          node1    node2
   * 两个节点水平融合且node2与node1的多个输入都来自于同一个节点，类似于same_input_map=[(0, 0), (0, 1), (1, 0), (1, 1)]，此时需要对
   * same_input_map进行去重，按照pair中的第2个元素（node2的输入index）进行去重，比如这个例子中需要将后两个(1, 0)和(1, 1)去掉。
   */
  std::unordered_set<int32_t> seen;
  same_input_map.erase(
      std::remove_if(same_input_map.begin(), same_input_map.end(),
                     [&seen](const auto &p) { return !seen.insert(p.second).second; }), same_input_map.end());
  GELOGI_IF(open_log_, "after removing duplicates, node %s(%s) and node %s(%s), same input map: %s.",
            node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            AutofuseUtils::VectorPairToStr(same_input_map).c_str());
  return SUCCESS;
}

Status NodeFuseInfo::GetSubgraphLinkIndex(const NodePtr &node1, const NodePtr &node2,
                                          std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map) {
  auto node1_out_data_size = node1->GetAllOutDataAnchorsSize();

  auto out_index = 0;
  for (auto node1_output = 0U; node1_output < node1_out_data_size; node1_output++) {
    const auto node1_out_anchor = node1->GetOutDataAnchor(static_cast<int32_t>(node1_output));
    GE_ASSERT_NOTNULL(node1_out_anchor);
    const auto &node1_peer_anchors = node1_out_anchor->GetPeerInDataAnchors();
    if (node1_peer_anchors.empty()) {
      out_index++;
      continue;
    }
    for (const auto &node1_peer_anchor : node1_peer_anchors) {
      const auto &node1_peer_node = node1_peer_anchor->GetOwnerNode();
      if (node1_peer_node == node2) {
        auto in_index = 0;
        for (const auto &node2_in_anchor : node2->GetAllInDataAnchors()) {
          if (node1_peer_anchor == node2_in_anchor) {
            subgraph_link_map.emplace_back(out_index, in_index);
            // 如果是多个输出，不能删除store节点
            if (node1_peer_anchors.size() > 1U) {
              GELOGI_IF(open_log_, "node1 %s(%s) mul reference link to node2 %s(%s), store will not delete.",
                        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
              is_single_reference_.emplace_back(out_index, false);
            }
          }
          in_index++;
        }
      }
      out_index++;
    }
  }
  GELOGI_IF(open_log_, "node1 %s(%s) link to node2 %s(%s), link map: %s.", node1->GetNamePtr(),
            node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            AutofuseUtils::VectorPairToStr(subgraph_link_map).c_str());
  return SUCCESS;
}

Status NodeFuseInfo::GetFuseNodeInput(const NodePtr &node1, const NodePtr &node2, std::vector<int32_t> &node1_input_map,
                                      std::vector<int32_t> &node2_input_map) const {
  std::vector<int32_t> node1_map(node1_in_data_size_, 0);
  std::vector<int32_t> node2_map(node2_in_data_size_, 0);
  for (auto node1_input = 0U; node1_input < node1_in_data_size_; node1_input++) {
    const auto node1_in_anchor = node1->GetInDataAnchor(static_cast<int32_t>(node1_input));
    GE_ASSERT_NOTNULL(node1_in_anchor);
    auto node1_peer_out_anchor = node1_in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(node1_peer_out_anchor);
    for (auto node2_input = 0U; node2_input < node2_in_data_size_; node2_input++) {
      // 如果node2输入以及标记为无效，无需再判断
      if (node2_map[node2_input] == -1) {
        continue;
      }
      auto node2_in_anchor = node2->GetInDataAnchor(static_cast<int32_t>(node2_input));
      GE_ASSERT_NOTNULL(node2_in_anchor);
      auto node2_peer_out_anchor = node2_in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(node2_peer_out_anchor);
      // 如果node1和node2的输入来自同一个节点的输出，fuse node只使用node1的，做多引用的合并
      if (node1_peer_out_anchor == node2_peer_out_anchor && CanDoHorizontalMapping()) {
        node2_map[node2_input] = -1;
      }
      // node1是node2的祖先，node2的输入无效
      if (node2_peer_out_anchor->GetOwnerNode() == node1) {
        node2_map[node2_input] = -1;
      }
    }
  }

  auto process_zero_inputs = [](const NodePtr &zero_input_node, const NodePtr &node, uint32_t in_data_size,
                                std::vector<int32_t> &input_map) -> Status {
    for (auto input = 0U; input < in_data_size; input++) {
      auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(input));
      GE_ASSERT_NOTNULL(in_anchor);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      if (peer_out_anchor->GetOwnerNode() == zero_input_node) {
        input_map[input] = -1;
      }
    }
    return SUCCESS;
  };

  // const node没有输入的场景
  if (node1_in_data_size_ == 0U) {
    GE_ASSERT_SUCCESS(process_zero_inputs(node1, node2, node2_in_data_size_, node2_map));
  }

  for (size_t i = 0U; i < node1_map.size(); i++) {
    if (node1_map[i] != -1) {
      node1_input_map.push_back(static_cast<int32_t>(i));
      node2_input_map.push_back(-1);
    }
  }
  for (size_t i = 0U; i < node2_map.size(); i++) {
    if (node2_map[i] != -1) {
      node2_input_map.push_back(static_cast<int32_t>(i));
    }
  }
  return SUCCESS;
}

Status NodeFuseInfo::GetFuseNodeOutput(const NodePtr &node1, const NodePtr &node2,
                                       std::vector<int32_t> &node1_output_map,
                                       std::vector<int32_t> &node2_output_map) const {
  std::vector<int32_t> node1_map(node1_out_data_size_, 0);
  std::vector<int32_t> node2_map(node2_out_data_size_, 0);
  GE_ASSERT_SUCCESS(GetNodeOutputMap(node1, node2, node1_map));
  GE_ASSERT_SUCCESS(GetNodeOutputMap(node2, node1, node2_map));
  for (size_t i = 0U; i < node1_map.size(); i++) {
    if (node1_map[i] == -1) {
      continue;
    }
    node1_output_map.push_back(static_cast<int32_t>(i));
    node2_output_map.push_back(-1);
  }
  for (size_t i = 0U; i < node2_map.size(); i++) {
    if (node2_map[i] == -1) {
      continue;
    }
    node2_output_map.push_back(static_cast<int32_t>(i));
  }
  return SUCCESS;
}

Status NodeFuseInfo::GetNodeOutputMap(const NodePtr &node1, const NodePtr &node2,
                                      std::vector<int32_t> &node_output_map) const {
  for (auto node1_output = 0U; node1_output < node_output_map.size(); node1_output++) {
    node_output_map[node1_output] = 0;
    const auto node1_out_anchor = node1->GetOutDataAnchor(static_cast<int32_t>(node1_output));
    GE_ASSERT_NOTNULL(node1_out_anchor);
    const auto &node1_peer_anchors = node1_out_anchor->GetPeerInDataAnchors();
    bool all_match = true;
    for (const auto &node1_peer_anchor : node1_peer_anchors) {
      const auto &node1_peer_node = node1_peer_anchor->GetOwnerNode();
      if (node1_peer_node != node2) {
        all_match = false;
        GELOGI("node1 %s(%s) output %zu peer in anchor %s is not one of in anchors of node2 %s(%s), "
               "map node1 output %zu to fused node outputs list.", node1->GetName().c_str(), node1->GetType().c_str(), node1_output,
               loop::BufferName(node1_peer_anchor).c_str(), node2->GetName().c_str(), node2->GetType().c_str(),
               node1_output);
        break;
      }
    }
    if (!node1_peer_anchors.empty() && all_match) {
      GELOGI("All of peer in anchors of out anchor %zu of node1 %s(%s) are in anchors of node2 %s(%s), "
             "do not map node1 output %zu to fused node outputs list.", node1_output, node1->GetName().c_str(),
             node1->GetType().c_str(), node2->GetName().c_str(), node2->GetType().c_str(), node1_output);
      node_output_map[node1_output] = -1;
    }
  }
  return SUCCESS;
}

// 如果anchor没有连边关系，占据一个位置
Status NodeFuseInfo::GetNodeOutputIndex(const NodePtr &node, uint32_t &node_out_node_size,
                                        const uint32_t node_out_data_size,
                                        std::vector<uint32_t> &node_output_index) const {
  node_out_node_size = 0U;
  for (auto node_output = 0U; node_output < node_out_data_size; node_output++) {
    const auto node_out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(node_output));
    GE_ASSERT_NOTNULL(node_out_anchor);
    const auto out_size = node_out_anchor->GetPeerInDataAnchors().size();
    if (out_size == 0U) {
      node_out_node_size++;
      node_output_index.push_back(node_output);
      continue;
    }
    for (size_t i = 0U; i < out_size; i++) {
      node_out_node_size++;
      node_output_index.push_back(node_output);
    }
  }
  return SUCCESS;
}

bool HasCubeType(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);
  if (attr1->HasFuseType(loop::FuseType::kCube) || attr2->HasFuseType(loop::FuseType::kCube)) {
    return true;
  }
  return false;
}

void NodeFuseInfo::RollbackNode2InputMap(int32_t data_input) {
  // node2_input_map_是升序的数组，融合的时候默认是删除node2子图上的节点，回退时只需要回退node2
  GELOGD_IF(open_log_, "origin node2 input map: %s.", AutofuseUtils::VectorToStr(node2_input_map_).c_str());
  auto insert_pos = std::lower_bound(node2_input_map_.begin(), node2_input_map_.end(), data_input);
  node2_input_map_.insert(insert_pos, data_input);
  GELOGD_IF(open_log_, "rollback node2 input map: %s.", AutofuseUtils::VectorToStr(node2_input_map_).c_str());
}

// 在can fuse前获取node间的连接信息，后期融合的时候直接获取
Status NodeFuseInfo::UpdateNodeFuseInfo(const NodePtr &node1, const NodePtr &node2) {
  Clear();
  GE_ASSERT_SUCCESS(GetSubgraphLinkIndex(node2, node1, node1_to_node2_link_map_));
  // 融合的两个节点如果垂直融合，一定是node1是前序，node2是后续
  GE_ASSERT_TRUE(node1_to_node2_link_map_.empty());
  Clear();
  GE_ASSERT_SUCCESS(GetSubgraphLinkIndex(node1, node2, node1_to_node2_link_map_));
  // 垂直融合场景，node1是前序节点
  node1_in_data_size_ = node1->GetAllInDataAnchorsSize();
  node2_in_data_size_ = node2->GetAllInDataAnchorsSize();
  node1_out_data_size_ = node1->GetAllOutDataAnchorsSize();
  node2_out_data_size_ = node2->GetAllOutDataAnchorsSize();
  GE_ASSERT_SUCCESS(GetNodeOutputIndex(node1, node1_out_node_size_, node1_out_data_size_, node1_output_index_));
  GE_ASSERT_SUCCESS(GetNodeOutputIndex(node2, node2_out_node_size_, node2_out_data_size_, node2_output_index_));
  GE_ASSERT_SUCCESS(GetSubgraphSameInputIndex(node1, node2, same_input_map_));
  has_slice_vertical_ = BackendUtils::SliceHasSameLoad(node1, node2, same_input_map_);
  can_do_horizontal_mapping_ = !HasCubeType(node1, node2); // matmul节点同时有水平融合和垂直融合时不做水平融合轴映射
  // 如果两个node毫无关联返回FAILED
  if (same_input_map_.empty() && node1_to_node2_link_map_.empty()) {
    GELOGD_IF(open_log_, "node1 %s(%s) and node2 %s(%s) have no relation.", node1->GetNamePtr(),
              node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return FAILED;
  }
  GE_ASSERT_SUCCESS(GetFuseNodeInput(node1, node2, node1_input_map_, node2_input_map_));
  GE_ASSERT_SUCCESS(GetFuseNodeOutput(node1, node2, node1_output_map_, node2_output_map_));
  // 获取两个节点的前序节点信息
  GE_ASSERT_SUCCESS(GetNodePreInfo(node1, node1_in_data_size_, node1_pre_nodes_));
  GE_ASSERT_SUCCESS(GetNodePreInfo(node2, node2_in_data_size_, node2_pre_nodes_));
  GELOGI_IF(open_log_,
            "get fuse info: node1 %s(%s) input size %u, output size %u, input map %s, output map %s, output index %s; "
            "node2 %s(%s) input size %u, output size %u, input map %s, output map %s, output index %s.",
            node1->GetNamePtr(), node1->GetType().c_str(), node1_in_data_size_, node1_out_node_size_,
            AutofuseUtils::VectorToStr(node1_input_map_).c_str(), AutofuseUtils::VectorToStr(node1_output_map_).c_str(),
            AutofuseUtils::VectorToStr(node1_output_index_).c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            node2_in_data_size_, node2_out_node_size_, AutofuseUtils::VectorToStr(node2_input_map_).c_str(),
            AutofuseUtils::VectorToStr(node2_output_map_).c_str(),
            AutofuseUtils::VectorToStr(node2_output_index_).c_str());
  return SUCCESS;
}

Status NodeFuseInfo::GetNodePreInfo(const NodePtr &node, const uint32_t &in_nums,
                                    std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes) const {
  for (auto input_idx = 0U; input_idx < in_nums; input_idx++) {
    const auto &input_anchor = node->GetInDataAnchor(static_cast<int32_t>(input_idx));
    GE_ASSERT_NOTNULL(input_anchor);
    const auto output_anchor = input_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(output_anchor);
    auto pre_node = output_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(pre_node);
    pre_nodes.emplace_back(pre_node, output_anchor->GetIdx());
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::GetCurAscNodeAttrs(const NodePtr &parent_node, const int32_t index,
                                               std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats) const {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(parent_node, graph));
  const auto subgraph_input_nodes = graph->GetInputNodes();
  GE_ASSERT_TRUE(subgraph_input_nodes.size() > static_cast<size_t>(index));
  const auto data_node = subgraph_input_nodes.at(index);
  GE_ASSERT_NOTNULL(data_node);
  GELOGD_IF(open_log_, "get cur node %s(%s) index=%d.", data_node->GetNamePtr(), data_node->GetType().c_str(), index);
  auto node_op_desc = data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_output_desc = node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(node_output_desc);
  auto node_tensor_attr = node_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(node_tensor_attr);
  repeats = node_tensor_attr->repeats;
  axis = node_tensor_attr->axis;
  GE_ASSERT_TRUE(axis.size() == repeats.size());
  return SUCCESS;
}

Status AscGraphAxisMapping::GetPreNodeAttrs(const NodePtr &node, const int32_t index, std::vector<ge::Expression> &dims,
                                            std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats) const {
  NodePtr peer_node;
  InDataAnchorPtr in_anchor;
  GE_ASSERT_SUCCESS(BackendUtils::GetPreNodeAndAnchor(node, index, peer_node, in_anchor));
  const auto peer_node_desc = peer_node->GetOpDesc();
  GE_ASSERT_NOTNULL(peer_node_desc);
  const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  const auto output_tensor_desc = peer_node_desc->MutableOutputDesc(peer_out_anchor->GetIdx());
  GE_ASSERT_NOTNULL(output_tensor_desc);
  const auto attr_group = output_tensor_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  dims = attr_group->symbolic_tensor.GetOriginSymbolShape().GetDims();

  if (peer_node->GetType() == kAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_node, graph));
    // 获取前面ascbackendnode信息的时候需要创建子图属性，要不然output索引无效
    GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph, peer_node));
    GE_ASSERT_SUCCESS(BackendUtils::GetPreAscNodeAttrs(peer_node, in_anchor, axis, repeats));
  } else if (peer_node->GetType() == kFusedAscBackendType) {
    NodePtr asc_node;
    InDataAnchorPtr netoutput_in_anchor;
    GE_ASSERT_SUCCESS(BackendUtils::GetPreAscBackendNodeAndAnchor(node, peer_node, in_anchor, asc_node, netoutput_in_anchor));
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(asc_node, graph));
    GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph, asc_node));
    GE_ASSERT_SUCCESS(BackendUtils::GetPreAscNodeAttrs(asc_node, netoutput_in_anchor, axis, repeats));
  } else {
    // nothing, 共同输入是普通ge node，没有axis和repeats属性
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::GetPreNodeAscGraphAttrs(const NodePtr &node, const int32_t index,
                                                    std::vector<int64_t> &axis, std::vector<ge::Expression> &size,
                                                    NodePtr &ascback_node) const {
  NodePtr peer_node;
  InDataAnchorPtr in_anchor;
  GE_ASSERT_SUCCESS(BackendUtils::GetPreNodeAndAnchor(node, index, peer_node, in_anchor));

  if (peer_node->GetType() == kAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_node, graph));
    ascback_node = peer_node;
    auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(graph_attr);
    for (auto &axis_info : graph_attr->axis) {
      axis.push_back(axis_info->id);
      size.push_back(axis_info->size);
    }
  } else if (peer_node->GetType() == kFusedAscBackendType) {
    NodePtr asc_node;
    InDataAnchorPtr netoutput_in_anchor;
    GE_ASSERT_SUCCESS(BackendUtils::GetPreAscBackendNodeAndAnchor(node, peer_node, in_anchor, asc_node, netoutput_in_anchor));
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(asc_node, graph));
    ascback_node = asc_node;
    auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(graph_attr);
    for (auto &axis_info : graph_attr->axis) {
      axis.push_back(axis_info->id);
      size.push_back(axis_info->size);
    }
  } else {
    // nothing
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::GetCurNodeAttrs(const NodePtr &node, const int32_t index, std::vector<int64_t> &axis,
                                            std::vector<ge::Expression> &repeats) const {
  if (node->GetType() == kAscBackendType) {
    GE_ASSERT_SUCCESS(GetCurAscNodeAttrs(node, index, axis, repeats));
  } else if (node->GetType() == kFusedAscBackendType) {
    int32_t in_anchor_idx;
    const auto asc_node = BackendUtils::GetFusedAscBackendInputNode(node, index, in_anchor_idx);
    GE_ASSERT_NOTNULL(asc_node);
    GE_ASSERT_SUCCESS(GetCurAscNodeAttrs(asc_node, in_anchor_idx, axis, repeats));
  } else {
    // nothing, 共同输入是普通ge node，没有axis和repeats属性
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::GetCurNodeAscGraphAttrs(const NodePtr &node, const int32_t index,
                                                    std::vector<int64_t> &axis, std::vector<ge::Expression> &size,
                                                    NodePtr &ascback_node) {
  if (node->GetType() == kAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
    ascback_node = node;
    auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(graph_attr);
    for (auto &axis_info : graph_attr->axis) {
      axis.push_back(axis_info->id);
      size.push_back(axis_info->size);
    }
  } else if (node->GetType() == kFusedAscBackendType) {
    int32_t in_anchor_idx;
    const auto asc_node = BackendUtils::GetFusedAscBackendInputNode(node, index, in_anchor_idx);
    GE_ASSERT_NOTNULL(asc_node);
    GE_ASSERT_SUCCESS(GetCurNodeAscGraphAttrs(asc_node, in_anchor_idx, axis, size, ascback_node));
  } else {
    // nothing, 共同输入是普通ge node，没有axis和repeats属性
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::FindAxisIndex(std::vector<ge::Expression> &node_repeats,
                                          std::vector<ge::Expression> &base_repeats,
                                          std::vector<uint32_t> &axis_index) const {
  std::unordered_set<size_t> used_indices;
  size_t start_index = 0;

  // 1. 优先尝试找到连续匹配（从右到左）
  for (int i = static_cast<int>(node_repeats.size()) - 1; i >= 0; --i) {
    bool found = false;
    for (int j = static_cast<int>(base_repeats.size()) - 1; j >= static_cast<int>(start_index); --j) {
      if ((base_repeats[j] == node_repeats[i]) && (used_indices.find(j) == used_indices.end())) {
        axis_index.insert(axis_index.begin(), j);
        used_indices.insert(j);
        found = true;
        start_index = j;
        break;
      }
    }
    if (!found) {
      break;
    }
  }

  if (axis_index.size() == node_repeats.size()) {
    GELOGD_IF(open_log_, "find axis index info: repeats(%s), base repeats(%s), axis_index(%s).",
              AutofuseUtils::VectorToStr(node_repeats).c_str(), AutofuseUtils::VectorToStr(base_repeats).c_str(),
              AutofuseUtils::VectorToStr(axis_index).c_str());
    return SUCCESS;
  }

  // 2. 如果连续匹配失败，回退到随机匹配（从右到左）
  axis_index.clear();
  used_indices.clear();

  for (int i = static_cast<int>(node_repeats.size()) - 1; i >= 0; --i) {
    bool found = false;
    for (int j = static_cast<int>(base_repeats.size()) - 1; j >= 0; --j) {
      if ((base_repeats[j] == node_repeats[i]) && (used_indices.find(j) == used_indices.end())) {
        axis_index.insert(axis_index.begin(), j);
        used_indices.insert(j);
        found = true;
        break;
      }
    }
    if (!found) {
      GELOGD_IF(open_log_, "some axis repeat(%s) don't find from base repeats(%s).",
                AutofuseUtils::VectorToStr(node_repeats).c_str(), AutofuseUtils::VectorToStr(base_repeats).c_str());
      return FAILED;
    }
  }
  GELOGD("find axis index info: repeats(%s), base repeats(%s), axis_index(%s).",
         AutofuseUtils::VectorToStr(node_repeats).c_str(), AutofuseUtils::VectorToStr(base_repeats).c_str(),
         AutofuseUtils::VectorToStr(axis_index).c_str());
  return SUCCESS;
}

bool AscGraphAxisMapping::CanAxisMap(std::vector<int64_t> &node1_axis, std::vector<ge::Expression> &node1_repeats,
                                     std::vector<int64_t> &node2_axis, std::vector<ge::Expression> &node2_repeats,
                                     AxisPairSet &node1_map, AxisPairSet &node2_map, AxisPairSet &temp_node1_map,
                                     AxisPairSet &temp_node2_map) const {
  temp_node1_map.clear();
  temp_node2_map.clear();
  if (node1_repeats.size() >= node2_repeats.size()) {
    std::vector<uint32_t> axis_index;
    if (FindAxisIndex(node2_repeats, node1_repeats, axis_index) != SUCCESS) {
      return false;
    }
    // 如果已经做了轴对齐，替换轴
    if (!node1_map.empty()) {
      if (BackendUtils::ConvertAxis(node1_map, node1_axis) != SUCCESS) {
        return false;
      }
    }
    for (auto i = 0U; i < axis_index.size(); i++) {
      GE_ASSERT_TRUE(static_cast<size_t>(axis_index[i]) < node1_axis.size());
      temp_node2_map.insert(std::pair<int64_t, int64_t>(node2_axis[i], node1_axis[axis_index[i]]));
    }
    for (long &node1_axi : node1_axis) {
      temp_node1_map.insert(std::pair<int64_t, int64_t>(node1_axi, node1_axi));
    }
  } else {
    std::vector<uint32_t> axis_index;
    if (FindAxisIndex(node1_repeats, node2_repeats, axis_index) != SUCCESS) {
      return false;
    }
    // 如果已经做了轴对齐，替换轴
    if (!node2_map.empty()) {
      if (BackendUtils::ConvertAxis(node2_map, node2_axis) != SUCCESS) {
        return false;
      }
    }
    for (auto i = 0U; i < axis_index.size(); i++) {
      GE_ASSERT_TRUE(static_cast<size_t>(axis_index[i]) < node2_axis.size());
      temp_node1_map.insert(std::pair<int64_t, int64_t>(node1_axis[i], node2_axis[axis_index[i]]));
    }
    for (long &node2_axi : node2_axis) {
      temp_node2_map.insert(std::pair<int64_t, int64_t>(node2_axi, node2_axi));
    }
  }
  GELOGD_IF(open_log_,
            "find axis map info: left axis(%s), repeats(%s), right axis(%s), repeats(%s), axis map1(%s), axis map2(%s).",
            AutofuseUtils::VectorToStr(node1_axis).c_str(), AutofuseUtils::VectorToStr(node1_repeats).c_str(),
            AutofuseUtils::VectorToStr(node2_axis).c_str(), AutofuseUtils::VectorToStr(node2_repeats).c_str(),
            AutofuseUtils::VectorPairToStr(temp_node1_map).c_str(),
            AutofuseUtils::VectorPairToStr(temp_node2_map).c_str());
  return true;
}

bool AscGraphAxisMapping::IsSameMapAxis(AxisPairSet &map1, AxisPairSet &map2) const {
  GELOGD_IF(open_log_, "check axis map(%s) and axis map(%s).", AutofuseUtils::VectorPairToStr(map1).c_str(),
            AutofuseUtils::VectorPairToStr(map2).c_str());
  std::set<int64_t> second_values_map1;
  for (const auto &pair : map1) {
    second_values_map1.insert(pair.second);
  }

  std::set<int64_t> second_values_map2;
  for (const auto &pair : map2) {
    second_values_map2.insert(pair.second);
  }

  // 合并两个 set
  std::set<int64_t> merged_set;
  merged_set.insert(second_values_map1.begin(), second_values_map1.end());
  merged_set.insert(second_values_map2.begin(), second_values_map2.end());

  // 判断合并后的 set 大小是否等于原两个 set 的最大值
  size_t max_size = std::max(second_values_map1.size(), second_values_map2.size());
  if (merged_set.size() == max_size) {
    return true;
  }
  return false;
}

Status AscGraphAxisMapping::FlashContinueAxisId() {
  AxisPairSet *max_map;
  AxisPairSet *min_map;
  GELOGD_IF(open_log_, "flash continue axis before, node1 map(%s) and node2 map(%s).",
            AutofuseUtils::VectorPairToStr(node1_map_).c_str(), AutofuseUtils::VectorPairToStr(node2_map_).c_str());
  if (node1_map_.size() >= node2_map_.size()) {
    max_map = &node1_map_;
    min_map = &node2_map_;
  } else {
    max_map = &node2_map_;
    min_map = &node1_map_;
  }
  std::unordered_map<int64_t, int64_t> value_map;
  AxisPairSet temp_map;
  int64_t i = 0;
  for (auto &pair : *max_map) {
    value_map[pair.second] = i;
    temp_map.insert(std::pair<int64_t, int64_t>(pair.first, i));
    i++;
  }
  *max_map = temp_map;
  GE_ASSERT_TRUE(value_map.size() == (*max_map).size());

  temp_map.clear();
  for (auto &pair : *min_map) {
    auto it = value_map.find(pair.second);
    GE_ASSERT_TRUE(it != value_map.end());
    temp_map.insert(std::pair<int64_t, int64_t>(pair.first, it->second));
  }
  *min_map = temp_map;
  GELOGD_IF(open_log_, "flash continue axis after, node1 map(%s) and node2 map(%s).",
            AutofuseUtils::VectorPairToStr(node1_map_).c_str(), AutofuseUtils::VectorPairToStr(node2_map_).c_str());
  return SUCCESS;
}

Status AscGraphAxisMapping::CheckSubGraphtVerticalAxisMapping(const NodePtr &node, AscNodeAxisInfo &pre_axis_info,
                                                              AscNodeAxisInfo &cur_axis_info, AxisPairSet &node1_map,
                                                              AxisPairSet &node2_map) {
  AxisPairSet temp_node1_map;
  AxisPairSet temp_node2_map;

  if (CanAxisMap(pre_axis_info.node_axis, pre_axis_info.node_repeats, cur_axis_info.node_axis,
                   cur_axis_info.node_repeats, node1_map, node2_map, temp_node1_map, temp_node2_map)) {
    // 1.graph1 store和 graph2 data轴是否能映射
    GELOGD_IF(open_log_, "node %s(%s) pre store and cur data axis can axis map.", node->GetNamePtr(),
              node->GetType().c_str());
    if (pre_node_is_reduction_ &&
      !CanAxisMap(pre_axis_info.node_axis, pre_axis_info.node_repeats, cur_axis_info.graph_axis,
                  cur_axis_info.graph_size, node1_map_, node2_map_, temp_node1_map, temp_node2_map)) {
        GELOGI_IF(open_log_, "pre node is reduction, graph sched axis can't map, can't fuse.");
        return FAILED;
    }
  } else {
    // 2. graph1 store节点和graph2轴能直接映射
    if (CanAxisMap(pre_axis_info.node_axis, pre_axis_info.node_repeats, cur_axis_info.graph_axis,
                 cur_axis_info.graph_size, node1_map, node2_map, temp_node1_map, temp_node2_map)) {
      GELOGD_IF(open_log_, "node %s(%s) pre store and ascgraph sched axis can axis map.", node->GetNamePtr(),
                node->GetType().c_str());
    } else {
      // 3.graph1 和 graph2调度轴是否能映射
      if (!CanAxisMap(pre_axis_info.graph_axis, pre_axis_info.graph_size, cur_axis_info.graph_axis,
                      cur_axis_info.graph_size, node1_map, node2_map, temp_node1_map, temp_node2_map)) {
        GELOGI_IF(open_log_,
                  "store and graph can't map, store and data can map, graph sched axis can't map, can't fuse.");
        return FAILED;
      } else {
        GELOGD_IF(open_log_, "node %s(%s) pre subgraph sched axis and cur subgraph sched axis can axis map.",
                  node->GetNamePtr(), node->GetType().c_str());
        if ((pre_axis_info.node_axis == pre_axis_info.graph_axis) &&
            (cur_axis_info.node_axis != cur_axis_info.graph_axis)) {
          GELOGI(
              "store and graph can't map, store and data can map, graph sched axis can map, but cur graph axis not "
              "same cur graph data axis, can't fuse.");
          return FAILED;
        }
      }
    }
  }

  return CheckAndFillAxisMap(node1_map, node2_map, temp_node1_map, temp_node2_map);
}

Status AscGraphAxisMapping::CheckAndFillAxisMap(AxisPairSet &node1_map, AxisPairSet &node2_map,
                                                AxisPairSet &temp_node1_map, AxisPairSet &temp_node2_map) const {
  if (node1_map.empty()) {
    node1_map = temp_node1_map;
    node2_map = temp_node2_map;
  } else {
    if (!IsSameMapAxis(node1_map, temp_node1_map)) {
      GELOGD("axis map(%s) and axis map(%s) mismatch.", AutofuseUtils::VectorPairToStr(node1_map).c_str(),
             AutofuseUtils::VectorPairToStr(temp_node1_map).c_str());
      return FAILED;
    }
    if (!IsSameMapAxis(node2_map, temp_node2_map)) {
      GELOGD("axis map(%s) and axis map(%s) mismatch.", AutofuseUtils::VectorPairToStr(node2_map).c_str(),
             AutofuseUtils::VectorPairToStr(temp_node2_map).c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::GetVerticalAxisMapInfo(const NodePtr &node, const int32_t index, AxisPairSet &node1_map,
                                                   AxisPairSet &node2_map, NodePtr &asc_node1, NodePtr &asc_node2) {
  AscNodeAxisInfo pre_node_axis_info;
  std::vector<ge::Expression> dims;
  GE_ASSERT_SUCCESS(GetPreNodeAttrs(node, index, dims, pre_node_axis_info.node_axis, pre_node_axis_info.node_repeats));
  GELOGD_IF(open_log_, "get node %s(%s) pre info: index=%d, tensor dim=%s, repeats=%s, axis=%s.", node->GetNamePtr(),
            node->GetType().c_str(), index, AutofuseUtils::VectorToStr(dims).c_str(),
            AutofuseUtils::VectorToStr(pre_node_axis_info.node_repeats).c_str(),
            AutofuseUtils::VectorToStr(pre_node_axis_info.node_axis).c_str());
  GE_ASSERT_TRUE(!pre_node_axis_info.node_axis.empty());

  GE_ASSERT_SUCCESS(
      GetPreNodeAscGraphAttrs(node, index, pre_node_axis_info.graph_axis, pre_node_axis_info.graph_size, asc_node1));
  GELOGD_IF(open_log_, "get node %s(%s) pre graph info: index=%d, size=%s, axis=%s.", node->GetNamePtr(),
            node->GetType().c_str(), index, AutofuseUtils::VectorToStr(pre_node_axis_info.graph_size).c_str(),
            AutofuseUtils::VectorToStr(pre_node_axis_info.graph_axis).c_str());

  AscNodeAxisInfo cur_node_axis_info;
  GE_ASSERT_SUCCESS(GetCurNodeAttrs(node, index, cur_node_axis_info.node_axis, cur_node_axis_info.node_repeats));
  GELOGD_IF(open_log_, "get node %s(%s) cur info: index=%d, repeats=%s, axis=%s.", node->GetNamePtr(),
            node->GetType().c_str(), index, AutofuseUtils::VectorToStr(cur_node_axis_info.node_repeats).c_str(),
            AutofuseUtils::VectorToStr(cur_node_axis_info.node_axis).c_str());

  GE_ASSERT_SUCCESS(
      GetCurNodeAscGraphAttrs(node, index, cur_node_axis_info.graph_axis, cur_node_axis_info.graph_size, asc_node2));
  GELOGD_IF(open_log_, "get node %s(%s) cur graph info: index=%d, size=%s, axis=%s.", node->GetNamePtr(),
            node->GetType().c_str(), index, AutofuseUtils::VectorToStr(cur_node_axis_info.graph_size).c_str(),
            AutofuseUtils::VectorToStr(cur_node_axis_info.graph_axis).c_str());

  if (CheckSubGraphtVerticalAxisMapping(node, pre_node_axis_info, cur_node_axis_info, node1_map, node2_map) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::FlushAscSubGraphAxisInfo(const ComputeGraphPtr &graph, const AxisPairSet &node_map,
                                                     bool need_flash) const {
  for (auto &asc_node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(asc_node);
    auto asc_node_op_desc = asc_node->GetOpDesc();
    GE_ASSERT_NOTNULL(asc_node_op_desc);
    AscNodeAttr *asc_node_attr = asc_node_op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(asc_node_attr);
    if (BackendUtils::ConvertAxis(node_map, asc_node_attr->sched.axis, need_flash) != SUCCESS) {
      return FAILED;
    }

    for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_desc_tensor_attr);
      if (BackendUtils::ConvertAxis(node_map, output_desc_tensor_attr->axis, need_flash) != SUCCESS) {
        return FAILED;
      }
    }
  }
  auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  for (auto &axis_info : graph_attr->axis) {
    if (BackendUtils::ConvertAxis(node_map, axis_info->id, need_flash) != SUCCESS) {
      return FAILED;
    }
    GELOGD_IF(open_log_, "  \nflash graph axis info: axis name(%s), axis id(%ld), axis size(%s), graph name(%s).",
              axis_info->name.c_str(), axis_info->id, std::string(axis_info->size.Str().get()).c_str(),
              graph->GetName().c_str());
  }

  return SUCCESS;
}

Status AscGraphAxisMapping::FlushSubGraphAxisInfo(const NodePtr &node, const AxisPairSet &node_map, bool need_flash) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  if (node->GetType() == kAscBackendType) {
    if (FlushAscSubGraphAxisInfo(graph, node_map, need_flash) != SUCCESS) {
      return FAILED;
    }
  } else if (node->GetType() == kFusedAscBackendType) {
    for (auto &sub_node : graph->GetAllNodes()) {
      if (sub_node->GetType() == kAscBackendType) {
        if (FlushSubGraphAxisInfo(sub_node, node_map, need_flash) != SUCCESS) {
          return FAILED;
        }
      }
    }
  } else {
    GELOGE(FAILED, "node %s(%s) not autofuse node.", node->GetNamePtr(), node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

bool AscGraphAxisMapping::CanLoopMerge(const NodePtr &node1, const NodePtr &node2) {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));

  auto graph_attr1 = graph1->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr1);
  auto graph_attr2 = graph2->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr2);
  GELOGD_IF(open_log_, "loop merge before, node %s(%s) axis map info=%s and node %s(%s) axis map info=%s.",
            node1->GetNamePtr(), node1->GetType().c_str(), AutofuseUtils::VectorPairToStr(node1_map_).c_str(),
            node2->GetNamePtr(), node2->GetType().c_str(), AutofuseUtils::VectorPairToStr(node2_map_).c_str());
  auto process_axis = [](AxisPairSet &node_map, const std::vector<AxisPtr> &axis_info_list) -> std::vector<int64_t> {
    std::vector<int64_t> axes;
    for (auto &axis_info : axis_info_list) {
      int64_t axis_id = axis_info->id;
      if (BackendUtils::ConvertAxis(node_map, axis_id, true) != SUCCESS) {
        return {};
      }
      axes.push_back(axis_id);
    }
    return axes;
  };
  std::vector<int64_t> axis1 = process_axis(node1_map_, graph_attr1->axis);
  std::vector<int64_t> axis2 = process_axis(node2_map_, graph_attr2->axis);

  GELOGD_IF(open_log_, "after axis process node %s(%s) axis(%s), node %s(%s) axis(%s).", node1->GetNamePtr(),
            node1->GetType().c_str(), AutofuseUtils::VectorToStr(axis1).c_str(), node2->GetNamePtr(),
            node2->GetType().c_str(), AutofuseUtils::VectorToStr(axis2).c_str());

  if (axis1.empty() || axis2.empty()) {
    GELOGI("sched axis convert failed, can't merge.");
    return false;
  }

  if (axis1 != axis2) {
    // 判断轴是否是顺序子集关系，子集关系认为也是可以循环合并的，后期schedue adapter补轴实现
    if (!BackendUtils::CheckAxisSubsetRelation(axis1, axis2)) {
      GELOGI_IF(open_log_, "sched axis different and not subset relation, can't merge.");
      return false;
    }
  }

  GELOGI_IF(open_log_, "node %s(%s) and node %s(%s) can cyclic merge.", node1->GetNamePtr(), node1->GetType().c_str(),
            node2->GetNamePtr(), node2->GetType().c_str());
  return true;
}

Status AscGraphAxisMapping::CheckSubGraphHorizontalAxisMapping(const NodePtr &node1, const NodePtr &node2,
                                                               AscNodeAxisInfo &node1_cur_info,
                                                               AscNodeAxisInfo &node2_cur_info) {
  AxisPairSet temp_node1_map;
  AxisPairSet temp_node2_map;
  // 1.graph1 data节点和graph2 data节点轴是否能直接映射
  if (CanAxisMap(node1_cur_info.node_axis, node1_cur_info.node_repeats, node2_cur_info.node_axis,
                 node2_cur_info.node_repeats, node1_map_, node2_map_, temp_node1_map, temp_node2_map)) {
    GELOGD_IF(open_log_, "node %s(%s) data node and node %s(%s) data node can axis map.", node1->GetNamePtr(),
              node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    // 2.graph1 和 graph2调度轴是否能映射
    if (!CanAxisMap(node1_cur_info.graph_axis, node1_cur_info.graph_size, node2_cur_info.graph_axis,
                    node2_cur_info.graph_size, node1_map_, node2_map_, temp_node1_map, temp_node2_map)) {
      GELOGI("data node can map, but graph sched axis can't, can't fuse.");
      return FAILED;
    }
    GELOGD_IF(open_log_, "node %s(%s) ascgraph sched axis and node %s(%s) ascgraph sched axis can axis map.",
              node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
  } else {
    if (CanAxisMap(node1_cur_info.graph_axis, node1_cur_info.graph_size, node2_cur_info.graph_axis,
                   node2_cur_info.graph_size, node1_map_, node2_map_, temp_node1_map, temp_node2_map)) {
      GELOGD_IF(open_log_, "node %s(%s) ascgraph sched axis and node %s(%s) ascgraph sched axis can axis map.",
                node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    } else {
      GELOGI("data node can't map, graph sched axis can't map, can't fuse.");
      return FAILED;
    }
  }
  return CheckAndFillAxisMap(node1_map_, node2_map_, temp_node1_map, temp_node2_map);
}

Status AscGraphAxisMapping::ProcessSubGraphHorizontalMapInfo(const NodePtr &node1, const NodePtr &node2,
                                                             const NodeFuseInfo &fuse_info) {
  for (const auto &same_input : fuse_info.GetSameInputMap()) {
    AscNodeAxisInfo cur_node1_info;
    GE_ASSERT_SUCCESS(GetCurNodeAttrs(node1, same_input.first, cur_node1_info.node_axis, cur_node1_info.node_repeats));
    GELOGD_IF(open_log_, "get node %s(%s) cur info: index=%d, repeats=%s, axis=%s.", node1->GetNamePtr(),
              node1->GetType().c_str(), same_input.first,
              AutofuseUtils::VectorToStr(cur_node1_info.node_repeats).c_str(),
              AutofuseUtils::VectorToStr(cur_node1_info.node_axis).c_str());

    NodePtr asc_node1;
    GE_ASSERT_SUCCESS(GetCurNodeAscGraphAttrs(node1, same_input.first, cur_node1_info.graph_axis,
                                              cur_node1_info.graph_size, asc_node1));
    GELOGD_IF(open_log_, "get node %s(%s) cur graph info: index=%d, size=%s, axis=%s.", node1->GetNamePtr(),
              node1->GetType().c_str(), same_input.first, AutofuseUtils::VectorToStr(cur_node1_info.graph_size).c_str(),
              AutofuseUtils::VectorToStr(cur_node1_info.graph_axis).c_str());

    AscNodeAxisInfo cur_node2_info;
    GE_ASSERT_SUCCESS(GetCurNodeAttrs(node2, same_input.second, cur_node2_info.node_axis, cur_node2_info.node_repeats));
    GELOGD_IF(open_log_, "get node %s(%s) cur info: index=%d, repeats=%s, axis=%s.", node2->GetNamePtr(),
              node2->GetType().c_str(), same_input.second,
              AutofuseUtils::VectorToStr(cur_node2_info.node_repeats).c_str(),
              AutofuseUtils::VectorToStr(cur_node2_info.node_axis).c_str());

    NodePtr asc_node2;
    GE_ASSERT_SUCCESS(GetCurNodeAscGraphAttrs(node2, same_input.second, cur_node2_info.graph_axis,
                                              cur_node2_info.graph_size, asc_node2));
    GELOGD_IF(open_log_, "get node %s(%s) cur graph info: index=%d, size=%s, axis=%s.", node2->GetNamePtr(),
              node2->GetType().c_str(), same_input.second,
              AutofuseUtils::VectorToStr(cur_node2_info.graph_size).c_str(),
              AutofuseUtils::VectorToStr(cur_node2_info.graph_axis).c_str());

    if (CheckSubGraphHorizontalAxisMapping(node1, node2, cur_node1_info, cur_node2_info) != SUCCESS) {
      return FAILED;
    }

    // Todo,  transpose支持后需要删除
    std::vector<ViewOpAttrInfo> attr_infos1, attr_infos2;
    if (!BackendUtils::CurNodeInputIsSimplestLoad(node1, same_input.first, attr_infos1) &&
        !BackendUtils::CurNodeInputIsSimplestLoad(node2, same_input.second, attr_infos2)) {
      if (!BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2)) {
        GELOGD_IF(open_log_, "node1 and node2 input all view op, broadcast infos are different, can fuse false.");
        return FAILED;
      }
      GELOGD_IF(open_log_, "node1 and node2 input all view op, broadcast infos are same.");
    }
    GELOGD_IF(open_log_, "node %s(%s) axis map info=%s and node %s(%s) axis map info=%s.", node1->GetNamePtr(),
              node1->GetType().c_str(), AutofuseUtils::VectorPairToStr(node1_map_).c_str(), node2->GetNamePtr(),
              node2->GetType().c_str(), AutofuseUtils::VectorPairToStr(node2_map_).c_str());
    if (node1_map_.empty() || node2_map_.empty()) {
      GELOGD("axis map info empty, can fuse false.");
      return FAILED;
    }
    // 轴映射完成后看是否能循环合并
    if (!CanLoopMerge(asc_node1, asc_node2)) {
      GELOGI("graph sched axis can't loop merge, can fuse false.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::ProcessSubGraphVerticalMapInfo(const NodePtr &node1, const NodePtr &node2,
                                                           const NodeFuseInfo &fuse_info, AxisPairSet &node1_map,
                                                           AxisPairSet &node2_map) {
  auto is_reduction = false;
  for (const auto &subgraph_link : fuse_info.GetNode1ToNode2LinkMap()) {
    auto autofuse_attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
    GE_ASSERT_NOTNULL(autofuse_attr1);
    is_reduction = GetInterAttrs(autofuse_attr1).IsReduction();
    pre_node_is_reduction_ = is_reduction;
    NodePtr asc_node1, asc_node2;
    if (GetVerticalAxisMapInfo(node2, subgraph_link.second, node1_map, node2_map, asc_node1, asc_node2) != SUCCESS) {
      return FAILED;
    }
    GELOGD_IF(open_log_, "node %s(%s) axis map info=%s and node %s(%s) axis map info=%s.", node1->GetNamePtr(),
              node1->GetType().c_str(), AutofuseUtils::VectorPairToStr(node1_map).c_str(), node2->GetNamePtr(),
              node2->GetType().c_str(), AutofuseUtils::VectorPairToStr(node2_map).c_str());
    if (node1_map.empty() || node2_map.empty()) {
      GELOGD("axis map info empty, can fuse false.");
      return FAILED;
    }
    if (is_reduction) {
      if (!CanLoopMerge(asc_node1, asc_node2)) {
        GELOGI("pre is reduce, graph sched axis can't loop merge, can't fuse.");
        return FAILED;
      }
    } else {
      std::vector<ViewOpAttrInfo> attr_infos;
      if (!CanLoopMerge(asc_node1, asc_node2)) {
        std::vector<ViewOpAttrInfo> pre_attr_infos;
        auto pre_node_input_is_simplest_load = BackendUtils::PreNodeInputIsSimplestLoad(node2, subgraph_link.second, pre_attr_infos);
        auto cur_node_input_is_simplest_load = BackendUtils::CurNodeInputIsSimplestLoad(node2, subgraph_link.second, attr_infos);
        // 不能循环合并时，如果前序子图没有view、后续子图没有view可以融合(transpose场景)
        if (pre_node_input_is_simplest_load && cur_node_input_is_simplest_load) {
          GELOGD_IF(open_log_, "pre node is simplest load, cur node is simplest load, map info success.");
          return SUCCESS;
        }
        // 不能循环合并时，如果输入node单引用输出、前序子图没有view、后续子图有view可以融合
        if (!(pre_node_input_is_simplest_load && !cur_node_input_is_simplest_load && !fuse_info.HasMulReference() &&
              (node1->GetType() == kAscBackendType) && (node2->GetType() == kAscBackendType))) {
          GELOGD(
              "pre node not single reference or pre node is no simplest load or cur node is simplest load, can't "
              "fuse.");
          return FAILED;
        }
      } else {
        //        node1                         node1                        node1
        //         /  |                          /   |                         |
        //        /   |                         /    |                         |
        // 原图 node2 node3   -> 场景1：        node2 |            场景2：  FusedAscBackend
        //       |    |                         |    |
        //       concat                     FusedAscendBackend
        // 上面原图里node1的输出多引用，连接了node2和node3，此时有两种融合场景：
        // 场景1：node3和concat融合，然后node2不能和concat融合，此时需要判断node1和Fused节点是否能融合。如果node1的输出是多引用，同时Fused
        // 节点中与node1连接的节点中包含view op，那么node1和Fused节点不能融合。
        // 场景2：node3和node2与concat节点融合了，最后会判断node1和Fused节点是否能融合，这个时候node1和Fused节点只有一条边，fuse_info中没有
        // 多引用的信息。这个时候不能只用fuse_info.HasMulReference()来判断，还需要判断Fused节点中与node1连接的data节点后面是否有多个节点。
        // 如果后面有多个节点（> 1U），同时对应节点中也包含view op，那么node1和Fused节点不能融合。
        bool has_mul_reference = fuse_info.HasMulReference();
        auto asc_node = node2;
        auto index = subgraph_link.second;
        if (node2->GetType() == kFusedAscBackendType) {
          ComputeGraphPtr graph;
          GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph));
          const auto subgraph_input_nodes = graph->GetInputNodes();
          GE_ASSERT_TRUE(subgraph_input_nodes.size() > static_cast<size_t>(subgraph_link.second));
          auto data_node = subgraph_input_nodes.at(subgraph_link.second);
          GE_ASSERT_NOTNULL(data_node);
          has_mul_reference = has_mul_reference || (data_node->GetOutNodesSize() > 1U);
          int32_t in_anchor_idx;
          asc_node = BackendUtils::GetFusedAscBackendInputNode(node2, subgraph_link.second, in_anchor_idx);
          index = in_anchor_idx;
        }
        // 能循环合并时，如果输入node多引用输出、后续子图有view，不能融合
        if ((!BackendUtils::CurNodeInputIsSimplestLoad(asc_node, index, attr_infos) && has_mul_reference)) {
          GELOGD_IF(open_log_, "pre node not single reference and cur node not simplest load, can't fuse.");
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status AscGraphAxisMapping::CreateSubGraphAxisMapInfo(const NodePtr &node1, const NodePtr &node2,
                                                      const NodeFuseInfo &fuse_info) {
  node1_map_.clear();
  node2_map_.clear();
  pre_node_is_reduction_ = false;
  // 垂直融合轴映射
  if (ProcessSubGraphVerticalMapInfo(node1, node2, fuse_info, node1_map_, node2_map_) != SUCCESS) {
    return FAILED;
  }

  // concat如果同时存在水平融合以及垂直融合，只做垂直轴映射
  const auto fuse_type = BackendUtils::GetAllFuseType(node1, node2);
  bool is_both_horizontal_and_vertical = false;
  for (const auto fusion_strategy : FusionStrategyRegistry::Instance().Get(fuse_type)) {
    if (fusion_strategy != nullptr) {
      if (fusion_strategy->OnlyVerticalMapping(node1, node2, fuse_info)) {
        is_both_horizontal_and_vertical = true;
        break;
      }
    }
  }
  if (fuse_info.CanDoHorizontalMapping() && !is_both_horizontal_and_vertical) {
    // 水平融合轴映射
    if (ProcessSubGraphHorizontalMapInfo(node1, node2, fuse_info) != SUCCESS) {
      return FAILED;
    }
  }

  // 把axis id刷成连续的id
  GE_ASSERT_SUCCESS(FlashContinueAxisId());
  GELOGD_IF(open_log_, "node %s(%s) continue axis map info=%s and node %s(%s) continue axis map info=%s.",
            node1->GetNamePtr(), node1->GetType().c_str(), AutofuseUtils::VectorPairToStr(node1_map_).c_str(),
            node2->GetNamePtr(), node2->GetType().c_str(), AutofuseUtils::VectorPairToStr(node2_map_).c_str());
  return SUCCESS;
}
}  // namespace ge
