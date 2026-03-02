/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/buffer_pool_memory_pass.h"

#include <string>
#include <vector>
#include "common/checker.h"
#include "common/omg_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/math/math_util.h"

namespace ge {
namespace {
const size_t kBufferPoolNodeInSize = 1;
const size_t kBufferPoolNodeOutSize = 1;
constexpr const char_t *kOpNameSplitD = "SplitD";
} // namespace

Status BufferPoolMemoryPass::Run(ComputeGraphPtr graph) {
  if (graph == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Graph]Graph is nullptr");
    REPORT_INNER_ERR_MSG("E19999", "Input graph is nullptr");
    return PARAM_INVALID;
  }
  // The cache prefetching scheme is developed for very large models, which gets the weight data in advance
  // and allocates it to a special memory pool. When the large model is dynamic shape, it need to go through
  // the executor flow and is not allocated memory statically. This is another development point, so we will
  // skip the dynamic shape model processing here.
  if (graph->GetParentGraph() != nullptr || graph->GetGraphUnknownFlag()) {
    return SUCCESS;
  }
  if (!IsBufferPoolMemEnable(graph)) {
    GELOGD("[Check][Enable]Buffer pool memory is not enable, graph:%s.", graph->GetName().c_str());
    return SUCCESS;
  }
  Status ret = graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "[TopologicalSort][Graph]Graph name:%s.", graph->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Failed to topological sort for graph:%s.", graph->GetName().c_str());
    return ret;
  }

  ret = CopyOutForMultiUsedOutput(graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Copy][Output]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  GE_CHK_BOOL_RET_STATUS(DisableBufferPreventingCycle(graph) == SUCCESS,
                         FAILED,
                         "Failed to invoke DisableBufferPreventingCycle");
  ret = GetBufferPoolAndPeerCalcNodes(graph);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][BufferPoolNode]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  if (calc_nodes_.empty()) {
    GELOGE(FAILED, "[Check][BufferPoolNode]Graph:%s.", graph->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "All Buffer pool nodes are isolated nodes in graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  GroupBufferPoolNodes();
  ret = AllocateAllBufferPoolSpace();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Alloc][BufferPoolMem]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }

  ret = SetResultOfMemoryAndEvent();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Set][Result]Graph:%s.", graph->GetName().c_str());
    return FAILED;
  }
  ret = graph->TopologicalSorting();
  if (ret != SUCCESS) {
    GELOGE(ret, "[TopologicalSort][Graph]Graph name:%s.", graph->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Failed to topological sort for graph:%s.", graph->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

void BufferPoolMemoryPass::ClearQueue(std::queue<std::pair<std::string, uint32_t>> &q) {
  while (!q.empty()) {
    q.pop();
  }
}

bool BufferPoolMemoryPass::IsBufferPoolMemEnable(const ComputeGraphPtr &graph) {
  for (NodePtr &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    if (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE)) {
      return true;
    }
  }
  return false;
}

Status BufferPoolMemoryPass::CheckBufferPoolSize(int64_t total_size, int64_t pool_id, int64_t buffer_pool_size,
                                                 std::unordered_map<int64_t, int64_t> &calc_total_size) {
  auto iter = calc_total_size.find(pool_id);
  if (iter == calc_total_size.end()) {
    calc_total_size[pool_id] = total_size;
  } else {
    FMK_INT64_ADDCHECK(calc_total_size[pool_id], total_size);
    calc_total_size[pool_id] += total_size;
  }
  if (calc_total_size[pool_id] > buffer_pool_size) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]The memory required at the same is greater than buffer pool size, "
          "pool id:%ld, pool size:%ld, required size:%ld.", pool_id, buffer_pool_size, calc_total_size[pool_id]);
    REPORT_INNER_ERR_MSG("E19999", "The memory required at the same is greater than buffer pool size, pool id:%ld,"
                       " pool size:%ld, required size:%ld.", pool_id, buffer_pool_size, calc_total_size[pool_id]);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::TryToFixNodeOrder(NodePtr &pre_node, NodePtr &curr_node, bool &not_change) {
  auto pre_node_graph = pre_node->GetOwnerComputeGraph();
  auto curr_node_graph = curr_node->GetOwnerComputeGraph();
  std::string pre_node_stream_label;
  (void) AttrUtils::GetStr(pre_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, pre_node_stream_label);
  std::string curr_node_stream_label;
  (void) AttrUtils::GetStr(curr_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, curr_node_stream_label);
  not_change = true;
  if ((pre_node_graph == curr_node_graph) &&
      ((pre_node_stream_label == curr_node_stream_label) || pre_node_stream_label.empty())) {
    // Same subgraph, including simultaneously in the root graph.
    auto ret = ge::GraphUtils::AddEdge(pre_node->GetOutControlAnchor(), curr_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Add][Edge]Src:%s, dst:%s.", pre_node->GetName().c_str(), curr_node->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Failed to add ctrl edge from %s to %s.", pre_node->GetName().c_str(),
                        curr_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    not_change = false;
  } else {
    // Two nodes are located on different graphs.
    auto pre_op_desc = pre_node->GetOpDesc();
    auto cur_op_desc = curr_node->GetOpDesc();
    GE_CHECK_NOTNULL(pre_op_desc);
    GE_CHECK_NOTNULL(cur_op_desc);
    // The topo order is correct to ensure node dependency,
    // there is no need to add control edges.
    if (pre_op_desc->GetId() > cur_op_desc->GetId()) {
      GELOGE(INTERNAL_ERROR, "[Check][Dependency]Invalid dependency, pre node:%s, curr node:%s.",
             pre_node->GetName().c_str(), curr_node->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Invalid dependency, pre node:%s, curr node:%s.",
                         pre_node->GetName().c_str(), curr_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GELOGI("[Check][Dependency]The two nodes are located in sub graphs of different parent nodes and meet the "
           "dependency relationship. pre:%s, curr:%s.", pre_node->GetName().c_str(), curr_node->GetName().c_str());
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::InsertMemCpyNodeAfter(NodePtr &node) const {
  auto out_anchor = node->GetOutDataAnchor(kBufferPoolNodeOutIndex);
  OpDescBuilder op_desc_builder(node->GetName() + "_memcpy_async", MEMCPYASYNC);
  auto mem_copy_op = op_desc_builder.AddInput("x", node->GetOpDesc()->GetOutputDesc(kBufferPoolNodeOutIndex))
    .AddOutput("y", node->GetOpDesc()->GetOutputDesc(kBufferPoolNodeOutIndex))
    .Build();
  std::string batch_label;
  bool get_attr = AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, batch_label);
  if (get_attr && !batch_label.empty()) {
    (void) AttrUtils::SetStr(mem_copy_op, ATTR_NAME_STREAM_LABEL, batch_label);
  }
  GE_ASSERT_NOTNULL(out_anchor);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  std::vector<InDataAnchorPtr> in_anchors(peer_in_anchors.begin(), peer_in_anchors.end());
  if (GraphUtils::InsertNodeAfter(out_anchor, in_anchors, mem_copy_op) == nullptr) {
    GELOGE(FAILED, "[Insert][Node] Node:%s.", node->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Failed to insert mem copy node after %s.", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::CopyOutForMultiUsedOutput(ComputeGraphPtr &graph) const {
  bool changed = false;
  for (NodePtr &node : graph->GetAllNodes()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    bool use_buffer_pool = op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE);
    if (use_buffer_pool) {
      if ((node->GetInDataNodes().size() == kBufferPoolNodeInSize) &&
          (node->GetOutDataNodes().size() == kBufferPoolNodeOutSize)) {
        continue;
      } else if ((node->GetAllInDataAnchors().size() == kBufferPoolNodeInSize) &&
                 (node->GetAllOutDataAnchors().size() == kBufferPoolNodeOutSize)) {
        // A prefetching output is used in multiple places. Copy one so that the prefetching node remains
        // single input and single output.
        if (InsertMemCpyNodeAfter(node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Insert][MemCpy]Node:%s.", node->GetName().c_str());
          return INTERNAL_ERROR;
        }
        changed = true;
        GELOGI("[Insert][Node]Insert mem copy node after %s.", node->GetName().c_str());
      } else {
        GELOGE(PARAM_INVALID, "[Check][InputOutput]Only support single input and single output, "
               "node:%s.", node->GetName().c_str());
        REPORT_INNER_ERR_MSG("E19999", "Only support single input and single output, node:%s.", node->GetName().c_str());
        return PARAM_INVALID;
      }
    }
  }
  if (changed) {
    Status ret = graph->TopologicalSorting();
    if (ret != SUCCESS) {
      GELOGE(ret, "[TopologicalSort][Graph]Graph name:%s.", graph->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Failed to topological sort for graph:%s.", graph->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::GetBufferPoolAndPeerCalcNodes(const ComputeGraphPtr &graph) {
  std::unordered_map<std::string, std::unordered_map<int64_t, std::set<NodePtr>>> unique_calc_nodes;
  for (const NodePtr &node : graph->GetAllNodes()) {
    auto in_data_nodes = node->GetInDataNodes();
    for (NodePtr &in_node : in_data_nodes) {
      int64_t buffer_pool_id = 0;
      int64_t buffer_pool_size = 0;
      bool get_attr = AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_ID, buffer_pool_id);
      get_attr = get_attr && (AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_SIZE, buffer_pool_size));
      if (get_attr) {
        const auto calc_node = BypassRefIoNodes(node);
        GE_CHECK_NOTNULL(calc_node);
        std::string batch_label;
        (void) AttrUtils::GetStr(calc_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
        peer_buffer_node_item_[batch_label][calc_node].emplace_back(in_node, 0, 0);
        buffer_node_to_calc_[batch_label][in_node] = calc_node;
        if (unique_calc_nodes[batch_label][buffer_pool_id].count(calc_node) == 0) {
          calc_nodes_[batch_label][buffer_pool_id].emplace_back(calc_node);
          unique_calc_nodes[batch_label][buffer_pool_id].insert(calc_node);
        }
        GELOGI("[Get][BufferNode]Calc node:%s, pool node:%s.",
               calc_node->GetName().c_str(),
               in_node->GetName().c_str());
        Status ret = SetBufferPoolSize(batch_label, buffer_pool_id, buffer_pool_size);
        if (ret != SUCCESS) {
          GELOGE(ret, "[Set][BufferPoolSize]Node:%s", in_node->GetName().c_str());
          return ret;
        }
      }
    }
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::SetBufferPoolSize(const std::string &batch_label, int64_t id, int64_t size) {
  auto iter = buffer_pool_size_[batch_label].find(id);
  if (iter != buffer_pool_size_[batch_label].end() && iter->second != size) {
    GELOGE(PARAM_INVALID, "[Check][BufferPoolSize]Get different size with the same id, "
           "id:%ld, original size:%ld, this size:%ld.", id, iter->second, size);
    REPORT_INNER_ERR_MSG("E19999", "Get different size with the same id, "
                       "id:%ld, original size:%ld, this size:%ld.", id, iter->second, size);
    return PARAM_INVALID;
  }
  buffer_pool_size_[batch_label][id] = size;
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateAllBufferPoolSpace() {
  for (const auto &iter : calc_nodes_) {
    std::string batch_label = iter.first;
    Status ret = AllocateSpaceInBatch(calc_nodes_[batch_label],
                                      buffer_pool_size_[batch_label],
                                      buffer_node_to_calc_[batch_label]);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Alloc][InBatch]Batch_label:%s.", batch_label.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Failed to allocate space in batch, batch_label:%s.", batch_label.c_str());
      return ret;
    }
    GELOGI("[Alloc][InBatch]Alloc space in batch successfully, batch label:%s.", batch_label.c_str());
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateSpaceInBatch(
    const std::map<int64_t, std::vector<NodePtr>> &calc_nodes,
    const std::unordered_map<int64_t, int64_t> &buffer_pool_size_map,
    const std::unordered_map<NodePtr, NodePtr> &buffer_node_to_calc) {
  for (const auto &calc_node_in_pool : calc_nodes) {
    int64_t pool_id = calc_node_in_pool.first;
    int64_t buffer_pool_size = buffer_pool_size_map.at(pool_id);
    ClearQueue(mem_ctrl_event_);
    ClearQueue(stream_ctrl_event_);
    BufferPool buffer_pool(pool_id, buffer_pool_size, buffer_node_to_calc);
    Status ret = AllocateSpaceInBufferPool(buffer_pool, calc_node_in_pool.second);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Alloc][InBufferPool]Pool id:%ld, pool size:%ld.", pool_id, buffer_pool_size);
      REPORT_INNER_ERR_MSG("E19999", "Failed to allocate space in buffer pool, id:%ld, pool size:%ld.",
                         pool_id, buffer_pool_size);
      return ret;
    }
    GELOGI("[Alloc][InBufferPool]Alloc space in buffer pool successfully, pool id:%ld.", pool_id);
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateSpaceInBufferPool(
    const BufferPool &buffer_pool,
    const std::vector<NodePtr> &calc_nodes_in_pool) {
  int64_t pool_id = buffer_pool.pool_id;
  int64_t buffer_pool_size = buffer_pool.pool_size;
  int64_t next_start = 0;
  NodePtr pre_buffer_pool_node = nullptr;
  std::queue<BufferPoolNodeItem> node_mem_range_in_pool;
  node_mem_range_in_pool.emplace(nullptr, 0, buffer_pool_size);
  for (auto &calc_node : calc_nodes_in_pool) {
    auto &peer_buffer_node_items = peer_buffer_node_items_[calc_node];
    std::unordered_map<int64_t, int64_t> calc_total_size;
    size_t input_buffer_node_num = 0;
    for (auto &node_item_group : peer_buffer_node_items) {
      int64_t group_total_mem_size = 0;
      ++input_buffer_node_num;
      std::vector<int64_t> total_sizes;
      for (auto &node_item : node_item_group.node_items) {
        auto peer_buffer_node = node_item.node;
        GE_CHECK_NOTNULL(peer_buffer_node);
        int64_t total_size = 0;
        GE_CHK_STATUS_RET(GetMemorySize(peer_buffer_node, total_size),
                          "[Get][MemSize]Node:%s, calc_node:%s.",
                          peer_buffer_node->GetName().c_str(), calc_node->GetName().c_str());
        total_sizes.emplace_back(total_size);
        FMK_INT64_ADDCHECK(group_total_mem_size, total_size);
        group_total_mem_size += total_size;
        GE_CHK_STATUS_RET(CheckBufferPoolSize(total_size, pool_id, buffer_pool_size, calc_total_size),
                          "[Check][BufferPoolSize]Capacity is not enough for all data, calc_node:%s.",
                          calc_node->GetName().c_str());
        node_item.total_size = total_size;
      }
      for (size_t i = 0U; i < node_item_group.node_items.size(); ++i) {
        const auto &peer_buffer_node = node_item_group.node_items[i].node;
        bool is_last_input = (input_buffer_node_num == peer_buffer_node_items.size()) &&
            (i == node_item_group.node_items.size() - 1U);
        BufferPoolNodeItem buffer_pool_node_item(peer_buffer_node, calc_node, pre_buffer_pool_node,
                                                 total_sizes[i], 0, 0, is_last_input);
        buffer_pool_node_item.is_first_in_group = (i == 0U);
        GE_CHK_STATUS_RET(AllocateSpaceForBufferPoolNode(next_start, buffer_pool, buffer_pool_node_item,
                                                         group_total_mem_size, node_mem_range_in_pool),
                          "[Alloc][ForNode]Pool node:%s, calc_node:%s.",
                          peer_buffer_node->GetName().c_str(), calc_node->GetName().c_str());
        pre_buffer_pool_node = peer_buffer_node;
      }
    }
  }
  return SUCCESS;
}

Status BufferPoolMemoryPass::AllocateSpaceForBufferPoolNode(int64_t &next_start,
                                                            const BufferPool &buffer_pool,
                                                            BufferPoolNodeItem &buffer_pool_node_item,
                                                            const int64_t group_total_mem_size,
                                                            std::queue<BufferPoolNodeItem> &node_mem_range_in_pool) {
  NodePtr buffer_node = buffer_pool_node_item.node;
  /// In the scenario where there are multiple PREFETCH operators in the inputs of the calculation operator,
  /// the addition of events is optimized to only add events after the last PREFETCH operator.
  ///              w1         w2         w3         w4         w5
  ///              |          |          |          |          |
  ///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5   xxx
  ///               \         /           \         /          \       /
  ///                \       /             \       /            \     /
  ///                 \     /               \     /              \   /
  ///                  node1                 node2               node3
  ///                   |                     |                   |
  ///                   |                     |                   |
  ///                    ---------------  other nodes  ------------
  ///
  /// The event id of the PREFETCH operator to the calculation operator needs to be generated before
  /// FixTheTimingOfDependentNodes, because FixTheTimingOfDependentNodes may add a new id to stream_ctrl_event_,
  /// and this id cannot be reused until the next PREFETCH operator in the sequence.
  if (buffer_pool_node_item.is_last_input) {
    const auto &calc_node = buffer_pool_node_item.out_calc_node;
    const auto logic_event = GenerateEventId(buffer_node->GetName(), stream_ctrl_event_);
    node_event_multiplexing_[buffer_node].push_back(string("SendTo;" + calc_node->GetName() +
        ";" + std::to_string(logic_event)));
    mem_ctrl_event_.emplace(calc_node->GetName(), logic_event);
    GELOGI("[Alloc][ForNode]Buffer pool node %s send to %s, offset start:%ld, send event id:%u.",
           buffer_node->GetName().c_str(), calc_node->GetName().c_str(),
           buffer_pool_node_item.offset_start, logic_event);
  }
  const auto mem_size_needed = buffer_pool_node_item.is_first_in_group ?
                               group_total_mem_size : buffer_pool_node_item.total_size;
  NodePtr dependent_calc_node = GetOffsetAndDependency(next_start,
                                                       mem_size_needed,
                                                       buffer_pool.pool_size,
                                                       buffer_pool.buffer_node_to_calc,
                                                       node_mem_range_in_pool);
  if (buffer_pool_node_item.is_first_in_group && (dependent_calc_node != nullptr)) {
    GE_CHK_STATUS_RET(FixTheTimingOfDependentNodes(dependent_calc_node, buffer_node),
                      "[Fix][Timing]Pool_id:%ld, pool node:%s, dependent node:%s.",
                      buffer_pool.pool_id, buffer_node->GetName().c_str(), dependent_calc_node->GetName().c_str());
  }

  buffer_pool_node_item.offset_start = next_start;
  buffer_node_logical_offset_[buffer_node].push_back(buffer_pool_node_item.total_size);
  buffer_node_logical_offset_[buffer_node].push_back(next_start);
  FMK_INT64_ADDCHECK(next_start, buffer_pool_node_item.total_size);
  next_start += buffer_pool_node_item.total_size;
  buffer_pool_node_item.offset_end = next_start;
  node_mem_range_in_pool.push(buffer_pool_node_item);
  if (buffer_pool_node_item.pre_buffer_pool_node != nullptr) {
    bool not_change = true;
    auto ret = TryToFixNodeOrder(buffer_pool_node_item.pre_buffer_pool_node, buffer_node, not_change);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Fix][BufferPoolNodeOrder]Pre node:%s, curr node:%s.",
             buffer_pool_node_item.pre_buffer_pool_node->GetName().c_str(), buffer_node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

/// When generating the event ID, determine whether the name of the queue head node is the same as the name of
/// the operator, in order to handle such scenarios:
///              w1         w2         w3        w4         w5
///              |          |          |         |          |
///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///             |          |          |         |          |
///           node1      node2      node3     node4      node5
///
///  Memory distribution:
///
///      |____w1_____|__|
///      |____w2_____|__|
///      |____w3_____|__|
///      |______w4______|
///      |______w5______|
/// In this scenario, prefetch2 depends on node1. If the dependency is handled by adding an event of node1 to prefetch2,
/// the id sent by prefetch2 will be the same as the id it receives.Although Runtime supports this through WaitReset,
/// we consider this a dangerous operation and avoid it.
uint32_t BufferPoolMemoryPass::GenerateEventId(const std::string &node_name,
                                               std::queue<std::pair<std::string, uint32_t>> &event_queue) {
  uint32_t logic_event = logic_event_num_;
  if (!event_queue.empty()) {
    auto item = event_queue.front();
    if (item.first != node_name) {
      logic_event = item.second;
      event_queue.pop();
      return logic_event;
    }
  }
  ++logic_event_num_;
  return logic_event;
}

NodePtr BufferPoolMemoryPass::GetOffsetAndDependency(int64_t &next_start,
    int64_t total_mem_size,
    int64_t buffer_pool_size,
    const std::unordered_map<NodePtr, NodePtr> &buffer_node_to_calc,
    std::queue<BufferPoolMemoryPass::BufferPoolNodeItem> &nodes_in_buffer) const {
  // The buffer pool can no longer fit this Tensor and needs to turn back.
  if (next_start + total_mem_size > buffer_pool_size) {
    next_start = 0;
    if (!nodes_in_buffer.empty()) {
      // Take up the rest of the space at the end,
      nodes_in_buffer.back().offset_end = buffer_pool_size;
      // Pop the first tensor memory in the previous round of the previous round.
      nodes_in_buffer.pop();
    }
    while (!nodes_in_buffer.empty()) {
      auto node_item = nodes_in_buffer.front();
      // Go to the begin of previous round.
      if (node_item.offset_start == 0) {
        break;
      }
      nodes_in_buffer.pop();
    }
  }

  while (!nodes_in_buffer.empty()) {
    auto node_item = nodes_in_buffer.front();
    if (next_start + total_mem_size <= node_item.offset_end) {
      auto pool_node = node_item.node;
      if (pool_node == nullptr) {
        return nullptr;
      }
      auto output_calc = buffer_node_to_calc.find(pool_node);
      if (output_calc != buffer_node_to_calc.end()) {
        return output_calc->second;
      }
      return nullptr;
    }
    nodes_in_buffer.pop();
  }
  return nullptr;
}

Status BufferPoolMemoryPass::FixTheTimingOfDependentNodes(NodePtr &dependent_calc_node, NodePtr &curr_pool_node) {
  // The previous process ensures that all pointers are not null.
  bool not_change = false;
  Status ret = TryToFixNodeOrder(dependent_calc_node, curr_pool_node, not_change);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Fix][NodeOrder]Src:%s, dst:%s.",
           dependent_calc_node->GetName().c_str(), curr_pool_node->GetName().c_str());
    return ret;
  }
  if (not_change) {
    return SUCCESS;
  }
  uint32_t logic_event = GenerateEventId(dependent_calc_node->GetName(), mem_ctrl_event_);
  node_event_multiplexing_[curr_pool_node].push_back(string("RecvFrom;" + dependent_calc_node->GetName() +
      ";" + std::to_string(logic_event)));
  stream_ctrl_event_.emplace(curr_pool_node->GetName(), logic_event);
  GELOGI("[Fix][Timing]Add ctrl edge for buffer pool memory from %s to %s, buffer pool node recv event:%u.",
         dependent_calc_node->GetName().c_str(), curr_pool_node->GetName().c_str(), logic_event);
  return SUCCESS;
}

Status BufferPoolMemoryPass::SetResultOfMemoryAndEvent() {
  for (auto &iter : node_event_multiplexing_) {
    auto node = iter.first;
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool ret = AttrUtils::SetListStr(op_desc, ATTR_NAME_EVENT_MULTIPLEXING, iter.second);
    if (!ret) {
      GELOGE(INTERNAL_ERROR, "[Set][Attr]Node:%s.", node->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Failed to set event reuse info, node:%s.", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
  }
  for (const auto &node_and_logical_offset : buffer_node_logical_offset_) {
    const auto &node = node_and_logical_offset.first;
    const auto &logical_offset = node_and_logical_offset.second;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListInt(node->GetOpDesc(),
                                                 ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET,
                                                 logical_offset),
                           INTERNAL_ERROR, "[Set][Attr]Node:%s.", node->GetName().c_str());
  }
  return SUCCESS;
}

bool BufferPoolMemoryPass::HasDependency(const NodePtr &from_node, const NodePtr &to_node) {
  std::vector<NodePtr> nodes{to_node};
  std::unordered_set<const Node *> visited;
  while (!nodes.empty()) {
    if (FindOrBackward(from_node, nodes, visited)) {
      return true;
    }
  }
  return false;
}

bool BufferPoolMemoryPass::FindOrBackward(const NodePtr &target_node,
                                          std::vector<NodePtr> &nodes,
                                          std::unordered_set<const Node *> &visited) {
  std::vector<NodePtr> in_nodes;
  std::unordered_set<const Node *> unique_in_nodes;
  for (const auto &node : nodes) {
    const auto *p_node = node.get();
    visited.emplace(p_node);
    for (const auto &in_node : node->GetInAllNodes()) {
      const auto *p_in_node = in_node.get();
      if (visited.find(p_in_node) != visited.cend()) {
        continue;
      }
      if (in_node == target_node) {
        return true;
      }
      if (unique_in_nodes.find(p_in_node) == unique_in_nodes.cend()) {
        unique_in_nodes.emplace(p_in_node);
        in_nodes.emplace_back(in_node);
      }
    }
  }
  nodes = std::move(in_nodes);
  return false;
}

Status BufferPoolMemoryPass::DisableBufferPreventingCycle(const ComputeGraphPtr &compute_graph) {
  GE_CHK_STATUS_RET(GetBufferPoolAndPeerCalcNodes(compute_graph));
  for (const auto &batch_label_and_calc_nodes : calc_nodes_) {
    const auto &batch_label = batch_label_and_calc_nodes.first;
    auto &calc_node_to_buffer_node_items = peer_buffer_node_item_[batch_label];
    for (auto &pool_id_and_calc_nodes : batch_label_and_calc_nodes.second) {
      auto &calc_nodes = pool_id_and_calc_nodes.second;
      GE_CHK_STATUS_RET_NOLOG(DoCheckAndDisable(calc_nodes, calc_node_to_buffer_node_items));
    }
  }

  calc_nodes_.clear();
  peer_buffer_node_item_.clear();
  buffer_node_to_calc_.clear();
  buffer_pool_size_.clear();
  return SUCCESS;
}

Status BufferPoolMemoryPass::DoCheckAndDisable(
    const std::vector<NodePtr> &calc_nodes,
    std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>> &calc_node_to_buffer_node_items) {
  NodePtr pre_buffer_pool_node = nullptr;
  std::vector<std::pair<NodePtr, NodePtr>> edges_added;
  for (const auto &calc_node : calc_nodes) {
    auto &buffer_node_items = calc_node_to_buffer_node_items[calc_node];
    for (auto &node_item : buffer_node_items) {
      const auto &cur_node = node_item.node;
      if (pre_buffer_pool_node == nullptr) {
        pre_buffer_pool_node = cur_node;
        continue;
      }

      if (HasDependency(cur_node, pre_buffer_pool_node)) { // reversed dependency
        GELOGW("Cannot add control edge, would cause cycle: [%s]->[%s]",
               pre_buffer_pool_node->GetName().c_str(),
               cur_node->GetName().c_str());
        GELOGW("[%s] remove buffer pool attrs", cur_node->GetName().c_str());
        const auto &op_desc = cur_node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        (void) op_desc->DelAttr(ATTR_NAME_BUFFER_POOL_SIZE);
        (void) op_desc->DelAttr(ATTR_NAME_BUFFER_POOL_ID);
        continue;  // do not update pre_buffer_pool_node
      }
      if (!pre_buffer_pool_node->GetOutControlAnchor()->IsLinkedWith(cur_node->GetInControlAnchor())) {
        GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(pre_buffer_pool_node->GetOutControlAnchor(),
                                                    cur_node->GetInControlAnchor()),
                                "Failed to add edge: [%s]->[%s]",
                                pre_buffer_pool_node->GetName().c_str(),
                                cur_node->GetName().c_str());
        edges_added.emplace_back(pre_buffer_pool_node, cur_node);
      }
      pre_buffer_pool_node = cur_node;
    }
  }
  // rollback
  for (const auto &edge : edges_added) {
    (void) GraphUtils::RemoveEdge(edge.first->GetOutControlAnchor(), edge.second->GetInControlAnchor());
  }
  return SUCCESS;
}

void BufferPoolMemoryPass::GroupBufferPoolNodes() {
  for (auto &label_and_calc_nodes : calc_nodes_) {
    const auto &label = label_and_calc_nodes.first;
    auto &buffer_node_items = peer_buffer_node_item_[label];
    for (auto &pool_id_and_calc_nodes : label_and_calc_nodes.second) {
      auto &calc_nodes = pool_id_and_calc_nodes.second;
      DoGroupBufferPoolNodes(calc_nodes, buffer_node_items);
    }
  }
  peer_buffer_node_item_.clear();
}

void BufferPoolMemoryPass::DoGroupBufferPoolNodes(
    std::vector<NodePtr> &calc_nodes,
    std::unordered_map<NodePtr, std::vector<BufferPoolNodeItem>> &buffer_node_items) {
  std::map<NodePtr, std::vector<BufferPoolNodeItem>> all_parent_buffer_node_items;
  std::vector<NodePtr> to_remove;
  std::vector<NodePtr> real_calc_nodes;
  for (const auto &calc_node : calc_nodes) {  // already in order
    const auto &op_desc = calc_node->GetOpDesc();
    bool is_buffer_node = op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE);
    auto &input_buffer_node_item_group = peer_buffer_node_items_[calc_node];
    auto &input_buffer_node_items = buffer_node_items[calc_node];
    for (auto &input_buffer_node_item : input_buffer_node_items) {  // iterate by inputs
      auto &parent_buffer_node_items = all_parent_buffer_node_items[input_buffer_node_item.node];
      parent_buffer_node_items.emplace_back(input_buffer_node_item);
      if (is_buffer_node) {
        all_parent_buffer_node_items.emplace(calc_node, parent_buffer_node_items);
      } else {
        BufferPoolNodeItemGroup node_item_group;
        node_item_group.node_items = parent_buffer_node_items;
        input_buffer_node_item_group.emplace_back(std::move(node_item_group));
      }
    }
    if (is_buffer_node) {
      GELOGD("remove prefetch node [%s] from calc nodes", calc_node->GetName().c_str());
      buffer_pool_nodes_.emplace(calc_node);
    } else {
      real_calc_nodes.emplace_back(calc_node);
    }
  }
  calc_nodes = std::move(real_calc_nodes);
}

NodePtr BufferPoolMemoryPass::GetLastOutDataNode(const NodePtr &node) {
  int64_t max_node_id = -1;
  NodePtr last_out_data_node;
  for (const auto &out_data_node : node->GetOutDataNodes()) {
    const auto &out_op_desc = out_data_node->GetOpDesc();
    GE_ASSERT_NOTNULL(out_op_desc);
    auto node_id = out_op_desc->GetId();
    if (node_id > max_node_id) {
      last_out_data_node = out_data_node;
      max_node_id = node_id;
    }
  }
  return last_out_data_node;
}

NodePtr BufferPoolMemoryPass::BypassRefIoNodes(const NodePtr &node) {
  const auto &op_type = NodeUtils::GetNodeType(node);
  // 对于其他ref类输入, 自动切分不会插入AllGather
  if ((op_type != SPLIT) && (op_type != kOpNameSplitD)) {
    return node;
  }

  auto calc_node = GetLastOutDataNode(node);
  GE_ASSERT_NOTNULL(calc_node);
  GELOGI("%s(%s) output memory could ref input memory, after bypassing it, calc node = %s",
         node->GetNamePtr(), op_type.c_str(), calc_node->GetNamePtr());
  return calc_node;
}

REG_PASS_OPTION("BufferPoolMemoryPass").LEVELS(OoLevel::kO3);
}  // namespace ge
