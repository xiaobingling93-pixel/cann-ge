/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logical_stream_allocator.h"
#include "assign_attached_stream_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "api/gelib/gelib.h"
#include "base/err_msg.h"

namespace {
constexpr const char_t *kHcomParallelGroupName = "-1";
constexpr const char_t kNewStreamId[] = "NewStreamId";
constexpr const char_t *kDisableIneffectiveMultiStreamOptimize = "DISABLE_INEFFECTIVE_MULTI_STREAM_OPTIMIZE";
constexpr int64_t kMainStreamId = 0;
constexpr uint32_t kWhileBodyIndex = 1;

// 逻辑流分配pass内打印info日志使用，可默认将pass name打印出来
#define GE_STREAM_PASS_LOGI(fmt, ...) \
  GELOGI("[%s]" fmt, this->GetName().c_str(), ##__VA_ARGS__)

bool HasUserStreamLabel(const ge::NodePtr &node) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  return ge::AttrUtils::HasAttr(node->GetOpDesc(), ge::public_attr::USER_STREAM_LABEL);
}

const std::set<std::string> kWhileOpTypes{"While"};
bool IsDynamicWhileBodyNetOutput(const ge::NodePtr &node) {
  if (node->GetType() == "NetOutput") {
    const auto owner_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(owner_graph);
    const auto parent_node = owner_graph->GetParentNode();
    if ((parent_node != nullptr) && (kWhileOpTypes.count(parent_node->GetType()) > 0)) {
      // index of body_subgraph is 1
      auto body_graph = ge::NodeUtils::GetSubgraph(*parent_node, kWhileBodyIndex);
      ge::ComputeGraphPtr origin_owner_graph = owner_graph->TryGetExtAttr("part_src_graph", ge::ComputeGraphPtr());
      auto while_owner_graph = parent_node->GetOwnerComputeGraph();
      if (while_owner_graph->GetGraphUnknownFlag() && (origin_owner_graph == body_graph)) {
        return true;
      }
    }
  }
  return false;
}

bool HasSameStreamId(const ge::Node::Vistor<ge::NodePtr> &nodes, int64_t stream_id) {
  for (const auto &node : nodes) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetStreamId() == stream_id) {
      return true;
    }
  }
  return false;
}

/**
 * @brief 收集当前节点的输入输出里是相同stream id且与当前节点在拓扑序上最接近的输入输出节点
 *
 * @param node 当前节点
 * @param stream_id_to_io_nodes 输出参数，键为stream id，值为该stream id在当前节点输入输出方向上拓扑序最接近的一对节点
 *                             第一个元素为输入节点，第二个元素为输出节点
 * @return ge::Status 成功返回SUCCESS
 *
 * 该函数会遍历当前节点的所有输入和输出节点，对于每一个有效的stream id，
 * 在输入方向上寻找拓扑序最大的节点（即最靠近当前节点），在输出方向上寻找拓扑序最小的节点（即最靠近当前节点）。
 * 最终将这些信息存储在stream_id_to_io_nodes映射中。
 */
ge::Status CollectStreamIdToIoNodes(const ge::NodePtr &node,
                                    std::map<int64_t, std::pair<ge::NodePtr, ge::NodePtr>> &stream_id_to_io_nodes) {
  const auto &in_nodes = node->GetInNodes();
  const auto &out_nodes = node->GetOutNodes();
  for (const auto &input : in_nodes) {
    auto input_op_desc = input->GetOpDesc();
    GE_ASSERT_NOTNULL(input_op_desc);
    auto input_stream_id = input_op_desc->GetStreamId();
    if (input_stream_id == ge::kInvalidStream) {
      continue;
    }
    auto iter = stream_id_to_io_nodes.find(input_stream_id);
    if (iter == stream_id_to_io_nodes.end()) {
      stream_id_to_io_nodes[input_stream_id] = {input, nullptr};
    } else {
      const auto &io_nodes = iter->second;
      if ((io_nodes.first == nullptr) || (input->GetOpDesc()->GetId() > io_nodes.first->GetOpDesc()->GetId())) {
        iter->second.first = input;
      }
    }
  }
  for (const auto &output : out_nodes) {
    auto output_op_desc = output->GetOpDesc();
    GE_ASSERT_NOTNULL(output_op_desc);
    auto output_stream_id = output_op_desc->GetStreamId();
    if (output_stream_id == ge::kInvalidStream) {
      continue;
    }
    auto iter = stream_id_to_io_nodes.find(output_stream_id);
    if (iter == stream_id_to_io_nodes.end()) {
      stream_id_to_io_nodes[output_stream_id] = {nullptr, output};
    } else {
      const auto &io_nodes = iter->second;
      if ((io_nodes.second == nullptr) || (output->GetOpDesc()->GetId() < io_nodes.second->GetOpDesc()->GetId())) {
        iter->second.second = output;
      }
    }
  }
  return ge::SUCCESS;
}

bool HasOtherNodeBetweenIOInThisStream(const std::pair<ge::NodePtr, ge::NodePtr> &io_nodes, const std::set<int64_t> &ordered_node_ids) {
  if ((io_nodes.first == nullptr) || (io_nodes.second == nullptr)) {
    return true;
  }
  auto input_node_id = io_nodes.first->GetOpDesc()->GetId();
  auto output_node_id = io_nodes.second->GetOpDesc()->GetId();
  auto iter = ordered_node_ids.find(input_node_id);
  GE_ASSERT_TRUE(iter != ordered_node_ids.end());
  auto next_node_id = *(++iter);
  if (next_node_id != output_node_id) {
    return true;
  }
  return false;
}
}  // namespace
namespace ge {
LogicalStreamPass::LogicalStreamPass(const std::string &name) : name_(name) {}

const std::string &LogicalStreamPass::GetName() const {
  return name_;
}

Status AssignByLabelPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) {
  (void)graph;
  bool changed = false;
  int64_t &next_stream = context.next_stream;
  std::map<std::string, int64_t> label_streams;

  for (const SubgraphPtr &subgraph : subgraphs) {
    const std::string &stream_label = subgraph->subgraph_info.GetStreamLabel();
    if (!stream_label.empty()) {
      // Subgraphs of the same stream_label are assigned to the same stream,
      // and different stream_labels are assigned new streams.
      std::map<std::string, int64_t>::const_iterator iter = label_streams.find(stream_label);
      if (iter == label_streams.cend()) {
        subgraph->stream_id = next_stream;
        GE_STREAM_PASS_LOGI("[Assign][NewStreamId] %ld for label %s (engine: %s).", next_stream, stream_label.c_str(),
                            subgraph->engine_conf.id.c_str());

        label_streams.emplace(stream_label, next_stream);
        next_stream++;
      } else {
        subgraph->stream_id = iter->second;
      }
      changed = true;
    }
  }

  return changed ? SUCCESS : NOT_CHANGED;
}

Status IndependentStreamPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) {
  (void)graph;
  bool changed = false;
  int64_t &next_stream = context.next_stream;

  // <engine, <label, stream>>
  std::map<std::string, std::map<std::string, int64_t>> engine_streams;

  for (const SubgraphPtr &subgraph : subgraphs) {
    if (StreamUtils::HasAssignedUserStream(*subgraph)) {
      continue;
    }
    if (!StreamUtils::IsEngineIndependent(*subgraph)) {
      continue;
    }

    const std::string &engine = subgraph->engine_conf.id;
    const std::string &stream_label = subgraph->subgraph_info.GetStreamLabel();
    auto &label_streams = engine_streams[engine];
    std::map<std::string, int64_t>::const_iterator iter = label_streams.find(stream_label);
    if (iter == label_streams.cend()) {
      subgraph->stream_id = next_stream;
      GE_STREAM_PASS_LOGI("[Assign][NewStreamId:independent] %ld for engine %s (label: %s).", next_stream,
                          engine.c_str(), stream_label.c_str());

      label_streams.emplace(stream_label, next_stream);
      next_stream++;
    } else {
      subgraph->stream_id = iter->second;
    }
    changed = true;
  }

  return changed ? SUCCESS : NOT_CHANGED;
}

Status AssignByDependencyPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) {
  (void)graph;
  bool changed = false;
  is_memory_priority_ = IsMemoryPriority();
  const auto end_subgraph_map = StreamUtils::InitEndSubgraphMap(subgraphs);
  const auto pld_subgraph_map = StreamUtils::InitPldSubgraphMap(subgraphs);

  for (const SubgraphPtr &subgraph : subgraphs) {
    if (StreamUtils::HasAssignedStream(*subgraph)) {
      continue;
    }

    SubgraphPtr reusable_subgraph = GetReusableSubgraph(subgraph, end_subgraph_map, pld_subgraph_map);
    if (reusable_subgraph == nullptr) {
      (void)AssignNewStream(subgraph);
    } else {
      if (StreamUtils::HasAssignedStream(*reusable_subgraph)) {
        subgraph->stream_id = reusable_subgraph->stream_id;
      } else {
        int64_t stream_id = AssignNewStream(reusable_subgraph);
        subgraph->stream_id = stream_id;
        GE_STREAM_PASS_LOGI("[Assign][NewStreamId] %ld for Reusable subgraph %s cause has not been assigned before.",
               stream_id, reusable_subgraph->name.c_str());
      }

      if (reusable_subgraph->reused_subgraph != nullptr) {
        reusable_subgraph = reusable_subgraph->reused_subgraph;
      }

      subgraph->reused_subgraph = reusable_subgraph;
      reused_subgraphs_.emplace_back(subgraph, reusable_subgraph);
      GE_STREAM_PASS_LOGI("[Reuse][Stream]Subgraph %s of engine %s reuses stream of subgraph %s of engine %s.",
             subgraph->name.c_str(),
             subgraph->engine_conf.id.c_str(), reusable_subgraph->name.c_str(),
             reusable_subgraph->engine_conf.id.c_str());
    }
    changed = true;
  }

  UpdateAssignedSubgraphs(context);
  UpdateReusedSubgraphs();

  return changed ? SUCCESS : NOT_CHANGED;
}

bool AssignByDependencyPass::IsForceAttach(const SubgraphPtr &subgraph) const {
  for (const auto &node : subgraph->subgraph_info.GetSubGraph()->GetDirectNode()) {
    if ((node->GetOpDesc() != nullptr) && (node->GetOpDesc()->HasAttr(ATTR_NAME_FORCE_ATTACH_STREAM))) {
      return true;
    }
  }
  return false;
}

bool AssignByDependencyPass::SubGraphCouldReuse(
    const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
    const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  for (const auto &end_pld_pair : pred_subgraph->subgraph_info.GetEnd2PldMap()) {
    auto iter = pld_subgraph_map.find(end_pld_pair.second);
    if (iter != pld_subgraph_map.end()) {
      const SubgraphPtr &pred_subgraph_succ = iter->second;
      if ((pred_subgraph_succ != subgraph) && (pred_subgraph_succ->engine_conf.id == pred_subgraph->engine_conf.id)) {
        return false;
      }
    }
  }
  return true;
}

bool AssignByDependencyPass::IsMemoryPriority() const {
  std::string memory_optimization_policy;
  (void) ge::GetContext().GetOption(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  return (memory_optimization_policy == kMemoryPriority);
}

bool AssignByDependencyPass::CouldReuse(const SubgraphPtr &subgraph, const SubgraphPtr &pred_subgraph,
                                        const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  if (subgraph->engine_conf.scheduler_id != pred_subgraph->engine_conf.scheduler_id) {
    return false;
  }

  if (StreamUtils::IsEngineIndependent(*pred_subgraph) || StreamUtils::HasStreamLabel(*pred_subgraph)) {
    return false;
  }

  // If the engine of the predecessor subgraph is the same as the other successor subgraphs, the stream is not reused.
  // there is one exception: if current subgraph's has node which is forced to attach it's predecessor node
  if ((visited_subgraphs_.count(subgraph) == 0) && IsForceAttach(subgraph)) {
    GELOGI("Subgraph %s is set to be forced to attach its predecessor subgraph %s", subgraph->name.c_str(),
           pred_subgraph->name.c_str());
    return true;
  } else {
    (void)visited_subgraphs_.insert(subgraph);
  }

  // if use notify, do not execute the SubGraphCouldReuse
  if ((!is_memory_priority_) && (!SubGraphCouldReuse(subgraph, pred_subgraph, pld_subgraph_map))) {
    return false;
  }

  if ((subgraph->engine_conf.id == pred_subgraph->engine_conf.id) ||
      StreamUtils::IsEngineAttach(*subgraph)) {
    return true;
  }

  if ((pred_subgraph->reused_subgraph != nullptr) &&
      (pred_subgraph->reused_subgraph->engine_conf.id == subgraph->engine_conf.id)) {
    return true;
  }

  return false;
}

SubgraphPtr AssignByDependencyPass::GetReusableSubgraph(
    const SubgraphPtr &subgraph, const std::unordered_map<NodePtr, SubgraphPtr> &end_subgraph_map,
    const std::unordered_map<NodePtr, SubgraphPtr> &pld_subgraph_map) const {
  std::set<SubgraphPtr> reusable_subgraphs;
  const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
  for (const auto &pld_2_end : subgraph_info.GetPld2EndMap()) {
    const auto iter = end_subgraph_map.find(pld_2_end.second);
    if ((iter != end_subgraph_map.end()) && (iter->second != nullptr) &&
        (reusable_subgraphs.find(iter->second) == reusable_subgraphs.end())) {
      if (CouldReuse(subgraph, iter->second, pld_subgraph_map)) {
        reusable_subgraphs.emplace(iter->second);
      }
    }
  }

  return StreamUtils::GetTopPrioritySubgraph(reusable_subgraphs);
}

int64_t AssignByDependencyPass::AssignNewStream(SubgraphPtr subgraph) {
  const std::string &engine_name = subgraph->engine_conf.id;
  int64_t max_parallel_num = subgraph->max_parallel_num;

  int64_t stream_id = 0;
  std::map<std::string, int64_t>::const_iterator next_iter = engine_next_streams_.find(engine_name);
  if (next_iter != engine_next_streams_.cend()) {
    stream_id = next_iter->second;
  }

  if (stream_id >= max_parallel_num) {
    stream_id = 0;
  }

  subgraph->stream_id = stream_id;
  engine_next_streams_[engine_name] = stream_id + 1;
  assigned_subgraphs_.emplace_back(subgraph);

  if ((stream_id + 1) > engine_stream_num_[engine_name]) {
    engine_stream_num_[engine_name] = stream_id + 1;
  }

  GE_STREAM_PASS_LOGI("[Assign][NewStreamId:temp]id:%ld for Subgraph %s (engine: %s).", stream_id,
                      subgraph->name.c_str(), engine_name.c_str());
  return stream_id;
}

void AssignByDependencyPass::UpdateAssignedSubgraphs(Context &context) {
  // If the default stream is valid, the first assigned stream will reuse the default stream id
  // and other streams use new id. To ensure that the id of the new stream is continuous,
  // we first subtract one from next_stream.
  int64_t to_be_updated_stream = kInvalidStream;
  if (context.default_stream != kInvalidStream) {
    context.next_stream--;
    to_be_updated_stream = context.next_stream;
  }

  // Update the starting stream id for each engine.
  int64_t &next_stream = context.next_stream;
  std::map<std::string, int64_t> engine_start_streams;
  for (const auto &item : engine_stream_num_) {
    int64_t stream_count = item.second;
    engine_start_streams[item.first] = next_stream;
    next_stream += stream_count;
  }

  // Update the subgraph streams assigned by engine.
  for (auto &subgraph : assigned_subgraphs_) {
    subgraph->stream_id += engine_start_streams[subgraph->engine_conf.id];
    if (subgraph->stream_id == to_be_updated_stream) {
      subgraph->stream_id = context.default_stream;
      GE_STREAM_PASS_LOGI("Subgraph %s of engine %s reuses default stream %ld.", subgraph->name.c_str(),
             subgraph->engine_conf.id.c_str(), context.default_stream);
    } else {
      GE_STREAM_PASS_LOGI("[Update][StreamId]id:%ld for subgraph %s.", subgraph->stream_id, subgraph->name.c_str());
    }
  }
}

void AssignByDependencyPass::UpdateReusedSubgraphs() {
  // Update streams for the subgraphs of reusing stream.
  for (const auto &item : reused_subgraphs_) {
    auto &cur_subgraph = item.first;
    auto &reused_graph = item.second;
    cur_subgraph->stream_id = reused_graph->stream_id;
    GE_STREAM_PASS_LOGI("[Update][StreamId]id:%ld for subgraph %s.", cur_subgraph->stream_id,
                        cur_subgraph->name.c_str());
  }
}

Status SingleStreamPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) {
  (void)graph;
  // context.default_stream can be kInvalidStream only when graph is the root graph.
  int64_t new_stream = context.default_stream;
  if (new_stream == kInvalidStream) {
    new_stream = context.next_stream;
    ++context.next_stream;
  }

  for (const SubgraphPtr &subgraph : subgraphs) {
    if (!StreamUtils::HasAssignedStream(*subgraph)) {
      const std::string &stream_label = subgraph->subgraph_info.GetStreamLabel();
      if (!stream_label.empty()) {
        REPORT_INNER_ERR_MSG("E19999", "Stream labels are not supported in SingleStream mode "
                             "(subgraph: %s, stream label: %s)", subgraph->name.c_str(), stream_label.c_str());
        GELOGE(INTERNAL_ERROR, "[Get][Label] Stream labels are not supported (subgraph: %s, stream label: %s).",
               subgraph->name.c_str(), stream_label.c_str());
        return INTERNAL_ERROR;
      }
      subgraph->stream_id = new_stream;
    }
  }

  return SUCCESS;
}

Status NodeStreamUpdatePass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) {
  // Check if all subgraphs have been assigned a stream.
  for (const SubgraphPtr &subgraph : subgraphs) {
    const std::string &engine_name = subgraph->engine_conf.id;

    if ((!StreamUtils::IsEngineSkip(*subgraph)) && (!StreamUtils::HasAssignedStream(*subgraph))) {
      REPORT_INNER_ERR_MSG("E19999", "Subgraph %s has not yet been assigned a stream (engine: %s)",
                         subgraph->name.c_str(), engine_name.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Subgraph %s has not yet been assigned a stream (engine: %s).",
             subgraph->name.c_str(), engine_name.c_str());
      return INTERNAL_ERROR;
    } else {
      GE_STREAM_PASS_LOGI("[Assign][StreamId] %ld for Subgraph %s (engine: %s).", subgraph->stream_id,
                          subgraph->name.c_str(), engine_name.c_str());
    }
  }

  // Init the stream id of node.
  for (NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    node->GetOpDesc()->SetStreamId(kInvalidStream);
  }

  // Set the stream id of the subgraph to the node.
  for (const SubgraphPtr &subgraph : subgraphs) {
    int64_t stream_id = subgraph->stream_id;
    const std::string &engine_name = subgraph->engine_conf.id;
    const auto &compute_graph = subgraph->subgraph_info.GetSubGraph();
    GE_CHECK_NOTNULL(compute_graph);
    for (const NodePtr &node : compute_graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node->GetOpDesc());
      if (node->GetOpDesc()->HasAttr(ATTR_NAME_RTS_LABEL_NODE)) {
        node->GetOpDesc()->SetStreamId(context.default_stream);
        GELOGD("Node %s of type %s in subgraph %s is assigned parent stream %ld (engine: %s).", node->GetName().c_str(),
               node->GetType().c_str(), subgraph->name.c_str(), context.default_stream, engine_name.c_str());
      } else {
        node->GetOpDesc()->SetStreamId(stream_id);
        // 动态图上的while算子的静态body子图的NetOutput需要在stream id 0上，否则可能会引入动态shape多流构图的bug
        if (IsDynamicWhileBodyNetOutput(node)) {
          node->GetOpDesc()->SetStreamId(kMainStreamId);
          GELOGI("node %s set stream id from %ld to %ld", node->GetNamePtr(), stream_id, kMainStreamId);
        }
        GELOGD("[Assign][StreamId]id:%ld for Node %s of type %s in subgraph %s (engine: %s).",
               node->GetOpDesc()->GetStreamId(), node->GetName().c_str(), node->GetType().c_str(),
               subgraph->name.c_str(), engine_name.c_str());
      }
    }
  }

  return SUCCESS;
}

Status UpdateForParallelGroupPass::UpdateStreamIdFromPreNode(
    const NodePtr &cur_node, const std::unordered_map<ge::NodePtr, ge::NodePtr> &total_pld_to_end) const {
  // 在给hcom分配新的流之前，考虑是否可以复用前面的流。
  //   要求：
  //      (1) 当前算子仅有一个输入
  //      (2) 输入算子的streamid != -1
  GELOGD("cur_node:%s.", cur_node->GetName().c_str());
  const OpDescPtr &cur_op_desc = cur_node->GetOpDesc();
  if (cur_node->GetInDataNodesSize() == 1UL) {
    auto pre_node = cur_node->GetInDataNodes().at(0UL);
    GE_CHECK_NOTNULL(pre_node);
    if (pre_node->GetType() == PLACEHOLDER) {
      auto iter = total_pld_to_end.find(pre_node);
      GE_ASSERT_TRUE(iter != total_pld_to_end.end());
      pre_node = iter->second;
      GE_CHECK_NOTNULL(pre_node);
      auto pre_nodes = pre_node->GetInNodes();
      GE_ASSERT_TRUE(!pre_nodes.empty());
      pre_node = pre_nodes.at(0U);
    }
    const auto &pre_op_desc = pre_node->GetOpDesc();
    GE_CHECK_NOTNULL(pre_op_desc);
    const int64_t pre_stream_id = pre_op_desc->GetStreamId();
    GELOGD("cur_node:%s, pre_node:%s, pre_stream_id:%ld", cur_node->GetName().c_str(), pre_node->GetName().c_str(),
           pre_stream_id);
    if (pre_stream_id != kInvalidStream) {
      int64_t old_stream_id = cur_op_desc->GetStreamId();
      cur_op_desc->SetStreamId(pre_stream_id);
      GE_STREAM_PASS_LOGI("pre_stream_id != -1, Node %s assigned stream %ld from stream %ld.",
                          cur_op_desc->GetName().c_str(), pre_stream_id, old_stream_id);
      return SUCCESS;
    }
  }
  return FAILED;
}

Status UpdateForParallelGroupPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs,
                                       Context &context) {
  (void)graph;
  std::unordered_map<ge::NodePtr, ge::NodePtr> total_pld_to_end;
  std::map<int32_t, std::vector<NodePtr>> stream_op_map;
  for (const SubgraphPtr &subgraph : subgraphs) {
    const auto &compute_graph = subgraph->subgraph_info.GetSubGraph();
    const auto &pld_to_end = subgraph->subgraph_info.GetPld2EndMap();
    total_pld_to_end.insert(pld_to_end.begin(), pld_to_end.end());
    GE_CHECK_NOTNULL(compute_graph);
    for (const NodePtr &node : compute_graph->GetDirectNode()) {
      const OpDescPtr &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (op_desc->HasAttr(ATTR_NAME_PARALLEL_GROUP)) {
        int64_t op_desc_stream_id = op_desc->GetStreamId();
        if (!HasUserStreamLabel(node)) {
          stream_op_map[op_desc_stream_id].push_back(node);
        }
      }
    }
  }
  for (const auto &itr : stream_op_map) {
    if (itr.first == kInvalidStream) {
      continue;
    }
    std::map<std::string, int64_t> group_2_stream_id;
    for (const auto &node : itr.second) {
      const OpDescPtr &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::string group_name;
      if (!AttrUtils::GetStr(op_desc, ATTR_NAME_PARALLEL_GROUP, group_name)) {
        GELOGE(FAILED, "[Get][Attr] ATTR_NAME_PARALLEL_GROUP of node %s failed.", op_desc->GetName().c_str());
        REPORT_INNER_ERR_MSG("E19999", "Get node %s ATTR_NAME_PARALLEL_GROUP failed.", op_desc->GetName().c_str());
        return FAILED;
      }
      // node type is hcom and the parallel_group_name is set to -1, then do reuse stream_id action
      if ((group_name == kHcomParallelGroupName) && (UpdateStreamIdFromPreNode(node, total_pld_to_end) == SUCCESS)) {
        continue;
      }
      const std::map<std::string, int64_t>::const_iterator &it_find = group_2_stream_id.find(group_name);
      int64_t new_stream_id = kInvalidStream;
      int64_t old_stream_id = op_desc->GetStreamId();
      if (it_find != group_2_stream_id.cend()) {
        new_stream_id = it_find->second;
      } else {
        new_stream_id = context.next_stream++;
        group_2_stream_id[group_name] = new_stream_id;
      }
      op_desc->SetStreamId(new_stream_id);
      GE_STREAM_PASS_LOGI("Node %s assigned stream %ld from stream %ld.", op_desc->GetName().c_str(), new_stream_id,
                          old_stream_id);
    }
  }
  return SUCCESS;
}

int64_t UpdateForSkippedEnginePass::GetSingleInoutStream(const NodePtr &node) const {
  std::set<int64_t> stream_ids;

  for (const auto &in_node : node->GetInAllNodes()) {
    GE_CHECK_NOTNULL_EXEC(in_node->GetOpDesc(), return kInvalidStream);
    int64_t stream_id = in_node->GetOpDesc()->GetStreamId();
    if (stream_id != kInvalidStream) {
      stream_ids.insert(stream_id);
    }
  }

  for (const auto &out_node : node->GetOutAllNodes()) {
    GE_CHECK_NOTNULL_EXEC(out_node->GetOpDesc(), return kInvalidStream);
    int64_t stream_id = out_node->GetOpDesc()->GetStreamId();
    if (stream_id != kInvalidStream) {
      stream_ids.insert(stream_id);
    }
  }

  if (stream_ids.size() == 1) {
    int64_t stream_id = *(stream_ids.cbegin());
    GE_STREAM_PASS_LOGI("[Get][SingleStreamId]The stream of all input and output nodes of node %s (type: %s) is %ld.",
                        node->GetName().c_str(), node->GetType().c_str(), stream_id);
    return stream_id;
  }

  return kInvalidStream;
}


Status UpdateForMdeGroupPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs,
                                  Context &context) {
  (void)graph;
  std::map<int32_t, std::vector<OpDescPtr>> stream_op_map;
  for (const auto &subgraph : subgraphs) {
    const auto &compute_graph = subgraph->subgraph_info.GetSubGraph();
    GE_CHECK_NOTNULL(compute_graph);
    for (const auto &node : compute_graph->GetDirectNode()) {
      const OpDescPtr &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      if (!op_desc->HasAttr(ATTR_NAME_PARALLEL_GROUP) && op_desc->HasAttr(kNewStreamId)) {
        int64_t op_desc_stream_id = op_desc->GetStreamId();
        stream_op_map[op_desc_stream_id].push_back(op_desc);
      }
    }
  }
  for (const auto &iter : stream_op_map) {
    if (iter.first == kInvalidStream) {
      continue;
    }
    std::map<int64_t, int64_t> group_2_stream_id;
    for (const auto &op_desc : iter.second) {
      int64_t attr_new_stream_id;
      GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, kNewStreamId, attr_new_stream_id),
                     "[Get][Attr] %s of node %s failed,", kNewStreamId, op_desc->GetNamePtr());
      const std::map<int64_t, int64_t>::const_iterator &it_find = group_2_stream_id.find(attr_new_stream_id);
      int64_t new_stream_id = kInvalidStream;
      int64_t old_stream_id = op_desc->GetStreamId();
      if (it_find != group_2_stream_id.cend()) {
        new_stream_id = it_find->second;
      } else {
        new_stream_id = context.next_stream++;
        group_2_stream_id[attr_new_stream_id] = new_stream_id;
      }
      op_desc->SetStreamId(new_stream_id);
      GE_STREAM_PASS_LOGI("Node = %s assigned stream %ld from stream %ld", op_desc->GetNamePtr(), new_stream_id,
                          old_stream_id);
    }
  }
  return SUCCESS;
}

Status UpdateForSkippedEnginePass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs,
                                       Context &context) {
  (void)graph;
  (void)context;
  std::set<OpDescPtr> ops_without_label;

  // Check if subgraph is engine skipped and without stream label or not
  for (const SubgraphPtr &subgraph : subgraphs) {
    if (StreamUtils::IsEngineSkip(*subgraph)) {
      const auto &compute_graph = subgraph->subgraph_info.GetSubGraph();
      GE_CHECK_NOTNULL(compute_graph);
      for (const NodePtr &node : compute_graph->GetDirectNode()) {
        const auto &op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        auto stream_id = op_desc->GetStreamId();
        if ((stream_id != kInvalidStream) && !StreamUtils::HasStreamLabel(*subgraph)) {
          ops_without_label.emplace(op_desc);
        }
      }
    }
  }

  // Try reassign the stream id
  for (const NodePtr &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int64_t stream_id = op_desc->GetStreamId();
    if (ops_without_label.find(op_desc) != ops_without_label.end()) {
      if ((op_desc->GetSubgraphInstanceNames().empty()) && AreAllPredStreamsInvalid(node)) {
        op_desc->SetStreamId(kInvalidStream);
        GELOGI("Node %s of type %s reassign to stream %ld from stream %ld.", node->GetName().c_str(),
               node->GetType().c_str(), kInvalidStream, stream_id);
      } else if (node->GetOutNodesSize() != 0U) {
        int64_t inout_stream = GetSingleInoutStream(node);
        if (inout_stream != kInvalidStream) {
          op_desc->SetStreamId(inout_stream);
          GE_STREAM_PASS_LOGI("[Reassign][StreamId]%ld for Node %s of type %s from stream %ld.", inout_stream,
                              node->GetName().c_str(), node->GetType().c_str(), stream_id);
        }
      }
    }
  }

  return SUCCESS;
}

bool UpdateForSkippedEnginePass::AreAllPredStreamsInvalid(const NodePtr &node) const {
  const auto &in_data_anchors = node->GetAllInDataAnchorsPtr();
  for (const auto in_data_anchor : in_data_anchors) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    const auto &out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    const auto in_node = out_anchor->GetOwnerNodeBarePtr();  // in_node and GetOpDescBarePtr must not be null
    int64_t stream_id = in_node->GetOpDescBarePtr()->GetStreamId();
    if (stream_id != kInvalidStream) {
      return false;
    }
  }
  const auto &in_control_anchor = node->GetInControlAnchor();
  if (in_control_anchor != nullptr) {
    for (const auto out_control_anchor : in_control_anchor->GetPeerOutControlAnchorsPtr()) {
      const auto in_node = out_control_anchor->GetOwnerNodeBarePtr();  // in_node and GetOpDescBarePtr must not be null
      int64_t stream_id = in_node->GetOpDescBarePtr()->GetStreamId();
      if (stream_id != kInvalidStream) {
        return false;
      }
    }
  }
  return true;
}

// if "fusion" attr is 1 or 2, need to optimize, if is 0 or -1, do not optimize
int64_t AllReduceParallelPass::GetFusion(const NodePtr &node) const {
  const auto &op_desc = node->GetOpDesc();
  if (op_desc != nullptr) {
    int64_t fusion = -1;
    if (AttrUtils::GetInt(op_desc, HCOM_ATTR_FUSION, fusion)) {
      return fusion;
    }
  }
  return -1;
}

Status AllReduceParallelPass::Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) {
  (void)graph;
  (void)subgraphs;
  if (!context.enable_hcom_parallel) {
    return NOT_CHANGED;
  }

  GE_STREAM_PASS_LOGI("[Run][AllReduceParallelPass] start");

  // All successors of HcomAllReduce.
  std::set<NodePtr> all_reduce_succs;

  for (const NodePtr &node : graph->GetDirectNode()) {
    if (!IsHcomNode(node->GetType()) || (GetFusion(node) <= 0)) {
      continue;
    }
    if (HasUserStreamLabel(node)) {
      continue;
    }

    std::string reduce_stream_label;
    GE_CHECK_NOTNULL(node->GetOpDesc());
    (void)AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, reduce_stream_label);

    std::set<NodePtr> cur_nodes = {node};
    while (!cur_nodes.empty()) {
      std::set<NodePtr> all_out_data_nodes;
      for (auto &curr_node : cur_nodes) {
        for (const NodePtr &out_node : curr_node->GetOutDataNodes()) {
          std::string out_stream_label;
          GE_CHECK_NOTNULL(out_node->GetOpDesc());
          (void)AttrUtils::GetStr(out_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, out_stream_label);
          // normally, Allreduce do not have streamLabel. when in horovod scenario Allreduce will have streamLabel
          bool isSuccessorParallel =
              (out_stream_label == reduce_stream_label) || (!reduce_stream_label.empty() && out_stream_label.empty());
          if (isSuccessorParallel) {
            all_reduce_succs.emplace(out_node);
            all_out_data_nodes.emplace(out_node);
          }
        }
      }
      cur_nodes = all_out_data_nodes;
    }
  }

  std::map<int64_t, int64_t> old_stream_to_new;
  for (const NodePtr &node : all_reduce_succs) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto old_stream = node->GetOpDesc()->GetStreamId();
    if (old_stream != kInvalidStream) {
      int64_t new_stream = kInvalidStream;
      std::map<int64_t, int64_t>::const_iterator iter = old_stream_to_new.find(old_stream);
      if (iter != old_stream_to_new.cend()) {
        new_stream = iter->second;
      } else {
        new_stream = context.next_stream;
        context.next_stream++;
        old_stream_to_new.emplace(old_stream, new_stream);
      }

      if (!IsHcomNode(node->GetType())) {
        GELOGI("Stream of node %s has been updated from %ld to %ld.", node->GetName().c_str(), old_stream, new_stream);
        node->GetOpDesc()->SetStreamId(new_stream);
      }
    }
  }

  return !all_reduce_succs.empty() ? SUCCESS : NOT_CHANGED;
}

bool AllReduceParallelPass::IsHcomNode(const std::string& node_type) const {
  return (node_type == HCOMALLREDUCE || node_type == HVDCALLBACKALLREDUCE);
}

LogicalStreamAllocator::LogicalStreamAllocator(const std::map<std::string, int32_t> &max_parallel_num)
  : max_parallel_num_(max_parallel_num) {
}

void LogicalStreamAllocator::EnableSingleStream(bool enable) { context_.enable_single_stream = enable; }

void LogicalStreamAllocator::EnableHcomParallel(bool enable) { context_.enable_hcom_parallel = enable; }

Status LogicalStreamAllocator::Assign(const ComputeGraphPtr &root_graph, const Graph2SubGraphInfoList &subgraph_map,
                                      int64_t &total_stream_num, int64_t &main_stream_num) {
  GE_CHECK_NOTNULL(root_graph);
  const auto engine_confs = StreamUtils::GetEngineConfs();

  Status status = DoAssign(root_graph, subgraph_map, engine_confs);
  if (status != SUCCESS) {
    GELOGE(status, "[Assign][Streams] failed, graph:%s.", root_graph->GetName().c_str());
    return status;
  }

  std::vector<ComputeGraphPtr> subgraphs = root_graph->GetAllSubgraphs();
  for (const ComputeGraphPtr &subgraph : subgraphs) {
    status = DoAssign(subgraph, subgraph_map, engine_confs);
    if (status != SUCCESS) {
      GELOGE(status, "[Assign][Streams] failed, graph:%s.", subgraph->GetName().c_str());
      return status;
    }
  }

  RefreshContinuousStreams(root_graph);
  if (!context_.enable_single_stream) {
    // stream id从0开始分配，next_stream为stream_num， (next_stream - 1)为当前图上最大stream id
    GE_ASSERT_SUCCESS(StreamUtils::RunCustomStreamPass(root_graph, context_.next_stream));
  }
  main_stream_num = context_.next_stream;
  // AssignAttachedStreamPass应该放在最后, 因为在分配attached从流之前，主流需要保证分配完成
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  GE_ASSERT_NOTNULL(assign_attached_stream_pass);
  // 对一个已经分过主流的节点，尝试进行从流的分配，即：一个节点可能产生多个流
  GE_ASSERT_SUCCESS(assign_attached_stream_pass->Run(root_graph, {}, context_));
  for (const auto &subgraph : subgraphs) {
    const auto &assign_attached_stream_pass_sub = MakeShared<AssignAttachedStreamPass>();
    GE_ASSERT_SUCCESS(assign_attached_stream_pass_sub->Run(subgraph, {}, context_));
  }
  total_stream_num = context_.next_stream;
  GE_ASSERT_TRUE(total_stream_num >= main_stream_num);
  const int64_t attached_stream_number = total_stream_num - main_stream_num;
  GELOGI("[Assign][LogicalStream] At last, total stream num: %ld, main stream num: %ld, attached stream num: %ld.",
         total_stream_num, main_stream_num, attached_stream_number);

  return SUCCESS;
}

Status LogicalStreamAllocator::DoAssign(const ComputeGraphPtr &graph, const Graph2SubGraphInfoList &subgraph_map,
                                        const std::map<std::string, EngineConfPtr> &engine_confs) {
  GE_CHECK_NOTNULL(graph);

  NodePtr parent_node = graph->GetParentNode();
  if ((parent_node == nullptr) || (parent_node->GetOpDesc() == nullptr) ||
      (parent_node->GetOwnerComputeGraph()->GetGraphUnknownFlag())) {
    context_.default_stream = kInvalidStream;
  } else {
    context_.default_stream = parent_node->GetOpDesc()->GetStreamId();
  }

  std::vector<SubgraphPtr> subgraphs;
  GE_TRACE_START(ConvertSubgraphs);
  Status status = StreamUtils::ConvertSubgraphs(graph, subgraph_map, engine_confs, max_parallel_num_, subgraphs);
  GE_COMPILE_TRACE_TIMESTAMP_END(ConvertSubgraphs, "GraphBuilder::AssignStreamConvertSubgraphs");
  if (status != SUCCESS) {
    GELOGE(status, "[Convert][SubGraphs] failed.");
    return status;
  }

  GELOGD("[Show][Subgraphs] in graph %s", graph->GetName().c_str());
  for (const auto &subgraph : subgraphs) {
    if (subgraph != nullptr) {
      GELOGD("subgraph: %s", subgraph->name.c_str());
    }
  }

  GE_ASSERT_SUCCESS(RunPasses(graph, subgraphs));
  GE_ASSERT_SUCCESS(RunOptimizeByTopoPasses(graph));
  return SUCCESS;
}

Status OptimizeIneffectiveMultiStreamPass::Run(const ComputeGraphPtr &graph) {
  char disable_flag_str[MMPA_MAX_PATH] = {"0"};
  mmGetEnv(kDisableIneffectiveMultiStreamOptimize, &disable_flag_str[0], static_cast<uint32_t>(MMPA_MAX_PATH));
  int32_t disable_flag;
  GE_ASSERT_SUCCESS(ge::ConvertToInt32(std::string(disable_flag_str), disable_flag));
  if (disable_flag == 1) {
    GELOGI("Disable optimize ineffective multi stream");
    return NOT_CHANGED;
  }
  std::map<int64_t, std::set<int64_t>> stream_id_to_node_ids;
  std::set<int64_t> stream_label_stream_ids;
  bool changed = false;
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto stream_id = op_desc->GetStreamId();
    if (stream_id == kInvalidStream) {
      continue;
    }
    if (StreamUtils::HasStreamLabelOrUserStreamLabel(node)) {
      stream_label_stream_ids.insert(stream_id);
    }
    stream_id_to_node_ids[stream_id].insert(op_desc->GetId());
  }
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto cur_stream_id = op_desc->GetStreamId();
    // 如下情况不根据图结构关系重新分流： 1. 通过StreamLabel指定分流 2. 带kNewStreamId属性(UpdateForMdeGroupPass) 3. 带ATTR_NAME_PARALLEL_GROUP属性(UpdateForParallelGroupPass)
    if (StreamUtils::HasStreamLabelOrUserStreamLabel(node) || op_desc->HasAttr(kNewStreamId) || op_desc->
        HasAttr(ATTR_NAME_PARALLEL_GROUP) || (cur_stream_id == kInvalidStream)) {
      continue;
    }
    const auto &in_nodes = node->GetInNodes();
    const auto &out_nodes = node->GetOutNodes();
    // 如果输入或者输出节点中存在节点与当前node是同一条流，那么这个node挪到别的流上也会与当前流之间插入event，并不会减少整体的event数量
    if (HasSameStreamId(in_nodes, cur_stream_id) || HasSameStreamId(out_nodes, cur_stream_id)) {
      continue;
    }
    std::map<int64_t, std::pair<ge::NodePtr, ge::NodePtr>> stream_id_to_io_nodes;
    GE_ASSERT_SUCCESS(CollectStreamIdToIoNodes(node, stream_id_to_io_nodes));
    for (const auto &iter : stream_id_to_io_nodes) {
      auto io_stream_id = iter.first;
      // 如果是stream_label分配的流，则不能将当前节点挪过去
      if (stream_label_stream_ids.find(io_stream_id) != stream_label_stream_ids.end()) {
        continue;
      }
      const auto &io_nodes = iter.second;
      auto node_ids_iter = stream_id_to_node_ids.find(io_stream_id);
      GE_ASSERT_TRUE(node_ids_iter != stream_id_to_node_ids.end(), "node %s io's stream id %ld cannot found",
                     op_desc->GetNamePtr(), io_stream_id);
      // 如果输入输出节点之间在这条流上没有别的节点，则可以挪过去
      if (!HasOtherNodeBetweenIOInThisStream(io_nodes, node_ids_iter->second)) {
        op_desc->SetStreamId(io_stream_id);
        GE_STREAM_PASS_LOGI("node %s optimize ineffective multi stream , set stream id from %ld to %ld",
                            op_desc->GetNamePtr(), cur_stream_id, io_stream_id);
        changed = true;
      }
    }
  }
  return changed ? SUCCESS : NOT_CHANGED;
}

Status LogicalStreamAllocator::RunPasses(const ComputeGraphPtr &graph, const std::vector<SubgraphPtr> &subgraphs) {
  std::vector<LogicalStreamPassPtr> passes;
  if (context_.enable_single_stream) {
    passes.emplace_back(MakeShared<SingleStreamPass>());
    passes.emplace_back(MakeShared<NodeStreamUpdatePass>());
    passes.emplace_back(MakeShared<UpdateForSkippedEnginePass>());
  } else {
    passes.emplace_back(MakeShared<UpdateForMdeGroupPass>());
    passes.emplace_back(MakeShared<AssignByLabelPass>());
    passes.emplace_back(MakeShared<IndependentStreamPass>());
    passes.emplace_back(MakeShared<AssignByDependencyPass>());
    passes.emplace_back(MakeShared<NodeStreamUpdatePass>());
    passes.emplace_back(MakeShared<UpdateForParallelGroupPass>());
    passes.emplace_back(MakeShared<AllReduceParallelPass>());
    passes.emplace_back(MakeShared<UpdateForSkippedEnginePass>());
  }

  for (auto &pass : passes) {
    GE_CHECK_NOTNULL(pass);

    Status status = pass->Run(graph, subgraphs, context_);
    if (status == SUCCESS) {
      GELOGI("[Show][Status]Stream pass %s return SUCCESS.", pass->GetName().c_str());
    } else if (status == NOT_CHANGED) {
      GELOGI("[Show][Status]Stream pass %s return NOT_CHANGED.", pass->GetName().c_str());
    } else {
      REPORT_INNER_ERR_MSG("E19999", "The %s of stream pass run failed.", pass->GetName().c_str());
      GELOGE(status, "[Call][Run] The %s of stream pass run failed.", pass->GetName().c_str());
      return status;
    }
  }

  return SUCCESS;
}

Status LogicalStreamAllocator::RunOptimizeByTopoPasses(const ComputeGraphPtr &graph) {
  if (context_.enable_single_stream) {
    return SUCCESS;
  }
  std::vector<OptimizeByTopoPassPtr> passes;
  passes.emplace_back(MakeShared<OptimizeIneffectiveMultiStreamPass>());

  for (auto &pass : passes) {
    GE_CHECK_NOTNULL(pass);

    Status status = pass->Run(graph);
    if (status == SUCCESS) {
      GELOGI("[Show][Status]Stream pass %s return SUCCESS.", pass->GetName().c_str());
    } else if (status == NOT_CHANGED) {
      GELOGI("[Show][Status]Stream pass %s return NOT_CHANGED.", pass->GetName().c_str());
    } else {
      REPORT_INNER_ERR_MSG("E19999", "The %s of stream pass run failed.", pass->GetName().c_str());
      GELOGE(status, "[Call][Run] The %s of stream pass run failed.", pass->GetName().c_str());
      return status;
    }
  }
  return SUCCESS;
}


void LogicalStreamAllocator::RefreshContinuousStreams(const ComputeGraphPtr &graph) {
  int64_t stream_num = context_.next_stream;
  std::vector<bool> stream_has_node(stream_num);

  for (const NodePtr &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    const auto &op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      int64_t stream_id = op_desc->GetStreamId();
      if ((stream_id != kInvalidStream) && (stream_id < stream_num)) {
        stream_has_node[stream_id] = true;
      }
    }
  }

  context_.next_stream = 0;
  std::vector<int64_t> old_to_new_streams(stream_num, kInvalidStream);
  for (size_t old_stream = 0; old_stream < stream_has_node.size(); old_stream++) {
    if (stream_has_node[old_stream]) {
      old_to_new_streams[old_stream] = context_.next_stream;
      context_.next_stream++;
    }
  }

  for (const NodePtr &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    const auto &op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      int64_t stream_id = op_desc->GetStreamId();
      if ((stream_id != kInvalidStream) && (stream_id < stream_num)) {
        op_desc->SetStreamId(old_to_new_streams[stream_id]);
      }
    }
  }
}

OptimizeByTopoPass::OptimizeByTopoPass(const std::string &name) : name_(name) {}

const std::string &OptimizeByTopoPass::GetName() const {
  return name_;
}
}  // namespace ge
