/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_allocator.h"
#include <algorithm>
#include <memory>
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "logical_stream_allocator.h"
#include "common/omg_util/omg_util.h"
#include "common/sgt_slice_type.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/build/task_generator_utils.h"
#include "api/gelib/gelib.h"
#include "assign_attached_notify_pass.h"
#include "assign_attached_event_pass.h"
#include "common/util.h"

namespace {
constexpr int64_t kTaskNumPerNormalNode = 3;
constexpr uint32_t kMaxSubgraphDepth = 10U;
constexpr uint32_t kMaxNotifyNum = 1024U;
constexpr size_t kEventMultiplexingItemCount = 3;
constexpr size_t kKeyWordIndex = 0;
constexpr size_t kNodeNameIndex = 1;
constexpr size_t kEventIdIndex = 2;
const char *const kSend = "SendTo";
const char *const kRecv = "RecvFrom";
constexpr char_t kNotify[] = "notify";
constexpr char_t kEvent[] = "event";
const char kDelim = ';';
const std::string ATTR_NAME_ATTACHED_STREAM_DEPEND_VALUE_LIST = "_attached_stream_depend_value_list";

inline std::string GetEventTypeStr(ge::EventType event_type) {
  if (event_type == ge::EventType::kNotify) {
    return kNotify;
  }
  return kEvent;
}

inline bool HasContinuousStreamLabel(const ge::OpDescPtr &op_desc, std::string &continuous_stream_label) {
  if (ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CONTINUOUS_STREAM_LABEL, continuous_stream_label)) {
    GELOGD("node[%s] get continuous_stream_label %s", op_desc->GetName().c_str(), continuous_stream_label.c_str());
    return true;
  }
  return false;
}

std::string PrintStreamIdToTaskSize(const std::map<int64_t, size_t> &stream_id_to_task_size) {
  std::string log;
  for (const auto &iter : stream_id_to_task_size) {
    log += "stream id: " + std::to_string(iter.first) + " task size: " + std::to_string(iter.second) + ", ";
  }
  return log;
}

ge::Status ParseNodeEventMultiplexing(
    const ge::NodePtr &node, const std::vector<std::string> &raw_event_multiplexing,
    std::unordered_map<ge::NodePtr, std::vector<std::pair<std::string, uint32_t>>> &node_to_send,
    std::unordered_map<ge::NodePtr, std::vector<std::pair<std::string, uint32_t>>> &node_to_recv) {
  GE_CHECK_NOTNULL(node);
  for (const auto &str : raw_event_multiplexing) {
    std::vector<std::string> ele = ge::StringUtils::Split(str, kDelim);
    if (ele.size() != kEventMultiplexingItemCount) {
      GELOGE(ge::PARAM_INVALID, "[Check][RawMultiplexing]Size error, node:%s, require size:%zu, actually:%zu.",
             node->GetName().c_str(), kEventMultiplexingItemCount, ele.size());
      REPORT_INNER_ERR_MSG("E19999", "Raw event multiplexing is invalid, node:%s, require size:%zu, actually:%zu.",
                         node->GetName().c_str(), kEventMultiplexingItemCount, ele.size());
      return ge::PARAM_INVALID;
    }
    int32_t value;
    try {
      value = std::stoi(ele[kEventIdIndex]);
    } catch (std::invalid_argument &) {
      GELOGE(ge::PARAM_INVALID, "[Throw][Exception]Event id is invalid, node:%s, raw:%s.",
             node->GetName().c_str(), ele[kEventIdIndex].c_str());
      REPORT_INNER_ERR_MSG("E19999", "Event id is invalid, node:%s, raw:%s.",
                         node->GetName().c_str(), ele[kEventIdIndex].c_str());
      return ge::PARAM_INVALID;
    } catch (std::out_of_range &) {
      GELOGE(ge::PARAM_INVALID, "[Throw][Exception]Event id is out of range, node:%s, raw:%s.",
             node->GetName().c_str(), ele[kEventIdIndex].c_str());
      REPORT_INNER_ERR_MSG("E19999", "Event id is out of range, node:%s, raw:%s.",
                         node->GetName().c_str(), ele[kEventIdIndex].c_str());
      return ge::PARAM_INVALID;
    }
    if (value < 0) {
      GELOGE(ge::PARAM_INVALID, "[Check][EventId]Event id is out of range, node:%s, raw:%s, value:%d.",
             node->GetName().c_str(), ele[kEventIdIndex].c_str(), value);
      REPORT_INNER_ERR_MSG("E19999", "Event id is out of range, node:%s, raw:%s, value:%d.",
                         node->GetName().c_str(), ele[kEventIdIndex].c_str(), value);
      return ge::PARAM_INVALID;
    }
    if (ele[kKeyWordIndex] == kSend) {
      node_to_send[node].emplace_back(std::make_pair(ele[kNodeNameIndex], static_cast<uint32_t>(value)));
    } else if (ele[kKeyWordIndex] == kRecv) {
      node_to_recv[node].emplace_back(std::make_pair(ele[kNodeNameIndex], static_cast<uint32_t>(value)));
    } else {
      GELOGE(ge::PARAM_INVALID, "[Check][KeyWord]Key word is not supported, node:%s, key:%s.",
             node->GetName().c_str(), ele[kEventIdIndex].c_str());
      REPORT_INNER_ERR_MSG("E19999", "Key word is not supported, node:%s, key:%s.",
                         node->GetName().c_str(), ele[kEventIdIndex].c_str());
      return ge::PARAM_INVALID;
    }
  }
  return ge::SUCCESS;
}

ge::Status ParseAllNodeEventMultiplexing(
    const ge::ComputeGraphPtr &graph, std::unordered_map<std::string, ge::NodePtr> &name_to_node_map,
    std::unordered_map<ge::NodePtr, std::vector<std::pair<std::string, uint32_t>>> &node_to_send,
    std::unordered_map<ge::NodePtr, std::vector<std::pair<std::string, uint32_t>>> &node_to_recv) {
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    name_to_node_map.insert({node->GetName(), node});
    std::vector<std::string> raw_event_multiplexing;
    if (!(op_desc->HasAttr(ge::ATTR_NAME_EVENT_MULTIPLEXING))) {
      continue;
    }
    bool get_attr = ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_EVENT_MULTIPLEXING, raw_event_multiplexing);
    if (!get_attr) {
      GELOGE(ge::PARAM_INVALID, "[Get][Attr]Node:%s.", node->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Failed to get raw event multiplexing, node:%s.", node->GetName().c_str());
      return ge::PARAM_INVALID;
    }
    auto parse_ret = ParseNodeEventMultiplexing(node, raw_event_multiplexing, node_to_send, node_to_recv);
    if (parse_ret != ge::SUCCESS) {
      GELOGE(parse_ret, "[Parse][Eventmultiplexing]Node:%s.", node->GetName().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Failed to parse node event multiplexing, node:%s.", node->GetName().c_str());
      return parse_ret;
    }
  }
  return ge::SUCCESS;
}

std::vector<uint32_t> GetIntersection(std::vector<uint32_t> &a, std::vector<uint32_t> &b) {
  std::unordered_set<uint32_t> ele_of_a(a.begin(), a.end());
  std::vector<uint32_t> res;
  for (auto &ele : b) {
    if (ele_of_a.count(ele) > 0) {
      res.emplace_back(ele);
    }
  }
  return res;
}

std::string PrintAttachedStreamId(const std::vector<int64_t> &attached_streams) {
  std::stringstream s;
  for (const auto stream_id : attached_streams) {
    s << (std::to_string(stream_id) + " ");
  }
  return s.str();
}

ge::Status GetLastExecRefNodeFromInput(const ge::NodePtr &target_node, const ge::NodePtr &input_node,
                                       const ge::OutDataAnchorPtr &out_anchor_from_input, ge::NodePtr &out_ref_node) {
  if (!ge::OpTypeUtils::IsVarLikeNode(input_node->GetType())) {
    return ge::SUCCESS;
  }
  const auto target_op_desc = target_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(target_op_desc);
  const auto target_node_id = target_op_desc->GetId();
  int64_t last_node_id = -1;
  for (const auto &in_anchor : out_anchor_from_input->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(in_anchor);
    auto owner_node = in_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(owner_node);
    const auto op_desc = owner_node->GetOpDescBarePtr();
    GE_CHECK_NOTNULL(op_desc);
    const auto node_id = op_desc->GetId();
    if ((node_id > target_node_id) || (last_node_id > node_id)) {
      continue;
    }

    bool is_ref = false;
    (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_REFERENCE, is_ref);
    if (!is_ref) {
      continue;
    }
    const std::string &input_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(in_anchor->GetIdx()));
    for (const auto &output_info : op_desc->GetAllOutputName()) {
      if ((!output_info.first.empty()) && (output_info.first == input_name)) {
        out_ref_node = owner_node;
        last_node_id = node_id;
        break;
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status GetDependNodesWithStreamAcitveInSubgraph(ge::NodePtr &depend_node) {
  auto graph = depend_node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  ge::NodePtr stream_active_node;
  for (const auto &node : graph->GetDirectNode()) {
    if (strcmp(node->GetTypePtr(), ge::STREAMACTIVE) != 0) {
      continue;
    }
    bool is_subgraph_first_active = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ge::ATTR_NAME_SUBGRAPH_FIRST_ACTIVE,
                                 is_subgraph_first_active);
    if (is_subgraph_first_active) {
      stream_active_node = node;
      break;
    }
  }
  if (stream_active_node == nullptr) {
    return ge::SUCCESS;
  }
  if (depend_node->GetOpDescBarePtr()->GetId() < stream_active_node->GetOpDescBarePtr()->GetId()) {
    depend_node = stream_active_node;
  }
  return ge::SUCCESS;
}

std::vector<int64_t> ConvertValueStr2List(const std::string &value_list_str) {
  std::vector<std::string> index_list = ge::StringUtils::Split(value_list_str, ',');
  std::vector<int64_t> value_list(index_list.size());
  (void)std::transform(index_list.begin(), index_list.end(), value_list.begin(), [](const std::string &str) {
    int64_t index = std::numeric_limits<int64_t>::max();
    (void) ge::ConvertToInt64(str, index);
    return index;
  });
  return value_list;
}

ge::Status GetDependNodesByValueList(const ge::NodePtr &cur_node, const std::vector<int64_t> &index_list,
                                     std::unordered_set<ge::NodePtr> &depend_nodes) {
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetIrInputRawDescRange(cur_node->GetOpDesc(), ir_input_2_range));
  for (const auto &index : index_list) {
    auto iter = ir_input_2_range.find(static_cast<size_t>(index));
    GE_ASSERT(iter != ir_input_2_range.end());
    GELOGI("node name[%s], type[%s], ir index[%ld], instance index[%zu][%zu]",
            cur_node->GetNamePtr(), cur_node->GetTypePtr(), index, iter->second.first, iter->second.second);
    for (size_t i = iter->second.first; i < iter->second.first + iter->second.second; ++i) {
      const auto &in_data_anchor = cur_node->GetInDataAnchor(i);
      GE_ASSERT_NOTNULL(in_data_anchor);
      const auto &peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_data_anchor);
      auto in_node = peer_out_data_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(in_node);
      ge::NodePtr ref_node = nullptr;
      GE_ASSERT_GRAPH_SUCCESS(GetLastExecRefNodeFromInput(cur_node, in_node, peer_out_data_anchor, ref_node));
      ge::NodePtr depend_node = (ref_node != nullptr) ? ref_node : in_node;
      GE_ASSERT_GRAPH_SUCCESS(GetDependNodesWithStreamAcitveInSubgraph(depend_node));
      (void)depend_nodes.insert(depend_node);
    }
  }
  return ge::SUCCESS;
}
// 兼容处理，引擎不设置sqe_num则表示只有1个任务
size_t GetTaskSqeNum(const domi::TaskDef &task) {
  size_t sqe_num = static_cast<size_t>(task.sqe_num());
  return sqe_num == 0U ? 1U : sqe_num;
}

void CollectStreamIdToTaskSize(const std::vector<domi::TaskDef> &task_defs,
                               std::map<int64_t, size_t> &stream_id_to_task_size) {
  for (const auto &task : task_defs) {
    int64_t stream_id = static_cast<int64_t>(task.stream_id());
    auto task_sqe_num = GetTaskSqeNum(task);
    stream_id_to_task_size[stream_id] += task_sqe_num;
    // event_wait在Distribute时下了wait和reset两个任务
    auto task_type = static_cast<ge::ModelTaskType>(task.type());
    if (task_type == ge::ModelTaskType::MODEL_TASK_EVENT_WAIT) {
      stream_id_to_task_size[stream_id]++;
    }
  }
}
}  // namespace

namespace ge {
namespace {
constexpr int64_t kDfxTaskNum = 7; // profiling 7点轨迹法
constexpr int64_t kReservedTaskNum = kDfxTaskNum + 20;

void CollectSubgraphStreams(const OpDescPtr &op_desc, std::set<int64_t> &subgraph_streams) {
  // 外部保证op_desc不为空
  std::vector<int64_t> all_streams;
  all_streams.emplace_back(op_desc->GetStreamId());
  const auto attached_streams = op_desc->GetAttachedStreamIds();
  all_streams.insert(all_streams.end(), attached_streams.begin(), attached_streams.end());
  for (auto stream_id : all_streams) {
    if (stream_id != kInvalidStream) {
      subgraph_streams.emplace(stream_id);
      GELOGI("Get valid stream %ld from node %s %s", stream_id, op_desc->GetNamePtr(), op_desc->GetTypePtr());
    }
  }
}
bool IsBelongToSameGraph(const NodePtr &node1, const NodePtr &node2) {
  return node1->GetOwnerComputeGraphBarePtr() == node2->GetOwnerComputeGraphBarePtr();
}
}  // namespace
StreamAllocator::StreamAllocator(ComputeGraphPtr whole_graph, const Graph2SubGraphInfoList &subgraphs)
    : whole_graph_(std::move(whole_graph)), subgraphs_(subgraphs) {
  enable_single_stream_ = StreamUtils::EnableSingleStream();
}

void StreamAllocator::BuildEventReuseMap(const EventType event_type, const std::vector<uint32_t> &events,
                                         std::map<uint32_t, uint32_t> &event_seen, uint32_t &event_id) const {
  for (size_t i = 0U; i < events.size(); ++i) {
    std::map<uint32_t, uint32_t>::const_iterator iter = event_seen.find(events[i]);
    if (iter == event_seen.cend()) {
      event_seen.emplace(events[i], event_id);
      GELOGI("Refresh %s id from %u to %u.", GetEventTypeStr(event_type).c_str(), events[i], event_id);
      ++event_id;
    }
  }
}

Status StreamAllocator::AssignLogicalStreams(const std::map<std::string, int32_t> &max_parallel_num,
                                             bool hcom_parallel) {
  GE_CHECK_NOTNULL(whole_graph_);
  auto gelib = GELib::GetInstance();
  if (gelib == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Check GELib instance nullptr, graph:%s", whole_graph_->GetName().c_str());
    GELOGE(FAILED, "[Get][Instance] of GELib failed. graph:%s", whole_graph_->GetName().c_str());
    return FAILED;
  }

  LogicalStreamAllocator logical_allocator(max_parallel_num);
  logical_allocator.EnableSingleStream(enable_single_stream_);
  logical_allocator.EnableHcomParallel(hcom_parallel);

  Status status = logical_allocator.Assign(whole_graph_, subgraphs_, stream_num_, main_stream_num_);
  if (status != SUCCESS) {
    GELOGE(status, "[Assign][LogicalStreams] failed. graph:%s", whole_graph_->GetName().c_str());
    return status;
  }
  return SUCCESS;
}

Status StreamAllocator::PreProcessOfInsertSyncNodes() {
  auto status = SetActiveStreamsByLabel();
  if (status != SUCCESS) {
    GELOGE(status, "[Set][ActiveStreams] By Label failed! graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  status = SetActiveStreamsForSubgraphs();
  if (status != SUCCESS) {
    GELOGE(status, "[Set][ActiveStreams] For Subgraphs failed. graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  return SUCCESS;
}

// 模型的输出执行的时候，应该保证模型中所有的流都已经执行完成，一般来说图中的节点本身都会跟模型输出节点产生直接或者间接的连边关系;
// 但是当一个节点有多个流的时候，需要处理一下附属流的同步
Status StreamAllocator::CoverAllStreamByNetoutput() {
  GE_ASSERT_NOTNULL(whole_graph_);
  std::set<int64_t> attached_stream;
  std::map<int64_t, NodePtr> attached_stream_id_to_last_node;
  for (const auto &cur_node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    const auto op_desc = cur_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->HasValidAttachedStreamId()) {
      for (auto attached_stream_id : op_desc->GetAttachedStreamIds()) {
        attached_stream.emplace(attached_stream_id);
        attached_stream_id_to_last_node[attached_stream_id] = cur_node;
      }
    }
  }
  for (const int64_t stream : attached_stream) {
    const auto iter = attached_stream_id_to_last_node.find(stream);
    GE_ASSERT_TRUE(iter != attached_stream_id_to_last_node.end());
    const auto &last_node_of_this_stream = iter->second;
    GE_ASSERT_NOTNULL(last_node_of_this_stream);
    const auto &this_graph = last_node_of_this_stream->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(this_graph);
    // 附属流最后一个节点跟普通流上的当前图输出节点进行event同步
    const auto &output_node = this_graph->GetOrUpdateNetOutputNode();
    if (output_node == nullptr) {
      GELOGW("Graph %s has no netoutput, just return.", this_graph->GetName().c_str());
      continue;
    }
    GE_ASSERT_SUCCESS(AddEventPairBetweenAttachedAndMain(last_node_of_this_stream, output_node, stream,
                                                         attached_node_to_stream_id_to_send_event_id_,
                                                         node_to_recv_events_));
  }
  return SUCCESS;
}

Status StreamAllocator::AssignAttachedNotifyResource() {
  AssignAttachedNotifyPass assign_attached_notify_pass;
  const uint32_t normal_notify_num = notify_num_;
  GE_ASSERT_SUCCESS(assign_attached_notify_pass.Run(whole_graph_, notify_num_, notify_types_));
  std::vector<ComputeGraphPtr> subgraphs = whole_graph_->GetAllSubgraphs();
  for (const auto &subgraph : subgraphs) {
    AssignAttachedNotifyPass assign_attached_notify_pass_sub;
    GE_ASSERT_SUCCESS(assign_attached_notify_pass_sub.Run(subgraph, notify_num_, notify_types_));
  }
  GE_ASSERT_TRUE(notify_num_ >= normal_notify_num);
  GELOGI("At last, total notify num: %u, normal notify num: %u, attached notify num: %u.", notify_num_,
         normal_notify_num, notify_num_ - normal_notify_num);
  return SUCCESS;
}

Status StreamAllocator::AssignAttachedEventResource() {
  AssignAttachedEventPass assign_attached_event_pass;
  const uint32_t normal_event_num = event_num_;
  GE_ASSERT_SUCCESS(assign_attached_event_pass.Run(whole_graph_, event_num_));
  std::vector<ComputeGraphPtr> subgraphs = whole_graph_->GetAllSubgraphs();
  for (const auto &subgraph : subgraphs) {
    AssignAttachedEventPass assign_attached_event_pass_sub;
    GE_ASSERT_SUCCESS(assign_attached_event_pass_sub.Run(subgraph, event_num_));
  }
  GE_ASSERT_TRUE(event_num_ >= normal_event_num);
  GELOGI("At last, total event num: %u, normal event num: %u, attached event num: %u.", event_num_,
         normal_event_num, event_num_ - normal_event_num);
  return SUCCESS;
}

Status StreamAllocator::PostProcessOfSplitStreams() {
  DumpEvents(EventType::kEvent, node_to_send_events_, node_to_recv_events_);
  DumpEvents(EventType::kNotify, node_to_send_notifies_, node_to_recv_notifies_);

  for (const NodePtr &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto stream_id = node->GetOpDesc()->GetStreamId();
    if (stream_id == kInvalidStream) {
      node->GetOpDesc()->SetStreamId(0);
    }
  }

  if (stream_num_ == 0) {
    GELOGI("None of nodes need to assign stream, stream num is 0, it will cause problem, so change it to 1");
    stream_num_ = 1;
  }

  GE_ASSERT_TRUE(notify_num_ <= kMaxNotifyNum,
                 "notify_num:%u is bigger than kMaxNotifyNum:%u.",
                 notify_num_,
                 kMaxNotifyNum);

  return SUCCESS;
}

Status StreamAllocator::InserSyncNodesWithoutNotify() {
  GELOGI("InserSyncNodesWithoutNotify.");
  Status status = InsertSyncEvents(EventType::kEvent);
  if (status != SUCCESS) {
    GELOGE(status, "[Insert][SyncEventId] failed! graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  status = OptimizeSyncEvents(EventType::kEvent, node_to_send_events_, node_to_recv_events_);
  if (status != SUCCESS) {
    GELOGE(status, "[Optimize][SyncEventId] failed! graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  return SUCCESS;
}

Status StreamAllocator::InsertSyncNodesWithNotify() {
  GELOGI("InsertSyncNodesWithNotify.");
  Status status = InsertSyncEvents(EventType::kNotify);
  if (status != SUCCESS) {
    GELOGE(status, "[Insert][NotifyId] failed! graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  status = OptimizeSyncEvents(EventType::kNotify, node_to_send_notifies_, node_to_recv_notifies_);
  if (status != SUCCESS) {
    GELOGE(status, "[Optimize][SyncNotifyId] failed! graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  return SUCCESS;
}

Status StreamAllocator::InsertSyncEventsWithAttachedStream(const EventType insert_event_type) {
  auto ffts_filter = [](const Node &node, const char *, const ComputeGraphPtr &) {
    return !(node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) ||
             node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH));
  };

  for (const auto &cur_node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag(), nullptr, ffts_filter)) {
    if (!(AttrUtils::HasAttr(cur_node->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO) ||
          AttrUtils::HasAttr(cur_node->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO_LIST))) {
      continue;
    }
    std::vector<NamedAttrs> stream_info_attrs_list;
    std::unordered_set<NodePtr> depend_nodes;
    if (AttrUtils::GetListNamedAttrs(cur_node->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                     stream_info_attrs_list)) {
      for (const auto &attr : stream_info_attrs_list) {
        std::vector<int64_t> value_list;
        if (!AttrUtils::GetListInt(attr, ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT, value_list)) {
          continue;
        }
        GELOGI("get attr[%s], value_list[%s], name[%s], type[%s]",
               ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT.c_str(), ToString(value_list).c_str(),
               cur_node->GetNamePtr(), cur_node->GetTypePtr());
        GE_ASSERT_SUCCESS(GetDependNodesByValueList(cur_node, value_list, depend_nodes));
      }
    } else {
      (void)AttrUtils::GetListNamedAttrs(cur_node->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, stream_info_attrs_list);
      for (const auto &attr : stream_info_attrs_list) {
        // todo:此处可以优化为ListInt，需要和算子配合整改
        std::string value_list;
        if (!AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_STREAM_DEPEND_VALUE_LIST, value_list)) {
          continue;
        }
        GELOGI("get attr[%s], valuelist[%s], name[%s], type[%s]", ATTR_NAME_ATTACHED_STREAM_DEPEND_VALUE_LIST.c_str(),
               value_list.c_str(), cur_node->GetNamePtr(), cur_node->GetTypePtr());
        GE_ASSERT_SUCCESS(GetDependNodesByValueList(cur_node, ConvertValueStr2List(value_list), depend_nodes));
      }
    }
    for (auto &depend_node : depend_nodes) {
      GELOGD("insert event between node[%s][%s] to target node[%s][%s]",
              depend_node->GetNamePtr(), depend_node->GetTypePtr(), cur_node->GetNamePtr(), cur_node->GetTypePtr());
      GE_ASSERT_SUCCESS(InsertOneEventInTwoNodesWithAttachedStream(insert_event_type, depend_node, cur_node));
    }
    // ATTR_NAME_ATTACHED_STREAM_INFO属性的生命周期到这里应该就结束了，为了减少OM体积，删除他
    // 新的方案下ATTR_NAME_ATTACHED_STREAM_INFO_LIST属性生命週期并未结束，FE后续需要
    (void)cur_node->GetOpDesc()->DelAttr(ATTR_NAME_ATTACHED_STREAM_INFO);
  }
  return SUCCESS;
}

Status StreamAllocator::InsertOneEventInTwoNodesWithAttachedStream(const EventType insert_event_type,
                                                                   const NodePtr &src_node,
                                                                   const NodePtr &dst_node) {
  if (insert_event_type != EventType::kEvent) { // 当前不支持非event
    return SUCCESS;
  }
  const auto src_desc = src_node->GetOpDesc();
  GE_CHECK_NOTNULL(src_desc);
  if ((src_desc->GetType() == DATA) || OpTypeUtils::IsConstNode(src_desc->GetType())) {
    return SUCCESS;
  }
  const auto dst_desc = dst_node->GetOpDesc();
  GE_CHECK_NOTNULL(dst_desc);

  // tips: multi_attached_stream control order in self, so only one attached stream need add event by GE
  const auto src_attached_stream_ids = src_desc->GetAttachedStreamIds();
  const auto dst_attached_stream_ids = dst_desc->GetAttachedStreamIds();
  GE_ASSERT_TRUE(src_attached_stream_ids.size() <= 1U, "Src_node [%s %s] only support <= 1 attached stream but got %zu",
                 src_desc->GetName().c_str(), src_desc->GetType().c_str(), src_attached_stream_ids.size());
  GE_ASSERT_TRUE(dst_attached_stream_ids.size() <= 1U, "Dst_node [%s %s] only support <= 1 attached stream but got %zu",
                 dst_desc->GetName().c_str(), dst_desc->GetType().c_str(), dst_attached_stream_ids.size());
  const auto src_attached_stream_id =
      src_attached_stream_ids.size() == 1U ? src_attached_stream_ids[0U] : kInvalidStream;
  const auto dst_attached_stream_id =
      dst_attached_stream_ids.size() == 1U ? dst_attached_stream_ids[0U] : kInvalidStream;
  GELOGI("src_node:%s, dst_node:%s, src_attached_stream_id:%d, dst_attached_stream_id:%d, insert_event_type:%d",
          src_node->GetName().c_str(), dst_node->GetName().c_str(), src_attached_stream_id, dst_attached_stream_id,
          insert_event_type);
  if (src_attached_stream_id == dst_attached_stream_id) {
    return SUCCESS;
  }
  if (src_attached_stream_id == kInvalidStream && dst_attached_stream_id != kInvalidStream) {
    // Add send and receive events.
    GE_ASSERT_SUCCESS(AddEventPair(src_node, dst_node, node_to_send_events_, attached_node_to_recv_events_));
    GELOGI("Insert event %u between node %s(stream %ld) and %s(stream %ld)", event_num_, src_node->GetName().c_str(),
            src_desc->GetStreamId(), dst_node->GetName().c_str(), dst_attached_stream_id);
  } else if (src_attached_stream_id != kInvalidStream && dst_attached_stream_id != kInvalidStream) {
    GE_ASSERT_SUCCESS(AddEventPair(src_node, dst_node, attached_node_to_send_events_, attached_node_to_recv_events_));
    GELOGI("Insert event %u between node %s(stream %ld) and %s(stream %ld)", event_num_, src_node->GetName().c_str(),
            src_attached_stream_id, dst_node->GetName().c_str(), dst_attached_stream_id);
  } else {
    GE_ASSERT_SUCCESS(AddEventPair(src_node, dst_node, attached_node_to_send_events_, node_to_recv_events_));
    GELOGI("Insert event %u between node %s(stream %ld) and %s(stream %ld)", event_num_, src_node->GetName().c_str(),
            src_attached_stream_id, dst_node->GetName().c_str(), dst_desc->GetStreamId());
  }
  return SUCCESS;
}

Status StreamAllocator::UpdateStreamSwitchByLogicStream() {
  for (auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    if (node->GetType() == STREAMSWITCH) {
      GE_ASSERT_SUCCESS(UpdateActiveStreamsForSwitchNode(node), "[Update][ActiveStreams] for switch node: %s failed.",
                        node->GetName().c_str());
    }
  }

  GE_ASSERT_SUCCESS(SetActiveStreamsForLoop(), "[Set][ActiveStreams] For Loop failed! graph:%s",
                    whole_graph_->GetName().c_str());
  return SUCCESS;
}

Status StreamAllocator::SetLogicStreamIdAttr() {
  for (const ge::NodePtr &node : whole_graph_->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const int64_t stream_id = op_desc->GetStreamId();
    if (stream_id != ge::kInvalidStream) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "_logic_stream_id", stream_id));
      GELOGI("Op [%s] OpType [%s] logic stream id is %lld, ", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             stream_id);
      split_stream_id_to_logic_stream_id_[stream_id] = stream_id;
    }
    const auto attached_stream_ids = op_desc->GetAttachedStreamIds();
    for (const auto attached_stream_id : attached_stream_ids) {
      split_stream_id_to_logic_stream_id_[attached_stream_id] = attached_stream_id;
    }
    auto attached_stream_num = attached_stream_ids.size();
    if (attached_stream_num == 1U && (attached_stream_ids[0U] != ge::kInvalidStream)) {
      // 兼容处理，老的om在加载时会根据这个属性获取原始逻辑stream id
      GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc, "_logic_attached_stream_id", attached_stream_ids[0U]));
      GELOGI("Op [%s] OpType [%s] logic attached stream id is %lld", op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), attached_stream_ids[0U]);
    } else if (attached_stream_num > 1U) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(op_desc, "_logic_attached_stream_ids", attached_stream_ids));
      GELOGI("Op [%s] OpType [%s] logic attached stream id is %s", op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), PrintAttachedStreamId(attached_stream_ids).c_str());
    }
  }
  return SUCCESS;
}

// After allocating the logical stream in the graph, insert the synchronization node.
Status StreamAllocator::InsertSyncNodesByLogicStream(int64_t &stream_num, int64_t &event_num, int64_t &notify_num) {
  GE_ASSERT_NOTNULL(whole_graph_);
  GE_ASSERT_SUCCESS(SetLogicStreamIdAttr());

  GE_ASSERT_SUCCESS(SetNewTopoId());
  GE_ASSERT_SUCCESS(PreProcessOfInsertSyncNodes());
  std::string event_optimizer_option;
  (void)GetContext().GetOption(EVENT, event_optimizer_option);
  if ((!event_optimizer_option.empty()) && (event_optimizer_option == kNotify)) {
    event_type_ = ge::EventType::kNotify;
    GE_ASSERT_SUCCESS(InsertSyncNodesWithNotify());
  } else {
    GE_ASSERT_SUCCESS(InserSyncNodesWithoutNotify());
  }

  GE_ASSERT_SUCCESS(RefreshContinuousNotifies(), "[Refresh][ContinuousNotifies] failed! graph:%s",
                    whole_graph_->GetName().c_str());
  event_num_ = 0U;
  GE_ASSERT_SUCCESS(StreamUtils::RefreshContinuousEvents(event_num_, node_to_send_events_, node_to_recv_events_),
                    "[Refresh][ContinuousEvents] failed! graph:%s", whole_graph_->GetName().c_str());
  GE_ASSERT_SUCCESS(
      StreamUtils::RefreshContinuousEvents(event_num_, attached_node_to_send_events_, attached_node_to_recv_events_),
      "[Refresh][ContinuousEvents] attached stream failed! graph:%s", whole_graph_->GetName().c_str());
  GE_ASSERT_SUCCESS(RefreshEventsAndNotifiesWithReuse(), "[Refresh][EventsAndNotifies] With Reuse failed! graph:%s",
                    whole_graph_->GetName().c_str());
  GE_ASSERT_SUCCESS(InsertSyncEventsWithAttachedStream(EventType::kEvent),
                    "[InsertSyncEventsWithAttachedStream] failed! graph:%s", whole_graph_->GetName().c_str());

  GE_ASSERT_SUCCESS(UpdateStreamSwitchByLogicStream(), "UpdateStreamSwitchByLogicStream failed! graph:%s",
                    whole_graph_->GetName().c_str());
  GE_ASSERT_SUCCESS(CoverAllStreamByNetoutput());
  GE_ASSERT_SUCCESS(GenerateSyncEventNodes(), "[GenerateSyncEventNodes] failed! graph:%s",
                    whole_graph_->GetName().c_str());
  notify_types_.resize(notify_num_, RT_NOTIFY_DEFAULT);
  GE_ASSERT_SUCCESS(AssignAttachedNotifyResource());
  GE_ASSERT_SUCCESS(AssignAttachedEventResource());

  stream_num = stream_num_;
  event_num = static_cast<int64_t>(event_num_);
  notify_num = static_cast<int64_t>(notify_num_);

  GELOGI("After InsertSyncNodesByLogicStream, graph:%s, stream num:%ld, notify num:%u, event num:%u.",
         whole_graph_->GetName().c_str(), stream_num_, notify_num_, event_num_);
  return SUCCESS;
}

// 流拆分产生的新的节点的id要主动设置为图上最后的id，因为流拆分前已经GenTask了，对于已经GenTask的节点不能改变id，
// task def里打上了op_index
Status StreamAllocator::SplitStreamAndRefreshTaskDef(
    std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks, int64_t &stream_num,
    int64_t &event_num, int64_t &notify_num) {
  GE_ASSERT_SUCCESS(AssignSingleStream(node_id_2_node_tasks), "[Assign][SingleStream] failed! graph:%s",
                    whole_graph_->GetName().c_str());

  ClearNodes2SyncEvents();
  GE_ASSERT_SUCCESS(SetNewTopoId());
  std::vector<std::set<int64_t>> split_streams(stream_num_);
  GE_ASSERT_SUCCESS(SplitStreams(node_id_2_node_tasks, split_streams), "[Split][Streams] failed! graph:%s",
                    whole_graph_->GetName().c_str());

  GE_ASSERT_SUCCESS(UpdateActiveStreams(split_streams), "[Update][ActiveStreams] failed! graph:%s",
                    whole_graph_->GetName().c_str());

  // GenTask后不能改变原来节点的topo id，因为task def里打上了op_index
  GE_ASSERT_SUCCESS(GenerateSyncEventNodes(false), "[GenerateSyncEventNodes] failed! graph:%s", whole_graph_->GetName().c_str());
  GE_ASSERT_SUCCESS(PostProcessOfSplitStreams());
  stream_num = stream_num_;
  event_num = static_cast<int64_t>(event_num_);
  notify_num = static_cast<int64_t>(notify_num_);
  GELOGI("After SplitStreamAndRefreshTaskDef, graph:%s, stream num:%ld, notify num:%u, event num:%u.",
         whole_graph_->GetName().c_str(), stream_num_, notify_num_, event_num_);
  return SUCCESS;
}

Status StreamAllocator::SetNewTopoId() {
  const auto &all_nodes = whole_graph_->GetAllNodesPtr();
  GE_ASSERT_TRUE(!all_nodes.empty());
  const auto last_node = all_nodes.back();
  GE_ASSERT_NOTNULL(last_node);
  auto op_desc = last_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  new_topo_id_ = op_desc->GetId() + 1;
  return SUCCESS;
}

void StreamAllocator::ClearNodes2SyncEvents() {
  node_to_send_events_.clear();
  node_to_recv_events_.clear();
  attached_node_to_send_events_.clear();
  attached_node_to_recv_events_.clear();
  attached_node_to_stream_id_to_send_event_id_.clear();
  attached_node_to_stream_id_to_recv_event_id_.clear();
  node_to_send_notifies_.clear();
  node_to_recv_notifies_.clear();
}

Status StreamAllocator::AssignSingleStream(
    const std::unordered_map<int64_t , std::vector<domi::TaskDef>> &node_id_2_node_tasks) {
  if (!enable_single_stream_) {
    return SUCCESS;
  }

  if (main_stream_num_ > 1) {
    REPORT_INNER_ERR_MSG("E19999", "The number of ts streams is %ld, only one is supported",
                       main_stream_num_);
    GELOGE(FAILED, "[Check][Param] The number of ts streams is %ld, only one is supported.", main_stream_num_);
    return FAILED;
  }

  int64_t task_count = 0;
  for (const NodePtr &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    const auto &iter = node_id_2_node_tasks.find(node->GetOpDesc()->GetId());
    if (iter == node_id_2_node_tasks.end()) {
      continue;
    }
    std::map<int64_t, size_t> stream_id_to_task_size;
    stream_id_to_task_size[node->GetOpDesc()->GetStreamId()] = 0U;
    CollectStreamIdToTaskSize(iter->second, stream_id_to_task_size);
    size_t total_task_size = 0U;
    for (const auto &pair : stream_id_to_task_size) {
      total_task_size += pair.second;
    }
    AddTaskNum(node, task_count, total_task_size, false);
  }
  task_count += kReservedTaskNum;

  uint32_t max_normal_stream_count = 0;
  uint32_t max_normal_task_count = 0;
  Status status = GetMaxStreamAndTask(false, max_normal_stream_count, max_normal_task_count);
  if (status != SUCCESS) {
    GELOGE(status, "[Get][MaxCount] of normal stream and task failed. graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  if (task_count > static_cast<int64_t>(max_normal_task_count)) {
    uint32_t max_huge_stream_count = 0;
    uint32_t max_huge_task_count = 0;
    status = GetMaxStreamAndTask(true, max_huge_stream_count, max_huge_task_count);
    if (status == SUCCESS) {
      int64_t huge_stream = 0;
      GELOGI("Use huge stream %ld.", huge_stream);
      huge_streams_.emplace_back(huge_stream);
    } else {
      GELOGW(
          "The estimated task count %ld is greater than the max count of normal stream,"
          " but the huge stream is not supported.",
          task_count);
    }
  }

  return SUCCESS;
}

Status StreamAllocator::SetActiveStreamsByLabel() {
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::string stream_label;
    if (AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label) && !stream_label.empty()) {
      int64_t stream_id = op_desc->GetStreamId();
      if (stream_id != kInvalidStream) {
        labeled_streams_[stream_label].emplace(stream_id);
      }
    }
  }

  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::vector<std::string> activated_label_list;
    if (!AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, activated_label_list) ||
        activated_label_list.empty()) {
      continue;
    }

    std::vector<uint32_t> activated_stream_list;
    for (std::string &activated_label : activated_label_list) {
      specific_activated_labels_[activated_label].emplace(node);
      for (int64_t activated_stream : labeled_streams_[activated_label]) {
        activated_stream_list.push_back(static_cast<uint32_t>(activated_stream));
        specific_activated_streams_.emplace(activated_stream);
        specific_activated_streams_nodes_map_[activated_stream].emplace(node);
        GELOGI("Node %s active stream %ld by %s.", node->GetName().c_str(), activated_stream, activated_label.c_str());
      }
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, activated_stream_list),
                     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s for op:%s(%s) failed",
                                        ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                                        node->GetName().c_str(), node->GetType().c_str());
                     GELOGE(FAILED, "[Set][Attr] %s for op:%s(%s) failed",
                            ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                            node->GetName().c_str(), node->GetType().c_str());
                     return FAILED);
  }

  return SUCCESS;
}

Status StreamAllocator::SetActiveStreamsForSubgraphs() {
  for (auto &subgraph : whole_graph_->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(subgraph);
    GELOGI("Start to collect streams of graph %s", subgraph->GetName().c_str());
    NodePtr first_active_node = nullptr;

    // Get all streams in subgraph.
    std::set<int64_t> subgraph_streams;
    for (auto &node : subgraph->GetDirectNode()) {
      OpDescPtr op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      // Skip streams with label
      std::string stream_label;
      if (AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label) && !stream_label.empty()) {
        continue;
      }
      CollectSubgraphStreams(op_desc, subgraph_streams);
      bool is_first_active = false;
      if (AttrUtils::GetBool(op_desc, ATTR_NAME_SUBGRAPH_FIRST_ACTIVE, is_first_active) && is_first_active) {
        first_active_node = node;
      }
    }

    if (first_active_node == nullptr) {
      continue;
    }

    subgraph_first_active_node_map_[subgraph] = first_active_node;

    // Set active streams for StreamActive.
    subgraph_streams.erase(first_active_node->GetOpDesc()->GetStreamId());

    std::vector<uint32_t> active_streams;
    (void)AttrUtils::GetListInt(first_active_node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams);
    for (int64_t active_stream : subgraph_streams) {
      active_streams.emplace_back(static_cast<uint32_t>(active_stream));
      specific_activated_streams_.emplace(active_stream);
    }

    if (!AttrUtils::SetListInt(first_active_node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s for op:%s(%s) failed",
                         ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                         first_active_node->GetName().c_str(), first_active_node->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] active streams for node %s failed.", first_active_node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

// Insert the send/recv event id to the graph
Status StreamAllocator::InsertSyncEvents(const EventType insert_event_type) {
  auto ffts_filter = [](const Node &node, const char *, const ComputeGraphPtr &) {
    return !(node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) ||
             node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH));
  };

  for (const auto &cur_node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag(), nullptr, ffts_filter)) {
    // Take the adjacent points, then judge whether need to insert the event
    for (const OutDataAnchorPtr &anchor : cur_node->GetAllOutDataAnchors()) {
      for (const InDataAnchorPtr &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
        NodePtr next_node = peer_in_anchor->GetOwnerNode();
        Status status = InsertOneEventInTwoNodes(insert_event_type, cur_node, next_node);
        if (status != SUCCESS) {
          GELOGE(status, "[Insert][One %s] In Two Nodes failed! cur node:%s", cur_node->GetName().c_str(),
                 GetEventTypeStr(insert_event_type).c_str());
          return status;
        }
      }
    }

    /// If the two nodes of the control side belong to two streams,
    /// you also need to add the send/recv event.
    if (cur_node->GetOutControlAnchor() != nullptr) {
      for (const auto peer_in_anchor : cur_node->GetOutControlAnchor()->GetPeerAnchorsPtr()) {
        NodePtr next_node = peer_in_anchor->GetOwnerNode();
        Status status = InsertOneEventInTwoNodes(insert_event_type, cur_node, next_node);
        if (status != SUCCESS) {
          GELOGE(status, "[Insert][One %s] In Two Nodes failed! cur node:%s", cur_node->GetName().c_str(),
                 GetEventTypeStr(insert_event_type).c_str());
          return status;
        }
      }
    }
  }

  Status status = InsertEventsForSubgraph(insert_event_type);
  if (status != SUCCESS) {
    GELOGE(status, "[Insert][%ss] Between Sub And Parent GraphNodes failed! graph:%s",
           GetEventTypeStr(insert_event_type).c_str(), whole_graph_->GetName().c_str());
    return status;
  }

  return SUCCESS;
}

// Insert one send/recv event in two nodes
Status StreamAllocator::InsertOneEventInTwoNodes(const EventType insert_event_type, const NodePtr &cur_node,
                                                 const NodePtr &next_node) {
  const auto cur_desc = cur_node->GetOpDesc();
  GE_CHECK_NOTNULL(cur_desc);
  const auto next_desc = next_node->GetOpDesc();
  GE_CHECK_NOTNULL(next_desc);

  // No need to insert events after node that do not assign streams.
  int64_t cur_stream_id = cur_desc->GetStreamId();
  if (cur_stream_id == kInvalidStream) {
    GELOGD("No need to insert %s after node %s.", GetEventTypeStr(insert_event_type).c_str(),
           cur_node->GetName().c_str());
    return SUCCESS;
  }

  // No need to insert events between nodes in the same stream.
  int64_t next_stream_id = next_desc->GetStreamId();
  if (cur_stream_id == next_stream_id) {
    return SUCCESS;
  }

  if (((cur_node->GetType() == ENTER) || (cur_node->GetType() == REFENTER)) && (next_node->GetType() != STREAMACTIVE)) {
    GELOGD("No need to insert %s between %s and %s.", GetEventTypeStr(insert_event_type).c_str(),
           cur_node->GetName().c_str(), next_node->GetName().c_str());
    return SUCCESS;
  }

  if (next_stream_id == kInvalidStream) {
    std::string ffts_str;
    if (next_desc->HasAttr(ATTR_NAME_THREAD_SCOPE_ID)) {
      GELOGI("FFTS+ node: %s, skip inserting events.", next_desc->GetName().c_str());
      return SUCCESS;
    }
    REPORT_INNER_ERR_MSG("E19999", "Stream id of next_node %s(%s) is invalid, need check why assign invalid stream",
                       next_node->GetName().c_str(), next_node->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Stream id of next_node %s is invalid, need check why assign invalid stream",
           next_node->GetName().c_str());
    return FAILED;
  }

  // No event needs to be inserted between the active node and the activated stream.
  std::string next_node_label;
  if (AttrUtils::GetStr(next_desc, ATTR_NAME_STREAM_LABEL, next_node_label) && !next_node_label.empty()) {
    auto iter = specific_activated_labels_.find(next_node_label);
    if (iter != specific_activated_labels_.end()) {
      for (const auto &active_node : iter->second) {
        OpDescPtr active_op = active_node->GetOpDesc();
        GE_CHECK_NOTNULL(active_op);
        if (IsBelongToSameGraph(cur_node, active_node) && (cur_stream_id == active_op->GetStreamId()) &&
            (cur_desc->GetId() <= active_op->GetId())) {
          GELOGI("No need to insert %s between node %s and %s.", GetEventTypeStr(insert_event_type).c_str(),
                 cur_node->GetName().c_str(), next_node->GetName().c_str());
          return SUCCESS;
        }
      }
    }
  }

  if (insert_event_type == EventType::kNotify) {
    // Add send and receive notifies.
    StreamUtils::AddSendEventId(cur_node, notify_num_, node_to_send_notifies_);
    StreamUtils::AddRecvEventId(next_node, notify_num_, node_to_recv_notifies_);
    ++notify_num_;
    GELOGI("Insert notify %u between node %s(stream %ld) and %s(stream %ld)", notify_num_, cur_node->GetName().c_str(),
           cur_stream_id, next_node->GetName().c_str(), next_stream_id);
  } else {
    // Add send and receive events.
    StreamUtils::AddSendEventId(cur_node, event_num_, node_to_send_events_);
    StreamUtils::AddRecvEventId(next_node, event_num_, node_to_recv_events_);
    ++event_num_;
    GELOGI("Insert event %u between node %s(stream %ld) and %s(stream %ld)", event_num_, cur_node->GetName().c_str(),
           cur_stream_id, next_node->GetName().c_str(), next_stream_id);
  }

  return SUCCESS;
}

Status StreamAllocator::InsertEventsForSubgraph(const EventType insert_event_type) {
  for (const auto &subgraph : whole_graph_->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(subgraph);
    const auto &parent_node = subgraph->GetParentNode();
    if (parent_node != nullptr) {
      const auto &op_desc = parent_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const auto is_ffts_subgraph = op_desc->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) ||
                                    op_desc->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH) ||
                                    op_desc->HasAttr(ATTR_NAME_THREAD_SCOPE_ID);
      if (is_ffts_subgraph) {
        GELOGD("Skip ffts subgraph, parent node is %s.", op_desc->GetName().c_str());
        continue;
      }
    }

    for (const auto &node : subgraph->GetDirectNode()) {
      const auto &op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      bool is_subgraph_end_node = false;
      if (!AttrUtils::GetBool(op_desc, ATTR_NAME_SUBGRAPH_END_NODE, is_subgraph_end_node) || !is_subgraph_end_node) {
        continue;
      }
      const auto &sub_parent_node = subgraph->GetParentNode();
      GE_CHECK_NOTNULL(sub_parent_node);

      // Insert events between subgraph end node and parent node's out nodes
      for (const auto &next_node : sub_parent_node->GetOutAllNodes()) {
        Status status = InsertOneEventInTwoNodes(insert_event_type, node, next_node);
        if (status != SUCCESS) {
          GELOGE(status, "[Insert][One %s] In Two Nodes failed! node:%s", GetEventTypeStr(insert_event_type).c_str(),
                 node->GetName().c_str());
          return status;
        }
      }

      break;
    }
  }

  return SUCCESS;
}

// Optimize the event in the graph, delete the redundant sync event according to the stream information
Status StreamAllocator::OptimizeSyncEvents(
    const EventType event_type, std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  std::map<int64_t, std::vector<NodePtr>> stream_nodes;

  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    int64_t stream_id = node->GetOpDesc()->GetStreamId();
    stream_nodes[stream_id].emplace_back(node);
  }

  Status status = StreamUtils::OptimizeBySendEvents(stream_nodes, node_to_send_events, node_to_recv_events);
  if (status != SUCCESS) {
    GELOGE(status, "[Optimize][StreamNodes] By send %s failed! graph:%s", whole_graph_->GetName().c_str(),
           GetEventTypeStr(event_type).c_str());
    return status;
  }

  status = StreamUtils::OptimizeByRecvEvents(stream_nodes, node_to_send_events, node_to_recv_events);
  if (status != SUCCESS) {
    GELOGE(status, "[Optimize][StreamNodes] By recv %ss failed! graph:%s", whole_graph_->GetName().c_str(),
           GetEventTypeStr(event_type).c_str());
    return status;
  }

  status = OptimizeByStreamActivate(event_type, node_to_send_events, node_to_recv_events);
  if (status != SUCCESS) {
    GELOGE(status, "[Call][OptimizeByStreamActivate] failed! graph:%s", whole_graph_->GetName().c_str());
    return status;
  }

  for (auto pair : node_to_send_events) {
    if (pair.first->GetType() == STREAMSWITCH) {
      GELOGI("node_to_send_events STREAMSWITCH.");

      for (auto event_id : pair.second) {
        GELOGI("Curren switch node is %s, remove send %s_id %d.", GetEventTypeStr(event_type).c_str(),
               pair.first->GetName().c_str(), event_id);
        StreamUtils::RmvSendEventId(pair.first, event_id, node_to_send_events);
        auto recv_node = StreamUtils::GetNodeFromSyncId(event_id, node_to_recv_events);
        GE_CHECK_NOTNULL(recv_node);
        GELOGI("Curren recv_node is %s, remove recv %s_id %d.", GetEventTypeStr(event_type).c_str(),
               recv_node->GetName().c_str(), event_id);
        StreamUtils::RmvRecvEventId(recv_node, event_id, node_to_recv_events);
      }
    }
  }
  return SUCCESS;
}

Status StreamAllocator::OptimizeByStreamActivate(
    const EventType event_type, std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) const {
  auto node_to_send_events_temp = node_to_send_events;
  for (const auto &node_event_id_pair : node_to_send_events_temp) {
    const NodePtr &send_node_ptr = node_event_id_pair.first;
    for (const auto &event_id : node_event_id_pair.second) {
      NodePtr recv_node_ptr = StreamUtils::GetNodeFromSyncId(event_id, node_to_recv_events);
      GE_CHECK_NOTNULL(recv_node_ptr);
      if (IsRecvNodeActivatedBySendNode(send_node_ptr, recv_node_ptr)) {
        StreamUtils::RmvSendEventId(send_node_ptr, event_id, node_to_send_events);
        StreamUtils::RmvRecvEventId(recv_node_ptr, event_id, node_to_recv_events);
        GELOGI("Remove %s %u between node %s and node %s.", GetEventTypeStr(event_type).c_str(), event_id,
               send_node_ptr->GetName().c_str(), recv_node_ptr->GetName().c_str());
      }
    }
  }
  return SUCCESS;
}

// In situation : stream(normal) -> stream(streamActivate)->
// -> stream(streamSwitch) -> stream(streamActivate) -> stream(stream true or false)
// No need to insert an event between node in stream(normal) and node in stream(stream true or false)
bool StreamAllocator::IsRecvNodeActivatedBySendNode(const NodePtr &send_node_ptr, const NodePtr &recv_node_ptr) const {
  GE_CHECK_NOTNULL_EXEC(send_node_ptr->GetOpDesc(),
                        REPORT_INNER_ERR_MSG("E19999", "Check param send_node_ptr nullptr");
                        GELOGE(FAILED, "[Check][Param] op desc is nullptr");
                        return false);
  GE_CHECK_NOTNULL_EXEC(recv_node_ptr->GetOpDesc(),
                        REPORT_INNER_ERR_MSG("E19999", "Check param recv_node_ptr nullptr");
                        GELOGE(FAILED, "[Check][Param] op desc is nullptr");
                        return false);
  auto cur_stream_id = send_node_ptr->GetOpDesc()->GetStreamId();
  if (AttrUtils::HasAttr(recv_node_ptr->GetOpDesc(), ATTR_NAME_STREAM_LABEL)) {
    // find streamActivate node
    auto iter = specific_activated_streams_nodes_map_.find(recv_node_ptr->GetOpDesc()->GetStreamId());
    std::set<NodePtr> activate_stream_nodes;
    if (iter != specific_activated_streams_nodes_map_.end()) {
      activate_stream_nodes = iter->second;
    }
    std::set<NodePtr> visited_nodes{recv_node_ptr};
    while (!activate_stream_nodes.empty()) {
      std::set<NodePtr> activate_stream_nodes_temp;
      for (const auto &activate_stream_node : activate_stream_nodes) {
        if (!IsBelongToSameGraph(send_node_ptr, activate_stream_node)) continue;
        GE_IF_BOOL_EXEC(activate_stream_node->GetOpDesc() == nullptr, continue);
        if (visited_nodes.find(activate_stream_node) != visited_nodes.end() ||
            AttrUtils::HasAttr(activate_stream_node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE)) {
          return false;
        }

        ///
        /// stream_0  -->  stream_2  -->  stream_3  -->  stream_4
        ///                   /\             |
        ///                   |             \/
        ///                   |           stream_1  -->  stream_5  -->  stream_6  -->  stream_7
        ///                   |                             /\             |              |
        ///                   |                             |             \/              |
        ///                   |                             |---------- stream_8          |
        ///                   |                                                           |
        ///                   |-----------------------------------------------------------|
        ///
        ///  Exit1(S7) Exit2(S7)  Exit3(S7)
        ///     \       /           |
        ///     AddN(S1)     NextIteration(S7)
        ///       |                 |
        ///     NextIteration(S1)  /
        ///          |            /
        ///          |           /
        ///        StreamActive(S7)
        ///
        /// Event between Exit1/Exit2 and AddN should not be optimized
        ///
        if (IsActiveAfterNextIteration(activate_stream_node)) {
          continue;
        }

        visited_nodes.insert(activate_stream_node);
        // nodes in stream link to streamActivate no need to add event/notify before activated node
        for (const auto &pre_activate_stream_node : activate_stream_node->GetInNodes()) {
          GE_IF_BOOL_EXEC(pre_activate_stream_node->GetOpDesc() == nullptr, continue);
          if (pre_activate_stream_node->GetOpDesc()->GetStreamId() == cur_stream_id &&
              pre_activate_stream_node->GetOpDesc()->GetId() >= send_node_ptr->GetOpDesc()->GetId()) {
            return true;
          }
          auto in_nodes_of_pre = pre_activate_stream_node->GetInNodes();
          if (std::find(in_nodes_of_pre.begin(), in_nodes_of_pre.end(), send_node_ptr) != in_nodes_of_pre.end()) {
            return true;
          }
        }
        auto iterator = specific_activated_streams_nodes_map_.find(activate_stream_node->GetOpDesc()->GetStreamId());
        if (iterator != specific_activated_streams_nodes_map_.end()) {
          auto active_nodes = iterator->second;
          for (const auto &active_node : active_nodes) {
            activate_stream_nodes_temp.emplace(active_node);
          }
        }
      }
      activate_stream_nodes = activate_stream_nodes_temp;
    }
  }
  return false;
}

bool StreamAllocator::IsActiveAfterNextIteration(const NodePtr &active_node_ptr) const {
  if ((active_node_ptr == nullptr) || active_node_ptr->GetInControlNodes().empty()) {
    return false;
  }
  for (const auto &in_node : active_node_ptr->GetInControlNodes()) {
    if ((in_node->GetType() != NEXTITERATION) && (in_node->GetType() != REFNEXTITERATION)) {
      return false;
    }
  }
  return true;
}

Status StreamAllocator::SplitStreamForOneNode(StreamSplitNodeInfo &stream_split_node_info, StreamSplitHelper &helper,
                                              std::vector<std::set<int64_t>> &split_streams,
                                              std::vector<domi::TaskDef> &task_defs) {
  const auto &cur_node = stream_split_node_info.cur_node;
  GE_ASSERT_NOTNULL(cur_node);
  const auto &op_desc = cur_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t stream_id = stream_split_node_info.stream_id;
  GELOGD("Try split stream for node {%s %s} with stream_id: %ld, is_split_for_attached_stream: %d",
         op_desc->GetNamePtr(), op_desc->GetTypePtr(), stream_id, stream_split_node_info.split_for_attached_stream);
  if (stream_id == kInvalidStream) {
    return SUCCESS;
  }
  GE_ASSERT_TRUE(stream_id <= helper.last_stream_id, "op %s type %s stream_id(%ld) should <= last_stream_id(%ld)",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), stream_id, helper.last_stream_id);
  stream_split_node_info.is_stream_first_node = (helper.stream_task_num_vec[stream_id] == 0);
  AddTaskNum(cur_node, helper.stream_task_num_vec[stream_id], stream_split_node_info.assigned_task_num,
             stream_split_node_info.split_for_attached_stream);
  helper.stream_2_nodes_map[stream_id].push_back(cur_node);
  if (stream_split_node_info.split_for_attached_stream) {
    (void)helper.attached_stream_.insert(stream_id);
  }
  std::string continuous_stream_label;
  if (HasContinuousStreamLabel(op_desc, continuous_stream_label)) {
    helper.stream_continuous_2_node_num_map[continuous_stream_label]++;
    helper.stream_continuous_2_nodes_map[continuous_stream_label].push_back(cur_node);
  }

  GE_ASSERT_SUCCESS(SplitNodesToNewStream(stream_split_node_info, split_streams, helper),
                    "Split nodes to new stream failed.");

  /// If the split stream num is greater than 1, the node behind the same
  /// stream must reset the new stream id.
  bool has_attached_stream = (op_desc->HasValidAttachedStreamId()) || (op_desc->GetType() == "SuperKernel");
  if (helper.added_stream_num_vec[stream_id] >= 1) {
    StreamUtils::SetStreamId(op_desc, helper.latest_stream_id_vec[stream_id],
                             stream_split_node_info.split_for_attached_stream, stream_id);
    GELOGI("op name [%s] is split to new stream id %lld, is attached stream: %zu", op_desc->GetName().c_str(),
           helper.latest_stream_id_vec[stream_id], static_cast<size_t>(stream_split_node_info.split_for_attached_stream));
    RefreshTaskDefStreamId(has_attached_stream, stream_id,
                           helper.latest_stream_id_vec[stream_id], task_defs);
  } else {
    RefreshTaskDefStreamId(has_attached_stream, stream_id, stream_id, task_defs);
  }

  helper.pre_node_vec[stream_id] = cur_node;
  return SUCCESS;
}

Status StreamAllocator::RefreshStreamActiveNodeTaskSize(
    const std::vector<NodePtr> &stream_active_nodes,
    const std::map<int64_t, size_t> &logical_stream_id_to_real_stream_num) {
  for (auto stream_active_node : stream_active_nodes) {
    auto op_desc = stream_active_node->GetOpDescBarePtr();
    auto node_id = op_desc->GetId();
    auto stream_id = op_desc->GetStreamId();
    const auto &node_id_to_task_num_info = node_id_to_task_num_infos_.find(node_id);
    GE_ASSERT_TRUE(node_id_to_task_num_info != node_id_to_task_num_infos_.end());
    auto &task_num_infos = node_id_to_task_num_info->second;
    std::vector<uint32_t> active_streams;
    size_t task_size = 0U;
    if (AttrUtils::GetListInt(stream_active_node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
      for (auto activated_stream_id : active_streams) {
        const auto &iter = logical_stream_id_to_real_stream_num.find(activated_stream_id);
        if (iter != logical_stream_id_to_real_stream_num.end()) {
          task_size += iter->second;
        }
      }
    }
    task_num_infos[stream_id] = task_size;
    GELOGI("stream active node %s reserve %zu task size", stream_active_node->GetNamePtr(), task_size);
  }
  return SUCCESS;
}

Status StreamAllocator::CollectTaskSize(
    std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks,
    uint32_t per_stream_max_task_size) {
  std::vector<NodePtr> stream_active_nodes;
  std::map<int64_t, size_t> logical_stream_id_to_total_task_num;
  for (const auto &cur_node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    auto op_desc = cur_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_id = op_desc->GetId();
    auto &task_defs = node_id_2_node_tasks[node_id];
    if (task_defs.size() > 0U) {
      // const或者data节点的stream是-1，所以可能导致在其后面插入的event的stream是-1，不过流拆分的最后会把所有-1刷成0，
      // 所以这里计算task数量时针对有task的算子，需要将其stream id设置成0
      if (op_desc->GetStreamId() == kInvalidStream) {
        GELOGW("node %s task size %zu, stream id %ld is invalid, will set 0", op_desc->GetNamePtr(), task_defs.size(),
               op_desc->GetStreamId());
        op_desc->SetStreamId(0);
      }
    }
    auto stream_id = op_desc->GetStreamId();
    if (stream_id == kInvalidStream) {
      continue;
    }
    if (op_desc->GetType() == STREAMACTIVE) {
      stream_active_nodes.emplace_back(cur_node);
    }
    auto &task_num_info = node_id_to_task_num_infos_[node_id];
    task_num_info[op_desc->GetStreamId()] = 0U;
    CollectStreamIdToTaskSize(task_defs, task_num_info);
    auto assigned_task_num = StreamUtils::GetAssignedTaskNum(cur_node, false);
    if (assigned_task_num > 0U) {
      task_num_info[op_desc->GetStreamId()] = assigned_task_num;
    }
    auto assigned_attached_task_num = StreamUtils::GetAssignedTaskNum(cur_node, true);
    for (const auto attached_stream_id : op_desc->GetAttachedStreamIds()) {
      if (assigned_attached_task_num > 0U) {
        task_num_info[attached_stream_id] = assigned_attached_task_num;
      }
    }
    for (const auto &iter : task_num_info) {
      logical_stream_id_to_total_task_num[iter.first] += iter.second;
    }
  }
  // 给原图上的StreamActive预留task位置
  std::map<int64_t, size_t> logical_stream_id_to_real_stream_num;
  for (const auto &iter : logical_stream_id_to_total_task_num) {
    logical_stream_id_to_real_stream_num[iter.first] =
        std::ceil(static_cast<double>(iter.second) / static_cast<double>(per_stream_max_task_size));
  }
  GE_ASSERT_SUCCESS(RefreshStreamActiveNodeTaskSize(stream_active_nodes, logical_stream_id_to_real_stream_num));
  return SUCCESS;
}

// Split the stream according to the maximum number of tasks in the stream.
Status StreamAllocator::SplitStreams(
    std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks,
    std::vector<std::set<int64_t>> &split_streams) {
  if (stream_num_ == 0) {
    GELOGI("The number of streams is 0, no need to split streams.");
    return SUCCESS;
  }

  auto &helper = helper_;
  helper.Init(stream_num_);

  bool is_huge_stream = false;
  if (enable_single_stream_ && (!huge_streams_.empty())) {
    is_huge_stream = true;
    GELOGI("use huge stream");
  }
  GE_ASSERT_SUCCESS(GetMaxStreamAndTask(is_huge_stream, helper.max_stream_count, helper.max_task_count),
                    "[Get][MaxCount] of stream and task failed.");

  GE_ASSERT_SUCCESS(CollectTaskSize(node_id_2_node_tasks, helper.max_task_count));
  for (const auto &cur_node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    auto op_desc = cur_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto &task_defs = node_id_2_node_tasks[op_desc->GetId()];
    const auto stream_id_to_task_size = node_id_to_task_num_infos_[op_desc->GetId()];
    GELOGI("node: %s, type %s, topo id: %ld, logical stream id: %ld, real stream size: %zu, stream id to task size: %s",
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_desc->GetId(), op_desc->GetStreamId(),
           stream_id_to_task_size.size(), PrintStreamIdToTaskSize(stream_id_to_task_size).c_str());

    for (const auto &iter : stream_id_to_task_size) {
      auto stream_id = iter.first;
      auto task_size = iter.second;
      bool is_for_attached_stream = (op_desc->GetStreamId() != stream_id);
      StreamSplitNodeInfo stream_split_node_info = {cur_node, false, is_for_attached_stream, task_size, stream_id};
      GE_ASSERT_SUCCESS(SplitStreamForOneNode(stream_split_node_info, helper, split_streams, task_defs),
                        "op %s type %s split stream failed", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    }
  }

  if (helper.last_stream_id >= 0) {
    stream_num_ = helper.last_stream_id + 1;
  }

  return SUCCESS;
}

Status StreamAllocator::SplitNodesToNewStream(const StreamSplitNodeInfo &stream_split_node_info,
                                              std::vector<std::set<int64_t>> &split_streams,
                                              StreamSplitHelper &helper) {
  const auto &cur_node = stream_split_node_info.cur_node;
  GE_ASSERT_NOTNULL(cur_node);
  const auto &op_desc = cur_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const int64_t stream_id = stream_split_node_info.stream_id;
  GE_ASSERT_TRUE(stream_id >= 0);
  split_stream_id_to_logic_stream_id_[stream_id] = stream_id;
  // Split the stream if it exceeds the maximum number of tasks in the stream.
  if (NeedSpiltNewStream(helper.stream_task_num_vec[stream_id], helper.max_task_count, op_desc,
                         stream_split_node_info.is_stream_first_node)) {
    helper.last_stream_id++;
    GELOGI(
        "stream[%ld]'s task num[%ld] > max_task_num_one_stream[%u], split stream to %ld, first node[name: %s, "
        "type: %s, owner graph: %s].",
        stream_id, (helper.stream_task_num_vec[stream_id] + kReservedTaskNum), helper.max_task_count,
        helper.last_stream_id, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
        cur_node->GetOwnerComputeGraph()->GetName().c_str());
    NodePtr pre_node = helper.pre_node_vec[stream_id];
    helper.stream_task_num_vec[stream_id] = 0;
    AddTaskNum(cur_node, helper.stream_task_num_vec[stream_id], stream_split_node_info.assigned_task_num,
               stream_split_node_info.split_for_attached_stream);
    // try spilt a new stream and move same continuous stream label nodes from this stream
    bool not_use_cur = false;
    NodePtr not_cur = nullptr;
    std::string cur_continuous_stream_label;
    if (HasContinuousStreamLabel(op_desc, cur_continuous_stream_label)) {
      // get stored nodes
      auto nodes = helper.stream_continuous_2_nodes_map[cur_continuous_stream_label];
      GE_RETURN_WITH_LOG_IF_FALSE(!nodes.empty(), "[Check][Param] split stream with continuous stream label %s failed",
                                  cur_continuous_stream_label.c_str());
      for (const auto &node : nodes) {
        auto stored_op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(stored_op_desc);
        StreamUtils::SetStreamId(stored_op_desc, helper.last_stream_id,
                                 stream_split_node_info.split_for_attached_stream, stream_id);
        AddTaskNum(node, helper.stream_task_num_vec[stream_id], stream_split_node_info.assigned_task_num,
                   stream_split_node_info.split_for_attached_stream);
      }
      not_use_cur = true;
      not_cur = nodes.front();
      GE_CHECK_NOTNULL(not_cur);
      GELOGI("split from first node %s with continuous stream label %s", not_cur->GetName().c_str(),
             cur_continuous_stream_label.c_str());
      auto iter = std::find(helper.stream_2_nodes_map[stream_id].begin(),
                            helper.stream_2_nodes_map[stream_id].end(), not_cur);
      GE_RETURN_WITH_LOG_IF_FALSE(
          (iter != helper.stream_2_nodes_map[stream_id].end()) &&
          (iter != helper.stream_2_nodes_map[stream_id].begin()),
          "[Check][Param] split stream with continuous stream label %s failed", cur_continuous_stream_label.c_str());
      iter--;
      pre_node = *iter;
    }

    helper.added_stream_num_vec[stream_id]++;
    auto pre_stream_id = helper.latest_stream_id_vec[stream_id];
    helper.latest_stream_id_vec[stream_id] = helper.last_stream_id;
    split_streams[stream_id].emplace(helper.last_stream_id);
    split_stream_id_to_logic_stream_id_[helper.last_stream_id] = stream_id;
    node_split_stream_map_[cur_node] = helper.last_stream_id;

    if (pre_node != nullptr) {
      // Add the send/recv event to the first and last nodes of the split stream.
      auto next_stream_id = helper.last_stream_id;
      GE_CHK_STATUS_RET(
          AddEventIdWhenStreamSplit({pre_node, not_cur, cur_node, not_use_cur,
                                     stream_split_node_info.split_for_attached_stream, pre_stream_id, next_stream_id}),
          "[Add][EventId] failed, pre node:%s, not cur node:%s, cur node:%s.", pre_node->GetName().c_str(),
          not_cur->GetName().c_str(), cur_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

bool StreamAllocator::NeedSpiltNewStream(int64_t stream_node_num, int64_t max_node_num_one_stream,
                                         const OpDescPtr &op_desc, bool is_stream_first_node) const {
  if (is_stream_first_node) {
    GELOGD("First node of stream does not need to split new stream");
    return false;
  }
  static const std::set<std::string> label_op_types({LABELSET, LABELGOTOEX, LABELSWITCHBYINDEX});
  bool is_first_active_node = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_SUBGRAPH_FIRST_ACTIVE, is_first_active_node);
  return (((stream_node_num + kReservedTaskNum) > max_node_num_one_stream) && op_desc->GetSubgraphInstanceNames().empty() &&
          !is_first_active_node && (label_op_types.count(op_desc->GetType()) == 0));
}

Status StreamAllocator::UpdateActiveStreams(const std::vector<std::set<int64_t>> &split_streams) {
  UpdateLabelStreams(split_streams);

  for (auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    if (node->GetType() == STREAMSWITCH) {
      GE_ASSERT_SUCCESS(UpdateActiveStreamsForSwitchNode(node) != SUCCESS,
                        "[Update][ActiveStreams] for switch node: %s failed.", node->GetName().c_str());
    } else {
      GE_ASSERT_SUCCESS(UpdateActiveStreamsForActiveNode(split_streams, node) != SUCCESS,
                        "[Update][ActiveStreams] for active node: %s failed.", node->GetName().c_str());
    }
  }

  GE_ASSERT_SUCCESS(UpdateActiveStreamsForSubgraphs(), "[Update][ActiveStreams] for subgraphs failed! graph:%s",
                    whole_graph_->GetName().c_str());

  GE_ASSERT_SUCCESS(SetActiveStreamsForLoop(false, split_streams), "[Set][ActiveStreams] For Loop failed! graph:%s",
                    whole_graph_->GetName().c_str());

  return CheckStreamActived();
}

void StreamAllocator::UpdateLabelStreams(const std::vector<std::set<int64_t>> &split_streams) {
  for (size_t i = 0; i < split_streams.size(); i++) {
    auto &streams = split_streams[i];
    if (streams.empty()) {
      continue;
    }
    if (specific_activated_streams_.count(static_cast<int64_t>(i)) > 0) {
      specific_activated_streams_.insert(streams.cbegin(), streams.cend());
    }
    for (auto &labeled_stream : labeled_streams_) {
      if (labeled_stream.second.count(static_cast<int64_t>(i)) > 0) {
        labeled_stream.second.insert(streams.cbegin(), streams.cend());
        break;
      }
    }
  }
}

Status StreamAllocator::UpdateActiveStreamsForSwitchNode(const NodePtr &switch_node) {
  std::vector<NodePtr> active_nodes;
  if (InsertActiveNodesAfterSwitch(switch_node, active_nodes) != SUCCESS) {
    GELOGE(FAILED, "[Insert][ActiveNodes] after node %s failed.", switch_node->GetName().c_str());
    return FAILED;
  }
  if (active_nodes.empty()) {
    return SUCCESS;
  }
  std::vector<int64_t> stream_ids;
  for (auto &active_node : active_nodes) {
    GE_CHECK_NOTNULL(active_node->GetOpDesc());
    active_node->GetOpDesc()->SetStreamId(stream_num_);
    stream_ids.emplace_back(stream_num_);
    specific_activated_streams_.emplace(stream_num_);
    stream_num_++;
  }
  auto op_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (!AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, stream_ids)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status StreamAllocator::InsertActiveNodesAfterSwitch(const NodePtr &switch_node, std::vector<NodePtr> &active_nodes) {
  GE_CHECK_NOTNULL(switch_node);
  OpDescPtr switch_desc = switch_node->GetOpDesc();
  GE_CHECK_NOTNULL(switch_desc);
  std::vector<std::string> ori_active_label_list;
  if (!AttrUtils::GetListStr(switch_desc, ATTR_NAME_ACTIVE_LABEL_LIST, ori_active_label_list) ||
      ori_active_label_list.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s fail from op:%s(%s)", ATTR_NAME_ACTIVE_LABEL_LIST.c_str(),
                       switch_node->GetName().c_str(), switch_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] active label list of switch %s failed.", switch_node->GetName().c_str());
    return INTERNAL_ERROR;
  }

  std::vector<std::string> active_label_list;
  std::vector<NodePtr> added_active_nodes;
  if (AddActiveNodes(switch_node, ori_active_label_list, active_label_list, added_active_nodes) != SUCCESS) {
    GELOGE(FAILED, "[Add][ActiveNodes] after node %s failed.", switch_node->GetName().c_str());
    return FAILED;
  }

  if (SetActiveLabelList(switch_node, active_label_list) != SUCCESS) {
    GELOGE(FAILED, "[Set][ActiveLabelList] failed, node:%s", switch_node->GetName().c_str());
    return FAILED;
  }

  if (added_active_nodes.empty()) {
    return SUCCESS;
  }

  for (auto &active_node : added_active_nodes) {
    GE_CHECK_NOTNULL(switch_node->GetOutControlAnchor());
    if (switch_node->GetOutControlAnchor()->LinkTo(active_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Link from %s to %s failed",
                            switch_node->GetName().c_str(), active_node->GetName().c_str());
      GELOGE(FAILED, "[Link][Nodes] from %s to %s failed.",
             switch_node->GetName().c_str(), active_node->GetName().c_str());
      return FAILED;
    }
    active_nodes.emplace_back(active_node);
  }
  return SUCCESS;
}

Status StreamAllocator::UpdateActiveStreamsForActiveNode(const std::vector<std::set<int64_t>> &split_streams,
                                                         const NodePtr &node) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::vector<uint32_t> active_streams;
  if (AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
    std::vector<uint32_t> new_active_streams = active_streams;
    for (uint32_t logical_stream : active_streams) {
      GE_ASSERT_SUCCESS(static_cast<size_t>(logical_stream) >= split_streams.size(), "[Check][Param] logical stream:%u is out of range(0, %zu).",
                        logical_stream, split_streams.size());
      const std::set<int64_t> &new_split_streams = split_streams[logical_stream];
      for (int64_t split_stream : new_split_streams) {
        for (const auto &node_stream : node_split_stream_map_) {
          if (split_stream == node_stream.second) {
            if (node_stream.first->GetOwnerComputeGraph() == node->GetOwnerComputeGraph()) {
              new_active_streams.emplace_back(static_cast<uint32_t>(split_stream));
              GELOGI("Add stream %ld to active_stream_list of node %s of graph %s", split_stream,
                     node->GetName().c_str(), node->GetOwnerComputeGraph()->GetName().c_str());
            }
            break;
          }
        }
      }
    }
    (void) AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, new_active_streams);
  }
  return SUCCESS;
}

Status StreamAllocator::UpdateActiveStreamsForSubgraphs() {
  // Update active stream list for active nodes
  for (auto &node_stream_pair : node_split_stream_map_) {
    auto node = node_stream_pair.first;
    auto subgraph = node->GetOwnerComputeGraph();
    if (subgraph->GetParentNode() == nullptr) {
      continue;
    }
    // Skip streams with label
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string stream_label;
    if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label) && !stream_label.empty()) {
      continue;
    }
    std::map<ComputeGraphPtr, NodePtr>::const_iterator it = subgraph_first_active_node_map_.find(subgraph);
    if (it == subgraph_first_active_node_map_.cend()) {
      continue;
    }
    const auto &active_node = it->second;
    GE_CHECK_NOTNULL(active_node);
    GELOGI("Subgraph[%s]'s first active node is %s.", subgraph->GetName().c_str(), active_node->GetName().c_str());
    auto active_op = active_node->GetOpDesc();
    GE_CHECK_NOTNULL(active_op);
    std::vector<uint32_t> active_streams;
    (void)AttrUtils::GetListInt(active_op, ATTR_NAME_ACTIVE_STREAM_LIST, active_streams);
    std::set<uint32_t> new_active_streams(active_streams.begin(), active_streams.end());
    // specific_activated_streams_ has already contained new split activated stream
    int64_t new_split_stream = node_stream_pair.second;
    if (IsActivated(new_split_stream)) {
      continue;
    }
    specific_activated_streams_.emplace(new_split_stream);
    if (new_split_stream == active_op->GetStreamId()) {
      GELOGD("Node[%s] can not active its own stream[%ld].", active_op->GetName().c_str(), new_split_stream);
      continue;
    }
    new_active_streams.emplace(static_cast<uint32_t>(new_split_stream));
    active_streams.assign(new_active_streams.begin(), new_active_streams.end());
    (void)AttrUtils::SetListInt(active_op, ATTR_NAME_ACTIVE_STREAM_LIST, active_streams);
  }

  return SUCCESS;
}

bool StreamAllocator::IsActivated(int64_t stream_id) const {
  const auto &iter = split_stream_id_to_logic_stream_id_.find(stream_id);
  if (iter == split_stream_id_to_logic_stream_id_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Find original stream_id failed, split_stream_id=%ld", stream_id);
    GELOGE(INTERNAL_ERROR, "[CheckActivated][Check] Find original stream_id failed, split_stream_id=%ld", stream_id);
    return false;
  }
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    auto op_desc = node->GetOpDesc();
    std::vector<uint32_t> active_streams;
    if (op_desc == nullptr || !AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
      continue;
    }
    if (std::find(active_streams.begin(), active_streams.end(), stream_id) != active_streams.end()) {
      GELOGD("Stream: %ld is activated by %s.", stream_id, node->GetName().c_str());
      return true;
    }
  }
  return false;
}

// Iteraotor loop :
// StreamSwitch  ->  StreamActive
// FpBp loop:
// StreamSwitch  ->  AssignAdd  ->  StreamActive
static NodePtr FindSwitchNodeBeforeLoopActiveNode(const NodePtr &active_node) {
  for (auto pre_node : active_node->GetInControlNodes()) {
    if (pre_node->GetType() == STREAMSWITCH) {
      return pre_node;
    }
    for (auto pre_pre_node : pre_node->GetInControlNodes()) {
      if (pre_pre_node->GetType() == STREAMSWITCH) {
        return pre_pre_node;
      }
    }
  }
  return nullptr;
}

Status StreamAllocator::SetActiveStreamsForLoop(bool is_before_split_stream,
                                                const std::vector<std::set<int64_t>> &split_streams) {
  std::vector<uint32_t> loop_active_streams;
  for (int64_t stream_id = 0; stream_id < stream_num_; stream_id++) {
    if (specific_activated_streams_.count(stream_id) == 0) {
      loop_active_streams.emplace_back(static_cast<uint32_t>(stream_id));
    }
  }
  std::map<int64_t, NodePtr> stream_id_to_last_node;
  std::set<int64_t> streams_skip_iterator_event;
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    int64_t stream_id = node->GetOpDesc()->GetStreamId();
    if (find(loop_active_streams.begin(), loop_active_streams.end(), stream_id) != loop_active_streams.end()) {
      stream_id_to_last_node[stream_id] = node;
      // last node in stream which has streamswitch or IF may be not execute, it will cause block if add event on them
      if (node->GetOpDesc()->GetType() == STREAMSWITCH) {
        streams_skip_iterator_event.insert(stream_id);
      }
    }
  }
  // Set the stream that needs to be activated
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    bool is_loop_active = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_IS_LOOP_ACTIVE, is_loop_active);
    if (is_loop_active) {
      std::vector<std::string> activated_label_list;

      NodePtr pre_switch_node = FindSwitchNodeBeforeLoopActiveNode(node);
      if (pre_switch_node == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Find switch node before loop active node %s fail",
                           node->GetName().c_str());
      }

      if (!AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_ACTIVE_LABEL_LIST, activated_label_list) ||
          activated_label_list.empty()) {
        GE_CHK_BOOL_EXEC(AttrUtils::SetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, loop_active_streams),
                         REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)",
                                            ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                                            node->GetName().c_str(), node->GetType().c_str());
                             GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                                    node->GetName().c_str(), node->GetType().c_str());
                             return FAILED);
        for (const auto &stream_id : loop_active_streams) {
          GELOGI("Active stream %u for node: %s.", stream_id, node->GetName().c_str());
        }

        // In switch group optimize case, some data input branch may exec slowly.
        // when condition input branch judge false and some switch has no false branch,
        // In this condition, data branch has no synchronize point,
        // it may cause some stream activated by iterator next step when this stream still alive.
        // If above situation happen, active message will lose, cause process block in next iteration.
        // In order to avoid this abnormal happen,
        // add event between each last node and iterator switch node
        GELOGI("there are %zu next iterator target streams has streamswitch node.", streams_skip_iterator_event.size());
        for (auto iter : stream_id_to_last_node) {
          auto stream_id = iter.first;
          if (!is_before_split_stream && (static_cast<size_t>(stream_id) < split_streams.size()) &&
              split_streams[stream_id].empty()) {
            GELOGI("Skip stream %ld which has add event to next iterator active node before split stream", iter.first);
            continue;
          }
          if (streams_skip_iterator_event.find(iter.first) != streams_skip_iterator_event.end()) {
            GELOGI("Skip stream %ld which has streamswitch node when adding event to next iterator active node",
                   iter.first);
            continue;
          }
          if (iter.second->GetOwnerComputeGraph()->GetParentGraph() != nullptr) {
            GELOGI("Skip stream %ld which is last node in subgraph when adding event to next iterator active node",
                   iter.first);
            continue;
          }
          StreamUtils::AddSendEventId(iter.second, event_num_, node_to_send_events_);
          StreamUtils::AddRecvEventId(pre_switch_node, event_num_, node_to_recv_events_);
          event_num_++;
        }

        break;
      }
    }
  }

  return SUCCESS;
}

Status StreamAllocator::CheckStreamActived() const {
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::vector<uint32_t> active_streams;
    if (AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
      uint32_t stream_id = static_cast<uint32_t>(node->GetOpDesc()->GetStreamId());
      auto iter = find(active_streams.begin(), active_streams.end(), stream_id);
      if (iter != active_streams.end()) {
        REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) cannot active its own stream %u, check invalid ",
                           node->GetName().c_str(), node->GetType().c_str(), stream_id);
        GELOGE(FAILED, "[Check][Param] Node %s cannot active its own stream %u.", node->GetName().c_str(), stream_id);
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status StreamAllocator::ReuseEvent(
    bool send_to, const std::unordered_map<std::string, ge::NodePtr> &name_to_node_map,
    const std::unordered_map<ge::NodePtr, std::vector<std::pair<std::string, uint32_t>>> &node_to_event_id) {
  for (const auto &node_event_id : node_to_event_id) {
    ge::NodePtr curr_node = node_event_id.first;
    NodePtr send_node = send_to ? curr_node : nullptr;
    NodePtr recv_node = send_to ? nullptr : curr_node;
    for (const auto &event_pair : node_event_id.second) {
      auto peer_node_iter = name_to_node_map.find(event_pair.first);
      if (peer_node_iter == name_to_node_map.end()) {
        GELOGE(PARAM_INVALID, "[Get][Node]Name:%s.", event_pair.first.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Failed to find node, name:%s.", event_pair.first.c_str());
        return PARAM_INVALID;
      }
      recv_node = send_to ? peer_node_iter->second : recv_node;
      send_node = send_to ? send_node : peer_node_iter->second;
      GE_CHECK_NOTNULL(send_node);
      GE_CHECK_NOTNULL(recv_node);
      auto event_id = GetIntersection(node_to_send_events_[send_node], node_to_recv_events_[recv_node]);
      uint32_t new_event = event_pair.second + event_num_;
      if (event_id.empty()) {
        GELOGI("[Check][Optimized]Send:%s, recv:%s.", send_node->GetName().c_str(), recv_node->GetName().c_str());
        continue;
      } else if (event_id.size() != 1) {
        GELOGW("[Check][Event]More than one event are found between %s and %s, event num:%zu.",
               send_node->GetName().c_str(), recv_node->GetName().c_str(), event_id.size());
      }
      uint32_t old_event = event_id[0];
      auto reuse_event_id = [](std::vector<uint32_t> &event_list, uint32_t event_old, uint32_t event_new) -> void {
        event_list.erase(std::remove(event_list.begin(), event_list.end(), event_old), event_list.cend());
        event_list.push_back(event_new);
        return;
      };
      reuse_event_id(node_to_send_events_[send_node], old_event, new_event);
      reuse_event_id(node_to_recv_events_[recv_node], old_event, new_event);
      GELOGI("[Reuse][Event]Replace event successfully, send node:%s, recv node:%s, old id:%u, new id:%u.",
             send_node->GetName().c_str(), recv_node->GetName().c_str(), old_event, new_event);
    }
  }
  return ge::SUCCESS;
}

// Refresh (events && notifies) to reuse (events && notifies)
Status StreamAllocator::RefreshEventsAndNotifiesWithReuse() {
  GELOGI("[Refresh][Events]Refresh events with reuse, stream num:%ld, original event num:%u, original notify num:%u.",
         stream_num_, event_num_, notify_num_);

  GE_CHK_STATUS_RET(ReuseEventForMultiDims(EventType::kEvent, node_to_send_events_, node_to_recv_events_),
                    "Reuse event for multi dims failed.");
  GE_CHK_STATUS_RET(ReuseEventForMultiDims(EventType::kNotify, node_to_send_notifies_, node_to_recv_notifies_),
                    "Reuse notify for multi dims failed.");
  if (event_num_ <= kEventReuseThreshold) {
    GELOGI("[Check][ReuseThreshold]Event used num is %u, less than %u, skip reuse.",
           event_num_, kEventReuseThreshold);
    return SUCCESS;
  }
  std::unordered_map<std::string, NodePtr> name_to_node_map;
  std::unordered_map<NodePtr, std::vector<std::pair<std::string, uint32_t>>> node_to_send;
  std::unordered_map<NodePtr, std::vector<std::pair<std::string, uint32_t>>> node_to_recv;
  Status ret = ParseAllNodeEventMultiplexing(whole_graph_, name_to_node_map, node_to_send, node_to_recv);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][AllNodeEventMultiplexing]Graph:%s.", whole_graph_->GetName().c_str());
    return ret;
  }
  if (node_to_send.empty() && node_to_recv.empty()) {
    return SUCCESS;
  }

  ret = ReuseEvent(true, name_to_node_map, node_to_send);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Reuse][Event]Phase:Send, graph:%s.", whole_graph_->GetName().c_str());
    return ret;
  }

  ret = ReuseEvent(false, name_to_node_map, node_to_recv);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Reuse][Event]Phase:Recv, graph:%s.", whole_graph_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Failed to reuse event, phase:Recv, graph:%s.", whole_graph_->GetName().c_str());
    return ret;
  }

  event_num_ = 0U;
  Status status = StreamUtils::RefreshContinuousEvents(event_num_, node_to_send_events_, node_to_recv_events_);
  if (status != SUCCESS) {
    GELOGE(status, "[Refresh][ContinuousEvents]Graph:%s.", whole_graph_->GetName().c_str());
    return status;
  }
  GE_ASSERT_SUCCESS(
      StreamUtils::RefreshContinuousEvents(event_num_, attached_node_to_send_events_, attached_node_to_recv_events_),
      "[Refresh][ContinuousEvents]Graph:%s.", whole_graph_->GetName().c_str());
  GELOGI("[Refresh][EventsAndNotifies]RefreshEventsAndNotifiesWithReuse successfully, event num:%u, notify num:%u.",
         event_num_, notify_num_);
  return SUCCESS;
}

// only one dim can execute in per model execution, so event id of different dims can be reused
// we refresh event id of every dim to start from 0, for example:
// dim0: 0, 1, 2, 3         --->        0, 1, 2, 3
// dim1: 4, 5, 6, 7, 8      --->        0, 1, 2, 3, 4
// dim2: 9, 10, 11          --->        0, 1, 2
// at last event_num_ is the max event_id of all dims, so above event_num_ is 4
Status StreamAllocator::ReuseEventForMultiDims(
    const EventType event_type, std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  // event_type_ == kEvent means does not open the notify option, so it is no need to do ReuseNotifyForMultiDims
  if ((event_type == EventType::kNotify) && (event_type_ == EventType::kEvent)) {
    GELOGI("event_type_ = false, there is no need to ReuseNotifyForMultiDims.");
    return SUCCESS;
  }
  uint32_t max_event_id = 0U;
  std::map<uint32_t, uint32_t> event_seen;
  for (const auto &node : whole_graph_->GetAllNodes()) {
    // current only has one Case
    if (node->GetType() != CASE) {
      continue;
    }
    const auto &func_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(func_desc);
    if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
      GELOGD("Not multi-batch, skip Case: %s.", node->GetName().c_str());
      continue;
    }

    GELOGI("Start to reuse event for multi dims of Case: %s.", node->GetName().c_str());
    uint32_t cur_max_event_id = max_event_id;
    for (const auto &sub_name : func_desc->GetSubgraphInstanceNames()) {
      const auto &subgraph = whole_graph_->GetSubgraph(sub_name);
      GE_CHECK_NOTNULL(subgraph);
      uint32_t cur_event_id = max_event_id;
      if (event_type == EventType::kEvent) {
        GE_CHK_STATUS_RET(BuildEventReuseMapOfOneDim(subgraph, 0U, cur_event_id, event_seen),
                          "Build event reuse map of subgraph: %s failed.", sub_name.c_str());
      } else if (event_type == EventType::kNotify) {
        GE_CHK_STATUS_RET(BuildNotifyReuseMapOfOneDim(subgraph, 0U, cur_event_id, event_seen),
                          "Build notify reuse map of subgraph: %s failed.", sub_name.c_str());
      }
      GELOGD("Subgraph: %s, cur_event_id: %u, cur_max_event_id: %u, max_event_id: %u.", sub_name.c_str(), cur_event_id,
             cur_max_event_id, max_event_id);
      if (cur_event_id > cur_max_event_id) {
        cur_max_event_id = cur_event_id;
      }
    }
    max_event_id = cur_max_event_id;
    GELOGI("Success to reuse event for multi dims of %s, max event id: %u.", node->GetName().c_str(), max_event_id);
  }

  GE_CHK_STATUS_RET(BuildEventReuseMapOutOfDims(event_type, max_event_id, node_to_send_events,
                                                node_to_recv_events, event_seen),
                    "Build event reuse map out of dims failed.");

  GE_CHK_STATUS_RET(StreamUtils::RefreshEventByReuseMap(event_seen, node_to_send_events, node_to_recv_events),
                    "Refresh event by reuse map failed.");

  return SUCCESS;
}

std::vector<uint32_t> StreamAllocator::GetSyncIdWithinSameGraph(
    const ComputeGraphPtr &graph, const std::vector<uint32_t> &sync_ids,
    const std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &peer_sync_info) const {
  std::vector<uint32_t> sync_id_within_same_graph;
  for (const uint32_t cur_sync_id : sync_ids) {
    const auto node = StreamUtils::GetNodeFromSyncId(cur_sync_id, peer_sync_info);
    if ((node != nullptr) && (node->GetOwnerComputeGraph() == graph)) {
      sync_id_within_same_graph.emplace_back(cur_sync_id);
    }
  }
  return sync_id_within_same_graph;
}

Status StreamAllocator::BuildEventReuseMapOfOneDim(const ComputeGraphPtr &subgraph, uint32_t depth,
                                                   uint32_t &cur_event_id,
                                                   std::map<uint32_t, uint32_t> &event_seen) const {
  if (depth > kMaxSubgraphDepth) {
    REPORT_INNER_ERR_MSG("E19999", "Check invalid depth %u, exceed max subgraph depth %u.", depth, kMaxSubgraphDepth);
    GELOGE(FAILED, "[Check][Param] Check invalid depth %u, exceed max subgraph depth %u.", depth, kMaxSubgraphDepth);
    return FAILED;
  }

  for (const auto &node : subgraph->GetDirectNode()) {
    const auto &iter_send = node_to_send_events_.find(node);
    if (iter_send != node_to_send_events_.cend()) {
      const std::vector<uint32_t> &send_events =
          GetSyncIdWithinSameGraph(subgraph, iter_send->second, node_to_recv_events_);
      BuildEventReuseMap(EventType::kEvent, send_events, event_seen, cur_event_id);
    }

    const auto &iter_recv = node_to_recv_events_.find(node);
    if (iter_recv != node_to_recv_events_.cend()) {
      const std::vector<uint32_t> &recv_events =
          GetSyncIdWithinSameGraph(subgraph, iter_recv->second, node_to_send_events_);
      BuildEventReuseMap(EventType::kEvent, recv_events, event_seen, cur_event_id);
    }

    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &sub_name : op_desc->GetSubgraphInstanceNames()) {
      const auto &subgraph_in_branch = whole_graph_->GetSubgraph(sub_name);
      GE_CHECK_NOTNULL(subgraph_in_branch);
      GE_CHK_STATUS_RET(BuildEventReuseMapOfOneDim(subgraph_in_branch, depth + 1, cur_event_id, event_seen),
                        "Build event reuse map of subgraph: %s failed.", sub_name.c_str());
    }
  }

  return SUCCESS;
}

Status StreamAllocator::BuildNotifyReuseMapOfOneDim(const ComputeGraphPtr &subgraph, uint32_t depth,
                                                    uint32_t &cur_notify_id,
                                                    std::map<uint32_t, uint32_t> &notify_seen) const {
  if (depth > kMaxSubgraphDepth) {
    REPORT_INNER_ERR_MSG("E19999", "Check invalid depth %u, exceed max subgraph depth %u.", depth, kMaxSubgraphDepth);
    GELOGE(FAILED, "[Check][Param] Check invalid depth %u, exceed max subgraph depth %u.", depth, kMaxSubgraphDepth);
    return FAILED;
  }

  for (const auto &node : subgraph->GetDirectNode()) {
    const auto &iter_send = node_to_send_notifies_.find(node);
    if (iter_send != node_to_send_notifies_.cend()) {
      const std::vector<uint32_t> &send_notifies =
          GetSyncIdWithinSameGraph(subgraph, iter_send->second, node_to_recv_notifies_);
      BuildEventReuseMap(EventType::kNotify, send_notifies, notify_seen, cur_notify_id);
    }

    const auto &iter_recv = node_to_recv_notifies_.find(node);
    if (iter_recv != node_to_recv_notifies_.cend()) {
      const std::vector<uint32_t> &recv_notifies =
          GetSyncIdWithinSameGraph(subgraph, iter_recv->second, node_to_send_notifies_);
      BuildEventReuseMap(EventType::kNotify, recv_notifies, notify_seen, cur_notify_id);
    }

    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &sub_name : op_desc->GetSubgraphInstanceNames()) {
      const auto &subgraph_in_branch = whole_graph_->GetSubgraph(sub_name);
      GE_CHECK_NOTNULL(subgraph_in_branch);
      GE_CHK_STATUS_RET(BuildNotifyReuseMapOfOneDim(subgraph_in_branch, depth + 1, cur_notify_id, notify_seen),
                        "Build notify reuse map of subgraph: %s failed.", sub_name.c_str());
    }
  }

  return SUCCESS;
}

Status StreamAllocator::BuildEventReuseMapOutOfDims(
    const EventType event_type, uint32_t max_event_id,
    const map<NodePtr, vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    const map<NodePtr, vector<uint32_t>, NodeCompareKey> &node_to_recv_events,
    std::map<uint32_t, uint32_t> &event_seen) {
  // no event reuse
  if (max_event_id == 0U) {
    return SUCCESS;
  }

  GELOGI("Start to refresh event out of multi dims, max_event_id: %u.", max_event_id);
  // Refresh send event id
  for (auto &one_pair : node_to_send_events) {
    const std::vector<uint32_t> &send_events = one_pair.second;
    BuildEventReuseMap(event_type, send_events, event_seen, max_event_id);
  }

  // Refresh recv event id
  for (auto &one_pair : node_to_recv_events) {
    const std::vector<uint32_t> &recv_events = one_pair.second;
    BuildEventReuseMap(event_type, recv_events, event_seen, max_event_id);
  }

  if (event_type == EventType::kEvent) {
    event_num_ = max_event_id;
    GELOGI("After refresh event for multi dims, event num is: %u.", max_event_id);
  } else if (event_type == EventType::kNotify) {
    notify_num_ = max_event_id;
    GELOGI("After refresh notify for multi dims, event num is: %u.", max_event_id);
  }

  return SUCCESS;
}

// Refresh notifies to continuous notifies
Status StreamAllocator::RefreshContinuousNotifies() {
  // Establish a mapping relationship from old to new notify id
  std::map<uint32_t, uint32_t> old_to_new_notifies;
  uint32_t new_notify_id = 0U;
  for (const auto &one_pair : node_to_send_notifies_) {
    for (const auto &notify_id : one_pair.second) {
      if (old_to_new_notifies.find(notify_id) == old_to_new_notifies.end()) {
        old_to_new_notifies[notify_id] = new_notify_id;
        new_notify_id++;
      }
    }
  }

  GE_CHK_STATUS_RET(
      StreamUtils::RefreshEventByReuseMap(old_to_new_notifies, node_to_send_notifies_, node_to_recv_notifies_),
      "Refresh continuous notify failed.");

  notify_num_ = static_cast<uint32_t>(old_to_new_notifies.size());

  return SUCCESS;
}

Status StreamAllocator::InsertSyncSendEventNode(const NodePtr &node, const std::vector<uint32_t> &event_id_list,
                                                int64_t stream_id, int32_t &total_num,
                                                std::unordered_map<std::string, uint32_t> &sync_event_name) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GE_CHECK_NOTNULL(node->GetInControlAnchor());
  GE_CHECK_NOTNULL(node->GetOutControlAnchor());
  for (const uint32_t event_id : event_id_list) {
    std::string recv_node_name = whole_graph_->GetName().append("_Recv_").append(to_string(event_id));
    auto iter = sync_event_name.find(recv_node_name);
    if (iter == sync_event_name.end()) {
      sync_event_name[recv_node_name] = 1;
    } else {
      recv_node_name = recv_node_name.append("_Reuse_").append(to_string(iter->second));
      ++(iter->second);
    }
    OpDescPtr op_desc_ptr = MakeShared<OpDesc>(recv_node_name, RECV);
    GE_CHECK_NOTNULL(op_desc_ptr);

    const int64_t temp_stream_id = stream_id;
    op_desc_ptr->SetStreamId(temp_stream_id);
    GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc_ptr, RECV_ATTR_EVENT_ID, event_id),
                     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s for op:%s(%s) failed, event_id:%u,",
                                        RECV_ATTR_EVENT_ID.c_str(),
                                        node->GetName().c_str(), node->GetType().c_str(), event_id);
                         GELOGE(FAILED, "[Set][Attr] %s for op:%s(%s) failed, event_id:%u,",
                                RECV_ATTR_EVENT_ID.c_str(), node->GetName().c_str(), node->GetType().c_str(), event_id);
                         return FAILED);
    (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                std::move(std::vector<std::string>()));
    NodePtr recv_node = node->GetOwnerComputeGraph()->AddNode(op_desc_ptr);
    GE_CHECK_NOTNULL(recv_node);
    GE_CHECK_NOTNULL(recv_node->GetOutControlAnchor());
    Status status = GraphUtils::AddEdge(recv_node->GetOutControlAnchor(), node->GetInControlAnchor());
    if (status != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node %s to node %s failed",
                         recv_node->GetName().c_str(), node->GetName().c_str());
      GELOGE(status, "[Add][Edge] for node %s and node %s failed.", recv_node->GetName().c_str(),
             node->GetName().c_str());
      return status;
    }
    // topo id设置为当前图的最后，后面插完所有event可能会调用topo排序重新排序
    recv_node->GetOpDesc()->SetId(new_topo_id_);
    ++total_num;
    GELOGI("Insert recv event node %s topo id %ld event id %u before node: %s with stream %ld.", recv_node_name.c_str(),
           new_topo_id_, event_id, node->GetName().c_str(), temp_stream_id);
    new_topo_id_++;
  }
  return SUCCESS;
}

Status StreamAllocator::InsertSyncRecvEventNode(const NodePtr &node, const std::vector<uint32_t> &event_id_list,
                                                int64_t stream_id, int32_t &total_num,
                                                std::unordered_map<std::string, uint32_t> &sync_event_name) {
  GE_ASSERT_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  for (const uint32_t event_id : event_id_list) {
    std::string send_node_name = whole_graph_->GetName() + "_Send_" + to_string(event_id);
    auto iter = sync_event_name.find(send_node_name);
    if (iter == sync_event_name.end()) {
      sync_event_name[send_node_name] = 1;
    } else {
      send_node_name = send_node_name.append("_Reuse_").append(to_string(iter->second));
      ++(iter->second);
    }
    OpDescPtr op_desc_ptr = MakeShared<OpDesc>(send_node_name, SEND);
    GE_CHECK_NOTNULL(op_desc_ptr);

    const int64_t temp_stream_id = stream_id;
    op_desc_ptr->SetStreamId(temp_stream_id);
    GE_CHK_BOOL_EXEC(AttrUtils::SetInt(op_desc_ptr, SEND_ATTR_EVENT_ID, event_id), GELOGE(FAILED, "SetInt failed.");
        return FAILED);
    (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                std::move(std::vector<std::string>()));
    NodePtr send_node = node->GetOwnerComputeGraph()->InsertNode(node, op_desc_ptr);
    GE_CHECK_NOTNULL(send_node);
    GE_CHECK_NOTNULL(send_node->GetInControlAnchor());
    Status status = GraphUtils::AddEdge(node->GetOutControlAnchor(), send_node->GetInControlAnchor());
    if (status != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node %s to node %s failed", node->GetName().c_str(),
                         send_node->GetName().c_str());
      GELOGE(status, "[Add][Edge] for node %s and node %s failed.", node->GetName().c_str(),
             send_node->GetName().c_str());
      return status;
    }
    // topo id设置为当前图的最后，后面插完所有event可能会调用topo排序重新排序
    send_node->GetOpDesc()->SetId(new_topo_id_);
    ++total_num;
    GELOGI("Insert send event node: %s topo id %ld event id %u after node: %s with stream: %ld.",
           send_node_name.c_str(), new_topo_id_, event_id, node->GetName().c_str(), temp_stream_id);
    new_topo_id_++;
  }
  return SUCCESS;
}

Status StreamAllocator::InsertSyncSendNotifyNode(const NodePtr &node, int32_t &total_num,
                                                 std::unordered_map<std::string, uint32_t> &sync_notify_name) {
  // Add the node corresponding to the send notify
  const auto send_notify_id_list = StreamUtils::GetSyncIdList(node, node_to_send_notifies_);

  for (const uint32_t notify_id : send_notify_id_list) {
    std::string send_node_name = whole_graph_->GetName().append("_Send_Notify_").append(to_string(notify_id));
    auto iter = sync_notify_name.find(send_node_name);
    if (iter == sync_notify_name.end()) {
      sync_notify_name[send_node_name] = 1;
    } else {
      send_node_name = send_node_name.append("_Reuse_").append(to_string(iter->second));
      ++(iter->second);
    }
    OpDescPtr op_desc_ptr = MakeShared<OpDesc>(send_node_name, SENDNOTIFY);
    GE_CHECK_NOTNULL(op_desc_ptr);

    int64_t temp_stream_id = node->GetOpDesc()->GetStreamId();
    op_desc_ptr->SetStreamId(temp_stream_id);
    GE_CHK_BOOL_EXEC(
        AttrUtils::SetInt(op_desc_ptr, SEND_ATTR_NOTIFY_ID, notify_id),
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s for op:%s(%s) failed, notify_id:%u,", SEND_ATTR_NOTIFY_ID.c_str(),
                           node->GetName().c_str(), node->GetType().c_str(), notify_id);
            GELOGE(FAILED, "[Set][Attr] %s for op:%s(%s) failed, notify_id:%u,", SEND_ATTR_NOTIFY_ID.c_str(),
                   node->GetName().c_str(), node->GetType().c_str(), notify_id);
            return FAILED);
    (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                std::move(std::vector<std::string>()));
    NodePtr send_node = node->GetOwnerComputeGraph()->InsertNode(node, op_desc_ptr);
    GE_CHECK_NOTNULL(send_node);
    GE_CHECK_NOTNULL(send_node->GetInControlAnchor());
    Status status = GraphUtils::AddEdge(node->GetOutControlAnchor(), send_node->GetInControlAnchor());
    if (status != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node %s to node %s failed", node->GetName().c_str(),
                         send_node->GetName().c_str());
      GELOGE(status, "[Add][Edge] for node %s and node %s failed.", node->GetName().c_str(),
             send_node->GetName().c_str());
      return status;
    }
    // topo id设置为当前图的最后，后面插完所有event可能会调用topo排序重新排序
    send_node->GetOpDesc()->SetId(new_topo_id_);
    ++total_num;
    GELOGI("Insert send notify %u topo id %ld after node: %s.", notify_id, new_topo_id_, node->GetName().c_str());
    new_topo_id_++;
  }
  return SUCCESS;
}

Status StreamAllocator::InsertSyncRecvNotifyNode(const NodePtr &node, int32_t &total_num,
                                                 std::unordered_map<std::string, uint32_t> &sync_notify_name) {
  // Add the node corresponding to the recv notify
  const auto recv_notify_id_list = StreamUtils::GetSyncIdList(node, node_to_recv_notifies_);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  GE_CHECK_NOTNULL(node->GetInControlAnchor());
  GE_CHECK_NOTNULL(node->GetOutControlAnchor());
  for (const uint32_t notify_id : recv_notify_id_list) {
    std::string recv_node_name = whole_graph_->GetName().append("_Recv_Notify_").append(to_string(notify_id));
    auto iter = sync_notify_name.find(recv_node_name);
    if (iter == sync_notify_name.end()) {
      sync_notify_name[recv_node_name] = 1;
    } else {
      recv_node_name = recv_node_name.append("_Reuse_").append(to_string(iter->second));
      ++(iter->second);
    }
    OpDescPtr op_desc_ptr = MakeShared<OpDesc>(recv_node_name, RECVNOTIFY);
    GE_CHECK_NOTNULL(op_desc_ptr);

    int64_t temp_stream_id = node->GetOpDesc()->GetStreamId();
    op_desc_ptr->SetStreamId(temp_stream_id);
    GE_CHK_BOOL_EXEC(
        AttrUtils::SetInt(op_desc_ptr, RECV_ATTR_NOTIFY_ID, notify_id),
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s for op:%s(%s) failed, notify_id:%u,", RECV_ATTR_NOTIFY_ID.c_str(),
                           node->GetName().c_str(), node->GetType().c_str(), notify_id);
            GELOGE(FAILED, "[Set][Attr] %s for op:%s(%s) failed, notify_id:%u,", RECV_ATTR_NOTIFY_ID.c_str(),
                   node->GetName().c_str(), node->GetType().c_str(), notify_id);
            return FAILED);
    (void)AttrUtils::SetListStr(op_desc_ptr, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                std::move(std::vector<std::string>()));
    NodePtr recv_node = node->GetOwnerComputeGraph()->AddNode(op_desc_ptr);
    GE_CHECK_NOTNULL(recv_node);
    GE_CHECK_NOTNULL(recv_node->GetOutControlAnchor());
    Status status = GraphUtils::AddEdge(recv_node->GetOutControlAnchor(), node->GetInControlAnchor());
    if (status != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge from node %s to node %s failed", recv_node->GetName().c_str(),
                         node->GetName().c_str());
      GELOGE(status, "[Add][Edge] for node %s and node %s failed.", recv_node->GetName().c_str(),
             node->GetName().c_str());
      return status;
    }
    // topo id设置为当前图的最后，后面插完所有event可能会调用topo排序重新排序
    recv_node->GetOpDesc()->SetId(new_topo_id_);
    ++total_num;
    GELOGI("Insert recv notify %u topo id %ld before node: %s.", notify_id, new_topo_id_, node->GetName().c_str());
    new_topo_id_++;
  }
  return SUCCESS;
}

// Insert the real send/recv node in the graph
Status StreamAllocator::GenerateSyncEventNodes(bool change_topo) {
  std::unordered_map<std::string, uint32_t> sync_event_name;
  std::unordered_map<std::string, uint32_t> sync_notify_name;
  int32_t total_send = 0;
  int32_t total_recv = 0;
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    // Add the node corresponding to the recv event
    GE_ASSERT_SUCCESS(InsertSyncRecvEventNode(node, StreamUtils::GetSyncIdList(node, node_to_send_events_),
                                              node->GetOpDesc()->GetStreamId(), total_recv, sync_event_name));
    const auto attached_streams = node->GetOpDesc()->GetAttachedStreamIds();
    if (attached_streams.size() == 1U) {
      GE_ASSERT_SUCCESS(InsertSyncRecvEventNode(node, StreamUtils::GetSyncIdList(node, attached_node_to_send_events_),
                                                attached_streams[0], total_recv, sync_event_name));
    }

    const auto &send_iter = attached_node_to_stream_id_to_send_event_id_.find(node);
    if (send_iter != attached_node_to_stream_id_to_send_event_id_.end()) {
      for (const auto &stream_iter : send_iter->second) {
        const auto stream_id = stream_iter.first;
        const auto &event_id_list = stream_iter.second;
        GE_ASSERT_SUCCESS(InsertSyncRecvEventNode(node, event_id_list, stream_id, total_recv, sync_event_name));
      }
    }

    // Add the node corresponding to the send event
    GE_ASSERT_SUCCESS(InsertSyncSendEventNode(node, StreamUtils::GetSyncIdList(node, node_to_recv_events_),
                                              node->GetOpDesc()->GetStreamId(), total_send, sync_event_name));
    if (attached_streams.size() == 1U) {
      GE_ASSERT_SUCCESS(InsertSyncSendEventNode(node, StreamUtils::GetSyncIdList(node, attached_node_to_recv_events_),
                                                attached_streams[0], total_send, sync_event_name));
    }
    const auto &recv_iter = attached_node_to_stream_id_to_recv_event_id_.find(node);
    if (recv_iter != attached_node_to_stream_id_to_recv_event_id_.end()) {
      for (const auto &stream_iter : recv_iter->second) {
        const auto stream_id = stream_iter.first;
        const auto &event_id_list = stream_iter.second;
        GE_ASSERT_SUCCESS(InsertSyncSendEventNode(node, event_id_list, stream_id, total_send, sync_event_name));
      }
    }

    if (event_type_ == EventType::kNotify) {
      // Add the node corresponding to the recv notify
      GE_CHK_STATUS_RET(InsertSyncRecvNotifyNode(node, total_recv, sync_notify_name),
                        "Insert recv notify nodes failed.");
      // Add the node corresponding to the send notify
      GE_CHK_STATUS_RET(InsertSyncSendNotifyNode(node, total_send, sync_notify_name),
                        "Insert send notify nodes failed.");
    }
  }
  GE_ASSERT_TRUE(total_send == total_recv);

  if (change_topo) {
    GE_ASSERT_SUCCESS(whole_graph_->InsertGraphEvents(),
                      "[Insert][GraphEvents] Graph ReorderEventNodes failed, graph:%s,",
                      whole_graph_->GetName().c_str());
  } else {
    // 只topo排序，但不重新设置id
    GE_ASSERT_SUCCESS(whole_graph_->ReorderEventNodes(), "graph %s ReorderEventNodes failed",
                      whole_graph_->GetName().c_str());
    for(const auto &sub_graph : whole_graph_->GetAllSubgraphs()) {
      GE_ASSERT_SUCCESS(sub_graph->ReorderEventNodes(), "sub graph %s ReorderEventNodes failed",
                        sub_graph->GetName().c_str());
    }
  }

  return SUCCESS;
}

void StreamAllocator::DumpEvents(const EventType event_type,
                                 std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                 std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) const {
  std::map<int64_t, std::vector<NodePtr>> after_refresh_stream_nodes;
  for (const auto &node : whole_graph_->GetNodes(whole_graph_->GetGraphUnknownFlag())) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, continue);
    int64_t stream_id = node->GetOpDesc()->GetStreamId();
    after_refresh_stream_nodes[stream_id].emplace_back(node);
  }

  for (const auto &one_pair : after_refresh_stream_nodes) {
    int64_t stream_id = one_pair.first;
    GELOGD("After RefreshRealStream: stream %ld.", stream_id);

    for (const auto &node : one_pair.second) {
      if (node == nullptr || node->GetOpDesc() == nullptr) {
        continue;
      }
      std::string send_event_str;
      for (const auto &send_event_id : node_to_send_events[node]) {
        send_event_str += " " + to_string(send_event_id);
      }
      if (!send_event_str.empty()) {
        GELOGI("node: %s, id: %ld, stream id: %ld, send %ss:%s.", node->GetName().c_str(), node->GetOpDesc()->GetId(),
               node->GetOpDesc()->GetStreamId(), GetEventTypeStr(event_type).c_str(), send_event_str.c_str());
      }

      std::string recv_event_str;
      for (const auto &recv_event_id : node_to_recv_events[node]) {
        recv_event_str += " " + to_string(recv_event_id);
      }
      if (!recv_event_str.empty()) {
        GELOGI("node: %s, id: %ld, stream id: %ld, recv %ss:%s.", node->GetName().c_str(), node->GetOpDesc()->GetId(),
               node->GetOpDesc()->GetStreamId(), GetEventTypeStr(event_type).c_str(), recv_event_str.c_str());
      }
    }
  }
}

Status StreamAllocator::GetMaxStreamAndTask(bool huge_stream, uint32_t &max_stream_count,
                                            uint32_t &max_task_count) const {
  uint32_t stream_type = RT_NORMAL_STREAM;
  if (huge_stream) {
    stream_type = RT_HUGE_STREAM;
  }
  rtError_t ret = rtGetMaxStreamAndTask(stream_type, &max_stream_count, &max_task_count);
  if (ret != RT_ERROR_NONE) {
    if (!huge_stream) {
      REPORT_INNER_ERR_MSG("E19999", "call rtGetMaxStreamAndTask fail, ret:%d, stream_type:%u,", static_cast<int32_t>(ret),
                        stream_type);
      GELOGE(FAILED,
             "[Call][RtGetMaxStreamAndTask] Get max stream and task count by rts failed, ret:%d, stream_type:%u,",
             static_cast<int32_t>(ret), stream_type);
    }
    return FAILED;
  }
  GELOGD("Allowed max stream count: %u, max task count per stream: %u.", max_stream_count, max_task_count);

  return SUCCESS;
}

void StreamAllocator::AddTaskNum(const NodePtr &node, int64_t &task_num, size_t task_size,
                                 bool is_attached_stream) const {
  auto assigned_task_num = StreamUtils::GetAssignedTaskNum(node, is_attached_stream);
  if (assigned_task_num > 0L) {
    task_num += assigned_task_num;
  } else {
    task_num += static_cast<int64_t>(task_size);
  }
  GELOGD("node %s, assigned_task_num %lld, generated task size %zu", node->GetName().c_str(), assigned_task_num,
         task_size);
}

Status StreamAllocator::AddEventPair(const NodePtr &send_node, const NodePtr &recv_node,
                                     Nodes2SyncInfos &nodes_2_send_sync_infos,
                                     Nodes2SyncInfos &nodes_2_recv_sync_infos) {
  GE_ASSERT_NOTNULL(send_node);
  GELOGI("Add send event %u for node %s", event_num_, send_node->GetName().c_str());
  StreamUtils::AddSendEventId(send_node, event_num_, nodes_2_send_sync_infos);

  GE_ASSERT_NOTNULL(recv_node);
  GELOGI("Add recv event %u for node %s", event_num_, recv_node->GetName().c_str());
  StreamUtils::AddRecvEventId(recv_node, event_num_, nodes_2_recv_sync_infos);
  ++event_num_;
  return SUCCESS;
}

Status StreamAllocator::AddEventPairBetweenAttachedAndMain(const NodePtr &send_node, const NodePtr &recv_node,
                                                           int64_t pre_stream_id,
                                                           Node2AttachedStreamId2EventId &nodes_2_send_event,
                                                           Nodes2SyncInfos &nodes_2_recv_event) {
  GE_ASSERT_NOTNULL(send_node);
  GELOGI("Add send event %u for attached stream %ld node %s", event_num_, pre_stream_id, send_node->GetName().c_str());
  auto &attached_stream_id_to_send_event_id = nodes_2_send_event[send_node];
  attached_stream_id_to_send_event_id[pre_stream_id].emplace_back(event_num_);

  GE_ASSERT_NOTNULL(recv_node);
  GELOGI("Add recv event %u for node %s", event_num_, recv_node->GetName().c_str());
  StreamUtils::AddRecvEventId(recv_node, event_num_, nodes_2_recv_event);
  ++event_num_;
  return SUCCESS;
}

Status StreamAllocator::AddAttachedStreamEventPair(const NodePtr &send_node, const NodePtr &recv_node,
                                                   int64_t pre_stream_id, int64_t next_stream_id,
                                                   Node2AttachedStreamId2EventId &nodes_2_send_event,
                                                   Node2AttachedStreamId2EventId &nodes_2_recv_event) {
  GE_ASSERT_NOTNULL(send_node);
  GELOGI("Add send event %u for attached stream %ld node %s", event_num_, pre_stream_id, send_node->GetName().c_str());
  auto &attached_stream_id_to_send_event_id = nodes_2_send_event[send_node];
  attached_stream_id_to_send_event_id[pre_stream_id].emplace_back(event_num_);

  GE_ASSERT_NOTNULL(recv_node);
  GELOGI("Add recv event %u for attached stream %ld node %s", event_num_, next_stream_id, recv_node->GetName().c_str());
  auto &attached_stream_id_to_recv_event_id = nodes_2_recv_event[recv_node];
  attached_stream_id_to_recv_event_id[next_stream_id].emplace_back(event_num_);
  ++event_num_;
  return SUCCESS;
}

Status StreamAllocator::AddEventIdWhenStreamSplit(const StreamSplitSyncInfo &stream_split_sync_info) {
  const auto &recv_node =
      stream_split_sync_info.not_use_cur ? stream_split_sync_info.not_cur : stream_split_sync_info.cur_node;
  GE_CHECK_NOTNULL(recv_node);
  const auto &send_node = stream_split_sync_info.pre_node;
  if (stream_split_sync_info.split_for_attached_stream) {
    GE_ASSERT_SUCCESS(AddAttachedStreamEventPair(
        send_node, recv_node, stream_split_sync_info.pre_stream_id, stream_split_sync_info.next_stream_id,
        attached_node_to_stream_id_to_send_event_id_, attached_node_to_stream_id_to_recv_event_id_));
  } else {
    GE_ASSERT_SUCCESS(AddEventPair(send_node, recv_node, node_to_send_events_, node_to_recv_events_));
  }
  return SUCCESS;
}

Status StreamAllocator::SetActiveNodeStreamLabel(const ge::NodePtr &node, const std::string &label,
                                                 std::set<std::string> &new_active_stream_labels) const {
  GE_CHECK_NOTNULL(node);
  const OpDescPtr tmp_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_desc);
  GE_ASSERT_TRUE(AttrUtils::SetStr(tmp_desc, ge::ATTR_NAME_STREAM_LABEL, label), "[Set][Attr] %s fail for op:%s(%s)",
                 ATTR_NAME_STREAM_LABEL.c_str(), node->GetName().c_str(), node->GetType().c_str());
  new_active_stream_labels.insert(label);
  return SUCCESS;
}

Status StreamAllocator::AddActiveNodes(const NodePtr &switch_node,
                                       const std::vector<std::string> &ori_active_label_list,
                                       std::vector<std::string> &active_label_list,
                                       std::vector<NodePtr> &added_active_nodes) {
  size_t label_num = ori_active_label_list.size();
  auto &has_set_labels = switch_to_has_set_labels_[switch_node];
  auto &new_active_stream_labels = switch_to_new_active_stream_labels_[switch_node];
  for (size_t i = 0; i < label_num; i++) {
    const std::string &active_label = ori_active_label_list[i];
    if (labeled_streams_.find(active_label) == labeled_streams_.end()) {
      if (new_active_stream_labels.find(active_label) != new_active_stream_labels.end()) {
        active_label_list.emplace_back(active_label);
        continue;
      }
      REPORT_INNER_ERR_MSG("E19999", "can not find stream label:%s", active_label.c_str());
      GELOGE(FAILED, "[Check][Param] can not find stream label %s", active_label.c_str());
      return FAILED;
    }
    if (labeled_streams_[active_label].size() <= 1) {
      active_label_list.emplace_back(active_label);
      continue;
    }

    if (has_set_labels.find(active_label) != has_set_labels.end()) {
      active_label_list.emplace_back(active_label);
      continue;
    }

    has_set_labels.insert(active_label);

    std::string name = switch_node->GetName() + "_" + STREAMACTIVE + "_" + std::to_string(i);
    GELOGI("Create StreamActive op %s after node %s.", name.c_str(), switch_node->GetName().c_str());
    OpDescPtr active_op_desc = MakeShared<OpDesc>(name, STREAMACTIVE);
    GE_CHECK_NOTNULL(active_op_desc);
    NodePtr active_node = switch_node->GetOwnerComputeGraph()->AddNode(active_op_desc);
    GE_CHECK_NOTNULL(active_node);

    for (NodePtr &node : switch_node->GetOutControlNodes()) {
      OpDescPtr op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::string stream_label;
      // If GetStr failed, stream_label is empty.
      (void)AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL, stream_label);
      if (stream_label != active_label) {
        continue;
      }
      GE_CHECK_NOTNULL(switch_node->GetOutControlAnchor());
      if (switch_node->GetOutControlAnchor()->Unlink(node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Unlink %s to %s failed", switch_node->GetName().c_str(), node->GetName().c_str());
        GELOGE(FAILED, "[Unlink][Nodes] %s to %s failed.", switch_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
      GE_CHECK_NOTNULL(active_node->GetOutControlAnchor());
      if (active_node->GetOutControlAnchor()->LinkTo(node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Link %s to %s failed", active_node->GetName().c_str(), node->GetName().c_str());
        GELOGE(FAILED, "[Link][Nodes] %s to %s failed.", active_node->GetName().c_str(), node->GetName().c_str());
        return FAILED;
      }
    }

    if (SetSwitchBranchNodeLabel(active_node, name) != SUCCESS) {
      GELOGE(FAILED, "[Set][SwitchBranchNodeLabel] failed, node:%s.", active_node->GetName().c_str());
      return FAILED;
    }
    if (SetActiveNodeStreamLabel(active_node, name, new_active_stream_labels) != SUCCESS) {
      GELOGE(FAILED, "[Set][StreamLabel] failed, node:%s.", active_node->GetName().c_str());
      return FAILED;
    }
    if (SetActiveLabelList(active_node, {active_label}) != SUCCESS) {
      GELOGE(FAILED, "[Set][ActiveLabelList] failed, node:%s.", active_node->GetName().c_str());
      return FAILED;
    }
    if (SetActiveStreamList(active_node, active_label) != SUCCESS) {
      GELOGE(FAILED, "[Set][ActiveStreamList] failed, node:%s.", active_node->GetName().c_str());
      return FAILED;
    }

    added_active_nodes.emplace_back(active_node);
    active_node->GetOpDesc()->SetId(new_topo_id_);
    new_topo_id_++;
    active_label_list.emplace_back(name);
  }
  return SUCCESS;
}

Status StreamAllocator::SetActiveStreamList(const NodePtr &active_node, const std::string &active_label) {
  if (labeled_streams_.find(active_label) == labeled_streams_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Can not find stream label:%s", active_label.c_str());
    GELOGE(FAILED, "[Check][Param] Can not find stream label %s.", active_label.c_str());
    return FAILED;
  }
  std::set<int64_t> &streams = labeled_streams_[active_label];
  std::vector<int64_t> active_streams(streams.cbegin(), streams.cend());
  GE_CHECK_NOTNULL(active_node->GetOpDesc());
  if (!AttrUtils::SetListInt(active_node->GetOpDesc(), ATTR_NAME_ACTIVE_STREAM_LIST, active_streams)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                       active_node->GetName().c_str(), active_node->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s failed for op:%s(%s).", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
           active_node->GetName().c_str(), active_node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
