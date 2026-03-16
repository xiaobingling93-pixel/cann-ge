/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_utils.h"

#include "graph/utils/node_adapter.h"
#include "register/custom_pass_helper.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "api/gelib/gelib.h"
#include "register/register_custom_pass.h"
#include "register/custom_pass_context_impl.h"

namespace {
constexpr const ge::char_t *const kTrueStr = "true";
constexpr const ge::char_t *const kFalseStr = "false";
const std::set<std::string> hccl_op_types({ge::HCOMBROADCAST, ge::HCOMALLGATHER, ge::HCOMALLREDUCE,
                                           ge::HCOMREDUCESCATTER, ge::HCOMREDUCE, ge::HCOMALLTOALLV,
                                           ge::HCOMGATHERALLTOALLV, ge::HCOMALLTOALLVC, ge::HCOMALLTOALL});
}  // namespace

namespace ge {
std::mutex StreamUtils::mutex_;
std::map<std::string, PriorityEnum> StreamUtils::engine_priority_;

Status StreamUtils::ConvertSubgraphs(const ComputeGraphPtr &graph, const Graph2SubGraphInfoList &subgraph_map,
                                     const std::map<std::string, EngineConfPtr> &engine_confs,
                                     const std::map<std::string, int32_t> &max_parallel_num,
                                     std::vector<SubgraphPtr> &subgraphs) {
  const auto iter = subgraph_map.find(graph);
  GE_ASSERT_TRUE(iter != subgraph_map.end(), "Can not find graph: %s.", graph->GetName().c_str());

  const std::vector<SubGraphInfoPtr> &subgraph_infos = iter->second;
  for (auto &subgraph_info : subgraph_infos) {
    GE_CHECK_NOTNULL(subgraph_info);

    std::string subgraph_name;
    ComputeGraphPtr computer_graph = subgraph_info->GetSubGraph();
    if (computer_graph != nullptr) {
      subgraph_name = computer_graph->GetName();
    }

    const std::string &engine_name = subgraph_info->GetEngineName();
    auto engine_conf_iter = engine_confs.find(engine_name);
    GE_ASSERT_TRUE(engine_conf_iter != engine_confs.end(), "Can not find engine: %s.", engine_name.c_str());
    GE_CHECK_NOTNULL(engine_conf_iter->second);

    SubgraphPtr subgraph = MakeShared<Subgraph>(*subgraph_info, *engine_conf_iter->second);
    GE_CHECK_NOTNULL(subgraph);
    subgraph->name = subgraph_name;

    auto parallel_iter = max_parallel_num.find(engine_name);
    if (parallel_iter != max_parallel_num.end()) {
      subgraph->max_parallel_num = parallel_iter->second;
    }

    subgraphs.emplace_back(subgraph);
    GELOGI("subgraph: %s, max_parallel_num: %lld.", subgraph->name.c_str(), subgraph->max_parallel_num);
  }

  return SUCCESS;
}

SubgraphPtr StreamUtils::GetTopPrioritySubgraph(const std::set<SubgraphPtr> &subgraphs) {
  if (subgraphs.empty()) {
    return nullptr;
  }

  auto instance_ptr = ge::GELib::GetInstance();
  GE_ASSERT_NOTNULL(instance_ptr);
  SubgraphPtr priority_subgraph = *(subgraphs.begin());
  PriorityEnum current_priority = PriorityEnum::COST_10;
  for (auto &reusable_subgraph : subgraphs) {
    auto engine = instance_ptr->DNNEngineManagerObj().GetEngine(reusable_subgraph->engine_conf.id);
    GE_ASSERT_NOTNULL(engine);
    DNNEngineAttribute attr;
    engine->GetAttributes(attr);
    if (attr.compute_cost < current_priority) {
      current_priority = attr.compute_cost;
      priority_subgraph = reusable_subgraph;
    }
  }
  return priority_subgraph;
}

std::map<std::string, EngineConfPtr> StreamUtils::GetEngineConfs() {
  auto gelib = GELib::GetInstance();
  if (gelib == nullptr) {
    return {};
  }
  const auto &scheduler_confs = gelib->DNNEngineManagerObj().GetSchedulers();
  GELOGD("scheduler_confs size: %zu.", scheduler_confs.size());

  std::map<std::string, EngineConfPtr> engine_confs;
  for (const auto &item : scheduler_confs) {
    const SchedulerConf &scheduler = item.second;
    for (const auto &engine_pair : scheduler.cal_engines) {
      EngineConfPtr engine_conf = engine_pair.second;
      if (engine_conf != nullptr) {
        engine_confs[engine_pair.first] = engine_conf;
        GELOGI("Add engine: %s.", engine_pair.first.c_str());
      }
    }
  }
  return engine_confs;
}

const std::map<std::string, PriorityEnum> &StreamUtils::GetEnginePriority() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!engine_priority_.empty()) {
    return engine_priority_;
  }

  auto gelib = GELib::GetInstance();
  if (gelib == nullptr) {
    return engine_priority_;
  }
  for (const auto &engine_info : gelib->DNNEngineManagerObj().GetAllEngines()) {
    DNNEngineAttribute attr;
    engine_info.second->GetAttributes(attr);
    engine_priority_.emplace(engine_info.first, attr.compute_cost);
    GELOGI("Engine: %s, priority: %d.", engine_info.first.c_str(), static_cast<int32_t>(attr.compute_cost));
  }

  return engine_priority_;
}

bool StreamUtils::IsEngineSkip(const Subgraph &subgraph) { return subgraph.engine_conf.skip_assign_stream; }

bool StreamUtils::IsEngineAttach(const Subgraph &subgraph) { return subgraph.engine_conf.attach; }

bool StreamUtils::IsEngineIndependent(const Subgraph &subgraph) { return subgraph.engine_conf.independent; }

bool StreamUtils::HasStreamLabel(const Subgraph &subgraph) { return !subgraph.subgraph_info.GetStreamLabel().empty(); }

bool StreamUtils::HasUserStreamLabel(const Subgraph &subgraph) { return !subgraph.subgraph_info.GetUserStreamLabel().empty(); }

bool StreamUtils::HasStreamLabelOrUserStreamLabel(const ge::NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  return AttrUtils::HasAttr(op_desc, ATTR_NAME_STREAM_LABEL) ||
         AttrUtils::HasAttr(op_desc, public_attr::USER_STREAM_LABEL);
}

bool StreamUtils::HasAssignedStream(const Subgraph &subgraph) { return subgraph.stream_id != kInvalidStream; }

bool StreamUtils::HasAssignedUserStream(const Subgraph &subgraph) {
  return subgraph.stream_id != kInvalidStream && HasUserStreamLabel(subgraph);
}

bool StreamUtils::IsDynamicSubGraph(const ComputeGraphPtr &graph) {
  const auto &functional_node = graph->GetParentNode();
  if (functional_node == nullptr) {
    return false;
  }
  return graph->GetGraphUnknownFlag();
}

void StreamUtils::SetStreamId(const OpDescPtr &op_desc, const int64_t new_stream_id, const bool is_attached_stream,
                              const int64_t origin_stream_id) {
  // 外部保证op_desc非空
  if (is_attached_stream) {
    const auto origin_attached_streams = op_desc->GetAttachedStreamIds();
    std::vector<int64_t> new_attached_streams;
    for (const auto stream_id : origin_attached_streams) {
      if (stream_id == origin_stream_id) {
        new_attached_streams.emplace_back(new_stream_id);
      } else {
        new_attached_streams.emplace_back(stream_id);
      }
    }
    return op_desc->SetAttachedStreamIds(new_attached_streams);
  } else {
    return op_desc->SetStreamId(new_stream_id);
  }
}

bool StreamUtils::IsHcclOp(const ge::OpDesc *const op_desc) {
  const auto &lib_name = op_desc->GetOpKernelLibName();
  if (lib_name == ge::kEngineNameHccl) {
    return true;
  }
  const auto &op_type = op_desc->GetType();
  return hccl_op_types.find(op_type) != hccl_op_types.end();
}

bool StreamUtils::IsEventWaitNode(const ge::NodePtr &node) {
  const auto &node_type = node->GetTypePtr();
  return (strcmp(node_type, "Recv") == 0) || (strcmp(node_type, "RecvMem") == 0);
}

int64_t StreamUtils::GetAssignedTaskNum(const NodePtr &node, bool is_attached_stream) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  int64_t task_num = 0L;
  if (AttrUtils::GetInt(op_desc, ATTR_NAME_NODE_SQE_NUM, task_num)) {
    GELOGD("Node: %s, type: %s, sqe num: %lld.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), task_num);
  } else if (IsHcclOp(op_desc)) {
    if (is_attached_stream) {
      (void)AttrUtils::GetInt(op_desc, ATTR_NAME_HCCL_ATTACHED_TASK_NUM, task_num);
    } else {
      (void)AttrUtils::GetInt(op_desc, ATTR_NAME_HCCL_TASK_NUM, task_num);
    }
  }
  // event_wait在Distribute时下了wait和reset两个任务
  if ((task_num > 0U) && IsEventWaitNode(node)) {
    task_num++;
  }
  GELOGD("Node: %s, type: %s, task num: %lld.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), task_num);
  return task_num;
}

bool StreamUtils::EnableSingleStream() {
  std::string single_stream_str;
  (void)GetContext().GetOption(ENABLE_SINGLE_STREAM, single_stream_str);

  const std::set<std::string> stream_options = {"", kTrueStr, kFalseStr};
  if (stream_options.find(single_stream_str) == stream_options.end()) {
    GELOGW("The value %s of the %s option is invalid, it should be true or false.", single_stream_str.c_str(),
           ENABLE_SINGLE_STREAM);
  }

  GELOGI("Enable single stream: %s.", single_stream_str.c_str());
  return (single_stream_str == kTrueStr);
}

bool StreamUtils::EnableDynamicShapeMultiStream() {
  const char_t *enable_multi_stream_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, enable_multi_stream_env);
  if (enable_multi_stream_env != nullptr) {
    const std::string env_str(enable_multi_stream_env);
    if (env_str == "1") {
      GEEVENT("Enable multi-stream in dynamic graph.");
      return true;
    }
  }
  return false;
}

bool StreamUtils::EnableCvParallel() {
  std::string multi_stream_mode;
  if ((ge::GetContext().GetOption("ge.autoMultistreamParallelMode", multi_stream_mode) == ge::GRAPH_SUCCESS) &&
      (multi_stream_mode == "cv")) {
    GELOGI("auto multistream parallel mode is %s", multi_stream_mode.c_str());
    return true;
  }
  return false;
}

// trans string to map, "0:0,1:0,2:1" to {{0,0}, {1,0}, {2,1}}
Status StreamUtils::TransStrToMap(const std::string &map_str, std::map<int64_t, int64_t> &result) {
  std::stringstream ss(map_str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t pos = item.find(':');
    if (pos != std::string::npos) {
      int64_t key = -1L;
      GE_ASSERT_SUCCESS(ge::ConvertToInt64(item.substr(0, pos), key));
      int64_t value = -1L;
      GE_ASSERT_SUCCESS(ge::ConvertToInt64(item.substr(pos + 1), value));
      result[key] = value;
    }
  }
  return SUCCESS;
}

// trans map to string, {{0,0}, {1,0}, {2,1}} to "0:0,1:0,2:1"
std::string StreamUtils::TransMapToStr(const std::map<int64_t, int64_t> &map) {
  std::stringstream ss;
  bool first = true;
  for (const auto& pair : map) {
    if (!first) {
      ss << ",";
    }
    ss << pair.first << ":" << pair.second;
    first = false;
  }
  return ss.str();
}

// trans vector to string, {0,1,2} to "0,1,2"
std::string StreamUtils::TransVecToStr(const std::vector<int64_t> &vec) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) oss << ",";
    oss << vec[i];
  }
  return oss.str();
}

// trans string to vector, "0,1,2" to {0,1,2}
Status StreamUtils::TransStrToVec(const std::string &vec_str, std::vector<int64_t> &result) {
  std::istringstream iss(vec_str);
  std::string token;

  while (std::getline(iss, token, ',')) {
    int64_t val = -1L;
    GE_ASSERT_SUCCESS(ge::ConvertToInt64(token, val));
    result.push_back(val);
  }
  return SUCCESS;
}

std::unordered_map<NodePtr, SubgraphPtr> StreamUtils::InitEndSubgraphMap(const vector<SubgraphPtr> &subgraphs) {
  std::unordered_map<NodePtr, SubgraphPtr> end_subgraph_map;
  for (const auto &subgraph : subgraphs) {
    const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
    for (const auto &item : subgraph_info.GetEnd2PldMap()) {
      end_subgraph_map.emplace(item.first, subgraph);
    }
  }
  return end_subgraph_map;
}

std::unordered_map<NodePtr, SubgraphPtr> StreamUtils::InitPldSubgraphMap(const vector<SubgraphPtr> &subgraphs) {
  std::unordered_map<NodePtr, SubgraphPtr> pld_subgraph_map;
  for (const auto &subgraph : subgraphs) {
    const SubGraphInfo &subgraph_info = subgraph->subgraph_info;
    for (const auto &item : subgraph_info.GetPld2EndMap()) {
      pld_subgraph_map.emplace(item.first, subgraph);
    }
  }
  return pld_subgraph_map;
}

// Insert send event id on a node
void StreamUtils::AddSendEventId(const NodePtr &node, uint32_t event_id,
                                 std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events) {
  node_to_send_events[node].emplace_back(event_id);
}

// Insert recv event id on a node
void StreamUtils::AddRecvEventId(const NodePtr &node, uint32_t event_id,
                                 std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  node_to_recv_events[node].emplace_back(event_id);
}

// Remove send event id from a node
void StreamUtils::RmvSendEventId(const NodePtr &node, uint32_t event_id,
                                 std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events) {
  const auto find_it = node_to_send_events.find(node);
  if (find_it == node_to_send_events.end()) {
    return;
  }

  std::vector<uint32_t> &send_events = find_it->second;
  for (auto it = send_events.begin(); it != send_events.end(); ++it) {
    if (*it == event_id) {
      send_events.erase(it);
      return;
    }
  }
}

// Remove recv event id from a node
void StreamUtils::RmvRecvEventId(const NodePtr &node, uint32_t event_id,
                                 std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  const auto find_it = node_to_recv_events.find(node);
  if (find_it == node_to_recv_events.end()) {
    return;
  }

  std::vector<uint32_t> &recv_events = find_it->second;
  for (auto it = recv_events.begin(); it != recv_events.end(); ++it) {
    if (*it == event_id) {
      recv_events.erase(it);
      return;
    }
  }
}

// Get sync id list from a node
std::vector<uint32_t> StreamUtils::GetSyncIdList(
    const NodePtr &node, std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_sync_ids) {
  const auto find_it = node_to_sync_ids.find(node);
  if (find_it != node_to_sync_ids.end()) {
    return find_it->second;
  }
  return {};
}

// Get a specific sync node according to the sync id
NodePtr StreamUtils::GetNodeFromSyncId(
    const uint32_t sync_id, const std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_sync_ids) {
  for (const auto &one_pair : node_to_sync_ids) {
    const std::vector<uint32_t> &sync_ids = one_pair.second;
    for (const auto &cur_sync_id : sync_ids) {
      if (cur_sync_id == sync_id) {
        return one_pair.first;
      }
    }
  }
  GELOGI("GetNodeFromSyncId, sync_id:%u, return nullptr.", sync_id);
  return nullptr;
}

/* Optimization scenario: one stream has multiple send events in one node,
   and multiple nodes for recv events on another stream
   Example:
   Stream0            Stream1
     N1 - event|notify   -- > N1
       \                     |
        \                    v
          - -event|notify- > N2 */
Status StreamUtils::OptimizeBySendEvents(
    const std::map<int64_t, std::vector<NodePtr>> &stream_nodes,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  for (const auto &one_pair : stream_nodes) {
    // The nodes on a stream in order
    const std::vector<NodePtr> &nodes = one_pair.second;
    std::map<NodePtr, uint32_t> send_node_to_event_id;
    for (const auto &recv_node : nodes) {
      GE_CHECK_NOTNULL(recv_node);
      // Get all recv events of the current node, then traverse the event
      const auto recv_events = StreamUtils::GetSyncIdList(recv_node, node_to_recv_events);

      for (const uint32_t event_id : recv_events) {
        NodePtr send_node = StreamUtils::GetNodeFromSyncId(event_id, node_to_send_events);
        GE_CHECK_NOTNULL(send_node);

        /// If the record to the stream is found in the map,
        /// and the recv node is the node, then remove sync event
        if (send_node_to_event_id.find(send_node) == send_node_to_event_id.end()) {
          send_node_to_event_id[send_node] = event_id;
        } else {
          StreamUtils::RmvSendEventId(send_node, event_id, node_to_send_events);
          StreamUtils::RmvRecvEventId(recv_node, event_id, node_to_recv_events);
          GELOGI("Remove %u between node %s and node %s", event_id, send_node->GetName().c_str(),
                 recv_node->GetName().c_str());
        }
      }
    }
  }

  return SUCCESS;
}

/* Scenario: multiple send nodes on a stream sent to a single recv node on the destination stream
   Example:
   Stream0            Stream1
     N1 - -
     |    |
     |    - - event - - -
     |                  |
     V                  V
     N2 - - - event - > N2 */
Status StreamUtils::OptimizeByRecvEvents(
    const std::map<int64_t, std::vector<NodePtr>> &stream_nodes,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  for (const auto &one_pair : stream_nodes) {
    // The nodes on a stream in order
    const std::vector<NodePtr> &nodes = one_pair.second;
    std::map<NodePtr, uint32_t> recv_node_to_event_id;
    for (const auto &send_node : nodes) {
      GE_CHECK_NOTNULL(send_node);
      //  Get all send events of the current node, then traverse the event
      const auto send_id_list = StreamUtils::GetSyncIdList(send_node, node_to_send_events);

      for (const uint32_t event_id : send_id_list) {
        NodePtr recv_node = StreamUtils::GetNodeFromSyncId(event_id, node_to_recv_events);
        GE_CHECK_NOTNULL(recv_node);

        /// If the record to the stream is found in the map,
        /// and the send node is the node, then remove sync event
        std::map<NodePtr, uint32_t>::const_iterator it = recv_node_to_event_id.find(recv_node);
        if (it != recv_node_to_event_id.cend()) {
          uint32_t pre_event_id = it->second;
          NodePtr pre_send_node = StreamUtils::GetNodeFromSyncId(pre_event_id, node_to_send_events);
          GE_CHECK_NOTNULL(pre_send_node);

          StreamUtils::RmvSendEventId(pre_send_node, pre_event_id, node_to_send_events);
          StreamUtils::RmvRecvEventId(recv_node, pre_event_id, node_to_recv_events);
          GELOGI("Remove %u between node %s and node %s.", event_id, pre_send_node->GetName().c_str(),
                 recv_node->GetName().c_str());
        }
        recv_node_to_event_id[recv_node] = event_id;
      }
    }
  }

  return SUCCESS;
}

// Refresh events to continuous events
Status StreamUtils::RefreshContinuousEvents(
    uint32_t &event_num, std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  // Establish a mapping relationship from old to new event id
  std::map<uint32_t, uint32_t> old_to_new_events;
  uint32_t new_event_id = event_num;
  for (const auto &one_pair : node_to_send_events) {
    for (const auto &event_id : one_pair.second) {
      if (old_to_new_events.find(event_id) == old_to_new_events.end()) {
        old_to_new_events[event_id] = new_event_id;
        GELOGD("Refresh event id: %u to %u.", event_id, new_event_id);
        ++new_event_id;
      }
    }
  }

  GE_ASSERT_SUCCESS(RefreshEventByReuseMap(old_to_new_events, node_to_send_events, node_to_recv_events));

  event_num = new_event_id;
  GELOGI("[Refresh][ContinuousEvents] event num: %u", event_num);

  return SUCCESS;
}

Status StreamUtils::RefreshEventByReuseMap(
    const std::map<uint32_t, uint32_t> &old_to_new_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
    std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events) {
  if (old_to_new_events.empty()) {
    return SUCCESS;
  }

  // Refresh send event id
  for (auto &one_pair : node_to_send_events) {
    std::vector<uint32_t> &send_events = one_pair.second;
    for (size_t i = 0U; i < send_events.size(); i++) {
      std::map<uint32_t, uint32_t>::const_iterator find_it = old_to_new_events.find(send_events[i]);
      GE_ASSERT_TRUE(find_it != old_to_new_events.cend(), "Can not find send id: %u.", send_events[i]);
      send_events[i] = find_it->second;
    }
  }

  // Refresh recv event id
  for (auto &one_pair : node_to_recv_events) {
    std::vector<uint32_t> &recv_events = one_pair.second;
    for (size_t i = 0U; i < recv_events.size(); i++) {
      std::map<uint32_t, uint32_t>::const_iterator find_it = old_to_new_events.find(recv_events[i]);
      GE_ASSERT_TRUE(find_it != old_to_new_events.cend(), "Can not find recv id: %u.", recv_events[i]);
      recv_events[i] = find_it->second;
    }
  }

  return SUCCESS;
}

Status StreamUtils::TransUserStreamLabel(const ComputeGraphPtr &root_graph) {
  GE_ASSERT_NOTNULL(root_graph);
  for (const auto &node: root_graph->GetAllNodesPtr()) {
    GE_ASSERT_NOTNULL(node);
    std::string user_stream_label;
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    // todo use attr define after metadef submodule update
    if (AttrUtils::GetStr(op_desc, public_attr::USER_STREAM_LABEL, user_stream_label)) {
      std::string inner_stream_label;
      if (AttrUtils::GetStr(op_desc, ATTR_NAME_STREAM_LABEL,inner_stream_label)) {
        GELOGI(
          "Node %s(%s) has both user stream label %s and inner stream label: %s. User stream label will take effect.",
          op_desc->GetNamePtr(), op_desc->GetTypePtr(), user_stream_label.c_str(), inner_stream_label.c_str());
      }
      GELOGI(
        "Node %s(%s) has user stream label: %s. User stream label will take effect.",
        op_desc->GetNamePtr(), op_desc->GetTypePtr(), user_stream_label.c_str());
      AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, user_stream_label);
    }
  }
  return SUCCESS;
}

Status StreamUtils::RunCustomStreamPass(const ComputeGraphPtr &root_graph, int64_t &next_stream_id) {
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(root_graph);
  const int64_t orgin_max_stream_id = next_stream_id - 1;
  StreamPassContext context(orgin_max_stream_id);
  GE_TRACE_START(RunCustomStreamPass);
  GE_ASSERT_SUCCESS(CustomPassHelper::Instance().Run(graph, context, CustomPassStage::kAfterAssignLogicStream),
                    "Run allocate stream pass for graph [%s] failed, reason: %s.", root_graph->GetName().c_str(),
                    context.GetErrorMessage().GetString());
  GE_COMPILE_TRACE_TIMESTAMP_END(RunCustomStreamPass, "RunCustomPass_AfterAssignLogicStream");
  const int64_t new_stream_num = context.GetCurrMaxStreamId() - orgin_max_stream_id;
  GE_ASSERT_TRUE(new_stream_num >= 0);
  next_stream_id += new_stream_num;

  std::vector<int64_t> custom_logical_stream_ids;
  for (int64_t i = (orgin_max_stream_id + 1); i < next_stream_id; i++) {
    custom_logical_stream_ids.emplace_back(i);
  }
  GE_ASSERT_TRUE(AttrUtils::SetStr(root_graph, "_custom_logical_stream_ids",
                                   StreamUtils::TransVecToStr(custom_logical_stream_ids)));
  return SUCCESS;
}
}  // namespace ge
