/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_STREAM_STREAM_UTILS_H_
#define GE_GRAPH_BUILD_STREAM_STREAM_UTILS_H_

#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "ge_common/ge_api_types.h"
#include "graph/any_value.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
struct Subgraph;
using SubgraphPtr = std::shared_ptr<Subgraph>;

constexpr int64_t kInvalidStream = -1;
constexpr int64_t kMainStream = 0;
constexpr int64_t kDefaultMaxParalleNum = 1;

struct Subgraph {
  std::string name;
  int64_t stream_id = kInvalidStream;

  const SubGraphInfo &subgraph_info;
  const EngineConf &engine_conf;
  int64_t max_parallel_num = kDefaultMaxParalleNum;

  SubgraphPtr reused_subgraph = nullptr;

  Subgraph(const SubGraphInfo &sub_info, const EngineConf &conf)
      : subgraph_info(sub_info), engine_conf(conf) {}
};

class StreamUtils {
 public:
  static Status ConvertSubgraphs(const ComputeGraphPtr &graph, const Graph2SubGraphInfoList &subgraph_map,
                                 const std::map<std::string, EngineConfPtr> &engine_confs,
                                 const std::map<std::string, int32_t> &max_parallel_num,
                                 std::vector<SubgraphPtr> &subgraphs);
  static SubgraphPtr GetTopPrioritySubgraph(const std::set<SubgraphPtr> &subgraphs);
  static std::map<std::string, EngineConfPtr> GetEngineConfs();
  static const std::map<std::string, PriorityEnum> &GetEnginePriority();

  static bool IsDynamicSubGraph(const ComputeGraphPtr &graph);
  static void SetStreamId(const OpDescPtr &op_desc, const int64_t new_stream_id, const bool is_attached_stream,
                          const int64_t origin_stream_id);
  static int64_t GetAssignedTaskNum(const NodePtr &node, bool is_attached_stream);
  static bool IsHcclOp(const ge::OpDesc *const op_desc);

  static bool IsEngineSkip(const Subgraph &subgraph);
  static bool IsEngineAttach(const Subgraph &subgraph);
  static bool IsEngineIndependent(const Subgraph &subgraph);
  static bool HasStreamLabel(const Subgraph &subgraph);
  static bool HasUserStreamLabel(const Subgraph &subgraph);
  static bool HasStreamLabelOrUserStreamLabel(const NodePtr &node);
  static bool HasAssignedStream(const Subgraph &subgraph);
  static bool HasAssignedUserStream(const Subgraph &subgraph);
  static bool IsEventWaitNode(const ge::NodePtr &node);

  static bool EnableSingleStream();
  static bool EnableDynamicShapeMultiStream();
  static bool EnableCvParallel();

  static Status TransStrToMap(const std::string &map_str, std::map<int64_t, int64_t> &result);
  static std::string TransMapToStr(const std::map<int64_t, int64_t> &map);

  static Status TransStrToVec(const std::string &vec_str, std::vector<int64_t> &result);
  static std::string TransVecToStr(const std::vector<int64_t> &vec);

  static std::unordered_map<NodePtr, SubgraphPtr> InitEndSubgraphMap(const std::vector<SubgraphPtr> &subgraphs);
  static std::unordered_map<NodePtr, SubgraphPtr> InitPldSubgraphMap(const std::vector<SubgraphPtr> &subgraphs);

  static void AddSendEventId(const NodePtr &node, uint32_t event_id,
                             std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events);
  static void AddRecvEventId(const NodePtr &node, uint32_t event_id,
                             std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  static void RmvSendEventId(const NodePtr &node, uint32_t event_id,
                             std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events);
  static void RmvRecvEventId(const NodePtr &node, uint32_t event_id,
                             std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  static std::vector<uint32_t> GetSyncIdList(
      const NodePtr &node, std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_sync_ids);
  static NodePtr GetNodeFromSyncId(const uint32_t sync_id,
                                   const std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_sync_ids);

  static Status OptimizeBySendEvents(const std::map<int64_t, std::vector<NodePtr>> &stream_nodes,
                                     std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                     std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  static Status OptimizeByRecvEvents(const std::map<int64_t, std::vector<NodePtr>> &stream_nodes,
                                     std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                     std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  static Status RefreshContinuousEvents(uint32_t &event_num,
                                        std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                        std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);
  static Status RefreshEventByReuseMap(const std::map<uint32_t, uint32_t> &old_to_new_events,
                                       std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_send_events,
                                       std::map<NodePtr, std::vector<uint32_t>, NodeCompareKey> &node_to_recv_events);

  static Status TransUserStreamLabel(const ComputeGraphPtr &root_graph);

  static Status RunCustomStreamPass(const ComputeGraphPtr &root_graph, int64_t &next_stream_id);

 private:
  static std::mutex mutex_;
  static std::map<std::string, PriorityEnum> engine_priority_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_STREAM_STREAM_UTILS_H_