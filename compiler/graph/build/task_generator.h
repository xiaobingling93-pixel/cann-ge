/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_TASK_GENERATOR_H_
#define GE_GRAPH_BUILD_TASK_GENERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "framework/common/types.h"
#include "graph/ge_local_context.h"
#include "graph/build/profiling_task_utils.h"
#include "graph/compute_graph.h"
#include "graph/model.h"
#include "proto/task.pb.h"
#include "common/thread_pool/thread_pool.h"
#include "common/preload/model/pre_model_utils.h"

namespace ge {
class TaskGenerator {
 public:
  TaskGenerator() = default;
  TaskGenerator(const TaskGenerator &) = delete;
  TaskGenerator &operator=(const TaskGenerator &) = delete;
  virtual ~TaskGenerator();
  TaskGenerator(uint8_t *var_mem_base, uint64_t var_mem_size, RunContext *run_context);
  /**
   * 对graph内的节点生成task，并设置到model上
   * @param graph
   * @param session_id
   * @param model
   * @return
   */
  Status GetTaskInfo(const ComputeGraphPtr &graph, uint64_t session_id, Model &model);
  /**
   * profiling task的查找接口
   * @param graph
   * @param profiling_point
   * @return
   */
  Status FindProfilingNodeIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point);
  Status GenModelTaskDef(const ComputeGraphPtr &graph, uint64_t session_id, Model &model);
  Status GenerateTaskForNodes(const std::vector<Node *> nodes);
  Status ReGetTaskInfo(const ComputeGraphPtr &comp_graph);
  std::unordered_map<int64_t, std::vector<domi::TaskDef>> &MutableNodeId2TaskDefs();
 private:
  /**
   * task 生成接口
   * @param graph
   * @param task_def_list
   * @return
   */
  Status GenerateTask(const ComputeGraphPtr &graph, Model &model);
  Status UpdateTaskDef();
  Status SaveFusionNodes(std::map<int64_t, std::vector<NodePtr>> &fusion_nodes, const std::vector<Node *> nodes) const;
  Status GenTaskForFusionNodes(const std::map<int64_t, std::vector<NodePtr>> &fusion_nodes);
  Status GenerateTaskForFusionNode(Node *const node, const std::map<int64_t, std::vector<NodePtr>> &fusion_nodes,
                                   std::unordered_set<Node *> &fusion_nodes_seen);
  Status PrepareForGenerateTask(const ComputeGraphPtr &graph);
  Status GenerateTaskForFftsNode(Node *ffts_node, const std::string &tag,
                                 std::vector<domi::TaskDef> &task_def_list_per_node,
                                 const GEThreadLocalContext &ge_context,
                                 const error_message::ErrorManagerContext &error_context, int32_t device_id);
  Status GenerateTaskForNormalNode(Node *const node, const std::string &tag,
                                   std::vector<domi::TaskDef> &task_def_list_per_node,
                                   const GEThreadLocalContext &ge_context,
                                   const error_message::ErrorManagerContext &error_context, int32_t device_id);
  Status GenTaskForNormalNode(Node *const node, const std::string &tag,
                              std::vector<domi::TaskDef> &task_def_list_per_node);
  Status FilterCandidatesNodes(const ComputeGraphPtr &graph);
  Status SetKernelInfo();
  Status GenTaskForPartiallySupportedNode(const NodePtr &node, RunContext &context,
                                          std::vector<domi::TaskDef> &tasks) const;
  Status GenTaskForNodeByAliasEngine(const NodePtr &node, RunContext &context, std::vector<domi::TaskDef> &tasks) const;
  Status AddModelTaskToModel(const domi::ModelTaskDef &model_task_def, uint64_t session_id, ge::Model &model,
                             RunContext &run_context) const;
  Status UpdateAnchorStatus(const NodePtr &node) const;
  Status UpdateOpIsVarAttr(const OpDescPtr &op_desc, uint64_t session_id) const;
  Status MarkNodeAndSetIndex(const ComputeGraphPtr &graph) const;
  // Mark first and last op according to the same stream and engine
  Status MarkFirstAndLastOps(const std::vector<Node *> &nodes, bool is_single_stream,
                             std::unordered_set<Node *> &target_nodes) const;
  Status MarkFirstAndLastOpsForGraph(const ComputeGraphPtr &graph, std::unordered_set<Node *> &target_nodes) const;

  Status InitZeroCopyInfo(const ComputeGraphPtr &graph, const PreRuntimeParam &runtime_param);
  Status GenZeroCopyTable(const OpDescPtr &op_desc, uint32_t &search_id, const bool is_input);
  Status InitRuntimeParams(const ModelPtr &model_ptr, PreRuntimeParam &runtime_param);
  Status SetAnchorsOffset(const NodePtr &owner_node, const bool is_input, const uint32_t index,
                          const PreRuntimeParam &runtime_param, const OpDescPtr &op_desc);
  Status SetOpOffset(const OpDescPtr &op_desc, const bool is_input, const int64_t offset,
                     const uint32_t offset_to_id);
  Status SetNetOutputNodeInAnchorAndPeerOffset(const InDataAnchorPtr &in_anchor, const PreRuntimeParam &runtime_param,
                                               SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol);
  Status SetNetOutputNodePeerNodeOffset(const NodePtr &peer_node, const bool is_input, uint32_t index,
                                        const int64_t ori_offset, const uint32_t offset_to_id,
                                        const PreRuntimeParam &runtime_param);

  uint8_t *var_mem_base_ = nullptr;
  uint64_t var_mem_size_ = 0;
  RunContext *run_context_ = nullptr;
  ProfilingPoint profiling_point_;
  // record node need gen task
  std::vector<Node *> nodes_;
  std::unordered_set<Node *> fusion_nodes_seen_;
  std::unordered_map<int64_t , std::vector<domi::TaskDef>> node_id_2_node_tasks_;
  // fusion node场景下用来记录model task def插入的node顺序
  std::vector<int64_t> fusion_ordered_node_list_;
  // fusion node场景下用来记录task对应的node name， 不一定是产生该task的node name
  std::vector<std::string> fusion_task_node_name_list_;
  std::unique_ptr<ThreadPool> thread_pool_{nullptr};
  std::unique_ptr<ThreadPool> ffts_inner_thread_pool_{nullptr};
  std::mutex ffts_mutex_;
  uint64_t session_id_{0U};
  // record node name by taskdefs one by one, name may be duplicate when more than one task is generated
  std::list<std::string> op_names_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_TASK_GENERATOR_H_
