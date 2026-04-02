/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_EXECUTE_MODEL_EXECUTOR_H
#define GE_GRAPH_EXECUTE_MODEL_EXECUTOR_H

#include <thread>

#include "common/model/executor.h"
#include "graph/execute/graph_executor.h"

namespace ge {
class ModelExecutor : public Executor {
 public:
  ModelExecutor() = default;
  /// @ingroup ge
  /// @brief graph executor init
  /// @param [in] options user config params
  /// @return Status result of function
  Status Initialize(const std::map<std::string, std::string> &options, const uint64_t session_id);

  /// @ingroup ge
  /// @brief graph executor finalize
  /// @return Status result of function
  Status Finalize();

  /// @ingroup ge
  /// @brief Load mode for graph.
  /// @param [in] ge_root_model: root model of graph compiled.
  /// @param [in] GraphNode: node of graph.
  /// @return Status result of function
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                   const aclrtStream stream = nullptr) override;

  /// @ingroup ge
  /// @brief Unload mode for graph.
  /// @param [in] ge_root_model: root model of graph compiled.
  /// @param [in] graph_id: graph identifier.
  /// @return Status result of function
  Status UnloadGraph(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) override;

  /// @brief Push model execution params to queue.
  /// @param [in] RunArgs of for model execution.
  /// @return Status result of function
  Status PushRunArgs(const std::shared_ptr<RunArgs> &args) override;

  /// @ingroup ge
  /// @brief Run graph for synchronize model.
  /// @param [in] graph_node: node of graph.
  /// @param [in] graph_id: graph identifier.
  /// @param [in] inputs: input data for the graph running.
  /// @param [out] outputs: output data of the graph running
  /// @return Status result of function
  Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                  const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) override;

  /// @ingroup ge
  /// @brief Run graph for NN synchronize model.
  /// @param [in] graph_node: node of graph.
  /// @param [in] graph_id: graph identifier.
  /// @param [in] stream: Stream for model running.
  /// @param [in] inputs: input data for the graph running.
  /// @param [out] outputs: output data of the graph running
  /// @return Status result of function
  Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, aclrtStream const stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) override;
  
   Status ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                                         aclrtStream const stream, const std::vector<gert::Tensor> &inputs,
                                         std::vector<gert::Tensor> &outputs) override;

  /**
   * @ingroup ge
   * @brief Update feature memory base after load graph
   * @param [in] graph_node: node of graph.
   * @param [in] mem_base: graph feature memory base(without input and output).
   * @param [in] size: memory size
   * @return Status result of function
   */
  Status UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base, const size_t size) override;

  /// @ingroup ge
  /// @brief Start executor run thread.
  void StartRunThread();

  /// @ingroup ge
  /// @brief Run graph for distributed model.
  /// @param [in] graph_node: node of graph.
  /// @param [in] va virtual memory
  /// @param [in] new_pa new physical memory
  /// @param [in] len the lens of va to remap
  /// @return Status result of function
  Status PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                    const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) override;

 private:
  static Status GetDeviceMemorySize(size_t &free_mem, size_t &total_mem_size);

  void AddGraphNode(const GraphId graph_id, const GraphNodePtr &graph_node);
  void RemoveGraphNode(const GraphId graph_id);

  Status ModelLoad(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node, const aclrtStream stream = nullptr);
  Status MallocFixedFeatureMemoryIfNeed(const GraphNodePtr &graph_node, const GeRootModelPtr &ge_root_model,
                                        const aclrtStream stream) const;
  static Status MallocByDiffAllocator(const uint64_t session_id,
                                      const aclrtStream stream,
                                      const FeatureMemoryPtr &fixed_feature_mem,
                                      const rtMemType_t rt_mem_type,
                                      const GeRootModelPtr &ge_root_model);
  static Status FreeFixedFeatureMemoryIfNeed(const GeRootModelPtr &ge_root_model);
  Status GetStreamNum(const GeRootModelPtr &ge_root_model, uint32_t &stream_num, uint64_t &hccl_follow_stream) const;
  Status GetEventNum(const GeRootModelPtr &ge_root_model, uint32_t &event_num) const;
  static Status UnloadModel(const GeRootModelPtr &ge_root_model, const uint32_t graph_id);
  static Status UnloadPneModel(const uint32_t model_id, const uint64_t session_id, const uint32_t graph_id);
  bool ReleaseMemory(const GeRootModelPtr &ge_root_model, const GraphNodePtr &loaded_graph_node) const;
  bool ReleaseModel(const GeRootModelPtr &ge_root_model, const GraphNodePtr &loaded_graph_node) const;
  Status CheckAndReleaseMemory(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);
  Status CheckAndReleaseStream(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);
  Status CheckAndReleaseEvent(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node);
  static Status GetMemoryInfo(size_t &free);

  Status CheckFreeMemory(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                                bool &is_enough, bool &release_all) const;
  Status GetMemorySizeAfterReuse(const std::vector<GeModelPtr> &ge_models, const GraphNodePtr &graph_node,
                                        int64_t &sum_size, bool &reuse) const;

  void RunThread();
  void StopQueue();
  void ReturnError(const RunAsyncCallbackV2 &callback,
      const Status ret, const std::string &log_info) const;
  bool DoReleaseModel(const GeRootModelPtr &ge_root_model, const GraphNodePtr &loaded_graph_node) const;

  bool init_flag_{false};
  uint64_t session_id_{0U};
  GraphExecutor graph_executor_;

  std::mutex mutex_;
  std::map<GraphId, GraphNodePtr> graph_nodes_;

  std::thread run_thread_;
  std::atomic_bool thread_run_flag_{false};
  BlockingQueue<std::shared_ptr<RunArgs>> run_args_q_;
};
}
#endif // GE_GRAPH_EXECUTE_MODEL_EXECUTOR_H
