/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_GRAPH_MANAGER_H_
#define GE_GRAPH_MANAGER_GRAPH_MANAGER_H_

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/blocking_queue.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api_types.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"
#include "graph/build/graph_builder.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/partition/engine_partitioner.h"
#include "graph/preprocess/graph_prepare.h"
#include "graph/tuning_utils.h"
#include "common/model/ge_model.h"
#include "common/model/executor.h"
#include "graph/resource_context_mgr.h"
#include "ge/ge_allocator.h"
#include "exe_graph/runtime/tensor.h"

namespace ge {
class GraphManager {
 public:
  GraphManager() = default;
  ~GraphManager();

  /// @ingroup ge_graph
  /// @brief graph manager init
  /// @param [in] options user config params
  /// @return Status result of function
  Status Initialize(const std::map<std::string, std::string> &options, Executor *executor = nullptr);

  /// @ingroup ge_graph
  /// @brief graph manager finalize
  /// @return Status result of function
  Status Finalize();

  /// @ingroup ge_graph
  /// @brief add specific graph
  /// @param [in] graph_id graph id
  /// @param [out] Graph output graph
  /// @return Status result of function
  Status AddGraph(const GraphId &graph_id, const Graph &graph, const std::map<std::string, std::string> &options,
                  const OmgContext &omg_context);

  Status ForkGraph(uint32_t origin_graph_id, uint32_t forked_graph_id);

  Status AddGraphForBuild(const GraphId &graph_id,
                          const ComputeGraphPtr &compute_graph,
                          const std::map<std::string, std::string> &options,
                          bool graph_normalized = false);
  Status InitDynamicParams(const ComputeGraphPtr &compute_graph,
                           const std::map<std::string, std::string> &graph_options) const;

  /// @ingroup ge_graph
  /// @brief add a copy graph
  /// @param [in] graph_id graph id
  /// @param [out] Graph output graph
  /// @return Status result of function
  Status AddGraphWithCopy(const GraphId &graph_id, const Graph &graph,
                          const std::map<std::string, std::string> &options, const OmgContext &omg_context);

  /// @ingroup ge_graph
  /// @brief remove specific graph
  /// @param [in] graph_id graph id
  /// @return Status result of function
  Status RemoveGraph(const GraphId &graph_id);

  /// @ingroup ge_graph
  /// @brief run specific graph
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  Status RunGraph(const GraphId &graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs,
    uint64_t session_id);
  Status RunGraph(const GraphId &graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  /// @ingroup ge_graph
  /// @brief run specific graph with specific session id and stream
  /// @param [in] graph_id graph id
  /// @param [in] stream specific stream
  /// @param [in] session_id session id
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  Status RunGraphWithStreamAsync(const GraphId &graph_id, const rtStream_t stream, uint64_t session_id,
                                 const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);

  Status ExecuteGraphWithStreamAsync(const GraphId &graph_id, const rtStream_t stream,
                                     const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  /// @ingroup ge_graph
  /// @brief build specific graph
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] models build result
  /// @return Status result of function
  Status BuildGraph(const GraphId &graph_id, const std::vector<GeTensor> &inputs, GeRootModelPtr &ge_root_model,
                    uint64_t session_id = 0, bool async = false);

  Status BuildGraphWithoutLoad(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                               GeRootModelPtr &ge_root_model, uint64_t session_id = 0, bool async = false);

  Status InnerLoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                        const rtStream_t stream = nullptr) const;

  Status LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options,
      const rtStream_t stream);

  Status BuildGraphForUnregisteredOp(const GraphId &graph_id, const std::vector<GeTensor> &inputs,
                                     GeRootModelPtr &ge_root_model, uint64_t session_id);

  /// @ingroup ge_graph
  /// @brief Save extra attribute to Model
  /// @param [in] model: Model attribues will save to.
  /// @param [in] type: type of OpDesc.
  /// @param [in] attrs: attributes of OpDesc
  /// @param [in] inputs: input tensor
  /// @param [in] outputs: output tensor
  /// @return: Status
  Status SaveParams(GeModel &model, const std::string &type, const std::map<std::string, GeAttrValue> &attrs,
                    const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs) const;

  /// @ingroup ge_graph
  /// @brief get variable value from the session with specific session id
  /// @param [in] sessionId session id
  /// @param [in] name op name
  /// @param [out] val out value tensor
  /// @return Status result of function
  Status GetVariable(const std::string &name, Tensor &val) const;

  /// @ingroup ge_graph
  /// @brief run graph async on session with specific session id
  /// @param [in] graph_id graph id
  /// @param [in] inputs input data
  /// @param [out] callback: callback while run graph async finish
  /// @return Status result of function
  Status RunGraphAsync(const GraphId &graph_id, std::vector<gert::Tensor> &&inputs,
                       uint64_t session_id, const RunAsyncCallbackV2 &callback);

  /// @ingroup ge_graph
  /// @brief me register the callback function to get the result of summary or checkpoin
  /// @param [in] key: summary or checkpoint
  /// @param [in] callbak: The real callback object of me
  /// @return Status result of function
  Status RegisterCallBackFunc(const std::string &key,
                              const std::function<Status(uint32_t,
                              const std::map<AscendString, gert::Tensor> &)> &callback);

  bool GetTrainFlag() const { return options_.train_graph_flag; }

  bool IsGraphNeedRebuild(uint32_t graph_id);

  Status GenerateInfershapeGraph(GraphId &graph_id);

  Status CheckGraphVaildBeforeExecute(const GraphId &graph_id, GraphNodePtr &graph_node) const;

  const std::map<std::string, std::string> *GetGraphOptions(uint32_t graph_id);

  void SetOptionsRunGraphFlag(bool run_graph_flag);

  Status GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables, Graph &graph) const;

  Status SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                       const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values) const;

  Status SaveCheckPointResult(const Graph &graph, const std::vector<Tensor> &outputs,
                              std::map<std::string, Tensor> &var_results) const;

  void RemoveGraphCount(GraphId graph_id);

  void IncreaseGraphCount(GraphId graph_id);

  void DecreaseGraphCount(GraphId graph_id);

  Status GetGraphCount(GraphId graph_id, uint32_t &count);

  void SetAddGraphCondition(GraphId graph_id, uint32_t cond);

  uint32_t GetAddGraphCondition(GraphId graph_id);

  void RemoveAddGraphCondition(GraphId graph_id);

  Status OptimizeGraph(const std::vector<GeTensor> &inputs, ComputeGraphPtr &compute_graph);

  Status BuildGraph(ComputeGraphPtr &compute_graph, GeRootModelPtr &ge_root_model);

  Status GetGraphsMemInfo(std::map<uint32_t, std::vector<uint64_t>> &graphs_mem_info) const;

  Status CompileGraph(uint32_t graph_id, uint64_t session_id, const vector<ge::Tensor> &inputs);

  Status GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary);

  Status SetConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status UpdateFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status CheckFixFeatureMemoryBaseHasBeenSet(const GraphNodePtr graph_node, const rtMemType_t rt_mem_type,
                                             bool &has_been_set, bool &user_alloc, void *&mem_base) const;

  Status SetFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory, size_t size);

  Status UpdateRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  static Status SetRunContext(const GraphNodePtr &graph_node);
  Status GetGraphNode(const GraphId &graph_id, GraphNodePtr &out) const;

  // GeSession约束RunGraph/RunGraphAsync/RunGraphWithStreamAsync不能混用
  Status GetRunGraphMode(uint32_t graph_id, RunGraphMode &mode) const;
  Status SetRunGraphMode(uint32_t graph_id, const RunGraphMode &mode);
  Status GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer);
  const std::vector<GraphId> &GetOrderedGraphIds() const;

  Status PaRemapped(const GraphId graph_id, const uint64_t va, const uint64_t new_pa, const uint64_t len,
                    std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) const;

  // temporary solution, set rebuild_ctrl from external to share variable change between different graph_manager object
  void SetExternalGraphRebuildStateCtrl(std::shared_ptr<GraphRebuildStateCtrl> &rebuild_ctrl);
  Status SetFrozenInputAttrs(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) const;
  void UpdateLocalOmgContext(GraphId graph_id);
  Status StartForRunGraph(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                          GeRootModelPtr &ge_root_model, uint64_t session_id = INVALID_SESSION_ID,
                          const rtStream_t stream = nullptr);
  Status TranFrameOp(const GraphNodePtr &graph_node);
  Status RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const;
  Status UnregisterExternalAllocator(const void * const stream) const;
  Status GetOmeContextByGraphId(const GraphId &graph_id, OmeContext &ome_context) const;
  bool GetLoadFlag(uint32_t graph_id) const;
  bool GetBuildFlag(uint32_t graph_id) const;
  Status GetCompiledFlag(uint32_t graph_id, bool &flag) const;
  Status SetCompiledFlag(uint32_t graph_id, bool flag);
 private:
  struct CompilerStages {
    GraphPrepare preparer;
    GraphOptimize optimizer;
    EnginePartitioner partitioner;
    GraphBuilder builder;
  };

  void UpdateDynamicParams(std::string &input_shape, std::string &dynamic_dims, int32_t &dynamic_node_type,
                           const std::map<std::string, std::string> &graph_options) const;
  void AddGraphNode(GraphId graph_id, const GraphNodePtr &graph_node);
  void RemoveGraphNode(GraphId graph_id);
  bool HasGraphNode(GraphId graph_id) const;
  void WarningForDeprecatedOptions(const std::map<std::string, std::string> &options) const;
  static Status ProcessSubGraphWithMultiThreads(GraphManager *graph_manager, GraphId root_graph_id,
                                                const SubGraphInfoPtr &sub_graph_info_ptr,
                                                const std::string &root_graph_name, uint64_t session_id,
                                                const struct error_message::ErrorManagerContext &error_context,
                                                const GEThreadLocalContext &ge_context, int32_t device_id);
  Status CheckGraphExisted(const GraphId &graph_id, bool &is_added);
  Status RunCustomPassAfterOriginGraphOptimize(ConstGraphPtr const_graph) const;
  Status RunCustomPass(ConstGraphPtr const_graph) const;
  Status PreRun(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs, GeRootModelPtr &ge_root_model,
                uint64_t session_id = INVALID_SESSION_ID);
  Status SaveOriginCommunicationNodes(const ComputeGraphPtr &compute_graph) const;
  Status VerifyCommNodesOrderAfterEngineAssigned(const ComputeGraphPtr &compute_graph) const;
  Status VerifyCommNodesOrderAfterBuild(const ComputeGraphPtr &compute_graph) const;
  Status BuildModel(const GraphNodePtr &graph_node, const std::vector<GeTensor> &input_tensors,
                    ComputeGraphPtr &root_graph, GeRootModelPtr &ge_root_model);
  Status DoBuildModel(ComputeGraphPtr &compute_graph, const std::vector<GeTensor> &input_tensors,
                      GeRootModelPtr &ge_root_model);
  static Status ResortAndUpdateMultiBatchContext(const GraphNodePtr &graph_node);
  static Status ResortDynamicBatchInput(const std::vector<std::vector<int64_t>> &batch_shapes,
                                        std::vector<NodePtr> &data_nodes);
  static Status UpdateMultiBatchContext(const std::vector<NodePtr> &data_nodes,
      const std::vector<std::vector<int64_t>> &batch_shapes,
      const std::map<std::string, std::vector<vector<int64_t>>> &data_to_dynamic_info);
  Status OptimizeSubgraph(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph, uint64_t session_id);

  Status DoDynamicShapePartition(const GraphNodePtr &graph_node, const ComputeGraphPtr &compute_graph);
  Status DoSubgraphPartitionWithMode(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                                     uint64_t session_id, EnginePartitioner::Mode mode, const char *mode_name);

  Status SubgraphPartitionAndOptimization(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
                                          uint64_t session_id, EnginePartitioner::Mode mode);

  Status Build(const GraphNodePtr &graph_node, ComputeGraphPtr &compute_graph,
               GeRootModelPtr &ge_root_model, uint64_t session_id);

  Status InnerRunGraphWithStream(const GraphNodePtr &graph_node, const GraphId &graph_id, rtStream_t stream,
                                 const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);
  
  Status ParseOptions(const std::map<std::string, std::string> &options);

  static void ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                          std::string &option);

  static Status ParseOption(const std::map<std::string, std::string> &options, const std::string &key, bool &option);

  static Status ParseOption(const std::map<std::string, std::string> &options, const std::string &key, int32_t &option);

  static Status ParseOption(const std::map<std::string, std::string> &options, const std::string &key,
                            std::map<std::string, int32_t> &option);

  static void Trim(std::string &str);

  static Status CheckEngineName(const std::string &engine_name, const std::string &key,
                                const std::map<std::string, int32_t> &option);

  static Status ParseParallelNum(const std::string &parallel_num, const std::string &key, int32_t &num);

  static Status ParseTrainGraphFlag(bool &train_flag);

  static bool IsPerfLevelInvalid(int32_t perf_level);

  Status SummaryHandle(const GraphId &graph_id, std::vector<gert::Tensor> &outputs);

  Status CheckpointHandle(const GraphId &graph_id, const ComputeGraphPtr &compute_graph,
                          const std::vector<gert::Tensor> &outputs);

  // call the callback function of ME to push summary result data to ME
  Status PushSummaryData2ME(const GraphId &graph_id, std::map<std::string, gert::Tensor> &summary_data);

  // call the callback function of ME to push save result data to ME
  Status PushSaveData2ME(const GraphId &graph_id, std::map<std::string, gert::Tensor> &save_data);

  bool IsCheckpointGraph(ComputeGraphPtr &compute_graph);

  bool CheckNetOutputForCheckpointGraph(const NodePtr &node) const;

  bool CheckVariableForCheckpointGraph(const NodePtr &node) const;

  bool CheckTransOpForCheckpointGraph(const NodePtr &node) const;

  Status MergeSubGraph(ComputeGraphPtr &compute_graph, const ComputeGraphPtr &original_compute_graph,
                       GraphId root_graph_id,
                       EnginePartitioner::Mode mode = EnginePartitioner::Mode::kAtomicEnginePartitioning);

  Status ConvertGraphToFile(ComputeGraphPtr &compute_graph, EnginePartitioner &partitioner, std::string path,
                            bool exe_flag = false) const;

  Status SetSubgraph(uint64_t session_id, ComputeGraphPtr compute_graph, EnginePartitioner &partitioner);

  void SetAttrForHcomBroadCastOp(ComputeGraphPtr &compute_graph);

  bool IsBroadCastOpData(const NodePtr &var_node) const;

  void AdjustBroadCastOpData(const NodePtr &var_node) const;

  bool IsAssignOpData(const NodePtr &var_node);

  void AdjustAssignOpData(const NodePtr &var_node) const;

  bool ConfirmUseOpAndIndexByAnchor(const InDataAnchorPtr &in_anchor,
                                    const std::map<std::string, std::set<int32_t>> &confirm_ops,
                                    NodePtr &use_node) const;

  bool ConfirmUseOpAndIndexByNode(const NodePtr &var_node, const std::map<std::string, std::set<int32_t>> &confirm_ops,
                                  NodePtr &use_node) const;

  // graph context
  std::shared_ptr<GraphContext> GetGraphContext() const { return graph_context_; }

  Status RemoveIsolatedConst(ComputeGraphPtr &compute_graph);
  Status RemoveIsolatedConstInThisGraph(const ComputeGraphPtr &compute_graph) const;

  Status AutofuseWithExtOptimize(ComputeGraphPtr &compute_graph, const std::vector<GeTensor> &inputs);
  Status OptimizeStage1(ComputeGraphPtr &compute_graph);
  Status OptimizeStage2(ComputeGraphPtr &compute_graph);
  Status MemConflictProc(ComputeGraphPtr &compute_graph);

  Status SubexpressionMigration(ComputeGraphPtr &compute_graph) const;

  bool CheckModelLoad(const GeRootModelPtr &ge_root_model, bool load_flag) const;

  bool IsGraphNeedBuild(const GraphNodePtr &graph_node) const;

  void PushRunArgs(const std::shared_ptr<RunArgs> &args) const;
  void PreRunThreadV2();
  void StopQueue();
  void ReturnError(RunAsyncCallbackV2 callback, Status ret,
      const std::string &log);
  void ChangeConstTypeWhenTraining(const ComputeGraphPtr &compute_graph) const;

  Status PreRunOptimizeOriginalGraph(const GraphNodePtr &graph_node, const std::vector<GeTensor> &inputs,
                                     ComputeGraphPtr &compute_graph, uint64_t session_id);
  Status PreRunOptimizeSubGraph(const GraphNodePtr &graph_node,
                                ComputeGraphPtr &compute_graph,
                                uint64_t session_id);
  Status PreRunAfterOptimizeSubGraph(const GraphNodePtr &graph_node,
                                     ComputeGraphPtr &compute_graph,
                                     GeRootModelPtr &ge_root_model,
                                     uint64_t session_id);

  Status UnfoldDynamicShapeGraph(ComputeGraphPtr &compute_graph) const;

  Status OptimizeSubGraphWithMultiThreads(ComputeGraphPtr compute_graph,
                                          Graph2SubGraphInfoList &sub_graph_map,
                                          uint64_t session_id);

  void AddLocalOmgContext(GraphId graph_id, const OmgContext &omg_context);

  CompilerStages &GetCompilerStages(GraphId graph_id);
  void RemoveCompilerStages(GraphId graph_id);

  Status CheckIncreBuildAndPreRun(const std::shared_ptr<RunArgs> &args, GraphNodePtr &graph_node);

  Status CheckRepeatAdd(uint32_t graph_id, bool &is_added);

  Status NotifyWaittingGraph(uint32_t graph_id);

  Status CreateGraphNode(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status SetStagesOptions(uint32_t graph_id);

  Status UnloadModel(GeRootModelPtr ge_root_model, uint32_t graph_id);

  void SetSessionGraphId(ComputeGraphPtr compute_graph, uint32_t graph_id) const;

  Status ModifyDataIndex(const Graph &graph, const std::map<std::string, std::string> &graph_option) const;

  static Status CheckGraphAdded(const GraphId &graph_id, const Graph &graph);
  std::string GetBuildMode(const GraphNodePtr &graph_node) const;
  std::string GetBuildStep(const GraphNodePtr &graph_node) const;
  std::string GetTuningPath(const GraphNodePtr &graph_node) const;
  void GetExcludeEngines(const GraphNodePtr &graph_node, GraphManagerOptions &refreshed_options) const;
  void RefreshOptionByGraph(uint32_t graph_id, GraphManagerOptions &refreshed_options);

  bool IsContainVariable(const ComputeGraphPtr &compute_graph) const;

  Status ConstructInputTensors(const ComputeGraphPtr &compute_graph, const std::vector<GeShape> &hint_shape,
                               std::vector<GeTensor> &inputs, bool support_unknown_shape = false) const;

  Status UpdateInputWithHintShape(const std::vector<GeShape> &hint_shape, std::vector<GeTensor> &inputs) const;

  Status NormalizeInputsOutputs(const ComputeGraphPtr &compute_graph,
                                const std::vector<GeTensor> &inputs,
                                const std::vector<GeTensor> &outputs,
                                std::vector<GeTensor> &normalized_inputs) const;

  Status CheckOptionsValid(const ComputeGraphPtr &compute_graph,
                           const std::map<std::string, std::string> &options) const;
  Status CheckFixedFeatureMemoryBase(const uint32_t graph_id, const MemoryType type, const void *const memory,
                                     const size_t size, bool &fixed_mem_not_exist);

  static Status ComputeHashForConstNodes(const ComputeGraphPtr &compute_graph);
  void SaveCompiledMemSize(const GraphNodePtr &graph_node, const CompiledGraphSummaryPtr &summary) const;
  Status TryUnloadModel(GraphId graph_id, const GraphNodePtr &graph_node);

  Status InnerRemoveGraph(const GraphId &graph_id);

  std::atomic_bool thread_run_flag_{false};
  BlockingQueue<std::shared_ptr<RunArgs>> prerun_args_v2_q_{};
  std::thread prerun_thread_v2_;
  std::map<GraphId, GraphNodePtr> graph_map_;
  std::vector<GraphId> graph_ids_;
  std::unordered_map<GraphId, std::unordered_set<GraphId>> graph_ids_to_forked_ids_;

  // summary and checkpoint callback function list for ME, key is summary or checkpoint
  std::map<std::string, std::function<Status(uint32_t, const std::map<std::string, gert::Tensor> &)>> me_callback_map2_;

  std::map<std::string, std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)>> callback_map2_;

  bool init_flag_{false};
  GraphManagerOptions options_;
  GraphContextPtr graph_context_ = nullptr;
  std::map<GraphId, OmgContext> omg_contexts_;

  std::shared_ptr<GraphRebuildStateCtrl> graph_rebuild_state_ctrl_;
  ResourceContextMgr resource_context_mgr_;
  std::map<GraphId, CompilerStages> compiler_stages_;
  Executor *executor_{nullptr};

  mutable std::shared_mutex callback_mutex_;
  mutable std::mutex member_mutex_;
  std::mutex unload_model_mutex_;
  // avoid repeatively add same graph (owns same graph id)
  std::mutex add_graph_mutex_;
  std::mutex add_graph_cond_mutex_;
  std::condition_variable add_graph_cv_;

  std::map<GraphId, uint32_t> graph_id_to_add_graph_cond_;
  // use for multi-thread online-infer scenario
  std::set<GraphId> to_be_deleted_graphs_;
  std::map<GraphId, uint32_t> graph_count_;
  std::mutex graph_count_mutex_;
  uint8_t logLevel_ = DLOG_DEBUG;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_MANAGER_H_
