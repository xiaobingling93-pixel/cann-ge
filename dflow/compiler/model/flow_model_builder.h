/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_COMPILER_MODEL_FLOW_MODEL_BUILDER_H_
#define DFLOW_COMPILER_MODEL_FLOW_MODEL_BUILDER_H_

#include <future>
#include "graph/compute_graph.h"
#include "dflow/compiler/pne/process_node_engine.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"

namespace ge {
class FlowModelBuilder {
 public:
  Status InitProcessNodeEngines(const std::map<std::string, std::string> &options,
                                const ProcessNodeEngineImplPtr &pneImpl);
  void Finalize();
  Status BuildModel(Graph &graph, const std::vector<GeTensor> &input_tensors,
                    const std::map<std::string, std::string> &options, FlowModelPtr &flow_model) const;

 private:
  struct CacheParam {
    bool enable_cache;
    bool manual_check;
    bool debug_mode;
  };
  struct DataFlowGraphParam {
    std::string df_scope;
    std::string deploy_info;
    uint32_t df_depth;
  };

  static Status GetOrAssignDefaultEngine(const ComputeGraphPtr &compute_graph, std::string &process_node_engine_id);
  static void UpdateThreadLocalOptions(const std::string &pne_id);
  static void ClearThreadLocalOptions(const std::string &pne_id);
  static Status MergeDataFlowLoadedModel(const DataFlowGraph &data_flow_graph, const FlowModelPtr &flow_model);
  static Status MergeInvokedModel(const FlowModelPtr &flow_model, const std::string &invoke_key,
                                  const FlowModelPtr &invoked_flow_model, bool invoked_by_built_in);
  static Status CheckAndSetUdfInvokeKeys(const std::shared_ptr<PneModel> &pne_model,
                                         const std::shared_ptr<ModelRelation> &model_relation);
  static Status SetUdfInvokeKeysRecursively(const std::shared_ptr<PneModel> &pne_model,
                                            const std::shared_ptr<ModelRelation> &model_relation, int32_t depth);
  /**
   * @brief Get the Input Data Tensor Descs.
   * @param graph graph
   * @param input_tensor_descs input tensor desc in order
   * @return Status get result.
   */
  static Status GetInputDataTensorDescs(const ComputeGraph &graph, std::vector<GeTensorDesc> &input_tensor_descs);

  static Status MakeInputTensors(const ComputeGraphPtr &graph, const std::map<std::string, std::string> &options,
                                 std::vector<GeTensor> &input_tensors);
  static Status UpdateTensorDescByOption(std::vector<GeTensorDesc> &input_tensor_descs,
                                         const std::map<std::string, std::string> &options);
  Status BuildModel(ComputeGraphPtr &root_graph, const std::vector<GeTensor> &input_tensors,
                    const std::map<std::string, std::string> &options, const FlowModelPtr &flow_model,
                    const CacheParam &cache_param) const;
  Status BuildHeterogeneousModel(ComputeGraphPtr &root_graph,
                                 const std::vector<GeTensor> &input_tensors,
                                 const std::map<std::string, std::string> &options,
                                 const FlowModelPtr &flow_model) const;

  Status DoBuildGraph(ComputeGraphPtr &compute_graph,
                      const std::map<std::string, std::string> &options,
                      const std::vector<GeTensor> &input_tensors,
                      bool is_sub_graph,
                      const FlowModelPtr &flow_model) const;
  Status GetEngine(const std::string &pne_id, ProcessNodeEnginePtr &engine) const;

  static Status GetEschedPriority(const ComputeGraphPtr &graph, const std::string &attr_name,
                                  std::map<std::string, int32_t> &esched_priority);
  static Status GetModelEschedPriority(const PneModelPtr &pne_model, std::map<std::string, int32_t> &esched_priority);
  static Status BuildModelEschedPriority(const FlowModelPtr &flow_model);

  Status BuildDataFlowGraph(const ComputeGraphPtr &root_graph, const std::map<std::string, std::string> &options,
                            const FlowModelPtr &flow_model, const CacheParam &cache_param,
                            const DataFlowGraphParam &df_param) const;
  static Status RemoveDataFlowSubgraphs(const FlowModelPtr &flow_model, const CacheParam &cache_param);
  Status BuildFlowSubgraph(ComputeGraphPtr &graph, const std::vector<GeTensor> &input_tensors,
                           const std::map<std::string, std::string> &options, const FlowModelPtr &flow_model) const;
  Status BuildFlowSubgraph(ComputeGraphPtr graph, const std::map<std::string, std::string> &options,
                           const FlowModelPtr &flow_model) const;
  Status BuildDataFlowSubGraphs(const DataFlowGraph &data_flow_graph, const std::map<std::string, std::string> &options,
                                const FlowModelPtr &flow_model, const CacheParam &cache_param) const;
  static Status PostOfDataFlowSubGraphsBuild(const DataFlowGraph &data_flow_graph,
                                             std::vector<std::future<Status>> &vector_future,
                                             const std::vector<FlowModelPtr> &sub_flow_models,
                                             const FlowModelPtr &flow_model);
  static Status PostProcessSubFlowModel(const DataFlowGraph &data_flow_graph, const FlowModelPtr &flow_model,
                                        const ComputeGraphPtr &subgraph, const FlowModelPtr &sub_flow_model);
  static Status UpdateDeployInfo(const ComputeGraphPtr &graph, const FlowModelPtr &flow_model);

  Status BuildGraph(ComputeGraphPtr &graph, const vector<GeTensor> &input_tensors,
                    const map<std::string, std::string> &options, bool is_sub_graph,
                    const FlowModelPtr &flow_model) const;

  static Status CheckCacheGraphIoNodesWithGraphAdded(const ComputeGraphPtr &cached_graph,
                                                     const ComputeGraphPtr &added_graph);
  static Status FindInvokesAndGetSubDataFlowDeployInfos(const DataFlowGraph &data_flow_graph,
                                                        std::map<std::string, DataFlowGraphParam> &deploy_infos);
  static Status CheckInvokedDataFlowDepth(uint32_t depth);
  static Status ProcessNetOutput(const ComputeGraphPtr &compute_graph);
  static Status ModifyDataIndex(const ComputeGraphPtr &compute_graph);

  std::map<std::string, ProcessNodeEnginePtr> process_node_engines_;
};
}  // namespace ge

#endif  // DFLOW_COMPILER_MODEL_FLOW_MODEL_BUILDER_H_
