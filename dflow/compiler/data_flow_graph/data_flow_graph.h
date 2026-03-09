/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_H

#include <map>
#include <string>
#include <vector>
#include <mutex>
#include "framework/common/debug/log.h"
#include "graph/compute_graph.h"
#include "dflow/inc/data_flow/model/flow_model.h"
#include "proto/dflow.pb.h"
#include "common/thread_pool/thread_pool.h"

namespace ge {
constexpr uint32_t kDataFlowGraphThreadPoolSize = 8U;
class DataFlowGraph {
 public:
  explicit DataFlowGraph(const ComputeGraphPtr &compute_graph,
                         const std::string &data_flow_scope = "",
                         const bool enable_cache = false,
                         const bool cache_manual_check = false,
                         const uint32_t data_flow_depth = 0U)
      : root_graph_(compute_graph),
        data_flow_scope_(data_flow_scope),
        data_flow_depth_(data_flow_depth),
        enable_cache_(enable_cache),
        cache_manual_check_(cache_manual_check),
        thread_pool_("df_pre_load_", kDataFlowGraphThreadPoolSize, true) {}
  ~DataFlowGraph() = default;
  Status Initialize();

  const ComputeGraphPtr &GetRootGraph() const {
    return root_graph_;
  }

  const std::string &GetName() const {
    return graph_name_;
  }

  const std::map<std::string, ComputeGraphPtr> &GetAllSubgraphs() const {
    return subgraphs_;
  }

  const std::map<std::string, std::vector<ComputeGraphPtr>> &GetNodeSubgraphs() const {
    return node_subgraphs_;
  }

  const std::map<std::string, std::vector<std::pair<ComputeGraphPtr, uint32_t>>> &GetNodesInputs() const {
    return nodes_inputs_;
  }

  const std::map<std::string, std::vector<std::pair<ComputeGraphPtr, uint32_t>>> &GetNodesOutputs() const {
    return nodes_outputs_;
  }

  const std::vector<std::string> &GetInvokeKeys(const std::string &graph_name) const;

  bool InvokedByBuiltIn(const std::string &invoke_key) const;

  const std::map<std::string, std::string> &GetGraphBuildOptions(const std::string &graph_name) const;

  const std::string &GetInvokedGraphKey(const std::string &graph_name) const;

  const std::string &GetInvokedKeyOriginName(const std::string &invoke_key) const;

  const std::map<std::string, FlowModelPtr> &GetAllLoadedModels() const {
    return loaded_models_;
  }

  const std::map<std::string, std::vector<FlowModelPtr>> &GetNodeLoadedModels() const {
    return node_loaded_models_;
  }

  Status AddLoadedModel(const std::string &node_name, const std::string &graph_name, const FlowModelPtr &model);

  bool IsInvokedGraph(const std::string &graph_name) const;

  bool EnableCache() const {
    return enable_cache_;
  }

  bool CacheManualCheck() const {
    return cache_manual_check_;
  }

  bool IsRootDataFlow() const {
    return data_flow_scope_.empty();
  }

  const std::string &GetDataFlowScope() const {
    return data_flow_scope_;
  }

  uint32_t GetDataFlowDepth() const {
    return data_flow_depth_;
  }

  Status CommitPreprocessTask(const std::string &name, std::function<Status()> &task);
  Status GetInvokedModelFusionAttrs(const std::vector<std::string> &invoke_keys,
                                    std::string &invoked_model_attrs) const;

 private:
  friend class ProcessPointLoader;
  Status CheckGraph() const;
  Status MapNodeInputsAndOutputs(const NodePtr &node, const dataflow::ProcessPoint &process_point);
  Status MapNodeInputs(const NodePtr &node, const dataflow::ProcessPoint &process_point);
  Status MapNodeOutputs(const NodePtr &node, const dataflow::ProcessPoint &process_point);
  void GetInOrOutIndex(const std::vector<std::pair<ComputeGraphPtr, uint32_t>> &vec, size_t &index) const;
  Status UpdateInputsFlowAttrs(const NodePtr &node);
  Status CheckFlowNode(const NodePtr &node) const;
  Status CheckAlignAttrs(bool &align_enable) const;
  Status CheckAndFixDataFlowAttrs() const;
  Status InitializeFlowNode(const NodePtr &node);

  Status WaitPreprocessTaskFinish();

  static bool NeedSkip(const std::string &op_type);
  ComputeGraphPtr root_graph_;
  std::string data_flow_scope_;
  uint32_t data_flow_depth_;
  std::string graph_name_;
  // key: subgraph name, value: subgraph
  std::map<std::string, ComputeGraphPtr> subgraphs_;
  // key: flow node name, value: subgraph list
  std::map<std::string, std::vector<ComputeGraphPtr>> node_subgraphs_;
  // key: node name, value: a vector which size is input num, value is {subgraph, subgraph_input_index}
  std::map<std::string, std::vector<std::pair<ComputeGraphPtr, uint32_t>>> nodes_inputs_;
  // key: node name, value: a vector which size is output num, value is {subgraph, subgraph_output_index}
  std::map<std::string, std::vector<std::pair<ComputeGraphPtr, uint32_t>>> nodes_outputs_;
  // key: udf graph name, value: invoked keys
  std::map<std::string, std::vector<std::string>> invokes_;
  // key: invoked graph name, value:invoked key
  std::map<std::string, std::string> invoked_keys_;
  // key: invoked key, value:invoked graph name
  std::map<std::string, std::string> invoked_graphs_;
  // key: invoked key, value:invoked by built in udf
  std::map<std::string, bool> invoked_by_built_in_;
  // key: invoked key with scope, value:usr set invoked key
  std::map<std::string, std::string> invoke_origins_;
  // key: subgraph name, value: build_options
  std::map<std::string, std::map<std::string, std::string>> graphs_build_options_;

  // guard for loaded_models_ and node_loaded_models_ when loading.
  std::mutex loaded_models_mt_;
  // key: graph name, value: model
  std::map<std::string, FlowModelPtr> loaded_models_;
  // key: flow node name, value: loaded model
  std::map<std::string, std::vector<FlowModelPtr>> node_loaded_models_;
  bool enable_cache_;
  bool cache_manual_check_;

  ThreadPool thread_pool_;
  std::map<std::string, std::future<Status>> preprocess_tasks_;
};
}  // namespace ge
#endif
