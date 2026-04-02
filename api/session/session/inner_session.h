/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SESSION_INNER_SESSION_H_
#define GE_SESSION_INNER_SESSION_H_

#include <map>
#include <string>
#include <vector>
#include <set>
#include "common/dump/dump_properties.h"
#include "framework/common/ge_types.h"
#include "ge/ge_api_types.h"
#include "ge/ge_data_flow_api.h"
#include "graph/manager/graph_manager.h"
#include "graph/execute/model_executor.h"
#include "ge/ge_allocator.h"

#include "jit_execution/user_graphs_manager.h"
#include "user_hybrid_graph_manager.h"
#include "acl/acl_rt.h"

namespace ge {

class DFlowSessionImpl;

class InnerSession {
 public:
  InnerSession(uint64_t session_id, const std::map<std::string, std::string> &options);

  ~InnerSession() = default;

  Status Initialize();
  // only ge_api.cc call
  Status AddGraph(uint32_t graph_id, const Graph &graph);

  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status LoadGraph(const uint32_t graph_id,
                   const std::map<AscendString, AscendString> &options, void *stream);

  Status RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

  Status RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  Status RunGraphWithStreamAsync(uint32_t graph_id, aclrtStream stream, const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &outputs);

  Status ExecuteGraphWithStreamAsync(uint32_t graph_id, const aclrtStream stream,
                                     const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  Status RemoveGraph(uint32_t graph_id);

  Status BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs);

  Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);

  Status RunGraphAsync(uint32_t graph_id, std::vector<gert::Tensor> &&inputs, const RunAsyncCallbackV2 &callback);

  Status Finalize();

  Status GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables);

  Status GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables, Graph &graph);

  Status SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                       const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values);

  Status RegisterCallBackFunc(
      const std::string &key,
      const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback);

  Status RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, ge::Tensor> &)> &callback);

  Status RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback);

  const GraphManager &getGraphManagerObj() const;

  bool IsGraphNeedRebuild(uint32_t graph_id);

  Status AddDumpProperties(const DumpProperties &dump_properties);

  Status RemoveDumpProperties();

  static void SetRtSocVersion();

  static Status SetSessionGraphId(const Graph &graph, uint64_t session_id, uint32_t graph_id);

  Status CompileGraph(uint32_t graph_id, const vector<ge::Tensor> &inputs);

  Status GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary);

  Status SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status SetGraphFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory, size_t size);

  Status UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const;

  Status UnregisterExternalAllocator(const void * const stream) const;

  Status PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len) const;

  /*
   * @brief 将origin_graph_id图的fork一份，fork出的图与原始图共享编译model，fork出的图可以独立加载出新实例并执行
   * 原始图应该是已编译的状态
   * 当原始图被卸载的时候，fork图也会被卸载
   */
  Status ForkGraph(uint32_t origin_graph_id, uint32_t forked_graph_id);

  uint64_t GetSessionId() const {
    return session_id_;
  }

  void UpdateGlobalSessionContext() const;

  Status GetCompiledFlag(uint32_t graph_id, bool &flag) const;

  Status SetCompiledFlag(uint32_t graph_id, bool flag);

  // Get and Set dflow_session_impl_
  std::shared_ptr<DFlowSessionImpl> GetDFlowSession() const;

  void SetDFlowSession(const std::shared_ptr<DFlowSessionImpl> &dflow_session_impl);

  Status GetRunGraphMode(uint32_t graph_id, RunGraphMode &mode) const;

  Status SetRunGraphMode(uint32_t graph_id, const RunGraphMode &mode);

  Status GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer);

  bool GetBuildFlag(uint32_t graph_id) const;

  bool GetLoadFlag(uint32_t graph_id) const;

 private:
  Status InnerInitialize();
  Status InnerFinalize();
  static void SetTrainFlagOption();
  static Status InitializeExecutionRuntime(const std::map<std::string, std::string> &options);

  // 仅用于防重复初始化，若初始化失败，inner session对象不应被获取到，其成员方法也不会被调用, 因此初始化成功后成员方法内不必再判断
  bool is_initialized_{false};
  uint64_t session_id_;
  uint8_t logLevel_ = DLOG_DEBUG;
  std::map<std::string, std::string> options_;
  // 在UserGraphsManager/UserHybridGraphManager场景中，用户持有的graph_id不能直接传递给graph_manager_,容易犯错
  GraphManager graph_manager_;
  ModelExecutor model_executor_;
  std::mutex resource_mutex_;  // AddGraph, RemoveGraph and Finalize use
  Status CheckPaRemappedResult(const uint64_t va, const uint64_t len,
                               std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) const;
  Status InitializeVarManager();
  static bool is_dump_server_inited_;
  std::shared_ptr<DFlowSessionImpl> dflow_session_impl_;
  UserGraphsManagerPtr user_graphs_manager_{nullptr};
  UserHybridGraphManagerPtr user_hybrid_graph_manager_{nullptr};
};
using SessionPtr = std::shared_ptr<InnerSession>;
void CopyGeOutputsMemToUserOutputs(const aclrtStream stream, const std::vector<GeTensor> &ge_outputs,
                                   std::vector<Tensor> &outputs);
}  // namespace ge

#endif  // GE_SESSION_INNER_SESSION_H_
