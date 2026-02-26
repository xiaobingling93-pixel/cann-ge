/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef USER_GRAPH_CTRL_H
#define USER_GRAPH_CTRL_H
#include <vector>
#include <queue>
#include <stack>
#include <memory>

#include "exe_graph/runtime/tensor.h"
#include "ge/ge_api_types.h"

#include "graph/compute_graph.h"

#include "compile_context.h"
#include "exe_points/execution_order.h"
#include "cache/compiled_model_cache.h"
#include "jit_executor.h"
#include "common/thread_pool.h"

namespace ge {
namespace {
constexpr uint32_t kDefaultJitExeThreadPoolSize = 8U;
}
class JitExecutorPool {
 public:
  Status AddJitExecutor(std::unique_ptr<JitExecutor> &jit_executor);
  JitExecutor *GetIdleExecutor();
  void BackToIdle(JitExecutor *jit_executor);
  Status Finalize();
  size_t Size();
  bool IsGraphNeedRebuild();

 private:
  std::mutex executors_mutex_;
  std::vector<std::unique_ptr<JitExecutor>> executors;
  std::mutex idle_mutex_;
  std::stack<JitExecutor *> idle_executors;
};
class UserGraphControl {
 public:
  UserGraphControl(uint32_t user_graph_id, const ComputeGraphPtr &graph, CompileContext &compile_context,
                   GraphManager &graph_manager)
      : user_graph_id_(user_graph_id),
        order_(UserGraph({user_graph_id, graph})),
        compile_context_(compile_context),
        graph_manager_(graph_manager),
        jit_exe_thread_pool_("jit_exe", kDefaultJitExeThreadPoolSize, true),
        cmc_(user_graph_id_, compile_context_, graph_manager_) {
    auto ge_context = ge::GetThreadLocalContext();
    user_graph_exe_thread_ = std::thread(
      [this, ge_context]() mutable {
        ge::GetThreadLocalContext() = ge_context;
        this->ExecuteUserGraph();
      });
    const auto ret = cmc_.RestoreCache(order_);
    if (ret != SUCCESS) {
      GELOGW("CompiledModelCache RestoreCache failed. The cache files are not valid. user_graph[%u].", user_graph_id);
    }
  }
  ~UserGraphControl() {
    (void)Finalize();
  }

  Status AddGraphInstance();
  void RunGraphAsync(std::unique_ptr<UserGraphExecution> &task);
  Status ExecuteGraphWithStreamAsync(std::unique_ptr<UserGraphExecution> task);
  Status CompileGraph(uint64_t session_id);
  CompiledGraphSummaryPtr GetCompiledGraphSummary();
  Status LoadGraph(const std::map<AscendString, AscendString> &options, void *stream);
  Status Finalize();
  bool IsUserGraphNeedRebuild();
  bool GetCompiledFlag() const;
  void SetCompiledFlag(bool flag);
  std::map<AscendString, AscendString> GetLoadOptions() const;

 private:
  void StopQueue();
  void ExecuteUserGraph();
  void SetLoadOptions(const std::map<AscendString, AscendString> &load_options);
  Status CompileCompleteGraph(uint64_t session_id);

  uint32_t user_graph_id_;
  ExecutionOrder order_;
  CompileContext &compile_context_;
  GraphManager &graph_manager_;
  std::mutex compile_mutex_;

  std::mutex add_execution_mutex_;
  std::atomic_bool thread_is_stopped_{false};
  UserGraphExecutionQueue executions_;
  std::thread user_graph_exe_thread_;

  // mutext is inside
  JitExecutorPool jit_executor_pool_;
  ThreadPool jit_exe_thread_pool_;
  std::mutex jit_futures_mutex_;
  std::queue<std::future<Status>> jit_futures_;
  mutable std::mutex options_mutex_;
  std::map<AscendString, AscendString> load_options_;
  std::map<uint32_t, uint32_t> user_graph_id_to_ins_id;
  // std::vector<std::unique_ptr<JitExecutor>> executors_; // 实例
  CompiledModelCache cmc_;

  // set true only when Session::CompileGraph is called
  // only set or check in ge_api.cc
  bool compiled_flag_{false};
};
}  // namespace ge

#endif  // USER_GRAPH_CTRL_H
