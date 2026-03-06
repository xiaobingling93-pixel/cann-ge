/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef JIT_EXECUTOR_H
#define JIT_EXECUTOR_H
#include <vector>
#include <queue>
#include <mutex>

#include "runtime/base.h"
#include "exe_graph/runtime/tensor.h"
#include "ge/ge_api_types.h"

#include "exe_points/execution_order.h"
#include "cache/compiled_model_cache.h"
#include "compile_context.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/type_utils.h"

namespace ge {
struct UserGraphExecution {
  UserGraphExecution(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
    const RunAsyncCallbackV2 &callback_func, uint64_t session_id_param)
      : user_graph_id(graph_id), callback(callback_func), session_id(session_id_param), external_rt_inputs(&inputs) {
    inputs_memblocks.resize(inputs.size());
  }
  // for RunGraphAsync
  UserGraphExecution(uint32_t graph_id, std::vector<gert::Tensor> &&inputs,
    const RunAsyncCallbackV2 &callback_func, uint64_t session_id_param)
      : user_graph_id(graph_id), callback(callback_func), session_id(session_id_param),
    input_tensors_holder(std::move(inputs)), external_rt_inputs(&input_tensors_holder) {
    inputs_memblocks.resize(input_tensors_holder.size());
  }
  ~UserGraphExecution() = default;
  uint32_t user_graph_id;
  std::vector<MemBlock *> inputs_memblocks;
  RunAsyncCallbackV2 callback{nullptr};
  void *stream{nullptr};
  uint64_t session_id;
  // 仅RunGraphAsync，inputs需要保存在input_tensors_holder中
  std::vector<gert::Tensor> input_tensors_holder;
  const std::vector<gert::Tensor> *external_rt_inputs{nullptr};
  std::vector<gert::Tensor> *rt_outputs{nullptr};
  std::map<AscendString, AscendString> load_options;
};
using UserGraphExecutionQueue = std::queue<std::unique_ptr<UserGraphExecution>>;

template <typename K, typename V>
std::vector<std::pair<K, V>> SortMapByValue(const std::map<K, V> &input_map, bool is_ascend = true) {
  std::vector<std::pair<K, V>> target_vec(input_map.begin(), input_map.end());
  std::sort(target_vec.begin(), target_vec.end(),
    [is_ascend](const std::pair<K, V> &a, const std::pair<K, V> &b) {
      return is_ascend ? (a.second < b.second) : (a.second > b.second);
    });
  return target_vec;
}

class JitExecutor {
 public:
  static std::unique_ptr<JitExecutor> Create(GraphManager &graph_manager, UserGraphExecutionQueue &task_queue,
                                             ExecutionOrder &order, CompileContext &compile_context,
                                             CompiledModelCache &cmc, std::mutex &mutex);

  Status RunWithCallback(UserGraphExecution &&task);

  Status Finalize();

  bool IsUserGraphNeedRebuild();

  Status CompileGraph(UserGraphExecution &task, uint64_t session_id);

  Status LoadGraph(UserGraphExecution &task);

  Status Execute(UserGraphExecution &&task);
 private:
  JitExecutor(GraphManager &graph_manager, UserGraphExecutionQueue &task_queue, ExecutionOrder &order,
              CompileContext &compile_context, CompiledModelCache &cmc, std::mutex &mutex);
  Status CompileAndLoad(const std::vector<gert::Tensor> &inputs, GuardedExecutionPoint *gep, uint32_t &instance_id,
      const rtStream_t stream, const std::map<AscendString, AscendString> &load_options, uint64_t session_id);
  Status Compile(const std::vector<ge::Tensor> &inputs, GuardedExecutionPoint *gep, uint64_t session_id);
  Status ProcessAndExecuteGraphAsync(UserGraphExecution &task, rtStream_t const stream,
                                     const std::vector<gert::Tensor> &inputs,
                                     std::vector<gert::Tensor> &outputs, ExecutionPoint *ep,
                                     bool need_malloc_output = false);
  Status ExecuteFirstPoint(UserGraphExecution &task, rtStream_t const stream,
                           std::vector<gert::Tensor> &outputs, std::vector<GeTensor> &ge_tensors,
                           ExecutionPoint *&ep, bool need_malloc_output);
  Status TryExecuteWithoutProcess(UserGraphExecution &task);
  Status MallocOutputsForStatic(uint32_t guarded_ep_instance_id, const GuardedExecutionPoint *gep,
                                std::vector<gert::Tensor> &outputs);
 private:
  GraphManager &graph_manager_;
  UserGraphExecutionQueue &task_queue_;
  ExecutionOrder &order_;
  CompileContext &compile_context_;
  CompiledModelCache &cmc_;
  std::mutex &mutex_;
  std::map<const GuardedExecutionPoint *, uint32_t> geps_to_inner_ge_graph_id_;
  rtStream_t stream_{nullptr};
  std::shared_ptr<ge::Allocator> device_allocator_;
  int32_t device_id_{-1};
  std::vector<uint32_t> compiled_ge_graph_id_;
  std::shared_ptr<ge::Allocator> external_allocator_{nullptr};
};
}  // namespace ge

#endif  // JIT_EXECUTOR_H
