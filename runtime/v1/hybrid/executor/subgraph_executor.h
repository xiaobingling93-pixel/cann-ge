/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_EXECUTOR_SUBGRAPH_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_EXECUTOR_SUBGRAPH_EXECUTOR_H_

#include <vector>

#include "common/blocking_queue.h"
#include "common/thread_pool/thread_pool.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/worker/shape_inference_engine.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
namespace hybrid {
// Executor for executing a subgraph
class SubgraphExecutor : SubgraphContext, ShapeInferenceEngine {
 public:
  SubgraphExecutor(const GraphItem *const graph_item, GraphExecutionContext *const context,
                   const bool force_infer_shape = false, ThreadPool *pre_run_pool = nullptr);
  virtual ~SubgraphExecutor();

  Status Init();

  Status InitInputs(const std::vector<TensorValue> &inputs, const std::vector<ConstGeTensorDescPtr> &input_desc);

  Status PartialExecuteAsync(const int32_t task_group);

  /**
   * Execute subgraph async, output tensor address(not data) and output tensor descriptions are
   * valid after this method returned
   * @param inputs          input tensors
   * @param input_desc      input tensor descriptions
   * @return SUCCESS on success, error code otherwise
   */
  Status ExecuteAsync(const std::vector<TensorValue> &inputs,
                      const std::vector<ConstGeTensorDescPtr> &input_desc);

  /**
   * Execute subgraph async, output tensor address(not data) and output tensor descriptions are
   * valid after this method returned
   * @param inputs          input tensors
   * @param input_desc      input tensor descriptions
   * @return SUCCESS on success, error code otherwise
   */
  Status ExecuteAsync(const std::vector<TensorValue> &inputs,
                      const std::vector<ConstGeTensorDescPtr> &input_desc,
                      const std::vector<TensorValue> &outputs);

  /**
   * Execute subgraph async, output tensor address(not data) and output tensor descriptions are
   * valid after this method returned
   * @param task_context    instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  Status ExecuteAsync(const TaskContext &task_context);

  /**
   * Synchronize all tasks in the subgraph. output tensor data are valid after this method returned
   * @return SUCCESS on success, error code otherwise
   */
  Status Synchronize() const;

  /**
   * Get output tensors
   * @param outputs         output tensors
   * @return SUCCESS on success, error code otherwise
   */
  Status GetContextOutputs(std::vector<TensorValue> &outputs);

  /**
   * Get output tensors and output tensor descriptions
   * @param outputs         output tensors
   * @param output_desc     output tensor descriptions
   * @return SUCCESS on success, error code otherwise
   */
  Status GetOutputs(std::vector<TensorValue> &outputs, std::vector<ConstGeTensorDescPtr> &output_desc);

  void Reset();

 protected:
  Status NodeEnqueue(NodeState *const node_state);
  Status PrepareForExecution(const GraphExecutionContext *const ctx, NodeState &node_state) const;
  Status LaunchTasks();
  Status ScheduleTasks(const int32_t group = -1);
  Status PrepareNodes(const int32_t group = -1);

  virtual Status PreRunSubgraph() { return SUCCESS; }
  virtual Status InitCallback(NodeState *const node_state, std::function<void()> &callback,
                              const std::shared_ptr<ScopeGuard> tensor_guard = nullptr);
  virtual Status PrepareNode(const NodeItem &node_item);

 private:
  Status EnableOutputZeroCopy(const std::vector<TensorValue> &outputs);
  Status InitInputsForKnownShape(const std::vector<TensorValue> &inputs);
  Status ExecuteAsyncForKnownShape(const std::vector<TensorValue> &inputs);
  Status SetOutputsToParentNode(const TaskContext &task_context) const;
  Status ReportTimeoutInfo(const NodeState *const node_state) const;

  BlockingQueue<const NodeItem *> &GetPrepareQueue(const int32_t group);

  Status NodeScheduled(NodeState *const node_state);
  Status AfterPrepared(NodeState *const node_state);
  void AfterExecuted(NodeState *const node_state);
  void OnNodeDone(NodeState *const node_state);
  bool HasOnNodeDoneCallback(const NodeState &node_state) const;

 private:
  const GraphItem *graph_item_;
  GraphExecutionContext *context_ = nullptr;
  SubgraphContext *subgraph_context_ = nullptr;
  ThreadPool *pre_run_pool_;
  bool own_thread_pool_;
  BlockingQueue<NodeState *> ready_queue_;
  ShapeInferenceEngine *shape_inference_engine_ = nullptr;

  std::mutex mu_; // Guard for prepare_queues_.
  std::map<int32_t, BlockingQueue<const NodeItem *>> prepare_queues_;
  friend class FftsPlusSubgraphExecutor;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_EXECUTOR_SUBGRAPH_EXECUTOR_H_
