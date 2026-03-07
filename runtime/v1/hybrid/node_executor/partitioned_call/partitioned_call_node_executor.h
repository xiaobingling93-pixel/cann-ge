/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_NODE_EXECUTOR_PARTITIONED_CALL_NODE_EXECUTOR_H_
#define GE_HYBRID_NODE_EXECUTOR_PARTITIONED_CALL_NODE_EXECUTOR_H_

#include "hybrid/node_executor/node_executor.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/subgraph_executor.h"
#include "common/thread_pool/thread_pool.h"

namespace ge {
namespace hybrid {
class PartitionedCallNodeTask : public NodeTask {
 public:
  explicit PartitionedCallNodeTask(const GraphItem * const graph_item);
  ~PartitionedCallNodeTask() override;

  Status Init(TaskContext &context) override;

  Status UpdateArgs(TaskContext &context) override;

  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

 private:
  Status Callback(const std::function<void()> &done_callback);

  const GraphItem *graph_item_;
  std::unique_ptr<SubgraphExecutor> subgraph_executor_;
};

class PartitionedCallNodeExecutor : public NodeExecutor {
 public:
  PartitionedCallNodeExecutor() noexcept : NodeExecutor() {}
  ~PartitionedCallNodeExecutor() override = default;
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_NODE_EXECUTOR_PARTITIONED_CALL_NODE_EXECUTOR_H_
