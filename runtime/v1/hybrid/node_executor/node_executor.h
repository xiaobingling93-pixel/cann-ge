/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_NODE_EXECUTOR_NODE_EXECUTOR_H_
#define GE_HYBRID_NODE_EXECUTOR_NODE_EXECUTOR_H_

#include "ge/ge_api_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "graph/node.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
namespace hybrid {
class HybridModel;
using NodeTaskPtr = std::shared_ptr<NodeTask>;

// Base class of Node Task
class NodeTask {
 public:
  NodeTask() = default;
  virtual ~NodeTask() = default;

  virtual Status SelectBin(TaskContext &task_context, const GraphExecutionContext *const ctx) {
    (void)task_context;
    (void)ctx;
    return SUCCESS;
  }

  /**
   * Is need update tiling data
   * @return default is false
   */
  virtual bool IsNeedTilling() {
    return false;
  }

  /**
   * Update tiling data
   * @param context             instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status UpdateTilingData(TaskContext &context) {
    (void)context;
    return SUCCESS;
  }

  /**
   * Init
   * @param context             instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status Init(TaskContext &context) {
    (void)context;
    return SUCCESS;
  }

  /**
   * Whether this task supports dynamic shape
   * @return true if this task supports dynamic shape, false otherwise
   */
  virtual bool IsSupportDynamicShape() {
    return true;
  }

  /**
   * Whether this task supports host mem input optimise
   * @return true if this task supports host mem input optimise, false otherwise
   */
  virtual bool IsSupportHostMemInputOpt() const {
    return false;
  }

  /**
   * Whether this task's args extended for host mem input optimization
   * @return true if this task's args extended for host mem input optimization, false otherwise
   */
  virtual bool IsArgsExtendedForHostMemInput() const {
    return false;
  }

  /**
   * Set need host memory optimization
   */
  virtual void SetNeedHostMemOpt(const bool need_host_mem_opt) {
    (void)need_host_mem_opt;
  }

  /**
   * Update args for execution
   * @param context             instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status UpdateArgs(TaskContext &context) = 0;

  /**
   * Execute task async
   * @param context             instance of TaskContext
   * @param done_callback       callback function, will be invoked after task is done
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) = 0;

  /**
   * init task info during load phase
   * @param node             node of the task
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status InitTaskBasicInfo(const NodePtr &node) {
    (void)node;
    return SUCCESS;
  }

  virtual Status ReportProfilingData() {
    return SUCCESS;
  }
 private:
  NodeTask &operator=(const NodeTask&) = default;
  NodeTask(const NodeTask&) = default;
};

class NoOpTask : public NodeTask {
 public:
  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
};

// Node executor
class NodeExecutor {
 public:
  NodeExecutor() noexcept = default;
  virtual ~NodeExecutor() = default;

  /**
   * Initialize node executor
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status Initialize() {
    return SUCCESS;
  }

  /**
   * Finalize node executor
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status Finalize() {
    return SUCCESS;
  }

  /**
   * Load task in load stage
   * @param model       instance of HybridModel
   * @param node        node
   * @param task        generated node task
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status LoadTask(const HybridModel &model,
                          const NodePtr &node,
                          std::shared_ptr<NodeTask> &task) const;

  /**
   * Preparation actions before execution
   * @param task        instance of NodeTask
   * @param context     instance of TaskContext
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status PrepareTask(NodeTask &task, TaskContext &context) const;

  /**
   * Execute task
   * @param task        instance of NodeTask
   * @param context     instance of TaskContext
   * @param callback    callback function which will be invoked after computation is done
   * @return SUCCESS on success, error code otherwise
   */
  virtual Status ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const;

  virtual Status ReportProfilingData(const NodeItem &node_item) const {
    (void)node_item;
    return SUCCESS;
  }
};

class NodeExecutorManager {
 public:
  enum class ExecutorType {
    AICORE,
    AICPU_TF,
    AICPU_CUSTOM,
    COMPILED_SUBGRAPH,
    DYNAMIC_SUBGRAPH,
    GE_LOCAL,
    CONTROL_OP,
    HCCL,
    RTS,
    HOST_CPU,
    FFTS,
    RESERVED
  };

  static NodeExecutorManager &GetInstance() {
    static NodeExecutorManager instance;
    return instance;
  }

  /**
   * Register build of executor
   * @param executor_type   type of executor
   * @param builder         build function
   */
  void RegisterExecutorBuilder(const ExecutorType executor_type,
                               const std::function<std::unique_ptr<NodeExecutor>()> &builder);

  /**
   * Initialize executor if needed
   * @return SUCCESS on success, error code otherwise
   */
  Status EnsureInitialized();

  void FinalizeExecutors();

  /**
   * Get executor by node
   * @param node            node
   * @param executor        executor
   * @return SUCCESS on success, error code otherwise
   */
  Status GetExecutor(const NodeItem &node_item, const NodeExecutor *&executor);

  /**
   * Resolve executor type by node
   * @param node            node
   * @return executor type
   */
  ExecutorType ResolveExecutorType(const NodeItem &node_item) const;

  Status GetOrCreateExecutor(const ExecutorType executor_type, const NodeExecutor *&out_executor);

  bool IsExecutorInitialized(const ExecutorType executor_type) const;

 private:
  std::map<ExecutorType, std::unique_ptr<NodeExecutor>> executors_;
  std::map<ExecutorType, std::function<std::unique_ptr<NodeExecutor>()>> builders_;
  std::map<std::string, NodeExecutorManager::ExecutorType> engine_mapping_;
  mutable std::mutex mu_;
  bool initialized_ = false;
  int32_t ref_count_ = 0;
};

class NodeExecutorRegistrar {
 public:
  NodeExecutorRegistrar(const NodeExecutorManager::ExecutorType executor_type,
                        std::unique_ptr<NodeExecutor> (*builder)());
  ~NodeExecutorRegistrar() = default;
};
}  // namespace hybrid
}  // namespace ge

#define REGISTER_NODE_EXECUTOR_BUILDER(engine_type, executor) \
    REGISTER_NODE_EXECUTOR_BUILDER_UNIQ_HELPER(__COUNTER__, engine_type, executor)

#define REGISTER_NODE_EXECUTOR_BUILDER_UNIQ_HELPER(ctr, engine_type, executor) \
    REGISTER_NODE_EXECUTOR_BUILDER_UNIQ(ctr, engine_type, executor)

#define REGISTER_NODE_EXECUTOR_BUILDER_UNIQ(ctr, engine_type, executor)                         \
  static ::ge::hybrid::NodeExecutorRegistrar register_##executor##ctr                           \
      __attribute__((unused)) =                                                                 \
          ::ge::hybrid::NodeExecutorRegistrar((engine_type), []()->std::unique_ptr<::ge::hybrid::NodeExecutor> {  \
            return MakeUnique<executor>();                                               \
          })

#endif // GE_HYBRID_NODE_EXECUTOR_NODE_EXECUTOR_H_
