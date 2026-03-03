/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/node_executor.h"
#include "framework/common/ge_types.h"
#include "common/math/math_util.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"

namespace ge {
namespace hybrid {
const std::string kRtsFftsPlusOpKernelName = "DNN_VM_RTS_FFTS_PLUS_OP_STORE";

Status NodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GE_CHK_STATUS_RET_NOLOG(context.AllocateOutputs());
  GE_CHK_STATUS_RET_NOLOG(context.AllocateWorkspaces());
  GE_CHK_STATUS_RET_NOLOG(task.UpdateArgs(context));
  return SUCCESS;
}

Status NodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const {
  HYBRID_CHK_STATUS_RET(task.ExecuteAsync(context, callback),
                        "[Execute][Task] failed. node = %s", context.GetNodeItem().NodeName().c_str());
  return SUCCESS;
}

Status NodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, std::shared_ptr<NodeTask> &task) const {
  (void)model;
  (void)node;
  (void)task;
  return UNSUPPORTED;
}

Status NodeExecutorManager::EnsureInitialized() {
  const std::lock_guard<std::mutex> lk(mu_);
  ++ref_count_;
  if (initialized_) {
    return SUCCESS;
  }

  (void)engine_mapping_.emplace(kEngineNameAiCore, NodeExecutorManager::ExecutorType::AICORE);
  (void)engine_mapping_.emplace(kEngineNameVectorCore, NodeExecutorManager::ExecutorType::AICORE);  // reuse AIC
  (void)engine_mapping_.emplace(kEngineNameGeLocal, NodeExecutorManager::ExecutorType::GE_LOCAL);
  (void)engine_mapping_.emplace(kEngineNameAiCpuTf, NodeExecutorManager::ExecutorType::AICPU_TF);
  (void)engine_mapping_.emplace(kEngineNameAiCpu, NodeExecutorManager::ExecutorType::AICPU_TF);
  (void)engine_mapping_.emplace(kEngineNameHccl, NodeExecutorManager::ExecutorType::HCCL);
  (void)engine_mapping_.emplace(kEngineNameRts, NodeExecutorManager::ExecutorType::RTS);
  (void)engine_mapping_.emplace(kRtsFftsPlusOpKernelName, NodeExecutorManager::ExecutorType::RTS);
  (void)engine_mapping_.emplace(kEngineNameHostCpu, NodeExecutorManager::ExecutorType::HOST_CPU);

  initialized_ = true;
  GELOGI("Initializing NodeExecutors successfully");
  return SUCCESS;
}

NodeExecutorManager::ExecutorType NodeExecutorManager::ResolveExecutorType(const NodeItem &node_item) const {
  if (node_item.IsFftsSubNode()) {
    return ExecutorType::FFTS;
  }

  const auto &node = *node_item.node;
  const auto op_type = node.GetType();
  if (op_type == PARTITIONEDCALL) {
    const auto &subgraph = NodeUtils::GetSubgraph(node, 0UL);
    if ((subgraph != nullptr) && subgraph->GetGraphUnknownFlag()) {
      return ExecutorType::DYNAMIC_SUBGRAPH;
    }
    bool is_dynamic = false;
    (void)NodeUtils::GetNodeUnknownShapeStatus(node, is_dynamic);
    return is_dynamic ? ExecutorType::DYNAMIC_SUBGRAPH : ExecutorType::COMPILED_SUBGRAPH;
  }

  // rts kernel store is assigned to NetOutput
  if ((op_type == NETOUTPUT) || (op_type == VARIABLE)) {
    return ExecutorType::GE_LOCAL;
  }

  if (IsControlFlowV2Op(op_type)) {
    return ExecutorType::CONTROL_OP;
  }

  const auto op_desc = node.GetOpDesc(); // checked before
  const auto &lib_name = op_desc->GetOpKernelLibName();
  const auto it = engine_mapping_.find(lib_name);
  if (it == engine_mapping_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to get ExecutorType by lib_name:%s, node:%s(%s)",
                       lib_name.c_str(), node.GetName().c_str(), node.GetType().c_str());
    GELOGE(UNSUPPORTED, "[Find][ExecutorType]Failed to get ExecutorType by lib_name:%s, node:%s(%s)",
           lib_name.c_str(), node.GetName().c_str(), node.GetType().c_str());
    return ExecutorType::RESERVED;
  }

  return it->second;
}

Status NodeExecutorManager::GetExecutor(const NodeItem &node_item, const NodeExecutor *&executor) {
  const auto executor_type = ResolveExecutorType(node_item);
  GELOGD("[%s] Set node executor by type: %d.", node_item.NodeName().c_str(), static_cast<int32_t>(executor_type));
  return GetOrCreateExecutor(executor_type, executor);
}

void NodeExecutorManager::RegisterExecutorBuilder(const NodeExecutorManager::ExecutorType executor_type,
                                                  const std::function<std::unique_ptr<NodeExecutor>()> &builder) {
  (void)builders_.emplace(executor_type, builder);
}

bool NodeExecutorManager::IsExecutorInitialized(const NodeExecutorManager::ExecutorType executor_type) const {
  const std::lock_guard<std::mutex> lk(mu_);
  return executors_.find(executor_type) != executors_.end();
}

Status NodeExecutorManager::GetOrCreateExecutor(const ExecutorType executor_type, const NodeExecutor *&out_executor) {
  const std::lock_guard<std::mutex> lk(mu_);
  const std::map<ExecutorType, std::unique_ptr<NodeExecutor>>::const_iterator executor_it =
      executors_.find(executor_type);
  if (executor_it != executors_.cend()) {
    out_executor = executor_it->second.get();
    return SUCCESS;
  }

  GELOGI("Start to Initialize NodeExecutor, type = %d", static_cast<int32_t>(executor_type));
  const std::map<ExecutorType, std::function<std::unique_ptr<NodeExecutor>()>>::const_iterator it =
      builders_.find(executor_type);
  if (it == builders_.cend()) {
    REPORT_INNER_ERR_MSG("E19999", "Create NodeExecutor failed for executor type = %d",
                      static_cast<int32_t>(executor_type));
    GELOGE(INTERNAL_ERROR, "[Create][NodeExecutor] failed for executor type = %d", static_cast<int32_t>(executor_type));
    return INTERNAL_ERROR;
  }

  const auto build_fn = it->second;
  GE_CHECK_NOTNULL(build_fn);
  auto executor = std::unique_ptr<NodeExecutor>(build_fn());
  if (executor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Create NodeExecutor failed for executor type = %d",
                      static_cast<int32_t>(executor_type));
    GELOGE(INTERNAL_ERROR, "[Create][NodeExecutor] failed for engine type = %d", static_cast<int32_t>(executor_type));
    return INTERNAL_ERROR;
  }

  GELOGD("Executor of engine type = %d was created successfully", static_cast<int32_t>(executor_type));
  const auto ret = executor->Initialize();
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Initialize NodeExecutor failed for type = %d", static_cast<int32_t>(executor_type));
    GELOGE(ret, "[Initialize][NodeExecutor] failed for type = %d", static_cast<int32_t>(executor_type));
    return ret;
  }

  out_executor = executor.get();
  (void)executors_.emplace(executor_type, std::move(executor));
  GELOGI("Initializing NodeExecutor successfully, type = %d", static_cast<int32_t>(executor_type));
  return SUCCESS;
}

void NodeExecutorManager::FinalizeExecutors() {
  const std::lock_guard<std::mutex> lk(mu_);
  if (ref_count_ <= 0) {
    GELOGD("No need for finalizing for not initialized.");
    return;
  }

  --ref_count_;
  if (ref_count_ > 0) {
    GELOGD("Ref count = %d, do not finalize executors.", ref_count_);
    return;
  }

  GELOGD("Start to invoke Finalize on executors.");
  for (auto &it : executors_) {
    (void)it.second->Finalize();
  }
  executors_.clear();
  GELOGD("Done invoking Finalize successfully.");
}

NodeExecutorRegistrar::NodeExecutorRegistrar(const NodeExecutorManager::ExecutorType executor_type,
                                             std::unique_ptr<NodeExecutor> (*builder)()) {
  NodeExecutorManager::GetInstance().RegisterExecutorBuilder(executor_type, builder);
}
Status NoOpTask::UpdateArgs(TaskContext &context) {
  GELOGD("[%s] Skipping UpdateArgs for op with empty outputs", context.GetNodeName());
  return SUCCESS;
}

Status NoOpTask::ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) {
  GELOGD("[%s] Skipping execution for op with empty outputs", context.GetNodeName());
  return context.TryExecuteCallback(done_callback);
}
}  // namespace hybrid
}  // namespace ge
