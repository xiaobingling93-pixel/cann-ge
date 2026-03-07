/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_model_manager.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "graph/ge_global_options.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "dflow/base/model/flow_model_om_loader.h"

namespace ge {
FlowModelManager &FlowModelManager::GetInstance() {
  static FlowModelManager instance;
  return instance;
}

Status FlowModelManager::DoLoadFlowModel(uint32_t model_id, const FlowModelPtr &flow_model) {
  auto *const execution_runtime = ExecutionRuntime::GetInstance();
  GE_CHECK_NOTNULL(execution_runtime);
  auto &model_deployer = execution_runtime->GetModelDeployer();
  DeployResult deploy_result{};
  GE_CHK_STATUS_RET(model_deployer.DeployModel(flow_model, deploy_result), "deploy model failed, model_id:%u.",
                    model_id);
  auto executor = MakeShared<HeterogeneousModelExecutor>(flow_model, deploy_result);

  ScopeGuard load_guard([this, executor, &deploy_result]() { StopAndUnloadModel(executor, deploy_result.model_id); });

  GE_CHECK_NOTNULL(executor);
  executor->SetModelId(model_id);
  GE_CHK_STATUS_RET(executor->Initialize(), "executor init failed, model_id:%u.", model_id);
  GE_CHK_STATUS_RET(executor->ModelRunStart(), "[Start][Model] failed, model_id:%u.", model_id);
  load_guard.Dismiss();
  flow_model->SetModelId(model_id);
  const std::lock_guard<std::mutex> lk(map_mutex_);
  (void)heterogeneous_model_map_.emplace(model_id, std::move(executor));
  return SUCCESS;
}

Status FlowModelManager::LoadFlowModel(uint32_t &model_id, const FlowModelPtr &flow_model) {
  GenModelId(model_id);
  GELOGD("Generate new model_id:%u", model_id);

  domi::GetContext().is_online_model = true;
  GE_CHK_STATUS_RET(DoLoadFlowModel(model_id, flow_model), "load flow model failed, model_id=%u.", model_id);
  GELOGI("load flow model success, model_id:%u", model_id);
  return SUCCESS;
}

Status FlowModelManager::ExecuteFlowModel(uint32_t model_id, const std::vector<GeTensor> &inputs,
                                          std::vector<GeTensor> &outputs) {
  const auto executor = GetHeterogeneousModelExecutor(model_id);
  GE_CHECK_NOTNULL(executor, ", find flow model executor failed, model_id=%u.", model_id);
  return executor->Execute(inputs, outputs);
}

FlowModelPtr FlowModelManager::GetFlowModelByModelId(uint32_t model_id) {
  const std::lock_guard<std::mutex> lk(map_mutex_);
  const auto iter = heterogeneous_model_map_.find(model_id);
  if ((iter == heterogeneous_model_map_.cend()) || (iter->second == nullptr)) {
    GELOGW("Can not get valid flow model.");
    return nullptr;
  }
  return iter->second->GetFlowModel();
}

bool FlowModelManager::IsLoadedByFlowModel(uint32_t model_id) {
  const std::lock_guard<std::mutex> lk(map_mutex_);
  return heterogeneous_model_map_.find(model_id) != heterogeneous_model_map_.cend();
}

Status FlowModelManager::StopAndUnloadModel(const std::shared_ptr<HeterogeneousModelExecutor> &executor,
                                            uint32_t deployed_model_id) const {
  if (executor != nullptr) {
    (void)executor->ModelRunStop();
  }
  auto *execution_runtime = ExecutionRuntime::GetInstance();
  GE_ASSERT_NOTNULL(execution_runtime);
  (void)execution_runtime->GetModelDeployer().Undeploy(deployed_model_id);
  return SUCCESS;
}

Status FlowModelManager::Unload(uint32_t model_id) {
  const auto executor = GetHeterogeneousModelExecutor(model_id);
  if (executor == nullptr) {
    GELOGW("heterogeneous model %u is not loaded, no need unload.", model_id);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(StopAndUnloadModel(executor, executor->GetDeployedModelId()), "Unload model executor failed.");

  const std::lock_guard<std::mutex> lk(map_mutex_);
  (void)heterogeneous_model_map_.erase(model_id);
  return SUCCESS;
}

std::shared_ptr<HeterogeneousModelExecutor> FlowModelManager::GetHeterogeneousModelExecutor(uint32_t model_id) {
  const std::lock_guard<std::mutex> lk(map_mutex_);
  const auto it = heterogeneous_model_map_.find(model_id);
  if (it == heterogeneous_model_map_.end()) {
    return nullptr;
  }
  return it->second;
}

void FlowModelManager::GenModelId(uint32_t &id) {
  id = max_model_id_.fetch_add(1);
}
}  // namespace ge