/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/event_handler.h"

#include <string>
#include <future>
#include "executor/executor_context.h"
#include "common/utils/rts_api_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/thread_pool.h"
#include "cpu_tasks.h"

namespace ge {
namespace {
constexpr uint32_t kClearDavinciModelThreadNum = 1U;
constexpr uint32_t kSingleThreadNum = 1U;
constexpr int32_t kClearTypeStop = 1;
constexpr int32_t kClearTypeClear = 2;
constexpr int32_t kMaxParseModelThreadPoolSize = 8;
}
Status EventHandler::Initialize() {
  context_ = MakeUnique<ExecutorContext>();
  GE_CHECK_NOTNULL(context_);
  context_->SetBaseDir(base_dir_);
  GE_CHK_STATUS_RET_NOLOG(context_->Initialize());
  return SUCCESS;
}

void EventHandler::Finalize() {
  context_->Finalize();
  context_.reset();
}

void EventHandler::SetBaseDir(const std::string &base_dir) {
  base_dir_ = base_dir;
}

void EventHandler::HandleEvent(deployer::ExecutorRequest &request,
                               deployer::ExecutorResponse &response) {
  if (request.has_batch_load_model_message()) {
    HandleBatchLoadRequest(request, response);
  } else if (request.has_unload_model_message()) {
    HandleUnloadRequest(request, response);
  } else if (request.has_sync_var_manager_message()) {
    HandleSyncVarManagerRequest(request, response);
  } else if (request.has_clear_model_message()) {
    HandleClearModelRequest(request, response);
  } else if (request.has_exception_notify_request()) {
    HandleDataFlowExceptionNotifyRequest(request, response);
  } else if (request.has_update_prof_message()) {
    HandleProfInfo(request, response);
  } else {
    GELOGE(PARAM_INVALID, "[Handle][Event] failed, request has no content");
    response.set_error_code(UNSUPPORTED);
    response.set_error_message("request has no content");
  }
}

void EventHandler::HandleUnloadRequest(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response) const {
  auto &unload_req = request.unload_model_message();
  auto root_model_id = unload_req.model_id();
  std::map<uint32_t, std::unique_ptr<ExecutorContext::ModelHandle>> *submodel_map = nullptr;
  if (context_->GetModel(root_model_id, submodel_map) != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to get model");
    return;
  }
  GEEVENT("[Batch][UnloadModel] begin, model size = %zu.", submodel_map->size());

  std::mutex failed_mu;
  std::vector<uint32_t> failed;
  std::vector<std::thread> threads;
  rtContext_t ctx = nullptr;
  (void) rtCtxGetCurrent(&ctx);
  const auto &thread_local_ctx = GetThreadLocalContext();
  for (auto it = submodel_map->begin(); it != submodel_map->end(); ++it) {
    auto submodel_id = it->first;
    ExecutorContext::ModelHandle *handle = it->second.get();
    try {
      threads.emplace_back([handle, submodel_id, &failed, &failed_mu, ctx, &thread_local_ctx]() {
        (void) rtCtxSetCurrent(ctx);
        GetThreadLocalContext() = thread_local_ctx;
        if (handle->UnloadModel() != SUCCESS) {
          std::lock_guard<std::mutex> lk(failed_mu);
          failed.emplace_back(submodel_id);
        }
      });
    } catch (std::exception &e) {
      GELOGW("Failed to unload model, submodel_id = %u, exception = %s", submodel_id, e.what());
      std::lock_guard<std::mutex> lk(failed_mu);
      failed.emplace_back(submodel_id);
    }
  }

  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  for (auto it = submodel_map->begin(); it != submodel_map->end();) {
    it = submodel_map->erase(it);
  }
  ExecutorContext().LocalContext().RemoveLocalModel(root_model_id);

  if (!failed.empty()) {
    response.set_error_code(FAILED);
    const std::string error_msg = "Unload model failed, failed model id = " + ToString(failed);
    response.set_error_message(error_msg);
    GELOGW("%s", error_msg.c_str());
    return;
  }

  response.set_error_code(SUCCESS);
  GEEVENT("[Batch][UnloadModel] success.");
}

void EventHandler::HandleSyncVarManagerRequest(deployer::ExecutorRequest &request,
                                               deployer::ExecutorResponse &response) {
  GELOGD("[Handle][Init VarManager] begin.");
  if (context_->SyncSharedVarManager(request) != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to init VarManger");
    return;
  }
  response.set_error_code(SUCCESS);
  GELOGD("[Handle][Init VarManager] success.");
}

void EventHandler::HandleBatchLoadRequest(deployer::ExecutorRequest &request,
                                          deployer::ExecutorResponse &response) {
  if (BatchLoadModels(request) != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to batch load models");
    return;
  }
  response.set_error_code(SUCCESS);
  GELOGD("[Handle][BatchLoadModel] success.");
}

Status EventHandler::BatchParseAndLoadModels(const deployer::ExecutorRequest_BatchLoadModelMessage &model_messages) {
  int32_t pool_size = model_messages.models_size() > kMaxParseModelThreadPoolSize ?
                                                     kMaxParseModelThreadPoolSize :
                                                     model_messages.models_size();
  ThreadPool pool("ge_dpl_prsm", static_cast<uint32_t>(pool_size), false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &load_model_req : model_messages.models()) {
    auto fut = pool.commit([this, &load_model_req]() -> Status {
      GE_CHK_STATUS_RET(context_->ParseModel(load_model_req), "Failed to parse model message.");
      GE_CHK_STATUS_RET(context_->AttachQueues(load_model_req), "Failed to attach model queues.");
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }
  for (int32_t i = 0; i < model_messages.models_size(); ++i) {
    GE_CHK_STATUS_RET(fut_rets[i].get(), "Failed to parse model");
    const auto &load_model_req = model_messages.models(i);
    GE_CHK_STATUS_RET(context_->LoadModel(load_model_req), "Failed to process load model message.");
  }
  return SUCCESS;
}

Status EventHandler::BatchLoadModels(deployer::ExecutorRequest &request) {
  auto &message = request.batch_load_model_message();
  GEEVENT("[Batch][LoadModel] begin, model size = %d.", message.models_size());
  context_->UpdateOptions(message.options());
  if (!once_inited_) {
    GE_CHK_STATUS(context_->SetOpTimeout(), "Failed to set op timeout");
    GE_CHK_STATUS(context_->SetDeviceSatMode(), "Failed to set device sat mode");
    once_inited_ = true;
  }

  GE_CHK_STATUS_RET(BatchParseAndLoadModels(message), "Failed to parse and load models");
  GEEVENT("[Batch][LoadModel] success, model size = %d.", message.models_size());
  return SUCCESS;
}

void EventHandler::HandleClearModelRequest(deployer::ExecutorRequest &request,
                                           deployer::ExecutorResponse &response) {
  GELOGD("[Handle][Clear Model] begin.");
  const auto &clear_model_req = request.clear_model_message();
  // check type
  const auto clear_msg_type = clear_model_req.clear_msg_type();
  if ((clear_msg_type != kClearTypeStop) && (clear_msg_type != kClearTypeClear)) {
    GELOGE(FAILED, "Failed to clear model, invalid type: %d", clear_msg_type);
    const std::string err_msg = "Failed to clear model, invalid type: " +
      std::to_string(clear_msg_type);
    response.set_error_code(FAILED);
    response.set_error_message(err_msg.c_str());
    return;
  }
  const auto root_model_id = clear_model_req.model_id();
  std::map<uint32_t, std::unique_ptr<ExecutorContext::ModelHandle>> *submodel_map = nullptr;
  if (context_->GetModel(root_model_id, submodel_map) != SUCCESS) {
    GELOGE(FAILED, "Failed to get model: %u", root_model_id);
    const std::string err_msg = "Failed to get model, model id: " +
      std::to_string(root_model_id);
    response.set_error_code(FAILED);
    response.set_error_message(err_msg.c_str());
    return;
  }
  std::vector<uint32_t> davinci_model_runtime_ids;
  std::vector<ExecutorContext::ModelHandle *> dynamic_model_handles;
  for (auto it = submodel_map->begin(); it != submodel_map->end(); it++) {
    ExecutorContext::ModelHandle *handle = it->second.get();
    if (handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids, dynamic_model_handles) != SUCCESS) {
      response.set_error_code(FAILED);
      response.set_error_message("Failed to get clear model handle");
      return;
    }
  }
  DoClearModel(davinci_model_runtime_ids, dynamic_model_handles, clear_msg_type, response);
  GELOGD("[Handle][Clear Model] end.");
}

void EventHandler::DoClearModel(const std::vector<uint32_t> &davinci_model_runtime_ids,
                                const std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles,
                                const int32_t clear_msg_type,
                                deployer::ExecutorResponse &response) const {
  uint32_t parallel_num = static_cast<uint32_t>(dynamic_model_handles.size());
  if (!davinci_model_runtime_ids.empty()) {
    parallel_num += kClearDavinciModelThreadNum;
  }
  if (parallel_num > kSingleThreadNum) {
    return DoClearModelPara(davinci_model_runtime_ids, dynamic_model_handles, clear_msg_type, response);
  }
  for (const auto dynamic_model_handle : dynamic_model_handles) {
    const auto ret = dynamic_model_handle->ClearModel(clear_msg_type);
    if (ret != SUCCESS) {
      response.set_error_code(FAILED);
      response.set_error_message("Failed to clear model");
      return;
    }
  }
  const auto ret = CpuTasks::ExecuteModelClearTask(clear_msg_type, davinci_model_runtime_ids);
  if (ret != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to clear model");
    return;
  }
  response.set_error_code(SUCCESS);
}

void EventHandler::DoClearModelPara(const std::vector<uint32_t> &davinci_model_runtime_ids,
                                    const std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles,
                                    const int32_t clear_msg_type,
                                    deployer::ExecutorResponse &response) const {
  // clean dynamic model async
  const uint32_t parallel_num = static_cast<uint32_t>(dynamic_model_handles.size());
  ThreadPool pool("ge_dpl_clrm", static_cast<uint32_t>(parallel_num), false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto dynamic_model_handle : dynamic_model_handles) {
    auto fut = pool.commit([dynamic_model_handle, clear_msg_type]() -> Status {
      GE_CHK_STATUS_RET_NOLOG(dynamic_model_handle->ClearModel(clear_msg_type));
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }
  // clean davinci model sync
  const auto ret = CpuTasks::ExecuteModelClearTask(clear_msg_type, davinci_model_runtime_ids);
  if (ret != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to clear model");
    return;
  }
  // wait until clean dynamic model sync
  for (auto &fut : fut_rets) {
    if (fut.get() != SUCCESS) {
      response.set_error_code(FAILED);
      response.set_error_message("Failed to clear model");
      return;
    }
  }
  response.set_error_code(SUCCESS);
}

void EventHandler::HandleDataFlowExceptionNotifyRequest(deployer::ExecutorRequest &request,
                                                        deployer::ExecutorResponse &response) {
  GELOGD("[Handle][Exception notify] begin.");
  const auto &exception_notify_request = request.exception_notify_request();
  const auto &exception_notify = exception_notify_request.exception_notify();
  const auto root_model_id = exception_notify_request.root_model_id();
  std::map<uint32_t, std::unique_ptr<ExecutorContext::ModelHandle>> *submodel_map = nullptr;
  if (context_->GetModel(root_model_id, submodel_map) != SUCCESS) {
    GELOGE(FAILED, "No model found, model: %u", root_model_id);
    response.set_error_code(FAILED);
    response.set_error_message("No model found");
    return;
  }
  std::vector<uint32_t> davinci_model_runtime_ids;
  std::vector<ExecutorContext::ModelHandle *> dynamic_model_handles;
  for (auto it = submodel_map->begin(); it != submodel_map->end(); it++) {
    ExecutorContext::ModelHandle *handle = it->second.get();
    if ((handle->GetScope() != exception_notify.scope()) || handle->IsInvokedNN()) {
      GELOGI("scope[%s] is mismatch exception scope[%s] or is invoked nn[%d], ignore it, model: %u",
             handle->GetScope().c_str(), exception_notify.scope().c_str(), static_cast<int32_t>(handle->IsInvokedNN()),
             root_model_id);
      continue;
    }
    auto get_ret = handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids, dynamic_model_handles);
    if (get_ret != SUCCESS) {
      GELOGE(get_ret, "Failed to get model: %u", root_model_id);
      response.set_error_code(get_ret);
      response.set_error_message("get model runtime id or handle failed");
      return;
    }
  }

  Status notify_ret = DoDataFlowExceptionNotify(davinci_model_runtime_ids, dynamic_model_handles, exception_notify);
  if (notify_ret != SUCCESS) {
    response.set_error_code(notify_ret);
    response.set_error_message("data flow exception notify failed");
  } else {
    response.set_error_code(SUCCESS);
  }
  GELOGD("[Handle][Exception notify] end.");
}
Status EventHandler::DoDataFlowExceptionNotify(const std::vector<uint32_t> &davinci_model_runtime_ids,
                                               const std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles,
                                               const deployer::DataFlowExceptionNotify &exception_notify) const {
  uint32_t type = exception_notify.type();
  uint64_t trans_id = exception_notify.trans_id();
  // notify dynamic model async
  const uint32_t parallel_num = static_cast<uint32_t>(dynamic_model_handles.size());
  std::vector<std::future<Status>> fut_rets;
  std::unique_ptr<ThreadPool> thread_pool;
  if (parallel_num > 0) {
    fut_rets.reserve(parallel_num);
    thread_pool = MakeUnique<ThreadPool>("ge_dpl_hexn", parallel_num, false);
    if (thread_pool == nullptr) {
      GELOGE(FAILED, "make unique for ThreadPool failed, parallel_num=%u", parallel_num);
      return FAILED;
    }
    for (const auto dynamic_model_handle : dynamic_model_handles) {
      auto fut = thread_pool->commit([dynamic_model_handle, type, trans_id]() -> Status {
        GE_CHK_STATUS_RET_NOLOG(dynamic_model_handle->ExceptionNotify(type, trans_id));
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
  }
  // davinci model sync
  const auto ret = CpuTasks::ExceptionNotify(davinci_model_runtime_ids, type, trans_id);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "ExecuteDataFlowExceptionNotify failed, model runtime ids=%s",
           ToString(davinci_model_runtime_ids).c_str());
    return ret;
  }
  // wait until dynamic model sync
  for (auto &fut : fut_rets) {
    auto fut_ret = fut.get();
    if (fut_ret != SUCCESS) {
      GELOGE(fut_ret, "DataFlowExceptionNotify to dynamic model failed, trans_id=%lu", trans_id);
      return fut_ret;
    }
  }
  return SUCCESS;
}

void EventHandler::HandleProfInfo(deployer::ExecutorRequest &request,
                                  deployer::ExecutorResponse &response) {
  GELOGI("[Handle][Set Proiling Info] begin.");
  if (context_->UpdateProfInfo(request) != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to set prof");
    return;
  }
  response.set_error_code(SUCCESS);
  GELOGI("[Handle][Set Proiling Info] end.");
}
}  // namespace ge
