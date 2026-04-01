/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/execfwk/builtin_executor_client.h"
#include <signal.h>
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/mem_grp/memory_group_manager.h"
#include "common/subprocess/subprocess_manager.h"
#include "common/utils/rts_api_utils.h"
#include "dflow/base/utils/process_utils.h"
#include "mmpa/mmpa_api.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "deploy/flowrm/flowgw_client.h"
#include "prof_common.h"

namespace ge {
namespace {
constexpr uint32_t kMaxExecutorShutdownWaitTimeInSec = 60;
constexpr const char_t * const kArgsKeyBaseDir = "--base_dir";
constexpr const char_t * const kArgsKeyMsgQueueDeviceId = "--msg_queue_device_id";

PneExecutorClientCreatorRegistrar<BuiltinExecutorClient> __attribute__((unused)) npu_client_reg(PNE_ID_NPU);
PneExecutorClientCreatorRegistrar<HostCpuExecutorClient> __attribute__((unused)) cpu_client_reg(PNE_ID_CPU);
}  // namespace

BuiltinExecutorClient::BuiltinExecutorClient(int32_t device_id, bool is_host)
    : PneExecutorClient(device_id), is_host_(is_host),
      sub_proc_stat_(ProcStatus::NORMAL) {
}

Status BuiltinExecutorClient::Initialize() {
  if (GetContext().device_type != static_cast<int32_t>(CPU)) {
    // 1971 helper host config set
    MsprofConfigParam param = {};
    param.deviceId = static_cast<uint32_t>(GetDeviceId());
    param.type = MsprofConfigParamType::DEV_CHANNEL_RESOURCE;
    param.value = 1;
    (void) MsprofSetConfig(0, reinterpret_cast<const char *>(&param), sizeof(param));
  }
  GE_CHK_STATUS_RET(ForAndInit(GetDeviceId(), message_client_), "Failed to fork and init");
  heartbeat_listening_ = true;
  return SUCCESS;
}

void BuiltinExecutorClient::Shutdown() {
  (void)SubprocessManager::GetInstance().ShutdownSubprocess(pid_, kMaxExecutorShutdownWaitTimeInSec);
}

Status BuiltinExecutorClient::Finalize() {
  GEEVENT("BuiltinExecutorClient begin, executor pid is %d.", GetPid());
  heartbeat_listening_ = false;
  if (message_client_ != nullptr) {
    // notify queue wake up
    (void) message_client_->NotifyFinalize();
  }
  Shutdown();
  message_client_.reset();
  return SUCCESS;
}

Status BuiltinExecutorClient::ForAndInit(int32_t device_id, std::unique_ptr<ExecutorMessageClient> &message_client) {
  // 1. create message queue
  int32_t msg_queue_device_id = 0;
  message_client = CreateExecutorMessageClient(msg_queue_device_id);
  GE_CHECK_NOTNULL(message_client);
  const std::string name_suffix = "executor_" + std::to_string(device_id);
  uint32_t req_msg_queue_id = UINT32_MAX;
  uint32_t rsp_msg_queue_id = UINT32_MAX;
  GE_CHK_STATUS_RET(message_client->CreateMessageQueue(name_suffix, req_msg_queue_id, rsp_msg_queue_id),
                    "Failed to create message queue");

  // 2. for child executor process
  const auto &group_name = MemoryGroupManager::GetInstance().GetQsMemGroupName();
  GE_CHK_STATUS_RET_NOLOG(DoForkChildProcess(device_id, req_msg_queue_id, rsp_msg_queue_id, group_name));
  GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().MemGrpAddProc(group_name, pid_, false, true),
                    "Failed to add group, pid = %d", pid_);
  GELOGD("[Fork][Process] succeeded, pid = %d.", pid_);

  // 3. initialize message client
  const auto &get_stat_func = [this, device_id]() -> Status {
    return (sub_proc_stat_.load() == ProcStatus::EXITED) ? FAILED : SUCCESS;
  };
  GE_CHK_STATUS_RET(message_client->Initialize(pid_, get_stat_func), "Failed to initialize executor process");
  return SUCCESS;
}

Status BuiltinExecutorClient::DoForkChildProcess(int32_t device_id,
                                                 uint32_t req_msg_queue_id,
                                                 uint32_t rsp_msg_queue_id,
                                                 const std::string &group_name) {
  std::string bin_dir;
  SubprocessManager::SubprocessConfig config{};
  config.process_type = is_host_ ? PNE_ID_CPU : PNE_ID_NPU;
  config.death_signal = SIGKILL;
  std::string process_name = is_host_ ? "host_cpu_executor" : "npu_executor";

  // get options
  auto process_cfg = GetDevMaintenanceCfg();
  if (process_cfg != nullptr) {
    GE_CHK_STATUS_RET(process_cfg->DecodeConfig(config.envs, config.kv_args), "Decode config failed.");
  }
  const std::string enable = "1";
  config.envs.emplace("AICPU_ADD_BUFFERGROUP", enable);
  config.args = {process_name,
                 group_name,
                 std::to_string(req_msg_queue_id),
                 std::to_string(rsp_msg_queue_id),
                 std::to_string(device_id)};
  (void) GenerateKvArgs(config.kv_args);
  GE_CHK_STATUS_RET_NOLOG(SubprocessManager::GetInstance().ForkSubprocess(config, pid_));
  // watch subprocess
  GELOGI("Start to watch subprocess, pid[%d].", static_cast<int32_t>(pid_));
  std::function<void(const ProcStatus &)> excpt_handle_callback = [this](const ProcStatus &proc_status) {
    GEEVENT("Executor process status is %d.", static_cast<int32_t>(proc_status));
    sub_proc_stat_ = proc_status;
  };
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid_, excpt_handle_callback);
  GELOGI("Fork subprocess successfully, engine_type = %s, pid = %d", process_name.c_str(), pid_);
  return SUCCESS;
}

Status BuiltinExecutorClient::LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) {
  GELOGD("[Load][Model] begin.");
  GE_CHECK_NOTNULL(message_client_);
  GE_CHK_STATUS_RET_NOLOG(GrantQueues(load_model_desc));
  GE_CHK_STATUS_RET_NOLOG(GrantDynamicSchedQueues(load_model_desc));
  GE_TIMESTAMP_START(PreLoadModel);
  GE_CHK_STATUS_RET(PreLoadProcess(load_model_desc), "Failed to pre load process.");
  GE_TIMESTAMP_EVENT_END(PreLoadModel, "PreLoadModel");
  GE_MAKE_GUARD(after_process, ([this, &load_model_desc]() { AfterLoadProcess(load_model_desc); }));
  deployer::ExecutorRequest executor_request;
  GEEVENT("Load model on executor, model_type = %s, pid = %d, graph_id = %u, deployer pid = %d, device_id = %d.",
          PNE_ID_NPU.c_str(), GetPid(), load_model_desc.graph_id(), GetDeployerPid(), GetDeviceId());
  *executor_request.mutable_batch_load_model_message() = load_model_desc;
  deployer::ExecutorResponse executor_response;
  constexpr int64_t kTimeoutSec = 8400; // s
  GE_CHK_STATUS_RET(message_client_->SendRequest(executor_request, executor_response, kTimeoutSec),
                    "[Load][Model] Failed to send request");
  if (executor_response.error_code() != SUCCESS) {
    GELOGE(FAILED, "[Load][Model] failed, error_message = %s", executor_response.error_message().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::UpdateProfilingFromExecutor(deployer::ExecutorRequest_UpdateProfRequest &prof_message) {
  GELOGD("[Update][profiling] begin.");
  if (!is_host_) {
    GELOGI("[Update][profiling] only support host for now.");
    return SUCCESS;
  }
  deployer::ExecutorRequest executor_request;
  *executor_request.mutable_update_prof_message() = prof_message;
  deployer::ExecutorResponse executor_response;
  constexpr int64_t kTimeoutSec = 300; // s
  GE_CHK_STATUS_RET(message_client_->SendRequest(executor_request, executor_response, kTimeoutSec),
                    "[Update][profiling] Failed to send request to executor, device_id = %d", GetDeviceId());
  GE_CHK_BOOL_RET_STATUS(executor_response.error_code() == SUCCESS, FAILED,
                         "[Update][profiling] failed, error message=%s.",
                         executor_response.error_message().c_str());
  return SUCCESS;
}

Status BuiltinExecutorClient::UnloadModel(uint32_t model_id) {
  GELOGD("[Unload][Model] begin.");
  deployer::ExecutorRequest executor_request;
  auto exec_req_body = executor_request.mutable_unload_model_message();
  GE_CHECK_NOTNULL(exec_req_body);
  exec_req_body->set_model_id(model_id);
  deployer::ExecutorResponse executor_response;
  constexpr int64_t kTimeoutSec = 300; // s
  GE_CHK_STATUS_RET(message_client_->SendRequest(executor_request, executor_response, kTimeoutSec),
                    "[Unload][Model] Failed to send request to executor, device_id = %d", GetDeviceId());
  if (executor_response.error_code() != SUCCESS) {
    GELOGE(FAILED, "[Unload][Model] failed, request = %s, error_message = %s",
           exec_req_body->DebugString().c_str(), executor_response.error_message().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::ClearModelRunningData(uint32_t model_id, int32_t type,
                                                    const std::set<int32_t> &device_ids) {
  GELOGD("[ClearModelExceptionData] begin.");
  (void)device_ids;
  if (!is_valid_) {
    GELOGI("Executor is not valid.");
    return SUCCESS;
  }
  deployer::ExecutorRequest executor_request;
  auto clear_req_body = executor_request.mutable_clear_model_message();
  GE_CHECK_NOTNULL(clear_req_body);
  clear_req_body->set_model_id(model_id);
  clear_req_body->set_clear_msg_type(type);

  deployer::ExecutorResponse executor_response;
  constexpr int64_t kTimeoutSec = 60; // s
  GE_CHK_STATUS_RET(message_client_->SendRequest(executor_request, executor_response, kTimeoutSec),
                    "[ClearModelExceptionData] Failed to send request to executor, device_id = %d, "
                    "root_model_id = %u, msg_type = %d", GetDeviceId(), model_id, type);
  if (executor_response.error_code() != SUCCESS) {
    GELOGE(FAILED, "[ClearModelExceptionData] failed, request = %s, error_message = %s",
           clear_req_body->DebugString().c_str(), executor_response.error_message().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) {
  if (!is_valid_) {
    GELOGI("Executor is not valid.");
    return SUCCESS;
  }
  deployer::ExecutorRequest executor_request;
  executor_request.set_type(deployer::ExecutorRequestType::kExecutorExceptionNotify);
  auto exception_notify_req_body = executor_request.mutable_exception_notify_request();
  GE_CHECK_NOTNULL(exception_notify_req_body);
  *exception_notify_req_body = req_body;

  deployer::ExecutorResponse executor_response;
  GE_CHK_STATUS_RET(
      message_client_->SendRequest(executor_request, executor_response),
      "[DataFlowExceptionNotify] Failed to send request to executor, device_id = %d, trans_id = %lu, type=%u",
      GetDeviceId(), exception_notify_req_body->exception_notify().trans_id(),
      exception_notify_req_body->exception_notify().type());
  if (executor_response.error_code() != SUCCESS) {
    GELOGE(FAILED, "[DataFlowExceptionNotify] failed, request = %s, error_code=%u, error_message = %s",
           executor_request.DebugString().c_str(), executor_response.error_code(),
           executor_response.error_message().c_str());
    return FAILED;
  }
  GELOGI("send request to executor end, device_id = %d, trans_id = %lu, type=%u", GetDeviceId(),
         exception_notify_req_body->exception_notify().trans_id(),
         exception_notify_req_body->exception_notify().type());
  return SUCCESS;
}

ProcStatus BuiltinExecutorClient::GetSubProcStat() {
  if (!heartbeat_listening_) {
    return ProcStatus::NORMAL;
  }

  is_valid_ = is_valid_ ? (sub_proc_stat_.load() == ProcStatus::NORMAL) : is_valid_;
  return sub_proc_stat_.load();
}

Status BuiltinExecutorClient::GetPidOwningIoQueues(int32_t &pid) {
  if (is_host_) {
    GELOGD("Get cpu schedule pid = %d success", pid_);
    pid = pid_;
    return SUCCESS;
  }

  if (aicpu_pid_ != -1) {
    pid = aicpu_pid_;
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(RtsApiUtils::GetAicpuSchedulePid(GetDeviceId(), pid_, pid),
                    "Query aicpu schedule failed, deviceId=%d, hostPid=%d.", GetDeviceId(), pid_);
  aicpu_pid_ = pid;
  GELOGD("Get cpu schedule pid = %d success.", pid);

  // load model on aicpu, need to add group for aicpu
  if (GetContext().device_type != static_cast<int32_t>(CPU)) {
    const auto &group_name = MemoryGroupManager::GetInstance().GetRemoteMemGroupName(GetDeviceId());
    GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().RemoteMemGrpAddProc(GetDeviceId(),
                                                                            group_name,
                                                                            aicpu_pid_, false, true),
                      "Failed to add group for aicpu, pid = %d", aicpu_pid_);
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) {
  deployer::ExecutorRequest executor_request;
  *executor_request.mutable_sync_var_manager_message() = std::move(sync_var_manage_desc);
  deployer::ExecutorResponse executor_response;
  GE_CHK_STATUS_RET(message_client_->SendRequest(executor_request, executor_response),
                    "[Sync][VarManager] Failed to send request to executor, device_id = %d", GetDeviceId());
  if (executor_response.error_code() != SUCCESS) {
    GELOGE(FAILED, "[Sync][VarManager] failed, request = %s, error_message = %s",
           executor_request.DebugString().c_str(), executor_response.error_message().c_str());
    return FAILED;
  }
  GELOGD("[Sync][VarManager] succeeded, device_id = %d.", GetDeviceId());
  return SUCCESS;
}

Status BuiltinExecutorClient::DoGrantQueues(int32_t pid, const std::vector<DeployQueueAttr> &queue_attrs) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &queue_info = grant_queues_map_[pid];
  for (const auto &queue_attr : queue_attrs) {
    const auto &key = queue_attr.GetKey();
    const auto &it = queue_info.find(key);
    if (it != queue_info.cend()) {
      continue;
    }
    GE_CHK_STATUS_RET(FlowGwClient::GrantQueue(static_cast<uint32_t>(queue_attr.device_id), queue_attr.queue_id,
                                               pid, GrantType::kReadAndWrite),
                      "Grant queue failed, device id=%d, queue id=%u, pid = %d",
                      queue_attr.device_id, queue_attr.queue_id, pid);
    queue_info.emplace(key);
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::GrantQueues(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc) {
  for (const auto &model : load_model_desc.models()) {
    int32_t pid = 0;
    GE_CHK_STATUS_RET(GetPidOwningIoQueues(pid), "Failed to get pid");
    GE_CHK_STATUS_RET(GrantQueuesForProcess(pid, GetContext().device_type, model.model_queues_attrs()),
                      "Failed to grant queues, pid = %d", pid);
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::GrantDynamicSchedQueues(
    const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc) {
  for (const auto &model : load_model_desc.models()) {
    int32_t pid = 0;
    GE_CHK_STATUS_RET(GetPidOwningIoQueues(pid), "Failed to get pid");
    for (auto input_queue : model.status_queues().input_queues_attrs()) {
      GELOGI("DynamicSched Grant queues, status input queue id=%u, pid=%d", input_queue.queue_id(), pid);
    }
    for (auto output_queue : model.status_queues().output_queues_attrs()) {
      GELOGI("DynamicSched Grant queues, status output queue id=%u, pid=%d", output_queue.queue_id(), pid);
    }
    GE_CHK_STATUS_RET(GrantQueuesForProcess(pid, GetContext().device_type, model.status_queues()),
                      "Failed to grant queues, pid = %d", pid);
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::GenerateKvArgs(std::map<std::string, std::string> &kv_args) {
  const auto &context = GetContext();
  kv_args[kArgsKeyBaseDir] = context.base_dir;
  int32_t msg_queue_device_id = 0;
  kv_args[kArgsKeyMsgQueueDeviceId] = std::to_string(msg_queue_device_id);
  const auto &mode_it = context.options.find(OPTION_EXEC_PROFILING_MODE);
  if (mode_it != context.options.cend()) {
    kv_args[OPTION_EXEC_PROFILING_MODE] = mode_it->second;
  }
  const auto &options_it = context.options.find(OPTION_EXEC_PROFILING_OPTIONS);
  if (options_it != context.options.cend()) {
    kv_args[OPTION_EXEC_PROFILING_OPTIONS] = options_it->second;
  }
  return SUCCESS;
}

Status BuiltinExecutorClient::DoBindHostPid(const int32_t pid) {
  std::lock_guard<std::mutex> guard(pid_mutex_);
  const auto &it = bind_pids_.find(pid);
  if (it == bind_pids_.cend()) {
    GE_CHK_STATUS_RET(BindHostPid(pid), "Failed to bind host pid");
    bind_pids_.emplace(pid);
  }
  return SUCCESS;
}

HostCpuExecutorClient::HostCpuExecutorClient(int32_t device_id) : BuiltinExecutorClient(device_id, true) {}
}  // namespace ge
