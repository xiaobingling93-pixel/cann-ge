/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/cpu_sched_event_dispatcher.h"
#include "mmpa/mmpa_api.h"
#include "common/utils/rts_api_utils.h"
#include "executor/dynamic_model_executor.h"
#include "runtime/rt_mem_queue.h"
#include "rt_error_codes.h"
#include "aicpu/aicpu_schedule/aicpusd_interface.h"
#include "graph/ge_context.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
uint32_t kAiCpuSubEventIdEndGraph = 6U;
uint32_t kAiCpuSubEventIdActivateModel = 7U;
}  // namespace

CpuSchedEventDispatcher::~CpuSchedEventDispatcher() {
  if (running_) {
    Finalize();
  }
}

Status CpuSchedEventDispatcher::Initialize(int32_t device_id, bool host_exec_flag) {
  host_exec_flag_ = host_exec_flag;
  const std::string kAicpu = "libaicpu_scheduler.so";
  const std::string kHostAicpu = "libhost_aicpu_scheduler.so";
  const std::string aicpu_so_name = host_exec_flag_ ? kHostAicpu : kAicpu;
  aicpu_handle_ = mmDlopen(aicpu_so_name.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
      static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  if (aicpu_handle_ == nullptr) {
    GELOGW("Dispatcher dlopen %s failed with error %s.", aicpu_so_name.c_str(), mmDlerror());
    return SUCCESS;
  }

  device_id_ = device_id;
  aicpu_sd_pid_ = mmGetPid();
  GE_CHK_STATUS_RET(RtsApiUtils::EschedCreateGroup(device_id_, event_group_id_, RT_GRP_TYPE_BIND_CP_CPU),
                    "Failed to create group, device_id = %d", device_id);
  uint64_t event_bitmap = 1ULL << static_cast<uint32_t>(RT_EVENT_AICPU_MSG);
  GE_CHK_STATUS_RET(RtsApiUtils::EschedSubscribeEvent(device_id_, event_group_id_, 0, event_bitmap),
                    "Failed to subscribe event, device_id = %d", device_id_);
  CpuSchedInitParam init_param{};
  init_param.deviceId = device_id_;
  init_param.hostPid = aicpu_sd_pid_;
  init_param.profilingMode = PROFILING_CLOSE;
  const std::string kInitFunc = "InitCpuScheduler";
  const auto init_func =
      reinterpret_cast<int32_t (*)(const CpuSchedInitParam * const)>(mmDlsym(aicpu_handle_, kInitFunc.c_str()));
  GE_CHECK_NOTNULL(init_func);
  auto ret = init_func(&init_param);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED, "Failed to invoke InitHostAICPUScheduler, ret = %d", ret);
  running_ = true;
  event_handle_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_ehdl");
    this->ProcessEvents();
  });
  return SUCCESS;
}

Status CpuSchedEventDispatcher::OnInputsReady(rtEschedEventSummary_t &in_event) {
  if (in_event.msgLen < sizeof(AICPUSubEventInfo)) {
    GELOGE(PARAM_INVALID, "event msg length is insufficient for AICPUSubEventInfo, msgLen = %u", in_event.msgLen);
    return PARAM_INVALID;
  }

  auto model_id = reinterpret_cast<AICPUSubEventInfo *>(in_event.msg)->modelId;
  GELOGD("On activate model event, model_id = %u", model_id);

  std::lock_guard<std::mutex> lk(mu_);
  const auto &it = models_.find(model_id);
  if (it == models_.end()) {
    GELOGW("model id not found, id = %u", model_id);
    return SUCCESS;
  }

  auto callback = [model_id, this](Status status, void *req_mbuf, void *resp_mbuf) {
    (void) req_mbuf;
    (void) resp_mbuf;
    OnModelExecuted(model_id, status);
    if (status != SUCCESS) {
      GELOGE(FAILED, "Execute model failed, model_id = %u", model_id);
      running_ = false;
    }
  };

  auto &model_executor = *it->second;
  GE_CHK_STATUS_RET(model_executor.ExecuteAsync(callback), "Failed to submit task, model id = %u", model_id);
  GELOGD("Activate model success, model_id = %u", model_id);
  return SUCCESS;
}

void CpuSchedEventDispatcher::ProcessEvents() {
  GELOGI("Process thread started.");
  const int32_t timeout = 10 * 1000;
  while (running_) {
    rtEschedEventSummary_t in_event = {};
    char_t msg[RT_EVENT_MAX_MSG_LEN] = {};
    in_event.msg = msg;
    in_event.msgLen = RT_EVENT_MAX_MSG_LEN;
    const auto ret = rtEschedWaitEvent(device_id_, event_group_id_, 0, timeout, &in_event);
    if (ret == ACL_ERROR_RT_REPORT_TIMEOUT) {
      GELOGI("wait timeout, continue");
      continue;
    }
    if (ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Failed to invoke rtEschedWaitEvent, device_id = %d, group_id = %u, ret = 0x%X",
             device_id_, event_group_id_, ret);
      running_ = false;
      return;
    }
    if (in_event.eventId != RT_EVENT_AICPU_MSG || in_event.subeventId != kAiCpuSubEventIdActivateModel) {
      continue;
    }

    if (OnInputsReady(in_event) != SUCCESS) {
      running_ = false;
      return;
    }
  }
  GELOGI("Process thread ended.");
}

Status CpuSchedEventDispatcher::OnModelExecuted(uint32_t model_id, Status status) const {
  // notify aicpu-sd
  (void) model_id;
  GELOGD("Notify model execution ended, model_id = %u, status = %u.", model_id, status);
  AICPUSubEventInfo sub_event_info{};
  sub_event_info.modelId = model_id;
  sub_event_info.para.endGraphInfo.result = status;

  rtEschedEventSummary_t event_info{};
  event_info.eventId = RT_EVENT_AICPU_MSG;
  event_info.pid = aicpu_sd_pid_;
  event_info.grpId = 0U;  // aicpu event group
  event_info.subeventId = kAiCpuSubEventIdEndGraph;  // AICPU_SUB_EVENT_END_GRAPH
  event_info.msg = reinterpret_cast<char_t *>(&sub_event_info);
  event_info.msgLen = sizeof(sub_event_info);
  if (host_exec_flag_) {
    event_info.dstEngine = static_cast<uint32_t>(RT_MQ_DST_ENGINE_CCPU_HOST);
  }

  GE_CHK_STATUS_RET(RtsApiUtils::EschedSubmitEvent(device_id_, event_info),
                    "[Send][Event] failed, device_id = %d", device_id_);
  GELOGD("[Send][Event] succeeded, device_id = %d", device_id_);
  return SUCCESS;
}

void CpuSchedEventDispatcher::Finalize() {
  running_ = false;
  if (event_handle_thread_.joinable()) {
    event_handle_thread_.join();
  }

  if (aicpu_handle_ != nullptr) {
    const std::string kStopFunc = "StopCPUScheduler";
    const auto stop_func =
        reinterpret_cast<int32_t (*)(const uint32_t deviceId,
                                     const pid_t hostPid)>(mmDlsym(aicpu_handle_, kStopFunc.c_str()));
    if (stop_func != nullptr) {
      (void) stop_func(device_id_, aicpu_sd_pid_);
    }
    (void) mmDlclose(aicpu_handle_);
    aicpu_handle_ = nullptr;
  }
}

void CpuSchedEventDispatcher::Register(uint32_t model_id, DynamicModelExecutor *executor) {
  std::lock_guard<std::mutex> lk(mu_);
  models_[model_id] = executor;
}

void CpuSchedEventDispatcher::Deregister(uint32_t model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  models_.erase(model_id);
}
}  // namespace ge
