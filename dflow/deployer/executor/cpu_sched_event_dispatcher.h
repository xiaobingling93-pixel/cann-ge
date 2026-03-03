/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_EVENT_DISPATCHER_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_EVENT_DISPATCHER_H_

#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include "runtime/rt_mem_queue.h"
#include "ge/ge_api_error_codes.h"
#include "executor/dynamic_model_executor.h"

namespace ge {
class CpuSchedEventDispatcher {
 public:
  static CpuSchedEventDispatcher &GetInstance() {
    static CpuSchedEventDispatcher instance;
    return instance;
  }

  ~CpuSchedEventDispatcher();

  Status Initialize(int32_t device_id, bool host_exec_flag);

  void Finalize();

  void Register(uint32_t model_id, DynamicModelExecutor *executor);
  void Deregister(uint32_t model_id);

 private:
  CpuSchedEventDispatcher() = default;
  Status OnInputsReady(rtEschedEventSummary_t &in_event);
  Status OnModelExecuted(uint32_t model_id, Status status) const;
  void ProcessEvents();

  int32_t device_id_ = -1;
  int32_t aicpu_sd_pid_ = -1;
  uint32_t event_group_id_ = 10U;
  bool host_exec_flag_ = false;

  std::thread event_handle_thread_;
  std::mutex mu_;
  std::atomic_bool running_{};
  // key: model_id
  std::map<uint32_t, DynamicModelExecutor *> models_;
  void *aicpu_handle_ = nullptr;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_EVENT_DISPATCHER_H_
