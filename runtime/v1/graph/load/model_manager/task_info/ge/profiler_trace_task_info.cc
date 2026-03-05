/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/ge/profiler_trace_task_info.h"
#include "acl/acl_rt.h"
#include "graph/load/model_manager/davinci_model.h"

namespace {
constexpr uint64_t kProfilingMaxLogid = 5U;  // step trace中tagId的最大值
constexpr uint64_t kProfilingArStartLogid = 10000U;
constexpr uint64_t kProfilingArMaxLogid = 29999U;
} // namespace

namespace ge {
Status ProfilerTraceTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                   const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                   const IowAddrs &iow_addrs) {
  GELOGI("ProfilerTraceTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  model_id_ = davinci_model->GetModelId();
  GELOGD("model id is %u", model_id_);

  const auto &log_time_stamp_def = task_def.log_timestamp();
  log_id_ = log_time_stamp_def.logid();
  notify_ = log_time_stamp_def.notify();
  flat_ = log_time_stamp_def.flat();
  davinci_model_ = davinci_model;

  GELOGI("ProfilerTraceTaskInfo Init Success, logic stream id: %u, stream: %p.", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status ProfilerTraceTaskInfo::Distribute() {
  GELOGI("ProfilerTraceTaskInfo Distribute Start. logid = %" PRIu64 ". notify = %d.",
    log_id_, static_cast<int32_t>(notify_));
  is_support_redistribute_ = true;
  if (((log_id_ > kProfilingMaxLogid) && (log_id_ < kProfilingArStartLogid)) ||
    (log_id_ > kProfilingArMaxLogid)) {
    GELOGD("ProfilerTraceTaskInfo logid:%" PRIu64 " is out of range.", log_id_);
    return SUCCESS;
  }
  if ((davinci_model_ != nullptr) && (!davinci_model_->CheckModelNoInputAndOutput())) {
    GELOGD("ProfilerTraceTaskInfo load model with queue no need distribute.");
    return SUCCESS;
  }

  const rtError_t rt_ret =
    rtProfilerTraceEx(1UL, static_cast<uint64_t>(model_id_), static_cast<uint16_t>(log_id_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(ge::RT_FAILED, "[Call][rtProfilerTraceEx]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtProfilerTraceEx failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGI("ProfilerTraceTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_PROFILER_TRACE, ProfilerTraceTaskInfo);
REGISTER_TASK_INFO(MODEL_TASK_PROFILER_TRACE_EX, ProfilerTraceTaskInfo);
}  // namespace ge

