/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/rt_callback_manager.h"

#include "base/err_mgr.h"
#include "base/err_msg.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/profiling_definitions.h"
#include "graph/def_types.h"
#include "rt_error_codes.h"
#include "graph/ge_context.h"
#include "framework/runtime/subscriber/global_profiler.h"

namespace ge {
namespace {
constexpr int32_t kDefaultTimeOut = -1;
}
namespace hybrid {
Status RtCallbackManager::RegisterCallback(const rtStream_t stream,
                                           const rtCallback_t callback,
                                           void *const user_data) {
  GELOGD("To register callback");
  rtEvent_t event = nullptr;
  GE_PROFILING_START(kRtEventCreateRecord);
  GE_CHK_RT_RET(rtEventCreateWithFlag(&event, RT_EVENT_STREAM_MARK));
  const auto rt_ret = rtEventRecord(event, stream);
  GE_PROFILING_END(gert::profiling::kUnknownName, gert::profiling::kRtEventCreateRecord, kRtEventCreateRecord);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Invoke][rtEventRecord] failed, error code = %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Invoke rtEventRecord failed, error code = %d", rt_ret);
    (void)rtEventDestroy(event);
    return RT_FAILED;
  }

  auto cb = std::pair<rtCallback_t, void *>(callback, user_data);
  const auto entry = std::pair<rtEvent_t, std::pair<rtCallback_t, void *>>(event, std::move(cb));
  if (!callback_queue_.Push(entry)) {
    (void) rtEventDestroy(event);
    return INTERNAL_ERROR;
  }

  GELOGD("Registering callback successfully");
  return SUCCESS;
}

Status RtCallbackManager::Init() {
  rtContext_t ctx = nullptr;
  GE_CHK_RT_RET(rtCtxGetCurrent(&ctx));
  ret_future_ = std::async(std::launch::async, [this](const rtContext_t context,
      const struct error_message::ErrorManagerContext &error_context) ->Status {
    error_message::SetErrMgrContext(error_context);
    return CallbackProcess(context);
  }, ctx, error_message::GetErrMgrContext());
  if (!ret_future_.valid()) {
    GELOGE(INTERNAL_ERROR, "[Check][ShareState]Failed to init callback manager.");
    REPORT_INNER_ERR_MSG("E19999", "Failed to init callback manager.");
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status RtCallbackManager::CallbackProcess(const rtContext_t context) {
  GE_CHK_RT_RET(rtCtxSetCurrent(context));
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> entry;
  bool rt_timeout = false;
  while (true) {
    GELOGD("start to pop");
    if (!callback_queue_.Pop(entry)) {
      GELOGI("CallbackManager stopped");
      return INTERNAL_ERROR;
    }
    GELOGD("end to pop");
    const auto event = entry.first;
    if (event == nullptr) {
      GELOGD("receive eos entry");
      return rt_timeout ? FAILED : SUCCESS;
    }

    GE_PROFILING_START(kRtEventSync);
    std::string stream_synchronize_timeout;
    (void)ge::GetContext().GetOption(OPTION_EXEC_STREAM_SYNC_TIMEOUT, stream_synchronize_timeout);
    auto timeout = (!stream_synchronize_timeout.empty())
                       ? static_cast<int32_t>(std::strtol(stream_synchronize_timeout.c_str(), nullptr, 10))
                       : kDefaultTimeOut;
    const auto rt_err = rtEventSynchronizeWithTimeout(event, timeout);
    if (rt_err == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
      GELOGE(rt_err, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_err);
      REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, ret:%d.", rt_err);
      rt_timeout = true;
    } else if (rt_err != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Invoke][rtEventSynchronize] failed. ret = %d", rt_err);
      REPORT_INNER_ERR_MSG("E19999", "Invoke rtEventSynchronize failed, ret = %d.", rt_err);
      GE_CHK_RT(rtEventDestroy(event));
      return RT_FAILED;
    } else {
      // do nothing
    }
    GE_PROFILING_END(gert::profiling::kUnknownName, gert::profiling::kRtEventSync, kRtEventSync);

    GE_PROFILING_START(kRtEventDestroy);
    GE_CHK_RT(rtEventDestroy(event));
    GE_PROFILING_END(gert::profiling::kUnknownName, gert::profiling::kRtEventDestroy, kRtEventDestroy);

    const auto cb_func = entry.second.first;
    const auto cb_args = entry.second.second;
    cb_func(cb_args);
  }
}

Status RtCallbackManager::Destroy() {
  GELOGI("To destroy callback manager.");
  if (!ret_future_.valid()) {
    GELOGI("RtCallbackManager not initialized.");
    return SUCCESS;
  }

  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  (void) callback_queue_.Push(eof_entry);

  const auto ret = ret_future_.get();
  GELOGI("Callback manager ended. ret = %u", ret);
  return ret;
}

void RtCallbackManager::RtCallbackFunc(void *const data) {
  GELOGD("To invoke callback function");
  const auto callback_func = PtrToPtr<void, std::function<void()>>(data);
  (*callback_func)();
  delete callback_func;
}

Status RtCallbackManager::RegisterCallbackFunc(const rtStream_t stream, const std::function<void()> &callback) {
  auto func = MakeUnique<std::function<void()>>(std::function<void()>(callback));
  GE_CHECK_NOTNULL(func);
  GELOGD("Callback registered");
  return RegisterCallback(stream, &RtCallbackFunc, func.release());
}
}  // namespace hybrid
}  // namespace ge
