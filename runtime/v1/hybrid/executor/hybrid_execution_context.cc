/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/hybrid_execution_context.h"

#include <atomic>

#include "common/dump/dump_manager.h"
#include "graph/ge_context.h"
#include "rt_error_codes.h"

namespace ge {
namespace hybrid {
namespace {
const int32_t kEndOfSequence = 0x0704000a;
const int32_t kEndOfSequenceNew = 507005;
const int32_t kModelAbortNormal = 0x0704000e;
const int32_t kModelAbortNormalNew = 507024;
const int32_t kIntBase = 10;
const int32_t kDefaultTimeOut = -1;
}  // namespace

int64_t GraphExecutionContext::profiling_level = 0;

GraphExecutionContext::GraphExecutionContext() {
  static std::atomic_ulong context_id_gen{};
  context_id = context_id_gen++;
}

GraphExecutionContext::~GraphExecutionContext() {
  if (own_callback_manager && (callback_manager != nullptr)) {
    delete callback_manager;
    callback_manager = nullptr;
  }
}

Status GraphExecutionContext::InitProfiler() const {
  const char_t *env_profiling_level = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HYBRID_PROFILING_LEVEL, env_profiling_level);
  if (env_profiling_level != nullptr) {
    GraphExecutionContext::profiling_level = std::strtol(env_profiling_level, nullptr, kIntBase);
    GELOGD("Got profiling level = %ld", GraphExecutionContext::profiling_level);
    if (GraphExecutionContext::profiling_level > 0) {
      profiler = MakeUnique<HybridProfiler>();
      GE_CHECK_NOTNULL(profiler);
    }
  }
  return SUCCESS;
}

void GraphExecutionContext::SetErrorCode(const Status error_code) {
  const std::lock_guard<std::mutex> lk(mu);
  this->status = error_code;
}

Status GraphExecutionContext::GetStatus() const {
  const std::lock_guard<std::mutex> lk(mu);
  return this->status;
}

Status GraphExecutionContext::Synchronize(const rtStream_t rt_stream) {
  std::string stream_synchronize_timeout;
  (void)ge::GetContext().GetOption(OPTION_EXEC_STREAM_SYNC_TIMEOUT, stream_synchronize_timeout);
  auto timeout = (!stream_synchronize_timeout.empty())
                     ? static_cast<int32_t>(std::strtol(stream_synchronize_timeout.c_str(), nullptr, 10))
                     : kDefaultTimeOut;
  const auto rt_ret = rtStreamSynchronizeWithTimeout(rt_stream, timeout);
  if (rt_ret == RT_ERROR_NONE) {
    return SUCCESS;
  }

  if ((rt_ret == kEndOfSequence) || (rt_ret == kEndOfSequenceNew)) {
    GELOGI("Got end of sequence");
    is_eos_ = true;
    return END_OF_SEQUENCE;
  }

  if ((rt_ret == kModelAbortNormal) || (rt_ret == kModelAbortNormalNew)) {
    GELOGI("The model with multiple datasets aborts normally");
    return SUCCESS;
  }

  if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
    GELOGE(rt_ret, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret:%d.", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, ret:%d.", rt_ret);
    return FAILED;
  }

  GELOGE(RT_FAILED, "[Invoke][rtStreamSynchronizeWithTimeout] failed, ret = %d", rt_ret);
  REPORT_INNER_ERR_MSG("E19999", "invoke rtStreamSynchronizeWithTimeout failed, ret = %d", rt_ret);
  return RT_FAILED;
}

bool GraphExecutionContext::IsDumpEnabled() const {
  return dump_properties.IsDumpOpen() || DumpManager::GetInstance().IsDumpExceptionOpen() ||
         dump_properties.IsOpDebugOpen();
}
}  // namespace hybrid
}  // namespace ge
