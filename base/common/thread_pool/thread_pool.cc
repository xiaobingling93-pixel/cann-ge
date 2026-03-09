/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/thread_pool/thread_pool.h"

#include <atomic>
#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"

namespace ge {
namespace {
const char* const kMultiThreadCompile = "MULTI_THREAD_COMPILE";
const char* const kDisEnableFlag = "0";
bool IsSingleThreadCompile() {
  std::string compile_thread;
  return ((ge::GetContext().GetOption(kMultiThreadCompile, compile_thread) == GRAPH_SUCCESS)
      && (compile_thread.compare(kDisEnableFlag) == 0));
}
}

ThreadPool::ThreadPool(std::string thread_name_prefix, const uint32_t size, const bool with_context)
    : thread_name_prefix_(std::move(thread_name_prefix)), is_stoped_(false) {
  idle_thrd_num_ = ((size < 1U) || IsSingleThreadCompile()) ? 1U : size;

  thread_local_context_ = nullptr;
  local_omg_context_ = nullptr;
  error_message_context_ = nullptr;
  if (with_context) {
    thread_local_context_ = MakeUnique<GEThreadLocalContext>(GetThreadLocalContext());
    local_omg_context_ = MakeUnique<OmgContext>(GetLocalOmgContext());
    error_message_context_ = MakeUnique<error_message::ErrorManagerContext>(error_message::GetErrMgrContext());
  }

  for (uint32_t i = 0U; i < idle_thrd_num_; ++i) {
    (void)pool_.emplace_back(&ThreadFunc, this, i);
  }
}

ThreadPool::~ThreadPool() {
  Destroy();
}

void ThreadPool::Destroy() {
  if (is_stoped_.load() == true) {
    return;
  }
  is_stoped_.store(true);
  {
    const std::unique_lock<std::mutex> lock{m_lock_};
    cond_var_.notify_all();
  }

  for (std::thread &thd : pool_) {
    if (thd.joinable()) {
      try {
        thd.join();
      } catch (const std::system_error &) {
        GELOGW("system_error");
      } catch (...) {
        GELOGW("exception");
      }
    }
  }
}

void ThreadPool::ThreadFunc(ThreadPool *const thread_pool, uint32_t thread_idx) {
  if (thread_pool == nullptr) {
    return;
  }
  if (!thread_pool->thread_name_prefix_.empty()) {
    auto thread_name = thread_pool->thread_name_prefix_ + std::to_string(thread_idx);
    const int32_t set_ret = pthread_setname_np(pthread_self(), thread_name.c_str());
    GELOGD("set thread name to [%s], ret=%d", thread_name.c_str(), set_ret);
  }
  if (thread_pool->thread_local_context_ != nullptr) {
    GetThreadLocalContext() = *(thread_pool->thread_local_context_);
  }
  std::unique_ptr<OmgContext> local_omg_context;
  if (thread_pool->local_omg_context_ != nullptr) {
    // must copy OmgContext for every thread as local omg context is threadlocal ptr.
    local_omg_context = MakeUnique<OmgContext>(*(thread_pool->local_omg_context_));
    if (local_omg_context != nullptr) {
      SetLocalOmgContext(*local_omg_context);
    }
  }
  if (thread_pool->error_message_context_ != nullptr) {
    error_message::SetErrMgrContext(*(thread_pool->error_message_context_));
  }
  while (!thread_pool->is_stoped_) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock{thread_pool->m_lock_};
      thread_pool->cond_var_.wait(
          lock, [thread_pool]() -> bool { return thread_pool->is_stoped_.load() || (!thread_pool->tasks_.empty()); });
      if (thread_pool->is_stoped_ && thread_pool->tasks_.empty()) {
        return;
      }
      task = std::move(thread_pool->tasks_.front());
      thread_pool->tasks_.pop();
    }
    --thread_pool->idle_thrd_num_;
    task();
    ++thread_pool->idle_thrd_num_;
  }
}
}  // namespace ge
