/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_THREAD_POOL_H_
#define GE_COMMON_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "graph/ge_local_context.h"
#include "common/context/local_context.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api_error_codes.h"
#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"

namespace ge {
using ThreadTask = std::function<void()>;

class ThreadPool {
 public:
  explicit ThreadPool(std::string thread_name_prefix, const uint32_t size = 4U, const bool with_context = true);
  ~ThreadPool();
  void Destroy();

  template <class Func, class... Args>
  auto commit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
    GELOGD("commit run task enter.");
    using retType = decltype(func(args...));
    std::future<retType> fail_future;
    if (is_stoped_.load()) {
      GELOGE(ge::FAILED, "thread pool has been stopped.");
      return fail_future;
    }

    const auto bindFunc = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    const auto task = ge::MakeShared<std::packaged_task<retType()>>(bindFunc);
    if (task == nullptr) {
      GELOGE(ge::FAILED, "Make shared failed.");
      return fail_future;
    }
    std::future<retType> future = task->get_future();
    {
      const std::lock_guard<std::mutex> lock{m_lock_};
      (void)tasks_.emplace([task]() { (*task)(); });
    }
    cond_var_.notify_one();
    GELOGD("commit run task end");
    return future;
  }

  static void ThreadFunc(ThreadPool *const thread_pool, uint32_t thread_idx);

 private:
  std::string thread_name_prefix_;
  std::vector<std::thread> pool_;
  std::queue<ThreadTask> tasks_;
  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_;
  std::atomic<uint32_t> idle_thrd_num_;
  std::unique_ptr<GEThreadLocalContext> thread_local_context_;
  std::unique_ptr<OmgContext> local_omg_context_;
  std::unique_ptr<error_message::ErrorManagerContext> error_message_context_;
};
}  // namespace ge

#endif  // GE_COMMON_THREAD_POOL_H_
