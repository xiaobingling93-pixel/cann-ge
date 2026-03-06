/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_RT_CALLBACK_MANAGER_H_
#define GE_HYBRID_EXECUTOR_RT_CALLBACK_MANAGER_H_

#include <condition_variable>
#include <future>

#include "hybrid/executor/callback_manager.h"
#include "common/blocking_queue.h"
#include "ge/ge_api_error_codes.h"
#include "runtime/rt.h"
namespace ge {
namespace hybrid {
class RtCallbackManager : public CallbackManager {
 public:
  RtCallbackManager() = default;
  ~RtCallbackManager() override = default;

  Status Init() override;

  Status Destroy() override;

  Status RegisterCallbackFunc(const rtStream_t stream, const std::function<void()> &callback) override;

 private:
  Status RegisterCallback(const rtStream_t stream, const rtCallback_t callback, void *const user_data);
  Status CallbackProcess(const rtContext_t context);
  static void RtCallbackFunc(void *const data);

  BlockingQueue<std::pair<rtEvent_t, std::pair<rtCallback_t, void *>>> callback_queue_;
  std::future<Status> ret_future_;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_RT_CALLBACK_MANAGER_H_
