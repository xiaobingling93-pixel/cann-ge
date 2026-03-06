/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_NOTIFY_ALLOCATOR_H
#define AIR_CXX_RUNTIME_NOTIFY_ALLOCATOR_H

#include <cstdlib>
#include "runtime/event.h"
#include "common/ge_visibility.h"
#include "framework/common/ge_inner_error_codes.h"
#include "exe_graph/runtime/continuous_vector.h"

namespace gert {
class VISIBILITY_EXPORT NotifyAllocator {
 public:
  static constexpr size_t kMaxNotifyNum = 1024U;
  explicit NotifyAllocator() : notifies_holder_(ContinuousVector::Create<rtNotify_t>(kMaxNotifyNum)) {}
  NotifyAllocator(const NotifyAllocator &) = delete;
  NotifyAllocator &operator=(const NotifyAllocator &) = delete;
  ~NotifyAllocator() noexcept;

  TypedContinuousVector<rtNotify_t> *AcquireNotifies(const int32_t device_id, const size_t notify_num) const;

 private:
  TypedContinuousVector<rtNotify_t> *Notifies() const;

 private:
  std::unique_ptr<uint8_t[]> notifies_holder_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_NOTIFY_ALLOCATOR_H
