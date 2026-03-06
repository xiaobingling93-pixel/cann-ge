/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/notify_allocator.h"

#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge_common/debug/log.h"

namespace gert {
NotifyAllocator::~NotifyAllocator() noexcept {
  auto notifies = Notifies();
  for (size_t i = 0U; i < notifies->GetSize(); ++i) {
    (void)rtNotifyDestroy(notifies->MutableData()[i]);
  }
}

TypedContinuousVector<rtNotify_t> *NotifyAllocator::AcquireNotifies(const int32_t device_id,
                                                                    const size_t notify_num) const {
  GE_ASSERT_TRUE(notify_num < kMaxNotifyNum);
  auto notifies = Notifies();
  for (size_t i = notifies->GetSize(); i < notify_num; ++i) {
    rtNotify_t notify = nullptr;
    GE_ASSERT_RT_OK(rtNotifyCreate(device_id, &notify));
    notifies->MutableData()[i] = notify;
    GE_ASSERT_SUCCESS(notifies->SetSize(i + 1U));
  }
  return notifies;
}

TypedContinuousVector<rtNotify_t> *NotifyAllocator::Notifies() const {
  return reinterpret_cast<TypedContinuousVector<rtNotify_t> *>(notifies_holder_.get());
}
}  // namespace gert
