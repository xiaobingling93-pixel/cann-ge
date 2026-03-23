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
    (void)aclrtDestroyNotify(notifies->MutableData()[i]);
  }
}

TypedContinuousVector<aclrtNotify> *NotifyAllocator::AcquireNotifies(const int32_t device_id,
                                                                    const size_t notify_num) const {
  (void)device_id;
  GE_ASSERT_TRUE(notify_num < kMaxNotifyNum);
  auto notifies = Notifies();
  for (size_t i = notifies->GetSize(); i < notify_num; ++i) {
    aclrtNotify notify = nullptr;
    GE_ASSERT_RT_OK(aclrtCreateNotify(&notify, 0U));
    notifies->MutableData()[i] = notify;
    GE_ASSERT_SUCCESS(notifies->SetSize(i + 1U));
  }
  return notifies;
}

TypedContinuousVector<aclrtNotify> *NotifyAllocator::Notifies() const {
  return reinterpret_cast<TypedContinuousVector<aclrtNotify> *>(notifies_holder_.get());
}
}  // namespace gert
