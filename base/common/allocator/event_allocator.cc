/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/event_allocator.h"

#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
EventAllocator::~EventAllocator() {
  const auto events = Events();
  for (size_t i = 0U; i < events->GetSize(); ++i) {
    (void)rtEventDestroy(events->MutableData()[i]);
  }
}

TypedContinuousVector<rtEvent_t> *EventAllocator::AcquireEvents(const size_t event_num) const {
  auto events = Events();
  for (size_t i = events->GetSize(); i < event_num; ++i) {
    rtEvent_t event = nullptr;
    GE_ASSERT_RT_OK(rtEventCreateExWithFlag(&event, default_flag_));
    events->MutableData()[i] = event;
    GE_ASSERT_SUCCESS(events->SetSize(i + 1U));
  }
  return events;
}

TypedContinuousVector<rtEvent_t> *EventAllocator::Events() const {
  return reinterpret_cast<TypedContinuousVector<rtEvent_t> *>(events_holder_.get());
}
}  // namespace gert
