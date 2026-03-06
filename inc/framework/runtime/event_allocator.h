/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_

#include <cstdlib>
#include "runtime/event.h"
#include "common/ge_visibility.h"
#include "exe_graph/runtime/continuous_vector.h"

namespace gert {
class VISIBILITY_EXPORT EventAllocator {
 public:
  static constexpr size_t kMaxEventNum = 4096U;
  explicit EventAllocator(uint32_t flag = RT_EVENT_DDSYNC_NS)
      : events_holder_(ContinuousVector::Create<rtEvent_t>(kMaxEventNum)), default_flag_(flag) {}
  EventAllocator(const EventAllocator &) = delete;
  EventAllocator &operator=(const EventAllocator &) = delete;
  ~EventAllocator();

  TypedContinuousVector<rtEvent_t> *AcquireEvents(size_t event_num) const;

 private:
  TypedContinuousVector<rtEvent_t> *Events() const;

 private:
  std::unique_ptr<uint8_t[]> events_holder_;
  uint32_t default_flag_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_EVENT_ALLOCATOR_H_
