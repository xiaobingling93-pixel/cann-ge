/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reusable_stream_allocator.h"

#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "common/math/math_util.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr int32_t kInvalidStream = -1;
}  // namespace

ReusableStreamAllocator *ReusableStreamAllocator::Create() {
  ReusableStreamAllocator *reusable_stream_allocator = new (std::nothrow) ReusableStreamAllocator();
  GE_ASSERT_NOTNULL(reusable_stream_allocator);
  GE_DISMISSABLE_GUARD(reusable_stream_allocator_guard, [&reusable_stream_allocator]() {
    delete reusable_stream_allocator;
    reusable_stream_allocator = nullptr;
  });
  GELOGD("Create reusable_stream_allocator which does not need to reuse stream.");
  GE_DISMISS_GUARD(reusable_stream_allocator_guard);
  return reusable_stream_allocator;
}

RtStreamStatusPtr ReusableStreamAllocator::CreateNewStream(rtStream_t &stream, const uint32_t rt_model_id,
                                                           const int32_t priority, const uint32_t stream_flag,
                                                           const uint32_t task_num) const {
  GE_ASSERT_RT_OK(rtStreamCreateWithFlags(&stream, priority, stream_flag));
  int32_t rt_stream_id = kInvalidStream;
  GE_ASSERT_RT_OK(rtGetStreamId(stream, &rt_stream_id));
  GELOGI("Create new stream: %p, rt stream id: %d, rt model id: %u, priority: %d, stream flag: %u, task num: %u.",
         stream, rt_stream_id, rt_model_id, priority, stream_flag, task_num);
  const auto stream_status = RtStreamStatus::Create(stream, rt_stream_id, rt_model_id, task_num);
  return stream_status;
}

Status ReusableStreamAllocator::GetOrCreateRtStream(rtStream_t &stream, const uint32_t rt_model_id,
                                                    const int32_t priority, const uint32_t stream_flag,
                                                    const uint32_t task_num) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto stream_status = CreateNewStream(stream, rt_model_id, priority, stream_flag, task_num);
  GE_ASSERT_NOTNULL(stream_status);
  stream_ref_[stream] = stream_status;
  rt_stream_list_[std::make_pair(priority, stream_flag)].emplace(std::move(stream_status));
  return SUCCESS;
}
Status ReusableStreamAllocator::DestroyStream(rtStream_t &stream, const bool is_force_destroy) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GE_ASSERT_NOTNULL(stream);
  const auto iter = stream_ref_.find(stream);
  GE_ASSERT_TRUE(iter != stream_ref_.end());
  const auto &stream_status = iter->second;
  GE_ASSERT_NOTNULL(stream_status);
  if (!stream_status->is_valid) {
    GELOGD("Stream: %p, id: %d, has already been destroyed.", stream, stream_status->rt_stream_id);
    return SUCCESS;
  }

  if (is_force_destroy) {
    GE_ASSERT_RT_OK(rtStreamDestroyForce(stream));
  } else {
    GE_ASSERT_RT_OK(rtStreamDestroy(stream));
  }
  stream_status->is_valid = false;
  GELOGD("Succ to destroy stream: %p, id: %d, is_force_destroy: %d.", stream, stream_status->rt_stream_id,
         static_cast<int32_t>(is_force_destroy));

  return SUCCESS;
}
}  // namespace ge
