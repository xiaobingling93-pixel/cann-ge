/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_REUSABLE_STREAM_ALLOCATOR_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_REUSABLE_STREAM_ALLOCATOR_H_

#include <map>
#include <set>

#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "acl/acl_rt.h"

namespace ge {
struct RtStreamStatus;
using RtStreamStatusPtr = std::shared_ptr<RtStreamStatus>;

struct RtStreamStatus {
  aclrtStream stream = nullptr;
  int32_t rt_stream_id;
  std::set<uint32_t> rt_model_id;  // can not reuse streams of its own model
  uint32_t task_num = 0U;
  bool is_valid = true;

  static RtStreamStatusPtr Create(aclrtStream &stream, const int32_t rt_stream_id, const uint32_t rt_model_id,
                                  const uint32_t task_num = 0U) {
    RtStreamStatusPtr stream_status = std::make_shared<RtStreamStatus>();
    GE_ASSERT_NOTNULL(stream_status);

    stream_status->stream = stream;
    stream_status->rt_stream_id = rt_stream_id;
    stream_status->rt_model_id.emplace(rt_model_id);
    stream_status->task_num = task_num;

    return stream_status;
  }
};

struct StreamCompareKey {
  bool operator()(const RtStreamStatusPtr &status1, const RtStreamStatusPtr &status2) const {
    if (status1->task_num == status2->task_num) {
      return status1->stream < status2->stream;
    }
    return (status1->task_num < status2->task_num);
  }
};

class ReusableStreamAllocator {
 public:
  static ReusableStreamAllocator *Create();
  Status GetOrCreateRtStream(aclrtStream &stream, const uint32_t rt_model_id, const int32_t priority,
                             const uint32_t stream_flag, const uint32_t task_num = 0U);
  Status DestroyStream(aclrtStream &stream, const bool is_force_destroy = false);

 private:
  ReusableStreamAllocator() = default;
  ReusableStreamAllocator(const ReusableStreamAllocator &) = delete;
  ReusableStreamAllocator(ReusableStreamAllocator &&) = delete;
  ReusableStreamAllocator operator=(const ReusableStreamAllocator &) = delete;
  ReusableStreamAllocator operator=(ReusableStreamAllocator &&) = delete;

  RtStreamStatusPtr CreateNewStream(aclrtStream &stream, const uint32_t rt_model_id, const int32_t priority,
                                    const uint32_t stream_flag, const uint32_t task_num = 0U) const;
  // <<priority, stream_flag>, std::set<RtStreamStatus>>
  std::map<std::pair<int32_t, uint32_t>, std::set<RtStreamStatusPtr, StreamCompareKey>> rt_stream_list_{};
  // to get stream_status easier
  std::map<aclrtStream, RtStreamStatusPtr> stream_ref_;
  std::mutex mutex_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_REUSABLE_STREAM_ALLOCATOR_H_
