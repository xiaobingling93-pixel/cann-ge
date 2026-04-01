/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_GERT_STREAM_H
#define AIR_CXX_GERT_STREAM_H
#include "graph/small_vector.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "kernel/memory/multi_stream_mem_block.h"
namespace gert {
// may delete later
struct GertStream {
  int64_t logic_id;
  aclrtStream stream;
};

struct EventInfo {
  int64_t logic_src_stream;
  int64_t logic_dst_stream;
};
static_assert(std::is_trivial<EventInfo>::value, "The class EventInfo must be a POD");

struct EventSpace {
  ge::SmallVector<memory::VersionBlock, 10> block_free_by_src_stream;
  ge::SmallVector<memory::MultiStreamMemBlock *, 10> block_need_return_birth;
};

struct GertEvent {
  int64_t logic_id;
  EventInfo compile_time_event_info;
  EventSpace space;
};
}  // namespace gert
#endif  // AIR_CXX_GERT_STREAM_H
