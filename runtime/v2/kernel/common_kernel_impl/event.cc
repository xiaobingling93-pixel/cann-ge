/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "event.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "common/checker.h"
#include "utils/extern_math_util.h"

#include "kernel/gert_event.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "core/debug/kernel_tracing.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"

namespace gert {
namespace kernel {
ge::graphStatus CreateGertEvents(KernelContext *context) {
  auto event_ids_to_info = context->GetInputPointer<TypedContinuousVector<EventInfo>>(0U);
  GE_ASSERT_NOTNULL(event_ids_to_info);
  auto events_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(events_chain);
  auto events = new (std::nothrow) std::vector<GertEvent>();
  GE_ASSERT_NOTNULL(events);
  events->resize(event_ids_to_info->GetSize());
  events_chain->SetWithDefaultDeleter(events);

  GE_ASSERT_TRUE(ge::IntegerChecker<int64_t>::Compat(event_ids_to_info->GetSize()));
  for (size_t i = 0UL; i < event_ids_to_info->GetSize(); ++i) {
    events->at(i).logic_id = static_cast<int64_t>(i);
    events->at(i).compile_time_event_info.logic_src_stream = event_ids_to_info->GetData()[i].logic_src_stream;
    events->at(i).compile_time_event_info.logic_dst_stream = event_ids_to_info->GetData()[i].logic_dst_stream;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateGertEvents).RunFunc(CreateGertEvents);

/**
 * 批量发送events
 * lowering时保证两点，因此kernel中不做检查：
 * 1. 所有event中的logic src stream均相同，且与输入的src stream对应
 * 2. 所有要发出的event,dst stream id均不相同（不存在重复向一个dst stream发送的情况）
 * @param context
 * @return
 */
using DoFuncType = ge::graphStatus(KernelContext *context, GertEvent &event, rtEvent_t rt_event, rtStream_t stream,
                                   memory::MultiStreamL2Allocator *allocator);
template <DoFuncType DoFunc>
ge::graphStatus DoEvents(KernelContext *context) {
  auto event_ids =
      context->GetInputPointer<TypedContinuousVector<int64_t>>(static_cast<size_t>(SendEventsInput::kLogicEventIds));
  auto events = context->MutableInputPointer<std::vector<GertEvent>>(static_cast<size_t>(SendEventsInput::kAllEvents));
  auto rt_events =
      context->GetInputPointer<TypedContinuousVector<rtEvent_t>>(static_cast<size_t>(SendEventsInput::kAllRtEvents));
  auto allocator =
      context->MutableInputPointer<memory::MultiStreamL2Allocator>(static_cast<size_t>(SendEventsInput::kAllocator));
  if (SECUREC_UNLIKELY((events == nullptr) || (event_ids == nullptr) || (rt_events == nullptr) ||
                       (allocator == nullptr))) {
    return ge::GRAPH_FAILED;
  }
  if (SECUREC_UNLIKELY(events->size() > rt_events->GetSize())) {
    GELOGE(ge::PARAM_INVALID, "Failed to send events, rt events(%zu) num not enough, at least %zu",
           rt_events->GetSize(), events->size());
    return ge::GRAPH_FAILED;
  }
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(SendEventsInput::kStream));

  for (size_t i = 0UL; i < event_ids->GetSize(); ++i) {
    auto event_id = event_ids->GetData()[i];
    if (SECUREC_UNLIKELY(static_cast<size_t>(event_id) >= events->size())) {
      GELOGE(ge::PARAM_INVALID, "Failed to send events, invalid logic event id %lld", event_id);
      return ge::GRAPH_FAILED;
    }

    if (SECUREC_UNLIKELY(DoFunc(context, events->at(event_id), rt_events->GetData()[event_id], stream, allocator) !=
                         ge::GRAPH_SUCCESS)) {
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CallRtsSendEvent(KernelContext *context, GertEvent &event, rtEvent_t rt_event, rtStream_t stream,
                                 memory::MultiStreamL2Allocator *) {
  GE_ASSERT_RT_OK(rtEventRecord(rt_event, stream));
  KERNEL_TRACE("Sent event %" PRId64 " RT event %p from stream %" PRId64, event.logic_id, rt_event,
               event.compile_time_event_info.logic_src_stream);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(LastSendEvents).RunFunc(DoEvents<CallRtsSendEvent>).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus SendEvent(KernelContext *context, GertEvent &event, rtEvent_t rt_event, rtStream_t stream,
                          memory::MultiStreamL2Allocator *allocator) {
  auto blocks = allocator->GetClearLocalRecycleBlocks(event.compile_time_event_info.logic_dst_stream);
  for (auto iter = blocks.Begin(); iter != blocks.End(); blocks.Next(iter)) {
    KERNEL_TRACE("[MEM]Send memory recycling event from stream %" PRId64 ", event id %" PRId64 ", address %p",
                 event.compile_time_event_info.logic_src_stream, event.logic_id, iter->version_block.block->GetAddr());
    event.space.block_free_by_src_stream.emplace_back(iter->version_block);
  }

  // todo 这里的list仍然存在从操作系统申请和释放内存，考虑专门做个Allocator的池子，给所有l2 allocator的borrow
  // allocator使用
  //      另外borrow allocator中还存在size到block的映射map，效率也低，同样需要解决
  auto migration_blocks = allocator->GetAndClearBorrowBlocks(event.compile_time_event_info.logic_dst_stream);
  if (KERNEL_TRACE_ENABLE) {
    for (const auto block : migration_blocks) {
      KERNEL_TRACE("[MEM]Send memory migration event from stream %" PRId64 ", event id %" PRId64 ", address %p",
                   event.compile_time_event_info.logic_src_stream, event.logic_id, block->GetAddr());
    }
  }
  event.space.block_need_return_birth.insert(event.space.block_need_return_birth.end(), migration_blocks.begin(),
                                             migration_blocks.end());

  return CallRtsSendEvent(context, event, rt_event, stream, allocator);
}
REGISTER_KERNEL(SendEvents).RunFunc(DoEvents<SendEvent>).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus CallRtsWaitEvent(KernelContext *context, GertEvent &event, rtEvent_t rt_event, rtStream_t stream,
                                 memory::MultiStreamL2Allocator *) {
  GE_ASSERT_RT_OK(rtStreamWaitEvent(stream, rt_event));
  KERNEL_TRACE("Waited event %" PRId64 " RT event %p at stream %" PRId64, event.logic_id, rt_event,
               event.compile_time_event_info.logic_dst_stream);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus WaitEvent(KernelContext *context, GertEvent &event, rtEvent_t rt_event, rtStream_t stream,
                          memory::MultiStreamL2Allocator *allocator) {
  GE_ASSERT_GRAPH_SUCCESS(CallRtsWaitEvent(context, event, rt_event, stream, nullptr));

  for (auto &version_block : event.space.block_free_by_src_stream) {
    if (SECUREC_UNLIKELY(version_block.version != version_block.block->GetVersion())) {
      /*
       * 对于同一个block，在s0上已经完成了本地回收，考虑如下时序：
       * SendEventToS1
       * SendEventToS2
       * SendEventToS3
       * WaitEventOnS1: 完成了birth/borrow回收，block版本号增加
       * WaitEventOnS2: block version与发送时的version不匹配，说明该block已经进入下一个阶段，该消息为过时消息，跳过
       * WaitEventOnS3: 同上，跳过
       */
      GELOGD("Ignore block %p in waited event space from stream %" PRId64 "to %" PRId64, version_block.block,
             event.compile_time_event_info.logic_src_stream, event.compile_time_event_info.logic_dst_stream);
      continue;
    }
    KERNEL_TRACE("[MEM]Wait memory recycling event at stream %" PRId64 ", event id %" PRId64 ", address %p",
                 event.compile_time_event_info.logic_dst_stream, event.logic_id, version_block.block->GetAddr());
    GE_ASSERT_SUCCESS(version_block.block->SyncLocalRecycleStatus(event.compile_time_event_info.logic_src_stream,
                                                                  event.compile_time_event_info.logic_dst_stream));
  }
  event.space.block_free_by_src_stream.clear();

  for (auto &block : event.space.block_need_return_birth) {
    KERNEL_TRACE("[MEM]Wait memory migration event at stream %" PRId64 ", event id %" PRId64 ", address %p",
                 event.compile_time_event_info.logic_dst_stream, event.logic_id, block->GetAddr());
    GE_ASSERT_SUCCESS(allocator->BirthRecycle(block),
                      "Failed to birth recycle memory %p at stream %" PRId64 ", block %p, birth stream %" PRId64,
                      block->GetAddr(), event.compile_time_event_info.logic_dst_stream, block,
                      block->GetBirthStreamId());
  }
  event.space.block_need_return_birth.clear();

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(WaitEvents).RunFunc(DoEvents<WaitEvent>).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus LastWaitEvents(KernelContext *context) {
  // 在 DoEvents 中获取最后一个参数 allocators 时，会取错类型，但是只会判空，并不会真正使用，所以没关系
  GE_ASSERT_GRAPH_SUCCESS(DoEvents<CallRtsWaitEvent>(context));

  /*
   * 理想情况下，本kernel会把所有二级内存池中的内存做birth回收，并还给一级内存池，如果发现二级内存池中仍然无法
   */
  auto allocators = context->MutableInputPointer<TypedContinuousVector<memory::MultiStreamL2Allocator *>>(
      static_cast<size_t>(SendEventsInput::kAllocator));

  // 我们认为在LastWaitEvents后，辅流就不再有内存的申请动作了，因此所有birth在辅流上的内存都可以立刻同步local、borrow
  // recycle状态，并作回收 而birth在主流的内存由于收到了所有辅流的event，也可以同步所有的local、borrow recycle状态
  auto stream_num = static_cast<int64_t>(allocators->GetSize());
  GE_ASSERT_TRUE(stream_num > 0);
  for (int64_t src_stream_id = 0; src_stream_id < stream_num; ++src_stream_id) {
    auto src_allocator = allocators->MutableData()[src_stream_id];

    // lr_blocks means local recycle blocks
    auto lr_blocks = src_allocator->GetClearLocalRecycleBlocks();
    for (auto iter = lr_blocks.Begin(); iter != lr_blocks.End(); lr_blocks.Next(iter)) {
      auto block = iter->version_block.block;
      GE_ASSERT_SUCCESS(block->SyncAllLocalRecycleStatus(block->GetBirthStreamId()));
    }

    for (int64_t dst_stream_id = 0; dst_stream_id < stream_num; ++dst_stream_id) {
      for (auto block : src_allocator->GetAndClearBorrowBlocks(dst_stream_id)) {
        GE_ASSERT_SUCCESS(allocators->MutableData()[block->GetBirthStreamId()]->BirthRecycle(block));
      }
    }
  }

  for (int64_t i = 0; i < stream_num; ++i) {
    GE_ASSERT_SUCCESS(allocators->MutableData()[i]->RecycleFreeMem());
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(LastWaitEvents).RunFunc(LastWaitEvents).ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert
