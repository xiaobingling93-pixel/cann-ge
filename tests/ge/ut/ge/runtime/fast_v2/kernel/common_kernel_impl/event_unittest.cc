/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/kernel_registry.h"
#include <gtest/gtest.h>
#include "exe_graph/runtime/continuous_vector.h"
#include "faker/kernel_run_context_facker.h"
#include "faker/continuous_vector_builder.h"
#include "stub/gert_runtime_stub.h"
#include "kernel/gert_event.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "faker/multi_stream_allocator_faker.h"
#include "checker/memory_profiling_log_matcher.h"
#include "kernel/memory/multi_stream_mem_block_helper.h"
namespace gert {
namespace memory {
bool operator==(const VersionBlock &vb1, const VersionBlock &vb2) {
  return vb1.block == vb2.block && vb1.version == vb2.version;
}
bool operator!=(const VersionBlock &vb1, const VersionBlock &vb2) {
  return !(vb1 == vb2);
}
}  // namespace memory
namespace kernel {
template <typename T, size_t N>
std::set<T> ToSet(ge::SmallVector<T, N> &vec) {
  std::set<T> s;
  for (const auto &e : vec) {
    s.insert(e);
  }
  return s;
}
extern ge::graphStatus CreateGertEvents(KernelContext *context);
struct L2AllocatorHolder {
  struct BlockInfo {
    BlockInfo(memory::MultiStreamMemBlock *block) {
      this->block = block;
      addr = block->GetAddr();
    }
    void *GetAddr() {
      return addr;
    }
    BlockInfo *operator->() {
      return this;
    }
    memory::MultiStreamMemBlock *get() {
      return block;
    }
    void *addr;
    memory::MultiStreamMemBlock *block;
  };
  memory::MultiStreamAllocatorFaker::Holder allocators;
  std::vector<BlockInfo> stream_0_local_recycle_blocks;
  std::vector<BlockInfo> stream_0_borrow_recycle_blocks;

  void FakeStream0LocalRecycleBlocks(size_t num) {
    for (size_t i = 0U; i < num; ++i) {
      auto block = allocators.AllocBlock(0, 1024);
      block->NewAccessStream(0, 1);
      block->Free(0);
      stream_0_local_recycle_blocks.emplace_back(block.release());
    }
  }

  void FakeStream0BorrowRecycleBlocks(size_t num) {
    for (size_t i = 0U; i < num; ++i) {
      auto block = allocators.AllocBlock(1, 1024);
      block->NewAccessStream(1, 0);
      block->Free(1);
      block->SyncLocalRecycleStatus(1, 0);
      block->Free(0);
      stream_0_borrow_recycle_blocks.emplace_back(block.release());
    }
  }
};
class EventUT : public testing::Test {
 protected:
  L2AllocatorHolder FakeL2Allocator() {
    L2AllocatorHolder holder;
    holder.allocators = memory::MultiStreamAllocatorFaker().StreamNum(4).Build();
    holder.FakeStream0LocalRecycleBlocks(2);
    holder.FakeStream0BorrowRecycleBlocks(2);
    return holder;
  }
};
#define CHECK_RESULT(index, src, dst)                                           \
  do {                                                                          \
    ASSERT_EQ(events->at(index).logic_id, index);                               \
    ASSERT_EQ(events->at(index).compile_time_event_info.logic_src_stream, src); \
    ASSERT_EQ(events->at(index).compile_time_event_info.logic_dst_stream, dst); \
    ASSERT_TRUE(events->at(index).space.block_free_by_src_stream.empty());      \
    ASSERT_TRUE(events->at(index).space.block_need_return_birth.empty());       \
  } while (0)
TEST_F(EventUT, CreateGertEvent_Ok) {
  auto events_info = ContinuousVectorBuilder::Create<EventInfo>({//
                                                                 {0, 1},
                                                                 {0, 1},
                                                                 {1, 0},
                                                                 {2, 3}});
  auto context = KernelRunContextFaker().KernelIONum(1, 4).Inputs({events_info.get()}).Build();

  auto kc = context.GetContext<KernelContext>();
  ASSERT_EQ(CreateGertEvents(kc), ge::GRAPH_SUCCESS);
  auto events = kc->GetOutputPointer<std::vector<GertEvent>>(0);
  ASSERT_NE(events, nullptr);
  ASSERT_EQ(events->size(), 4);
  CHECK_RESULT(0, 0, 1);
  CHECK_RESULT(1, 0, 1);
  CHECK_RESULT(2, 1, 0);
  CHECK_RESULT(3, 2, 3);
}
TEST_F(EventUT, CreateGertEvent_Failed_nullinput) {
  auto context = KernelRunContextFaker().KernelIONum(1, 4).Inputs({nullptr}).Build();
  auto kc = context.GetContext<KernelContext>();
  ASSERT_NE(CreateGertEvents(kc), ge::GRAPH_SUCCESS);
}
TEST_F(EventUT, SendEvents_Success) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({0, 2});
  std::vector<GertEvent> events{
      {0, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {1, {0, 2}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {0, 3}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>({(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300});
  auto allocator_holder = FakeL2Allocator();
  auto context =
      KernelRunContextFaker()
          .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), allocator_holder.allocators.at(0)})
          .Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func, nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);
  std::map<rtEvent_t, std::vector<rtStream_t>> expect_send_event_records = {
      {(rtEvent_t)0x100, {(rtStream_t)0x1000}},
      {(rtEvent_t)0x300, {(rtStream_t)0x1000}},
  };
  // rts record correct
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetRtEventRecordRecords(), expect_send_event_records);

  // event space correct
  ge::SmallVector<memory::VersionBlock, 10> expect_local_recycle_blocks{
      {0, allocator_holder.stream_0_local_recycle_blocks.at(0).get()},
      {0, allocator_holder.stream_0_local_recycle_blocks.at(1).get()}};
  EXPECT_EQ(events[0].space.block_free_by_src_stream, expect_local_recycle_blocks);
  EXPECT_EQ(events[2].space.block_free_by_src_stream, expect_local_recycle_blocks);

  std::set<memory::MultiStreamMemBlock *> expect_borrow_recycle_blocks{
      allocator_holder.stream_0_borrow_recycle_blocks.at(0).get(),
      allocator_holder.stream_0_borrow_recycle_blocks.at(1).get()};
  EXPECT_EQ(events[0].space.block_need_return_birth.size(), expect_local_recycle_blocks.size());
  EXPECT_EQ(ToSet(events[0].space.block_need_return_birth), expect_borrow_recycle_blocks);
  EXPECT_TRUE(events[2].space.block_need_return_birth.empty());

  // allocator correct
  auto blocks1 = allocator_holder.allocators.at(0)->GetClearLocalRecycleBlocks(1);
  EXPECT_EQ(blocks1.Begin(), blocks1.End());
  auto blocks2 = allocator_holder.allocators.at(0)->GetClearLocalRecycleBlocks(2);
  EXPECT_NE(blocks2.Begin(), blocks2.End());

  EXPECT_TRUE(allocator_holder.allocators.at(0)->GetAndClearBorrowBlocks(1).empty());

  // log correct
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleSendEvent,
                  {{2, "0"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(0)->GetAddr())}}) >=
              0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleSendEvent,
                  {{2, "0"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(1)->GetAddr())}}) >=
              0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleSendEvent,
                  {{2, "0"}, {3, "2"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(0)->GetAddr())}}) >=
              0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleSendEvent,
                  {{2, "0"}, {3, "2"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(1)->GetAddr())}}) >=
              0);

  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kMigrationSendEvent,
                  {{2, "0"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_borrow_recycle_blocks.at(0)->GetAddr())}}) >=
              0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kMigrationSendEvent,
                  {{2, "0"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_borrow_recycle_blocks.at(1)->GetAddr())}}) >=
              0);

  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kSendEvent, {{2, "0"}, {3, "0x100"}, {7, "0"}}) >= 0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kSendEvent, {{2, "2"}, {3, "0x300"}, {7, "0"}}) >= 0);
}

TEST_F(EventUT, RecvEvents_Success) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({0, 2});
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>({(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300});
  auto allocator_holder = FakeL2Allocator();
  std::vector<GertEvent> events{
      {0,
       {0, 1},
       {ge::SmallVector<memory::VersionBlock, 10>{{0, allocator_holder.stream_0_local_recycle_blocks.at(0).get()},
                                                  {0, allocator_holder.stream_0_local_recycle_blocks.at(1).get()}},
        ge::SmallVector<memory::MultiStreamMemBlock *, 10>{
            allocator_holder.stream_0_borrow_recycle_blocks.at(0).get(),
            allocator_holder.stream_0_borrow_recycle_blocks.at(1).get()}}},
      {1, {1, 3}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {2, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto context =
      KernelRunContextFaker()
          .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), allocator_holder.allocators.at(1)})
          .Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("WaitEvents")->run_func, nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  auto borrow_block_0_addr = ToHex(allocator_holder.stream_0_borrow_recycle_blocks.at(0)->GetAddr());
  auto borrow_block_1_addr = ToHex(allocator_holder.stream_0_borrow_recycle_blocks.at(1)->GetAddr());
  ASSERT_TRUE(
      memory::MultiStreamMemBlockHelper::IsOccupied(allocator_holder.stream_0_local_recycle_blocks.at(0).get(), 1, 0));
  ASSERT_TRUE(
      memory::MultiStreamMemBlockHelper::IsOccupied(allocator_holder.stream_0_local_recycle_blocks.at(1).get(), 1, 0));
  ASSERT_EQ(KernelRegistry::GetInstance().FindKernelFuncs("WaitEvents")->run_func(context), ge::GRAPH_SUCCESS);

  // todo rtStreamWaitEvent 暂时无法被接管，需要对stub库做重构，暂时无法检查了

  // check allocator
  EXPECT_FALSE(
      memory::MultiStreamMemBlockHelper::IsOccupied(allocator_holder.stream_0_local_recycle_blocks.at(0).get(), 1, 0));
  EXPECT_FALSE(
      memory::MultiStreamMemBlockHelper::IsOccupied(allocator_holder.stream_0_local_recycle_blocks.at(1).get(), 1, 0));
  // todo 判断allocator的birth recycle已经被调用

  // check event space
  EXPECT_TRUE(events.at(0).space.block_free_by_src_stream.empty());
  EXPECT_TRUE(events.at(0).space.block_need_return_birth.empty());

  // check log
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleRecvEvent,
                  {{2, "1"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(0)->GetAddr())}}) >=
              0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleRecvEvent,
                  {{2, "1"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(1)->GetAddr())}}) >=
              0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kMigrationRecvEvent,
                                                          {{2, "1"}, {3, "0"}, {4, borrow_block_0_addr}}) >= 0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kMigrationRecvEvent,
                                                          {{2, "1"}, {3, "0"}, {4, borrow_block_1_addr}}) >= 0);

  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kWaitEvent, {{2, "0"}, {3, "0x100"}, {7, "1"}}) >= 0);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kWaitEvent, {{2, "2"}, {3, "0x300"}, {7, "1"}}) >= 0);
}
TEST_F(EventUT, RecvEvents_DropOutDatedBlocks) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({0, 2});
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>({(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300});
  auto allocator_holder = FakeL2Allocator();
  memory::MultiStreamMemBlockHelper::PlusVersion(allocator_holder.stream_0_local_recycle_blocks.at(0).get());
  std::vector<GertEvent> events{
      {0,
       {0, 1},
       {ge::SmallVector<memory::VersionBlock, 10>{{0, allocator_holder.stream_0_local_recycle_blocks.at(0).get()},
                                                  {0, allocator_holder.stream_0_local_recycle_blocks.at(1).get()}},
        ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {1, {1, 3}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {2, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto context =
      KernelRunContextFaker()
          .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), allocator_holder.allocators.at(1)})
          .Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("WaitEvents")->run_func, nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  ASSERT_EQ(KernelRegistry::GetInstance().FindKernelFuncs("WaitEvents")->run_func(context), ge::GRAPH_SUCCESS);

  // todo rtStreamWaitEvent 暂时无法被接管，需要对stub库做重构，暂时无法检查了

  EXPECT_TRUE(
      memory::MultiStreamMemBlockHelper::IsOccupied(allocator_holder.stream_0_local_recycle_blocks.at(0).get(), 1, 0));
  EXPECT_FALSE(
      memory::MultiStreamMemBlockHelper::IsOccupied(allocator_holder.stream_0_local_recycle_blocks.at(1).get(), 1, 0));

  EXPECT_TRUE(events.at(0).space.block_free_by_src_stream.empty());
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(
                  kRecycleRecvEvent,
                  {{2, "1"}, {3, "0"}, {4, ToHex(allocator_holder.stream_0_local_recycle_blocks.at(1)->GetAddr())}}) >=
              0);
}
TEST_F(EventUT, SendEvents_Failed_InvalidEventId) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({4});
  std::vector<GertEvent> events{
      {0, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {1, {1, 0}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {0, 2}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {3, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>(
      {(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300, (rtEvent_t)0x400, (rtEvent_t)0x500});
  auto context = KernelRunContextFaker()
                     .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), (void*)1})
                     .Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func, nullptr);

  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);
}
TEST_F(EventUT, SendEvents_Failed_RtEventsNotEnough) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({0, 2, 3});
  std::vector<GertEvent> events{
      {0, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {1, {1, 0}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {0, 2}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {3, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {4, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>(
      {(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300, (rtEvent_t)0x400});
  auto context = KernelRunContextFaker()
                     .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), (void*)1})
                     .Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func, nullptr);

  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);
}
TEST_F(EventUT, SendEvents_Failed_NullInput) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({0, 2, 3});
  std::vector<GertEvent> events{
      {0, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {1, {1, 0}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {0, 2}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {3, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {4, {0, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>(
      {(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300, (rtEvent_t)0x400, (rtEvent_t)0x500});
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func, nullptr);

  auto context =
      KernelRunContextFaker().Inputs({(rtStream_t)0x1000, nullptr, &events, rt_events.get(), (void*)1}).Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);

  context = KernelRunContextFaker()
                .Inputs({(rtStream_t)0x1000, event_ids.get(), nullptr, rt_events.get(), (void*)1})
                .Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);

  context = KernelRunContextFaker().Inputs({(rtStream_t)0x1000, event_ids.get(), &events, nullptr, (void*)1}).Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);

  context =
      KernelRunContextFaker().Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), nullptr}).Build();
  ASSERT_NE(KernelRegistry::GetInstance().FindKernelFuncs("SendEvents")->run_func(context), ge::GRAPH_SUCCESS);
}
//todo just for ut coverage
TEST_F(EventUT, LastWaitEvents) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({0, 2});
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>({(rtEvent_t)0x100, (rtEvent_t)0x200, (rtEvent_t)0x300});
  auto allocator_holder = FakeL2Allocator();
  memory::MultiStreamMemBlockHelper::PlusVersion(allocator_holder.stream_0_local_recycle_blocks.at(0).get());
  std::vector<GertEvent> events{
      {0,
       {0, 1},
       {ge::SmallVector<memory::VersionBlock, 10>{{0, allocator_holder.stream_0_local_recycle_blocks.at(0).get()},
                                                  {0, allocator_holder.stream_0_local_recycle_blocks.at(1).get()}},
        ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {1, {1, 3}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
      {2, {2, 1}, {ge::SmallVector<memory::VersionBlock, 10>{}, ge::SmallVector<memory::MultiStreamMemBlock *, 10>{}}},
  };
  auto allocator_size = allocator_holder.allocators.stream_ids_to_allocator.size();
  auto l2_allocators_holder = ContinuousVector::Create<GertAllocator *>(allocator_size);
  auto l2_allocators = reinterpret_cast<TypedContinuousVector<GertAllocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(static_cast<size_t>(allocator_size));
  auto l2_allocators_vec = l2_allocators->MutableData();
  for (size_t i = 0U; i < allocator_size; i++) {
    l2_allocators_vec[i] = allocator_holder.allocators.at(i);
  }
  auto context =
      KernelRunContextFaker()
          .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), l2_allocators})
          .Build();
  ASSERT_EQ(KernelRegistry::GetInstance().FindKernelFuncs("LastWaitEvents")->run_func(context), ge::GRAPH_SUCCESS);
}
TEST_F(EventUT, LastWaitEvents_RecycleL2ToL1) {
  auto event_ids = ContinuousVectorBuilder::Create<int64_t>({});
  auto rt_events = ContinuousVectorBuilder::Create<rtEvent_t>({});
  auto l1_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto allocator_holder = memory::MultiStreamAllocatorFaker().StreamNum(4).L1Allocator(l1_allocator).Build();
  std::vector<GertEvent> events{};
  auto allocator_size = allocator_holder.stream_ids_to_allocator.size();
  auto l2_allocators_holder = ContinuousVector::Create<GertAllocator *>(allocator_size);
  auto l2_allocators = reinterpret_cast<TypedContinuousVector<GertAllocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(static_cast<size_t>(allocator_size));
  auto l2_allocators_vec = l2_allocators->MutableData();
  ASSERT_NE(l1_allocator->GetScalableAllocator(), nullptr);
  auto origin_occupied_size = l1_allocator->GetScalableAllocator()->GetOccupiedMemSize();
  for (size_t i = 0U; i < allocator_size; i++) {
    l2_allocators_vec[i] = allocator_holder.at(i);
    auto block = l2_allocators_vec[i]->Malloc(1024U);
    ASSERT_NE(block, nullptr);
    block->Free(l2_allocators_vec[i]->GetStreamId());
  }
  ASSERT_NE(l1_allocator->GetScalableAllocator()->GetOccupiedMemSize(), origin_occupied_size);
  auto context =
      KernelRunContextFaker()
          .Inputs({(rtStream_t)0x1000, event_ids.get(), &events, rt_events.get(), l2_allocators})
          .Build();
  ASSERT_EQ(KernelRegistry::GetInstance().FindKernelFuncs("LastWaitEvents")->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(l1_allocator->GetScalableAllocator()->GetOccupiedMemSize(), origin_occupied_size);
}
}  // namespace kernel
}  // namespace gert