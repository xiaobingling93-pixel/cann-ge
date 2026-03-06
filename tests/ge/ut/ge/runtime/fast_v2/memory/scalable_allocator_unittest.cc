/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <math.h>
#include <thread>

#include "macro_utils/dt_public_scope.h"

#include "kernel/memory/allocator/scalable_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "kernel/memory/rts_caching_mem_allocator.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/graph_var_manager.h"

#include "macro_utils/dt_public_unscope.h"

#include "depends/runtime/src/runtime_stub.h"
#include "ge_local_context.h"
#include "stub/gert_runtime_stub.h"

using namespace gert;
using namespace testing;

namespace {
class MockRuntime : public ge::RuntimeStub {
 public:
  rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
    *free = 64UL * 1024UL * 1024UL;
    *total = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U];
    return RT_ERROR_NONE;
  }
};
class MockRtsCachingMemAllocator : public gert::memory::RtsCachingMemAllocator {
   public:
    MockRtsCachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type)
        : RtsCachingMemAllocator(device_id, memory_type) {
    }
    PageSpan *BlockAlloc(ge::Allocator& allocator, const BlockAddr block_addr, const MemAddr addr, const size_t size) override {
      return nullptr;
    }
};

// stub rtMalloc, always return failed
class RuntimeMock : public ge::RuntimeStub {
 public:
  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
    *dev_ptr = nullptr;
    return -1;
  }
};

void ThreadFunction() {
  static size_t count = 0;
  ++count;
  rtStream stream = (rtStream)count;
  gert::memory::CachingMemAllocator allocator(0);
  allocator.SetStream(stream);
  for (int i = 0; i < 10; ++i) {
    auto ptr = allocator.Malloc(1024 * count); // Malloc 1024 bytes
    if (ptr != nullptr) {
      allocator.Free(ptr);
    }
  }
}
}
struct ScaleAllocatorTest : public testing::Test {
 protected:
  void SetUp() {
    if (caching_allocator == nullptr) {
      ge::VarManagerPool::Instance().Destory();
      memory::RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
      memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
      auto mock_runtime = std::make_shared<MockRuntime>();
      ge::RuntimeStub::SetInstance(mock_runtime);
      caching_allocator.reset(new memory::CachingMemAllocator{0, RT_MEMORY_HBM});
      //allocator.reset(new ScalableAllocator{caching_allocator->span_allocator_, caching_allocator->rts_mem_allocator_});
      ge::RuntimeStub::Reset();
    }
  }
  void TearDown() {
  }

  timespec interval(const timespec &start, const timespec &end) {
    timespec temp_time;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
      temp_time.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
      temp_time.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp_time;
  }

 protected:
  size_t pageIdem{PAGE_SIZE_IDEM_DEFAULT};
  MemSize pageSize{PageLen_GetMemSize(1, pageIdem)};

  std::unique_ptr<memory::CachingMemAllocator> caching_allocator = nullptr;
  std::unique_ptr<ScalableAllocator> allocator = nullptr;
};

TEST_F(ScaleAllocatorTest, should_init_in_correct_state) {
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleMemSize());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());
}

TEST_F(ScaleAllocatorTest, should_alloc_and_free_0_size) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 0);

  ASSERT_TRUE(span != nullptr);
  ASSERT_EQ(1, span->GetCount());
  ASSERT_EQ(1, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleMemSize());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());

  span->Free();

  ASSERT_EQ(0, span->GetCount());
  ASSERT_EQ(1, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdlePageCountOfLayer(1));
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetIdleMemSizeOfLayer(1));

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetIdleMemSize());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());
}

TEST_F(ScaleAllocatorTest, should_alloc_and_free_size_less_than_one_page) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize - 1);

  ASSERT_TRUE(span != nullptr);
  ASSERT_EQ(1, span->GetCount());
  ASSERT_EQ(1, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleMemSize());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());

  span->Free();

  ASSERT_EQ(0, span->GetCount());
  ASSERT_EQ(1, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdlePageCountOfLayer(1));
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetIdleMemSizeOfLayer(1));

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetIdleMemSize());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());
}

TEST_F(ScaleAllocatorTest, should_alloc_and_free_size_larger_than_one_page) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize + 1);

  ASSERT_TRUE(span != nullptr);
  ASSERT_EQ(1, span->GetCount());
  ASSERT_EQ(2, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleMemSize());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(pageSize * 2, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());

  span->Free();

  ASSERT_EQ(0, span->GetCount());
  ASSERT_EQ(2, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(2));
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdlePageCountOfLayer(2));
  ASSERT_EQ(pageSize * 2, caching_allocator->GetScalableAllocator()->GetIdleMemSizeOfLayer(2));

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(pageSize * 2, caching_allocator->GetScalableAllocator()->GetIdleMemSize());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());
}

TEST_F(ScaleAllocatorTest, should_alloc_and_free_large_size) {
  size_t pageCount = 1024 * 30;
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * pageCount - 1);

  ASSERT_TRUE(span != nullptr);
  ASSERT_EQ(1, span->GetCount());
  ASSERT_EQ(pageCount, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleMemSize());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(pageCount, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(pageSize * pageCount, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());

  span->Free();

  ASSERT_EQ(0, span->GetCount());
  ASSERT_EQ(pageCount, span->GetPageLen());
  ASSERT_FALSE(span->HasSplited());
  ASSERT_TRUE(span->IsBuddyHeader());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(pageCount));
  ASSERT_EQ(pageCount, caching_allocator->GetScalableAllocator()->GetIdlePageCountOfLayer(pageCount));
  ASSERT_EQ(pageSize * pageCount, caching_allocator->GetScalableAllocator()->GetIdleMemSizeOfLayer(pageCount));

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(pageCount, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(pageSize * pageCount, caching_allocator->GetScalableAllocator()->GetIdleMemSize());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());
}

TEST_F(ScaleAllocatorTest, should_alloc_and_free_overflow_size) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U] + 1);

  ASSERT_TRUE(span == nullptr);

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdlePageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleMemSize());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedPageCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedMemSize());
}

TEST_F(ScaleAllocatorTest, should_not_free_referenced_span) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 0);

  span->AddCount();
  span->Free();

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  span->Free();
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
}

TEST_F(ScaleAllocatorTest, should_alloc_and_free_multiple_times_by_same_size) {
  MemSize size = pageSize * 16;

  for (size_t i = 0; i < 10; i++) {
    auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, size);
    ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
    ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

    span->Free();
    ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(16));
    ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
    ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  }
}

TEST_F(ScaleAllocatorTest, should_alloc_multiple_times_then_free_by_same_size) {
  MemSize size = pageSize * 16;

  constexpr size_t ALLOC_COUNT = 10;
  PageSpan* spans[ALLOC_COUNT] = {nullptr};

  for (size_t i = 0; i < ALLOC_COUNT; i++) {
    spans[i] = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, size);
    ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
    ASSERT_EQ(i + 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  }

  for (size_t i = 0; i < ALLOC_COUNT; i++) {
    spans[i]->Free();
    ASSERT_EQ(i + 1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
    ASSERT_EQ(ALLOC_COUNT - i - 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  }

  ASSERT_EQ(ALLOC_COUNT, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(16));
}

TEST_F(ScaleAllocatorTest, should_alloc_multiple_times_then_free_by_different_size) {
  constexpr size_t ALLOC_COUNT = 10;
  PageSpan* spans[ALLOC_COUNT] = {nullptr};

  for (size_t i = 0; i < ALLOC_COUNT; i++) {
    spans[i] = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * (i + 1));
    ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
    ASSERT_EQ(i + 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  }

  for (size_t i = 0; i < ALLOC_COUNT; i++) {
    spans[i]->Free();
    ASSERT_EQ(i + 1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
    ASSERT_EQ(ALLOC_COUNT - i - 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
    ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(i + 1));
  }
}

TEST_F(ScaleAllocatorTest, should_split_from_larger_span) {
  size_t buddyPageCount1 = 16;
  size_t buddyPageCount2 = 2;

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  ASSERT_EQ(buddyPageCount1, span1->GetPageLen());
  ASSERT_EQ(1,  span1->GetCount());
  ASSERT_TRUE(span1->IsBuddyHeader());
  ASSERT_FALSE(span1->HasSplited());

  span1->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1));

  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount2);
  ASSERT_EQ(1,  span2->GetCount());
  ASSERT_EQ(buddyPageCount2, span2->GetPageLen());
  ASSERT_TRUE(span2->HasSplited());
  ASSERT_FALSE(span2->IsBuddyHeader());

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1));
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1 - buddyPageCount2));

  ASSERT_EQ(buddyPageCount1 - buddyPageCount2, span1->GetPageLen());
  ASSERT_EQ(0,  span1->GetCount());
  ASSERT_TRUE(span1->IsBuddyHeader());
  ASSERT_TRUE(span1->HasSplited());

  ASSERT_EQ(static_cast<const uint8_t *>(span1->GetAddr()) + (buddyPageCount1 - buddyPageCount2) * pageSize,
            static_cast<const uint8_t *>(span2->GetAddr()));
  ASSERT_EQ(span1->GetNextBuddy() , span2);
  ASSERT_EQ(span2->GetPrevBuddy() , span1);

  span2->Free();
  ASSERT_EQ(0,  span2->GetCount());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1));
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount2));

  ASSERT_EQ(buddyPageCount1, span1->GetPageLen());
  ASSERT_EQ(0,  span1->GetCount());
  ASSERT_TRUE(span1->IsBuddyHeader());
  ASSERT_FALSE(span1->HasSplited());
}

TEST_F(ScaleAllocatorTest, should_split_from_a_splited_span) {
  size_t buddyPageCount1 = 16;
  size_t buddyPageCount2 = 2;
  size_t buddyPageCount3 = 1;

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  span1->Free();

  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount2);
  auto span3 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount3);

  ASSERT_EQ(buddyPageCount1 - buddyPageCount2 - buddyPageCount3, span1->GetPageLen());

  ASSERT_TRUE(span1->IsBuddyHeader());
  ASSERT_TRUE(span1->HasSplited());
  ASSERT_TRUE(span2->HasSplited());
  ASSERT_TRUE(span3->HasSplited());

  ASSERT_EQ(static_cast<const uint8_t *>(span1->GetAddr()) + (buddyPageCount1 - buddyPageCount2 - buddyPageCount3) * pageSize,
            static_cast<const uint8_t *>(span3->GetAddr()));
  ASSERT_EQ(static_cast<const uint8_t *>(span3->GetAddr()) + buddyPageCount3 * pageSize,
            static_cast<const uint8_t *>(span2->GetAddr()));

  ASSERT_EQ(span1->GetNextBuddy() , span3);
  ASSERT_EQ(span3->GetPrevBuddy() , span1);

  ASSERT_EQ(span3->GetNextBuddy() , span2);
  ASSERT_EQ(span2->GetPrevBuddy() , span3);

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1 - buddyPageCount2 - buddyPageCount3));

  span2->Free();

  ASSERT_EQ(buddyPageCount1 - buddyPageCount2 - buddyPageCount3, span1->GetPageLen());
  ASSERT_EQ(buddyPageCount2, span2->GetPageLen());
  ASSERT_EQ(buddyPageCount3, span3->GetPageLen());

  ASSERT_TRUE(span1->IsBuddyHeader());
  ASSERT_TRUE(span1->HasSplited());
  ASSERT_TRUE(span2->HasSplited());
  ASSERT_TRUE(span3->HasSplited());

  ASSERT_EQ(static_cast<const uint8_t *>(span1->GetAddr()) + (buddyPageCount1 - buddyPageCount2 - buddyPageCount3) * pageSize,
            static_cast<const uint8_t *>(span3->GetAddr()));
  ASSERT_EQ(static_cast<const uint8_t *>(span3->GetAddr()) + buddyPageCount3 * pageSize,
            static_cast<const uint8_t *>(span2->GetAddr()));

  ASSERT_EQ(span1->GetNextBuddy() , span3);
  ASSERT_EQ(span3->GetPrevBuddy() , span1);

  ASSERT_EQ(span3->GetNextBuddy() , span2);
  ASSERT_EQ(span2->GetPrevBuddy() , span3);

  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1 - buddyPageCount2 - buddyPageCount3));
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount2));

  span3->Free();

  ASSERT_EQ(buddyPageCount1, span1->GetPageLen());

  ASSERT_FALSE(span1->HasSplited());
  ASSERT_FALSE(span2->HasSplited());
  ASSERT_FALSE(span3->HasSplited());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1));
}

TEST_F(ScaleAllocatorTest, should_alloc_buddy_header) {
  size_t buddyPageCount1 = 16;
  size_t buddyPageCount2 = 2;
  size_t buddyPageCount3 = 1;

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  span1->Free();

  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount2);
  auto span3 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount3);

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1 - buddyPageCount2 - buddyPageCount3));

  auto span4 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * (buddyPageCount1 - buddyPageCount2 - buddyPageCount3));

  ASSERT_EQ(span1, span4);
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());

  ASSERT_TRUE(span4->IsBuddyHeader());
  ASSERT_TRUE(span4->HasSplited());
  ASSERT_TRUE(span2->HasSplited());
  ASSERT_TRUE(span3->HasSplited());

  ASSERT_EQ(static_cast<const uint8_t *>(span1->GetAddr()) + (buddyPageCount1 - buddyPageCount2 - buddyPageCount3) * pageSize,
            static_cast<const uint8_t *>(span3->GetAddr()));
  ASSERT_EQ(static_cast<const uint8_t *>(span3->GetAddr()) + buddyPageCount3 * pageSize,
            static_cast<const uint8_t *>(span2->GetAddr()));

  ASSERT_EQ(span1->GetNextBuddy() , span3);
  ASSERT_EQ(span3->GetPrevBuddy() , span1);

  ASSERT_EQ(span3->GetNextBuddy() , span2);
  ASSERT_EQ(span2->GetPrevBuddy() , span3);

  span3->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount3));

  ASSERT_EQ(buddyPageCount1 - buddyPageCount2 - buddyPageCount3, span1->GetPageLen());
  ASSERT_EQ(buddyPageCount2, span2->GetPageLen());
  ASSERT_EQ(buddyPageCount3, span3->GetPageLen());

  ASSERT_TRUE(span1->IsBuddyHeader());
  ASSERT_TRUE(span1->HasSplited());
  ASSERT_TRUE(span2->HasSplited());
  ASSERT_TRUE(span3->HasSplited());

  ASSERT_EQ(static_cast<const uint8_t *>(span4->GetAddr()) + (buddyPageCount1 - buddyPageCount2 - buddyPageCount3) * pageSize,
            static_cast<const uint8_t *>(span3->GetAddr()));
  ASSERT_EQ(static_cast<const uint8_t *>(span3->GetAddr()) + buddyPageCount3 * pageSize,
            static_cast<const uint8_t *>(span2->GetAddr()));

  ASSERT_EQ(span4->GetNextBuddy() , span3);
  ASSERT_EQ(span3->GetPrevBuddy() , span4);

  ASSERT_EQ(span3->GetNextBuddy() , span2);
  ASSERT_EQ(span2->GetPrevBuddy() , span3);

  span2->Free();

  ASSERT_EQ(buddyPageCount1 - buddyPageCount2 - buddyPageCount3, span1->GetPageLen());
  ASSERT_EQ(buddyPageCount2 + buddyPageCount3, span3->GetPageLen());

  ASSERT_TRUE(span4->HasSplited());
  ASSERT_TRUE(span3->HasSplited());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount2 + buddyPageCount3));
  ASSERT_EQ(static_cast<const uint8_t *>(span4->GetAddr()) + (buddyPageCount1 - buddyPageCount2 - buddyPageCount3) * pageSize,
            static_cast<const uint8_t *>(span3->GetAddr()));

  span4->Free();
  ASSERT_FALSE(span4->HasSplited());
  ASSERT_EQ(buddyPageCount1, span1->GetPageLen());

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(buddyPageCount1));
}

TEST_F(ScaleAllocatorTest, should_not_recyle_occupied_spans) {
  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  span2->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  caching_allocator->GetScalableAllocator()->Recycle();

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  ASSERT_EQ(1, span1->GetCount());

  span1->Free();
}

TEST_F(ScaleAllocatorTest, should_not_recyle_splited_spans) {
  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 2 * pageSize);
  span1->Free();

  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  caching_allocator->GetScalableAllocator()->Recycle();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  span2->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  caching_allocator->GetScalableAllocator()->Recycle();

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
}

TEST_F(ScaleAllocatorTest, should_alarm_memory_leaks) {
  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 2 * pageSize);
  span1->Free();

  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  span2->Free();
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
}

TEST_F(ScaleAllocatorTest, should_alloc_recycle_when_exceeds_total_threshold) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  auto largeSpan = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0] - pageSize + 1);
  ASSERT_EQ(1, span->GetCount());
  ASSERT_TRUE(largeSpan != nullptr);
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetRecycleCount());
  ASSERT_EQ(PageLen_GetLenFromSize(PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0], pageIdem), largeSpan->GetPageLen());
  span->Free();
  largeSpan->Free();
}

TEST_F(ScaleAllocatorTest, should_alloc_free_recycle_when_exceeds_total_threshold) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  auto largeSpan = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0] - pageSize + 1);
  ASSERT_EQ(1, span->GetCount());
  ASSERT_TRUE(largeSpan != nullptr);
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetRecycleCount());
  ASSERT_EQ(PageLen_GetLenFromSize(PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0], pageIdem), largeSpan->GetPageLen());

  span->Free();
  largeSpan->Free();
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  auto largeSpan1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0] - pageSize + 1);
  ASSERT_TRUE(span->GetAddr() == span1->GetAddr());
  ASSERT_TRUE(largeSpan->GetAddr() == largeSpan1->GetAddr());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  span1->Free();
  largeSpan1->Free();
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  caching_allocator->GetScalableAllocator()->Recycle();
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
}

TEST_F(ScaleAllocatorTest, should_release_when_free_because_of_layer_span_count_exceeds_threshold) {
  PageSpan* spans[SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT] = {nullptr};

  for (auto& span : spans) {
    span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  }

  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT + 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());


  for (auto& span : spans) {
    span->Free();
  }

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));

  span->Free();

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT + 1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT + 1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));

  caching_allocator->GetScalableAllocator()->Recycle();
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
}

TEST_F(ScaleAllocatorTest, should_reuse_even_if_layer_span_count_exceeds_threshold) {
  PageSpan* spans[SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT] = {nullptr};

  for (auto& span : spans) {
    span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  }

  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT + 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  span->Free();
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  ASSERT_TRUE(span->GetAddr() == span1->GetAddr());
  ASSERT_EQ(SPAN_COUNT_IN_LAYER_THRESHOLD_DEFAULT + 1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));

  for (auto& span : spans) {
    span->Free();
  }
  span1->Free();
  caching_allocator->GetScalableAllocator()->Recycle();

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
}

TEST_F(ScaleAllocatorTest, should_release_when_free_because_of_layer_mem_size_exceeds_threshold) {
  MemSize size = PAGE_MEM_SIZE_IN_LAYER_THRESHOLD_DEFAULT / 2;

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, size);
  ASSERT_TRUE(span1 != nullptr);

  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, size);
  ASSERT_TRUE(span2 != nullptr);

  auto span3 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, size);
  ASSERT_TRUE(span3 != nullptr);

  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(3, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  span1->Free();
  span2->Free();

  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());

  span3->Free();

  ASSERT_EQ(3, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(0, caching_allocator->GetScalableAllocator()->GetOccupiedSpanCount());
}

TEST_F(ScaleAllocatorTest, ffts_allocator_memory_reuse) {
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_TRUE(level_1_allocator != nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 4U);
  ASSERT_TRUE(level_2_allocator != nullptr);

  auto span1 = level_2_allocator->Malloc(1024U);
  ASSERT_TRUE(span1 != nullptr);
  const size_t kBlockSize = 65536U;
  ASSERT_TRUE(span1->Addr(0U) == span1->GetAddr());
  auto base_addr = static_cast<const uint8_t *>(span1->GetAddr());
  ASSERT_TRUE(span1->Addr(1U) == static_cast<const void *>(base_addr + kBlockSize));
  ASSERT_TRUE(span1->Addr(2U) == static_cast<const void *>(base_addr + kBlockSize * 2));
  ASSERT_TRUE(span1->Addr(3U) == static_cast<const void *>(base_addr + kBlockSize * 3));
  ASSERT_TRUE(span1->Addr(4U) == span1->GetAddr());
  ASSERT_TRUE(span1->Addr(5U) == static_cast<const void *>(base_addr + kBlockSize));
  ASSERT_TRUE(span1->Addr(6U) == static_cast<const void *>(base_addr + kBlockSize * 2));
  ASSERT_TRUE(span1->Addr(7U) == static_cast<const void *>(base_addr + kBlockSize * 3));

  auto span2 = level_2_allocator->Malloc(65536U);
  ASSERT_TRUE(span2 != nullptr);
  ASSERT_TRUE(span1->GetAddr() != span2->GetAddr());
  span1->Free();
  auto span3 = level_2_allocator->Malloc(1024U);
  ASSERT_TRUE(span3 != nullptr);
  ASSERT_TRUE(span1->GetAddr() == span3->GetAddr());
  span2->Free();
  span3->Free();
  level_2_allocator->Recycle();
  ASSERT_EQ(0, level_2_allocator->GetOccupiedSpanCount());
}

TEST_F(ScaleAllocatorTest, split_span_size) {
  size_t buddyPageCount1 = 4;
  size_t buddyPageCount2 = 2;
  size_t buddyPageCount3 = 1;

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
  span1->Free();
  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount2);
  ASSERT_EQ(span2->GetSize(), pageSize * buddyPageCount2);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount2);
  auto span3 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, buddyPageCount3);
  ASSERT_EQ(span3->GetSize(), pageSize);
  ASSERT_EQ(span1->GetSize(), pageSize);
  auto span4 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, buddyPageCount3);
  ASSERT_EQ(span4->GetSize(), pageSize);
  ASSERT_EQ(span1->GetSize(), pageSize);
  span4->Free();
  span3->Free();
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount2);
  span2->Free();
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
}

TEST_F(ScaleAllocatorTest, FetchSplitedSpan_CheckMemBlockSize_OK) {
  size_t buddyPageCount1 = 4;
  size_t buddyPageCount2 = 2;
  size_t buddyPageCount3 = 1;

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
  auto mem_block_1 = dynamic_cast<ge::MemBlock *>(span1);
  ASSERT_NE(mem_block_1, nullptr);
  ASSERT_EQ(mem_block_1->GetSize(), span1->GetSize());

  span1->Free();
  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount2);
  ASSERT_EQ(span2->GetSize(), pageSize * buddyPageCount2);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount2);
  auto mem_block_2 = dynamic_cast<ge::MemBlock *>(span2);
  ASSERT_NE(mem_block_2, nullptr);
  ASSERT_EQ(mem_block_2->GetSize(), span2->GetSize());

  auto span3 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, buddyPageCount3);
  ASSERT_EQ(span3->GetSize(), pageSize);
  ASSERT_EQ(span1->GetSize(), pageSize);
  auto mem_block_3 = dynamic_cast<ge::MemBlock *>(span3);
  ASSERT_NE(mem_block_3, nullptr);
  ASSERT_EQ(mem_block_3->GetSize(), span3->GetSize());

  auto span4 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, buddyPageCount3);
  ASSERT_EQ(span4->GetSize(), pageSize);
  ASSERT_EQ(span1->GetSize(), pageSize);
  auto mem_block_4 = dynamic_cast<ge::MemBlock *>(span4);
  ASSERT_NE(mem_block_4, nullptr);
  ASSERT_EQ(mem_block_4->GetSize(), span4->GetSize());

  span4->Free();
  span3->Free();
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount2);
  span2->Free();
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
}

TEST_F(ScaleAllocatorTest, ScalableConfig_Less32G) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 64UL * 1024UL * 1024UL;
      *total = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0U];
      return RT_ERROR_NONE;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  const ScalableConfig &cfg = ScalableConfig();
  auto page_mem_size_total_threshold =
      static_cast<size_t>(floor(static_cast<float64_t>(PAGE_MEM_SIZE_THRESHOLD_DEFAULT[0U]) * kMaxMemorySizeRatio));
  ASSERT_EQ(cfg.page_mem_size_total_threshold, page_mem_size_total_threshold);
  ASSERT_EQ(cfg.uncacheable_size_threshold, SPAN_UNCACHEABLE_MEM_SIZE_DEFAULT[0U]);
}

TEST_F(ScaleAllocatorTest, ScalableConfig_Greater32G_Less64G) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 64UL * 1024UL * 1024UL;
      *total = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U];
      return RT_ERROR_NONE;
    }
  };

  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  const ScalableConfig &cfg = ScalableConfig();
  auto page_mem_size_total_threshold =
    static_cast<size_t>(floor(static_cast<float64_t>(PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U]) * kMaxMemorySizeRatio));
  ASSERT_EQ(cfg.page_mem_size_total_threshold, page_mem_size_total_threshold);
  ASSERT_EQ(cfg.uncacheable_size_threshold, SPAN_UNCACHEABLE_MEM_SIZE_DEFAULT[1U]);
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, ffts_allocator_split_memory_reuse) {
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_TRUE(level_1_allocator != nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 4U);
  ASSERT_TRUE(level_2_allocator != nullptr);

  auto span1 = level_2_allocator->Malloc(1024U * 1024U);
  ASSERT_TRUE(span1 != nullptr);
  auto base_addr = span1->GetAddr();

  ASSERT_TRUE(span1->Addr(0U) == span1->GetAddr());
  span1->Free();

  auto span2 = level_2_allocator->Malloc(65536U);
  ASSERT_EQ(static_cast<const uint8_t *>(span1->GetAddr()) + 983040U, static_cast<const uint8_t *>(span2->GetAddr()));
  ASSERT_TRUE(span2 != nullptr);
  span2->Free();

  auto span3 = level_2_allocator->Malloc(1024U * 1024U);
  ASSERT_TRUE(span3 != nullptr);
  ASSERT_EQ(span3->GetAddr(), base_addr);
  span3->Free();
}

TEST_F(ScaleAllocatorTest, alloc_total_exceed_thresold) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 64UL * 1024UL * 1024UL;
      *total = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U];
      return RT_ERROR_NONE;
    }
  };

  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  auto alloc_size = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[1U]/2;
  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 1024U);
  span1->Free();
  span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, alloc_size);
  ASSERT_TRUE(span1 != nullptr);
  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, alloc_size);
  ASSERT_TRUE(span2 == nullptr);
  span1->Free();
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, alloc_manager_create_allocator_success) {
  memory::RtsAllocatorManager rts_allocator_manager;
  auto allocator = rts_allocator_manager.CreateAllocator(0U, RT_MEMORY_HBM);
  ASSERT_TRUE(allocator != nullptr);
  memory::RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
  memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
}

TEST_F(ScaleAllocatorTest, rts_caching_allocator_alloc_total_exceed_thresold) {
  gert::memory::RtsCachingMemAllocator allocator(0U, RT_MEMORY_HBM);
  auto span1 = allocator.Malloc(1024UL * 1024UL + pageSize);
  ASSERT_TRUE(span1 != nullptr);
  auto alloc_size = allocator.config_.page_mem_size_total_threshold - 4 * 1024UL*1024UL;
  auto span2 = allocator.Malloc(alloc_size);
  ASSERT_TRUE(span2 != nullptr);
  auto span3 = allocator.Malloc(1024UL * 1024UL);
  ASSERT_TRUE(span3 == nullptr);
  span2->Free();
  span1->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, rts_caching_allocator_alloc_multiple_times_by_same_size) {
  gert::memory::RtsCachingMemAllocator allocator(0U, RT_MEMORY_HBM);
  auto span1 = allocator.Malloc(512UL * 1024UL);
  ASSERT_TRUE(span1 != nullptr);
  auto span2 = allocator.Malloc(512UL * 1024UL);
  ASSERT_TRUE(span2 != nullptr);
  auto span3 = allocator.Malloc(512UL * 1024UL);
  ASSERT_TRUE(span3 != nullptr);
  auto span4 = allocator.Malloc(512UL * 1024UL);
  ASSERT_TRUE(span4 != nullptr);
  ASSERT_EQ(static_cast<const uint8_t *>(span1->GetAddr()),
            static_cast<const uint8_t *>(span4->GetAddr()) + 3 * 512UL *1024UL);
  span1->Free();
  span2->Free();
  span3->Free();
  span4->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(ScaleAllocatorTest, rts_caching_allocator_alloc_fail) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      return -1;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  gert::memory::RtsCachingMemAllocator allocator(0U, RT_MEMORY_HBM);
  auto span = allocator.Malloc(1024U);
  ASSERT_TRUE(span == nullptr);
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, rts_caching_allocator_block_alloc_fail) {
  auto allocator = std::make_shared<MockRtsCachingMemAllocator>(0U, RT_MEMORY_HBM);
  auto span = allocator->Malloc(1024U);
  ASSERT_TRUE(span == nullptr);
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator->Finalize());
}

TEST_F(ScaleAllocatorTest, delay_split_large_span) {
  size_t buddyPageCount1 = 10 * 1024; // 640M
  size_t buddyPageCount2 = 32; // 2M
  size_t buddyPageCount3 = 64; // 4M
  size_t buddyPageCount4 = 64; // 4M

  auto span1 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
  auto addr1 = span1->GetAddr();
  span1->Free();
  auto span2 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount2);
  ASSERT_EQ(span2->GetSize(), pageSize * buddyPageCount2);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
  ASSERT_EQ(span1->try_split_page_len_, buddyPageCount2);
  auto span3 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount3);
  ASSERT_EQ(span3->GetSize(), pageSize * buddyPageCount3);
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
  ASSERT_EQ(span1->try_split_page_len_, (buddyPageCount2 + buddyPageCount3));
  auto span4 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount4);
  ASSERT_EQ(span4->GetSize(), pageSize * buddyPageCount4);
  ASSERT_EQ(span1->GetSize(), pageSize * (buddyPageCount1 - buddyPageCount4));
  ASSERT_EQ(span1->try_split_page_len_, (buddyPageCount2 + buddyPageCount3 + buddyPageCount4));
  auto span5 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount4);
  ASSERT_EQ(span5->GetSize(), pageSize * buddyPageCount4);
  ASSERT_EQ(span1->GetSize(), pageSize * (buddyPageCount1 - buddyPageCount4 - buddyPageCount4));
  ASSERT_EQ(span1->try_split_page_len_, (buddyPageCount2 + buddyPageCount3 + buddyPageCount4));
  span4->Free();
  span5->Free();
  ASSERT_EQ(span1->GetSize(), pageSize * buddyPageCount1);
  ASSERT_EQ(span1->try_split_page_len_, 0);
  span2->Free();
  span3->Free();
  auto span6 = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize * buddyPageCount1);
  ASSERT_TRUE(span6 != nullptr);
  ASSERT_TRUE(addr1 == span6->GetAddr());
  span6->Free();
}

TEST_F(ScaleAllocatorTest, test_init_fail) {
  caching_allocator->GetScalableAllocator()->span_layer_lut_ = nullptr;
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);
  ASSERT_EQ(span, nullptr);
  caching_allocator->GetScalableAllocator()->Recycle();
  ASSERT_EQ(caching_allocator->GetScalableAllocator()->Finalize(), ge::FAILED);
}

TEST_F(ScaleAllocatorTest, test_recycle_time) {
  struct timespec t_begin;
  struct timespec t_end;
  clock_gettime(CLOCK_REALTIME, &t_begin);
  caching_allocator->GetScalableAllocator()->Recycle();
  clock_gettime(CLOCK_REALTIME, &t_end);
  ASSERT_LE(interval(t_begin, t_end).tv_nsec, 100000U);
}

TEST_F(ScaleAllocatorTest, convert_span_to_root_block_success) {
  size_t buddyPageCount = 10U * 1024U;
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  auto span = scalable_allocator->Alloc(*caching_allocator, pageSize * buddyPageCount);
  ASSERT_NE(span, nullptr);
  auto block = span->GetBlockAddr();
  auto root_block = scalable_allocator->ConvertToRootBlock(span);
  ASSERT_EQ(block, root_block);
  block->Free();
}

TEST_F(ScaleAllocatorTest, convert_splited_span_to_root_block_failed) {
  size_t buddyPageCount = 10U * 1024U;
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  auto span = scalable_allocator->Alloc(*caching_allocator, pageSize * buddyPageCount);
  ASSERT_NE(span, nullptr);
  scalable_allocator->Free(span);
  auto splited_span = scalable_allocator->Alloc(*caching_allocator, pageSize * buddyPageCount / 2U);
  ASSERT_EQ(span->HasSplited(), true);
  auto root_block = scalable_allocator->ConvertToRootBlock(span);
  ASSERT_EQ(root_block, nullptr);
  scalable_allocator->Free(splited_span);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 15U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  auto addr2 = span2->GetAddr();
  span1->Free();
  span2->Free();
  auto span3 = caching_allocator->Malloc(alloc_size * 2);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  auto addr3 = span3->GetAddr();
  ASSERT_EQ(addr2, addr3);
  span3->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0);

  auto span4 = caching_allocator->Malloc(alloc_size * 2);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  auto addr4 = span4->GetAddr();
  ASSERT_EQ(addr4, addr3);
  span4->Free();
  caching_allocator.reset(nullptr);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_onepage) {
  size_t alloc_size = ge::kLargePageSize;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), ge::kLargePageSize);
  span1->Free();
  caching_allocator.reset(nullptr);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_recycle) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 15U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  auto addr2 = span2->GetAddr();
  span1->Free();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span2->Free();
  auto span3 = caching_allocator->Malloc(alloc_size * 2);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  auto addr3 = span3->GetAddr();
  ASSERT_EQ(addr2, addr3);
  span3->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_merge_ref_count_test) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span3 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span3, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span4 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span4, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  span3->Free();
  span2->Free();

  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  span1->Free();
  span4->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);

  span4->Free();
  caching_allocator.reset(nullptr);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_reuse_ref_count_test) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  span1->Free();
  auto span3 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span3, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  span3->Free();

  auto span4 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span4, nullptr);
  span4->Free();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);

  span2->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);
  caching_allocator.reset(nullptr);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_fail) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      static size_t cnt = 0U;
      ++cnt;
      if (cnt >= 3U) {
        return -1;
      }
      *handle = (rtDrvMemHandle) new uint8_t[8];;
      return 0;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 10U * 2U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  ASSERT_EQ(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);

  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);
  caching_allocator.reset(nullptr);
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_hole_12M) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 12U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |--11|1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U); // 12M
  auto span2 = caching_allocator->Malloc(10U * 1024U *1024U);
  // va |-222|2211|1111|
  ASSERT_NE(span2, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U); // 24M
  auto addr2 = span2->GetAddr();
  span1->Free(); // 12M
  // va |-222|22--|----|

  auto span3 = caching_allocator->Malloc(alloc_size * 2); // 36M
  // new_va |3333|3333|3333|
  // va |3333|3333|-222|22--|3333|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 40U * 1024U * 1024U);
  auto addr3 = span3->GetAddr();
  ASSERT_NE(addr2, addr3);
  span2->Free(); // 24M
  // new_va |3333|3333|3333|
  // va |3333|3333|----|----|3333|

  auto span4 = caching_allocator->Malloc(32U * 1024U * 1024U); // 56M
  // new_va |3333|3333|3333|
  // va |4444|4444|3333|3333|4444|4444|3333|

  auto addr4 = span4->GetAddr();
  ASSERT_NE(addr4, addr2);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 56U * 1024U * 1024U);
  span3->Free();// 32M
  // new_va |----|----|----|
  // va |4444|4444|----|----|4444|4444|----|

  auto span5 = caching_allocator->Malloc(alloc_size * 2); // 56M
  // new_va |5555|5555|5555|
  // va |4444|4444|5555|5555|4444|4444|5555|

  auto addr5 = span5->GetAddr();
  ASSERT_EQ(addr5, addr3);
  ASSERT_NE(static_cast<const uint8_t *>(span5->GetAddr()),
      static_cast<const uint8_t *>(span4->GetAddr()) + 16 * 1024U *1024U);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 56U * 1024U * 1024U);
  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));
  span5->Free();
  span4->Free();
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_hole_2M) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 2U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |---1|
  ASSERT_NE(span1, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |--21|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size * 3);
  // va |---3|3321|
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);
  span2->Free();
  // va |---3|33-1|

  auto span4 = caching_allocator->Malloc(alloc_size * 2);
  // va |-443|33-1|
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  span1->Free();
  span2->Free();
  span3->Free();
  span4->Free();
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_hole_12M_and_2M) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 12U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |--11|1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(2U * 1024U *1024U);
  // va |-211|1111|
  ASSERT_NE(span2, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(2U * 1024U *1024U);
  // va |3211|1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  span1->Free();
  // va |32--|----|

  auto span4 = caching_allocator->Malloc(24U * 1024U * 1024U);
  // new_va |4444|4444|4444|
  // va |4444|4444|32--|4444|
  auto addr4 = span4->GetAddr();

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 32U * 1024U * 1024U);
  span4->Free();
  // new_va |----|----|----|
  // va |----|----|32--|----|

  auto span5 = caching_allocator->Malloc(6U * 1024U * 1024U);
  // new_va |----|----|-555|
  // va |----|----|32--|-555|

  auto span6 = caching_allocator->Malloc(24U * 1024U * 1024U);
  // new_va |----|----|-555|
  // new_va |6666|6666|6666|
  // va |6666|6666|6666|32--|-555|

  auto addr6 = span5->GetAddr();
  ASSERT_NE(addr4, addr6);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 40U * 1024U * 1024U);
  span2->Free();
  span3->Free();
  span5->Free();
  span6->Free();
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_hole_8M) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  auto addr2 = span2->GetAddr();
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span3->Free();
  span1->Free();
  // va |----|2222|----|

  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |4444|4444|
  // va |4444|2222|4444|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span4->Free();
  // new_va |----|----|
  // va |----|2222|----|

  auto span5 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |5555|5555|
  // va |5555|2222|5555|
  auto addr5 = span5->GetAddr();
  span2->Free();
  // new_va |5555|5555|
  // va |5555|----|5555|

  auto span6 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |5555|5555|
  // new_va |6666|6666|
  // va |6666|5555|6666|5555|

  auto addr6 = span6->GetAddr();
  ASSERT_NE(addr2, addr6);
  ASSERT_NE(addr2, addr5);
  span5->Free();
  // new_va |----|----|
  // new_va |6666|6666|
  // va |6666|----|6666|----|

  auto span7 = caching_allocator->Malloc(alloc_size);
  // new_va |----|----|
  // new_va |6666|6666|
  // va |6666|----|6666|7777|

  auto span8 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |----|----|
  // new_va |6666|6666|
  // new_va |8888|8888|
  // va |8888|6666|8888|6666|7777|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 40U * 1024U * 1024U);
  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));

  span6->Free();
  span7->Free();
  span8->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);

  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_hole_oom) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtReserveMemAddress(void** devPtr, size_t size, size_t alignment, void *devAddr, uint64_t flags) {
      static size_t cnt = 0U;
      ++cnt;
      if (cnt == 4U) {
        return -1;
      }
      *devPtr = new uint8_t[1];
      return 0;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span3->Free();
  span1->Free();
  // va |----|2222|----|

  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |4444|4444|
  // va |4444|2222|4444|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span4->Free();
  // new_va |----|----|
  // va |----|2222|----|

  auto span5 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |5555|5555|
  // va |5555|2222|5555|
  span2->Free();
  // new_va |5555|5555|
  // va |5555|----|5555|

  auto span6 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |5555|5555|
  // new_va |6666|6666|
  // va |6666|5555|6666|5555|

  span5->Free();
  // new_va |----|----|
  // new_va |6666|6666|
  // va |6666|----|6666|----|

  auto span7 = caching_allocator->Malloc(alloc_size * 3U);
  // new_va |6666|6666|
  // new_va |7777|7777|7777|
  // va |7777|6666|7777|6666|7777|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 40U * 1024U * 1024U);
  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));
  span7->Free();
  span6->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);

  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_hole_map_fail) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMapMem(void* devPtr, size_t size, size_t offset, rtDrvMemHandle handle, uint64_t flags) {
      static size_t cnt = 0U;
      ++cnt;
      if (cnt >= 4U) {
        return -1;
      }
      return RT_ERROR_NONE;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);
  span1->Free();
  // va |2222|----|
  auto span3 = caching_allocator->Malloc(alloc_size * 2U);
  // va |3333|2222|3333|
  ASSERT_EQ(span3, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  span2->Free();
  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  ge::RuntimeStub::Reset();
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_recycle_full) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  auto addr2 = span2->GetAddr();
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span3->Free();
  span1->Free();
  // va |----|2222|----|

  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |4444|4444|
  // va |4444|2222|4444|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span4->Free();
  // new_va |----|----|
  // va |----|2222|----|

  auto span5 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |5555|5555|
  // va |5555|2222|5555|
  auto addr5 = span5->GetAddr();
  span2->Free();
  // new_va |5555|5555|
  // va |5555|----|5555|

  auto span6 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |5555|5555|
  // new_va |6666|6666|
  // va |6666|5555|6666|5555|

  auto addr6 = span6->GetAddr();
  ASSERT_NE(addr2, addr6);
  ASSERT_NE(addr2, addr5);
  span5->Free();
  // new_va |----|----|
  // new_va |6666|6666|
  // va |6666|----|6666|----|

  auto span7 = caching_allocator->Malloc(alloc_size);
  // new_va |----|----|
  // new_va |6666|6666|
  // va |6666|----|6666|7777|

  auto span8 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |----|----|
  // new_va |6666|6666|
  // new_va |8888|8888|
  // va |8888|6666|8888|6666|7777|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 40U * 1024U * 1024U);
  ASSERT_EQ(scalable_allocator->GetReachTheoryRate(), static_cast<float>(ge::kRatioBase));

  span6->Free();
  span7->Free();
  span8->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);
  auto physical_memory_size = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  auto caching_allocator1 = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator1 = caching_allocator1->GetScalableAllocator();
  auto span11 = caching_allocator1->Malloc(alloc_size * 2U);
  auto span12 = caching_allocator1->Malloc(alloc_size * 3U);
  auto physical_memory_size1 = scalable_allocator1->device_allocator_.GetExpandableAllocator()
     .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size, physical_memory_size1);
  span11->Free();
  span12->Free();
  scalable_allocator1->Recycle();
  ASSERT_EQ(scalable_allocator1->device_allocator_.GetOccupiedSize(), 0U);
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  scalable_allocator1->Finalize(false);
  caching_allocator1.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_recycle_partial) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span3->Free();
  span1->Free();
  // va |----|2222|----|

  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |4444|4444|
  // va |4444|2222|4444|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span4->Free();

  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), alloc_size);
  auto physical_memory_size = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  auto caching_allocator1 = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator1 = caching_allocator1->GetScalableAllocator();
  auto span11 = caching_allocator1->Malloc(alloc_size);
  auto physical_memory_size1 = scalable_allocator1->device_allocator_.GetExpandableAllocator()
     .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size, physical_memory_size1);
  auto span12 = caching_allocator1->Malloc(alloc_size * 2U);
  auto physical_memory_size2 = scalable_allocator1->device_allocator_.GetExpandableAllocator()
     .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size + 1U, physical_memory_size2);
  ASSERT_EQ(scalable_allocator1->device_allocator_.GetOccupiedSize(), alloc_size * 3U);
  span11->Free();
  span12->Free();
  span2->Free();
  scalable_allocator1->Recycle();
  ASSERT_EQ(scalable_allocator1->device_allocator_.GetOccupiedSize(), 0U);
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  scalable_allocator1->Finalize(false);
  caching_allocator1.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_paindex_selfreuse) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  // step1 malloc 3 pa, free, recycle; malloc 1 pa not free
  /*
   *     ┌────┐ ┌──────────────────┐
   * va  │occu│ │span1 free        │
   *     └─▲──┘ └──▲─────▲──────▲──┘
   *       │       │     │      │
   *       │       │     │      │
   *     ┌─┴──┐ ┌──┼─┐ ┌─┼──┐ ┌─┼──┐
   * pa  │ 3  │ │  2 │ │ 1  │ │ 0  │
   *     └────┘ └────┘ └────┘ └────┘
   */
  GELOGI("==========================span1 malloc 24M begin");
  auto span1 = caching_allocator->Malloc(alloc_size * 3);
  GELOGI("==========================span1 malloc 24M success");
  auto occupy1 = caching_allocator->Malloc(alloc_size);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), alloc_size * 4);
  GELOGI("==========================span1 free 24M begin");
  span1->Free();
  GELOGI("==========================span1 free 24M success");
  GELOGI("==========================scalable_allocator begin recycle");
  scalable_allocator->Recycle(); // put pa0/1/2 to pool
  GELOGI("==========================scalable_allocator begin success");

  // step2 malloc 5 pa
  /*
   *
   *          ┌────────────────────────────────┐ ┌────┐ ┌──────────────────┐
   *  va      │ span2 occupy 5 pa              │ │occu│ │span1 free        │
   *          └──▲─────▲──────▲───────▲─────▲──┘ └─▲──┘ └──▲─────▲──────▲──┘
   *             │     │      │       │     └──────┼───────┼─────┼──────┼
   *             │     │      │       └────────────┼───────┼─────┼      │
   *             │     │      └────────────────────┼───────┼     │      │
   *             │     └────────────────────┐      │       │     │      │
   *             └───────────────────▲      │      │       │     │      │
   *                                 │      │      │       │     │      │
   *                                 │      │      │       │     │      │
   *                                 │      │      │       │     │      │
   *                                 │      │      │       │     │      │
   *                                 │      │      │       │     │      │
   *                               ┌─┼──┐ ┌─┼──┐ ┌─┼──┐ ┌──┼─┐ ┌─┼──┐ ┌─┼──┐
   *   pa                          │ 5  │ │ 4  │ │ 3  │ │  2 │ │ 1  │ │ 0  │
   *                               └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
   */
  GELOGI("==========================span2 malloc 40M begin");
  auto span2 = caching_allocator->Malloc(alloc_size * 5); // MallocPhysical:get from pool pa_index:0.
  GELOGI("==========================span2 malloc 40M success");
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), alloc_size * 6); // span2 + occupy1

  // step3 malloc new_span1, will reuse span1
  /* before fix (get from pool not do physical_memory.ref_count++;)
   *
   *       ┌────────────────────────────────┐ ┌────┐ ┌──────────────────┐
   *  va   │ span2 free                     │ │occu│ │new span1         │
   *       └──▲─────▲──────▲───────▲─────▲──┘ └─▲──┘ └──▲─────▲──────▲──┘
   *          │     │      │       │     └──────┼───────┼─────┼──────┼
   *          │     │      │       └────────────┼───────┼─────┼      │
   *          │     │      └────────────────────┼───────┼     │      │
   *          │     └────────────────────┐      │       │     │      │
   *          └───────────────────▲      │      │       │     │      │
   *                              │      │      │       │     │      │
   *                              │      │      │       │     │      │
   *                              │      │      │       │     │      │
   *                              │      │      │       │     │      │
   *                              │      │      │       │     │      │
   *                            ┌─┼──┐ ┌─┼──┐ ┌─┼──┐ ┌──┼─┐ ┌─┼──┐ ┌─┼──┐
   *   pa                       │ 5  │ │ 4  │ │ 3  │ │  2 │ │ 1  │ │ 0  │
   *                            └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
   *
   *  after fix
   *
   *                ┌────────────────────────────────┐   ┌────┐   ┌──────────────────┐
   *    va          │span2 ccupy 5 pa                │   │    │   │                  │
   *                │                                │   │occu│   │new span1         │
   *                └─▲──────▲───────▲─────▲──────▲──┘   └─▲──┘   └─▲──────▲─────▲───┘
   *                  │      │       │     │      │        │        │      │     │
   *                  │      │       │     │      │        │        │      │     │
   *                  │      │       │     │      │        │        │      │     │
   *                ┌─┼──┐ ┌─┼──┐ ┌──┼─┐ ┌─┼──┐ ┌─┼──┐   ┌─┼──┐   ┌─┼──┐ ┌─┼──┐ ┌┴───┐
   *     pa         │ 5  │ │ 4  │ │  2 │ │ 1  │ │ 0  │   │ 3  │   │ 8  │ │ 7  │ │ 6  │
   *                └────┘ └────┘ └────┘ └────┘ └────┘   └────┘   └────┘ └────┘ └────┘
   */
  GELOGI("==========================new_span1 malloc 24M begin");
  auto new_span1 = caching_allocator->Malloc(alloc_size * 3); // MallocPhysical:reuse self pa_index:0.
  GELOGI("==========================new_span1 malloc 24M success");
  auto physical_memorys = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_;
  // key checkpoints
  ASSERT_EQ(physical_memorys.size(), 9); // before fix, size is 6

  // release
  span2->Free();
  new_span1->Free();
  occupy1->Free();
  scalable_allocator->Recycle();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_add_page_record_failed) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  // step1 malloc 3 pa, free
  /*
   *     ┌──────────────────┐
   * va  │span1 free        │
   *     └──▲─────▲──────▲──┘
   *        │     │      │
   *        │     │      │
   *     ┌──┼─┐ ┌─┼──┐ ┌─┼──┐
   * pa  │  2 │ │ 1  │ │ 0  │
   *     └────┘ └────┘ └────┘
   */
  GELOGI("==========================span1 malloc 24M begin");
  auto span1 = caching_allocator->Malloc(alloc_size * 3);
  GELOGI("==========================span1 malloc 24M success");
  GELOGI("==========================span1 free 24M begin");
  span1->Free();
  GELOGI("==========================span1 free 24M success");

  // step2 Injection error
  auto &physical_allocator = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator();
  ge::PageRecord error_page_record{(const uint8_t *)0x123, 0U, alloc_size, (const uint8_t *)0x123, alloc_size};
  physical_allocator.AddPageRecord(0, error_page_record);

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  auto new_span1 = caching_allocator->Malloc(alloc_size * 3);
  ASSERT_EQ(new_span1, nullptr);
  runtime_stub.GetSlogStub().FindErrorLogEndsWith("ProcPageRecord: ErrorNo: 4294967295(failed) virtual and physical page mapping check failed");

  // release
  scalable_allocator->Recycle();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_ProcPageRecordByPaList_failed) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);

  auto span3 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span3, nullptr);
  // va |3333|2222|1111|

  span3->Free();
  span1->Free();
  // va |----|2222|----|

  // Injection error
  auto &physical_allocator = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator();
  ge::PageRecord error_page_record{(const uint8_t *)0x123, 0U, alloc_size, (const uint8_t *)0x123, alloc_size};
  physical_allocator.AddPageRecord(0, error_page_record);

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  ASSERT_EQ(span4, nullptr);
  runtime_stub.GetSlogStub().FindErrorLogEndsWith("ProcPageRecordByPaList: ErrorNo: 4294967295(failed) virtual and physical page mapping check failed");

  scalable_allocator->Recycle();
  span2->Free();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_ProcPageRecordByPaList_SeccondMalloc_failed) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);

  auto span3 = caching_allocator->Malloc(alloc_size);
  ASSERT_NE(span3, nullptr);
  // va |3333|2222|1111|

  span3->Free();
  span1->Free();
  // va |----|2222|----|

  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  ASSERT_NE(span4, nullptr);
  span4->Free();

  // Injection error
  auto &physical_allocator = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator();
  ge::PageRecord error_page_record{(const uint8_t *)0x123, 0U, alloc_size, (const uint8_t *)0x123, alloc_size};
  physical_allocator.AddPageRecord(0, error_page_record);

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  span4 = caching_allocator->Malloc(alloc_size * 2U);
  ASSERT_EQ(span4, nullptr);
  runtime_stub.GetSlogStub().FindErrorLogEndsWith("ProcPageRecordByPaList: ErrorNo: 4294967295(failed) virtual and physical page mapping check failed");

  scalable_allocator->Recycle();
  span2->Free();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_physical_malloc_fail) {
  class MockRuntime : public ge::RuntimeStub {
   public:
    rtError_t rtMallocPhysical(rtDrvMemHandle* handle, size_t size, rtDrvMemProp_t* prop, uint64_t flags) {
      static size_t cnt = 0U;
      ++cnt;
      if (cnt >= 3U) {
        return -1;
      }
      *handle = (rtDrvMemHandle) new uint8_t[8];
      return RT_ERROR_NONE;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|
  ASSERT_EQ(span3, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);
  span2->Free();
  span1->Free();

  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_physical_memory_free) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|
  ASSERT_NE(span3, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span2->Free();
  span1->Free();
  span3->Free();
  auto &phycal_allocator = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator();
  caching_allocator.reset(nullptr);
  ASSERT_EQ(phycal_allocator.physical_memorys_.size(), 0U);

  auto caching_allocator1 = memory::CachingMemAllocator::GetAllocator();
  auto span11 = caching_allocator1->Malloc(alloc_size);
  span11->Free();
  caching_allocator1.reset(nullptr);
  ASSERT_EQ(phycal_allocator.physical_memorys_.size(), 0U);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, expandable_memory_allocator_free_after_recycle) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "3";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 8U * 1024U * 1024U);
  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 16U * 1024U * 1024U);

  auto span3 = caching_allocator->Malloc(alloc_size);
  // va |3333|2222|1111|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span3->Free();
  span1->Free();
  // va |----|2222|----|

  auto span4 = caching_allocator->Malloc(alloc_size * 2U);
  // new_va |4444|4444|
  // va |4444|2222|4444|

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 24U * 1024U * 1024U);
  span2->Free();
  // new_va |4444|4444|
  // va |4444|----|4444|

  scalable_allocator->Recycle();
  span4->Free();

  span1 = caching_allocator->Malloc(alloc_size);
  span2 = caching_allocator->Malloc(alloc_size);
  span1->Free();
  span4 = caching_allocator->Malloc(alloc_size * 2U);
  scalable_allocator->Recycle();
  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), alloc_size * 3U);
  span2->Free();
  span4->Free();
  scalable_allocator->Recycle();

  ASSERT_EQ(scalable_allocator->device_allocator_.GetOccupiedSize(), 0U);
  auto physical_memory_size = scalable_allocator->device_allocator_.GetExpandableAllocator()
    .GetPhysicalMemoryAllocator().physical_memorys_.size();
  auto caching_allocator1 = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator1 = caching_allocator1->GetScalableAllocator();
  auto span11 = caching_allocator1->Malloc(alloc_size);
  auto physical_memory_size1 = scalable_allocator1->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size, physical_memory_size1);
  auto span12 = caching_allocator1->Malloc(alloc_size * 2U);
  auto physical_memory_size2 = scalable_allocator1->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size, physical_memory_size2);
  ASSERT_EQ(scalable_allocator1->device_allocator_.GetOccupiedSize(), alloc_size * 3U);
  span11->Free();
  span12->Free();
  scalable_allocator1->Recycle();
  ASSERT_EQ(scalable_allocator1->device_allocator_.GetOccupiedSize(), 0U);
  ASSERT_EQ(scalable_allocator->GetStatics().empty(), false);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  scalable_allocator1->Finalize(false);
  caching_allocator1.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, static_and_dynamic_memory_full_reuse) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "4";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  auto mem_allocator = ge::SessionMemAllocator<ge::ActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  ge::LogicalMemorys logical_memorys;
  logical_memorys.emplace_back(0, 2U * 1024U * 1024U);
  std::vector<std::pair<uint8_t *, size_t>> mem_size;
  auto temp_mem_base = mem_allocator->MallocMemory("test", logical_memorys, mem_size, 0);
  ASSERT_NE(temp_mem_base, nullptr);
  mem_allocator->Recycle(mem_size);
  logical_memorys.clear();
  logical_memorys.emplace_back(0, 2U * 1024U * 1024U);
  logical_memorys.emplace_back(2U * 1024U * 1024U, 14 * 1024U * 1024U);
  mem_size.clear();
  temp_mem_base = mem_allocator->MallocMemory("test", logical_memorys, mem_size, 0);
  ASSERT_NE(temp_mem_base, nullptr);
  mem_allocator->Recycle(mem_size);

  auto physical_memory_size1 = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size1, 2);
  mem_allocator->Recycle(mem_size);

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  auto span2 = caching_allocator->Malloc(alloc_size);
  // va |2222|1111|
  ASSERT_NE(span2, nullptr);
  auto physical_memory_size2 = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size1, physical_memory_size2);
  span1->Free();
  span2->Free();
  scalable_allocator->Recycle();
  ASSERT_EQ(mem_allocator->MallocPhysicalMemory("test", mem_size), ge::SUCCESS);
  auto physical_memory_size3 = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(physical_memory_size2, physical_memory_size3);

  mem_allocator->FreeMemory(0);
  mem_allocator->FreeMemory(0);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, static_and_dynamic_memory_no_reuse) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "2";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  auto mem_allocator = ge::SessionMemAllocator<ge::ActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  ge::LogicalMemorys logical_memorys;
  logical_memorys.emplace_back(0, alloc_size * 2);
  std::vector<std::pair<uint8_t *, size_t>> mem_size;
  auto temp_mem_base = mem_allocator->MallocMemory("test", logical_memorys, mem_size, 0);
  ASSERT_NE(temp_mem_base, nullptr);

  mem_allocator->Recycle(mem_size);

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);

  ASSERT_EQ(mem_allocator->IsSupportExpandableMemoryFull(), false);
  span1->Free();
  scalable_allocator->Recycle();

  mem_allocator->MallocPhysicalMemory("test", mem_size);
  ASSERT_EQ(mem_allocator->IsSupportExpandableMemoryFull(), false);

  mem_allocator->FreeMemory(0);
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(ScaleAllocatorTest, static_and_dynamic_memory_part_reuse) {
  std::map<std::string, std::string> graph_options;
  graph_options[ge::STATIC_MEMORY_POLICY] = "4";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
  size_t alloc_size = 8U * 1024U * 1024U;
  auto caching_allocator = memory::CachingMemAllocator::GetAllocator();
  auto scalable_allocator = caching_allocator->GetScalableAllocator();
  auto mem_allocator = ge::SessionMemAllocator<ge::ActiveMemoryAllocator>::Instance().GetMemAllocator(1, 0);
  // open va pa check
  caching_allocator->GetScalableAllocator()->device_allocator_.GetExpandableAllocator().log_level_ = 1;

  ge::LogicalMemorys logical_memorys;
  logical_memorys.emplace_back(0, 2 * 1024 * 1024);
  std::vector<std::pair<uint8_t *, size_t>> mem_size;
  auto temp_mem_base = mem_allocator->MallocMemory("test", logical_memorys, mem_size, 0);
  ASSERT_NE(temp_mem_base, nullptr);

  mem_allocator->Recycle(mem_size);

  auto span1 = caching_allocator->Malloc(alloc_size);
  // va |1111|
  ASSERT_NE(span1, nullptr);
  logical_memorys.clear();
  logical_memorys.emplace_back(0, 2 * alloc_size);
  temp_mem_base = mem_allocator->MallocMemory("test", logical_memorys, mem_size, 0);
  auto physical_memory_size = scalable_allocator->device_allocator_.GetExpandableAllocator()
      .GetPhysicalMemoryAllocator().physical_memorys_.size();
  ASSERT_EQ(4, physical_memory_size);

  span1->Free();
  scalable_allocator->Recycle();
  mem_allocator->FreeMemory(0);
  span1->Free();
  scalable_allocator->Finalize(false);
  caching_allocator.reset(nullptr);
  graph_options[ge::STATIC_MEMORY_POLICY] = "0";
  ge::GetThreadLocalContext().SetGraphOption(graph_options);
}
/*
 * 1. stub rtMalloc allways return failed
 * 2. crate 10 threads, each thread create a CachingMemAllocator, and malloc 1000 times, each malloc size is 1024
 */
TEST_F(ScaleAllocatorTest, multithread_allocator_recycle) {
  gert::memory::CachingMemAllocator allocator(0);
  const auto old_object_size = allocator.all_caching_mem_allocators_.size();
  auto stub = std::make_shared<RuntimeMock>();
  ge::RuntimeStub::SetInstance(stub);
  const int num_threads = 10;
  std::vector<std::thread> threads;

  // Create 10 threads
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(ThreadFunction);
  }

  // Wait for all threads to finish
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(allocator.all_caching_mem_allocators_.size() == old_object_size);
  ge::RuntimeStub::Reset();
}