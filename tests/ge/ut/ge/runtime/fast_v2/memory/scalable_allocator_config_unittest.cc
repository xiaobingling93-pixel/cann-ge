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

#include "macro_utils/dt_public_scope.h"

#include "kernel/memory/allocator/scalable_allocator.h"
#include "depends/runtime/src/runtime_stub.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "graph/ge_local_context.h"

#include "macro_utils/dt_public_unscope.h"

using namespace gert;

namespace {
class MockRuntime : public ge::RuntimeStub {
 public:
  rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
    *free = 64UL * 1024UL * 1024UL;
    *total = 56UL * 1024UL * 1024UL * 1024UL;
    return RT_ERROR_NONE;
  }
};
}

 struct ScaleAllocatorConfigTest : public memory::MemSynchronizer, public testing::Test {
  void SetUp() {
    auto mock_runtime = std::make_shared<MockRuntime>();
    ge::RuntimeStub::SetInstance(mock_runtime);
    config.reset(new ScalableConfig());
    memory::RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
    memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
    config->page_idem_num = page_idem_num;
    config->span_layer_lift_max = span_layer_lift_max;
    config->unsplitable_size_threshold = unsplitable_size_threshold;
    config->uncacheable_size_threshold = uncacheable_size_threshold;
    //allocator.reset(new ScalableAllocator{span_allocator_, device_allocator, *config});
    caching_allocator.reset(new memory::CachingMemAllocator{0, kOnDeviceHbm, *config});
    ge::RuntimeStub::Reset();
  }

  void TearDown() {
  }

  ge::Status Synchronize() const { return ge::SUCCESS; }

  void Recycle() {};

 protected:
  size_t page_idem_num = 10;
  size_t span_layer_lift_max = 2;
  MemSize unsplitable_size_threshold = 1_GB;
  MemSize uncacheable_size_threshold = 2_GB - 1_MB;
  MemSize pageSize{(MemSize) 1 << page_idem_num};

  size_t unsplitableLayerId{SpanLayerId_GetIdFromSize(unsplitable_size_threshold, page_idem_num)};
  size_t unCacheableLayerId{SpanLayerId_GetIdFromSize(uncacheable_size_threshold, page_idem_num)};

 protected:
  std::unique_ptr<ScalableConfig> config;
  const uint32_t device_id{0};
  memory::RtsCachingMemAllocator rts_caching_mem_allocator{0, RT_MEMORY_HBM};
  RtMemAllocator device_allocator{rts_caching_mem_allocator, 0U};
  SpanAllocatorImp span_allocator_;
  std::unique_ptr<memory::CachingMemAllocator> caching_allocator;
};

TEST_F(ScaleAllocatorConfigTest, should_alloc_and_free_0_size) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 0);
  span->Free();
  ASSERT_EQ(pageSize, caching_allocator->GetScalableAllocator()->GetIdleMemSizeOfLayer(1));
}

TEST_F(ScaleAllocatorConfigTest, should_not_lift_to_unsplitable_layer) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, unsplitable_size_threshold);
  span->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(unsplitableLayerId));

  span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, pageSize);

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(unsplitableLayerId));

  span->Free();

  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(unsplitableLayerId));
}

TEST_F(ScaleAllocatorConfigTest, should_not_cache_the_uncacheable_size) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, uncacheable_size_threshold);
  span->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
}

TEST_F(ScaleAllocatorConfigTest, test_set_memory_pool_threshold) {
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  ScalableConfig default_cfg;
  constexpr const char *kOptionDisableMemoryPoolThreshold = "ge.experiment.memory_pool_threshold";
  const auto back_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = back_options;
  new_options[kOptionDisableMemoryPoolThreshold] = "3";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  ScalableConfig cfg1;
  EXPECT_EQ(cfg1.page_mem_size_total_threshold, 3 * MEM_SIZE_GB);

  new_options[kOptionDisableMemoryPoolThreshold] = "abc";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  ScalableConfig cfg2;
  EXPECT_EQ(cfg2.page_mem_size_total_threshold, default_cfg.page_mem_size_total_threshold);

  new_options[kOptionDisableMemoryPoolThreshold] = "0";
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  ScalableConfig cfg3;
  EXPECT_EQ(cfg3.page_mem_size_total_threshold, default_cfg.page_mem_size_total_threshold);

  ge::RuntimeStub::Reset();
  ge::GetThreadLocalContext().SetGlobalOption(back_options);
}

struct ScaleAllocatorFixConfigTest : public testing::Test {
  void SetUp() {
    config.span_layer_lift_max = 0;
    caching_allocator = new memory::CachingMemAllocator{0, kOnDeviceHbm, config};
  }

  void TearDown() {
    delete caching_allocator;
  }

 protected:
  ScalableConfig config;
  const uint32_t device_id{0};
  memory::RtsCachingMemAllocator rts_caching_mem_allocator{0, RT_MEMORY_HBM};
  RtMemAllocator device_allocator{rts_caching_mem_allocator, 0U};
  SpanAllocatorImp span_allocator_;
  memory::CachingMemAllocator *caching_allocator;
};

TEST_F(ScaleAllocatorFixConfigTest, should_not_lift_when_limit_lift_threhold_to_0) {
  auto span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 1_MB);
  span->Free();

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(16));

  span = caching_allocator->GetScalableAllocator()->Alloc(*caching_allocator, 1_KB);

  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(16));

  span->Free();
  ASSERT_EQ(2, caching_allocator->GetScalableAllocator()->GetIdleSpanCount());
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(1));
  ASSERT_EQ(1, caching_allocator->GetScalableAllocator()->GetIdleSpanCountOfLayer(16));
}
