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
#include "graph/load/model_manager/sink_only_allocator.h"
#include <kernel/memory/single_stream_l2_allocator.h>
#include "graph/load/model_manager/memory_block_manager.h"

namespace gert::memory {
 class SinkOnlyAllocatorUT : public testing::Test {};


 TEST_F(SinkOnlyAllocatorUT, SinkOnlyAllocator_Malloc) {
   auto sink_alloc_  = new SinkOnlyAllocator();
   auto caching_mem_allocator = std::make_shared<ge::MemoryBlockManager>(0);
   sink_alloc_->SetAllocator(caching_mem_allocator);
   auto mem_block = sink_alloc_->Malloc(1024);
   ASSERT_NE(mem_block, nullptr);
   void* addr = mem_block->GetAddr();
   if (addr != nullptr) {
     caching_mem_allocator->Release();
   }
   delete sink_alloc_;
   sink_alloc_ = nullptr;
 }

 TEST_F(SinkOnlyAllocatorUT, SinkOnlyAllocator_Malloc_with_error) {
   auto sink_alloc_  = new gert::memory::SinkOnlyAllocator();
   auto caching_mem_allocator = std::make_shared<ge::MemoryBlockManager>(0);
   sink_alloc_->SetAllocator(caching_mem_allocator);
   auto malloc = sink_alloc_->Malloc(0);
   ASSERT_EQ(malloc, nullptr);
   delete sink_alloc_;
   sink_alloc_ = nullptr;
 }
}  // namespace gert::memory

