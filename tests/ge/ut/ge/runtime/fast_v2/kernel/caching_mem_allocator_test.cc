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
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "runtime/mem.h"
#include <cmath>
#include "stub/gert_runtime_stub.h"
#include "faker/multi_stream_allocator_faker.h"
#include "runtime/src/runtime_stub.h"

namespace gert {
namespace memory {
class MultiStreamL2AllocatorsFaker : public MultiStreamAllocatorFaker {
 public:
  MultiStreamL2AllocatorsFaker() : MultiStreamAllocatorFaker() {
    L1Allocator(std::make_shared<CachingMemAllocator>(0, RT_MEMORY_HBM));
  }
};

struct CacheMemoryAllocatorTest : public testing::Test {
 protected:
  void SetUp() {
    RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
    RtsCachingMemAllocator::device_id_to_allocators_.clear();
  }
  void TearDown() {
    RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
    RtsCachingMemAllocator::device_id_to_allocators_.clear();
  }
};

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_fail_when_size_bigger_then_kMaxHbmMemorySize) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  //  auto allocator = CachingMemAllocator::GetExternalL1Allocator();
  ASSERT_EQ(nullptr, allocator.Malloc(1024UL * 1024UL * 1024UL * 1024UL));
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(1024UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_NE(ge::GRAPH_SUCCESS, allocator.Finalize());
  mem_block->Free();

  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_3145728_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(3145728UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), 3145728UL);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

const int64_t MByteSize = 1024UL * 1024UL;

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_1M_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(MByteSize - 64);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), MByteSize - 64);
  mem_block->Free();
  allocator.Finalize();
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_1M_without_padding_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(MByteSize);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), MByteSize);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_2M_without_padding_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(2 * MByteSize);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), 2 * MByteSize);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_3M_without_padding_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(3 * MByteSize);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), 3 * MByteSize);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_4M_without_padding_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(4 * MByteSize + 1);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), 4 * MByteSize + 1);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}
// constexpr size_t GByteSize = 1024UL * 1024UL * 1024UL;
// TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_1G_without_padding_sucess){
//   CachingMemAllocator allocator(0, RT_MEMORY_HBM);
//   auto mem_block = allocator.Malloc(GByteSize);
//   ASSERT_NE(mem_block->GetAddr(), nullptr);
//   ASSERT_GE(mem_block->GetSize(), GByteSize);
//   mem_block->Free();
//   ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
// }

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_size_4816896UL_sucess) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(4816896UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), 4816896UL);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(CacheMemoryAllocatorTest, test_get_allocator_failed_when_rtGetDevice_failed) {
  RTS_STUB_RETURN_VALUE(rtGetDevice, rtError_t, 0x01);
  ASSERT_EQ(CachingMemAllocator::GetAllocator(), nullptr);
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_free_then_alloc_again) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(4816896UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  ASSERT_GE(mem_block->GetSize(), 4816896UL);
  mem_block->Free();
  ASSERT_EQ(allocator.Malloc(4816896UL), mem_block);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_hbm_first_success_second_failed) {
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(4816896UL);
  ASSERT_NE(mem_block, nullptr);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
  RtsCachingMemAllocator::GetAllocator(0, RT_MEMORY_HBM)->Recycle();
  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      return -1;
    }
    rtError_t rtFree(void *dev_ptr) {
      return -1;
    }
  };
  GertRuntimeStub stub(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime()));
  mem_block = allocator.Malloc(4816896UL);
  ASSERT_EQ(mem_block, nullptr);
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_try_recycle_then_malloc_when_mem_cannot_alloc) {
  static uint64_t total_size = 0;
  static uint8_t malloc_count = 0;
  class MockRuntime : public ge::RuntimeStub {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      malloc_count++;
      total_size += size;
      if (total_size > 32 * 1024UL * 1024UL * 1024UL) {
        total_size = (malloc_count == 2) ? 0 : total_size;
        *dev_ptr = nullptr;
        return -1;
      }
      *dev_ptr = new uint8_t[1];
      return RT_ERROR_NONE;
    }
    rtError_t rtFree(void *dev_ptr) {
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 56UL * 1024UL * 1024UL;
      *total = 56UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }
  };

  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  allocator.SetStream((void *)1);
  auto mem_block = allocator.Malloc(20 * 1024UL * 1024UL * 1024UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  auto mem_block1 = allocator.Malloc(21 * 1024UL * 1024UL * 1024UL);
  ASSERT_NE(mem_block1->GetAddr(), nullptr);
  mem_block->Free();
  mem_block1->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
  ge::RuntimeStub::Reset();
}

TEST_F(CacheMemoryAllocatorTest, test_allocate_mem_block_try_recycle_other_then_malloc_when_mem_cannot_alloc) {
  static uint64_t total_size = 0;
  static uint8_t malloc_count = 0;
  class MockRuntime : public ge::RuntimeStub {
    rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) {
      malloc_count++;
      total_size += size;
      if (total_size > 32 * 1024UL * 1024UL * 1024UL) {
        total_size = (malloc_count == 3) ? 0 : total_size;
        *dev_ptr = nullptr;
        return -1;
      }
      *dev_ptr = new uint8_t[1];
      return RT_ERROR_NONE;
    }
    rtError_t rtFree(void *dev_ptr) {
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }
    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) override {
      *free = 56UL * 1024UL * 1024UL;
      *total = 56UL * 1024UL * 1024UL * 1024UL;
      return RT_ERROR_NONE;
    }
  };
  auto mock_runtime = std::make_shared<MockRuntime>();
  ge::RuntimeStub::SetInstance(mock_runtime);
  CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  CachingMemAllocator allocator1(0, RT_MEMORY_HBM);
  auto mem_block = allocator.Malloc(20 * 1024UL * 1024UL * 1024UL);
  ASSERT_NE(mem_block->GetAddr(), nullptr);
  auto mem_block1 = allocator1.Malloc(21 * 1024UL * 1024UL * 1024UL);
  ASSERT_NE(mem_block1->GetAddr(), nullptr);
  mem_block->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator.Finalize());
  mem_block1->Free();
  ASSERT_EQ(ge::GRAPH_SUCCESS, allocator1.Finalize());
  ge::RuntimeStub::Reset();
}
}  // namespace memory
}  // namespace gert
