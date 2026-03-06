/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "l2_mem_pool.h"
#include "caching_mem_allocator.h"
#include "common/checker.h"
#include "rts_caching_mem_allocator.h"
#include "utils/utils.h"

namespace gert {
namespace memory {
L2MemPool::L2MemPool(ge::Allocator *allocator, rtStream_t stream, TypedContinuousVector<L2MemPool *> *all_l2_mem_pool)
    : first_level_pool_(allocator, stream, all_l2_mem_pool),
      memory_pool_(new ScalableAllocator(span_allocator_, first_level_pool_, ScalableConfig())),
      stream_(stream) {
  GELOGI("create l2 allocator:%s", memory_pool_->GetId().c_str());
}

L2MemPool::~L2MemPool() {
  (void)Finalize(true);
}

ge::MemBlock *L2MemPool::Malloc(size_t size) {
  auto addr = memory_pool_->Alloc(*this, size);
  if (addr != nullptr) {
    return addr;
  }
  GELOGE(ge::MEMALLOC_FAILED,
         "stream %p 's L2 allocator failed to apply for memory. We will try to free memory from memory pool, the above "
         "error log can be ignored. Try to free cached memory...",
         stream_);
  GE_ASSERT_SUCCESS(Synchronize());
  Recycle();
  addr = memory_pool_->Alloc(*this, size);
  return addr;
}

void L2MemPool::Free(ge::MemBlock *block) {
  memory_pool_->Free(reinterpret_cast<PageSpan *>(block));
}

void L2MemPool::Recycle() {
  memory_pool_->Recycle();
}

ge::Status L2MemPool::Synchronize() const {
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(stream_));
  return ge::SUCCESS;
}

ge::Status L2MemPool::Finalize(bool no_log) {
  return memory_pool_->Finalize(no_log);
}

rtStream_t L2MemPool::GetStream() {
  return stream_;
}

void L2MemPool::SetStream(rtStream_t stream) {
  stream_ = stream;
  first_level_pool_.SetStream(stream);
}

ge::MemBlock *L2MemPool::MoveL2ToL1(ge::MemBlock *block) {
  GE_ASSERT_NOTNULL(block);
  auto l1_block = memory_pool_->ConvertToRootBlock(block);
  if (l1_block == nullptr) {
    auto size = block->GetSize();
    l1_block = first_level_pool_.Alloc(size);
    GE_ASSERT_NOTNULL(l1_block);
    GE_ASSERT_RT_OK(aclrtMemcpyAsync(l1_block->GetAddr(), size, block->GetAddr(), block->GetSize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));
    GELOGI("l2 block %p addr %p is splited, it has been moved to L1 block %p addr %p", block, block->GetAddr(),
           l1_block, l1_block->GetAddr());
    block->Free();
  }
  return l1_block;
}

BlockAddr MultiStreamL1Allocator::Alloc(const MemSize size) {
  auto block = l1_allocator_->Malloc(size);
  if ((block == nullptr) && (all_l2_mem_pool_ != nullptr) && !is_rt2_multi_thread_) {
    GELOGI("malloc memory failed, try to free l2 mem pool and malloc again");
    for (size_t i = 0U; i < all_l2_mem_pool_->GetSize(); ++i) {
      auto l2_mem_pool = all_l2_mem_pool_->MutableData()[i];
      GE_ASSERT_NOTNULL(l2_mem_pool);
      GE_ASSERT_SUCCESS(l2_mem_pool->Synchronize());
      l2_mem_pool->Recycle();
      block = l1_allocator_->Malloc(size);
      if (block == nullptr) {
        continue;
      }
      break;
    }
  }
  GE_ASSERT_NOTNULL(block,
                    "Failed to expand memory for l2 allocator, stream %p, size %zu, is enable rt2 multi thread: %zu",
                    l2_stream_, size, static_cast<size_t>(is_rt2_multi_thread_));
  GE_ASSERT_NOTNULL(block->GetAddr());
  GELOGI("[MEM]Expand memory pool at stream %p, address %p, size %zu, block %p. allocator addr %p", l2_stream_,
         block->GetAddr(), size, block, l1_allocator_);
  return block;
}
bool MultiStreamL1Allocator::Free(ge::MemBlock *const block) {
  if (block != nullptr) {
    GELOGI("[MEM]Shrink memory pool at stream %p, address %p, block %p", l2_stream_, block->GetAddr(), block);
    block->Free();
  }
  return true;
}
DeviceId MultiStreamL1Allocator::GetDeviceId() const {
  return -1;
}
}  // namespace memory
}  // namespace gert
