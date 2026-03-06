/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_block_manager.h"
#include "common/checker.h"
#include "ge_common/debug/log.h"
#include "runtime/rt.h"
#include "graph/manager/mem_manager.h"

namespace ge {
namespace {
constexpr size_t kAlignSize = 512;
}
void *MemoryBlockManager::Malloc(const std::string &purpose, const size_t size) {
  const auto aligned_size = (size + kAlignSize - 1U) / kAlignSize * kAlignSize;
  void *ptr = FindFreeMem(aligned_size);
  if (ptr != nullptr) {
    return ptr;
  }
  const auto block_size = (aligned_size + block_size_ - 1U) / block_size_ * block_size_;
  const auto rt_ret = rtMalloc(&ptr, block_size, mem_type_, GE_MODULE_NAME_U16);
  GE_ASSERT_TRUE((rt_ret == RT_ERROR_NONE) && (ptr != nullptr),
                 "call rtMalloc failed, size: %zu, memory type: %u, rt_ret: %d",
                 block_size, mem_type_, rt_ret);
  GE_ASSERT(rtMemset(ptr, block_size, 0U, block_size) == RT_ERROR_NONE);
  mem_blocks_.emplace_back(RtMemBlock{ptr, block_size, aligned_size});
  int32_t device_id = 0;
  (void) rtGetDevice(&device_id);
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, ge::ToMallocMemInfo(purpose, ptr, device_id, GE_MODULE_NAME_U16).c_str(),
                          block_size);
  GELOGI("malloc memory success, ptr: %p, size: %zu, aligned size: %zu, block_size: %zu, memory type: %u",
         ptr, size, aligned_size, block_size, mem_type_);
  return ptr;
}

void *MemoryBlockManager::FindFreeMem(const size_t aligned_size) {
  for (auto &block : mem_blocks_) {
    if (block.size >= aligned_size + block.used_size) {
      void *ptr = ValueToPtr(PtrToValue(block.addr) + block.used_size);
      block.used_size += aligned_size;
      GELOGI("find free memory, ptr: %p, aligned_size: %zu, block info[base: %p, size: %zu, used_size: %zu].",
             ptr, aligned_size, block.addr, block.size, block.used_size);
      return ptr;
    }
  }
  return nullptr;
}

void MemoryBlockManager::Release() {
  for (auto &block : mem_blocks_) {
    (void)rtFree(block.addr);
    GELOGI("free memory success, ptr: %p, size: %zu, memory type: %u", block.addr, block.size, mem_type_);
  }
  mem_blocks_.clear();
}
} // namespace ge