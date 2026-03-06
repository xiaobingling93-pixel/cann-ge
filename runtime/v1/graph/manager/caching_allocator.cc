/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/caching_allocator.h"

#include <set>
#include <vector>
#include <string>

#include "graph/manager/mem_manager.h"
#include "graph/def_types.h"
#include "common/debug/log.h"
#include "common/math/math_util.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
// The memory size calculated is 64 bits larger than the memory pool gear after byte alignment and byte filling
// adjust gear of memory pool by adding addition 512 offset
constexpr size_t bin_ranges[kNumBins] = {kRoundBlockSize * kKByteSize,
                                         kBinSizeUnit8 * kMByteSize + kRoundBlockSize,
                                         kBinSizeUnit32 * kMByteSize + kRoundBlockSize,
                                         kBinSizeUnit128 * kMByteSize + kRoundBlockSize,
                                         kBinSizeUnit256 * kMByteSize + kRoundBlockSize,
                                         kBinSizeUnit512 * kMByteSize + kRoundBlockSize,
                                         kGByteSize};

bool BlockComparator(const Block *const left, const Block *const right) {
  GE_CHECK_NOTNULL_EXEC(left, return false);
  GE_CHECK_NOTNULL_EXEC(right, return false);
  if (left->size != right->size) {
    return left->size < right->size;
  }
  return PtrToValue(left->ptr) < PtrToValue(right->ptr);
}

bool CanMergeBlock(const Block *const block) {
  if ((block == nullptr) || block->allocated || (!block->IsSplit())) {
    return false;
  }
  return true;
}

size_t GetBinIndex(const size_t size) {
  size_t index = 0U;
  for (const size_t range : bin_ranges) {
    if (size <= range) {
      break;
    }
    index++;
  }
  if (index > (kNumBins - 1U)) {
    index = kNumBins - 1U;
  }
  return index;
}

size_t GetAllocationSize(const size_t size) {
  const size_t index = GetBinIndex(size);
  if (bin_ranges[index] >= size) {
    return bin_ranges[index];
  }
  if (CheckSizeTAddOverflow(size, kGByteSize) != SUCCESS) {
    return SIZE_MAX;
  }
  return static_cast<size_t>(kGByteSize * ((size + kGByteSize - 1U) / kGByteSize));
}

/// @ingroup ge_graph
/// @brief block size based on alignment
/// @param [in] original malloc size
/// @return allocation size
size_t GetBlockSize(const size_t size) {
  if (size == 0U) {
    return kRoundBlockSize;
  }
  if (CheckSizeTAddOverflow(size, kRoundBlockSize) != SUCCESS) {
    return SIZE_MAX;
  }
  return kRoundBlockSize * ((size + kRoundBlockSize - 1U) / kRoundBlockSize);
}

bool ShouldSplitBlock(const Block &block, const size_t size) {
  if (CheckDoubleMulOverflow(static_cast<float64_t>(block.size), kSplitThreshold) != SUCCESS) {
    return true;
  }
  return static_cast<float64_t>(size) <= (static_cast<float64_t>(block.size) * kSplitThreshold);
}

void IncreaseCount(std::map<size_t, size_t> &count, size_t size) {
  const auto it = count.find(size);
  if (it == count.end()) {
    (void)count.emplace(size, 1);
  } else  {
    if (CheckSizeTAddOverflow(it->second, 1) == SUCCESS) {
      it->second++;
    }
  }
}

void PrintCount(const std::map<size_t, size_t> &count, const std::string &name, const size_t total_size,
                const size_t total_count) {
  GELOGD("%6s total[size:%11zu count:%11zu].", name.c_str(), total_size, total_count);
  for (auto &it : count) {
    GELOGD("    |- block[size:%11zu count:%11zu].", it.first, it.second);
  }
}
}

CachingAllocator::CachingAllocator(const rtMemType_t memory_type)
    : memory_type_(memory_type) {
}

Status CachingAllocator::Initialize(const uint32_t device_id) {
  // when redo Initialize free old memory
  FreeBlocks();
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &bin : free_block_bins_) {
    if (bin != nullptr) {
      continue;
    }
    const auto bin_ptr = new (std::nothrow) BlockBin(&BlockComparator);
    if (bin_ptr == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "New BlockBin fail, device_id:%u", device_id);
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Alloc][BlockBin] failed, device_id:%u", device_id);
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
    bin = bin_ptr;
  }
  memory_allocator_ = &MemManager::Instance().MemInstance(memory_type_);
  if (memory_allocator_ == nullptr) {
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  called_malloc_counts_ = 0U;
  called_free_counts_ = 0U;
  return ge::SUCCESS;
}

void CachingAllocator::Finalize() {
  PrintStatics();
  FreeBlocks();
  FreeBlockBins();
}

uint8_t *CachingAllocator::Malloc(size_t size, uint8_t *const org_ptr, const uint32_t device_id) {
  GELOGI("Start malloc pool memory, size = %zu, device id = %u", size, device_id);
  if (CheckSizeTAddOverflow(called_malloc_counts_, 1) == SUCCESS) {
    called_malloc_counts_++;
  }
  size = GetBlockSize(size);
  uint8_t *ptr = nullptr;
  Block *block = FindFreeBlock(size, org_ptr, device_id);
  if (block == nullptr) {
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (TryExtendCache(size, device_id) == ge::SUCCESS) {
      block = FindFreeBlock(size, org_ptr, device_id);
      if (block != nullptr) {
        ptr = block->ptr;
      }
    }
  } else {
    ptr = block->ptr;
  }
  if (ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "FindFreeBlock fail, size:%zu, device_id:%u", size, device_id);
    GELOGE(FAILED, "[Check][Param] FindFreeBlock failed device id = %u, size= %zu", device_id, size);
  }
  return ptr;
}

Status CachingAllocator::Free(uint8_t *const memory_addr, const uint32_t device_id) {
  GELOGI("Free device id = %u", device_id);
  called_free_counts_++;
  if (memory_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param memory_addr is nullptr, device_id:%u, check invalid", device_id);
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid memory pointer, device_id:%u", device_id);
    return ge::PARAM_INVALID;
  }

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = allocated_blocks_.find(memory_addr);
  if (it == allocated_blocks_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Param ptr not allocated before, device_id:%u, check invalid", device_id);
    GELOGE(PARAM_INVALID, "[Check][Param] Param ptr not allocated before, device_id:%u", device_id);
    return ge::PARAM_INVALID;
  }
  Block *const block = it->second;
  (void)allocated_blocks_.erase(it);
  FreeBlock(block);
  return ge::SUCCESS;
}

void CachingAllocator::FreeBlock(Block *const block) const {
  if ((block == nullptr) || (!block->allocated) || (block->bin == nullptr)) {
    return;
  }
  GELOGI("Free block size = %zu", block->size);

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  block->allocated = false;
  auto &bin = *block->bin;
  const std::vector<Block *> merge_blocks {block->prev, block->next};
  for (Block *const merge_block : merge_blocks) {
    MergeBlocks(block, merge_block, bin);
  }
  (void)bin.insert(block);
}

void CachingAllocator::MergeBlocks(Block *const dst, Block *const src, BlockBin &bin) const {
  if ((!CanMergeBlock(src)) || (!CanMergeBlock(dst))) {
    return;
  }

  if (dst->prev == src) {
    dst->ptr = src->ptr;
    dst->prev = src->prev;
    if (dst->prev != nullptr) {
      dst->prev->next = dst;
    }
  } else {
    dst->next = src->next;
    if (dst->next != nullptr) {
      dst->next->prev = dst;
    }
  }

  if (CheckSizeTAddOverflow(dst->size, src->size) == SUCCESS) {
    dst->size += src->size;
  }
  (void)bin.erase(src);
  delete src;
}

BlockBin *CachingAllocator::GetBlockBin(const size_t size) const {
  const size_t index = GetBinIndex(size);
  return free_block_bins_[index];
}

Block *CachingAllocator::FindFreeBlock(const size_t size, uint8_t *const org_ptr, const uint32_t device_id) {
  Block key(device_id, size, org_ptr);
  BlockBin *const bin = GetBlockBin(size);
  if (bin == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "GetBlockBin fail, size:%zu, device_id:%u", size, device_id);
    GELOGE(ge::FAILED, "[Get][BlockBin] failed, size:%zu, device_id:%u", size, device_id);
    return nullptr;
  }
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = bin->lower_bound(&key);
  if (it != bin->end()) {
    Block *block = *it;
    (void)bin->erase(it);
    if (block != nullptr) {
      GELOGI("Find block size = %zu", block->size);
      if (ShouldSplitBlock(*block, size)) {
        block = SplitBlock(*block, size, *bin, device_id);
      }

      if (block->ptr != nullptr) {
        block->allocated = true;
        allocated_blocks_[block->ptr] = block;
        GELOGI("Malloc device id = %u, size= %zu", device_id, size);
      }
    }

    return block;
  }
  return nullptr;
}

Block *CachingAllocator::SplitBlock(Block &block, const size_t size, BlockBin &bin, const uint32_t device_id) const {
  // block has been checked, should not be nullptr
  Block *const remaining = &block;
  Block *const new_block = new (std::nothrow) Block(device_id, size, &bin, block.ptr);
  if (new_block == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New Block fail, size:%zu, device_id:%u", size, device_id);
    GELOGE(ge::FAILED, "[Alloc][Block] failed, size:%zu, device_id:%u", size, device_id);
    return remaining;
  }
  new_block->prev = remaining->prev;
  if (new_block->prev != nullptr) {
    new_block->prev->next = new_block;
  }
  new_block->next = remaining;
  remaining->prev = new_block;
  remaining->ptr = PtrAdd(remaining->ptr, remaining->size, size);
  if (CheckSizeTSubOverflow(remaining->size, size) == SUCCESS) {
    remaining->size -= size;
  } else {
    remaining->size = 0;
  }
  (void)bin.insert(remaining);
  return new_block;
}

Status CachingAllocator::TryExtendCache(const size_t size, const uint32_t device_id) {
  GELOGI("Try to extend cache. size = %zu, device id = %u", size, device_id);
  const auto memory_size = GetAllocationSize(size);
  const std::string purpose = "Memory for caching";
  auto memory_addr = memory_allocator_->MallocMemory(purpose, memory_size, device_id);
  if (memory_addr == nullptr) {
    if (bind_stream_) {
      GELOGE(ge::FAILED, "[Malloc][Memory] failed, no enough memory for size = %zu, device_id = %u", memory_size,
             device_id);
      PrintStatics(GeLogLevel::kError);
      return ge::FAILED;
    }
    // try to free caches and malloc again when malloc memory failed
    GELOGE(MEMALLOC_FAILED,
           "Failed to apply for memory. We will try to free memory from memory pool, the above error log can be "
           "ignored. Try to free cached memory...");
    memory_addr = memory_allocator_->MallocMemory(purpose, memory_size, device_id);
    if (memory_addr == nullptr) {
      GELOGE(ge::FAILED, "[Malloc][Memory] failed, no enough memory for size = %zu, device_id = %u", memory_size,
             device_id);
      PrintStatics(GeLogLevel::kError);
      return ge::FAILED;
    }
  }

  if (AddToBlockBin(memory_addr, memory_size, device_id) != ge::SUCCESS) {
    (void)memory_allocator_->FreeMemory(memory_addr);
    return ge::FAILED;
  }
  PrintStatics();
  return ge::SUCCESS;
}

Status CachingAllocator::AddToBlockBin(uint8_t *const ptr, const size_t size, const uint32_t device_id) {
  BlockBin *const bin = GetBlockBin(size);
  if (bin == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "GetBlockBin fail, size:%zu, device_id:%u", size, device_id);
    GELOGE(ge::FAILED, "[Get][BlockBin] failed, size:%zu, device_id:%u", size, device_id);
    return ge::FAILED;
  }
  Block *block = new (std::nothrow) Block(device_id, size, bin, nullptr);
  if (block == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New Block fail, size:%zu, device_id:%u", size, device_id);
    GELOGE(ge::FAILED, "[Alloc][Block] failed, size:%zu, device_id:%u", size, device_id);
    return ge::FAILED;
  }

  GELOGI("Block size = %zu", size);
  block->ptr = ptr;
  block->size = size;

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  IncreaseCount(malloced_memory_, block->size);
  (void)bin->insert(block);
  return ge::SUCCESS;
}

size_t CachingAllocator::FreeCachedBlocks() {
  GELOGI("Free cached blocks");
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  size_t free_cached_memory_size = 0U;
  for (const auto pool : free_block_bins_) {
    if (pool == nullptr) {
      continue;
    }
    auto it = pool->cbegin();
    while (it != pool->cend()) {
      const Block *const block = *it;
      // free block memory that has not been split
      if ((block != nullptr) && (block->ptr != nullptr) &&
          (block->prev == nullptr) && (block->next == nullptr) &&
          (memory_allocator_->FreeMemory(block->ptr) == ge::SUCCESS)) {
        const auto itcount = malloced_memory_.find(block->size);
        free_cached_memory_size += block->size;
        if (itcount != malloced_memory_.end()) {
          itcount->second--;
          if (itcount->second == 0U) {
            (void)malloced_memory_.erase(itcount);
          }
        }
        (void)pool->erase(it++);
        delete block;
        continue;
      }
      ++it;
    }
  }
  return free_cached_memory_size;
}

void CachingAllocator::FreeBlocks() {
  GELOGI("Free blocks.");
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  // free allocated blocks and put to cache
  for (auto &it : allocated_blocks_) {
    FreeBlock(it.second);
  }
  allocated_blocks_.clear();
  (void)FreeCachedBlocks();
}

void CachingAllocator::TryFreeBlocks() {
  GELOGI("Try free blocks.");
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  (void)FreeCachedBlocks();
  PrintStatics(GeLogLevel::kEvent);
}

Status CachingAllocator::FreeBlocksAfterSynchronize(rtStream_t const stream) {
  GELOGW("Stream synchronize and try free blocks! stream: %p.", stream);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  (void)FreeCachedBlocks();
  PrintStatics(GeLogLevel::kEvent);
  return SUCCESS;
}

void CachingAllocator::SetBindStream(const bool bind_stream) {
  bind_stream_ = bind_stream;
}

void CachingAllocator::FreeBlockBins() {
  GELOGI("Free block bins.");
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &bin : free_block_bins_) {
    if (bin != nullptr) {
      delete bin;
      bin = nullptr;
    }
  }
}

void CachingAllocator::PrintStatics(const GeLogLevel ge_log_level) {
  int32_t level = static_cast<int32_t>(ge_log_level);
  if (!IsLogEnable(GE_MODULE_NAME, level)) {
    return;
  }
  size_t total_using_size = 0U;
  size_t total_using_count = 0U;
  size_t total_free_size = 0U;
  size_t total_free_count = 0U;
  size_t total_malloc_size = 0U;
  size_t total_malloc_count = 0U;
  std::map<size_t, size_t> using_block_stat;
  std::map<size_t, size_t> free_block_stat;
  std::map<size_t, size_t> malloc_block_stat;
  {
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (const auto &pool : free_block_bins_) {
      if (pool == nullptr) {
        continue;
      }
      for (auto it = pool->cbegin(); it != pool->cend(); it++) {
        if ((*it) != nullptr) {
          total_free_size += (*it)->size;
          IncreaseCount(free_block_stat, (*it)->size);
          total_free_count++;
        }
      }
    }

    for (auto &it : allocated_blocks_) {
      if (it.second != nullptr) {
        total_using_size += it.second->size;
        IncreaseCount(using_block_stat, it.second->size);
        total_using_count++;
      }
    }

    for (auto &it : malloced_memory_) {
      total_malloc_size += it.first * it.second;
      total_malloc_count += it.second;
      malloc_block_stat[it.first] = it.second;
    }
  }
  GELOGI("Called counts[malloc:%11zu free:%11zu].", called_malloc_counts_.load(), called_free_counts_.load());
  PrintCount(malloc_block_stat, "Malloc", total_malloc_size, total_malloc_count);
  PrintCount(using_block_stat, "Using", total_using_size, total_using_count);
  PrintCount(free_block_stat, "Free", total_free_size, total_free_count);
}
}  // namespace ge
