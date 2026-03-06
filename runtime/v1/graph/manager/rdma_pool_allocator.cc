/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/rdma_pool_allocator.h"

#include "graph/types.h"
#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "graph/def_types.h"
#include "graph/ge_context.h"
#include "runtime/dev.h"
#include "graph/manager/mem_manager.h"
#include "common/math/math_util.h"
#include "graph_metadef/common/ge_common/util.h"

namespace {
constexpr size_t kAlignedSize = 512U;
constexpr ge::float32_t kSplitBlockThreshold = 0.5F;

inline size_t GetAlignedBlockSize(const size_t size) {
  if (size == 0U) {
    return kAlignedSize;
  }
  if (ge::CheckSizeTAddOverflow(size, kAlignedSize) != ge::SUCCESS) {
    return SIZE_MAX;
  }
  return kAlignedSize * ((size + kAlignedSize - 1U) / kAlignedSize);
}

inline bool ShouldSplit(const ge::Block &block, const size_t size) {
  return static_cast<ge::float64_t>(size) <= (static_cast<ge::float64_t>(block.size) * kSplitBlockThreshold);
}

inline bool CanMergeBlock(const ge::Block &block) { return !block.allocated; }

bool BlockComp(const ge::Block *const left, const ge::Block *const right) {
  if (left->size != right->size) {
    return left->size < right->size;
  }
  return ge::PtrToValue(left->ptr) < ge::PtrToValue(right->ptr);
}
}  // namespace

namespace ge {
RdmaPoolAllocator::RdmaPoolAllocator(const rtMemType_t memory_type)
    : memory_type_(memory_type), block_bin_(BlockBin(&BlockComp)) {}

Status RdmaPoolAllocator::Initialize() {
  memory_allocator_ = &MemManager::Instance().MemInstance(memory_type_);
  return SUCCESS;
}

void RdmaPoolAllocator::Finalize() {
  GELOGD("Rdma pool finalize start.");
  auto it_block = allocated_blocks_.begin();
  while (it_block != allocated_blocks_.end()) {
    const auto block = it_block->second;
    it_block = allocated_blocks_.erase(it_block);
    delete block;
  }
  auto it_bin = block_bin_.begin();
  while (it_bin != block_bin_.end()) {
    const auto block = *it_bin;
    it_bin = block_bin_.erase(it_bin);
    delete block;
  }

  if (rdma_base_addr_ != nullptr) {
    GELOGD("Start to free rdma pool memory.");
    if ((memory_allocator_ == nullptr) || (memory_allocator_->FreeMemory(rdma_base_addr_) != SUCCESS)) {
      GELOGW("Free rdma pool memory failed");
    }
    rdma_base_addr_ = nullptr;
  }
}

Status RdmaPoolAllocator::InitMemory(const size_t mem_size) {
  const auto device_id = GetContext().DeviceId();
  GELOGD("Init Rdma Memory with size [%zu] for devid:[%u].", mem_size, device_id);
  if (rdma_base_addr_ != nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param rdma_base_addr_ is not nullptr, devid:%u, check invalid", device_id);
    GELOGE(GE_MULTI_INIT, "[Check][Param] Rdma pool has been malloced, devid:%u", device_id);
    return GE_MULTI_INIT;
  }
  const std::string purpose = "Memory for rdma pool";
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const int32_t dev_id = static_cast<int32_t>(device_id);
  GE_CHK_RT_RET(rtSetDevice(dev_id));
  // DeviceReset before memory finished!
  GE_MAKE_GUARD(not_used_var, [&dev_id]() { GE_CHK_RT(rtDeviceReset(dev_id)); });

  GE_CHECK_NOTNULL(memory_allocator_);
  rdma_base_addr_ = memory_allocator_->MallocMemory(purpose, mem_size, device_id);
  if (rdma_base_addr_ == nullptr) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[Malloc][Memory] failed, size:%zu, device_id:%u", mem_size, device_id);
    return GE_GRAPH_MALLOC_FAILED;
  }
  rdma_mem_size_ = mem_size;
  // Init with a base block.
  auto *const base_block = new (std::nothrow) Block(device_id, mem_size, rdma_base_addr_);
  if (base_block == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New Block failed, size:%zu, device_id:%u", mem_size, device_id);
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[New][Block] failed, size:%zu, device_id:%u", mem_size, device_id);
    return GE_GRAPH_MALLOC_FAILED;
  }
  (void)block_bin_.insert(base_block);
  return SUCCESS;
}

uint8_t *RdmaPoolAllocator::Malloc(const size_t size, const uint32_t device_id) {
  GELOGI("start to malloc rdma memory size:%zu, device id = %u.", size, device_id);
  const auto aligned_size = GetAlignedBlockSize(size);
  Block key(device_id, aligned_size, nullptr);
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = block_bin_.lower_bound(&key);
  if (it != block_bin_.end()) {
    Block *block = *it;
    (void)block_bin_.erase(it);
    block->allocated = true;
    if (block->ptr == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Rdmapool memory address is nullptr, device_id:%u, check invalid",
                         device_id);
      GELOGE(INTERNAL_ERROR, "[Check][Param] Rdmapool memory address is nullptr, device_id:%u", device_id);
      return nullptr;
    }
    (void)allocated_blocks_.emplace(block->ptr, block);

    if (ShouldSplit(*block, aligned_size)) {
      GELOGD("Block will be splited block size = %zu, aligned_size:%zu.", block->size, aligned_size);
      auto *const new_block = new (std::nothrow) Block(device_id, block->size - aligned_size, nullptr,
                                                       PtrAdd(block->ptr, block->size, aligned_size));
      if (new_block == nullptr) {
        GELOGW("Block split failed");
        return block->ptr;
      }
      new_block->next = block->next;
      if (block->next != nullptr) {
        block->next->prev = new_block;
      }
      new_block->prev = block;
      block->next = new_block;
      block->size = aligned_size;
      (void)block_bin_.insert(new_block);
    }
    GELOGD("Find block size = %zu", block->size);
    return block->ptr;
  }
  GELOGW("Memory block not founded.");
  return nullptr;
}

Status RdmaPoolAllocator::Free(uint8_t *const memory_addr, const uint32_t device_id) {
  GELOGI("Free rdma memory, device id = %u.", device_id);
  if (memory_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param memory_addr is nullptr, device_id:%u, check invalid", device_id);
    GELOGE(GE_GRAPH_FREE_FAILED, "[Check][Param] Invalid memory pointer, device id:%u", device_id);
    return GE_GRAPH_FREE_FAILED;
  }

  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = allocated_blocks_.find(memory_addr);
  if (it == allocated_blocks_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Param memory_addr is not allocated before, device_id:%u, "
                       "check invalid", device_id);
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid memory pointer, device id:%u", device_id);
    return PARAM_INVALID;
  }

  Block *const block = it->second;
  block->allocated = false;
  (void)allocated_blocks_.erase(it);

  const std::vector<Block *> merge_blocks = {block->prev, block->next};
  for (Block *const merge_block : merge_blocks) {
    if (merge_block != nullptr) {
      MergeBlocks(*block, *merge_block);
    }
  }
  (void)block_bin_.insert(block);

  return SUCCESS;
}

void RdmaPoolAllocator::MergeBlocks(Block &dst, Block &src) {
  if ((!CanMergeBlock(dst)) || (!CanMergeBlock(src))) {
    return;
  }

  if (dst.prev == &src) {
    dst.ptr = src.ptr;
    dst.prev = src.prev;
    if (dst.prev != nullptr) {
      dst.prev->next = &dst;
    }
  } else {
    dst.next = src.next;
    if (dst.next != nullptr) {
      dst.next->prev = &dst;
    }
  }

  if (CheckSizeTAddOverflow(dst.size, src.size) == SUCCESS) {
    dst.size += src.size;
  } else {
    dst.size = SIZE_MAX;
  }
  (void)block_bin_.erase(&src);
  delete &src;
}

Status RdmaPoolAllocator::GetBaseAddr(uint64_t &base_addr, uint64_t &mem_size) const {
  if (rdma_base_addr_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param rdma_base_addr_ is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Rdma base addr is nullptr.");
    return INTERNAL_ERROR;
  }
  base_addr = PtrToValue(rdma_base_addr_);
  mem_size = rdma_mem_size_;
  return SUCCESS;
}
}  // namespace ge
