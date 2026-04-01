/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef B0F387D2672945138C8316EF65776372_H
#define B0F387D2672945138C8316EF65776372_H

#include <cstdint>
#include <array>
#include <memory>
#include <unordered_map>
#include "runtime/mem.h"
#include "mem_block.h"
#include "ge/ge_api_types.h"
#include "ge/ge_allocator.h"
#include "kernel/memory/allocator/scalable_allocator.h"
#include "runtime/base.h"
#include "runtime/mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "rts_caching_mem_allocator.h"
#include "multi_stream_mem_block_pool.h"
#include "framework/runtime/device_memory_recorder.h"

namespace gert {
constexpr uint32_t MEM_QUEUE_NUM = 21U;
namespace memory {
struct CachingMemAllocator : public ge::Allocator, public MemSynchronizer {
  CachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type);
  CachingMemAllocator(const std::string &graph_name, const uint32_t device_id, const rtMemType_t memory_type);
  CachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type, ScalableConfig &config);
  explicit CachingMemAllocator(const uint32_t device_id);
  ~CachingMemAllocator() override {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0U; i < same_thread_allocators_.size(); ++i) {
      if (same_thread_allocators_[i] == this) {
        same_thread_allocators_.erase(same_thread_allocators_.begin() + i);
        break;
      }
    }
    for (size_t i = 0U; i < all_caching_mem_allocators_.size(); ++i) {
      if (all_caching_mem_allocators_[i] == this) {
        all_caching_mem_allocators_.erase(all_caching_mem_allocators_.begin() + i);
        break;
      }
    }
    (void)Finalize(true);
  }
  ge::MemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override {
    DeviceMemoryRecorder::ReduceTotalAllocateMemory(static_cast<uint64_t>(block->GetSize()));
    memory_pool_->Free(dynamic_cast<PageSpan *>(block));
  }
  ge::Status Finalize(bool no_log = false);
  static std::unique_ptr<CachingMemAllocator> GetAllocator();
  static std::unique_ptr<CachingMemAllocator> GetAllocator(const uint32_t device_id);
  static std::unique_ptr<CachingMemAllocator> GetAllocator(const std::string &graph_name, const uint32_t device_id,
                                                           const rtMemType_t rt_mem_type);
  thread_local static std::vector<CachingMemAllocator *> same_thread_allocators_;
  static std::vector<CachingMemAllocator *> all_caching_mem_allocators_;
  ge::Status Synchronize() const override;
  void Recycle() override;
  void SetStream(const aclrtStream stream) {
    stream_ = stream;
  }

  DeviceId GetDeviceId() const {
    return rts_mem_allocator_.GetDeviceId();
  }

 public:
  static std::mutex mutex_;
  ScalableAllocator *GetScalableAllocator() {
    return reinterpret_cast<ScalableAllocator *>(memory_pool_.get());
  }

 private:
  void RecallMemBlocks(size_t start_queue_index = 0);
  ge::Status TryExtendCache(size_t queue_index);
  ge::MemBlock *AllocateWithTryRecycle(size_t size);

 private:
  RtsFirstLevelPool rts_mem_allocator_;
  SpanAllocatorImp span_allocator_;
  std::unique_ptr<MemoryPool> memory_pool_;
  aclrtStream stream_ = nullptr;
};
}  // namespace memory
}  // namespace gert

#endif
