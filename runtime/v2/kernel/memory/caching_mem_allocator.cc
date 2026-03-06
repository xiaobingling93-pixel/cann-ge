/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>

#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "caching_mem_allocator.h"

#include "runtime/rt.h"

#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "utils/utils.h"
#include "multi_stream_mem_block_helper.h"
#include "graph/load/model_manager/model_utils.h"

namespace gert {
namespace memory {
thread_local std::vector<CachingMemAllocator *> CachingMemAllocator::same_thread_allocators_ = {};
std::vector<CachingMemAllocator *> CachingMemAllocator::all_caching_mem_allocators_ = {};
std::mutex CachingMemAllocator::mutex_;

ge::MemBlock *CachingMemAllocator::AllocateWithTryRecycle(size_t size) {
  auto addr = memory_pool_->Alloc(*this, size);
  if (addr != nullptr) {
    return addr;
  }

  GELOGE(ge::MEMALLOC_FAILED,
         "%s Failed to apply for memory. We will try to free memory from memory pool, the above or this error log can "
         "be ignored. Try to free cached memory...",
         memory_pool_->GetId().c_str());
  memory_pool_->PrintDetails(DLOG_INFO);
  GELOGI("will synchronize on stream %p", stream_);
  GE_ASSERT_SUCCESS(Synchronize());
  Recycle();
  addr = memory_pool_->Alloc(*this, size);
  if (addr == nullptr) {
    GELOGI("addr is nullptr, try to free other allocator memory and malloc again");
    const std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0U; i < same_thread_allocators_.size(); ++i) {
      if (same_thread_allocators_[i] == this) {
        continue;
      }
      GE_ASSERT_SUCCESS(same_thread_allocators_[i]->Synchronize());
      same_thread_allocators_[i]->memory_pool_->Recycle();
      addr = memory_pool_->Alloc(*this, size);
      if (addr == nullptr) {
        continue;
      }
      break;
    }
  }
  return addr;
}

ge::MemBlock *CachingMemAllocator::Malloc(size_t size) {
  GELOGI("Malloc size:%zu.", size);
  auto block_mem = AllocateWithTryRecycle(size);
  if (block_mem != nullptr) {
    DeviceMemoryRecorder::AddTotalAllocateMemory(static_cast<uint64_t>(block_mem->GetSize()));
  }
  return block_mem;
}

CachingMemAllocator::CachingMemAllocator(const uint32_t device_id)
    : rts_mem_allocator_(*RtsCachingMemAllocator::GetAllocator(device_id, RT_MEMORY_HBM), device_id, "rt2 memory pool"),
      // 不加nothrow，理由：由于构造函数无法返回失败，且这是关键资源申请，如果申请失败允许进程退出。
      memory_pool_(new ScalableAllocator(span_allocator_, rts_mem_allocator_, ScalableConfig())) {
  GELOGI("create caching memory allocator, %s", memory_pool_->GetId().c_str());
}

CachingMemAllocator::CachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type)
    : CachingMemAllocator("", device_id, memory_type) {}

CachingMemAllocator::CachingMemAllocator(const std::string &graph_name, const uint32_t device_id,
                                         const rtMemType_t memory_type)
    : rts_mem_allocator_(*RtsCachingMemAllocator::GetAllocator(device_id, memory_type), device_id, "rt2 memory pool"),
    // 不加nothrow，理由：由于构造函数无法返回失败，且这是关键资源申请，如果申请失败允许进程退出。
      memory_pool_(new ScalableAllocator(span_allocator_, rts_mem_allocator_, ScalableConfig(), graph_name)) {
  const std::lock_guard<std::mutex> lock(mutex_);
  same_thread_allocators_.emplace_back(this);
  all_caching_mem_allocators_.emplace_back(this);
  GELOGI("create caching memory allocator, %s", memory_pool_->GetId().c_str());
}

CachingMemAllocator::CachingMemAllocator(const uint32_t device_id, const rtMemType_t memory_type,
                                         ScalableConfig &config)
    : rts_mem_allocator_(*RtsCachingMemAllocator::GetAllocator(device_id, memory_type), device_id, "rt2 memory pool"),
    // 不加nothrow，理由：由于构造函数无法返回失败，且这是关键资源申请，如果申请失败允许进程退出。
      memory_pool_(new ScalableAllocator(span_allocator_, rts_mem_allocator_, config)) {
  const std::lock_guard<std::mutex> lock(mutex_);
  same_thread_allocators_.emplace_back(this);
  all_caching_mem_allocators_.emplace_back(this);
  GELOGI("create caching memory allocator, %s", memory_pool_->GetId().c_str());
}

std::unique_ptr<CachingMemAllocator> CachingMemAllocator::GetAllocator(const uint32_t device_id) {
  return GetAllocator("", device_id, RT_MEMORY_HBM);
}

std::unique_ptr<CachingMemAllocator> CachingMemAllocator::GetAllocator(const std::string &graph_name,
                                                                       const uint32_t device_id,
                                                                       const rtMemType_t rt_mem_type) {
  auto caching_allocator = ge::MakeUnique<CachingMemAllocator>(graph_name, device_id, rt_mem_type);
  if ((caching_allocator != nullptr) && ge::ModelUtils::IsGeUseExtendSizeMemory(true)) {
    auto allocator = caching_allocator->GetScalableAllocator();
    if (allocator != nullptr) {
      allocator->InitExpandableAllocator(*caching_allocator, rt_mem_type);
    }
  }
  return caching_allocator;
}

std::unique_ptr<CachingMemAllocator> CachingMemAllocator::GetAllocator() {
  int32_t device_id = 0;
  const auto rt_result = rtGetDevice(&device_id);
  if (rt_result != RT_ERROR_NONE) {
    GELOGE(ge::RT_FAILED, "[Get][Device] Failed, result:%d.", rt_result);
    REPORT_INNER_ERR_MSG("E19999", "rtGetDevice failed, result:%d.", rt_result);
    return nullptr;
  }
  return GetAllocator(device_id);
}

ge::Status CachingMemAllocator::Finalize(bool no_log) {
  return memory_pool_->Finalize(no_log);
}

ge::Status CachingMemAllocator::Synchronize() const {
  // call rtStreamSynchronize
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(stream_));
  return ge::SUCCESS;
}
void CachingMemAllocator::Recycle() {
  memory_pool_->Recycle();
}
}  // namespace memory
}  // namespace gert
