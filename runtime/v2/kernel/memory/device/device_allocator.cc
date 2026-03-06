/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/device/device_allocator.h"
#include "framework/common/debug/ge_log.h"
#include "framework/runtime/device_memory_recorder.h"
#include "subscriber/profiler/cann_memory_profiler.h"
#include "kernel/memory/rts_caching_mem_allocator.h"
#include "graph/manager/mem_manager.h"

namespace gert {
DeviceAllocator::DeviceAllocator(DeviceMemAllocator &mem_allocator)
    : mem_allocator_(mem_allocator) {
}

BlockAddr DeviceAllocator::Alloc(const MemSize size) {
  ++alloc_count_;
  const auto ptr = mem_allocator_.Alloc(size);
  if (ptr != nullptr) {
    GELOGI("DeviceAllocator::Alloc device_id:%u, size:%llu, block_addr:%p", mem_allocator_.GetDeviceId(), size, ptr);
    occupied_size_ += size;
  } else {
    if (log_error_if_alloc_failed_) {
      GELOGE(ge::FAILED, "DeviceAllocator::Alloc failed device_id:%u, size:%llu", mem_allocator_.GetDeviceId(), size);
    }
  }
  return ptr;
}

void DeviceAllocator::Free(ge::MemBlock *const addr) {
  ++free_count_;
  if (addr == nullptr) {
    return;
  }
  auto size = addr->GetSize();
  if (mem_allocator_.Free(addr)) {
    GELOGI("DeviceAllocator::free device_id:%u ,size:%llu, block_addr:%p", mem_allocator_.GetDeviceId(), size, addr);
    if (occupied_size_ >= size) {
      occupied_size_ -= size;
    }
  }
}

void DeviceAllocator::SetLogErrorIfAllocFailed(bool log_error_if_alloc_failed) {
  log_error_if_alloc_failed_ = log_error_if_alloc_failed;
}

RtMemAllocator::RtMemAllocator(ge::Allocator &allocator, const DeviceId device_id, const uint32_t mem_type)
    : allocator_{allocator}, device_id_{device_id}, mem_type_{mem_type} {
}

BlockAddr RtMemAllocator::Alloc(const MemSize size) {
  void *ptr = nullptr;
  const auto rt_ret = rtMalloc(&ptr, size, mem_type_, GE_MODULE_NAME_U16);
  if (rt_ret == RT_ERROR_NONE) {
    DeviceMemoryRecorder::AddTotalReserveMemory(static_cast<uint64_t>(size));
    GE_PRINT_DYNAMIC_MEMORY(rtMalloc, ge::ToMallocMemInfo("page caching", ptr, device_id_, GE_MODULE_NAME_U16).c_str(),
                            size);

    // The construction of the MemBlock class requires a reference to the allocator
    // but in reality, the allocator here does not need to be used
    auto block = new (block_allocator_.Alloc()) ge::MemBlock{allocator_, ptr, static_cast<size_t>(size)};
    return block;
  }
  REPORT_INNER_ERR_MSG("E19999", "Call rtMalloc fail, purpose: page caching, type = %u, size:%llu, device_id:%u",
                    mem_type_, size, device_id_);
  GELOGE(ge::INTERNAL_ERROR, "[Malloc][Memory] failed, rt_ret:%d, device_id:%u, size:%llu", rt_ret, device_id_,
         size);
  return nullptr;
}

bool RtMemAllocator::Free(ge::MemBlock *const addr) {
  auto size = addr->GetSize();
  GELOGI("RtMemAllocator::free device_id:%u ,size:%llu, mem_addr:%p", GetDeviceId(), size, addr->GetAddr());
  const auto rt_ret = rtFree(addr->GetAddr());
  if (rt_ret == RT_ERROR_NONE) {
    block_allocator_.Free(dynamic_cast<ge::MemBlock &>(*addr));
    DeviceMemoryRecorder::ReduceTotalReserveMemory(static_cast<uint64_t>(size));
    return true;
  }
  REPORT_INNER_ERR_MSG("E19999", "Call aclrtFree fail, device_id:%u", device_id_);
  GELOGE(ge::FAILED, "[Call][RtFree] failed, rt_ret:%d, device_id:%u, addr:%p, size:%llu",
         rt_ret, device_id_, addr->GetAddr(), size);
  return false;
}
}

