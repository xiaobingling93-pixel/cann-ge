/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_COMMON_NPU_MEMORY_ALLOCATOR_H_
#define GE_HYBRID_COMMON_NPU_MEMORY_ALLOCATOR_H_

#include <mutex>
#include "framework/memory/memory_api.h"
#include "graph/manager/caching_allocator.h"

namespace ge {
namespace hybrid {
class AllocationAttr {
 public:
  AllocationAttr() = default;
  explicit AllocationAttr(const int32_t padding);
  explicit AllocationAttr(void *const try_reuse_addr);
  AllocationAttr(const int32_t padding, void *const try_reuse_addr,
                 const MemStorageType mem_type = MemStorageType::HBM);
  ~AllocationAttr() = default;
  void SetMemType(const MemStorageType memType) { mem_type_ = memType; }
  MemStorageType GetMemType() const { return mem_type_; }

 private:
  friend class NpuMemoryAllocator;
  int32_t padding_ = 0;
  void *try_reuse_addr_ = nullptr;
  MemStorageType mem_type_ = MemStorageType::HBM;
};

class NpuMemoryAllocator {
 public:
  NpuMemoryAllocator(const uint32_t device_id, const aclrtStream stream);
  ~NpuMemoryAllocator();
  static NpuMemoryAllocator *GetAllocator(const uint32_t device_id, const aclrtStream stream);
  static NpuMemoryAllocator *GetAllocator(const aclrtStream stream);
  static NpuMemoryAllocator *GetAllocator();
  static void Finalize();
  static void FreeCachedMem();
  static void ClearStream(const aclrtStream stream);

  static AllocationAttr* AttrWithDefaultPadding() {
    static AllocationAttr attr(kDefaultPadding, nullptr);
    return &attr;
  }

  Status InitCachingllocator();
  void *Allocate(const uint64_t size, const AllocationAttr *const attr = nullptr) const;
  void Deallocate(void *const data, const MemStorageType mem_storage_type = MemStorageType::HBM) const;

  static constexpr int32_t kDefaultPadding = 32;
 private:
  Status TryFreeAndMalloc(const size_t size, void **buffer) const;
  Status TryFreeCachingMem() const;
  void *AllocateCachingMem(const std::size_t size, void *const try_reuse_addr) const;

  uint32_t device_id_;
  aclrtStream stream_;
  std::unique_ptr<CachingAllocator> caching_allocator_;

  using DeviceidAllocatorMap = std::map<uint32_t, std::unique_ptr<NpuMemoryAllocator>>;
  static DeviceidAllocatorMap default_allocators_;
  static std::map<aclrtStream, std::unique_ptr<DeviceidAllocatorMap>> allocators_;
  static std::set<aclrtStream> streams_;
  static std::mutex mu_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_COMMON_NPU_MEMORY_ALLOCATOR_H_
