/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_L2_MEM_POOL_H
#define AIR_CXX_L2_MEM_POOL_H

#include "ge/ge_allocator.h"
#include "kernel/memory/allocator/scalable_allocator.h"
#include "caching_mem_allocator.h"
#include "utils/utils.h"

namespace gert {
namespace memory {
class L2MemPool;
class MultiStreamL1Allocator : public DeviceMemAllocator {
 public:
  MultiStreamL1Allocator(ge::Allocator *l1_allocator, aclrtStream l2_stream,
                         TypedContinuousVector<L2MemPool *> *all_l2_mem_pool = nullptr)
      : l1_allocator_(l1_allocator),
        l2_stream_(l2_stream),
        all_l2_mem_pool_(all_l2_mem_pool),
        is_rt2_multi_thread_(false) {
    if (IsEnableRmLaunchFreeEdge()) {
      is_rt2_multi_thread_ = true;
    }
  }
  BlockAddr Alloc(const MemSize size) override;
  bool Free(ge::MemBlock *const addr) override;
  DeviceId GetDeviceId() const override;
  void SetL1Allocator(ge::Allocator *allocator) {
    l1_allocator_ = allocator;
  }
  void SetStream(aclrtStream stream) {
    l2_stream_ = stream;
  }

 private:
  ge::Allocator *l1_allocator_;
  aclrtStream l2_stream_;
  TypedContinuousVector<L2MemPool *> *all_l2_mem_pool_;
  bool is_rt2_multi_thread_;
};
class L2MemPool : public ge::Allocator, public MemSynchronizer {
 public:
  explicit L2MemPool(ge::Allocator *allocator, aclrtStream stream,
                     TypedContinuousVector<L2MemPool *> *all_l2_mem_pool = nullptr);

  ~L2MemPool() override;

  ge::MemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override;
  void Recycle() override;
  ge::Status Synchronize() const override;
  ge::Status Finalize(bool no_log = false);
  void SetL1Allocator(ge::Allocator *allocator) {
    first_level_pool_.SetL1Allocator(allocator);
    const auto caching_mem_allocator = dynamic_cast<CachingMemAllocator *>(allocator);
    if ((caching_mem_allocator != nullptr) && (caching_mem_allocator->GetScalableAllocator() != nullptr)) {
      GELOGI("set l1 allocator %s for l2 allocator %s",
             caching_mem_allocator->GetScalableAllocator()->GetId().c_str(), memory_pool_->GetId().c_str());
    }
  }
  aclrtStream GetStream();
  void SetStream(aclrtStream stream);
  ge::MemBlock *MoveL2ToL1(ge::MemBlock *block);

 private:
  MultiStreamL1Allocator first_level_pool_;
  SpanAllocatorImp span_allocator_;
  std::unique_ptr<MemoryPool> memory_pool_;
  aclrtStream stream_;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_L2_MEM_POOL_H
