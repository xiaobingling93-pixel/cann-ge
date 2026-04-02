/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scalable_allocator_manager.h"
#include "framework/runtime/gert_api.h"
#include "framework/runtime/mem_allocator.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/model/external_allocator_manager.h"

namespace ge {
gert::Allocators *ScalableAllocatorManager::GetAllocator(const std::string &graph_name, const aclrtStream stream) {
  const std::unique_lock<std::mutex> lk(allocators_lock_);
  const auto iter = allocators_.find(stream);
  if (iter != allocators_.end()) {
    return iter->second.get();
  }

  std::shared_ptr<ge::Allocator> device_allocator;
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator != nullptr) {
    GELOGD("Use external device allocator = %p with stream = %p", external_allocator.get(), stream);
    device_allocator = external_allocator;
  } else {
    device_allocator.reset(gert::AllocatorFactory::Create(graph_name, gert::kOnDeviceHbm).release());
    GELOGD("Create device allocator = %p with stream = %p", device_allocator.get(), stream);
  }
  GE_ASSERT_NOTNULL(device_allocator);

  std::shared_ptr<ge::Allocator> host_allocator(
      gert::AllocatorFactory::Create(graph_name, gert::kOnHost).release());
  GE_ASSERT_NOTNULL(host_allocator);

  std::shared_ptr<ge::Allocator> p2p_device_allocator(
      gert::AllocatorFactory::Create(gert::kOnDeviceP2p).release());
  GE_ASSERT_NOTNULL(p2p_device_allocator);

  // create a new allocator
  std::shared_ptr<gert::Allocators> allocators = MakeShared<gert::Allocators>();
  GE_ASSERT_NOTNULL(allocators);
  for (size_t i = 0U; i < static_cast<size_t>(gert::AllocatorUsage::kEnd); ++i) {
    (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kOnDeviceHbm), i, device_allocator);
    (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kOnDeviceP2p), i, p2p_device_allocator);
    (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kOnHost), i, host_allocator);
    (void)allocators->SetAllocator(static_cast<gert::TensorPlacement>(gert::kFollowing), i, host_allocator);
  }
  allocators_[stream] = allocators;
  return allocators.get();
}
} // ge namespace
