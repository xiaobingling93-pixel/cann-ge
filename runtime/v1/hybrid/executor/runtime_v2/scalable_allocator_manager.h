/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_SCALABLE_ALLOCATOR_MANAGER_H_
#define GE_HYBRID_EXECUTOR_SCALABLE_ALLOCATOR_MANAGER_H_

#include <mutex>
#include <map>
#include "framework/runtime/gert_api.h"

namespace ge {
class ScalableAllocatorManager {
 public:
  ScalableAllocatorManager () = default;
  ~ScalableAllocatorManager() {
    const std::unique_lock<std::mutex> lk(allocators_lock_);
    allocators_.clear();
  }
  gert::Allocators *GetAllocator(const std::string &graph_name, const aclrtStream stream);
 private:
  std::map<aclrtStream, std::shared_ptr<gert::Allocators>> allocators_;
  std::mutex allocators_lock_;
};
} // namespace ge
#endif  // GE_HYBRID_EXECUTOR_SCALABLE_ALLOCATOR_MANAGER_H_
