/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEVICE_MEMORY_PTR_H_
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEVICE_MEMORY_PTR_H_
#include <cstdint>
#include "runtime/mem.h"
#include "graph/def_types.h"
#include "acl/acl_rt.h"

namespace ge {
class DeviceMemoryPtr {
 public:
  DeviceMemoryPtr(DeviceMemoryPtr &&other) noexcept : addr_(other.addr_) {
    other.addr_ = 0UL;
  }
  DeviceMemoryPtr &operator=(DeviceMemoryPtr &&other) & noexcept {
    Free();
    addr_ = other.addr_;
    other.addr_ = 0UL;
    return *this;
  }
  DeviceMemoryPtr(const DeviceMemoryPtr &) = delete;
  DeviceMemoryPtr &operator=(const DeviceMemoryPtr &) & = delete;
  ~DeviceMemoryPtr() {
    Free();
  }
  DeviceMemoryPtr() : DeviceMemoryPtr(0UL) {}
  explicit DeviceMemoryPtr(uint64_t addr) : addr_(addr) {}
  uint64_t Get() const {
    return addr_;
  }
  void Reset(uint64_t addr) {
    Free();
    addr_ = addr;
  }

 private:
  void Free() noexcept {
    if (addr_ != 0UL) {
      if (aclrtFree(ValueToPtr(addr_)) == RT_ERROR_NONE) {
        addr_ = 0UL;
      }
    }
  }

 private:
  uint64_t addr_;
};
}  // namespace ge
#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEVICE_MEMORY_PTR_H_
