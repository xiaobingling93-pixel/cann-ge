/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_COMMON_TENSOR_VALUE_H_
#define GE_HYBRID_COMMON_TENSOR_VALUE_H_

#include <atomic>
#include "framework/memory/memory_api.h"
#include "framework/common/util.h"
#include "hybrid/common/npu_memory_allocator.h"

namespace ge {
namespace hybrid {

class TensorBuffer {
 public:
  static std::unique_ptr<TensorBuffer> Create(NpuMemoryAllocator * const allocator,
                                              const size_t size,
                                              const AllocationAttr * const attr = nullptr);

  static std::unique_ptr<TensorBuffer> Create(void *const buffer, const size_t size);

  TensorBuffer(NpuMemoryAllocator *const allocator, void *const buffer, const size_t size,
               const MemStorageType mem_type = MemStorageType::HBM);
  TensorBuffer(const TensorBuffer &) = delete;
  TensorBuffer &operator = (const TensorBuffer &) = delete;
  ~TensorBuffer();

  void* Release() {
    const auto ret = buffer_;
    buffer_ = nullptr;
    return ret;
  }

  void *GetData() const {
    return buffer_;
  }

  size_t GetSize() const {
    return size_;
  }

  MemStorageType GetMemType() const {
    return mem_type_;
  }

 private:
  NpuMemoryAllocator *allocator_ = nullptr;
  void *buffer_ = nullptr;
  size_t size_ = 0U;
  MemStorageType mem_type_;
};

class TensorValue {
 public:
  TensorValue() = default;

  explicit TensorValue(const std::shared_ptr<TensorBuffer> buffer);

  TensorValue(void *const buffer, const size_t size, const MemStorageType mem_type = MemStorageType::HOST_DDR);

  ~TensorValue();

  void Destroy();

  void *Release() {
    if ((buffer_ == nullptr) && (ref_buffer_ != nullptr)) {
      return ref_buffer_;
    }
    return buffer_->Release();
  }

  const void *GetData() const;

  std::string DebugString() const;

  void SetName(const std::string &name) {
    name_ = name;
  }

  MemStorageType GetMemType() const {
    return mem_type_;
  }

  void *MutableData();

  size_t GetSize() const;

  template<typename T>
  Status CopyScalarValueToHost(T &value) const {
    GE_CHECK_GE(this->GetSize(), sizeof(value));
    GE_CHK_RT_RET(aclrtMemcpy(PtrToPtr<T, void>(&value), sizeof(value), this->GetData(), sizeof(value),
        ACL_MEMCPY_DEVICE_TO_HOST));
    return SUCCESS;
  }

 private:
  std::shared_ptr<TensorBuffer> buffer_;
  std::string name_;
  // 1. for weights and variables
  // 2. for rt2
  void *ref_buffer_ = nullptr;
  size_t ref_size_ = 0U;
  MemStorageType mem_type_ = MemStorageType::HBM;
  // shape
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_COMMON_TENSOR_VALUE_H_
