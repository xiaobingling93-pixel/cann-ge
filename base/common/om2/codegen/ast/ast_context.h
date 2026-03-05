/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_OM2_CODEGEN_AST_AST_CONTEXT_H
#define BASE_COMMON_OM2_CODEGEN_AST_AST_CONTEXT_H
#include <vector>
#include <cstdint>

#include "securec.h"
#include "ge_common/ge_api_types.h"
#include "common/checker.h"

namespace ge {
class AstNodePool {
 public:
  AstNodePool() = default;
  ~AstNodePool();

  AstNodePool(const AstNodePool &) = delete;
  AstNodePool &operator=(const AstNodePool &) = delete;

  uint8_t *Allocate(size_t mem_size);

  size_t GetMemoryUsage() const {
    return total_mem_usage_;
  }

 private:
  size_t CreateNewBlock(size_t min_size);
  static size_t AlignTo(const size_t size, const size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
  }

 private:
  struct Block {
    uint8_t *data;
    size_t length;
    size_t offset;
  };

  std::vector<Block> blocks_;
  size_t total_mem_usage_ = 0UL;
};

class StringRef {
 public:
  StringRef() : data_(nullptr), length_(0UL) {}
  explicit StringRef(const char_t *str) : data_(str), length_(str ? std::strlen(str) : 0UL) {}
  StringRef(const char_t *str, const size_t len) : data_(str), length_(len) {}

  const char_t *Data() const {
    return data_;
  }

  size_t Length() const {
    return length_;
  }

  bool Empty() const {
    return length_ == 0;
  }

  bool operator==(const StringRef &other) const {
    if (length_ != other.length_) {
      return false;
    }
    if (length_ == 0) {
      return true;
    }
    if (data_ == nullptr || other.data_ == nullptr) {
      return false;
    }
    return std::memcmp(data_, other.data_, length_) == 0;
  }

 private:
  const char_t *data_;
  size_t length_;
};

template <typename T>
class ArrayRef {
 public:
  using iterator = const T *;

  ArrayRef() : data_(nullptr), length_(0) {}
  ArrayRef(const T *start, const size_t len) : data_(start), length_(len) {}

  const T *Data() const {
    return data_;
  }

  size_t Size() const {
    return length_;
  }

  const T &operator[](size_t index) const {
    if (index >= length_) {
      throw std::out_of_range("Index out of range");
    }
    return data_[index];
  }

  bool Empty() const {
    return length_ == 0;
  }

 protected:
  const T *data_;
  size_t length_;
};

template <typename T>
class MutableArrayRef : public ArrayRef<T> {
 public:
  MutableArrayRef(T *mutable_data, size_t length) : ArrayRef<T>(mutable_data, length) {}

  T &operator[](size_t index) const {
    if (index >= this->length_) {
      throw std::out_of_range("Index out of range");
    }
    // data_ itself is writable, so const_cast is safe.
    return const_cast<T &>(this->data_[index]);
  }
};

class AstContext {
 public:
  AstContext() = default;
  ~AstContext() = default;

  AstContext(const AstContext &) = delete;
  AstContext &operator=(const AstContext &) = delete;
  AstContext(AstContext &&) = delete;
  AstContext &operator=(AstContext &&) = delete;

  void *Allocate(const size_t size) {
    return node_pool_.Allocate(size);
  }

  size_t GetMemoryUsage() const {
    return node_pool_.GetMemoryUsage();
  }

  template <typename T>
  MutableArrayRef<T> AllocateMutableArray(const size_t count) {
    static_assert(std::is_pointer<T>::value || std::is_same<std::decay_t<T>, StringRef>::value,
                  "ArrayRef element type must be a pointer or StringRef.");
    static_assert(std::is_trivially_destructible<T>::value, "Array elements T must be trivially destructible");

    if (count == 0) {
      return MutableArrayRef<T>(nullptr, 0);
    }

    void *mem = Allocate(count * sizeof(T));
    if (mem == nullptr) {
      return MutableArrayRef<T>(nullptr, 0);
    }

    T *data = static_cast<T *>(mem);
    for (size_t i = 0; i < count; ++i) {
      (void)new (&data[i]) T();
    }

    return MutableArrayRef<T>(data, count);
  }

  StringRef CopyString(const char_t *s) {
    if (!s) {
      return {};
    }
    const size_t len = std::strlen(s);
    const auto dest = static_cast<char_t *>(Allocate(len + 1));
    if (dest == nullptr) {
      return {};
    }
    GE_ASSERT_EOK(memcpy_s(dest, len, s, len));
    dest[len] = '\0';

    return {dest, len};
  }

 private:
  AstNodePool node_pool_;
};
}  // namespace ge
#endif  // BASE_COMMON_OM2_CODEGEN_AST_AST_CONTEXT_H
