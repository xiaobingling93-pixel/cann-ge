/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_MATH_HIF8_T_H_
#define GE_COMMON_MATH_HIF8_T_H_

#include <cstdint>
#include "common/fp16_t/fp16_t.h"

namespace ge {
using hif8_t = class HiF8 final {
private:
  HiF8() = default;
  uint8_t u8_;
public:
  // Raw construction from bits.
  static HiF8 FromRawBits(uint8_t bits);

  explicit HiF8(float f32);
  explicit HiF8(fp16_t f16);
  static uint8_t BitsFromFp16(uint16_t f16);
  static uint8_t BitsFromFp32(uint32_t f32);

  bool IsNaN() const;
  bool IsInf() const;
  explicit operator float() const;
  explicit operator fp16_t() const;

  friend bool operator==(const HiF8 lhs, const HiF8 rhs) noexcept {
    return lhs.u8_ == rhs.u8_;
  }
};

static_assert(sizeof(hif8_t) == sizeof(uint8_t), "sizeof hif8_t must be 1");
} // namespace ge

#endif
