/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ATAN2_H__
#define __ASCENDC_API_REGBASE_ATAN2_H__

using namespace AscendC;

constexpr float ATAN2_PI = 3.14159265358979323846264338327950288;
constexpr float ATAN2_HALF_PI = 1.57079632679489661923132169163975144;
constexpr float ATAN2_NEG_HALF_PI = 0 - ATAN2_HALF_PI;
constexpr float ATAN2_NEG_PI = 0 - ATAN2_PI;
constexpr float ATAN2_FLOAT_NAN = 0x7fc00000;
constexpr float ATAN2_HALF_NAN = 0x7e00;

/* atan2(y, x) 的实现：

1. 如果 x > 0：
   atan2(y, x) = atan(y/x)

2. 如果 x < 0：
   atan2(y, x) = atan(y/x) + π  (如果 y ≥ 0)
   atan2(y, x) = atan(y/x) - π  (如果 y < 0)

3. 如果 x == 0：
   atan2(y, x) = π/2   (如果 y > 0)
   atan2(y, x) = -π/2  (如果 y < 0)
   atan2(0, 0) = NaN    (未定义)
*/

template <typename T>
__aicore__ inline void AtanPostProcess(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* src_atan,
                                       uint32_t calc_cnt) {
  uint32_t vl_size = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));

  MicroAPI::RegTensor<T> src0_reg, src1_reg, atan_reg, half_PI_reg, neg_half_PI_reg, PI_reg, neg_PI_reg;
  MicroAPI::RegTensor<T> dst_reg, nan_reg;
  MicroAPI::MaskReg full_mask, dst_mask;
  full_mask = MicroAPI::CreateMask<uint8_t>();
  if constexpr (std::is_same_v<T, float>) {
    MicroAPI::Duplicate(nan_reg, ATAN2_FLOAT_NAN, full_mask);
  } else if constexpr (std::is_same_v<T, half>) {
    MicroAPI::Duplicate(nan_reg, ATAN2_HALF_NAN, full_mask);
  }
  MicroAPI::Duplicate(half_PI_reg, ATAN2_HALF_PI, full_mask);
  MicroAPI::Duplicate(neg_half_PI_reg, ATAN2_NEG_HALF_PI, full_mask);
  MicroAPI::Duplicate(PI_reg, ATAN2_PI, full_mask);
  MicroAPI::Duplicate(neg_PI_reg, ATAN2_NEG_PI, full_mask);

  MicroAPI::MaskReg mask;
  // mainBlock
  for (uint16_t i = 0U; i < repeat_time; ++i) {
    mask = AscendC::MicroAPI::UpdateMask<T>(calc_cnt);
    MicroAPI::LoadAlign(src0_reg, src0 + i * vl_size);
    MicroAPI::LoadAlign(src1_reg, src1 + i * vl_size);
    MicroAPI::LoadAlign(atan_reg, src_atan + i * vl_size);

    // 按照y > =0或y < 0填充PI寄存器
    MicroAPI::Compares<T, CMPMODE::GE>(dst_mask, src1_reg, (T)0.0, mask);
    MicroAPI::Select(dst_reg, PI_reg, neg_PI_reg, dst_mask);
    // atan结果与PI寄存器相加
    MicroAPI::Add(dst_reg, dst_reg, atan_reg, mask);

    // x > 0时，atan2(y, x) = atan(y/x)，否则与PI寄存器相加
    MicroAPI::Compares<T, CMPMODE::GT>(dst_mask, src0_reg, (T)0.0, mask);
    MicroAPI::Select(dst_reg, atan_reg, dst_reg, dst_mask);

    // 处理x == 0的情况
    // y > 0时，atan2(y, x) = π/2
    MicroAPI::Compares<T, CMPMODE::GT>(dst_mask, src1_reg, (T)0.0, mask);
    MicroAPI::Select(nan_reg, half_PI_reg, nan_reg, dst_mask);

    // y < 0时，atan2(y, x) = -π/2
    // x == 0时，y == 0时，atan2(y, x) = NaN
    MicroAPI::Compares<T, CMPMODE::LT>(dst_mask, src1_reg, (T)0.0, mask);
    MicroAPI::Select(nan_reg, neg_half_PI_reg, nan_reg, dst_mask);

    MicroAPI::Compares<T, CMPMODE::EQ>(dst_mask, src0_reg, (T)0.0, mask);
    MicroAPI::Select(dst_reg, nan_reg, dst_reg, dst_mask);

    MicroAPI::StoreAlign(dst + i * vl_size, dst_reg, mask);
  }
}

template <typename T>
__aicore__ inline void Atan2Extend(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calc_cnt) {
  AscendC::Div(dst, src1, src0, calc_cnt);
  AscendC::Atan(dst, dst, tmp_buf, calc_cnt);
  VF_CALL<AtanPostProcess<T>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src0.GetPhyAddr(),
                              (__ubuf__ T*)src1.GetPhyAddr(), (__ubuf__ T*)dst.GetPhyAddr(), calc_cnt);
}
#endif
