/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_EXPM_H__
#define __ASCENDC_API_REGBASE_EXPM_H__

using namespace AscendC;

template <typename T>
__aicore__ inline void ExpmImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t calc_cnt) {
  uint32_t vl_size = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));

  MicroAPI::RegTensor<T> src_reg;
  MicroAPI::RegTensor<T> dst_reg;
  MicroAPI::RegTensor<T> one_reg;
  MicroAPI::MaskReg full_mask;
  full_mask = MicroAPI::CreateMask<uint8_t>();

  MicroAPI::Duplicate(one_reg, (T)1.0, full_mask);
  MicroAPI::MaskReg mask;
  // mainBlock
  for (uint16_t i = 0U; i < repeat_time; ++i) {
    mask = AscendC::MicroAPI::UpdateMask<T>(calc_cnt);
    MicroAPI::LoadAlign(src_reg, src + i * vl_size);

    MicroAPI::Exp(src_reg, src_reg, mask);
    MicroAPI::Sub(dst_reg, src_reg, one_reg, mask);

    MicroAPI::StoreAlign(dst + i * vl_size, dst_reg, mask);
  }
}

template <typename T>
__aicore__ inline void ExpmExtend(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t calc_cnt) {
  VF_CALL<ExpmImplVF<T>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src.GetPhyAddr(), calc_cnt);
}
#endif
