/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_COPY_SIGN_H__
#define __ASCENDC_API_REGBASE_COPY_SIGN_H__

using namespace AscendC;

template <typename T>
__aicore__ inline void CopySignImplVF(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint32_t calc_cnt) {
  uint32_t vl_size = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vl_size));

  MicroAPI::RegTensor<T> src0_reg, src1_reg, dst_reg;
  MicroAPI::RegTensor<T> get_sign_reg, clear_sign_reg;
  MicroAPI::MaskReg full_mask = MicroAPI::CreateMask<uint8_t>();
  if (sizeof(T) == 2) {
    MicroAPI::Duplicate(get_sign_reg, 0x8000, full_mask);
    MicroAPI::Duplicate(clear_sign_reg, 0x7FFF, full_mask);
  } else if (sizeof(T) == 4) {
    MicroAPI::Duplicate(get_sign_reg, 0x80000000, full_mask);
    MicroAPI::Duplicate(clear_sign_reg, 0x7FFFFFFF, full_mask);
  }

  MicroAPI::MaskReg mask;
  for (uint16_t i = 0U; i < repeat_time; ++i) {
    mask = AscendC::MicroAPI::UpdateMask<T>(calc_cnt);
    MicroAPI::LoadAlign(src0_reg, src0 + i * vl_size);
    MicroAPI::LoadAlign(src1_reg, src1 + i * vl_size);

    // 提取 src1 的符号位
    MicroAPI::And(src1_reg, src1_reg, get_sign_reg, mask);
    // 清除 src0 的符号位
    MicroAPI::And(src0_reg, src0_reg, clear_sign_reg, mask);
    // 将 src1 的符号位合并到 dst 上
    MicroAPI::Or(dst_reg, src0_reg, src1_reg, mask);
    // 存储 dst
    MicroAPI::StoreAlign(dst + i * vl_size, dst_reg, mask);
  }
}

template <typename T>
__aicore__ inline void CopySignExtend(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                      const uint32_t calc_cnt) {
  if (sizeof(T) == 2) {
    LocalTensor<uint16_t> dst_ext = dst.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> src0_ext = src0.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> src1_ext = src1.template ReinterpretCast<uint16_t>();
    VF_CALL<CopySignImplVF<uint16_t>>((__ubuf__ uint16_t*)dst_ext.GetPhyAddr(),
                                      (__ubuf__ uint16_t*)src0_ext.GetPhyAddr(),
                                      (__ubuf__ uint16_t*)src1_ext.GetPhyAddr(), calc_cnt);
  } else if (sizeof(T) == 4) {
    LocalTensor<uint32_t> dst_ext = dst.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> src0_ext = src0.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> src1_ext = src1.template ReinterpretCast<uint32_t>();
    VF_CALL<CopySignImplVF<uint32_t>>((__ubuf__ uint32_t*)dst_ext.GetPhyAddr(),
                                      (__ubuf__ uint32_t*)src0_ext.GetPhyAddr(),
                                      (__ubuf__ uint32_t*)src1_ext.GetPhyAddr(), calc_cnt);
  }
}
#endif
