/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_EXP2_H__
#define __ASCENDC_API_EXP2_H__

static constexpr AscendC::PowerConfig exp2_config = {PowerAlgo::DOUBLE_FLOAT_TECH};

template <typename T>
inline __aicore__ void Exp2(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  if constexpr (AscendC::SupportType<T, float, half, bfloat16_t>()) {
    Power<T, false, exp2_config>(dst, (T)2, src, tmp_buf, calCount);
  } else {
    Power(dst, (T)2, src, tmp_buf, calCount);
  }
}
#endif  // __ASCENDC_API_POW_H__
