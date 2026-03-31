
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_TRUE_DIV_H__
#define __ASCENDC_API_TRUE_DIV_H__

template <typename T, typename U>
inline __aicore__ void TrueDivExtend(const AscendC::LocalTensor<U> &dst, const AscendC::LocalTensor<T> &src1,
                                     const AscendC::LocalTensor<T> &src2, const uint32_t size,
                                     AscendC::LocalTensor<uint8_t> &tmp_buf) {
  static_assert(std::is_same<T, int32_t>::value, "Unsupported data type for TrueDivExtend");
  constexpr uint32_t cast_dst_align = 32U / sizeof(U);
  uint32_t cast_buf_size = (size + cast_dst_align - 1) / cast_dst_align * cast_dst_align;

  AscendC::LocalTensor<U> cast_src1 = tmp_buf.template ReinterpretCast<U>();
  uint32_t offset = cast_buf_size * sizeof(U);
  AscendC::LocalTensor<U> cast_src2 = tmp_buf[offset].template ReinterpretCast<U>();

  AscendC::Cast(cast_src1, src1, AscendC::RoundMode::CAST_RINT, size);
  AscendC::Cast(cast_src2, src2, AscendC::RoundMode::CAST_RINT, size);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Div(dst, cast_src1, cast_src2, size);
}

template <typename T>
inline __aicore__ void TrueDivExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                    const AscendC::LocalTensor<T> &src2, const uint32_t size) {
    AscendC::Div(dst, src1, src2, size);
}

#endif  // __ASCENDC_API_TRUE_DIV_H__
