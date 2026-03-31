/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REDUCEANY_H__
#define __ASCENDC_API_REDUCEANY_H__

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceAnyExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                      const LocalTensor<uint8_t> &tmp, const uint32_t src_shape[], bool src_inner_pad) {
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(),
                "failed to check the reduce pattern, it only supports AR/RA pattern");
  static_assert(SupportType<T, int32_t>(), "failed to check the data type, current api supports data type is int32_t");

  LocalTensor<float> dst_buff = dst.template ReinterpretCast<float>();
  LocalTensor<float> src_buff = src.template ReinterpretCast<float>();
  AscendC::ReduceAny<float, pattern, isReuseSource>(dst_buff, src_buff, tmp, src_shape, src_inner_pad);
}

#endif  // __ASCENDC_API_REDUCEANY_H__