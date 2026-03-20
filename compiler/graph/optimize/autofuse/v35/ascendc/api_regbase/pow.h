/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_POW_H__
#define __ASCENDC_API_POW_H__

static constexpr AscendC::PowerConfig pow_config = {PowerAlgo::DOUBLE_FLOAT_TECH};

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                           const AscendC::LocalTensor<T> &src2, const uint32_t calCount,
                           AscendC::LocalTensor<uint8_t> &tmp_buf) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1, const T &src2,
                           const uint32_t calCount, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  if(static_cast<float>(src2) == 0.0f) {
    Duplicate(dst, static_cast<T>(1.0), calCount);
  } else if (static_cast<float>(src2) == 1.0f) {
    DataCopy(dst, src1, calCount);
  } else if (static_cast<float>(src2) == -1.0f) {
    Reciprocal(dst, src1, calCount);
  } else if (static_cast<float>(src2) == 2.0f) {
    Mul(dst, src1, src1, calCount);
  } else if (static_cast<float>(src2) == 0.5f) {
    Sqrt(dst, src1, calCount);
  } else if (static_cast<float>(src2) == -0.5f) {
    Rsqrt(dst, src1, calCount);
  } else if (static_cast<float>(src2) == 3.0f) {
    Mul(dst, src1, src1, calCount);
    Mul(dst, dst, src1, calCount);
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const AscendC::LocalTensor<T> &src2,
                           const uint32_t calCount, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const T &src2, const uint32_t calCount,
                           AscendC::LocalTensor<uint8_t> &tmp_buf) {
  auto block_cnt = KernelUtils::BlkSize<T>();
  // 将src1扩充为一个blockSize的tensor
  LocalTensor<T> src1_buf = tmp_buf.template ReinterpretCast<T>();
  src1_buf.SetSize(block_cnt);
  LocalTensor<T> dst_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<T>();
  dst_buf.SetSize(block_cnt);

  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[2*ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src1_buf, src1, block_cnt);
  // 调用Power基础API：tensor(block size) + scalar
  if(static_cast<float>(src2) == 0.0f) {
    Duplicate(dst_buf, static_cast<T>(1.0), calCount);
  } else if (static_cast<float>(src2) == 1.0f) {
    DataCopy(dst_buf, src1_buf, calCount);
  } else if (static_cast<float>(src2) == -1.0f) {
    Reciprocal(dst_buf, src1_buf, calCount);
  } else if (static_cast<float>(src2) == 2.0f) {
    Mul(dst_buf, src1_buf, src1_buf, calCount);
  } else if (static_cast<float>(src2) == 0.5f) {
    Sqrt(dst_buf, src1_buf, calCount);
  } else if (static_cast<float>(src2) == -0.5f) {
    Rsqrt(dst_buf, src1_buf, calCount);
  } else if (static_cast<float>(src2) == 3.0f) {
    Mul(dst_buf, src1_buf, src1_buf, calCount);
    Mul(dst_buf, dst_buf, src1_buf, calCount);
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst_buf, src1_buf, src2, left_tmp_buf, block_cnt);
  } else {
    Power(dst_buf, src1_buf, src2, left_tmp_buf, block_cnt);
  }
  // 取block tensor中的一个scalar元素，扩充为dst size长度的tensor
  Duplicate(dst, dst_buf.GetValue(0), dst.GetSize());
}
#endif  // __ASCENDC_API_POW_H__
