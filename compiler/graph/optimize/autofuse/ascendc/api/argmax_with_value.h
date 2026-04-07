/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_ARGMAX_WITH_VALUE_H__
#define __ASCENDC_API_ARGMAX_WITH_VALUE_H__

namespace AscendC {

/**
 * @brief UpdateMaxIndexAndValue - 更新最大值及其索引（tensor版本）
 * @note 使用CompareExtend和Where算子来更新保存的最大值和索引
 * @param maxIndexTemp 当次计算的索引tensor（局部索引，会被修改为加上offset后的值）
 * @param maxValueTemp 当次计算的值tensor
 * @param maxIndexSaved 之前保存的索引tensor（会被更新）
 * @param maxValueSaved 之前保存的值tensor（会被更新）
 * @param offset 索引偏移量（标量），用于将局部索引转换为全局索引
 * @param tmp_buf 临时buffer
 * @param cal_cnt 计算的数据量
 *
 * @note maxIndexTemp 会被原地修改为 maxIndexTemp + offset，调用后不应再使用原始值
 * @note 仅支持 cal_cnt > 1 的 tensor 情况
 */
template <typename T>
inline __aicore__ void UpdateMaxIndexAndValue(const LocalTensor<int64_t> &maxIndexTemp,
                                             const LocalTensor<T> &maxValueTemp,
                                             const LocalTensor<int64_t> &maxIndexSaved,
                                             const LocalTensor<T> &maxValueSaved,
                                             const int64_t offset,
                                             LocalTensor<uint8_t> &tmp_buf,
                                             const uint32_t cal_cnt) {
  // cal_cnt 是元素个数，已经32B对齐

  // 计算 mask 需要的空间（按位，转换为字节数，并对齐到32字节）
  uint32_t mask_size = ((cal_cnt + 7) / 8 + 31) / 32 * 32;

  // 计算 int32 临时索引需要的空间（按字节）
  uint32_t index32_size = cal_cnt * sizeof(int32_t);

  // 1. 分配 int32 临时索引空间（用于计算 offset，因为 Adds 不支持 int64）
  LocalTensor<uint8_t> index32_tmp_uint8 = tmp_buf[0];
  index32_tmp_uint8.SetSize(index32_size);
  LocalTensor<int32_t> index32_tmp = index32_tmp_uint8.ReinterpretCast<int32_t>();

  // 2. 分配 mask 空间（必须独立，不能被复用）
  LocalTensor<uint8_t> mask = tmp_buf[index32_size];
  mask.SetSize(mask_size);

  // 3. 分配 CompareExtend 和 Where 共用的临时空间
  LocalTensor<uint8_t> common_tmp_buf = tmp_buf[index32_size + mask_size];

  // 4. 将 maxIndexTemp 从 int64_t cast 为 int32_t（临时存储）
  // 注意：假设索引值在 int32 范围内
  Cast(index32_tmp, maxIndexTemp, RoundMode::CAST_NONE, cal_cnt);

  // 5. 对 int32_t 临时索引加上 offset（标量加到 tensor）
  Adds(index32_tmp, index32_tmp, static_cast<int32_t>(offset), cal_cnt);

  // 6. 将计算结果 cast 回 int64_t，并原地修改 maxIndexTemp
  Cast(maxIndexTemp, index32_tmp, RoundMode::CAST_NONE, cal_cnt);

  // 7. 使用 CompareExtend 比较 (maxValueTemp > maxValueSaved)，生成 mask
  CompareExtend(mask, maxValueTemp, maxValueSaved, CMPMODE::GT, cal_cnt, common_tmp_buf);

  // 8. 使用 Where 更新 maxValueSaved（复用 common_tmp_buf）
  Where(maxValueSaved, mask, maxValueTemp, maxValueSaved, cal_cnt, common_tmp_buf);

  // 9. 使用 Where 更新 maxIndexSaved（复用 common_tmp_buf）
  Where(maxIndexSaved, mask, maxIndexTemp, maxIndexSaved, cal_cnt, common_tmp_buf);
}

}  // namespace AscendC

#endif  // __ASCENDC_API_ARGMAX_WITH_VALUE_H__
