/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REMAINDER_H__
#define __ASCENDC_API_REMAINDER_H__

constexpr uint32_t REMAINDER_TMP_BUF_FACTOR = 3U;  // Need 3 temp buffers for div_res, floor_res, mul_res

/**
 * Remainder operation: dst = dividend - floor(dividend / divisor) * divisor
 * Implementation steps:
 * 1. div_res = dividend / divisor
 * 2. floor_res = Cast(div_res, CAST_FLOOR)
 * 3. mul_res = floor_res * divisor
 * 4. dst = dividend - mul_res
 * 
 * Supported data types: float, int32
 * For int32 input, Cast API is used to convert int32 to float for computation,
 * and output is float type.
 * 
 * Note: 
 * - Div API only supports half and float types, not int32
 * - For int32 input, computation is performed in float precision, output is float
 * - Large int32 values (outside [-2^24, 2^24]) may have precision loss when converted to float
 * - Division by zero: Follows IEEE 754 standard behavior:
 *   - dividend > 0, divisor = 0: result = +inf
 *   - dividend < 0, divisor = 0: result = -inf
 *   - dividend = 0, divisor = 0: result = NaN
 *   Caller should ensure divisor does not contain zero values if undefined behavior is not desired.
 * 
 * @tparam T Input data type (float or int32)
 * @tparam DstType Output data type (defaults to float for int32 input, same as T for float input)
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<typename std::conditional<AscendC::IsSameType<T, int32_t>::value, float, T>::type> &dst, const AscendC::LocalTensor<T> &dividend,
                                 const AscendC::LocalTensor<T> &divisor, const AscendC::LocalTensor<uint8_t> &tmp_buf,
                                 const uint32_t calCount) {
  // Static type check: only support float and int32 for input
  static_assert(AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, int32_t>::value,
                "Remainder only supports float and int32 input data types");
  
  // For int32 type: Div API only supports half/float, so we need to use float for computation
  // For float type: use float directly
  using ComputeType = typename std::conditional<AscendC::IsSameType<T, int32_t>::value, float, T>::type;
  constexpr bool isInt32Input = AscendC::IsSameType<T, int32_t>::value;
  
  // Split tmp_buf into 3 parts for intermediate results
  // Note: AscendC::ONE_BLK_SIZE = 32 bytes, buffer should be 32-byte aligned for optimal performance
  uint32_t totalBufSize = tmp_buf.GetSize();
  
  // Calculate aligned buffer size for each of 3 buffers
  // Must be at least AscendC::ONE_BLK_SIZE (32 bytes) for hardware requirements
  uint32_t alignedTotalSize = totalBufSize / AscendC::ONE_BLK_SIZE * AscendC::ONE_BLK_SIZE;
  uint32_t bufSize = alignedTotalSize / REMAINDER_TMP_BUF_FACTOR;
  
  // Ensure bufSize is at least AscendC::ONE_BLK_SIZE to meet hardware alignment requirements
  if (bufSize < AscendC::ONE_BLK_SIZE) {
    // If buffer is too small, we cannot process safely - this indicates caller error
    // For safety, use minimal buffer size (may cause overflow, but caller should ensure enough space)
    bufSize = AscendC::ONE_BLK_SIZE;
  }
  
  uint32_t bufElementCount = bufSize / sizeof(ComputeType);
  
  AscendC::LocalTensor<ComputeType> buf0 = tmp_buf.ReinterpretCast<ComputeType>();
  AscendC::LocalTensor<ComputeType> buf1 = tmp_buf[bufSize].ReinterpretCast<ComputeType>();
  AscendC::LocalTensor<ComputeType> buf2 = tmp_buf[bufSize * 2].ReinterpretCast<ComputeType>();
  
  constexpr uint32_t ONE_RPT_SIZE = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(ComputeType);
  uint32_t maxBufRptNum = bufElementCount / ONE_RPT_SIZE;
  uint32_t maxDoRptNum = AscendC::MAX_REPEAT_TIMES > maxBufRptNum ? maxBufRptNum : AscendC::MAX_REPEAT_TIMES;
  uint32_t maxDoSize = maxDoRptNum * ONE_RPT_SIZE;

  uint32_t doSize = 0;
  uint32_t calcSize = 0;

  // ==================== Float type processing ====================
  if constexpr (!isInt32Input) {
    // For float type, use buffers directly for intermediate results
    AscendC::LocalTensor<ComputeType>& divRes = buf0;
    AscendC::LocalTensor<ComputeType>& floorRes = buf1;
    AscendC::LocalTensor<ComputeType>& mulRes = buf2;
    
    // Process in chunks with max repeat times (only if buffer can hold at least one repeat)
    if (maxDoRptNum > 0 && maxDoSize <= calCount) {
      doSize = maxDoSize;
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<ComputeType, AscendC::MaskMode::NORMAL>(ONE_RPT_SIZE);
      
      for (; calcSize + doSize < calCount; calcSize += doSize) {
        // Step 1: divRes = dividend / divisor
        AscendC::Div<ComputeType, false>(divRes, dividend[calcSize], divisor[calcSize], 
                                         AscendC::MASK_PLACEHOLDER, maxDoRptNum, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        
        // Step 2: floorRes = Cast(divRes, CAST_FLOOR)
        AscendC::Cast<ComputeType, ComputeType, false>(floorRes, divRes, AscendC::RoundMode::CAST_FLOOR,
                                                       AscendC::MASK_PLACEHOLDER, maxDoRptNum, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        
        // Step 3: mulRes = floorRes * divisor
        AscendC::Mul<ComputeType, false>(mulRes, floorRes, divisor[calcSize], 
                                         AscendC::MASK_PLACEHOLDER, maxDoRptNum, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        
        // Step 4: dst = dividend - mulRes
        AscendC::Sub<ComputeType, false>(dst[calcSize], dividend[calcSize], mulRes, 
                                         AscendC::MASK_PLACEHOLDER, maxDoRptNum, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::ResetMask();
    }

    // Process remaining repeats (only if buffer can hold at least one repeat)
    if (bufElementCount >= ONE_RPT_SIZE && calcSize + ONE_RPT_SIZE <= calCount) {
      uint32_t leftRptNum = (calCount - calcSize) / ONE_RPT_SIZE;
      doSize = leftRptNum * ONE_RPT_SIZE;
      
      AscendC::Div(divRes, dividend[calcSize], divisor[calcSize], ONE_RPT_SIZE, leftRptNum, {1, 1, 1, 8, 8, 8});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast<ComputeType, ComputeType>(floorRes, divRes, AscendC::RoundMode::CAST_FLOOR, doSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(mulRes, floorRes, divisor[calcSize], ONE_RPT_SIZE, leftRptNum, {1, 1, 1, 8, 8, 8});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(dst[calcSize], dividend[calcSize], mulRes, ONE_RPT_SIZE, leftRptNum, {1, 1, 1, 8, 8, 8});
      AscendC::PipeBarrier<PIPE_V>();
      
      calcSize += doSize;
    }

    // Process remaining elements (use simple element count API)
    if (calcSize < calCount) {
      uint32_t leftSize = calCount - calcSize;
      
      AscendC::Div(divRes, dividend[calcSize], divisor[calcSize], leftSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(floorRes, divRes, AscendC::RoundMode::CAST_FLOOR, leftSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(mulRes, floorRes, divisor[calcSize], leftSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(dst[calcSize], dividend[calcSize], mulRes, leftSize);
    }
  } 
  // ==================== int32 type processing ====================
  else {
    // For int32 input, use Cast API to convert int32 <-> float
    // Buffer reuse strategy:
    //   buf0 (divRes):    dividendFloat -> floorResult -> subResult
    //   buf1 (divisorFloat): divisorFloat (kept throughout batch)
    //   buf2 (mulRes):    divResult -> mulResult
    //
    // Processing in batches to avoid buffer overflow:
    //   batchSize = min(bufElementCount, calCount)
    //   If bufElementCount == 0, process all data in one batch using calCount
    
    uint32_t batchSize = (bufElementCount > 0) ? bufElementCount : calCount;
    // For DataCopy alignment: float requires 32-byte alignment, so 8 elements
    constexpr uint32_t alignElements = AscendC::ONE_BLK_SIZE / sizeof(ComputeType);  // 8 for float
    
    for (calcSize = 0; calcSize < calCount; calcSize += batchSize) {
      uint32_t currentSize = (calCount - calcSize) > batchSize ? batchSize : (calCount - calcSize);
      
      // Step 1: Cast int32 dividend to float (buf0 = dividendFloat)
      AscendC::Cast(buf0, dividend[calcSize], AscendC::RoundMode::CAST_NONE, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      
      // Step 2: Cast int32 divisor to float (buf1 = divisorFloat)
      AscendC::Cast(buf1, divisor[calcSize], AscendC::RoundMode::CAST_NONE, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      
      // Step 3: divResult = dividendFloat / divisorFloat (buf2 = divResult)
      AscendC::Div(buf2, buf0, buf1, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      
      // Step 4: floorResult = floor(divResult) (reuse buf0)
      AscendC::Cast(buf0, buf2, AscendC::RoundMode::CAST_FLOOR, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      
      // Step 5: mulResult = floorResult * divisorFloat (reuse buf2)
      AscendC::Mul(buf2, buf0, buf1, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      
      // Step 6: subResult = dividendFloat - mulResult
      // Need to re-cast dividend since buf0 was overwritten in Step 4
      AscendC::Cast(buf0, dividend[calcSize], AscendC::RoundMode::CAST_NONE, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(buf0, buf0, buf2, currentSize);
      AscendC::PipeBarrier<PIPE_V>();
      
      // Step 7: Copy float result to dst directly (no cast back to int32)
      // dst is already float type for int32 input, buf0 is also float
      // DataCopy requires 32-byte alignment, handle non-aligned cases
      uint32_t alignedSize = ((currentSize + alignElements - 1) / alignElements) * alignElements;
      AscendC::DataCopy<ComputeType>(dst[calcSize], buf0, alignedSize);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
}

#endif  // __ASCENDC_API_REMAINDER_H__
