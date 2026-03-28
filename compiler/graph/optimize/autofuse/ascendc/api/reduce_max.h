/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/**
 * @brief ReduceMaxExtend implementation for int32_t data type.
 *        Uses vmax-based binary reduction since vcgmax does not support int32_t.
 *        Supported types: int32_t
 *        Supported patterns: Pattern::Reduce::AR, Pattern::Reduce::RA
 */
#ifndef __ASCENDC_API_REDUCE_MAX_H_
#define __ASCENDC_API_REDUCE_MAX_H_

#include "kernel_operator_vec_binary_scalar_intf.h"
#include "kernel_operator_vec_brcb_intf.h"
#include "kernel_basic_intf.h"

namespace AscendC {
namespace __AutoFusionApiImpl {

constexpr uint32_t INT32_BYTES = 4;
constexpr uint32_t INT32_ELEM_PER_BLK = ONE_BLK_SIZE / INT32_BYTES;
constexpr uint32_t INT32_ELEM_PER_REPEAT = ONE_REPEAT_BYTE_SIZE / INT32_BYTES;

// Binary reduction across 8 datablocks to get single element
template <typename T>
__aicore__ inline void BinaryReduceAcrossDatablocks(const LocalTensor<T>& brcbDst,
                                                      const BinaryRepeatParams& smallDataParam) {
    SetMaskCount();
    uint32_t halfElems = 4 * INT32_ELEM_PER_BLK;
    SetVectorMask<T, MaskMode::COUNTER>(halfElems);
    Max<T, false>(brcbDst, brcbDst, brcbDst[halfElems], MASK_PLACEHOLDER, 1, smallDataParam);
    PipeBarrier<PIPE_V>();
    
    halfElems = 2 * INT32_ELEM_PER_BLK;
    SetVectorMask<T, MaskMode::COUNTER>(halfElems);
    Max<T, false>(brcbDst, brcbDst, brcbDst[halfElems], MASK_PLACEHOLDER, 1, smallDataParam);
    PipeBarrier<PIPE_V>();
    
    halfElems = INT32_ELEM_PER_BLK;
    SetVectorMask<T, MaskMode::COUNTER>(halfElems);
    Max<T, false>(brcbDst, brcbDst, brcbDst[halfElems], MASK_PLACEHOLDER, 1, smallDataParam);
    PipeBarrier<PIPE_V>();
}

// Reduce elements to single value using Brcb + Max for count <= 8
template <typename T>
__aicore__ inline void BrcbReduceToSingle(const LocalTensor<T>& dst,
                                           const LocalTensor<T>& src,
                                           const LocalTensor<T>& tmpBuffer,
                                           uint32_t count) {
    BinaryRepeatParams smallDataParam;
    smallDataParam.dstBlkStride = 1;
    smallDataParam.src0BlkStride = 1;
    smallDataParam.src1BlkStride = 1;
    smallDataParam.dstRepStride = 1;
    smallDataParam.src0RepStride = 1;
    smallDataParam.src1RepStride = 1;
    UnaryRepeatParams smallUnaryParam;
    smallUnaryParam.dstBlkStride = 1;
    smallUnaryParam.srcBlkStride = 1;
    smallUnaryParam.dstRepStride = 1;
    smallUnaryParam.srcRepStride = 1;

    if (count == 0) {
        return;
    }
    if (count == 1) {
        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(1);
        Adds<T, false>(dst, src, static_cast<T>(0), MASK_PLACEHOLDER, 1, smallUnaryParam);
        PipeBarrier<PIPE_V>();
        return;
    }

    // Prepare 8 elements with padding
    LocalTensor<T> prepareBuffer = tmpBuffer;
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(8);
    Adds<T, false>(prepareBuffer, src, static_cast<T>(0), MASK_PLACEHOLDER, 1, smallUnaryParam);
    PipeBarrier<PIPE_V>();

    // Brcb expand 8 elements to 8 datablocks
    LocalTensor<T> brcbDst = tmpBuffer[8];
    BrcbRepeatParams brcbParams;
    brcbParams.dstBlkStride = 1;
    brcbParams.dstRepStride = 8;
    Brcb<T>(brcbDst, prepareBuffer, 1, brcbParams);
    PipeBarrier<PIPE_V>();
    
    // Binary reduction across datablocks
    BinaryReduceAcrossDatablocks(brcbDst, smallDataParam);
    
    // Copy result to dst
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(1);
    Adds<T, false>(dst, brcbDst, static_cast<T>(0), MASK_PLACEHOLDER, 1, smallUnaryParam);
    PipeBarrier<PIPE_V>();
}

// Binary reduction to single value for count > 8
template <typename T>
__aicore__ inline void BinaryReduceToSingle(const LocalTensor<T>& dst,
                                             const LocalTensor<T>& src,
                                             const LocalTensor<T>& tmpBuffer,
                                             uint32_t count) {
    BinaryRepeatParams defaultParam;
    UnaryRepeatParams smallUnaryParam;
    smallUnaryParam.dstBlkStride = 1;
    smallUnaryParam.srcBlkStride = 1;
    smallUnaryParam.dstRepStride = 1;
    smallUnaryParam.srcRepStride = 1;

    if (count <= 1) {
        if (count == 1) {
            SetMaskCount();
            SetVectorMask<T, MaskMode::COUNTER>(1);
            Adds<T, false>(dst, src, static_cast<T>(0), MASK_PLACEHOLDER, 1, smallUnaryParam);
            PipeBarrier<PIPE_V>();
        }
        return;
    }
    if (count <= 8) {
        BrcbReduceToSingle<T>(dst, src, tmpBuffer, count);
        return;
    }
    
    SetMaskCount();
    LocalTensor<T> currentSrc = src;
    LocalTensor<T> currentDst = tmpBuffer;
    uint32_t currentCount = count;
    bool useTmpDst = true;
    
    while (currentCount > 8) {
        uint32_t halfCount = currentCount / 2;
        SetVectorMask<T, MaskMode::COUNTER>(halfCount);
        LocalTensor<T> src0 = currentSrc;
        LocalTensor<T> src1 = currentSrc[halfCount];
        Max<T, false>(currentDst, src0, src1, MASK_PLACEHOLDER, 1, defaultParam);
        PipeBarrier<PIPE_V>();
        currentCount = halfCount;
        if (useTmpDst) {
            currentSrc = tmpBuffer;
            currentDst = dst;
        } else {
            currentSrc = dst;
            currentDst = tmpBuffer;
        }
        useTmpDst = !useTmpDst;
    }
    
    LocalTensor<T> finalSrc = useTmpDst ? tmpBuffer : dst;
    LocalTensor<T> finalTmp = useTmpDst ? dst : tmpBuffer;
    BrcbReduceToSingle<T>(dst, finalSrc, finalTmp, currentCount);
}

// Process a single row for AR reduction
template <bool isReuseSource>
__aicore__ inline void ReduceMaxARProcessRow(const LocalTensor<int32_t>& rowDst,
                                              const LocalTensor<int32_t>& rowSrc,
                                              const LocalTensor<int32_t>& workBuffer,
                                              uint32_t lastAxis) {
    constexpr uint32_t elePerBlk = INT32_ELEM_PER_BLK;
    BinaryRepeatParams defaultParam;
    UnaryRepeatParams defaultUnaryParam;
    
    if (lastAxis <= elePerBlk) {
        if (lastAxis == 1) {
            SetMaskCount();
            SetVectorMask<int32_t, MaskMode::COUNTER>(1);
            Adds<int32_t, false>(rowDst, rowSrc, static_cast<int32_t>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
            PipeBarrier<PIPE_V>();
        } else {
            BinaryReduceToSingle<int32_t>(rowDst, rowSrc, workBuffer, lastAxis);
        }
    } else {
        uint32_t blkCount = lastAxis / elePerBlk;
        uint32_t blkTail = lastAxis % elePerBlk;
        LocalTensor<int32_t> accumBuffer = workBuffer;
        
        SetMaskCount();
        SetVectorMask<int32_t, MaskMode::COUNTER>(elePerBlk);
        Adds<int32_t, false>(accumBuffer, rowSrc, static_cast<int32_t>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
        PipeBarrier<PIPE_V>();
        
        for (uint32_t blk = 1; blk < blkCount; blk++) {
            LocalTensor<int32_t> blkSrc = rowSrc[blk * elePerBlk];
            SetVectorMask<int32_t, MaskMode::COUNTER>(elePerBlk);
            Max<int32_t, false>(accumBuffer, accumBuffer, blkSrc, MASK_PLACEHOLDER, 1, defaultParam);
            PipeBarrier<PIPE_V>();
        }
        if (blkTail > 0) {
            LocalTensor<int32_t> tailSrc = rowSrc[blkCount * elePerBlk];
            SetVectorMask<int32_t, MaskMode::COUNTER>(blkTail);
            Max<int32_t, false>(accumBuffer, accumBuffer, tailSrc, MASK_PLACEHOLDER, 1, defaultParam);
            PipeBarrier<PIPE_V>();
        }
        BinaryReduceToSingle<int32_t>(rowDst, accumBuffer, workBuffer[elePerBlk], elePerBlk);
    }
    PipeBarrier<PIPE_V>();
}

// Reduce along last axis: [M, N] -> [M]
template <bool isReuseSource = false>
__aicore__ inline void ReduceMaxInt32AR(const LocalTensor<int32_t>& dstTensor,
                                         const LocalTensor<int32_t>& srcTensor,
                                         const LocalTensor<int32_t>& tmpTensor,
                                         uint32_t firstAxis,
                                         uint32_t lastAxis,
                                         uint32_t padLast) {
    constexpr uint32_t elePerBlk = INT32_ELEM_PER_BLK;
    
    LocalTensor<int32_t> rowResults = tmpTensor;
    LocalTensor<int32_t> workBuffer = isReuseSource ? tmpTensor : tmpTensor[firstAxis * elePerBlk];
    LocalTensor<int32_t> finalResStored = isReuseSource ? srcTensor : rowResults;
    uint16_t countStride = isReuseSource ? static_cast<uint16_t>(padLast / elePerBlk) : 1;
    
    for (uint32_t row = 0; row < firstAxis; row++) {
        LocalTensor<int32_t> rowSrc = srcTensor[row * padLast];
        LocalTensor<int32_t> rowDst = isReuseSource ? rowSrc : rowResults[row * elePerBlk];
        ReduceMaxARProcessRow<isReuseSource>(rowDst, rowSrc, workBuffer, lastAxis);
    }
    
    // Gather results to dstTensor
    LocalTensor<uint32_t> patternTensor = workBuffer.template ReinterpretCast<uint32_t>();
    SetMaskCount();
    SetVectorMask<uint32_t, MaskMode::COUNTER>(elePerBlk);
    Duplicate<uint32_t>(patternTensor, 1u, elePerBlk);
    PipeBarrier<PIPE_V>();
    
    GatherMaskParams gatherMaskParam = {1, static_cast<uint16_t>(firstAxis), countStride, 0};
    uint64_t rsvdCnt;
    GatherMask<int32_t, uint32_t>(dstTensor, finalResStored, patternTensor, true, elePerBlk, gatherMaskParam, rsvdCnt);
}

// Reduce along first axis: [M, N] -> [N]
template <bool isReuseSource = false>
__aicore__ inline void ReduceMaxInt32RA(const LocalTensor<int32_t>& dstTensor,
                                         const LocalTensor<int32_t>& srcTensor,
                                         const LocalTensor<int32_t>& tmpTensor,
                                         uint32_t firstAxis,
                                         uint32_t lastAxis,
                                         uint32_t padLast) {
    BinaryRepeatParams defaultParam;
    UnaryRepeatParams defaultUnaryParam;
    LocalTensor<int32_t> currBuff = isReuseSource ? srcTensor : tmpTensor;
    
    uint32_t k = AscendC::Internal::FindClosestPowerOfTwo(firstAxis);
    uint32_t splitK = 1 << k;
    uint32_t remain = firstAxis - splitK;
    
    SetMaskCount();
    if constexpr (isReuseSource) {
        if (remain != 0) {
            SetVectorMask<int32_t, MaskMode::COUNTER>(padLast * remain);
            Max<int32_t, false>(srcTensor, srcTensor, srcTensor[splitK * padLast], MASK_PLACEHOLDER, 1, defaultParam);
            PipeBarrier<PIPE_V>();
        }
    } else {
        if (remain != 0) {
            SetVectorMask<int32_t, MaskMode::COUNTER>(splitK * padLast);
            Adds<int32_t, false>(tmpTensor, srcTensor, static_cast<int32_t>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
            PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t, MaskMode::COUNTER>(padLast * remain);
            Max<int32_t, false>(tmpTensor, tmpTensor, srcTensor[splitK * padLast], MASK_PLACEHOLDER, 1, defaultParam);
            PipeBarrier<PIPE_V>();
        } else if (splitK > 1) {
            splitK >>= 1;
            SetVectorMask<int32_t, MaskMode::COUNTER>(padLast * splitK);
            Max<int32_t, false>(tmpTensor, srcTensor, srcTensor[splitK * padLast], MASK_PLACEHOLDER, 1, defaultParam);
            PipeBarrier<PIPE_V>();
        } else {
            SetVectorMask<int32_t, MaskMode::COUNTER>(lastAxis);
            Adds<int32_t, false>(dstTensor, srcTensor, static_cast<int32_t>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
            PipeBarrier<PIPE_V>();
            return;
        }
    }
    
    while (splitK > 1) {
        splitK >>= 1;
        SetVectorMask<int32_t, MaskMode::COUNTER>(padLast * splitK);
        Max<int32_t, false>(currBuff, currBuff, currBuff[splitK * padLast], MASK_PLACEHOLDER, 1, defaultParam);
        PipeBarrier<PIPE_V>();
    }
    
    SetVectorMask<int32_t, MaskMode::COUNTER>(lastAxis);
    Adds<int32_t, false>(dstTensor, currBuff, static_cast<int32_t>(0), MASK_PLACEHOLDER, 1, defaultUnaryParam);
    PipeBarrier<PIPE_V>();
}

template <class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMaxExtendImpl(const LocalTensor<int32_t>& dstTensor,
                                           const LocalTensor<int32_t>& srcTensor,
                                           const LocalTensor<uint8_t>& sharedTmpBuffer,
                                           const uint32_t srcShape[],
                                           bool srcInnerPad) {
    uint32_t firstAxis = srcShape[0];
    uint32_t lastAxis = srcShape[1];
    constexpr uint32_t elePerBlk = INT32_ELEM_PER_BLK;
    uint32_t padLast = (lastAxis + elePerBlk - 1) / elePerBlk * elePerBlk;
    LocalTensor<int32_t> tmpTensor = sharedTmpBuffer.ReinterpretCast<int32_t>();
    
    if constexpr (IsSameType<pattern, Pattern::Reduce::AR>::value) {
        ReduceMaxInt32AR<isReuseSource>(dstTensor, srcTensor, tmpTensor, firstAxis, lastAxis, padLast);
    } else {
        ReduceMaxInt32RA<isReuseSource>(dstTensor, srcTensor, tmpTensor, firstAxis, lastAxis, padLast);
    }
    
    SetMaskNorm();
    ResetMask();
}

} // namespace __AutoFusionApiImpl
} // namespace AscendC

// Public API
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMaxExtend(const AscendC::LocalTensor<T>& dstTensor,
                                       const AscendC::LocalTensor<T>& srcTensor,
                                       const AscendC::LocalTensor<uint8_t>& sharedTmpBuffer,
                                       const uint32_t srcShape[],
                                       bool srcInnerPad = true) {
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(AscendC::IsSameType<T, int32_t>::value,
        "ReduceMaxExtend: unsupported data type! Only int32_t is supported.");
    static_assert(AscendC::SupportType<pattern, AscendC::Pattern::Reduce::AR, AscendC::Pattern::Reduce::RA>(),
        "ReduceMaxExtend: unsupported pattern! Only AR and RA patterns are supported.");
    AscendC::__AutoFusionApiImpl::ReduceMaxExtendImpl<pattern, isReuseSource>(
        dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad);
}

#endif // IMPL_REDUCE_MAX_EXTEND_H_
