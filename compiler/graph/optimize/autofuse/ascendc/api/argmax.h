/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_ARGMAX_H__
#define __ASCENDC_API_ARGMAX_H__

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

namespace AscendC {
// AR, float, R less than one repeat(64)
template <typename T, bool isReuseSource = false>
inline __aicore__ void ArgmaxLEOneRepeat(const LocalTensor<int64_t>& dstTensor, const LocalTensor<T>& srcTensor, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t repeatTimes, const uint32_t last) 
{
    LocalTensor<float> tailTmp = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<uint8_t> tmpMask = sharedTmpBuffer[256];
    LocalTensor<float> dstTensorLocal = sharedTmpBuffer[512].ReinterpretCast<float>();

    uint32_t copyNum = last;

    for (uint32_t i = 0; i < repeatTimes; ++i) {
        // tailTmp used for ±0 first, then process tail
        if constexpr (std::is_same_v<T, float>) {
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(tailTmp, 0.0f, 64);
            AscendC::PipeBarrier<PIPE_V>();
            Compare(tmpMask, srcTensor, tailTmp, AscendC::CMPMODE::NE, 64);
            AscendC::PipeBarrier<PIPE_V>();
            Select(srcTensor, tmpMask, srcTensor, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(tailTmp, -INFINITY, 64);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::PipeBarrier<PIPE_V>();
            DataCopy(tailTmp, srcTensor[i * last], copyNum);
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (std::is_same_v<T, int32_t>) {
            AscendC::PipeBarrier<PIPE_V>();
            Cast<float, int32_t>(tailTmp, srcTensor[i * last], AscendC::RoundMode::CAST_NONE, copyNum);
            AscendC::PipeBarrier<PIPE_V>();
        }
        WholeReduceMax<float, true>(dstTensorLocal, tailTmp, 64, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_INDEX);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        uint32_t index = *reinterpret_cast<__ubuf__ uint32_t*>(&dstTensorLocal(0));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        dstTensor(i) = static_cast<int64_t>(index);
    }

    // add for test, fill illegal index dstTensor with 0
    // if u won't access dstTensor after this, u can remove this loop
    for (uint32_t i = repeatTimes; i < dstTensor.GetSize(); ++i) {
        dstTensor(i) = static_cast<int64_t>(0);
    }
}

// DONE: cast dstTensor to int64_t, current dstTensor is float
// AR/RA, float, R greater than one repeat(64)
template <typename T, class pattern, bool isReuseSource = false>
__aicore__ void ArgmaxGTOneRepeat(const LocalTensor<int64_t>& dstTensor, const LocalTensor<T>& srcTensor, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t first, const uint32_t last, const uint32_t repeatTimesOneRow) 
{
    LocalTensor<float> tmpLocal = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<float> maxLocal = sharedTmpBuffer[256].ReinterpretCast<float>();
    LocalTensor<float> intTmp = sharedTmpBuffer[512].ReinterpretCast<float>();
    LocalTensor<float> dupIndexTensor = sharedTmpBuffer[768].ReinterpretCast<float>();
    LocalTensor<uint8_t> tmpMask = sharedTmpBuffer[1024].ReinterpretCast<uint8_t>();
    LocalTensor<float> maxBefore = sharedTmpBuffer[1280].ReinterpretCast<float>();
    LocalTensor<float> maxAfter = sharedTmpBuffer[1536].ReinterpretCast<float>();
    LocalTensor<float> tailTmp = sharedTmpBuffer[1792].ReinterpretCast<float>();
    LocalTensor<float> tmpdstTensor = sharedTmpBuffer[2048].ReinterpretCast<float>();

    uint32_t tailNum = last % 64;
    tailNum = (tailNum == 0) ? 64 : tailNum;
    uint32_t copyNum = (tailNum == 0) ? 64 : tailNum;
    uint32_t tailCtrl = 1; // force tail process

    for (uint32_t i = 0; i < first; ++i) {
        Duplicate(tmpdstTensor, 0.0f, tmpdstTensor.GetSize());
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(maxBefore, srcTensor[i * last], 64);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            Cast<float, int32_t>(maxBefore, srcTensor[i * last], AscendC::RoundMode::CAST_NONE, 64);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < repeatTimesOneRow - 2; ++j) {
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(dupIndexTensor, (j + 1) * 64.0f, 64);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (std::is_same_v<T, float>) {
                Max(maxAfter, maxBefore, srcTensor[i * last + (j + 1) * 64], 64);
            } else if constexpr (std::is_same_v<T, int32_t>) {
                Cast<float, int32_t>(intTmp, srcTensor[i * last + (j + 1) * 64], AscendC::RoundMode::CAST_NONE, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Max(maxAfter, maxBefore, intTmp, 64);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Compare(tmpMask, maxAfter, maxBefore, AscendC::CMPMODE::NE, 64);
            AscendC::PipeBarrier<PIPE_V>();
            Select(tmpdstTensor, tmpMask, dupIndexTensor, tmpdstTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
            AscendC::PipeBarrier<PIPE_V>();
            DataCopy(maxBefore, maxAfter, 64);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_V>();
        // tail process
        for (uint32_t j = 0; j < tailCtrl; ++j) {
            Duplicate(tailTmp, -INFINITY, 64);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (std::is_same_v<T, float>) {
                DataCopy(tailTmp, srcTensor[(i + 1) * last - copyNum], copyNum);
            } else if constexpr (std::is_same_v<T, int32_t>) {
                Cast<float, int32_t>(intTmp, srcTensor[(i + 1) * last - copyNum], AscendC::RoundMode::CAST_NONE, copyNum);
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(tailTmp, intTmp, copyNum);
            }
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(dupIndexTensor, (repeatTimesOneRow - 1) * 64.0f, 64);
            AscendC::PipeBarrier<PIPE_V>();
            Max(maxAfter, maxBefore, tailTmp, 64);
            AscendC::PipeBarrier<PIPE_V>();
            Compare(tmpMask, maxAfter, maxBefore, AscendC::CMPMODE::NE, 64);
            AscendC::PipeBarrier<PIPE_V>();
            Select(tmpdstTensor, tmpMask, dupIndexTensor, tmpdstTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_V>();
        WholeReduceMax<float, true>(tmpLocal, maxAfter, 64, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        float maxValue = tmpLocal(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Compares(tmpMask, maxAfter, maxValue, AscendC::CMPMODE::EQ, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        uint64_t minMask[1] = {(static_cast<uint64_t>(tmpMask(0)) << 0)
                            | (static_cast<uint64_t>(tmpMask(1)) << 8)
                            | (static_cast<uint64_t>(tmpMask(2)) << 16)
                            | (static_cast<uint64_t>(tmpMask(3)) << 24)
                            | (static_cast<uint64_t>(tmpMask(4)) << 32)
                            | (static_cast<uint64_t>(tmpMask(5)) << 40)
                            | (static_cast<uint64_t>(tmpMask(6)) << 48)
                            | (static_cast<uint64_t>(tmpMask(7)) << 56)};
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        WholeReduceMin<float, true>(maxLocal, tmpdstTensor, minMask, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        uint32_t relative_index = *reinterpret_cast<__ubuf__ uint32_t*>(&maxLocal(0));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        float finalIndex = tmpdstTensor(relative_index) + relative_index * 1.0f;
        dstTensor(i) = static_cast<int64_t>(finalIndex);
    }

    // add for test, fill illegal index dstTensor with 0
    // if u won't access dstTensor after this, you can remove this loop
    if constexpr (SupportType<pattern, Pattern::Reduce::AR>()) {
        for (uint32_t i = first; i < dstTensor.GetSize(); ++i) {
            dstTensor(i) = static_cast<int64_t>(0);
        }
    }
}

template <typename T, class pattern, bool isReuseSource = false>
__aicore__ void ArgmaxRA(const LocalTensor<int64_t>& dstTensor, const LocalTensor<T>& srcTensor, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t first, const uint32_t last, const uint32_t repeatTimesOneRow) 
{
    LocalTensor<float> tmpLocal = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<float> maxLocal = sharedTmpBuffer[256].ReinterpretCast<float>();
    LocalTensor<float> intTmp = sharedTmpBuffer[512].ReinterpretCast<float>();
    LocalTensor<float> dupIndexTensor = sharedTmpBuffer[768].ReinterpretCast<float>();
    LocalTensor<uint8_t> tmpMask = sharedTmpBuffer[1024].ReinterpretCast<uint8_t>();
    LocalTensor<float> maxBefore = sharedTmpBuffer[1280].ReinterpretCast<float>();
    LocalTensor<float> maxAfter = sharedTmpBuffer[1536].ReinterpretCast<float>();
    LocalTensor<float> tailTmp = sharedTmpBuffer[1792].ReinterpretCast<float>();
    LocalTensor<float> tmpdstTensor = sharedTmpBuffer[2048].ReinterpretCast<float>();
    
    Duplicate(tmpdstTensor, 0.0f, tmpdstTensor.GetSize());

    uint32_t tailNum = last % 64;
    tailNum = (tailNum == 0) ? 64 : tailNum;
    uint32_t copyNum = (tailNum == 0) ? 64 : tailNum;
    uint32_t tailCtrl = 1; // force tail process
    uint32_t tailOffset = last - tailNum;

    if (first == 1) {
        Duplicate(tmpdstTensor, 0.0f, dstTensor.GetSize());
        AscendC::PipeBarrier<PIPE_V>();
        Cast<int64_t, float>(dstTensor, tmpdstTensor, AscendC::RoundMode::CAST_RINT, dstTensor.GetSize());
    } else {
        for (uint32_t i = 0; i < repeatTimesOneRow - 1; ++i) {
            Duplicate(tmpdstTensor, 0.0f, 64);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (std::is_same_v<T, int32_t>) {
                Cast<float, int32_t>(intTmp, srcTensor[i * 64], AscendC::RoundMode::CAST_RINT, 64);
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(maxBefore, intTmp, 64);
            } else if constexpr (std::is_same_v<T, float>) {
                DataCopy(maxBefore, srcTensor[i * 64], 64);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < first - 1; ++j) {
                Duplicate(dupIndexTensor, (j + 1) * 1.0f, 64);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (std::is_same_v<T, int32_t>) {
                    Cast<float, int32_t>(intTmp, srcTensor[i * 64 + (j + 1) * last], AscendC::RoundMode::CAST_RINT, 64);
                    AscendC::PipeBarrier<PIPE_V>();
                    Max(maxAfter, maxBefore, intTmp, 64);
                } else if constexpr (std::is_same_v<T, float>) {
                    Max(maxAfter, maxBefore, srcTensor[i * 64 + (j + 1) * last], 64);
                }
                AscendC::PipeBarrier<PIPE_V>();
                Compare(tmpMask, maxAfter, maxBefore, AscendC::CMPMODE::NE, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Select(tmpdstTensor, tmpMask, dupIndexTensor, tmpdstTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(maxBefore, maxAfter, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Cast<int64_t, float>(dstTensor[i * 64], tmpdstTensor, AscendC::RoundMode::CAST_RINT, 64);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t j = 0; j < tailCtrl; ++j) {
            Duplicate(tmpdstTensor, 0.0f, 64);
            Duplicate(tailTmp, -INFINITY, 64);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (std::is_same_v<T, int32_t>) {
                Cast<float, int32_t>(intTmp, srcTensor[tailOffset], AscendC::RoundMode::CAST_NONE, tailNum);
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(tailTmp, intTmp, tailNum);
            } else if constexpr (std::is_same_v<T, float>) {
                DataCopy(tailTmp, srcTensor[tailOffset], tailNum);
            }
            AscendC::PipeBarrier<PIPE_V>(); 
            DataCopy(maxBefore, tailTmp, 64);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t k = 0; k < first - 1; ++k) {
                AscendC::PipeBarrier<PIPE_V>();
                Duplicate(dupIndexTensor, (k + 1) * 1.0f, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Duplicate(tailTmp, -INFINITY, 64);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (std::is_same_v<T, int32_t>) {
                    Cast<float, int32_t>(intTmp, srcTensor[(k + 1) * last + tailOffset], AscendC::RoundMode::CAST_NONE, tailNum);
                    AscendC::PipeBarrier<PIPE_V>();
                    DataCopy(tailTmp, intTmp, tailNum);
                } else if constexpr (std::is_same_v<T, float>) {
                    DataCopy(tailTmp, srcTensor[(k + 1) * last + tailOffset], tailNum);
                }
                AscendC::PipeBarrier<PIPE_V>();
                Max(maxAfter, maxBefore, tailTmp, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Compare(tmpMask, maxAfter, maxBefore, AscendC::CMPMODE::NE, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Select(tmpdstTensor, tmpMask, dupIndexTensor, tmpdstTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(maxBefore, maxAfter, 64);
                AscendC::PipeBarrier<PIPE_V>();
                Cast<int64_t, float>(dstTensor[tailOffset], tmpdstTensor, AscendC::RoundMode::CAST_RINT, 64);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template<typename T, typename U, class pattern>
__aicore__ inline void ArgMaxExtend(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(std::is_same_v<U, float> || std::is_same_v<U, int32_t>, "ArgMaxExtend src only support float and int32_t");
    static_assert(std::is_same_v<T, int64_t>, "ArgMaxExtend dst only support int64_t dst type");
    static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(), "failed to check Reduce pattern, only support AR/RA pattern!");
    uint32_t first = srcShape[0];
    uint32_t last = srcShape[1];

    if constexpr (SupportType<pattern, Pattern::Reduce::AR>()) {
        if constexpr (std::is_same_v<U, float>) {
            if (last <= 64) {
                uint32_t repeatTimes = first;
                ArgmaxLEOneRepeat<float, false>(dst, src, sharedTmpBuffer, repeatTimes, last);
            } else {
                uint32_t repeatTimesOneRow = CeilDivision(last, 64);
                ArgmaxGTOneRepeat<float, Pattern::Reduce::AR, false>(dst, src, sharedTmpBuffer, first, last, repeatTimesOneRow);
            }
        } else if constexpr (std::is_same_v<U, int32_t>) {
            if (last <= 64) {
                uint32_t repeatTimes = first;
                ArgmaxLEOneRepeat<int32_t, false>(dst, src, sharedTmpBuffer, repeatTimes, last);
            } else {
                uint32_t repeatTimesOneRow = CeilDivision(last, 64);
                ArgmaxGTOneRepeat<int32_t, Pattern::Reduce::AR, false>(dst, src, sharedTmpBuffer, first, last, repeatTimesOneRow);
            }
        }
    } else if constexpr (SupportType<pattern, Pattern::Reduce::RA>()) {
        if constexpr (std::is_same_v<U, float>) {
                uint32_t repeatTimesOneRow = CeilDivision(last, 64);
                ArgmaxRA<float, Pattern::Reduce::RA, false>(dst, src, sharedTmpBuffer, first, last, repeatTimesOneRow);
        } else if constexpr (std::is_same_v<U, int32_t>) {
                uint32_t repeatTimesOneRow = CeilDivision(last, 64);
                ArgmaxRA<int32_t, Pattern::Reduce::RA, false>(dst, src, sharedTmpBuffer, first, last, repeatTimesOneRow);
        }
    }
}

} // namespace AscendC

#endif  // __ASCENDC_API_ARGMAX_H__