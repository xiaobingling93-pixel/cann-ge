/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_LOGICAL_H__
#define __ASCENDC_API_REGBASE_LOGICAL_H__

template <typename T>
inline __aicore__ void LogicalOrExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                       const LocalTensor<T> &src2, const uint32_t size) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    AscendC::LogicalOr(dst_tmp, src1, src2, size);
}

template <typename T>
inline __aicore__ void LogicalAndExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                        const LocalTensor<T> &src2, const uint32_t size) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    AscendC::LogicalAnd(dst_tmp, src1, src2, size);
}

template <typename T>
inline __aicore__ void LogicalXorExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                        const LocalTensor<T> &src2, const uint32_t size) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    AscendC::LogicalXor(dst_tmp, src1, src2, size);
}

template <typename T>
inline __aicore__ void LogicalOrExtends(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                        const T src2, const uint32_t size) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    AscendC::LogicalOrs(dst_tmp, src1, src2, size);
}

template <typename T>
inline __aicore__ void LogicalAndExtends(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                        const T src2, const uint32_t size) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    AscendC::LogicalAnds(dst_tmp, src1, src2, size);
}

template <typename T>
inline __aicore__ void LogicalOrExtends(const LocalTensor<uint8_t> &dst, const T src1, const T src2) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    uint8_t res = src1 || src2;
    AscendC::Duplicate(dst, res, dst.GetSize());
}

template <typename T>
inline __aicore__ void LogicalAndExtends(const LocalTensor<uint8_t> &dst, const T src1, const T src2) {
    auto dst_tmp = dst.template ReinterpretCast<bool>();
    uint8_t res = src1 && src2;
    AscendC::Duplicate(dst, res, dst.GetSize());
}


#endif  // __ASCENDC_API_REGBASE_LOGICAL_H__