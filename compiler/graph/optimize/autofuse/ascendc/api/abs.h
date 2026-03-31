/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to License for details. You may not use this file except in compliance with License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ASCENDC_API_ABS_H__
#define __ASCENDC_API_ABS_H__

/**
* @brief 扩展的绝对值计算接口，支持int32等整型数据类型，支持原地计算
*
* @tparam T 数据类型，支持int32_t、half、float
* @param dst 目的操作数tensor
* @param src 源操作数tensor
* @param size 参与计算的元素个数
* @param tmp_buf 临时buffer，用于类型转换和中间存储，原地计算时用于暂存原始数据
*
* @note tmp_buf大小必须 >= size * sizeof(T)，用于原地计算时的数据暂存
* @note int32类型实现方式：使用Not(x) + 1（负数变正）然后用Max选择正负数的正确结果
* @note 支持原地计算：当dst和src指向同一块内存时，先将数据复制到tmp_buf，计算后再写回
*/
template <typename T>
inline __aicore__ void AbsExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  const uint32_t size, AscendC::LocalTensor<uint8_t> &tmp_buf) {
    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, float>::value || std::is_same<T, half>::value,
                "Unsupported data type for AbsExtend");
    if constexpr (AscendC::IsSameType<T, int32_t>::value) {
        if (dst.GetPhyAddr() == src.GetPhyAddr()) {
            AscendC::LocalTensor<int16_t> tmp_buf_i16 = tmp_buf.template ReinterpretCast<int16_t>();
            AscendC::LocalTensor<int32_t> tmp_buf_i32 = tmp_buf.template ReinterpretCast<int32_t>();
            AscendC::Not(tmp_buf_i16, src.template ReinterpretCast<int16_t>(), size * 2);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp_buf_i32, tmp_buf_i32, 1, size);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(dst, src, tmp_buf_i32, size);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<int16_t> dst_i16 = dst.template ReinterpretCast<int16_t>();
            AscendC::Not(dst_i16, src.template ReinterpretCast<int16_t>(), size * 2);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(dst, dst, 1, size);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(dst, dst, src, size);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) {
        // float/half类型：直接调用AscendC原生Abs接口
        AscendC::Abs(dst, src, size);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

#endif  // __ASCENDC_API_ABS_H__