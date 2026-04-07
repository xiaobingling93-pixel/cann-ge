/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_REMAINDER_H
#define __ASCENDC_API_REGBASE_REMAINDER_H

// Remainder(x1, x2) = x1 - x2 * trunc(x1/x2)

template <typename T>
__aicore__ inline void RemainderImplVF(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2,
                                        uint32_t count, uint16_t repeat_time) {
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
    AscendC::MicroAPI::RegTensor<T> srcVreg1;
    AscendC::MicroAPI::RegTensor<T> srcVreg2;
    AscendC::MicroAPI::RegTensor<T> dstVreg;
    AscendC::MicroAPI::MaskReg mask;
    AscendC::MicroAPI::RegTensor<T> quotient;
    AscendC::MicroAPI::RegTensor<T> truncatedQuotient;
    AscendC::MicroAPI::RegTensor<T> mulResult;

    for (uint16_t i = 0; i < repeat_time; i++) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
        AscendC::MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);

        // Calculate x1 / x2
        AscendC::MicroAPI::Div(quotient, srcVreg1, srcVreg2, mask);

        // Truncate to floor (trunc towards zero)
        AscendC::MicroAPI::Truncate<T, AscendC::RoundMode::CAST_FLOOR>(truncatedQuotient, quotient, mask);

        // Calculate x2 * trunc(x1/x2)
        AscendC::MicroAPI::Mul(mulResult, srcVreg2, truncatedQuotient, mask);

        // Calculate remainder: x1 - x2 * trunc(x1/x2)
        AscendC::MicroAPI::Sub(dstVreg, srcVreg1, mulResult, mask);

        AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <typename T>
__aicore__ inline void RemainderExtend(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src1,
                                   const AscendC::LocalTensor<T>& src2, const uint32_t size) {
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
    uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(size, oneRepElm));
    VF_CALL<RemainderImplVF<T>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src1.GetPhyAddr(),
                                     (__ubuf__ T*)src2.GetPhyAddr(), size, repeat_time);
}

#endif  // __ASCENDC_API_REGBASE_REMAINDER_H
