/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_LOG1P_H__
#define __ASCENDC_API_LOG1P_H__

template <typename T>
inline __aicore__ void Log1p(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src, const uint32_t calCount) {
  AscendC::Adds(src, src, (T)1.0, calCount);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Ln(dst, src, calCount);
}
#endif  // __ASCENDC_API_LOG1P_H__
