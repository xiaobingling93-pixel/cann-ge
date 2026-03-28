/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void trunc_bf16_test(GM_ADDR x0, GM_ADDR output, GM_ADDR workspace, GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_TruncBf16_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_TruncBf16_Code, CalculateCorrect){
 auto test_shape = GetParam();

 uint64_t block_dim = 48;

 int test_size = test_shape[0] * test_shape[1];

 AutofuseTilingData tiling_data;
 bfloat16_t* x = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
 bfloat16_t* y = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
 bfloat16_t *expect = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);

 // Prepare test and expect data
 for (int i = 0; i < test_size; i++) {
   // 使用包含小数部分的数值，测试 Trunc 截断功能
   float val = (i - 5.0f) / 2.0f;  // 产生包含正负小数的数
   x[i] = static_cast<bfloat16_t>(val);
   expect[i] = static_cast<bfloat16_t>(std::trunc(val));
 }

 // Launch
 uint32_t ws_size = 0;
 AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
 printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

 AscendC::SetKernelMode(KernelMode::AIV_MODE);
 ICPU_RUN_KF(trunc_bf16_test, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

 // Count difference
 uint32_t diff_count = 0;
 for (int i = 0; i < test_size; i++) {
   auto diff = (double)(static_cast<float>(y[i]) - static_cast<float>(expect[i]));
   if(diff < -1e-5 || diff > 1e-5) {
     diff_count++;
   }
 }

 EXPECT_EQ(diff_count, 0) << " of " << test_size;

 AscendC::GmFree(x);
 AscendC::GmFree(y);
 AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_TruncBf16_Code,
                        ::testing::Values(std::vector<int>{2,8}));

}
