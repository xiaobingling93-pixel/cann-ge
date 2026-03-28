/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void frexp_float_test(GM_ADDR x, GM_ADDR y, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

namespace {
class E2E_FrexpFloat_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_FrexpFloat_Code, CalculateCorrect){
 auto test_shape = GetParam();

 uint64_t block_dim = 48;

 int test_size = test_shape[0] * test_shape[1];

 AutofuseTilingData tiling_data;
 float* x = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
 float* mantissa = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
 float* exponent = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
 float* expect_mantissa = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
 float* expect_exponent = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

 for (int i = 0; i < test_size; i++) {
   x[i] = static_cast<float>(i) / test_size * 20.0f - 10.0f;
   int exp;
   expect_mantissa[i] = std::frexp(x[i], &exp);
   expect_exponent[i] = static_cast<float>(exp);
 }

 uint32_t ws_size = 0;
 AutofuseTiling(test_shape[0], test_shape[1], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
 printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

 AscendC::SetKernelMode(KernelMode::AIV_MODE);
 ICPU_RUN_KF(frexp_float_test, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)mantissa, (uint8_t *)exponent, nullptr, (uint8_t *)&tiling_data);

 uint32_t diff_count = 0;
 for (int i = 0; i < test_size; i++) {
   auto diff_mantissa = (double)(mantissa[i] - expect_mantissa[i]);
   auto diff_exponent = (double)(exponent[i] - expect_exponent[i]);
   if(diff_mantissa < -1e-5 || diff_mantissa > 1e-5) {
       printf("mantissa diff at index %d: x: %f, mantissa: %f, expect: %f, diff: %f\n", i, static_cast<float>(x[i]),
              static_cast<float>(mantissa[i]),
              static_cast<float>(expect_mantissa[i]), diff_mantissa);
       diff_count++;
   }
   if(diff_exponent < -1e-5 || diff_exponent > 1e-5) {
       printf("exponent diff at index %d: x: %f, exponent: %f, expect: %f, diff: %f\n", i, static_cast<float>(x[i]),
              static_cast<float>(exponent[i]),
              static_cast<float>(expect_exponent[i]), diff_exponent);
       diff_count++;
   }
 }

 EXPECT_EQ(diff_count, 0) << " of " << test_size;

 AscendC::GmFree(x);
 AscendC::GmFree(mantissa);
 AscendC::GmFree(exponent);
 AscendC::GmFree(expect_mantissa);
 AscendC::GmFree(expect_exponent);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_FrexpFloat_Code,
                        ::testing::Values(std::vector<int>{2,8,16}));

}
