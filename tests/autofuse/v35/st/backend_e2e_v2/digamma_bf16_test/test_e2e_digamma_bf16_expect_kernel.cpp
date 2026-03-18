/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void digamma_bf16_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

// Digamma function implementation (ψ function) - derivative of log-gamma
static double digamma(double x) {
  const double epsilon = 1e-6;
  const double euler_mascheroni = 0.577215664901532860606512090082;
  double result = 0.0;

  if (x < epsilon) {
    return -euler_mascheroni - 1.0 / x;
  }

  // Use asymptotic expansion for large x
  while (x < 10.0) {
    result -= 1.0 / x;
    x += 1.0;
  }

  // Asymptotic expansion: ψ(x) ≈ ln(x) - 1/(2x) + 1/(12x²) - 1/(120x⁴) + ...
  double inv_x = 1.0 / x;
  double inv_x2 = inv_x * inv_x;
  result += std::log(x) - 0.5 * inv_x + inv_x2 / 12.0 - inv_x2 * inv_x2 / 120.0;

  return result;
}

class E2E_BackendDigamma_bf16_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendDigamma_bf16_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  bfloat16_t* input = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t* y = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  float *expect = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
      float val = rand() / (double)RAND_MAX * 5.0f + 1.0f;  // [1, 6] - positive values for digamma
      input[i] = static_cast<bfloat16_t>(val);
      expect[i] = static_cast<float>(digamma(val));
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(digamma_bf16_test, tiling_data.block_dim, (uint8_t *)input, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  //精度校验: bfloat16_t使用容差比较
  uint32_t diff_count = 0;
  const float EPS = 1e-2f;
  for (int i = 0; i < test_size; i++) {
      float y_val = static_cast<float>(y[i]);
      if (std::fabs(y_val - expect[i]) > EPS) {
        printf("diff at index %d: x: %f, y: %f, expect: %f, diff: %f\n", i, static_cast<float>(input[i]),
                y_val, expect[i], std::fabs(y_val - expect[i]));
        diff_count++;
      }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendDigamma_bf16_Code,
    ::testing::Values(std::vector<int>{32, 16, 16}));
