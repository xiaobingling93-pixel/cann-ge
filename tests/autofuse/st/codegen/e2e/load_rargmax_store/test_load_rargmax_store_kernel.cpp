/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_rargmax_store(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData& tiling_data);

class E2E_LoadRargmaxStore_Code : public testing::Test,
    public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadRargmaxStore_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1];

  AutofuseTilingData tiling_data;
  float *x = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  int64_t *y = (int64_t *)AscendC::GmAlloc(test_shape[0] * sizeof(int64_t) + 32);
  int64_t *expect = (int64_t *)AscendC::GmAlloc(test_shape[0] * sizeof(int64_t) + 32);

  // Prepare test and expect data
  for (int i = 0; i < test_shape[0] * test_shape[1]; i++) {
    x[i] = (float)1.0;
  }
  // x[5] = (float)2.0;
  // x[6] = (float)(-2.0);
  // x[20] = (float)2.0;
  // x[52] = (float)3.0;

  // Calculate expected argmax indices
  for (int i = 0; i < test_shape[0]; i++) {
    float max_value = -INFINITY;
    int64_t max_index = 0;
    for (int j = 0; j < test_shape[1]; j++) {
        int idx = i * test_shape[1] + j;
        if (x[idx] > max_value) {
            max_value = x[idx];
            max_index = j;
        }
    }
    expect[i] = max_index;
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_rargmax_store, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr, (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_shape[0]; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(x);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadRargmaxStore_Code,
    ::testing::Values(
        std::vector<int>{4, 2}
        ));
