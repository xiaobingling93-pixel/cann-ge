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
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void concat_v2_test(GM_ADDR data0,
                                                     GM_ADDR data1,
                                                     GM_ADDR output,
                                                     GM_ADDR workspace,
                                                     GM_ADDR gm_tiling_data);
extern "C" int64_t AutofuseTiling(uint32_t s0,
                                  uint32_t s1,
                                  uint32_t s2,
                                  AutofuseTilingData *tiling,
                                  uint32_t *workspaceSize,
                                  uint64_t *blockDim,
                                  uint32_t aiv_num,
                                  uint32_t ub_size);

class E2E_BackendConcat_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendConcat_Code, CalculateCorrect) {
  auto test_shape = GetParam();
  int32_t col_0 = test_shape[1];
  int32_t col_1 = test_shape.size() == 2 ? test_shape[1] : test_shape[2];
  int32_t input_size_0 = test_shape[0] * col_0;
  int32_t input_size_1 = test_shape[0] * col_1;
  int32_t output_size = input_size_0 + input_size_1;

  AutofuseTilingData tiling_data;
  int32_t *input1 = (int32_t *) AscendC::GmAlloc(test_shape[0] * col_0 * sizeof(int32_t) + 32);
  int32_t *input2 = (int32_t *) AscendC::GmAlloc(test_shape[0] * col_1 * sizeof(int32_t) + 32);
  int32_t *y = (int32_t *) AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);
  int32_t *expect = (int32_t *) AscendC::GmAlloc(output_size * sizeof(int32_t) + 32);

  // Prepare test and expect data
  int32_t value = 0;
  for (int i = 0; i < test_shape[0]; ++i) {
    for (int j = 0; j < col_0; ++j) {
      input1[i * col_0 + j] = value;
      expect[i * (col_0 + col_1) + j] = -1 * input1[i * col_0 + j];
      value++;
    }
    for (int j = 0; j < col_1; ++j) {
      input2[i * col_1 + j] = 100000 + value;
      expect[i * (col_0 + col_1) + col_0 + j] = -1 * input2[i * col_1 + j];
      value++;
    }
  }

  // Launch
  uint32_t ws_size = 0;
  uint64_t block_dim = 48;
  if (test_shape.size() == 2) {
    AutofuseTiling(test_shape[0], test_shape[1], test_shape[1], &tiling_data, &ws_size, &block_dim, 1, 192 * 1024);
  } else {
    AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 1, 192 * 1024);
  }

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(concat_v2_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < output_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << output_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendConcat_Code,
                         ::testing::Values(
                             std::vector<int>{16, 100}
                             // std::vector<int>{1, 1},
                             // std::vector<int>{100, 1},
                             // std::vector<int>{2, 2},
                             // std::vector<int>{200, 2},
                             // std::vector<int>{15, 15},
                             // std::vector<int>{16, 16},
                             // std::vector<int>{17, 17},
                             // std::vector<int>{29, 31},
                             // std::vector<int>{29, 31, 33},
                             // std::vector<int>{30, 32},
                             // std::vector<int>{31, 33},
                             // std::vector<int>{511, 63},
                             // std::vector<int>{1025, 64}
                         ));