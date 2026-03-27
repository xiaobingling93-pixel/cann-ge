/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void fma_int8_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendFmaInt8_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendFmaInt8_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;
  
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  int8_t* input1 = (int8_t *)AscendC::GmAlloc(test_size * sizeof(int8_t) + 32);
  int8_t* input2 = (int8_t *)AscendC::GmAlloc(test_size * sizeof(int8_t) + 32);
  int8_t* input3 = (int8_t *)AscendC::GmAlloc(test_size * sizeof(int8_t) + 32);
  int8_t* y = (int8_t *)AscendC::GmAlloc(test_size * sizeof(int8_t) + 32);
  int8_t *expect = (int8_t *)AscendC::GmAlloc(test_size * sizeof(int8_t) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    input1[i] = rand() % 10;
    input2[i] = rand() % 10;
    input3[i] = rand() % 10;

    // Fma computes: x1 * x2 + x3
    // Convert to float16 for computation, then back to int8
    expect[i] = static_cast<int8_t>(static_cast<float>(input1[i]) * static_cast<float>(input2[i]) + static_cast<float>(input3[i]));
  }

  // Launch
  uint32_t ws_size = 0;
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(fma_int8_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)input3, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(input3);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendFmaInt8_Code,
    ::testing::Values(std::vector<int>{32, 16, 16}  // 用例输入的维度需要与构图接口的dims_size匹配
                      ));
