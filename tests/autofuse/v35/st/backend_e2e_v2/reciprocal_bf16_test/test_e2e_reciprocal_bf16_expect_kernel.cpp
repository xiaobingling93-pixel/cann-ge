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
// 输入输出数量与{graph}中描述一致
extern "C" __global__ __aicore__ void reciprocal_bf16_test(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
// 确保轴数量与3匹配, 与reciprocal_bf16_backend_generator.cpp中的shape_info声明的轴数量一致
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendReciprocalBf16_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendReciprocalBf16_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  //  test_shape的size与3匹配
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  // 输入输出数量与{graph}中描述一致
  // 根据{dtype_map}和{dtype}生成正确的dtype_val, 如
  // uint8_t* input1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  bfloat16_t* input1 = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t* y = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);
  bfloat16_t *expect = (bfloat16_t *)AscendC::GmAlloc(test_size * sizeof(bfloat16_t) + 32);

  // Prepare test and expect data
  srand(1);  // 固定随机种子
  for (int i = 0; i < test_size; i++) {
      float f_val = static_cast<float>(rand()) / RAND_MAX * 10.0f + 0.1f; // 避免零值
      input1[i] = static_cast<bfloat16_t>(f_val);
      expect[i] = static_cast<bfloat16_t>(1.0f / input1[i]);
  }

  // Launch
  uint32_t ws_size = 0;
  // test_shape的size与3匹配, 入参数量与函数声明保持一致
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  // 输入输出数量与{graph}中描述和函数声明一致
  ICPU_RUN_KF(reciprocal_bf16_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // 根据精度校验规则
  uint32_t diff_count = 0;
  const float EPS = 1e-2f; // bfloat16的容差值
  for (int i = 0; i < test_size; i++) {
      float f_y = static_cast<float>(y[i]);
      float f_expect = static_cast<float>(expect[i]);

      if (std::fabs(f_y - f_expect) > EPS) {
          diff_count++;
          printf("Index %d: y=%f, expect=%f, diff=%f\n", i, f_y, f_expect, std::fabs(f_y - f_expect));
      }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  // 保持与变量声明一致
  AscendC::GmFree(input1);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

// 用例输入的维度需要与构图接口的dims_size匹配
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendReciprocalBf16_Code,
    ::testing::Values(std::vector<int>{32, 16, 16}));
