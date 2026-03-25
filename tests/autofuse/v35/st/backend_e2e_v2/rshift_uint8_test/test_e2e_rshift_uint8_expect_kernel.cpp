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
extern "C" __global__ __aicore__ void rshift_uint8_test(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
// 确保轴数量与{dim_nums}匹配, 与{case_name}_backend_generator.cpp中的shape_info声明的轴数量一致
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint64_t *blockDim, uint32_t aiv_num, uint32_t ub_size);

class E2E_BackendRshiftUint8_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {
};

TEST_P(E2E_BackendRshiftUint8_Code, CalculateCorrect) {
  auto test_shape = GetParam();

  uint64_t block_dim = 48;

  //  test_shape的size与{dim_nums}匹配
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  // 输入输出数量与{graph}中描述一致
  // 根据{dtype_map}和{dtype}生成正确的dtype_val, 如
  // uint8_t* input1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  uint8_t* input1 = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  int8_t* input2 = (int8_t *)AscendC::GmAlloc(test_size * sizeof(int8_t) + 32);
  uint8_t* y = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  uint8_t *expect = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

  // Prepare test and expect data
  srand(1);  // 固定随机种子
  for (int i = 0; i < test_size; i++) {
      input1[i] = static_cast<uint8_t>((i * 17) % 255 + 1); // 生成1-255之间的uint8数据
      input2[i] = static_cast<int8_t>((i * 3) % 8); // 生成0-7之间的移位位数
      // 计算期望值
      if (input2[i] < 0) {
          // 负数移动位返回0
          expect[i] = 0;
      } else if (input2[i] >= 8) {
          // 移动8位或更多位，对于8位类型返回0
          expect[i] = 0;
      } else {
          // 正数移动位，无符号右移
          expect[i] = input1[i] >> static_cast<int8_t>(input2[i]);
      }
  }

  // Launch
  uint32_t ws_size = 0;
  // test_shape的size与{dim_nums}匹配, 入参数量与函数声明保持一致
  AutofuseTiling(test_shape[0], test_shape[1], test_shape[2], &tiling_data, &ws_size, &block_dim, 48, 192*1024);
  printf("tiling key: %d, core_num: %d\n", tiling_data.tiling_key, tiling_data.block_dim);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  // 输入输出数量与{graph}中描述和函数声明一致
  ICPU_RUN_KF(rshift_uint8_test, tiling_data.block_dim, (uint8_t *)input1, (uint8_t *)input2, (uint8_t *)y, nullptr, (uint8_t*)&tiling_data);

  // 根据精度校验规则
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      // 打印相关数据，方便问题处理
      printf("Index %d: input1=%d, input2=%d, y=%d, expect=%d, diff=%f\n", i, input1[i], input2[i], y[i], expect[i], std::fabs(y[i] - expect[i]));
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  // 保持与变量声明一致
  AscendC::GmFree(input1);
  AscendC::GmFree(input2);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

// 用例输入的维度需要与构图接口的dims_size匹配
INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_BackendRshiftUint8_Code,
    ::testing::Values(std::vector<int>{2, 8, 8}));
