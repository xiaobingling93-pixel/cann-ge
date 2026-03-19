/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"

using namespace AscendC;

template <typename T>
struct ReluInputParam {
  T *y{};
  T *x{};
  T *exp{};
  uint32_t size{};
};

class TestApiReluUT : public testing::Test {
 protected:
  template <typename T>
  static void InvokeTensorTensorKernel(ReluInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x, param.x, param.size);
    AscendC::Relu(l_y, l_x, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(ReluInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      // 生成测试数据，范围覆盖正数、零和负数
      // 对于无符号类型，负数范围会被截断为正数
      std::uniform_real_distribution<double> data_distr(-10.0, 10.0);
      double val = data_distr(eng);
      param.x[i] = static_cast<T>(val);
      // Relu(x) = max(x, 0)
      param.exp[i] = (param.x[i] > T(0)) ? param.x[i] : T(0);
    }
  }

  template <typename T>
  static uint32_t Valid(ReluInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      const double atol = 1e-5;
      const double delta = static_cast<double>(param.y[i]) - static_cast<double>(param.exp[i]);
      if (delta > atol && delta < (-atol)) {
        diff_count++;
      }
    }
    return diff_count;
  }

  template <typename T>
  static void ReluTensorTensorTest(uint32_t size) {
    ReluInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);
  }

};

// ============ Tensor - Tensor 测试 ============
TEST_F(TestApiReluUT, Relu_TensorTensor_Test) {
  // int32
  ReluTensorTensorTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  ReluTensorTensorTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  ReluTensorTensorTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  ReluTensorTensorTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  ReluTensorTensorTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
  ReluTensorTensorTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));

  // float
  ReluTensorTensorTest<float>(ONE_BLK_SIZE / sizeof(float));
  ReluTensorTensorTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  ReluTensorTensorTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  ReluTensorTensorTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  ReluTensorTensorTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  ReluTensorTensorTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));

  // half
  ReluTensorTensorTest<half>(ONE_BLK_SIZE / sizeof(half));
  ReluTensorTensorTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  ReluTensorTensorTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  ReluTensorTensorTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  ReluTensorTensorTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  ReluTensorTensorTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));

  // int64
  ReluTensorTensorTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  ReluTensorTensorTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  ReluTensorTensorTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  ReluTensorTensorTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  ReluTensorTensorTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
  ReluTensorTensorTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
}
