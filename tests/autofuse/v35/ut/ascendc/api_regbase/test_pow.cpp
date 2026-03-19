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
#include "api_regbase/pow.h"

using namespace AscendC;

template <typename T>
struct PowInputParam {
  T *y{};
  T *x1{};
  T *x2{};
  T *exp{};
  uint32_t size{};
};

class TestApiPowUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(PowInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.x1, param.size);
    GmToUb(l_x2, param.x2, param.size);
    Pow(l_y, l_x1, l_x2, param.size, l_tmp);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(PowInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x2 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution<float> data_distr(0.0f, 10.0f);
      std::uniform_real_distribution<float> pow_distr(0.0f, 2.0f);
      param.x1[i] = static_cast<T>(data_distr(eng));
      param.x2[i] = static_cast<T>(pow_distr(eng));
      param.exp[i] = static_cast<T>(std::pow(static_cast<float>(param.x1[i]), static_cast<float>(param.x2[i])));
    }
  }

  template <typename T>
  static uint32_t Valid(PowInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      const double atol = 1e-3;
      const double delta = static_cast<double>(param.y[i]) - static_cast<double>(param.exp[i]);
      if (delta > atol && delta < (-atol)) {
        diff_count++;
      }
    }
    return diff_count;
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void PowTensorTensorTest(uint32_t size) {
    PowInputParam<T> param{};
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
TEST_F(TestApiPowUT, Pow_TensorTensor_Test) {
  // int8
  PowTensorTensorTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
  PowTensorTensorTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  PowTensorTensorTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  PowTensorTensorTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
  PowTensorTensorTest<int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));
  PowTensorTensorTest<int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));

  // int16
  PowTensorTensorTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  PowTensorTensorTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  PowTensorTensorTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  PowTensorTensorTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  PowTensorTensorTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  PowTensorTensorTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));

  // int32
  PowTensorTensorTest<int>(ONE_BLK_SIZE / sizeof(int));
  PowTensorTensorTest<int>(ONE_REPEAT_BYTE_SIZE / sizeof(int));
  PowTensorTensorTest<int>((ONE_BLK_SIZE - sizeof(int)) / sizeof(int));
  PowTensorTensorTest<int>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int));
  PowTensorTensorTest<int>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int));
  PowTensorTensorTest<int>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int));

  // uint8
  PowTensorTensorTest<uint8_t>(ONE_BLK_SIZE / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>((ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint8_t));
  PowTensorTensorTest<uint8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint8_t));

  // uint16
  PowTensorTensorTest<uint16_t>(ONE_BLK_SIZE / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>((ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));
  PowTensorTensorTest<uint16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));

  // uint32
  PowTensorTensorTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));
  PowTensorTensorTest<uint32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));

  // float
  PowTensorTensorTest<float>(ONE_BLK_SIZE / sizeof(float));
  PowTensorTensorTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  PowTensorTensorTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  PowTensorTensorTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  PowTensorTensorTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  PowTensorTensorTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));

  // half
  PowTensorTensorTest<half>(ONE_BLK_SIZE / sizeof(half));
  PowTensorTensorTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  PowTensorTensorTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  PowTensorTensorTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  PowTensorTensorTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  PowTensorTensorTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));

  // BF16
  PowTensorTensorTest<bfloat16_t>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>((ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(bfloat16_t));
  PowTensorTensorTest<bfloat16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(bfloat16_t));
}