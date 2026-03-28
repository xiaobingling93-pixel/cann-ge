/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
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

namespace ge {
template <typename T>
struct TanInputParam {
  T *x{};
  T *y{};
  T *exp{};
  uint32_t size{};
};

class TestApiTanUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(TanInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));
    tpipe.InitBuffer(tmp, TMP_UB_SIZE);

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, param.x, param.size);
    AscendC::Tan(l_y, l_x, l_tmp, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(TanInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      std::uniform_real_distribution distr(-1.5f, 1.5f);
      param.x[i] = distr(eng);
      param.exp[i] = static_cast<T>(std::tan(static_cast<double>(param.x[i])));
    }
  }

  template <typename T>
  static uint32_t Valid(TanInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      auto diff = (double)(param.y[i] - param.exp[i]);
      if(diff < -1e-2 || diff > 1e-2) {
        diff_count++;
        printf("diff at index %d: x: %f, y: %f, expect: %f, diff: %f\n", i, static_cast<float>(param.x[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]),
               static_cast<float>(param.y[i] - param.exp[i]));
      }
    }
    return diff_count;
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void TanTest(uint32_t size) {
    TanInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造 API 调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用 kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0) << " of " << size;

    AscendC::GmFree(param.x);
    AscendC::GmFree(param.y);
    AscendC::GmFree(param.exp);
  }
};

TEST_F(TestApiTanUT, Tan_TensorTensor_Test) {
  // 测试 float16 类型
  TanTest<half>(ONE_BLK_SIZE / sizeof(half));
  TanTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  TanTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  TanTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  TanTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  TanTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  TanTest<half>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(half))) / 2 /sizeof(half));

  // 测试 float 类型
  TanTest<float>(ONE_BLK_SIZE / sizeof(float));
  TanTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  TanTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  TanTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  TanTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  TanTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  TanTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}
}  // namespace ge
