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
struct SquareInputParam {
  T *x{};
  T *y{};
  T *exp{};
  uint32_t size{};
};

class TestRegbaseApiSquareUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(SquareInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf;
    tpipe.InitBuffer(xbuf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x = xbuf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x, param.x, param.size);
    AscendC::Mul(l_y, l_x, l_x, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(SquareInputParam<T> &param) {
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);

    for (uint32_t i = 0; i < param.size; i++) {
      if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<float> distr(-5.0f, 5.0f);
        param.x[i] = static_cast<T>(distr(eng));
        param.exp[i] = param.x[i] * param.x[i];
      } else {
        std::uniform_int_distribution<int> distr(-100, 100);
        param.x[i] = static_cast<T>(distr(eng));
        param.exp[i] = param.x[i] * param.x[i];
      }
    }
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void SquareTensorTensorTest(uint32_t size) {
    SquareInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeTensorTensorKernel(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

// ============ Tensor - Tensor 测试 (支持的数据类型: DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16) ============
TEST_F(TestRegbaseApiSquareUT, Square_TensorTensor_Test) {
  // float
  SquareTensorTensorTest<float>(ONE_BLK_SIZE / sizeof(float));
  SquareTensorTensorTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  SquareTensorTensorTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  SquareTensorTensorTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  SquareTensorTensorTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  SquareTensorTensorTest<float>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));

  // float16
  SquareTensorTensorTest<half>(ONE_BLK_SIZE / sizeof(half));
  SquareTensorTensorTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  SquareTensorTensorTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  SquareTensorTensorTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  SquareTensorTensorTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  SquareTensorTensorTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));

  // bf16
  SquareTensorTensorTest<bfloat16_t>(ONE_BLK_SIZE / sizeof(bfloat16_t));
  SquareTensorTensorTest<bfloat16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(bfloat16_t));
  SquareTensorTensorTest<bfloat16_t>((ONE_BLK_SIZE - sizeof(bfloat16_t)) / sizeof(bfloat16_t));
  SquareTensorTensorTest<bfloat16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(bfloat16_t));
  SquareTensorTensorTest<bfloat16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(bfloat16_t));
  SquareTensorTensorTest<bfloat16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(bfloat16_t));

  // int32
  SquareTensorTensorTest<int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  SquareTensorTensorTest<int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  SquareTensorTensorTest<int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  SquareTensorTensorTest<int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  SquareTensorTensorTest<int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
  SquareTensorTensorTest<int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));

  // int16
  SquareTensorTensorTest<int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  SquareTensorTensorTest<int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  SquareTensorTensorTest<int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  SquareTensorTensorTest<int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  SquareTensorTensorTest<int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  SquareTensorTensorTest<int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));

  // uint16
  SquareTensorTensorTest<uint16_t>(ONE_BLK_SIZE / sizeof(uint16_t));
  SquareTensorTensorTest<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
  SquareTensorTensorTest<uint16_t>((ONE_BLK_SIZE - sizeof(uint16_t)) / sizeof(uint16_t));
  SquareTensorTensorTest<uint16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint16_t));
  SquareTensorTensorTest<uint16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));
  SquareTensorTensorTest<uint16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint16_t));

  // uint32
  SquareTensorTensorTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t));
  SquareTensorTensorTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  SquareTensorTensorTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  SquareTensorTensorTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
  SquareTensorTensorTest<uint32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));
  SquareTensorTensorTest<uint32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));

  // int64
  SquareTensorTensorTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  SquareTensorTensorTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  SquareTensorTensorTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  SquareTensorTensorTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  SquareTensorTensorTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
  SquareTensorTensorTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));

  // uint64
  SquareTensorTensorTest<uint64_t>(ONE_BLK_SIZE / sizeof(uint64_t));
  SquareTensorTensorTest<uint64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  SquareTensorTensorTest<uint64_t>((ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t));
  SquareTensorTensorTest<uint64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t));
  SquareTensorTensorTest<uint64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t));
  SquareTensorTensorTest<uint64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t));
}

}  // namespace ge
