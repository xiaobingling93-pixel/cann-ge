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
#include "api_regbase/remainder.h"

using namespace AscendC;

template <typename T>
struct TensorRemainderInputParam {
  T *y{};
  T *exp{};
  T *src0{};
  T *src1{};
  uint32_t size{0};
  uint32_t out_size{0};
};

class TestRegbaseApiRemainder :public testing::Test {
 protected:
  template <typename T>
  static void InvokeKernelWithTwoTensorInput(TensorRemainderInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_x2 = x2buf.Get<T>();

    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);
    RemainderExtend(l_y, l_x1, l_x2, param.size);
    UbToGm(param.y, l_y, param.size);
}


  template <typename T>
  static void CreateTensorInput(TensorRemainderInputParam<T> &param) {
    // 构造测试输入和预期结果
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src0 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.src1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);
    std::uniform_int_distribution distr(0, 20);  // Define range [0, 20]

    // 构造src1的随机生成器，避免除以零
    std::mt19937 eng1(2);
    std::uniform_int_distribution distr1(1, 20);  // Define range [1, 20], avoid division by zero

    for (int i = 0; i < param.size; i++) {
      T input = static_cast<T>(distr(eng));
      T input1 = static_cast<T>(distr1(eng1));
      param.src0[i] = input;
      param.src1[i] = input1;
      // Remainder: x1 - x2 * trunc(x1/x2)
      param.exp[i] = static_cast<T>(static_cast<double>(input) -
                                    static_cast<double>(input1) *
                                        std::floor(static_cast<double>(input) / static_cast<double>(input1)));
    }
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if (std::fabs(static_cast<double>(y[i]) - static_cast<double>(exp[i])) > 1e-3) {
          diff_count++;
        }
      } else {
        if (y[i] != exp[i]) {
          diff_count++;
        }
      }
    }
    return diff_count;
  }

  template <typename T>
  static void RemainderTest(uint32_t size) {
    TensorRemainderInputParam<T> param{};
    param.size = size;
    CreateTensorInput(param);

    // 构造Api调用函数
    auto kernel = [&param] { InvokeKernelWithTwoTensorInput(param); };

    // 调用kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    // 验证结果
    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

TEST_F(TestRegbaseApiRemainder, Remainder_Test) {
  RemainderTest<half>(ONE_BLK_SIZE / sizeof(half));
  RemainderTest<half>(ONE_REPEAT_BYTE_SIZE / sizeof(half));
  RemainderTest<half>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  RemainderTest<half>((ONE_BLK_SIZE - sizeof(half)) / sizeof(half));
  RemainderTest<half>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(half));
  RemainderTest<half>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(half));
  RemainderTest<float>(ONE_BLK_SIZE / sizeof(float));
  RemainderTest<float>(ONE_REPEAT_BYTE_SIZE / sizeof(float));
  RemainderTest<float>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
  RemainderTest<float>((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  RemainderTest<float>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
  RemainderTest<float>(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + (ONE_BLK_SIZE - sizeof(float))) / 2 /sizeof(float));
}
