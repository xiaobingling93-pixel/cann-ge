/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "remainder.h"

using namespace AscendC;

template <typename T, typename DstType = T>
struct TensorRemainderInputParam {
  DstType *y{};
  DstType *exp{};
  T *src0{};  // dividend
  T *src1{};  // divisor
  uint32_t size{0};
};

class TestApiRemainder : public testing::Test {
 protected:
  // Float type test: input float, output float
  static void InvokeKernelWithTwoTensorInputFloat(TensorRemainderInputParam<float, float> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(float) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(float) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(float) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(float)));
    // Need 3 temp buffers for div_res, floor_res, mul_res
    tpipe.InitBuffer(tmp, 3 * sizeof(float) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(float)));

    LocalTensor<float> l_x1 = x1buf.Get<float>();
    LocalTensor<float> l_x2 = x2buf.Get<float>();
    LocalTensor<float> l_y = ybuf.Get<float>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);

    RemainderExtend(l_y, l_x1, l_x2, l_tmp, param.size);

    UbToGm(param.y, l_y, param.size);
  }

  // Int32 type test: input int32, output float
  static void InvokeKernelWithTwoTensorInputInt32(TensorRemainderInputParam<int32_t, float> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, x2buf, ybuf, tmp;

    tpipe.InitBuffer(x1buf, sizeof(int32_t) * param.size);
    tpipe.InitBuffer(x2buf, sizeof(int32_t) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(float) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(float)));
    // Need 3 temp buffers for intermediate float results
    tpipe.InitBuffer(tmp, 3 * sizeof(float) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(float)));

    LocalTensor<int32_t> l_x1 = x1buf.Get<int32_t>();
    LocalTensor<int32_t> l_x2 = x2buf.Get<int32_t>();
    LocalTensor<float> l_y = ybuf.Get<float>();
    LocalTensor<uint8_t> l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x1, param.src0, param.size);
    GmToUb(l_x2, param.src1, param.size);

    RemainderExtend(l_y, l_x1, l_x2, l_tmp, param.size);

    UbToGm(param.y, l_y, param.size);
  }

  static void CreateTensorInputFloat(TensorRemainderInputParam<float, float> &param) {
    param.y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.exp = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src0 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));

    std::mt19937 eng(1);
    std::uniform_real_distribution<float> distr(-100.0f, 100.0f);

    std::mt19937 eng1(3);
    std::uniform_real_distribution<float> distr1(1.0f, 100.0f);  // Avoid zero divisor

    for (uint32_t i = 0; i < param.size; i++) {
      float input = distr(eng);
      float input1 = distr1(eng1);
      param.src0[i] = input;
      param.src1[i] = input1;
      // remainder = dividend - floor(dividend / divisor) * divisor
      // Avoid division by zero
      if (input1 == 0.0f) {
        param.exp[i] = 0.0f;
        continue;
      }
      param.exp[i] = input - std::floor(input / input1) * input1;
    }
  }

  static void CreateTensorInputInt32(TensorRemainderInputParam<int32_t, float> &param) {
    param.y = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.exp = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * param.size));
    param.src0 = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * param.size));
    param.src1 = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * param.size));

    std::mt19937 eng(1);
    std::uniform_int_distribution<int32_t> distr(-10000, 10000);

    std::mt19937 eng1(3);
    std::uniform_int_distribution<int32_t> distr1(1, 100);  // Avoid zero divisor

    for (uint32_t i = 0; i < param.size; i++) {
      int32_t input = distr(eng);
      int32_t input1 = distr1(eng1);
      param.src0[i] = input;
      param.src1[i] = input1;
      // remainder = dividend - floor(dividend / divisor) * divisor
      // For int32, computation is in float precision
      float fInput = static_cast<float>(input);
      float fInput1 = static_cast<float>(input1);
      param.exp[i] = fInput - std::floor(fInput / fInput1) * fInput1;
    }
  }

  template <typename T>
  static uint32_t Valid(T *y, T *exp, size_t comp_size) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < comp_size; i++) {
      if (std::fabs(static_cast<double>(y[i]) - static_cast<double>(exp[i])) > 1e-5) {
        diff_count++;
      }
    }
    return diff_count;
  }

  static void RemainderTestFloat(uint32_t size) {
    TensorRemainderInputParam<float, float> param{};
    param.size = size;
    CreateTensorInputFloat(param);

    auto kernel = [&param] { InvokeKernelWithTwoTensorInputFloat(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }

  static void RemainderTestInt32(uint32_t size) {
    TensorRemainderInputParam<int32_t, float> param{};
    param.size = size;
    CreateTensorInputInt32(param);

    auto kernel = [&param] { InvokeKernelWithTwoTensorInputInt32(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param.y, param.exp, param.size);
    EXPECT_EQ(diff_count, 0);
  }
};

// ==================== Float type tests ====================
TEST_F(TestApiRemainder, Remainder_Float_OneBlockSize) {
  RemainderTestFloat(ONE_BLK_SIZE / sizeof(float));
}

TEST_F(TestApiRemainder, Remainder_Float_OneRepeatSize) {
  RemainderTestFloat(ONE_REPEAT_BYTE_SIZE / sizeof(float));
}

TEST_F(TestApiRemainder, Remainder_Float_MaxRepeatSize) {
  RemainderTestFloat(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(float));
}

TEST_F(TestApiRemainder, Remainder_Float_AlignedSize) {
  RemainderTestFloat((ONE_BLK_SIZE - sizeof(float)) / sizeof(float));
  RemainderTestFloat((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(float));
}

TEST_F(TestApiRemainder, Remainder_Float_LargeAlignedSize) {
  RemainderTestFloat(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + 
                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + 
                      (ONE_BLK_SIZE - sizeof(float))) / 2 / sizeof(float));
}

// ==================== Int32 type tests ====================
TEST_F(TestApiRemainder, Remainder_Int32_OneBlockSize) {
  RemainderTestInt32(ONE_BLK_SIZE / sizeof(int32_t));
}

TEST_F(TestApiRemainder, Remainder_Int32_OneRepeatSize) {
  RemainderTestInt32(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
}

TEST_F(TestApiRemainder, Remainder_Int32_MaxRepeatSize) {
  RemainderTestInt32(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
}

TEST_F(TestApiRemainder, Remainder_Int32_AlignedSize) {
  RemainderTestInt32((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  RemainderTestInt32((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
}

TEST_F(TestApiRemainder, Remainder_Int32_LargeAlignedSize) {
  RemainderTestInt32(((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + 
                      (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) + 
                      (ONE_BLK_SIZE - sizeof(int32_t))) / 2 / sizeof(int32_t));
}
