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

template <typename InT, typename OutT>
struct RoundToIntInputParam {
  InT *x{};
  OutT *y{};
  OutT *exp{};
  uint32_t size{};
};

class TestRegbaseApiRoundToIntUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename InT, typename OutT>
  static void InvokeTensorTensorKernel(RoundToIntInputParam<InT, OutT> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf;
    tpipe.InitBuffer(xbuf, sizeof(InT) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(OutT) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(OutT)));

    LocalTensor<InT> l_x = xbuf.Get<InT>();
    LocalTensor<OutT> l_y = ybuf.Get<OutT>();

    GmToUb(l_x, param.x, param.size);
    // 使用 AscendC::Cast 函数，设置 round 模式为 CAST_RINT
    AscendC::Cast(l_y, l_x, AscendC::RoundMode::CAST_RINT, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename InT, typename OutT>
  static void CreateTensorInput(RoundToIntInputParam<InT, OutT> &param) {
    param.exp = static_cast<OutT *>(AscendC::GmAlloc(sizeof(OutT) * param.size));
    param.x = static_cast<InT *>(AscendC::GmAlloc(sizeof(InT) * param.size));
    param.y = static_cast<OutT *>(AscendC::GmAlloc(sizeof(OutT) * param.size));

    std::mt19937 eng(1);
    std::uniform_real_distribution<float> distr(-10.0f, 10.0f);

    for (uint32_t i = 0; i < param.size; i++) {
      float val = distr(eng);
      if constexpr (AscendC::IsSameType<OutT, uint8_t>::value) {
        param.x[i] = static_cast<InT>(std::abs(val));
      } else {
        param.x[i] = static_cast<InT>(val);
      }
      // 计算期望值：四舍五入取整
      param.exp[i] = static_cast<OutT>(static_cast<int64_t>(std::rint(static_cast<float>(param.x[i]))));
    }
  }

  template <typename InT, typename OutT>
  static uint32_t Valid(RoundToIntInputParam<InT, OutT> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.size; i++) {
      auto diff = (double)(param.y[i] - param.exp[i]);
      if(diff < -1e-5 || diff > 1e-5) {
        diff_count++;
        printf("diff at index %d: x: %f, y: %f, expect: %f, diff: %f\n", i, static_cast<float>(param.x[i]),
               static_cast<float>(param.y[i]), static_cast<float>(param.exp[i]),
               static_cast<float>(param.y[i] - param.exp[i]));
      }
    }
    return diff_count;
  }

  // Tensor - Tensor 测试
  template <typename InT, typename OutT>
  static void RoundToIntTensorTensorTest(uint32_t size) {
    RoundToIntInputParam<InT, OutT> param{};
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

// ============ Tensor - Tensor 测试 (输入: DT_FLOAT, 输出: DT_INT16, DT_INT32, DT_INT64) ============
TEST_F(TestRegbaseApiRoundToIntUT, RoundToInt_TensorTensor_Test) {
  // float -> int16
  RoundToIntTensorTensorTest<float, int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  RoundToIntTensorTensorTest<float, int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  RoundToIntTensorTensorTest<float, int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  RoundToIntTensorTensorTest<float, int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  RoundToIntTensorTensorTest<float, int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  RoundToIntTensorTensorTest<float, int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));

  // float -> int32
  RoundToIntTensorTensorTest<float, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  RoundToIntTensorTensorTest<float, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  RoundToIntTensorTensorTest<float, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  RoundToIntTensorTensorTest<float, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  RoundToIntTensorTensorTest<float, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
  RoundToIntTensorTensorTest<float, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));

  // float -> int64
  RoundToIntTensorTensorTest<float, int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  RoundToIntTensorTensorTest<float, int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  RoundToIntTensorTensorTest<float, int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  RoundToIntTensorTensorTest<float, int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  RoundToIntTensorTensorTest<float, int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
  RoundToIntTensorTensorTest<float, int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
}

// ============ Tensor - Tensor 测试 (输入: DT_FLOAT16, 输出:DT_INT16, DT_INT32, DT_INT8, DT_UINT8) ============
TEST_F(TestRegbaseApiRoundToIntUT, RoundToInt_TensorTensor_Float16_Test) {
  // float16 -> int16
  RoundToIntTensorTensorTest<half, int16_t>(ONE_BLK_SIZE / sizeof(int16_t));
  RoundToIntTensorTensorTest<half, int16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int16_t));
  RoundToIntTensorTensorTest<half, int16_t>((ONE_BLK_SIZE - sizeof(int16_t)) / sizeof(int16_t));
  RoundToIntTensorTensorTest<half, int16_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int16_t));
  RoundToIntTensorTensorTest<half, int16_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));
  RoundToIntTensorTensorTest<half, int16_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int16_t));

  // float16 -> int32
  RoundToIntTensorTensorTest<half, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  RoundToIntTensorTensorTest<half, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  RoundToIntTensorTensorTest<half, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  RoundToIntTensorTensorTest<half, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  RoundToIntTensorTensorTest<half, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
  RoundToIntTensorTensorTest<half, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));

  // float16 -> int8
  RoundToIntTensorTensorTest<half, int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
  RoundToIntTensorTensorTest<half, int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  RoundToIntTensorTensorTest<half, int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  RoundToIntTensorTensorTest<half, int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
  RoundToIntTensorTensorTest<half, int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));
  RoundToIntTensorTensorTest<half, int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));

  // float16 -> uint8
  RoundToIntTensorTensorTest<half, uint8_t>(ONE_BLK_SIZE / sizeof(uint8_t));
  RoundToIntTensorTensorTest<half, uint8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t));
  RoundToIntTensorTensorTest<half, uint8_t>((ONE_BLK_SIZE - sizeof(uint8_t)) / sizeof(uint8_t));
  RoundToIntTensorTensorTest<half, uint8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint8_t));
  RoundToIntTensorTensorTest<half, uint8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint8_t));
  RoundToIntTensorTensorTest<half, uint8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint8_t));
}

// ============ Tensor - Tensor 测试 (输入: DT_BF16, 输出: DT_INT32) ============
TEST_F(TestRegbaseApiRoundToIntUT, RoundToInt_TensorTensor_BF16_Test) {
  // bf16 -> int32
  RoundToIntTensorTensorTest<bfloat16_t, int32_t>(ONE_BLK_SIZE / sizeof(int32_t));
  RoundToIntTensorTensorTest<bfloat16_t, int32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int32_t));
  RoundToIntTensorTensorTest<bfloat16_t, int32_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t));
  RoundToIntTensorTensorTest<bfloat16_t, int32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t));
  RoundToIntTensorTensorTest<bfloat16_t, int32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
  RoundToIntTensorTensorTest<bfloat16_t, int32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int32_t));
}

} // namespace ge