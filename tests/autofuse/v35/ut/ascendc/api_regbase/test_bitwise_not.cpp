/**
* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * test_bitwise_not.cpp
 */

#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"

using namespace AscendC;

namespace ge {

class TestRegbaseApiBitwiseNotUT : public testing::Test {
 protected:
  // Tensor - Tensor 场景
  template <typename T>
  static void InvokeTensorTensorKernel(UnaryInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> x1buf, ybuf;
    tpipe.InitBuffer(x1buf, sizeof(T) * param.size);
    tpipe.InitBuffer(ybuf, sizeof(T) * AlignUp(param.size, ONE_BLK_SIZE / sizeof(T)));

    LocalTensor<T> l_x1 = x1buf.Get<T>();
    LocalTensor<T> l_y = ybuf.Get<T>();

    GmToUb(l_x1, param.x1, param.size);
    BitwiseNot(l_y, l_x1, param.size);
    UbToGm(param.y, l_y, param.size);
  }

  template <typename T>
  static void CreateTensorInput(UnaryInputParam<T> &param) {
    param.y = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.exp = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));
    param.x1 = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.size));

    std::mt19937 eng(1);
    int input_range = 100;

    for (uint32_t i = 0; i < param.size; i++) {
      if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> distr(-50.0f, 50.0f);
        param.x1[i] = distr(eng);
        param.exp[i] = ~param.x1[i];
      } else if constexpr (std::is_same_v<T, int64_t>) {
        std::uniform_int_distribution<int64_t> distr(-10000, 10000);
        param.x1[i] = distr(eng);
        param.exp[i] = ~param.x1[i];
      } else {
        std::uniform_int_distribution<int> distr(-input_range, input_range);
        param.x1[i] = static_cast<T>(distr(eng));
        param.exp[i] = ~param.x1[i];
      }
    }
  }

  // Tensor - Tensor 测试
  template <typename T>
  static void BitwiseNotTensorTensorTest(uint32_t size) {
    UnaryInputParam<T> param{};
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

// ============ Tensor - Tensor 测试 (新增数据类型: DT_INT8, DT_INT64, DT_BF16) ============
TEST_F(TestRegbaseApiBitwiseNotUT, BitwiseNot_TensorTensor_Test) {
  // int8
  BitwiseNotTensorTensorTest<int8_t>(ONE_BLK_SIZE / sizeof(int8_t));
  BitwiseNotTensorTensorTest<int8_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int8_t));
  BitwiseNotTensorTensorTest<int8_t>((ONE_BLK_SIZE - sizeof(int8_t)) / sizeof(int8_t));
  BitwiseNotTensorTensorTest<int8_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int8_t));
  BitwiseNotTensorTensorTest<int8_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));
  BitwiseNotTensorTensorTest<int8_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int8_t));

  // int64
  BitwiseNotTensorTensorTest<int64_t>(ONE_BLK_SIZE / sizeof(int64_t));
  BitwiseNotTensorTensorTest<int64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(int64_t));
  BitwiseNotTensorTensorTest<int64_t>((ONE_BLK_SIZE - sizeof(int64_t)) / sizeof(int64_t));
  BitwiseNotTensorTensorTest<int64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int64_t));
  BitwiseNotTensorTensorTest<int64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));
  BitwiseNotTensorTensorTest<int64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(int64_t));

  // uint32
  BitwiseNotTensorTensorTest<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t));
  BitwiseNotTensorTensorTest<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
  BitwiseNotTensorTensorTest<uint32_t>((ONE_BLK_SIZE - sizeof(uint32_t)) / sizeof(uint32_t));
  BitwiseNotTensorTensorTest<uint32_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint32_t));
  BitwiseNotTensorTensorTest<uint32_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t)); 
  BitwiseNotTensorTensorTest<uint32_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint32_t));

  // uint64
  BitwiseNotTensorTensorTest<uint64_t>(ONE_BLK_SIZE / sizeof(uint64_t));
  BitwiseNotTensorTensorTest<uint64_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint64_t));
  BitwiseNotTensorTensorTest<uint64_t>((ONE_BLK_SIZE - sizeof(uint64_t)) / sizeof(uint64_t));
  BitwiseNotTensorTensorTest<uint64_t>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(uint64_t));
  BitwiseNotTensorTensorTest<uint64_t>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t));
  BitwiseNotTensorTensorTest<uint64_t>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t));
}

}  // namespace ge