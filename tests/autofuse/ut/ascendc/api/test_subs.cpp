/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include "scalar_sub.h"

template<class T>
void GmToUb(LocalTensor<T>& local, T* gm, int size) {
  for (int i = 0; i < size; i++) {
    local.SetValue(i, gm[i]);
  }
}

template<class T>
void UbToGm(T* gm, LocalTensor<T>& local, int size) {
  for (int i = 0; i < size; i++) {
    gm[i] = local.GetValue(i);
  }
}

TEST(TestApiSubs, Test_scalar_latter) {
  // 构造测试输入和预期结果
  int a = 2, b = 32;
  auto *x = (half*)AscendC::GmAlloc(sizeof(half) * a * b); // 全局内存输入
  auto *y = (half*)AscendC::GmAlloc(sizeof(half) * a * b);

  half expect[a][b];

  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      x[i * b + j] = (double)(i + 1);
      expect[i][j] = (double)((i + 1) - 2.0);
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, half *x, half *y) {
    half constant_x = (double)(2.0);
    uint32_t cal_cnt = a * b;
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * b);  // 输入缓冲区
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b);  // 输出缓冲区
    tpipe.InitBuffer(tmp, 8 * 1024);               // 临时工作区

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a * b);

    Subs<half>(l_y, l_x, constant_x, cal_cnt, l_tmp);

    UbToGm(y, l_y, a * b);                          // 数据搬出
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (double)(y[i*b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiSubs, Test_scalar_front) {
  // 构造测试输入和预期结果
  int a = 2, b = 4;
  auto *x = (half*)AscendC::GmAlloc(sizeof(half) * a * b);
  auto *y = (half*)AscendC::GmAlloc(sizeof(half) * a * b);

  half expect[a][b];

  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      x[i * b + j] = (double)(i + 1);
      expect[i][j] = (double)(2.0 - (i + 1));
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, half *x, half *y) {
    half constant_x = (double)(2.0);
    uint32_t cal_cnt = 2 * 4;
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * b);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * b);
    GmToUb(l_y, y, a * b);

    Subs<half, false>(l_y, l_x, constant_x, cal_cnt, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (double)(y[i*b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
}

TEST(TestApiSubs, Test_scalar_front_float) {
  // 构造测试输入和预期结果
  int calc_size = 3840;
  auto *x = (float*)AscendC::GmAlloc(sizeof(float) * calc_size);
  auto *y = (float*)AscendC::GmAlloc(sizeof(float) * calc_size);

  float expect[calc_size];

  for (int i = 0; i < calc_size; i++) {
    x[i] = (float)(i + 1);
    expect[i] = (float)(4000.0 - (i + 1));
  }

  // 构造Api调用函数
  auto kernel = [](float *x, float *y) {
    float constant_x = (float)(4000.0);
    uint32_t cal_cnt = 3840;
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(float) * cal_cnt);
    tpipe.InitBuffer(ybuf, sizeof(float) * cal_cnt);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<float>();
    auto l_y = ybuf.Get<float>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, cal_cnt);
    GmToUb(l_y, y, cal_cnt);

    Subs<float, false>(l_y, l_x, constant_x, cal_cnt, l_tmp);

    UbToGm(y, l_y, cal_cnt);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < calc_size; i++) {
    auto diff = (float)(y[i] - expect[i]);
    if (diff < -1e-5 || diff > 1e-5) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0);
}


TEST(TestApiSubs, Test_scalar_both) {
  // 构造Api调用函数
  auto kernel = [](int a, int b, half *y) {
    half constant_x = (double)(2.0);
    half constant_y = (double)(1.0);
    uint32_t cal_cnt = 2 * 4;
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> ybuf, tmp;
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_y, y, a * b);

    Subs<half>(l_y, constant_x,  constant_y);

    UbToGm(y, l_y, a * b);
  };

  // 构造测试输入和预期结果
  int a = 2, b = 4;
  auto *y = (half*)AscendC::GmAlloc(sizeof(half) * a * b);

  half expect[a][b];

  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      expect[i][j] = (double)(2.0 - (1.0));
    }
  }

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (double)(y[i*b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
}