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
#include <cstdlib>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace AscendC;

#include "test_api_utils.h"
#include "utils.h"
#include "abs.h"

constexpr inline double AbsExpectGen(const double x) {
    // INT_MIN(-2147483648)的Not+1在int32中溢出回绕，AbsExtend返回-2147483648
    if (x == -2147483648.0) return -2147483648.0;
    return x >= 0 ? x : -x;
}

constexpr inline double AbsSrcGen(int index) {
    constexpr int values[] = {0, 1, -1, 2, -2, 100, -100, 42, -42, 7, -7,
                              2147483647, -2147483648, 2147483646, -2147483647,
                              0, 1, -1, 2, -2, 100, -100, 42, -42, 7, -7,
                              2147483647, -2147483648, 2147483646, -2147483647};
    return static_cast<double>(values[index % 32]);
}

// Non-in-place test using UnaryTest utility
class TestAbsExtendNonInplace : public testing::Test, public testing::WithParamInterface<size_t> {};
TEST_P(TestAbsExtendNonInplace, Calc) {
    int size = this->GetParam();
    UnaryTest<int32_t>(size, AbsExtend<int32_t>, AbsExpectGen, AbsSrcGen);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestAbsExtendNonInplace,
        ::testing::Values(
            /* 1 block */ ONE_BLK_SIZE / sizeof(int32_t),
            /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
            /* max repeat */ MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
            /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
            /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t),
            /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
            /* mix block, repeat, max repeat */
                ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                sizeof(int32_t)));

// In-place test: dst and src share the same buffer
void InplaceAbsExtendCalc(int32_t* x, int32_t* y, int size) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * size);
    tpipe.InitBuffer(tmp, sizeof(int32_t) * size);

    auto l_x = xbuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, size);
    AbsExtend<int32_t>(l_x, l_x, size, l_tmp);
    UbToGm(y, l_x, size);
}

class TestAbsExtendInplace : public testing::Test, public testing::WithParamInterface<size_t> {};
TEST_P(TestAbsExtendInplace, Calc) {
    int size = this->GetParam();

    auto *x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * size);
    auto *y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * size);

    int32_t expect[32];
    for (int i = 0; i < 32; i++) {
        expect[i] = static_cast<int32_t>(AbsExpectGen(AbsSrcGen(i)));
    }

    for (int i = 0; i < size; i++) {
        x[i] = static_cast<int32_t>(AbsSrcGen(i));
    }

    auto kernel = [](int size, int32_t *x, int32_t *y) {
        InplaceAbsExtendCalc(x, y, size);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, size, x, y);

    int diff_count = 0;
    for (int i = 0; i < size; i++) {
        if (y[i] != expect[i % 32]) {
            diff_count++;
        }
    }
    EXPECT_EQ(diff_count, 0);
}

INSTANTIATE_TEST_SUITE_P(DiffLength, TestAbsExtendInplace,
        ::testing::Values(
            /* 1 block */ ONE_BLK_SIZE / sizeof(int32_t),
            /* 1 repeat */ ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
            /* max repeat */ MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
            /* less than 1 block */ (ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t),
            /* less than 1 repeat */ (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(int32_t),
            /* less than max repeat */ (MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / sizeof(int32_t),
            /* mix block, repeat, max repeat */
                ((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE + (ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) +
                 (ONE_BLK_SIZE - sizeof(int32_t))) /
                sizeof(int32_t)));
