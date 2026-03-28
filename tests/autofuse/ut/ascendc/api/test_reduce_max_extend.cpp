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
#include "reduce_max.h"

constexpr uint32_t ELE_PER_BLK = 8;  // 32 bytes / 4 bytes = 8 int32_t

// Initialize padded input data
static void InitPaddedInput(int32_t* x, uint32_t first, uint32_t last, uint32_t padLast) {
    for (uint32_t row = 0; row < first; row++) {
        for (uint32_t col = 0; col < padLast; col++) {
            x[row * padLast + col] = (col < last) ? static_cast<int32_t>((row * 100 + col) % 200 - 50) : INT32_MIN;
        }
    }
}

// Compute expected result for AR mode: max along each row
static void ComputeExpectedAR(const int32_t* src, int32_t* dst, uint32_t first, uint32_t last, uint32_t padLast) {
    for (uint32_t row = 0; row < first; row++) {
        int32_t maxVal = src[row * padLast];
        for (uint32_t col = 0; col < last; col++) {
            maxVal = std::max(maxVal, src[row * padLast + col]);
        }
        dst[row] = maxVal;
    }
}

// Compute expected result for RA mode: max along each column
static void ComputeExpectedRA(const int32_t* src, int32_t* dst, uint32_t first, uint32_t last, uint32_t padLast) {
    for (uint32_t col = 0; col < last; col++) {
        int32_t maxVal = src[col];
        for (uint32_t row = 0; row < first; row++) {
            maxVal = std::max(maxVal, src[row * padLast + col]);
        }
        dst[col] = maxVal;
    }
}

// Test parameters structure
struct ReduceMaxTestParams {
    uint32_t first;
    uint32_t last;
    bool isAr;
};

// AR mode tests
class TestReduceMaxAR : public testing::Test, public testing::WithParamInterface<ReduceMaxTestParams> {};

TEST_P(TestReduceMaxAR, Calc) {
    auto param = GetParam();
    uint32_t first = param.first;
    uint32_t last = param.last;
    uint32_t padLast = (last + ELE_PER_BLK - 1) / ELE_PER_BLK * ELE_PER_BLK;

    auto* x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * first * padLast);
    auto* y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * first * ELE_PER_BLK);

    InitPaddedInput(x, first, last, padLast);

    std::vector<int32_t> expected(first);
    ComputeExpectedAR(x, expected.data(), first, last, padLast);

    auto kernel = [first, last, padLast](int32_t* x, int32_t* y) {
        TPipe tpipe;
        TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
        tpipe.InitBuffer(xbuf, sizeof(int32_t) * first * padLast);
        tpipe.InitBuffer(ybuf, sizeof(int32_t) * first * ELE_PER_BLK);
        tpipe.InitBuffer(tmp, sizeof(int32_t) * (first * ELE_PER_BLK + 72 + ELE_PER_BLK));
        auto l_x = xbuf.Get<int32_t>();
        auto l_y = ybuf.Get<int32_t>();
        auto l_tmp = tmp.Get<uint8_t>();
        GmToUb(l_x, x, first * padLast);
        uint32_t shape[] = {first, last};
        ReduceMaxExtend<int32_t, AscendC::Pattern::Reduce::AR, false>(l_y, l_x, l_tmp, shape, true);
        UbToGm(y, l_y, first * ELE_PER_BLK);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, x, y);

    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < first; i++) {
        if (y[i] != expected[i]) diff_count++;
    }
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(DiffShape, TestReduceMaxAR,
    ::testing::Values(
        ReduceMaxTestParams{8, 32, true},
        ReduceMaxTestParams{8, 16, true},
        ReduceMaxTestParams{4, 8, true},
        ReduceMaxTestParams{7, 1, true},
        ReduceMaxTestParams{1, 8, true}
    ));

// RA mode tests
class TestReduceMaxRA : public testing::Test, public testing::WithParamInterface<ReduceMaxTestParams> {};

TEST_P(TestReduceMaxRA, Calc) {
    auto param = GetParam();
    uint32_t first = param.first;
    uint32_t last = param.last;
    uint32_t padLast = (last + ELE_PER_BLK - 1) / ELE_PER_BLK * ELE_PER_BLK;

    auto* x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * first * padLast);
    auto* y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * padLast);

    InitPaddedInput(x, first, last, padLast);

    std::vector<int32_t> expected(last);
    ComputeExpectedRA(x, expected.data(), first, last, padLast);

    auto kernel = [first, last, padLast](int32_t* x, int32_t* y) {
        TPipe tpipe;
        TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
        tpipe.InitBuffer(xbuf, sizeof(int32_t) * first * padLast);
        tpipe.InitBuffer(ybuf, sizeof(int32_t) * padLast);
        tpipe.InitBuffer(tmp, sizeof(int32_t) * (first * padLast + 72));
        auto l_x = xbuf.Get<int32_t>();
        auto l_y = ybuf.Get<int32_t>();
        auto l_tmp = tmp.Get<uint8_t>();
        GmToUb(l_x, x, first * padLast);
        uint32_t shape[] = {first, last};
        ReduceMaxExtend<int32_t, AscendC::Pattern::Reduce::RA, false>(l_y, l_x, l_tmp, shape, true);
        UbToGm(y, l_y, padLast);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, x, y);

    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < last; i++) {
        if (y[i] != expected[i]) diff_count++;
    }
    EXPECT_EQ(diff_count, 0);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
}

INSTANTIATE_TEST_SUITE_P(DiffShape, TestReduceMaxRA,
    ::testing::Values(
        ReduceMaxTestParams{8, 32, false},
        ReduceMaxTestParams{8, 16, false},
        ReduceMaxTestParams{4, 64, false},
        ReduceMaxTestParams{16, 7, false},
        ReduceMaxTestParams{1, 16, false},
        ReduceMaxTestParams{16, 1, false}
    ));
