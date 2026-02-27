/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "acl/ops/acl_cblas.h"


using namespace testing;
using namespace std;

static aclFloat16 ALPHA_HALF = 0;
static aclFloat16 BETA_HALF = 0;

static int32_t ALPHA_INT = 1;
static int32_t BETA_INT = 0;

class UTEST_ACL_BlasApiTest : public testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(UTEST_ACL_BlasApiTest, TestHgemmCreateHandle)
{
    aclopHandle *handle = nullptr;
    int m = 16;
    int k = 32;
    int n = 64;
    ASSERT_NE(aclblasCreateHandleForHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, nullptr), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreateHandleForHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_ERROR_OP_NOT_FOUND);
    ASSERT_NE(aclblasCreateHandleForHgemm(ACL_TRANS_NZ_T, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_NZ_T, m, n, k, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_LOW_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, -1, -1, k, ACL_COMPUTE_LOW_PRECISION, &handle), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_BlasApiTest, TestS8gemmCreateHandle)
{
    int m = 16;
    int k = 32;
    int n = 64;
    aclopHandle *handle = nullptr;
    ASSERT_NE(aclblasCreateHandleForS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, nullptr), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreateHandleForS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestHgemm)
{
    int m = 16;
    int k = 32;
    int n = 64;
    aclFloat16 *alphaHalf = &ALPHA_HALF;
    aclFloat16 *betaHalf = &BETA_HALF;
    aclrtStream stream = nullptr;
    aclFloat16 matrixA[16] = {0};
    aclFloat16 matrixB[16] = {0};
    aclFloat16 matrixC[16] = {0};

    ASSERT_NE(aclblasHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alphaHalf, nullptr, -1, matrixB, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alphaHalf, matrixA, -1, nullptr, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alphaHalf, matrixA, -1, matrixB, -1, betaHalf, nullptr, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);

    ASSERT_NE(aclblasHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alphaHalf, matrixA, -1, matrixB, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alphaHalf, matrixA, -1, matrixB, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestS8gemm)
{
    int m = 16;
    int k = 32;
    int n = 64;
    int32_t *alpha = &ALPHA_INT;
    int32_t *beta = &BETA_INT;
    aclrtStream stream = nullptr;
    int8_t matrixA8[16] = {0};
    int8_t matrixB8[16] = {0};
    int32_t matrixC8[16] = {0};
    ASSERT_NE(aclblasS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alpha, nullptr, -1, matrixB8, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alpha, matrixA8, -1, nullptr, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alpha, matrixA8, -1, matrixB8, -1, beta, nullptr, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);

    ASSERT_NE(aclblasS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alpha, matrixA8, -1, matrixB8, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k, alpha, matrixA8, -1, matrixB8, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestHgemvCreateHandle)
{
    int m = 16;
    int n = 64;
    aclopHandle *handle = nullptr;
    ASSERT_NE(aclblasCreateHandleForHgemv(ACL_TRANS_N, m, n, ACL_COMPUTE_HIGH_PRECISION, nullptr), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForHgemv(ACL_TRANS_N, m, n, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreateHandleForHgemv(ACL_TRANS_N, m, n, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestS8gemvCreateHandle)
{
    int m = 16;
    int n = 64;
    aclopHandle *handle = nullptr;
    ASSERT_NE(aclblasCreateHandleForS8gemv(ACL_TRANS_N, m, n, ACL_COMPUTE_HIGH_PRECISION, nullptr), ACL_SUCCESS);
    ASSERT_NE(aclblasCreateHandleForS8gemv(ACL_TRANS_N, m, n, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreateHandleForS8gemv(ACL_TRANS_N, m, n, ACL_COMPUTE_HIGH_PRECISION, &handle), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestHgemv)
{
    int m = 16;
    int n = 64;
    aclFloat16 *alphaHalf = &ALPHA_HALF;
    aclFloat16 *betaHalf = &BETA_HALF;
    aclrtStream stream = nullptr;
    aclFloat16 matrixA[16] = {0};
    aclFloat16 matrixB[16] = {0};
    aclFloat16 matrixC[16] = {0};
    ASSERT_NE(aclblasHgemv(ACL_TRANS_N, m, n, alphaHalf, nullptr, -1, matrixB, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasHgemv(ACL_TRANS_N, m, n, alphaHalf, matrixA, -1, nullptr, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasHgemv(ACL_TRANS_N, m, n, alphaHalf, matrixA, -1, matrixB, -1, betaHalf, nullptr, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);

    ASSERT_NE(aclblasHgemv(ACL_TRANS_N, m, n, alphaHalf, matrixA, -1, matrixB, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasHgemv(ACL_TRANS_N, m, n, alphaHalf, matrixA, -1, matrixB, -1, betaHalf, matrixC, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestS8gemv)
{
    int m = 16;
    int n = 64;
    int32_t *alpha = &ALPHA_INT;
    int32_t *beta = &BETA_INT;
    aclrtStream stream = nullptr;
    int8_t matrixA8[16] = {0};
    int8_t matrixB8[16] = {0};
    int32_t matrixC8[16] = {0};
    ASSERT_NE(aclblasS8gemv(ACL_TRANS_N, m, n, alpha, nullptr, -1, matrixB8, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasS8gemv(ACL_TRANS_N, m, n, alpha, matrixA8, -1, nullptr, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);
    ASSERT_NE(aclblasS8gemv(ACL_TRANS_N, m, n, alpha, matrixA8, -1, matrixB8, -1, beta, nullptr, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_SUCCESS);

    ASSERT_NE(aclblasS8gemv(ACL_TRANS_N, m, n, alpha, matrixA8, -1, matrixB8, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION,stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasS8gemv(ACL_TRANS_N, m, n, alpha, matrixA8, -1, matrixB8, -1, beta, matrixC8, -1, ACL_COMPUTE_HIGH_PRECISION, stream), ACL_ERROR_OP_NOT_FOUND);
}

TEST_F(UTEST_ACL_BlasApiTest, TestGemmEx)
{
    int m = 16;
    int k = 32;
    int n = 64;
    aclFloat16 *alphaHalf = &ALPHA_HALF;
    aclFloat16 *betaHalf = &BETA_HALF;
    aclrtStream stream = nullptr;
    aclFloat16 matrixA[16] = {0};
    aclFloat16 matrixB[16] = {0};
    aclFloat16 matrixC[16] = {0};
    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, 0, n, k,
                            alphaHalf, matrixA, -1, ACL_FLOAT16,
                            matrixB, -1, ACL_FLOAT16, betaHalf,
                            matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
              ACL_SUCCESS);
    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, 0, k,
                            alphaHalf, matrixA, -1, ACL_FLOAT16,
                            matrixB, -1, ACL_FLOAT16, betaHalf,
                            matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
              ACL_SUCCESS);

    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, 0,
                            alphaHalf, matrixA, -1, ACL_FLOAT16,
                            matrixB, -1, ACL_FLOAT16, betaHalf,
                            matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
              ACL_SUCCESS);

    ASSERT_NE(aclblasGemmEx(ACL_TRANS_T, ACL_TRANS_T, ACL_TRANS_N, m, n, k,
                        alphaHalf, matrixA, -1, ACL_FLOAT16,
                        matrixB, -1, ACL_FLOAT16, betaHalf,
                        matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
            ACL_SUCCESS);

    ASSERT_EQ(aclblasGemmEx(ACL_TRANS_T, ACL_TRANS_T, ACL_TRANS_N, m, n, k,
                    alphaHalf, matrixA, -1, ACL_FLOAT16,
                    matrixB, -1, ACL_FLOAT16, betaHalf,
                    matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
        ACL_ERROR_OP_NOT_FOUND);

    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k,
                        alphaHalf, matrixA, 1, ACL_FLOAT16,
                        matrixB, -1, ACL_FLOAT16, betaHalf,
                        matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
            ACL_SUCCESS);

    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k,
                        alphaHalf, matrixA, 1, ACL_FLOAT16,
                        matrixB, -1, ACL_FLOAT16, betaHalf,
                        matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_LOW_PRECISION, stream),
            ACL_SUCCESS);

    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k,
                    alphaHalf, matrixA, -1, ACL_FLOAT16,
                    matrixB, -1, ACL_FLOAT16, betaHalf,
                    matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_LOW_PRECISION, stream),
        ACL_SUCCESS);
    
    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_T, m, n, k,
                    alphaHalf, matrixA, -1, ACL_FLOAT16,
                    matrixB, -1, ACL_FLOAT16, betaHalf,
                    matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_LOW_PRECISION, stream),
        ACL_SUCCESS);
    ASSERT_NE(aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_T, -1, -1, k,
                    alphaHalf, matrixA, -1, ACL_FLOAT16,
                    matrixB, -1, ACL_FLOAT16, betaHalf,
                    matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_LOW_PRECISION, stream),
        ACL_SUCCESS);
    ASSERT_EQ(aclblasGemmEx(ACL_TRANS_NZ, ACL_TRANS_NZ, ACL_TRANS_NZ, m, n, k,
                            alphaHalf, matrixA, -1, ACL_FLOAT16,
                            matrixB, -1, ACL_FLOAT16, betaHalf,
                            matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
            ACL_ERROR_OP_NOT_FOUND);
    ASSERT_EQ(aclblasGemmEx(ACL_TRANS_NZ, ACL_TRANS_NZ, ACL_TRANS_NZ, m, n, k,
                            alphaHalf, matrixA, -1, ACL_INT8,
                            matrixB, -1, ACL_INT8, betaHalf,
                            matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
            ACL_ERROR_OP_NOT_FOUND);

    ASSERT_EQ(aclblasGemmEx(ACL_TRANS_NZ_T, ACL_TRANS_N, ACL_TRANS_N, m, n, k,
                            alphaHalf, matrixA, -1, ACL_FLOAT16,
                            matrixB, -1, ACL_FLOAT16, betaHalf,
                            matrixC, -1, ACL_FLOAT16, ACL_COMPUTE_HIGH_PRECISION, stream),
              ACL_ERROR_INVALID_PARAM);
}
