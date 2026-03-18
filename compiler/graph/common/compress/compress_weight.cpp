/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <sstream>
#include <securec.h>
#include "compress.h"
#include "log.h"
#include "compress_weight.h"

#define CHECK_NOTNULLPTR(val, msg, expr) \
    do {                                 \
        if ((val) == nullptr) {          \
            LogFatal(msg);               \
            expr;                        \
        }                                \
    } while (0)

namespace {
struct CompressParametersConv2d {
    uint64_t weightSizeTotal = 0;
    size_t weightK = 0U;
    size_t weightN = 0U;
    size_t weightCo0 = 0U;
    size_t weightCo1 = 0U;
    size_t kbFrac = 0U;
    size_t nbFrac = 0U;
    size_t nRemainders = 0U;
    size_t wDtypeSize = 0U;
    size_t totalCompressedLength = 0U;
    size_t dataBase = 0U;
    size_t indexBase = 0U;
    size_t tailBlockWeightSize = 0U;
    size_t tailBlockSize = 0U;
    size_t blockSize = 0U;
    size_t firstWeightSize = 0U;
    size_t indexUnitSize = 0U;
};

const int INDEX_SIZE_COMPACT_MODE = 8;
const int IDX_WEIGHT_SHAPE_K = 0;
const int IDX_WEIGHT_SHAPE_N = 1;
const int IDX_WEIGHT_SHAPE_CO0 = 2;
const int IDX_WEIGHT_SHAPE_CO1 = 3;
const size_t WEIGHT_SPARSE_BLOCK = 4;
const size_t NONZERO_THRESHOLD = 2;
const size_t K0N0_PRODUCT = 512;
const size_t ZERO_REMAINDER = 0;

CmpStatus RearrangeConv2d(const char *const input, char *const weightRe,
                          size_t compressTilingK, size_t compressTilingN,
                          CompressParametersConv2d& compressParameters)
{
    if (compressTilingK <= 0 || compressTilingN <= 0) {
        LogFatal("compressTiling is invalid: compressTilingK("
                 << compressTilingK << ") compressTilingN(" << compressTilingN
                 << ")");
        return RET_ERROR;
    }

    int weightNum = 0;
    size_t jRange = compressTilingN;
    for (size_t n = 0; n < compressParameters.nbFrac; n++) {
        if ((n == compressParameters.nbFrac - 1) && (compressParameters.nRemainders > 0)) {
            jRange = compressParameters.nRemainders;
        }
        for (size_t k = 0; k < compressParameters.kbFrac; k++) {
            for (size_t i = 0; i < compressTilingK; i++) {
                for (size_t j = 0; j < jRange; j++) {
                    size_t start = (j + i * compressParameters.weightN + n * compressTilingN +
                                    k * compressTilingK * compressParameters.weightN) *
                                   (compressParameters.weightCo0 * compressParameters.weightCo1);
                    size_t end = start + (compressParameters.weightCo0 * compressParameters.weightCo1);
                    for (size_t x = start; x < end; x++) {
                        weightRe[weightNum] = input[x];
                        weightNum += 1;
                    }
                }
            }
        }
    }
    return RET_SUCCESS;
}

void LogCompressOpConfig(CompressOpConfig *const param)
{
    std::stringstream ss;
    ss << std::endl << "==== compress op param ====" << std::endl;
    ss << "shape of weight: ";
    for (int i = 0; i < SHAPE_SIZE_WEIGHT; ++i) {
        ss << param->wShape[i] << " ";
    }
    ss << std::endl;
    ss << "tiling k: " << param->compressTilingK << std::endl;
    ss << "tiling n: " << param->compressTilingN << std::endl;
    ss << "compress param:" << std::endl;
    ss << "    inputSize: " << param->compressConfig.inputSize << std::endl;
    ss << "    engineNum: " << param->compressConfig.engineNum << std::endl;
    ss << "    maxRatio: " << param->compressConfig.maxRatio << std::endl;
    ss << "    channel: " << param->compressConfig.channel << std::endl;
    ss << "    fractalSize: " << param->compressConfig.fractalSize << std::endl;
    ss << "    isTight: " << param->compressConfig.isTight << std::endl;
    ss << "    initOffset: " << param->compressConfig.init_offset << std::endl;
    ss << "==== compress op param ====" << std::endl;
    Log(ss.str());
}

CmpStatus CompressConv2d(CompressOpConfig *const param, char *const weightRe, CompressParametersConv2d&
                         compressParameters, char *const zipBuffer,  char *const infoBuffer)
{
    char* weightData = const_cast<char*>(weightRe) + compressParameters.dataBase;
    if (param->compressConfig.inputSize == 0) {
        LogFatal("The inputSize of compressConfig is zero.");
        return RET_ERROR;
    }
    if (param->compressConfig.fractalSize == 0) {
        LogFatal("The fractalSize of compressConfig is zero.");
        return RET_ERROR;
    }
    size_t indexSize = param->compressConfig.inputSize / param->compressConfig.fractalSize * INDEX_SIZE_COMPACT_MODE;
    std::unique_ptr<char[]> compIdxBuffer(new (std::nothrow) char[indexSize]());
    CHECK_NOTNULLPTR(compIdxBuffer.get(), "compIdxBuffer is null pointers.", return RET_ERROR);
    std::unique_ptr<char[]> compDataBuffer(new (std::nothrow) char[param->compressConfig.inputSize]());
    CHECK_NOTNULLPTR(compDataBuffer.get(), "compDataBuffer is null pointers.", return RET_ERROR);
    size_t compressedLength = 0;
    param->compressConfig.init_offset = compressParameters.totalCompressedLength;
    if (compressParameters.dataBase + param->compressConfig.inputSize > compressParameters.weightSizeTotal) {
        LogFatal("weightSizeTotal is:"<<compressParameters.weightSizeTotal<<", datebase is:"<<
        compressParameters.dataBase<<", size is:"<<param->compressConfig.inputSize<<".");
        return RET_ERROR;
    }
    auto ret = CompressWeights(
        weightData, param->compressConfig, compIdxBuffer.get(),
        compDataBuffer.get(), compressedLength);
    if (ret != 0) {
        LogFatal("weight compress of Conv2d failed");
        return RET_ERROR;
    }
    auto res = memcpy_s(infoBuffer + compressParameters.indexBase, indexSize, compIdxBuffer.get(), indexSize);
    if (res != 0) {
        LogFatal("copy compress index failed.");
        return RET_ERROR;
    }
    if (compressParameters.totalCompressedLength + compressedLength > compressParameters.weightSizeTotal) {
        LogFatal("weightSizeTotal is:"<<compressParameters.weightSizeTotal<<", totalCompressedLength is:"<<
        compressParameters.totalCompressedLength<<", compressedLength is:"<<compressedLength<<".");
        return RET_ERROR;
    }
    res = memcpy_s(
        zipBuffer + compressParameters.totalCompressedLength, compressedLength,
        compDataBuffer.get(), compressedLength);
    if (res != 0) {
        LogFatal("copy compress data failed.");
        return RET_ERROR;
    }
    compressParameters.dataBase += param->compressConfig.inputSize;
    compressParameters.totalCompressedLength += compressedLength;
    compressParameters.indexBase += indexSize;
    return RET_SUCCESS;
}

CmpStatus CompressWeightConv2d(char *const weightRe, char *const zipBuffer,
                               char *const infoBuffer, CompressOpConfig *const param,
                               CompressParametersConv2d &compressParameters)
{
    if (compressParameters.nRemainders == 0) {
        size_t compressedLength = 0;
        auto ret = CompressWeights(weightRe, param->compressConfig, infoBuffer, zipBuffer, compressedLength);
        if (ret != 0) {
            LogFatal("weight compress of Conv2d failed");
            return RET_ERROR;
        }
    } else {
        // 一次性处理所有连续的正常块
        if (compressParameters.firstWeightSize != 0) {
            param->compressConfig.inputSize = compressParameters.firstWeightSize;
            auto ret = CompressConv2d(param, weightRe, compressParameters, zipBuffer, infoBuffer);
            if (ret != 0) {
                LogFatal("compress firstWeight failed");
                return RET_ERROR;
            }
        }
        for (size_t i = 0; i < compressParameters.kbFrac; i++) {
            // 先将block_size为param->compressConfig.fractalSize的压缩起来。
            param->compressConfig.inputSize = compressParameters.tailBlockWeightSize - compressParameters.tailBlockSize;
            param->compressConfig.fractalSize = compressParameters.blockSize;
            if (param->compressConfig.inputSize != 0) {
                auto ret = CompressConv2d(param, weightRe, compressParameters, zipBuffer, infoBuffer);
                if (ret != 0) {
                    LogFatal("compress tail failed");
                    return RET_ERROR;
                }
            }
            // 将尾块的尾块单独压缩
            param->compressConfig.inputSize = compressParameters.tailBlockSize;
            param->compressConfig.fractalSize = compressParameters.tailBlockSize;
            if (compressParameters.tailBlockSize != 0) {
                auto ret = CompressConv2d(param, weightRe, compressParameters, zipBuffer, infoBuffer);
                if (ret != 0) {
                    LogFatal("CompressTailOfTail failed");
                    return RET_ERROR;
                }
            }
        }
    }
    return RET_SUCCESS;
}

bool SetCompressParametersConv2dConv2d(CompressParametersConv2d& compressParameters, CompressOpConfig *const param)
{
    if (param->compressTilingK <= 0) {
        LogFatal("CompressTilingK in parameter must > 0.");
        return false;
    }
    if (param->compressTilingN <= 0) {
        LogFatal("CompressTilingN in parameter must > 0.");
        return false;
    }
    for (auto v : param->wShape) {
        if (v <= 0) {
            LogFatal("Dim of weight must > 0.");
            return false;
        }
    }
    size_t compressWeightSize = compressParameters.weightK * compressParameters.weightN *
    compressParameters.weightCo0 * compressParameters.weightCo1;
    if (compressWeightSize <= 0) {
        LogFatal("compressWeightSize must > 0.");
        return false;
    }
    compressParameters.weightSizeTotal = param->compressConfig.inputSize;
    compressParameters.weightK = param->wShape[IDX_WEIGHT_SHAPE_K];
    compressParameters.weightN = param->wShape[IDX_WEIGHT_SHAPE_N];
    compressParameters.weightCo0 = param->wShape[IDX_WEIGHT_SHAPE_CO0];
    compressParameters.weightCo1 = param->wShape[IDX_WEIGHT_SHAPE_CO1];
    compressParameters.kbFrac = compressParameters.weightK / param->compressTilingK;
    compressParameters.nbFrac = compressParameters.weightN / param->compressTilingN;
    compressParameters.nRemainders = compressParameters.weightN % param->compressTilingN;
    compressParameters.wDtypeSize = compressParameters.weightSizeTotal / compressWeightSize;
    if (compressParameters.nRemainders != 0) {
        compressParameters.firstWeightSize = param->compressTilingN * param->compressTilingK *
        compressParameters.weightCo0 * compressParameters.weightCo1 * compressParameters.kbFrac *
        compressParameters.nbFrac * compressParameters.wDtypeSize;
        compressParameters.nbFrac += 1;
    }
    compressParameters.blockSize = param->compressConfig.fractalSize;
    compressParameters.tailBlockWeightSize = compressParameters.nRemainders * param->compressTilingK *
    compressParameters.weightCo0 * compressParameters.weightCo1 * compressParameters.wDtypeSize;
    compressParameters.tailBlockSize = compressParameters.tailBlockWeightSize % compressParameters.blockSize;
    compressParameters.dataBase = 0;
    compressParameters.indexBase = 0;
    compressParameters.totalCompressedLength = 0;

    return true;
}
} // namespace

CmpStatus CompressWeightsConv2D(const char *const input, char *const zipBuffer,
                                char *const infoBuffer,
                                CompressOpConfig *const param)
{
    if (input == nullptr || zipBuffer == nullptr || infoBuffer == nullptr ||
        param == nullptr) {
        LogFatal("The input parameters contain null pointers.");
        return RET_ERROR;
    }
    if (param->compressConfig.inputSize <= 0) {
        LogFatal("inputSize in CompressConfig is invalid: "
                 << param->compressConfig.inputSize);
        return RET_ERROR;
    }

    LogCompressOpConfig(param);

    std::unique_ptr<char[]> weightRe(
        new (std::nothrow) char[param->compressConfig.inputSize]());
    CHECK_NOTNULLPTR(weightRe.get(), "weightRe is null pointers.", return RET_ERROR);
    CompressParametersConv2d compressParameters;
    if (!SetCompressParametersConv2dConv2d(compressParameters, param)) {
        return RET_ERROR;
    }

    auto res = RearrangeConv2d(input, weightRe.get(), param->compressTilingK,
                               param->compressTilingN, compressParameters);
    if (res == RET_ERROR) {
        LogFatal("rearrange weight failed");
        return RET_ERROR;
    }

    res = CompressWeightConv2d(weightRe.get(), zipBuffer, infoBuffer, param, compressParameters);
    if (res == RET_ERROR) {
        LogFatal("weight compress of Conv2d failed");
        return RET_ERROR;
    }

    return RET_SUCCESS;
}

CmpStatus SparseWeightsConv2D(const char *const input, size_t weight_size)
{
    /*
    input: weight data, int8 dtype.
    weight_size: The size of weight_data. Ci1HkWk*Co1*Co0*Ci0
    Check if there are over 2 nonzero elements in 4 elements, then return error.
    */

    if (input == nullptr) {
        LogFatal("Input weight is a null pointer!");
        return RET_ERROR;
    }

    if (weight_size % K0N0_PRODUCT != ZERO_REMAINDER) {
        // n0 * k0
        LogFatal("Invalid weight_size, which is not n0*k0 aligned!");
        return RET_ERROR;
    }

    for (size_t i = 0; i < weight_size / WEIGHT_SPARSE_BLOCK; i++) {
        size_t nonzeroNum = 0;

        for (size_t j = 0; j < WEIGHT_SPARSE_BLOCK; j++) {
            if (input[i * WEIGHT_SPARSE_BLOCK + j] != 0) {
                nonzeroNum += 1;
            }
        }

        if (nonzeroNum > NONZERO_THRESHOLD) {
            LogFatal("Weight sparse is not satisfied! Over 2 nonzero elements in 4 elements.");
            return RET_ERROR;
        }
    }

    return RET_SUCCESS;
}
