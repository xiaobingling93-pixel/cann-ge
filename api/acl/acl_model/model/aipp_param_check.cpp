/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aipp_param_check.h"
#include <string>
#include "utils/math_utils.h"
#include "platform/soc_spec.h"
#include "graph/types.h"

#define NPUARCH_TO_STR(arch) std::to_string(static_cast<uint32_t>(arch))

namespace {
    constexpr int32_t MIN_ALIGNMENT_YUV = 2;
    constexpr uint32_t TWO_CHANNEL = 2U;
    constexpr uint32_t THREE_CHANNEL = 3U;
    constexpr uint32_t FOUR_CHANNEL = 4U;
    constexpr uint32_t MULTIPLE = 16U;
}

namespace acl {
static aclError AippInputFormatCheck(const enum CceAippInputFormat inputFormat, const std::string &npuArch)
{
    bool flag = false;
    if (inputFormat < CCE_YUV420SP_U8) {
        ACL_LOG_INNER_ERROR("[Check][InputFormat]inputFormat must be set, cceInputFormat = %d",
            static_cast<int32_t>(inputFormat));
        return ACL_ERROR_INVALID_PARAM;
    }

    if (npuArch == NPUARCH_TO_STR(NpuArch::DAV_1001) || npuArch == NPUARCH_TO_STR(NpuArch::DAV_3002) ||
        npuArch == NPUARCH_TO_STR(NpuArch::DAV_2002) || npuArch == NPUARCH_TO_STR(NpuArch::DAV_2201) ||
        npuArch == NPUARCH_TO_STR(NpuArch::DAV_3510)) {
        flag = ((inputFormat != CCE_YUV420SP_U8) && (inputFormat != CCE_XRGB8888_U8) &&
               (inputFormat != CCE_RGB888_U8) && (inputFormat != CCE_YUV400_U8));
        if (flag) {
            ACL_LOG_INNER_ERROR("[Check][InputFormat]arch[%s] only support YUV420SP_U8, XRGB8888_U8, "
                "RGB888_U8, YUV400_U8, cceInputFormat = %d", npuArch.c_str(), static_cast<int32_t>(inputFormat));
            return ACL_ERROR_INVALID_PARAM;
        }
    } else {
        ACL_LOG_INNER_ERROR("[Check][Aipp]dynamic aipp not support arch[%s]", npuArch.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

static aclError AippSrcImageSizeCheck(const enum CceAippInputFormat inputFormat, const int32_t srcImageSizeW,
    const int32_t srcImageSizeH)
{
    bool flag = false;
    flag = ((srcImageSizeW == 0) || (srcImageSizeH == 0));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]srcImageSizeW and srcImageSizeH must be setted!");
        return ACL_ERROR_INVALID_PARAM;
    }

    if (inputFormat == CCE_YUV420SP_U8) {
        // determine whether it is even
        flag = (((srcImageSizeW % 2) != 0) || ((srcImageSizeH % 2) != 0));
        if (flag) {
            ACL_LOG_INNER_ERROR("[Check][Params]srcImageSizeH[%d] and srcImageSizeW[%d] must be even for YUV420SP_U8!",
                srcImageSizeH, srcImageSizeW);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    flag = ((inputFormat == CCE_YUV422SP_U8) || (inputFormat == CCE_YUYV_U8));
    if (flag) {
        // determine whether it is even
        if ((srcImageSizeW % 2) != 0) {
            ACL_LOG_INNER_ERROR("[Check][Params]srcImageSizeW[%d] must be even for YUV422SP_U8 and YUYV_U8!",
                srcImageSizeW);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}

aclError AippScfSizeCheck(const aclmdlAIPP *const aippParmsSet, const size_t batchIndex)
{
    if (aippParmsSet->aippBatchPara.empty()) {
        ACL_LOG_INNER_ERROR("[Check][Params]the size of aippBatchPara can't be zero");
        return ACL_ERROR_INVALID_PARAM;
    }

    const int32_t scfInputSizeW = aippParmsSet->aippBatchPara[batchIndex].scfInputSizeW;
    const int32_t scfInputSizeH = aippParmsSet->aippBatchPara[batchIndex].scfInputSizeH;
    const int32_t scfOutputSizeW = aippParmsSet->aippBatchPara[batchIndex].scfOutputSizeW;
    const int32_t scfOutputSizeH = aippParmsSet->aippBatchPara[batchIndex].scfOutputSizeH;
    const int32_t srcImageSizeW = aippParmsSet->aippParms.srcImageSizeW;
    const int32_t srcImageSizeH = aippParmsSet->aippParms.srcImageSizeH;

    const int8_t cropSwitch = aippParmsSet->aippBatchPara[batchIndex].cropSwitch;
    const int32_t cropSizeW = aippParmsSet->aippBatchPara[batchIndex].cropSizeW;
    const int32_t cropSizeH = aippParmsSet->aippBatchPara[batchIndex].cropSizeH;

    bool flag = false;
    if (cropSwitch == 1) {
        flag = ((scfInputSizeW != cropSizeW) || (scfInputSizeH != cropSizeH));
        if (flag) {
            ACL_LOG_INNER_ERROR("[Check][Params]when enable crop and scf, scfInputSizeW[%d] must be "
                "equal to cropSizeW[%d] and scfInputSizeH[%d] must be equal to cropSizeH[%d]!",
                scfInputSizeW, cropSizeW, scfInputSizeH, cropSizeH);
            return ACL_ERROR_INVALID_PARAM;
        }
    } else {
        flag = ((scfInputSizeW != srcImageSizeW) || (scfInputSizeH != srcImageSizeH));
        if (flag) {
            ACL_LOG_INNER_ERROR("[Check][Params]when disable crop and enable scf, scfInputSizeW[%d] "
                "must be equal to srcImageSizeW[%d] and scfInputSizeH[%d] must be equal to srcImageSizeH[%d]!",
                scfInputSizeW, srcImageSizeW, scfInputSizeH, srcImageSizeH);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    // scfInputSizeH mini value is 16, scfInputSizeH max value is 4096
    flag = ((scfInputSizeH < 16) || (scfInputSizeH > 4096));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]resize_input_h[%d] should be within [16, 4096]!",
            scfInputSizeH);
        return ACL_ERROR_INVALID_PARAM;
    }

    // scfInputSizeW mini value is 16, scfInputSizeW max value is 4096
    flag = ((scfInputSizeW < 16) || (scfInputSizeW > 4096));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]resize_input_w[%d] should be within [16, 4096]!", scfInputSizeW);
        return ACL_ERROR_INVALID_PARAM;
    }

    // scfOutputSizeW mini value is 16, scfOutputSizeW max value is 1920
    flag = ((scfOutputSizeW < 16) || (scfOutputSizeW > 1920));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]resize_output_w[%d] should be within [16, 1920]!", scfOutputSizeW);
        return ACL_ERROR_INVALID_PARAM;
    }

    // scfOutputSizeH mini value is 16, scfOutputSizeH max value is 4096
    flag = ((scfOutputSizeH < 16) || (scfOutputSizeH > 4096));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]resize_output_h[%d] should be within [16, 4096]!", scfOutputSizeH);
        return ACL_ERROR_INVALID_PARAM;
    }

    ge::float32_t scfRatio = (static_cast<ge::float32_t>(scfOutputSizeW) * 1.0F) / static_cast<ge::float32_t>(scfInputSizeW);
    // scf factor is within [1/16, 16]
    flag = ((scfRatio < (1.0F / 16.0F)) || (scfRatio > 16.0F));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]resize_output_w/resize_input_w[%f] should be within [1/16, 16]!",
            static_cast<ge::float64_t>(scfRatio));
        return ACL_ERROR_INVALID_PARAM;
    }

    scfRatio = (static_cast<ge::float32_t>(scfOutputSizeH) * 1.0F) / static_cast<ge::float32_t>(scfInputSizeH);
    // scf factor is within [1/16, 16]
    flag = ((scfRatio < (1.0F / 16.0F)) || (scfRatio > 16.0F));
    if (flag) {
        ACL_LOG_INNER_ERROR("[Check][Params]resize_output_h/resize_input_h[%f] should be within [1/16, 16]!",
            static_cast<ge::float64_t>(scfRatio));
        return ACL_ERROR_INVALID_PARAM;
    }

    return ACL_SUCCESS;
}

static aclError AippCropSizeCheck(const aclmdlAIPP *const aippParmsSet, const size_t batchIndex)
{
    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(!aippParmsSet->aippBatchPara.empty(), ACL_ERROR_INVALID_PARAM,
                                            "[Check][Params]the size of aippBatchPara can't be zero");

    const int32_t srcImageSizeW = aippParmsSet->aippParms.srcImageSizeW;
    const int32_t srcImageSizeH = aippParmsSet->aippParms.srcImageSizeH;

    const int32_t cropStartPosW = aippParmsSet->aippBatchPara[batchIndex].cropStartPosW;
    const int32_t cropStartPosH = aippParmsSet->aippBatchPara[batchIndex].cropStartPosH;
    const int32_t cropSizeW = aippParmsSet->aippBatchPara[batchIndex].cropSizeW;
    const int32_t cropSizeH = aippParmsSet->aippBatchPara[batchIndex].cropSizeH;

    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN((cropStartPosW + cropSizeW) <= srcImageSizeW, ACL_ERROR_INVALID_PARAM,
                                            "[Check][Params]the sum of cropStartPosW[%d] and cropSizeW[%d] can not be "
                                            "larger than srcImageSizeW[%d]", cropStartPosW, cropSizeW, srcImageSizeW);
    ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN((cropStartPosH + cropSizeH) <= srcImageSizeH, ACL_ERROR_INVALID_PARAM,
                                            "[Check][Params]the sum of cropStartPosH[%d] and cropSizeH[%d] can not be "
                                            "larger than srcImageSizeH[%d]", cropStartPosH, cropSizeH, srcImageSizeH);

    const enum CceAippInputFormat inputFormat =
        static_cast<enum CceAippInputFormat>(aippParmsSet->aippParms.inputFormat);
    if (inputFormat == CCE_YUV420SP_U8) {
        // determine whether it is even
        if (((cropStartPosW % MIN_ALIGNMENT_YUV) != 0) || ((cropStartPosH % MIN_ALIGNMENT_YUV) != 0)) {
            ACL_LOG_INNER_ERROR("[Check][Params]cropStartPosW[%d], cropStartPosH[%d] must be even for YUV420SP_U8!",
                cropStartPosW, cropStartPosH);
            return ACL_ERROR_INVALID_PARAM;
        }
    }
    if ((inputFormat == CCE_YUV422SP_U8) || (inputFormat == CCE_YUYV_U8)) {
        // determine whether it is even
        if ((cropStartPosW % MIN_ALIGNMENT_YUV) != 0) {
            ACL_LOG_INNER_ERROR("[Check][Params]cropStartPosW[%d] must be even for YUV422SP_U8 and YUYV_U8!",
                cropStartPosW);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}


aclError GetAippOutputHW(const aclmdlAIPP *const aippParmsSet, const size_t batchIndex, const std::string &npuArch,
                         int32_t &aippOutputW, int32_t &aippOutputH)
{
    if (aippParmsSet->aippBatchPara.empty()) {
        ACL_LOG_INNER_ERROR("[Check][Params]aippParmsSet->aippBatchPara is empty!");
        return ACL_ERROR_INVALID_PARAM;
    }

    const int8_t scfSwitch = aippParmsSet->aippBatchPara[batchIndex].scfSwitch;
    const int32_t scfOutputSizeW = aippParmsSet->aippBatchPara[batchIndex].scfOutputSizeW;
    const int32_t scfOutputSizeH = aippParmsSet->aippBatchPara[batchIndex].scfOutputSizeH;
    if (aippParmsSet->aippBatchPara[batchIndex].cropSwitch == 1) {
        aippOutputW = aippParmsSet->aippBatchPara[batchIndex].cropSizeW;
        aippOutputH = aippParmsSet->aippBatchPara[batchIndex].cropSizeH;

        if (scfSwitch == 1) {
            aippOutputW = scfOutputSizeW;
            aippOutputH = scfOutputSizeH;
        }
    } else if (scfSwitch == 1) {
        aippOutputW = scfOutputSizeW;
        aippOutputH = scfOutputSizeH;
    } else {
        aippOutputW = aippParmsSet->aippParms.srcImageSizeW;
        aippOutputH = aippParmsSet->aippParms.srcImageSizeH;
    }

    if (aippParmsSet->aippBatchPara[batchIndex].paddingSwitch == 1) {
	aippOutputW += aippParmsSet->aippBatchPara[batchIndex].paddingSizeLeft
	    + aippParmsSet->aippBatchPara[batchIndex].paddingSizeRight;
	aippOutputH += aippParmsSet->aippBatchPara[batchIndex].paddingSizeTop
	    + aippParmsSet->aippBatchPara[batchIndex].paddingSizeBottom;
        const bool flag =
            (npuArch == NPUARCH_TO_STR(NpuArch::DAV_1001)) || (npuArch == NPUARCH_TO_STR(NpuArch::DAV_2002)) ||
            (npuArch == NPUARCH_TO_STR(NpuArch::DAV_2201)) || (npuArch == NPUARCH_TO_STR(NpuArch::DAV_3510));
        if (flag) {
            ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(aippOutputW <= 1080, ACL_ERROR_INVALID_PARAM,
	              "[Check][Params]after padding, aipp output W[%d] should be less than or equal to 1080 for arch[%s]",
	              aippOutputW, npuArch.c_str());
        } else {
            ACL_LOG_INFO("no need to check aipp output width.");
        }
    }

    return ACL_SUCCESS;
}

static aclError AippDynamicBatchParaCheck(const aclmdlAIPP *const aippParmsSet, const std::string &npuArch)
{
    int8_t scfSwitch = 0;
    int8_t cropSwitch = 0;
    int32_t aippBatchOutputW = 0;
    int32_t aippBatchOutputH = 0;
    int32_t aippFirstOutputW = 0;
    int32_t aippFirstOutputH = 0;

    aclError result = GetAippOutputHW(aippParmsSet, 0UL, npuArch, aippFirstOutputW, aippFirstOutputH);
    if (result != ACL_SUCCESS) {
        return result;
    }

    const uint64_t batchSize = aippParmsSet->batchSize;
    bool flag = false;
    ACL_REQUIRES_LE(batchSize, aippParmsSet->aippBatchPara.size());
    for (uint64_t i = 0UL; i < batchSize; i++) {
        scfSwitch = aippParmsSet->aippBatchPara[i].scfSwitch;
        if (scfSwitch == 1) {
            ACL_LOG_INNER_ERROR("[Check][Params]Not support scf!");
            return ACL_ERROR_INVALID_PARAM;
        }

        cropSwitch = aippParmsSet->aippBatchPara[i].cropSwitch;
        if (cropSwitch == 1) {
            result = AippCropSizeCheck(aippParmsSet, static_cast<size_t>(i));
            if (result != ACL_SUCCESS) {
                return result;
            }
        }

        result = GetAippOutputHW(aippParmsSet, static_cast<size_t>(i), npuArch, aippBatchOutputW, aippBatchOutputH);
        if (result != ACL_SUCCESS) {
            return result;
        }

        flag = ((aippBatchOutputW != aippFirstOutputW) || (aippBatchOutputH != aippFirstOutputH));
        if (flag) {
            ACL_LOG_INNER_ERROR("[Check][Params]the %lu batch output size must be equal to the first "
                "batch aipp output size! aippBatchOutputW = %d, aippBatchOutputH = %d, aippFirstOutputW = %d, "
                "aippFirstOutputH = %d.", (i + 1UL), aippBatchOutputW, aippBatchOutputH, aippFirstOutputW,
                aippFirstOutputH);
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    return ACL_SUCCESS;
}

aclError AippParamsCheck(const aclmdlAIPP *const aippParmsSet, const std::string &npuArch)
{
    ACL_LOG_INFO("start to execute aclAippParamsCheck");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(aippParmsSet);

    const enum CceAippInputFormat inputFormat =
        static_cast<enum CceAippInputFormat>(aippParmsSet->aippParms.inputFormat);
    aclError result = AippInputFormatCheck(inputFormat, npuArch);
    if (result != ACL_SUCCESS) {
        return result;
    }

    const int8_t cscSwitch = aippParmsSet->aippParms.cscSwitch;
    bool flag = false;
    flag = ((inputFormat == CCE_YUV400_U8) || (inputFormat == CCE_RAW10) ||
        (inputFormat == CCE_RAW12) || (inputFormat == CCE_RAW16));
    if (flag) {
        if (cscSwitch == 1) {
            ACL_LOG_INNER_ERROR("[Check][Params]YUV400 or raw not support csc switch!");
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    const int32_t srcImageSizeW = aippParmsSet->aippParms.srcImageSizeW;
    const int32_t srcImageSizeH = aippParmsSet->aippParms.srcImageSizeH;

    result = AippSrcImageSizeCheck(inputFormat, srcImageSizeW, srcImageSizeH);
    if (result != ACL_SUCCESS) {
        return result;
    }

    result = AippDynamicBatchParaCheck(aippParmsSet, npuArch);
    if (result != ACL_SUCCESS) {
        return result;
    }

    return ACL_SUCCESS;
}


uint64_t GetSrcImageSize(const aclmdlAIPP *const aippParmsSet)
{
    if (aippParmsSet == nullptr) {
        return 0UL;
    }

    const enum CceAippInputFormat inputFormat =
        static_cast<enum CceAippInputFormat>(aippParmsSet->aippParms.inputFormat);
    const size_t srcImageSizeW = static_cast<size_t>(aippParmsSet->aippParms.srcImageSizeW);
    const size_t srcImageSizeH = static_cast<size_t>(aippParmsSet->aippParms.srcImageSizeH);
    const size_t batch = aippParmsSet->batchSize;
    size_t size = 0UL;

    if (inputFormat == CCE_YUV420SP_U8) {
        // YUV420SP_U8, one pixel use uint16, 2 bytes
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, THREE_CHANNEL * srcImageSizeH * srcImageSizeW / 2UL, size);
    } else if (inputFormat == CCE_XRGB8888_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, FOUR_CHANNEL * srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_RGB888_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, THREE_CHANNEL * srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_YUV400_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_ARGB8888_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, FOUR_CHANNEL * srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_YUYV_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, TWO_CHANNEL * srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_YUV422SP_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, TWO_CHANNEL * srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_AYUV444_U8) {
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, FOUR_CHANNEL * srcImageSizeH * srcImageSizeW, size);
    } else if (inputFormat == CCE_RAW10) {
        // RAW10, one pixel use uint16, 2 bytes
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, srcImageSizeH * srcImageSizeW * 2UL, size);
    } else if (inputFormat == CCE_RAW12) {
        // RAW12, one pixel use uint16, 2 bytes
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, srcImageSizeH * srcImageSizeW * 2UL, size);
    } else if (inputFormat == CCE_RAW16) {
        // RAW16, one pixel use uint16, 2 bytes
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, srcImageSizeH * srcImageSizeW * 2UL, size);
    } else if (inputFormat == CCE_RAW24) {
        // RAW24, one pixel use uint32, 4 bytes
        ACL_CHECK_ASSIGN_SIZET_MULTI_RET_NUM(batch, srcImageSizeH * srcImageSizeW * 4UL, size);
    } else {
        ACL_LOG_INFO("no need to check assign size.");
    }

    ACL_LOG_INFO("Input SrcImageSize = %lu, cce_InputFormat = %d", size, static_cast<int32_t>(inputFormat));
    return size;
}
} // namespace acl
