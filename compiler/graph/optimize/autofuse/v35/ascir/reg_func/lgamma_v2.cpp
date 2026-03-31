/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func_v2.h"

namespace ge {
namespace ascir {
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLgammaTmpSizeV2(const ge::AscNode &node)
{
    constexpr uint32_t LGAMMA_ONE_REPEAT_BYTE_SIZE = 256;
    constexpr uint32_t HALF_CALC_FAC = 13U;
    constexpr uint32_t FLOAT_CALC_PROC = 8U;
    constexpr uint32_t NUM_TWO = 2U;
    auto node_inputs = node.inputs;
    GE_ASSERT_TRUE(node_inputs.Size() > 0, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
    uint32_t calcTmpBuf = (node_inputs[0].attr.dtype == ge::DT_FLOAT) ? FLOAT_CALC_PROC * LGAMMA_ONE_REPEAT_BYTE_SIZE
                                                                      : HALF_CALC_FAC * LGAMMA_ONE_REPEAT_BYTE_SIZE * NUM_TWO;
    GELOGD("Node %s[%s] temp buffer size: %u", node.GetTypePtr(), node.GetNamePtr(), calcTmpBuf);
    Expression TmpSize = ge::Symbol(calcTmpBuf);
    ge::TmpBufDesc desc = {TmpSize, -1};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> tmpBufDescs;
    tmpBufDescs.emplace_back(std::make_unique<ge::TmpBufDesc>(desc));
    return tmpBufDescs;
}
}
}