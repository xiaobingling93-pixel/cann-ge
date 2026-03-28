/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__
#define __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__

#include "ascendc_ir.h"

namespace ge {
namespace ascir {

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLog2TmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcModTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcReduceTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcErfTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGeluTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTanhTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcGatherTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcPowTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcExp2TmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcLgammaTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcVoidTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> GetCompareSizeV2([[maybe_unused]]const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSinTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAsinTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAsinhTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAtanTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAtanhTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcCosTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAcosTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcCoshTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcDigammaTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcErfcTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAcoshTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAtan2TmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcCeilTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcSinhTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTanTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTruncTmpSizeV2(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcXorTmpSizeV2(const ge::AscNode &node);
}  // namespace ascir
}  // namespace ge
#endif  // __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__
