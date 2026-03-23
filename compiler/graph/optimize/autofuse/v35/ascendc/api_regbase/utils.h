/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_REGBASE_API_UTILS_H__
#define __ASCENDC_REGBASE_API_UTILS_H__

namespace {

constexpr static AscendC::MicroAPI::CastTrait cast_trait_none = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_float_2_half = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_float_2_int64 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_TRUNC,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_float_2_int32 = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_TRUNC,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_int32_2_float = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_int32_2_int16 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_int64_2_float = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_bf16_2_float = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_float_2_bf16 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_half_2_int8 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_TRUNC,
};

constexpr static AscendC::MicroAPI::CastTrait cast_trait_int16_2_uint8 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::UNKNOWN,
};

static constexpr MicroAPI::DivSpecificMode high_precision_div_mode = {MicroAPI::MaskMergeMode::ZEROING, true};

}
#endif  // __ASCENDC_REGBASE_API_UTILS_H__