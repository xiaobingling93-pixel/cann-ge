/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PRELOAD_PRE_MODEL_TYPES_H_
#define GE_COMMON_PRELOAD_PRE_MODEL_TYPES_H_
#include <string>
#include <map>
#include "framework/common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
constexpr uint32_t DEFAULT_INFO_VALUE_ZERO = 0U;
constexpr uint32_t DEFAULT_INFO_VALUE_ONE = 1U;
constexpr uint32_t DEFAULT_INFO_VALUE_EIGHT = 8U;
constexpr uint32_t kAlignBy4B = 4U;

enum class NanoTaskDescType : uint16_t { NANO_AI_CORE, NANO_AI_CPU, NANO_PLACE_HOLDER, NANO_RESEVERD };

enum class NanoTaskPreStatus : uint16_t { NANO_PRE_DISABLE, NANO_PRE_ENABLE };

enum class NanoTaskSoftUserStatus : uint16_t { NANO_SOFTUSER_DEFAULT, NANO_SOFTUSER_HOSTFUNC };

enum class EngineType : uint32_t { kDefaultEngine, kNanoEngine };
const std::string kPreEngineAiCore = "aicore_engine";
const std::string kPreEngineAiCpu = "aicpu_engine";
const std::string kPreEngineNano = "nano_engine";
const std::string kPreEngineDefault = "default_engine";

// nano engine
const std::string kPreEngineNanoAiCore = "nano_aicore_engine";
const std::string kPreEngineNanoAiCpu = "nano_aicpu_engine";

// prefect attr name
const std::string kAttrWeightPrefetchType = "_weight_prefetch_type";
const std::string kAttrWeightPrefetchSrcOffset = "_weight_prefetch_src_offset";
const std::string kAttrWeightPrefetchDstOffset = "_weight_prefetch_dst_offset";
const std::string kAttrWeightPrefetchDataSize = "_weight_prefetch_data_size";
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_MODEL_UTILS_H_