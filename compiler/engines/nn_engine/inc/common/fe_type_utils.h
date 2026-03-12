/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_FE_TYPE_UTILS_H_
#define FUSION_ENGINE_INC_COMMON_FE_TYPE_UTILS_H_

#include "common/aicore_util_types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
std::string GetRealPath(const std::string &path);
std::string RemoveCharacters(const std::string &param_key);
Status String2DataType(const std::string &dtype_str, ge::DataType &dtype);
Status String2Bool(const std::string &bool_str, bool &bool_res);
std::string GetStrByFormatVec(const std::vector<ge::Format>& format_vec);
std::string GetStrBySubFormatVec(const std::vector<uint32_t>& sub_format_vec);
std::string GetStrByDataTypeVec(const std::vector<ge::DataType>& data_type_vec);

std::string GetOpPatternString(OpPattern op_pattern);

std::string GetPrecisionPolicyString(PrecisionPolicy precision_policy);

std::string L2CacheReadMode2Str(const L2CacheReadMode &read_mode);

std::string GetBufferOptimizeString(const BufferOptimize &buffer_optimize);

bool IsMemoryEmpty(const ge::GeTensorDesc &tensor_desc);

bool HasNullableOutput(const ge::GeTensorDesc &tensor_desc);

bool IsSubGraphData(const ge::OpDescPtr &op_desc_ptr);

bool IsSubGraphNetOutput(const ge::OpDescPtr &op_desc_ptr);

bool CheckFallbackAclnn(const ge::OpDescPtr &op_desc_ptr);

int32_t GetAxisIndexByFormat(const ge::Format &format, const string &axis);
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_FE_TYPE_UTILS_H_
