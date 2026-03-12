/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_type_utils.h"
#include <vector>
#include <sstream>
#include <climits>
#include <cctype>
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "common/fe_log.h"
#include "common/constants_define.h"
#include "common/string_utils.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_attr_define.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/attr_utils.h"
#include "base/err_msg.h"

#include "graph/utils/op_type_utils.h"
namespace fe {
namespace {
const std::string DATA = "Data";
const std::map<ge::Format, std::map<std::string, int32_t>> AXIS_INDEX_OF_FORMAT = {
        {ge::FORMAT_NCHW, {{"N", NCHW_DIM_N}, {"C", NCHW_DIM_C}, {"H", NCHW_DIM_H}, {"W", NCHW_DIM_W}}},
        {ge::FORMAT_HWCN, {{"N", HWCN_DIM_N}, {"C", HWCN_DIM_C}, {"H", HWCN_DIM_H}, {"W", HWCN_DIM_W}}},
        {ge::FORMAT_NHWC, {{"N", NHWC_DIM_N}, {"C", NHWC_DIM_C}, {"H", NHWC_DIM_H}, {"W", NHWC_DIM_W}}},
        {ge::FORMAT_CHWN, {{"N", CHWN_DIM_N}, {"C", CHWN_DIM_C}, {"H", CHWN_DIM_H}, {"W", CHWN_DIM_W}}},
        {ge::FORMAT_NDHWC, {{"N", NDHWC_DIM_N}, {"C", NDHWC_DIM_C}, {"H", NDHWC_DIM_H}, {"W", NDHWC_DIM_W},
                                  {"D", NDHWC_DIM_D}}},
        {ge::FORMAT_NCDHW, {{"N", NCDHW_DIM_N}, {"C", NCDHW_DIM_C}, {"H", NCDHW_DIM_H}, {"W", NCDHW_DIM_W},
                                  {"D", NCDHW_DIM_D}}},
        {ge::FORMAT_DHWCN, {{"N", DHWCN_DIM_N}, {"C", DHWCN_DIM_C}, {"H", DHWCN_DIM_H}, {"W", DHWCN_DIM_W},
                                  {"D", DHWCN_DIM_D}}},
        {ge::FORMAT_DHWNC, {{"N", DHWNC_DIM_N}, {"C", DHWNC_DIM_C}, {"H", DHWNC_DIM_H}, {"W", DHWNC_DIM_W},
                                  {"D", DHWNC_DIM_D}}}};

}
std::string GetRealPath(const std::string &path) {
  if (path.empty()) {
    FE_LOGI("path string is nullptr.");
    return "";
  }
  if (path.size() >= PATH_MAX) {
    FE_LOGI("file path %s is too long!", path.c_str());
    return "";
  }

  // PATH_MAX is the system marco，indicate the maximum length for file path
  // pclint check one param in stack can not exceed 1K bytes
  char resoved_path[PATH_MAX] = {0x00};

  std::string res;

  // path not exists or not allowed to read return nullptr
  // path exists and readable, return the resoved path
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    FE_LOGI("Path [%s] does not exist.", path.c_str());
  }
  return res;
}

std::string RemoveCharacters(const std::string &param_key) {
  std::string param_key_str;
  std::string key1 = "ge.";
  std::string key2 = "ge.exec.";
  size_t pos = param_key.find(key1);
  size_t pos1 = param_key.find(key2);
  if (pos == 0) {
    if (pos1 == 0) {
      param_key_str = param_key.substr(key2.size());
    } else {
      param_key_str = param_key.substr(key1.size());
    }
  } else {
    param_key_str = param_key;
  }
  return param_key_str;
}

Status String2DataType(const std::string &dtype_str, ge::DataType &dtype) {
  string fe_dtype_str = const_cast<string&>(dtype_str);
  if (fe_dtype_str == "float32") {
    dtype = ge::DT_FLOAT;
  } else {
    transform(fe_dtype_str.begin(), fe_dtype_str.end(), fe_dtype_str.begin(), ::toupper);
    std::string ge_dtype_string = "DT_" + fe_dtype_str;
    dtype = ge::TypeUtils::SerialStringToDataType(ge_dtype_string);
    if (ge::DT_UNDEFINED == dtype) {
      FE_LOGE("Did not find dtype %s in struct STR_DTYPE_MAP.", dtype_str.c_str());
      return fe::FAILED;
    }
  }
  return fe::SUCCESS;
}

Status String2Bool(const std::string &bool_str, bool &bool_res) {
  if (STR_BOOL_MAP.end() == STR_BOOL_MAP.find(bool_str)) {
    FE_LOGE("Fallback flag %s not found in struct STR_BOOL_MAP.", bool_str.c_str());
    return fe::FAILED;
  }
  bool_res = STR_BOOL_MAP.at(bool_str);
  return fe::SUCCESS;
}

std::string GetStrByFormatVec(const std::vector<ge::Format>& format_vec) {
  string result;
  size_t size = format_vec.size();
  for (size_t i = 0; i < size; ++i) {
    string format = ge::TypeUtils::FormatToSerialString(format_vec[i]);
    result += ge::TypeUtils::FormatToSerialString(format_vec[i]);
    if (i != size - 1) {
      result += ",";
    }
  }
  return result;
}

std::string GetStrBySubFormatVec(const std::vector<uint32_t>& sub_format_vec) {
  string result;
  size_t size = sub_format_vec.size();
  for (size_t i = 0; i < size; ++i) {
    result += std::to_string(sub_format_vec[i]);
    if (i != size - 1) {
      result += ",";
    }
  }
  return result;
}

std::string GetStrByDataTypeVec(const std::vector<ge::DataType>& data_type_vec) {
  std::string result;
  size_t size = data_type_vec.size();
  for (size_t i = 0; i < size; ++i) {
    std::string data_type = ge::TypeUtils::DataTypeToSerialString(data_type_vec[i]);
    result += data_type;
    if (i != size - 1) {
      result += ",";
    }
  }
  return result;
}

std::string GetOpPatternString(OpPattern op_pattern) {
  auto iter = OP_PATTERN_STRING_MAP.find(op_pattern);
  if (iter == OP_PATTERN_STRING_MAP.end()) {
    return "unknown-op-pattern";
  } else {
    return iter->second;
  }
}

std::string GetPrecisionPolicyString(PrecisionPolicy precision_policy) {
  auto iter = PRECISION_POLICY_STRING_MAP.find(precision_policy);
  if (iter == PRECISION_POLICY_STRING_MAP.end()) {
    return "unknown-precision-policy";
  } else {
    return iter->second;
  }
}

std::string L2CacheReadMode2Str(const L2CacheReadMode &read_mode) {
  if (L2CACHE_READ_MODE_STRING_MAP.end() == L2CACHE_READ_MODE_STRING_MAP.find(read_mode)) {
    return "UNDEFINED";
  }
  return L2CACHE_READ_MODE_STRING_MAP.at(read_mode);
}

std::string GetBufferOptimizeString(const BufferOptimize &buffer_optimize) {
  auto iter = BUFFER_OPTIMIZE_STRING_MAP.find(buffer_optimize);
  if (iter == BUFFER_OPTIMIZE_STRING_MAP.end()) {
    return BUFFER_OPTIMIZE_UNKNOWN;
  } else {
    return iter->second;
  }
}

bool IsMemoryEmpty(const ge::GeTensorDesc &tensor_desc) {
    auto memory_size_calc_type = static_cast<int64_t>(ge::MemorySizeCalcType::NORMAL);
    (void)ge::AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE, memory_size_calc_type);
    return memory_size_calc_type == static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY);
}

bool HasNullableOutput(const ge::GeTensorDesc &tensor_desc) {
 	bool is_null_output = false;
 	bool has_null_output_attr = ge::AttrUtils::GetBool(tensor_desc, ATTR_NAME_IS_NULL_OUTPUT, is_null_output);
 	if (!has_null_output_attr) {
 	  return false;
 	}
 	return is_null_output;
}

bool IsSubGraphData(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr || op_desc_ptr->GetType() != DATA) {
    return false;
  }
  return op_desc_ptr->HasAttr(ge::ATTR_NAME_PARENT_NODE_INDEX);
}

bool IsSubGraphNetOutput(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr || op_desc_ptr->GetType() != "NetOutput") {
    return false;
  }
  for (auto &tensor : op_desc_ptr->GetAllInputsDescPtr()) {
    if (ge::AttrUtils::HasAttr(tensor, ge::ATTR_NAME_PARENT_NODE_INDEX)) {
      return true;
    }
  }
  return false;
}

bool CheckFallbackAclnn(const ge::OpDescPtr &op_desc_ptr) {
  bool fallback_aclnn_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_FALLBACK_ACLNN, fallback_aclnn_flag);
  return fallback_aclnn_flag;
}

int32_t GetAxisIndexByFormat(const ge::Format& format, const string& axis) {
  auto iter = AXIS_INDEX_OF_FORMAT.find(format);
  if (iter != AXIS_INDEX_OF_FORMAT.end()) {
    auto iter2 = iter->second.find(axis);
    if (iter2 != iter->second.end()) {
      return iter2->second;
    } else {
      FE_LOGW("Unsupported axis: %s", axis.c_str());
      return -1;
    }
  } else {
    FE_LOGW("Do not support this format %s", ge::TypeUtils::FormatToSerialString(format).c_str());
    return -1;
  }
}
}  // namespace fe
