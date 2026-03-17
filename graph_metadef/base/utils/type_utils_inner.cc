/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils_inner.h"
#include <algorithm>
#include <map>
#include "graph/types.h"
#include "base/err_msg.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {

const std::map<std::string, Format> kStringToFormatMap = {
    {"NCHW", FORMAT_NCHW},
    {"NHWC", FORMAT_NHWC},
    {"ND", FORMAT_ND},
    {"NC1HWC0", FORMAT_NC1HWC0},
    {"FRACTAL_Z", FORMAT_FRACTAL_Z},
    {"NC1C0HWPAD", FORMAT_NC1C0HWPAD},
    {"NHWC1C0", FORMAT_NHWC1C0},
    {"FSR_NCHW", FORMAT_FSR_NCHW},
    {"FRACTAL_DECONV", FORMAT_FRACTAL_DECONV},
    {"C1HWNC0", FORMAT_C1HWNC0},
    {"FRACTAL_DECONV_TRANSPOSE", FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {"FRACTAL_DECONV_SP_STRIDE_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {"NC1HWC0_C04", FORMAT_NC1HWC0_C04},
    {"FRACTAL_Z_C04", FORMAT_FRACTAL_Z_C04},
    {"CHWN", FORMAT_CHWN},
    {"DECONV_SP_STRIDE8_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {"NC1KHKWHWC0", FORMAT_NC1KHKWHWC0},
    {"BN_WEIGHT", FORMAT_BN_WEIGHT},
    {"FILTER_HWCK", FORMAT_FILTER_HWCK},
    {"HWCN", FORMAT_HWCN},
    {"LOOKUP_LOOKUPS", FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {"LOOKUP_KEYS", FORMAT_HASHTABLE_LOOKUP_KEYS},
    {"LOOKUP_VALUE", FORMAT_HASHTABLE_LOOKUP_VALUE},
    {"LOOKUP_OUTPUT", FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {"LOOKUP_HITS", FORMAT_HASHTABLE_LOOKUP_HITS},
    {"MD", FORMAT_MD},
    {"C1HWNCoC0", FORMAT_C1HWNCoC0},
    {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
    {"FRACTAL_NZ_C0_16", FORMAT_FRACTAL_NZ_C0_16},
    {"FRACTAL_NZ_C0_32", FORMAT_FRACTAL_NZ_C0_32},
    {"FRACTAL_NZ_C0_2", FORMAT_FRACTAL_NZ_C0_2},
    {"FRACTAL_NZ_C0_4", FORMAT_FRACTAL_NZ_C0_4},
    {"FRACTAL_NZ_C0_8", FORMAT_FRACTAL_NZ_C0_8},
    {"NDHWC", FORMAT_NDHWC},
    {"NCDHW", FORMAT_NCDHW},
    {"DHWCN", FORMAT_DHWCN},
    {"DHWNC", FORMAT_DHWNC},
    {"NDC1HWC0", FORMAT_NDC1HWC0},
    {"FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
    {"FRACTAL_Z_3D_TRANSPOSE", FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {"CN", FORMAT_CN},
    {"NC", FORMAT_NC},
    {"FRACTAL_ZN_LSTM", FORMAT_FRACTAL_ZN_LSTM},
    {"FRACTAL_Z_G", FORMAT_FRACTAL_Z_G},
    {"FORMAT_RESERVED", FORMAT_RESERVED},
    {"ALL", FORMAT_ALL},
    {"NULL", FORMAT_NULL},
    // add for json input
    {"ND_RNN_BIAS", FORMAT_ND_RNN_BIAS},
    {"FRACTAL_ZN_RNN", FORMAT_FRACTAL_ZN_RNN},
    {"NYUV", FORMAT_NYUV},
    {"NYUV_A", FORMAT_NYUV_A},
    {"NCL", FORMAT_NCL},
    {"FRACTAL_Z_WINO", FORMAT_FRACTAL_Z_WINO},
    {"C1HWC0", FORMAT_C1HWC0},
    {"RESERVED", FORMAT_RESERVED},
    {"UNDEFINED", FORMAT_RESERVED}
  };

const std::map<std::string, DataType> kStringTodataTypeMap = {
    {"DT_UNDEFINED", DT_UNDEFINED},            // Used to indicate a DataType field has not been set.
    {"DT_FLOAT", DT_FLOAT},                    // float type
    {"DT_FLOAT16", DT_FLOAT16},                // fp16 type
    {"DT_INT8", DT_INT8},                      // int8 type
    {"DT_INT16", DT_INT16},                    // int16 type
    {"DT_UINT16", DT_UINT16},                  // uint16 type
    {"DT_UINT8", DT_UINT8},                    // uint8 type
    {"DT_INT32", DT_INT32},                    // uint32 type
    {"DT_INT64", DT_INT64},                    // int64 type
    {"DT_UINT32", DT_UINT32},                  // unsigned int32
    {"DT_UINT64", DT_UINT64},                  // unsigned int64
    {"DT_BOOL", DT_BOOL},                      // bool type
    {"DT_DOUBLE", DT_DOUBLE},                  // double type
    {"DT_DUAL", DT_DUAL},                      // dual output type
    {"DT_DUAL_SUB_INT8", DT_DUAL_SUB_INT8},    // dual output int8 type
    {"DT_DUAL_SUB_UINT8", DT_DUAL_SUB_UINT8},  // dual output uint8 type
    {"DT_COMPLEX32", DT_COMPLEX32},            // complex32 type
    {"DT_COMPLEX64", DT_COMPLEX64},            // complex64 type
    {"DT_COMPLEX128", DT_COMPLEX128},          // complex128 type
    {"DT_QINT8", DT_QINT8},                    // qint8 type
    {"DT_QINT16", DT_QINT16},                  // qint16 type
    {"DT_QINT32", DT_QINT32},                  // qint32 type
    {"DT_QUINT8", DT_QUINT8},                  // quint8 type
    {"DT_QUINT16", DT_QUINT16},                // quint16 type
    {"DT_RESOURCE", DT_RESOURCE},              // resource type
    {"DT_STRING_REF", DT_STRING_REF},          // string ref type
    {"DT_STRING", DT_STRING},                  // string type
    // add for json input
    {"DT_FLOAT32", DT_FLOAT},
    {"DT_VARIANT", DT_VARIANT},                // dt_variant type
    {"DT_BFLOAT16", DT_BF16},                  // dt_bf16 type
    {"DT_INT4", DT_INT4},                      // dt_int4 type
    {"DT_UINT1", DT_UINT1},                    // dt_uint1 type
    {"DT_INT2", DT_INT2},                      // dt_int2 type
    {"DT_UINT2", DT_UINT2},                    // dt_uint2 type
    {"DT_HIFLOAT8", DT_HIFLOAT8},
    {"DT_FLOAT8_E5M2", DT_FLOAT8_E5M2},
    {"DT_FLOAT8_E4M3FN", DT_FLOAT8_E4M3FN},
    {"DT_FLOAT8_E8M0", DT_FLOAT8_E8M0},
    {"DT_FLOAT6_E3M2", DT_FLOAT6_E3M2},  // mxfp6
    {"DT_FLOAT6_E2M3", DT_FLOAT6_E2M3},  // mxfp6
    {"DT_FLOAT4_E2M1", DT_FLOAT4_E2M1},  // mxfp4
    {"DT_FLOAT4_E1M2", DT_FLOAT4_E1M2},  // mxfp4
    {"RESERVED", DT_UNDEFINED},                // RESERVED will be deserialized to DT_UNDEFINED
};

}


bool TypeUtilsInner::IsDataTypeValid(const DataType dt) {
  const uint32_t num = static_cast<uint32_t>(dt);
  GE_CHK_BOOL_EXEC((num < DT_MAX),
                   REPORT_INNER_ERR_MSG("E18888", "param dt:%u >= DT_MAX:%d, check invalid", num, DT_MAX);
                   return false, "[Check][Param] The DataType is invalid, dt:%u >= DT_MAX:%d", num, DT_MAX);
  return true;
}

bool TypeUtilsInner::IsFormatValid(const Format format) {
  const uint32_t num = static_cast<uint32_t>(GetPrimaryFormat(format));
  GE_CHK_BOOL_EXEC((num <= FORMAT_RESERVED),
                   REPORT_INNER_ERR_MSG("E18888", "The Format is invalid, num:%u > FORMAT_RESERVED:%d",
                                      num, FORMAT_RESERVED);
                   return false,
                   "[Check][Param] The Format is invalid, num:%u > FORMAT_RESERVED:%d", num, FORMAT_RESERVED);
  return true;
}

bool TypeUtilsInner::IsDataTypeValid(std::string dt) {
  (void)transform(dt.begin(), dt.end(), dt.begin(), &::toupper);
  const std::string key = "DT_" + dt;
  const auto it = kStringTodataTypeMap.find(key);
  if (it == kStringTodataTypeMap.end()) {
    return false;
  }
  return true;
}

bool TypeUtilsInner::IsFormatValid(std::string format) {
  (void)transform(format.begin(), format.end(), format.begin(), &::toupper);
  const auto it = kStringToFormatMap.find(format);
  if (it == kStringToFormatMap.end()) {
    return false;
  }
  return true;
}

graphStatus TypeUtilsInner::SplitFormatFromStr(const std::string &str,
                                               std::string &primary_format_str, int32_t &sub_format) {
  const size_t split_pos = str.find_first_of(':');
  if (split_pos != std::string::npos) {
    const std::string sub_format_str = str.substr(split_pos + 1U);
    try {
      primary_format_str = str.substr(0U, split_pos);
      if (std::any_of(sub_format_str.cbegin(), sub_format_str.cend(),
                      [](const char_t c) { return !static_cast<bool>(isdigit(static_cast<unsigned char>(c))); })) {
        REPORT_INNER_ERR_MSG("E18888", "sub_format: %s is not digital.", sub_format_str.c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is not digital.", sub_format_str.c_str());
        return GRAPH_FAILED;
      }
      sub_format = std::stoi(sub_format_str);
    } catch (std::invalid_argument &) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %s is invalid.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is invalid.", sub_format_str.c_str());
      return GRAPH_FAILED;
    } catch (std::out_of_range &) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %s is out of range.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is out of range.", sub_format_str.c_str());
      return GRAPH_FAILED;
    } catch (...) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %s cannot change to int.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s cannot change to int.", sub_format_str.c_str());
      return GRAPH_FAILED;
    }
    if (sub_format > 0xFFFF) {
      REPORT_INNER_ERR_MSG("E18888", "sub_format: %d is out of range [0, 0xffff].", sub_format);
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %d is out of range [0, 0xffff].", sub_format);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

bool TypeUtilsInner::CheckUint64MulOverflow(const uint64_t a, const uint32_t b) {
  // Not overflow
  if (a == 0U) {
    return false;
  }
  if (b <= (std::numeric_limits<uint64_t>::max() / a)) {
    return false;
  }
  return true;
}
}  // namespace ge
