/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/datatype_util.h"
#include "proto/ge_ir.pb.h"

#include <map>

namespace {
std::map<ge::DataType, ge::proto::DataType> g_dump_data_type_map = {
    // key:ge datatype,value:proto datatype
    {ge::DT_UNDEFINED, ge::proto::DT_UNDEFINED},
    {ge::DT_FLOAT, ge::proto::DT_FLOAT},
    {ge::DT_FLOAT16, ge::proto::DT_FLOAT16},
    {ge::DT_INT8, ge::proto::DT_INT8},
    {ge::DT_UINT8, ge::proto::DT_UINT8},
    {ge::DT_INT16, ge::proto::DT_INT16},
    {ge::DT_UINT16, ge::proto::DT_UINT16},
    {ge::DT_INT32, ge::proto::DT_INT32},
    {ge::DT_INT64, ge::proto::DT_INT64},
    {ge::DT_UINT32, ge::proto::DT_UINT32},
    {ge::DT_UINT64, ge::proto::DT_UINT64},
    {ge::DT_BOOL, ge::proto::DT_BOOL},
    {ge::DT_DOUBLE, ge::proto::DT_DOUBLE},
    {ge::DT_DUAL, ge::proto::DT_DUAL},
    {ge::DT_DUAL_SUB_INT8, ge::proto::DT_DUAL_SUB_INT8},
    {ge::DT_DUAL_SUB_UINT8, ge::proto::DT_DUAL_SUB_UINT8},
    {ge::DT_COMPLEX32, ge::proto::DT_COMPLEX32},
    {ge::DT_COMPLEX64, ge::proto::DT_COMPLEX64},
    {ge::DT_COMPLEX128, ge::proto::DT_COMPLEX128},
    {ge::DT_QINT8, ge::proto::DT_QINT8},
    {ge::DT_QINT16, ge::proto::DT_QINT16},
    {ge::DT_QINT32, ge::proto::DT_QINT32},
    {ge::DT_QUINT8, ge::proto::DT_QUINT8},
    {ge::DT_QUINT16, ge::proto::DT_QUINT16},
    {ge::DT_RESOURCE, ge::proto::DT_RESOURCE},
    {ge::DT_STRING_REF, ge::proto::DT_STRING_REF},
    {ge::DT_STRING, ge::proto::DT_STRING},
    {ge::DT_VARIANT, ge::proto::DT_VARIANT},
    {ge::DT_BF16, ge::proto::DT_BF16},
    {ge::DT_INT4, ge::proto::DT_INT4},
    {ge::DT_UINT1, ge::proto::DT_UINT1},
    {ge::DT_INT2, ge::proto::DT_INT2},
    {ge::DT_UINT2, ge::proto::DT_UINT2},
    {ge::DT_HIFLOAT8, ge::proto::DT_HIFLOAT8},
    {ge::DT_FLOAT8_E5M2, ge::proto::DT_FLOAT8_E5M2},
    {ge::DT_FLOAT8_E4M3FN, ge::proto::DT_FLOAT8_E4M3FN},
    {ge::DT_FLOAT8_E8M0, ge::proto::DT_FLOAT8_E8M0},
    {ge::DT_FLOAT6_E3M2, ge::proto::DT_FLOAT6_E3M2},
    {ge::DT_FLOAT6_E2M3, ge::proto::DT_FLOAT6_E2M3},
    {ge::DT_FLOAT4_E2M1, ge::proto::DT_FLOAT4_E2M1},
    {ge::DT_FLOAT4_E1M2, ge::proto::DT_FLOAT4_E1M2},
};
}  // namespace

namespace ge {
int32_t DataTypeUtil::GetIrDataType(const ge::DataType data_type) {
  const auto const_iter = g_dump_data_type_map.find(data_type);
  if (const_iter == g_dump_data_type_map.end()) {
    return static_cast<int32_t>(ge::proto::DT_UNDEFINED);
  }

  return static_cast<int32_t>(const_iter->second);
}
}  // namespace ge
