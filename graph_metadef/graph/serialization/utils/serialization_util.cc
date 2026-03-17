/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "serialization_util.h"
#include <map>

namespace ge {
const std::map<DataType, ::ge::proto::DataType> kDataTypeMap = {
    {DT_UNDEFINED, proto::DT_UNDEFINED},
    {DT_FLOAT, proto::DT_FLOAT},
    {DT_FLOAT16, proto::DT_FLOAT16},
    {DT_INT8, proto::DT_INT8},
    {DT_UINT8, proto::DT_UINT8},
    {DT_INT16, proto::DT_INT16},
    {DT_UINT16, proto::DT_UINT16},
    {DT_INT32, proto::DT_INT32},
    {DT_INT64, proto::DT_INT64},
    {DT_UINT32, proto::DT_UINT32},
    {DT_UINT64, proto::DT_UINT64},
    {DT_BOOL, proto::DT_BOOL},
    {DT_DOUBLE, proto::DT_DOUBLE},
    {DT_DUAL, proto::DT_DUAL},
    {DT_DUAL_SUB_INT8, proto::DT_DUAL_SUB_INT8},
    {DT_DUAL_SUB_UINT8, proto::DT_DUAL_SUB_UINT8},
    {DT_COMPLEX32, proto::DT_COMPLEX32},
    {DT_COMPLEX64, proto::DT_COMPLEX64},
    {DT_COMPLEX128, proto::DT_COMPLEX128},
    {DT_QINT8, proto::DT_QINT8},
    {DT_QINT16, proto::DT_QINT16},
    {DT_QINT32, proto::DT_QINT32},
    {DT_QUINT8, proto::DT_QUINT8},
    {DT_QUINT16, proto::DT_QUINT16},
    {DT_RESOURCE, proto::DT_RESOURCE},
    {DT_STRING_REF, proto::DT_STRING_REF},
    {DT_STRING, proto::DT_STRING},
    {DT_VARIANT, proto::DT_VARIANT},
    {DT_BF16, proto::DT_BF16},
    {DT_INT4, proto::DT_INT4},
    {DT_UINT1, proto::DT_UINT1},
    {DT_INT2, proto::DT_INT2},
    {DT_UINT2, proto::DT_UINT2},
    {DT_HIFLOAT8, proto::DT_HIFLOAT8},
    {DT_FLOAT8_E5M2, proto::DT_FLOAT8_E5M2},
    {DT_FLOAT8_E4M3FN, proto::DT_FLOAT8_E4M3FN},
    {DT_FLOAT8_E8M0, proto::DT_FLOAT8_E8M0},
    {DT_FLOAT6_E3M2, proto::DT_FLOAT6_E3M2},
    {DT_FLOAT6_E2M3, proto::DT_FLOAT6_E2M3},
    {DT_FLOAT4_E2M1, proto::DT_FLOAT4_E2M1},
    {DT_FLOAT4_E1M2, proto::DT_FLOAT4_E1M2}
};

void SerializationUtil::GeDataTypeToProto(const ge::DataType ge_type, proto::DataType &proto_type) {
  auto iter = kDataTypeMap.find(ge_type);
  if (iter != kDataTypeMap.end()) {
    proto_type = iter->second;
    return;
  }
  proto_type = proto::DT_UNDEFINED;
}

void SerializationUtil::ProtoDataTypeToGe(const proto::DataType proto_type, ge::DataType &ge_type) {
  for (auto iter : kDataTypeMap) {
    if (iter.second == proto_type) {
      ge_type = iter.first;
      return;
    }
  }
  ge_type = DT_UNDEFINED;
}
}  // namespace ge
