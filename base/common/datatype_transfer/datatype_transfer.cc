/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/datatype_transfer/datatype_transfer.h"
#include "base/err_msg.h"
#include <cstdint>
#include <map>
#include <utility>

#include "formats/utils/formats_trans_utils.h"
#include "common/fp16_t/fp16_t.h"
#include "common/math/hif8_t.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"
#include "securec.h"

namespace ge {
namespace formats {
namespace {
enum class DataTypeTransMode : uint16_t {
  kTransferWithDatatypeFloatToFloat16,
  kTransferWithDatatypeFloatToInt32,
  kTransferWithDatatypeFloat16ToFloat,
  kTransferWithDatatypeFloat16ToInt32,
  kTransferWithDatatypeInt32ToFloat,
  kTransferWithDatatypeInt32ToFloat16,
  kTransferWithDatatypeInt32ToUint8,
  kTransferWithDatatypeInt32ToInt8,
  kTransferWithDatatypeUint8ToFloat,
  kTransferWithDatatypeUint8ToInt32,
  kTransferWithDatatypeInt8ToFloat,
  kTransferWithDatatypeInt8ToInt32,
  kTransferWithDatatypeInt64ToInt32,
  kTransferWithDatatypeInt32ToInt64,
  kTransferWithDatatypeInt32ToDouble,
  kTransferWithDatatypeDoubleToInt32,
  kTransferWithDatatypeFloatToHiF8,
  kTransferWithDatatypeFloat16ToHiF8,
  kTransferWithDatatypeHiF8ToFloat,
  kTransferWithDatatypeHiF8ToFloat16,
};

std::map<std::pair<DataType, DataType>, DataTypeTransMode> trans_mode_map {
  {std::pair<DataType, DataType>(DT_FLOAT, DT_FLOAT16), DataTypeTransMode::kTransferWithDatatypeFloatToFloat16},
  {std::pair<DataType, DataType>(DT_FLOAT, DT_INT32), DataTypeTransMode::kTransferWithDatatypeFloatToInt32},
  {std::pair<DataType, DataType>(DT_FLOAT16, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeFloat16ToFloat},
  {std::pair<DataType, DataType>(DT_FLOAT16, DT_INT32), DataTypeTransMode::kTransferWithDatatypeFloat16ToInt32},
  {std::pair<DataType, DataType>(DT_INT32, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeInt32ToFloat},
  {std::pair<DataType, DataType>(DT_INT32, DT_FLOAT16), DataTypeTransMode::kTransferWithDatatypeInt32ToFloat16},
  {std::pair<DataType, DataType>(DT_INT32, DT_UINT8), DataTypeTransMode::kTransferWithDatatypeInt32ToUint8},
  {std::pair<DataType, DataType>(DT_INT32, DT_INT8), DataTypeTransMode::kTransferWithDatatypeInt32ToInt8},
  {std::pair<DataType, DataType>(DT_UINT8, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeUint8ToFloat},
  {std::pair<DataType, DataType>(DT_UINT8, DT_INT32), DataTypeTransMode::kTransferWithDatatypeUint8ToInt32},
  {std::pair<DataType, DataType>(DT_INT8, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeInt8ToFloat},
  {std::pair<DataType, DataType>(DT_INT8, DT_INT32), DataTypeTransMode::kTransferWithDatatypeInt8ToInt32},
  {std::pair<DataType, DataType>(DT_INT64, DT_INT32), DataTypeTransMode::kTransferWithDatatypeInt64ToInt32},
  {std::pair<DataType, DataType>(DT_INT32, DT_INT64), DataTypeTransMode::kTransferWithDatatypeInt32ToInt64},
  {std::pair<DataType, DataType>(DT_INT32, DT_DOUBLE), DataTypeTransMode::kTransferWithDatatypeInt32ToDouble},
  {std::pair<DataType, DataType>(DT_DOUBLE, DT_INT32), DataTypeTransMode::kTransferWithDatatypeDoubleToInt32},
  {std::pair<DataType, DataType>(DT_FLOAT, DT_HIFLOAT8), DataTypeTransMode::kTransferWithDatatypeFloatToHiF8},
  {std::pair<DataType, DataType>(DT_FLOAT16, DT_HIFLOAT8), DataTypeTransMode::kTransferWithDatatypeFloat16ToHiF8},
  {std::pair<DataType, DataType>(DT_HIFLOAT8, DT_FLOAT), DataTypeTransMode::kTransferWithDatatypeHiF8ToFloat},
  {std::pair<DataType, DataType>(DT_HIFLOAT8, DT_FLOAT16), DataTypeTransMode::kTransferWithDatatypeHiF8ToFloat16},
};

template <typename SrcT, typename DstT>
Status TransDataSrc2Dst(const CastArgs &args, uint8_t * const dst_raw, const size_t data_size) {
  auto *src = reinterpret_cast<const SrcT *>(args.data);
  auto *dst = reinterpret_cast<DstT *>(dst_raw);
  for (size_t idx = 0U; idx != data_size; idx++) {
    dst[idx] = static_cast<DstT>(src[idx]);
  }
  return SUCCESS;
}

template <typename SrcT>
Status TransDataSrc2Fp16(const CastArgs &args, uint8_t *const dst, const size_t data_size) {
  fp16_t src_data;
  for (size_t idx = 0U; idx != data_size; idx++) {
    src_data = reinterpret_cast<const SrcT *>(args.data)[idx];
    reinterpret_cast<uint16_t *>(dst)[idx] = src_data.val;
  }
  return SUCCESS;
}

Status CastKernel(const CastArgs &args, uint8_t *const dst, const size_t data_size,
                  const DataTypeTransMode trans_mode) {
  static std::map<DataTypeTransMode, std::function<Status(const CastArgs &, uint8_t *, const size_t)>>
  transfer_handle = {
      {DataTypeTransMode::kTransferWithDatatypeFloatToFloat16, &TransDataSrc2Fp16<float>},
      {DataTypeTransMode::kTransferWithDatatypeFloatToInt32, &TransDataSrc2Dst<float, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeFloat16ToFloat, &TransDataSrc2Dst<fp16_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeFloat16ToInt32, &TransDataSrc2Dst<fp16_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToFloat, &TransDataSrc2Dst<int32_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToFloat16, &TransDataSrc2Fp16<int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToUint8, &TransDataSrc2Dst<int32_t, uint8_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToInt8, &TransDataSrc2Dst<int32_t, int8_t>},
      {DataTypeTransMode::kTransferWithDatatypeUint8ToFloat, &TransDataSrc2Dst<uint8_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeUint8ToInt32, &TransDataSrc2Dst<uint8_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt8ToFloat, &TransDataSrc2Dst<int8_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeInt8ToInt32, &TransDataSrc2Dst<int8_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt64ToInt32, &TransDataSrc2Dst<int64_t, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToInt64, &TransDataSrc2Dst<int32_t, int64_t>},
      {DataTypeTransMode::kTransferWithDatatypeInt32ToDouble, &TransDataSrc2Dst<int32_t, double>},
      {DataTypeTransMode::kTransferWithDatatypeDoubleToInt32, &TransDataSrc2Dst<double, int32_t>},
      {DataTypeTransMode::kTransferWithDatatypeFloatToHiF8, &TransDataSrc2Dst<float, hif8_t>},
      {DataTypeTransMode::kTransferWithDatatypeFloat16ToHiF8, &TransDataSrc2Dst<fp16_t, hif8_t>},
      {DataTypeTransMode::kTransferWithDatatypeHiF8ToFloat, &TransDataSrc2Dst<hif8_t, float>},
      {DataTypeTransMode::kTransferWithDatatypeHiF8ToFloat16, &TransDataSrc2Dst<hif8_t, fp16_t>},
  };
  const auto it = transfer_handle.find(trans_mode);
  if (it == transfer_handle.end()) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  } else {
    return (it->second)(args, dst, data_size);
  }
}
}  // namespace

Status DataTypeTransfer::TransDataType(const CastArgs &args, TransResult &result) {
  GELOGD("Begin trans data from %s to %s, data size %zu", TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
         TypeUtils::DataTypeToSerialString(args.dst_data_type).c_str(), args.src_data_size);
  const std::pair<DataType, DataType> trans_info(args.src_data_type, args.dst_data_type);
  const auto iter = trans_mode_map.find(trans_info);
  if (iter == trans_mode_map.end()) {
    const std::string error = "Failed to trans data from datatype " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.src_data_type)) + " to " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type)) + " , it is not supported.";
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_DATATYPE_INVALID, error.c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  const auto trans_mode = iter->second;

  const int32_t size = GetSizeByDataType(args.dst_data_type);
  if (size <= 0) {
    const std::string error = "Failed to calc size from data type" +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type)) + ", it is not supported.";
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_DATATYPE_INVALID, error.c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (args.src_data_size > (SIZE_MAX / static_cast<size_t>(size))) {
    const std::string error = "args.src_data_size" + FmtToStr(args.src_data_size) +
        " or data type size" + FmtToStr(size) + " is too big";
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_PARAM_INVALID, error.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  const size_t total_size = args.src_data_size * static_cast<size_t>(size);
  result.length = total_size;
  if (total_size == 0U) {
    GELOGI("In TransDataType, total_size is zero, has no data.");
    return SUCCESS;
  }

  const std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Allocate][DSTMemory]Failed, memory for dst buf %zu, data size %zu",
           total_size, args.src_data_size);
    REPORT_INNER_ERR_MSG("E19999", "Failed to allocate memory for dst buf %zu, data size %zu",
                      total_size, args.src_data_size);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  if (CastKernel(args, dst.get(), args.src_data_size, trans_mode) != SUCCESS) {
    const std::string error = "Failed to cast data from datatype " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.src_data_type)) + " to " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type)) + ", data size is " +
        FmtToStr(std::to_string(args.src_data_size));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_INTERNAL_ERROR, error.c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  result.data = dst;
  return SUCCESS;
}

bool DataTypeTransfer::DataTypeTransferExists(const CastArgs &args) {
  const std::pair<DataType, DataType> trans_info(args.src_data_type, args.dst_data_type);
  const auto iter = trans_mode_map.find(trans_info);
  return iter != trans_mode_map.end();
}

bool IsTransDataTypeSupport(const CastArgs &args) {
  return DataTypeTransfer::DataTypeTransferExists(args);
}

Status TransTensorDataType(const CastArgs &args, TransResult &result) {
  if (!DataTypeTransfer::DataTypeTransferExists(args)) {
    const std::string error = "Failed to trans data from datatype " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.src_data_type)) + " to " +
        FmtToStr(TypeUtils::DataTypeToSerialString(args.dst_data_type));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_DATATYPE_INVALID, error.c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  if ((args.data == nullptr) && (args.src_data_size != 0UL)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param]Failed, input data is null "
           "or data size not equal to 0, src_data_size %ld", args.src_data_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  return DataTypeTransfer::TransDataType(args, result);
}
}  // namespace formats
}  // namespace ge
