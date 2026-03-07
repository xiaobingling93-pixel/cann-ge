/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_
#define GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_

#include "formats/register_format_transfer.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace formats {
class DataTypeTransfer {
 public:
  static bool DataTypeTransferExists(const CastArgs &args);
  static Status TransDataType(const CastArgs &args, TransResult &result);
};

bool IsTransDataTypeSupport(const CastArgs &args);

Status TransTensorDataType(const CastArgs &args, TransResult &result);
}  // namespace formats
}  // namespace ge

#endif  // GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_
