/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_CONST_PLACE_HOLDER_UTILS_H_
#define GE_COMMON_CONST_PLACE_HOLDER_UTILS_H_

#include "framework/common/types.h"
#include "graph/op_desc.h"
#include "ge/ge_api_types.h"

namespace ge {
  ge::Status GetConstPlaceHolderAddr(const OpDescPtr &op_desc, uint8_t* &dev_address);
}  // namespace ge

#endif  // GE_COMMON_CONST_PLACE_HOLDER_UTILS_H_
