/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/const_place_holder_utils/const_place_holder_utils.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_context.h"
#include "common/math/math_util.h"
#include "formats/utils/formats_trans_utils.h"

namespace ge {
ge::Status GetConstPlaceHolderAddr(const OpDescPtr &op_desc, uint8_t* &dev_address) {
  vector<int64_t> storage_shape;
  GE_ASSERT_TRUE(ge::AttrUtils::GetListInt(op_desc, "storage_shape", storage_shape));
  DataType data_type = DT_UNDEFINED;
  GE_ASSERT_TRUE(ge::AttrUtils::GetDataType(op_desc, "dtype", data_type));
  int64_t data_length = 0L;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, "size", data_length));
  int64_t element_cnt = 1L;
  for (const int64_t dim : storage_shape) {
    FMK_INT64_MULCHECK(element_cnt, dim);
    element_cnt *= dim;
  }
  const int64_t size = ge::GetSizeInBytes(static_cast<int64_t>(element_cnt), data_type);
  GE_ASSERT_TRUE(data_length >= size, "GetConstPlaceHolder check data_lengths failed, attr size is [%zu], "
                 "size calculated by shape and data type is [%zu], shape is %s, data_type is %d.",
                 data_length, size, ge::formats::JoinToString(storage_shape).c_str(), data_type);

  int64_t placement = 0L;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, "placement", placement));
  GE_ASSERT_TRUE(placement == ge::Placement::kPlacementDevice, "GetConstPlaceHolderAttr placement expect %d, "
                 "but got %d.", ge::Placement::kPlacementDevice, placement);

  int64_t device_addr = 0L;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, "addr", device_addr));
  dev_address = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(device_addr));
  GE_ASSERT_TRUE(dev_address != nullptr, "GetConstPlaceHolderAttr dev_address ptr is null.");
  GELOGI("Get op %s addr, dev addr is %p.", op_desc->GetName().c_str(), dev_address);
  return SUCCESS;
}
}  // namespace ge
