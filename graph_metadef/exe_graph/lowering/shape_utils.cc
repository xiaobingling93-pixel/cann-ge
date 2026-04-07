/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/shape_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/math_util.h"
#include "graph/utils/tensor_utils.h"
#include "common/checker.h"

namespace gert {
const Shape g_vec_1_shape = {1};

ge::graphStatus CalcAlignedSizeByShape(const Shape &shape, ge::DataType data_type, uint64_t &ret_tensor_size) {
  constexpr uint64_t kAlignBytes = 32U;
  auto shape_size = shape.GetShapeSize();
  int64_t cal_size = 0;
  if (data_type == ge::DT_STRING) {
    uint32_t type_size = 0U;
    GE_ASSERT_TRUE(ge::TypeUtils::GetDataTypeLength(data_type, type_size));
    if (ge::MulOverflow(shape_size, static_cast<int64_t>(type_size), cal_size)) {
      GELOGE(ge::GRAPH_FAILED, "[Calc][TensorSizeByShape] shape_size[%ld] multiplied by type_size[%u] overflowed!",
             shape_size, type_size);
      return ge::GRAPH_FAILED;
    }
  } else {
    cal_size = ge::GetSizeInBytes(shape_size, data_type);
  }
  if (cal_size < 0) {
    GELOGE(ge::GRAPH_FAILED, "[Calc][TensorSizeByShape] shape_size[%" PRId64 "] data_type[%s] failed", shape_size,
           ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ge::GRAPH_FAILED;
  }

  const uint64_t padding_size = static_cast<uint64_t>(ge::TensorUtils::GetPaddingSize());
  // 不可能溢出，因为ret最大值也只有int64的最大值
  ret_tensor_size = ge::RoundUp(static_cast<uint64_t>(cal_size), kAlignBytes) + padding_size;
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert