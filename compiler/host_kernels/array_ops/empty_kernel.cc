/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "empty_kernel.h"

#include <memory>

#include "common/fp16_t/fp16_t.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "host_kernels/kernel_utils.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kEmptyFirstInput = 0;
const size_t kEmptyFirstOutput = 0;
const size_t kEmptyInputsSize = 1;
const size_t kEmptyOutputsSize = 1;
const size_t kShapeMaxDims = 1;
}  // namespace
Status EmptyKernel::EmptyCheck(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) const {
  if (op_desc_ptr == nullptr) {
    GELOGW("Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  // check input size
  bool size_check =
      ((op_desc_ptr->GetAllInputsSize() != kEmptyInputsSize) || (input.size() != kEmptyInputsSize) ||
       (op_desc_ptr->GetAllOutputsDescSize() != kEmptyOutputsSize));
  if (size_check) {
    GELOGW("Input/Output size error. InDesc size:%zu, OutDesc size:%zu, in size:%zu ",
           op_desc_ptr->GetAllInputsSize(), op_desc_ptr->GetAllOutputsDescSize(), input.size());
    return PARAM_INVALID;
  }

  if (input.at(kEmptyFirstInput) == nullptr) {
    GELOGW("Parameter's invalid, first input is nullptr.");
    return PARAM_INVALID;
  }
  ConstGeTensorPtr shape = input.at(kEmptyFirstInput);
  // Check if the dimension is 1-D
  if (shape->GetTensorDesc().GetShape().GetDimNum() > kShapeMaxDims) {
    GELOGW("Check if the dimension is 1-D failed, dims:%zu",
           shape->GetTensorDesc().GetShape().GetDimNum());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status EmptyKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                            std::vector<GeTensorPtr> &v_output) {
  GELOGD("Empty kernel in.");
  Status ret = EmptyCheck(op_desc_ptr, input);
  if (ret != SUCCESS) {
    return NOT_CHANGED;
  }

  ConstGeTensorPtr shape = input.at(kEmptyFirstInput);
  GE_CHECK_NOTNULL(shape);
  int64_t total_data_size = 1;
  std::vector<int64_t> shape_vec;
  DataType shape_type = shape->GetTensorDesc().GetDataType();
  // Calculate user input dim
  if (shape_type == DT_INT32) {
    ret = KernelUtils::CalcDims<int32_t>(shape, shape_vec, total_data_size);
  } else if (shape_type == DT_INT64) {
    ret = KernelUtils::CalcDims<int64_t>(shape, shape_vec, total_data_size);
  } else {
    GELOGW("shape type must be DT_INT32 or DT_INT64.");
    return NOT_CHANGED;
  }

  if (ret != SUCCESS) {
    GELOGE(ret, "CalcDims failed, dim_type: %s", TypeUtils::DataTypeToSerialString(shape_type).c_str());
    return ret;
  }

  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(kEmptyFirstOutput);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "make_shared ge::GeTensor failed");
    return MEMALLOC_FAILED;
  }

  DataType data_type = op_desc_ptr->GetOutputDesc(kEmptyFirstOutput).GetDataType();
  ret = PARAM_INVALID;
  uint64_t data = 0;
  switch (data_type) {
#define CASE(dtype, type)                                                \
  case dtype:                                                            \
    ret = KernelUtils::GenData(total_data_size, static_cast<type>(data), output_ptr); \
    break
    CASE(DT_FLOAT, float);
    CASE(DT_FLOAT16, ge::fp16_t);
    CASE(DT_INT8, int8_t);
    CASE(DT_INT16, int16_t);
    CASE(DT_UINT16, uint16_t);
    CASE(DT_UINT8, uint8_t);
    CASE(DT_INT32, int32_t);
    CASE(DT_INT64, int64_t);
    CASE(DT_UINT32, uint32_t);
    CASE(DT_UINT64, uint64_t);
    CASE(DT_BOOL, bool);
    CASE(DT_DOUBLE, double);
#undef CASE
    default:
      GELOGW("invalid data type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return NOT_CHANGED;
  }

  if (ret != SUCCESS) {
    GELOGE(ret, "GenData failed, data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ret;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(shape_vec));
  output_ptr->MutableTensorDesc().SetDataType(DataType(data_type));
  Format format = op_desc_ptr->GetOutputDesc(kEmptyFirstOutput).GetFormat();
  output_ptr->MutableTensorDesc().SetFormat(format);
  v_output.push_back(output_ptr);
  GELOGD("Empty kernel success");
  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(EMPTY, EmptyKernel);
}  // namespace ge
