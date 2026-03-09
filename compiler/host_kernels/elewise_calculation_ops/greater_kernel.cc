/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_kernels/elewise_calculation_ops/greater_kernel.h"

#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/fp16_t/fp16_t.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/b_cast/b_cast.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kGreaterInputNum = 2;

#define DEFINE_FUNC_BY_TYPE(TYPE)                                                                                \
  std::function<uint8_t(TYPE const &, TYPE const &)> func_##TYPE = [](TYPE const &a, TYPE const &b) -> uint8_t { \
    return a > b;                                                                                                \
  }

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)               \
  case DTYPE:                                             \
    ret = bcast.BCastCompute(input, y_data, func_##TYPE); \
    break

DEFINE_FUNC_BY_TYPE(int8_t);
DEFINE_FUNC_BY_TYPE(int16_t);
DEFINE_FUNC_BY_TYPE(int32_t);
DEFINE_FUNC_BY_TYPE(int64_t);
DEFINE_FUNC_BY_TYPE(uint8_t);
DEFINE_FUNC_BY_TYPE(uint16_t);
DEFINE_FUNC_BY_TYPE(uint32_t);
DEFINE_FUNC_BY_TYPE(uint64_t);
DEFINE_FUNC_BY_TYPE(fp16_t);
DEFINE_FUNC_BY_TYPE(float);
DEFINE_FUNC_BY_TYPE(double);
DEFINE_FUNC_BY_TYPE(bool);
}  // namespace

Status GreaterKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                              std::vector<GeTensorPtr> &v_output) {
  GELOGD("GreaterKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = GreaterCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<uint8_t> y_data;
  GE_CHECK_NOTNULL(input[0]);
  DataType data_type = input[0]->GetTensorDesc().GetDataType();
  BCast bcast;
  switch (data_type) {
    SET_BCAST_COMPUTE_CASE(DT_INT8, int8_t);
    SET_BCAST_COMPUTE_CASE(DT_INT16, int16_t);
    SET_BCAST_COMPUTE_CASE(DT_INT32, int32_t);
    SET_BCAST_COMPUTE_CASE(DT_INT64, int64_t);
    SET_BCAST_COMPUTE_CASE(DT_UINT8, uint8_t);
    SET_BCAST_COMPUTE_CASE(DT_UINT16, uint16_t);
    SET_BCAST_COMPUTE_CASE(DT_UINT32, uint32_t);
    SET_BCAST_COMPUTE_CASE(DT_UINT64, uint64_t);
    SET_BCAST_COMPUTE_CASE(DT_FLOAT16, fp16_t);
    SET_BCAST_COMPUTE_CASE(DT_FLOAT, float);
    SET_BCAST_COMPUTE_CASE(DT_DOUBLE, double);
    SET_BCAST_COMPUTE_CASE(DT_BOOL, bool);
    default:
      ret = NOT_CHANGED;
      break;
  }

  if (ret != SUCCESS) {
    GELOGW("BCastCompute fail, data_type:%s, ret:%s", TypeUtils::DataTypeToSerialString(data_type).c_str(),
           GET_ERRORNO_STR(ret).c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed %s.", op_desc_ptr->GetName().c_str());
    return MEMALLOC_FAILED;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast.GetOutputShape()));
  // only return GRAPH_SUCCESS here
  GE_CHK_STATUS_RET(output_ptr->SetData(y_data));
  output_ptr->MutableTensorDesc().SetDataType(DT_BOOL);
  v_output.push_back(output_ptr);
  GELOGD("GreaterKernel success");

  return SUCCESS;
}

Status GreaterKernel::GreaterCheck(const std::vector<ConstGeTensorPtr> &input) {
  // check input number
  if (input.size() != kGreaterInputNum) {
    GELOGI("The number of input for greater must be %zu.", kGreaterInputNum);
    return NOT_CHANGED;
  }

  GE_CHECK_NOTNULL(input[0]);
  GE_CHECK_NOTNULL(input[1]);

  ConstGeTensorPtr input_x1 = input.at(0);
  ConstGeTensorPtr input_x2 = input.at(1);
  // check whether there is data in Tensor
  if (input_x1->GetData().size() == 0 || input_x2->GetData().size() == 0) {
    GELOGI("Check data size fail. x1: %zu, x2:%zu", input_x1->GetData().size(), input_x2->GetData().size());
    return NOT_CHANGED;
  }

  // check whether the data types are the same
  if (input_x1->GetTensorDesc().GetDataType() != input_x2->GetTensorDesc().GetDataType()) {
    GELOGI("Data type of inputs for greater not matched.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  DataType type = input_x1->GetTensorDesc().GetDataType();
  if (greater_supported_type.find(type) == greater_supported_type.end()) {
    GELOGI("Greater does not support this Data type:%s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(GREATER, GreaterKernel);
}  // namespace ge
