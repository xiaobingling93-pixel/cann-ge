/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mul_kernel.h"

#include <memory>
#include <set>

#include "framework/common/debug/log.h"
#include "common/math/math_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/b_cast/b_cast.h"
#include "common/checker.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const std::set<DataType> kMulSupportedType = {DT_INT8,   DT_INT16,     DT_INT32,     DT_INT64,     DT_UINT8,
                                              DT_UINT16, DT_UINT32,    DT_UINT64,    DT_FLOAT16,   DT_FLOAT,
                                              DT_DOUBLE, DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128};
template <typename T>
Status OverflowCheckMul(T const &x, T const &y, DataType const &type) {
  switch (type) {
    case DT_INT8:
      FMK_INT8_MULCHECK(x, y)
      break;
    case DT_INT16:
      FMK_INT16_MULCHECK(x, y)
      break;
    case DT_INT32:
      FMK_INT32_MULCHECK(x, y)
      break;
    case DT_INT64:
      FMK_INT64_MULCHECK(x, y)
      break;
    case DT_UINT8:
      FMK_UINT8_MULCHECK(x, y)
      break;
    case DT_UINT16:
      FMK_UINT16_MULCHECK(x, y)
      break;
    case DT_UINT32:
      FMK_UINT32_MULCHECK(x, y)
      break;
    case DT_UINT64:
      FMK_UINT64_MULCHECK(x, y)
      break;
    case DT_FLOAT16:
    case DT_COMPLEX32:
      FMK_FP16_MULCHECK(x, y)
      break;
    case DT_FLOAT:
    case DT_COMPLEX64:
      FMK_FLOAT_MULCHECK(x, y)
      break;
    case DT_DOUBLE:
    case DT_COMPLEX128:
      FMK_DOUBLE_MULCHECK(x, y)
      break;
    default:
      break;
  }

  return SUCCESS;
}

template <typename T>
Status OverflowCheckAdd(T const &x, T const &y, DataType const &type) {
  switch (type) {
    case DT_COMPLEX32:
      FMK_FP16_ADDCHECK(x, y)
      break;
    case DT_COMPLEX64:
      FMK_FLOAT_ADDCHECK(x, y)
      break;
    case DT_COMPLEX128:
      FMK_DOUBLE_ADDCHECK(x, y)
      break;
    default:
      break;
  }
  return SUCCESS;
}

template <typename T>
Status OverflowCheckSub(T const &x, T const &y, DataType const &type) {
  switch (type) {
    case DT_COMPLEX32:
      FMK_FP16_SUBCHECK(x, y)
      break;
    case DT_COMPLEX64:
      FMK_FLOAT_SUBCHECK(x, y)
      break;
    case DT_COMPLEX128:
      FMK_DOUBLE_SUBCHECK(x, y)
      break;
    default:
      break;
  }
  return SUCCESS;
}

#define DEFINE_FUNC_WITH_STATUS_BY_TYPE(TYPE)                                         \
  std::function<TYPE(TYPE const &, TYPE const &, DataType &, Status &)> func_##TYPE = \
      [](TYPE const &a, TYPE const &b, DataType &type, Status &ret) -> TYPE {         \
    ret = OverflowCheckMul(a, b, type);                                               \
    if (ret != SUCCESS) {                                                             \
      GELOGE(PARAM_INVALID, "Result of mul is overflow.");                            \
      return static_cast<TYPE>(0);                                                    \
    }                                                                                 \
    return static_cast<TYPE>(a) * static_cast<TYPE>(b);                               \
  }

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)                              \
  case DTYPE:                                                            \
    ret = bcast.BCastComputeCheck(input, y_data_##TYPE##_, func_##TYPE); \
    break

#define SET_OUTPUT(DTYPE, TYPE)                                                                                        \
  case DTYPE:                                                                                                          \
    (void)output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE##_.data()), y_data_##TYPE##_.size() * length); \
    break
// [no need to check result]
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int8_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int16_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int32_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int64_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint8_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint16_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint32_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint64_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(fp16_t);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(float);
DEFINE_FUNC_WITH_STATUS_BY_TYPE(double);

template <typename InT>
Status ComplexCompute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                      std::vector<GeTensorPtr> &v_output) {
  // only broadcast shape
  BCast bcast;
  Status ret = bcast.GenerateBcastInfo(BCast::TransShapeToDimVec(input[0U]->GetTensorDesc()),
                                       BCast::TransShapeToDimVec(input[1U]->GetTensorDesc()));
  GE_ASSERT_SUCCESS(ret, "Greater broadcasting failed.");

  std::vector<int64_t> x_indexes;
  std::vector<int64_t> y_indexes;
  bcast.BCastIndexes(x_indexes, y_indexes);

  auto x1_data = reinterpret_cast<const InT *>(input[0U]->GetData().data());
  auto x2_data = reinterpret_cast<const InT *>(input[1U]->GetData().data());

  size_t data_num = x_indexes.size();
  constexpr int64_t kComplexWidth = 2;
  const size_t value_num = data_num * kComplexWidth;
  std::unique_ptr<InT[]> buf = ge::MakeUnique<InT[]>(value_num);
  GE_CHECK_NOTNULL(buf, "New sizeof(T) * data_num(%zu) memory failed", static_cast<size_t>(sizeof(InT) * value_num));
  DataType data_type = input[0U]->GetTensorDesc().GetDataType();
  GE_ASSERT_TRUE((value_num * sizeof(InT)) == input[0U]->GetData().size(),
                 "complex value size should be an integer multiple of 2.");
  for (size_t i = 0U; i < data_num; ++i) {
    auto x_index_real = x1_data[x_indexes[i] * kComplexWidth];
    auto x_index_imaginary = x1_data[(x_indexes[i] * kComplexWidth) + 1U];
    auto y_index_real = x2_data[y_indexes[i] * kComplexWidth];
    auto y_index_imaginary = x2_data[(y_indexes[i] * kComplexWidth) + 1U];
    if ((OverflowCheckMul<InT>(x_index_real, y_index_real, data_type) != SUCCESS) ||
        (OverflowCheckMul<InT>(x_index_real, y_index_imaginary, data_type) != SUCCESS) ||
        (OverflowCheckMul<InT>(x_index_imaginary, y_index_real, data_type) != SUCCESS) ||
        (OverflowCheckMul<InT>(x_index_imaginary, y_index_imaginary, data_type) != SUCCESS)) {
      GELOGE(PARAM_INVALID, "Result of mul is overflow.");
      return PARAM_INVALID;
    }
    auto xr_yr = x_index_real * y_index_real;
    auto xi_yi = x_index_imaginary * y_index_imaginary;
    auto xr_yi = x_index_real * y_index_imaginary;
    auto xi_yr = x_index_imaginary * y_index_real;
    if ((OverflowCheckSub<InT>(xr_yr, xi_yi, data_type) != SUCCESS) ||
        (OverflowCheckAdd<InT>(xr_yi, xi_yr, data_type) != SUCCESS)) {
      GELOGE(PARAM_INVALID, "Result of mul is overflow.");
      return PARAM_INVALID;
    }
    buf[i * kComplexWidth]= xr_yr - xi_yi;        // complex real part
    buf[(i * kComplexWidth) + 1UL] = xr_yi + xi_yr;  // complex imaginary part
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  GE_CHECK_NOTNULL(output_ptr, "Make shared failed");

  output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), value_num * sizeof(InT));
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast.GetOutputShape()));
  v_output.push_back(output_ptr);
  GELOGD("MulKernel success");

  return SUCCESS;
}
}  // namespace

Status MulKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                          std::vector<GeTensorPtr> &v_output) {
  GELOGD("MulKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = MulCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  DataType data_type = input[0]->GetTensorDesc().GetDataType();
  BCast bcast;
  switch (data_type) {
    case DT_COMPLEX32:
      return ComplexCompute<fp16_t>(op_desc_ptr, input, v_output);
    case DT_COMPLEX64:
      return ComplexCompute<float>(op_desc_ptr, input, v_output);
    case DT_COMPLEX128:
      return ComplexCompute<double>(op_desc_ptr, input, v_output);
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
    default:
      ret = NOT_CHANGED;
      break;
  }

  if (ret != SUCCESS) {
    GELOGW("BCastCompute fail, data_type: %s, ret: %s", TypeUtils::DataTypeToSerialString(data_type).c_str(),
           GET_ERRORNO_STR(ret).c_str());
    return NOT_CHANGED;
  }

  uint32_t length = 1;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGW("Can't GetDataTypeLength of data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast.GetOutputShape()));
  // only return GRAPH_SUCCESS here
  switch (data_type) {
    SET_OUTPUT(DT_INT8, int8_t);
    SET_OUTPUT(DT_INT16, int16_t);
    SET_OUTPUT(DT_INT32, int32_t);
    SET_OUTPUT(DT_INT64, int64_t);
    SET_OUTPUT(DT_UINT8, uint8_t);
    SET_OUTPUT(DT_UINT16, uint16_t);
    SET_OUTPUT(DT_UINT32, uint32_t);
    SET_OUTPUT(DT_UINT64, uint64_t);
    SET_OUTPUT(DT_FLOAT16, fp16_t);
    SET_OUTPUT(DT_FLOAT, float);
    SET_OUTPUT(DT_DOUBLE, double);
    default:
      break;
  }
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  v_output.push_back(output_ptr);
  GELOGD("MulKernel success");

  return SUCCESS;
}

Status MulKernel::MulCheck(const std::vector<ConstGeTensorPtr> &input) const {
  // check input number
  if (input.size() != static_cast<size_t>(MUL_INPUT_NUM)) {
    GELOGI("The number of input for Mul must be %u.", MUL_INPUT_NUM);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr input_x1 = input.at(0);
  ConstGeTensorPtr input_x2 = input.at(1);
  GE_CHECK_NOTNULL(input_x1);
  GE_CHECK_NOTNULL(input_x2);
  // check whether there is data in Tensor
  if (input_x1->GetData().size() == 0 || input_x2->GetData().size() == 0) {
    GELOGI("Check data size fail. x1: %zu, x2: %zu", input_x1->GetData().size(), input_x2->GetData().size());
    return NOT_CHANGED;
  }

  // check whether the data types are the same
  DataType type = input_x1->GetTensorDesc().GetDataType();
  if (type != input_x2->GetTensorDesc().GetDataType()) {
    GELOGI("Data type of inputs for Mul not matched.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  if (kMulSupportedType.find(type) == kMulSupportedType.end()) {
    GELOGI("Mul does not support this Data type: %s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(MUL, MulKernel);
}  // namespace ge
