/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_kernels/kernel_utils.h"

#include <vector>
#include <memory>
#include <climits>

#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"

namespace {
const uint32_t kDimensionShapeIndex = 0U;
const size_t kDimensionDimsIndex = 1U;
const size_t kDimensionNodeInputSize = 2UL;
}  // namespace

namespace ge {
Status KernelUtils::ConstructTensorDescWithData(const GeTensorDesc &out_desc, const std::vector<int64_t> &data,
                                                std::vector<GeTensorPtr> &v_output, const bool scalar_output) {
  Status ret = SUCCESS;
  const size_t dim_size = data.size();
  const DataType data_type = out_desc.GetDataType();
  if (data_type == DT_INT32) {
    std::vector<int32_t> buf(dim_size);
    for (size_t i = 0U; i < dim_size; i++) {
      if (data[i] >= INT_MAX) {
        REPORT_INNER_ERR_MSG("E19999", "Param data:%s will overflow after multi", ToString(data).c_str());
        GELOGE(PARAM_INVALID, "[Check][Param] int32 overflow, data[%zu]:%ld", i, data[i]);
        return PARAM_INVALID;
      }
      buf[i] = static_cast<int32_t>(data[i]);
    }
    ret = ConstructTensorDescWithData(out_desc, buf.data(), dim_size, v_output, scalar_output);
  } else if (data_type == DT_INT64) {
    std::vector<int64_t> buf(dim_size);
    for (size_t i = 0U; i < dim_size; i++) {
      buf[i] = data[i];
    }
    ret = ConstructTensorDescWithData(out_desc, buf.data(), dim_size, v_output, scalar_output);
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Only support DT_INT32 and DT_INT64. Input data_type:%s not support",
                      ToString(data).c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Only support DT_INT32 and DT_INT64. data_type:%s not support",
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return PARAM_INVALID;
  }

  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][ShapeTensor] failed, ret:%u.", ret);
    return ret;
  }

  return SUCCESS;
}

template <typename T>
Status KernelUtils::ConstructTensorDescWithData(const GeTensorDesc &out_desc, const T *const buf, const size_t len,
                                                std::vector<GeTensorPtr> &v_output, const bool scalar_output) {
  // construct TensorDesc
  const GeShape out_shape = (scalar_output ? GeShape() : GeShape({static_cast<int64_t>(len)}));
  GeTensorDesc output_tensor_desc(out_desc);
  output_tensor_desc.SetShape(out_shape);
  output_tensor_desc.SetOriginShape(out_shape);

  const GeTensorPtr output_tensor_ptr = MakeShared<GeTensor>(
      output_tensor_desc, reinterpret_cast<const uint8_t *const>(buf), sizeof(T) * len);
  if (output_tensor_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New GeTensor failed");
    GELOGE(MEMALLOC_FAILED, "[New][GeTensor] failed");
    return MEMALLOC_FAILED;
  }

  v_output.push_back(output_tensor_ptr);
  return SUCCESS;
}

Status KernelUtils::CheckDimensionNodeInfo(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  auto input_nodes = node_ptr->GetInDataNodes();
  if (input_nodes.size() != kDimensionNodeInputSize) {
    GELOGW("op:%s type: %s, dimension input size must be %zu, but get %zu inputs", node_ptr->GetName().c_str(),
           node_ptr->GetType().c_str(), kDimensionNodeInputSize, input_nodes.size());
    return NOT_CHANGED;
  }

  const NodePtr dim_node = input_nodes.at(kDimensionDimsIndex);
  if (dim_node == nullptr) {
    GELOGE(PARAM_INVALID, "dim node is nullptr");
    return PARAM_INVALID;
  }

  std::vector<ConstGeTensorPtr> const_ge_tensor;
  if ((dim_node->GetType() == CONSTANT) || (dim_node->GetType() == CONSTANTOP)) {
    const_ge_tensor = OpDescUtils::GetWeights(dim_node);
  } else if (dim_node->GetType() == DATA) {
    auto parent_node_anchor = NodeUtils::GetParentInputAndAnchor(dim_node);
    auto parent_node = parent_node_anchor.first;
    while ((parent_node != nullptr) && (parent_node->GetType() == DATA)) {
      parent_node_anchor = NodeUtils::GetParentInputAndAnchor(parent_node);
      parent_node = parent_node_anchor.first;
    }

    if ((parent_node != nullptr) && ((parent_node->GetType() == CONSTANT) || (parent_node->GetType() == CONSTANTOP))) {
      GELOGD("Get parent const node[%s].", parent_node->GetName().c_str());
      const_ge_tensor = OpDescUtils::GetWeights(parent_node);
    }
  } else {
    // do nothing
  }
  if (const_ge_tensor.empty()) {
    GELOGE(PARAM_INVALID, "dim node must be const op");
    return PARAM_INVALID;
  }
  const ConstGeTensorPtr &input_dim = const_ge_tensor.at(0U);
  if (input_dim->GetData().size() == 0U) {
    GELOGE(PARAM_INVALID, "dim data size is 0");
    return PARAM_INVALID;
  }

  return SUCCESS;
}

bool KernelUtils::CheckFormatSupported(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return false;
  }
  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "op_desc is null");
    return false;
  }
  const auto &input_desc = op_desc->MutableInputDesc(kDimensionShapeIndex);
  GE_CHECK_NOTNULL_EXEC(input_desc, return false);
  const Format fmt = input_desc->GetFormat();
  if ((fmt == FORMAT_NC1HWC0) || (fmt == FORMAT_FRACTAL_Z)) {
    GELOGW("invalid format, fmt: %s", TypeUtils::FormatToSerialString(fmt).c_str());
    return false;
  }

  return true;
}

bool KernelUtils::CheckSizeForTransOp(const ge::ConstGeTensorPtr &const_weight_ptr,
                                      const ge::OpDescPtr &op_desc_ptr) {
  if ((const_weight_ptr == nullptr) || (op_desc_ptr == nullptr)) {
    GELOGE(FAILED, "parameter invalid");
    return false;
  }
  const auto data_size = const_weight_ptr->GetData().GetSize();
  const auto &input_desc = op_desc_ptr->MutableInputDesc(0U);
  GE_CHECK_NOTNULL_EXEC(input_desc, return false);
  const DataType data_type = input_desc->GetDataType();
  const GeShape data_shape = input_desc->GetShape();
  const Format data_format = input_desc->GetFormat();
  const auto shape_size = input_desc->GetShape().GetShapeSize();
  int64_t cal_size = 0;

  const auto ret = TensorUtils::CalcTensorMemSize(data_shape, data_format, data_type, cal_size);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "CalcTensorMemSize failed");
    return false;
  }

  uint32_t length = 1U;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGE(PARAM_INVALID, "Input datatype %d is not supported.", data_type);
    return false;
  }

  GELOGI("Const real value Size:%zu, op_desc Shape Size:%ld, data_type:%s.", data_size, cal_size,
         TypeUtils::DataTypeToSerialString(data_type).c_str());
  if (shape_size != 0) {
    // Standard tensor
    if ((data_size != static_cast<size_t>(cal_size)) || (data_size == 0U)) {
      GELOGW("Const input data size is not equal with tensor desc shape");
      return false;
    }
  } else if (data_shape.GetDimNum() != 0U) {
    // Empty tensor, has zero in shape vector
    if (data_size != 0U) {
      GELOGW("Const input data size is not equal with tensor desc shape");
      return false;
    }
  } else {
    // Scalar tensor, has only one element in tensor
    if ((length != 0U) && ((data_size / static_cast<size_t>(length)) != 1U)) {
      GELOGW("Const input data size is not equal with tensor desc shape");
      return false;
    }
  }

  return true;
}

bool KernelUtils::IsUnknownShape(const ge::GeShape &shape) {
  const std::vector<int64_t> dims = shape.GetDims();
  for (const auto dim : dims) {
    if (dim < 0) {
      GELOGW("Shape kernel recognize unknown shape. Ignore shape kernel.");
      return true;
    }
  }
  return false;
}
}  // namespace ge
