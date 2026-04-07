/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/infer/node_shape_infer.h"
#include "graph/utils/tensor_utils.h"
#include "common/math/math_util.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace hybrid {

namespace {
const int32_t kAlignment = 32;

Status CalcTensorSize(const DataType data_type, const std::vector<int64_t> &shape,
                      int64_t &tensor_size) {
  GELOGD("To calc tensor size by shape = [%s]",
         GeShape(shape).ToString().c_str());
  uint32_t type_size;
  if (!TypeUtils::GetDataTypeLength(data_type, type_size)) {
    GELOGE(INTERNAL_ERROR, "[Get][DataTypeLength] failed for type:%s.",
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    REPORT_INNER_ERR_MSG("E19999", "GetDataTypeLength failed for type:%s.",
                      TypeUtils::DataTypeToSerialString(data_type).c_str());
    return INTERNAL_ERROR;
  }

  tensor_size = static_cast<int64_t>(type_size);
  for (const auto &dim : shape) {
    GE_CHECK_GE(dim, 0);
    GE_CHK_STATUS_RET(CheckInt64MulOverflow(tensor_size, dim),
                      "[Check][Overflow] Shape size overflow, shape = [%s]",
                      GeShape(shape).ToString().c_str());
    tensor_size *= dim;
  }

  const int64_t padding_size = TensorUtils::GetPaddingSize();
  const int64_t append_size = kAlignment + padding_size;
  GE_CHK_STATUS_RET(
      CheckInt64AddOverflow(tensor_size, append_size - 1),
      "[Check][Overflow]Tensor size is too large:%" PRId64 ", shape = [%s]"
      "Shape size will overflow when add align.",
      tensor_size, GeShape(shape).ToString().c_str());
  tensor_size = (tensor_size + append_size - 1) / kAlignment * kAlignment;
  return SUCCESS;
}

Status CanonicalizeShape(GeTensorDesc &tensor_desc, std::vector<int64_t> &shape,
                         const bool fallback_with_range) {
  const auto &tensor_shape = tensor_desc.MutableShape();
  if (tensor_shape.IsUnknownShape()) {
    if (!fallback_with_range) {
      GELOGE(INTERNAL_ERROR,
             "[Is][UnknownShape] Output shape is still unknown after shape "
             "inference. shape = [%s].",
             tensor_shape.ToString().c_str());
      REPORT_INNER_ERR_MSG(
          "E19999",
          "Output shape is still unknown after shape inference. shape = [%s].",
          tensor_shape.ToString().c_str());
      return INTERNAL_ERROR;
    }

    GELOGD("Calc output size by range");
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    GE_CHK_GRAPH_STATUS_RET(tensor_desc.GetShapeRange(shape_range),
                            "Failed to get shape range");
    if (shape_range.size() != shape.size()) {
      GELOGE(INTERNAL_ERROR,
             "[Check][Size] Number of shape ranges (%zu) mismatches that of "
             "dims (%zu).",
             shape_range.size(), shape.size());
      REPORT_INNER_ERR_MSG(
          "E19999",
          "Number of shape ranges (%zu) mismatches that of dims (%zu)",
          shape_range.size(), shape.size());
      return INTERNAL_ERROR;
    }

    for (size_t dim_index = 0U; dim_index < shape.size(); ++dim_index) {
      if (shape[dim_index] == ge::UNKNOWN_DIM) {
        shape[dim_index] = shape_range[dim_index].second;
      }
    }

    GELOGD("After canonicalization, shape = [%s], before = [%s]",
           GeShape(shape).ToString().c_str(), tensor_shape.ToString().c_str());
  }

  return SUCCESS;
}
} // namespace

Status NodeShapeInfer::CalcOutputTensorSizes(const bool fallback_with_range) const {
  for (size_t output_index = 0U; output_index < op_desc->GetOutputsSize(); ++output_index) {
    const auto &tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(output_index));
    GE_CHECK_NOTNULL(tensor_desc);
    const auto &shape = tensor_desc->MutableShape();
    // modify on copy
    auto dims = shape.GetDims();
    auto status_result = CanonicalizeShape(*tensor_desc, dims, fallback_with_range);
    if (status_result != SUCCESS) {
      GELOGE(ge::FAILED, "[Canonicalize][Shape] failed for [%s(%s)], output %zu.",
             NodeName().c_str(), NodeType().c_str(), output_index);
      return status_result;
    }
    int64_t tensor_size;
    status_result = CalcTensorSize(tensor_desc->GetDataType(), dims, tensor_size);
    if (status_result != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Invoke CalcTensorSize failed, node:%s(%s), output:%zu.",
                        NodeName().c_str(), NodeType().c_str(), output_index);
      GELOGE(ge::FAILED, "[Calc][TensorSize] failed for [%s(%s)], output %zu.",
             NodeName().c_str(), NodeType().c_str(), output_index);
      return status_result;
    }
    GELOGD("[%s] Tensor size of output %zu = %ld", NodeName().c_str(), output_index, tensor_size);
    (void)TensorUtils::SetSize(*tensor_desc, tensor_size);
  }
  return SUCCESS;
}

Status NodeShapeInfer::OnNodeDone() const {
  if ((shape_inference_type == DEPEND_SHAPE_RANGE) || (shape_inference_type == DEPEND_COMPUTE)) {
    GE_CHK_STATUS_RET_NOLOG(CalcOutputTensorSizes());
    GE_CHK_STATUS_RET_NOLOG(const_cast<NodeShapeInfer*>(this) ->DoPropagate());
  }
  return SUCCESS;
}

bool NodeShapeInfer::IsInputShapeStatic(const int32_t index) const {
  if (!is_dynamic) {
    return true;
  }

  if (static_cast<size_t>(index) >= is_input_shape_static_.size()) {
    GELOGE(PARAM_INVALID, "[Check][Param:index]Input index(%d) out of range: [0, %zu)",
           index, is_input_shape_static_.size());
    REPORT_INNER_ERR_MSG("E19999", "Input index(%d) out of range: [0, %zu).", index, is_input_shape_static_.size());
    return false;
  }

  return is_input_shape_static_[static_cast<size_t>(index)];
}
} // namespace hybrid
} // namespace ge
