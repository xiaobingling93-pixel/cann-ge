/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

graphStatus ProcessUnsqueezeAxes(const gert::SymbolShape *in_shape, gert::SymbolShape *out_shape,
                                 const std::vector<int64_t> &axes) {
  const auto in_dim_num = in_shape->GetDimNum();
  const auto out_dim_num = in_dim_num + axes.size();
  GE_ASSERT(out_dim_num <= gert::Shape::kMaxDimNum,
            "unsqueeze failed, DimNum of output shape is %zu, larger than kMaxDimNum is %zu!", out_dim_num,
            gert::Shape::kMaxDimNum);
  out_shape->Clear();
  for (size_t i = 0; i < out_dim_num; i++) {
    out_shape->AppendDim(ge::Symbol(0));
  }
  const auto out_dim_num_signed = static_cast<int64_t>(out_dim_num);
  for (size_t i = 0; i < axes.size(); i++) {
    const int64_t raw_axis = axes[i];
    const int64_t real_axis = raw_axis >= 0 ? raw_axis : raw_axis + out_dim_num_signed;
    // validate axis is in range of out shape dim num
    GE_ASSERT(real_axis >= 0 && real_axis < out_dim_num_signed,
              "unsqueeze failed, as axes val[%zu] is out of range[-%zu, %zu].", raw_axis, out_dim_num, out_dim_num);
    // validate axis is not repeated in out shape
    GE_ASSERT(out_shape->GetDim(real_axis) != 1, "unsqueeze failed, axis repeated");
    out_shape->MutableDim(real_axis) = ge::Symbol(1);
  }
  size_t in_index = 0;
  for (size_t i = 0; i < out_dim_num; i++) {
    if (out_shape->GetDim(i) != 1) {
      out_shape->MutableDim(i) = in_shape->GetDim(in_index++);
    }
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4UnSqueeze(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto axes = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(axes);
  // if axes is empty, out shape is same as input shape
  if (axes->GetSize() == 0) {
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
  }
  // get axes value
  std::vector<int64_t> axes_vec;
  axes_vec.reserve(axes->GetSize());
  for (size_t i = 0; i < axes->GetSize(); i++) {
    axes_vec.push_back(axes->GetData()[i]);
  }
  return ProcessUnsqueezeAxes(in_shape, out_shape, axes_vec);
}

graphStatus InferShape4UnsqueezeV3(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto axes_tensor = context->GetInputSymbolTensor(1);
  GE_ASSERT_NOTNULL(axes_tensor);
  const auto axes = axes_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(axes_tensor);
  if (axes->empty()) {
    GELOGD("UnsqueezeV3: axes is empty, out shape is same as input shape. node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
  }

  // get axes value
  std::vector<int64_t> axes_vec;
  axes_vec.reserve(axes->size());
  for (size_t i = 0; i < axes->size(); i++) {
    int64_t raw_axis;
    if (!axes->at(i).GetConstValue<int64_t>(raw_axis)) {
      return ge::UNSUPPORTED;
    }
    axes_vec.push_back(raw_axis);
  }
  return ProcessUnsqueezeAxes(in_shape, out_shape, axes_vec);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Unsqueeze).InferSymbolShape(InferShape4UnSqueeze);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(UnsqueezeV3).InferSymbolShape(InferShape4UnsqueezeV3);
}  // namespace
}  // namespace ge