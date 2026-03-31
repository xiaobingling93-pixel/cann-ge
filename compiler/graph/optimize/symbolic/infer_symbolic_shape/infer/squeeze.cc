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
graphStatus ProcessSqueezeAxes(const gert::SymbolShape *in_shape, gert::SymbolShape *out_shape,
                               const std::vector<int64_t> &axes) {
  if (axes.empty()) {
    // axes is empty, squeeze all dims that are 1
    out_shape->Clear();
    for (size_t i = 0; i < in_shape->GetDimNum(); i++) {
      if (EXPECT_SYMBOL_NE(in_shape->GetDim(i), Symbol(1))) {
        out_shape->AppendDim(in_shape->GetDim(i));
      }
    }
  } else {
    // axes is not empty, squeeze dims that are 1 by axes
    bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
    const auto dim_num = static_cast<int64_t>(in_shape->GetDimNum());
    for (size_t i = 0; i < axes.size(); i++) {
      const int64_t raw_axis = axes[i];
      const int64_t real_axis = raw_axis >= 0 ? raw_axis : raw_axis + dim_num;
      GE_ASSERT(real_axis >= 0 && real_axis < dim_num, "Squeeze failed, as axes val[%ld] is out of range[-%ld, %ld].",
                raw_axis, dim_num, dim_num);
      squeeze_dims[real_axis] = true;
    }
    out_shape->Clear();
    for (size_t i = 0; i < in_shape->GetDimNum(); i++) {
      if (!squeeze_dims[i]) {
        out_shape->AppendDim(in_shape->GetDim(i));
      } else {
        const auto dim = in_shape->GetDim(i);
        ASSERT_SYMBOL_EQ(dim, Symbol(1));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4Squeeze(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto axes = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(axes);

  std::vector<int64_t> axes_vec;
  for (size_t i = 0; i < axes->GetSize(); i++) {
    axes_vec.push_back(axes->GetData()[i]);
  }
  return ProcessSqueezeAxes(in_shape, out_shape, axes_vec);
}

graphStatus InferShape4SqueezeV3(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto axes_tensor = context->GetInputSymbolTensor(1);
  std::vector<int64_t> axes_vec;
  if (axes_tensor == nullptr) {
    GELOGD("SqueezeV3: axes is empty, squeeze all dims with value 1. node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
  } else {
    GE_UNSUPPORTED_IF_NULL(axes_tensor->GetSymbolicValue());
    const auto axes = axes_tensor->GetSymbolicValue();
    GELOGD("SqueezeV3: axes is not empty, squeeze dims with axes. axes size: %ld. node %s[%s]", axes->size(),
           context->GetNodeName(), context->GetNodeType());
    for (size_t i = 0; i < axes->size(); i++) {
      int64_t raw_axis;
      if (!(axes->at(i).GetConstValue<int64_t>(raw_axis))) {
        return ge::UNSUPPORTED;
      }
      axes_vec.push_back(raw_axis);
    }
  }
  return ProcessSqueezeAxes(in_shape, out_shape, axes_vec);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Squeeze).InferSymbolShape(InferShape4Squeeze);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SqueezeV3).InferSymbolShape(InferShape4SqueezeV3);
}  // namespace
}  // namespace ge