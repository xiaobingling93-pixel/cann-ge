/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
graphStatus InferShape4FlattenV2(gert::InferSymbolShapeContext *context) {
  auto const in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto const out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto attr_axis_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(attr_axis_ptr);
  const auto attr_end_axis_ptr = attrs->GetInt(1);
  GE_ASSERT_NOTNULL(attr_end_axis_ptr);

  const int64_t axis = *attr_axis_ptr;
  const int64_t end_axis = *attr_end_axis_ptr;
  GELOGD("FlattenV2: axis=%ld, end_axis=%ld. node %s[%s]", axis, end_axis, context->GetNodeName(), context->GetNodeType());

  const int64_t dim_num = static_cast<int64_t>(in_shape->GetDimNum());
  const int64_t real_axis = axis >= 0 ? axis : axis + dim_num;
  const int64_t real_end_axis = end_axis >= 0 ? end_axis : end_axis + dim_num;

  GE_ASSERT(real_axis >= 0 && real_axis < dim_num,
            "FlattenV2 failed, as axes val[%ld] is out of range[-%ld, %ld]. node %s[%s]", real_axis, dim_num, dim_num,
            context->GetNodeName(), context->GetNodeType());

  GE_ASSERT(real_axis <= real_end_axis,
            "FlattenV2 failed, as axes val[%ld] must be less than or equal to end_axes val[%ld]. node %s[%s]",
            real_axis, real_end_axis, context->GetNodeName(), context->GetNodeType());

  GE_ASSERT(real_end_axis >= 0 && real_end_axis < dim_num,
            "FlattenV2 failed, as end_axes val[%ld] is out of range[-%ld, %ld]. node %s[%s]", real_end_axis, dim_num,
            dim_num, context->GetNodeName(), context->GetNodeType());
  
  out_shape->Clear();
  auto product = Expression(Symbol(1));
  for (int64_t i = 0; i < dim_num; i++) {
    if (i >= real_axis && i <= real_end_axis) {
      product = product * in_shape->GetDim(i);
      if (i == real_end_axis) {
        out_shape->AppendDim(product);
      }
    } else {
      out_shape->AppendDim(in_shape->GetDim(i));
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FlattenV2).InferSymbolShape(InferShape4FlattenV2);
}  // namespace
}  // namespace ge