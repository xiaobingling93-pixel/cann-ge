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
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
/**
 * GatherShapes算子的符号化shape推导
 * 【算子功能】收集指定的输入dim, 作为输出
 * 【算子约束】
 *      1. 至少有一个输入
 *      2. 属性axes存放要输出的shape的dim值，格式为[input_index, dim_index]，input_index要在输入个数范围内，
 *         dim_index要在输入dim_num范围内
 *      举例： 输入0=(3)，输入1=(1,2)，属性axes={{0, 0},{1, 1}}，输出=(3,2)
 * 【推导逻辑】
 *      1. 按照算子约束校验输入和axes属性
 *      2. 属性axes的size作为输出shape
 */
graphStatus InferShape4GatherShapes(gert::InferSymbolShapeContext* context) {
  auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  auto input_num = context->GetComputeNodeInputNum();
  GE_ASSERT(input_num != 0U);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto* attr_axes = attrs->GetListListInt(0);
  GE_ASSERT_NOTNULL(attr_axes);
  auto axes_size = attr_axes->GetSize();
  GE_ASSERT(axes_size != 0U);

  for (size_t i = 0U; i < axes_size; ++i) {
    const auto cv = attr_axes->Get(i);
    GE_ASSERT_NOTNULL(cv);
    GE_ASSERT_EQ(cv->GetSize(), 2U); // [input_index, dim_index]
    const uint64_t *data = reinterpret_cast<const uint64_t *>(cv->GetData());
    GE_ASSERT_NOTNULL(data);
    const auto input_index = *data;
    GE_ASSERT(input_index < context->GetComputeNodeInputNum(), "Node %s input_index[%lu] must less than input num[%zu],"
              " i: %zu", context->GetNodeName(), input_index, context->GetComputeNodeInputNum(), i);
    const auto dim_index = *(data + 1U);
    auto in_shape = context->GetInputSymbolShape(input_index);
    GE_UNSUPPORTED_IF_NULL(in_shape);
    GE_ASSERT(dim_index < in_shape->GetDimNum(), "Node %s dim_index[%lu] must less than input shape"
              " dim number[%zu], i: %zu", context->GetNodeName(), dim_index, in_shape->GetDimNum(), i);
  }

  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  out_shape->Clear();
  out_shape->AppendDim(Symbol(axes_size));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(GatherShapes).InferSymbolShape(InferShape4GatherShapes);
}
}  // namespace ge

