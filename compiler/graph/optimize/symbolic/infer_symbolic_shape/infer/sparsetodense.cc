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
graphStatus GetConstInt(const Expression &expr, DataType dt, int64_t &value) {
  if (dt == DT_INT32) {
    int32_t tmp_value = 0;
    if (expr.GetConstValue<int32_t>(tmp_value) == false) {
      return UNSUPPORTED;
    }
    value = static_cast<int64_t>(tmp_value);
  } else {
    if (expr.GetConstValue<int64_t>(value) == false) {
      return UNSUPPORTED;
    }
  }
  return GRAPH_SUCCESS;
}

/**
* SparseToDense算子的符号化shape推导
* 【算子功能】将稀疏张量转换为稠密张量，即根据稀疏表示中的索引和值，填充到指定形状的稠密输出张量中，未指定位置的元素使用默认值填充
* 【算子约束】
*      1. 输出shape由名为output_shape的输入决定，该输入shape的dim_num为1，同时值依赖该输入
*      2. output_shape输入的数据类型为DT_INT32或DT_INT64
* 【推导逻辑】
*      1. 按照算子约束校验名为output_shape的输入是否满足符号化推导
*      2. 获取到值依赖的SymbolicValue,按照shape的长度将存储的dim值作为输出shape
*/
graphStatus InferShape4SparseToDense(gert::InferSymbolShapeContext* context) {
  auto input1_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(input1_shape);

  size_t input1_dim_size = input1_shape->GetDimNum();
  GE_ASSERT(input1_dim_size == 1U, "the shape rank of output_shape should be 1, node %s", context->GetNodeName());

  auto dim_num_expr = input1_shape->GetDim(0);
  if (!dim_num_expr.IsConstExpr()) {
    GELOGW("Symbol Infer unsupported, input 1 shape is not ConstExpr, node %s", context->GetNodeName());
    return UNSUPPORTED;
  }

  auto input1_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(input1_tensor);
  if (input1_tensor->GetSymbolicValue() == nullptr || input1_tensor->GetSymbolicValue()->empty()) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr or empty, node %s", context->GetNodeName());
    return UNSUPPORTED;
  }

  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  out_shape->Clear();

  auto shape_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(shape_desc);
  auto dt = shape_desc->GetDataType();
  if (dt != DT_INT32 && dt != DT_INT64) {
    GELOGE(PARAM_INVALID, "DataType must in [int32, int64], node %s", context->GetNodeName());
    return PARAM_INVALID;
  }

  int64_t dim_num = -2;
  auto ret = GetConstInt(dim_num_expr, DT_INT64, dim_num);
  if (ret == UNSUPPORTED) {
    GELOGW("Symbol Infer unsupported, input 1 shape is not constvalue, node %s", context->GetNodeName());
    return ret;
  }
  dim_num = (dim_num > 0) ? dim_num : 1;
  auto value_num = input1_tensor->GetSymbolicValue()->size();
  GE_ASSERT(static_cast<size_t>(dim_num) <= value_num, "dim_num[%ld] should less than value_num[zu], node %s",
            dim_num, value_num, context->GetNodeName());

  for (int64_t i = 0; i < dim_num; i++) {
    auto dim_expr = input1_tensor->GetSymbolicValue()->at(i);
    if (dim_expr.IsConstExpr()) {
      int64_t dim = -2;
      auto ret = GetConstInt(dim_expr, dt, dim);
      if (ret == UNSUPPORTED) {
        GELOGW("Symbol Infer unsupported, get dim at index[%ld] is not constvalue, node %s", i, context->GetNodeName());
        return ret;
      }
    }
    out_shape->AppendDim(dim_expr);
  }

  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SparseToDense).InferSymbolShape(InferShape4SparseToDense);
}
}  // namespace ge

