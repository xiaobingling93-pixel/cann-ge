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
#include "graph/utils/type_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

graphStatus ValidateInputShapes(const gert::SymbolShape *diagonal_shape, const gert::SymbolShape *k_shape) {
  GE_ASSERT_TRUE(diagonal_shape->GetDimNum() >= 1,
         "[InferSymbolShape4MatrixDiagV2] diagonal_shape is invalid, dim should be at least 1");

  GE_ASSERT_TRUE(k_shape->GetDimNum() <= 1,
         "[InferSymbolShape4MatrixDiagV2] k_shape is invalid, dim should be at most 1");

  return GRAPH_SUCCESS;
}

graphStatus ExtractDiagIndices(const std::vector<Expression> *k_value, int32_t &lower_diag_index, int32_t &upper_diag_index) {
  int32_t num_elements = k_value->size();

  GE_ASSERT_TRUE(num_elements >= 1 && num_elements <= 2,
          "[InferSymbolShape4MatrixDiagV2] input[k] must be scalar or a vector with one or two elements");
  GE_ASSERT_TRUE(k_value->at(0).GetConstValue<int32_t>(lower_diag_index),
          "[InferSymbolShape4MatrixDiagV2] k_value[0] must be a scalar");

  if (num_elements == 1) {
    upper_diag_index = lower_diag_index;
  } else {
    GE_ASSERT_TRUE(k_value->at(1).GetConstValue<int32_t>(upper_diag_index),
            "[InferSymbolShape4MatrixDiagV2] k_value[1] must be a scalar");
  }

  GE_ASSERT_TRUE(lower_diag_index <= upper_diag_index,
          "[InferSymbolShape4MatrixDiagV2] lower_diag_index must be less than or equal to upper_diag_index");

  GELOGI("lower_diag_index %d, upper_diag_index %d", lower_diag_index, upper_diag_index);

  return GRAPH_SUCCESS;
}

graphStatus ValidateMultiDiag(const std::vector<Expression> &diagonal_dims, size_t diagonal_rank,
                 int32_t lower_diag_index, int32_t upper_diag_index) {
  GE_ASSERT_TRUE(diagonal_rank >= 2,
          "[InferSymbolShape4MatrixDiagV2] diagonal_shape is invalid, dim should be at least 2");

  auto num_diags = diagonal_dims[diagonal_rank - 2];
  auto expected_num_diags = Symbol(upper_diag_index - lower_diag_index + 1);
  ASSERT_SYMBOL_EQ(num_diags, expected_num_diags);

  return GRAPH_SUCCESS;
}

void ComputeMinDimensions(const std::vector<Expression> &diagonal_dims, size_t diagonal_rank,
              int32_t lower_diag_index, int32_t upper_diag_index,
              Expression &min_num_rows, Expression &min_num_cols) {
  auto max_diag_len = diagonal_dims[diagonal_rank - 1];
  min_num_rows = max_diag_len - Symbol(std::min(lower_diag_index, 0));
  min_num_cols = max_diag_len + Symbol(std::max(upper_diag_index, 0));

  GELOGI("max_diag_len %s, min_num_rows %s, min_num_cols %s",
       max_diag_len.Serialize().get(), min_num_rows.Serialize().get(), min_num_cols.Serialize().get());
}

graphStatus ProcessNumRowsCols(const gert::SymbolTensor *num_rows_tensor, const gert::SymbolTensor *num_cols_tensor,
                const Expression &min_num_rows, const Expression &min_num_cols,
                Expression &num_rows, Expression &num_cols) {
  auto num_rows_value = num_rows_tensor->GetSymbolicValue();
  auto num_cols_value = num_cols_tensor->GetSymbolicValue();

  if (num_rows_value != nullptr) {
    num_rows = num_rows_value->at(0);
  }
  if (num_cols_value != nullptr) {
    num_cols = num_cols_value->at(0);
  }

  if (num_rows_value == nullptr && num_cols_value == nullptr) {
    num_rows = ge::sym::Max(min_num_rows, min_num_cols);
    num_cols = num_rows;
  } else if (num_rows_value == nullptr) {
    num_rows = min_num_cols;
  } else if (EXPECT_SYMBOL_LT(num_rows, min_num_rows)) {
    GELOGE(GRAPH_PARAM_INVALID, "[InferSymbolShape4MatrixDiagV2] num_rows%s is invalid, it must be greater than or equal to min_num_rows %s",
         num_rows.Serialize().get(), min_num_rows.Serialize().get());
    return GRAPH_PARAM_INVALID;
  }

  if (num_cols_value == nullptr) {
    num_cols = min_num_cols;
  } else if (EXPECT_SYMBOL_LT(num_cols, min_num_cols)) {
    GELOGE(GRAPH_PARAM_INVALID, "[InferSymbolShape4MatrixDiagV2] num_cols%s is invalid, it must be greater than or equal to min_num_cols %s",
         num_cols.Serialize().get(), min_num_cols.Serialize().get());
    return GRAPH_PARAM_INVALID;
  }

  if (EXPECT_SYMBOL_LE(num_rows, Symbol(0))) {
    num_rows = min_num_rows;
    GELOGI("num_rows is <= 0, use min_num_rows %s", min_num_rows.Serialize().get());
  } else {
    GELOGI("check num_rows is valid, num_rows : %s , min_num_rows : %s", num_rows.Serialize().get(), min_num_rows.Serialize().get());
    ASSERT_SYMBOL_GE(num_rows, min_num_rows);
  }

  if (EXPECT_SYMBOL_LE(num_cols, Symbol(0))) {
    num_cols = min_num_cols;
    GELOGI("num_cols is <= 0, use min_num_cols %s", min_num_cols.Serialize().get());
  } else {
    GELOGI("check num_cols is valid, num_cols : %s , min_num_cols : %s", num_cols.Serialize().get(), min_num_cols.Serialize().get());
    ASSERT_SYMBOL_GE(num_cols, min_num_cols);
  }

  return GRAPH_SUCCESS;
}

void BuildOutputShape(const std::vector<Expression> &diagonal_dims, size_t diagonal_rank,
            int32_t lower_diag_index, int32_t upper_diag_index,
            const Expression &num_rows, const Expression &num_cols,
            gert::SymbolShape *output_shape) {
  if (lower_diag_index == upper_diag_index) {
    for (size_t i = 0; i < diagonal_rank - 1; i++) {
      output_shape->MutableDims().push_back(diagonal_dims[i]);
    }
    output_shape->MutableDims().push_back(num_rows);
    output_shape->MutableDims().push_back(num_cols);
  } else {
    for (size_t i = 0; i < diagonal_rank; i++) {
      if (i == diagonal_rank - 2) {
        output_shape->MutableDims().push_back(num_rows);
      } else if (i == diagonal_rank - 1) {
        output_shape->MutableDims().push_back(num_cols);
      } else {
        output_shape->MutableDims().push_back(diagonal_dims[i]);
      }
    }
  }
}

graphStatus InferShape4MatrixDiagV2(gert::InferSymbolShapeContext *context) {
  auto diagonal_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(diagonal_shape);
  auto k_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(k_shape);
  auto k_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(k_tensor);
  auto num_rows_tensor = context->GetInputSymbolTensor(2);
  GE_UNSUPPORTED_IF_NULL(num_rows_tensor);
  auto num_cols_tensor = context->GetInputSymbolTensor(3);
  GE_UNSUPPORTED_IF_NULL(num_cols_tensor);
  auto output_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_shape);
  output_shape->MutableDims().clear();

  GE_ASSERT_GRAPH_SUCCESS(ValidateInputShapes(diagonal_shape, k_shape));

  auto k_value = k_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(k_value);

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  GE_ASSERT_GRAPH_SUCCESS(ExtractDiagIndices(k_value, lower_diag_index, upper_diag_index));

  auto diagonal_dims = diagonal_shape->GetDims();
  size_t diagonal_rank = diagonal_shape->GetDimNum();

  if (lower_diag_index < upper_diag_index) {
    GE_ASSERT_GRAPH_SUCCESS(ValidateMultiDiag(diagonal_dims, diagonal_rank, lower_diag_index, upper_diag_index));
  }

  Expression min_num_rows;
  Expression min_num_cols;
  ComputeMinDimensions(diagonal_dims, diagonal_rank, lower_diag_index, upper_diag_index, min_num_rows, min_num_cols);

  Expression num_rows;
  Expression num_cols;
  GE_ASSERT_GRAPH_SUCCESS(ProcessNumRowsCols(num_rows_tensor, num_cols_tensor, min_num_rows, min_num_cols, num_rows, num_cols));

  BuildOutputShape(diagonal_dims, diagonal_rank, lower_diag_index, upper_diag_index, num_rows, num_cols, output_shape);

  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MatrixDiagV2).InferSymbolShape(InferShape4MatrixDiagV2);
}
} 