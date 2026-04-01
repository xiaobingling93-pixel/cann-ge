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
graphStatus ExpandShapeInference(const gert::SymbolShape *input_shape,
                const gert::SymbolTensor *target_tensor, gert::SymbolShape *output_shape) {
  size_t input_dim = input_shape->GetDimNum();
  auto target_dim = target_tensor->GetSymbolicValue()->size();
  auto diff = (target_dim > input_dim) ? (target_dim - input_dim) : (input_dim - target_dim);

  if (target_dim < input_dim) {
    for (size_t i = 0; i < diff; ++i) {
      output_shape->MutableDims().emplace_back(input_shape->GetDim(i));
    }
    for (size_t i = diff; i < input_dim; i++) {
      auto dim = target_tensor->GetSymbolicValue()->at(i - diff);
      if (dim == 0) {
        output_shape->MutableDims().emplace_back(dim);
        continue;
      }
      
      if (EXPECT_SYMBOL_EQ(input_shape->GetDim(i), Symbol(1))) {
        output_shape->MutableDims().emplace_back(dim);
      } else if (EXPECT_SYMBOL_EQ(dim, Symbol(1))) {
        output_shape->MutableDims().emplace_back(input_shape->GetDim(i));
      } else if (EXPECT_SYMBOL_EQ(input_shape->GetDim(i), dim)) {
        output_shape->MutableDims().emplace_back(dim);
      } else {
        GELOGE(GRAPH_PARAM_INVALID, "[InferSymbolShape4Expand], input dims cannot exand to target's dim, when input dim is %s, target dim is %s",
             input_shape->GetDim(i).Serialize().get(), dim.Serialize().get());
        return GRAPH_PARAM_INVALID;
      }
    }
  } else {
    for (size_t i = 0; i < diff; ++i) {
      output_shape->MutableDims().emplace_back(target_tensor->GetSymbolicValue()->at(i));
    }

    for (size_t i = diff; i < target_dim; i++) {
      auto dim = target_tensor->GetSymbolicValue()->at(i);
      if (dim == 0) {
        output_shape->MutableDims().emplace_back(dim);
        continue;
      }
        
      if (EXPECT_SYMBOL_EQ(input_shape->GetDim(i - diff), Symbol(1))) {
        output_shape->MutableDims().emplace_back(dim);
      } else if (EXPECT_SYMBOL_EQ(dim, Symbol(1))) {
        output_shape->MutableDims().emplace_back(input_shape->GetDim(i - diff));
      } else if (EXPECT_SYMBOL_EQ(input_shape->GetDim(i - diff), dim)) {
        output_shape->MutableDims().emplace_back(input_shape->GetDim(i - diff));
      } else {
        GELOGE(GRAPH_PARAM_INVALID, "[InferSymbolShape4Expand], input dims cannot exand to target's dim, when input dim is %s, target dim is %s",
             input_shape->GetDim(i - diff).Serialize().get(), dim.Serialize().get());
        return GRAPH_PARAM_INVALID;
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus InferShape4Expand(gert::InferSymbolShapeContext *context) {
  auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  auto target_shape = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(target_shape);
  auto output_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_shape);

  output_shape->MutableDims().clear();

  auto ret = ExpandShapeInference(input_shape, target_shape, output_shape);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Expand).InferSymbolShape(InferShape4Expand);
}

}