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
graphStatus InferShape4ElementWise(gert::InferSymbolShapeContext *context) {
  auto in_shape1 = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape1);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  *out_shape = *in_shape1;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Abs).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SoftmaxV2).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Relu).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Elu).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(HcomAllReduce).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SigmoidGrad).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(IsNan).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Sign).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Exp).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Cast).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LogicalNot).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Neg).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Sqrt).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Rsqrt).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ZerosLike).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Sigmoid).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BiasAdd).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Square).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Erf).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Reciprocal).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(LeakyRelu).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Tanh).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(RsqrtGrad).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Muls).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Identity).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(TensorMove).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Log).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Log1p).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(SoftmaxGradExt).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(IsFinite).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(StopGradient).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ApplyGradientDescent).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(AssignAdd).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BNTrainingReduceGrad).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Gelu).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FusedMulAddN).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Softplus).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(BNInferenceD).InferSymbolShape(InferShape4ElementWise);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Adds).InferSymbolShape(InferShape4ElementWise);
} // namespace
} // namespace ge
