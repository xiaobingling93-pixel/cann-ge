/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascendc_api_perf.h"
namespace att {
// 这是一个可选文件
// 下列是内置公式，可以按照自己需要选择，若仍不满足要求，用户可以自己定义
//REGISTER_EVAL_FUNC(kMoveGmToL1, CopyGMtoL1, PipeType::AICORE_MTE2);
//REGISTER_EVAL_FUNC(kMoveL2ToL1, Copy, PipeType::AICORE_MTE2);
//REGISTER_EVAL_FUNC(kMoveL1ToL0a, CopyFromL1, PipeType::AICORE_MTE1);
//REGISTER_EVAL_FUNC(kMoveL1ToL0b, CopyFromL1, PipeType::AICORE_MTE1);
//REGISTER_EVAL_FUNC(kMoveL0cToL2, Copy, PipeType::AIC_FIXPIPE);
//REGISTER_EVAL_FUNC(kMoveL0cToGm, Copy, PipeType::AIC_FIXPIPE);
//REGISTER_EVAL_FUNC(kComputeVector, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kComputeCube, CubeCompute, PipeType::AICORE_CUBE);
//REGISTER_EVAL_FUNC(kAbs, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kMoveUbToGm, StoreApi, PipeType::AIV_MTE3);
//REGISTER_EVAL_FUNC(kMoveGmToUb, LoadApi, PipeType::AIV_MTE2);
//REGISTER_EVAL_FUNC(kExp, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kSub, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kDiv, DivCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kMax, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kSum, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kCast, CastCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kBroadcast, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kMul, MulCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kNeg, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kAdd, AddCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kWhere, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kRsqrt, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kConstant, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kReduction, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kStoreReduce, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kToType, VectorCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kSigmoid, Sigmoid, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kMuls, MulsCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kFlashSoftmax, SoftmaxFlashV2, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kDropOut, DropoutCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kSelect, SelectCompute, PipeType::AICORE_VEC);
//REGISTER_EVAL_FUNC(kMatMul, MatmulCompute, PipeType::AIC_MTE2);
// 如果你希望自己自定义性能公式, 你应该调用REGISTER_EVAL_FUNC并且传入下列信息:
// 1. 表达式名
// 2. 性能计算公式
// 3. 这个API使用的PipeType
// 示例:
constexpr char kVfCall1[] = "VfCall1";
ge::Status VfCall1Compute(const std::vector<TensorShapeInfo> &input_shapes,
              const std::vector<TensorShapeInfo> &output_shapes,
              const ge::AscNodePtr &node,
              std::map<PipeType, Expr> &res,
              std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
  (void)input_shapes;
  (void)output_shapes;
  (void)node;
  return ge::SUCCESS;
}
REGISTER_EVAL_FUNC(kVfCall1, VfCall1Compute);

}