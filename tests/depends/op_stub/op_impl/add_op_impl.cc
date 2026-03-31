/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "common/checker.h"
using namespace gert;
namespace ops {
ge::graphStatus OpExecuteDoNothing(OpExecuteContext *) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferShapeDoNothing(InferShapeContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferDatatypeDoNothing(InferDataTypeContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferShapeForAdd(InferShapeContext *context) {
  GELOGD("InferShapeForAdd");
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    auto min_num = std::min(input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    auto max_num = std::max(input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    if (min_num != 1) {
      GELOGE(ge::PARAM_INVALID, "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
             input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    } else {
      if (input_shape_1.GetDimNum() > 1) {
        *output_shape = input_shape_1;
      } else {
        *output_shape = input_shape_0;
      }
      return ge::GRAPH_SUCCESS;
    }
  }
  if (input_shape_0.GetDimNum() == 0) {
    *output_shape = input_shape_1;
    return ge::GRAPH_SUCCESS;
  }
  if (input_shape_1.GetDimNum() == 0) {
    *output_shape = input_shape_0;
    return ge::GRAPH_SUCCESS;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }

  return ge::GRAPH_SUCCESS;
}

struct AddCompileInfo {
  int64_t a;
  int64_t b;
};
ge::graphStatus TilingForAdd(TilingContext *context) {
  GELOGD("TilingForAdd");
  auto ci = context->GetCompileInfo<AddCompileInfo>();
  GE_ASSERT_NOTNULL(ci);
  auto tiling_data = context->GetRawTilingData();
  GE_ASSERT_NOTNULL(tiling_data);
  tiling_data->Append(*ci);
  tiling_data->Append(ci->a);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingParseForAdd(TilingParseContext *context) {
  auto compile_info = context->GetCompiledInfo<AddCompileInfo>();
  compile_info->a = 10;
  compile_info->b = 200;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Add).InferShape(InferShapeForAdd).Tiling(TilingForAdd).TilingParse<AddCompileInfo>(TilingParseForAdd)
            .OpExecuteFunc(OpExecuteDoNothing);

ge::graphStatus InferShapeForCast(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForCast(TilingContext *context) {
  return ge::GRAPH_SUCCESS;
}

struct CastCompileInfo {
  int64_t a;
};

ge::graphStatus TilingParseForCast(TilingParseContext *context) {
  auto compile_info = context->GetCompiledInfo<CastCompileInfo>();
  compile_info->a = 10;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Cast).InferShape(InferShapeForCast).Tiling(TilingForCast).TilingParse<CastCompileInfo>(TilingParseForCast);

ge::graphStatus InferShapeForMul(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);

  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    return ge::GRAPH_FAILED;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i) * input_shape_1.GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Mul).InferShape(InferShapeForMul);

ge::graphStatus InferShapeForIdentityN(InferShapeContext *context) {
  auto extend_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto input_num = extend_kernel_context->GetComputeNodeInputNum();
  for (size_t i = 0U; i < input_num; ++i) {
    auto x_shape = context->GetInputShape(i);
    auto y_shape = context->GetOutputShape(i);
    if ((x_shape == nullptr) || (y_shape == nullptr)) {
      return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(Identity).InferShape(InferShapeForIdentityN);
IMPL_OP(MemcpyAsync).InferShape(InferShapeForIdentityN);
IMPL_OP(IdentityN).InferShape(InferShapeForIdentityN);


ge::graphStatus InferShapeForMinimumGrad(InferShapeContext *context) {
  if (context->GetAttrs()->GetBool(0)) {  // grad_x
    *context->GetOutputShape(0) = *context->GetInputShape(1);
  }
  if (context->GetAttrs()->GetBool(1)) {  // grad_y
    *context->GetOutputShape(1) = *context->GetInputShape(2);
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(MinimumGrad).InferShape(InferShapeForMinimumGrad);

constexpr int64_t kPrivateAttrValue = 0;
IMPL_OP(FileConstant)
    .PrivateAttr("offset", kPrivateAttrValue)
    .PrivateAttr("length", kPrivateAttrValue)
    .PrivateAttr("location", "");

IMPL_OP(ConstPlaceHolder);

IMPL_OP(ConditionCalc);

IMPL_OP(StatelessIf).InputsDataDependency({0});

IMPL_OP(If).InputsDataDependency({0});
IMPL_OP(_If).InputsDataDependency({0});

IMPL_OP(Case).InputsDataDependency({0});

IMPL_OP(NetOutput).InferDataType(InferDatatypeDoNothing).InferShape(InferShapeDoNothing);
IMPL_OP(Data).InferDataType(InferDatatypeDoNothing).InferShape(InferShapeDoNothing);
IMPL_OP(Const).InferDataType(InferDatatypeDoNothing).InferShape(InferShapeDoNothing);
IMPL_OP(Constant).InferDataType(InferDatatypeDoNothing).InferShape(InferShapeDoNothing);
IMPL_OP(Variable).InferDataType(InferDatatypeDoNothing).InferShape(InferShapeDoNothing);
IMPL_OP(FileConstant).InferDataType(InferDatatypeDoNothing).InferShape(InferShapeDoNothing);

IMPL_OP(PartitionedCall);

IMPL_OP(While);
IMPL_OP(MoeFFN).TilingInputsDataDependency({1});
IMPL_OP(IFN).TilingInputsDataDependency({1}, {TilingPlacement::TILING_ON_HOST, TilingPlacement::TILING_ON_AICPU});
ge::graphStatus InferShapeForAsString(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  GE_ASSERT_NOTNULL(output_shape);
  *output_shape = input_shape_0;
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(AsString).InferShape(InferShapeForAsString);

IMPL_OP(Const).InputsDataDependency({});
IMPL_OP(Variable).InputsDataDependency({});
IMPL_OP(VariableV2).InputsDataDependency({});
IMPL_OP(Constant).InputsDataDependency({});
IMPL_OP(Add).InputsDataDependency({});
IMPL_OP(Mul).InputsDataDependency({});
IMPL_OP(MatMul).InputsDataDependency({});
IMPL_OP(MatMulV2).InputsDataDependency({});
IMPL_OP(Abs).InputsDataDependency({});
IMPL_OP(Less).InputsDataDependency({});
IMPL_OP(Relu).InputsDataDependency({});
IMPL_OP(ReduceSumD).InputsDataDependency({});
IMPL_OP(ReduceMaxD).InputsDataDependency({});
IMPL_OP(ReduceMinD).InputsDataDependency({});
IMPL_OP(ReduceProdD).InputsDataDependency({});
IMPL_OP(ReduceMeanD).InputsDataDependency({});
IMPL_OP(ReduceAllD).InputsDataDependency({});
IMPL_OP(ReduceAnyD).InputsDataDependency({});
IMPL_OP(ReduceSum).InputsDataDependency({1});
IMPL_OP(ReduceMax).InputsDataDependency({1});
IMPL_OP(ReduceMin).InputsDataDependency({1});
IMPL_OP(ReduceProd).InputsDataDependency({1});
IMPL_OP(ReduceMean).InputsDataDependency({1});
IMPL_OP(ReduceAll).InputsDataDependency({1});
IMPL_OP(ReduceAny).InputsDataDependency({1});
IMPL_OP(IsNan).InputsDataDependency({});
IMPL_OP(Sign).InputsDataDependency({});
IMPL_OP(Exp).InputsDataDependency({});
IMPL_OP(Sub).InputsDataDependency({});
IMPL_OP(RealDiv).InputsDataDependency({});
IMPL_OP(Cast).InputsDataDependency({});
IMPL_OP(Equal).InputsDataDependency({});
IMPL_OP(NotEqual).InputsDataDependency({});
IMPL_OP(Greater).InputsDataDependency({});
IMPL_OP(Maximum).InputsDataDependency({});
IMPL_OP(Minimum).InputsDataDependency({});
IMPL_OP(LogicalAnd).InputsDataDependency({});
IMPL_OP(LogicalOr).InputsDataDependency({});
IMPL_OP(LogicalNot).InputsDataDependency({});
IMPL_OP(Div).InputsDataDependency({});
IMPL_OP(Neg).InputsDataDependency({});
IMPL_OP(Sqrt).InputsDataDependency({});
IMPL_OP(Rsqrt).InputsDataDependency({});
IMPL_OP(Pow).InputsDataDependency({});
IMPL_OP(ZerosLike).InputsDataDependency({});
IMPL_OP(Slice).InputsDataDependency({1, 2});
IMPL_OP(Split).InputsDataDependency({0});
IMPL_OP(ConcatV2D).InputsDataDependency({});
IMPL_OP(ConcatV2).InputsDataDependency({1});
IMPL_OP(Range).InputsDataDependency({0, 1, 2});
IMPL_OP(Fill).InputsDataDependency({0});
IMPL_OP(SplitV).InputsDataDependency({1, 2});
IMPL_OP(Select).InputsDataDependency({});
IMPL_OP(Tile).InputsDataDependency({1});
IMPL_OP(TileD).InputsDataDependency({});
IMPL_OP(Transpose).InputsDataDependency({1});
IMPL_OP(TransposeD).InputsDataDependency({1});
IMPL_OP(Reshape).InputsDataDependency({1});
IMPL_OP(Squeeze).InputsDataDependency({});
IMPL_OP(SqueezeV3).InputsDataDependency({1});
IMPL_OP(Unsqueeze).InputsDataDependency({});
IMPL_OP(UnsqueezeV3).InputsDataDependency({1});
IMPL_OP(GatherV2).InputsDataDependency({2});
IMPL_OP(Pack).InputsDataDependency({});
IMPL_OP(Unpack).InputsDataDependency({}).PrivateAttr("axis", static_cast<int64_t>(0));
IMPL_OP(LessEqual).InputsDataDependency({});
IMPL_OP(ExpandDims).InputsDataDependency({1});
IMPL_OP(Pad).InputsDataDependency({1});
IMPL_OP(StridedSlice).InputsDataDependency({1, 2, 3});
IMPL_OP(BatchMatMulV2).InputsDataDependency({});
IMPL_OP(Sigmoid).InputsDataDependency({});
IMPL_OP(foo1).InferShape(nullptr);
IMPL_OP(BroadcastTo).InputsDataDependency({1});
IMPL_OP(Repeat).InputsDataDependency({1});
IMPL_OP(PadV3).InputsDataDependency({1});
IMPL_OP(UnsortedSegmentMin).InputsDataDependency({2});
IMPL_OP(UnsortedSegmentMax).InputsDataDependency({2});
IMPL_OP(FlattenV2).InputsDataDependency({});
}  // namespace ops
