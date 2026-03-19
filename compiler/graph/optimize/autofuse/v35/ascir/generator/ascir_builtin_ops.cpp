/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascir_register.h"
#include "v2_ascir_codegen_impl.h"
#include "v2_ascir_att_impl.h"
#include "graph/types.h"

namespace ge {
namespace ascir {
EXPORT_GENERATOR()

// todo:: 暂时先定义一个AscIrAttStubV2用于注册, 后面att根据需要替换成具体的impl子类
class AscIrAttStubV2 : public ge::ascir::AscIrAtt {
  void *GetApiPerf() const override {
    return nullptr;
  }
  void *GetAscendCApiPerfTable() const override {
    return nullptr;
  }
};

const std::vector<std::string> v2_soc_versions{"3510", "5102"};

REG_ASC_IR(VectorFunc)
    .DynamicInput("x", "T")
    .DynamicOutput("y", "T")
    .Attr<std::string>("sub_graph_name")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::VectorFuncAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::VfAscIrCodegenImpl>(),
                            {{"T", TensorType::ALL()}}});

REG_ASC_IR(Data)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::DataAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::DataAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Scalar)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ScalarAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ScalarAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(IndexExpr)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::IndexExprAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::IndexExprAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Output)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::OutputAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::OutputAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Workspace)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::WorkspaceAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::WorkspaceAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Load)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LoadAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LoadAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16}}}});

REG_ASC_IR(Nddma)
    .Input("x", "T")
    .Output("y", "T")
    .Attr<Expression>("offset")
    .ComputeType(ge::ComputeType::kComputeLoad)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NddmaAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NddmaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16}}}});

REG_ASC_IR(Store)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::StoreAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::StoreAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16, DT_UINT64}}}});

// todo: Broadcast DT_INT64 后面根据需要放开
REG_ASC_IR(Broadcast)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BroadcastAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BroadcastAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_UINT8, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT16,
                                              DT_UINT32, DT_UINT64, DT_INT64}}}});

REG_ASC_IR(RemovePad)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RemovePadAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RemovePadAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Pad)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::PadAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::PadAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Round)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT, DT_BF16, DT_FLOAT16})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RoundAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RoundAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_BF16, DT_FLOAT16}}}});

// todo: Nop DT_INT64 后面根据需要放开
REG_ASC_IR(Nop)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NopAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NopAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_FLOAT16, DT_FLOAT}}}});

/* cast 暂时先放开int64->float, 以下类型, 暂不放开
 * T1:DT_INT64, DT_INT64, DT_INT64, DT_INT64,
 * T2:DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_UINT64,
 */
REG_ASC_IR(Cast)
    .Impl(v2_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::CastAscIrAttImplV2>(),
           ge::ascir::AscIrImplCreator<ge::ascir::CastAscIrCodegenImplV2>(),
           {{"T1", OrderedTensorTypeList{DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                                         DT_FLOAT,   DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                         DT_FLOAT16, DT_FLOAT16, DT_INT4,    DT_UINT8,   DT_UINT8,   DT_UINT8,
                                         DT_UINT8,   DT_UINT8,   DT_UINT8,   DT_INT8,    DT_INT8,    DT_INT8,
                                         DT_INT8,    DT_INT16,   DT_INT16,   DT_INT16,   DT_INT16,   DT_INT16,
                                         DT_INT32,   DT_INT32,   DT_INT32,   DT_INT32,   DT_INT32,   DT_INT64,
                                         DT_INT64,   DT_INT64,   DT_INT64,   DT_INT64,   DT_BF16,    DT_BF16,
                                         DT_UINT32,  DT_UINT16,  DT_UINT64}},
            {"T2", OrderedTensorTypeList{DT_FLOAT,   DT_FLOAT16, DT_INT64,   DT_INT32,   DT_INT16,   DT_BF16,
                                         DT_INT8,    DT_FLOAT,   DT_INT32,   DT_INT16,   DT_INT8,    DT_UINT8,
                                         DT_INT4,    DT_INT64,   DT_FLOAT16, DT_FLOAT16, DT_FLOAT,   DT_INT32,
                                         DT_INT16,   DT_INT8,    DT_INT4,    DT_FLOAT16, DT_UINT8,   DT_FLOAT,
                                         DT_INT16,   DT_FLOAT16, DT_FLOAT,   DT_UINT16,  DT_INT8,    DT_UINT8,
                                         DT_FLOAT,   DT_INT64,   DT_INT16,   DT_FLOAT16, DT_UINT32,  DT_INT32,
                                         DT_FLOAT,   DT_UINT8,   DT_UINT64,  DT_FLOAT16, DT_FLOAT,   DT_INT32,
                                         DT_INT32,   DT_INT16,   DT_INT64}}}});

REG_ASC_IR(Abs)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AbsAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AbsAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                                              DT_UINT8}}}});

REG_ASC_IR(Exp)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ExpAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ExpAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Exp2)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::Exp2AscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::Exp2AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Floor)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::FloorAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::FloorAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Fma)
    .Input("x1", "T")
    .Input("x2", "T")
    .Input("x3", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::FmaAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::FmaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Ln)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LnAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LnAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Log2)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::Log2AscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::Log2AscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(LShift)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .DataType("T1", TensorType{DT_INT8, DT_INT16, DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_INT8, DT_INT16, DT_INT32, DT_INT64})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LShiftAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LShiftAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_INT8, DT_INT16, DT_INT32, DT_INT64}}, 
                             {"T2", TensorType{DT_INT8, DT_INT16, DT_INT32, DT_INT64}}}});

REG_ASC_IR(Mod)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_UINT8})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ModAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ModAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT16, DT_UINT8}}}});

REG_ASC_IR(Sqrt)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SqrtAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SqrtAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Rsqrt)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RsqrtAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RsqrtAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Reciprocal)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ReciprocalAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ReciprocalAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT64, DT_UINT64}}}});

REG_ASC_IR(Erf)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ErfAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ErfAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Sign)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SignAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SignAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT8, DT_BF16}}}});

REG_ASC_IR(Tanh)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<AscIrAttStubV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::TanhAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Isnan)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::IsnanAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::IsnanAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(IsFinite)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::IsFiniteAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::IsFiniteAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Relu)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ReluAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ReluAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT64}}}});

REG_ASC_IR(Neg)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NegAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NegAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_INT8, DT_INT64, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

// todo: LogicalNot DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalNot)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LogicalNotAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LogicalNotAscIrCodegenImplV2>(),
                            {{"T1", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT16, DT_INT32, DT_INT8, DT_INT64}},
                             {"T2", OrderedTensorTypeList{DT_UINT8,   DT_UINT8, DT_UINT8, DT_UINT8, DT_UINT8, DT_UINT8, DT_UINT8}}}});

REG_ASC_IR(Max)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MaxAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MaxAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Sum)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SumAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SumAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_INT16, DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Min)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ReciprocalAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MinAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Mean)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MeanAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MeanAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Prod)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ProdAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ProdAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Sigmoid)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SigmoidAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SigmoidAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Any)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AnyAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AnyAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(All)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AllAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AllAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Add)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AddAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AddAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT64,
                                              DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(Sub)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SubAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SubAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(Div)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::DivAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::DivAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Mul)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MulAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_BF16, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT64}}}});

REG_ASC_IR(Minimum)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MinimumAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MinimumAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_INT8, DT_INT16, DT_INT64, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8}}}});

REG_ASC_IR(Maximum)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MaximumAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MaximumAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_BF16, DT_INT8, DT_INT16, DT_INT64, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8}}}});

REG_ASC_IR(TrueDiv)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::TrueDivAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::TrueDivAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

// todo:LogicalOr DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalOr)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LogicalOrAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LogicalOrAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, DT_UINT32, DT_INT64}},
                             {"T2", TensorType{DT_UINT8}}}});

// todo:LogicalAnd DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalAnd)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LogicalAndAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LogicalAndAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT8, DT_INT64,
                                               DT_UINT32, DT_BF16}},
                             {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Pow)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::PowAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::PowAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_UINT16, DT_UINT32, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(ClipByValue)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ClipByValueAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ClipByValueAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}}});

// todo:Ge Eq Ne Gt Le  DT_INT64 后面根据需要放开
REG_ASC_IR(Ge)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::GeAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::GeAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Eq)
    .Impl(v2_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::EqAscIrAttImplV2>(),
           ge::ascir::AscIrImplCreator<ge::ascir::EqAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_INT16, DT_UINT8, DT_UINT16,
                              DT_UINT32, DT_UINT64}},
            {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Ne)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NeAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NeAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Gt)
    .Impl(v2_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::GtAscIrAttImplV2>(),
           ge::ascir::AscIrImplCreator<ge::ascir::GtAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Le)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LeAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LeAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Lt)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LtAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LtAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32,
                                               DT_INT64, DT_UINT8}},
                            {"T2", TensorType{DT_UINT8}}}});

// todo:Concat DT_INT64 后面根据需要放开
REG_ASC_IR(Concat)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ConcatAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ConcatAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                              DT_INT64, DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Split)
    .Input("x", "T")
    .DynamicOutput("y", "T")
    .Attr<int64_t>("index")
    .Attr<int64_t>("gid")   // global_id, SplitOp的全局编号
    .ComputeType(ge::ComputeType::kComputeSplit)
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SplitAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SplitAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                            DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                            DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                            DT_BF16,       DT_BOOL}}}});

REG_ASC_IR(Select)
    .Impl(v2_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::SelectAscIrAttImplV2>(),
           ge::ascir::AscIrImplCreator<ge::ascir::SelectAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64}}}});

REG_ASC_IR(Where)
    .Impl(v2_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::WhereAscIrAttImplV2>(),
           ge::ascir::AscIrImplCreator<ge::ascir::WhereAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_BF16, DT_INT8, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}}}});

// Ub2ub是在sched阶段添加的，不需要在py构图中对外体现
// todo:Ub2ub DT_INT64 后面根据需要放开
REG_ASC_IR(Ub2ub)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::Ub2ubAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::Ub2ubAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(LeakyRelu)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LeakyReluAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LeakyReluAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo:BitwiseAnd DT_INT64 后面根据需要放开
REG_ASC_IR(BitwiseAnd)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BitwiseAndAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BitwiseAndAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64}}}});

REG_ASC_IR(BitwiseNot)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BitwiseNotAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BitwiseNotAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64}}}});

REG_ASC_IR(BitwiseOr)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BitwiseOrAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BitwiseOrAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64}}}});

REG_ASC_IR(BitwiseXor)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BitwiseXorAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BitwiseXorAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8, DT_INT8, DT_INT64, DT_UINT32,
                                              DT_UINT64}}}});

REG_ASC_IR(Ceil)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::CeilAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::CeilAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Cos)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::CosAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::CosAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Acos)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AcosAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AcosAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Cosh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::CoshAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::CoshAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Digamma)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::DigammaAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::DigammaAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Erfc)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ErfcAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ErfcAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Gather)
    .Impl(v2_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::GatherAscIrAttImplV2>(),
           ge::ascir::AscIrImplCreator<ge::ascir::GatherAscIrCodegenImplV2>(),
           {{"T1", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT}},
            {"T2", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(Transpose)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::TransposeAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::TransposeAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});
// todo:目前前端dt构图用到了FlashSoftmax，暂时无法删除
REG_ASC_IR(FlashSoftmax)
    .Impl({}, {ge::ascir::AscIrImplCreator<ge::ascir::AbsAscIrAttImplV2>(),
               ge::ascir::AscIrImplCreator<ge::ascir::AscIrCodegen>(),
               {{"T1", TensorType{DT_INT8, DT_INT16}}, {"T2", TensorType{DT_UINT8, DT_INT16}}}});

REG_ASC_IR(FloorDiv)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::FloorDivAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::FloorDivAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_UINT8}}}});

REG_ASC_IR(Gelu)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::GeluAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::GeluAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Axpy)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AxpyAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AxpyAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16, DT_UINT64, DT_INT64}}}});
REG_ASC_IR(MatMul)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulBias)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulOffset)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(MatMulOffsetBias)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMul)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulBias)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulOffset)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMulOffsetBias)
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImplV2>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(Sin)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SinAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SinAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Acosh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AcoshAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AcoshAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Asin)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AsinAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AsinAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Asinh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AsinhAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AsinhAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Atan)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AtanAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AtanAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(Atanh)
    .Input("x", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AtanhAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AtanhAscIrCodegenImplV2>(),
                            {{"T", TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}}}});

REG_ASC_IR(RShift)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", OrderedTensorTypeList{DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64})
    .DataType("T2", OrderedTensorTypeList{DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_INT16, DT_INT32, DT_INT64})
    .Impl(v2_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RShiftAscIrAttImplV2>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RShiftAscIrCodegenImplV2>(),
                            {{"T1", OrderedTensorTypeList{DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                                                          DT_UINT32, DT_UINT64}},
                             {"T2", OrderedTensorTypeList{DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_INT16,
                                                          DT_INT32, DT_INT64}}}});
}  // namespace ascir
}
