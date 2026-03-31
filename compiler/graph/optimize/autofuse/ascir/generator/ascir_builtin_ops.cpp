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
#include "v1_ascir_codegen_impl.h"
#include "v1_ascir_att_impl.h"
#include "graph/types.h"
#include "graph/tensor.h"

namespace ge {
namespace ascir {
EXPORT_GENERATOR()

const std::vector<std::string> v1_soc_versions{"2201"};

REG_ASC_IR(Data)
    .Inputs({})
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                              DT_FLOAT16, DT_FLOAT})
    .StartNode()
    .Attr<int64_t>("index")
    .ComputeType(ge::ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::DataAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::DataAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(VectorFunc)
    .DynamicInput("x", "T")
    .DynamicOutput("y", "T")
    .Attr<std::string>("sub_graph_name")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {nullptr, nullptr, {{"T", TensorType::ALL()}}});

REG_ASC_IR(Scalar)
    .Inputs({})
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                              DT_FLOAT16, DT_FLOAT})
    .StartNode()
    .Attr<std::string>("value")
    .Attr<int64_t>("index")
    .ComputeType(ge::ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ScalarAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ScalarAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(IndexExpr)
    .Inputs({})
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                              DT_FLOAT16, DT_FLOAT})
    .StartNode()
    .Attr<int64_t>("expr")
    .ComputeType(ge::ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::IndexExprAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::IndexExprAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Output)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                              DT_FLOAT16, DT_FLOAT})
    .Attr<int64_t>("index")
    .ComputeType(ge::ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::OutputAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::OutputAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Workspace)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64,
                              DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeInvalid)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::WorkspaceAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::WorkspaceAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64,
                                              DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(Load)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT, DT_INT64})
    .Attr<Expression>("offset")
    .ComputeType(ge::ComputeType::kComputeLoad)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LoadAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LoadAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16}}}});

REG_ASC_IR(Store)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT, DT_INT64})
    .Attr<Expression>("offset")
    .ComputeType(ge::ComputeType::kComputeStore)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::StoreAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::StoreAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16,
                                              DT_FLOAT, DT_INT64, DT_BF16}}}});

// todo: Broadcast DT_INT64 后面根据需要放开
REG_ASC_IR(Broadcast)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcBroadCastTmpSize")
    .DataType("T",
              TensorType{DT_UINT8, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64})
    .ComputeType(ge::ComputeType::kComputeBroadcast)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BroadcastAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BroadcastAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_UINT8, DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT16,
                                              DT_UINT32, DT_UINT64}}}});

REG_ASC_IR(RemovePad)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RemovePadAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RemovePadAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT,
                                              DT_BF16}}}});

REG_ASC_IR(Pad)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcPadTmpSize")
    .DataType("T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::PadAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::PadAscIrCodegenImpl>(),
           {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

// todo: Nop DT_INT64 后面根据需要放开
REG_ASC_IR(Nop)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NopAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NopAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_FLOAT16, DT_FLOAT}}}});

/* cast 暂时先放开int64->float, 以下类型, 暂不放开
 * T1:DT_INT64, DT_INT64, DT_INT64, DT_INT64,
 * T2:DT_FLOAT, DT_UINT8, DT_FLOAT16, DT_UINT64,
 */
REG_ASC_IR(Cast)
    .Input("x", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcCastTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", OrderedTensorTypeList{DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                                          DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                          DT_FLOAT16, DT_INT4,    DT_UINT8,   DT_UINT8,   DT_UINT8,   DT_UINT8,
                                          DT_UINT8,   DT_UINT8,   DT_INT8,    DT_INT8,    DT_INT16,   DT_INT16,
                                          DT_INT16,   DT_INT32,   DT_INT32,   DT_INT32,   DT_INT32,   DT_INT32,
                                          DT_INT64,   DT_BF16,    DT_BF16,    DT_UINT32,  DT_UINT16,  DT_UINT64})
    .DataType("T2", OrderedTensorTypeList{DT_FLOAT,  DT_FLOAT16, DT_INT64,   DT_INT32, DT_INT16,   DT_BF16,
                                          DT_FLOAT,  DT_INT32,   DT_INT16,   DT_INT8,  DT_UINT8,   DT_INT4,
                                          DT_INT64,  DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_INT32,   DT_INT16,
                                          DT_INT8,   DT_INT4,    DT_FLOAT16, DT_UINT8, DT_FLOAT16, DT_FLOAT,
                                          DT_UINT16, DT_FLOAT,   DT_INT64,   DT_INT16, DT_FLOAT16, DT_UINT32,
                                          DT_INT32,  DT_FLOAT,   DT_INT32,   DT_INT32, DT_INT16,   DT_INT64})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::CastAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::CastAscIrCodegenImpl>(),
           {{"T1", OrderedTensorTypeList{DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,   DT_FLOAT,
                                         DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16,
                                         DT_FLOAT16, DT_INT4,    DT_UINT8,   DT_UINT8,   DT_UINT8,   DT_UINT8,
                                         DT_UINT8,   DT_UINT8,   DT_INT8,    DT_INT8,    DT_INT16,   DT_INT16,
                                         DT_INT16,   DT_INT32,   DT_INT32,   DT_INT32,   DT_INT32,   DT_INT32,
                                         DT_INT64,   DT_BF16,    DT_BF16,    DT_UINT32,  DT_UINT16,  DT_UINT64}},
            {"T2", OrderedTensorTypeList{DT_FLOAT,  DT_FLOAT16, DT_INT64,   DT_INT32, DT_INT16,   DT_BF16,
                                         DT_FLOAT,  DT_INT32,   DT_INT16,   DT_INT8,  DT_UINT8,   DT_INT4,
                                         DT_INT64,  DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_INT32,   DT_INT16,
                                         DT_INT8,   DT_INT4,    DT_FLOAT16, DT_UINT8, DT_FLOAT16, DT_FLOAT,
                                         DT_UINT16, DT_FLOAT,   DT_INT64,   DT_INT16, DT_FLOAT16, DT_UINT32,
                                         DT_INT32,  DT_FLOAT,   DT_INT32,   DT_INT32, DT_INT16,   DT_INT64}}}});

REG_ASC_IR(Abs)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcAbsTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AbsAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AbsAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Exp)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ExpAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ExpAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Ln)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LnAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LnAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Sqrt)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SqrtAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SqrtAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Rsqrt)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcRsqrtTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RsqrtAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RsqrtAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Reciprocal)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcDefaultTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ReciprocalAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ReciprocalAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Erf)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcErfTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ErfAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ErfAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo: Sign DT_INT64 后面根据需要放开
REG_ASC_IR(Sign)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcSignTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SignAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SignAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Tanh)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcTanhTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::TanhAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::TanhAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Isnan)
    .Input("x", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcIsnanTmpSize")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT})
    .DataType("T2", TensorType{DT_UINT8})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::IsnanAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::IsnanAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(IsFinite)
    .Input("x", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcIsFiniteTmpSize")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT})
    .DataType("T2", TensorType{DT_UINT8})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::IsFiniteAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::IsFiniteAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Relu)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ReluAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ReluAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Neg)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NegAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NegAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

// todo: LogicalNot DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalNot)
    .Input("x", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcLogicalNotTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT16, DT_INT32})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LogicalNotAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LogicalNotAscIrCodegenImpl>(),
                            {{"T1", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT16, DT_INT32}},
                             {"T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT, DT_UINT8, DT_INT16, DT_INT32}}}});

REG_ASC_IR(Max)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MaxAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MaxAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Sum)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT, DT_INT32})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SumAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SumAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(Min)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MinAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MinAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Mean)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MeanAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MeanAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Prod)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ProdAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ProdAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Sigmoid)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcSigmoidTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SigmoidAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SigmoidAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Any)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AnyAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AnyAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}}});

REG_ASC_IR(All)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcReduceTmpSize")
    .DataType("T", TensorType{DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeReduce)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AllAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AllAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT}}}});

REG_ASC_IR(Add)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AddAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AddAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Sub)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcSubTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::SubAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::SubAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Div)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcDivTmpSize")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::DivAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::DivAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Mul)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MulAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Minimum)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MinimumAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MinimumAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Maximum)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::MaximumAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MaximumAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(TrueDiv)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcTrueDivTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::TrueDivAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::TrueDivAscIrCodegenImpl>(),
                            {{"T1", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT, DT_INT32}},
                             {"T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT, DT_FLOAT}}}});

REG_ASC_IR(Remainder)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcRemainderTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT, DT_INT32})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::RemainderAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::RemainderAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT, DT_INT32}}}});
// todo:LogicalOr DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalOr)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcLogicalOrTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::LogicalOrAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::LogicalOrAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8}}, {"T2", TensorType{DT_UINT8}}}});

// todo:LogicalAnd DT_INT64 后面根据需要放开
REG_ASC_IR(LogicalAnd)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcLogicalAndTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::LogicalAndAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::LogicalAndAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_INT16, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_UINT8}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Pow)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcPowTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::PowAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::PowAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT32, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(ClipByValue)
    .Input("x1", "T")
    .Input("x2", "T")
    .Input("x3", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcClipByValueTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ClipByValueAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ClipByValueAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo:Ge Eq Ne Gt Le  DT_INT64 后面根据需要放开
REG_ASC_IR(Ge)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcGeTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::GeAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::GeAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Eq)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcEqTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::EqAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::EqAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Ne)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcNeTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::NeAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::NeAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Gt)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcGtTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::GtAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::GtAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Le)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcLeTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::LeAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::LeAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_UINT8}}}});

REG_ASC_IR(Lt)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcLtTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32})
    .DataType("T2", TensorType{DT_UINT8})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LtAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LtAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT32}}, {"T2", TensorType{DT_UINT8}}}});

// todo:Concat DT_INT64 后面根据需要放开
REG_ASC_IR(Concat)
    .DynamicInput("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcConcatTmpSize")
    .ComputeType(ge::ComputeType::kComputeConcat)
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::ConcatAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::ConcatAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Select)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T2")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcSelectTmpSize")
    .DataType("T1", TensorType{DT_UINT8})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::SelectAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::SelectAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64}}}});

REG_ASC_IR(Where)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T2")
    .Output("y", "T2")
    .CalcTmpBufSize("CalcWhereTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T1", TensorType{DT_UINT8})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64})
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::WhereAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::WhereAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_UINT8}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_INT16, DT_INT32, DT_INT64}}}});

// Ub2ub是在sched阶段添加的，不需要在py构图中对外体现
// todo:Ub2ub DT_INT64 后面根据需要放开
REG_ASC_IR(Ub2ub)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::Ub2ubAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::Ub2ubAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(LeakyRelu)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .Attr<float>("negative_slope")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::LeakyReluAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::LeakyReluAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

// todo:BitwiseAnd DT_INT64 后面根据需要放开
REG_ASC_IR(BitwiseAnd)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcDefaultTmpSize")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .DataType("T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8})
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::BitwiseAndAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BitwiseAndAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT8}}}});

REG_ASC_IR(Gather)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y", "T1")
    .CalcTmpBufSize("CalcGatherTmpSize")
    .DataType("T1", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT})
    .DataType("T2", TensorType{DT_INT32, DT_INT64})
    .Attr<int64_t>("axis")
    .Attr<bool>("negative_index_support")
    .ComputeType(ge::ComputeType::kComputeGather)
    .Impl(v1_soc_versions,
          {ge::ascir::AscIrImplCreator<ge::ascir::GatherAscIrAttImpl>(),
           ge::ascir::AscIrImplCreator<ge::ascir::GatherAscIrCodegenImpl>(),
           {{"T1", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT}},
            {"T2", TensorType{DT_INT32, DT_INT64}}}});

REG_ASC_IR(Transpose)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT})
    .ApiTilingDataType("ConfusionTransposeTiling")
    .ComputeType(ge::ComputeType::kComputeTranspose)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::TransposeAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::TransposeAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FLOAT16, DT_FLOAT}}}});

// todo:目前前端dt构图用到了FlashSoftmax，暂时无法删除
REG_ASC_IR(FlashSoftmax)
    .Inputs({"x1", "x2", "x3"})
    .Outputs({"y1", "y2", "y3"})
    .UseFirstInputDataType()
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl({}, {ge::ascir::AscIrImplCreator<ge::ascir::AbsAscIrAttImpl>(),
               ge::ascir::AscIrImplCreator<ge::ascir::AscIrCodegen>(),
               {{"T1", TensorType{DT_INT8, DT_INT16}}, {"T2", TensorType{DT_UINT8, DT_INT16}}}});

REG_ASC_IR(FloorDiv)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("GetInputDataSizeTmpBuffer")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::FloorDivAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::FloorDivAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Gelu)
    .Input("x", "T")
    .Output("y", "T")
    .CalcTmpBufSize("GetInputDataSizeTmpBuffer")
    .DataType("T", TensorType{DT_FLOAT16, DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::GeluAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::GeluAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(Axpy)
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .CalcTmpBufSize("CalcAxpyTmpSize")
    .Attr<float>("alpha")
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<ge::ascir::AxpyAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::AxpyAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});
REG_ASC_IR(MatMul)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(MatMulOffset)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T3", TensorType{DT_INT8, DT_INT4})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(MatMulOffsetBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T3", TensorType{DT_INT8, DT_INT4})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("transpose_x1")
    .Attr<int64_t>("transpose_x2")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::MatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMul)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});

REG_ASC_IR(BatchMatMulBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}}}});


REG_ASC_IR(BatchMatMulOffset)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T3", TensorType{DT_INT8, DT_INT4})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(BatchMatMulOffsetBias)
    .Input("x1", "T1")
    .Input("x2", "T1")
    .Input("bias", "T2")
    .Input("offset_w", "T3")
    .Output("y", "T2")
    .DataType("T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16})
    .DataType("T3", TensorType{DT_INT8, DT_INT4})
    .Attr<int64_t>("offset_x")
    .Attr<int64_t>("has_relu")
    .Attr<int64_t>("enable_hf32")
    .Attr<int64_t>("adj_x1")
    .Attr<int64_t>("adj_x2")
    .ComputeType(ge::ComputeType::kComputeCube)
    .Impl(v1_soc_versions, {ge::ascir::AscIrImplCreator<MatMulAscIrAttImpl>(),
                            ge::ascir::AscIrImplCreator<ge::ascir::BatchMatMulAscIrCodegenImpl>(),
                            {{"T1", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_BF16}},
                             {"T3", TensorType{DT_INT8, DT_INT4}}}});

REG_ASC_IR(Split)
    .Input("x", "T")
    .DynamicOutput("y", "T")
    .Attr<int64_t>("index")
    .Attr<int64_t>("gid")  // global_id, SplitOp的全局编号
    .DataType("T",
              TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT});
}  // namespace ascir
}  // namespace ge
