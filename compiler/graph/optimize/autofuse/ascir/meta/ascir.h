/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCIR_H__
#define __ASCIR_H__

#include <cstdint>
#include <memory>
#include <vector>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/symbolizer/symbolic.h"

namespace ascir {
  using Graph = ge::AscGraph;
  using HintGraph = ge::AscGraph;
  using ImplGraph = ge::AscGraph;
  using NodeView = ge::AscNodePtr;
  using NodeViewVisitorConst = ge::AscNodeVisitor;
  using AxisId = ge::AxisId;
  using Axis = ge::Axis;
  using TensorAttr = ge::AscTensor;
  using SizeExpr = ge::Expression;
  //using TensorView = ge::AscTensorAttr*;
  using TensorView = ge::AscTensor;
  using TensorPtr = ge::AscTensorAttr*;
  using SizeVar = ge::SizeVar;

  // enum
  using ComputeType = ge::ComputeType;
  using ComputeUnit = ge::ComputeUnit;
  using ApiType = ge::ApiType;
  using AllocType = ge::AllocType;
  using MemHardware = ge::MemHardware;
  using Position = ge::Position;

  // id  int64_t
  using Identifier = int64_t;
  using TensorId = Identifier;
  using BufId = Identifier;
  using QueId = Identifier;
  using MergeScopeId = Identifier;
  using ReuseId = Identifier;

}

#endif