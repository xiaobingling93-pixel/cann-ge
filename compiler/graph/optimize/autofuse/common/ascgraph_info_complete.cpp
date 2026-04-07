/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascgraph_info_complete.h"
#include <map>
#include <queue>
#include "ascir_ops.h"
#include "ascendc_ir_def.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "ascir_ops_utils.h"

using namespace ge::ascir_op;

namespace optimize {
namespace {
static Status GetNodeIrAttrOffset(const ge::NodePtr &node, ge::Expression &offset) {
  auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
  GE_ASSERT_NOTNULL(asc_node);
  GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
  return asc_node->attr.ir_attr->GetAttrValue("offset", offset);
}

void InsertFreeSymbolsIntoVarSet(const ge::Expression &exp, SizeVarSet &size_vars) {
  std::vector<ge::Expression> free_symbols = exp.FreeSymbols();
  size_vars.insert(free_symbols.begin(), free_symbols.end());
}

void CompleteDataApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeBuffer;
  node->attr.api.unit = ge::ComputeUnit::kUnitNone;
}

void CompleteLoadApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
}

void CompleteStoreApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
}

void CompleteElewiseApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteBroadcastApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteReduceApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteConcatApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteSplitApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteGatherApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
}

void CompleteCubeApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitCube;
}
}  // namespace

using CompleteApiInfoFunc = std::function<void(ge::AscNodePtr &)>;
struct Completer {
  CompleteApiInfoFunc complete_api_info;
};

void CompleteTransposeApiInfo(ge::AscNodePtr &node) {
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

static const std::map<std::string, ge::ComputeType> kOpTypeToComputeType = {
    {Workspace::Type, ge::ComputeType::kComputeInvalid},      {Data::Type, ge::ComputeType::kComputeInvalid},
    {Scalar::Type, ge::ComputeType::kComputeInvalid},         {Output::Type, ge::ComputeType::kComputeInvalid},
    {IndexExpr::Type, ge::ComputeType::kComputeInvalid},

    {Load::Type, ge::ComputeType::kComputeLoad},           {Store::Type, ge::ComputeType::kComputeStore},

    {Sum::Type, ge::ComputeType::kComputeReduce},          {Max::Type, ge::ComputeType::kComputeReduce},
    {ArgMax::Type, ge::ComputeType::kComputeReduce},       {Mean::Type, ge::ComputeType::kComputeReduce},         {Min::Type, ge::ComputeType::kComputeReduce},
    {Prod::Type, ge::ComputeType::kComputeReduce},         {All::Type, ge::ComputeType::kComputeReduce},
    {Any::Type, ge::ComputeType::kComputeReduce},

    {Broadcast::Type, ge::ComputeType::kComputeBroadcast},
    {RemovePad::Type, ge::ComputeType::kComputeElewise},
    {Pad::Type, ge::ComputeType::kComputeElewise},
    {Round::Type, ge::ComputeType::kComputeElewise},

    {Cast::Type, ge::ComputeType::kComputeElewise},        {Abs::Type, ge::ComputeType::kComputeElewise},
    {Neg::Type, ge::ComputeType::kComputeElewise},         {Exp::Type, ge::ComputeType::kComputeElewise},
    {Sqrt::Type, ge::ComputeType::kComputeElewise},        {Rsqrt::Type, ge::ComputeType::kComputeElewise},
    {Relu::Type, ge::ComputeType::kComputeElewise},        {Reciprocal::Type, ge::ComputeType::kComputeElewise},
    {Erf::Type, ge::ComputeType::kComputeElewise},         {Sign::Type, ge::ComputeType::kComputeElewise},
    {Tanh::Type, ge::ComputeType::kComputeElewise},        {Isnan::Type, ge::ComputeType::kComputeElewise},
    {IsFinite::Type, ge::ComputeType::kComputeElewise},    {Ln::Type, ge::ComputeType::kComputeElewise},
    {LogicalNot::Type, ge::ComputeType::kComputeElewise},

    {Add::Type, ge::ComputeType::kComputeElewise},         {Sub::Type, ge::ComputeType::kComputeElewise},
    {Mul::Type, ge::ComputeType::kComputeElewise},         {Div::Type, ge::ComputeType::kComputeElewise},
    {TrueDiv::Type, ge::ComputeType::kComputeElewise},     {Minimum::Type, ge::ComputeType::kComputeElewise},
    {Maximum::Type, ge::ComputeType::kComputeElewise},     {LogicalOr::Type, ge::ComputeType::kComputeElewise},
    {LogicalAnd::Type, ge::ComputeType::kComputeElewise},

    {Ge::Type, ge::ComputeType::kComputeElewise},          {Eq::Type, ge::ComputeType::kComputeElewise},
    {Ne::Type, ge::ComputeType::kComputeElewise},          {Gt::Type, ge::ComputeType::kComputeElewise},
    {Le::Type, ge::ComputeType::kComputeElewise},          {Lt::Type, ge::ComputeType::kComputeElewise},
    {Broadcast::Type, ge::ComputeType::kComputeElewise},   {Sigmoid::Type, ge::ComputeType::kComputeElewise},
    {Concat::Type, ge::ComputeType::kComputeConcat},       {Gather::Type, ge::ComputeType::kComputeGather},

    {Where::Type, ge::ComputeType::kComputeElewise},       {Select::Type, ge::ComputeType::kComputeElewise},
    {ClipByValue::Type, ge::ComputeType::kComputeElewise}, {Pow::Type, ge::ComputeType::kComputeElewise},
    {Transpose::Type, ge::ComputeType::kComputeTranspose},
    {BitwiseAnd::Type, ge::ComputeType::kComputeElewise},  {LeakyRelu::Type, ge::ComputeType::kComputeElewise},
    {FloorDiv::Type, ge::ComputeType::kComputeElewise},    {Gelu::Type, ge::ComputeType::kComputeElewise},
    {Axpy::Type, ge::ComputeType::kComputeElewise},
    {Split::Type, ge::ComputeType::kComputeSplit},
    {MatMul::Type, ge::ComputeType::kComputeCube},         {MatMulBias::Type, ge::ComputeType::kComputeCube},
    {MatMulOffset::Type, ge::ComputeType::kComputeCube},   {MatMulOffsetBias::Type, ge::ComputeType::kComputeCube},
    {BatchMatMul::Type, ge::ComputeType::kComputeCube},    {BatchMatMulBias::Type, ge::ComputeType::kComputeCube},
    {BatchMatMulOffset::Type, ge::ComputeType::kComputeCube},
    {BatchMatMulOffsetBias::Type, ge::ComputeType::kComputeCube},
};

static const std::map<ge::ComputeType, Completer> kComputeTypeToCompleter = {
    {ge::ComputeType::kComputeInvalid, {&CompleteDataApiInfo}},
    {ge::ComputeType::kComputeLoad, {&CompleteLoadApiInfo}},
    {ge::ComputeType::kComputeStore, {&CompleteStoreApiInfo}},
    {ge::ComputeType::kComputeReduce, {&CompleteReduceApiInfo}},
    {ge::ComputeType::kComputeBroadcast, {&CompleteBroadcastApiInfo}},
    {ge::ComputeType::kComputeElewise, {&CompleteElewiseApiInfo}},
    {ge::ComputeType::kComputeConcat, {&CompleteConcatApiInfo}},
    {ge::ComputeType::kComputeGather, {&CompleteGatherApiInfo}},
    {ge::ComputeType::kComputeTranspose, {&CompleteTransposeApiInfo}},
    {ge::ComputeType::kComputeSplit, {&CompleteSplitApiInfo}},
    {ge::ComputeType::kComputeCube, {&CompleteCubeApiInfo}},
};

Status AscGraphInfoComplete::CompleteApiInfo(const ge::AscGraph &optimize_graph) {
  for (auto node : optimize_graph.GetAllNodes()) {
    auto node_compute_type = &node->attr.api.compute_type;
    if (*node_compute_type >= ge::ComputeType::kComputeInvalid) {
      auto item = kOpTypeToComputeType.find(node->GetType());
      GE_ASSERT_TRUE((item != kOpTypeToComputeType.end()), "Failed get node compute type, node name:[%s], type: [%s].",
                     node->GetNamePtr(), node->GetTypePtr());
      *node_compute_type = item->second;
    }
    auto it = kComputeTypeToCompleter.find(*node_compute_type);
    GE_ASSERT_TRUE((it != kComputeTypeToCompleter.end()), "CompleteApiInfo unsupported node name:[%s], type: [%s].",
                   node->GetNamePtr(), node->GetTypePtr());
    it->second.complete_api_info(node);
  }
  return ge::SUCCESS;
}

void AscGraphInfoComplete::AppendOriginalSizeVar(const ge::AscGraph &graph, SizeVarSet &size_vars) {
  auto axes = graph.GetAllAxis();
  for (const auto &axis : axes) {
    InsertFreeSymbolsIntoVarSet(axis->size, size_vars);
  }
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!ge::ops::IsOps<Nddma>(node) && !ge::ops::IsOps<Store>(node) && !ge::ops::IsOps<Load>(node) &&
        !ge::ops::IsOps<Gather>(node)) {
      continue;
    }

    ge::Expression cur_load_offset;
    if (GetNodeIrAttrOffset(node, cur_load_offset) == ge::SUCCESS) {
      InsertFreeSymbolsIntoVarSet(cur_load_offset, size_vars);
    }

    if (ge::ops::IsOps<Gather>(node)) {
      for (const auto &exp : node->inputs[0].attr.repeats) {
        InsertFreeSymbolsIntoVarSet(exp, size_vars);
      }
    }

    for (const auto &exp : node->outputs[0].attr.repeats) {
      InsertFreeSymbolsIntoVarSet(exp, size_vars);
    }
    for (const auto &exp : node->outputs[0].attr.strides) {
      InsertFreeSymbolsIntoVarSet(exp, size_vars);
    }
  }
}
}  // namespace optimize
