/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>

#include "compiler/graph/passes/feature/algebraic_simplification_pass.h"
#include "es_ge_test_ops.h"
#include "graph_utils_ex.h"
#include "tensor_adapter.h"
#include "common/types.h"
#include "graph/operator_reg.h"

namespace ge {
class AlgebraicSimplificationPassTest : public testing::Test {
 protected:
  template <typename T>
  static EsCTensorHolder* CreateConst(es::EsGraphBuilder &graph_builder, ge::DataType dtype, const std::vector<int64_t> &dims,
                                const std::vector<T> &value) {
    EsCGraphBuilder* esb_graph = graph_builder.GetCGraphBuilder();
    return ge::es::EsCreateConst<T>(esb_graph, value.data(), dims.data(), static_cast<int64_t>(dims.size()), dtype, FORMAT_ND);
  }

  template <typename T>
  static ComputeGraphPtr BuildGraph(const std::string &op_type, DataType dtype,
                                    T value, bool lhs_is_data,
                                    bool ref_const = false,
                                    const std::vector<int64_t> &const_shape = {8, 16},
                                    const std::vector<int64_t> &data_shape = {8, 16}) {
    GeTensorDesc const_desc(GeShape(const_shape), FORMAT_ND, dtype);
    std::vector<T> buffer(GeShape(const_shape).GetShapeSize(), value);
    ge::es::EsGraphBuilder es_graph("graph");
    {
      auto data_0 = EsCreateGraphInput(es_graph.GetCGraphBuilder(), 0);
      auto abs_0 = EsAbs(data_0);
      auto const_0 = es::EsTensorHolder(CreateConst(es_graph, dtype, GeShape(const_shape).GetDims(), buffer));
      const ge::es::EsTensorHolder &lhs = lhs_is_data ? abs_0 : const_0;
      const ge::es::EsTensorHolder &rhs = lhs_is_data ? const_0 : abs_0;
      if (op_type == ADD) {
        const auto out_0 = es::Add(lhs, rhs);
        es::EsGraphBuilder::SetOutput(out_0, 0);
      } else if (op_type == MUL) {
        const auto out_0 = es::Mul(lhs, rhs);
        es::EsGraphBuilder::SetOutput(out_0, 0);
      } else if (op_type == SUB) {
        const auto out_0 = es::Sub(lhs, rhs);
        es::EsGraphBuilder::SetOutput(out_0, 0);
      } else if (op_type == "Div") {
        const auto out_0 = es::Div(lhs, rhs);
        es::EsGraphBuilder::SetOutput(out_0, 0);
      }
      if (ref_const) {
        es::EsGraphBuilder::SetOutput(const_0, 1);
      }
    }
    const auto test_graph = es_graph.BuildAndReset();
    const auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
    std::vector<int64_t> output_shape(data_shape.size());
    for (size_t i = 0; i < data_shape.size(); ++i) {
      output_shape[i] = (data_shape[i] != 1 ? data_shape[i] : const_shape[i]);
    }
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == op_type) {
        const auto &lhs_shape = lhs_is_data ? data_shape : const_shape;
        const auto &rhs_shape = lhs_is_data ? const_shape : data_shape;
        node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(lhs_shape));
        node->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape(rhs_shape));
        node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(output_shape));
      }
    }
    return graph;
  }
};

namespace {
REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Mul)

REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Add)

REG_OP(Div)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Div)

REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sub)

REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)

REG_OP(BroadcastTo)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                      DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(BroadcastTo)
}  // namespace

TEST_F(AlgebraicSimplificationPassTest, HandleAdd) {
  // A + 0 -> A
  {
    const auto graph = BuildGraph(ADD, DT_INT32, 0, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 0 + A -> A
  {
    const auto graph = BuildGraph<int64_t>(ADD, DT_INT64, 0, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // A + 1 -> A + 1
  {
    const auto graph = BuildGraph(ADD, DT_INT32, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleSub) {
  // A - 0 -> A
  {
    const auto graph = BuildGraph<int16_t>(SUB, DT_INT16, 0, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 0 - A -> 0 - A
  {
    const auto graph = BuildGraph<int8_t>(SUB, DT_INT8, 0, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
  // A - 1 -> A - 1
  {
    const auto graph = BuildGraph(SUB, DT_INT32, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleMul) {
  // A * 1 -> A
  {
    const auto graph = BuildGraph<int8_t>(MUL, DT_INT8, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // A * 1 -> A
  {
    const auto graph = BuildGraph(MUL, DT_DOUBLE, 1.0, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 1 * A -> A
  {
    const auto graph = BuildGraph<uint16_t>(MUL, DT_FLOAT16, 15360, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // A * 1.1 -> A * 1.1
  {
    const auto graph = BuildGraph(ADD, DT_FLOAT, 1.1f, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleMul_WithBrc) {
  // 1 * A -> A
  {
    const auto graph = BuildGraph<uint16_t>(MUL, DT_BF16, 16256, false, false, {8, 16}, {1, 1});
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
    EXPECT_TRUE(graph->FindFirstNodeMatchType("BroadcastTo") != nullptr);
  }
  {
    const auto graph = BuildGraph<uint16_t>(MUL, DT_BF16, 16256, false, false, {1, 1}, {8, 16});
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  {
    const auto graph = BuildGraph<uint16_t>(MUL, DT_BF16, 16256, false, false, {8, 1}, {1, 16});
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
    EXPECT_TRUE(graph->FindFirstNodeMatchType("BroadcastTo") != nullptr);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleDiv) {
  // A / 1 -> A
  {
    const auto graph = BuildGraph<int16_t>("Div", DT_INT16, 1, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 3);
  }
  // 1 / A -> 1 / A
  {
    const auto graph = BuildGraph<int8_t>("Div", DT_INT8, 1, false);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
  // A / 2-> A / 2
  {
    const auto graph = BuildGraph("Div", DT_INT32, 2, true);
    EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
  }
}

TEST_F(AlgebraicSimplificationPassTest, HandleUnsupportedDtype) {
  const auto graph = BuildGraph(ADD, DT_BOOL, 0, true);
  EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
}

TEST_F(AlgebraicSimplificationPassTest, HandleAdd_ConstOutputMultiRef) {
  // A + 0 -> A
  const auto graph = BuildGraph(ADD, DT_INT32, 0, true, true);
  EXPECT_EQ(AlgebraicSimplificationPass::Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
}
}  // namespace ge
