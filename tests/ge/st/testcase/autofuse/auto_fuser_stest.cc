/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <gtest/gtest.h>
#include "faker/space_registry_faker.h"
#include <map>

#include "compliant_node_builder.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/utils/graph_utils.h"
#include "utils/autofuse_attrs.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "compiler/graph/passes/feature/auto_fuse_pass.h"
#include "graph/ge_local_context.h"
#include "dlog_pub.h"
#include "graph_utils_ex.h"
#include "framework/ge_runtime_stub/include/common/env_path.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "tests/framework/ge_runtime_stub/include/common/topo_checker.h"
#include "tests/framework/ge_runtime_stub/include/common/summary_checker.h"
#include "tests/framework/ge_runtime_stub/include/faker/space_registry_faker.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/ge_runtime_stub/include/common/compliant_share_graph.h"
#include "attribute_group/attr_group_shape_env.h"
#include "es_ge_test_ops_c.h"
#include "es_ge_test_ops.h"

namespace ge {
class AutofuserUT : public testing::Test {
  void SetUp() override {
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  }
  void TearDown() override {
    unsetenv("AUTOFUSE_FLAGS");
  }
};

namespace {
REG_OP(BroadcastTo)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8, DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                      DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(BroadcastTo)

template <typename T>
ComputeGraphPtr BuildGraph(const std::string &op_type, DataType dtype, T value,
                           bool lhs_is_data, bool ref_const = false) {
  std::vector<int64_t> const_shape = {8, 16};
  std::vector<int64_t> data_shape = {8, 1};
  GeTensorDesc const_desc(GeShape(const_shape), FORMAT_ND, dtype);
  std::vector<T> buffer(GeShape(const_shape).GetShapeSize(), value);
  es::EsGraphBuilder es_graph("graph");
  {
    auto data_0 = es_graph.CreateInput(0, "data_0", nullptr);
    data_0.SetShape(data_shape);
    auto abs_0 = es::Abs(data_0);
    auto const_0 = es_graph.CreateConst(buffer, GeShape(const_shape).GetDims(), dtype);
    const_0.SetShape(const_shape);
    const es::EsTensorHolder &lhs = lhs_is_data ? abs_0 : const_0;
    const es::EsTensorHolder &rhs = lhs_is_data ? const_0 : abs_0;
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
}  // namespace

/*
 * 普通的fuse流程
 */
//TEST_F(AutofuserUT, AddFuse) {
//  const auto graph = cg::BuildAbsAddReluReluGraph({4, 5, 6});
//  ASSERT_NE(graph, nullptr);
//  std::vector<GeTensor> inputs;
//  GeTensorDesc td;
//  td.SetShape((GeShape({4, 5, 6})));
//  td.SetOriginShape((GeShape({4, 5, 6})));
//  inputs.emplace_back(td);
//
//  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, inputs), SUCCESS);
//  AutoFusePass pass;
//  EXPECT_EQ(pass.Run(graph), ge::SUCCESS);
//}

TEST_F(AutofuserUT, Symbolic_AllConstDimValue_AllConstSymbol) {
  const auto graph = cg::BuildAddGraph({-1, -1, -1}, {-1, -1, -1});
  ASSERT_NE(graph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, inputs), SUCCESS);
  SymbolicShapeInference symbolic_shape_inference;
  ASSERT_EQ(symbolic_shape_inference.Infer(graph), SUCCESS);

  auto data0 = graph->FindNode("input_0");
  ASSERT_NE(data0, nullptr);
  auto attr0 = data0->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  auto input_symbol_shape0 = attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(input_symbol_shape0.GetDimNum(), 3);
  EXPECT_EQ(std::string(input_symbol_shape0.GetDim(0).Serialize().get()), "s3");
  EXPECT_EQ(std::string(input_symbol_shape0.GetDim(1).Serialize().get()), "s4");
  EXPECT_EQ(std::string(input_symbol_shape0.GetDim(2).Serialize().get()), "s5");

  auto data1 = graph->FindNode("input_1");
  ASSERT_NE(data1, nullptr);
  auto attr1 = data1->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto input_symbol_shape1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(input_symbol_shape1.GetDimNum(), 3);
  EXPECT_EQ(std::string(input_symbol_shape1.GetDim(0).Serialize().get()), "s3");
  EXPECT_EQ(std::string(input_symbol_shape1.GetDim(1).Serialize().get()), "s4");
  EXPECT_EQ(std::string(input_symbol_shape1.GetDim(2).Serialize().get()), "s5");
}

TEST_F(AutofuserUT, Symbolic_DynamicShapeTest_InputIsEmpty_Fail) {
  const auto graph = cg::BuildAddGraph({-1, -1, -1}, {1, -1, 3});
  ASSERT_NE(graph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape()));
  td.SetOriginShape((GeShape()));
  inputs.emplace_back(td);
  inputs.emplace_back(td);
  ASSERT_NE(SymbolicShapeSymbolizer::Symbolize(graph, inputs), SUCCESS); // test input not match
}

TEST_F(AutofuserUT, PreProcess_Success) {
  {
    AutoFusePass pass;
    const auto graph = BuildGraph(ADD, DT_INT32, 0, true, true);
    EXPECT_EQ(pass.Run(graph), SUCCESS);
  }
  {
    AutoFusePass pass;
    const auto graph = BuildGraph(ADD, DT_BF16, 0, true, true);
    EXPECT_EQ(pass.Run(graph), SUCCESS);
  }
}
} // namespace ge