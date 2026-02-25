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

#include <memory>
#include <gtest/gtest.h>
#include <dlfcn.h>
#include "graph/utils/graph_utils_ex.h"
#include "es_ge_test_ops_c.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "compiler/graph/passes/feature/auto_fuse_pass.h"
#include "framework/common/types.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "common/env_path.h"
#include "common/topo_checker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/omg_util.h"
#include "mmpa/mmpa_api.h"
#include "graph/node.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h"
#include "graph/operator_reg.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_post_processor.h"

namespace ge {
namespace {
graphStatus InferShape4FooTest(gert::InferSymbolShapeContext *context) {
  auto shape0 = context->GetInputSymbolShape(0);
  GE_ASSERT_NOTNULL(shape0);
  auto shape1 = context->GetInputSymbolShape(1);
  GE_ASSERT_NOTNULL(shape1);
  GE_ASSERT_TRUE(shape0->GetDimNum() == shape1->GetDimNum());
  for (size_t i = 0U; i < shape0->GetDimNum(); ++i) {
    if (EXPECT_SYMBOL_EQ(shape0->GetDim(i), shape1->GetDim(i))) {
      context->GetOutputSymbolShape(0)->AppendDim(shape0->GetDim(i));
    } else if (EXPECT_SYMBOL_EQ(shape0->GetDim(i), Symbol(1))) {
      context->GetOutputSymbolShape(0)->AppendDim(shape1->GetDim(i));
    } else if (EXPECT_SYMBOL_EQ(shape1->GetDim(i), Symbol(1))) {
      context->GetOutputSymbolShape(0)->AppendDim(shape0->GetDim(i));
    } else {
      return GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FooTest).InferSymbolShape(InferShape4FooTest);
} // namespace
class SymbolicInfoPostProcessorUT : public testing::Test {
 public:
  void SetUp() override {
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  }

  void TearDown() override {
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("LD_LIBRARY_PATH");
  }
};

TEST_F(SymbolicInfoPostProcessorUT, run_test) {
  auto data1 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 0)
      .InCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .Build("data1");
  auto data2 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 1)
      .InCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .Build("data2");
  auto foo1 = OP_CFG("FooTest")
      .InCnt(2)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1})
      .Build("foo1");

  DEF_GRAPH(g_test_guard) {
    CHAIN(NODE(data1)->NODE(foo1)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data2)->NODE(foo1)->NODE("NetOutput", "NetOutput"));
  };
  auto graph = ToComputeGraph(g_test_guard);
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(shape_env_attr, nullptr);
  ShapeEnvGuarder guard(shape_env_attr);
  auto s0 = shape_env_attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto s1 = shape_env_attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto s2 = shape_env_attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 0));
  auto s3 = shape_env_attr->CreateSymbol(7, MakeShared<InputShapeSource>(1, 1));

  auto foo1_node = graph->FindNode("foo1");
  auto op_desc = foo1_node->GetOpDesc();
  auto sym_attr0 = op_desc->MutableOutputDesc(0U)->template GetOrCreateAttrsGroup<SymbolicDescAttr>();
  sym_attr0->symbolic_tensor.MutableOriginSymbolShape() = {s0, s1, s2, s3};
  auto sym_attr1 = op_desc->MutableOutputDesc(1U)->template GetOrCreateAttrsGroup<SymbolicDescAttr>();
  sym_attr1->symbolic_tensor.MutableOriginSymbolShape() = {s0, s1, s2, s3};

  ASSERT_EQ(SymbolicInfoPostProcessor::Run(graph), SUCCESS);

  std::string infer_key;
  (void)AttrUtils::GetStr(op_desc, "_symbol_infer_shape_merge_key", infer_key);
  ASSERT_EQ(infer_key, "[s0_s1_s2_s3][s0_s1_s2_s3]");

  size_t all_sym_num;
  (void)AttrUtils::GetInt(graph, "_all_symbol_num", all_sym_num);
  ASSERT_EQ(all_sym_num, 4);

  std::string buffer;
  AttrUtils::GetStr(graph, "_guard_check_so_data", buffer);
  ASSERT_TRUE(!buffer.empty());
}
} // namespac ge