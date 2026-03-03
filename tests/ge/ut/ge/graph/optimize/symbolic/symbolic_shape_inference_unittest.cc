/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <utility>
#include <gtest/gtest.h>
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "es_ge_test_ops_c.h"
#include "es_ge_test_ops.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_pre_processor.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "attribute_group/attr_group_shape_env.h"
#include "framework/common/types.h"
#include "faker/space_registry_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/operator_reg.h"

#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/autofuse/autofuse_optimize.h"
#include "common/env_path.h"
#include "mmpa/mmpa_api.h"
#include "ge_local_context.h"
#include "register/optimization_option_registry.h"
#include "ge_types.h"
#include "expect_node_info_check_test.h"
#include "ge_running_env/op_reg.h"
#include "common/share_graph.h"
#include "api/aclgrph/option_utils.h"
#include "common/context/local_context.h"

namespace ge {
bool EnableSliceSchedule() { // 桩函数
  return true;
}
class SymbolicShapeInferenceUT : public testing::Test {
 public:
 protected:
  static void SetUpTestSuite() {
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  }
  static void TearDownTestSuite() {

  }
  void SetUp() override {
    global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();

    std::map<std::string, std::string> tmp_global_option;
    tmp_global_option.insert(std::make_pair(ge::OPTION_TOPOSORTING_MODE, "0"));
    ge::GetThreadLocalContext().SetGlobalOption(tmp_global_option);
    GetThreadLocalContext().GetOo().Initialize(GetThreadLocalContext().GetAllOptions(),
                                               OptionRegistry::GetInstance().GetRegisteredOptTable());
    graph_ = EsCreateGraphBuilder("Hello");
  }
  void TearDown() override {
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
    GetThreadLocalContext().GetOo().Initialize(GetThreadLocalContext().GetAllOptions(),
                                               OptionRegistry::GetInstance().GetRegisteredOptTable());
    EsDestroyGraphBuilder(graph_);
    graph_ = nullptr;
  }
  EsCGraphBuilder *graph_{nullptr};
 private:
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
};

/**
 *      Data0     Data1
 *        |  \     /
 *        |   Add
 *        |   /
 *         Mul
 *          |
 *         Relu
 *          |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForSimpleGraph) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  auto add = EsAdd(data0, data1);
  auto mul = EsMul(data0, add);
  auto relu = EsRelu(mul);
  ASSERT_EQ(EsSetGraphOutput(relu, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT32, {-1, -1, -1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 1, -1};
  SetNoStorage(cg, "data1", di, 1);
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  std::vector<Expression> expect_dims = {Symbol("s0"), Symbol("s4"), Symbol("s2"), Symbol("s5")};
  ExpectNodeInfo expect_node("Relu_2", expect_dims, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0     Const(scalar)
 *         \     /
 *        ReduceSum    Const(list_int)
 *              \       /
 *              ReduceMax
 *                  |
 *              NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForGetConstInput) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto const_scalar_int32 = EsCreateScalarInt32(graph_, 1);
  std::vector<int64_t> dims{0, 1, 2};
  int64_t dims_size = 3;
  auto const_int64_list = EsCreateConstInt64(graph_, dims.data(), &dims_size, 1);

  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s2", "s1", "s0", "s0"}).data(), 4), 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(const_scalar_int32, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto reduce_sum = EsReduceSum(data0, const_scalar_int32, true, true);
  ge::TensorDesc reduce_sum_desc;
  reduce_sum->GetProducer().GetInputDesc(1, reduce_sum_desc);
  reduce_sum_desc.SetDataType(DT_INT32);
  reduce_sum->GetProducer().UpdateInputDesc(1, reduce_sum_desc);

  auto reduce_max = EsReduceMax(reduce_sum, const_int64_list, true, true);
  ge::TensorDesc reduce_max_desc;
  reduce_max->GetProducer().GetInputDesc(1, reduce_max_desc);
  reduce_max_desc.SetDataType(DT_INT64);
  reduce_max->GetProducer().UpdateInputDesc(1, reduce_max_desc);

  ASSERT_EQ(EsSetGraphOutput(reduce_max, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  ExpectNodeInfo expect_node1("ReduceSum", {s2, one, s0, s0}, {}, {}, {});
  ExpectNodeInfo expect_node2("ReduceMax", {one, one, one, s0}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  expect_node_vec.push_back(expect_node2);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

ge::graphStatus InferShapeDoNothing(gert::InferShapeContext *context) {
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(ReduceSum).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Const).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Sub).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(ReduceMax).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Exp).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Div).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Cast).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Data).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();

/**
 *      Data0     Const(scalar)
 *         \     /
 *        ReduceSum    Const(list_int)
 *              \       /
 *              ReduceMax
 *                  |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForConst) {
  auto es_graph = std::unique_ptr<es::EsGraphBuilder>(new es::EsGraphBuilder("cpp_graph"));
  auto data0 = es_graph->CreateInput(0, "data0");
  data0.SetOriginSymbolShape({"s0", "s1", "s2", "s3"});
  auto const_scalar = es_graph->CreateScalar(1);
  auto const_list = es_graph->CreateVector(std::vector<int64_t>{0, 1});
  auto reduce_sum = es::ReduceSum(data0, const_scalar, true);
  // 在infer symbol时，非decompose场景，图上节点的dtype在前面的infer shape流程处理过，ut时需要显示设置
  // decompose图中，infer symbol代码里面会调用infer dtype流程，因此要保证完成了decompose图中算子的infer dtype注册
  ge::TensorDesc reduce_sum_input_desc;
  reduce_sum.GetProducer()->GetInputDesc(1, reduce_sum_input_desc);
  reduce_sum_input_desc.SetDataType(DT_INT32);
  reduce_sum.GetProducer()->UpdateInputDesc(1, reduce_sum_input_desc);
  auto reduce_max = es::ReduceMax(reduce_sum, const_list, true);
  ge::TensorDesc reduce_max_input_desc;
  reduce_max.GetProducer()->GetInputDesc(1, reduce_max_input_desc);
  reduce_max_input_desc.SetDataType(DT_INT64);
  reduce_max.GetProducer()->UpdateInputDesc(1, reduce_max_input_desc);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_max, 0), 0);
  auto graph = es_graph->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ExpectNodeInfo expect_node1("Const_0", {}, {}, {}, {});
  ExpectNodeInfo expect_node2("Const_1",{ge::Symbol(2)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  expect_node_vec.push_back(expect_node2);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, InferShapeForVariable) {
  auto es_graph = std::unique_ptr<es::EsGraphBuilder>(new es::EsGraphBuilder("cpp_graph"));
  std::vector<int64_t> dims{4, 5, 6};
  auto variable = es_graph->CreateVariable(0, "variable_0");
  auto relu = es::Relu(variable);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = es_graph->BuildAndReset();
  ASSERT_NE(variable.GetProducer(), nullptr);
  ge::TensorDesc variable_input_desc;
  ASSERT_EQ(variable.GetProducer()->GetInputDesc(0, variable_input_desc), ge::GRAPH_SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto var_node = cg->FindNode("variable_0");
  if (var_node != nullptr) {
    auto op_desc = var_node->GetOpDesc();
    if (op_desc != nullptr && op_desc->GetOutputsSize() > 0) {
      op_desc->MutableOutputDesc(0)->SetShape(GeShape({4, 5, 6}));
      op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({4, 5, 6}));
    }
  }
  ExpectNodeInfo expect_node("variable_0", {Symbol(4), Symbol(5), Symbol(6)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

/*
 * 如果未实现infer symbol shape，分两种情况：
 * 1. 输出为静态shape，则使用输出shape构造常量symbol继续推导
 * 2. 输出为动态shape，则报错
 * 后续该点需要废弃，当前属于静态shape网络的规避处理
 * */
namespace {
auto data = OP_CFG("Data")
                .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                .InCnt(0)
                .OutCnt(1)
                .InNames({"x"})
                .OutNames({"y"})
                .Build("Data");
auto foo1 =
    OP_CFG("foo1").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2, 3, 4}).InCnt(1).OutCnt(1).OutNames({"y"}).Build("FOO1");
auto foo2 = OP_CFG("foo2")
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1, 3, 4})
                .InCnt(1)
                .OutCnt(1)
                .InNames({"x"})
                .OutNames({"y"})
                .Build("FOO2");
}  // namespace

namespace {
void SetBatchMatMulV2IrAttrs(const ComputeGraphPtr &cg) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "BatchMatMulV2") continue;
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) continue;
    op_desc->AppendIrAttrName("adj_x1");
    op_desc->AppendIrAttrName("adj_x2");
    op_desc->AppendIrAttrName("offset_x");
    AttrUtils::SetBool(op_desc, "adj_x1", false);
    AttrUtils::SetBool(op_desc, "adj_x2", false);
    AttrUtils::SetInt(op_desc, "offset_x", 0);
  }
}
void SetMatMulIrAttrs(const ComputeGraphPtr &cg) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "MatMul") continue;
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) continue;
    op_desc->AppendIrAttrName("trans_a");
    op_desc->AppendIrAttrName("trans_b");
    AttrUtils::SetBool(op_desc, "trans_a", false);
    AttrUtils::SetBool(op_desc, "trans_b", false);
    AttrUtils::SetBool(op_desc, "transpose_x1", false);
    AttrUtils::SetBool(op_desc, "transpose_x2", false);
  }
}
void SetMatMulV2IrAttrs(const ComputeGraphPtr &cg) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "MatMulV2") continue;
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) continue;
    op_desc->AppendIrAttrName("transpose_x1");
    op_desc->AppendIrAttrName("transpose_x2");
    op_desc->AppendIrAttrName("offset_x");
    if (!op_desc->HasAttr("transpose_x1")) {
      AttrUtils::SetBool(op_desc, "transpose_x1", false);
    }
    if (!op_desc->HasAttr("transpose_x2")) {
      AttrUtils::SetBool(op_desc, "transpose_x2", false);
    }
    if (!op_desc->HasAttr("offset_x")) {
      AttrUtils::SetInt(op_desc, "offset_x", 0);
    }
  }
}
void SetGatherIrAttrs(const ComputeGraphPtr &cg, int64_t batch_dims = 0) {
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() != "Gather") continue;
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) continue;
    op_desc->AppendIrAttrName("axis");
    op_desc->AppendIrAttrName("batch_dims");
    AttrUtils::SetInt(op_desc, "axis", 0);
    AttrUtils::SetInt(op_desc, "batch_dims", batch_dims);
  }
}
}  // namespace

TEST_F(SymbolicShapeInferenceUT, InferShapeForNoInferFunc) {
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(foo1));
    CHAIN(NODE(foo1)->EDGE(0, 0)->NODE(foo2)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto data0 = graph->FindFirstNodeMatchType(ge::DATA);
  ASSERT_NE(data0, nullptr);
  gert::SymbolShape symbol_shape({
      Symbol("s0"),
      Symbol(1),
      Symbol("s1"),
      Symbol("s2"),
  });
  data0->GetOpDescBarePtr()
      ->MutableOutputDesc(0)
      ->template GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolShape(symbol_shape);
  ExpectNodeInfo expect_node1("FOO1", {Symbol(2), Symbol(2), Symbol(3), Symbol(4)}, {}, {}, {});
  ExpectNodeInfo expect_node2("FOO2", {}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  expect_node_vec.push_back(expect_node2);
  ASSERT_NE(RunSymbolInferenceTest(graph, expect_node_vec, {}), SUCCESS);
}

namespace {
auto const11 = OP_CFG("Const").InCnt(0).OutCnt(1).OutNames({"y"}).Build("Const");
auto const22 = OP_CFG("Constant").InCnt(0).OutCnt(1).OutNames({"y"}).Build("Constant");

}  // namespace

// 如果不是按照IR注册的方式造的node，后续造context时拿不到IR属性
// infer symbol shape时，会尝试从IR中恢复IR属性，这对const\constant节点非常重要，此处添加ut校验
TEST_F(SymbolicShapeInferenceUT, InferShapeForConstantWithRecoverIrAttr) {
  DEF_GRAPH(g1) {
    CHAIN(NODE(const11)->EDGE(0, 0)->NODE(foo1))->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT);
    CHAIN(NODE(const22)->EDGE(0, 0)->NODE(foo2))->EDGE(0, 1)->NODE("NetOutput", NETOUTPUT);
  };
  auto graph = ToComputeGraph(g1);
  auto const_node = graph->FindFirstNodeMatchType(ge::CONSTANT);
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({2, 2, 3, 4}));
  const_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({2, 2, 3, 4}));
  auto constant_node = graph->FindFirstNodeMatchType(ge::CONSTANTOP);
  constant_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({2, 2, 3, 4}));
  constant_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({2, 2, 3, 4}));
  ExpectNodeInfo expect_node1("Const", {Symbol(2), Symbol(2), Symbol(3), Symbol(4)}, {}, {}, {});
  ExpectNodeInfo expect_node2("Constant", {Symbol(2), Symbol(2), Symbol(3), Symbol(4)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  expect_node_vec.push_back(expect_node2);
  ASSERT_EQ(RunSymbolInferenceTest(graph, expect_node_vec, {}), SUCCESS);
}

/**
 *      Data0    Data1
 *        |    /   |
 *        |  /     |
 *       Pow     Tanh
 *        \       /
 *    SquaredDifference
 *           |
 *          Neg
 *           |
 *        NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForDecomposedGraph) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s2", "s1", "s0", "s0"}).data(), 4), 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data1, std::vector<const char *>({"s1", "1", "s0"}).data(), 3), 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto pow = EsPow(data0, data1);
  auto tanh = EsTanh(data1);
  auto squaredD = EsSquaredDifference(pow, tanh);
  auto neg = EsNeg(squaredD);

  ASSERT_EQ(EsSetGraphOutput(neg, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  ExpectNodeInfo expect_node1("Neg_3", {s2, s1, s0, s0}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

//
//                         ┌────────────┐
//                         │   const2   │ ────────────────────────────────────────────────────────────────────────┐
//                         └────────────┘                                                                         │
//                           │                                                                                    │
//                           │ (0,2)                                                                              │
//                           ∨                                                                                    │
//     ┌────────┐  (0,1)   ┌────────────┐  (0,0)   ┌────────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────────────┐  │
//     │ const1 │ ───────> │ gatherv2_1 │ ───────> │   pack1    │ ───────> │ reshape1 │ ───────> │ Node_Output │  │
//     └────────┘          └────────────┘          └────────────┘          └──────────┘          └─────────────┘  │
//       │                   ∧                       ∧                       ∧                                    │
//  ┌────┘                   │ (0,0)                 │                       │ (0,0)                              │
//  │                        │                       │                       │                                    │
//  │  ┌────────┐  (0,0)   ┌────────────┐            │                     ┌──────────┐                           │
//  │  │ data_1 │ ───────> │   shape1   │            │                     │  data_3  │                           │
//  │  └────────┘          └────────────┘            │                     └──────────┘                           │
//  │                                                │                                                            │
//  │                                                │ (0,1)                                                      │
//  │                                                │                                                            │
//  │  ┌────────┐  (0,0)   ┌────────────┐  (0,0)   ┌────────────┐  (0,2)                                          │
//  │  │ data_2 │ ───────> │   shape2   │ ───────> │ gatherv2_2 │ <───────────────────────────────────────────────┘
//  │  └────────┘          └────────────┘          └────────────┘
//  │   (0,1)                                        ∧
//  └────────────────────────────────────────────────┘
TEST_F(SymbolicShapeInferenceUT, simple_symbolic_kernel_compute) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data_1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data2 = EsCreateGraphInputWithDetails(graph_, 2, "data_2", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto const1 = EsCreateScalarInt32(graph_, 0);

  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"(s0 * s1)", "s2"}).data(), 2), 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data1, std::vector<const char *>({"(s3 * s4)", "s5"}).data(), 2), 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data1, std::vector<const char *>({"s0", "(s1 * s3 * s4)"}).data(), 2), 0);

  auto shape1 = EsShape(data0, 3);  // DT_INT32
  auto gather_1 = EsGatherV2(shape1, const1, const1, 0, false, false);

  auto shape2 = EsShape(data2, 3);  // DT_INT32
  auto gather_2 = EsGatherV2(shape2, const1, const1, 0, false, false);

  std::vector<EsCTensorHolder *> esb;
  esb.push_back(gather_1);
  esb.push_back(gather_2);
  auto pack1 = EsPack(esb.data(), 2, 0, 2);

  auto reshape = EsReshape(data2, pack1, 0, -1);
  ASSERT_EQ(EsSetGraphOutput(reshape, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  // todo: 如果补充了gather、pack等的symbol kernel，继续完善该用例
}

//
// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ExpandDims│ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ———————————
// └────────┘
TEST_F(SymbolicShapeInferenceUT, test_expanddims_with_symbols_value) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2"}).data(), 3), 0);
  auto const0 = EsCreateScalarInt32(graph_, 1);
  ASSERT_NE(const0, nullptr);

  const auto expandDims = EsExpandDims(data0, const0);
  ge::TensorDesc expandDims_input_desc;
  expandDims->GetProducer().GetInputDesc(1, expandDims_input_desc);
  expandDims_input_desc.SetDataType(DT_INT32);
  expandDims->GetProducer().UpdateInputDesc(1, expandDims_input_desc);
  ASSERT_EQ(EsSetGraphOutput(expandDims, 0), 0);
  const auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  ExpectNodeInfo expect_node1(EXPANDDIMS, {Symbol("s0"), Symbol(1), Symbol("s1"), Symbol("s2")}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

//
// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ Pad      │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ───────————
// └────────┘
TEST_F(SymbolicShapeInferenceUT, test_pad_with_symbols_value) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2"}).data(), 3), 0);

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {3, 2};
  auto const0 = EsCreateConstInt32(graph_, const_data0.data(), const_dim.data(), const_dim.size());

  const auto pad = EsPad(data0, const0);

  ASSERT_EQ(EsSetGraphOutput(pad, 0), 0);
  const auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  ExpectNodeInfo expect_node1(PAD, {Symbol("s0") + Symbol(3), Symbol("s1") + Symbol(3), Symbol("s2") + Symbol(2)},
                              {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

//
// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ Pad      │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ───────————
// └────────┘
TEST_F(SymbolicShapeInferenceUT, test_pad_with_symbols_value_but_error_shape) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2"}).data(), 3), 0);

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 4}; // paddings.size != data0.dims * 2 校验报错
  auto const0 = EsCreateConstInt32(graph_, const_data0.data(), const_dim.data(), const_dim.size());

  const auto pad = EsPad(data0, const0);

  ASSERT_EQ(EsSetGraphOutput(pad, 0), 0);
  const auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferenceUT, test_pad_with_vector) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2"}).data(), 3), 0);

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {6};
  auto const0 = EsCreateConstInt32(graph_, const_data0.data(), const_dim.data(), const_dim.size());

  const auto pad = EsPad(data0, const0);

  ASSERT_EQ(EsSetGraphOutput(pad, 0), 0);
  const auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, test_unsupport_subgraph) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2"}).data(), 3), 0);

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3};  // error shape， paddings 的shape应该是{3, 2} 对应 {inputDimNum, 2}
  auto const0 = EsCreateConstInt32(graph_, const_data0.data(), const_dim.data(), const_dim.size());

  const auto pad = EsPad(data0, const0);

  ASSERT_EQ(EsSetGraphOutput(pad, 0), 0);
  const auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto reshape_node = cg->FindFirstNodeMatchType(PAD);
  reshape_node->GetOpDesc()->AddSubgraphName("subgraph");
  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, test_abnormal_reshape) {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);

  auto reshape = OP_CFG(RESHAPE).InCnt(1).OutCnt(1).OutNames({"y"}).Build("reshape1");
  GeTensorDesc in_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  GeTensorDesc out_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);

  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(reshape));
    CHAIN(NODE(reshape)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(reshape)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto reshape_node = graph->FindFirstNodeMatchType(RESHAPE);
  ASSERT_NE(reshape_node, nullptr);
  reshape_node->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 3, 4}));
  reshape_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 3, 4}));
  reshape_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 24}));
  reshape_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 24}));
  reshape_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  reshape_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  reshape_node->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_ND);
  reshape_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
  reshape_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  reshape_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);

  DataInfo di = {ge::FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}};
  SetNoStorage(graph, "Data", di, 0);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({1, 2, 3, 4})));
  td.SetOriginShape((GeShape({1, 2, 3, 4})));
  inputs.emplace_back(td);
  ge::GetThreadLocalContext().GetOo().Initialize(ge::GetThreadLocalContext().GetAllOptions(),
                                             OptionRegistry::GetInstance().GetRegisteredOptTable());
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, inputs), ge::GRAPH_SUCCESS);

  auto op_desc = reshape_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(1), Symbol(24)}));
  unsetenv("ASCEND_OPP_PATH");
  unsetenv("LD_LIBRARY_PATH");
}

//         ┌────────┐  (0,0)
//         │ data_0 │ ────────
//         └────────┘         |
//                            |
// ┌────────┐  (0,1)   ┌─────────────┐ (0,0)    ┌─────────────┐
// │ data_1 │ ───────> │Stridedslice │ ───────> │ Node_Output │
// └────────┘          └─────────────┘          └─────────────┘
//                            |  |
//      ┌────────┐  (0,2)     |  |
//      │ data_2 │ ———————————   |
//      └────────┘               |
//                               |
//      ┌────────┐  (0,2)        |
//      │ data_3 │ ——————————————
//      └────────┘
TEST_F(SymbolicShapeInferenceUT, test_stridedslice_infershape) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);

  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3};
  auto data1 = EsCreateVectorInt64(graph_, data1_value.data(), 4);
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5};
  auto data2 = EsCreateVectorInt64(graph_, data2_value.data(), 4);
  // strides
  std::vector<int64_t> data3_value = {1, 1, 1, 1};
  auto data3 = EsCreateVectorInt64(graph_, data3_value.data(), 4);

  auto strided_slice =
      EsStridedSlice(data0, data1, data2, data3, static_cast<int64_t>(0b0010), static_cast<int64_t>(0b0010),
                     static_cast<int64_t>(0b0100), static_cast<int64_t>(0b1101), static_cast<int64_t>(0b0111));
  ASSERT_EQ(EsSetGraphOutput(strided_slice, 0), 0);

  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  // 跳过hostcompute
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, -1, -1, -1}};
  SetNoStorage(cg, "data_0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  const std::vector<Expression> expect_output_shape = {Symbol(1),
                                                       Symbol("s1"),
                                                       Symbol("s2"),
                                                       Symbol("s3"),
                                                       Symbol("s4"),
                                                       Symbol(1)};
  ExpectNodeInfo expect_node1(STRIDEDSLICE, expect_output_shape, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0    Data1
 *        |    /   |
 *        |  /     |
 *       Pow     Tanh
 *        |       |
 *        |      Foo1
 *        \       /
 *    SquaredDifference
 *           |
 *          Neg
 *           |
 *        NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForGraphWithNodeNotSupportSymbolInfer) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s2", "s1", "s0", "s0"}).data(), 4), 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data1, std::vector<const char *>({"s1", "1", "s0"}).data(), 3), 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto pow = EsPow(data0, data1);
  auto tanh = EsTanh(data1);
  auto squaredD = EsSquaredDifference(pow, tanh);
  auto neg = EsNeg(squaredD);

  ASSERT_EQ(EsSetGraphOutput(neg, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  GeTensorDesc foo_input_desc(GeShape({-1, -1, 3, 6}), FORMAT_ND, DT_INT32);
  foo_input_desc.SetOriginShape(GeShape({-1, -1, 3, 6}));
  auto foo_op_desc = std::make_shared<OpDesc>("foo", "Foo");
  foo_op_desc->AddInputDesc(foo_input_desc);
  foo_op_desc->AddOutputDesc(foo_input_desc);
  auto foo_node = cg->AddNode(foo_op_desc);
  auto tanh_node = cg->FindFirstNodeMatchType("Tanh");
  ASSERT_NE(tanh_node, nullptr);

  auto squared_difference_node = cg->FindFirstNodeMatchType("SquaredDifference");
  ASSERT_NE(squared_difference_node, nullptr);

  ASSERT_EQ(GraphUtils::InsertNodeBetweenDataAnchors(tanh_node->GetOutDataAnchor(0),
                                                     squared_difference_node->GetInDataAnchor(1), foo_node),
            SUCCESS);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto foo_attr = foo_op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(foo_attr, nullptr);
  auto neg_node = cg->FindNode("Neg_3");
  ASSERT_NE(neg_node, nullptr);
  auto neg_op_desc = neg_node->GetOpDesc();
  ASSERT_NE(neg_op_desc, nullptr);
  auto neg_attr = neg_op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(neg_attr, nullptr);
  auto squared_difference_op_desc = squared_difference_node->GetOpDesc();
  ASSERT_NE(squared_difference_op_desc, nullptr);
  auto sd_input_attr_0 = squared_difference_op_desc->GetInputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(sd_input_attr_0, nullptr);
  ASSERT_EQ(sd_input_attr_0->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
  auto sd_input_attr_1 = squared_difference_op_desc->GetInputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(sd_input_attr_1, nullptr);
  auto sd_output_attr_0 = squared_difference_op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(sd_output_attr_0, nullptr);
}

//         ┌────────┐  (0,0)
//         │ data_0 │ ────────
//         └────────┘         |
//                            |
// ┌────────┐  (0,1)   ┌─────────────┐ (0,0)    ┌─────────────┐
// │ data_1 │ ───────> │BatchMatMulV2│ ───────> │ Node_Output │
// └────────┘          └─────────────┘          └─────────────┘
TEST_F(SymbolicShapeInferenceUT, test_batchmatmulv2_infershape) {
  const auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2", "s3", "s4"}).data(), 5), 0);

  const auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data_1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data1, std::vector<const char *>({"s2", "s4", "s3"}).data(), 3), 0);

  auto bmmv2 = EsBatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(EsSetGraphOutput(bmmv2, 0), 0);

  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SetBatchMatMulV2IrAttrs(cg);
  std::vector<Expression> expect_output_shape = {Symbol("s0"), Symbol("s1"), Symbol("s2"), Symbol("s3"), Symbol("s3")};
  ExpectNodeInfo expect_node1("BatchMatMulV2", expect_output_shape, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, test_reshape_1) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data_2", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);

  ASSERT_EQ(EsSetOriginSymbolShape(data0,std::vector<const char *>({"s3","(s1 * s2)","s0",}).data(),3),0);
  ASSERT_EQ(EsSetOriginSymbolShape(data1, std::vector<const char *>({"2"}).data(), 1), 0);

  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto sym2 = Symbol("s2");
  auto sym3 = Symbol("s3");

  // auto shape = EsShape(data0, 3);  // DT_INT32
  auto reshape = EsReshape(data0, data1, 0, -1);
  ASSERT_EQ(EsSetGraphOutput(reshape, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  // Set symbolic value using internal node access after graph is built
  auto data_node = cg->FindNode("data_2");
  ASSERT_NE(data_node, nullptr);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym1 * sym3);
  ptr->emplace_back(sym2 * sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->template GetOrCreateAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  ExpectNodeInfo expect_node1(RESHAPE, {s1 * s3, s0 * s2}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}
/**
 *      Data0     Data1
 *         \     /
 *           Add
 *            |
 *            |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForBroadCastGraphWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto add = EsAdd(data0, data1);
  ASSERT_EQ(EsSetGraphOutput(add, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, 1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 1, -1, -1};
  SetNoStorage(cg, "data1", di, 1);
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {3, 2, 3, 1};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {3, 1, 3, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  const std::set<std::string> expect_guard = {"ExpectNe(1, s4)", "ExpectEq(3, s3)", "ExpectEq(s0, s2)", "ExpectNe(1, s1)", "ExpectNe(0, s0)",
                                              "ExpectNe(0, s1)", "ExpectNe(0, s2)", "ExpectNe(0, s3)", "ExpectNe(0, s4)"};
  std::vector<Expression> expect_output_shape = {Symbol("s2"), Symbol("s1"), Symbol(3), Symbol("s4")};
  ExpectNodeInfo expect_node1("Add", expect_output_shape, expect_guard, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0     Data1
 *         \     /
 *           Matmul
 *            |
 *            |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForMatmulGraphWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto matmul = EsMatMul(data0, data1, nullptr, false, false);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 3};
  SetNoStorage(cg, "data1", di, 1);
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SetMatMulIrAttrs(cg);
  ExpectNodeInfo expect_node1("MatMul", {Symbol("s0"), Symbol(3)}, {}, {"ExpectEq(s1, s2)"}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0     Data1
 *         \     /
 *           Add
 *            |
 *            |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, TestMultiInfer) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  std::vector<int64_t> dims{1, 16};
  int64_t dims_size = 2;
  auto const_0 = EsCreateConstInt64(graph_, dims.data(), &dims_size, 1);
  ASSERT_NE(const_0, nullptr);

  auto matmul = EsMatMulV2(data0, data1, nullptr, nullptr, 0, 0, 0);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {128, 16}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {16, 1};
  SetNoStorage(cg, "data1", di, 1);
  auto matmul_node = cg->FindFirstNodeMatchType("MatMulV2");
  ASSERT_NE(matmul_node, nullptr);
  auto matmul_op_desc = matmul_node->GetOpDesc();
  ASSERT_NE(matmul_op_desc, nullptr);
  SetMatMulV2IrAttrs(cg);
  matmul_op_desc->MutableInputDesc(0)->SetShape(GeShape({128, 16}));
  matmul_op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({128, 16}));
  matmul_op_desc->MutableInputDesc(0)->SetDataType(DT_INT64);
  matmul_op_desc->MutableInputDesc(1)->SetShape(GeShape({16, 1}));
  matmul_op_desc->MutableInputDesc(1)->SetOriginShape(GeShape({16, 1}));
  matmul_op_desc->MutableInputDesc(1)->SetDataType(DT_INT64);
  matmul_op_desc->MutableOutputDesc(0)->SetShape(GeShape({128, 1}));
  matmul_op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({128, 1}));
  matmul_op_desc->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {128, 16};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {16, 1};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  ExpectNodeInfo expect_node1("MatMulV2", {Symbol(128), Symbol(1)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  SetMatMulV2IrAttrs(cg);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);

  auto reshape_desc = std::make_shared<ge::OpDesc>("Reshape_" + matmul_node->GetName(), "Reshape");
  reshape_desc->AddInputDesc("x", matmul_op_desc->GetInputDesc(1));
  reshape_desc->AddInputDesc("shape", ge::GeTensorDesc(ge::GeShape({2}), FORMAT_ND, DT_INT64));
  ge::GeTensorDesc reshape_output_desc = matmul_op_desc->GetInputDesc(1).Clone();
  auto matmul_x2_dims = matmul_op_desc->GetInputDesc(1).GetShape().GetDims();
  std::vector<int64_t> shape = {matmul_x2_dims[1], matmul_x2_dims[0]};
  reshape_output_desc.SetShape(ge::GeShape(shape));
  reshape_output_desc.SetOriginShape(ge::GeShape(shape));
  reshape_desc->AddOutputDesc("y", reshape_output_desc);
  (void)ge::AttrUtils::SetListInt(reshape_desc, "shape", shape);
  ge::NodePtr reshape_node = cg->InsertNodeBefore(matmul_node, reshape_desc);
  auto matmul_in_anchor = matmul_node->GetInDataAnchor(1);
  auto out_anchor = matmul_in_anchor->GetPeerOutAnchor();
  EXPECT_EQ(ge::GraphUtils::InsertNodeBetweenDataAnchors(out_anchor, matmul_in_anchor, reshape_node), ge::GRAPH_SUCCESS);

  auto const_node = cg->FindNode("Const_0");
  ASSERT_NE(const_node, nullptr);
  EXPECT_EQ(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(1)), ge::GRAPH_SUCCESS);

  AttrUtils::SetBool(matmul_op_desc, "transpose_x2", true);
  matmul_op_desc->MutableInputDesc(1)->SetShape(ge::GeShape(shape));
  matmul_op_desc->MutableInputDesc(1)->SetOriginShape(ge::GeShape(shape));

  ExpectNodeInfo expect_node2("MatMulV2", {Symbol(128), Symbol(1)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec1;
  expect_node_vec1.push_back(expect_node2);
  SetMatMulV2IrAttrs(cg);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec1, input_vec), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, TestMultiInfer2) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto matmul = EsMatMulV2(data0, data1, nullptr, nullptr, 0, 0, 0);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {128, 16}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {16, 1};
  SetNoStorage(cg, "data1", di, 1);

  auto matmul_node = cg->FindFirstNodeMatchType("MatMulV2");
  ASSERT_NE(matmul_node, nullptr);
  auto matmul_op_desc = matmul_node->GetOpDesc();
  ASSERT_NE(matmul_op_desc, nullptr);
  SetMatMulV2IrAttrs(cg);
  matmul_op_desc->MutableInputDesc(0)->SetShape(GeShape({128, 16}));
  matmul_op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({128, 16}));
  matmul_op_desc->MutableInputDesc(0)->SetDataType(DT_INT64);
  matmul_op_desc->MutableInputDesc(1)->SetShape(GeShape({16, 1}));
  matmul_op_desc->MutableInputDesc(1)->SetOriginShape(GeShape({16, 1}));
  matmul_op_desc->MutableInputDesc(1)->SetDataType(DT_INT64);
  matmul_op_desc->MutableOutputDesc(0)->SetShape(GeShape({128, 1}));
  matmul_op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({128, 1}));
  matmul_op_desc->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {128, 16};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {16, 1};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  ExpectNodeInfo expect_node1("MatMulV2", {Symbol(128), Symbol(1)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec1;
  expect_node_vec1.push_back(expect_node1);
  SetMatMulV2IrAttrs(cg);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec1, input_vec), SUCCESS);

  auto reshape_desc = std::make_shared<ge::OpDesc>("Reshape_" + matmul_node->GetName(), "Reshape");
  reshape_desc->AddInputDesc("x", matmul_op_desc->GetInputDesc(1));
  reshape_desc->AddInputDesc("shape", ge::GeTensorDesc(ge::GeShape({2}), FORMAT_ND, DT_INT64));
  ge::GeTensorDesc reshape_output_desc = matmul_op_desc->GetInputDesc(1).Clone();
  auto matmul_x2_dims = matmul_op_desc->GetInputDesc(1).GetShape().GetDims();
  std::vector<int64_t> shape = {matmul_x2_dims[1], matmul_x2_dims[0]};
  reshape_output_desc.SetShape(ge::GeShape(shape));
  reshape_output_desc.SetOriginShape(ge::GeShape(shape));
  reshape_desc->AddOutputDesc("y", reshape_output_desc);
  (void)ge::AttrUtils::SetListInt(reshape_desc, "shape", shape);
  ge::NodePtr reshape_node = cg->InsertNodeBefore(matmul_node, reshape_desc);
  auto matmul_in_anchor = matmul_node->GetInDataAnchor(1);
  auto out_anchor = matmul_in_anchor->GetPeerOutAnchor();
  EXPECT_EQ(ge::GraphUtils::InsertNodeBetweenDataAnchors(out_anchor, matmul_in_anchor, reshape_node), ge::GRAPH_SUCCESS);
  AttrUtils::SetBool(matmul_op_desc, "transpose_x2", true);
  matmul_op_desc->MutableInputDesc(1)->SetShape(ge::GeShape(shape));
  matmul_op_desc->MutableInputDesc(1)->SetOriginShape(ge::GeShape(shape));

  ExpectNodeInfo expect_node2("MatMulV2", {Symbol(128), Symbol(1)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec2;
  expect_node_vec2.push_back(expect_node2);
  ASSERT_NE(RunSymbolInferenceTest(cg, expect_node_vec2, input_vec), SUCCESS);
}

/**
 *      Data0     Data1    Data2
 *        |        /        /
 *        \       /        /
 *         \     /        /
 *           ConcatV2D             
 *                |
 *                |
 *             NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForConcatV2DWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data2 = EsCreateGraphInputWithDetails(graph_, 2, "data2", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(data2, nullptr);
  std::vector<EsCTensorHolder *> concat_input;
  concat_input.push_back(data0);
  concat_input.push_back(data1);
  concat_input.push_back(data2);
  const auto concat0 = EsConcatV2D(concat_input.data(), 3, 2, 3);
  ASSERT_EQ(EsSetGraphOutput(concat0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, 2}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 3, 2, -1};
  SetNoStorage(cg, "data1", di, 1);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data2", di, 2);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 2, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {2, 3, 4, 2};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));
  const std::set<std::string> expect_assert = {"ExpectEq(2, s7)", "ExpectEq(3, s1)", "ExpectEq(s0, s2)",
                                               "ExpectEq(2, s3)", "ExpectEq(s0, s4)", "ExpectEq(3, s5)"};
  ExpectNodeInfo expect_node("ConcatV2D", {Symbol("s4"), Symbol(3), (Symbol(5) + Symbol("s6")), Symbol(2)}, {},
                              expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0            Const0
 *        |              /
 *        \             /
 *         \           /
 *          \         /          
 *            \      /
 *             Reshape
 *                |
 *             NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForReShapeConstWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int64_t> dims{2, -1, 3};
  int64_t dims_size = 3;
  auto const_0 = EsCreateConstInt64(graph_, dims.data(), &dims_size, 1);
  ASSERT_NE(const_0, nullptr);
  auto reshape0 = EsReshape(data0, const_0, 0, -1);
  ASSERT_EQ(EsSetGraphOutput(reshape0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, -1}};
  SetNoStorage(cg, "data0", di, 0);

  auto reshape_op_0 = cg->FindFirstNodeMatchType("Reshape");
  ASSERT_NE(reshape_op_0, nullptr);
  auto reshape_op_desc0 = reshape_op_0->GetOpDesc();
  reshape_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  reshape_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 2, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  const std::vector<Expression> expect_symbol_output_shape = {Symbol(2), sym::Rational(1, 2) * Symbol("s0") * Symbol("s1") * Symbol("s2"), Symbol(3)};
  ExpectNodeInfo expect_node("Reshape", expect_symbol_output_shape, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0            Data1     Data2
 *        |              /         /
 *        \             /         /
 *         \           /         /
 *          \         /         / 
 *            \       |         /
 *                  Select
 *                    |
 *                 NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForSelectWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data2 = EsCreateGraphInputWithDetails(graph_, 2, "data2", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(data2, nullptr);

  const auto select0 = EsSelect(data0, data1, data2);
  ASSERT_EQ(EsSetGraphOutput(select0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 2, 2}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 3, 2, -1};
  SetNoStorage(cg, "data1", di, 1);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data2", di, 2);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 2, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 2, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {2, 3, 2, 2};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));
  const std::set<std::string> expect_assert =
      {"ExpectEq(2, s3)", "ExpectEq(2, s6)", "ExpectEq(3, s1)", "ExpectEq(s0, s2)",
       "ExpectEq(s3, s7)","ExpectEq(3, s5)","ExpectEq(s2, s4)"};
  const std::vector<Expression> expect_symbol_output_shape = {Symbol("s4"), Symbol(3), Symbol(2), Symbol(2)};
  ExpectNodeInfo expect_node("Select", expect_symbol_output_shape, {}, expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0            Data1     Const
 *        |              /         /
 *        \             /         /
 *         \          Shape      /
 *          \         /         / 
 *            \       |        /
 *                  Slice
 *                    |
 *                 NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForSliceWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  std::vector<int64_t> dims{1, 2, -1, -1};
  int64_t dims_size = 4;
  auto const_0 = EsCreateConstInt64(graph_, dims.data(), &dims_size, 1);
  ASSERT_NE(const_0, nullptr);

  const auto shape_0 = EsShape(data1, 3);

  const auto slice0 = EsSlice(data0, shape_0, const_0);
  ASSERT_EQ(EsSetGraphOutput(slice0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, 4}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {2, 1, -1, -1};
  SetNoStorage(cg, "data1", di, 1);

  auto shape_node_0 = cg->FindFirstNodeMatchType("Shape");
  ASSERT_NE(shape_node_0, nullptr);
  auto shape_op_desc0 = shape_node_0->GetOpDesc();
  shape_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto slice_op_0 = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_op_0, nullptr);
  auto slice_op_desc0 = slice_op_0->GetOpDesc();
  slice_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  slice_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT64);
  slice_op_desc0->MutableInputDesc(2)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 4, 3, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 1, 1, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  const std::set<std::string> expect_assert =
      {"ExpectLe(1, s1)", "ExpectLe(s3, 4)", "ExpectLe(2, s0)", "ExpectLe(0, s2)",
       "ExpectLe(0, s3)", "ExpectLe(s2, 3)"};
  const std::vector<Expression> expect_symbol_output_shape = {Symbol(1), Symbol(2), (Symbol(3) - Symbol("s2")), (Symbol(4) - Symbol("s3"))};
  ExpectNodeInfo expect_node("Slice", expect_symbol_output_shape, {}, expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0            
 *        |          
 *        \           
 *         \          
 *          \         
 *            \       
 *            Squeeze
 *               |
 *           NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForSqueezeAllDimWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int64_t> axis = {};
  const auto squeeze = EsSqueeze(data0, axis.data(), axis.size());
  ASSERT_EQ(EsSetGraphOutput(squeeze, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, 1, -1}};
  SetNoStorage(cg, "data0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 1, 3, 1, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  const std::set<std::string> expect_guard =
      {"ExpectNe(1, s2)", "ExpectNe(1, s0)", "ExpectEq(1, s1)", "ExpectNe(0, s0)", "ExpectNe(0, s1)", "ExpectNe(0, s2)"};
  const std::vector<Expression> expect_symbol_output_shape = {Symbol("s0"), Symbol(3), Symbol("s2")};
  ExpectNodeInfo expect_node("Squeeze", expect_symbol_output_shape, expect_guard, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0            
 *        |          
 *        \           
 *         \          
 *          \         
 *            \       
 *            Squeeze
 *               |
 *           NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForSqueezeWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int64_t> axis = {1, 3};
  const auto squeeze = EsSqueeze(data0, axis.data(), axis.size());
  ASSERT_EQ(EsSetGraphOutput(squeeze, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, 1, -1}};
  SetNoStorage(cg, "data0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 1, 3, 1, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  const std::set<std::string> expect_assert = {"ExpectEq(1, s1)"};
  const std::vector<Expression> expect_symbol_output_shape = {Symbol("s0"), Symbol(3), Symbol("s2")};
  ExpectNodeInfo expect_node("Squeeze", expect_symbol_output_shape, {}, expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Const            Data1     Const
 *        |              /         /
 *        \             /         /
 *         \          Shape      /
 *          \         /         / 
 *            \       |        /
 *                  Range
 *                    |
 *                 NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForRangeWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int64_t> dims_start{2};
  std::vector<int64_t> dims_end{8};
  int64_t dims_size = 1;
  auto const_0 = EsCreateConstInt64(graph_, dims_end.data(), &dims_size, 1);
  ASSERT_NE(const_0, nullptr);
  auto const_1 = EsCreateConstInt64(graph_, dims_start.data(), &dims_size, 1);
  ASSERT_NE(const_1, nullptr);
  const auto shape_0 = EsShape(data0, 3);

  const auto range0 = EsRange(const_1, const_0, shape_0, false);
  ASSERT_EQ(EsSetGraphOutput(range0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1}};
  SetNoStorage(cg, "data0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  const std::set<std::string> expect_assert = {"ExpectNe(0, s0)", "ExpectLt(0, s0)"};
  const std::vector<Expression> expect_symbol_output_shape = {sym::Ceiling(Symbol(6) / Symbol("s0"))};
  ExpectNodeInfo expect_node("Range", expect_symbol_output_shape, {}, expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/**
 *      Data0     Data1    Data2
 *        |        /        /
 *        \       /        /
 *         \     /        /
 *               Pack             
 *                |
 *                |
 *             NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForPackWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data2 = EsCreateGraphInputWithDetails(graph_, 2, "data2", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(data2, nullptr);
  std::vector<EsCTensorHolder *> pack_input;
  pack_input.push_back(data0);
  pack_input.push_back(data1);
  pack_input.push_back(data2);
  const auto pack0 = EsPack(pack_input.data(), 3, 2, 3);
  ASSERT_EQ(EsSetGraphOutput(pack0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, 2}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 3, 3, -1};
  SetNoStorage(cg, "data1", di, 1);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data2", di, 2);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 3, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {2, 3, 3, 2};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  auto pack_node_0 = cg->FindFirstNodeMatchType(PACK);
  ASSERT_NE(pack_node_0, nullptr);
  auto pack_op_desc_0 = pack_node_0->GetOpDesc();
  pack_op_desc_0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 3, 2}));
  pack_op_desc_0->MutableInputDesc(1)->SetShape(GeShape({-1, 3, 3, -1}));
  pack_op_desc_0->MutableInputDesc(2)->SetShape(GeShape({-1, -1, -1, -1}));
  const std::set<std::string> expect_assert = {"ExpectEq(2, s7)", "ExpectEq(3, s1)", "ExpectEq(s0, s2)",
    "ExpectEq(2, s3)", "ExpectEq(3, s6)", "ExpectEq(s1, s5)", "ExpectEq(s0, s4)"};
  const std::vector<Expression> expect_symbol_output_shape = {Symbol("s4"), Symbol(3), Symbol(3), Symbol(3), Symbol(2)};
  ExpectNodeInfo expect_node("Pack", expect_symbol_output_shape, {}, expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}
/**
 *      Data0     Data1
 *         \     /
 *           BatchMatmul
 *            |
 *            |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceUT, InferShapeForBatchMatmulGraphWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto matmul = EsBatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 1, -1, 3, -1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 1, -1, 2, -1};
  SetNoStorage(cg, "data1", di, 1);
  SetBatchMatMulV2IrAttrs(cg);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {4, 2, 1, 2, 3, 4, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {3, 1, 1, 2, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  const std::set<std::string> expect_guard =
    {"ExpectEq(1, s6)", "ExpectNe(3, s6)", "ExpectNe(1, s5)", "ExpectNe(1, s2)", "ExpectNe(0, s0)", "ExpectNe(0, s1)",
     "ExpectNe(0, s2)", "ExpectNe(0, s3)", "ExpectNe(0, s4)", "ExpectNe(0, s5)", "ExpectNe(0, s6)", "ExpectNe(0, s7)"};
  const std::set<std::string> expect_assert = {"ExpectEq(2, s4)"};
  const std::vector<Expression> expect_symbol_output_shape = {Symbol("s0"), Symbol("s1"), Symbol("s5"), Symbol("s2"),
    Symbol(3), Symbol("s3"), Symbol("s7")};
  ExpectNodeInfo expect_node("BatchMatMulV2", expect_symbol_output_shape, expect_guard, expect_assert, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

REG_OP(MultisliceConcat)
   .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
   .REQUIRED_ATTR(concat_size, ListInt)
   .REQUIRED_ATTR(slice_begin, ListInt)
   .REQUIRED_ATTR(silce_size, ListInt)
   .REQUIRED_ATTR(concat_num, Int)
   .REQUIRED_ATTR(max_concat_size, Int)
   .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
   .OP_END_FACTORY_REG(MultisliceConcat)

TEST_F(SymbolicShapeInferenceUT, test_multisliceconcat) {
  vector<int64_t> vec_concatsize = {2, 4};
  vector<int64_t> vec_slicebegin = {4, 5, 12, 5, 20, 28};
  vector<int64_t> vec_slicesize = {1, 1, 2, 2, 2, 4};
  auto multisliceconcat = OP_CFG("MultisliceConcat").InCnt(1).OutCnt(2)
                                  .Attr("concat_size", vec_concatsize)
                                  .Attr("slice_begin", vec_slicebegin)
                                  .Attr("silce_size", vec_slicesize)
                                  .Attr("concat_num", 2)
                                  .Attr("max_concat_size", 4).Build("MultisliceConcat1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(multisliceconcat));
    CHAIN(NODE(multisliceconcat)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(multisliceconcat)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto multisliceconcat_node = graph->FindFirstNodeMatchType("MultisliceConcat");
  ASSERT_NE(multisliceconcat_node, nullptr);
  multisliceconcat_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  auto data_node = graph->FindFirstNodeMatchType(DATA);
  ASSERT_NE(data_node, nullptr);
  gert::SymbolShape symol_shape({Symbol("s0"), Symbol("s1")});
  data_node->GetOpDesc()->MutableOutputDesc(0)->template GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.
             SetSymbolShape(symol_shape);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
  multisliceconcat_node = graph->FindFirstNodeMatchType("MultisliceConcat");
  ASSERT_NE(multisliceconcat_node, nullptr);
  auto op_desc = multisliceconcat_node->GetOpDesc();
  auto attr0 = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  EXPECT_EQ(attr0->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), (Symbol(1)+Symbol(1))}));

  auto attr1 = op_desc->GetOutputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), (Symbol(2)+Symbol(2)+Symbol(2)+Symbol(4))}));


  auto symbol_shape0 = attr0->symbolic_tensor.GetOriginSymbolShape();
  auto outshapeout = std::string((symbol_shape0.GetDim(0).Serialize().get()));
  auto outshapeout1 = std::string((symbol_shape0.GetDim(1).Serialize().get()));

  auto symbol_shape1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  auto outshapeout_1 = std::string((symbol_shape1.GetDim(0).Serialize().get()));
  auto outshapeout1_1 = std::string((symbol_shape1.GetDim(1).Serialize().get()));
}

TEST_F(SymbolicShapeInferenceUT, test_sliced_infer_shape) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int64_t> offset = {0, 0, 0};
  std::vector<int64_t> size = {3, 3, 3};

  auto sliceD = EsSliceD(data0, offset.data(), offset.size(), size.data(), size.size());
  ASSERT_EQ(EsSetGraphOutput(sliceD, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 6}};
  SetNoStorage(cg, "data0", di, 0);
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {9, 5, 6};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  const std::vector<Expression> expect_symbol_output_shape = {Symbol(3), Symbol(3), Symbol(3)};
  ExpectNodeInfo expect_node("SliceD", expect_symbol_output_shape, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}
TEST_F(SymbolicShapeInferenceUT, InferShapeForBatchMatmul_DimNumCheckSUCCESS) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto matmul = EsBatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, 1, -1, 3}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, -1, 2};
  SetNoStorage(cg, "data1", di, 1);

  auto matmul_node = cg->FindFirstNodeMatchType("BatchMatMulV2");
  ASSERT_NE(matmul_node, nullptr);
  auto matmul_op_desc1 = matmul_node->GetOpDesc();
  SetBatchMatMulV2IrAttrs(cg);
  matmul_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, 2}));
  matmul_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, 2}));
  matmul_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 1, 4, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  dlog_setlevel(-1, 0, 0);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  dlog_setlevel(-1, 3, 0);
}

TEST_F(SymbolicShapeInferenceUT, InferShapeForBatchMatmul_DimNumCheckFailed) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto matmul = EsBatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, 1, -1, 3}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, -1, 2};
  SetNoStorage(cg, "data1", di, 1);

  auto matmul_node = cg->FindFirstNodeMatchType("BatchMatMulV2");
  ASSERT_NE(matmul_node, nullptr);
  auto matmul_op_desc1 = matmul_node->GetOpDesc();
  SetBatchMatMulV2IrAttrs(cg);
  matmul_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 2}));
  matmul_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 2}));
  matmul_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 1, 4, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  dlog_setlevel(-1, 0, 0);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  dlog_setlevel(-1, 3, 0);
}

TEST_F(SymbolicShapeInferenceUT, InferShapeForBatchMatmul_UnknowRank) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);

  auto matmul = EsBatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(EsSetGraphOutput(matmul, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, 1, -1, 3}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, -1, 2};
  SetNoStorage(cg, "data1", di, 1);

  auto matmul_node = cg->FindFirstNodeMatchType("BatchMatMulV2");
  ASSERT_NE(matmul_node, nullptr);
  auto matmul_op_desc1 = matmul_node->GetOpDesc();
  SetBatchMatMulV2IrAttrs(cg);
  matmul_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-2}));
  matmul_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-2}));
  matmul_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 1, 4, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  dlog_setlevel(-1, 0, 0);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  dlog_setlevel(-1, 3, 0);
}

REG_OP(Repeat)
  .INPUT(x, TensorType::ALL())
  .INPUT(repeat_times, TensorType::ALL())
  .OUTPUT(y, TensorType::ALL())
  .OP_END_FACTORY_REG(Repeat)

/*
 *
 *                           data0
 *                             |
 *                            abs
 *                            |
 *             -----------------------------------
 *      data1  |   data2   |   data3  |   data4  |
 *        |   /      |    /      |   /      |   /
 *        | /        |  /        |  /       |  /
 *     repeat1     repeat2      repeat3    repeat4
 *           |      |                 |    |
 *            \     /                  \  /
 *             mul1                    add1
 *               |---------    ----------|
 *                        |    |
 *                       Netoutput
 */
namespace {
REG_OP(Repeat)
.INPUT(x, TensorType::ALL())
.INPUT(repeat_times, TensorType::ALL())
.OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Repeat)

IMPL_OP(Repeat).InputsDataDependency({1}); // repeat归属自定义二类算子，符号化推导需要获取
graphStatus TestRepeatInferSymbolShapeFunc(gert::InferSymbolShapeContext *context) {
  auto input0 = context->GetInputSymbolShape(0);
  GE_ASSERT_NOTNULL(input0);
  auto input1 = context->GetInputSymbolTensor(1);
  GE_ASSERT_NOTNULL(input1);
  auto symbol_value = input1->GetSymbolicValue();
  if (symbol_value == nullptr) {
    return UNSUPPORTED;
  }
  auto output = context->GetOutputSymbolShape(0);
  *output = *input0;
  Expression expr(Symbol(0));
  for (const auto &sym : *symbol_value) {
    expr = expr + sym;
  }
  GE_ASSERT_TRUE(!output->GetDims().empty());
  output->MutableDim(0) = expr;
  return ge::SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Repeat).InferSymbolShape(TestRepeatInferSymbolShapeFunc);
}
TEST_F(SymbolicShapeInferenceUT, test_symbolize_value_and_repeat_infer) {
  dlog_setlevel(0, 0, 0);
  auto data0 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).
      OutCnt(1).OutNames({"y"}).Build("data0");
  auto data1 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {192}).OutCnt(1).
      OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 2).TensorDesc(FORMAT_ND, DT_INT64, {192}).OutCnt(1).
      OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 3).TensorDesc(FORMAT_ND, DT_UINT32, {192}).OutCnt(1).
      OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 4).TensorDesc(FORMAT_ND, DT_UINT64, {192}).OutCnt(1).
      OutNames({"y"}).Build("data4");

  auto repeat1 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat1");
  auto repeat2 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat2");
  auto repeat3 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat3");
  auto repeat4 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat4");
  auto abs = OP_CFG("Abs").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(1).OutCnt(1).OutNames({"y"}).
      Build("abs");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("add");
  auto mul = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("mul");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat2));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat3));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat4));

    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(repeat2));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(repeat3));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(repeat4));

    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(repeat2)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(repeat3)->EDGE(0, 0)->NODE(mul));
    CHAIN(NODE(repeat4)->EDGE(0, 1)->NODE(mul));

    CHAIN(NODE(add)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(mul)->EDGE(0, 1)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({192, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  GeTensor tensor1(GeTensorDesc(GeShape({192}), FORMAT_ND, DT_INT32));
  vector<int32_t> data_int32(192, 1);
  tensor1.SetData(reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));

  GeTensor tensor2(GeTensorDesc(GeShape({192}), FORMAT_ND, DT_INT64));
  vector<int64_t> data_int64(192, 1);
  tensor2.SetData(reinterpret_cast<uint8_t *>(data_int64.data()), data_int64.size() * sizeof(int64_t));

  GeTensor tensor3(GeTensorDesc(GeShape({192}), FORMAT_ND, DT_UINT32));
  vector<uint32_t> data_uint32(192, 2);
  tensor3.SetData(reinterpret_cast<uint8_t *>(data_uint32.data()), data_uint32.size() * sizeof(uint32_t));

  GeTensor tensor4(GeTensorDesc(GeShape({192}), FORMAT_ND, DT_UINT64));
  vector<uint64_t> data_uint64(192, 2);
  tensor4.SetData(reinterpret_cast<uint8_t *>(data_uint64.data()), data_uint64.size() * sizeof(uint64_t));
  std::vector<GeTensor> input_vec = {tensor0, tensor1, tensor2, tensor3, tensor4};
   ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, {tensor0, tensor1, tensor2, tensor3, tensor4}), ge::SUCCESS);
   SymbolicShapeInference ssi;
   ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
   auto attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
   EXPECT_NE(attr, nullptr);
   ShapeEnvGuarder guard(attr);
   auto repeat1_node = graph->FindNode("repeat1");
   ASSERT_NE(repeat1_node, nullptr);
   auto op_desc1 = repeat1_node->GetOpDesc();
   ASSERT_NE(op_desc1, nullptr);
   auto attr1 = op_desc1->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
   ASSERT_NE(attr1, nullptr);
   auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape().GetDim(0);
   int64_t hint = -1;
   EXPECT_EQ(symbol_expr1.GetHint(hint), true);
   EXPECT_EQ(hint, 192);
   auto repeat3_node = graph->FindNode("repeat3");
   ASSERT_NE(repeat3_node, nullptr);
   auto opdesc3 = repeat3_node->GetOpDesc();
   ASSERT_NE(opdesc3, nullptr);
   auto attr3 = opdesc3->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
   ASSERT_NE(attr3, nullptr);
   auto symbol_expr3 = attr3->symbolic_tensor.GetOriginSymbolShape().GetDim(0);
   EXPECT_EQ(symbol_expr3.GetHint(hint), true);
   EXPECT_EQ(hint, 192 * 2);
}

TEST_F(SymbolicShapeInferenceUT, test_symbolize_value_not_support_placement_device) {
  auto data0 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).
      OutCnt(1).OutNames({"y"}).Build("data0");
  auto data1 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).OutCnt(1).
      OutNames({"y"}).Build("data1");

  auto repeat1 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({192, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  GeTensor tensor1(GeTensorDesc(GeShape({192}), FORMAT_ND, DT_INT32));
  vector<int32_t> data_int32(192, 1);
  tensor1.SetData(reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));
  tensor1.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, {tensor0, tensor1}), ge::SUCCESS);
  auto repeat = graph->FindFirstNodeMatchType("Repeat");
  ASSERT_NE(repeat, nullptr);
  auto opdesc = repeat->GetOpDesc();
  ASSERT_NE(opdesc, nullptr);
  auto attr = opdesc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeInferenceUT, EmptyTensorSymbolic) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetGraphOutput(data0, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, -1}};
  SetNoStorage(cg, "data_0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {0, 0, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbols = shape_env_attr->GetAllSym2Src();
  ASSERT_EQ(symbols.size(), 3);
  auto sym2 = symbols[0].first;
  auto sym1 = symbols[1].first;
  auto sym0 = symbols[2].first;
  const auto guards = shape_env_attr->GetAllSymbolCheckInfos();
  EXPECT_EQ(guards.size(), 3);
  EXPECT_EQ(shape_env_attr->HasSymbolCheckInfo(sym::Eq(sym0, Symbol(0))), true);
  EXPECT_EQ(shape_env_attr->HasSymbolCheckInfo(sym::Eq(sym1, Symbol(0))), true);
  EXPECT_EQ(shape_env_attr->HasSymbolCheckInfo(sym::Ne(sym2, Symbol(0))), true);

  // 化简能力
  ge::ShapeEnvGuarder shape_env_guarder(shape_env_attr);
  auto expr1 = sym::Ceiling(sym0 * Symbol(5)) + sym::Min(Symbol(2), sym0) * sym::Max(Symbol(2), sym0);
  EXPECT_EQ(expr1.Simplify(), Symbol(0));
  EXPECT_EQ(std::string(expr1.Simplify().Serialize().get()), "0");
  auto expr2 = sym::Min(sym::Pow(sym0, Symbol(2)), sym::Max(Symbol(3), sym0) * sym0);
  EXPECT_EQ(expr2.Simplify(), Symbol(0));
  EXPECT_EQ(std::string(expr2.Simplify().Serialize().get()), "0");
  auto expr3 = sym::Floor(sym::Neg(sym0) + Symbol(2)) - sym::Mod(Symbol(5), Symbol(3));
  EXPECT_EQ(expr3.Simplify(), Symbol(0));
  EXPECT_EQ(std::string(expr3.Simplify().Serialize().get()), "0");
  auto expr4 = sym::Abs(sym0 + Symbol(2)) / Symbol(2) * sym0;
  EXPECT_EQ(expr4.Simplify(), Symbol(0));
  EXPECT_EQ(std::string(expr4.Simplify().Serialize().get()), "0");
}

TEST_F(SymbolicShapeInferenceUT, test_symbolize_value_not_support_float16) {
  auto data0 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).
      OutCnt(1).OutNames({"y"}).Build("data0");
  auto data1 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT16, {192}).OutCnt(1).
      OutNames({"y"}).Build("data1");

  auto repeat1 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({192, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  GeTensor tensor1(GeTensorDesc(GeShape({192}), FORMAT_ND, DT_FLOAT16));
  vector<float> data_int32(192, 1);
  tensor1.SetData(reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));
  tensor1.MutableTensorDesc().SetPlacement(Placement::kPlacementDevice);

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, {tensor0, tensor1}), ge::SUCCESS);
  auto repeat = graph->FindFirstNodeMatchType("Repeat");
  ASSERT_NE(repeat, nullptr);
  auto opdesc = repeat->GetOpDesc();
  ASSERT_NE(opdesc, nullptr);
  auto attr = opdesc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeInferenceUT, test_symbolize_value_not_support_non_const_size) {
  auto data0 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).
      OutCnt(1).OutNames({"y"}).Build("data0");
  auto data1 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {-1}).OutCnt(1).
      OutNames({"y"}).Build("data1");

  auto repeat1 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({258, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  vector<int32_t> data_int32(258, 1);

  GeTensor tensor1(GeTensorDesc(GeShape({258}), FORMAT_ND, DT_INT32), reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));
  tensor1.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, {tensor0, tensor1}), ge::SUCCESS);
  auto repeat = graph->FindFirstNodeMatchType("Repeat");
  ASSERT_NE(repeat, nullptr);
  auto opdesc = repeat->GetOpDesc();
  ASSERT_NE(opdesc, nullptr);
  auto attr = opdesc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeInferenceUT, test_symbolize_value_not_support_compute_size_greater_than_max_size) {
  auto data0 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).
      OutCnt(1).OutNames({"y"}).Build("data0");
  auto data1 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1025}).OutCnt(1).
      OutNames({"y"}).Build("data1");

  auto repeat1 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("repeat1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({1025, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  vector<int32_t> data_int32(1025, 1);

  GeTensor tensor1(GeTensorDesc(GeShape({1025}), FORMAT_ND, DT_INT32), reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));
  tensor1.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, {tensor0, tensor1}), ge::SUCCESS);
  auto repeat = graph->FindFirstNodeMatchType("Repeat");
  ASSERT_NE(repeat, nullptr);
  auto opdesc = repeat->GetOpDesc();
  ASSERT_NE(opdesc, nullptr);
  auto attr = opdesc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

// data0 为非-2， 将泛化输出shape
// data1 为-2， 泛化输出shape
TEST_F(SymbolicShapeInferenceUT, test_symbolize_shape_support_mius_2) {
  auto data0 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).
      OutCnt(1).OutNames({"y"}).Build("data0");
  auto data1 = OP_CFG("Data").InCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT16, {-2}).OutCnt(1).
      OutNames({"y"}).Build("data1");

  auto add1 = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).
      Build("add1");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add1));
    CHAIN(NODE(add1)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({1025, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  vector<int32_t> data_int32(1025, 1);

  GeTensor tensor1(GeTensorDesc(GeShape({1025, 2}), FORMAT_ND, DT_INT32), reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));
  tensor1.MutableTensorDesc().SetPlacement(Placement::kPlacementHost);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, {tensor0, tensor1}), ge::SUCCESS);

  auto data0_node = graph->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  auto op_desc0 = data0_node->GetOpDesc();
  ASSERT_NE(op_desc0, nullptr);
  auto attr0 = op_desc0->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);

  auto data1_node = graph->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  auto op_desc1 = data1_node->GetOpDesc();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = op_desc1->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_expr1.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_expr1.GetDim(0).Serialize().get()), "s2");
  EXPECT_EQ(std::string(symbol_expr1.GetDim(1).Serialize().get()), "s3");
}

TEST_F(SymbolicShapeInferenceUT, test_repeat_infer) {
  auto data0 = OP_CFG("Data")
                .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 2, 3, 4})
                .InCnt(0)
                .OutCnt(1)
                .InNames({"x"})
                .OutNames({"y"})
                .Build("data0");
  auto data1 = OP_CFG("Data")
              .TensorDesc(FORMAT_ND, DT_FLOAT, {4})
              .InCnt(0)
              .OutCnt(1)
              .InNames({"x"})
              .OutNames({"y"})
              .Build("data1");
  auto repeat1 = OP_CFG("Repeat").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).Build("repeat1");

  DEF_GRAPH(graph) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };

  auto cg = ToComputeGraph(graph);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape0({
    Symbol(4),
    Symbol(2),
    Symbol(3),
    Symbol(4)
  });
  data_op_desc0->MutableOutputDesc(0)
  ->template GetOrCreateAttrsGroup<SymbolicDescAttr>()
  ->symbolic_tensor.SetSymbolShape(data_symbol_shape0);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto sym0 = Symbol(1);
  auto sym1 = Symbol(2);
  auto sym2 = Symbol(3);
  auto sym3 = Symbol(4);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);
  ptr->emplace_back(sym1);
  ptr->emplace_back(sym2);
  ptr->emplace_back(sym3);

  gert::SymbolShape data_symbol_shape1({
    Symbol(4)
  });

  data_node1->GetOpDescBarePtr()
      ->MutableOutputDesc(0)
      ->template GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolShape(data_symbol_shape1);
  data_node1->GetOpDescBarePtr()
    ->MutableOutputDesc(0)
    ->template GetOrCreateAttrsGroup<SymbolicDescAttr>()
    ->symbolic_tensor.SetSymbolicValue(std::move(ptr));

  ExpectNodeInfo expect_node1("repeat1", {Symbol(10), Symbol(2), Symbol(3), Symbol(4)}, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

//
// ┌────────┐  (0,0)   
// | const_0│ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │data_1  │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘          
// param: [2, 4, 3, 2], indice: [2, 3], axis: 2 batch_dim = 0
TEST_F(SymbolicShapeInferenceUT, test_gather_symbol_infer_batch_dim_0_success) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int32_t> indice_const_data = {2, 2, 1, 0, 0, 1};
  std::vector<int64_t> indice_const_dim = {2, 3};
  auto indice_const = EsCreateConstInt32(graph_, indice_const_data.data(),
      indice_const_dim.data(), indice_const_dim.size());
  auto axis_const = EsCreateScalarInt32(graph_, 2);
  auto gather_v2 = EsGatherV2(data0, indice_const, axis_const, 0, false, false);
  ASSERT_EQ(EsSetGraphOutput(gather_v2, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 3, -1}};
  SetNoStorage(cg, "data_0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 4, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);

  std::vector<Expression> expect_dim = {Symbol("s0"), Symbol("s1"), Symbol(2), Symbol(3), Symbol("s2")};
  ExpectNodeInfo expect_node1(GATHERV2, expect_dim, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

//
// ┌────────┐  (0,0)   
// | const_0│ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │data_1  │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘          
// param: [2, 4, 3, 2], indice: [2, 4, 2], axis: 2 batch_dim = 2
TEST_F(SymbolicShapeInferenceUT, test_gather_symbol_infer_batch_dim_2_success) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int32_t> indice_const_data = {2, 2, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 0, 2};
  std::vector<int64_t> indice_const_dim = {2, 4, 2};
  auto indice_const = EsCreateConstInt32(graph_, indice_const_data.data(),
      indice_const_dim.data(), indice_const_dim.size());
  auto axis_const = EsCreateScalarInt32(graph_, 2);
  auto gather_v2 = EsGatherV2(data0, indice_const, axis_const, 2, false, false);
  ASSERT_EQ(EsSetGraphOutput(gather_v2, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT64, {2, 4, 3, 2}};
  SetNoStorage(cg, "data_0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 4, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);

  std::vector<Expression> expect_dim = {Symbol(2), Symbol(4), Symbol(2), Symbol(2)};
  ExpectNodeInfo expect_node1(GATHERV2, expect_dim, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

//
// ┌────────┐  (0,0)   
// | data_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘          
// param: [2, 2, 3, 2], indice: [2, 3], axis: 2 batch_dim = 1
TEST_F(SymbolicShapeInferenceUT, test_gather_symbol_infer_batch_dim_1_success) {
  std::vector<int32_t> param_const_data;
  for (size_t i = 0UL; i < 24; i++) {
    param_const_data.emplace_back(i);
  }
  std::vector<int64_t> param_const_dim = {2, 2, 3, 2};
  auto param_const = EsCreateConstInt32(graph_, param_const_data.data(),
      param_const_dim.data(), param_const_dim.size());
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  auto axis_const = EsCreateScalarInt32(graph_, 2);
  auto gather_v2 = EsGatherV2(param_const, data0, axis_const, 1, false, false);
  ASSERT_EQ(EsSetGraphOutput(gather_v2, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1}};
  SetNoStorage(cg, "data_0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);

  std::vector<Expression> expect_dim = {Symbol(2), Symbol(2), Symbol("s1"), Symbol(2)};
  ExpectNodeInfo expect_node1(GATHERV2, expect_dim, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

//
// ┌────────┐  (0,0)   
// | data_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘          
// param: [2, 2, 3, 2], indice: [2, 3], axis: 2 batch_dim = 1
TEST_F(SymbolicShapeInferenceUT, test_gather_symbol_infer_data_axis_unsupport) {
  std::vector<int32_t> param_const_data;
  for (size_t i = 0UL; i < 24; i++) {
    param_const_data.emplace_back(i);
  }
  std::vector<int64_t> param_const_dim = {2, 2, 3, 2};
  auto param_const = EsCreateConstInt32(graph_, param_const_data.data(),
      param_const_dim.data(), param_const_dim.size());
  std::vector<int32_t> indice_const_data = {2, 2, 1, 0, 0, 1};
  std::vector<int64_t> indice_const_dim = {2, 3};
  auto indice_const = EsCreateConstInt32(graph_, indice_const_data.data(),
      indice_const_dim.data(), indice_const_dim.size());
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data_0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  auto gather_v2 = EsGatherV2(param_const, indice_const, data0, 0, false, false);
  ASSERT_EQ(EsSetGraphOutput(gather_v2, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT64, {}};
  SetNoStorage(cg, "data_0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  ge::Shape shape0;
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeInferenceUT, test_BNTrainingReduce_NHWC) {
  auto x = EsCreateGraphInputWithDetails(graph_, 0, "x", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(x, nullptr);
  auto s3 = ge::Symbol("s3");
  auto bntrainingreduce = EsBNTrainingReduce(x);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_NHWC, DT_FLOAT, {-1, -1, -1, -1}};
  SetNoStorage(cg, "x", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NHWC, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  auto bnt_node = cg->FindFirstNodeMatchType("BNTrainingReduce");
  ASSERT_NE(bnt_node, nullptr);
  auto op_desc = bnt_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
}

TEST_F(SymbolicShapeInferenceUT, test_BNTrainingReduce_NCHW) {
  auto x = EsCreateGraphInputWithDetails(graph_, 0, "x", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(x, nullptr);
  auto s1 = ge::Symbol("s1");
  auto bntrainingreduce = EsBNTrainingReduce(x);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1}};
  SetNoStorage(cg, "x", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NCHW, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  auto bnt_node = cg->FindFirstNodeMatchType("BNTrainingReduce");
  ASSERT_NE(bnt_node, nullptr);
  auto op_desc = bnt_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
}

TEST_F(SymbolicShapeInferenceUT, test_BNTrainingReduce_NDHWC) {
  auto x = EsCreateGraphInputWithDetails(graph_, 0, "x", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(x, nullptr);
  auto s4 = ge::Symbol("s4");
  auto bntrainingreduce = EsBNTrainingReduce(x);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_NDHWC, DT_FLOAT, {-1, -1, -1, -1, -1}};
  SetNoStorage(cg, "x", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NDHWC, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  auto bnt_node = cg->FindFirstNodeMatchType("BNTrainingReduce");
  ASSERT_NE(bnt_node, nullptr);
  auto op_desc = bnt_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NDHWC);
  op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NDHWC);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s4}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s4}).GetDims());
}

TEST_F(SymbolicShapeInferenceUT, test_BNTrainingReduce_NDC1HWC0) {
  auto x = EsCreateGraphInputWithDetails(graph_, 0, "x", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(x, nullptr);
  auto k1 = ge::Symbol(1);
  auto s2 = ge::Symbol("s2");
  auto s5 = ge::Symbol("s5");
  auto bntrainingreduce = EsBNTrainingReduce(x);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_NDC1HWC0, DT_FLOAT, {-1, -1, -1, -1, -1, -1}};
  SetNoStorage(cg, "x", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NDC1HWC0, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  auto bnt_node = cg->FindFirstNodeMatchType("BNTrainingReduce");
  ASSERT_NE(bnt_node, nullptr);
  auto op_desc = bnt_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NDC1HWC0);
  op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NDC1HWC0);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({k1, k1, s2, k1, k1, s5}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({k1, k1, s2, k1, k1, s5}).GetDims());
}

TEST_F(SymbolicShapeInferenceUT, test_BNTrainingReduce_NCDHW) {
  auto x = EsCreateGraphInputWithDetails(graph_, 0, "x", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(x, nullptr);
  auto s1 = ge::Symbol("s1");
  auto bntrainingreduce = EsBNTrainingReduce(x);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_NCDHW, DT_FLOAT, {-1, -1, -1, -1, -1}};
  SetNoStorage(cg, "x", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NCDHW, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  auto bnt_node = cg->FindFirstNodeMatchType("BNTrainingReduce");
  ASSERT_NE(bnt_node, nullptr);
  auto op_desc = bnt_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NCDHW);
  op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCDHW);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).template GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
}

TEST_F(SymbolicShapeInferenceUT, test_BNTrainingReduce_ND) {
  auto x = EsCreateGraphInputWithDetails(graph_, 0, "x", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(x, nullptr);
  auto bntrainingreduce = EsBNTrainingReduce(x);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(EsSetGraphOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_FLOAT, {-1, -1, -1, -1, -1}};
  SetNoStorage(cg, "x", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  auto bnt_node = cg->FindFirstNodeMatchType("BNTrainingReduce");
  ASSERT_NE(bnt_node, nullptr);
  auto op_desc = bnt_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_ND);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->template GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

/**
 *      Data0            Data1     Data2
 *        |              /         /
 *        \             /         /
 *         \           /         /
 *          \         /         /
 *            \       |         /
 *                  SelectV2
 *                    |
 *                 NetOutput
 */
TEST_F(SymbolicShapeInferenceUT, InferShapeForSelectV2) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data2 = EsCreateGraphInputWithDetails(graph_, 2, "data2", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(data2, nullptr);

  const auto selectv2 = EsSelectV2(data0, data1, data2);
  ASSERT_EQ(EsSetGraphOutput(selectv2, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, -1, 2, 2}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 3, 2, -1};
  SetNoStorage(cg, "data1", di, 1);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data2", di, 2);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 2, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3, 2, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {2, 3, 2, 2};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  std::vector<Expression> expect_dim = {Symbol("s4"), Symbol(3), Symbol(2), Symbol(2)};
  ExpectNodeInfo expect_node1("SelectV2", expect_dim, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, InferShapeForGather) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_EQ(EsSetOriginSymbolShape(data0, std::vector<const char *>({"s0", "s1", "s2", "s3"}).data(), 4), 0);
  ASSERT_NE(data0, nullptr);
  std::vector<int64_t> const_data = {2, 3};
  auto const0 = EsCreateVectorInt64(graph_, const_data.data(), 2);
  ASSERT_NE(const0, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto gather = EsGather(data0, const0, false, 0,true, true);
  ASSERT_EQ(EsSetGraphOutput(gather, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("Gather");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SetGatherIrAttrs(cg, 0);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  std::vector<Expression> expect_dim = {Symbol(2), s1, s2, s3};
  ExpectNodeInfo expect_node1("Gather", expect_dim, {}, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, {}), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, InferShapeForBroadCastToGraphWithGuard) {
  auto data0 = EsCreateGraphInputWithDetails(graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  std::vector<int32_t> const_data0 = {3, 2, 3, 4};
  std::vector<int64_t> const_dim = {4};
  auto const0 = EsCreateConstInt32(graph_, const_data0.data(), const_dim.data(), const_dim.size());
  ASSERT_NE(data0, nullptr);
  auto broadcast_to = EsBroadcastTo(data0, const0);
  ASSERT_EQ(EsSetGraphOutput(broadcast_to, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  DataInfo di = {ge::FORMAT_ND, DT_INT64, {-1, 3, 1}};
  SetNoStorage(cg, "data0", di, 0);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {1, 3, 1};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  const std::set<std::string> expect_guard = {"LogicOr(ExpectEq(1, s0), ExpectEq(2, s0))", "ExpectNe(0, s0)"};
  std::vector<Expression> expect_output_shape = {Symbol(3), Symbol(2), Symbol(3), Symbol(4)};

  ExpectNodeInfo expect_node1("BroadcastTo", expect_output_shape, expect_guard, {}, {});
  std::vector<ExpectNodeInfo> expect_node_vec;
  expect_node_vec.push_back(expect_node1);
  ASSERT_EQ(RunSymbolInferenceTest(cg, expect_node_vec, input_vec), SUCCESS);
}

/* 
 * if的条件输入是data
 *  
 * then_grpah:
 *   data -> sqrt -> output
 * else_graph:
 *   data -> sqrt -> output
 * 
 * data1 -> if -> output
 *           |
 * data  ----
 * 
 */
TEST_F(SymbolicShapeInferenceUT, NestIfGraphTest) {
  EnableSliceScheduleEnv();
  auto root_graph = gert::ShareGraph::BuildNestIfGraph();

  // data
  DataInfo di0 = {FORMAT_NCHW, DT_INT32, {}};
  SetNoStorage(root_graph, "data_0", di0, 0);
  // data1
  DataInfo di1 = {FORMAT_NCHW, DT_INT32, {2, 3}};
  SetNoStorage(root_graph, "data_1", di1, 1);

  // data
  std::vector<GeTensor> input_vec;
  auto input0 = BuildGeTensor<int32_t, DT_INT32>({}, {1});
  auto input1 =  BuildGeTensor<int32_t, DT_INT32>({2, 3}, {});
  input_vec.emplace_back(input0);
  input_vec.emplace_back(input1);

  SymbolicShapeSymbolizer symboilzer;
  ASSERT_EQ(symboilzer.Symbolize(root_graph, input_vec), SUCCESS);
  ASSERT_EQ(SymbolicInfoPreProcessor::Run(root_graph, input_vec), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(root_graph), SUCCESS);

  ASSERT_EQ(root_graph->FindNode("if1"), nullptr);

  auto sqrt_node = root_graph->FindNode("then_subgraph_sqrt1");
  ASSERT_NE(sqrt_node, nullptr);
  DisableSliceScheduleEnv();
}

TEST_F(SymbolicShapeInferenceUT, NestCaseGraphTest) {
  EnableSliceScheduleEnv();
  auto root_graph = gert::ShareGraph::BuildNestCaseGraph();

  // data
  DataInfo di0 = {FORMAT_NCHW, DT_INT32, {}};
  SetNoStorage(root_graph, "data_0", di0, 0);
  // data1
  DataInfo di1 = {FORMAT_NCHW, DT_INT32, {2, 3}};
  SetNoStorage(root_graph, "data_1", di1, 1);

  // data
  std::vector<GeTensor> input_vec;
  auto input0 = BuildGeTensor<int32_t, DT_INT32>({}, {1});
  auto input1 =  BuildGeTensor<int32_t, DT_INT32>({2, 3}, {});
  input_vec.emplace_back(input0);
  input_vec.emplace_back(input1);

  SymbolicShapeSymbolizer symboilzer;
  ASSERT_EQ(symboilzer.Symbolize(root_graph, input_vec), SUCCESS);
  ASSERT_EQ(SymbolicInfoPreProcessor::Run(root_graph, input_vec), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(root_graph), SUCCESS);

  ASSERT_EQ(root_graph->FindNode("case1"), nullptr);

  auto sqrt_node = root_graph->FindNode("batch2_subgraph_sqrt1");
  ASSERT_NE(sqrt_node, nullptr);
  DisableSliceScheduleEnv();
}

// if的条件输入是其它算子的输出
TEST_F(SymbolicShapeInferenceUT, NestIfGraph1Test) {
  EnableSliceScheduleEnv();
  auto root_graph = gert::ShareGraph::BuildNestIfGraph1();

  // data
  DataInfo di0 = {FORMAT_NCHW, DT_INT32, {}};
  SetNoStorage(root_graph, "data_0", di0, 0);
  // data1
  DataInfo di1 = {FORMAT_NCHW, DT_INT32, {}};
  SetNoStorage(root_graph, "data_1", di1, 1);
  // data2
  DataInfo di2 = {FORMAT_NCHW, DT_INT32, {2, 3}};
  SetNoStorage(root_graph, "data_2", di2, 2);


  // data
  std::vector<GeTensor> input_vec;
  auto input0 = BuildGeTensor<int32_t, DT_INT32>({}, {1});
  auto input1 = BuildGeTensor<int32_t, DT_INT32>({}, {1});
  auto input2 =  BuildGeTensor<int32_t, DT_INT32>({2, 3}, {});
  input_vec.emplace_back(input0);
  input_vec.emplace_back(input1);
  input_vec.emplace_back(input2);

  SymbolicShapeSymbolizer symboilzer;
  ASSERT_EQ(symboilzer.Symbolize(root_graph, input_vec), SUCCESS);
  ASSERT_EQ(SymbolicInfoPreProcessor::Run(root_graph, input_vec), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(root_graph), SUCCESS);
  
  ASSERT_NE(root_graph->FindNode("if1"), nullptr);

  DisableSliceScheduleEnv();
}

TEST_F(SymbolicShapeInferenceUT, AippOpTest) {
  auto data0 = EsCreateGraphInputWithDetails(
      graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(
      graph_, 1, "data1", AIPPDATA, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  auto aipp = EsAipp(data0, data1, "");
  ASSERT_EQ(EsSetGraphOutput(aipp, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT32, {-1, -1, -1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, 1, -1};
  SetNoStorage(cg, "data1", di, 1);
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, MulitBatchOpTest) {
  auto data0 = EsCreateGraphInputWithDetails(
      graph_, 0, "data0", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(
      graph_, 1, "data1", nullptr, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);

  std::vector<int32_t> dims{0, 1, 2};
  int64_t dims_size = 3;
  auto const_int32_list = EsCreateConstInt32(graph_, dims.data(), &dims_size, 1);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  auto mapindex = EsMapIndex(data1, const_int32_list, nullptr, false);
  auto add = EsAdd(data0, mapindex);
  ASSERT_EQ(EsSetGraphOutput(add, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT32, {-1, -1, -1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data1", di, 1);
  auto data1_node = cg->FindNode("data1");
  (void)AttrUtils::SetBool(data1_node->GetOpDesc(), "_is_multi_batch_shape_data", true);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, RefDataCopyOpTest) {
  auto data0 = EsCreateGraphInputWithDetails(
      graph_, 0, "data0", REFDATA, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(
      graph_, 1, "data1", REFDATA, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  auto assign = EsAssign(data0, data0, true, true);
  auto add = EsAdd(assign, data1);
  ASSERT_EQ(EsSetGraphOutput(add, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT32, {-1, -1, -1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data1", di, 1);
  auto data1_node = cg->FindNode("data1");
  (void)AttrUtils::SetStr(data1_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "data");

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), SUCCESS);
}

TEST_F(SymbolicShapeInferenceUT, DummyShapeTest) {
  auto data0 = EsCreateGraphInputWithDetails(
      graph_, 0, "data0", REFDATA, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  auto data1 = EsCreateGraphInputWithDetails(
      graph_, 1, "data1", REFDATA, C_DataType::C_DT_FLOAT, C_Format::C_FORMAT_ND, nullptr, 0);
  ASSERT_NE(data0, nullptr);
  ASSERT_NE(data1, nullptr);
  auto add = EsAdd(data0, data1);
  ASSERT_EQ(EsSetGraphOutput(add, 0), 0);
  auto graph = std::unique_ptr<Graph>(reinterpret_cast<Graph *>(EsBuildGraphAndReset(graph_)));
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  DataInfo di = {ge::FORMAT_ND, DT_INT32, {-1, -1, -1, -1}};
  SetNoStorage(cg, "data0", di, 0);
  di.shape = {-1, -1, -1, -1};
  SetNoStorage(cg, "data0", di, 0);
  SetNoStorage(cg, "data1", di, 1);

  std::vector<ge::GeTensor> input_vec;
  ge::Shape shape0({2, 3, 4, 4});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};

  ge::Shape shape1({-3});
  ge::TensorDesc td1{shape0, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  ASSERT_NE(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), SUCCESS);
}

// 动态分档场景符号化推导
TEST_F(SymbolicShapeInferenceUT, MultiBatchInferTest) {
  GetLocalOmgContext().need_multi_batch = true;
  auto graph = gert::ShareGraph::BuildMultiBatchShapesGraph();

  std::vector<ge::GeTensor> input_vec;
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, input_vec), SUCCESS);

  for (auto &subgraph : graph->GetAllSubgraphs()) {
    for (auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() == DATA) {
        const auto attr = node->GetOpDesc()->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
        ASSERT_NE(attr, nullptr);
      }
    }
  }
  GetLocalOmgContext().need_multi_batch = false;
}
} // namespace ge
