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
#include <gtest/gtest.h>
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "es_ge_test_ops.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "framework/common/types.h"
#include "faker/space_registry_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/node_adapter.h"
#include "ge_running_env/include/ge_running_env/ge_running_env_faker.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "ge_running_env/op_reg.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "faker/space_registry_faker.h"
#include "ge/ut/ge/graph/optimize/symbolic/expect_node_info_check_test.h"
#include "common/share_graph.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_pre_processor.h"
#include "api/aclgrph/option_utils.h"
#include "ge_local_context.h"
#include "register/optimization_option_registry.h"
#include "common/context/local_context.h"

namespace ge {
using ge::es::EsTensorHolder;
bool EnableSliceSchedule() { // 桩函数
  const char_t *auto_fuse_options = nullptr;
  MM_SYS_GET_ENV(MM_ENV_AUTOFUSE_FLAGS, auto_fuse_options);
  if (auto_fuse_options == nullptr) {
    return false;
  }
  std::string option = std::string(auto_fuse_options);
  return option.find("--experimental_enable_jit_executor_v2=true") != std::string::npos;
}
class SymbolicShapeInferenceST : public testing::Test {
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
    builder_ = std::make_unique<ge::es::EsGraphBuilder>("Hello");
  }
  void TearDown() override {
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
    GetThreadLocalContext().GetOo().Initialize(GetThreadLocalContext().GetAllOptions(),
                                               OptionRegistry::GetInstance().GetRegisteredOptTable());
    builder_.reset();
  }
  std::unique_ptr<ge::es::EsGraphBuilder> builder_;
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

TEST_F(SymbolicShapeInferenceST, InferShapeForSimpleGraph) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
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
IMPL_OP(BiasAddGrad).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("data_format", "NHWC");
IMPL_OP(ConcatV2D).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("concat_dim", static_cast<int64_t>(0));
IMPL_OP(Conv2DBackpropInputD).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("input_size", std::vector<int64_t>({1, 1, 1, 1}));
IMPL_OP(Conv2DBackpropFilterD).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("filter_size", std::vector<int64_t>({1, 1, 1, 1}));
IMPL_OP(Gather).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("validate_indices", true).PrivateAttr("batch_dims", static_cast<int64_t>(0));

TEST_F(SymbolicShapeInferenceST, InferShapeForVariable) {
  auto es_graph = std::unique_ptr<es::EsGraphBuilder>(new es::EsGraphBuilder("cpp_graph"));
  auto variable = es_graph->CreateVariable(0, "Variable0");
  auto relu = es::Relu(variable);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = es_graph->BuildAndReset();
  ASSERT_NE(NodeAdapter::GNode2Node(*variable.GetProducer())->GetOpDesc(), nullptr);
  ASSERT_NE(NodeAdapter::GNode2Node(*variable.GetProducer())->GetOpDesc()->MutableInputDesc(0), nullptr);
  // variable类节点存在输入的tensor_desc，但是输入上没有symbol_desc_attr
  //  NodeAdapter::GNode2Node(*variable.GetProducer())->GetOpDesc()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto var_node = cg->FindNode("Variable0");
  ASSERT_NE(var_node, nullptr);
  std::vector<int64_t> dims{4, 5, 6};
  var_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(dims));
  var_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape(dims));
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  var_node = cg->FindNode("Variable0");
  ASSERT_NE(var_node, nullptr);

  auto attr = var_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::cout << SymbolicInferUtil::VectorExpressionToStr(attr->symbolic_tensor.GetOriginSymbolShape().GetDims()) <<
      std::endl;
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(4), Symbol(5), Symbol(6)}));
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
} // namespace

TEST_F(SymbolicShapeInferenceST, InferShapeForNoInferFunc) {
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
       ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
       ->symbolic_tensor.SetSymbolShape(symbol_shape);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);

  auto foo1_node = graph->FindNode("FOO1");
  ASSERT_NE(foo1_node, nullptr);
  auto attr = foo1_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol(2), Symbol(2), Symbol(3), Symbol(4)}));

  auto foo2_node = graph->FindNode("FOO2");
  ASSERT_NE(foo1_node, nullptr);
  attr = foo2_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

namespace {
auto const11 = OP_CFG("Const")
               .InCnt(0)
               .OutCnt(1)
               .OutNames({"y"})
               .Build("Const");
auto const22 = OP_CFG("Constant")
               .InCnt(0)
               .OutCnt(1)
               .OutNames({"y"})
               .Build("Constant");
} // namespace

// 如果不是按照IR注册的方式造的node，后续造context时拿不到IR属性
// infer symbol shape时，会尝试从IR中恢复IR属性，这对const\constant节点非常重要，此处添加ut校验
TEST_F(SymbolicShapeInferenceST, InferShapeForConstantWithRecoverIrAttr) {
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
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);

  auto const11_node = graph->FindNode("Const");
  ASSERT_NE(const11_node, nullptr);
  auto attr = const11_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol(2), Symbol(2), Symbol(3), Symbol(4)}));

  auto const22_node = graph->FindNode("Constant");
  ASSERT_NE(const22_node, nullptr);
  attr = const22_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol(2), Symbol(2), Symbol(3), Symbol(4)}));
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
TEST_F(SymbolicShapeInferenceST, InferShapeForDecomposedGraph) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto pow = es::Pow(data0, data1);
  auto tanh = es::Tanh(data1);
  auto squaredD = es::SquaredDifference(pow, tanh);
  auto neg = es::Neg(squaredD);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(neg, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Neg_3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

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
TEST_F(SymbolicShapeInferenceST, simple_symbolic_kernel_compute) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_1");
  auto data2 = builder_->CreateInput(2, "data_2");
  auto const1 = builder_->CreateScalar(static_cast<int32_t>(0));

  auto shape_1 = es::Shape(data0, 3); // DT_INT32
  auto gather_1 = es::GatherV2(shape_1, const1, const1, 0, false, false);

  auto shape_2 = es::Shape(data1, 3); // DT_INT32
  auto gather_2 = es::GatherV2(shape_2, const1, const1, 0, false, false);

  std::vector<EsTensorHolder> esb;
  esb.push_back(gather_1);
  esb.push_back(gather_2);
  auto pack1 = es::Pack(esb, 0, 2);

  auto reshape = es::Reshape(data2, pack1, 0, -1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reshape, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data_1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("data_2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3};
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
  std::vector<int64_t> dims_vec2 = {1, 1, 4};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  auto reshape_op_0 = cg->FindFirstNodeMatchType("Reshape");
  ASSERT_NE(reshape_op_0, nullptr);
  auto reshape_op_desc0 = reshape_op_0->GetOpDesc();
  reshape_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  reshape_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT64);

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto gather0 = cg->FindNode("GatherV2_2");
  ASSERT_NE(gather0, nullptr);
  auto gather_0_op_desc = gather0->GetOpDesc();
  auto attr_0 = gather_0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr_0, nullptr);
  const auto result_dim_0 = attr_0->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim_0.empty(), true);
  std::vector<Expression> expect_symbolic_value_0 = {Symbol("s0")};
  const auto result_value_0 = attr_0->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value_0, nullptr);
  EXPECT_EQ(*result_value_0, expect_symbolic_value_0);

  auto gather1 = cg->FindNode("GatherV2_4");
  ASSERT_NE(gather1, nullptr);
  auto gather_1_op_desc = gather1->GetOpDesc();
  auto attr_1 = gather_1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr_1, nullptr);
  const auto result_dim_1 = attr_1->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim_1.empty(), true);
  std::vector<Expression> expect_symbolic_value_1 = {Symbol("s2")};
  const auto result_value_1 = attr_1->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value_1, nullptr);
  EXPECT_EQ(*result_value_1, expect_symbolic_value_1);

  auto reshape_node = cg->FindNode("Reshape_6");
  ASSERT_NE(reshape_node, nullptr);
  auto reshape_node_op_desc = reshape_node->GetOpDesc();
  auto attr_reshape = reshape_node_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr_reshape, nullptr);
  std::vector<Expression> expect_shape_reshape = {Symbol("s0"), Symbol("s2")};
  const auto result_dim_reshape = attr_reshape->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim_reshape, expect_shape_reshape);

  const auto result_value_reshape = attr_reshape->symbolic_tensor.GetSymbolicValue();
  EXPECT_EQ(result_value_reshape, nullptr);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ Pad      │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ───────————
// └────────┘
TEST_F(SymbolicShapeInferenceST, test_pad_with_symbols_value_but_error_shape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 4}; // // paddings.size != data0.dims * 2 校验报错
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  const auto pad = es::Pad(data0, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferenceST, test_pad_with_vector) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {6};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  const auto pad = es::Pad(data0, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, GatherV2_get_input_failed) {
  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3}; // error shape， paddings 的shape应该是{3, 2} 对应 {inputDimNum ,2}
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto data1 = builder_->CreateInput(0, "data_1");
  auto data2 = builder_->CreateInput(1, "data_2");

  data1.SetOriginSymbolShape(std::vector<const char *>({"(s3 * s4)", "s5"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "(s1 * s3 * s4)"}));

  auto gather_1 = es::GatherV2(const0, data1, data2, 0, false, false);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_1, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, ExpandDims_get_input_failed) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  auto data01 = builder_->CreateInput(1, "data_0");
  data01.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  const auto expandDims = es::ExpandDims(data0, data01);
  NodeAdapter::GNode2Node(*expandDims.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expandDims, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_expanddims_with_dtint32_symbolvalue) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(1));
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);

  const auto expandDims = es::ExpandDims(data0, const0);
  NodeAdapter::GNode2Node(*expandDims.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expandDims, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto expand_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(expand_node, nullptr);
  auto op_desc = expand_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol("s0"), Symbol(1), Symbol("s1"), Symbol("s2")}));
}

TEST_F(SymbolicShapeInferenceST, test_expanddims_with_dtint64_symbolvalue) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(1));
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);

  const auto expandDims = es::ExpandDims(data0, const0);
  NodeAdapter::GNode2Node(*expandDims.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT64);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expandDims, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto expand_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(expand_node, nullptr);
  auto op_desc = expand_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol("s0"), Symbol(1), Symbol("s1"), Symbol("s2")}));
}

TEST_F(SymbolicShapeInferenceST, ConcatV2_get_input_failed) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"128", "32"}));
  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"128", "32"}));

  auto concat_dim = builder_->CreateInput(2, "data_0");
  concat_dim.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  std::vector<EsTensorHolder> inputs = {data0, data1};

  auto concatv2 = es::ConcatV2(inputs, concat_dim, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, Fill_get_input_failed) {
  std::vector<int64_t> const_data = {2, 4, 2};
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto fill = es::Fill(data0, shape1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto fill_node = cg->FindFirstNodeMatchType(FILL);
  ASSERT_NE(fill_node, nullptr);
  auto op_desc = fill_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({1}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, ReduceSum_get_input_failed) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  auto reduce_sum = es::ReduceSum(data0, data0, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_sum, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, Pad_get_input_failed) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  auto pad = es::Pad(data0, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, Tile_get_input_failed) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  std::vector<int64_t> dims{0, 1, 2};
  auto pad = es::Tile(data0, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, Transpose_get_input_failed) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  std::vector<int64_t> dims{0, 1, 2};
  auto pad = es::Transpose(data0, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, Slice_get_input_failed_offsets) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  std::vector<int64_t> dims{0, 1, 2};
  auto pad = es::Slice(data0, data0, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, Slice_get_input_failed_size) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  std::vector<int64_t> dims{0, 1, 2};
  int64_t dims_size = 3;
  auto offsets = builder_->CreateConst(dims, std::vector<int64_t>{dims_size});
  ASSERT_NE(offsets.GetCTensorHolder(), nullptr);
  auto pad = es::Slice(data0, offsets, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_reshape_no_symbolicvalue) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_2");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s3","(s1 * s2)","s0",}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  auto reshape = es::Reshape(data0, data1, 0, -1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reshape, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_reshape_have_not_constvalue) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_2");

  data0.SetOriginSymbolShape(std::vector<const char *>({"s3","(s1 * s2)","s0",}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto sym2 = Symbol("s2");
  auto sym3 = Symbol("s3");
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym1 * sym3);
  ptr->emplace_back(sym2 * sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto reshape = es::Reshape(data0, data1, 0, -1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reshape, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  auto reshape_node = cg->FindFirstNodeMatchType(RESHAPE);
  ASSERT_NE(reshape_node, nullptr);
  auto op_desc = reshape_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s1 * s3, s0 * s2}));
}

TEST_F(SymbolicShapeInferenceST, test_reshape_constvalue_not_valid) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_2");

  data0.SetOriginSymbolShape(std::vector<const char *>({"s3","(s1 * s2)","s0",}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"4"}));

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol(1.1);
  auto sym1 = Symbol(0);
  auto sym2 = Symbol(-1);
  auto sym3 = Symbol(2);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);
  ptr->emplace_back(sym1);
  ptr->emplace_back(sym2);
  ptr->emplace_back(sym3);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT64);

  auto reshape = es::Reshape(data0, data1, 2, 4);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reshape, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reshape_node = NodeAdapter::GNode2Node(*reshape.GetProducer());
  ASSERT_NE(reshape_node, nullptr);
  auto op_desc = reshape_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0)->SetDataType(DT_INT64);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT64);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
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
TEST_F(SymbolicShapeInferenceST, InferShapeForGraphWithNodeNotSupportSymbolInfer) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto pow = es::Pow(data0, data1);
  auto tanh = es::Tanh(data1);
  auto squaredD = es::SquaredDifference(pow, tanh);
  auto neg = es::Neg(squaredD);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(neg, 0), 0);
  auto graph = builder_->BuildAndReset();
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
              squared_difference_node->GetInDataAnchor(1), foo_node), SUCCESS);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto foo_attr = foo_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(foo_attr, nullptr);
  auto neg_node = cg->FindNode("Neg_3");
  ASSERT_NE(neg_node, nullptr);
  auto neg_op_desc = neg_node->GetOpDesc();
  ASSERT_NE(neg_op_desc, nullptr);
  auto neg_attr = neg_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(neg_attr, nullptr);
  auto squared_difference_op_desc = squared_difference_node->GetOpDesc();
  ASSERT_NE(squared_difference_op_desc, nullptr);
  auto sd_input_attr_0 = squared_difference_op_desc->GetInputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(sd_input_attr_0, nullptr);
  ASSERT_EQ(sd_input_attr_0->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
  auto sd_input_attr_1 = squared_difference_op_desc->GetInputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(sd_input_attr_1, nullptr);
  auto sd_output_attr_0 = squared_difference_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(sd_output_attr_0, nullptr);
}

//         ┌────────┐  (0,0)
//         │ data_0 │ ────────
//         └────────┘         |
//                            |
// ┌────────┐  (0,1)   ┌─────────────┐ (0,0)    ┌─────────────┐
// │ data_1 │ ───────> │BatchMatMulV2│ ───────> │ Node_Output │
// └────────┘          └─────────────┘          └─────────────┘
TEST_F(SymbolicShapeInferenceST, test_batchmatmulv2_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3", "s4"}));

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({ "s2", "s4", "s3"}));

  auto bmmv2 = es::BatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bmmv2, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto bmmv2_node = cg->FindFirstNodeMatchType("BatchMatMulV2");
  ASSERT_NE(bmmv2_node, nullptr);
  auto op_desc = bmmv2_node->GetOpDesc();

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 5);
  ASSERT_EQ(out_shape.GetDim(0), Symbol("s0"));
  ASSERT_EQ(out_shape.GetDim(1), Symbol("s1"));
  ASSERT_EQ(out_shape.GetDim(2), Symbol("s2"));
  ASSERT_EQ(out_shape.GetDim(3), Symbol("s3"));
  ASSERT_EQ(out_shape.GetDim(4), Symbol("s3"));
}

//         ┌────────┐  ()
//         │ data_0 │ ────────
//         └────────┘         |
//                            |
// ┌────────┐(-1,-1,-1)┌─────────────┐ (s0,s1,s2)┌─────────────┐
// │ data_1 │ ───────> │     Mul     │ ───────>  │ Node_Output │
// └────────┘          └─────────────┘           └─────────────┘
TEST_F(SymbolicShapeInferenceST, test_mul_input_tensor_scalar_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({}));

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({ "s0", "s1", "s2"}));

  auto mul = es::Mul(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(mul, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto mul_node = cg->FindFirstNodeMatchType("Mul");
  ASSERT_NE(mul_node, nullptr);
  auto op_desc = mul_node->GetOpDesc();

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 3);
  ASSERT_EQ(out_shape.GetDim(0), Symbol("s0"));
  ASSERT_EQ(out_shape.GetDim(1), Symbol("s1"));
  ASSERT_EQ(out_shape.GetDim(2), Symbol("s2"));
}

TEST_F(SymbolicShapeInferenceST, test_batchmatmulv2_2) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0"}));

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0"}));

  auto bmmv2 = es::BatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bmmv2, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto bmmv2_node = cg->FindFirstNodeMatchType("BatchMatMulV2");
  ASSERT_NE(bmmv2_node, nullptr);
  auto op_desc = bmmv2_node->GetOpDesc();

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2);
  ASSERT_EQ(out_shape.GetDim(0), Symbol(1));
  ASSERT_EQ(out_shape.GetDim(1), Symbol(1));
}

TEST_F(SymbolicShapeInferenceST, test_unsupport_subgraph) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3};  // error shape， paddings 的shape应该是{3, 2} 对应 {inputDimNum ,2}
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  const auto pad = es::Pad(data0, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto reshape_node = cg->FindFirstNodeMatchType(PAD);
  reshape_node->GetOpDesc()->AddSubgraphName("subgraph");
  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_concatv2) {
  dlog_setlevel(0, 0, 0);
  std::vector<EsTensorHolder> data_nodes;

  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s2", "s1"}));
  data_nodes.emplace_back(data0);

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s3", "s1"}));
  data_nodes.emplace_back(data1);

  auto data2 = builder_->CreateInput(2, "data_2");
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s4", "s1"}));
  data_nodes.emplace_back(data2);

  auto data3 = builder_->CreateInput(3, "data_3");
  data3.SetOriginSymbolShape(std::vector<const char *>({"1"}));

  auto data_node = NodeAdapter::GNode2Node(*data3.GetProducer());
  auto sym0 = Symbol(1);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);
  auto concatv2 = es::ConcatV2(data_nodes, data3, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  NodeAdapter::GNode2Node(*concatv2.GetProducer())->GetOpDesc()->MutableInputDesc(3)->SetDataType(DT_INT32);
  auto concat_node = cg->FindFirstNodeMatchType("ConcatV2");
  ASSERT_NE(concat_node, nullptr);
  auto op_desc = concat_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  auto s4 = ge::Symbol("s4");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s2+s3+s4, s1}));
  dlog_setlevel(0, 3, 0);
}

REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

IMPL_OP(Concat).InputsDataDependency({0});

TEST_F(SymbolicShapeInferenceST, test_concat) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto funcs = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("Concat");
  funcs->IsInputDataDependency({0});
  std::vector<EsTensorHolder> data_nodes;

  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s2", "s1"}));
  data_nodes.emplace_back(data0);

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s3", "s1"}));
  data_nodes.emplace_back(data1);

  auto data2 = builder_->CreateInput(2, "data_2");
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s4", "s1"}));
  data_nodes.emplace_back(data2);

  auto data3 = builder_->CreateInput(3, "data_3");
  data3.SetOriginSymbolShape(std::vector<const char *>({"1"}));

  auto data_node = NodeAdapter::GNode2Node(*data3.GetProducer());
  auto sym0 = Symbol(1);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto concat = es::Concat(data3, data_nodes, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concat, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  NodeAdapter::GNode2Node(*concat.GetProducer())->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_INT32);
  auto concat_node = cg->FindFirstNodeMatchType("Concat");
  ASSERT_NE(concat_node, nullptr);
  auto op_desc = concat_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  auto s4 = ge::Symbol("s4");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s2+s3+s4, s1}));
}

TEST_F(SymbolicShapeInferenceST, test_concat_dim_failed) {
  std::vector<EsTensorHolder> data_nodes;

  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s2", "s1"}));
  data_nodes.emplace_back(data0);

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s3", "s1"}));
  data_nodes.emplace_back(data1);

  auto data2 = builder_->CreateInput(2, "data_2");
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s4", "s1"}));
  data_nodes.emplace_back(data2);

  auto data3 = builder_->CreateInput(3, "data_3");
  data3.SetOriginSymbolShape(std::vector<const char *>({"1.1"}));

  auto data_node = NodeAdapter::GNode2Node(*data3.GetProducer());
  auto sym0 = Symbol(1.1);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto concat = es::Concat(data3, data_nodes, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concat, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  NodeAdapter::GNode2Node(*concat.GetProducer())->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_INT32);
  auto concat_node = cg->FindFirstNodeMatchType("Concat");
  ASSERT_NE(concat_node, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::PARAM_INVALID);
}

REG_OP(SplitD)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitD)

TEST_F(SymbolicShapeInferenceST, test_splitd) {
  // INPUT
  // ATTR split_dim
  // ATTR num_split
  auto splitd = OP_CFG("SplitD").InCnt(1).OutCnt(2)
                                .Attr("split_dim", 0)
                                .Attr("num_split", 2).Build("splitd1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(splitd));
    CHAIN(NODE(splitd)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(splitd)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto splitd_node = graph->FindFirstNodeMatchType("SplitD");
  ASSERT_NE(splitd_node, nullptr);
  splitd_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  auto data_node = graph->FindFirstNodeMatchType(DATA);
  ASSERT_NE(data_node, nullptr);
  gert::SymbolShape symol_shape({Symbol("s0"), Symbol("s1")});
  data_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(symol_shape);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
  splitd_node = graph->FindFirstNodeMatchType("SplitD");
  ASSERT_NE(splitd_node, nullptr);
  auto op_desc = splitd_node->GetOpDesc();
  auto attr0 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  EXPECT_EQ(attr0->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0")/Symbol(2), Symbol("s1")}));
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0")/Symbol(2), Symbol("s1")}));
}

REG_OP(SplitVD)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(size_splits, ListInt)
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitVD)
REG_OP(Assign)
    .INPUT(ref, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .INPUT(value, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .OUTPUT(ref, TensorType({BasicType(), DT_BOOL, DT_STRING}))
    .ATTR(validate_shape, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(Assign)

TEST_F(SymbolicShapeInferenceST, test_splitvd) {
  // INPUT
  // ATTR size_splits
  // ATTR split_dim
  // ATTR num_split
  vector<int64_t> vec = {1, -1, 2};
  auto splitvd = OP_CFG("SplitVD").InCnt(1).OutCnt(3)
                                  .Attr("size_splits", vec)
                                  .Attr("split_dim", 1)
                                  .Attr("num_split", 3).Build("splitvd1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(splitvd));
    CHAIN(NODE(splitvd)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(splitvd)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto splitvd_node = graph->FindFirstNodeMatchType("SplitVD");
  ASSERT_NE(splitvd_node, nullptr);
  splitvd_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  auto data_node = graph->FindFirstNodeMatchType(DATA);
  ASSERT_NE(data_node, nullptr);
  gert::SymbolShape symol_shape({Symbol("s0"), Symbol("s1")});
  data_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.
             SetSymbolShape(symol_shape);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
  splitvd_node = graph->FindFirstNodeMatchType("SplitVD");
  ASSERT_NE(splitvd_node, nullptr);
  auto op_desc = splitvd_node->GetOpDesc();
  auto attr0 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  EXPECT_EQ(attr0->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), Symbol(1)}));
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), Symbol("s1")-Symbol(3)}));
  auto attr2 = op_desc->GetOutputDesc(2).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr2, nullptr);
  EXPECT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), Symbol(2)}));
}

REG_OP(PadD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(paddings, ListListInt)
    .OP_END_FACTORY_REG(PadD)

TEST_F(SymbolicShapeInferenceST, test_padd) {
  std::vector<vector<int64_t>> paddings = {{1, 2}, {2, 1}, {3, 3}};
  auto padd = OP_CFG("PadD").InCnt(1).OutCnt(1).Build("PadD");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(padd));
    CHAIN(NODE(padd)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(padd)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  auto padd_node = graph->FindFirstNodeMatchType("PadD");
  ASSERT_NE(padd_node, nullptr);
  padd_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  padd_node->GetOpDesc()->SetAttr("paddings",GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(paddings));
  auto data_node = graph->FindFirstNodeMatchType(DATA);
  ASSERT_NE(data_node, nullptr);
  gert::SymbolShape symol_shape({Symbol("s0"), Symbol("s1"), Symbol("s2")});
  data_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.
             SetSymbolShape(symol_shape);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
  padd_node = graph->FindFirstNodeMatchType("PadD");
  ASSERT_NE(padd_node, nullptr);
  auto op_desc = padd_node->GetOpDesc();
  auto attr0 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  EXPECT_EQ(attr0->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({ge::Symbol("s0") + ge::Symbol(1) + ge::Symbol(2), Symbol("s1") + ge::Symbol(2) + ge::
              Symbol(1),
              Symbol("s2") + ge::Symbol(3) + ge::Symbol(3)}));
}

REG_OP(LayerNorm)
    .INPUT(x, TensorType::BasicType())
    .INPUT(gamma, TensorType::BasicType())
    .INPUT(beta, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OUTPUT(mean, TensorType::BasicType())
    .OUTPUT(variance, TensorType::BasicType())
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.0000001f)
    .OP_END_FACTORY_REG(LayerNorm)

TEST_F(SymbolicShapeInferenceST, test_layernorm) {
  // input x
  // input gamma
  // input beta
  // output y
  // output mean
  // output variance
  // attr begin_norm_axis
  // attr begin_params_axis
  // attr epsilon
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s5", "s3"}));

  auto data2 = builder_->CreateInput(2, "data_2");
  data2.SetOriginSymbolShape(std::vector<const char *>({"1", "s3"}));

  auto layer0 = es::LayerNorm(data0,data1,data2,-1,0,0.000001);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layer0.y, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layer0.mean, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layer0.variance, 2), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto layer_node = cg->FindFirstNodeMatchType("LayerNorm");
  ASSERT_NE(layer_node, nullptr);
  auto op_desc = layer_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto expect_shape1 = gert::SymbolShape({ge::Symbol("s0") , Symbol("s1") ,
                       Symbol("s2") });
  auto expect_shape2 = gert::SymbolShape({ge::Symbol("s0") , Symbol("s1") ,
                       Symbol(1) });
  auto attr0 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  auto attr2 = op_desc->GetOutputDesc(2).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  EXPECT_EQ(attr0->symbolic_tensor.GetOriginSymbolShape(),expect_shape1);
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(),expect_shape2);
  EXPECT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape(),expect_shape2);
}

TEST_F(SymbolicShapeInferenceST, test_AddN) {
  std::vector<EsTensorHolder> data_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_1");
  auto data2 = builder_->CreateInput(2, "data_2");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  data_nodes.emplace_back(data0);
  data_nodes.emplace_back(data1);
  data_nodes.emplace_back(data2);
  auto addn0 = es::AddN(data_nodes, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addn0, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node1 = cg->FindNode("data_1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node2 = cg->FindNode("data_2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT32);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 1, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {1, 3};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT32};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto addn_node = cg->FindFirstNodeMatchType("AddN");
  ASSERT_NE(addn_node, nullptr);
  auto op_desc = addn_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result_dims = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<std::string> expect_dims = {"s0", "s2", "s4"};
  ASSERT_EQ(result_dims.size(), expect_dims.size());
  for (size_t i = 0UL; i < result_dims.size(); i++) {
    EXPECT_EQ(std::string(result_dims[i].Serialize().get()), expect_dims[i]);
  }
}

TEST_F(SymbolicShapeInferenceST, test_clipbyvalue) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_1");
  auto data2 = builder_->CreateInput(2, "data_2");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  auto clipbyvalue0 = es::ClipByValue(data0, data1, data2);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(clipbyvalue0, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node1 = cg->FindNode("data_1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node2 = cg->FindNode("data_2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT32);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 1, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {1, 3};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT32};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  auto clip_node = cg->FindFirstNodeMatchType("ClipByValue");
  ASSERT_NE(clip_node, nullptr);
  auto op_desc = clip_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result_dims = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<std::string> expect_dims = {"s0", "s2", "s4"};
  ASSERT_EQ(result_dims.size(), expect_dims.size());
  for (size_t i = 0UL; i < result_dims.size(); i++) {
    EXPECT_EQ(std::string(result_dims[i].Serialize().get()), expect_dims[i]);
  }
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBiasAdd) {
  dlog_setlevel(0, 0, 0);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto biasadd = es::BiasAdd(data0, data1, "NHWC");
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(biasadd, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("BiasAdd");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
  dlog_setlevel(0, 3, 0);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBiasAddGradNHWC) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");

  auto tensor = es::BiasAddGrad(data0, "NHWC");
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tensor, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("BiasAddGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBiasAddGradNCHW) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s2 = ge::Symbol("s2");

  auto tensor = es::BiasAddGrad(data0, "NCHW");
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tensor, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("BiasAddGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBiasAddGradFailure) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s2 = ge::Symbol("s2");

  auto biasaddgrad = es::BiasAddGrad(data0, "TEST");
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(biasaddgrad, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_broadcastto_no_symbolicvalue) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "shape");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto broadcastto = es::BroadcastTo(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcastto, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadcastTo) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "shape");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto sym2 = Symbol("s2");
  auto sym3 = Symbol("s3");
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym3);
  ptr->emplace_back(sym1);
  ptr->emplace_back(sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto broadcastto = es::BroadcastTo(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcastto, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto node = cg->FindFirstNodeMatchType("BroadcastTo");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({sym3, sym1, sym0}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadcastTo_shape_size_ge_input_szie) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "shape");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto sym2 = Symbol("s2");
  auto sym3 = Symbol("s3");
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym3);
  ptr->emplace_back(sym2);
  ptr->emplace_back(sym1);
  ptr->emplace_back(sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto broadcastto = es::BroadcastTo(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcastto, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto node = cg->FindFirstNodeMatchType("BroadcastTo");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({sym3, sym2, sym1, sym0}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadcastTo_shape_size_lt_input_size) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "shape");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto sym2 = Symbol("s2");
  auto sym3 = Symbol("s3");
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym1);
  ptr->emplace_back(sym0);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto broadcastto = es::BroadcastTo(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcastto, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadcastTo_input_to_shape_unsupport) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "shape");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto sym2 = Symbol("s2");
  auto sym3 = Symbol("s3");
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym1);
  ptr->emplace_back(sym0);
  ptr->emplace_back(sym3);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto broadcastto = es::BroadcastTo(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcastto, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadcastTo_with_negone_shape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "shape");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"3"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto data_node = NodeAdapter::GNode2Node(*data1.GetProducer());
  auto sym0 = Symbol("s0");
  auto sym1 = Symbol("s1");
  auto neg_one = Symbol(-1);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(neg_one);
  ptr->emplace_back(neg_one);
  ptr->emplace_back(sym1);
  auto output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  auto out_desc_attr = output_desc->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(out_desc_attr, nullptr);
  out_desc_attr->symbolic_tensor.SetSymbolicValue(std::move(ptr));
  output_desc->SetDataType(DT_INT32);

  auto broadcastto = es::BroadcastTo(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcastto, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto node = cg->FindFirstNodeMatchType("BroadcastTo");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(1), sym0, sym1}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForLayerNormBetaGammaBackpropV2Neg) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s1 = ge::Symbol("s1");
  auto kone = ge::Symbol(1);
  vector<int64_t> gamma_shape = {-1, 0, -1};

  auto biasaddgrad = es::LayerNormBetaGammaBackpropV2(data0, data1, gamma_shape);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(biasaddgrad.pd_gamma, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(biasaddgrad.pd_beta, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LayerNormBetaGammaBackpropV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({kone, s1, kone}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({kone, s1, kone}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForLayerNormBetaGammaBackpropV2) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto ktwo = ge::Symbol(2);
  vector<int64_t> gamma_shape = {2};

  auto biasaddgrad = es::LayerNormBetaGammaBackpropV2(data0, data1, gamma_shape);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(biasaddgrad.pd_beta, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(biasaddgrad.pd_gamma, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LayerNormBetaGammaBackpropV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({ktwo}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({ktwo}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForLayerNormV3) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto kone = ge::Symbol(1);

  auto layernormv3 = es::LayerNormV3(data0, data1, data2, 2, 2, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormv3.y, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormv3.mean, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormv3.rstd, 2), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LayerNormV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto attr2 = op_desc->GetOutputDesc(2).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr2, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, kone}).GetDims());
  EXPECT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, kone}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForLayerNormXBackpropV3) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  auto data3 = builder_->CreateInput(3, "data3");
  auto data4 = builder_->CreateInput(4, "data4");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data3.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data4.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  ASSERT_NE(data3.GetCTensorHolder(), nullptr);
  ASSERT_NE(data4.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto layernormxbackparopv3 = es::LayerNormXBackpropV3(data0, data1, data2, data3, data4);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormxbackparopv3.pd_x, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormxbackparopv3.res_for_gamma, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LayerNormXBackpropV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_stridedsliceD_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3};
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5};
  // strides
  std::vector<int64_t> data3_value = {1, 1, 1, 1};

  auto strided_sliced = es::StridedSliceD(data0, data1_value, data2_value, data3_value,
                                        static_cast<int64_t>(0b0010),
                                        static_cast<int64_t>(0b0010),
                                        static_cast<int64_t>(0b0100),
                                        static_cast<int64_t>(0b1101),
                                        static_cast<int64_t>(0b0111));
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_sliced, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_sliced_node = cg->FindFirstNodeMatchType("StridedSliceD");
  ASSERT_NE(strided_sliced_node, nullptr);
  auto op_desc = strided_sliced_node->GetOpDesc();
  // 跳过hostcompute
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 6);
  ASSERT_EQ(out_shape.GetDim(0), Symbol(1));
  ASSERT_EQ(out_shape.GetDim(1), Symbol("s1"));
  ASSERT_EQ(out_shape.GetDim(2), Symbol("s2"));
  ASSERT_EQ(out_shape.GetDim(3), Symbol("s3"));
  ASSERT_EQ(out_shape.GetDim(4), Symbol("s4"));
  ASSERT_EQ(out_shape.GetDim(5), Symbol(1));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxCrossEntropyWithLogits2) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"3", "2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2", "4"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s2 = ge::Symbol(2);
  auto s3 = ge::Symbol(3);
  auto s4 = ge::Symbol(4);
  auto softmaxcrossentropywithlogits = es::SoftmaxCrossEntropyWithLogits(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.loss, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.backprop, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxCrossEntropyWithLogits");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3, s4}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxCrossEntropyWithLogits2big) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"2", "4"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"3", "2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s2 = ge::Symbol(2);
  auto s3 = ge::Symbol(3);
  auto s4 = ge::Symbol(4);
  auto softmaxcrossentropywithlogits = es::SoftmaxCrossEntropyWithLogits(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.loss, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.backprop, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxCrossEntropyWithLogits");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3, s4}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxCrossEntropyWithLogits) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"2", "3", "4", "5"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"3", "4", "5", "6"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s2 = ge::Symbol(2);
  auto s3 = ge::Symbol(3);
  auto s4 = ge::Symbol(4);
  auto s5 = ge::Symbol(5);
  auto s6 = ge::Symbol(6);
  auto softmaxcrossentropywithlogits = es::SoftmaxCrossEntropyWithLogits(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.loss, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.backprop, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxCrossEntropyWithLogits");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3, s4, s5, s6}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3, s4, s5, s6}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxCrossEntropyWithLogitsbig) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"3", "4", "5", "6"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"2", "3", "4", "5"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s2 = ge::Symbol(2);
  auto s3 = ge::Symbol(3);
  auto s4 = ge::Symbol(4);
  auto s5 = ge::Symbol(5);
  auto s6 = ge::Symbol(6);
  auto softmaxcrossentropywithlogits = es::SoftmaxCrossEntropyWithLogits(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.loss, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxcrossentropywithlogits.backprop, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxCrossEntropyWithLogits");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3, s4, s5, s6}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3, s4, s5, s6}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_stridedsliceD_infershape2) {
  auto data0 = builder_->CreateInput(0, "data_0");
  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3, 4};
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5, 5};
  // strides
  std::vector<int64_t> data3_value = {1, 2, 1, 1, 1};

  auto strided_sliced = es::StridedSliceD(data0, data1_value, data2_value, data3_value,
                                        static_cast<int64_t>(0),
                                        static_cast<int64_t>(0b10110),
                                        static_cast<int64_t>(0),
                                        static_cast<int64_t>(0),
                                        static_cast<int64_t>(0));
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_sliced, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_sliced_node = cg->FindFirstNodeMatchType("StridedSliceD");
  ASSERT_NE(strided_sliced_node, nullptr);
  auto op_desc = strided_sliced_node->GetOpDesc();
  // 跳过hostcompute
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 5);
  ASSERT_EQ(out_shape.GetDim(0), Symbol("s0"));
  ASSERT_EQ(out_shape.GetDim(1), sym::Ceiling((Symbol("s1") - Symbol(1)) / Symbol(2)));
  ASSERT_EQ(out_shape.GetDim(2), Symbol("s2") - Symbol(2));
  ASSERT_EQ(out_shape.GetDim(3), Symbol(2));
  ASSERT_EQ(out_shape.GetDim(4), Symbol("s4") - Symbol(4));
}

TEST_F(SymbolicShapeInferenceST, test_stridedslicev2_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3, 4};
  auto data1 = builder_->CreateVector(data1_value);
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5, 5};
  auto data2 = builder_->CreateVector(data2_value);
  // axes
  std::vector<int64_t> data3_value = {0, 1, 2, 3, 4};
  auto data3 = builder_->CreateVector(data3_value);
  // strides
  std::vector<int64_t> data4_value = {1, 2, 1, 1, 1};
  auto data4 = builder_->CreateVector(data4_value);

  auto strided_sliced = es::StridedSliceV2(data0, data1, data2, data3, data4,
                                          static_cast<int64_t>(0b0010),
                                          static_cast<int64_t>(0b0010),
                                          static_cast<int64_t>(0b0100),
                                          static_cast<int64_t>(0b1101),
                                          static_cast<int64_t>(0b0111));
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_sliced, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_sliced_node = cg->FindFirstNodeMatchType("StridedSliceV2");
  ASSERT_NE(strided_sliced_node, nullptr);
  auto op_desc = strided_sliced_node->GetOpDesc();
  // 跳过hostcompute
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  const std::vector<ge::Expression> expect_shape = {Symbol(1), Symbol("s1"), Symbol("s2"), Symbol("s3"), Symbol(1), Symbol(1)};
  ASSERT_EQ(out_shape.GetDims(), expect_shape);
}

TEST_F(SymbolicShapeInferenceST, test_stridedslicev2_infershape_no_optional_input) {
  auto data0 = builder_->CreateInput(0, "data_0");
  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3, 4};
  auto data1 = builder_->CreateVector(data1_value);
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5, 5};
  auto data2 = builder_->CreateVector(data2_value);

  auto strided_sliced = es::StridedSliceV2(data0, data1, data2, nullptr, nullptr,
                                           static_cast<int64_t>(0b0010),
                                           static_cast<int64_t>(0b0010),
                                           static_cast<int64_t>(0b0100),
                                           static_cast<int64_t>(0b1101),
                                           static_cast<int64_t>(0b0111));
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_sliced, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_sliced_node = cg->FindFirstNodeMatchType("StridedSliceV2");
  ASSERT_NE(strided_sliced_node, nullptr);
  auto op_desc = strided_sliced_node->GetOpDesc();
  // 跳过hostcompute
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  const std::vector<ge::Expression> expect_shape = {Symbol(1), Symbol("s1"), Symbol("s2"), Symbol("s3"), Symbol(1), Symbol("s4")};
  ASSERT_EQ(out_shape.GetDims(), expect_shape);
}

TEST_F(SymbolicShapeInferenceST, test_stridedslicev3_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3, 4};
  auto data1 = builder_->CreateVector(data1_value);
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5, 5};
  auto data2 = builder_->CreateVector(data2_value);
  // axes
  std::vector<int64_t> data3_value = {0, 1, 2, 3, 4};
  auto data3 = builder_->CreateVector(data3_value);
  // strides
  std::vector<int64_t> data4_value = {1, 2, 1, 1, 1};
  auto data4 = builder_->CreateVector(data4_value);

  auto strided_sliced = es::StridedSliceV3(data0, data1, data2, data3, data4);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_sliced, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_sliced_node = cg->FindFirstNodeMatchType("StridedSliceV3");
  ASSERT_NE(strided_sliced_node, nullptr);
  auto op_desc = strided_sliced_node->GetOpDesc();
  // 跳过hostcompute
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  const std::vector<ge::Expression> expect_shape = {Symbol(5), Symbol(2), Symbol(3), Symbol(2), Symbol(1)};
  ASSERT_EQ(out_shape.GetDims(), expect_shape);
}

TEST_F(SymbolicShapeInferenceST, test_stridedslicev3_infershape_no_optional_input) {
  auto data0 = builder_->CreateInput(0, "data_0");
  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3, 4};
  auto data1 = builder_->CreateVector(data1_value);
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5, 5};
  auto data2 = builder_->CreateVector(data2_value);

  auto strided_sliced = es::StridedSliceV3(data0, data1, data2);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_sliced, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_sliced_node = cg->FindFirstNodeMatchType("StridedSliceV3");
  ASSERT_NE(strided_sliced_node, nullptr);
  auto op_desc = strided_sliced_node->GetOpDesc();
  // 跳过hostcompute
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 6, 7, 8, 9};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  auto out_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  const std::vector<ge::Expression> expect_shape = {Symbol(5), Symbol(4), Symbol(3), Symbol(2), Symbol(1)};
  ASSERT_EQ(out_shape.GetDims(), expect_shape);
}

/*
   (2, 1, 4, 3)   (1, 3, 1, 3)       (8, 3)
    Data        Data      Data
      |           |         |
      |           |         |
       \         /          |
           Add              |        Const
           |                |       /
            \               ReShape
             \                /
              \             /
              ----BatchMatmul
                      \
                    Squeeze
*/
TEST_F(SymbolicShapeInferenceST, test_guard_check_graph1) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  auto add_0 = es::Add(data0, data1);
  std::vector<int64_t> dims_data{1, 1, 3, 3, -1};
  int64_t dims_size = 5;
  auto const_0 = builder_->CreateConst(dims_data, std::vector<int64_t>{dims_size});
  ASSERT_NE(const_0.GetCTensorHolder(), nullptr);
  auto reshape_0 = es::Reshape(data2, const_0, 0, -1);
  auto batch_matmul = es::BatchMatMulV2(add_0, reshape_0, nullptr, nullptr, false, false, 0);
  std::vector<int64_t> axis;
  auto squeeze_0 = es::Squeeze(batch_matmul, axis);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squeeze_0, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({1, -1, -1, 3}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({1, -1, -1, 3}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({1, -1, -1, 3}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, -1, 3}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto reshape_op_0 = cg->FindFirstNodeMatchType("Reshape");
  ASSERT_NE(reshape_op_0, nullptr);
  auto reshape_op_desc0 = reshape_op_0->GetOpDesc();
  reshape_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  reshape_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 1, 4, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {1, 3, 1, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {12, 3};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "s2");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "1");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "3");

  auto data_symbol_attr2 = data_op_desc2->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr2, nullptr);
  auto symbol_shape2 = data_symbol_attr2->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape2.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape2.GetDim(0).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(1).Serialize().get()), "s6");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto add_node = cg->FindFirstNodeMatchType("Add");
  ASSERT_NE(add_node, nullptr);
  auto add_op_desc = add_node->GetOpDesc();
  ASSERT_NE(add_op_desc, nullptr);
  auto add_attr = add_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(add_attr, nullptr);
  auto add_symbol_shape = add_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(add_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(2).Serialize().get()), "s1");
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(3).Serialize().get()), "3");

  auto reshape_node = cg->FindFirstNodeMatchType("Reshape");
  ASSERT_NE(reshape_node, nullptr);
  auto reshape_op_desc = reshape_node->GetOpDesc();
  ASSERT_NE(reshape_op_desc, nullptr);
  auto reshape_attr = reshape_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_attr, nullptr);
  auto reshape_symbol_shape = reshape_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(reshape_symbol_shape.GetDimNum(), 5);
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(0).Serialize().get()), "1");
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(1).Serialize().get()), "1");
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(3).Serialize().get()), "3");
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(4).Serialize().get()), "(Rational(1 , 9) * s5 * s6)");
  auto matmul_node = cg->FindFirstNodeMatchType("BatchMatMulV2");
  ASSERT_NE(matmul_node, nullptr);
  auto matmul_op_desc = matmul_node->GetOpDesc();
  ASSERT_NE(matmul_op_desc, nullptr);
  auto matmul_attr = matmul_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(matmul_attr, nullptr);
  auto matmul_symbol_shape = matmul_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(matmul_symbol_shape.GetDimNum(), 5);
  EXPECT_EQ(std::string(matmul_symbol_shape.GetDim(0).Serialize().get()), "1");
  EXPECT_EQ(std::string(matmul_symbol_shape.GetDim(1).Serialize().get()), "s0");
  EXPECT_EQ(std::string(matmul_symbol_shape.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(matmul_symbol_shape.GetDim(3).Serialize().get()), "s1");
  EXPECT_EQ(std::string(matmul_symbol_shape.GetDim(4).Serialize().get()), "(Rational(1 , 9) * s5 * s6)");

  auto squeeze_node = cg->FindFirstNodeMatchType("Squeeze");
  ASSERT_NE(squeeze_node, nullptr);
  auto squeeze_op_desc = squeeze_node->GetOpDesc();
  ASSERT_NE(squeeze_op_desc, nullptr);
  auto squeeze_attr = squeeze_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(squeeze_attr, nullptr);
  auto squeeze_symbol_shape = squeeze_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(squeeze_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(squeeze_symbol_shape.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(squeeze_symbol_shape.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(std::string(squeeze_symbol_shape.GetDim(2).Serialize().get()), "s1");
  EXPECT_EQ(std::string(squeeze_symbol_shape.GetDim(3).Serialize().get()), "(Rational(1 , 9) * s5 * s6)");

  const std::vector<SymbolCheckInfo> assert_guard_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_guard_infos.size(), 0);

  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 15);
  const std::set<std::string> expect_guard =
      {"ExpectNe((Rational(1 , 9) * s5 * s6), 1)", "ExpectNe(1, s1)",
       "ExpectEq(3, s3)", "ExpectNe(1, s3)", "ExpectNe(1, s0)",
       "ExpectNe(s1, s4)", "ExpectEq(1, s4)", "ExpectEq(3, s2)", "ExpectNe(0, s0)",
       "ExpectNe(0, s1)", "ExpectNe(0, s2)", "ExpectNe(0, s3)", "ExpectNe(0, s4)",
       "ExpectNe(0, s5)", "ExpectNe(0, s6)"};
  for (auto &iter : guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
}

/*    [-1(2)]       [8]       [-1(2)]
     Data        Data      Data
      |           |         |
      |           |         |
    Shape         |       Shape
       \         /          |       [-1(5)]
        Slice     ----------      Data
            \                ------
             \              /        [(-1)8]    [-1(2), 8]
                ConcatV2             Data   Data
                     \             /        |
                          Pack             /       [2, -1(8)]
                             \            /       Data
                                Select  ----------
                                  |
                              Output
*/
TEST_F(SymbolicShapeInferenceST, test_guard_check_graph2) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  auto data3 = builder_->CreateInput(3, "data3");
  auto data4 = builder_->CreateInput(4, "data4");
  auto data5 = builder_->CreateInput(5, "data5");
  auto data6 = builder_->CreateInput(6, "data6");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  ASSERT_NE(data3.GetCTensorHolder(), nullptr);
  ASSERT_NE(data4.GetCTensorHolder(), nullptr);
  ASSERT_NE(data5.GetCTensorHolder(), nullptr);
  ASSERT_NE(data6.GetCTensorHolder(), nullptr);
  auto shape_0 = es::Shape(data0, 3);
  auto shape_1 = es::Shape(data2, 3);
  auto slice_0 = es::Slice(data1, shape_0, shape_1);
  std::vector<EsTensorHolder> concat_input;
  concat_input.push_back(slice_0);
  concat_input.push_back(data3);
  auto concat0 = es::ConcatV2D(concat_input, 0, 2);
  std::vector<EsTensorHolder> pack_input;
  pack_input.push_back(concat0);
  pack_input.push_back(data4);
  auto pack_0 = es::Pack(pack_input, 0, 2);
  auto select_0 = es::Select(data6, pack_0, data5);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(select_0, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({8}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({8}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({8}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({8}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node3 = cg->FindNode("data3");
  ASSERT_NE(data_node3, nullptr);
  auto data_op_desc3 = data_node3->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc3, "index", 3);
  data_op_desc3->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc3->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc3->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc3->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc3->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc3->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node4 = cg->FindNode("data4");
  ASSERT_NE(data_node4, nullptr);
  auto data_op_desc4 = data_node4->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc4, "index", 4);
  data_op_desc4->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc4->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc4->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc4->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc4->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc4->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node5 = cg->FindNode("data5");
  ASSERT_NE(data_node5, nullptr);
  auto data_op_desc5 = data_node5->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc5, "index", 5);
  data_op_desc5->MutableInputDesc(0)->SetShape(GeShape({-1, 8}));
  data_op_desc5->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 8}));
  data_op_desc5->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc5->MutableOutputDesc(0)->SetShape(GeShape({-1, 8}));
  data_op_desc5->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 8}));
  data_op_desc5->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node6 = cg->FindNode("data6");
  ASSERT_NE(data_node6, nullptr);
  auto data_op_desc6 = data_node6->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc6, "index", 6);
  data_op_desc6->MutableInputDesc(0)->SetShape(GeShape({2, -1}));
  data_op_desc6->MutableInputDesc(0)->SetOriginShape(GeShape({2, -1}));
  data_op_desc6->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc6->MutableOutputDesc(0)->SetShape(GeShape({2, -1}));
  data_op_desc6->MutableOutputDesc(0)->SetOriginShape(GeShape({2, -1}));
  data_op_desc6->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto slice_op_0 = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_op_0, nullptr);
  auto slice_op_desc0 = slice_op_0->GetOpDesc();
  slice_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT32);
  slice_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT32);
  slice_op_desc0->MutableInputDesc(2)->SetDataType(DT_INT32);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {8};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {3};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT32};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  std::vector<int64_t> dims_vec3 = {5};
  ge::Shape shape3({dims_vec3});
  ge::TensorDesc td3{shape3, ge::FORMAT_ND, DT_INT32};
  td3.SetOriginShape(shape3);
  ge::Tensor tensor3{td3};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor3));

  std::vector<int64_t> dims_vec4 = {8};
  ge::Shape shape4({dims_vec4});
  ge::TensorDesc td4{shape4, ge::FORMAT_ND, DT_INT32};
  td4.SetOriginShape(shape4);
  ge::Tensor tensor4{td4};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor4));

  std::vector<int64_t> dims_vec5 = {2, 8};
  ge::Shape shape5({dims_vec5});
  ge::TensorDesc td5{shape5, ge::FORMAT_ND, DT_INT32};
  td5.SetOriginShape(shape5);
  ge::Tensor tensor5{td5};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor5));

  std::vector<int64_t> dims_vec6 = {2, 8};
  ge::Shape shape6({dims_vec6});
  ge::TensorDesc td6{shape6, ge::FORMAT_ND, DT_INT32};
  td6.SetOriginShape(shape6);
  ge::Tensor tensor6{td6};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor6));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);

  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "8");

  auto data_symbol_attr2 = data_op_desc2->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr2, nullptr);
  auto symbol_shape2 = data_symbol_attr2->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape2.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_shape2.GetDim(0).Serialize().get()), "s1");

  auto data_symbol_attr3 = data_op_desc3->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr3, nullptr);
  auto symbol_shape3 = data_symbol_attr3->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape3.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_shape3.GetDim(0).Serialize().get()), "s2");

  auto data_symbol_attr4 = data_op_desc4->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr4, nullptr);
  auto symbol_shape4 = data_symbol_attr4->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape4.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_shape4.GetDim(0).Serialize().get()), "s3");

  auto data_symbol_attr5 = data_op_desc5->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr5, nullptr);
  auto symbol_shape5 = data_symbol_attr5->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape5.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape5.GetDim(0).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape5.GetDim(1).Serialize().get()), "8");

  auto data_symbol_attr6 = data_op_desc6->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr6, nullptr);
  auto symbol_shape6 = data_symbol_attr6->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape6.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape6.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape6.GetDim(1).Serialize().get()), "s5");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  auto slice_op_desc = slice_node->GetOpDesc();
  ASSERT_NE(slice_op_desc, nullptr);
  auto slice_attr = slice_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(slice_attr, nullptr);
  auto slice_symbol_shape = slice_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(slice_symbol_shape.GetDimNum(), 1);
  EXPECT_EQ(std::string(slice_symbol_shape.GetDim(0).Serialize().get()), "s1");

  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto concatv2d_op_desc = concatv2d_node->GetOpDesc();
  ASSERT_NE(concatv2d_op_desc, nullptr);
  auto concatv2d_attr = concatv2d_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(concatv2d_attr, nullptr);
  auto concatv2d_symbol_shape = concatv2d_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(concatv2d_symbol_shape.GetDimNum(), 1);
  EXPECT_EQ(std::string(concatv2d_symbol_shape.GetDim(0).Serialize().get()), "(s1 + s2)");

  auto pack_node = cg->FindFirstNodeMatchType("Pack");
  ASSERT_NE(pack_node, nullptr);
  auto pack_op_desc = pack_node->GetOpDesc();
  ASSERT_NE(pack_op_desc, nullptr);
  auto pack_attr = pack_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(pack_attr, nullptr);
  auto pack_symbol_shape = pack_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(pack_symbol_shape.GetDimNum(), 2);
  EXPECT_EQ(std::string(pack_symbol_shape.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(pack_symbol_shape.GetDim(1).Serialize().get()), "(s1 + s2)");

  auto select_node = cg->FindFirstNodeMatchType("Select");
  ASSERT_NE(select_node, nullptr);
  auto select_op_desc = select_node->GetOpDesc();
  ASSERT_NE(select_op_desc, nullptr);
  auto select_attr = select_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(select_attr, nullptr);
  auto select_symbol_shape = select_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(select_symbol_shape.GetDimNum(), 2);
  EXPECT_EQ(std::string(select_symbol_shape.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(select_symbol_shape.GetDim(1).Serialize().get()), "(s1 + s2)");

  const std::vector<SymbolCheckInfo> assert_guard_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_guard_infos.size(), 6);
  const std::set<std::string> expect_guard = {"ExpectEq((s1 + s2), 8)", "ExpectLe(0, s0)",
                                              "ExpectEq(2, s4)", "ExpectEq((s1 + s2), s3)",
                                              "ExpectLe(s0, 8)", "ExpectEq((s1 + s2), s5)"};
  for (auto &iter : assert_guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
}

/*   [-1, 29, 39]
     Data0      Data1[41, 29, 39]
      |        /
       |      /             Data2[41, 29, 39]
       ConcatV2               /
          \     \            /
            \      Sigmoid  /
             \      |     /
                ConcatV2 [82, 87, 39]
                    |
                 Output
*/
TEST_F(SymbolicShapeInferenceST, test_ConcatV2_const_shape_replace) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);

  std::vector<EsTensorHolder> inputs = {data0, data1};
  auto concat = es::ConcatV2D(inputs, 0, 1);
  auto sigmod = es::Sigmoid(concat);
  std::vector<EsTensorHolder> inputs2 = {concat, sigmod, data2};
  auto concat2 = es::ConcatV2D(inputs2, 1, 1);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concat2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, 29, 39}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 29, 39}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, 29, 39}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 29, 39}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({41, 29, 39}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({41, 29, 39}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({41, 29, 39}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({41, 29, 39}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({82, 29, 39}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({82, 29, 39}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({82, 29, 39}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({82, 29, 39}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT32);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {41, 29, 39};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {41, 29, 39};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {82, 29, 39};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT32};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);

  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 3);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 3);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "41");

  auto data_symbol_attr2 = data_op_desc2->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr2, nullptr);
  auto symbol_shape2 = data_symbol_attr2->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape2.GetDimNum(), 3);
  EXPECT_EQ(std::string(symbol_shape2.GetDim(0).Serialize().get()), "82");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto concatv2d_node = cg->FindNode("ConcatV2D_2");
  ASSERT_NE(concatv2d_node, nullptr);
  auto concatv2d_op_desc = concatv2d_node->GetOpDesc();
  ASSERT_NE(concatv2d_op_desc, nullptr);
  auto concatv2d_attr = concatv2d_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(concatv2d_attr, nullptr);
  auto concatv2d_symbol_shape = concatv2d_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(concatv2d_symbol_shape.GetDimNum(), 3);
  EXPECT_EQ(std::string(concatv2d_symbol_shape.GetDim(0).Serialize().get()), "82");
  EXPECT_EQ(std::string(concatv2d_symbol_shape.GetDim(1).Serialize().get()), "87");
  EXPECT_EQ(std::string(concatv2d_symbol_shape.GetDim(2).Serialize().get()), "39");
}

TEST_F(SymbolicShapeInferenceST, InferShapeForElu) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto elu = es::Elu(mul, 0, 0, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(elu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Elu");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForFourSquareErfReciprocalLeakyRelu) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto square = es::Square(data0);
  auto Erf = es::Erf(square);
  auto Reciprocal = es::Reciprocal(Erf);
  auto LeakyRelu = es::LeakyRelu(Reciprocal,0.01);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(LeakyRelu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LeakyRelu");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSimpleBro) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto relugrad = es::ReluGrad(data0,data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relugrad, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("ReluGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSimpleEle) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "s1", "s0","s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto rsqrtgrad = es::RsqrtGrad(data0,data1);
  auto muls = es::Muls(rsqrtgrad,0.01);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(muls, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Muls");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTanh) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto tanh = es::Tanh(data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tanh, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Tanh");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForGreaterequalDivnonanLeakyrelugrad) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto greaterequal = es::GreaterEqual(data0,data1);
  auto divnonan = es::DivNoNan(data0,data1);
  auto leakyrelugrad = es::LeakyReluGrad(greaterequal,divnonan,0.01);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(leakyrelugrad, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LeakyReluGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForConv2DBackpropInputD) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0", "1"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto c1 = ge::Symbol(1);
  auto c2 = ge::Symbol(2);
  auto c3 = ge::Symbol(3);
  auto c4 = ge::Symbol(4);
  std::vector<int64_t> data1_value = {1,2,3,4};
  std::vector<int64_t> data2_value = {1,1,1,1};
  std::vector<int64_t> data3_value = {1,1,1,1};
  std::vector<int64_t> data4_value = {1,1,1,1};
  auto conv2dback = es::Conv2DBackpropInputD(data0, data1, data1_value, data2_value, data3_value, data4_value, 1, "NHWC");

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2dback, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Conv2DBackpropInputD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({c1, c2, c3, c4}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForAddV2) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto addv2 = es::AddV2(data0, data1);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addv2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("AddV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxV2) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int64_t> data_value = {1,1,1,1};
  auto softmaxV2 = es::SoftmaxV2(data0, data_value, false);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxV2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSigmoidGrad) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto sigmoidgrad = es::SigmoidGrad(data0, data1, false);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(sigmoidgrad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SigmoidGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForHcomAllReduce) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  const char* data1_value = "1";
  const char* data2_value = "sum";
  auto hcomAllReduce = es::HcomAllReduce(data0, data1_value, data2_value, 1, 1);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(hcomAllReduce, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  SymbolicShapeInference ssi;
  ASSERT_NE(cg, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("HcomAllReduce");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForConv2DBackpropFilterD) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0", "1"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto c1 = ge::Symbol(1);
  auto c2 = ge::Symbol(2);
  auto c3 = ge::Symbol(3);
  auto c4 = ge::Symbol(4);
  std::vector<int64_t> data1_value = {1,2,3,4};
  std::vector<int64_t> data2_value = {1,1,1,1};
  std::vector<int64_t> data3_value = {1,1,1,1};
  std::vector<int64_t> data4_value = {1,1,1,1};
  auto conv2dback = es::Conv2DBackpropFilterD(data0, data1, data1_value, data2_value, data3_value, data4_value, 1, "NHWC");

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2dback, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Conv2DBackpropFilterD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({c1, c2, c3, c4}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForLog) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s1"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto log = es::Log(data0,-1,1,0);
  auto log1p = es::Log1p(log);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(log1p, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Log1p");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s1}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxGradExt) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data2.SetOriginSymbolShape({});
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto softmaxGradExt = es::SoftmaxGradExt(data0, data1, data2, 1, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxGradExt, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxGradExt");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
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

TEST_F(SymbolicShapeInferenceST, test_multisliceconcat) {
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
  data_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.
    SetSymbolShape(symol_shape);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
  multisliceconcat_node = graph->FindFirstNodeMatchType("MultisliceConcat");
  ASSERT_NE(multisliceconcat_node, nullptr);
  auto op_desc = multisliceconcat_node->GetOpDesc();
  auto attr0 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  EXPECT_EQ(attr0->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), (Symbol(1)+Symbol(1))}));

  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol("s0"), (Symbol(2)+Symbol(2)+Symbol(2)+Symbol(4))}));

  auto symbol_shape0 = attr0->symbolic_tensor.GetOriginSymbolShape();
  auto outshapeout = std::string((symbol_shape0.GetDim(0).Serialize().get()));
  auto outshapeout1 = std::string((symbol_shape0.GetDim(1).Serialize().get()));
  printf("output shape 1 is %s\n, outputshape 2 is %s\n", outshapeout.c_str(), outshapeout1.c_str());

  auto symbol_shape1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  auto outshapeout_1 = std::string((symbol_shape1.GetDim(0).Serialize().get()));
  auto outshapeout1_1 = std::string((symbol_shape1.GetDim(1).Serialize().get()));
  printf("output shape 1 is %s\n, outputshape 2 is %s\n", outshapeout_1.c_str(), outshapeout1_1.c_str());
}

TEST_F(SymbolicShapeInferenceST, test_sliced_infer_shape) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int64_t> offset = {0, 0, 0};
  std::vector<int64_t> size = {3, 3, 3};

  auto sliceD = es::SliceD(data0, offset, size);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(sliceD, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 6}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 6}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 6}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 6}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {9, 5, 6};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);

  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 3);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "6");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto sliced_node = cg->FindFirstNodeMatchType("SliceD");
  ASSERT_NE(sliced_node, nullptr);
  auto sliced_op_desc = sliced_node->GetOpDesc();
  ASSERT_NE(sliced_op_desc, nullptr);
  auto sliced_attr = sliced_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(sliced_attr, nullptr);
  auto sliced_symbol_shape = sliced_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(sliced_symbol_shape.GetDimNum(), 3);
  EXPECT_EQ(std::string(sliced_symbol_shape.GetDim(0).Serialize().get()), "3");
  EXPECT_EQ(std::string(sliced_symbol_shape.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(std::string(sliced_symbol_shape.GetDim(2).Serialize().get()), "3");
}

TEST_F(SymbolicShapeInferenceST, InferShapeForIsFinite) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto isfinite = es::IsFinite(data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(isfinite, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("IsFinite");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1, s2, s0}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForStopGradient) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto stopgradient = es::StopGradient(data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(stopgradient, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("StopGradient");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSquareSumV1) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto kone = ge::Symbol(1);
  std::vector<int64_t> reduce_axis = {0,1};

  auto squaresumv1 = es::SquareSumV1(data0, reduce_axis, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squaresumv1, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SquareSumV1");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({kone, kone, s2}).GetDims());
}

REG_OP(Repeat)
  .INPUT(x, TensorType::ALL())
  .INPUT(repeat_times, TensorType::ALL())
  .OUTPUT(y, TensorType::ALL())
  .OP_END_FACTORY_REG(Repeat)

TEST_F(SymbolicShapeInferenceST, test_repeat_infer) {
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
  ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
  ->symbolic_tensor.SetSymbolShape(data_symbol_shape0);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto sym0 = Symbol(8);
  auto sym1 = Symbol(2);
  auto sym2 = Symbol(2);
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
      ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolShape(data_symbol_shape1);
  data_node1->GetOpDescBarePtr()
    ->MutableOutputDesc(0)
    ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
    ->symbolic_tensor.SetSymbolicValue(std::move(ptr));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto repeat1_node = cg->FindNode("repeat1");
  ASSERT_NE(repeat1_node, nullptr);
  auto op_desc1 = repeat1_node->GetOpDesc();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_expr1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_expr1.GetDim(0).Serialize().get()), "16");
  EXPECT_EQ(std::string(symbol_expr1.GetDim(1).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_expr1.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(symbol_expr1.GetDim(3).Serialize().get()), "4");
}

TEST_F(SymbolicShapeInferenceST, test_range_with_symbolic_value_1) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto es_shape1 = es::Shape(data0, 3); // DT_INT32
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  auto range = es::Range(es_shape1, data1, data2, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(range, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto range_node = cg->FindFirstNodeMatchType(RANGE);
  ASSERT_NE(range_node, nullptr);
  auto op_desc = range_node->GetOpDesc();

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {10};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {1};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto attr1 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_expr1.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_expr1.GetDim(0).Serialize().get()), "Ceiling(s0)");

  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 3);
  const std::set<std::string> expect_guard = {"ExpectLt(0, s0)", "ExpectNe(0, s0)", "ExpectNe(0, s1)"};
  for (auto &iter : guard_infos) {
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 0);
}

TEST_F(SymbolicShapeInferenceST, test_range_with_symbolic_value_2) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto es_shape1 = es::Shape(data0, 3); // DT_INT32
  auto data1 = builder_->CreateInput(1, "data1");
  auto es_shape2 = es::Shape(data1, 3); // DT_INT32
  auto data2 = builder_->CreateInput(2, "data2");
  auto range = es::Range(es_shape1, es_shape2, data2, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(range, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto range_node = cg->FindFirstNodeMatchType(RANGE);
  ASSERT_NE(range_node, nullptr);
  auto op_desc = range_node->GetOpDesc();

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {1};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {10};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto attr1 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_expr1.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_expr1.GetDim(0).Serialize().get()), "(-1 * Floor(((s1 - s0) * -1)))");

  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 3);
  const std::set<std::string> expect_guard = {"ExpectLt(s0, s1)", "ExpectNe(0, s0)", "ExpectNe(0, s1)"};
  for (auto &iter : guard_infos) {
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 0);
}

TEST_F(SymbolicShapeInferenceST, test_range_with_symbolic_value_3) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto es_shape1 = es::Shape(data0, 3); // DT_INT32
  auto data1 = builder_->CreateInput(1, "data1");
  auto es_shape2 = es::Shape(data1, 3); // DT_INT32
  std::vector<int32_t> const0 = {-1};
  std::vector<int64_t> data_dims = {};
  auto delta = builder_->CreateConst(const0, data_dims);
  auto range = es::Range(es_shape1, es_shape2, delta, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(range, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto range_node = cg->FindFirstNodeMatchType(RANGE);
  ASSERT_NE(range_node, nullptr);
  auto op_desc = range_node->GetOpDesc();

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {10};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {1};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto attr1 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_expr1.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_expr1.GetDim(0).Serialize().get()), "(-1 * Floor((s1 - s0)))");

  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 4);
  const std::set<std::string> expect_guard = {"ExpectLt(s1, s0)", "ExpectLe(s1, s0)", "ExpectNe(0, s0)", "ExpectNe(0, s1)"};
  for (auto &iter : guard_infos) {
    std::string str = std::string(iter.expr.Serialize().get());
    std::cout << str << std::endl;
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 0);
}

TEST_F(SymbolicShapeInferenceST, test_range_with_symbolic_value_4) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto es_shape1 = es::Shape(data0, 3); // DT_INT32
  auto data1 = builder_->CreateInput(1, "data1");
  auto es_shape2 = es::Shape(data1, 3); // DT_INT32
  auto data2 = builder_->CreateInput(2, "data2");
  auto es_shape3 = es::Shape(data2, 3); // DT_INT32

  auto range = es::Range(es_shape1, es_shape2, es_shape3, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(range, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto range_node = cg->FindFirstNodeMatchType(RANGE);
  ASSERT_NE(range_node, nullptr);
  auto op_desc = range_node->GetOpDesc();

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({-1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {7};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {3};
  ge::Shape shape2({dims_vec1});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto attr1 = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_expr1.GetDimNum(), 1);
  EXPECT_EQ(std::string(symbol_expr1.GetDim(0).Serialize().get()), "Ceiling(((s1 - s0) / (s2)))");

  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 4);
  const std::set<std::string> expect_guard = {"ExpectLt(s0, s1)", "ExpectNe(0, s0)", "ExpectNe(0, s1)", "ExpectNe(0, s2)"};
  for (auto &iter : guard_infos) {
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 2);
  const std::set<std::string> assert_guard = {"ExpectNe(0, s2)", "ExpectLt(0, s2)"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_select_host_compute) {
  std::vector<int32_t> cond_value = {1, 1, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {1, 3, 1};
  std::vector<int64_t> data_dims = {2, 3, 4};

  auto cond = builder_->CreateConst(cond_value, cond_dims);
  auto data0 = builder_->CreateConst(const0, data_dims);
  auto data1 = builder_->CreateConst(const1, data_dims);

  auto select = es::Select(cond, data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(select, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto select_node = cg->FindFirstNodeMatchType(SELECT);
  ASSERT_NE(select_node, nullptr);
  auto op_desc = select_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), SUCCESS);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0,
                                           c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  auto symbolic_tensor = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>()->symbolic_tensor;
  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}
/**
 *      Data0     Data1
 *         \     /
 *       BitwiseAnd
 *            |
 *            |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadCastGraphWithGuard) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto add = es::BitwiseAnd(data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(add, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

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
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);

  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "1");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "1");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s4");
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto add_node = cg->FindFirstNodeMatchType("BitwiseAnd");
  ASSERT_NE(add_node, nullptr);
  auto add_op_desc = add_node->GetOpDesc();
  ASSERT_NE(add_op_desc, nullptr);
  auto add_attr = add_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(add_attr, nullptr);
  auto add_symbol_shape = add_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(add_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(0).Serialize().get()), "s2");
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(add_symbol_shape.GetDim(3).Serialize().get()), "s4");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  const std::set<std::string> expect_guard = {"ExpectNe(1, s4)", "ExpectEq(3, s3)",
                                              "ExpectEq(s0, s2)", "ExpectNe(1, s1)",
                                              "ExpectNe(0, s0)", "ExpectNe(0, s1)",
                                              "ExpectNe(0, s2)", "ExpectNe(0, s3)",
                                              "ExpectNe(0, s4)"};
  ASSERT_EQ(guard_infos.size(), 9);
  for (auto &iter : guard_infos) {
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
}

IMPL_OP(Conv2D).PrivateAttr("padding", "");
TEST_F(SymbolicShapeInferenceST, test_conv2d_NHWC) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1, 1, 1};

  auto conv2d = es::Conv2D(data0, data1, nullptr, nullptr, strides, pads, dilations, 1, "NHWC", 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 28, 28, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NHWC, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {3, 3, 3, 16};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_HWCN, DT_FLOAT};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  auto conv_node = cg->FindFirstNodeMatchType("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  auto conv_op_desc = conv_node->GetOpDesc();
  ASSERT_NE(conv_op_desc, nullptr);
  conv_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
  conv_op_desc->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
  ge::AttrUtils::SetStr(conv_op_desc, "padding", "");
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto conv_attr = conv_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(conv_attr, nullptr);
  auto conv2d_symbol_shape = conv_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(conv2d_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(1).Serialize().get()), "Floor((s1 - (-1 + s4)))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(2).Serialize().get()), "Floor((s2 - (-1 + s5)))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(3).Serialize().get()), "s7");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 12);
  const std::set<std::string> expect_guard = {"ExpectNe(0, s7)", "ExpectNe(0, s1)",
                                              "ExpectNe(s2, s5)", "ExpectNe(0, s0)",
                                              "ExpectNe(0, s3)", "ExpectNe(s1, s4)",
                                              "ExpectNe(0, s2)", "ExpectNe((s3 / (s6)), 0)",
                                              "ExpectEq(0, Mod(s3, s6))",
                                              "ExpectNe(0, s6)", "ExpectNe(0, s5)", "ExpectNe(0, s4)"};
  for (auto &iter : guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 5);
  const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                              "ExpectLt(0, s5)",
                                              "ExpectLe(s4, s1)",
                                              "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                              "ExpectLe(s5, s2)"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_conv2d_NCHW) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1, 1, 1};

  auto conv2d = es::Conv2D(data0, data1, nullptr, nullptr, strides, pads, dilations, 1, "NCHW", 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 28, 28};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NCHW, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {16, 3, 3, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_NCHW, DT_FLOAT};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  auto conv_node = cg->FindFirstNodeMatchType("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  auto conv_op_desc = conv_node->GetOpDesc();
  ASSERT_NE(conv_op_desc, nullptr);
  conv_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  conv_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  conv_op_desc->MutableInputDesc(1)->SetFormat(FORMAT_NCHW);
  conv_op_desc->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto conv_attr = conv_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(conv_attr, nullptr);
  auto conv2d_symbol_shape = conv_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(conv2d_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(1).Serialize().get()), "s4");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(2).Serialize().get()), "Floor((s2 - (-1 + s6)))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(3).Serialize().get()), "Floor((s3 - (-1 + s7)))");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 12);
  const std::set<std::string> expect_guard = {"ExpectNe(0, s4)", "ExpectNe(0, s1)",
                                              "ExpectNe(s3, s7)", "ExpectNe(0, s0)",
                                              "ExpectNe(0, s3)", "ExpectNe(s2, s6)",
                                              "ExpectNe(0, s2)", "ExpectNe((s1 / (s5)), 0)",
                                              "ExpectEq(0, Mod(s1, s5))", "ExpectNe(0, s5)", "ExpectNe(0, s6)", "ExpectNe(0, s7)"};
  for (auto &iter : guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 5);
  const std::set<std::string> assert_guard = {"ExpectLt(0, s6)",
                                              "ExpectLt(0, s7)",
                                              "ExpectLe(s7, s3)",
                                              "ExpectEq(0, Mod(s4, (s1 / (s5))))",
                                              "ExpectLe(s6, s2)"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_conv2d_zero) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1, 1, 1};

  auto conv2d = es::Conv2D(data0, data1, nullptr, nullptr, strides, pads, dilations, 1, "NHWC", 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data_op_desc1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {0, 28, 28, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NHWC, DT_FLOAT};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {3, 3, 3, 16};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_HWCN, DT_FLOAT};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  auto conv_node = cg->FindFirstNodeMatchType("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  auto conv_op_desc = conv_node->GetOpDesc();
  ASSERT_NE(conv_op_desc, nullptr);
  conv_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
  conv_op_desc->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto conv_attr = conv_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(conv_attr, nullptr);
  auto conv2d_symbol_shape = conv_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(conv2d_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(0).Serialize().get()), "0");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(1).Serialize().get()), "Floor((s1 - (-1 + s4)))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(2).Serialize().get()), "Floor((s2 - (-1 + s5)))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(3).Serialize().get()), "s7");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 12);
  const std::set<std::string> expect_guard = {"ExpectEq(0, s0)",
                                              "ExpectNe(0, s3)",
                                              "ExpectNe(s1, s4)",
                                              "ExpectNe((s3 / (s6)), 0)",
                                              "ExpectEq(0, Mod(s3, s6))",
                                              "ExpectNe(0, s6)",
                                              "ExpectNe(s2, s5)",
                                              "ExpectNe(0, s1)", "ExpectNe(0, s2)",
                                              "ExpectNe(0, s4)", "ExpectNe(0, s5)",
                                              "ExpectNe(0, s7)"};
  for (auto &iter : guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 6);
  const std::set<std::string> assert_guard = {"ExpectLe(0, s7)",
                                              "ExpectLe(0, s0)",
                                              "ExpectLe(0, Floor((s1 - (-1 + s4))))",
                                              "ExpectLe(0, Floor((s2 - (-1 + s5))))",
                                              "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                              "ExpectLt(0, s5)",
                                              "ExpectLt(0, s4)"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_conv2d_padding_same) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> strides = {1, 2, 2, 1};
  std::vector<int64_t> pads = {-1, -1, -1, -1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};

  auto conv2d = es::Conv2D(data0, data1, nullptr, nullptr, strides, pads, dilations, 1, "NHWC", 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {48, 112, 112, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NHWC, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {7, 7, 3, 64};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_HWCN, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  auto conv_node = cg->FindFirstNodeMatchType("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  auto conv_op_desc = conv_node->GetOpDesc();
  ASSERT_NE(conv_op_desc, nullptr);
  conv_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
  conv_op_desc->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
  ge::AttrUtils::SetStr(conv_op_desc, "padding", "SAME");

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto conv_attr = conv_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(conv_attr, nullptr);
  auto conv2d_symbol_shape = conv_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(conv2d_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(1).Serialize().get()),
            "(1 + Floor((((2 * Floor(((-2 + s4) * Rational(1 , 2)))) + -1 + Mod((-2 + s4), 2) + s1 - (-1 + s4)) * Rational(1 , 2))))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(2).Serialize().get()),
            "(1 + Floor((((2 * Floor(((-2 + s5) * Rational(1 , 2)))) + -1 + Mod((-2 + s5), 2) + s2 - (-1 + s5)) * Rational(1 , 2))))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(3).Serialize().get()),
            "s7");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 15);
  const std::set<std::string> expect_guard = {"ExpectNe(0, s7)",
                                              "ExpectNe(0, s1)",
                                              "ExpectNe(((Mod((-2 + s4), 2) * Rational(1 , 2)) + (Rational(1 , 2) * s1) + Floor(((-2 + s4) * Rational(1 , 2)))), (Rational(1 , 2) * s4))",
                                              "ExpectNe(0, s0)",
                                              "ExpectNe(0, s3)",
                                              "ExpectNe(((Mod((-2 + s5), 2) * Rational(1 , 2)) + (Rational(1 , 2) * s2) + Floor(((-2 + s5) * Rational(1 , 2)))), (Rational(1 , 2) * s5))",
                                              "ExpectNe(0, s2)",
                                              "ExpectNe((s3 / (s6)), 0)",
                                              "ExpectEq(0, Mod(s3, s6))",
                                              "ExpectLt(2, s5)",
                                              "ExpectLe(Mod(s1, 2), 0)",
                                              "ExpectLt(2, s4)",
                                              "ExpectNe(0, s6)",
                                              "ExpectNe(0, s4)",
                                              "ExpectNe(0, s5)"};
  for (auto &iter : guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 9);
  const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                              "ExpectLe(0, Floor(((-2 + s4) * Rational(1 , 2))))",
                                              "ExpectLt(0, s5)",
                                              "ExpectLe(0, Floor(((-2 + s5) * Rational(1 , 2))))",
                                              "ExpectLe(0, (Floor(((-2 + s4) * Rational(1 , 2))) + Mod((-2 + s4), 2)))",
                                              "ExpectLe(0, (Floor(((-2 + s5) * Rational(1 , 2))) + Mod((-2 + s5), 2)))",
                                              "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                              "ExpectLe(s4, ((2 * Floor(((-2 + s4) * Rational(1 , 2)))) + Mod((-2 + s4), 2) + s1))",
                                              "ExpectLe(s5, ((2 * Floor(((-2 + s5) * Rational(1 , 2)))) + Mod((-2 + s5), 2) + s2))"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_conv2d_padding_valid) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> strides = {1, 2, 2, 1};
  std::vector<int64_t> pads = {-1, -1, -1, -1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};

  auto conv2d = es::Conv2D(data0, data1, nullptr, nullptr, strides, pads, dilations, 1, "NHWC", 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {48, 112, 112, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NHWC, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {7, 7, 3, 64};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_HWCN, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  auto conv_node = cg->FindFirstNodeMatchType("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  auto conv_op_desc = conv_node->GetOpDesc();
  ASSERT_NE(conv_op_desc, nullptr);
  conv_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
  conv_op_desc->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
  ge::AttrUtils::SetStr(conv_op_desc, "padding", "VALID");

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto conv_attr = conv_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(conv_attr, nullptr);
  auto conv2d_symbol_shape = conv_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(conv2d_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(1).Serialize().get()),
            "(1 + Floor(((-1 + s1 - (-1 + s4)) * Rational(1 , 2))))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(2).Serialize().get()),
            "(1 + Floor(((-1 + s2 - (-1 + s5)) * Rational(1 , 2))))");
  EXPECT_EQ(std::string(conv2d_symbol_shape.GetDim(3).Serialize().get()),
            "s7");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  ASSERT_EQ(guard_infos.size(), 12);
  const std::set<std::string> expect_guard = {
    "ExpectNe(0, s7)", "ExpectNe(0, s1)", "ExpectNe((Rational(1 , 2) * s2), (Rational(1 , 2) * s5))",
    "ExpectNe(0, s0)", "ExpectNe(0, s3)", "ExpectNe((Rational(1 , 2) * s1), (Rational(1 , 2) * s4))",
    "ExpectNe(0, s2)", "ExpectNe((s3 / (s6)), 0)", "ExpectEq(0, Mod(s3, s6))",
    "ExpectNe(0, s6)", "ExpectNe(0, s4)", "ExpectNe(0, s5)"};
  for (auto &iter : guard_infos) {
    const std::string guard_str = std::string(iter.expr.Serialize().get());
    std::cout << "guard info: " << guard_str << std::endl;
    EXPECT_NE(expect_guard.find(guard_str), expect_guard.end());
  }
  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 5);
  const std::set<std::string> assert_guard = {"ExpectLt(0, s4)",
                                              "ExpectLt(0, s5)",
                                              "ExpectLe(s4, s1)",
                                              "ExpectEq(0, Mod(s7, (s3 / (s6))))",
                                              "ExpectLe(s5, s2)"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_conv2d_padding_invalid) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> strides = {1, 2, 2, 1};
  std::vector<int64_t> pads = {-1, -1, -1, -1};
  std::vector<int64_t> dilations = {1, 1, 1, 1};

  auto conv2d = es::Conv2D(data0, data1, nullptr, nullptr, strides, pads, dilations, 1, "NHWC", 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(conv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {48, 112, 112, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_NHWC, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {7, 7, 3, 64};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_HWCN, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  auto conv_node = cg->FindFirstNodeMatchType("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  auto conv_op_desc = conv_node->GetOpDesc();
  ASSERT_NE(conv_op_desc, nullptr);
  conv_op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
  conv_op_desc->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
  conv_op_desc->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
  ge::AttrUtils::SetStr(conv_op_desc, "padding", "AABB");

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferenceST, test_split_no_input) {
  auto split = OP_CFG("Split").InCnt(0).OutCnt(2).Attr("num_split", 2).Build("split1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(split)->EDGE(0, 0)->NODE("NetOutput1", NETOUTPUT));
    CHAIN(NODE(split)->EDGE(1, 0)->NODE("NetOutput2", NETOUTPUT));
  };
  auto cg = ToComputeGraph(g1);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);
  auto split_node = cg->FindFirstNodeMatchType("Split");
  ASSERT_NE(split_node, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_split_no_symbolicvalue) {
  auto split =
      OP_CFG("Split").TensorDesc(FORMAT_ND, DT_INT32, {}).InCnt(2).OutCnt(2).Attr("num_split", 2).Build("split1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(split)->EDGE(0, 0)->NODE("NetOutput1", NETOUTPUT));
    CHAIN(NODE(split)->EDGE(1, 0)->NODE("NetOutput2", NETOUTPUT));
  };
  auto cg = ToComputeGraph(g1);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);
  auto split_node = cg->FindFirstNodeMatchType("Split");
  ASSERT_NE(split_node, nullptr);
  split_node->GetOpDescBarePtr()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  split_node->GetOpDescBarePtr()->MutableInputDesc(1)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_splitv_no_symbolicvalue) {
  auto splitv =
      OP_CFG("SplitV").TensorDesc(FORMAT_ND, DT_INT32, {}).InCnt(2).OutCnt(2).Attr("num_split", 2).Build("splitv1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(splitv)->EDGE(0, 0)->NODE("NetOutput1", NETOUTPUT));
    CHAIN(NODE(splitv)->EDGE(1, 0)->NODE("NetOutput2", NETOUTPUT));
  };
  auto cg = ToComputeGraph(g1);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);
  auto split_node = cg->FindFirstNodeMatchType("SplitV");
  ASSERT_NE(split_node, nullptr);
  split_node->GetOpDescBarePtr()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  split_node->GetOpDescBarePtr()->MutableInputDesc(1)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_splitv_input1_no_symbolic_value) {
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 7})
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
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto splitv = OP_CFG("SplitV")
                    .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1})
                    .InCnt(3)
                    .OutCnt(2)
                    .Attr("num_split", 3)
                    .Build("splitv1");

  DEF_GRAPH(graph) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(splitv));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(splitv));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(splitv));
    CHAIN(NODE(splitv)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(splitv)->EDGE(1, 0)->NODE("NetOutput", NETOUTPUT));
  };

  auto cg = ToComputeGraph(graph);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape0({Symbol(4), Symbol(7)});
  data_op_desc0->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape0);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape1({});
  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape1);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto sym0 = Symbol(1);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);

  gert::SymbolShape data_symbol_shape2({Symbol(1)});

  data_node2->GetOpDescBarePtr()
      ->MutableOutputDesc(0)
      ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolShape(data_symbol_shape2);
  data_node2->GetOpDescBarePtr()
      ->MutableOutputDesc(0)
      ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolicValue(std::move(ptr));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_splitv_input1_with_not_const_value) {
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 7})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {2})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto splitv = OP_CFG("SplitV")
                    .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1})
                    .InCnt(3)
                    .OutCnt(2)
                    .Attr("num_split", 2)
                    .Build("splitv1");

  DEF_GRAPH(graph) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(splitv));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(splitv));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(splitv));
    CHAIN(NODE(splitv)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(splitv)->EDGE(1, 0)->NODE("NetOutput", NETOUTPUT));
  };

  auto cg = ToComputeGraph(graph);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape0({Symbol(4), Symbol(7)});
  data_op_desc0->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape0);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape1({Symbol(2)});

  auto sym0 = Symbol("s0");
  auto sym1 = Symbol(-1);
  auto ptr1 = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr1, nullptr);
  ptr1->emplace_back(sym0);
  ptr1->emplace_back(sym1);

  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape1);
  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolicValue(
      std::move(ptr1));

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym0);

  gert::SymbolShape data_symbol_shape2({Symbol(1)});

  data_node2->GetOpDescBarePtr()
      ->MutableOutputDesc(0)
      ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolShape(data_symbol_shape2);
  data_node2->GetOpDescBarePtr()
      ->MutableOutputDesc(0)
      ->GetOrCreateAttrsGroup<SymbolicDescAttr>()
      ->symbolic_tensor.SetSymbolicValue(std::move(ptr));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_splitv_with_symbolic_value) {
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 7})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {2})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto splitv = OP_CFG("SplitV")
                    .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1})
                    .InCnt(3)
                    .OutCnt(2)
                    .Attr("num_split", 2)
                    .Build("splitv1");

  DEF_GRAPH(graph) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(splitv));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(splitv));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(splitv));
    CHAIN(NODE(splitv)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(splitv)->EDGE(1, 0)->NODE("NetOutput", NETOUTPUT));
  };

  auto cg = ToComputeGraph(graph);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDescBarePtr();
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape1({Symbol(2)});

  auto sym0 = Symbol(2);
  auto sym1 = Symbol(-1);
  auto ptr1 = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr1, nullptr);
  ptr1->emplace_back(sym0);
  ptr1->emplace_back(sym1);

  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape1);
  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolicValue(
      std::move(ptr1));

  auto sym2 = Symbol(0);
  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym2);

  auto data_op_desc2 = data_node2->GetOpDescBarePtr();
  ASSERT_NE(data_op_desc2, nullptr);
  gert::SymbolShape data_symbol_shape2({Symbol(1)});
  data_op_desc2->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape2);
  data_op_desc2->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolicValue(
      std::move(ptr));

  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);


  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {4, 7};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {1};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto splitv_node = cg->FindFirstNodeMatchType("SplitV");
  ASSERT_NE(splitv_node, nullptr);
  auto splitv_op_desc = splitv_node->GetOpDesc();
  ASSERT_NE(splitv_op_desc, nullptr);

  auto attr0 = splitv_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr0, nullptr);
  auto output0 = attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(output0.GetDimNum(), 2);
  EXPECT_EQ(std::string(output0.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(output0.GetDim(1).Serialize().get()), "s1");

  auto attr1 = splitv_op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto output1 = attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(output1.GetDimNum(), 2);
  EXPECT_EQ(std::string(output1.GetDim(0).Serialize().get()), "(-2 + s0)");
  EXPECT_EQ(std::string(output1.GetDim(1).Serialize().get()), "s1");

  const std::vector<SymbolCheckInfo> assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(assert_infos.size(), 1);
  const std::set<std::string> assert_guard = {"ExpectLe(2, s0)"};
  for (auto &iter : assert_infos) {
    EXPECT_NE(assert_guard.find(std::string(iter.expr.Serialize().get())), assert_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, test_splitv_with_wrong_size_splits) {
  auto data0 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 7})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {2})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto splitv = OP_CFG("SplitV")
                    .TensorDesc(FORMAT_ND, DT_INT32, {-1, -1})
                    .InCnt(3)
                    .OutCnt(2)
                    .Attr("num_split", 2)
                    .Build("splitv1");

  DEF_GRAPH(graph) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(splitv));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(splitv));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(splitv));
    CHAIN(NODE(splitv)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(splitv)->EDGE(1, 0)->NODE("NetOutput", NETOUTPUT));
  };

  auto cg = ToComputeGraph(graph);
  cg->TopologicalSorting();
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDescBarePtr();
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDescBarePtr();
  gert::SymbolShape data_symbol_shape1({Symbol(2)});

  auto sym0 = Symbol(5);
  auto sym1 = Symbol(-1);
  auto ptr1 = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr1, nullptr);
  ptr1->emplace_back(sym0);
  ptr1->emplace_back(sym1);

  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape1);
  data_op_desc1->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolicValue(
      std::move(ptr1));

  auto sym2 = Symbol(0);
  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto ptr = std::make_unique<std::vector<ge::Expression>>();
  ASSERT_NE(ptr, nullptr);
  ptr->emplace_back(sym2);

  auto data_op_desc2 = data_node2->GetOpDescBarePtr();
  ASSERT_NE(data_op_desc2, nullptr);
  gert::SymbolShape data_symbol_shape2({Symbol(1)});
  data_op_desc2->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(
      data_symbol_shape2);
  data_op_desc2->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolicValue(
      std::move(ptr));

  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);


  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {4, 7};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {1};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForAddLayerNorm) {
  auto x1 = builder_->CreateInput(0, "x1");
  auto x2 = builder_->CreateInput(1, "x2");
  auto gamma = builder_->CreateInput(2, "gamma");
  auto beta = builder_->CreateInput(3, "beta");
  x1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  x2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  gamma.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  beta.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(x1.GetCTensorHolder(), nullptr);
  ASSERT_NE(x2.GetCTensorHolder(), nullptr);
  ASSERT_NE(gamma.GetCTensorHolder(), nullptr);
  ASSERT_NE(beta.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto kone = ge::Symbol(1);

  auto addlayernorm = es::AddLayerNorm(x1, x2, gamma, beta, nullptr, 1e-5, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.y, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.mean, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.rstd, 2), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.x, 3), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("AddLayerNorm");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto attr2 = op_desc->GetOutputDesc(2).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr2, nullptr);
  auto attr3 = op_desc->GetOutputDesc(3).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr3, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, kone}).GetDims());
  EXPECT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, kone}).GetDims());
  EXPECT_EQ(attr3->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForAddLayerNormWithGuard) {
  auto x1 = builder_->CreateInput(0, "x1");
  auto x2 = builder_->CreateInput(1, "x2");
  auto gamma = builder_->CreateInput(2, "gamma");
  auto beta = builder_->CreateInput(3, "beta");
  ASSERT_NE(x1.GetCTensorHolder(), nullptr);
  ASSERT_NE(x2.GetCTensorHolder(), nullptr);
  ASSERT_NE(gamma.GetCTensorHolder(), nullptr);
  ASSERT_NE(beta.GetCTensorHolder(), nullptr);

  auto addlayernorm = es::AddLayerNorm(x1, x2, gamma, beta, nullptr, 1e-5, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.y, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.mean, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.rstd, 2), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(addlayernorm.x, 3), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("x1");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, 1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("x2");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, 3, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, 3, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, 3, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, 3, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("gamma");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node3 = cg->FindNode("beta");
  ASSERT_NE(data_node3, nullptr);
  auto data_op_desc3 = data_node3->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc3, "index", 3);
  data_op_desc3->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc3->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc3->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc3->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc3->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc3->MutableOutputDesc(0)->SetDataType(DT_INT64);

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

  std::vector<int64_t> dims_vec2 = {3, 2, 4, 3};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT64};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  std::vector<int64_t> dims_vec3 = {3, 2, 4, 3};
  ge::Shape shape3({dims_vec3});
  ge::TensorDesc td3{shape3, ge::FORMAT_ND, DT_INT64};
  td3.SetOriginShape(shape3);
  ge::Tensor tensor3{td3};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor3));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);

  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "1");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "1");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s3");

  auto data_symbol_attr2 = data_op_desc2->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr2, nullptr);
  auto symbol_shape2 = data_symbol_attr2->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape2.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape2.GetDim(0).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(1).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(2).Serialize().get()), "s6");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(3).Serialize().get()), "s7");

  auto data_symbol_attr3 = data_op_desc3->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr3, nullptr);
  auto symbol_shape3 = data_symbol_attr3->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape3.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape3.GetDim(0).Serialize().get()), "s8");
  EXPECT_EQ(std::string(symbol_shape3.GetDim(1).Serialize().get()), "s9");
  EXPECT_EQ(std::string(symbol_shape3.GetDim(2).Serialize().get()), "s10");
  EXPECT_EQ(std::string(symbol_shape3.GetDim(3).Serialize().get()), "s11");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto addlayernorm_node = cg->FindFirstNodeMatchType("AddLayerNorm");
  ASSERT_NE(addlayernorm_node, nullptr);
  auto addlayernorm_op_desc = addlayernorm_node->GetOpDesc();
  ASSERT_NE(addlayernorm_op_desc, nullptr);

  const std::vector<SymbolCheckInfo> guard_infos1 = shape_env_attr->GetAllSymbolAssertInfos();
  const std::set<std::string> expect_guard = {"ExpectEq(s4, s8)", "ExpectEq(s5, s9)",
                                              "ExpectEq(s10, s6)", "ExpectEq(s11, s7)"};
  ASSERT_EQ(guard_infos1.size(), 4);
  for (auto &iter : guard_infos1) {
    auto tmp = std::string(iter.expr.Serialize().get());
    EXPECT_NE(expect_guard.find(std::string(iter.expr.Serialize().get())), expect_guard.end());
  }
}

TEST_F(SymbolicShapeInferenceST, InferShapeForApplyAdagradD) {
  auto var = builder_->CreateInput(0, "var");
  auto accum = builder_->CreateInput(1, "accum");
  auto lr = builder_->CreateInput(2, "lr");
  auto grad = builder_->CreateInput(3, "grad");
  var.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  accum.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  lr.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  grad.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(var.GetCTensorHolder(), nullptr);
  ASSERT_NE(accum.GetCTensorHolder(), nullptr);
  ASSERT_NE(lr.GetCTensorHolder(), nullptr);
  ASSERT_NE(grad.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto applyadagradd = es::ApplyAdagradD(var, accum, lr, grad, true, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(applyadagradd.ref_var, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(applyadagradd.ref_accum, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("ApplyAdagradD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForApplyAdamD) {
  auto var = builder_->CreateInput(0, "var");
  auto m = builder_->CreateInput(1, "m");
  auto v = builder_->CreateInput(2, "v");
  auto beta1_power = builder_->CreateInput(3, "beta1_power");
  auto beta2_power = builder_->CreateInput(4, "beta2_power");
  auto lr = builder_->CreateInput(5, "lr");
  auto beta1 = builder_->CreateInput(6, "beta1");
  auto beta2 = builder_->CreateInput(7, "beta2");
  auto epsilon = builder_->CreateInput(8, "epsilon");
  auto grad = builder_->CreateInput(9, "beta1_power");
  var.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  m.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  v.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  beta1_power.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  beta2_power.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  lr.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  beta1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  beta2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  epsilon.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  grad.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(var.GetCTensorHolder(), nullptr);
  ASSERT_NE(m.GetCTensorHolder(), nullptr);
  ASSERT_NE(v.GetCTensorHolder(), nullptr);
  ASSERT_NE(beta1_power.GetCTensorHolder(), nullptr);
  ASSERT_NE(beta2_power.GetCTensorHolder(), nullptr);
  ASSERT_NE(lr.GetCTensorHolder(), nullptr);
  ASSERT_NE(beta1.GetCTensorHolder(), nullptr);
  ASSERT_NE(beta2.GetCTensorHolder(), nullptr);
  ASSERT_NE(epsilon.GetCTensorHolder(), nullptr);
  ASSERT_NE(grad.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto applyadamd = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(applyadamd.ref_var, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(applyadamd.ref_m, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(applyadamd.ref_v, 2), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("ApplyAdamD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto attr2 = op_desc->GetOutputDesc(2).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr2, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBNTrainingUpdate) {
  auto x = builder_->CreateInput(0, "x");
  auto sum = builder_->CreateInput(1, "sum");
  auto square_sum = builder_->CreateInput(2, "square_sum");
  auto scale = builder_->CreateInput(3, "scale");
  auto offset = builder_->CreateInput(4, "offset");
  auto mean = builder_->CreateInput(5, "mean");
  auto variance = builder_->CreateInput(6, "variance");
  x.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  sum.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  square_sum.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  scale.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s3"}));
  offset.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  mean.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  variance.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  ASSERT_NE(sum.GetCTensorHolder(), nullptr);
  ASSERT_NE(square_sum.GetCTensorHolder(), nullptr);
  ASSERT_NE(scale.GetCTensorHolder(), nullptr);
  ASSERT_NE(offset.GetCTensorHolder(), nullptr);
  ASSERT_NE(mean.GetCTensorHolder(), nullptr);
  ASSERT_NE(variance.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto bntrainingupdate = es::BNTrainingUpdate(x, sum, square_sum, scale, offset, mean, variance, 1, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingupdate.y, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingupdate.ref_mean, 1), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingupdate.ref_variance, 2), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingupdate.batch_mean, 3), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingupdate.batch_variance, 4), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("BNTrainingUpdate");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto attr2 = op_desc->GetOutputDesc(2).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr2, nullptr);
  auto attr3 = op_desc->GetOutputDesc(3).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr3, nullptr);
  auto attr4 = op_desc->GetOutputDesc(4).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr4, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s3}).GetDims());
  EXPECT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s3}).GetDims());
  EXPECT_EQ(attr3->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s3}).GetDims());
  EXPECT_EQ(attr4->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s3}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_BNTrainingReduce_NHWC) {
  auto x = builder_->CreateInput(0, "x");
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  auto s3 = ge::Symbol("s3");
  auto bntrainingreduce = es::BNTrainingReduce(x);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("x");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);

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
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_BNTrainingReduce_NCHW) {
  auto x = builder_->CreateInput(0, "x");
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  auto s1 = ge::Symbol("s1");
  auto bntrainingreduce = es::BNTrainingReduce(x);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("x");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);

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
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_BNTrainingReduce_NDHWC) {
  auto x = builder_->CreateInput(0, "x");
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  auto s4 = ge::Symbol("s4");
  auto bntrainingreduce = es::BNTrainingReduce(x);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("x");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NDHWC);

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
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s4}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s4}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_BNTrainingReduce_NDC1HWC0) {
  auto x = builder_->CreateInput(0, "x");
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  auto k1 = ge::Symbol(1);
  auto s2 = ge::Symbol("s2");
  auto s5 = ge::Symbol("s5");
  auto bntrainingreduce = es::BNTrainingReduce(x);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("x");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NDC1HWC0);

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
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({k1, k1, s2, k1, k1, s5}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({k1, k1, s2, k1, k1, s5}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_BNTrainingReduce_NCDHW) {
  auto x = builder_->CreateInput(0, "x");
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  auto s1 = ge::Symbol("s1");
  auto bntrainingreduce = es::BNTrainingReduce(x);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("x");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCDHW);

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
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto bnt_attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr, nullptr);
  auto bnt_attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(bnt_attr1, nullptr);
  EXPECT_EQ(bnt_attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
  EXPECT_EQ(bnt_attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, test_BNTrainingReduce_ND) {
  auto x = builder_->CreateInput(0, "x");
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  auto bntrainingreduce = es::BNTrainingReduce(x);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.sum, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bntrainingreduce.square_sum, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("x");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1, -1}));

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
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForGelu) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto gelu = es::Gelu(data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gelu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Gelu");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1, s2, s0}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForFusedMulAddN) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s0"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto fusedMulAddN = es::FusedMulAddN(data0, data1, data2);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fusedMulAddN, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("FusedMulAddN");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s1, s2, s0}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForFloorDiv) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto floorDiv = es::FloorDiv(data0, data1);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(floorDiv, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("FloorDiv");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForFloorMod) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto floorMod = es::FloorMod(data0, data1);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(floorMod, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("FloorMod");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForLayerNormXBackpropV2) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  auto data3 = builder_->CreateInput(3, "data3");
  auto data4 = builder_->CreateInput(4, "data4");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data3.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  data4.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  ASSERT_NE(data3.GetCTensorHolder(), nullptr);
  ASSERT_NE(data4.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto layernormxbackparopv2 = es::LayerNormXBackpropV2(data0, data1, data2, data3, data4);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormxbackparopv2.pd_x, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(layernormxbackparopv2.res_for_gamma, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("LayerNormXBackpropV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1, s2}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferShapeForL2Loss) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto l2Loss = es::L2Loss(data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(l2Loss, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("L2Loss");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDimNum(), 0);
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
TEST_F(SymbolicShapeInferenceST, InferShapeForSelectV2Success) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);

  const auto selectv2 = es::SelectV2(data0, data1, data2);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(selectv2, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 2, 2}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 2, 2}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 2, 2}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 2, 2}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 3, 2, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 3, 2, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 3, 2, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 3, 2, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node2 = cg->FindNode("data2");
  ASSERT_NE(data_node2, nullptr);
  auto data_op_desc2 = data_node2->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc2, "index", 2);
  data_op_desc2->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc2->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1, -1}));
  data_op_desc2->MutableOutputDesc(0)->SetDataType(DT_INT64);

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

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);

  auto data_symbol_attr0 = data_op_desc0->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "2");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s3");

  auto data_symbol_attr2 = data_op_desc2->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr2, nullptr);
  auto symbol_shape2 = data_symbol_attr2->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape2.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape2.GetDim(0).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(1).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(2).Serialize().get()), "s6");
  EXPECT_EQ(std::string(symbol_shape2.GetDim(3).Serialize().get()), "s7");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto select_node = cg->FindFirstNodeMatchType("SelectV2");
  ASSERT_NE(select_node, nullptr);
  auto select_op_desc = select_node->GetOpDesc();
  ASSERT_NE(select_op_desc, nullptr);
  auto select_attr = select_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(select_attr, nullptr);
  auto select_symbol_shape = select_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(select_symbol_shape.GetDimNum(), 4);
  EXPECT_EQ(std::string(select_symbol_shape.GetDim(0).Serialize().get()), "s4");
  EXPECT_EQ(std::string(select_symbol_shape.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(std::string(select_symbol_shape.GetDim(2).Serialize().get()), "2");
  EXPECT_EQ(std::string(select_symbol_shape.GetDim(3).Serialize().get()), "2");
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSelectV2Fail) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s3", "s2"}));

  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto selectv2 = es::SelectV2(data0, data1, data2);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(selectv2, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBNTrainingUpdateGrad) {
  auto grads = builder_->CreateInput(0, "grads");
  auto x = builder_->CreateInput(1, "x");
  auto batch_mean = builder_->CreateInput(2, "batch_mean");
  auto batch_variance = builder_->CreateInput(3, "batch_variance");
  grads.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  x.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  batch_mean.SetOriginSymbolShape(std::vector<const char *>({"s3"}));
  batch_variance.SetOriginSymbolShape(std::vector<const char *>({"s3"}));

  ASSERT_NE(grads.GetCTensorHolder(), nullptr);
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  ASSERT_NE(batch_mean.GetCTensorHolder(), nullptr);
  ASSERT_NE(batch_variance.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto bn_training_update_grad_output = es::BNTrainingUpdateGrad(grads, x, batch_mean, batch_variance, 0.0001);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bn_training_update_grad_output.diff_scale, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(bn_training_update_grad_output.diff_offset, 1), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("BNTrainingUpdateGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s3}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferSymbolicShapeForDiagPartDSuccess) {
  auto x = builder_->CreateInput(0, "x");
  auto assist = builder_->CreateInput(1, "assist");
  x.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  assist.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  ASSERT_NE(assist.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto diag_partd = es::DiagPartD(x, assist);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(diag_partd, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("DiagPartD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), gert::SymbolShape({s0, s1}).GetDims());
}

TEST_F(SymbolicShapeInferenceST, InferSymbolicShapeForDiagPartDFail_input_dims_num_ne_assist_dims_num) {
  auto x = builder_->CreateInput(0, "x");
  auto assist = builder_->CreateInput(1, "assist");
  x.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  assist.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  ASSERT_NE(assist.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto diag_partd = es::DiagPartD(x, assist);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(diag_partd, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferSymbolicShapeForDiagPartDFail_input_dims_ne_assist_dims) {
  auto x = builder_->CreateInput(0, "x");
  auto assist = builder_->CreateInput(1, "assist");
  x.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  assist.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s3", "s0"}));
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  ASSERT_NE(assist.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto diag_partd = es::DiagPartD(x, assist);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(diag_partd, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferSymbolicShapeForDiagPartDFail_input_dims_is_not_even) {
  auto x = builder_->CreateInput(0, "x");
  auto assist = builder_->CreateInput(1, "assist");
  x.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  assist.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s3", "s0"}));
  ASSERT_NE(x.GetCTensorHolder(), nullptr);
  ASSERT_NE(assist.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto diag_partd = es::DiagPartD(x, assist);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(diag_partd, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForEluGrad) {
  auto grads = builder_->CreateInput(0, "grads");
  auto activations = builder_->CreateInput(1, "activations");
  grads.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  activations.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(grads.GetCTensorHolder(), nullptr);
  ASSERT_NE(activations.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto eluGrad = es::EluGrad(grads, activations);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(eluGrad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("EluGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftmaxGrad) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int64_t> axes = {-1};
  auto softmaxGrad = es::SoftmaxGrad(data0, data1, axes);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softmaxGrad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SoftmaxGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSoftplus) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto softplus = es::Softplus(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(softplus, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Softplus");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTanhGrad) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto tanhGrad = es::TanhGrad(data0, data1, false);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tanhGrad, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("TanhGrad");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, s0}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForPadV3Success) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {3, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto padv3 = es::PadV3(data0, const0, nullptr, "edge",true);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(padv3, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("PadV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto expect_shape = gert::SymbolShape({s2 + Symbol(1) + Symbol(2),
    s1 + Symbol(2) + Symbol(1), s0 + Symbol(1) + Symbol(1)});
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), expect_shape);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForPadV3Fail_padding_type_err) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {3, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto padv3 = es::PadV3(data0, const0, nullptr, "edge",true);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(padv3, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("PadV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT16);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForPadV3Fail_padding_value_dim_error1) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {1, 2, 2, 1};
  std::vector<int64_t> const_dim = {2, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto padv3 = es::PadV3(data0, const0, nullptr, "edge",true);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(padv3, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("PadV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForPadV3Fail_padding_value_dim_error2) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {1, 2, 2};
  std::vector<int64_t> const_dim = {1, 3};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto padv3 = es::PadV3(data0, const0, nullptr, "edge",true);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(padv3, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("PadV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

REG_OP(PadV3)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(constant_values, TensorType::BasicType())
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(mode, String, "constant")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3)

TEST_F(SymbolicShapeInferenceST, InferShapeForPadV3Fail_padding_unsupported) {
auto data0 = OP_CFG("Data")
                 .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 2, 3, 4})
                 .InCnt(0)
                 .OutCnt(1)
                 .InNames({"x"})
                 .OutNames({"y"})
                 .Build("data0");
auto padv3 = OP_CFG("PadV3").InCnt(1).OutCnt(1).Build("PadV3");
DEF_GRAPH(g1) {
  CHAIN(NODE(data0)->EDGE(0, 0)->NODE(padv3));
  CHAIN(NODE(padv3)->NODE("NetOutput", NETOUTPUT));
};
auto graph = ToComputeGraph(g1);
graph->TopologicalSorting();
ASSERT_NE(graph, nullptr);
auto padd_node = graph->FindFirstNodeMatchType("PadV3");
ASSERT_NE(padd_node, nullptr);
padd_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
std::string mode = "edge";
padd_node->GetOpDesc()->SetAttr("mode",GeAttrValue::CreateFrom<std::string>(mode));
padd_node->GetOpDesc()->SetAttr("paddings_contiguous",GeAttrValue::CreateFrom<bool>(false));
auto data_node = graph->FindNode("data0");
ASSERT_NE(data_node, nullptr);
gert::SymbolShape symol_shape({Symbol("s0"), Symbol("s1"), Symbol("s2")});
data_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.
    SetSymbolShape(symol_shape);
SymbolicShapeInference ssi;
ASSERT_EQ(ssi.Infer(graph), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForUnsortedSegmentMinSuccess) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  auto data1 = builder_->CreateInput(1, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {3};
  std::vector<int64_t> const_dim = {1};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto unsortedsegmentmin = es::UnsortedSegmentMin(data0, data1, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsortedsegmentmin, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("UnsortedSegmentMin");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto expect_shape = gert::SymbolShape({Symbol(3), s1, s0});
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), expect_shape);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForUnsortedSegmentMinFail_num_segments_shape_ne_1) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  auto data1 = builder_->CreateInput(1, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {3, 3, 2, 2};
  std::vector<int64_t> const_dim = {2, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto unsortedsegmentmin = es::UnsortedSegmentMin(data0, data1, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsortedsegmentmin, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("UnsortedSegmentMin");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForUnsortedSegmentMinFail_num_segments_value_size_ne_1) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  auto data1 = builder_->CreateInput(1, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {3, 3};
  std::vector<int64_t> const_dim = {2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto unsortedsegmentmin = es::UnsortedSegmentMin(data0, data1, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsortedsegmentmin, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("UnsortedSegmentMin");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForUnsortedSegmentMinFail_num_segments_dtype_error) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0"}));
  auto data1 = builder_->CreateInput(1, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<int32_t> const_data0 = {3};
  std::vector<int64_t> const_dim = {1};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  auto unsortedsegmentmin = es::UnsortedSegmentMin(data0, data1, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsortedsegmentmin, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("UnsortedSegmentMin");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT16);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSparseSoftmaxCrossEntropyWithLogits) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1"}));
  auto data1 = builder_->CreateInput(1, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto res = es::SparseSoftmaxCrossEntropyWithLogits(data0, data1);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(res.loss, 0), 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(res.backprop, 1), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("SparseSoftmaxCrossEntropyWithLogits");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0}));
  auto attr1 = op_desc->GetOutputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  EXPECT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForAssign) {

  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto assign = es::Assign(data0, data1, false, false);
  // auto assign = es::Abs(data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(assign, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Assign");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2, s3}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForGatherV2) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int64_t> const_data = {2, 3};
  auto const0 = builder_->CreateVector(const_data);
  auto const1 = builder_->CreateScalar(static_cast<int32_t>(-1));
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);
  ASSERT_NE(const1.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  auto gatherv2 = es::GatherV2(data0, const0, const1, 0,true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gatherv2, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("GatherV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  op_desc->MutableInputDesc(2)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2, Symbol(2)}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForGatherNd) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int64_t> const_data = {2, 3};
  auto const0 = builder_->CreateVector(const_data);
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);

  auto gatherNd = es::GatherNd(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gatherNd, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("GatherNd");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({ge::Symbol("s2"), ge::Symbol("s3")}));
}

REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)

TEST_F(SymbolicShapeInferenceST, InferShapeForMatMul) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  auto data2 = builder_->CreateInput(2, "data2");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  ASSERT_NE(data2.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s1", "s0"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1"}));
  data2.SetOriginSymbolShape(std::vector<const char *>({"s2"}));

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto matmul = es::MatMul(data0, data1, data2, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matmul, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("MatMul");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s2}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSqueeze) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "1", "s2", "1"}));

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  const vector<int64_t> axis = {2, 4};
  auto squeeze = es::Squeeze(data0, axis);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squeeze, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("Squeeze");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForSqueezeV3) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "1", "s2", "1"}));
  std::vector<int32_t> const_data0 = {2, 4};
  std::vector<int64_t> const_dim = {1, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto squeezeV3 = es::SqueezeV3(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squeezeV3, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("SqueezeV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForUnsqueezeV3) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  std::vector<int32_t> const_data0 = {2, 4};
  std::vector<int64_t> const_dim = {1, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto unsqueezeV3 = es::UnsqueezeV3(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsqueezeV3, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("UnsqueezeV3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({s0, s1, ge::Symbol(1), s2, ge::Symbol(1)}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTileDInDimsGtMultiSize) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto symbol_two = ge::Symbol(2);
  auto symbol_four = ge::Symbol(4);
  const vector<int64_t> multiples = {2, 4};
  auto tiled = es::TileD(data0, multiples);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tiled, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("TileD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1 * symbol_two, s2 * symbol_four}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTileDInDimsLtMultiSize) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto symbol_one = ge::Symbol(1);
  auto symbol_two = ge::Symbol(2);
  auto symbol_three = ge::Symbol(3);
  auto symbol_four = ge::Symbol(4);
  const vector<int64_t> multiples = {1, 2, 3, 4};
  auto tiled = es::TileD(data0, multiples);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tiled, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("TileD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({symbol_one, symbol_two * s0, symbol_three * s1, symbol_four * s2}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTileDInDimsEqMultiSize) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto symbol_one = ge::Symbol(1);
  auto symbol_two = ge::Symbol(2);
  auto symbol_three = ge::Symbol(3);
  const vector<int64_t> multiples = {1, 2, 3};
  auto tiled = es::TileD(data0, multiples);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tiled, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("TileD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0 * symbol_one, s1 * symbol_two, s2 * symbol_three}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTileDMultiSizeGt8) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  const vector<int64_t> multiples = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto tiled = es::TileD(data0, multiples);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tiled, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("TileD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTranspose) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  std::vector<int32_t> perm = {1, 0, 2};
  int64_t perm_dim = 3;
  auto perm_tensor = builder_->CreateConst(perm, std::vector<int64_t>{perm_dim});

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto transpose = es::Transpose(data0, perm_tensor);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(transpose, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("Transpose");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  {
    op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
    SymbolicShapeInference ssi;
    ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

    const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s1, s0, s2}));
  }
  {
    op_desc->MutableInputDesc(1)->SetDataType(DT_INT64);
    SymbolicShapeInference ssi;
    ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

    const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s1, s0, s2}));
  }
  {
    op_desc->MutableInputDesc(1)->SetDataType(DT_INT16);
    SymbolicShapeInference ssi;
    ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
  }
}

TEST_F(SymbolicShapeInferenceST, InferShapeForTransposeD) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  std::vector<int64_t> perm = {1, 0, 2};

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");

  auto transpose = es::TransposeD(data0, perm);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(transpose, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("TransposeD");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s1, s0, s2}));
}

REG_OP(Unpack)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num, Int)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(Unpack)
TEST_F(SymbolicShapeInferenceST, InferShapeForUnpack) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node = cg->FindNode("data0");
  ASSERT_NE(data_node, nullptr);
  GeTensorDesc unpack_input_desc(GeShape({-1, -1, 3}), FORMAT_ND, DT_INT32);
  auto unpack_op_desc = std::make_shared<OpDesc>("unpack_0", UNPACK);
  unpack_op_desc->AppendIrAttrName("num");
  AttrUtils::SetInt(unpack_op_desc, "num", 3);
  unpack_op_desc->AppendIrAttrName("axis");
  AttrUtils::SetInt(unpack_op_desc, "axis", 2);
  unpack_op_desc->AddInputDesc(unpack_input_desc);
  GeTensorDesc unpack_output_desc(GeShape({-1, -1, 6}), FORMAT_ND, DT_INT32);
  for (auto i = 0; i < 3; ++i) {
    unpack_op_desc->AddOutputDesc(unpack_output_desc);
  }
  auto unpack_node = cg->AddNode(unpack_op_desc);
  ASSERT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), unpack_node->GetInDataAnchor(0)), SUCCESS);

  auto node = cg->FindFirstNodeMatchType(UNPACK);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  for (auto i = 0; i < 3; i++) {
    const auto attr = op_desc->GetOutputDesc(i).GetAttrsGroup<SymbolicDescAttr>();
    ASSERT_NE(attr, nullptr);
    EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1}));
  }
}

TEST_F(SymbolicShapeInferenceST, InferShapeForGather) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2", "s3"}));
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<int64_t> const_data = {2, 3};
  auto const0 = builder_->CreateVector(const_data);
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto gather = es::Gather(mul, const0, false, 0,true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("Gather");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), s1, s2, s3}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForBroadCastToGraphWithGuard) {
  auto data0 = builder_->CreateInput(0, "data0");
  std::vector<int32_t> const_data0 = {3, 2, 3, 4};
  std::vector<int64_t> const_dim = {4};
  auto const0 = builder_->CreateConst(const_data0, const_dim);
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto broadcast_to = es::BroadcastTo(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(broadcast_to, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto op_desc0 = data_node0->GetOpDesc();
  ASSERT_NE(op_desc0, nullptr);
  ge::AttrUtils::SetInt(op_desc0, "index", 0);
  op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, 3, 1}));
  op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 3, 1}));
  op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  op_desc0->MutableInputDesc(0)->SetOriginDataType(DT_INT64);
  op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, 3, 1}));
  op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 3, 1}));
  op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);
  op_desc0->MutableOutputDesc(0)->SetOriginDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {1, 3, 1};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  const std::set<std::string> expect_guard = {"LogicOr(ExpectEq(1, s0), ExpectEq(2, s0))", "ExpectNe(0, s0)"};
  std::vector<Expression> expect_output_shape = {Symbol(3), Symbol(2), Symbol(3), Symbol(4)};

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr1 = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(attr1, nullptr);
  ShapeEnvGuarder guard(attr1);

  auto node_ptr = cg->FindFirstNodeMatchType("BroadcastTo");
  ASSERT_NE(node_ptr, nullptr);
  auto op_desc = node_ptr->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto symbol_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(symbol_shape.GetDimNum(), expect_output_shape.size());
  EXPECT_TRUE(symbol_shape.GetDims() == expect_output_shape);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
  EXPECT_EQ(guard_infos.size(), expect_guard.size());
  for (auto &iter : guard_infos) {
    EXPECT_TRUE(expect_guard.find(std::string(iter.expr.Serialize().get())) != expect_guard.end());
  }
}

/*
 * if的条件输入是data
 */
TEST_F(SymbolicShapeInferenceST, NestIfGraphTest) {
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

  ASSERT_NE(root_graph->FindNode("if1"), nullptr);

  auto sqrt_node = root_graph->FindNode("then_subgraph_sqrt1");
  ASSERT_EQ(sqrt_node, nullptr);
  DisableSliceScheduleEnv();
}

TEST_F(SymbolicShapeInferenceST, NestCaseGraphTest) {
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

  ASSERT_NE(root_graph->FindNode("case1"), nullptr);

  auto sqrt_node = root_graph->FindNode("batch2_subgraph_sqrt1");
  ASSERT_EQ(sqrt_node, nullptr);
  DisableSliceScheduleEnv();
}

// index 小于0
TEST_F(SymbolicShapeInferenceST, NestCaseGraphTestNegativeIndex) {
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
  auto input0 = BuildGeTensor<int32_t, DT_INT32>({}, {-1});
  auto input1 =  BuildGeTensor<int32_t, DT_INT32>({2, 3}, {});
  input_vec.emplace_back(input0);
  input_vec.emplace_back(input1);

  SymbolicShapeSymbolizer symboilzer;
  ASSERT_EQ(symboilzer.Symbolize(root_graph, input_vec), SUCCESS);
  ASSERT_EQ(SymbolicInfoPreProcessor::Run(root_graph, input_vec), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(root_graph), SUCCESS);

  ASSERT_NE(root_graph->FindNode("case1"), nullptr);

  auto sqrt_node = root_graph->FindNode("batch2_subgraph_sqrt1");
  ASSERT_EQ(sqrt_node, nullptr);
  DisableSliceScheduleEnv();
}

// if的条件输入是其它算子的输出
TEST_F(SymbolicShapeInferenceST, NestIfGraph1Test) {
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

TEST_F(SymbolicShapeInferenceST, AippOpTest) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1", AIPPDATA);
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto aipp = es::Aipp(data0, data1, "");
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(aipp, 0), 0);
  auto graph = builder_->BuildAndReset();
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

TEST_F(SymbolicShapeInferenceST, MulitBatchOpTest) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");

  std::vector<int32_t> dims{0, 1, 2};
  auto const_int32_list = builder_->CreateConst(dims, {static_cast<int64_t>(dims.size())});
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto mapindex = es::MapIndex(data1, const_int32_list, nullptr, false);
  auto add = es::Add(data0, mapindex);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(add, 0), 0);
  auto graph = builder_->BuildAndReset();
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

TEST_F(SymbolicShapeInferenceST, RefDataCopyOpTest) {
  auto data0 = builder_->CreateInput(0, "data0", REFDATA);
  auto data1 = builder_->CreateInput(1, "data1", REFDATA);
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto assign = es::Assign(data0, data0, true, true);
  auto add = es::Add(assign, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(add, 0), 0);
  auto graph = builder_->BuildAndReset();
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

// 动态分档场景符号化推导
TEST_F(SymbolicShapeInferenceST, MultiBatchInferTest) {
  GetLocalOmgContext().need_multi_batch = true;
  auto graph = gert::ShareGraph::BuildMultiBatchShapesGraph();

  std::vector<ge::GeTensor> input_vec;
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));
  input_vec.emplace_back(BuildTensor({8, 3, 1, 100}, FORMAT_NCHW, DT_FLOAT));

  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, input_vec), SUCCESS);

  std::vector<Expression> expect_shape = {Symbol(8), Symbol(3), Symbol(1), Symbol(100)};

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

TEST_F(SymbolicShapeInferenceST, InferShapeForFlattenV2) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3"}));

  int64_t axis = 2;
  int64_t end_axis = 3;
  auto flattenV2 = es::FlattenV2(data0, axis, end_axis);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(flattenV2, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto node = cg->FindFirstNodeMatchType("FlattenV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");

  const auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2 * s3}));
}

/**
 *      Data0    Const(float)
 *        |  \    /
 *        |   Adds
 *        |   /
 *         Mul
 *          |
 *         Relu
 *          |
 *       NetOutput
 */

TEST_F(SymbolicShapeInferenceST, InferShapeForSimpleGraphWithAdds) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "1", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  float const_float = 1.0f;

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto one = ge::Symbol(1);

  auto adds = es::Adds(data0, const_float);
  auto mul = es::Mul(data0, adds);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s2, s1, one, s0}));
}

TEST_F(SymbolicShapeInferenceST, Expand_symbolic_dims_equal) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "4"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);

  std::vector<int32_t> const_data0 = {1, 1, 1, 4};
  std::vector<int64_t> const_dim = {4};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto expand = es::Expand(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Expand");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto s3 = ge::Symbol("s3");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2, Symbol(4)}));
}

TEST_F(SymbolicShapeInferenceST, Expand_symbolic_dims_greater_than_input) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int32_t> const_data0 = {0, 1, 1, 2};
  std::vector<int64_t> const_dim = {4};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto expand = es::Expand(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Expand");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(0),s0, s1, Symbol(2)}));
}

TEST_F(SymbolicShapeInferenceST, Expand_target_less_than_input) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "1", "1"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int32_t> const_data0 = {1, 2};
  std::vector<int64_t> const_dim = {2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto expand = es::Expand(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForExpandSymbolInputEq1) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "s1", "2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int32_t> const_data0 = {0, 1, 1, 2};
  std::vector<int64_t> const_dim = {4};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto expand = es::Expand(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindFirstNodeMatchType("Expand");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);

  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(0), Symbol(1), s1, Symbol(2)}));
}

TEST_F(SymbolicShapeInferenceST, InferShapeForExpandSymbolInputGt1) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "s1", "2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int32_t> const_data0 = {0, 1, 1, 3};
  std::vector<int64_t> const_dim = {4};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto expand = es::Expand(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, InferShapeForExpandSymbolInputException) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "2"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  std::vector<int32_t> const_data0 = {1, 3};
  std::vector<int64_t> const_dim = {2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  auto expand = es::Expand(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

/**
 *      diagonal    k(0)    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |               |
 *                 \|           |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_single_diag) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s1}));
}

/**
 *      diagonal    k([-1,1])    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |               |                |               |
 *            \     |               |                |               |
 *             \    |               |                |               |
 *              \   |               |                |               |
 *               \  |               |                |               |
 *                \ |               |                |               |
 *                 \|               |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_multi_diag) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "3", "s1"}));
  
  auto k = builder_->CreateVector(std::vector<int32_t>{-1, 1});
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, Symbol(1) + s1, Symbol(1) + s1}));
}

/**
 *      diagonal    k(0)    num_rows(5)    num_cols(6)    padding_value(0.0)
 *           \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |.              |
 *                 \|                     |                |               |
 *              MatrixDiagV2
 *                   |
|              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_with_explicit_dims) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "5"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(5));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(5));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, Symbol(5), Symbol(5)}));
}

/**
 *      diagonal    k(-2)    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |               |
 *                 \|           |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_negative_k) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(-2));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, Symbol(2) + s1, s1}));
}

/**
 *      diagonal    k(3)    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *                     \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |               |
 *                 \|           |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_positive_k) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(3));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, Symbol(3) + s1}));
}

/**
 *      diagonal    k([0,0])    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |               |                |               |
 *            \     |               |                |               |
 *             \    |               |                |               |
 *              \   |               |                |               |
 *               \  |               |                |               |
 *                \ |               |                |               |
 *                 \|               |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_single_element_k) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "1", "s1"}));
  
  auto k = builder_->CreateVector(std::vector<int32_t>{0, 0});
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, Symbol(1), s1, s1}));
}

/**
 *      diagonal    k([-2,2])    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |                 |                |               |
 *            \     |                 |                |               |
 *             \    |                 |                |               |
 *              \   |                 |                |               |
 *               \  |                 |                |               |
 *                \ |                 |                |               |
 *                 \|                 |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_larger_diag_range) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "5", "s1"}));
  
  auto k = builder_->CreateVector(std::vector<int32_t>{-2, 2});
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, Symbol(2) + s1, Symbol(2) + s1}));
}

/**
 *      diagonal    k(0)    num_rows(5)    num_cols(-1)    padding_value(0.0)
 *           \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |               |
 *                 \|           |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_specify_num_rows_only) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "6"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(6));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(6));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, Symbol(6), Symbol(6)}));
}

/**
 *      diagonal    k(0)    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |               |
 *                 \|           |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_3d_input) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s2, s2}));
}

/**
 *      diagonal    k([-1,1])    num_rows(-1)    num_cols(-1)    padding_value(0.0)
 *           \      |               |                |               |
 *            \     |               |                |               |
 *             \    |               |                |               |
 *              \   |               |                |               |
 *               \  |               |                |               |
 *                \ |               |                |               |
 *                 \|               |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_4d_input_multi_diag) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "3", "s2"}));
  
  auto k = builder_->CreateVector(std::vector<int32_t>{-1, 1});
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, Symbol(1) + s2, Symbol(1) + s2}));
}

/**
 *      diagonal    k(0)    num_rows(-1)    num_cols(0)    padding_value(0.0)
 *           \      |           |                |               |
 *            \     |           |                |               |
 *             \    |           |                |               |
 *              \   |           |                |               |
 *               \  |           |                |               |
 *                \ |           |                |               |
 *                 \|           |                |               |
 *              MatrixDiagV2
 *                   |
 *              NetOutput
 */
TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_zero_num_cols) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows = builder_->CreateScalar(static_cast<int32_t>(-1));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(0));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  
  auto node = cg->FindFirstNodeMatchType("MatrixDiagV2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0, s1, s1}));
}

TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_col_less_than_min) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1" "3"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows =  builder_->CreateScalar(static_cast<int32_t>(3));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(0));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeInferenceST, test_matrixdiagv2_row_less_than_min) {
  auto diagonal = builder_->CreateInput(0, "diagonal");
  diagonal.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1" "3"}));
  
  auto k = builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_rows =  builder_->CreateScalar(static_cast<int32_t>(0));
  auto num_cols = builder_->CreateScalar(static_cast<int32_t>(3));
  auto padding_value = builder_->CreateScalar(static_cast<float>(0.0f));
  
  auto matrix_diag_v2 = es::MatrixDiagV2(diagonal, k, num_rows, num_cols, padding_value);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(matrix_diag_v2, 0), 0);
  
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
} 
}  // namespace ge
