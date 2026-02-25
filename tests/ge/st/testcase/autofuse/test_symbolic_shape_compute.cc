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
#include "common/plugin/ge_make_unique_util.h"
#include "es_ge_test_ops_c.h"
#include "es_ge_test_ops.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "framework/common/types.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "common/env_path.h"
#include "compliant_node_builder.h"
#include "common/topo_checker.h"
#include "graph/utils/node_adapter.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "stub/gert_runtime_stub.h"

namespace ge {
using ge::es::EsTensorHolder;
class SymbolicShapeComputeST : public testing::Test {
public:
protected:
  static void SetUpTestSuite() {
    setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
    setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp", 1);
    dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
    auto work_path = EnvPath().GetAirBasePath() + "/output";
    setenv("ASCEND_WORK_PATH", work_path.c_str(), 1);
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  }
  static void TearDownTestSuite() {
  }
  void SetUp() override {
    builder_ = std::make_unique<ge::es::EsGraphBuilder>("Hello");
  }
  void TearDown() override {
    builder_.reset();
  }
  std::unique_ptr<ge::es::EsGraphBuilder> builder_;
};

graphStatus UnsqueezeTest(gert::InferSymbolComputeContext *context) {
  return ge::SUCCESS;
}

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
TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraph) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
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

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<std::string> expect_dim = {"s0", "s4", "s2", "s5"};
  ASSERT_EQ(result.size(), expect_dim.size());
  for (size_t i = 0UL; i < result.size(); i++) {
    EXPECT_EQ(std::string(result[i].Serialize().get()), expect_dim[i]);
  }
}

TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraphUnkownDimNum) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<std::string> expect_dim = {"s1", "s5", "s3", "s6"};
  ASSERT_EQ(result.size(), expect_dim.size());
  for (size_t i = 0UL; i < result.size(); i++) {
    EXPECT_EQ(std::string(result[i].Serialize().get()), expect_dim[i]);
  }
}

TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraphInvalidShapeDynamic) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {-3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraphInvalidShapeStatic) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({2, 3, 4, 4}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({2, 3, 4, 4}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({2, 3, 4, 4}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({2, 3, 4, 4}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {-3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(3), Symbol(4), Symbol(4)};
  ASSERT_EQ(result.size(), expect_dim.size());
  ASSERT_EQ(result, expect_dim);
}

TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraphInputData0IsEmpty) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<Expression> expect_dim = {Symbol("s1"), Symbol(1), Symbol("s2")};
  ASSERT_EQ(result.size(), expect_dim.size());
  ASSERT_EQ(result, expect_dim);
}

TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraphScalar) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape());
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape());
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape());
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape());
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {-3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 1, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto result = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  std::vector<Expression> expect_dim = {Symbol("s0"), Symbol(1), Symbol("s1")};
  ASSERT_EQ(result.size(), expect_dim.size());
  ASSERT_EQ(result, expect_dim);
}

TEST_F(SymbolicShapeComputeST, InferShapeForSimpleGraphShapeNEOriginShape) {
  REGISTER_SYMBOLIC_KERNEL(Unpack, nullptr);
  REGISTER_SYMBOLIC_KERNEL(Unsqueeze, UnsqueezeTest);
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto one = ge::Symbol(1);
  auto add = es::Add(data0, data1);
  auto mul = es::Mul(data0, add);
  auto relu = es::Relu(mul);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-2}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 4, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));
  std::vector<int64_t> dims_vec1 = {3, 4};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Relu_2");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
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
TEST_F(SymbolicShapeComputeST, InferShapeForGetConstInput) {
  auto data0 = builder_->CreateInput(0, "data0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto const_scalar_int32 = builder_->CreateScalar(static_cast<int32_t>(1));
  ASSERT_NE(const_scalar_int32.GetCTensorHolder(), nullptr);
  std::vector<int64_t> dims{0, 1, 2};
  int64_t dims_size = 3;
  auto const_int64_list = builder_->CreateConst(dims, std::vector<int64_t>{dims_size});
  ASSERT_NE(const_int64_list.GetCTensorHolder(), nullptr);
  auto reduce_sum = es::ReduceSum(data0, const_scalar_int32, true, true);
  NodeAdapter::GNode2Node(*reduce_sum.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  auto reduce_max = es::ReduceMax(reduce_sum, const_int64_list, true, true);
  NodeAdapter::GNode2Node(*reduce_max.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT64);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_max, 0), 0);
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
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {4, 2, 3, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto sum_node = cg->FindFirstNodeMatchType("ReduceSum");
  ASSERT_NE(sum_node, nullptr);
  auto op_desc = sum_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<std::string> expect_dim = {"s0", "1", "s2", "s3"};
  auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim.size(), expect_dim.size());
  for (size_t i = 0UL; i < result_dim.size(); i++) {
    EXPECT_EQ(std::string(result_dim[i].Serialize().get()), expect_dim[i]);
  }
  auto max_node = cg->FindFirstNodeMatchType("ReduceMax");
  ASSERT_NE(max_node, nullptr);
  op_desc = max_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  expect_dim = {"1", "1", "1", "s3"};
  result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim.size(), expect_dim.size());
  for (size_t i = 0UL; i < result_dim.size(); i++) {
    EXPECT_EQ(std::string(result_dim[i].Serialize().get()), expect_dim[i]);
  }
}

inline ge::graphStatus InferShapeDoNothing(gert::InferShapeContext *context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(ReduceSum).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Const).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Sub).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(ReduceMax).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(ReduceProd).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("keep_dims", false).PrivateAttr("noop_with_empty_axes", true);
IMPL_OP(Exp).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Div).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Cast).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(Data).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput();
IMPL_OP(BatchMatMulV2).InferShape(InferShapeDoNothing).InferOutDataTypeSameWithFirstInput()
    .PrivateAttr("adj_x1", false).PrivateAttr("adj_x2", false);

/**
*      Data0     Const(scalar)
*         \     /
*        ReduceSum    Const(list_int)
*              \       /
*              ReduceMax
*                  |
*              NetOutput
*/
TEST_F(SymbolicShapeComputeST, InferShapeForConst) {
  auto es_graph = std::unique_ptr<es::EsGraphBuilder>(new es::EsGraphBuilder("cpp_graph"));
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  data0.SetOriginSymbolShape({"s0", "s1", "s2", "s3"});
  auto const_scalar = es_graph->CreateScalar(1);
  auto const_list = es_graph->CreateVector(std::vector<int64_t>({0, 1}));
  auto reduce_sum = es::ReduceSum(data0, const_scalar, true, true);
  // 在infer symbol时，非decompose场景，图上节点的dtype在前面的infer shape流程处理过，ut时需要显示设置
  // decompose图中，infer symbol代码里面会调用infer dtype流程，因此要保证完成了decompose图中算子的infer dtype注册
  NodeAdapter::GNode2Node(*reduce_sum.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  auto reduce_max = es::ReduceMax(reduce_sum, const_list, true, true);
  NodeAdapter::GNode2Node(*reduce_max.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT64);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_max, 0), 0);
  auto graph = es_graph->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto const0_node = cg->FindNode("Const_0");
  ASSERT_NE(const0_node, nullptr);
  auto attr = const0_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  ASSERT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().IsScalar(), true);

  auto const1_node = cg->FindNode("Const_1");
  ASSERT_NE(const0_node, nullptr);
  attr = const1_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  ASSERT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDimNum(), 1);
  ASSERT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDim(0), ge::Symbol(2));
}

TEST_F(SymbolicShapeComputeST, InferShapeForVariable) {
  auto es_graph = std::unique_ptr<es::EsGraphBuilder>(new es::EsGraphBuilder("cpp_graph"));
  std::vector<int64_t> dims{4, 5, 6};
  auto var = ge::ComGraphMakeUnique<int64_t[]>(4 * 5 * 6);
  auto variable = es_graph->CreateVariable(0, "Variable0");
  auto relu = es::Relu(variable);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);
  auto graph = es_graph->BuildAndReset();
  ASSERT_NE(NodeAdapter::GNode2Node(*variable.GetProducer())->GetOpDesc(), nullptr);
  ASSERT_NE(NodeAdapter::GNode2Node(*variable.GetProducer())->GetOpDesc()->MutableInputDesc(0), nullptr);
  // variable类节点存在输入的tensor_desc，但是输入上没有symbol_desc_attr
  //  variable.GetProducer()->GetOpDesc()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto var_node = cg->FindNode("Variable0");
  ASSERT_NE(var_node, nullptr);
  var_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(dims));
  var_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape(dims));
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  var_node = cg->FindNode("Variable0");
  ASSERT_NE(var_node, nullptr);

  auto attr = var_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::cout << SymbolicInferUtil::VectorExpressionToStr(attr->symbolic_tensor.GetOriginSymbolShape().GetDims())
            << std::endl;
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
}  // namespace

TEST_F(SymbolicShapeComputeST, InferShapeForNoInferFunc) {
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
  ASSERT_NE(foo2_node, nullptr);
  attr = foo2_node->GetOpDescBarePtr()->GetOutputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

namespace {
auto const11 = OP_CFG("Const").InCnt(0).OutCnt(1).OutNames({"y"}).Build("Const");
auto const22 = OP_CFG("Constant").InCnt(0).OutCnt(1).OutNames({"y"}).Build("Constant");

}  // namespace

// 如果不是按照IR注册的方式造的node，后续造context时拿不到IR属性
// infer symbol shape时，会尝试从IR中恢复IR属性，这对const\constant节点非常重要，此处添加ut校验
TEST_F(SymbolicShapeComputeST, InferShapeForConstantWithRecoverIrAttr) {
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
TEST_F(SymbolicShapeComputeST, InferShapeForDecomposedGraph) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  auto pow = es::Pow(data0, data1);
  auto tanh = es::Tanh(data1);
  auto squaredD = es::SquaredDifference(pow, tanh);
  auto neg = es::Neg(squaredD);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(neg, 0), 0);
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

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {4, 2, 3, 3};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {2, 1, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto node = cg->FindNode("Neg_3");
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<std::string> expect_dim = {"s0", "s4", "s2", "s5"};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim.size(), expect_dim.size());
  for (size_t i = 0UL; i < result_dim.size(); i++) {
    EXPECT_EQ(std::string(result_dim[i].Serialize().get()), expect_dim[i]);
  }
}

/*
* 测试GeTensorDesc的拷贝函数，需要带着属性组深拷贝
* */
TEST_F(SymbolicShapeComputeST, TestGeTensorDesc_Copy_With_Attr) {
  auto data0 = builder_->CreateInput(0, "data0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);

  auto relu = es::Relu(data0);
  relu.SetOriginSymbolShape(std::vector<const char *>({"s2", "s1", "s0", "s0"}));
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(relu, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto relu1 = cg->FindFirstNodeMatchType("Relu");
  ASSERT_NE(relu1, nullptr);
  auto attr = relu1->GetOpDesc()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  attr->symbolic_tensor.SetSymbolShape({
      Symbol(1),
      Symbol(2),
      Symbol(3),
      Symbol(4),
  });

  auto attr1 = relu1->GetOpDesc()->GetInputDescPtr(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  std::cout << attr1 << std::endl;
  ASSERT_EQ(attr1->symbolic_tensor.GetOriginSymbolShape(), attr->symbolic_tensor.GetOriginSymbolShape());

  auto desc = relu1->GetOpDesc()->GetInputDesc(0);
  auto attr2 = desc.GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr2->symbolic_tensor.GetOriginSymbolShape(), attr->symbolic_tensor.GetOriginSymbolShape());

  auto all_desc = relu1->GetOpDesc()->GetAllInputsDesc();
  ASSERT_EQ(all_desc.size(), 1);
  auto attr3 = all_desc.at(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr3, nullptr);
  auto ret1 = attr3->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(attr3->symbolic_tensor.GetOriginSymbolShape().GetDims(),
            attr->symbolic_tensor.GetOriginSymbolShape().GetDims());

  auto attr4 = relu1->GetOpDesc()->GetAllInputsDescPtr().at(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr4, nullptr);
  ASSERT_EQ(attr4->symbolic_tensor.GetOriginSymbolShape(), attr->symbolic_tensor.GetOriginSymbolShape());
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │ data_1 │ ───────> │   mul    │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐  (0,0)   ┌──────────┐
// │ data_2 │ ───────> │  shape1  │
// └────────┘          └──────────┘
TEST_F(SymbolicShapeComputeST, test_shape_mul) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_1");
  auto shape = es::Shape(data0, 3);  // DT_INT32
  auto mul = es::Mul(data1, shape);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(mul, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data_1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {2, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {4, 3, 1};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto mul_node = cg->FindFirstNodeMatchType(MUL);
  ASSERT_NE(mul_node, nullptr);
  auto op_desc = mul_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<std::string> expect_dim = {"s3", "s4", "3"};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim.size(), expect_dim.size());
  for (size_t i = 0UL; i < result_dim.size(); i++) {
    EXPECT_EQ(std::string(result_dim[i].Serialize().get()), expect_dim[i]);
  }
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │ data_1 │ ───────> │ reshape1 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐  (0,0)   ┌──────────┐
// │ data_2 │ ───────> │  shape1  │
// └────────┘          └──────────┘
TEST_F(SymbolicShapeComputeST, test_shape_reshape) {
  auto data0 = builder_->CreateInput(0, "data0");
  auto data1 = builder_->CreateInput(1, "data1");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  const auto shape_0 = es::Shape(data1, 3);
  auto reshape0 = es::Reshape(data0, shape_0, 0, -1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reshape0, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT32);

  auto data_node1 = cg->FindNode("data1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({-1, 2, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({-1, 2, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT32);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({-1, 2, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, 2, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT32);

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

  std::vector<int64_t> dims_vec1 = {4, 2, 3};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
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
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "s2");

  auto data_symbol_attr1 = data_op_desc1->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 3);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s4");

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto reshape_node = cg->FindFirstNodeMatchType("Reshape");
  ASSERT_NE(reshape_node, nullptr);
  auto reshape_op_desc = reshape_node->GetOpDesc();
  ASSERT_NE(reshape_op_desc, nullptr);
  auto reshape_attr = reshape_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_attr, nullptr);
  auto reshape_symbol_shape = reshape_attr->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(reshape_symbol_shape.GetDimNum(), 3);
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(0).Serialize().get()), "s3");
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(1).Serialize().get()), "2");
  EXPECT_EQ(std::string(reshape_symbol_shape.GetDim(2).Serialize().get()), "s4");
  const std::vector<SymbolCheckInfo> guard_infos = shape_env_attr->GetAllSymbolAssertInfos();
  ASSERT_EQ(guard_infos.size(), 1);
  EXPECT_EQ(std::string(guard_infos[0].expr.Serialize().get()), "ExpectEq((2 * s3 * s4), (3 * s0 * s1 * s2))");
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
TEST_F(SymbolicShapeComputeST, simple_symbolic_kernel_compute) {
  auto data0 = builder_->CreateInput(0, "data_0");
  auto data1 = builder_->CreateInput(1, "data_1");
  auto data2 = builder_->CreateInput(2, "data_2");
  auto const1 = builder_->CreateScalar(static_cast<int32_t>(0));

  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "s2"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"(s3 * s4)", "s5"}));
  data1.SetOriginSymbolShape(std::vector<const char *>({"s0", "(s1 * s3 * s4)"}));

  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto gather_1 = es::GatherV2(shape1, const1, const1, 0, false, false);

  auto shape2 = es::Shape(data2, 3);  // DT_INT32
  auto gather_2 = es::GatherV2(shape2, const1, const1, 0, false, false);

  std::vector<EsTensorHolder> esb;
  esb.push_back(gather_1);
  esb.push_back(gather_2);
  auto pack1 = es::Pack(esb, 0, 2);

  auto reshape = es::Reshape(data2, pack1, 0, -1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reshape, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  // todo: 如果补充了gather、pack等的symbol kernel，继续完善该用例
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │   fill   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                        ∧
//                        │ (0,1)
//                        │
// ┌────────┐  (0,0)      |
// │const_1 │ ————————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_fill_with_const) {
  auto const0 = builder_->CreateScalar(static_cast<int64_t>(2));
  std::vector<int64_t> const_data = {2, 4, 2};
  auto const1 = builder_->CreateVector(const_data);
  auto fill = es::Fill(const1, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto fill_node = cg->FindFirstNodeMatchType(FILL);
  ASSERT_NE(fill_node, nullptr);
  auto op_desc = fill_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape());

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(4), Symbol(2)}));
  std::vector<Expression> expect_symbolic_value(16, Symbol(2));
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │   fill   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                        ∧
//                        │ (0,1)
//                        │
// ┌────────┐  (0,0)      |
// │const_1 │ ————————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_fill_output_size_exceed) {
  auto const0 = builder_->CreateScalar(static_cast<int64_t>(2));
  std::vector<int64_t> const_data = {2, 4, 1024};
  auto const1 = builder_->CreateVector(const_data);
  auto fill = es::Fill(const1, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto fill_node = cg->FindFirstNodeMatchType(FILL);
  ASSERT_NE(fill_node, nullptr);
  auto op_desc = fill_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape());

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(4), Symbol(1024)}));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │ fill     │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐  (0,0)   ┌──────────┐
// │ data_2 │ ───────> │  shape1  │
// └────────┘          └──────────┘
TEST_F(SymbolicShapeComputeST, test_fill_with_value_symbols) {
  std::vector<int64_t> const_data = {2, 4, 2};
  auto const0 = builder_->CreateVector(const_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto fill = es::Fill(const0, shape1);
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
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(4), Symbol(2)}));
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  std::vector<Expression> expect_symbolic_value(16, (s0 * s1));
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_fill_dim_not_const_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)"}));
  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"(s0 + s1)"}));
  auto dims = es::Shape(data0, 3);
  auto fill = es::Fill(dims, data1);
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
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  std::vector<Expression> expect_symbolic_value(1, (s0 * s1));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({s0 * s1}));
}

TEST_F(SymbolicShapeComputeST, test_fill_node_const_input_failed) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)"}));
  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"(s0 + s1)"}));
  auto fill = es::Fill(data0, data1);
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

TEST_F(SymbolicShapeComputeST, test_fill_node_const_input_over_limit) {
  std::vector<int64_t> const_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto const0 = builder_->CreateVector(const_data);
  auto data1 = builder_->CreateInput(0, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"(s0 + s1)"}));
  auto fill = es::Fill(const0, data1);
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
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1),
                              Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1),
                              Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1),
                              Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1), Symbol(1)}));
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │   tile   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                        ∧
//                        │ (0,1)
//                        │
// ┌────────┐  (0,0)      |
// │const_1 │ ————————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_tile_with_const) {
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim = {2, 3, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  std::vector<int64_t> const_data1 = {2, 3};
  auto const1 = builder_->CreateVector(const_data1);
  auto tile = es::Tile(const0, const1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(6), Symbol(6)}));
  std::vector<Expression> expect_symbolic_value = {
      Symbol(1),  Symbol(2), Symbol(1),  Symbol(2),  Symbol(1),  Symbol(2),  Symbol(3),  Symbol(4),  Symbol(3),
      Symbol(4),  Symbol(3), Symbol(4),  Symbol(5),  Symbol(6),  Symbol(5),  Symbol(6),  Symbol(5),  Symbol(6),
      Symbol(1),  Symbol(2), Symbol(1),  Symbol(2),  Symbol(1),  Symbol(2),  Symbol(3),  Symbol(4),  Symbol(3),
      Symbol(4),  Symbol(3), Symbol(4),  Symbol(5),  Symbol(6),  Symbol(5),  Symbol(6),  Symbol(5),  Symbol(6),
      Symbol(7),  Symbol(8), Symbol(7),  Symbol(8),  Symbol(7),  Symbol(8),  Symbol(9),  Symbol(10), Symbol(9),
      Symbol(10), Symbol(9), Symbol(10), Symbol(11), Symbol(12), Symbol(11), Symbol(12), Symbol(11), Symbol(12),
      Symbol(7),  Symbol(8), Symbol(7),  Symbol(8),  Symbol(7),  Symbol(8),  Symbol(9),  Symbol(10), Symbol(9),
      Symbol(10), Symbol(9), Symbol(10), Symbol(11), Symbol(12), Symbol(11), Symbol(12), Symbol(11), Symbol(12)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │   tile   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                        ∧
//                        │ (0,1)
//                        │
// ┌────────┐  (0,0)      |
// │const_1 │ ————————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_tile_check_output_size_failed) {
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim = {2, 3, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  std::vector<int64_t> const_data1 = {2, 1024};
  auto const1 = builder_->CreateVector(const_data1);
  auto tile = es::Tile(const0, const1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(6), Symbol(2048)}));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

// ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │ tile     │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐  (0,0)   ┌──────────┐
// │ data_0 │ ───────> │  shape1  │
// └────────┘          └──────────┘
TEST_F(SymbolicShapeComputeST, test_tile_with_value_symbols) {
  std::vector<int64_t> const_data = {2, 2, 3};
  auto const0 = builder_->CreateVector(const_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "2", "s2"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto tile = es::Tile(shape1, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(2), Symbol(9)}));
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<Expression> expect_symbolic_value = {
      s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2,
      s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2,
      s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2, s0 * s1, Symbol(2), s2};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_tile_without_symbols_value) {
  std::vector<int64_t> const_data = {2, 2, 3};
  auto const0 = builder_->CreateVector(const_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "2", "s2"}));
  auto tile = es::Tile(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

TEST_F(SymbolicShapeComputeST, test_tile_create_expression_failed_double_not_support) {
  auto const0 = builder_->CreateScalar(1.0f);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));
  auto tile = es::Tile(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  EXPECT_NE(attr, nullptr);
}

TEST_F(SymbolicShapeComputeST, test_tile_without_symbols_value_input2) {
  std::vector<int64_t> const_data = {2, 2, 3};
  auto const0 = builder_->CreateVector(const_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));
  auto tile = es::Tile(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

TEST_F(SymbolicShapeComputeST, test_tile_2nd_input_dims_invalid) {
  std::vector<int64_t> const_data = {2, 2, 3};
  auto const0 = builder_->CreateVector(const_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));
  auto tile = es::Tile(const0, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeComputeST, test_tile_kMultiplesInputIndex_symbols_value_invalid) {
  std::vector<int64_t> const_data = {2, 2, 3};
  auto const0 = builder_->CreateVector(const_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1"}));
  auto tile = es::Tile(const0, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeComputeST, test_tile_multiples_symbols_over_limit) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1"}));
  std::vector<int64_t> const_data = {2, 2, 3};
  auto const1 = builder_->CreateVector(const_data);
  auto tile = es::Tile(const1, data0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  auto data_node = cg->FindNode("data_0");
  auto data_attr = data_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  std::vector<ge::Expression> input_symbol_value({Symbol("s1"), Symbol("s1"), Symbol(1), Symbol(1), Symbol("s1"),
                                                  Symbol("s1"), Symbol(1), Symbol(1), Symbol("s1"), Symbol("s1"),
                                                  Symbol(1), Symbol(1)});
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(input_symbol_value);
  data_attr->symbolic_tensor.SetSymbolicValue(std::move(symbolic_value_unique));
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

TEST_F(SymbolicShapeComputeST, test_tile_GetMultiplesDims_failed) {
  std::vector<int64_t> const_data = {2, 2, 3, 3, 4, 4};
  std::vector<int64_t> shape_data = {1, 2, 3};
  auto const0 = builder_->CreateConst(const_data, shape_data);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "2", "s2"}));
  auto tile = es::Tile(data0, const0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto op_desc = tile_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3, 2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

// ┌────────┐  (0,0)
// │const_0 │ ————————————
// └────────┘             |
//                        |
//                        │
//                        |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │ Pack     │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                        ^
//                        |
//                        | (0,2)
// ┌────────┐  (0,0)   ┌──────────┐
// │const_2 │ ───────> │ fill     │
// └────────┘          └──────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐     (0,0)  ┌────────┐
// │ data_0 │ ───────—>  | shape  |
// └────────┘            └────────┘

TEST_F(SymbolicShapeComputeST, test_pack_with_symbols_value) {
  std::vector<EsTensorHolder> input_nodes;
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> const_dim0 = {2, 3};
  input_nodes.emplace_back(builder_->CreateConst(const_data0, const_dim0));
  std::vector<int32_t> const_data1 = {7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim1 = {2, 3};
  input_nodes.emplace_back(builder_->CreateConst(const_data1, const_dim1));
  std::vector<int64_t> const_data2 = {2, 3};
  auto const2 = builder_->CreateVector(const_data2);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto fill = es::Fill(const2, shape1);
  input_nodes.emplace_back(fill);
  auto pack = es::Pack(input_nodes, -2, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pack, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto pack_node = cg->FindFirstNodeMatchType(PACK);
  ASSERT_NE(pack_node, nullptr);
  auto op_desc = pack_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({2, 3}));
  auto fill_node = cg->FindFirstNodeMatchType(FILL);
  ASSERT_NE(fill_node, nullptr);
  auto fill_op_desc = fill_node->GetOpDesc();
  fill_op_desc->MutableInputDesc(0)->SetShape(GeShape({2}));
  fill_op_desc->MutableInputDesc(1)->SetShape(GeShape({1}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(3), Symbol(3)}));
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  std::vector<Expression> expect_symbolic_value = {Symbol(1),  Symbol(2),  Symbol(3),  Symbol(7), Symbol(8), Symbol(9),
                                                  s0 * s1,    s0 * s1,    s0 * s1,    Symbol(4), Symbol(5), Symbol(6),
                                                  Symbol(10), Symbol(11), Symbol(12), s0 * s1,   s0 * s1,   s0 * s1};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)
// │const_0 │ ————————————
// └────────┘             |
//                        |
//                        │
//                        |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │ Pack     │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,2)     |
// │const_2 │ ───────————
// └────────┘

TEST_F(SymbolicShapeComputeST, test_pack_with_const) {
  std::vector<es::EsTensorHolder> const_nodes;
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> const_dim0 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data0, const_dim0));
  std::vector<int32_t> const_data1 = {7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim1 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data1, const_dim1));
  std::vector<int32_t> const_data2 = {13, 14, 15, 16, 17, 18};
  std::vector<int64_t> const_dim2 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data2, const_dim2));
  auto pack = es::Pack(const_nodes, 2, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pack, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto pack_node = cg->FindFirstNodeMatchType(PACK);
  ASSERT_NE(pack_node, nullptr);
  auto op_desc = pack_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(3), Symbol(3)}));
  std::vector<Expression> expect_symbolic_value = {
      Symbol(1), Symbol(7),  Symbol(13), Symbol(2), Symbol(8),  Symbol(14), Symbol(3), Symbol(9),  Symbol(15),
      Symbol(4), Symbol(10), Symbol(16), Symbol(5), Symbol(11), Symbol(17), Symbol(6), Symbol(12), Symbol(18)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_pack_with_const_abormal_input) {
  std::vector<EsTensorHolder> const_nodes;
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> const_dim0 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data0, const_dim0));
  std::vector<int32_t> const_data1 = {7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim1 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data1, const_dim1));
  std::vector<int32_t> const_data2 = {13, 14, 15, 16, 17, 18, 19, 20};
  std::vector<int64_t> const_dim2 = {2, 4};
  const_nodes.emplace_back(builder_->CreateConst(const_data2, const_dim2));
  auto pack = es::Pack(const_nodes, 2, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pack, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto pack_node = cg->FindFirstNodeMatchType(PACK);
  ASSERT_NE(pack_node, nullptr);
  auto op_desc = pack_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({2, 4}));

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_pack_without_symbolic_value) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"2", "3"}));

  const_nodes.emplace_back(data0);
  std::vector<int32_t> const_data1 = {7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim1 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data1, const_dim1));
  std::vector<int32_t> const_data2 = {13, 14, 15, 16, 17, 18};
  std::vector<int64_t> const_dim2 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data2, const_dim2));
  auto pack = es::Pack(const_nodes, 2, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pack, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto pack_node = cg->FindFirstNodeMatchType(PACK);
  ASSERT_NE(pack_node, nullptr);
  auto op_desc = pack_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(3), Symbol(3)}));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

TEST_F(SymbolicShapeComputeST, test_pack_get_const_value_failed) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({}));

  const_nodes.emplace_back(data0);
  std::vector<int32_t> const_data1 = {7, 8, 9, 10, 11, 12};
  std::vector<int64_t> const_dim1 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data1, const_dim1));
  std::vector<int32_t> const_data2 = {13, 14, 15, 16, 17, 18};
  std::vector<int64_t> const_dim2 = {2, 3};
  const_nodes.emplace_back(builder_->CreateConst(const_data2, const_dim2));
  auto pack = es::Pack(const_nodes, 2, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pack, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto pack_node = cg->FindFirstNodeMatchType(PACK);
  ASSERT_NE(pack_node, nullptr);
  auto op_desc = pack_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({2, 3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

// ┌────────┐  (0,3)
// │const_3 │ ─────────────────────
// └────────┘                     |
//                                |
//                                |
//                                |
// ┌────────┐  (0,2)              |
// │const_2 │ ───────-----        |
// └────────┘             |       |
//                        |       |
//                        |       |
//                        |       |
// ┌────────┐  (0,1)   ┌─────────────────┐
// │const_1 │ ───────> │ stridedslice    │ (0,0)    ┌─────────────┐
// └────────┘          └─────────────────┘ ───────> │ Node_Output │
//                                                  └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐     (0,0)  ┌────────┐
// │ data_0 │ ───────—>  | shape  |
// └────────┘            └────────┘

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_symbols_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "(s0 + s1)", "s0", "3"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  std::vector<int64_t> const_data0 = {1};
  auto const_node0 = builder_->CreateVector(const_data0);
  std::vector<int64_t> const_data1 = {4};
  auto const_node1 = builder_->CreateVector(const_data1);
  std::vector<int64_t> const_data2 = {2};
  auto const_node2 = builder_->CreateVector(const_data2);
  auto strided_slice = es::StridedSlice(shape1, const_node0, const_node1, const_node2, 0, 0, 0, 0, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({1}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({1}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({1}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({}));
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  std::vector<Expression> expect_symbolic_value = {s0 + s1};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_AttrStartInput_symbols_value_invalid) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "(s0 + s1)", "s0", "3"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto const_node0 = builder_->CreateInput(1, "data_0");
  const_node0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));
  std::vector<int64_t> const_data1 = {4};
  auto const_node1 = builder_->CreateVector(const_data1);
  std::vector<int64_t> const_data2 = {2};
  auto const_node2 = builder_->CreateVector(const_data2);
  auto strided_slice = es::StridedSlice(shape1, const_node0, const_node1, const_node2, 0, 0, 0, 0, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({1}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({1}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({1}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}

TEST_F(SymbolicShapeComputeST, test_stridedsliced_with_symbols_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "(s0 + s1)", "s0", "3"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  std::vector<int64_t> const_data0 = {1};
  vector<int64_t> begin = {1};
  vector<int64_t> end = {4};
  vector<int64_t> strides = {2};
  auto strided_slice = es::StridedSliceD(shape1, begin, end, strides, 0, 0, 0, 0, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICED);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({}));
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  std::vector<Expression> expect_symbolic_value = {s0 + s1};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedsliced_without_symbols_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "3", "4"}));
  std::vector<int64_t> const_data0 = {1};
  vector<int64_t> begin = {1};
  vector<int64_t> end = {4};
  vector<int64_t> strides = {2};
  auto strided_slice = es::StridedSliceD(data0, begin, end, strides, 0, 0, 0, 0, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICED);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

// ┌────────┐  (0,3)
// │const_3 │ ─────────────────────
// └────────┘                     |
//                                |
//                                |
//                                |
// ┌────────┐  (0,2)              |
// │const_2 │ ───────-----        |
// └────────┘             |       |
//                        |       |
//                        |       |
//                        |       |
// ┌────────┐  (0,1)   ┌─────────────────┐
// │const_1 │ ───────> │ stridedslice    │ (0,0)    ┌─────────────┐
// └────────┘          └─────────────────┘ ───────> │ Node_Output │
//                                                  └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐     (0,0)  ┌────────┐
// │ data_0 │ ───────—>  | shape  |
// └────────┘            └────────┘
TEST_F(SymbolicShapeComputeST, test_stridedslice_normal) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 24; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {4, 3, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {2, 1, 0};
  std::vector<int64_t> const_dim1 = {3};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {4, 3, 1};
  std::vector<int64_t> const_dim2 = {3};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {1, 2, 1};
  std::vector<int64_t> const_dim3 = {3};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0, 0, 0, 0, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4, 3, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(1), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(14), Symbol(20)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_need_normalize) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 24; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {4, 3, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {-2, 1, 0};
  std::vector<int64_t> const_dim1 = {3};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {4, 3, -1};
  std::vector<int64_t> const_dim2 = {3};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {1, 2, 1};
  std::vector<int64_t> const_dim3 = {3};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0, 0, 0, 0, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4, 3, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(1), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(14), Symbol(20)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_ellipsis_mask) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 24; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {4, 3, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {3, 1, 0};
  std::vector<int64_t> const_dim1 = {3};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {4, 5, 5};
  std::vector<int64_t> const_dim2 = {3};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {1, 2, 3};
  std::vector<int64_t> const_dim3 = {3};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0, 0, 0b010, 0, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4, 3, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(1), Symbol(3), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(18), Symbol(20), Symbol(22)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_ellipsis_mask_with_begin_mask_and_end_mask) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 24; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {3, 2, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {1, 1, 1};
  std::vector<int64_t> const_dim1 = {3};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {0, 2, 2};
  std::vector<int64_t> const_dim2 = {3};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {-1, 1, 1};
  std::vector<int64_t> const_dim3 = {3};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0b001, 0b110, 0b010, 0, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3, 2, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(2), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(9), Symbol(11), Symbol(5), Symbol(7)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_ellipsis_mask_with_new_axis_mask) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 24; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {3, 2, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {5, 1, 0};
  std::vector<int64_t> const_dim1 = {3};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {1, 4, 2};
  std::vector<int64_t> const_dim2 = {3};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {-2, 1, 1};
  std::vector<int64_t> const_dim3 = {3};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0, 0, 0b010, 0b1110, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3, 2, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(1), Symbol(2), Symbol(2), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(8), Symbol(9), Symbol(10), Symbol(11)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_ellipsis_mask_with_new_axis_mask_shrink_axis_mask) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 24; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {3, 2, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {1, 1, 0};
  std::vector<int64_t> const_dim1 = {3};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {5, 4, 2};
  std::vector<int64_t> const_dim2 = {3};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {2, 1, 1};
  std::vector<int64_t> const_dim3 = {3};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0, 0, 0b100, 0b1110, 0b10111);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3, 2, 2}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({3}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(1), Symbol(2), Symbol(2)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(4), Symbol(5), Symbol(6), Symbol(7)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_new_axis_mask_when_input_dims_less_than_begin) {
  std::vector<int32_t> const_data0;
  const_data0.reserve(12);
  for (int32_t i = 0; i < 12; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {3, 2, 2};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {0, 0, 0, 0};
  std::vector<int64_t> const_dim1 = {4};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {1, 4, 2, -1};
  std::vector<int64_t> const_dim2 = {4};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {1, 1, 1, 1};
  std::vector<int64_t> const_dim3 = {4};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0, 0, 0, 0b10, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({3, 2, 4}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({4}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({4}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({4}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(1), Symbol(1), Symbol(2), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(0), Symbol(2)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_stridedslice_with_new_axis_mask_begin_mask_end_mask_when_input_dims_less_than_begin) {
  std::vector<int32_t> const_data0;
  const_data0.reserve(16);
  for (int32_t i = 0; i < 16; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {4, 4};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int32_t> const_data1 = {0, 0, 0, 0};
  std::vector<int64_t> const_dim1 = {4};
  auto const_node1 = builder_->CreateConst(const_data1, const_dim1);
  std::vector<int32_t> const_data2 = {0, 1, 0, 1};
  std::vector<int64_t> const_dim2 = {4};
  auto const_node2 = builder_->CreateConst(const_data2, const_dim2);
  std::vector<int32_t> const_data3 = {1, 1, 1, 1};
  std::vector<int64_t> const_dim3 = {4};
  auto const_node3 = builder_->CreateConst(const_data3, const_dim3);
  auto strided_slice = es::StridedSlice(const_node0, const_node1, const_node2, const_node3, 0b101, 0b101, 0, 0b1100, 0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4, 4}));
  op_desc->MutableInputDesc(1)->SetShape(GeShape({4}));
  op_desc->MutableInputDesc(2)->SetShape(GeShape({4}));
  op_desc->MutableInputDesc(3)->SetShape(GeShape({4}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(4), Symbol(1), Symbol(1), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(0), Symbol(4), Symbol(8), Symbol(12)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

//                     ┌──────────┐ (0,0)    ┌─────────────┐
//                     │ UnPack   │ ───────> │ Node_Output │
//                     └──────────┘          └─────────────┘
//                        ^
//                        |
//                        | (0,0)
// ┌────────┐  (0,0)   ┌──────────┐
// │const_0 │ ───────> │ tile     │
// └────────┘          └──────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐     (0,0)  ┌────────┐
// │ data_0 │ ───────—>  | shape  |
// └────────┘            └────────┘

TEST_F(SymbolicShapeComputeST, test_unpack_with_symbols_value) {
  std::vector<int64_t> const_data0 = {2, 2, 3, 2};
  auto const_node0 = builder_->CreateVector(const_data0);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 + s1)", "2", "s2"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto tile = es::Tile(shape1, const_node0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(tile, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto tile_op_desc = tile_node->GetOpDesc();
  tile_op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  tile_op_desc->MutableInputDesc(1)->SetShape(GeShape({4}));
  GeTensorDesc unpack_input_desc(GeShape({2, 2, 3, 6}), FORMAT_ND, DT_INT32);
  auto unpack_op_desc = std::make_shared<OpDesc>("unpack_0", UNPACK);
  unpack_op_desc->AppendIrAttrName("num");
  AttrUtils::SetInt(unpack_op_desc, "num", 3);
  unpack_op_desc->AppendIrAttrName("axis");
  AttrUtils::SetInt(unpack_op_desc, "axis", 2);
  unpack_op_desc->AddInputDesc(unpack_input_desc);
  GeTensorDesc unpack_output_desc(GeShape({2, 2, 6}), FORMAT_ND, DT_INT32);
  for (auto i = 0; i < 3; ++i) {
    unpack_op_desc->AddOutputDesc(unpack_output_desc);
  }
  auto unpack_node = cg->AddNode(unpack_op_desc);
  ASSERT_EQ(GraphUtils::AddEdge(tile_node->GetOutDataAnchor(0), unpack_node->GetInDataAnchor(0)), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  for (size_t i = 0UL; i < 3; i++) {
    auto attr = unpack_op_desc->GetOutputDesc(i).GetAttrsGroup<SymbolicDescAttr>();
    ASSERT_NE(attr, nullptr);
    std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(2), Symbol(6)};
    EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
    auto s0 = ge::Symbol("s0");
    auto s1 = ge::Symbol("s1");
    auto s2 = ge::Symbol("s2");
    std::vector<Expression> expect_symbolic_value = {(s0 + s1), ge::Symbol(2), s2, (s0 + s1), ge::Symbol(2), s2,
                                                    (s0 + s1), ge::Symbol(2), s2, (s0 + s1), ge::Symbol(2), s2,
                                                    (s0 + s1), ge::Symbol(2), s2, (s0 + s1), ge::Symbol(2), s2,
                                                    (s0 + s1), ge::Symbol(2), s2, (s0 + s1), ge::Symbol(2), s2};
    EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
    EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
  }
}

TEST_F(SymbolicShapeComputeST, test_unpack_get_const_value_failed) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 12; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  const_node0.SetOriginSymbolShape(std::vector<const char *>({}));
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  GeTensorDesc unpack_input_desc(GeShape({2, 2, 3}), FORMAT_ND, DT_INT32);
  auto unpack_op_desc = std::make_shared<OpDesc>("unpack_0", UNPACK);
  unpack_op_desc->AppendIrAttrName("num");
  AttrUtils::SetInt(unpack_op_desc, "num", 3);
  unpack_op_desc->AppendIrAttrName("axis");
  AttrUtils::SetInt(unpack_op_desc, "axis", -1);
  unpack_op_desc->AddInputDesc(unpack_input_desc);
  GeTensorDesc unpack_output_desc(GeShape({2, 2}), FORMAT_ND, DT_INT32);
  for (auto i = 0; i < 3; ++i) {
    unpack_op_desc->AddOutputDesc(unpack_output_desc);
  }
  auto unpack_node = cg->AddNode(unpack_op_desc);
  auto input_const_node = cg->FindFirstNodeMatchType(CONSTANT);
  ASSERT_NE(input_const_node, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(input_const_node->GetOutDataAnchor(0), unpack_node->GetInDataAnchor(0)), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_unpack_GetConstInputDims_failed) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "3"}));
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  GeTensorDesc unpack_input_desc(GeShape({2, 2, 3}), FORMAT_ND, DT_INT32);
  auto unpack_op_desc = std::make_shared<OpDesc>("unpack_0", UNPACK);
  unpack_op_desc->AddRequiredAttr("num");
  AttrUtils::SetInt(unpack_op_desc, "num", 3);
  unpack_op_desc->AppendIrAttrName("axis");
  AttrUtils::SetInt(unpack_op_desc, "axis", 0);
  unpack_op_desc->AddInputDesc(unpack_input_desc);
  GeTensorDesc unpack_output_desc(GeShape({2, 2}), FORMAT_ND, DT_INT32);
  for (auto i = 0; i < 3; ++i) {
    unpack_op_desc->AddOutputDesc(unpack_output_desc);
  }
  auto unpack_node = cg->AddNode(unpack_op_desc);
  auto input_const_node = cg->FindFirstNodeMatchType(DATA);
  ASSERT_NE(input_const_node, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(input_const_node->GetOutDataAnchor(0), unpack_node->GetInDataAnchor(0)), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │ Unpack   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
TEST_F(SymbolicShapeComputeST, test_unpack_with_const) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 12; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {2, 2, 3};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(const_node0, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  GeTensorDesc unpack_input_desc(GeShape({2, 2, 3}), FORMAT_ND, DT_INT32);
  auto unpack_op_desc = std::make_shared<OpDesc>("unpack_0", UNPACK);
  unpack_op_desc->AppendIrAttrName("num");
  AttrUtils::SetInt(unpack_op_desc, "num", 3);
  unpack_op_desc->AppendIrAttrName("axis");
  AttrUtils::SetInt(unpack_op_desc, "axis", -1);
  unpack_op_desc->AddInputDesc(unpack_input_desc);
  GeTensorDesc unpack_output_desc(GeShape({2, 2}), FORMAT_ND, DT_INT32);
  for (auto i = 0; i < 3; ++i) {
    unpack_op_desc->AddOutputDesc(unpack_output_desc);
  }
  auto unpack_node = cg->AddNode(unpack_op_desc);
  auto input_const_node = cg->FindFirstNodeMatchType(CONSTANT);
  ASSERT_NE(input_const_node, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(input_const_node->GetOutDataAnchor(0), unpack_node->GetInDataAnchor(0)), SUCCESS);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  for (size_t i = 0UL; i < 3; i++) {
    auto attr = unpack_op_desc->GetOutputDesc(i).GetAttrsGroup<SymbolicDescAttr>();
    ASSERT_NE(attr, nullptr);
    std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(2)};
    EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
    std::vector<Expression> expect_symbolic_value = {Symbol(i), Symbol(3 + i), Symbol(6 + i), Symbol(9 + i)};
    EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
    EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
  }
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ExpandDims│ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ———————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_expanddims_with_symbols_value) {
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

  auto pad_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(pad_node, nullptr);
  auto op_desc = pad_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol("s0"), Symbol(1), Symbol("s1"), Symbol("s2")}));
}

TEST_F(SymbolicShapeComputeST, test_expanddims_without_symbols_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(-1));
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);

  const auto expandDims = es::ExpandDims(data0, const0);
  NodeAdapter::GNode2Node(*expandDims.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expandDims, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto pad_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(pad_node, nullptr);
  auto op_desc = pad_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol("s0"), Symbol("s1"), Symbol("s2"), Symbol(1)}));
}

TEST_F(SymbolicShapeComputeST, test_expanddims_with_symbols_value3) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({}));
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(-1));
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);

  const auto expandDims = es::ExpandDims(data0, const0);
  NodeAdapter::GNode2Node(*expandDims.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expandDims, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto pad_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(pad_node, nullptr);
  auto op_desc = pad_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(1)}));
}

TEST_F(SymbolicShapeComputeST, test_expanddims_with_axis_symbols_error) {
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> const_dim0 = {2, 3};
  auto data0 = builder_->CreateConst(const_data0, const_dim0);
  auto const0 = builder_->CreateScalar(-1);
  ASSERT_NE(const0.GetCTensorHolder(), nullptr);
  const auto expandDims = es::ExpandDims(data0, const0);
  NodeAdapter::GNode2Node(*expandDims.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expandDims, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), UNSUPPORTED);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │  Shape   │ ───────> │ ExpandDIms  │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘          └─────────────┘
//                                               ^
//                                               |
// ┌────────┐  (0,1)                             |
// │const_0 │ ————————————————————————————————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_expanddims_host_compute1) {
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(0));

  std::vector<int64_t> const1_data = {1, 2, 3};
  auto const1 = builder_->CreateVector(const1_data);

  auto shape = es::Shape(const1, 3);
  auto expand_dims = es::ExpandDims(shape, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand_dims, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto expanddims_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(expanddims_node, nullptr);
  auto op_desc = expanddims_node->GetOpDesc();
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(1), Symbol(1)}));
  std::vector<Expression> expect_symbolic_value(1, Symbol(3));
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ const_1│ ───────> │ExpandDims│ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ———————————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_expanddims_host_compute2) {
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(0));

  auto const1 = builder_->CreateScalar(static_cast<int32_t>(1));

  auto expand_dims = es::ExpandDims(const1, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(expand_dims, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto expanddims_node = cg->FindFirstNodeMatchType(EXPANDDIMS);
  ASSERT_NE(expanddims_node, nullptr);
  auto op_desc = expanddims_node->GetOpDesc();
  op_desc->MutableInputDesc(1)->SetDataType(DT_INT32);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(1)}));
  std::vector<Expression> expect_symbolic_value(1, Symbol(1));
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

//                         ┌────────┐  (0,1)
//                         │const_1 │ ──────────────
//                         └────────┘               |
//                                                  |
// ┌────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────────────┐          ┌─────────────┐
// │const_0 │ ───────> │ tile     │ ───────> │ ReduceProd  │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘          └─────────────┘
//                       ∧
//                       │ (0,1)
//                       │
// ┌────────┐  (0,0)   ┌──────────┐
// │ data_0 │ ───────> │  shape1  │
// └────────┘          └──────────┘
TEST_F(SymbolicShapeComputeST, test_reduce_prod_with_value_symbols) {
  std::vector<int64_t> const_data0 = {2, 2, 3, 2};
  auto const_node0 = builder_->CreateVector(const_data0);
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 + s1)", "2", "s2"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto tile = es::Tile(shape1, const_node0);
  std::vector<int64_t> const_data1 = {2, -4};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(tile, const_node1, false, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto tile_node = cg->FindFirstNodeMatchType(TILE);
  ASSERT_NE(tile_node, nullptr);
  auto tile_op_desc = tile_node->GetOpDesc();
  tile_op_desc->MutableInputDesc(0)->SetShape(GeShape({3}));
  tile_op_desc->MutableInputDesc(1)->SetShape(GeShape({4}));

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3, 6}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = reduce_prod_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(6)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  std::vector<Expression> expect_symbolic_value = {
      ((s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1)).Simplify(),
      ge::Symbol(64),
      s2 * s2 * s2 * s2 * s2 * s2,
      ((s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1)).Simplify(),
      ge::Symbol(64),
      s2 * s2 * s2 * s2 * s2 * s2,
      ((s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1)).Simplify(),
      ge::Symbol(64),
      s2 * s2 * s2 * s2 * s2 * s2,
      ((s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1) * (s0 + s1)).Simplify(),
      ge::Symbol(64),
      s2 * s2 * s2 * s2 * s2 * s2};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

//           ┌────────┐  (0,1)
//           │const_1 │ ────────
//           └────────┘         |
//                              |
// ┌────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │ReduceProd│ ───────> │ NetOutput   │
// └────────┘          └──────────┘          └─────────────┘

TEST_F(SymbolicShapeComputeST, test_reduce_prod_with_const) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 12; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {2, 2, 3};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int64_t> const_data1 = {-1, 1};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(const_node0, const_node1, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = reduce_prod_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape = {Symbol(2), Symbol(1), Symbol(1)};
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(0), Symbol(332640)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_reduce_prod_without_symbolic_value) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"2", "3"}));
  std::vector<int64_t> const_data1 = {-1, 1};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(data0, const_node1, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_reduce_prod_AxisDims_invalid) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"2", "3"}));
  std::vector<int64_t> const_data1 = {-1, 1};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(const_node1, data0, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_reduce_prod_AxisInputIndex_invalid_symbolic_value) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"2"}));
  std::vector<int64_t> const_data1 = {-1, 1};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(const_node1, data0, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_reduce_prod_get_const_failed) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({}));
  std::vector<int64_t> const_data1 = {-1, 1};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(data0, const_node1, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_reduce_prod_ConstInputDims_failed) {
  std::vector<EsTensorHolder> const_nodes;
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s1", "s2"}));
  std::vector<int64_t> const_data1 = {-1, 1};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(data0, const_node1, true, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({2}));

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

//           ┌────────┐  (0,1)
//           │const_1 │ ────────
//           └────────┘         |
//                              |
// ┌────────┐  (0,1)   ┌──────────┐  (0,0)   ┌─────────────┐
// │const_0 │ ───────> │ReduceProd│ ───────> │ NetOutput   │
// └────────┘          └──────────┘          └─────────────┘

TEST_F(SymbolicShapeComputeST, test_reduce_prod_with_const_all_reduce) {
  std::vector<int32_t> const_data0;
  for (int32_t i = 0; i < 12; i++) {
    const_data0.emplace_back(i);
  }
  std::vector<int64_t> const_dim0 = {2, 2, 3};
  auto const_node0 = builder_->CreateConst(const_data0, const_dim0);
  std::vector<int64_t> const_data1 = {0, 1, 2};
  auto const_node1 = builder_->CreateVector(const_data1);
  auto reduce_pro = es::ReduceProd(const_node0, const_node1, false, true);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(reduce_pro, 0), 0);
  auto graph = builder_->BuildAndReset();

  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto reduce_prod_node = cg->FindFirstNodeMatchType(REDUCEPROD);
  ASSERT_NE(reduce_prod_node, nullptr);
  auto reduce_prod_op_desc = reduce_prod_node->GetOpDesc();
  reduce_prod_op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  reduce_prod_op_desc->MutableInputDesc(1)->SetShape(GeShape({3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = reduce_prod_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_symbolic_shape;
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_symbolic_shape);
  std::vector<Expression> expect_symbolic_value = {Symbol(0)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ Pad      │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ───────————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_pad_with_symbols_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {3, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  const auto pad = es::Pad(data0, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto pad_node = cg->FindFirstNodeMatchType(PAD);
  ASSERT_NE(pad_node, nullptr);
  auto op_desc = pad_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol("s0") + Symbol(3), Symbol("s1") + Symbol(3), Symbol("s2") + Symbol(2)}));
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │ data_0 │ ───────> │ Pad      │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
//                       ^
//                       |
// ┌────────┐  (0,1)     |
// │const_0 │ ───────————
// └────────┘
TEST_F(SymbolicShapeComputeST, test_pad_with_symbols_value_but_error_shape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2"}));

  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 4};  // // paddings.size != data0.dims * 2 校验报错
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  const auto pad = es::Pad(data0, const0);

  ASSERT_EQ(es::EsGraphBuilder::SetOutput(pad, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::PARAM_INVALID);
}

TEST_F(SymbolicShapeComputeST, test_host_compute_input_size_over_limit) {
  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2,
                                      1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {1, 30};
  auto data0 = builder_->CreateConst(const_data0, const_dim);
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

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_host_compute_type_not_valid) {
  std::vector<int32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3};
  auto data0 = builder_->CreateConst(const_data0, const_dim);
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto fill = es::Fill(data0, shape1);
  NodeAdapter::GNode2Node(*fill.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT16);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_host_compute_type_uint32) {
  std::vector<uint32_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3};
  auto data0 = builder_->CreateConst(const_data0, const_dim);
  auto shape1 = es::Shape(data0, DT_UINT32);
  auto fill = es::Fill(data0, shape1);
  NodeAdapter::GNode2Node(*fill.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_UINT32);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_host_compute_type_int64_t) {
  std::vector<int64_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3};
  auto data0 = builder_->CreateConst(const_data0, const_dim);
  auto shape1 = es::Shape(data0, DT_INT64);
  auto fill = es::Fill(data0, shape1);
  NodeAdapter::GNode2Node(*fill.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT64);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_host_compute_type_uint64_t) {
  std::vector<uint64_t> const_data0 = {1, 2, 2, 1, 1, 1};
  std::vector<int64_t> const_dim = {2, 3};
  auto data0 = builder_->CreateConst(const_data0, const_dim);
  auto shape1 = es::Shape(data0, DT_UINT64);
  auto fill = es::Fill(data0, shape1);
  NodeAdapter::GNode2Node(*fill.GetProducer())->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_UINT64);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(fill, 0), 0);
  const auto graph = builder_->BuildAndReset();
  const auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  const SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_0 │ ───────> │Unsqueeze │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
TEST_F(SymbolicShapeComputeST, test_unsqueeze_hostcompute) {
  auto const0 = builder_->CreateScalar(static_cast<int32_t>(1));
  std::vector<int64_t> axes = {0, 1};

  auto unsqueeze = es::Unsqueeze(const0, axes);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsqueeze, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto unsqueeze_node = cg->FindFirstNodeMatchType(UNSQUEEZE);
  ASSERT_NE(unsqueeze_node, nullptr);
  auto op_desc = unsqueeze_node->GetOpDesc();

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(1), Symbol(1)}));
  std::vector<Expression> expect_symbolic_value(1, Symbol(1));
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_unsqueeze_hostcompute_without_symbolic_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"1", "1"}));

  std::vector<int64_t> axes = {0, 1};

  auto unsqueeze = es::Unsqueeze(data0, axes);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(unsqueeze, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto unsqueeze_node = cg->FindFirstNodeMatchType(UNSQUEEZE);
  ASSERT_NE(unsqueeze_node, nullptr);
  auto op_desc = unsqueeze_node->GetOpDesc();

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(),
            gert::SymbolShape({Symbol(1), Symbol(1), Symbol(1), Symbol(1)}));
  std::vector<Expression> expect_symbolic_value(1, Symbol(1));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
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


TEST_F(SymbolicShapeComputeST, test_stridedslice_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");

  // begin
  std::vector<int64_t> data1_value = {0, 1, 2, 3};
  auto data1 = builder_->CreateVector(data1_value);
  // end
  std::vector<int64_t> data2_value = {5, 5, 5, 5};
  auto data2 = builder_->CreateVector(data2_value);
  // strides
  std::vector<int64_t> data3_value = {1, 1, 1, 1};
  auto data3 = builder_->CreateVector(data3_value);

  auto strided_slice =
      es::StridedSlice(data0, data1, data2, data3, static_cast<int64_t>(0b0010), static_cast<int64_t>(0b0010),
                    static_cast<int64_t>(0b0100), static_cast<int64_t>(0b1101), static_cast<int64_t>(0b0111));
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto op_desc = strided_slice_node->GetOpDesc();
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
TEST_F(SymbolicShapeComputeST, InferShapeForGraphWithNodeNotSupportSymbolInfer) {
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
                                                    squared_difference_node->GetInDataAnchor(1), foo_node),
            SUCCESS);

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
TEST_F(SymbolicShapeComputeST, test_batchmatmulv2_infershape) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"s0", "s1", "s2", "s3", "s4"}));

  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"s2", "s4", "s3"}));

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

void ConcatV2DHostComputeTestCommon(ge::es::EsGraphBuilder &builder_, const GeShape &data1_shape, const GeShape &data2_shape,
                                    const GeShape &data3_shape, int64_t concat_dim,
                                    const gert::SymbolShape &expect_out_shape,
                                    const std::vector<Expression> &expect_out_values) {
  std::vector<int64_t> const_data1 = {1, 1, 1, 1, 1, 1, 1, 1};
  auto data1 = builder_.CreateConst(const_data1, data1_shape.GetDims());
  std::vector<int64_t> const_data2 = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  auto data2 = builder_.CreateConst(const_data2, data2_shape.GetDims());
  std::vector<int64_t> const_data3 = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  auto data3 = builder_.CreateConst(const_data3, data3_shape.GetDims());
  std::vector<EsTensorHolder> inputs = {data1, data2, data3};

  auto concatv2d = es::ConcatV2D(inputs, concat_dim, inputs.size());
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", concat_dim);
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));

  op_desc->MutableInputDesc(0)->SetShape(data1_shape);
  op_desc->MutableInputDesc(1)->SetShape(data2_shape);
  op_desc->MutableInputDesc(2)->SetShape(data3_shape);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), expect_out_shape);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_out_values);
}

// ┌────────┐  (0,1)   ┌─────────────┐ (0,0)    ┌─────────────┐
// │ data_1 │ ───────> │ConcatV2D    │ ───────> │ Node_Output │
// └────────┘          └─────────────┘          └─────────────┘
TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute1) {
  // 拼接0轴
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  std::vector<Expression> expect_out_values = {
      c1, c1, c1, c1, c1, c1, c1, c1, c2, c2, c2, c2, c2, c2, c2, c2, c2, c2,
      c2, c2, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3, c3,
  };

  ConcatV2DHostComputeTestCommon(*builder_, GeShape({2, 2, 2}), GeShape({3, 2, 2}), GeShape({4, 2, 2}), 0,
                                gert::SymbolShape({Symbol(9), Symbol(2), Symbol(2)}), expect_out_values);
}
TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute2) {
  // 拼接1轴
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  std::vector<Expression> expect_out_values = {c1, c1, c1, c1, c2, c2, c2, c2, c2, c2, c3, c3, c3, c3, c3, c3, c3, c3,
                                              c1, c1, c1, c1, c2, c2, c2, c2, c2, c2, c3, c3, c3, c3, c3, c3, c3, c3};

  ConcatV2DHostComputeTestCommon(*builder_, GeShape({2, 2, 2}), GeShape({2, 3, 2}), GeShape({2, 4, 2}), 1,
                                gert::SymbolShape({Symbol(2), Symbol(9), Symbol(2)}), expect_out_values);
}
TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute3) {
  // 拼接3轴
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  std::vector<Expression> expect_out_values = {c1, c1, c2, c2, c2, c3, c3, c3, c3, c1, c1, c2, c2, c2, c3, c3, c3, c3,
                                              c1, c1, c2, c2, c2, c3, c3, c3, c3, c1, c1, c2, c2, c2, c3, c3, c3, c3};

  ConcatV2DHostComputeTestCommon(*builder_, GeShape({2, 2, 2}), GeShape({2, 2, 3}), GeShape({2, 2, 4}), 2,
                                gert::SymbolShape({Symbol(2), Symbol(2), Symbol(9)}), expect_out_values);
}
TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute4) {
  // 只有一个输入
  std::vector<int64_t> const_data2 = {1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  auto shape = GeShape({2, 2, 3});
  auto data2 = builder_->CreateConst(const_data2, shape.GetDims());

  std::vector<EsTensorHolder> inputs = {data2};

  auto concatv2d = es::ConcatV2D(inputs, 0, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(0));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(2), Symbol(2), Symbol(3)}));

  std::vector<Expression> expect_symbolic_value = {c1, c2, c3, c2, c2, c2, c2, c2, c2, c2, c2, c2};
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}
TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute5) {
  // 输入为scalar
  auto scalar1 = builder_->CreateScalar(static_cast<int64_t>(1));
  auto scalar2 = builder_->CreateScalar(static_cast<int64_t>(2));
  auto scalar3 = builder_->CreateScalar(static_cast<int64_t>(3));

  std::vector<EsTensorHolder> inputs = {scalar1, scalar2, scalar3};

  auto concatv2d = es::ConcatV2D(inputs, 0, 3);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(0));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(3)}));

  std::vector<Expression> expect_symbolic_value = {c1, c2, c3};
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute6_without_value) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"128", "32"}));
  auto data1 = builder_->CreateInput(1, "data_1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"128", "32"}));

  std::vector<EsTensorHolder> inputs = {data0, data1};

  auto concatv2d = es::ConcatV2D(inputs, 1, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(0));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(256), Symbol(32)}));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute7_concat_dim_not_valid) {
  // 只有一个输入
  std::vector<int64_t> const_data2 = {1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  auto shape = GeShape({2, 2, 3});
  auto data1 = builder_->CreateConst(const_data2, shape.GetDims());

  auto data2 = builder_->CreateConst(const_data2, shape.GetDims());

  std::vector<EsTensorHolder> inputs = {data1, data2};

  auto concatv2d = es::ConcatV2D(inputs, 1, 2);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(3));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::SUCCESS);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute8) {
  // 仅一维,在0轴拼接
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  auto c3 = Symbol(3);
  std::vector<int64_t> const_data1 = {1, 1};
  auto data1_shape = GeShape({2});
  auto data1 = builder_->CreateConst(const_data1, data1_shape.GetDims());
  std::vector<int64_t> const_data2 = {2};
  auto data2_shape = GeShape({1});
  auto data2 = builder_->CreateConst(const_data2, data2_shape.GetDims());
  std::vector<int64_t> const_data3 = {3};
  auto data3_shape = GeShape({1});
  auto data3 = builder_->CreateConst(const_data3, data3_shape.GetDims());
  std::vector<EsTensorHolder> inputs = {data1, data2, data3};
  int64_t concat_dim = 0L;
  auto concatv2d = es::ConcatV2D(inputs, concat_dim, inputs.size());
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", concat_dim);
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));

  op_desc->MutableInputDesc(0)->SetShape(data1_shape);
  op_desc->MutableInputDesc(1)->SetShape(data2_shape);
  op_desc->MutableInputDesc(2)->SetShape(data3_shape);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(4)}));

  std::vector<Expression> expect_symbolic_value = {c1, c1, c2, c3};
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_hostcompute9) {
  // 仅一维,在0轴拼接
  auto c1 = Symbol(1);
  auto c2 = Symbol(2);
  std::vector<int64_t> const_data1 = {1};
  auto data1_shape = GeShape({1});
  auto data1 = builder_->CreateConst(const_data1, data1_shape.GetDims());
  std::vector<int64_t> const_data2 = {2, 2, 2, 2};
  auto data2_shape = GeShape({4});
  auto data2 = builder_->CreateConst(const_data2, data2_shape.GetDims());
  std::vector<EsTensorHolder> inputs = {data1, data2};
  int64_t concat_dim = 0L;
  auto concatv2d = es::ConcatV2D(inputs, concat_dim, inputs.size());
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", concat_dim);
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));

  op_desc->MutableInputDesc(0)->SetShape(data1_shape);
  op_desc->MutableInputDesc(1)->SetShape(data2_shape);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(5)}));

  std::vector<Expression> expect_symbolic_value = {c1, c2, c2, c2, c2};
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_GetConstInputDims_failed_n1) {
  std::vector<int64_t> const_data2 = {2, 2, 2, 2};
  auto variable = builder_->CreateVariable(0, "Variable0");
  std::vector<EsTensorHolder> inputs = {variable};

  auto concatv2d = es::ConcatV2D(inputs, 1, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(1));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  auto var_node = cg->FindNode("Variable0");
  ASSERT_NE(var_node, nullptr);
  auto attr = var_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  attr->symbolic_tensor.SetSymbolShape({
      Symbol("s1"),
      Symbol("s2"),
      Symbol(3),
      Symbol(4),
  });
  std::vector<ge::Expression> input_symbol_value({Symbol("s1"), Symbol("s1"), Symbol(1), Symbol(1)});
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(input_symbol_value);
  attr->symbolic_tensor.SetSymbolicValue(std::move(symbolic_value_unique));
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::UNSUPPORTED);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_GetConstInputDims_failed_n2) {
  std::vector<int64_t> const_data2 = {2, 2, 2, 2};
  auto variable = builder_->CreateVariable(0, "Variable0");
  std::vector<EsTensorHolder> inputs = {variable, variable};

  auto concatv2d = es::ConcatV2D(inputs, 1, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(1));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));
  auto var_node = cg->FindNode("Variable0");
  ASSERT_NE(var_node, nullptr);
  auto attr = var_node->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  attr->symbolic_tensor.SetSymbolShape({
      Symbol("s1"),
      Symbol("s2"),
      Symbol(3),
      Symbol(4),
  });
  std::vector<ge::Expression> input_symbol_value({Symbol("s1"), Symbol("s1"), Symbol(1), Symbol(1)});
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression> >(input_symbol_value);
  attr->symbolic_tensor.SetSymbolicValue(std::move(symbolic_value_unique));
  SymbolicShapeInference ssi;
  ASSERT_NE(ssi.Infer(cg), ge::UNSUPPORTED);
}

TEST_F(SymbolicShapeComputeST, test_concatv2d_get_symbolic_input_failed) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({}));
  std::vector<EsTensorHolder> inputs = {data0};
  auto concatv2d = es::ConcatV2D(inputs, 1, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(concatv2d, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto concatv2d_node = cg->FindFirstNodeMatchType("ConcatV2D");
  ASSERT_NE(concatv2d_node, nullptr);
  auto op_desc = concatv2d_node->GetOpDesc();
  op_desc->AddRequiredAttr("concat_dim");
  AttrUtils::SetInt(op_desc, "concat_dim", static_cast<int64_t>(0));
  op_desc->AppendIrAttrName("N");
  AttrUtils::SetInt(op_desc, "N", static_cast<int64_t>(inputs.size()));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({2, 2, 3}));

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();

  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({}));
  EXPECT_EQ(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_0 │ ───────> │squeeze   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
TEST_F(SymbolicShapeComputeST, test_squeeze_hostcompute) {
  std::vector<int32_t> const_data0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int64_t> const_dim = {5, 1, 1, 2};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  std::vector<int64_t> axis = {1, 2};

  auto squeeze = es::Squeeze(const0, axis);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squeeze, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto squeeze_node = cg->FindFirstNodeMatchType(SQUEEZE);
  ASSERT_NE(squeeze_node, nullptr);
  auto op_desc = squeeze_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(5), Symbol(2)}));
  std::vector<Expression> expect_symbolic_value = {Symbol(1), Symbol(2), Symbol(3), Symbol(4), Symbol(5),
                                                  Symbol(6), Symbol(7), Symbol(8), Symbol(9), Symbol(10)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_0 │ ───────> │squeeze   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
TEST_F(SymbolicShapeComputeST, test_squeeze_all_hostcompute) {
  std::vector<int32_t> const_data0 = {1, 2, 3, 4};
  std::vector<int64_t> const_dim = {1, 1, 1, 4};
  auto const0 = builder_->CreateConst(const_data0, const_dim);

  std::vector<int64_t> axis = {};

  auto squeeze = es::Squeeze(const0, axis);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squeeze, 0), 0);

  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto squeeze_node = cg->FindFirstNodeMatchType(SQUEEZE);
  ASSERT_NE(squeeze_node, nullptr);
  auto op_desc = squeeze_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->symbolic_tensor.GetOriginSymbolShape(), gert::SymbolShape({Symbol(4)}));
  std::vector<Expression> expect_symbolic_value = {Symbol(1), Symbol(2), Symbol(3), Symbol(4)};
  EXPECT_NE(attr->symbolic_tensor.GetSymbolicValue(), nullptr);
  EXPECT_EQ(*attr->symbolic_tensor.GetSymbolicValue(), expect_symbolic_value);
}

// ┌────────┐  (0,0)   ┌──────────┐ (0,0)    ┌─────────────┐
// │data_0  │ ───────> │squeeze   │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
TEST_F(SymbolicShapeComputeST, test_squeeze_data_hostcompute) {
  auto data0 = builder_->CreateInput(0, "data_0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto data1 = builder_->CreateInput(1, "data_1");
  ASSERT_NE(data1.GetCTensorHolder(), nullptr);
  std::vector<EsTensorHolder> concat_input;
  concat_input.push_back(data0);
  concat_input.push_back(data1);
  const auto concat0 = es::ConcatV2D(concat_input, 1, 2);
  std::vector<int64_t> axis = {};
  auto squeeze = es::Squeeze(concat0, axis);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(squeeze, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({1, -1, -1, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, -1, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  auto data_node1 = cg->FindNode("data_1");
  ASSERT_NE(data_node1, nullptr);
  auto data_op_desc1 = data_node1->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc1, "index", 1);
  data_op_desc1->MutableInputDesc(0)->SetShape(GeShape({1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetOriginShape(GeShape({1, -1, -1, -1}));
  data_op_desc1->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc1->MutableOutputDesc(0)->SetShape(GeShape({1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetOriginShape(GeShape({1, -1, -1, -1}));
  data_op_desc1->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {1, 2, 3, 2};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {1, 4, 3, 2};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT64};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));
  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);

  auto squeeze_node = cg->FindFirstNodeMatchType(SQUEEZE);
  ASSERT_NE(squeeze_node, nullptr);
  auto op_desc = squeeze_node->GetOpDesc();

  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  std::vector<std::string> expect_dim = {"(s0 + s3)", "s4", "s5"};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim.size(), expect_dim.size());
  for (size_t i = 0UL; i < result_dim.size(); i++) {
    EXPECT_EQ(std::string(result_dim[i].Serialize().get()), expect_dim[i]);
  }
}
namespace {
void SelectTestCommon(ge::es::EsGraphBuilder &builder_, const vector<int32_t> &cond_value, const vector<int32_t> &const0,
                      const vector<int32_t> &const1, const vector<int64_t> &cond_dims, const vector<int64_t> &data_dims,
                      gert::SymbolTensor &symbolic_tensor, graphStatus status = SUCCESS) {
  auto cond = builder_.CreateConst(cond_value, cond_dims);
  auto data0 = builder_.CreateConst(const0, data_dims);
  auto data1 = builder_.CreateConst(const1, data_dims);

  auto select = es::Select(cond, data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(select, 0), 0);
  auto graph = builder_.BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto select_node = cg->FindFirstNodeMatchType(SELECT);
  ASSERT_NE(select_node, nullptr);
  auto op_desc = select_node->GetOpDesc();
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), status);
  if (status == SUCCESS) {
    symbolic_tensor = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>()->symbolic_tensor;
  }
}
}  // namespace

TEST_F(SymbolicShapeComputeST, test_select_1_n1) {
  std::vector<int32_t> cond_value = {0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {1};
  std::vector<int64_t> data_dims = {6};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  std::vector<Expression> expect_values = {c0, c0, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(6)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_1_n1xn2) {
  std::vector<int32_t> cond_value = {1};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {1};
  std::vector<int64_t> data_dims = {2, 3};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto s1 = Symbol(1);
  std::vector<Expression> expect_values = {s1, s1, s1, s1, s1, s1};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_1_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {1};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1,
                                          c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_n3_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 0, 1, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {4};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1, c0,
                                          c1, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_n2xn3_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {3, 4};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c0, c1, c0, c0, c0, c0, c0,
                                          c1, c1, c1, c1, c1, c0, c1, c0, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_n1x1x1_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {2, 1, 1};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1,
                                          c0, c0, c0, c0, c0, c0, c0, c0, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_1xn2x1_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 1, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {1, 3, 1};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0,
                                          c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_n1xn2x1_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 1, 0, 1, 0, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {2, 3, 1};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c1, c1, c1, c0, c0, c0, c0,
                                          c1, c1, c1, c1, c0, c0, c0, c0, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_n1x1xn3_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 1, 1, 1, 1, 0, 1, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {2, 1, 4};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1, c1,
                                          c1, c0, c1, c0, c1, c0, c1, c0, c1, c0, c1, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_n1xn2xn3_n1xn2xn3) {
  std::vector<int32_t> cond_value = {1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0};
  std::vector<int32_t> const0 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {2, 3, 4};
  std::vector<int64_t> data_dims = {2, 3, 4};

  gert::SymbolTensor symbolic_tensor;
  SelectTestCommon(*builder_, cond_value, const0, const1, cond_dims, data_dims, symbolic_tensor);

  auto c0 = Symbol(0);
  auto c1 = Symbol(1);
  std::vector<Expression> expect_values = {c1, c1, c1, c1, c1, c0, c1, c0, c0, c0, c0, c0,
                                          c1, c0, c1, c0, c1, c1, c1, c1, c0, c0, c0, c0};
  std::vector<Expression> expect_shape = {Symbol(2), Symbol(3), Symbol(4)};

  EXPECT_EQ(symbolic_tensor.GetOriginSymbolShape().GetDims(), expect_shape);
  EXPECT_EQ(*(symbolic_tensor.GetSymbolicValue()), expect_values);
}

TEST_F(SymbolicShapeComputeST, test_select_inputs_invalid) {
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  auto condition = builder_->CreateInput(0, "condition");
  auto data0 = builder_->CreateInput(1, "data0");
  auto data1 = builder_->CreateInput(2, "data1");
  condition.SetOriginSymbolShape(std::vector<const char *>({}));
  data0.SetOriginSymbolShape(std::vector<const char *>({}));
  data1.SetOriginSymbolShape(std::vector<const char *>({}));

  auto select = es::Select(condition, data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(select, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto select_node = cg->FindFirstNodeMatchType(SELECT);
  ASSERT_NE(select_node, nullptr);
  SymbolicShapeInference ssi;
  ssi.Infer(cg);
  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_WARN, "Select not support, inputs symbol value is empty."), -1);
  stub.GetSlogStub().Clear();
}

TEST_F(SymbolicShapeComputeST, test_select_shape_invalid) {
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  std::vector<int32_t> cond_value = {1};
  std::vector<int32_t> const0 = {1};
  std::vector<int32_t> const1 = {0, 0};

  std::vector<int64_t> cond_dims = {1};
  std::vector<int64_t> data0_dims = {1};
  std::vector<int64_t> data1_dims = {2};

  auto cond = builder_->CreateConst(cond_value, cond_dims);
  auto data0 = builder_->CreateConst(const0, data0_dims);
  auto data1 = builder_->CreateConst(const1, data1_dims);

  auto select = es::Select(cond, data0, data1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(select, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto select_node = cg->FindFirstNodeMatchType(SELECT);
  ASSERT_NE(select_node, nullptr);
  SymbolicShapeInference ssi;
  ssi.Infer(cg);
  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_WARN, "Select not support, check inputs shape failed,"), -1);
  stub.GetSlogStub().Clear();
}

TEST_F(SymbolicShapeComputeST, test_select_broadcast_failed1) {
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  std::vector<int32_t> cond_value = {1, 1, 1};
  std::vector<int32_t> const0 = {1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {3};
  std::vector<int64_t> data_dims = {2, 2};

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
  ssi.Infer(cg);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_WARN, "Select not support, broadcast failed."), -1);
  stub.GetSlogStub().Clear();
}

TEST_F(SymbolicShapeComputeST, test_select_broadcast_failed2) {
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevelInfo();

  std::vector<int32_t> cond_value = {1, 1, 1};
  std::vector<int32_t> const0 = {1, 1, 1, 1};
  std::vector<int32_t> const1 = {0, 0, 0, 0};

  std::vector<int64_t> cond_dims = {3, 1, 1};
  std::vector<int64_t> data_dims = {2, 2};

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
  ssi.Infer(cg);

  ASSERT_NE(stub.GetSlogStub().FindLog(DLOG_WARN, "Select not support, broadcast failed."), -1);
  stub.GetSlogStub().Clear();
}

// ┌────────┐  (0,0)
// │const_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [2, 2, 3, 2], indice: [2, 3], axis: 2 batch_dim = 0
TEST_F(SymbolicShapeComputeST, test_gather_const_hostcompute_batch_dim_0_success) {
  std::vector<int32_t> param_const_data;
  for (size_t i = 0UL; i < 24; i++) {
    param_const_data.emplace_back(i);
  }
  std::vector<int64_t> param_const_dim = {2, 2, 3, 2};
  auto param_const =
      builder_->CreateConst(param_const_data, param_const_dim);

  std::vector<int32_t> indice_const_data = {2, 2, 1, 0, 0, 1};
  std::vector<int64_t> indice_const_dim = {2, 3};
  auto indice_const =
      builder_->CreateConst(indice_const_data, indice_const_dim);

  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(2));

  auto gather_v2 = es::GatherV2(param_const, indice_const, axis_const, 0, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(2), Symbol(2), Symbol(3), Symbol(2)};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim, expect_dim);
  std::vector<Expression> expect_symbolic_value = {
      Symbol(4),  Symbol(5),  Symbol(4),  Symbol(5),  Symbol(2),  Symbol(3),  Symbol(0),  Symbol(1),
      Symbol(0),  Symbol(1),  Symbol(2),  Symbol(3),  Symbol(10), Symbol(11), Symbol(10), Symbol(11),
      Symbol(8),  Symbol(9),  Symbol(6),  Symbol(7),  Symbol(6),  Symbol(7),  Symbol(8),  Symbol(9),
      Symbol(16), Symbol(17), Symbol(16), Symbol(17), Symbol(14), Symbol(15), Symbol(12), Symbol(13),
      Symbol(12), Symbol(13), Symbol(14), Symbol(15), Symbol(22), Symbol(23), Symbol(22), Symbol(23),
      Symbol(20), Symbol(21), Symbol(18), Symbol(19), Symbol(18), Symbol(19), Symbol(20), Symbol(21)};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

// ┌────────┐  (0,0)
// │const_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [2, 2, 3, 2], indice: [2, 2, 3], axis: 2 batch_dim = 1
TEST_F(SymbolicShapeComputeST, test_gather_const_hostcompute_batch_dim_1_success) {
  std::vector<int32_t> param_const_data;
  for (size_t i = 0UL; i < 24; i++) {
    param_const_data.emplace_back(i);
  }
  std::vector<int64_t> param_const_dim = {2, 2, 3, 2};
  auto param_const =
      builder_->CreateConst(param_const_data, param_const_dim);

  std::vector<int32_t> indice_const_data = {2, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 2};
  std::vector<int64_t> indice_const_dim = {2, 2, 3};
  auto indice_const =
      builder_->CreateConst(indice_const_data, indice_const_dim);

  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(2));

  auto gather_v2 = es::GatherV2(param_const, indice_const, axis_const, -2, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(2), Symbol(2), Symbol(3), Symbol(2)};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim, expect_dim);
  std::vector<Expression> expect_symbolic_value = {
      Symbol(4),  Symbol(5),  Symbol(4),  Symbol(5),  Symbol(2),  Symbol(3),  Symbol(0),  Symbol(1),
      Symbol(0),  Symbol(1),  Symbol(2),  Symbol(3),  Symbol(10), Symbol(11), Symbol(10), Symbol(11),
      Symbol(8),  Symbol(9),  Symbol(6),  Symbol(7),  Symbol(6),  Symbol(7),  Symbol(8),  Symbol(9),
      Symbol(14), Symbol(15), Symbol(14), Symbol(15), Symbol(12), Symbol(13), Symbol(12), Symbol(13),
      Symbol(12), Symbol(13), Symbol(16), Symbol(17), Symbol(20), Symbol(21), Symbol(20), Symbol(21),
      Symbol(18), Symbol(19), Symbol(18), Symbol(19), Symbol(18), Symbol(19), Symbol(22), Symbol(23)};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

// ┌────────┐  (0,0)
// │const_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [2, 2, 3, 2], indice: [2, 2, 2], axis: 2 batch_dim = 2
TEST_F(SymbolicShapeComputeST, test_gather_const_hostcompute_batch_dim_2_success) {
  std::vector<int32_t> param_const_data;
  for (size_t i = 0UL; i < 24; i++) {
    param_const_data.emplace_back(i);
  }
  std::vector<int64_t> param_const_dim = {2, 2, 3, 2};
  auto param_const =
      builder_->CreateConst(param_const_data, param_const_dim);

  std::vector<int32_t> indice_const_data = {2, 2, 1, 0, 0, 1, 0, 2};
  std::vector<int64_t> indice_const_dim = {2, 2, 2};
  auto indice_const =
      builder_->CreateConst(indice_const_data, indice_const_dim);

  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(-2));

  auto gather_v2 = es::GatherV2(param_const, indice_const, axis_const, 2, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(2), Symbol(2), Symbol(2)};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim, expect_dim);
  std::vector<Expression> expect_symbolic_value = {
      Symbol(4),  Symbol(5),  Symbol(4),  Symbol(5),  Symbol(8),  Symbol(9),  Symbol(6),  Symbol(7),
      Symbol(12), Symbol(13), Symbol(14), Symbol(15), Symbol(18), Symbol(19), Symbol(22), Symbol(23)};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

// ┌────────┐  (0,0)
// | data_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [4], indice: [2, 3, 2], axis: 0 batch_dim = 0
TEST_F(SymbolicShapeComputeST, test_gather_symbol_value_hostcompute_batch_dim_0_success) {
  auto data0 = builder_->CreateInput(0, "data_0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto shape1 = es::Shape(data0, 3);  // DT_INT32

  std::vector<int32_t> indice_const_data = {3, 2, 1, 0, 0, 1, 3, 2, 3, 3, 1, 1};
  std::vector<int64_t> indice_const_dim = {2, 3, 2};
  auto indice_const =
      builder_->CreateConst(indice_const_data, indice_const_dim);

  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(0));

  auto gather_v2 = es::GatherV2(shape1, indice_const, axis_const, 0, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 2, 3, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(3), Symbol(2)};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim, expect_dim);
  std::vector<Expression> expect_symbolic_value = {Symbol("s2"), Symbol(3),    Symbol("s1"), Symbol("s0"),
                                                  Symbol("s0"), Symbol("s1"), Symbol("s2"), Symbol(3),
                                                  Symbol("s2"), Symbol("s2"), Symbol("s1"), Symbol("s1")};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

// ┌────────┐  (0,0)
// | data_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [], indice: [2, 3], axis: 0 batch_dim = 0

TEST_F(SymbolicShapeComputeST, test_gather_scalar_param_hostcompute_batch_dim_0_success) {
  auto param_const = builder_->CreateScalar(static_cast<int32_t>(2));

  std::vector<int32_t> indice_const_data = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> indice_const_dim = {2, 3};
  auto indice_const =
      builder_->CreateConst(indice_const_data, indice_const_dim);

  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(0));

  auto gather_v2 = es::GatherV2(param_const, indice_const, axis_const, 0, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(3)};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim, expect_dim);
  std::vector<Expression> expect_symbolic_value = {Symbol(2), Symbol(2), Symbol(2), Symbol(2), Symbol(2), Symbol(2)};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

// ┌────────┐  (0,0)
// | data_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [4], indice: [], axis: 0 batch_dim = 0

TEST_F(SymbolicShapeComputeST, test_gather_scalar_indices_hostcompute_batch_dim_0_success) {
  auto data0 = builder_->CreateInput(0, "data_0");
  ASSERT_NE(data0.GetCTensorHolder(), nullptr);
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  auto indice_const = builder_->CreateScalar(static_cast<int32_t>(1));
  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(0));

  auto gather_v2 = es::GatherV2(shape1, indice_const, axis_const, 0, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);
  auto data_node0 = cg->FindNode("data_0");
  ASSERT_NE(data_node0, nullptr);
  auto data_op_desc0 = data_node0->GetOpDesc();
  ge::AttrUtils::SetInt(data_op_desc0, "index", 0);
  data_op_desc0->MutableInputDesc(0)->SetShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  data_op_desc0->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetOriginShape(GeShape({-1, -1, 3, -1}));
  data_op_desc0->MutableOutputDesc(0)->SetDataType(DT_INT64);

  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {5, 2, 3, 4};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT64};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  SymbolicShapeInference ssi;
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);

  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim.empty(), true);
  std::vector<Expression> expect_symbolic_value = {Symbol("s1")};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

// ┌────────┐  (0,0)
// │const_0 │ ────────────
// └────────┘             |
// ┌────────┐  (0,1)   ┌──────────┐ (0,0)    ┌─────────────┐
// │const_1 │ ───────> │gather_v2 │ ───────> │ Node_Output │
// └────────┘          └──────────┘          └─────────────┘
// ┌────────┐  (0,2)        |
// │const_2 │ ──────────────
// └────────┘
// param: [2, 2, 3, 2], indice: [2, 2], axis: 2 batch_dim = 2
TEST_F(SymbolicShapeComputeST, test_gather_const_hostcompute_batch_dim_2_equal_to_indice_rank_success) {
  std::vector<int32_t> param_const_data;
  for (size_t i = 0UL; i < 24; i++) {
    param_const_data.emplace_back(i);
  }
  std::vector<int64_t> param_const_dim = {2, 2, 3, 2};
  auto param_const =
      builder_->CreateConst(param_const_data, param_const_dim);

  std::vector<int32_t> indice_const_data = {2, 2, 1, 0};
  std::vector<int64_t> indice_const_dim = {2, 2};
  auto indice_const =
      builder_->CreateConst(indice_const_data, indice_const_dim);

  auto axis_const = builder_->CreateScalar(static_cast<int32_t>(-2));

  auto gather_v2 = es::GatherV2(param_const, indice_const, axis_const, 2, false, false);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(gather_v2, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_NE(cg, nullptr);

  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto gather_v2_node = cg->FindFirstNodeMatchType(GATHERV2);
  ASSERT_NE(gather_v2_node, nullptr);
  auto op_desc = gather_v2_node->GetOpDesc();
  auto attr = op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr, nullptr);
  std::vector<Expression> expect_dim = {Symbol(2), Symbol(2), Symbol(2)};
  const auto result_dim = attr->symbolic_tensor.GetOriginSymbolShape().GetDims();
  ASSERT_EQ(result_dim, expect_dim);
  std::vector<Expression> expect_symbolic_value = {Symbol(4),  Symbol(5),  Symbol(10), Symbol(11),
                                                  Symbol(14), Symbol(15), Symbol(18), Symbol(19)};
  const auto result_value = attr->symbolic_tensor.GetSymbolicValue();
  EXPECT_NE(result_value, nullptr);
  EXPECT_EQ(*result_value, expect_symbolic_value);
}

TEST_F(SymbolicShapeComputeST, test_symbols_value_format_invalid) {
  auto data0 = builder_->CreateInput(0, "data_0");
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "(s0 + s1)", "s0", "3"}));
  auto shape1 = es::Shape(data0, 3);  // DT_INT32
  data0.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "2", "s2"}));
  auto tile = es::Tile(shape1, shape1);
  auto const_node0 = builder_->CreateInput(1, "const_node0");
  const_node0.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));

  std::vector<int64_t> const_data1 = {4};
  auto const_node1 = builder_->CreateVector(const_data1);
  const_node1.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));
  std::vector<int64_t> const_data2 = {2};
  auto const_node2 = builder_->CreateVector(const_data2);
  const_node2.SetOriginSymbolShape(std::vector<const char *>({"1", "2", "2"}));
  auto data1 = builder_->CreateInput(2, "data1");
  data1.SetOriginSymbolShape(std::vector<const char *>({"(s0 * s1)", "(s0 + s1)", "s0", "3"}));
  auto shape2 = es::Shape(data1, 3);  // DT_INT32
  std::vector<int64_t> const_data3 = {-1, 1};
  auto const_node3 = builder_->CreateVector(const_data3);
  auto reduce_pro = es::ReduceProd(const_node3, shape2, true, true);
  auto strided_slice = es::StridedSlice(tile, shape1, const_node1, reduce_pro, 0, 0, 0, 0, 1);
  ASSERT_EQ(es::EsGraphBuilder::SetOutput(strided_slice, 0), 0);
  auto graph = builder_->BuildAndReset();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType(REDUCEPROD)->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT32);
  ASSERT_NE(cg, nullptr);
  SymbolicShapeInference ssi;
  ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto strided_slice_node = cg->FindFirstNodeMatchType(STRIDEDSLICE);
  ASSERT_NE(strided_slice_node, nullptr);
  auto strided_slice_op_desc = strided_slice_node->GetOpDesc();
  auto attr = strided_slice_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_EQ(attr, nullptr);
}
}  // namespace ge
