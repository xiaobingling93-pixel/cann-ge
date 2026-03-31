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

#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/debug/ge_op_types.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/anchor_utils.h"
#include "test_std_structs.h"
#include "graph/operator_reg.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "faker/space_registry_faker.h"

namespace ge {
class UtestOpDescUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


namespace {
///     Data    const1
///        \  /
///        addn
///
ComputeGraphPtr BuildGraph1() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto addn = builder.AddNode("addn", "AddN", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor0});

  builder.AddDataEdge(data, 0, addn, 0);
  builder.AddDataEdge(const1, 0, addn, 1);
  return builder.GetGraph();
}
///   (p_const)addn    const1
///          /     \   /
///        cast     mul
///
ComputeGraphPtr BuildGraph2() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto addn = builder.AddNode("addn", "AddN", 0, 2);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto cast = builder.AddNode("cast", "Cast", 1, 1);
  auto mul = builder.AddNode("mul", "Mul", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  AttrUtils::SetBool(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  AttrUtils::SetListInt(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0,1});
  AttrUtils::SetListTensor(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor0, tensor0});
  OpDescUtils::SetWeights(const1, {tensor0});

  builder.AddDataEdge(addn, 0, cast, 0);
  builder.AddDataEdge(addn, 1, mul, 0);
  builder.AddDataEdge(const1, 0, mul, 1);
  return builder.GetGraph();
}
///   (p_const)addn    const1
///          /     \   /
///        enter     mul
///         |
///       cast
ComputeGraphPtr BuildGraph3() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto addn = builder.AddNode("addn", "AddN", 0, 2);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto enter = builder.AddNode("enter", "Enter", 1, 1);
  auto cast = builder.AddNode("cast", "Cast", 1, 1);
  auto mul = builder.AddNode("mul", "Mul", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  AttrUtils::SetBool(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  AttrUtils::SetListInt(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0,1});
  AttrUtils::SetListTensor(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor0, tensor0});
  OpDescUtils::SetWeights(const1, {tensor0});

  AttrUtils::SetBool(enter->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, true);

  builder.AddDataEdge(addn, 0, enter, 0);
  builder.AddDataEdge(addn, 1, mul, 0);
  builder.AddDataEdge(const1, 0, mul, 1);
  builder.AddDataEdge(enter, 0, cast, 0);
  return builder.GetGraph();
}

///     x0   a   bias  b
///      \    \   /  /
///        DynamicOpUt
///
ComputeGraphPtr BuildGraph4(size_t dynamic_input_num, bool has_optional_input) {
  size_t optional_input_num = has_optional_input ? 1u : 0U;
  ut::GraphBuilder builder = ut::GraphBuilder("graph");

  auto data2 = builder.AddNode("a", "Data", 1, 1);
  auto data4 = builder.AddNode("b", "Data", 1, 1);
  auto dynamic_op_ut = builder.AddNode("dynamic_op_ut", "DynamicOpUt",
                                       2 + dynamic_input_num + optional_input_num, 1);

  size_t dst_index = 0;
  // dynamic input
  for (size_t i = 0U; i < dynamic_input_num; ++i) {
    auto data1 = builder.AddNode("x", "Data", 1, 1);
    builder.AddDataEdge(data1, 0, dynamic_op_ut, dst_index++);
  }

  // required input
  builder.AddDataEdge(data2, 0, dynamic_op_ut, dst_index++);

  // optional input
  for (size_t i = 0U; i < optional_input_num; ++i) {
    auto data3 = builder.AddNode("bias", "Data", 1, 1);
    builder.AddDataEdge(data3, 0, dynamic_op_ut, dst_index++);
  }

  // required input
  builder.AddDataEdge(data4, 0, dynamic_op_ut, dst_index++);

  auto graph = builder.GetGraph();
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputDynamic);
  op_desc->AppendIrInput("a", kIrInputRequired);
  op_desc->AppendIrInput("bias", kIrInputOptional);
  op_desc->AppendIrInput("b", kIrInputRequired);
  return graph;
}
///     Data
///       |
///       | ctrl_edge
///      noop
///
ComputeGraphPtr BuildGraph5() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto noop = builder.AddNode("noop", "NoOp", 1, 0);

  builder.AddControlEdge(data, noop);
  return builder.GetGraph();
}
///        a       b
///          \   /
///        DynamicOpUt
///
ComputeGraphPtr BuildGraph6(size_t dynamic_output_num) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data0 = builder.AddNode("a", "Data", 1, 1);
  auto data1 = builder.AddNode("b", "Data", 1, 1);
  auto dynamic_op_ut = builder.AddNode("dynamic_op_ut", "DynamicOpUt",
                                       2, 1 + dynamic_output_num + 1);

  size_t dst_index = 0;
  builder.AddDataEdge(data0, 0, dynamic_op_ut, dst_index++);
  builder.AddDataEdge(data1, 0, dynamic_op_ut, dst_index++);

  auto graph = builder.GetGraph();
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();
  op_desc->AppendIrInput("a", kIrInputRequired);
  op_desc->AppendIrInput("b", kIrInputRequired);
  op_desc->AppendIrOutput("x", kIrOutputRequired);
  op_desc->AppendIrOutput("y", kIrOutputDynamic);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  return graph;
}
}
TEST_F(UtestOpDescUtils, SetWeight) {
  auto graph = BuildGraph1();

  auto addn_node = graph->FindNode("addn");
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  map<int, ge::GeTensorPtr> weight0;
  weight0[-1] = tensor;
  auto ret = ge::OpDescUtils::SetWeights(*addn_node, weight0);
  EXPECT_NE(ret, 0);

  map<int, ge::GeTensorPtr> weight1;
  weight1[1] = tensor;
  ret = ge::OpDescUtils::SetWeights(*addn_node, weight1);
  EXPECT_EQ(ret, 0);
  auto const_node = graph->FindNode("const1");
  auto const_tensor = OpDescUtils::MutableWeights(const_node);
  EXPECT_EQ(const_tensor[0]->MutableData().size(), 3);
  auto in_nodes = addn_node->GetInAllNodes();
  EXPECT_EQ(in_nodes.size(), 2);

  map<int, ge::GeTensorPtr> weight2;
  weight2[2] = tensor;
  ret = ge::OpDescUtils::SetWeights(*addn_node, weight2);
  EXPECT_EQ(ret, 0);
  auto in_nodes1 = addn_node->GetInAllNodes();
  EXPECT_EQ(in_nodes1.size(), 3);
}

TEST_F(UtestOpDescUtils, GetRealConstInputNodeAndAnchor) {
  auto graph = BuildGraph1();
  auto add_node = graph->FindNode("addn");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*add_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 1);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "const1");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 0);
}
TEST_F(UtestOpDescUtils, GetMixConstInputNodeAndAnchor) {
  auto graph = BuildGraph2();
  auto mul_node = graph->FindNode("mul");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*mul_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 2);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 1);
  EXPECT_EQ(nodes_2_out_anchor[1].first->GetName(), "const1");
  EXPECT_EQ(nodes_2_out_anchor[1].second->GetIdx(), 0);
}
TEST_F(UtestOpDescUtils, GetInputDataByIndexForMixInputConst) {
  auto graph = BuildGraph2();
  auto mul_node = graph->FindNode("mul");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*mul_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 2);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 1);
  EXPECT_EQ(nodes_2_out_anchor[1].first->GetName(), "const1");
  EXPECT_EQ(nodes_2_out_anchor[1].second->GetIdx(), 0);

  auto weights = OpDescUtils::GetWeightsFromNodes(nodes_2_out_anchor);
  EXPECT_EQ(weights.size(), 2);
  EXPECT_EQ(weights[0]->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(weights[1]->GetTensorDesc().GetDataType(), DT_INT32);
}
TEST_F(UtestOpDescUtils, GetPotentailWeightByIndexAccrossEnter) {
  auto graph = BuildGraph3();
  auto cast_node = graph->FindNode("cast");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*cast_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 1);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 0);

  auto weights = OpDescUtils::GetWeightsFromNodes(nodes_2_out_anchor);
  EXPECT_EQ(weights.size(), 1);
  EXPECT_EQ(weights[0]->GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(UtestOpDescUtils, GetInputConstDataByIndex_01) {
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 23;
  data_buf[10] = 32;
  auto ge_tensor = std::make_shared<GeTensor>();
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", ge_tensor);
  auto case_node = builder.AddNode("Case", "Case", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(const_node, 0, case_node, 0);
  builder.AddDataEdge(case_node, 0, netoutput, 0);
  auto parent_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph_graph");
  auto sub_data = sub_builder.AddNode("sub_data", "Data", 0, 1);
  auto sub_const = sub_builder.AddNode("sub_const", "Const", 0, 1);
  AttrUtils::SetTensor(sub_const->GetOpDesc(), "value", ge_tensor);
  auto add = sub_builder.AddNode("Add", "Add", 2, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_netoutput", "NetOutput", 1, 0);
  sub_builder.AddDataEdge(sub_data, 0, add, 0);
  sub_builder.AddDataEdge(sub_const, 0, add, 1);
  sub_builder.AddDataEdge(add, 0, sub_netoutput, 0);

  auto subgraph = sub_builder.GetGraph();
  subgraph->SetParentNode(case_node);
  subgraph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(subgraph->GetName(), subgraph);
  AttrUtils::SetInt(sub_data->GetOpDesc(), "_parent_node_index", 0);

  auto op_desc = add->GetOpDesc();
  op_desc->impl_->input_name_idx_["sub_data"] = 0;
  op_desc->impl_->input_name_idx_["sub_const"] = 1;
  auto op = OpDescUtils::CreateOperatorFromNode(add);
  RuntimeInferenceContext runtime_ctx;
  // define callback
  OpDescUtils::GetConstInputOnRuntimeFun func_get_input_const =
      [&runtime_ctx](const ConstNodePtr &node, const size_t index, ge::GeTensorPtr &dst_tensor) {
        // from runtime context
        const auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(index));
        const auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
        auto peer_node = out_data_anchor->GetOwnerNode();
        GeTensorPtr tensor_value = nullptr;
        if (runtime_ctx.GetTensor(peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx(), tensor_value) ==
            GRAPH_SUCCESS) {
          dst_tensor = tensor_value;
          return GRAPH_SUCCESS;
        }
        return ge::GRAPH_SUCCESS;
      };
  OpDescUtils::SetCallbackGetConstInputFuncToOperator(op, func_get_input_const);
  GeTensorDesc desc;
  GeTensorPtr tensor = std::make_shared<GeTensor>(desc);
  tensor->SetData(data_buf, 4096);

  int64_t node_id = 1;
  int output_id = 0;
  runtime_ctx.SetTensor(node_id, output_id, std::move(tensor));
  ConstGeTensorBarePtr ge_tensor_res = nullptr;
  ge_tensor_res = OpDescUtils::GetInputConstData(op, 1);

  ASSERT_TRUE(ge_tensor_res != nullptr);
  const TensorData tmp(ge_tensor_res->GetData());
  const uint8_t* res_buf = tmp.GetData();
  ASSERT_EQ(res_buf[0], 23);
  ASSERT_EQ(res_buf[10], 32);
}

TEST_F(UtestOpDescUtils, GetInputConstDataByIndex_02) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto enter = builder.AddNode("Enter", "Enter", 1, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data2, 0, enter, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(enter, 0, transdata, 1);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 23;
  data_buf[10] = 32;
  ge_tensor->SetData(data_buf, 4096);

  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto tensor_desc = op_desc->MutableInputDesc(0);
  AttrUtils::SetTensor(tensor_desc, "_value", ge_tensor);

  auto op = OpDescUtils::CreateOperatorFromNode(transdata);
  ConstGeTensorBarePtr ge_tensor_res = nullptr;
  ConstGeTensorBarePtr ge_tensor_res2 = nullptr;
  ge_tensor_res = OpDescUtils::GetInputConstData(op, 0);
  ge_tensor_res2 = OpDescUtils::GetInputConstData(op, 1);
  ASSERT_TRUE(ge_tensor_res != nullptr);
  ASSERT_TRUE(ge_tensor_res2 == nullptr);
  const TensorData tmp(ge_tensor_res->GetData());
  const uint8_t* res_buf = tmp.GetData();
  ASSERT_EQ(res_buf[0], 23);
  ASSERT_EQ(res_buf[10], 32);
}

// for partiton graph get const
TEST_F(UtestOpDescUtils, GetInputConstDataByIndex_03) {
  ut::GraphBuilder builder = ut::GraphBuilder("partiton_graph0");
  auto pld = builder.AddNode(PLACEHOLDER, PLACEHOLDER, 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode(NETOUTPUT, NETOUTPUT, 1, 0);
  builder.AddDataEdge(pld, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto op_desc = transdata->GetOpDesc();

  ut::GraphBuilder builder1 = ut::GraphBuilder("partiton_graph1");
  auto const_node = builder1.AddNode(CONSTANT, CONSTANT, 0, 1);
  auto end = builder1.AddNode(END, END, 1, 0);
  builder.AddDataEdge(const_node, 0, end, 0);
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096U] = {0};
  data_buf[0] = 23U;
  data_buf[10] = 32U;
  ge_tensor->SetData(data_buf, 4096U);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ATTR_NAME_WEIGHTS, ge_tensor);

  pld->GetOpDesc()->SetExtAttr("parentNode", const_node);
  auto op = OpDescUtils::CreateOperatorFromNode(transdata);
  ConstGeTensorBarePtr ge_tensor_res = nullptr;
  // case 0
  ge_tensor_res = OpDescUtils::GetInputConstData(op, 0U);
  ASSERT_TRUE(ge_tensor_res != nullptr);
  const TensorData tmp(ge_tensor_res->GetData());
  const uint8_t *res_buf = tmp.GetData();
  ASSERT_EQ(res_buf[0], 23U);
  ASSERT_EQ(res_buf[10], 32U);

  // case 1
  op_desc->impl_->input_name_idx_[PLACEHOLDER] = 0U;
  Tensor tensor;
  ASSERT_EQ(op.GetInputConstData(PLACEHOLDER, tensor), GRAPH_SUCCESS);
  const uint8_t *buf = tensor.GetData();
  ASSERT_EQ(buf[0], 23U);
  ASSERT_EQ(buf[10], 32U);
}

TEST_F(UtestOpDescUtils, DefaultInferFormat) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape());
  tensor_desc->SetFormat(FORMAT_ND);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  EXPECT_EQ(op_desc->DefaultInferFormat(), 0);
  auto input_desc = op_desc->MutableInputDesc(0);
  EXPECT_EQ(input_desc->GetFormat(), FORMAT_ND);
  auto output_desc = op_desc->MutableOutputDesc(0);
  EXPECT_EQ(output_desc->GetFormat(), FORMAT_ND);
}


TEST_F(UtestOpDescUtils, OpDescBuilder) {
  OpDescBuilder builder("name", "type");
  builder.AddDynamicInput("AddDy", 1);
  EXPECT_NE(&builder, nullptr);
  const GeTensorDesc ten = GeTensorDesc(GeShape());
  builder.AddDynamicInput(std::string("AddDy2"), 2, ten);
  EXPECT_NE(&builder, nullptr);
  builder.AddDynamicOutput("AddDyOut", 3);
  EXPECT_NE(&builder, nullptr);
  builder.AddDynamicOutput(std::string("AddDyOut2"), 4, ten);
  EXPECT_NE(&builder, nullptr);
}

TEST_F(UtestOpDescUtils, OpDescUtils) {
  OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
  EXPECT_EQ(OpDescUtils::SetSubgraphInstanceName("subgraph_name", "subgraph_instance_name", odp), GRAPH_PARAM_INVALID);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  GeTensorPtr tp = std::make_shared<GeTensor>();
  OpDescPtr odp1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(OpDescUtils::MutableWeights(odp1), nullptr);
  EXPECT_EQ(OpDescUtils::ClearWeights(data_node), GRAPH_SUCCESS);
  NodePtr np = std::make_shared<Node>();
  EXPECT_EQ(OpDescUtils::ClearWeights(np), GRAPH_PARAM_INVALID);
  EXPECT_EQ(OpDescUtils::ClearInputDesc(data_node), true);
  odp->AddInputDesc(GeTensorDesc());
  EXPECT_EQ(OpDescUtils::GetWeights(data_node).size(), 0);
  EXPECT_EQ(OpDescUtils::GetWeights(nullptr).size(), 0);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*data_node).size(), 0);
  EXPECT_EQ(OpDescUtils::SetWeights(*odp, nullptr), GRAPH_FAILED);
  EXPECT_EQ(OpDescUtils::ClearInputDesc(odp, 0), true);
  EXPECT_EQ(OpDescUtils::ClearInputDesc(odp, 1), false);
  EXPECT_EQ(odp->impl_->inputs_desc_.size(), 0);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(odp), false);
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(data_node), true);
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(odp, 0), false);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(*odp), false);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(*data_node, 1), false);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(data_node, 1), false);
}

TEST_F(UtestOpDescUtils, OpDescUtilsSupply) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  auto one_node = builder.AddNode("One", "One", 3, 3);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*attr_node).size(), 0);
  std::vector<ge::NodePtr> node_v;
  node_v.push_back(data_node);
  node_v.push_back(attr_node);
  EXPECT_EQ(OpDescUtils::GetInputData(node_v).size(), 0);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(*attr_node), 1);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(attr_node), 1);
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(*attr_node, 1), GeTensorDesc());
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(attr_node, 1), GeTensorDesc());
  size_t st = 0;
  EXPECT_EQ(OpDescUtils::GetNonConstInputIndex(attr_node, 1, st), false);
  EXPECT_EQ(OpDescUtils::GetConstInputs(nullptr).size(), 0);
  EXPECT_EQ(OpDescUtils::GetNonConstTensorDesc(attr_node).size(), 1);
  Operator op("name", "type");
  op.operator_impl_ = nullptr;
  EXPECT_EQ(OpDescUtils::GetInputConstData(op, 0), nullptr);
}

TEST_F(UtestOpDescUtils, ClearInputDesc_Nullptr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 1);
  data_node->impl_->op_->impl_ = nullptr;
  EXPECT_EQ(OpDescUtils::ClearInputDesc(data_node), false);
}

TEST_F(UtestOpDescUtils, ClearOutputDesc_Nullptr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 1);
  data_node->impl_->op_->impl_ = nullptr;
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(data_node), false);
}

TEST_F(UtestOpDescUtils, ClearOutputDesc_Normal) {
  OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
  EXPECT_NE(odp, nullptr);
  EXPECT_NE(odp->impl_, nullptr);
  EXPECT_EQ(odp->impl_->outputs_desc_.size(), 0);
  odp->impl_->outputs_desc_.push_back(std::make_shared<GeTensorDesc>());
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(odp, 0), true);
}

TEST_F(UtestOpDescUtils, GetWeightsFromNodes) {
  auto graph = BuildGraph3();
  auto cast_node = graph->FindNode("cast");
  auto enter_node = graph->FindNode("enter");
  auto in_nodes_and_anchors = cast_node->GetInDataNodesAndAnchors();
  EXPECT_EQ(in_nodes_and_anchors.size(), 1);
  EXPECT_EQ(in_nodes_and_anchors.begin()->first->GetName(), "enter");
  EXPECT_EQ(in_nodes_and_anchors.begin()->second->GetIdx(), 0);
  auto opdsc1 = in_nodes_and_anchors.begin()->first->GetOpDesc();
  bool is_potential_const1 = false;
  auto has_attr1 = AttrUtils::GetBool(opdsc1, ATTR_NAME_POTENTIAL_CONST, is_potential_const1);
  EXPECT_EQ(has_attr1, false);

  EXPECT_EQ(in_nodes_and_anchors.size(), 1);
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*cast_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 1);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 0);

  auto opdsc = nodes_2_out_anchor[0].first->GetOpDesc();
  bool is_potential_const = false;
  auto has_attr = AttrUtils::GetBool(opdsc, ATTR_NAME_POTENTIAL_CONST, is_potential_const);
  EXPECT_EQ(has_attr, true);
  auto weights = OpDescUtils::GetWeightsFromNodes(nodes_2_out_anchor);
}

TEST_F(UtestOpDescUtils, GetConstInputNode_Const_Enter_Other) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*const1).size(), 1);

  auto enter1 = builder.AddNode("enter1", "Enter", 1, 1);
  auto enter2 = builder.AddNode("enter2", "Enter", 1, 1);
  EXPECT_EQ(enter1->AddLinkFrom(enter2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*enter1).size(), 0);

  auto other1 = builder.AddNode("other1", "Enter", 1, 1);
  auto other2 = builder.AddNode("other2", "other", 1, 1);
  EXPECT_EQ(other1->AddLinkFrom(other2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*other1).size(), 0);
}

TEST_F(UtestOpDescUtils, GetInputData_Weight) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor0});

  std::vector<ge::NodePtr> vec;
  vec.push_back(const1);
  EXPECT_EQ(OpDescUtils::GetInputData(vec).size(), 1);
}

TEST_F(UtestOpDescUtils, GetNonConstInputsSize) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);

  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(nullptr), 0);
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*const1), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(*const1), 0);
  EXPECT_EQ(const1->GetAllInDataAnchors().size(), 1);
  auto in_anch = const1->GetAllInDataAnchors().at(0);
  EXPECT_EQ(AnchorUtils::SetStatus(in_anch, ANCHOR_DATA), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(*const1), 1);
}

TEST_F(UtestOpDescUtils, AddConstOpToAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  EXPECT_EQ(const1->GetAllInDataAnchors().size(), 1);
  auto in_anch = const1->GetAllInDataAnchors().at(0);
  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));

  EXPECT_EQ(OpDescUtils::AddConstOpToAnchor(in_anch, tensor0), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtils, GetNonConstInputIndex) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);

  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*attr_node), GRAPH_SUCCESS);
  size_t st = 0;
  EXPECT_EQ(OpDescUtils::GetNonConstInputIndex(attr_node, 1, st), false);
}

TEST_F(UtestOpDescUtils, GetNonConstInputTensorDesc) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(), 2);
  auto in_anch = attr_node->GetAllInDataAnchors().at(0);
  EXPECT_NE(in_anch, nullptr);

  EXPECT_EQ(AnchorUtils::SetStatus(in_anch, ANCHOR_DATA), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(attr_node, 1), GeTensorDesc());
}

TEST_F(UtestOpDescUtils, GetNonConstInputTensorDesc_SetStatus) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(), 2);
  auto in_anch = attr_node->GetAllInDataAnchors().at(0);
  EXPECT_NE(in_anch, nullptr);

  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(attr_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(attr_node, 1), GeTensorDesc());
}

TEST_F(UtestOpDescUtils, IsNonConstInput) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(attr_node, 1), false);


  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*attr_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(attr_node, 1), false);

  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(const1, 0), false);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(const2, 0), false);
}

TEST_F(UtestOpDescUtils, GetNonConstTensorDesc) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(OpDescUtils::GetNonConstTensorDesc(nullptr).size(), 0);
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*attr_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstTensorDesc(attr_node).size(), 0);
}

TEST_F(UtestOpDescUtils, GetConstInputs_Const) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(const1->GetType(), "Const");
  EXPECT_EQ(OpDescUtils::GetConstInputs(*const1).size(), 1);
}

TEST_F(UtestOpDescUtils, GetConstInputs_Switch) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto sw1 = builder.AddNode("sw1", "Switch", 1, 1);
  auto mm1 = builder.AddNode("mm1", "MatMul", 1, 1);
  EXPECT_EQ(sw1->AddLinkFrom(mm1), GRAPH_SUCCESS);
  EXPECT_EQ(sw1->GetType(), "Switch");
  EXPECT_EQ(mm1->GetType(), "MatMul");
  EXPECT_EQ(OpDescUtils::GetConstInputs(*sw1).size(), 0);
}

TEST_F(UtestOpDescUtils, MutableWeights) {
  auto node = std::make_shared<Node>();
  node = nullptr;
  EXPECT_EQ(OpDescUtils::MutableWeights(node).size(), 0);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ph = builder.AddNode("ph", "PlaceHolder", 1, 1);
  EXPECT_EQ(OpDescUtils::MutableWeights(*ph).size(), 0);
}

TEST_F(UtestOpDescUtils, MutableWeights_Nullptr) {
  OpDescPtr odp = std::make_shared<OpDesc>();
  odp = nullptr;
  EXPECT_EQ(OpDescUtils::MutableWeights(odp), nullptr);
}

TEST_F(UtestOpDescUtils, SetWeights) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  std::map<int, ge::GeTensorPtr> weights_map;
  weights_map[1] = std::make_shared<GeTensor>();
  EXPECT_EQ(OpDescUtils::SetWeights(*const1, weights_map), GRAPH_SUCCESS);

  auto non1 = builder.AddNode("nonconst1", "NonConst", 1, 1);
  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  EXPECT_EQ(OpDescUtils::SetWeights(non1, {tensor0}), GRAPH_SUCCESS);

  weights_map[2] = tensor0;
  EXPECT_EQ(OpDescUtils::SetWeights(*const1, weights_map), GRAPH_PARAM_INVALID);
}

TEST_F(UtestOpDescUtils, CopyConstructOpdesc) {
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetFormat(FORMAT_NCHW);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);
  vector<int64_t> input_size = {12};
  AttrUtils::SetListInt(td, "input_size", input_size);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOptionalInputDesc("x3", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "SAME");

  OpDescPtr new_desc = std::make_shared<OpDesc>(*op_desc);
  EXPECT_TRUE(new_desc->OpDescMembersAreEqual(*op_desc));
  EXPECT_TRUE(new_desc->OpDescAttrsAreEqual(*op_desc));
  EXPECT_TRUE(new_desc->OpDescGenTensorDescsAreEqual(*op_desc));
  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  EXPECT_EQ(new_desc->GetInputsSize(), 3);
  EXPECT_EQ(new_desc->GetOutputsSize(), 1);

  EXPECT_EQ(new_desc->GetInputDescPtr("x1"), new_desc->GetInputDescPtr(0));
  EXPECT_EQ(new_desc->GetInputDescPtr("x2"), new_desc->GetInputDescPtr(1));
  EXPECT_EQ(new_desc->MutableOutputDesc("y"), new_desc->MutableOutputDesc(0));

  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  vector<int64_t> new_input_size;
  EXPECT_TRUE(AttrUtils::GetListInt(new_desc->GetInputDescPtr(0), "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));

  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  new_input_size.clear();
  auto new_input_desc = new_desc->GetInputDescPtr(1);
  EXPECT_TRUE(AttrUtils::GetListInt(new_input_desc, "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));

  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  new_input_size.clear();
  EXPECT_TRUE(AttrUtils::GetListInt(new_desc->GetInputDescPtr(0), "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));
  op_desc->MutableInputDesc(0)->SetFormat(FORMAT_NC1HWC0_C04);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_FALSE(new_desc->OpDescGenTensorDescsAreEqual(*op_desc));
}

TEST_F(UtestOpDescUtils, CopyOpdesc) {
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetFormat(FORMAT_NCHW);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);
  vector<int64_t> input_size = {12};
  AttrUtils::SetListInt(td, "input_size", input_size);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOptionalInputDesc("x3", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "SAME");

  auto new_desc = OpDescUtils::CopyOpDesc(op_desc);

  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  EXPECT_EQ(new_desc->GetInputsSize(), 3);
  EXPECT_EQ(new_desc->GetOutputsSize(), 1);

  EXPECT_EQ(new_desc->GetInputDescPtr("x1"), new_desc->GetInputDescPtr(0));
  EXPECT_EQ(new_desc->GetInputDescPtr("x2"), new_desc->GetInputDescPtr(1));
  EXPECT_EQ(new_desc->MutableOutputDesc("y"), new_desc->MutableOutputDesc(0));

  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  vector<int64_t> new_input_size;
  EXPECT_TRUE(AttrUtils::GetListInt(new_desc->GetInputDescPtr(0), "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));

  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  new_input_size.clear();
  auto new_input_desc = new_desc->GetInputDescPtr(1);
  EXPECT_TRUE(AttrUtils::GetListInt(new_input_desc, "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));

  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  new_input_size.clear();
  EXPECT_TRUE(AttrUtils::GetListInt(new_desc->GetInputDescPtr(0), "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));
}


TEST_F(UtestOpDescUtils, CopyOpdesc2) {
  GeTensorDesc td = StandardTd_5d_1_1_224_224();

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "VALID");

  auto new_desc1 = OpDescUtils::CopyOpDesc(op_desc);

  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "VALID");

  AttrUtils::SetStr(new_desc1, "padding", "SAME");
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  auto new_desc2 = OpDescUtils::CopyOpDesc(new_desc1);
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc2, "padding", padding));
  EXPECT_EQ(padding, "SAME");
}

TEST_F(UtestOpDescUtils, CloneOpdesc) {
  GeTensorDesc td = StandardTd_5d_1_1_224_224();

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "VALID");

  auto new_desc1 = OpDescUtils::CloneOpDesc(op_desc);

  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "VALID");

  AttrUtils::SetStr(new_desc1, "padding", "SAME");
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  auto new_desc2 = OpDescUtils::CloneOpDesc(new_desc1);
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc2, "padding", padding));
  EXPECT_EQ(padding, "SAME");
}

REG_OP(DynamicOpUt)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .INPUT(a, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .INPUT(b, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
        .ATTR(transpose_x1, Bool, false)
        .ATTR(transpose_x2, Bool, false)
        .OP_END_FACTORY_REG(DynamicOpUt)

TEST_F(UtestOpDescUtils, GetInputIrIndexes2InstanceIndexesPairMap_NullOpDescFailed) {
  auto ir_index_to_instance_index_pair_map = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(nullptr);
  ASSERT_TRUE(ir_index_to_instance_index_pair_map.empty());
}

TEST_F(UtestOpDescUtils, GetOutputIrIndexes2InstanceIndexesPairMap_NullOpDescFailed) {
  auto ir_index_to_instance_index_pair_map = OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(nullptr);
  ASSERT_TRUE(ir_index_to_instance_index_pair_map.empty());
}

void IrIndexAndInstanceIndexCheck(size_t dynamic_input_num, bool has_optional_input) {
  size_t optional_input_num = has_optional_input ? 1U : 0U;
  auto graph = BuildGraph4(dynamic_input_num, has_optional_input);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  size_t index = 0;
  auto &name_index = op_desc->MutableAllInputName();
  name_index.clear();
  for (size_t i = 0U; i < dynamic_input_num; ++i) {
    name_index["x" + std::to_string(i)] = index++;
  }
  name_index["a"] = index++;
  if (optional_input_num == 1) {
    name_index["bias"] = index++;
  }
  name_index["b"] = index++;

  auto ir_index_to_instance_index_pair_map = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  ASSERT_FALSE(ir_index_to_instance_index_pair_map.empty());

  std::map<size_t, std::pair<size_t, size_t>> expect_map;
  expect_map[0] = std::pair<size_t, size_t>(0, dynamic_input_num);
  expect_map[1] = std::pair<size_t, size_t>(dynamic_input_num, 1);
  expect_map[2] = std::pair<size_t, size_t>(dynamic_input_num + 1, optional_input_num);
  expect_map[3] = std::pair<size_t, size_t>(dynamic_input_num + 1 + optional_input_num, 1);
  EXPECT_EQ(ir_index_to_instance_index_pair_map, expect_map);
}

TEST_F(UtestOpDescUtils, GetInputIrIndexes2InstanceIndexesPairMap_Success) {
  IrIndexAndInstanceIndexCheck(0, true);
  IrIndexAndInstanceIndexCheck(0, false);
  IrIndexAndInstanceIndexCheck(1, true);
  IrIndexAndInstanceIndexCheck(1, false);
  IrIndexAndInstanceIndexCheck(3, true);
  IrIndexAndInstanceIndexCheck(3, false);
}

TEST_F(UtestOpDescUtils, GetInputIrIndexes2InstanceIndexesPairMap_DynamicInputNameNotMatch_Failed) {
  size_t dynamic_input_num = 1;
  size_t has_optional_input = true;
  size_t optional_input_num = has_optional_input ? 1U : 0U;
  auto graph = BuildGraph4(dynamic_input_num, has_optional_input);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  size_t index = 0;
  auto &name_index = op_desc->MutableAllInputName();
  name_index.clear();

  name_index["x0"] = index++;
  name_index["x2"] = index++; // error name

  name_index["a"] = index++;
  if (optional_input_num == 1) {
    name_index["bias"] = index++;
  }
  name_index["b"] = index++;

  auto ir_index_to_instance_index_pair_map = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  ASSERT_TRUE(ir_index_to_instance_index_pair_map.empty());
}

void GetIrIndexCheck(size_t dynamic_input_num, bool has_optional_input) {
  size_t optional_input_num = has_optional_input ? 1U : 0U;
  auto graph = BuildGraph4(dynamic_input_num, has_optional_input);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  size_t index = 0;
  auto &name_index = op_desc->MutableAllInputName();
  name_index.clear();
  for (size_t i = 0U; i < dynamic_input_num; ++i) {
    name_index["x" + std::to_string(i)] = index++;
  }
  name_index["a"] = index++;
  if (optional_input_num == 1) {
    name_index["bias"] = index++;
  }
  name_index["b"] = index++;

  index = 0U;
  std::map<size_t, size_t> expect_instance_index_to_ir_index_map;
  for (size_t i = 0U; i < dynamic_input_num; ++i) {
    expect_instance_index_to_ir_index_map[index++] = 0;
  }
  expect_instance_index_to_ir_index_map[index++] = 1;
  if (has_optional_input) {
    expect_instance_index_to_ir_index_map[index++] = 2;
  }
  expect_instance_index_to_ir_index_map[index++] = 3;
  for (auto &instance_index_to_ir_index : expect_instance_index_to_ir_index_map) {
    auto input_index = instance_index_to_ir_index.first;
    size_t ir_index;
    auto ret = OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, input_index, ir_index);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
    ASSERT_EQ(ir_index, instance_index_to_ir_index.second);
  }
}

TEST_F(UtestOpDescUtils, GetInputIrIndexeByInstanceIndexe_Success) {
  GetIrIndexCheck(0, true);
  GetIrIndexCheck(0, false);
  GetIrIndexCheck(1, true);
  GetIrIndexCheck(1, false);
  GetIrIndexCheck(3, true);
  GetIrIndexCheck(3, false);
}

TEST_F(UtestOpDescUtils, GetInputIrIndexeByInstanceIndexe_DynamicNameNotmatch_Failed) {
  size_t dynamic_input_num = 1;
  size_t has_optional_input = true;
  size_t optional_input_num = has_optional_input ? 1U : 0U;
  auto graph = BuildGraph4(dynamic_input_num, has_optional_input);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  size_t index = 0;
  auto &name_index = op_desc->MutableAllInputName();
  name_index.clear();

  name_index["x0"] = index++;
  name_index["x2"] = index++; // error name

  name_index["a"] = index++;
  if (optional_input_num == 1) {
    name_index["bias"] = index++;
  }
  name_index["b"] = index++;
  size_t ir_index;
  auto ret = OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, 2, ir_index);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOpDescUtils, GetInputIrIndexeByInstanceIndexe_ActualInputsIsMoreThanIrInputsNum_Success) {
  size_t dynamic_input_num = 1;
  size_t has_optional_input = true;
  size_t optional_input_num = has_optional_input ? 1U : 0U;
  auto graph = BuildGraph4(dynamic_input_num, has_optional_input);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  size_t index = 0;
  auto &name_index = op_desc->MutableAllInputName();
  name_index.clear();

  name_index["x0"] = index++;
  name_index["x1"] = index++; // error name

  name_index["a"] = index++;
  if (optional_input_num == 1) {
    name_index["bias"] = index++;
  }
  name_index["b"] = index++;
  name_index["assist_matrix"] = index++;
  size_t ir_index;

  int32_t event_level;
  int32_t old_level = dlog_getlevel(GE_MODULE_NAME, &event_level);
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, event_level);
  auto ret = OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, 5, ir_index);
  dlog_setlevel(GE_MODULE_NAME, old_level, event_level);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ir_index, std::numeric_limits<size_t>::max());
}

void GetOutputIrIndexCheck(size_t dynamic_output_num) {
  auto graph = BuildGraph6(dynamic_output_num);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  // 只处理output
  size_t index = 0;
  auto &name_index = op_desc->MutableAllOutputName();
  name_index.clear();
  name_index["x"] = index++;
  for (size_t i = 0U; i < dynamic_output_num; ++i) {
    name_index["y" + std::to_string(i)] = index++;
  }
  name_index["z"] = index++;

  index = 0U;
  std::map<size_t, size_t> expect_instance_index_to_ir_index_map;
  expect_instance_index_to_ir_index_map[index++] = 0; // x
  for (size_t i = 0U; i < dynamic_output_num; ++i) { // y
    expect_instance_index_to_ir_index_map[index++] = 1;
  }
  expect_instance_index_to_ir_index_map[index++] = 2;

  for (auto &instance_index_to_ir_index : expect_instance_index_to_ir_index_map) {
    auto input_index = instance_index_to_ir_index.first;
    size_t ir_index;
    auto ret = OpDescUtils::GetOutputIrIndexByInstanceIndex(op_desc, input_index, ir_index);
    ASSERT_EQ(ret, GRAPH_SUCCESS);
    ASSERT_EQ(ir_index, instance_index_to_ir_index.second);
  }
}

TEST_F(UtestOpDescUtils, GetOutputIrIndexByInstanceIndex_Success) {
  GetOutputIrIndexCheck(0);
  GetOutputIrIndexCheck(1);
  GetOutputIrIndexCheck(2);
  GetOutputIrIndexCheck(3);
}

TEST_F(UtestOpDescUtils, GetOutputIrIndexByInstanceIndexDynamicNameNotmatch_Failed) {
  auto graph = BuildGraph6(2);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  // 只处理output
  size_t index = 0;
  auto &name_index = op_desc->MutableAllOutputName();
  name_index.clear();
  name_index["x"] = index++;
  name_index["y0"] = index++;
  name_index["y2"] = index++;
  name_index["z"] = index++;

  size_t ir_index;
  auto ret = OpDescUtils::GetOutputIrIndexByInstanceIndex(op_desc, 1, ir_index);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ir_index, std::numeric_limits<size_t>::max());
}

TEST_F(UtestOpDescUtils, GetOutputIrIndexByInstanceIndexActualInputsIsMoreThanIrOutputsNum_Success) {
  auto graph = BuildGraph6(2);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();

  // 只处理output
  size_t index = 0;
  auto &name_index = op_desc->MutableAllOutputName();
  name_index.clear();
  name_index["x"] = index++;
  name_index["y0"] = index++;
  name_index["y1"] = index++;
  name_index["z"] = index++;
  name_index["u"] = index++;

  size_t ir_index;
  auto ret = OpDescUtils::GetOutputIrIndexByInstanceIndex(op_desc, 4, ir_index);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(ir_index, std::numeric_limits<size_t>::max());
}

TEST_F(UtestOpDescUtils, GetOutputIrIndexeByInstanceIndexe_NoOutput_Success) {
  auto graph = BuildGraph5();
  auto node_without_outputs = graph->FindNode("noop");
  auto op_desc = node_without_outputs->GetOpDesc();

  auto ir_index_to_instance_index_pair_map = OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(op_desc);
  ASSERT_TRUE(ir_index_to_instance_index_pair_map.empty());
}

TEST_F(UtestOpDescUtils, GetOutputIrIndexeByInstanceIndexe_UnknownOutputIrType_Failed) {
  auto graph = BuildGraph4(2, false);
  auto dynamic_op_ut_node = graph->FindNode("dynamic_op_ut");
  auto op_desc = dynamic_op_ut_node->GetOpDesc();
  op_desc->AppendIrOutput("y", kIrOutputTypeEnd);// invalid IrType

  auto ir_index_to_instance_index_pair_map = OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(op_desc);
  ASSERT_TRUE(ir_index_to_instance_index_pair_map.empty());
}

#define CHECK_IR_RANGE(Idx, Start, Num)                                                                                \
  EXPECT_EQ(ir_ranges[Idx].first, Start);                                                                              \
  EXPECT_EQ(ir_ranges[Idx].second, Num)

REG_OP(DescUtilTestDynamicFirst)
    .DYNAMIC_INPUT(input0, "T")
    .INPUT(input1, "T")
    .INPUT(input2, "T")
    .DATATYPE(T, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DescUtilTestDynamicFirst);
TEST_F(UtestOpDescUtils, get_input_desc_range_for_dynamic_first_ir_desc_end) {
  auto op = op::DescUtilTestDynamicFirst();
  op.create_dynamic_input_input0(2);  // Dynamic desc出现在尾部
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  // |0       |1      |2        |3
  // [input1, input2, input0:0, input0:1]
  // dynamic input1
  CHECK_IR_RANGE(0, 2, 2);
  // static input2
  CHECK_IR_RANGE(1, 0, 1);
  // static input2
  CHECK_IR_RANGE(2, 1, 1);
}
TEST_F(UtestOpDescUtils, get_input_desc_range_for_dynamic_first_ir_desc_begin) {
  auto op = op::DescUtilTestDynamicFirst();
  op.create_dynamic_input_input0(2, false); // Dynamic desc出现在头部
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  // |0         |1        |2      |3
  // [input0:0, input0:1, input1, input2]
  // dynamic input1
  CHECK_IR_RANGE(0, 0, 2);
  // static input2
  CHECK_IR_RANGE(1, 2, 1);
  // static input2
  CHECK_IR_RANGE(2, 3, 1);
}
TEST_F(UtestOpDescUtils, get_input_desc_range_for_dynamic_first_ir_desc_middle) {
  auto op = op::DescUtilTestDynamicFirst();
  op.create_dynamic_input_byindex_input0(2, 1);  // Dynamic desc出现在中间
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  // |0       |1        |2        |3
  // [input1, input0:0, input0:1, input2]
  // dynamic input1
  CHECK_IR_RANGE(0, 1, 2);
  // static input2
  CHECK_IR_RANGE(1, 0, 1);
  // static input2
  CHECK_IR_RANGE(2, 3, 1);
}

REG_OP(DescUtilTestMultiDynamic)
    .DYNAMIC_INPUT(input0, "T")
    .DYNAMIC_INPUT(input1, "T")
    .DATATYPE(T, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DescUtilTestMultiDynamic);
TEST_F(UtestOpDescUtils, get_input_desc_range_for_mulit_dynamic) {
  auto op = op::DescUtilTestMultiDynamic();
  op.create_dynamic_input_input0(2);
  op.create_dynamic_input_input1(2);
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 2);
  // |0         |1        |2        |3
  // [input0:0, input0:1, input1:0, input1:1]
  // dynamic input0
  CHECK_IR_RANGE(0, 0, 2);
  // dynamic input1
  CHECK_IR_RANGE(1, 2, 2);
}

TEST_F(UtestOpDescUtils, get_input_desc_range_for_mulit_dynamic_mis_order) {
  auto op = op::DescUtilTestMultiDynamic();
  op.create_dynamic_input_input1(2); // 首先创建input2
  op.create_dynamic_input_input0(2);
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 2);
  // |0         |1        |2        |3
  // [input1:0, input1:1, input0:0, input0:1]
  // dynamic input0
  CHECK_IR_RANGE(0, 2, 2);
  // dynamic input1
  CHECK_IR_RANGE(1, 0, 2);
}

REG_OP(DescUtilTestUnfedOptional)
    .OPTIONAL_INPUT(input0, "T")
    .OPTIONAL_INPUT(input1, "T")
    .OPTIONAL_INPUT(input2, "T")
    .DATATYPE(T, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(DescUtilTestUnfedOptional);
TEST_F(UtestOpDescUtils, get_input_desc_instance_range_for_unfed_optional) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);  // 全部为optional且未feed
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 0);
  CHECK_IR_RANGE(1, 0, 0);
  CHECK_IR_RANGE(2, 0, 0);
}
TEST_F(UtestOpDescUtils, get_input_desc_raw_range_for_unfed_optional) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);  // 全部为optional且未feed
  std::map<size_t, std::pair<size_t, size_t>> ir_ranges;
  ASSERT_EQ(OpDescUtils::GetIrInputRawDescRange(desc, ir_ranges), GRAPH_SUCCESS);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 0);  // Raw range会存储其desc在数据上的位置
  CHECK_IR_RANGE(1, 1, 0);
  CHECK_IR_RANGE(2, 2, 0);
}
TEST_F(UtestOpDescUtils, get_input_desc_instance_range_for_unfed_optional_begin) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input1", GeTensorDesc());
  desc->UpdateInputDesc("input2", GeTensorDesc());
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 0);
  CHECK_IR_RANGE(1, 0, 1);
  CHECK_IR_RANGE(2, 1, 1);
}
TEST_F(UtestOpDescUtils, get_input_desc_raw_range_for_unfed_optional_begin) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input1", GeTensorDesc());
  desc->UpdateInputDesc("input2", GeTensorDesc());
  std::map<size_t, std::pair<size_t, size_t>> ir_ranges;
  ASSERT_EQ(OpDescUtils::GetIrInputRawDescRange(desc, ir_ranges), GRAPH_SUCCESS);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 0);
  CHECK_IR_RANGE(1, 1, 1);
  CHECK_IR_RANGE(2, 2, 1);
}
TEST_F(UtestOpDescUtils, get_input_desc_instance_range_for_unfed_optional_middle) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input0", GeTensorDesc());
  desc->UpdateInputDesc("input2", GeTensorDesc());
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 1);
  CHECK_IR_RANGE(1, 1, 0);
  CHECK_IR_RANGE(2, 1, 1);
}
TEST_F(UtestOpDescUtils, get_input_desc_raw_range_for_unfed_optional_middle) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input0", GeTensorDesc());
  desc->UpdateInputDesc("input2", GeTensorDesc());
  std::map<size_t, std::pair<size_t, size_t>> ir_ranges;
  ASSERT_EQ(OpDescUtils::GetIrInputRawDescRange(desc, ir_ranges), GRAPH_SUCCESS);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 1);
  CHECK_IR_RANGE(1, 1, 0);
  CHECK_IR_RANGE(2, 2, 1);
}
TEST_F(UtestOpDescUtils, get_input_desc_instance_range_for_unfed_optional_end) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input0", GeTensorDesc());
  desc->UpdateInputDesc("input1", GeTensorDesc());
  auto ir_ranges = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(desc);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 1);
  CHECK_IR_RANGE(1, 1, 1);
  CHECK_IR_RANGE(2, 2, 0);
}
TEST_F(UtestOpDescUtils, get_input_desc_raw_range_for_unfed_optional_end) {
  auto op = op::DescUtilTestUnfedOptional();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input0", GeTensorDesc());
  desc->UpdateInputDesc("input1", GeTensorDesc());
  std::map<size_t, std::pair<size_t, size_t>> ir_ranges;
  ASSERT_EQ(OpDescUtils::GetIrInputRawDescRange(desc, ir_ranges), GRAPH_SUCCESS);

  EXPECT_EQ(ir_ranges.size(), 3);
  CHECK_IR_RANGE(0, 0, 1);
  CHECK_IR_RANGE(1, 1, 1);
  CHECK_IR_RANGE(2, 2, 0);
}

REG_OP(OpTesGetPromoteInputList1)
    .INPUT(input1, "T1")
    .DYNAMIC_INPUT(input2, "T2")
    .OUTPUT(output1, "T3")
    .DATATYPE(T1, TensorType({DT_INT32, DT_FLOAT}))
    .DATATYPE(T2, TensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(OpTesGetPromoteInputList1);

TEST_F(UtestOpDescUtils, get_promote_input_list_one_output) {
  auto op = op::OpTesGetPromoteInputList1();
  op.create_dynamic_input_input2(2);
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  std::vector<std::vector<size_t>> ir_input_list;
  OpDescUtils::GetPromoteIrInputList(desc, ir_input_list);
  EXPECT_EQ(ir_input_list.size(), 1);
  EXPECT_EQ(ir_input_list[0].size(), 2);
  EXPECT_EQ(ir_input_list[0][0], 0);
  EXPECT_EQ(ir_input_list[0][1], 1);

  std::vector<std::vector<size_t>> instance_input_list;
  OpDescUtils::GetPromoteInstanceInputList(desc, instance_input_list);
  EXPECT_EQ(instance_input_list.size(), 1);
  EXPECT_EQ(instance_input_list[0].size(), 3);
  EXPECT_EQ(instance_input_list[0][0], 0);
  EXPECT_EQ(instance_input_list[0][1], 1);
  EXPECT_EQ(instance_input_list[0][2], 2);
}

REG_OP(OpTesGetPromoteInputList2)
    .INPUT(input1, "T1")
    .OPTIONAL_INPUT(input2, "T2")
    .INPUT(input3, "T3")
    .OPTIONAL_INPUT(input4, "T4")
    .OUTPUT(output1, "T5")
    .OUTPUT(output2, "T6")
    .DATATYPE(T1, TensorType({DT_INT32, DT_FLOAT}))
    .DATATYPE(T2, TensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T3, TensorType({DT_INT32, DT_FLOAT}))
    .DATATYPE(T4, TensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T5, Promote({"T1", "T2"}))
    .DATATYPE(T6, Promote({"T3", "T4"}))
    .OP_END_FACTORY_REG(OpTesGetPromoteInputList2);

TEST_F(UtestOpDescUtils, get_promote_input_list_outputs) {
  auto op = op::OpTesGetPromoteInputList2();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  desc->UpdateInputDesc("input2", GeTensorDesc());

  std::vector<std::vector<size_t>> ir_input_list;
  OpDescUtils::GetPromoteIrInputList(desc, ir_input_list);
  EXPECT_EQ(ir_input_list.size(), 2);
  EXPECT_EQ(ir_input_list[0].size(), 2);
  EXPECT_EQ(ir_input_list[1].size(), 2);
  EXPECT_EQ(ir_input_list[0][0], 0);
  EXPECT_EQ(ir_input_list[0][1], 1);
  EXPECT_EQ(ir_input_list[1][0], 2);
  EXPECT_EQ(ir_input_list[1][1], 3);

  std::vector<std::vector<size_t>> instance_input_list;
  OpDescUtils::GetPromoteInstanceInputList(desc, instance_input_list);
  EXPECT_EQ(instance_input_list.size(), 2);
  EXPECT_EQ(instance_input_list[0].size(), 2);
  EXPECT_EQ(instance_input_list[1].size(), 1);
  EXPECT_EQ(instance_input_list[0][0], 0);
  EXPECT_EQ(instance_input_list[0][1], 1);
  EXPECT_EQ(instance_input_list[1][0], 2);
}

REG_OP(OpTesGetPromoteInputList3)
    .INPUT(input1, "T")
    .INPUT(input2, "T")
    .OUTPUT(output1, "T")
    .DATATYPE(T, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(OpTesGetPromoteInputList3);

TEST_F(UtestOpDescUtils, get_promote_input_list_none_output) {
  auto op = op::OpTesGetPromoteInputList3();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);

  std::vector<std::vector<size_t>> ir_input_list;
  OpDescUtils::GetPromoteIrInputList(desc, ir_input_list);
  EXPECT_TRUE(ir_input_list.empty());
  std::vector<std::vector<size_t>> instance_input_list;
  OpDescUtils::GetPromoteInstanceInputList(desc, instance_input_list);
  EXPECT_TRUE(instance_input_list.empty());
}

TEST_F(UtestOpDescUtils, CreateConstOpWithOutCopy) {
  ge::GeTensorDesc ge_tensor(GeShape({8,8,8}), FORMAT_ND, DT_FLOAT16);
  ge_tensor.SetName("test");
  ge::GeTensorPtr const_tensor_ptr = ge::MakeShared<ge::GeTensor>(ge_tensor);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOpZeroCopy(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  (void) const_op.SetInput("test", const_op);
  ConstGeTensorPtr weight;
  EXPECT_TRUE(ConstantUtils::GetWeight(const_op_desc, 0UL, weight) == true);
  EXPECT_TRUE(weight->GetData().GetData() == const_tensor_ptr->GetData().GetData());
}

TEST_F(UtestOpDescUtils, CreateConstOpWithCopy) {
  ge::GeTensorDesc ge_tensor(GeShape({8,8,8}), FORMAT_ND, DT_FLOAT16);
  ge_tensor.SetName("test");
  ge::GeTensorPtr const_tensor_ptr = ge::MakeShared<ge::GeTensor>(ge_tensor);
  constexpr int32_t kAlignedSize = 256;
  auto aligned_ptr = std::make_shared<AlignedPtr>(kAlignedSize);
  const_tensor_ptr->SetData(aligned_ptr, 256);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  (void) const_op.SetInput("test", const_op);
  ConstGeTensorPtr weight;
  EXPECT_TRUE(ConstantUtils::GetWeight(const_op_desc, 0UL, weight) == true);
  EXPECT_TRUE(weight->GetData().GetData() != const_tensor_ptr->GetData().GetData());
}
}  // namespace ge
