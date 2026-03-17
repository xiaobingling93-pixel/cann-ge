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
#include <string>
#include "ge_graph_dsl/graph_dsl.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/control_flow_and_stream/cond_remove_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"

using namespace std;
using namespace ge;

class UtestCondRemovePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
  OpDescPtr CreateOpDesc(const string &name, const string &type, const GeTensorDesc &in_tensor, uint32_t input_num,
                         const GeTensorDesc &out_tensor, uint32_t output_num) {
    OpDescPtr op_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, type));
    if (op_desc == nullptr) {
      return nullptr;
    }
    for (uint32_t i = 0; i < input_num; i++) {
      op_desc->AddInputDesc(in_tensor);
    }
    for (uint32_t i = 0; i < output_num; i++) {
      op_desc->AddOutputDesc(out_tensor);
    }
    return op_desc;
  }

  Status RunGraphPass(const GeTensorDesc &tensor_desc, const ComputeGraphPtr &graph){
    NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, tensor_desc, 1, tensor_desc, 1));
    NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
    EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

    CondRemovePass pass;
    return pass.Run(if_node);
  }
};

TEST_F(UtestCondRemovePass, no_cond_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, GeTensorDesc(), 1, GeTensorDesc(), 1));

  CondRemovePass pass;
  auto ret = pass.Run(data_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondRemovePass, no_graph_fail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  auto pre_op_desc = std::make_shared<OpDesc>("pre", IF);
  NodePtr pre_node = graph->AddNode(pre_op_desc);

  CondRemovePass pass;
  auto ret = pass.Run(pre_node);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestCondRemovePass, if_cond_not_const_succ) {
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("data", DATA, tensor_desc, 1, tensor_desc, 1));
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

  CondRemovePass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondRemovePass, if_cond_bool_const_succ) {
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("const", CONSTANTOP, tensor_desc, 1, tensor_desc, 1));
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
  NodePtr output_node = graph->AddNode(CreateOpDesc("output", NETOUTPUT, tensor_desc, 1, tensor_desc, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(if_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)), SUCCESS);

  CondRemovePass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondRemovePass, while_cond_bool_const_succ) {
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_BOOL);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  NodePtr data_node = graph->AddNode(CreateOpDesc("const", CONSTANTOP, tensor_desc, 1, tensor_desc, 1));
  NodePtr while_node = graph->AddNode(CreateOpDesc("while", WHILE, tensor_desc, 2, tensor_desc, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), while_node->GetInDataAnchor(0)), SUCCESS);

  CondRemovePass pass;
  auto ret = pass.Run(while_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondRemovePass, if_cond_int_const_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  vector<int32_t> data_vec = {0};
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  auto op_value = CreateOpDesc("const", CONSTANTOP, tensor_desc, 1, tensor_desc, 1);
  OpDescUtils::SetWeights(op_value, value_tensor);
  NodePtr data_node = graph->AddNode(op_value);
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

  std::string then_name = "then";
  ComputeGraphPtr then_graph = std::make_shared<ComputeGraph>(then_name);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(graph);
  if_node->GetOpDesc()->AddSubgraphName(then_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, then_name);

  std::string else_name = "else";
  ComputeGraphPtr else_graph = std::make_shared<ComputeGraph>(else_name);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(graph);
  if_node->GetOpDesc()->AddSubgraphName(else_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, else_name);
  EXPECT_EQ(graph->AddSubgraph(then_name, then_graph), GRAPH_SUCCESS);
  EXPECT_EQ(graph->AddSubgraph(else_name, else_graph), GRAPH_SUCCESS);

  CondRemovePass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondRemovePass, if_cond_int_const_succ_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  vector<int32_t> data_vec = {0};
  GeTensorDesc tensor_desc(GeShape({0}), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  auto op_value = CreateOpDesc("const", CONSTANTOP, tensor_desc, 1, tensor_desc, 1);
  OpDescUtils::SetWeights(op_value, value_tensor);
  NodePtr data_node = graph->AddNode(op_value);
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

  std::string then_name = "then";
  ComputeGraphPtr then_graph = std::make_shared<ComputeGraph>(then_name);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(graph);
  if_node->GetOpDesc()->AddSubgraphName(then_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0, then_name);

  std::string else_name = "else";
  ComputeGraphPtr else_graph = std::make_shared<ComputeGraph>(else_name);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(graph);
  if_node->GetOpDesc()->AddSubgraphName(else_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(1, else_name);
  EXPECT_EQ(graph->AddSubgraph(then_name, then_graph), GRAPH_SUCCESS);
  EXPECT_EQ(graph->AddSubgraph(else_name, else_graph), GRAPH_SUCCESS);

  CondRemovePass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestCondRemovePass, if_cond_int_const_fail_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("g");
  vector<int32_t> data_vec = {0};
  GeTensorDesc tensor_desc(GeShape({0}), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  auto op_value = CreateOpDesc("const", CONSTANTOP, tensor_desc, 1, tensor_desc, 1);
  OpDescUtils::SetWeights(op_value, value_tensor);
  NodePtr data_node = graph->AddNode(op_value);
  NodePtr if_node = graph->AddNode(CreateOpDesc("if", IF, tensor_desc, 2, tensor_desc, 1));
  EXPECT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0)), SUCCESS);

  std::string then_name = "then";
  ComputeGraphPtr then_graph = std::make_shared<ComputeGraph>(then_name);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(graph);
  if_node->GetOpDesc()->AddSubgraphName(then_name);

  std::string else_name = "else";
  ComputeGraphPtr else_graph = std::make_shared<ComputeGraph>(else_name);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(graph);
  if_node->GetOpDesc()->AddSubgraphName(else_name);
  EXPECT_EQ(graph->AddSubgraph(then_name, then_graph), GRAPH_SUCCESS);
  EXPECT_EQ(graph->AddSubgraph(else_name, else_graph), GRAPH_SUCCESS);

  CondRemovePass pass;
  auto ret = pass.Run(if_node);
  EXPECT_EQ(ret, FAILED);
}

// cond(const, int32)->case --  case1
//                   /      |
// data1 -----------        |
//                   /      |
// data2 -----------         --  case2
//                          |
//                           --  case3
TEST_F(UtestCondRemovePass, case_const_multidata_succ) {
  GeTensorDesc tensor(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  
  vector<int32_t> data_vec = {1};
  GeTensorDesc tensor_desc(GeShape({1}), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  auto op_value = CreateOpDesc("const", CONSTANTOP, tensor_desc, 1, tensor_desc, 1);
  OpDescUtils::SetWeights(op_value, value_tensor);
  NodePtr const_node = graph->AddNode(op_value);

  NodePtr data1_node = graph->AddNode(CreateOpDesc("data1", DATA, tensor, 1, tensor, 1));
  NodePtr data2_node = graph->AddNode(CreateOpDesc("data2", DATA, tensor, 1, tensor, 1));
  NodePtr case_node = graph->AddNode(CreateOpDesc("case", CASE, tensor, 3, tensor, 1));
  EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), case_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), case_node->GetInDataAnchor(1)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), case_node->GetInDataAnchor(2)), SUCCESS);

  std::string case1_name = "case1";
  ComputeGraphPtr case1_graph = std::make_shared<ComputeGraph>(case1_name);
  case1_graph->SetParentNode(case_node);
  case1_graph->SetParentGraph(graph);
  auto case1_data1_op = CreateOpDesc("case1_data1", DATA, tensor, 1, tensor, 1);
  auto case1_data2_op = CreateOpDesc("case1_data2", DATA, tensor, 1, tensor, 1);
  NodePtr case1_data1_node = case1_graph->AddNode(case1_data1_op);
  NodePtr case1_data2_node = case1_graph->AddNode(case1_data2_op);
  ge::AttrUtils::SetInt(case1_data1_node->GetOpDesc(), "_parent_node_index", 1);
  ge::AttrUtils::SetInt(case1_data2_node->GetOpDesc(), "_parent_node_index", 2);
  case1_graph->AddInputNode(case1_data1_node);
  case1_graph->AddInputNode(case1_data2_node);
  case_node->GetOpDesc()->AddSubgraphName(case1_name);
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, case1_name);

  std::string case2_name = "case2";
  ComputeGraphPtr case2_graph = std::make_shared<ComputeGraph>(case2_name);
  case2_graph->SetParentNode(case_node);
  case2_graph->SetParentGraph(graph);
  auto case2_data1_op = CreateOpDesc("case2_data1", DATA, tensor, 1, tensor, 1);
  auto case2_data2_op = CreateOpDesc("case2_data2", DATA, tensor, 1, tensor, 1);
  NodePtr case2_data1_node = case2_graph->AddNode(case2_data1_op);
  NodePtr case2_data2_node = case2_graph->AddNode(case2_data2_op);
  ge::AttrUtils::SetInt(case2_data1_node->GetOpDesc(), "_parent_node_index", 1);
  ge::AttrUtils::SetInt(case2_data2_node->GetOpDesc(), "_parent_node_index", 2);
  case2_graph->AddInputNode(case2_data1_node);
  case2_graph->AddInputNode(case2_data2_node);
  case_node->GetOpDesc()->AddSubgraphName(case2_name);
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, case2_name);

  std::string case3_name = "case3";
  ComputeGraphPtr case3_graph = std::make_shared<ComputeGraph>(case3_name);
  case3_graph->SetParentNode(case_node);
  case3_graph->SetParentGraph(graph);
  auto case3_data1_op = CreateOpDesc("case3_data1", DATA, tensor, 1, tensor, 1);
  auto case3_data2_op = CreateOpDesc("case3_data2", DATA, tensor, 1, tensor, 1);
  NodePtr case3_data1_node = case3_graph->AddNode(case3_data1_op);
  NodePtr case3_data2_node = case3_graph->AddNode(case3_data2_op);
  ge::AttrUtils::SetInt(case3_data1_node->GetOpDesc(), "_parent_node_index", 1);
  ge::AttrUtils::SetInt(case3_data2_node->GetOpDesc(), "_parent_node_index", 2);
  case3_graph->AddInputNode(case3_data1_node);
  case3_graph->AddInputNode(case3_data2_node);
  case_node->GetOpDesc()->AddSubgraphName(case3_name);
  case_node->GetOpDesc()->SetSubgraphInstanceName(2, case3_name);

  (void) graph->AddSubGraph(case1_graph);
  (void) graph->AddSubGraph(case2_graph);
  (void) graph->AddSubGraph(case3_graph);
  CondRemovePass cond_pass;
  EXPECT_EQ(cond_pass.Run(case_node), SUCCESS);
}

template <typename T>
GeTensorPtr ConstructTensorPtr(DataType dtype) {
  vector<T> data_vec = {0};
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, dtype);
  GeTensorPtr value_tensor =
    std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(T));
  return value_tensor;
}

TEST_F(UtestCondRemovePass, test_get_idx_succ) {
  CondRemovePass pass;
  int32_t ret = 0;
  GeTensorPtr value_tensor;
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, -1);
  
  value_tensor = ConstructTensorPtr<int64_t>(DT_INT64);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  value_tensor = ConstructTensorPtr<uint32_t>(DT_UINT32);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  value_tensor = ConstructTensorPtr<int16_t>(DT_INT16);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  value_tensor = ConstructTensorPtr<int8_t>(DT_INT8);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  value_tensor = ConstructTensorPtr<double>(DT_DOUBLE);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  value_tensor = ConstructTensorPtr<float>(DT_FLOAT);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  value_tensor = ConstructTensorPtr<float>(DT_DUAL);
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  vector<int32_t> datas = {0};
  GeTensorDesc tensor_desc(GeShape({1}), ge::FORMAT_NCHW, DT_BOOL);
  value_tensor = std::make_shared<GeTensor>(tensor_desc, (uint8_t *)datas.data(), datas.size() * sizeof(int32_t));
  ret = pass.GetCondIndex(value_tensor.get());
  EXPECT_EQ(ret, 0);
  
  GeTensorDesc str_tensor_desc(GeShape({1}), ge::FORMAT_NCHW, DT_STRING);
  value_tensor = std::make_shared<GeTensor>(str_tensor_desc, (uint8_t *)datas.data(), datas.size() * sizeof(int32_t));
  ret = pass.GetCondIndex(value_tensor.get());
}

void BuildAndCheckConstScalerIfGraphWithDataType(DataType dtype, void *value, bool should_take) {
  GeTensorDesc desc(GeShape(), FORMAT_NCHW, dtype);
  auto tensor = std::make_shared<GeTensor>(desc, static_cast<uint8_t *>(value), GetSizeByDataType(dtype));
  auto const_0 = OP_CFG(CONSTANT).TensorDesc(desc.GetFormat(), dtype, {}).OutCnt(1)
                              .Weight(tensor)
                              .Build("const_0");
  DEF_GRAPH(taken) {
    CHAIN(NODE("data_0", DATA)->NODE("output_0", NETOUTPUT));
  };
  DEF_GRAPH(not_taken) {
    CHAIN(NODE("data_1", DATA)->NODE("output_1", NETOUTPUT));
  };
  DEF_GRAPH(g) {
    CHAIN(NODE(const_0)->EDGE(0, 0)->NODE("if", IF, taken, not_taken));
  };

  CondRemovePass pass;
  auto graph = ToComputeGraph(g);
  auto node = graph->FindNode("if");
  EXPECT_NE(node, nullptr);
  EXPECT_EQ(pass.Run(node), SUCCESS);

  EXPECT_EQ(graph->FindNode("if"), nullptr);
  EXPECT_EQ(graph->GetSubgraph("taken") != nullptr, should_take);
  EXPECT_EQ(graph->GetSubgraph("not_taken") != nullptr, !should_take);
}

TEST_F(UtestCondRemovePass, if_cond_with_scaler_data_types) {
  uint8_t non_zero = 1;
  uint8_t zero = 0;

  BuildAndCheckConstScalerIfGraphWithDataType(DT_HIFLOAT8, &non_zero, true);
  BuildAndCheckConstScalerIfGraphWithDataType(DT_HIFLOAT8, &zero, false);

  BuildAndCheckConstScalerIfGraphWithDataType(DT_FLOAT8_E5M2, &non_zero, true);
  BuildAndCheckConstScalerIfGraphWithDataType(DT_FLOAT8_E5M2, &zero, false);

  BuildAndCheckConstScalerIfGraphWithDataType(DT_FLOAT8_E4M3FN, &non_zero, true);
  BuildAndCheckConstScalerIfGraphWithDataType(DT_FLOAT8_E4M3FN, &zero, false);
}
