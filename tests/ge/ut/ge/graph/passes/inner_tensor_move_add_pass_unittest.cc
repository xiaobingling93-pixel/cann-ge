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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/feature/inner_tensor_move_add_pass.h"
#include "depends/op_stub/op_proto/inc/elewise_calculation_ops.h"

using namespace ge;

namespace {
ComputeGraphPtr BuildTensormoveOnlyOutputToAssignGraph() {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->NODE("tensormove", TENSORMOVE)->NODE(assign)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA))->EDGE(0, 1)->NODE(assign);
  };
  return ToComputeGraph(g1);
}

ComputeGraphPtr BuildMultiRefFromVarGraph() {
  DEF_GRAPH(g1) {
    auto assign1 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign1");
    auto assign2 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign2");
    CHAIN(NODE("var", VARIABLE)->NODE(assign1)->NODE(assign2)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data1", DATA))->EDGE(0, 1)->NODE(assign1);
    CHAIN(NODE("data1", DATA))->EDGE(0, 1)->NODE(assign2);
  };
  return ToComputeGraph(g1);
}

ComputeGraphPtr BuildReluOnlyOutputToAssignGraph() {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(assign)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
  };
  return ToComputeGraph(g1);
}

ComputeGraphPtr BuildMultiRefOpGraph() {
  DEF_GRAPH(g1) {
    auto test_op = OP_CFG("TESTOP")
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(2)
                      .InNames({"ref1", "ref2"})
                      .OutNames({"ref1", "ref2"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("test_op");
    CHAIN(NODE("data1", DATA)->NODE(test_op)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data1")->EDGE(0, 1)->NODE(test_op));
  };
  return ToComputeGraph(g1);
}

ComputeGraphPtr BuildReluMultiOutputToAssignGraph1() {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(assign));
    CHAIN(NODE("relu")->EDGE(0, 0)->NODE("add", ADD)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA))->EDGE(0, 1)->NODE(assign)->EDGE(0, 1)->NODE("add");
  };
  return ToComputeGraph(g1);
}


ComputeGraphPtr BuildReluMultiOutputToAssignGraph2() {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    CHAIN(NODE("data", DATA)
              ->NODE("relu1", RELU)
              ->NODE(assign)
              ->EDGE(0, 0)
              ->NODE("add", ADD)
              ->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("relu1")->EDGE(0, 1)->NODE("add"));
    CHAIN(NODE("relu1")->NODE("relu2", RELU)->EDGE(0, 1)->NODE(assign));
  };
  return ToComputeGraph(g1);
}
ComputeGraphPtr BuildReluMultiOutputToAssignGraph3() {
  DEF_GRAPH(g1) {
    auto assign1 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign1");
    auto assign2 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign2");
    CHAIN(NODE("relu", RELU)->NODE(assign1)->CTRL_EDGE()->NODE(assign2));
    CHAIN(NODE("relu")->EDGE(0, 0)->NODE(assign2));
  };

  return ToComputeGraph(g1);
}
}
class UtestGraphPassesInnerTensorMoveAddPass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

/**
 *       Data1                      Data1
 *        |                           |
 *       Relu                        Relu
 *        |                           |
 *        |     Data2      ->     TensorMove     Data2
 *        \        /                  \            /
 *          Assign                       Assign
 *            |                            |
 *        NetOutput                     NetOutput
 *
 * relu单输出单引用
 */
TEST_F(UtestGraphPassesInnerTensorMoveAddPass, add_tensormove_success_1) {
  auto graph = BuildReluOnlyOutputToAssignGraph();
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
  ASSERT_NE(tensor_move, nullptr);
  bool is_inner_tensor_move = false;
  EXPECT_EQ(AttrUtils::GetBool(tensor_move->GetOpDesc(), "_inner_tensor_move", is_inner_tensor_move), true);
  EXPECT_EQ(is_inner_tensor_move, true);
  auto relu = graph->FindNode("relu");
  ASSERT_NE(relu, nullptr);
  EXPECT_EQ(relu->GetOutDataNodes().at(0)->GetName(), tensor_move->GetName());
  auto assign = graph->FindNode("assign");
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->GetInDataNodes().at(0), tensor_move);
}

/**
 *              Data1                      Data1
 *               |                           |
 *              Relu                        Relu
 *               |                           |
 *               |      Data2      ->     TensorMove     Data2
 *               | \     /                   | \         /
 *               |  Assign                   |    Assign
 *               \    |                       \     /
 *                 Add                          Add
 *                  |                            |
 *              NetOutput                    NetOutput
 *
 * relu单输出多引用，且relu的另一个输出节点依赖ref算子
 */
TEST_F(UtestGraphPassesInnerTensorMoveAddPass, add_tensormove_success_2) {
  auto graph = BuildReluMultiOutputToAssignGraph1();
  graph->TopologicalSorting();
  auto relu = graph->FindNode("relu");
  ASSERT_NE(relu, nullptr);
  ASSERT_EQ(relu->GetOutDataNodesSize(), 2U);
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
  ASSERT_NE(tensor_move, nullptr);
  EXPECT_EQ(relu->GetOutDataNodes().at(0), tensor_move);
  auto assign = graph->FindNode("assign");
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->GetInDataNodes().at(0), tensor_move);
  auto add = graph->FindNode("add");
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(add->GetInDataNodes().at(0), tensor_move);
}

/*
 *                    data                   data
 *                     |                       |
 *                    relu1                  relu1
 *                  /  |   |              /    |
 *                 / relu2 |   -> TensorMove  relu2
 *                /  /     |           |   \ /
 *            assign       |         assign   |
 *                   \    /              \    |
 *                     add                 add
 **/
TEST_F(UtestGraphPassesInnerTensorMoveAddPass, add_tensormove_success_3) {
  auto graph = BuildReluMultiOutputToAssignGraph2();
  graph->TopologicalSorting();
  auto relu1 = graph->FindNode("relu1");
  ASSERT_NE(relu1, nullptr);
  ASSERT_EQ(relu1->GetOutDataNodesSize(), 3U);
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  ASSERT_EQ(relu1->GetOutDataNodesSize(), 2U);
  auto tensor_move = graph->FindFirstNodeMatchType(TENSORMOVE);
  ASSERT_NE(tensor_move, nullptr);
  EXPECT_EQ(relu1->GetOutDataNodes().at(0), tensor_move);
  auto assign = graph->FindNode("assign");
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->GetInDataNodes().at(0), tensor_move);
  auto add = graph->FindNode("add");
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(add->GetInDataNodes().at(1), tensor_move);
}

TEST_F(UtestGraphPassesInnerTensorMoveAddPass, multi_ref_op_add_tensor_move_succ) {
  auto graph = BuildMultiRefOpGraph();
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  auto test_op = graph->FindNode("test_op");
  ASSERT_NE(test_op, nullptr);
  auto input_1 = test_op->GetInDataNodes().at(0);
  auto input_2 = test_op->GetInDataNodes().at(1);
  ASSERT_NE(input_1, nullptr);
  ASSERT_NE(input_2, nullptr);
  EXPECT_EQ(input_1->GetType(), TENSORMOVE);
  EXPECT_EQ(input_2->GetType(), TENSORMOVE);
  EXPECT_NE(input_1->GetName(), input_2->GetName());
}

/**
 *             relu                    relu
 *          /      \                  /
 *    assign1      |              TensorMove
 *            \   |                  |       \
 *              assign2            assign1   TensorMove
 *                                      \     |
 *                                        assign2
 **/
TEST_F(UtestGraphPassesInnerTensorMoveAddPass, add_tensormove_success_4) {
  auto graph = BuildReluMultiOutputToAssignGraph3();
  graph->TopologicalSorting();
  auto relu = graph->FindNode("relu");
  ASSERT_NE(relu, nullptr);
  ASSERT_EQ(relu->GetOutDataNodesSize(), 2U);
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  ASSERT_EQ(relu->GetOutDataNodesSize(), 1U);
  auto tensormove1 = relu->GetOutDataNodes().at(0);
  ASSERT_EQ(tensormove1->GetType(), TENSORMOVE);
  ASSERT_EQ(tensormove1->GetOutDataNodes().at(0)->GetName(), "assign1");
  auto assign2 = graph->FindNode("assign2");
  ASSERT_NE(assign2, nullptr);
  auto tensormove2 = assign2->GetInDataNodes().at(0);
  ASSERT_EQ(tensormove1->GetOutDataNodes().at(1), tensormove2);
}

TEST_F(UtestGraphPassesInnerTensorMoveAddPass, not_add_tensormove) {
  auto graph = BuildTensormoveOnlyOutputToAssignGraph();
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  size_t inner_tensor_move_count = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == TENSORMOVE) {
      bool is_inner_tensor_move = false;
      if (AttrUtils::GetBool(node->GetOpDesc(), "_inner_tensor_move", is_inner_tensor_move) && is_inner_tensor_move) {
        inner_tensor_move_count++;
      }
    }
  }
  ASSERT_EQ(inner_tensor_move_count, 0U);
}

TEST_F(UtestGraphPassesInnerTensorMoveAddPass, multi_ref_fromvar_not_add_tensormove) {
  auto graph = BuildMultiRefFromVarGraph();
  InnerTensorMoveAddPass pass;
  ASSERT_EQ(pass.Run(graph), SUCCESS);
  size_t inner_tensor_move_count = 0;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == TENSORMOVE) {
      bool is_inner_tensor_move = false;
      if (AttrUtils::GetBool(node->GetOpDesc(), "_inner_tensor_move", is_inner_tensor_move) && is_inner_tensor_move) {
        inner_tensor_move_count++;
      }
    }
  }
  ASSERT_EQ(inner_tensor_move_count, 0U);
}