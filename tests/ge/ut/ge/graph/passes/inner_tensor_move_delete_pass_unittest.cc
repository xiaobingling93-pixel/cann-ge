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
#include "graph/passes/feature/inner_tensor_move_delete_pass.h"
#include "depends/op_stub/op_proto/inc/elewise_calculation_ops.h"
#include "graph/ge_local_context.h"
#include "common/ge_common/ge_types.h"

using namespace ge;

class UtestGraphPassesInnerTensorMoveDeletePass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

// 内置TensorMove的输入节点单引用，TensorMove也是单引用
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(tensormove)->NODE(assign)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
}

// 内置TensorMove的输入节点单引用，TensorMove多引用
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete2) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(tensormove)->NODE(assign)->CTRL_EDGE()->NODE("relu2", RELU));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE(tensormove)->EDGE(0, 0)->NODE("relu2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
  auto relu2 = compute_graph->FindNode("relu2");
  EXPECT_EQ(relu2->GetInDataNodes().at(0)->GetName(), "relu1");
}

// 内置TensorMove的输入节点A多引用，TensorMove单引用，A除TensorMove外其他输出节点都指向ref
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete3) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(tensormove)->NODE(assign));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("relu1")->NODE("relu2", RELU)->CTRL_EDGE()->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
  auto assign = compute_graph->FindNode("assign");
  EXPECT_EQ(assign->GetInDataNodes().at(0)->GetName(), "relu1");
}

// 内置TensorMove的输入节点A多引用，TensorMove单引用，稳定拓扑排序
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete4) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(tensormove)->NODE(assign));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("relu1")->NODE("relu2", RELU));
  };
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting(TopoSortingMode::kStableRDFS);
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
  auto assign = compute_graph->FindNode("assign");
  EXPECT_EQ(assign->GetInDataNodes().at(0)->GetName(), "relu1");

  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

// 内置TensorMove的输入节点单引用，TensorMove多引用,其中还连给另外一个内置TensorMove
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete5) {
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
    auto tensormove1 = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove1");
    auto tensormove2 = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove2");
    CHAIN(NODE("relu", RELU)->NODE(tensormove1)->NODE(assign1)->CTRL_EDGE()->NODE(assign2));
    CHAIN(NODE(tensormove1)->EDGE(0, 0)->NODE(assign2));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
  auto assign1 = compute_graph->FindNode("assign1");
  EXPECT_EQ(assign1->GetInDataNodes().at(0)->GetName(), "relu");
  auto assign2 = compute_graph->FindNode("assign2");
  EXPECT_EQ(assign2->GetInDataNodes().at(0)->GetName(), "relu");
}

// 内置TensorMove的输入节点A多引用，TensorMove多引用，TensorMove其他所有输出节点受ref控制，A节点除TensorMove之外的其他所有输出节点都指向了ref
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete6) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("relu1", RELU)->NODE(tensormove)->NODE(assign)->CTRL_EDGE()->NODE("relu2", RELU));
    CHAIN(NODE(tensormove)->EDGE(0, 0)->NODE("relu2"));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("relu3", RELU)->EDGE(0, 1)->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
  auto assign = compute_graph->FindNode("assign");
  auto relu1 = compute_graph->FindNode("relu1");
  auto relu2 = compute_graph->FindNode("relu2");
  EXPECT_EQ(assign->GetInDataNodes().at(0), relu1);
  EXPECT_EQ(relu2->GetInDataNodes().at(0), relu1);
}

TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveDelete7) {
  DEF_GRAPH(g1) {
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(tensormove)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_EQ(tensor_move, nullptr);
}

/**
 *        data1      data2
 *          |           |
 *         relu1        |
 *        /    \        |
 *       /      \       |
 * tensormove  relu2    |
 *       |             |
 *     assign <--------+
 *
 * 预期行为：
 * - 内置TensorMove的输入节点(relu1)多引用，TensorMove单引用
 * - relu1的另一输出relu2不受ref算子(assign)控制，因此不能删除tensormove
 * - 保留tensormove
 */
TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveCannotDelete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto tensormove = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(tensormove)->NODE(assign));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("relu2", RELU));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move = compute_graph->FindFirstNodeMatchType(TENSORMOVE);
  EXPECT_NE(tensor_move, nullptr);
}

TEST_F(UtestGraphPassesInnerTensorMoveDeletePass, InnerTensorMoveCannotDelete2) {
  DEF_GRAPH(g1) {
    auto test_op = OP_CFG("TESTOP")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
        .InCnt(2)
        .OutCnt(2)
        .InNames({"ref1", "ref2"})
        .OutNames({"ref1", "ref2"})
        .Attr(ATTR_NAME_REFERENCE, true)
        .Build("test_op");
    auto tensormove1 = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove1");
    auto tensormove2 = OP_CFG(TENSORMOVE).InCnt(1).OutCnt(1).Attr("_inner_tensor_move", true).Build("tensormove2");
    auto transdata1 = OP_CFG(TRANSDATA).InCnt(1).OutCnt(1).OutputAttr(0, ATTR_NAME_TENSOR_MEMORY_SCOPE, 2).
        Build("transdata1");
    auto transdata2 = OP_CFG(TRANSDATA).InCnt(1).OutCnt(1).OutputAttr(0, ATTR_NAME_TENSOR_MEMORY_SCOPE, 2).
        Build("transdata2");
    CHAIN(NODE("data1", DATA)->NODE(tensormove1)->NODE(transdata1)->NODE(test_op)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data1")->NODE(tensormove2)->NODE(transdata2)->EDGE(0, 1)->NODE(test_op));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerTensorMoveDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto tensor_move_1 = compute_graph->FindNode("tensormove1");
  EXPECT_NE(tensor_move_1, nullptr);
  auto tensor_move_2 = compute_graph->FindNode("tensormove2");
  EXPECT_NE(tensor_move_2, nullptr);
}