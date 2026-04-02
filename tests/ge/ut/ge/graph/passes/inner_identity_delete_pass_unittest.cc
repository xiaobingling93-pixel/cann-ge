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
#include "graph/passes/feature/inner_identity_delete_pass.h"
#include "depends/op_stub/op_proto/inc/elewise_calculation_ops.h"
#include "graph/ge_local_context.h"
#include "common/ge_common/ge_types.h"

using namespace ge;

class UtestGraphPassesInnerIdentityDeletePass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

// 内置Identity的输入节点单引用，Identity也是单引用
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(identity)->NODE(assign)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
}

// 内置Identity的输入节点单引用，Identity多引用
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete2) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(identity)->NODE(assign)->CTRL_EDGE()->NODE("relu2", RELU));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE(identity)->EDGE(0, 0)->NODE("relu2"));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
  auto relu2 = compute_graph->FindNode("relu2");
  EXPECT_EQ(relu2->GetInDataNodes().at(0)->GetName(), "relu1");
}

// 内置Identity的输入节点A多引用，Identity单引用，A除Identity外其他输出节点都指向ref
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete3) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(identity)->NODE(assign));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("relu1")->NODE("relu2", RELU)->CTRL_EDGE()->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
  auto assign = compute_graph->FindNode("assign");
  EXPECT_EQ(assign->GetInDataNodes().at(0)->GetName(), "relu1");
}

// 内置Identity的输入节点A多引用，Identity单引用，稳定拓扑排序
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete4) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(identity)->NODE(assign));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("relu1")->NODE("relu2", RELU));
  };
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting(TopoSortingMode::kStableRDFS);
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
  auto assign = compute_graph->FindNode("assign");
  EXPECT_EQ(assign->GetInDataNodes().at(0)->GetName(), "relu1");

  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

// 内置Identity的输入节点单引用，Identity多引用,其中还连给另外一个内置Identity
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete5) {
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
    auto identity1 = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity1");
    auto identity2 = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity2");
    CHAIN(NODE("relu", RELU)->NODE(identity1)->NODE(assign1)->CTRL_EDGE()->NODE(assign2));
    CHAIN(NODE(identity1)->EDGE(0, 0)->NODE(assign2));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
  auto assign1 = compute_graph->FindNode("assign1");
  EXPECT_EQ(assign1->GetInDataNodes().at(0)->GetName(), "relu");
  auto assign2 = compute_graph->FindNode("assign2");
  EXPECT_EQ(assign2->GetInDataNodes().at(0)->GetName(), "relu");
}

// 内置Identity的输入节点A多引用，Identity多引用，Identity其他所有输出节点受ref控制，A节点除Identity之外的其他所有输出节点都指向了ref
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete6) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("relu1", RELU)->NODE(identity)->NODE(assign)->CTRL_EDGE()->NODE("relu2", RELU));
    CHAIN(NODE(identity)->EDGE(0, 0)->NODE("relu2"));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("relu3", RELU)->EDGE(0, 1)->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
  auto assign = compute_graph->FindNode("assign");
  auto relu1 = compute_graph->FindNode("relu1");
  auto relu2 = compute_graph->FindNode("relu2");
  EXPECT_EQ(assign->GetInDataNodes().at(0), relu1);
  EXPECT_EQ(relu2->GetInDataNodes().at(0), relu1);
}

TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityDelete7) {
  DEF_GRAPH(g1) {
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(identity)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_EQ(identity, nullptr);
}

/**
 *        data1      data2
 *          |           |
 *         relu1        |
 *        /    \        |
 *       /      \       |
 * identity  relu2    |
 *       |             |
 *     assign <--------+
 *
 * 预期行为：
 * - 内置Identity的输入节点(relu1)多引用，Identity单引用
 * - relu1的另一输出relu2不受ref算子(assign)控制，因此不能删除identity
 * - 保留identity
 */
TEST_F(UtestGraphPassesInnerIdentityDeletePass, InnerIdentityCannotDelete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Attr(ATTR_NAME_REFERENCE, true)
                      .Build("assign");
    auto identity = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE(identity)->NODE(assign));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("relu2", RELU));
  };
  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindFirstNodeMatchType(IDENTITY);
  EXPECT_NE(identity, nullptr);
}

// Data(ioa) -> (ioa)transdata(workspace) -> (workspace)identity(ioa) -> (ioa)ref
TEST_F(UtestGraphPassesInnerIdentityDeletePass, NanoCannotDelete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
        .InCnt(2)
        .OutCnt(1)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Attr(ATTR_NAME_REFERENCE, true)
        .Build("assign");
    auto identity = OP_CFG(IDENTITY)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_inner_identity", true)
        .OutputAttr(0, ATTR_NAME_TENSOR_MEMORY_SCOPE, 2)
        .Build("identity");
    CHAIN(NODE("data1", DATA)->NODE("transdata", TRANSDATA)->NODE(identity)->NODE(assign)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity = compute_graph->FindNode("identity");
  EXPECT_NE(identity, nullptr);
}

TEST_F(UtestGraphPassesInnerIdentityDeletePass, NanoCannotDelete2) {
  DEF_GRAPH(g1) {
    auto test_op = OP_CFG("TESTOP")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
        .InCnt(2)
        .OutCnt(2)
        .InNames({"ref1", "ref2"})
        .OutNames({"ref1", "ref2"})
        .Attr(ATTR_NAME_REFERENCE, true)
        .Build("test_op");
    auto identity1 = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity1");
    auto identity2 = OP_CFG(IDENTITY).InCnt(1).OutCnt(1).Attr("_inner_identity", true).Build("identity2");
    auto transdata1 = OP_CFG(TRANSDATA).InCnt(1).OutCnt(1).OutputAttr(0, ATTR_NAME_TENSOR_MEMORY_SCOPE, 2).
        Build("transdata1");
    auto transdata2 = OP_CFG(TRANSDATA).InCnt(1).OutCnt(1).OutputAttr(0, ATTR_NAME_TENSOR_MEMORY_SCOPE, 2).
        Build("transdata2");
    CHAIN(NODE("data1", DATA)->NODE(identity1)->NODE(transdata1)->NODE(test_op)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data1")->NODE(identity2)->NODE(transdata2)->EDGE(0, 1)->NODE(test_op));
  };
  auto compute_graph = ToComputeGraph(g1);
  InnerIdentityDeletePass pass;
  ASSERT_EQ(pass.Run(compute_graph), SUCCESS);
  auto identity_1 = compute_graph->FindNode("identity1");
  EXPECT_NE(identity_1, nullptr);
  auto identity_2 = compute_graph->FindNode("identity2");
  EXPECT_NE(identity_2, nullptr);
}