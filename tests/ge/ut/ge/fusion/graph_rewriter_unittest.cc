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
#include <gmock/gmock.h>
#include "common/share_graph.h"
#include "es_ge_test_ops_c.h"
#include "es_ge_test_ops.h"
#include "stub/gert_runtime_stub.h"

#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/constant_utils.h"
#include "common/topo_checker.h"
#include "graph/debug/ge_attr_define.h"

#include "ge/fusion/graph_rewriter.h"

namespace ge {
namespace fusion{
namespace {
using AnchorDesc = std::pair<NodePtr, int64_t>;
SubgraphBoundary BuildBoundary(const std::vector<std::vector<AnchorDesc>> &inputs,
                               const std::vector<AnchorDesc> &outputs) {
  SubgraphBoundary boundary;
  size_t i = 0u;
  for (const auto &in_anchors : inputs) {
    SubgraphInput subgraph_input;
    for (const auto &in_anchor : in_anchors) {
      subgraph_input.AddInput({NodeAdapter::Node2GNode(in_anchor.first), in_anchor.second});
    }
    boundary.AddInput(i, std::move(subgraph_input));
    ++i;
  }
  i = 0u;
  for (const auto &out_anchor : outputs) {
    SubgraphOutput subgraph_output;
    subgraph_output.SetOutput({NodeAdapter::Node2GNode(out_anchor.first), out_anchor.second});
    boundary.AddOutput(i, std::move(subgraph_output));
    ++i;
  }
  return boundary;
}
} // namespace
class UtestMatchReplacer : public testing::Test {
 public:
  static void SetUpTestSuite() {
  }
  static void TearDownTestSuite() {
  }
};

/**
   target graph:            boundary not include exp

        data
         |                    |
       abs1                  abs1
       /  \                   |
    exp   relu               relu
      \   /                   |
      add
       |
     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, InvalidBoundary_NotSelfContained) {
  ComputeGraphPtr target_compute_graph =  gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // build boundary
  NodePtr abs, relu;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Abs") {
      abs = node;
    }
    if (node->GetType() == "Relu") {
      relu = node;
    }

  }
  SubgraphBoundary subgraph = BuildBoundary({{{abs, 0}}}, {{relu, 0}});
  gert::GertRuntimeStub runtime_stub;
  EXPECT_NE(SubgraphRewriter::Replace(subgraph, *replace_graph), SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "Boundary is not self contained")>= 0);
}

/**
   target graph:            boundary not include exp

    data                   add
     |                     |
    abs1                  netoutput
   /  \
 exp   relu
   \   /
    add
    |
  netoutput

*/
TEST_F(UtestMatchReplacer, InvalidBoundary_HasMultiOwnerGraph) {
  ComputeGraphPtr target_compute_graph =  gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  ComputeGraphPtr target_compute_graph1 =  gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph1 = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // build boundary
  // build boundary
  NodePtr abs, relu, exp;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Abs") {
      abs = node;
    }
    if (node->GetType() == "Exp") {
      exp = node;
    }
  }
  for (const auto &node : target_compute_graph1->GetDirectNode()) {
    if (node->GetType() == "Relu") {
      relu = node;
    }
  }
  SubgraphBoundary subgraph = BuildBoundary({{{abs, 0}}}, {{relu, 0}, {exp, 0}});

  gert::GertRuntimeStub runtime_stub;
  EXPECT_NE(SubgraphRewriter::Replace(subgraph, *replace_graph), SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "as output node of boundary, is not in same graph with others")>= 0);
}

/**
 *  1 to 1
 * single node match
 *      data
 *        |
 *     transdata     => relu
 *        |
 *       out
 */
TEST_F(UtestMatchReplacer, SingleNode_1Input_1Output_Replace) {
  ComputeGraphPtr target_compute_graph =  gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "tests");

  // build replacement graph
  auto replace_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = replace_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto relu = EsRelu(data);
  esb_graph->SetGraphOutput(relu, 0);
  auto graph = replace_graph.BuildAndReset();
  auto replace_compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());

  // build boundary
  std::vector<SubgraphBoundary> subgraphs;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() != TRANSDATA) {
      continue;
    }
    // set stream label on transdata
    AttrUtils::SetStr(node->GetOpDesc(), public_attr::USER_STREAM_LABEL, "test_label");
    auto boundary = BuildBoundary({{{node, 0}}}, {{node,0}});
    subgraphs.emplace_back(std::move(boundary));
  }
  for (const auto &boundary : subgraphs) {
    EXPECT_EQ(SubgraphRewriter::Replace(boundary, *graph), SUCCESS);
  }
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");

  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {"Relu"}, {"Relu"}, {CONSTANT}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(1, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(2, {{"Relu"}}), "success");
    }
    if (node->GetName() == "x_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
    }
    if (node->GetName() == "y_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}}), "success");
    }
    if (node->GetType() == "Relu") {
      std::cout << node->GetName() << std::endl;
      // check attr
      std::vector<std::string> origin_types;
      const bool has_origin_op_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_types);
      EXPECT_TRUE(has_origin_op_attr);
      EXPECT_TRUE(origin_types.size() == 1);
      EXPECT_STREQ(origin_types[0].c_str(), TRANSDATA);

      std::string user_stream_label;
      const bool has_inherited_attr =
          ge::AttrUtils::GetStr(node->GetOpDesc(), public_attr::USER_STREAM_LABEL, user_stream_label);
      EXPECT_TRUE(has_inherited_attr);
      EXPECT_STREQ(user_stream_label.c_str(), "test_label");
    }
  }
}

/**
 *      data    const           data   const
 *        |    /                  |     /
 *     reshape                    add
 *        |             =>         |
 *     transdata                 relu
 *       |                         |
 *      out                       out
 */
TEST_F(UtestMatchReplacer, TwoNode_2Input_1Output_WithConst_Replace) {
  ComputeGraphPtr target_compute_graph =  gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build replacement
  std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
  std::vector<int64_t> x_reshape_shape({3});
  auto replace_graph_builder = es::EsGraphBuilder("replace");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto add_const = EsCreateConstInt64(replace_esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
  auto add_tensor = EsAdd(EsCreateGraphInput(replace_esb_graph, 0), add_const);
  replace_esb_graph->SetGraphOutput(EsRelu(add_tensor), 0);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // build boundary
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "LSTM");
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "x_reshape") {
      continue;
    }
    auto transdata = *node->GetOutDataNodes().begin();
    EXPECT_EQ(transdata->GetType(), TRANSDATA);
    boundary = BuildBoundary({{{node, 0}, {node, 1}}}, {{transdata, 0}});
  }

  // replace
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_add");
  EXPECT_EQ(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_add");

  // check replace result
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {TRANSDATA}, {TRANSDATA}, {CONSTANT}}),
                "success");
    }
    if (node->GetType() == "Relu") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Add"}}),
                "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"DynamicRNNV3"}}), "success");
    }
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{DATA}, {CONSTANT}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
      auto new_const = node->GetInDataNodes().at(1);
      EXPECT_STREQ(new_const->GetType().c_str(), CONSTANT);

      // check value
      ConstGeTensorPtr weight;
      EXPECT_TRUE(ConstantUtils::GetWeight(new_const->GetOpDesc(), 0, weight));
      int64_t *out_data = (int64_t *)weight->GetData().data();
      vector<int64_t> shape_in_const;
      for (size_t i = 0; i < x_reshape_const_data.size(); ++i) {
        shape_in_const.emplace_back(out_data[i]);
      }
      EXPECT_EQ(shape_in_const, x_reshape_const_data);
    }
  }
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *    data                data                 data
 *     |                   |                   /   \
 *    abs1                abs1              relu   abs
 *   /   \               /  \                |     |
 *  exp  relu          exp   relu            netoutput
 *                       \   /
 *                       add
 *                        |
 *                     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, InputMultiConsumer_1Input_2Output_Replace) {
  ComputeGraphPtr target_compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "abs1") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  auto abs_r = EsAbs(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(abs_r, 1);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // replace
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  EXPECT_EQ(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() == "data1") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}, {"Abs"}}), "success");
    }
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"},{"Abs"}}),
                "success");
    }
  }
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *    data                   data                      data
 *     |                   /     \                     /   \
 *    abs1                abs1   cast               relu   abs
 *   /   \               /  \      |                 |      |
 *  exp  relu          exp   relu  |                netoutput
 *                       \   /     |
 *                       add      |
 *                        |     /
 *                     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, ProducerHasConsumerOutOfBoundary) {
  // build target graph
  auto target_graph_builder = es::EsGraphBuilder("target");
  auto target_esb_graph = target_graph_builder.GetCGraphBuilder();
  auto data = EsCreateGraphInput(target_esb_graph, 0);
  auto abs = EsAbs(data);
  auto exp = EsExp(abs,0,0,0);
  auto relu = EsRelu(abs);
  auto add = EsAdd(exp, relu);
  auto cast = EsCast(data, DT_INT64);
  target_esb_graph->SetGraphOutput(add, 0);
  target_esb_graph->SetGraphOutput(cast, 1);
  auto target_graph = target_graph_builder.BuildAndReset();
  auto target_compute_graph = GraphUtilsEx::GetComputeGraph(*target_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "Abs_0") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  auto abs_r = EsAbs(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(abs_r, 1);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // replace
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  EXPECT_EQ(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() == "data1") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}, {"Abs"}, {"Cast"}}), "success");
    }
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"},{"Abs"}}),
                "success");
    }
  }
}

/**
 * pattern graph:    target graph:          replace graph:            expect:
 *
 *    data                   data                      data                     data
 *     |                   /     \                     /   \                  /  \   \
 *    abs1                abs1   cast                 |   abs                |   abs  cast
 *   /   \               /  \      |                   \  /                   \   /    \
 *  exp  relu          exp   relu  |                  netoutput                add     |
 *                       \   /     |                                            \     /
 *                       add      |                                             netoutput
 *                        |     /
 *                     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, ReplaceDataConnectToNetoutput) {
  // build target graph
  auto target_graph_builder = es::EsGraphBuilder("target");
  auto target_esb_graph = target_graph_builder.GetCGraphBuilder();
  auto data = EsCreateGraphInput(target_esb_graph, 0);
  auto abs = EsAbs(data);
  auto exp = EsExp(abs,0,0,0);
  auto relu = EsRelu(abs);
  auto add = EsAdd(exp, relu);
  auto cast = EsCast(data, DT_INT64);
  target_esb_graph->SetGraphOutput(add, 0);
  target_esb_graph->SetGraphOutput(cast, 1);
  auto target_graph = target_graph_builder.BuildAndReset();
  auto target_compute_graph = GraphUtilsEx::GetComputeGraph(*target_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "Abs_0") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto abs_r = EsAbs(data_replace);
  replace_esb_graph->SetGraphOutput(data_replace, 0);
  replace_esb_graph->SetGraphOutput(abs_r, 1);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // replace
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  EXPECT_EQ(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() == "data1") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}, {"Abs"}, {"Cast"}}), "success");
    }
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Data"},{"Abs"}}),
                "success");
    }
  }
}

/**
 * N to 0
  * pattern graph:    target graph:          replace graph:
  *
  *    data                data                 data
  *     |                   |                   /   \
  *    abs1                abs1               netoutput
  *   /   \               /  \
  *  exp  relu          exp   relu
  *                       \   /
  *                       add
  *                        |
  *                     netoutput
 */
TEST_F(UtestMatchReplacer, delete_nodes_in_boundary) {
  ComputeGraphPtr target_compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "abs1") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  replace_esb_graph->SetGraphOutput(data_replace, 0);
  replace_esb_graph->SetGraphOutput(data_replace, 1);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  GraphUtils::DumpGEGraphToOnnx(*GraphUtilsEx::GetComputeGraph(*replace_graph), "replace");
  // replace
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "before_replace_abs");
  EXPECT_EQ(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace_abs");

  // check replace result
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Add") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Data"},{"Data"}}),
                "success");
    }
  }
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *    data                data                   data
 *     |                   |                   /     \
 *    abs1                abs1              relu <->  abs
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, InvalidReplacement_has_cycle) {
  ComputeGraphPtr target_compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "abs1") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  auto abs_r = EsAbs(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(abs_r, 1);
  auto replace_graph = replace_graph_builder.BuildAndReset();
  GraphUtils::AddEdge(NodeAdapter::GNode2Node(relu_r->GetProducer())->GetOutControlAnchor(), NodeAdapter::GNode2Node(abs_r->GetProducer())->GetInControlAnchor());
  GraphUtils::AddEdge(NodeAdapter::GNode2Node(abs_r->GetProducer())->GetOutControlAnchor(), NodeAdapter::GNode2Node(relu_r->GetProducer())->GetInControlAnchor());

  // replace
  gert::GertRuntimeStub runtime_stub;
  EXPECT_NE(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "There may exist cycle on replacement graph")>= 0);
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *    data                data            data  data
 *     |                   |                 \     \
 *    abs1                abs1              relu   abs
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, InvalidReplacement_InputSizeNotMatch) {
  ComputeGraphPtr target_compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "abs1") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto data_replace1 = EsCreateGraphInput(replace_esb_graph, 1);
  auto relu_r = EsRelu(data_replace);
  auto abs_r = EsAbs(data_replace1);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  replace_esb_graph->SetGraphOutput(abs_r, 1);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // replace
  gert::GertRuntimeStub runtime_stub;
  EXPECT_NE(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "not equal with Boundary input size")>= 0);
}

/**
 * pattern graph:    target graph:          replace graph:
 *
 *    data                data               data
 *     |                   |                  |
 *    abs1                abs1               relu
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                     netoutput
 *
 *
 */
TEST_F(UtestMatchReplacer, InvalidReplacement_OutputSizeNotMatch) {
  ComputeGraphPtr target_compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build boundary
  SubgraphBoundary boundary;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetName() != "abs1") {
      continue;
    }
    auto out_nodes = node->GetOutDataNodes();
    EXPECT_EQ(out_nodes.size(), 2);
    boundary = BuildBoundary({{{node, 0}}}, {{out_nodes.at(0), 0}, {out_nodes.at(1), 0}});
  }

  // build replacement graph
  auto replace_graph_builder = es::EsGraphBuilder("repalce");
  auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
  auto data_replace = EsCreateGraphInput(replace_esb_graph, 0);
  auto relu_r = EsRelu(data_replace);
  replace_esb_graph->SetGraphOutput(relu_r, 0);
  auto replace_graph = replace_graph_builder.BuildAndReset();

  // replace
  gert::GertRuntimeStub runtime_stub;
  EXPECT_NE(SubgraphRewriter::Replace(boundary, *replace_graph), SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "not equal with Boundary output size")>= 0);
}
} // namespace fusion
} // namespace ge