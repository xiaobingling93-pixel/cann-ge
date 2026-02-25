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
#include "es_ge_test_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"

#include "stub/gert_runtime_stub.h"

#include "ge/fusion/pattern_matcher.h"
#include "ge/fusion/pattern.h"
namespace ge {
namespace fusion {
namespace {
void ExpectMatchedNodeEqualWith(const std::unique_ptr<MatchResult> &match, const NodePtr &p_node,
                                const std::string &expect_matched_node_name) {
  GNode matched_node;
  EXPECT_EQ(match->GetMatchedNode(NodeAdapter::Node2GNode(p_node), matched_node), SUCCESS);
  EXPECT_STREQ(NodeAdapter::GNode2Node(matched_node)->GetNamePtr(), expect_matched_node_name.c_str());
}

void ExpectMatchedBoundaryEqualWith(const SubgraphBoundary &actual_boundary, const SubgraphBoundary &expect_boundary) {
  std::vector<SubgraphInput> actual_inputs;
  std::vector<SubgraphInput> expect_inputs;
  EXPECT_EQ(actual_boundary.GetAllInputs(actual_inputs), SUCCESS);
  EXPECT_EQ(expect_boundary.GetAllInputs(expect_inputs), SUCCESS);
  EXPECT_EQ(actual_inputs.size(), expect_inputs.size());
  for (size_t i = 0U; i < actual_inputs.size(); ++i) {
    auto actual_node_inputs = actual_inputs[i].GetAllInputs();
    auto expect_node_inputs = expect_inputs[i].GetAllInputs();
    EXPECT_EQ(actual_node_inputs.size(), expect_node_inputs.size());
    for (size_t j = 0U; j < actual_node_inputs.size(); ++j) {
      EXPECT_EQ(NodeAdapter::GNode2Node(actual_node_inputs[j].node),
                NodeAdapter::GNode2Node(expect_node_inputs[j].node));
      EXPECT_EQ(actual_node_inputs[j].index, expect_node_inputs[j].index);
    }
  }

  std::vector<SubgraphOutput> actual_outputs;
  std::vector<SubgraphOutput> expect_outputs;
  EXPECT_EQ(actual_boundary.GetAllOutputs(actual_outputs), SUCCESS);
  EXPECT_EQ(expect_boundary.GetAllOutputs(expect_outputs), SUCCESS);
  EXPECT_EQ(actual_outputs.size(), expect_outputs.size());
  for (size_t i = 0U; i < actual_outputs.size(); ++i) {
    NodeIo actual_node_output;
    EXPECT_EQ(actual_outputs[i].GetOutput(actual_node_output), SUCCESS);
    NodeIo expect_node_output;
    EXPECT_EQ(expect_outputs[i].GetOutput(expect_node_output), SUCCESS);
    EXPECT_EQ(NodeAdapter::GNode2Node(actual_node_output.node),
              NodeAdapter::GNode2Node(expect_node_output.node));
    EXPECT_EQ(actual_node_output.index, expect_node_output.index);
  }
}

/**
*                          data3   data4
*                              \     /
*                     data2    add0
*                        \      /
*                          add
*                           |
*        data0  data1     cast1
*            \     \      /
*             addlayernorm
*              /   |  |  |
*           cast2  |  |  |
*              \   |  | /
*               netoutput
 * @param cast_dst_type
 * @return
 */
std::unique_ptr<Graph> BuildPartialOutputAnchorsAsPatternOutputGraph(DataType cast_dst_type) {
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data0 = EsCreateGraphInput(esb_graph, 0);
  auto data1 = EsCreateGraphInput(esb_graph, 1);
  auto data2 = EsCreateGraphInput(esb_graph, 2);
  auto data3 = EsCreateGraphInput(esb_graph, 3);
  auto data4 = EsCreateGraphInput(esb_graph, 4);
  auto add = EsAdd(data2, EsAdd(data3, data4));
  auto cast1 = EsCast(add, cast_dst_type);
  auto add_layer_norm = EsAddLayerNorm(data0, data1, cast1, EsCreateScalarInt32(esb_graph, 1), EsCreateScalarInt32(esb_graph, 1), 0, 0);
  auto cast2 = EsCast(add_layer_norm.x, cast_dst_type);
  esb_graph->SetGraphOutput(cast2, 0);
  esb_graph->SetGraphOutput(add_layer_norm.mean, 1);
  esb_graph->SetGraphOutput(add_layer_norm.rstd, 2);
  esb_graph->SetGraphOutput(add_layer_norm.x, 3);
  return pattern_graph.BuildAndReset();
}
}  // namespace
using namespace es;
class UtestPatternMatcher : public testing::Test {
 public:
  static void SetUpTestSuite() {
    target_graph0 = GraphUtilsEx::CreateGraphFromComputeGraph(gert::ShareGraph::LstmpGraph());
    target_graph1 = GraphUtilsEx::CreateGraphFromComputeGraph(gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph());
  }
  static void TearDownTestSuite() {}

 private:
  static std::unordered_map<std::string, EsCTensorHolder *> case_2_tensor_;
 protected:
  static Graph target_graph0;
  static Graph target_graph1;
};
/**

                                                                                            g1

                                                                   ┌──────────────┐     ┌─────────┐                                    ┌─────────────────┐
                                                                   │   c2894_4    │     │ c2894_5 │                                    │ y_reshape_const │
                                                                   └──────────────┘     └─────────┘                                    └─────────────────┘
                                                                     │                    │                                              │
                                                                     │ (0,2)              │ (0,5)                                        │ (0,1)
                                                                     ∨                    ∨                                              ∨
                                 ┌──────────────────────┐  (0,1)   ┌──────────────────────────────┐  (0,0)   ┌──────────────┐  (0,0)   ┌─────────────────┐  (0,0)   ┌───────────┐
                                 │       c2894_3        │ ───────> │                              │ ───────> │ transdata_17 │ ───────> │    y_reshape    │ ───────> │ NetOutput │ <┐
                                 └──────────────────────┘          │                              │          └──────────────┘          └─────────────────┘          └───────────┘  │
    (0,3)    │                              │                                                                   ∧            │
  ┌──────────────────────────────────────────────────────────────> │                              │                                                                   │            │
  │                                                                │                              │                                                                   │            │
  │                              ┌──────────────────────┐          │                              │  (2,0)   ┌──────────────┐  (0,2)                                  │            │
  │                              │   x_reshape_const    │          │            drnnv3            │ ───────> │ transdata_13 │ ────────────────────────────────────────┘            │ (0,1)
  │                              └──────────────────────┘          │                              │          └──────────────┘                                                      │
  │                                │                               │                              │                                                                                │
  │                                │ (0,1)                         │                              │                                                                                │
  │                                ∨                               │                              │                                                                                │
  │  ┌────────────────┐  (0,0)   ┌──────────────────────┐          │                              │  (1,0)   ┌──────────────┐                                                      │
  │  │ inputs_float32 │ ───────> │      x_reshape       │          │                              │ ───────> │ transdata_15 │ ─────────────────────────────────────────────────────┘
  │  └────────────────┘          └──────────────────────┘          └──────────────────────────────┘          └──────────────┘
  │                                │                                 ∧                    ∧
  │                                │ (0,0)                           │ (0,0)              │ (0,4)
  │                                ∨                                 │                    │
  │                              ┌──────────────────────┐            │                    │
  │                              │     transdata_4      │ ───────────┘                    │
  │                              └──────────────────────┘                                 │
  │                              ┌──────────────────────┐  (0,0)   ┌──────────────┐       │
  │                              │  cell_state_float32  │ ───────> │ transdata_10 │ ──────┘
  │                              └──────────────────────┘          └──────────────┘
  │
  └──────────────────────────────────────────────────────────────────┐
                                                                     │
                                 ┌──────────────────────┐  (0,0)   ┌──────────────┐
                                 │ hidden_state_float32 │ ───────> │ transdata_8  │
                                 └──────────────────────┘          └──────────────┘
 */
Graph UtestPatternMatcher::target_graph0;

/**

┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌─────┐  (0,0)   ┌─────┐  (0,0)   ┌───────────┐
│ data1 │ ───────> │  abs1  │ ───────> │ exp │ ───────> │ add │ ───────> │ NetOutput │
└───────┘          └────────┘          └─────┘          └─────┘          └───────────┘
                     │                                    ∧
                     │ (0,0)                              │
                     ∨                                    │
                   ┌────────┐  (0,1)                      │
                   │  relu  │ ────────────────────────────┘
                   └────────┘
 */
Graph UtestPatternMatcher::target_graph1;
/**
 * single node match
 *      data
 *        |
 *     transdata
 *        |
 *       out
 */
TEST_F(UtestPatternMatcher, SingleNode_1Input_1Output_Match) {
  auto target_compute_graph = GraphUtilsEx::GetComputeGraph(target_graph0);
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
  esb_graph->SetGraphOutput(transdata, 0);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;

  while ((match = matcher.MatchNext()), match != nullptr) {
    AscendString node_name;
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 6);
  std::vector<std::string> expect_match_node_name = {"transdata_10", "transdata_13", "transdata_15",
                                                    "transdata_17", "transdata_4",  "transdata_8"};
  std::vector<std::string> expect_match_node_input_name = {"cell_state_float32", "drnnv3", "drnnv3",
                                                    "drnnv3", "x_reshape",  "hidden_state_float32"};
  GNode transdata10;
  for (size_t i = 0u; i < match_ret.size(); ++i) {
    if (i == 0) {
      match_ret[i]->GetMatchedNode(NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(transdata->GetProducer())), transdata10);
    }
    ExpectMatchedNodeEqualWith(match_ret[i], NodeAdapter::GNode2Node(transdata->GetProducer()), expect_match_node_name[i]);
    std::cout << match_ret[i]->ToAscendString().GetString() << std::endl;
  }
  auto actual_boundary = match_ret[0]->ToSubgraphBoundary();
  EXPECT_NE(actual_boundary, nullptr);
  SubgraphBoundary expect_boundary;
  SubgraphInput expect_subgraph_input;
  expect_subgraph_input.AddInput({transdata10, 0});
  expect_boundary.AddInput(0, std::move(expect_subgraph_input));
  SubgraphOutput expect_subgraph_output;
  expect_subgraph_output.SetOutput({transdata10, 0});
  expect_boundary.AddOutput(0, std::move(expect_subgraph_output));
  ExpectMatchedBoundaryEqualWith(*actual_boundary, expect_boundary);
}

/**
 * single node match
 *      data    const
 *        |    /
 *     reshape
 *        |
 *       out
 */
TEST_F(UtestPatternMatcher, SingleNode_2Input_1Output_WithConst_Match) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);

  std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
  std::vector<int64_t> x_reshape_shape({3});

  auto shape_const = EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
  auto reshape = EsReshape(data, shape_const, 0, 0);
  esb_graph->SetGraphOutput(reshape, 0);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 2);
  std::vector<std::string> expect_match_node_name = {"x_reshape", "y_reshape"};
  for (size_t i = 0u; i < match_ret.size(); ++i) {
    ExpectMatchedNodeEqualWith(match_ret[i], NodeAdapter::GNode2Node(reshape->GetProducer()), expect_match_node_name[i]);
  }
}

/**
 * single node match
 *      data    const
 *        |    /
 *     reshape
 *        |
 *       out
 */
TEST_F(UtestPatternMatcher, SingleNode_2Input_1Output_WithConst_EnableValueMatch_ValueMiss) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);

  std::vector<int64_t> x_reshape_const_data({-1, 2, 256});
  std::vector<int64_t> x_reshape_shape({3});

  auto shape_const =
      EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
  auto reshape = EsReshape(data, shape_const, 0, 0);
  esb_graph->SetGraphOutput(reshape, 0);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  auto matcher_config = PatternMatcherConfigBuilder().EnableConstValueMatch().Build();
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0), std::move(matcher_config));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 0);
}


/**
 *      data    const
 *        |    /
 *     reshape
 *        |
 *     transdata
 *       |
 *      out
 */
TEST_F(UtestPatternMatcher, TwoNode_2Input_1Output_WithConst_Match) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);

  std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
  std::vector<int64_t> x_reshape_shape({3});
  auto shape_const = EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
  auto reshape = EsReshape(data, shape_const, 0, 0);
  auto transdata = EsTransData(reshape, "0", "29", 0, 0, 0);
  esb_graph->SetGraphOutput(transdata, 0);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  GraphUtils::DumpGEGraphToOnnx(*GraphUtilsEx::GetComputeGraph(target_graph0).get(), "LSTM");
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(reshape->GetProducer()), "x_reshape");
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(transdata->GetProducer()), "transdata_4");
}

/**
 * pattern graph:    target graph:
 *
 *    data                data
 *     |                   |
 *    abs1                abs1
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                     netoutput

 */
TEST_F(UtestPatternMatcher, 3Node_1Input_2Output_Match) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto abs1 = EsAbs(data);
  auto exp = EsExp(abs1, 0 , 0, 0);
  auto relu = EsRelu(abs1);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph1));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(abs1->GetProducer()), "abs1");
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(exp->GetProducer()), "exp");
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(relu->GetProducer()), "relu");
}

/**
 * pattern graph:    target graph:
 *
 *                       data
 *                        |
 *    data                abs1
 *   /   \               /  \
 *  exp  relu          exp   relu
 *                       \   /
 *                       add
 *                        |
 *                     netoutput

*/
TEST_F(UtestPatternMatcher, PatternInput_MultiConsumer_Match) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto exp = EsExp(data, 0 , 0, 0);
  auto relu = EsRelu(data);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph1));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(exp->GetProducer()), "exp");
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(relu->GetProducer()), "relu");

  std::vector<std::string> expected_matched_nodes({"exp", "relu"});
  auto matched_nodes = match_ret[0]->GetMatchedNodes();
  EXPECT_EQ(matched_nodes.size(), expected_matched_nodes.size());
  for (size_t i = 0; i < matched_nodes.size(); ++i) {
    AscendString node_name;
    matched_nodes[i].GetName(node_name);
    EXPECT_STREQ(node_name.GetString(), expected_matched_nodes[i].c_str());
  }
}
/**
 * pattern graph:    target graph:
 *
 *                       data
 *                       |   \
 *    data             abs1  abs2
 *   /   \              |      |
 *  exp  relu          exp   relu
 *                       \   /
 *                     netoutput
*/
// todo check later how can this ut work
TEST_F(UtestPatternMatcher, PatternInput_MultiConsumer_TargetNode_ProducerIsDiff_Miss) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto exp = EsExp(data, 0 , 0, 0);
  auto relu = EsRelu(data);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  // build target graph
  auto target_graph_builder = es::EsGraphBuilder("target_graph");
  auto target_esb_graph = target_graph_builder.GetCGraphBuilder();
  auto t_data = EsCreateGraphInput(target_esb_graph, 0);
  auto t_abs1 = EsAbs(t_data);
  auto t_abs2 = EsAbs(t_data);
  auto t_exp = EsExp(t_abs1, 0 , 0, 0);
  auto t_relu = EsRelu(t_abs2);
  target_esb_graph->SetGraphOutput(t_exp, 0);
  target_esb_graph->SetGraphOutput(t_relu, 1);
  auto target_graph = target_graph_builder.BuildAndReset();

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(*target_graph));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 0);
}

/**
 * pattern graph:    target graph:
 *
 *     data                data
 *     |   \                 |
 *    abs1  relu           abs1
 *    |                    /  \
 *   exp                exp   relu
 *                       \   /
 *                       add
 *                        |
 *                     netoutput

*/
TEST_F(UtestPatternMatcher, 3Node_1Input_2Output_Miss) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);
  auto abs1 = EsAbs(data);
  auto exp = EsExp(abs1, 0 , 0, 0);
  auto relu = EsRelu(data);
  esb_graph->SetGraphOutput(exp, 0);
  esb_graph->SetGraphOutput(relu, 1);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph1));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 0);
}

/**
 * pattern0                      pattern1
 *      data    const             data
 *        |    /                   \
 *     reshape                    transdata  data
 *        |                          \        /
 *     transdata                      reshape
 *       |                              \
 *      out                            out
 */
TEST_F(UtestPatternMatcher, 2Patterns_Match) {
  // build pattern graph0
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data = EsCreateGraphInput(esb_graph, 0);

  std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
  std::vector<int64_t> x_reshape_shape({3});
  auto shape_const = EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
  auto reshape = EsReshape(data, shape_const, 0, 0);
  auto transdata = EsTransData(reshape, "0", "29", 0, 0, 0);
  esb_graph->SetGraphOutput(transdata, 0);
  auto graph = pattern_graph.BuildAndReset();
  auto pattern = std::make_unique<Pattern>(std::move(*graph));

  // build pattern graph1
  auto pattern_graph1= es::EsGraphBuilder("pattern1");
  auto esb_graph1 = pattern_graph1.GetCGraphBuilder();
  auto transdata1 = EsTransData(EsCreateGraphInput(esb_graph1, 0), "0", "29", 0, 0, 0);
  auto reshape1 = EsReshape(transdata1, EsCreateGraphInput(esb_graph1, 1), 0, 0);
  esb_graph1->SetGraphOutput(reshape1, 0);
  auto graph1 = pattern_graph1.BuildAndReset();
  auto pattern1 = std::make_unique<Pattern>(std::move(*graph1));

  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  PatternMatcher matcher1(std::move(pattern1), std::make_shared<Graph>(target_graph0));
  // dlog_setlevel(GE_MODULE_NAME, 0, 0);
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }
  while ((match = matcher1.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 2);
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(reshape->GetProducer()), "x_reshape");
  ExpectMatchedNodeEqualWith(match_ret[0], NodeAdapter::GNode2Node(transdata->GetProducer()), "transdata_4");
  ExpectMatchedNodeEqualWith(match_ret[1], NodeAdapter::GNode2Node(reshape1->GetProducer()), "y_reshape");
  ExpectMatchedNodeEqualWith(match_ret[1], NodeAdapter::GNode2Node(transdata1->GetProducer()), "transdata_17");
}

TEST_F(UtestPatternMatcher, InvalidPattern_ContrlEdgeInPattern_Miss) {
  auto compute_graph = gert::ShareGraph::SimpleVariableAssignGraph("TMP");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto pattern = std::make_unique<Pattern>(std::move(graph));

  gert::GertRuntimeStub runtime_stub;
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  EXPECT_EQ(matcher.MatchNext(), nullptr);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "has control edge")>= 0);
}

TEST_F(UtestPatternMatcher, InvalidPattern_SubgraphInPattern_Miss) {
  auto compute_graph = gert::ShareGraph::IfGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto pattern = std::make_unique<Pattern>(std::move(graph));

  gert::GertRuntimeStub runtime_stub;
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  EXPECT_EQ(matcher.MatchNext(), nullptr);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "It has subgraph")>= 0);
}

TEST_F(UtestPatternMatcher, InvalidPattern_DynamicInputNodeInPattern_Miss) {
  auto compute_graph = gert::ShareGraph::ConcatV2ConstDependencyGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto pattern = std::make_unique<Pattern>(std::move(graph));

  gert::GertRuntimeStub runtime_stub;
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  EXPECT_EQ(matcher.MatchNext(), nullptr);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "is dynamic input")>= 0);
}

TEST_F(UtestPatternMatcher, InvalidPattern_DynamicOutputNodeInPattern_Miss) {
  auto compute_graph = gert::ShareGraph::GroupedMatMulAllReduceSingleGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto pattern = std::make_unique<Pattern>(std::move(graph));

  gert::GertRuntimeStub runtime_stub;
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph0));
  EXPECT_EQ(matcher.MatchNext(), nullptr);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindLogRegex(DLOG_ERROR, "is dynamic output")>= 0);
}

/**
 * 该测试用例中的pattern，add_layer_norm共4个输出，其中后3个作为pattern的输出
 *  pattern:
 *     data0  data1           data3   data4
 *                              \     /
 *                     data2    add0
 *                        \      /
 *                          add
 *                           |
 *        data0  data1     cast1
 *            \     \      /
 *             addlayernorm
 *              /   |  |  |
 *           cast2  |  |  |
 *              \   |  | /
 *               netoutput
 */
TEST_F(UtestPatternMatcher, PatternOutNode_PartialOutputAnchorsAsPatternOutput_Match) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data0 = EsCreateGraphInput(esb_graph, 0);
  auto data1 = EsCreateGraphInput(esb_graph, 1);
  auto data2 = EsCreateGraphInput(esb_graph, 2);
  auto data3 = EsCreateGraphInput(esb_graph, 3);
  auto data4 = EsCreateGraphInput(esb_graph, 4);
  auto add = EsAdd(data2, EsAdd(data3, data4));
  auto cast1 = EsCast(add, DT_FLOAT);
  auto add_layer_norm = EsAddLayerNorm(data0, data1, cast1, EsCreateScalarInt32(esb_graph, 1), EsCreateScalarInt32(esb_graph, 1), 0, 0);
  auto cast2 = EsCast(add_layer_norm.x, DT_FLOAT);
  esb_graph->SetGraphOutput(cast2, 0);
  esb_graph->SetGraphOutput(add_layer_norm.mean, 1);
  esb_graph->SetGraphOutput(add_layer_norm.rstd, 2);
  esb_graph->SetGraphOutput(add_layer_norm.x, 3);
  auto graph = pattern_graph.BuildAndReset();

  // build target graph
  ComputeGraphPtr target_graph = MakeShared<ComputeGraph>("target_graph");
  GraphUtils::CopyComputeGraph(GraphUtilsEx::GetComputeGraph(*graph), target_graph);
  auto pattern = std::make_unique<Pattern>(std::move(*graph));
  PatternMatcher matcher(std::move(pattern), GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_graph));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
}

/**
 * 该测试用例中的pattern，add_layer_norm共4个输出，其中后3个作为pattern的输出
 *  pattern:
 *     data0  data1           data3   data4
 *                              \     /
 *                     data2    add0
 *                        \      /
 *                          add
 *                           |
 *        data0  data1     cast1
 *            \     \      /
 *             addlayernorm
 *              /   |  |  |
 *           cast2  |  |  |
 *              \   |  | /
 *               netoutput
 */
TEST_F(UtestPatternMatcher, PatternOutNode_PartialOutputAnchorsAsPatternOutput_EnableIrAttrMatch_Match) {
  // build pattern graph
  auto pattern_graph = es::EsGraphBuilder("pattern");
  auto esb_graph = pattern_graph.GetCGraphBuilder();
  auto data0 = EsCreateGraphInput(esb_graph, 0);
  auto data1 = EsCreateGraphInput(esb_graph, 1);
  auto data2 = EsCreateGraphInput(esb_graph, 2);
  auto data3 = EsCreateGraphInput(esb_graph, 3);
  auto data4 = EsCreateGraphInput(esb_graph, 4);
  auto add = EsAdd(data2, EsAdd(data3, data4));
  auto cast1 = EsCast(add, DT_FLOAT);
  auto add_layer_norm = EsAddLayerNorm(data0, data1, cast1, EsCreateScalarInt32(esb_graph, 1), EsCreateScalarInt32(esb_graph, 1), 0, 0);
  auto cast2 = EsCast(add_layer_norm.x, DT_FLOAT);
  esb_graph->SetGraphOutput(cast2, 0);
  esb_graph->SetGraphOutput(add_layer_norm.mean, 1);
  esb_graph->SetGraphOutput(add_layer_norm.rstd, 2);
  esb_graph->SetGraphOutput(add_layer_norm.x, 3);
  auto graph = pattern_graph.BuildAndReset();

  // build target graph
  ComputeGraphPtr target_graph = MakeShared<ComputeGraph>("target_graph");
  GraphUtils::CopyComputeGraph(GraphUtilsEx::GetComputeGraph(*graph), target_graph);
  auto pattern = std::make_unique<Pattern>(std::move(*graph));
  PatternMatcher matcher(std::move(pattern), GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_graph));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
}

/**
 * 该测试用例中的pattern，add_layer_norm共4个输出，其中后3个作为pattern的输出
 *  pattern:
 *     data0  data1           data3   data4
 *                              \     /
 *                     data2    add0
 *                        \      /
 *                          add
 *                           |
 *        data0  data1     cast1
 *            \     \      /
 *             addlayernorm
 *              /   |  |  |
 *           cast2  |  |  |
 *              \   |  | /
 *               netoutput
 */
TEST_F(UtestPatternMatcher, PatternOutNode_PartialOutputAnchorsAsPatternOutput_EnableIrAttrMatch_AttrValueMiss) {
  // build pattern graph
  auto pattern_graph = BuildPartialOutputAnchorsAsPatternOutputGraph(DT_FLOAT);
  // build target graph
  auto target_graph = BuildPartialOutputAnchorsAsPatternOutputGraph(DT_INT64);

  auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph));
  auto matcher_config = PatternMatcherConfigBuilder().EnableIrAttrMatch().Build();
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(*target_graph), std::move(matcher_config));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 0);
  // todo check log
}
/**
   target graph:        pattern:

    data                  data
     |                     |
    abs1                  abs1
     /  \                  |
    exp   relu            relu
      \   /                |
      add                 netoutput
       |
      abs2
       |
      relu2
       |
    netoutput
*
*
*/
TEST_F(UtestPatternMatcher, InvalidBoundary_NotSelfContained_Miss) {
  // build target graph
  auto target_graph_builder = es::EsGraphBuilder("target");
  auto target_esb_graph = target_graph_builder.GetCGraphBuilder();
  auto data_target = EsCreateGraphInput(target_esb_graph, 0);
  auto abs1_target = EsAbs(data_target);
  auto relu_target = EsRelu(abs1_target);
  auto exp_target = EsExp(abs1_target, 0, 0, 0);
  auto add_target = EsAdd(exp_target, relu_target);
  auto abs2_target = EsAbs(add_target);
  auto relu2_target = EsRelu(abs2_target);
  target_esb_graph->SetGraphOutput(relu2_target, 0);
  ComputeGraphPtr target_compute_graph = GraphUtilsEx::GetComputeGraph(*target_graph_builder.BuildAndReset());
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build pattern graph
  auto pattern_graph_builder = es::EsGraphBuilder("pattern");
  auto pattern_esb_graph = pattern_graph_builder.GetCGraphBuilder();
  auto data_pattern = EsCreateGraphInput(pattern_esb_graph, 0);
  auto abs1 = EsAbs(data_pattern);
  auto relu = EsRelu(abs1);
  pattern_esb_graph->SetGraphOutput(relu, 0);
  auto pattern_graph = pattern_graph_builder.BuildAndReset();

  auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph));
  PatternMatcher matcher(std::move(pattern), target_graph);
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
}

/**
   target graph:        pattern:

      data                  data
        |                     |
       abs1                 abs1
      /  \                   |
    exp   relu             relu
      \   /                  |
       add                 netoutput
       |
    netoutput
*
*
*/
TEST_F(UtestPatternMatcher, InvalidBoundary_ForceSelfContained_Match) {
  ComputeGraphPtr target_compute_graph = gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  // build pattern graph
  auto pattern_graph_builder = es::EsGraphBuilder("pattern");
  auto pattern_esb_graph = pattern_graph_builder.GetCGraphBuilder();
  auto data_pattern = EsCreateGraphInput(pattern_esb_graph, 0);
  auto abs1 = EsAbs(data_pattern);
  auto relu = EsRelu(abs1);
  pattern_esb_graph->SetGraphOutput(relu, 0);
  pattern_esb_graph->SetGraphOutput(abs1, 1);
  auto pattern_graph = pattern_graph_builder.BuildAndReset();

  auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph));
  PatternMatcher matcher(std::move(pattern), target_graph);
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  // dlog_setlevel(GE_MODULE_NAME, 0, 0);
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }

  EXPECT_EQ(match_ret.size(), 1);
}
} // namespace fusion
} // namespace ge
