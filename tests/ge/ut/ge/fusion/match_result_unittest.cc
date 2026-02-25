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

#include "ge/fusion/pattern_matcher.h"
#include "ge/fusion/pattern.h"
namespace ge {
namespace fusion {

using namespace es;
class UtestMatchResult : public testing::Test {
public:
  static void SetUpTestSuite() {
  }
  static void TearDownTestSuite() {}
};

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
TEST_F(UtestMatchResult, GetCapturedTensor) {
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

  // capture
  pattern->CaptureTensor({NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(abs1->GetProducer())), 0})
      .CaptureTensor({NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(exp->GetProducer())), 0});

  auto target_graph = GraphUtilsEx::CreateGraphFromComputeGraph(gert::ShareGraph::BuildStaticAbsReluExpAddNodeGraph());
  PatternMatcher matcher(std::move(pattern), std::make_shared<Graph>(target_graph));
  std::vector<std::unique_ptr<MatchResult>> match_ret;
  std::unique_ptr<MatchResult> match;
  while ((match = matcher.MatchNext()), match != nullptr) {
    match_ret.emplace_back(std::move(match));
  }
  EXPECT_EQ(match_ret.size(), 1);

  // get captured tensor
  NodeIo asb1_node_output;
  EXPECT_EQ(match_ret[0]->GetCapturedTensor(0, asb1_node_output), SUCCESS);
  NodeIo exp_node_output;
  EXPECT_EQ(match_ret[0]->GetCapturedTensor(1, exp_node_output), SUCCESS);

  AscendString asb1_type;
  asb1_node_output.node.GetType(asb1_type);
  EXPECT_STREQ(asb1_type.GetString(), "Abs");

  AscendString exp_type;
  exp_node_output.node.GetType(exp_type);
  EXPECT_STREQ(exp_type.GetString(), "Exp");

  NodeIo invalie_node_output;
  EXPECT_NE(match_ret[0]->GetCapturedTensor(2, invalie_node_output), SUCCESS);
}
} // namespace fusion
} // namespace ge