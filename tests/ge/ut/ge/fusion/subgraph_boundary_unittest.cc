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
#include <memory>
#include "ge/fusion/subgraph_boundary.h"

#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "ge_common/ge_api_error_codes.h"

#include "common/share_graph.h"

namespace ge {
namespace fusion {
class UtestSubgraphBoundary : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
  }
};

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphInput_SingleConsumer) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphInput subgraph_input;
  EXPECT_EQ(subgraph_input.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  auto all_subgraph_inputs = subgraph_input.GetAllInputs();
  EXPECT_EQ(all_subgraph_inputs.size(), 1);
  EXPECT_EQ(NodeAdapter::GNode2Node(all_subgraph_inputs[0].node), add);
  EXPECT_EQ(all_subgraph_inputs[0].index, 0);
}

/*
 *             data1
 *                |
 *            shape
 *            /   \      data2
 *        relu     \      /
 *         \        add
 *          \       /
 *          netoutput
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphInput_MultiConsumer) {
  auto graph = gert::ShareGraph::ShapeToMultiAiCoreGraph();
  auto relu = graph->FindNode("relu");
  auto add = graph->FindNode("add1");

  SubgraphInput subgraph_input;
  EXPECT_EQ(subgraph_input.AddInput({NodeAdapter::Node2GNode(relu), 0}), SUCCESS);
  EXPECT_EQ(subgraph_input.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  auto all_subgraph_inputs = subgraph_input.GetAllInputs();
  EXPECT_EQ(all_subgraph_inputs.size(), 2);
  EXPECT_EQ(NodeAdapter::GNode2Node(all_subgraph_inputs[0].node), relu);
  EXPECT_EQ(NodeAdapter::GNode2Node(all_subgraph_inputs[1].node), add);
  EXPECT_EQ(all_subgraph_inputs[0].index, 0);
  EXPECT_EQ(all_subgraph_inputs[0].index, 0);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphInput_MultiProducer_ERROR) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  auto gnode_add = NodeAdapter::Node2GNode(add);

  SubgraphInput subgraph_input;
  EXPECT_EQ(subgraph_input.AddInput({gnode_add, 0}), SUCCESS);
  // add-1 from other producer, not valid input
  EXPECT_NE(subgraph_input.AddInput({gnode_add, 1}), SUCCESS);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphOutput) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  NodeIo node_output;
  EXPECT_EQ(subgraph_output.GetOutput(node_output), SUCCESS);
  EXPECT_EQ(NodeAdapter::GNode2Node(node_output.node), add);
  EXPECT_EQ(node_output.index, 0);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphOutput_Twice_ERROR) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  EXPECT_NE(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphOutput_NonExistOutput_ERROR) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphOutput subgraph_output;
  EXPECT_NE(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphBoundary) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphInput subgraph_input0;
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  SubgraphInput subgraph_input1;
  EXPECT_EQ(subgraph_input1.AddInput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  SubgraphBoundary boundary;
  EXPECT_EQ(boundary.AddInput(0, subgraph_input0), SUCCESS);
  EXPECT_EQ(boundary.AddInput(1, subgraph_input1), SUCCESS);
  EXPECT_EQ(boundary.AddOutput(0, subgraph_output), SUCCESS);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphBoundary_AnotherConstruct) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphInput subgraph_input0;
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  SubgraphInput subgraph_input1;
  EXPECT_EQ(subgraph_input1.AddInput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  SubgraphBoundary boundary({subgraph_input0, subgraph_input1}, {subgraph_output});
  std::vector<SubgraphInput> inputs;
  boundary.GetAllInputs(inputs);
  EXPECT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[1].GetAllInputs().size(), 1);
  EXPECT_EQ(inputs[1].GetAllInputs()[0].index, 1);
  SubgraphOutput output;
  EXPECT_EQ(boundary.GetOutput(0, output), SUCCESS);
  NodeIo node_output;
  EXPECT_EQ(output.GetOutput(node_output), SUCCESS);
  EXPECT_STREQ(NodeAdapter::GNode2Node(node_output.node)->GetTypePtr(), "Add");
  EXPECT_EQ(node_output.index, 0);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphBoundary_GetInput) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphInput subgraph_input0;
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  SubgraphInput subgraph_input1;
  EXPECT_EQ(subgraph_input1.AddInput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  SubgraphBoundary boundary;
  SubgraphInput subgraph_input_got;
  EXPECT_EQ(boundary.GetInput(0, subgraph_input_got), FAILED);

  EXPECT_EQ(boundary.AddInput(0, subgraph_input0), SUCCESS);
  EXPECT_EQ(boundary.GetInput(0, subgraph_input_got), SUCCESS);
  EXPECT_EQ(subgraph_input_got.GetAllInputs().size(), 1);

  AscendString input_node_type;
  subgraph_input_got.GetAllInputs()[0].node.GetType(input_node_type);
  EXPECT_STREQ(input_node_type.GetString(), "Add");
  EXPECT_EQ(subgraph_input_got.GetAllInputs()[0].index, 0);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphBoundary_GetOutput) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphInput subgraph_input0;
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  SubgraphInput subgraph_input1;
  EXPECT_EQ(subgraph_input1.AddInput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  SubgraphBoundary boundary;
  SubgraphOutput subgraph_output_got;
  EXPECT_EQ(boundary.GetOutput(0, subgraph_output_got), FAILED);

  EXPECT_EQ(boundary.AddOutput(0, subgraph_output), SUCCESS);
  EXPECT_EQ(boundary.GetOutput(0, subgraph_output_got), SUCCESS);
  NodeIo node_output;
  EXPECT_EQ(subgraph_output_got.GetOutput(node_output), SUCCESS);
  AscendString output_node_type;
  node_output.node.GetType(output_node_type);
  EXPECT_STREQ(output_node_type.GetString(), "Add");
  EXPECT_EQ(node_output.index, 0);

  EXPECT_NE(boundary.GetOutput(2, subgraph_output_got), SUCCESS);
}

/*
 *    add1
 *    /  \
 * data1 data2
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphBoundary_AddSameInputTwice_Failed) {
  auto graph = gert::ShareGraph::AicoreGraph();
  auto add = graph->FindNode("add1");
  SubgraphInput subgraph_input0;
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  SubgraphInput subgraph_input1;
  EXPECT_EQ(subgraph_input1.AddInput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
  SubgraphOutput subgraph_output;
  EXPECT_EQ(subgraph_output.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);

  SubgraphBoundary boundary;
  EXPECT_EQ(boundary.AddInput(0, subgraph_input0), SUCCESS);
  EXPECT_NE(boundary.AddInput(0, subgraph_input1), SUCCESS);
  EXPECT_EQ(boundary.AddOutput(0, subgraph_output), SUCCESS);
  EXPECT_NE(boundary.AddOutput(0, subgraph_output), SUCCESS);
}
/*
 *             data1
 *                |
 *            shape
 *            /   \      data2
 *        relu     \      /
 *         \        add
 *          \       /
 *          netoutput
 */
TEST_F(UtestSubgraphBoundary, CreateSubgraphBoundary_MultiInputFromOneTensor) {
  auto graph = gert::ShareGraph::ShapeToMultiAiCoreGraph();
  auto relu = graph->FindNode("relu");
  auto add = graph->FindNode("add1");

  SubgraphInput subgraph_input0;
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(relu), 0}), SUCCESS);
  EXPECT_EQ(subgraph_input0.AddInput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);
  SubgraphInput subgraph_input1;
  EXPECT_EQ(subgraph_input1.AddInput({NodeAdapter::Node2GNode(add), 1}), SUCCESS);
  SubgraphOutput subgraph_output1;
  EXPECT_EQ(subgraph_output1.SetOutput({NodeAdapter::Node2GNode(relu), 0}), SUCCESS);
  SubgraphOutput subgraph_output2;
  EXPECT_EQ(subgraph_output2.SetOutput({NodeAdapter::Node2GNode(add), 0}), SUCCESS);


  SubgraphBoundary boundary;
  EXPECT_EQ(boundary.AddInput(0, subgraph_input0), SUCCESS);
  EXPECT_EQ(boundary.AddInput(1, subgraph_input1), SUCCESS);
  EXPECT_EQ(boundary.AddOutput(0, subgraph_output1), SUCCESS);
  EXPECT_EQ(boundary.AddOutput(1, subgraph_output2), SUCCESS);
}
} // namespace fusion
} // namespace ge
