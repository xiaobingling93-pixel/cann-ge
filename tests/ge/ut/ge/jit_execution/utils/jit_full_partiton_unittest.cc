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
#include "faker/space_registry_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "es_ge_test_ops.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/common/types.h"
#include "jit_execution/exe_points/execution_order.h"
#include "api/session/jit_execution/utils/jit_infer_utils.h"
#include "framework/ge_runtime_stub/include/common/compliant_share_graph.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "tests/framework/ge_runtime_stub/include/common/summary_checker.h"
#include "tests/framework/ge_runtime_stub/include/faker/space_registry_faker.h"
#include "jit_execution/utils/partitioner/binary_partitioner.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include <vector>
#include <stack>
using namespace std;
using namespace testing;
using namespace ge;

std::vector<NodePtr> getPreviousNodes(ComputeGraphPtr graph, size_t nodeNum);

class JitFullPartitionUT : public testing::Test {
 protected:
  void SetUp() override {
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  }
  void TearDown() override {
  }
};

ComputeGraphPtr BuildAddAbsReLuReshapeAbsGraph(const vector<int64_t> &shape) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  const auto abs0 = EsAbs(data0);
  const auto add = EsAdd(abs0, abs0);
  const auto abs = EsAbs(add);
  const auto relu = EsRelu(abs);
  const auto reshape = EsReshape(relu, relu, 3, 3);
  const auto abs1 = EsAbs(reshape);
  EsSetGraphOutput(abs1, 0);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  return GraphUtilsEx::GetComputeGraph(*ge_graph);
}

/**
 * 包含可推导的二类算子的图推导前置, 推导时到该节点断图
 * 输入8个节点的图，二类算子为reshape，断图在输出节点，输出7个节点的sliced图,输出3个节点的remaining图
 * 分别对sliced图与remaining图再做推导，保证推导结果符合预期
 *       data
 *        |
 *       abs0
 *        ||
 *       add
 *        |
 *       abs
 *        |
 *       reLu
 *        ||
 *      reshape
 *        |
 *       abs
 *        |
 *      netoutput
 *
 */
TEST_F(JitFullPartitionUT, return_partition_two_graph_can_infer_success_when_origin_graph_infered_not_full) {
  const auto graph = BuildAddAbsReLuReshapeAbsGraph({-1, -1, -1});
  ASSERT_NE(graph, nullptr);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  auto abs5 = graph->FindNode("Abs_5");
  abs5->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({-1,-1,-1}));
  abs5->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1,-1,-1}));
  
  std::vector<NodePtr> nodes = getPreviousNodes(graph, 6);
  
  std::vector<NodePtr> infered_nodes;
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(graph, inputs, infered_nodes), SUCCESS);
  ASSERT_EQ(infered_nodes, nodes);  

  PartionResult partition_result;
  ASSERT_EQ(BinaryPartitioner::Partition(graph, infered_nodes, partition_result), SUCCESS);

  // check sliced graph
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodes().size(), 7);
  // check remaining graph
  EXPECT_EQ(partition_result.remaining_graph->GetAllNodes().size(), 3);
  // check io
  EXPECT_EQ(partition_result.out_idx_2_in_idxs.size(), 1);

  // infer sliced_graph
  infered_nodes.clear();
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.sliced_graph, inputs, infered_nodes), SUCCESS);
  for (size_t i = 0; i < nodes.size(); ++i) {
    ASSERT_EQ(infered_nodes.at(i)->GetName(), nodes.at(i)->GetName());
  }

  // infer remaining_graph
  infered_nodes.clear();
  std::vector<NodePtr> uninfered_nodes;
  for (auto it : partition_result.remaining_graph->GetAllNodes()) {
    uninfered_nodes.push_back(it);
  }
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.remaining_graph, inputs, infered_nodes), SUCCESS);
  for (size_t i = 0; i < infered_nodes.size(); ++i) {
    ASSERT_EQ(infered_nodes.at(i)->GetName(), uninfered_nodes.at(i)->GetName());
  }
}

/**
 *           data
 *            |
 *           abs0
 *            ||
 *           add
 *          /   \
 *       relu0  relu1
 *        |       |
 *       abs1   abs2
 *        |       |
 *    output0  output1
 */
TEST_F(JitFullPartitionUT, return_partition_one_graph) {
  const auto graph = cg::BuildAbsAddReluReluGraph({-1, -1, -1});
  ASSERT_NE(graph, nullptr);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);

  std::vector<NodePtr> nodes;
  for (const auto &node : graph->GetDirectNode()) {
    nodes.push_back(node);
  }
  std::sort(nodes.begin(), nodes.end());

  std::vector<NodePtr> infered_nodes;
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(graph, inputs, infered_nodes), SUCCESS);
  std::sort(infered_nodes.begin(), infered_nodes.end());
  ASSERT_EQ(infered_nodes, nodes);

  PartionResult partition_result;
  ASSERT_EQ(BinaryPartitioner::Partition(graph, infered_nodes, partition_result), SUCCESS);

  // check sliced graph
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodes().size(), 8);
  // check remaining graph
  EXPECT_EQ(partition_result.remaining_graph, nullptr);

  // infer sliced_graph again
  infered_nodes.clear();
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.sliced_graph, inputs, infered_nodes), SUCCESS);
  std::sort(infered_nodes.begin(), infered_nodes.end());
  ASSERT_EQ(infered_nodes, nodes);
  for (size_t i = 0; i < infered_nodes.size(); ++i) {
    ASSERT_EQ(infered_nodes.at(i)->GetName(), nodes.at(i)->GetName());
  }
}

/**
 * 包含可推导的二类算子的图推导前置, 推导时到该节点不断图，继续向后推导
 * 输入7个节点的图，二类算子为reshape，断图在输出节点，输出7个节点的子图
 *       data
 *        |
 *       abs0
 *        ||
 *       add
 *        |
 *       abs
 *        |
 *       reLu
 *        ||
 *      reshape
 *        |
 *      netoutput
 *
 */
ComputeGraphPtr BuildAddAbsReLuReshapeGraph2(const vector<int64_t> &shape) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  const auto abs0 = EsAbs(data0);
  const auto add = EsAdd(abs0, abs0);
  const auto abs = EsAbs(add);
  const auto relu = EsRelu(abs);
  const auto reshape = EsReshape(relu, data0, 3, 3);
  EsSetGraphOutput(reshape, 0);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  GE_ASSERT_NOTNULL(ge_graph);
  return GraphUtilsEx::GetComputeGraph(*ge_graph);
}
TEST_F(JitFullPartitionUT, return_partition_two_graph_conatins_uninferable_reshape) {
  const auto graph = BuildAddAbsReLuReshapeGraph2({-1, -1, -1});
  ASSERT_NE(graph, nullptr);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  auto output = graph->GetOrUpdateNetOutputNode();
  ASSERT_NE(output, nullptr);
  output->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({-1, -1, -1}));
  output->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1, -1, -1}));

  std::vector<NodePtr> nodes = getPreviousNodes(graph, 7);
  std::sort(nodes.begin(), nodes.end());

  std::vector<NodePtr> infered_nodes;
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(graph, inputs, infered_nodes), SUCCESS);
  std::sort(infered_nodes.begin(), infered_nodes.end());
  ASSERT_EQ(infered_nodes, nodes);

  PartionResult partition_result;
  ASSERT_EQ(BinaryPartitioner::Partition(graph, infered_nodes, partition_result), SUCCESS);

  // check sliced graph
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodes().size(), 7);
  // check remaining graph
  EXPECT_EQ(partition_result.remaining_graph, nullptr);

  // infer sliced_graph again
  infered_nodes.clear();
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.sliced_graph, inputs, infered_nodes), SUCCESS);
  std::sort(infered_nodes.begin(), infered_nodes.end());
  for (size_t i = 0; i < infered_nodes.size(); ++i) {
    ASSERT_EQ(infered_nodes.at(i)->GetName(), nodes.at(i)->GetName());
  }
}

/**
 * 包含可推导的二类算子的图推导前置, 推导时到该节点不断图，继续向后推导
 * 输入8个节点的图，二类算子为reshape，且value参数是常量const，断图在输出节点，输出8个节点的sliced图,输出0个节点的remaining图
 * 对sliced图再做推导，保证推导结果符合预期
 *       data
 *        |
 *       abs0
 *        ||
 *       add
 *        |
 *       abs
 *        |
 *       reLu  const
 *        |    /
 *      reshape
 *        |
 *      netoutput
 *
 */
TEST_F(JitFullPartitionUT, return_partition_one_graph_can_infer_success_when_origin_graph_infered_full) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);
  auto data0 = EsCreateGraphInputWithDetails(graph.get(), 0, "data0", nullptr, C_DT_FLOAT, C_FORMAT_ND, nullptr, 0);
  std::vector<int64_t> shape = {-1, -1, -1, -1, -1};
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  es::EsTensorHolder data0_holder(data0);
  const auto abs0 = es::Abs(data0_holder);
  const auto add = es::Add(abs0, abs0);
  const auto abs = es::Abs(add);
  const auto relu = es::Relu(abs);

  std::vector<int64_t> dims_data{1, 1, 3, 3, -1};
  int64_t dims_size = 5;
  es::EsTensorHolder const_0(EsCreateConstInt64(graph.get(), dims_data.data(), &dims_size, 1));

  auto reshape = es::Reshape(relu, const_0, 0, -1);
  EsSetGraphOutput(reshape.GetCTensorHolder(), 0);

  auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  ASSERT_NE(ge_graph, nullptr);
  const auto computerGraph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  ASSERT_NE(computerGraph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 5, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 5, 5, 6})));
  inputs.emplace_back(td);

  auto reshape_op_0 = computerGraph->FindFirstNodeMatchType("Reshape");
  ASSERT_NE(reshape_op_0, nullptr);
  auto reshape_op_desc0 = reshape_op_0->GetOpDesc();
  reshape_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  reshape_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT64);

  std::vector<NodePtr> nodes;
  for (const auto &node : computerGraph->GetDirectNode()) {
    nodes.push_back(node);
  }

  std::vector<NodePtr> infered_nodes;
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(computerGraph, inputs, infered_nodes), SUCCESS);
  ASSERT_EQ(infered_nodes, nodes);

  PartionResult partition_result;
  ASSERT_EQ(BinaryPartitioner::Partition(computerGraph, infered_nodes, partition_result), SUCCESS);
  // check sliced graph
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodes().size(), 8);
  // check remaining graph
  EXPECT_EQ(partition_result.remaining_graph, nullptr);
  // check io
  EXPECT_EQ(partition_result.out_idx_2_in_idxs.size(), 0);

  // infer sliced_graph
  infered_nodes.clear();
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.sliced_graph, inputs, infered_nodes), SUCCESS);
  ASSERT_EQ(infered_nodes, nodes);
}

/**
 * 包含不可推导的二类算子的图推导前置, 推导时到该节点断图
 * 剩余节点存在继续推导的情况（手动构造abs输出有符号），inferednode与uninfernode存在成环情况
 * 输入8个节点的图，二类算子为reshape
 * 输出切图失败，因为存在成环
 *       data0
 *        |
 *       reLu
 *        ||
 *      reshape
 *        |
 *       reLu
 *        |
 *       abs   data1
 *          \   /
 *           add
 *            |
 *          netout
 */
TEST_F(JitFullPartitionUT, return_partition_two_graph_can_infer_success_when_origin_graph_uninfernode_output_has_symbol) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  const auto data1 = EsCreateGraphInput(graph.get(), 1);
  std::vector<int64_t> shape = {-1, -1, -1};
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  EsSetShape(data1, shape.data(), static_cast<int64_t>(shape.size()));
  const auto relu = EsRelu(data0);
  const auto reshape = EsReshape(relu, relu, 3, 3);
  const auto relu1 = EsRelu(reshape);
  const auto abs = EsAbs(relu1);
  const auto add = EsAdd(abs, data1);

  EsSetGraphOutput(add, 0);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  ASSERT_NE(ge_graph, nullptr);
  const auto computerGraph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  ASSERT_NE(computerGraph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);
  auto abs5 = computerGraph->FindNode("Abs_3");
  abs5->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({-1,-1,-1}));
  abs5->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1,-1,-1}));
  auto attr = abs5->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>(); // 模拟构造abs节点不依赖输入能继续推导
  ASSERT_NE(attr, nullptr);
  auto add3 = computerGraph->FindNode("Add_4");
  add3->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(GeShape({-1,-1,-1}));
  add3->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({-1,-1,-1}));
  
  
  std::vector<NodePtr> infered_nodes;
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(computerGraph, inputs, infered_nodes), SUCCESS);

  PartionResult partition_result;
  ASSERT_EQ(BinaryPartitioner::Partition(computerGraph, infered_nodes, partition_result), SUCCESS);
}

/**
 * 包含不可推导的二类算子的图推导前置, 推导时到该节点断图，const节点断在上一张图中
 * 剩余节点仍然存在二类算子继续推导的情况，期望剩余节点能完成推导，const继承到下一张图的输入
 *       data0
 *        |
 *       reLu
 *        ||
 *      reshape    const
 *          \      /
 *          reshape
 *             |
 *            relu
 *             |
 *           netout
 */
TEST_F(JitFullPartitionUT, return_partition_second_graph_can_infer_success_when_origin_graph_second_reshape_has_const_input) {
  auto graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("graph"), EsDestroyGraphBuilder);

  const auto data0 = EsCreateGraphInput(graph.get(), 0);
  std::vector<int64_t> shape = {-1, -1, -1, -1, -1};
  EsSetShape(data0, shape.data(), static_cast<int64_t>(shape.size()));
  es::EsTensorHolder data0_holder(data0);
  const auto relu = es::Relu(data0_holder);
  const auto reshape = es::Reshape(relu, relu, 0, -1);
  std::vector<int64_t> dims_data{1, 1, 3, 3, -1};
  int64_t dims_size = 5;
  es::EsTensorHolder const_0(EsCreateConstInt64(graph.get(), dims_data.data(), &dims_size, 1));

  auto reshape1 = es::Reshape(reshape, const_0, 0, -1);
  const auto relu1 = es::Relu(reshape1);
  EsSetGraphOutput(relu1.GetCTensorHolder(), 0);

  const auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph.get()))));
  ASSERT_NE(ge_graph, nullptr);
  const auto computerGraph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  ASSERT_NE(computerGraph, nullptr);

  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 5, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 5, 5, 6})));
  inputs.emplace_back(td);
  auto reshape2 = computerGraph->FindNode("Reshape_3");
  reshape2->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({-1,-1,-1,-1,-1}));
  reshape2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1,-1,-1,-1,-1}));
  auto reshape_op_desc0 = reshape2->GetOpDesc();
  reshape_op_desc0->MutableInputDesc(0)->SetDataType(DT_INT64);
  reshape_op_desc0->MutableInputDesc(1)->SetDataType(DT_INT64);
  
  std::vector<NodePtr> nodes = getPreviousNodes(computerGraph, 4);
  std::vector<NodePtr> infered_nodes;
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(computerGraph, inputs, infered_nodes), SUCCESS);

  PartionResult partition_result;
  ASSERT_EQ(BinaryPartitioner::Partition(computerGraph, infered_nodes, partition_result), SUCCESS);

  // check sliced graph
  EXPECT_EQ(partition_result.sliced_graph->GetAllNodes().size(), 5);
  // check remaining graph
  EXPECT_EQ(partition_result.remaining_graph->GetAllNodes().size(), 5);
  // check io
  EXPECT_EQ(partition_result.out_idx_2_in_idxs.size(), 1);

  // infer sliced_graph
  infered_nodes.clear();
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.sliced_graph, inputs, infered_nodes), SUCCESS);
  for (size_t i = 0; i < nodes.size(); ++i) {
    ASSERT_EQ(infered_nodes.at(i)->GetName(), nodes.at(i)->GetName());
  }

  // infer remaining_graph
  infered_nodes.clear();
  std::vector<NodePtr> uninfered_nodes;
  for (auto it : partition_result.remaining_graph->GetAllNodes()) {
    uninfered_nodes.push_back(it);
  }
  ASSERT_EQ(JitInferUtils::InferGraphAndGetInferredNodes(partition_result.remaining_graph, inputs, infered_nodes), SUCCESS);
  ASSERT_EQ(infered_nodes.size(), 5);
  for (size_t i = 0; i < infered_nodes.size(); ++i) {
    ASSERT_EQ(infered_nodes.at(i)->GetName(), uninfered_nodes.at(i)->GetName());
  }
}