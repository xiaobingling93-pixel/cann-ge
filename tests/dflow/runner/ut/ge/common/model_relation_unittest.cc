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

#include "compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "framework/common/types.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "utils/graph_utils.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "macro_utils/dt_public_scope.h"
#include "dflow/base/model/model_relation.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/debug/ge_attr_define.h"

namespace ge {
class ModelRelationTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

 protected:

  void SetSubGraph(ComputeGraphPtr graph, OpDesc &op_desc, const std::string &name) {
    auto subgraph = std::make_shared<ComputeGraph>(name);
    subgraph->SetParentGraph(graph);
    subgraph->SetParentNode(graph->FindNode(op_desc.GetName()));
    op_desc.AddSubgraphName(name);
    op_desc.SetSubgraphInstanceName(0, name);
    graph->AddSubgraph(name, subgraph);
  }

  void SetSubGraph(ComputeGraphPtr graph, ComputeGraphPtr subgraph, OpDesc &op_desc, const std::string &name) {
    subgraph->SetParentGraph(graph);
    subgraph->SetParentNode(graph->FindNode(op_desc.GetName()));
    op_desc.AddSubgraphName(name);
    op_desc.SetSubgraphInstanceName(0, name);
    graph->AddSubgraph(name, subgraph);
  }

  ComputeGraphPtr BuildGraph() {
    auto builder = ut::GraphBuilder("g1");
    auto data1 = builder.AddNode("data1", DATA, 0, 1);
    auto data2 = builder.AddNode("data2", DATA, 0, 1);
    AttrUtils::SetBool(data2->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 10);
    AttrUtils::SetStr(data2->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO");
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    const GeTensorDesc tensor_desc;
    auto output_desc = data1->GetOpDesc()->MutableOutputDesc(0);
    AttrUtils::SetBool(output_desc, ATTR_NAME_FLOW_ATTR, true);
    AttrUtils::SetInt(output_desc, ATTR_NAME_FLOW_ATTR_DEPTH, 6);
    AttrUtils::SetStr(output_desc, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO");
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
    auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 2, 1);
    auto partitioned_call_2 = builder.AddNode("PartitionedCall2", PARTITIONEDCALL, 2, 1);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

    SetSubGraph(builder.GetGraph(), *partitioned_call_1->GetOpDesc(), "subgraph-1");
    SetSubGraph(builder.GetGraph(), *partitioned_call_2->GetOpDesc(), "subgraph-2");

    builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
    builder.AddDataEdge(data2, 0, partitioned_call_1, 1);
    builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_2, 0);
    builder.AddDataEdge(data2, 0, partitioned_call_2, 1);
    builder.AddDataEdge(partitioned_call_2, 0, net_output, 0);
    return builder.GetGraph();
  }

  static ComputeGraphPtr BuildControlOpIfGraph(const ComputeGraphPtr &root_graph, const NodePtr &if_parent) {
    DEF_GRAPH(then_branch) {
      auto data = OP_CFG(DATA)
          .InCnt(1)
          .OutCnt(1)
          .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});
      auto net_output = OP_CFG(NETOUTPUT)
          .InCnt(1)
          .OutCnt(1)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});
      CHAIN(NODE("then_arg_0", data)->NODE("then_Node_Output", net_output));
    };

    DEF_GRAPH(else_branch) {
      auto data = OP_CFG(DATA)
          .InCnt(1)
          .OutCnt(1)
          .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});
      auto neg_op = OP_CFG("Neg")
          .InCnt(1)
          .OutCnt(2)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});
      auto net_output = OP_CFG(NETOUTPUT)
          .InCnt(1)
          .OutCnt(1)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});
      CHAIN(NODE("else_arg_0", data)->NODE("neg", neg_op)->NODE("else_Node_Output", net_output));
    };

    auto then_graph = ToComputeGraph(then_branch);
    auto else_graph = ToComputeGraph(else_branch);

    DEF_GRAPH(if_graph) {
      auto pred_data = OP_CFG(DATA)
          .InCnt(1)
          .OutCnt(1)
          .Attr(ATTR_NAME_INDEX, 0)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});

      auto value_data = OP_CFG(DATA)
          .InCnt(1)
          .OutCnt(1)
          .Attr(ATTR_NAME_INDEX, 1)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});

      auto if_op = OP_CFG(IF)
          .InCnt(2)
          .OutCnt(1)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8})
          .Build("if");

      if_op->MutableOutputDesc(0)->SetShape(GeShape({4, 8}));
      if_op->RegisterSubgraphIrName("then_branch", SubgraphType::kStatic);
      if_op->RegisterSubgraphIrName("else_branch", SubgraphType::kStatic);
      if_op->AddSubgraphName(then_graph->GetName());
      if_op->SetSubgraphInstanceName(0, then_graph->GetName());
      if_op->AddSubgraphName(else_graph->GetName());
      if_op->SetSubgraphInstanceName(1, else_graph->GetName());

      auto net_output = OP_CFG(NETOUTPUT)
          .InCnt(1)
          .OutCnt(1)
          .TensorDesc(FORMAT_ND, DT_FLOAT, {4, 8});

      CHAIN(NODE("arg_pred", pred_data)->NODE(if_op)->NODE("Node_Output", net_output));
      CHAIN(NODE("arg_value", value_data)->NODE(if_op));
    };

    auto if_ge_graph = ToComputeGraph(if_graph);
    if_ge_graph->SetParentGraph(root_graph);
    if_ge_graph->SetParentNode(if_parent);
    auto if_node = if_ge_graph->FindFirstNodeMatchType(IF);
    EXPECT_TRUE(if_node != nullptr);
    then_graph->SetParentNode(if_node);
    then_graph->SetParentGraph(if_ge_graph);
    else_graph->SetParentNode(if_node);
    else_graph->SetParentGraph(if_ge_graph);
    if_ge_graph->TopologicalSorting();
    root_graph->AddSubgraph(then_graph);
    root_graph->AddSubgraph(else_graph);
    return if_ge_graph;
  }

  ComputeGraphPtr BuildGraphWithIf() {
    auto builder = ut::GraphBuilder("g1");
    auto data1 = builder.AddNode("data1", DATA, 0, 1);
    auto data2 = builder.AddNode("data2", DATA, 0, 1);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    const GeTensorDesc tensor_desc;
    auto output_desc = data1->GetOpDesc()->MutableOutputDesc(0);
    AttrUtils::SetBool(output_desc, ATTR_NAME_FLOW_ATTR, true);
    AttrUtils::SetInt(output_desc, ATTR_NAME_FLOW_ATTR_DEPTH, 6);
    AttrUtils::SetStr(output_desc, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO");
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
    auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 2, 1);

    auto cpu_subgraph = std::make_shared<ComputeGraph>("cpu-subgraph");
    auto graph = builder.GetGraph();
    auto if_graph = BuildControlOpIfGraph(graph, partitioned_call_1);
    SetSubGraph(graph, if_graph, *partitioned_call_1->GetOpDesc(), "subgraph-1");
    // add cpu subgraph
    auto if_node = if_graph->FindFirstNodeMatchType(IF);
    cpu_subgraph->SetParentGraph(if_graph);
    cpu_subgraph->SetParentNode(if_node);
    if_node->GetOpDesc()->AddSubgraphName("cpu-subgraph");
    if_node->GetOpDesc()->SetSubgraphInstanceName(2, "cpu-subgraph");
    graph->AddSubgraph("cpu-subgraph", cpu_subgraph);
    // add engine type to cpu graph
    (void)AttrUtils::SetStr(cpu_subgraph, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_CPU);
    builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
    builder.AddDataEdge(data2, 0, partitioned_call_1, 1);
    builder.AddDataEdge(partitioned_call_1, 0, net_output, 0);
    return builder.GetGraph();
  }

  ComputeGraphPtr BuildGraphWithNoInput() {
    auto builder = ut::GraphBuilder("g1");
    auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 0, 1);
    auto partitioned_call_2 = builder.AddNode("PartitionedCall2", PARTITIONEDCALL, 1, 1);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

    SetSubGraph(builder.GetGraph(), *partitioned_call_1->GetOpDesc(), "subgraph-1");
    SetSubGraph(builder.GetGraph(), *partitioned_call_2->GetOpDesc(), "subgraph-2");

    builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_2, 0);
    builder.AddDataEdge(partitioned_call_2, 0, net_output, 0);
    return builder.GetGraph();
  }

  ComputeGraphPtr BuildGraphWithManyToOnePartitionedCalls() {
    auto builder = ut::GraphBuilder("g1");
    auto data1 = builder.AddNode("data1", DATA, 0, 1);
    auto data2 = builder.AddNode("data2", DATA, 0, 1);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
    auto partitioned_call_1 = builder.AddNode("PartitionedCall1", PARTITIONEDCALL, 1, 1);
    auto partitioned_call_2 = builder.AddNode("PartitionedCall2", PARTITIONEDCALL, 1, 1);
    auto partitioned_call_3 = builder.AddNode("PartitionedCall3", PARTITIONEDCALL, 2, 1);
    auto net_output = builder.AddNode("NetOutput", NETOUTPUT, 1, 1);

    SetSubGraph(builder.GetGraph(), *partitioned_call_1->GetOpDesc(), "subgraph-1");
    SetSubGraph(builder.GetGraph(), *partitioned_call_2->GetOpDesc(), "subgraph-2");
    SetSubGraph(builder.GetGraph(), *partitioned_call_3->GetOpDesc(), "subgraph-3");

    builder.AddDataEdge(data1, 0, partitioned_call_1, 0);
    builder.AddDataEdge(data2, 0, partitioned_call_2, 0);
    builder.AddDataEdge(partitioned_call_1, 0, partitioned_call_3, 0);
    builder.AddDataEdge(partitioned_call_2, 0, partitioned_call_3, 1);
    builder.AddDataEdge(partitioned_call_3, 0, net_output, 0);
    return builder.GetGraph();
  }
};

/**
 *      NetOutput
 *         |
 *         |
 *        PC_2
 *        |  \
 *       PC_1 |
 *     /     \
 *    |      |
 *  data1  data2
 */
TEST_F(ModelRelationTest, TestBuildFromRootGraph) {
  ComputeGraphPtr graph = BuildGraph();
  std::unique_ptr<ModelRelation> model_relation;
  auto ret = ModelRelationBuilder().BuildFromRootGraph(*graph, model_relation);
  ASSERT_EQ(ret, SUCCESS);
  auto queue_defs = model_relation->endpoints;
  ASSERT_EQ(queue_defs.size(), 4);
  EXPECT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 2);
  EXPECT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  EXPECT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
}

/**
 *      NetOutput
 *         |
 *        PC_2
 *        |
 *       PC_1
 */
TEST_F(ModelRelationTest, TestBuildFromRootGraphWithNoInput) {
  ComputeGraphPtr graph = BuildGraphWithNoInput();
  std::unique_ptr<ModelRelation> model_relation;
  auto ret = ModelRelationBuilder().BuildFromRootGraph(*graph, model_relation);
  ASSERT_EQ(ret, SUCCESS);
  auto queue_defs = model_relation->endpoints;
  ASSERT_EQ(queue_defs.size(), 2);
  EXPECT_EQ(model_relation->root_model_endpoint_info.input_endpoint_names.size(), 0);
  EXPECT_EQ(model_relation->root_model_endpoint_info.output_endpoint_names.size(), 1);
  EXPECT_EQ(model_relation->submodel_endpoint_infos.size(), 2);
}
}  // namespace ge
