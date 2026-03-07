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
#include "macro_utils/dt_public_scope.h"
#include "graph/partition/dynamic_shape_partition.h"
#include "compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/omg_util/omg_util.h"
#include <gmock/gmock.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph/ge_local_context.h"
#include "register/op_impl_registry.h"
#include "register/optimization_option_registry.h"
#include "faker/space_registry_faker.h"
#include "graph/ir_definitions_recover.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "common/share_graph.h"
#include "runtime/config.h"
#include "runtime/dev.h"
#include "runtime_stub.h"
#include "common/ge_common/ge_types.h"
#include "depends/slog/src/slog_stub.h"
#include <set>

namespace ge {
namespace {
// todo 把注册做成stub的庄能力，不影响其他流程
IMPL_OP(AddTilingDepend).TilingInputsDataDependency({1});
IMPL_OP(AddTilingDependPlacementHasAicpu)
  .TilingInputsDataDependency({1}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU});

GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, Format format = FORMAT_NCHW,
                                 DataType data_type = DT_FLOAT) {
  GeShape ge_shape{vector<int64_t>(shape)};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  return tensor_desc;
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                            DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                             DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(GeTensorDescPtr tensor_desc) {
    op_desc_->AddOutputDesc(tensor_desc->Clone());
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};
}  // namespace

class UtestDynamicShapePartition : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
  static void ReInitOo() {
    GetThreadLocalContext().GetOo().Initialize(GetThreadLocalContext().GetAllOptions(),
                                               OptionRegistry::GetInstance().GetRegisteredOptTable());
  }
};

TEST_F(UtestDynamicShapePartition, single_op_scene_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({-1}).AddOutputDesc({-1}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({-1}).AddOutputDesc({-1}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({-1}).AddOutputDesc({-1}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  (void)AttrUtils::SetBool(graph, ATTR_SINGLE_OP_SCENE, true);

  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
}

TEST_F(UtestDynamicShapePartition, not_single_op_scene_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root");
  NodePtr node1_root =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(root_graph);
  NodePtr add_n_node_root =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(root_graph);
  NodePtr node2_root =
      NodeBuilder("node2", RELU).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(root_graph);

  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "123");

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node4 =
      NodeBuilder("node4", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(graph);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutControlAnchor(), node2->GetInControlAnchor());
  GraphUtils::AddEdge(add_n_node->GetOutControlAnchor(), node4->GetInControlAnchor());
  GraphUtils::AddEdge(node4->GetOutControlAnchor(), node2->GetInControlAnchor());


  GraphUtils::AddEdge(node1_root->GetOutDataAnchor(0), add_n_node_root->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node_root->GetOutDataAnchor(0), node2_root->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node_root->GetOutControlAnchor(), node2_root->GetInControlAnchor());

  root_graph->AddSubGraph(graph);

  DynamicShapePartitioner partitioner(root_graph);

  bool is_unknown = false;
  EXPECT_EQ(partitioner.IsUnknownShapeNode(add_n_node_root, is_unknown), SUCCESS);
  EXPECT_EQ(is_unknown, true);

  auto opdesc = add_n_node_root->GetOpDesc();
  opdesc->MutableOutputDesc(0)->SetDataType(DT_RESOURCE);
  opdesc->SetOpEngineName("DNN_VM_HOST_CPU");
  EXPECT_EQ(partitioner.IsUnknownShapeNode(add_n_node_root, is_unknown), SUCCESS);
  EXPECT_EQ(is_unknown, true);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
}

TEST_F(UtestDynamicShapePartition, TestSingleOpWithSubGraph) {
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_0) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE("branch_0_Node_Output", net_output));
  };

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b0 = ToComputeGraph(branch_0);
  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b0->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b0->GetName());
    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(1, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b0->SetParentNode(node_partitioned_call_op2);
  sub_graph_b0->SetParentGraph(root_graph);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b0), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  (void)AttrUtils::SetBool(root_graph, ATTR_SINGLE_OP_SCENE, true);
  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  for (auto &subgraph : root_graph->GetAllSubgraphs()) {
    AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  }
  DynamicShapePartitioner partitioner(root_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(sub_graph->GetGraphUnknownFlag(), true);
  EXPECT_EQ(sub_graph_b1->GetGraphUnknownFlag(), true);
  EXPECT_EQ(sub_graph_b0->GetGraphUnknownFlag(), true);
}

TEST_F(UtestDynamicShapePartition, TestSubGraphUnKnownShape) {
  const auto old_level = ge::SlogStub::GetInstance()->GetLevel();
  ge::SlogStub::GetInstance()->SetLevel(DLOG_INFO);
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_0) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE("branch_0_Node_Output", net_output));
  };

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b0 = ToComputeGraph(branch_0);
  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b0->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b0->GetName());
    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(1, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b0->SetParentNode(node_partitioned_call_op2);
  sub_graph_b0->SetParentGraph(root_graph);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b0), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  for (auto &subgraph : root_graph->GetAllSubgraphs()) {
    AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  }
  DynamicShapePartitioner partitioner(root_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(sub_graph_b0->GetGraphUnknownFlag(), true);
  EXPECT_EQ(sub_graph_b1->GetGraphUnknownFlag(), true);
  ge::SlogStub::GetInstance()->SetLevel(old_level);
}

TEST_F(UtestDynamicShapePartition, TestSubGraphForceUnKnownShape) {
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_0) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE("branch_0_Node_Output", net_output));
  };

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b0 = ToComputeGraph(branch_0);
  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b0->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b0->GetName());
    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(1, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b0->SetParentNode(node_partitioned_call_op2);
  sub_graph_b0->SetParentGraph(root_graph);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b0), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  auto myneg_op = sub_graph->FindFirstNodeMatchType("MyNeg");
  AttrUtils::SetBool(myneg_op->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);

  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitioner(root_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(sub_graph_b1->GetGraphUnknownFlag(), true);
}

TEST_F(UtestDynamicShapePartition, TestSubGraphHostCpuNode) {
  DEF_GRAPH(partitioned_call) {
    auto cond_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto neg = OP_CFG("MyNeg")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("partitioned_call_data", cond_data)->NODE("neg", neg)->NODE("partitioned_call_Node_Output",
                                                                           net_output));
  };
  auto sub_graph = ToComputeGraph(partitioned_call);

  DEF_GRAPH(branch_0) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE("branch_0_Node_Output", net_output));
  };

  DEF_GRAPH(branch_1) {
    auto data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto partitioned_call_op = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op");

    partitioned_call_op->RegisterSubgraphIrName("f", SubgraphType::kStatic);
    partitioned_call_op->AddSubgraphName(sub_graph->GetName());
    partitioned_call_op->SetSubgraphInstanceName(0, sub_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("branch_0_arg_0", data)->NODE(partitioned_call_op)->NODE("branch_0_Node_Output", net_output));
  };

  auto sub_graph_b0 = ToComputeGraph(branch_0);
  auto sub_graph_b1 = ToComputeGraph(branch_1);

  auto partitioned_call_node = sub_graph_b1->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(partitioned_call_node != nullptr);
  sub_graph->SetParentNode(partitioned_call_node);
  sub_graph->SetParentGraph(sub_graph_b1);

  DEF_GRAPH(graph1) {
    auto arg_branch_index = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto arg_value = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto unique_op = OP_CFG("Unique")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto partitioned_call_op2 = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16})
        .Build("partitioned_call_op2");

    partitioned_call_op2->AddSubgraphName(sub_graph_b0->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(0, sub_graph_b0->GetName());
    partitioned_call_op2->AddSubgraphName(sub_graph_b1->GetName());
    partitioned_call_op2->SetSubgraphInstanceName(1, sub_graph_b1->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("arg_branch_index", arg_branch_index)->NODE(partitioned_call_op2)->NODE("Node_Output", net_output));
    CHAIN(NODE("arg_value", arg_value)->NODE("unique_op", unique_op)->NODE(partitioned_call_op2));
  };

  auto root_graph = ToComputeGraph(graph1);
  auto node_partitioned_call_op2 = root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  EXPECT_TRUE(node_partitioned_call_op2 != nullptr);
  sub_graph_b0->SetParentNode(node_partitioned_call_op2);
  sub_graph_b0->SetParentGraph(root_graph);
  sub_graph_b1->SetParentNode(node_partitioned_call_op2);
  sub_graph_b1->SetParentGraph(root_graph);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b0), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph_b1), GRAPH_SUCCESS);
  EXPECT_EQ(root_graph->AddSubgraph(sub_graph), GRAPH_SUCCESS);

  auto myneg_op = sub_graph->FindFirstNodeMatchType("MyNeg");
  myneg_op->GetOpDesc()->SetOpEngineName("DNN_VM_HOST_CPU");

  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  for (auto &subgraph : root_graph->GetAllSubgraphs()) {
    AttrUtils::SetStr(*subgraph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  }
  DynamicShapePartitioner partitioner(root_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(sub_graph_b1->GetGraphUnknownFlag(), true);
}

/*******************************************************************************
 *                |
 *              Merge1
 *      Active /      \ Active
 *            /        \.
 *           /          \.
 *        Merge2         \.
 *  Active/   \Active     \.
 *       /     \           \.
 *     Add      Sub       Relu
 *      |        |          |
 *      |        |          |
 * Switch_f2  Switch_t2     |
 *       \      /           |
 *        \    /            |
 *         Less2            |
 *           |              |
 *           |              |
 *       Switch_f      Switch_t
 *           |   \      /   |
 *           |    Active    |
 *           |       |      |
 *           |     Less1    |
 *           |     /   \    |
 *           |    /     \   |
 *           | Data    Data |
 ******************************************************************************/

TEST_F(UtestDynamicShapePartition, merge_control_flow_group) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");

  auto data1 = NodeBuilder("data1", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto data2 = NodeBuilder("data2", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);

  auto less1 = NodeBuilder("less1", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto square0 = NodeBuilder("square0", SQUARE).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto square1 = NodeBuilder("square1", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto square2 = NodeBuilder("square2", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto square3 = NodeBuilder("square3", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active1 = NodeBuilder("active1", STREAMACTIVE).Build(graph);
  auto switch_t = NodeBuilder("switch_t", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto switch_f = NodeBuilder("switch_f", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto const_01 = NodeBuilder("const_01", CONSTANT).AddOutputDesc({1}).Build(graph);
  auto const_11 = NodeBuilder("const_11", CONSTANT).AddOutputDesc({1}).Build(graph);


  auto less2 = NodeBuilder("less2", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active2 = NodeBuilder("active2", STREAMACTIVE).Build(graph);
  auto switch_t2 = NodeBuilder("switch_t2", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto switch_f2 = NodeBuilder("switch_f2", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto const_02 = NodeBuilder("const_02", CONSTANT).AddOutputDesc({1}).Build(graph);
  auto const_12 = NodeBuilder("const_12", CONSTANT).AddOutputDesc({1}).Build(graph);

  auto add2 = NodeBuilder("add2", ADD).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto sub2 = NodeBuilder("sub2", SUB).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto merge2 = NodeBuilder("merge2", STREAMMERGE).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active_f2 = NodeBuilder("active_f2", STREAMACTIVE).Build(graph);
  auto active_t2 = NodeBuilder("active_t2", STREAMACTIVE).Build(graph);

  auto relu1 = NodeBuilder("relu1", RELU).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto merge1 = NodeBuilder("merge1", STREAMMERGE).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active_f1 = NodeBuilder("active_f1", STREAMACTIVE).Build(graph);
  auto active_t1 = NodeBuilder("active_t1", STREAMACTIVE).Build(graph);

  auto output1 = NodeBuilder("noutput1", NETOUTPUT).AddInputDesc({1}).Build(graph);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), square0->GetInDataAnchor(0));
  GraphUtils::AddEdge(square0->GetOutDataAnchor(0), square1->GetInDataAnchor(0));
  GraphUtils::AddEdge(square1->GetOutDataAnchor(0), square2->GetInDataAnchor(0));
  GraphUtils::AddEdge(square2->GetOutDataAnchor(0), square3->GetInDataAnchor(0));
  GraphUtils::AddEdge(square3->GetOutDataAnchor(0), switch_t->GetInDataAnchor(0));
  GraphUtils::AddEdge(square3->GetOutDataAnchor(0), switch_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_01->GetOutDataAnchor(0), switch_t->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_11->GetOutDataAnchor(0), switch_f->GetInDataAnchor(1));
  GraphUtils::AddEdge(square3->GetOutControlAnchor(), active1->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_t->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_f->GetInControlAnchor());


  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), less2->GetInDataAnchor(0));
  GraphUtils::AddEdge(square3->GetOutDataAnchor(0), less2->GetInDataAnchor(1));
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), switch_t2->GetInDataAnchor(0));
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), switch_f2->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_02->GetOutDataAnchor(0), switch_t2->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_12->GetOutDataAnchor(0), switch_f2->GetInDataAnchor(1));
  GraphUtils::AddEdge(less2->GetOutControlAnchor(), active2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_t2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_f2->GetInControlAnchor());


  GraphUtils::AddEdge(switch_f2->GetOutControlAnchor(), add2->GetInControlAnchor());
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), add2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add2->GetOutDataAnchor(0), merge2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add2->GetOutControlAnchor(), active_f2->GetInControlAnchor());
  GraphUtils::AddEdge(active_f2->GetOutControlAnchor(), merge2->GetInControlAnchor());

  GraphUtils::AddEdge(switch_t2->GetOutControlAnchor(), sub2->GetInControlAnchor());
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), sub2->GetInDataAnchor(0));
  GraphUtils::AddEdge(sub2->GetOutDataAnchor(0), merge2->GetInDataAnchor(1));
  GraphUtils::AddEdge(sub2->GetOutControlAnchor(), active_t2->GetInControlAnchor());
  GraphUtils::AddEdge(active_t2->GetOutControlAnchor(), merge2->GetInControlAnchor());

  GraphUtils::AddEdge(switch_t->GetOutControlAnchor(), less2->GetInControlAnchor());
  GraphUtils::AddEdge(switch_f->GetOutControlAnchor(), relu1->GetInControlAnchor());


  GraphUtils::AddEdge(merge2->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge2->GetOutControlAnchor(), active_f1->GetInControlAnchor());
  GraphUtils::AddEdge(active_f1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), relu1->GetInDataAnchor(1));
  GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu1->GetOutControlAnchor(), active_t1->GetInControlAnchor());
  GraphUtils::AddEdge(active_t1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  AttrUtils::SetBool(merge2->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  SetControlFlowGroup(merge2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_f2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_t2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(active2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(active_t2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(active_f2, merge2->GetOpDesc()->GetId());

  SetControlFlowGroup(merge1, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_f, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_t, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(active1, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(active_f1, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(active_t1, merge1->GetOpDesc()->GetId());

  EXPECT_EQ(graph->impl_->sub_graph_.size(), 0);
  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(graph->impl_->sub_graph_.size(), 3);   // input  less1  uknown
}

TEST_F(UtestDynamicShapePartition, mark_unknown_shape_nodes) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  gert::SpaceRegistryFaker().UpdateOpImplToDefaultSpaceRegistry();
  auto data = NodeBuilder("data", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto const_op = NodeBuilder("const", CONSTANT).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto const_op2 = NodeBuilder("const2", CONSTANT).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto add_tiling_depend = NodeBuilder("add", "AddTilingDepend").AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto add_tiling_depend2 = NodeBuilder("add2", "AddTilingDepend").AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto add_tiling_depend3 = NodeBuilder("add3", "AddTilingDependPlacementHasAicpu")
                                        .AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto where = NodeBuilder("where", WHERE).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto output = NodeBuilder("output", NETOUTPUT).AddInputDesc({1}).Build(graph);

  std::vector<int64_t> known_shape = {2, 5};
  data->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  const_op->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  ge::GeTensorDesc const_desc0(GeShape(known_shape), ge::FORMAT_ND, DT_INT32);
  uint8_t c_data[40] = {0};
  c_data[0] = 8;
  ge::ConstGeTensorPtr const_tensor =
          std::make_shared<GeTensor>(const_desc0, c_data, 40);
  ge::AttrUtils::SetTensor(const_op->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, const_tensor);
  const_op2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  ge::AttrUtils::SetTensor(const_op2->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, const_tensor);
  (void)ge::AttrUtils::SetBool(add_tiling_depend->GetOpDesc(), ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
  (void)ge::AttrUtils::SetBool(add_tiling_depend3->GetOpDesc(), ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
  add_tiling_depend->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(known_shape));
  add_tiling_depend->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape(known_shape));
  add_tiling_depend->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  add_tiling_depend2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(known_shape));
  add_tiling_depend2->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape(known_shape));
  add_tiling_depend2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  add_tiling_depend3->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(known_shape));
  add_tiling_depend3->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape(known_shape));
  add_tiling_depend3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  where->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(known_shape));
  std::vector<int64_t> unknown_shape = {-1, 2};
  where->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(unknown_shape));
  output->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(unknown_shape));

  GraphUtils::AddEdge(const_op->GetOutDataAnchor(0), add_tiling_depend->GetInDataAnchor(0));
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), add_tiling_depend->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_op2->GetOutDataAnchor(0), add_tiling_depend2->GetInDataAnchor(1));
  GraphUtils::AddEdge(add_tiling_depend->GetOutDataAnchor(0), add_tiling_depend2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_tiling_depend2->GetOutDataAnchor(0), add_tiling_depend3->GetInDataAnchor(0));
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), add_tiling_depend3->GetInDataAnchor(1));
  GraphUtils::AddEdge(add_tiling_depend3->GetOutDataAnchor(0), where->GetInDataAnchor(0));
  GraphUtils::AddEdge(where->GetOutDataAnchor(0), output->GetInDataAnchor(0));
  RecoverOpDescIrDefinition(add_tiling_depend->GetOpDesc(), "Add");
  RecoverOpDescIrDefinition(add_tiling_depend2->GetOpDesc(), "Add");
  RecoverOpDescIrDefinition(add_tiling_depend3->GetOpDesc(), "Add");
  GEThreadLocalContext &context = GetThreadLocalContext();
  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>(ge::TILING_SCHEDULE_OPTIMIZE, "1"));
  context.SetGlobalOption(options);
  for (int i = 0; i < 3; ++i) { // tiling depend node is 3
    RTS_STUB_RETURN_VALUE(rtGetDeviceCapability, rtError_t, RT_ERROR_NONE);
    RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_DEV_CAP_SUPPORT);
  }
  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.MarkUnknownShapeNodes(), SUCCESS);
  EXPECT_EQ(partitioner.unknown_shape_nodes_.size(), 4);
  options.clear();
  context.SetGlobalOption(options);
}

TEST_F(UtestDynamicShapePartition, mark_no_tiling_nodes) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

  auto data = NodeBuilder("data", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto where = NodeBuilder("where", WHERE).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto output = NodeBuilder("output", NETOUTPUT).AddInputDesc({1}).Build(graph);

  std::vector<int64_t> known_shape = {2, 5};
  data->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  where->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(known_shape));
  std::vector<int64_t> unknown_shape = {-1, 2};
  where->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(unknown_shape));
  output->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(unknown_shape));

  GraphUtils::AddEdge(data->GetOutDataAnchor(0), where->GetInDataAnchor(0));
  GraphUtils::AddEdge(where->GetOutDataAnchor(0), output->GetInDataAnchor(0));

  auto where_desc = where->GetOpDesc();
  vector<std::string> tiling_inline;
  vector<std::string> export_shape;
  AttrUtils::GetListStr(where_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
  tiling_inline.push_back("DNN_VM_AICPU");
  AttrUtils::SetListStr(where_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
  AttrUtils::GetListStr(where_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
  export_shape.push_back("DNN_VM_AICPU");
  AttrUtils::SetListStr(where_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
  where_desc->SetOpEngineName("DNN_VM_AICPU");
  AttrUtils::SetStr(where_desc, ATTR_NAME_OP_MAX_SHAPE, "10, 0");

  auto out_desc = output->GetOpDesc();

  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.MarkUnknownShapeNodes(), SUCCESS);
  EXPECT_EQ(partitioner.unknown_shape_nodes_.size(), 0);
  bool is_no_tiling = false;
  EXPECT_EQ(AttrUtils::GetBool(where_desc, ATTR_NAME_OP_NO_TILING, is_no_tiling), true);
  EXPECT_EQ(is_no_tiling, true);
  EXPECT_EQ(AttrUtils::GetBool(out_desc, ATTR_NAME_OP_NO_TILING, is_no_tiling), true);
  EXPECT_EQ(is_no_tiling, true);
}

TEST_F(UtestDynamicShapePartition, mark_unknown_and_no_tiling_nodes) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

  auto data = NodeBuilder("data", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto where = NodeBuilder("where", WHERE).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto where1 = NodeBuilder("where", WHERE).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto output = NodeBuilder("output", NETOUTPUT).AddInputDesc({1}).Build(graph);

  std::vector<int64_t> known_shape = {2, 5};
  std::vector<int64_t> unknown_shape = {-1, 2};
  data->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  where->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(known_shape));
  where->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(unknown_shape));
  where1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(unknown_shape));
  where1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(unknown_shape));
  output->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(unknown_shape));

  GraphUtils::AddEdge(data->GetOutDataAnchor(0), where->GetInDataAnchor(0));
  GraphUtils::AddEdge(where->GetOutDataAnchor(0), where1->GetInDataAnchor(0));
  GraphUtils::AddEdge(where1->GetOutDataAnchor(0), output->GetInDataAnchor(0));

  auto where_desc = where->GetOpDesc();
  vector<std::string> tiling_inline;
  vector<std::string> export_shape;
  AttrUtils::GetListStr(where_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
  tiling_inline.push_back("DNN_VM_AICPU");
  AttrUtils::SetListStr(where_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
  AttrUtils::GetListStr(where_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
  export_shape.push_back("DNN_VM_AICPU");
  AttrUtils::SetListStr(where_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
  where_desc->SetOpEngineName("DNN_VM_AICPU");
  AttrUtils::SetStr(where_desc, ATTR_NAME_OP_MAX_SHAPE, "10, 0");

  auto where1_desc = where1->GetOpDesc();

  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.MarkUnknownShapeNodes(), SUCCESS);
  EXPECT_EQ(partitioner.unknown_shape_nodes_.size(), 3);
  bool is_no_tiling = false;
  EXPECT_EQ(AttrUtils::GetBool(where_desc, ATTR_NAME_OP_NO_TILING, is_no_tiling), true);
  EXPECT_TRUE(is_no_tiling == false);
}

TEST_F(UtestDynamicShapePartition, test_node_support_no_tiling) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  DynamicShapePartitioner partitioner(graph);

  auto data = NodeBuilder("data", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto where = NodeBuilder("where", WHERE).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto output = NodeBuilder("output", NETOUTPUT).AddInputDesc({1}).Build(graph);

  GraphUtils::AddEdge(data->GetOutDataAnchor(0), where->GetInDataAnchor(0));
  GraphUtils::AddEdge(where->GetOutDataAnchor(0), output->GetInDataAnchor(0));

  auto where_desc = where->GetOpDesc();
  where_desc->SetOpEngineName("DNN_VM_AICPU");

  vector<std::string> tiling_inline;
  AttrUtils::GetListStr(where_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
  tiling_inline.push_back("DNN_VM_AICPU");
  AttrUtils::SetListStr(where_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);

  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(where), false);

  vector<std::string> export_shape;
  AttrUtils::GetListStr(where_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
  export_shape.push_back("DNN_VM_AICPU");
  AttrUtils::SetListStr(where_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
  std::vector<int64_t> known_shape = {2, 5};
  where->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(known_shape));
  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(where), true);
  
  std::vector<int64_t> unknown_shape = {-1, 2};
  where->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(unknown_shape));
  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(where), false);

  std::vector<std::pair<int64_t, int64_t>> range = {{1, -1}, {2, 2}};
  where->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(range);
  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(where), false);

  std::vector<int64_t> unknown_rank_shape = {-2};
  where->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(unknown_rank_shape));
  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(where), false);
}

TEST_F(UtestDynamicShapePartition, test_node_support_no_tiling_01) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  DynamicShapePartitioner partitioner(graph);

  auto memcpyasync = NodeBuilder("memcpyasync", MEMCPYASYNC).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);

  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(memcpyasync), true);

  auto data = NodeBuilder("data", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto data_desc = data->GetOpDesc();
  AttrUtils::SetBool(data_desc, ATTR_NAME_OP_NO_TILING, true);
  EXPECT_EQ(partitioner.IsNodeSupportNoTiling(data), true);
}

TEST_F(UtestDynamicShapePartition, special_process_resource_op)
{
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto add_3 = OP_CFG(ADD);
  auto add_4 = OP_CFG(ADD).Attr("_force_unknown_shape", true);
  auto add_5 = OP_CFG(ADD).Attr("_force_unknown_shape", true);
  auto add_6 = OP_CFG(ADD).Attr("_force_unknown_shape", true);
  auto stack = OP_CFG("Stack");
  auto stackpush = OP_CFG("StackPush");
  auto stackpop = OP_CFG("StackPop");
  auto stack1 = OP_CFG("Stack");
  auto stackpush1 = OP_CFG("StackPush").Attr("_force_unknown_shape", true);
  auto stackpop1 = OP_CFG("StackPop").Attr("_force_unknown_shape", true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto square0 = OP_CFG(SQUARE);
  auto square1 = OP_CFG(SQUARE);
  auto square2 = OP_CFG(SQUARE);
  auto square3 = OP_CFG(SQUARE);
  auto square4 = OP_CFG(SQUARE);
  auto op_ptr = OP_CFG(DATA)
    .InCnt(1)
    .OutCnt(1)
    .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
    .Attr("compile_info_key", "ddd")
    .Attr("compile_info_json", "cccc")
    .Attr("_force_unknown_shape", true)
    .Build("data3");
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)
          ->NODE("add_2", add_2)->EDGE(0, 0)->NODE("add_3", add_3)
          ->NODE("add_4", add_4)->EDGE(0, 0)->NODE("add_5", add_5)
          ->NODE("add_6", add_6));

    CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE(op_ptr)->EDGE(0, 1)->NODE("add_4", add_4));
    CHAIN(NODE(op_ptr)->EDGE(0, 1)->NODE("add_5", add_5));

    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackpush", stackpush));
    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackpop", stackpop));
    CHAIN(NODE("add_1", add_1)->EDGE(0, 1)->NODE("stackpush", stackpush));
    CHAIN(NODE("stackpop", stackpop)->EDGE(0, 1)->NODE("add_3", add_3));
    CHAIN(NODE("square0", square0)
              ->EDGE(0, 0)
              ->NODE("square1", square1)
              ->NODE("square2", square2)
              ->NODE("square3", square3)
              ->NODE("stack1", stack));
    CHAIN(NODE("stack1", stack)->EDGE(0, 0)->NODE("stackpush1", stackpush1));
    CHAIN(NODE("stack1", stack)->EDGE(0, 0)->NODE("stackpop1", stackpop1));
    CHAIN(NODE("add_4", add_4)->EDGE(0, 1)->NODE("stackpush1", stackpush1));
    CHAIN(NODE("stackpop1", stackpop1)->EDGE(0, 1)->NODE("add_6", add_6));

  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
   for (auto &node : graph->GetAllNodes()) {
     if (node->GetName() == "stack" || node->GetName() == "stackpush" || node->GetName() == "stackpop") {
       (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, 1);
     }
     if (node->GetName() == "stack1" || node->GetName() == "stackpush1" || node->GetName() == "stackpop1") {
       (void)AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_DATA_FLOW_HANDLE, 2);
     }

   }
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);

  bool forced_unknown = false;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "stack1") {
      AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown);
    }
  }
  EXPECT_EQ(forced_unknown, true);
}

TEST_F(UtestDynamicShapePartition, merge_known_shape_first_success) {
  auto dyn_1 = OP_CFG(PACK);
  auto dyn_2 = OP_CFG(ADD);
  auto static_1 = OP_CFG(RELU).Attr("_user_stream_label", "11");
  auto static_2 = OP_CFG(RELU).Attr("_user_stream_label", "22");
  auto static_3 = OP_CFG(RELU);
  auto static_4 = OP_CFG(RELU);
  auto static_5 = OP_CFG(RELU);

  DEF_GRAPH(g1) {
                  CHAIN(NODE("dyn_1", dyn_1)->EDGE(0, 0)->NODE("static_1", static_1)->EDGE(0, 0)
                            ->NODE("static_2", static_2)->EDGE(0, 0)->NODE("static_3", static_3)
                            ->NODE("static_4", static_4)->EDGE(0, 0)->NODE("static_5", static_5)
                            ->NODE("dyn_2", dyn_2));
                  CHAIN(NODE("dyn_1", dyn_1)->EDGE(0, 1)->NODE("dyn_2", dyn_2));
                };
  ComputeGraphPtr graph = ToComputeGraph(g1);

  for (auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "dyn_1") {
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 224, 224}));
    }
    if (node->GetName() == "dyn_2") {
      node->GetOpDescBarePtr()->MutableInputDesc(1)->SetShape(GeShape({-1, -1, 224, 224}));
    }
  }
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitioner(graph, true);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
}

TEST_F(UtestDynamicShapePartition, merge_small_known_shape_to_unknown_success) {
  auto dyn_1 = OP_CFG(PACK);
  auto dyn_2 = OP_CFG(ADD);
  auto static_1 = OP_CFG(RELU);
  auto static_2 = OP_CFG(RELU);

  DEF_GRAPH(g1) {
    CHAIN(NODE("static_1", static_1)->EDGE(0, 0)->NODE("dyn_1", dyn_1)->EDGE(0, 0)
              ->NODE("dyn_2", dyn_2)->NODE("static_2", static_2));
    CHAIN(NODE("dyn_1", dyn_1)->EDGE(0, 1)->NODE("dyn_2", dyn_2));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);

  for (auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "dyn_1") {
      node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape({-1, -1, 224, 224}));
    }
    if (node->GetName() == "dyn_2") {
      node->GetOpDescBarePtr()->MutableInputDesc(1)->SetShape(GeShape({-1, -1, 224, 224}));
    }
  }
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitior(graph, false);
  EXPECT_EQ(partitior.Partition(), SUCCESS);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  EXPECT_EQ(graph->GetDirectNodesSize(), 1);
  auto s0 = graph->GetDirectNode().at(0);
  bool is_unknown_shape = false;
  EXPECT_TRUE(AttrUtils::GetBool(s0->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape));
  EXPECT_EQ(is_unknown_shape, true);
}

TEST_F(UtestDynamicShapePartition, run_on_host_return_sucess) {
  std::map<std::string, std::string> config;
  config["ge.exec.placement"] = "HOST";
  ge::GetThreadLocalContext().SetGraphOption(config);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root");
  NodePtr node1_root =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(root_graph);
  NodePtr add_n_node_root =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(root_graph);
  NodePtr node2_root =
      NodeBuilder("node2", RELU).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(root_graph);

  AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "123");

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({-1, -1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node4 =
      NodeBuilder("node4", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({-1, -1, 224, 224}).Build(graph);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutControlAnchor(), node2->GetInControlAnchor());
  GraphUtils::AddEdge(add_n_node->GetOutControlAnchor(), node4->GetInControlAnchor());
  GraphUtils::AddEdge(node4->GetOutControlAnchor(), node2->GetInControlAnchor());


  GraphUtils::AddEdge(node1_root->GetOutDataAnchor(0), add_n_node_root->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node_root->GetOutDataAnchor(0), node2_root->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node_root->GetOutControlAnchor(), node2_root->GetInControlAnchor());

  root_graph->AddSubGraph(graph);
  DynamicShapePartitioner partitioner(root_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

/*
 *     relu
 *      |
 *     add
 *      |
 *   constant
 */
TEST_F(UtestDynamicShapePartition, host_scheduling_dynamic) {
  std::map<std::string, std::string> options;
  options["ge.exec.hostSchedulingMaxThreshold"] = "15";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(graph->GetGraphUnknownFlag(), true);
  GetThreadLocalContext().SetGraphOption({});
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

/*
 *     relu
 *      |
 *     add
 *      |
 *   constant
 */
TEST_F(UtestDynamicShapePartition, host_scheduling_static) {
  std::map<std::string, std::string> options;
  options["ge.exec.hostSchedulingMaxThreshold"] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(graph->GetGraphUnknownFlag(), false);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestDynamicShapePartition, partition_unknown_graph_only_contain_data_netoutput) {
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto data2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  auto data1 = graph->FindNode("data1");
  ASSERT_NE(data1, nullptr);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  auto net_output = graph->FindNode("net_output");
  ASSERT_NE(net_output, nullptr);
  net_output->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({-1}));

  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitior(graph, false);
  EXPECT_EQ(partitior.Partition(), SUCCESS);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 0);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  const char *const kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
  bool is_owner_graph_unknown = false;
  EXPECT_EQ(AttrUtils::GetBool(data1->GetOpDesc(), kOwnerGraphIsUnknown, is_owner_graph_unknown), true);
  EXPECT_EQ(is_owner_graph_unknown, true);
  EXPECT_EQ(graph->GetGraphUnknownFlag(), true);
}

TEST_F(UtestDynamicShapePartition, partition_unknown_graph_set_static_model_ops_lower_limit) {
  std::map<std::string, std::string> options;
  options[OPTION_STATIC_MODEL_OPS_LOWER_LIMIT] = "-1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto neg = OP_CFG(NEG)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("data1", data1)->NODE("neg", neg)->NODE("net_output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  auto neg_node = graph->FindNode("neg");
  ASSERT_NE(neg_node, nullptr);

  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  std::string option_value;
  ge::GEThreadLocalContext().GetOption(OPTION_STATIC_MODEL_OPS_LOWER_LIMIT, option_value);
  std::cout << "option_value=" << option_value << std::endl;
  DynamicShapePartitioner partitioner(graph, false);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 0);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  const char *const kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
  bool is_owner_graph_unknown = false;
  EXPECT_EQ(AttrUtils::GetBool(neg_node->GetOpDesc(), kOwnerGraphIsUnknown, is_owner_graph_unknown), true);
  EXPECT_EQ(is_owner_graph_unknown, true);
  EXPECT_EQ(graph->GetGraphUnknownFlag(), true);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestDynamicShapePartition, partition_unknown_graph_set_static_model_ops_lower_limit_10) {
  std::map<std::string, std::string> options;
  options[OPTION_STATIC_MODEL_OPS_LOWER_LIMIT] = "10";
  ge::GetThreadLocalContext().SetGraphOption(options);
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto neg = OP_CFG(NEG)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("data1", data1)->NODE("neg", neg)->NODE("net_output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  auto neg_node = graph->FindNode("neg");
  ASSERT_NE(neg_node, nullptr);

  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  std::string option_value;
  ge::GEThreadLocalContext().GetOption(OPTION_STATIC_MODEL_OPS_LOWER_LIMIT, option_value);
  std::cout << "option_value=" << option_value << std::endl;
  DynamicShapePartitioner partitioner(graph, false);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);
  const char *const kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
  bool is_owner_graph_unknown = false;
  EXPECT_EQ(AttrUtils::GetBool(neg_node->GetOpDesc(), kOwnerGraphIsUnknown, is_owner_graph_unknown), true);
  EXPECT_EQ(is_owner_graph_unknown, true);
  EXPECT_EQ(graph->GetGraphUnknownFlag(), true);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestDynamicShapePartition, partition_unknown_graph_set_invalid_static_model_ops_lower_limit) {
  std::map<std::string, std::string> options;
  options[OPTION_STATIC_MODEL_OPS_LOWER_LIMIT] = "";
  ge::GetThreadLocalContext().SetGraphOption(options);
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    auto neg = OP_CFG(NEG)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("data1", data1)->NODE("neg", neg)->NODE("net_output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  auto neg_node = graph->FindNode("neg");
  ASSERT_NE(neg_node, nullptr);

  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  std::string option_value;
  ge::GEThreadLocalContext().GetOption(OPTION_STATIC_MODEL_OPS_LOWER_LIMIT, option_value);
  std::cout << "option_value=" << option_value << std::endl;
  DynamicShapePartitioner partitioner(graph, false);
  EXPECT_NE(partitioner.Partition(), SUCCESS);
  GetThreadLocalContext().SetGraphOption({});
}

TEST_F(UtestDynamicShapePartition, partition_mark_support_addr_refresh) {
  constexpr char_t kIsSupportAddrRefresh[] = "_is_support_addr_refresh";
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_INT32, {10,10});

    auto neg = OP_CFG(NEG)
        .InCnt(1)
        .OutCnt(1)
        .Attr(kIsSupportAddrRefresh, false)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10,10})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto neg1 = OP_CFG(NEG)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {10,10})
        .Attr(ATTR_NAME_STREAM_LABEL, "aaa");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    CHAIN(NODE("data1", data1)->NODE("neg", neg)->NODE("neg1", neg1)->NODE("net_output", net_output));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  auto neg_node = graph->FindNode("neg");
  ASSERT_NE(neg_node, nullptr);

  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  DynamicShapePartitioner partitioner(graph, false);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  bool is_force_unknown = false;
  EXPECT_EQ(AttrUtils::GetBool(neg_node->GetOpDescBarePtr(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, is_force_unknown), true);
  EXPECT_EQ(is_force_unknown, true);
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);
  ASSERT_EQ(graph->GetDirectNodePtr().size(), 3);
  bool is_unknown = false;
  EXPECT_EQ(
      AttrUtils::GetBool(graph->GetDirectNodePtr().at(1)->GetOpDescBarePtr(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown),
      true);
  EXPECT_EQ(is_unknown, true);
}

ComputeGraphPtr BuildCaseSubGraph(const bool is_multi_batch) {
  auto main_graph = std::make_shared<ge::ComputeGraph>("multi_batch_graph");
  auto data = gert::NodeBuilder("data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 0).Output().Build(main_graph);
  auto data2 = gert::NodeBuilder("data2", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 1).Output().Build(main_graph);
  auto data3 = gert::NodeBuilder("data3", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 2).Output().Build(main_graph);
  auto data4 = gert::NodeBuilder("data4", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 3).Output().Build(main_graph);
  auto data_shape = gert::NodeBuilder("shape_data", ge::DATA).Attr(ge::ATTR_NAME_INDEX, 4).Output().Build(main_graph);
  GeTensor weight;
  //std::vector<uint8_t> data = {2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<int32_t> const_data = {8, 4, 4, 0, 0, 0, 0, 0,
                                     100, 100, 10, 0, 0, 0, 0, 0,
                                     8, 4, 4, 0, 0, 0, 0, 0,
                                     100, 100, 10, 0, 0, 0, 0, 0,
                                     100, 100, 10, 0, 0, 0, 0, 0};

  weight.SetData((uint8_t *)const_data.data(), const_data.size() * sizeof(int32_t));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({40}));
  weight_desc.SetOriginShape(GeShape({40}));
  weight.SetTensorDesc(weight_desc);

  auto shape_const = gert::NodeBuilder("shape_const", ge::CONSTANT).Attr("value", weight).Output().Build(main_graph);
  auto mapIndex = gert::NodeBuilder("mapIndex", "MapIndex").Input(data_shape).Input(shape_const).Output().Build(main_graph);

  auto sub_builder_case = [](uint32_t index) {
    std::string graph_name = "branch" + std::to_string(index); 
    auto graph = std::make_shared<ge::ComputeGraph>(graph_name);
    auto data = gert::NodeBuilder("data_branch" + std::to_string(index), ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 0)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 1)
                    .Output()
                    .Build(graph);
    auto data2 = gert::NodeBuilder("data2_branch" + std::to_string(index), ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 1)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 2)
                    .Output()
                    .Build(graph);
    auto data3 = gert::NodeBuilder("data3_branch" + std::to_string(index), ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 2)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 3)
                    .Output()
                    .Build(graph);
    auto data4 = gert::NodeBuilder("data4_branch" + std::to_string(index), ge::DATA)
                    .Attr(ge::ATTR_NAME_INDEX, 3)
                    .Attr(ge::ATTR_NAME_PARENT_NODE_INDEX, 4)
                    .Output()
                    .Build(graph);
    auto add1 = gert::NodeBuilder("add1_branch" + std::to_string(index), ADD)
                                  .Input(data).Input(data2).Output().Build(graph);
    auto add2 = gert::NodeBuilder("add2_branch" + std::to_string(index), ADD)
                                  .Input(add1).Input(data3).Output().Build(graph); 
    auto add3 = gert::NodeBuilder("add3_branch" + std::to_string(index), "AddTilingDepend")
                                  .Input(add2).Input(data4).Output().Build(graph);
    RecoverOpDescIrDefinition(add3->GetOpDesc(), "Add");
    auto output = gert::NodeBuilder("output_branch" + std::to_string(index), ge::NETOUTPUT).Input(add3).Build(graph);
    graph->SetGraphID(20);
    AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
    return graph;
  };

  auto case_node = gert::NodeBuilder("case", "Case")
          .Input(mapIndex).Input(data).Input(data2).Input(data3).Input(data4).Output()
          .Output()
          .Attr("batch0", sub_builder_case(0))
          .Attr("batch1", sub_builder_case(1))
          .AttrBool(ATTR_INSERT_BY_MBATCH, is_multi_batch)
          .Build(main_graph);
  auto output = gert::NodeBuilder("output", ge::NETOUTPUT).Input(case_node).Build(main_graph);
  AttrUtils::SetStr(*main_graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");
  (void)ge::AttrUtils::SetBool(*main_graph, "_enable_dynamic_batch", is_multi_batch);
  main_graph->SetGraphID(20);
  return main_graph;
}

TEST_F(UtestDynamicShapePartition, partition_multi_batch_graph) {
  auto compute_graph = BuildCaseSubGraph(true);
  DynamicShapePartitioner partitior(compute_graph, false);
  EXPECT_EQ(partitior.Partition(), SUCCESS);
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 6);
  EXPECT_EQ(compute_graph->GetGraphUnknownFlag(), true);
  auto compute_graph_not_multi_batch = BuildCaseSubGraph(false);
  DynamicShapePartitioner partitior2(compute_graph_not_multi_batch, false);
  EXPECT_EQ(partitior2.Partition(), SUCCESS);
  EXPECT_EQ(compute_graph_not_multi_batch->GetAllSubgraphs().size(), 4);
  EXPECT_EQ(compute_graph_not_multi_batch->GetGraphUnknownFlag(), true);
}

TEST_F(UtestDynamicShapePartition, not_single_op_scene_success_stable_topo_bfs) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  vector<float> value{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1, 1, 2, 2}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(value.data()), sizeof(float) * value.size());
  auto const3 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).Weight(const_tensor1);
  auto const2 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).Weight(const_tensor1);
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast4 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu7 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu8 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu9 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu10 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu11 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu12 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});

  auto add1 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto add2 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto add3 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto variable1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto ref_data = OP_CFG(REFDATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2}).Attr("ref_var_src_var_name", "variable1");
  auto add4 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1)->EDGE(0, 0)->
        NODE("relu1", relu1)->EDGE(0, 0)->NODE("relu2", relu2)->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("data1", data0)->EDGE(0, 0)->NODE("cast2", cast2)->EDGE(0, 0)->
        NODE("relu4", relu4)->EDGE(0, 0)->NODE("relu5", relu5)->EDGE(0, 0)->NODE("relu6", relu6)->
        EDGE(0, 0)->NODE("add1", add1)->EDGE(0, 0)->NODE("add3", add3));
    CHAIN(NODE("const2", const2)->EDGE(0, 0)->NODE("cast3", cast3)->EDGE(0, 0)->
        NODE("relu7", relu7)->EDGE(0, 0)->NODE("relu8", relu8)->EDGE(0, 0)->NODE("relu9", relu9));
    CHAIN(NODE("const3", const3)->EDGE(0, 0)->NODE("cast4", cast4)->EDGE(0, 0)->
        NODE("relu10", relu10)->EDGE(0, 0)->NODE("relu11", relu11)->EDGE(0, 0)->NODE("relu12", relu12)->
        EDGE(0, 0)->NODE("add2", add2)->EDGE(0, 1)->NODE("add3", add3));
    CHAIN(NODE("relu3")->EDGE(0, 1)->NODE("add1"));
    CHAIN(NODE("relu9")->EDGE(0, 1)->NODE("add2"));
    CHAIN(NODE("variable1", variable1)->NODE("ref_data", ref_data));
    CHAIN(NODE("ref_data")->EDGE(0, 0)->NODE("add4", add4));
    CHAIN(NODE("add3")->EDGE(0, 1)->NODE("add4"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  // bfs排序
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ReInitOo();
  compute_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ReInitOo();
  DynamicShapePartitioner partitioner(compute_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 4);
  std::map<std::string, std::vector<std::string>> subgraph_to_node = {
      {"g1_sub_0_unknow", {"cast1", "relu1", "relu2", "relu3", "cast2", "relu4", "relu5", "relu6", "add1"}},
      {"g1_sub_1_input", {"const2", "const3"}},
      {"g1_sub_2_know", {"cast3", "relu7", "relu8", "relu9", "cast4", "relu10", "relu11", "relu12", "add2"}},
      {"g1_sub_3_unknow", {"add3", "variable1", "add4", "ref_data"}}};

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == PARTITIONEDCALL) {
      auto subgraph = compute_graph->GetSubgraph(node->GetOpDesc()->GetSubgraphInstanceName(0U));
      ASSERT_NE(subgraph, nullptr);
      EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
      std::vector<std::string> actual_node_name;
      for (const auto &sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          actual_node_name.push_back(sub_node->GetName());
        }
      }
      EXPECT_EQ(actual_node_name.size(), subgraph_to_node[subgraph->GetName()].size());
      for (size_t i = 0UL; i < actual_node_name.size(); i++) {
        std::cout << "sub graph node name: " << actual_node_name[i] << std::endl;
        EXPECT_EQ(subgraph_to_node[subgraph->GetName()][i], actual_node_name[i]);
      }
    }
  }
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDynamicShapePartition, not_single_op_scene_success_stable_topo_bfs2) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data2 = OP_DATA(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto addn = OP_CFG(ADDN).InCnt(3).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("cast2", cast2));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("cast3", cast3));
    CHAIN(NODE("cast1")->EDGE(0, 0)->NODE("relu1", relu1));
    CHAIN(NODE("cast2")->EDGE(0, 0)->NODE("relu2", relu2));
    CHAIN(NODE("cast3")->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("addn", addn));
    CHAIN(NODE("relu2")->EDGE(0, 1)->NODE("addn", addn));
    CHAIN(NODE("relu3")->EDGE(0, 2)->NODE("addn", addn));    
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  // bfs排序
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
  compute_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DynamicShapePartitioner partitioner(compute_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  std::map<std::string, std::vector<std::string>> subgraph_to_node = {
      {"g1_sub_0_unknow", {"cast1", "relu1", "cast2", "relu2", "cast3", "relu3", "addn"}}};

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == PARTITIONEDCALL) {
      auto subgraph = compute_graph->GetSubgraph(node->GetOpDesc()->GetSubgraphInstanceName(0U));
      ASSERT_NE(subgraph, nullptr);
      EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
      std::vector<std::string> actual_node_name;
      for (const auto &sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          actual_node_name.push_back(sub_node->GetName());
        }
      }
      EXPECT_EQ(actual_node_name.size(), subgraph_to_node[subgraph->GetName()].size());
      for (size_t i = 0UL; i < actual_node_name.size(); i++) {
        std::cout << "sub graph node name: " << actual_node_name[i] << std::endl;
        EXPECT_EQ(subgraph_to_node[subgraph->GetName()][i], actual_node_name[i]);
      }
    }
  }
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDynamicShapePartition, not_single_op_scene_success_stable_topo_dfs) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data2 = OP_DATA(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto addn = OP_CFG(ADDN).InCnt(3).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("cast2", cast2));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("cast3", cast3));
    CHAIN(NODE("cast1")->EDGE(0, 0)->NODE("relu1", relu1));
    CHAIN(NODE("cast2")->EDGE(0, 0)->NODE("relu2", relu2));
    CHAIN(NODE("cast3")->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("addn", addn));
    CHAIN(NODE("relu2")->EDGE(0, 1)->NODE("addn", addn));
    CHAIN(NODE("relu3")->EDGE(0, 2)->NODE("addn", addn));    
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  // bfs排序
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  compute_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DynamicShapePartitioner partitioner(compute_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  compute_graph->TopologicalSortingGraph();
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  std::map<std::string, std::vector<std::string>> subgraph_to_node = {
      {"g1_sub_0_unknow", {"cast1", "relu1", "cast2", "relu2", "cast3", "relu3", "addn"}}};

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == PARTITIONEDCALL) {
      auto subgraph = compute_graph->GetSubgraph(node->GetOpDesc()->GetSubgraphInstanceName(0U));
      ASSERT_NE(subgraph, nullptr);
      EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
      std::vector<std::string> actual_node_name;
      for (const auto &sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          actual_node_name.push_back(sub_node->GetName());
        }
      }
      EXPECT_EQ(actual_node_name.size(), subgraph_to_node[subgraph->GetName()].size());
      for (size_t i = 0UL; i < actual_node_name.size(); i++) {
        std::cout << "sub graph node name: " << actual_node_name[i] << std::endl;
        EXPECT_EQ(subgraph_to_node[subgraph->GetName()][i], actual_node_name[i]);
      }
    }
  }
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDynamicShapePartition, not_single_op_scene_success_stable_topo_rdfs) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data2 = OP_DATA(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto addn = OP_CFG(ADDN).InCnt(3).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("cast2", cast2));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("cast3", cast3));
    CHAIN(NODE("cast1")->EDGE(0, 0)->NODE("relu1", relu1));
    CHAIN(NODE("cast2")->EDGE(0, 0)->NODE("relu2", relu2));
    CHAIN(NODE("cast3")->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("addn", addn));
    CHAIN(NODE("relu2")->EDGE(0, 1)->NODE("addn", addn));
    CHAIN(NODE("relu3")->EDGE(0, 2)->NODE("addn", addn));    
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  // bfs排序
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  compute_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  DynamicShapePartitioner partitioner(compute_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  compute_graph->TopologicalSortingGraph();
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  std::map<std::string, std::vector<std::string>> subgraph_to_node = {
      {"g1_sub_0_unknow", {"cast1", "relu1", "cast2", "relu2", "cast3", "relu3", "addn"}}};

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == PARTITIONEDCALL) {
      auto subgraph = compute_graph->GetSubgraph(node->GetOpDesc()->GetSubgraphInstanceName(0U));
      ASSERT_NE(subgraph, nullptr);
      EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
      std::vector<std::string> actual_node_name;
      for (const auto &sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          actual_node_name.push_back(sub_node->GetName());
        }
      }
      EXPECT_EQ(actual_node_name.size(), subgraph_to_node[subgraph->GetName()].size());
      for (size_t i = 0UL; i < actual_node_name.size(); i++) {
        std::cout << "sub graph node name: " << actual_node_name[i] << std::endl;
        EXPECT_EQ(subgraph_to_node[subgraph->GetName()][i], actual_node_name[i]);
      }
    }
  }
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestDynamicShapePartition, not_single_op_scene_success_stable_topo_center_static) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data2 = OP_DATA(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu5 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu6 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto addn = OP_CFG(ADDN).InCnt(3).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("cast2", cast2));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("cast3", cast3));
    CHAIN(NODE("cast1")->EDGE(0, 0)->NODE("relu1", relu1));
    CHAIN(NODE("cast2")->EDGE(0, 0)->NODE("relu2", relu2));
    CHAIN(NODE("cast3")->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("addn", addn));
    CHAIN(NODE("relu2")->EDGE(0, 1)->NODE("addn", addn));
    CHAIN(NODE("relu3")->EDGE(0, 2)->NODE("addn", addn));    
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  // bfs排序
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ReInitOo();
  compute_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ReInitOo();
  DynamicShapePartitioner partitioner(compute_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  compute_graph->TopologicalSortingGraph();
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  std::map<std::string, std::vector<std::string>> subgraph_to_node = {
      {"g1_sub_0_unknow", {"cast1", "relu1", "cast3", "relu3", "cast2", "relu2", "addn"}}};

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == PARTITIONEDCALL) {
      auto subgraph = compute_graph->GetSubgraph(node->GetOpDesc()->GetSubgraphInstanceName(0U));
      ASSERT_NE(subgraph, nullptr);
      EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
      std::vector<std::string> actual_node_name;
      for (const auto &sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          actual_node_name.push_back(sub_node->GetName());
        }
      }
      EXPECT_EQ(actual_node_name.size(), subgraph_to_node[subgraph->GetName()].size());
      for (size_t i = 0UL; i < actual_node_name.size(); i++) {
        std::cout << "sub graph node name: " << actual_node_name[i] << std::endl;
        EXPECT_EQ(subgraph_to_node[subgraph->GetName()][i], actual_node_name[i]);
      }
    }
  }
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
}
TEST_F(UtestDynamicShapePartition, not_single_op_scene_success_stable_topo_tail_static_3) {
  auto data0 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data1 = OP_DATA(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto data2 = OP_DATA(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast1 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto cast3 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto relu3 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, -1, 2, 2});
  auto addn = OP_CFG(ADDN).InCnt(3).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto cast4 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  auto relu4 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 2, 2});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("cast1", cast1));
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("cast2", cast2));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("cast3", cast3));
    CHAIN(NODE("cast1")->EDGE(0, 0)->NODE("relu1", relu1));
    CHAIN(NODE("cast2")->EDGE(0, 0)->NODE("relu2", relu2));
    CHAIN(NODE("cast3")->EDGE(0, 0)->NODE("relu3", relu3));
    CHAIN(NODE("relu1")->EDGE(0, 0)->NODE("addn", addn));
    CHAIN(NODE("relu2")->EDGE(0, 1)->NODE("addn", addn));
    CHAIN(NODE("relu3")->EDGE(0, 2)->NODE("addn", addn));
    CHAIN(NODE("addn")->EDGE(0, 0)->NODE("cast4", cast4)->NODE("relu4", relu4));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void) AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  // bfs排序
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ReInitOo();
  compute_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ReInitOo();
  DynamicShapePartitioner partitioner(compute_graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  compute_graph->TopologicalSortingGraph();
  EXPECT_EQ(compute_graph->GetAllSubgraphs().size(), 1);
  std::map<std::string, std::vector<std::string>> subgraph_to_node = {
      {"g1_sub_0_unknow", {"cast1", "relu1", "cast2", "relu2", "cast3", "relu3", "addn", "cast4", "relu4"}}};

  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == PARTITIONEDCALL) {
      auto subgraph = compute_graph->GetSubgraph(node->GetOpDesc()->GetSubgraphInstanceName(0U));
      ASSERT_NE(subgraph, nullptr);
      EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
      std::vector<std::string> actual_node_name;
      for (const auto &sub_node : subgraph->GetDirectNode()) {
        if (sub_node->GetType() != DATA && sub_node->GetType() != NETOUTPUT) {
          actual_node_name.push_back(sub_node->GetName());
        }
      }
      EXPECT_EQ(actual_node_name.size(), subgraph_to_node[subgraph->GetName()].size());
      for (size_t i = 0UL; i < actual_node_name.size(); i++) {
        std::cout << "sub graph node name: " << actual_node_name[i] << std::endl;
        EXPECT_EQ(subgraph_to_node[subgraph->GetName()][i], actual_node_name[i]);
      }
    }
  }
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
}
} // namespace ge
