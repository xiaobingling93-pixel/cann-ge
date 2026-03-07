/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include <gtest/gtest.h>
#include "macro_utils/dt_public_scope.h"
#include "compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/graph_utils_ex.h"
#include <gmock/gmock.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_utils.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph/partition/engine_partitioner.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "api/gelib/gelib.h"
#include "common/share_graph.h"
#include "ge/ge_api.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/attribute_group/attr_group_shape_env.h"

namespace ge {
namespace airut {

class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  NodePtr AddNode(const std::string &name, const std::string &type,
                  std::initializer_list<std::string> input_names,
                  std::initializer_list<std::string> output_names,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};

NodePtr GraphBuilder::AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt, Format format,
                              DataType data_type, std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginFormat(format);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }

  return graph_->AddNode(op_desc);
}
void GraphBuilder::AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx) {
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
}
void GraphBuilder::AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
  GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
}
NodePtr GraphBuilder::AddNode(const string &name, const string &type, std::initializer_list<std::string> input_names,
                              std::initializer_list<std::string> output_names, Format format, DataType data_type,
                              std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  tensor_desc->SetOriginFormat(format);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (auto &input_name : input_names) {
    op_desc->AddInputDesc(input_name, tensor_desc->Clone());
  }
  for (auto &output_name :output_names) {
    op_desc->AddOutputDesc(output_name, tensor_desc->Clone());
  }

  return graph_->AddNode(op_desc);
}

}  // namespace airut
namespace {
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

/*                                  -------------------------
*                                  |  partitioncall_0_const1* |
*     partitioncall_0--------------|             |           |
*           |                      |          netoutput      |
*           |                      --------------------------
*           |                       ------------------         -------------
*           |                      |        data      |       |    data     |
*           |                      |          |       |       |     |       |
*     partitioncall_1--------------|        case -----|-------|   squeeze*  |
*                                  |          |       |       |     |       |
*                                  |      netoutput   |       |  netoutput  |
*                                   ------------------         -------------
*/
ComputeGraphPtr BuildGraphPartitionCall() {
  auto root_builder = airut::GraphBuilder("root");
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 0, 1);
  const auto &partitioncall_1 = root_builder.AddNode("partitioncall_1", PARTITIONEDCALL, 1, 1);
  root_builder.AddDataEdge(partitioncall_0, 0, partitioncall_1, 0);
  const auto &root_graph = root_builder.GetGraph();

  // 1.build partitioncall_0 sub graph
  auto p1_sub_builder = airut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
  const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);
  p1_sub_builder.AddDataEdge(partitioncall_0_const1, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = p1_sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("f");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  // 2.build partitioncall_1 sub graph
  auto p2_sub_builder = airut::GraphBuilder("partitioncall_1_sub");
  const auto &partitioncall_1_data = p2_sub_builder.AddNode("partitioncall_1_data", DATA, 0, 1);
  AttrUtils::SetInt(partitioncall_1_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &partitioncall_1_case = p2_sub_builder.AddNode("partitioncall_1_case", "Case", 1, 1);
  const auto &partitioncall_1_netoutput = p2_sub_builder.AddNode("partitioncall_1_netoutput", NETOUTPUT, 1, 1);
  p2_sub_builder.AddDataEdge(partitioncall_1_data, 0, partitioncall_1_case, 0);
  p2_sub_builder.AddDataEdge(partitioncall_1_case, 0, partitioncall_1_netoutput, 0);
  const auto &sub_graph2 = p2_sub_builder.GetGraph();
  sub_graph2->SetParentNode(partitioncall_1);
  sub_graph2->SetParentGraph(root_graph);
  partitioncall_1->GetOpDesc()->AddSubgraphName("f");
  partitioncall_1->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_1_sub");

  // 2.1 build case sub graph
  auto case_sub_builder = airut::GraphBuilder("case_sub");
  const auto &case_data = case_sub_builder.AddNode("case_data", DATA, 0, 1);
  AttrUtils::SetInt(case_data->GetOpDesc(), "_parent_node_index", 0);
  const auto &case_squeeze = case_sub_builder.AddNode("case_squeeze", SQUEEZE, 1, 1);
  const auto &case_netoutput = case_sub_builder.AddNode("case_netoutput", NETOUTPUT, 1, 1);
  case_sub_builder.AddDataEdge(case_data, 0, case_squeeze, 0);
  case_sub_builder.AddDataEdge(case_squeeze, 0, case_netoutput, 0);
  const auto &case_sub_graph = case_sub_builder.GetGraph();
  case_sub_graph->SetParentNode(partitioncall_1_case);
  case_sub_graph->SetParentGraph(sub_graph2);
  partitioncall_1_case->GetOpDesc()->AddSubgraphName("branches");
  partitioncall_1_case->GetOpDesc()->SetSubgraphInstanceName(0, "case_sub");

  root_graph->AddSubgraph(case_sub_graph->GetName(), case_sub_graph);
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->AddSubgraph(sub_graph2->GetName(), sub_graph2);
  return root_graph;
}
}  // namespace

using namespace airut;
class UtestGraphPartition : public testing::Test {
  protected:
    void SetUp() {
     std::map<std::string, std::string> options;
      EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
      auto ge_dev = GeRunningEnvFaker();
      ge_dev.Reset()
          .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
          .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
          .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
          .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
          .Install(FakeOp("FakeOpNpu").InfoStoreAndBuilder("AIcoreEngine"))
          .Install(FakeOp("FakeOpRts").InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
          .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
    }

    void TearDown() {
      if (GELib::GetInstance() != nullptr) {
        GELib::GetInstance()->Finalize();
      }
      auto ge_dev = GeRunningEnvFaker();
      ge_dev.Reset();
    }
};

TEST_F(UtestGraphPartition, check_if_end2pld_empty_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = nullptr;
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("default");
  EnginePartitioner::GraphPartitionInfo graph_info;
  EXPECT_NE(EnginePartitioner.CheckValidIfEnd2PldEmpty(graph_info, graph), SUCCESS);
  ge::PartitionMap partitionMap;
  partitionMap[graph] =  1;
  graph_info.partitions_ = partitionMap;
  EXPECT_NE(EnginePartitioner.CheckValidIfEnd2PldEmpty(graph_info, graph), SUCCESS);
  ge::PartitionMap partitionMap1;
  partitionMap1[graph1] =  2;
  graph_info.partitions_ = partitionMap1;
  EXPECT_EQ(EnginePartitioner.CheckValidIfEnd2PldEmpty(graph_info, graph), SUCCESS);
}

TEST_F(UtestGraphPartition, has_second_path_test) {
  size_t src = 0;
  size_t dst = 1;
  size_t upper_bound = 0UL;
  EnginePartitioner EnginePartitioner;
  string test = "test";
  ClusterPtr cluster = std::make_shared<Cluster>(1, test, test);
  ClusterSet out_set;
  out_set.insert(0);
  out_set.insert(1);
  ClusterSet in_set;
  in_set.insert(0);
  in_set.insert(1);
  cluster->out_clu_ = out_set;
  cluster->in_clu_ = in_set;
  ClusterPtr cluster1 = std::make_shared<Cluster>(1, test, test);
  ClusterSet out_set1;
  out_set1.insert(0);
  out_set1.insert(1);
  ClusterSet in_set1;
  in_set1.insert(0);
  in_set1.insert(1);
  cluster1->out_clu_ = out_set1;
  cluster1->in_clu_ = in_set1;
  std::unordered_map<size_t, ClusterPtr> clusters;
  clusters[src] = cluster;
  clusters[dst] = cluster1;
  EnginePartitioner.graph_info_.clusters_ = clusters;
  EXPECT_NE(EnginePartitioner.HasSecondPath(src, dst, upper_bound), SUCCESS);
}
TEST_F(UtestGraphPartition, merge_sub_graph_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.MergeSubGraph(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1");
  EXPECT_EQ(EnginePartitioner.MergeSubGraph(graph, graph1), SUCCESS);
  AttrUtils::SetInt(graph1, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph1, "globalworkspace_size", 1);
  EXPECT_EQ(EnginePartitioner.MergeSubGraph(graph, graph1), SUCCESS);
}

TEST_F(UtestGraphPartition, get_overflow_attr) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.FindOverflowAttr(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1");
  EXPECT_EQ(EnginePartitioner.FindOverflowAttr(graph, graph1), SUCCESS);
  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  AttrUtils::SetInt(add_n_node->GetOpDesc(), "globalworkspace_type", 0);
  AttrUtils::SetInt(add_n_node->GetOpDesc(), "globalworkspace_size", 1);
  EXPECT_EQ(EnginePartitioner.FindOverflowAttr(graph, graph1), SUCCESS);
  EXPECT_EQ(AttrUtils::HasAttr(graph1, "globalworkspace_type"), true);
  int64_t globalworkspace_type = -1;
  AttrUtils::GetInt(graph1, "globalworkspace_type", globalworkspace_type);
  ASSERT_EQ(globalworkspace_type, 0);
  int64_t globalworkspace_size = -1;
  AttrUtils::GetInt(graph1, "globalworkspace_size", globalworkspace_size);
  ASSERT_EQ(globalworkspace_size, 1);
}

TEST_F(UtestGraphPartition, merge_overflow_attr) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.MergeOverflowAttr(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1");
  EXPECT_EQ(EnginePartitioner.MergeOverflowAttr(graph, graph1), SUCCESS);
  
  EnginePartitioner.global_workspace_size_ = 1;
  EnginePartitioner.global_workspace_type_ = 0;
  AttrUtils::SetInt(graph, "globalworkspace_type", 0);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  ComputeGraphPtr graph2 = std::make_shared<ComputeGraph>("default2");
  EXPECT_EQ(EnginePartitioner.MergeOverflowAttr(graph, graph2), SUCCESS);
  AttrUtils::GetInt(graph2, "globalworkspace_size", EnginePartitioner.global_workspace_size_);
  AttrUtils::GetInt(graph2, "globalworkspace_type", EnginePartitioner.global_workspace_type_);
  ASSERT_EQ(AttrUtils::HasAttr(graph2, "globalworkspace_size"), true);
  ASSERT_EQ(EnginePartitioner.global_workspace_size_, 1);
  ASSERT_EQ(EnginePartitioner.global_workspace_type_, 0);
}

TEST_F(UtestGraphPartition, merge_after_sub_graph_optimization_test_with_func_sub_graph) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = std::make_shared<ComputeGraph>("default1");
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  AttrUtils::SetBool(sub_graph, ATTR_NAME_NO_NEED_MERGE, true);
  graph1->AddSubGraph(sub_graph);
  EXPECT_EQ(EnginePartitioner.MergeAfterSubGraphOptimization(graph, graph1), SUCCESS);
}

TEST_F(UtestGraphPartition, merge_after_sub_graph_optimization_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.MergeAfterSubGraphOptimization(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1"); 
  EXPECT_EQ(EnginePartitioner.MergeAfterSubGraphOptimization(graph, graph1), SUCCESS);
}

TEST_F(UtestGraphPartition, update_end_op_desc_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.MergeAfterSubGraphOptimization(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1");
  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  (void)AttrUtils::SetBool(add_n_node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);

  NodePtr node3 = nullptr;
  EXPECT_NE(EnginePartitioner.UpdateEndOpDesc(node3, 1, add_n_node->GetOpDesc()), SUCCESS);
  EXPECT_EQ(EnginePartitioner.UpdateEndOpDesc(add_n_node, 1, add_n_node->GetOpDesc()), SUCCESS);
}

TEST_F(UtestGraphPartition, update_pld_op_desc_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.MergeAfterSubGraphOptimization(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1");
  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  (void)AttrUtils::SetBool(add_n_node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);

  NodePtr node3 = nullptr;
  EXPECT_NE(EnginePartitioner.UpdatePldOpDesc(node3, 1, add_n_node->GetOpDesc()), SUCCESS);
  EXPECT_EQ(EnginePartitioner.UpdatePldOpDesc(add_n_node, 1, add_n_node->GetOpDesc()), SUCCESS);
}


TEST_F(UtestGraphPartition, link_input2_end_remove_orginal_linktest) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;  
  EXPECT_NE(EnginePartitioner.MergeAfterSubGraphOptimization(graph, graph1), SUCCESS);
  graph1 = std::make_shared<ComputeGraph>("default1");
  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  (void)AttrUtils::SetBool(add_n_node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);
  NodePtr node3 = nullptr;
  EXPECT_NE(EnginePartitioner.LinkInput2EndRemoveOrginalLink(node3, graph, graph1), SUCCESS);
  EXPECT_EQ(EnginePartitioner.LinkInput2EndRemoveOrginalLink(add_n_node, graph, graph1), SUCCESS);
}


TEST_F(UtestGraphPartition, put_input_nodes_in_sub_graph_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EXPECT_NE(EnginePartitioner.PutInputNodesInSubGraph(graph, graph1), SUCCESS);  ;
  graph1 = std::make_shared<ComputeGraph>("default1");
  EXPECT_EQ(EnginePartitioner.PutInputNodesInSubGraph(graph, graph1), SUCCESS);
}

TEST_F(UtestGraphPartition, add_new_graph_to_partition_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  std::string engine_name = "test";
  EXPECT_NO_THROW(EnginePartitioner.AddNewGraphToPartition(graph1, engine_name));
  EXPECT_NO_THROW(EnginePartitioner.AddNewGraphToPartition(graph, engine_name));
}

TEST_F(UtestGraphPartition, add_partitions_to_graph_node_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  std::string engine_name = "test";
  std::vector<ge::SubGraphInfoPtr> output_subgraphs;
  auto ret = EnginePartitioner.AddPartitionsToGraphNode(output_subgraphs, graph);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphPartition, split_sub_graphs_with_empty_graph) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto ret = EnginePartitioner.SplitSubGraphs(graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPartition, add_place_holder_end_test) {
  EnginePartitioner EnginePartitioner;
  AnchorPtr out_anchor;
  AnchorPtr in_anchor;
  auto ret = EnginePartitioner.AddPlaceHolderEnd(out_anchor, in_anchor);
  ASSERT_NE(ret, SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ge::NodePtr aipp1 = NodeBuilder("aipp1", AIPP)
                  .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                  .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                  .Build(graph);
  ge::NodePtr data1 = NodeBuilder("data1", DATA)
                  .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                  .AddOutputDesc({1}, FORMAT_NCHW, DT_FLOAT)
                  .Build(graph);
  OutDataAnchorPtr outptr = std::make_shared<OutDataAnchor>(aipp1, 0);
  OutDataAnchorPtr inptr = std::make_shared<OutDataAnchor>(data1, 0);
  ret = EnginePartitioner.AddPlaceHolderEnd(outptr->GetFirstPeerAnchor(), inptr->GetFirstPeerAnchor());
  ASSERT_NE(ret, SUCCESS);  
}

TEST_F(UtestGraphPartition, sort_sub_graphs_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  auto ret = EnginePartitioner.SortSubGraphs(graph);
  ASSERT_EQ(ret, SUCCESS);
  ret = EnginePartitioner.SortSubGraphs(graph1);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphPartition, get_end_in_anchor_test) {
  AnchorPtr out_anchor;
  NodePtr in_node;
  EnginePartitioner EnginePartitioner;
  auto ret = EnginePartitioner.GetEndInAnchor(out_anchor, in_node);
  ASSERT_EQ(ret, nullptr);
}

TEST_F(UtestGraphPartition, get_pld_out_anchor_test) {
  AnchorPtr in_anchor;
  EnginePartitioner EnginePartitioner;
  NodePtr node3 = nullptr;
  auto ret = EnginePartitioner.GetPldOutAnchor(node3, in_anchor);
  ASSERT_EQ(ret, nullptr);
}

TEST_F(UtestGraphPartition, add_end_pld_information_to_sub_graph_info_test) {
  EnginePartitioner EnginePartitioner;
  ge::SubGraphInfoPtr subgraph_info = nullptr;
  EXPECT_NO_THROW(EnginePartitioner.AddEndPldInformationToSubGraphInfo(subgraph_info));
}

TEST_F(UtestGraphPartition, partition_with_empty_graph) {
  EnginePartitioner EnginePartitioner;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  auto ret = EnginePartitioner.Partition(graph, mode);
  ASSERT_NE(ret, SUCCESS);
  ret = EnginePartitioner.Partition(graph1, mode);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphPartition, partition_with_graph)
{
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpRts")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op4 = OP_CFG("FakeOpRts")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op5 = OP_CFG("FakeOpRts")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)
              ->NODE("fused_op1", fake_type2_op1)
              ->NODE("fused_op2", fake_type2_op2)
              ->NODE("fused_op3", fake_type2_op3)
              ->NODE("fused_op4", fake_type2_op4)
              ->NODE("fused_op5", fake_type2_op5)
              ->NODE("Node_Output", net_output));
  };
  auto root_graph = ToComputeGraph(graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options;
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  EXPECT_EQ(EnginePartitioner.GetSubGraphMap().begin()->second.size(), 3);
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, second_partition_graph_with_user_stream_label) {
  airut::GraphBuilder graph_builder("test");
  auto data0 = graph_builder.AddNode("data_0", DATA, 1, 1, FORMAT_ND, DT_INT32, {16});
  auto fake_type2_op1 = graph_builder.AddNode("fake_type2_op1", "FakeOpNpu", 1, 1, FORMAT_ND, DT_INT32, {16});
  fake_type2_op1->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  auto fake_type2_op2 = graph_builder.AddNode("fake_type2_op2", "FakeOpNpu", 1, 1, FORMAT_ND, DT_INT32, {16});
  fake_type2_op2->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  auto fake_type2_op3 = graph_builder.AddNode("fake_type2_op3", "FakeOpRts", 1, 1, FORMAT_ND, DT_INT32, {16});
  fake_type2_op3->GetOpDesc()->SetOpEngineName("RTS");
  auto fake_type2_op4 = graph_builder.AddNode("fake_type2_op4", "FakeOpRts", 1, 1, FORMAT_ND, DT_INT32, {16});
  fake_type2_op4->GetOpDesc()->SetOpEngineName("RTS");
  AttrUtils::SetStr(fake_type2_op4->GetOpDesc(), public_attr::USER_STREAM_LABEL, "label1");
  AttrUtils::SetStr(fake_type2_op4->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label1");
  auto fake_type2_op5 = graph_builder.AddNode("fake_type2_op5", "FakeOpRts", 1, 1, FORMAT_ND, DT_INT32, {16});
  fake_type2_op5->GetOpDesc()->SetOpEngineName("RTS");
  AttrUtils::SetStr(fake_type2_op5->GetOpDesc(), public_attr::USER_STREAM_LABEL, "label1");
  AttrUtils::SetStr(fake_type2_op5->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "label1");
  auto net_output = graph_builder.AddNode("net_output", NETOUTPUT, 1, 1, FORMAT_ND, DT_INT32, {16});
  net_output->GetOpDesc()->SetOpEngineName("GELOCAL");

  graph_builder.AddDataEdge(data0, 0, fake_type2_op1, 0);
  graph_builder.AddDataEdge(fake_type2_op1, 0, fake_type2_op2, 0);
  graph_builder.AddDataEdge(fake_type2_op2, 0, fake_type2_op3, 0);
  graph_builder.AddDataEdge(fake_type2_op3, 0, fake_type2_op4, 0);
  graph_builder.AddDataEdge(fake_type2_op4, 0, fake_type2_op5, 0);
  graph_builder.AddDataEdge(fake_type2_op5, 0, net_output, 0);

  auto root_graph = graph_builder.GetGraph();
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options;
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  dlog_setlevel(GE_MODULE_NAME_U16, 0, 0);
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  auto subgraph_infos = EnginePartitioner.GetSubGraphMap().begin()->second;
  EXPECT_EQ(subgraph_infos.size(), 4);
  for (const auto &subgraph : subgraph_infos) {
    std::cout << "engine name " << subgraph->GetEngineName() << "label:" << subgraph->GetStreamLabel()
              << " user label: " << subgraph->GetUserStreamLabel() << std::endl;
  }
  EXPECT_EQ(subgraph_infos[2]->GetUserStreamLabel(), "label1");
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, partition_with_graph_stable_topo_bfs) {
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpRts")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op4 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op5 = OP_CFG("FakeOpRts")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op6 = OP_CFG("FakeOpRts")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op7 = OP_CFG("FakeOpNpu")
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op8 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)
              ->NODE("fused_op1", fake_type2_op1)
              ->NODE("fused_op2", fake_type2_op2)
              ->NODE("fused_op3", fake_type2_op3)
              ->NODE("fused_op4", fake_type2_op4)
              ->EDGE(0, 0)->NODE("fused_op7", fake_type2_op7)
              ->NODE("fused_op8", fake_type2_op8)
              ->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", data_1)
              ->NODE("fused_op5", fake_type2_op5)
              ->NODE("fused_op6", fake_type2_op6)
              ->EDGE(0, 1)->NODE("fused_op7"));
  };
  auto root_graph = ToComputeGraph(graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options = {};
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  // bfs
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
  root_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  std::map<std::string, std::set<std::string>> subgraph_to_node = {
      {"partition0_rank1_new_sub_graph1", {"fused_op1", "fused_op2"}},
      {"partition0_rank2_new_sub_graph2", {"fused_op3"}},
      {"partition0_rank3_new_sub_graph3", {"fused_op4"}},
      {"partition0_rank4_new_sub_graph5", {"fused_op5", "fused_op6"}},
      {"partition0_rank5_new_sub_graph6", {"fused_op7", "fused_op8"}},
      {"partition0_rank6_new_sub_graph7", {}}
  };
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  EXPECT_EQ(EnginePartitioner.GetSubGraphMap().begin()->second.size(), 6);
  for (const auto &sub_info : EnginePartitioner.GetSubGraphMap().begin()->second) {
    const auto subgraph = sub_info->GetSubGraph();
    ASSERT_NE(subgraph, nullptr);
    int32_t subgraph_node_num = 0;
    EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
    for (const auto &sub_node : subgraph->GetDirectNode()) {
      if (sub_node->GetType() != PLACEHOLDER && sub_node->GetType() != END && sub_node->GetType() != NETOUTPUT) {
        subgraph_node_num++;
        EXPECT_NE(subgraph_to_node[subgraph->GetName()].find(sub_node->GetName()),
            subgraph_to_node[subgraph->GetName()].end());
      }
    }
    EXPECT_EQ(subgraph_node_num, subgraph_to_node[subgraph->GetName()].size());
  }

  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, partition_with_graph_stable_topo_bfs2) {
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op4 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op5 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op6 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op7 = OP_CFG("FakeOpNpu")
        .InCnt(3)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)
          ->NODE("fused_op1", fake_type2_op1));
    CHAIN(NODE("_arg_1", data_1)
          ->NODE("fused_op2", fake_type2_op2));
    CHAIN(NODE("_arg_2", data_2)
          ->NODE("fused_op3", fake_type2_op3));
    CHAIN(NODE("fused_op1")->NODE("fused_op4", fake_type2_op4)
          ->EDGE(0, 0)->NODE("fused_op7", fake_type2_op7)->NODE("Node_Output", net_output));
    CHAIN(NODE("fused_op2")->NODE("fused_op5", fake_type2_op5)
          ->EDGE(0, 1)->NODE("fused_op7"));
    CHAIN(NODE("fused_op3")->NODE("fused_op6", fake_type2_op6)
          ->EDGE(0, 2)->NODE("fused_op7"));
  };
  auto root_graph = ToComputeGraph(graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options = {};
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  // bfs
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "0";
  GetThreadLocalContext().SetGraphOption(graph_options);
  root_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  std::map<std::string, std::set<std::string>> subgraph_to_node = {
      {"partition1_rank1_new_sub_graph1",
          {"fused_op1", "fused_op2", "fused_op3", "fused_op4", "fused_op5", "fused_op6", "fused_op7"}},
      {"partition1_rank2_new_sub_graph4", {}}
  };
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  EXPECT_EQ(EnginePartitioner.GetSubGraphMap().begin()->second.size(), 2);
  for (const auto &sub_info : EnginePartitioner.GetSubGraphMap().begin()->second) {
    const auto subgraph = sub_info->GetSubGraph();
    ASSERT_NE(subgraph, nullptr);
    int32_t subgraph_node_num = 0;
    EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
    for (const auto &sub_node : subgraph->GetDirectNode()) {
      if (sub_node->GetType() != PLACEHOLDER && sub_node->GetType() != END && sub_node->GetType() != NETOUTPUT) {
        subgraph_node_num++;
        EXPECT_NE(subgraph_to_node[subgraph->GetName()].find(sub_node->GetName()),
            subgraph_to_node[subgraph->GetName()].end());
      }
    }
    EXPECT_EQ(subgraph_node_num, subgraph_to_node[subgraph->GetName()].size());
  }

  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, partition_with_graph_stable_topo_dfs) {
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op4 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op5 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op6 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op7 = OP_CFG("FakeOpNpu")
        .InCnt(3)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)
          ->NODE("fused_op1", fake_type2_op1));
    CHAIN(NODE("_arg_1", data_1)
          ->NODE("fused_op2", fake_type2_op2));
    CHAIN(NODE("_arg_2", data_2)
          ->NODE("fused_op3", fake_type2_op3));
    CHAIN(NODE("fused_op1")->NODE("fused_op4", fake_type2_op4)
          ->EDGE(0, 0)->NODE("fused_op7", fake_type2_op7)->NODE("Node_Output", net_output));
    CHAIN(NODE("fused_op2")->NODE("fused_op5", fake_type2_op5)
          ->EDGE(0, 1)->NODE("fused_op7"));
    CHAIN(NODE("fused_op3")->NODE("fused_op6", fake_type2_op6)
          ->EDGE(0, 2)->NODE("fused_op7"));
  };
  auto root_graph = ToComputeGraph(graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options = {};
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  // bfs
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);
  root_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  std::map<std::string, std::set<std::string>> subgraph_to_node = {
      {"partition1_rank1_new_sub_graph1",
          {"fused_op1", "fused_op2", "fused_op3", "fused_op4", "fused_op5", "fused_op6", "fused_op7"}},
      {"partition1_rank2_new_sub_graph4", {}}
  };
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  EXPECT_EQ(EnginePartitioner.GetSubGraphMap().begin()->second.size(), 2);
  for (const auto &sub_info : EnginePartitioner.GetSubGraphMap().begin()->second) {
    const auto subgraph = sub_info->GetSubGraph();
    ASSERT_NE(subgraph, nullptr);
    int32_t subgraph_node_num = 0;
    EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
    for (const auto &sub_node : subgraph->GetDirectNode()) {
      if (sub_node->GetType() != PLACEHOLDER && sub_node->GetType() != END && sub_node->GetType() != NETOUTPUT) {
        subgraph_node_num++;
        EXPECT_NE(subgraph_to_node[subgraph->GetName()].find(sub_node->GetName()),
            subgraph_to_node[subgraph->GetName()].end());
      }
    }
    EXPECT_EQ(subgraph_node_num, subgraph_to_node[subgraph->GetName()].size());
  }

  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, partition_with_graph_stable_topo_rdfs) {
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto data_2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op4 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op5 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op6 = OP_CFG("FakeOpNpu")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op7 = OP_CFG("FakeOpNpu")
        .InCnt(3)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("_arg_0", data_0)
          ->NODE("fused_op1", fake_type2_op1));
    CHAIN(NODE("_arg_1", data_1)
          ->NODE("fused_op2", fake_type2_op2));
    CHAIN(NODE("_arg_2", data_2)
          ->NODE("fused_op3", fake_type2_op3));
    CHAIN(NODE("fused_op1")->NODE("fused_op4", fake_type2_op4)
          ->EDGE(0, 0)->NODE("fused_op7", fake_type2_op7)->NODE("Node_Output", net_output));
    CHAIN(NODE("fused_op2")->NODE("fused_op5", fake_type2_op5)
          ->EDGE(0, 1)->NODE("fused_op7"));
    CHAIN(NODE("fused_op3")->NODE("fused_op6", fake_type2_op6)
          ->EDGE(0, 2)->NODE("fused_op7"));
  };
  auto root_graph = ToComputeGraph(graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options = {};
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  // bfs
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  root_graph->TopologicalSortingGraph();
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  std::map<std::string, std::set<std::string>> subgraph_to_node = {
      {"partition1_rank1_new_sub_graph1",
          {"fused_op1", "fused_op2", "fused_op3", "fused_op4", "fused_op5", "fused_op6", "fused_op7"}},
      {"partition1_rank2_new_sub_graph4", {}}
  };
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  EXPECT_EQ(EnginePartitioner.GetSubGraphMap().begin()->second.size(), 2);
  for (const auto &sub_info : EnginePartitioner.GetSubGraphMap().begin()->second) {
    const auto subgraph = sub_info->GetSubGraph();
    ASSERT_NE(subgraph, nullptr);
    int32_t subgraph_node_num = 0;
    EXPECT_NE(subgraph_to_node.find(subgraph->GetName()), subgraph_to_node.end());
    for (const auto &sub_node : subgraph->GetDirectNode()) {
      if (sub_node->GetType() != PLACEHOLDER && sub_node->GetType() != END && sub_node->GetType() != NETOUTPUT) {
        subgraph_node_num++;
        EXPECT_NE(subgraph_to_node[subgraph->GetName()].find(sub_node->GetName()),
            subgraph_to_node[subgraph->GetName()].end());
      }
    }
    EXPECT_EQ(subgraph_node_num, subgraph_to_node[subgraph->GetName()].size());
  }

  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, partition_sub_graph_test) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr graph1 = nullptr;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kSecondPartitioning;
  graph->SetOutputSize(0);
  auto ret = EnginePartitioner.PartitionSubGraph(graph, mode);
  ASSERT_NE(ret, SUCCESS);
  ret = EnginePartitioner.PartitionSubGraph(graph1, mode);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphPartition, partition_vector_engine_graph_test) {
  const auto back_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  auto options = back_options;
  options["ge.autoMultistreamParallelMode"] = "cv";
  ge::GetThreadLocalContext().SetSessionOption(options);
  EnginePartitioner EnginePartitioner;
  auto graph = gert::ShareGraph::BuildCVParallelGraph();
  ComputeGraphPtr root_graph = GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_TRUE(root_graph != nullptr);
  auto aiv1 = root_graph->FindNode("aiv1");
  ASSERT_TRUE(aiv1 != nullptr);
  auto aiv2 = root_graph->FindNode("aiv2");
  ASSERT_TRUE(aiv2 != nullptr);
  auto aiv3 = root_graph->FindNode("aiv3");
  ASSERT_TRUE(aiv3 != nullptr);
  auto reshape = root_graph->FindNode("reshape");
  ASSERT_TRUE(reshape != nullptr);
  aiv1->GetOpDesc()->SetOpEngineName("VectorEngine");
  aiv2->GetOpDesc()->SetOpEngineName("VectorEngine");
  aiv3->GetOpDesc()->SetOpEngineName("VectorEngine");
  reshape->GetOpDesc()->SetOpEngineName("DNN_VM_GE_LOCAL");

  auto aic1 = root_graph->FindNode("aic1");
  ASSERT_TRUE(aiv1 != nullptr);
  auto aic2 = root_graph->FindNode("aic2");
  ASSERT_TRUE(aiv2 != nullptr);
  aic1->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  aic2->GetOpDesc()->SetOpEngineName("AIcoreEngine");
  root_graph->TopologicalSorting();
  auto ret = EnginePartitioner.Initialize(root_graph, EnginePartitioner::Mode::kSecondPartitioning);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPartition, partition_sub_graph_test_second_path) {
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("FakeOpNpu").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op2 = OP_CFG("FakeOpRts").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op3 = OP_CFG("FakeOpNpu").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});

    CHAIN(NODE("_arg_0", data_0)
              ->NODE("fused_op1", fake_type2_op1)
              ->NODE("fused_op2", fake_type2_op2)
              ->NODE("fused_op3", fake_type2_op3)
              ->NODE("Node_Output", net_output));
    CHAIN(NODE("fused_op1")->NODE("fused_op3"));
  };
  auto root_graph = ToComputeGraph(graph);
  (void) AttrUtils::SetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, "0");
  map<string, string> options;
  EXPECT_EQ(ge::GELib::Initialize(options), SUCCESS);
  EnginePartitioner EnginePartitioner;
  EnginePartitioner::Mode mode = EnginePartitioner::Mode::kAtomicEnginePartitioning;
  ASSERT_EQ(EnginePartitioner.Partition(root_graph, mode), SUCCESS);
  // op3 and op1 cannot merge because second path [op1->op2->op3]
  // we get 4 engine subgraphs and 1 input-node subgraph
  EXPECT_EQ(EnginePartitioner.GetSubGraphMap().begin()->second.size(), 4);
  EXPECT_EQ(ge::GELib::GetInstance()->Finalize(), SUCCESS);
}

TEST_F(UtestGraphPartition, RemoveNodeAndEdgeBetweenEndPld) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr output_merged_compute_graph = nullptr;
  std::vector<SubGraphInfoPtr> sub_graph_list;
  EnginePartitioner::GraphPartitionInfo graph_info;
  EXPECT_EQ(EnginePartitioner.RemoveNodeAndEdgeBetweenEndPld(output_merged_compute_graph, sub_graph_list, graph_info),
            PARAM_INVALID);
}

TEST_F(UtestGraphPartition, MergeAfterSubGraphOptimization) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr output_merged_compute_graph = std::make_shared<ComputeGraph>("graph");
  ComputeGraphPtr original_compute_graph = BuildGraphPartitionCall();
  EXPECT_EQ(EnginePartitioner.MergeAfterSubGraphOptimization(output_merged_compute_graph, original_compute_graph),
            PARAM_INVALID);
}

TEST_F(UtestGraphPartition, HasNoInput) {
  EnginePartitioner EnginePartitioner;
  NodePtr node = nullptr;
  EXPECT_EQ(EnginePartitioner.HasNoInput(node), true);
}

TEST_F(UtestGraphPartition, DataShouldClearUserStreamLabel) {
  EnginePartitioner EnginePartitioner;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  NodePtr data =
      NodeBuilder("data", DATA).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr relu =
      NodeBuilder("relu", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr netoutput =
      NodeBuilder("netoutput", NETOUTPUT).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), relu->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));

  AttrUtils::SetStr(data->GetOpDesc(), public_attr::USER_STREAM_LABEL, "label1");
  AttrUtils::SetStr(relu->GetOpDesc(), public_attr::USER_STREAM_LABEL, "label1");

  EnginePartitioner.Initialize(graph, EnginePartitioner::Mode::kSecondPartitioning);

  std::string user_stream_label;
  AttrUtils::GetStr(data->GetOpDesc(), public_attr::USER_STREAM_LABEL, user_stream_label);
  EXPECT_TRUE(user_stream_label.empty());
}
} // namespace ge
