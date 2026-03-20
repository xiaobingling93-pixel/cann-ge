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
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "common/debug/ge_log.h"
#include "graph/passes/format_optimize/dim1_transpose_to_squeeze_pass.h"
#include "transfer_shape_utils.h"

using namespace std;
using namespace ge;

/******************
            abs1
                \
data1-->cast1-->transdata1-->netoutput1
        \
        cast2-->transdata2-->abs----cast1
                /
            abs2
******************/
namespace {
Graph BuildTransopFusionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("cast1", CAST));
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("cast2", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0, 0)->NODE("transdata1", TRANSDATA));
    CHAIN(NODE("cast2", CAST)->EDGE(0, 0)->NODE("transdata2", TRANSDATA));
    CHAIN(NODE("transdata1", TRANSDATA)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("transdata2", TRANSDATA)->EDGE(0, 0)->NODE("abs", ABSVAL));
    CHAIN(NODE("abs", ABSVAL)->CTRL_EDGE()->NODE("cast1", CAST));
    CHAIN(NODE("abs2", ABSVAL)->CTRL_EDGE()->NODE("transdata2", TRANSDATA));
    CHAIN(NODE("abs1", ABSVAL)->CTRL_EDGE()->NODE("transdata1", TRANSDATA));
  };
  return ToGeGraph(g1);
}

/******************
      abs
        \
data1-->cast1-->netoutput1
     \         /
      cast2----
******************/
Graph BuildTransopBreadthFusionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("cast1", CAST));
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("cast2", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("cast2", CAST)->EDGE(0, 1)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("abs", ABSVAL)->CTRL_EDGE()->NODE("cast1", CAST));
  };
  return ToGeGraph(g1);
}

/******************
      abs
        \
 data|-->cast1-->transdata1->addN-->netoutput
     |                          /
     |-->cast2-->transdata2-->relu
******************/
Graph BuildTransDataAndCastopBreadthFusionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast1", CAST));
    CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast2", CAST));
    CHAIN(NODE("cast1", CAST)->EDGE(0, 0)->NODE("transdata1", TRANSDATA));
    CHAIN(NODE("cast2", CAST)->EDGE(0, 0)->NODE("transdata2", TRANSDATA));
    CHAIN(NODE("transdata1", TRANSDATA)->EDGE(0, 0)->NODE("addn", ADDN));
    CHAIN(NODE("transdata2", TRANSDATA)->EDGE(0, 0)->NODE("relu", RELU));
    CHAIN(NODE("addn", ADDN)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("relu", RELU)->EDGE(0, 1)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("abs", ABSVAL)->CTRL_EDGE()->NODE("cast1", CAST));
  };

  return ToGeGraph(g1);
}

using NodeOutIndex = std::pair<std::string, uint32_t>;
using UtPath = std::vector<NodeOutIndex>;
using UtPaths = std::vector<UtPath>;
bool CheckConnection(const ComputeGraphPtr &graph, std::vector<NodeOutIndex> &path) {
  for (size_t i = 0U; i < path.size() - 1U; ++i) {
    auto firt_node = graph->FindNode(path[i].first);
    if (firt_node == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << " is not found." << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
    if (path[i].second >= firt_node->GetOutDataNodesAndAnchors().size()) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << " index: " << path[i].second << " is larger than  actrual output size: "
                << firt_node->GetOutDataNodesAndAnchors().size()
                << ", i: " << i << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
    auto first_out = firt_node->GetOutDataAnchor(path[i].second);
    if (first_out == nullptr) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << " index: " << path[i].second << " out_anchor is null. output size: "
                << firt_node->GetOutDataNodesAndAnchors().size()
                << ", i: " << i << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
    bool find = false;
    for (const auto &in_anchor : first_out->GetPeerInDataAnchors()) {
      if (in_anchor->GetOwnerNode()->GetName() == path[i + 1].first) {
        find = true;
      }
    }

    if (!find) {
      std::cout << "========================================" << std::endl;
      std::cout << path[i].first << "[" << path[i].second << "] is not connected to " << path[i + 1].first << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(graph, "CheckConnection_failed");
      return false;
    }
  }
  return true;
}

bool CheckConnection(const ComputeGraphPtr &graph, std::vector<std::vector<NodeOutIndex>> &paths) {
  for (auto &path : paths) {
    if (!CheckConnection(graph, path)) {
      return false;
    }
  }
  return true;
}

using NetoutputParentIndexes = std::vector<std::pair<std::string, std::vector<uint32_t>>>;
bool AddParentIndexForNetoutput(ComputeGraphPtr &root_graph, NetoutputParentIndexes &indexes) {
  std::map<std::string, NodePtr> netoutput_map;
  for (auto &node : root_graph->GetAllNodes()) {
    netoutput_map[node->GetName()] = node;
  }
  for (auto &name_indexes_pair : indexes) {
    const auto iter = netoutput_map.find(name_indexes_pair.first);
    if (iter == netoutput_map.end()) {
      std::cout << "========================================" << std::endl;
      std::cout << "can not find " << name_indexes_pair.first << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "AddParentIndexForNetoutput_failed");
      return false;
    }
    auto op_desc = iter->second->GetOpDesc();
    size_t input_index = 0U;
    if (name_indexes_pair.second.size() != op_desc->GetInputsSize()) {
      std::cout << "========================================" << std::endl;
      std::cout << name_indexes_pair.first << " real inputs size: " << op_desc->GetInputsSize()
                << ", but name_indexes_pair.second.size(): " << name_indexes_pair.second.size() << std::endl;
      std::cout << "========================================" << std::endl;
      GE_DUMP(root_graph, "AddParentIndexForNetoutput_failed");
      return false;
    }
    for (auto parent_index : name_indexes_pair.second) {
      auto tensor_desc = op_desc->MutableInputDesc(input_index++);
      AttrUtils::SetInt(tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    }
  }
  return true;
}

}  // namespace

class TransopFusionOptimizeTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}

};
/******************
      abs
        \c
data1-->cast1-->netoutput1
        \         /
        cast2----
******************/
TEST_F(TransopFusionOptimizeTest, test_transop_breadth_fusion_pass_normal) {
  Graph test_graph = BuildTransopBreadthFusionGraph();

  DUMP_GRAPH_WHEN("OptimizeStage1_1");
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, test_graph, options);
  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);

  CHECK_GRAPH(OptimizeStage1_1) {
    auto ret = graph->TopologicalSorting();
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(graph->GetDirectNodesSize(), 4);
    auto data1 = graph->FindNode("data1");
    auto remain_cast_node = data1->GetOutNodes().at(0);
    EXPECT_EQ(remain_cast_node->GetType(), "Cast");
    EXPECT_EQ(remain_cast_node->GetInControlNodes().size(), 0);
    auto netoutput1 = graph->FindNode("netoutput1");
    std::unordered_set<std::string> in_control_node_name;
    for (const auto &in_control_node : netoutput1->GetInControlNodes()) {
      in_control_node_name.insert(in_control_node->GetName());
    }
    EXPECT_TRUE(in_control_node_name.count("abs") > 0);
  };
}

TEST_F(TransopFusionOptimizeTest, test_transop_and_cast_breadth_fusion_pass_normal) {
  Graph test_graph = BuildTransDataAndCastopBreadthFusionGraph();

  DUMP_GRAPH_WHEN("OptimizeStage1_1");
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, test_graph, options);
  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);

  CHECK_GRAPH(OptimizeStage1_1) {
      auto ret = graph->TopologicalSorting();
      EXPECT_EQ(ret, SUCCESS);
      auto data = graph->FindNode("data");
      auto remain_node = data->GetOutNodes().at(0);
      EXPECT_EQ(remain_node->GetType(), "TransData");
      EXPECT_EQ(data->GetOutDataNodesSize(), 1);
  };
}

/*
 *  data--cast--transdata--netoutput
 *    |
 * partitioned_call--+--transdata
 *    |              |
 *    |              +--transpose
 *    |
 * +-------------------------------------+
 * | data--partitioned_call---netoutput  |
 * |              |                      |
 * |      +-----------------+            |
 * |      | data--netoutput |            |
 * |      +-----------------+            |
 * +-------------------------------------+
 *             ||
 *             \/
 *  data--transdata--cast--netoutput
 *            |
 * partitioned_call
 *    |
 * +-------------------------------------+
 * | data--partitioned_call---netoutput  |
 * |              |                      |
 * |      +-----------------+            |
 * |      | data--netoutput |            |
 * |      +-----------------+            |
 * +-------------------------------------+
 */
TEST_F(TransopFusionOptimizeTest, DiffGraph_ExtractTransdataThroughDoubleSubGraph_AddNewData_DoFusion) {
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_netoutput", NETOUTPUT));
                       };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                            ->NODE("transdata1", TRANSDATA)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("partitioned_call", PARTITIONEDCALL)->EDGE(0, 0)->NODE("transpose", TRANSPOSE)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);

  GeTensorDesc tensor_desc(GeShape({1, 1, 224, 224}), Format::FORMAT_NCHW);
  const auto shape = GeShape({1, 1, 224, 224});
  auto data_node = compute_graph->FindNode("data");
  data_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(shape);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(shape);
  data_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape);
  data_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape);

  auto cast_node = compute_graph->FindNode("cast");
  cast_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(shape);
  cast_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(shape);
  cast_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape);
  cast_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape);

  auto transdata_node = compute_graph->FindNode("transdata");
  transdata_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(shape);
  transdata_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(shape);
  transdata_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape);
  transdata_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape);

  auto transdata_node1 = compute_graph->FindNode("transdata1");
  transdata_node1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(shape);
  transdata_node1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(shape);
  transdata_node1->GetOpDesc()->MutableInputDesc(0)->SetShape(shape);
  transdata_node1->GetOpDesc()->MutableOutputDesc(0)->SetShape(shape);

  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;

  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  ASSERT_EQ(compute_graph->FindNode("transdata1"), nullptr);
  ASSERT_EQ(compute_graph->FindNode("transdata"), nullptr);
  ASSERT_EQ(compute_graph->FindNode("partitioned_call")->GetInDataNodes().size(), 1U);

  ASSERT_EQ(sub_sub_1_graph->FindNode("sub_sub_1_transdata_fusion_arg_1"), nullptr);
}

/*
 *  data--relu--if--transdata
 *
 * if_subgraph                            then_subgraph
 * +-------------------------------+      +------------------------+
 * | data-+--if_transdata-if_relu  |      | data--relu---netoutput |
 * |      |                        |      +------------------------+
 * |     netoutput                 |
 * +-------------------------------+
 * 关注点：由于then_subgraph中没有到达transdata的路径，所以两个transdata不能融合
 */
TEST_F(TransopFusionOptimizeTest, IfTwoSubgraphs_NotFuse) {
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
                      CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)
                                ->NODE("if_sub_netoutput", NETOUTPUT));
                      CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)->NODE("if_transdata", TRANSDATA)->NODE("if_relu", RELU)->Ctrl()
                                ->NODE("if_sub_netoutput", NETOUTPUT));
                    };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
                        CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
                      };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;

  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  UtPaths root_graph_path = {{{"if", 0}, {"transdata", 0}}};
  ASSERT_TRUE(CheckConnection(compute_graph, root_graph_path));

  std::vector<NodeOutIndex> path2 = {{"if_sub_data", 0}, {"if_transdata", 0}, {"if_relu", 0}};
  ASSERT_TRUE(CheckConnection(if_sub_graph, path2));
}

TEST_F(TransopFusionOptimizeTest, test_same_transop_fusion_pass_not_cause_loop) {
  Graph test_graph = BuildTransopFusionGraph();
  
  DUMP_GRAPH_WHEN("OptimizeStage1_1");
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, test_graph, options);
  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);

  CHECK_GRAPH(OptimizeStage1_1) {
    auto ret = graph->TopologicalSorting();
    EXPECT_EQ(ret, SUCCESS);
  };
}

// 用例需要重新构造,infershape之后插入transdata
// TEST_F(TransopFusionOptimizeTest, test_transop_symmetry_elimination_pass) {
//   Graph test_graph = BuildTransopSymmetryGraph();
  
//   DUMP_GRAPH_WHEN("OptimizeStage1_2");
//   map<AscendString, AscendString> options;
//   Session session(options);
//   session.AddGraph(1, test_graph, options);
//   std::vector<InputTensorInfo> inputs;
//   session.BuildGraph(1, inputs);

//   CHECK_GRAPH(OptimizeStage1_2) {
//     // check transop symmtry fusion
//     auto netoutput = graph->FindNode("netoutput1");
//     auto transdata = netoutput->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
//     auto data2 = netoutput->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode();
//     EXPECT_EQ(transdata->GetName(), "transdata2");
//     EXPECT_EQ(data2->GetName(), "data2");

//     auto transdata1 = transdata->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
//     EXPECT_EQ(transdata1->GetName(), "transdata1");

//     auto data3 = graph->FindNode("data3");
//     EXPECT_EQ(data3->GetOutDataNodes().at(0)->GetType(), TRANSDATA);
//   };
// }

TEST_F(TransopFusionOptimizeTest, test_framework_transop_breath_fusion_not_fuse) {
  vector<int64_t> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_tensor1 = 
    std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANT).Weight(const_tensor1);

  vector<int32_t> perm2{0, 2, 1, 3};
  GeTensorDesc tensor_desc2(GeShape(vector<int64_t>{4}), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor2 = 
    std::make_shared<GeTensor>(tensor_desc2, reinterpret_cast<uint8_t *>(perm2.data()), sizeof(int32_t)*perm2.size());
  auto const2 = OP_CFG(CONSTANT).Weight(const_tensor2);

  auto transpose1 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 128, 52, 52});
  auto transpose2 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 128, 52, 52});
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(2).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1, 128, 52, 52});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("transpose1", transpose1));
    CHAIN(NODE("data1", DATA)->EDGE(1, 0)->NODE("transpose2", transpose2));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("transpose1", transpose1));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("transpose2", transpose2));
    CHAIN(NODE("transpose1", transpose1)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("transpose2", transpose2)->EDGE(0, 1)->NODE("netoutput1", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);

  map<string, uint32_t> name_index;
  name_index.emplace("x", 0);
  name_index.emplace("perm", 1);
  for (auto &gn : graph.GetAllNodes()) {
    AscendString type;
    (void)gn.GetType(type);
    if (type == TRANSPOSE) {
      auto node = NodeAdapter::GNode2Node(gn);
      if (node != nullptr && node->GetOpDesc()) {
        node->GetOpDesc()->UpdateInputName(name_index);
      }
    }
  }

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(4, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // transpose1 and transpose2 not fusion, the num of nodes not changed
  EXPECT_EQ(graph.GetDirectNode().size(), 6);
}

TEST_F(TransopFusionOptimizeTest, test_framework_dim1_transpose_noreplace) {
  auto transpose1 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {16, 32, 64, 128});
  auto transpose2 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {16, 32, 64, 128});
  auto data2 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {-1, -1, -1, -1});
  DEF_GRAPH(transGraph) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("transpose1", transpose1));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("transpose2", transpose2));
    CHAIN(NODE("transpose1", transpose1)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("transpose2", transpose2)->EDGE(0, 1)->NODE("netoutput1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(transGraph);
  GE_DUMP(graph, "Input ComputeGraph");
  map<string, uint32_t> name_index;
  name_index.emplace("x", 0);
  name_index.emplace("perm", 1);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == TRANSPOSE) {
      if (node != nullptr && node->GetOpDesc()) {
        node->GetOpDesc()->UpdateInputName(name_index);
      }
      node->GetOpDesc()->MutableInputDesc(0)->SetUnknownDimNumShape();
    }
    if (node->GetName() == "transpose2") {
      gert::SymbolShape symbol_shape({
        Symbol("s0"),
        Symbol(1),
        Symbol("s1"),
        Symbol(10),
      });
      const auto attr = node->GetOpDesc()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
      attr->symbolic_tensor.SetSymbolShape(symbol_shape);
    }
  }
  std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
  for (auto &node : graph->GetAllNodes()) {
    Status ret = pass->Run(node);
    EXPECT_EQ(ret, SUCCESS);
  }
  GE_DUMP(graph, "Input ComputeGraph");
  EXPECT_EQ(graph->GetDirectNode().size(), 5);
  auto transpose_1 = graph->FindNode("transpose1");
  auto transpose_2 = graph->FindNode("transpose2");
  EXPECT_NE(transpose_1, nullptr);
  EXPECT_NE(transpose_2, nullptr);
}

TEST_F(TransopFusionOptimizeTest, test_framework_dim1_transpose_replace_squeeze_unsqueeze) {
  vector<int16_t> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}), FORMAT_ND, DT_UINT16);
  GeTensorPtr const_tensor1 =
    std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int16_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANT).Weight(const_tensor1);

  vector<int32_t> perm2{0, 2, 1, 3};
  GeTensorDesc tensor_desc2(GeShape(vector<int64_t>{4}), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor2 =
    std::make_shared<GeTensor>(tensor_desc2, reinterpret_cast<uint8_t *>(perm2.data()), sizeof(int32_t)*perm2.size());
  auto const2 = OP_CFG(CONSTANT).Weight(const_tensor2);

  auto transpose1 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {16, 32, 64, 128});
  auto transpose2 = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1});
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {16, 32, 64, 128});
  auto data2 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {-1, -1, -1, -1});
  DEF_GRAPH(transGraph) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("transpose1", transpose1));
    CHAIN(NODE("data2", data2)->EDGE(0, 0)->NODE("transpose2", transpose2));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("transpose1", transpose1));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("transpose2", transpose2));
    CHAIN(NODE("transpose1", transpose1)->EDGE(0, 0)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE("transpose2", transpose2)->EDGE(0, 1)->NODE("netoutput1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(transGraph);

  map<string, uint32_t> name_index;
  name_index.emplace("x", 0);
  name_index.emplace("perm", 1);
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == TRANSPOSE) {
      if (node != nullptr && node->GetOpDesc()) {
        node->GetOpDesc()->UpdateInputName(name_index);
      }
      node->GetOpDesc()->MutableInputDesc(0)->SetUnknownDimNumShape();
    }
    if (node->GetName() == "transpose2") {
      gert::SymbolShape symbol_shape({
        Symbol("s0"),
        Symbol(1),
        Symbol("s1"),
        Symbol(10),
      });
      const auto attr = node->GetOpDesc()->MutableInputDesc(0)->GetOrCreateAttrsGroup<SymbolicDescAttr>();
      attr->symbolic_tensor.SetSymbolShape(symbol_shape);
    }
  }
  std::shared_ptr<Dim1TransposeToSqueezePass> pass = std::make_shared<Dim1TransposeToSqueezePass>();
  for (auto &node : graph->GetAllNodes()) {
    Status ret = pass->Run(node);
    EXPECT_EQ(ret, SUCCESS);
  }
  EXPECT_EQ(graph->GetDirectNode().size(), 7);
  auto squeeze = graph->FindNode("transpose2_replaced_Squeeze");
  auto unsqueeze = graph->FindNode("transpose2_replaced_Unsqueeze");
  // 替换逻辑修改，所以该场景预期不会替换
  EXPECT_EQ(squeeze, nullptr);
  EXPECT_EQ(unsqueeze, nullptr);
}

/**
 *              data2(static)
 *                 |
 *   data1         |
 * (dynamic)     unsqueese
 *      \        /
 *         less
 *          |
 *        netoutput
 */
TEST_F(TransopFusionOptimizeTest, test_reshape_recover_on_single_op_graph) {
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {5, -1}).Attr(ATTR_NAME_INDEX, 0).Attr("OwnerGraphIsUnknown", true);
  auto data2 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {5}).Attr(ATTR_NAME_INDEX, 1).Attr("OwnerGraphIsUnknown", true);
  auto unsqueeze = OP_CFG(UNSQUEEZE)
                       .InCnt(1)
                       .TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {5})
                       .OutCnt(1)
                       .TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {5, 1});
  auto less = OP_CFG(LESS)
                  .InCnt(2)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {5, -1})
                  .OutCnt(1)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
                  .Attr("_unknown_shape", true)
                  .Attr(ATTR_SINGLE_OP_SCENE, true);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->NODE("less", less)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data2", data2)->NODE("unsqueeze", UNSQUEEZE)->EDGE(0, 1)->NODE("less"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ToComputeGraph(g1);
  auto data1_node = compute_graph->FindNode("data1");
  auto data_input_desc_0 = data1_node->GetOpDesc()->MutableInputDesc(0);
  data_input_desc_0->SetShape(GeShape({5, -1}));
  auto data_output_desc_0 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  data_output_desc_0->SetShape(GeShape({5, -1}));

  auto less_node = compute_graph->FindFirstNodeMatchType(LESS);
  auto input_desc_0 = less_node->GetOpDesc()->MutableInputDesc(0);
  input_desc_0->SetShape(GeShape({5, -1}));
  auto input_desc_1 = less_node->GetOpDesc()->MutableInputDesc(1);
  input_desc_1->SetShape(GeShape({5,1}));
  //compute_graph->SetGraphUnknownFlag(true);
  AttrUtils::SetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  DUMP_GRAPH_WHEN("PreRunAfterOptimize2");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  (void)session.BuildGraph(1, inputs);
  CHECK_GRAPH(PreRunAfterOptimize2) {
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == LESS) {
        EXPECT_NE(node, nullptr);
        auto peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
        EXPECT_NE(peer_anchor, nullptr);
        auto peer_node = peer_anchor->GetOwnerNode();
        EXPECT_NE(peer_node, nullptr);
        EXPECT_EQ(peer_node->GetType(), RESHAPE);
        /*
        auto const_in_anchor = peer_node->GetInDataAnchor(1);
        auto shape_const = const_in_anchor->GetPeerOutAnchor()->GetOwnerNode();
        // 动静拆分以后，const被拆到partitioncall里面，所以这里类型是data
        EXPECT_EQ(shape_const->GetType(), "Data");
        */
      }
    }
  };
}

void FakeTransData5DNodeEngine(GeRunningEnvFaker &ge_env) {
  auto ffo = MakeShared<FakeFormatsOptimizer>();
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_NHWC, 5));
  ffo->OpFormatByName(
      "conv1", {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({1,1,100,190,16}))},
              {dst_format, GeShape(std::vector<int64_t>({9,16,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({1,16,100,190,16}))}
          }
      });
  ffo->OpFormatByName(
      "conv2", {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,2,100,190,16}))},
              {dst_format, GeShape(std::vector<int64_t>({18,16,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,32,100,190}))}
          }
      });
  ffo->OpFormatByName(
      "conv3", {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({1,16,190,100,16}))},
              {dst_format, GeShape(std::vector<int64_t>({18,16,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({1,256,190,100}))}
          }
      });
  ge_env.InstallDefault();
  ge_env.Install(FakeEngine("TestForTransdata").GraphOptimizer("TestForTransdata", ffo));
}

TEST_F(TransopFusionOptimizeTest, test_transdata_transdata_fusion_5hd_nchw) {
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,3,100,190})
                   .Attr(ATTR_NAME_INDEX, 0);
  auto conv1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,3,100,190});
  auto reshape1 = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,256,100,190});
  auto conv2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {8,32,100,190});
  auto const1 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, ge::DT_FLOAT, {4});
  transformer::TransferShapeUtils::GetC0Value(DT_FLOAT, ge::FORMAT_FRACTAL_NZ_C0_2);
  transformer::TransferShapeUtils::GetC0Value(DT_FLOAT, ge::FORMAT_FRACTAL_NZ_C0_8);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("conv1", conv1)->EDGE(0, 0)->NODE("reshape1", reshape1)
              ->EDGE(0, 0)->NODE("conv2", conv2)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape1", reshape1));
    CHAIN(NODE("conv1_weight", CONSTANT)->EDGE(0, 1)->NODE("conv1", conv1));
    CHAIN(NODE("conv2_weight", CONSTANT)->EDGE(0, 1)->NODE("conv2", conv2));
  };
  auto compute_graph = ToComputeGraph(g1);
  auto test_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto reshape = compute_graph->FindNode("reshape1");
  EXPECT_NE(reshape, nullptr);
  GeShape reshape_in({1,256,100,190});
  GeShape reshape_out({8,32,100,190});
  reshape->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(reshape_in, FORMAT_NCHW, DT_FLOAT));
  reshape->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(reshape_out, FORMAT_NCHW, DT_FLOAT));

  auto conv1_node = compute_graph->FindNode("conv1");
  EXPECT_NE(conv1_node, nullptr);
  conv1_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_in);
  conv1_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

  GeRunningEnvFaker ge_env;
  FakeTransData5DNodeEngine(ge_env);

  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  map<AscendString, AscendString> options;
  Session session(options);
  dlog_setlevel(0,0,0);
  session.AddGraph(99, test_graph, options);
  std::vector<InputTensorInfo> inputs;

  (void)session.BuildGraph(99, inputs);
  dlog_setlevel(3,3,0);

  CHECK_GRAPH(OptimizeStage1_2) {
    auto reshape_end = graph->FindNode("reshape1");
    EXPECT_EQ(reshape_end, nullptr);
    auto conv1_end = graph->FindNode("conv1");
    EXPECT_NE(conv1_end, nullptr);
    EXPECT_EQ(conv1_end->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims(), std::vector<int64_t>({1,16,100,190,16}));
    auto conv2_end = graph->FindNode("conv2");
    EXPECT_NE(conv2_end, nullptr);
    EXPECT_EQ(conv2_end->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims(), std::vector<int64_t>({8,2,100,190,16}));

    auto in_node = NodeUtils::GetInDataNodeByIndex(*conv2_end, 0);
    EXPECT_EQ(in_node->GetName(), conv1_end->GetName());
  };

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(TransopFusionOptimizeTest, test_transdata_transdata_fusion_5hd_nchw_c_padding) {
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,3,100,190})
      .Attr(ATTR_NAME_INDEX, 0);
  auto conv1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,3,100,190});
  auto reshape1 = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,256,100,190});
  auto conv3 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, ge::DT_FLOAT, {1,256,190,100});
  auto const1 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, ge::DT_FLOAT, {4});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("conv1", conv1)->EDGE(0, 0)->NODE("reshape1", reshape1)
                            ->EDGE(0, 0)->NODE("conv3", conv3)->NODE("net_output", NETOUTPUT));
                  CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape1", reshape1));
                  CHAIN(NODE("conv1_weight", CONSTANT)->EDGE(0, 1)->NODE("conv1", conv1));
                  CHAIN(NODE("conv3_weight", CONSTANT)->EDGE(0, 1)->NODE("conv3", conv3));
                };
  auto compute_graph = ToComputeGraph(g1);
  auto test_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto reshape = compute_graph->FindNode("reshape1");
  EXPECT_NE(reshape, nullptr);
  GeShape reshape_in({1,256,100,190});
  GeShape reshape_out({1,256,190,100});
  reshape->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(reshape_in, FORMAT_NCHW, DT_FLOAT));
  reshape->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(reshape_out, FORMAT_NCHW, DT_FLOAT));

  auto conv1_node = compute_graph->FindNode("conv1");
  EXPECT_NE(conv1_node, nullptr);
  conv1_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_in);
  conv1_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

  GeRunningEnvFaker ge_env;
  FakeTransData5DNodeEngine(ge_env);

  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  map<AscendString, AscendString> options;
  Session session(options);
  dlog_setlevel(0,0,0);
  session.AddGraph(99, test_graph, options);
  std::vector<InputTensorInfo> inputs;

  (void)session.BuildGraph(99, inputs);
  dlog_setlevel(3,3,0);

  CHECK_GRAPH(OptimizeStage1_2) {
    auto reshape_end = graph->FindNode("reshape1");
    EXPECT_EQ(reshape_end, nullptr);
    auto conv1_end = graph->FindNode("conv1");
    EXPECT_NE(conv1_end, nullptr);
    EXPECT_EQ(conv1_end->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims(), std::vector<int64_t>({1,16,100,190,16}));
    auto conv3_end = graph->FindNode("conv3");
    EXPECT_NE(conv3_end, nullptr);
    EXPECT_EQ(conv3_end->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims(), std::vector<int64_t>({1,16,190,100,16}));

    auto in_node = NodeUtils::GetInDataNodeByIndex(*conv3_end, 0);
    EXPECT_EQ(in_node->GetName(), conv1_end->GetName());
  };

  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(TransopFusionOptimizeTest, test_transdata_transdata_fusion_5hd_nhwc) {
  auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {1,100,190,3})
      .Attr(ATTR_NAME_INDEX, 0);
  auto conv1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {1,100,190,3});
  auto reshape1 = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {1,100,190,3});
  auto conv2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NHWC, ge::DT_FLOAT, {8,100,190,32});
  auto const1 = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, ge::DT_FLOAT, {4});

  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("conv1", conv1)->EDGE(0, 0)->NODE("reshape1", reshape1)
                            ->EDGE(0, 0)->NODE("conv2", conv2)->NODE("net_output", NETOUTPUT));
                  CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape1", reshape1));
                  CHAIN(NODE("conv1_weight", CONSTANT)->EDGE(0, 1)->NODE("conv1", conv1));
                  CHAIN(NODE("conv2_weight", CONSTANT)->EDGE(0, 1)->NODE("conv2", conv2));
                };
  auto compute_graph = ToComputeGraph(g1);
  auto test_graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto reshape = compute_graph->FindNode("reshape1");
  EXPECT_NE(reshape, nullptr);
  GeShape reshape_in({1,100,190,256});
  GeShape reshape_out({8,100,190,32});
  reshape->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(reshape_in, FORMAT_NHWC, DT_FLOAT));
  reshape->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(reshape_out, FORMAT_NHWC, DT_FLOAT));

  auto conv1_node = compute_graph->FindNode("conv1");
  EXPECT_NE(conv1_node, nullptr);
  conv1_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_in);
  conv1_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);

  GeRunningEnvFaker ge_env;
  FakeTransData5DNodeEngine(ge_env);

  DUMP_GRAPH_WHEN("OptimizeStage1_2");
  map<AscendString, AscendString> options;
  Session session(options);
  dlog_setlevel(0,0,0);
  session.AddGraph(99, test_graph, options);
  std::vector<InputTensorInfo> inputs;

  (void)session.BuildGraph(99, inputs);
  dlog_setlevel(3,3,0);

  CHECK_GRAPH(OptimizeStage1_2) {
    auto reshape_end = graph->FindNode("reshape1");
    EXPECT_EQ(reshape_end, nullptr);
    auto conv1_end = graph->FindNode("conv1");
    EXPECT_NE(conv1_end, nullptr);
    EXPECT_EQ(conv1_end->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims(), std::vector<int64_t>({1,16,100,190,16}));
    auto conv2_end = graph->FindNode("conv2");
    EXPECT_NE(conv2_end, nullptr);
    EXPECT_EQ(conv2_end->GetOpDesc()->MutableInputDesc(0)->GetShape().GetDims(), std::vector<int64_t>({8,2,100,190,16}));

    auto in_node = NodeUtils::GetInDataNodeByIndex(*conv2_end, 0);
    EXPECT_EQ(in_node->GetType(), TRANSDATA);
  };

  ge_env.Reset();
  ge_env.InstallDefault();
}