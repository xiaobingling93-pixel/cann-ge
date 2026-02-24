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
#include <iostream>
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/graph.h"
#include "graph/graph_buffer.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph_builder_utils.h"
#include "graph/utils/file_utils.h"
#include "graph/ge_attr_value.h"
#include "ge_ir.pb.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/tensor.h"
#include "graph/ascend_string.h"
#include "graph/types.h"
#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"
#include <debug/ge_attr_define.h>
#include <dirent.h>
#include <google/protobuf/text_format.h>
#include "graph_metadef/graph/utils/file_utils.h"
#include "proto/onnx/ge_onnx.pb.h"
using namespace ge;
namespace {
std::stringstream GetFilePathWhenDumpPathSet(const string &ascend_work_path) {
  std::stringstream dump_file_path;
  dump_file_path << ascend_work_path << "/pid_" << mmGetPid() << "_deviceid_" << GetContext().DeviceId() << "/";
  return dump_file_path;
}
std::string GetSpecificFilePath(const std::string &file_path, const string &suffix) {
  DIR *dir;
  struct dirent *ent;
  dir = opendir(file_path.c_str());
  if (dir == nullptr) {
    return "";
  }
  while ((ent = readdir(dir)) != nullptr) {
    if (strstr(ent->d_name, suffix.c_str()) != nullptr) {
      std::string d_name(ent->d_name);
      closedir(dir);
      return file_path + "/" + d_name;
    }
  }
  closedir(dir);
  return "";
}

ComputeGraphPtr BuildComputeGraphWithoutNetOutput() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data1 = builder.AddNode("Data1", "Data", 0, 1);
  auto transdata1 = builder.AddNode("Transdata1", "Transdata", 1, 1);
  builder.AddDataEdge(data1, 0, transdata1, 0);

  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto transdata2 = builder.AddNode("Transdata2", "Transdata", 1, 1);
  builder.AddDataEdge(data2, 0, transdata2, 0);
  auto graph = builder.GetGraph();
  return graph;
}

ComputeGraphPtr BuildComputeGraphWithNetOutput() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data1 = builder.AddNode("Data1", "Data", 0, 1);
  auto transdata1 = builder.AddNode("Transdata1", "Transdata", 1, 1);
  auto net_output = builder.AddNode("Netoutput", NETOUTPUT, 1, 0);
  builder.AddDataEdge(data1, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, net_output, 0);

  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto transdata2 = builder.AddNode("Transdata2", "Transdata", 1, 1);
  builder.AddDataEdge(data2, 0, transdata2, 0);
  auto graph = builder.GetGraph();
  return graph;
}
} // namespace
class UtestGraph : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

struct ExpectNodeInfo {
  std::string name;
  std::string type;
  std::map<int32_t, std::pair<std::string, int32_t>> input_node_name;
  std::map<int32_t, std::vector<std::pair<std::string, int32_t>>> output_node_name;
  std::vector<std::string> control_input_node_name;
  std::vector<std::string> control_output_node_name;
  int32_t input_desc_size;
  int32_t output_desc_size;
  ExpectNodeInfo(const std::string &in_name, const std::string &in_type,
                 const std::map<int32_t, std::pair<std::string, int32_t>> &in_input_node_name,
                 const std::map<int32_t, std::vector<std::pair<std::string, int32_t>>> &in_output_node_name,
                 const std::vector<std::string> &in_control_input_node_name,
                 const std::vector<std::string> &in_control_output_node_name,
                 const int32_t in_input_desc_size,
                 const int32_t in_output_desc_size)
    : name(in_name), type(in_type), input_node_name(in_input_node_name),
      output_node_name(in_output_node_name),
      control_input_node_name(in_control_input_node_name),
      control_output_node_name(in_control_output_node_name),
      input_desc_size(in_input_desc_size), output_desc_size(in_output_desc_size) {}
};

static ComputeGraphPtr BuildSubComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  return graph;
}

static void CheckNodeResult(const ComputeGraphPtr &compute_graph,
    std::vector<ExpectNodeInfo> &expect_result) {
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), expect_result.size());
  size_t i = 0UL;
  for (const auto &node : compute_graph->GetDirectNode()) {
    std::cout << "node name: " << node->GetName() << ", expect name: " << expect_result[i].name << std::endl;
    EXPECT_EQ(node->GetName(), expect_result[i].name);
    EXPECT_EQ(node->GetType(), expect_result[i].type);
    for (uint32_t in_index = 0UL; in_index < node->GetAllInDataAnchorsSize(); in_index++) {
      const auto in_anchor = node->GetInDataAnchor(in_index);
      ASSERT_NE(in_anchor, nullptr);
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      const auto iter = expect_result[i].input_node_name.find(in_index);
      ASSERT_EQ(peer_out_anchor == nullptr, iter == expect_result[i].input_node_name.end());
      if (iter != expect_result[i].input_node_name.end()) {
        EXPECT_EQ(iter->second.first, peer_out_anchor->GetOwnerNode()->GetName());
        EXPECT_EQ(iter->second.second, peer_out_anchor->GetIdx());
      }
    }
    for (uint32_t out_index = 0UL; out_index < node->GetAllOutDataAnchorsSize(); out_index++) {
      const auto out_anchor = node->GetOutDataAnchor(out_index);
      ASSERT_NE(out_anchor, nullptr);
      const auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
      const auto iter = expect_result[i].output_node_name.find(out_index);
      ASSERT_EQ(peer_in_anchors.size(), iter->second.size());
      for (size_t peer_in_index = 0UL; peer_in_index < peer_in_anchors.size(); peer_in_index++) {
        EXPECT_EQ(iter->second[peer_in_index].first, peer_in_anchors.at(peer_in_index)->GetOwnerNode()->GetName());
        EXPECT_EQ(iter->second[peer_in_index].second, peer_in_anchors.at(peer_in_index)->GetIdx());
      }
    }
    const auto in_control_anchor = node->GetInControlAnchor();
    ASSERT_NE(in_control_anchor, nullptr);
    const auto peer_out_control_anchors = in_control_anchor->GetPeerOutControlAnchors();
    ASSERT_EQ(peer_out_control_anchors.size(), expect_result[i].control_input_node_name.size());
    for (size_t control_out_index = 0UL; control_out_index < peer_out_control_anchors.size(); control_out_index++) {
      EXPECT_EQ(expect_result[i].control_input_node_name.at(control_out_index),
          peer_out_control_anchors.at(control_out_index)->GetOwnerNode()->GetName());
    }
    const auto out_control_anchor = node->GetOutControlAnchor();
    ASSERT_NE(out_control_anchor, nullptr);
    const auto peer_in_control_anchors = out_control_anchor->GetPeerInControlAnchors();
    ASSERT_EQ(peer_in_control_anchors.size(), expect_result[i].control_output_node_name.size());
    for (size_t control_in_index = 0UL; control_in_index < peer_in_control_anchors.size(); control_in_index++) {
      EXPECT_EQ(expect_result[i].control_output_node_name[control_in_index],
          peer_in_control_anchors.at(control_in_index)->GetOwnerNode()->GetName());
    }
    const auto op_desc = node->GetOpDesc();
    ASSERT_NE(op_desc, nullptr);
    EXPECT_EQ(op_desc->GetAllInputsSize(), expect_result[i].input_desc_size);
    EXPECT_EQ(op_desc->GetOutputsSize(), expect_result[i].output_desc_size);
    i++;
  }
}

// construct graph which contains subgraph
static ComputeGraphPtr BuildComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  transdata->GetOpDesc()->AddSubgraphName("subgraph");
  transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  // add subgraph
  transdata->SetOwnerComputeGraph(graph);
  ComputeGraphPtr subgraph = BuildSubComputeGraph();
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(transdata);
  graph->AddSubgraph("subgraph", subgraph);
  return graph;
}

TEST_F(UtestGraph, copy_graph_01) {
  ge::OpDescPtr add_op(new ge::OpDesc("add1", "Add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  ge::Graph copy_graph("copy_graph");
  ASSERT_EQ(copy_graph.CopyFrom(graph), ge::GRAPH_SUCCESS);
  Graph graph2("graph2");
  ASSERT_EQ(copy_graph.CopyFrom(graph2), GRAPH_FAILED);

  auto cp_compute_graph = ge::GraphUtilsEx::GetComputeGraph(copy_graph);
  ASSERT_NE(cp_compute_graph, nullptr);
  ASSERT_NE(cp_compute_graph, compute_graph);
  ASSERT_EQ(cp_compute_graph->GetDirectNodesSize(), 2);
  auto cp_add_node = cp_compute_graph->FindNode("add1");
  ASSERT_NE(cp_add_node, nullptr);
  ASSERT_NE(cp_add_node, add_node);
}

TEST_F(UtestGraph, copy_graph_02) {
  ge::OpDescPtr if_op(new ge::OpDesc("if", "If"));
  if_op->AddDynamicInputDesc("input", 1);
  if_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto if_node = compute_graph->AddNode(if_op);
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  ge::Graph copy_graph("copy_graph");

  if_op->AddSubgraphName("then_branch");
  if_op->AddSubgraphName("else_branch");
  if_op->SetSubgraphInstanceName(0, "then");
  if_op->SetSubgraphInstanceName(1, "else");

  ge::OpDescPtr add_op1(new ge::OpDesc("add1", "Add"));
  add_op1->AddDynamicInputDesc("input", 2);
  add_op1->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> then_compute_graph(new ge::ComputeGraph("then"));
  auto add_node1 = then_compute_graph->AddNode(add_op1);
  then_compute_graph->SetParentNode(if_node);
  then_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(then_compute_graph);

  ge::OpDescPtr add_op2(new ge::OpDesc("add2", "Add"));
  add_op2->AddDynamicInputDesc("input", 2);
  add_op2->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> else_compute_graph(new ge::ComputeGraph("else"));
  auto add_node2 = else_compute_graph->AddNode(add_op2);
  else_compute_graph->SetParentNode(if_node);
  else_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(else_compute_graph);

  ASSERT_EQ(copy_graph.CopyFrom(graph), ge::GRAPH_SUCCESS);

  auto cp_compute_graph = ge::GraphUtilsEx::GetComputeGraph(copy_graph);
  ASSERT_NE(cp_compute_graph, nullptr);
  ASSERT_NE(cp_compute_graph, compute_graph);
  ASSERT_EQ(cp_compute_graph->GetDirectNodesSize(), 2);
  auto cp_if_node = cp_compute_graph->FindNode("if");
  ASSERT_NE(cp_if_node, nullptr);
  ASSERT_NE(cp_if_node, if_node);

  auto cp_then_compute_graph = cp_compute_graph->GetSubgraph("then");
  ASSERT_NE(cp_then_compute_graph, nullptr);
  ASSERT_NE(cp_then_compute_graph, then_compute_graph);
  ASSERT_EQ(cp_then_compute_graph->GetDirectNodesSize(), 2);
  auto cp_add_node1 = cp_then_compute_graph->FindNode("add1");
  ASSERT_NE(cp_add_node1, nullptr);
  ASSERT_NE(cp_add_node1, add_node1);

  auto cp_else_compute_graph = cp_compute_graph->GetSubgraph("else");
  ASSERT_NE(cp_else_compute_graph, nullptr);
  ASSERT_NE(cp_else_compute_graph, else_compute_graph);
  ASSERT_EQ(cp_else_compute_graph->GetDirectNodesSize(), 2);
  auto cp_add_node2 = cp_else_compute_graph->FindNode("add2");
  ASSERT_NE(cp_add_node2, nullptr);
  ASSERT_NE(cp_add_node2, add_node2);
}

REG_OP(Mul)
    .OP_END_FACTORY_REG(Mul)
IMPL_INFER_VALUE_RANGE_FUNC(Mul, func){
  std::cout << "test" << std::endl;
  return GRAPH_SUCCESS;
}

REG_OP(Test2)
    .OP_END_FACTORY_REG(Test2)
IMPL_INFER_VALUE_RANGE_FUNC(Test2, func2){
  std::cout << "test" << std::endl;
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraph, test_infer_value_range_register_succ) {
  string op_type = "Add";
  INFER_VALUE_RANGE_DEFAULT_REG(Add);
  INFER_VALUE_RANGE_DEFAULT_REG(Test1);
  auto para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, true);
  ASSERT_EQ(para.infer_value_func, nullptr);

  op_type = "Mul";
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Mul, INPUT_HAS_VALUE_RANGE, func);
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Test2, INPUT_IS_DYNAMIC, func2);
  para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, true);
  ASSERT_NE(para.infer_value_func, nullptr);

  op_type = "Sub";
  para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, false);
}

TEST_F(UtestGraph, IsRefFromRefData_HasNoAttr_ReturnFalse) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  auto out_data_anchor = transdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = true;
  EXPECT_EQ(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
  EXPECT_FALSE(is_ref_from_other);
}

TEST_F(UtestGraph, IsRefFromRefData_VarNameNotExist_ReturnFalse) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  ge::AttrUtils::SetStr(transdata->GetOpDesc()->MutableOutputDesc(0), "ref_var_src_var_name", "not_exist");
  auto out_data_anchor = transdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = true;
  EXPECT_EQ(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
  EXPECT_FALSE(is_ref_from_other);
}

TEST_F(UtestGraph, IsRefFromRefData_VarNameNodeIsNotRefData_ReturnFalse) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  ge::AttrUtils::SetStr(transdata->GetOpDesc()->MutableOutputDesc(0), "ref_var_src_var_name", "NetOutput");
  auto out_data_anchor = transdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = true;
  EXPECT_EQ(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
  EXPECT_FALSE(is_ref_from_other);
}

TEST_F(UtestGraph, IsRefFromRefData_ReturnTrue) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  ge::AttrUtils::SetStr(transdata->GetOpDesc()->MutableOutputDesc(0), "ref_var_src_var_name", "ref_data");
  auto out_data_anchor = transdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = false;
  EXPECT_EQ(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
  EXPECT_TRUE(is_ref_from_other);
}

TEST_F(UtestGraph, RefDataInSubgraph_IsRefFromInnerData_ReturnTrue) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto partitioned_call = builder.AddNode("partitionedcall", "PartitionedCall", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, partitioned_call, 0);
  builder.AddDataEdge(partitioned_call, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph");
  auto sub_data = sub_builder.AddNode("sub_Data", "Data", 0, 1);
  AttrUtils::SetInt(sub_data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sub_refdata = sub_builder.AddNode("sub_RefData", "RefData", 0, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_Netoutput", "NetOutput", 1, 0);
  builder.AddControlEdge(sub_data, sub_refdata);
  builder.AddDataEdge(sub_refdata, 0, sub_netoutput, 0);
  auto sub_graph = sub_builder.GetGraph();

  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(partitioned_call);
  graph->AddSubgraph("subgraph", sub_graph);

  auto out_data_anchor = sub_refdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = false;
  EXPECT_EQ(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
  EXPECT_TRUE(is_ref_from_other);
}

TEST_F(UtestGraph, RefDataInSubgraph_IsRefFromInnerData_PeerInCtrolNotData_InvalidGraph_ReturnFalse) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto partitioned_call = builder.AddNode("partitionedcall", "PartitionedCall", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, partitioned_call, 0);
  builder.AddDataEdge(partitioned_call, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph");
  auto sub_cast = sub_builder.AddNode("sub_Data", "Cast", 0, 1);
  auto sub_refdata = sub_builder.AddNode("sub_RefData", "RefData", 0, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_Netoutput", "NetOutput", 1, 0);
  builder.AddControlEdge(sub_cast, sub_refdata);
  builder.AddDataEdge(sub_refdata, 0, sub_netoutput, 0);
  auto sub_graph = sub_builder.GetGraph();

  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(partitioned_call);
  graph->AddSubgraph("subgraph", sub_graph);

  auto out_data_anchor = sub_refdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = false;
  EXPECT_NE(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
}

TEST_F(UtestGraph, RefDataInSubgraph_IsRefFromInnerData_MultiPeerInCtrl_InvalidGraph_ReturnFalse) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ref_data = builder.AddNode("ref_data", "RefData", 0, 1);
  auto partitioned_call = builder.AddNode("partitionedcall", "PartitionedCall", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(ref_data, 0, partitioned_call, 0);
  builder.AddDataEdge(partitioned_call, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph");
  auto sub_data = sub_builder.AddNode("sub_Data", "Data", 0, 1);
  AttrUtils::SetInt(sub_data->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto sub_cast = sub_builder.AddNode("sub_cast", "Cast", 0, 1);
  auto sub_refdata = sub_builder.AddNode("sub_RefData", "RefData", 0, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_Netoutput", "NetOutput", 1, 0);
  builder.AddControlEdge(sub_cast, sub_refdata);
  builder.AddDataEdge(sub_refdata, 0, sub_netoutput, 0);
  auto sub_graph = sub_builder.GetGraph();

  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(partitioned_call);
  graph->AddSubgraph("subgraph", sub_graph);

  auto out_data_anchor = sub_refdata->GetOutDataAnchor(0);
  ASSERT_NE(out_data_anchor, nullptr);

  NodePtr node = nullptr;
  bool is_ref_from_other = false;
  EXPECT_NE(GraphUtils::CheckIsRefFromOther(out_data_anchor, node, is_ref_from_other), GRAPH_SUCCESS);
}

REG_OP(Shape)
    .OP_END_FACTORY_REG(Shape)
IMPL_INFER_VALUE_RANGE_FUNC(Shape, ShapeValueInfer){
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range;
  op_desc->MutableInputDesc(0)->GetShapeRange(in_shape_range);
  if (!in_shape_range.empty()) {
    output_tensor_desc->SetValueRange(in_shape_range);
  }
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraph, test_value_range_infer_and_set_get) {
  using std::make_pair;
  string op_type = "Shape";
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueInfer);
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto shape_op_desc = std::make_shared<OpDesc>("node_name", op_type);
  GeTensorDesc tensor_desc(GeShape({-1, -1, 4, 192}), ge::FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 100), make_pair(1, 240),
                                                          make_pair(4, 4),   make_pair(192, 192)};
  tensor_desc.SetShapeRange(shape_range);
  shape_op_desc->AddInputDesc(tensor_desc);
  GeTensorDesc out_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, DT_INT32);
  shape_op_desc->AddOutputDesc(out_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(shape_node);
  auto ret = OpDescUtilsEx::CallInferValueRangeFunc(shape_node->GetOpDesc(), op);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto output_0_desc = shape_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> value_range;
  output_0_desc.GetValueRange(value_range);
  EXPECT_EQ(value_range.size(), 4);

  std::vector<int64_t> target_value_range = {1, 100, 1, 240, 4, 4, 192, 192};
  std::vector<int64_t> output_value_range;
  for (auto pair : value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);
}

TEST_F(UtestGraph, get_all_graph_nodes) {
  ComputeGraphPtr graph = BuildComputeGraph();
  auto nodes = graph->GetAllNodes();
  EXPECT_EQ(nodes.size(), 5);

  Graph graph2("Test");
  auto nodes_empty = graph2.GetAllNodes();
  EXPECT_EQ(nodes_empty.size(), 0);
}

TEST_F(UtestGraph, get_all_subgraphs) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto subgraphs = graph.GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  Graph graph2("empty");
  auto subgraphs_empty = graph2.GetAllSubgraphs();
  EXPECT_EQ(subgraphs_empty.size(), 0);
}

TEST_F(UtestGraph, get_subgraph) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto subgraph = graph.GetSubGraph("subgraph");
  EXPECT_NE(subgraph, nullptr);

  auto subgraph1 = graph.GetSubGraph("subgraph1");
  EXPECT_EQ(subgraph1, nullptr);
}

static ComputeGraphPtr BuildSubComputeGraph1() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph1");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UtestGraph, add_subgraph) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto subgraphs = graph.GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  ComputeGraphPtr sub_compute_graph = BuildSubComputeGraph1();
  sub_compute_graph->SetParentGraph(compute_graph);
  sub_compute_graph->SetParentNode(compute_graph->GetSubgraph("subgraph")->GetParentNode());
  Graph subgraph = GraphUtilsEx::CreateGraphFromComputeGraph(sub_compute_graph);

  EXPECT_EQ(graph.AddSubGraph(subgraph), GRAPH_SUCCESS);
  subgraphs = graph.GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  EXPECT_EQ(graph.AddSubGraph(subgraph), GRAPH_FAILED);
  subgraphs = graph.GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);
}

TEST_F(UtestGraph, remove_subgraph) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto subgraphs = graph.GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 1);

  EXPECT_EQ(graph.RemoveSubgraph("subgraph"), GRAPH_SUCCESS);
  subgraphs = graph.GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 0);
}

TEST_F(UtestGraph, SetOutputs_ops) {
  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<ge::Operator> outputs = {op1, op2, op3};

  Graph graph;
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 0);
  // EXPECT_TRUE(graph.impl_->output_name_.empty()); // impl缺少头文件，找不到声明
}

TEST_F(UtestGraph, SetOutputsRepalceExistedNetOutputDataEdge) {
  auto compute_graph = BuildComputeGraphWithNetOutput();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto transdata2 = compute_graph->FindNode("Transdata2");
  auto transdata2_op = OpDescUtils::CreateOperatorFromNode(transdata2);
  graph.SetOutputs({transdata2_op});
  auto net_output = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(net_output, nullptr);
  ASSERT_EQ(net_output->GetInDataNodes().size(), 1U);
  ASSERT_EQ(net_output->GetInDataNodes().at(0)->GetName(), "Transdata2");
  ASSERT_EQ(net_output->GetInControlNodes().size(), 0U);
}

TEST_F(UtestGraph, SetTargetsAndSetOutputs) {
  auto compute_graph = BuildComputeGraphWithoutNetOutput();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto transdata1 = compute_graph->FindNode("Transdata1");
  auto transdata2 = compute_graph->FindNode("Transdata2");
  auto transdata1_op = OpDescUtils::CreateOperatorFromNode(transdata1);
  auto transdata2_op = OpDescUtils::CreateOperatorFromNode(transdata2);

  graph.SetTargets({transdata1_op}).SetOutputs({transdata2_op});
  auto net_output = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(net_output, nullptr);
  ASSERT_EQ(net_output->GetInDataNodes().size(), 1U);
  ASSERT_EQ(net_output->GetInDataNodes().at(0)->GetName(), "Transdata2");
  ASSERT_EQ(net_output->GetInControlNodes().size(), 1U);
}

TEST_F(UtestGraph, MultiSetOutputs) {
  auto compute_graph = BuildComputeGraphWithoutNetOutput();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto transdata1 = compute_graph->FindNode("Transdata1");
  auto transdata2 = compute_graph->FindNode("Transdata2");
  auto transdata1_op = OpDescUtils::CreateOperatorFromNode(transdata1);
  auto transdata2_op = OpDescUtils::CreateOperatorFromNode(transdata2);

  graph.SetOutputs({transdata1_op, transdata2_op}).SetOutputs({transdata1_op});
  auto net_output = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(net_output, nullptr);
  ASSERT_EQ(net_output->GetInDataNodesSize(), 1U);
  ASSERT_EQ(net_output->GetInDataNodes().at(0)->GetName(), "Transdata1");
}

TEST_F(UtestGraph, SetOutputs_string) {
  using std::make_pair;
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::string op_n1 = std::string("add");
  std::string op_n2 = std::string("op2");
  std::string op_n3 = std::string("op3");

  std::vector<std::pair<Operator, std::string>> outputs = {make_pair(op1, op_n1), make_pair(op2, op_n2),
                                                          make_pair(op3, op_n3)};
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, SetOutputs_AscendString) {
  using std::make_pair;
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  AscendString op_n1 = AscendString("add");
  AscendString op_n2 = AscendString("op2");
  AscendString op_n3 = AscendString("op3");

  std::vector<std::pair<Operator, AscendString>> outputs = {make_pair(op1, op_n1), make_pair(op2, op_n2),
                                                          make_pair(op3, op_n3)};
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, SetOutputs_Index) {
  using std::make_pair;
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  Graph graph2;

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<size_t> vec_index1 = {0,1,2};
  std::vector<size_t> vec_index2 = {0};
  std::vector<size_t> vec_index3 = {0};

  std::vector<std::pair<Operator, std::vector<size_t>>> outputs = {make_pair(op1, vec_index1),
    make_pair(op2, vec_index2),  make_pair(op3, vec_index3)};
  graph2.SetOutputs(outputs);
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 2); // add + netoutput
}

TEST_F(UtestGraph, SetTargets) {
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  Graph graph2;

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<size_t> vec_index1 = {0,1,2};
  std::vector<size_t> vec_index2 = {0};
  std::vector<size_t> vec_index3 = {0};

  std::vector<ge::Operator> targets = {op1, op2, op3};

  graph2.SetTargets(targets);
  graph.SetTargets(targets);
  EXPECT_EQ(graph.GetAllNodes().size(), 2); // add + netoutput
}

TEST_F(UtestGraph, SetNeedIteration) {
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  Graph graph2;

  graph2.SetNeedIteration(true);
  graph.SetNeedIteration(false);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, GetDirectNode) {
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  ge::OpDescPtr add_op2(new ge::OpDesc("add_1", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph2 = nullptr;
  Graph graph2 = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph2);

  Graph graph3;

  std::vector<GNode> gnodes, gnodes2, gnodes3;

  gnodes = graph.GetDirectNode();
  gnodes2 = graph2.GetDirectNode();
  gnodes3 = graph3.GetDirectNode();
  EXPECT_EQ(gnodes.size(), 1);
}

TEST_F(UtestGraph, RemoveNode) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  auto v_nodes = cgp->GetAllNodes();
  EXPECT_EQ(v_nodes.size(), 5);

  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  auto nodes = graph.GetAllNodes();
  graph.RemoveNode(nodes[4]);
  EXPECT_EQ(graph.GetAllNodes().size(), 4);

  graph.RemoveNode(nodes[0], true);
  EXPECT_EQ(graph.GetAllNodes().size(), 3);
}

TEST_F(UtestGraph, AddRemoveEdge1) {
  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");

  Graph graph("a_graph");
  Graph graph2;

  GNode node1 = graph.AddNodeByOp(op1);
  GNode node2 = graph.AddNodeByOp(op2);
  GNode node3 = graph.AddNodeByOp(op3);

  auto ret =graph.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.AddControlEdge(node2, node3);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);

  graph2.AddNodeByOp(op1);
  ret =graph2.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph2.AddControlEdge(node2, node3);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph2.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, AddRemoveEdge2) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  ComputeGraphPtr cgp = builder.GetGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  auto nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 1);

  GNode node1 = nodes[0];
  GNode node2;

  auto ret =graph.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.AddControlEdge(node1, node2);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, AddRemoveEdge3) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  ComputeGraphPtr cgp = builder.GetGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  auto nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  GNode node1 = nodes[0];
  GNode node2 = nodes[1];
  GNode node3 = nodes[2];

  auto ret = graph.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph.AddControlEdge(node2, node3);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, ConstructFromInputs1) {
  Graph graph;
  Operator op1 = Operator("op1");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  auto ret = graph.ConstructFromInputs({}, name);
  EXPECT_EQ(ret, nullptr);

  ret = graph.ConstructFromInputs(inputs, AscendString(nullptr));
  EXPECT_EQ(ret, nullptr);

  ret = graph.ConstructFromInputs(inputs, name);
  EXPECT_EQ(ret, nullptr);
}

REG_OP(Phony0)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Phony0);

REG_OP(Phony1)
    .DYNAMIC_INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Phony1);

REG_OP(Phony2)
    .INPUT(x,
           TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                       DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Phony2);

TEST_F(UtestGraph, ConstructFromInputs2) {
  Graph graph;
  Operator op1 = op::Phony0("op1");
  Operator op2 = op::Phony1("op2");
  Operator op3 = op::Phony2("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  auto ret = graph.ConstructFromInputs(inputs, name);
  EXPECT_NE(ret, nullptr);
}

TEST_F(UtestGraph, SaveLoadFile) {
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");

  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  auto ret = graph.SaveToFile(nullptr);
  EXPECT_EQ(ret, GRAPH_FAILED);

  ret = graph.SaveToFile("./ut_graph1.txt");
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = graph.SaveToFile(std::string("./ut_graph2.txt"));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  Graph graph2;
  ret = graph2.LoadFromFile(nullptr);
  EXPECT_EQ(ret, GRAPH_FAILED);

  Graph graph3;
  ret = graph3.LoadFromFile("./ut_graph1.txt");
  EXPECT_NE(ret, GRAPH_FAILED);

  Graph graph4;
  ret = graph4.LoadFromFile(std::string("./ut_graph2.txt"));
  EXPECT_NE(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, LoadFromSerializedModelArray_InvalidParams) {
  ge::proto::ModelDef model_def;
  auto *graph_def = model_def.add_graph();
  graph_def->set_name("serialized_model_array_graph");

  Graph graph;
  EXPECT_NE(graph.LoadFromSerializedModelArray(nullptr, 0), GRAPH_SUCCESS);

  std::string serialized;
  EXPECT_NE(graph.LoadFromSerializedModelArray(reinterpret_cast<const uint8_t*>(serialized.c_str()), 0), GRAPH_SUCCESS);

  serialized = "abc";
  EXPECT_NE(graph.LoadFromSerializedModelArray(reinterpret_cast<const uint8_t*>(serialized.c_str()), serialized.size()), GRAPH_SUCCESS);
}


std::vector<std::string> CreateOpDef(ge::proto::GraphDef *def, const std::string &type, const std::vector<std::string> &inputs,
                                     size_t num_outputs, std::vector<std::string> subgraphs = {}) {
  auto name = type + std::to_string(def->op_size());
  auto *op_def = def->add_op();
  op_def->set_name(name);
  op_def->set_type(type);


  auto op_desc_attr = op_def->mutable_attr();
  proto::AttrDef input_desc_name;
  proto::AttrDef input_desc_index;
  proto::AttrDef output_desc_name;
  proto::AttrDef output_desc_index;

  for (size_t i = 0U; i < inputs.size(); ++i) {
    op_def->add_input_desc();
    *op_def->add_input() = inputs[i];

    input_desc_name.mutable_list()->add_s(std::string("x") + std::to_string(i));
    input_desc_index.mutable_list()->add_i(i);
  }
  std::vector<std::string> outputs;
  for (size_t i = 0U; i < num_outputs; ++i) {
    op_def->add_output_desc();
    outputs.push_back(op_def->name() + ":" + std::to_string(i));

    output_desc_name.mutable_list()->add_s(std::string("y") + std::to_string(i));
    output_desc_index.mutable_list()->add_i(i);
  }

  (void) op_desc_attr->insert({"_input_name_key", input_desc_name});
  (void) op_desc_attr->insert({"_input_name_value", input_desc_index});

  (void) op_desc_attr->insert({"_output_name_key", output_desc_name});
  (void) op_desc_attr->insert({"_output_name_value", output_desc_index});

  for (auto &subgraph : subgraphs) {
      op_def->add_subgraph_name(subgraph);
  }

  if (num_outputs == 0) {
    outputs.push_back(op_def->name());
  }

  return outputs;
}


std::string GetStringBeforeColon(const std::string& str) {
    size_t pos = str.find(':');
    if (pos != std::string::npos) {
        return str.substr(0, pos);
    } else {
        return str;
    }
}


void AssertOpMatch(ge::ComputeGraphPtr &compute_graph, const std::vector<std::string> &op,
                   const std::vector<std::string> &inputs, size_t num_outputs) {
  auto op_name = GetStringBeforeColon(op[0]);
  auto data = compute_graph->FindNode(op_name);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(data->GetInDataNodesAndAnchors().size(), inputs.size());
  size_t index = 0U;
  for (auto &node_and_anchor : data->GetInDataNodesAndAnchors()) {
    auto input = node_and_anchor.first->GetName() + ":" + std::to_string(node_and_anchor.second->GetIdx());
    ASSERT_EQ(input, inputs[index]);
    index++;
  }
  auto in_name_idx = data->GetOpDesc()->GetAllInputName();
  ASSERT_EQ(in_name_idx.size(), inputs.size());
  index = 0U;
  for (auto &name_idx : in_name_idx) {
    ASSERT_EQ(name_idx.first, "x" + std::to_string(index));
    ASSERT_EQ(name_idx.second, index);
    index++;
  }
  auto out_name_idx = data->GetOpDesc()->GetAllOutputName();
  ASSERT_EQ(out_name_idx.size(), num_outputs);
  index = 0U;
  for (auto &name_idx : out_name_idx) {
    ASSERT_EQ(name_idx.first, "y" + std::to_string(index));
    ASSERT_EQ(name_idx.second, index);
    index++;
  }
}


TEST_F(UtestGraph, LoadFromSerializedModelArray_NoSubGraph) {
  ge::proto::ModelDef model_def;
  auto *graph_def = model_def.add_graph();
  graph_def->set_name("root_graph");

  auto data = CreateOpDef(graph_def, "Data", {}, 1);
  auto abs = CreateOpDef(graph_def, "Abs", data, 1);
  auto sqrt = CreateOpDef(graph_def, "Add", {data[0], abs[0]}, 1);
  auto netoutput = CreateOpDef(graph_def, "NetOutput", {abs[0], sqrt[0]}, 0);

  Graph graph;
  auto serialized = model_def.SerializeAsString();
  ASSERT_EQ(graph.LoadFromSerializedModelArray(serialized.c_str(), serialized.size()), GRAPH_SUCCESS);

  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_EQ(compute_graph->GetName(), graph_def->name());

  AssertOpMatch(compute_graph, data, {}, 1);
  AssertOpMatch(compute_graph, abs, data, 1);
  AssertOpMatch(compute_graph, sqrt, {data[0], abs[0]}, 1);
  AssertOpMatch(compute_graph, netoutput, {abs[0], sqrt[0]}, 0);
}

TEST_F(UtestGraph, LoadFromSerializedModelArray_WithSubGraph) {
  ge::proto::ModelDef model_def;
  auto *graph_def = model_def.add_graph();
  graph_def->set_name("root_graph");
  auto func = CreateOpDef(graph_def, "FuncOp", {}, 0, {"sub_graph"});

  auto *sub_graph = model_def.add_graph();
  sub_graph->set_name("sub_graph");
  auto data = CreateOpDef(sub_graph, "Data", {}, 1);
  auto abs = CreateOpDef(sub_graph, "Abs", data, 1);
  auto sqrt = CreateOpDef(sub_graph, "Add", {data[0], abs[0]}, 1);
  auto netoutput = CreateOpDef(sub_graph, "NetOutput", {abs[0], sqrt[0]}, 0);

  Graph graph;
  auto serialized = model_def.SerializeAsString();
  ASSERT_EQ(graph.LoadFromSerializedModelArray(serialized.c_str(), serialized.size()), GRAPH_SUCCESS);

  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_EQ(compute_graph->GetName(), graph_def->name());

  ASSERT_EQ(compute_graph->GetAllSubgraphs().size(), 1U);
  auto sub_compute_graph = compute_graph->GetSubgraph("sub_graph");
  ASSERT_NE(sub_compute_graph, nullptr);
  ASSERT_EQ(sub_compute_graph->GetName(), "sub_graph");

  auto func_op = compute_graph->FindNode(GetStringBeforeColon(func[0]));
  ASSERT_NE(func_op, nullptr);
  ASSERT_EQ(sub_compute_graph->GetParentNode(), func_op);
  ASSERT_EQ(sub_compute_graph->GetParentGraph(), compute_graph);

  AssertOpMatch(sub_compute_graph, data, {}, 1);
  AssertOpMatch(sub_compute_graph, abs, data, 1);
  AssertOpMatch(sub_compute_graph, sqrt, {data[0], abs[0]}, 1);
  AssertOpMatch(sub_compute_graph, netoutput, {abs[0], sqrt[0]}, 0);
}

TEST_F(UtestGraph, LoadFromSerializedModelArray_WithNestedSubGraph) {
  ge::proto::ModelDef model_def;
  auto *graph_def = model_def.add_graph();
  graph_def->set_name("root_graph");
  auto func = CreateOpDef(graph_def, "FuncOp", {}, 0, {"sub_graph"});

  auto *sub_graph0 = model_def.add_graph();
  sub_graph0->set_name("sub_graph");
  auto func1 = CreateOpDef(sub_graph0, "FuncOp1", {}, 0, {"sub_graph1"});

  auto *sub_graph1 = model_def.add_graph();
  sub_graph1->set_name("sub_graph1");
  auto data = CreateOpDef(sub_graph1, "Data", {}, 1);
  auto abs = CreateOpDef(sub_graph1, "Abs", data, 1);
  auto sqrt = CreateOpDef(sub_graph1, "Add", {data[0], abs[0]}, 1);
  auto netoutput = CreateOpDef(sub_graph1, "NetOutput", {abs[0], sqrt[0]}, 0);

  Graph graph;
  auto serialized = model_def.SerializeAsString();
  ASSERT_EQ(graph.LoadFromSerializedModelArray(serialized.c_str(), serialized.size()), GRAPH_SUCCESS);

  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_EQ(compute_graph->GetName(), graph_def->name());

  ASSERT_EQ(compute_graph->GetAllSubgraphs().size(), 2U);
  auto sub_compute_graph = compute_graph->GetSubgraph("sub_graph");
  ASSERT_NE(sub_compute_graph, nullptr);
  ASSERT_EQ(sub_compute_graph->GetName(), "sub_graph");

  auto sub_compute_graph1 = compute_graph->GetSubgraph("sub_graph1");
  ASSERT_NE(sub_compute_graph1, nullptr);
  ASSERT_EQ(sub_compute_graph1->GetName(), "sub_graph1");

  auto func_op = compute_graph->FindNode(GetStringBeforeColon(func[0]));
  ASSERT_NE(func_op, nullptr);
  ASSERT_EQ(sub_compute_graph->GetParentNode(), func_op);
  ASSERT_EQ(sub_compute_graph->GetParentGraph(), compute_graph);

  auto func_op1 = sub_compute_graph->FindNode(GetStringBeforeColon(func1[0]));
  ASSERT_NE(func_op1, nullptr);
  ASSERT_EQ(sub_compute_graph1->GetParentNode(), func_op1);
  ASSERT_EQ(sub_compute_graph1->GetParentGraph(), sub_compute_graph);

  AssertOpMatch(sub_compute_graph1, data, {}, 1);
  AssertOpMatch(sub_compute_graph1, abs, data, 1);
  AssertOpMatch(sub_compute_graph1, sqrt, {data[0], abs[0]}, 1);
  AssertOpMatch(sub_compute_graph1, netoutput, {abs[0], sqrt[0]}, 0);
}

TEST_F(UtestGraph, SaveAndLoadMemWithBuffer) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph1 = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  GraphBuffer buf1;
  auto ret = graph1.SaveToMem(buf1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  Graph graph2;
  ret = graph2.LoadFromMem(buf1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  GraphBuffer buf2;
  ret = graph2.SaveToMem(buf2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  EXPECT_EQ(buf1.GetSize(), buf2.GetSize());
  EXPECT_EQ(memcmp(buf1.GetData(), buf2.GetData(), buf1.GetSize()), 0);
}

TEST_F(UtestGraph, SaveAndLoadMemWithData) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph1 = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  GraphBuffer buf1;
  auto ret = graph1.SaveToMem(buf1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  Graph graph2;
  ret = graph2.LoadFromMem(buf1.GetData(), buf1.GetSize());
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  GraphBuffer buf2;
  ret = graph2.SaveToMem(buf2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  EXPECT_EQ(buf1.GetSize(), buf2.GetSize());
  EXPECT_EQ(memcmp(buf1.GetData(), buf2.GetData(), buf1.GetSize()), 0);
}

TEST_F(UtestGraph, LoadFromMemFailed) {
  GraphBuffer buf;
  Graph graph;
  auto ret = graph.LoadFromMem(buf.GetData(), buf.GetSize());
  EXPECT_NE(ret, GRAPH_SUCCESS);

  ret = graph.LoadFromMem(nullptr, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, ErrorCodeCheck) {
  EXPECT_EQ(ge::FAILED, 4294967295);
  EXPECT_EQ(ge::END_OF_SEQUENCE, 1343225863);
  EXPECT_EQ(ge::GE_GRAPH_SAVE_WEIGHTS_FAILED, 1343242286);

  EXPECT_EQ(strcmp(GE_GET_ERRORNO_STR(ge::END_OF_SEQUENCE).c_str(), "End of sequence!"), 0);
  EXPECT_EQ(strcmp(GE_GET_ERRORNO_STR(ge::FAILED).c_str(), "failed"), 0);
  EXPECT_EQ(strcmp(GE_GET_ERRORNO_STR(ge::GE_GRAPH_SAVE_WEIGHTS_FAILED).c_str(),
    "OMG Save Weights to Model failed."), 0);
}

TEST_F(UtestGraph, GetName) {
  Graph graph;
  AscendString name;
  auto ret = graph.GetName(name);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, RecoverGraphOperators) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  auto ret = GraphUtilsEx::RecoverGraphOperators(graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, GetOpName) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  Operator op1("add");
  auto ret = graph.AddOp(op1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<std::string> op_names1;
  ret = graph.GetAllOpName(op_names1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<AscendString> op_names2;
  ret = graph.GetAllOpName(op_names2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, FindOpByName) {
  Graph graph;
  Operator op1 = op::Phony0("op1");
  Operator op2 = op::Phony1("op2");
  Operator op3 = op::Phony2("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  GraphPtr gptr = Graph::ConstructFromInputs(inputs, name);

  EXPECT_EQ(gptr->GetAllNodes().size(), 2);

  Operator op1_2;
  auto ret = gptr->FindOpByName(nullptr, op1_2);
  ret = gptr->FindOpByName("op1", op1_2);
  EXPECT_EQ(ret, GRAPH_FAILED);

  Operator op2_2;
  ret = gptr->FindOpByName(std::string("op2"), op2_2);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, FindOpByType) {
  Graph graph;
  Operator op1 = op::Phony0("op1");
  Operator op2 = op::Phony1("op2");
  Operator op3 = op::Phony2("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  GraphPtr gptr = Graph::ConstructFromInputs(inputs, name);

  std::vector<ge::Operator> op1_2;
  auto ret = gptr->FindOpByType(nullptr, op1_2);
  ret = gptr->FindOpByType("const", op1_2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::Operator> op2_2;
  ret = gptr->FindOpByType(std::string("data"), op2_2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, SaveInvalidPath) {
  std::vector<Operator> inputs{};
  std::vector<Operator> outputs{};
  Graph graph("empty_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  std::string file_name = "....1263713612～";
  EXPECT_EQ(graph.SaveToFile(file_name), GRAPH_FAILED);
}

/*
  Data Data Const Variable
   |     |    |    / |
    \    |   /    /  |
      ConcatV2 --    |       DATA
     |   \          /         |
     |     IdentityN    -------  
     \         |
      ---- MatmulV2                                          
*/
TEST_F(UtestGraph, TestGenerateGraphWithControlEdge) {
  ge::Operator data1 = ge::Operator("Data_0", "Data");
  ge::Operator data2 = ge::Operator("Data_1", "Data");
  ge::Operator const_op = ge::Operator("Constant_0", "Constant");
  ge::Operator data3 = ge::Operator("Data_2", "Data");
  ge::Operator variable = ge::Operator("Variable_0", "Variable");
  ge::Operator concat_v2 = ge::Operator("ConcatV2_0", "ConcatV2");
  ge::Operator identity_n = ge::Operator("IdentityN_0", "IdentityN");
  ge::Operator matmul_v2 = ge::Operator("MatmulV2_0", "MatmulV2");

  data1.InputRegister("x");
  data1.OutputRegister("y");
  data2.InputRegister("x");
  data2.OutputRegister("y");
  data3.InputRegister("x");
  data3.OutputRegister("y");
  const_op.OutputRegister("y");
  variable.InputRegister("x");
  variable.OutputRegister("y");
  concat_v2.DynamicInputRegister("x", 2);
  concat_v2.InputRegister("concat_dim");
  concat_v2.OutputRegister("y");
  identity_n.DynamicInputRegister("x", 3);
  identity_n.DynamicOutputRegister("y", 3);
  matmul_v2.InputRegister("x1");
  matmul_v2.InputRegister("x2");
  matmul_v2.OptionalInputRegister("bias");
  matmul_v2.OptionalInputRegister("offset_w");
  matmul_v2.OutputRegister("y");
  concat_v2.SetInput(0U, data1, 0U);
  concat_v2.SetInput(1U, data2, 0U);
  concat_v2.SetInput(2U, const_op, 0U);
  identity_n.SetInput(0U, concat_v2, 0U);
  identity_n.SetInput(1U, variable, 0U);
  identity_n.SetInput(2U, data3, 0U);
  matmul_v2.SetInput(0U, identity_n, 0U);
  matmul_v2.SetInput(1U, identity_n, 1U);
  matmul_v2.SetInput(2U, identity_n, 2U);
  matmul_v2.AddControlInput(concat_v2);
  concat_v2.AddControlInput(variable);
  std::vector<Operator> ops{data1, const_op, data2, variable, concat_v2, data3, identity_n, matmul_v2};
  Graph graph("stable_sort_graph");
  EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, ops), SUCCESS);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetName(), "stable_sort_graph");
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 8);
  std::vector<ExpectNodeInfo> expect_node_info;
  std::map<int32_t, std::pair<std::string, int32_t>> input_node_name;
  std::map<int32_t, std::vector<std::pair<std::string, int32_t>>> output_node_name;
  std::vector<std::string> control_input_node_name;
  std::vector<std::string> control_output_node_name;
  std::vector<std::pair<std::string, int32_t>> temp_vector = {{"ConcatV2_0", 0}};
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("ConcatV2_0", 2));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Constant_0", "Constant",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 0, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("ConcatV2_0", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_1", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("IdentityN_0", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  control_output_node_name.emplace_back("ConcatV2_0");
  expect_node_info.emplace_back(ExpectNodeInfo("Variable_0", "Variable",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("Data_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("Data_1", 0)));
  input_node_name.emplace(std::make_pair(2, std::make_pair("Constant_0", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("IdentityN_0", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  control_input_node_name.emplace_back("Variable_0");
  control_output_node_name.clear();
  control_output_node_name.emplace_back("MatmulV2_0");
  expect_node_info.emplace_back(ExpectNodeInfo("ConcatV2_0", "ConcatV2",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 3, 1));

  input_node_name.clear();
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("IdentityN_0", 2));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  control_input_node_name.clear();
  control_output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("Data_2", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("ConcatV2_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("Variable_0", 0)));
  input_node_name.emplace(std::make_pair(2, std::make_pair("Data_2", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("MatmulV2_0", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  temp_vector.clear();
  temp_vector.emplace_back(std::make_pair("MatmulV2_0", 1));
  output_node_name.emplace(std::make_pair(1, temp_vector));
  temp_vector.clear();
  temp_vector.emplace_back(std::make_pair("MatmulV2_0", 2));
  output_node_name.emplace(std::make_pair(2, temp_vector));
  control_input_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("IdentityN_0", "IdentityN",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 3, 3));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("IdentityN_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("IdentityN_0", 1)));
  input_node_name.emplace(std::make_pair(2, std::make_pair("IdentityN_0", 2)));
  output_node_name.clear();
  control_input_node_name.emplace_back("ConcatV2_0");
  expect_node_info.emplace_back(ExpectNodeInfo("MatmulV2_0", "MatmulV2",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 4, 1));
  CheckNodeResult(compute_graph, expect_node_info);
  EXPECT_EQ(compute_graph->GetInputSize(), 3);
}

/*
  Data Data Data 
   |     |    |                               
    \    |   /            branch0:                          branch1:                 branch1_0:          branch1_1:         
        If                    DATA     DATA                 DATA   DATA                   DATA             DATA
         |                      \       /                     |    /                        |                |
        Relu                      Add                           if                        Relu             Relu                                                  
*/
TEST_F(UtestGraph, TestGenerateGraphWithSubGraph) {
  ge::Operator data_0 = ge::Operator("Data_0", "Data");
  ge::Operator data_1 = ge::Operator("Data_1", "Data");
  ge::Operator data_2 = ge::Operator("Data_2", "Data");
  ge::Operator if_op = ge::Operator("If_0", "If");
  ge::Operator relu_0 = ge::Operator("Relu_0", "Relu");
  data_0.InputRegister("x");
  data_0.OutputRegister("y");
  data_1.InputRegister("x");
  data_1.OutputRegister("y");
  data_2.InputRegister("x");
  data_2.OutputRegister("y");

  if_op.InputRegister("cond");
  if_op.DynamicInputRegister("input", 2);
  if_op.DynamicOutputRegister("output", 1);
  if_op.SubgraphRegister("then_branch", false);
  if_op.SubgraphRegister("else_branch", false);
  if_op.SubgraphCountRegister("then_branch", 1);
  if_op.SubgraphCountRegister("else_branch", 1);
  if_op.SetSubgraphBuilder("then_branch", 0, [] ()->Graph {
    ge::Operator then_branch_data_0 = ge::Operator("then_branch_data_0", "Data");
    ge::Operator then_branch_data_1 = ge::Operator("then_branch_data_1", "Data");
    ge::Operator add_0 = ge::Operator("Add_0", "Add");
    then_branch_data_0.InputRegister("x");
    then_branch_data_0.OutputRegister("y");
    then_branch_data_1.InputRegister("x");
    then_branch_data_1.OutputRegister("y");
    add_0.InputRegister("x1");
    add_0.InputRegister("x2");
    add_0.OutputRegister("y");
    add_0.SetInput(0U, then_branch_data_0, 0U);
    add_0.SetInput(1U, then_branch_data_1, 0U);
    std::vector<Operator> then_branch_ops{then_branch_data_0, then_branch_data_1, add_0};
    Graph graph("if_op_then_branch");
    EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, then_branch_ops), SUCCESS);
    return graph;
  });
  if_op.SetSubgraphBuilder("else_branch", 0, [] ()->Graph {
    ge::Operator else_branch_data_0 = ge::Operator("else_branch_data_0", "Data");
    ge::Operator else_branch_data_1 = ge::Operator("else_branch_data_1", "Data");
    ge::Operator if_op_1 = ge::Operator("else_branch_if", "If");
    else_branch_data_0.InputRegister("x");
    else_branch_data_0.OutputRegister("y");
    else_branch_data_1.InputRegister("x");
    else_branch_data_1.OutputRegister("y");
    if_op_1.InputRegister("cond");
    if_op_1.DynamicInputRegister("input", 1);
    if_op_1.DynamicOutputRegister("output", 1);
    if_op_1.SubgraphRegister("then_branch", false);
    if_op_1.SubgraphRegister("else_branch", false);
    if_op_1.SubgraphCountRegister("then_branch", 1);
    if_op_1.SubgraphCountRegister("else_branch", 1);
    if_op_1.SetSubgraphBuilder("then_branch", 0, [] ()->Graph {
      ge::Operator if_1_then_branch_data_0 = ge::Operator("if_1_then_branch_data_0", "Data");
      ge::Operator if_1_then_branch_relu = ge::Operator("if_1_then_branch_relu", "Relu");
      if_1_then_branch_data_0.InputRegister("x");
      if_1_then_branch_data_0.OutputRegister("y");
      if_1_then_branch_relu.InputRegister("x");
      if_1_then_branch_relu.OutputRegister("y");
      if_1_then_branch_relu.SetInput(0U, if_1_then_branch_data_0, 0U);
      std::vector<Operator> if_1_then_branch_ops{if_1_then_branch_data_0, if_1_then_branch_relu};
      Graph graph("if_1_then_branch");
      EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, if_1_then_branch_ops), SUCCESS);
      return graph;
    });
    if_op_1.SetSubgraphBuilder("else_branch", 0, [] ()->Graph {
      ge::Operator if_1_else_branch_data_0 = ge::Operator("if_1_else_branch_data_0", "Data");
      ge::Operator if_1_else_branch_relu = ge::Operator("if_1_else_branch_relu", "Relu");
      if_1_else_branch_data_0.InputRegister("x");
      if_1_else_branch_data_0.OutputRegister("y");
      if_1_else_branch_relu.InputRegister("x");
      if_1_else_branch_relu.OutputRegister("y");
      if_1_else_branch_relu.SetInput(0U, if_1_else_branch_data_0, 0U);
      std::vector<Operator> if_1_else_branch_ops{if_1_else_branch_data_0, if_1_else_branch_relu};
      Graph graph("if_1_else_branch");
      EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, if_1_else_branch_ops), SUCCESS);
      return graph;
    });
    if_op_1.SetInput(0U, else_branch_data_0, 0U);
    if_op_1.SetInput(1U, else_branch_data_1, 0U);
    std::vector<Operator> else_branch_ops{else_branch_data_0, else_branch_data_1, if_op_1};
    Graph graph("if_op_else_branch");
    EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, else_branch_ops), SUCCESS);
    return graph;
  });
  relu_0.InputRegister("x");
  relu_0.OutputRegister("y");
  if_op.SetInput(0U, data_0, 0U);
  if_op.SetInput(1U, data_1, 0U);
  if_op.SetInput(2U, data_2, 0U);
  relu_0.SetInput(0U, if_op, 0U);

  std::vector<Operator> ops{data_0, data_1, data_2, if_op, relu_0};
  Graph graph("stable_sort_graph_with_subgraph");
  EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, ops), SUCCESS);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  std::map<std::string, std::vector<ExpectNodeInfo>> graph_expect_info;
  // root_graph
  std::vector<ExpectNodeInfo> expect_node_info;
  std::map<int32_t, std::pair<std::string, int32_t>> input_node_name;
  std::map<int32_t, std::vector<std::pair<std::string, int32_t>>> output_node_name;
  std::vector<std::string> control_input_node_name;
  std::vector<std::string> control_output_node_name;
  std::vector<std::pair<std::string, int32_t>> temp_vector = {{"If_0", 0}};
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("If_0", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_1", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("If_0", 2));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_2", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("Data_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("Data_1", 0)));
  input_node_name.emplace(std::make_pair(2, std::make_pair("Data_2", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Relu_0", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("If_0", "If",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 3, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("If_0", 0)));
  output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("Relu_0", "Relu",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));
  graph_expect_info.emplace("stable_sort_graph_with_subgraph", expect_node_info);

  // if_0_then_branch
  expect_node_info.clear();
  input_node_name.clear();
  temp_vector.clear();
  temp_vector.emplace_back(std::make_pair("Add_0", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("then_branch_data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Add_0", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("then_branch_data_1", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("then_branch_data_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("then_branch_data_1", 0)));
  output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("Add_0", "Add",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 2, 1));
  graph_expect_info.emplace("if_op_then_branch", expect_node_info);

  // if_0_else_branch
  expect_node_info.clear();
  input_node_name.clear();
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("else_branch_if", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("else_branch_data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("else_branch_if", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("else_branch_data_1", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("else_branch_data_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("else_branch_data_1", 0)));
  output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("else_branch_if", "If",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 2, 1));
  graph_expect_info.emplace("if_op_else_branch", expect_node_info);

  // if_1_then_branch
  expect_node_info.clear();
  input_node_name.clear();
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("if_1_then_branch_relu", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("if_1_then_branch_data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("if_1_then_branch_data_0", 0)));
  output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("if_1_then_branch_relu", "Relu",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));
  graph_expect_info.emplace("if_1_then_branch", expect_node_info);
  // if_1_else_branch
  expect_node_info.clear();
  input_node_name.clear();
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("if_1_else_branch_relu", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("if_1_else_branch_data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.emplace(std::make_pair(0, std::make_pair("if_1_else_branch_data_0", 0)));
  output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("if_1_else_branch_relu", "Relu",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));
  graph_expect_info.emplace("if_1_else_branch", expect_node_info);

  EXPECT_EQ(compute_graph->GetName(), "stable_sort_graph_with_subgraph");
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 5);
  EXPECT_EQ(compute_graph->GetInputSize(), 3);
  CheckNodeResult(compute_graph, graph_expect_info["stable_sort_graph_with_subgraph"]);
  const auto subgraphs = compute_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 4);
  for (const auto &subgraph : subgraphs) {
    const auto iter = graph_expect_info.find(subgraph->GetName());
    ASSERT_NE(iter, graph_expect_info.end());
    CheckNodeResult(subgraph, iter->second);
  }
}

/*
  Data Data    Const
   |     |       |
  Relu  Relu    Relu
   |     |       |  \
  Cast  Cast -- Add  Cast
   \    /        
    Add                               
*/
TEST_F(UtestGraph, TestGenerateGraphWithOutputMultiRef) {
  ge::Operator data_0 = ge::Operator("Data_0", "Data");
  ge::Operator data_1 = ge::Operator("Data_1","Data");
  ge::Operator const_op = ge::Operator("Constant_0", "Constant");
  ge::Operator relu_0 = ge::Operator("Relu_0", "Relu");
  ge::Operator relu_1 = ge::Operator("Relu_1", "Relu");
  ge::Operator relu_2 = ge::Operator("Relu_2", "Relu");
  ge::Operator cast_0 = ge::Operator("Cast_0", "Cast");
  ge::Operator cast_1 = ge::Operator("Cast_1", "Cast");
  ge::Operator cast_2 = ge::Operator("Cast_2", "Cast");
  ge::Operator add_0 = ge::Operator("Add_0", "Add");
  ge::Operator add_1 = ge::Operator("Add_1", "Add");

  data_0.InputRegister("x");
  data_0.OutputRegister("y");
  data_1.InputRegister("x");
  data_1.OutputRegister("y");
  const_op.OutputRegister("y");
  relu_0.InputRegister("x");
  relu_0.OutputRegister("y");
  relu_1.InputRegister("x");
  relu_1.OutputRegister("y");
  relu_2.InputRegister("x");
  relu_2.OutputRegister("y");
  cast_0.InputRegister("x");
  cast_0.OutputRegister("y");
  cast_1.InputRegister("x");
  cast_1.OutputRegister("y");
  cast_2.InputRegister("x");
  cast_2.OutputRegister("y");
  add_0.InputRegister("x1");
  add_0.InputRegister("x2");
  add_0.OutputRegister("y");
  add_1.InputRegister("x1");
  add_1.InputRegister("x2");
  add_1.OutputRegister("y");

  relu_0.SetInput(0U, data_0, 0U);
  relu_1.SetInput(0U, data_1, 0U);
  relu_2.SetInput(0U, const_op, 0U);
  cast_0.SetInput(0U, relu_0, 0U);
  cast_1.SetInput(0U, relu_1, 0U);
  cast_2.SetInput(0U, relu_2, 0U);
  add_0.SetInput(0U, cast_1, 0U);
  add_0.SetInput(1U, relu_2, 0U);
  add_1.SetInput(0U, cast_0, 0U);
  add_1.SetInput(1U, cast_1, 0U);

  std::vector<Operator> ops{data_0, const_op, data_1, relu_0, relu_1, relu_2, cast_0, cast_1, cast_2, add_0, add_1};
  Graph graph("stable_sort_graph_multi_output_ref");
  EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(graph, ops), SUCCESS);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetName(), "stable_sort_graph_multi_output_ref");
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 11);
  std::vector<ExpectNodeInfo> expect_node_info;
  std::map<int32_t, std::pair<std::string, int32_t>> input_node_name;
  std::map<int32_t, std::vector<std::pair<std::string, int32_t>>> output_node_name;
  std::vector<std::string> control_input_node_name;
  std::vector<std::string> control_output_node_name;
  std::vector<std::pair<std::string, int32_t>> temp_vector = {{"Relu_0", 0}};
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_0", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Relu_2", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Constant_0", "Constant",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 0, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Relu_1", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Data_1", "Data",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Cast_0", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  input_node_name.emplace(std::make_pair(0, std::make_pair("Data_0", 0)));
  expect_node_info.emplace_back(ExpectNodeInfo("Relu_0", "Relu",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Data_1", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Cast_1", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Relu_1", "Relu",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Constant_0", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Cast_2", 0));
  temp_vector.emplace_back(std::make_pair("Add_0", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Relu_2", "Relu",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Relu_0", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Add_1", 0));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Cast_0", "Cast",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Relu_1", 0)));
  temp_vector.clear();
  output_node_name.clear();
  temp_vector.emplace_back(std::make_pair("Add_0", 0));
  temp_vector.emplace_back(std::make_pair("Add_1", 1));
  output_node_name.emplace(std::make_pair(0, temp_vector));
  expect_node_info.emplace_back(ExpectNodeInfo("Cast_1", "Cast",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Relu_2", 0)));
  output_node_name.clear();
  expect_node_info.emplace_back(ExpectNodeInfo("Cast_2", "Cast",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 1, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Cast_1", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("Relu_2", 0)));
  expect_node_info.emplace_back(ExpectNodeInfo("Add_0", "Add",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 2, 1));

  input_node_name.clear();
  input_node_name.emplace(std::make_pair(0, std::make_pair("Cast_0", 0)));
  input_node_name.emplace(std::make_pair(1, std::make_pair("Cast_1", 0)));
  expect_node_info.emplace_back(ExpectNodeInfo("Add_1", "Add",
      input_node_name, output_node_name, control_input_node_name, control_output_node_name, 2, 1));
  CheckNodeResult(compute_graph, expect_node_info);
  EXPECT_EQ(compute_graph->GetInputSize(), 2);
}

TEST_F(UtestGraph, TestSameNameNode_fail) {
  std::string op_type(__FUNCTION__);
  std::string op_name("the_dummy");
  OperatorFactoryImpl::RegisterOperatorCreator(op_type, [op_type](const std::string &name) -> Operator {
    auto op_desc = std::make_shared<OpDesc>(name, op_type);
    op_desc->AddOutputDesc("output", {});
    return OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  });

  auto node_0 = Operator(op_name, op_type);
  auto node_1 = Operator(op_name, op_type);
  std::vector<Operator> ops_0 = { node_0, node_1 };
  EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperator("graph_with_same_name_node", ops_0), nullptr);

  auto node_2 = Operator(op_name, op_type);
  node_2.SubgraphRegister("sub_graph", false);
  node_2.SubgraphCountRegister("sub_graph", 1);
  node_2.SetSubgraphBuilder("sub_graph", 0, [op_name, op_type]() {
    ut::GraphBuilder builder = ut::GraphBuilder("sub_graph_with_same_name_node");
    builder.AddNode(op_name, op_type, 0, 1);
    builder.AddNode(op_name, op_type, 0, 1);
    return GraphUtilsEx::CreateGraphFromComputeGraph(builder.GetGraph());
  });
  std::vector<Operator> ops_1 = { node_2 };
  EXPECT_EQ(GraphUtilsEx::CreateGraphFromOperator("graph_with_same_name_node_in_subgraph", ops_1), nullptr);
}
// extern "C" wrapper functions to avoid C++ name mangling
extern "C" {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_Graph_SetValid(void *graph_ptr) {
  if (graph_ptr == nullptr) {
    return GRAPH_FAILED;
  }
  auto *graph = static_cast<Graph *>(graph_ptr);
  return graph->SetValid();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_Graph_SetAttr_AttrValue(void *graph_ptr,
                                                                                          const char *name,
                                                                                          const void *attr_value) {
  if (graph_ptr == nullptr || name == nullptr || attr_value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *graph = static_cast<Graph *>(graph_ptr);
  auto *value = static_cast<const ge::AttrValue *>(attr_value);
  return graph->SetAttr(ge::AscendString(name), *value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus aclCom_Graph_GetAttr_AttrValue(void *graph_ptr,
                                                                                          const char *name,
                                                                                          void *attr_value) {
  if (graph_ptr == nullptr || name == nullptr || attr_value == nullptr) {
    return GRAPH_FAILED;
  }
  auto *graph = static_cast<Graph *>(graph_ptr);
  auto *value = static_cast<ge::AttrValue *>(attr_value);
  return graph->GetAttr(ge::AscendString(name), *value);
}
}
TEST_F(UtestGraph, TestGraphSetAttrAndGetAttr_AttrValue) {
  // 创建测试图
  Graph graph("test_graph");
  auto data_op = Operator("Data", "Data");
  auto const_op = Operator("Const", "Const");
  std::vector<Operator> inputs = {data_op, const_op};
  graph.SetInputs(inputs);

  // 测试AttrValue类型的SetAttr和GetAttr
  AttrValue attr_value;
  attr_value.SetAttrValue(static_cast<int64_t>(12345));

  // 测试成功情况
  EXPECT_EQ(graph.SetAttr("test_attr", attr_value), GRAPH_SUCCESS);
  AttrValue get_attr_value;
  EXPECT_EQ(graph.GetAttr("test_attr", get_attr_value), GRAPH_SUCCESS);
  int64_t int_value = 0;
  EXPECT_EQ(get_attr_value.GetAttrValue(int_value), GRAPH_SUCCESS);
  EXPECT_EQ(int_value, 12345);

  // 测试extern "C"接口
  EXPECT_EQ(aclCom_Graph_SetAttr_AttrValue(&graph, "test_attr_c", &attr_value), GRAPH_SUCCESS);
  EXPECT_EQ(aclCom_Graph_GetAttr_AttrValue(&graph, "test_attr_c", &get_attr_value), GRAPH_SUCCESS);
  int_value = 0;
  EXPECT_EQ(get_attr_value.GetAttrValue(int_value), GRAPH_SUCCESS);
  EXPECT_EQ(int_value, 12345);

  // 测试nullptr参数
  EXPECT_EQ(aclCom_Graph_SetAttr_AttrValue(nullptr, "test_attr", &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_Graph_SetAttr_AttrValue(&graph, nullptr, &attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_Graph_SetAttr_AttrValue(&graph, "test_attr", nullptr), GRAPH_FAILED);
  EXPECT_EQ(aclCom_Graph_GetAttr_AttrValue(nullptr, "test_attr", &get_attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_Graph_GetAttr_AttrValue(&graph, nullptr, &get_attr_value), GRAPH_FAILED);
  EXPECT_EQ(aclCom_Graph_GetAttr_AttrValue(&graph, "test_attr", nullptr), GRAPH_FAILED);
}

TEST_F(UtestGraph, TestGraphSetAttrAndGetAttr_AttrValue_ComplexTypes) {
  // 创建测试图
  Graph graph("test_graph");
  graph.SetValid();

  // 测试复杂类型的AttrValue
  AttrValue complex_attr;
  std::vector<int64_t> vec_value = {1, 2, 3, 4, 5};
  complex_attr.SetAttrValue(vec_value);

  EXPECT_EQ(graph.SetAttr("complex_attr", complex_attr), GRAPH_SUCCESS);
  AttrValue get_complex_attr;
  EXPECT_EQ(graph.GetAttr("complex_attr", get_complex_attr), GRAPH_SUCCESS);
  std::vector<int64_t> get_vec_value;
  EXPECT_EQ(get_complex_attr.GetAttrValue(get_vec_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_vec_value, vec_value);

  // 测试extern "C"接口
  EXPECT_EQ(aclCom_Graph_SetAttr_AttrValue(&graph, "complex_attr_c", &complex_attr), GRAPH_SUCCESS);
  EXPECT_EQ(aclCom_Graph_GetAttr_AttrValue(&graph, "complex_attr_c", &get_complex_attr), GRAPH_SUCCESS);
  get_vec_value.clear();
  EXPECT_EQ(get_complex_attr.GetAttrValue(get_vec_value), GRAPH_SUCCESS);
  EXPECT_EQ(get_vec_value, vec_value);
}

TEST_F(UtestGraph, TestGraphSetAttrAndGetAttr_AttrValue_InvalidCases) {
  // 创建测试图
  Graph graph("test_graph");
  EXPECT_EQ(graph.SetValid(), GRAPH_SUCCESS);
  EXPECT_EQ(aclCom_Graph_SetValid(&graph), GRAPH_SUCCESS);

  // 测试获取不存在的属性
  AttrValue attr_value;
  EXPECT_NE(graph.GetAttr("non_existent_attr", attr_value), GRAPH_SUCCESS);

  // 测试空图
  Graph empty_graph;
  AttrValue test_attr;
  test_attr.SetAttrValue(static_cast<int64_t>(12345));
  EXPECT_NE(empty_graph.SetAttr("test_attr", test_attr), GRAPH_SUCCESS);
  EXPECT_NE(empty_graph.GetAttr("test_attr", attr_value), GRAPH_SUCCESS);
}

TEST_F(UtestGraph, TestDumpOnnxGraphToFile) {
  std::string ascend_work_path = "./test_ge_graph_path";
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);

  // 创建测试图
  auto compute_graph = BuildComputeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(graph.IsValid(), true);

  std::string suffix = "test_onnx";
  // 测试dump onnx
  EXPECT_EQ(graph.DumpToFile(Graph::DumpFormat::kOnnx, suffix.c_str()), GRAPH_SUCCESS);

  // test existed dir
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("GeTestGraph1");
  onnx::ModelProto model_proto;
  ASSERT_EQ(model_proto.ByteSize(), 0);
  // static thing, so follow DumpGEGraphUserGraphNameNull_AscendWorkPathNotNull this case path
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = GetSpecificFilePath(ge::RealPath(dump_file_path.str().c_str()), suffix.c_str());
  bool state = GraphUtils::ReadProtoFromTextFile(dump_graph_path.c_str(), &model_proto);
  ASSERT_EQ(state, true);
  ASSERT_NE(model_proto.ByteSize(), 0);
  EXPECT_STREQ(model_proto.graph().name().c_str(), "graph");

  system(("rm -rf " + ascend_work_path).c_str());
  unsetenv("DUMP_GRAPH_PATH");
}

TEST_F(UtestGraph, TestDumpProtoGraphToFile) {
  std::string ascend_work_path = "./test_ge_graph_path";
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);

  // 创建测试图
  auto compute_graph = BuildComputeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(graph.IsValid(), true);

  std::string suffix = "test_txt";
  // 测试dump onnx
  EXPECT_EQ(graph.DumpToFile(Graph::DumpFormat::kTxt, suffix.c_str()), GRAPH_SUCCESS);

  // test existed dir
  onnx::ModelProto model_proto;
  ASSERT_EQ(model_proto.ByteSize(), 0);
  // static thing, so follow DumpGEGraphUserGraphNameNull_AscendWorkPathNotNull this case path
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = GetSpecificFilePath(ge::RealPath(dump_file_path.str().c_str()), suffix.c_str());

  ComputeGraphPtr com_graph2 = std::make_shared<ComputeGraph>("GeTestGraph2");
  bool state = GraphUtils::LoadGEGraph(dump_graph_path.c_str(), *com_graph2);
  EXPECT_EQ(state, true);
  EXPECT_EQ(com_graph2->GetDirectNodesSize(), 3);

  system(("rm -rf " + ascend_work_path).c_str());
  unsetenv("DUMP_GRAPH_PATH");
}

TEST_F(UtestGraph, TestDumpProtoGraphToOstream) {
  // 创建测试图
  auto compute_graph = BuildComputeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(graph.IsValid(), true);

  // 测试dump txt
  std::ostringstream stream;
  EXPECT_EQ(graph.Dump(Graph::DumpFormat::kTxt, stream), GRAPH_SUCCESS);

  ge::proto::ModelDef txt_model_proto;
  google::protobuf::TextFormat::ParseFromString(stream.str(), &txt_model_proto);

  Model model;
  EXPECT_EQ(model.Load(txt_model_proto), SUCCESS);
  EXPECT_EQ(model.GetGraph()->GetDirectNodesSize(), 3);
}

TEST_F(UtestGraph, TestDumpOnnxGraphToOstream) {
  // 创建测试图
  auto compute_graph = BuildComputeGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(graph.IsValid(), true);

  // 测试dump txt
  std::ostringstream stream;
  EXPECT_EQ(graph.Dump(Graph::DumpFormat::kOnnx, stream), GRAPH_SUCCESS);

  onnx::ModelProto onnx_model_proto;
  google::protobuf::TextFormat::ParseFromString(stream.str(), &onnx_model_proto);
  EXPECT_STREQ(onnx_model_proto.graph().name().c_str(), "graph");
}
REG_OP(subTest)
    .INPUT(inx, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .GRAPH(subgraph)
    .OP_END_FACTORY_REG(subTest);

REG_OP(dataOp)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Int, 0)
    .OP_END_FACTORY_REG(dataOp);

TEST_F(UtestGraph, TestGraphFindNodeByName_failure) {
  Graph graph_invalid("graph_invalid");
  AscendString empty_graph_node_name = "blablabla";
  ASSERT_EQ(nullptr, graph_invalid.FindNodeByName(empty_graph_node_name));

  auto op = op::dataOp("dataOp");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs = {op_input1};
  AscendString name = "graph";
  GraphPtr graph = Graph::ConstructFromInputs(inputs, name);
  auto node = graph->AddNodeByOp(op);

  AscendString wrong_name("wrong_name");
  ASSERT_EQ(nullptr, graph->FindNodeByName(wrong_name));
}

TEST_F(UtestGraph, TestGraphFindNodeByName_success) {
  auto op = op::dataOp("dataOp");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs = {op_input1};
  AscendString name = "graph";
  GraphPtr graph = Graph::ConstructFromInputs(inputs, name);
  auto node = graph->AddNodeByOp(op);

  AscendString find_name;
  op.GetName(find_name);
  auto get_node = graph->FindNodeByName(find_name);
  AscendString exp_name("dataOp");
  ASSERT_TRUE(find_name == exp_name);
}

TEST_F(UtestGraph, TestGraphGetParentGraph_failure) {
  Graph graph_invalid("graph_invalid");
  AscendString empty_graph_node_name = "blablabla";
  ASSERT_EQ(nullptr, graph_invalid.GetParentGraph());
}

TEST_F(UtestGraph, TestGraphGetParentGraph_success) {
  auto op = op::subTest("subTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs = {op_input1};
  AscendString name = "graph";
  GraphPtr graph = Graph::ConstructFromInputs(inputs, name);
  auto gnode = graph->AddNodeByOp(op);

  name = "subgraph1";
  Operator op_input2 = op::dataOp("op_input2");
  inputs = {op_input2};
  auto subgraph = Graph::ConstructFromInputs(inputs, name);
  ASSERT_EQ(GRAPH_SUCCESS, gnode.SetSubgraph("subgraph", *subgraph.get()));

  auto subgraph_parent = subgraph->GetParentGraph();
  AscendString ret_parent_graph_name;
  AscendString exp_parent_graph_name("graph");
  subgraph_parent->GetName(ret_parent_graph_name);
  ASSERT_TRUE(exp_parent_graph_name == ret_parent_graph_name);
}

TEST_F(UtestGraph, TestGraphGetParentNode_failure) {
  Graph graph_invalid("graph_invalid");
  AscendString empty_graph_node_name = "blablabla";
  ASSERT_EQ(nullptr, graph_invalid.GetParentNode());
}

TEST_F(UtestGraph, TestGraphGetParentNode_success) {
  auto op = op::subTest("subTest");
  Operator op_input1 = op::dataOp("op_input1");
  std::vector<Operator> inputs = {op_input1};
  AscendString name = "graph";
  GraphPtr graph = Graph::ConstructFromInputs(inputs, name);
  auto gnode = graph->AddNodeByOp(op);

  name = "subgraph1";
  Operator op_input2 = op::dataOp("op_input2");
  inputs = {op_input2};
  auto subgraph = Graph::ConstructFromInputs(inputs, name);
  ASSERT_EQ(GRAPH_SUCCESS, gnode.SetSubgraph("subgraph", *subgraph.get()));

  auto subgraph_parent_node = subgraph->GetParentNode();
  AscendString ret_parent_node_name;
  AscendString exp_parent_node_name("subTest");
  subgraph_parent_node->GetName(ret_parent_node_name);
  ASSERT_TRUE(ret_parent_node_name == exp_parent_node_name);
}
