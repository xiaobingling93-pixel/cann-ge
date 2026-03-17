/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "common/util/mem_utils.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "proto/dflow.pb.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "graph/ge_global_options.h"
#include "framework/common/ge_types.h"

using namespace testing;
namespace ge {
class DataFlowGraphTest : public Test {
 protected:
  static void SetUpTestSuite() {
    std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
    (void) system(cmd.c_str());
    std::ofstream cmakefile("./temp/CMakeLists.txt");
    {
      cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
      // Prevent cmake from testing the toolchain
      cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
      cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
      cmakefile << "project(test)\n";
      cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
      cmakefile << "execute_process(\n";
      cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
      cmakefile << ")\n";
      cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
      cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
    }
  }
  static void TearDownTestSuite() {
    std::string cmd = "rm -rf temp";
    (void) system(cmd.c_str());
  }
  void SetUp() override {
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
  }
  void TearDown() override {
  }
};

TEST_F(DataFlowGraphTest, Initialize_SUCCESS) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, 1024);
  (void)AttrUtils::SetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, 30 * 1000);
  (void)AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT, true);
  auto pp0 = dataflow::ProcessPoint();
  std::string pp_name(128, 'x');
  pp0.set_name(pp_name);
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 2},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  AttrUtils::SetBool(node0->GetOpDesc(), dflow::ATTR_NAME_BALANCE_GATHER, true);

  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                    {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                     {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  pp1.set_compile_cfg_file(pp1_config_file);
  pp1.add_graphs("pp1");
  auto pp1_input0 = pp1.add_in_edges();
  pp1_input0->set_node_name("node1");
  pp1_input0->set_index(0);
  auto pp1_input1 = pp1.add_in_edges();
  pp1_input1->set_node_name("node1");
  pp1_input1->set_index(1);
  auto pp1_output0 = pp1.add_out_edges();
  pp1_output0->set_node_name("node1");
  pp1_output0->set_index(0);
  std::string pp1_str;
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp1_attr{pp1_str};
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetListStr(node1->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp1_attr);
  AttrUtils::SetBool(node1->GetOpDesc(), dflow::ATTR_NAME_BALANCE_SCATTER, true);

  DEF_GRAPH(sub_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto sub_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph->SetParentNode(node1);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph);

  dataflow::ProcessPoint pp2;
  pp2.set_name("func_invoke_graph_pp");
  pp2.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp2_config_file = "./pp2_config.json";
  {
    nlohmann::json pp2_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp2_config_file);
    json_file << pp2_cfg_json << std::endl;
  }
  pp2.set_compile_cfg_file(pp2_config_file);

  pp1.set_name("invoked_graph_pp");
  pp1.set_graphs(0, "invoked_graph_pp");
  auto invoke_pps = pp2.mutable_invoke_pps();
  (*invoke_pps)["invoked_graph_pp"] = pp1;
  std::string pp2_str;
  pp2.SerializeToString(&pp2_str);
  std::vector<std::string> pp2_attr{pp2_str};
  auto node2 = root_graph->FindNode("node2");
  AttrUtils::SetListStr(node2->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp2_attr);
  auto sub_graph1 = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  (void)AttrUtils::SetStr(sub_graph1, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph1->SetParentNode(node2);
  sub_graph1->SetParentGraph(root_graph);
  sub_graph1->SetName("invoked_graph_pp");
  root_graph->AddSubgraph("invoked_graph_pp", sub_graph1);

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  remove(pp0_config_file.c_str());
  remove(pp1_config_file.c_str());
  remove(pp2_config_file.c_str());
}

TEST_F(DataFlowGraphTest, Initialize_FAILED) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data0", data0)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 1},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), FAILED);
  remove(pp0_config_file.c_str());
}

TEST_F(DataFlowGraphTest, Initialize_failed_as_pp_name_over_length) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data0", data0)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  std::string pp_name(129, 'x');
  pp0.set_name(pp_name);
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 1},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), PARAM_INVALID);
  remove(pp0_config_file.c_str());
}

TEST_F(DataFlowGraphTest, Initialize_failed_as_balance_only_part) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node0", node0)->EDGE(2, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 3},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);

  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp1_config_file = "./pp1_config.json";
  {
    nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                    {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                     {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
    std::ofstream json_file(pp1_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  pp1.set_compile_cfg_file(pp1_config_file);
  pp1.add_graphs("pp1");
  auto pp1_input0 = pp1.add_in_edges();
  pp1_input0->set_node_name("node1");
  pp1_input0->set_index(0);
  auto pp1_input1 = pp1.add_in_edges();
  pp1_input1->set_node_name("node1");
  pp1_input1->set_index(1);
  auto pp1_output0 = pp1.add_out_edges();
  pp1_output0->set_node_name("node1");
  pp1_output0->set_index(0);
  std::string pp1_str;
  pp1.SerializeToString(&pp1_str);
  std::vector<std::string> pp1_attr{pp1_str};
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetListStr(node1->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp1_attr);
  AttrUtils::SetBool(node1->GetOpDesc(), dflow::ATTR_NAME_BALANCE_SCATTER, true);

  DEF_GRAPH(sub_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto sub_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  (void)AttrUtils::SetStr(sub_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph->SetParentNode(node1);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph);

  dataflow::ProcessPoint pp2;
  pp2.set_name("func_invoke_graph_pp");
  pp2.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp2_config_file = "./pp2_config.json";
  {
    nlohmann::json pp2_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp2_config_file);
    json_file << pp2_cfg_json << std::endl;
  }
  pp2.set_compile_cfg_file(pp2_config_file);

  pp1.set_name("invoked_graph_pp");
  pp1.set_graphs(0, "invoked_graph_pp");
  auto invoke_pps = pp2.mutable_invoke_pps();
  (*invoke_pps)["invoked_graph_pp"] = pp1;
  std::string pp2_str;
  pp2.SerializeToString(&pp2_str);
  std::vector<std::string> pp2_attr{pp2_str};
  auto node2 = root_graph->FindNode("node2");
  AttrUtils::SetListStr(node2->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp2_attr);
  auto sub_graph1 = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def));
  (void)AttrUtils::SetStr(sub_graph1, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph1->SetParentNode(node2);
  sub_graph1->SetParentGraph(root_graph);
  sub_graph1->SetName("invoked_graph_pp");
  root_graph->AddSubgraph("invoked_graph_pp", sub_graph1);

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_NE(data_flow_graph.Initialize(), SUCCESS);
  remove(pp0_config_file.c_str());
  remove(pp1_config_file.c_str());
  remove(pp2_config_file.c_str());
}

TEST_F(DataFlowGraphTest, AddLoadedModel_repeat) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data0", data0)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  DataFlowGraph data_flow_graph(root_graph);
  FlowModelPtr flow_model = MakeShared<FlowModel>(root_graph);
  EXPECT_EQ(data_flow_graph.AddLoadedModel("FlowNode", "loaded_name", flow_model), SUCCESS);
  EXPECT_NE(data_flow_graph.AddLoadedModel("FlowNode", "loaded_name", flow_model), SUCCESS);
}

TEST_F(DataFlowGraphTest, Initialize_Exception_Catch_Without_Align) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, true);
  (void)AttrUtils::SetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, 30 * 1000);
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 2},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  AttrUtils::SetBool(node0->GetOpDesc(), dflow::ATTR_NAME_BALANCE_GATHER, true);

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), PARAM_INVALID);
  remove(pp0_config_file.c_str());
}

TEST_F(DataFlowGraphTest, Initialize_Exception_auto_add_n_mapping) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, true);
  (void)AttrUtils::SetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, 1024);
  (void)AttrUtils::SetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, 100 * 1000);
  (void)AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT, true);

  DataFlowGraph data_flow_graph(root_graph);
  EXPECT_EQ(data_flow_graph.Initialize(), SUCCESS);
  bool contains_n_mapping_node = false;
  (void)AttrUtils::GetBool(root_graph, dflow::ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE, contains_n_mapping_node);
  EXPECT_TRUE(contains_n_mapping_node);
}
}  // namespace ge
