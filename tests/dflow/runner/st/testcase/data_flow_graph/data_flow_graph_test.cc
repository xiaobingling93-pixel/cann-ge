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
#include <fstream>
#include "ge/ge_api.h"
#include "flow_graph/data_flow.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "proto/dflow.pb.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "nlohmann/json.hpp"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/model/flow_model_cache.h"
#include "dflow/compiler/data_flow_graph/function_compile.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/ge_global_options.h"
#include "framework/common/ge_types.h"
#include "init_ge.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/env_path.h"
#include "acl/acl.h"

using namespace testing;
using namespace std;
using namespace ge; 
namespace ge {
namespace {
Graph BuildSubGraph(const std::string &name) {
  DEF_GRAPH(tmp_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto compute_graph = ToComputeGraph(tmp_graph_def);
  compute_graph->SetName(name);
  return GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}

Graph BuildDataFlowGraphInvokeKeyRepeat(const std::string &name) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0,data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("node1", 2, 2).SetInput(0,data0).SetInput(1, data1);

  auto invoked_graph_pp0 = dflow::GraphPp("invoked_graph_pp0", []() {
                             return BuildSubGraph("invoked_graph_pp0");
                           }).SetCompileConfig("./pp1_config.json");
  // function pp
  auto pp0 = dflow::FunctionPp("func_pp0")
                 .SetCompileConfig("./pp0_config.json")
                 .AddInvokedClosure("invoke_graph", invoked_graph_pp0);
  node0.AddPp(pp0);

  auto invoked_graph_pp1 = dflow::GraphPp("invoked_graph_pp1", []() {
                             return BuildSubGraph("invoked_graph_pp1");
                           }).SetCompileConfig("./pp1_config.json");
  auto pp1 = dflow::FunctionPp("func_pp1")
                 .SetCompileConfig("./pp2_config.json")
                 .AddInvokedClosure("invoke_graph", invoked_graph_pp1);
  node1.AddPp(pp1);
  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0, node1};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Graph BuildDataFlowGraphWithHostUdfCallNn(const std::string &name, bool enable_exception = false) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0,data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("node1", 2, 2).SetInput(0,data0).SetInput(1, data1);
  auto node2 = dflow::FlowNode("node2", 2, 2).SetInput(0,node0, 0).SetInput(1, node1, 0);

  auto invoked_graph_pp0 = dflow::GraphPp("invoked_graph_pp0", []() {
                             return BuildSubGraph("invoked_graph_pp0");
                           }).SetCompileConfig("./pp1_config.json");
  // function pp
  auto host_udf_pp = dflow::FunctionPp("host_udf_pp")
                         .SetCompileConfig("./host_udf_config.json")
                         .AddInvokedClosure("invoke_graph", invoked_graph_pp0);
  node0.AddPp(host_udf_pp);
  node0.SetBalanceScatter();

  auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig("./pp0_config.json");
  node1.AddPp(pp0);
  node1.SetBalanceGather();

  auto host_udf_pp2 = dflow::FunctionPp("host_udf_pp2").SetCompileConfig("./host_udf_config.json");
  node2.AddPp(host_udf_pp2);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node2};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  flow_graph.SetInputsAlignAttrs(6, 30 * 1000);
  // todo use interface to set catch exception
  if (enable_exception) {
    auto compute_graph = GraphUtilsEx::GetComputeGraph(flow_graph.ToGeGraph());
    AttrUtils::SetBool(compute_graph, "_enable_exception_catch", enable_exception);
  }
  return flow_graph.ToGeGraph();
}

Graph BuildDataFlowGraphWithHostUdf(const std::string &name) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0,data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("node1", 2, 2).SetInput(0,data0).SetInput(1, data1);

  // function pp
  auto host_udf_pp = dflow::FunctionPp("host_udf_pp").SetCompileConfig("./host_udf_config.json");
  node0.AddPp(host_udf_pp);

  auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig("./pp0_config.json");
  node1.AddPp(pp0);
  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0, node1};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Graph BuildDataFlowGraphWithBuiltinUdf(const std::string &name) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0,data0).SetInput(1, data1);

  // function pp
  auto builtin_udf_pp = dflow::FunctionPp("builtin_udf_pp").SetCompileConfig("./builtin_udf_config.json");
  node0.AddPp(builtin_udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

static ComputeGraphPtr BuildAbnormalDataFlowGraph(std::string invalid_config_file) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)(AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  pp0.set_compile_cfg_file(invalid_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  AttrUtils::SetListStr(node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  return root_graph;
}

/*
  该函数构造dataflow graph 包含graph point和function point
  测试用例构图时如果不是测UDF的特有功能请优先使用 BuildDataFlowGraphWithoutUDFNodes
  UDF涉及CMAKE，集群负载高时会严重影响测试用例用例执行速度
*/
static ComputeGraphPtr BuildDataFlowGraph() {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
    CHAIN(NODE("node2", node2)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)(AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  auto op_desc = node0->GetOpDesc();
  AttrUtils::SetListStr(op_desc, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
  AttrUtils::SetInt(op_desc, "_npu_sched_model", 1);
  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name("pp1");
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  std::string pp1_config_file = "./pp1_config.json";
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
  auto op_desc1 = node1->GetOpDesc();
  AttrUtils::SetListStr(op_desc1, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp1_attr);
  AttrUtils::SetInt(op_desc1, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
  AttrUtils::SetInt(op_desc1, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
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
  AttrUtils::SetBool(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR, true);
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128);
  AttrUtils::SetStr(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO");
  sub_graph->SetParentNode(node1);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName("pp1");
  root_graph->AddSubgraph("pp1", sub_graph);

  dataflow::ProcessPoint pp2;
  pp2.set_name("func_invoke_graph_pp");
  pp2.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp2_config_file = "./pp2_config.json";
  pp2.set_compile_cfg_file(pp2_config_file);

  pp1.set_name("invoked_graph_pp");
  pp1.set_graphs(0, "invoked_graph_pp");
  auto invoke_pps = pp2.mutable_invoke_pps();
  (*invoke_pps)["invoked_graph_pp"] = pp1;
  std::string pp2_str;
  pp2.SerializeToString(&pp2_str);
  std::vector<std::string> pp2_attr{pp2_str};
  auto node2 = root_graph->FindNode("node2");
  auto op_desc2 = node2->GetOpDesc();
  AttrUtils::SetListStr(op_desc2, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp2_attr);
  AttrUtils::SetInt(op_desc2, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
  AttrUtils::SetInt(op_desc2, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
  DEF_GRAPH(invoked_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
  };
  auto invoked_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(invoked_graph_def));
  (void)AttrUtils::SetStr(invoked_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  invoked_graph->SetName("invoked_graph_pp_invoked");
  auto invoked_node = invoked_graph->FindNode("node0");
  DEF_GRAPH(sub_graph_def1) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    ADD_OUTPUT(add, 0);
  };
  auto sub_graph1 = GraphUtilsEx::GetComputeGraph(ToGeGraph(sub_graph_def1));
  (void)AttrUtils::SetStr(sub_graph1, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  sub_graph1->SetParentNode(invoked_node);
  sub_graph1->SetParentGraph(invoked_graph);
  sub_graph1->SetName("invoked_graph_pp");
  invoked_graph->AddSubgraph("invoked_graph_pp", sub_graph1);
  invoked_graph->SetParentNode(node2);
  invoked_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph("invoked_graph_pp_invoked", invoked_graph);
  root_graph->AddSubgraph("invoked_graph_pp", sub_graph1);
  (void)node2->GetOpDesc()->AddSubgraphName("invoked_graph_pp_invoked");
  (void)node2->GetOpDesc()->SetSubgraphInstanceName(0, "invoked_graph_pp_invoked");
  invoked_graph->RemoveSubgraph("invoked_graph_pp");

  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  return root_graph;
}

static void BuildSubGrpahForGraphPoint(const std::string &name, const std::string &parent_node_name,
                                                  ComputeGraphPtr root_graph, dataflow::ProcessPoint &pp) {
  pp.set_name(name);
  pp.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  pp.add_graphs(name);
  std::string pp_config_file = "./pp1_config.json";
  pp.set_compile_cfg_file(pp_config_file);
  std::string pp_str;
  pp.SerializeToString(&pp_str);
  std::vector<std::string> pp_attr{pp_str};
  auto parent_node = root_graph->FindNode(parent_node_name);
  auto op_desc = parent_node->GetOpDesc();
  AttrUtils::SetListStr(op_desc, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp_attr);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);

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
  sub_graph->SetParentNode(parent_node);
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetName(name);
  root_graph->AddSubgraph(name, sub_graph);
}

static ComputeGraphPtr BuildDataFlowGraphWithOneUDF() {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(1, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
    CHAIN(NODE("node2", node2)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  auto node2 = root_graph->FindNode("node2");
  root_graph->SetGraphOutNodesInfo({{node2, 0}});
  (void)(AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  auto pp0 = dataflow::ProcessPoint();
  pp0.set_name("pp0");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  pp0.set_compile_cfg_file(pp0_config_file);
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  auto op_desc = node0->GetOpDesc();
  AttrUtils::SetListStr(op_desc, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  AttrUtils::SetInt(op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
 
  auto pp1 = dataflow::ProcessPoint();
  BuildSubGrpahForGraphPoint("pp1", "node1", root_graph, pp1);
  auto pp1_input0 = pp1.add_in_edges();
  pp1_input0->set_node_name("node1");
  pp1_input0->set_index(0);
  auto pp1_input1 = pp1.add_in_edges();
  pp1_input1->set_node_name("node1");
  pp1_input1->set_index(1);
  auto pp1_output0 = pp1.add_out_edges();
  pp1_output0->set_node_name("node1");
  pp1_output0->set_index(0);
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128);
  AttrUtils::SetStr(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO");

  dataflow::ProcessPoint pp2;
  BuildSubGrpahForGraphPoint("pp2", "node2", root_graph, pp2);

  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  return root_graph;
}

ComputeGraphPtr BuildFlowGraphWithUdfCallFlowGraph(const std::string &flow_graph_pp_name) {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0));
  };

  DEF_GRAPH(invoked_graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node1", node1));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  dataflow::ProcessPoint pp0;
  pp0.set_name("func_invoke_flow_graph_pp");
  pp0.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp0_config_file = "./pp0_config.json";
  {
    nlohmann::json pp0_cfg_json = {{"workspace", "./temp"},
                                   {"target_bin", "libxxx.so"},
                                   {"input_num", 2},
                                   {"output_num", 1},
                                   {"cmakelist_path", "CMakeLists.txt"},
                                   {"func_list",
                                    {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
                                     {{"func_name", "func2"}, {"inputs_index", {1}}, {"outputs_index", {0}}}}}};
    std::ofstream json_file(pp0_config_file);
    json_file << pp0_cfg_json << std::endl;
  }
  pp0.set_compile_cfg_file(pp0_config_file);

  auto pp1 = dataflow::ProcessPoint();
  pp1.set_name(flow_graph_pp_name);
  pp1.set_type(dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);
  pp1.add_graphs("invoked_flow_graph_pp");
  auto invoke_pps = pp0.mutable_invoke_pps();
  (*invoke_pps)["invoke_flow_graph_key"] = pp1;
  std::string pp0_str;
  pp0.SerializeToString(&pp0_str);
  std::vector<std::string> pp0_attr{pp0_str};
  auto node0 = root_graph->FindNode("node0");
  auto op_desc0 = node0->GetOpDesc();
  AttrUtils::SetListStr(op_desc0, dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp0_attr);
  AttrUtils::SetInt(op_desc0, ATTR_NAME_ESCHED_EVENT_PRIORITY, 0);
  AttrUtils::SetInt(op_desc0, "_npu_sched_model", 0);
  auto invoked_graph = GraphUtilsEx::GetComputeGraph(ToGeGraph(invoked_graph_def));
  (void)AttrUtils::SetStr(invoked_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  (void)AttrUtils::SetBool(invoked_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true);
  auto pp2 = dataflow::ProcessPoint();
  std::string pp2_name(128, 'i');
  pp2.set_name(pp2_name);
  pp2.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  std::string pp2_config_file = "./pp2_config.json";
  std::string target_bin_path = "./libxxx.so";
  {
    nlohmann::json pp2_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 2},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func2"}}}}};
    std::ofstream json_file(pp2_config_file);
    json_file << pp2_cfg_json << std::endl;
    std::ofstream target_bin(target_bin_path);
    target_bin << target_bin_path;
  }
  pp2.set_compile_cfg_file(pp2_config_file);
  std::string pp2_str;
  pp2.SerializeToString(&pp2_str);
  std::vector<std::string> pp2_attr{pp2_str};
  auto func_node0 = invoked_graph->FindNode("node1");
  AttrUtils::SetListStr(func_node0->GetOpDesc(), dflow::ATTR_NAME_DATA_FLOW_PROCESS_POINTS, pp2_attr);
  invoked_graph->SetParentNode(node0);
  invoked_graph->SetParentGraph(root_graph);
  invoked_graph->SetName("invoked_flow_graph_pp");
  root_graph->AddSubgraph("invoked_flow_graph_pp", invoked_graph);

  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  return root_graph;
}

static ComputeGraphPtr BuildDataFlowGraphWithoutUDFNodes() {
  DEF_GRAPH(flow_graph) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node0 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node1 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto node2 = OP_CFG("FlowNode").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("node1", node1));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE("node0", node0)->EDGE(0, 1)->NODE("node1", node1));
    CHAIN(NODE("node1", node1)->EDGE(0, 0)->NODE("node2", node2));
    CHAIN(NODE("node1", node1)->EDGE(0, 1)->NODE("node2", node2));
    CHAIN(NODE("node2", node2)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  auto root_graph = ToComputeGraph(flow_graph);
  (void)(AttrUtils::SetBool(root_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(root_graph, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");

  auto pp0 = dataflow::ProcessPoint();
  BuildSubGrpahForGraphPoint("pp0", "node0", root_graph, pp0);

  auto pp1 = dataflow::ProcessPoint();
  BuildSubGrpahForGraphPoint("pp1", "node1", root_graph, pp1);
  auto pp1_input0 = pp1.add_in_edges();
  pp1_input0->set_node_name("node1");
  pp1_input0->set_index(0);
  auto pp1_input1 = pp1.add_in_edges();
  pp1_input1->set_node_name("node1");
  pp1_input1->set_index(1);
  auto pp1_output0 = pp1.add_out_edges();
  pp1_output0->set_node_name("node1");
  pp1_output0->set_index(0);
  auto node1 = root_graph->FindNode("node1");
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR_DEPTH, 128);
  AttrUtils::SetStr(node1->GetOpDesc(), ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, "FIFO");

  dataflow::ProcessPoint pp2;
  BuildSubGrpahForGraphPoint("pp2", "node2", root_graph, pp2);

  uint32_t graph_id = 1U;
  root_graph->SetGraphID(graph_id);
  return root_graph;
}


/*
data0    data1
   \       /
     node0     data2
     /    \    /  |
netoutput  node1  |
             \    |
               node2
*/
Graph BuildDataFlowGraphWithRedundantNode(const std::string &name) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto data2 = dflow::FlowData("Data2", 2);
  auto node0 = dflow::FlowNode("node0", 2, 2).SetInput(0,data0).SetInput(1, data1);
  auto node1 = dflow::FlowNode("node1", 2, 1).SetInput(0,node0).SetInput(1, data2);
  auto node2 = dflow::FlowNode("node2", 2, 1).SetInput(0,node1).SetInput(1, data2);

  // function pp
  auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig("./pp0_config.json");
  node0.AddPp(pp0);

  auto pp1 = dflow::FunctionPp("func_pp1").SetCompileConfig("./pp0_config.json");
  node1.AddPp(pp1);
  auto pp2 = dflow::FunctionPp("func_pp2").SetCompileConfig("./pp0_config.json");
  node2.AddPp(pp2);
  std::vector<dflow::FlowOperator> inputsOperator{data0, data1, data2};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

bool ReadIndexFile(const std::string &index_file, std::vector<ge::CacheFileIndex> &cache_file_list) {
  nlohmann::json json_obj;
  std::ifstream file_stream(index_file);
  if (!file_stream.is_open()) {
    std::cout << "Failed to open cache index file:" << index_file << std::endl;
    return false;
  }

  try {
    file_stream >> json_obj;
  } catch (const nlohmann::json::exception &e) {
    std::cout << "Failed to read cache index file:" << index_file << ", err msg:" << e.what() << std::endl;
    file_stream.close();
    return false;
  }

  try {
    cache_file_list = json_obj["cache_file_list"].get<std::vector<ge::CacheFileIndex>>();
  } catch (const nlohmann::json::exception &e) {
    std::cout << "Failed to read cache index file:" << index_file << ", err msg:" << e.what() << std::endl;
    file_stream.close();
    return false;
  }
  file_stream.close();
  return true;
}

bool CheckCacheResult(const std::string &cache_dir, const std::string &graph_key, size_t expect_cache_size) {
  const auto cache_idx_file = cache_dir + "/" + graph_key + ".idx";
  auto check_ret = mmAccess(cache_idx_file.c_str());
  if (check_ret != 0) {
    std::cout << "Cache index file:" << cache_idx_file << " is not exist" << std::endl;
    return false;
  }

  std::vector<ge::CacheFileIndex> cache_file_list;
  if (!ReadIndexFile(cache_idx_file, cache_file_list)) {
    std::cout << "Faile to read cache index file:" << cache_idx_file << std::endl;
    return false;
  }
  for (auto &idx : cache_file_list) {
    idx.cache_file_name = cache_dir + "/" + idx.cache_file_name;
  }
  if (cache_file_list.size() != expect_cache_size) {
    std::cout << "Cache file size[" << cache_file_list.size() << "] error, expect = " << expect_cache_size << std::endl;
    return false;
  }

  for (const auto &cache_index : cache_file_list) {
    if (cache_index.graph_key != graph_key) {
      std::cout << "Cache graph_key[" << cache_index.graph_key << "] error, expect = " << graph_key << std::endl;
      return false;
    }

    if (cache_index.cache_file_name.empty()) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is empty" << std::endl;
      return false;
    }

    check_ret = mmAccess(cache_index.cache_file_name.c_str());
    if (check_ret != 0) {
      std::cout << "Cache om file:" << cache_index.cache_file_name << " is not exist" << std::endl;
      return false;
    }
  }
  return true;
}

struct StubMbufInfo {
  size_t size = 0;
  size_t len = 0;
};
class RuntimeMock : public RuntimeStub {
 public:
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) override {
    *mbuf = malloc(size);
    std::lock_guard<std::mutex> lk(mt_);
    mbuf_info_.emplace(*mbuf, StubMbufInfo{size, size});
    return 0;
  }

  rtError_t rtMbufFree(rtMbufPtr_t mbuf) override {
    if (mbuf == nullptr) {
      return 0;
    }
    std::lock_guard<std::mutex> lk(mt_);
    mbuf_info_.erase(mbuf);
    free(mbuf);
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    *databuf = mbuf;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    std::lock_guard<std::mutex> lk(mt_);
    auto find_ret = mbuf_info_.find(mbuf);
    if (find_ret == mbuf_info_.cend()) {
      return RT_FAILED;
    }
    *size = find_ret->second.size;
    return 0;
  }

  rtError_t rtMbufGetPrivInfo(rtMbufPtr_t mbuf, void **priv, uint64_t *size) override {
    static char priv_fake[256] = {};
    *priv = priv_fake;
    *size = 256;
    return 0;
  }

 private:
  std::mutex mt_;
  std::map<rtMbufPtr_t, StubMbufInfo> mbuf_info_;

};

void *mock_handle = nullptr;

class ExchangeServiceMock : public ExchangeService {
 public:
  Status CreateQueue(int32_t device_id, const string &name, const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, const FillFunc &fill_func,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status DestroyQueue(int32_t device_id, uint32_t queue_id) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, const void *data, size_t size,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout){
    return 0;
  }

  MOCK_METHOD4(DequeueTensor, Status(int32_t, uint32_t, GeTensor & , ExchangeService::ControlInfo &));
  MOCK_METHOD4(DequeueMbuf, Status(int32_t, uint32_t, rtMbufPtr_t *, int32_t));

  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }

  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) {
    return SUCCESS;
  }

  MOCK_METHOD5(Dequeue, Status(int32_t, uint32_t, void *, size_t, ExchangeService::ControlInfo &));
};
#include "dflow/flow_graph/data_flow_attr_define.h"
class ModelDeployerMock : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model, DeployResult &deploy_result) override {
    deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    deploy_result.output_queue_attrs = {{4, 0, 0}, {5, 0, 0}};
    GetModelInputAlignAttrs(flow_model->GetRootGraph(), deploy_result.input_align_attrs);
    (void)AttrUtils::GetBool(flow_model->GetRootGraph(), dflow::ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH,
                             deploy_result.is_exception_catch);
    if (deploy_result.is_exception_catch) {
      deploy_result.status_output_queue_attrs = {{101, 0, 0}};
      deploy_result.exception_notify_callback = [](const UserExceptionNotify &notify) {};
    }
    deploy_result.dev_abnormal_callback = []() { return SUCCESS; };
    return SUCCESS;
  }

  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
  }

  Status GetModelInputAlignAttrs(const ComputeGraphPtr &root_graph, InputAlignAttrs &input_align_attrs) {
    GE_CHECK_NOTNULL(root_graph);
    int64_t cache_num = 0;
    if (AttrUtils::GetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, cache_num)) {
      // max cache num is 1024.
      GE_CHK_BOOL_RET_STATUS((cache_num >= 0) && (cache_num <= 1024), PARAM_INVALID,
                             "attr[%s]=%ld is out of range [0, 1024]",
                             dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, cache_num);
    } else {
      GELOGI("no align attrs configured, graph=%s.", root_graph->GetName().c_str());
      return SUCCESS;
    }
    int64_t timeout = 0;
    if (AttrUtils::GetInt(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, timeout)) {
      // -1 means no time out, max value is 600 * 1000ms
      GE_CHK_BOOL_RET_STATUS((timeout == (-1)) || ((timeout > 0) && (timeout <= 600 * 1000)), PARAM_INVALID,
                             "attr[%s]=%ld is invalid, must be -1 or in range(0, 600 * 1000]",
                             dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, timeout);
    } else {
      GELOGE(PARAM_INVALID, "attr[%s] is not configured, graph=%s.",
             dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, root_graph->GetName().c_str());
      return PARAM_INVALID;
    }
    bool drop_when_not_aligned = false;
    (void)AttrUtils::GetBool(root_graph, dflow::ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT, drop_when_not_aligned);
    GELOGI("graph[%s] align attrs, align_max_cache_num=%ld, align_timeout=%ld, drop_when_not_align=%d.",
           root_graph->GetName().c_str(), cache_num, timeout, static_cast<int32_t>(drop_when_not_aligned));
    input_align_attrs.align_max_cache_num = static_cast<int32_t>(cache_num);
    input_align_attrs.align_timeout = static_cast<int32_t>(timeout);
    input_align_attrs.drop_when_not_align = drop_when_not_aligned;
    return SUCCESS;
  }
};

class ExecutionRuntimeMock : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return 0;
  }
  Status Finalize() override {
    return 0;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }
  const std::string &GetCompileHostResourceType() const override{
    if (set_host_) {
      return host_stub_;
    }
    return host_stub2_;
  }
  const std::map<std::string, std::string> &GetCompileDeviceInfo() const override{
    if (set_dev_) {
      return logic_dev_id_to_res_type_;
    }
    return logic_dev_id_to_res_type2_;
  }

 public:
  ExchangeServiceMock exchange_service_;
  ModelDeployerMock model_deployer_;
  bool set_host_ = false;
  bool set_dev_ = false;
  std::string host_stub_ = "stub_host_type";
  std::map<std::string, std::string> logic_dev_id_to_res_type_ = {{"0:0:0", "stub_dev_type"},
                                                                  {"0:0:1", "stub_dev_type"}};
  std::string host_stub2_ = "";
  std::map<std::string, std::string> logic_dev_id_to_res_type2_ = {};
};

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<ExecutionRuntimeMock>());
  return SUCCESS;
}


class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libmodel_deployer.so") == file_name) {
      return mock_handle;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    return 0;
  }
};

Status GetDirInfo(const std::string &dir_path, const std::string &hide_file, std::string &dir_info) {
  std::string cmd = "cd " + dir_path + ";ls -Rl --full-time --ignore=pp[1-9].om";
  if (!hide_file.empty()) {
    cmd = cmd + " --hide=" + hide_file;
  }
  cmd += std::string(" ./ 2>&1");
  std::cout << "cmd:" << cmd << std::endl;
  FILE *pipe = popen(cmd.c_str(), "r");
  GE_CHECK_NOTNULL(pipe);
  dir_info = "";
  constexpr int32_t buffer_len = 128;
  char buffer[buffer_len];
  while (!feof(pipe)) {
    if (fgets(buffer, buffer_len, pipe) != nullptr) {
      dir_info += buffer;
    }
  }
  const auto ret = pclose(pipe);
  std::cout << "cmd result:" << dir_info << std::endl;
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED, "Failed to get release info, ret[%d], errmsg[%s]", ret, dir_info.c_str());
  return SUCCESS;
}

}  // namespace
class DataFlowGraphTest : public testing::Test {
 public:
  static void PrepareForCacheConfig(bool cache_manual_check, bool cache_debug_mode) {
    std::string cache_config_file = "./build_cache_dir/cache.conf";
    {
      nlohmann::json cfg_json = {
                                  {"cache_manual_check", cache_manual_check},
                                  {"cache_debug_mode", cache_debug_mode}};
      std::ofstream json_file(cache_config_file);
      json_file << cfg_json << std::endl;
    }
  }

  static void RemoveCacheConfig() {
    remove("./build_cache_dir/cache.conf");
  }

  static void SetUpTestSuite() {
    std::string cmd = R"(
mkdir -p ./temp/build/_udf1/X86/release
cd ./temp/build/_udf1/X86/release
touch pp0_release.om
touch pp0_release.so
echo "Hello" > pp0_release.om
echo "test1_release" > pp0_release.so
tar -cvf pp0_release.tar.gz pp0_release.om pp0_release.so
rm -rf pp0_release.om pp0_release.so
cd -
mkdir -p ./temp/build/_udf1/Ascend/release
cp ./temp/build/_udf1/X86/release/pp0_release.tar.gz ./temp/build/_udf1/Ascend/release/
mkdir -p ./temp/build/_udf2/Ascend/release
cd ./temp/build/_udf2/Ascend/release
touch func_invoke_graph_pp_release.om
touch func_invoke_graph_pp_release.so
echo "Hello" > func_invoke_graph_pp_release.om
echo "test1_release" > func_invoke_graph_pp_release.so
tar -cvf func_invoke_graph_pp_release.tar.gz func_invoke_graph_pp_release.om func_invoke_graph_pp_release.so
rm -rf func_invoke_graph_pp_release.om func_invoke_graph_pp_release.so
cd -
)";
    (void)system(cmd.c_str());
  }

  static void TearDownTestSuite() {
    (void)system("rm -rf ./temp");
  }

  void SetUp() {
    ExecutionRuntime::instance_ = ge::MakeShared<ExecutionRuntimeMock>();
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    CreateCompilerJson("./cpu_compile.json");
    PrepareForOnlyHostUdf();
    PrepareForBuiltInUdf();
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
      cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so -n ${RELEASE_DIR}\n";
      cmakefile << ")\n";
      cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
      cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
    }

    std::string pp0_config_file = "./pp0_config.json";
    std::string target_bin_path = "./libxxx.so";
    {
      nlohmann::json pp0_cfg_json = {{"workspace", "./temp"},
                                     {"target_bin", "libudf1.so"},
                                     {"input_num", 2},
                                     {"output_num", 2},
                                     {"cmakelist_path", "CMakeLists.txt"},
                                     {"compiler", "./cpu_compile.json"},
                                     {
                                         "running_resources_info",
                                         {{
                                              {"type", "cpu"},
                                              {"num", 2},
                                          },
                                          {
                                              {"type", "memory"},
                                              {"num", 100},
                                          }},
                                     },
                                     {"func_list",
                                      {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}, {"stream_input", true}},
                                       {{"func_name", "func2"}, {"inputs_index", {1}}, {"outputs_index", {0}}}}},
                                     {"buf_cfg", 
                                      {{{"total_size", 2097152}, {"blk_size", 256},
                                        {"max_buf_size", 8192}, {"page_type", "normal"}},
                                      {{"total_size", 33554432}, {"blk_size", 8192},
                                       {"max_buf_size", 8388608}, {"page_type", "normal"}},
                                      {{"total_size", 2097152}, {"blk_size", 256},
                                       {"max_buf_size", 8192}, {"page_type", "huge"}},
                                      {{"total_size", 33554432}, {"blk_size", 8192},
                                       {"max_buf_size", 8388608}, {"page_type", "huge"}}
                                      }
                                     }
                                   };
      std::ofstream json_file(pp0_config_file);
      json_file << pp0_cfg_json << std::endl;
      std::ofstream target_bin(target_bin_path);
      target_bin << target_bin_path;
    }
    std::string pp1_config_file = "./pp1_config.json";
    {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}}};
      std::ofstream json_file(pp1_config_file);
      json_file << pp1_cfg_json << std::endl;
    }
    std::string pp2_config_file = "./pp2_config.json";
    {
      nlohmann::json pp2_cfg_json = {{"workspace", "./temp"},
                                     {"target_bin", "libudf2.so"},
                                     {"input_num", 2},
                                     {"output_num", 1},
                                     {"cmakelist_path", "CMakeLists.txt"},
                                     {"compiler", "./cpu_compile.json"},
                                     {"func_list",
                                      {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
                                       {{"func_name", "func2"}, {"inputs_index", {1}}, {"outputs_index", {0}}}}}};
      std::ofstream json_file(pp2_config_file);
      json_file << pp2_cfg_json << std::endl;
    }
    std::string invalid_pp0_config1_file = "./invalid_pp0_config1.json";
    {
      nlohmann::json invalid_pp0_cfg1_json = {
          {"workspace", "./temp"},
          {"target_bin", "libxxx.so"},
          {"input_num", 2},
          {"output_num", 1},
          {"cmakelist_path", "CMakeLists.txt"},
          {"compiler", "./cpu_compile.json"},
          {"func_list",
           {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
            {{"func_name", "func2"}, {"inputs_index", {2}}, {"outputs_index", {0}}}}}};
      std::ofstream json_file(invalid_pp0_config1_file);
      json_file << invalid_pp0_cfg1_json << std::endl;
    }
    std::string invalid_pp0_config2_file = "./invalid_pp0_config2.json";
    {
      nlohmann::json invalid_pp0_cfg2_json = {
          {"workspace", "./temp"},
          {"target_bin", "libxxx.so"},
          {"input_num", 2},
          {"output_num", 1},
          {"cmakelist_path", "CMakeLists.txt"},
          {"compiler", "./cpu_compile.json"},
          {"func_list",
           {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
            {{"func_name", "func2"}, {"inputs_index", {0}}, {"outputs_index", {0}}}}}};
      std::ofstream json_file(invalid_pp0_config2_file);
      json_file << invalid_pp0_cfg2_json << std::endl;
    }
    std::string invalid_pp0_config3_file = "./invalid_pp0_config3.json";
    {
      nlohmann::json invalid_pp0_cfg3_json = {
          {"workspace", "./temp"},
          {"target_bin", "libxxx.so"},
          {"input_num", 2},
          {"output_num", 1},
          {"cmakelist_path", "CMakeLists.txt"},
          {"compiler", "./cpu_compile.json"},
          {"func_list",
           {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
            {{"func_name", "func2"}, {"inputs_index", {1}}, {"outputs_index", {2}}}}}};
      std::ofstream json_file(invalid_pp0_config3_file);
      json_file << invalid_pp0_cfg3_json << std::endl;
    }
    std::string invalid_pp0_config4_file = "./invalid_pp0_config4.json";
    {
      nlohmann::json invalid_pp0_cfg4_json = {
          {"workspace", "./temp"},
          {"target_bin", "libxxx.so"},
          {"input_num", 4},
          {"output_num", 1},
          {"cmakelist_path", "CMakeLists.txt"},
          {"compiler", "./cpu_compile.json"},
          {"func_list",
           {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
            {{"func_name", "func2"}, {"inputs_index", {1}}, {"outputs_index", {0}}}}}};
      std::ofstream json_file(invalid_pp0_config4_file);
      json_file << invalid_pp0_cfg4_json << std::endl;
    }
    std::string invalid_pp0_config5_file = "./invalid_pp0_config5.json";
    {
      nlohmann::json invalid_pp0_cfg5_json = {
          {"workspace", "./temp"},
          {"target_bin", "libxxx.so"},
          {"input_num", 2},
          {"output_num", 1},
          {"cmakelist_path", "CMakeLists.txt"},
          {"compiler", "./cpu_compile.json"},
          {"func_list",
           {{{"func_name", "func1"}, {"inputs_index", {0}}, {"outputs_index", {0}}},
            {{"func_name", "func1"}, {"inputs_index", {1}}, {"outputs_index", {0}}}}}};
      std::ofstream json_file(invalid_pp0_config5_file);
      json_file << invalid_pp0_cfg5_json << std::endl;
    }
    (void)system("mkdir ./build_cache_dir");
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
    auto path = PathUtils::Join(
        {EnvPath().GetAirBasePath(), "tests/dflow/runner/st/st_run_data/json/helper_runtime/host/numa_config.json"});
    setenv("RESOURCE_CONFIG_PATH", path.c_str(), MMPA_MAX_PATH);
    mock_handle = (void *)0xffffffff;
    ReInitGe();
  }

  void TearDown() {
    ExecutionRuntime::instance_ = nullptr;
    ExecutionRuntime::FinalizeExecutionRuntime();
    MmpaStub::GetInstance().Reset();
    RuntimeStub::Reset();
    mock_handle = nullptr;
    remove("./pp0_config.json");
    remove("./libxxx.so");
    remove("./pp1_config.json");
    remove("./pp2_config.json");
    remove("./invalid_pp0_config1.json");
    remove("./invalid_pp0_config2.json");
    remove("./invalid_pp0_config3.json");
    remove("./invalid_pp0_config4.json");
    remove("./invalid_pp0_config5.json");
    remove("./cpu_compile.json");
    remove("./builtin_udf_config.json");
    std::string cmd = "rm -fr `ls ./temp/* | grep -v build`";
    (void) system(cmd.c_str());
    (void)system("rm -fr ./temp_host_udf");
    (void)system("rm -fr ./build_cache_dir");
    dflow::DFlowFinalize();
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  static void PrepareForOnlyHostUdf() {
    std::string cmd = "mkdir -p temp_host_udf; cd temp_host_udf; touch libtest.so";
    (void)system(cmd.c_str());
    std::ofstream cmakefile("./temp_host_udf/CMakeLists.txt");
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "if (\"x${RESOURCE_TYPE}\" STREQUAL \"xAscend\") \n";
    cmakefile << "message(FATAL_ERROR \"Unsupport compile Ascend target!\") \n";
    cmakefile << "endif() \n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";

    std::string host_udf_config_file = "./host_udf_config.json";
    {
      nlohmann::json host_udf_cfg_json = {{"workspace", "./temp_host_udf"},
                                          {"target_bin", "libxxx.so"},
                                          {"input_num", 2},
                                          {"output_num", 2},
                                          {"heavy_load", true},
                                          {"cmakelist_path", "CMakeLists.txt"},
                                          {"compiler", "./cpu_compile.json"},
                                          {
                                              "running_resources_info",
                                              {{
                                                   {"type", "cpu"},
                                                   {"num", 2},
                                               },
                                               {
                                                   {"type", "memory"},
                                                   {"num", 100},
                                               }},
                                          },
                                          {"func_list", {{{"func_name", "func1"}}}}};
      std::ofstream json_file(host_udf_config_file);
      json_file << host_udf_cfg_json << std::endl;
    }
  }

  static void CreateCompilerJson(const std::string &cpu_compile_config_file) {
    nlohmann::json cpu_compiler_json = {
        {"compiler",
         {
             {
                 {"resource_type", "X86"},
                 {"toolchain", "/usr/bin/g++"},
             },
             {
                 {"resource_type", "Aarch64"},
                 {"toolchain", "/usr/bin/g++"},
             },
             {
                 {"resource_type", "Ascend"},
                 {"toolchain", "/usr/local/Ascend/hcc"},
             },
         }},
    };
    std::ofstream json_file(cpu_compile_config_file);
    json_file << cpu_compiler_json << std::endl;
  }

  static void PrepareForBuiltInUdf() {
    std::string builtin_udf_config_file = "./builtin_udf_config.json";
    {
      nlohmann::json builtin_udf_cfg_json = {
                                          {"input_num", 2},
                                          {"output_num", 2},
                                          {"built_in_flow_func", true},
                                          {"heavy_load", false},
                                          {"func_list", {{{"func_name", "_BuiltIn_func1"}}}}};
      std::ofstream json_file(builtin_udf_config_file);
      json_file << builtin_udf_cfg_json << std::endl;
    }
  }
};

TEST_F(DataFlowGraphTest, Build_SUCCESS) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Compile_SUCCESS) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  auto ret = session.CompileGraph(1);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_SUCCESS_AfterPrune) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphWithRedundantNode("PruneDataFlow");
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_Failed_out_of_range_input_index) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildAbnormalDataFlowGraph("./invalid_pp0_config1.json");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_Failed_repeat_input_index) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildAbnormalDataFlowGraph("./invalid_pp0_config2.json");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_Failed_out_of_range_output_index) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildAbnormalDataFlowGraph("./invalid_pp0_config3.json");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_Failed_invalid_input_num) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildAbnormalDataFlowGraph("./invalid_pp0_config4.json");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_Failed_duplicate_func_names) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildAbnormalDataFlowGraph("./invalid_pp0_config5.json");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_with_cache_in_heterogeneous) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);

  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_host_ = true;
  session.AddGraph(1, graph, graph_options);
  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_dev_ = true;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  auto check_ret = CheckCacheResult("./build_cache_dir", "data_flow_graph_cache_key1", 1);
  EXPECT_EQ(check_ret, true);
  // remove udf om file to simulate compile failed scenario
  system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1/pp0/pp0_release.om");
  system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.om");
  system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.idx");
  auto g2 = BuildDataFlowGraph();
  auto graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(g2);
  session.AddGraph(2, graph2, graph_options);
  ret = session.BuildGraph(2, inputs);
  ASSERT_EQ(ret, SUCCESS);
  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_host_ = false;
  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_dev_ = false;
}

TEST_F(DataFlowGraphTest, Build_with_cache_in_heterogeneous_with_fake_input_err) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);

  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_host_ = true;
  session.AddGraph(1, graph, graph_options);
  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_dev_ = true;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  auto check_ret = CheckCacheResult("./build_cache_dir", "data_flow_graph_cache_key1", 1);
  EXPECT_EQ(check_ret, true);
  ComputeGraphPtr g2 = nullptr;
  {
    DEF_GRAPH(flow_graph) {
      auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
      auto node0 = OP_CFG("FlowNode").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
      auto net_output = OP_CFG("NetOutput").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
      CHAIN(NODE("arg0", data0)->EDGE(0, 0)->NODE("node0", node0)->EDGE(0, 0)->NODE("net_output", net_output));
  };
  g2 = ToComputeGraph(flow_graph);
  (void)(AttrUtils::SetBool(g2, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, true), true);
  (void)AttrUtils::SetStr(g2, ATTR_NAME_SESSION_GRAPH_ID, "xxxx");
  }
  auto graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(g2);
  session.AddGraph(2, graph2, graph_options);
  ret = session.BuildGraph(2, inputs);
  ASSERT_EQ(ret, FAILED);
  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_host_ = false;
  static_cast<ExecutionRuntimeMock *>(ExecutionRuntime::GetInstance())->set_dev_ = false;
}

TEST_F(DataFlowGraphTest, Build_with_builtin_udf) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphWithBuiltinUdf("with_builtin_udf");
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_invoke_df) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "redundant_logic_device_list":"0:0:1~2:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "logic_device_list":"0:0:1:0",
                "redundant_logic_device_list":"0:0:2:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  std::string pp1_name(128, 'x');
  auto g1 = BuildFlowGraphWithUdfCallFlowGraph(pp1_name);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_invoke_df_error) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0~1:0",
            "redundant_logic_device_list":"0:0:2~3:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./data_flow_deploy_info_df.json",
                "redundant_logic_device_list":"0:0:2:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  std::string pp1_name(128, 'x');
  auto g1 = BuildFlowGraphWithUdfCallFlowGraph(pp1_name);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_invoke_invoke_repeat_error) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0~1:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "deploy_info_file":"./data_flow_deploy_info_df.json"
              },
              {
                "invoke_name":"invoke_flow_graph_key",
                "logic_device_list":"0:0:2~3:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  std::string pp1_name(128, 'x');
  auto g1 = BuildFlowGraphWithUdfCallFlowGraph(pp1_name);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_invoke_flow_graph_pp_over_length) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0",
            "redundant_logic_device_list":"0:0:1~2:0",
            "invoke_list":[
              {
                "invoke_name":"invoke_flow_graph_key",
                "logic_device_list":"0:0:1:0",
                "redundant_logic_device_list":"0:0:2:0"
              }
            ]
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  std::string pp1_name(129, 'x');
  auto g1 = BuildFlowGraphWithUdfCallFlowGraph(pp1_name);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_cache_and_deploy_info) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"},
                                                   {"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  auto check_ret = CheckCacheResult("./build_cache_dir", "data_flow_graph_cache_key1", 1);
  EXPECT_EQ(check_ret, true);

  auto g2 = BuildDataFlowGraphWithOneUDF();
  auto graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(g2);
  session.AddGraph(2, graph2, graph_options);
  ret = session.BuildGraph(2, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_manual_cache) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"},
                                                   {"ge.experiment.data_flow_deploy_info_path", file_name}};

  PrepareForCacheConfig(true, true);
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  auto check_ret = CheckCacheResult("./build_cache_dir", "data_flow_graph_cache_key1", 1);
  EXPECT_EQ(check_ret, false);

  auto g2 = BuildDataFlowGraphWithOneUDF();
  auto graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(g2);
  session.AddGraph(2, graph2, graph_options);
  ret = session.BuildGraph(2, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
  RemoveCacheConfig();
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_format_error) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:1:0:-1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_ascend_format_error) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0,0:aa:1:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_range_format_error) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0,0:0:1~2~3:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_inputshape_success) {
  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "1~3,2,1~6;1~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_with_inputshape_and_name_success) {
  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "data0:1~3,2,4~10;data1:2~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.inputShape", "data0:1~3,2,1~10;data1:1~3,2~3,3"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(DataFlowGraphTest, Build_with_inputshape_and_name_failed_lack_input) {
  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "data3:1~3,2,4~10;data1:2~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.inputShape", "data3:1~3,2,4~10;data1:2~3,2~3,3"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(DataFlowGraphTest, Build_with_inputshape_and_name_failed_error_config) {
  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.inputShape", "data0:1~,2,4~10;data1:2~3,2~3,3"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.inputShape", "data0:1~,2,4~10;data1:2~3,2~3,3"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(DataFlowGraphTest, Build_with_shape_range_success) {
  remove("./pp1_config.json");
  std::string pp_config_file = "./pp1_config.json";
  {
      nlohmann::json pp1_cfg_json = {{"inputs_tensor_desc",
                                      {{{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}, {"format", "ND"}},
                                       {{"shape", {1, 2, 3}}, {"data_type", "DT_INT32"}}}},
                                       {"build_options", {{"ge.exec.dynamicGraphExecuteMode", "dynamic_execute"},
                                                   {"ge.exec.dataInputsShapeRange", "[1,2,2~3],[1,2,2~3]"}}}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

rtMbufPtr_t GenerateExceptionBuf(const std::string &scope, uint64_t trans_id, int32_t exception_code) {
  domi::SubmodelStatus submodel_status;
  submodel_status.set_msg_type(1);
  auto exception = submodel_status.mutable_exception();
  exception->set_scope(scope);
  exception->set_trans_id(trans_id);
  exception->set_exception_code(exception_code);
  uint8_t user_context[256] = {};
  exception->set_exception_context(user_context, sizeof(user_context));
  rtMbufPtr_t req_msg_mbuf = nullptr;
  ScopeGuard guard([&req_msg_mbuf]() {
    if (req_msg_mbuf != nullptr) {
      rtMbufFree(req_msg_mbuf);
    }
  });
  void *input_buffer = nullptr;
  auto req_msg_mbuf_size = submodel_status.ByteSizeLong();
  auto rt_ret = rtMbufAlloc(&req_msg_mbuf, req_msg_mbuf_size);
  if (rt_ret != RT_ERROR_NONE) {
    return nullptr;
  }
  rt_ret = rtMbufSetDataLen(req_msg_mbuf, req_msg_mbuf_size);
  if (rt_ret != RT_ERROR_NONE) {
    return nullptr;
  }
  rt_ret = rtMbufGetBuffAddr(req_msg_mbuf, &input_buffer);
  if (rt_ret != RT_ERROR_NONE) {
    return nullptr;
  }
  bool serial_ret = submodel_status.SerializeToArray(input_buffer, static_cast<int32_t>(req_msg_mbuf_size));
  if (!serial_ret) {
    return nullptr;
  }
  guard.Dismiss();
  return req_msg_mbuf;
}

TEST_F(DataFlowGraphTest, Build_host_udf_call_nn_with_deploy_info) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node2"],
            "logic_device_list":"0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:0:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  auto runtime = (ExecutionRuntimeMock *)ExecutionRuntime::GetInstance();
  bool dequeue_mbuf_empty = true;
  auto dequeue_mbuf_stub = [&dequeue_mbuf_empty](int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf,
                                                 int32_t timeout) {
    if (dequeue_mbuf_empty) {
      return RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY);
    }
    static uint64_t cnt = 0;
    GELOGI("generate exception buf begin, queue_id=%u, cnt=%d", queue_id, cnt);
    if (cnt < 1024) {
      *m_buf = GenerateExceptionBuf("ignore scope", cnt, 100);
    } else if (cnt == 1024) {
      *m_buf = GenerateExceptionBuf("", 4, 200);
    } else {
      usleep(300 * 1000);
      return RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY);
    }

    ++cnt;
    if (*m_buf == nullptr) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "generate exception buf failed");
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    GELOGI("generate exception buf success, queue_id=%u, cnt=%d", queue_id, cnt);
    return SUCCESS;
  };
  EXPECT_CALL(runtime->exchange_service_, DequeueMbuf).WillRepeatedly(Invoke(dequeue_mbuf_stub));
  EXPECT_CALL(runtime->exchange_service_, DequeueTensor).WillRepeatedly(Return(SUCCESS));

  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphWithHostUdfCallNn("host_call_nn", true);

  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  std::vector<Tensor> output;
  DataFlowInfo info;
  int32_t timeout = 2000;
  ret = session.FetchDataFlowGraph(1, output, info, timeout);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(output.size(), 2);
  output.clear();
  // align must fetch same index
  ret = session.FetchDataFlowGraph(1, {0},output, info, timeout);
  EXPECT_NE(ret, SUCCESS);
  ret = session.FetchDataFlowGraph(1, {0, 1}, output, info, timeout);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(output.size(), 2);
  output.clear();

  EXPECT_CALL(runtime->exchange_service_, DequeueTensor).WillRepeatedly(Return(ACL_ERROR_GE_REDEPLOYING));
  ret = session.FetchDataFlowGraph(1, output, info, timeout);
  EXPECT_EQ(ret, ACL_ERROR_GE_REDEPLOYING);

  EXPECT_CALL(runtime->exchange_service_, DequeueTensor)
      .WillOnce(Return(RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)))
      .WillRepeatedly(Return(ACL_ERROR_GE_REDEPLOYING));
  ret = session.FetchDataFlowGraph(1, output, info, timeout);
  EXPECT_EQ(ret, ACL_ERROR_GE_REDEPLOYING);

  struct TmpHeadInfo {
    uint64_t trans_id;
    uint32_t data_label;
    int32_t ret_code;
    uint64_t start_time;
  };
  std::mutex queue_mt;
  std::queue<TmpHeadInfo> head_info_queue;
  head_info_queue.emplace(TmpHeadInfo{1, 100, 0, 1}); // 0
  head_info_queue.emplace(TmpHeadInfo{1, 101, 999, 2}); // 1
  head_info_queue.emplace(TmpHeadInfo{2, 100, 0, 3}); // 0
  head_info_queue.emplace(TmpHeadInfo{1, 100, 0, 1}); // 1
  head_info_queue.emplace(TmpHeadInfo{1, 101, 999, 2}); // 0
  head_info_queue.emplace(TmpHeadInfo{2, 100, 0, 3}); // 1

  auto func = [&head_info_queue, &queue_mt](const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                                               ExchangeService::ControlInfo &control_info) {
    std::lock_guard<std::mutex> lk(queue_mt);
    if (head_info_queue.empty()) {
      return RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY);
    }
    const auto &trans_id_data_label = head_info_queue.front();
    control_info.msg_info->trans_id = trans_id_data_label.trans_id;
    control_info.msg_info->data_label = trans_id_data_label.data_label;
    control_info.msg_info->ret_code = trans_id_data_label.ret_code;
    control_info.msg_info->start_time = trans_id_data_label.start_time;
    head_info_queue.pop();
    return SUCCESS;
  };
  EXPECT_CALL(runtime->exchange_service_, DequeueTensor).WillRepeatedly(Invoke(func));
  std::map<uint64_t, int32_t> start_time_ret_map;
  for (int32_t i = 0; i < 3; ++i) {
    ret = session.FetchDataFlowGraph(1, output, info, timeout);
    start_time_ret_map[info.GetStartTime()] = ret;
    output.clear();
  }
  std::map<uint64_t, int32_t> expect_map = {{1, 0}, {2, 999}, {3, 0}};
  EXPECT_EQ(start_time_ret_map, expect_map);
  EXPECT_TRUE(head_info_queue.empty());
  ret = session.FetchDataFlowGraph(1, output, info, 100);
  EXPECT_EQ(ret, ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT);
  EXPECT_EQ(output.size(), 0);
  {
    std::lock_guard<std::mutex> lk(queue_mt);
    head_info_queue.emplace(TmpHeadInfo{1, 100, 0, 4});
    head_info_queue.emplace(TmpHeadInfo{1, 101, 0, 5});
    head_info_queue.emplace(TmpHeadInfo{1, 100, 0, 6});
    head_info_queue.emplace(TmpHeadInfo{2, 0, 0, 7});
    head_info_queue.emplace(TmpHeadInfo{3, 0, 0, 8});
    head_info_queue.emplace(TmpHeadInfo{4, 0, 0, 9});
    head_info_queue.emplace(TmpHeadInfo{5, 0, 0, 10});
    head_info_queue.emplace(TmpHeadInfo{6, 0, 0, 11});
  }
  // data not align and over limit take not finish
  ret = session.FetchDataFlowGraph(1, output, info, timeout);
  EXPECT_EQ(ret, ACL_ERROR_GE_DATA_NOT_ALIGNED);
  EXPECT_EQ(info.GetStartTime(), 4);
  // data not align and over limit take finish
  ret = session.FetchDataFlowGraph(1, output, info, timeout);
  EXPECT_EQ(ret, ACL_ERROR_GE_DATA_NOT_ALIGNED);
  EXPECT_EQ(info.GetStartTime(), 6);

  RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
  dequeue_mbuf_empty = false;
  ret = session.FetchDataFlowGraph(1, output, info, timeout);
  EXPECT_EQ(ret, ACL_ERROR_GE_USER_RAISE_EXCEPTION);
  {
    std::lock_guard<std::mutex> lk(queue_mt);
    head_info_queue.emplace(TmpHeadInfo{4, 0, 0, 9});
  }
  // ignore exception trans id
  ret = session.FetchDataFlowGraph(1, output, info, 500);
  EXPECT_EQ(ret, ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT);
  RuntimeStub::Reset();
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_deploy_info_not_all) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphWithHostUdfCallNn("host_call_nn");
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_repeat_config) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_repeat_config) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1:0"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:1"
          },
          {
            "flow_node_list":["node1", "node2"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_both_deploy_info) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node1", "node2"],
            "logic_device_list":"0:0:0:0~1,0:0:2~4:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_start_over_end) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node1", "node2"],
            "logic_device_list":"0:0:0:0~1,0:0:4~2:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_end_out_range) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node1", "node2"],
            "logic_device_list":"0:0:0~1,0:4~222222222:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_both_deploy_info_repeat_config) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          }
        ],
        "batch_deploy_info": [
          {
            "flow_node_list":["node0", "node1", "node2"],
            "logic_device_list":"0:0:0:0~1,0:0:2~4:0~1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, FAILED_InvokeKeysRepeat) {
  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphInvokeKeyRepeat("repeat_invoke_keys");
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(DataFlowGraphTest, Build_with_deploy_info_with_head_node) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphWithHostUdf("test_host_udf");
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_cache_exception_without_align_attrs) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto graph = BuildDataFlowGraphWithHostUdf("test_host_udf");
  // todo use interface to set catch exception
  const bool exception_catch = true;
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  AttrUtils::SetBool(compute_graph, "_enable_exception_catch", exception_catch);

  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_sub_cache_and_deploy_info) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0,0:0:1:1"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithOneUDF();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  auto pp1_graph = g1->GetSubgraph("pp1");
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"},
                                                   {"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.om");
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.idx");
  // udf build info will update compile info
  std::string old_dir_info;
  ASSERT_EQ(GetDirInfo("./build_cache_dir/data_flow_graph_cache_key1", "pp0", old_dir_info), SUCCESS);

  auto g2 = BuildDataFlowGraphWithOneUDF();
  auto graph2 = GraphUtilsEx::CreateGraphFromComputeGraph(g2);
  session.AddGraph(2, graph2, graph_options);
  ret = session.BuildGraph(2, inputs);
  ASSERT_EQ(ret, SUCCESS);
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.om");
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.idx");
  // udf build info will update compile info
  std::string new_dir_info;
  ASSERT_EQ(GetDirInfo("./build_cache_dir/data_flow_graph_cache_key1", "pp0", new_dir_info), SUCCESS);
  ASSERT_EQ(old_dir_info, new_dir_info);

  constexpr const char *file_name_new = "./st_data_flow_deploy_info_new.json";
  std::ofstream json_file_new(file_name_new);
  std::string content_new = R"(
      {
        "deploy_info": [
          {
            "flow_node_name":"node0",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node1",
            "logic_device_id":"0:0:1:0"
          },
          {
            "flow_node_name":"node2",
            "logic_device_id":"0:0:0:0"
          }
        ]
      })";
  json_file_new << content_new << std::endl;
  json_file_new.close();
  map<AscendString, AscendString> graph_options_new = {{"ge.graph_key", "data_flow_graph_cache_key1"},
                                                       {"ge.experiment.data_flow_deploy_info_path", file_name_new}};
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.om");
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.idx");
  // udf build info will update compile info
  ASSERT_EQ(GetDirInfo("./build_cache_dir/data_flow_graph_cache_key1", "pp0", old_dir_info), SUCCESS);
  auto g3 = BuildDataFlowGraphWithOneUDF();
  auto graph3 = GraphUtilsEx::CreateGraphFromComputeGraph(g3);
  session.RemoveGraph(1);
  session.AddGraph(1, graph3, graph_options_new);
  ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.om");
  (void)system("rm -rf ./build_cache_dir/data_flow_graph_cache_key1*.idx");
  // udf build info will update compile info
  ASSERT_EQ(GetDirInfo("./build_cache_dir/data_flow_graph_cache_key1", "pp0", new_dir_info), SUCCESS);
  ASSERT_EQ(old_dir_info, new_dir_info);
  remove(file_name);
  remove(file_name_new);
}

namespace {
Graph BuildGeGraph() {
  DEF_GRAPH(graph_def) {
    auto data0 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto data1 = OP_CFG("Data").InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3});
    auto add = OP_CFG("Add").InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1, 2, 3}).Build("add");
    auto netoutput = OP_CFG("NetOutput").InCnt(1).OutCnt(1);
    CHAIN(NODE("data0", data0)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE("data1", data1)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 0)->NODE("netoutput", netoutput));
  };
  Graph sub_graph = ToGeGraph(graph_def);
  auto sub_compute_graph = GraphUtilsEx::GetComputeGraph(sub_graph);
  (void)ge::AttrUtils::SetStr(sub_compute_graph, ge::ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_NPU);
  auto add_node = sub_compute_graph->FindNode("add");
  sub_compute_graph->RemoveOutputNode(add_node);
  NodePtr output_node = sub_compute_graph->FindFirstNodeMatchType("NetOutput");
  EXPECT_TRUE(output_node != nullptr);
  return sub_graph;
}
}  // namespace

TEST_F(DataFlowGraphTest, BuildModelRelationNodeNotUse) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto data1 = dflow::FlowData("Data1", 1);
  auto node0 = dflow::FlowNode("Node0", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto graph_pp0 = dflow::GraphPp("graph_pp0", BuildGeGraph);
  node0.AddPp(graph_pp0);
  auto node1 = dflow::FlowNode("Node1", 2, 1).SetInput(0, data0).SetInput(1, data1);
  auto graph_pp1 = dflow::GraphPp("graph_pp1", BuildGeGraph);
  node1.AddPp(graph_pp1);

  std::vector<dflow::FlowOperator> inputsOperator{data0, data1};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph("FlowGraph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  std::map<AscendString, AscendString> options = {{"ge.runFlag", "0"}};
  Session session(options);
  session.AddGraph(0, flow_graph.ToGeGraph());
  std::vector<InputTensorInfo> inputs;
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
  auto ret = session.BuildGraph(0, inputs);
  ASSERT_EQ(ret, SUCCESS);
  RuntimeStub::Reset();
}

TEST_F(DataFlowGraphTest, DataFlowInfoSetUserData_Failed_InvalidParams) {
  DataFlowInfo info;
  ASSERT_EQ(info.SetUserData(nullptr, 0U), ACL_ERROR_GE_PARAM_INVALID);
  int8_t user_data[65];
  ASSERT_EQ(info.SetUserData(user_data, 0U), ACL_ERROR_GE_PARAM_INVALID);
  ASSERT_EQ(info.SetUserData(user_data, 65U), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(DataFlowGraphTest, DataFlowInfoGetUserData_Failed_InvalidParams) {
  DataFlowInfo info;
  ASSERT_EQ(info.GetUserData(nullptr, 0U), ACL_ERROR_GE_PARAM_INVALID);
  int8_t user_data[65];
  ASSERT_EQ(info.GetUserData(user_data, 0U), ACL_ERROR_GE_PARAM_INVALID);
  ASSERT_EQ(info.GetUserData(user_data, 65U), ACL_ERROR_GE_PARAM_INVALID);
}

/*
 * 用例描述: 构造dataflow图，用户指定部署节点和device上内存配置
 * 预期结果：
 * 1. 模型Build成功
 */
TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_and_mem_cfg) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}

/*
 * 用例描述: 构造dataflow图，用户指定部署节点和device上内存配置，device 内存配置错误
 * 预期结果：
 * 1. 模型Build失败
 */
TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_and_mem_cfg_Invalid1) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:0:0,0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"afwg",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  remove(file_name);
}

/*
 * 用例描述: 构造dataflow图，用户指定部署节点和device上内存配置，配置内存限制的device数量小于部署模型的device数量，报错
 * 预期结果：
 * 1. 模型Build失败
 */
TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_and_mem_cfg_Invalid2) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  remove(file_name);
}

/*
 * 用例描述: 构造dataflow图, 内存配置值异常 超过整形范围，执行报错
 * 预期结果：
 * 1. 模型Build失败
 */
TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_and_mem_cfg_Invalid3) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"9999999999999999999999999999999999999999999999999999999999999999999",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:2:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  remove(file_name);
}

/*
 * 用例描述: 构造dataflow图, 相通device 内存重复配置，且值不相同，build报错
 * 预期结果：
 * 1. 模型Build失败
 */
TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_and_mem_cfg_Invalid4) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:0:1:0~1,0:0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0:0"
          }
        ],
        "deploy_mem_info":[
          {
            "std_mem_size":"1024",
            "shared_mem_size":"1024",
            "logic_device_id":"0:0:1:0"
          },
          {
            "std_mem_size":"2048",
            "shared_mem_size":"2048",
            "logic_device_id":"0:0:1:0,0:0:0~1:1"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  remove(file_name);
}

TEST_F(DataFlowGraphTest, Build_with_cache_open_lock_file_failed) {
  class MockMmpaForOpenFailed : public MockMmpa {
   public:
    INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
      return -1;
    }
  };
  mock_handle = (void *)0xffffffff;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForOpenFailed>());
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);

  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
}

TEST_F(DataFlowGraphTest, Build_with_cache_lock_file_failed) {
  class MockMmpaForFlockFailed : public MockMmpa {
   public:
    INT32 Open2(const CHAR *path_name, INT32 flags, MODE mode) override {
      return INT32_MAX;
    }
  };
  mock_handle = (void *)0xffffffff;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaForFlockFailed>());
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraphWithoutUDFNodes();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {{"ge.graph_compiler_cache_dir", "./build_cache_dir"}};
  map<AscendString, AscendString> graph_options = {{"ge.graph_key", "data_flow_graph_cache_key1"}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);

  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, FAILED);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
}

TEST_F(DataFlowGraphTest, Build_with_batch_deploy_info_and_dynamic_sched) {
  constexpr const char *file_name = "./st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "dynamic_schedule_enable": true,
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:1"
          },
          {
            "flow_node_list":["node1"],
            "logic_device_list":"0:1:0~1,0:0:1"
          },
          {
            "flow_node_list":["node2"],
            "logic_device_list":"0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  mock_handle = (void *)0xffffffff;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  auto g1 = BuildDataFlowGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  map<AscendString, AscendString> session_options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Session session(session_options);
  session.AddGraph(1, graph, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
  remove(file_name);
}
TEST_F(DataFlowGraphTest, GetAscendLatestInstallPath) {
  constexpr const char_t *kAscendHomePath = "ASCEND_HOME_PATH";
  std::string ascend_home_path("/test/ascend_path");
  mmSetEnv(kAscendHomePath, ascend_home_path.c_str(), 1);
  std::string path = FunctionCompile::GetAscendLatestInstallPath();
  EXPECT_EQ(path, ascend_home_path);

  unsetenv(kAscendHomePath);
  path = FunctionCompile::GetAscendLatestInstallPath();
  EXPECT_EQ(path, "/usr/local/Ascend/cann");
}
}  // namespace ge
