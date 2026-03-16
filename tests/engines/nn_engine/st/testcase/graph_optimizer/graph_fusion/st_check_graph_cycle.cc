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
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <map>
#include <memory>

#include "proto/om.pb.h"

#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "mmpa/src/mmpa_stub.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/conv_weight_compress_fusion_pass.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
#include "../../../../graph_constructor/graph_constructor.h"
#undef protected
#undef private
using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;

class STEST_st_check_graph_cycle : public testing::Test {
 protected:
  std::shared_ptr<fe::GraphComm> graph_comm_ptr_;
  virtual void SetUp() {
    std::shared_ptr<std::mutex> ffts_lock_ptr = std::make_shared<std::mutex>();
    graph_comm_ptr_ = std::make_shared<fe::GraphComm>(fe::AI_CORE_NAME, ffts_lock_ptr);
    graph_comm_ptr_->Initialize();
  }

  virtual void TearDown() {

  }
  void SetPattern(ge::OpDescPtr opdef, string optype) {
    auto key_pattern = "_pattern";
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
  }
  void SetTvmType(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  }

  void BuildGraph(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", fe::DATA);
    OpDescPtr addn = std::make_shared<OpDesc>("addn", "AddN");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1","ReLU");
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2","ReLU");
    SetPattern(addn, "ElemWise");
    SetPattern(elemwise, "ElemWise");

    SetTvmType(addn);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc tenosr_desc(shape);

    data->AddOutputDesc(tenosr_desc);
    data1->AddOutputDesc(tenosr_desc);
    data2->AddOutputDesc(tenosr_desc);
    addn->AddInputDesc(tenosr_desc);
    addn->AddInputDesc(tenosr_desc);
    addn->AddOutputDesc(tenosr_desc);
    elemwise->AddInputDesc(tenosr_desc);
    elemwise->AddOutputDesc(tenosr_desc);
    relu->AddInputDesc(tenosr_desc);
    relu->AddInputDesc(tenosr_desc);
    relu->AddOutputDesc(tenosr_desc);
    relu1->AddInputDesc(tenosr_desc);
    relu1->AddInputDesc(tenosr_desc);
    relu1->AddOutputDesc(tenosr_desc);
    relu2->AddInputDesc(tenosr_desc);
    relu2->AddOutputDesc(tenosr_desc);
    AttrUtils::SetInt(addn, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr addn_node = graph->AddNode(addn);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(addn_node->GetName(), std::move(buffer));
    addn_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), addn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), addn_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(addn_node->GetOutDataAnchor(0), elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(1));
  }

  void BuildGraph2(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", fe::DATA);
    OpDescPtr addn = std::make_shared<OpDesc>("addn", "AddN");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1","ReLU");
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2","ReLU");
    SetPattern(addn, "ElemWise");
    SetPattern(elemwise, "ElemWise");

    SetTvmType(addn);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc tenosr_desc(shape);

    data->AddOutputDesc(tenosr_desc);
    data1->AddOutputDesc(tenosr_desc);
    data2->AddOutputDesc(tenosr_desc);
    addn->AddInputDesc(tenosr_desc);
    addn->AddInputDesc(tenosr_desc);
    addn->AddOutputDesc(tenosr_desc);
    elemwise->AddInputDesc(tenosr_desc);
    elemwise->AddOutputDesc(tenosr_desc);
    relu->AddInputDesc(tenosr_desc);
    relu->AddOutputDesc(tenosr_desc);
    relu1->AddInputDesc(tenosr_desc);
    relu1->AddOutputDesc(tenosr_desc);
    relu2->AddInputDesc(tenosr_desc);
    relu2->AddOutputDesc(tenosr_desc);
    AttrUtils::SetInt(addn, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr addn_node = graph->AddNode(addn);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(addn_node->GetName(), std::move(buffer));
    addn_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), addn_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), addn_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(addn_node->GetOutDataAnchor(0), elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
  }

  static void CreateGraphOnlyControlEdge(ComputeGraphPtr &graph) {
    ge::GeShape shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, shape);
    test.AddOpDesc("a", "Test", 1, 1)
        .AddOpDesc("data", "Data", 0, 0)
        .SetInput("a:-1", "data:-1")
        .AddOpDesc("b", "Test", 1, 1)
        .SetInputs({"a"})
        .AddOpDesc("netout", "NetOutput", 0, 0)
        .SetInput("netout:-1", "b:-1");
  }

  static void CreateGraphTwoControlEdgesToOuterNodes(ComputeGraphPtr &graph) {
    ge::GeShape shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, shape);
    test.AddOpDesc("a", "Test", 1, 1)
        .AddOpDesc("data", "Data", 0, 0)
        .SetInput("a:-1", "data:-1")
        .AddOpDesc("b", "Test", 1, 1)
        .SetInputs({"a"})
        .AddOpDesc("c", "TestC", 0, 0)
        .AddOpDesc("netout", "NetOutput", 0, 0)
//        .SetInput("c:-1", "a:-1")
        .SetInput("netout:-1", "b:-1")
        .SetInput("c:-1", "b:-1");
  }

  static void CreateGraphContainsControlEdge(ComputeGraphPtr &graph) {
    ge::GeShape shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, shape);
    test.AddOpDesc("a", "Test", 1, 1)
        .SetInputs({"Data0"})
        .AddOpDesc("datacontrol0", "DataControl0", 0, 0)
        .AddOpDesc("datacontrol1", "DataControl1", 0, 0)
        .SetInput("a:-1", "datacontrol0:-1")
        .AddOpDesc("b", "Test", 1, 1)
        .SetInputs({"a"})
        .SetInput("b:-1", "datacontrol1:-1")
        .AddOpDesc("netout", "NetOutput", 0, 0)
        .SetInput("netout:-1", "b:-1");
  }

  static void CreateGraphDataEdge(ComputeGraphPtr &graph) {
    ge::GeShape shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, shape);
    test.AddOpDesc("a", "Test", 1, 1)
        .SetInputs({"Data0"})
        .AddOpDesc("b", "Test", 1, 1)
        .SetInputs({"a"})
        .AddOpDesc("netout", "NetOutput", 1, 0)
        .SetInput("netout:0", "b:0");
  }

  void RunAndCheck(ComputeGraphPtr &graph) {
    ge::AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "0_0");
    std::vector<ge::NodePtr> node_vec = {};
    for (auto &node : graph->GetDirectNode()) {
      auto op_desc_ptr = node->GetOpDesc();
      if (op_desc_ptr->GetName() == "a" || op_desc_ptr->GetName() == "b") {
        node_vec.emplace_back(node);
      }
    }
//    ge::GraphUtils::DumpGEGraphToOnnx(*graph, "before");
    ge::NodePtr node = graph_comm_ptr_->TransSingleSubGraph(*(graph.get()), node_vec);
//    ge::GraphUtils::DumpGEGraphToOnnx(*graph, "after");
    int i = 0;
    for (const auto &sub_graph_func : (graph)->GetAllSubgraphs()) {
      const auto sub_graph_func_name = std::string("test") + std::string("_sub_graph_") + std::to_string(i++);
//      ge::GraphUtils::DumpGEGraphToOnnx(*sub_graph_func, sub_graph_func_name);
    }

    EXPECT_EQ(graph->TopologicalSorting(), fe::SUCCESS);
    EXPECT_NE(node, nullptr);
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kInFixpipeSubGraph, true);
    const auto &filter = [](const ge::ComputeGraphPtr &graph_ptr) {
      return true;
    };
    auto sub_graphs = graph->GetAllSubgraphs();
    for (const auto &sub_graph : sub_graphs) {
      if (sub_graph == nullptr || sub_graph->GetParentNode() == nullptr) {
        continue;
      }
      if (sub_graph->GetParentNode()->GetName() != node->GetName()) {
        continue;
      }

      if (ge::GraphUtils::UnfoldGraph(sub_graph, graph, node, filter) != ge::GRAPH_SUCCESS) {
        continue;
      }
    }

//    ge::GraphUtils::DumpGEGraphToOnnx(*graph, "after2");
    EXPECT_EQ(graph->TopologicalSorting(), fe::SUCCESS);
  }
};

string GraphCycleGetAscendPath() {
  const char *ascend_custom_path_ptr = std::getenv("ASCEND_INSTALL_PATH");
  string ascend_path;
  if (ascend_custom_path_ptr != nullptr) {
    ascend_path = fe::RealPath(string(ascend_custom_path_ptr));
  } else {
    const char *ascend_home_path_ptr = std::getenv("ASCEND_HOME");
    if (ascend_home_path_ptr != nullptr) {
      ascend_path = fe::RealPath(string(ascend_home_path_ptr));
    } else {
      ascend_path = "/mnt/d/Ascend";
    }
  }
  return ascend_path;
}

TEST_F(STEST_st_check_graph_cycle, check_graph_cycle_yes) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out);
  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", true);
  Configuration config(fe::AI_CORE_NAME);
  std::map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config.Initialize(options);
  config.content_map_.clear();
  config.content_map_.emplace("op.store.tbe-builtin", "2|6|/tests/engines/nn_engine/config/fe_config|/tests/engines/nn_engine/config/fe_config|true|true");
  config.ascend_ops_path_ = GetCurpath() + "../../../../../..";
  bool ret = false;
  if (config.IsEnableNetworkAnalysis()) {
    ret = graphFusion.CheckGraphCycle(*graph_out);
  }

  EXPECT_EQ(ret, true);
}

TEST_F(STEST_st_check_graph_cycle, check_graph_cycle_no) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph2(graph_out);
  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", true);
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config.Initialize(options);
  config.content_map_.clear();
  config.content_map_.emplace("op.store.tbe-builtin", "2|6|/tests/engines/nn_engine/config/fe_config|/tests/engines/nn_engine/config/fe_config|true|true");
  config.ascend_ops_path_ = GetCurpath() + "../../../../../..";
  bool ret = true;
  if (config.IsEnableNetworkAnalysis()) {
    ret = graphFusion.CheckGraphCycle(*graph_out);
  }

  EXPECT_EQ(ret, false);
}

TEST_F(STEST_st_check_graph_cycle, check_graph_cycle_pass_success) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out);
  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", true);
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config.Initialize(options);
  auto create_func = []() -> ::fe::GraphPass * { return new (std::nothrow) ConvWeightCompressFusionPass(); };
  FusionPassRegistry::PassDesc pass_desc = {0, create_func};
  fe::FusionPassOrRule pass_or_rule("PaddDepthwiseConv2dFusionPass", 0, PASS_METHOD, BUILT_IN_PASS_PRIORITY_MIN,
                                    pass_desc);
  Status status = graphFusion.RunOnePassFusion(*graph_out, pass_or_rule, {});

  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(STEST_st_check_graph_cycle, report_cycle_after_pass_fusion) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out);
  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", true);
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  EXPECT_EQ(PlatformUtils::Instance().soc_version_, "Ascend310");
  config.Initialize(options);
  auto create_func = []() -> ::fe::GraphPass * { return new (std::nothrow) ConvWeightCompressFusionPass(); };
  FusionPassRegistry::PassDesc pass_desc = {0, create_func};
  FusionPassOrRule pass_or_rule("ConcatQuantFusionPass",
                               0, PASS_METHOD,
                               BUILT_IN_PASS_PRIORITY_MIN,
                               pass_desc);
  graphFusion.ReportAfterCheckGraphCycle(*graph_out, pass_or_rule);
}

TEST_F(STEST_st_check_graph_cycle, only_control_edges_case)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphOnlyControlEdge(graph);
  RunAndCheck(graph);
  int count = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "afunction_graph_0/a") {
      EXPECT_EQ(node->GetInControlNodes().size(), 1);
      ++count;
    }
    if (node->GetName() == "netout") {
      EXPECT_EQ(node->GetInControlNodes().size(), 1);
      ++count;
    }
  }
  EXPECT_EQ(count, 2);
}

TEST_F(STEST_st_check_graph_cycle, two_control_edges_to_outer_nodes)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphTwoControlEdgesToOuterNodes(graph);
  RunAndCheck(graph);
  int count = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "afunction_graph_0/a") {
      EXPECT_EQ(node->GetInControlNodes().size(), 1);
      ++count;
    }
    if (node->GetName() == "c") {
      EXPECT_EQ(node->GetInControlNodes().size(), 1);
      ++count;
    }
    if (node->GetName() == "netout") {
      EXPECT_EQ(node->GetInControlNodes().size(), 1);
      ++count;
    }
  }
  EXPECT_EQ(count, 3);
}


TEST_F(STEST_st_check_graph_cycle, input_contain_both_data_and_control_edges_case)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphContainsControlEdge(graph);
  RunAndCheck(graph);
  int count = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "afunction_graph_0/a") {
      EXPECT_EQ(node->GetInControlNodes().size(), 2);
      ++count;
    } else if (node->GetName() == "afunction_graph_0/b") {
      EXPECT_EQ(node->GetInControlNodes().size(), 0);
      ++count;
    }
    if (node->GetName() == "netout") {
      EXPECT_EQ(node->GetInControlNodes().size(), 1);
      ++count;
    }
  }
  EXPECT_EQ(count, 3);
}

TEST_F(STEST_st_check_graph_cycle, data_edges_case)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphDataEdge(graph);
  RunAndCheck(graph);
}
