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

#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "mmpa/src/mmpa_stub.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/ub_fusion/automatic_buffer_fusion.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/conv_weight_compress_fusion_pass.h"

#undef protected
#undef private
using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;

class UTEST_check_graph_cycle_unittest : public testing::Test {
 protected:
  virtual void SetUp() {

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

};

TEST_F(UTEST_check_graph_cycle_unittest, check_graph_cycle_yes) {
  std::string path = GetCurpath() + "../../../../../..";
  setenv("ASCEND_OPP_PATH", path.c_str(), 1);
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out);
  GraphFusion graphFusion(nullptr, nullptr, nullptr);
  setenv("ENABLE_NETWORK_ANALYSIS_DEBUG", "1", true);
  Configuration config(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config.Initialize(options);
  bool ret = false;
  if (config.IsEnableNetworkAnalysis()) {
    ret = graphFusion.CheckGraphCycle(*graph_out);
  }

  EXPECT_EQ(ret, true);
  unsetenv("ASCEND_OPP_PATH");
}

TEST_F(UTEST_check_graph_cycle_unittest, check_graph_cycle_pass_success) {
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
  FusionPassRegistry::PassDesc pass_desc = {4, create_func};
  fe::FusionPassOrRule pass_or_rule("PaddDepthwiseConv2dFusionPass", 0, PASS_METHOD, BUILT_IN_PASS_PRIORITY_MIN,
                                    pass_desc);
  Status status = graphFusion.RunOnePassFusion(*graph_out, pass_or_rule, {});

  EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(UTEST_check_graph_cycle_unittest, report_cycle_after_pass_fusion) {
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
  FusionPassRegistry::PassDesc pass_desc = {4, create_func};
  fe::FusionPassOrRule pass_or_rule("ConcatQuantFusionPass",
                                    0, PASS_METHOD,
                                    BUILT_IN_PASS_PRIORITY_MIN,
                                    pass_desc);
  graphFusion.ReportAfterCheckGraphCycle(*graph_out, pass_or_rule);
}

TEST_F(UTEST_check_graph_cycle_unittest, report_cycle_after_rule_fusion) {
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
  fe::FusionPassOrRule pass_or_rule("AdamApplyoneRuleCond2", 0, RULE_METHOD, BUILT_IN_RULE_PRIORITY_MIN,
                                    pass_desc);
  fe::FusionPassOrRule pass_or_rule_2("AdamApplyoneRuleCond2", 0, RULE_METHOD, BUILT_IN_RULE_PRIORITY_MIN,
                                      pass_desc);
  graphFusion.ReportAfterCheckGraphCycle(*graph_out, pass_or_rule);
  graphFusion.ReportAfterCheckGraphCycle(*graph_out, pass_or_rule_2);
}
