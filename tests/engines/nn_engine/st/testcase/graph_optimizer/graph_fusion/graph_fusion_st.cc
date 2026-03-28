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
#include "fe_llt_utils.h"

#define protected public
#define private public
#include "pass_manager.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_local_context.h"
#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
#include "common/configuration.h"
#include "common/platform_utils.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_store/ops_kernel_manager.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/constants_define.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/quant_host_cpu_op_common.h"

#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace fe;
using namespace std;

namespace fe {

using FEGraphOptimizerPtr = std::shared_ptr<FEGraphOptimizer>;
class GRAPH_FUSION_ST : public testing::Test {
 public:

 protected:

  void SetUp() {
    std::map<std::string, std::string> options;
    options.emplace(ge::PRECISION_MODE, ALLOW_FP32_TO_FP16);
    ge::GetThreadLocalContext().SetGraphOption(options);
    fe_ops_kernel_info_store_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo heavy_op_info{
        6,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/fusion_rule_manager",
        "",
        false,
        false,
        false};

    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(heavy_op_info);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    // initialize fusion rules
    string file_path =
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_parser/cycle_detection.json";
    fe_ops_kernel_info_store_->Initialize(options);
    fusion_rule_mgr_ = std::make_shared<FusionRuleManager>(fe_ops_kernel_info_store_);
    Configuration::Instance(AI_CORE_NAME).ascend_ops_path_ = "";
    Configuration::Instance(AI_CORE_NAME).content_map_[custom_path_key_] = file_path;
    Configuration::Instance(AI_CORE_NAME).content_map_[built_in_path_key_] = file_path;
    Configuration::Instance(AI_CORE_NAME).content_map_[FUSION_CONFIG_BUILT_IN_FILE] =
        "lib64/plugin/opskernel/fusion_pass/config/fusion_config.json";
    fusion_rule_mgr_->Initialize(AI_CORE_NAME);

    fusion_priority_mgr_ =
        std::make_shared<FusionPriorityManager>(AI_CORE_NAME, fusion_rule_mgr_);

    fusion_priority_mgr_vec_ =
        std::make_shared<FusionPriorityManager>(VECTOR_CORE_NAME, fusion_rule_mgr_);

    // initialize fusion configuration
    Configuration::Instance(fe::AI_CORE_NAME).lib_path_ =
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/builtin_config3/";
    ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config3.json";
    fusion_priority_mgr_->fusion_config_parser_ptr_ = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
    fusion_priority_mgr_->fusion_config_parser_ptr_->ParseFusionConfigFile();

    Configuration::Instance(fe::VECTOR_CORE_NAME).lib_path_ =
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/builtin_config3/";
    ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config3.json";
    fusion_priority_mgr_vec_->fusion_config_parser_ptr_ = std::make_unique<FusionConfigParser>(fe::VECTOR_CORE_NAME);
    fusion_priority_mgr_vec_->fusion_config_parser_ptr_->ParseFusionConfigFile();

    graph_fusion_ = std::make_shared<GraphFusion>(
        fusion_rule_mgr_, fe_ops_kernel_info_store_,
        fusion_priority_mgr_);
    graph_fusion_->SetEngineName(AI_CORE_NAME);

    graph_fusion_vec_ = std::make_shared<GraphFusion>(
        fusion_rule_mgr_, fe_ops_kernel_info_store_,
        fusion_priority_mgr_vec_);
    graph_fusion_vec_->SetEngineName(VECTOR_CORE_NAME);

    graph_optimizer_ = std::make_shared<FEGraphOptimizer>(fe_ops_kernel_info_store_);
    graph_optimizer_->init_flag_ = true;
    graph_optimizer_->graph_fusion_ptr_ = graph_fusion_;
    graph_optimizer_->fusion_priority_mgr_ptr_ = fusion_priority_mgr_;
    graph_optimizer_->fusion_rule_mgr_ptr_ = fusion_rule_mgr_;
  }

  void TearDown() {
  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_;
  FusionRuleManagerPtr fusion_rule_mgr_;
  FusionPriorityMgrPtr fusion_priority_mgr_;
  FusionPriorityMgrPtr fusion_priority_mgr_vec_;
  FEGraphOptimizerPtr graph_optimizer_;
  shared_ptr<GraphFusion> graph_fusion_;
  shared_ptr<GraphFusion> graph_fusion_vec_;
  string ori_path_;
  string ori_opp_path_;
  const string custom_path_key_ = "fusionrulemgr.aicore.customfilepath";
  const string built_in_path_key_ = "fusionrulemgr.aicore.graphfilepath";
};

TEST_F(GRAPH_FUSION_ST, converage_03) {
  int32_t priority = CUSTOM_CFG_DOWN_PRIORITY_MIN;
  fusion_priority_mgr_->AdjustDownStagePriority(priority);
  EXPECT_EQ(fusion_priority_mgr_->GetRealPriority(RESERVED_FOR_DOWN_PRIORITY + 1), 1);
}

class TestPass : public PatternFusionBasePass {
 protected:

  vector<FusionPattern *> DefinePatterns() override {
    return {};
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    return SUCCESS;
  }
};

class TestFailedPass : public PatternFusionBasePass {
 protected:

  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("FailedPattern");
    FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatQuatFus][DfnPtn] Fail to new an object."),
             return patterns);

    pattern->AddOpDesc("pattern_dequant", {ASCEND_DEQUANT})
        .SetOutput("pattern_dequant");
    patterns.push_back(pattern);

    return patterns;
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    return FAILED;
  }
};

using CreateFn = GraphPass *(*)();

fe::GraphPass *CreateFunc() {
  return new(std::nothrow) TestPass();
}

fe::GraphPass *CreateFailedFunc() {
  return new(std::nothrow) TestFailedPass();

}

void RegisterPassFunc(CreateFn create_fn) {
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS2", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_GRAPH_PASS, "CUSTOM_PASS3", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS2", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(SECOND_ROUND_BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS3", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(SECOND_ROUND_BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS4", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "BUILT_IN_PASS3", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "BUILT_IN_PASS4", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS2", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_PREPARE_GRAPH_PASS, "PREPARE_PASS3", create_fn, 0);

  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "BEFORE_QUANT_1", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "BEFORE_QUANT_2", create_fn, 0);
}

TEST_F(GRAPH_FUSION_ST, converage_04) {
  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  Configuration::Instance(AI_CORE_NAME).enable_network_analysis_ = true;
  EXPECT_EQ(SUCCESS, graph_fusion_->FusionEachGraph(*graph));
  string stage = "test";
  EXPECT_EQ(SUCCESS, graph_fusion_->RunGraphFusionPassByType(stage, *graph, SECOND_ROUND_BUILT_IN_GRAPH_PASS));
  EXPECT_EQ(SUCCESS, graph_fusion_->RunGraphFusionPassByType(stage, *graph,
                                                             BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS));
  EXPECT_EQ(SUCCESS, graph_fusion_->FusionQuantOp(*graph));
  Configuration::Instance(AI_CORE_NAME).enable_network_analysis_ = false;
}

ge::NodePtr AddOneNode(ge::ComputeGraphPtr &graph,
                       string node_name, const string &node_type) {
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>(node_name, node_type);
  ge::GeTensorDesc tensor_desc(ge::GeShape({10}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  op->AddInputDesc(tensor_desc);
  op->AddOutputDesc(tensor_desc);
  return graph->AddNode(op);
}

int count = 0;
class TestPruningPass : public PatternFusionBasePass {
 protected:

  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("FailedPattern");
    FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatQuatFus][DfnPtn] Fail to new an object."),
             return patterns);

    pattern->AddOpDesc("pattern_dequant", {ASCEND_DEQUANT})
        .SetOutput("pattern_dequant");
    patterns.push_back(pattern);

    return patterns;
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    count++;
    return SUCCESS;
  }
};

class TestPruningFailedPass : public PatternFusionBasePass {
protected:

  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("FailedPattern");
    FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatQuatFus][DfnPtn] Fail to new an object."),
             return patterns);

    pattern->AddOpDesc("pattern_dequant", {ASCEND_DEQUANT})
        .SetOutput("pattern_dequant");
    patterns.push_back(pattern);

    return patterns;
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    count++;
    return FAILED;
  }
};

fe::GraphPass *CreatePruningFunc() {
  return new(std::nothrow) TestPruningPass();
}

fe::GraphPass *CreatePruningFailedFunc() {
  return new(std::nothrow) TestPruningFailedPass();
}

void RegisterPruningPassFunc(CreateFn create_fn) {
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILT_IN_PASS", create_fn, 0);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "PRUNING_PASS", create_fn, 0 | PRUNING);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BEFORE_QUANT", create_fn, 0);
}

TEST_F(GRAPH_FUSION_ST, PruningPassSuccess) {
  RegisterPruningPassFunc(CreatePruningFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");

  auto node1 = AddOneNode(graph, "dequant1", ASCEND_DEQUANT);
  auto node2 = AddOneNode(graph, "relu", RELU);
  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  count= 0;
  EXPECT_EQ(SUCCESS, graph_optimizer_->OptimizeOriginalGraph(*graph));
  EXPECT_EQ(4, count);
}

TEST_F(GRAPH_FUSION_ST, PruningPassFailed) {
  RegisterPruningPassFunc(CreatePruningFailedFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");

  auto node1 = AddOneNode(graph, "dequant1", ASCEND_DEQUANT);
  auto node2 = AddOneNode(graph, "relu", RELU);
  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  count= 0;
  EXPECT_EQ(FAILED, graph_optimizer_->OptimizeOriginalGraph(*graph));
  EXPECT_EQ(1, count);
}

TEST_F(GRAPH_FUSION_ST, converage_05) {
  RegisterPassFunc(CreateFailedFunc);

  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");


  auto node1 = AddOneNode(graph, "dequant1", ASCEND_DEQUANT);
  auto node2 = AddOneNode(graph, "dequant2", ASCEND_DEQUANT);
  auto node3 = AddOneNode(graph, "dequant3", ASCEND_DEQUANT);

  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  EXPECT_EQ(FAILED, graph_fusion_->FusionEachGraph(*graph));
  string stage = "a";
  EXPECT_EQ(FAILED, graph_fusion_->RunGraphFusionPassByType(stage, *graph, CUSTOM_AI_CORE_GRAPH_PASS));
  EXPECT_EQ(FAILED, graph_fusion_->RunGraphFusionPassByType(stage, *graph, SECOND_ROUND_BUILT_IN_GRAPH_PASS));
  EXPECT_EQ(FAILED, graph_fusion_->RunGraphFusionPassByType(stage, *graph, BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS));
  EXPECT_EQ(FAILED, graph_fusion_->RunGraphFusionPassByType(stage, *graph, BUILT_IN_PREPARE_GRAPH_PASS));
  EXPECT_EQ(FAILED, graph_fusion_->RunGraphFusionPassByType(stage, *graph, BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS));
}

TEST_F(GRAPH_FUSION_ST, converage_06) {
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V100);
  EXPECT_EQ(SUCCESS, graph_fusion_->FusionQuantOp(*graph));
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V200);
  EXPECT_EQ(SUCCESS, graph_fusion_->FusionQuantOp(*graph));
}

TEST_F(GRAPH_FUSION_ST, converage_07) {
  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeTensorDesc tensor_desc(ge::GeShape({10}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data", DATA);
  ge::GeTensorPtr const_out_tenosr = nullptr;
  const_out_tenosr = std::make_shared<ge::GeTensor>(tensor_desc);
  vector<uint64_t> scale_data;
  scale_data.emplace_back(0xFF0120312);
  const_out_tenosr->SetData(reinterpret_cast<uint8_t *>(scale_data.data()),
                            sizeof(uint64_t));
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_out_tenosr);
  ge::OpDescPtr dequant = std::make_shared<ge::OpDesc>("dequant", ASCEND_DEQUANT);
  ge::OpDescPtr other = std::make_shared<ge::OpDesc>("other", "Other");

  data->AddOutputDesc(tensor_desc);
  dequant->AddInputDesc(tensor_desc);
  dequant->AddInputDesc(tensor_desc);
  dequant->AddOutputDesc(tensor_desc);
  other->AddInputDesc(tensor_desc);

  auto data_node = graph->AddNode(data);
  auto const_node = graph->AddNode(const_op_desc);
  auto dequant_node = graph->AddNode(dequant);
  auto other_node = graph->AddNode(other);

  ge::OpDescUtils::SetWeights(dequant_node->GetOpDesc(), const_out_tenosr);
  ASSERT_EQ(ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                                    dequant_node->GetInDataAnchor(0)), SUCCESS);
  ASSERT_EQ(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                                    dequant_node->GetInDataAnchor(1)), SUCCESS);
  ASSERT_EQ(ge::GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0),
                                    other_node->GetInDataAnchor(0)), SUCCESS);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V200);
  EXPECT_EQ(fe::SUCCESS, graph_fusion_->JudgeQuantMode(*graph));
}

TEST_F(GRAPH_FUSION_ST, converage_08) {
  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeTensorDesc tensor_desc(ge::GeShape({10, 20}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr dequant = std::make_shared<ge::OpDesc>("dequant", ASCEND_DEQUANT);
  dequant->AddInputDesc(tensor_desc);
  dequant->AddOutputDesc(tensor_desc);

  ge::OpDescPtr other = std::make_shared<ge::OpDesc>("other", "Other");
  other->AddInputDesc(tensor_desc);
  other->AddOutputDesc(tensor_desc);

  auto dequant_node = graph->AddNode(dequant);
  graph->AddNode(other);

  ge::GeTensorPtr const_out_tenosr = nullptr;
  const_out_tenosr = std::make_shared<ge::GeTensor>(tensor_desc);

  vector<uint64_t> data;
  data.emplace_back(0xFF0120312);

  const_out_tenosr->SetData(reinterpret_cast<uint8_t *>(data.data()),
                            sizeof(uint64_t));

  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_out_tenosr);
  auto const_node = graph->AddNode(const_op_desc);
  ge::OpDescUtils::SetWeights(dequant_node->GetOpDesc(), const_out_tenosr);
  ASSERT_EQ(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                                    dequant_node->GetInDataAnchor(0)), SUCCESS);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V200);
  EXPECT_EQ(PARAM_INVALID, graph_fusion_->JudgeQuantMode(*graph));
}

TEST_F(GRAPH_FUSION_ST, converage_10) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  fe::GraphNodeMapUtil::ReCreateNodeTypeMapInGraph(*graph);
  ge::OpDescPtr op = std::make_shared<OpDesc>("test_op", "TestOp");
  auto node = graph->AddNode(op);

  std::map<std::string, ge::NodePtr> inner_map;
  inner_map["test"] = node;
  std::unordered_map<std::string, std::map<std::string, ge::NodePtr>> node_map;
  node_map["test"] = inner_map;

  NodeMapInfoPtr info = std::make_shared<NodeMapInfo>();

  NodeTypeMapPtr node_type_map = std::make_shared<NodeTypeMap>(node_map);
  info->node_type_map = node_type_map;
  graph->SetExtAttr("NodeMapInfo", info);
  EXPECT_EQ(fe::GraphNodeMapUtil::ReCreateNodeTypeMapInGraph(*graph), fe::SUCCESS);
}

TEST_F(GRAPH_FUSION_ST, converage_11) {
  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  Configuration::Instance(AI_CORE_NAME).enable_network_analysis_ = true;
  EXPECT_EQ(SUCCESS, graph_fusion_->FusionEachGraph(*graph));
  string stage = "test";
  EXPECT_EQ(SUCCESS, graph_fusion_->RunGraphFusionPassByType(stage, *graph, BUILT_IN_PREPARE_GRAPH_PASS));
  EXPECT_EQ(SUCCESS, graph_fusion_->RunGraphFusionPassByType(stage, *graph, BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS));
  EXPECT_EQ(SUCCESS, graph_fusion_->FusionQuantOp(*graph));
  Configuration::Instance(AI_CORE_NAME).enable_network_analysis_ = false;

  vector<FusionPassOrRule> graph_fusion_pass_vector;
  fusion_priority_mgr_->GetGraphFusionPassInfosByType(BUILT_IN_PREPARE_GRAPH_PASS, false,
                                                       graph_fusion_pass_vector);
  fusion_priority_mgr_->GetGraphFusionPassInfosByType(BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, false,
                                                      graph_fusion_pass_vector);

  EXPECT_EQ(SUCCESS, graph_optimizer_->OptimizeOriginalGraph(*graph));

  EXPECT_EQ(SUCCESS, graph_optimizer_->GraphFusionBeforeTransnodesInsertion(*graph));
}

TEST_F(GRAPH_FUSION_ST, converage_12) {
  EXPECT_EQ(kStagePrepare, "[GraphOpt][Prepare]");
  EXPECT_EQ(kStageBeforeQuant, "[GraphOpt][BeforeQuant]");
  EXPECT_EQ(kStageOrigin, "[GraphOpt][Origin]");
  EXPECT_EQ(kStageAftFmtSlct, "[GraphOpt][AftFmtSlct]");
  EXPECT_EQ(kStageJudgeInsert, "[GraphOpt][JdgInst]");
  EXPECT_EQ(kStageSetOpSlc, "[SubGraphOpt][SetOpSlc]");
  EXPECT_EQ(kStagePreCompile, "[SubGraphOpt][PreComp]");
  EXPECT_EQ(kStageParseCompRst, "[SubGraphOpt][ParseCompRst]");
  EXPECT_EQ(kStageLx, "[SubGraphOpt][Lx]");
  EXPECT_EQ(kStageCompile, "[SubGraphOpt][Compile]");
}

TEST_F(GRAPH_FUSION_ST, converage_13) {
  std::unordered_set<string> types = {CONSTANT};
  bool matched = false;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("test_op", DATA);
  ge::AttrUtils::SetInt(op, ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  auto node = graph->AddNode(op);
  FeGraphUtils::IsNodeSpecificType(types, node, matched);
  (void)ge::AttrUtils::SetStr(*graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id_100");
  std::string graph_id = "";
  FeGraphUtils::GetGraphIdFromAttr(*graph, graph_id);
  EXPECT_EQ(matched, false);
  EXPECT_EQ(graph_id, "graph_id_100");
}

bool TestAippAttrValue(NamedAttrs &aipp_attr1)
{
  int64_t value = 0;
  AippGetInt64Value(aipp_attr1, "aipp_mode", value);
  EXPECT_EQ(value, 2);
  float var_chn0;
  AippGetFloatVecFirstValue(aipp_attr1, "var_reci_chn_0", var_chn0);
  EXPECT_FLOAT_EQ(var_chn0, 1.123);
  AippGetInt64Value(aipp_attr1, "mean_chn_0", value);
  EXPECT_EQ(value, 3);
  return true;
}
TEST_F(GRAPH_FUSION_ST, converage_14) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("aipp", "Aipp");
  NamedAttrs aipp_attr;
  aipp_attr.SetAttr("aipp_mode", GeAttrValue::CreateFrom<int64_t>(2));
  aipp_attr.SetAttr("support_rotation", GeAttrValue::CreateFrom<int64_t>(1));
  std::vector<float> var_chn = { 1.123, 3.345, 5.589};
  aipp_attr.SetAttr("var_reci_chn_0", GeAttrValue::CreateFrom<std::vector<float>>(var_chn));
  std::vector<int64_t> aipp_mean_vec = { 2, 3, 4, 50};
  AippSetAttrValue(aipp_attr, "mean_chn_0", aipp_mean_vec[3]);
  AippSetAttrValue(aipp_attr, "mean_chn_0", aipp_mean_vec[1]);
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr));
  NamedAttrs aipp_attr1;
  AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr1);
  EXPECT_EQ(TestAippAttrValue(aipp_attr1), true);
}

TEST_F(GRAPH_FUSION_ST, converage_16) {
  std::unordered_set<string> types = {CONSTANT};
  bool matched = false;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr placeholder1 = std::make_shared<OpDesc>("placeholder1", OP_TYPE_PLACE_HOLDER);
  ge::NodePtr pld1 = graph->AddNode(placeholder1);
  ge::OpDescPtr partitioned_node_desc = std::make_shared<OpDesc>("PartitionedCall", "PartitionedCall");
  ge::NodePtr partitioned_node = graph->AddNode(partitioned_node_desc);
  auto sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  sub_graph->SetParentGraph(graph);
  sub_graph->SetParentNode(partitioned_node);
  graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  graph->SetExtAttr("part_src_graph", graph);
  pld1->GetOpDesc()->SetExtAttr(ATTR_NAME_PARENT_NODE, partitioned_node);
  OpDescPtr out = std::make_shared<OpDesc>("out", "NetOutput");
  auto const_op_desc = std::make_shared<OpDesc>("Constant", "Const");
  vector<int64_t> dims_vec = {4};
  GeTensorDesc const_tensor_desc(ge::GeShape(dims_vec), ge::FORMAT_NCHW, ge::DT_INT64);
  const_op_desc->AddOutputDesc(const_tensor_desc);
  GeTensorDesc shape_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT64);
  out->AddInputDesc(shape_tensor_desc);
  auto const_node = sub_graph->AddNode(const_op_desc);
  auto out_node = sub_graph->AddNode(out);
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  (void)ge::AttrUtils::SetInt(pld1->GetOpDesc(), "anchorIndex", 0);
  FeGraphUtils::IsNodeSpecificType(types, pld1, matched);
  EXPECT_EQ(matched, true);
}

TEST_F(GRAPH_FUSION_ST, converage_17) {
  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeTensorDesc tensor_desc(ge::GeShape({10}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data", DATA);
  ge::OpDescPtr dequant = std::make_shared<ge::OpDesc>("dequant", ASCEND_DEQUANT);
  ge::OpDescPtr other = std::make_shared<ge::OpDesc>("other", "Other");

  data->AddOutputDesc(tensor_desc);
  dequant->AddInputDesc(tensor_desc);
  dequant->AddOutputDesc(tensor_desc);
  other->AddInputDesc(tensor_desc);

  auto data_node = graph->AddNode(data);
  auto dequant_node = graph->AddNode(dequant);
  auto other_node = graph->AddNode(other);

  ASSERT_EQ(ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                                    dequant_node->GetInDataAnchor(0)), SUCCESS);
  ASSERT_EQ(ge::GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0),
                                    other_node->GetInDataAnchor(0)), SUCCESS);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V200);
  EXPECT_EQ(fe::FAILED, graph_fusion_->FusionQuantOp(*graph));
}

TEST_F(GRAPH_FUSION_ST, converage_18) {
  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeTensorDesc tensor_desc(ge::GeShape({10}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data", DATA);
  ge::GeTensorPtr const_out_tenosr = nullptr;
  const_out_tenosr = std::make_shared<ge::GeTensor>(tensor_desc);
  vector<uint64_t> scale_data;
  scale_data.emplace_back(0xFF0120312);
  const_out_tenosr->SetData(reinterpret_cast<uint8_t *>(scale_data.data()),
                            sizeof(uint64_t));
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_out_tenosr);
  ge::OpDescPtr dequant = std::make_shared<ge::OpDesc>("dequant", ASCEND_DEQUANT);
  ge::OpDescPtr other = std::make_shared<ge::OpDesc>("other", "Other");

  data->AddOutputDesc(tensor_desc);
  dequant->AddInputDesc(tensor_desc);
  dequant->AddInputDesc(tensor_desc);
  dequant->AddOutputDesc(tensor_desc);
  other->AddInputDesc(tensor_desc);

  auto data_node = graph->AddNode(data);
  auto const_node = graph->AddNode(const_op_desc);
  auto dequant_node = graph->AddNode(dequant);
  auto other_node = graph->AddNode(other);

  ge::OpDescUtils::SetWeights(dequant_node->GetOpDesc(), const_out_tenosr);
  ASSERT_EQ(ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                                    dequant_node->GetInDataAnchor(0)), SUCCESS);
  ASSERT_EQ(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                                    dequant_node->GetInDataAnchor(1)), SUCCESS);
  ASSERT_EQ(ge::GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0),
                                    other_node->GetInDataAnchor(0)), SUCCESS);

  ge::ComputeGraphPtr sub_graph = std::make_shared<ge::ComputeGraph>("sub_graph");
  sub_graph->SetParentNode(dequant_node);
  sub_graph->SetParentGraph(graph);
  ge::OpDescPtr data2 = std::make_shared<ge::OpDesc>("data2", DATA);
  ge::OpDescPtr dequant2 = std::make_shared<ge::OpDesc>("dequant2", ASCEND_DEQUANT);
  ge::OpDescPtr other2 = std::make_shared<ge::OpDesc>("other2", "Other");

  data2->AddOutputDesc(tensor_desc);
  dequant2->AddInputDesc(tensor_desc);
  dequant2->AddOutputDesc(tensor_desc);
  other2->AddInputDesc(tensor_desc);

  auto data_node2 = sub_graph->AddNode(data2);
  auto dequant_node2 = sub_graph->AddNode(dequant2);
  auto other_node2 = sub_graph->AddNode(other2);

  ASSERT_EQ(ge::GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0),
                                    dequant_node2->GetInDataAnchor(0)), SUCCESS);
  ASSERT_EQ(ge::GraphUtils::AddEdge(dequant_node2->GetOutDataAnchor(0),
                                    other_node2->GetInDataAnchor(0)), SUCCESS);
  graph->AddSubgraph("sub_graph", sub_graph);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V200);
  EXPECT_EQ(fe::FAILED, graph_fusion_->FusionQuantOp(*graph));
}

TEST_F(GRAPH_FUSION_ST, converage_19) {
  ge::Format filter_format = ge::FORMAT_NCHW;
  std::vector<int64_t> filter_dims = {1, 1, 1, 1, 1};
  std::vector<int64_t> filter_dims4_d;
  Status ret = PadShapeTo4Dim(filter_format, filter_dims, filter_dims4_d);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(GRAPH_FUSION_ST, converage_20) {
  const std::string engine_name = fe::AI_CORE_NAME;
  const std::string current_dir = "./";
  const std::string json_path = current_dir + "plugin/opskernel/fusion_pass/config/";
  CreateDir(json_path);
  const std::string supportFusionPassFileName = json_path + "support_fusion_pass.json";
  std::string ori_support_fusion_pass_json_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/support_fusion_pass.json";
  std::ifstream ifs1(ori_support_fusion_pass_json_path);
  if (!ifs1.is_open()) {
    printf("open json[%s] failed, %s", ori_support_fusion_pass_json_path.c_str(), strerror(errno));
  }
  nlohmann::json ori_support_fusion_pass_json_value;
  ifs1 >> ori_support_fusion_pass_json_value;
  CreateFileAndFillContent(supportFusionPassFileName, ori_support_fusion_pass_json_value, true);
  Configuration::Instance(engine_name).lib_path_ = current_dir;

  fusion_priority_mgr_->fusion_config_parser_ptr_->ParseSupportFusionPassFile();

  RegisterPassFunc(CreateFunc);
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  size_t hash_key = FusionPriorityManager::GetCurrentHashedKey();
  for (auto pass : fusion_priority_mgr_->sorted_graph_fusion_map_[hash_key]) {
    std::cout << "pass " << pass.name << " is on." << std::endl;
  }

  fusion_priority_mgr_->sorted_graph_fusion_map_.erase(hash_key);
  string soc_version = "Ascend910B";
  PlatformUtils::Instance().short_soc_version_ = soc_version;
  fusion_priority_mgr_->fusion_config_parser_ptr_->ParseSupportFusionPassFile();
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  for (auto pass : fusion_priority_mgr_->sorted_graph_fusion_map_[hash_key]) {
    std::cout << "pass " << pass.name << " is on." << std::endl;
  }
  fusion_priority_mgr_->sorted_graph_fusion_map_.erase(hash_key);
  soc_version = "Ascend910B";
  PlatformUtils::Instance().short_soc_version_ = soc_version;
  fusion_priority_mgr_->fusion_config_parser_ptr_->ParseSupportFusionPassFile();
  EXPECT_EQ(SUCCESS, fusion_priority_mgr_->SortGraphFusion());
  for (auto pass : fusion_priority_mgr_->sorted_graph_fusion_map_[hash_key]) {
    std::cout << "pass " << pass.name << " is on." << std::endl;
  }
  system(("rm -rf " + current_dir + "plugin").c_str());
}
}
