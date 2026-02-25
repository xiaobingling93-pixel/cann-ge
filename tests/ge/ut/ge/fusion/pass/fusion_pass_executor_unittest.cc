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
#include "common/share_graph.h"
#include "es_ge_test_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "nlohmann/json.hpp"

#include "stub/gert_runtime_stub.h"
#include "ge/fusion/pattern.h"
#include "common/topo_checker.h"
#include "ge/fusion/pass/fusion_pass_reg.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "register/custom_pass_context_impl.h"
#include "graph/ge_local_context.h"
#include "compiler/graph/fusion/pass/fusion_pass_executor.h"
#include "register/optimization_option_registry.h"

// 单例，为了保证ut效果，需要清理其成员
#define private public
#include "compiler/graph/fusion/pass/pass_registry.h"
#include "ge/fusion/pass/decompose_pass.h"
#undef private

namespace ge {
namespace fusion {
namespace {
std::string GetCodeDir() {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  return current_path;
}
} // namespace
using namespace ge::es;
class UtestFusionPassExecutor : public testing::Test {
 public:
  void SetUp() override {
    PassRegistry::GetInstance().name_2_fusion_pass_regs_.clear();
    global_options_bak_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    session_options_bak_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    graph_options_bak_ = ge::GetThreadLocalContext().GetAllGraphOptions();
  }
  void TearDown() override {
    PassRegistry::GetInstance().name_2_fusion_pass_regs_.clear();

    GetThreadLocalContext().SetGlobalOption(global_options_bak_);
    GetThreadLocalContext().SetSessionOption(session_options_bak_);
    GetThreadLocalContext().SetGraphOption(graph_options_bak_);
    GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
  }
 private:
  std::map<std::string, std::string> global_options_bak_;
  std::map<std::string, std::string> graph_options_bak_;
  std::map<std::string, std::string> session_options_bak_;
};
/**
 * single node match
 *      data
 *        |
 *     transdata
 *        |
 *       out
 */
TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {"Relu"}, {"Relu"}, {CONSTANT}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(1, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(2, {{"Relu"}}), "success");
    }
    if (node->GetName() == "x_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
    }
    if (node->GetName() == "y_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}}), "success");
    }
    if (node->GetType() == "Relu") {
      std::cout << node->GetName() << std::endl;
      // check attr
      std::vector<std::string> origin_types;
      const bool has_origin_op_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_types);
      EXPECT_TRUE(has_origin_op_attr);
      EXPECT_TRUE(origin_types.size() == 1);
      EXPECT_STREQ(origin_types[0].c_str(), TRANSDATA);
    }
  }
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
}
TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_OptionToOff) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(pattern.release());
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  GetThreadLocalContext().GetOo().Initialize({{ge::OPTIMIZATION_SWITCH, "TransDataToReluPass:off"}},
                                            OptionRegistry::GetInstance().GetRegisteredOptTable());
  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  bool has_relu = false;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Relu") {
      has_relu = true;
    }
  }
  EXPECT_FALSE(has_relu); // 图没有被改
}
TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_ConfigToOff) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(pattern.release());
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  std::string fusion_config_json_str = "{\n"
      "      \"Switch\":{\n"
      "          \"GraphFusion\":{\n"
      "            \"TransDataToReluPass\" : \"off\"\n"
      "          },\n"
      "          \"UBFusion\":{\n"
      "          }\n"
      "      }}";
  std::ofstream json_file("./fusion_switch_config.json");
  json_file << fusion_config_json_str << std::endl;

  // with option config
  std::string config_file_path = GetCodeDir() + "/fusion_switch_config.json";
  GetThreadLocalContext().SetGlobalOption({{FUSION_SWITCH_FILE, config_file_path}});

  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  bool has_relu = false;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Relu") {
      has_relu = true;
    }
  }
  EXPECT_FALSE(has_relu); // 图没有被改
  remove("./fusion_switch_config.json");
}

TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_ConfigAllOff_PassToOn) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(pattern.release());
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  std::string fusion_config_json_str = "{\n"
      "      \"Switch\":{\n"
      "          \"GraphFusion\":{\n"
      "            \"TransDataToReluPass\" : \"on\"\n"
      "            \"ALL\" : \"off\"\n"
      "          },\n"
      "          \"UBFusion\":{\n"
      "          }\n"
      "      }}";
  std::ofstream json_file("./fusion_switch_config.json");
  json_file << fusion_config_json_str << std::endl;

  // with option config
  std::string config_file_path = GetCodeDir() + "/fusion_switch_config.json";
  GetThreadLocalContext().SetGlobalOption({{FUSION_SWITCH_FILE, config_file_path}});

  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  bool has_relu = false;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Relu") {
      has_relu = true;
    }
  }
  EXPECT_TRUE(has_relu); // 图被改
  remove("./fusion_switch_config.json");
}

TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_ConfigAllOff) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(pattern.release());
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  std::string fusion_config_json_str = "{\n"
      "      \"Switch\":{\n"
      "          \"GraphFusion\":{\n"
      "            \"ALL\" : \"off\"\n"
      "          },\n"
      "          \"UBFusion\":{\n"
      "          }\n"
      "      }}";
  std::ofstream json_file("./fusion_switch_config.json");
  json_file << fusion_config_json_str << std::endl;

  // with option config
  std::string config_file_path = GetCodeDir() + "/fusion_switch_config.json";
  GetThreadLocalContext().SetGlobalOption({{FUSION_SWITCH_FILE, config_file_path}});

  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  bool has_relu = false;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "Relu") {
      has_relu = true;
    }
  }
  EXPECT_FALSE(has_relu); // 图没有被改
  remove("./fusion_switch_config.json");
}

/**
 * shape to relu
 */
TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_WithSubgraph) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto shape = EsShape(data, DT_INT64);
      esb_graph->SetGraphOutput(shape, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::IfGraph2();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  NodePtr relu_node = nullptr;
  for (const auto &node : target_compute_graph->GetAllNodes()) {
    if (node->GetType() == "Relu") {
      relu_node = node;
    }
  }
  EXPECT_NE(relu_node, nullptr);
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
}

TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_WithSubgraph_Failed) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto shape = EsShape(data, DT_INT64);
      esb_graph->SetGraphOutput(shape, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override { // invalid replacement
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto data1 = EsCreateGraphInput(esb_graph, 1);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      esb_graph->SetGraphOutput(data1, 1);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::IfGraph2();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  FusionPassExecutor pass_executor;
  EXPECT_NE(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  EXPECT_NE(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), NOT_CHANGED);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  NodePtr relu_node = nullptr;
  for (const auto &node : target_compute_graph->GetAllNodes()) {
    if (node->GetType() == "Relu") {
      relu_node = node;
    }
  }
  EXPECT_EQ(relu_node, nullptr);
}

TEST_F(UtestFusionPassExecutor, PatternFusionPassReg_Run_BeforeInferShape_FAILED) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override { // invalid replacement
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto data1 = EsCreateGraphInput(esb_graph, 1);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      esb_graph->SetGraphOutput(data1, 1);
      return replace_graph.BuildAndReset();
    }
  };
  REG_FUSION_PASS(TransDataToReluPass).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  FusionPassExecutor pass_executor;
  auto ret = pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_NE(ret, NOT_CHANGED);
}

/**
 * single node match
 *      data
 *        |
 *     transdata
 *        |
 *       out
 */
TEST_F(UtestFusionPassExecutor, DecomposePass_Run_AfterInferShape) {
  // define pass
  class RunDecomposeTransDataPass : public DecomposePass {
   public:
    RunDecomposeTransDataPass(const std::vector<AscendString> &op_types) : DecomposePass(op_types) {}
   protected:
    bool MeetRequirements(const GNode &matched_node) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const GNode &matched_node) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };
  REG_DECOMPOSE_PASS(RunDecomposeTransDataPass, {"TransData"}).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  FusionPassExecutor pass_executor;
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {"Relu"}, {"Relu"}, {CONSTANT}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(1, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(2, {{"Relu"}}), "success");
    }
    if (node->GetName() == "x_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
    }
    if (node->GetName() == "y_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}}), "success");
    }
    if (node->GetType() == "Relu") {
      std::cout << node->GetName() << std::endl;
      // check attr
      std::vector<std::string> origin_types;
      const bool has_origin_op_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_types);
      EXPECT_TRUE(has_origin_op_attr);
      EXPECT_TRUE(origin_types.size() == 1);
      EXPECT_STREQ(origin_types[0].c_str(), TRANSDATA);
    }
  }
  EXPECT_EQ(pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape), SUCCESS);
}

TEST_F(UtestFusionPassExecutor, DecomposePass_Run_AfterInferShape_Failed) {
  // define pass
  class RunDecomposeTransDataPass : public DecomposePass {
   public:
    RunDecomposeTransDataPass(const std::vector<AscendString> &op_type) : DecomposePass(op_type) {}
   protected:
    bool MeetRequirements(const GNode &matched_node) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const GNode &matched_node) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto data1 = EsCreateGraphInput(esb_graph, 1);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      esb_graph->SetGraphOutput(data1, 1);
      return replace_graph.BuildAndReset();
    }
  };
  REG_DECOMPOSE_PASS(RunDecomposeTransDataPass, {"TransData"}).Stage(CustomPassStage::kAfterInferShape);

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  FusionPassExecutor pass_executor;
  auto ret = pass_executor.RunPasses(target_compute_graph, CustomPassStage::kAfterInferShape);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_NE(ret, NOT_CHANGED);
}
} // namespace fusion
} // namespace ge