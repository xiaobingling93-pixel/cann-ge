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
#include "common/aicore_util_constants.h"
#define protected public
#define private public
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"
#include "graph/ge_local_context.h"

#include "common/configuration.h"
#include "common/util/op_info_util.h"
#include "fusion_config_manager/fusion_config_parser.h"
#include "register/optimization_option_registry.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace fe;
using FusionConfigParserPtr = std::shared_ptr<FusionConfigParser>;

class UTestFusionConfigParser : public testing::Test {
 protected:
  static void SetUpTestCase() {
    Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/builtin_config/";
  }
  static void TearDownTestCase() {
    Configuration::Instance(fe::AI_CORE_NAME).InitLibPath();
    Configuration::Instance(fe::AI_CORE_NAME).InitAscendOpsPath();
  }
  std::string allStr = "ALL";
  std::string nullStr = "";
};

TEST_F(UTestFusionConfigParser, fusion_switch_01) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config.json";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();
  bool ret = fusionConfigParserPtr->GetFusionSwitchByName("PassThroughFusionPass", "GraphFusion");
  EXPECT_EQ(ret, true);
  bool ret1 = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion");
  EXPECT_EQ(ret1, true);
  bool ret2 = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret2, false);
  bool ret3 = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS2", "GraphFusion");
  bool ret4 = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion");
  EXPECT_EQ(ret4, true);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = nullStr;
}

TEST_F(UTestFusionConfigParser, fusion_switch_02) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config2.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config1.json";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;

  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();
  bool ret = fusionConfigParserPtr->GetFusionSwitchByName("PassThroughFusionPass", "GraphFusion");
  EXPECT_EQ(ret, true);
  bool ret1 = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion");
  EXPECT_EQ(ret1, true);
  ret1 = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS2", "GraphFusion");
  EXPECT_EQ(ret1, true);
  bool ret2 = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret2, false);
  bool ret3 = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS2", "GraphFusion");
  EXPECT_EQ(ret3, true);
  bool ret4 = fusionConfigParserPtr->GetFusionSwitchByName("UB_FUSION_PASS1", "UBFusion");
  EXPECT_EQ(ret4, false);
  ret4 = fusionConfigParserPtr->GetFusionSwitchByName("UB_FUSION_PASS2", "UBFusion");
  EXPECT_EQ(ret4, true);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = nullStr;
}

TEST_F(UTestFusionConfigParser, fusion_switch_03) {
  std::string  allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config3.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config2.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();
  bool ret = fusionConfigParserPtr->GetFusionSwitchByName("PassThroughFusionPass", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS2", "GraphFusion");
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS2", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS3", "UBFusion");
  EXPECT_EQ(ret, false);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = nullStr;
}

TEST_F(UTestFusionConfigParser, fusion_switch_04) {
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config3.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config2.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();
  bool ret = fusionConfigParserPtr->GetFusionSwitchByName("PassThroughFusionPass", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS2", "GraphFusion");
  ret = fusionConfigParserPtr->GetFusionSwitchByName("LayerNormV4FusionPass", "GraphFusion");
  EXPECT_EQ(ret, true);
  
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS2", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS3", "UBFusion");
  EXPECT_EQ(ret, false);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("MatMulBiasAddFusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("OneHotFusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = "3:5";
  Configuration::Instance(fe::AI_CORE_NAME).ParseFusionLicense(true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("MatMulBiasAddFusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("OneHotFusionPass", "UBFusion");
  EXPECT_EQ(ret, false);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = "";
  Configuration::Instance(fe::AI_CORE_NAME).ParseFusionLicense(true);
}

TEST_F(UTestFusionConfigParser, fusion_switch_05) {
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config3.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config2.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();

  bool ret = false;
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion", 1);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion", 4);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion", 4, true);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion", 0, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion", 4, true);
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion", 1);
  EXPECT_EQ(ret, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion", 1);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion", 4);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion", 4, true);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion", 0, true);
  EXPECT_EQ(ret, false);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion", 4, true);
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion", 1);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion", 0x1234F);
  EXPECT_EQ(ret, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS2", "UBFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS2", "UBFusion", 1);
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS2", "UBFusion", 0, true);
  EXPECT_EQ(ret, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS3", "UBFusion", 0x1234E);
  EXPECT_EQ(ret, false);

  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_["UB_PASS4"] = "on";
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS4", "UBFusion");
  EXPECT_EQ(ret, true);
  ge::GetThreadLocalContext().GetOo().working_opt_names_to_value_.clear();

  BufferFusionPassRegistry instance;
  FusionPassRegistry instance2;
  BufferFusionPassRegistry::GetInstance().impl_.swap(instance.impl_);
  FusionPassRegistry::GetInstance().impl_.swap(instance2.impl_);
}

TEST_F(UTestFusionConfigParser, fusion_switch_fail1) {
  Configuration::Instance(fe::AI_CORE_NAME).lib_path_ =
    GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
    GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_switch.cfg";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  EXPECT_EQ(fusionConfigParserPtr->ParseFusionConfigFile(), FAILED);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = nullStr;
}

TEST_F(UTestFusionConfigParser, fusion_switch_fail2) {
  Configuration::Instance(fe::AI_CORE_NAME).lib_path_ =
    GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
    GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_switch1.cfg";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  EXPECT_EQ(fusionConfigParserPtr->ParseFusionConfigFile(), FAILED);
  nlohmann::json custom_fusion_config_json;
  std::map<std::string, std::string> error_key_map;
  std::map<string, bool> old_fusion_switch_map;
  fusionConfigParserPtr->VerifyAndParserCustomFile("./::test", custom_fusion_config_json, old_fusion_switch_map);
  fusionConfigParserPtr->VerifyAndParserCustomFile("test", custom_fusion_config_json, old_fusion_switch_map);
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = nullStr;
}

TEST_F(UTestFusionConfigParser, fusion_switch_fail3) {
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config3.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config4.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  EXPECT_EQ(fusionConfigParserPtr->ParseFusionConfigFile(), FAILED);
}

TEST_F(UTestFusionConfigParser, fusion_switch_06) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
  GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_switch.cfg";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  Status status = fusionConfigParserPtr->ParseFusionConfigFile();
  EXPECT_EQ(status, SUCCESS);
  vector<string> close_graph_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("GraphFusion", false, close_graph_fusion_vec);
  vector<string> close_ub_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("UBFusion", false, close_ub_fusion_vec);
}

TEST_F(UTestFusionConfigParser, fusion_switch_07) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
  GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_switch.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  Status status = fusionConfigParserPtr->ParseFusionConfigFile();
  EXPECT_EQ(status, SUCCESS);
  vector<string> close_graph_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("GraphFusion", false, close_graph_fusion_vec);
  vector<string> close_ub_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("UBFusion", false, close_ub_fusion_vec);
}

class TestCompileLevelPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern1 = new (std::nothrow) FusionPattern("TestGenPattern1");
    FE_CHECK(pattern1 == nullptr, FE_LOGE("New a pattern1 object failed."),  return patterns);
    pattern1->AddOpDesc("TestGen", {"TestGen"})
        .SetOutput("TestGen");
    patterns.push_back(pattern1);
  };

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override {
    return fe::SUCCESS;
  }
};

TEST_F(UTestFusionConfigParser, fusion_switch_case) {
  const std::string engine_name = fe::AI_CORE_NAME;
  const std::string current_dir = "./";
  const std::string json_path = current_dir + "plugin/opskernel/fusion_pass/config/";
  CreateDir(json_path);
  const std::string fileName = json_path + "fusion_config.json";
  std::string ori_json_path = GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/fusion_config.json";
  std::ifstream ifs(ori_json_path);
  if (!ifs.is_open()) {
    printf("open json[%s] failed, %s", ori_json_path.c_str(), strerror(errno));
  }
  ASSERT_TRUE(ifs.is_open());
  nlohmann::json ori_json_value;
  ifs >> ori_json_value;
  CreateFileAndFillContent(fileName, ori_json_value, true);
  Configuration::Instance(engine_name).lib_path_ = current_dir;

  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(engine_name);
  Status status = fusionConfigParserPtr->ParseFusionConfigFile();
  EXPECT_EQ(status, SUCCESS);

  system(("rm -rf " + current_dir + "plugin").c_str());
}