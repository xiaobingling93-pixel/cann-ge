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
#include "register/optimization_option_registry.h"
#include "common/configuration.h"
#include "common/util/op_info_util.h"
#include "fusion_config_manager/fusion_config_parser.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace fe;
using FusionConfigParserPtr = std::shared_ptr<FusionConfigParser>;
class STestFusionConfigParser : public testing::Test {
 protected:
  static void SetUpTestCase() {
    Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
            GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
    Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = "ALL";
  }
  static void TearDownTestCase() {
    Configuration::Instance(fe::AI_CORE_NAME).InitLibPath();
    Configuration::Instance(fe::AI_CORE_NAME).InitAscendOpsPath();
  }
};

TEST_F(STestFusionConfigParser, fusion_switch_01) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config.json";
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
}

TEST_F(STestFusionConfigParser, fusion_switch_02) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config2.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config1.json";
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
}

TEST_F(STestFusionConfigParser, fusion_switch_03) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config3.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config2.json";
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
}

TEST_F(STestFusionConfigParser, fusion_switch_04) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config2.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config1.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();
  bool ret = fusionConfigParserPtr->GetFusionSwitchByName("PassThroughFusionPass", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret, false);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS2", "GraphFusion");
  EXPECT_EQ(ret, true);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion");
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion");
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS2", "UBFusion");
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS3", "UBFusion");
}

TEST_F(STestFusionConfigParser, fusion_switch_case) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config2.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config1.json";

  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  Status status = fusionConfigParserPtr->ParseFusionConfigFile();
  EXPECT_EQ(status, SUCCESS);
  vector<string> close_graph_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("GraphFusion", false, close_graph_fusion_vec);
  vector<string> close_ub_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("UBFusion", false, close_ub_fusion_vec);
}

TEST_F(STestFusionConfigParser, fusion_switch_fail1) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_switch.cfg";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  EXPECT_EQ(fusionConfigParserPtr->ParseFusionConfigFile(), FAILED);
}

TEST_F(STestFusionConfigParser, fusion_switch_fail2) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config1.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_switch1.cfg";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();
  nlohmann::json custom_fusion_config_json;
  std::map<std::string, std::string> error_key_map;
  std::map<string, bool> old_fusion_switch_map;
  EXPECT_NE(
      fusionConfigParserPtr->VerifyAndParserCustomFile("./::test", custom_fusion_config_json, old_fusion_switch_map),
      SUCCESS);
  EXPECT_NE(fusionConfigParserPtr->VerifyAndParserCustomFile("test", custom_fusion_config_json, old_fusion_switch_map),
            SUCCESS);
}

TEST_F(STestFusionConfigParser, fusion_switch_fail3) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config3.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config4.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  EXPECT_EQ(fusionConfigParserPtr->ParseFusionConfigFile(), FAILED);
}

TEST_F(STestFusionConfigParser, fusion_switch_05) {
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config2.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusionConfigParserPtr->ParseFusionConfigFile();

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "CUSTOM_PASS1", nullptr, 2);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "CUSTOM_PASS2", nullptr, 2);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "TbeCommonRules2FusionPass", nullptr, 2);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "UB_PASS1", nullptr, 2);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "UB_PASS2", nullptr, 2);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "UB_PASS3", nullptr, 2);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_GRAPH_PASS, "BUILTIN_PASS1", nullptr, 2);
  Status ret = fusionConfigParserPtr->GetFusionSwitchByName("BUILTIN_PASS1", "GraphFusion", 0);
  EXPECT_EQ(ret, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion");
  EXPECT_EQ(ret, false);
  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_VECTOR_CORE_GRAPH_PASS, "CUSTOM_PASS1", nullptr, 2);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion", 2);
  EXPECT_EQ(ret, false);

  FusionPassRegistry::GetInstance().RegisterPass(BUILT_IN_VECTOR_CORE_GRAPH_PASS, "CUSTOM_PASS1", nullptr, 1);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("CUSTOM_PASS1", "GraphFusion", 1);
  EXPECT_EQ(ret, true);


  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion");
  EXPECT_EQ(ret, true);
  BufferFusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_BUFFER_FUSION_PASS,
                                                       "TbeCommonRules2FusionPass", nullptr, 1);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("TbeCommonRules2FusionPass", "UBFusion", 1);
  EXPECT_EQ(ret, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion");
  EXPECT_EQ(ret, false);
  BufferFusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "UB_PASS1", nullptr, 0x1234F);
  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS1", "UBFusion", 0x1234F);
  EXPECT_EQ(ret, true);

  ret = fusionConfigParserPtr->GetFusionSwitchByName("UB_PASS3", "UBFusion", 2);
  EXPECT_EQ(ret, false);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "UB_PASS3", nullptr, 0x1234E);
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

TEST_F(STestFusionConfigParser, fusion_switch_06) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
    GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_switch2.cfg";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  Status status = fusionConfigParserPtr->ParseFusionConfigFile();
  EXPECT_EQ(status, SUCCESS);
  vector<string> close_graph_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("GraphFusion", false, close_graph_fusion_vec);
  vector<string> close_ub_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("UBFusion", false, close_ub_fusion_vec);
}

TEST_F(STestFusionConfigParser, fusion_switch_07) {
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config.json";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
    GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/custom_config/fusion_config3.json";
  FusionConfigParserPtr fusionConfigParserPtr = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  Status status = fusionConfigParserPtr->ParseFusionConfigFile();
  EXPECT_EQ(status, SUCCESS);
  vector<string> close_graph_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("GraphFusion", false, close_graph_fusion_vec);
  vector<string> close_ub_fusion_vec;
  fusionConfigParserPtr->GetFusionPassNameBySwitch("UBFusion", false, close_ub_fusion_vec);
}
