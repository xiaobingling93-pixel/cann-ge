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
#include <memory>
#include <fstream>
#include "nlohmann/json.hpp"
#include "mmpa_api.h"
#include "graph/fusion/fusion_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/ge_local_context.h"

namespace ge {
namespace fusion {
namespace {
std::string GetCodeDir() {
  ge::char_t current_path[MMPA_MAX_PATH] = {'\0'};
  getcwd(current_path, MMPA_MAX_PATH);
  return current_path;
}
}
class UtestFusionUtils : public testing::Test {
 protected:
  void SetUp() {
    global_options_bak_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
  }

  void TearDown() {
    GetThreadLocalContext().SetGlobalOption(global_options_bak_);
  }
 private:
  std::map<std::string, std::string> global_options_bak_;
};

TEST_F(UtestFusionUtils, GetFusionSwitchFilePathFromOption) {
  // No option config
  EXPECT_TRUE(FusionUtils::GetFusionSwitchFileFromOption().empty());

  // with option config
  std::string config_file_path = "./fusion_config_file/fusion_config.json";
  GetThreadLocalContext().SetGlobalOption({{FUSION_SWITCH_FILE, config_file_path}});
  EXPECT_STREQ(FusionUtils::GetFusionSwitchFileFromOption().c_str(), config_file_path.c_str());
}

TEST_F(UtestFusionUtils, ParseFusionSwitch) {
  // No option config
  EXPECT_TRUE(FusionUtils::ParseFusionSwitch().empty());
  std::string json_str = "{\n"
      "      \"Switch\": {\n"
      "          \"GraphFusion\": {\n"
      "            \"CUSTOM_PASS1\" : \"off\",\n"
      "            \"CUSTOM_PASS2\" : \"on\",\n"
      "            \"CUSTOM_PASS3\" : \"off\"\n"
      "          }\n"
      "      }\n"
      "  }";
  std::ofstream json_file("./fusion_switch_config.json");
  json_file << json_str << std::endl;

  // with option config
  std::string config_file_path = GetCodeDir() + "/fusion_switch_config.json";
  std::cout << "cur path " << config_file_path << std::endl;
  GetThreadLocalContext().SetGlobalOption({{FUSION_SWITCH_FILE, config_file_path}});
  auto pass_name_2_switches = FusionUtils::ParseFusionSwitch();
  EXPECT_EQ(pass_name_2_switches.size(), 3);
  EXPECT_EQ(pass_name_2_switches["CUSTOM_PASS1"], false);
  EXPECT_EQ(pass_name_2_switches["CUSTOM_PASS2"], true);
  EXPECT_EQ(pass_name_2_switches["CUSTOM_PASS3"], false);
  remove("./fusion_switch_config.json");
}
} // namespace fusion
} // namespace ge

