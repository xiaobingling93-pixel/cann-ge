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
#include <string>
#include <unordered_map>
#define private public
#include "utils/auto_fuse_config.h"
#undef private

namespace ge {
namespace autofuse {

class AutoFuseConfigTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }
};

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigValidNumber) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "100";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 100U);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 100U);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigValidWithPunctuation) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "1,000";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 1U);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 1U);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigInvalidCharacters) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "abc";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigMixedInvalidCharacters) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "100abc";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigZero) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "0";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 0U);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 0U);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigMaxUint64) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "18446744073709551615";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 18446744073709551615ULL);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 18446744073709551615ULL);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigNoFlag) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--other_flag"] = "100";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigEmptyMap) {
  std::unordered_map<std::string, std::string> all_flags;
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigNegativeNumber) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "-100";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigFloatNumber) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "1.5";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigMultipleCommas) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "1,000,000";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 1U);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 1U);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigNumberWithSemicolon) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "1000;";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 1000U);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 1000U);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigNumberWithComma) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "1000,";
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, 1000U);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, 1000U);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigNumberWithDot) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "1000.";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

TEST_F(AutoFuseConfigTest, UpdateMaxFusionSizeConfigEmptyString) {
  std::unordered_map<std::string, std::string> all_flags;
  all_flags["--max_fusion_size"] = "";
  uint64_t original_fusion_size = AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size;
  uint64_t original_loop_ops = AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops;
  AutoFuseConfig::Instance().UpdateMaxFusionSizeConfig(all_flags);
  EXPECT_EQ(AutoFuseConfig::Instance().fusion_strategy_solver_.max_fusion_size, original_fusion_size);
  EXPECT_EQ(AutoFuseConfig::Instance().lowering_strategy_config_.max_fused_loop_ops, original_loop_ops);
}

}  // namespace autofuse
}  // namespace ge