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
#include <fstream>
#include <cstdio>
#include "utils/auto_fuse_config.h"

namespace ge {
namespace autofuse {

class AutoFuseConfigTest : public testing::Test {
 protected:
  void SetUp() override {
    test_config_file_ = "./test_skip_config.ini";
  }
  
  void TearDown() override {
    std::remove(test_config_file_.c_str());
  }
  
  void CreateTestConfigFile(const std::string &content) {
    std::ofstream file(test_config_file_);
    ASSERT_TRUE(file.is_open());
    file << content;
    file.close();
  }
  
  void ParseAndVerifyConfig(const std::string &config_content, 
                         size_t expected_types_size,
                         size_t expected_names_size,
                         const std::vector<std::string> &expected_types = {},
                         const std::vector<std::string> &expected_names = {}) {
    CreateTestConfigFile(config_content);
    
    std::unordered_set<std::string> skip_types;
    std::unordered_set<std::string> skip_names;
    
    ParseSkipNodeNamesConfig(test_config_file_, skip_types, skip_names);
    
    EXPECT_EQ(skip_types.size(), expected_types_size);
    for (const auto &type : expected_types) {
      EXPECT_TRUE(skip_types.count(type));
    }
    
    EXPECT_EQ(skip_names.size(), expected_names_size);
    for (const auto &name : expected_names) {
      EXPECT_TRUE(skip_names.count(name));
    }
  }
  
  std::string test_config_file_;
};

TEST_F(AutoFuseConfigTest, TrimFunction_RemoveLeadingTrailingSpaces) {
  std::string test_str = "  hello world  ";
  Trim(test_str);
  EXPECT_EQ(test_str, "hello world");
}

TEST_F(AutoFuseConfigTest, TrimFunction_RemoveTabs) {
  std::string test_str = "\thello\tworld\t";
  Trim(test_str);
  EXPECT_EQ(test_str, "hello\tworld");
}

TEST_F(AutoFuseConfigTest, TrimFunction_RemoveNewlines) {
  std::string test_str = "\nhello\nworld\n";
  Trim(test_str);
  EXPECT_EQ(test_str, "hello\nworld");
}

TEST_F(AutoFuseConfigTest, TrimFunction_EmptyString) {
  std::string test_str = "";
  Trim(test_str);
  EXPECT_EQ(test_str, "");
}

TEST_F(AutoFuseConfigTest, TrimFunction_OnlySpaces) {
  std::string test_str = "     ";
  Trim(test_str);
  EXPECT_EQ(test_str, "");
}

TEST_F(AutoFuseConfigTest, ParseConfig_BothSections) {
  std::string config_content = 
    "# Test configuration file\n"
    "[ByNodeType]\n"
    "Concat\n"
    "ReduceSum\n"
    "Add\n"
    "\n"
    "[ByNodeName]\n"
    "emb_gather1\n"
    "flash_concat2\n"
    "special_layer_1\n";
  
  ParseAndVerifyConfig(config_content, 3U, 3U, 
                     {"Concat", "ReduceSum", "Add"},
                     {"emb_gather1", "flash_concat2", "special_layer_1"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_OnlyByNodeType) {
  std::string config_content = 
    "# Only node types\n"
    "[ByNodeType]\n"
    "Concat\n"
    "ReduceSum\n"
    "Add\n"
    "Mul\n";
  
  ParseAndVerifyConfig(config_content, 4U, 0U, 
                     {"Concat", "ReduceSum", "Add", "Mul"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_OnlyByNodeName) {
  std::string config_content = 
    "# Only node names\n"
    "[ByNodeName]\n"
    "emb_gather1\n"
    "flash_concat2\n"
    "special_layer_1\n"
    "test_node_42\n";
  
  ParseAndVerifyConfig(config_content, 0U, 4U, {}, 
                     {"emb_gather1", "flash_concat2", "special_layer_1", "test_node_42"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_WithComments) {
  std::string config_content = 
    "# This is a comment\n"
    "[ByNodeType]\n"
    "Concat\n"
    "# Another comment\n"
    "ReduceSum\n"
    "# Skip this: Add\n";
  
  ParseAndVerifyConfig(config_content, 2U, 0U, {"Concat", "ReduceSum"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_WithEmptyLines) {
  std::string config_content = 
    "[ByNodeType]\n"
    "\n"
    "Concat\n"
    "\n"
    "ReduceSum\n"
    "\n"
    "[ByNodeName]\n"
    "\n"
    "test_node\n"
    "\n";
  
  ParseAndVerifyConfig(config_content, 2U, 1U, {"Concat", "ReduceSum"}, {"test_node"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_EmptyFile) {
  std::string config_content = "";
  ParseAndVerifyConfig(config_content, 0U, 0U);
}

TEST_F(AutoFuseConfigTest, ParseConfig_NonexistentFile) {
  std::string nonexistent_file = "./nonexistent_config_file.ini";
  
  std::unordered_set<std::string> skip_types;
  std::unordered_set<std::string> skip_names;
  
  ParseSkipNodeNamesConfig(nonexistent_file, skip_types, skip_names);
  
  EXPECT_EQ(skip_types.size(), 0U);
  EXPECT_EQ(skip_names.size(), 0U);
}

TEST_F(AutoFuseConfigTest, ParseConfig_DuplicateEntries) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "Concat\n"
    "Add\n"
    "Add\n"
    "\n"
    "[ByNodeName]\n"
    "test_node\n"
    "test_node\n";
  
  ParseAndVerifyConfig(config_content, 2U, 1U, {"Concat", "Add"}, {"test_node"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_WithSpacesInNames) {
  std::string config_content = 
    "[ByNodeType]\n"
    "  Concat  \n"
    "  ReduceSum  \n"
    "\n"
    "[ByNodeName]\n"
    "  test_node_1  \n"
    "  test_node_2  \n";
  
  ParseAndVerifyConfig(config_content, 2U, 2U, {"Concat", "ReduceSum"}, {"test_node_1", "test_node_2"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_InvalidSection) {
  std::string config_content = 
    "[InvalidSection]\n"
    "SomeNode\n"
    "\n"
    "[ByNodeType]\n"
    "Concat\n\n"
    "[ByNodeName]\n"
    "test_node\n";
  
  ParseAndVerifyConfig(config_content, 1U, 1U, {"Concat"}, {"test_node"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_MixedCaseSection) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "\n"
    "[UnknownSection]\n"
    "UnknownNode\n"
    "\n"
    "[ByNodeName]\n"
    "test_node\n";
  
  ParseAndVerifyConfig(config_content, 1U, 1U, {"Concat"}, {"test_node"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_LongNodeNames) {
  std::string config_content = 
    "[ByNodeName]\n"
    "very_long_node_name_with_many_characters_that_should_still_work_correctly\n"
    "another_extremely_long_node_name_for_testing_purposes_123456789\n";
  
  ParseAndVerifyConfig(config_content, 0U, 2U, {}, 
                     {"very_long_node_name_with_many_characters_that_should_still_work_correctly",
                      "another_extremely_long_node_name_for_testing_purposes_123456789"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_SpecialCharacters) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "Add_v2\n"
    "Reduce-Sum\n"
    "Test.Node\n";
  
  ParseAndVerifyConfig(config_content, 4U, 0U, {"Concat", "Add_v2", "Reduce-Sum", "Test.Node"});
}

TEST_F(AutoFuseConfigTest, ParseConfig_MultipleSections) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "\n"
    "[ByNodeType]\n"
    "Add\n"
    "\n"
    "[ByNodeName]\n"
    "test_node\n"
    "\n"
    "[ByNodeName]\n"
    "another_node\n";
  
  CreateTestConfigFile(config_content);
  
  std::unordered_set<std::string> skip_types;
  std::unordered_set<std::string> skip_names;
  
  ParseSkipNodeNamesConfig(test_config_file_, skip_types, skip_names);
  
  EXPECT_EQ(skip_types.size(), 2U);
  EXPECT_TRUE(skip_types.count("Concat"));
  EXPECT_TRUE(skip_types.count("Add"));
  
  EXPECT_EQ(skip_names.size(), 2U);
  EXPECT_TRUE(skip_names.count("test_node"));
  EXPECT_TRUE(skip_names.count("another_node"));
}

TEST_F(AutoFuseConfigTest, AutoFuseConfig_SkipNodeTypes) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "ReduceSum\n"
    "Add\n";
  
  CreateTestConfigFile(config_content);
  
  auto &mutable_config = AutoFuseConfig::MutableConfig();
  auto &mutable_lowering_config = mutable_config.MutableLoweringConfig();
  
  mutable_lowering_config.skip_node_types.clear();
  mutable_lowering_config.skip_node_names.clear();
  
  ParseSkipNodeNamesConfig(test_config_file_, mutable_lowering_config.skip_node_types, 
                             mutable_lowering_config.skip_node_names);
  
  const auto &config = AutoFuseConfig::Config();
  const auto &lowering_config = config.LoweringConfig();
  
  EXPECT_EQ(lowering_config.skip_node_types.size(), 3U);
  EXPECT_TRUE(lowering_config.skip_node_types.count("Concat"));
  EXPECT_TRUE(lowering_config.skip_node_types.count("ReduceSum"));
  EXPECT_TRUE(lowering_config.skip_node_types.count("Add"));
  
  EXPECT_EQ(lowering_config.skip_node_names.size(), 0U);
}

TEST_F(AutoFuseConfigTest, AutoFuseConfig_SkipNodeNames) {
  std::string config_content = 
    "[ByNodeName]\n"
    "emb_gather1\n"
    "flash_concat2\n"
    "special_layer_1\n";
  
  CreateTestConfigFile(config_content);
  
  auto &mutable_config = AutoFuseConfig::MutableConfig();
  auto &mutable_lowering_config = mutable_config.MutableLoweringConfig();
  
  mutable_lowering_config.skip_node_types.clear();
  mutable_lowering_config.skip_node_names.clear();
  
  ParseSkipNodeNamesConfig(test_config_file_, mutable_lowering_config.skip_node_types, 
                             mutable_lowering_config.skip_node_names);
  
  const auto &config = AutoFuseConfig::Config();
  const auto &lowering_config = config.LoweringConfig();
  
  EXPECT_EQ(lowering_config.skip_node_types.size(), 0U);
  
  EXPECT_EQ(lowering_config.skip_node_names.size(), 3U);
  EXPECT_TRUE(lowering_config.skip_node_names.count("emb_gather1"));
  EXPECT_TRUE(lowering_config.skip_node_names.count("flash_concat2"));
  EXPECT_TRUE(lowering_config.skip_node_names.count("special_layer_1"));
}

TEST_F(AutoFuseConfigTest, AutoFuseConfig_MixedSkipConfig) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "Add\n"
    "\n"
    "[ByNodeName]\n"
    "test_node_1\n"
    "test_node_2\n";
  
  CreateTestConfigFile(config_content);
  
  auto &mutable_config = AutoFuseConfig::MutableConfig();
  auto &mutable_lowering_config = mutable_config.MutableLoweringConfig();
  
  mutable_lowering_config.skip_node_types.clear();
  mutable_lowering_config.skip_node_names.clear();
  
  ParseSkipNodeNamesConfig(test_config_file_, mutable_lowering_config.skip_node_types, 
                             mutable_lowering_config.skip_node_names);
  
  const auto &config = AutoFuseConfig::Config();
  const auto &lowering_config = config.LoweringConfig();
  
  EXPECT_EQ(lowering_config.skip_node_types.size(), 2U);
  EXPECT_TRUE(lowering_config.skip_node_types.count("Concat"));
  EXPECT_TRUE(lowering_config.skip_node_types.count("Add"));
  
  EXPECT_EQ(lowering_config.skip_node_names.size(), 2U);
  EXPECT_TRUE(lowering_config.skip_node_names.count("test_node_1"));
  EXPECT_TRUE(lowering_config.skip_node_names.count("test_node_2"));
}

TEST_F(AutoFuseConfigTest, AutoFuseConfig_ClearSkipConfig) {
  std::string config_content = 
    "[ByNodeType]\n"
    "Concat\n"
    "\n"
    "[ByNodeName]\n"
    "test_node\n";
  
  CreateTestConfigFile(config_content);
  
  auto &mutable_config = AutoFuseConfig::MutableConfig();
  auto &mutable_lowering_config = mutable_config.MutableLoweringConfig();
  
  mutable_lowering_config.skip_node_types.clear();
  mutable_lowering_config.skip_node_names.clear();
  
  ParseSkipNodeNamesConfig(test_config_file_, mutable_lowering_config.skip_node_types, 
                             mutable_lowering_config.skip_node_names);
  
  EXPECT_EQ(mutable_lowering_config.skip_node_types.size(), 1U);
  EXPECT_EQ(mutable_lowering_config.skip_node_names.size(), 1U);
  
  mutable_lowering_config.skip_node_types.clear();
  mutable_lowering_config.skip_node_names.clear();
  
  EXPECT_EQ(mutable_lowering_config.skip_node_types.size(), 0U);
  EXPECT_EQ(mutable_lowering_config.skip_node_names.size(), 0U);
}

} // namespace autofuse
} // namespace ge