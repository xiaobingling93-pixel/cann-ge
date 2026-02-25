/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history/history_registry_interface.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "history/history_registry_reader.h"

using namespace ge::es::history;

namespace {
void EnsureDir(const std::string &path) {
  std::string cmd = "mkdir -p " + path;
  (void)std::system(cmd.c_str());
}
}  // namespace

class HistoryRegistryInterfaceUT : public ::testing::Test {
 protected:
  void SetUp() override {
    fixture_root_ = "./fixtures/history_registry/math/";
  }

  std::string fixture_root_;
};

TEST_F(HistoryRegistryInterfaceUT, LoadHistoryChainReturnsMatchingVersionsAndProtos) {
  std::vector<VersionMeta> all = HistoryRegistryReader::LoadIndex(fixture_root_);
  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "8.0.RC2");
  std::vector<std::string> warnings;
  HistoryContext ctx = LoadHistoryChain(fixture_root_, window, "phony_1i_1o", warnings);
  EXPECT_TRUE(warnings.empty());
  ASSERT_EQ(ctx.versions.size(), ctx.proto_chain.size());
  EXPECT_EQ(ctx.versions.size(), 2U);

  EXPECT_EQ(ctx.versions[0].release_version, "8.0.RC1");
  EXPECT_EQ(ctx.versions[0].release_date, "2024-09-30");
  EXPECT_EQ(ctx.proto_chain[0].op_type, "phony_1i_1o");
  EXPECT_EQ(ctx.proto_chain[0].inputs.size(), 1U);

  EXPECT_EQ(ctx.versions[1].release_version, "8.0.RC2");
  EXPECT_EQ(ctx.versions[1].release_date, "2025-01-15");
  EXPECT_EQ(ctx.proto_chain[1].op_type, "phony_1i_1o");
  EXPECT_EQ(ctx.proto_chain[1].inputs.size(), 2U);
}

TEST_F(HistoryRegistryInterfaceUT, LoadHistoryChainPassesVersionMetaDirectly) {
  std::vector<VersionMeta> all = HistoryRegistryReader::LoadIndex(fixture_root_);
  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "8.0.RC1");
  ASSERT_FALSE(window.empty());

  std::vector<std::string> warnings;
  HistoryContext ctx = LoadHistoryChain(fixture_root_, window, "phony_1i_1o", warnings);
  EXPECT_TRUE(warnings.empty());
  ASSERT_EQ(ctx.versions.size(), 1U);
  EXPECT_EQ(ctx.versions[0].release_version, "8.0.RC1");
  EXPECT_EQ(ctx.versions[0].release_date, "2024-09-30");
  EXPECT_EQ(ctx.versions[0].branch_name, "master");
}

TEST_F(HistoryRegistryInterfaceUT, LoadHistoryChainEmptyWindowReturnsEmpty) {
  std::vector<VersionMeta> window;
  std::vector<std::string> warnings;
  HistoryContext ctx = LoadHistoryChain(fixture_root_, window, "phony_1i_1o", warnings);
  EXPECT_TRUE(ctx.versions.empty());
  EXPECT_TRUE(ctx.proto_chain.empty());
  EXPECT_TRUE(warnings.empty());
}

TEST_F(HistoryRegistryInterfaceUT, LoadHistoryChainOpNotFoundAddsWarning) {
  std::vector<VersionMeta> all = HistoryRegistryReader::LoadIndex(fixture_root_);
  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "8.0.RC1");
  ASSERT_EQ(window.size(), 1U);

  std::vector<std::string> warnings;
  HistoryContext ctx = LoadHistoryChain(fixture_root_, window, "NonExistentOp", warnings);
  EXPECT_TRUE(ctx.versions.empty());
  EXPECT_TRUE(ctx.proto_chain.empty());
  ASSERT_EQ(warnings.size(), 1U);
  EXPECT_NE(warnings[0].find("op NonExistentOp skip version 8.0.RC1: Op NonExistentOp not found in operators.json"), std::string::npos);
}

TEST_F(HistoryRegistryInterfaceUT, LoadHistoryChainSkipsInvalidVersion) {
  std::string pkg_dir = "./load_history_chain_partial_" + std::to_string(getpid()) + "/";
  EnsureDir(pkg_dir);
  EnsureDir(pkg_dir + "registry/8.0.RC1/");
  EnsureDir(pkg_dir + "registry/8.0.RC2/");
  std::ofstream(pkg_dir + "index.json") << R"({"version":"1.0.0","releases":[
    {"release_version":"8.0.RC1","release_date":"2024-09-30"},
    {"release_version":"8.0.RC2","release_date":"2025-01-15"}
  ]})";
  std::ofstream(pkg_dir + "registry/8.0.RC1/metadata.json") << R"({"release_version":"8.0.RC1"})";
  std::ofstream(pkg_dir + "registry/8.0.RC1/operators.json")
      << R"({"operators":[{"op_type":"phony_1i_1o","inputs":[],"outputs":[],"attrs":[],"subgraphs":[]}]})";
  std::ofstream(pkg_dir + "registry/8.0.RC2/metadata.json") << R"({"release_version":"8.0.RC2"})";
  std::ofstream(pkg_dir + "registry/8.0.RC2/operators.json") << "invalid json";

  std::vector<VersionMeta> all = HistoryRegistryReader::LoadIndex(pkg_dir);
  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "8.0.RC2");
  ASSERT_EQ(window.size(), 2U);

  std::vector<std::string> warnings;
  HistoryContext ctx = LoadHistoryChain(pkg_dir, window, "phony_1i_1o", warnings);
  ASSERT_EQ(warnings.size(), 1U);
  ASSERT_EQ(ctx.versions.size(), 1U);
  EXPECT_EQ(ctx.versions.size(), ctx.proto_chain.size());
  EXPECT_EQ(ctx.versions[0].release_version, "8.0.RC1");
  EXPECT_EQ(ctx.proto_chain[0].op_type, "phony_1i_1o");

  (void)std::system(("rm -rf " + pkg_dir).c_str());
}
