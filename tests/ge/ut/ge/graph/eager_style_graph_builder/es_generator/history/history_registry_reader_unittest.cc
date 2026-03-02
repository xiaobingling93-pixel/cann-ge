/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history/history_registry_reader.h"
#include "history/history_registry_interface.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ge::es::history;

namespace {
void EnsureDir(const std::string &path) {
  std::string cmd = "mkdir -p " + path;
  (void)std::system(cmd.c_str());
}

std::string DirName(const std::string &path) {
  const auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return ".";
  }
  return path.substr(0, pos);
}

template <typename F>
void ExpectRuntimeErrorContains(F &&fn, const std::string &expected_substr) {
  try {
    fn();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(expected_substr), std::string::npos);
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

std::string BuildDateFromOffsetDays(int offset_days) {
  std::time_t now = std::time(nullptr);
  std::tm *local_tm = std::localtime(&now);
  if (local_tm == nullptr) {
    return "";
  }
  std::tm shifted_tm = *local_tm;
  shifted_tm.tm_mday += offset_days;
  if (std::mktime(&shifted_tm) < 0) {
    return "";
  }
  char date[11] = {};
  if (std::strftime(date, sizeof(date), "%Y-%m-%d", &shifted_tm) == 0) {
    return "";
  }
  return date;
}
}  // namespace

class HistoryRegistryReaderUT : public ::testing::Test {
 protected:
  void SetUp() override {
    fixture_root_ = DirName(__FILE__) + "/fixtures/history_registry/math/";
  }

  void TearDown() override {
    if (!temp_dir_.empty()) {
      (void)std::system(("rm -rf " + temp_dir_).c_str());
    }
  }

  std::string fixture_root_;
  std::string temp_dir_;
};

TEST_F(HistoryRegistryReaderUT, LoadIndexReturnsVersionsSortedByDate) {
  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex(fixture_root_);
  ASSERT_FALSE(versions.empty());
  EXPECT_EQ(versions.size(), 2U);

  EXPECT_EQ(versions[0].release_version, "8.0.RC1");
  EXPECT_EQ(versions[0].release_date, "2024-09-30");
  EXPECT_EQ(versions[0].branch_name, "master");

  EXPECT_EQ(versions[1].release_version, "8.0.RC2");
  EXPECT_EQ(versions[1].release_date, "2025-01-15");
  EXPECT_EQ(versions[1].branch_name, "");
}

TEST_F(HistoryRegistryReaderUT, LoadIndexWithNonexistentPkgDirReturnsEmpty) {
  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex("/fake_dir");
  ASSERT_TRUE(versions.empty());
}

TEST_F(HistoryRegistryReaderUT, LoadIndexWithInvalidJsonReturnsEmpty) {
  temp_dir_ = "./load_index_bad_json_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_);
  std::ofstream(temp_dir_ + "index.json") << "not valid json {";

  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex(temp_dir_);
  EXPECT_TRUE(versions.empty());
}

TEST_F(HistoryRegistryReaderUT, LoadIndexWithoutReleasesFieldReturnsEmpty) {
  temp_dir_ = "./load_index_no_releases_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_);
  std::ofstream(temp_dir_ + "index.json") << R"({"version":"1.0.0"})";

  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex(temp_dir_);
  EXPECT_TRUE(versions.empty());
}

TEST_F(HistoryRegistryReaderUT, LoadIndexSkipsVersionWithoutMetadata) {
  temp_dir_ = "./load_index_no_meta_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_);
  EnsureDir(temp_dir_ + "registry/v1/");
  EnsureDir(temp_dir_ + "registry/v2/");
  std::ofstream(temp_dir_ + "index.json") << R"({"version":"1.0.0","releases":[
    {"release_version":"v1","release_date":"2024-01-01"},
    {"release_version":"v2","release_date":"2024-06-01"}
  ]})";
  std::ofstream(temp_dir_ + "registry/v1/metadata.json") << R"({"release_version":"v1","branch_name":"br1"})";

  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex(temp_dir_);
  ASSERT_EQ(versions.size(), 1U);
  EXPECT_EQ(versions[0].release_version, "v1");
  EXPECT_EQ(versions[0].release_date, "2024-01-01");
  EXPECT_EQ(versions[0].branch_name, "br1");
}

TEST_F(HistoryRegistryReaderUT, LoadIndexSkipsInvalidReleaseEntry) {
  temp_dir_ = "./load_index_bad_entry_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_);
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "index.json") << R"({"version":"1.0.0","releases":[
    {"release_version":"v1","release_date":"2024-01-01"},
    "not_an_object",
    {"release_version":123,"release_date":"2024-06-01"}
  ]})";
  std::ofstream(temp_dir_ + "registry/v1/metadata.json") << R"({"release_version":"v1"})";

  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex(temp_dir_);
  ASSERT_EQ(versions.size(), 1U);
  EXPECT_EQ(versions[0].release_version, "v1");
  EXPECT_EQ(versions[0].release_date, "2024-01-01");
}

TEST_F(HistoryRegistryReaderUT, LoadIndexSkipsVersionWithMismatchedMetadataReleaseVersion) {
  temp_dir_ = "./load_index_mismatch_meta_version_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_);
  EnsureDir(temp_dir_ + "registry/v1/");
  EnsureDir(temp_dir_ + "registry/v2/");
  std::ofstream(temp_dir_ + "index.json") << R"({"version":"1.0.0","releases":[
    {"release_version":"v1","release_date":"2024-01-01"},
    {"release_version":"v2","release_date":"2024-06-01"}
  ]})";
  std::ofstream(temp_dir_ + "registry/v1/metadata.json") << R"({"release_version":"v1"})";
  std::ofstream(temp_dir_ + "registry/v2/metadata.json") << R"({"release_version":"v2_other"})";

  std::vector<VersionMeta> versions = HistoryRegistryReader::LoadIndex(temp_dir_);
  ASSERT_EQ(versions.size(), 1U);
  EXPECT_EQ(versions[0].release_version, "v1");
  EXPECT_EQ(versions[0].release_date, "2024-01-01");
}

TEST_F(HistoryRegistryReaderUT, LoadMetadataReadsVersionMeta) {
  VersionMeta meta = HistoryRegistryReader::LoadMetadata(fixture_root_, "8.0.RC1");
  EXPECT_EQ(meta.release_version, "8.0.RC1");
  EXPECT_EQ(meta.branch_name, "master");
}

TEST_F(HistoryRegistryReaderUT, LoadMetadataMissingFileThrows) {
  ExpectRuntimeErrorContains(
    [&]() { (void)HistoryRegistryReader::LoadMetadata(fixture_root_, "NonExistentVersion"); },
    "Failed to load metadata.json: cannot open file: " + fixture_root_ + "registry/NonExistentVersion/metadata.json");
}

TEST_F(HistoryRegistryReaderUT, LoadMetadataReleaseVersionMismatchThrows) {
  temp_dir_ = "./load_metadata_mismatch_version" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/metadata.json") << R"({"release_version":"v2"})";

  ExpectRuntimeErrorContains([&]() { (void)HistoryRegistryReader::LoadMetadata(temp_dir_, "v1"); },
                             "Failed to load metadata.json: release_version is not equal to version");
}

TEST_F(HistoryRegistryReaderUT, LoadMetadataInvalidBranchNameTypeThrows) {
  temp_dir_ = "./load_metadata_branch_type_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/metadata.json") << R"({"release_version":"v1","branch_name":123})";

  ExpectRuntimeErrorContains([&]() { (void)HistoryRegistryReader::LoadMetadata(temp_dir_, "v1"); },
                             "Failed to load metadata.json: branch_name must be a string");
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoFindsOpAndParses) {
  IrOpProto proto = HistoryRegistryReader::LoadOpProto(fixture_root_, "8.0.RC1", "phony_op");
  EXPECT_EQ(proto.op_type, "phony_op");
  ASSERT_EQ(proto.inputs.size(), 2U);
  EXPECT_EQ(proto.inputs[0].name, "input");
  EXPECT_EQ(proto.inputs[0].type, ge::kIrInputRequired);
  EXPECT_EQ(proto.inputs[0].dtype, "TensorType({DT_FLOAT, DT_INT32})");
  EXPECT_EQ(proto.inputs[1].name, "dy_input");
  EXPECT_EQ(proto.inputs[1].type, ge::kIrInputDynamic);
  EXPECT_EQ(proto.inputs[1].dtype, "TensorType({DT_FLOAT, DT_INT32, DT_INT64})");
  ASSERT_EQ(proto.outputs.size(), 1U);
  EXPECT_EQ(proto.outputs[0].name, "output");
  EXPECT_EQ(proto.outputs[0].type, ge::kIrOutputDynamic);
  EXPECT_EQ(proto.outputs[0].dtype, "TensorType({DT_FLOAT, DT_INT32})");
  ASSERT_EQ(proto.attrs.size(), 1U);
  EXPECT_EQ(proto.attrs[0].name, "index");
  EXPECT_EQ(proto.attrs[0].av_type, "Int");
  EXPECT_EQ(proto.attrs[0].required, false);
  EXPECT_EQ(proto.attrs[0].default_value, "0");
  ASSERT_EQ(proto.subgraphs.size(), 2U);
  EXPECT_EQ(proto.subgraphs[0].name, "branch_a");
  EXPECT_EQ(proto.subgraphs[0].type, ge::kStatic);
  EXPECT_EQ(proto.subgraphs[1].name, "branch_b");
  EXPECT_EQ(proto.subgraphs[1].type, ge::kStatic);
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoNonExistentOpThrows) {
  ExpectRuntimeErrorContains(
      [&]() { (void)HistoryRegistryReader::LoadOpProto(fixture_root_, "8.0.RC1", "NonExistentOp"); },
      "Op NonExistentOp not found in operators.json");
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoDuplicateOpTypeThrows) {
  temp_dir_ = "./load_op_dup_op_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/operators.json") << R"({"operators":[
    {"op_type":"Foo","inputs":[],"outputs":[],"attrs":[],"subgraphs":[]},
    {"op_type":"Foo","inputs":[],"outputs":[],"attrs":[],"subgraphs":[]}
  ]})";

  ExpectRuntimeErrorContains([&]() { (void)HistoryRegistryReader::LoadOpProto(temp_dir_, "v1", "Foo"); },
                             "Found duplicate op Foo in operators.json");
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoSkipsInvalidEntries) {
  temp_dir_ = "./load_op_bad_op_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/operators.json") << R"({"operators":[
    "not_an_object",
    {"no_op_type_field": true},
    {"op_type":"Good","inputs":[],"outputs":[],"attrs":[],"subgraphs":[]}
  ]})";

  IrOpProto proto = HistoryRegistryReader::LoadOpProto(temp_dir_, "v1", "Good");
  EXPECT_EQ(proto.op_type, "Good");
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoInvalidOperatorsRootThrows) {
  temp_dir_ = "./load_op_invalid_op_root_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/operators.json") << R"([])";

  ExpectRuntimeErrorContains([&]() { (void)HistoryRegistryReader::LoadOpProto(temp_dir_, "v1", "Foo"); },
                             "Failed to load operators.json: operators.json is not a JSON object");
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoInvalidOperatorsFieldThrows) {
  temp_dir_ = "./load_op_invalid_op_field_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/operators.json") << R"({"operators":{}})";

  ExpectRuntimeErrorContains([&]() { (void)HistoryRegistryReader::LoadOpProto(temp_dir_, "v1", "Foo"); },
                             "Failed to load operators.json: operators is required and must be an array");
}

TEST_F(HistoryRegistryReaderUT, LoadOpProtoMatchedOpMalformedThrows) {
  temp_dir_ = "./load_op_bad_op_parse_" + std::to_string(getpid()) + "/";
  EnsureDir(temp_dir_ + "registry/v1/");
  std::ofstream(temp_dir_ + "registry/v1/operators.json") << R"({"operators":[
    {"op_type":"Foo","inputs":[{"name":"x"}],"outputs":[],"attrs":[],"subgraphs":[]}
  ]})";

  ExpectRuntimeErrorContains([&]() { (void)HistoryRegistryReader::LoadOpProto(temp_dir_, "v1", "Foo"); },
                             "Failed to parse json for op Foo: inputs[0].type is required and must be a string");
}

TEST_F(HistoryRegistryReaderUT, SelectWindowVersionsFiltersByDate) {
  std::vector<VersionMeta> all = HistoryRegistryReader::LoadIndex(fixture_root_);
  ASSERT_EQ(all.size(), 2U);

  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "8.0.RC2");
  ASSERT_EQ(window.size(), 2U);
  EXPECT_TRUE(window[0].release_version == "8.0.RC1");
  EXPECT_TRUE(window[1].release_version == "8.0.RC2");

  window = HistoryRegistryReader::SelectWindowVersions(all, "8.0.RC1");
  ASSERT_EQ(window.size(), 1U);
  EXPECT_TRUE(window[0].release_version == "8.0.RC1");
}

TEST_F(HistoryRegistryReaderUT, SelectWindowVersionsCurrentNotInListReturnsEmpty) {
  std::vector<VersionMeta> all = HistoryRegistryReader::LoadIndex(fixture_root_);
  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "99.0.NotFound");
  EXPECT_TRUE(window.empty());
}

TEST_F(HistoryRegistryReaderUT, SelectWindowVersionsCurrentVersionEmptyUsesCurrentDateAnchor) {
  std::vector<VersionMeta> all = {
    {"v_old", BuildDateFromOffsetDays(-500), ""},
    {"v_in_window", BuildDateFromOffsetDays(-120), ""},
    {"v_recent", BuildDateFromOffsetDays(-1), ""},
    {"v_future", BuildDateFromOffsetDays(7), ""},
  };
  ASSERT_FALSE(all[0].release_date.empty());
  ASSERT_FALSE(all[1].release_date.empty());
  ASSERT_FALSE(all[2].release_date.empty());
  ASSERT_FALSE(all[3].release_date.empty());

  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(all, "");
  ASSERT_EQ(window.size(), 2U);
  EXPECT_EQ(window[0].release_version, "v_in_window");
  EXPECT_EQ(window[1].release_version, "v_recent");
}

TEST_F(HistoryRegistryReaderUT, SelectWindowVersionsEmptyListReturnsEmpty) {
  std::vector<VersionMeta> empty;
  std::vector<VersionMeta> window = HistoryRegistryReader::SelectWindowVersions(empty, "8.0.RC1");
  EXPECT_TRUE(window.empty());
}
