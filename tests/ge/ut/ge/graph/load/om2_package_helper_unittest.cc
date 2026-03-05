/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/om2_package_helper.h"
#include "common/helper/om2/json_file.h"
#include "file_utils.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "common/env_path.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/file_utils.h"

#include "ge_runtime_stub/include/common/share_graph.h"
#include "ge_runtime_stub/include/faker/ge_model_builder.h"
#include "ge_runtime_stub/include/faker/aicore_taskdef_faker.h"

namespace ge {
namespace {
class ScopedEnvVar {
 public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old_value = getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
      has_old_value_ = true;
    }
    (void)setenv(name, value, 1);
  }

  ~ScopedEnvVar() {
    if (has_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
      return;
    }
    (void)unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_ = false;
};

GeRootModelPtr CreateGeRootModelWithAicoreOp() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({1024});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint64_t> weights_value(64, 1024);
  const size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom(reinterpret_cast<uint8_t *>(weights_value.data()), weight_size));

  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  return ge_root_model;
}

GeRootModelPtr CreateInvalidGeRootModel() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  return ge_root_model;
}

}  // namespace

class Om2PackageHelperUt : public testing::Test {
 public:
  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_case_name = test_info->test_case_name();  // Om2PackageHelperUt
    test_work_dir = EnvPath().GetOrCreateCaseTmpPath(test_case_name);
    const auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_HOME_PATH", ascend_install_path.c_str(), 1);
    asan_guard_ = std::make_unique<ScopedEnvVar>("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
    lsan_guard_ = std::make_unique<ScopedEnvVar>("LSAN_OPTIONS", "exitcode=0");
  }
  void TearDown() override {
    lsan_guard_.reset();
    asan_guard_.reset();
    EnvPath().RemoveRfCaseTmpPath(test_case_name);
    unsetenv("ASCEND_HOME_PATH");
  }

 public:
  std::string test_case_name;
  std::string test_work_dir;
  const std::string kZipFileBaseName = "fake_test";

 private:
  std::unique_ptr<ScopedEnvVar> asan_guard_;
  std::unique_ptr<ScopedEnvVar> lsan_guard_;
};

TEST_F(Om2PackageHelperUt, ConvertOm2Model_Ok_GenOm2WithAicoreNode) {
  Om2PackageHelper om2_packager;
  const auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + ".om2"});
  ASSERT_EQ(om2_packager.SaveToOmRootModel(ge_root_model, output_file, model_data, false), SUCCESS);
  ASSERT_EQ(mmAccess2(output_file.c_str(), M_F_OK), EOK);

  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  RAIIZipArchive archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  const std::set<std::string> expect_files = {
      "fake_test/data/model_0/runtime/g1_kernel_reg.cpp",
      "fake_test/data/model_0/runtime/g1_resources.cpp",
      "fake_test/data/model_0/runtime/g1_args_manager.cpp",
      "fake_test/data/model_0/runtime/g1_load_and_run.cpp",
      "fake_test/data/model_0/runtime/g1_interface.h",
      "fake_test/data/model_0/runtime/Makefile",
      "fake_test/data/model_0/runtime/libg1_om2.so",
      "fake_test/data/constants/constant_0",
      "fake_test/data/constants/model_0_constants_config.json",
      "fake_test/data/kernels_npu_arch/add1_faked_kernel.o",
      "fake_test/data/model_0/model_meta.json",
      "fake_test/manifest.json",
  };
  EXPECT_EQ(file_names.size(), expect_files.size());
  for (const auto &file_name : file_names) {
    EXPECT_EQ(expect_files.count(file_name), 1);
  }

  size_t manifest_size = 0;
  const auto manifest_buf = archive.ExtractToMem("fake_test/manifest.json", manifest_size);
  ASSERT_NE(manifest_buf, nullptr);
  const JsonFile manifest_json(reinterpret_cast<const uint8_t *>(manifest_buf.get()), manifest_size);
  ASSERT_TRUE(manifest_json.IsValid());
  std::string atc_command;
  ASSERT_TRUE(manifest_json.Get("atc_command", atc_command));
  EXPECT_EQ(atc_command, "");
  int model_num = 0;
  ASSERT_TRUE(manifest_json.Get("model_num", model_num));
  EXPECT_EQ(model_num, 1);
  std::string om2_version;
  ASSERT_TRUE(manifest_json.Get("om2_version", om2_version));
  EXPECT_EQ(om2_version, "0");

  size_t model_meta_size = 0;
  const auto model_meta_buf = archive.ExtractToMem("fake_test/data/model_0/model_meta.json", model_meta_size);
  ASSERT_NE(model_meta_buf, nullptr);
  const JsonFile model_meta_json(reinterpret_cast<const uint8_t *>(model_meta_buf.get()), model_meta_size);
  ASSERT_TRUE(model_meta_json.IsValid());
  EXPECT_EQ(model_meta_json.Raw().at("name"), JsonFile::json("g1"));
  EXPECT_EQ(model_meta_json.Raw().at("dynamic_batch_info"), JsonFile::json::array());
  EXPECT_EQ(model_meta_json.Raw().at("dynamic_output_shape"), JsonFile::json::array());
  EXPECT_EQ(model_meta_json.Raw().at("dynamic_type"), JsonFile::json(0));
  EXPECT_EQ(model_meta_json.Raw().at("user_designate_shape_order"), JsonFile::json::array());

  const JsonFile::json expected_inputs = JsonFile::json::array({
                                                                   {{"data_type", "DT_FLOAT"},
                                                                       {"format", "ND"},
                                                                       {"index", 0},
                                                                       {"name", "data1"},
                                                                       {"shape", JsonFile::json::array({1, 2, 3, 4})},
                                                                       {"shape_range", JsonFile::json::array()},
                                                                       {"shape_v2", JsonFile::json::array({1, 2, 3, 4})},
                                                                       {"size", 0}},
                                                                   {{"data_type", "DT_FLOAT"},
                                                                       {"format", "NCHW"},
                                                                       {"index", 1},
                                                                       {"name", "data2"},
                                                                       {"shape", JsonFile::json::array({1, 1, 224, 224})},
                                                                       {"shape_range", JsonFile::json::array()},
                                                                       {"shape_v2", JsonFile::json::array({1, 1, 224, 224})},
                                                                       {"size", 0}},
                                                               });
  EXPECT_EQ(model_meta_json.Raw().at("inputs"), expected_inputs);

  const JsonFile::json expected_outputs = JsonFile::json::array({
                                                                    {{"data_type", "DT_FLOAT"},
                                                                        {"format", "ND"},
                                                                        {"index", 0},
                                                                        {"name", "output_0_reshape1_0"},
                                                                        {"shape", JsonFile::json::array()},
                                                                        {"shape_range", JsonFile::json::array()},
                                                                        {"size", 4}},
                                                                });
  EXPECT_EQ(model_meta_json.Raw().at("outputs"), expected_outputs);
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_Fail_GenFailedAndRemoveOm2File) {
  Om2PackageHelper om2_packager;
  const auto ge_root_model = CreateInvalidGeRootModel();
  ASSERT_NE(ge_root_model, nullptr);
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_invalid.om2"});
  ASSERT_NE(om2_packager.SaveToOmRootModel(ge_root_model, output_file, model_data, false), SUCCESS);
  ASSERT_NE(mmAccess2(output_file.c_str(), M_F_OK), EOK);
}
}  // namespace ge
