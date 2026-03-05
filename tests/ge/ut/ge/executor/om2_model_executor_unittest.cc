/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "framework/runtime/om2_model_executor.h"
#include "common/env_path.h"
#include "common/helper/om2/zip_archive.h"
#include "common/path_utils.h"
#include "graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

namespace ge {
namespace {
constexpr const char *kOm2BaseName = "om2_model_executor_test";
constexpr const char *kModelName = "g1";

struct ModelDataHolder {
  ge::ModelData model_data{};
  std::unique_ptr<char[]> buffer;
};

std::string GetParentDir(const std::string &path) {
  const auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return {};
  }
  return path.substr(0, pos);
}

void WriteTextFile(const std::string &file_path, const std::string &content) {
  const auto parent_dir = GetParentDir(file_path);
  ASSERT_FALSE(parent_dir.empty());
  ASSERT_EQ(CreateDir(parent_dir), 0);
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs << content;
  ASSERT_TRUE(ofs.good());
}

void WriteBinaryFile(const std::string &file_path, const std::vector<uint8_t> &content) {
  const auto parent_dir = GetParentDir(file_path);
  ASSERT_FALSE(parent_dir.empty());
  ASSERT_EQ(CreateDir(parent_dir), 0);
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs.write(reinterpret_cast<const char *>(content.data()), static_cast<std::streamsize>(content.size()));
  ASSERT_TRUE(ofs.good());
}

void RunCommandOrAssert(const std::string &command) {
  const std::string wrapped_command =
      "env ASAN_OPTIONS=detect_leaks=0:halt_on_error=0 LSAN_OPTIONS=exitcode=0 " + command;
  ASSERT_EQ(system(wrapped_command.c_str()), 0) << wrapped_command;
}

std::string MakeManifestJson() {
  return R"({
    "atc_command": "",
    "model_num": 1,
    "om2_version": "0"
})";
}

std::string MakeModelMetaJson() {
  return R"({
    "dynamic_batch_info": [],
    "dynamic_output_shape": [],
    "dynamic_type": 0,
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "data1",
            "shape": [1, 2, 3, 4],
            "shape_range": [],
            "shape_v2": [1, 2, 3, 4],
            "size": 0
        },
        {
            "data_type": "DT_FLOAT",
            "format": "NCHW",
            "index": 1,
            "name": "data2",
            "shape": [1, 1, 224, 224],
            "shape_range": [],
            "shape_v2": [1, 1, 224, 224],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output_0_reshape1_0",
            "shape": [],
            "shape_range": [],
            "size": 4
        }
    ],
    "user_designate_shape_order": []
})";
}

std::string MakeInterfaceHeader() {
  return R"(#pragma once

#include <cstddef>
#include <cstdint>

namespace om2 {
struct FakeModel {
  uint64_t session_id;
};
}

extern "C" {
int Om2ModelCreate(void **model_handle, const char **bin_files, const void **bin_data, size_t *bin_size, int bin_num,
                   void *host_weight_mem_ptr, uint64_t *session_id);
int Om2ModelRunAsync(void **model_handle, void *stream, int input_count, void **input_data, int output_count,
                     void **output_data);
int Om2ModelRun(void **model_handle, int input_count, void **input_data, int output_count, void **output_data);
int Om2ModelDestroy(void **model_handle);
}
)";
}

std::string MakeLoadAndRunCpp() {
  return R"(#include "g1_interface.h"

#include <new>

extern "C" int Om2ModelCreate(void **model_handle, const char **, const void **, size_t *, int, void *, uint64_t *session_id) {
  if (model_handle == nullptr) {
    return 1;
  }
  auto *model = new (std::nothrow) om2::FakeModel();
  if (model == nullptr) {
    return 1;
  }
  model->session_id = (session_id == nullptr) ? 0UL : *session_id;
  *model_handle = model;
  return 0;
}

extern "C" int Om2ModelRunAsync(void **model_handle, void *, int input_count, void **input_data, int output_count,
                                void **output_data) {
  if ((model_handle == nullptr) || (*model_handle == nullptr) || (input_data == nullptr) || (output_data == nullptr)) {
    return 1;
  }
  return (input_count == 2 && output_count == 1) ? 0 : 1;
}

extern "C" int Om2ModelRun(void **model_handle, int input_count, void **input_data, int output_count,
                           void **output_data) {
  if ((model_handle == nullptr) || (*model_handle == nullptr) || (input_data == nullptr) || (output_data == nullptr)) {
    return 1;
  }
  return (input_count == 2 && output_count == 1) ? 0 : 1;
}

extern "C" int Om2ModelDestroy(void **model_handle) {
  if ((model_handle == nullptr) || (*model_handle == nullptr)) {
    return 0;
  }
  delete static_cast<om2::FakeModel *>(*model_handle);
  *model_handle = nullptr;
  return 0;
}
)";
}

std::string MakeEmptyCpp(const std::string &header_name) {
  return "#include \"" + header_name + "\"\n";
}

std::string MakeCMakeLists() {
  return R"(cmake_minimum_required(VERSION 3.10)
project(g1_om2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(g1_om2 SHARED
  g1_resources.cpp
  g1_kernel_reg.cpp
  g1_load_and_run.cpp
  g1_args_manager.cpp
)

set_target_properties(g1_om2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
)";
}
}  // namespace

class Om2ModelExecutorUt : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    test_work_dir_ = EnvPath().GetOrCreateCaseTmpPath("Om2ModelExecutorUt");
    setenv("ASCEND_WORK_PATH", test_work_dir_.c_str(), 1);
    om2_file_path_ = PathUtils::Join({test_work_dir_, std::string(kOm2BaseName) + ".om2"});
    PrepareOm2File();
  }

  static void TearDownTestSuite() {
    unsetenv("ASCEND_WORK_PATH");
    EnvPath().RemoveRfCaseTmpPath("Om2ModelExecutorUt");
  }

  static void PrepareOm2File() {
    std::call_once(prepare_once_, []() {
      const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime"});
      const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
      const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
      const std::string archive_constant_path = PathUtils::Join({test_work_dir_, "constant_0"});
      const std::string archive_constant_cfg_path = PathUtils::Join({test_work_dir_, "model_0_constants_config.json"});

      (void)PathUtils::RemoveDirectories(runtime_dir);
      ASSERT_EQ(CreateDir(runtime_dir), 0);
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
      WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

      const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
      const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
      RunCommandOrAssert(cmake_config_cmd);
      RunCommandOrAssert(cmake_build_cmd);
      ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

      WriteBinaryFile(archive_constant_path, std::vector<uint8_t>(16U, 0U));
      WriteTextFile(archive_constant_cfg_path, R"({"constant_file":"constant_0"})");

      ZipArchiveWriter zip_writer(om2_file_path_);
      ASSERT_TRUE(zip_writer.IsMemFileOpened());
      const auto manifest = MakeManifestJson();
      const auto model_meta = MakeModelMetaJson();
      ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
      ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt", PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h", PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp", PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp", PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp", PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp", PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
      ASSERT_TRUE(zip_writer.WriteFile("data/constants/constant_0", archive_constant_path, false));
      ASSERT_TRUE(zip_writer.WriteFile("data/constants/model_0_constants_config.json", archive_constant_cfg_path, false));
      ASSERT_TRUE(zip_writer.SaveModelDataToFile());
      ASSERT_EQ(mmAccess2(om2_file_path_.c_str(), M_F_OK), EOK);
    });
  }

  static ModelDataHolder LoadValidModelData() {
    PrepareOm2File();
    uint32_t model_buf_size = 0U;
    auto model_buf = GetBinDataFromFile(om2_file_path_, model_buf_size);
    EXPECT_NE(model_buf, nullptr);
    EXPECT_GT(model_buf_size, 0U);

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.get();
    holder.model_data.model_len = model_buf_size;
    holder.buffer = std::move(model_buf);
    return holder;
  }

  static void ConstructIoTensors(std::vector<gert::Tensor> &input_tensors, std::vector<gert::Tensor> &output_tensors,
                                 std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) {
    input_tensors.resize(2);
    output_tensors.resize(1);
    TensorCheckUtils::ConstructGertTensor(input_tensors[0], {2, 16}, DataType::DT_FLOAT, Format::FORMAT_ND);
    TensorCheckUtils::ConstructGertTensor(input_tensors[1], {2, 16}, DataType::DT_FLOAT, Format::FORMAT_ND);
    TensorCheckUtils::ConstructGertTensor(output_tensors[0], {2, 16}, DataType::DT_FLOAT, Format::FORMAT_ND);

    inputs = {&input_tensors[0], &input_tensors[1]};
    outputs = {&output_tensors[0]};
  }

  static std::string test_work_dir_;
  static std::string om2_file_path_;
  static std::once_flag prepare_once_;
};

std::string Om2ModelExecutorUt::test_work_dir_;
std::string Om2ModelExecutorUt::om2_file_path_;
std::once_flag Om2ModelExecutorUt::prepare_once_;

TEST_F(Om2ModelExecutorUt, load_invalid_model_data) {
  gert::Om2ModelExecutor executor;
  ModelData invalid_model_data{};
  EXPECT_NE(executor.Load(invalid_model_data), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  EXPECT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_before_load_failed) {
  gert::Om2ModelExecutor executor;
  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_NE(executor.Run(inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_ok_after_load) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  ASSERT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);

  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_EQ(executor.Run(inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_async_before_load_failed) {
  gert::Om2ModelExecutor executor;
  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_NE(executor.RunAsync(nullptr, inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_async_ok_after_load) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  ASSERT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);

  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_EQ(executor.RunAsync(nullptr, inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, get_model_desc_info_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  ASSERT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);

  std::vector<ge::TensorDesc> input_desc;
  std::vector<ge::TensorDesc> output_desc;
  EXPECT_EQ(executor.GetModelDescInfo(input_desc, output_desc, false), SUCCESS);
  ASSERT_EQ(input_desc.size(), 2U);
  ASSERT_EQ(output_desc.size(), 1U);
  EXPECT_EQ(input_desc[0].GetName(), "data1");
  EXPECT_EQ(input_desc[1].GetName(), "data2");
  EXPECT_EQ(output_desc[0].GetName(), "output_0_reshape1_0");

  std::vector<ge::TensorDesc> input_desc_v2;
  std::vector<ge::TensorDesc> output_desc_v2;
  EXPECT_EQ(executor.GetModelDescInfo(input_desc_v2, output_desc_v2, true), SUCCESS);
  EXPECT_EQ(input_desc_v2.size(), input_desc.size());
  EXPECT_EQ(output_desc_v2.size(), output_desc.size());
}

TEST_F(Om2ModelExecutorUt, get_model_attrs_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  ASSERT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);

  std::vector<std::string> dynamic_output_shape;
  EXPECT_EQ(executor.GetModelAttrs(dynamic_output_shape), SUCCESS);
  EXPECT_TRUE(dynamic_output_shape.empty());
}

TEST_F(Om2ModelExecutorUt, get_dynamic_batch_info_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  ASSERT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);

  std::vector<std::vector<int64_t>> dynamic_batch_info;
  int32_t dynamic_type = -1;
  EXPECT_EQ(executor.GetDynamicBatchInfo(dynamic_batch_info, dynamic_type), SUCCESS);
  EXPECT_TRUE(dynamic_batch_info.empty());
  EXPECT_EQ(dynamic_type, 0);
}

TEST_F(Om2ModelExecutorUt, get_user_designate_shape_order_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  ASSERT_EQ(executor.Load(model_data_holder.model_data), SUCCESS);

  std::vector<std::string> user_designate_shape_order;
  EXPECT_EQ(executor.GetUserDesignateShapeOrder(user_designate_shape_order), SUCCESS);
  EXPECT_TRUE(user_designate_shape_order.empty());
}
}  // namespace ge
