/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <fstream>
#include <regex>
#include "registry/op_impl_space_registry_v2.h"
#include "runtime/om2_model_executor.h"
#include "common/checker.h"
#include "mmpa/mmpa_api.h"
#include "ge/ge_error_codes.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/helper/om2/om2_utils.h"
#include "graph/utils/type_utils.h"
#include "runtime/mem.h"
#include "common/helper/om2/zip_archive.h"
#include "common/helper/om2/json_file.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace gert {
namespace {
constexpr size_t kMaxErrorStringLen = 128U;
constexpr size_t FILE_MAGIC_HEADER_SIZE = 4U;
constexpr uint8_t OM2_MAGIC[] = {0x50, 0x4B, 0x03, 0x04};

using Om2ModelHandle = void *;
using CreateFunc = ge::graphStatus (*)(Om2ModelHandle *, const char **, const void **, size_t *, int, void *,
                                       uint64_t *);
using DestroyFunc = ge::graphStatus (*)(Om2ModelHandle *);
using RunFunc = ge::graphStatus (*)(Om2ModelHandle *, int, void **, int, void **);
using RunAsyncFunc = ge::graphStatus (*)(Om2ModelHandle *, rtStream_t, int, void **, int, void **);

struct RunModelInfo {
  std::string so_file;
  ge::JsonFile model_meta_json;
  void *so_handle = nullptr;
  std::string model_name;
  Om2ModelHandle model_handle = nullptr;
  CreateFunc create_func = nullptr;
  DestroyFunc destroy_func = nullptr;
  RunFunc run_func = nullptr;
  RunAsyncFunc run_async_func = nullptr;
};

struct KernelBinInfo {
  std::string file;
  ge::UniqueByteBuffer data;
  size_t data_size;
};

std::pair<std::string, std::string> ExtractParentDirAndFileName(const std::string &abs_path) {
  if (abs_path.empty()) {
    return {"", ""};
  }
  size_t last_slash = abs_path.find_last_of('/');
  if (last_slash == std::string::npos) {
    return {"", abs_path};
  }
  if (last_slash == 0) {
    return {"/", abs_path.substr(1)};
  }
  return {abs_path.substr(0, last_slash + 1), abs_path.substr(last_slash + 1)};
}

bool IsFileNameEndsWith(const std::string &file_name, const std::string &ext) {
  return file_name.size() >= ext.size() && file_name.compare(file_name.size() - ext.size(), ext.size(), ext) == 0;
}

template <typename T>
ge::Status GetModelJsonValue(const std::string &key, T &out, const ge::JsonFile &json_file) {
  GE_ASSERT_TRUE(json_file.IsValid());
  GE_ASSERT_TRUE(json_file.Get(key, out));
  return ge::SUCCESS;
}

ge::Status SetTensorDesc(ge::JsonFile::json &tensor_array_json, std::vector<ge::TensorDesc> &tensor_desc_array,
                         const bool new_model_desc = false) {
  for (ge::JsonFile::json &tensor : tensor_array_json) {
    ge::JsonFile tensor_obj{tensor};
    ge::TensorDesc tensor_desc;
    std::string name;
    GE_ASSERT_TRUE(tensor_obj.Get("name", name));
    tensor_desc.SetName(name.c_str());
    std::string data_type;
    GE_ASSERT_TRUE(tensor_obj.Get("data_type", data_type));
    tensor_desc.SetDataType(ge::TypeUtils::SerialStringToDataType(data_type));
    std::string format;
    GE_ASSERT_TRUE(tensor_obj.Get("format", format));
    tensor_desc.SetFormat(ge::TypeUtils::SerialStringToFormat(format));
    int64_t size;
    GE_ASSERT_TRUE(tensor_obj.Get("size", size));
    tensor_desc.SetSize(size);
    std::string shape_key = new_model_desc ? "shape_v2" : "shape";
    std::vector<int64_t> shape_dims;
    GE_ASSERT_TRUE(tensor_obj.Get(shape_key, shape_dims));
    const ge::Shape ge_shape(shape_dims);
    tensor_desc.SetShape(ge_shape);
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    GE_ASSERT_TRUE(tensor_obj.Get("shape_range", shape_range));
    tensor_desc.SetShapeRange(shape_range);
    tensor_desc_array.emplace_back(tensor_desc);
  }
  return ge::SUCCESS;
}

uint64_t GetNextSessionId() {
  static std::atomic<uint64_t> atomic_session_id(0);
  return atomic_session_id.fetch_add(1);
}
}  // namespace

class Om2ModelExecutor::Impl {
 public:
  ge::Status ParseModel(ge::ModelData &model_data, ge::UniqueByteBuffer &weight_buf,
                        std::vector<KernelBinInfo> &kernel_bin_info) {
    has_model_ = false;
    run_model_info_ = RunModelInfo();
    weight_buf.reset(nullptr);
    kernel_bin_info.clear();

    GE_ASSERT_SUCCESS(ge::Om2Utils::CreateOm2WorkspaceDir(ws_dir_));
    ge::RAIIZipArchive archive(static_cast<uint8_t *>(model_data.model_data), model_data.model_len);
    GE_ASSERT_TRUE(archive.IsGood());
    const auto file_names = archive.ListFiles();

    for (const auto &file_name : file_names) {
      if (IsFileNameEndsWith(file_name, "model_meta.json")) {
        size_t buff_size = 0UL;
        auto buff_data = archive.ExtractToMem(file_name, buff_size);
        GE_ASSERT_TRUE(buff_data != nullptr && buff_size != 0U);
        run_model_info_.model_meta_json = ge::JsonFile(buff_data.get(), buff_size);
        GE_ASSERT_TRUE(run_model_info_.model_meta_json.IsValid());
        GE_ASSERT_SUCCESS(GetModelJsonValue("name", run_model_info_.model_name, run_model_info_.model_meta_json));
        has_model_ = true;
        break;
      }
    }
    GE_ASSERT_TRUE(has_model_);

    for (const auto &file_name : file_names) {
      if (IsFileNameEndsWith(file_name, "manifest.json") || IsFileNameEndsWith(file_name, "model_meta.json")) {
        continue;
      }

      size_t buff_size = 0UL;
      if (file_name.find("constant_") != std::string::npos) {
        auto buff_data = archive.ExtractToMem(file_name, buff_size);
        GE_ASSERT_TRUE(buff_data != nullptr && buff_size != 0U);
        weight_buf = std::move(buff_data);
        continue;
      }
      if (IsFileNameEndsWith(file_name, ".o")) {
        auto buff_data = archive.ExtractToMem(file_name, buff_size);
        GE_ASSERT_TRUE(buff_data != nullptr && buff_size != 0U);
        auto file_info = ExtractParentDirAndFileName(ws_dir_ + file_name);
        kernel_bin_info.push_back({file_info.second, std::move(buff_data), buff_size});
        continue;
      }
      if (IsFileNameEndsWith(file_name, "lib" + run_model_info_.model_name + "_om2.so")) {
        GE_ASSERT_TRUE(archive.ExtractToFile(file_name, ws_dir_));
        run_model_info_.so_file = ws_dir_ + file_name;
      }
    }
    GE_ASSERT_TRUE(!run_model_info_.so_file.empty(), "[OM2] Om2 compiled so not found, need to check .om2 file.");
    return ge::SUCCESS;
  }

  ge::Status LoadSharedObject() {
    GELOGI("[OM2] Begin loading so file %s", run_model_info_.so_file.c_str());
    GE_ASSERT_TRUE(!run_model_info_.so_file.empty());
    run_model_info_.so_handle = mmDlopen(run_model_info_.so_file.c_str(), MMPA_RTLD_NOW);
    if (run_model_info_.so_handle == nullptr) {
      const char_t *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGE(ge::FAILED, "[OM2][Invoke][DlOpen] Failed to  load so, path = [%s], error = [%s]",
             run_model_info_.so_file.c_str(), error);
      return ge::FAILED;
    }
    return ge::SUCCESS;
  }

  ge::Status ResolveSymbols() {
    GE_ASSERT_TRUE(run_model_info_.so_handle != nullptr);
    run_model_info_.create_func = reinterpret_cast<CreateFunc>(mmDlsym(run_model_info_.so_handle, "Om2ModelCreate"));
    GE_ASSERT_NOTNULL(run_model_info_.create_func);
    run_model_info_.destroy_func =
        reinterpret_cast<DestroyFunc>(mmDlsym(run_model_info_.so_handle, "Om2ModelDestroy"));
    GE_ASSERT_NOTNULL(run_model_info_.destroy_func);
    run_model_info_.run_func = reinterpret_cast<RunFunc>(mmDlsym(run_model_info_.so_handle, "Om2ModelRun"));
    GE_ASSERT_NOTNULL(run_model_info_.run_func);
    run_model_info_.run_async_func =
        reinterpret_cast<RunAsyncFunc>(mmDlsym(run_model_info_.so_handle, "Om2ModelRunAsync"));
    GE_ASSERT_NOTNULL(run_model_info_.run_async_func);
    return ge::SUCCESS;
  }

  ge::Status CreateModel(ge::UniqueByteBuffer &weight_buf, std::vector<KernelBinInfo> &kernel_bin_info) {
    GE_ASSERT_TRUE(has_model_);
    std::vector<const char *> bin_files(kernel_bin_info.size());
    std::vector<const void *> bin_data(kernel_bin_info.size());
    std::vector<size_t> bin_sizes(kernel_bin_info.size());
    for (auto i = 0U; i < kernel_bin_info.size(); ++i) {
      bin_files[i] = kernel_bin_info[i].file.c_str();
      bin_data[i] = kernel_bin_info[i].data.get();
      bin_sizes[i] = kernel_bin_info[i].data_size;
    }
    uint64_t new_session_id = GetNextSessionId();
    GE_ASSERT_SUCCESS(run_model_info_.create_func(&run_model_info_.model_handle, bin_files.data(), bin_data.data(),
                                                  bin_sizes.data(), bin_data.size(), weight_buf.get(),
                                                  &new_session_id));
    weight_buf.reset(nullptr);
    kernel_bin_info.clear();
    return ge::GRAPH_SUCCESS;
  }

  ge::Status Run(std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_NOTNULL(run_model_info_.run_func);
    GE_ASSERT_NOTNULL(run_model_info_.model_handle);
    GE_ASSERT_SUCCESS(run_model_info_.run_func(&run_model_info_.model_handle, inputs.size(),
                                               reinterpret_cast<void **>(inputs.data()), outputs.size(),
                                               reinterpret_cast<void **>(outputs.data())));
    return ge::GRAPH_SUCCESS;
  }

  ge::Status RunAsync(void *const stream, std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_NOTNULL(run_model_info_.run_async_func);
    GE_ASSERT_NOTNULL(run_model_info_.model_handle);
    GE_ASSERT_SUCCESS(run_model_info_.run_async_func(&run_model_info_.model_handle, stream, inputs.size(),
                                                     reinterpret_cast<void **>(inputs.data()), outputs.size(),
                                                     reinterpret_cast<void **>(outputs.data())));
    return ge::GRAPH_SUCCESS;
  }

  ge::Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &dynamic_batch_info, int32_t &dynamic_type) const {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_SUCCESS(GetModelJsonValue("dynamic_batch_info", dynamic_batch_info, run_model_info_.model_meta_json));
    return GetModelJsonValue("dynamic_type", dynamic_type, run_model_info_.model_meta_json);
  }

  ge::Status GetModelAttrs(std::vector<std::string> &dynamic_output_shape) const {
    GE_ASSERT_TRUE(has_model_);
    return GetModelJsonValue("dynamic_output_shape", dynamic_output_shape, run_model_info_.model_meta_json);
  }

  ge::Status GetModelDescInfo(std::vector<ge::TensorDesc> &input_desc, std::vector<ge::TensorDesc> &output_desc,
                              const bool new_model_desc) const {
    GE_ASSERT_TRUE(has_model_);
    GE_ASSERT_TRUE(run_model_info_.model_meta_json.IsValid());
    ge::JsonFile::json inputs;
    GE_ASSERT_TRUE(run_model_info_.model_meta_json.Get("inputs", inputs));
    GE_ASSERT_SUCCESS(SetTensorDesc(inputs, input_desc, new_model_desc));
    ge::JsonFile::json outputs;
    GE_ASSERT_TRUE(run_model_info_.model_meta_json.Get("outputs", outputs));
    GE_ASSERT_SUCCESS(SetTensorDesc(outputs, output_desc));
    return ge::SUCCESS;
  }

  ge::Status GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const {
    GE_ASSERT_TRUE(has_model_);
    return GetModelJsonValue("user_designate_shape_order", user_designate_shape_order, run_model_info_.model_meta_json);
  }

  void Cleanup() {
    if (run_model_info_.destroy_func != nullptr && run_model_info_.model_handle != nullptr) {
      const auto destroy_ret = run_model_info_.destroy_func(&run_model_info_.model_handle);
      if (destroy_ret != ge::GRAPH_SUCCESS) {
        GELOGI("[OM2] Releasing resources failed, so file: %s", run_model_info_.so_file.c_str());
      }
    } else {
      GELOGI("[OM2] Destroy func not found or model not created, so file: %s", run_model_info_.so_file.c_str());
    }
    if (run_model_info_.so_handle != nullptr) {
      if (mmDlclose(run_model_info_.so_handle) != 0) {
        const char_t *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        GELOGI("[OM2][Invoke][Dlclose] failed. path = %s, error = %s", run_model_info_.so_file.c_str(), error);
      }
      run_model_info_.so_handle = nullptr;
    }
    if (mmAccess2(ws_dir_.c_str(), M_F_OK) == EN_OK) {
      ge::Om2Utils::RmOm2WorkspaceDir(ws_dir_);
    }
  }

 private:
  std::string ws_dir_;
  RunModelInfo run_model_info_;
  bool has_model_ = false;
};

Om2ModelExecutor::Om2ModelExecutor() : impl_(std::make_unique<Impl>()) {}

Om2ModelExecutor::~Om2ModelExecutor() {
  if (impl_) {
    impl_->Cleanup();
  }
}

ge::Status Om2ModelExecutor::Load(ge::ModelData &model_data) const {
  ge::UniqueByteBuffer weight_buf;
  std::vector<KernelBinInfo> kernel_bin_info;
  GE_ASSERT_SUCCESS(impl_->ParseModel(model_data, weight_buf, kernel_bin_info));
  GE_ASSERT_SUCCESS(impl_->LoadSharedObject());
  GE_ASSERT_SUCCESS(impl_->ResolveSymbols());
  GE_ASSERT_SUCCESS(impl_->CreateModel(weight_buf, kernel_bin_info));
  return ge::SUCCESS;
}

ge::Status Om2ModelExecutor::Run(std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) const {
  return impl_->Run(inputs, outputs);
}

ge::Status Om2ModelExecutor::RunAsync(void *stream, std::vector<gert::Tensor *> &inputs,
                                      std::vector<gert::Tensor *> &outputs) const {
  return impl_->RunAsync(stream, inputs, outputs);
}

ge::Status Om2ModelExecutor::GetModelDescInfo(std::vector<ge::TensorDesc> &input_desc,
                                              std::vector<ge::TensorDesc> &output_desc, bool new_model_desc) const {
  return impl_->GetModelDescInfo(input_desc, output_desc, new_model_desc);
}

ge::Status Om2ModelExecutor::GetModelAttrs(std::vector<std::string> &dynamic_output_shape) const {
  return impl_->GetModelAttrs(dynamic_output_shape);
}

ge::Status Om2ModelExecutor::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &dynamic_batch_info,
                                                 int32_t &dynamic_type) const {
  return impl_->GetDynamicBatchInfo(dynamic_batch_info, dynamic_type);
}

ge::Status Om2ModelExecutor::GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const {
  return impl_->GetUserDesignateShapeOrder(user_designate_shape_order);
}

ge::Status LoadOm2DataFromFile(const std::string &model_path, ge::ModelData &model_data) {
  GELOGI("Begin to load om2 model data from file, path: [%s]", model_path.c_str());
  const std::string file_path = ge::RealPath(model_path.c_str());
  if (file_path.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E13026", std::vector<const char_t *>({"pathname", "reason"}),
        std::vector<const char_t *>({model_path.c_str(), "It is not a real path. Please check your model path."}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID,
           "[Call][RealPath] File path is invalid. Please check your text file '%s'.", model_path.c_str());
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  std::ifstream fs(file_path, std::ios::binary);
  if (!fs.is_open()) {
    std::array<char_t, kMaxErrorStringLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Open][File]Failed, file %s, error %s", model_path.c_str(), err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Open file %s failed, error %s", model_path.c_str(), err_msg);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  (void)fs.seekg(0, std::ifstream::end);
  const int64_t len = fs.tellg();
  GE_ASSERT_TRUE(len > 1U);
  (void)fs.seekg(0, std::ifstream::beg);

  auto *buffer = new (std::nothrow) char_t[static_cast<size_t>(len)];
  if (buffer == nullptr) {
    GELOGE(ge::FAILED, "[Alloc][Mem] Failed to alloc memory for om2, size: %lld", len);
    return ge::FAILED;
  }

  (void)fs.read(buffer, len);

  model_data.model_data = buffer;
  model_data.model_len = len;
  GELOGI("Load om2 model data success, path: %s, size: %zu", model_path.c_str(), static_cast<size_t>(len));

  return ge::SUCCESS;
}

std::unique_ptr<Om2ModelExecutor> LoadOm2ExecutorFromData(ge::ModelData &model_data, ge::Status &error_code) {
  auto executor = std::unique_ptr<Om2ModelExecutor>(new (std::nothrow) Om2ModelExecutor());
  if (executor == nullptr) {
    error_code = ge::FAILED;
    GELOGE(ge::FAILED, "Constructing Om2ModelExecutor failed.");
    return executor;
  }
  error_code = executor->Load(model_data);
  GE_ASSERT_SUCCESS(error_code);
  return executor;
}

ge::Status IsOm2Model(const void *const data, const size_t size, bool &is_support) {
  if (data == nullptr || size < FILE_MAGIC_HEADER_SIZE) {
    return ge::PARAM_INVALID;
  }
  is_support = std::memcmp(data, OM2_MAGIC, FILE_MAGIC_HEADER_SIZE) == 0;
  return ge::SUCCESS;
}

ge::Status IsOm2Model(const char *file_path, bool &is_support) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "Opening file %s failed!", file_path);
    return ge::FAILED;
  }
  uint8_t magic[FILE_MAGIC_HEADER_SIZE] = {};
  file.read(reinterpret_cast<char *>(magic), FILE_MAGIC_HEADER_SIZE);
  const auto read_len = static_cast<size_t>(file.gcount());

  return IsOm2Model(magic, read_len, is_support);
}
}  // namespace gert
