/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_bin_handler.h"

#include "mmpa/mmpa_api.h"
#include "common/checker.h"
#include "graph/ge_error_codes.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "framework/common/debug/log.h"
#include "runtime/dev.h"
#include "runtime/context.h"

 
using namespace ge;

namespace gert {
namespace {
constexpr size_t kSocVersionLen = 128U;
constexpr int32_t kCpuKernelModeJson = 0;
constexpr int32_t kCpuKernelModeData = 2;
const std::string kAicpuBuiltInCustKernelFile = "/built-in/op_impl/aicpu/kernel/";
}

bool OpJsonBinHandler::is_support_ = true;
std::once_flag OpJsonBinHandler::init_flag_;

bool OpJsonBinHandler::IsSupportBinHandle()
{
  std::call_once(init_flag_,
    []() {
      char soc_version[kSocVersionLen] = {0};
      if (rtGetSocVersion(soc_version, kSocVersionLen) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::FAILED, "Get soc version failed.");
        return;
      }

      if ((strncmp(soc_version, "Ascend950", (sizeof("Ascend950") - 1UL)) == 0)
        || (strncmp(soc_version, "Ascend910_96", (sizeof("Ascend910_96") - 1UL)) == 0)) {
        is_support_ = false;
      }
      GELOGI("Init bin handle support status, val=%u, socVersion=%s", static_cast<uint32_t>(is_support_), soc_version);
    }
  );
  return is_support_;
}

ge::graphStatus OpJsonBinHandler::LoadBinary(const std::string &json_path) {
  std::unique_lock lock(bin_mutex_);
  if (bin_handle_ != nullptr) {
    GELOGI("Load binary from json success, not need to load again. json_path=%s", json_path.c_str());
    return ge::GRAPH_SUCCESS;
  }

  if (json_path.empty()) {
    GELOGE(ge::FAILED, "Load binary failed by json path is empty.");
    return ge::FAILED; 
  }

  rtLoadBinaryOption_t bin_option = {};
  bin_option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
  bin_option.value.cpuKernelMode = kCpuKernelModeJson;
  const rtLoadBinaryConfig_t load_bin_cfg = {&bin_option, 1U};
  GE_ASSERT_SUCCESS(rtsBinaryLoadFromFile(json_path.c_str(), &load_bin_cfg, &bin_handle_));
  GE_ASSERT_NOTNULL(bin_handle_);

  GELOGI("Load json binary success for %s.", json_path.c_str());

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpDataBinHandler::LoadBinary(const std::string &so_name, const ge::OpKernelBinPtr &kernel_bin)
{
  std::unique_lock lock(bin_mutex_);
    if (bin_handle_ != nullptr) {
    GELOGI("Load binary from data success, not need to load again. so=%s", so_name.c_str());
    return ge::GRAPH_SUCCESS;
  }

  if (kernel_bin == nullptr) {
      GELOGE(ge::FAILED, "Load custom aicpu binary failed by nullptr. so=%s", so_name.c_str());
      return ge::FAILED; 
  }

  if (so_name.empty()) {
    GELOGE(ge::FAILED, "Load custom aicpu binary failed by so name is empty.");
    return ge::FAILED; 
  }

  rtLoadBinaryOption_t bin_option = {};
  bin_option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
  bin_option.value.cpuKernelMode = kCpuKernelModeData;
  const rtLoadBinaryConfig_t load_bin_cfg = {&bin_option, 1U};
  GE_ASSERT_SUCCESS(rtsBinaryLoadFromData(kernel_bin->GetBinData(), kernel_bin->GetBinDataSize(),
                                          &load_bin_cfg, &bin_handle_));
  GE_ASSERT_NOTNULL(bin_handle_);

  GELOGI("Load data binary success for %s.", so_name.c_str());

  return ge::GRAPH_SUCCESS;
}

TfJsonBinHandler &TfJsonBinHandler::Instance() {
  static TfJsonBinHandler inst;
  return inst;
}

AicpuJsonBinHandler &AicpuJsonBinHandler::Instance() {
  static AicpuJsonBinHandler inst;
  return inst;
}

CustBinHandlerManager &CustBinHandlerManager::Instance()
{
  static CustBinHandlerManager inst;
  return inst;
}

ge::graphStatus CustBinHandlerManager::LoadAndGetBinHandle(const std::string &so_name,
  const ge::OpKernelBinPtr &kernel_bin, rtBinHandle &handle) {
  handle = nullptr;
  if ((so_name.empty()) || (kernel_bin == nullptr)) {
    GELOGE(ge::FAILED, "Load bin failed by so name is empty or bin is nullptr, so_name=%s", so_name.c_str());
    return ge::FAILED;
  }

  rtContext_t current_ctx = nullptr;
  GE_CHK_RT_RET(aclrtGetCurrentContext(&current_ctx));
  const uintptr_t resource_id = reinterpret_cast<uintptr_t>(current_ctx);

  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &so_bin_maps = bin_manager_.find(resource_id);
  if (so_bin_maps == bin_manager_.end()) {
    OpDataBinHandlerPtr bin_handler = std::make_shared<OpDataBinHandler>();
    GE_ASSERT_NOTNULL(bin_handler);
    GE_ASSERT_SUCCESS(bin_handler->LoadBinary(so_name, kernel_bin));
    bin_manager_.emplace(resource_id, std::unordered_map<std::string, OpDataBinHandlerPtr>{{so_name, bin_handler}});
    handle = bin_handler->GetBinHandle();
    GELOGI("Add resource and bin handle success, resource_id=%lu, so=%s, size=%lu",
      resource_id, so_name.c_str(), bin_manager_.size());
    return ge::GRAPH_SUCCESS;
  }

  const auto &iter = so_bin_maps->second.find(so_name);
  if (iter == so_bin_maps->second.end()) {
    OpDataBinHandlerPtr bin_handler = std::make_shared<OpDataBinHandler>();
    GE_ASSERT_NOTNULL(bin_handler);
    GE_ASSERT_SUCCESS(bin_handler->LoadBinary(so_name, kernel_bin));
    so_bin_maps->second.emplace(so_name, bin_handler);
    handle = bin_handler->GetBinHandle();
    GELOGI("Add bin handle for so %s success, resource_id=%lu, size=%lu",
      so_name.c_str(), resource_id, so_bin_maps->second.size());
    return ge::GRAPH_SUCCESS;
  }

  GELOGI("Bin handle for so %s has exist, will not repeat load. resource_id=%lu", so_name.c_str(), resource_id);
  handle = iter->second->GetBinHandle();

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustBinHandlerManager::GetBinHandle(const std::string &so_name, rtBinHandle &handle) {
  handle = nullptr;
  if (so_name.empty()) {
    GELOGE(ge::FAILED, "Get bin handle failed by empty so name, so_name=%s", so_name.c_str());
    return ge::FAILED;
  }

  rtContext_t current_ctx = nullptr;
  GE_CHK_RT_RET(aclrtGetCurrentContext(&current_ctx));
  const uintptr_t resource_id = reinterpret_cast<uintptr_t>(current_ctx);

  const std::lock_guard<std::recursive_mutex> lk(mutex_);
  const auto &so_bin_maps = bin_manager_.find(resource_id);
  if (so_bin_maps == bin_manager_.end()) { // no resource id
    ge::OpKernelBinPtr kernel_bin = GetKernelBin(so_name);
    GE_ASSERT_NOTNULL(kernel_bin);
    (void)LoadAndGetBinHandle(so_name, kernel_bin, handle);
    GELOGI("Bin handle not find in resource id, but read so success, resource_id=%lu, so=%s, size=%lu",
      resource_id, so_name.c_str(), bin_manager_.size());
    return ge::SUCCESS;
  }

  const auto &iter = so_bin_maps->second.find(so_name);
  if (iter == so_bin_maps->second.end()) {
    ge::OpKernelBinPtr kernel_bin = GetKernelBin(so_name);
    GE_ASSERT_NOTNULL(kernel_bin);
    (void)LoadAndGetBinHandle(so_name, kernel_bin, handle);
    GELOGI("Bin handle in current resource id, but read so success, resource_id=%lu, so=%s, size=%lu",
      resource_id, so_name.c_str(), so_bin_maps->second.size());
    return ge::GRAPH_SUCCESS;
  }

  GELOGI("Get bind handle success, resource_id=%lu, so=%s", resource_id, so_name.c_str());
  handle = iter->second->GetBinHandle();

  return ge::GRAPH_SUCCESS;
}

ge::OpKernelBinPtr CustBinHandlerManager::GetKernelBin(const std::string &so_name) {
  const auto &iter = kernel_manager_.find(so_name);
  if (iter == kernel_manager_.end()) {
    ge::OpKernelBinPtr kernel_bin = nullptr;
    (void)GetCustAicpuBinFromFile(so_name, kernel_bin);
    if (kernel_bin == nullptr) {
      GELOGI("Get cust aicpu bin from model, so=%s", so_name.c_str());
      (void)ModelManager::GetInstance().GetCustAicpuSo(so_name, kernel_bin);
      return kernel_bin;
    }

    kernel_manager_.emplace(so_name, kernel_bin);
    return kernel_bin;
  }

  return iter->second;
}

bool CustBinHandlerManager::GetCustAicpuBinFromFile(const std::string &so_name, ge::OpKernelBinPtr &kernel_bin) {
  const std::string folder_path = GetCustSoFolderPath();
  if (folder_path.empty()) {
    GELOGI("The folder path is empty, will not read");
    return true;
  }

  const std::string file_path = folder_path + so_name;
  const std::string real_path = ge::RealPath(file_path.c_str());
  if (real_path.empty()) {
    GELOGI("The file[%s] is not exist, will not read.", file_path.c_str());
    return true;
  }

  uint32_t len = 0U;
  std::unique_ptr<char_t[]> bin_data = ge::GetBinDataFromFile(real_path, len);
  if (bin_data == nullptr) {
    GELOGE(ge::FAILED, "Read cust so[%s] failed", file_path.c_str());
    return false;
  }

  std::vector<char_t> buffer(bin_data.get(), bin_data.get()+len);
  kernel_bin = std::make_shared<ge::OpKernelBin>(so_name, std::move(buffer));
  if (kernel_bin == nullptr) {
    GELOGE(ge::FAILED, "Malloc memory for cust so failed, so=%s, len=%u", file_path.c_str(), len);
    return false;
  }

  GELOGI("Read so[%s] success, len is %u", so_name.c_str(), len);
  return true;
}

std::string CustBinHandlerManager::GetCustSoFolderPath() const {
  std::string dir_path;
  const char *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  if (path_env != nullptr) {
    dir_path = path_env;
    if (dir_path[dir_path.size() - 1UL] != '/') {
      dir_path += "/";
    }

    dir_path += kAicpuBuiltInCustKernelFile;
  }

  GELOGI("Get cust so path is %s", dir_path.c_str());

  return dir_path;
}
} // namespace gert