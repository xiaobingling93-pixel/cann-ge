/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/graph_external_weight_manager.h"
#include <fstream>
#include "framework/common/util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "graph/ge_local_context.h"
#include "base/err_msg.h"

namespace ge {
void from_json(const nlohmann::json &j, FileConstantMeta &meta) {
  const auto hash_to_weight_file = j.find("hash_to_weight_file");
  if (hash_to_weight_file != j.end()) {
    meta.hash_to_weight_file = hash_to_weight_file->get<std::map<std::string, std::string>>();
  }
  const auto hash_to_weight_offset = j.find("hash_to_weight_offset");
  if (hash_to_weight_offset != j.end()) {
    meta.hash_to_weight_offset = hash_to_weight_offset->get<std::map<std::string, size_t>>();
  }
}

void to_json(nlohmann::json &j, const FileConstantMeta &meta) {
  j = nlohmann::json();
  j["hash_to_weight_file"] = meta.hash_to_weight_file;
  j["hash_to_weight_offset"] = meta.hash_to_weight_offset;
}

ExternalWeightManager::ExternalWeightManager(const uint64_t session_id) : session_id_(session_id) {
  std::string option_str = ExternalWeightManager::GetWeightPathFromOption();
  if(option_str.empty()){
    weight_path_ = FileConstantUtils::GetTmpWeightDir(mmGetPid(), session_id_);
  } else {
    weight_path_ = option_str;
  }
}

void ExternalWeightManager::Finalize() noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  loaded_external_weight_files_.clear();
  shard_info_to_fileconstant_info_.clear();
  const std::string tmp_weight_dir = FileConstantUtils::GetTmpWeightDir(mmGetPid(), session_id_);
  if (mmAccess(tmp_weight_dir.c_str()) == EN_OK) {
    (void) mmRmdir(tmp_weight_dir.c_str());
    GELOGI("Success to remove dir:%s", tmp_weight_dir.c_str());
  }
}

Status ExternalWeightManager::CreateWeightPath() {
  const std::lock_guard<std::mutex> lock(mutex_);
  GE_CHK_BOOL_RET_STATUS(CreateDirectory(weight_path_) == 0, FAILED, "Failed to create weight path.");
  return SUCCESS;
}

bool ExternalWeightManager::CheckAndSetWeightLoaded(const std::string &file_name, const uint32_t device_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto &iter = loaded_external_weight_files_.find(device_id);
  if ((iter == loaded_external_weight_files_.end()) || (iter->second.count(file_name) == 0U)) {
    (void)loaded_external_weight_files_[device_id].insert(file_name);
    return false;
  }
  GELOGI("Weight file[%s] has been loaded on device[%u], do not need load again", file_name.c_str(), device_id);
  return true;
}

void ExternalWeightManager::SetWeightPath(const std::string &weight_path) {
  const std::lock_guard<std::mutex> lock(mutex_);
  weight_path_ = weight_path;
}

const std::string &ExternalWeightManager::GetWeightPath() {
  const std::lock_guard<std::mutex> lock(mutex_);
  return weight_path_;
}

void ExternalWeightManager::SaveSlicedFileConstantInfo(const std::string &shard_info,
                                                       const FileConstantInfo &weight_info) {
  const std::lock_guard<std::mutex> lock(mutex_);
  shard_info_to_fileconstant_info_[shard_info] = weight_info;
}

bool ExternalWeightManager::TryGetSlicedFileConstantInfo(const std::string &shard_info, FileConstantInfo &weight_info) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (shard_info_to_fileconstant_info_.count(shard_info) > 0U) {
    weight_info = shard_info_to_fileconstant_info_.at(shard_info);
    return true;
  }
  return false;
}

std::string ExternalWeightManager::GetWeightPathFromOption(){
  std::string option_str;
  (void)GetThreadLocalContext().GetOption(OPTION_EXTERNAL_WEIGHT_DIR, option_str);
  return option_str;
}

ExternalWeightManagerPool &ExternalWeightManagerPool::Instance() {
  static ExternalWeightManagerPool external_weight_manager_pool;
  return external_weight_manager_pool;
}

ExternalWeightManagerPool::~ExternalWeightManagerPool() {
  Destroy();
}

void ExternalWeightManagerPool::Destroy() noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &session_and_manager : session_id_to_manager_) {
    if (session_and_manager.second != nullptr) {
      session_and_manager.second->Finalize();
    }
  }
  session_id_to_manager_.clear();
  GELOGI("Success to destroy external weight manager pool");
}

ExternalWeightManagerPtr ExternalWeightManagerPool::GetManager(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto &external_weight_manager = session_id_to_manager_[session_id];
  if (external_weight_manager == nullptr) {
    external_weight_manager = MakeShared<ExternalWeightManager>(session_id);
    if (external_weight_manager == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "New ExternalWeightManager fail, session_id:%" PRIu64 "", session_id);
      GELOGE(INTERNAL_ERROR, "[New][ExternalWeightManager] fail, session_id:%" PRIu64 "", session_id);
      return nullptr;
    }
  }
  GELOGD("Success to get external weight manager, session id:%" PRIu64 "", session_id);
  return external_weight_manager;
}

void ExternalWeightManagerPool::RemoveManager(const uint64_t session_id) {
  ExternalWeightManagerPtr external_weight_manager = nullptr;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto &session_and_manager = session_id_to_manager_.find(session_id);
    if (session_and_manager != session_id_to_manager_.end()) {
      external_weight_manager = session_and_manager->second;
      (void) session_id_to_manager_.erase(session_and_manager);
      GELOGI("Success to remove external weight manager, session id:%" PRIu64 "", session_id);
    }
  }
  if (external_weight_manager != nullptr) {
    external_weight_manager->Finalize();
  }
}
}  // namespace ge
