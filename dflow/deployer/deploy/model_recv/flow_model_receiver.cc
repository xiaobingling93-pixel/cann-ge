/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/model_recv/flow_model_receiver.h"
#include <cstdio>
#include <cstring>
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "dflow/base/utils/process_utils.h"

namespace ge {
namespace {
constexpr size_t kMaxReceivingFiles = 1024U;
constexpr size_t kMaxRootModelSize = 64U;
constexpr int32_t kMaxSubmodelDescSize = 1024;  // max head node processes * pg * submodels
constexpr size_t kMaxFlowRoutePlanSize = 64U;
}  // namespace

Status FlowModelReceiver::UpdateDeployPlan(const deployer::UpdateDeployPlanRequest &request) {
  std::lock_guard<std::mutex> lk(mu_);
  uint32_t root_model_id = request.root_model_id();
  uint32_t graph_id = request.graph_id();
  auto &deploy_state = deploy_states_[root_model_id];
  GE_CHECK_LE(deploy_states_.size(), kMaxRootModelSize);
  deploy_state.SetRootModelId(root_model_id);
  deploy_state.SetGraphId(graph_id);
  deploy_state.SetSessionId(request.session_id());
  deploy_state.SetOptions(request.options());
  GE_CHECK_LE(request.submodel_descs_size(), kMaxSubmodelDescSize);
  GE_CHECK_GE(request.device_id(), 0);
  for (auto &submodel_desc : request.submodel_descs()) {
    deploy_state.AddLocalSubmodelDesc(request.device_id(), request.device_type(), submodel_desc);
  }
  GELOGI("DeployPlan updated, session_id = %lu, root_model_id = %u", request.session_id(), root_model_id);
  return SUCCESS;
}

Status FlowModelReceiver::AddFlowRoutePlan(const deployer::AddFlowRoutePlanRequest &request) {
  std::lock_guard<std::mutex> lk(mu_);
  uint32_t root_model_id = request.root_model_id();
  GE_CHECK_LE(deploy_states_.size() + 1, kMaxRootModelSize);
  auto &deploy_state = deploy_states_[root_model_id];
  deploy_state.SetRootModelId(root_model_id);
  deploy_state.SetLocalFlowRoutePlan(request.node_id(), request.flow_route_plan());
  GELOGI("Add FlowRoutePlan success, root_model_id = %u, node = %d",
         root_model_id, request.node_id());
  return SUCCESS;
}

Status FlowModelReceiver::GetDeployState(uint32_t root_model_id, DeployState *&deploy_state) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto &it = deploy_states_.find(root_model_id);
  if (it == deploy_states_.end()) {
    GELOGE(FAILED, "DeployPlan does not exist, root_model_id = %u", root_model_id);
    return FAILED;
  }
  deploy_state = &it->second;
  return SUCCESS;
}

Status FlowModelReceiver::AddDataGwSchedInfos(const deployer::DataGwSchedInfos &request) {
  std::lock_guard<std::mutex> lk(mu_);
  uint32_t root_model_id = request.root_model_id();
  GE_CHECK_LE(deploy_states_.size() + 1, kMaxRootModelSize);
  auto &deploy_state = deploy_states_[root_model_id];
  deploy_state.SetRootModelId(root_model_id);
  deploy_state.SetIsDynamicSched(request.is_dynamic_sched());
  GE_CHECK_GE(request.device_id(), 0);
  DeployPlan::DeviceInfo device_info(request.device_type(), 0, request.device_id());
  deploy_state.AddDataGwSchedInfos(device_info, request);
  GE_CHECK_LE(deploy_state.GetDataGwSchedInfos().size(), kMaxFlowRoutePlanSize);
  GELOGI("Add DataGwSchedInfos success, root_model_id = %u, device = %s",
         root_model_id, device_info.GetKey().c_str());
  return SUCCESS;
}

Status FlowModelReceiver::AppendToFile(const std::string &path, const char_t *file_content, size_t size, bool is_eof) {
  GE_CHK_STATUS_RET(ProcessUtils::IsValidPath(path), "File path[%s] is invalid.", path.c_str());
  std::ofstream *output_file = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = receiving_files_.find(path);
    if (it == receiving_files_.end()) {
      GE_CHECK_LE(receiving_files_.size(), kMaxReceivingFiles);
      auto dir = GetDirectory(path);
      constexpr auto dir_mode =
          static_cast<uint32_t>(M_UMASK_USRREAD | M_UMASK_USRWRITE | M_UMASK_USREXEC | M_UMASK_GRPEXEC);
      GE_CHK_BOOL_RET_STATUS(
          (mmAccess2(dir.c_str(), M_F_OK) == EN_OK) || (ProcessUtils::CreateDir(dir, dir_mode) == SUCCESS), FAILED,
          "Failed to create directory: %s", dir.c_str());
      constexpr mmMode_t kAccess = M_IRUSR | M_IWUSR;
      constexpr int32_t kOpenFlags = M_WRONLY | M_CREAT |O_TRUNC;
      const int32_t fd = mmOpen2(path.c_str(), kOpenFlags, kAccess);
      GE_CHK_BOOL_RET_STATUS(fd >= 0, FAILED, "Failed to open file, path = %s", path.c_str());
      (void) mmClose(fd);
      auto receiving_file_stream = MakeUnique<std::ofstream>(path, std::ios::out | std::ios::binary);
      GE_CHECK_NOTNULL(receiving_file_stream);
      GE_CHK_BOOL_RET_STATUS(receiving_file_stream->good(), FAILED, "Failed to open file for write, path = %s",
                             path.c_str());
      output_file = receiving_file_stream.get();
      receiving_files_[path] = std::move(receiving_file_stream);
    } else {
      output_file = it->second.get();
    }
  }

  GE_DISMISSABLE_GUARD(file_guard, [&path]() {
    (void) std::remove(path.c_str());
  });
  output_file->write(file_content, static_cast<std::streamsize>(size));
  GE_CHK_BOOL_RET_STATUS(!output_file->fail(), FAILED,
                         "Failed to write to file[%s], size = %zu bytes, current file size = %ld bytes, "
                         "error msg = %s. If there  is not enough space, "
                         "you can specify another path through the configuration file.",
                         path.c_str(), size, static_cast<int64_t>(output_file->tellp()), strerror(errno));
  GELOGD("[TransferFile] success, path = %s, block_size = %zu bytes, current_size = %ld bytes, is_eof = %d",
         path.c_str(), size, static_cast<int64_t>(output_file->tellp()), static_cast<int32_t>(is_eof));
  if (is_eof) {
    // auto flush and close
    GEEVENT("[TransferFile] success, path = %s, file_size = %ld bytes.", path.c_str(),
            static_cast<int64_t>(output_file->tellp()));
    std::lock_guard<std::mutex> lk(mu_);
    receiving_files_.erase(path);
  }
  GE_DISMISS_GUARD(file_guard);
  return SUCCESS;
}

std::string FlowModelReceiver::GetDirectory(const std::string &file_path) const {
  auto pos = file_path.rfind('/');
  if (pos == std::string::npos) {
    return "";
  }
  return file_path.substr(0, pos);
}

void FlowModelReceiver::DestroyDeployState(uint32_t model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  deploy_states_.erase(model_id);
}

void FlowModelReceiver::DestroyAllDeployStates() {
  std::lock_guard<std::mutex> lk(mu_);
  deploy_states_.clear();
}
}  // namespace ge
