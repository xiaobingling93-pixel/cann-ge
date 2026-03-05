/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "daemon/daemon_client_manager.h"
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "base/err_msg.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/util.h"
#include "common/config/configurations.h"
#include "dflow/base/utils/process_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr int64_t kHeartbeatIntervalSec = 20;
constexpr int32_t kIpIndex = 1;
constexpr int32_t kPortIndex = 2;
constexpr size_t kAddressSize = 3UL;
constexpr size_t kMaxClientSize = 32UL;
}  // namespace

Status DaemonClientManager::Initialize() {
  if (running_) {
    GELOGE(INTERNAL_ERROR, "Repeat initialize");
    return INTERNAL_ERROR;
  }
  running_ = true;
  (void) UpdateJsonFile();
  evict_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_evict");
    while (running_) {
      std::unique_lock<std::mutex> lk(mu_cv_);
      running_cv_.wait_for(lk, std::chrono::seconds(kHeartbeatIntervalSec), [this] {
        return !running_;
      });
      EvictExpiredClients();
    }
  });
  return SUCCESS;
}

DaemonClientManager::~DaemonClientManager() {
  Finalize();
}

void DaemonClientManager::Finalize() {
  try {
    running_ = false;
    running_cv_.notify_all();
    if (evict_thread_.joinable()) {
      evict_thread_.join();
    }
    if (client_fd_ >= 0) {
      (void) mmClose(client_fd_);
      client_fd_ = -1;
    }
    DeleteAllClientInfo();
    {
      std::lock_guard<std::mutex> lk(mu_);
      clients_.clear();
    }
  } catch (const std::exception &e) {
    GELOGW("Exception caught during DaemonClientManager Finalize: %s", e.what());
  } catch (...) {
    GELOGW("Unknown exception caught during DaemonClientManager Finalize");
  }
}

Status DaemonClientManager::CreateAndInitClient(const std::string &peer_uri,
                                                const std::map<std::string, std::string> &deployer_envs,
                                                int64_t &client_id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (clients_.size() == kMaxClientSize) {
    REPORT_INNER_ERR_MSG("E19999", "Client size has reached the upper limit[%zu]", kMaxClientSize);
    GELOGE(FAILED, "[Create][Client]Client size has reached the upper limit[%zu]", kMaxClientSize);
    return FAILED;
  }
  int64_t new_client_id = client_id_gen_;
  auto new_client = CreateClient(new_client_id);
  GE_CHECK_NOTNULL(new_client);
  GE_CHK_STATUS_RET_NOLOG(new_client->Initialize(deployer_envs));
  clients_.emplace(new_client_id, std::move(new_client));
  ++client_id_gen_;
  client_id = new_client_id;
  GE_CHK_STATUS_RET_NOLOG(RecordClientInfo(new_client_id, peer_uri));
  GELOGD("Client added, id = %ld", client_id);
  return SUCCESS;
}

Status DaemonClientManager::CloseClient(const int64_t client_id) {
  GELOGI("Close client begin.");
  std::unique_ptr<DeployerDaemonClient> client;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = clients_.find(client_id);
    if (it == clients_.cend()) {
      REPORT_INNER_ERR_MSG("E19999", "Client[%ld] does not exist in client manager.", client_id);
      GELOGE(FAILED, "[Close][Client]Client[%ld] does not exist in client manager.", client_id);
      return FAILED;
    }

    client = std::move(it->second);
    clients_.erase(it);
    (void)DeleteClientInfo(client_id);
  }
  (void)client->Finalize();
  return SUCCESS;
}

DeployerDaemonClient *DaemonClientManager::GetClient(const int64_t client_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = clients_.find(client_id);
  if (it == clients_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Get client[%ld] failed.", client_id);
    GELOGE(FAILED, "[Get][Client]Get client[%ld] failed.", client_id);
    return nullptr;
  }
  return it->second.get();
}

void DaemonClientManager::GenDgwPortOffset(const int32_t dev_count, int32_t &offset) {
  std::lock_guard<std::mutex> lk(mu_);
  offset = dgw_port_offset_gen_;
  dgw_port_offset_gen_ += dev_count;
}

Status DaemonClientManager::GetClientIpAndPort(const std::string &uri,
                                               ClientAddr &client) {
  std::vector<std::string> address = StringUtils::Split(uri, ':');
  if ((address.size() == kAddressSize) && (!address[kIpIndex].empty()) &&
      (!address[kPortIndex].empty())) {
    client.ip = address[kIpIndex];
    client.port = address[kPortIndex];
    return SUCCESS;
  }
  GELOGE(FAILED, "[Get][Client]Get client[%s] ip and port failed.", uri.c_str());
  return FAILED;
}

Status DaemonClientManager::RecordClientInfo(const int64_t client_id, const std::string &peer_uri) {
  ClientAddr client;
  GE_CHK_STATUS_RET_NOLOG(GetClientIpAndPort(peer_uri, client));
  client_addrs_.emplace(client_id, client);
  GELOGI("Add connection[%s] success.", peer_uri.c_str());
  GE_CHK_STATUS_RET_NOLOG(UpdateJsonFile());
  return SUCCESS;
}

Status DaemonClientManager::UpdateJsonFile() {
  const auto &dir = Configurations::GetInstance().GetDeployResDir();
  const std::string kClientFile = dir + "client.json";
  try {
    nlohmann::json json;
    for (const auto &client : client_addrs_) {
      nlohmann::json client_addr = nlohmann::json{{"ip", client.second.ip}, {"port", client.second.port}};
      json["connections"].push_back(client_addr);
    }
    if (client_fd_ < 0) {
      GE_CHK_BOOL_RET_STATUS(mmAccess2(dir.c_str(), M_F_OK) == EN_OK ||
                            ProcessUtils::CreateDir(dir) == SUCCESS,
                            FAILED,
                            "Failed to create directory: %s", dir.c_str());
      const mmMode_t kAccess = static_cast<mmMode_t>(M_IRUSR | M_IWUSR);
      client_fd_ = mmOpen2(kClientFile.c_str(), static_cast<int32_t>(M_WRONLY | M_CREAT | O_TRUNC), kAccess);
      if (client_fd_ < 0) {
        int32_t error_code = mmGetErrorCode();
        GELOGE(FAILED, "Open %s failed, ret = %d, error = %d(%s).", kClientFile.c_str(), client_fd_, error_code,
              GetErrorNumStr(error_code).c_str());
        return FAILED;
      }
    }
    std::ofstream file(kClientFile);
    if (!client_addrs_.empty()) {
      file << json << std::endl;
    }
    GELOGI("Update %s success.", kClientFile.c_str());
  } catch (const std::exception &e) {
    GELOGE(FAILED, "Update %s failed, error = %s.", kClientFile.c_str(), e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status DaemonClientManager::DeleteClientInfo(const int64_t client_id) {
  auto it = client_addrs_.find(client_id);
  if (it == client_addrs_.end()) {
    return SUCCESS;
  }
  client_addrs_.erase(it);
  GE_CHK_STATUS_RET_NOLOG(UpdateJsonFile());
  GELOGI("Delete client[%ld] success.", client_id);
  return SUCCESS;
}

void DaemonClientManager::DeleteAllClientInfo() {
  GELOGI("DeleteAllClientInfo begin.");
  std::lock_guard<std::mutex> lk(mu_);
  if (client_addrs_.empty()) {
    return;
  }
  client_addrs_.clear();
  (void) UpdateJsonFile();
}

void DaemonClientManager::EvictExpiredClients() {
  std::vector<int64_t> expired_clients;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &it : clients_) {
      if (it.second->IsExpired()) {
        if (it.second->IsExecuting()) {
          GELOGD("Client is still executing, check next time, client_id = %ld.", it.first);
        } else {
          GELOGW("Client is not executing, client_id = %ld.", it.first);
          expired_clients.push_back(it.first);
        }
      }
    }
  }
  for (int64_t client_id : expired_clients) {
    GEEVENT("Client expired, close it, client_id = %ld.", client_id);
    (void) CloseClient(client_id);
  }
}
} // namespace ge
