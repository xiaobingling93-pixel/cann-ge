/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor_manager.h"
#include <vector>
#include <thread>
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
std::map<std::string, std::function<std::string(const ExecutorManager::ExecutorKey &)>>
    ExecutorManager::ExecutorKeyComparator::executor_get_key_funcs_ = {
      { PNE_ID_UDF, &ExecutorManager::UdfExecutorKeyGetter::GetKey }
    };

std::string ExecutorManager::ExecutorKeyGetter::GetKey(const ExecutorManager::ExecutorKey &executor_key) {
  std::string key = std::to_string(executor_key.device_id) + "_" +
                    std::to_string(executor_key.device_type) + "_" +
                    std::to_string(executor_key.context_id) + "_" +
                    executor_key.engine_name + "_" +
                    std::to_string(executor_key.process_id) + "_" +
                    executor_key.rank_id;
  return key;
}

std::string ExecutorManager::UdfExecutorKeyGetter::GetKey(const ExecutorManager::ExecutorKey &executor_key) {
  std::string key = std::to_string(executor_key.device_id) + "_" +
                    std::to_string(executor_key.device_type) + "_" +
                    std::to_string(executor_key.context_id) + "_" +
                    executor_key.engine_name;
  return key;
}

std::string ExecutorManager::ExecutorKeyComparator::GetKeyByEngine(const ExecutorManager::ExecutorKey &executor_key) {
  const auto &it = executor_get_key_funcs_.find(executor_key.engine_name);
  if (it != executor_get_key_funcs_.cend()) {
    return it->second(executor_key);
  }
  return ExecutorManager::ExecutorKeyGetter::GetKey(executor_key);
}

bool ExecutorManager::ExecutorKeyComparator::Compare(const ExecutorKey &lhs, const ExecutorKey &rhs) {
  return GetKeyByEngine(lhs) < GetKeyByEngine(rhs);
}

Status ExecutorManager::GetOrCreateExecutorClient(const ExecutorManager::ExecutorKey &executor_key,
                                                  const PneExecutorClient::ClientContext &client_context,
                                                  PneExecutorClient **client) {
  const auto &device_id = executor_key.device_id;
  const auto &context_id = executor_key.context_id;
  const auto &engine_name = executor_key.engine_name;
  const auto &rank_id = executor_key.rank_id;
  const auto &process_id = executor_key.process_id;
  {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = executor_clients_.find(executor_key);
    if (it != executor_clients_.end()) {
      *client = it->second.get();
      return SUCCESS;
    }
    GELOGI("Executor process does not exist, start to create context_id = %ld, "
           "device id = %d, device_type = %d, engine_name = %s, rank_id = %s, process_id = %d",
           context_id, device_id, executor_key.device_type, engine_name.c_str(), rank_id.c_str(), process_id);
    std::unique_ptr<PneExecutorClient>
        executor_client = PneExecutorClientFactory::GetInstance().CreateClient(engine_name,
                                                                               executor_key.is_proxy,
                                                                               device_id);
    GE_CHECK_NOTNULL(executor_client);
    executor_client->SetContext(client_context);
    executor_clients_[executor_key] = std::move(executor_client);
    *client = executor_clients_[executor_key].get();
  }

  GE_CHK_STATUS_RET((*client)->Initialize(), "Failed to init client");
  GELOGI("Executor process started successfully, context_id = %ld, "
         "device id = %d, device_type = %d, engine_name = %s, rank_id = %s, process_id = %d",
         context_id, device_id, executor_key.device_type, engine_name.c_str(), rank_id.c_str(), process_id);
  return SUCCESS;
}

Status ExecutorManager::GetExecutorClient(const ExecutorManager::ExecutorKey &executor_key,
                                          PneExecutorClient **client) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = executor_clients_.find(executor_key);
  if (it != executor_clients_.end()) {
    *client = it->second.get();
    return SUCCESS;
  }

  GELOGW("Executor process does not exist, context_id = %ld, device id = %d, "
         "engine name = %s, rank id = %s, process_id = %d",
         executor_key.context_id,
         executor_key.device_id,
         executor_key.engine_name.c_str(),
         executor_key.rank_id.c_str(),
         executor_key.process_id);
  return FAILED;
}

void ExecutorManager::ResponseErrorInfoFormat(deployer::DeployerResponse &response) const {
  response.mutable_heartbeat_response()->set_abnormal_type(kAbnormalTypeModelInstance);
  response.set_error_code(FAILED);
  response.set_error_message("Executor process abnormal.");
  return;
}

void ExecutorManager::GetExecutorStatus(deployer::DeployerResponse &response,
    std::map<ExecutorManager::ExecutorKey, bool> &abnormal_pids,
    std::map<uint32_t, std::vector<std::string>> &model_instance_name) {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto &executor_client : executor_clients_) {
    const auto status = executor_client.second->GetSubProcStat();
    if (status == ProcStatus::NORMAL) {
      continue;
    } else if (status == ProcStatus::EXITED) {
      const auto &key = executor_client.first;
      abnormal_pids.emplace(key, false);
      GELOGW("Subprocess[%d] exited, device id[%d], name[%s].", key.process_id, key.device_id,
          key.engine_name.c_str());
    } else if (status == ProcStatus::STOPPED) {
      const auto &key = executor_client.first;
      abnormal_pids.emplace(key, false);
      GELOGW("Subprocess[%d] stopped, device id[%d], name[%s].", key.process_id, key.device_id,
          key.engine_name.c_str());
    }
    executor_client.second->GetAbnormalModelInsName(model_instance_name);
  }

  if (!abnormal_pids.empty()) {
    ResponseErrorInfoFormat(response);
  }
}

void ExecutorManager::Finalize() {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::thread> threads;
  for (auto &executor_client : executor_clients_) {
    threads.emplace_back([&executor_client]() {
        GE_CHK_STATUS(executor_client.second->Finalize(), "Finalize failed");
    });
  }
  for (std::thread &th:threads) {
    if (th.joinable()) {
      th.join();
    }
  }
  executor_clients_.clear();
  GEEVENT("Executor manager finalize success.");
}
}  // namespace ge
