/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_DAEMON_EXECUTOR_MANAGER_H_
#define AIR_RUNTIME_DEPLOY_DAEMON_EXECUTOR_MANAGER_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "common/config/device_debug_config.h"
#include "deploy/execfwk/pne_executor_client.h"
#include "proto/deployer.pb.h"

namespace ge {
class ExecutorManager {
 public:
  struct ExecutorKey {
    int32_t device_id;
    int32_t device_type;
    int64_t context_id;
    std::string engine_name;
    std::string rank_id;
    int32_t process_id;
    bool is_proxy = false;

    bool operator < (const ExecutorKey &other) const {
      auto lhs = *this;
      auto rhs = other;
      if (lhs.rank_id.empty() && (!rhs.rank_id.empty())) {
        rhs.rank_id = "";
      }
      if ((!lhs.rank_id.empty()) && rhs.rank_id.empty()) {
        lhs.rank_id = "";
      }
      return ExecutorKeyComparator::Compare(lhs, rhs);
    }
  };

  class ExecutorKeyGetter {
   public:
    static std::string GetKey(const ExecutorKey &executor_key);
  };

  class UdfExecutorKeyGetter {
   public:
    static std::string GetKey(const ExecutorKey &executor_key);
  };

  class ExecutorKeyComparator {
   public:
    static bool Compare(const ExecutorKey &lhs, const ExecutorKey &rhs);
    static std::string GetKeyByEngine(const ExecutorKey &executor_key);
   private:
    static std::map<std::string, std::function<std::string(const ExecutorKey &)>> executor_get_key_funcs_;
  };

  Status GetOrCreateExecutorClient(const ExecutorManager::ExecutorKey &executor_key,
                                   const PneExecutorClient::ClientContext &client_context,
                                   PneExecutorClient **client);
  Status GetExecutorClient(const ExecutorManager::ExecutorKey &executor_key, PneExecutorClient **client);
  void GetExecutorStatus(deployer::DeployerResponse &response,
      std::map<ExecutorManager::ExecutorKey, bool> &abnormal_pids,
      std::map<uint32_t, std::vector<std::string>> &model_instance_name);
  void ResponseErrorInfoFormat(deployer::DeployerResponse &response) const;
  void Finalize();

 private:
  mutable std::mutex mu_;
  std::map<const ExecutorManager::ExecutorKey, std::unique_ptr<PneExecutorClient>> executor_clients_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_DAEMON_EXECUTOR_MANAGER_H_
