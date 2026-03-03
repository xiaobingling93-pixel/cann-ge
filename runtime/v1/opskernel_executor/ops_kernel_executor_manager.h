/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_EXECUTOR_OPSKERNEL_EXECUTOR_OPSKERNEL_EXECUTOR_MANAGER_H_
#define AIR_EXECUTOR_OPSKERNEL_EXECUTOR_OPSKERNEL_EXECUTOR_MANAGER_H_

#include <map>
#include <memory>
#include <mutex>
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

namespace ge {
using OpsKernelExecutor = OpsKernelInfoStore;
using OpsKernelExecutorPtr = std::shared_ptr<OpsKernelExecutor>;

class OpsKernelExecutorManager {
 public:
  ~OpsKernelExecutorManager() = default;
  static OpsKernelExecutorManager &GetInstance() {
    static OpsKernelExecutorManager instance;
    return instance;
  }

  Status Initialize(const std::map<std::string, std::string> &options);
  void Finalize();
  Status GetExecutor(const std::string &name, OpsKernelExecutor *&executor);

 private:
  OpsKernelExecutorManager() = default;
  static std::string GetHcclExecutorPluginLibPath();
  static std::string GetExecutorPluginLibPaths(const std::map<std::string, std::string> &options);
  static Status CheckExecutors(const std::map<std::string, OpsKernelExecutorPtr> &executors);
  Status InitializePlugin(PluginManager &plugin_manager, const std::string &plugin_paths);

  std::mutex mu_;
  std::map<std::string, std::string> options_;
  std::unique_ptr<PluginManager> hccl_executor_plugin_;  // load on-demand
  PluginManager executor_plugin_;
  std::map<std::string, OpsKernelExecutorPtr> executors_;
};
}  // namespace ge

#endif  // AIR_EXECUTOR_OPSKERNEL_EXECUTOR_OPSKERNEL_EXECUTOR_MANAGER_H_
