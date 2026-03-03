/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_executor_manager.h"
#include "common/debug/log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
const std::string kExecutorPluginFuncInitialize = "Initialize";
const std::string kExecutorPluginFuncGetExecutors = "GetOpsKernelInfoStores";
const std::string kExecutorPluginFuncFinalize = "Finalize";
const std::set<std::string> kHcclKernelInfoStoreNames = {"ops_kernel_info_hccl",
                                                         "ops_kernel_info_hccl_gradtune",
                                                         "hvd_ops_kernel_info"};
}  // namespace

std::string OpsKernelExecutorManager::GetExecutorPluginLibPaths(const std::map<std::string, std::string> &options) {
  const std::map<std::string, std::string>::const_iterator &iter = options.find(OPTION_EXEC_HCCL_FLAG);
  // libhcom_graph_adaptor.so need to be loaded when lowering on RT2
  if ((iter != options.cend()) && (iter->second == "1")) {
    return RealPath(GetModelPath().append("libhcom_graph_adaptor.so").c_str());
  }
  return "";
}

Status OpsKernelExecutorManager::Initialize(const std::map<std::string, std::string> &options) {
  options_ = options;
  const std::string &lib_paths = GetExecutorPluginLibPaths(options);
  if (lib_paths.empty()) {
    GELOGI("No library to load");
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(InitializePlugin(executor_plugin_, lib_paths), "Failed to initialize executor plugins");
  return SUCCESS;
}

Status OpsKernelExecutorManager::GetExecutor(const std::string &name, OpsKernelExecutor *&executor) {
  const std::lock_guard<std::mutex> lk(mu_);
  // ensure hccl executor plugin was initialized when needed
  if ((kHcclKernelInfoStoreNames.count(name) > 0U) && (hccl_executor_plugin_ == nullptr)) {
    auto hccl_executor_plugin = MakeUnique<PluginManager>();
    GE_CHECK_NOTNULL(hccl_executor_plugin);
    GE_CHK_STATUS_RET(InitializePlugin(*hccl_executor_plugin, GetHcclExecutorPluginLibPath()),
                      "Failed to initialize hccl executor plugins");
    hccl_executor_plugin_ = std::move(hccl_executor_plugin);
  }

  const std::map<std::string, OpsKernelExecutorPtr>::const_iterator it = executors_.find(name);
  if (it == executors_.cend()) {
    GELOGE(FAILED, "Failed to get executor, name = %s", name.c_str());
    return FAILED;
  }
  executor = it->second.get();
  return SUCCESS;
}

void OpsKernelExecutorManager::Finalize() {
  GELOGI("ge invoke ops kernel executor finalize.");
  (void) executor_plugin_.InvokeAll<Status>(kExecutorPluginFuncFinalize);
  if (hccl_executor_plugin_ != nullptr) {
    (void) hccl_executor_plugin_->InvokeAll<Status>(kExecutorPluginFuncFinalize);
    hccl_executor_plugin_.reset();
  }
}

std::string OpsKernelExecutorManager::GetHcclExecutorPluginLibPath() {
  const std::string path_base = GetModelPath();
  return path_base + "libhcom_executor.so";
}

Status OpsKernelExecutorManager::InitializePlugin(PluginManager &plugin_manager, const std::string &plugin_paths) {
  const std::vector<std::string> func_check_list =
      {kExecutorPluginFuncInitialize, kExecutorPluginFuncGetExecutors, kExecutorPluginFuncFinalize};
  GE_CHK_STATUS_RET(plugin_manager.LoadSo(plugin_paths, func_check_list),
                    "[Check][SoFile] not find any valid so file.");
  if (plugin_manager.InvokeAll<std::map<std::string, std::string> &, Status>(kExecutorPluginFuncInitialize, options_)
      != SUCCESS) {
    GELOGE(GE_OPS_GET_NO_VALID_SO, "[Invoke][OpsKernelInfo]PluginManager InvokeAll failed.");
    REPORT_INNER_ERR_MSG("E19999", "PluginManager InvokeAll failed.");
    return GE_OPS_GET_NO_VALID_SO;
  }

  std::map<std::string, OpsKernelExecutorPtr> new_executors;
  const auto ret = plugin_manager.InvokeAll<std::map<std::string, std::shared_ptr<OpsKernelExecutor>> &>(
      kExecutorPluginFuncGetExecutors, new_executors);
  GE_CHK_STATUS_RET(ret, "Failed to get OpsKernelExecutors");
  GE_CHK_STATUS_RET_NOLOG(CheckExecutors(new_executors));

  executors_.insert(new_executors.cbegin(), new_executors.cend());
  return SUCCESS;
}

Status OpsKernelExecutorManager::CheckExecutors(const std::map<std::string, OpsKernelExecutorPtr> &executors) {
  for (const auto &it : executors) {
    if (it.second == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Check][PluginPtr] OpsKernelExecutor key=%s is null", it.first.c_str());
      REPORT_INNER_ERR_MSG("E19999", "CheckPluginPtr OpsKernelExecutor key=%s is null", it.first.c_str());
      return FAILED;
    } else {
      GELOGI("Executor initialized, name = %s", it.first.c_str());
    }
  }
  return SUCCESS;
}
}  // namespace ge
