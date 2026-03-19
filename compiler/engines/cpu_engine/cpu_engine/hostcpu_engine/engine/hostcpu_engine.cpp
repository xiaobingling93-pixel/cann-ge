/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hostcpu_engine.h"
#include <mutex>
#include "common/util/log.h"
#include "platform/platform_info.h"
#include "common/util/util.h"

using fe::PlatformInfoManager;
using namespace ge;
namespace aicpu {

BaseEnginePtr HostCpuEngine::instance_ = nullptr;

BaseEnginePtr HostCpuEngine::Instance() {
  static std::once_flag flag;
  std::call_once(flag,
                 []() { instance_.reset(new (std::nothrow) HostCpuEngine); });
  return instance_;
}

ge::Status HostCpuEngine::Initialize(
    [[maybe_unused]] const std::map<std::string, std::string> &options) {
  if (ops_kernel_info_store_ == nullptr) {
    ops_kernel_info_store_ =
        std::make_shared<AicpuOpsKernelInfoStore>(kHostCpuEngine);
    if (ops_kernel_info_store_ == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("Create HostCpuOpsKernelInfoStore object failed");
      return ge::FAILED;
    }
  }
  if (graph_optimizer_ == nullptr) {
    graph_optimizer_ = std::make_shared<AicpuGraphOptimizer>(kHostCpuEngine);
    if (graph_optimizer_ == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("Create HostCpuGraphOptimizer object failed");
      return ge::FAILED;
    }
  }
  if (ops_kernel_builder_ == nullptr) {
    ops_kernel_builder_ = std::make_shared<AicpuOpsKernelBuilder>(kHostCpuEngine);
    if (ops_kernel_builder_ == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("Create HostCpuOpsKernelBuilder object failed");
      return ge::FAILED;
    }
  }
  BaseEnginePtr (*instance_ptr)() = &HostCpuEngine::Instance;
  return LoadConfigFile(reinterpret_cast<void *>(instance_ptr));
}

void HostCpuEngine::GetOpsKernelInfoStores(
    std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_info_stores)
    const {
  if (ops_kernel_info_store_ != nullptr) {
    ops_kernel_info_stores[kHostCpuOpsKernelInfo] = ops_kernel_info_store_;
  }
}

void HostCpuEngine::GetGraphOptimizerObjs(
    std::map<std::string, GraphOptimizerPtr> &graph_optimizers) const {
  if (graph_optimizer_ != nullptr) {
    graph_optimizers[kHostCpuGraphOptimizer] = graph_optimizer_;
  }
}

void HostCpuEngine::GetOpsKernelBuilderObjs(
    std::map<std::string, OpsKernelBuilderPtr> &ops_kernel_builders) const {
  if (ops_kernel_builder_ != nullptr) {
    ops_kernel_builders[kHostCpuOpsKernelBuilder] = ops_kernel_builder_;
  }
}

ge::Status HostCpuEngine::Finalize() {
  ops_kernel_info_store_ = nullptr;
  graph_optimizer_ = nullptr;
  ops_kernel_builder_ = nullptr;
  return ge::SUCCESS;
}

FACTORY_ENGINE_CLASS_KEY(HostCpuEngine, kHostCpuEngine)

}  // namespace aicpu

ge::Status Initialize(const std::map<std::string, std::string> &options) {
  AICPU_IF_BOOL_EXEC(
      PlatformInfoManager::Instance().InitializePlatformInfo() != ge::GRAPH_SUCCESS,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call fe::PlatformInfoManager::InitializePlatformInfo function failed");
      return ge::FAILED)
  return aicpu::HostCpuEngine::Instance()->Initialize(options);
}

void GetOpsKernelInfoStores(
    std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_info_stores) {
  aicpu::HostCpuEngine::Instance()->GetOpsKernelInfoStores(
      ops_kernel_info_stores);
}

void GetGraphOptimizerObjs(
    std::map<std::string, GraphOptimizerPtr> &graph_optimizers) {
  aicpu::HostCpuEngine::Instance()->GetGraphOptimizerObjs(graph_optimizers);
}

void GetOpsKernelBuilderObjs(
    std::map<std::string, OpsKernelBuilderPtr> &ops_kernel_builders) {
  aicpu::HostCpuEngine::Instance()->GetOpsKernelBuilderObjs(ops_kernel_builders);
}

ge::Status Finalize() {
  (void)PlatformInfoManager::Instance().Finalize();
  return aicpu::HostCpuEngine::Instance()->Finalize();
}
