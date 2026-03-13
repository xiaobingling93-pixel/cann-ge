/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "graph/ge_global_options.h"
#include "exec_runtime/execution_runtime.h"

using namespace testing;

namespace ge {
class ExchangeServiceStub : public ExchangeService {
 public:
  MOCK_METHOD4(CreateQueue, Status(int32_t device_id, const string &name, const MemQueueAttr &mem_queue_attr,
                                   uint32_t &queue_id));
  MOCK_METHOD5(Enqueue, Status(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                               const ControlInfo &control_info));
  MOCK_METHOD5(Enqueue, Status(int32_t device_id, uint32_t queue_id, size_t size, const FillFunc &fill_func,
                               const ControlInfo &control_info));
  MOCK_METHOD4(Enqueue, Status(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                               const ControlInfo &control_info));
  MOCK_METHOD2(DestroyQueue, Status(int32_t device_id, uint32_t queue_id));
  MOCK_METHOD5(Enqueue, Status(int32_t device_id, uint32_t queue_id, const void *data, size_t size,
                               const ControlInfo &control_info));
  MOCK_METHOD4(EnqueueMbuf, Status(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout));
  MOCK_METHOD4(DequeueTensor, Status(int32_t, uint32_t, GeTensor & , ExchangeService::ControlInfo &));
  MOCK_METHOD4(DequeueMbuf, Status(int32_t, uint32_t, rtMbufPtr_t *, int32_t));
  MOCK_METHOD2(ResetQueueInfo, void(const int32_t device_id, const uint32_t queue_id));
  MOCK_METHOD5(DequeueMbufTensor, Status(const int32_t device_id, const uint32_t queue_id,
                                         std::shared_ptr<AlignedPtr> &aligned_ptr,
                                         const size_t size, ControlInfo &control_info));
  MOCK_METHOD5(Dequeue, Status(int32_t, uint32_t, void *, size_t, ExchangeService::ControlInfo &));
};

class ModelDeployerStub : public ModelDeployer {
 public:
  MOCK_METHOD2(DeployModel, Status(const FlowModelPtr &flow_model, DeployResult &deploy_result));
  MOCK_METHOD1(Undeploy, Status(uint32_t model_id));
  MOCK_METHOD2(GetDeviceMeshIndex, Status(const int32_t device_id, std::vector<int32_t> &node_mesh_index));
  MOCK_METHOD1(GetValidLogicDeviceId, Status(std::string &device_id));
};

class ExecutionRuntimeStub : public ExecutionRuntime {
 public:
  ExecutionRuntimeStub()
      :ExecutionRuntimeStub(std::make_shared<ExchangeServiceStub>(), std::make_shared<ModelDeployerStub>()) {}

  explicit ExecutionRuntimeStub(std::shared_ptr<ExchangeService> exchange_service)
      : ExecutionRuntimeStub(exchange_service, std::make_shared<ModelDeployerStub>()) {}

  explicit ExecutionRuntimeStub(std::shared_ptr<ModelDeployer> model_deployer)
      : ExecutionRuntimeStub(std::make_shared<ExchangeServiceStub>(), model_deployer) {}

  ExecutionRuntimeStub(std::shared_ptr<ExchangeService> exchange_service,
                       std::shared_ptr<ModelDeployer> model_deployer)
      : exchange_service_(exchange_service), model_deployer_(model_deployer) {}

  Status Initialize(const map<std::string, std::string> &options) override {
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
    return 0;
  }

  Status Finalize() override {
    return 0;
  }

  ModelDeployer &GetModelDeployer() override {
    return *model_deployer_;
  }

  ExchangeService &GetExchangeService() override {
    return *exchange_service_;
  }

  ModelDeployerStub &GetModelDeployerStub() {
    auto stub = std::dynamic_pointer_cast<ModelDeployerStub>(model_deployer_);
    if (stub == nullptr) {
      std::cout << "model deployer already seted, reset to stub." << std::endl;
      model_deployer_ = std::make_shared<ModelDeployerStub>();
      stub = std::dynamic_pointer_cast<ModelDeployerStub>(model_deployer_);
    }
    return *stub;
  }

  ExchangeServiceStub &GetExchangeServiceStub() {
    auto stub = std::dynamic_pointer_cast<ExchangeServiceStub>(exchange_service_);
    if (stub == nullptr) {
      std::cout << "exchange service already seted, reset to stub." << std::endl;
      exchange_service_ = std::make_shared<ExchangeServiceStub>();
      stub = std::dynamic_pointer_cast<ExchangeServiceStub>(exchange_service_);
    }
    return *stub;
  }

 public:
  std::shared_ptr<ExchangeService> exchange_service_;
  std::shared_ptr<ModelDeployer> model_deployer_;
};

Status InitializeExecutionRuntimeStub(const std::map<std::string, std::string> &options = {});
}  // namespace ge
