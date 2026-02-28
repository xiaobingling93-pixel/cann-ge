/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/execfwk/pne_executor_client.h"
#include "common/debug/ge_log.h"
#include "deploy/flowrm/flowgw_client.h"
#include "common/config/configurations.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/utils/rts_api_utils.h"
namespace ge {
namespace {
constexpr uint32_t kBindAllDevice = 0xffffffff;
}

PneExecutorClient::PneExecutorClient(int32_t device_id) : device_id_(device_id) {
}

void PneExecutorClient::SetContext(const ClientContext& context) {
  context_ = context;
}

const DeviceMaintenanceClientCfg *PneExecutorClient::GetDevMaintenanceCfg() const {
  return context_.dev_maintenance_cfg;
}

int32_t PneExecutorClient::GetDeviceId() const {
  return device_id_;
}

int32_t PneExecutorClient::GetProcessId() const {
  return context_.process_id;
}

int32_t PneExecutorClient::GetDeployerPid() const {
  return context_.deployer_pid;
}

const PneExecutorClient::ClientContext &PneExecutorClient::GetContext() const {
  return context_;
}

bool PneExecutorClient::SupportSyncVarManager() {
  return true;
}

Status PneExecutorClient::GrantQueuesForProcess(int32_t use_queue_pid, int32_t use_queue_process_device_type,
                                                const deployer::ExecutorRequest_ModelQueuesAttrs &model_queues_attrs) {
  std::vector<DeployQueueAttr> queue_attrs;
  bool need_bind_host_pid = false;
  for (const auto &input_queue : model_queues_attrs.input_queues_attrs()) {
    if (use_queue_process_device_type == input_queue.device_type() && input_queue.queue_id() != UINT32_MAX) {
      DeployQueueAttr queue_attr = {};
      queue_attr.queue_id = input_queue.queue_id();
      queue_attr.device_type = input_queue.device_type();
      queue_attr.device_id = input_queue.device_id();
      queue_attrs.emplace_back(queue_attr);
    }

    if ((use_queue_process_device_type == static_cast<int32_t>(CPU)) &&
        (use_queue_process_device_type != input_queue.device_type())) {
      need_bind_host_pid = true;
    }
  }

  for (const auto &output_queue : model_queues_attrs.output_queues_attrs()) {
    if (use_queue_process_device_type == output_queue.device_type() && output_queue.queue_id() != UINT32_MAX) {
      DeployQueueAttr queue_attr = {};
      queue_attr.queue_id = output_queue.queue_id();
      queue_attr.device_type = output_queue.device_type();
      queue_attr.device_id = output_queue.device_id();
      queue_attrs.emplace_back(queue_attr);
    }

    if ((use_queue_process_device_type == static_cast<int32_t>(CPU)) &&
        (use_queue_process_device_type != output_queue.device_type())) {
      need_bind_host_pid = true;
    }
  }
  GE_CHK_STATUS_RET(DoGrantQueues(use_queue_pid, queue_attrs), "Failed to grant queues, pid = %d.", use_queue_pid);
  if (need_bind_host_pid) {
    GE_CHK_STATUS_RET(DoBindHostPid(use_queue_pid), "Failed to bind host pid[%d].", use_queue_pid);
  }
  return SUCCESS;
}

Status PneExecutorClient::BindHostPid(const int32_t pid) const {
  rtBindHostpidInfo info{};
  info.cpType = RT_DEVDRV_PROCESS_USER;
  uint32_t host_pid = static_cast<uint32_t>(getpid());
  info.hostPid = host_pid;
  info.chipId = kBindAllDevice;
  info.len = static_cast<uint32_t>(RT_PROCESS_SIGN_LENGTH);
  const auto ret = memcpy_s(info.sign, static_cast<size_t>(RT_PROCESS_SIGN_LENGTH), &pid, sizeof(pid_t));
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "Failed to memcpy_s, ret = %d", ret);
  GE_CHK_RT_RET(rtBindHostPid(info));
  return SUCCESS;
}

PneExecutorClientFactory &PneExecutorClientFactory::GetInstance() {
  static PneExecutorClientFactory instance;
  return instance;
}

std::string PneExecutorClientFactory::GenerateClientKey(const std::string &engine_name,
                                                        bool is_proxy) const {
  std::string key = engine_name + "_" + std::to_string(static_cast<uint32_t>(is_proxy));
  return key;
}

void PneExecutorClientFactory::RegisterCreateFunc(const std::string &engine_name,
                                                  bool is_proxy,
                                                  PneExecutorClientFactory::CreateFunc func) {
  std::string key = GenerateClientKey(engine_name, is_proxy);
  create_funcs_[key] = std::move(func);
}

std::unique_ptr<PneExecutorClient> PneExecutorClientFactory::CreateClient(const std::string &engine_name,
                                                                          bool is_proxy,
                                                                          int32_t device_id) {
  std::string key = GenerateClientKey(engine_name, is_proxy);
  auto func = create_funcs_[key];
  if (func == nullptr) {
    GELOGE(UNSUPPORTED, "Unsupported client, engine type = %s, is_proxy = %d",
           engine_name.c_str(), is_proxy);
    return nullptr;
  }
  return func(device_id);
}
}  // namespace ge
