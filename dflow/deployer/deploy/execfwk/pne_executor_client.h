/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_PNE_EXECUTOR_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_PNE_EXECUTOR_CLIENT_H_

#include "common/config/device_debug_config.h"
#include "proto/deployer.pb.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/subprocess/subprocess_manager.h"
#include "common/message_handle/message_client.h"
#include "framework/common/ge_types.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
class PneExecutorClient {
 public:
  struct ClientContext {
    int32_t device_id = 0;
    int32_t device_type = 0;
    int32_t process_id = 0;
    int32_t deployer_pid = -1;
    const DeviceMaintenanceClientCfg *dev_maintenance_cfg = nullptr;
    std::string base_dir;
    std::map<std::string, std::string> options;
  };

  explicit PneExecutorClient(int32_t device_id);
  virtual ~PneExecutorClient() = default;

  virtual Status Initialize() = 0;
  virtual Status Finalize() = 0;
  virtual Status SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) = 0;
  virtual bool SupportSyncVarManager();
  virtual Status PreProcess(const std::vector<deployer::SubmodelDesc> &model_descs,
                            const std::string &base_dir) {
    (void) model_descs;
    (void) base_dir;
    return SUCCESS;
  }
  virtual Status LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) = 0;
  virtual Status UnloadModel(uint32_t model_id) = 0;
  virtual ProcStatus GetSubProcStat() = 0;
  virtual void GetAbnormalModelInsName(std::map<uint32_t, std::vector<std::string>> &abnormal_model_instances_name) = 0;
  virtual Status ClearModelRunningData(uint32_t model_id, int32_t type, const std::set<int32_t> &device_ids) = 0;
  virtual Status DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) = 0;
  virtual Status UpdateProfilingFromExecutor(deployer::ExecutorRequest_UpdateProfRequest &prof_message) = 0;

  void SetContext(const ClientContext& context);

  int32_t GetDeviceId() const;

  int32_t GetProcessId() const;

  int32_t GetDeployerPid() const;

  const ClientContext &GetContext() const;

 protected:
  const DeviceMaintenanceClientCfg *GetDevMaintenanceCfg() const;
  Status GrantQueuesForProcess(int32_t use_queue_pid, int32_t use_queue_process_device_type,
                               const deployer::ExecutorRequest_ModelQueuesAttrs &model_queues_attrs);
  Status BindHostPid(const int32_t pid) const;
  virtual std::unique_ptr<ExecutorMessageClient> CreateExecutorMessageClient(int32_t device_id) {
    return MakeUnique<ExecutorMessageClient>(device_id);
  }
  virtual Status DoGrantQueues(int32_t pid, const std::vector<DeployQueueAttr> &queue_attrs) {
    (void) pid;
    (void) queue_attrs;
    return SUCCESS;
  }
  virtual Status DoBindHostPid(const int32_t pid) {
    (void) pid;
    return SUCCESS;
  }
  bool is_valid_ = true;
 private:

  int32_t device_id_ = 0;
  ClientContext context_;
};

using PneExecutorClientPtr = std::unique_ptr<PneExecutorClient>;

class PneExecutorClientFactory {
 public:
  static PneExecutorClientFactory &GetInstance();
  std::unique_ptr<PneExecutorClient> CreateClient(const std::string &engine_name,
                                                  bool is_proxy,
                                                  int32_t device_id);

  using CreateFunc = std::function<PneExecutorClientPtr(int32_t device_id)>;

  void RegisterCreateFunc(const std::string &engine_name,
                          bool is_proxy,
                          CreateFunc func);

 private:
  std::string GenerateClientKey(const std::string &engine_name, bool is_proxy) const;

  std::map<std::string, CreateFunc> create_funcs_;
};

template<typename T>
class PneExecutorClientCreatorRegistrar {
 public:
  explicit PneExecutorClientCreatorRegistrar(const std::string &engine_name) {
    PneExecutorClientCreatorRegistrar(engine_name, false);
  }

  PneExecutorClientCreatorRegistrar(const std::string &engine_name,
                                    bool is_proxy) {
    auto func = [](int32_t device_id) -> PneExecutorClientPtr  {
      return ::ge::MakeUnique<T>(device_id);
    };
    PneExecutorClientFactory::GetInstance().RegisterCreateFunc(engine_name,
                                                               is_proxy,
                                                               std::move(func));
  }
  ~PneExecutorClientCreatorRegistrar() = default;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_PNE_EXECUTOR_CLIENT_H_
