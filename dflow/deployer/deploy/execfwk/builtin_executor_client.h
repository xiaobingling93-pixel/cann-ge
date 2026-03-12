/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_BUILTIN_EXECUTOR_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_BUILTIN_EXECUTOR_CLIENT_H_

#include <atomic>
#include <thread>
#include "deploy/execfwk/pne_executor_client.h"
#include "common/message_handle/message_client.h"
#include "common/config/device_debug_config.h"

namespace ge {
class BuiltinExecutorClient : public PneExecutorClient {
 public:
  explicit BuiltinExecutorClient(int32_t device_id, bool is_host = false);
  ~BuiltinExecutorClient() override = default;
  Status Initialize() override;
  Status Finalize() override;
  Status LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) override;
  Status UnloadModel(uint32_t model_id) override;
  Status SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) override;
  ProcStatus GetSubProcStat() override;
  void GetAbnormalModelInsName(std::map<uint32_t, std::vector<std::string>> &abnormal_model_instances_name) override {
    (void) abnormal_model_instances_name;
  }
  Status ClearModelRunningData(uint32_t model_id, int32_t type, const std::set<int32_t> &device_ids) override;
  Status DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) override;
  Status UpdateProfilingFromExecutor(deployer::ExecutorRequest_UpdateProfRequest &prof_message) override;

 protected:
  virtual Status ForAndInit(int32_t device_id, std::unique_ptr<ExecutorMessageClient> &message_client);
  virtual void Shutdown();
  virtual Status DoForkChildProcess(int32_t device_id,
                                    uint32_t req_msg_queue_id,
                                    uint32_t rsp_msg_queue_id,
                                    const std::string &group_name);
  virtual Status GenerateKvArgs(std::map<std::string, std::string> &kv_args);
  virtual Status GetPidOwningIoQueues(int32_t &pid);
  virtual Status PreLoadProcess(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc) {
    (void) load_model_desc;
    return SUCCESS;
  }
  virtual Status AfterLoadProcess(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc) {
    (void) load_model_desc;
    return SUCCESS;
  }
  virtual pid_t GetPid() const { return pid_; }
  Status DoGrantQueues(int32_t pid, const std::vector<DeployQueueAttr> &queue_attrs) override;

  Status DoBindHostPid(const int32_t pid) override;

 private:
  Status GrantQueues(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc);
  Status GrantDynamicSchedQueues(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc);

  std::atomic_bool heartbeat_listening_{false};
  pid_t pid_ = -1;
  std::unique_ptr<ExecutorMessageClient> message_client_;
  bool is_host_ = false;
  std::atomic<ProcStatus> sub_proc_stat_;
  int32_t aicpu_pid_ = -1;
  // guard grant_queues_map_
  std::mutex mutex_;
  // key: pid, value: queue_attr key
  std::map<int32_t, std::set<std::string>> grant_queues_map_;
  // guard bind pids
  std::mutex pid_mutex_;
  std::set<int32_t> bind_pids_;
};

class HostCpuExecutorClient : public BuiltinExecutorClient {
 public:
  explicit HostCpuExecutorClient(int32_t device_id);
  ~HostCpuExecutorClient() override = default;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_BUILTIN_EXECUTOR_CLIENT_H_
