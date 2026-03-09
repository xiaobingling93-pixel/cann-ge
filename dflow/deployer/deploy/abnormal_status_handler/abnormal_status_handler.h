/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_ABNORMAL_STATUS_HANDLER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_ABNORMAL_STATUS_HANDLER_H_

#include "common/thread_pool/thread_pool.h"
#include "common/config/device_debug_config.h"
#include "dflow/base/deploy/model_deployer.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "graph/ge_local_context.h"
#include "common/config/config_parser.h"

namespace ge {
class AbnormalStatusHandler {
 public:
  explicit AbnormalStatusHandler(std::mutex &mu);
  virtual ~AbnormalStatusHandler();

  void Initialize();

  void Finalize();

  Status ClearModelExceptionData(uint32_t root_model_id, const std::vector<DeployPlan::DeviceInfo> &device_infos);
  void DelCallback(const uint32_t root_model_id);
  void IncDeployingRootModelNum();
  void DecreaseDeployingRootModelNum();
  DeployPlan::AbnormalStatusCallbackInfo* GetAbnormalStatusCallbackInfo();
  void SetDynamicSchedFlag(bool flag);
  void AddDeployedModelInfo(uint32_t model_id,
      const DeployPlan::ModelDeployInfo &model_deploy_infos,
      const std::set<int32_t> &deployed_remote_nodes);
  void DelDeployedModelInfo(uint32_t model_id);

 private:
  struct DeployedModel {
    uint32_t model_id = UINT32_MAX;  // root_model_id
    // key: model_name, (key: model_instance_name, value: device_info)
    DeployPlan::ModelDeployInfo model_deploy_infos;
    std::set<int32_t> deployed_remote_nodes;
  };
  Status ParallelClearData(const std::pair<uint32_t, std::set<uint32_t>> &need_clear_root_models,
                           const std::vector<DeployPlan::DeviceInfo> &device_infos, const int32_t type) const;
  void FindOldDevice(DeployPlan::DeviceStateList &device_state_list,
      NodeConfig node_new, NodeConfig node_old) const;
  Status FindAbnormalDeviceOnServer(DeployPlan::DeviceStateList &device_state_list,
      DeployerConfig information_new, DeployerConfig information_old) const;
  Status FindAbnormalDevice(DeployPlan::DeviceStateList &device_state_list,
      DeployerConfig information_new, DeployerConfig information_old) const;
  Status ParseDeviceStateList(const std::string &file_path, DeployPlan::DeviceStateList &device_state_list);
  bool IsModelMulInstace(std::map<const std::string, bool> &abnormal_submodel_instances_name,
      DeployPlan::ModelDeployInfo model_deploy_infos) const;
  bool IsHeartbeatNormal() const;
  void ParseAbnormalNodeConfig(DeployPlan::DeviceStateList &device_state_list) const;
  void ParseAbnormalDeviceInfo(DeployPlan::DeviceStateList &device_state_list) const;
  void ParseAbnormalModelInstances(bool &is_new_abnormal);
  uint32_t CheckAbnormalDevices(DeployPlan::DeviceStateList &device_state_list) const;
  void ShowNodeInfo(DeployerConfig &information) const;
  Status RedeployProc(uint32_t root_model_id, uint32_t check_devices_flag);
  void ParseHeartbeatAbnormalInfo(bool &is_new_abnormal, DeployPlan::DeviceStateList &device_state_list);
  bool IsSupportDynamicSchedRecover(const uint32_t &root_model_id);
  Status GenerateFile(const std::string &file_path, const char_t *const file_name) const;
  void PreHandleAbnormalInfo();
  Status AfterHandleAbnormalInfo(const std::string &file_path, const char_t *const file_name);
  Status FailedHandleAbnormal(uint32_t root_model_id);
  void GetDeviceListDiff(const DeployPlan::DeviceStateList &device_state_list_new,
      DeployPlan::DeviceStateList &device_state_list_old, DeployPlan::DeviceStateList &device_state_list_diff) const;
  bool IsInDeviceList(std::set<DeployPlan::DeviceInfo> &instance_device_infos,
      DeployPlan::DeviceStateList &device_state_list_diff) const;
  bool IsInModelInstanceList(uint32_t root_model_id, const std::string &model_instance_name,
      RootModelId2SubmodelName &abnormal_submodel_instances_name) const;
  void Add2ModelInstanceList(uint32_t root_model_id, const std::string &model_instance_name,
      RootModelId2SubmodelName &abnormal_submodel_instances_name) const;
  void AbnormalDiffDevices2ModelInstances(uint32_t root_model_id,
      std::map<std::string, std::set<DeployPlan::DeviceInfo>> &model_deploy_info,
      DeployPlan::DeviceStateList &device_state_list_diff, bool &is_new_abnormal);
  void AbnormalDevices2ModelInstances(DeployPlan::DeviceStateList &device_state_list, bool &is_new_abnormal);
  Status RedeployStart(const uint32_t &root_model_id);
  Status FileMonitorProc(const std::string &file_path);
  Status ParallelAbnormalStatusHandle(uint32_t check_devices_flag);
  Status WaitReployFileGenerate(const std::string &file_path);
  bool IsReployFileGeneratedThenRemove(const std::string &file_path) const;
  Status GetFilePath(std::string &config_dir, const char_t *const path_env) const;
  void MonitorFileAndHeartbeatProc(const std::string &file_path, const int32_t &fd);
  Status GetMonitorFilePath(std::string &file_path);
  void AbnormalStatusMonitorRun();
  bool IsDeployingRootModel();
  bool IsAllCallbackInitFinished();
  Status DynamicSchedRecoverProc(uint32_t root_model_id);
  Status HeartbeatMonitorProc();

  std::mutex &mu_;
  std::map<uint32_t, DeployedModel> deployed_models_;

  std::thread file_monitor_thread_;
  std::atomic_bool file_monitor_flag_{true};
  bool is_dynamic_sched_ = false;
  DeployPlan::AbnormalStatusCallbackInfo abnormal_status_callback_info_;
  std::atomic<uint32_t> deploying_root_model_cnt_{0U};
  GEThreadLocalContext run_context_;
  DeployPlan::DeviceStateList device_state_list_;
  RootModelId2SubmodelName abnormal_submodel_instances_name_; // 记录已存在异常的submodel instance
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_ABNORMAL_STATUS_HANDLER_H_
