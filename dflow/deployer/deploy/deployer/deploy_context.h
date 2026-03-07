/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOY_CONTEXT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOY_CONTEXT_H_

#include <cstdint>
#include <map>
#include <set>
#include <mutex>
#include <functional>
#include <thread>
#include <atomic>
#include "deploy/deployer/deploy_state.h"
#include "deploy/deployer/deployer_var_manager.h"
#include "deploy/flowrm/heterogeneous_exchange_deployer.h"
#include "deploy/flowrm/flowgw_client_manager.h"
#include "common/config/device_debug_config.h"
#include "deploy/execfwk/executor_manager.h"
#include "deploy/model_recv/flow_model_receiver.h"
#include "ge/ge_api_error_codes.h"
#include "proto/deployer.pb.h"
#include "common/data_flow/route/rank_table_builder.h"

namespace ge {
using RootModelId2SubmodelDesc =
    std::map<uint32_t, std::map<ExecutorManager::ExecutorKey, std::vector<deployer::SubmodelDesc>>>;
constexpr int32_t EXCEPTION_HANDLE_STOP = 1;
constexpr int32_t EXCEPTION_HANDLE_CLEAR = 2;
class DeployContext {
 public:
  static DeployContext &LocalContext();

  /*
   *  @ingroup ge
   *  @brief   initialize context
   *  @param   [in]  None
   *  @return  None
   */
  void Initialize();

  /*
   *  @ingroup ge
   *  @brief   finalize context
   *  @param   [in]  None
   *  @return  None
   */
  void Finalize();

  void SetName(const std::string &name);

  const std::string &GetName() const;

  void SetDeployerPid(int32_t pid);

  int32_t GetDeployerPid() const;

  const std::string &GetBaseDir() const;

  /*
   *  @ingroup ge
   *  @brief   download device debug config
   *  @param   [in]  const DeployerRequest *
   *  @param   [in]  DeployerResponse *
   *  @return  SUCCESS or FAILED
   */
  Status DownloadDevMaintenanceCfg(const deployer::DeployerRequest &request, deployer::DeployerResponse &response);

  Status PreDeployLocalFlowRoute(uint32_t root_model_id);

  Status DeployLocalFlowRoute(uint32_t root_model_id);

  const ExchangeRoute *QueryFlowRoute(uint32_t root_model_id);

  Status LoadLocalModel(uint32_t root_model_id);

  Status UnloadSubmodels(uint32_t root_model_id);

  DeployerVarManager *GetVarManager(int32_t device_id, uint64_t session_id);

  Status ProcessMultiVarManager(const deployer::MultiVarManagerRequest &request);

  Status ProcessSharedContent(const deployer::SharedContentDescRequest &request,
                              const deployer::DeployerResponse &response);

  Status ProcessHeartbeat(const deployer::DeployerRequest &request,
                          deployer::DeployerResponse &response);

  void DestroyDeployState(uint32_t model_id);

  RankTableBuilder& GetRankTableBuilder();

  FlowGwClientManager &GetFlowGwClientManager();

  Status DataGwSchedInfo(const DeployState &deploy_state, const deployer::DataGwSchedInfos &req_body);

  Status ClearModelRunningData(const deployer::ClearModelDataRequest &req_body);
  Status DataFlowExceptionNotifyProcess(const deployer::DataFlowExceptionNotifyRequest &req_body);

  std::mutex &GetAbnormalHeartbeatInfoMu();

  void AddAbnormalSubmodelInstanceName(uint32_t root_model_id, const std::string model_instance_name);

  RootModelId2SubmodelName &GetAbnormalSubmodelInstanceName();

  void ClearAbnormalSubmodelInstanceName();

  void AddAbnormalNodeConfig(NodeConfig node_config);

  const std::map<NodeConfig, bool> &GetAbnormalNodeConfig();

  void ClearAbnormalNodeConfig();

  void AddAbnormalDeviceInfo(DeployPlan::DeviceInfo device_info);

  const std::map<DeployPlan::DeviceInfo, bool> &GetAbnormalDeviceInfo();

  void ClearAbnormalDeviceInfo();

  Status UpdateProfilingInfoProcess(const deployer::SendProfInfoRequest &req_body);

  Status UpdateLocalProfiling(const bool is_prof_start, const std::string &prof_data,
                              const std::vector<uint32_t> &model_ids);

  Status UpdateLocalProfilingInfo(const bool is_prof_start, const std::string &prof_data,
                                  const uint32_t model_id);

  Status UpdateProfiling(const bool is_prof_start, const std::string &prof_data,
                         const ExecutorManager::ExecutorKey &key);

 private:
  friend class DeployerServiceImpl;

  void Clear() const;

  static Status GetQueues(const ExchangeRoute &flow_route,
                          const deployer::SubmodelDesc &submodel_desc,
                          std::vector<DeployQueueAttr> &input_queues,
                          std::vector<DeployQueueAttr> &output_queues);

  static Status GetQueues(const ExchangeRoute &flow_route,
                          std::vector<int32_t> &input_indices,
                          std::vector<int32_t> &output_indices,
                          std::vector<uint32_t> &input_queues,
                          std::vector<uint32_t> &output_queues);

  static Status GetSchedQueues(const ExchangeRoute &flow_route,
                               const deployer::SubmodelDesc &submodel_desc,
                               std::vector<DeployQueueAttr> &input_queues,
                               std::vector<DeployQueueAttr> &output_queues);

  static Status GetInputFusionOffsets(const ExchangeRoute &flow_route,
                                     const deployer::SubmodelDesc &submodel_desc,
                                     std::vector<int32_t> &fusion_offsets);

  static Status GetFlowRoute(const DeployState &deploy_state,
                             const ExchangeRoute *&flow_route);

  Status InitProcessResource(const deployer::InitProcessResourceRequest &request, deployer::DeployerResponse &response);

  Status LoadLocalSubmodels(DeployState &deploy_state);

  Status UnloadSubmodelsFromExecutor(const ExecutorManager::ExecutorKey &executor_key, uint32_t root_model_id);
  Status PrepareExecutors(const DeployState &deploy_state);

  void SetVarMemoryInfo(int32_t device_id,
                        uint64_t session_id,
                        deployer::ExecutorRequest_BatchLoadModelMessage &request);

  Status SetModelInfo(const DeployState &deploy_state,
                      const ExecutorManager::ExecutorKey &key,
                      deployer::ExecutorRequest_BatchLoadModelMessage &request);
  Status VarManagersPreAlloc(DeployState &deploy_state);
  Status VarManagerPreAlloc(DeployerVarManager &var_manager) const;
  Status PrepareStateWorkingDir(const DeployState &deploy_state) const;
  Status SyncVarManagers(PneExecutorClient &executor_client, const DeployState &deploy_state);
  Status DoLoadSubmodels(const DeployState &deploy_state,
                         const ExecutorManager::ExecutorKey &key);
  Status GetOrCreateTransferQueue(int32_t device_id, uint32_t &queue_id);
  static void SetOptions(const DeployState &deploy_state, deployer::ExecutorRequest_BatchLoadModelMessage &request);

  Status SetDynamicSchedModelInfo(deployer::ExecutorRequest_LoadModelRequest *&model_info,
                                  const deployer::SubmodelDesc &submodel_desc,
                                  const ExchangeRoute *flow_route,
                                  const DeployState &deploy_state) const;

  void SetModelQueuesAttrs(const std::string &model_name,
                           const std::vector<DeployQueueAttr> &model_input_queues,
                           const std::vector<DeployQueueAttr> &model_output_queues,
                           deployer::ExecutorRequest_ModelQueuesAttrs &model_queues_attrs_def,
                           bool is_invoked = false) const;

  void GetModelClearInfo(const deployer::ClearModelDataRequest &req_body,
                         std::map<uint32_t, std::set<ExecutorManager::ExecutorKey>> &models_info,
                         std::set<uint32_t> &model_ids);

  Status SyncSubmitClearModelTasks(std::map<uint32_t, std::set<ExecutorManager::ExecutorKey>> &models_info,
                                   uint32_t parallel_num, int32_t type,
                                   const std::set<int32_t> &device_ids);

  Status SyncSubmitClearFlowgwTasks(const std::set<uint32_t> &model_ids, int32_t type);

  bool CheckExecutorKeyIsException(const deployer::ClearModelDataRequest &req_body,
                                   const ExecutorManager::ExecutorKey &executor_key) const;
  void AddAbnormalSubmodelInstance(deployer::DeployerResponse &response,
      const std::map<uint32_t, std::vector<std::string>> &model_instance_name);
  void AbnormalPidsToSubmodelInstance(deployer::DeployerResponse &response,
      std::map<ExecutorManager::ExecutorKey, bool> &abnormal_pids);
  void GetExceptionDevices(const deployer::ClearModelDataRequest &req_body,
                           std::vector<FlowGwClient::ExceptionDeviceInfo> &devices);

  Status SyncUpdateExceptionRoutes(const std::set<uint32_t> &model_ids,
                                   const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices);
  
  void GetExceptionDevId(const deployer::ClearModelDataRequest &req_body,
                         std::set<int32_t> &device_ids);

  std::mutex mu_;
  std::string name_ = "unnamed";
  int32_t deployer_pid_ = -1;
  std::string base_dir_;
  std::string client_base_dir_;

  // key: root_model_id
  std::map<uint32_t, std::set<ExecutorManager::ExecutorKey>> submodel_devices_;
  // key: model_id, value: route_id
  std::map<uint32_t, int64_t> submodel_routes_;
  // key: device_id, sub_key: session_id
  std::map<int32_t, std::map<uint64_t, std::unique_ptr<DeployerVarManager>>> var_managers_;
  // key: model_name
  std::map<std::string, PneModelPtr> submodels_;
  ExecutorManager executor_manager_;
  DeviceMaintenanceClientCfg dev_maintenance_cfg_;
  FlowModelReceiver flow_model_receiver_;
  int32_t device_cout_ = -1;
  FlowGwClientManager flowgw_client_manager_;
  RankTableBuilder rank_table_builder_;
  std::map<int64_t, ExchangeRoute> tansfer_routes_;
  std::map<int32_t, uint32_t> transfer_queues_;
  std::mutex abnormal_heartbeat_info_mu_;
  // root_model_id
  RootModelId2SubmodelDesc local_rootmodel_to_submodel_descs_;
  RootModelId2SubmodelName abnormal_submodel_instances_name_;
  std::map<NodeConfig, bool> abnormal_node_config_;
  std::map<DeployPlan::DeviceInfo, bool> abnormal_device_info_;
  std::map<ExecutorManager::ExecutorKey, bool> abnormal_pids_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOY_CONTEXT_H_
