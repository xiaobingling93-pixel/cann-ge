/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOY_STATE_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOY_STATE_H_

#include <memory>
#include "dflow/inc/data_flow/model/flow_model.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "proto/deployer.pb.h"
#include "deploy/flowrm/heterogeneous_exchange_deployer.h"
#include "deploy/execfwk/executor_manager.h"
#include "dflow/base/model/model_deploy_resource.h"

namespace ge {
class DeployState {
 public:
  using FlowRoutePlans = std::vector<std::pair<int32_t, deployer::FlowRoutePlan>>;
  DeployState() = default;
  explicit DeployState(FlowModelPtr flow_model);
  ~DeployState() = default;

  const FlowModelPtr &GetFlowModel() const;

  void SetFlowModel(const FlowModelPtr &flow_model);

  void SetInputModelName(const std::string &input_model_name);
  void SetOutputModelName(const std::string &output_model_name);
  const std::string &GetInputModelName() const;
  const std::string &GetOutputModelName() const;

  const DeployPlan &GetDeployPlan() const;
  void SetDeployPlan(DeployPlan &&deploy_plan);
  DeployPlan &MutableDeployPlan();

  const std::pair<int32_t, deployer::FlowRoutePlan> &GetLocalFlowRoutePlan() const;

  void SetRootModelId(uint32_t root_model_id);
  uint32_t GetRootModelId() const;

  void SetGraphId(uint32_t graph_id);
  uint32_t GetGraphId() const;

  bool GetIsDynamicSched() const;
  void SetIsDynamicSched(const bool is_dynamic_sched);

  void SetEnableExceptionCatch(const bool enable_exception_catch);
  bool IsEnableExceptionCatch() const;

  void SetContainsNMappingNode(bool contains_n_mapping_node);
  bool IsContainsNMappingNode() const;

  const InputAlignAttrs &GetInputAlignAttrs() const {
    return input_align_attrs_;
  }
  InputAlignAttrs &MutableInputAlignAttrs() {
    return input_align_attrs_;
  }
  void SetLocalFlowRoutePlan(const int32_t node_id, const deployer::FlowRoutePlan &flow_route_plan);
  void AddFlowRoutePlanToDeploy(const int32_t node_id,
                                const deployer::FlowRoutePlan &flow_route_plan);

  const FlowRoutePlans &GetFlowRoutePlansToDeploy() const;
  DeployState::FlowRoutePlans &MutableFlowRoutePlansToDeploy();
  void AddLocalSubmodelDesc(int32_t device_id, int32_t device_type, const deployer::SubmodelDesc &submodel_desc);

  const std::set<int32_t> &GetDeployedNodeIds() const;

  uint32_t GetSubmodelId(const std::string &submodel_name) const;

  uint64_t GetSessionId() const;
  void SetSessionId(uint64_t session_id);

  std::string GetRelativeWorkingDir() const;

  bool GetDynamicProxyControlledFlag(const uint32_t submodel_id) const;

  void SetOptions(const deployer::Options &options);
  const std::map<std::string, std::string> &GetAllGlobalOptions() const;
  const std::map<std::string, std::string> &GetAllSessionOptions() const;
  const std::map<std::string, std::string> &GetAllGraphOptions() const;

  const std::vector<deployer::SubmodelDesc> *GetLocalSubmodels(const ExecutorManager::ExecutorKey &key) const;
  const std::vector<std::pair<DeployPlan::DeviceInfo, deployer::DataGwSchedInfos>> &GetDataGwSchedInfos() const;
  void AddDataGwSchedInfos(const DeployPlan::DeviceInfo &device_info, const deployer::DataGwSchedInfos &sched_info);

 private:
  void UpdateClientRank(const ExecutorManager::ExecutorKey &key);
  friend class HeterogeneousModelDeployer;
  friend class DeployContext;
  friend class FlowRouteManager;
  uint64_t session_id_ = UINT64_MAX;
  uint32_t root_model_id_ = UINT32_MAX;
  uint32_t graph_id_ = UINT32_MAX;
  bool is_dynamic_sched_ = false;
  bool enable_exception_catch_ = false;
  bool contains_n_mapping_node_ = false;
  InputAlignAttrs input_align_attrs_{};
  FlowModelPtr flow_model_;
  std::string input_model_name_;
  std::string output_model_name_;

  DeployPlan deploy_plan_;
  FlowRoutePlans flow_route_plans_to_deploy_;
  std::pair<int32_t, deployer::FlowRoutePlan> local_flow_route_plan_;
  std::map<ExecutorManager::ExecutorKey, std::vector<deployer::SubmodelDesc>> local_submodel_descs_;
  std::pair<int32_t, std::unique_ptr<HeterogeneousExchangeDeployer>> node_exchange_deployer_pair_;
  std::set<int32_t> deployed_node_ids_;
  mutable std::map<std::string, uint32_t> submodel_ids_;
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> session_options_;
  std::map<std::string, std::string> graph_options_;
  // proxy process control model execution (dynamic model loaded on host)
  std::map<uint32_t, bool> dynamic_proxy_controlled_flags_;
  std::vector<std::pair<DeployPlan::DeviceInfo, deployer::DataGwSchedInfos>> datagw_sched_infos_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOY_STATE_H_
