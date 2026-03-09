/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/deploy_state.h"

namespace ge {
DeployState::DeployState(FlowModelPtr flow_model) : flow_model_(std::move(flow_model)) {
}

const FlowModelPtr &DeployState::GetFlowModel() const {
  return flow_model_;
}

void DeployState::SetFlowModel(const FlowModelPtr &flow_model) {
  flow_model_ = flow_model;
}

void DeployState::SetDeployPlan(DeployPlan &&deploy_plan) {
  deploy_plan_ = std::move(deploy_plan);
}

const DeployPlan &DeployState::GetDeployPlan() const {
  return deploy_plan_;
}

DeployPlan &DeployState::MutableDeployPlan() {
  return deploy_plan_;
}

const std::pair<int32_t, deployer::FlowRoutePlan> &DeployState::GetLocalFlowRoutePlan() const {
  return local_flow_route_plan_;
}

const DeployState::FlowRoutePlans &DeployState::GetFlowRoutePlansToDeploy() const {
  return flow_route_plans_to_deploy_;
}

DeployState::FlowRoutePlans &DeployState::MutableFlowRoutePlansToDeploy() {
  return flow_route_plans_to_deploy_;
}

uint32_t DeployState::GetRootModelId() const {
  return root_model_id_;
}

void DeployState::SetRootModelId(uint32_t root_model_id) {
  root_model_id_ = root_model_id;
}

uint32_t DeployState::GetGraphId() const {
  return graph_id_;
}

void DeployState::SetGraphId(uint32_t graph_id) {
  graph_id_ = graph_id;
}

bool DeployState::GetIsDynamicSched() const {
  return is_dynamic_sched_;
}

void DeployState::SetIsDynamicSched(const bool is_dynamic_sched) {
  is_dynamic_sched_ = is_dynamic_sched;
}

void DeployState::SetEnableExceptionCatch(const bool enable_exception_catch) {
  enable_exception_catch_ = enable_exception_catch;
}

bool DeployState::IsEnableExceptionCatch() const {
  return enable_exception_catch_;
}

void DeployState::SetContainsNMappingNode(bool contains_n_mapping_node) {
  contains_n_mapping_node_ = contains_n_mapping_node;
}
bool DeployState::IsContainsNMappingNode() const {
  return contains_n_mapping_node_;
}

void DeployState::SetLocalFlowRoutePlan(const int32_t node_id, const deployer::FlowRoutePlan &flow_route_plan) {
  local_flow_route_plan_ = std::make_pair(node_id, flow_route_plan);
}

void DeployState::AddFlowRoutePlanToDeploy(const int32_t node_id,
                                           const deployer::FlowRoutePlan &flow_route_plan) {
  flow_route_plans_to_deploy_.emplace_back(std::make_pair(node_id, flow_route_plan));
}

void DeployState::UpdateClientRank(const ExecutorManager::ExecutorKey &key) {
  auto it = local_submodel_descs_.find(key);
  if (it != local_submodel_descs_.end() && it->first.rank_id.empty() && !key.rank_id.empty()) {
    auto update_key = it->first;
    update_key.rank_id = key.rank_id;
    auto descs = std::move(it->second);
    local_submodel_descs_.erase(it);
    local_submodel_descs_[update_key] = std::move(descs);
  }
}

void DeployState::AddLocalSubmodelDesc(int32_t device_id, int32_t device_type,
                                       const deployer::SubmodelDesc &submodel_desc) {
  ExecutorManager::ExecutorKey key = {};
  key.device_id = device_id;
  key.device_type = device_type;
  key.engine_name = submodel_desc.engine_name();
  key.rank_id = submodel_desc.rank_id();
  key.process_id = submodel_desc.process_id();

  key.is_proxy = device_type != CPU;
  // 当前只有udf存在 proxy场景
  if ((key.is_proxy) && (key.engine_name != PNE_ID_UDF)) {
    key.is_proxy = false;
    dynamic_proxy_controlled_flags_[submodel_desc.submodel_id()] = submodel_desc.is_dynamic();
  } else {
    dynamic_proxy_controlled_flags_[submodel_desc.submodel_id()] = false;
  }
  UpdateClientRank(key);
  GELOGI("Add submodel[%s] desc success, is_dynamic = %d, is_dynamic_proxy_controlled = %d",
         submodel_desc.model_name().c_str(),
         static_cast<int32_t>(submodel_desc.is_dynamic()),
         static_cast<int32_t>(dynamic_proxy_controlled_flags_[submodel_desc.submodel_id()]));
  local_submodel_descs_[key].emplace_back(submodel_desc);
}

const std::set<int32_t> &DeployState::GetDeployedNodeIds() const {
  return deployed_node_ids_;
}

uint32_t DeployState::GetSubmodelId(const std::string &submodel_name) const {
  auto it = submodel_ids_.find(submodel_name);
  if (it == submodel_ids_.end()) {
    uint32_t model_id = submodel_ids_.size() + 1;
    submodel_ids_[submodel_name] = model_id;
    return model_id;
  } else {
    return it->second;
  }
}

void DeployState::SetOptions(const deployer::Options &options) {
  for (const auto &option : options.global_options()) {
    global_options_[option.first] = option.second;
  }
  for (const auto &option : options.session_options()) {
    session_options_[option.first] = option.second;
  }
  for (const auto &option : options.graph_options()) {
    graph_options_[option.first] = option.second;
  }
}

const std::map<std::string, std::string> &DeployState::GetAllGlobalOptions() const {
  return global_options_;
}

const std::map<std::string, std::string> &DeployState::GetAllSessionOptions() const {
  return session_options_;
}

const std::map<std::string, std::string> &DeployState::GetAllGraphOptions() const {
  return graph_options_;
}

bool DeployState::GetDynamicProxyControlledFlag(const uint32_t submodel_id) const {
  auto iter = dynamic_proxy_controlled_flags_.find(submodel_id);
  if (iter != dynamic_proxy_controlled_flags_.end()) {
    GELOGD("Get dynamic proxy controlled flag, flag = %d, root_model_id = %u, submodel_id = %u.",
           static_cast<int32_t>(iter->second), root_model_id_, submodel_id);
    return iter->second;
  }
  GEEVENT("Not find dynamic proxy controlled flag, root_model_id = %u, submodel_id = %u.", root_model_id_, submodel_id);
  return false;
}

uint64_t DeployState::GetSessionId() const {
  return session_id_;
}

void DeployState::SetSessionId(uint64_t session_id) {
  session_id_ = session_id;
}

std::string DeployState::GetRelativeWorkingDir() const {
  return std::to_string(GetSessionId()) + "/" + std::to_string(GetRootModelId()) + "/";
}

const std::vector<deployer::SubmodelDesc> *DeployState::GetLocalSubmodels(
    const ExecutorManager::ExecutorKey &key) const {
  const auto &it = local_submodel_descs_.find(key);
  if (it == local_submodel_descs_.cend()) {
    return nullptr;
  }
  return &it->second;
}

const std::string &DeployState::GetInputModelName() const {
  return input_model_name_;
}

const std::string &DeployState::GetOutputModelName() const {
  return output_model_name_;
}

void DeployState::SetInputModelName(const std::string &input_model_name) {
  input_model_name_ = input_model_name;
}

void DeployState::SetOutputModelName(const std::string &output_model_name) {
  output_model_name_ = output_model_name;
}

void DeployState::AddDataGwSchedInfos(const DeployPlan::DeviceInfo &device_info,
                                      const deployer::DataGwSchedInfos &sched_info) {
  datagw_sched_infos_.emplace_back(std::make_pair(device_info, sched_info));
}

const std::vector<std::pair<DeployPlan::DeviceInfo, deployer::DataGwSchedInfos>> &DeployState::GetDataGwSchedInfos(
    ) const {
  return datagw_sched_infos_;
}
}  // namespace ge
