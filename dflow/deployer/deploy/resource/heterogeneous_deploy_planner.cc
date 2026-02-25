/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/resource/heterogeneous_deploy_planner.h"
#include <regex>
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "dflow/base/model/model_deploy_resource.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/resource/device_info.h"
#include "deploy/deployer/deployer_proxy.h"

namespace ge {
namespace {
constexpr int64_t kDepDefQueDepth = 128L;

void RewriteQueueName(const std::string &model_name,
                      const std::map<std::string, std::string> &name_map,
                      std::string &queue_name) {
  const auto &it = name_map.find(queue_name);
  if (it == name_map.end()) {
    // inner queue, add model name to avoid name conflict
    queue_name.insert(0, model_name + ":");
  } else {
    queue_name = it->second;
  }
}

void RewriteQueueNames(const std::string &model_name,
                       const std::map<std::string, std::string> &name_map,
                       std::vector<std::string> &queue_names) {
  std::for_each(queue_names.begin(), queue_names.end(), [&model_name, &name_map](std::string &queue_name) {
    RewriteQueueName(model_name, name_map, queue_name);
  });
}

Status RewriteQueueNames(const std::string &model_name,
                         const ModelRelation::ModelEndpointInfo &model_queue_info,
                         ModelRelation &model_relation,
                         std::vector<Endpoint> &internal_queue_defs) {
  if (model_relation.root_model_endpoint_info.input_endpoint_names.size() !=
      model_queue_info.input_endpoint_names.size()) {
    GELOGE(PARAM_INVALID, "model input number mismatches, parent input size = %zu, submodel input size = %zu",
           model_queue_info.input_endpoint_names.size(),
           model_relation.root_model_endpoint_info.input_endpoint_names.size());
    return PARAM_INVALID;
  }
  std::map<std::string, std::string> queue_name_mapping;
  for (size_t i = 0; i < model_relation.root_model_endpoint_info.input_endpoint_names.size(); ++i) {
    queue_name_mapping.emplace(model_relation.root_model_endpoint_info.input_endpoint_names[i],
                               model_queue_info.input_endpoint_names[i]);
    model_relation.root_model_endpoint_info.input_endpoint_names[i] = model_queue_info.input_endpoint_names[i];
  }
  for (size_t i = 0; i < model_relation.root_model_endpoint_info.output_endpoint_names.size(); ++i) {
    queue_name_mapping.emplace(model_relation.root_model_endpoint_info.output_endpoint_names[i],
                               model_queue_info.output_endpoint_names[i]);
    model_relation.root_model_endpoint_info.output_endpoint_names[i] = model_queue_info.output_endpoint_names[i];
  }

  std::for_each(model_relation.endpoints.begin(),
                model_relation.endpoints.end(),
                [&model_name, &queue_name_mapping, &internal_queue_defs](Endpoint &queue_def) {
    RewriteQueueName(model_name, queue_name_mapping, queue_def.MutableName());
    if (queue_name_mapping.find(queue_def.GetName()) == queue_name_mapping.end()) {
      internal_queue_defs.emplace_back(queue_def);
    }
  });
  for (auto &submodel_it : model_relation.submodel_endpoint_infos) {
    auto &queue_info = submodel_it.second;
    RewriteQueueNames(model_name, queue_name_mapping, queue_info.input_endpoint_names);
    RewriteQueueNames(model_name, queue_name_mapping, queue_info.output_endpoint_names);
  }
  return SUCCESS;
}
}  // namespace

HeterogeneousDeployPlanner::HeterogeneousDeployPlanner(const FlowModelPtr &flow_model,
                                                       std::vector<DeployPlan::DeviceInfo> device_list) noexcept
    : HeterogeneousDeployPlanner({flow_model}, nullptr, std::move(device_list)) {}

HeterogeneousDeployPlanner::HeterogeneousDeployPlanner(std::vector<FlowModelPtr> models,
                                                       const ModelRelation *root_model_relation,
                                                       std::vector<DeployPlan::DeviceInfo> device_list) noexcept
    : DeployPlannerBase(),
      models_(std::move(models)),
      root_model_relation_(root_model_relation),
      device_list_(std::move(device_list)) {}

Status HeterogeneousDeployPlanner::PrepareModelsAndRelation(ModelRelation &model_relation) {
  GE_CHK_BOOL_RET_STATUS(!models_.empty(), PARAM_INVALID, "models is empty");
  std::map<std::string, PneModelPtr> name_to_models;
  if (models_.size() == 1U) {
    GELOGD("start to build deploy plan for single model");
    GE_CHK_STATUS_RET(PrepareForSingleFlowModel(name_to_models, model_relation),
                      "Failed to init for single model");
  } else {
    GELOGD("start to build deploy plan for multiply models");
    GE_CHK_STATUS_RET(MergeModels(name_to_models, model_relation), "Failed to merge models by relation");
    GE_CHK_STATUS_RET(ValidateModelAndRelation(name_to_models, model_relation),
                      "Failed to validate model and relation after merging submodels");
  }
  GE_CHK_STATUS_RET_NOLOG(PrepareTargetDevices(name_to_models, model_relation));
  GE_CHK_STATUS_RET_NOLOG(ValidateModelAndRelation(name_to_models, model_relation));
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::GetDeployEngineType(const PneModel &pne_model, std::string &deploy_type) {
  const auto &engine_type = pne_model.GetModelType().empty() ? PNE_ID_NPU : pne_model.GetModelType();
  if (engine_type != PNE_ID_UDF) {
    deploy_type = engine_type;
    return SUCCESS;
  }

  deploy_type = PNE_ID_NPU;
  auto deploy_resource = pne_model.GetDeployResource().get();
  if ((deploy_resource == nullptr) || (deploy_resource->resource_type.empty())) {
    GELOGI("Model deploy resource type is empty, deploy on NPU default.");
    return SUCCESS;
  }

  const auto &devices = ResourceManager::GetInstance().GetDeviceInfoList();
  std::map<std::string, std::set<std::string>> resource_to_deploy_type;
  for (const auto &device_info : devices) {
    GELOGI("Get device info = %s.", device_info.DebugString().c_str());
    auto tmp_deploy_type = (device_info.GetDeviceType() == NPU) ? PNE_ID_NPU : PNE_ID_CPU;
    resource_to_deploy_type[device_info.GetResourceType()].emplace(tmp_deploy_type);
  }

  auto it = resource_to_deploy_type.find(deploy_resource->resource_type);
  if (it != resource_to_deploy_type.end() && it->second.size() == 1U) {
    deploy_type = *(it->second.begin());
    return SUCCESS;
  }

  GELOGE(PARAM_INVALID, "Failed to match device according to resource type[%s].",
         deploy_resource->resource_type.c_str());
  return PARAM_INVALID;
}

Status HeterogeneousDeployPlanner::SelectTargetDevice(
    const std::map<std::string, PneModelPtr> &name_to_models,
    std::map<std::string, std::vector<std::pair<DeployPlan::DeviceInfo,
                                                bool>>> &target_devices) {
  std::map<std::string, std::vector<DeployPlan::DeviceInfo>> engine_to_devices;
  for (auto &device_info : device_list_) {
    const std::string &engine_type = (device_info.GetType() == CPU) ? PNE_ID_CPU : PNE_ID_NPU;
    engine_to_devices[engine_type].emplace_back(device_info);
  }

  GE_CHK_STATUS_RET_NOLOG(ReindexDevices());
  for (const auto &it : name_to_models) {
    const auto &model_name = it.first;
    std::string engine_type;
    GE_CHK_STATUS_RET_NOLOG(GetDeployEngineType(*it.second, engine_type));
    auto &available_devices = engine_to_devices[engine_type];
    GE_CHK_BOOL_RET_STATUS(!available_devices.empty(), FAILED,
                           "No available device for engine: %s", it.second->GetModelType().c_str());
    GE_CHK_STATUS_RET(AssignDevices(model_name, it.second, engine_type,
                                    available_devices, target_devices),
                      "Failed to assign device for %s", model_name.c_str());
  }
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::ReindexDevices() {
  for (const auto &device_info : device_list_) {
    auto device = ResourceManager::GetInstance().GetDeviceInfo(device_info.GetNodeId(),
                                                               device_info.GetDeviceId(),
                                                               device_info.GetType());
    GE_CHECK_NOTNULL(device);
    auto index_key = device->ToIndex();
    index_to_devices_[index_key] = device_info;
    GELOGD("Add device = %s, index key = %s", device_info.GetDesc().c_str(), index_key.c_str());
  }
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::GetLogicalDeviceId(const PneModel &pne_model,
                                                      const std::string &engine_type,
                                                      std::vector<std::string> &logical_device_ids) {
  // 1. get from PneModel
  auto logical_device_id_str = pne_model.GetLogicDeviceId();
  // 2. get from option
  if ((logical_device_id_str.empty()) && (engine_type == PNE_ID_NPU)) {
    GE_CHK_STATUS_RET_NOLOG(GetLogicalDeviceIdFromOption(logical_device_id_str));
    GE_CHK_STATUS_RET_NOLOG(ParseLogicalDeviceIds(logical_device_id_str, logical_device_ids));
    NormalizeLogicalDeviceId(logical_device_ids);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET_NOLOG(ParseLogicalDeviceIds(logical_device_id_str, logical_device_ids));

  // heavy load UDF not in soc must assign logic device.
  if (logical_device_ids.empty() && (pne_model.GetModelType() == PNE_ID_UDF)) {
    const auto deploy_resource = pne_model.GetDeployResource();
    if ((deploy_resource != nullptr)) {
      GE_CHK_BOOL_RET_STATUS(!(deploy_resource->is_heavy_load), FAILED,
                             "heavy load must assign logic device id, model name=%s", pne_model.GetModelName().c_str());
      constexpr const char *kResourceTypeAscend = "Ascend";
      GE_CHK_BOOL_RET_STATUS(deploy_resource->resource_type == kResourceTypeAscend, FAILED,
                             "udf[%s] resource_type=%s must be heavy load and assign logic device id",
                             pne_model.GetModelName().c_str(), deploy_resource->resource_type.c_str());
    }
  }
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::GetRedundantLogicalDeviceId(const PneModel &pne_model,
                                                               std::vector<std::string> &logical_device_ids) {
  auto logical_device_id_str = pne_model.GetRedundantLogicDeviceId();
  GE_CHK_STATUS_RET_NOLOG(ParseLogicalDeviceIds(logical_device_id_str, logical_device_ids));
  return SUCCESS;
}

void HeterogeneousDeployPlanner::NormalizeLogicalDeviceId(std::vector<std::string> &logical_device_ids) {
  for (auto &device_id : logical_device_ids) {
    constexpr size_t kIncompletedSize = 2U;
    if (StringUtils::Split(device_id, ':').size() == kIncompletedSize) {
      device_id = "0:0:" + device_id;  // apend cluster_id:server_id
    }
  }
}

Status HeterogeneousDeployPlanner::GetLogicalDeviceIdFromOption(std::string &logical_device_id) {
  (void) GetContext().GetOption(OPTION_EXEC_LOGICAL_DEVICE_ID, logical_device_id);
  if (logical_device_id.empty()) {
    return SUCCESS;
  }
  GELOGI("Got %s from option = %s", OPTION_EXEC_LOGICAL_DEVICE_ID, logical_device_id.c_str());
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::ParseLogicalDeviceIds(const std::string &logical_device_id_str,
                                                  std::vector<std::string> &logical_device_ids) {
  auto normalized = StringUtils::ReplaceAll(logical_device_id_str, " ", "");
  if (normalized.empty()) {
    return SUCCESS;
  }

  std::regex remove_brackets(R"(\[?([0-9:,-]+)\]?)");
  std::smatch match_result;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(normalized, match_result, remove_brackets),
                         PARAM_INVALID, "Invalid logical_device_id: %s", logical_device_id_str.c_str());
  logical_device_ids = StringUtils::Split(match_result.str(1), ',');
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::CheckAssignedDevice(const std::string &model_name,
                                                       const std::string &engine_type,
                                                       const std::string &device_index,
                                                       const DeployPlan::DeviceInfo &device_info) {
  const static std::map<std::string, int32_t> kTypeEngineToDevice = {{PNE_ID_NPU, static_cast<int32_t>(NPU)},
                                                                     {PNE_ID_CPU, static_cast<int32_t>(CPU)}};
  const auto &it = kTypeEngineToDevice.find(engine_type);
  if (it != kTypeEngineToDevice.cend()) {
    GE_CHK_BOOL_RET_STATUS(it->second == device_info.GetType(), FAILED,
                           "Failed to assign model[%s] to device[%s], "
                           "model should be assigned to device of type[%s], device desc=[%s].",
                           model_name.c_str(), device_index.c_str(), engine_type.c_str(),
                           device_info.GetDesc().c_str());
  }
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::AssignDevices(const std::string &model_name,
                                                 const PneModelPtr &pne_model,
                                                 const std::string &engine_type,
                                                 const std::vector<DeployPlan::DeviceInfo> &device_list,
                                                 std::map<std::string,
                                                          std::vector<std::pair<DeployPlan::DeviceInfo,
                                                                                bool>>> &target_devices) {
  // assign deploy resources for npu or cpu model
  std::vector<std::string> logical_device_ids;
  std::vector<std::string> redundant_logical_device_ids;
  GE_CHK_STATUS_RET_NOLOG(GetLogicalDeviceId(*pne_model, engine_type, logical_device_ids));
  GE_CHK_STATUS_RET_NOLOG(GetRedundantLogicalDeviceId(*pne_model, redundant_logical_device_ids));
  if (logical_device_ids.empty()) {
    std::vector<std::pair<DeployPlan::DeviceInfo, bool>> device_list_new;
    for (const auto &device : device_list) {
      device_list_new.emplace_back(std::make_pair(device, false));
    }
    target_devices.emplace(model_name, std::move(device_list_new));
    GELOGI("No device is assigned to model[%s]", model_name.c_str());
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(redundant_logical_device_ids.size() < logical_device_ids.size(),
                         PARAM_INVALID, "redundant logical device ids size[%u] is less than "
                         "logical device ids size[%u].",
                         static_cast<uint32_t>(redundant_logical_device_ids.size()),
                         static_cast<uint32_t>(logical_device_ids.size()));
  bool is_heavy_load = false;
  if (pne_model->GetModelType() == PNE_ID_UDF) {
    const auto deploy_resource = pne_model->GetDeployResource();
    if (deploy_resource != nullptr) {
      is_heavy_load = deploy_resource->is_heavy_load;
    }
  }
  const size_t main_instance_max_index = logical_device_ids.size() - redundant_logical_device_ids.size();
  size_t logical_device_id_index = 0U;
  for (std::string &logical_device_id : logical_device_ids) {
    const bool is_redundant = logical_device_id_index >= main_instance_max_index;
    GetValidLogicDeviceId(logical_device_id);
    auto it = index_to_devices_.find(logical_device_id);
    if (it == index_to_devices_.cend()) {
      GELOGE(PARAM_INVALID, "Failed to assign physical device for %s", logical_device_id.c_str());
      return PARAM_INVALID;
    }
    // UDF heavy load need deploy on host
    if (is_heavy_load) {
      const auto &proxy_device_info = it->second;
      if (proxy_device_info.GetType() != NPU) {
        GELOGE(PARAM_INVALID, "heavy load model[%s] assign physical device[%s] is not NPU, device desc = %s",
               model_name.c_str(), logical_device_id.c_str(), proxy_device_info.GetDesc().c_str());
        return PARAM_INVALID;
      }
      // head node type is CPU, device id is 0.
      DeployPlan::DeviceInfo deploy_info = {CPU, proxy_device_info.GetNodeId(), 0, proxy_device_info.GetDeviceId()};
      GELOGI("The heavy load model[%s] assigned to logic device[%s] will be deploy on device desc = %s.",
             model_name.c_str(), logical_device_id.c_str(), deploy_info.GetDesc().c_str());

      GE_CHK_STATUS_RET(CheckAssignedDevice(model_name, engine_type, logical_device_id, deploy_info),
                        "Failed to assign device.");
      // check deploy info exist.
      target_devices[model_name].emplace_back(std::make_pair(deploy_info, is_redundant));
      GELOGI("The device[%s] is assigned to heavy load model[%s], device desc = %s, redundant flag = %d.",
             logical_device_id.c_str(), model_name.c_str(), deploy_info.GetDesc().c_str(),
             static_cast<int32_t>(is_redundant));
    } else {
      GE_CHK_STATUS_RET(CheckAssignedDevice(model_name, engine_type, logical_device_id, it->second),
                        "Failed to assign device.");
      target_devices[model_name].emplace_back(std::make_pair(it->second, is_redundant));
      GELOGI("The device[%s] is assigned to model[%s], device desc = %s, redundant flag = %d.",
             logical_device_id.c_str(), model_name.c_str(), it->second.GetDesc().c_str(),
             static_cast<int32_t>(is_redundant));
    }
    logical_device_id_index++;
  }
  return SUCCESS;
}

void HeterogeneousDeployPlanner::GetValidLogicDeviceId(std::string &device_id) {
  constexpr uint8_t invalid_device_id_size = 3;
  if (StringUtils::Split(device_id, ':').size() == invalid_device_id_size) {
    device_id.append(":0");
  }
}

Status HeterogeneousDeployPlanner::PrepareForSingleFlowModel(std::map<std::string, PneModelPtr> &name_to_models,
                                                             ModelRelation &model_relation) {
  const auto &flow_model = models_.front();
  GE_CHECK_NOTNULL(flow_model);
  if (flow_model->GetSubmodels().size() == 1U) {
    GELOGD("Prepare for single flow model with single submodel");
    const auto &root_model = flow_model->GetSubmodels().begin()->second;
    const auto &root_graph = root_model->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph);
    if (flow_model->GetModelRelation() == nullptr) {
      flow_model->SetModelRelation(MakeShared<ModelRelation>());
      GE_CHECK_NOTNULL(flow_model->GetModelRelation());
      GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*root_graph, *flow_model->GetModelRelation()),
                        "Failed to build model relation");
    } else if (flow_model->GetModelRelation()->IsEmpty()) {
      // model relation will always not be nullptr in offline mode
      GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*root_graph, *flow_model->GetModelRelation()),
                        "Failed to build model relation");
    } else {
      GELOGD("Model relation is not build for single model.");
    }
  }
  ModelRelationFlattener flattener(flow_model);
  GE_CHK_STATUS_RET_NOLOG(flattener.Flatten(model_relation, name_to_models));
  GE_CHK_STATUS_RET(ValidateModelAndRelation(name_to_models, model_relation),
                    "Failed to validate model and relation");
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::UnfoldSubModel(const ModelRelation::ModelEndpointInfo &model_endpoint_info,
                                                  const FlowModel &model,
                                                  std::map<std::string, PneModelPtr> &name_to_models) {
  GE_CHECK_NOTNULL(model.GetModelRelation());
  GE_CHK_STATUS_RET(ValidateModelAndRelation(model.GetSubmodels(), *model.GetModelRelation()),
                    "Failed to validate model and relation for submodel: %s", model.GetModelName().c_str());
  ModelRelation model_relation = *model.GetModelRelation(); // copy
  GE_CHK_STATUS_RET(RewriteQueueNames(model.GetModelName(), model_endpoint_info,
                                      model_relation,
                                      merged_model_relation_.endpoints),
                    "Failed to rewrite queue names, model name = %s",
                    model.GetModelName().c_str());
  ModelRelationReader reader(model_relation);
  GE_CHK_STATUS_RET_NOLOG(reader.Initialize());
  for (const auto &submodel_it : model.GetSubmodels()) {
    const auto &inner_model_name = submodel_it.first;
    auto queue_info = reader.GetSubmodelQueueInfo(inner_model_name);
    GE_CHECK_NOTNULL(queue_info);
    std::string merged_model_name = model.GetModelName() + ":" + inner_model_name;
    // add inner submodel to merged model relation
    name_to_models.emplace(merged_model_name, submodel_it.second);
    merged_model_relation_.submodel_endpoint_infos[merged_model_name] = *queue_info;
  }
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::MergeModels(std::map<std::string, PneModelPtr> &name_to_models,
                                               ModelRelation &model_relation) {
  // initialize merged model relation
  GE_CHECK_NOTNULL(root_model_relation_);
  merged_model_relation_.root_model_endpoint_info.input_endpoint_names =
      root_model_relation_->root_model_endpoint_info.input_endpoint_names;
  merged_model_relation_.root_model_endpoint_info.output_endpoint_names =
      root_model_relation_->root_model_endpoint_info.output_endpoint_names;
  merged_model_relation_.endpoints.insert(merged_model_relation_.endpoints.end(),
                                          root_model_relation_->endpoints.begin(),
                                          root_model_relation_->endpoints.end());
  std::map<std::string, FlowModelPtr> submodels;
  for (const auto &model : models_) {
    submodels.emplace(model->GetModelName(), model);
  }
  for (const auto &it : root_model_relation_->submodel_endpoint_infos) {
    const auto &model_name = it.first;
    const auto &model_endpoint_info = it.second;
    auto &submodel = submodels[model_name];
    if (submodel == nullptr) {
      GELOGE(PARAM_INVALID, "Failed to get model, name = %s", model_name.c_str());
      return PARAM_INVALID;
    }
    if (submodel->GetSubmodels().empty()) {
      GELOGD("Submodel [%s] contains no child model", model_name.c_str());
      name_to_models.emplace(model_name, submodel);
      merged_model_relation_.submodel_endpoint_infos[model_name] = model_endpoint_info;
    } else {
      GELOGD("Submodel [%s] contains child model, start to unfold them to root relation", model_name.c_str());
      GE_CHK_STATUS_RET(UnfoldSubModel(model_endpoint_info, *submodel, name_to_models), "Failed to unfold submodels");
    }
  }
  merged_model_relation_.invoked_model_queue_infos =  std::move(model_relation.invoked_model_queue_infos);
  model_relation = std::move(merged_model_relation_);
  return SUCCESS;
}

bool HeterogeneousDeployPlanner::HasDeployedModelOnDevice(const DeployPlan::DeviceInfo &device_info) const {
  if (((device_info.GetType() == static_cast<int32_t>(CPU)) && (device_info.GetNodeId() != 0)) ||
      (device_info.GetType() == static_cast<int32_t>(NPU))) {
    auto iter = device_deployed_model_num_.find(device_info);
    if ((iter == device_deployed_model_num_.end()) ||
        ((iter != device_deployed_model_num_.end()) && (iter->second == 0U))) {
      GELOGI("DynamicSched no deployed model on device [%s], don't add dynamic sched info.",
             device_info.GetKey().c_str());
      return false;
    }
  }
  GELOGI("DynamicSched have deployed model on device [%s].", device_info.GetKey().c_str());
  return true;
}

void HeterogeneousDeployPlanner::AddDynamicSchedModel(std::map<std::string, PneModelPtr> &model_instance_to_models,
                                                      ModelRelation &resolved_model_relation,
                                                      std::map<std::string,
                                                      ModelRelation::ModelEndpointInfo> &device_endpoint_info,
                                                      std::vector<DeployPlan::DeviceInfo> &flowgw_device_infos) {
  uint32_t queue_id = 0U;
  for (const auto &device_info : flowgw_device_infos) {
    auto device = ResourceManager::GetInstance().GetDeviceInfo(device_info.GetNodeId(),
                                                               device_info.GetDeviceId(),
                                                               device_info.GetType());
    GE_RT_VOID_CHECK_NOTNULL(device);
    if (!device->SupportFlowgw() && (device_info.GetType() == static_cast<int32_t>(CPU))) {
      GELOGI("DynamicSched no host flowgw don't create sched model.");
      continue;
    }
    if (HasDeployedModelOnDevice(device_info)) {
      const auto model_instance_name = "Sched@" + device_info.GetKey() + "@" + std::to_string(queue_id++);
      model_instance_to_models.emplace(model_instance_name, nullptr);
      auto &submodel_info = MutableSubmodelInfo(model_instance_name);
      submodel_info.model = nullptr;
      submodel_info.device_info = device_info;
      (void)AssignSubmodelsQueueDeviceInfo(model_instance_name, submodel_info);
      auto &endpoint_info = device_endpoint_info[device_info.GetKey()];
      endpoint_info.model_name = model_instance_name;
      resolved_model_relation.submodel_endpoint_infos.emplace(model_instance_name, endpoint_info);
      GELOGI("DynamicSched create a sched model, model instance name=%s.", model_instance_name.c_str());
    }
  }
}

void HeterogeneousDeployPlanner::AddDynamicSchedRelation(std::map<std::string, PneModelPtr> &model_instance_to_models,
                                                         ModelRelation &resolved_model_relation) {
  if (!GetIsDynamicSched()) {
    GEEVENT("DynamicSched flag close, don't add sched model.");
    return;
  }
  uint32_t sched_queue_id = 0U;
  std::map<std::string, ModelRelation::ModelEndpointInfo> device_endpoints_info;

  for (const auto &device_info : device_list_) {
    auto device = ResourceManager::GetInstance().GetDeviceInfo(device_info.GetNodeId(),
                                                               device_info.GetDeviceId(),
                                                               device_info.GetType());
    GE_RT_VOID_CHECK_NOTNULL(device);
    if (!device->SupportFlowgw() && (device_info.GetType() == static_cast<int32_t>(CPU))) {
      GELOGI("DynamicSched no host flowgw don't create host request queue.");
      continue;
    }
    if (HasDeployedModelOnDevice(device_info)) {
      const auto queue_name = "queue_response@" + device_info.GetKey() + "@" + std::to_string(sched_queue_id);
      const auto request_queue_name = "queue_request@" + device_info.GetKey() + "@" + std::to_string(sched_queue_id);
      sched_queue_id++;
      device_endpoints_info[device_info.GetKey()].sched_output_queue_names.emplace_back(request_queue_name);
      device_endpoints_info[device_info.GetKey()].sched_input_queue_names.emplace_back(queue_name);
      resolved_model_relation.root_model_endpoint_info.sched_input_queue_names.emplace_back(queue_name);
      resolved_model_relation.root_model_endpoint_info.sched_output_queue_names.emplace_back(request_queue_name);
      GELOGI("DynamicSched model datagw request queue_name=%s, reponse queue_name=%s.",
             request_queue_name.c_str(), queue_name.c_str());

      Endpoint queue_def(queue_name, EndpointType::kQueue);
      QueueNodeUtils(queue_def).SetDepth(kDepDefQueDepth).SetEnqueuePolicy("FIFO").
        SetNodeAction(kQueueActionSched);
      resolved_model_relation.endpoints.emplace_back(queue_def);

      Endpoint request_queue_def(request_queue_name, EndpointType::kQueue);
      QueueNodeUtils(request_queue_def).SetDepth(kDepDefQueDepth).SetEnqueuePolicy("FIFO").
        SetNodeAction(kQueueActionSched);
      resolved_model_relation.endpoints.emplace_back(request_queue_def);
    }
  }

  AddDynamicSchedModel(model_instance_to_models, resolved_model_relation, device_endpoints_info, device_list_);
}

Status HeterogeneousDeployPlanner::AssignSubmodelsQueueDeviceInfo(const std::string &model_name,
                                                                  DeployPlan::SubmodelInfo &submodel_info) const {
  auto queue_device_info = submodel_info.device_info;
  if (queue_device_info.GetType() == static_cast<int32_t>(CPU)) {
    auto device = ResourceManager::GetInstance().GetDeviceInfo(queue_device_info.GetNodeId(),
                                                               queue_device_info.GetDeviceId(),
                                                               queue_device_info.GetType());
    GE_CHECK_NOTNULL(device);
    queue_device_info.SetSupportFlowgw(device->SupportFlowgw());
    if (queue_device_info.GetProxyDeviceId() == -1) {
      auto head_npu_device = ResourceManager::GetInstance().GetHeadNpuDeviceInfo(queue_device_info.GetNodeId());
      if (head_npu_device != nullptr) {
        queue_device_info.SetProxyDeviceId(head_npu_device->GetDeviceId());
      }
    }
  }
  submodel_info.queue_device_info = queue_device_info;
  GELOGI("Model[%s] queues will be deployed on device[%s]",
         model_name.c_str(), submodel_info.queue_device_info.GetDesc().c_str());
  return SUCCESS;
}

void HeterogeneousDeployPlanner::RecordDeviceDeployedModelNum(const DeployPlan::DeviceInfo &device_info) {
  device_deployed_model_num_[device_info]++;
  if (device_info.WithProxy()) {
    device_deployed_model_num_[device_info.ProxyDevice()]++;
  }
}

Status HeterogeneousDeployPlanner::PrepareTargetDevices(std::map<std::string, PneModelPtr> &name_to_models,
                                                        ModelRelation &model_relation) {
  // device selection is not supported yet, deploy to all given devices
  std::map<std::string, std::vector<std::pair<DeployPlan::DeviceInfo, bool>>> target_devices;
  GE_CHK_STATUS_RET_NOLOG(SelectTargetDevice(name_to_models, target_devices));

  // resolve target device after merging
  ModelRelation resolved_model_relation;
  std::map<std::string, DeployPlan::DeviceInfo> model_instance_locations;
  std::map<std::string, std::pair<PneModelPtr, bool>> model_instance_to_models;
  resolved_model_relation.endpoints = model_relation.endpoints;
  resolved_model_relation.root_model_endpoint_info = model_relation.root_model_endpoint_info;
  resolved_model_relation.invoked_model_queue_infos = model_relation.invoked_model_queue_infos;
  for (const auto &it : target_devices) {
    const auto &model_name = it.first;
    int32_t process_id = 0;
    for (const auto &target_device : it.second) {
      const bool is_redundant = target_device.second;
      const auto model_instance_name = model_name + "@" + std::to_string(process_id++) + "@" + target_device.first.GetKey() + "@" +
        std::to_string(is_redundant);
      // add submodel
      (deploy_plan_.GetModelDeployInfos())[model_name][model_instance_name] = {};
      model_instance_to_models.emplace(model_instance_name, std::make_pair(name_to_models[model_name], is_redundant));
      model_instance_locations.emplace(model_instance_name, target_device.first);
      resolved_model_relation.submodel_endpoint_infos.emplace(model_instance_name,
                                                              model_relation.submodel_endpoint_infos.at(model_name));
      auto &model_queue_info = resolved_model_relation.submodel_endpoint_infos.at(model_instance_name);
      model_queue_info.model_name = model_name;
      RecordDeviceDeployedModelNum(target_device.first);
      GELOGI("Model[%s] emplace model instance[%s].", model_name.c_str(), model_instance_name.c_str());
    }
  }
  name_to_models.clear();
  for (auto &model_instance : model_instance_to_models) {
    name_to_models.emplace(model_instance.first, model_instance.second.first);
  }
  DeployPlan::DeviceInfo model_io_device_info{};
  SelectHeadAndTailDevice(model_io_device_info);
  // dynamic sched need send relation to model io flowgw
  RecordDeviceDeployedModelNum(model_io_device_info);
  AddDynamicSchedRelation(name_to_models, resolved_model_relation);
  model_relation = std::move(resolved_model_relation);

  for (const auto &it : model_instance_to_models) {
    const auto &model_instance_name = it.first;
    if (it.second.first == nullptr) {
      continue;
    }
    auto &submodel_info = MutableSubmodelInfo(model_instance_name);
    submodel_info.model = it.second.first;
    submodel_info.device_info = model_instance_locations[model_instance_name];
    submodel_info.is_redundant = it.second.second;
    GE_CHK_STATUS_RET(AssignSubmodelsQueueDeviceInfo(model_instance_name, submodel_info),
                      "Failed to assign model queue device info.");
    GELOGD("Model [%s] will be deployed on device [%s]", model_instance_name.c_str(),
           submodel_info.device_info.GetDesc().c_str());
  }
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::BuildTransferPlan(const DeployPlan::DeviceInfo &local_device,
                                                     const std::vector<DeployPlan::DeviceInfo> &target_devices,
                                                     DeployPlan &deploy_plan) {
  GE_CHK_STATUS_RET_NOLOG(CreateTransferInfo(local_device, target_devices));
  deploy_plan = std::move(deploy_plan_);
  plan_id_gen_++;
  return SUCCESS;
}

Status HeterogeneousDeployPlanner::CreateTransferInfo(const DeployPlan::DeviceInfo &src_device_info,
                                                      const std::vector<DeployPlan::DeviceInfo> &dst_device_infos) {
  if (dst_device_infos.empty()) {
    return SUCCESS;
  }
  const DeployPlan::DeviceInfo &dst_device_info = dst_device_infos[0];
  const std::string route_name = src_device_info.GetKey() + "_TO_" + dst_device_info.GetKey()
                                 + "_T" + std::to_string(plan_id_gen_);
  DeployPlan::QueueInfo src_queue_info{};
  src_queue_info.name = route_name;
  src_queue_info.device_info = src_device_info;
  src_queue_info.depth = 128U;

  DeployPlan::QueueInfo dst_queue_info{};
  dst_queue_info.name = route_name;
  dst_queue_info.device_info = dst_device_info;
  dst_queue_info.depth = 128U;

  int32_t src_queue_index = -1;
  if (src_device_info.GetNodeId() != dst_device_info.GetNodeId()) {
    GE_CHK_STATUS_RET(CreateEndpointInfo(src_queue_info, src_queue_index));
    int32_t dst_queue_index = -1;
    GE_CHK_STATUS_RET(CreateEndpointInfo(dst_queue_info, dst_queue_index));
    int32_t dst_tag_index = -1;
    GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(dst_queue_info, dst_tag_index));
    int32_t dst_tag_group_index = -1;
    std::vector<int32_t> dst_tag_group = {dst_tag_index};
    GE_CHK_STATUS_RET_NOLOG(CreateGroupInfo(src_queue_info, dst_tag_group, dst_tag_group_index));
    AddEndpointBindings(src_queue_index, dst_tag_group_index);

    int32_t src_tag_index = -1;
    GE_CHK_STATUS_RET_NOLOG(CreateGroupEntry(src_queue_info, src_tag_index));
    int32_t src_tag_group_index = -1;
    std::vector<int32_t> src_tag_group = {src_tag_index};
    GE_CHK_STATUS_RET_NOLOG(CreateGroupInfo(dst_queue_info, src_tag_group, src_tag_group_index));
    AddEndpointBindings(src_tag_group_index, dst_queue_index);
  } else {
    GE_CHK_STATUS_RET(CreateEndpointInfo(src_queue_info, src_queue_index));
  }
  return SUCCESS;
}

void HeterogeneousDeployPlanner::SelectHeadAndTailDevice(DeployPlan::DeviceInfo &device_info) {
  // default use current hostcpu device, not soc use first npu device
  DeployPlan::DeviceInfo default_device_info = {};
  auto device = ResourceManager::GetInstance().GetDeviceInfo(default_device_info.GetNodeId(),
                                                             default_device_info.GetDeviceId(),
                                                             default_device_info.GetType());
  if (device != nullptr) {
    default_device_info.SetSupportFlowgw(device->SupportFlowgw());
    auto head_npu_device = ResourceManager::GetInstance().GetHeadNpuDeviceInfo(default_device_info.GetNodeId());
    if (head_npu_device != nullptr) {
      default_device_info.SetProxyDeviceId(head_npu_device->GetDeviceId());
    }
  }
  device_info = default_device_info;
  GELOGI("ModelIO will be deployed on device[%s]", device_info.GetDesc().c_str());
}
}  // namespace ge
