/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy_planner.h"
#include <algorithm>
#include <exception>
#include "nlohmann/json.hpp"
#include "dflow/base/model/endpoint.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr int32_t kLocalNodeId = 0;
constexpr int64_t kDepDefQueDepth = 128L;
constexpr uint32_t kMaxQueueNameLen = 127U;
const DeployPlan::DeviceInfo kLocalDeviceInfo{CPU, kLocalNodeId, 0};
constexpr const char_t *kEnableFusionTrue = "true";
constexpr const char_t *kAttrNameInvokedModelFusionInputs = "_invoked_model_fusion_inputs";
constexpr const char_t *kAttrValueDevicePlacement = "device";
const std::string kDynamicSchedRelationSuffix = "_dynamic_sched";
bool HasIntersection(const std::vector<std::string> &submodel_input, const std::vector<std::string> &root_intput) {
  std::unordered_set<std::string> submodel_inputs(submodel_input.begin(), submodel_input.end());
  for (auto &ele : root_intput) {
    if (submodel_inputs.count(ele) > 0) {
      return true;
    }
  }
  return false;
}

bool IsNeedDeviceQueue(const DeployPlan::SubmodelInfo &submodel_info) {
  if (submodel_info.model == nullptr) {
    return false;
  }
  std::string placement;
  (void)AttrUtils::GetStr(submodel_info.model->GetRootGraph(), ATTR_NAME_FLOW_ATTR_IO_PLACEMENT, placement);
  return placement == kAttrValueDevicePlacement;
}
}  // namespace

std::atomic<int64_t> DeployPlannerBase::endpoint_name_id_gen_{};
std::atomic<int64_t> DeployPlannerBase::plan_id_gen_{};
using DynamicSchedInfo = std::map<std::string, DeployPlan::SubmodelInfo>;

const std::vector<DeployPlan::QueueInfo> &DeployPlan::GetQueueInfoList() const {
  return queues_;
}

const std::vector<DeployPlan::QueueInfo> &DeployPlan::GetGroupEntryInfoList() const {
  return group_entries_;
}

const std::vector<std::pair<int32_t, int32_t>> &DeployPlan::GetQueueBindings() const {
  return queue_bindings_;
}

const std::vector<int32_t> DeployPlan::GetBroadcastIndices(int32_t src_endpoint_index) const {
  std::vector<int32_t> empty;
  std::vector<int32_t> broadcast_indices;
  const auto &it = src_to_dst_endpoints_.find(src_endpoint_index);
  if (it == src_to_dst_endpoints_.cend()) {
    return empty;
  }
  for (const auto &ins_it : it->second) {
    // not support multi-instance model broadcast
    if (ins_it.second.size() > 1U) {
      return empty;
    }
    broadcast_indices.emplace_back(ins_it.second[0]);
  }
  return broadcast_indices;
}

const std::vector<int32_t> &DeployPlan::GetInputQueueIndices() const {
  return root_model_info_.input_queue_indices;
}

const std::vector<int32_t> &DeployPlan::GetOutputQueueIndices() const {
  return root_model_info_.output_queue_indices;
}

const std::map<std::string, std::vector<std::string>> &DeployPlan::GetTrimmingEdgesModelInstances() const {
  return trimming_edges_model_instance_names_;
}

const std::map<std::string, DeployPlan::SubmodelInfo> &DeployPlan::GetSubmodels() const {
  return submodels_;
}

std::map<std::string, DeployPlan::SubmodelInfo> &DeployPlan::MutableSubmodels() {
  return submodels_;
}

const std::map<int32_t, std::vector<int32_t>> &DeployPlan::GetGroups() const {
  return groups_;
}

bool DeployPlan::IsGroupEndpoint(const int32_t queue_index) const {
  return groups_.find(queue_index) != groups_.end();
}

Status DeployPlan::GetQueueInfo(const int32_t queue_index, const DeployPlan::QueueInfo *&queue_info) const {
  if ((queue_index < 0) || (static_cast<size_t>(queue_index) >= queues_.size())) {
    GELOGE(PARAM_INVALID, "Queue index(%d) out of range: [0, %zu)", queue_index, queues_.size());
    return PARAM_INVALID;
  }
  queue_info = &queues_[static_cast<size_t>(queue_index)];
  return SUCCESS;
}

std::vector<int32_t> DeployPlan::GetAllInputQueueIndices() const {
  auto all_indices = root_model_info_.input_queue_indices;
  (void)all_indices.insert(all_indices.cend(), root_model_info_.control_input_queue_indices.cbegin(),
                           root_model_info_.control_input_queue_indices.cend());
  return all_indices;
}

const std::vector<int32_t> &DeployPlan::GetControlInputQueueIndices() const {
  return root_model_info_.control_input_queue_indices;
}

const std::vector<int32_t> &DeployPlan::GetControlOutputQueueIndices() const {
  return root_model_info_.control_output_queue_indices;
}

const DeployPlan::DeviceInfo &DeployPlan::GetRootModelQueueDeviceInfo() const {
  return root_model_info_.queue_device_info;
}

const DeployPlan::DynamicSchedPlan &DeployPlan::GetDynamicSchedPlan() const {
  return dynamic_sched_plan_;
}

void DeployPlan::SetIsDynamicSched(const bool is_dynamic_sched) {
  is_dynamic_sched_ = is_dynamic_sched;
}

const bool &DeployPlan::GetIsDynamicSched() const {
  return is_dynamic_sched_;
}

void DeployPlan::SetEnableExceptionCatch(bool enable_exception_catch) {
  enable_exception_catch_ = enable_exception_catch;
}

bool DeployPlan::IsEnableExceptionCatch() const {
  return enable_exception_catch_;
}

DeployPlan::ModelDeployInfo &DeployPlan::GetModelDeployInfos() {
  return model_deploy_infos_;
}

bool DeployPlan::DeviceInfo::WithProxy() const {
  return (GetType() == static_cast<int32_t>(CPU)) && (GetProxyDeviceId() != -1);
}

DeployPlan::DeviceInfo DeployPlan::DeviceInfo::ProxyDevice() const {
  return DeployPlan::DeviceInfo(static_cast<int32_t>(NPU), node_id_, proxy_device_id_);
}

DeployPlan::DeviceInfo::DeviceInfo(const int32_t type, const int32_t node_id, const int32_t device_id) noexcept
    : DeviceInfo(type, node_id, device_id, (type == static_cast<int32_t>(CPU)) ? -1 : device_id) {}

DeployPlan::DeviceInfo::DeviceInfo(const int32_t type, const int32_t node_id, const int32_t device_id,
                                   const int32_t proxy_device_id) noexcept
    : type_(type),
      node_id_(node_id),
      device_id_(device_id),
      proxy_device_id_(proxy_device_id),
      hcom_device_id_(device_id) {
  key_ = std::to_string(type) + "_" + std::to_string(node_id) + "_" + std::to_string(device_id);
  desc_ = key_ + "(" + std::to_string(proxy_device_id_) + ")";
}

int32_t DeployPlan::DeviceInfo::GetType() const {
  return type_;
}

int32_t DeployPlan::DeviceInfo::GetNodeId() const {
  return node_id_;
}

int32_t DeployPlan::DeviceInfo::GetDeviceId() const {
  return device_id_;
}

int32_t DeployPlan::DeviceInfo::GetProxyDeviceId() const {
  return proxy_device_id_;
}

int32_t DeployPlan::DeviceInfo::GetHcomDeviceId() const {
  return hcom_device_id_;
}

void DeployPlan::DeviceInfo::SetHcomDeviceId(int32_t hcom_device_id) {
  hcom_device_id_ = hcom_device_id;
}

int32_t DeployPlan::DeviceInfo::GetOsId() const {
  return os_id_;
}

void DeployPlan::DeviceInfo::SetOsId(int32_t os_id) {
  os_id_ = os_id;
}

const std::string &DeployPlan::DeviceInfo::GetKey() const {
  return key_;
}

const std::string &DeployPlan::DeviceInfo::GetDesc() const {
  return desc_;
}

const std::vector<int32_t> &DeployPlan::DynamicSchedPlan::GetStatusOutputQueueIndices() const {
  return root_model_info_.status_output_queue_indices;
}

const std::vector<int32_t> &DeployPlan::DynamicSchedPlan::GetSchedOutputQueueIndices() const {
  return root_model_info_.sched_output_queue_indices;
}

const std::vector<int32_t> &DeployPlan::DynamicSchedPlan::GetSchedInputQueueIndices() const {
  return root_model_info_.sched_input_queue_indices;
}

const std::map<int32_t, int32_t> &DeployPlan::DynamicSchedPlan::GetDatagwRequestBindings() const {
  return datagw_request_bindings_;
}

const std::map<int32_t, int32_t> &DeployPlan::DynamicSchedPlan::GetEntryBindings() const {
  return entry_to_dst_index_;
}

const DeployPlan::DynamicSchedIndex &DeployPlan::DynamicSchedPlan::GetModelIndexInfo() const {
  return model_index_info_;
}

const std::map<std::string, uint32_t> &DeployPlan::DynamicSchedPlan::GetModelInstanceNum() const {
  return model_instances_num_;
}

DeployPlanner::DeployPlanner(const PneModelPtr &root_model) : DeployPlannerBase(), root_model_(root_model) {}

Status DeployPlannerBase::BuildPlan(DeployPlan &deploy_plan) {
  deploy_plan_.is_dynamic_sched_ = deploy_plan.GetIsDynamicSched();
  deploy_plan_.enable_exception_catch_ = deploy_plan.IsEnableExceptionCatch();
  GE_CHK_STATUS_RET(Initialize(), "Failed to initialize deploy planner.");
  GE_CHK_STATUS_RET(ParseModelRelation(), "Failed to parse model relation.");
  plan_id_gen_++;
  deploy_plan = std::move(deploy_plan_);
  return SUCCESS;
}

Status DeployPlannerBase::Initialize() {
  GE_CHK_STATUS_RET(PrepareModelsAndRelation(model_relation_), "Failed to prepare");
  UpdateRelationForControlIo();  // add control input/output for submodels if needed
  UpdateRelationForDynamicSched();
  relation_reader_ = MakeUnique<ModelRelationReader>(model_relation_);
  GE_CHECK_NOTNULL(relation_reader_);
  GE_CHK_STATUS_RET(relation_reader_->Initialize(), "Failed to initialize model relation reader");
  const auto &root_model_endpoint_info = model_relation_.root_model_endpoint_info;
  head_model_queue_info_.output_endpoint_names = root_model_endpoint_info.input_endpoint_names;
  head_model_queue_info_.external_output_queue_names = root_model_endpoint_info.external_input_queue_names;
  head_model_queue_info_.model_name = "__head";
  tail_model_queue_info_.input_endpoint_names = root_model_endpoint_info.output_endpoint_names;
  tail_model_queue_info_.external_input_queue_names = root_model_endpoint_info.external_output_queue_names;
  tail_model_queue_info_.model_name = "__tail";
  SelectHeadAndTailDevice(head_model_info_.queue_device_info);
  SelectHeadAndTailDevice(tail_model_info_.queue_device_info);
  return SUCCESS;
}

Status DeployPlanner::PrepareModelsAndRelation(ModelRelation &model_relation) {
  GE_CHECK_NOTNULL(root_model_->GetModelRelation().get());
  ModelRelationFlattener flattener(root_model_);
  std::map<std::string, PneModelPtr> name_to_models;
  GE_CHK_STATUS_RET_NOLOG(flattener.Flatten(model_relation, name_to_models));
  GE_CHK_STATUS_RET_NOLOG(ValidateModelAndRelation(name_to_models, model_relation));
  for (const auto &it : name_to_models) {
    const auto &model_name = it.first;
    const auto &submodel = it.second;
    auto &submodel_info = MutableSubmodelInfo(model_name);
    submodel_info.model = submodel;
    submodel_info.device_info = kLocalDeviceInfo;
    submodel_info.queue_device_info = kLocalDeviceInfo;
    GELOGD("Model[%s] will be deployed on device[%s]", model_name.c_str(), submodel_info.device_info.GetDesc().c_str());
    submodel_info.is_head = false;
  }
  return SUCCESS;
}

void DeployPlannerBase::UpdateForInputControlIo() {
  std::vector<std::string> models_without_input;
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &submodel_endpoint_info = it.second;
    if (submodel_endpoint_info.input_endpoint_names.empty() &&
        submodel_endpoint_info.external_input_queue_names.empty()) {
      // need control input queue
      // all empty goes to LoadModelWithoutQ for now
      if (!submodel_endpoint_info.output_endpoint_names.empty()) {
        GELOGI("submodel [%s] needs control input", it.first.c_str());
        models_without_input.emplace_back(it.first);
      }
    }
  }

  if (!models_without_input.empty()) {
    const std::string control_input_queue_name = "__control_input";
    Endpoint queue_def(control_input_queue_name, EndpointType::kQueue);
    auto queue_def_utils =
        QueueNodeUtils(queue_def).SetDepth(kDepDefQueDepth).SetEnqueuePolicy("FIFO").
        SetNodeAction(kQueueActionControl);

    model_relation_.endpoints.emplace_back(queue_def);
    model_relation_.root_model_endpoint_info.input_endpoint_names.emplace_back(control_input_queue_name);
    for (const auto &model_name : models_without_input) {
      model_relation_.submodel_endpoint_infos[model_name].input_endpoint_names.emplace_back(control_input_queue_name);
      GELOGD("model_name:%s, control_input_name:%s, is_control:%d.", model_name.c_str(),
             control_input_queue_name.c_str(), queue_def_utils.GetIsControl());
    }
  }
}

void DeployPlannerBase::UpdateForOutputControlIo() {
  std::set<std::string> invoked_inputs;
  for (const auto &it : model_relation_.invoked_model_queue_infos) {
    invoked_inputs.insert(it.second.input_queue_names.cbegin(), it.second.input_queue_names.cend());
  }
  std::map<std::string, std::vector<std::string>> models_without_output;
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &submodel_endpoint_info = it.second;
    if (submodel_endpoint_info.output_endpoint_names.empty()) {
      // need control input queue
      // all empty goes to LoadModelWithoutQ for now
      // do not add control output for invoked model
      const auto is_invoked = std::any_of(submodel_endpoint_info.input_endpoint_names.cbegin(),
                                          submodel_endpoint_info.input_endpoint_names.cend(),
                                          [&invoked_inputs](const std::string &endpoint_name) -> bool {
                                            return invoked_inputs.find(endpoint_name) != invoked_inputs.cend();
                                          });
      if ((!is_invoked) &&
          (!submodel_endpoint_info.input_endpoint_names.empty() ||
              !submodel_endpoint_info.external_input_queue_names.empty())) {
        GELOGI("Submodel[%s] needs control output", it.first.c_str());
        models_without_output[submodel_endpoint_info.model_name].emplace_back(it.first);
      }
    }
  }

  for (const auto &it : models_without_output) {
    const auto &model_name = it.first;
    const std::string control_output_queue_name = "__" + model_name + "_control_output";
    Endpoint queue_def(control_output_queue_name, EndpointType::kQueue);
    QueueNodeUtils(queue_def).SetDepth(kDepDefQueDepth).SetEnqueuePolicy("FIFO").
      SetNodeAction(kQueueActionControl);

    model_relation_.endpoints.emplace_back(queue_def);
    model_relation_.root_model_endpoint_info.output_endpoint_names.emplace_back(control_output_queue_name);
    for (const auto &model_instance_name : it.second) {
      model_relation_.submodel_endpoint_infos[model_instance_name].output_endpoint_names.emplace_back(
          control_output_queue_name);
    }
  }
}

void DeployPlannerBase::UpdateRelationForControlIo() {
  UpdateForInputControlIo();
  UpdateForOutputControlIo();
}

Status DeployPlannerBase::ValidateModelAndRelation(const std::map<std::string, PneModelPtr> &models,
                                                   const ModelRelation &model_relation) {
  // check all model in model_relation exist in RootModel
  for (const auto &it : model_relation.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel = models.find(model_instance_name);
    if (submodel == models.end()) {
      GELOGE(PARAM_INVALID, "model exists in ModelRelation bot not found in RootModel, name = %s",
             model_instance_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::ParseModelRelation() {
  MarkMultiDeployedModels();
  GenerateDynamicSchedModelId();
  GE_CHK_STATUS_RET(AssignEnqueueQueues(), "Failed to assign enqueue queues");
  GE_CHK_STATUS_RET(AssignDynamicSchedEnqueueQueues(), "Failed to assign dynamic sched enqueue queues");
  GE_CHK_STATUS_RET(ResolveEnqueueFusion(), "Failed to resolve enqueue fusion");
  GE_CHK_STATUS_RET(ResolveInvokedFusion(), "Failed to resolve invoked fusion");
  GE_CHK_STATUS_RET(ResolveDataFlows(), "Failed to resolve flow relations");
  GE_CHK_STATUS_RET(AdjustEnqueueDevices(), "Failed to adjust enqueue devices");
  LogDataFlow();
  GE_CHK_STATUS_RET(ResolveReusableQueues(), "Failed to resolve reusable queues");
  GE_CHK_STATUS_RET(AssignDequeueQueues(), "Failed to assign dequeue queues");
  GE_CHK_STATUS_RET(AssignDynamicSchedDequeueQueues(), "Failed to assign dynamic sched dequeue queues");
  GE_CHK_STATUS_RET(BindOutputToRemoteInputs(), "Failed to bind output groups");
  GE_CHK_STATUS_RET(BindRemoteOutputGroupToInput(), "Failed to bind input groups");
  UpdateDeployPlan();
  UpdateDynamicSchedDeployPlan();
  GE_CHK_STATUS_RET(BuildDynamicSchedInfo(), "Failed to build dynamic sched info");
  GE_CHK_STATUS_RET(SetHeadNodeInfo(), "Failed to set head node info");
  return SUCCESS;
}

Status DeployPlannerBase::AssignEnqueueQueues() {
  GE_CHK_STATUS_RET_NOLOG(
      CreateOutputQueueDefs(head_model_queue_info_.model_name, head_model_queue_info_.output_endpoint_names));
  GE_CHK_STATUS_RET_NOLOG(CreateOutputQueueDefs(head_model_queue_info_.model_name,
                                                head_model_queue_info_.external_output_queue_names, false));
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    GE_CHK_STATUS_RET_NOLOG(CreateOutputQueueDefs(model_instance_name, it.second.output_endpoint_names));
    const auto &invoke_model_keys = it.second.invoke_model_keys;
    for (const auto &invoke_model_key : invoke_model_keys) {
      auto invoked_model_queue_info = relation_reader_->GetInvokedModelQueueInfo(invoke_model_key);
      GE_CHECK_NOTNULL(invoked_model_queue_info, ", get invoked model queue info is null, model_name = %s",
                       model_instance_name.c_str());
      // invoked model input is as feed for this model
      GE_CHK_STATUS_RET_NOLOG(
          CreateFeedEndpoints(model_instance_name, invoked_model_queue_info->input_queue_names, invoke_model_key));
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::GetInvokedModelFusionInputs(const PneModelPtr model,
                                                      std::map<std::string, std::string> &fusion_inputs) {
  std::string invoked_model_fusion_inputs_str;
  if (model != nullptr && model->GetRootGraph() != nullptr) {
    (void) AttrUtils::GetStr(model->GetRootGraph(), kAttrNameInvokedModelFusionInputs, invoked_model_fusion_inputs_str);
  }
  if (invoked_model_fusion_inputs_str.empty()) {
    return SUCCESS;
  }

  nlohmann::json js;
  try {
    js = nlohmann::json::parse(invoked_model_fusion_inputs_str);
    for (const auto &item : js.items()) {
      fusion_inputs[item.key()] = item.value().get<std::string>();
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Invalid json format of invoked model fusion inputs, exception:%s", e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status DeployPlannerBase::ResolveInvokedFusion() {
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    auto &submodel_info = MutableSubmodelInfo(model_instance_name);
    std::map<std::string, std::string> invoked_model_fusion_inputs;
    GE_CHK_STATUS_RET(GetInvokedModelFusionInputs(submodel_info.model, invoked_model_fusion_inputs),
                      "Failed to get invoked model fusion inputs");
    if (invoked_model_fusion_inputs.empty()) {
      GELOGI("Model[%s] is no need fusion invoked inputs without attr[%s].",
             model_instance_name.c_str(), kAttrNameInvokedModelFusionInputs);
      continue;
    }

    const auto &invoke_model_keys = it.second.invoke_model_keys;
    for (const auto &invoke_model_key : invoke_model_keys) {
      const auto &it2 = invoked_model_fusion_inputs.find(invoke_model_key);
      if (it2 == invoked_model_fusion_inputs.cend()) {
        GELOGI("Invoked model[invoke key:%s] is no need fusion invoked inputs without attr[%s].",
               model_instance_name.c_str(), kAttrNameInvokedModelFusionInputs);
        continue;
      }
      const auto &fusion_inputs = it2->second;
      auto invoked_model_queue_info = relation_reader_->GetInvokedModelQueueInfo(invoke_model_key);
      GE_CHECK_NOTNULL(invoked_model_queue_info, ", get invoked model queue info is null, model_name = %s",
                       model_instance_name.c_str());
      // invoked model input is as feed for this model
      GE_CHK_STATUS_RET(ResolveModelInvokedFusion(model_instance_name,
                                                  invoked_model_queue_info->input_queue_names,
                                                  invoke_model_key,
                                                  fusion_inputs),
                        "Failed to resolve model invoked fusion");
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::ParseInputIndexWithRange(const std::string &fusion_input_str,
                                                   std::vector<size_t> &fusion_input_index_list) {
  std::string range_begin_str;
  std::string range_end_str;
  auto range_str = ge::StringUtils::Split(fusion_input_str, '~');
  GE_CHECK_GE(range_str.size(), 1UL);
  GE_CHECK_LE(range_str.size(), 2UL);
  range_begin_str = range_str[0];
  if (range_str.size() == 1UL) {
    range_end_str = range_str[0];
  } else {
    range_end_str = range_str[1];
  }

  int32_t range_begin = 0;
  int32_t range_end = 0;
  try {
    range_begin = std::stoi(range_begin_str);
    range_end = std::stoi(range_end_str);
  } catch (...) {
    GELOGE(FAILED, "Fusion input str[%s] is illegal.", fusion_input_str.c_str());
    return FAILED;
  }
  GE_CHECK_GE(range_end, range_begin);
  std::set<size_t> index_list_set;
  for (int32_t i = range_begin; i <= range_end; ++i) {
    index_list_set.emplace(static_cast<size_t>(i));
  }
  fusion_input_index_list.insert(fusion_input_index_list.end(), index_list_set.begin(), index_list_set.end());
  return SUCCESS;
}

Status DeployPlannerBase::ParseInvokedModelFusionInputs(const std::string &fusion_inputs_str,
                                                        std::vector<std::vector<size_t>> &fusion_inputs_list) {
  auto normalized_fusion_inputs_str = StringUtils::ReplaceAll(fusion_inputs_str, " ", "");
  GE_CHK_BOOL_RET_STATUS(!normalized_fusion_inputs_str.empty(), PARAM_INVALID,
                         "Fusion inputs[%s] is empty after normalized.", fusion_inputs_str.c_str());
  auto fusion_inputs_str_list = ge::StringUtils::Split(normalized_fusion_inputs_str, ';');
  GE_CHK_BOOL_RET_STATUS(fusion_inputs_str_list.size() != 0UL, PARAM_INVALID,
                         "Invalid format of fusion inputs:%s to separated by ';'", fusion_inputs_str.c_str());
  for (const auto &fusion_input_str_list_str : fusion_inputs_str_list) {
    std::vector<size_t> fusion_input_index_list;
    auto fusion_input_str_list = ge::StringUtils::Split(fusion_input_str_list_str, ',');
    GE_CHK_BOOL_RET_STATUS(fusion_input_str_list.size() != 0UL, PARAM_INVALID,
                          "Invalid format of fusion inputs:%s to separated by ','", fusion_inputs_str.c_str());
    for (const auto &fusion_input_str : fusion_input_str_list) {
      GE_CHK_STATUS_RET(ParseInputIndexWithRange(fusion_input_str, fusion_input_index_list),
                        "Failed to parse input index with range");
    }
    GELOGI("Parse fusion input string[%s] success, fusion list = %s",
           fusion_input_str_list_str.c_str(), ToString(fusion_input_index_list).c_str());
    fusion_inputs_list.emplace_back(fusion_input_index_list);
  }
  return SUCCESS;
}

Status DeployPlannerBase::ResolveModelInvokedFusion(const std::string &model_instance_name,
                                                    const std::vector<std::string> &queue_names,
                                                    const std::string &invoke_key,
                                                    const std::string &fusion_inputs) {
  std::vector<const Endpoint *> endpoints;
  GE_CHK_STATUS_RET(relation_reader_->BatchGetEndpoints(queue_names, endpoints), "Failed to batch get endpoints");
  auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  if (endpoints.size() <= 1UL) {
    GELOGI("Model[%s] input size[%zu] <= 1, no need to fusion.", model_instance_name.c_str(),
           endpoints.size());
    return SUCCESS;
  }

  std::vector<std::vector<size_t>> fusion_inputs_list;
  GE_CHK_STATUS_RET(ParseInvokedModelFusionInputs(fusion_inputs, fusion_inputs_list),
                    "Failed to parse invoked model fusion inputs");
  for (const auto &fusion_list : fusion_inputs_list) {
    GE_CHECK_GE(fusion_list.size(), 1UL);
    auto begin_index = *fusion_list.begin();
    GE_CHECK_LE(begin_index + 1, submodel_info.invoked_model_queue_infos[invoke_key].feed_queue_indices.size());
    auto fusion_index = submodel_info.invoked_model_queue_infos[invoke_key].feed_queue_indices[begin_index];
    int32_t i = -1;
    for (auto input_index : fusion_list) {
      // first is fusion index
      i++;
      if (i == 0) {
        continue;
      }
      const auto &feed_queue_indices = submodel_info.invoked_model_queue_infos[invoke_key].feed_queue_indices;
      GE_CHK_BOOL_RET_STATUS(input_index < feed_queue_indices.size(), FAILED,
                             "Failed to check input index[%zu], must < %zu", input_index, feed_queue_indices.size());
      auto index = feed_queue_indices[input_index];
      auto &endpoint = deploy_plan_.queues_[static_cast<size_t>(index)];
      endpoint.ref_index = fusion_index;
      endpoint.fusion_offset = i;
      GELOGI("Input[%zu] is fused, index = %d, fusion index = %d, fusion offset = %d.",
             input_index, index, fusion_index, endpoint.fusion_offset);
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::ResolveEnqueueFusion() {
  std::string enable_fusion;
  (void)ge::GetContext().GetOption(OPTION_EXEC_ENABLE_FUSION, enable_fusion);
  if (enable_fusion != kEnableFusionTrue) {
    GELOGI("Option[%s] value[%s] means no need to fusion.", OPTION_EXEC_ENABLE_FUSION, enable_fusion.c_str());
    return SUCCESS;
  }

  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel_endpoint_info = it.second;
    GELOGI("Resolve model[%s] placement begin.", model_instance_name.c_str());
    GE_CHK_STATUS_RET_NOLOG(ResolveInputsPlacement(model_instance_name, submodel_endpoint_info));
  }
  GE_CHK_STATUS_RET_NOLOG(ResolveInputsPlacement(tail_model_queue_info_.model_name, tail_model_queue_info_));
  GE_CHK_STATUS_RET_NOLOG(ResolveModelFusion(head_model_queue_info_.model_name, head_model_queue_info_));
  return SUCCESS;
}

Status DeployPlannerBase::ResolveInputsPlacement(const std::string &model_instance_name,
                                                 const ModelRelation::ModelEndpointInfo &model_endpoint_info) {
  const std::set<std::string> kSupportFusionEngines = {PNE_ID_NPU};
  const auto &model_type = GetSubmodelType(model_instance_name);
  bool support_fusion = kSupportFusionEngines.find(model_type) != kSupportFusionEngines.end();
  auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  std::vector<const Endpoint *> model_input_endpoints;
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.input_endpoint_names, model_input_endpoints));
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.sched_input_queue_names, model_input_endpoints));
  for (size_t i = 0UL; i < model_input_endpoints.size(); ++i) {
    const auto *queue_def = model_input_endpoints[i];
    const auto &queue_name = queue_def->GetName();
    if (!support_fusion) {
      disable_fusion_queues_.emplace(queue_name);
    }
    dequeue_placements_[queue_name].emplace(submodel_info.queue_device_info.GetKey());
  }
  return SUCCESS;
}

Status DeployPlannerBase::ResolveModelFusion(const std::string &model_instance_name,
                                             const ModelRelation::ModelEndpointInfo &model_endpoint_info) {
  std::vector<const Endpoint *> model_output_endpoints;
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.output_endpoint_names, model_output_endpoints));
  auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  if (model_output_endpoints.size() <= 1UL) {
    GELOGI("Model[%s] input size[%zu] <= 1, no need to fusion.", model_instance_name.c_str(),
           model_output_endpoints.size());
    return SUCCESS;
  }

  auto fusion_index = submodel_info.output_queue_indices[0];
  auto fusion_name = model_output_endpoints[0]->GetName();
  for (size_t i = 1UL; i < model_output_endpoints.size(); ++i) {
    const auto *queue_def = model_output_endpoints[i];
    const auto &queue_name = queue_def->GetName();
    if (!CanBeFused(fusion_name, queue_name)) {
      GELOGI("Endpoint[%s] can not be fused.", queue_name.c_str());
      break;
    }
    auto index = submodel_info.output_queue_indices[i];
    auto &endpoint = deploy_plan_.queues_[static_cast<size_t>(index)];
    endpoint.ref_index = fusion_index;
    endpoint.fusion_offset = static_cast<int32_t>(i);
    GELOGI("Endpoint[%s] is fused, fusion endpoint name = %s, index = %d, fusion index = %d, fusion offset = %d.",
           queue_name.c_str(), fusion_name.c_str(), index, fusion_index, endpoint.fusion_offset);
  }
  return SUCCESS;
}

bool DeployPlannerBase::CanBeFused(const std::string &fusion_name, const std::string &endpoint_name) {
  if (disable_fusion_queues_.find(endpoint_name) != disable_fusion_queues_.end()) {
    GELOGI("Endpoint[%s] can not be fused.", endpoint_name.c_str());
    return false;
  }
  return dequeue_placements_[fusion_name] == dequeue_placements_[endpoint_name];
}

void DeployPlannerBase::MarkMultiDeployedModels() {
  // key: model_name value: {key: device, value:{model_instace_name}}
  std::map<std::string, std::map<std::string, std::vector<std::string>>> model_instances;
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    const auto &device_info = MutableSubmodelInfo(model_instance_name).queue_device_info;
    const auto &submodel_endpoint_info = it.second;
    model_instances[submodel_endpoint_info.model_name][device_info.GetKey()].emplace_back(model_instance_name);
    instance_to_model_name_[model_instance_name] = submodel_endpoint_info.model_name;
    model_deploy_locations_[submodel_endpoint_info.model_name].emplace_back(std::make_pair(model_instance_name,
                                                                                           device_info));
  }

  int32_t model_id = 1;
  for (const auto &it : model_instances) {
    model_name_to_id_[it.first] = model_id++;
    const auto &model_name = it.first;
    const auto &model_instance_list = it.second;
    if (model_instance_list.size() > 1U || model_instance_list.begin()->second.size() > 1U) {
      GELOGI("Submodel[%s] is multiple deployed.", model_name.c_str());
      for (const auto &dev_to_model_instance_name : model_instance_list) {
        const auto &device_key = dev_to_model_instance_name.first;
        int32_t process_id = 0;
        for (const auto &model_instance_name : dev_to_model_instance_name.second) {
          auto &submodel_info = MutableSubmodelInfo(model_instance_name);
          submodel_info.process_id = process_id++;
          GELOGD("Submodel[%s] is deploy to device[%s], instance = %s, process_id = %d.",
                 model_name.c_str(), device_key.c_str(), model_instance_name.c_str(), submodel_info.process_id);
        }
      }
    }
  }
}

bool DeployPlannerBase::CanConnectWithQ(const DeployPlan::DeviceInfo &src_device_info,
                                        const DeployPlan::DeviceInfo &dst_device_info) {
  GELOGI("Check can connect with queue, src device[%s], dst device[%s].",
         src_device_info.GetDesc().c_str(), dst_device_info.GetDesc().c_str());
  return (src_device_info.GetNodeId() == dst_device_info.GetNodeId()) &&
         ((src_device_info.GetType() != dst_device_info.GetType()) ||
          (src_device_info.GetDeviceId() == dst_device_info.GetDeviceId()) ||
          (src_device_info.GetOsId() == dst_device_info.GetOsId() &&
           src_device_info.GetHcomDeviceId() == dst_device_info.GetHcomDeviceId()));
}

bool DeployPlannerBase::CanConnectWithLocalQ(const DeployPlan::DeviceInfo &src_device_info,
                                             const DeployPlan::DeviceInfo &dst_device_info) {
  GELOGI("Check can directly connect with queue, src device[%s], dst device[%s].",
         src_device_info.GetDesc().c_str(), dst_device_info.GetDesc().c_str());
  return (src_device_info.GetKey() == dst_device_info.GetKey()) ||
         ((src_device_info.GetType() == dst_device_info.GetType()) &&
          (src_device_info.GetOsId() == dst_device_info.GetOsId()) &&
          (src_device_info.GetHcomDeviceId() == dst_device_info.GetHcomDeviceId()));
}

bool DeployPlannerBase::IsContainInvokedModel(const std::string &src_model_instance_name,
                                              const std::string &dst_model_instance_name) {
  std::vector<std::string> model_instance_names;
  model_instance_names.push_back(src_model_instance_name);
  model_instance_names.push_back(dst_model_instance_name);
  GELOGI("Try find if containing invoked nn model, src model instance name[%s], dst model instance name[%s].",
         src_model_instance_name.c_str(), dst_model_instance_name.c_str());
  for (const auto &model_instance_name : model_instance_names) {
    auto &submodel_info = MutableSubmodelInfo(model_instance_name);
    if (submodel_info.model != nullptr) {
      bool is_invoked_model = false;
      (void)AttrUtils::GetBool(submodel_info.model->GetRootGraph(),
                               ATTR_NAME_DATA_FLOW_UDF_INVOKED_NN, is_invoked_model);
      if (is_invoked_model) {
        return true;
      }
    }
  }
  return false;
}

void DeployPlannerBase::AddTrimmingEdgesModelInstance(const std::string &src_model_instance_name,
                                                      const std::string &dst_model_instance_name) {
  deploy_plan_.trimming_edges_model_instance_names_[src_model_instance_name].emplace_back(dst_model_instance_name);
  GELOGI("Added trimming edges model instance, src model instance name[%s], dst model instance name[%s].",
         src_model_instance_name.c_str(), dst_model_instance_name.c_str());
}

bool DeployPlannerBase::CheckSkipBinding(const std::string &src_model_instance_name,
                                         const std::string &dst_model_instance_name) {
  const auto &src_it = instance_to_model_name_.find(src_model_instance_name);
  const auto &dst_it = instance_to_model_name_.find(dst_model_instance_name);
  if (src_it == instance_to_model_name_.cend() || dst_it == instance_to_model_name_.cend()) {
    GELOGI("Can not find model name according instance, src[%s], dst[%s].",
           src_model_instance_name.c_str(), dst_model_instance_name.c_str());
    return false;
  }
  const auto &src_model_name = src_it->second;
  const auto &dst_model_name = dst_it->second;
  const auto &src_model_location_it = model_deploy_locations_.find(src_model_name);
  const auto &dst_model_location_it = model_deploy_locations_.find(dst_model_name);
  if (src_model_location_it == model_deploy_locations_.cend() ||
      dst_model_location_it == model_deploy_locations_.cend()) {
    GELOGI("Failed to find model location, src model_name = %s, dst model_name = %s.",
           src_model_name.c_str(), dst_model_name.c_str());
    return false;
  }

  const auto &src_model_location = src_model_location_it->second;
  const auto &dst_model_location = dst_model_location_it->second;
  if (src_model_location.size() != dst_model_location.size()) {
    GELOGI("Model deployed instance num is different, src[%s] is %zu, dst[%s] is %zu.",
           src_model_name.c_str(), src_model_location.size(), dst_model_name.c_str(), dst_model_location.size());
    return false;
  }
  if (src_model_location.size() <= 1U) {
    GELOGI("Model is not muilti deployed, model name = %s.", src_model_name.c_str());
    return false;
  }

  size_t src_model_instance_index = 0U;
  size_t dst_model_instance_index = 0U;
  const bool has_invoked = IsContainInvokedModel(src_model_instance_name, dst_model_instance_name);
  for (size_t i = 0U; i < src_model_location.size(); ++i) {
    if (!has_invoked && !CanConnectWithQ(src_model_location[i].second, dst_model_location[i].second)) {
      GELOGI("Model instances can not connect with queue.");
      return false;
    }
    if (src_model_location[i].first == src_model_instance_name) {
      src_model_instance_index = i;
    }
    if (dst_model_location[i].first == dst_model_instance_name) {
      dst_model_instance_index = i;
    }
  }
  auto skip_binding = src_model_instance_index != dst_model_instance_index;
  if (!skip_binding && !has_invoked) {
    // record model instance name in same device
    AddTrimmingEdgesModelInstance(src_model_instance_name, dst_model_instance_name);
  }
  GELOGI("Model instances deployed on same devices, skip binding = %d, src instance[%s] index = %zu, "
         "dst instance[%s] index = %zu.", static_cast<int32_t>(skip_binding), src_model_instance_name.c_str(),
         src_model_instance_index, dst_model_instance_name.c_str(), dst_model_instance_index);
  return skip_binding;
}

Status DeployPlannerBase::ResolveDataFlows() {
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel_endpoint_info = it.second;
    GE_CHK_STATUS_RET_NOLOG(ResolveModelInputs(model_instance_name, submodel_endpoint_info));
    GE_CHK_STATUS_RET_NOLOG(ResolveModelDynamicInputs(model_instance_name, submodel_endpoint_info));
  }
  GE_CHK_STATUS_RET_NOLOG(ResolveModelInputs(tail_model_queue_info_.model_name, tail_model_queue_info_));
  GE_CHK_STATUS_RET_NOLOG(ResolveModelDynamicInputs(tail_model_queue_info_.model_name, tail_model_queue_info_));
  return SUCCESS;
}

Status DeployPlannerBase::ResolveModelInputs(const std::string &model_instance_name,
                                             const ModelRelation::ModelEndpointInfo &model_endpoint_info) {
  std::vector<const Endpoint *> model_input_endpoints;
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.input_endpoint_names, model_input_endpoints));
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.external_input_queue_names, model_input_endpoints));
  std::vector<ModelQueueIndex> model_queue_ids;
  model_queue_ids.reserve(model_input_endpoints.size());
  for (size_t input_index = 0UL; input_index < model_endpoint_info.input_endpoint_names.size(); ++input_index) {
    ModelQueueIndex input_queue_index{model_endpoint_info.model_name, "", static_cast<int32_t>(input_index)};
    model_queue_ids.emplace_back(std::move(input_queue_index));
  }

  // external index mark as -1;
  ModelQueueIndex external_queue_index{model_endpoint_info.model_name, "", -1};
  model_queue_ids.resize(model_input_endpoints.size(), external_queue_index);

  for (const auto &invoke_model_key : model_endpoint_info.invoke_model_keys) {
    auto invoked_model_queue_info = relation_reader_->GetInvokedModelQueueInfo(invoke_model_key);
    GE_ASSERT_NOTNULL(invoked_model_queue_info,
                      ", failed to get invoked model queue info, model_instance_name=%s, invoke_model_key=%s.",
                      model_instance_name.c_str(), invoke_model_key.c_str());
    // invoke output is as fetch input here
    GE_CHK_STATUS_RET_NOLOG(
        relation_reader_->BatchGetEndpoints(invoked_model_queue_info->output_queue_names, model_input_endpoints));
    for (size_t feed_index = 0UL; feed_index < invoked_model_queue_info->output_queue_names.size(); ++feed_index) {
      ModelQueueIndex feed_queue_index{model_endpoint_info.model_name, invoke_model_key,
                                       static_cast<int32_t>(feed_index)};
      model_queue_ids.emplace_back(std::move(feed_queue_index));
    }
  }

  GE_CHK_BOOL_RET_STATUS(model_queue_ids.size() == model_input_endpoints.size(), INTERNAL_ERROR,
                         "model_queue_ids.size=%zu is not same as model_input_endpoints.size=%zu, model=%s",
                         model_queue_ids.size(), model_input_endpoints.size(), model_instance_name.c_str());

  auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  for (size_t i = 0UL; i < model_input_endpoints.size(); ++i) {
    const auto &model_queue_id = model_queue_ids[i];
    const auto endpoint = model_input_endpoints[i];
    const auto &endpoint_name = endpoint->GetName();
    if (endpoint->GetEndpointType() == EndpointType::kEvent) {
      GELOGD("Endpoint name = %s is event def. Skip bind endpoints.", endpoint_name.c_str());
      continue;
    }
    const auto &src_endpoint_indices = src_endpoint_indices_[endpoint_name];
    if (src_endpoint_indices.empty()) {
      GELOGE(PARAM_INVALID, "Failed to find enqueue operation for queue [%s]", endpoint_name.c_str());
      return PARAM_INVALID;
    }
    for (auto src_endpoint_index : src_endpoint_indices) {
      const auto &src_endpoint = deploy_plan_.queues_[static_cast<size_t>(src_endpoint_index)];
      // group to group : only deal with bind_relation in same node and same device
      if (CheckSkipBinding(src_endpoint.model_instance_name, model_instance_name)) {
        GELOGI("Skip bind endpoints: name = %s, from %s to %s:%d@%s", endpoint_name.c_str(),
               src_endpoint.model_instance_name.c_str(), model_endpoint_info.model_name.c_str(), model_queue_id.id,
               submodel_info.queue_device_info.GetDesc().c_str());
        continue;
      }

      auto &dst_endpoint_groups = endpoint_pairs_[src_endpoint_index];
      auto queue_info = BuildQueueInfo(*endpoint, model_instance_name);
      GE_CHK_STATUS_RET(AdjustDequeueDevice(queue_info, src_endpoint_indices), "Failed to adjust dequeue device");
      queue_info.name = GetEndpointFullName(queue_info, model_queue_id);
      relation_dst_to_src_[queue_info.name].emplace(src_endpoint_index);
      GELOGD("Bind endpoints: name = %s, from %s to %s:%d@%s, invoke_key=%s, queue device info=%s.",
             endpoint_name.c_str(), src_endpoint.model_instance_name.c_str(), model_endpoint_info.model_name.c_str(),
             model_queue_id.id, submodel_info.device_info.GetDesc().c_str(), model_queue_id.invoke_key.c_str(),
             queue_info.device_info.GetDesc().c_str());
      dst_endpoint_groups[model_queue_id].emplace_back(std::move(queue_info));
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::AdjustDequeueDevice(DeployPlan::QueueInfo &dst_endpoint,
                                              const std::vector<int32_t> &src_endpoint_indices) {
  if (!dst_endpoint.device_info.WithProxy()) {
    return SUCCESS;
  }

  // prioritizing the use of local queues
  if (dst_endpoint.device_info.SupportFlowgw()) {
    return SUCCESS;
  }

  std::map<DeployPlan::DeviceInfo, size_t> local_device_used;
  for (auto src_endpoint_index : src_endpoint_indices) {
    const auto &src_endpoint = deploy_plan_.queues_[static_cast<size_t>(src_endpoint_index)];
    if (src_endpoint.device_info.GetNodeId() == dst_endpoint.device_info.GetNodeId()) {
      local_device_used[src_endpoint.device_info]++;
    }
  }

  // The fault recovery scenario relies on overloading devices with multiple instances, use overloaded devices
  if ((!IsMultiDeployed(dst_endpoint.model_instance_name)) &&
      (local_device_used.size() == 1U) &&
      (local_device_used.begin()->first.GetType() == static_cast<int32_t>(NPU))) {
    GELOGI("Adjust queue[%s] device from [%s] to [%s]",
           dst_endpoint.name.c_str(),
           dst_endpoint.device_info.GetDesc().c_str(), local_device_used.begin()->first.GetDesc().c_str());
    dst_endpoint.device_info = local_device_used.begin()->first;
    return SUCCESS;
  }

  // no flowgw use proxy queue default
  dst_endpoint.device_info = dst_endpoint.device_info.ProxyDevice();
  return SUCCESS;
}

Status DeployPlannerBase::AdjustEnqueueDevice(
    DeployPlan::QueueInfo &src_endpoint,
    const std::map<ModelQueueIndex, std::vector<DeployPlan::QueueInfo>> &dst_endpoints) const {
  if (!src_endpoint.device_info.WithProxy()) {
    return SUCCESS;
  }

  // prioritizing the use of local queues
  if (src_endpoint.device_info.SupportFlowgw()) {
    return SUCCESS;
  }

  std::map<DeployPlan::DeviceInfo, size_t> local_device_used;
  for (const auto &dst_loc_and_queue_info : dst_endpoints) {
    for (const auto &dst_queue_info : dst_loc_and_queue_info.second) {
      if (src_endpoint.device_info.GetNodeId() == dst_queue_info.device_info.GetNodeId()) {
        local_device_used[dst_queue_info.device_info]++;
      }
    }
  }

  // The fault recovery scenario relies on overloading devices with multiple instances, use overloaded devices
  if ((!IsMultiDeployed(src_endpoint.model_instance_name)) &&
      (local_device_used.size() == 1U) &&
      (local_device_used.begin()->first.GetType() == static_cast<int32_t>(NPU))) {
    GELOGI("Adjust queue[%s] device from [%s] to [%s]",
           src_endpoint.name.c_str(),
           src_endpoint.device_info.GetDesc().c_str(), local_device_used.begin()->first.GetDesc().c_str());
    src_endpoint.device_info = local_device_used.begin()->first;
    return SUCCESS;
  }

  // no flowgw use proxy queue default
  src_endpoint.device_info = src_endpoint.device_info.ProxyDevice();
  return SUCCESS;
}

Status DeployPlannerBase::AdjustEnqueueDevices() {
  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    auto &src_endpoint = deploy_plan_.queues_[src_endpoint_idx];
    GE_CHK_STATUS_RET(AdjustEnqueueDevice(src_endpoint, endpoint_pair.second), "Failed to adjust enqueue device");
  }

  for (const auto &endpoint_pair : deploy_plan_.dynamic_sched_plan_.endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    auto &src_endpoint = deploy_plan_.queues_[src_endpoint_idx];
    GE_CHK_STATUS_RET(AdjustEnqueueDevice(src_endpoint, endpoint_pair.second), "Failed to adjust dynamic enqueue device");
  }
  return SUCCESS;
}

void DeployPlannerBase::LogDataFlow() const {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }

  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    const auto &src_endpoint_info = deploy_plan_.queues_[src_endpoint_idx];

    std::map<ModelQueueIndex, std::vector<std::string>> group_by_dst_loc;
    for (const auto &dst_loc_and_queue_info : endpoint_pair.second) {
      for (const auto &dst_queue_info : dst_loc_and_queue_info.second) {
        group_by_dst_loc[dst_loc_and_queue_info.first].emplace_back(dst_queue_info.device_info.GetDesc());
      }
    }
    GELOGD("Bindings for queue [%s@%s] are:", src_endpoint_info.name.c_str(),
           src_endpoint_info.device_info.GetDesc().c_str());
    for (const auto &it : group_by_dst_loc) {
      GELOGD("    %s:%d@%s", it.first.model_name.c_str(), it.first.id, ToString(it.second).c_str());
    }
  }
}

Status DeployPlannerBase::ResolveReusableQueues() {
  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    auto &src_endpoint_info = deploy_plan_.queues_[src_endpoint_idx];
    const auto &queue_name = src_endpoint_info.name;
    if (!src_endpoint_info.owned) {
      GELOGD("Queue[%s@%s] is external.", queue_name.c_str(), src_endpoint_info.device_info.GetDesc().c_str());
      continue;
    }

    if (endpoint_pair.second.size() != 1U) {
      GELOGD("Queue[%s@%s] has one-to-many relation to models", queue_name.c_str(),
             src_endpoint_info.device_info.GetDesc().c_str());
      continue;
    }

    const auto &dst_queue_infos = *endpoint_pair.second.begin();
    if (dst_queue_infos.second.size() != 1U) {
      GELOGD("Queue[%s@%s] has multi-device dest endpoints", queue_name.c_str(),
             src_endpoint_info.device_info.GetDesc().c_str());
      continue;
    }

    const auto &dst_endpoint_name = dst_queue_infos.second.begin()->name;
    const auto &it = relation_dst_to_src_.find(dst_endpoint_name);
    if (it != relation_dst_to_src_.end() && it->second.size() != 1U) {
      GELOGD("Queue[%s@%s] has many-to-one src endpoints.", dst_endpoint_name.c_str(),
             dst_queue_infos.second.begin()->device_info.GetDesc().c_str());
      continue;
    }

    const auto &dst_device_info = dst_queue_infos.second.begin()->device_info;
    if (src_endpoint_info.device_info.GetNodeId() != dst_device_info.GetNodeId()) {
      GELOGD("Queue[%s@%d] has diff node[%d] dest endpoints",
             queue_name.c_str(), src_endpoint_info.device_info.GetNodeId(), dst_device_info.GetNodeId());
      continue;
    }

    if ((src_endpoint_info.device_info.GetType() == dst_device_info.GetType()) &&
        (src_endpoint_info.device_info.GetType() == static_cast<int32_t>(NPU)) &&
        (src_endpoint_info.device_info.GetDeviceId() != dst_device_info.GetDeviceId())) {
      GELOGD("Queue[%s@%s] on npu has diff device endpoints, device = [%s]", queue_name.c_str(),
             src_endpoint_info.device_info.GetDesc().c_str(), dst_device_info.GetDesc().c_str());
      continue;
    }

    if (src_endpoint_info.device_info.GetType() != dst_device_info.GetType() &&
        dst_device_info.GetType() == static_cast<int32_t>(NPU)) {
      GELOGI("Queue[%s@%s] reuse npu device[%s] on same node.",
             queue_name.c_str(), src_endpoint_info.device_info.GetDesc().c_str(), dst_device_info.GetDesc().c_str());
      src_endpoint_info.device_info = dst_device_info;
    }
    GELOGD("Queue[%s@%s] is reusable, index = %d", queue_name.c_str(), src_endpoint_info.device_info.GetDesc().c_str(),
           src_endpoint_idx);
    (void)reusable_queue_indices_.emplace(src_endpoint_idx);
  }
  return SUCCESS;
}

bool DeployPlannerBase::IsOutputMultiConnected(const int32_t src_endpoint_idx) {
  const auto &it = endpoint_pairs_.find(src_endpoint_idx);
  if (it == endpoint_pairs_.cend()) {
    return false;
  }

  const auto &src_endpoint_info = deploy_plan_.queues_[src_endpoint_idx];
  if (it->second.size() != 1U) {
    GELOGD("Queue[%s@%s] has one-to-many relation to models", src_endpoint_info.name.c_str(),
           src_endpoint_info.device_info.GetDesc().c_str());
    return true;
  }

  const auto &dst_queue_infos = *(it->second.begin());
  if (dst_queue_infos.second.size() != 1U) {
    GELOGD("Queue[%s@%s] has multi-device dest endpoints", src_endpoint_info.name.c_str(),
           src_endpoint_info.device_info.GetDesc().c_str());
    return true;
  }

  return false;
}

bool DeployPlannerBase::IsInputMultiConnected(const int32_t dst_endpoint_idx) {
  const auto &dst_endpoint_info = deploy_plan_.queues_[dst_endpoint_idx];
  const auto &it = relation_dst_to_src_.find(dst_endpoint_info.name);
  return (it != relation_dst_to_src_.cend()) && (it->second.size() > 1U);
}

Status DeployPlannerBase::AssignDequeueQueues() {
  // key order [model_instance_name][invoke_key][index]
  std::map<std::string, std::map<std::string, std::map<int32_t, int32_t>>> model_input_indices;
  std::map<std::string, int32_t> external_input_indices;
  std::map<std::string, std::set<int32_t>> model_control_input_indices;
  for (const auto &endpoint_pair : endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    // group by model_and_input_idx
    for (const auto &queue_loc_and_queue_infos : endpoint_pair.second) {
      const auto &model_queue_loc = queue_loc_and_queue_infos.first;
      for (size_t i = 0; i < queue_loc_and_queue_infos.second.size(); ++i) {
        const auto &queue_info = queue_loc_and_queue_infos.second[i];
        int32_t dst_endpoint_idx = -1;
        const auto &model_instance_name = queue_info.model_instance_name;
        if (reusable_queue_indices_.count(src_endpoint_idx) > 0UL) {
          GELOGD("Reuse src queue, queue name = %s, queue index = %d",
                 deploy_plan_.queues_[src_endpoint_idx].name.c_str(), src_endpoint_idx);
          dst_endpoint_idx = src_endpoint_idx;
        } else {
          GE_CHK_STATUS_RET_NOLOG(GetOrCreateInputEndpoint(model_queue_loc, queue_info, dst_endpoint_idx));
          GE_CHK_STATUS_RET_NOLOG(ResolveDequeueFusion(src_endpoint_idx, dst_endpoint_idx));
          // aggregation endpoint need to known current intance index
          InputGroupAttr input_group_attr = {};
          input_group_attr.instance_num = static_cast<int32_t>(queue_loc_and_queue_infos.second.size());
          input_group_attr.instance_idx = static_cast<int32_t>(i);
          GE_CHK_STATUS_RET(PrepareRelations(src_endpoint_idx,
                                             dst_endpoint_idx,
                                             model_queue_loc,
                                             queue_info,
                                             input_group_attr),
                            "Failed to prepare relations");
          GELOGI("Prepare relation success, src = %s, index = %d, dst = %s, index = %d",
                 ToEndpointDesc(src_endpoint_idx).c_str(), src_endpoint_idx,
                 ToEndpointDesc(dst_endpoint_idx).c_str(), dst_endpoint_idx);
        }
        deploy_plan_.src_to_dst_endpoints_[src_endpoint_idx][model_queue_loc].emplace_back(dst_endpoint_idx);
        if (queue_info.queue_action == DeployPlan::QueueAction::kControl) {
          model_control_input_indices[model_instance_name].emplace(dst_endpoint_idx);
        } else if (model_queue_loc.id >= 0) {
          model_input_indices[model_instance_name][model_queue_loc.invoke_key][model_queue_loc.id] = dst_endpoint_idx;
        } else {
          external_input_indices[model_instance_name] = dst_endpoint_idx;
        }
      }
    }
  }
  for (const auto &name_and_input_indices : model_control_input_indices) {
    auto &submodel_info = MutableSubmodelInfo(name_and_input_indices.first);
    submodel_info.control_input_queue_indices.assign(name_and_input_indices.second.cbegin(),
                                                     name_and_input_indices.second.cend());
  }
  for (const auto &name_and_input_indices : model_input_indices) {
    auto &submodel_info = MutableSubmodelInfo(name_and_input_indices.first);
    // group by invoke key
    for (const auto &group_indices : name_and_input_indices.second) {
      const auto &invoke_key = group_indices.first;
      for (const auto &input_index_and_endpoint_index : group_indices.second) {
        if (!invoke_key.empty()) {
          // invoke model's fetch queue as input
          submodel_info.invoked_model_queue_infos[invoke_key].fetch_queue_indices.emplace_back(
              input_index_and_endpoint_index.second);
          continue;
        }
        // already sorted by input index, it's OK to use emplace_back
        submodel_info.input_queue_indices.emplace_back(input_index_and_endpoint_index.second);
      }
    }
  }
  for (const auto &name_and_input_index : external_input_indices) {
    auto &submodel_info = MutableSubmodelInfo(name_and_input_index.first);
    submodel_info.input_queue_indices.emplace_back(name_and_input_index.second);
  }
  return SUCCESS;
}

bool DeployPlannerBase::IsMultiDeployed(const std::string &model_instance_name) const {
  const auto &ins_it = instance_to_model_name_.find(model_instance_name);
  if (ins_it == instance_to_model_name_.cend()) {
    return false;
  }
  const auto &model_name = ins_it->second;
  const auto &location_it = model_deploy_locations_.find(model_name);
  return (location_it != model_deploy_locations_.cend()) && (location_it->second.size() > 1U);
}

void DeployPlannerBase::AddInputGroups(const int32_t dst_endpoint_idx,
                                       const int32_t src_tag_idx,
                                       const InputGroupAttr &input_group_attr) {
  auto &input_group = input_groups_[dst_endpoint_idx];
  auto it = std::find(input_group.begin(), input_group.end(), src_tag_idx);
  if (it == input_group.end()) {
    input_groups_[dst_endpoint_idx].emplace_back(src_tag_idx);
    input_groups_attr_[dst_endpoint_idx] = input_group_attr;
  }
}

bool DeployPlannerBase::CheckAndAddRelation(const int32_t src_endpoint_idx,
                                            const int32_t dst_endpoint_idx,
                                            const std::string &suffix) {
  const std::string relation_key = std::to_string(src_endpoint_idx) + "_to_" +
                                   std::to_string(dst_endpoint_idx) + suffix;
  const bool relation_added = relations_.find(relation_key) != relations_.end();
  if (!relation_added) {
    (void) relations_.emplace(relation_key);
  }
  GELOGD("Check and add relaton[%s] success, relation added = %d.",
         relation_key.c_str(), static_cast<int32_t>(relation_added));
  return relation_added;
}

void DeployPlannerBase::GenTagEntityPair(int32_t endpoint_idx,
                                         const DeployPlan::QueueInfo &mapping_queue_info,
                                         std::pair<DeployPlan::QueueInfo, DeployPlan::QueueInfo> &entity_pair) {
  auto tag_name = mapping_queue_info.name + "_" + std::to_string(endpoint_idx);
  const auto &queue_info = deploy_plan_.queues_[endpoint_idx];
  entity_pair.first = queue_info;
  entity_pair.first.name = tag_name;

  entity_pair.second = mapping_queue_info;
  entity_pair.second.name = tag_name;
}

Status DeployPlannerBase::GetOrCreateMappingTagPairEntry(const int32_t endpoint_idx,
                                                         const DeployPlan::QueueInfo &mapping_queue_info,
                                                         std::pair<int32_t, int32_t> &tag_pair,
                                                         bool use_balanced) {
  const auto &queue_info = deploy_plan_.queues_[endpoint_idx];
  auto src_is_multi_deployed = IsMultiDeployed(queue_info.model_instance_name);
  auto dst_is_multi_deployed = IsMultiDeployed(mapping_queue_info.model_instance_name);

  auto device_key = mapping_queue_info.device_info.GetKey();
  auto mapping_key = std::make_pair(endpoint_idx, device_key);
  auto get_from_cache = (!src_is_multi_deployed && !dst_is_multi_deployed) || (!use_balanced);
  if (get_from_cache) {
    const auto &it = endpoint_device_tags_mapping_.find(mapping_key);
    if (it != endpoint_device_tags_mapping_.cend()) {
      tag_pair =  it->second;
      return SUCCESS;
    }
  }

  std::pair<DeployPlan::QueueInfo, DeployPlan::QueueInfo> entity_pair;
  GenTagEntityPair(endpoint_idx, mapping_queue_info, entity_pair);
  GE_CHK_STATUS_RET(CreateGroupEntry(entity_pair.first, tag_pair.first), "Failed to create group entity.");
  GE_CHK_STATUS_RET(CreateGroupEntry(entity_pair.second, tag_pair.second), "Failed to create group entity.");
  if (get_from_cache) {
    endpoint_device_tags_mapping_[mapping_key] = tag_pair;
  }
  GELOGI("Endpoint[%d] add mapping tag pair[%d,%d] success, mapping device = %s",
         endpoint_idx, tag_pair.first, tag_pair.second, mapping_queue_info.device_info.GetDesc().c_str());
  return SUCCESS;
}

Status DeployPlannerBase::GetOrCreateMappingEntry(const int32_t endpoint_idx,
                                                  const DeployPlan::QueueInfo &mapping_queue_info,
                                                  int32_t &mapping_idx) {
  const auto &queue_info = deploy_plan_.queues_[endpoint_idx];
  auto src_is_multi_deployed = IsMultiDeployed(queue_info.model_instance_name);
  auto dst_is_multi_deployed = IsMultiDeployed(mapping_queue_info.model_instance_name);
  auto device_key = mapping_queue_info.device_info.GetKey();
  auto mapping_key = std::make_pair(endpoint_idx, device_key);
  auto get_from_cache = (!src_is_multi_deployed && !dst_is_multi_deployed);
  if (get_from_cache) {
    const auto &it = endpoint_device_mapping_.find(mapping_key);
    if (it != endpoint_device_mapping_.cend()) {
      mapping_idx =  it->second;
      return SUCCESS;
    }
  }

  auto entry_info = queue_info;
  entry_info.device_info = mapping_queue_info.device_info;
  int32_t queue_index = 0;
  auto mapping_desc = mapping_queue_info.device_info.GetDesc();
  GE_CHK_STATUS_RET(CreateGroupQueueEntry(entry_info, queue_index, mapping_idx),
                    "Failed to create group entity.");
  if (get_from_cache) {
    endpoint_device_mapping_[mapping_key] = mapping_idx;
  }
  GELOGI("Endpoint[%d] add mapping endpoint[%d] success, mapping device = %s",
         endpoint_idx, queue_index, mapping_desc.c_str());
  return SUCCESS;
}

Status DeployPlannerBase::PrepareDiffNodeRelation(const int32_t src_endpoint_idx,
                                                  const int32_t dst_endpoint_idx,
                                                  const ModelQueueIndex &model_queue_loc,
                                                  const DeployPlan::QueueInfo &queue_info,
                                                  const InputGroupAttr &input_group_attr) {
  auto deploy_location_key = queue_info.device_info.GetKey() + "_" + std::to_string(queue_info.process_id);
  // create endpoint may cause vector of deploy_plan_.queues_ memory expansion.
  // As a result, the referenced endpoint becomes invalid, use copy
  const auto src_queue_info = deploy_plan_.queues_[src_endpoint_idx];
  auto src_queue_index = src_endpoint_idx;
  auto dst_queue_index = dst_endpoint_idx;
  if (src_queue_info.device_info.WithProxy()) {
    auto proxy_queue_info = queue_info;
    proxy_queue_info.device_info = src_queue_info.device_info.ProxyDevice();
    int32_t mapping_index = 0;
    GE_CHK_STATUS_RET(GetOrCreateMappingEntry(src_endpoint_idx, proxy_queue_info, mapping_index),
                      "Failed to create mapping entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, mapping_index);
    src_queue_index = deploy_plan_.group_entries_[mapping_index].ref_index;
    GELOGI("Prepare relation, src index = %d, src name = %s, mapping index = %d, "
           "model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), mapping_index,
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[mapping_index] = dst_queue_index;
    GELOGI("DynamicSched, Step1, add group entry index=%d, dest endpoint idx=%d.", mapping_index, dst_queue_index);
  } else if (!src_queue_info.owned) {
    // for external queue, additional binding need to be added to block data until model loaded.
    int32_t mapping_index = 0;
    auto inner_queue_info = queue_info;
    inner_queue_info.device_info = src_queue_info.device_info;
    GE_CHK_STATUS_RET(GetOrCreateMappingEntry(src_endpoint_idx, inner_queue_info, mapping_index),
                      "Failed to create mapping entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, mapping_index);
    src_queue_index = deploy_plan_.group_entries_[mapping_index].ref_index;
    deploy_plan_.queues_[src_queue_index].owned = true;
    GELOGI("Prepare relation, src index = %d, src name = %s, mapping index = %d, "
           "model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), mapping_index,
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[mapping_index] = dst_queue_index;
  }

  auto dst_queue_info = queue_info;
  if (queue_info.device_info.WithProxy()) {
    dst_queue_info.device_info = queue_info.device_info.ProxyDevice();
  }

  std::pair<int32_t, int32_t> tag_pair;
  GE_CHK_STATUS_RET(GetOrCreateMappingTagPairEntry(src_queue_index, dst_queue_info, tag_pair),
                    "Failed to create mapping tag pair entity.");
  (void)output_groups_[src_queue_index][model_queue_loc].emplace(deploy_location_key, tag_pair.second);
  GELOGI("Prepare relation, src index = %d, src name = %s, entity index = %d, model name = %s, input index = %d",
         src_queue_index, ToEndpointDesc(src_queue_index).c_str(), tag_pair.second,
         model_queue_loc.model_name.c_str(), model_queue_loc.id);
  deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[tag_pair.second] = dst_queue_index;
  GELOGI("DynamicSched, Step1, add group entry index=%d, dest endpoint idx=%d.", tag_pair.second, dst_queue_index);
  if (queue_info.device_info.WithProxy()) {
    int32_t mapping_index = 0;
    GE_CHK_STATUS_RET(GetOrCreateMappingEntry(src_queue_index, dst_queue_info, mapping_index),
                      "Failed to create mapping entity.");
    dst_queue_index = deploy_plan_.group_entries_[mapping_index].ref_index;
    AddInputGroups(dst_endpoint_idx, mapping_index, input_group_attr);
    GELOGI("Prepare relation, mapping index = %d, dst index = %d, dst name = %s",
           mapping_index, dst_endpoint_idx, ToEndpointDesc(dst_endpoint_idx).c_str());
  }
  AddInputGroups(dst_queue_index, tag_pair.first, input_group_attr);
  GELOGI("Prepare relation, src tag index = %d, dst index = %d, dst name = %s",
         tag_pair.first, dst_queue_index, ToEndpointDesc(dst_queue_index).c_str());
  return SUCCESS;
}

Status DeployPlannerBase::PrepareSameNodeRelation(const int32_t src_endpoint_idx,
                                                  const int32_t dst_endpoint_idx,
                                                  const ModelQueueIndex &model_queue_loc,
                                                  const DeployPlan::QueueInfo &queue_info,
                                                  const InputGroupAttr &input_group_attr) {
  auto deploy_location_key = queue_info.device_info.GetKey() + "_" + std::to_string(queue_info.process_id);
  // create endpoint may cause vector of deploy_plan_.queues_ memory expansion.
  // As a result, the referenced endpoint becomes invalid, use copy
  const auto src_queue_info = deploy_plan_.queues_[src_endpoint_idx];
   // Queue -> Queue
  if (CanConnectWithLocalQ(src_queue_info.device_info, queue_info.device_info)) {
    GE_CHK_STATUS_RET(PrepareQueuesRelation(src_endpoint_idx,
                                            dst_endpoint_idx,
                                            model_queue_loc,
                                            queue_info,
                                            input_group_attr),
                      "Failed to prepare relation of queues");
  } else if (src_queue_info.device_info.WithProxy()) {
    int32_t mapping_index = 0;
    GE_CHK_STATUS_RET(GetOrCreateMappingEntry(src_endpoint_idx, queue_info, mapping_index),
                      "Failed to create mapping entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, mapping_index);
    AddInputGroups(dst_endpoint_idx, mapping_index, input_group_attr);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[mapping_index] = dst_endpoint_idx;
    GELOGI("Prepare relation, src index = %d, src name = %s, mapping index = %d, dst index = %d, dst name = %s, "
           "model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), mapping_index,
           dst_endpoint_idx, ToEndpointDesc(dst_endpoint_idx).c_str(),
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
  } else if (queue_info.device_info.WithProxy()) {
    // src in device, route must sched by dev flowgw,
    // because one to multi relationships need to be scheduled by the same flowgw
    int32_t mapping_index = 0;
    GE_CHK_STATUS_RET(GetOrCreateMappingEntry(dst_endpoint_idx, src_queue_info, mapping_index),
                      "Failed to create mapping entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, mapping_index);
    AddInputGroups(dst_endpoint_idx, mapping_index, input_group_attr);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[mapping_index] = dst_endpoint_idx;
    GELOGI("Prepare relation, src index = %d, src name = %s, mapping index = %d, dst index = %d, dst name = %s, "
           "model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), mapping_index,
           dst_endpoint_idx, ToEndpointDesc(dst_endpoint_idx).c_str(),
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
  } else {
    std::pair<int32_t, int32_t> tag_pair;
    GE_CHK_STATUS_RET(GetOrCreateMappingTagPairEntry(src_endpoint_idx, queue_info, tag_pair),
                      "Failed to create mapping tag pair entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, tag_pair.second);
    AddInputGroups(dst_endpoint_idx, tag_pair.first, input_group_attr);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[tag_pair.second] = dst_endpoint_idx;
    GELOGI("Prepare relation, src index = %d, src name = %s, src tag index = %d, "
           "dst index = %d, dst name = %s, dst tag index = %d, "
           "model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), tag_pair.first,
           dst_endpoint_idx, ToEndpointDesc(dst_endpoint_idx).c_str(), tag_pair.second,
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
  }
  return SUCCESS;
}

Status DeployPlannerBase::PrepareQueuesRelation(const int32_t src_endpoint_idx,
                                                const int32_t dst_endpoint_idx,
                                                const ModelQueueIndex &model_queue_loc,
                                                const DeployPlan::QueueInfo &queue_info,
                                                const InputGroupAttr &input_group_attr) {
  auto deploy_location_key = queue_info.device_info.GetKey() + "_" + std::to_string(queue_info.process_id);
  auto src_is_multi_connected = IsOutputMultiConnected(src_endpoint_idx);
  auto dst_is_multi_connected = IsInputMultiConnected(dst_endpoint_idx);
  if (src_is_multi_connected && dst_is_multi_connected) {
    int32_t mapping_index = 0;
    GE_CHK_STATUS_RET(GetOrCreateMappingEntry(src_endpoint_idx, queue_info, mapping_index),
                      "Failed to create mapping entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, mapping_index);
    AddInputGroups(dst_endpoint_idx, mapping_index, input_group_attr);
    GELOGI("Prepare relation, src index = %d, src name = %s, mapping index = %d, dst index = %d, dst name = %s, "
           "model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), mapping_index,
           dst_endpoint_idx, ToEndpointDesc(dst_endpoint_idx).c_str(),
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[mapping_index] = dst_endpoint_idx;
    GELOGI("DynamicSched, Step1, add group entry index=%d, dest endpoint idx=%d.", mapping_index, dst_endpoint_idx);
  } else if (dst_is_multi_connected) {
    int32_t entry_index = 0;
    GE_CHK_STATUS_RET(CreateGroupRefEntry(queue_info, src_endpoint_idx, entry_index),
                      "Failed to create group entity.");
    AddInputGroups(dst_endpoint_idx, entry_index, input_group_attr);
    GELOGI("Prepare relation, ref entity index = %d, dst index = %d, dst name = %s",
           entry_index, dst_endpoint_idx, ToEndpointDesc(dst_endpoint_idx).c_str());
  } else {
    int32_t entry_index = 0;
    GE_CHK_STATUS_RET(CreateGroupRefEntry(queue_info, dst_endpoint_idx, entry_index),
                      "Failed to create group entity.");
    (void)output_groups_[src_endpoint_idx][model_queue_loc].emplace(deploy_location_key, entry_index);
    GELOGI("Prepare relation, src index = %d, src name = %s, ref entity index = %d, model name = %s, input index = %d",
           src_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(), entry_index,
           model_queue_loc.model_name.c_str(), model_queue_loc.id);
    deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_[entry_index] = dst_endpoint_idx;
    GELOGI("DynamicSched, Step1, add group entry index=%d, dest endpoint idx=%d.", entry_index, dst_endpoint_idx);
  }
  return SUCCESS;
}

Status DeployPlannerBase::PrepareRelations(const int32_t src_endpoint_idx,
                                           const int32_t dst_endpoint_idx,
                                           const ModelQueueIndex &model_queue_loc,
                                           const DeployPlan::QueueInfo &queue_info,
                                           const InputGroupAttr &input_group_attr) {
  UpdateFusionOffset(src_endpoint_idx, dst_endpoint_idx);
  const int32_t src_ref_idx = deploy_plan_.queues_[src_endpoint_idx].ref_index;
  const int32_t dst_ref_idx = deploy_plan_.queues_[dst_endpoint_idx].ref_index;
  const int32_t src_queue_idx = src_ref_idx >= 0 ? src_ref_idx : src_endpoint_idx;
  const int32_t dst_queue_idx = dst_ref_idx >= 0 ? dst_ref_idx : dst_endpoint_idx;
  if (CheckAndAddRelation(src_queue_idx, dst_queue_idx)) {
    return SUCCESS;
  }
  const auto &src_device_info = deploy_plan_.queues_[src_queue_idx].device_info;
  if (src_device_info.GetNodeId() == queue_info.device_info.GetNodeId()) {
    GELOGI("Begin to prepare relation of same node, node_id = %d", src_device_info.GetNodeId());
    GE_CHK_STATUS_RET(PrepareSameNodeRelation(src_queue_idx,
                                              dst_queue_idx,
                                              model_queue_loc,
                                              queue_info,
                                              input_group_attr),
                      "Failed to prepare relation of same node");
  } else {
    GELOGI("Begin to prepare relation of diff node, src node_id = %d, dst node_id = %d",
           src_device_info.GetNodeId(), queue_info.device_info.GetNodeId());
    GE_CHK_STATUS_RET(PrepareDiffNodeRelation(src_queue_idx,
                                              dst_queue_idx,
                                              model_queue_loc,
                                              queue_info,
                                              input_group_attr),
                      "Failed to prepare relation of diff node");
  }
  GELOGI("%s add input queue %s", ToEndpointDesc(dst_queue_idx).c_str(), ToEndpointDesc(src_queue_idx).c_str());
  return SUCCESS;
}

void DeployPlannerBase::UpdateFusionOffset(int32_t src_index, int32_t dst_index) {
  auto it = deploy_plan_.groups_.find(dst_index);
  if (it != deploy_plan_.groups_.end()) {
    const auto &group_elements = it->second;
    for (auto index : group_elements) {
      deploy_plan_.queues_[index].fusion_offset = deploy_plan_.queues_[src_index].fusion_offset;
      GELOGI("Update queue[%d] fusion offset, offset = %d", index, deploy_plan_.queues_[index].fusion_offset);
    }
  }
  deploy_plan_.queues_[dst_index].fusion_offset = deploy_plan_.queues_[src_index].fusion_offset;
  GELOGI("Update queue[%d] fusion offset, offset = %d", dst_index, deploy_plan_.queues_[dst_index].fusion_offset);
}

void DeployPlannerBase::AddEndpointBindings(int32_t src_index, int32_t dst_index, bool skip_if_dst_exists) {
  GELOGI("Begin to add bind relation[%d -> %d].", src_index, dst_index);
  auto real_index = dst_index;
  if (deploy_plan_.queues_[dst_index].ref_index >= 0) {
    real_index = deploy_plan_.queues_[dst_index].ref_index;
    GELOGI("Dst[%d] is ref endpoint, ref index = %d.", dst_index, real_index);
  }

  if (skip_if_dst_exists) {
    auto it = deploy_plan_.dst_to_src_bindings_.find(real_index);
    if (it != deploy_plan_.dst_to_src_bindings_.end()) {
      GELOGI("Bind relation[%d -> %d] has been added already.", it->second, real_index);
      return;
    }
    deploy_plan_.dst_to_src_bindings_[dst_index] = src_index;
  }

  deploy_plan_.queue_bindings_.emplace_back(src_index, dst_index);
  GELOGI("Add bind relation[%d -> %d] success.", src_index, real_index);
}

Status DeployPlannerBase::BindRemoteOutputGroupToInput() {
  for (const auto &it : input_groups_) {
    const auto endpoint_index = it.first;
    // create endpoint may cause vector of deploy_plan_.queues_ memory expansion.
    // As a result, the referenced endpoint becomes invalid, use copy
    const auto input_endpoint_info = deploy_plan_.queues_[endpoint_index];
    const auto &group_attr = input_groups_attr_[endpoint_index];
    DeployPlan::QueueInfo group_info{};
    group_info.device_info = input_endpoint_info.device_info;
    group_info.model_instance_name = input_endpoint_info.model_instance_name;
    group_info.instance_num = group_attr.instance_num;
    group_info.instance_idx = group_attr.instance_idx;

    // Handle entries that need to bind separately
    std::vector<int32_t> group_entry_indices;
    for (const auto group_entry_index : it.second) {
      const auto &entry_info = deploy_plan_.group_entries_[group_entry_index];
      if (no_group_endpoint_names_.find(entry_info.name) != no_group_endpoint_names_.cend()) {
        group_info.name = input_endpoint_info.name + "_" + std::to_string(group_entry_index);
        GE_CHK_STATUS_RET_NOLOG(CreateAndBindGroup(group_info, {group_entry_index}, endpoint_index, false));
      } else {
        group_entry_indices.emplace_back(group_entry_index);
      }
    }
    // Handle remaining entries
    if (!group_entry_indices.empty()) {
      group_info.name = input_endpoint_info.name;
      GE_CHK_STATUS_RET_NOLOG(CreateAndBindGroup(group_info, group_entry_indices, endpoint_index));
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::CreateAndBindGroup(const DeployPlan::QueueInfo &group_info,
                                             const std::vector<int32_t> &group_entry_index,
                                             const int32_t dst_endpoint_index,
                                             const bool skip_if_dst_exists) {
  int32_t group_index = -1;
  GE_CHK_STATUS_RET_NOLOG(CreateGroupInfo(group_info, group_entry_index, group_index));
  AddEndpointBindings(group_index, dst_endpoint_index, skip_if_dst_exists);
  GELOGD("Input group binding added, peer = %s, local = %s@%s",
         ToString(ToEndpointDescs(deploy_plan_.groups_[group_index], true)).c_str(), group_info.name.c_str(),
         deploy_plan_.queues_[dst_endpoint_index].device_info.GetDesc().c_str());
  return SUCCESS;
}

Status DeployPlannerBase::BindOutputToRemoteInputs() {
  for (auto &it : output_groups_) {
    const auto endpoint_index = it.first;
    for (auto &grouped_peer_inputs : it.second) {
      const auto &model_queue_loc = grouped_peer_inputs.first;
      const auto &grouped_inputs = grouped_peer_inputs.second;
      DeployPlan::QueueInfo group_info{};
      // create endpoint may cause vector of deploy_plan_.queues_ memory expansion.
      // As a result, the referenced endpoint becomes invalid, use copy
      const auto output_endpoint_info = deploy_plan_.queues_[endpoint_index];
      group_info.name = output_endpoint_info.name;
      group_info.device_info = output_endpoint_info.device_info;
      group_info.model_instance_name = output_endpoint_info.model_instance_name;
      int32_t group_index = -1;
      std::vector<int32_t> grouped_inputs_order_by_device;
      for (const auto &device_and_index : grouped_inputs) {
        grouped_inputs_order_by_device.emplace_back(device_and_index.second);
      }
      GE_CHK_STATUS_RET(CreateGroupInfo(group_info, grouped_inputs_order_by_device, group_index));
      AddEndpointBindings(endpoint_index, group_index);
      GELOGD("Output group binding added, local = %s@%s, peer model = %s:%d, peer input indices = %s.",
             group_info.name.c_str(), deploy_plan_.queues_[endpoint_index].device_info.GetDesc().c_str(),
             model_queue_loc.model_name.c_str(), model_queue_loc.id,
             ToString(ToEndpointDescs(deploy_plan_.groups_[group_index], true)).c_str());
    }
  }
  return SUCCESS;
}

void DeployPlannerBase::UpdateDeployPlan() {
  deploy_plan_.root_model_info_.input_queue_indices = std::move(head_model_info_.output_queue_indices);
  deploy_plan_.root_model_info_.control_input_queue_indices = std::move(head_model_info_.control_output_queue_indices);
  deploy_plan_.root_model_info_.output_queue_indices = std::move(tail_model_info_.input_queue_indices);
  deploy_plan_.root_model_info_.control_output_queue_indices = std::move(tail_model_info_.control_input_queue_indices);
  deploy_plan_.root_model_info_.queue_device_info = std::move(head_model_info_.queue_device_info);
}

DeployPlan::SubmodelInfo &DeployPlannerBase::MutableSubmodelInfo(const std::string &name) {
  if (name == head_model_queue_info_.model_name) {
    return head_model_info_;
  } else if (name == tail_model_queue_info_.model_name) {
    return tail_model_info_;
  } else {
    return deploy_plan_.submodels_[name];
  }
}

bool DeployPlannerBase::IsHeadOrTail(const std::string &name) const {
  return (name == head_model_queue_info_.model_name) || (name == tail_model_queue_info_.model_name);
}

const std::string &DeployPlannerBase::GetSubmodelType(const std::string &name) {
  if (name == head_model_queue_info_.model_name || name == tail_model_queue_info_.model_name) {
    return PNE_ID_CPU;
  }
  if (deploy_plan_.submodels_[name].model == nullptr) {
    return PNE_ID_CPU;
  }
  return deploy_plan_.submodels_[name].model->GetModelType();
}

std::string DeployPlannerBase::ToEndpointDesc(const int32_t endpoint_indices, const bool is_group_entry) const {
  const auto &endpoint_info =
      is_group_entry ? deploy_plan_.group_entries_[endpoint_indices] : deploy_plan_.queues_[endpoint_indices];
  auto desc = endpoint_info.name;
  desc += ("@" + endpoint_info.device_info.GetDesc());
  return desc;
}

std::vector<std::string> DeployPlannerBase::ToEndpointDescs(const std::vector<int32_t> &endpoint_indices,
                                                            const bool is_group_entry) const {
  std::vector<std::string> ret;
  (void)std::transform(endpoint_indices.cbegin(), endpoint_indices.cend(), std::back_inserter(ret),
                       [this, is_group_entry](const int32_t index) { return ToEndpointDesc(index, is_group_entry); });
  return ret;
}

DeployPlan::QueueInfo DeployPlannerBase::BuildQueueInfo(const Endpoint &queue_def,
                                                        const std::string &model_instance_name) {
  DeployPlan::QueueInfo queue_info{};
  const auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  queue_info.device_info = submodel_info.queue_device_info;
  queue_info.model_instance_name = model_instance_name;
  const auto &name = queue_def.GetName();
  queue_info.name = name.length() <= kMaxQueueNameLen ? name : GenShortName(name);
  queue_info.process_id = submodel_info.process_id;

  // 需要安全校验
  queue_info.depth = QueueNodeUtils::GetDepth(queue_def);
  queue_info.enqueue_policy = QueueNodeUtils::GetEnqueuePolicy(queue_def);
  bool use_proxy_device = false;
  if (QueueNodeUtils::GetIsControl(queue_def)) {
    queue_info.queue_action = DeployPlan::QueueAction::kControl;
  } else if (QueueNodeUtils::GetIsStatus(queue_def)) {
    // all status queue use device queue
    use_proxy_device = true;
    queue_info.queue_action = DeployPlan::QueueAction::kStatus;
  } else if (QueueNodeUtils::GetIsSched(queue_def)) {
    queue_info.queue_action = DeployPlan::QueueAction::kSched;
  } else {
    queue_info.queue_action = DeployPlan::QueueAction::kDefault;
  }
  use_proxy_device = use_proxy_device || IsNeedDeviceQueue(submodel_info);
  if (use_proxy_device && queue_info.device_info.WithProxy()) {
    queue_info.device_info = queue_info.device_info.ProxyDevice();
  }

  queue_info.is_dummy = (queue_def.GetEndpointType() == EndpointType::kDummyQueue);
  queue_info.model_id = deploy_plan_.dynamic_sched_plan_.submodels_id_[model_instance_name];
  GELOGI("[%s] queue depth = [%u], policy = [%s], model_id=[%u], queue device info=%s, use_proxy_device=%d.",
         queue_def.GetName().c_str(), queue_info.depth, queue_info.enqueue_policy.c_str(), queue_info.model_id,
         queue_info.device_info.GetDesc().c_str(), static_cast<int32_t>(use_proxy_device));
  return queue_info;
}

Status DeployPlannerBase::CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info, int32_t &queue_idx) {
  const auto queue_size = deploy_plan_.queues_.size();
  GE_CHK_STATUS_RET(CreateEndpointInfo(queue_info), "Create endpoint info failed.");
  queue_idx = static_cast<int32_t>(queue_size);
  GELOGI("Create endpoint success, index = %d", queue_idx);
  return SUCCESS;
}

Status DeployPlannerBase::CreateEndpointInfo(const DeployPlan::QueueInfo &queue_info) {
  const auto queue_size = deploy_plan_.queues_.size();
  GE_CHECK_LE(queue_size, static_cast<size_t>(INT32_MAX));
  deploy_plan_.queues_.emplace_back(queue_info);
  return SUCCESS;
}

Status DeployPlannerBase::CreateGroupQueueEntry(const DeployPlan::QueueInfo &queue_info,
                                                int32_t &queue_index,
                                                int32_t &entry_index) {
  GE_CHK_STATUS_RET(CreateEndpointInfo(queue_info, queue_index), "Failed to create endpoint.");
  auto entry_info = queue_info;
  entry_info.ref_index = queue_index;
  GE_CHK_STATUS_RET(CreateGroupEntry(entry_info, entry_index), "Failed to create group entity.");
  return SUCCESS;
}

Status DeployPlannerBase::CreateGroupRefEntry(const DeployPlan::QueueInfo &queue_info,
                                              int32_t endpoint_index,
                                              int32_t &entry_index) {
  auto entry_info = queue_info;
  entry_info.ref_index = endpoint_index;
  GE_CHK_STATUS_RET(CreateGroupEntry(entry_info, entry_index), "Failed to create group entity.");
  return SUCCESS;
}

Status DeployPlannerBase::CreateGroupEntry(const DeployPlan::QueueInfo &queue_info, int32_t &entry_index) {
  const auto entry_size = deploy_plan_.group_entries_.size();
  GE_CHECK_LE(entry_size, static_cast<size_t>(INT32_MAX));
  deploy_plan_.group_entries_.emplace_back(queue_info);
  entry_index = static_cast<int32_t>(entry_size);
  return SUCCESS;
}

Status DeployPlannerBase::CreateGroupInfo(const DeployPlan::QueueInfo &queue_info,
                                          const std::vector<int32_t> &grouped_indices,
                                          int32_t &group_index) {
  // check whether group already exist
  auto group_key = ToString(grouped_indices);
  auto it = deploy_plan_.groups_key_to_idx_.find(group_key);
  if (it != deploy_plan_.groups_key_to_idx_.end()) {
    group_index = it->second;
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(CreateEndpointInfo(queue_info, group_index));
  deploy_plan_.groups_[group_index] = grouped_indices;
  deploy_plan_.groups_key_to_idx_[group_key] = group_index;
  GELOGD("Group created, name = %s, group_index = %d, endpoint_indices = %s, endpoint_descs = %s",
         queue_info.name.c_str(), group_index, ToString(grouped_indices).c_str(),
         ToString(ToEndpointDescs(grouped_indices, true)).c_str());
  return SUCCESS;
}

Status DeployPlannerBase::ResolveDequeueFusion(int32_t src_endpoint_idx, int32_t dst_endpoint_idx) {
  GELOGD("Begin to resolve dequeue fusion, src index = %d, dst index %d.", src_endpoint_idx, dst_endpoint_idx);
  const std::set<std::string> kSupportFusionEngines = {PNE_ID_NPU};
  auto &endpoint = deploy_plan_.queues_[static_cast<size_t>(dst_endpoint_idx)];
  const auto &dst_model_instance_name = endpoint.model_instance_name;
  const auto &model_type = GetSubmodelType(dst_model_instance_name);
  if (kSupportFusionEngines.find(model_type) == kSupportFusionEngines.end()) {
    GELOGI("Dequeue fusion is unsupported for engine type[%s].", model_type.c_str());
    return SUCCESS;
  }

  auto real_src_idx = src_endpoint_idx;
  if (deploy_plan_.queues_[src_endpoint_idx].ref_index >= 0) {
    real_src_idx = deploy_plan_.queues_[src_endpoint_idx].ref_index;
    GELOGI("Src[%d] is ref endpoint, ref index = %d.", src_endpoint_idx, real_src_idx);
  }
  auto src_it = dequeue_ref_indices_.find(real_src_idx);
  if (src_it == dequeue_ref_indices_.end()) {
    GELOGI("Relation added, dst model_name = %s, src index = %d, dst index = %d.", dst_model_instance_name.c_str(),
           real_src_idx, dst_endpoint_idx);
    dequeue_ref_indices_[real_src_idx][dst_model_instance_name] = dst_endpoint_idx;
    return SUCCESS;
  }
  const auto &dst_info = src_it->second;
  auto dst_info_it = dst_info.find(dst_model_instance_name);
  if (dst_info_it == dst_info.end()) {
    GELOGI("Relation added, dst model_name = %s, src index = %d, dst index = %d.", dst_model_instance_name.c_str(),
           real_src_idx, dst_endpoint_idx);
    dequeue_ref_indices_[real_src_idx][dst_model_instance_name] = dst_endpoint_idx;
    return SUCCESS;
  }
  endpoint.ref_index = dst_info_it->second;
  GELOGI("Fusion dequeue enpoint[%d] success, ref index = %d.", dst_endpoint_idx, endpoint.ref_index);
  return SUCCESS;
}

Status DeployPlannerBase::GetOrCreateInputEndpoint(const ModelQueueIndex &model_queue_index,
                                                   const DeployPlan::QueueInfo &queue_info, int32_t &endpoint_index) {
  auto key = std::make_tuple(model_queue_index, queue_info.device_info.GetKey(), queue_info.process_id);
  const auto &it = input_endpoint_indices_.find(key);
  if (it != input_endpoint_indices_.cend()) {
    endpoint_index = it->second;
    return SUCCESS;
  }

  GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
  GELOGD("Input endpoint created, queue name = %s, device = %s, index = %d", queue_info.name.c_str(),
         queue_info.device_info.GetDesc().c_str(), endpoint_index);
  input_endpoint_indices_[key] = endpoint_index;
  return SUCCESS;
}

Status DeployPlannerBase::CreateOutputQueueDefs(const std::string &model_instance_name,
                                                const std::vector<std::string> &queue_names, const bool is_owned) {
  std::vector<const Endpoint *> endpoints;
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetEndpoints(queue_names, endpoints));
  for (size_t output_idx = 0U; output_idx < endpoints.size(); ++output_idx) {
    const auto endpoint = endpoints[output_idx];
    int32_t endpoint_index = -1;
    auto queue_info = BuildQueueInfo(*endpoint, model_instance_name);
    queue_info.owned = is_owned;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
    src_endpoint_indices_[endpoint->GetName()].emplace_back(endpoint_index);
    if (queue_info.owned) {
      auto &submodel_info = MutableSubmodelInfo(queue_info.model_instance_name);
      if (queue_info.queue_action == DeployPlan::QueueAction::kControl) {
        submodel_info.control_output_queue_indices.emplace_back(endpoint_index);
      } else {
        submodel_info.output_queue_indices.emplace_back(endpoint_index);
      }
    }
    GELOGD("Output endpoint created, model = %s, output_index = %zu, queue name = %s, queue index = %d",
           model_instance_name.c_str(), output_idx, endpoint->GetName().c_str(), endpoint_index);
  }
  return SUCCESS;
}

Status DeployPlannerBase::CreateFeedEndpoints(const std::string &model_instance_name,
                                              const std::vector<std::string> &queue_names,
                                              const std::string &invoke_key) {
  std::vector<const Endpoint *> endpoints;
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetEndpoints(queue_names, endpoints));
  for (size_t output_idx = 0UL; output_idx < endpoints.size(); ++output_idx) {
    const auto endpoint = endpoints[output_idx];
    if (endpoint->GetEndpointType() != EndpointType::kQueue) {
      continue;
    }
    int32_t endpoint_index = -1;
    auto queue_info = BuildQueueInfo(*endpoint, model_instance_name);
    queue_info.owned = true;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
    src_endpoint_indices_[endpoint->GetName()].emplace_back(endpoint_index);
    auto &submodel_info = MutableSubmodelInfo(queue_info.model_instance_name);
    submodel_info.invoked_model_queue_infos[invoke_key].feed_queue_indices.emplace_back(endpoint_index);
    GELOGD("Feed endpoint created, model = %s, invoke_key = %s, feed_index = %zu, queue name = %s, queue index = %d",
           model_instance_name.c_str(), invoke_key.c_str(), output_idx, endpoint->GetName().c_str(), endpoint_index);
  }
  return SUCCESS;
}

std::string DeployPlannerBase::GenShortName(const std::string &name) {
  auto &short_name = short_names_[name];
  if (short_name.empty()) {
    short_name = "deploy_planner.auto_generated:" + std::to_string(endpoint_name_id_gen_++);
    GELOGD("endpoint name too long, change from %s to %s", name.c_str(), short_name.c_str());
  }
  return short_name;
}

std::string DeployPlannerBase::GetEndpointFullName(const DeployPlan::QueueInfo &endpoint_info,
                                                   const ModelQueueIndex &model_queue_index) {
  std::stringstream ss;
  ss << endpoint_info.model_instance_name << ":" << model_queue_index.id << "_FROM_" << endpoint_info.name << "@"
     << endpoint_info.device_info.GetDesc() << "_T" << std::to_string(plan_id_gen_);
  const auto &name = ss.str();
  if (name.length() <= kMaxQueueNameLen) {
    return name;
  }
  return GenShortName(name);
}

ModelRelationFlattener::ModelRelationFlattener(PneModelPtr root_model) : root_model_(std::move(root_model)) {}

Status ModelRelationFlattener::Flatten(ModelRelation &flattened_model_relation,
                                       std::map<std::string, PneModelPtr> &name_to_models) {
  const auto &model_relation = root_model_->GetModelRelation();
  GE_CHECK_NOTNULL(model_relation, ", FlowModel's ModelRelation is nullptr, model_name = %s",
                   root_model_->GetModelName().c_str());
  flattened_model_relation_.root_model_endpoint_info = model_relation->root_model_endpoint_info;
  flattened_model_relation_.invoked_model_queue_infos = model_relation->invoked_model_queue_infos;
  MergeEndpoints({}, model_relation->endpoints);
  for (auto &it : model_relation->submodel_endpoint_infos) {
    const auto &submodel_info_info = it.second;
    const auto &model_name = it.first;
    const auto submodel = root_model_->GetSubmodel(model_name);
    GE_CHECK_NOTNULL(submodel, ", Failed to get submodel, submodel_name = %s", model_name.c_str());
    GE_CHK_STATUS_RET(FlattenSubmodel(submodel_info_info, submodel, 0), "Failed to flatten submodel %s",
                      model_name.c_str());
  }

  flattened_model_relation = std::move(flattened_model_relation_);
  name_to_models = std::move(leaf_models_);
  return SUCCESS;
}

bool ModelRelationFlattener::NeedFlatten(const PneModelPtr &root_model) {
  const auto &submodels = root_model->GetSubmodels();
  for (const auto &submodel : submodels) {
    if (!submodel.second->GetSubmodels().empty()) {
      return true;
    }
  }
  return false;
}

Status ModelRelationFlattener::Flatten(const PneModelPtr &root_model) {
  GE_CHECK_NOTNULL(root_model);
  const auto is_need_flatten = NeedFlatten(root_model);
  if (!is_need_flatten) {
    GELOGD("model is no need flatten, model %s", root_model->GetModelName().c_str());
    return SUCCESS;
  }
  if (root_model->GetModelRelation() == nullptr) {
    GELOGD("model need flatten but relation is null, need build relation, model %s",
           root_model->GetModelName().c_str());
    const auto &root_graph = root_model->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph, ", need build model relation, but root graph is null, model %s",
                     root_model->GetModelName().c_str());
    auto model_relation = MakeShared<ModelRelation>();
    GE_CHECK_NOTNULL(model_relation, ", need build model relation, but make shared failed, model %s",
                     root_model->GetModelName().c_str());
    GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*root_graph, *model_relation),
                      "Failed to build ModelRelation from root graph: %s.", root_graph->GetName().c_str());
    root_model->SetModelRelation(model_relation);
    GELOGD("make model relation success, model %s", root_model->GetModelName().c_str());
  }
  auto flattened_model_relation = MakeShared<ModelRelation>();
  GE_CHECK_NOTNULL(flattened_model_relation, ", Failed to make flatten model relation for model %s",
                   root_model->GetModelName().c_str());
  ModelRelationFlattener flattener(root_model);
  std::map<std::string, PneModelPtr> flattened_submodels;
  GE_CHK_STATUS_RET_NOLOG(flattener.Flatten(*flattened_model_relation, flattened_submodels));
  root_model->SetModelRelation(flattened_model_relation);
  root_model->SetSubmodels(flattened_submodels);
  GELOGD("model flatten end, model %s", root_model->GetModelName().c_str());
  return SUCCESS;
}

Status ModelRelationFlattener::FlattenSubmodel(const ModelRelation::ModelEndpointInfo &parent_model_queue_info,
                                               const PneModelPtr &pne_model, const int32_t depth) {
  const auto &submodels = pne_model->GetSubmodels();
  if (submodels.empty()) {  // is_leaf
    const auto &model_name = pne_model->GetModelName();
    (void)leaf_models_.emplace(model_name, pne_model);
    (void)flattened_model_relation_.submodel_endpoint_infos.emplace(model_name, parent_model_queue_info);
    GELOGD("Leaf submodel %s(%s) flattened to parent model %s", pne_model->GetModelName().c_str(),
           pne_model->GetModelType().c_str(), parent_model_queue_info.model_name.c_str());
    return SUCCESS;
  }

  if (depth >= max_depth_) {
    GELOGE(UNSUPPORTED, "Depth limit(%d) reached", max_depth_);
    return UNSUPPORTED;
  }

  GELOGD("To flatten submodel %s(%s) to parent model %s", pne_model->GetModelName().c_str(),
         pne_model->GetModelType().c_str(), parent_model_queue_info.model_name.c_str());
  const auto &model_relation = pne_model->GetModelRelation();
  GE_CHECK_NOTNULL(model_relation);
  GE_CHK_STATUS_RET_NOLOG(CheckConsistency(parent_model_queue_info, model_relation->root_model_endpoint_info));
  auto name_refs = BuildNameRefs(parent_model_queue_info, model_relation->root_model_endpoint_info);
  // add inner queue defs
  MergeEndpoints(name_refs, model_relation->endpoints);
  flattened_model_relation_.invoked_model_queue_infos.insert(model_relation->invoked_model_queue_infos.begin(),
                                                             model_relation->invoked_model_queue_infos.end());
  // process submodels
  for (auto &it : model_relation->submodel_endpoint_infos) {
    auto &submodel_info_info = it.second;
    ReplaceQueueNames(name_refs, submodel_info_info.input_endpoint_names);
    ReplaceQueueNames(name_refs, submodel_info_info.output_endpoint_names);
    auto submodel = pne_model->GetSubmodel(submodel_info_info.model_name);
    GE_CHECK_NOTNULL(submodel, "Failed to get submodel, parent_model = %s, submodel_name = %s",
                     pne_model->GetModelName().c_str(), submodel_info_info.model_name.c_str());
    GE_CHK_STATUS_RET(FlattenSubmodel(submodel_info_info, submodel, depth + 1), "Failed to flatten submodel %s",
                      submodel_info_info.model_name.c_str());
  }
  return SUCCESS;
}

void ModelRelationFlattener::ReplaceQueueNames(const std::map<std::string, std::string> &name_refs,
                                               std::vector<std::string> &names) {
  for (auto &name : names) {
    auto it = name_refs.find(name);
    if (it != name_refs.cend()) {
      name = it->second;
    }
  }
}

void ModelRelationFlattener::MergeEndpoints(const map<std::string, std::string> &name_refs,
                                            const vector<Endpoint> &endpoints) {
  for (const auto &endpoint : endpoints) {
    auto it = name_refs.find(endpoint.GetName());
    if (it == name_refs.cend()) {  // inner queue defs
      flattened_model_relation_.endpoints.emplace_back(endpoint);
    }
  }
}

std::map<std::string, std::string> ModelRelationFlattener::BuildNameRefs(
    const ModelRelation::ModelEndpointInfo &parent_model_queue_info,
    const ModelRelation::ModelEndpointInfo &root_model_queue_info) {
  std::map<std::string, std::string> name_refs;
  const auto &input_endpoint_names = root_model_queue_info.input_endpoint_names;
  const auto &output_endpoint_names = root_model_queue_info.output_endpoint_names;
  for (size_t i = 0; i < input_endpoint_names.size(); ++i) {
    name_refs[input_endpoint_names[i]] = parent_model_queue_info.input_endpoint_names[i];
  }
  for (size_t i = 0; i < output_endpoint_names.size(); ++i) {
    name_refs[output_endpoint_names[i]] = parent_model_queue_info.output_endpoint_names[i];
  }
  return name_refs;
}

Status ModelRelationFlattener::CheckConsistency(const ModelRelation::ModelEndpointInfo &parent_model_queue_info,
                                                const ModelRelation::ModelEndpointInfo &root_model_queue_info) {
  if (root_model_queue_info.input_endpoint_names.size() != parent_model_queue_info.input_endpoint_names.size()) {
    GELOGE(PARAM_INVALID, "input queue size(%zu) mismatches that of parent's (%zu)",
           root_model_queue_info.input_endpoint_names.size(), parent_model_queue_info.input_endpoint_names.size());
    return PARAM_INVALID;
  }

  if (root_model_queue_info.output_endpoint_names.size() != parent_model_queue_info.output_endpoint_names.size()) {
    GELOGE(PARAM_INVALID, "output queue size(%zu) mismatches that of parent's (%zu)",
           root_model_queue_info.output_endpoint_names.size(), parent_model_queue_info.output_endpoint_names.size());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

const bool &DeployPlannerBase::GetIsDynamicSched() const {
  return deploy_plan_.is_dynamic_sched_;
}

void DeployPlannerBase::GenerateDynamicSchedModelId() {
  std::map<std::string, std::vector<std::string>> model_instances;
  uint32_t id = 0U;
  deploy_plan_.dynamic_sched_plan_.submodels_id_["__head"] = id;
  deploy_plan_.dynamic_sched_plan_.submodels_id_["__tail"] = id;
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    const auto &submodel_endpoint_info = it.second;
    deploy_plan_.dynamic_sched_plan_.submodels_id_[model_instance_name] = ++id;
    model_instances[submodel_endpoint_info.model_name].emplace_back(model_instance_name);
    GELOGI("Submodel instance [%s], model_uuid=%u.", model_instance_name.c_str(), id);
  }
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &submodel_endpoint_info = it.second;
    deploy_plan_.dynamic_sched_plan_.model_instances_num_[it.first] =
      model_instances[submodel_endpoint_info.model_name].size();
    GELOGI("Submodel instance [%s] num=%u.", it.first.c_str(),
           deploy_plan_.dynamic_sched_plan_.model_instances_num_[it.first]);
  }
}

void DeployPlannerBase::UpdateRelationForDynamicSched() {
  if (!GetIsDynamicSched() && (!deploy_plan_.IsEnableExceptionCatch())) {
    GELOGI("DynamicSched flag close and exception catch is disable, don't add status queues.");
    return;
  }
  GELOGD("DynamicSched flag=%d, exception catch flag=%d.", static_cast<int32_t>(GetIsDynamicSched()),
         static_cast<int32_t>(deploy_plan_.IsEnableExceptionCatch()));
  const std::string status_queue_name = "__status_output";
  Endpoint queue_def(status_queue_name, EndpointType::kQueue);
  QueueNodeUtils(queue_def).SetDepth(kDepDefQueDepth).SetEnqueuePolicy("FIFO").
    SetNodeAction(kQueueActionStatus);
  model_relation_.endpoints.emplace_back(queue_def);

  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    model_relation_.submodel_endpoint_infos[it.first].status_output_queue_names.emplace_back(status_queue_name);
  }
  model_relation_.root_model_endpoint_info.status_output_queue_names.emplace_back(status_queue_name);
  GELOGI("DynamicSched add status report queue, name=%s.", status_queue_name.c_str());

  head_model_queue_info_.sched_output_queue_names = model_relation_.root_model_endpoint_info.sched_input_queue_names;
  head_model_queue_info_.status_output_queue_names = model_relation_.root_model_endpoint_info.status_input_queue_names;
  tail_model_queue_info_.sched_input_queue_names = model_relation_.root_model_endpoint_info.sched_output_queue_names;
  tail_model_queue_info_.status_input_queue_names = model_relation_.root_model_endpoint_info.status_output_queue_names;
}

Status DeployPlannerBase::AssignDynamicSchedEnqueueQueues() {
  GE_CHK_STATUS_RET_NOLOG(
      CreateDynamicSchedOutputQueueDefs(head_model_queue_info_.model_name,
                                        head_model_queue_info_.sched_output_queue_names));
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    GE_CHK_STATUS_RET_NOLOG(CreateDynamicSchedOutputQueueDefs(model_instance_name,
                                                              it.second.status_output_queue_names));
    GE_CHK_STATUS_RET_NOLOG(CreateDynamicSchedOutputQueueDefs(model_instance_name,
                                                              it.second.sched_output_queue_names));
  }
  return SUCCESS;
}

void DeployPlannerBase::DynamicSchedGroupFormat(const int32_t &real_entry_index,
                                                const int32_t &entry_index,
                                                const DeployPlan::QueueInfo *src_queue_info,
                                                const int32_t &src_q_idx,
                                                const int32_t &dst_q_idx) {
  auto &modelInfo = deploy_plan_.dynamic_sched_plan_.model_index_info_;
  auto iter = deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_.find(entry_index);
  if (iter == deploy_plan_.dynamic_sched_plan_.entry_to_dst_index_.end()) {
    GELOGI("DynamicSched, Step2, can't find entry index=%d from entry bindings.", entry_index);
  } else {
    const auto &dst_endpoint_info = deploy_plan_.queues_[iter->second];
    auto mul_submodel_id = 0;
    auto submodel_info = instance_to_model_name_.find(dst_endpoint_info.model_instance_name);
    if (submodel_info != instance_to_model_name_.end()) {
      auto model_info = model_name_to_id_.find(submodel_info->second);
      if (model_info != model_name_to_id_.end()) {
        mul_submodel_id = model_info->second;
      }
    }
    const auto model_id = src_queue_info->model_id;
    modelInfo[model_id][src_q_idx].first.device_info = src_queue_info->device_info;
    modelInfo[model_id][src_q_idx].first.submodel_instance_name = src_queue_info->model_instance_name;
    modelInfo[model_id][src_q_idx].first.is_normal = true;
    modelInfo[model_id][src_q_idx].second[dst_q_idx].model_id = mul_submodel_id;
    DeployPlan::ExtendedIndexInfo index_info;
    index_info.device_info = deploy_plan_.queues_[iter->second].device_info;
    index_info.submodel_instance_name = deploy_plan_.queues_[iter->second].model_instance_name;
    index_info.is_normal = true;
    auto &dst_submodule_info = MutableSubmodelInfo(dst_endpoint_info.model_instance_name);
    const bool is_redundant = dst_submodule_info.is_redundant;
    DeployPlan::DynamicGroupRouteInfo group_route_info = {
      real_entry_index,
      iter->second,
      index_info,
      is_redundant
    };
    modelInfo[model_id][src_q_idx].second[dst_q_idx].routes.push_back(group_route_info);
    GELOGI("DynamicSched, Step2, add src index bind info: src endpoint index=%d, logic group id=%d, "
           "group entry endpoint index=%d, group dst endpoint index=%d, src model instance name=%s, "
           "model_id=%u, dst model instance name=%s, redundant flag=%d.",
           src_q_idx, dst_q_idx, real_entry_index, iter->second,
           src_queue_info->model_instance_name.c_str(), model_id,
           dst_endpoint_info.model_instance_name.c_str(), static_cast<int32_t>(is_redundant));
  }
  return;
}

void DeployPlannerBase::AddDependentDevice(std::set<DeployPlan::DeviceInfo> &device_infos,
                                           const std::vector<int32_t> &queue_indexs) {
  for (const auto index : queue_indexs) {
    device_infos.emplace(deploy_plan_.queues_[index].device_info);
  }
}

void DeployPlannerBase::BuildModelDeployInfos() {
  for (auto &model_iter : deploy_plan_.GetModelDeployInfos()) {
    for (auto &model_instance_iter : model_iter.second) {
      const auto &model_instance_name = model_instance_iter.first;
      const auto &submodel_info = MutableSubmodelInfo(model_instance_name);
      AddDependentDevice(model_instance_iter.second, submodel_info.input_queue_indices);
      AddDependentDevice(model_instance_iter.second, submodel_info.control_input_queue_indices);
      AddDependentDevice(model_instance_iter.second, submodel_info.output_queue_indices);
      AddDependentDevice(model_instance_iter.second, submodel_info.control_output_queue_indices);
      AddDependentDevice(model_instance_iter.second, submodel_info.status_input_queue_indices);
      AddDependentDevice(model_instance_iter.second, submodel_info.status_output_queue_indices);
    }
  }
}

Status DeployPlannerBase::BuildDynamicSchedInfo() {
  for (auto &binding : deploy_plan_.GetQueueBindings()) {
    auto src_q_idx = binding.first;
    auto dst_q_idx = binding.second;
    const DeployPlan::QueueInfo *src_queue_info = nullptr;
    GE_CHK_STATUS_RET_NOLOG(deploy_plan_.GetQueueInfo(src_q_idx, src_queue_info));
    if (deploy_plan_.IsGroupEndpoint(src_q_idx) || !deploy_plan_.IsGroupEndpoint(dst_q_idx)) {
      continue;
    }
    auto queue_list_iter = deploy_plan_.GetGroups().find(dst_q_idx);
    GE_CHK_BOOL_RET_STATUS(queue_list_iter != deploy_plan_.GetGroups().end(),
                           FAILED, "DynamicSched, Step2, Get group[%d] info failed.", dst_q_idx);
    auto group_entries_index_start = static_cast<int32_t>(deploy_plan_.GetQueueInfoList().size());
    for (auto entry_index : queue_list_iter->second) {
      auto &entry_info = deploy_plan_.GetGroupEntryInfoList()[entry_index];
      auto real_entry_index = 0;
      if (entry_info.ref_index >= 0) {
        real_entry_index = entry_info.ref_index;
      } else {
        real_entry_index = group_entries_index_start + entry_index;
      }
      DynamicSchedGroupFormat(real_entry_index, entry_index, src_queue_info, src_q_idx, dst_q_idx);
    }
  }
  BuildModelDeployInfos();
  return SUCCESS;
}

Status DeployPlannerBase::SetHeadNodeInfo() {
  for (const auto &it : model_relation_.submodel_endpoint_infos) {
    const auto &model_instance_name = it.first;
    const auto input_names = it.second.input_endpoint_names;
    auto &submodel_info = MutableSubmodelInfo(model_instance_name);
    submodel_info.is_head = HasIntersection(input_names, model_relation_.root_model_endpoint_info.input_endpoint_names);
    GELOGI("SetHeadNodeInfo, name = %s set head %d", model_instance_name.c_str(), submodel_info.is_head);
  }
  return SUCCESS;
}

Status DeployPlannerBase::ResolveModelDynamicInputs(const std::string &model_instance_name,
                                                    const ModelRelation::ModelEndpointInfo &model_endpoint_info) {
  std::vector<const Endpoint *> model_input_endpoints;
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.sched_input_queue_names, model_input_endpoints));
  GE_CHK_STATUS_RET_NOLOG(
      relation_reader_->BatchGetEndpoints(model_endpoint_info.status_input_queue_names, model_input_endpoints));
  std::vector<ModelQueueIndex> model_queue_indexs;
  model_queue_indexs.reserve(model_input_endpoints.size());

  for (size_t input_index = 0UL; input_index < model_endpoint_info.sched_input_queue_names.size(); ++input_index) {
    ModelQueueIndex input_queue_index{model_endpoint_info.model_name + "_sched", "",
      static_cast<int32_t>(input_index)};
    model_queue_indexs.emplace_back(std::move(input_queue_index));
  }
  for (size_t input_index = 0UL; input_index < model_endpoint_info.status_input_queue_names.size(); ++input_index) {
    ModelQueueIndex input_queue_index{model_endpoint_info.model_name + "_status", "",
      static_cast<int32_t>(input_index)};
    model_queue_indexs.emplace_back(std::move(input_queue_index));
  }

  GE_CHK_BOOL_RET_STATUS(model_queue_indexs.size() == model_input_endpoints.size(), INTERNAL_ERROR,
                         "model_queue_indexs.size=%zu is not same as model_input_endpoints.size=%zu, model=%s",
                         model_queue_indexs.size(), model_input_endpoints.size(), model_instance_name.c_str());

  auto &submodel_info = MutableSubmodelInfo(model_instance_name);
  for (size_t index = 0UL; index < model_input_endpoints.size(); ++index) {
    const auto &model_queue_id = model_queue_indexs[index];
    const auto endpoint = model_input_endpoints[index];
    const auto &endpoint_name = endpoint->GetName();
    const auto &src_endpoint_indices = deploy_plan_.dynamic_sched_plan_.src_endpoint_indices_[endpoint_name];
    if (src_endpoint_indices.empty()) {
      GELOGE(PARAM_INVALID, "Failed to find enqueue operation for queue [%s]", endpoint_name.c_str());
      return PARAM_INVALID;
    }

    for (auto src_endpoint_index : src_endpoint_indices) {
      const auto &src_endpoint = deploy_plan_.queues_[static_cast<size_t>(src_endpoint_index)];
      auto &dst_endpoint_groups = deploy_plan_.dynamic_sched_plan_.endpoint_pairs_[src_endpoint_index];
      auto queue_info = BuildQueueInfo(*endpoint, model_instance_name);
      GE_CHK_STATUS_RET(AdjustDequeueDevice(queue_info, src_endpoint_indices), "Failed to adjust dequeue device");
      queue_info.name = GetEndpointFullName(queue_info, model_queue_id);
      relation_dst_to_src_[queue_info.name].emplace(src_endpoint_index);
      dst_endpoint_groups[model_queue_id].emplace_back(std::move(queue_info));
      GELOGD("DynamicSched Bind endpoints: name = %s, from %s to %s:%d@%s, invoke_key=%s.", endpoint_name.c_str(),
             src_endpoint.model_instance_name.c_str(), model_endpoint_info.model_name.c_str(), model_queue_id.id,
             submodel_info.device_info.GetDesc().c_str(), model_queue_id.invoke_key.c_str());
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::AssignDynamicSchedDequeueQueue(const DeployPlan::QueueInfo &queue_info,
                                                         const ModelQueueIndex &model_queue_loc,
                                                         const int32_t &src_endpoint_idx) {
  int32_t dst_endpoint_idx = -1;
  const auto &model_instance_name = queue_info.model_instance_name;
  if (reusable_queue_indices_.count(src_endpoint_idx) > 0UL) {
    GELOGI("DynamicSched, Reuse src queue, queue name = %s, queue index = %d",
           deploy_plan_.queues_[src_endpoint_idx].name.c_str(), src_endpoint_idx);
    dst_endpoint_idx = src_endpoint_idx;
  } else {
    GE_CHK_STATUS_RET_NOLOG(GetOrCreateInputEndpoint(model_queue_loc, queue_info, dst_endpoint_idx));
    GE_CHK_STATUS_RET_NOLOG(CreateDynamicSchedTags(src_endpoint_idx,
                                                   dst_endpoint_idx,
                                                   queue_info));
    GELOGI("DynamicSched, Endpoint binding added, src = %s, dst = %s", ToEndpointDesc(src_endpoint_idx).c_str(),
           ToEndpointDesc(dst_endpoint_idx).c_str());
  }
  if (queue_info.queue_action == DeployPlan::QueueAction::kStatus) {
    auto &submodel_info = MutableSubmodelInfo(model_instance_name);
    submodel_info.status_input_queue_indices.push_back(dst_endpoint_idx);
    GELOGI("DynamicSched, add status input indices, model instance name=%s, input indice=%d.",
           model_instance_name.c_str(), dst_endpoint_idx);
  } else {
    auto &submodel_info = MutableSubmodelInfo(model_instance_name);
    submodel_info.sched_input_queue_indices.push_back(dst_endpoint_idx);
    GELOGI("DynamicSched, add sched input indices, model instance name=%s, input indice=%d.",
           model_instance_name.c_str(), dst_endpoint_idx);
    deploy_plan_.dynamic_sched_plan_.datagw_request_bindings_[src_endpoint_idx] = dst_endpoint_idx;
    GELOGI("DynamicSched, datagw request bindings, datagw input=%d, sched app output=%d.",
           dst_endpoint_idx, src_endpoint_idx);
  }
  return SUCCESS;
}

Status DeployPlannerBase::AssignDynamicSchedDequeueQueues() {
  for (const auto &endpoint_pair : deploy_plan_.dynamic_sched_plan_.endpoint_pairs_) {
    const auto src_endpoint_idx = endpoint_pair.first;
    // group by model_and_input_idx
    for (const auto &queue_loc_and_queue_infos : endpoint_pair.second) {
      const auto &model_queue_loc = queue_loc_and_queue_infos.first;
      for (size_t i = 0; i < queue_loc_and_queue_infos.second.size(); ++i) {
        const auto &queue_info = queue_loc_and_queue_infos.second[i];
        GE_CHK_STATUS_RET_NOLOG(AssignDynamicSchedDequeueQueue(queue_info, model_queue_loc,
                                                               src_endpoint_idx));
      }
    }
  }
  return SUCCESS;
}

Status DeployPlannerBase::DynamicSchedBindGroup2Queue(const int32_t src_idx,
                                                      const int32_t dst_idx,
                                                      int32_t &group_index) {
  DeployPlan::QueueInfo group_info{};
  const auto &dst_endpoint_info = deploy_plan_.queues_[src_idx];
  group_info.name = dst_endpoint_info.name;
  group_info.device_info = dst_endpoint_info.device_info;
  group_info.model_instance_name = dst_endpoint_info.model_instance_name;
  GE_CHK_STATUS_RET(CreateGroupInfo(group_info, {dst_idx}, group_index));
  deploy_plan_.queue_bindings_.emplace_back(group_index, src_idx);
  return SUCCESS;
}

Status DeployPlannerBase::DynamicSchedBindQueue2Group(const int32_t src_idx,
                                                      const int32_t dst_idx,
                                                      int32_t &group_index) {
  DeployPlan::QueueInfo group_info{};
  const auto &src_endpoint_info = deploy_plan_.queues_[src_idx];
  group_info.name = src_endpoint_info.name;
  group_info.device_info = src_endpoint_info.device_info;
  group_info.model_instance_name = src_endpoint_info.model_instance_name;
  GE_CHK_STATUS_RET(CreateGroupInfo(group_info, {dst_idx}, group_index));
  deploy_plan_.queue_bindings_.emplace_back(src_idx, group_index);
  return SUCCESS;
}

void DeployPlannerBase::BindDynamicSchedDevQueue(const int32_t src_endpoint_idx,
                                                 const int32_t dst_endpoint_idx) {
  auto src_is_multi_connected = IsOutputMultiConnected(src_endpoint_idx);
  auto dst_is_multi_connected = IsInputMultiConnected(dst_endpoint_idx);
  GELOGI("DynamicSched, Src endpoint[%s] is one to many = %d, dst endpoint[%s] is many to one = %d",
          ToEndpointDesc(src_endpoint_idx).c_str(), src_is_multi_connected, ToEndpointDesc(dst_endpoint_idx).c_str(),
          dst_is_multi_connected);
  if (src_is_multi_connected && dst_is_multi_connected) {
    GELOGW("DynamicSched, shouldn't many to many relation.");
  }
  // 动态调度直接添加绑定关系（host场景）
  deploy_plan_.queue_bindings_.emplace_back(src_endpoint_idx, dst_endpoint_idx);
  GELOGI("DynamicSched, Add bind relation[%d -> %d] success, src endpoint[%s], dst endpoint[%s].",
          src_endpoint_idx, dst_endpoint_idx, ToEndpointDesc(src_endpoint_idx).c_str(),
          ToEndpointDesc(dst_endpoint_idx).c_str());
}

Status DeployPlannerBase::BindDynamicSchedHostQueue(const DeployPlan::DeviceInfo &src_device_info,
                                                    const DeployPlan::DeviceInfo &dst_device_info,
                                                    DeployPlan::QueueInfo &entry_info,
                                                    int32_t &src_endpoint_idx,
                                                    int32_t &dst_endpoint_idx) {
  if (src_device_info.WithProxy()) {
    auto src_proxy_queue_info = deploy_plan_.queues_[src_endpoint_idx];
    src_proxy_queue_info.device_info = src_device_info.ProxyDevice();
    int32_t proxy_index = -1;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(src_proxy_queue_info, proxy_index));
    deploy_plan_.queue_bindings_.emplace_back(src_endpoint_idx, proxy_index);
    GELOGI("DynamicSched, Mul-server add host bind relation[%d -> %d] success, src endpoint[%s], dst endpoint[%s].",
          src_endpoint_idx, proxy_index, ToEndpointDesc(src_endpoint_idx).c_str(),
          ToEndpointDesc(proxy_index).c_str());
    src_endpoint_idx = proxy_index;
  }
  if (dst_device_info.WithProxy()) {
    auto dst_proxy_queue_info = deploy_plan_.queues_[dst_endpoint_idx];
    dst_proxy_queue_info.device_info = dst_device_info.ProxyDevice();
    entry_info.device_info = dst_proxy_queue_info.device_info;
    int32_t proxy_index = -1;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(dst_proxy_queue_info, proxy_index));
    deploy_plan_.queue_bindings_.emplace_back(proxy_index, dst_endpoint_idx);
    GELOGI("DynamicSched, Mul-server add host bind relation[%d -> %d] success, src endpoint[%s], dst endpoint[%s].",
          proxy_index, dst_endpoint_idx, ToEndpointDesc(proxy_index).c_str(),
          ToEndpointDesc(dst_endpoint_idx).c_str());
    dst_endpoint_idx = proxy_index;
  }
  return SUCCESS;
}

Status DeployPlannerBase::CreateDynamicSchedTags(const int32_t src_endpoint_idx,
                                                 const int32_t dst_endpoint_idx,
                                                 const DeployPlan::QueueInfo &queue_info) {
  if (CheckAndAddRelation(src_endpoint_idx, dst_endpoint_idx)) {
    return SUCCESS;
  }
  int32_t group_index = -1;
  auto &src_device_info = deploy_plan_.queues_[src_endpoint_idx].device_info;
  const auto is_same_node = (src_device_info.GetNodeId() == queue_info.device_info.GetNodeId());
  const auto is_same_type = (src_device_info.GetType() == queue_info.device_info.GetType());
  // Queue -> Queue
  if ((src_device_info.GetKey() == queue_info.device_info.GetKey()) || (is_same_node && !is_same_type)) {
    BindDynamicSchedDevQueue(src_endpoint_idx, dst_endpoint_idx);
    return SUCCESS;
  }

  auto proxy_src_endpoint_idx = src_endpoint_idx;
  auto proxy_dst_endpoint_idx = dst_endpoint_idx;
  auto entry_info = queue_info;
  if (!is_same_node && (src_device_info.WithProxy() || queue_info.device_info.WithProxy())) {
    GE_CHK_STATUS_RET_NOLOG(BindDynamicSchedHostQueue(src_device_info, queue_info.device_info, entry_info,
                                                      proxy_src_endpoint_idx, proxy_dst_endpoint_idx));
  }

  // use dst to create tag, to make multi src to one dst use the same tag
    entry_info = deploy_plan_.queues_[proxy_src_endpoint_idx];
  // 多对一场景为了保证group中tag id不一样，需要queue name唯一
  if (queue_info.queue_action == DeployPlan::QueueAction::kStatus) {
    entry_info.name += deploy_plan_.queues_[proxy_src_endpoint_idx].model_instance_name;
  }
  // In src device, create output Queue -> Group of Tags
  std::pair<int32_t, int32_t> tag_pair;
  GE_CHK_STATUS_RET(GetOrCreateMappingTagPairEntry(proxy_dst_endpoint_idx, entry_info, tag_pair, false),
                    "Failed to create mapping tag pair entity.");
  GELOGI("DynamicSched, src endpoint [%d] [%s] add output tag [%d] [%s], dst endpoint [%d] [%s]"
         " add input tag [%d] [%s].", proxy_src_endpoint_idx, ToEndpointDesc(proxy_src_endpoint_idx).c_str(),
         tag_pair.first, ToEndpointDesc(tag_pair.first, true).c_str(), proxy_dst_endpoint_idx,
         ToEndpointDesc(proxy_dst_endpoint_idx).c_str(), tag_pair.second, ToEndpointDesc(tag_pair.second, true).c_str());

  // 直接创建一个元素的group，然后做绑定，要创建group的原因是目前部署信息加载是根据group做tag资源遍历的
  GE_CHK_STATUS_RET(DynamicSchedBindQueue2Group(proxy_src_endpoint_idx, tag_pair.first, group_index));
  GELOGI("DynamicSched, Output group binding added, local = %s, peer = %s.",
         deploy_plan_.queues_[proxy_src_endpoint_idx].device_info.GetDesc().c_str(),
         ToString(ToEndpointDescs(deploy_plan_.groups_[group_index], true)).c_str());

  if (!CheckAndAddRelation(tag_pair.second, proxy_dst_endpoint_idx, kDynamicSchedRelationSuffix)) {
    GE_CHK_STATUS_RET(DynamicSchedBindGroup2Queue(proxy_dst_endpoint_idx, tag_pair.second, group_index));
  }
  GELOGI("DynamicSched, Input group binding added, peer = %s, local = %s.",
         ToString(ToEndpointDescs(deploy_plan_.groups_[group_index], true)).c_str(),
         deploy_plan_.queues_[proxy_dst_endpoint_idx].device_info.GetDesc().c_str());
  return SUCCESS;
}

Status DeployPlannerBase::CreateDynamicSchedOutputQueueDefs(const std::string &model_instance_name,
                                                            const std::vector<std::string> &queue_names,
                                                            const bool is_owned) {
  std::vector<const Endpoint *> endpoints;
  GE_CHK_STATUS_RET_NOLOG(relation_reader_->BatchGetEndpoints(queue_names, endpoints));
  for (size_t output_idx = 0U; output_idx < endpoints.size(); ++output_idx) {
    const auto endpoint = endpoints[output_idx];
    int32_t endpoint_index = -1;
    auto queue_info = BuildQueueInfo(*endpoint, model_instance_name);
    queue_info.owned = is_owned;
    GE_CHK_STATUS_RET_NOLOG(CreateEndpointInfo(queue_info, endpoint_index));
    deploy_plan_.dynamic_sched_plan_.src_endpoint_indices_[endpoint->GetName()].emplace_back(endpoint_index);
    if (queue_info.owned) {
      auto &submodel_info = MutableSubmodelInfo(queue_info.model_instance_name);
      if (queue_info.queue_action == DeployPlan::QueueAction::kStatus) {
        submodel_info.status_output_queue_indices.emplace_back(endpoint_index);
      } else {
        submodel_info.sched_output_queue_indices.emplace_back(endpoint_index);
      }
    }
    GELOGD("DynamicSched Output endpoint created, model = %s, output_index = %zu, queue name = %s, queue index = %d",
           model_instance_name.c_str(), output_idx, endpoint->GetName().c_str(), endpoint_index);
  }
  return SUCCESS;
}

void DeployPlannerBase::UpdateDynamicSchedDeployPlan() {
  deploy_plan_.dynamic_sched_plan_.root_model_info_.status_input_queue_indices =
    std::move(head_model_info_.status_output_queue_indices);
  deploy_plan_.dynamic_sched_plan_.root_model_info_.sched_input_queue_indices =
    std::move(head_model_info_.sched_output_queue_indices);
  deploy_plan_.dynamic_sched_plan_.root_model_info_.status_output_queue_indices =
    std::move(tail_model_info_.status_input_queue_indices);
  deploy_plan_.dynamic_sched_plan_.root_model_info_.sched_output_queue_indices =
    std::move(tail_model_info_.sched_input_queue_indices);
}
}  // namespace ge
