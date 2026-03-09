/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/heterogeneous_model_deployer.h"
#include <thread>
#include <future>
#include "graph/ge_context.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/thread_pool/thread_pool.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "deploy/deployer/deploy_context.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/flowrm/flow_route_manager.h"
#include "deploy/flowrm/flow_route_planner.h"
#include "deploy/model_send/flow_model_sender.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
Status HeterogeneousModelDeployer::DeployModel(DeployContext &deploy_context, DeployState &deploy_state) {
  GE_TIMESTAMP_START(DeployModel);
  auto ret = DoDeployModel(deploy_context, deploy_state);
  GE_TIMESTAMP_EVENT_END(DeployModel, "deploying model");
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Error occurred while deploying model, model_id = %u", deploy_state.GetRootModelId());
    GELOGE(FAILED,
           "Error occurred while deploying model, now start rollback, model_id = %u",
           deploy_state.GetRootModelId());
    UndeployModel(deploy_state.deployed_node_ids_, deploy_state.GetRootModelId());
  }
  return ret;
}

Status HeterogeneousModelDeployer::DoDeployModelWithFlow(DeployContext &deploy_context, DeployState &deploy_state) {
  auto model_id = deploy_state.GetRootModelId();
  GELOGI("Start to deploy model, model_id = %u.", model_id);
  // 1. build deploy plan
  GE_TIMESTAMP_START(BuildPlan);
  GE_CHK_BOOL_RET_STATUS(BuildDeployPlan(deploy_state) == SUCCESS, FAILED, "Failed to build DeployPlan");
  GE_TIMESTAMP_EVENT_END(BuildPlan, "deploying in BuildDeployPlan stage");

  // 2. build flow route plan for each device
  GE_TIMESTAMP_START(ResolveFlowRoutePlans);
  GE_CHK_BOOL_RET_STATUS(FlowRoutePlanner::ResolveFlowRoutePlans(deploy_state) == SUCCESS,
                         FAILED, "Failed to build FlowRoutePlan");
  GE_TIMESTAMP_EVENT_END(ResolveFlowRoutePlans, "deploying in ResolveFlowRoutePlans stage");

  // 3. fork executor process and deploy hcom rank table
  GE_TIMESTAMP_START(DeployDevMaintenanceCfg);
  GE_CHK_STATUS_RET(FlowModelSender::DeployDevMaintenanceCfg(deploy_state), "Failed to deploy dev cfg");
  GE_TIMESTAMP_EVENT_END(DeployDevMaintenanceCfg, "deploying in DeployDevMaintenanceCfg stage");

  // 4. distribute models and metadata to each device
  // 4_1. distribute flow route plan
  GE_TIMESTAMP_START(TransferPlan);
  GE_CHK_BOOL_RET_STATUS(FlowModelSender::TransferFlowRoutePlan(deploy_state) == SUCCESS,
                         FAILED, "Failed to dispatched FlowRoutePlan");
  // 4_2. distribute deploy plan
  GE_CHK_BOOL_RET_STATUS(FlowModelSender::TransferDeployPlan(deploy_state) == SUCCESS,
                         FAILED, "Failed to dispatched DeployPlan");
  GE_TIMESTAMP_EVENT_END(TransferPlan, "deploying in TransferPlan stage");

  // pre-deploy local flow route
  GE_TIMESTAMP_START(PreDeployLocalFlowRoute);
  GE_CHK_BOOL_RET_STATUS(deploy_context.PreDeployLocalFlowRoute(deploy_state.GetRootModelId()) == SUCCESS,
                         FAILED, "Failed to pre-deploy flow route");
  GE_TIMESTAMP_EVENT_END(PreDeployLocalFlowRoute, "deploying in PreDeployLocalFlowRoute stage");

  // 4_3. distribute submodels
  GE_TIMESTAMP_START(TransferSubmodels);
  GE_CHK_BOOL_RET_STATUS(FlowModelSender::TransferSubmodels(deploy_state) == SUCCESS,
                         FAILED, "Failed to transfer Submodels");
  GE_TIMESTAMP_EVENT_END(TransferSubmodels, "deploying in TransferSubmodels stage");
  // 4_4. distribute var manager
  GE_TIMESTAMP_START(DeployRemoteVarManager);
  GE_CHK_BOOL_RET_STATUS(FlowModelSender().DeployRemoteVarManager(deploy_state) == SUCCESS, FAILED,
                         "Failed to sync remote VarManager");
  GE_TIMESTAMP_EVENT_END(DeployRemoteVarManager, "deploying in DeployRemoteVarManager stage");

  // 5. distribute dynamic datagw config
  GE_TIMESTAMP_START(TransferDataGwDeployPlan);
  GE_CHK_BOOL_RET_STATUS(FlowModelSender::TransferDataGwDeployPlan(deploy_state) == SUCCESS,
                         FAILED, "Failed to TransferDataGwDeployPlan");
  GE_TIMESTAMP_EVENT_END(TransferDataGwDeployPlan, "deploying in TransferDataGwDeployPlan stage");

  // 6. start to load models in each device
  GE_TIMESTAMP_START(LoadSubmodels);
  GE_CHK_BOOL_RET_STATUS(LoadSubmodels(deploy_context, deploy_state) == SUCCESS, FAILED, "Failed to load submodels");
  GE_TIMESTAMP_EVENT_END(LoadSubmodels, "deploying in LoadSubmodels stage");

  // 7. finish deployment of local flow route
  GE_TIMESTAMP_START(DeployLocalFlowRoute);
  GE_CHK_BOOL_RET_STATUS(deploy_context.DeployLocalFlowRoute(deploy_state.GetRootModelId()) == SUCCESS,
                         FAILED, "Failed to deploy flow route");
  GE_TIMESTAMP_EVENT_END(DeployLocalFlowRoute, "deploying in DeployLocalFlowRoute stage");
  GELOGI("Deploy model successfully, model_id = %u", deploy_state.GetRootModelId());
  return SUCCESS;
}

void HeterogeneousModelDeployer::BuildModelAttrs(const FlowModelPtr &flow_model, DeployPlan &deploy_plan) {
  const auto &models_esched_priority = flow_model->GetModelsEschedPriority();
  auto &submodels = deploy_plan.MutableSubmodels();
  for (auto &submodel : submodels) {
    if (submodel.second.model == nullptr) {
      GELOGI("model is null, model name[%s], no need build model attrs", submodel.first.c_str());
      continue;
    }
    const std::map<std::string, std::map<std::string, int32_t>>::const_iterator iter =
        models_esched_priority.find(submodel.second.model->GetModelName());
    if (iter != models_esched_priority.cend()) {
      for (const auto &priority : iter->second) {
        submodel.second.attrs[priority.first] = std::to_string(priority.second);
        GELOGD("[ModelEschedPriority]: model name[%s], %s[%d].", submodel.second.model->GetModelName().c_str(),
               priority.first.c_str(), priority.second);
      }
    }
    const auto &graph = submodel.second.model->GetRootGraph();
    int64_t npu_sched_model = 0;
    (void)AttrUtils::GetInt(graph, "_npu_sched_model", npu_sched_model);
    if (npu_sched_model != 0) {
      submodel.second.attrs["_npu_sched_model"] = std::to_string(npu_sched_model);
      GELOGD("model name[%s], npu_sched_model[%ld].", submodel.second.model->GetModelName().c_str(), npu_sched_model);
    }
  }
}

Status HeterogeneousModelDeployer::DoDeployModel(DeployContext &deploy_context, DeployState &deploy_state) {
  GE_CHK_STATUS_RET_NOLOG(DoDeployModelWithFlow(deploy_context, deploy_state));
  return SUCCESS;
}

Status HeterogeneousModelDeployer::BuildDeployPlan(DeployState &deploy_state) {
  // recursive deployment are not supported yet
  if (!deploy_state.local_submodel_descs_.empty()) {
    GELOGD("Deploy plan already resolved");
    return SUCCESS;
  }

  const auto &flow_model = deploy_state.GetFlowModel();
  GE_CHECK_NOTNULL(flow_model);
  bool is_host_cpu = GetContext().GetHostExecFlag();
  for (const auto &it : flow_model->GetSubmodels()) {
    auto &pne_model = it.second;
    if (pne_model->GetModelType().empty()) {
      pne_model->SetModelType(is_host_cpu ? PNE_ID_CPU : PNE_ID_NPU);
    }
    GELOGI("Model [%s] will deployed on engine [%s]", it.first.c_str(), pne_model->GetModelType().c_str());
  }

  // build deploy plan
  DeployPlan deploy_plan;
  deploy_plan.SetIsDynamicSched(deploy_state.GetIsDynamicSched());
  deploy_plan.SetEnableExceptionCatch(deploy_state.IsEnableExceptionCatch());
  GE_CHK_STATUS_RET_NOLOG(ResourceManager::GetInstance().AllocateResources(flow_model, deploy_plan));
  BuildModelAttrs(flow_model, deploy_plan);
  deploy_state.SetDeployPlan(std::move(deploy_plan));
  return SUCCESS;
}

Status HeterogeneousModelDeployer::LoadSubmodels(DeployContext &deploy_context, DeployState &deploy_state) {
  int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  auto root_model_id = deploy_state.root_model_id_;
  const auto &submodels = deploy_state.GetDeployPlan().GetSubmodels();
  std::set<int32_t> unique_node_ids;
  for (const auto &it : submodels) {
    if (it.second.model == nullptr) {
      continue;
    }
    const auto &target_device = it.second.device_info;
    unique_node_ids.emplace(target_device.GetNodeId());
    int32_t device_id = it.second.device_info.GetDeviceId();
    it.second.model->SetDeviceId(device_id);
    GELOGI("Success to set device id:%d, submodel:%s", device_id, it.second.model->GetModelName().c_str());
  }
  if (!deploy_state.local_submodel_descs_.empty()) {
    unique_node_ids.emplace(local_node_id);
  }
  ThreadPool thread_pool("ge_dpl_ldmd", unique_node_ids.size(), false);
  std::vector<int32_t> target_node_ids(unique_node_ids.cbegin(), unique_node_ids.cend());
  std::vector<std::future<Status>> deploy_futures;
  for (const auto &target_node_id : target_node_ids) {
    std::future<Status> fut;
    if (target_node_id == local_node_id) {
      fut = thread_pool.commit([&deploy_context, root_model_id]() -> Status {
        return LoadLocalModel(deploy_context, root_model_id);
      });
    } else {
      fut = thread_pool.commit([target_node_id, root_model_id]() -> Status {
        return LoadRemoteModel(target_node_id, root_model_id);
      });
    }
    deploy_futures.emplace_back(std::move(fut));
  }
  Status ret = SUCCESS;
  for (size_t i = 0U; i < deploy_futures.size(); ++i) {
    auto fut_ret = deploy_futures[i].get();
    if (fut_ret == SUCCESS) {
      deploy_state.deployed_node_ids_.emplace(target_node_ids[i]);
    } else {
      ret = fut_ret;
    }
  }
  return ret;
}

Status HeterogeneousModelDeployer::LoadLocalModel(DeployContext &deploy_context, uint32_t root_model_id) {
  auto ret = deploy_context.LoadLocalModel(root_model_id);
  GEEVENT("Local load model ended, model_id = %u, result = %u", root_model_id, ret);
  return ret;
}

Status HeterogeneousModelDeployer::LoadRemoteModel(int32_t node_id, uint32_t root_model_id) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kLoadModel);
  auto load_model_request = request.mutable_load_model_request();
  GE_CHECK_NOTNULL(load_model_request);
  load_model_request->set_root_model_id(root_model_id);
  deployer::DeployerResponse response;
  auto ret = DeployerProxy::GetInstance().SendRequest(node_id, request, response);
  GEEVENT("Remote load model ended, target_node = %d, model_id = %u, send_result = %u, response_code = %u",
          node_id, root_model_id, ret, response.error_code());
  GE_CHK_STATUS_RET(ret, "[Load] failed to send request, device_id = %d", node_id);
  const auto *node_info = DeployerProxy::GetInstance().GetNodeInfo(node_id);
  std::string node_ip_addr;
  if (node_info != nullptr) {
    const auto &device_list = node_info->GetDeviceList();
    if (!device_list.empty()) {
      node_ip_addr = device_list[0].GetHostIp();
    }
  }
  GE_CHK_BOOL_RET_STATUS(response.error_code() == SUCCESS, FAILED,
                         "Remote load model failed, node_id = %d, node_ip = %s, model_id = %u, error code = %u, "
                         "error message = %s", node_id, node_ip_addr.c_str(), root_model_id, response.error_code(),
                         response.error_message().c_str());
  GELOGD("Remote load model succeeded, node_id = %d, root_model_id = %u",
         node_id,
         root_model_id);
  return SUCCESS;
}

void HeterogeneousModelDeployer::UndeployModel(const std::set<int32_t> &deployed_node_ids, uint32_t root_model_id) {
  for (auto node_id : deployed_node_ids) {
    (void) DoUndeployModel(node_id, root_model_id);
  }
}

Status HeterogeneousModelDeployer::ClearNodelExceptionData(uint32_t node_id, uint32_t model_id,
                                                           const std::vector<DeployPlan::DeviceInfo> &device_infos,
                                                           const int32_t type) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kClearModelData);
  auto model_data_clear_req = request.mutable_model_data_clear();
  GE_CHECK_NOTNULL(model_data_clear_req);
  model_data_clear_req->mutable_root_model_ids()->Add(model_id);
  
  for (auto device_info : device_infos) {
    auto exception_devices = model_data_clear_req->add_exception_dev_info();
    exception_devices->set_device_id(device_info.GetDeviceId());
    exception_devices->set_device_type(device_info.GetType());
    exception_devices->set_node_id(device_info.GetNodeId());
  }
  model_data_clear_req->set_clear_type(type);
  model_data_clear_req->set_node_id(node_id);

  deployer::DeployerResponse response;
  GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(node_id, request, response),
                    "[ClearModelExceptionData] failed to send request, node_id = %u, msg_type = %d",
                    node_id, type);
  GE_CHK_BOOL_RET_STATUS(response.error_code() == SUCCESS,
                        FAILED,
                        "ClearModelExceptionData failed, node_id = %u, msg_type = %d, "
                        "error code = %u, error message = %s", node_id, type,
                        response.error_code(),
                        response.error_message().c_str());
  GELOGD("[ClearModelExceptionData] succeeded, node_id = %u, msg_type = %d", node_id, type);
  return SUCCESS;
}

Status HeterogeneousModelDeployer::DoUndeployModel(int32_t node_id, uint32_t root_model_id) {
  GELOGD("[Undeploy][Submodels] start, node_id = %d, root_model_id = %u.", node_id, root_model_id);
  deployer::DeployerRequest request;
  request.set_type(deployer::kUnloadModel);
  auto unload_model_req = request.mutable_unload_model_request();
  GE_CHECK_NOTNULL(unload_model_req);
  unload_model_req->set_model_id(root_model_id);
  deployer::DeployerResponse response;
  GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(node_id, request, response),
                    "[Undeploy][Submodels] failed to send request, node_id = %d, root_model_id = %u",
                    node_id, root_model_id);
  GE_CHK_BOOL_RET_STATUS(response.error_code() == SUCCESS,
                         FAILED,
                         "Undeploy submodels failed, device_id = %d, model_id = %u, error code = %u error message = %s",
                         node_id,
                         root_model_id,
                         response.error_code(),
                         response.error_message().c_str());
  GELOGD("[Undeploy][Submodels] succeeded, device_id = %d, model_id = %u", node_id, root_model_id);
  return SUCCESS;
}

Status HeterogeneousModelDeployer::UpdateRemoteProfiling(const bool is_prof_start, const std::string &prof_data,
           const std::map<uint32_t, std::set<int32_t>> &model_id_to_nodes) {
  for (const auto &it : model_id_to_nodes) {
    for (const auto &node_id : it.second) {
      deployer::DeployerRequest request;
      request.set_type(deployer::kUpdateProfilingInfo);
      auto prof_info = request.mutable_prof_info();
      prof_info->set_is_prof_start(is_prof_start);
      prof_info->set_prof_data(prof_data);
      prof_info->set_model_id(it.first);
      deployer::DeployerResponse response;
      GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(node_id, request, response),
                        "[InitProcessResource] failed to send request, node id=%d.", node_id);
      GE_CHK_BOOL_RET_STATUS(response.error_code() == SUCCESS, FAILED,
                             "[InitProcessResource] failed, node id=%d, error code=%u, error message=%s.",
                             node_id, response.error_code(), response.error_message().c_str());
      GELOGI("[UpdateRemoteProfiling] success, node id=%d, model_id=%u.", node_id, it.first);
    }
  }
  return SUCCESS;
}
}  // namespace ge
