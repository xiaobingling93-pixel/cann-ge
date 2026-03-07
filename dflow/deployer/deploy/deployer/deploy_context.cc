/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/deploy_context.h"
#include <string>
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/thread_pool/thread_pool.h"
#include "nlohmann/json.hpp"
#include "graph_metadef/graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "securec.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/config/configurations.h"
#include "dflow/base/utils/process_utils.h"
#include "deploy/execfwk/udf_executor_client.h"
#include "deploy/deployer/heterogeneous_model_deployer.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"
#include "deploy/flowrm/flow_route_manager.h"
#include "common/data_flow/route/rank_table_builder.h"
#include "common/utils/rts_api_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
namespace {
constexpr int32_t kDequeueTimeout = 30 * 1000; // ms
constexpr size_t kMaxDeviceSize = 8U;
constexpr size_t kMaxFlowRouteSize = 8 * 1024U;

using HService = HeterogeneousExchangeService;

void GetQueueIdsFromAttrs(const std::vector<DeployQueueAttr> &queue_attrs, std::vector<uint32_t> &queue_id) {
  for (const auto attr : queue_attrs) {
    queue_id.emplace_back(attr.queue_id);
  }
}
}  // namespace

DeployContext &DeployContext::LocalContext() {
  static DeployContext context;
  return context;
}

void DeployContext::SetName(const std::string &name) {
  name_ = name;
}

const std::string &DeployContext::GetName() const {
  return name_;
}

void DeployContext::SetDeployerPid(int32_t pid) {
  deployer_pid_ = pid;
}

int32_t DeployContext::GetDeployerPid() const {
  return deployer_pid_;
}

void DeployContext::Initialize() {
  client_base_dir_ = Configurations::GetInstance().GetDeployResDir();
  base_dir_ = client_base_dir_ + name_ + "/";
  GELOGI("[%s] DeployContext initialized.", name_.c_str());
}

void DeployContext::Finalize() {
  GEEVENT("[%s] DeployContext finalize begin.", name_.c_str());
  executor_manager_.Finalize();
  // destroy queue after executor exit
  for (const auto &model_routes : submodel_routes_) {
    (void)FlowRouteManager::GetInstance().UndeployRoute(FlowRouteType::kModelFlowRoute, model_routes.second,
                                                        flowgw_client_manager_);
  }
  submodel_routes_.clear();
  for (const auto &it : tansfer_routes_) {
    (void) HeterogeneousExchangeDeployer::Undeploy(HService::GetInstance(), it.second, flowgw_client_manager_);
  }
  tansfer_routes_.clear();
  flow_model_receiver_.DestroyAllDeployStates();
  for (const auto &device_id_and_var_manager: var_managers_) {
    for (const auto &session_id_and_var_manager : device_id_and_var_manager.second) {
      session_id_and_var_manager.second->Finalize();
    }
  }
  for (const auto &it : transfer_queues_) {
    (void) HService::GetInstance().DestroyQueue(it.first, it.second);
  }
  transfer_queues_.clear();
  Clear();
  (void) flowgw_client_manager_.Finalize();
  GEEVENT("[%s] DeployContext finalized.", name_.c_str());
}

Status DeployContext::DownloadDevMaintenanceCfg(const deployer::DeployerRequest &request,
                                                deployer::DeployerResponse &response) {
  auto &download_conf_req = request.download_config_request();
  const auto sub_type = download_conf_req.sub_type();
  auto &config_data = download_conf_req.config_data();
  const auto device_id = download_conf_req.device_id();
  // get device debug config handle
  GELOGI("DownloadDevMaintenanceCfg enter, device_id = %d, sub_type = %d.", device_id, sub_type);
  const void *config_buffer = config_data.data();
  uint64_t config_buffer_size = config_data.size();
  std::string config_str(static_cast<const char_t *>(config_buffer), config_buffer_size);
  GELOGI("DownloadDevMaintenanceCfg config_buffer_size = %lu, config_str = %s.",
         config_buffer_size, config_str.c_str());
  // load json data to device debug config
  if (dev_maintenance_cfg_.LoadJsonData(config_str) != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Parse model failed");
    return FAILED;
  }
  response.set_error_code(SUCCESS);
  GELOGI("[Handle][DownloadDevMaintenanceCfg] success.");
  return SUCCESS;
}

Status DeployContext::ProcessMultiVarManager(const deployer::MultiVarManagerRequest &request) {
  GELOGD("[process][var_manager] Begin.");
  const auto &multi_var_manager_info = request.multi_var_manager_info();
  for (const auto &info : multi_var_manager_info.var_manager_info()) {
    auto session_id = info.session_id();
    for (int32_t device_id : request.device_ids()) {
      auto var_manager_info = info;
      var_manager_info.set_device_id(device_id);
      std::lock_guard<std::mutex> lk(mu_);
      auto &var_manager = var_managers_[device_id][session_id];
      GE_CHECK_LE(var_managers_.size(), kMaxDeviceSize);
      if (var_manager == nullptr) {
        auto new_manager = MakeUnique<DeployerVarManager>();
        GE_CHECK_NOTNULL(new_manager);
        GE_CHK_STATUS_RET(new_manager->Initialize(std::move(var_manager_info)),
                          "Failed to initialize VarManager, device_id = %d, session_id = %lu",
                          device_id, session_id);
        var_manager = std::move(new_manager);
        var_manager->SetBasePath(base_dir_);
        var_manager->SetShareVarMem(true);
      } else {
        var_manager->SetVarManagerInfo(std::move(var_manager_info));
      }
    }
  }
  GELOGD("[process][var_manager] SUCCESS.");
  return SUCCESS;
}

Status DeployContext::GetOrCreateTransferQueue(int32_t device_id, uint32_t &queue_id) {
  const auto &it = transfer_queues_.find(device_id);
  if (it != transfer_queues_.end()) {
    queue_id = it->second;
    GELOGI("Get transfer queue_id[%u] success, device_id = %d.", queue_id, device_id);
    return SUCCESS;
  }

  const uint32_t kMsgQueueDepth = 128U;
  std::string queue_name = "queue.transfer_" + std::to_string(device_id);
  MemQueueAttr mem_queue_attr{};
  mem_queue_attr.depth = kMsgQueueDepth;
  mem_queue_attr.work_mode = RT_MQ_MODE_PULL;
  mem_queue_attr.is_client = true;
  GE_CHK_STATUS_RET(HService::GetInstance().CreateQueue(device_id, queue_name, mem_queue_attr, queue_id),
                    "[Create][TransferQueue] failed, device_id = %d", device_id);
  transfer_queues_[device_id] = queue_id;
  return SUCCESS;
}

Status DeployContext::ProcessSharedContent(const deployer::SharedContentDescRequest &request,
                                           const deployer::DeployerResponse &response) {
  (void) response;
  GE_CHK_BOOL_RET_STATUS(request.has_shared_content_desc(), PARAM_INVALID, "Request shared_content_desc is not set");
  GE_CHK_BOOL_RET_STATUS(!request.device_ids().empty(), PARAM_INVALID, "request device id not set");

  const auto &flow_route = request.flow_route();
  auto &exchange_service = HService::GetInstance();
  GELOGD("start to deploy transfer flow route.");
  auto local_route =
      FlowRouteManager::GetInstance().QueryRoute(FlowRouteType::kTransferFlowRoute, flow_route.route_id());
  if (local_route == nullptr) {
    GE_CHECK_LE(tansfer_routes_.size() + 1, kMaxFlowRouteSize);
    const auto &it = tansfer_routes_.find(flow_route.route_id());
    if (it == tansfer_routes_.cend()) {
      GELOGD("Start to create transfer route.");
      HeterogeneousExchangeDeployer deployer(exchange_service, request.flow_route(),
                                      flowgw_client_manager_);
      ExchangeRoute exchange_route;
      GE_TIMESTAMP_START(DeployRoute);
      GE_CHK_STATUS_RET_NOLOG(deployer.Deploy(exchange_route));
      GE_TIMESTAMP_EVENT_END(DeployRoute, "DeployRoute");
      tansfer_routes_[flow_route.route_id()] = std::move(exchange_route);
    }
    local_route = &tansfer_routes_[flow_route.route_id()];
  }

  GE_TIMESTAMP_START(ReceiveFile);

  auto &content_desc = request.shared_content_desc();
  uint64_t session_id = content_desc.session_id();
  uint64_t total_size = content_desc.total_length();
  uint64_t offset = 0U;
  uint64_t data_offset = 0U;
  while (offset < total_size) {
    rtMbufPtr_t m_buf = nullptr;
    std::vector<DeployQueueAttr> queue_attrs;
    local_route->GetQueueAttrs(queue_attrs);
    GE_CHK_BOOL_RET_STATUS(queue_attrs.size() >= 1U, FAILED,
                           "Check transfer deploy queue size[%zu] failed.",
                           queue_attrs.size());
    const auto &deq_attr = queue_attrs[0U];
    GE_CHK_STATUS_RET_NOLOG(exchange_service.DequeueMbuf(deq_attr.device_id, deq_attr.queue_id, &m_buf,
                                                         kDequeueTimeout));
    GE_MAKE_GUARD(m_buf, [m_buf]() { GE_CHK_RT(rtMbufFree(m_buf)); });
    uint64_t buffer_size = 0U;
    GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferSize(m_buf, buffer_size));
    GE_CHK_BOOL_RET_STATUS(buffer_size > 0U, FAILED, "Get buff size is 0.");
    GE_CHECK_LE(offset, UINT64_MAX - buffer_size);
    offset += buffer_size;
    GELOGD("Dequeue shared content buffer size[%lu] of total size[%lu] from queue[%u] in device id[%d]", buffer_size,
           total_size, deq_attr.queue_id, deq_attr.device_id);
    GE_CHECK_LE(offset, total_size);
    GELOGD("Process shared content dequeue successfully, size = %lu, current/total = %zu/%zu",
           buffer_size, offset, total_size);

    for (int32_t device_id : request.device_ids()) {
      auto var_manager = GetVarManager(device_id, session_id);
      GE_CHECK_NOTNULL(var_manager);
      uint32_t transfer_queue_id = 0U;
      GE_CHK_STATUS_RET(GetOrCreateTransferQueue(device_id, transfer_queue_id), "Failed to get transfer queue");
      GELOGD("enqueue to queue[%u] in device[%d]", transfer_queue_id, device_id);
      GE_CHK_STATUS_RET(HService::EnqueueMbufToClientQueue(device_id, transfer_queue_id, m_buf, kDequeueTimeout),
                        "Failed to enqueue mbuf to client, device_id=%d, queue_id=%u, timeout=%d", device_id,
                        transfer_queue_id, kDequeueTimeout);

      GE_CHK_STATUS_RET(var_manager->ProcessSharedContent(content_desc, buffer_size, data_offset, transfer_queue_id),
                        "Failed to process shared content, device_id=%d, queue_id=%u,buffer_size=%lu, data_offset=%lu.",
                        device_id, transfer_queue_id, buffer_size, data_offset);
    }
    data_offset += buffer_size;
    GELOGD("Process shared content successfully, size = %lu, current/total = %lu/%lu",
           buffer_size, offset, total_size);
  }
  GE_TIMESTAMP_EVENT_END(ReceiveFile, "ReceiveFile");
  return SUCCESS;
}

void DeployContext::AbnormalPidsToSubmodelInstance(deployer::DeployerResponse &response,
    std::map<ExecutorManager::ExecutorKey, bool> &abnormal_pids) {
  GELOGI("Start change abnormal pids to submodel instance, abnormal_pids size=%zu,"
      " rootmodel_to_submodel list size=%zu",
      abnormal_pids.size(), local_rootmodel_to_submodel_descs_.size());
  auto abnormal_submodel_instance_name =
      response.mutable_heartbeat_response()->mutable_abnormal_submodel_instance_name();
  std::lock_guard<std::mutex> lk(abnormal_heartbeat_info_mu_);
  for (auto &iter_rootmodel_to_submodel_descs_ : local_rootmodel_to_submodel_descs_) {
    auto root_model_id = iter_rootmodel_to_submodel_descs_.first;
    GELOGI("root_model_id=%u, submodel size=%zu,",
        root_model_id, iter_rootmodel_to_submodel_descs_.second.size());
    deployer::SubmodelInstanceName submodel_instance_name_rsp;
    for (auto &iter_submodel_descs : iter_rootmodel_to_submodel_descs_.second) {
      auto iter_pid = abnormal_pids.find(iter_submodel_descs.first);
      if (iter_pid == abnormal_pids.end()) {
        continue;
      }
      for (auto &submodel_instance : iter_submodel_descs.second) {
        (*(submodel_instance_name_rsp.mutable_submodel_instance_name()))[submodel_instance.
            model_instance_name()] = false;
        GELOGI("Add to response: root_model_id:%d, submodel_instance_name:%s",
          root_model_id, submodel_instance.model_instance_name().c_str());
      }
    }
    GELOGI("submodel instance name list size:%zu",
        (*(submodel_instance_name_rsp.mutable_submodel_instance_name())).size());
    if (!(*(submodel_instance_name_rsp.mutable_submodel_instance_name())).empty()) {
      ((*abnormal_submodel_instance_name)[root_model_id]) = submodel_instance_name_rsp;
    }
  }

  return;
}

void DeployContext::AddAbnormalSubmodelInstance(deployer::DeployerResponse &response,
    const std::map<uint32_t, std::vector<std::string>> &model_instance_name) {
  GELOGI("Start add abnormal submodel instance to response, model_instance_name size=%zu",
      model_instance_name.size());
  auto abnormal_submodel_instance_name =
      response.mutable_heartbeat_response()->mutable_abnormal_submodel_instance_name();
  std::lock_guard<std::mutex> lk(abnormal_heartbeat_info_mu_);
  for (auto &iter : model_instance_name) {
    auto root_model_id = iter.first;
    GELOGI("root_model_id=%u, submodel size=%zu,", root_model_id, iter.second.size());
    deployer::SubmodelInstanceName submodel_instance_name_rsp;
    for (auto &submodel_instance : iter.second) {
      (*(submodel_instance_name_rsp.mutable_submodel_instance_name()))[submodel_instance] = false;
      GELOGI("Add to response: root_model_id:%u, submodel_instance_name:%s",
        root_model_id, submodel_instance.c_str());
    }
    GELOGI("submodel instance name list size:%zu",
        (*(submodel_instance_name_rsp.mutable_submodel_instance_name())).size());
    if (!(*(submodel_instance_name_rsp.mutable_submodel_instance_name())).empty()) {
      ((*abnormal_submodel_instance_name)[root_model_id]) = submodel_instance_name_rsp;
    }
  }
  return;
}

Status DeployContext::ProcessHeartbeat(const deployer::DeployerRequest &request,
                                       deployer::DeployerResponse &response) {
  (void) request;
  // take device error msg back by heartbeat response
  std::map<uint32_t, std::vector<uint32_t>> abnormal_device_info;
  DeviceAbnormalStatusHandler::Instance().HandleDeviceAbnormal(abnormal_device_info);
  if (!abnormal_device_info.empty()) {
    auto proto_abnormal_device = response.mutable_heartbeat_response()->mutable_abnormal_device();
    for (const auto &abnormal_info : abnormal_device_info) {
      GELOGW("device[%u] error[%s]", abnormal_info.first, ToString(abnormal_info.second).c_str());
      (*proto_abnormal_device)[abnormal_info.first].mutable_error_code()->Add(abnormal_info.second.cbegin(),
                                                                              abnormal_info.second.cend());
    }
  }

  flowgw_client_manager_.GetFlowGwStatus(response);
  if (response.error_code() == FAILED) {
    return SUCCESS;
  }

  std::map<ExecutorManager::ExecutorKey, bool> abnormal_pids;
  std::map<uint32_t, std::vector<std::string>> model_instance_name;
  executor_manager_.GetExecutorStatus(response, abnormal_pids, model_instance_name);
  if (response.error_code() == FAILED) {
    if (!model_instance_name.empty()) { // 对于executor client下多个执行进程场景通过异常执行进程获取模型实例名
      AddAbnormalSubmodelInstance(response, model_instance_name);
      return SUCCESS;
    }
    AbnormalPidsToSubmodelInstance(response, abnormal_pids);
  }
  return SUCCESS;
}

Status DeployContext::LoadLocalModel(uint32_t root_model_id) {
  DeployState *deploy_state = nullptr;
  GE_CHK_STATUS_RET(flow_model_receiver_.GetDeployState(root_model_id, deploy_state),
                    "Failed to get deploy plan, root_model_id = %u",
                    root_model_id);
  GE_CHK_STATUS_RET(LoadLocalSubmodels(*deploy_state),
                    "Failed to load local submodels, root_model_id = %u",
                    root_model_id);
  return SUCCESS;
}

Status DeployContext::GetFlowRoute(const DeployState &deploy_state,
                                   const ExchangeRoute *&flow_route) {
  flow_route = deploy_state.node_exchange_deployer_pair_.second->GetRoute();
  return SUCCESS;
}

Status DeployContext::VarManagersPreAlloc(DeployState &deploy_state) {
  GE_TIMESTAMP_START(VarManagersPreAlloc);
  std::set<DeployerVarManager *> alloc_var_managers;
  for (const auto &it : deploy_state.local_submodel_descs_) {
    const auto &executor_key = it.first;
    if (executor_key.device_type != static_cast<int32_t>(NPU) || executor_key.engine_name != PNE_ID_NPU) {
      continue;
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto &var_managers = var_managers_[executor_key.device_id];
    const auto &var_it = var_managers.find(deploy_state.GetSessionId());
    if (var_it != var_managers.cend()) {
      alloc_var_managers.emplace(var_it->second.get());
    }
  }
  ThreadPool pool("ge_dpl_avm", alloc_var_managers.size(), false);
  std::vector<std::future<Status>> fut_rets;
  for (auto var_manager_ptr : alloc_var_managers) {
    auto fut = pool.commit([this, var_manager_ptr]() -> Status {
      GE_CHK_STATUS_RET(VarManagerPreAlloc(*var_manager_ptr), "Failed to pre alloc var manager.");
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }

  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to pre alloc var manager");
  }
  GE_TIMESTAMP_EVENT_END(VarManagersPreAlloc, "VarManagers pre alloc");
  return SUCCESS;
}

Status DeployContext::VarManagerPreAlloc(DeployerVarManager &var_manager) const {
  auto &var_manager_info = var_manager.MutableVarManagerInfo();
  auto var_resource = var_manager_info.mutable_var_resource();
  auto var_addr_mgr_map = var_resource->mutable_var_dev_addr_mgr_map();
  for (auto &var_addr_mgr : *var_addr_mgr_map) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&var_addr_mgr.second.desc(), tensor_desc);
    int64_t total_size = 0L;
    (void)TensorUtils::GetSize(tensor_desc, total_size);
    GE_CHK_BOOL_RET_STATUS(total_size > 0, PARAM_INVALID, "Tensor size is empty");
    void *dev_addr = nullptr;
    GE_CHK_STATUS_RET(var_manager.GetVarMemAddr(var_addr_mgr.first,
        static_cast<uint64_t>(total_size), &dev_addr, false),
        "[GetVarMemAddr]Get or Malloc shared memory failed");
    var_addr_mgr.second.set_dev_addr(PtrToValue(dev_addr));
  }
  return SUCCESS;
}

Status DeployContext::LoadLocalSubmodels(DeployState &deploy_state) {
  local_rootmodel_to_submodel_descs_[deploy_state.GetRootModelId()] = deploy_state.local_submodel_descs_;
  GE_CHK_STATUS_RET(VarManagersPreAlloc(deploy_state), "Failed to pre alloc var managers");
  GE_CHK_STATUS_RET(PrepareExecutors(deploy_state), "Failed to prepare executors");
  auto parallel_num = deploy_state.local_submodel_descs_.size();
  if (parallel_num > 1U) {
    GELOGI("Load model in parallel, num = %zu", parallel_num);
    ThreadPool pool("ge_dpl_ldlm", static_cast<uint32_t>(parallel_num), false);
    std::vector<std::future<Status>> fut_rets;
    for (const auto &it : deploy_state.local_submodel_descs_) {
      const auto &key = it.first;
      auto fut = pool.commit([this, &deploy_state, &key]() -> Status {
        GE_CHK_STATUS_RET_NOLOG(DoLoadSubmodels(deploy_state, key));
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    for (auto &fut : fut_rets) {
      GE_CHK_STATUS_RET_NOLOG(fut.get());
    }
  } else {
    GELOGI("Load model in sequence");
    for (const auto &it : deploy_state.local_submodel_descs_) {
      GE_CHK_STATUS_RET_NOLOG(DoLoadSubmodels(deploy_state, it.first));
    }
  }
  GELOGI("All submodels are loaded successfully.");
  return SUCCESS;
}

Status DeployContext::PrepareStateWorkingDir(const DeployState &deploy_state) const {
  auto state_working_dir = base_dir_ + deploy_state.GetRelativeWorkingDir();
  GE_CHK_STATUS_RET(ProcessUtils::IsValidPath(state_working_dir),
                   "State working path[%s] is invalid.", state_working_dir.c_str());
  GE_CHK_BOOL_RET_STATUS((mmAccess2(state_working_dir.c_str(), M_F_OK) == EN_OK) ||
                         (ProcessUtils::CreateDir(state_working_dir) == 0), FAILED,
                        "Failed to create directory: %s", state_working_dir.c_str());
  return SUCCESS;
}

Status DeployContext::PrepareExecutors(const DeployState &deploy_state) {
  GE_TIMESTAMP_START(PrepareExecutors);
  GE_CHK_STATUS_RET(PrepareStateWorkingDir(deploy_state), "Failed to prepare state working dir");
  ThreadPool pool("ge_dpl_pree", deploy_state.local_submodel_descs_.size(), false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &it : deploy_state.local_submodel_descs_) {
    const auto &executor_key = it.first;
    const auto &submodel_descs = it.second;
    auto fut = pool.commit([this, &executor_key, &deploy_state, &submodel_descs]() -> Status {
      PneExecutorClient *executor_client = nullptr;
      PneExecutorClient::ClientContext context = {};
      context.device_id = executor_key.device_id;
      context.device_type = executor_key.device_type;
      context.process_id = executor_key.process_id;
      context.deployer_pid = deployer_pid_;
      context.dev_maintenance_cfg = &dev_maintenance_cfg_;
      context.base_dir = client_base_dir_;
      context.options = deploy_state.GetAllGlobalOptions();
      GE_CHK_STATUS_RET(executor_manager_.GetOrCreateExecutorClient(executor_key, context, &executor_client),
                        "Failed to get executor");
      GE_CHK_STATUS_RET(SyncVarManagers(*executor_client, deploy_state), "Failed to sync var manager.");
      GE_CHK_STATUS_RET(executor_client->PreProcess(submodel_descs, base_dir_), "Failed to client pre process.");
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }

  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to create client and sync var manager");
  }
  GE_TIMESTAMP_EVENT_END(PrepareExecutors, "Prepare executors");
  return SUCCESS;
}

Status DeployContext::UpdateLocalProfiling(const bool is_prof_start, const std::string &prof_data,
                                           const std::vector<uint32_t> &model_ids) {
  for (const auto it : model_ids) {
    GELOGD("UpdateLocalProfiling, is_prof_start = %d, model_id = %u", is_prof_start, it);
    GE_CHK_STATUS(UpdateLocalProfilingInfo(is_prof_start, prof_data, it), "UpdateLocalProf failed, model id %u", it);
  }
  return SUCCESS;
}

Status DeployContext::UpdateLocalProfilingInfo(const bool is_prof_start, const std::string &prof_data,
                                               const uint32_t model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = submodel_devices_.find(model_id);
  if (it == submodel_devices_.end()) {
    GELOGI("Model not deployed, model_id = %u", model_id);
  } else {
    auto parallel_num = it->second.size();
    GELOGI("Update local profiling info in parallel, num = %zu", parallel_num);
    ThreadPool pool("ge_dpl_prof", static_cast<uint32_t>(parallel_num), false);
    std::vector<std::future<Status>> fut_rets;
    for (auto &executor_key : it->second) {
      auto fut = pool.commit([this, &executor_key, model_id, is_prof_start, &prof_data]() -> Status {
        GE_CHK_STATUS_RET(UpdateProfiling(is_prof_start, prof_data, executor_key),
                          "Update profiling failed, model_id %u.", model_id);
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    for (auto &fut : fut_rets) {
      GE_CHK_STATUS_RET(fut.get(), "Failed to UpdateLocalProfilingInfo");
    }
  }
  GELOGI("Update local profiling info successfully.");
  return SUCCESS;
}
Status DeployContext::UpdateProfiling(const bool is_prof_start, const std::string &prof_data,
                                      const ExecutorManager::ExecutorKey &key) {
  PneExecutorClient *executor = nullptr;
  GE_CHK_STATUS_RET(executor_manager_.GetExecutorClient(key, &executor),
                    "[Update][profiling] Failed to get executor");

  deployer::ExecutorRequest_UpdateProfRequest prof_message;
  prof_message.set_is_prof_start(is_prof_start);
  prof_message.set_prof_data(prof_data);
  GE_TIMESTAMP_START(UpdateProfilingFromExecutor);
  GE_CHK_STATUS_RET(executor->UpdateProfilingFromExecutor(prof_message),
                    "[Update][Profiling] Failed to update profiling info");
  GE_TIMESTAMP_EVENT_END(UpdateProfilingFromExecutor, "update profiling info");
  GELOGD("[Update][Profiling] succeeded, device_id = %d, device_type = %d, is_prof_start = %d",
         key.device_id, key.device_type, is_prof_start);
  return SUCCESS;
}

void DeployContext::SetOptions(const DeployState &deploy_state,
                               deployer::ExecutorRequest_BatchLoadModelMessage &request) {
  auto options = request.mutable_options();
  for (const auto &item : deploy_state.GetAllGlobalOptions()) {
    options->mutable_global_options()->insert({item.first, item.second});
    GELOGI("Add global option to batch load request, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }

  for (const auto &item : deploy_state.GetAllSessionOptions()) {
    options->mutable_session_options()->insert({item.first, item.second});
    GELOGI("Add session option to batch load request, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }

  for (const auto &item : deploy_state.GetAllGraphOptions()) {
    options->mutable_graph_options()->insert({item.first, item.second});
    GELOGI("Add graph option to batch load request, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }
}

Status DeployContext::DoLoadSubmodels(const DeployState &deploy_state,
                                      const ExecutorManager::ExecutorKey &key) {
  PneExecutorClient *executor = nullptr;
  GE_CHK_STATUS_RET(executor_manager_.GetExecutorClient(key, &executor),
                    "[Download][Model] Failed to get executor");
  // assemble load model
  auto root_model_id = deploy_state.GetRootModelId();
  deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc;
  SetVarMemoryInfo(key.device_id, deploy_state.GetSessionId(), load_model_desc);
  SetOptions(deploy_state, load_model_desc);
  GE_CHK_STATUS_RET_NOLOG(SetModelInfo(deploy_state, key, load_model_desc));
  GE_TIMESTAMP_START(LoadModel);
  GE_CHK_STATUS_RET(executor->LoadModel(load_model_desc),
                    "[Load][Model] Failed to load model");
  GE_TIMESTAMP_EVENT_END(LoadModel, "deploying in loading single submodel");
  GELOGI("[Load][Model] succeeded, device_id = %d, device_type = %d, root_model_id = %u",
         key.device_id, key.device_type, root_model_id);
  std::lock_guard<std::mutex> lk(mu_);
  submodel_devices_[root_model_id].emplace(key);
  return SUCCESS;
}

Status DeployContext::SetDynamicSchedModelInfo(deployer::ExecutorRequest_LoadModelRequest *&model_info,
                                               const deployer::SubmodelDesc &submodel_desc,
                                               const ExchangeRoute *flow_route,
                                               const DeployState &deploy_state) const {
  std::vector<DeployQueueAttr> model_input_queues;
  std::vector<DeployQueueAttr> model_output_queues;

  GE_CHK_STATUS_RET_NOLOG(GetSchedQueues(*flow_route, submodel_desc, model_input_queues, model_output_queues));
  auto *const status_queues_def = model_info->mutable_status_queues();
  SetModelQueuesAttrs(submodel_desc.model_instance_name(), model_input_queues,
                      model_output_queues, *status_queues_def);
  for (size_t i = 0U; i < model_input_queues.size(); i++) {
    GELOGI("DynamicSched, add model info to load request, name=%s, status input indice=%d, phy input id=%d",
           submodel_desc.model_name().c_str(), submodel_desc.status_input_queue_indices()[i],
           model_input_queues[i].queue_id);
  }
  for (size_t i = 0U; i < model_output_queues.size(); i++) {
    GELOGI("DynamicSched, add model info to load request, name=%s, status output indice=%d, phy output id=%d",
           submodel_desc.model_name().c_str(), submodel_desc.status_output_queue_indices()[i],
           model_output_queues[i].queue_id);
  }
  for (auto &indice : submodel_desc.input_queue_indices()) {
    GELOGI("DynamicSched, add model info to load request, name=%s, logic input indice=%d",
           submodel_desc.model_name().c_str(), indice);
  }
  uint32_t model_id = 0U;
  if (model_output_queues.size() == 1U) {
    model_info->set_need_report_status(true);
    model_id = flow_route->GetModelId(submodel_desc.status_output_queue_indices()[0]);
    model_info->set_model_uuid(model_id);
  } else {
    model_info->set_need_report_status(false);
    model_info->set_model_uuid(model_id);
  }
  model_info->set_is_dynamic_sched(deploy_state.GetIsDynamicSched());
  GELOGI("DynamicSched, model [%s], is dynamic sched=%d, need report status=%d, model_uuid=%u",
         submodel_desc.model_name().c_str(), deploy_state.GetIsDynamicSched(),
         model_info->need_report_status(), model_id);
  return SUCCESS;
}

Status DeployContext::SetModelInfo(const DeployState &deploy_state,
                                   const ExecutorManager::ExecutorKey &key,
                                   deployer::ExecutorRequest_BatchLoadModelMessage &request) {
  auto root_model_id = deploy_state.GetRootModelId();
  auto graph_id = deploy_state.GetGraphId();
  auto submodel_descs = deploy_state.GetLocalSubmodels(key);
  GE_CHECK_NOTNULL(submodel_descs);
  request.set_root_model_id(root_model_id);
  request.set_graph_id(graph_id);
  for (const auto &submodel_desc : *submodel_descs) {
    auto submodel_id = submodel_desc.submodel_id();
    std::vector<DeployQueueAttr> model_input_queues;
    std::vector<DeployQueueAttr> model_output_queues;
    auto model_path = name_ + "/" + submodel_desc.model_path();
    auto saved_model_path = submodel_desc.saved_model_file_path();
    if (submodel_desc.is_remote_model() && (!submodel_desc.is_builtin_udf())) {
      saved_model_path = base_dir_ + submodel_desc.saved_model_file_path();
    }

    const bool is_dynamic_proxy_controlled = deploy_state.GetDynamicProxyControlledFlag(submodel_id);
    GELOGI("Add model to load, device_id = %d, root_model_id = %u, sub_model_id = %u, model_path = %s, "
           "is_dynamic = %d, is_dynamic_proxy_controlled = %d, saved_model_path = %s, is_builtin = %ds.",
           key.device_id, root_model_id, submodel_id, model_path.c_str(),
           static_cast<int32_t>(submodel_desc.is_dynamic()), static_cast<int32_t>(is_dynamic_proxy_controlled),
           saved_model_path.c_str(), static_cast<int32_t>(submodel_desc.is_builtin_udf()));
    auto model_info = request.add_models();
    model_info->set_root_model_id(root_model_id);
    model_info->set_model_id(submodel_desc.submodel_id());
    model_info->set_model_path(model_path);
    model_info->set_saved_model_file_path(saved_model_path);
    model_info->set_is_builtin_udf(submodel_desc.is_builtin_udf());
    model_info->set_enable_exception_catch(submodel_desc.enable_exception_catch());
    model_info->set_scope(submodel_desc.scope());
    model_info->set_is_dynamic(submodel_desc.is_dynamic());
    model_info->set_replica_num(submodel_desc.replica_num());
    model_info->set_replica_idx(submodel_desc.replica_idx());
    model_info->set_execute_times(submodel_desc.execute_times());
    model_info->set_phy_device_id(submodel_desc.phy_device_id());
    model_info->set_is_dynamic_proxy_controlled(is_dynamic_proxy_controlled);
    model_info->set_is_head(submodel_desc.is_head());
    model_info->set_model_instance_name(submodel_desc.model_instance_name());

    const ExchangeRoute *flow_route = nullptr;
    GE_CHK_STATUS_RET_NOLOG(GetFlowRoute(deploy_state, flow_route));
    GE_CHK_STATUS_RET_NOLOG(GetQueues(*flow_route, submodel_desc, model_input_queues, model_output_queues));
    auto *const model_queues_attrs_def = model_info->mutable_model_queues_attrs();
    std::vector<uint32_t> input_queue_ids;
    std::vector<uint32_t> output_queue_ids;
    SetModelQueuesAttrs(submodel_desc.model_instance_name(), model_input_queues,
                        model_output_queues, *model_queues_attrs_def);
    GetQueueIdsFromAttrs(model_input_queues, input_queue_ids);
    GetQueueIdsFromAttrs(model_output_queues, output_queue_ids);
    GE_CHK_STATUS_RET_NOLOG(SetDynamicSchedModelInfo(model_info, submodel_desc, flow_route, deploy_state));
    std::vector<int32_t> fusion_offsets;
    GE_CHK_STATUS_RET_NOLOG(GetInputFusionOffsets(*flow_route, submodel_desc, fusion_offsets));
    model_info->mutable_input_fusion_offsets()->Add(fusion_offsets.begin(), fusion_offsets.end());
    auto attrs = model_info->mutable_attrs();
    *attrs = submodel_desc.attrs();
    if (submodel_desc.has_input_align_attrs()) {
      auto input_align_attrs = model_info->mutable_input_align_attrs();
      *input_align_attrs = submodel_desc.input_align_attrs();
    }

    const auto &invoked_model_queues_indices = submodel_desc.invoked_model_queues();
    if (!invoked_model_queues_indices.empty()) {
      auto invoked_model_queues = model_info->mutable_invoked_model_queues_attrs();
      for (const auto &queue_indices : invoked_model_queues_indices) {
        std::vector<DeployQueueAttr> input_queues;
        std::vector<DeployQueueAttr> output_queues;
        std::vector<int32_t> input_queue_indices(queue_indices.second.input_queue_indices().begin(),
                                                 queue_indices.second.input_queue_indices().end());
        GE_CHK_STATUS_RET(flow_route->GetQueueAttrs(input_queue_indices, input_queues),
                          "Failed to get input queue ids, invoked model key=%s", queue_indices.first.c_str());
        std::vector<int32_t> output_queue_indices(queue_indices.second.output_queue_indices().begin(),
                                                  queue_indices.second.output_queue_indices().end());
        GE_CHK_STATUS_RET(flow_route->GetQueueAttrs(output_queue_indices, output_queues),
                          "Failed to get output queue ids, invoked model key=%s", queue_indices.first.c_str());
        deployer::ExecutorRequest_ModelQueuesAttrs invoked_model_queue_attrs;
        SetModelQueuesAttrs(submodel_desc.model_instance_name(), input_queues, output_queues,
                            invoked_model_queue_attrs, true);
        (*invoked_model_queues)[queue_indices.first] = std::move(invoked_model_queue_attrs);
      }
    }
    GEEVENT("[IO info] Add model info to load, graph_id = %u, model_name = %s, model_type = %s, "
            "input_queues = %s, output_queues = %s, device_id = %d, device_type = %d, "
            "root_model_id = %u, sub_model_id = %u, model_path = %s, model_instance_name = %s.",
            graph_id, submodel_desc.model_name().c_str(), submodel_desc.engine_name().c_str(),
            ToString(input_queue_ids).c_str(), ToString(output_queue_ids).c_str(), key.device_id, key.device_type,
            root_model_id, submodel_id, model_path.c_str(), submodel_desc.model_instance_name().c_str());
  }
  return SUCCESS;
}

void DeployContext::SetModelQueuesAttrs(const std::string &model_name,
                                        const std::vector<DeployQueueAttr> &model_input_queues,
                                        const std::vector<DeployQueueAttr> &model_output_queues,
                                        deployer::ExecutorRequest_ModelQueuesAttrs &model_queues_attrs_def,
                                        bool is_invoked) const {
  for (size_t i = 0UL; i < model_input_queues.size(); ++i) {
    const auto &input_queue = model_input_queues[i];
    auto *const input_queue_def =  model_queues_attrs_def.mutable_input_queues_attrs()->Add();
    input_queue_def->set_queue_id(input_queue.queue_id);
    input_queue_def->set_device_type(input_queue.device_type);
    input_queue_def->set_device_id(input_queue.device_id);
    input_queue_def->set_global_logic_id(input_queue.global_logic_id);
    GEEVENT("[IO info] model info = [name:%s], input info = [index:%zu], queue info = [%s], is_invoked = [%d].",
            model_name.c_str(), i, input_queue.DebugString().c_str(), is_invoked);
  }
  for (size_t i = 0UL; i < model_output_queues.size(); ++i) {
    const auto &output_queue = model_output_queues[i];
    auto *const output_queue_def =  model_queues_attrs_def.mutable_output_queues_attrs()->Add();
    output_queue_def->set_queue_id(output_queue.queue_id);
    output_queue_def->set_device_type(output_queue.device_type);
    output_queue_def->set_device_id(output_queue.device_id);
    output_queue_def->set_global_logic_id(output_queue.global_logic_id);
    GEEVENT("[IO info] model info = [name:%s], output info = [index:%zu], queue info = [%s], is_invoked = [%d].",
            model_name.c_str(), i, output_queue.DebugString().c_str(), is_invoked);
  }
}

void DeployContext::SetVarMemoryInfo(int32_t device_id,
                                     uint64_t session_id,
                                     deployer::ExecutorRequest_BatchLoadModelMessage &request) {
  DeployerVarManager *var_manager = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto &var_managers = var_managers_[device_id];
    const auto &it = var_managers.find(session_id);
    if (it == var_managers.cend()) {
      GELOGI("No need for synchronizing VarManager, session_id = %lu", session_id);
      return;
    }
    var_manager = it->second.get();
  }
  if (!var_manager->IsShareVarMem()) {
    *request.mutable_var_manager_info() = var_manager->GetVarManagerInfo();
  }
}

Status DeployContext::GetQueues(const ExchangeRoute &flow_route,
                                const deployer::SubmodelDesc &submodel_desc,
                                std::vector<DeployQueueAttr> &input_queues,
                                std::vector<DeployQueueAttr> &output_queues) {
  std::vector<int32_t>
      input_queue_indices(submodel_desc.input_queue_indices().begin(), submodel_desc.input_queue_indices().end());
  GE_CHK_STATUS_RET(flow_route.GetQueueAttrs(input_queue_indices, input_queues),
                    "Failed to get input queue ids");
  std::vector<int32_t>
      output_queue_indices(submodel_desc.output_queue_indices().begin(), submodel_desc.output_queue_indices().end());
  GE_CHK_STATUS_RET(flow_route.GetQueueAttrs(output_queue_indices, output_queues),
                    "Failed to get output queue ids");
  return SUCCESS;
}

Status DeployContext::GetQueues(const ExchangeRoute &flow_route,
                                std::vector<int32_t> &input_indices,
                                std::vector<int32_t> &output_indices,
                                std::vector<uint32_t> &input_queues,
                                std::vector<uint32_t> &output_queues) {
  std::vector<int32_t>
      input_queue_indices(input_indices.begin(), input_indices.end());
  GE_CHK_STATUS_RET(flow_route.GetQueueIds(input_queue_indices, input_queues),
                    "Failed to get input queue ids");
  std::vector<int32_t>
      output_queue_indices(output_indices.begin(), output_indices.end());
  GE_CHK_STATUS_RET(flow_route.GetQueueIds(output_queue_indices, output_queues),
                    "Failed to get output queue ids");
  return SUCCESS;
}

Status DeployContext::GetSchedQueues(const ExchangeRoute &flow_route,
                                     const deployer::SubmodelDesc &submodel_desc,
                                     std::vector<DeployQueueAttr> &input_queues,
                                     std::vector<DeployQueueAttr> &output_queues) {
  std::vector<int32_t>
      input_queue_indices(submodel_desc.status_input_queue_indices().begin(),
                          submodel_desc.status_input_queue_indices().end());
  GE_CHK_STATUS_RET(flow_route.GetQueueAttrs(input_queue_indices, input_queues),
                    "Failed to get status input queue ids");
  std::vector<int32_t>
      output_queue_indices(submodel_desc.status_output_queue_indices().begin(),
                           submodel_desc.status_output_queue_indices().end());
  GE_CHK_STATUS_RET(flow_route.GetQueueAttrs(output_queue_indices, output_queues),
                    "Failed to get status output queue ids");
  return SUCCESS;
}

Status DeployContext::GetInputFusionOffsets(const ExchangeRoute &flow_route,
                                            const deployer::SubmodelDesc &submodel_desc,
                                            std::vector<int32_t> &fusion_offsets) {
  std::vector<int32_t>
      input_queue_indices(submodel_desc.input_queue_indices().begin(), submodel_desc.input_queue_indices().end());
  GE_CHK_STATUS_RET(flow_route.GetFusionOffsets(input_queue_indices, fusion_offsets),
                    "Failed to get input queue ids");
  return SUCCESS;
}

Status DeployContext::UnloadSubmodelsFromExecutor(const ExecutorManager::ExecutorKey &executor_key,
                                                  uint32_t root_model_id) {
  auto device_id = executor_key.device_id;
  GEEVENT("[Unload][Model] start, device_id = %d, model_id = %u", device_id, root_model_id);
  PneExecutorClient *executor_client = nullptr;
  GE_CHK_STATUS_RET(executor_manager_.GetExecutorClient(executor_key, &executor_client),
                    "[Unload][Model] Failed to get executor, model_id = %u", root_model_id);
  GE_CHK_STATUS_RET(executor_client->UnloadModel(root_model_id),
                    "Failed to unload model, model_id = %u", root_model_id);
  GEEVENT("[Unload][Model] succeeded, device_id = %d, model_id = %u.", device_id, root_model_id);
  return SUCCESS;
}

Status DeployContext::UnloadSubmodels(uint32_t root_model_id) {
  Status status = SUCCESS;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = submodel_devices_.find(root_model_id);
  if (it == submodel_devices_.end()) {
    GELOGI("Submodel not deployed, root_model_id = %u", root_model_id);
  } else {
    auto parallel_num = it->second.size();
    GELOGI("Unload submodel in parallel, num = %zu", parallel_num);
    ThreadPool pool("ge_dpl_unld", static_cast<uint32_t>(parallel_num), false);
    std::vector<std::future<Status>> fut_rets;
    for (auto &executor_key : it->second) {
      auto fut = pool.commit([this, &executor_key, root_model_id]() -> Status {
        GE_CHK_RT_RET(rtSetDevice(executor_key.device_id));
        GE_CHK_STATUS_RET(UnloadSubmodelsFromExecutor(executor_key, root_model_id), "Failed to unload submodel.");
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    for (auto &fut : fut_rets) {
      auto ret = fut.get();
      if (ret != SUCCESS) {
        status = ret;
      }
    }
    (void) submodel_devices_.erase(it);
  }

  auto route_it = submodel_routes_.find(root_model_id);
  if (route_it != submodel_routes_.end()) {
    (void)FlowRouteManager::GetInstance().UndeployRoute(FlowRouteType::kModelFlowRoute,
                                                        route_it->second,
                                                        flowgw_client_manager_);
    GELOGI("flow route undeployed, model_id = %u, route_id = %ld", root_model_id, route_it->second);
    (void) submodel_routes_.erase(route_it);
  }
  return status;
}

bool DeployContext::CheckExecutorKeyIsException(const deployer::ClearModelDataRequest &req_body,
                                                const ExecutorManager::ExecutorKey &executor_key) const {
  const auto device_infos = req_body.exception_dev_info();
  const auto node_id = req_body.node_id();
  for (auto &device_info : device_infos) {
    if ((node_id == device_info.node_id())
        && (executor_key.device_id == device_info.device_id())
        && (executor_key.device_type == device_info.device_type())) {
      return true;
    }
  }
  return false;
}

void DeployContext::GetModelClearInfo(const deployer::ClearModelDataRequest &req_body,
                                      std::map<uint32_t, std::set<ExecutorManager::ExecutorKey>> &models_info,
                                      std::set<uint32_t> &model_ids) {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto &root_model_id : req_body.root_model_ids()) {
    auto it = submodel_devices_.find(root_model_id);
    if (it == submodel_devices_.end()) {
      GELOGI("Submodel not deployed, root_model_id = %u", root_model_id);
      continue;
    }
    model_ids.emplace(root_model_id);
    for (auto &executor_key : it->second) {
      if (CheckExecutorKeyIsException(req_body, executor_key)) {
        continue;
      }
      models_info[root_model_id].insert(executor_key);
    }
  }
}

void DeployContext::GetExceptionDevices(const deployer::ClearModelDataRequest &req_body,
                                        std::vector<FlowGwClient::ExceptionDeviceInfo> &devices) {
  const auto device_infos = req_body.exception_dev_info();
  for (auto &device_info : device_infos) {
    FlowGwClient::ExceptionDeviceInfo exception_device;
    exception_device.device_id = device_info.device_id();
    exception_device.device_type = device_info.device_type();
    exception_device.node_id = device_info.node_id();
    devices.emplace_back(std::move(exception_device));
    GELOGI("Exception device info: node_id = %d, device_id = %d, device_type = %d",
        exception_device.node_id, exception_device.device_id, exception_device.device_type);
  }
}

Status DeployContext::SyncSubmitClearModelTasks(std::map<uint32_t,
                                                std::set<ExecutorManager::ExecutorKey>> &models_info,
                                                uint32_t parallel_num, int32_t type,
                                                const std::set<int32_t> &device_ids) {
  std::vector<std::future<Status>> model_fut_rets;
  ThreadPool pool("ge_dpl_clmd", parallel_num, false);
  for (const auto &it : models_info) {
    const auto &model_id = it.first;
    for (const auto &client_key : it.second) {
      PneExecutorClient *executor_client = nullptr;
      GELOGI("Clear model data, root_model_id = %u, msg_type = %d", model_id, type);
      GE_CHK_STATUS_RET(executor_manager_.GetExecutorClient(client_key, &executor_client),
                        "[ClearModelExceptionData] Failed to get executor, model_id = %u", model_id);
      auto fut = pool.commit([this, &model_id, executor_client, type, &device_ids]() -> Status {
        GE_CHK_STATUS_RET_NOLOG(executor_client->ClearModelRunningData(model_id, type, device_ids));
        return SUCCESS;
      });
      model_fut_rets.emplace_back(std::move(fut));
    }
  }

  for (auto &fut : model_fut_rets) {
    if (fut.get() != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status DeployContext::SyncSubmitClearFlowgwTasks(const std::set<uint32_t> &model_ids, int32_t type) {
  const auto &flowgw_clients = flowgw_client_manager_.GetClients();
  ThreadPool pool("ge_dpl_clfg", flowgw_clients.size(), false);
  std::vector<std::future<Status>> flow_gw_fut_rets;
  for (const auto &client : flowgw_clients) {
    auto fut = pool.commit([this, &model_ids, &client, type]() -> Status {
      GE_CHK_STATUS_RET_NOLOG(client->ClearFlowgwModelData(model_ids, type));
      return SUCCESS;
    });
    flow_gw_fut_rets.emplace_back(std::move(fut));
  }

  for (auto &fut : flow_gw_fut_rets) {
    if (fut.get() != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status DeployContext::SyncUpdateExceptionRoutes(const std::set<uint32_t> &model_ids,
                                                const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices) {
  std::lock_guard<std::mutex> lk(mu_);
  for (const auto &model_id : model_ids) {
    auto route_it = submodel_routes_.find(model_id);
    if (route_it == submodel_routes_.end()) {
      continue;
    }
    auto &route_id = route_it->second;
    GELOGI("[UpdateExceptionRoutes] model_id = %u, route_id = %ld", model_id, route_id);
    FlowRouteManager::GetInstance().UpdateExceptionRoutes(FlowRouteType::kModelFlowRoute,
                                                          route_id,
                                                          flowgw_client_manager_,
                                                          devices);
  }
  return SUCCESS;
}

void DeployContext::GetExceptionDevId(const deployer::ClearModelDataRequest &req_body,
                                      std::set<int32_t> &device_ids) {
  const auto device_infos = req_body.exception_dev_info();
  const auto node_id = req_body.node_id();
  for (const auto &device_info : device_infos) {
    if (node_id == device_info.node_id()) {
      device_ids.emplace(device_info.device_id());
    }
  }
}

Status DeployContext::ClearModelRunningData(const deployer::ClearModelDataRequest &req_body) {
  std::map<uint32_t, std::set<ExecutorManager::ExecutorKey>> clear_submodel_devices;
  int32_t type = req_body.clear_type();
  uint32_t model_clear_num = 0U;
  std::vector<FlowGwClient::ExceptionDeviceInfo> exception_devices;
  std::set<uint32_t> model_ids;

  GetModelClearInfo(req_body, clear_submodel_devices, model_ids);
  GetExceptionDevices(req_body, exception_devices);
  for (const auto &it : clear_submodel_devices) {
      model_clear_num += it.second.size();
  }
  if ((model_ids.size() > 0U) && (exception_devices.size() > 0U) && (type == EXCEPTION_HANDLE_CLEAR)) {
    GE_CHK_STATUS_RET_NOLOG(SyncUpdateExceptionRoutes(model_ids, exception_devices));
  }
  if (model_clear_num > 0U) {
    std::set<int32_t> device_ids;
    GetExceptionDevId(req_body, device_ids);
    GE_CHK_STATUS_RET_NOLOG(SyncSubmitClearModelTasks(clear_submodel_devices, model_clear_num, type, device_ids));
  }
  if (model_ids.size() > 0U) {
    GE_CHK_STATUS_RET_NOLOG(SyncSubmitClearFlowgwTasks(model_ids, type));
  }

  return SUCCESS;
}

Status DeployContext::DataFlowExceptionNotifyProcess(const deployer::DataFlowExceptionNotifyRequest &req_body) {
  uint32_t root_model_id = req_body.root_model_id();
  std::set<ExecutorManager::ExecutorKey> executor_keys;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = submodel_devices_.find(root_model_id);
    if (it == submodel_devices_.end()) {
      GELOGI("Submodel is not deployed, root_model_id = %u", root_model_id);
      return SUCCESS;
    }
    executor_keys = it->second;
  }
  std::vector<std::future<Status>> exception_notify_fut_rets;
  exception_notify_fut_rets.reserve(executor_keys.size());
  ThreadPool pool("ge_dpl_exnn", executor_keys.size(), false);
  for (auto &executor_key : executor_keys) {
    PneExecutorClient *executor_client = nullptr;
    GE_CHK_STATUS_RET(executor_manager_.GetExecutorClient(executor_key, &executor_client),
                      "[ExceptionNotify] Failed to get executor, model_id = %u, context_id = %ld, device id = %d, "
                      "engine name = %s, process_id = %d",
                      root_model_id, executor_key.context_id, executor_key.device_id, executor_key.engine_name.c_str(),
                      executor_key.process_id);

    auto fut = pool.commit([&req_body, executor_client]() -> Status {
      GE_CHK_STATUS_RET_NOLOG(executor_client->DataFlowExceptionNotify(req_body));
      return SUCCESS;
    });
    exception_notify_fut_rets.emplace_back(std::move(fut));
  }

  for (auto &fut : exception_notify_fut_rets) {
    if (fut.get() != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status DeployContext::UpdateProfilingInfoProcess(const deployer::SendProfInfoRequest &req_body) {
  bool is_prof_start = req_body.is_prof_start();
  const std::string &prof_data = req_body.prof_data();
  uint32_t model_id = req_body.model_id();
  return UpdateLocalProfilingInfo(is_prof_start, prof_data, model_id);
}

Status DeployContext::PreDeployLocalFlowRoute(uint32_t root_model_id) {
  DeployState *deploy_state = nullptr;
  GE_CHK_STATUS_RET_NOLOG(flow_model_receiver_.GetDeployState(root_model_id, deploy_state));
  const auto &local_flow_route_plan = deploy_state->GetLocalFlowRoutePlan();
  GELOGI("Start to deploy flow route on node: %d", local_flow_route_plan.first);
  auto exchange_deployer = MakeUnique<HeterogeneousExchangeDeployer>(
      HService::GetInstance(), local_flow_route_plan.second, flowgw_client_manager_);
  GE_CHECK_NOTNULL(exchange_deployer);
  GE_CHK_STATUS_RET(exchange_deployer->PreDeploy());
  deploy_state->node_exchange_deployer_pair_ = std::make_pair(local_flow_route_plan.first,
                                                              std::move(exchange_deployer));
  return SUCCESS;
}

Status DeployContext::DeployLocalFlowRoute(uint32_t root_model_id) {
  DeployState *deploy_state = nullptr;
  GE_CHK_STATUS_RET_NOLOG(flow_model_receiver_.GetDeployState(root_model_id, deploy_state));
  if (deploy_state->GetIsDynamicSched()) {
    for (auto iter : deploy_state->GetDataGwSchedInfos()) {
      GE_CHK_STATUS_RET(DataGwSchedInfo(*deploy_state, iter.second),
                        "Failed to deploy dynamic sched, model_id = %u", root_model_id);
    }
  }

  // auto release resources on failure
  auto node_exchange_deployer_pair = std::move(deploy_state->node_exchange_deployer_pair_);
  auto node_id = node_exchange_deployer_pair.first;
  auto &flow_route_deployer = node_exchange_deployer_pair.second;
  ExchangeRoute flow_route;
  GE_CHK_STATUS_RET(flow_route_deployer->Deploy(flow_route, true),
                    "Failed to deploy flow route, node_id = %d, model_id = %u",
                    node_id, root_model_id);
  int64_t route_id = -1;
  FlowRouteManager::GetInstance().AddRoute(flow_route, FlowRouteType::kModelFlowRoute, route_id);
  {
    std::lock_guard<std::mutex> lk(mu_);
    submodel_routes_[root_model_id] = route_id;
  }
  deploy_state->deployed_node_ids_.emplace(node_id);  // ensure local route can be released
  return SUCCESS;
}

const ExchangeRoute *DeployContext::QueryFlowRoute(uint32_t root_model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto route_id = submodel_routes_[root_model_id];
  return FlowRouteManager::GetInstance().QueryRoute(FlowRouteType::kModelFlowRoute, route_id);
}

const std::string &DeployContext::GetBaseDir() const {
  return base_dir_;
}

void DeployContext::Clear() const {
  std::string clear_dir_cmd = "rm -rf -- ";
  clear_dir_cmd += base_dir_;
  GELOGI("Clear context: %s, execute: %s", base_dir_.c_str(), clear_dir_cmd.c_str());
  (void) ProcessUtils::System(clear_dir_cmd.c_str());
}

Status DeployContext::InitProcessResource(const deployer::InitProcessResourceRequest &request,
                                          deployer::DeployerResponse &response) {
  GE_DISMISSABLE_GUARD(deploy_flowgw, ([&response]() {
                         response.set_error_code(FAILED);
                         response.set_error_message("Failed to deploy rank table");
                       }));
  GE_CHK_STATUS(MemoryGroupManager::GetInstance().SetRemoteGroupCacheConfig(request.remote_group_cache_alloc_config()),
                "[Init][ProcessResource] Failed to set remote cache config, config = %s.",
                request.remote_group_cache_alloc_config().c_str());
  const bool profiling_on = request.profiling_on();
  const int32_t device_id = request.device_id();
  const int32_t device_type = request.device_type();
  std::vector<int32_t> res_ids(request.res_ids().begin(), request.res_ids().end());
  bool is_proxy = static_cast<int32_t>(NPU) == device_type;
  auto client = flowgw_client_manager_.GetOrCreateClient(device_id, device_type, res_ids, is_proxy);
  GE_CHECK_NOTNULL(client);
  const auto &rank_table = request.rank_table();
  if (!rank_table.empty()) {
    GELOGI("[Init][ProcessResource] Deploy kFlowGwInit rank table start, rank table = %s", rank_table.c_str());
    client->SetHcomInfo(rank_table, request.rank_id());
  }
  if (profiling_on) {
    GE_CHK_STATUS(client->UpdateProfiling(), "[Init][ProcessResource] Failed to update profiling.");
  }
  GE_DISMISS_GUARD(deploy_flowgw);
  response.set_error_code(SUCCESS);
  return SUCCESS;
}

Status DeployContext::SyncVarManagers(PneExecutorClient &executor_client, const DeployState &deploy_state) {
  auto device_id = executor_client.GetDeviceId();
  DeployerVarManager *var_manager = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto &var_managers = var_managers_[device_id];
    const auto &it = var_managers.find(deploy_state.GetSessionId());
    if (it == var_managers.cend()) {
      GELOGI("No need for synchronizing VarManager, session_id = %lu", deploy_state.GetSessionId());
      return SUCCESS;
    }
    var_manager = it->second.get();
  }

  if (!var_manager->IsShareVarMem()) {
    GELOGI("Var memory is not shared");
    return SUCCESS;
  }

  const auto &client_context = executor_client.GetContext();
  if (static_cast<DeviceType>(client_context.device_type) == CPU) {
    deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_request;
    GE_CHK_STATUS_RET(executor_client.SyncVarManager(std::move(sync_var_manage_request)),
                      "Failed to sync var manager, device_id = %d, session_id = %lu",
                      device_id, deploy_state.GetSessionId());
    return SUCCESS;
  }

  if (!executor_client.SupportSyncVarManager()) {
    GELOGI("No need to sync var manager.");
    return SUCCESS;
  }

  deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_request;
  auto var_manager_info = var_manager->GetVarManagerInfo();
  *sync_var_manage_request.mutable_var_manager_info() = std::move(var_manager_info);
  for (const auto &name_and_shared_content_desc : var_manager->GetSharedContentDescs()) {
    *sync_var_manage_request.add_shared_content_descs() = name_and_shared_content_desc.second;
  }
  GE_CHK_STATUS_RET(executor_client.SyncVarManager(std::move(sync_var_manage_request)),
                    "Failed to sync var manager, device_id = %d, session_id = %lu",
                    device_id, deploy_state.GetSessionId());
  GELOGD("[Sync][VarManager] succeeded, device_id = %d, session_id = %lu",
      device_id, deploy_state.GetSessionId());
  return SUCCESS;
}

void DeployContext::DestroyDeployState(uint32_t model_id) {
  flow_model_receiver_.DestroyDeployState(model_id);
}

DeployerVarManager *DeployContext::GetVarManager(int32_t device_id, uint64_t session_id) {
  std::lock_guard<std::mutex> lk(mu_);
  return var_managers_[device_id][session_id].get();
}

RankTableBuilder &DeployContext::GetRankTableBuilder() {
  return rank_table_builder_;
}

FlowGwClientManager &DeployContext::GetFlowGwClientManager() {
  return flowgw_client_manager_;
}

std::mutex &DeployContext::GetAbnormalHeartbeatInfoMu() {
  return abnormal_heartbeat_info_mu_;
}

void DeployContext::AddAbnormalSubmodelInstanceName(uint32_t root_model_id, const std::string model_instance_name) {
  abnormal_submodel_instances_name_[root_model_id].emplace(model_instance_name, false);
  return;
}

RootModelId2SubmodelName &DeployContext::GetAbnormalSubmodelInstanceName() {
  return abnormal_submodel_instances_name_;
}

void DeployContext::ClearAbnormalSubmodelInstanceName() {
  abnormal_submodel_instances_name_.clear();
  return;
}

void DeployContext::AddAbnormalNodeConfig(NodeConfig node_config) {
  abnormal_node_config_.emplace(node_config, false);
  return;
}

const std::map<NodeConfig, bool> &DeployContext::GetAbnormalNodeConfig() {
  return abnormal_node_config_;
}

void DeployContext::ClearAbnormalNodeConfig() {
  abnormal_node_config_.clear();
  return;
}

void DeployContext::AddAbnormalDeviceInfo(DeployPlan::DeviceInfo device_info) {
  abnormal_device_info_.emplace(device_info, false);
  return;
}

const std::map<DeployPlan::DeviceInfo, bool> &DeployContext::GetAbnormalDeviceInfo() {
  return abnormal_device_info_;
}

void DeployContext::ClearAbnormalDeviceInfo() {
  abnormal_device_info_.clear();
  return;
}

Status DeployContext::DataGwSchedInfo(const DeployState &deploy_state, const deployer::DataGwSchedInfos &req_body) {
  const uint32_t dev_id = req_body.device_id();
  const int32_t dev_type = req_body.device_type();
  const int32_t input_queue_indice = req_body.input_queue_indice();
  const int32_t output_queue_indice = req_body.output_queue_indice();
  const uint32_t root_model_id = req_body.root_model_id();
  const bool is_proxy = req_body.is_proxy();
  std::vector<uint32_t> input_queues;
  std::vector<uint32_t> output_queues;
  std::vector<int32_t> input_indices{input_queue_indice};
  std::vector<int32_t> output_indices{output_queue_indice};

  const ExchangeRoute *flow_route = nullptr;
  GE_CHK_STATUS_RET(GetFlowRoute(deploy_state, flow_route), "DynamicSched GetFlowRoute failed.");
  GE_CHK_STATUS_RET(GetQueues(*flow_route, input_indices, output_indices, input_queues, output_queues),
                    "DynamicSched GetQueues failed.");

  GELOGI("DynamicSched config sched info to datagw, device_id=%d, device_type=%d, logic input indice=%d, "
         "phy input indice=%u, phy output indice=%u, logic output indice=%d, is_proxy=%d", dev_id, dev_type,
         input_queue_indice, input_queues[0], output_queues[0], output_queue_indice, is_proxy);
  return flowgw_client_manager_.ConfigSchedInfoToDataGw(dev_id, dev_type, is_proxy,
                                                        input_queue_indice, input_queues[0],
                                                        output_queues[0], root_model_id);
}
}  // namespace ge
