/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/master_model_deployer.h"
#include <algorithm>
#include <future>
#include "framework/common/util.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/thread_pool/thread_pool.h"
#include "deploy/deployer/heterogeneous_model_deployer.h"
#include "deploy/flowrm/network_manager.h"
#include "deploy/flowrm/heterogeneous_exchange_deployer.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/resource/heterogeneous_deploy_planner.h"
#include "deploy/resource/deployer_port_distributor.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "securec.h"
#include "common/config/configurations.h"
#include "deploy/model_send/flow_model_sender.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "deploy/deployer/deployer_authentication.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"
#include "common/utils/memory_statistic_manager.h"
#include "common/profiling/profiling_properties.h"

namespace ge {
namespace {
constexpr const char_t *const kIsVarInitGraph = "1";
constexpr const char_t *const kProfilingOn = "1";
constexpr const char *ATTR_NAME_DATA_FLOW_DYNAMIC_SCHEDULE_CFG = "dynamic_schedule_enable";
constexpr const char *ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH = "_enable_exception_catch";
constexpr const char *ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE = "_contains_n-mapping_node";
constexpr const char *ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM = "_inputs_align_max_cache_num";
constexpr const char *ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT = "_inputs_align_timeout";
constexpr const char *ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT = "_inputs_align_dropout";

void DfsFind(const std::string &node, const std::map<std::string, std::vector<std::string>> &org_graph,
             std::unordered_set<std::string> &visited, std::unordered_set<std::string> &tree) {
  (void)visited.insert(node);
  (void)tree.insert(node);
  const auto iter = org_graph.find(node);
  if (iter != org_graph.cend()) {
    for (const auto &neighbor : iter->second) {
      if (visited.count(neighbor) == 0UL) {
        DfsFind(neighbor, org_graph, visited, tree);
      }
    }
  }
}

Status GetModelInputAlignAttrs(const ComputeGraphPtr &root_graph, InputAlignAttrs &input_align_attrs) {
  GE_CHECK_NOTNULL(root_graph);
  int64_t cache_num = 0;
  if (AttrUtils::GetInt(root_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM, cache_num)) {
    // max cache num is 1024.
    GE_CHK_BOOL_RET_STATUS((cache_num >= 0) && (cache_num <= 1024), PARAM_INVALID,
                           "attr[%s]=%ld is out of range [0, 1024]", ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM,
                           cache_num);
  } else {
    GELOGI("no align attrs configured, graph=%s.", root_graph->GetName().c_str());
    return SUCCESS;
  }
  int64_t timeout = 0;
  if (AttrUtils::GetInt(root_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, timeout)) {
    // -1 means no time out, max value is 600 * 1000ms
    GE_CHK_BOOL_RET_STATUS((timeout == (-1)) || ((timeout > 0) && (timeout <= 600 * 1000)), PARAM_INVALID,
                           "attr[%s]=%ld is invalid, must be -1 or in range(0, 600 * 1000]",
                           ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_TIMEOUT, timeout);
  } else {
    GELOGE(PARAM_INVALID, "attr[%s] is not configured, graph=%s.", ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_MAX_CACHE_NUM,
           root_graph->GetName().c_str());
    return PARAM_INVALID;
  }
  bool drop_when_not_aligned = false;
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DATA_FLOW_INPUTS_ALIGN_DROPOUT, drop_when_not_aligned);
  GELOGI("graph[%s] align attrs, align_max_cache_num=%ld, align_timeout=%ld, drop_when_not_align=%d.",
         root_graph->GetName().c_str(), cache_num, timeout, static_cast<int32_t>(drop_when_not_aligned));
  input_align_attrs.align_max_cache_num = static_cast<int32_t>(cache_num);
  input_align_attrs.align_timeout = static_cast<int32_t>(timeout);
  input_align_attrs.drop_when_not_align = drop_when_not_aligned;
  return SUCCESS;
}
}  // namespace

void MasterModelDeployer::AbnormalStatusMonitorInitialize() {
  abnormal_status_handler_.Initialize();
}

void MasterModelDeployer::AbnormalStatusMonitorFinalize() {
  abnormal_status_handler_.Finalize();
}

void MasterModelDeployer::DelCallback(const uint32_t root_model_id) {
  abnormal_status_handler_.DelCallback(root_model_id);
}

void MasterModelDeployer::IncDeployingRootModelNum() {
  abnormal_status_handler_.IncDeployingRootModelNum();
}

void MasterModelDeployer::DecreaseDeployingRootModelNum() {
  abnormal_status_handler_.DecreaseDeployingRootModelNum();
}

DeployPlan::AbnormalStatusCallbackInfo* MasterModelDeployer::GetAbnormalStatusCallbackInfo() {
  return abnormal_status_handler_.GetAbnormalStatusCallbackInfo();
}

void MasterModelDeployer::SetDynamicSchedFlag(bool flag) {
  abnormal_status_handler_.SetDynamicSchedFlag(flag);
}

void MasterModelDeployer::AddDeployedModelInfo(uint32_t root_model_id) {
  auto deployed_model = deployed_models_[root_model_id];
  abnormal_status_handler_.AddDeployedModelInfo(deployed_model.model_id,
                                                deployed_model.model_deploy_infos,
                                                deployed_model.deployed_remote_nodes);
}

void MasterModelDeployer::DelDeployedModelInfo(uint32_t root_model_id) {
  abnormal_status_handler_.DelDeployedModelInfo(root_model_id);
}

MasterModelDeployer::MasterModelDeployer() : mu_(), abnormal_status_handler_(mu_) {}

Status MasterModelDeployer::Initialize(const std::map<std::string, std::string> &options) {
  (void) options;
  GE_CHK_STATUS_RET(NetworkManager::GetInstance().Initialize(), "Failed to init NetworkManager");
  GE_CHK_STATUS_RET(DeviceMaintenanceMasterCfg::InitGlobalMaintenanceConfigs());
  const auto &node = Configurations::GetInstance().GetLocalNode();
  GE_CHK_STATUS_RET(DeployerAuthentication::GetInstance().Initialize(node.auth_lib_path, true),
                    "Failed to deployer auth.");
  GE_CHK_STATUS_RET(DeployerProxy::GetInstance().Initialize(Configurations::GetInstance().GetAllNodeConfigs()),
                    "Failed to initialize device proxy.");
  GE_CHK_STATUS_RET(ResourceManager::GetInstance().Initialize(), "Failed to initialize ResourceManager");
  AbnormalStatusMonitorInitialize();
  DeviceAbnormalStatusHandler::Instance().Initialize();
  MemoryStatisticManager::Instance().Initialize(MemoryGroupManager::GetInstance().GetQsMemGroupName());
  return SUCCESS;
}

Status MasterModelDeployer::PrepareFlowGwInfos(const std::string &rank_table_str,
                                               std::map<std::string, DeployFlowGwInfo> &deploy_flowgw_infos) const {
  const auto &devices = ResourceManager::GetInstance().GetDeviceInfoList();
  std::string profiling_on_str = "0";
  (void)GetContext().GetOption(OPTION_EXEC_PROFILING_MODE, profiling_on_str);
  bool profiling_on = (profiling_on_str == kProfilingOn);
  auto &rank_builder = DeployContext::LocalContext().GetRankTableBuilder();
  for (const auto &device_info : devices) {
    GELOGI("Get device rank info, device[%s]", device_info.DebugString().c_str());
    if (!device_info.SupportFlowgw()) {
      continue;
    }
    auto key = std::to_string(device_info.GetNodeId()) + "_" +
               device_info.GetDeviceIp() + "_" +
               std::to_string(device_info.GetHcomDeviceId());
    const auto &it = deploy_flowgw_infos.find(key);
    if (it == deploy_flowgw_infos.cend()) {
      DeployFlowGwInfo flowgw_info = {};
      flowgw_info.profiling_on = profiling_on;
      if (device_info.SupportHcom()) {
        int32_t rank_id = 0;
        GE_CHK_BOOL_RET_STATUS(rank_builder.GetRankIdByDeviceId(device_info.GetDeviceIp(),
                                                                device_info.GetHcomDeviceId(),
                                                                rank_id),
                               FAILED,
                               "Failed to get rank id, device info[%s]", device_info.DebugString().c_str());
        flowgw_info.rank_table = rank_table_str;
        flowgw_info.rank_id = rank_id;
        GELOGI("Get device rank info success, device[%s], rank_id = %d", device_info.DebugString().c_str(), rank_id);
      }
      flowgw_info.device_info = &device_info;
      deploy_flowgw_infos[key] = flowgw_info;
    }
    auto &flowgw_info = deploy_flowgw_infos[key];
    flowgw_info.res_ids.emplace_back(device_info.GetDeviceId());
  }
  return SUCCESS;
}

Status MasterModelDeployer::InitFlowGwInfo() {
  std::lock_guard<std::mutex> lk(mu_for_init_);
  if (flow_gw_inited_) {
    GELOGI("Init flow gw success, flow_gw_inited = %d", flow_gw_inited_);
    return SUCCESS;
  }
  HcomRankTable rank_table = {};
  GE_CHK_STATUS_RET(CreateRankTable(rank_table), "Failed to create rank table");

  auto &rank_builder = DeployContext::LocalContext().GetRankTableBuilder();
  std::string rank_table_json_str;
  rank_builder.SetRankTable(rank_table);
  bool ret = rank_builder.GetRankTable(rank_table_json_str);
  GE_CHK_BOOL_RET_STATUS(ret, FAILED, "failed to get rank table");
  GELOGI("[InitFlowGwInfo] rank_table is %s", rank_table_json_str.c_str());
  std::string remote_group_cache_config;
  GE_CHK_STATUS_RET(GetRemoteGroupCacheConfig(remote_group_cache_config), "Get remote group cache config failed.");
  GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().SetRemoteGroupCacheConfig(remote_group_cache_config),
                    "Failed to set remote group cache config, remote_group_cache_config=%s.",
                    remote_group_cache_config.c_str());

  std::map<std::string, DeployFlowGwInfo> deploy_flowgw_infos;
  GE_CHK_STATUS_RET(PrepareFlowGwInfos(rank_table_json_str, deploy_flowgw_infos),
                    "Failed to prepare flowgw infos.");
  // multi device send request
  ThreadPool pool("ge_dpl_ipr", deploy_flowgw_infos.size(), false);
  std::vector<std::future<Status>> fut_rets;
  for (const auto &it : deploy_flowgw_infos) {
    const auto &deploy_flowgw_info = it.second;
    auto fut = pool.commit([this, &deploy_flowgw_info, &remote_group_cache_config]() -> Status {
      GE_CHK_STATUS_RET(InitProcessResourceRequest(deploy_flowgw_info, remote_group_cache_config), "Failed to deploy rank table");
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }
  for (auto &fut : fut_rets) {
    GE_CHK_STATUS_RET(fut.get(), "Failed to deploy rank table.");
  }
  flow_gw_inited_ = true;
  return SUCCESS;
}

int32_t MasterModelDeployer::GetRankTableOrder() {
  std::lock_guard<std::mutex> lk(mu_);
  static int32_t creat_rank_table_cnt = 0;
  creat_rank_table_cnt++;
  GELOGD("[CreateRankTable] begin, creat_rank_table_cnt is %d.", creat_rank_table_cnt);
  return creat_rank_table_cnt;
}

Status MasterModelDeployer::CreateRankTable(HcomRankTable &rank_table) {
  int32_t rank_table_order = GetRankTableOrder();
  rank_table = {};
  rank_table.status = "completed";
  rank_table.version = "1.1";

  std::string master_ip;
  GE_CHK_STATUS_RET_NOLOG(NetworkManager::GetInstance().GetDataPanelIp(master_ip));

  const pid_t pid = getpid();
  const std::string collective_id = master_ip + "-" + std::to_string(pid) + "-" + std::to_string(rank_table_order);
  GELOGD("[CreateRankTable] collective_id is %s.", collective_id.c_str());
  rank_table.collective_id = collective_id;

  const auto num_nodes = DeployerProxy::GetInstance().NumNodes();
  std::map<std::string, std::vector<HcomRank>> ip_to_ranks;
  for (int32_t node_id = 0; node_id < num_nodes; ++node_id) {
    const auto *node_info = DeployerProxy::GetInstance().GetNodeInfo(node_id);
    GE_CHECK_NOTNULL(node_info);
    const auto &device_list = node_info->GetDeviceList();
    for (const auto &device_info : device_list) {
      if (!device_info.SupportHcom()) {
        continue;
      }
      HcomRank hcom_rank{};
      hcom_rank.port = std::to_string(device_info.GetDgwPort());
      hcom_rank.device_id = std::to_string(device_info.GetPhyDeviceId());
      hcom_rank.hcom_device_id = device_info.GetHcomDeviceId();
      hcom_rank.device_type = device_info.GetDeviceType();
      ip_to_ranks[device_info.GetDeviceIp()].emplace_back(hcom_rank);
      GELOGI("Create rank table, node addr = %s, port = %s, device_id = %s",
             device_info.GetDeviceIp().c_str(), hcom_rank.port.c_str(), hcom_rank.device_id.c_str());
    }
  }

  for (auto &it : ip_to_ranks) {
    HcomNode hcom_node = {};
    hcom_node.node_addr = it.first;
    hcom_node.node_ranks = std::move(it.second);
    rank_table.node_list.emplace_back(hcom_node);
  }
  return SUCCESS;
}

Status MasterModelDeployer::Finalize() {
  MemoryStatisticManager::Instance().Finalize();
  DeviceAbnormalStatusHandler::Instance().Finalize();
  AbnormalStatusMonitorFinalize();

  std::lock_guard<std::mutex> lk(mu_);
  for (auto &it : deployed_models_) {
    auto &deployed_model = it.second;
    (void) UndeployModel(deployed_model);
  }

  deployed_models_.clear();
  (void) ResourceManager::GetInstance().Finalize();
  (void) DeployerProxy::GetInstance().Finalize();
  DeployerPortDistributor::GetInstance().Finalize();
  DeployerAuthentication::GetInstance().Finalize();
  return SUCCESS;
}

Status MasterModelDeployer::GetReplicaNum(const DeployState &state, size_t &replica_num) {
  std::string is_var_init_graph;
  replica_num = 1U;
  const char_t *const kOptionExecIsVarInitGraph = "ge.exec.isVarInitGraph";
  (void) ge::GetContext().GetOption(kOptionExecIsVarInitGraph, is_var_init_graph);
  if (is_var_init_graph == kIsVarInitGraph) {
    const auto &deploy_plan = state.GetDeployPlan();
    replica_num = deploy_plan.GetSubmodels().size();
  }
  GELOGD("[Deploy][Model] get replica num = %zu", replica_num);
  return SUCCESS;
}

Status MasterModelDeployer::DeployStateUpdate(DeployState &deploy_state, uint32_t root_model_id,
                                              const FlowModelPtr &flow_model) {
  deploy_state.SetRootModelId(root_model_id);
  deploy_state.SetGraphId(flow_model->GetRootGraph()->GetGraphID());
  deploy_state.SetSessionId(GetContext().SessionId());
  bool is_dynamic_sched = false;
  (void)AttrUtils::GetBool(flow_model->GetRootGraph(), ATTR_NAME_DATA_FLOW_DYNAMIC_SCHEDULE_CFG, is_dynamic_sched);
  deploy_state.SetIsDynamicSched(is_dynamic_sched);
  SetDynamicSchedFlag(is_dynamic_sched);
  bool enable_exception_catch = false;
  (void)AttrUtils::GetBool(flow_model->GetRootGraph(), ATTR_NAME_DATA_FLOW_ENABLE_EXCEPTION_CATCH, enable_exception_catch);
  deploy_state.SetEnableExceptionCatch(enable_exception_catch);

  bool contains_n_mapping_node = false;
  (void)AttrUtils::GetBool(flow_model->GetRootGraph(), ATTR_NAME_DATA_FLOW_CONTAINS_N_MAPPING_NODE,
                           contains_n_mapping_node);
  deploy_state.SetContainsNMappingNode(contains_n_mapping_node);

  GE_CHK_STATUS_RET(GetModelInputAlignAttrs(flow_model->GetRootGraph(), deploy_state.MutableInputAlignAttrs()),
                    "Failed to get model input align attrs.");
  return SUCCESS;
}

Status MasterModelDeployer::DeployModel(const FlowModelPtr &flow_model, DeployResult &deploy_result) {
  GE_TIMESTAMP_START(DeployMallocTrim);
  // to avoid memory fragment and ensure optimal deployment performance, use malloc_trim to back free stack to system
  (void)malloc_trim(0);
  GE_TIMESTAMP_EVENT_END(DeployMallocTrim, "malloc trim before deploying master model");

  IncDeployingRootModelNum();
  GE_DISMISSABLE_GUARD(guard, [this]() {
    DecreaseDeployingRootModelNum();
  });

  GE_CHK_STATUS_RET(InitFlowGwInfo());

  // register device status callback
  auto root_model_id = ++model_id_gen_;

  GE_CHECK_NOTNULL(flow_model->GetRootGraph());
  DeployState deploy_state(flow_model);
  GE_CHK_STATUS_RET(DeployStateUpdate(deploy_state, root_model_id, flow_model));

  auto &local_context = DeployContext::LocalContext();
  GE_TIMESTAMP_START(MasterDeployModel);
  GE_CHK_STATUS_RET_NOLOG(HeterogeneousModelDeployer::DeployModel(local_context, deploy_state));
  GE_TIMESTAMP_EVENT_END(MasterDeployModel, "deploying master model");
  GE_DISMISSABLE_GUARD(deployed_model, [&deploy_state]() {
    GELOGE(FAILED, "Error occurred while deploying model, now start rollback");
    HeterogeneousModelDeployer::UndeployModel(deploy_state.GetDeployedNodeIds(), deploy_state.GetRootModelId());
    DeployContext::LocalContext().DestroyDeployState(deploy_state.GetRootModelId());
  });
  GE_CHK_STATUS_RET_NOLOG(SetDeployResult(deploy_state, deploy_result));
  GE_DISMISS_GUARD(deployed_model);

  DeployedModel deployed_model;
  deployed_model.model_id = root_model_id;
  deployed_model.deployed_remote_nodes = deploy_state.GetDeployedNodeIds();
  deployed_model.model_deploy_infos =
      deploy_state.MutableDeployPlan().GetModelDeployInfos();
  {
    std::lock_guard<std::mutex> lk(mu_);
    deployed_models_.emplace(root_model_id, std::move(deployed_model));
    AddDeployedModelInfo(root_model_id);
  }
  local_context.DestroyDeployState(root_model_id);
  return SUCCESS;
}

Status MasterModelDeployer::Undeploy(const uint32_t model_id) {
  MasterModelDeployer::DeployedModel deployed_model;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = deployed_models_.find(model_id);
    if (it == deployed_models_.end()) {
      GELOGE(PARAM_INVALID, "Failed to undeploy model, model id not found, id = %u", model_id);
      return PARAM_INVALID;
    }
    DelDeployedModelInfo(model_id);
    deployed_model = std::move(it->second);
    deployed_models_.erase(it);
  }
  DelCallback(model_id);
  UndeployModel(deployed_model);
  return SUCCESS;
}

Status MasterModelDeployer::UpdateProfilingInfo(const bool is_prof_start) {
  std::string rank_id;
  (void)GetContext().GetOption(OPTION_EXEC_RANK_ID, rank_id);
  // embedding service rank 0 control all ps executor
  if (!rank_id.empty() && rank_id != "0") {
    return SUCCESS;
  }
  const std::string &config_data = ProfilingProperties::Instance().GetDeviceConfigData();
  std::map<uint32_t, std::set<int32_t>> model_id_to_nodes;
  std::vector<uint32_t> model_ids;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &it : deployed_models_) {
      auto &deployed_model = it.second;
      model_id_to_nodes[it.first] = deployed_model.deployed_remote_nodes;
      model_ids.emplace_back(it.first);
      GELOGI("[UpdateProfilingInfo] model id = %u.", it.first);
    }
  }
  GE_CHK_STATUS(DeployContext::LocalContext().UpdateLocalProfiling(is_prof_start, config_data, model_ids),
                "Filed to UpdateLocalProfilingInfo");
  GE_CHK_STATUS(HeterogeneousModelDeployer::UpdateRemoteProfiling(is_prof_start, config_data,
                model_id_to_nodes), "Filed to UpdateRemoteProfiling");
  return SUCCESS;
}

void MasterModelDeployer::UndeployModel(MasterModelDeployer::DeployedModel &deployed_model) {
  const int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  auto nodes_to_undeploy = deployed_model.deployed_remote_nodes;
  nodes_to_undeploy.emplace(local_node_id);  // ensure local route can be released
  (void) HeterogeneousModelDeployer::UndeployModel(nodes_to_undeploy, deployed_model.model_id);
  GELOGD("[Undeploy][Model] ended, model id = %u.", deployed_model.model_id);
}

Status MasterModelDeployer::GetBroadcastInputQueueAttrs(
    const DeployPlan &deploy_plan,
    const ExchangeRoute &route,
    std::vector<std::vector<DeployQueueAttr>> &broadcast_input_queue_attrs) {
  const auto &input_queue_indices = deploy_plan.GetInputQueueIndices();
  broadcast_input_queue_attrs.resize(input_queue_indices.size());
  for (size_t i = 0U; i < input_queue_indices.size(); ++i) {
    auto input_queue_index = input_queue_indices[i];
    auto dst_endpoint_indices = deploy_plan.GetBroadcastIndices(input_queue_index);
    bool is_all_proxy_queue = true;
    for (auto dst_index : dst_endpoint_indices) {
      if (!route.IsProxyQueue(dst_index)) {
        is_all_proxy_queue = false;
        break;
      }
    }
    if (is_all_proxy_queue) {
      GELOGI("All input[%zu] dst endpoint is proxy queue, indices = %s.", i, ToString(dst_endpoint_indices).c_str());
      GE_CHK_STATUS_RET(route.GetQueueAttrs(dst_endpoint_indices, broadcast_input_queue_attrs[i]),
                        "Failed to get model broadcast input queue attrs.");
    }
  }
  return SUCCESS;
}

Status MasterModelDeployer::GetModelIoQueueAttrs(const DeployPlan &deploy_plan,
                                                 const ExchangeRoute &route,
                                                 DeployResult &deploy_result) {
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetInputQueueIndices(), deploy_result.input_queue_attrs),
                    "Failed to get model input queue ids.");
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetControlInputQueueIndices(),
                                        deploy_result.control_input_queue_attrs),
                    "Failed to get model control input queue ids.");
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetOutputQueueIndices(), deploy_result.output_queue_attrs),
                    "Failed to get model output queue ids");
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetControlOutputQueueIndices(),
                                        deploy_result.control_output_queue_attrs),
                    "Failed to get model control output queue ids.");
  GE_CHK_STATUS_RET(GetBroadcastInputQueueAttrs(deploy_plan,
                                                route,
                                                deploy_result.broadcast_input_queue_attrs),
                    "Failed to get model broadcast input queue attrs.");
  return SUCCESS;
}

Status MasterModelDeployer::GetDynamicSchedModelIoQueueIds(const DeployPlan &deploy_plan,
                                                           const ExchangeRoute &route,
                                                           DeployResult &deploy_result) {
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetDynamicSchedPlan().GetStatusOutputQueueIndices(),
                                        deploy_result.status_output_queue_attrs),
                    "Failed to get model input queue ids.");
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetDynamicSchedPlan().GetSchedOutputQueueIndices(),
                                        deploy_result.sched_output_queue_attrs),
                    "Failed to get model control input queue ids.");
  GE_CHK_STATUS_RET(route.GetQueueAttrs(deploy_plan.GetDynamicSchedPlan().GetSchedInputQueueIndices(),
                                        deploy_result.sched_input_queue_attrs),
                    "Failed to get model output queue ids");
  deploy_result.datagw_request_bindings = deploy_plan.GetDynamicSchedPlan().GetDatagwRequestBindings();
  deploy_result.model_index_info = deploy_plan.GetDynamicSchedPlan().GetModelIndexInfo();
  return SUCCESS;
}

Status MasterModelDeployer::InitProcessResourceRequest(const DeployFlowGwInfo &flowgw_info,
                                                       const std::string &remote_group_cache_config) const {
  deployer::DeployerRequest request;
  request.set_type(deployer::kInitProcessResource);
  auto init_process_resource = request.mutable_init_process_resource_request();
  const auto &device_info = *flowgw_info.device_info;
  init_process_resource->set_device_id(device_info.GetDeviceId());
  init_process_resource->set_device_type(static_cast<int32_t>(device_info.GetDeviceType()));
  init_process_resource->set_profiling_on(flowgw_info.profiling_on);
  init_process_resource->set_remote_group_cache_alloc_config(remote_group_cache_config);
  if (device_info.SupportHcom()) {
    init_process_resource->set_rank_table(flowgw_info.rank_table);
    init_process_resource->set_rank_id(flowgw_info.rank_id);
  }
  init_process_resource->mutable_res_ids()->Add(flowgw_info.res_ids.begin(), flowgw_info.res_ids.end());
  deployer::DeployerResponse response;
  GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(device_info.GetNodeId(), request, response),
                    "[InitProcessResource] failed to send request, node id=%d.", device_info.GetNodeId());
  GE_CHK_BOOL_RET_STATUS(response.error_code() == SUCCESS, FAILED,
                         "[InitProcessResource] failed, node id=%d, error code=%u, error message=%s.",
                         device_info.GetNodeId(), response.error_code(), response.error_message().c_str());
  GELOGI("[InitProcessResource] success, node id=%d, profiling_on=%d.", device_info.GetNodeId(),
         static_cast<int32_t>(flowgw_info.profiling_on));
  return SUCCESS;
}

Status MasterModelDeployer::SetDeployResult(const DeployState &state, DeployResult &deploy_result) {
  uint32_t model_id = state.GetRootModelId();
  deploy_result.model_id = model_id;
  deploy_result.is_exception_catch = state.IsEnableExceptionCatch();
  deploy_result.is_dynamic_sched = state.GetIsDynamicSched();
  deploy_result.contains_n_mapping_node = state.IsContainsNMappingNode();
  deploy_result.dev_abnormal_callback = []() -> Status {
    return DeployerProxy::GetInstance().GetDeviceAbnormalCode();
  };
  deploy_result.input_align_attrs = state.GetInputAlignAttrs();
  deploy_result.abnormal_status_callback_info = GetAbnormalStatusCallbackInfo();
  if (deploy_result.is_exception_catch) {
    deploy_result.exception_notify_callback = [model_id, this](const UserExceptionNotify &user_exception_notify) {
      NotifyException(model_id, user_exception_notify);
    };
  }
  auto *route = DeployContext::LocalContext().QueryFlowRoute(state.GetRootModelId());
  GE_CHECK_NOTNULL(route);
  GE_CHK_BOOL_RET_STATUS(GetModelIoQueueAttrs(state.GetDeployPlan(), *route, deploy_result) == SUCCESS, FAILED,
                         "Failed to get model queues");
  GE_CHK_BOOL_RET_STATUS(GetDynamicSchedModelIoQueueIds(state.GetDeployPlan(), *route, deploy_result) == SUCCESS,
                         FAILED, "Failed to get model queues");
  GE_CHK_STATUS_RET(GetReplicaNum(state, deploy_result.replica_num), "Failed to get model replica num.");
  SetTrimmingModelInstanceNames(state.GetDeployPlan().GetTrimmingEdgesModelInstances(),
      deploy_result.model_trimming_edges_model_instances);
  return SUCCESS;
}

void MasterModelDeployer::SetTrimmingModelInstanceNames(
    const std::map<std::string, std::vector<std::string>> &org_model_instance_names,
    std::vector<std::unordered_set<std::string>> &processed_model_instance_names) {
  if (org_model_instance_names.empty()) {
    GELOGD("There is no trimming model relation need to be processed.");
    return;
  }
  std::unordered_set<std::string> visited;
  for (const auto &instance_name_relation : org_model_instance_names) {
    const auto &src_name = instance_name_relation.first;
    if (visited.count(src_name) == 0UL) {
      std::unordered_set<std::string> instance_tree;
      DfsFind(src_name, org_model_instance_names, visited, instance_tree);
      processed_model_instance_names.emplace_back(std::move(instance_tree));
    }
  }
  GELOGI("Get Trimming model instance set number[%zu].", processed_model_instance_names.size());
}

Status MasterModelDeployer::GetDeviceMeshIndex(const int32_t device_id, std::vector<int32_t> &node_mesh_index) {
  const auto &resource_manager = ResourceManager::GetInstance();
  auto device_info = resource_manager.GetDeviceInfo(Configurations::GetInstance().GetLocalNode().node_id,
                                                    device_id, NPU);
  if (device_info == nullptr) {
    return FAILED;
  }
  node_mesh_index = device_info->GetNodeMeshIndex();
  return SUCCESS;
}

Status MasterModelDeployer::GetValidLogicDeviceId(std::string &device_id) {
  HeterogeneousDeployPlanner::GetValidLogicDeviceId(device_id);
  return SUCCESS;
}

Status MasterModelDeployer::NotifyException(uint32_t root_model_id, const UserExceptionNotify &user_exception_notify) {
  std::set<int32_t> deploy_nodes;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto find_ret = deployed_models_.find(root_model_id);
    if (find_ret == deployed_models_.cend()) {
      GELOGE(FAILED, "Notify exception failed as model not found, root_model_id=%u", root_model_id);
      return FAILED;
    }
    deploy_nodes = find_ret->second.deployed_remote_nodes;
  }
  const int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  deploy_nodes.emplace(local_node_id);
  deployer::DeployerRequest request;
  request.set_type(deployer::kDataFlowExceptionNotify);
  auto notify_req = request.mutable_exception_notify_request();
  notify_req->set_root_model_id(root_model_id);
  auto exception_notify = notify_req->mutable_exception_notify();
  exception_notify->set_type(user_exception_notify.type);
  exception_notify->set_exception_code(user_exception_notify.exception_code);
  exception_notify->set_trans_id(user_exception_notify.trans_id);
  exception_notify->set_user_context_id(user_exception_notify.user_context_id);
  exception_notify->set_scope(user_exception_notify.scope);
  if (user_exception_notify.exception_context != nullptr) {
    exception_notify->set_exception_context(user_exception_notify.exception_context,
                                            user_exception_notify.exception_context_len);
  }
  GELOGI("notify exception to nodes[%s], root_model_id=%u, trans_id=%lu, scope=%s",
         ToString(std::vector<int32_t>(deploy_nodes.cbegin(), deploy_nodes.cend())).c_str(), root_model_id,
         user_exception_notify.trans_id, user_exception_notify.scope.c_str());
  for (auto node_id : deploy_nodes) {
    deployer::DeployerResponse response;
    (void)DeployerProxy::GetInstance().SendRequest(node_id, request, response);
  }
  return SUCCESS;
}
Status MasterModelDeployer::GetRemoteGroupCacheConfig(std::string &remote_group_cache_config) {
  if (ge::GetContext().GetOption(OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE, remote_group_cache_config) == GRAPH_SUCCESS) {
    if (remote_group_cache_config.empty()) {
      GELOGE(PARAM_INVALID, "option[%s] config value is empty.", OPTION_FLOW_GRAPH_MEMORY_MAX_SIZE);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}
}  // namespace ge
