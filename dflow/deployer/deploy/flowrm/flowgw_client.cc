/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/flowgw_client.h"
#include <signal.h>
#include "mmpa/mmpa_api.h"
#include "common/utils/rts_api_utils.h"
#include "common/debug/log.h"
#include "common/subprocess/subprocess_manager.h"
#include "deploy/flowrm/tsd_client.h"
#include "deploy/deployer/deployer.h"
#include "acl/acl.h"
#include "common/df_chk.h"

namespace ge {
namespace {
const uint32_t kWaitTimeInMilliSecond = 1000U;
constexpr int32_t kForkQsWaitTimeInMilliSec = 2000;
constexpr int32_t kInnerForkQsWaitTimeInMilliSec = 5000;
constexpr int32_t kWaitTimeoutInSec = 30;
constexpr int32_t kDestroyHcomWaitTimeoutInSec = 40;
constexpr int32_t kFlowGwNotSupport = 20;
constexpr const char_t *kProtocolTypeRdma = "RDMA";
constexpr const char_t *kFlowGwProcessName = "queue_schedule";
constexpr uint32_t kBindAllDevice = 0xffffffff;
constexpr uint64_t kInvalidHcomHandle = UINT64_MAX;
}  // namespace

FlowGwClient::FlowGwClient(uint32_t device_id, int32_t device_type, const std::vector<int32_t> &res_ids, bool is_proxy)
    : pid_(0),
      proc_status_(ProcStatus::NORMAL),
      hcom_handle_(kInvalidHcomHandle),
      device_id_(device_id),
      device_type_(device_type),
      is_proxy_(is_proxy),
      is_inited_(false),
      is_exception_(false),
      res_ids_(res_ids) {
}

std::string FlowGwClient::FormatListParam(const std::vector<int32_t> &list) {
  std::string param;
  for (size_t i = 0U; i < list.size(); ++i) {
    param += std::to_string(list[i]);
    if (i != list.size() - 1U) {
      param += ",";
    }
  }
  return param;
}

Status FlowGwClient::InnerStartFlowGw(const ProcessParam &param) {
  SubprocessManager::SubprocessConfig config{};
  config.process_type = kFlowGwProcessName;
  config.death_signal = SIGKILL;
  config.args = {
      kFlowGwProcessName,
      std::string("--deviceId=") + std::to_string(param.device_id),
      std::string("--vfId=") + std::to_string(param.vf_id),
      std::string("--pid=") + std::to_string(getpid()),
      std::string("--qsInitGroupName=") + param.group_name,
      std::string("--schedPolicy=") + std::to_string(static_cast<uint64_t>(bqs::SchedPolicy::POLICY_SUB_BUF_EVENT)),
      std::string("--starter=1"),
      std::string("--resIds=") + FormatListParam(param.res_ids),
      std::string("--devIds=") + FormatListParam(param.dev_ids),
  };
  GE_CHK_STATUS_RET(SubprocessManager::GetInstance().ForkSubprocess(config, pid_), "Failed to fork flowgw.");
  GE_CHK_STATUS_RET(ge::MemoryGroupManager::GetInstance().MemGrpAddProc(param.group_name, pid_, false, true),
                    "Failed to add flowgw[%d] to memory group[%s].", pid_, param.group_name.c_str());
  if (!param.dev_ids.empty()) {
    rtBindHostpidInfo info{};
    info.cpType = RT_DEVDRV_PROCESS_USER;
    info.hostPid = static_cast<uint32_t>(getpid());
    info.chipId = kBindAllDevice;
    info.len = static_cast<uint32_t>(RT_PROCESS_SIGN_LENGTH);
    const auto ret = memcpy_s(info.sign, static_cast<size_t>(RT_PROCESS_SIGN_LENGTH), &pid_, sizeof(pid_t));
    GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "Failed to memcpy_s, ret = %d", ret);
    GE_CHK_RT_RET(rtBindHostPid(info));
  }

  (void) mmSleep(kInnerForkQsWaitTimeInMilliSec);  // wait for qs fully initialized

  std::function<void(const ProcStatus &)> excpt_handle_callback = [param, this](const ProcStatus &proc_status) {
    GEEVENT("Queue schedule process status is %d.", static_cast<int32_t>(proc_status));
    proc_status_ = proc_status;
  };
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid_, excpt_handle_callback);
  status_func_ = [this]() -> ProcStatus {
    return proc_status_;
  };
  return SUCCESS;
}

int32_t FlowGwClient::KillProcess(pid_t pid, int32_t signal) {
  return kill(pid, signal);
}

Status FlowGwClient::Shutdown(pid_t pid) {
  int32_t status = 0;
  if (KillProcess(pid, SIGTERM) == 0) {
    GELOGI("kill queue schedule success, pid=%d", pid);
    const pid_t ret = mmWaitPid(pid, &status, 0);
    GELOGI("mmWaitPid queue schedule[%d] ret[%d] status[%d] finish", pid, ret, status);
    return SUCCESS;
  }

  GELOGI("queue schedule[%d] stopping", pid);
  uint32_t times = 0U;
  while (times < 10U) {
    const pid_t ret = mmWaitPid(pid, &status, M_WAIT_NOHANG);
    if (ret != 0) {
      GELOGI("mmWaitPid success, ret[%d] pid[%d] status[%d]", ret, pid, status);
      return SUCCESS;
    }
    (void) mmSleep(100U);
    ++times;
  }
  GELOGI("queue schedule[%d] stopping timeout, ready to kill", pid);
  if (KillProcess(pid, SIGKILL) != 0) {
    const pid_t ret = mmWaitPid(pid, &status, 0);
    GELOGI("mmWaitPid queue schedule[%d] ret[%d] status[%d] finish", pid, ret, status);
  }
  GELOGI("queue schedule[%d] stopped", pid);
  return SUCCESS;
}

Status FlowGwClient::StartFlowGwFromTsd(uint32_t device_id, const std::string &group_name) {
  GE_CHK_STATUS_RET(TsdClient::GetInstance().StartFlowGw(device_id, group_name, pid_),
                    "Failed to start flowgw, device_id = %d, group name = %s.",
                    device_id, group_name.c_str());
  GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().RemoteMemGrpAddProc(device_id, group_name, pid_, false, true),
                    "Failed to add remote group, device_id = %d, group name = %s, pid = %d.",
                    device_id, group_name.c_str(), pid_);
  (void) mmSleep(kForkQsWaitTimeInMilliSec);  // wait for qs fully initialized;
  status_func_ = [this]() -> ProcStatus {
    ProcStatus stat = ProcStatus::NORMAL;
    auto ret = TsdClient::GetInstance().GetProcStatus(device_id_, pid_, stat, kFlowGwProcessName);
    if (ret == SUCCESS) {
      return stat;
    }
    GELOGE(FAILED, "Failed to get flowgw status, device_id = %d, pid = %d.", device_id_, pid_);
    return ProcStatus::EXITED;
  };
  return SUCCESS;
}

Status FlowGwClient::ToVisibleDeviceId(const std::vector<int32_t> &logical_device_ids,
                                       std::vector<int32_t> &visible_device_ids) {
  for (auto logical_device_id : logical_device_ids) {
    int32_t visible_device_id = logical_device_id;
    DF_CHK_ACL_RET(aclrtGetLogicDevIdByUserDevId(logical_device_id, &visible_device_id));
    visible_device_ids.emplace_back(visible_device_id);
  }
  return SUCCESS;
}

Status FlowGwClient::StartFlowGw() {
  if (is_proxy_) {
    const auto &mem_group_name = MemoryGroupManager::GetInstance().GetRemoteMemGroupName(device_id_);
    GE_CHK_STATUS_RET(StartFlowGwFromTsd(device_id_, mem_group_name),
                      "Failed to start flowgw from tsd, device_id = %d, memory group = %s.",
                      device_id_, mem_group_name.c_str());
    return SUCCESS;
  }
  const auto &mem_group_name = MemoryGroupManager::GetInstance().GetQsMemGroupName();
  const auto &node_config = Configurations::GetInstance().GetLocalNode();
  ProcessParam param = {};
  param.device_id = device_id_;
  param.group_name = mem_group_name;
  param.res_ids = res_ids_;
  GE_CHK_STATUS_RET(ToVisibleDeviceId(node_config.proxy_device_ids, param.dev_ids),
                    "Failed to covert logical device id to visible device id");
  GE_CHK_STATUS_RET(InnerStartFlowGw(param),
                    "Failed to start flowgw, device_id = %d, memory group = %s.",
                    device_id_, mem_group_name.c_str());
  return SUCCESS;
}

Status FlowGwClient::Initialize() {
  GE_CHK_STATUS_RET(StartFlowGw(), "Failed to start flowgw client, device_id = %d, is_proxy = %d.",
                    device_id_, is_proxy_);
  is_inited_ = true;
  GE_CHK_STATUS_RET(InitFlowGwClient(), "Failed to init flowgw client, pid = %d, device_id = %d, is_proxy = %d.",
                    pid_, device_id_, is_proxy_);
  return SUCCESS;
}

Status FlowGwClient::InitFlowGwClient() const {
  std::string proc_sign = "";
  uint32_t count = 0;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  constexpr uint32_t kMaxInitTimes = 30U;
  while (count++ < kMaxInitTimes) {
    GEEVENT("Init datagw client begin, flowgw server pid=%d, "
            "device id=%u, is_proxy = %d, number of initializations=%u.",
            pid_, device_id_, static_cast<int32_t>(is_proxy_), count);
    int32_t res = dgw_client->Initialize(pid_, proc_sign, is_proxy_, kWaitTimeoutInSec);
    if (res == 0) {
      GE_CHK_STATUS_RET(SetHcclProtocol(), "Failed to set hccl protocol");
      GEEVENT("Init datagw client success, flowgw server pid=%u, device id=%u, number of initializations=%u.",
              pid_, device_id_, count);
      return SUCCESS;
    }
    (void) mmSleep(kWaitTimeInMilliSecond);
  }

  GELOGE(FAILED, "Init datagw client failed, dgw server pid=%d, device id=%u.", pid_, device_id_);
  REPORT_INNER_ERR_MSG("E19999", "Init datagw client failed, dgw server pid=%d, device id=%u.", pid_, device_id_);
  return FAILED;
}

Status FlowGwClient::SetHcclProtocol() const {
  const auto &node_config = Configurations::GetInstance().GetLocalNode();
  const auto &protocol = node_config.protocol;
  bqs::ConfigInfo cfg = {};
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_SET_HCCL_PROTOCOL;
  cfg.cfg.hcclProtocolCfg.protocol = protocol == kProtocolTypeRdma ?
                                     bqs::HcclProtocolType::RDMA :
                                     bqs::HcclProtocolType::TCP;
  std::vector<int32_t> results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Set][HcclProtocol] failed, ret = %d, hccl protocol = %u, device_id = %u.",
                         ret, static_cast<uint32_t>(cfg.cfg.hcclProtocolCfg.protocol), device_id_);
  GELOGI("[Set][HcclProtocol] success, hccl protocol = %u, device_id = %u.",
         static_cast<uint32_t>(cfg.cfg.hcclProtocolCfg.protocol), device_id_);
  return SUCCESS;
}

Status FlowGwClient::WaitConfigEffect() {
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->WaitConfigEffect(0, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0 || ret == kFlowGwNotSupport, FAILED,
                         "[Wait][ConfigEffect] failed, ret = %d, device_id = %u, pid = %d.", ret, device_id_, pid_);
  return SUCCESS;
}

Status FlowGwClient::UpdateProfiling() const {
  bqs::ConfigInfo cfg = {};
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_UPDATE_PROFILING;
  cfg.cfg.profCfg.profMode = bqs::ProfilingMode::PROFILING_OPEN;
  std::vector<int32_t> results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Update][Profiling] failed, ret = %d, device_id = %u.", ret, device_id_);
  GELOGI("[Update][Profiling] success, device_id = %u.", device_id_);
  return SUCCESS;
}

Status FlowGwClient::CreateHcomHandle() {
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t res = dgw_client->CreateHcomHandle(rank_table_, rank_id_, nullptr, hcom_handle_, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(res == 0, FAILED,
                         "Init datagw handle failed, rank_table is %s, device_id is %u, rank_id is %d",
                         rank_table_.c_str(), device_id_, rank_id_);
  GELOGI("[Create][Hcomm] handle success, device id=%u, hcom_handle=%lu", device_id_, hcom_handle_);
  return SUCCESS;
}

Status FlowGwClient::CreateFlowGwGroup(const std::vector<const ExchangeEndpoint *> &endpoint_list,
                                       int32_t &group_id) const {
  GE_CHK_BOOL_RET_STATUS(!endpoint_list.empty(), FAILED,
                         "[Create][Group] failed, endpoint list is empty, device id=%u.", device_id_);
  std::vector<bqs::Endpoint> endpoints;
  ToFlowGwEndpoints(endpoint_list, endpoints);
  bqs::ConfigInfo cfg = {};
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_ADD_GROUP;
  cfg.cfg.groupCfg.endpointNum = static_cast<uint32_t>(endpoints.size());
  cfg.cfg.groupCfg.endpoints = const_cast<bqs::Endpoint *>(&endpoints[0]);
  std::vector<int32_t> results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Create][Group] failed, ret = %d, endpoint size=%zu, device id=%u.",
                         ret, endpoint_list.size(), device_id_);
  group_id = cfg.cfg.groupCfg.groupId;
  GELOGD("[Create][Group] success, endpoint size=%zu, device id=%u.", endpoint_list.size(), device_id_);
  return SUCCESS;
}

Status FlowGwClient::DestroyFlowGwGroup(int32_t group_id) const {
  bqs::ConfigInfo cfg = {};
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_DEL_GROUP;
  cfg.cfg.groupCfg.groupId = group_id;
  std::vector<int32_t> results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Destroy][Group] failed, ret = %d, group_id = %d, device id = %u, datagw pid = %d.",
                         ret, group_id, device_id_, pid_);
  GELOGI("[Destroy][Group] success, group_id = %d, device id = %u, datagw pid = %d.", group_id, device_id_, pid_);
  return SUCCESS;
}

Status FlowGwClient::ClearFlowgwModelData(const std::set<uint32_t> &model_ids, const int32_t type) {
  GE_IF_BOOL_EXEC(!is_inited_, return SUCCESS);
  bqs::ConfigInfo cfg = {};

  if (is_exception_ && type == EXCEPTION_HANDLE_STOP) {
    (void) DestroyHcomHandle();
    (void) Finalize();
    return SUCCESS;
  }

  if (type == EXCEPTION_HANDLE_STOP) {
    cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_STOP_SCHEDULE;
  } else {
    cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_CLEAR_AND_RESTART_SCHEDULE;
  }
  auto model_num = model_ids.size();
  std::unique_ptr<int32_t[]> model_ids_ptr = MakeUnique<int32_t[]>(model_num);
  auto index = 0;
  for (auto &model_id : model_ids) {
    model_ids_ptr[index++] = static_cast<int32_t>(model_id);
  }
  cfg.cfg.reDeployCfg.rootModelNum = model_num;
  cfg.cfg.reDeployCfg.rootModelIdsAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((model_ids_ptr.get())));

  std::vector<int32_t> results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[ClearModelExceptionData] failed, ret = %d, type = %d, device id = %u, datagw pid = %d.",
                         ret, type, device_id_, pid_);
  GELOGI("[ClearModelExceptionData] success, type = %d, device id = %u, datagw pid = %d.", type, device_id_, pid_);
  return SUCCESS;
}

Status FlowGwClient::GrantQueueForRoute(const std::pair<const ExchangeEndpoint *,
                                                        const ExchangeEndpoint *> &queue_route) const {
  GELOGD("[Grant][Queue] for flowgw route.");
  if ((queue_route.first->type == ExchangeEndpointType::kEndpointTypeExternalQueue ||
       queue_route.first->type == ExchangeEndpointType::kEndpointTypeQueue) &&
      queue_route.first->device_type == device_type_) {
    GE_CHK_STATUS_RET(GrantQueue(queue_route.first->device_id,
                                 queue_route.first->id,
                                 pid_,
                                 GrantType::kReadOnly),
                      "Grant src queue failed, endpoint=[%s], datagw pid = %d",
                      queue_route.first->DebugString().c_str(), pid_);
  }

  if ((queue_route.second->type == ExchangeEndpointType::kEndpointTypeExternalQueue ||
       queue_route.second->type == ExchangeEndpointType::kEndpointTypeQueue) &&
      queue_route.second->device_type == device_type_) {
    GE_CHK_STATUS_RET(GrantQueue(queue_route.second->device_id,
                                 queue_route.second->id,
                                 pid_,
                                 GrantType::kWriteOnly),
                      "Grant dst queue failed, endpoint=[%s], datagw pid = %d",
                      queue_route.second->DebugString().c_str(), pid_);
  }
  return SUCCESS;
}

void FlowGwClient::PrintFlowRoute(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes,
    bool print_error) {
  for (auto &queue_route : queue_routes) {
    if (print_error) {
      GELOGE(FAILED, "Flow route info, src endpoint = [%s], dst endpoint = [%s]",
             queue_route.first->DebugString().c_str(), queue_route.second->DebugString().c_str());
    } else {
      GELOGI("Flow route info, src endpoint = [%s], dst endpoint = [%s]",
             queue_route.first->DebugString().c_str(), queue_route.second->DebugString().c_str());
    }
  }
}

Status FlowGwClient::FillCommChannelAttrEndpoint(const ExchangeEndpoint &endpoint, bqs::Endpoint &ret) {
  ret.type = bqs::EndpointType::COMM_CHANNEL;
  ret.attr.channelAttr.handle = endpoint.hcom_handle;
  ret.attr.channelAttr.localTagId = endpoint.tag_id;
  ret.attr.channelAttr.localRankId = endpoint.rank_id;
  ret.attr.channelAttr.peerTagId = endpoint.peer_tag_id;
  ret.attr.channelAttr.peerRankId = endpoint.peer_rank_id;
  ret.attr.channelAttr.localTagDepth = endpoint.depth;
  ret.attr.channelAttr.peerTagDepth = endpoint.depth;
  return SUCCESS;
}

bqs::Endpoint FlowGwClient::ToFlowGwEndpoint(const ExchangeEndpoint &endpoint) const {
  bqs::Endpoint ret = {};
  ret.resId = 0x8000 |
              static_cast<uint16_t>(endpoint.device_id) |
              (static_cast<uint16_t>(endpoint.device_type) << 14U);
  ret.globalId = endpoint.index;
  ret.modelId = endpoint.model_id;
  ret.rootModelId = endpoint.root_model_id;
  switch (endpoint.type) {
    case ExchangeEndpointType::kEndpointTypeTag: {
      (void) FillCommChannelAttrEndpoint(endpoint, ret);
      break;
    }
    case ExchangeEndpointType::kEndpointTypeExternalQueue:  // fall through
    case ExchangeEndpointType::kEndpointTypeQueue: {
      ret.type = bqs::EndpointType::MEM_QUEUE;
      ret.attr.memQueueAttr.queueId = static_cast<int32_t>(endpoint.id);
      if (device_type_ != endpoint.device_type) {
        ret.attr.memQueueAttr.queueType = 1U;
      }
      break;
    }
    case ExchangeEndpointType::kEndpointTypeGroup: {
      ret.type = bqs::EndpointType::GROUP;
      ret.attr.groupAttr.groupId = static_cast<int32_t>(endpoint.id);
      ret.peerNum = endpoint.instance_num;
      ret.localId = endpoint.instance_idx;
      ret.attr.groupAttr.policy = endpoint.is_dynamic_sched ? bqs::GroupPolicy::DYNAMIC : bqs::GroupPolicy::HASH;
      break;
    }
    default: {
      break;
    }
  }
  return ret;
}

void FlowGwClient::ToFlowGwRoutes(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes,
    std::vector<bqs::Route> &flowgw_routes) const {
  for (const auto &route : queue_routes) {
    bqs::Route flowgw_route = {};
    flowgw_route.src = ToFlowGwEndpoint(*route.first);
    flowgw_route.dst = ToFlowGwEndpoint(*route.second);
    flowgw_routes.emplace_back(flowgw_route);
  }
}

void FlowGwClient::ToFlowGwEndpoints(
    const std::vector<const ExchangeEndpoint *> &endpoints,
    std::vector<bqs::Endpoint> &flowgw_endpoints) const {
  for (const auto &endpoint : endpoints) {
    bqs::Endpoint flowgw_endpoint = {};
    flowgw_endpoint = ToFlowGwEndpoint(*endpoint);
    flowgw_endpoints.emplace_back(flowgw_endpoint);
  }
}

Status FlowGwClient::BindQueues(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes) const {
  GELOGD("[Bind][Queues] device_id = %u, routes size = %zu", device_id_, queue_routes.size());
  for (size_t i = 0U; i < queue_routes.size(); ++i) {
    const auto &queue_route = queue_routes[i];
    GE_CHK_STATUS_RET(GrantQueueForRoute(queue_route),
                      "[Grant][Queue] failed, src=[%s], dst=[%s].",
                      queue_route.first->DebugString().c_str(), queue_route.second->DebugString().c_str());
  }

  std::vector<bqs::Route> routes;
  ToFlowGwRoutes(queue_routes, routes);

  bqs::ConfigInfo cfg = {};
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_BIND_ROUTE;
  cfg.cfg.routesCfg.routeNum = routes.size();
  cfg.cfg.routesCfg.routes = const_cast<bqs::Route *>(routes.data());
  std::vector<int32_t> bind_results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, bind_results, kWaitTimeoutInSec);
  PrintFlowRoute(queue_routes, ret != 0);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Bind][Route] failed, ret = %d, device_id = %u, route size = %zu.",
                         ret, device_id_, queue_routes.size());
  GELOGI("[Bind][Route] success, route size = %zu.", queue_routes.size());
  return SUCCESS;
}

Status FlowGwClient::UnbindQueues(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes) const {
  std::vector<bqs::Route> routes;
  ToFlowGwRoutes(queue_routes, routes);
  bqs::ConfigInfo cfg = {};
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_UNBIND_ROUTE;
  cfg.cfg.routesCfg.routeNum = routes.size();
  cfg.cfg.routesCfg.routes = const_cast<bqs::Route *>(routes.data());
  std::vector<int32_t> results;
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  PrintFlowRoute(queue_routes, ret != 0);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Unbind][Route] failed, ret = %d, device_id = %u, route size = %zu.",
                         ret, device_id_, queue_routes.size());
  GELOGI("[Unbind][Route] success, route size = %zu.", queue_routes.size());
  return SUCCESS;
}

ExchangeEndpoint *FlowGwClient::GetEndpoint(std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints,
                                            int32_t index) const {
  auto it = endpoints.find(index);
  if (it == endpoints.end()) {
    GELOGE(PARAM_INVALID, "Failed to get endpoint by index: %d", index);
    return nullptr;
  }
  return it->second.get();
}

bool FlowGwClient::IsExceptionGroup(const ExchangeEndpoint *group_endpoint,
    std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) const {
  for (auto indice : group_endpoint->endpoint_indices) {
    auto endpoint = GetEndpoint(endpoints, indice);
    GE_CHECK_NOTNULL(endpoint);
    if (endpoint->is_del) {
      return true;
    }
  }
  return false;
}

Status FlowGwClient::UpdateGroupRoute(const ExchangeEndpoint *group_endpoint,
    std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) const {
  GELOGI("[UpdateExceptionRoutes] try update group, endpoint = %s.", group_endpoint->DebugString().c_str());
  std::vector<int32_t> new_endpoint_indices;
  for (auto indice : group_endpoint->endpoint_indices) {
    auto endpoint = GetEndpoint(endpoints, indice);
    GE_CHECK_NOTNULL(endpoint);
    if (!endpoint->is_del) {
      new_endpoint_indices.emplace_back(indice);
    }
  }
  if (new_endpoint_indices.size() == group_endpoint->endpoint_indices.size()) {
    GELOGI("[UpdateExceptionRoutes] group not need update.");
    return SUCCESS;
  }
  GELOGI("[UpdateExceptionRoutes] delete old group, group id = %u.", group_endpoint->id);
  GE_CHK_STATUS_RET(DestroyFlowGwGroup(static_cast<int32_t>(group_endpoint->id)),
                    "[UpdateExceptionRoutes] Failed to destroy flowgw old group.");
  int32_t group_id = 0;
  std::vector<const ExchangeEndpoint *> endpoint_list;
  for (const auto index : new_endpoint_indices) {
    auto endpoint = GetEndpoint(endpoints, index);
    GE_CHECK_NOTNULL(endpoint);
    endpoint_list.emplace_back(endpoint);
  }
  GE_CHK_STATUS_RET(CreateFlowGwGroup(endpoint_list, group_id),
                    "[UpdateExceptionRoutes] Failed to create flowgw new group.");
  GELOGI("[UpdateExceptionRoutes] create new group, group id = %d.", group_id);
  auto mutable_group_endpoint = GetEndpoint(endpoints, group_endpoint->index);
  GE_CHECK_NOTNULL(mutable_group_endpoint);
  mutable_group_endpoint->id = static_cast<uint32_t>(group_id);
  mutable_group_endpoint->endpoint_indices = new_endpoint_indices;
  return SUCCESS;
}

Status FlowGwClient::UpdateExceptionRoutes(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes,
    std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) const {
  for (auto &queue_route : queue_routes) {
    auto &src = queue_route.first;
    auto &dst = queue_route.second;
    GELOGI("[UpdateExceptionRoutes] start handle exception route, src endpoint = %s, dst endpoint = %s,"
           " flowgw client info: device_id = %u, device_type = %d.",
           src->DebugString().c_str(), dst->DebugString().c_str(), device_id_, device_type_);
    if (src->is_del || dst->is_del) {
      GELOGI("[UpdateExceptionRoutes] dst endpoint or src endpoint need delete, unbind route.");
      GE_CHK_STATUS_RET(UnbindQueues({queue_route}), "[UpdateExceptionRoutes] Failed to bind routes.");
      continue;
    }
    if (dst->type == ExchangeEndpointType::kEndpointTypeGroup) {
      if ((src->type != ExchangeEndpointType::kEndpointTypeGroup) ||
          ((src->type == ExchangeEndpointType::kEndpointTypeGroup) && !IsExceptionGroup(src, endpoints))) {
        GELOGI("[UpdateExceptionRoutes] dst is a group and src is normal, don't need update group, dynaimic process.");
        continue;
      }
    }
    GELOGI("[UpdateExceptionRoutes] dst endpoint or src endpoint need update group, unbind route first.");
    GE_CHK_STATUS_RET(UnbindQueues({queue_route}), "[UpdateExceptionRoutes] Failed to unbind old route.");
    if (src->type == ExchangeEndpointType::kEndpointTypeGroup) {
      GE_CHK_STATUS_RET(UpdateGroupRoute(src, endpoints),
                        "[UpdateExceptionRoutes] Failed to update src group route.");
    }
    GELOGI("[UpdateExceptionRoutes] bind new route after update group, src endpoint = %s, dst endpoint = %s,",
           src->DebugString().c_str(), dst->DebugString().c_str());
    GE_CHK_STATUS_RET(BindQueues({queue_route}), "Failed to bind new route.");
  }
  return SUCCESS;
}

Status FlowGwClient::GrantQueue(uint32_t device_id, uint32_t qid, pid_t pid, GrantType grant_type) {
  rtMemQueueShareAttr_t attr = {};
  GE_CHK_BOOL_RET_STATUS(grant_type == GrantType::kReadOnly ||
                         grant_type == GrantType::kWriteOnly ||
                         grant_type == GrantType::kReadAndWrite,
                         FAILED,
                         "[Grant][Queue] type[%d] error.", static_cast<int32_t>(grant_type));
  if (grant_type == GrantType::kReadOnly) {
    attr.read = 1;
  } else if (grant_type == GrantType::kWriteOnly) {
    attr.write = 1;
  } else {
    attr.read = 1;
    attr.write = 1;
  }
  GE_CHK_RT_RET(rtMemQueueGrant(device_id, qid, pid, &attr));
  return SUCCESS;
}

Status FlowGwClient::Finalize() {
  if (!is_inited_) {
    GELOGI("FlowGw not start.");
    return SUCCESS;
  }

  GELOGI("FlowGw shutdown begin.");
  if (!is_proxy_) {
    SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid_);
    GE_CHK_STATUS(Shutdown(pid_));
  } else {
    TsdClient::GetInstance().ShutdownSubprocess(static_cast<int32_t>(device_id_), pid_, kFlowGwProcessName);
  }
  is_inited_ = false;
  GELOGI("FlowGw shutdown success.");
  return SUCCESS;
}

Status FlowGwClient::DestroyHcomHandle() {
  if (hcom_handle_ == kInvalidHcomHandle) {
    return SUCCESS;
  }
  GEEVENT("[Destroy][Handle] begin, hcom_handle = %lu, device_id = %u, pid = %d", hcom_handle_, device_id_, pid_);
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);
  const int32_t ret = dgw_client->DestroyHcomHandle(hcom_handle_, kDestroyHcomWaitTimeoutInSec);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED,
                         "[Destroy][Handle] failed, hcom_handle=%lu, device id=%u",
                         hcom_handle_, device_id_);
  hcom_handle_ = kInvalidHcomHandle;
  GEEVENT("[Destroy][Handle] success, hcom_handle=%lu, device id=%u, pid = %d", hcom_handle_, device_id_, pid_);
  return SUCCESS;
}

Status FlowGwClient::GetOrCreateHcomHandle(uint64_t &hcom_handle) {
  if (hcom_handle_ == kInvalidHcomHandle) {
    GE_CHK_STATUS_RET(CreateHcomHandle(), "Failed to create hcom handle");
  }
  hcom_handle = hcom_handle_;
  return SUCCESS;
}

ProcStatus FlowGwClient::GetSubProcStat() const {
  GE_IF_BOOL_EXEC(!is_inited_ || is_exception_, return ProcStatus::INVALID);
  return status_func_();
}

void FlowGwClient::SetExceptionFlag() {
  is_exception_ = true;
}

Status FlowGwClient::ConfigSchedInfoToDataGw(const uint32_t device_id, const int32_t input_indice,
                                             const uint32_t input, const uint32_t output,
                                             const uint32_t root_model_id, const bool is_proxy) const {
  auto dgw_client = bqs::DgwClient::GetInstance(device_id_, pid_, is_proxy_);
  GE_CHECK_NOTNULL(dgw_client);

  GE_CHK_STATUS_RET(GrantQueue(device_id, input, pid_, GrantType::kReadOnly),
                    "DynamicSched Grant src queue failed, device id=%u, input queue id=%d, datagw pid=%d",
                    device_id, input, pid_);
  GE_CHK_STATUS_RET(GrantQueue(device_id, output, pid_, GrantType::kWriteOnly),
                    "DynamicSched Grant src queue failed, device id=%u, output queue id=%d, datagw pid=%d",
                    device_id, output, pid_);
  GELOGI("DynamicSched Grant src queue succ, device id=%u, input queue id=%d, output queue id=%d, datagw pid=%d",
         device_id, input, output, pid_);

  bqs::ConfigInfo cfg = {};
  bqs::DynamicSchedConfigV2 dynamic_sched_cfg = {};
  dynamic_sched_cfg.rootModelId = root_model_id;
  dynamic_sched_cfg.requestQ.queueId = output;
  dynamic_sched_cfg.requestQ.deviceId = device_id;
  dynamic_sched_cfg.requestQ.deviceType = device_type_;
  dynamic_sched_cfg.requestQ.isClientQ = is_proxy;
  dynamic_sched_cfg.responseQ.queueId = input;
  dynamic_sched_cfg.responseQ.deviceId = device_id;
  dynamic_sched_cfg.responseQ.deviceType = device_type_;
  dynamic_sched_cfg.responseQ.globalLogicId = input_indice;
  dynamic_sched_cfg.responseQ.isClientQ = is_proxy;
  cfg.cmd = bqs::ConfigCmd::DGW_CFG_CMD_INIT_DYNAMIC_SCHEDULE;
  cfg.cfg.dynamicSchedCfgV2 = &dynamic_sched_cfg;

  std::vector<int32_t> results;
  int32_t ret = dgw_client->UpdateConfig(cfg, results, kWaitTimeoutInSec);
  if (ret != 0) {
    GELOGE(FAILED, "DynamicSched cfg failed, ret = %d, device id=%u, input queue id=%u, output queue id=%u,"
           " datagw pid=%d.", ret, device_id, input, output, pid_);
    REPORT_INNER_ERR_MSG("E19999", "DynamicSched cfg failed, ret = %d, device id=%u, input queue id=%u, "
                      "output queue id=%u, datagw pid=%d.", ret, device_id, input, output, pid_);
    return FAILED;
  }
  GELOGD("DynamicSched cfg succ, root_model_id=%u, device id=%u, input queue id=%u, output queue id=%u,"
         " datagw pid=%d, is_proxy=%d.", root_model_id, device_id, input, output, pid_, is_proxy);
  return SUCCESS;
}
}  // namespace ge
