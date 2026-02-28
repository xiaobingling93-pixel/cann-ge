/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/model_send/flow_model_sender.h"
#include <fstream>
#include <algorithm>
#include "common/thread_pool.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/file_constant_utils.h"
#include "framework/common/types.h"
#include "securec.h"
#include "deploy/deployer/deployer_proxy.h"
#include "deploy/flowrm/flow_route_planner.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "dflow/base/utils/process_utils.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/flowrm/flow_route_manager.h"
#include "deploy/resource/heterogeneous_deploy_planner.h"
#include "executor/executor_context.h"

namespace ge {
namespace {
constexpr int32_t kEnqueueTimeout = 5 * 60 * 1000;  // ms
// todo:临时方案新增option，待HCCL正式方案上库后删除
const std::string STATIC_MODEL_ADDR_FIXED = "ge.exec.static_model_addr_fixed";
constexpr const char *OPTION_MOMORY_POOL_THRESHOLD = "ge.experiment.memory_pool_threshold";
constexpr const char *OPTION_FLOAT_OVERFLOW_MODE = "ge.exec.float_overflow_mode";
constexpr const char *OPTION_FLOAT_OVERFLOW_MODE_SATURATION = "saturation";
constexpr const char *ATTR_NAME_IS_DATA_FLOW_GRAPH = "_dflow_is_data_flow_graph";
constexpr const char_t *ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE = "_dflow_data_flow_scope";
const std::string kUdfBuildInFuncNamePrefix = "_BuiltIn_";
const std::string kUdfResourceSubDir = "udf_resource";
constexpr size_t kMaxTransferPoolSize = 12U;
constexpr size_t kMaxSerializePoolSize = 8U;

const std::unordered_set<string> kTransferOptionsWhiteList{
    STATIC_MEMORY_POLICY,   FILE_CONSTANT_PATH, OP_WAIT_TIMEOUT, OP_EXECUTE_TIMEOUT, OPTION_EXEC_REUSE_ZERO_COPY_MEMORY,
    STATIC_MODEL_ADDR_FIXED, OPTION_EXEC_STREAM_SYNC_TIMEOUT, OPTION_MOMORY_POOL_THRESHOLD,
    OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE, OPTION_EXEC_PROFILING_MODE, OPTION_EXEC_PROFILING_OPTIONS};
}  // namespace

FlowModelSender::~FlowModelSender() {
  for (const auto &it : node_id_to_plan_) {
    const auto &remote_plan = it.second;
    FlowRouteManager::GetInstance().UndeployRoute(FlowRouteType::kTransferFlowRoute, remote_plan.route_id(),
                                                  DeployContext::LocalContext().GetFlowGwClientManager());
  }
}

Status FlowModelSender::SerializeModel(const PneModelPtr &model, ModelBufferData &model_buff) {
  return model->SerializeModel(model_buff);
}

Status FlowModelSender::DeployRemoteVarManager(DeployState &deploy_state) {
  std::map<std::string, std::vector<const DeployPlan::SubmodelInfo *>> model_groups;
  for (const auto &it : deploy_state.GetDeployPlan().GetSubmodels()) {
    const auto &submodel = it.second;
    model_groups[submodel.device_info.GetKey()].emplace_back(&submodel);
  }
  return DeployRemoteVarManager(model_groups);
}

Status FlowModelSender::DeployRemoteVarManager(const std::map<std::string,
                                                              std::vector<const DeployPlan::SubmodelInfo *>> &models) {
  std::map<int32_t, std::set<uint64_t>> sessions;
  std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> node_need_transfer_memory;
  std::map<int32_t, std::set<int32_t>> device_ids;
  for (const auto &it : models) {
    const auto &submodels = it.second;
    GE_CHK_BOOL_RET_STATUS(!submodels.empty(), FAILED, "The submodels must be not empty.");
    const auto &target_device = submodels[0]->device_info;
    GELOGD("[Deploy][RemoteVarManager] started, target_device = %s, submodel count = %zu.",
           target_device.GetDesc().c_str(),
           submodels.size());
    GE_CHK_STATUS_RET(GetAllRelatedVarManager(target_device, submodels, sessions,
                                              node_need_transfer_memory, device_ids),
                      "Failed to GetAllRelatedVarManager");
    GELOGD("[Deploy][RemoteVarManager] Success, target_device = %s.", target_device.GetDesc().c_str());
  }
  GE_CHK_STATUS_RET(TransferFileConstants(device_ids, node_need_transfer_memory),
                    "Failed to GetVarManagerAndSendToRemote.");
  return SUCCESS;
}

Status FlowModelSender::GetAllRelatedVarManager(
    const DeployPlan::DeviceInfo &device_info,
    const std::vector<const DeployPlan::SubmodelInfo *> &submodels,
    std::map<int32_t, std::set<uint64_t>> &sessions,
    std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> &node_need_transfer_memory,
    std::map<int32_t, std::set<int32_t>> &device_ids) {
  const auto node_id = device_info.GetNodeId();
  int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  for (const auto &submodel : submodels) {
    auto model = submodel->model;
    if (model == nullptr) {
      continue;
    }
    // session id is same for every submodel
    uint64_t session = GetContext().SessionId();
    (void) sessions[node_id].emplace(session);
    device_ids[node_id].emplace(device_info.GetDeviceId());
    // static graph in non soc device no need to transfer fileconst, load on host thread
    auto root_graph = model->GetRootGraph();
    GE_CHECK_NOTNULL(root_graph);
    if (node_id != local_node_id) {
      for (const auto &node : root_graph->GetAllNodes()) {
        if (node->GetType() == FILECONSTANT) {
          node_need_transfer_memory[node_id][session][node->GetOpDesc()].emplace(device_info.GetDeviceId());
          GELOGI("FileConstant[%s] need to transfer to device[%d].",
                 node->GetOpDesc()->GetName().c_str(), device_info.GetDeviceId());
        }
      }
    }
  }
  return SUCCESS;
}

Status FlowModelSender::GetOpFileInfo(const OpDescPtr &op_desc,
                                      const std::map<std::string, std::string> &file_id_to_path,
                                      std::string &file_path,
                                      size_t &offset,
                                      size_t &length) {
  ge::ConstGeTensorDescPtr tensor_desc = op_desc->GetOutputDescPtr(0U);
  GE_CHECK_NOTNULL(tensor_desc);
  int64_t total_length = 0;
  GE_CHK_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*tensor_desc, total_length),
                    "Failed to get size of file constant(%s).", op_desc->GetName().c_str());
  GE_CHK_STATUS_RET(FileConstantUtils::GetFilePath(op_desc, file_id_to_path, file_path, offset, length),
                    "Failed to get file path.");
  length = (length == 0U ? total_length : length);
  return SUCCESS;
}

Status FlowModelSender::TransferFileConstants(
    const std::map<int32_t, std::set<int32_t>> &device_ids,
    const std::map<int32_t, std::map<uint64_t, std::map<OpDescPtr, std::set<int32_t>>>> &node_need_transfer_memory) {
  std::map<std::string, std::string> file_id_to_path;
  GE_CHK_STATUS_RET(FileConstantUtils::GetFileIdToPathMapFromOption(file_id_to_path), "Failed to get file path");
  for (const auto &it : node_need_transfer_memory) {
    GELOGI("[VarManager] process shared memory.");
    auto node_id = it.first;
    auto session_op_desc_map = it.second;
    auto dev_iter = device_ids.find(node_id);
    if (dev_iter == device_ids.end()) {
      continue;
    }
    for (const auto &session_iter : session_op_desc_map) {
      auto session_id = session_iter.first;
      auto op_desc_map = session_iter.second;
      std::map<std::string, std::set<int32_t>> op_transfer_device_list;
      for (const auto &op_it : op_desc_map) {
        const auto &op_desc = op_it.first;
        std::string file_path;
        size_t offset = 0U;
        size_t length = 0U;
        GE_CHK_STATUS_RET(GetOpFileInfo(op_desc, file_id_to_path, file_path, offset, length),
                          "Failed to get file path");
        auto send_key = op_desc->GetName() + "_" +
                        file_path + "_" +
                        std::to_string(offset) + "_" +
                        std::to_string(length) + "_" +
                        std::to_string(node_id);
        op_transfer_device_list[send_key].insert(op_it.second.begin(), op_it.second.end());
        GELOGI("FileConstant[%s] need to transfer to device, device size = %zu, send key = %s.",
               op_desc->GetName().c_str(), op_it.second.size(), send_key.c_str());
      }

      for (const auto &op_desc_to_device : op_desc_map) {
        const auto &op_desc = op_desc_to_device.first;
        std::string file_path;
        size_t offset = 0U;
        size_t length = 0U;
        GE_CHK_STATUS_RET(GetOpFileInfo(op_desc, file_id_to_path, file_path, offset, length),
                          "Failed to get file path");
        auto send_key = op_desc->GetName() + "_" +
                        file_path + "_" +
                        std::to_string(offset) + "_" +
                        std::to_string(length) + "_" +
                        std::to_string(node_id);
        ge::ConstGeTensorDescPtr tensor_desc = op_desc->GetOutputDescPtr(0U);
        GE_CHECK_NOTNULL(tensor_desc);

        SendInfo send_info;
        send_info.session_id = session_id;
        send_info.node_id = node_id;
        send_info.device_ids = std::vector<int32_t>(dev_iter->second.begin(), dev_iter->second.end());
        std::unique_ptr<std::istream> input_stream;
        GE_CHK_STATUS_RET_NOLOG(CreateInputStream(file_path, offset, input_stream));
        GE_CHK_STATUS_RET(CopyOneWeightToTransfer(send_info,
                                                  *input_stream,
                                                  length,
                                                  op_desc,
                                                  op_transfer_device_list[send_key]),
                          "Failed to send data.");
      }
    }
  }
  return SUCCESS;
}

Status FlowModelSender::TransferWeightWithQ(std::istream &input_stream,
                                            int64_t file_constant_size,
                                            const DeployQueueAttr &queue_attr) const {
  std::string compress_nodes;
  size_t used_memory = 0U;
  size_t copy_len_once = 0U;
  auto file_size = static_cast<size_t>(file_constant_size);
  const int64_t kBlockSize = 1024 * 1024 * 10;
  compress_nodes.reserve(kBlockSize);
  input_stream.seekg(0, input_stream.end);
  int64_t stream_size = input_stream.tellg();
  GE_CHK_BOOL_RET_STATUS(stream_size == file_constant_size, FAILED,
                         "The file size[%ld] is inconsistent with the specified size[%ld] of tensor dec",
                         stream_size, file_constant_size);
  input_stream.seekg(0, input_stream.beg);

  auto &exchange_service = HeterogeneousExchangeService::GetInstance();
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kEnqueueTimeout;
  while ((!input_stream.eof()) && (used_memory != file_size)) {
    input_stream.read(&compress_nodes[0], kBlockSize);
    copy_len_once = input_stream.gcount();
    if (file_size - used_memory < copy_len_once) {
      copy_len_once = file_size - used_memory;
    }
    used_memory += copy_len_once;
    GELOGD("Enqueue shared content size[%zu] of total size[%lu] to queue[%u] in device[%d]",
           copy_len_once, file_constant_size, queue_attr.queue_id, queue_attr.device_id);
    GE_CHK_STATUS_RET(exchange_service.Enqueue(queue_attr.device_id,
                                               queue_attr.queue_id,
                                               &compress_nodes[0],
                                               copy_len_once,
                                               control_info),
                      "Failed to enqueue weight, device_id = %d, queue_id = %d",
                      queue_attr.device_id, queue_attr.queue_id);
  }
  return SUCCESS;
}

Status FlowModelSender::GetOrCreateFlowRoutePlan(const SendInfo &send_info,
                                                 deployer::FlowRoutePlan &remote_route) {
  auto it = node_id_to_plan_.find(send_info.node_id);
  if (it != node_id_to_plan_.end()) {
    remote_route = it->second;
    return SUCCESS;
  }

  ExchangeRoute local_route;
  int64_t route_id = -1;
  GE_CHK_STATUS_RET_NOLOG(TransferPreDeploy(send_info, local_route, remote_route));
  GE_CHK_STATUS_RET_NOLOG(FlowRouteManager::GetInstance().AddRoute(local_route, FlowRouteType::kTransferFlowRoute,
                                                                   route_id));
  GELOGD("Add local transfer route:%ld", route_id);
  remote_route.set_route_id(route_id);
  node_id_to_plan_[send_info.node_id] = remote_route;
  return SUCCESS;
}

Status FlowModelSender::CopyOneWeightToTransfer(const SendInfo &send_info,
                                                std::istream &input_stream,
                                                int64_t file_constant_size,
                                                const OpDescPtr &op_desc,
                                                const std::set<int32_t> &devices) {
  auto session_id = static_cast<int64_t>(send_info.session_id);
  GELOGI("Enter to CopyOneWeightToTransfer, file constant size = %ld", file_constant_size);

  deployer::FlowRoutePlan remote_route;
  GE_CHK_STATUS_RET(GetOrCreateFlowRoutePlan(send_info, remote_route),
                    "Get or create transfer weight flow route plan failed.");
  auto local_route =
      FlowRouteManager::GetInstance().QueryRoute(FlowRouteType::kTransferFlowRoute, remote_route.route_id());
  GE_CHECK_NOTNULL(local_route);
  std::vector<DeployQueueAttr> queue_attrs;
  local_route->GetQueueAttrs(queue_attrs);
  GE_CHK_BOOL_RET_STATUS(queue_attrs.size() >= 1U, FAILED,
                         "Check transfer pre deploy queue size[%zu] failed.",
                         queue_attrs.size());
  auto f = std::async(std::launch::async, [this, &input_stream, file_constant_size, &queue_attrs]() {
    GE_CHK_STATUS_RET(TransferWeightWithQ(input_stream, file_constant_size, queue_attrs[0]),
                      "Failed to transfer weight, file size = %ld.", file_constant_size);
    return SUCCESS;
  });

  auto tensor_desc = op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_desc);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  auto mem_type = static_cast<uint32_t>(RT_MEMORY_DEFAULT);
  if (AttrUtils::GetInt(op_desc, ATTR_OUTPUT_MEMORY_TYPE, mem_type) && (mem_type == 1)) {  // 1: rdma
    memory_type = RT_MEMORY_RDMA_HBM;
  }

  deployer::DeployerRequest request;
  request.set_type(deployer::kDownloadSharedContent);
  auto shared_content_desc_request = request.mutable_shared_content_desc_request();
  GE_CHECK_NOTNULL(shared_content_desc_request);
  shared_content_desc_request->mutable_device_ids()->Add(devices.begin(), devices.end());
  *shared_content_desc_request->mutable_flow_route() = remote_route;
  auto shared_content_description = shared_content_desc_request->mutable_shared_content_desc();
  GE_CHECK_NOTNULL(shared_content_description);
  shared_content_description->set_session_id(session_id);
  shared_content_description->set_node_name(op_desc->GetName());
  shared_content_description->set_total_length(file_constant_size);
  shared_content_description->set_mem_type(memory_type);
  proto::TensorDescriptor *tensor_desc_proto = shared_content_description->mutable_tensor_desc();
  GeTensorSerializeUtils::GeTensorDescAsProto(*tensor_desc, tensor_desc_proto);
  deployer::DeployerResponse response;
  GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(send_info.node_id, request, response),
                    "[Send] [shared_content] failed.");
  if ((response.error_code() != SUCCESS) || (f.get() != SUCCESS)) {
    GELOGE(FAILED, "[CopyOneWeightToTransfer] failed, node_id = %d, error code = %u, error message = %s",
           send_info.node_id, response.error_code(), response.error_message().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FlowModelSender::CreateInputStream(const std::string &constant_file_path, size_t offset,
                                          unique_ptr<std::istream> &in_stream) {
  std::string real_path = RealPath(constant_file_path.c_str());
  auto file_stream = MakeUnique<std::ifstream>(real_path, std::ifstream::binary);
  GE_CHECK_NOTNULL(file_stream);
  if (!file_stream->is_open()) {
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed.", real_path.c_str());
    REPORT_INNER_ERR_MSG("E19999", "open file:%s failed.", real_path.c_str());
    return GRAPH_FAILED;
  }
  file_stream->clear();
  file_stream->seekg(offset, file_stream->beg);
  in_stream = std::move(file_stream);
  return SUCCESS;
}

Status FlowModelSender::DeployDevMaintenanceCfg(const DeployState &deploy_state) {
  for (const auto &it : deploy_state.GetDeployPlan().GetSubmodels()) {
    auto target_node_id = it.second.device_info.GetNodeId();
    GE_CHK_STATUS_RET(DownloadDevMaintenanceCfg(target_node_id), "[Download][DevMaintenanceCfg] failed, node_id[%d].",
                      target_node_id);
  }
  return SUCCESS;
}

Status FlowModelSender::DownloadDevMaintenanceCfg(int32_t dev_id) {
  GELOGD("[Download][Device debug Config] start, device id[%d]", dev_id);
  GE_MAKE_GUARD(close_config, [dev_id]() {
    DeviceMaintenanceCfgManager::GetInstance().CloseDevMaintenanceConfig(dev_id);
  });
  GE_CHK_STATUS_RET(DeviceMaintenanceCfgManager::GetInstance().CreateDevMaintenanceConfig(dev_id),
                    "[Create][MaintenanceCfg] failed, device id[%d].", dev_id);
  GE_CHK_STATUS_RET(DeployDevCfg(dev_id, DeviceDebugConfig::ConfigType::kLogConfigType),
                    "[Download][LogConfig] failed, device id[%d].", dev_id);
  GE_CHK_STATUS_RET(DeployDevCfg(dev_id, DeviceDebugConfig::ConfigType::kDumpConfigType),
                    "[Download][DumpConfig] failed, device id[%d].", dev_id);
  GE_CHK_STATUS_RET(DeployDevCfg(dev_id, DeviceDebugConfig::ConfigType::kProfilingConfigType),
                    "[Download][ProfilingConfig] failed, device id[%d].", dev_id);
  return SUCCESS;
}

Status FlowModelSender::DeployDevCfg(int32_t dev_id, DeviceDebugConfig::ConfigType conf_type) {
  std::map<DeviceDebugConfig::ConfigType, deployer::DeviceConfigType> conf_type_map = {
      {DeviceDebugConfig::ConfigType::kLogConfigType, deployer::kLogConfig},
      {DeviceDebugConfig::ConfigType::kDumpConfigType, deployer::kDumpConfig},
      {DeviceDebugConfig::ConfigType::kProfilingConfigType, deployer::kProfilingConfig},
  };
  if (conf_type_map.find(conf_type) == conf_type_map.end()) {
    GELOGW("Init device config failed, can not find config type=%d.", static_cast<int32_t>(conf_type));
    return SUCCESS;
  }
  std::string conf_data;
  const auto &conf = DeviceMaintenanceCfgManager::GetInstance().GetDevMaintenanceConfig(dev_id);
  GE_CHECK_NOTNULL(conf);
  if (conf->GetJsonDataByType(conf_type, conf_data) != SUCCESS) {
    GELOGI("Do not have device cfg, cfg type[%d].", static_cast<int32_t>(conf_type));
    return SUCCESS;
  }
  deployer::DeployerRequest request;
  request.set_type(deployer::kDownloadConf);
  deployer::DeployerResponse response;
  auto download_config_request = request.mutable_download_config_request();
  download_config_request->set_sub_type(conf_type_map[conf_type]);
  download_config_request->set_device_id(dev_id);
  download_config_request->set_config_data(&conf_data[0], conf_data.size());
  if (DeployerProxy::GetInstance().SendRequest(dev_id, request, response) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Send disconnect request failed, response info:%s",
                       response.error_message().c_str());
    GELOGE(FAILED, "[Send][Request]Send disconnect request failed, response info:%s", response.error_message().c_str());
    return FAILED;
  }
  auto error_code = response.error_code();
  if (error_code != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Check response failed. error code =%u, error message=%s", error_code,
                       response.error_message().c_str());
    GELOGE(FAILED, "[Check][Response]Check response failed. error code =%u, error message=%s", error_code,
           response.error_message().c_str());
    return FAILED;
  }
  GELOGI("DeployDevCfg successfully, conf_type=%d, dev_id=%d.", static_cast<int32_t>(conf_type), dev_id);
  return SUCCESS;
}

Status FlowModelSender::SendDatagwSchedInfo(std::map<std::string, const DeployPlan::DeviceInfo *> &datagw_devices_used,
                                            std::map<std::string, deployer::DeployerRequest> &datagw_sched_infos) {
  for (const auto &it : datagw_devices_used) {
    const auto &target_device = *it.second;

    deployer::DeployerResponse response;
    GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(target_device.GetNodeId(),
                                                               datagw_sched_infos[it.first], response),
                      "DynamicSched Failed to send request to target_device = %s", target_device.GetDesc().c_str());
    auto ret = response.error_code();
    if (ret != SUCCESS) {
      GELOGE(ret, "DynamicSched [Transfer][Sched info] failed, target_device = %s, request = %s, error_message = %s",
             target_device.GetDesc().c_str(), datagw_sched_infos[it.first].DebugString().c_str(),
             response.error_message().c_str());
      return ret;
    }
    GEEVENT("DynamicSched [Transfer][Sched info] success, target_device = %s",
            target_device.GetDesc().c_str());
  }
  return SUCCESS;
}

Status FlowModelSender::TransferDataGwDeployPlan(DeployState &deploy_state) {
  if (!deploy_state.GetIsDynamicSched() && !deploy_state.IsEnableExceptionCatch()) {
    GEEVENT(
        "DynamicSched dynamic sched model close and exception catch disable, root_model_id=%u, "
        "don't need send config to flowgw.",
        deploy_state.GetRootModelId());
    return SUCCESS;
  }
  std::map<std::string, const DeployPlan::DeviceInfo *> datagw_devices_used;
  std::map<std::string, deployer::DeployerRequest> datagw_sched_infos;
  for (const auto &name_and_submodel_desc : deploy_state.GetDeployPlan().GetSubmodels()) {
    const auto &submodel_desc = name_and_submodel_desc.second;
    if (submodel_desc.model == nullptr) {
      GEEVENT("DynamicSched, process datagw model=%s", name_and_submodel_desc.first.c_str());
      deployer::DeployerRequest request;
      request.set_type(deployer::kDatagwSchedInfo);
      auto datagw_config_info = request.mutable_datagw_sched_info();
      datagw_config_info->set_root_model_id(deploy_state.GetRootModelId());
      datagw_config_info->set_device_id(submodel_desc.device_info.GetDeviceId());
      datagw_config_info->set_device_type(submodel_desc.device_info.GetType());
      datagw_config_info->set_is_dynamic_sched(deploy_state.GetIsDynamicSched());
      auto is_proxy_q = false;
      if (submodel_desc.device_info.WithProxy()) {
        is_proxy_q = true;
      }
      datagw_config_info->set_is_proxy(is_proxy_q);
      if (submodel_desc.sched_input_queue_indices.size() == 1) {
        datagw_config_info->set_input_queue_indice(submodel_desc.sched_input_queue_indices[0]);
        GELOGI("DynamicSched set datagw sched info, input queue indice=%d.",
               submodel_desc.sched_input_queue_indices[0]);
      } else {
        GELOGE(FAILED, "DynamicSched set datagw sched info failed, input indice num %u!",
               submodel_desc.sched_input_queue_indices.size());
      }
      if (submodel_desc.sched_output_queue_indices.size() == 1) {
        datagw_config_info->set_output_queue_indice(submodel_desc.sched_output_queue_indices[0]);
        GELOGI("DynamicSched set datagw sched info, output queue indice=%d.",
               submodel_desc.sched_output_queue_indices[0]);
      } else {
        GELOGE(FAILED, "DynamicSched set datagw sched info failed, output indice num %u!",
               submodel_desc.sched_output_queue_indices.size());
      }
      datagw_devices_used[submodel_desc.device_info.GetKey()] = &submodel_desc.device_info;
      datagw_sched_infos[submodel_desc.device_info.GetKey()] = std::move(request);
      GEEVENT("DynamicSched add sched model info, root_model_id=%u, model name=%s, device_id=%d, device_type=%d, "
              "is_proxy=%d.", deploy_state.GetRootModelId(), name_and_submodel_desc.first.c_str(),
              submodel_desc.device_info.GetDeviceId(), submodel_desc.device_info.GetType(), is_proxy_q);
    }
  }
  return SendDatagwSchedInfo(datagw_devices_used, datagw_sched_infos);
}

Status FlowModelSender::TransferDeployPlan(const DeployState &deploy_state) {
  std::map<std::string, std::vector<deployer::SubmodelDesc>> grouped_by_target_device;
  std::map<std::string, const DeployPlan::DeviceInfo *> devices_used;
  GE_CHK_STATUS_RET_NOLOG(BuildSubmodelDescs(deploy_state, grouped_by_target_device, devices_used));
  for (const auto &it : devices_used) {
    const auto &target_device = *it.second;
    const auto *node_info = DeployerProxy::GetInstance().GetNodeInfo(target_device.GetNodeId());
    GE_CHECK_NOTNULL(node_info);
    auto &submodel_descs = grouped_by_target_device[target_device.GetKey()];
    for (const auto &submodel_desc : submodel_descs) {
      GEEVENT("Model deployment info, model_name = %s, model_type = %s, graph_id = %u, %s, device_id = %d.",
              submodel_desc.model_name().c_str(), submodel_desc.engine_name().c_str(), deploy_state.GetGraphId(),
              node_info->DebugString().c_str(), target_device.GetDeviceId());
    }

    deployer::DeployerRequest request;
    GE_CHK_STATUS_RET(BuildUpdateDeployPlanRequest(deploy_state, target_device, submodel_descs, request),
                      "Failed to build request");
    GEEVENT("[Transfer][DeployPlan] in deploying start, root_model_id = %u, target_device = %s",
            deploy_state.GetRootModelId(), target_device.GetDesc().c_str());
    deployer::DeployerResponse response;
    GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(target_device.GetNodeId(), request, response),
                      "Failed to send request to target_device = %s", target_device.GetDesc().c_str());
    auto ret = response.error_code();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Transfer][DeployPlan] failed, target_device = %s, request = %s, error_message = %s",
             target_device.GetDesc().c_str(), request.DebugString().c_str(), response.error_message().c_str());
      return ret;
    }
    GEEVENT("[Transfer][DeployPlan] in deploying success, root_model_id = %u, target_device = %s",
            deploy_state.GetRootModelId(), target_device.GetDesc().c_str());
  }
  return SUCCESS;
}

Status FlowModelSender::TransferFlowRoutePlan(const DeployState &deploy_state) {
  for (const auto &it : deploy_state.GetFlowRoutePlansToDeploy()) {
    const auto &node_id = it.first;
    const auto &flow_route_plan = it.second;
    deployer::DeployerRequest request;
    request.set_type(deployer::kAddFlowRoutePlan);
    auto flow_route_plan_request = request.mutable_add_flow_route_plan_request();
    flow_route_plan_request->set_node_id(node_id);
    flow_route_plan_request->set_root_model_id(deploy_state.GetRootModelId());
    *(flow_route_plan_request->mutable_flow_route_plan()) = flow_route_plan;

    GEEVENT("[Transfer][FlowRoutePlan] start, root_model_id = %u, target_node = %d",
            deploy_state.GetRootModelId(), node_id);
    deployer::DeployerResponse response;
    GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(node_id, request, response),
                      "Failed to send request to target_device = %d", node_id);
    auto ret = response.error_code();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Transfer][FlowRoutePlan] failed, target_node = %d, request = %s, error_message = %s",
             node_id, request.DebugString().c_str(), response.error_message().c_str());
      return ret;
    }
    GEEVENT("[Transfer][FlowRoutePlan] success, root_model_id = %u, target_node = %d",
            deploy_state.GetRootModelId(), node_id);
  }
  return SUCCESS;
}

void FlowModelSender::AddDynamicSchedInfo(const DeployState &deploy_state, const std::string &model_instance_name,
                                          deployer::SubmodelDesc &submodel_desc) {
  uint32_t instance_num = 0U;
  auto &model_instances_num = deploy_state.GetDeployPlan().GetDynamicSchedPlan().GetModelInstanceNum();
  auto ins_iter = model_instances_num.find(model_instance_name);
  if (ins_iter != model_instances_num.end()) {
    instance_num = ins_iter->second;
  } else {
    GELOGI("DynamicSched single instance model no need report status, model name=%s.",
           model_instance_name.c_str());
  }
  if ((instance_num > 1U) || deploy_state.IsEnableExceptionCatch()) {
    const auto &dynamic_sched_model = deploy_state.GetDeployPlan().GetSubmodels();
    auto iter = dynamic_sched_model.find(model_instance_name);
    if (iter != dynamic_sched_model.end()) {
      for (const auto &idx : iter->second.status_input_queue_indices) {
        submodel_desc.add_status_input_queue_indices(idx);
        GELOGI("DynamicSched add status input queue indice=%d, model name=%s.",
               idx, model_instance_name.c_str());
      }
      for (const auto &idx : iter->second.status_output_queue_indices) {
        submodel_desc.add_status_output_queue_indices(idx);
        GELOGI("DynamicSched add status output queue indice=%d, model name=%s.",
               idx, model_instance_name.c_str());
      }
    }
  }
}

Status FlowModelSender::BuildSubmodelDescs(
    const DeployState &deploy_state,
    std::map<std::string, std::vector<deployer::SubmodelDesc>> &submodel_descs,
    std::map<std::string, const DeployPlan::DeviceInfo *> &devices_used) {
  std::map<std::string, std::map<std::string, std::vector<std::pair<const DeployPlan::SubmodelInfo *,
    const std::string>>>> grouped_submodels;
  std::map<std::string, std::vector<std::string>> model_instances;
  for (const auto &name_and_submodel_desc : deploy_state.GetDeployPlan().GetSubmodels()) {
    const auto &submodel_desc = name_and_submodel_desc.second;
    if (submodel_desc.model == nullptr) {
      continue;
    }
    grouped_submodels[submodel_desc.model->GetModelName()]
                     [submodel_desc.device_info.GetKey()].
                     emplace_back(std::make_pair(&submodel_desc, name_and_submodel_desc.first));
    devices_used[submodel_desc.device_info.GetKey()] = &submodel_desc.device_info;
    model_instances[submodel_desc.model->GetModelName()].emplace_back(name_and_submodel_desc.first);
  }
  const InputAlignAttrs &input_align_attrs = deploy_state.GetInputAlignAttrs();
  bool enable_exception_catch = deploy_state.IsEnableExceptionCatch();
  for (const auto &it : grouped_submodels) {
    const auto &submodel_instances = it.second;
    auto replica_num = static_cast<uint32_t>(model_instances[it.first].size());
    uint32_t replica_idx = 0;
    std::map<int32_t, std::vector<std::string>> model_instances_on_node;
    for (const auto &model_instance : submodel_instances) {  // ordered by device key
      const auto &submodel_infos = model_instance.second;
      for (const auto &submodel_info_ptr : submodel_infos) {
        const auto &submodel_info = *submodel_info_ptr.first;
        deployer::SubmodelDesc submodel_desc;
        GE_CHECK_NOTNULL(submodel_info.model, "Submodel is nullptr.");
        const auto &model_name = submodel_info.model->GetModelName();
        submodel_desc.set_model_name(model_name);
        submodel_desc.set_model_instance_name(submodel_info_ptr.second);
        submodel_desc.set_process_id(submodel_info.process_id);
        model_instances_on_node[submodel_info.device_info.GetNodeId()].emplace_back(model_name);
        uint32_t replica_idx_on_node = model_instances_on_node[submodel_info.device_info.GetNodeId()].size() - 1U;
        submodel_desc.set_replica_idx_on_node(replica_idx_on_node);

        const std::string gen_model_file_path = GetModelFilePath(deploy_state, model_name);
        // user define function: change model file name to om name already generated
        if ((submodel_info.model->GetModelType() == PNE_ID_UDF) && (!submodel_info.model->GetIsBuiltinModel())) {
            const std::string untar_path =  std::to_string(deploy_state.GetSessionId()) + "/" +
                std::to_string(deploy_state.GetRootModelId()) + "/" + kUdfResourceSubDir + "/" +
                submodel_info.model->GetNormalizedModelName() + ".om";
            submodel_desc.set_model_path(untar_path);
            GELOGD("Set model file path [%s] by normalized name.", untar_path.c_str());
        } else {
          submodel_desc.set_model_path(gen_model_file_path);
          GELOGD("Set model file path [%s].", gen_model_file_path.c_str());
        }
        std::string saved_model_path;
        GE_CHK_STATUS_RET(GetSavedFilePath(submodel_info, gen_model_file_path, saved_model_path),
                          "Failed to get saved path result of cache is not compatible.");
        submodel_desc.set_saved_model_file_path(saved_model_path);
        submodel_desc.set_is_builtin_udf(submodel_info.model->GetIsBuiltinModel());
        submodel_desc.set_enable_exception_catch(enable_exception_catch);
        GELOGD("Set saved model file path [%s].", saved_model_path.c_str());
        submodel_desc.set_is_remote_model(saved_model_path == gen_model_file_path);
        submodel_desc.set_submodel_id(deploy_state.GetSubmodelId(model_name));
        submodel_desc.set_engine_name(submodel_info.model->GetModelType());
        submodel_desc.set_replica_num(replica_num);
        submodel_desc.set_replica_idx(replica_idx);
        submodel_desc.set_phy_device_id(submodel_info.device_info.GetProxyDeviceId());
        if (input_align_attrs.align_max_cache_num > 0) {
          auto *proto_input_align_attrs = submodel_desc.mutable_input_align_attrs();
          proto_input_align_attrs->set_align_max_cache_num(input_align_attrs.align_max_cache_num);
          proto_input_align_attrs->set_align_timeout(input_align_attrs.align_timeout);
          proto_input_align_attrs->set_drop_when_not_align(input_align_attrs.drop_when_not_align);
        }
        submodel_desc.set_execute_times(-1);
        const ComputeGraphPtr root_graph = submodel_info.model->GetRootGraph();
        GE_CHECK_NOTNULL(root_graph);
        bool is_dynamic = false;
        (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic);
        is_dynamic = (is_dynamic || root_graph->GetGraphUnknownFlag());
        GELOGI("Model[%s] set dynamic flag is %d.", model_name.c_str(), is_dynamic);
        submodel_desc.set_is_dynamic(is_dynamic);
        GELOGI("Model[%s] set head info is %d.", model_name.c_str(), submodel_info.is_head);
        submodel_desc.set_is_head(submodel_info.is_head);
        std::string scope = "";
        (void)AttrUtils::GetStr(root_graph, ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE, scope);
        submodel_desc.set_scope(scope); // df scope + user set scope
        for (auto idx : submodel_info.input_queue_indices) {
          submodel_desc.add_input_queue_indices(idx);
        }
        for (auto idx : submodel_info.control_input_queue_indices) {
          submodel_desc.add_input_queue_indices(idx);
        }
        for (auto idx : submodel_info.output_queue_indices) {
          submodel_desc.add_output_queue_indices(idx);
        }
        for (auto idx : submodel_info.control_output_queue_indices) {
          submodel_desc.add_output_queue_indices(idx);
        }
        AddDynamicSchedInfo(deploy_state, submodel_info_ptr.second, submodel_desc);

        auto attrs = submodel_desc.mutable_attrs();
        for (const auto &attr : submodel_info.attrs) {
          (*attrs)[attr.first] = attr.second;
          GELOGD("Add attr[%s], value[%s] for model name[%s].", attr.first.c_str(), attr.second.c_str(),
                 submodel_desc.model_name().c_str());
        }

        if (!submodel_info.invoked_model_queue_infos.empty()) {
          auto proto_invoked_model_queues = submodel_desc.mutable_invoked_model_queues();
          for (const auto &invoked_model_queue_info : submodel_info.invoked_model_queue_infos) {
            deployer::ModelQueueIndices proto_model_queue_indices;
            for (auto idx : invoked_model_queue_info.second.feed_queue_indices) {
              proto_model_queue_indices.add_input_queue_indices(idx);
            }
            for (auto idx : invoked_model_queue_info.second.fetch_queue_indices) {
              proto_model_queue_indices.add_output_queue_indices(idx);
            }
            (*proto_invoked_model_queues)[invoked_model_queue_info.first] = std::move(proto_model_queue_indices);
          }
        }
        submodel_descs[model_instance.first].emplace_back(std::move(submodel_desc));
        ++replica_idx;
      }
    }
  }
  return SUCCESS;
}

void FlowModelSender::BuildDeployPlanOptions(const DeployState &deploy_state,
                                             deployer::UpdateDeployPlanRequest &request) {
  auto options = request.mutable_options();
  for (const auto &item : GetThreadLocalContext().GetAllGlobalOptions()) {
    const auto &it = kTransferOptionsWhiteList.find(item.first);
    if (it == kTransferOptionsWhiteList.cend()) {
      continue;
    }
    options->mutable_global_options()->insert({item.first, item.second});
    GELOGI("Add global option to proto, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }

  bool is_data_flow_graph = false;
  (void)AttrUtils::GetBool(deploy_state.GetFlowModel()->GetRootGraph(),
                           ATTR_NAME_IS_DATA_FLOW_GRAPH,
                           is_data_flow_graph);
  if (is_data_flow_graph) {
    options->mutable_global_options()->insert({OPTION_FLOAT_OVERFLOW_MODE, OPTION_FLOAT_OVERFLOW_MODE_SATURATION});
    GELOGI("Add float overflow mode option to proto, value = %s.", OPTION_FLOAT_OVERFLOW_MODE_SATURATION);
  }

  for (const auto &item : GetThreadLocalContext().GetAllSessionOptions()) {
    const auto &it = kTransferOptionsWhiteList.find(item.first);
    if (it == kTransferOptionsWhiteList.cend()) {
      continue;
    }
    options->mutable_session_options()->insert({item.first, item.second});
    GELOGI("Add session option to proto, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }

  for (const auto &item : GetThreadLocalContext().GetAllGraphOptions()) {
    const auto &it = kTransferOptionsWhiteList.find(item.first);
    if (it == kTransferOptionsWhiteList.cend()) {
      continue;
    }
    options->mutable_graph_options()->insert({item.first, item.second});
    GELOGI("Add graph option to proto, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }
}

Status FlowModelSender::BuildUpdateDeployPlanRequest(
    const DeployState &deploy_state,
    const DeployPlan::DeviceInfo &target_device,
    std::vector<deployer::SubmodelDesc> &submodel_descs,
    deployer::DeployerRequest &request) {
  request.set_type(deployer::kUpdateDeployPlan);
  auto update_deploy_plan_request = request.mutable_update_deploy_plan_request();
  update_deploy_plan_request->set_graph_id(deploy_state.GetGraphId());
  update_deploy_plan_request->set_device_id(target_device.GetDeviceId());
  update_deploy_plan_request->set_device_type(target_device.GetType());
  update_deploy_plan_request->set_session_id(deploy_state.GetSessionId());
  update_deploy_plan_request->set_root_model_id(deploy_state.GetRootModelId());
  auto device_info = ResourceManager::GetInstance().GetDeviceInfo(target_device.GetNodeId(),
                                                                  target_device.GetDeviceId(), target_device.GetType());
  if (device_info != nullptr) {
    auto mem_config = deploy_state.GetFlowModel()->GetLogicDeviceToMemCfg();
    auto model_mem_size = update_deploy_plan_request->mutable_model_mem_size();
    model_mem_size->set_std_mem_size(mem_config[device_info->ToIndex()].first);
    model_mem_size->set_shared_mem_size(mem_config[device_info->ToIndex()].second);
  }
  for (auto &submodel_desc : submodel_descs) {
    *update_deploy_plan_request->add_submodel_descs() = std::move(submodel_desc);
  }

  BuildDeployPlanOptions(deploy_state, *update_deploy_plan_request);
  return SUCCESS;
}

bool FlowModelSender::CacheLocalModel(const DeployPlan::SubmodelInfo &submodel) {
  int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  if (submodel.device_info.GetNodeId() == local_node_id) {
    GE_CHECK_NOTNULL(submodel.model);
    // saved model path is empty for cache from old version
    if (!submodel.model->GetSavedModelPath().empty()) {
      return true;
    }
  }
  return false;
}

Status FlowModelSender::TransferSubmodels(DeployState &deploy_state) {
  const auto &submodels = deploy_state.GetDeployPlan().GetSubmodels();
  std::map<PneModelPtr, std::set<int32_t>> model_to_devices;
  for (const auto &it : submodels) {
    const auto &submodel = it.second;
    if (submodel.model == nullptr) {
      continue;
    }
    if (CacheLocalModel(submodel)) {
      continue;
    }
    model_to_devices[submodel.model].emplace(submodel.device_info.GetNodeId());
  }

  GELOGD("[Transfer][Submodels] start, root_model_id = %u, submodel number = %zu",
         deploy_state.GetRootModelId(),
         model_to_devices.size());
  std::vector<std::future<std::tuple<Status, PneModelPtr, ModelBufferData>>> model_buffer_futs;
  ThreadPool serialize_thread_pool("ge_dpl_serm", model_to_devices.size() > kMaxSerializePoolSize ?
                                   kMaxSerializePoolSize :
                                   model_to_devices.size(), false);
  for (const auto &it : model_to_devices) {
    const auto &pne_model = it.first;
    GE_CHECK_NOTNULL(pne_model);
    std::future<std::tuple<Status, PneModelPtr, ModelBufferData>> fut =
        serialize_thread_pool.commit([&pne_model]() -> std::tuple<Status, PneModelPtr, ModelBufferData> {
      ModelBufferData model_buff;
      const auto ret = SerializeModel(pne_model, model_buff);
      return std::make_tuple(ret, pne_model, model_buff);
    });
    model_buffer_futs.emplace_back(std::move(fut));
  }

  size_t tranfer_times = 0U;
  for (const auto &it : model_to_devices) {
    tranfer_times += it.second.size();
  }
  ThreadPool thread_pool("ge_dpl_trfm", tranfer_times > kMaxTransferPoolSize ? kMaxTransferPoolSize : tranfer_times,
                         false);
  std::vector<std::future<Status>> deploy_futures;
  for (auto &buffer_fut : model_buffer_futs) {
    auto ret = buffer_fut.get();
    const auto &pne_model = std::get<1>(ret);
    const auto &model_buff = std::get<2>(ret);
    GE_CHK_STATUS_RET(std::get<0>(ret), "Failed to serialize model, model_name = %s",
                      pne_model->GetModelName().c_str());
    auto submodel_id = deploy_state.GetSubmodelId(pne_model->GetModelName());
    const auto &target_node_ids = model_to_devices[pne_model];
    for (auto target_node_id : target_node_ids) {
      std::future<Status> fut = thread_pool.commit([target_node_id, &deploy_state, pne_model,
                                                    submodel_id, model_buff]() -> Status {
        GEEVENT("[Transfer][Submodel] in deploying start, root_model_id = %u, submodel_id = %u, target_node_id = %d",
                deploy_state.GetRootModelId(), submodel_id, target_node_id);
        const auto trans_ret = TransferModel(target_node_id, deploy_state, pne_model, model_buff);
        GE_CHK_BOOL_RET_STATUS(trans_ret == SUCCESS, trans_ret,
                               "[Transfer][Submodel] failed, root_model_id = %u, submodel_id = %u, target_node_id = %d",
                               deploy_state.GetRootModelId(), submodel_id, target_node_id);
        GEEVENT("[Transfer][Submodel] in deploying success, root_model_id = %u, submodel_id = %u, target_node_id = %d",
                deploy_state.GetRootModelId(), submodel_id, target_node_id);
        return SUCCESS;
      });
      deploy_futures.emplace_back(std::move(fut));
    }
  }
  std::vector<Status> future_rets;
  for (size_t i = 0U; i < deploy_futures.size(); ++i) {
    future_rets.emplace_back(deploy_futures[i].get());
  }
  for (size_t i = 0U; i < future_rets.size(); ++i) {
    GE_CHK_STATUS_RET(future_rets[i], "Failed to transfer models");
  }
  return SUCCESS;
}

Status FlowModelSender::TransferModel(int32_t node_id,
                                      const DeployState &deploy_state,
                                      const PneModelPtr &model,
                                      const ModelBufferData model_buff) {
  GELOGD("[Transfer][Submodel] start, model_name = [%s]", model->GetModelName().c_str());
  std::string path = GetModelFilePath(deploy_state, model->GetModelName());
  GE_CHK_STATUS_RET_NOLOG(TransferFile(node_id, path, model_buff.data.get(), model_buff.length));
  GELOGD("[Transfer][Submodel] succeeded, model_name = [%s]", model->GetModelName().c_str());
  return SUCCESS;
}

Status FlowModelSender::TransferFile(int32_t target_node_id,
                                     const std::string &path,
                                     const void *content,
                                     size_t size) {
  GELOGI("[Download] start, node_id = %d, path = %s, size = %lu",
         target_node_id, path.c_str(), size);
  size_t remaining_size = size;
  size_t offset = 0;
  const size_t block_size = 2 * 1024 * 1024;  // 2M
  deployer::DeployerRequest request;
  request.set_type(deployer::kTransferFile);
  auto download_request = request.mutable_transfer_file_request();
  GE_CHECK_NOTNULL(download_request);
  download_request->set_path(path);
  download_request->set_eof(false);
  auto *buffer = static_cast<const uint8_t *>(content);
  while (remaining_size > 0) {
    size_t size_to_send = std::min(block_size, remaining_size);
    remaining_size -= size_to_send;
    download_request->set_content(buffer + offset, size_to_send);
    download_request->set_eof(remaining_size == 0);
    deployer::DeployerResponse response;
    GE_CHK_STATUS_RET(DeployerProxy::GetInstance().SendRequest(target_node_id, request, response),
                      "[TransferFile] failed to send request, node_id = %d, path = %s, offset = %zu",
                      target_node_id,
                      path.c_str(),
                      offset);
    if (response.error_code() != SUCCESS) {
      GELOGE(FAILED,
             "[TransferFile] failed, node_id = %d, path = %s, error code = %u, error message = %s",
             target_node_id, path.c_str(), response.error_code(), response.error_message().c_str());
      return FAILED;
    }
    offset += size_to_send;
    GELOGD("[TransferFile] succeeded, node_id = %d, path = %s, progress: %zu/%zu",
           target_node_id, path.c_str(), offset, size);
  }
  GELOGI("[TransferFile] succeeded, node_id = %d, path = %s, total size = %zu",
         target_node_id, path.c_str(), size);
  return SUCCESS;
}

Status FlowModelSender::GetDeviceInfo(int32_t node_id, int32_t device_id, int32_t device_type,
                                      DeployPlan::DeviceInfo &deploy_device_info) {
  auto device_info = ResourceManager::GetInstance().GetDeviceInfo(node_id, device_id, device_type);
  GE_CHECK_NOTNULL(device_info);
  deploy_device_info = DeployPlan::DeviceInfo(device_info->GetDeviceType(),
                                              device_info->GetNodeId(),
                                              device_info->GetDeviceId());
  return SUCCESS;
}

Status FlowModelSender::TransferPreDeploy(const SendInfo &send_info,
                                          ExchangeRoute &local_route,
                                          deployer::FlowRoutePlan &remote_plan) const {
  // 1. build deploy plan
  int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId();
  const auto &node_config = Configurations::GetInstance().GetLocalNode();
  DeployPlan::DeviceInfo local_device(CPU, local_node_id, 0);
  // AI server head device is arbitary npu device.
  if (send_info.node_id != local_node_id) {
    for (const auto &device_config : node_config.device_list) {
      if (device_config.device_type == NPU) {
        local_device = DeployPlan::DeviceInfo(NPU, local_node_id, device_config.device_id);
        break;
      }
    }
  }
  std::vector<DeployPlan::DeviceInfo> remote_devices;
  for (const int32_t &device_id : send_info.device_ids) {
    DeployPlan::DeviceInfo remote_device;
    GE_CHK_STATUS_RET(GetDeviceInfo(send_info.node_id, device_id, NPU, remote_device),
                      "Failed to get device info, device id = %d.", send_info.node_id);
    remote_devices.emplace_back(remote_device);
  }
  DeployPlan deploy_plan;
  GE_CHK_STATUS_RET(HeterogeneousDeployPlanner().BuildTransferPlan(local_device, remote_devices, deploy_plan),
                    "Failed to build transfer deploy plan.");
  GELOGD("Transfer deploy plan built successfully");

  // 2. resolve local route plan
  deployer::FlowRoutePlan local_plan;
  GE_CHK_STATUS_RET(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan, local_node_id, local_plan),
                    "Resolve local route plan failed.");
  GELOGD("Local flow route plan resolved successfully");

  // 3. resolve remote route plan
  if (send_info.node_id != local_node_id) {
    GE_CHK_STATUS_RET(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan, send_info.node_id, remote_plan),
                      "Failed to resolve remote plan.");
    GELOGD("Remote flow route plan resolved successfully");
  }

  auto &exchange_service = HeterogeneousExchangeService::GetInstance();
  HeterogeneousExchangeDeployer deployer(exchange_service, local_plan,
                                  DeployContext::LocalContext().GetFlowGwClientManager());
  GE_CHK_STATUS_RET_NOLOG(deployer.Deploy(local_route));
  return SUCCESS;
}

std::string FlowModelSender::GetModelFilePath(const DeployState &deploy_state, const std::string &model_name) {
  std::string file_name = ProcessUtils::NormalizePath(model_name);
  replace(file_name.begin(), file_name.end(), '/', '_');
  replace(file_name.begin(), file_name.end(), '\\', '_');
  replace(file_name.begin(), file_name.end(), '.', '_');
  std::string path = deploy_state.GetRelativeWorkingDir() +
      file_name + "_" + std::to_string(deploy_state.GetSubmodelId(model_name)) + ".om";
  GELOGI("Get model file path[%s] success, model name = %s", path.c_str(), model_name.c_str());
  return path;
}

Status FlowModelSender::GetSavedFilePath(const DeployPlan::SubmodelInfo &submodel_info,
                                              const std::string &model_file_path,
                                              std::string &saved_model_path) {
  if (submodel_info.model->GetIsBuiltinModel()) {
    saved_model_path = kUdfBuildInFuncNamePrefix;
    return SUCCESS;
  }
  if (submodel_info.device_info.GetNodeId() != ResourceManager::GetInstance().GetLocalNodeId()) {
    saved_model_path = model_file_path;
    return SUCCESS;
  }
  saved_model_path = submodel_info.model->GetSavedModelPath();
  if ((saved_model_path.empty()) && (submodel_info.model->GetModelType() == PNE_ID_UDF)) {
    GELOGE(FAILED, "Saved model file path must be not empty in current version."
           "Please generate cache based on current compiler version.");
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace ge
