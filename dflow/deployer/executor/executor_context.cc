/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/executor_context.h"
#include <fstream>
#include "rt_error_codes.h"
#include "nlohmann/json.hpp"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "executor/cpu_sched_event_dispatcher.h"
#include "executor/proxy_dynamic_model_executor.h"
#include "executor/dynamic_model_executor.h"
#include "cpu_tasks.h"
#include "framework/executor/ge_executor.h"
#include "graph/ge_context.h"
#include "graph/manager/mem_manager.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "common/checker.h"
#include "common/thread_pool/thread_pool.h"
#include "common/helper/model_parser_base.h"
#include "framework/common/helper/model_helper.h"
#include "toolchain/prof_api.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "external/graph/types.h"
#include "dflow/base/model/flow_model_om_loader.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
namespace {
constexpr int32_t kUserUnsetPriority = -1;
constexpr const char *OPTION_FLOAT_OVERFLOW_MODE = "ge.exec.float_overflow_mode";
constexpr const char *OPTION_FLOAT_OVERFLOW_MODE_SATURATION = "saturation";
constexpr const char *OPTION_FLOAT_OVERFLOW_MODE_INFNAN = "inf_nan";
const std::string kGatherDequeue = "gatherDequeue";
constexpr int32_t kDefaultAttachQueueTimeout = 3000;  // 3s
constexpr const char_t *kAttrNameInvokedByBuiltIn = "_dflow_invoked_by_built_in";
constexpr int32_t kDefaultHostDeviceId = 64;
}  // namespace

class SharedMemoryManager : public MemManager {
 public:
  static SharedMemoryManager &GetInstance() {
    static SharedMemoryManager instance;
    return instance;
  }
  uint8_t *MallocMemory(rtMemType_t memory_type,
                        const std::string &purpose,
                        const std::string &memory_key,
                        size_t memory_size,
                        uint32_t device_id) override {
    if (memory_type != RT_MEMORY_HBM) {
      return MemManager::MallocMemory(memory_type, purpose, memory_key, memory_size, device_id);
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = var_mem_bases_.find(memory_key);
      if (it != var_mem_bases_.cend()) {
        return it->second;
      }
    }
    return MemManager::MallocMemory(memory_type, purpose, memory_key, memory_size, device_id);
  }

  Status FreeMemory(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (memory_type != RT_MEMORY_HBM || var_mem_bases_.count(memory_key) == 0) {
      return MemManager::FreeMemory(memory_type, memory_key, device_id);
    }
    return SUCCESS;
  }
  uint8_t *GetMemoryBase(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) override {
    if (memory_type != RT_MEMORY_HBM) {
      return MemManager::GetMemoryBase(memory_type, memory_key, device_id);
    }
    return GetMemoryAddr(memory_type, memory_key, device_id);
  }
  uint8_t *GetMemoryAddr(rtMemType_t memory_type, const std::string &memory_key, uint32_t device_id) override {
    if (memory_type != RT_MEMORY_HBM) {
      return MemManager::GetMemoryAddr(memory_type, memory_key, device_id);
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto it = var_mem_bases_.find(memory_key);
    if (it == var_mem_bases_.cend()) {
      GELOGW("MemoryAllocator::GetMemoryAddr failed, memory_key[%s] was does not exist", memory_key.c_str());
      return nullptr;
    }
    return it->second;
  }

  void AddSharedMemory(const std::string &key, uint8_t *memory_base) {
    std::lock_guard<std::mutex> lk(mu_);
    var_mem_bases_[key] = memory_base;
  }

 private:
  std::mutex mu_;
  std::map<std::string, uint8_t *> var_mem_bases_;
};

ExecutorContext::ModelHandle *ExecutorContext::GetOrCreateModelHandle(uint32_t root_model_id, uint32_t model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  auto &submodels = model_handles_[root_model_id];
  const auto &it = submodels.find(model_id);
  if (it != submodels.cend()) {
    return it->second.get();
  }
  auto new_handle = MakeUnique<ModelHandle>();
  if (new_handle == nullptr) {
    return nullptr;
  }
  if (!model_handles_[root_model_id].emplace(model_id, std::move(new_handle)).second) {
    GELOGE(FAILED, "[Add][Model] failed, model already added, sub_model_id = %u", model_id);
    return nullptr;
  }
  ExecutorContext::ModelHandle *handle = model_handles_[root_model_id][model_id].get();
  return handle;
}

ExecutorContext &ExecutorContext::LocalContext() {
  static ExecutorContext context;
  return context;
}

PneModelPtr ExecutorContext::GetLocalModel(uint32_t root_model_id, uint32_t model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  const auto &root_it = models_.find(root_model_id);
  if (root_it == models_.cend()) {
    return nullptr;
  }
  const auto &sub_it = root_it->second.find(model_id);
  if (sub_it == root_it->second.cend()) {
    return nullptr;
  }
  return sub_it->second;
}

void ExecutorContext::AddLocalModel(uint32_t root_model_id, uint32_t model_id, const PneModelPtr &model) {
  std::lock_guard<std::mutex> lk(mu_);
  models_[root_model_id][model_id] = model;
}

void ExecutorContext::RemoveLocalModel(uint32_t root_model_id) {
  std::lock_guard<std::mutex> lk(mu_);
  models_.erase(root_model_id);
}

Status ExecutorContext::Initialize() const {
  return SUCCESS;
}

void ExecutorContext::Finalize() const {}

void ExecutorContext::SetBaseDir(const std::string &base_dir) {
  base_dir_ = base_dir;
}

Status ExecutorContext::ParseModel(uint32_t root_model_id, uint32_t model_id, const std::string &model_path) {
  GELOGI("Parse model begin, root_model_id = %u, submodel_id = %u", root_model_id, model_id);
  auto handle = GetOrCreateModelHandle(root_model_id, model_id);
  GE_CHECK_NOTNULL(handle);
  auto model = LocalContext().GetLocalModel(root_model_id, model_id);
  if (model != nullptr) {
    auto graph_model = std::dynamic_pointer_cast<GraphModel>(model);
    GE_CHECK_NOTNULL(graph_model, "cast to graph model failed, model name=%s", model->GetModelName().c_str());
    handle->SetModelData(graph_model->GetModelData());
    handle->SetRootGraph(graph_model->GetRootGraph());
    GELOGI("Parse model from cache success");
    return SUCCESS;
  }

  GE_CHK_BOOL_RET_STATUS((model_path.find("..") == std::string::npos), FAILED,
                         "Model path[%s] is invalid, include relative path.", model_path.c_str());
  std::string model_real_path = RealPath(model_path.c_str());
  GE_CHK_BOOL_RET_STATUS((!model_real_path.empty()), FAILED, "Model path[%s] is invalid.", model_path.c_str());
  GE_CHK_STATUS_RET_NOLOG(handle->ParseModel(model_path));
  GELOGI("Parse model from om success, path = %s, root_model_id = %u, submodel_id = %u",
         model_path.c_str(), root_model_id, model_id);
  return SUCCESS;
}

std::unique_ptr<std::istream> ExecutorContext::CreateInputStream(const std::string &path) const {
  return MakeUnique<std::ifstream>(path, std::ios::in | std::ios::binary);
}

Status ExecutorContext::GetModel(uint32_t root_model_id,
                                 std::map<uint32_t, std::unique_ptr<ModelHandle>> *&submodel_map) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = model_handles_.find(root_model_id);
  if (it == model_handles_.end()) {
    GELOGE(FAILED, "[Get][Model] failed, root model id = %u.", root_model_id);
    return FAILED;
  }

  submodel_map = &(it->second);
  return SUCCESS;
}

ExecutorContext::ModelHandle::~ModelHandle() {
  GELOGD("Begin to destruct model_handle.");
  if (loaded_) {
    (void) UnloadModel();
  }
  if (model_data_from_cache_ && model_data_.model_data != nullptr) {
    delete[] static_cast<char_t *>(model_data_.model_data);
    model_data_.model_data = nullptr;
    model_data_from_cache_ = false;
  }
}

Status ExecutorContext::SyncSharedVarManager(const deployer::ExecutorRequest &request) const {
  auto &sync_var_manager_request = request.sync_var_manager_message();
  const auto &var_manager_info = sync_var_manager_request.var_manager_info();
  auto session_id = var_manager_info.session_id();
  GELOGI("Init var manager, session id is %zu.", session_id);

  const std::vector<rtMemType_t> mem_type{RT_MEMORY_HBM, RT_MEMORY_P2P_DDR};
  auto &var_mem_manager = SharedMemoryManager::GetInstance();
  GE_CHK_STATUS_RET(var_mem_manager.Initialize(mem_type),
                    "[Init][MemManager] MemoryAllocatorManager initialize failed.");
  int32_t device_id = -1;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  GELOGD("SyncSharedVarManager get device_id[%d] success", device_id);
  for (const auto &shared_content_desc : sync_var_manager_request.shared_content_descs()) {
    GeTensorDesc tensor_desc;
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&shared_content_desc.tensor_desc(), tensor_desc);
  }

  GELOGI("Init var manager successfully, session_id = %lu.", session_id);
  return SUCCESS;
}

void ExecutorContext::UpdateGraphOptions(const std::string &key, const std::string &value) {
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[key] = value;
  GetThreadLocalContext().SetGraphOption(graph_options);
}

void ExecutorContext::UpdateOptions(const deployer::Options &options) {
  std::map<std::string, std::string> global_options = GetThreadLocalContext().GetAllGlobalOptions();
  for (const auto &item : options.global_options()) {
    global_options[item.first] = item.second;
    GELOGI("Insert global option, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }
  GetThreadLocalContext().SetGlobalOption(global_options);

  std::map<std::string, std::string> session_options = GetThreadLocalContext().GetAllSessionOptions();
  for (const auto &item : options.session_options()) {
    session_options[item.first] = item.second;
    GELOGI("Insert session option, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }
  GetThreadLocalContext().SetSessionOption(session_options);

  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  for (const auto &item : options.graph_options()) {
    graph_options[item.first] = item.second;
    GELOGI("Insert graph option, key = %s, value = %s.", item.first.c_str(), item.second.c_str());
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
}

Status ExecutorContext::SetOpTimeout() {
  std::string op_wait_timeout_str;
  (void)GetContext().GetOption(OP_WAIT_TIMEOUT, op_wait_timeout_str);
  if (!op_wait_timeout_str.empty()) {
    int64_t op_wait_timeout_sec = 0;
    GE_CHK_STATUS_RET(ConvertToInt64(op_wait_timeout_str, op_wait_timeout_sec),
                      "Convert result[%s] to int32 failed.", op_wait_timeout_str.c_str());
    GE_CHECK_LE(op_wait_timeout_sec, static_cast<int64_t>(UINT32_MAX));
    auto ret = rtSetOpWaitTimeOut(static_cast<uint32_t>(op_wait_timeout_sec));
    GELOGI("Set rtSetOpWaitTimeOut[%s], ret = %d.", op_wait_timeout_str.c_str(), ret);
  }

  std::string op_execute_timeout_str;
  (void)GetContext().GetOption(OP_EXECUTE_TIMEOUT, op_execute_timeout_str);
  if (!op_execute_timeout_str.empty()) {
    int64_t op_execute_timeout_sec = 0;
    GE_CHK_STATUS_RET(ConvertToInt64(op_execute_timeout_str, op_execute_timeout_sec),
                      "Convert result[%s] to int32 failed.", op_execute_timeout_str.c_str());
    GE_CHECK_LE(op_execute_timeout_sec, static_cast<int64_t>(UINT32_MAX));
    auto ret = rtSetOpExecuteTimeOut(static_cast<uint32_t>(op_execute_timeout_sec));
    GELOGI("Set rtSetOpExecuteTimeOut[%s], ret = %d.", op_execute_timeout_str.c_str(), ret);
  }
  return SUCCESS;
}

Status ExecutorContext::SetDeviceSatMode() {
  std::string sat_mode;
  (void)GetContext().GetOption(OPTION_FLOAT_OVERFLOW_MODE, sat_mode);
  if (!sat_mode.empty()) {
    static std::map<std::string, rtFloatOverflowMode_t> mode_transfer = {
      {OPTION_FLOAT_OVERFLOW_MODE_SATURATION, RT_OVERFLOW_MODE_SATURATION},
      {OPTION_FLOAT_OVERFLOW_MODE_INFNAN, RT_OVERFLOW_MODE_INFNAN}};
    const auto &it = mode_transfer.find(sat_mode);
    if (it != mode_transfer.cend()) {
      GE_CHK_RT_RET(rtSetDeviceSatMode(it->second));
      GEEVENT("rtSetDeviceSatMode success, mode = %d.", static_cast<int32_t>(it->second));
    }
  }
  return SUCCESS;
}

Status ExecutorContext::ParseModelEschedPriority(const deployer::ExecutorRequest_LoadModelRequest &request,
                                                 ModelHandle &handle) const {
  auto attrs = request.attrs();
  int32_t process_priority = kUserUnsetPriority;
  int32_t event_priority = kUserUnsetPriority;
  if (attrs.contains(ATTR_NAME_ESCHED_PROCESS_PRIORITY)) {
    GE_CHK_STATUS_RET(ConvertToInt32(attrs[ATTR_NAME_ESCHED_PROCESS_PRIORITY], process_priority),
                      "Failed to parse model esched process priority.");
  }
  if (attrs.contains(ATTR_NAME_ESCHED_EVENT_PRIORITY)) {
    GE_CHK_STATUS_RET(ConvertToInt32(attrs[ATTR_NAME_ESCHED_EVENT_PRIORITY], event_priority),
                      "Failed to parse model esched event priority.");
  }
  handle.SetEschedPriority(process_priority, event_priority);
  GELOGD("[ModelEschedPriority]: process priority[%d], event priority[%d].", process_priority, event_priority);
  return SUCCESS;
}

Status ExecutorContext::ParseInputAlignAttrs(const deployer::ExecutorRequest_LoadModelRequest &request,
                                             InputAlignAttrs &input_align_attrs) {
  if (!request.has_input_align_attrs()) {
    return SUCCESS;
  }
  const auto &proto_input_align_attrs = request.input_align_attrs();
  input_align_attrs.align_max_cache_num = proto_input_align_attrs.align_max_cache_num();
  input_align_attrs.align_timeout = proto_input_align_attrs.align_timeout();
  input_align_attrs.drop_when_not_align = proto_input_align_attrs.drop_when_not_align();
  return SUCCESS;
}

Status ExecutorContext::ParseModel(const deployer::ExecutorRequest_LoadModelRequest &request) {
  GE_TIMESTAMP_START(ParseModel);
  auto root_model_id = request.root_model_id();
  auto sub_model_id = request.model_id();
  auto model_path = base_dir_ + request.model_path();
  const auto &saved_model_path = request.saved_model_file_path();
  const auto &model_ins_name = request.model_instance_name();
  std::string real_path = RealPath(saved_model_path.c_str());
  if (!real_path.empty()) {
    // can use compile cache
    model_path = real_path;
  }
  std::string trace_log = "Parse model[" + model_ins_name + "] from path[" + model_path + "]";
  GE_CHK_STATUS_RET(ParseModel(root_model_id, sub_model_id, model_path), "Failed to parse model");
  GE_TIMESTAMP_EVENT_END(ParseModel, trace_log.c_str());
  return SUCCESS;
}

Status ExecutorContext::AttachQueues(const deployer::ExecutorRequest_LoadModelRequest &request) {
  std::map<int32_t, std::set<uint32_t>> attach_queue_list;
  for (const auto &queue_attr : request.model_queues_attrs().input_queues_attrs()) {
    if ((queue_attr.device_type() == static_cast<int32_t>(NPU)) &&
        (queue_attr.queue_id() != UINT32_MAX)) {
      attach_queue_list[queue_attr.device_id()].emplace(queue_attr.queue_id());
    }
  }
  for (const auto &queue_attr : request.model_queues_attrs().output_queues_attrs()) {
    if ((queue_attr.device_type() == static_cast<int32_t>(NPU)) &&
        (queue_attr.queue_id() != UINT32_MAX)) {
      attach_queue_list[queue_attr.device_id()].emplace(queue_attr.queue_id());
    }
  }
  for (const auto &queue_attr : request.status_queues().output_queues_attrs()) {
    if ((queue_attr.device_type() == static_cast<int32_t>(NPU)) &&
        (queue_attr.queue_id() != UINT32_MAX)) {
      attach_queue_list[queue_attr.device_id()].emplace(queue_attr.queue_id());
    }
  }
  for (const auto &it : attach_queue_list) {
    auto device_id = it.first;
    const auto &queue_ids = it.second;
    const auto ret = rtMemQueueInit(device_id);
    if (ret != RT_ERROR_NONE && ret != ACL_ERROR_RT_REPEATED_INIT) {
      return RT_ERROR_TO_GE_STATUS(ret);
    }
    for (auto queue_id : queue_ids) {
      GE_CHK_RT_RET(rtMemQueueAttach(device_id, queue_id, kDefaultAttachQueueTimeout));
    }
  }
  return SUCCESS;
}

Status ExecutorContext::LoadModel(const deployer::ExecutorRequest_LoadModelRequest &request) {
  ModelHandle::LoadParam param;
  for (const auto &input_queue : request.model_queues_attrs().input_queues_attrs()) {
    QueueAttrs in_queue_attrs;
    in_queue_attrs.queue_id = input_queue.queue_id();
    in_queue_attrs.device_type = input_queue.device_type();
    in_queue_attrs.device_id = input_queue.device_id();
    in_queue_attrs.logic_id = input_queue.global_logic_id();
    param.input_queues.emplace_back(in_queue_attrs);
  }
  for (const auto &output_queue : request.model_queues_attrs().output_queues_attrs()) {
    QueueAttrs out_queue_attrs;
    out_queue_attrs.queue_id = output_queue.queue_id();
    out_queue_attrs.device_type = output_queue.device_type();
    out_queue_attrs.device_id = output_queue.device_id();
    out_queue_attrs.logic_id = output_queue.global_logic_id();
    param.output_queues.emplace_back(out_queue_attrs);
  }

  param.input_fusion_offsets = std::move(std::vector<int32_t>(request.input_fusion_offsets().cbegin(),
                                                              request.input_fusion_offsets().cend()));
  param.replica_num = request.replica_num();
  param.replica_idx = request.replica_idx();
  param.model_uuid = request.model_uuid();
  std::vector<QueueAttrs> status_output_queues_attrs;
  for (const auto &status_output_queue : request.status_queues().output_queues_attrs()) {
    QueueAttrs status_output_queue_attrs;
    status_output_queue_attrs.queue_id = status_output_queue.queue_id();
    status_output_queue_attrs.device_type = status_output_queue.device_type();
    status_output_queue_attrs.device_id = status_output_queue.device_id();
    status_output_queue_attrs.logic_id = status_output_queue.global_logic_id();
    status_output_queues_attrs.emplace_back(status_output_queue_attrs);
  }
  auto root_model_id = request.root_model_id();
  auto sub_model_id = request.model_id();
  const auto &model_path = base_dir_ + request.model_path();
  GE_CHK_STATUS_RET(ParseInputAlignAttrs(request, param.input_align_attrs),
                    "Failed to parse input align attrs, model_id = %d, sub_model_id = %d", root_model_id, sub_model_id);
  if (status_output_queues_attrs.size() == 1U) {
    param.status_output_queue = status_output_queues_attrs[0];
    param.need_report_status = true;
  } else {
    param.need_report_status = false;
  }
  param.is_dynamic_sched = request.is_dynamic_sched();
  param.is_head = request.is_head();
  GELOGI("DynamicSched model load info: model_uuid=%u, status output queue=%u, need report status=%d,"
         " is dynamic sched=%d, is_head=%d", param.model_uuid, param.status_output_queue.queue_id,
         param.need_report_status, param.is_dynamic_sched, param.is_head);

  auto handle = GetOrCreateModelHandle(root_model_id, sub_model_id);
  GE_CHECK_NOTNULL(handle);
  handle->SetExecuteTimes(request.execute_times());
  handle->SetIsDynamicProxyControlled(request.is_dynamic_proxy_controlled());
  handle->SetScope(request.scope());
  handle->SetEnableExceptionCatch(request.enable_exception_catch());
  GE_CHK_STATUS_RET_NOLOG(ParseModelEschedPriority(request, *handle));
  GELOGD("Begin to load model, root_model_id = %u, submodel_id = %u, path = %s, is_dynamic_proxy_controlled = %d.",
         root_model_id, sub_model_id, model_path.c_str(), static_cast<int32_t>(request.is_dynamic_proxy_controlled()));
  GE_CHK_STATUS_RET(handle->LoadModel(param), "Failed to load model");
  GELOGD("Success to load model, proot_model_id = %u, submodel_id = %u, path = %s, is_dynamic_proxy_controlled = %d.",
         root_model_id, sub_model_id, model_path.c_str(), static_cast<int32_t>(request.is_dynamic_proxy_controlled()));
  return SUCCESS;
}

Status ExecutorContext::UpdateProfInfo(const deployer::ExecutorRequest &request) {
  const auto &prof_info_request = request.update_prof_message();
  const std::string &prof_data = prof_info_request.prof_data();
  const bool is_prof_start = prof_info_request.is_prof_start();
  static bool is_start_status = false;
  GELOGI("Is prof start[%d], current status %d.", is_prof_start, is_start_status);
  if (is_start_status == is_prof_start) {
    return SUCCESS;
  }
  MsprofConfig param = {};
  param.devNums = 1;
  param.devIdList[0] = kDefaultHostDeviceId;
  auto ret = strcpy_s(param.dumpPath, sizeof(param.dumpPath), std::to_string(kDefaultHostDeviceId).c_str());
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "strcpy_s ret:%d", ret);
  ret = strcpy_s(param.sampleConfig, sizeof(param.sampleConfig), prof_data.c_str());
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "strcpy_s prof_data ret:%d", ret);
  ret = is_prof_start ? MsprofStart(MSPROF_CTRL_INIT_PURE_CPU, reinterpret_cast<const char *>(&param), sizeof(param)) :
                        MsprofStop(MSPROF_CTRL_INIT_PURE_CPU, reinterpret_cast<const char *>(&param), sizeof(param));
  GE_CHK_BOOL_RET_STATUS(ret == MSPROF_ERROR_NONE, FAILED,
                         "MsprofStart/MsprofStop failed, is prof start : %d", is_prof_start);
  is_start_status = is_prof_start;
  return SUCCESS;
}

Status ExecutorContext::ModelHandle::GetModelData(ModelData &model_data) {
  model_data = model_data_;
  return SUCCESS;
}

void ExecutorContext::ModelHandle::SetModelData(const ModelData &model_data) {
  model_data_ = model_data;
}

Status ExecutorContext::ModelHandle::GetRootGraph(ComputeGraphPtr &root_graph) {
  root_graph = root_graph_;
  return SUCCESS;
}

void ExecutorContext::ModelHandle::SetRootGraph(const ComputeGraphPtr &root_graph) {
  root_graph_ = root_graph;
}

Status ExecutorContext::ModelHandle::GetModelRuntimeIdOrHandle(std::vector<uint32_t> &davinci_model_runtime_ids,
  std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) {
  if (inner_model_id_ != UINT32_MAX) {
    uint32_t model_runtime_id = UINT32_MAX;
    GE_CHK_STATUS_RET(GeExecutor::GetRuntimeModelId(inner_model_id_, model_runtime_id),
      "Failed to get runtime model id.");
    davinci_model_runtime_ids.emplace_back(model_runtime_id);
    return SUCCESS;
  }
  if (dynamic_model_executor_ != nullptr) {
    if (is_dynamic_proxy_controlled_) {
      const ProxyDynamicModelExecutor *const proxyDynamicModel =
        dynamic_cast<const ProxyDynamicModelExecutor *>(dynamic_model_executor_.get());
      GE_CHECK_NOTNULL(proxyDynamicModel);
      davinci_model_runtime_ids.emplace_back(proxyDynamicModel->GetRuntimeModelId());
    }
    dynamic_model_handles.emplace_back(this);
    return SUCCESS;
  }
  GELOGE(FAILED, "This model is not davinci model or dynamic model.");
  return FAILED;
}

Status ExecutorContext::ModelHandle::ClearModel(const int32_t clear_type) {
  return dynamic_model_executor_->ClearModel(clear_type);
}
Status ExecutorContext::ModelHandle::ExceptionNotify(uint32_t type, uint64_t trans_id) {
  return dynamic_model_executor_->ExceptionNotify(type, trans_id);
}
Status ExecutorContext::ModelHandle::ParseModel(const std::string &model_path) {
  GELOGD("Begin to to parse model[%s].", model_path.c_str());
  ge::ModelData model{};
  GE_CHK_STATUS_RET(ModelParserBase::LoadFromFile(model_path.c_str(), 0, model),
                    "Load model from file[%s] failed.", model_path.c_str());
  GE_CHK_STATUS_RET(FlowModelOmLoader::TransModelDataToComputeGraph(model, root_graph_), "Failed to trans Modeldata to ComputeGraph");
  GE_CHECK_NOTNULL(root_graph_, "load root graph is null");
  model_data_ = model;
  model_data_from_cache_ = true;
  return SUCCESS;
}

Status ExecutorContext::ModelHandle::LoadModel(const LoadParam &param) {
  if (loaded_) {
    GELOGW("Already loaded");
    return SUCCESS;
  }
  ModelData model_data;
  ComputeGraphPtr root_graph;
  GE_CHK_STATUS_RET_NOLOG(GetRootGraph(root_graph));
  GE_CHK_STATUS_RET_NOLOG(GetModelData(model_data));
  GE_CHK_STATUS_RET(DoLoadModel(model_data, root_graph, param), "Do load model failed.");
  loaded_ = true;
  return SUCCESS;
}

Status ExecutorContext::ModelHandle::DoLoadModel(const ModelData &model_data,
                                                 const ComputeGraphPtr &root_graph,
                                                 const LoadParam &params) {
  return DoLoadModelWithQ(model_data, root_graph, params);
}

Status ExecutorContext::ModelHandle::UnloadModel() {
  if (dynamic_model_executor_ != nullptr) {
    dynamic_model_executor_->UnloadModel();
    dynamic_model_executor_->Finalize();
    dynamic_model_executor_.reset();
  } else {
    GE_CHK_STATUS_RET(DoUnloadModel(inner_model_id_),
                      "[Unload][Model] failed, model_id = %u", inner_model_id_);
    GELOGD("[Unload][Model] success, model_id = %u", inner_model_id_);
    inner_model_id_ = UINT32_MAX;
  }
  loaded_ = false;
  return SUCCESS;
}

Status ExecutorContext::ModelHandle::CheckAicpuAlignTask(const InputAlignAttrs &input_align_attrs) {
  // input_align_attrs.cache_num will be set 0 until aicpusd support gather dequeue task
  if (input_align_attrs.align_max_cache_num == 0U) {
    GELOGD("No need to check gather dequeue task result cache number is 0.");
    return SUCCESS;
  }
  bool is_gather_supported = false;
  GE_CHK_STATUS_RET(CpuTasks::ExecuteCheckSupported(kGatherDequeue, is_gather_supported));
  GE_ASSERT_TRUE(is_gather_supported, "Gather dequeue is not supported in current version. "
                 "Please update software or unset input align attrs.");
  return SUCCESS;
}

Status ExecutorContext::ModelHandle::DoLoadModelWithQ(const ModelData &model_data,
                                                      const ComputeGraphPtr &root_graph,
                                                      const LoadParam &params) {
  ModelQueueParam model_queue_param;
  model_queue_param.group_total_count = params.replica_num;
  model_queue_param.group_index = params.replica_idx;
  for (const auto &queue : params.input_queues) {
    model_queue_param.input_queues.emplace_back(queue.queue_id);
    model_queue_param.input_queues_attrs.emplace_back(queue);
  }
  for (const auto &queue : params.output_queues) {
    model_queue_param.output_queues.emplace_back(queue.queue_id);
    model_queue_param.output_queues_attrs.emplace_back(queue);
  }

  model_queue_param.input_fusion_offsets = params.input_fusion_offsets;
  model_queue_param.model_uuid = params.model_uuid;
  model_queue_param.status_output_queue = params.status_output_queue;
  model_queue_param.is_dynamic_sched = params.is_dynamic_sched;
  model_queue_param.need_report_status = params.need_report_status;
  model_queue_param.is_head = params.is_head;
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DATA_FLOW_UDF_INVOKED_NN, is_invoked_nn_);
  // invoked nn no need align
  if (!is_invoked_nn_) {
    model_queue_param.input_align_attrs = params.input_align_attrs;
  }
  bool dflow_invoked_by_built_in = false;
  (void)AttrUtils::GetBool(root_graph, kAttrNameInvokedByBuiltIn,
                           dflow_invoked_by_built_in);
  model_queue_param.need_check_inputs = !dflow_invoked_by_built_in;
  model_queue_param.need_model_config = true;
  model_queue_param.mark_dump_step = true;
  model_queue_param.io_with_tensor_desc = true;
  model_queue_param.copy_inputs_for_non_zero_copy = true;
  bool is_host = GetContext().GetHostExecFlag();
  bool is_dynamic = false;
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic);
  is_dynamic = (is_dynamic || (root_graph->GetGraphUnknownFlag()));
  GE_TIMESTAMP_START(LoadModel);
  // dynamic model loaded on host
  if (is_dynamic_proxy_controlled_) {
    GELOGD("Load proxy dynamic model, is_dynamic = %d, is_dynamic_proxy_controlled = %d.",
           static_cast<int32_t>(is_dynamic), static_cast<int32_t>(is_dynamic_proxy_controlled_));
    if (IsEnableExceptionCatch()) {
      GE_CHK_STATUS_RET(CpuTasks::CheckSupportExceptionNotify(), "aicpu does not support exception catch.");
    }
    // dynamic proxy and static model check kernel supported by task.
    GE_CHK_STATUS_RET(CheckAicpuAlignTask(model_queue_param.input_align_attrs),
                      "[Check][Align] attrs failed.");
    dynamic_model_executor_ = CreateProxyDynamicModelExecutor();
    GE_CHECK_NOTNULL(dynamic_model_executor_);
    GE_CHK_STATUS_RET_NOLOG(dynamic_model_executor_->Initialize());
    GE_CHK_STATUS_RET_NOLOG(dynamic_model_executor_->LoadModel(model_data, root_graph, model_queue_param));
    GE_TIMESTAMP_EVENT_END(LoadModel, "LoadDynamicModel");
    GELOGD("[LoadProxyDynamicModel] success");
    return SUCCESS;
  }
  // host cpu executor or dynamic model loaded on device soc
  if (is_dynamic || is_host) {
    GELOGD("Load dynamic model, is_dynamic = %d, is_dynamic_proxy_controlled = %d.",
           static_cast<int32_t>(is_dynamic), static_cast<int32_t>(is_dynamic_proxy_controlled_));
    dynamic_model_executor_ = CreateDynamicModelExecutor(is_host);
    GE_CHECK_NOTNULL(dynamic_model_executor_);
    GE_CHK_STATUS_RET_NOLOG(dynamic_model_executor_->Initialize());
    if (IsEnableExceptionCatch()) {
      GE_CHK_STATUS_RET(dynamic_model_executor_->CheckLocalAicpuSupportExceptionNotify(),
                        "local aicpu does not support exception catch.");
    }
    dynamic_model_executor_->SetModelEschedPriority(esched_process_priority_, esched_event_priority_);
    GE_CHK_STATUS_RET_NOLOG(dynamic_model_executor_->LoadModel(model_data, root_graph, model_queue_param));
    GE_TIMESTAMP_EVENT_END(LoadModel, "LoadDynamicModel");
    GELOGD("[LoadDynamicModel] success");
    return SUCCESS;
  }
  // static model
  aclError ret;
  GELOGD("Load static model, replica_num = %u, replica_idx = %u, is_dynamic = %d, is_dynamic_proxy_controlled = %d.",
         params.replica_num, params.replica_idx, static_cast<int32_t>(is_dynamic),
         static_cast<int32_t>(is_dynamic_proxy_controlled_));
  std::vector<FileConstantMem> external_weight_mem_data{}; 
  GE_CHK_STATUS_RET(DynamicModelExecutor::InitExternalWeightMem(root_graph, external_weight_mem_data), "Failed to init external weright mem.");
  if (params.input_queues.empty() && params.output_queues.empty()) {
    handle_ = aclmdlCreateConfigHandle();
    GE_CHECK_NOTNULL(handle_, "Create acl load config handle failed.");
    GE_CHK_STATUS_RET(DynamicModelExecutor::GenerateLoadConfig(model_data, external_weight_mem_data, handle_));
    ret = aclmdlLoadWithConfig(handle_, &inner_model_id_);
    if (ret != ACL_SUCCESS) {
      GELOGE(FAILED, "Failed to load model");
      (void) aclmdlDestroyConfigHandle(handle_);
      handle_ = nullptr;
      return FAILED;
    }
    GELOGI("Load static model[%u] on success.", inner_model_id_);
  } else {
    GeExecutor executor;
    if (IsEnableExceptionCatch()) {
      GE_CHK_STATUS_RET(CpuTasks::CheckSupportExceptionNotify(), "aicpu not support exception catch.");
    }
    GE_CHK_STATUS_RET(CpuTasks::ExecuteModelEschedPriorityTask(esched_process_priority_, esched_event_priority_));
    GE_CHK_STATUS_RET(CheckAicpuAlignTask(model_queue_param.input_align_attrs),
                      "[Check][Algin] attrs failed.");
    if (!external_weight_mem_data.empty()) {
      model_queue_param.file_constant_mems = &external_weight_mem_data;
    }
    GE_CHK_STATUS_RET(executor.LoadModelWithQueueParam(inner_model_id_, model_data, model_queue_param), "[LoadModelWithQueueParam] failed");
    GE_TIMESTAMP_EVENT_END(LoadModel, "LoadModelWithQ");
  }
  GELOGI("[LoadModelWithQ] success, model_id = %u, model_name = %s",
         inner_model_id_, root_graph->GetName().c_str());
  return SUCCESS;
}

Status ExecutorContext::ModelHandle::DoUnloadModel(uint32_t model_id) {
  if (handle_ != nullptr) {
    (void) aclmdlUnload(model_id);
    (void) aclmdlDestroyConfigHandle(handle_);
    handle_ = nullptr;
  } else {
    GE_CHK_STATUS_RET(GeExecutor().UnloadModel(model_id), "Unload model[%u] faield.", model_id);
  }
  return SUCCESS;
}

std::unique_ptr<DynamicModelExecutor> ExecutorContext::ModelHandle::CreateDynamicModelExecutor(bool is_host) {
  return MakeUnique<DynamicModelExecutor>(is_host);
}

std::unique_ptr<ProxyDynamicModelExecutor> ExecutorContext::ModelHandle::CreateProxyDynamicModelExecutor() {
  return MakeUnique<ProxyDynamicModelExecutor>();
}

void ExecutorContext::ModelHandle::SetExecuteTimes(int32_t execute_times) {
  execute_times_ = execute_times;
}

void ExecutorContext::ModelHandle::SetIsDynamicProxyControlled(const bool is_dynamic_proxy_controlled) {
  is_dynamic_proxy_controlled_ = is_dynamic_proxy_controlled;
}

void ExecutorContext::ModelHandle::SetEschedPriority(int32_t esched_process_priority, int32_t esched_event_priority) {
  esched_process_priority_ = esched_process_priority;
  esched_event_priority_ = esched_event_priority;
}
void ExecutorContext::ModelHandle::SetScope(const std::string &scope) {
  scope_ = scope;
}
const std::string &ExecutorContext::ModelHandle::GetScope() const {
  return scope_;
}
bool ExecutorContext::ModelHandle::IsInvokedNN() const {
  return is_invoked_nn_;
}
void ExecutorContext::ModelHandle::SetEnableExceptionCatch(bool enable_exception_catch) {
  enable_exception_catch_ = enable_exception_catch;
}
bool ExecutorContext::ModelHandle::IsEnableExceptionCatch() const {
  return enable_exception_catch_;
}
}  // namespace ge
