/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "session/inner_session.h"

#include <map>
#include <memory>
#include <vector>

#include "analyzer/analyzer.h"
#include "adx_datadump_server.h"
#include "common/checker.h"
#include "acl/acl_rt.h"
#include "common/dump/dump_properties.h"
#include "common/dump/dump_manager.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/helper/model_helper.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "common/context/local_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "runtime/mem.h"
#include "api/aclgrph/option_utils.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_init.h"
#include "common/model/external_allocator_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/load/graph_loader.h"
#include "common/platform_info_util/platform_info_util.h"
#include <api/gelib/gelib.h>
#include "common/memory/tensor_trans_utils.h"
#include "register/core_num_utils.h"

namespace ge {
void CopyGeOutputsMemToUserOutputs(const rtStream_t stream, const std::vector<GeTensor> &ge_outputs,
                                   std::vector<Tensor> &outputs) {
  // if alloc output memory by external allocator, should copy to user.
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator == nullptr) {
    return;
  }

  if (outputs.size() == 0U) {
    outputs.reserve(ge_outputs.size());
    for (size_t i = 0UL; i < ge_outputs.size(); i++) {
      outputs.emplace_back(TensorAdapter::AsTensor(ge_outputs[i]));
      GELOGI("Return outputs memory malloc by external allocator success, mem:%p, size:%u", outputs[i].GetData(),
             outputs[i].GetSize());
    }
  }
}
namespace {
constexpr int32_t kDumpStatus = 0;
constexpr int32_t kDecimalSystem = 10;
constexpr int32_t kSocVersionLen = 50;

Status CheckReuseMemoryOption(const std::map<std::string, std::string> &options) {
  auto iter = options.find(OPTION_EXEC_DISABLE_REUSED_MEMORY);
  if (iter != options.end()) {
    if (iter->second == "0") {
      GELOGD("%s=0, reuse memory is open", OPTION_EXEC_DISABLE_REUSED_MEMORY);
    } else if (iter->second == "1") {
      GELOGD("%s=1, reuse memory is close", OPTION_EXEC_DISABLE_REUSED_MEMORY);
    } else {
      GELOGE(PARAM_INVALID, "[CheckReuse][MemoryOption]option %s=%s is invalid",
             OPTION_EXEC_DISABLE_REUSED_MEMORY, iter->second.c_str());
      const auto readable_name = ge::GetContext().GetReadableName(OPTION_EXEC_DISABLE_REUSED_MEMORY);
      std::string reason = readable_name + " only support 0 or 1";
      REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({readable_name.c_str(), iter->second.c_str(), reason.c_str()}));
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CheckAutoTuneMode(const std::map<std::string, std::string> &options) {
  auto option_key = options.find("ge.autoTuneMode");
  if (option_key != options.end() && !option_key->second.empty()) {
    const auto readable_name = ge::GetContext().GetReadableName("ge.autoTuneMode");
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({readable_name.c_str(), option_key->second.c_str(),
                                   "The Auto Tune function has been discarded. Please use the AOE tool for tuning."}));
    GELOGE(
        FAILED,
        "[Check][Param]Options[%s] unsupport, The Auto Tune function has been discarded. Please use the AOE tool for "
        "tuning.",
        option_key->first.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status CheckOpPrecisionMode(const std::map<std::string, std::string> &options) {
  auto iter = options.find(ge::OP_PRECISION_MODE);
  if (iter != options.end() && !iter->second.empty() && !ge::CheckInputPathValid(iter->second)) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::OP_PRECISION_MODE.c_str(), iter->second.c_str(), "path is not found"}));
    GELOGE(PARAM_INVALID, "[Check][OP_PRECISION_MODE] %s not found", iter->second.c_str());
    return FAILED;
  }
  if (iter != options.end()) {
    GELOGI("Option set successfully, option = %s, value=%s",
           ge::OP_PRECISION_MODE.c_str(), iter->second.c_str());
  }
  return CheckPrecisionModeParamValid(options);
}

void SetSessionDeviceId() {
  std::string str_session_device_id;
  if (GetContext().GetOption("ge.session_device_id", str_session_device_id) == SUCCESS) {
    GELOGI("Option session device id has set, value is %s.", str_session_device_id.c_str());
    try {
      const uint32_t session_device_id = static_cast<uint32_t>(std::stoi(str_session_device_id.c_str()));
      GetContext().SetCtxDeviceId(session_device_id);
    } catch (...) {
      GELOGW("Option session device id is invalid, value is %s.", str_session_device_id.c_str());
    }
  }
}

}

static std::mutex mutex_;  // BuildGraph and RunGraph use
bool InnerSession::is_dump_server_inited_ = false;
InnerSession::InnerSession(uint64_t session_id, const std::map<std::string, std::string> &options)
    : is_initialized_(false), session_id_(session_id), options_(options) {}

Status InnerSession::InitializeVarManager() {
  constexpr uint32_t version = static_cast<uint32_t>(SessionVersion::ClOUD_VERSION);
  constexpr uint32_t DEFAULT_JOB_ID = 0;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  const Status ret =
      VarManager::Instance(session_id_)->Init(version, session_id_, GetContext().DeviceId(), DEFAULT_JOB_ID);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][VarManager] failed.");
    REPORT_INNER_ERR_MSG("E19999", "VarManager init failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
  }
  return ret;
}

Status InnerSession::Initialize() {
  if (is_initialized_) {
    GELOGW("[InnerSession:%lu] session already initialize.", session_id_);
    return SUCCESS;
  }
  user_graphs_manager_ = MakeShared<UserGraphsManager>(graph_manager_);
  if (user_graphs_manager_ == nullptr) {
    return MEMALLOC_FAILED;
  }
  user_hybrid_graph_manager_ = MakeShared<UserHybridGraphManager>(*user_graphs_manager_);
  if (user_hybrid_graph_manager_ == nullptr) {
    return MEMALLOC_FAILED;
  }
  GE_CHK_STATUS_RET(CoreNumUtils::ParseAicoreNumFromOption(options_));

  const std::map<std::string, std::string>::const_iterator it = options_.find(ge::SOC_VERSION);
  if (it == options_.cend()) {
    char version[kSocVersionLen] = {0};
    rtError_t rt_ret = rtGetSocVersion(version, kSocVersionLen);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
        REPORT_INNER_ERR_MSG("E19999", "rtGetSocVersion failed.");
        GELOGE(rt_ret, "[Get][SocVersion]rtGetSocVersion failed");
        return FAILED;)
    GELOGI("Succeeded in getting SOC_VERSION[%s] from runtime in InnerSession::Initialize.", version);
    options_.insert(std::make_pair(ge::SOC_VERSION, version));
  }

  logLevel_ = static_cast<uint8_t>(dlog_getlevel(GE_MODULE_NAME, nullptr));
  // If the global options and the session options are duplicated, the session options is preferred.
  auto all_options = options_;
  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    all_options.insert(GetMutableGlobalOptions().cbegin(), GetMutableGlobalOptions().cend());
  }

  GE_ASSERT_SUCCESS(CheckAutoTuneMode(all_options));

  Status ret = CheckReuseMemoryOption(all_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[CheckReuse][MemoryOption] failed, [InnerSession:%lu].", session_id_);
    return ret;
  }

  GE_ASSERT_SUCCESS(CheckOpPrecisionMode(all_options));

  // Check option modify_mixlist
  if (ge::CheckModifyMixlistParamValid(all_options) != ge::SUCCESS) {
    return FAILED;
  }
  GE_ASSERT_SUCCESS(CheckOptionValidValues(all_options, OPTION_FEATURE_BASE_REFRESHABLE, kFeatureMapRefreshOptions));
  GE_ASSERT_SUCCESS(CheckOptionValidValues(all_options, OPTION_CONST_LIFECYCLE, kConstLifecycleOptions));
  GE_ASSERT_SUCCESS(CheckOptionValidThreshold(all_options, OPTION_HOST_SCHEDULING_MAX_THRESHOLD));
  GE_ASSERT_SUCCESS(CheckOptionValidValues(all_options, TILING_SCHEDULE_OPTIMIZE, kStateOptions));
  GE_ASSERT_GRAPH_SUCCESS(CheckOptimizationOptionValid(all_options));

  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption({});
  SetSessionDeviceId();
  GE_CHK_STATUS_RET(aclrtSetDevice(static_cast<int32_t>(GetContext().DeviceId())), "Set device failed.");

  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.GetHardwareInfo(options_), "[Get][Hardware]InnerSession Initialize: Get hardware info failed.");

  DumpProperties dump_properties;
  GE_CHK_STATUS_RET(dump_properties.InitByOptions(), "Init dump properties failed.");
  GE_CHK_STATUS_RET(AddDumpProperties(dump_properties), "[Add][DumpProperties] failed.");

  ret = InnerInitialize();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager initialize failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  GE_ASSERT_SUCCESS(InitializeVarManager());
  is_initialized_ = true;
  return SUCCESS;
}

Status InnerSession::Finalize() {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!is_initialized_) {
    GELOGW("[InnerSession:%lu] session does not initialize.", session_id_);
    return SUCCESS;
  }
  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption({});
  if (user_hybrid_graph_manager_ != nullptr) {
    user_hybrid_graph_manager_->Finalize();
  }
  if (user_graphs_manager_ != nullptr) {
    user_graphs_manager_->Finalize();
  }
  Status ret = InnerFinalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager Finalize failed, InnerSession:%lu.", session_id_);
  }

  is_initialized_ = false;
  // release analyzer saved info(Session Level)
  Analyzer::GetInstance()->DestroySessionJsonObject(session_id_);

  GE_CHK_RT(aclrtResetDevice(static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_STATUS_RET(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
  VarManagerPool::Instance().RemoveVarManager(session_id_);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(session_id_);
  SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id_);
  SessionMemAllocator<ActiveMemoryAllocator>::Instance().RemoveAllocator(session_id_);
  return ret;
}

Status InnerSession::InnerInitialize() {
  Status ret = model_executor_.Initialize(options_, session_id_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphExecutor] failed, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphExecutor initialize failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  ret = graph_manager_.Initialize(options_, &model_executor_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager initialize failed, InnerSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }
  // model executor thread should run later, in case graph_manager init failed.
  model_executor_.StartRunThread();
  return SUCCESS;
}

Status InnerSession::InnerFinalize() {
  Status ret = graph_manager_.Finalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphManager] failed, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager Finalize failed, InnerSession:%lu.", session_id_);
  }

  ret = model_executor_.Finalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphExecutor] failed, InnerSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphExecutor Finalize failed, InnerSession:%lu.", session_id_);
  }
  return SUCCESS;
}

Status InnerSession::AddGraph(uint32_t graph_id, const Graph &graph) {
  std::map<std::string, std::string> options;
  return AddGraph(graph_id, graph, options);
}

Status InnerSession::AddGraph(uint32_t graph_id, const Graph &graph,
                              const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);

  for (const auto &item : options) {
    GELOGI("GE option: %s, value: %s, innerSession:%lu, graphid: %u.", item.first.c_str(), item.second.c_str(),
           session_id_, graph_id);
  }

  auto iter = options.find("ge.autoTuneMode");
  if ((iter != options.end()) && (!iter->second.empty())) {
    const auto readable_name = ge::GetContext().GetReadableName("ge.autoTuneMode");
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({readable_name.c_str(), iter->second.c_str(),
                                   "The Auto Tune function has been discarded. Please use the AOE tool for tuning."}));
    GELOGE(
        FAILED,
        "[Check][Param]Options[%s] unsupport, The Auto Tune function has been discarded. Please use the AOE tool for "
        "tuning.",
        iter->first.c_str());
    return FAILED;
  }
  GE_ASSERT_SUCCESS(SetSessionGraphId(graph, session_id_, graph_id));
  UpdateGlobalSessionContext();

  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  Status ret = user_hybrid_graph_manager_->AddGraph(graph_id, graph, options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][Graph] failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager AddGraph failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    return ret;
  }
  const uint32_t device_id = GetContext().DeviceId();
  GELOGD("The device id is %u", device_id);
  (void)ProfilingInit::Instance().SetDeviceIdByModelId(graph_id, device_id);
  ProfilingManager::Instance().SetGraphIdToDeviceMap(graph_id, device_id);
  GELOGI("[InnerSession:%lu] Add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerSession::SetSessionGraphId(const Graph &graph, uint64_t session_id, uint32_t graph_id) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  std::string session_graph_id = std::to_string(session_id) + "_" + std::to_string(graph_id);
  if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Set graph session_graph_id attr failed.");
  } else {
    GELOGD("Set graph session_graph_id attr to [%s]", session_graph_id.c_str());
  }
  for (auto sub_graph : compute_graph->GetAllSubgraphs()) {
    (void)AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  return SUCCESS;
}

Status InnerSession::LoadGraph(const uint32_t graph_id,
                               const std::map<AscendString, AscendString> &options, void *stream) {
  GELOGI("[InnerSession] Load graph by graph_id=%u, stream = %p", graph_id, stream);
  UpdateGlobalSessionContext();
  GE_ASSERT_NOTNULL(user_graphs_manager_);
  const auto ret = user_graphs_manager_->LoadGraph(graph_id, options, stream);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Graph] Failed, graph_id:%u.", graph_id);
    return ret;
  }
  return SUCCESS;
}

Status InnerSession::AddGraphWithCopy(uint32_t graph_id, const Graph &graph,
                                      const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  GE_ASSERT_SUCCESS(SetSessionGraphId(graph, session_id_, graph_id));
  UpdateGlobalSessionContext();
  Status ret = graph_manager_.AddGraphWithCopy(graph_id, graph, options, domi::GetContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][Graph] failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager AddGraphWithCopy failed, InnerSession:%lu graphid: %u.", session_id_, graph_id);
    return ret;
  }

  GELOGI("[InnerSession:%lu] add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerSession::RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  GELOGI("[InnerSession:%lu] Run graph on session, graph_id=%u.", session_id_, graph_id);
  if (std::unique_lock<std::mutex> lock{mutex_, std::try_to_lock}) {
    UpdateGlobalSessionContext();
    return graph_manager_.RunGraph(graph_id, inputs, outputs, GetSessionId());
  } else {
    GELOGE(GE_SESS_ALREADY_RUNNING, "[Run][Graph]failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                       "RunGraph failed because mutex try_lock false, InnerSession:%lu, graph_id=%u.",
                       session_id_, graph_id);
    return GE_SESS_ALREADY_RUNNING;
  }
}

Status InnerSession::RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
                               std::vector<gert::Tensor> &outputs) {
  GELOGI("[InnerSession:%lu] Run graph on session, graph_id=%u.", session_id_, graph_id);
  if (std::unique_lock<std::mutex> lock{mutex_, std::try_to_lock}) {
    UpdateGlobalSessionContext();
    return graph_manager_.RunGraph(graph_id, inputs, outputs);
  } else {
    GELOGE(GE_SESS_ALREADY_RUNNING, "[Run][Graph]failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                       "RunGraph failed because mutex try_lock false, InnerSession:%lu, graph_id=%u.",
                       session_id_, graph_id);
    return GE_SESS_ALREADY_RUNNING;
  }
}

Status InnerSession::ExecuteGraphWithStreamAsync(uint32_t graph_id, const rtStream_t stream,
                                                 const std::vector<gert::Tensor> &inputs,
                                                 std::vector<gert::Tensor> &outputs) {
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Execute graph with stream begin, session id = %lu, graph id = %u,"
          "stream = %p, input size = %zu, output size = %zu",
          session_id_, graph_id, stream, inputs.size(), outputs.size());
  }
  GE_ASSERT_NOTNULL(user_graphs_manager_);
  const Status res = user_graphs_manager_->ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs, session_id_);
  if (res != SUCCESS) {
    GELOGE(res, "[Execute][GraphWithStreamAsync]failed,"
            "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager ExecuteGrapWithStreamhAsync failed,"
                      "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    return res;
  }

  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Execute graph with stream async success, session id = %lu, graph id = %u, stream = %p.",
          session_id_, graph_id, stream);
  }

  return SUCCESS;
}

Status InnerSession::RunGraphWithStreamAsync(uint32_t graph_id, rtStream_t stream,
                                             const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Run graph with stream begin, session id = %lu, graph id = %u,"
          "stream = %p, input size = %zu, output size = %zu",
          session_id_, graph_id, stream, inputs.size(), outputs.size());
  }
  UpdateGlobalSessionContext();
  std::vector<GeTensor> ge_inputs;
  ge_inputs.reserve(inputs.size());
  for (auto &item : inputs) {
    ge_inputs.emplace_back(TensorAdapter::AsGeTensorShared(item));
  }
  std::vector<GeTensor> ge_outputs;
  ge_outputs.reserve(outputs.size());
  for (auto &item : outputs) {
    ge_outputs.emplace_back(TensorAdapter::AsGeTensorShared(item));
  }
  const Status res = graph_manager_.RunGraphWithStreamAsync(graph_id, stream, session_id_, ge_inputs, ge_outputs);
  if (res != SUCCESS) {
    GELOGE(res, "[Run][GraphWithStreamAsync]failed,"
            "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager RunGrapWithStreamhAsync failed,"
                      "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    return res;
  }

  // if alloc output memory by external allocator, should return to user.
  CopyGeOutputsMemToUserOutputs(stream, ge_outputs, outputs);
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Run graph with stream async success, session id = %lu, graph id = %u, stream = %p.",
          session_id_, graph_id, stream);
  }
  return SUCCESS;
}

Status InnerSession::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  const auto device_id = GetContext().DeviceId();
  GELOGD("Remove device id %u", device_id);
  (void)ProfilingInit::Instance().UnsetDeviceIdByModelId(graph_id, device_id);
  UpdateGlobalSessionContext();
  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  const Status ret = user_hybrid_graph_manager_->RemoveGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Remove][Graph] failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RemoveGraph failed, InnerSession:%lu, graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] Remove graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerSession::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<std::string, ge::Tensor> &)> &callback) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption({});
  auto callback_func = [callback] (uint32_t graph_id, const std::map<AscendString, gert::Tensor>& params_list) {
    std::map<std::string, ge::Tensor> para_map;
    for (const auto &item : params_list) {
      ge::Tensor tensor;
      if (ge::TensorTransUtils::GertTensor2Tensor(item.second, tensor) != SUCCESS) {
        GELOGE(FAILED, "convert ge::Tensor to gert::Tensor failed");
        return FAILED;
      }
      para_map[item.first.GetString()] = std::move(tensor);
    }
    return callback(graph_id, para_map);
  };
  const Status ret = graph_manager_.RegisterCallBackFunc(key, callback_func);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Register][CallBackFunc] failed, InnerSession:%lu register %s.", session_id_, key.c_str());
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RegisterCallBackFunc failed, InnerSession:%lu register %s.",
                      session_id_, key.c_str());
    return ret;
  }

  GELOGI("[InnerSession:%lu] register %s callback function success.", session_id_, key.c_str());
  return SUCCESS;
}

Status InnerSession::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, ge::Tensor> &)> &callback) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption({});
  auto callback_func = [callback] (uint32_t graph_id, const std::map<AscendString, gert::Tensor>& params_list) {
    std::map<AscendString, ge::Tensor> para_map;
    for (const auto &item : params_list) {
      ge::Tensor tensor;
      if (ge::TensorTransUtils::GertTensor2Tensor(item.second, tensor) != SUCCESS) {
        GELOGE(FAILED, "convert ge::Tensor to gert::Tensor failed");
        return FAILED;
      }
      para_map[item.first] = std::move(tensor);
    }
    return callback(graph_id, para_map);
  };
  const Status ret = graph_manager_.RegisterCallBackFunc(key, callback_func);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Register][CallBackFunc] failed, InnerSession:%lu register %s.", session_id_, key.c_str());
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RegisterCallBackFunc failed, InnerSession:%lu register %s.",
                      session_id_, key.c_str());
    return ret;
  }

  GELOGI("[InnerSession:%lu] register %s callback function success.", session_id_, key.c_str());
  return SUCCESS;
}

Status InnerSession::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption({});
  const Status ret = graph_manager_.RegisterCallBackFunc(key, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Register][CallBackFunc] failed, InnerSession:%lu register %s.", session_id_, key.c_str());
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RegisterCallBackFunc failed, InnerSession:%lu register %s.",
                      session_id_, key.c_str());
    return ret;
  }

  GELOGI("[InnerSession:%lu] register %s callback function success.", session_id_, key.c_str());
  return SUCCESS;
}

Status InnerSession::BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs) {
  GELOGI("[InnerSession:%lu] Build graph on session, graph_id=%u.", session_id_, graph_id);
  std::vector<ge::GeTensor> ge_inputs;
  for (auto const &input : inputs) {
    std::vector<int64_t> input_dims;
    (void)std::transform(input.dims.begin(), input.dims.end(), std::back_inserter(input_dims),
                         [](int64_t x) -> int64_t { return x; });
    GeShape input_shape(input_dims);
    GeTensorDesc input_tensor_desc;
    input_tensor_desc.SetShape(input_shape);
    input_tensor_desc.SetDataType(static_cast<ge::DataType>(input.data_type));
    ge_inputs.emplace_back(input_tensor_desc);
  }
  UpdateGlobalSessionContext();
  GeRootModelPtr ge_root_model = nullptr;
  Status ret = graph_manager_.BuildGraph(graph_id, ge_inputs, ge_root_model, session_id_, true);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Graph] failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager BuildGraph failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] build graph success, graph_id=%u.", session_id_, graph_id);
  return ret;
}

Status InnerSession::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  GELOGI("[InnerSession:%lu] build graph on session, graph_id=%u.", session_id_, graph_id);

  std::vector<ge::GeTensor> ge_inputs;
  for (const auto &input : inputs) {
    ge_inputs.emplace_back(TensorAdapter::AsGeTensor(input));
  }
  UpdateGlobalSessionContext();
  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  GeRootModelPtr ge_root_model = nullptr;
  Status ret = user_hybrid_graph_manager_->BuildGraph(graph_id, ge_inputs, session_id_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Graph] failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager BuildGraph failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] build graph success, graph_id=%u.", session_id_, graph_id);
  return ret;
}

Status InnerSession::RunGraphAsync(uint32_t graph_id, std::vector<gert::Tensor> &&inputs,
                                   const RunAsyncCallbackV2 &callback) {
  GELOGI("[InnerSession:%lu] run graph on session, graph_id=%u.", session_id_, graph_id);
  UpdateGlobalSessionContext();

  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  Status ret = user_hybrid_graph_manager_->RunGraphAsync(graph_id, std::move(inputs), session_id_, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][GraphAsync]failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RunGraphAsync failed, InnerSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerSession:%lu] run graph async submit success, graph_id=%u.", session_id_, graph_id);
  return ret;
}

const GraphManager &InnerSession::getGraphManagerObj() const { return graph_manager_; }

void InnerSession::UpdateGlobalSessionContext() const {
  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  }
  GetThreadLocalContext().SetSessionOption(options_);
  GetContext().SetSessionId(session_id_);
  SetTrainFlagOption();
  SetRtSocVersion();
}

bool InnerSession::IsGraphNeedRebuild(uint32_t graph_id) {
  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  return user_hybrid_graph_manager_->IsGraphNeedRebuild(graph_id);
}

Status InnerSession::GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables) {
  const auto &instance = VarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(instance);
  return instance->GetAllVariables(all_variables);
}

Status InnerSession::GenCheckPointGraph(const std::map<std::string, GeTensorDesc> &all_variables, Graph &graph) {
  return graph_manager_.GenCheckPointGraph(all_variables, graph);
}

Status InnerSession::SaveVariables(const Graph &graph, const std::vector<std::string> &var_names,
                                   const std::vector<Tensor> &outputs, std::vector<Tensor> &var_values) {
  return graph_manager_.SaveVariables(graph, var_names, outputs, var_values);
}

Status InnerSession::AddDumpProperties(const DumpProperties &dump_properties) {
  if (!is_dump_server_inited_) {
    if ((dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen())) {
      GE_IF_BOOL_EXEC(AdxDataDumpServerInit() != kDumpStatus,
                      GELOGE(PARAM_INVALID, "[Init][AdxDataDumpServer] failed, session_id:%lu.", session_id_);
                      return PARAM_INVALID)
      GELOGI("Init adx data dump server success");
      is_dump_server_inited_ = true;
    }
  }
  if ((!dump_properties.GetEnableDump().empty()) || (!dump_properties.GetEnableDumpDebug().empty())) {
    // if dump option set, add dump property
    GE_IF_BOOL_EXEC(DumpManager::GetInstance().AddDumpProperties(session_id_, dump_properties) != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Add][DumpProperties] failed, session_id:%lu.", session_id_);
                    return PARAM_INVALID);
    if (DumpManager::GetInstance().CheckIfAclDumpSet()) {
      GELOGW("Set dump by options and acl simultaneously, will use the option setting.");
    }
    DumpManager::GetInstance().ClearAclDumpSet();
  }
  return SUCCESS;
}

Status InnerSession::RemoveDumpProperties() {
  DumpManager::GetInstance().RemoveDumpProperties(session_id_);
  if (is_dump_server_inited_ && DumpManager::GetInstance().GetDumpPropertiesMap().empty()) {
    GE_IF_BOOL_EXEC(AdxDataDumpServerUnInit() != kDumpStatus,
                    GELOGE(PARAM_INVALID, "[UnInit][AdxDataDumpServer] failed, session_id:%lu.", session_id_);
                    REPORT_INNER_ERR_MSG("E19999", "RemoveDumpProperties failed because AdxDataDumpServerUnInit failed,"
                                       "session_id:%lu", session_id_);
                    return PARAM_INVALID)
    GELOGI("UnInit adx data dump server success");
    is_dump_server_inited_ = false;
  }
  return SUCCESS;
}

void InnerSession::SetRtSocVersion() {
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  const auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(ge::SOC_VERSION);
  if (it != global_options.end()) {
    rtError_t rt_ret = rtSetSocVersion(it->second.c_str());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Set soc version %s failed. ret:0x%X", it->second.c_str(), rt_ret);
    }
    GELOGI("Set soc version %s success.", it->second.c_str());
  }
}

void InnerSession::SetTrainFlagOption() {
  auto train_flag = false;
  std::string run_mode;
  if ((GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == SUCCESS) && (!run_mode.empty())) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, kDecimalSystem)) >= TRAIN) {
      train_flag = true;
    }
  }
  domi::GetContext().train_flag = train_flag;
  GELOGI("train flag is %d in session", train_flag);
}

Status InnerSession::CompileGraph(uint32_t graph_id, const vector<ge::Tensor> &inputs) {
  UpdateGlobalSessionContext();
  GE_ASSERT_NOTNULL(user_graphs_manager_);
  const auto ret = user_graphs_manager_->CompileGraph(graph_id, session_id_, inputs);
  GE_CHK_STATUS_RET(ret, "[Compile][Graph]Failed, InnerSession:%lu, graph_id:%u, inputs size:%zu.",
                    session_id_, graph_id, inputs.size());
  GELOGI("[InnerSession:%lu]Compile graph success, session_id:%lu, graph_id:%u, inputs size:%zu.",
         session_id_, graph_id, inputs.size());
  return SUCCESS;
}

Status InnerSession::GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary) {
  UpdateGlobalSessionContext();
  GE_ASSERT_NOTNULL(user_graphs_manager_);
  return user_graphs_manager_->GetCompiledGraphSummary(graph_id, summary);
}

Status InnerSession::SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  UpdateGlobalSessionContext();
  const auto ret = graph_manager_.SetConstMemoryBase(graph_id, memory, size);
  GE_CHK_STATUS_RET(ret, "[Set][Memory]Failed, InnerSession:%lu, graph_id:%u, memory:%p, size:%zu.",
                    session_id_, graph_id, memory, size);
  GELOGI("[InnerSession:%lu]Set graph const memory base success, graph_id:%u, memory:%p, size:%zu.",
         session_id_, graph_id, memory, size);
  return SUCCESS;
}

Status InnerSession::UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  UpdateGlobalSessionContext();
  const auto ret = graph_manager_.UpdateFeatureMemoryBase(graph_id, memory, size);
  GE_CHK_STATUS_RET(ret, "[Update][Memory]Failed, InnerSession:%lu, graph_id:%u, memory:%p, size:%zu.",
                    session_id_, graph_id, memory, size);
  GELOGI("[InnerSession:%lu]Update graph feature memory base success, graph_id:%u, memory:%p, size:%zu.",
         session_id_, graph_id, memory, size);
  return SUCCESS;
}

Status InnerSession::SetGraphFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory,
                                                    size_t size) {
  UpdateGlobalSessionContext();
  const auto ret = graph_manager_.SetFixedFeatureMemoryBase(graph_id, type, memory, size);
  GE_CHK_STATUS_RET(ret, "[Set][Memory]Failed, InnerSession:%lu, graph_id:%u, type:%d, memory:%p, size:%zu.",
                    session_id_, graph_id, type, memory, size);
  return SUCCESS;
}

Status InnerSession::UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  UpdateGlobalSessionContext();
  const auto ret = graph_manager_.UpdateRefreshableFeatureMemoryBase(graph_id, memory, size);
  GE_CHK_STATUS_RET(ret, "[Update][Memory]Failed, InnerSession:%lu, graph_id:%u, memory:%p, size:%zu.",
                    session_id_, graph_id, memory, size);
  GELOGI("[InnerSession:%lu]Update graph refreshable feature memory base success, graph_id:%u, memory:%p, size:%zu.",
         session_id_, graph_id, memory, size);
  return SUCCESS;
}

Status InnerSession::RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const {
  GE_ASSERT_NOTNULL(stream, "stream is nullptr, session_id:%u.", session_id_);
  GE_ASSERT_NOTNULL(allocator, "allocator is nullptr, session_id:%u.", session_id_);

  return graph_manager_.RegisterExternalAllocator(stream, allocator);
}

Status InnerSession::UnregisterExternalAllocator(const void * const stream) const {
  GE_ASSERT_NOTNULL(stream, "stream is nullptr, session_id:%u.", session_id_);
  return graph_manager_.UnregisterExternalAllocator(stream);
}

Status InnerSession::PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len) const {
  const auto &ordered_graph_ids = graph_manager_.GetOrderedGraphIds();
  GE_ASSERT_TRUE(!(ordered_graph_ids.empty()), "[PaRemapped][Graph]there is no graph, InnerSession:%ld", session_id_);
  Status ret;
  std::vector<std::pair<uint64_t, uint64_t>> cross_ranges;
  for (const GraphId graph_id : ordered_graph_ids) {
    ret = graph_manager_.PaRemapped(graph_id, va, new_pa, len, cross_ranges);
    if (ret == FAILED) {
      GELOGW("[PaRemapped] va[%lu] pa[%lu] can not remap, graph id:%u.", va, new_pa, graph_id);
      return FAILED;
    }
  }
  return CheckPaRemappedResult(va, len, cross_ranges);
}

Status InnerSession::CheckPaRemappedResult(const uint64_t va, const uint64_t len,
                                           std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) const {
  if (cross_ranges.empty()) {
    return PARAM_INVALID;
  }
  std::vector<std::pair<uint64_t, uint64_t>> merged_ranges;
  // 排序的参数使用了lambda表达式
  std::sort(
      cross_ranges.begin(), cross_ranges.end(),
      [](const std::pair<uint64_t, uint64_t> &a, const std::pair<uint64_t, uint64_t> &b) { return a.first < b.first; });

  // 第一个区间就可以放进结果集里，后面如果重叠，在merged_ranges上直接合并
  merged_ranges.push_back(cross_ranges[0]);
  for (size_t i = 1UL; i < cross_ranges.size(); i++) {
    // 发现重叠区间或者区间相邻 地址为整数，相邻区间也合并
    if (merged_ranges.back().second >= cross_ranges[i].first ||
        merged_ranges.back().second + 1UL == cross_ranges[i].first) {
      // 合并区间，只更新右边界就好，因为merged_ranges.back()的左边界一定是最小值，因为我们按照左边界排序的
      merged_ranges.back().second = std::max(merged_ranges.back().second, cross_ranges[i].second);
    } else {
      merged_ranges.push_back(cross_ranges[i]);  // 区间不重叠
    }
  }

  if ((merged_ranges.size() == 1UL) && (merged_ranges[0].first == va) &&
      (merged_ranges[0].second == (va + len - 1UL))) {
    return SUCCESS;
  }
  return PARAM_INVALID;
}
Status InnerSession::ForkGraph(uint32_t origin_graph_id, uint32_t forked_graph_id) {
  return graph_manager_.ForkGraph(origin_graph_id, forked_graph_id);
}

Status InnerSession::GetCompiledFlag(uint32_t graph_id, bool &flag) const {
  flag = false;
  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  return user_hybrid_graph_manager_->GetCompiledFlag(graph_id, flag);
}

Status InnerSession::SetCompiledFlag(uint32_t graph_id, bool flag) {
  GE_ASSERT_NOTNULL(user_hybrid_graph_manager_);
  return user_hybrid_graph_manager_->SetCompiledFlag(graph_id, flag);
}

std::shared_ptr<DFlowSessionImpl> InnerSession::GetDFlowSession() const {
  return dflow_session_impl_;
}

void InnerSession::SetDFlowSession(const std::shared_ptr<DFlowSessionImpl> &dflow_session_impl) {
  dflow_session_impl_ = dflow_session_impl;
}

Status InnerSession::GetRunGraphMode(uint32_t graph_id, RunGraphMode &mode) const {
  return graph_manager_.GetRunGraphMode(graph_id, mode);
}

Status InnerSession::SetRunGraphMode(uint32_t graph_id, const RunGraphMode &mode) {
  return graph_manager_.SetRunGraphMode(graph_id, mode);
}

Status InnerSession::GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer) {
  GELOGI("Start to get the compiled model. graph_id: %u.", graph_id);
  UpdateGlobalSessionContext();
  return graph_manager_.GetCompiledModel(graph_id, model_buffer);
}

bool InnerSession::GetBuildFlag(uint32_t graph_id) const {
  return graph_manager_.GetBuildFlag(graph_id);
}

bool InnerSession::GetLoadFlag(uint32_t graph_id) const {
  return graph_manager_.GetLoadFlag(graph_id);
}
}  // namespace ge
