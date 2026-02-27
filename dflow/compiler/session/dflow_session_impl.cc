/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow_session_impl.h"
#include <map>
#include <memory>
#include <vector>
#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include "dflow/executor/flow_model_manager.h"
#include "graph/ge_global_options.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph/utils/tensor_adapter.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "external/ge/ge_api_v2.h"
#include "framework/common/helper/model_helper.h"
#include "external/ge/ge_ir_build.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
std::vector<GeTensor> ToGeTensors(const std::vector<Tensor> &tensors) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.reserve(tensors.size());
  for (auto &item : tensors) {
    ge_tensors.emplace_back(TensorAdapter::AsGeTensor(item));
  }
  return ge_tensors;
}

std::vector<Tensor> ToTensors(const std::vector<GeTensor> &ge_tensors) {
  std::vector<Tensor> tensors;
  tensors.reserve(ge_tensors.size());
  for (auto &item : ge_tensors) {
    tensors.emplace_back(TensorAdapter::AsTensor(item));
  }
  return tensors;
}

Status ConvertStringMap(const std::map<std::string, std::string> &options,
                        std::map<ge::AscendString, ge::AscendString> &ascend_options) {
	for (auto &option_item : options) {
		if (option_item.first.empty()) {
			GELOGE(ge::FAILED, "Construct session failed, option key is empty.");
			REPORT_INNER_ERR_MSG("E19999", "Construct session failed, option key is empty.");
			return FAILED;
		}
		const ge::AscendString &key = ge::AscendString(option_item.first.c_str());
		const ge::AscendString &val = ge::AscendString(option_item.second.c_str());
		ascend_options[key] = val;
	}
  return SUCCESS;
}

class DefaultNpuProcessNodeEngineImpl : public ProcessNodeEngineImpl {
 public:
  explicit DefaultNpuProcessNodeEngineImpl(std::shared_ptr<GeSession> ge_session)
      : ge_session_(std::move(ge_session)) {}
  ~DefaultNpuProcessNodeEngineImpl() override = default;

  Status BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                    const std::map<std::string, std::string> &options, const std::vector<GeTensor> &inputs,
                    PneModelPtr &model) override {
    GELOGD("Build graph begin, graph_id=%u, graph_name=%s.", graph_id, compute_graph->GetName().c_str());
    GE_CHECK_NOTNULL(ge_session_);

    // option
    std::map<ge::AscendString, ge::AscendString> ascend_options;
		GE_CHK_STATUS_RET(ConvertStringMap(options, ascend_options), "Convert string to ascend string map.");

    auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
    GE_CHK_STATUS_RET(ge_session_->AddGraph(graph_id, graph, ascend_options),
                      "Failed to add graph, graph_id=%u, graph_name=%s.", graph_id, compute_graph->GetName().c_str());
    ScopeGuard guard([this, graph_id]() { return ge_session_->RemoveGraph(graph_id); });
    GE_CHK_STATUS_RET(ge_session_->CompileGraph(graph_id, ge::ToTensors(inputs)),
                      "Compile graph failed, graph_id=%u, graph_name=%s.", graph_id, compute_graph->GetName().c_str());
    ge::ModelBufferData model_buffer_data{};
    GE_CHK_STATUS_RET(ge_session_->GetCompiledModel(graph_id, model_buffer_data), "Failed to get built model");
                
    ge::ModelData model_data{};
    model_data.model_data = model_buffer_data.data.get();
    model_data.model_len = model_buffer_data.length;
    model = FlowModelHelper::ToPneModel(model_data, compute_graph);
    GE_CHECK_NOTNULL(model);

    GELOGD("Build graph end, graph_id=%u, graph_name=%s.", graph_id, compute_graph->GetName().c_str());
    return SUCCESS;
  }

 private:
  std::shared_ptr<GeSession> ge_session_;
};

Status EnsureDflowInitialized(const std::map<std::string, std::string> &options) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (ExecutionRuntime::GetInstance() == nullptr) {
    GE_TIMESTAMP_START(InitializeExecutionRuntime);
    GE_CHK_STATUS_RET_NOLOG(ExecutionRuntime::InitHeterogeneousRuntime(options));
    GE_TIMESTAMP_EVENT_END(InitializeExecutionRuntime, "InitializeExecutionRuntime");
  }

  GE_TRACE_START(ProcessNodeEngine);
  Status init_pne_status = ProcessNodeEngineManager::GetInstance().Initialize(options);
  GE_INIT_TRACE_TIMESTAMP_END(ProcessNodeEngine, "InnerInitialize::ProcessNodeEngine");
  if (init_pne_status != SUCCESS) {
    GELOGE(init_pne_status, "[Init][EngineManager]GE process node engine manager initial failed.");
    REPORT_INNER_ERR_MSG("E19999", "Process node engine manager initial failed.");
    return init_pne_status;
  }
  return SUCCESS;
}
}

Status DFlowInitializeInner(const std::map<AscendString, AscendString> &options) {
  std::map<std::string, std::string> str_options;
  for (const auto &option_item : options) {
    if (option_item.first.GetLength() == 0) {
      GELOGE(FAILED, "[Check][Param] DFlowInitializeInner failed, option key is empty.");
      REPORT_INNER_ERR_MSG("E19999", "Check parameter's options invalid, option key is empty.");
      return FAILED;
    }
    std::string key = option_item.first.GetString();
    std::string val = option_item.second.GetString();
    str_options[key] = val;
  }
  GE_CHK_STATUS_RET(EnsureDflowInitialized(str_options), "Failed to init execution runtime");
  return SUCCESS;
}

void DFlowFinalizeInner() {
  (void)ProcessNodeEngineManager::GetInstance().Finalize();
  ExecutionRuntime::FinalizeExecutionRuntime();
}

DFlowSessionImpl::DFlowSessionImpl(uint64_t session_id, const std::map<std::string, std::string> &options)
    : session_id_(session_id), options_(options) {}

DFlowSessionImpl::~DFlowSessionImpl() {
  (void)Finalize();
}

Status DFlowSessionImpl::Initialize(const std::map<std::string, std::string> &options) {
  if (is_initialized_) {
    GELOGI("[DFlowSessionImpl:%lu] session already initialize.", session_id_);
    return SUCCESS;
  }
  
  std::map<ge::AscendString, ge::AscendString> ascend_options;
  GE_CHK_STATUS_RET(ConvertStringMap(options, ascend_options), "Convert string to ascend string map.");
  ge::AscendString graph_key("ge.graph_key");
  ge::AscendString cache_dir("ge.graph_compiler_cache_dir");
  ge::AscendString external_weight("ge.externalWeightDir");
  if (ascend_options.find(graph_key) != ascend_options.end() && ascend_options.find(cache_dir) != ascend_options.end()) {
    if (ascend_options.find(external_weight) != ascend_options.end()) {
      ascend_options[external_weight] = ascend_options[cache_dir];
      ascend_options.erase(graph_key);
      ascend_options.erase(cache_dir);
    }
  }

  ge_session_ = MakeShared<GeSession>(ascend_options);
  GE_CHECK_NOTNULL(ge_session_);

  GE_CHK_STATUS_RET(EnsureDflowInitialized(options_), "Failed to init dflow.");
  auto pneImpl = MakeShared<DefaultNpuProcessNodeEngineImpl>(ge_session_);
  GE_CHECK_NOTNULL(pneImpl);
  GE_CHK_STATUS_RET(dflow_graph_manager_.Initialize(options_, pneImpl), "Failed to init dflow graph manager");
  is_initialized_ = true;
  return SUCCESS;
}

Status DFlowSessionImpl::Finalize() {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!is_initialized_) {
    GELOGW("[DFlowSessionImpl:%lu] session does not initialize.", session_id_);
    return SUCCESS;
  }

  {
    std::lock_guard<std::mutex> model_lock(model_mutex_);
    for (uint32_t model_id : loaded_models_) {
      (void)FlowModelManager::GetInstance().Unload(model_id);
    }
    loaded_models_.clear();
  }
  dflow_graph_manager_.Finalize();

  is_initialized_ = false;
  return SUCCESS;
}

Status DFlowSessionImpl::AddGraph(uint32_t graph_id, const dflow::FlowGraph &graph,
                                  const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  for (const auto &item : options) {
    GELOGI("DFlow option: %s, value: %s, dflowInnerSession:%lu, graph id: %u.", item.first.c_str(), item.second.c_str(),
           session_id_, graph_id);
  }
  const auto &ge_graph = graph.ToGeGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(ge_graph);
  GE_CHECK_NOTNULL(compute_graph);
  const uint64_t ge_session_id = ge_session_->GetSessionId();
  // graph use ge session id.
  compute_graph->SetSessionID(ge_session_id);
  std::string session_graph_id = std::to_string(ge_session_id) + "_" + std::to_string(graph_id);
  (void)AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  for (auto &sub_graph : compute_graph->GetAllSubgraphs()) {
    sub_graph->SetSessionID(ge_session_id);
    (void)AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  UpdateThreadContext(options);
  GE_CHK_STATUS_RET(dflow_graph_manager_.AddGraph(graph_id, ge_graph, options),
      "[Add][Graph] failed, DFlowSessionImpl:%lu graph id: %u.", session_id_, graph_id);
  GELOGI("[DFlowSessionImpl:%lu] Add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status DFlowSessionImpl::AddGraph(uint32_t graph_id, const Graph &graph,
                                  const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  const uint64_t ge_session_id = ge_session_->GetSessionId();
  // graph use ge session id.
  compute_graph->SetSessionID(ge_session_id);
  std::string session_graph_id = std::to_string(ge_session_id) + "_" + std::to_string(graph_id);
  (void)AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  for (auto &sub_graph : compute_graph->GetAllSubgraphs()) {
    sub_graph->SetSessionID(ge_session_id);
    (void)AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  UpdateThreadContext(options);
  GE_CHK_STATUS_RET(dflow_graph_manager_.AddGraph(graph_id, graph, options), 
    "FlowGraphManager AddGraph failed, DFlowInnerSession:%lu graph id: %u.", session_id_, graph_id);
  GELOGI("[DFlowInnerSession:%lu] Add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status DFlowSessionImpl::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  uint32_t model_id = INVALID_MODEL_ID;
  (void)dflow_graph_manager_.GetGraphModelId(graph_id, model_id);
  if (model_id != INVALID_MODEL_ID) {
    (void)FlowModelManager::GetInstance().Unload(model_id);
    loaded_models_.erase(model_id);
  }
  Status ret = dflow_graph_manager_.RemoveGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Remove][Graph] failed, DFlowSessionImpl:%lu graph id: %u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "FlowGraphManager RemoveGraph failed, DFlowSessionImpl:%lu graph id: %u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[DFlowSessionImpl:%lu] Remove graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

FlowModelPtr DFlowSessionImpl::CompileAndLoadGraph(uint32_t graph_id, const std::vector<GeTensor> &inputs) {
  std::lock_guard<std::mutex> lock(build_run_mutex_);
  auto flow_model = dflow_graph_manager_.GetFlowModel(graph_id);
  if (flow_model == nullptr) {
    const auto ret = dflow_graph_manager_.CompileGraph(graph_id, inputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "Dflow graph manager compile graph failed, graph_id=%u.", graph_id);
      return nullptr;
    }
    flow_model = dflow_graph_manager_.GetFlowModel(graph_id);
    if (flow_model == nullptr) {
      GELOGE(FAILED, "Dflow graph manager compile graph success, but flow model is null, graph_id=%u.", graph_id);
      return nullptr;
    }
  }
  if (!dflow_graph_manager_.GetOptionsRunGraphFlag()) {
    GELOGI("RunFlag is false, no need load graph.");
    return flow_model;
  }
  if (flow_model->GetModelId() != INVALID_MODEL_ID) {
    return flow_model;
  }
  GELOGI("Start to load dflow model.");
  uint32_t model_id = INVALID_MODEL_ID;
  Status load_ret = FlowModelManager::GetInstance().LoadFlowModel(model_id, flow_model);
  if (load_ret != SUCCESS) {
    GELOGE(load_ret, "Load flow model failed, graph_id=%u.", graph_id);
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> model_guard(model_mutex_);
    loaded_models_.emplace(model_id);
  }
  GELOGI("Load flow model success, graph_id=%u, model_id=%u.", graph_id, model_id);
  return flow_model;
}

Status DFlowSessionImpl::CompileGraph(uint32_t graph_id, const std::vector<GeTensor> &ge_inputs) {
  UpdateThreadContext(graph_id);
  GE_CHK_STATUS_RET(dflow_graph_manager_.CompileGraph(graph_id, ge_inputs),
                    "[DFlowSessionImpl:%lu] compile graph failed, session_id_, graph_id=%u", graph_id);
  GELOGI("[DFlowSessionImpl:%lu] Compile graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status DFlowSessionImpl::BuildGraph(uint32_t graph_id, const std::vector<Tensor> &inputs) {
  UpdateThreadContext(graph_id);
  std::vector<ge::GeTensor> ge_inputs = ToGeTensors(inputs);
  return BuildGraph(graph_id, ge_inputs);
}

Status DFlowSessionImpl::BuildGraph(uint32_t graph_id, const std::vector<ge::GeTensor> &ge_inputs) {
  UpdateThreadContext(graph_id);
  const auto flow_model = CompileAndLoadGraph(graph_id, ge_inputs);
  if (flow_model == nullptr) {
    GELOGE(FAILED, "[Compile][Load] failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "FlowGraphManager BuildGraph failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    return FAILED;
  }
  GELOGI("[DFlowSessionImpl:%lu] Build graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}


Status DFlowSessionImpl::RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  UpdateThreadContext(graph_id);
  std::vector<ge::GeTensor> ge_inputs = ToGeTensors(inputs);
  const auto flow_model = CompileAndLoadGraph(graph_id, ge_inputs);
  if (flow_model == nullptr) {
    GELOGE(FAILED, "[Compile][Load] failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "FlowGraphManager BuildGraph failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    return FAILED;
  }
  if (!dflow_graph_manager_.GetOptionsRunGraphFlag()) {
    GEEVENT("Skip execute model result of run flag obtained is false.");
    return SUCCESS;
  }
  std::vector<GeTensor> ge_outputs;
  GE_CHK_STATUS_RET(
      FlowModelManager::GetInstance().ExecuteFlowModel(flow_model->GetModelId(), ge_inputs, ge_outputs),
      "execute flow model failed, graph_id=%u, model_id=%u", graph_id, flow_model->GetModelId());
  outputs = ToTensors(ge_outputs);
  GELOGI("run graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status DFlowSessionImpl::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                            const std::vector<Tensor> &inputs, const DataFlowInfo &info, int32_t timeout) {
  UpdateThreadContext(graph_id);
  std::vector<GeTensor> ge_inputs = ToGeTensors(inputs);
  const auto flow_model = CompileAndLoadGraph(graph_id, ge_inputs);
  GE_CHK_BOOL_RET_STATUS(flow_model != nullptr, FAILED,
                         "[Build][Graph] failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
  if (!dflow_graph_manager_.GetOptionsRunGraphFlag()) {
    GEEVENT("Skip loading model result of run flag obtained is false.");
    return SUCCESS;
  }
  auto model_id = flow_model->GetModelId();
  auto heterogeneous_executor = FlowModelManager::GetInstance().GetHeterogeneousModelExecutor(model_id);
  GE_CHECK_NOTNULL(heterogeneous_executor, ", model_id:%u.", model_id);
  auto ret = heterogeneous_executor->FeedData(indexes, ge_inputs, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(FAILED, "[Feed][Data]failed, DFlowSession:%lu, graph_id=%u, model_id=%u.", session_id_, graph_id, model_id);
  } else {
    GELOGI("[DFlowSessionImpl:%lu] feed data flow graph success, graph_id=%u, model_id=%u.", session_id_, graph_id,
           model_id);
  }
  return ret;
}

FlowModelPtr DFlowSessionImpl::GetFlowModel(uint32_t graph_id) const {
  return dflow_graph_manager_.GetFlowModel(graph_id);
}

Status DFlowSessionImpl::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                            const std::vector<FlowMsgPtr> &inputs, int32_t timeout) {
  UpdateThreadContext(graph_id);
  const std::vector<GeTensor> input_no_use = {};
  const auto flow_model = CompileAndLoadGraph(graph_id, input_no_use);
  GE_CHK_BOOL_RET_STATUS(flow_model != nullptr, FAILED,
                         "[Build][Graph] failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
  if (!dflow_graph_manager_.GetOptionsRunGraphFlag()) {
    GEEVENT("Skip loading model result of run flag obtained is false.");
    return SUCCESS;
  }

  auto model_id = flow_model->GetModelId();
  auto heterogeneous_executor = FlowModelManager::GetInstance().GetHeterogeneousModelExecutor(model_id);
  GE_CHECK_NOTNULL(heterogeneous_executor, ", model_id:%u.", model_id);
  auto ret = heterogeneous_executor->FeedFlowMsg(indexes, inputs, timeout);
  GE_CHK_BOOL_RET_STATUS((ret == SUCCESS || ret == ACL_ERROR_GE_REDEPLOYING || ret == ACL_ERROR_GE_SUBHEALTHY),
                         ret, "[Feed][FlowMsg]failed, DFlowSession:%lu, graph_id=%u, model_id=%u.",
                         session_id_, graph_id, model_id);

  GELOGI("[DFlowSession:%lu] feed data flow graph success, graph_id=%u, model_id=%u.", session_id_, graph_id, model_id);
  return ret;
}

Status DFlowSessionImpl::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                             std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout) {
  UpdateThreadContext(graph_id);
  uint32_t model_id = INVALID_MODEL_ID;
  Status ret = dflow_graph_manager_.GetGraphModelId(graph_id, model_id);
  if ((ret != SUCCESS) || (model_id == INVALID_MODEL_ID)) {
    GELOGE(ret, "[Get][model] failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Get model failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    return FAILED;
  }
  std::vector<GeTensor> ge_outputs;
  auto heterogeneous_executor = FlowModelManager::GetInstance().GetHeterogeneousModelExecutor(model_id);
  GE_CHECK_NOTNULL(heterogeneous_executor, ", model_id:%u.", model_id);

  ret = heterogeneous_executor->FetchData(indexes, ge_outputs, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(FAILED, "[Fetch][Data]failed, DFlowSession:%lu, graph_id=%u, model_id=%u.", session_id_, graph_id, model_id);
  } else {
    outputs = ToTensors(ge_outputs);
    GELOGI("[DFlowSession:%lu] Fetch data flow graph success, graph_id=%u, model_id=%u.", session_id_, graph_id,
           model_id);
  }
  return ret;
}

Status DFlowSessionImpl::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                            std::vector<FlowMsgPtr> &outputs, int32_t timeout) {
  UpdateThreadContext(graph_id);
  uint32_t model_id = INVALID_MODEL_ID;
  Status ret = dflow_graph_manager_.GetGraphModelId(graph_id, model_id);
  if ((ret != SUCCESS) || (model_id == INVALID_MODEL_ID)) {
    GELOGE(ret, "[Get][model] failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Get model failed, DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    return FAILED;
  }
  auto heterogeneous_executor = FlowModelManager::GetInstance().GetHeterogeneousModelExecutor(model_id);
  GE_CHECK_NOTNULL(heterogeneous_executor, ", model_id:%u.", model_id);

  ret = heterogeneous_executor->FetchFlowMsg(indexes, outputs, timeout);
  GE_CHK_BOOL_RET_STATUS((ret == SUCCESS || ret == ACL_ERROR_GE_REDEPLOYING || ret == ACL_ERROR_GE_SUBHEALTHY),
                         ret, "[Fetch][FlowMsg]failed, DFlowSession:%lu, graph_id=%u, model_id=%u.",
                         session_id_, graph_id, model_id);

  GELOGI("[DFlowSession:%lu] Fetch data flow graph success, graph_id=%u, model_id=%u.", session_id_, graph_id,
         model_id);
  return ret;
}

Status DFlowSessionImpl::FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,
                                     const DataFlowInfo &info, int32_t timeout) {
  UpdateThreadContext(graph_id);
  FlowModelPtr flow_model = dflow_graph_manager_.GetFlowModel(graph_id);
  if (flow_model == nullptr) {
    GELOGE(FAILED, "[Get][FlowModel] failed. Please make sure graph has been build before feed raw data, "
                   "DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "[Get][FlowModel] failed. Please make sure graph has been build before feed raw data, "
                       "DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
    return FAILED;
  }
  if (!dflow_graph_manager_.GetOptionsRunGraphFlag()) {
    GELOGI("Skip feed raw data as run flag is false.");
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(flow_model, ", DFlowSessionImpl:%lu graph_id=%u.", session_id_, graph_id);
  const auto model_id = flow_model->GetModelId();
  auto heterogeneous_executor = FlowModelManager::GetInstance().GetHeterogeneousModelExecutor(model_id);
  GE_CHECK_NOTNULL(heterogeneous_executor, ", model_id:%u.", model_id);
  const auto ret = heterogeneous_executor->FeedRawData(raw_data_list, index, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(FAILED, "[Feed][Data]failed, DFlowSessionImpl:%lu, graph_id=%u, model_id=%u.", session_id_, graph_id,
           model_id);
  } else {
    GELOGI("[DFlowSessionImpl:%lu] feed raw data flow graph success, graph_id=%u, model_id=%u.", session_id_, graph_id,
           model_id);
  }
  return ret;
}

void DFlowSessionImpl::UpdateThreadContext(const std::map<std::string, std::string> &options) const {
  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption(options);
}

void DFlowSessionImpl::UpdateGlobalSessionContext() const {
  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  }
  GetThreadLocalContext().SetSessionOption(options_);
  // Context use ge session id.
  GetContext().SetSessionId(ge_session_->GetSessionId());
  domi::GetContext().train_flag = false;
  SetRtSocVersion();
}

void DFlowSessionImpl::UpdateThreadContext(uint32_t graph_id) {
  auto options = dflow_graph_manager_.GetGraphOptions(graph_id);
  if (options == nullptr) {
    GELOGW("graph level options is null.");
    UpdateThreadContext(std::map<std::string, std::string>{});
  } else {
    UpdateThreadContext(*options);
  }
}

void DFlowSessionImpl::SetRtSocVersion() const {
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
}  // namespace ge
