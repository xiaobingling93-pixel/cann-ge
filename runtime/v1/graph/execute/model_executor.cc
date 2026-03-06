/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/execute/model_executor.h"
#include <sstream>
#include "base/err_mgr.h"
#include "graph/ge_context.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/model/external_allocator_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "graph/utils/tensor_adapter.h"
namespace ge {
namespace {
constexpr uint8_t kNeverLoaded = 0U;
constexpr uint32_t kExecuteStreamNumPerModel = 1U;

ge::GeModelPtr GetGeModel(const GeRootModelPtr &ge_root_model) {
  if (ge_root_model == nullptr) {
    return nullptr;
  }
  const auto &root_graph = ge_root_model->GetRootGraph();
  if (root_graph == nullptr) {
    return nullptr;
  }
  const auto &name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
  const auto it = name_to_model.find(root_graph->GetName());
  const GeModelPtr ge_model = (it != name_to_model.end()) ? it->second : nullptr;
  return ge_model;
}

std::string ToString(const std::vector<FeatureMemoryPtr> &all_feature_mem) {
  std::stringstream ss;
  for (const auto &feature_mem : all_feature_mem) {
    ss << "[type: " << MemTypeUtils::ToString(feature_mem->GetType()) << ", size: " << feature_mem->GetSize()
       << ", is_fixed: " << feature_mem->IsFixed() << "]";
  }
  if (all_feature_mem.empty()) {
    return "[empty]";
  }
  return ss.str();
}

}  // namespace

///
/// @ingroup ge
/// @brief graph executor init
/// @param [in] options user config params
/// @return Status result of function
///
Status ModelExecutor::Initialize(const std::map<std::string, std::string> &options, const uint64_t session_id) {
  if (init_flag_) {
    GELOGW("ModelExecutor has already initialized.");
    return SUCCESS;
  }

  session_id_ = session_id;
  const auto var_manager = VarManager::Instance(session_id);
  GE_ASSERT_NOTNULL(var_manager);
  size_t free_mem = 0U;
  size_t total_mem_size = 0U;
  GE_CHK_STATUS_RET_NOLOG(GetDeviceMemorySize(free_mem, total_mem_size));
  GE_CHK_STATUS_RET(var_manager->SetMemoryMallocSize(options, total_mem_size),
                    "VarManager SetMemoryMallocSize failed, InnerSession:%" PRIu64 ".", session_id_);
  thread_run_flag_.store(true);
  init_flag_ = true;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief graph executor finalize
/// @return Status result of function
///
Status ModelExecutor::Finalize() {
  if (!init_flag_) {
    GELOGW("ModelExecutor has not been initialized.");
    return SUCCESS;
  }

  StopQueue();
  if (run_thread_.joinable()) {
    run_thread_.join();
  }

  GELOGI("VarManager free var memory.");
  const auto var_manager = VarManager::Instance(session_id_);
  GE_ASSERT_NOTNULL(var_manager);
  (void)var_manager->FreeVarMemory();
  MemManager::Instance().FreeSessionMemory(session_id_);

  ModelManager::GetInstance().DestroyAicpuSession(session_id_);
  return SUCCESS;
}

Status ModelExecutor::GetDeviceMemorySize(size_t &free_mem, size_t &total_mem_size) {
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free_mem, &total_mem_size));
  if (total_mem_size == 0U) {
    GE_CHK_RT_RET(rtMemGetInfoEx(RT_MEMORYINFO_DDR, &free_mem, &total_mem_size));
  }
  GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));

  return SUCCESS;
}

void ModelExecutor::AddGraphNode(const GraphId graph_id, const GraphNodePtr &graph_node) {
  const std::lock_guard<std::mutex> lk(mutex_);
  graph_nodes_[graph_id] = graph_node;
}

void ModelExecutor::RemoveGraphNode(const GraphId graph_id) {
  const std::lock_guard<std::mutex> lk(mutex_);
  (void)graph_nodes_.erase(graph_id);
}

///
/// @ingroup ge
/// @brief Load mode for graph.
/// @param [in] ge_root_model: root model of graph compiled.
/// @param [in] GraphNode: node of graph.
/// @return Status result of function
///
Status ModelExecutor::LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                                const rtStream_t stream) {
  GE_CHECK_NOTNULL(graph_node);
  GE_CHECK_NOTNULL(ge_root_model);
  return ModelLoad(ge_root_model, graph_node, stream);
}

///
/// @ingroup ge
/// @brief Unload mode for graph.
/// @param [in] GeRootModel: root model of graph compiled.
/// @param [in] graph_id: graph identifier.
/// @return Status result of function
///
Status ModelExecutor::UnloadGraph(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) {
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
  RemoveGraphNode(graph_id);
  const Status ret = UnloadModel(ge_root_model, graph_id);
  if (ret != SUCCESS) {
    GELOGW("[GraphExecutor] unload model failed, graph_id=%u.", graph_id);
  }

  GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
  return ret;
}

Status ModelExecutor::UnloadModel(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) {
  GE_CHECK_NOTNULL(ge_root_model);
  // GeRootModel process
  for (const uint32_t &model_id : ge_root_model->GetAllModelId()) {
    GELOGI("Unload model %u.", model_id);
    GE_CHK_STATUS_RET(GraphLoader::UnloadModel(model_id),
                      "[GraphExecutor] unload model failed, modelId=%u, graphId=%u.", model_id, graph_id);
  }
  GE_ASSERT_SUCCESS(FreeFixedFeatureMemoryIfNeed(ge_root_model));

  return SUCCESS;
}

Status ModelExecutor::UnloadPneModel(const uint32_t model_id, const uint64_t session_id, const uint32_t graph_id) {
  const auto model = ModelManager::GetInstance().GetModel(model_id);
  if ((model != nullptr) && (model->GetAsyncMode())) {
    GELOGI("Unload model, async mode, start to synchronize stream before unload model, modelId[%u]", model_id);
    (void)rtStreamSynchronize(model->GetModelExecuteStream());
    GELOGI("Unload model, async mode, synchronize stream finished.");
  }
  if (ModelManager::GetInstance().DestroyAicpuKernel(session_id, model_id, 0U) != SUCCESS) {
    GELOGW("[GraphExecutor:] destroy aicpu kernel failed when unload model, modelId=%u, graphId=%u.", model_id,
           graph_id);
  }

  GE_CHK_STATUS_RET(GraphLoader::UnloadModel(model_id), "[GraphExecutor:] unload model failed, modelId=%u, graphId=%u.",
                    model_id, graph_id);

  GEEVENT("UnloadGraph[%u], model[%u] success.", graph_id, model_id);
  return SUCCESS;
}

void ModelExecutor::StopQueue() {
  thread_run_flag_.store(false);
  run_args_q_.Stop();
}

void ModelExecutor::ReturnError(const RunAsyncCallbackV2 &callback,
  const Status ret, const std::string &log_info) const {
  GELOGE(ret, "%s.", log_info.c_str());
  if (callback != nullptr) {
    std::vector<gert::Tensor> outputs;
    callback(ret, outputs);
  }
}

///
/// @ingroup ge
/// @brief Push model execution params to queue.
/// @param [in] RunArgs of for model execution.
/// @return Status result of function
///
Status ModelExecutor::PushRunArgs(const std::shared_ptr<RunArgs> &args) {
  return run_args_q_.Push(args) ? SUCCESS : FAILED;
}

void ModelExecutor::RunThread() {
  SET_THREAD_NAME(pthread_self(), "ge_mdlexecrun");
  if (mmSetCurrentThreadName("GE_Run") != EN_OK) {
    GELOGW("Set thread name failed.");
  }

  GELOGI("[RunThread] GE_Run start");
  while (thread_run_flag_) {
    std::shared_ptr<RunArgs> args;
    if (!run_args_q_.Pop(args)) {
      continue;
    }

    GELOGI("[RunThread] run graph async start, graph_id:%u.", args->graph_id);
    GE_MAKE_GUARD(args, [args]() { args->graph_node->Unlock(); });
    error_message::SetErrMgrContext(args->error_context);
    GetContext().SetSessionId(args->session_id);
    GetThreadLocalContext() = args->context;
    bool is_continue = false;
    if (is_continue) {
      GELOGI("graph [%u] is suspended, return success", args->graph_id);
      std::vector<gert::Tensor> outputs;
      args->callback(SUCCESS, outputs);
      args->graph_node->SetRunFlag(false);
      continue;
    }

    auto ge_root_model = args->graph_node->GetGeRootModel();
    if (ge_root_model == nullptr) {
      ReturnError(args->callback, PARAM_INVALID, "ge_root_model is invalid, thread exit.");
      continue;
    }
    Status ret = SUCCESS;
    args->graph_node->UpdateLoadFlag();
    if (!args->graph_node->GetLoadFlag()) {
      args->graph_node->SetAsync(true);
      ret = ModelLoad(ge_root_model, args->graph_node);
      if (ret != SUCCESS) {
        ReturnError(args->callback, ret, "LoadGraphAsync failed, thread exit.");
        continue;
      }
      GELOGI("LoadGraph[%u], model[%u] success and set LoadFlag to true.", args->graph_node->GetGraphId(),
             ge_root_model->GetModelId());
    }

    ret = graph_executor_.ExecuteGraphAsync(ge_root_model, args);
    args->graph_node->SetRunFlag(false);
    if (ret != SUCCESS) {
      ReturnError(args->callback, ret, "ExecuteGraphAsync failed, thread exit.");
      continue;
    }
    GELOGI("[GraphExecutor] Run graph async success, graph_id=%u.", args->graph_id);
  }
  GELOGI("[RunThread] GE_Run end");
}

///
/// @ingroup ge
/// @brief Run graph for synchronize model.
/// @param [in] graph_node: node of graph.
/// @param [in] graph_id: graph identifier.
/// @param [in] inputs: input data for the graph running.
/// @param [out] outputs: output data of the graph running
/// @return Status result of function
///
Status ModelExecutor::RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                               const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
  const auto ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  Status ret = graph_executor_.ExecuteGraph(graph_id, ge_root_model, inputs, outputs);
  graph_node->SetRunFlag(false);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Graph] failed, graph_id = %u.", graph_id);
  }
  return ret;
}

///
/// @ingroup ge
/// @brief Run graph for NN synchronize model.
/// @param [in] graph_node: node of graph.
/// @param [in] graph_id: graph identifier.
/// @param [in] stream: Stream for model running.
/// @param [in] inputs: input data for the graph running.
/// @param [out] outputs: output data of the graph running
/// @return Status result of function
///
Status ModelExecutor::RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                                         rtStream_t const stream, const std::vector<GeTensor> &inputs,
                                         std::vector<GeTensor> &outputs) {
  const auto ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  const auto ret = graph_executor_.ExecuteGraphWithStream(stream, graph_node, ge_root_model, inputs, outputs);
  graph_node->SetRunFlag(false);
  graph_node->SetIsSpecificStream(false);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Graph] With Stream failed, graph id = %u, stream = %p.", graph_id, stream);
    return ret;
  }
  return SUCCESS;
}

Status ModelExecutor::ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                                         rtStream_t const stream, const std::vector<gert::Tensor> &inputs,
                                         std::vector<gert::Tensor> &outputs) {
  auto ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  auto model_id = ge_root_model->GetModelId();
  const auto ret = ModelManager::GetInstance().ExecuteModelWithStreamAsync(model_id, graph_node, inputs,
    outputs, stream);
  graph_node->SetRunFlag(false);
  graph_node->SetIsSpecificStream(false);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Graph] With Stream failed, graph id = %u, stream = %p.", graph_id, stream);
    return ret;
  }
  return SUCCESS;
}

Status ModelExecutor::UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base,
                                              const size_t size) {
  const auto graph_id = graph_node->GetGraphId();
  const auto &ge_root_model = graph_node->GetGeRootModel();
  GE_ASSERT_NOTNULL(ge_root_model);
  const auto model_id = ge_root_model->GetModelId();
  GE_ASSERT_SUCCESS(ModelManager::GetInstance().UpdateFeatureMemoryBase(model_id, mem_base, size),
                    "Failed to update feature memory base, graph_id = %u, model_id = %u", graph_id, model_id);
  return SUCCESS;
}

/*
 * 5种不申请fix优先内存的情况：
 * 1.动态图，不在这里申请，在init图中申请
 * 2.fix优先内存size为0，fixed_feature_mem->GetSize() == 0U
 * 3.用户设置了feature memory base，user_has_set_feature_memory_base为true
 * 4.用户关闭GE申请fix优先内存。GeRootModel::IsNeedMallocFixedFeatureMemByType
 * 5.没有配置staticMemoryPolicy=4/2，也没有配置了OPTION_FEATURE_BASE_REFRESHABLE=1
 *
 * 2种申请fix优先内存的情况：
 * 1.配置了staticMemoryPolicy=4/2，一定要申请fix优先内存。如果注册了外置allocator，用外置allocator申请；
 *   如果没注册，使用Session级内存池，图间共用fix优先内存。
 * 2.没有配置staticMemoryPolicy=4/2，但是配置了OPTION_FEATURE_BASE_REFRESHABLE=1，申请fix优先内存。
 *   如果注册了外置allocator，就用外置allocator，否则用普通的内存分配器。
 */
Status ModelExecutor::MallocFixedFeatureMemoryIfNeed(const GraphNodePtr &graph_node,
                                                     const GeRootModelPtr &ge_root_model,
                                                     const rtStream_t stream) const {
  if (!ge_root_model->IsNeedMallocFixedFeatureMem()) {
    return SUCCESS;
  }

  std::vector<FeatureMemoryPtr> all_feature_mem;
  size_t hbm_fixed_size;
  GE_ASSERT_SUCCESS(ge_root_model->GetSummaryFeatureMemory(all_feature_mem, hbm_fixed_size),
                    "get summary feature memory failed, graph_id: %s", ge_root_model->GetModelName().c_str());
  (void)hbm_fixed_size;
  GELOGI("graph[%s] all fixed_feature_memory info: %s",
         ge_root_model->GetModelName().c_str(), ToString(all_feature_mem).c_str());
  for (const auto &fixed_feature_mem : all_feature_mem) {
    if (!fixed_feature_mem->IsFixed() || (fixed_feature_mem->GetSize() == 0U)) {
      continue;
    }
    const bool user_has_set_feature_memory_base = (fixed_feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT) &&
                                                  graph_node->IsAppRefreshFeatureMemory() &&
                                                  (graph_node->GetFeatureMemoryBase().first != nullptr);
    if (user_has_set_feature_memory_base) {
      GELOGI("user has set feature memory base, no need to malloc fixed_feature_memory");
      continue;
    }
    rtMemType_t rt_mem_type;
    GE_ASSERT_SUCCESS(MemTypeUtils::ExternalMemTypeToRtMemType(fixed_feature_mem->GetType(), rt_mem_type));
    if (!ge_root_model->IsNeedMallocFixedFeatureMemByType(rt_mem_type)) {
      GELOGI("no need to malloc fixed_feature_memory base, type:%s", MemTypeUtils::ToString(rt_mem_type).c_str());
      continue;
    }

    GE_ASSERT_SUCCESS(MallocByDiffAllocator(session_id_, stream, fixed_feature_mem, rt_mem_type, ge_root_model));
  }
  return SUCCESS;
}

Status ModelExecutor::MallocByDiffAllocator(const uint64_t session_id,
                                            const rtStream_t stream,
                                            const FeatureMemoryPtr &fixed_feature_mem,
                                            const rtMemType_t rt_mem_type,
                                            const GeRootModelPtr &ge_root_model) {
  void *addr = nullptr;
  MemBlock *block = nullptr;
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if ((external_allocator!= nullptr) && (fixed_feature_mem->GetType() == MemoryType::MEMORY_TYPE_DEFAULT)) {
    block = external_allocator->Malloc(fixed_feature_mem->GetSize());
    GE_ASSERT_NOTNULL(block, "malloc %zu bytes failed using external allocator", fixed_feature_mem->GetSize());
    addr = block->GetAddr();
    GE_ASSERT_NOTNULL(addr, "malloc %zu bytes failed using external allocator", fixed_feature_mem->GetSize());
    GELOGI("malloc %zu bytes success using external allocator", addr, fixed_feature_mem->GetSize());
    (void)ge_root_model->MutableFixedFeatureMemory().insert(
        {rt_mem_type, {rt_mem_type, addr, fixed_feature_mem->GetSize(), false, true, false, 0, block}});
    return SUCCESS;
  }

  if (VarManager::IsGeUseExtendSizeMemory(false)) {
    auto session_allocator = SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().
        GetMemAllocator(session_id, GetContext().DeviceId(), rt_mem_type);
    GE_ASSERT_NOTNULL(session_allocator);
    GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
    const auto mem_block = session_allocator->Malloc(fixed_feature_mem->GetSize());
    GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
    if ((mem_block != nullptr) && (mem_block->GetAddr() != nullptr)) {
      (void)ge_root_model->MutableFixedFeatureMemory().insert(
          {rt_mem_type, {rt_mem_type, mem_block->GetAddr(), fixed_feature_mem->GetSize(), false, true, true,
                         session_id, mem_block}});
      GELOGI("get fixed_feature_memory success, type: %s, addr: %p, size: %zu, session_id: %llu, using session"
          " allocator", MemTypeUtils::ToString(rt_mem_type).c_str(), mem_block->GetAddr(), fixed_feature_mem->GetSize(),
          session_id);
      return SUCCESS;
    } else {
      // 有些版本DRV不支持预留虚拟内存，需要再使用普通allocator申请内存
      GELOGW("malloc %zu bytes failed using inner session allocator", fixed_feature_mem->GetSize());
    }
  }

  const std::string purpose = MemTypeUtils::ToString(rt_mem_type) + " fixed feature base";
  auto &mem_instance = MemManager::Instance().MemInstance(rt_mem_type);
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
  addr = mem_instance.MallocMemory(purpose,
                                   fixed_feature_mem->GetSize(), GetContext().DeviceId());
  GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
  GE_ASSERT_NOTNULL(addr, "malloc %zu bytes failed using inner allocator", fixed_feature_mem->GetSize());
  GELOGI("malloc fixed_feature_memory success, type: %s, addr: %p, size: %zu",
         MemTypeUtils::ToString(rt_mem_type).c_str(), addr, fixed_feature_mem->GetSize());
  (void)ge_root_model->MutableFixedFeatureMemory().insert(
      {rt_mem_type, {rt_mem_type, addr, fixed_feature_mem->GetSize(), false, true, false, 0, block}});
  return SUCCESS;
}

Status ModelExecutor::FreeFixedFeatureMemoryIfNeed(const GeRootModelPtr &ge_root_model) {
  auto &all_fixed_mems = ge_root_model->MutableFixedFeatureMemory();
  for (auto iter = all_fixed_mems.begin(); iter != all_fixed_mems.end();) {
    if (!iter->second.ge_alloc) {
      ++iter;
      continue;
    }
    if (iter->second.block != nullptr) {
      iter->second.block->Free();
      if (iter->second.is_session_allocator) {
        GELOGI("free fixed_feature_memory by session allocator, %s", iter->second.ToString().c_str());
      } else {
        GELOGI("free fixed_feature_memory by external allocator success, addr: %p, size: %zu",
               iter->second.addr, iter->second.size);
      }
    } else {
      auto &mem_instance = MemManager::Instance().MemInstance(iter->second.type);
      GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
      GE_ASSERT_SUCCESS(mem_instance.FreeMemory(iter->second.addr, GetContext().DeviceId()));
      GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
      GELOGI("free fixed_feature_memory by inner allocator success, %s", iter->second.ToString().c_str());
    }
    iter = all_fixed_mems.erase(iter);
  }
  return SUCCESS;
}

Status ModelExecutor::ModelLoad(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                                const rtStream_t stream) {
  uint32_t model_id = INVALID_MODEL_ID;
  GE_CHECK_NOTNULL(ge_root_model);
  const auto root_graph = ge_root_model->GetRootGraph();
  bool is_unknown_shape = false;
  GE_CHK_STATUS_RET(ge_root_model->CheckIsUnknownShape(is_unknown_shape));
  if (!is_unknown_shape) {
    GE_CHK_STATUS_RET(CheckAndReleaseMemory(ge_root_model, graph_node));
    GE_CHK_STATUS_RET(CheckAndReleaseStream(ge_root_model, graph_node));
    GE_CHK_STATUS_RET(CheckAndReleaseEvent(ge_root_model, graph_node));
  }

  // 申请fix feature内存
  GE_ASSERT_SUCCESS(MallocFixedFeatureMemoryIfNeed(graph_node, ge_root_model, stream));

  if (!graph_node->IsAsync()) {
    ge_root_model->SetIsSpecificStream(graph_node->IsSpecificStream());
  }

  GE_TIMESTAMP_START(LoadModelOnline);
  Status ret = GraphLoader::LoadModelOnline(model_id, ge_root_model, graph_node, GetContext().DeviceId(),
                                     error_message::GetErrMgrContext(), stream);
  GE_TIMESTAMP_EVENT_END(LoadModelOnline, "GraphLoader::LoadModelOnline");

  if (ret != SUCCESS) {
    (void)FreeFixedFeatureMemoryIfNeed(ge_root_model);
    GELOGE(ret, "[Load][ModelOnline] Failed, model_id:%u", model_id);
    graph_node->SetRunFlag(false);
    return ret;
  }
  graph_node->SetLoaded();
  AddGraphNode(graph_node->GetGraphId(), graph_node);
  return SUCCESS;
}

bool ModelExecutor::ReleaseMemory(const GeRootModelPtr &ge_root_model, const GraphNodePtr &loaded_graph_node) const {
  if (ge_root_model == nullptr) {
    return false;
  }

  // unload model not release
  bool is_unknown_shape = false;
  (void)ge_root_model->CheckIsUnknownShape(is_unknown_shape);
  if (is_unknown_shape) {
    return false;
  }

  // unload static shape model
  const auto current_ge_model = GetGeModel(ge_root_model);
  if (current_ge_model == nullptr) {
    return false;
  }

  int64_t value = 0;
  uint64_t session_id =
      AttrUtils::GetInt(current_ge_model, MODEL_ATTR_SESSION_ID, value) ? static_cast<uint64_t>(value) : 0U;
  GEEVENT("Start to release static model memory.");
  const auto &model_ids = ge_root_model->GetAllModelId();
  const uint32_t graph_id = loaded_graph_node->GetGraphId();

  // 如果model中包含有hccl task，不做卸载
  for (const auto &model_id : model_ids) {
    const auto &model = ModelManager::GetInstance().GetModel(model_id);
    if ((model != nullptr) && (model->HasHcclTask())) {
      GELOGI("Cannot unload graph[%u], model[%u] which has hccl task.", graph_id, model_id);
      return false;
    }
  }

  for (const auto &model_id : model_ids) {
    uint64_t max_memory_size = 0U;
    Status result = ModelManager::GetInstance().GetMaxUsedMemory(model_id, max_memory_size);
    if (result != SUCCESS) {
      continue;
    }
    GELOGI("try to UnloadGraph[%u], model[%u] which MaxUsedMemory[%" PRIu64 "].", graph_id, model_id, max_memory_size);
    if (model_ids.size() > 1U) {
      result = current_ge_model->GetSessionId(model_id, session_id);
      if (result != SUCCESS) {
        GELOGW("[GraphExecutor:] get session failed when dynamic memory, modelId=%u, graphId=%u.", model_id, graph_id);
        continue;
      }
    }

    (void)UnloadPneModel(model_id, session_id, graph_id);
  }
  return true;
}

bool ModelExecutor::ReleaseModel(const GeRootModelPtr &ge_root_model, const GraphNodePtr &loaded_graph_node) const {
  if (ge_root_model == nullptr) {
    return false;
  }

  const auto current_ge_model = GetGeModel(ge_root_model);
  if (current_ge_model == nullptr) {
    return false;
  }
  const auto &model_ids = ge_root_model->GetAllModelId();
  const uint32_t graph_id = loaded_graph_node->GetGraphId();

  // 如果model中包含有hccl task，不做卸载
  for (const auto &model_id : model_ids) {
    const auto &davinci_model = ModelManager::GetInstance().GetModel(model_id);
    if ((davinci_model != nullptr) && (davinci_model->HasHcclTask())) {
      GELOGI("Cannot unload graph[%u], model[%u] which has hccl task.", graph_id, model_id);
      return false;
    }
  }

  for (const auto &model_id : model_ids) {
    const auto &davinci_model = ModelManager::GetInstance().GetModel(model_id);
    if (davinci_model == nullptr) {
      continue;
    }
    GELOGI("Unload graph[%u], model[%u] which stream num[%" PRIu64 "] event num[%" PRIu64 "].",
           graph_id, model_id, davinci_model->GetAllStreamNum(), davinci_model->GetEventList().size());

    if (davinci_model->GetAsyncMode()) {
      (void)rtStreamSynchronize(davinci_model->GetModelExecuteStream());
      GELOGI("Unload model, async mode, synchronize stream finished.");
    }

    (void)GraphLoader::UnloadModel(model_id);
    GEEVENT("Unload graph[%u], model[%u] finished.", graph_id, model_id);
  }
  ge_root_model->ClearAllModelId();

  return true;
}

Status ModelExecutor::GetMemoryInfo(size_t &free) {
  size_t total_mem = 0U;
  size_t free_mem = 0U;
  GE_CHK_STATUS_RET_NOLOG(GetDeviceMemorySize(free_mem, total_mem));

  const size_t limited_max_size = static_cast<size_t>(static_cast<float64_t>(total_mem) * kMaxMemorySizeRatio);
  free = ((free_mem + limited_max_size) > total_mem) ? (free_mem + limited_max_size - total_mem) : 0U;
  GELOGI("GetMemoryInfo free[%zu], total[%zu], limited_max_size[%zu], return free[%zu]",
         free_mem, total_mem, limited_max_size, free);
  return SUCCESS;
}

Status ModelExecutor::GetMemorySizeAfterReuse(const std::vector<GeModelPtr> &ge_models, const GraphNodePtr &graph_node,
                                              int64_t &sum_size, bool &reuse) const {
  auto mem_instance =
      SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(session_id_, GetContext().DeviceId());
  GE_ASSERT_NOTNULL(mem_instance);
  const int64_t malloced_feature_mem_size = mem_instance->MemorySize();
  const bool is_reuse_zero_copy_memory = ModelUtils::IsReuseZeroCopyMemory();

  int64_t total_non_zero_copy_mem_size = 0;
  int64_t total_weight_size = 0;
  for (const auto &ge_model : ge_models) {
    int64_t value = 0;
    const int64_t memory_size = AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, value) ? value : 0;
    const int64_t weight_size = AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, value) ? value : 0;
    const int64_t zero_copy_size = AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, value) ? value : 0;
    int64_t non_zero_copy_mem_size = memory_size;
    if (is_reuse_zero_copy_memory) {
      if (memory_size < zero_copy_size) {
            REPORT_INNER_ERR_MSG("E19999", "total mem size[%" PRId64 "] is less than zero copy size[%" PRId64 "] ",
                              memory_size, zero_copy_size);
        GELOGE(FAILED, "[Check] failed, total mem size[%" PRId64 "] is less than zero copy size[%" PRId64 "]",
               memory_size, zero_copy_size);
        return FAILED;
      }
      non_zero_copy_mem_size = memory_size - zero_copy_size;
    }
    GE_CHK_STATUS_RET(CheckInt64AddOverflow(total_non_zero_copy_mem_size, non_zero_copy_mem_size),
                      "total_non_zero_copy_mem_size[%" PRId64 "] and non_zero_copy_mem_size[%" PRId64
                      "] will overflow after add",
                      total_non_zero_copy_mem_size, non_zero_copy_mem_size);
    GE_CHK_STATUS_RET(CheckInt64AddOverflow(total_weight_size, weight_size),
                      "total_weight_size[%" PRId64 "] and weight_size[%" PRId64 "] will overflow after add",
                      total_weight_size, weight_size);
    total_non_zero_copy_mem_size += non_zero_copy_mem_size;
    total_weight_size += weight_size;
    GELOGI(
        "Graph[%u] non_zero_copy_mem_size[%" PRId64 "], memory_size[%" PRId64 "], zero_copy_size[%" PRId64 "], "
        "is_reuse_zero_copy_memory[%d], weight_size[%" PRId64 "], Device[%u]",
        graph_node->GetGraphId(), non_zero_copy_mem_size, memory_size, zero_copy_size,
        static_cast<int32_t>(is_reuse_zero_copy_memory), weight_size, GetContext().DeviceId());
  }

  if (total_non_zero_copy_mem_size <= malloced_feature_mem_size) {
    reuse = true;
  }
  GE_CHK_STATUS_RET(CheckInt64AddOverflow(total_weight_size, total_non_zero_copy_mem_size),
                    "total_weight_size[%" PRId64 "] and total_non_zero_copy_mem_size[%" PRId64
                    "] will overflow after add", total_weight_size, total_non_zero_copy_mem_size);
  sum_size = total_weight_size + (reuse ? 0 : total_non_zero_copy_mem_size - malloced_feature_mem_size);

  GELOGI("Graph[%u] reuse[%d], total_non_zero_copy_mem_size[%" PRId64 "], malloced_feature_mem_size[%" PRId64 "], "
         "sum_size[%" PRId64 "], total_weight_size[%" PRId64 "], Device[%u]",
         graph_node->GetGraphId(), static_cast<int32_t>(reuse), total_non_zero_copy_mem_size, malloced_feature_mem_size,
         sum_size, total_weight_size, GetContext().DeviceId());
  return SUCCESS;
}

Status ModelExecutor::CheckFreeMemory(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                                      bool &is_enough, bool &release_all) const {
  is_enough = false;
  release_all = false;
  size_t free_memory = 0U;
  GE_CHK_STATUS_RET(GetMemoryInfo(free_memory));
  GE_CHECK_NOTNULL(ge_root_model);
  std::vector<GeModelPtr> ge_models;
  const GeModelPtr ge_model = GetGeModel(ge_root_model);
  if (ge_model == nullptr) {
    return SUCCESS;
  }
  ge_models.emplace_back(ge_model);

  int64_t sum_size = 0;
  const auto feature_mem = graph_node->GetFeatureMemoryBase();
  const auto const_mem = graph_node->GetConstMemoryBase();

  void *hbm_fixed_mem = nullptr;
  const auto fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  const auto hbm_fixed_mem_iter = fixed_feature_mem.find(RT_MEMORY_HBM);
  if (hbm_fixed_mem_iter != fixed_feature_mem.end()) {
    hbm_fixed_mem = hbm_fixed_mem_iter->second.addr;
  }

  const auto refreshable_feature_mem = graph_node->GetRefreshableFeatureMemoryBase();
  const bool not_set_fm_and_const_mem = ((feature_mem.first == nullptr) && (const_mem.first == nullptr) &&
      (hbm_fixed_mem == nullptr) && (refreshable_feature_mem.first == nullptr));
  if (ModelUtils::IsGeUseExtendSizeMemory() && not_set_fm_and_const_mem) {
    bool reuse = false;
    GE_CHK_STATUS_RET(GetMemorySizeAfterReuse(ge_models, graph_node, sum_size, reuse));
    release_all = false;
    GELOGI("use static memory, release_all[%d], reuse[%d]", static_cast<int32_t>(release_all),
           static_cast<int32_t>(reuse));
  } else {
    const auto &ge_model_local = *ge_models.begin();
    int64_t value = 0;
    int64_t memory_size = AttrUtils::GetInt(ge_model_local, ATTR_MODEL_MEMORY_SIZE, value) ? value : 0;
    int64_t weight_size = AttrUtils::GetInt(ge_model_local, ATTR_MODEL_WEIGHT_SIZE, value) ? value : 0;
    const int64_t zero_copy_size = AttrUtils::GetInt(ge_model_local, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, value) ? value : 0;

    // 外部设置fm地址(fm, fixed fm, refreshable fm), IO内存均由外部分配
    if ((feature_mem.first != nullptr) || (refreshable_feature_mem.first != nullptr)) {
      memory_size = 0;
    } else if (hbm_fixed_mem != nullptr) {
      std::vector<FeatureMemoryPtr> all_feature_mem;
      size_t required_hbm_fixed_size;
      GE_ASSERT_SUCCESS(ge_root_model->GetSummaryFeatureMemory(all_feature_mem, required_hbm_fixed_size),
                        "get summary feature memory failed, graph_id: %s", ge_root_model->GetModelName().c_str());
      (void)all_feature_mem;
      GE_ASSERT_SUCCESS(CheckInt64SubOverflow(memory_size, zero_copy_size),
                        "sub overflow, memory_size: %lld, zero_copy_size: %lld", memory_size, zero_copy_size);
      GE_ASSERT_SUCCESS(CheckInt64SubOverflow(memory_size - zero_copy_size,
                                              static_cast<int64_t>(required_hbm_fixed_size)),
                        "sub overflow, memory_size - zero_copy_size: %lld, required_hbm_fixed_size: %lld",
                        memory_size - zero_copy_size, required_hbm_fixed_size);
      memory_size = memory_size - zero_copy_size - static_cast<int64_t>(required_hbm_fixed_size);
    } else {
      // misra
    }

    weight_size = (const_mem.first != nullptr) ? 0 : weight_size;
    GE_ASSERT_SUCCESS(CheckInt64AddOverflow(memory_size, weight_size),
        "memory_size[%" PRId64 "] and weight_size[%" PRId64 "] will overflow after add", memory_size, weight_size);
    sum_size = memory_size + weight_size;
    const auto var_manager = VarManager::Instance(session_id_);
    GE_ASSERT_NOTNULL(var_manager);
    const int64_t var_total_size = var_manager->GetVarMemSize(RT_MEMORY_HBM);
    const int64_t var_malloc_size = var_manager->GetVarMallocMemSize();
    const int64_t var_size = var_total_size - var_malloc_size;
    GE_ASSERT_TRUE(var_size >= 0LL, "var mem size[%" PRId64 "] "
        "should larger than var malloc size[%" PRId64 "], check invalid",
        var_total_size, var_malloc_size);
    GE_ASSERT_SUCCESS(CheckInt64AddOverflow(var_size, sum_size),
        "var_size[%" PRId64 "] and sum_size[%" PRId64 "] will overflow after add", var_size, sum_size);
    sum_size += var_size;
    GELOGI("Graph[%u] need memory_size[%" PRId64 "], "
        "weight_size[%" PRId64 "], var_size[%" PRId64 "], Device[%u] free_memory_size[%zu]",
        graph_node->GetGraphId(), memory_size, weight_size,
        var_size, GetContext().DeviceId(), free_memory);
  }

  if (free_memory >= static_cast<size_t>(sum_size)) {
    is_enough = true;
  }
  return SUCCESS;
}

Status ModelExecutor::CheckAndReleaseMemory(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
  bool is_enough = false;
  bool release_all = false;
  GE_CHK_STATUS_RET(CheckFreeMemory(ge_root_model, graph_node, is_enough, release_all));
  if ((!release_all) && is_enough) {
    GELOGI("graph id[%u] no need to unload other models, release_all[%d], is_enough[%d]",
           graph_node->GetGraphId(), static_cast<int32_t>(release_all), static_cast<int32_t>(is_enough));
    return SUCCESS;
  }
  GEEVENT("graph id[%u] need to unload other models, if have any, release_all[%d], is_enough[%d]",
          graph_node->GetGraphId(), static_cast<int32_t>(release_all), static_cast<int32_t>(is_enough));

  const std::lock_guard<std::mutex> lk(mutex_);
  for (const auto &it : graph_nodes_) {
    if ((it.second == nullptr) || (!it.second->GetLoadFlag())) {  // not loaded,no need unload
      GELOGI("CheckAndReleaseMemory graph[%u] has not been loaded.", it.first);
      continue;
    }
    GeRootModelPtr tmp_ge_root_model = it.second->GetGeRootModel();
    if (!DoReleaseModel(tmp_ge_root_model, it.second)) {
      continue;
    }
    it.second->SetLoadFlag(false);
    // Allow model to be loaded agagin without adding graph again
    it.second->SetLoadCount(it.second->GetLoadRecord());
    it.second->SetLoadRecord(kNeverLoaded);
    tmp_ge_root_model->ClearAllModelId();

    if ((!release_all)) {
      GE_CHK_STATUS_RET(CheckFreeMemory(ge_root_model, graph_node, is_enough, release_all));
    }
    if ((!release_all) && is_enough) {
      return SUCCESS;
    }
  }

  // unload unknown shape model
  (void)MemManager::Instance().CachingInstance(RT_MEMORY_HBM).TryFreeBlocks();
  (void)hybrid::NpuMemoryAllocator::FreeCachedMem();
  return SUCCESS;
}

bool ModelExecutor::DoReleaseModel(const GeRootModelPtr &ge_root_model,
                                   const GraphNodePtr &loaded_graph_node) const {
  return ReleaseMemory(ge_root_model, loaded_graph_node);
}

Status ModelExecutor::GetStreamNum(const GeRootModelPtr &ge_root_model, uint32_t &stream_num,
                                   uint64_t &hccl_follow_stream) const {
  const auto ge_model = GetGeModel(ge_root_model);
  GE_CHECK_NOTNULL(ge_model);

  uint32_t model_stream_num = 0U;
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_STREAM_NUM, model_stream_num);

  const Status status = ModelUtils::CalculateFollowStream(ge_model, hccl_follow_stream);
  if (status != SUCCESS) {
    GELOGE(FAILED, "[Calculate][stream] Calculate follow stream num failed");
    return FAILED;
  }
  stream_num = model_stream_num + static_cast<uint32_t>(hccl_follow_stream) + kExecuteStreamNumPerModel;
  GELOGI("model total stream num: %u, model stream num: %u, hccl follow stream num: %zu", stream_num, model_stream_num,
         hccl_follow_stream);

  return SUCCESS;
}

Status ModelExecutor::CheckAndReleaseStream(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
  uint32_t required_stream_num = 0U;
  uint64_t hccl_follow_stream_num = 0U;
  if (GetStreamNum(ge_root_model, required_stream_num, hccl_follow_stream_num) != SUCCESS) {
    return FAILED;
  }

  uint32_t free_stream_num = 0U;
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtGetAvailStreamNum(RT_NORMAL_STREAM, &free_stream_num));

  if (required_stream_num <= free_stream_num) {
    GELOGI("Graph id[%u] no need to unload other models, required stream num[%u], free stream num[%u]",
           graph_node->GetGraphId(), required_stream_num, free_stream_num);
    GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
    return SUCCESS;
  }

  GEEVENT("Graph id[%u] need to unload other models, if have any, required stream num[%u], free stream num[%u]",
          graph_node->GetGraphId(), required_stream_num, free_stream_num);

  const std::lock_guard<std::mutex> lk(mutex_);
  for (const auto &it : graph_nodes_) {
    if ((it.second == nullptr) || (!it.second->GetLoadFlag())) {  // not loaded,no need unload
      GELOGI("Check and release stream resource, graph[%u] has not been loaded.", it.first);
      continue;
    }

    if (!ReleaseModel(it.second->GetGeRootModel(), it.second)) {
      continue;
    }

    it.second->SetLoadFlag(false);
    // Allow model to be loaded agagin without adding graph again
    it.second->SetLoadCount(it.second->GetLoadRecord());
    it.second->SetLoadRecord(kNeverLoaded);

    GE_CHK_RT_RET(rtGetAvailStreamNum(RT_NORMAL_STREAM, &free_stream_num));
    if (required_stream_num <= free_stream_num) {
      GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
      return SUCCESS;
    }
  }

  GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
  GELOGE(FAILED,
         "Graph id[%u] check and release stream failed, required total stream num[%u], required hccl follow stream "
         "num[%u], free stream num[%u]",
         graph_node->GetGraphId(), required_stream_num, hccl_follow_stream_num, free_stream_num);

  return FAILED;
}

Status ModelExecutor::GetEventNum(const GeRootModelPtr &ge_root_model, uint32_t &event_num) const {
  const auto ge_model = GetGeModel(ge_root_model);
  GE_CHECK_NOTNULL(ge_model);

  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_EVENT_NUM, event_num);

  uint32_t aicpu_blocking_event_num = 0U;
  GE_ASSERT_SUCCESS(ModelUtils::CalculateAicpuBlockingEventNum(ge_model, aicpu_blocking_event_num),
                    "Calculate aicpu blocking event num failed");

  uint32_t hccl_group_ordered_event_num = 0U;
  GE_ASSERT_SUCCESS(ModelUtils::CalculateHcclGroupOrderedEventNum(ge_model, hccl_group_ordered_event_num),
    "Calculate hccl group ordered event num failed");
  GELOGI("Model event num[%u] aicpu blocking event num[%u], hccl group ordered event num[%u].",
    event_num, aicpu_blocking_event_num, hccl_group_ordered_event_num);
  event_num = event_num + aicpu_blocking_event_num + hccl_group_ordered_event_num;
  return SUCCESS;
}

Status ModelExecutor::CheckAndReleaseEvent(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node) {
  uint32_t required_event_num = 0U;
  if (GetEventNum(ge_root_model, required_event_num) != SUCCESS) {
    return FAILED;
  }

  uint32_t free_event_num = 0U;
  GE_CHK_RT_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtGetAvailEventNum(&free_event_num));

  if (required_event_num <= free_event_num) {
    GELOGI("Graph id[%u] no need to unload other models, required event nums[%u], free event nums[%u]",
           graph_node->GetGraphId(), required_event_num, free_event_num);
    GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
    return SUCCESS;
  }

  GEEVENT("Graph id[%u] need to unload other models, if have any, required event nums[%u], free event nums[%u]",
          graph_node->GetGraphId(), required_event_num, free_event_num);

  const std::lock_guard<std::mutex> lk(mutex_);
  for (const auto &it : graph_nodes_) {
    if ((it.second == nullptr) || (!it.second->GetLoadFlag())) {  // not loaded,no need unload
      GELOGI("Check and release event resource, graph[%u] has not been loaded.", it.first);
      continue;
    }

    if (!ReleaseModel(it.second->GetGeRootModel(), it.second)) {
      continue;
    }

    it.second->SetLoadFlag(false);
    // Allow model to be loaded agagin without adding graph again
    it.second->SetLoadCount(it.second->GetLoadRecord());
    it.second->SetLoadRecord(kNeverLoaded);

    GE_CHK_RT_RET(rtGetAvailEventNum(&free_event_num));
    if (required_event_num <= free_event_num) {
      GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
      return SUCCESS;
    }
  }

  GE_CHK_RT_RET(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
  GELOGE(FAILED, "Graph id[%u] check and release event failed, required event nums[%u], free event nums[%u]",
         graph_node->GetGraphId(), required_event_num, free_event_num);

  return FAILED;
}

void ModelExecutor::StartRunThread() {
  run_thread_ = std::thread(&ModelExecutor::RunThread, this);
}

Status ModelExecutor::PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                                 const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) {
  const auto &ge_root_model = graph_node->GetGeRootModel();
  GE_ASSERT_NOTNULL(ge_root_model);
  const auto model_id = ge_root_model->GetModelId();
  if (model_id == INVALID_MODEL_ID) {
    GELOGW("[GraphExecutor] model do not load, graphId=%u.", graph_node->GetGraphId());
    return PARAM_INVALID;
  }
  return ModelManager::GetInstance().PaRemapped(model_id, va, new_pa, len, cross_ranges);
}

} // namespace ge
