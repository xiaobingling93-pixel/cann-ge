/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_v2_executor_builder.h"

#include <cstring>
#include <sstream>
#include <utility>
#include "ge/ge_api_types.h"
#include "executor_builder.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "runtime/model_v2_executor.h"
#include "equivalent_data_edges.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph_async_value.h"
#include "graph_executor_builder.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "common/checker.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "runtime/subscriber/executor_subscribers_scheduler.h"
#include "graph/utils/node_utils.h"
#include "core/executor/sequential/execution_data/sequential_execution_data_builder.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "graph/manager/session_id_manager.h"

namespace gert {
ge::ExecuteGraphPtr GetExecuteGraph(ge::ExecuteGraph *const root_graph, SubExeGraphType eg_type) {
  auto graph_type_str = GetSubExeGraphTypeStr(eg_type);
  auto graph_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph, graph_type_str);
  GE_ASSERT_NOTNULL(graph_node, "Failed to find node %s from root graph, execute graph type %d", graph_type_str,
                    static_cast<int32_t>(eg_type));
  auto sub_graph = ge::FastNodeUtils::GetSubgraphFromNode(graph_node, 0);
  GE_ASSERT_NOTNULL(sub_graph, "Failed to get sub graph from node %s", graph_node->GetNamePtr());
  return sub_graph->shared_from_this();
}

ge::graphStatus AllocRootGraphAnyValues(const ge::ExecuteGraphPtr &exe_graph,
                                        SymbolsToValue &symbols_to_value,
                                        TopologicalResourceGuard *resource_guard) {
  auto executor_builder = GraphExecutorBuilder({nullptr, nullptr}, exe_graph, &symbols_to_value);
  auto base_ed_builder = ge::MakeUnique<SequentialExecutionDataBuilder>(executor_builder);
  GE_ASSERT_NOTNULL(base_ed_builder);
  GraphAsyncValue graph_async_value;
  GE_RETURN_IF_ERROR(base_ed_builder->AllocGraphAsyncValues(exe_graph->GetDirectNode(), graph_async_value));
  resource_guard->ResetAnyValue(std::move(graph_async_value.values_guarder), graph_async_value.total_num);
  return ge::SUCCESS;
}

ModelV2ExecutorBuilder::ModelV2ExecutorBuilder(const ge::ExecuteGraphPtr &exe_graph) : exe_graph_(exe_graph) {
}

ModelV2ExecutorBuilder &ModelV2ExecutorBuilder::ExecuteGraph(const ge::ExecuteGraphPtr &exe_graph) {
  exe_graph_ = exe_graph;
  return *this;
}
ModelV2ExecutorBuilder &ModelV2ExecutorBuilder::ModelData(const ge::ModelData &model_data) {
  model_data_ = model_data;
  return *this;
}
ModelV2ExecutorBuilder &ModelV2ExecutorBuilder::GeRootModel(const ge::GeRootModelPtr &root_model) {
  root_model_ = root_model;
  return *this;
}

std::unique_ptr<ModelV2Executor> ModelV2ExecutorBuilder::Build() const {
  return Build(ExecutorType::kEnd);
}

std::unique_ptr<ModelV2Executor> ModelV2ExecutorBuilder::Build(ExecutorType executor_type) const {
  ExecutorOption executor_option(executor_type);
  return Build(executor_option);
}

std::unique_ptr<ModelV2Executor> ModelV2ExecutorBuilder::Build(const ExecutorOption &executor_option) const {
  GE_TIMESTAMP_START(ModelV2ExecutorBuilderBuild);
  GE_ASSERT_NOTNULL(exe_graph_);
  auto executor = std::unique_ptr<ModelV2Executor>(new (std::nothrow) ModelV2Executor);
  GE_ASSERT_NOTNULL(executor);
  EquivalentDataEdges eq_data_edges;
  SymbolsToValue symbols_to_value;
  GE_TIMESTAMP_START(UpdateEquivalentEdges);
  GE_ASSERT_GRAPH_SUCCESS(eq_data_edges.UpdateEquivalentEdges(exe_graph_.get()));
  GE_TIMESTAMP_EVENT_END(UpdateEquivalentEdges, "ModelV2ExecutorBuilderBuild::UpdateEquivalentEdges");
  GE_TIMESTAMP_START(AllocRootGraphAnyValues);
  GE_ASSERT_GRAPH_SUCCESS(AllocRootGraphAnyValues(exe_graph_, symbols_to_value, &(executor->resource_guard_)));
  GE_TIMESTAMP_EVENT_END(AllocRootGraphAnyValues, "ModelV2ExecutorBuilderBuild::AllocRootGraphAnyValues");

  GE_TIMESTAMP_START(ReadInBuffer);
  auto buffer = ReadInBuffer(*executor.get());
  GE_TIMESTAMP_EVENT_END(ReadInBuffer, "ModelV2ExecutorBuilderBuild::ReadInBuffer");
  GE_ASSERT_NOTNULL(buffer);

  GE_TIMESTAMP_START(ReadInComputeNodeInfo);
  auto compute_node_info = ReadInComputeNodeInfo(*buffer, *executor.get());
  GE_TIMESTAMP_EVENT_END(ReadInComputeNodeInfo, "ModelV2ExecutorBuilderBuild::ReadInComputeNodeInfo");
  GE_ASSERT_NOTNULL(compute_node_info);

  GE_TIMESTAMP_START(ReadInKernelExtendInfo);
  auto kernel_extend_info = ReadInKernelExtendInfo(*buffer, *executor.get());
  GE_TIMESTAMP_EVENT_END(ReadInKernelExtendInfo, "ModelV2ExecutorBuilderBuild::ReadInKernelExtendInfo");
  GE_ASSERT_NOTNULL(kernel_extend_info);

  GE_TIMESTAMP_START(ReadInModelDesc);
  GE_ASSERT_SUCCESS(ReadInModelDesc(*executor.get()));
  GE_TIMESTAMP_EVENT_END(ReadInModelDesc, "ModelV2ExecutorBuilderBuild::ReadInModelDesc");

  GE_TIMESTAMP_START(BuildGraph);

  const ModelLevelData model_level_data = {compute_node_info, kernel_extend_info};
  for (int32_t i = 0; i < kSubExeGraphTypeEnd; ++i) {
    auto subgraph = GetExecuteGraph(exe_graph_.get(), static_cast<SubExeGraphType>(i));
    GE_ASSERT_NOTNULL(subgraph, "Failed to get subgraph %s", GetSubExeGraphTypeStr(static_cast<SubExeGraphType>(i)));

    ge::graphStatus ret;
    if (i != static_cast<SubExeGraphType>(kMainExeGraph)) {
      ret = GraphExecutorBuilder(model_level_data, subgraph, &symbols_to_value).Build(executor->graphs_[i]);
    } else {
      ret = GraphExecutorBuilder(model_level_data, subgraph, &symbols_to_value)
                .ExecutorOpt(executor_option)
                .Build(executor->graphs_[i]);
    }
    GE_ASSERT_SUCCESS(ret, "Failed to build subgraph executor %s",
                      GetSubExeGraphTypeStr(static_cast<SubExeGraphType>(i)));
  }
  GE_TIMESTAMP_EVENT_END(BuildGraph, "ModelV2ExecutorBuilderBuild::BuildGraph");
  GE_ASSERT_NOTNULL(root_model_);

  ge::ComputeGraphPtr root_graph = root_model_->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);
  GE_ASSERT_GRAPH_SUCCESS(RestoreDeviceVarMem(*executor));
  uint32_t cur_model_id = root_model_->GetCurModelId();
  std::string model_name = root_model_->GetModelName();
  // Init Aipp
  GE_ASSERT_SUCCESS(executor->InitAipp(root_graph));
  GE_TIMESTAMP_START(SubscribersSchedulerInit);
  const auto &subscriber_extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor.get(), exe_graph_, root_graph, model_data_, root_model_, symbols_to_value, cur_model_id, model_name,
      nullptr, std::unordered_map<std::string, TraceAttr>{});
  GE_ASSERT_NOTNULL(subscriber_extend_info);
  executor->subscribers_.Init(subscriber_extend_info);
  GE_TIMESTAMP_EVENT_END(SubscribersSchedulerInit, "ModelV2ExecutorBuilderBuild::SubscribersSchedulerInit");
  GE_TIMESTAMP_EVENT_END(ModelV2ExecutorBuilderBuild, "ModelV2ExecutorBuilderBuild::All");
  executor->host_resource_center_ = root_model_->GetHostResourceCenterPtr();
  SetOutputReuseInputMemIndexes(*executor);
  return executor;
}

const ContinuousBuffer *ModelV2ExecutorBuilder::ReadInBuffer(ModelV2Executor &executor) const {
  auto buffer_data = ReadBufferFromAttr(kBuffer);
  if (buffer_data == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<ContinuousBuffer *>(executor.resource_guard_.ResetBuffer(std::move(buffer_data)));
}
const ContinuousBuffer *ModelV2ExecutorBuilder::ReadInComputeNodeInfo(const ContinuousBuffer &buffer,
                                                                      ModelV2Executor &executor) const {
  auto buffer_data = ReadBufferFromAttr(kComputeNodeInfo);
  if (buffer_data == nullptr) {
    return nullptr;
  }
  auto c_nodes_info =
      reinterpret_cast<ContinuousBuffer *>(executor.resource_guard_.ResetComputeNodeInfo(std::move(buffer_data)));

  for (size_t i = 0U; i < c_nodes_info->GetNum(); ++i) {
    auto c_node_info = c_nodes_info->Get<ComputeNodeInfo>(i);
    auto node_name_index = reinterpret_cast<size_t>(c_node_info->GetNodeName());
    auto node_type_index = reinterpret_cast<size_t>(c_node_info->GetNodeType());
    auto node_name_p = buffer.Get<char>(node_name_index);
    auto node_type_p = buffer.Get<char>(node_type_index);
    if (node_name_p == nullptr || node_type_p == nullptr) {
      return nullptr;
    }
    c_node_info->SetNodeName(node_name_p);
    c_node_info->SetNodeType(node_type_p);
  }
  return c_nodes_info;
}
const ContinuousBuffer *ModelV2ExecutorBuilder::ReadInKernelExtendInfo(const ContinuousBuffer &buffer,
                                                                       ModelV2Executor &executor) const {
  auto buffer_data = ReadBufferFromAttr(kKernelExtendInfo);
  if (buffer_data == nullptr) {
    return nullptr;
  }
  auto kernels_extend_info =
      reinterpret_cast<ContinuousBuffer *>(executor.resource_guard_.ResetKernelExtendInfo(std::move(buffer_data)));

  for (size_t i = 0U; i < kernels_extend_info->GetNum(); ++i) {
    auto ke_info = kernels_extend_info->Get<KernelExtendInfo>(i);
    auto name_index = reinterpret_cast<size_t>(ke_info->GetKernelName());
    auto type_index = reinterpret_cast<size_t>(ke_info->GetKernelType());
    auto name_p = buffer.Get<char>(name_index);
    auto type_p = buffer.Get<char>(type_index);
    if (name_p == nullptr || type_p == nullptr) {
      return nullptr;
    }
    ke_info->SetKernelName(name_p);
    ke_info->SetKernelType(type_p);
  }

  return kernels_extend_info;
}

ge::graphStatus ModelV2ExecutorBuilder::ReadInModelDesc(ModelV2Executor &executor) const {
  auto buffer_data = ReadBufferFromAttr(kModelDesc);
  GE_ASSERT_NOTNULL(buffer_data);

  auto model_desc_buffer =
      reinterpret_cast<ContinuousBuffer *>(executor.resource_guard_.ResetModelDesc(std::move(buffer_data)));
  auto model_desc = model_desc_buffer->Get<ModelDesc>(model_desc_buffer->GetNum() - 1);
  GE_ASSERT_NOTNULL(model_desc);

  auto space_registries =
      exe_graph_->TryGetExtAttr<std::shared_ptr<OpImplSpaceRegistryV2Array>>(kSpaceRegistry, nullptr);
  if (space_registries != nullptr) {
    model_desc->SetSpaceRegistries(space_registries.get());
  } else {
    GELOGW("Attention: default registry does not exist. Model desc and executor did not set space registry");
  }
  auto file_constant_weight_dir = exe_graph_->TryGetExtAttr<std::string>(kExternalFileConstantDir, "");
  executor.SetFileConstantWeightDir(file_constant_weight_dir);
  model_desc->SetFileConstantWeightDir(executor.GetFileConstantWeightDir().c_str());
  executor.SetModelDesc(model_desc);

  size_t input_num = 0U;
  size_t output_num = 0U;
  ModelIoDesc *io_descs = model_desc->AllMutableIoDesc(input_num, output_num);
  GE_ASSERT_NOTNULL(io_descs);
  for (size_t i = 0U; i < input_num + output_num; ++i) {
    ModelIoDesc &io_desc = io_descs[i];
    auto name_index = reinterpret_cast<size_t>(io_desc.GetName());
    auto name = model_desc_buffer->Get<char>(name_index);
    GE_ASSERT_NOTNULL(name);
    io_desc.SetName(name);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelV2ExecutorBuilder::RestoreDeviceVarMem(ModelV2Executor &executor) const {
  if ((session_ != nullptr) && session_->GetVarManager() != nullptr) {
    GELOGI("var manager exist, no need to restore, session_id: %lu", session_->GetSessionId());
    return ge::GRAPH_SUCCESS;
  }
  const auto &models = root_model_->GetSubgraphInstanceNameToModel();
  uint32_t graph_id = root_model_->GetRootGraph()->GetGraphID();
  uint64_t logic_var_base{0UL};
  int64_t total_var_size{0};
  std::vector<ge::NodePtr> device_variables;
  for (const auto &iter : models) {
    int64_t model_var_size{0};
    (void)ge::AttrUtils::GetInt(models.begin()->second.get(), ge::ATTR_MODEL_VAR_SIZE, total_var_size);
    if (model_var_size > 0) {
      total_var_size = model_var_size;
    }
    int64_t model_var_logic_base{0};
    (void)ge::AttrUtils::GetInt(models.begin()->second.get(), ge::ATTR_MODEL_TASK_GEN_VAR_ADDR, model_var_logic_base);
    if (model_var_logic_base > 0) {
      logic_var_base = static_cast<uint64_t>(model_var_logic_base);
    }
    const auto &graph = iter.second->GetGraph();
    GE_ASSERT_NOTNULL(graph);
    for (const auto &node : graph->GetAllNodes()) {
      if ((node->GetType() != ge::VARIABLE) && (node->GetType() != ge::CONSTANTOP)) {
        continue;
      }
      const std::string *placement = ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_VARIABLE_PLACEMENT);
      if ((placement != nullptr) && (*placement == "host")) {
        continue;
      }
      device_variables.push_back(node);
    }
  }

  if (device_variables.empty()) {
    return ge::GRAPH_SUCCESS;
  }

  GE_ASSERT_NOTNULL(session_);
  executor.load_session_id_ = session_->GetSessionId();
  GELOGI("load_session_id_: %lu, session_: %p", executor.load_session_id_, session_);
  int32_t device_id{0};
  GE_ASSERT_RT_OK(rtGetDevice(&device_id));
  auto rt_var_manager = ModelRtVarManager::Instance(executor.load_session_id_);
  GE_ASSERT_NOTNULL(rt_var_manager);
  void *external_var_addr = nullptr;
  uint64_t external_var_size = 0;
  session_->GetExternalVar(external_var_addr, external_var_size);
  GE_ASSERT_SUCCESS(rt_var_manager->Init(device_id, logic_var_base, total_var_size, external_var_addr, external_var_size));
  GE_ASSERT_SUCCESS(rt_var_manager->RestoreDeviceVariables(device_variables, graph_id, device_id));

  return ge::GRAPH_SUCCESS;
}

std::unique_ptr<uint8_t[]> ModelV2ExecutorBuilder::ReadBufferFromAttr(const char *attr_name) const {
  ge::Buffer attr_buffer;
  if (!ge::AttrUtils::GetZeroCopyBytes(exe_graph_, attr_name, attr_buffer)) {
    GELOGE(ge::PARAM_INVALID, "Failed to get buffer %s from root graph", attr_name);
    return nullptr;
  }
  std::unique_ptr<uint8_t[]> buffer_data = ge::MakeUnique<uint8_t[]>(attr_buffer.GetSize());
  if (buffer_data == nullptr) {
    return nullptr;
  }
  size_t buffer_size = attr_buffer.GetSize();
  size_t temp_size = 0UL;
  while (temp_size < attr_buffer.GetSize()) {
    size_t copy_size = (buffer_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : buffer_size;
    if (memcpy_s(buffer_data.get() + temp_size, copy_size, attr_buffer.GetData() + temp_size, copy_size) != EOK) {
      return nullptr;
    }
    temp_size += copy_size;
    buffer_size -= copy_size;
  }
  return buffer_data;
}

void ModelV2ExecutorBuilder::SetOutputReuseInputMemIndexes(ModelV2Executor &executor) const {
  if (root_model_ == nullptr) {
    return;
  }
  const auto &models = root_model_->GetSubgraphInstanceNameToModel();
  for (const auto &iter : models) {
    std::string reuse_indexes_str;
    if (!ge::AttrUtils::GetStr(iter.second, ge::ATTR_MODEL_OUTPUT_REUSE_INPUT_MEM_INDEXES, reuse_indexes_str)) {
      continue;
    }
    ge::ParseOutputReuseInputMemIndexes(reuse_indexes_str, executor.io_same_addr_pairs_);
    GELOGI("Read output reuse input mem indexes from model %s, pairs count: %zu",
           iter.first.c_str(), executor.io_same_addr_pairs_.size());
    break;  // Only need to read from one model
  }
}
}  // namespace gert
