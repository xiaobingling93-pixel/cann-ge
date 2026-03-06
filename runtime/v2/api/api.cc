/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/gert_api.h"
#include "lowering/model_converter.h"
#include "framework/common/helper/model_helper.h"
#include "common/helper/model_parser_base.h"
#include "common/host_resource_center/host_resource_center.h"
#include "core/debug/kernel_tracing.h"
#include "core/builder/model_v2_executor_builder.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/runtime/allocator.h"
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/external_allocator.h"
#include "runtime/mem.h"
#include "graph/utils/graph_utils.h"
#include "graph/load/model_manager/model_manager.h"

namespace gert {
namespace {
ge::graphStatus LoadToModelV2ExecutorBuilder(const ge::ModelData &model_data, const ModelConverter::Args &args,
                                             ModelV2ExecutorBuilder &builder) {
  ge::ModelHelper model_helper;
  auto error_code = model_helper.LoadRootModel(model_data);
  if (error_code != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Failed to load root model from model data");
    return error_code;
  }
  // todo： 当前单算子是一个临时方案， 后续考虑与图流程归一
  auto root_model = model_helper.GetGeRootModel();
  GE_ASSERT_NOTNULL(root_model);
  auto root_graph = root_model->GetRootGraph();
  GE_ASSERT_NOTNULL(root_graph);

  if (ge::GraphUtils::IsSingleOpScene(root_graph)) {
    GELOGD("single op scene, set ge model as root model");
    auto root_ge_model = model_helper.GetGeModel();
    root_graph->SetGraphUnknownFlag(true);
    root_model->SetRootGraph(root_graph);
    root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), root_ge_model);
  }

  uint32_t model_id = 0U;
  ge::ModelManager::GetInstance().GenModelId(model_id);
  root_model->SetCurModelId(model_id);
  GELOGI("Current id of hybrid model %s is %u.", root_model->GetModelName().c_str(), root_model->GetCurModelId());
  for (auto &named_model : root_model->GetSubgraphInstanceNameToModel()) {
    GE_ASSERT_NOTNULL(named_model.second, "Compiled model of graph %s is nullptr", named_model.first.c_str());
    named_model.second->SetModelId(model_id);
  }
  GE_ASSERT_SUCCESS(ge::ModelManager::GetInstance().InitOpMasterDeviceSo(model_id, root_model),
                    "Init model [%u] op master device failed", model_id);

  auto graph = ModelConverter().ConvertGeModelToExecuteGraph(root_model, args);
  GE_ASSERT_NOTNULL(graph, "Failed to lowering to execute graph");

  builder.ExecuteGraph(graph).ModelData(model_data).GeRootModel(root_model);

  return ge::GRAPH_SUCCESS;
}
}  // namespace
void FreeModelData(ge::ModelData &model_data) {
  delete[] static_cast<ge::char_t *>(model_data.model_data);
  model_data.model_data = nullptr;
}

ge::graphStatus LoadDataFromFile(const ge::char_t *model_path, ge::ModelData &model_data) {
  const auto ret = ge::ModelParserBase::LoadFromFile(model_path, -1, model_data);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Failed to load model data");
  }
  return ret;
}

std::unique_ptr<ModelV2Executor> LoadExecutorFromFile(const ge::char_t *model_path, ge::graphStatus &error_code) {
  ge::ModelData model_data;
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });
  error_code = LoadDataFromFile(model_path, model_data);
  if (error_code != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Failed to load model data form model path");
    return nullptr;
  }

  auto ret = LoadExecutorFromModelData(model_data, error_code);
  return ret;
}

std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           ge::graphStatus &error_code) {
  return LoadExecutorFromModelData(model_data, ExecutorOption{}, error_code);
}

std::unique_ptr<ModelV2Executor> LoadExecutorFromModelDataWithRtSession(const ge::ModelData &model_data,
                                                                        RtSession *const rt_session,
                                                                        ge::graphStatus &error_code) {
  ModelV2ExecutorBuilder builder(rt_session);
  error_code = LoadToModelV2ExecutorBuilder(model_data, {}, builder);
  GE_ASSERT_SUCCESS(error_code);
  ExecutorOption option{};
  return builder.Build(option);
}

std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const LoadExecutorArgs &args,
                                                           ge::graphStatus &error_code) {
  ModelV2ExecutorBuilder builder(args.rt_session);
  ModelConverter::Args converter_args{{}, nullptr, nullptr, nullptr, &args.file_constant_mems};
  error_code = LoadToModelV2ExecutorBuilder(model_data, converter_args, builder);
  GE_ASSERT_SUCCESS(error_code);
  ExecutorOption option{};
  return builder.Build(option);
}

std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const ExecutorOption &executor_option,
                                                           ge::graphStatus &error_code) {
  ModelV2ExecutorBuilder builder;
  error_code = LoadToModelV2ExecutorBuilder(model_data, {}, builder);
  GE_ASSERT_SUCCESS(error_code);
  return builder.Build(executor_option);
}

std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const ExecutorOption &executor_option,
                                                           StreamAllocator *const stream_allocator,
                                                           EventAllocator *const event_allocator,
                                                           NotifyAllocator *const notify_allocator,
                                                           ge::graphStatus &error_code) {
  ModelV2ExecutorBuilder builder;
  ModelConverter::Args args{{}, stream_allocator, event_allocator, notify_allocator, nullptr};
  error_code =
      LoadToModelV2ExecutorBuilder(model_data, args, builder);
  GE_ASSERT_SUCCESS(error_code);
  return builder.Build(executor_option);
}

std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                const LoweringOption &optimize_option,
                                                                ge::graphStatus &error_code) {
  auto builder = ge::MakeUnique<ModelV2ExecutorBuilder>();
  GE_ASSERT_NOTNULL(builder);
  error_code = LoadToModelV2ExecutorBuilder(model_data,
      ModelConverter::Args{optimize_option, nullptr, nullptr, nullptr, nullptr}, *builder);
  GE_ASSERT_SUCCESS(error_code);
  return ge::MakeUnique<StreamExecutor>(builder.release());
}

std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                ge::graphStatus &error_code) {
  LoweringOption option;
  return LoadStreamExecutorFromModelData(model_data, option, error_code);
}

std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                const size_t weight_size, ge::graphStatus &error_code) {
  (void)weight_ptr;
  (void)weight_size;
  auto builder = ge::MakeUnique<ModelV2ExecutorBuilder>();
  GE_ASSERT_NOTNULL(builder);
  error_code = LoadToModelV2ExecutorBuilder(model_data, {}, *builder);
  GE_ASSERT_SUCCESS(error_code);
  return ge::MakeUnique<StreamExecutor>(builder.release());
}

ge::graphStatus IsDynamicModel(const void *const model, size_t model_size, bool &is_dynamic_model) {
  GE_ASSERT_NOTNULL(model, "[Check][Input] failed, model is null.");
  ge::ModelData model_data;
  if (model_size < sizeof(ge::ModelFileHeader)) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
           "[Check][Param] Invalid model size. Model data size %zu must be greater than or equal to %zu.", model_size,
           sizeof(ge::ModelFileHeader));
    return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID;
  }
  const auto file_head = reinterpret_cast<const ge::ModelFileHeader *>(model);
  is_dynamic_model = ge::ModelParserBase::IsDynamicModel(*file_head);
  GELOGD("Parser model[%p], model size[%zu], is_dynamic_model[%d].", model, model_size, is_dynamic_model);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsDynamicModel(const ge::char_t *model_path, bool &is_dynamic_model) {
  GE_ASSERT_NOTNULL(model_path, "[Check][ModelPath] failed, model_path is null.");
  ge::ModelData model_data;
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      FreeModelData(model_data);
    }
  });
  ge::graphStatus error_code = ge::ModelParserBase::LoadFromFile(model_path, -1, model_data);
  if (error_code != ge::GRAPH_SUCCESS) {
    GELOGE(error_code, "Failed to load model data from model path[%s]", model_path);
    return error_code;
  }
  const ge::ModelFileHeader *file_head = nullptr;
  GE_CHK_STATUS_RET(ge::ModelHelper::GetModelFileHead(model_data, file_head), "[Get][ModelFileHead] failed.");
  GE_ASSERT_NOTNULL(file_head);
  is_dynamic_model = ge::ModelParserBase::IsDynamicModel(*file_head);
  GELOGD("Parser model path[%s], is_dynamic_model[%d].", model_path, is_dynamic_model);
  return ge::GRAPH_SUCCESS;
}

std::unique_ptr<ge::Allocator> AllocatorFactory::Create(const TensorPlacement &placement) {
  return Create("", placement);
}

std::unique_ptr<ge::Allocator> AllocatorFactory::Create(const std::string &graph_name,
                                                        const TensorPlacement &placement) {
  int32_t device_id = 0;
  switch (placement) {
    case kOnDeviceHbm:
      (void) rtGetDevice(&device_id);
      return memory::CachingMemAllocator::GetAllocator(graph_name, device_id, RT_MEMORY_HBM);
    case kOnDeviceP2p:
      (void) rtGetDevice(&device_id);
      return memory::CachingMemAllocator::GetAllocator(graph_name, device_id, RT_MEMORY_P2P_DDR);
    case kOnHost:
    case kFollowing:
      return ge::MakeUnique<memory::HostMemAllocator>();
    default:
      GELOGE(ge::PARAM_INVALID, "Unsupported placement %d to create allocator", static_cast<int32_t>(placement));
      return nullptr;
  }
}

std::unique_ptr<ge::Allocator> CreateExternalAllocator(const ge::AllocatorDesc * const allocatorDesc) {
  return ge::MakeUnique<ExternalAllocator>(*allocatorDesc);
}
}  // namespace gert
