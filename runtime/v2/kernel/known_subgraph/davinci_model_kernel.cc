/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/known_subgraph/davinci_model_kernel.h"
#include <cstddef>
#include <iostream>
#include <algorithm>
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "exe_graph/runtime/tensor.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "kernel/memory/mem_block.h"
#include "common/checker.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/reusable_stream_allocator.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "common/model/ge_model.h"
#include "kernel/known_subgraph/davinci_model_tracing.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "common/dump/dump_manager.h"
#include "framework/common/ge_types.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "core/debug/kernel_tracing.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_context.h"

namespace gert {
namespace kernel {
namespace {
uintptr_t GetRunMemory(const ge::DavinciModel &davinci_model, const MemoryBaseTypeOffset &type_offset_pair) {
  const ge::RuntimeParam &param = davinci_model.GetRuntimeParam();
  switch (type_offset_pair.base_type) {
    case MemoryBaseType::kMemoryBaseTypeWeight:
      return param.weight_base + type_offset_pair.offset;
    case MemoryBaseType::kMemoryBaseTypeFileConstant:
      if (param.fileconstant_addr_mapping.find(type_offset_pair.offset) != param.fileconstant_addr_mapping.cend()) {
        return param.fileconstant_addr_mapping.at(type_offset_pair.offset);
      }
      GELOGE(ge::FAILED, "can not find offset[%ld] in fileconstant_addr_mapping", type_offset_pair.offset);
      return 0;
    default:
      return 0;
  }
  return 0;
}

void SetDaviciModel(ge::DavinciModel &davinci_model, const ge::GeModelPtr &model) {
  davinci_model.SetKnownNode(true);
  davinci_model.SetId(model->GetModelId());
  int32_t device_id = 0;
  rtGetDevice(&device_id);
  davinci_model.SetDeviceId(static_cast<uint32_t>(device_id));
  davinci_model.SetOmName(model->GetOmName());
}

ge::GeModelPtr GetGeModel(const KernelContext *const context) {
  const auto ge_model_holder = context->GetInputValue<ge::GeModel *>(0U);
  GE_ASSERT_NOTNULL(ge_model_holder);
  return ge_model_holder->shared_from_this();
}

ge::Status UpdateModelGraphInputIndex(const ge::ComputeGraphPtr &graph, std::set<uint32_t> &input_index_set) {
  GE_ASSERT_NOTNULL(graph);
  for (const auto node : graph->GetDirectNodePtr()) {
    if (node->GetType() != ge::DATA) {
      continue;
    }
    const auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    uint32_t parent_index = 0U;
    if (ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGI("Update model %s data %s index to %u", graph->GetName().c_str(), node->GetNamePtr(), parent_index);
      GE_ASSERT(ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, parent_index));
      (void)input_index_set.insert(parent_index);
    }
  }
  return ge::SUCCESS;
}

ge::Status GetValidFrozenIndicies(const std::string &frozen_inputs, const ge::ComputeGraphPtr &graph,
                                  std::set<uint32_t> &input_index_set,
                                  std::unordered_set<uint32_t> &valid_frozen_index_set) {
  if (frozen_inputs.empty()) {
    GELOGD("Frozen inputs option is empty. Skip to set frozen inputs fro static model.");
    return ge::SUCCESS;
  }
  // Parse frozen inputs option
  std::vector<std::string> frozen_inputs_list = ge::StringUtils::Split(frozen_inputs, ';');
  std::set<uint32_t> all_frozen_indicies;
  for (auto &index : frozen_inputs_list) {
    std::vector<std::string> index_list = ge::StringUtils::Split(index, ',');
    GE_ASSERT_TRUE(!index_list.empty(), "Split frozen string [%s] failed.", index.c_str());
    int32_t frozen_input_index = -1;
    GE_ASSERT_SUCCESS(ge::ConvertToInt32(index_list[0UL], frozen_input_index), "Convert frozen input index failed.");
    GE_ASSERT_TRUE(frozen_input_index >= 0, "Frozen input index should be greater than or equal to 0.");
    (void)all_frozen_indicies.insert(static_cast<uint32_t>(frozen_input_index));
  }
  GE_ASSERT_NOTNULL(graph);
  const auto parent_node = graph->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_node);
  for (const auto &parent_index : input_index_set) {
    // get data index in root graph
    const auto out_node = ge::NodeUtils::GetInDataNodeByIndex(*parent_node, parent_index);
    GE_ASSERT_NOTNULL(out_node);
    if (!ge::OpTypeUtils::IsDataNode(out_node->GetType())) {
      continue;
    }
    const auto &out_data_op_desc = out_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(out_data_op_desc);
    int32_t index = -1;
    (void)ge::AttrUtils::GetInt(out_data_op_desc, ge::ATTR_NAME_INDEX, index);
    GE_ASSERT_TRUE(index >= 0, "Input index should be greater than or equal to 0.");
    // check data index from root graph frozen or not
    if (all_frozen_indicies.find(static_cast<uint32_t>(index)) != all_frozen_indicies.cend()) {
      (void)valid_frozen_index_set.insert(parent_index);
      GELOGI("Set static model input index[%u] frozen mapping root graph input index[%u]", parent_index, index);
    }
  }
  return ge::SUCCESS;
}
}
bool IsNeedMallocFixedMemoryOnInitGraph(const void *fixed_feature_mem, const size_t fixed_size) {
  // 用户如果通过这个option设置了fixed内存，也不需要GE申请了
  std::string is_addr_fixed_opt;
  (void)ge::GetContext().GetOption("ge.exec.static_model_addr_fixed", is_addr_fixed_opt);
  if (!is_addr_fixed_opt.empty()) {
    GELOGI("user set ge.exec.static_model_addr_fixed option, return false");
    return false;
  }
  return (fixed_feature_mem == nullptr) && (fixed_size > 0U);
}

/*
 * 如果传递给davinci model的是正常的地址，使用这个地址, 如果设置了nullptr地址，davinci model中会再申请地址
 */
ge::graphStatus InitParam(KernelContext *context, ge::ModelParam &param) {
  const auto weight_tensor = context->GetInputPointer<GertTensorData>(
      static_cast<size_t>(DavinciModelCreateInput::kAssignMem));
  GE_ASSERT_NOTNULL(weight_tensor, "There is no weight info.");

  param.weight_base = ge::PtrToValue(weight_tensor->GetAddr());
  param.weight_size = weight_tensor->GetSize();

  const auto fixed_mem_addr =
      context->GetInputValue<uintptr_t>(static_cast<size_t>(DavinciModelCreateInput::kFixedMemAddr));
  const auto fixed_mem_size =
      context->GetInputValue<size_t>(static_cast<size_t>(DavinciModelCreateInput::kFixedMemSize));
  param.fixed_mem_base = fixed_mem_addr;
  param.fixed_mem_size = fixed_mem_size;

  // 是否使用init graph中的内存申请，可以搜索MallocFixedFeatureMemIfNeed日志
  if (IsNeedMallocFixedMemoryOnInitGraph(ge::ValueToPtr(fixed_mem_addr), fixed_mem_size)) {
    const auto tensor_data = context->GetInputPointer<TensorData>(
        static_cast<size_t>(DavinciModelCreateInput::kFixedMemTensorFromInit));
    GE_ASSERT_NOTNULL(tensor_data, "get hbm fixed_feautre_memory tensor data failed.");
    GE_ASSERT_NOTNULL(tensor_data->GetAddr(), "get hbm fixed_feautre_memory addr failed.");
    param.fixed_mem_base = ge::PtrToValue(tensor_data->GetAddr());
    param.fixed_mem_size = tensor_data->GetSize();
  }

  const auto p2p_fixed_mem_addr =
      context->GetInputValue<uintptr_t>(static_cast<size_t>(DavinciModelCreateInput::kP2pFixedMemAddr));
  const auto p2p_fixed_mem_size =
      context->GetInputValue<size_t>(static_cast<size_t>(DavinciModelCreateInput::kP2pFixedMemSize));
  param.p2p_fixed_mem_base = p2p_fixed_mem_addr;
  param.p2p_fixed_mem_size = p2p_fixed_mem_size;

  // 是否使用init graph中的内存申请，可以搜索MallocFixedFeatureMemIfNeed日志
  if (IsNeedMallocFixedMemoryOnInitGraph(ge::ValueToPtr(p2p_fixed_mem_addr), p2p_fixed_mem_size)) {
    const auto tensor_data = context->GetInputPointer<TensorData>(
        static_cast<size_t>(DavinciModelCreateInput::kP2pFixedMemTensorFromInit));
    GE_ASSERT_NOTNULL(tensor_data, "get p2p fixed_feautre_memory tensor data failed.");
    GE_ASSERT_NOTNULL(tensor_data->GetAddr(), "get p2p fixed_feautre_memory addr failed.");
    param.p2p_fixed_mem_base = ge::PtrToValue(tensor_data->GetAddr());
    param.p2p_fixed_mem_size = tensor_data->GetSize();
  }
  KERNEL_TRACE("[MEM] fixed_feature_memory hbm_addr %p, hbm_size %zu, p2p_addr %p, p2p_size %zu",
               param.fixed_mem_base, param.fixed_mem_size, param.p2p_fixed_mem_base, param.p2p_fixed_mem_size);
  return ge::SUCCESS;
}

ge::graphStatus DavinciModelCreate(KernelContext *context) {
  GE_CHECK_NOTNULL(context);
  ge::GeModelPtr ge_model = GetGeModel(context);
  GE_CHECK_NOTNULL(ge_model);
  std::set<uint32_t> input_index_set;
  GE_ASSERT_SUCCESS(UpdateModelGraphInputIndex(ge_model->GetGraph(), input_index_set));

  auto davinci_model_ptr = ge::MakeUnique<ge::DavinciModel>(0, nullptr);
  GE_CHECK_NOTNULL(davinci_model_ptr);
  davinci_model_ptr->Assign(ge_model);
  SetDaviciModel(*davinci_model_ptr.get(), ge_model);
  const size_t session_id_index = 1U;
  const auto session_id_ptr = context->GetInputPointer<uint64_t>(session_id_index);
  GE_CHECK_NOTNULL(session_id_ptr);
  davinci_model_ptr->UpdateSessionId(*session_id_ptr);
  const size_t step_id_index = static_cast<size_t>(DavinciModelCreateInput::kStepId);
  davinci_model_ptr->SetGlobalStep(ge::PtrToValue(context->GetInputValue<void *>(step_id_index)), sizeof(int64_t));
  const uint32_t root_graph_id =
      context->GetInputValue<uint32_t>(static_cast<size_t>(DavinciModelCreateInput::kRootGraphId));
  davinci_model_ptr->SetRootGraphId(root_graph_id);
  GE_ASSERT_SUCCESS(davinci_model_ptr->InitRuntimeParams());
  auto space_registries_ptr = context->GetInputValue<gert::OpImplSpaceRegistryV2Array *>(
    static_cast<size_t>(DavinciModelCreateInput::kSpaceRegistry));
  GE_CHECK_NOTNULL(space_registries_ptr);
  davinci_model_ptr->SetSpaceRegistries(std::make_shared<OpImplSpaceRegistryV2Array>(*space_registries_ptr));

  const auto file_constant_weight_dir_holder =
      context->GetInputPointer<ge::char_t *>(static_cast<size_t>(DavinciModelCreateInput::kFileConstantWeightDir));
  GE_ASSERT_NOTNULL(file_constant_weight_dir_holder);
  std::string file_constant_weight_dir(*file_constant_weight_dir_holder);
  GELOGD("Get file constant weight dir [%s] for davinci model.", file_constant_weight_dir.c_str());
  davinci_model_ptr->SetFileConstantWeightDir(file_constant_weight_dir);

  const auto reusable_stream_allocator = context->GetInputValue<ge::ReusableStreamAllocator *>(
      static_cast<size_t>(DavinciModelCreateInput::kRtStreamReuse));
  GE_CHECK_NOTNULL(reusable_stream_allocator);
  davinci_model_ptr->SetReusableStreamAllocator(reusable_stream_allocator);

  const auto &file_constant_names_and_mems =
      context->GetInputPointer<ContinuousVector>(static_cast<size_t>(DavinciModelCreateInput::kFileConstantUserMem));
  GE_CHECK_NOTNULL(file_constant_names_and_mems);
  GE_CHECK_NOTNULL(file_constant_names_and_mems->GetData());
  const FileConstantNameAndMem *const file_constant_names_and_mems_ptr =
      reinterpret_cast<const FileConstantNameAndMem *>(file_constant_names_and_mems->GetData());
  for (size_t i = 0U; i < file_constant_names_and_mems->GetSize(); ++i) {
    const FileConstantNameAndMem *const item = &file_constant_names_and_mems_ptr[i];
    GE_ASSERT_NOTNULL(item);
    davinci_model_ptr->SetFileConstantDeviceMem(std::string(item->name), item->mem, item->size);
    KERNEL_TRACE("FileConstant use user device memory. file name: %s, addr: %p, size: %zu",
                 item->name, item->mem, item->size);
  }

  auto ret = davinci_model_ptr->InitVariableMem();
  if (ret != ge::SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "davinci model init variable memory failed");
    return ret;
  }
  const auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  if (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen()) {
    davinci_model_ptr->SetDumpProperties(dump_properties);
  }
  ge::ModelParam param{};
  GE_ASSERT_SUCCESS(InitParam(context, param));
  const auto frozen_indices_holder = context->GetInputPointer<ge::char_t *>(
      static_cast<size_t>(DavinciModelCreateInput::kFrozenInputIndicies));
  GE_ASSERT_NOTNULL(frozen_indices_holder);
  std::string frozen_inputs(*frozen_indices_holder);
  std::unordered_set<uint32_t> valid_frozen_inputs;
  GE_ASSERT_SUCCESS(GetValidFrozenIndicies(frozen_inputs, ge_model->GetGraph(), input_index_set, valid_frozen_inputs),
                    "Get valid frozen indicies failed");
  davinci_model_ptr->SetNoFrozenInputIndexes(valid_frozen_inputs);
  ret = davinci_model_ptr->Init(param);
  GE_ASSERT_SUCCESS(ret, "davinci model init failed");
  auto chain = context->GetOutput(0U);
  GE_CHECK_NOTNULL(chain);
  chain->SetWithDefaultDeleter(davinci_model_ptr.get());
  GELOGI("create davinci model successfully, model id:%u.", davinci_model_ptr->GetModelId());
  davinci_model_ptr.release();
  return ge::SUCCESS;
}

ge::graphStatus UpdateMemBase(ge::DavinciModel &davinci_model,
                              std::vector<uint64_t> &workspaces_memory_type,
                              std::vector<void *> &workspaces) {
  std::vector<uint8_t *> hbm_fm_mem_bases;
  for (size_t i = 0U; i < workspaces_memory_type.size(); ++i) {
    if (workspaces_memory_type[i] == RT_MEMORY_HBM) {
      hbm_fm_mem_bases.push_back(reinterpret_cast<uint8_t *>(workspaces[i]));
      continue;
    }
    const auto ret = davinci_model.UpdateExMemBase(workspaces_memory_type[i], static_cast<uint8_t *>(workspaces[i]));
    if (ret != ge::SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "davinci model update memory base failed. memory type[0x%lx]",
             workspaces_memory_type[i]);
      return ret;
    }
  }
  GE_ASSERT_SUCCESS(davinci_model.UpdateHbmFmMemBases(hbm_fm_mem_bases));
  return ge::SUCCESS;
}

ge::graphStatus DavinciModelUpdateArgs(KernelContext *context) {
  GE_CHECK_NOTNULL(context);
  auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsCommon::kDavinciModel));
  GE_CHECK_NOTNULL(davinci_model);
  auto stream = context->GetInputValue<void *>(static_cast<int32_t>(ModelExecute::kStream));
  GE_CHECK_NOTNULL(stream);
  davinci_model->SetAsyncMode(true);
  auto ret = davinci_model->InitModelStream(stream);
  if (ret != ge::SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "davinci model init model stream:%p failed.", stream);
    return ret;
  }
  const auto input_num = context->GetInputPointer<size_t>(static_cast<int32_t>(ModelExecute::kInputNum));
  GE_CHECK_NOTNULL(input_num);
  const auto output_num = context->GetInputPointer<size_t>(static_cast<int32_t>(ModelExecute::kOutputNum));
  GE_CHECK_NOTNULL(output_num);

  std::vector<uint64_t> v_inputs(*input_num);
  for (size_t i = 0UL; i < *input_num; ++i) {
    const auto gtd = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(ModelExecute::kModelExecuteEnd) + i);
    GE_CHECK_NOTNULL(gtd);
    v_inputs[i] = ge::PtrToValue(gtd->GetAddr());
  }

  std::vector<uint64_t> v_outputs(*output_num);
  for (size_t i = 0UL; i < *output_num; ++i) {
    const auto gtd = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(ModelExecute::kModelExecuteEnd) + *input_num + i);
    GE_CHECK_NOTNULL(gtd);
    v_outputs[i] = ge::PtrToValue(gtd->GetAddr());
  }
  ret = davinci_model->UpdateKnownNodeArgs(v_inputs, v_outputs);
  if (ret != ge::SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "davinci model update known node args failed.");
  }
  return ret;
}

ge::graphStatus ConstructOutputData(const KernelContext *context, ge::OutputData &output_data) {
  GE_CHECK_NOTNULL(context);
  auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsCommon::kDavinciModel));
  GE_CHECK_NOTNULL(davinci_model);

  const auto input_num = context->GetInputPointer<size_t>(static_cast<int32_t>(ModelExecute::kInputNum));
  GE_CHECK_NOTNULL(input_num);
  const auto output_num = context->GetInputPointer<size_t>(static_cast<int32_t>(ModelExecute::kOutputNum));
  GE_CHECK_NOTNULL(output_num);

  output_data.blobs.resize(*output_num);
  for (size_t i = 0UL; i < *output_num; ++i) {
    const auto gtd = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(ModelExecute::kModelExecuteEnd) + *input_num + i);
    GE_CHECK_NOTNULL(gtd);
    const auto placement = TensorPlacementUtils::IsOnDevice(gtd->GetPlacement()) ? ge::Placement::kPlacementDevice :
                           ge::Placement::kPlacementHost;
    ge::DataBuffer data_buffer{gtd->GetAddr(), gtd->GetSize(), false, placement};
    output_data.blobs[i] = std::move(data_buffer);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DavinciModelUpdateWorkspaces(KernelContext *context) {
  GE_CHECK_NOTNULL(context);
  auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsCommon::kDavinciModel));
  GE_CHECK_NOTNULL(davinci_model);
  const auto workspace_num = context->GetInputPointer<size_t>(static_cast<int32_t>(UpdateWorkspaces::kWorkspacesNum));
  GE_CHECK_NOTNULL(workspace_num);

  std::vector<uint64_t> types;
  std::vector<void *> addresses;
  for (size_t i = 0U; i < *workspace_num;) {
    const auto memory_type = context->GetInputPointer<uint64_t>(
        static_cast<int32_t>(UpdateWorkspaces::kWorkspaceMemory) + (i++));
    const auto gtd = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(UpdateWorkspaces::kWorkspaceMemory) + (i++));
    GE_CHECK_NOTNULL(memory_type);
    GE_CHECK_NOTNULL(gtd);
    types.emplace_back(*memory_type);
    addresses.emplace_back(gtd->GetAddr());
  }
  const auto ret = UpdateMemBase(*davinci_model, types, addresses);
  if (ret != ge::SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "update memory base failed, memory type size[%zu].", types.size());
    return ret;
  }

  return ge::SUCCESS;
}

ge::graphStatus DavinciModelExecute(KernelContext *context) {
  GE_CHECK_NOTNULL(context);
  auto ret = DavinciModelUpdateArgs(context);
  if (ret != RT_ERROR_NONE) {
    GELOGE(ge::GRAPH_FAILED, "model update args failed. ret = %d", ret);
    return ge::GRAPH_FAILED;
  }

  auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsCommon::kDavinciModel));
  GE_CHECK_NOTNULL(davinci_model);
  auto stream = context->GetInputValue<void *>(static_cast<int32_t>(ModelExecute::kStream));
  GE_CHECK_NOTNULL(stream);

  ret = rtModelExecute(davinci_model->GetRtModelHandle(), stream, 0U);
  if (ret != RT_ERROR_NONE) {
    GELOGE(ge::GRAPH_FAILED, "model execute failed. ret = %d", ret);
    return ge::GRAPH_FAILED;
  }

  ge::OutputData output_data;
  GE_ASSERT_SUCCESS(ConstructOutputData(context, output_data));
  const std::vector<ge::GeTensor> output_tensor = {};
  GE_ASSERT_SUCCESS(davinci_model->CopyOutputData(output_data, output_tensor));
  GE_ASSERT_SUCCESS(davinci_model->LaunchEventForHcclGroupOrderedStream(stream));
  return ge::SUCCESS;
}

ge::graphStatus CreateGetRunAddressOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto chain = context->GetOutput(i);
    GE_CHECK_NOTNULL(chain);
    auto gtd = new (std::nothrow) GertTensorData(0, kOnDeviceHbm, -1, nullptr);
    GE_ASSERT_NOTNULL(gtd);
    chain->SetWithDefaultDeleter(gtd);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetRunAddress(KernelContext *context) {
  GE_CHECK_NOTNULL(context);

  const auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsSpecial::kDavinciModel));
  auto stream_id = context->GetInputPointer<int64_t>(static_cast<size_t>(InputsSpecial::kStreamId));
  GE_ASSERT_NOTNULL(stream_id);
  GE_CHECK_NOTNULL(davinci_model);

  for (size_t i = static_cast<size_t>(InputsSpecial::kInputsCommonEnd); i < context->GetInputNum(); ++i) {
    const auto type_offset_pair = context->GetInputPointer<MemoryBaseTypeOffset>(static_cast<size_t>(i));
    GE_CHECK_NOTNULL(type_offset_pair);
    size_t output_idx = i - static_cast<size_t>(InputsSpecial::kInputsCommonEnd);
    auto gtd = context->GetOutputPointer<GertTensorData>(output_idx);
    GE_CHECK_NOTNULL(gtd);
    const uintptr_t run_address = GetRunMemory(*davinci_model, *type_offset_pair);
    if (run_address == 0U) {
      GELOGE(ge::FAILED, "get run address is zero, type:%u, offset:%ld",
             static_cast<uint32_t>(type_offset_pair->base_type), type_offset_pair->offset);
      return ge::FAILED;
    }
    *gtd = GertTensorData{reinterpret_cast<void *>(run_address), static_cast<size_t>(type_offset_pair->size),
                          gtd->GetPlacement(), *stream_id};
  }
  return ge::SUCCESS;
}

ge::graphStatus DavinciModelCreateV2(KernelContext *context) {
  GE_CHECK_NOTNULL(context);
  ge::GeModelPtr ge_model = GetGeModel(context);
  GE_CHECK_NOTNULL(ge_model);
  std::set<uint32_t> input_index_set_no_use;
  GE_ASSERT_SUCCESS(UpdateModelGraphInputIndex(ge_model->GetGraph(), input_index_set_no_use));

  auto davinci_model_ptr = ge::MakeUnique<ge::DavinciModel>(0, nullptr);
  GE_CHECK_NOTNULL(davinci_model_ptr);
  davinci_model_ptr->Assign(ge_model);
  SetDaviciModel(*davinci_model_ptr.get(), ge_model);
  const size_t session_id_index = 1U;
  const auto session_id_ptr = context->GetInputPointer<uint64_t>(session_id_index);
  GE_CHECK_NOTNULL(session_id_ptr);
  davinci_model_ptr->UpdateSessionId(*session_id_ptr);
  const size_t step_id_index = static_cast<size_t>(DavinciModelCreateInput::kStepId);
  davinci_model_ptr->SetGlobalStep(ge::PtrToValue(context->GetInputValue<void *>(step_id_index)), sizeof(int64_t));
  const uint32_t root_graph_id =
      context->GetInputValue<uint32_t>(static_cast<size_t>(DavinciModelCreateInput::kRootGraphId));
  davinci_model_ptr->SetRootGraphId(root_graph_id);
  GE_ASSERT_SUCCESS(davinci_model_ptr->InitRuntimeParams());
  auto space_registries_ptr = context->GetInputValue<gert::OpImplSpaceRegistryV2Array *>(
    static_cast<size_t>(DavinciModelCreateInput::kSpaceRegistry));
  GE_CHECK_NOTNULL(space_registries_ptr);
  davinci_model_ptr->SetSpaceRegistries(std::make_shared<OpImplSpaceRegistryV2Array>(*space_registries_ptr));

  const auto file_constant_weight_dir_holder =
      context->GetInputPointer<ge::char_t *>(static_cast<size_t>(DavinciModelCreateInput::kFileConstantWeightDir));
  GE_ASSERT_NOTNULL(file_constant_weight_dir_holder);
  std::string file_constant_weight_dir(*file_constant_weight_dir_holder);
  GELOGD("Get file constant weight dir [%s] for davinci model.", file_constant_weight_dir.c_str());
  davinci_model_ptr->SetFileConstantWeightDir(file_constant_weight_dir);

  const auto reusable_stream_allocator = context->GetInputValue<ge::ReusableStreamAllocator *>(
      static_cast<size_t>(DavinciModelCreateInput::kRtStreamReuse));
  GE_CHECK_NOTNULL(reusable_stream_allocator);
  davinci_model_ptr->SetReusableStreamAllocator(reusable_stream_allocator);

  GE_ASSERT_SUCCESS(davinci_model_ptr->InitVariableMem());
  const auto dump_properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  if (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen()) {
    davinci_model_ptr->SetDumpProperties(dump_properties);
  }
  const auto weight_tensor = context->GetInputPointer<GertTensorData>(2U);
  GE_CHECK_NOTNULL(weight_tensor);
  ge::ModelParam param{};
  param.weight_base = ge::PtrToValue(weight_tensor->GetAddr());
  param.weight_size = weight_tensor->GetSize();
  const auto outer_fm_mem = context->GetInputPointer<TensorData>(8U);
  GE_ASSERT_NOTNULL(outer_fm_mem);
  GE_ASSERT_SUCCESS(davinci_model_ptr->Init(param, outer_fm_mem->GetAddr()));

  auto chain = context->GetOutput(0U);
  GE_CHECK_NOTNULL(chain);
  chain->SetWithDefaultDeleter(davinci_model_ptr.get());
  GELOGI("create davinci model successfully, model id:%u.", davinci_model_ptr->GetModelId());
  davinci_model_ptr.release();
  return ge::SUCCESS;
}

REGISTER_KERNEL(DavinciModelCreate).RunFunc(DavinciModelCreate).TracePrinter(PrintModelCreate);
REGISTER_KERNEL(DavinciModelCreateV2).RunFunc(DavinciModelCreateV2).TracePrinter(PrintModelCreate);
REGISTER_KERNEL(DavinciModelUpdateWorkspaces).RunFunc(DavinciModelUpdateWorkspaces).TracePrinter(PrintWorkspaces);
REGISTER_KERNEL(DavinciModelExecute).RunFunc(DavinciModelExecute).TracePrinter(PrintModelExecute);
REGISTER_KERNEL(DavinciModelGetRunAddress)
    .RunFunc(GetRunAddress)
    .OutputsCreator(CreateGetRunAddressOutputs)
    .TracePrinter(PrintGetRunAddress);
} // namespace kernel
} // namespace gert
