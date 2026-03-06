/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/model_v2_executor.h"
#include "runtime/exe_graph_executor.h"

#include <utility>
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "builder/model_v2_executor_builder.h"
#include "register/kernel_registry.h"
#include "common/model/ge_root_model.h"
#include "core/utils/tensor_utils.h"
#include "executor_error_code.h"
#include "framework/runtime/gert_const_types.h"
#include "core/executor/sequential/execution_data/sequential_execution_data.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/op/ge_op_utils.h"
#include "utils/utils.h"
#include "graph/load/model_manager/aipp_utils.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "framework/runtime/model_rt_var_manager.h"
#include "graph/manager/session_id_manager.h"

namespace gert {
namespace {
constexpr size_t kArgCount = static_cast<size_t>(ExecuteArgIndex::kNum);

ge::graphStatus CheckTensors(Tensor **const tensors, const size_t num, const char *const desc) {
  if (num > 0U) {
    if (tensors == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Failed to execute, %s is nullptr", desc);
      return ge::PARAM_INVALID;
    }
    for (size_t i = 0U; i < num; ++i) {
      if (tensors[i] == nullptr) {
        GELOGE(ge::PARAM_INVALID, "Failed to execute, %s[%zu] is nullptr", desc, i);
        return ge::PARAM_INVALID;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckModelOutputsNum(const void *void_ed, size_t num) {
  auto ed = static_cast<const SequentialExecutionData *>(void_ed);
  if (ed->output_num != num) {
    GELOGE(ge::PARAM_INVALID, "Failed to execute, outputs num %zu, expect num %zu", num, ed->output_num);
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckModelInputsNum(const void *void_ed, size_t tensor_num, size_t append_num) {
  auto ed = static_cast<const SequentialExecutionData *>(void_ed);
  size_t total_num;
  GE_ASSERT_TRUE(!ge::AddOverflow(tensor_num, append_num, total_num),
                 "Check model input num failed, add overflow, input tensor num(%zu), append input num(%zu)",
                 tensor_num, append_num);
  GE_ASSERT_TRUE((total_num == ed->input_num),
                 "Check model input num failed, input num not match, expect input num(%zu), "
                 "current total inputs num(%zu) = input tensor num(%zu) + append input num(%zu)",
                 ed->input_num, total_num, tensor_num, append_num);
  return ge::GRAPH_SUCCESS;
}
}  // namespace
using ge::Status;
ge::graphStatus ModelV2Executor::ArrangeModelLoadArg(const ModelLoadArg &arg, std::vector<void *> &const_inputs) {
  if (arg.rt_session == nullptr) {
    const_inputs.emplace_back(&default_rt_session_);
  } else {
    const_inputs.emplace_back(arg.rt_session);
  }
  const_inputs.emplace_back(const_cast<void *>(static_cast<const void *>(&(arg.outer_weight_mem))));
  const_inputs.emplace_back(const_cast<void *>(static_cast<const void *>(&(GetModelDesc()))));
  GE_ASSERT_TRUE(const_inputs.size() == static_cast<size_t>(ConstDataType::kTypeEnd));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelV2Executor::Load() {
  return Load({nullptr});
}
ge::graphStatus ModelV2Executor::Load(const ModelExecuteArg &arg) {
  return Load(arg, nullptr);
}

ge::graphStatus ModelV2Executor::OccupyStreamResource(const ModelExecuteArg &arg,
                                                      TypedContinuousVector<rtStream_t> *&streams,
                                                      TypedContinuousVector<rtEvent_t> *&events,
                                                      TypedContinuousVector<rtNotify_t> *&notifies) {
  StreamAllocator *stream_allocator;
  EventAllocator *event_allocator;
  NotifyAllocator *notifyAllocator;
  if ((arg.external_stream_allocator != nullptr) && (arg.external_event_allocator != nullptr) &&
      (arg.external_notify_allocator != nullptr)) {
    stream_allocator = arg.external_stream_allocator;
    event_allocator = arg.external_event_allocator;
    notifyAllocator = arg.external_notify_allocator;
  } else if ((arg.external_stream_allocator == nullptr) && (arg.external_event_allocator == nullptr) &&
             (arg.external_notify_allocator == nullptr)) {
    stream_allocator = &builtin_stream_allocator_;
    event_allocator = &builtin_event_allocator_;
    notifyAllocator = &builtin_notify_allocator_;
  } else {
    GELOGE(ge::PARAM_INVALID, "external_stream_allocator and external_event_allocator not allow only set one.");
    return ge::PARAM_INVALID;
  }

  size_t stream_num = model_desc_->GetReusableStreamNum() + model_desc_->GetAttachedStreamNum();
  streams = stream_allocator->AcquireStreams(stream_num);
  GE_ASSERT_NOTNULL(streams, "Failed to prepare reusable streams, num %zu. Maybe streams not enough on device",
                    stream_num);
  GE_ASSERT_TRUE(streams->GetSize() > 0u);
  streams->MutableData()[0] = arg.stream;

  events = event_allocator->AcquireEvents(model_desc_->GetReusableEventNum());
  GE_ASSERT_NOTNULL(events, "Failed to prepare reusable events, num %zu. Maybe events not enough on device",
                    model_desc_->GetReusableEventNum());
  int32_t device_id = 0;
  GE_ASSERT_RT_OK(rtGetDevice(&device_id));
  notifies = notifyAllocator->AcquireNotifies(device_id, model_desc_->GetReusableNotifyNum());
  GE_ASSERT_NOTNULL(notifies, "Failed to prepare reusable notifies, num %zu. Maybe notifies not enough on device",
                    model_desc_->GetReusableNotifyNum());

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelV2Executor::SpecifyArgsInputs(const ModelExecuteArg &arg, size_t input_num,
                                                   ExeGraphExecutor &graph_executor) {
  TypedContinuousVector<rtStream_t> *streams = nullptr;
  TypedContinuousVector<rtEvent_t> *events = nullptr;
  TypedContinuousVector<rtNotify_t> *notifies = nullptr;
  GE_RETURN_IF_ERROR(OccupyStreamResource(arg, streams, events, notifies));
  GE_RETURN_IF_ERROR(graph_executor.SpecifyInput(streams, input_num++));
  GE_RETURN_IF_ERROR(graph_executor.SpecifyInput(arg.external_allocator, input_num++));
  GE_RETURN_IF_ERROR(graph_executor.SpecifyInput(events, input_num++));
  GE_RETURN_IF_ERROR(graph_executor.SpecifyInput(notifies, input_num++));
  return ge::GRAPH_SUCCESS;
}

ge::Status ModelV2Executor::InitRtVarManager(const ModelLoadArg &load_arg) {
  RtSession *rt_session = load_arg.rt_session;
  if (rt_session == nullptr) {
    rt_session = &default_rt_session_;
  }
  if (load_session_id_ != std::numeric_limits<uint64_t>::max()) {
    GE_ASSERT(rt_session->GetSessionId() == load_session_id_,
              "Session id [%lu] from load arg mismatch with created value [%lu], this scene is not supported.",
              rt_session->GetSessionId(), load_session_id_);
    if (rt_session->GetVarManager() == nullptr) {
      auto rt_var_manager = ModelRtVarManager::Instance(load_session_id_);
      GE_ASSERT_NOTNULL(rt_var_manager);
      rt_session->SetVarManager(rt_var_manager.get());
    }
  }
  return ge::SUCCESS;
}

ge::graphStatus ModelV2Executor::Load(const ModelExecuteArg &arg, const ModelLoadArg &load_arg) {
  if (state_ != ExecutorState::kInit) {
    GELOGE(ge::PARAM_INVALID, "Can not load now, the model has been loaded(%d)", static_cast<int32_t>(state_));
    return ge::PARAM_INVALID;
  }

  auto &init_executor = graphs_[kInitExeGraph];

  auto ret = init_executor.Load();
  GE_ASSERT_SUCCESS(ret, "Failed to load init graph");

  GE_ASSERT_SUCCESS(InitRtVarManager(load_arg));
  std::vector<void *> const_inputs;
  GE_RETURN_IF_ERROR(ArrangeModelLoadArg(load_arg, const_inputs));
  GE_RETURN_IF_ERROR(CheckModelInputsNum(init_executor.GetExecutionData(), const_inputs.size(), kArgCount));
  GE_RETURN_IF_ERROR(init_executor.SpecifyInputs(const_inputs.data(), 0U, const_inputs.size()));
  GE_RETURN_IF_ERROR(SpecifyArgsInputs(arg, const_inputs.size(), init_executor));

  if (subscribers_.IsEnable()) {
    ret = init_executor.Execute(kInitExeGraph, &subscribers_.GetSubscriber(kInitExeGraph));
  } else {
    ret = init_executor.Execute();
  }
  GE_ASSERT_SUCCESS(ret, "Failed to execute init graph");
  ret = init_executor.UnLoad();
  GE_ASSERT_SUCCESS(ret, "Failed to unload init graph");
  ret = graphs_[kMainExeGraph].Load();
  GE_ASSERT_SUCCESS(ret, "Failed to load main graph");
  state_ = ExecutorState::kLoaded;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelV2Executor::UnLoad() {
  if (state_ != ExecutorState::kLoaded) {
    GELOGE(ge::PARAM_INVALID, "Can not unload now, the model state is not loaded(%d)", static_cast<int32_t>(state_));
    return ge::PARAM_INVALID;
  }
  if (default_stream_ != nullptr) {
    (void)rtStreamDestroy(default_stream_);
    default_stream_ = nullptr;
  }
  auto ret = graphs_[kMainExeGraph].UnLoad();
  GE_ASSERT_SUCCESS(ret, "Failed to unload main graph");
  ret = graphs_[kDeInitExeGraph].Load();
  GE_ASSERT_SUCCESS(ret, "Failed to load de-init graph");
  if (subscribers_.IsEnable()) {
    ret = graphs_[kDeInitExeGraph].Execute(kDeInitExeGraph, &subscribers_.GetSubscriber(kDeInitExeGraph));
  } else {
    ret = graphs_[kDeInitExeGraph].Execute();
  }
  GE_ASSERT_SUCCESS(ret, "Failed to execute de-init graph");
  ret = graphs_[kDeInitExeGraph].UnLoad();
  GE_ASSERT_SUCCESS(ret, "Failed to unload de-init graph");
  state_ = ExecutorState::kInit;

  // todo:OpImplRegistryHolderManager::GetInstance().UpdateOpImplRegistries();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelV2Executor::Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num,
                                         Tensor **outputs, size_t output_num) {
  if (state_ != ExecutorState::kLoaded) {
    GELOGE(ge::PARAM_INVALID, "Failed to execute model, you may need load model first");
    return ge::PARAM_INVALID;
  }

  auto &graph_executor = graphs_[kMainExeGraph];
  GE_RETURN_IF_ERROR(CheckModelInputsNum(graph_executor.GetExecutionData(), input_num, kArgCount));
  GE_RETURN_IF_ERROR(CheckTensors(inputs, input_num, "inputs"));
  GE_RETURN_IF_ERROR(graph_executor.SpecifyInputs(reinterpret_cast<void *const *>(inputs), 0U, input_num));
  GE_RETURN_IF_ERROR(SpecifyArgsInputs(arg, input_num, graph_executor));

  GE_RETURN_IF_ERROR(CheckModelOutputsNum(graph_executor.GetExecutionData(), output_num));
  GE_RETURN_IF_ERROR(CheckTensors(outputs, output_num, "outputs"));
  GE_RETURN_IF_ERROR(graph_executor.SpecifyOutputs(reinterpret_cast<void *const *>(outputs), output_num));

  GE_RETURN_IF_ERROR(CheckIoReuseAddrs(inputs, input_num, outputs, output_num));

  if (subscribers_.IsEnable()) {
    return graph_executor.Execute(kMainExeGraph, &subscribers_.GetSubscriber(kMainExeGraph));
  } else {
    return graph_executor.Execute();
  }
}
ge::graphStatus ModelV2Executor::ExecuteSync(Tensor **inputs, size_t input_num, Tensor **outputs, size_t output_num) {
  if (default_stream_ == nullptr) {
    GE_CHK_RT_RET(rtStreamCreateWithFlags(&default_stream_, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT),
                                          RT_STREAM_FAST_LAUNCH | RT_STREAM_FAST_SYNC));
  }

  auto ret = Execute({default_stream_}, inputs, input_num, outputs, output_num);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }
  GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(default_stream_));
  return ge::GRAPH_SUCCESS;
}
const ModelDesc &ModelV2Executor::GetModelDesc() const {
  if (model_desc_ != nullptr) {
    return *model_desc_;
  } else {
    // Usually impossible
    GELOGE(ge::GRAPH_FAILED, "[Get][ModelDesc] failed, model_desc is null.");
    static ModelDesc default_model_desc;
    return default_model_desc;
  }
}

void ModelV2Executor::SetModelDesc(ModelDesc *model_desc) {
  model_desc_ = model_desc;
}

std::unique_ptr<ModelV2Executor> ModelV2Executor::Create(const ge::ExecuteGraphPtr &exe_graph,
                                                         const ge::GeRootModelPtr &root_model,
                                                         RtSession *session) {
  if (exe_graph == nullptr) {
    return nullptr;
  }
  return ModelV2ExecutorBuilder(session).ExecuteGraph(exe_graph).GeRootModel(root_model).Build();
}
std::unique_ptr<ModelV2Executor> ModelV2Executor::Create(const ge::ExecuteGraphPtr &exe_graph,
                                                         const ExecutorOption &option,
                                                         const ge::GeRootModelPtr &root_model,
                                                         RtSession *session) {
  if (exe_graph == nullptr) {
    return nullptr;
  }
  return ModelV2ExecutorBuilder(session).ExecuteGraph(exe_graph).GeRootModel(root_model).Build(option);
}
std::unique_ptr<ModelV2Executor> ModelV2Executor::Create(const ge::ExecuteGraphPtr &exe_graph,
                                                         const ge::ModelData &model_data,
                                                         const ge::GeRootModelPtr &root_model,
                                                         RtSession *session) {
  if (exe_graph == nullptr) {
    return nullptr;
  }
  return ModelV2ExecutorBuilder(session).ExecuteGraph(exe_graph).ModelData(model_data).GeRootModel(root_model).Build();
}
const ExecutorSubscribersScheduler &ModelV2Executor::GetSubscribers() const {
  return subscribers_;
}
ExecutorSubscribersScheduler &ModelV2Executor::GetSubscribers() {
  return subscribers_;
}
ModelV2Executor::ModelV2Executor()
    : resource_guard_(), graphs_(), model_desc_(nullptr), default_stream_(nullptr), subscribers_() {}

ge::graphStatus ExeGraphExecutor::Execute() const {
  auto ret = execute_func_(execution_data_);
  if (ret != kStatusSuccess) {
    return ret;
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ExeGraphExecutor::Execute(SubExeGraphType sub_graph_type, ExecutorSubscriber *callback) const {
  auto ret = execute_with_callback_func_(sub_graph_type, execution_data_, callback);
  if (ret != kStatusSuccess) {
    return ret;
  }
  return ge::GRAPH_SUCCESS;
}
void ExeGraphExecutor::SetExecutionData(void *execution_data, ResourceGuardPtr resource_guard) {
  execution_data_ = execution_data;
  resource_guard_ = std::move(resource_guard);
}
ge::graphStatus ExeGraphExecutor::SpecifyInputs(void *const *inputs, size_t start, size_t num) const {
  auto ed = reinterpret_cast<SequentialExecutionData *>(execution_data_);
  for (size_t i = 0U; i < num; ++i) {
    reinterpret_cast<Chain *>(ed->input_values[start + i])->Set(inputs[i], nullptr);
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ExeGraphExecutor::SpecifyOutputs(void *const *outputs, size_t num) const {
  auto ed = reinterpret_cast<SequentialExecutionData *>(execution_data_);
  if (ed->output_num != num) {
    GELOGE(ge::PARAM_INVALID, "Failed to execute, outputs num %zu, expect num %zu", num, ed->output_num);
    return ge::PARAM_INVALID;
  }
  for (size_t i = 0U; i < num; ++i) {
    reinterpret_cast<Chain *>(ed->output_values[i])->Set(outputs[i], nullptr);
  }

  return ge::GRAPH_SUCCESS;
}
void ExeGraphExecutor::SetExecuteFunc(ExeGraphExecutor::ExecuteFunc execute_func,
                                      ExeGraphExecutor::ExecuteWithCallbackFunc callback_func) {
  execute_func_ = execute_func;
  execute_with_callback_func_ = callback_func;
}

ge::Status ModelV2Executor::InitAipp(const ge::ComputeGraphPtr &root_graph) {
  GE_ASSERT_NOTNULL(root_graph);
  uint32_t data_index = 0U;
  std::map<std::string, uint32_t> data_index_map;
  for (const auto &data_node : root_graph->GetInputNodes()) {
    data_index_map[data_node->GetName()] = data_index;
    data_index++;
  }
  data_index = 0U;
  for (const auto &data_node : root_graph->GetInputNodes()) {
    GE_ASSERT_NOTNULL(data_node);
    GE_ASSERT_SUCCESS(ge::AippUtils::SetAippInfoAndTypeFromOpDesc(data_index_map, data_node->GetOpDesc(), data_index,
        aipp_info_list_, aipp_type_list_));
    GE_ASSERT_SUCCESS(ge::AippUtils::SetAippInputOutputInfoFromOpDesc(data_node->GetOpDesc(), data_index,
        orig_aipp_input_info_, aipp_dims_info_));
    ++data_index;
  }
  return ge::SUCCESS;
}

ge::Status ModelV2Executor::GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info) const {
  return ge::AippUtils::GetAippInfo(aipp_info_list_, index, aipp_info);
}

ge::Status ModelV2Executor::GetAippType(const uint32_t index, ge::InputAippType &aipp_type, size_t &aipp_index) const {
  return ge::AippUtils::GetAippType(aipp_type_list_, index, aipp_type, aipp_index);
}

ge::Status ModelV2Executor::GetOriginAippInputInfo(const uint32_t index,
    ge::OriginInputInfo &orig_aipp_input_info) const {
  return ge::AippUtils::GetOrigInputInfo(orig_aipp_input_info_, index, orig_aipp_input_info);
}

ge::Status ModelV2Executor::GetAllAippInputOutputDims(const uint32_t index,
    std::vector<ge::InputOutputDims> &input_dims, std::vector<ge::InputOutputDims> &output_dims) const {
  return ge::AippUtils::GetAllAippInputOutputDims(aipp_dims_info_, index, input_dims, output_dims);
}

uint32_t ModelV2Executor::GetIterationNum() const {
  const auto cann_profiler = subscribers_.GetBuiltInSubscriber<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2);
  if (cann_profiler == nullptr) {
    return 0U;
  }
  return cann_profiler->GetIterationNum();
}

ge::graphStatus ModelV2Executor::CheckIoReuseAddrs(Tensor **inputs, size_t input_num,
                                                    Tensor **outputs, size_t output_num) const {
  if (io_same_addr_pairs_.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  if (inputs == nullptr || outputs == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (input_num == 0U || output_num == 0U) {
    return ge::GRAPH_SUCCESS;
  }

  AddrGetter input_getter;
  AddrGetter output_getter;
  for (size_t i = 0; i < input_num; ++i) {
    input_getter = [&](size_t i) { return inputs[i]->GetAddr(); };
  }
  for (size_t i = 0; i < output_num; ++i) {
    output_getter = [&](size_t i) { return outputs[i]->GetAddr(); };
  }

  GE_ASSERT_SUCCESS(ge::CheckIoReuseAddrPairs(io_same_addr_pairs_, input_getter, input_num,
                                              output_getter, output_num));
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
