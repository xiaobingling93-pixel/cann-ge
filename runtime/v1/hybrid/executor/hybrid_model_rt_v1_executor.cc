/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/hybrid_model_rt_v1_executor.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/tensor_utils.h"
#include "common/dump/dump_manager.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling_definitions.h"
#include "hybrid/executor/host_cpu_callback_manager.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "common/profiling_definitions.h"
#include "common/memory/tensor_trans_utils.h"
#include "formats/utils/formats_trans_utils.h"
namespace ge {
namespace {
InputData GetInputDataFromGertTensors(const std::vector<gert::Tensor> &inputs) {
  InputData input_data{};
  for (auto &tensor : inputs) {
    DataBuffer buffer;
    buffer.data = const_cast<void *>(tensor.GetAddr());
    buffer.length = tensor.GetSize();
    buffer.placement = static_cast<uint32_t>(gert::TensorPlacementUtils::IsOnDevice(tensor.GetPlacement()) ?
      kPlacementDevice : kPlacementHost);
    input_data.blobs.emplace_back(buffer);
    const auto &shape = tensor.GetStorageShape();
    std::vector<int64_t> dims(shape.GetDimNum());
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
      dims[i] = shape[i];
    }
    input_data.shapes.emplace_back(dims);
  }
  return input_data;
}

Status TensorValue2GeTensor(ge::hybrid::TensorValue &&tensor_value, const ConstGeTensorDescPtr &ge_tensor_desc,
                            ge::GeTensor &ge_tensor) {
  ge_tensor.MutableTensorDesc() = *ge_tensor_desc;
  ge_tensor.MutableTensorDesc().SetPlacement(kPlacementDevice);
  // 将tensor_value移动到deleter中保存，避免外部tensor_value析构时释放内存
  auto deleter = [tensor = std::move(tensor_value)](uint8_t *const device_data) mutable {
    (void)device_data;
    tensor.Destroy();
  };
  GE_CHK_STATUS_RET(
      ge_tensor.SetData(PtrToPtr<void, uint8_t>(const_cast<void *>(tensor_value.GetData())),
        tensor_value.GetSize(), deleter));
  return SUCCESS;
}
}
namespace hybrid {
HybridModelRtV1Executor::HybridModelRtV1Executor(HybridModel *const model, const uint32_t device_id,
                                                 const rtStream_t stream, ThreadPool *const thread_pool)
    : HybridModelExecutor(model, device_id, stream),
      executor_(model_->GetRootGraphItem(), &context_, false, thread_pool),
      data_dumper_(nullptr) {}

Status HybridModelRtV1Executor::Init(CallbackManager *const callback_manager) {
  GELOGD("Start to init HybridGraphEngine.");
  model_id_ = model_->GetModelId();
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext(callback_manager));
  GE_CHK_STATUS_RET_NOLOG(executor_.Init());
  GE_CHK_STATUS_RET(DumpOpDebug(), "[Dump][OpDebug] failed, model_id:%u.", model_->GetModelId());
  GE_CHK_STATUS_RET(InitInputDesc(), "[Init][InputDesc] failed, model_id:%u.", model_->GetModelId());
  GELOGD("HybridGraphEngine initialized successfully.");
  return SUCCESS;
}

Status HybridModelRtV1Executor::DumpOpDebug() {
  const DumpProperties &dump_properties = context_.dump_properties;
  if (dump_properties.IsOpDebugOpen()) {
    GELOGD("Opdebug is open in hybrid engine");
    const uint32_t op_debug_mode = dump_properties.GetOpDebugMode();
    GE_CHK_RT_RET(static_cast<rtError_t>(
        op_debug_register_.RegisterDebugForStream(stream_, op_debug_mode, data_dumper_)));
    is_op_debug_reg_ = true;
    data_dumper_.SetDumpProperties(dump_properties);
    data_dumper_.SetModelName(model_->GetModelName());
    data_dumper_.SetModelId(model_->GetModelId());
    data_dumper_.SetDeviceId(model_->GetDeviceId());

    uintptr_t global_step = 0U;
    if (dump_properties.IsInferOpDebug()) {
      GELOGD("Init global step when infer with op debug.");
      global_step = PtrToValue(context_.global_step);
    } else {
      const TensorValue *const variable_global_step = model_->GetVariable(NODE_NAME_GLOBAL_STEP);
      if (variable_global_step != nullptr) {
        global_step = PtrToValue(variable_global_step->GetData());
      }
    }
    const TensorValue *const variable_loop_iter = model_->GetVariable(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
    const uintptr_t loop_iter = (variable_loop_iter != nullptr) ? PtrToValue(variable_loop_iter->GetData()) : 0U;

    const TensorValue *const variable_loop_cond = model_->GetVariable(NODE_NAME_FLOWCTRL_LOOP_COND);
    const uintptr_t loop_cond = (variable_loop_cond != nullptr) ? PtrToValue(variable_loop_cond->GetData()) : 0U;

    data_dumper_.SetLoopAddr(global_step, loop_iter, loop_cond);
    GE_CHK_STATUS_RET(data_dumper_.LoadDumpInfo(),
                      "[Invoke][LoadDumpInfo] failed in hybrid engine, model_id = %u.", model_id_);
    GELOGD("Dump op debug SUCCESS in hybrid engine");
  }
  return SUCCESS;
}

void HybridModelRtV1Executor::Stop() {
  if (is_op_debug_reg_) {
    op_debug_register_.UnregisterDebugForStream(stream_);
  }
}

Status HybridModelRtV1Executor::ExecuteForSingleOp(const HybridModelExecutor::ExecuteArgs &args) {
  const auto ret = executor_.ExecuteAsync(args.inputs, args.input_desc, args.outputs);

  if (context_.has_observer) {
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kInitInferShapeContext);
    context_.runtime_context_.Release();
  }
  HYBRID_CHK_STATUS_RET(ret, "[Run][Execute] Single op model execute Failed.");

  // When dump on, must wait for the callback function to finish executing.
  if (context_.IsDumpEnabled()) {
    GE_CHK_STATUS_RET(context_.callback_manager->Destroy(), "[Destroy][Callback] for failed.");
    GE_CHK_STATUS_RET(context_.callback_manager->Init(), "[Init][Callback] for failed.");
  }
  executor_.Reset();
  return SUCCESS;
}

Status HybridModelRtV1Executor::PreRun(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args) {
  GE_CHK_STATUS_RET(SyncVarData(), "[Invoke][SyncVarData] failed, model_id:%u.", model_id_);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[SyncVarData] End");
  GE_CHK_STATUS_RET(PrepareExecuteArgs(current_data, args),
                    "[Invoke][PrepareExecuteArgs] failed to copy input data to model, model_id:%u.", model_id_);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[CopyInputData] End");
  return SUCCESS;
}

Status HybridModelRtV1Executor::ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
    std::shared_ptr<ModelListener> listener) {
  RT2_PROFILING_SCOPE_CONST(gert::profiling::kUnknownName, gert::profiling::kModelExecute);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[RunInternal] [iteration = %d] Start", iterator_count_);
  HybridModelExecutor::ExecuteArgs args;
  InputData input_data;
  OutputData output_data;
  GenDataInputOutputData(model_id_, inputs, input_data, output_data);
  auto ret = ProcessOnlineModel(input_data, args);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[RunInternal] [iteration = %d] End", iterator_count_);
  iterator_count_++;
  GELOGI("run iterator count is %lu, model_id:%u", iterator_count_, model_->GetModelId());
  ret = HandleResult(ret, input_data.index, args, &output_data, listener);
  return ret;
}

Status HybridModelRtV1Executor::ProcessOnlineModel(const InputData &input_data,
                                                   HybridModelExecutor::ExecuteArgs &args) {
  const auto ret = PreRun(input_data, args);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Invoke][PreRun] failed, model_id:%u.", model_id_);  // [No need to check value]
    return ret;
  }

  GELOGI("HybridModel will execute in rt1.0 singleline mode");
  ge::GetContext().SetSessionId(context_.session_id);
  ge::GetContext().SetContextId(context_.context_id);
  return Execute(args);
}

Status HybridModelRtV1Executor::Execute(const InputData &input_data, ExecuteArgs &args) {
  GE_CHK_STATUS_RET(PrepareExecuteArgs(input_data, args),
      "[Invoke][PrepareExecuteArgs]Failed to copy input data to model, model_id = %u", model_id_);
  GELOGD("Done copying input data successfully.");
  return Execute(args);
}

Status HybridModelRtV1Executor::Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) {
  ExecuteArgs args;
  args.ctrl_args = ctrl_args;
  InputData input_data = GetInputDataFromGertTensors(inputs);

  GE_ASSERT_SUCCESS(Execute(input_data, args));
  GE_ASSERT_EQ(args.output_desc.size(), args.outputs.size());
  ctrl_args = args.ctrl_args;
  outputs.clear();
  outputs.resize(args.outputs.size());
  size_t out_index = 0U;
  for (auto &gert_tensor : outputs) {
    const auto &tensor_desc = args.output_desc[out_index];
    GeTensor ge_tensor;
    GE_ASSERT_SUCCESS(TensorValue2GeTensor(std::move(args.outputs[out_index]), tensor_desc, ge_tensor));
    GE_ASSERT_SUCCESS(TensorTransUtils::GeTensor2GertTensor(ge_tensor, outputs[out_index]));
    GELOGD("Set output[%d], tensor size = %zu, shape = [%s]", out_index,
           gert_tensor.GetSize(), formats::GertShapeToString(gert_tensor.GetStorageShape()).c_str());
    ++out_index;
  }
  return SUCCESS;
}

Status HybridModelRtV1Executor::Execute(ExecuteArgs &args) {
  GELOGD("Start to execute model.");
  const auto root_graph_item = model_->GetRootGraphItem();
  GE_CHECK_NOTNULL(root_graph_item);

  // one-node-multiple-bin mode does not need to check shape range which will be modified in fuzz compile
  if (root_graph_item->IsDynamic() && (model_->GetNodeBinMode() == fuzz_compile::kOneNodeSingleBinMode)) {
    GE_CHK_STATUS_RET(CheckInputShapeByShapeRange(root_graph_item, args),
                      "[Check][InputShape] By ShapeRange for [%s] failed.", root_graph_item->GetName().c_str());
  }

  // In heterogeneous executor, gloabl_step is updated by markStep task
  if (!ExecutionRuntimeUtils::IsInHeterogeneousExecutor()) {
    if (context_.global_step != nullptr) {
      GE_CHK_RT_RET(rtMemcpyAsync(context_.global_step, sizeof(uint64_t), &context_.iteration,
                                  sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE_EX, context_.stream));
    }
  }

  const auto ret = ExecuteGraphInternal(executor_, args);
  (void)Cleanup();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");
  GELOGD("Model executed successfully.");
  if (context_.profiler != nullptr) {
    context_.profiler->Dump(std::cout);
    context_.profiler->Reset();
  }

  context_.iteration += 1;
  executor_.Reset();
  if (ret == END_OF_SEQUENCE) {
    args.ctrl_args.is_eos = true;
  } else {
    if (ret != ge::SUCCESS) {
      GELOGE(ret, "[Invoke][ExecuteGraphInternal] Failed, ret:%u.", ret);
      return ret;
    }
  }
  return SUCCESS;
}

Status HybridModelRtV1Executor::ExecuteGraphInternal(SubgraphExecutor &executor,
                                                     HybridModelExecutor::ExecuteArgs &args) {
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] Start");
  GE_CHK_STATUS_RET_NOLOG(ResetExecutionContext(context_));
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[InitContext] End");
  GE_CHECK_NOTNULL(model_->GetRootGraph());

  HYBRID_CHK_STATUS_RET(executor.ExecuteAsync(args.inputs, args.input_desc, args.outputs),
                        "[Call][ExecuteAsync] Failed to execute partitioned call.");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[ExecuteAsync] End");

  const Status ret = executor.Synchronize();
  if (ret != ge::SUCCESS) {
    if (ret == ge::END_OF_SEQUENCE) {
      GELOGD("Got end of sequence");
    } else {
      GELOGE(ret, "[Execute][GraphInternal] Synchronize failed.");
    }
    return ret;
  }
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End");

  args.outputs.clear();
  HYBRID_CHK_STATUS_RET(executor.GetOutputs(args.outputs, args.output_desc), "[Get][Outputs] failed");
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[GetOutput] End");
  return SUCCESS;
}

Status HybridModelRtV1Executor::Cleanup() {
  GELOGD("Start to cleanup.");
  (void)context_.callback_manager->Destroy();
  context_.runtime_context_.Release();
  GELOGD("Cleanup successfully.");
  return SUCCESS;
}

Status HybridModelRtV1Executor::InitExecutionContext(CallbackManager *const callback_manager) {
  GE_CHK_RT_RET(rtCtxGetCurrent(&context_.rt_context));
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));

  context_.is_host_cpu = ::ge::GetContext().GetHostExecFlag();
  context_.global_step = model_->GetGlobalStep();
  context_.stream = stream_;
  context_.model = model_;
  context_.is_eos_ = false;
  context_.session_id = ::ge::GetContext().SessionId();
  context_.ge_context = &GetThreadLocalContext();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(device_id_, stream_);
  GE_CHECK_NOTNULL(context_.allocator);
  if (callback_manager != nullptr) {
    context_.callback_manager = callback_manager;
  } else {
    if (context_.is_host_cpu) {
      context_.callback_manager = new (std::nothrow) HostCpuCallbackManager();
    } else {
      context_.callback_manager = new (std::nothrow) RtCallbackManager();
    }
    GE_CHECK_NOTNULL(context_.callback_manager);
    context_.own_callback_manager = true;
  }

  context_.dump_properties = DumpManager::GetInstance().GetDumpProperties(context_.session_id);

  GE_CHK_STATUS_RET_NOLOG(context_.InitProfiler());
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }
  context_.has_observer = model_->HasObserver();
  GE_CHK_STATUS_RET_NOLOG(context_.res_manager.Init(model_->GetRootGraphItem()));

  return SUCCESS;
}

Status HybridModelRtV1Executor::ResetExecutionContext(GraphExecutionContext &context) {
  GE_CHK_STATUS_RET_NOLOG(context.callback_manager->Init());

  context.runtime_context_.Release();
  for (auto &host_tensor : context.model->GetHostTensors()) {
    const auto node_id = host_tensor.first;
    for (const auto &output_idx_and_tensor : host_tensor.second) {
      const auto output_idx = output_idx_and_tensor.first;
      GELOGD("Preload const host tensor, node_id = %ld, output id = %d", node_id, output_idx);
      (void)context.runtime_context_.SetTensor(node_id, output_idx, output_idx_and_tensor.second);
    }
  }
  context.res_manager.ClearDataFlowResources();
  return SUCCESS;
}

Status HybridModelRtV1Executor::CheckInputShapeByShapeRange(const GraphItem *const graph_item,
                                                            const HybridModelExecutor::ExecuteArgs &args) {
  GE_CHECK_NOTNULL(graph_item);
  const auto &input_nodes = graph_item->GetInputNodes();
  for (size_t i = 0U; i < input_nodes.size(); ++i) {
    const auto &input_node = input_nodes[i];
    if (input_node == nullptr) {
      GELOGD("[%s] Input[%zu] is not needed by graph, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }
    if (!input_node->is_dynamic) {
      GELOGD("[%s] Input[%zu] is not dynamic, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }
    const GeTensorDescPtr &model_input_desc = input_node->MutableInputDesc(0);
    GE_CHECK_NOTNULL(model_input_desc);
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    if (model_input_desc->GetShapeRange(shape_range) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "[%s] Input[%zu] get shape range failed", graph_item->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[Get][ShapeRange] [%s] Input[%zu] get shape range failed",
             graph_item->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }
    if (shape_range.empty()) {
      GELOGD("[%s] Input[%zu] shape is not needed to check by shape range, skip it.", graph_item->GetName().c_str(), i);
      continue;
    }
    if (i >= args.input_desc.size()) {
      REPORT_INNER_ERR_MSG("E19999", "[%s] Inputs[%zu] is greater than or equal to input desc size[%zu].",
                         graph_item->GetName().c_str(), i, args.input_desc.size());
      GELOGE(INTERNAL_ERROR, "[Check][Param] [%s] inputs[%zu] is greater than or equal to input desc size[%zu].",
             graph_item->GetName().c_str(), i, args.input_desc.size());
      return INTERNAL_ERROR;
    }
    const ConstGeTensorDescPtr &args_tensor_desc = args.input_desc[i];
    GE_CHECK_NOTNULL(args_tensor_desc);
    const GeShape &shape = args_tensor_desc->GetShape();
    if (shape.IsUnknownShape()) {
      REPORT_INNER_ERR_MSG("E19999", "[%s] Input desc shape [%zu] designed by user must be static.",
                         graph_item->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[Check][Param] [%s] Input desc shape [%zu] designed by user must be static.",
             graph_item->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }

    if (TensorUtils::CheckShapeByShapeRange(shape, shape_range) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Check][InputShape] [%s] check input [%zu] shape failed by shape range.",
             graph_item->GetName().c_str(), i);
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

bool HybridModelRtV1Executor::NeedBuildDeviceTensorAsOutput() const {
  std::string execute_mode;
  ParserContextOption(OPTION_EXEC_DYNAMIC_EXECUTE_MODE, execute_mode);
  std::string is_copy_output_addr;
  ParserContextOption(OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR, is_copy_output_addr);
  return ((execute_mode == kLazyRecompile) && (is_copy_output_addr == kIsCopyOuputAddr));
}
}  // namespace hybrid
}  // namespace ge
