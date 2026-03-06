/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/hybrid_model_async_executor.h"
#include "base/err_mgr.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "graph/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/caching_allocator.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/rdma_pool_allocator.h"
#include "graph/manager/host_mem_allocator.h"
#include "graph/manager/mem_manager.h"
#include "common/profiling_definitions.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace hybrid {
namespace {
const size_t kMinimumPiplineStages = 2U;
const uint64_t kStopOnFailure = 1U;

Status CheckBlockingOp(const ComputeGraphPtr &graph, bool &has_blocking_op) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool is_blocking_op = false;
    (void)AttrUtils::GetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, is_blocking_op);
    if (is_blocking_op) {
      has_blocking_op = true;
      return SUCCESS;
    }
  }
  has_blocking_op = false;
  return SUCCESS;
}

ge::graphStatus GetGraphMaxParallelModeNum(int32_t &max_parallel_num) {
  std::string opt = "0";
  (void)GetContext().GetOption(GRAPH_MAX_PARALLEL_MODEL_NUM, opt);
  GE_ASSERT_SUCCESS(ge::ConvertToInt32(opt, max_parallel_num), "option %s, value %s is not int",
                    GetContext().GetReadableName(GRAPH_MAX_PARALLEL_MODEL_NUM).c_str(), opt.c_str());
  return ge::GRAPH_SUCCESS;
}
}  // namespace
std::map<std::pair<uint32_t, uint32_t>, HybridModelAsyncExecutor::DefaultStreamGuarder>
    HybridModelAsyncExecutor::default_stream_by_dev_;
std::mutex HybridModelAsyncExecutor::mu_for_guarder_;
HybridModelAsyncExecutor::HybridModelAsyncExecutor(HybridModel *const model)
    : model_(model), run_flag_(false) {
}

HybridModelAsyncExecutor::~HybridModelAsyncExecutor() {
  auto &default_stream_guarder = GetDefaultStreamGuarder();
  const std::lock_guard<std::mutex> lk(default_stream_guarder.mu);
  if (stream_ != nullptr) {
    if (owner_stream_) {
      NpuMemoryAllocator::ClearStream(stream_);
      GE_CHK_RT(rtStreamDestroy(stream_));
    } else if (default_stream_guarder.default_stream != nullptr) {
      default_stream_guarder.stream_ref_count--;
      if (default_stream_guarder.stream_ref_count == 0U) {
        NpuMemoryAllocator::ClearStream(default_stream_guarder.default_stream);
        GE_CHK_RT(rtStreamDestroy(default_stream_guarder.default_stream));
        default_stream_guarder.default_stream = nullptr;
      }
    } else {
      // nothing to do
    }
    stream_ = nullptr;
  } else {
    // nothing to do
  }
}

void HybridModelAsyncExecutor::SetDeviceId(const uint32_t device_id) {
  device_id_ = device_id;
}

void HybridModelAsyncExecutor::SetModelId(const uint32_t model_id) {
  model_id_ = model_id;
}

Status HybridModelAsyncExecutor::EnqueueData(const shared_ptr<RunArgs> &args) {
  if (data_inputer_->Push(args) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Data queue is full, please call again later, model_id %u.", model_id_);
    GELOGE(domi::DATA_QUEUE_ISFULL,
        "[Push][Data] Data queue is full, please call again later, model_id %u ", model_id_);
    return domi::DATA_QUEUE_ISFULL;
  }
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Start(const std::shared_ptr<ModelListener> &listener) {
  GELOGD("HybridModelExecutor::Start IN, has listener = %d", static_cast<int32_t>(listener != nullptr));
  const std::lock_guard<std::mutex> lk(GetDefaultStreamGuarder().mu);
  if (run_flag_) {
    REPORT_INNER_ERR_MSG("E19999", "Model already started, model_id:%u.", model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][RunState] Model already started, model_id:%u.", model_id_);
    return INTERNAL_ERROR;
  }
  run_flag_ = true;
  listener_ = listener;

  future_ = std::async(std::launch::async, [this, context_copy = *executor_->GetContext()->ge_context]
    (const struct error_message::ErrorManagerContext &error_context) -> Status {
    error_message::SetErrMgrContext(error_context);
    // rt1 return non-nullptr, rt2 and pipeline will return nullptr
    if (executor_->GetContext() != nullptr) {
      // context_copy使用值捕获，为了避免主线程中销毁executor_->GetContext()->ge_context后导致访问野指针
      GetThreadLocalContext() = context_copy;
      GetContext().SetSessionId(executor_->GetContext()->session_id);
      GetContext().SetContextId(executor_->GetContext()->context_id);
    }
    return RunInternal();
  }, error_message::GetErrMgrContext());

  GE_CHK_BOOL_RET_STATUS(future_.valid(), INTERNAL_ERROR,
                         "[Check][RunState] Failed to start, model_id:%u.", model_id_);
  GELOGD("HybridModelExecutor::Start successfully.");
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Stop() {
  auto &default_stream_guarder = GetDefaultStreamGuarder();
  const std::lock_guard<std::mutex> lk(default_stream_guarder.mu);
  run_flag_ = false;
  data_inputer_->Stop();

  Status ret = SUCCESS;
  if (future_.valid()) {
    ret = future_.get();
  }
  executor_->Stop();

  if (stream_ != nullptr) {
    if (owner_stream_) {
      NpuMemoryAllocator::ClearStream(stream_);
      GE_CHK_RT(rtStreamSynchronize(stream_));
      GE_CHK_RT(rtStreamDestroyForce(stream_));
    } else if (default_stream_guarder.default_stream != nullptr) {
      default_stream_guarder.stream_ref_count--;
      if (default_stream_guarder.stream_ref_count == 0U) {
        NpuMemoryAllocator::ClearStream(default_stream_guarder.default_stream);
        GE_CHK_RT(rtStreamSynchronize(default_stream_guarder.default_stream));
        GE_CHK_RT(rtStreamDestroyForce(default_stream_guarder.default_stream));
        default_stream_guarder.default_stream = nullptr;
      }
    } else {
      // nothing to do
    }
    stream_ = nullptr;
  } else {
    // nothing to do
  }

  return ret;
}

Status HybridModelAsyncExecutor::BuildExecutor() {
  if (model_->IsExecuteByRtV2()) {
    executor_ = MakeUnique<HybridModelRtV2Executor>(model_, device_id_, stream_);
  } else if (model_->GetRootGraphItem()->NumGroups() >= kMinimumPiplineStages) {
    GELOGI("HybridModel stage nums:%zu", model_->GetRootGraphItem()->NumGroups());
    executor_ = MakeUnique<HybridModelPipelineExecutor>(model_, device_id_, stream_);
  } else {
    executor_ = MakeUnique<HybridModelRtV1Executor>(model_, device_id_, stream_);
  }
  GE_ASSERT_NOTNULL(executor_);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Init(const rtStream_t stream) {
  data_inputer_ = MakeUnique<DataInputer>();
  GE_CHECK_NOTNULL(data_inputer_);
  // 如果用户传入了stream，将stream设置为用户传入的, 更新own_stream_为true。
  // 如果用户没有传入stream，则内部创建一条stream，如果有阻塞型算子时，两个网络在同一个stream上就卡死了。
  // （外置stream时属于用户构造的错误场景，不会存在问题）
  if (stream != nullptr) {
    GELOGD("load with external stream = %p", stream);
    stream_ = stream;
    owner_stream_ = false;
  } else {
    bool has_blocking_op = false;
    GE_CHECK_NOTNULL(model_);
    GE_CHK_STATUS_RET(CheckBlockingOp(model_->root_graph_, has_blocking_op));
    // 如果用户设置多实例并行，则这里需要为每个实例单独创建stream
    int32_t max_parallel_num = 0;
    GE_ASSERT_SUCCESS(GetGraphMaxParallelModeNum(max_parallel_num));
    const bool use_new_stream_for_parallel = (max_parallel_num > 1);
    const bool use_second_stream =
        ((!domi::GetContext().is_online_model) || has_blocking_op || use_new_stream_for_parallel);
    const uint32_t stream_flags =
        (GetContext().IsOverflowDetectionOpen() ? RT_STREAM_OVERFLOW : RT_STREAM_DEFAULT) | RT_STREAM_FAST_LAUNCH |
        RT_STREAM_FAST_SYNC;
    if (!use_second_stream) {
      auto &default_stream_guarder = GetDefaultStreamGuarder();
      const std::lock_guard<std::mutex> lk(default_stream_guarder.mu);
      if (default_stream_guarder.default_stream == nullptr) {
        GE_CHK_RT_RET(rtStreamCreateWithFlags(&default_stream_guarder.default_stream,
                                              static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), stream_flags));
        GE_CHK_RT_RET(rtStreamSetMode(default_stream_guarder.default_stream, kStopOnFailure));
        GELOGD("Create default stream=%p, device id = %u", default_stream_guarder.default_stream, device_id_);
      }
      default_stream_guarder.stream_ref_count++;
      stream_ = default_stream_guarder.default_stream;
    } else {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream_, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT), stream_flags));
      GE_CHK_RT_RET(rtStreamSetMode(stream_, kStopOnFailure));
      GELOGD("Create stream=%p, device id = %u", stream_, device_id_);
      owner_stream_ = true;
    }
  }
  GE_ASSERT_EQ(BuildExecutor(), SUCCESS);
  GE_CHK_STATUS_RET(executor_->Init(), "[Init][HybridModelExecutor] failed, model_id:%u.", model_id_);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::RunInternal() {
  const auto device_id = static_cast<int32_t>(device_id_);
  GELOGD("Hybrid model start. model_id = %u, device_id = %u", model_id_, device_id_);
  GE_CHK_RT_RET(rtSetDevice(device_id));
  // DeviceReset before thread run finished!
  GE_MAKE_GUARD(not_used_var, [&device_id] { GE_CHK_RT(rtDeviceReset(device_id)); });

  while (run_flag_) {
    // Model has not indeedly started running before received data
    SetRunningFlag(false);
    std::shared_ptr<RunArgs> args = nullptr;
    Status ret = data_inputer_->Pop(args);
    // Model indeedly start running
    GE_IF_BOOL_EXEC((args == nullptr) || (ret != SUCCESS),
      GELOGI("data_wrapper is null!, ret = %u", ret); continue);

    GELOGI("Getting the input data, model_id:[%u]", model_id_);
    GE_IF_BOOL_EXEC(!run_flag_, break);
    SetRunningFlag(true);
    ScopeGuard running_flag_guarder([this]() { running_flag_ = false; });
    GELOGI("Model thread Run begin, model id:[%u].", model_id_);

    ret = executor_->ExecuteOnlineModel(args->input_tensor, listener_);
    if (ret != SUCCESS) {
      GELOGI("Executor execute model:[%u] is not success.", model_id_);
      continue;
    }
  }
  GELOGI("Model run end, model id:[%u]", model_id_);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Execute(const std::vector<DataBuffer> &inputs,
                                         const std::vector<GeTensorDesc> &input_desc,
                                         std::vector<DataBuffer> &outputs,
                                         std::vector<GeTensorDesc> &output_desc,
                                         rtStream_t stream) {
  GELOGI("Start to execute model.");
  output_cache_.clear();
  HybridModelExecutor::ExecuteArgs args;
  args.ctrl_args.stream = stream;
  args.inputs.resize(inputs.size());
  args.outputs.resize(outputs.size());
  for (size_t i = 0U; i < inputs.size(); ++i) {
    MemStorageType mem_type = inputs[i].placement == static_cast<uint32_t>(Placement::kPlacementHost)
                                  ? MemStorageType::HOST_DDR
                                  : MemStorageType::HBM;
    const TensorValue tensor_value(inputs[i].data, inputs[i].length, mem_type);
    args.inputs[i] = tensor_value;
  }
  std::vector<size_t> allocate_by_executor;
  for (size_t i = 0U; i < outputs.size(); ++i) {
    if (outputs[i].data == nullptr) {
      allocate_by_executor.emplace_back(i);
    } else {
      const auto mem_type = (outputs[i].placement == kPlacementDevice) ?
                            MemStorageType::HBM : MemStorageType::HOST_DDR;
      args.outputs[i] = TensorValue(outputs[i].data, outputs[i].length, mem_type);
    }
  }
  // usr must designate input tensorDesc when input shape is dynamic in inference
  for (size_t i = 0U; i < input_desc.size(); ++i) {
    ConstGeTensorDescPtr tensor_desc_ptr = MakeShared<GeTensorDesc>(input_desc[i]);
    GE_CHECK_NOTNULL(tensor_desc_ptr);
    args.input_desc.emplace_back(tensor_desc_ptr);
  }

  GE_CHK_STATUS_RET(executor_->Execute(args), "[Invoke][Execute] Failed, model_id = %u.", model_id_);
  for (const size_t output_index : allocate_by_executor) {
    output_cache_.emplace_back(args.outputs[output_index]); // hold till next iteration
    outputs[output_index].length = args.outputs[output_index].GetSize();
    outputs[output_index].data = args.outputs[output_index].MutableData();
    outputs[output_index].placement = kPlacementDevice;
  }
  for (const auto &output_tensor_desc : args.output_desc) {
    output_desc.emplace_back(*output_tensor_desc);
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs,
                                                        std::vector<GeTensor> &outputs,
                                                        rtStream_t stream) {
    return executor_->ExecuteWithStreamAsync(inputs, outputs, stream);
}

Status HybridModelAsyncExecutor::ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                                        std::vector<gert::Tensor> &outputs,
                                                        rtStream_t stream) {
    return executor_->ExecuteWithStreamAsync(inputs, outputs, stream);
}

/*
 * outputs是要返回给用户的数据，为GE申请位于host上的内存。
 * executor_->Execute出参executor_outputs是位于device上的内存，需要调用CopyOutputs接口拷贝到host上，保存在outputs中，
 * 返回给用户，由用户释放。
 */
Status HybridModelAsyncExecutor::Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
  GELOGD("Start to execute model.");
  GE_CHECK_NOTNULL(executor_);
  std::vector<gert::Tensor> executor_outputs;
  HybridModelExecutor::CtrlArgs ctrl_args;
  GE_CHK_STATUS_RET(executor_->Execute(inputs, executor_outputs, ctrl_args),
    "[Invoke][Execute] Failed, model_id = %u.", model_id_);

  GE_CHK_STATUS_RET(executor_->CopyOutputs(executor_outputs, outputs),
      "[Invoke][CopyOutputs]Failed to copy outputs, model_id = %u.", model_id_);
  GELOGD("Done copying output data successfully. output count = %zu", outputs.size());
  return SUCCESS;
}

HybridModelAsyncExecutor::DefaultStreamGuarder &HybridModelAsyncExecutor::GetDefaultStreamGuarder() const {
  const std::lock_guard<std::mutex> lk(mu_for_guarder_);
  return default_stream_by_dev_[std::make_pair(device_id_, model_id_)];
}
}  // namespace hybrid
}  // namespace ge
