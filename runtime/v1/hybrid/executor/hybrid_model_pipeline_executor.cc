/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/hybrid_model_pipeline_executor.h"

#include "base/err_mgr.h"
#include "common/math/math_util.h"
#include "common/dump/dump_manager.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "common/profiling_definitions.h"
#include "common/profiling/profiling_manager.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int32_t kNumExecutors = 2;
const int32_t kMinLoopCount = 2;
constexpr int32_t kWaitTimeoutInSec = 600;
const int32_t kDefaultLoopCount = 10;
}

StageExecutor::StageExecutor(const int32_t id, HybridModel *const model, PipeExecutionConfig *const config,
                             StageSubject *const stage_subject)
    : id_(id), model_(model), pipe_config_(config), stage_subject_(stage_subject) {}

StageExecutor::~StageExecutor() {
  GELOGD("~StageExecutor(), id = %d", id_);
  if (stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(stream_));
    stream_ = nullptr;
  }
  if (hccl_stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(hccl_stream_));
    hccl_stream_ = nullptr;
  }
}

Status StageExecutor::Init() {
  GELOGD("[Executor: %d] Start to init StateExecutor", id_);
  context_.rt_context = pipe_config_->rt_context;
  GE_CHK_STATUS_RET_NOLOG(InitExecutionContext());
  GE_CHK_RT_RET(rtStreamCreate(&stream_, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)));
  GE_CHK_RT_RET(rtStreamCreate(&hccl_stream_, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)));
  context_.stream = stream_;
  context_.hccl_stream = hccl_stream_;

  root_graph_executor_ = MakeUnique<SubgraphExecutor>(model_->GetRootGraphItem(), &context_);
  GE_CHECK_NOTNULL(root_graph_executor_);
  GE_CHK_STATUS_RET(root_graph_executor_->Init(), "[Init][RootGraphExecutor]Failed.");
  GELOGD("[Executor: %d] Init stage executor successfully", id_);
  return SUCCESS;
}

Status StageExecutor::ResetExecutionContext(GraphExecutionContext &context) {
  GE_CHK_STATUS_RET_NOLOG(context.callback_manager->Init());
  const std::string ctx_id = std::to_string(context.context_id);
  context.runtime_context_.Release();
  for (auto &host_tensor : context.model->GetHostTensors()) {
    const auto node_id = host_tensor.first;
    for (const auto &output_idx_and_tensor : host_tensor.second) {
      const auto output_idx = output_idx_and_tensor.first;
      GELOGD("Preload const host tensor, node_id = %ld, output id = %d", node_id, output_idx);
      (void)context.runtime_context_.SetTensor(node_id, output_idx, output_idx_and_tensor.second);
    }
  }
  return SUCCESS;
}

void StageExecutor::ReleaseCallback() {
  (void)context_.callback_manager->Destroy();
  context_.runtime_context_.Release();
}

void StageExecutor::ExecuteEndTaskAndReleae() {
  StageTask end_task;
  end_task.is_eos = true;
  GELOGD("[Executor: %d] send end task", id_);
  (void)next_executor_->ExecuteAsync(end_task);
  ReleaseCallback();
}

Status StageExecutor::Start(const std::vector<TensorValue> &inputs, const std::vector<ConstGeTensorDescPtr> &input_desc,
                            const int32_t iteration_count) {
  GELOGD("[Executor: %d] thread start", id_);
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));
  int32_t num_loops = iteration_count / pipe_config_->num_executors;
  if (id_ < (iteration_count % iteration_count)) {
    num_loops += 1;
  }
  FMK_INT32_MULCHECK(num_loops, pipe_config_->num_stages);
  num_loops *= pipe_config_->num_stages;
  GELOGD("[Executor: %d] loop count = %d", id_, num_loops);

  std::function<void()> release_callback = [this]() {
    ExecuteEndTaskAndReleae();
  };
  const auto release_guard = MakeShared<ScopeGuard>(release_callback);
  GE_CHECK_NOTNULL(release_guard);

  for (int32_t loop_idx = 0; loop_idx < num_loops; ++loop_idx) {
    GELOGD("[Executor: %d] Start to wait for task.", id_);
    StageTask task_info;
    (void)task_queue_.Pop(task_info);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(task_info.is_eos, END_OF_SEQUENCE, "[Executor: %d] receive end task", id_);
    GELOGD("[Executor: %d] Got task, stage = %d, iteration = %ld", id_, task_info.stage, task_info.iteration);
    if (task_info.iteration >= pipe_config_->iteration_end) {
      GELOGE(INTERNAL_ERROR, "[Check][Range][Executor: %d] Unexpected iteration: %ld.", id_, task_info.iteration);
      REPORT_INNER_ERR_MSG("E19999", "[Executor: %d] Unexpected iteration: %" PRId64 ".", id_, task_info.iteration);
      return INTERNAL_ERROR;
    }

    if (task_info.event != nullptr) {
      GELOGD("[%d] Add StreamWaitEvent", id_);
      GE_CHK_RT_RET(rtStreamWaitEvent(stream_, task_info.event));
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] EventWait End", task_info.iteration,
                                   task_info.stage);
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] Start", task_info.iteration,
                                 task_info.stage);

    if (task_info.stage == 0) {
      GELOGD("[Executor: %d] To ResetExecutionContext", id_);
      GE_CHK_STATUS_RET(ResetExecutionContext(context_),
                        "[Invoke][ResetExecutionContext][Executor: %d] Failed to reset context", id_);
      context_.iteration = task_info.iteration;
      GE_CHK_STATUS_RET_NOLOG(SetInputs(inputs, input_desc));
    }
    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Stage = %d] PartialExecuteAsync Start", task_info.stage);
    const auto ret = root_graph_executor_->PartialExecuteAsync(task_info.stage);
    RECORD_MODEL_EXECUTION_EVENT(&context_, "[Stage = %d] PartialExecuteAsync End", task_info.stage);
    GELOGD("[Executor: %d] PartialExecuteAsync end, ret is %u", id_, ret);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(ret == END_OF_SEQUENCE, END_OF_SEQUENCE,
        "[Executor: %d] PartialExecuteAsync receive eos", id_);
    GE_CHK_STATUS_RET(ret);
    // notify next execution unit
    StageTask next_task;
    next_task.stage = task_info.stage;
    next_task.iteration = task_info.iteration + 1;
    GE_MAKE_GUARD(next_task_guard, [&next_task]() {
      if (next_task.event != nullptr) {
        (void)rtEventDestroy(next_task.event);
        next_task.event = nullptr;
      }
    });
    if (((task_info.iteration + 1) % iteration_count) > 0) {
      GE_CHK_RT_RET(rtEventCreate(&next_task.event));
      GE_CHK_RT_RET(rtEventRecord(next_task.event, context_.hccl_stream));
    }

    const auto sync_result = Synchronize();
    if (sync_result != SUCCESS) {
      GELOGE(sync_result,
             "[Invoke][Synchronize][Executor: %d] Failed to sync result:%u. iteration = %ld",
             id_, sync_result, task_info.iteration);
      REPORT_INNER_ERR_MSG("E19999", "[Executor: %d] Failed to sync result:%u. iteration = %" PRId64 "",
                        id_, sync_result, task_info.iteration);
      if (context_.profiler != nullptr) {
        context_.profiler->Dump(std::cout);
      }
      return sync_result;
    }
    stage_subject_->Release(task_info.stage);
    if (task_info.event != nullptr) {
      GE_CHK_RT_RET(rtEventDestroy(task_info.event));
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] EventDestroy End", task_info.iteration,
                                   task_info.stage);
    }

    RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] [Stage = %d] End", task_info.iteration, task_info.stage);

    // if end stage
    if (task_info.stage >= (pipe_config_->num_stages - 1)) {
      RECORD_MODEL_EXECUTION_EVENT(&context_, "[iteration = %ld] Schedule End",
                                   task_info.iteration);
      GELOGD("[Executor: %d] End of iteration [%ld]", id_, task_info.iteration);
      ReleaseCallback();
    } else {
      if (model_->GetRootGraphItem()->IsDynamic()) {
        GE_CHK_STATUS_RET_NOLOG(stage_subject_->Await(task_info.stage + 1));
        auto& stage_cache = model_->GetRootGraphItem()->GetStageCache();
        GE_CHK_STATUS_RET_NOLOG(stage_cache.DoPropagate(task_info.stage + 1));
      }
    }
    GE_CHK_STATUS_RET_NOLOG(next_executor_->ExecuteAsync(next_task));
    GELOGD("[Executor: %d] Push item successfully.", id_);
  }

  GELOGD("[Executor: %d] Process task ended.", id_);
  return SUCCESS;
}

Status StageExecutor::ExecuteAsync(const StageTask &args) {
  (void)task_queue_.Push(args);
  return SUCCESS;
}

Status StageExecutor::Synchronize() {
  const auto ret = root_graph_executor_->Synchronize();
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Synchronize] End, ret = %u", ret);
  return ret;
}

StageSubject::Cond &StageSubject::GetSubject(const int32_t stage) {
  const std::lock_guard<std::mutex> lk(mu_);
  return subjects_[stage];
}

Status StageSubject::Await(const int32_t stage) {
  GELOGD("Stage %d await start.", stage);
  const Status ret = GetSubject(stage).Await();
  GELOGD("Stage %d await ended.", stage);
  return ret;
}

void StageSubject::Release(const int32_t stage) {
  GetSubject(stage).Release();
}

Status StageSubject::Cond::Await() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  if ((!first_exe_) && (!cv_.wait_for(lk,
                                      std::chrono::seconds(kWaitTimeoutInSec),
                                      [this]() { return is_released_; }))) {
    GELOGE(INTERNAL_ERROR, "[Invoke][wait_for]Wait timed out.");
    REPORT_INNER_ERR_MSG("E19999", "wait timed out[%d].", kWaitTimeoutInSec);
    return INTERNAL_ERROR;
  }
  first_exe_ = false;
  is_released_ = false;
  return SUCCESS;
}

void StageSubject::Cond::Release() {
  const std::unique_lock<std::mutex> lk(cond_mu_);
  is_released_ = true;
  cv_.notify_all();
}

HybridModelPipelineExecutor::HybridModelPipelineExecutor(HybridModel *const model, const uint32_t device_id,
                                                         const rtStream_t stream)
    : HybridModelExecutor(model, device_id, stream) {
  config_.num_executors = kNumExecutors;
  config_.num_stages = static_cast<int32_t>(model_->GetRootGraphItem()->NumGroups());
  config_.device_id = device_id_;
  config_.iteration_end = 0;
}

Status StageExecutor::InitExecutionContext() {
  GE_CHK_RT_RET(rtCtxSetCurrent(context_.rt_context));

  context_.model = model_;
  context_.session_id = ::ge::GetContext().SessionId();
  GELOGD("session id from model = %lu, from context = %lu", model_->GetSessionId(), context_.session_id);
  context_.allocator = NpuMemoryAllocator::GetAllocator(pipe_config_->device_id, stream_);
  GE_CHECK_NOTNULL(context_.allocator);
  context_.callback_manager = new (std::nothrow) RtCallbackManager();
  GE_CHECK_NOTNULL(context_.callback_manager);
  context_.own_callback_manager = true;
  context_.dump_properties = DumpManager::GetInstance().GetDumpProperties(context_.session_id);
  context_.is_eos_ = false;
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    context_.trace_enabled = true;
  }
  return SUCCESS;
}

Status StageExecutor::SetInputs(const std::vector<TensorValue> &inputs,
                                const std::vector<ConstGeTensorDescPtr> &input_desc) const {
  root_graph_executor_->Reset();
  (void)root_graph_executor_->InitInputs(inputs, input_desc);
  return SUCCESS;
}

Status StageExecutor::GetOutputs(std::vector<TensorValue> &outputs,
                                 std::vector<ConstGeTensorDescPtr> &output_desc) const {
  return root_graph_executor_->GetOutputs(outputs, output_desc);
}

void StageExecutor::Reset() {
  task_queue_.Stop();
  task_queue_.Clear();
  task_queue_.Restart();
}

Status HybridModelPipelineExecutor::Init(CallbackManager *const callback_manager) {
  (void)callback_manager;
  GE_CHK_STATUS_RET_NOLOG(context_.InitProfiler());
  model_id_ = model_->GetModelId();
  GELOGD("Number of stages = %d, number of executors = %d", config_.num_stages, config_.num_executors);
  GE_CHK_RT_RET(rtCtxGetCurrent(&config_.rt_context));
  GE_CHK_STATUS_RET_NOLOG(InitStageExecutors());
  GE_CHK_STATUS_RET(InitInputDesc(), "[Init][InputDesc] failed, model_id:%u.", model_->GetModelId());
  return SUCCESS;
}

Status HybridModelPipelineExecutor::InitStageExecutors() {
  for (int32_t i = 0; i < config_.num_executors; ++i) {
    auto stage_executor = MakeUnique<StageExecutor>(i, model_, &config_, &stage_subject_);
    GE_CHECK_NOTNULL(stage_executor);
    GE_CHK_STATUS_RET_NOLOG(stage_executor->Init());

    if (context_.profiler != nullptr) {
      // will call unique_ptr::release later
      stage_executor->context_.profiler.reset(context_.profiler.get());
    }

    stage_executors_.emplace_back(std::move(stage_executor));
  }

  // build propagation loop
  for (int32_t i = 0; i < (config_.num_executors - 1); ++i) {
    const int32_t index = i + 1;
    stage_executors_[static_cast<size_t>(i)]->SetNext(stage_executors_[static_cast<size_t>(index)].get());
  }
  const int32_t executor_index = config_.num_executors - 1;
  stage_executors_[static_cast<size_t>(executor_index)]->SetNext(stage_executors_[0U].get());
  return SUCCESS;
}

Status HybridModelPipelineExecutor::PreRun(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args) {
  GE_CHK_STATUS_RET(SyncVarData(), "[Invoke][SyncVarData] failed, model_id:%u.", model_id_);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[SyncVarData] End");
  GE_CHK_STATUS_RET(PrepareExecuteArgs(current_data, args),
                    "[Invoke][PrepareExecuteArgs] failed to copy input data to model, model_id:%u.", model_id_);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[CopyInputData] End");
  return SUCCESS;
}

Status HybridModelPipelineExecutor::ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
    std::shared_ptr<ModelListener> listener) {
  PROFILING_SCOPE_CONST(-1, profiling::kModelExecute);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[RunInternal] [iteration = %d] Start", iterator_count_);
  InputData input_data;
  OutputData output_data;
  GenDataInputOutputData(model_id_, inputs, input_data, output_data);
  HybridModelExecutor::ExecuteArgs args;
  auto ret = ProcessOnlineModel(input_data, args);
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[RunInternal] [iteration = %d] End", iterator_count_);
  iterator_count_++;
  GELOGI("run iterator count is %lu, model_id:%u", iterator_count_, model_->GetModelId());
  ret = HandleResult(ret, input_data.index, args, &output_data, listener);
  return ret;
}

Status HybridModelPipelineExecutor::ProcessOnlineModel(const InputData &input_data,
                                                       HybridModelExecutor::ExecuteArgs &args) {
  const auto ret = PreRun(input_data, args);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Invoke][PreRun] failed, model_id:%u.", model_id_);  // [No need to check value]
    return ret;
  }
  GELOGI("HybridModel will execute in pipeline mode");
  const char_t *iter_per_run = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ITER_NUM, iter_per_run);
  if (iter_per_run != nullptr) {
    args.ctrl_args.num_loops = static_cast<int32_t>(strtol(iter_per_run, nullptr, kDefaultLoopCount));
  }
  return Execute(args);
}

Status HybridModelPipelineExecutor::Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) {
  (void)ctrl_args;
  (void)outputs;
  (void)inputs;
  GELOGE(FAILED, "Pipeline executor not support upper interface RunGraph");
  return FAILED;
}

Status HybridModelPipelineExecutor::Execute(ExecuteArgs &args) {
  const int32_t loop_count = args.ctrl_args.num_loops;
  GE_CHECK_GE(loop_count, kMinLoopCount);

  auto &inputs = args.inputs;
  auto &input_desc = args.input_desc;
  // Start schedulers
  std::vector<std::future<Status>> futures;
  for (size_t i = 0U; i < stage_executors_.size(); ++i) {
    GELOGD("Starting executor %zu", i);
    const auto executor = stage_executors_[i].get();
    executor->Reset();
    auto future = std::async([loop_count, executor, inputs,
                              input_desc](const struct error_message::ErrorManagerContext &error_context) {
      error_message::SetErrMgrContext(error_context);
      return executor->Start(inputs, input_desc, loop_count);
    }, error_message::GetErrMgrContext());

    futures.emplace_back(std::move(future));
  }

  // Push initial tasks
  GELOGD("Start to execute with loops, loop count = %d", loop_count);
  config_.iteration_end = iteration_ + loop_count;
  for (int32_t i = 0; i < config_.num_stages; ++i) {
    StageExecutor::StageTask task_info;
    task_info.stage = i;
    task_info.iteration = iteration_;
    (void)stage_executors_[0U]->ExecuteAsync(task_info);
  }

  // Wait for end of iterations
  bool has_error = false;
  for (size_t i = 0U; i < stage_executors_.size(); ++i) {
    GELOGD("Start to sync result of executor[%zu]", i);
    auto ret = futures[i].get();
    if (ret == END_OF_SEQUENCE) {
      args.ctrl_args.is_eos = true;
      continue;
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "[Check][Result][Executor: %zu] Failed to schedule tasks.", i);
      REPORT_INNER_ERR_MSG("E19999", "[Executor: %zu] Failed to schedule tasks.", i);
      has_error = true;
      continue;
    }

    ret = stage_executors_[i]->Synchronize();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Invoke][Synchronize] failed for [Executor: %zu].", i);
      REPORT_INNER_ERR_MSG("E19999", "[Executor: %zu] failed to Synchronize result.", i);
      has_error = true;
      continue;
    }
  }

  // record for profiling analyzer
  RECORD_MODEL_EXECUTION_EVENT(&context_, "[Cleanup] End");

  if (context_.profiler != nullptr) {
    context_.profiler->Dump(std::cout);
  }

  iteration_ = config_.iteration_end;

  if (has_error) {
    GELOGE(FAILED, "[Check][Error]Error occurred while execution.");
    REPORT_INNER_ERR_MSG("E19999", "Error occurred while execution.");
    return FAILED;
  }

  const auto last_iter_executor_idx = static_cast<size_t>(loop_count) % stage_executors_.size();
  GE_CHK_STATUS_RET(stage_executors_[last_iter_executor_idx]->GetOutputs(args.outputs, args.output_desc),
                    "[Get][Outputs]Failed from executor[%zu]", last_iter_executor_idx);
  GELOGD("execute with loops successfully, loop count = %d", loop_count);
  return SUCCESS;
}

HybridModelPipelineExecutor::~HybridModelPipelineExecutor() {
  GELOGD("~HybridModelPipelineExecutor()");
  for (auto &executor : stage_executors_) {
    (void)executor->context_.profiler.release();
  }
}
void HybridModelPipelineExecutor::Stop() {}
}  // namespace hybrid
}  // namespace ge
