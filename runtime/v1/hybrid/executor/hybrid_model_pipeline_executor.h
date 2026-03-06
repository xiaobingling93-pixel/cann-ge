/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_MODEL_PIPELINE_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_HYBRID_MODEL_PIPELINE_EXECUTOR_H_

#include "common/blocking_queue.h"
#include "common/thread_pool.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/executor/subgraph_executor.h"
#include "hybrid/executor/hybrid_model_executor.h"

namespace ge {
namespace hybrid {

struct PipeExecutionConfig {
  uint32_t device_id;
  rtContext_t rt_context;
  int32_t num_executors;
  int32_t num_stages;
  int64_t iteration_end;
};

class StageSubject {
 public:
  Status Await(const int32_t stage);
  void Release(const int32_t stage);

 private:
  class Cond {
   public:
    void Release();
    Status Await();
   private:
    std::mutex cond_mu_;
    std::condition_variable cv_;
    bool first_exe_ = true;
    bool is_released_ = false;
  };
  Cond &GetSubject(const int32_t stage);
  std::mutex mu_;
  std::unordered_map<int32_t, Cond> subjects_;
};

class StageExecutor {
 public:
  struct StageTask {
    rtEvent_t event = nullptr;
    int32_t stage = 0;
    int64_t iteration = 0;
    bool is_eos = false;
  };

  StageExecutor(const int32_t id, HybridModel *const model, PipeExecutionConfig *const config,
                StageSubject *const stage_subject);

  ~StageExecutor();

  Status Init();

  void Reset();

  Status Start(const std::vector<TensorValue> &inputs, const std::vector<ConstGeTensorDescPtr> &input_desc,
               const int32_t iteration_count);

  void ReleaseCallback();

  void ExecuteEndTaskAndReleae();

  Status SetInputs(const std::vector<TensorValue> &inputs, const std::vector<ConstGeTensorDescPtr> &input_desc) const;

  Status ExecuteAsync(const StageTask &args);

  Status GetOutputs(std::vector<TensorValue> &outputs, std::vector<ConstGeTensorDescPtr> &output_desc) const;

  Status Synchronize();

  void SetNext(StageExecutor *const next_executor) { next_executor_ = next_executor; }

 private:
  friend class HybridModelPipelineExecutor;
  static Status ResetExecutionContext(GraphExecutionContext &context);
  Status InitExecutionContext();

  int32_t id_;
  HybridModel *model_;

  PipeExecutionConfig *pipe_config_;
  StageSubject *stage_subject_;
  BlockingQueue<StageTask> task_queue_;
  std::unique_ptr<SubgraphExecutor> root_graph_executor_;
  GraphExecutionContext context_;
  StageExecutor *next_executor_ = nullptr;

  rtStream_t stream_ = nullptr;
  rtStream_t hccl_stream_ = nullptr;
};

class HybridModelPipelineExecutor : public HybridModelExecutor {
 public:
  HybridModelPipelineExecutor(HybridModel *const model, const uint32_t device_id, const rtStream_t stream);

  ~HybridModelPipelineExecutor() override;

  Status Init(CallbackManager *const callback_manager = nullptr) override;

  Status Execute(ExecuteArgs &args) override;
  Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) override;

  Status InitStageExecutors();
  Status ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
    std::shared_ptr<ModelListener> listener) override;
  void Stop() override;
 private:
  Status ProcessOnlineModel(const InputData &input_data, HybridModelExecutor::ExecuteArgs &args);
  Status PreRun(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args);
  uint32_t device_id_;
  std::vector<std::unique_ptr<StageExecutor>> stage_executors_;
  StageSubject stage_subject_;
  PipeExecutionConfig config_;
  GraphExecutionContext context_;
  int64_t iteration_ = 0;
};
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_EXECUTOR_HYBRID_MODEL_PIPELINE_EXECUTOR_H_
