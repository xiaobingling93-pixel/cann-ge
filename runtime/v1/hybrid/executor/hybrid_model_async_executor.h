/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_MODEL_HYBRID_MODEL_ASYNC_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_MODEL_HYBRID_MODEL_ASYNC_EXECUTOR_H_
#include <atomic>
#include <mutex>
#include <future>
#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "common/dump/opdebug_register.h"
#include "graph/load/model_manager/data_inputer.h"
#include "common/dump/data_dumper.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/hybrid_model_rt_v1_executor.h"
#include "hybrid/executor/hybrid_model_pipeline_executor.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"

namespace ge {
namespace hybrid {
class HybridModelAsyncExecutor {
 public:
  struct DefaultStreamGuarder {
    rtStream_t default_stream = nullptr;
    uint32_t stream_ref_count = 0U;
    std::mutex mu;
  };
  explicit HybridModelAsyncExecutor(HybridModel *const model);
  ~HybridModelAsyncExecutor();

  Status Init(const rtStream_t stream = nullptr);

  // dflow
  Status Execute(const std::vector<DataBuffer> &inputs,
                 const std::vector<GeTensorDesc> &input_desc,
                 std::vector<DataBuffer> &outputs,
                 std::vector<GeTensorDesc> &output_desc,
                 rtStream_t stream = nullptr);

  Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  Status ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                 rtStream_t stream = nullptr);
  
  Status ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                               std::vector<gert::Tensor> &outputs,
                                               rtStream_t stream = nullptr);

  Status Start(const std::shared_ptr<ModelListener> &listener);

  void SetDeviceId(const uint32_t device_id);

  void SetModelId(const uint32_t model_id);

  Status Stop();

  Status EnqueueData(const std::shared_ptr<RunArgs> &args);

  uint32_t GetDataInputerSize() const { return data_inputer_->Size(); }

  bool GetRunningFlag() const { return running_flag_; }

  void SetRunningFlag(const bool flag) { running_flag_ = flag; }

  const GraphExecutionContext *GeContext() const { return executor_->GetContext(); }

  GraphExecutionContext *GeContext() {return executor_->GetContext(); }

 private:
  Status RunInternal();

  Status BuildExecutor();

  DefaultStreamGuarder &GetDefaultStreamGuarder() const;

  static std::map<std::pair<uint32_t, uint32_t>, DefaultStreamGuarder> default_stream_by_dev_;
  static std::mutex mu_for_guarder_;

  HybridModel *model_;
  uint32_t device_id_ = 0U;
  uint32_t model_id_ = 0U;
  std::atomic_bool run_flag_;
  // check whether model is running with data
  bool running_flag_ = false;
  std::unique_ptr<DataInputer> data_inputer_;
  std::unique_ptr<HybridModelExecutor> executor_;
  std::future<Status> future_;

  rtStream_t stream_ = nullptr;
  bool owner_stream_ = false;
  std::shared_ptr<ModelListener> listener_;
  std::vector<TensorValue> output_cache_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_MODEL_HYBRID_MODEL_ASYNC_EXECUTOR_H_
