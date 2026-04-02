/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_MODEL_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_HYBRID_MODEL_EXECUTOR_H_
#include "common/thread_pool/thread_pool.h"
#include "graph/load/model_manager/data_inputer.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/callback_manager.h"
#include "hybrid/executor/subgraph_executor.h"

namespace ge {
namespace hybrid {
class HybridModelExecutor {
 public:
  struct CtrlArgs {
    bool is_eos = false;
    int32_t num_loops = 10;
    aclrtStream stream = nullptr;
  };

  struct ExecuteArgs {
    std::vector<TensorValue> inputs;
    std::vector<ConstGeTensorDescPtr> input_desc;
    std::vector<TensorValue> outputs;
    std::vector<ConstGeTensorDescPtr> output_desc;
    CtrlArgs ctrl_args;
  };

  explicit HybridModelExecutor(HybridModel *const model, uint32_t device_id, aclrtStream stream)
    : model_(model),
      device_id_(device_id),
      stream_(stream) {};
  virtual ~HybridModelExecutor() {};
  virtual Status Init(CallbackManager *const callback_manager = nullptr) = 0;
  // interface is called by sess.RunGraphWithStream with dynamic model, will return output tensor to usr
  // only support rt2 executor due to dynamic model is execute by rt2 defult, other executor will report error
  virtual Status ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                                        const aclrtStream stream = nullptr);
  virtual Status ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                                       std::vector<gert::Tensor> &outputs,
                                                       const aclrtStream stream = nullptr);
  // interface is called by sess.RunGraphWithStream and LoadModelWithQ
  virtual Status Execute(ExecuteArgs &args) = 0;
  // interface is called by sess.RunGraph
  virtual Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) = 0;
  // interface is called by sess.RunGraphAsync
  virtual Status ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
    std::shared_ptr<ModelListener> listener) = 0;
  virtual void Stop() = 0;
  virtual GraphExecutionContext* GetContext() {return nullptr;};
  virtual bool NeedBuildDeviceTensorAsOutput() const {return false;};
  virtual Status BuildDeviceTensor(TensorValue &output_tensor, GeTensorDesc &ge_tensor_desc, const int64_t output_size,
                                   std::vector<ge::Tensor> &outputs) const;
  Status CopyOutputs(HybridModelExecutor::ExecuteArgs &args, OutputData *const output_data,
                     std::vector<ge::Tensor> &outputs) const;
  Status CopyOutputs(const std::vector<gert::Tensor> &executor_outputs, std::vector<gert::Tensor> &uer_outputs) const;
  static void ParserContextOption(const std::string &option_name, std::string &option_value);
  void GenDataInputOutputData(const uint32_t model_id, const std::vector<gert::Tensor> &inputs,
      InputData &input_data, OutputData &output_data) const;
 protected:
  Status InitInputDesc();
  Status SyncVarData() const;
  Status PrepareExecuteArgs(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args);
  Status PrepareDynamicInput(HybridModelExecutor::ExecuteArgs &args, const size_t input_index,
                             const GeShape &shape, const DataBuffer &data_buf, int64_t &tensor_size);
  Status CopyDataToExecutArgs(const int64_t tensor_size, HybridModelExecutor::ExecuteArgs &args,
                              const size_t input_index, const DataBuffer &data_buf) const;
  virtual Status HandleResult(const Status exec_ret,
                              const uint32_t data_id, HybridModelExecutor::ExecuteArgs &args,
                              OutputData *const output_data, std::shared_ptr<ModelListener> listener) const;
  virtual Status HandleResult(const Status exec_ret, const uint32_t data_id, HybridModelExecutor::CtrlArgs &ctrl_args,
      std::vector<gert::Tensor> &outputs, std::shared_ptr<ModelListener> listener) const;
  Status OnComputeDone(const uint32_t data_index, const uint32_t result_code, std::vector<ge::Tensor> &outputs,
                       const std::shared_ptr<ModelListener> listener) const;
  Status OnComputeDone(const uint32_t data_index, const uint32_t result_code, std::vector<gert::Tensor> &outputs,
                       const std::shared_ptr<ModelListener> listener) const;
  uint64_t iterator_count_ = 0U;
  uint32_t model_id_ = 0U;

  HybridModel *model_ = nullptr;
  uint32_t device_id_;
  aclrtStream stream_;
  std::map<uint32_t, int64_t> index_to_tensor_size_;
  std::map<uint32_t, GeTensorDescPtr> index_to_tensor_desc_;
  std::vector<bool> is_input_dynamic_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_HYBRID_MODEL_EXECUTOR_H_
