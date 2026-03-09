/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_MODEL_RTV1_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_HYBRID_MODEL_RTV1_EXECUTOR_H_
#include "hybrid/executor/hybrid_model_executor.h"
#include "common/thread_pool/thread_pool.h"
#include "graph/load/model_manager/data_inputer.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/callback_manager.h"
#include "hybrid/executor/subgraph_executor.h"

namespace ge {
namespace hybrid {
class HybridModelRtV1Executor : public HybridModelExecutor {
 public:
  HybridModelRtV1Executor(HybridModel *const model, const uint32_t device_id, const rtStream_t stream,
                          ThreadPool *const thread_pool = nullptr);

  ~HybridModelRtV1Executor() override {
    // release the memory held by op_debug_register_.p2p_debug_addr_
    // hybrid_model_rt_v1_executor.h is deprecated and will be removed later
    Stop();
  };

  Status Init(CallbackManager *const callback_manager = nullptr) override;

  GraphExecutionContext* GetContext() override {
    return &context_;
  }

  bool NeedBuildDeviceTensorAsOutput() const override;

  Status Execute(ExecuteArgs &args) override;
  Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) override;

  Status Execute(const InputData &input_data, ExecuteArgs &args);

  Status ExecuteForSingleOp(const HybridModelExecutor::ExecuteArgs &args);

  Status ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
    std::shared_ptr<ModelListener> listener) override;
 private:
  Status ProcessOnlineModel(const InputData &input_data, HybridModelExecutor::ExecuteArgs &args);
  Status ExecuteGraphInternal(SubgraphExecutor &executor, ExecuteArgs &args);
  Status Cleanup();
  Status InitExecutionContext(CallbackManager *const callback_manager = nullptr);
  static Status ResetExecutionContext(GraphExecutionContext &context);
  static Status CheckInputShapeByShapeRange(const GraphItem *const graph_item,
                                            const HybridModelExecutor::ExecuteArgs &args);
  Status PreRun(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args);
  Status DumpOpDebug();
  void Stop() override;

  GraphExecutionContext context_;
  SubgraphExecutor executor_;
  DataDumper data_dumper_;
  OpdebugRegister op_debug_register_;
  bool is_op_debug_reg_ = false;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_EXECUTOR_HYBRID_MODEL_RTV1_EXECUTOR_H_
