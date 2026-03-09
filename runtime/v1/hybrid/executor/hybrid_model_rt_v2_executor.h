/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_HYBRID_MODEL_RT2_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_HYBRID_MODEL_RT2_EXECUTOR_H_

#include "common/blocking_queue.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/thread_pool/thread_pool.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/rt_session.h"
#include "framework/runtime/subscriber/built_in_subscriber_definitions.h"
#include "graph/manager/graph_var_manager.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/rt_callback_manager.h"
#include "hybrid/executor/subgraph_executor.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/runtime_v2/rt_v2_executor_factory.h"
#include "runtime_v2/scalable_allocator_manager.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace hybrid {
struct H2DCopyHelper {
  std::string node_name;
  void *dev_addr;
  std::string file_path;
  size_t offset;
  size_t file_length;
  size_t left_size;
};

struct SharedConstantCopyHelper {
  SharedConstantCopyHelper(const int64_t out_size, const uint8_t *out_addr, const ConstGeTensorPtr &out_weight,
                           const OpDescPtr &out_desc)
      : size(out_size), addr(const_cast<uint8_t *>(out_addr)), weight(out_weight), op_desc(out_desc) {}
  int64_t size = 0L;
  uint8_t *addr = nullptr;
  ConstGeTensorPtr weight = nullptr;
  OpDescPtr op_desc = nullptr;
};

class GraphVarVisitor : public gert::RtVarManager {
  enum class VariablePlacement {
    kOnHost,
    kOnDeviceHbm,
    kOnDeviceRdma,
    kVariablePlacementEnd
  };
  struct Variable {
    VariablePlacement placement;
    GeTensorDesc desc;
    int64_t size = 0U;
    uint8_t *addr = nullptr;
    std::string DebugString() const;
    static std::string ReadablePlacement(const VariablePlacement placement);
  };

 public:
  Status Init(const std::shared_ptr<ge::VarManager> &var_manager, uint32_t device_id, uint64_t session_id,
              uint32_t graph_id);
  Status GetVarShapeAndMemory(const std::string &id, gert::StorageShape &shape,
                              gert::TensorData &memory) const override;

  Status AssembleVariables(const std::vector<ge::NodePtr> &variables);
  Status AssembleHostVariables(const std::vector<ge::NodePtr> &variables);
  Status AssembleDeviceVariables(const std::vector<ge::NodePtr> &variables);
  Status AssembleSharedConstants(const std::vector<ge::NodePtr> &shared_constants);
  Status AssembleDeviceSharedConstants(const std::vector<ge::NodePtr> &shared_constants);
  Status AssembleHostSharedConstants(const std::vector<ge::NodePtr> &shared_constants);
  Status AssembleFileConstants(const std::vector<ge::NodePtr> &file_constants);

  Status AssignVarLogicalMemory(const ge::NodePtr &node, bool &assigned);
  Status AssignVarLogicalMemory(const ge::NodePtr &node);

  Status GetVarDeviceInstance(const ge::NodePtr &node, Variable &var_instance);

 private:
  void *GetOrCreateVarMem(const std::string &var_name,
                          const OpDescPtr &var_desc,
                          const rtMemType_t memory_type) const;

  Status LoadFileConstantToAddr(const ge::OpDescPtr &op_desc, const Variable &var_instance) const;
  Status LoadFileConstantToDevice(const ExternalWeightManagerPtr &manager, const uint32_t device_id,
                                  std::vector<H2DCopyHelper> &node_infos) const;
  Status PreLoadFileConstant(const ge::OpDescPtr &op_desc, const Variable &var_instance,
                             const std::map<std::string, std::string> &file_id_path, H2DCopyHelper &helper) const;
  Status MultiThreadH2DCopy(std::unordered_map<uint32_t, std::vector<H2DCopyHelper>> &multi_node_infos);
  Status CopySharedConstant(const std::shared_ptr<ge::VarManager> &var_manager, uint32_t device_id,
                            const std::vector<SharedConstantCopyHelper> &helpers) const;
  Status MultiThreadSharedConstantCopy(
      const std::unordered_map<uint32_t, std::vector<SharedConstantCopyHelper>> &helpers) const;
  Status AssembleFileConstantsOnDevice(const std::vector<ge::NodePtr> &file_constants);
  Status AssembleFileConstantsOnHost(const std::vector<ge::NodePtr> &file_constants);

  uint32_t device_id_ = 0U;
  uint64_t session_id_ = 0U;
  uint32_t graph_id_ = 0U;
  bool is_visitor_for_host_ = false;
  std::shared_ptr<ge::VarManager> session_var_manager_;
  std::map<std::string, Variable> named_variables_;
};
class HybridModelRtV2Executor : public HybridModelExecutor {
 public:
  explicit HybridModelRtV2Executor(HybridModel *const model, uint32_t device_id, const rtStream_t stream);
  ~HybridModelRtV2Executor() override {
    if (executor_ != nullptr) {
      (void)executor_->Unload();
    }
    if (guard_check_info_.guard_handle != nullptr) {
      (void)mmDlclose(guard_check_info_.guard_handle);
    }
    if (guard_check_info_.guard_so_fd != -1) {
      (void)mmClose(guard_check_info_.guard_so_fd);
    }
  }
  Status ExecuteOnlineModel(const std::vector<gert::Tensor> &inputs,
                            std::shared_ptr<ModelListener> listener) override;
  Status Init(CallbackManager *const callback_manager = nullptr) override;
  Status ExecuteWithStreamAsync(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs,
                                const rtStream_t stream = nullptr) override;
  Status ExecuteWithStreamAsync(const std::vector<gert::Tensor> &inputs,
                                std::vector<gert::Tensor> &outputs, const rtStream_t stream) override;
  Status Execute(const InputData &input_data, ExecuteArgs &args);
  Status Execute(ExecuteArgs &args) override;
  Status Execute(const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs,
    CtrlArgs &ctrl_args) override;
  void ResetMemcpyBatchParams();
  void Stop() override;
  bool NeedBuildDeviceTensorAsOutput() const override;
  Status BuildDeviceTensor(TensorValue &output_tensor, GeTensorDesc &ge_tensor_desc, const int64_t output_size,
                           std::vector<ge::Tensor> &outputs) const override;
  Status StepDone() const;
  void StepDoneV2();
  GraphExecutionContext* GetContext() override {
    return &context_;
  }
  Status HandleResult(const Status exec_ret, const uint32_t data_id, HybridModelExecutor::CtrlArgs &ctrl_args,
    std::vector<gert::Tensor> &outputs, std::shared_ptr<ModelListener> listener) const override;
 private:
  Status LoadGuardFunc(const ge::ComputeGraphPtr &graph);
  Status CheckGuard();
  Status PrepareInputData(InputData &current_data, const HybridModelExecutor::ExecuteArgs &args) const;
  Status PostProcResult(std::vector<GeTensor> &outputs) const;
  Status InitCtx();
  Status CheckInputIsOnDevice();
  Status AllocatorRecycle(const rtStream_t stream) const;
  Status RecycleOutputs(std::vector<gert::Tensor> &outputs, const rtStream_t stream) const;
  Status TryUpdateStreamCoreLimits(const rtStream_t stream);

  class RunCtx {
    friend class HybridModelRtV2Executor;

   private:
    Status Init(HybridModel *model);
    bool host_exec_flag_ = false;
    uint32_t device_id_ = 0U;
    uint64_t session_id_ = 0U;
    uint32_t graph_id_ = 0U;
    size_t iterations_per_loop_ = 1U;
    std::string model_name_;
    std::string graph_name_;
    GraphVarVisitor graph_var_visitor_;
    gert::RtSession session_;
    std::string execute_mode_;
    std::string is_copy_output_addr_;
    // rts resource finalize need follow event, stream, context, rtDeviceReset
    // unload graph will call rtDeviceReset, so resource allocator need finalize before
    DevResourceAllocator dev_resource_allocator_;
    bool enable_input_batch_cpy_;
    std::string aicore_num_str_;
    std::string vectorcore_num_str_;
  };

  struct GuardCheckInfo {
    // guard check 函数
    GuardCheckFunc guard_check_func{nullptr};
    // guard check so
    void *guard_handle{nullptr};
    // guard check内存系统文件符
    int32_t guard_so_fd{-1};
  };

  std::string name_;
  ScalableAllocatorManager allocator_manager_;
  // rts stream resource need destroy before mem allocator, otherwise may case rtunmap error in multistream scene
  RunCtx run_ctx_;
  std::unique_ptr<gert::RtV2ExecutorInterface> executor_;
  size_t num_inputs_;
  size_t num_outputs_;

  std::vector<GeTensorDescPtr> output_descs_;

  std::vector<gert::Tensor> inputs_holder_;
  std::vector<gert::Tensor> outputs_holder_;

  std::vector<gert::Tensor *> rt_inputs_;
  std::vector<gert::Tensor *> rt_outputs_;
  std::unique_ptr<ProfilerCollector> profiler_collector_;
  GraphExecutionContext context_;
  GEThreadLocalContext ge_context_;
  uint8_t logLevel_ = DLOG_DEBUG;
  GuardCheckInfo guard_check_info_;
  MemcpyBatchParam memcpy_batch_params_;
};
}  // namespace hybrid
}  // namespace ge

#endif  // GE_HYBRID_EXECUTOR_HYBRID_MODEL_RT2_EXECUTOR_H_
