/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_

#include "aicpu_engine_struct.h"
#include "graph/types.h"
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"

namespace ge {
namespace hybrid {
class AicpuNodeTaskBase : public NodeTask {
 public:
  AicpuNodeTaskBase(const NodeItem *const node_item, const domi::TaskDef &task_def)
      : NodeTask(), node_item_(node_item), task_def_(task_def),
        node_name_(node_item->node_name), node_type_(node_item->node_type),
        unknown_type_(node_item->shape_inference_type),
        aicpu_ext_handle_(node_item->node_name,
                          static_cast<uint32_t>(node_item->num_inputs),
                          static_cast<uint32_t>(node_item->num_outputs),
                          node_item->shape_inference_type),
        op_name_(node_item->node->GetOpDesc()->GetName()) {}

  ~AicpuNodeTaskBase() override;
  GE_DELETE_ASSIGN_AND_COPY(AicpuNodeTaskBase);

  using NodeTask::Init;

  virtual Status Init(const HybridModel &model) = 0;

  Status UpdateArgs(TaskContext &context) override;

  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;

  void SetNeedHostMemOpt(const bool need_host_mem_opt) override { need_host_mem_opt_ = need_host_mem_opt; }

 protected:
  virtual Status CheckOverflow(TaskContext &context) const;

  virtual Status InitExtInfo(const std::string &kernel_ext_info, const uint64_t session_id);

  virtual Status UpdateExtInfo();

  virtual Status UpdateOutputShapeFromExtInfo(TaskContext &context);

  Status UpdateShapeToOutputDesc(const TaskContext &context, const GeShape &shape_new,
                                 const int32_t output_index) const;

  Status UpdateBlockDimInfo(const int32_t block_dim_index);

  virtual Status LaunchTask(TaskContext &context) = 0;

  virtual Status InitForDependComputeTask() = 0;

  Status TaskCallback(TaskContext &context);

  virtual Status UpdateShapeAndDataByResultSummary(TaskContext &context);

  virtual Status UpdateIoAddr(TaskContext &context) = 0;

  Status AllocTensorBuffer(const size_t size, std::unique_ptr<TensorBuffer> &tensor_buffer,
                           NpuMemoryAllocator *const allocator = NpuMemoryAllocator::GetAllocator()) const;
  virtual Status CopyDataToHbm(TaskContext &context,
                               const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) = 0;

  /// read result summary and prepare copy task memory.
  /// @param context task context
  /// @param out_shape_hbm if scalar, TensorBuffer->data is null, size=0
  /// @return SUCCESS:success other:failed
  Status ReadResultSummaryAndPrepareMemory(const TaskContext &context,
                                           std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  Status AllocOutputBuffer(const TaskContext &context, const int32_t idx, const uint64_t data_size) const;

  Status UpdateShapeByHbmBuffer(const TaskContext &context,
                                const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  Status PrepareCopyInputs(const TaskContext &context,
                           const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm);

  Status DistributeWaitTaskForAicpuBlockingOp(rtStream_t stream) const;
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const;
  Status UpdateEventIdForBlockingAicpuOp();
  void SetTaskTag() const;
  void InitBlockAicpuOp(const OpDescPtr& op_desc);

 private:
  const NodeItem *node_item_;
  // just reference.
  const domi::TaskDef &task_def_;

  const std::string node_name_;

  const std::string node_type_;

  std::string stream_sync_timeout_ = "-1";

  // valid when node_item_->is_dynamic is true
  UnknowShapeOpType unknown_type_ = DEPEND_IN_SHAPE;

  // valid when node_item_->is_dynamic is true
  AicpuExtInfoHandler aicpu_ext_handle_;

  // ext info addr, device mem
  std::unique_ptr<TensorBuffer> ext_info_addr_dev_;

  std::vector<std::unique_ptr<TensorBuffer>> output_summary_;
  std::vector<aicpu::FWKAdapter::ResultSummary> output_summary_host_;

  std::unique_ptr<TensorBuffer> copy_input_release_flag_dev_;
  std::unique_ptr<TensorBuffer> copy_input_data_size_dev_;
  std::unique_ptr<TensorBuffer> copy_input_src_dev_;
  std::unique_ptr<TensorBuffer> copy_input_dst_dev_;
  // for blocking aicpu op
  bool is_blocking_aicpu_op_ = false;
  rtEvent_t rt_event_ = nullptr;
  std::string op_name_;
  friend class AicpuTfNodeTask;
  friend class AicpuNodeTask;
  friend class HostAicpuNodeTask;
  bool is_support_block_ = false;
  uint32_t block_num_ = 1U;
  int32_t axis_index_ = -1;
  // for host aicpu or device aicpu
  int32_t deploy_type_flag_ = 0;
  aclrtMemcpyKind memcpy_kind_{ACL_MEMCPY_HOST_TO_DEVICE};
  // for 3,4 aicpu callback
  aclrtMemcpyKind callback_memcpy_kind_{ACL_MEMCPY_DEVICE_TO_HOST};
  // for qos
  uint32_t qos_level_flag_ = 0U;
  bool need_host_mem_opt_ = false;
  size_t host_mem_input_data_offset_ = 0U;
};

class AicpuTfNodeTask : public AicpuNodeTaskBase {
 public:
  AicpuTfNodeTask(const NodeItem *const node_item, const domi::TaskDef &task_def)
      : AicpuNodeTaskBase(node_item, task_def) {}

  ~AicpuTfNodeTask() override = default;
  Status Init(const HybridModel &model) override;

  bool IsSupportHostMemInputOpt() const override { return true; }
  bool IsArgsExtendedForHostMemInput() const override { return host_mem_input_data_offset_ != 0U; }

 protected:

  Status LaunchTask(TaskContext &context) override;

  Status UpdateIoAddr(TaskContext &context) override;

  Status UpdateHostMemInputArgs(const TaskContext &context, void *const args,
                                const size_t args_size);

  Status InitForDependComputeTask() override;

  Status CopyDataToHbm(TaskContext &context,
                       const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) override;
 private:
  Status SetMemCopyTask(const domi::TaskDef &task_def);

  static Status EnsureSessionCreated(const uint64_t session_id);
  static uint64_t GetStepIdAddr(const HybridModel &model);
  Status AssembleWorkSpaceAddr(const domi::KernelExDef &kernel_ex_def);
  Status AssembleKernelBuffer(STR_FWK_OP_KERNEL *fwk_op_kernel);
  Status InitTopicTypAndExtInfo(const HybridModel &model);
  // kernel buf,
  // device mem when topic type is default
  // or svm smm when topic type is host only
  std::unique_ptr<TensorBuffer> kernel_buf_;

  std::unique_ptr<TensorBuffer> kernel_workspace_;

  // input and output addr,
  // device mem when topic type is default
  // or svm smm when topic type is host only
  std::unique_ptr<TensorBuffer> input_output_addr_;

  // just used for depend DEPEND_COMPUTE op
  std::unique_ptr<TensorBuffer> copy_task_args_buf_;
  // device mem when topic type is default
  // or svm smm when topic type is host only
  std::unique_ptr<TensorBuffer> copy_ioaddr_dev_;
  bool need_sync_ = false;

  std::unique_ptr<TensorBuffer> copy_workspace_buf_;
};

class AicpuNodeTask : public AicpuNodeTaskBase {
 public:
  AicpuNodeTask(const NodeItem *const node_item, const domi::TaskDef &task_def)
      : AicpuNodeTaskBase(node_item, task_def) {}

  ~AicpuNodeTask() override = default;

  Status Init(const HybridModel &model) override;

  bool IsSupportHostMemInputOpt() const override { return true; }
  bool IsArgsExtendedForHostMemInput() const override { return host_mem_input_data_offset_ != 0U; }

 protected:

  Status LaunchTask(TaskContext &context) override;

  Status CopyDataToHbm(TaskContext &context,
                       const std::vector<std::unique_ptr<TensorBuffer>> &out_shape_hbm) override;

  Status UpdateIoAddr(TaskContext &context) override;

  Status UpdateHostMemInputArgs(const TaskContext &context);

  Status InitForDependComputeTask() override;

 private:
  Status CheckOverflow(TaskContext &context) const override;

  Status SetMemCopyTask(const domi::TaskDef &task_def);

  Status InitTopicTypAndExtInfo(const HybridModel &model);

  // host mem
  std::unique_ptr<uint8_t[]> args_;
  // host memcpy mem
  std::unique_ptr<uint8_t[]> memcpy_args_;

  std::string memcpy_so_name_;

  std::string memcpy_kernel_name_;
  // args size
  uint32_t memcpy_args_size_ = 0U;

  std::vector<uint64_t> copy_io_addr_;
  // args size
  uint32_t args_size_ = 0U;

  std::unique_ptr<rtHostInputInfo_t[]> host_inputs_info_;
  rtArgsEx_t args_ex_ = {};
  friend class HostAicpuNodeTask;
};

class AiCpuNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model,
                  const NodePtr &node,
                  std::shared_ptr<NodeTask> &task) const override;

  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
};
}
}
#endif  // GE_HYBRID_KERNEL_AICPU_NODE_EXECUTOR_H_
