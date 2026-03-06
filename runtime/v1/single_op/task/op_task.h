/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_TASK_OP_TASK_H_
#define GE_SINGLE_OP_TASK_OP_TASK_H_

#include <memory>
#include <string>

#include "common/dump/dump_op.h"
#include "common/dump/dump_properties.h"
#include "common/dump/dump_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/profiling/profiling_properties.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/op_kernel_bin.h"
#include "runtime/stream.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "graph/node.h"
#include "graph/runtime_inference_context.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "graph/utils/op_desc_utils.h"
#include "aicpu_engine_struct.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "register/op_tiling.h"
#include "proto/task.pb.h"
#include "framework/common/ge_types.h"
#include "runtime/rt.h"
#include "single_op/stream_resource.h"
#include "platform/platform_info.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/load/model_manager/model_utils.h"
#include "mmpa/mmpa_api.h"
#include "aprof_pub.h"
#include "framework/runtime/subscriber/global_dumper.h"

namespace ge {
constexpr uint32_t kHostMemType = 1U;
constexpr size_t kAlignBytes4 = 4U;
constexpr size_t kAlignBytes64 = 64U;

struct SingleOpModelParam;
class AtomicAddrCleanOpTask;
class OpTask {
 public:
  OpTask() noexcept
      : op_(nullptr),
        op_desc_(nullptr),
        model_id_(0U),
        block_dim_(1U),
        tiling_key_(0U),
        need_tiling_(false),
        need_host_mem_opt_(false),
        extend_args_for_host_input_(false),
        clear_atomic_(false),
        task_id_(0U),
        stream_id_(0U) {};
  explicit OpTask(const NodePtr &node)
      : op_(MakeUnique<Operator>(
            OpDescUtils::CreateOperatorFromNode(node->shared_from_this()))),
        op_desc_(nullptr),
        model_id_(0U),
        block_dim_(1U),
        tiling_key_(0U),
        need_tiling_(false),
        need_host_mem_opt_(false),
        extend_args_for_host_input_(false),
        clear_atomic_(false),
        task_id_(0U),
        stream_id_(0U) {};
  virtual ~OpTask() noexcept = default;
  virtual Status LaunchKernel(rtStream_t const stream) = 0;
  virtual Status PostProcess(rtStream_t const stream);
  virtual Status PreProcess(uint64_t &launch_begin_time) {
    launch_begin_time = MsprofSysCycleTime();
    return ge::SUCCESS;
  }
  virtual void ResetDumperResource() {
    return;
  }
  Status SaveExceptionDumpInfo();
  virtual void GetHostArgsAndSize(uintptr_t &args, size_t &arg_size);
  virtual void GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) {
    tiling_key = 0U;
    tiling_data = "";
  }
  virtual void SaveForL0ExceptionDump() {
    GELOGD("task name is %s, no need to save for exception dump!", task_name_.c_str());
    return;
  };
  const std::string &GetModelName() const { return model_name_; }
  virtual Status UpdateRunInfo();
  virtual Status UpdateArgTable(const SingleOpModelParam &param);
  void SetModelArgs(const std::string &model_name, const uint32_t model_id);
  Status GetTaskIdAndStreamId(rtStream_t const stream);
  Status ReportProfilingData(const uint64_t begin_time) const;
  Status ReportProfAdditionalInfo(const uint64_t end_time, const uint64_t op_name_hash, const int32_t tid) const;
  virtual Status ReportProfExtendInfo(const uint64_t end_time, const uint64_t op_name_hash, const int32_t tid) const {
    (void)end_time;
    (void)op_name_hash;
    (void)tid;
    return SUCCESS;
  }
  const std::string &GetTaskName() const {return task_name_;}
  void SetOpDesc(const OpDescPtr &op_desc) {
    op_desc_ = op_desc;
  }
  const OpDescPtr &GetOpdesc() const {return op_desc_;}
  Status OpenDump(rtStream_t const stream);
  virtual void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) = 0;
  virtual Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                              const std::vector<DataBuffer> &input_buffers,
                              std::vector<GeTensorDesc> &output_desc,
                              std::vector<DataBuffer> &output_buffers,
                              rtStream_t const stream);
  virtual const std::string &GetTaskType() const;
  bool NeedReportAtomicTask() const { return clear_atomic_ && (atomic_task_ != nullptr); }
  AtomicAddrCleanOpTask *GetAtomicTask() const { return atomic_task_.get(); }
  virtual const std::string GetOpType() const;
  void SetNeedHostMemOpt(const bool need_host_mem_opt);
  void SetHostMemInputFlag(const bool has_host_mem_input);
  bool GetNeedTiling() const;
  void SetRuntimeContext(RuntimeInferenceContext *const context);
  virtual Status UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  virtual bool IsSupportHostMemInputOptimize() const { return false; }
  virtual size_t GetHostMemInputDataOffsetInIoAddr() const { return 0U; }
  virtual size_t GetInputAddrAlignBytes() const { return kAlignBytes4; }
  bool IsArgsExtendedForHostMemInput() const { return extend_args_for_host_input_; }
  virtual void SetPlatform(fe::PlatFormInfos &platform_infos) {
    (void)platform_infos;
  }
  virtual void SetSpaceRegistries(const std::shared_ptr<gert::OpImplSpaceRegistryV2Array> &space_registries) {
    (void)space_registries;
  }
 protected:
  Status DoUpdateArgTable(const SingleOpModelParam &param, const bool keep_workspace);
  void SetTaskTag() const;

 private:
  OpTask(const OpTask &) = delete;
  OpTask &operator=(const OpTask &)& = delete;

  friend class AiCpuTaskBuilder;
  friend class AiCpuCCTaskBuilder;
  friend class TbeTaskBuilder;
  friend class MixL2TaskBuilder;
  friend class SingleOpModel;
  friend class TbeOpTask;
  friend class MixL2OpTask;
  friend class AiCpuBaseTask;
  friend class AiCpuCCTask;
  friend class AiCpuTask;
  friend class AtomicAddrCleanOpTask;

  std::unique_ptr<Operator> op_;
  DumpProperties dump_properties_;
  DumpOp dump_op_;
  OpDescPtr op_desc_;
  std::string model_name_;
  uint32_t model_id_;
  uint32_t block_dim_;
  uint64_t tiling_key_;
  std::string task_name_;
  bool need_tiling_;
  bool need_host_mem_opt_;
  bool extend_args_for_host_input_;
  bool clear_atomic_;
  std::unique_ptr<AtomicAddrCleanOpTask> atomic_task_;
  uint32_t task_id_;
  uint32_t stream_id_;
};

struct ArgItemOffset {
  size_t overflow_addr_offset{0UL};
  size_t workspace_addr_offset{0UL};
  size_t tiling_addr_offset{0UL};
  size_t tiling_data_offset{0UL};
  size_t host_input_data_offset{0UL};
};

class TbeOpTask : public OpTask {
 public:
  TbeOpTask() = default;
  explicit TbeOpTask(const NodePtr &node) : OpTask(node) {}
  ~TbeOpTask() noexcept override;
  Status LaunchKernel(rtStream_t const stream) override;
  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers,
                      rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
  void SetStubFunc(const std::string &name, const void *const stub_func);
  void SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, const size_t arg_size,
                     const uint32_t block_dim, const OpDescPtr &op_desc);
  void SetKernelArgs(std::unique_ptr<uint8_t[]> &&args, const size_t arg_size,
                     const uint32_t block_dim, const OpDescPtr &op_desc,
                     const domi::KernelDef &kernel_def);
  void SetKernelWithHandleArgs(std::unique_ptr<uint8_t[]> &&args, const size_t arg_size,
                               const uint32_t block_dim, const OpDescPtr &op_desc,
                               const domi::KernelDefWithHandle& kernel_def_with_handle);
  void SetAtomicAddrCleanTask(AtomicAddrCleanOpTask *const task) { atomic_task_.reset(task); }
  void SaveForL0ExceptionDump() override;

  Status UpdateRunInfo() override;
  Status UpdateRunInfoByTilingResult();
  Status SetArgIndex();

  void EnableDynamicSupport(const NodePtr &node, const uint32_t max_tiling_size);
  const std::string &GetTaskType() const override;
  void SetHandle(void *const handle);

  void SetOverflowAddr(void *addr) {
    overflow_addr_ = addr;
  }
  Status UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) override;
  bool IsSupportHostMemInputOptimize() const override { return true; }
  size_t GetHostMemInputDataOffsetInIoAddr() const override { return args_item_offsets_.host_input_data_offset; }
  void UpdateArgsItemOffset(const size_t io_size, const size_t workspace_addr_size, size_t &arg_size);
  // for soft sync op
  void SetPlatform(fe::PlatFormInfos &platform_infos) override {
    platform_infos_ = platform_infos;
  }
  void SetSpaceRegistries(const std::shared_ptr<gert::OpImplSpaceRegistryV2Array> &space_registries) override {
    space_registries_ = space_registries;
  }
  void GetHostArgsAndSize(uintptr_t &args, size_t &arg_size) override {
    args = reinterpret_cast<uintptr_t>(args_ex_.args);
    arg_size = args_ex_.argsSize;
  }

  void GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) override {
    tiling_key = static_cast<uint32_t>(tiling_key_);
    if (run_info_ != nullptr) {
      tiling_data = run_info_->GetAllTilingData().str();
    }
  }

  Status PreProcess(uint64_t &launch_begin_time) override;
  Status ReportProfExtendInfo(const uint64_t end_time, const uint64_t op_name_hash, const int32_t tid) const override;
  void ResetDumperResource() override;
 protected:
  virtual Status DoLaunchKernel(rtStream_t const stream);

 private:
  NodePtr node_;
  // |ffts_addr|input addrs|output addrs|workspace addrs|tiling addr|overflow_addr|tiling data|host mem data|
  std::unique_ptr<uint8_t[]> args_;
  size_t arg_size_ = 0U;
  rtArgsEx_t args_ex_ = {};
  rtTaskCfgInfo_t cfg_ = {};
  std::unique_ptr<rtHostInputInfo_t[]> host_inputs_info_;
  ArgItemOffset args_item_offsets_;
  uint32_t arg_num_ = 0U;
  uint32_t max_tiling_size_ = 0U;
  size_t ffts_addr_num_{0UL};
  size_t input_num_ = 0U; // include const input
  size_t output_num_ = 0U;
  friend class SingleOpModel;
  friend class TbeTaskBuilder;
  friend class AtomicAddrCleanOpTask;
  friend class MixL2TaskBuilder;
  friend class MixL2OpTask;
  Status UpdateArgsItem(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status DoLaunchKernelWithArgsEx(rtStream_t const stream);
  Status CheckAndExecuteAtomic(const std::vector<GeTensorDesc> &input_desc,
                               const std::vector<DataBuffer> &input_buffers,
                               std::vector<GeTensorDesc> &output_desc,
                               std::vector<DataBuffer> &output_buffers,
                               rtStream_t const stream);
  virtual Status UpdateNodeByShape(const std::vector<GeTensorDesc> &input_desc,
                                   const std::vector<GeTensorDesc> &output_desc) const;
  virtual Status UpdateTilingArgs();
  virtual Status UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  virtual Status CalcTilingInfo();

  Status ExtendArgSizeIfNeed(size_t new_size);
  virtual void UpdateOverflowAddr() const;
  void UpdateWorkspaceArgs();
  const void *stub_func_ = nullptr;
  void *sm_desc_ = nullptr;
  std::string stub_name_;
  StreamResource *stream_resource_ = nullptr;

  std::vector<void *> workspaces_;
  fe::PlatFormInfos platform_infos_ = {};
  std::shared_ptr<gert::OpImplSpaceRegistryV2Array> space_registries_;

  uint64_t tiling_key_ = 0U;
  void* handle_ = nullptr;
  std::string node_info_;
  std::vector<size_t> arg_index_; // data index in args
  void *overflow_addr_ = nullptr;
  bool has_overflow_attr_ = false;
  std::unique_ptr<optiling::utils::OpRunInfo> run_info_;
  size_t tiling_data_idx_ = 0U;
  std::vector<uint64_t> l0_dump_list_;
};

class AtomicAddrCleanOpTask : public TbeOpTask {
 public:
  AtomicAddrCleanOpTask() = default;
  explicit AtomicAddrCleanOpTask(const NodePtr &node) : TbeOpTask(node) {}
  ~AtomicAddrCleanOpTask() noexcept override = default;
  Status InitAtomicAddrCleanIndices();
  void SetWorkSpaceAddr(const std::vector<void *> &workspaces) { workspaces_ = workspaces;}
  const std::string GetOpType() const override;
  bool IsSupportHostMemInputOptimize() const override { return false; }
 private:
  Status UpdateNodeByShape(const std::vector<GeTensorDesc> &input_desc,
                           const std::vector<GeTensorDesc> &output_desc) const override;
  Status UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) override;
  Status UpdateTilingArgs() override;
  void UpdateOverflowAddr() const override;
  Status CalcTilingInfo() override;

  std::vector<int32_t> atomic_output_indices_;
  std::vector<int32_t> atomic_workspace_indices_;
  std::vector<void *> workspaces_;
};

class AiCpuBaseTask : public OpTask {
 public:
  AiCpuBaseTask() = default;
  ~AiCpuBaseTask() noexcept override;
  UnknowShapeOpType GetUnknownType() const { return unknown_type_; }
  Status UpdateArgTable(const SingleOpModelParam &param) override;
  const std::string &GetTaskType() const override;
 protected:
  Status UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);
  Status SetInputConst();
  Status SetExtInfoAndType(const std::string &kernel_ext_info, const uint64_t kernel_id);

  Status UpdateExtInfo(const std::vector<GeTensorDesc> &input_desc,
                       const std::vector<GeTensorDesc> &output_desc,
                       rtStream_t const stream);
  Status UpdateOutputShape(std::vector<GeTensorDesc> &output_desc);
  Status UpdateShapeToOutputDesc(const GeShape &shape_new, GeTensorDesc &output_desc) const;
  Status UpdateShapeAndDataByResultSummary(std::vector<GeTensorDesc> &output_desc,
                                           std::vector<DataBuffer> &outputs,
                                           rtStream_t const stream);
  Status ReadResultSummaryAndPrepareMemory();

  Status PrepareCopyInputs(const std::vector<DataBuffer> &outputs);

  Status UpdateShapeByHbmBuffer(std::vector<GeTensorDesc> &output_desc);

  virtual Status CopyDataToHbm(std::vector<DataBuffer> &outputs, rtStream_t stream) = 0;
  // for blocking aicpu op
  Status DistributeWaitTaskForAicpuBlockingOp(rtStream_t const stream);
  Status UpdateEventIdForBlockingAicpuOp();
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const;

 private:
  AiCpuBaseTask(const AiCpuBaseTask &) = delete;
  AiCpuBaseTask &operator=(const AiCpuBaseTask &)& = delete;

  friend class AiCpuTaskBuilder;
  friend class AiCpuCCTaskBuilder;
  friend class AiCpuTask;
  friend class AiCpuCCTask;

  size_t num_inputs_ = 0U;
  size_t num_outputs_ = 0U;
  UnknowShapeOpType unknown_type_ = DEPEND_IN_SHAPE;
  std::unique_ptr<ge::hybrid::AicpuExtInfoHandler> aicpu_ext_handle_;
  void *ext_info_addr_dev_ = nullptr;
  std::vector<int8_t> input_is_const_; // 1 is const, 0 is not const
  // for blocking aicpu op
  bool is_blocking_aicpu_op_ = false;
  rtEvent_t rt_event_ = nullptr;
  std::vector<void *> output_summary_;
  std::vector<aicpu::FWKAdapter::ResultSummary> output_summary_host_;

  void *copy_input_release_flag_dev_ = nullptr;
  void *copy_input_data_size_dev_ = nullptr;
  void *copy_input_src_dev_ = nullptr;
  void *copy_input_dst_dev_ = nullptr;

  std::vector<void *> out_shape_hbm_;
  int32_t deploy_type_flag_{0};
  tagRtMemcpyKind memcpy_kind_{RT_MEMCPY_HOST_TO_DEVICE};
  rtMemType_t mem_type_{RT_MEMORY_HBM};
};

class AiCpuTask : public AiCpuBaseTask {
 public:
  AiCpuTask() = default;
  ~AiCpuTask() noexcept override;

  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;

  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers,
                      rtStream_t const stream) override;
  Status SetMemCopyTask(const domi::KernelExDef &kernel_def);
  Status UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) override;
  bool IsSupportHostMemInputOptimize() const override { return true; }
  size_t GetInputAddrAlignBytes() const override { return kAlignBytes64; }
  size_t GetHostMemInputDataOffsetInIoAddr() const override { return host_mem_input_data_offset_; }
  void GetHostArgsAndSize(uintptr_t &args, size_t &arg_size) override{
    args = reinterpret_cast<uintptr_t>(args_);
    arg_size = arg_size_;
  }
 private:
  // for copy task.
  Status InitForSummaryAndCopy();
  Status CopyDataToHbm(std::vector<DataBuffer> &outputs, rtStream_t const stream) override;

  friend class AiCpuTaskBuilder;
  void *workspace_addr_ = nullptr;
  std::string task_info_;
  // device addr
  void *args_ = nullptr;
  size_t arg_size_ = 0U;
  std::string op_type_;
  // device addr
  void *io_addr_ = nullptr;
  size_t io_addr_size_ = 0U;

  // host addr
  std::vector<void *> io_addr_host_;
  size_t host_mem_input_data_offset_ = 0U;

  // for copy task
  void *copy_task_args_buf_ = nullptr;
  void *copy_workspace_buf_ = nullptr;

  void *copy_ioaddr_dev_ = nullptr;

  uint64_t kernel_id_ = 0U;
};

class AiCpuCCTask : public AiCpuBaseTask {
 public:
  AiCpuCCTask() = default;
  ~AiCpuCCTask() noexcept override;
  AiCpuCCTask(const AiCpuCCTask &) = delete;
  AiCpuCCTask &operator=(const AiCpuCCTask &)& = delete;
  Status SetMemCopyTask(const domi::KernelDef &kernel_def);
  Status LaunchKernel(rtStream_t const stream) override;
  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers,
                      rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
  void SetKernelArgs(std::unique_ptr<uint8_t[]> args, const size_t arg_size);
  void SetSoName(const std::string &so_name);
  void SetkernelName(const std::string &kernel_Name);
  void SetIoAddr(uintptr_t *const io_addr);
  bool IsSupportHostMemInputOptimize() const override { return true; }
  size_t GetHostMemInputDataOffsetInIoAddr() const override { return host_mem_input_data_offset_; }
  Status UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) override;
 private:
  Status InitForSummaryAndCopy();
  Status CopyDataToHbm(std::vector<DataBuffer> &outputs, rtStream_t const stream) override;
  void GetHostArgsAndSize(uintptr_t &args, size_t &arg_size) override{
    args = reinterpret_cast<uintptr_t>(args_ex_.args);
    arg_size = args_ex_.argsSize;
  }
 private:
  friend class AiCpuCCTaskBuilder;
  std::string so_name_;
  std::string kernel_name_;
  std::unique_ptr<uint8_t[]> args_;
  std::unique_ptr<rtHostInputInfo_t[]> host_inputs_info_;
  rtArgsEx_t args_ex_ = {};
  size_t arg_size_ = 0U;
  void *sm_desc_ = nullptr;
  uintptr_t *io_addr_ = nullptr;
  size_t io_addr_num_ = 0U;
  size_t host_mem_input_data_offset_ = 0U;
  bool is_custom_ = false;
  uint32_t dump_flag_ = RT_KERNEL_DEFAULT;
  std::string op_type_;
  uint64_t kernel_id_ = 0U;
  // host memcpy mem
  std::unique_ptr<uint8_t[]> memcpy_args_;
  std::string memcpy_so_name_;
  std::string memcpy_kernel_name_;
  std::vector<uint64_t> copy_io_addr_;
  // args size
  uint32_t memcpy_args_size_ = 0U;
};

class MemcpyAsyncTask : public OpTask {
 public:
  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;

 private:
  friend class SingleOpModel;
  friend class RtsKernelTaskBuilder;

  std::vector<uintptr_t> addresses_ = {0U, 0U}; // src address and dst address
  size_t dst_max_;
  size_t count_;
  rtMemcpyKind_t kind_;
  rtTaskCfgInfo_t cfg_ = {};
  NodePtr node_;
};

class MixL2OpTask : public TbeOpTask {
 public:
  explicit MixL2OpTask(const NodePtr &node) : TbeOpTask(node) {}
  ~MixL2OpTask() noexcept override;
  Status LaunchKernel(rtStream_t const stream) override;
  Status LaunchKernel(const std::vector<GeTensorDesc> &input_desc, const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc, std::vector<DataBuffer> &output_buffers,
                      rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
  Status UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) override;

  Status UpdateIoAddr(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) override;
  void GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) override;
  void GetHostArgsAndSize(uintptr_t &args, size_t &arg_size) override;
  const std::string &GetTaskType() const override {
    if (ctx_type_ == RT_CTX_TYPE_MIX_AIC) {
      return kTaskTypeMixAic;
    } else if (ctx_type_ == RT_CTX_TYPE_MIX_AIV) {
      return kTaskTypeMixAiv;
    } else {
      return kTaskTypeInvalid;
    }
  }
  Status ReportProfExtendInfo(const uint64_t end_time, const uint64_t op_name_hash, const int32_t tid) const override;
  void SaveForL0ExceptionDump() override {};
  Status PreProcess(uint64_t &launch_begin_time) override;
 protected:
  Status DoLaunchKernel(rtStream_t const stream) override;

 private:
  friend class MixL2TaskBuilder;
  // |tiling data|host mem data|mode addrs|input addrs|output addrs|workspace addrs|tiling addr|
  std::vector<uintptr_t> host_args_; // host argtable
  void *device_args_{nullptr}; // device argtable
  size_t mode_addr_cnt_{0UL};
  size_t args_addr_base_idx_{0UL};  // base index for ioaddr
  size_t args_addr_cnt_{0UL}; // ioaddr&workspace cnts
  size_t host_mem_base_idx_{0UL};

  std::vector<uint64_t> io_addrs_from_taskdef_;
  std::set<size_t> mode_addr_idx_;
  std::vector<void *> ext_args_;
  rtFftsPlusTaskInfo_t ffts_plus_task_info_{};

  TBEKernelHandle bin_kernel_handle_;
  std::vector<std::string> names_prefix_;
  tagFftsPlusContextType ctx_type_{};
  std::vector<uint32_t> context_ids_{};
  std::vector<uint64_t> l0_dump_list_{};
};

class NpuGetFloatStatusTask : public OpTask {
 public:
  ~NpuGetFloatStatusTask() noexcept override;
  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
 private:
  friend class SingleOpModel;
  friend class RtsKernelTaskBuilder;

  uint32_t mode_{0U};
  void *args_{nullptr};
  size_t args_size_{0UL};
  uint8_t *output_addr_{nullptr};
  size_t output_size_{0UL};
};

class NpuClearFloatStatusTask : public OpTask {
 public:
  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override {
    arg_base = nullptr;
    arg_count = 0UL;
  }
 private:
  friend class SingleOpModel;
  friend class RtsKernelTaskBuilder;

  uint32_t mode_{0U};
};

class NpuGetFloatDebugStatusTask : public OpTask {
 public:
  ~NpuGetFloatDebugStatusTask() noexcept override;
  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
 private:
  friend class SingleOpModel;
  friend class RtsKernelTaskBuilder;

  uint32_t mode_{0U};
  void *args_{nullptr};
  size_t args_size_{0UL};
  uint8_t *output_addr_{nullptr};
  size_t output_size_{0UL};
};

class NpuClearFloatDebugStatusTask : public OpTask {
 public:
  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override {
    arg_base = nullptr;
    arg_count = 0UL;
  }
 private:
  friend class SingleOpModel;
  friend class RtsKernelTaskBuilder;

  uint32_t mode_{0U};
};

class DsaTask : public OpTask {
 public:
  Status LaunchKernel(rtStream_t const stream) override;
  void GetIoAddr(uintptr_t *&arg_base, size_t &arg_count) override;
  const std::string &GetTaskType() const override { return kTaskTypeDsa; }
 private:
  friend class DsaTaskBuilder;

  Status UpdateDsaSqe(rtStream_t const stream);
  std::vector<void *> io_addr_;
  rtStarsDsaSqe_t dsa_sqe_;
  size_t input_size_{0UL};
  size_t output_size_{0UL};
  size_t workspace_size_{0UL};
  uint32_t input1_value_or_ptr_{0U};
  uint32_t seed_value_or_ptr_{0U};
  uint32_t random_count_value_or_ptr_{0U};
  uint64_t input_data_[2] = {0U, 0U};
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_OP_TASK_H_
