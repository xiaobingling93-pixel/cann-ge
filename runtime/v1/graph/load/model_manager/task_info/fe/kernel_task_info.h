/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/args_format_desc.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "framework/omg/parser/parser_types.h"
#include "framework/common/types.h"
#include "register/op_tiling_registry.h"
#include "common/dump/kernel_tracing_utils.h"

namespace ge {
struct ArgsFormatInfo {
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  std::vector<ArgDesc> arg_descs;
  // header for shape infos
  std::vector<std::vector<int64_t>> shape_infos;
  size_t level1_addr_cnt{0UL};
  // tiling sink_addr
  std::vector<size_t> tiling_depends_input_idx;
  // ling sink tensor size
  size_t sink_tensor_size{0UL};
};

class KernelTaskInfo : public TaskInfo {
 public:
  KernelTaskInfo() : TaskInfo(), ctx_(), custom_info_() {}

  ~KernelTaskInfo() override {
    davinci_model_ = nullptr;
    stub_func_ = nullptr;
    args_ = nullptr;
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        void *const host_args,
                        const size_t host_args_max_len) override;

  void UpdateAtomicCleanArgs(std::vector<uint64_t> &input_data_addrs,
                             std::vector<uint64_t> &output_data_addrs,
                             std::vector<uint64_t> &workspace_data_addrs) const;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;
  Status CopyTilingDataIfNeeded();

  Status Release() override;

  const std::vector<FusionOpInfo> &GetAllFusionOpInfo() const override { return fusion_op_info_; }

  uint32_t GetTaskID() const override { return task_id_; }

  uint32_t GetStreamId() const override { return stream_id_; }

  uintptr_t GetDumpArgs() const override {
    return static_cast<uintptr_t>(PtrToValue(dump_args_));
  }

  uintptr_t GetArgs() const override {
    return static_cast<uintptr_t>(PtrToValue(args_) + io_addr_offset_);
  }

  size_t GetArgSize() const override {
    auto argsSize = customized_args_info_.customized_aligned ? customized_args_info_.kernel_def_args_size : args_size_;
    if (argsSize > io_addr_offset_) {
      return static_cast<size_t>(argsSize) - io_addr_offset_;
    } else {
      return 0U;
    }
  }

  void GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) const override {
    if (!has_tiling_) {
      return;
    }
    tiling_key = static_cast<uint32_t>(tiling_key_);
    const auto tiling_data_holder = MakeUnique<uint8_t[]>(static_cast<size_t>(tiling_data_size_));
    GE_CHECK_NOTNULL_JUST_RETURN(tiling_data_holder);
    if (aclrtMemcpy(tiling_data_holder.get(), static_cast<uint64_t>(tiling_data_size_), tiling_data_addr_,
        static_cast<uint64_t>(tiling_data_size_), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      return;
    }
    std::stringstream ss;
    gert::PrintHex(tiling_data_holder.get(), static_cast<size_t>(tiling_data_size_), ss);
    tiling_data = ss.str();
  }

  void ResetArgsEx() {
    args_ex_ = rtArgsEx_t{};
  }
  void PostProcess(const domi::TaskDef &task_def) override;

  bool CallSaveDumpInfo() const override  { return call_save_dump_; }

  bool IsAtomicCleanTask() const {
    return (op_desc_->GetType() == ATOMICADDRCLEAN) ||
           (op_desc_->GetType() == MEMSET) ||
           is_separately_clean_task_;
  }

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

  std::map<uint64_t, uint64_t> GetCustToRelevantOffset() const override {
    return cust_to_relevant_offset_;
  }

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  Status UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) override;

 private:
  struct tagOpContext {
   private:
    friend class KernelTaskInfo;
    uint32_t opIndex = 0U;
    std::vector<uint16_t> argsOffset;
  };
  tagOpContext ctx_;

  std::vector<FusionOpInfo> fusion_op_info_;
  void UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs);
  Status InitTVMTask(const domi::KernelDef &kernel_def);
  Status InitTVMTask(const domi::KernelDefWithHandle &kernel_def);
  Status InitTVMTask();
  bool HasOverflowAddr(const OpDescPtr &op_desc) const;
  Status InitArgsAddr(const std::vector<uint64_t> &tensor_device_addrs, uint8_t *io_addr,
                      std::vector<uint64_t> &io_addr_mem_types, const size_t args_size);

  Status SetTvmTaskZeroCopy(const OpDescPtr &op_desc, const std::vector<uint64_t> &virtual_io_addrs);

  Status InitAICPUCustomTask(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def);

  Status InitAicpuTask(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def);

  Status InitAicpuKfcTask(const domi::KernelDef &kernel_def);

  Status InitTVMContext(const domi::KernelContext &context);

  bool SetHasMemoryLog();
  Status InitAicpuTaskExtInfo(const std::string &ext_info);
  Status UpdateExtraInfo(const hybrid::AicpuExtInfoHandler &ext_handle);

  Status StoreInputOutputTensor(const std::vector<uint64_t> &input_data_addrs,
                                const std::vector<uint64_t> &output_data_addrs,
                                const std::vector<ccAICPUTensor> &input_descs,
                                const std::vector<ccAICPUTensor> &output_descs);

  bool IsL1OrUBFusionOp(const OpDescPtr &op_desc) const;
  Status SetIoAddrs();
  void GetAtomicOutAddrs(const std::vector<uint64_t> &output_data_addrs,
                         std::vector<uint64_t> &atomic_output_data_addrs) const;
  void GetAtomicOutAddrs(const std::vector<uint64_t> &output_data_addrs,
                         const std::vector<uint64_t> &output_addr_mem_types,
                         std::vector<uint64_t> &atomic_output_data_addrs,
                         std::vector<uint64_t> &atomic_output_addr_mem_types) const;
  void GetAtomicWorkspaceAddrs(const std::vector<uint64_t> &workspace_data_addrs,
                               std::vector<uint64_t> &atomic_workspace_data_addrs) const;

  void GetAtomicWorkspaceAddrs(const std::vector<uint64_t> &workspace_data_addrs,
                               const std::vector<uint64_t> &workspace_addr_types,
                               std::vector<uint64_t> &atomic_workspace_data_addrs,
                               std::vector<uint64_t> &atomic_workspace_addr_types) const;

  Status AssembleArgs(const std::vector<uint64_t> &io_addrs);
  Status AssembleKernelNamesAndLaunch();
  void InitFusionDumpInfo(const OpDescPtr &op_desc, const domi::TaskDef &task_def);
  void InitDumpArgs(const size_t offset);

  Status GetNoncontinuousArgsRefreshInfo(std::vector<TaskArgsRefreshInfo> &infos);

  Status GetcontinuousArgsRefreshInfo(std::vector<TaskArgsRefreshInfo> &infos);

  Status UpdateNoncontinuousArgs(const size_t offset, const std::vector<uint64_t> &active_mem_base_addr,
                                 void *const host_args, const size_t host_args_len);
  Status UpdateContinuousArgs(const std::vector<uint64_t> &active_mem_base_addr,
                              void *const host_args,
                              const size_t host_args_len);

  Status InitPreprocessTask(const OpDescPtr &op_desc);
  void UpdateTaskId();

  // for blocking aicpu op
  Status DistributeWaitTaskForAicpuBlockingOp() const;
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const;
  Status UpdateEventIdForAicpuBlockingOp(const hybrid::AicpuExtInfoHandler &ext_handle) const;

  // for dynamic kernel
  Status InitKernel(const domi::TaskDef &task_def, const PisToArgs &args);
  Status InitKernelWithHandle(const domi::TaskDef &task_def, const PisToArgs &args);
  Status InitKernelByContext(const domi::TaskDef &task_def, const domi::KernelContext &context,
                             const PisToArgs &args);
  Status UpdateRunInfoByTilingResult(const optiling::utils::OpRunInfo *const run_info);
  size_t GetExtraArgsSize(const DavinciModel &davinci_model, const OpDescPtr &op_desc, const ccKernelType kernel_type);
  Status UpdateArgsSizeWithCustomized(const OpDescPtr &op_desc);
  Status ParseAicpuExtInfoHandler(const OpDescPtr &op_desc, const std::string &ext_info,
                                  std::unique_ptr<hybrid::AicpuExtInfoHandler> &ex_handle) const;
  Status SetIoAddrsForCustomized();
  Status ParseArgsFormat(uint32_t op_index, DavinciModel *const davinci_model);
  size_t GetArgsSizeByFormat() const;
  Status AssembleShapeInfoAddrs(const std::vector<ArgDesc> &dynamic_args_desc,
                                const std::vector<size_t> &level2_addr_idx);
  Status AssembleIoByArgsFormat();
  Status SaveL0DumpListWithArgsFormat();
  Status GetTilingSinkAtomicIndex(bool &is_args_exception_enable, uint64_t &atomic_index);
  Status AssembleTilingContextArgs(const ArgDesc &arg_desc,
                                   std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor);
  Status AssembleTilingSinkTensors(std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor);
  void SaveL0DumpList(const size_t io_addr_size);
  void AppendIoAddr(const uint64_t addr, const uint64_t addr_type);
  Status AppendWorkspaceAddr(int32_t ir_idx);
  Status AppendInputOutputAddrByInstanceIndex(size_t ins_idx, bool is_input);
  Status AppendInputOutputAddr(size_t ir_idx, bool is_input);
  Status PreprocessForSkNode();
  Status FindSkSubNode(const OpDescPtr &sk_op, const int32_t id,  NodePtr &sub_node) const;
  rtBinHandle GetBinHandle(const domi::TaskDef &task_def) const;
  rtFuncHandle GetFuncHandle(const domi::TaskDef &task_def);
  void SetExceptionCallback(rtBinHandle bin_handle);
  Status DistributeTask();
  rtArgsEx_t args_ex_{};
  rtAicpuArgsEx_t aicpu_args_ex_{};
  const void *stub_func_{nullptr};
  void *args_{nullptr};
  uint32_t block_dim_{0U};
  uint32_t args_size_{0U};
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  std::string so_name_;
  std::string kernel_name_;
  ccKernelType kernel_type_{ccKernelType::CCE_AI_CORE};
  bool has_memory_log_ = false;
  uint32_t dump_flag_{RT_KERNEL_DEFAULT};
  void *dump_args_{nullptr};
  OpDescPtr super_kernel_op_desc_;   // Clear after distribute.
  OpDescPtr op_desc_;   // Clear after distribute.
  std::shared_ptr<Operator> operator_;
  std::shared_ptr<Operator> sk_sub_operator_;
  std::vector<uint64_t> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  DavinciModel *davinci_model_{nullptr};
  std::vector<uint8_t> args_addr_;
  size_t io_addr_offset_{0U};
  bool call_save_dump_ = false;
  int32_t deploy_type_flag_{0};
  uint32_t qos_level_flag_{0U};
  rtTaskCfgInfo_t cfg_ = {};

  // aicpu ext_info device mem
  void *aicpu_ext_info_addr_ = nullptr;
  void *launch_addr_ = nullptr;
  void *kernel_name_arg_ = nullptr;

  bool is_blocking_aicpu_op_ = false;
  bool own_args_memory_ = false;
  bool is_separately_clean_task_ = false;
  bool is_addrs_folded_ = false;

  // for dynamic kernel
  uint64_t tiling_key_ = 0U;
  void *tiling_data_addr_{nullptr};
  size_t tiling_data_size_ = 0UL;
  bool has_tiling_{false};
  std::string node_info_;
  void *handle_ = nullptr;
  ModelTaskType task_type_ = ModelTaskType::MODEL_TASK_KERNEL;
  uint32_t op_index_ = 0U;
  bool clear_atomic_ = false;
  bool is_soft_sync_op_ = false;
  uint32_t local_memory_size_ = 0U;  // for simt op
  bool is_block_task_prefetch_{false};
  bool is_data_dump_{false};
  struct AICPUCustomInfo {
   private:
    friend class KernelTaskInfo;
    void *input_descs = nullptr;
    void *input_addrs = nullptr;
    void *output_descs = nullptr;
    void *output_addrs = nullptr;
    void *attr_handle = nullptr;
  };
  AICPUCustomInfo custom_info_;
  ArgsIoAddrsUpdater args_io_addrs_updater_;
  std::vector<uint64_t> input_data_addrs_;
  std::vector<uint64_t> output_data_addrs_;
  std::vector<uint64_t> workspace_addrs_;
  std::vector<uint64_t> input_mem_types_;
  std::vector<uint64_t> output_mem_types_;
  std::vector<uint64_t> workspace_mem_types_;
  struct CustomizedKernelInfo {
    uint32_t kernel_def_args_size{0};
    uint32_t input_addr_size{0};
    uint32_t input_addr_offset{0};
    uint32_t output_addr_size{0};
    uint32_t output_addr_offset{0};
    bool customized_aligned{false};
  };
  CustomizedKernelInfo customized_args_info_;
  ArgsPlacement args_placement_{ArgsPlacement::kArgsPlacementHbm};

  ArgsFormatInfo args_format_holder_;
  std::map<uint64_t, uint64_t> cust_to_relevant_offset_;
  std::vector<uint64_t> l0_dump_list_;
  int64_t args_offset_from_pls_{0};
  rtFuncHandle func_handle_{nullptr};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_TASK_INFO_H_
