/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_EX_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_EX_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "acl/acl_rt.h"

namespace ge {
class KernelExTaskInfo : public TaskInfo {
 public:
  KernelExTaskInfo() = default;

  ~KernelExTaskInfo() override = default;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status Release() override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        void *const host_args, const size_t host_args_max_len) override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;
  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  Status UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) override;

  uint32_t GetTaskID() const override { return task_id_; }

  uint32_t GetStreamId() const override { return stream_id_; }

  uintptr_t GetDumpArgs() const override {
    return static_cast<uintptr_t>(PtrToValue(dump_args_));
  }

  uintptr_t GetArgs() const override {
    return reinterpret_cast<uintptr_t>(input_output_addr_);
  }

  size_t GetArgSize() const override {
    return addrs_size_;
  }

  bool CallSaveDumpInfo() const override {
    return true;
  }

  void PostProcess(const domi::TaskDef &task_def) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  Status AssembleWorkSpaceAddr(const domi::KernelExDef &kernel_def, const RuntimeParam &rts_param,
                               const OpDescPtr &op_desc);
  void InitDumpFlag(const OpDescPtr &op_desc);
  void InitDumpArgs(void *const addr, const OpDescPtr &op_desc);
  bool NeedUpdateAddr(const OpDescPtr &op_desc) const;
  Status InitTaskExtInfo(const std::string &ext_info, const OpDescPtr &op_desc);

  // for blocking aicpu op
  Status DistributeWaitTaskForAicpuBlockingOp() const;
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const;
  Status UpdateEventIdForAicpuBlockingOp(const OpDescPtr &op_desc, const hybrid::AicpuExtInfoHandler &ext_handle) const;
  Status InitKernelBufferAddr();
  Status AssembleKernelBuffer(const STR_FWK_OP_KERNEL * const fwk_op_kernel) const;
  Status InitInputOutputAddr(const PisToArgs &args, const IowAddrs &iow_addrs);
  Status AssembleInputOutputAddr();
  aclrtFuncHandle GetFuncHandle();
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  uint32_t kernel_buf_size_{0U};
  DavinciModel *davinci_model_{nullptr};
  OpDescPtr op_desc_;
  void *kernel_buf_{nullptr};
  void *input_output_addr_{nullptr};
  void *ext_info_addr_{nullptr};
  void *dump_args_{nullptr};
  std::vector<void *> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  int32_t deploy_type_flag_{0};
  tagRtMemcpyKind memcpy_kind_{RT_MEMCPY_HOST_TO_DEVICE};
  rtMemType_t mem_type_{RT_MEMORY_HBM};
  bool is_blocking_aicpu_op_{false};
  std::vector<void *> ext_args_;
  bool own_args_memory_{false};
  size_t addrs_size_{0};
  std::vector<void *> input_data_addrs_;
  std::vector<void *> output_data_addrs_;
  std::vector<uint64_t> input_addr_mem_types_;
  std::vector<uint64_t> output_addr_mem_types_;
  std::vector<void *> workspace_data_addrs_;
  ArgsIoAddrsUpdater args_io_addrs_updater_;
  ArgsPlacement pls_{ArgsPlacement::kArgsPlacementHbm};
  aclrtFuncHandle func_handle_{nullptr};
  bool is_data_dump_{false};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_KERNEL_EX_TASK_INFO_H_
