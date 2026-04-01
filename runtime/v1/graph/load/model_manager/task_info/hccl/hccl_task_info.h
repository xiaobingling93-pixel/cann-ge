/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "common/opskernel/ge_task_info.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/manager/util/hcom_ome_util.h"
#include "common/dump/dump_op.h"

namespace ge {
class HcclTaskInfo : public TaskInfo {
 public:
  HcclTaskInfo() = default;

  ~HcclTaskInfo() override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  uint32_t GetTaskID() const override { return id_; }

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        void *const host_args, const size_t host_args_max_len) override;
  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  Status UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) override;

  Status Release() override;
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

  Status GetTaskIowPaRemapInfos(std::vector<IowPaRemapInfo> &infos) override;
 private:
  void UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs);

  Status GetTypeSizeByDataType(const ge::DataType data_type, int64_t &type_size) const;

  Status InsertDumpOp(const std::string &dump_mode);

  Status SetAddrs(const std::shared_ptr<OpDesc> &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  void TransToGETaskInfo(GETaskInfo &ge_task) const;

  void PostProcess(const domi::TaskDef &task_def) override;

  void GetPrivateDefByTaskDef(const OpDescPtr &op_desc, const domi::TaskDef &task);

  Status CreateStream(const int64_t stream_num, const int64_t main_stream_id);

  Status SetFollowStream(const ConstOpDescPtr &op_desc);

  void CreateKernelHcclInfo(const ConstOpDescPtr &op_desc);

  Status SetWorkspace(const std::shared_ptr<OpDesc> &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  Status SetOverflowAddrs(const std::shared_ptr<OpDesc> &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  bool IsReduceOp(const std::string &hccl_type) const;

  bool UpdateOutputAddr(const std::string &hccl_type) const;

  Status InitZeroCopyInfos(const OpDescPtr &op_desc, const domi::KernelHcclDef &hccl_def);

  Status SetZeroCopyAddrs(const std::shared_ptr<OpDesc> &op_desc, std::vector<GETaskKernelHcclInfo> &kernel_hccl_infos);

  bool IsFeatureBaseRefreshable(const ge::DavinciModel *const davinci_model) const;
  Status AssembleAttachedRtStream(GETaskInfo &ge_task_info) const;
  void HcclWatcherModeProcess(const ModelTaskType task_type);
  DavinciModel *davinci_model_{nullptr};
  uint32_t id_{0U};
  uint32_t logic_stream_id_{0U};
  DumpOp input_hccl_dump_;
  DumpOp output_hccl_dump_;
  std::vector<aclrtStream> hccl_stream_list_;
  OpsKernelInfoStore *ops_kernel_store_{nullptr};
  void *private_def_{nullptr};
  uint32_t private_def_len_{0U};
  static std::mutex hccl_follow_stream_mutex_;
  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos_;
  std::vector<HcclDumpInfo> hccl_dump_infos_;
  std::vector<uint64_t> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  void *args_{nullptr};
  std::vector<void *> global_workspace_addr_;
  std::vector<int32_t> input_zero_copy_flag_;
  std::vector<int32_t> output_zero_copy_flag_;
  OpDescPtr hccl_op_desc_{nullptr};
  bool support_zero_copy_{false};
  ArgsIoAddrsUpdater args_io_addrs_updater_;
  std::vector<void *> input_data_addrs_;
  std::vector<void *> output_data_addrs_;
  std::vector<void *> workspace_addrs_;
  std::vector<uint64_t> input_mem_types_;
  std::vector<uint64_t> output_mem_types_;
  std::vector<uint64_t> workspace_mem_types_;
  uint32_t args_mem_type_{0U};
  uint32_t stream_flag_{0U};
  bool is_refresh_addr_op_{false};
  ArgsPlacement pls_{ArgsPlacement::kArgsPlacementHbm};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_HCCL_TASK_INFO_H_
