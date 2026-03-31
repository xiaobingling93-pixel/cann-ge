/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_SUPER_KERNEL_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_SUPER_KERNEL_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/args_format_desc.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "graph/ge_context.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/types.h"
#include "framework/omg/parser/parser_types.h"
#include "register/op_tiling_registry.h"
#include "common/dump/kernel_tracing_utils.h"
#include "acl/acl_rt.h"

namespace ge {
class SuperKernelV2TaskInfo : public TaskInfo {
 public:
  SuperKernelV2TaskInfo() : TaskInfo() {}

  ~SuperKernelV2TaskInfo() override {
    davinci_model_ = nullptr;
    args_ = nullptr;
  }

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status Release() override;

  uint32_t GetTaskID() const override { return task_id_; }

  uint32_t GetStreamId() const override { return stream_id_; }

  uintptr_t GetDumpArgs() const override {
    return static_cast<uintptr_t>(PtrToValue(dump_args_));
  }

  std::map<uint64_t, uint64_t> GetCustToRelevantOffset() const override {
    return cust_to_relevant_offset_;
  }

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  void PostProcess(const domi::TaskDef &task_def) override;
  bool CallSaveDumpInfo() const override  { return call_save_dump_; }
 private:
  struct ArgsFormatInfo {
    std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
    std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
    std::vector<ArgDesc> arg_descs;
    std::vector<std::vector<int64_t>> shape_infos;
    size_t level1_addr_cnt{0UL};
    // tiling sink_addr
    std::vector<size_t> tiling_depends_input_idx;
    // ling sink tensor size
    size_t sink_tensor_size{0UL};
  };

  struct SubNodeIoIndex {
    size_t node_idx;
    size_t io_idx;
    bool is_input;

    bool operator<(const SubNodeIoIndex &sub_node_io_index) const{
      return ((node_idx < sub_node_io_index.node_idx) ||
        ((node_idx == sub_node_io_index.node_idx) && (io_idx < sub_node_io_index.io_idx)) ||
        ((node_idx == sub_node_io_index.node_idx) && (io_idx == sub_node_io_index.io_idx)
        && static_cast<int32_t>(is_input) <  static_cast<int32_t>(sub_node_io_index.is_input)));
    }
  };

  Status UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs);
  Status InitTask(const domi::KernelDef &kernel_def);
  Status InitContext(const domi::KernelContext &context);

  Status SetHcomAttr(const size_t node_idx);

  void UpdateTaskId();

  Status InitKernel(const domi::TaskDef &task_def, const PisToArgs &args);
  Status FindSkSubNode(const OpDescPtr &sk_op, const int32_t id,  NodePtr &sub_node) const;
  Status GenSubNodeIoToSuperKernelIoMap(size_t node_idx, const NodePtr &sub_node);
  Status ParseArgsFormat(const std::vector<ArgDesc> &args_descs);
  size_t GetArgsSizeByFormat() const;
  Status AssembleShapeInfoAddrs(const std::vector<std::vector<ArgDesc>> &sub_node_dynamic_args_desc,
                                const std::vector<std::vector<size_t>> &sub_node_level2_addr_idx);

  Status AssembleTilingContextArgs(int32_t node_idx,const ArgDesc &arg_desc,
                                  std::map<size_t, gert::AddrRefreshedTensor> &index_to_tensor);

  Status AssembleTilingSinkTensors(std::map<int32_t ,std::map<size_t, gert::AddrRefreshedTensor>> &index_to_tensor);
  void GetAddrAlignedGertTensorSize(size_t &io_aligned_offset, size_t &double_aliged_tensor_size) const;
  Status AssembleIoByArgsFormat();
  void AppendIoAddr(const uint64_t addr, const uint64_t addr_type);
  Status AppendWorkspaceAddr(size_t node_idx, int32_t ir_idx);
  Status AppendInputOutputAddrByInstanceIndex(size_t node_idx, size_t ins_idx, bool is_input);
  Status AppendInputOutputAddr(size_t node_idx, size_t ir_idx, bool is_input);
  void InsertL0DumpList(size_t node_idx, size_t io_idx, bool is_input);
  void InsertCustToRelevantOffset(size_t node_idx, size_t io_idx, bool is_input);
  aclrtFuncHandle GetFuncHandle();
  void *args_{nullptr};
  uint32_t block_dim_{0U};
  uint32_t args_size_{0U};
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  OpDescPtr op_desc_;   // Clear after distribute.
  std::vector<uint64_t> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  DavinciModel *davinci_model_{nullptr};
  size_t io_addr_offset_{0U};
  ModelTaskType task_type_ = ModelTaskType::MODEL_TASK_KERNEL;
  std::shared_ptr<Operator> operator_;
  ArgsIoAddrsUpdater args_io_addrs_updater_;
  ArgsPlacement args_placement_{ArgsPlacement::kArgsPlacementHbm};
  ArgsFormatInfo args_format_holder_;
  ccKernelType kernel_type_{ccKernelType::CCE_AI_CORE};
  uint32_t local_memory_size_ = 0U;  // for simt op
  uint8_t schedule_mode_{0U};
  uint32_t block_dim_offset_{0UL};

  // super kernel 特有的数据结构
  std::vector<int32_t> sub_node_op_index_list_;  // 和args format里subtask的顺序保持一致
  std::vector<OpDescPtr> sub_node_op_desc_list_;
  std::vector<ArgsFormatInfo> sub_node_args_format_holder_list_;
  std::vector<std::vector<uint64_t>> sub_node_input_addrs_list_;
  std::vector<std::vector<uint64_t>> sub_node_output_addrs_list_;
  std::vector<std::vector<uint64_t>> sub_node_workspace_addrs_list_;
  std::vector<std::vector<uint64_t>> sub_node_input_mem_types_list_;
  std::vector<std::vector<uint64_t>> sub_node_output_mem_types_list_;
  std::vector<std::vector<uint64_t>> sub_node_workspace_mem_types_list_;

  // dump相关
  uint32_t dump_flag_{RT_KERNEL_DEFAULT};
  void *dump_args_{nullptr};
  bool call_save_dump_ = false;
  std::map<SubNodeIoIndex, size_t> sub_node_io_idx_to_super_kernel_io_idx_;
  std::map<uint64_t, uint64_t> cust_to_relevant_offset_;
  std::vector<uint64_t> l0_dump_list_;
  aclrtFuncHandle func_handle_{nullptr};
  bool is_block_task_prefetch_{false};
  bool is_data_dump_{false};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_SUPER__KERNEL_TASK_INFO_H_
