/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_DUMP_OP_H_
#define GE_COMMON_DUMP_DUMP_OP_H_

#include <string>

#include "graph/op_desc.h"
#include "common/dump/dump_properties.h"
#include "proto/op_mapping.pb.h"
#include "runtime/stream.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"

namespace ge {
struct RealAddressAndSize {
  uint64_t address;
  uint64_t size;
};

struct Context {
  uint32_t context_id;
  uint32_t thread_id;
  std::vector<RealAddressAndSize> input;
  std::vector<RealAddressAndSize> output;
};

struct FftsPlusDumpInfo {
  std::shared_ptr<OpDesc> op;
  std::vector<Context> context;
};

class DumpOp {
 public:
  DumpOp() = default;
  ~DumpOp();

  void SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc,
                   const std::vector<uintptr_t> &input_addrs, const std::vector<uintptr_t> &output_addrs,
                   rtStream_t const stream);
  Status LaunchDumpOp(const bool is_single_op_dump, bool need_device_args = false);
  void SetLoopAddr(const uintptr_t global_step, const uintptr_t loop_per_iter, const uintptr_t loop_cond);
  void SetDynamicModelInfo(const std::string &dynamic_model_name, const std::string &dynamic_om_name,
                           const uint32_t dynamic_model_id);
  void SaveFftsSubOpInfo(const OpDescPtr &op_desc, const std::vector<Context> &context);
  Status GenerateFftsDump(const DumpProperties &dump_properties, void *&load_dump_info, uint32_t &load_dump_len,
                          void *&unload_dump_info, uint32_t &unload_dump_len, const bool is_single_op_dump);
  bool IsFftsDumpInfoEmpty() const { return ffts_sub_op_list_.empty(); }
  void SetTaskId(const uint32_t task_id) {
    task_id_ = task_id;
  }
  void SetWorkspaceAddrs(const std::vector<std::pair<uintptr_t, int64_t>> &workspace_addr) {
    space_addrs_.assign(workspace_addr.cbegin(), workspace_addr.cend());
  }
  void SetStreamId(const uint32_t stream_id) {
    stream_id_ = stream_id;
  }
  Status UpdateAddrs(const std::vector<uintptr_t> &input_addrs,
                     const std::vector<uintptr_t> &output_addrs);
 private:
  Status ExecutorDumpOp(bool need_device_args);
  void DumpWorkspace(toolkit::aicpu::dump::Task &task);
  Status DumpOutput(toolkit::aicpu::dump::Task &task, const OpDescPtr &op_desc,
                    const std::vector<uintptr_t> &addrs, bool ffts_flag = false) const;
  Status DumpInput(toolkit::aicpu::dump::Task &task, const OpDescPtr &op_desc,
                   const std::vector<uintptr_t> &addrs, bool ffts_flag = false) const;
  void DumpTask(toolkit::aicpu::dump::Task &task, const uint32_t task_id);
  Status SetDumpModelName();
  Status ProtoMallocAndMemcpy(const size_t proto_size, const std::string &proto_msg);
  Status LaunchDump(toolkit::aicpu::dump::Task &task);
  Status BuildFftsSubOpTask(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  Status BuildUnLoadFftsDumpInfo(void *&unload_dump_info, uint32_t &unload_dump_len);
  toolkit::aicpu::dump::AddressType GetAddrType(const toolkit::aicpu::dump::Task &task, const GeTensorDesc &desc) const;
  Status ExecuteDump(toolkit::aicpu::dump::Task &task, bool need_device_args);

  DumpProperties dump_properties_;
  OpDescPtr op_desc_;
  std::vector<uintptr_t> input_addrs_;
  std::vector<uintptr_t> output_addrs_;
  std::vector<std::pair<uintptr_t, int64_t>> space_addrs_;
  std::vector<FftsPlusDumpInfo> ffts_sub_op_list_;

  void *proto_dev_mem_ = nullptr;
  void *proto_size_dev_mem_ = nullptr;
  void *dev_mem_unload_{nullptr};
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info_;
  rtStream_t stream_;
  uintptr_t global_step_ = 0U;
  uintptr_t loop_per_iter_ = 0U;
  uintptr_t loop_cond_ = 0U;
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  std::string dynamic_model_name_;
  std::string dynamic_om_name_;
  std::uint32_t dynamic_model_id_;
  void *launch_kernel_args_dev_mem_ = nullptr;
};
}  // namespace ge

#endif  // GE_COMMON_DUMP_DUMP_OP_H_
