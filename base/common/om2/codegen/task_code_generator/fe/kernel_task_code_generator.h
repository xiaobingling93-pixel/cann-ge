/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_RTS_KERNEL_TASK_CODE_GENERATOR_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_RTS_KERNEL_TASK_CODE_GENERATOR_H_

#include "common/om2/codegen/task_code_generator/task_code_generator.h"
#include "rts/rts_kernel.h"
#include "fwk_adpt_struct.h"
#include "ge/ge_error_codes.h"
#include "graph/utils/args_format_desc_utils.h"
#include "framework/common/taskdown_common.h"
#include "common/om2/codegen/om2_codegen_utils.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"

namespace ge {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
#define EMIT_CODE(ss, code) (ss << code << '\n')
constexpr int64_t kDimEndFlag = std::numeric_limits<int64_t>::min();
constexpr uint32_t kAddressLen = static_cast<uint32_t>(sizeof(uint64_t));

struct Om2LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  uint32_t local_memory_size{0U};  // ACL接口暂未支持
  std::string engine_type{"ACL_RT_ENGINE_TYPE_AIC"};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};  // 待补齐Data Dump能力后做
  uint16_t time_out{0U};
};

struct Om2LaunchKernelParam {
  uint32_t block_dim{0U};
  uint32_t stream_id{0U};
  Om2LaunchKernelConfig launch_config;
};

class KernelTaskCodeGenerator : public TaskCodeGenerator {
 public:
  KernelTaskCodeGenerator() : host_args_offset_(0UL), cust_value_var_index_(0), place_holder_var_index_(0) {}
  Status GenTaskDistributionCode(TaskDistributionContext &context) override;
  Status GenDistributionImplCode(TaskDistributionImplContext &context) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) override;

 private:
  struct ArgsFormatInfo {
    std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
    std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
    std::vector<ArgDesc> arg_descs;
    // header for shape infos
    std::vector<std::vector<int64_t>> shape_infos;
    size_t level1_addr_cnt{0UL};
  };
  Status AssembleAicpuArgsCode(TaskDistributionContext &context, const std::string iow_addr_var_name, const std::string args_var_name, std::string &aicpu_args_code);
  Status GenArgsCode(TaskDistributionContext &context);
  void AssembleLaunchKernelConfig(const OpDescPtr &op_desc, const domi::TaskDef &task_def,
                                  Om2LaunchKernelParam &launch_param);
  Status InitAicpuTaskExtInfo(uint8_t *ext_info, size_t ext_info_len, const OpDescPtr op_desc,
                              const domi::TaskDef &task_def, int32_t &session_info_offset);
  Status UpdateShapeAndType(const std::vector<int64_t> &dims, const DataType data_type,
                            AicpuShapeAndType &shape_and_type) const;
  Status UpdateShapeAndType(const GeShape &shape, const DataType data_type, AicpuShapeAndType *const shape_and_type);
  Status ParseArgsFormat(TaskDistributionContext &context, ArgsFormatInfo &args_format_holder);
  size_t GetArgsSizeByFormat(const OpDescPtr op_desc, ArgsFormatInfo &args_format_holder) const;
  size_t GetExtraArgsSize(const OpDescPtr &op_desc, const ccKernelType kernel_type, ArgsFormatInfo &args_format_holder);
  Status GenInputOutputAddrByInstanceIndex(TaskDistributionContext &context, size_t inst_idx, bool is_input);
  Status GenWorkspaceAddr(TaskDistributionContext &context, int32_t ir_idx);
  Status GenInputOutputAddr(TaskDistributionContext &context, ArgsFormatInfo &args_format_holder, size_t ir_idx, bool is_input);
  Status GenArgsByArgsFormat(TaskDistributionContext &context, ArgsFormatInfo &args_format_holder);
  void GenCustomValue(TaskDistributionContext &context, const uint64_t custom_value);
  void AppendPlaceholder(TaskDistributionContext &context);
  Status AssembleShapeInfoAddrs(TaskDistributionContext &context, const std::vector<ArgDesc> &dynamic_args_desc,
                                const std::vector<size_t> &level2_addr_idx, ArgsFormatInfo &args_format_holder);
  void AppendIoAddrNodes(TaskDistributionContext &context, const AddrGenInfo &src, uint64_t args_len = kAddressLen);
  Status CheckTaskSupport(TaskDistributionContext &context);
  Status GetKernelTaskMeta(const domi::TaskDef &task_def, domi::KernelContext &kernel_context,
                           uint32_t &args_size, uint32_t &kernel_type) const;
  std::string EmitLaunchConfigSetupCode(size_t op_index, const Om2LaunchKernelConfig &launch_config);
  std::string SerializeBytesToOctalString(const std::vector<uint8_t> &buffer);
 private:
  uint64_t host_args_offset_;
  std::vector<AddrGenInfo> input_addr_nodes_;
  std::vector<AddrGenInfo> output_addr_nodes_;
  std::vector<AddrGenInfo> workspace_addr_nodes_;
  std::vector<AddrGenInfo> args_addr_nodes_;
  int32_t cust_value_var_index_;
  int32_t place_holder_var_index_;
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_RTS_KERNEL_TASK_CODE_GENERATOR_H_
