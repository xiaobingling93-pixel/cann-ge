/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/task/aicpu_task_builder.h"
#include <vector>
#include "single_op/task/build_task_utils.h"
#include "runtime/mem.h"
#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/model_manager.h"

namespace ge {
AiCpuTaskBuilder::AiCpuTaskBuilder(const OpDescPtr &op_desc, const domi::KernelExDef &kernel_def)
    : op_desc_(op_desc), kernel_def_(kernel_def) {}

Status AiCpuTaskBuilder::SetFmkOpKernel(const void *const io_addr, const void *const ws_addr,
                                        STR_FWK_OP_KERNEL &fwk_op_kernel) const {
  const auto sec_ret = memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL),
                                kernel_def_.args().data(), kernel_def_.args().size());
  GE_CHK_BOOL_RET_STATUS(sec_ret == EOK, ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                         "[Memcpy_s][Param:fwk_op_kernel] failed, ret: %d", sec_ret);

  const uint64_t io_addr_val = PtrToValue(io_addr);
  fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = io_addr_val;
  const uint64_t ws_addr_val = PtrToValue(ws_addr);
  fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr = ws_addr_val;
  return SUCCESS;
}

Status AiCpuTaskBuilder::InitWorkspaceAndIO(AiCpuTask &task, const SingleOpModelParam &param) const {
  GE_CHECK_GE(kernel_def_.task_info().size(), kernel_def_.task_info_size());
  GE_CHK_RT_RET(rtMalloc(&task.workspace_addr_, static_cast<uint64_t>(kernel_def_.task_info_size()), task.mem_type_,
                         GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(aclrtMemcpy(task.workspace_addr_, static_cast<uint64_t>(kernel_def_.task_info_size()),
      kernel_def_.task_info().data(), static_cast<uint64_t>(kernel_def_.task_info_size()),
      task.memcpy_kind_));

  const auto addresses = BuildTaskUtils::GetAddresses(op_desc_, param, false);
  task.io_addr_host_ = BuildTaskUtils::JoinAddresses(addresses);
  if (task.extend_args_for_host_input_) {
    // input address of tf kernel cpu must be aligned to 64B
    GE_CHK_STATUS_RET(CheckUint32AddOverflow(static_cast<uint32_t>(task.io_addr_host_.size() * sizeof(uint64_t)),
                                             (static_cast<uint32_t>(kAlignBytes64) - 1U)),
                      "Padding size is beyond the UINT32_MAX.");
    task.host_mem_input_data_offset_ = (((task.io_addr_host_.size() * sizeof(void *)) + kAlignBytes64 - 1U) /
                                        kAlignBytes64) * kAlignBytes64;
    const size_t extend_len = (task.host_mem_input_data_offset_ - (task.io_addr_host_.size() * sizeof(void *))) +
                              kMaxHostMemInputLen;
    const size_t io_addr_host_size = task.io_addr_host_.size() + (extend_len / sizeof(void *));
    task.io_addr_host_.resize(io_addr_host_size, nullptr);
    GELOGD("%s has host memory input, io addr host is extended %zu, length = %zu, host_mem_input_data_offset = %zu.",
           op_desc_->GetName().c_str(), extend_len, task.io_addr_host_.size() * sizeof(void *),
           task.host_mem_input_data_offset_);
  }
  task.io_addr_size_ = task.io_addr_host_.size() * sizeof(void *);
  GE_CHK_RT_RET(rtMalloc(&task.io_addr_, task.io_addr_size_, task.mem_type_, GE_MODULE_NAME_U16));
  return SUCCESS;
}

Status AiCpuTaskBuilder::BuildTask(ge::AiCpuTask &task, const SingleOpModelParam &param,
                                   const uint64_t kernel_id) const {
  GE_CHECK_NOTNULL(op_desc_);
  task.op_desc_ = op_desc_;
  task.num_inputs_ = op_desc_->GetInputsSize();
  task.num_outputs_ = op_desc_->GetOutputsSize();

  // get kernel_ext_info
  auto &kernel_ext_info = kernel_def_.kernel_ext_info();
  const auto kernel_ext_info_size = kernel_def_.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS((kernel_ext_info.size() == kernel_ext_info_size), ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Size]task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.",
                         kernel_ext_info.size(), kernel_ext_info_size);
  GE_CHK_STATUS_RET(task.SetExtInfoAndType(kernel_ext_info, kernel_id), "[Set][ExtInfoAndType]failed.");
  GE_CHK_STATUS_RET_NOLOG(InitWorkspaceAndIO(task, param));
  STR_FWK_OP_KERNEL fwk_op_kernel = {};
  GE_CHK_STATUS_RET_NOLOG(SetFmkOpKernel(task.io_addr_, task.workspace_addr_, fwk_op_kernel));
  if (task.ext_info_addr_dev_ != nullptr) {
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = PtrToValue(task.ext_info_addr_dev_);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = kernel_ext_info_size;
  }
  GE_CHK_STATUS_RET(task.SetInputConst(), "[Set][InputConst] failed.");
  GE_CHK_STATUS_RET(task.InitForSummaryAndCopy(), "[Init][SummaryAndCopy] failed.");

  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = std::numeric_limits<uint64_t>::max();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = kernel_id;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.opType = aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_RUN_NO_SESS;
  GE_CHK_RT_RET(rtMalloc(&task.args_, sizeof(STR_FWK_OP_KERNEL), task.mem_type_, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(aclrtMemcpy(task.args_, sizeof(STR_FWK_OP_KERNEL),
      &fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), task.memcpy_kind_));

  task.arg_size_ = sizeof(STR_FWK_OP_KERNEL);
  task.op_type_ = op_desc_->GetName();
  task.task_info_ = kernel_def_.task_info();
  task.kernel_id_ = kernel_id;

  GELOGI("[TASK_INFO] %" PRIu64 "/%s %s",
    kernel_id, task.op_type_.c_str(), BuildTaskUtils::GetTaskInfo(op_desc_).c_str());
  return SUCCESS;
}
}  // namespace ge
