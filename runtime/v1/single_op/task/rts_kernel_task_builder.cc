/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/task/rts_kernel_task_builder.h"
#include "graph/def_types.h"
#include "single_op/task/build_task_utils.h"
#include "common/plugin/ge_make_unique_util.h"

namespace ge {
namespace {
constexpr size_t kNumAddresses = 2UL;
}  // namespace

void RtsKernelTaskBuilder::UpdateCopyKind(const OpDescPtr &op_desc, rtMemcpyKind_t &kind) {
  std::vector<int64_t> v_output_memory_type;
  std::vector<int64_t> v_input_memory_type;
  const bool has_output_mem_type_attr =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_output_memory_type);
  const bool has_input_mem_type_attr =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_input_memory_type);
  const bool is_output_svm =
      (v_output_memory_type.size() == 1U) && (v_output_memory_type[0U] == static_cast<int64_t>(RT_MEMORY_HOST_SVM));
  const bool is_input_svm =
      (v_input_memory_type.size() == 1U) && (v_input_memory_type[0U] == static_cast<int64_t>(RT_MEMORY_HOST_SVM));

  if (is_output_svm && (!has_input_mem_type_attr)) {
    kind = RT_MEMCPY_DEVICE_TO_HOST;
  } else if (is_input_svm && (!has_output_mem_type_attr)) {
    kind = RT_MEMCPY_HOST_TO_DEVICE;
  } else {
    // no need to update.
  }
}

Status RtsKernelTaskBuilder::BuildMemcpyAsyncTask(const GetOpDescFunc &get_op_desc_func,
                                                  const SingleOpModelParam &param, const domi::TaskDef &task_def,
                                                  OpTask *&op_task, StreamResource &resource) {
  (void)resource;
  const domi::MemcpyAsyncDef &kernel_def = task_def.memcpy_async();
  auto task = MakeUnique<MemcpyAsyncTask>();
  GE_CHECK_NOTNULL(task);
  OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET(get_op_desc_func(kernel_def.op_index(), op_desc), "get op desc failed");
  task->SetOpDesc(op_desc);
  task->dst_max_ = kernel_def.dst_max();
  task->count_ = kernel_def.count();
  task->kind_ = static_cast<rtMemcpyKind_t>(kernel_def.kind());
  UpdateCopyKind(op_desc, task->kind_);

  const auto addresses = BuildTaskUtils::JoinAddresses(BuildTaskUtils::GetAddresses(op_desc, param, false));
  if (addresses.size() != kNumAddresses) {
    GELOGE(INTERNAL_ERROR, "[Build][MemcpyAsyncTask] Invalid address count: %zu", addresses.size());
    return INTERNAL_ERROR;
  }

  task->addresses_[0U] = static_cast<uintptr_t>(PtrToValue(addresses[0UL]));
  task->addresses_[1U] = static_cast<uintptr_t>(PtrToValue(addresses[1UL]));
  op_task = task.release();
  return SUCCESS;
}

Status RtsKernelTaskBuilder::BuildNpuGetFloatStatusTask(const GetOpDescFunc &get_op_desc_func,
                                                        const SingleOpModelParam &param, const domi::TaskDef &task_def,
                                                        OpTask *&op_task, StreamResource &resource) {
  (void)resource;
  const domi::NpuGetFloatStatusDef &kernel_def = task_def.npu_get_float_status();
  auto task = MakeUnique<NpuGetFloatStatusTask>();
  GE_CHECK_NOTNULL(task);
  task->mode_ = kernel_def.mode();
  task->output_size_ = kernel_def.output_size();
  task->args_size_ = sizeof(uint8_t *);
  GE_CHK_RT_RET(aclrtMalloc(&task->args_, task->args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_CHK_STATUS_RET(ModelUtils::GetRtAddress(param.runtime_param, static_cast<uintptr_t>(kernel_def.output_addr()),
                                             task->output_addr_));
  OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET(get_op_desc_func(kernel_def.op_index(), op_desc), "get op desc failed");
  task->SetOpDesc(op_desc);
  op_task = task.release();
  return SUCCESS;
}

Status RtsKernelTaskBuilder::BuildNpuClearFloatStatusTask(const GetOpDescFunc &get_op_desc_func,
                                                          const domi::TaskDef &task_def, OpTask *&op_task,
                                                          StreamResource &resource) {
  (void)resource;
  const domi::NpuClearFloatStatusDef &kernel_def = task_def.npu_clear_float_status();
  std::unique_ptr<NpuClearFloatStatusTask> task = MakeUnique<NpuClearFloatStatusTask>();
  GE_CHECK_NOTNULL(task);
  task->mode_ = kernel_def.mode();
  OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET(get_op_desc_func(kernel_def.op_index(), op_desc), "get op desc failed");
  task->SetOpDesc(op_desc);
  op_task = task.release();
  return SUCCESS;
}

Status RtsKernelTaskBuilder::BuildNpuGetFloatDebugStatusTask(const GetOpDescFunc &get_op_desc_func,
                                                             const SingleOpModelParam &param,
                                                             const domi::TaskDef &task_def, OpTask *&op_task,
                                                             StreamResource &resource) {
  (void)resource;
  const domi::NpuGetFloatDebugStatusDef &kernel_def = task_def.npu_get_float_debug_status();
  auto task = MakeUnique<NpuGetFloatDebugStatusTask>();
  GE_CHECK_NOTNULL(task);
  task->mode_ = kernel_def.mode();
  task->output_size_ = kernel_def.output_size();
  task->args_size_ = sizeof(uint8_t *);
  GE_CHK_RT_RET(aclrtMalloc(&task->args_, task->args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_CHK_STATUS_RET(ModelUtils::GetRtAddress(param.runtime_param, static_cast<uintptr_t>(kernel_def.output_addr()),
                                             task->output_addr_));
  OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET(get_op_desc_func(kernel_def.op_index(), op_desc), "get op desc failed");
  task->SetOpDesc(op_desc);
  op_task = task.release();
  return SUCCESS;
}

Status RtsKernelTaskBuilder::BuildNpuClearFloatDebugStatusTask(const GetOpDescFunc &get_op_desc_func,
                                                               const domi::TaskDef &task_def, OpTask *&op_task,
                                                               StreamResource &resource) {
  (void)resource;
  const domi::NpuClearFloatDebugStatusDef &kernel_def = task_def.npu_clear_float_debug_status();
  std::unique_ptr<NpuClearFloatDebugStatusTask> task = MakeUnique<NpuClearFloatDebugStatusTask>();
  GE_CHECK_NOTNULL(task);
  task->mode_ = kernel_def.mode();
  OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET(get_op_desc_func(kernel_def.op_index(), op_desc), "get op desc failed");
  task->SetOpDesc(op_desc);
  op_task = task.release();
  return SUCCESS;
}
}  // namespace ge
