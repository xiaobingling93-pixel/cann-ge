/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/cpu_tasks.h"
#include "aicpu/aicpu_schedule/aicpusd_info.h"
#include "aicpu_task_struct.h"
#include "aicpu/queue_schedule/dgw_client.h"
#include "framework/common/debug/log.h"
#include "graph/def_types.h"
#include "runtime/rt.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr const char *kKernelNameModelEschedPriority = "AicpuModelEschedPriority";
constexpr const char *kKernelNameModelStop = "AICPUModelStop";
constexpr const char *kKernelNameModelClear = "AICPUModelClearInputAndRestart";
constexpr const char *kKernelNameCheckSupported = "CheckKernelSupported";
constexpr const char *kKernelNameProcessDataException = "ProcessDataException";
constexpr int32_t kDefaultPriority = 0;
constexpr uint32_t kKernelBlockDim = 1U;
constexpr int32_t kClearTypeStop = 1;
constexpr int32_t kClearTypeClear = 2;
}  // namespace

Status CpuTasks::ExecuteKernel(const std::string &kernel_name, std::vector<uint8_t> &args) {
  rtStream_t stream = nullptr;
  GE_CHK_RT_RET(rtStreamCreate(&stream, kDefaultPriority));
  GE_MAKE_GUARD_RTSTREAM(stream);
  rtArgsEx_t args_info = {};
  args_info.args = static_cast<void *>(args.data());
  args_info.argsSize = static_cast<uint32_t>(args.size());
  GE_CHK_RT_RET(rtCpuKernelLaunchWithFlag(nullptr,
      kernel_name.c_str(), kKernelBlockDim, &args_info, nullptr, stream, RT_KERNEL_DEFAULT));
  GELOGD("Launch cpu kernel successfully, kernel name = %s.", kernel_name.c_str());
  GE_CHK_RT_RET(rtStreamSynchronize(stream));
  GELOGD("Stream synchronize successfully, kernel name = %s.", kernel_name.c_str());
  return SUCCESS;
}

Status CpuTasks::ExecuteModelEschedPriorityTask(int32_t process_priority, int32_t event_priority) {
  const auto args_size = sizeof(aicpu::AicpuParamHead) + sizeof(AicpuPriInfo);
  std::vector<uint8_t> task_args(args_size, 0U);
  auto &param_head = *(PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data()));
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input
  auto &priority_info = *(PtrToPtr<uint8_t, AicpuPriInfo>(&(task_args[sizeof(aicpu::AicpuParamHead)])));
  priority_info.checkHead = PRIORITY_MSG_CHECKCODE;
  priority_info.pidPriority = process_priority;
  priority_info.eventPriority = event_priority;
  GELOGD("The process priority = %d, event priority = %d", process_priority, event_priority);
  return ExecuteKernel(kKernelNameModelEschedPriority, task_args);
}

Status CpuTasks::ExecuteModelClearTask(int32_t clear_type,
                                       const std::vector<uint32_t> &davinci_model_runtime_ids) {
  if (davinci_model_runtime_ids.empty()) {
    return SUCCESS;
  }
  std::string kernel_name_clear_model;
  if (clear_type == kClearTypeStop) {
    kernel_name_clear_model = kKernelNameModelStop;
  } else if (clear_type == kClearTypeClear) {
    kernel_name_clear_model = kKernelNameModelClear;
  }
  void *model_ids_addr = nullptr;
  const uint64_t model_ids_size = davinci_model_runtime_ids.size() * sizeof(uint32_t);
  GE_CHK_RT_RET(rtMalloc(&model_ids_addr, model_ids_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_MAKE_GUARD(model_ids_addr, [model_ids_addr]() {
    GE_CHK_RT(rtFree(model_ids_addr));
  });
  GE_CHK_RT_RET(rtMemcpy(model_ids_addr, model_ids_size,
                         davinci_model_runtime_ids.data(), model_ids_size,
                         RT_MEMCPY_HOST_TO_DEVICE));
  const auto args_size = sizeof(ReDeployConfig);
  std::vector<uint8_t> task_args(args_size, 0U);
  auto param_re_deploy_config = PtrToPtr<uint8_t, ReDeployConfig>(task_args.data());
  param_re_deploy_config->modelIdNum = davinci_model_runtime_ids.size();
  param_re_deploy_config->modelIdsAddr = PtrToValue(model_ids_addr);
  return ExecuteKernel(kernel_name_clear_model, task_args);
}

Status CpuTasks::ExceptionNotify(const std::vector<uint32_t> &davinci_model_runtime_ids, uint32_t type,
                                 uint64_t trans_id) {
  if (davinci_model_runtime_ids.empty()) {
    return SUCCESS;
  }

  void *model_ids_addr = nullptr;
  const uint64_t model_ids_size = davinci_model_runtime_ids.size() * sizeof(uint32_t);
  GE_CHK_RT_RET(rtMalloc(&model_ids_addr, model_ids_size, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_MAKE_GUARD(model_ids_addr, [model_ids_addr]() { GE_CHK_RT(rtFree(model_ids_addr)); });
  GE_CHK_RT_RET(rtMemcpy(model_ids_addr, model_ids_size, davinci_model_runtime_ids.data(), model_ids_size,
                         RT_MEMCPY_HOST_TO_DEVICE));
  const auto args_size = sizeof(DataFlowExceptionNotify);
  std::vector<uint8_t> task_args(args_size, 0U);
  auto notify_info = PtrToPtr<uint8_t, DataFlowExceptionNotify>(task_args.data());
  notify_info->transId = trans_id;
  notify_info->type = type;
  notify_info->modelIdNum = davinci_model_runtime_ids.size();
  notify_info->modelIdsAddr = PtrToValue(model_ids_addr);
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameProcessDataException, task_args),
                    "Failed to notify exception to aicpu, trans_id=%lu, type=%u.", trans_id, type);
  GELOGI("notify exception to aicpu end, trans_id=%lu, type=%u.", trans_id, type);
  return SUCCESS;
}

Status CpuTasks::CheckSupportExceptionNotify() {
  bool is_supported = false;
  GE_CHK_STATUS_RET(CheckAicpuTsKernelSupported(kKernelNameProcessDataException, is_supported));
  GE_CHK_BOOL_RET_STATUS(is_supported, FAILED,
                         "Aicpu does not support exception notify. Please update software or unset exception catch.");
  return SUCCESS;
}

Status CpuTasks::CheckAicpuTsKernelSupported(const std::string &kernel_name, bool &is_supported) {
  constexpr const char *ts_kernel_prefix = "tsKernel:";
  return ExecuteCheckSupported(ts_kernel_prefix + kernel_name, is_supported);
}

Status CpuTasks::ExecuteCheckSupported(const std::string &kernel_name, bool &is_supported) {
  const auto args_size = sizeof(CheckKernelSupportedConfig);
  std::vector<uint8_t> task_args(args_size, 0U);
  auto &check_cfg = *(PtrToPtr<uint8_t, CheckKernelSupportedConfig>(task_args.data()));
  // prepare device data
  int32_t result = -1;
  void *dev_result_ptr = nullptr;
  void *dev_name_ptr = nullptr;
  GE_CHK_RT_RET(rtMalloc(&dev_result_ptr, sizeof(int32_t), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_MAKE_GUARD(dev_result_ptr, ([dev_result_ptr]() {
    GE_CHK_RT(rtFree(dev_result_ptr));
  }));
  GE_CHK_RT_RET(rtMalloc(&dev_name_ptr, kernel_name.length(), RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_MAKE_GUARD(dev_name_ptr, ([dev_name_ptr]() {
    GE_CHK_RT(rtFree(dev_name_ptr));
  }));
  GE_CHK_RT_RET(rtMemcpy(dev_result_ptr, sizeof(int32_t), &result, sizeof(int32_t), RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT_RET(rtMemcpy(dev_name_ptr, kernel_name.length(), kernel_name.c_str(),
                         kernel_name.length(), RT_MEMCPY_HOST_TO_DEVICE));

  check_cfg.kernelNameAddr = PtrToValue(dev_name_ptr);
  check_cfg.kernelNameLen = kernel_name.length();
  check_cfg.checkResultAddr = PtrToValue(dev_result_ptr);
  check_cfg.checkResultLen = sizeof(result);
  GELOGI("Start to check[%s] is supported.", kernel_name.c_str());
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameCheckSupported, task_args), "Execute kernel for check supported failed.");

  GE_CHK_RT_RET(rtMemcpy(&result, sizeof(int32_t), dev_result_ptr, sizeof(int32_t), RT_MEMCPY_DEVICE_TO_HOST));
  GELOGD("Get result %d after cpu kernel task to be executed.", result);
  is_supported = (result == 0);
  return SUCCESS;
}
}  // namespace ge
