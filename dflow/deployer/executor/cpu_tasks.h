/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_TASKS_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_TASKS_H_

#include <vector>
#include <string>
#include "ge/ge_api_error_codes.h"

namespace ge {
class CpuTasks {
 public:
  static Status ExecuteModelEschedPriorityTask(int32_t process_priority, int32_t event_priority);

  static Status ExecuteModelClearTask(int32_t clear_type,
                                      const std::vector<uint32_t> &davinci_model_runtime_ids);
  static Status ExecuteCheckSupported(const std::string &kernel_name, bool &is_supported);
  static Status ExceptionNotify(const std::vector<uint32_t> &davinci_model_runtime_ids, uint32_t type,
                                uint64_t trans_id);
  static Status CheckSupportExceptionNotify();
 private:
  static Status CheckAicpuTsKernelSupported(const std::string &kernel_name, bool &is_supported);
  static Status ExecuteKernel(const std::string &kernel_name, std::vector<uint8_t> &args);
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_TASKS_H_
