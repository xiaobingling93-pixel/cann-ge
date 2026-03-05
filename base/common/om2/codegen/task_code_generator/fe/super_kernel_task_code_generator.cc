/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "super_kernel_task_code_generator.h"
#include "common/om2/codegen/code_generator_factory.h"

namespace ge {
Status SuperKernelV2TaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  (void)context;
  return SUCCESS;
}

Status SuperKernelV2TaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  (void)context;
  return SUCCESS;
}

int64_t SuperKernelV2TaskCodeGenerator::ParseOpIndex(const domi::TaskDef &task_def) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();
  return static_cast<int64_t>(context.op_index());
}

REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_SUPER_KERNEL, SuperKernelV2TaskCodeGenerator);
}  // namespace ge
