/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmo_addr_task_code_generator.h"
#include "common/om2/codegen/code_generator_factory.h"

namespace ge {
Status CmoAddrTaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  (void)context;
  return SUCCESS;
}

Status CmoAddrTaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  (void)context;
  return SUCCESS;
}

int64_t CmoAddrTaskCodeGenerator::ParseOpIndex(const domi::TaskDef &task_def) {
  const domi::CmoAddrTaskDef &cmo_addr_task = task_def.cmo_addr_task();
  return static_cast<int64_t>(cmo_addr_task.op_index());
}

REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_CMO_ADDR, CmoAddrTaskCodeGenerator);
}  // namespace ge
