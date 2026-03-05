/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_switch_task_code_generator.h"
#include "common/om2/codegen/code_generator_factory.h"

namespace ge {
Status StreamSwitchTaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  (void)context;
  return SUCCESS;
}

Status StreamSwitchTaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  (void)context;
  return SUCCESS;
}

int64_t StreamSwitchTaskCodeGenerator::ParseOpIndex(const domi::TaskDef &task_def) {
  const auto &stream_switch_def = task_def.stream_switch();
  return static_cast<int64_t>(stream_switch_def.op_index());
}

REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_STREAM_SWITCH, StreamSwitchTaskCodeGenerator);
}  // namespace ge
