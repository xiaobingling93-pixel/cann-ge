/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "label_switch_by_index_task_code_generator.h"
#include "common/om2/codegen/code_generator_factory.h"

namespace ge {
Status LabelSwitchByIndexTaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  (void)context;
  return SUCCESS;
}

Status LabelSwitchByIndexTaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  (void)context;
  return SUCCESS;
}

int64_t LabelSwitchByIndexTaskCodeGenerator::ParseOpIndex(const domi::TaskDef &task_def) {
  const domi::LabelSwitchByIndexDef &label_switch = task_def.label_switch_by_index();
  return static_cast<int64_t>(label_switch.op_index());
}

REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX, LabelSwitchByIndexTaskCodeGenerator);
}  // namespace ge
