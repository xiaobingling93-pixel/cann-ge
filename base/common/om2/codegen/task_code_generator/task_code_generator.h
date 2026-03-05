/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_TASK_CODE_GENERATOR_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_TASK_CODE_GENERATOR_H_

#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/om2/codegen/om2_codegen_types.h"
#include "proto/task.pb.h"

namespace ge {
class TaskCodeGenerator {
public:
  TaskCodeGenerator() = default;
  virtual ~TaskCodeGenerator() = default;

  virtual Status GenTaskDistributionCode(TaskDistributionContext &context) = 0;

  virtual Status GenDistributionImplCode(TaskDistributionImplContext &context) = 0;

  virtual int64_t ParseOpIndex(const domi::TaskDef &task_def) {
    (void)task_def;
    return -1L;
  }
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_GENERATOR_TASK_CODE_GENERATOR_H_