/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_start_task_code_generator.h"
#include "common/om2/codegen/code_generator_factory.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {

Status FusionStartTaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx,
    "  // ============================= FUSION_START ==============================="));
  uint32_t stream_id = context.task_def.stream_id();
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, "  OM2_CHK_STATUS(KernelFusionStartDistribute(stream_list_[" + std::to_string(stream_id)
     + "]));"));
  return SUCCESS;
}

Status FusionStartTaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  std::stringstream code_stream;
  code_stream << R"(aclError KernelFusionStartDistribute(aclrtStream stream) {
  (void)rtKernelFusionStart(stream);
  return ACL_SUCCESS;
}
)";
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, code_stream.str()));
  return SUCCESS;
}

REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_FUSION_START, FusionStartTaskCodeGenerator);
} // namespace ge