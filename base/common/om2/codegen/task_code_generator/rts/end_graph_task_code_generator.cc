/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "end_graph_task_code_generator.h"
#include "common/om2/codegen/code_generator_factory.h"

namespace ge {
Status EndGraphTaskCodeGenerator::GenTaskDistributionCode(TaskDistributionContext &context) {
  std::stringstream code_stream;
  code_stream << "  // EndGraph\n";
  code_stream << "  OM2_CHK_STATUS(EndGraphTaskDistribute(model_handle_, stream_list_[" << context.task_def.stream_id()
              << "]));\n";
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, code_stream.str()));
  return SUCCESS;
}

Status EndGraphTaskCodeGenerator::GenDistributionImplCode(TaskDistributionImplContext &context) {
  std::stringstream code_stream;
  code_stream << R"(
aclError EndGraphTaskDistribute(aclmdlRI mdl, aclrtStream stream) {
  OM2_CHK_STATUS(aclmdlRIEndTask(mdl, stream));
  return ACL_SUCCESS;
}
)";
  context.nodes.push_back(RAW_CODE_STMT(context.ast_ctx, code_stream.str()));
  return SUCCESS;
}
REGISTER_TASK_CODE_GENERATOR(MODEL_TASK_END_GRAPH, EndGraphTaskCodeGenerator);
}  // namespace ge
