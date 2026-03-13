/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/custom_engine/custom_ops_kernel_builder.h"
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "register/ops_kernel_builder_registry.h"
#include "common/ge_common/ge_types.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
  // A5上没有sqe的限制，此处后续要打开限制
  constexpr uint32_t kMaxCustomOpSqeNum = 5;
}
namespace custom {
REGISTER_OPS_KERNEL_BUILDER(kCustomOpKernelLibName, CustomOpsKernelBuilder);

CustomOpsKernelBuilder::~CustomOpsKernelBuilder() {
  GELOGI("CustomOpsKernelBuilder destroyed");
}

Status CustomOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  return SUCCESS;
}

Status CustomOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

Status CustomOpsKernelBuilder::CalcOpRunningParam(Node &node) {
  auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++) {
    if (op_desc->GetOutputDesc(i).GetShape().IsUnknownShape()) {
      GELOGI("Node[%s] output[%zu] is unknown shape.", op_desc->GetName().c_str(), i);
      continue;
    }
    int64_t tensor_size = 0;
    GE_ASSERT_SUCCESS(TensorUtils::GetTensorMemorySizeInBytes(*op_desc->MutableOutputDesc(i), tensor_size));
    TensorUtils::SetSize(*op_desc->MutableOutputDesc(i), tensor_size);
  }
  (void)node;
  return SUCCESS;
}

Status CustomOpsKernelBuilder::GenerateTask(const Node &node,
    RunContext &context, std::vector<domi::TaskDef> &tasks) {
  (void)node;
  (void)context;
  (void)tasks;
  GE_ASSERT_NOTNULL(node.GetOpDesc());
  domi::TaskDef task_def = {};
  task_def.set_stream_id(node.GetOpDesc()->GetStreamId());
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_sqe_num(kMaxCustomOpSqeNum);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(node.GetOpDesc()->GetId());
  tasks.push_back(task_def);
  return SUCCESS;
}
}  // namespace custom
}  // namespace ge
