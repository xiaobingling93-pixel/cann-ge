/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "model_exit_op.h"
#include "op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/compute_graph.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
ModelExitOp::ModelExitOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status ModelExitOp::Init() {
  RTS_LOGI("ModelExitOp Init start, node:%s.", name_.c_str());
  input_num_ = 0;
  output_num_ = 0;

  auto op_desc = node_.GetOpDesc();
  ComputeGraphPtr owner_graph = node_.GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    RTS_REPORT_CALL_ERROR("Model exit op Init failed, graph is nullptr.");
    return FAILED;
  }

  return SUCCESS;
}

Status ModelExitOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("ModelExitOp Run start, name=%s.", name_.c_str());
  const uint32_t streamId = op_desc_->GetStreamId();
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_MODEL_EXIT);
  taskDef.set_stream_id(streamId);
  tasks.push_back(taskDef);
  RTS_LOGI("end ModelExitOp streamId:%u.", streamId);
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
