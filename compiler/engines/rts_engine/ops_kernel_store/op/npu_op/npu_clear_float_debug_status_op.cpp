/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "npu_clear_float_debug_status_op.h"

#include "op_factory.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
NpuClearFloatDebugStatusOp::NpuClearFloatDebugStatusOp(const Node &node, RunContext &runContext)
    : Op(node, runContext), check_mode_(0) {}

Status NpuClearFloatDebugStatusOp::Init() {
  RTS_LOGI("NpuClearFloatDebugStatusOp Init start, node:%s.", name_.c_str());
  // NPUClearFloatStatusV2 has 0 output and 0 input
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();
  RTS_LOGI("input_num:%zu, output_num:%zu.", input_num_, output_num_);
  return Op::Init();
}

Status NpuClearFloatDebugStatusOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("NPU clear float debug status op run start, node: %s.", name_.c_str());
  domi::TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_NPU_CLEAR_DEBUG_FLOAT_STATUS);
  taskDef.set_stream_id(op_desc_->GetStreamId());
  domi::NpuClearFloatDebugStatusDef *clearFloatStatusDef = taskDef.mutable_npu_clear_float_debug_status();
  clearFloatStatusDef->set_mode(check_mode_);
  clearFloatStatusDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  tasks.push_back(taskDef);
  RTS_LOGI("NPU clear float debug status op run end stream_id:%u.", op_desc_->GetStreamId());
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
