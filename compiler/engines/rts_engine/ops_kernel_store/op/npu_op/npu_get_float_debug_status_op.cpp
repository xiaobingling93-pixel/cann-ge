/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "npu_get_float_debug_status_op.h"

#include "op_factory.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
NpuGetFloatDebugStatusOp::NpuGetFloatDebugStatusOp(const Node &node, RunContext &runContext)
    : Op(node, runContext), check_mode_(0) {}

Status NpuGetFloatDebugStatusOp::Init() {
  RTS_LOGI("NpuGetFloatStatusOp Init start, node:%s.", name_.c_str());
  // NPUGetFloatStatusV2 has 1 output and 0 input
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();
  RTS_LOGI("input_num:%zu, output_num:%zu.", input_num_, output_num_);
  return Op::Init();
}

Status NpuGetFloatDebugStatusOp::Run(vector<TaskDef> &tasks) {
  if (v_output_data_addr_.size() != v_output_size_.size() || v_output_data_addr_.size() != 1) {
    RTS_REPORT_INNER_ERROR(
        "Run npu get float status op failed, output data addr num(%zu) output num(%zu) is invalid, node:%s.",
        v_output_data_addr_.size(), v_output_size_.size(), name_.c_str());
    return INTERNAL_ERROR;
  }

  RTS_LOGI("NPU get float status op run start, node: %s.", name_.c_str());
  domi::TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_NPU_GET_DEBUG_FLOAT_STATUS);
  taskDef.set_stream_id(op_desc_->GetStreamId());
  domi::NpuGetFloatDebugStatusDef *npuGetFloatDebugStatusDef = taskDef.mutable_npu_get_float_debug_status();
  npuGetFloatDebugStatusDef->set_output_addr(reinterpret_cast<uintptr_t>(v_output_data_addr_[0]));
  npuGetFloatDebugStatusDef->set_output_size(v_output_size_[0]);
  npuGetFloatDebugStatusDef->set_mode(check_mode_);
  npuGetFloatDebugStatusDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  tasks.push_back(taskDef);
  RTS_LOGI("NPU get float status op run end, stream_id:%u.", op_desc_->GetStreamId());
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce
