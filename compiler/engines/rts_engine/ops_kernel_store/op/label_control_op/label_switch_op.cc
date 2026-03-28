/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "label_switch_op.h"

#include "op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
static const uint32_t LABEL_SWITCH_INPUT_NUM = 1;

LabelSwitchOp::LabelSwitchOp(const Node &node, RunContext &runContext) : Op(node, runContext), branch_max_(0) {}

Status LabelSwitchOp::Init() {
  RTS_LOGI("LabelSwitchOp Init, node: %s.", name_.c_str());
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init failed, node: %s, retCode:%#x.", name_.c_str(), ret);
    return ret;
  }

  if (input_num_ != LABEL_SWITCH_INPUT_NUM || v_input_data_addr_.size() != LABEL_SWITCH_INPUT_NUM) {
    RTS_REPORT_CALL_ERROR("LabelSwitchOp input_num should be %u, actually input num is %zu, input addr size is %zu.",
                          LABEL_SWITCH_INPUT_NUM, input_num_, v_input_data_addr_.size());
    return FAILED;
  }

  std::vector<uint32_t> label_idx_list;
  if (!AttrUtils::GetListInt(op_desc_, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list)) {
    RTS_REPORT_CALL_ERROR("LabelSwitchOp: %s get %s failed.", name_.c_str(), ATTR_NAME_LABEL_SWITCH_LIST.c_str());
    return FAILED;
  }

  if (label_idx_list.empty()) {
    RTS_REPORT_CALL_ERROR("LabelSwitchOp: %s label list is empty!", name_.c_str());
    return FAILED;
  }
  branch_max_ = label_idx_list.size();

  return SUCCESS;
}

Status LabelSwitchOp::Run(vector<TaskDef> &tasks) {
  const uint32_t streamId = op_desc_->GetStreamId();
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX);
  taskDef.set_stream_id(streamId);
  domi::LabelSwitchByIndexDef *labelSwitchDef = taskDef.mutable_label_switch_by_index();
  labelSwitchDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  labelSwitchDef->set_label_max(branch_max_);
  tasks.push_back(taskDef);
  RTS_LOGI("end LabelSwitchOp streamId:%u.", streamId);
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
