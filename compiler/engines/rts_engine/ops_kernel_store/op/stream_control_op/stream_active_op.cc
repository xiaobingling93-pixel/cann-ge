/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "stream_active_op.h"

#include "graph/debug/ge_attr_define.h"
#include "op_factory.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
StreamActiveOp::StreamActiveOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status StreamActiveOp::Init() {
  RTS_LOGI("StreamActiveOp Init, node:%s.", name_.c_str());
  // Set input number、output number
  input_num_ = op_desc_->GetInputsSize();
  output_num_ = op_desc_->GetOutputsSize();

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init StreamActiveOp failed. node:%s.", name_.c_str());
    return ret;
  }

  return SUCCESS;
}

Status StreamActiveOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("StreamActiveOp op:%s start run", name_.c_str());
  vector<uint32_t> activeStreamList;
  bool getAttrSucc = AttrUtils::GetListInt(op_desc_, ATTR_NAME_ACTIVE_STREAM_LIST, activeStreamList);
  if (!getAttrSucc) {
    RTS_LOGE("StreamActiveOp[node:%s] get attr ACTIVE_STREAM_LIST fail.", name_.c_str());
    return FAILED;
  }

  for (size_t index = 0; index < activeStreamList.size(); index++) {
    domi::TaskDef taskDef = {};
    taskDef.set_type(ACL_RT_MODEL_TASK_STREAM_ACTIVE);
    taskDef.set_stream_id(op_desc_->GetStreamId());

    domi::StreamActiveDef *streamActiveDef = taskDef.mutable_stream_active();
    streamActiveDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
    streamActiveDef->set_active_stream_id(static_cast<uint32_t>(activeStreamList[index]));
    tasks.push_back(taskDef);
  }
  RTS_LOGI("StreamActiveOp op:%s  size:%zu end.", name_.c_str(), activeStreamList.size());

  return SUCCESS;
}

Status StreamActiveOp::UpdateTaskDef(vector<TaskDef> &tasks) {
  tasks.clear();
  vector<uint32_t> activeStreamList;
  bool getAttrSucc = AttrUtils::GetListInt(op_desc_, ATTR_NAME_ACTIVE_STREAM_LIST, activeStreamList);
  if (!getAttrSucc) {
    RTS_LOGE("StreamActiveOp[node:%s] get attr ACTIVE_STREAM_LIST fail.", name_.c_str());
    return FAILED;
  }

  for (size_t index = 0; index < activeStreamList.size(); index++) {
    domi::TaskDef taskDef = {};
    taskDef.set_type(ACL_RT_MODEL_TASK_STREAM_ACTIVE);
    taskDef.set_stream_id(op_desc_->GetStreamId());

    domi::StreamActiveDef *streamActiveDef = taskDef.mutable_stream_active();
    streamActiveDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
    streamActiveDef->set_active_stream_id(static_cast<uint32_t>(activeStreamList[index]));
    tasks.push_back(taskDef);
  }
  RTS_LOGI("StreamActiveOp update op:%s  size:%zu end.", name_.c_str(), activeStreamList.size());

  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
