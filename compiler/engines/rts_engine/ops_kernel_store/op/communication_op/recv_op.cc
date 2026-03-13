/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "recv_op.h"

#include "common/constant/constant.h"
#include "op_factory.h"
#include "graph/args_format_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

constexpr uint8_t MEM_WAIT_SQE_NUM = 3U;

using namespace ge;
namespace cce {
namespace runtime {
RecvOp::RecvOp(const Node &node, RunContext &runContext) : Op(node, runContext), eventId_(0) {}

Status RecvOp::Init() {
  input_num_ = 0;
  output_num_ = 0;

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init failed.");
    return ret;
  }

  uint32_t eventId = 0;
  if (!AttrUtils::GetInt(op_desc_, ATTR_NAME_RECV_ATTR_EVENT_ID, eventId)) {
    RTS_REPORT_CALL_ERROR("Receive op init failed, get event_id attribute failed, node:%s.", name_.c_str());
    return FAILED;
  }

  eventId_ = eventId;

  RTS_LOGI("Recv op:%s init, event id:%u.", name_.c_str(), eventId);
  return SUCCESS;
}

Status RecvOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("Recv op:%s start wait record", name_.c_str());
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_EVENT_WAIT);
  taskDef.set_stream_id(op_desc_->GetStreamId());
  taskDef.set_event_id(eventId_);
  domi::EventExDef *eventDef = taskDef.mutable_event_ex();
  eventDef->set_event_type(ACL_RT_MODEL_TASK_EVENT_WAIT);
  eventDef->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));
  tasks.push_back(taskDef);
  RTS_LOGI("Recv op:%s wait record end, event_id:%u.", name_.c_str(), eventId_);
  return SUCCESS;
}

RecvOpMem::RecvOpMem(const Node &node, RunContext &runContext) : Op(node, runContext), eventId_(0) {}

Status RecvOpMem::Init() {
  input_num_ = 0;
  output_num_ = 0;

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    return ret;
  }

  int32_t eventId = 0;
  if (!AttrUtils::GetInt(op_desc_, ATTR_NAME_RECV_ATTR_EVENT_ID, eventId)) {
    RTS_REPORT_CALL_ERROR("RecvOpMem init failed, get event_id attribute failed, node:%s.", name_.c_str());
    return FAILED;
  }

  eventId_ = eventId;
  RTS_LOGI("Recv mem op:%s init, event id:%d.", name_.c_str(), eventId);
  return SUCCESS;
}

Status RecvOpMem::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("Recv mem op:%s wait value start", name_.c_str());
  TaskDef taskDef = {};
  taskDef.set_type(RT_MODEL_TASK_MEM_EVENT_WAIT);
  taskDef.set_stream_id(op_desc_->GetStreamId());
  taskDef.set_event_id(eventId_);

  OpDescPtr opDesc = node_.GetOpDesc();
  if (opDesc != nullptr) {
    (void)AttrUtils::SetInt(opDesc, ATTR_NAME_NODE_SQE_NUM, MEM_WAIT_SQE_NUM);
  }

  domi::EventExDef *event_ex_def = taskDef.mutable_event_ex();
  event_ex_def->set_event_type(RT_MODEL_TASK_MEM_EVENT_WAIT);
  event_ex_def->set_op_index(static_cast<uint32_t>(op_desc_->GetId()));

  tasks.push_back(taskDef);

  RTS_LOGI("Recv mem op:%s wait value end, event_id:%d.", name_.c_str(), eventId_);
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
