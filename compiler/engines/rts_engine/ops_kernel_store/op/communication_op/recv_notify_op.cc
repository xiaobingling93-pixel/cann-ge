/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "recv_notify_op.h"

#include "common/constant/constant.h"
#include "op_factory.h"
#include "common/util/log.h"
#include "../../../inc/framework/common/runtime_model_ge.h"

using namespace ge;
namespace cce {
namespace runtime {
RecvNotifyOp::RecvNotifyOp(const Node &node, RunContext &runContext) : Op(node, runContext), notifyId_(0) {}

Status RecvNotifyOp::Init() {
  input_num_ = 0;
  output_num_ = 0;

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init failed.");
    return ret;
  }

  uint32_t notifyId = 0;
  if (!AttrUtils::GetInt(op_desc_, ATTR_NAME_RECV_ATTR_NOTIFY_ID, notifyId)) {
    RTS_REPORT_CALL_ERROR("Receive op init failed, get notify_id attribute failed, node:%s.", name_.c_str());
    return FAILED;
  }
  notifyId_ = notifyId;

  RTS_LOGI("Recv op:%s init, notify id:%u.", name_.c_str(), notifyId);
  return SUCCESS;
}

Status RecvNotifyOp::Run(vector<TaskDef> &tasks) {
  RTS_LOGI("RecvNotify op:%s start wait notify", name_.c_str());
  TaskDef taskDef = {};
  taskDef.set_type(ACL_RT_MODEL_TASK_NOTIFY_WAIT);
  taskDef.set_stream_id(op_desc_->GetStreamId());
  taskDef.set_notify_id(notifyId_);
  tasks.push_back(taskDef);
  RTS_LOGI("Recv op:%s wait notify end, notify_id:%u.", name_.c_str(), notifyId_);
  return SUCCESS;
}

}  // namespace runtime
}  // namespace cce
