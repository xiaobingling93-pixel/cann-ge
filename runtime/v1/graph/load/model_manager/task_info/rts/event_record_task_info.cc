/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/event_record_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status EventRecordTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                 const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                 const IowAddrs &iow_addrs) {
  GELOGI("EventRecordTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  const auto &eventList = davinci_model_->GetEventList();
  if (task_def.event_id() >= eventList.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Task event_id:%u > model event size:%zu, check invalid",
                       task_def.event_id(), eventList.size());
    GELOGE(INTERNAL_ERROR, "[Check][Param] event list size:%zu, cur:%u!", eventList.size(), task_def.event_id());
    return INTERNAL_ERROR;
  }

  event_ = eventList[static_cast<size_t>(task_def.event_id())];

  if (task_def.has_event_ex()) {
    op_index_ = task_def.event_ex().op_index();
  }
  op_desc_ = davinci_model_->GetOpByIndex(op_index_);
  GE_CHECK_NOTNULL(op_desc_);

  GELOGI("EventRecordTaskInfo Init Success, node :%s, logic stream id: %u, stream: %p.",
    op_desc_->GetName().c_str(), task_def.stream_id(), stream_);

  return SUCCESS;
}

Status EventRecordTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("EventRecordTaskInfo op %s Distribute Start.", op_desc_->GetNamePtr());
  SetTaskTag(op_desc_->GetName().c_str());

  const rtError_t rt_ret = rtEventRecord(event_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtEventRecord failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtEventRecord] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
  }
  is_support_redistribute_ = true;
  GELOGI("EventRecordTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_EVENT_RECORD, EventRecordTaskInfo);
}  // namespace ge
