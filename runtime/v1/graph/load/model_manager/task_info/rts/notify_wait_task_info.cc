/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/notify_wait_task_info.h"
#include "hcom/hcom_topo_info.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"

namespace ge {
namespace {
constexpr const char_t *GROUP_POLICY = "group";
}
Status NotifyWaitTaskInfo::SetNotifyHandleByEngine(const std::string &custom_group_name) {
  const std::string *group_name_ptr = nullptr;
  if (custom_group_name.empty()) {
    group_name_ptr = AttrUtils::GetStr(op_desc_, GROUP_POLICY);
    GE_ASSERT_NOTNULL(group_name_ptr);
  } else {
    group_name_ptr = AttrUtils::GetStr(op_desc_, custom_group_name.c_str());
    GE_ASSERT_NOTNULL(group_name_ptr, "find no %s attr", custom_group_name.c_str());
  }
  std::vector<void *> context_addrs;
  GE_ASSERT_SUCCESS(ArgsFormatUtils::GetHcomHiddenInputs(op_desc_, *davinci_model_, context_addrs));
  GE_ASSERT_SUCCESS(HcomTopoInfo::Instance().GetGroupNotifyHandle(group_name_ptr->c_str(), notify_));
  GELOGI("Get notify %p for %s %s with stream %p and group %s successfully.", notify_, op_desc_->GetNamePtr(),
         op_desc_->GetTypePtr(), stream_, group_name_ptr->c_str());
  return SUCCESS;
}

Status NotifyWaitTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                                const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  op_index_ = task_def.id();
  GELOGD("NotifyWaitTaskInfo init for op with index: %u.", op_index_);
  op_desc_ = davinci_model_->GetOpByIndex(op_index_);
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("NotifyWaitTaskInfo init for %s %s with index: %u.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr(),
         op_index_);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));
  const uint32_t notify_id = task_def.notify_id();
  if (notify_id == UINT32_MAX) {
    return SetNotifyHandleByEngine(task_def.private_def());
  }
  const auto &notifyList = davinci_model->GetNotifyList();
  if (task_def.notify_id() >= notifyList.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Task notify_id:%u > model notify size:%zu, check invalid",
                       task_def.notify_id(), notifyList.size());
    GELOGE(INTERNAL_ERROR, "[Check][Param] notify list size:%zu, cur:%u!", notifyList.size(), task_def.notify_id());
    return INTERNAL_ERROR;
  }
  GELOGI("NotifyWaitTaskInfo Init Success, logic stream id: %u, stream: %p.", task_def.stream_id(), stream_);
  notify_ = notifyList[static_cast<size_t>(task_def.notify_id())];
  return SUCCESS;
}

Status NotifyWaitTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGD("NotifyWaitTaskInfo distribute for %s %s.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  SetTaskTag(op_desc_->GetName().c_str());
  const rtError_t rt_ret = rtNotifyWait(notify_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtNotifyWait failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtNotifyWait] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  is_support_redistribute_ = true;

  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_NOTIFY_WAIT, NotifyWaitTaskInfo);
}  // namespace ge
