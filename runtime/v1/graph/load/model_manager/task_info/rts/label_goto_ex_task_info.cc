/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/label_goto_ex_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
constexpr uint8_t kGotoBranchMax = 1U;

LabelGotoExTaskInfo::~LabelGotoExTaskInfo() {
  args_ = nullptr;
  index_value_ = nullptr;
}

Status LabelGotoExTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                 const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                 const IowAddrs &iow_addrs) {
  GELOGI("LabelGotoExTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  // Get LabelGotoEx task def
  const domi::LabelGotoExDef &label_goto = task_def.label_goto_ex();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(label_goto.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Can't get op_desc from davinci_model by index:%u",
                       label_goto.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range!", label_goto.op_index());
    return INTERNAL_ERROR;
  }

  uint32_t label_index = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail",
                       ATTR_NAME_LABEL_SWITCH_INDEX.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) fail.",
           ATTR_NAME_LABEL_SWITCH_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto memory_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT,
                                          static_cast<uint32_t>(sizeof(uint64_t)));
  GELOGI("memory_type: %u", memory_type);

  GE_CHK_STATUS_RET_NOLOG(davinci_model->GetLabelGotoAddr(label_index, memory_type, args_, args_size_));

  index_value_ = davinci_model->MallocDynamicMemory(sizeof(uint64_t), memory_type);
  GE_ASSERT_NOTNULL(index_value_);

  constexpr uint64_t branch_index = 0U;
  GE_CHK_RT_RET(rtMemcpy(index_value_, sizeof(uint64_t), &branch_index, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE));

  GELOGI("LabelGotoExTaskInfo %s Init Success, label id:%u, logic stream id: %u, stream: %p.",
    op_desc->GetNamePtr(), label_index, task_def.stream_id(), stream_);
  return SUCCESS;
}

Status LabelGotoExTaskInfo::Distribute() {
  GELOGI("LabelGotoExTaskInfo Distribute Start.");
  GE_CHECK_NOTNULL(args_);
  GE_CHECK_NOTNULL(index_value_);
  if (args_size_ == 0U) {
    REPORT_INNER_ERR_MSG("E19999", "Param args_size_ is 0, check fail");
    GELOGE(PARAM_INVALID, "[Check][Param] branch max:%u, args size:%u invalid.",
           static_cast<uint32_t>(kGotoBranchMax), args_size_);
    return PARAM_INVALID;
  }

  const rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, kGotoBranchMax, args_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtLabelSwitchByIndex failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelSwitchByIndex] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  is_support_redistribute_ = true;
  GELOGI("LabelGotoExTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_STREAM_LABEL_GOTO, LabelGotoExTaskInfo);
}  // namespace ge
