/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/label_set_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status LabelSetTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                              const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                              const IowAddrs &iow_addrs) {
  GELOGI("LabelSetTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  // Get LabelSet task def
  const domi::LabelSetDef &label_set = task_def.label_set();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(label_set.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Can't get op_desc from davinci_model by index:%u", label_set.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range!", label_set.op_index());
    return INTERNAL_ERROR;
  }

  uint32_t label_index = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail",
                       ATTR_NAME_LABEL_SWITCH_INDEX.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] LabelSetTaskInfo:%s attr [%s] does not exist.",
           op_desc->GetName().c_str(), ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return INTERNAL_ERROR;
  }

  const std::vector<rtLabel_t> &label_list = davinci_model->GetLabelList();
  if (label_index >= label_list.size()) {
    REPORT_INNER_ERR_MSG("E19999", "lable_index:%u >= label_list.size():%zu in model, op:%s(%s), "
                       "check invalid", label_index, label_list.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] LabelSetTaskInfo: Invalid label id:%u, label size:%zu, op:%s(%s)",
           label_index, label_list.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  label_ = label_list[static_cast<size_t>(label_index)];

  GELOGI("LabelSetTaskInfo %s Init success, label id:%u, label:%p, logic stream id: %u, stream: %p.",
    op_desc->GetNamePtr(), label_index, label_, task_def.stream_id(), stream_);
  return SUCCESS;
}

Status LabelSetTaskInfo::Distribute() {
  GELOGI("LabelSetTaskInfo Distribute Start.");
  const rtError_t rt_ret = rtLabelSet(label_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtLabelSet failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelSet] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  is_support_redistribute_ = true;
  GELOGI("LabelSetTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_LABEL_SET, LabelSetTaskInfo);
}  // namespace ge
