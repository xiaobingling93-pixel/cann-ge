/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/end_graph_task_info.h"

#include "graph/load/model_manager/davinci_model.h"

namespace {
constexpr uint32_t kDumpFlag = 2U;
}  // namespace
namespace ge {
Status EndGraphTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                              const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                              const IowAddrs &iow_addrs) {
  GELOGI("InitEndGraphTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  GELOGI("InitEndGraphTaskInfo Init Success, model: %p, logic stream id: %u, stream: %p",
    davinci_model_->GetRtModelHandle(), task_def.stream_id(), stream_);
  return SUCCESS;
}

Status EndGraphTaskInfo::Distribute() {
  GELOGI("EndGraphTaskInfo Distribute Start.");
  GE_CHECK_NOTNULL(davinci_model_);
  GE_ASSERT_SUCCESS(davinci_model_->SetStreamLockOrUnlocK(stream_, false));
  rtError_t rt_ret;
  if (davinci_model_->ModelNeedDump()) {
    GELOGI("Start to call rtEndGraphEx");
    rt_ret = rtEndGraphEx(davinci_model_->GetRtModelHandle(), stream_, kDumpFlag);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtEndGraphEx failed, ret:%d", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtEndGraphEx] failed, ret:%d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  } else {
    GELOGI("Start to call rtEndGraph");
    rt_ret = rtEndGraph(davinci_model_->GetRtModelHandle(), stream_);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtEndGraph failed, ret:%d", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtEndGraph] failed, ret:%d", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  GE_CHK_RT_RET(rtsGetThreadLastTaskId(&task_id_));
  GE_CHK_RT_RET(rtsStreamGetId(stream_, reinterpret_cast<int32_t*>(&stream_id_)));
  davinci_model_->SetEndGraphId(task_id_, stream_id_);

  is_support_redistribute_ = true;

  GELOGI("EndGraphTaskInfo Distribute Success, task id is %u, stream id is %u, stream: %p.",
    task_id_, stream_id_, stream_);
  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_END_GRAPH, EndGraphTaskInfo);
}  // namespace ge
