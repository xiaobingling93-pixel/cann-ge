/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/run_context_util.h"

#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/omg_util/omg_util.h"
#include "base/err_msg.h"

namespace ge {
Status RunContextUtil::InitMemInfo(uint8_t *data_mem_base, uint64_t data_mem_size,
                                   std::map<int64_t, uint8_t *> mem_type_to_data_mem_base,
                                   std::map<int64_t, uint64_t> mem_type_to_data_mem_size, uint8_t *weight_mem_base,
                                   uint64_t weight_mem_size) {
  if ((data_mem_size > 0) && (data_mem_base == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "InitMemInfo param data_mem_base is null but data_mem_size = %lu", data_mem_size);
    GELOGE(PARAM_INVALID, "[Check][Param] InitMemInfo param data_mem_base is null but data_mem_size = %lu.",
           data_mem_size);
    return PARAM_INVALID;
  }
  if ((weight_mem_size > 0) && (weight_mem_base == nullptr)) {
    REPORT_INNER_ERR_MSG("E19999", "InitMemInfo param weight_mem_base is null but weight_mem_size = %lu",
                       weight_mem_size);
    GELOGE(PARAM_INVALID, "[Check][Param] InitMemInfo param weight_mem_base is null but weight_mem_size = %lu.",
           weight_mem_size);
    return PARAM_INVALID;
  }
  if (mem_type_to_data_mem_base.empty() || mem_type_to_data_mem_size.empty() ||
      mem_type_to_data_mem_base.size() != mem_type_to_data_mem_size.size()) {
    REPORT_INNER_ERR_MSG("E19999", "InitMemInfo param mem_type_to_data_mem_base size[%zu] "
                       "is not equal to the size of mem_type_to_data_mem_size[%zu].",
                       mem_type_to_data_mem_base.size(), mem_type_to_data_mem_size.size());
    GELOGE(PARAM_INVALID,
           "[Check][Param] InitMemInfo param mem_type_to_data_mem_base size[%zu] is not equal to the size of "
           "mem_type_to_data_mem_size[%zu].", mem_type_to_data_mem_base.size(), mem_type_to_data_mem_size.size());
    return PARAM_INVALID;
  }
  data_mem_base_ = data_mem_base;
  data_mem_size_ = data_mem_size;
  weight_mem_base_ = weight_mem_base;
  weight_mem_size_ = weight_mem_size;
  mem_type_to_data_mem_base_ = mem_type_to_data_mem_base;
  mem_type_to_data_mem_size_ = mem_type_to_data_mem_size;
  return SUCCESS;
}

Status RunContextUtil::CreateRunContext(Model &model, const ComputeGraphPtr &graph, Buffer &buffer,
                                        const uint64_t session_id) {
  GELOGD("Begin to Create RunContext, session_id = %lu", session_id);
  // check params
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Check param graph nullptr, session_id:%lu,", session_id);
    GELOGE(PARAM_INVALID, "[Check][Param] CreateRunContext param graph is null. session_id=%lu", session_id);
    return PARAM_INVALID;
  }

  uint32_t notify_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_NOTIFY_NUM, notify_num)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s failed from model, session_id:%lu,", ATTR_MODEL_NOTIFY_NUM.c_str(),
                       session_id);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s failed from model, session_id:%lu,", ATTR_MODEL_NOTIFY_NUM.c_str(),
           session_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Notify_num = %u", notify_num);

  uint32_t event_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_EVENT_NUM, event_num)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s failed from model, session_id:%lu,",
                       ATTR_MODEL_EVENT_NUM.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s failed from model, session_id:%lu,",
           ATTR_MODEL_EVENT_NUM.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Event_num = %u", event_num);

  uint32_t label_num = 0;
  if (!AttrUtils::GetInt(&model, ATTR_MODEL_LABEL_NUM, label_num)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s failed from model, session_id:%lu,",
                       ATTR_MODEL_LABEL_NUM.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s failed from model, session_id:%lu,",
           ATTR_MODEL_LABEL_NUM.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  GELOGD("Label_num = %u", label_num);

  GELOGI("CreateRunContext: data_mem_base_ = %p, weight_mem_base_ = %p, memory_size = %lu, weight_size = %lu",
         data_mem_base_, weight_mem_base_, data_mem_size_, weight_mem_size_);

  PrintMemInfo();
  run_context_ = {};
  run_context_.sessionId = session_id;
  run_context_.dataMemSize = data_mem_size_;
  run_context_.dataMemBase = data_mem_base_;
  run_context_.mem_type_data_mem_size = mem_type_to_data_mem_size_;
  run_context_.mem_type_data_mem_base = mem_type_to_data_mem_base_;
  run_context_.weightMemSize = weight_mem_size_;
  run_context_.weightMemBase = weight_mem_base_;
  run_context_.weightsBuffer = buffer;
  return SUCCESS;
}

void RunContextUtil::PrintMemInfo() const {
  for (auto iter : mem_type_to_data_mem_base_) {
    GELOGD("CreateRunContext: memory type = %ld, data memory base = %p", iter.first, iter.second);
  }

  for (auto iter : mem_type_to_data_mem_size_) {
    GELOGD("CreateRunContext: memory type = %ld, data memory size = %lu", iter.first, iter.second);
  }
}

RunContext &RunContextUtil::GetRunContext() { return run_context_; }
}  // namespace ge
