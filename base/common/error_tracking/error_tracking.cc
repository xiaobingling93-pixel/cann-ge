/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "error_tracking.h"
#include "runtime/rt.h"
#include "framework/common/debug/log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "rt_error_codes.h"
#include "base/err_msg.h"

namespace {
void GetOpOriginName(const ge::OpDescPtr &op_desc, std::string &origin_op_name) {
  std::vector<std::string> original_names;
  origin_op_name = op_desc->GetName();
  if (ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names) &&
      !original_names.empty()) {
    origin_op_name.clear();
    for (const auto &name : original_names) {
      origin_op_name += name;
      origin_op_name += ";";
    }
    origin_op_name = origin_op_name.substr(0U, origin_op_name.length() - 1U);
  }
}

}
namespace ge {
constexpr uint32_t kMaxGraphOpDescInfoNum = 2048U * 2048U;

ErrorTracking &ErrorTracking::GetInstance() {
  static ErrorTracking instance;
  return instance;
}

ErrorTracking::ErrorTracking() {
}
void ErrorTracking::AddTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key,
                                      std::map<TaskKey, ErrorTrackingOpInfo> &map, uint32_t max_count) const {
  if (op != nullptr) {
    GELOGD("Add task opdesc info, opname %s, task_id=%u, stream_id=%u, thread_id %u, context_id %u.",
           op->GetName().c_str(), key.GetTaskId(), key.GetStreamId(), key.GetThreadId(), key.GetContextId());
    std::string origin_name;
    GetOpOriginName(op, origin_name);
    if (map.size() >= max_count) {
      (void)map.erase(map.begin());
    }
    ErrorTrackingOpInfo info = {op->GetName(), origin_name};
    map[key] = info;
  }
}

bool ErrorTracking::GetGraphTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, ErrorTrackingOpInfo &op_info) {
  TaskKey key(task_id, stream_id);
  return GetTaskOpdescInfo(key, graph_task_to_op_info_, op_info);
}

bool ErrorTracking::GetGraphTaskOpdescInfo(const TaskKey &key, ErrorTrackingOpInfo &op_info) {
  return GetTaskOpdescInfo(key, graph_task_to_op_info_, op_info);
}

bool ErrorTracking::GetSingleOpTaskOpdescInfo(const uint32_t task_id, const uint32_t stream_id, ErrorTrackingOpInfo &op_info) {
  TaskKey key(task_id, stream_id);
  return GetTaskOpdescInfo(key, single_op_task_to_op_info_, op_info);
}

bool ErrorTracking::GetTaskOpdescInfo(const TaskKey &key, const std::map<TaskKey, ErrorTrackingOpInfo> &map,
                                      ErrorTrackingOpInfo &op_info) {
    const std::lock_guard<std::mutex> lk(mutex_);
    auto iter = map.find(key);
    if (iter != map.end()) {
      op_info = iter->second;
      return true;
    }
    return false;
}

bool ErrorTracking::GetTaskOpdescInfo(const TaskKey &key,
                                      const std::map<uint32_t, std::map<TaskKey, ErrorTrackingOpInfo>> &map,
                                      ErrorTrackingOpInfo &op_info)
{
  const std::lock_guard<std::mutex> lk(mutex_);
  for (const auto& pair : map) {
    auto iter = pair.second.find(key);
    if (iter != pair.second.end()) {
      op_info = iter->second;
      return true;
    }
  }
  return false;
}

void ErrorTracking::SaveGraphTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id,
  const uint32_t model) {
    TaskKey key(task_id, stream_id);
    const std::lock_guard<std::mutex> lk(mutex_);
    AddTaskOpdescInfo(op, key, graph_task_to_op_info_[model], kMaxGraphOpDescInfoNum);
}

void ErrorTracking::UpdateTaskId(const uint32_t old_task_id, const uint32_t new_task_id, const uint32_t stream_id, const uint32_t model) {
    TaskKey old_key(old_task_id, stream_id);
    TaskKey new_key(new_task_id, stream_id);

    const std::lock_guard<std::mutex> lk(mutex_);

    auto model_it = graph_task_to_op_info_.find(model);
    if (model_it == graph_task_to_op_info_.end()) {
        GELOGW("[Update][TaskId] failed, model %u not found", model);
        return;
    }

    auto &task_map = model_it->second;
    auto it = task_map.find(old_key);
    if (it != task_map.end()) {
        const std::string opname = it->second.op_name;
        GELOGD("Update task id, old: %u -> new: %u, stream_id: %u, model: %u, opname: %s",
               old_task_id, new_task_id, stream_id, model, opname.c_str());

        ErrorTrackingOpInfo op_info = it->second;
        (void)task_map.erase(it);
        task_map[new_key] = op_info;
    } else {
        GELOGW("Failed to update task id, old task id %u not found in model %u", old_task_id, model);
    }
}

void ErrorTracking::SaveGraphTaskOpdescInfo(const OpDescPtr &op, const TaskKey &key, const uint32_t model) {
  const std::lock_guard<std::mutex> lk(mutex_);
  AddTaskOpdescInfo(op, key, graph_task_to_op_info_[model], kMaxGraphOpDescInfoNum);
}

void ErrorTracking::SaveSingleOpTaskOpdescInfo(const OpDescPtr &op, const uint32_t task_id, const uint32_t stream_id) {
  TaskKey key(task_id, stream_id);
  const std::lock_guard<std::mutex> lk(mutex_);
  AddTaskOpdescInfo(op, key, single_op_task_to_op_info_, single_op_max_count_);
}

void ErrorTracking::ClearUnloadedModelOpdescInfo(const uint32_t model) {
  const std::lock_guard<std::mutex> lk(mutex_);
  auto it = graph_task_to_op_info_.find(model);
  if (it != graph_task_to_op_info_.end()) {
    (void)graph_task_to_op_info_.erase(it);
  }
}

void ErrorTrackingCallback(rtExceptionInfo *const exception_data) {
  if (exception_data == nullptr) {
    return;
  }
  if ((exception_data->retcode == ACL_ERROR_RT_AICORE_OVER_FLOW) ||
    (exception_data->retcode == ACL_ERROR_RT_AIVEC_OVER_FLOW) || (exception_data->retcode == ACL_ERROR_RT_OVER_FLOW)) {
    return;
  }
  GELOGI("ErrorTracking callbak in, task_id %u, stream_id %u.", exception_data->taskid, exception_data->streamid);
  ErrorTrackingOpInfo op_info;
  bool founded = false;
  if (exception_data->expandInfo.type == RT_EXCEPTION_FFTS_PLUS) {
    const uint32_t context_id = static_cast<uint32_t>(exception_data->expandInfo.u.fftsPlusInfo.contextId);
    const uint32_t thread_id = static_cast<uint32_t>(exception_data->expandInfo.u.fftsPlusInfo.threadId);
    TaskKey key(exception_data->taskid, exception_data->streamid, context_id, thread_id);
    founded = ErrorTracking::GetInstance().GetGraphTaskOpdescInfo(key, op_info);
  } else {
    founded = ErrorTracking::GetInstance().GetGraphTaskOpdescInfo(exception_data->taskid, exception_data->streamid, op_info);
    if (!founded) {
      founded = ErrorTracking::GetInstance().GetSingleOpTaskOpdescInfo(exception_data->taskid, exception_data->streamid, op_info);
    }
  }
  if (founded) {
    GELOGE(FAILED, "Error happened, origin_op_name [%s], op_name [%s], task_id %u, stream_id %u.",
           op_info.op_origin_name.c_str(), op_info.op_name.c_str(), exception_data->taskid, exception_data->streamid);
    REPORT_INNER_ERR_MSG("E18888", "Op execute failed. origin_op_name [%s], op_name [%s], "
      "error_info: task_id %u, stream_id %u, tid %u, device_id %u, retcode 0x%x",
      op_info.op_origin_name.c_str(), op_info.op_name.c_str(), exception_data->taskid, exception_data->streamid,
      exception_data->tid, exception_data->deviceid, exception_data->retcode);
  }
}

uint32_t RegErrorTrackingCallBack() {
  GE_CHK_RT_RET(rtRegTaskFailCallbackByModule("GeErrorTracking", &ErrorTrackingCallback));
  return 0;
}
}
