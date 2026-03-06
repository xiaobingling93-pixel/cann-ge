/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/stream_switch_task_info.h"

#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace ge {
namespace {
constexpr size_t kTrueBranchStreamNum_1 = 1U;
}  // namespace

Status StreamSwitchTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                  const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                  const IowAddrs &iow_addrs) {
  GELOGI("StreamSwitchTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  const auto &stream_switch_def = task_def.stream_switch();
  op_index_ = stream_switch_def.op_index();
  const OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index_);
  GE_CHECK_NOTNULL(op_desc, ", stream switch op_index:%u out of range", op_index_);
  op_desc_ = op_desc;

  GE_CHK_STATUS_RET_NOLOG(InitInputValueAndType(op_desc, iow_addrs));

  uint32_t cond = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_STREAM_SWITCH_COND, cond)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail", ATTR_NAME_STREAM_SWITCH_COND.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) fail",
           ATTR_NAME_STREAM_SWITCH_COND.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  cond_ = static_cast<rtCondition_t>(cond);

  std::vector<uint32_t> active_stream_list;
  if ((!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) ||
      (active_stream_list.size() != kTrueBranchStreamNum_1)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s fail, active_stream_list.size():%zu",
                       ATTR_NAME_ACTIVE_STREAM_LIST.c_str(), op_desc->GetName().c_str(), active_stream_list.size());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s fail, active_stream_list.size():%zu",
           ATTR_NAME_ACTIVE_STREAM_LIST.c_str(), op_desc->GetName().c_str(), active_stream_list.size());
    return INTERNAL_ERROR;
  }

  const auto &stream_list = davinci_model_->GetStreamList();
  const size_t true_stream_index = static_cast<size_t>(active_stream_list.front());
  if (true_stream_index >= stream_list.size()) {
    REPORT_INNER_ERR_MSG("E19999", "active_stream_index:%zu in op:%s(%s) >= stream list size:%zu in model, check invalid",
                       true_stream_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), stream_list.size());
    GELOGE(INTERNAL_ERROR, "[Check][Param] active_stream_index:%zu in op:%s(%s) >= stream list size:%zu in model",
           true_stream_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), stream_list.size());
    return INTERNAL_ERROR;
  }

  true_stream_ = stream_list[true_stream_index];
  true_stream_id_ = stream_switch_def.true_stream_id();
  GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(input_ptr_));
  GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(value_ptr_));

  GELOGI("InitStreamSwitchTaskInfo %s Init Success, cond:%d, trueStream:%p, trueStreamID:%u, datatype:%d, "
         "logic stream id: %u, stream: %p.",
         op_desc->GetNamePtr(), cond_, true_stream_, true_stream_id_, data_type_, task_def.stream_id(), stream_);

  return SUCCESS;
}

Status StreamSwitchTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("StreamSwitchTaskInfo Distribute Start.");
  SetTaskTag(op_desc_->GetName().c_str());
  const rtError_t rt_ret = rtStreamSwitchEx(input_ptr_, cond_, value_ptr_, true_stream_, stream_, data_type_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtStreamSwitchEx fail, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtStreamSwitchEx] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
  }
  is_support_redistribute_ = true;
  GELOGI("StreamSwitchTaskInfo Distribute Success. cond: %d, true stream: %p, datatype: %d, stream: %p.",
    cond_, true_stream_, data_type_, stream_);
  return SUCCESS;
}

Status StreamSwitchTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                               TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  const auto &stream_switch_def = task_def.stream_switch();
  const uint32_t op_index = stream_switch_def.op_index();
  GELOGI("Begin to calculate args, op_index is: %u", op_index);
  const auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  const size_t input_size = op_desc->GetInputsSize();
  std::vector<uint64_t> mem_types;
  const auto input_data_addrs = ModelUtils::GetInputAddrsValue(davinci_model->GetRuntimeParam(), op_desc, mem_types);
  if ((input_data_addrs.size() != STREAM_SWITCH_INPUT_NUM) || (input_size != STREAM_SWITCH_INPUT_NUM)) {
    REPORT_INNER_ERR_MSG("E19999", "Op:%s, input_data_addrs.size():%zu or input size:%zu != %u, check invalid",
                       op_desc->GetName().c_str(), input_data_addrs.size(), input_size, STREAM_SWITCH_INPUT_NUM);
    GELOGE(FAILED, "[Check][Param] Op:%s, input_data_addrs.size():%zu, input size:%zu != %u.",
           op_desc->GetName().c_str(), input_data_addrs.size(), input_size, STREAM_SWITCH_INPUT_NUM);
    return FAILED;
  }

  task_run_param.parsed_input_addrs.push_back({input_data_addrs[0U], mem_types[0U], false, {0}});
  task_run_param.parsed_input_addrs.push_back({input_data_addrs[1U], mem_types[1U], false, {0}});
  input_ptr_ = ValueToPtr(input_data_addrs[0U]);
  value_ptr_ = ValueToPtr(input_data_addrs[1U]);
  GELOGD("parse task param, input_addrs[0] %llu, mem_types[0] %llu, input_addrs[1] %llu, mem_types[1] %llu",
         input_data_addrs[0U], mem_types[0U], input_data_addrs[1U], mem_types[1U]);

  return SUCCESS;
}

Status StreamSwitchTaskInfo::InitInputValueAndType(const OpDescPtr &op_desc, const IowAddrs &iow_addrs) {
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE)) {
    int64_t data_type = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_SWITCH_DATA_TYPE, data_type)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail, attribute value not int",
                         ATTR_NAME_SWITCH_DATA_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Get][Attr] %s in op:%s(%s) fail, attribute value not int",
             ATTR_NAME_SWITCH_DATA_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    data_type_ = static_cast<rtSwitchDataType_t>(data_type);
  }

  GE_ASSERT_TRUE((iow_addrs.input_logic_addrs.size() == STREAM_SWITCH_INPUT_NUM),
                 "[Check][Param] Op:%s(%s) input logic addrs list size:%zu != %d", op_desc->GetName().c_str(),
                 op_desc->GetType().c_str(), iow_addrs.input_logic_addrs.size(), STREAM_SWITCH_INPUT_NUM);

  if (davinci_model_->IsFeatureBaseRefreshable() || davinci_model_->GetPhysicalMemoryRefreshable()) {
    input_ptr_ = ValueToPtr(iow_addrs.input_logic_addrs[0U].logic_addr);
    value_ptr_ = ValueToPtr(iow_addrs.input_logic_addrs[1U].logic_addr);
  }

  return SUCCESS;
}

int64_t StreamSwitchTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const auto &stream_switch_def = task_def.stream_switch();
  return static_cast<int64_t>(stream_switch_def.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_STREAM_SWITCH, StreamSwitchTaskInfo);
}  // namespace ge
