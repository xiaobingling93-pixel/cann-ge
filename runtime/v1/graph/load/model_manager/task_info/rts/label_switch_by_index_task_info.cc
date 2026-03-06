/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/label_switch_by_index_task_info.h"

#include "common/checker.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace ge {
constexpr size_t kLabelSwitchIndexNum = 1U;

LabelSwitchByIndexTaskInfo::~LabelSwitchByIndexTaskInfo() {
  args_ = nullptr;
  index_value_ = nullptr;
}

Status LabelSwitchByIndexTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                        const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                        const IowAddrs &iow_addrs) {
  GELOGI("LabelSwitchByIndexTaskInfo Init Start.");
  (void)args;
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));

  // Get LabelSwitchByIndex task def
  const domi::LabelSwitchByIndexDef &label_switch = task_def.label_switch_by_index();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(label_switch.op_index());
  GE_CHECK_NOTNULL(op_desc, ", label switch op_index:%u out of range", label_switch.op_index());

  GE_ASSERT_TRUE((!(iow_addrs.input_logic_addrs.empty())),
                 "[Check][Param] Op:%s, input logic addr list is empty.", op_desc->GetName().c_str());

  GE_CHK_STATUS_RET_NOLOG(InitIndexValue(*davinci_model, iow_addrs));
  branch_max_ = label_switch.label_max();

  GE_ASSERT_SUCCESS(davinci_model->DisableZeroCopy(index_value_));

  std::vector<uint32_t> label_idx_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, label_idx_list)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail, attribute value not set",
                       ATTR_NAME_LABEL_SWITCH_LIST.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) failed, attribute value not set",
           ATTR_NAME_LABEL_SWITCH_LIST.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  const std::vector<rtLabel_t> &label_list = davinci_model->GetLabelList();
  const auto validate_label_id = [&label_list](const std::vector<uint32_t> &ids) -> bool {
    return std::all_of(ids.cbegin(), ids.cend(),
                       [&label_list](const uint32_t id) -> bool { return id < label_list.size(); });
  };
  if (label_idx_list.empty() || (label_idx_list.size() != branch_max_) || (!validate_label_id(label_idx_list))) {
    REPORT_INNER_ERR_MSG("E19999", "label_idx_list in op:%s(%s) is empty, or size:%zu != branch_max:%u, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), label_idx_list.size(), branch_max_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] %s(%s) label index size:%zu, task branch max:%u.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), label_idx_list.size(), branch_max_);
    return INTERNAL_ERROR;
  }

  std::vector<rtLabel_t> label_used(static_cast<size_t>(branch_max_), nullptr);
  for (size_t idx = 0U; idx < label_idx_list.size(); ++idx) {
    const uint32_t label_id = label_idx_list[idx];
    GE_CHECK_NOTNULL(label_list[static_cast<size_t>(label_id)]);
    label_used[idx] = label_list[static_cast<size_t>(label_id)];
  }

   // persistent_workspace内存申请适配好后删除
  const auto memory_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, static_cast<uint32_t>(args_size_));
  GELOGI("memory_type: %u", memory_type);
  GE_ASSERT_TRUE(!ge::MulOverflow(branch_max_, sizeof(rtLabelDevInfo), args_size_));
  args_ = davinci_model->MallocDynamicMemory(static_cast<size_t>(args_size_), memory_type);
  GE_ASSERT_NOTNULL(args_);

  const ArgsPlacement pls = (memory_type == RT_MEMORY_TS) ?
      ArgsPlacement::kArgsPlacementTs : ArgsPlacement::kArgsPlacementHbm;
  auto args_addr = (ValueToPtr(persistent_workspace[static_cast<uint32_t>(pls)].dev_addr) == nullptr) ?
      args_ : ValueToPtr(persistent_workspace[static_cast<uint32_t>(pls)].dev_addr);
  GE_CHK_RT_RET(rtLabelListCpy(label_used.data(), static_cast<uint32_t>(label_used.size()), args_addr, args_size_));

  GELOGI("LabelSwitchByIndexTaskInfo %s Init success, branch max: %u, logic stream id: %u, stream: %p.",
    op_desc->GetNamePtr(), task_def.stream_id(), stream_);
  return SUCCESS;
}

Status LabelSwitchByIndexTaskInfo::InitIndexValue(const DavinciModel &davinci_model, const IowAddrs &iow_addrs) {
  if (davinci_model.IsFeatureBaseRefreshable() || davinci_model.GetPhysicalMemoryRefreshable()) {
    index_value_ = ValueToPtr(iow_addrs.input_logic_addrs[0U].logic_addr);
  }

  return SUCCESS;
}

Status LabelSwitchByIndexTaskInfo::Distribute() {
  GELOGI("LabelSwitchByIndexTaskInfo Distribute Start, branch max: %u", branch_max_);
  GE_CHECK_NOTNULL(args_);
  GE_CHECK_NOTNULL(index_value_);
  if ((branch_max_ == 0U) || (args_size_ == 0U)) {
    REPORT_INNER_ERR_MSG("E19999", "branch_max_:%u or args_size_:%u is 0, check invalid", branch_max_, args_size_);
    GELOGE(PARAM_INVALID, "[Check][Param] branch max:%u, args size:%u invalid.", branch_max_, args_size_);
    return PARAM_INVALID;
  }

  const rtError_t rt_ret = rtLabelSwitchByIndex(index_value_, branch_max_, args_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtLabelSwitchByIndex failed, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelSwitchByIndex] failed, ret:%d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  is_support_redistribute_ = true;
  GELOGI("LabelSwitchByIndexTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

Status LabelSwitchByIndexTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                                     TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);

  const auto label_switch = task_def.label_switch_by_index();
  const uint32_t op_index = label_switch.op_index();
  GELOGI("Begin to calculate args, op_index is: %u", op_index);
  const auto op_desc = davinci_model->GetOpByIndex(op_index);
  GE_CHECK_NOTNULL(op_desc);
  GELOGI("Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  const size_t input_size = op_desc->GetInputsSize();
  std::vector<uint64_t> mem_types;
  const auto input_data_addrs = ModelUtils::GetInputAddrsValue(davinci_model->GetRuntimeParam(), op_desc, mem_types);
  if ((input_data_addrs.size() != kLabelSwitchIndexNum) || (input_size != kLabelSwitchIndexNum)) {
    REPORT_INNER_ERR_MSG("E19999", "Op:%s, input_data_addrs.size():%zu or input size:%zu != %u, check invalid",
                       op_desc->GetName().c_str(), input_data_addrs.size(), input_size, STREAM_SWITCH_INPUT_NUM);
    GELOGE(FAILED, "[Check][Param] Op:%s, input_data_addrs.size():%zu, input size:%zu != %u.",
           op_desc->GetName().c_str(), input_data_addrs.size(), input_size, STREAM_SWITCH_INPUT_NUM);
    return FAILED;
  }

  task_run_param.parsed_input_addrs.push_back({input_data_addrs[0U], mem_types[0U], false, {0}});
  index_value_ = ValueToPtr(input_data_addrs[0U]);
  GELOGD("parse task param, input_data_addrs %llu, mem_types %llu", input_data_addrs[0U], mem_types[0U]);

  const auto label_max = label_switch.label_max();
  const auto memory_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, static_cast<uint32_t>(args_size_));

  GE_ASSERT_TRUE(!ge::MulOverflow(label_max, sizeof(rtLabelDevInfo), args_size_));

  const ArgsPlacement pls = (memory_type == RT_MEMORY_TS) ?
      ArgsPlacement::kArgsPlacementTs : ArgsPlacement::kArgsPlacementHbm;
  task_run_param.persistent_workspace_descs.push_back({static_cast<int64_t>(args_size_), pls});

  return SUCCESS;
}

int64_t LabelSwitchByIndexTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::LabelSwitchByIndexDef &label_switch = task_def.label_switch_by_index();
  return static_cast<int64_t>(label_switch.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX, LabelSwitchByIndexTaskInfo);
}  // namespace ge
