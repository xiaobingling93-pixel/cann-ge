/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/model/pre_davinci_model.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/common/taskdown_common.h"
#include "common/preload/task_info/pre_generate_task_registry.h"

namespace ge {
namespace {
int64_t GetOpIndexKernel(const domi::TaskDef &task_def) {
  return task_def.kernel().context().op_index();
}

int64_t GetOpIndexDefault(const domi::TaskDef &task_def) {
  (void) task_def;
  return -1;
}

uint32_t GetOpIndexSwitchByIndex(const domi::TaskDef &task_def) {
  return task_def.label_switch_by_index().op_index();
}

uint32_t GetOpIndexLabelGoto(const domi::TaskDef &task_def) {
  return task_def.label_goto_ex().op_index();
}

const TypeToEngineNameToGetOpIndexFunc kKernelTypeToEngineNameToGetOpIndexFunc = {
    {static_cast<uint32_t>(ccKernelType::TE), {kPreEngineAiCore, &GetOpIndexKernel}},
    {static_cast<uint32_t>(ccKernelType::AI_CPU), {kPreEngineAiCpu, &GetOpIndexKernel}},
    {static_cast<uint32_t>(ccKernelType::CUST_AI_CPU), {kPreEngineAiCpu, &GetOpIndexKernel}}};
const TypeToEngineNameToGetOpIndexFunc kKernelTypeToNanoEngineNameToGetOpIndexFunc = {
    {static_cast<uint32_t>(ccKernelType::TE), {kPreEngineNanoAiCore, &GetOpIndexKernel}},
    {static_cast<uint32_t>(ccKernelType::AI_CPU), {kPreEngineNanoAiCpu, &GetOpIndexKernel}}};
const TypeToEngineNameToGetOpIndexFunc kTaskTypeToEngineNameToGetOpIndexFunc = {
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_EVENT_RECORD), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_EVENT_WAIT), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_SWITCH), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX), {kPreEngineNanoAiCore, &GetOpIndexSwitchByIndex}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_LABEL_GOTO), {kPreEngineNanoAiCore, GetOpIndexLabelGoto}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SWITCH), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_GOTO), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ADDR_ASYNC), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FUSION_START), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FUSION_END), {kPreEngineDefault, &GetOpIndexDefault}},
    {static_cast<uint32_t>(ModelTaskType::MODEL_TASK_END_GRAPH), {kPreEngineDefault, &GetOpIndexDefault}}};
}
void PreDavinciModel::Assign(const GeModelPtr &ge_model) {
  ge_model_ = ge_model;
}

void PreDavinciModel::DoReset() const {
  // other reset
  PreModelPartitionUtils::GetInstance().Reset();  // inst reset
}

Status PreDavinciModel::Init() {
  GELOGI("begin init pre davinci model.");
  GE_ASSERT_NOTNULL(ge_model_, "GeModel is null");
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph, "compute_graph is null");
  InitRuntimeParams();
  InitKernelOffset();
  DoReset();
  GE_CHK_STATUS_RET(InitNodes(compute_graph), "[Init][Nodes] failed, graph:%s.", compute_graph->GetName().c_str());
  GE_TIMESTAMP_START(DoTaskSink);
  GE_CHK_STATUS_RET(DoTaskSink(EngineType::kDefaultEngine), "[Call][DoTaskSink] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoTaskSink, "PreDavinciModel::DoTaskSink");

  GE_TIMESTAMP_START(DoPartitionProcess);
  GE_CHK_STATUS_RET(DoPartitionProcess(), "[Call][DoPartitionProcess] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoPartitionProcess, "PreDavinciModel::DoPartitionProcess");
  GELOGI("success init pre davinci model.");
  return SUCCESS;
}

void LogSegmentedMessage(const std::string& message, const std::string& prefix = "") {
    const size_t max_log_string_len = 800U;
    size_t index = 0U;

    std::string full_message = prefix + message;

    while (index < full_message.length()) {
        std::string segment = full_message.substr(index, max_log_string_len);
        GELOGE(FAILED, "%s", segment.c_str());
        index += max_log_string_len;
    }
}

Status PreDavinciModel::DoTaskSink(const EngineType engine_type) {
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def, "model_task_def is null");

  task_num_ = static_cast<uint32_t>(model_task_def->task_size());
  for (int32_t i = 0; i < static_cast<int32_t>(task_num_); ++i) {
    const auto &task_def = model_task_def->task(i);

    string engine_name;
    OpDescPtr op_desc = nullptr;
    std::string task_debug_info = task_def.ShortDebugString();

    if (GetEngineNameAndOpDesc(engine_type, task_def, engine_name, op_desc) != SUCCESS) {
      LogSegmentedMessage(task_debug_info, "[Call][GetEngineName] taskdef failed. Taskdef info: ");
      return FAILED;
    }

    PreTaskInput pre_task_input;
    pre_task_input.rts_param = runtime_param_;
    pre_task_input.names_to_bin_offset = names_to_bin_offset_;
    std::string op_name = (op_desc != nullptr ? op_desc->GetName() : task_debug_info);
    const auto func = PreGenerateTaskRegistry::GetInstance().FindPreGenerateTask(engine_name);

    if (func == nullptr) {
      std::stringstream error_ss;
      error_ss << "[Call][FindPreGenerateTask] op[" << op_name << "] can't find func from engine_name:" << engine_name;
      LogSegmentedMessage(error_ss.str());
      return FAILED;
    }

    const auto task_result = func(task_def, op_desc, pre_task_input);
    if (!task_result.status.IsSuccess()) {
      LogSegmentedMessage(task_result.status.GetErrorMessage(), "[Call][func] func execution failed, error message:");
      return FAILED;
    }

    PreModelPartitionUtils::GetInstance().AddPreTaskDescInfo(task_result.pre_task_desc_infos);
  }
  return SUCCESS;
}
Status PreDavinciModel::InitNodes(const ComputeGraphPtr &compute_graph) {
  const auto &nodes = compute_graph->GetAllNodes();
  for (size_t i = 0UL; i < nodes.size(); ++i) {
    const auto &node = nodes.at(i);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GELOGI("add op[%d] to list", op_desc->GetId());
    op_list_[op_desc->GetId()] = op_desc;
  }
  return SUCCESS;
}
void PreDavinciModel::InitRuntimeParams() {
  PreModelUtils::InitRuntimeParams(ge_model_, runtime_param_);
}
Status PreDavinciModel::DoPartitionProcess() {
  GE_CHK_STATUS_RET(
      PreModelPartitionUtils::GetInstance().InitTaskBuildMem(huge_stream_size_, runtime_param_.stream_num),
      "[Call][PreModelPartitionUtils][InitTaskBuildMem] failed.");
  // refresh partition data
  GE_CHK_STATUS_RET(PreModelPartitionUtils::GetInstance().PreparePartitionData(EngineType::kDefaultEngine),
                    "[Call][PreModelPartitionUtils][PreparePartitionData] failed.");
  return SUCCESS;
}
void PreDavinciModel::InitKernelOffset() {
  const TBEKernelStore tbe_kernel_store = ge_model_->GetTBEKernelStore();
  names_to_bin_offset_ = tbe_kernel_store.GetKernelOffset();
  GELOGI("names_to_bin_offset_ size:%u", names_to_bin_offset_.size());
}
// get Op
OpDescPtr PreDavinciModel::GetOpByIndex(const uint32_t op_index) const {
  const auto it = op_list_.find(static_cast<int64_t>(op_index));
  GE_ASSERT_TRUE(!(it == op_list_.end()));
  return it->second;
}

Status PreDavinciModel::GetEngineNameAndOpDesc(const EngineType engine_type, const domi::TaskDef &task_def,
                                               std::string &engine_name, OpDescPtr &op_desc) const {
  const auto task_type = task_def.type();
  uint32_t kernel_type = static_cast<uint32_t>(ccKernelType::INVALID);
  Status ret = FAILED;
  if (task_type == static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL)) {
    const domi::KernelDef &kernel_def = task_def.kernel();
    const domi::KernelContext &context = kernel_def.context();
    kernel_type = context.kernel_type();
    GELOGD("GetEngineName engine_type:%u, task_type:%u, kernel_type:%u.", engine_type, task_type, kernel_type);
    switch (engine_type) {
      case EngineType::kDefaultEngine:
        ret = GetEngineNameAndOpDescByType(kernel_type, kKernelTypeToEngineNameToGetOpIndexFunc, task_def, engine_name, op_desc);
        break;
      case EngineType::kNanoEngine:
        ret = GetEngineNameAndOpDescByType(kernel_type, kKernelTypeToNanoEngineNameToGetOpIndexFunc, task_def, engine_name, op_desc);
        break;
      default:
        GELOGE(FAILED, "there are unsupported engine_type in the model, engine_type:%u, kernel_type:%u.", engine_type,
               kernel_type);
        break;
    }
  } else {
    GELOGD("GetEngineName engine_type:%u, task_type:%u.", engine_type, task_type);
    switch (engine_type) {
      case EngineType::kDefaultEngine:
      case EngineType::kNanoEngine:
        ret = GetEngineNameAndOpDescByType(task_type, kTaskTypeToEngineNameToGetOpIndexFunc, task_def, engine_name, op_desc);
        break;
      default:
        GELOGE(FAILED, "there are unsupported engine_type in the model, engine_type:%u, task_type:%u.", engine_type,
               task_type);
        break;
    }
  }
  if (ret == FAILED) {
    GELOGE(FAILED, "[Call] there are unsupported task in the model, engine_type:%u, task_type:%u, kernel_type:%u.",
           engine_type, task_type, kernel_type);
    return FAILED;
  }
  GELOGD("success get engine name[%s]", engine_name.c_str());
  return SUCCESS;
}

Status PreDavinciModel::GetEngineNameAndOpDescByType(const uint32_t type,
                                                     const TypeToEngineNameToGetOpIndexFunc &type_to_engine_name_to_get_op_index_func,
                                                     const domi::TaskDef &task_def,
                                                     std::string &engine_name,
                                                     OpDescPtr &op_desc) const {
  const auto it = type_to_engine_name_to_get_op_index_func.find(type);
  if (it == type_to_engine_name_to_get_op_index_func.end()) {
    GELOGE(FAILED, "[Call][GetEngineNameAndOpDescByType] failed find engine name from type:%u.", type);
    return FAILED;
  }

  engine_name = it->second.first;
  op_desc = nullptr;
  int64_t op_index =  it->second.second(task_def);
  if (op_index != -1) {
    op_desc = GetOpByIndex(static_cast<uint32_t>(op_index));
    GE_ASSERT_NOTNULL(op_desc, "[Call][GetOpByIndex] get op fail, op index is %u", op_index);
  }
  return SUCCESS;
}
}  // namespace ge