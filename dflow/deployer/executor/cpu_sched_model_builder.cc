/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/cpu_sched_model_builder.h"
#include "securec.h"
#include "framework/common/debug/log.h"
#include "common/debug/ge_log.h"
#include "graph/def_types.h"
#include "common/dump/dump_manager.h"
#include "common/checker.h"
#include "executor/cpu_sched_model.h"
#include "executor/cpu_id_resource_manager.h"
#include "executor/dynamic_model_executor.h"
#include "executor/sched_task_info.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr const char_t *kCpuSdTaskModelEnqueue = "modelEnqueue";
constexpr const char_t *kCpuSdTaskModelEnqueueBuff = "modelEnqueueBuff";
constexpr const char_t *kCpuSdTaskWaitEndGraph = "modelWaitEndGraph";
constexpr const char_t *kCpuSdTaskModelDequeue = "modelDequeue";
constexpr const char_t *kCpuSdTaskModelBatchDequeue = "modelBatchDequeue";
constexpr const char_t *kCpuSdTaskModelBatchDequeueBuff = "modelBatchDequeueBuff";
constexpr const char_t *kCpuSdTaskMarkStep = "markStep";
constexpr const char_t *kCpuSdTaskModelRepeat = "modelRepeat";
constexpr const char_t *kCpuSdTaskActivateModel = "activeModel";
constexpr const char_t *kGatherDequeue = "gatherDequeue";

constexpr uint32_t kQueueFlagInput = 0U;
constexpr uint32_t kQueueFlagOutput = 1U;
constexpr uint32_t kClientQueueFlagInput = 2U;
constexpr uint32_t kClientQueueFlagOutput = 3U;
constexpr uint32_t kMaxDumpStepStrLen = 1024U;
constexpr uint32_t kReservedParamsNum = 30U;

struct QueueOpTaskParam {
  uint32_t queue_id;
  uint64_t mbuf_addr;
};

// for client queue mode
struct QueueOpBuffTaskParam {
  uint32_t queue_id;
  int32_t device_id;
  uint64_t mbuf_addr;
};

struct ModelBatchDequeueTaskParam {
  uint32_t num_inputs;
  uint32_t align_interval;
  uint64_t align_offsets_addr;
  uint64_t queue_ids_addr;
  uint64_t mbuf_addrs_addr;
};

// for client queue mode
struct ModelBatchDequeueBuffTaskParam {
  uint32_t num_inputs;
  uint32_t align_interval;
  uint64_t align_offsets_addr; // value equal to 0 when align is false
  uint64_t queue_ids_addr;
  uint64_t mbuf_addrs_addr;
  uint64_t device_ids_addr;
};

struct MarkStepTaskParam {
  uint32_t group_total_count{1U};
  uint32_t group_index{0U};
  uint32_t group_policy{0U};   // load balance policy
  uint64_t step_id_addr{0UL};  // current step id addr
  uint64_t rsv{0UL};           // for aicpu use
  uint8_t is_not_head{0};      // 非头节点为1
  uint64_t reserved[kReservedParamsNum]{0U};
  char_t dump_step[kMaxDumpStepStrLen]{'\0'};
};
}  // namespace

CpuSchedModelBuilder::CpuSchedModelBuilder(CpuSchedModel &model) : model_(model) {}

uint8_t *CpuSchedModelBuilder::NewTask(const char_t *kernel_name, size_t param_size, uint32_t stream_id) {
  model_.task_params_.emplace_back(std::vector<uint8_t>(param_size));
  auto &param_base = model_.task_params_.back();
  ::ModelTaskInfo task_info{};
  task_info.taskId = task_id_gen_++;
  task_info.kernelName = PtrToValue(kernel_name);
  task_info.paraBase = PtrToValue(param_base.data());
  model_.tasks_[stream_id].emplace_back(task_info);
  return param_base.data();
}

void CpuSchedModelBuilder::AddDequeueTasks(uint32_t stream_id) {
  for (const auto &input_queue_info : input_local_queue_infos_) {
    model_.queues_.emplace_back(::ModelQueueInfo{input_queue_info.first.queue_id, kQueueFlagInput});
    AddQueueOpTask(kCpuSdTaskModelDequeue, input_queue_info.first.queue_id, input_queue_info.second, stream_id);
  }
}

Status CpuSchedModelBuilder::AddMarkStepTask(uint32_t stream_id, bool is_head) {
  auto task_param =
      reinterpret_cast<MarkStepTaskParam *>(NewTask(kCpuSdTaskMarkStep, sizeof(MarkStepTaskParam), stream_id));
  task_param->group_total_count = model_queue_param_.group_total_count;
  task_param->group_index = model_queue_param_.group_index;
  task_param->group_policy = model_queue_param_.group_policy;
  const auto dump_step = DumpManager::GetInstance().GetDumpProperties(kInferSessionId).GetDumpStep();
  const auto ret = strcpy_s(task_param->dump_step, sizeof(task_param->dump_step), dump_step.c_str());
  if (ret != EOK) {
    REPORT_INNER_ERR_MSG("E19999", "Call strcpy failed, dump_step: %s is too long", dump_step.c_str());
    GELOGE(FAILED, "[Call][strcpy_s] strcpy failed, result: %d, dump_step: %s", ret, dump_step.c_str());
    return FAILED;
  }
  void * const global_step_addr = ValueToPtr(global_step_);
  if (global_step_addr != nullptr) {
    DF_CHK_ACL_RET(aclrtMemset(global_step_addr, sizeof(uint64_t), 0U, sizeof(uint64_t)));
  }
  task_param->step_id_addr = global_step_;
  task_param->is_not_head = is_head ? 0 : 1;
  GELOGI("[Add][MarkStep] group_total_count[%u], group_index[%u], dump_step: %s, is_not_head: %d.",
         task_param->group_total_count, task_param->group_index, task_param->dump_step, task_param->is_not_head);
  return SUCCESS;
}

void CpuSchedModelBuilder::AddActivateTask(uint32_t stream_id) {
  auto task_param = NewTask(kCpuSdTaskActivateModel, sizeof(uint32_t), stream_id);
  *reinterpret_cast<uint32_t *>(task_param) = model_.model_info_.modelId;
}

void CpuSchedModelBuilder::AddWaitEndGraph(uint32_t stream_id) {
  auto task_param = NewTask(kCpuSdTaskWaitEndGraph, sizeof(uint32_t), stream_id);
  *reinterpret_cast<uint32_t *>(task_param) = model_.model_info_.modelId;
}

void CpuSchedModelBuilder::AddEnqueueTasks(uint32_t stream_id) {
  for (const auto &output_queue_info : output_local_queue_infos_) {
    model_.queues_.emplace_back(::ModelQueueInfo{output_queue_info.first.queue_id, kQueueFlagOutput});
    AddQueueOpTask(kCpuSdTaskModelEnqueue, output_queue_info.first.queue_id, output_queue_info.second, stream_id);
  }
  for (const auto &output_queue_info : output_client_queue_infos_) {
    model_.queues_.emplace_back(::ModelQueueInfo{output_queue_info.first.queue_id, kClientQueueFlagOutput});
    AddQueueBuffOpTask(kCpuSdTaskModelEnqueueBuff, output_queue_info.first, output_queue_info.second, stream_id);
  }
}

void CpuSchedModelBuilder::AddModelRepeat(uint32_t stream_id) {
  auto task_param = NewTask(kCpuSdTaskModelRepeat, sizeof(uint32_t), stream_id);
  *reinterpret_cast<uint32_t *>(task_param) = model_.model_info_.modelId;
}

void CpuSchedModelBuilder::AddQueueOpTask(const char_t *kernel_name, uint32_t queue_id,
                                          uintptr_t mbuf_addr, uint32_t stream_id) {
  auto task_param = reinterpret_cast<QueueOpTaskParam *>(
      NewTask(kernel_name, sizeof(QueueOpTaskParam), stream_id));
  task_param->queue_id = queue_id;
  task_param->mbuf_addr = mbuf_addr;
}

void CpuSchedModelBuilder::AddQueueBuffOpTask(const char_t *kernel_name, const QueueAttrs &queue_attrs,
                                              uintptr_t mbuf_addr, uint32_t stream_id) {
  auto task_param = reinterpret_cast<QueueOpBuffTaskParam *>(
      NewTask(kernel_name, sizeof(QueueOpBuffTaskParam), stream_id));
  task_param->queue_id = queue_attrs.queue_id;
  task_param->device_id = queue_attrs.device_id;
  task_param->mbuf_addr = mbuf_addr;
}


void CpuSchedModelBuilder::AddBatchDequeueOpTask(uint32_t stream_id) {
  const auto num_inputs = static_cast<uint32_t>(input_local_queue_infos_.size());
  // kernel_args|mbuf_addr_buffer|queue_ids_buffer|align_offsets_buffer
  auto arg_size = sizeof(ModelBatchDequeueTaskParam);
  const auto mbuf_addrs_offset = arg_size;
  const auto mbuf_addrs_size = sizeof(uint64_t) * num_inputs;
  arg_size += mbuf_addrs_size;
  const auto queue_ids_offset = arg_size;
  const auto queue_ids_size = sizeof(uint32_t) * num_inputs;
  arg_size += queue_ids_size;
  const auto align_offsets_offset = arg_size;
  const auto align_offsets_size = sizeof(uint32_t) * num_inputs;
  arg_size += sizeof(uint32_t) * align_offsets_size;

  auto kernel_args = reinterpret_cast<ModelBatchDequeueTaskParam *>(
      NewTask(kCpuSdTaskModelBatchDequeue, arg_size, stream_id));
  kernel_args->num_inputs = num_inputs;
  kernel_args->align_interval = align_interval_;
  kernel_args->align_offsets_addr = PtrToValue(kernel_args) + align_offsets_offset;
  kernel_args->queue_ids_addr = PtrToValue(kernel_args) + queue_ids_offset;
  kernel_args->mbuf_addrs_addr = PtrToValue(kernel_args) + mbuf_addrs_offset;
  for (size_t i = 0; i < input_local_queue_infos_.size(); ++i) {
    const auto &input_queue_info = input_local_queue_infos_[i];
    model_.queues_.emplace_back(::ModelQueueInfo{input_queue_info.first.queue_id, kQueueFlagInput});
    reinterpret_cast<uint32_t *>(kernel_args->align_offsets_addr)[i] = align_offsets_[i];  // checked by caller
    reinterpret_cast<uint32_t *>(kernel_args->queue_ids_addr)[i] = input_queue_info.first.queue_id;
    reinterpret_cast<uint64_t *>(kernel_args->mbuf_addrs_addr)[i] = input_queue_info.second;
  }
}

void CpuSchedModelBuilder::AddBatchDequeueBuffOpTask(uint32_t stream_id) {
  if (input_client_queue_infos_.empty()) {
    GELOGD("There is no client queue to send batch dequeue buff task.");
    return;
  }
  const auto num_inputs = static_cast<uint32_t>(input_client_queue_infos_.size());
  // kernel_args|mbuf_addr_buffer|queue_ids_buffer|align_offsets_buffer
  auto arg_size = sizeof(ModelBatchDequeueBuffTaskParam);

  size_t align_offsets_offset = 0UL;
  if (has_align_attr_) {
    align_offsets_offset = arg_size;
    const auto align_offsets_size = sizeof(uint32_t) * num_inputs;
    arg_size += align_offsets_size;
  }

  const auto queue_ids_offset = arg_size;
  const auto queue_ids_size = sizeof(uint32_t) * num_inputs;
  arg_size += queue_ids_size;

  const auto mbuf_addrs_offset = arg_size;
  const auto mbuf_addrs_size = sizeof(uint64_t) * num_inputs;
  arg_size += mbuf_addrs_size;

  const auto device_id_offset = arg_size;
  const auto device_id_size = sizeof(int32_t) * num_inputs;
  arg_size += device_id_size;

  auto kernel_args = reinterpret_cast<ModelBatchDequeueBuffTaskParam *>(
      NewTask(kCpuSdTaskModelBatchDequeueBuff, arg_size, stream_id));
  kernel_args->num_inputs = num_inputs;
  kernel_args->align_interval = align_interval_;
  kernel_args->align_offsets_addr = has_align_attr_ ? (PtrToValue(kernel_args) + align_offsets_offset) :
                                                      align_offsets_offset;
  kernel_args->queue_ids_addr = PtrToValue(kernel_args) + queue_ids_offset;
  kernel_args->mbuf_addrs_addr = PtrToValue(kernel_args) + mbuf_addrs_offset;
  kernel_args->device_ids_addr = PtrToValue(kernel_args) + device_id_offset;
  for (size_t i = 0; i < input_client_queue_infos_.size(); ++i) {
    const auto &input_queue_info = input_client_queue_infos_[i];
    model_.queues_.emplace_back(::ModelQueueInfo{input_queue_info.first.queue_id, kClientQueueFlagInput});
    if (has_align_attr_) {
      reinterpret_cast<uint32_t *>(kernel_args->align_offsets_addr)[i] = align_offsets_[i];
    }
    reinterpret_cast<uint32_t *>(kernel_args->queue_ids_addr)[i] = input_queue_info.first.queue_id;
    reinterpret_cast<uint64_t *>(kernel_args->mbuf_addrs_addr)[i] = input_queue_info.second;
    reinterpret_cast<int32_t *>(kernel_args->device_ids_addr)[i] = input_queue_info.first.device_id;
  }
}

void CpuSchedModelBuilder::AddInputQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr) {
  input_local_queue_infos_.emplace_back(std::move(queue_attrs), mbuf_addr);
}

void CpuSchedModelBuilder::AddOutputQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr) {
  output_local_queue_infos_.emplace_back(std::move(queue_attrs), mbuf_addr);
}

void CpuSchedModelBuilder::AddInputClientQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr) {
  input_client_queue_infos_.emplace_back(std::move(queue_attrs), mbuf_addr);
}

void CpuSchedModelBuilder::AddOutputClientQueue(QueueAttrs queue_attrs, uintptr_t mbuf_addr) {
  output_client_queue_infos_.emplace_back(std::move(queue_attrs), mbuf_addr);
}

void CpuSchedModelBuilder::AddGatherDequeueTask(uint32_t stream_id) {
  const auto num_inputs = static_cast<uint32_t>(input_client_queue_infos_.size() + input_local_queue_infos_.size());
  GELOGD("Prepare gather dequeue task for input queue num %zu and client queue num %zu",
         input_local_queue_infos_.size(), input_client_queue_infos_.size());
  // memory layout: kernel_args | mbuf_addrs_buffer | mbufs_buffer | queue_ids_buffer | align_offsets_buffer
  auto arg_size = sizeof(GatherDequeueParam);

  const auto queue_ids_offset = arg_size;
  const auto queue_ids_size = sizeof(uint32_t) * num_inputs;
  arg_size += queue_ids_size;

  const auto mbuf_addrs_offset = arg_size;
  const auto mbuf_addrs_size = sizeof(uint64_t) * num_inputs;
  arg_size += mbuf_addrs_size;

  const auto device_id_offset = arg_size;
  const auto device_id_size = sizeof(uint32_t) * num_inputs;
  arg_size += device_id_size;

  const auto device_type_offset = arg_size;
  const auto device_type_size = sizeof(uint32_t) * num_inputs;
  arg_size += device_type_size;

  auto kernel_args = reinterpret_cast<GatherDequeueParam *>(
      NewTask(kGatherDequeue, arg_size, stream_id));
  kernel_args->input_nums = num_inputs;
  kernel_args->inputs_align_max_cache_num = input_align_attrs_.align_max_cache_num;
  kernel_args->inputs_align_timeout = input_align_attrs_.align_timeout;
  kernel_args->inputs_align_drop_out = static_cast<uint32_t>(input_align_attrs_.drop_when_not_align);
  kernel_args->queue_ids_addr = PtrToValue(kernel_args) + queue_ids_offset;
  kernel_args->mbuf_addrs_addr = PtrToValue(kernel_args) + mbuf_addrs_offset;
  kernel_args->queue_device_ids_addr = PtrToValue(kernel_args) + device_id_offset;
  kernel_args->queue_device_type_addr = PtrToValue(kernel_args) + device_type_offset;
  for (size_t i = 0; i < input_client_queue_infos_.size(); ++i) {
    const auto &input_queue_info = input_client_queue_infos_[i];
    model_.queues_.emplace_back(::ModelQueueInfo{input_queue_info.first.queue_id, kClientQueueFlagInput});
    reinterpret_cast<uint32_t *>(kernel_args->queue_ids_addr)[i] = input_queue_info.first.queue_id;
    reinterpret_cast<uint64_t *>(kernel_args->mbuf_addrs_addr)[i] = input_queue_info.second;
    reinterpret_cast<uint32_t *>(kernel_args->queue_device_ids_addr)[i] = input_queue_info.first.device_id;
    reinterpret_cast<uint32_t *>(kernel_args->queue_device_type_addr)[i] = input_queue_info.first.device_type;
  }
  for (size_t i = 0; i < input_local_queue_infos_.size(); ++i) {
    const auto &input_queue_info = input_local_queue_infos_[i];
    model_.queues_.emplace_back(::ModelQueueInfo{input_queue_info.first.queue_id, kQueueFlagInput});
    const auto index = input_client_queue_infos_.size() + i;
    reinterpret_cast<uint32_t *>(kernel_args->queue_ids_addr)[index] = input_queue_info.first.queue_id;
    reinterpret_cast<uint64_t *>(kernel_args->mbuf_addrs_addr)[index] = input_queue_info.second;
    reinterpret_cast<uint32_t *>(kernel_args->queue_device_ids_addr)[index] = input_queue_info.first.device_id;
    reinterpret_cast<uint32_t *>(kernel_args->queue_device_type_addr)[index] = input_queue_info.first.device_type;
  }
  GELOGD("Finish to prepare gather dequeue task, queue num:%zu, max cache num:%u, timeout:%d, drop flag:%u",
         kernel_args->input_nums, kernel_args->inputs_align_max_cache_num, kernel_args->inputs_align_timeout,
         kernel_args->inputs_align_drop_out);
}

void CpuSchedModelBuilder::AddDequeueTasksByAttrs(uint32_t stream_id) {
    if ((input_align_attrs_.align_max_cache_num != 0U) &&
        (input_client_queue_infos_.size() + input_local_queue_infos_.size() > 1UL)) {
      AddGatherDequeueTask(stream_id);
    } else {
      if (has_align_attr_) {
        AddBatchDequeueOpTask(stream_id);
      } else {
        AddDequeueTasks(stream_id);
      }
      AddBatchDequeueBuffOpTask(stream_id);
    }
}

Status CpuSchedModelBuilder::Build() {
  GE_ASSERT_TRUE((model_queue_param_.input_queues.size() + model_queue_param_.output_queues.size()) > 0);
  uint32_t stream_id = is_host_ ? GenerateStreamId() : static_cast<uint32_t>(aicpu_stream_id_);
  if (!model_queue_param_.input_queues.empty()) {
    AddDequeueTasksByAttrs(stream_id);
  }
  GE_CHK_STATUS_RET(AddMarkStepTask(stream_id, model_queue_param_.is_head), "Add mark step task failed.");
  AddActivateTask(stream_id);
  AddWaitEndGraph(stream_id);
  if (!model_queue_param_.output_queues.empty()) {
    AddEnqueueTasks(stream_id);
  }
  AddModelRepeat(stream_id);
  auto &model_info = model_.model_info_;
  model_info.queueNum = model_.queues_.size();
  model_info.queues = model_.queues_.data();

  model_.streams_.resize(1);
  auto &stream_info = model_.streams_.back();
  stream_info.streamId = stream_id;
  stream_info.streamFlag = ACL_STREAM_CPU_SCHEDULE;
  stream_info.taskNum = model_.tasks_[stream_id].size();
  stream_info.tasks = model_.tasks_[stream_id].data();

  model_info.aicpuStreamNum = model_.streams_.size();
  model_info.streams = model_.streams_.data();
  model_info.abnormalBreak = 0;
  model_info.abnormalEnqueue = 1;
  return SUCCESS;
}

void CpuSchedModelBuilder::SetModelId(uint32_t model_id) {
  model_.model_info_.modelId = model_id;
}

void CpuSchedModelBuilder::SetAicpuStreamId(int32_t stream_id) {
  aicpu_stream_id_ = stream_id;
}

void CpuSchedModelBuilder::SetAlignAttributes(uint32_t align_interval, const std::vector<uint32_t> &align_offsets) {
  align_interval_ = align_interval;
  align_offsets_ = align_offsets;
  has_align_attr_ = true;
}

void CpuSchedModelBuilder::SetModelQueueParam(const ModelQueueParam &model_queue_param) {
  model_queue_param_ = model_queue_param;
}

void CpuSchedModelBuilder::SetGlobalStep(uint64_t global_step) {
  global_step_ = global_step;
}

void CpuSchedModelBuilder::SetInputBufferAddrs(const std::vector<void *> &input_buf_addresses) {
  input_buf_addresses_ = input_buf_addresses;
}

void CpuSchedModelBuilder::SetOutputBufferAddrs(const std::vector<void *> &output_buf_addresses) {
  output_buf_addresses_ = output_buf_addresses;
}

void CpuSchedModelBuilder::SetInputTensor(const std::vector<GeTensorDesc> &input_tensor_descs) {
  input_tensor_descs_ = input_tensor_descs.data();
}

void CpuSchedModelBuilder::SetOutputTensor(const std::vector<GeTensorDesc> &output_tensor_descs) {
  output_tensor_descs_ = output_tensor_descs.data();
}

uint32_t CpuSchedModelBuilder::GenerateStreamId() const {
  static std::atomic<uint32_t> stream_id_gen{1};
  return stream_id_gen++;
}
} // namespace ge
