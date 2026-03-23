/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/npu_sched_model_loader.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "runtime/rt_external.h"
#include "common/checker.h"
#include "common/dump/dump_manager.h"
#include "framework/common/debug/ge_log.h"
#include "executor/npu_sched_model_configurator.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr int32_t kDefaultStreamPriority = 0;
constexpr uint32_t kMsgQueueDefaultDepth = 2U;
constexpr uint32_t kMsgQueueCtrlDropTimeout = 30U * 60U * 1000U;  // 30 min
}

void NpuSchedModelLoader::SetModelId(const uint32_t model_id) {
  model_id_ = model_id;
}

void NpuSchedModelLoader::SetDeviceId(int32_t device_id) {
  device_id_ = device_id;
}

void NpuSchedModelLoader::SetOutputTensorSizes(const std::vector<int64_t> &output_tensor_sizes) {
  output_tensor_sizes_ = output_tensor_sizes;
}

void NpuSchedModelLoader::SetOutputDynamicFlags(const std::vector<uint32_t> &output_dynamic_flags) {
  output_dynamic_flags_.assign(output_dynamic_flags.begin(), output_dynamic_flags.end());
}

void NpuSchedModelLoader::SetOutputQueueIds(const std::vector<uint32_t> &output_queue_ids) {
  output_queue_ids_ = output_queue_ids;
}

void NpuSchedModelLoader::SetInputDynamicFlags(const std::vector<bool> &input_dynamic_flags) {
  input_dynamic_flags_.assign(input_dynamic_flags.begin(), input_dynamic_flags.end());
}

void NpuSchedModelLoader::SetEnablePostProcessV2Flag(const bool enable) {
   enable_post_process_v2_ = enable; 
}

void NpuSchedModelLoader::SetSkipMarkStep(bool skip_mark_step) {
  skip_mark_step_ = skip_mark_step;
}

uint32_t NpuSchedModelLoader::GetReqMsgQueueId() const {
  return req_msg_queue_id_;
}

uint32_t NpuSchedModelLoader::GetRespMsgQueueId() const {
  return resp_msg_queue_id_;
}

void NpuSchedModelLoader::SetOutputStaticTensorDescs(const std::vector<RuntimeTensorDesc> &output_static_tensor_descs) {
  output_static_tensor_descs_ = const_cast<RuntimeTensorDesc *>(output_static_tensor_descs.data());
  output_static_tensor_num_ = output_static_tensor_descs.size();
}

void NpuSchedModelLoader::SetGlobalStep(const uint64_t global_step) {
  global_step_ = global_step;
}


Status NpuSchedModelLoader::CreateMsgQueues() {
  // create msg queues
  const std::string req_msg_queue_name =
      "queue.executor_req_" + std::to_string(device_id_) + "_" + std::to_string(model_id_);
  uint32_t req_msg_queue_id = 0U;
  GE_CHK_STATUS_RET(CreateQueue(device_id_, req_msg_queue_name, req_msg_queue_id),
                    "Fail to create req msg queue, queue name = %s.", req_msg_queue_name.c_str());
  req_msg_queue_id_ = static_cast<int32_t>(req_msg_queue_id);
  const std::string resp_msg_queue_name =
      "queue.executor_resp_" + std::to_string(device_id_) + "_" + std::to_string(model_id_);
  uint32_t resp_msg_queue_id = 0U;
  GE_CHK_STATUS_RET(CreateQueue(device_id_, resp_msg_queue_name, resp_msg_queue_id),
                    "Fail to create resp msg queue, queue name = %s.", resp_msg_queue_name.c_str());
  resp_msg_queue_id_ = static_cast<int32_t>(resp_msg_queue_id);
  GELOGD("Success to create message queue, req_msg_queue(name = %s, id = %d), resp_msg_queue(name = %s, id = %d).",
         req_msg_queue_name.c_str(), req_msg_queue_id_, resp_msg_queue_name.c_str(), resp_msg_queue_id_);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateQueue(const int32_t device_id, const std::string &name, uint32_t &queue_id) {
  GE_CHECK_LE(name.size(), static_cast<size_t>(RT_MQ_MAX_NAME_LEN) - 1);
  GELOGD("Start to create queue, device id = %d, queue name = %s", device_id, name.c_str());
  GE_CHK_STATUS_RET(EnsureQueueResourceInitialized(device_id), "Fail to initialize queue resource, queue name = %s",
                    name.c_str());
  rtMemQueueAttr_t attr;
  attr.depth = kMsgQueueDefaultDepth;
  attr.workMode = RT_MQ_MODE_PULL;  // default pull mode
  attr.flowCtrlFlag = false;
  attr.flowCtrlDropTime = kMsgQueueCtrlDropTimeout;
  attr.overWriteFlag = false;
  attr.deployType = RT_MQ_CLIENT_QUEUE_DEPLOY;
  // actually this won't fail, length was checked
  GE_ASSERT_EOK(strcpy_s(attr.name, sizeof(attr.name), name.c_str()),
                "Fail to copy queue name, name = %s.", name.c_str());
  GE_CHK_RT_RET(rtMemQueueCreate(device_id, &attr, &queue_id));
  GELOGD("Success to create queue, device id = %d, queue name = %s, depth = %u, queue_id = %u, deploy type = %u.",
         device_id, name.c_str(), attr.depth, queue_id, attr.deployType);
  return SUCCESS;
}

Status NpuSchedModelLoader::DestroyQueue(const int32_t device_id, const uint32_t queue_id) const {
  GELOGD("Start to destroy queue, device id = %d, queue_id = %u", device_id, queue_id);
  GE_CHK_RT_RET(rtMemQueueDestroy(device_id, queue_id));
  GELOGD("Success to destroy queue, device id = %d, queue_id = %u", device_id, queue_id);
  return SUCCESS;
}

Status NpuSchedModelLoader::EnsureQueueResourceInitialized(const int32_t device_id) {
  std::lock_guard<std::mutex> lock(queue_res_init_mutex_);
  if (queue_res_init_flag_) {
    GELOGD("Queue resource has been initialized, device_id = %d", device_id);
    return SUCCESS;
  }
  const auto ret = rtMemQueueInit(device_id);
  if (ret != RT_ERROR_NONE && ret != ACL_ERROR_RT_REPEATED_INIT) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemQueueInit fail, ret: 0x%X", static_cast<uint32_t>(ret));
    GELOGE(RT_FAILED, "[InitQueue] failed, rt_err = %d, device id = %d", ret, device_id);
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  GELOGI("Success to init queue, device id = %d", device_id);
  queue_res_init_flag_ = true;
  return SUCCESS;
}

Status NpuSchedModelLoader::LoadModel(const ModelQueueParam &model_queue_param, uint32_t &runtime_model_id) {
  GELOGD("Begin to load model, model_id = %u.", model_id_);
  GE_CHK_BOOL_RET_STATUS(!model_queue_param.input_queues.empty() || !model_queue_param.output_queues.empty(),
                         UNSUPPORTED, "Not exist input queue and output queue.");
  model_queue_param_ = model_queue_param;
  if (model_queue_param_.input_fusion_offsets.empty()) {
    model_queue_param_.input_fusion_offsets.resize(model_queue_param_.input_queues.size());
  }
  GE_CHK_STATUS_RET(CreateMsgQueues(), "Fail to create message queue.");
  // create model_handle
  DF_CHK_ACL_RET(aclmdlRIBuildBegin(&rt_model_handle_, 0U));
  GE_CHK_RT_RET(rtModelGetId(rt_model_handle_, &runtime_model_id_));
  runtime_model_id = runtime_model_id_;
  GE_CHK_STATUS_RET(SetModelConfig(), "Fail to set model config, model_id:%u, runtime_model_id:%u.", model_id_,
                    runtime_model_id_);
  // create aicpu entry stream and next stream
  DF_CHK_ACL_RET(aclrtCreateStreamWithConfig(&rt_entry_stream_, kDefaultStreamPriority, ACL_STREAM_CPU_SCHEDULE | ACL_STREAM_PERSISTENT));
  DF_CHK_ACL_RET(aclmdlRIBindStream(rt_model_handle_, rt_entry_stream_, static_cast<uint32_t>(ACL_MODEL_STREAM_FLAG_HEAD)));
  DF_CHK_ACL_RET(aclrtCreateStreamWithConfig(&rt_next_stream_, kDefaultStreamPriority, ACL_STREAM_CPU_SCHEDULE | ACL_STREAM_PERSISTENT));
  DF_CHK_ACL_RET(aclmdlRIBindStream(rt_model_handle_, rt_next_stream_, static_cast<uint32_t>(ACL_MODEL_STREAM_FLAG_DEFAULT)));
  // create tasks on entry stream and next stream
  GE_CHK_STATUS_RET(CreateSchedTasks(), "Fail to create sched tasks for model:%u.", model_id_);
  // distribute tasks
  GE_CHK_STATUS_RET(DistributeTasks(), "Fail to distribute sched tasks for model:%u.", model_id_);
  // create rt stream and distribute end graph task
  GE_CHK_STATUS_RET(DistributeEndGraph());
  // complete to load model
  DF_CHK_ACL_RET(aclmdlRIBuildEnd(rt_model_handle_, nullptr));
  GELOGD("Success to load model, model_id = %u, runtime_model_id = %u.", model_id_, runtime_model_id_);
  return SUCCESS;
}

void NpuSchedModelLoader::SetAlignAttributes(const uint32_t align_interval,
                                             const std::vector<uint32_t> &align_offsets) {
  align_interval_ = align_interval;
  align_offsets_ = align_offsets;
  has_align_attr_ = true;
}

Status NpuSchedModelLoader::SetModelConfig() const {
  NpuSchedModelConfigurator::AicpuModelConfig config = {};
  config.model_id = model_id_;
  config.runtime_model_id = runtime_model_id_;
  config.abnormal_exist = 0;
  config.abnormal_enqueue = 1;
  config.req_msg_queue = static_cast<int32_t>(req_msg_queue_id_);
  config.resp_msg_queue = static_cast<int32_t>(resp_msg_queue_id_);
  GE_CHK_STATUS_RET_NOLOG(NpuSchedModelConfigurator::SetModelConfig(config));
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateSchedTasks() {
  if (!skip_mark_step_) {
    // create mark step task on entry stream
    GE_CHK_STATUS_RET(CreateMarkStepTask(rt_entry_stream_));
  }
  // bind input queue
  GE_CHK_STATUS_RET(BindInputQueue(rt_entry_stream_), "Fail to bind input queue, model_id:%u.", model_id_);
  // create tasks on entry stream and next stream
  GE_CHK_STATUS_RET(CreatePrepareDynamicInputOutputTask(rt_entry_stream_),
                    "Fail to create prepare dynamic input output task, model_id:%u.", model_id_);
  GE_CHK_STATUS_RET(CreateEnqueueTask(rt_entry_stream_, req_msg_queue_id_, req_msg_mbuf_addr_));
  uint32_t first_notify_id = GenerateNotifyId();
  uint32_t next_notify_id = GenerateNotifyId();
  GE_CHK_STATUS_RET(CreateNotifyRecordTask(rt_entry_stream_, first_notify_id));
  GE_CHK_STATUS_RET(CreateNotifyWaitTask(rt_entry_stream_, next_notify_id));
  GE_CHK_STATUS_RET(CreateStreamRepeatTask(rt_entry_stream_));
  GE_CHK_STATUS_RET(CreateNotifyWaitTask(rt_next_stream_, first_notify_id));
  GE_CHK_STATUS_RET(CreateZeroCopyTask(rt_next_stream_, input_mbuf_addrs_, postproc_input_mbuf_addrs_));
  GE_CHK_STATUS_RET(CreateZeroCopyTask(rt_next_stream_, output_mbuf_addrs_, postproc_output_mbuf_addrs_));
  GE_CHK_STATUS_RET(CreateNotifyRecordTask(rt_next_stream_, next_notify_id));
  GE_CHK_STATUS_RET(CreateDequeueTask(rt_next_stream_, resp_msg_queue_id_, resp_msg_mbuf_addr_));
  GE_CHK_STATUS_RET(CreatePostprocessDynamicOutputTask(rt_next_stream_));
  // bind output queue
  GE_CHK_STATUS_RET(BindOutputQueue(rt_next_stream_), "Fail to bind output queue, model_id:%u.", model_id_);
  // create stream repeat task on next stream
  GE_CHK_STATUS_RET(CreateStreamRepeatTask(rt_next_stream_));
  return SUCCESS;
}

Status NpuSchedModelLoader::DistributeTasks() const {
  for (auto &task : sched_tasks_) {
    GE_CHECK_NOTNULL(task);
    GE_CHK_STATUS_RET(task->Distribute());
  }
  return SUCCESS;
}

Status NpuSchedModelLoader::DistributeEndGraph() {
  DF_CHK_ACL_RET(aclrtCreateStreamWithConfig(&rt_fake_stream_, kDefaultStreamPriority, ACL_STREAM_PERSISTENT));
  DF_CHK_ACL_RET(aclmdlRIBindStream(rt_model_handle_, rt_fake_stream_, static_cast<uint32_t>(ACL_MODEL_STREAM_FLAG_DEFAULT)));
  DF_CHK_ACL_RET(aclmdlRIEndTask(rt_model_handle_, rt_fake_stream_));
  return SUCCESS;
}

Status NpuSchedModelLoader::BindInputQueue(const aclrtStream stream) {
  // unique input queue ids for input fusion case
  std::vector<uint32_t> unique_input_queue_ids;
  std::vector<uint32_t> unique_align_offsets;  // input align and input fusion are non-coexisting
  const std::vector<uint32_t> &input_queue_ids = model_queue_param_.input_queues;
  for (size_t i = 0UL; i < input_queue_ids.size(); ++i) {
    const uint32_t queue_id = input_queue_ids[i];
    const auto iter = std::find(unique_input_queue_ids.begin(), unique_input_queue_ids.end(), queue_id);
    if (iter != unique_input_queue_ids.end()) {
      continue;
    }
    (void)unique_input_queue_ids.emplace_back(queue_id);
    GE_CHK_RT_RET(rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_INPUT_QUEUE));
    if (has_align_attr_) {
      (void)unique_align_offsets.emplace_back(align_offsets_[i]);
    }
  }
  // create model batch dequeue task
  std::vector<uint64_t> unique_input_mbuf_addrs;
  if ((input_align_attrs_.align_max_cache_num != 0U) && (model_queue_param_.input_queues_attrs.size() > 1UL)) {
    GELOGI("Add gather dequeue task result of input_align_attrs_ is set.");
    GE_CHK_STATUS_RET(CreateModelGatherDequeueTask(stream, model_queue_param_.input_queues_attrs,
                                                   unique_input_mbuf_addrs),
                      "Fail to add model gather dequeue task, model_id:%u.", model_id_);
  } else {
    GE_CHK_STATUS_RET(CreateModelBatchDequeueTask(stream, unique_input_queue_ids, align_interval_,
                                                  unique_align_offsets, unique_input_mbuf_addrs),
                      "Fail to add model batch dequeue task, model_id:%u.", model_id_);
  }

  // update input_mbuf_addrs
  GE_CHK_STATUS_RET(UpdateFusionInputsMbufAddr(unique_input_mbuf_addrs),
                    "Fail to update fusion input mbuf addrs, model_id:%u.", model_id_);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateModelBatchDequeueTask(const aclrtStream stream,
                                                        const std::vector<uint32_t> &queue_ids,
                                                        const uint32_t align_interval,
                                                        const std::vector<uint32_t> &align_offsets,
                                                        std::vector<uint64_t> &mbuf_addrs) {
  // create model batch dequeue task
  const auto task = MakeShared<SchedTaskModelBatchDequeue>(stream);
  GE_CHECK_NOTNULL(task);
  std::vector<uint64_t> unique_input_mbuf_addrs;
  GE_CHK_STATUS_RET_NOLOG(task->Init(queue_ids, align_interval, align_offsets, mbuf_addrs));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateModelGatherDequeueTask(const aclrtStream stream,
                                                         const std::vector<QueueAttrs> &queues,
                                                         std::vector<uint64_t> &mbuf_addrs) {
  // create model batch dequeue task
  const auto task = MakeShared<SchedTaskModelGatherDequeue>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(queues, input_align_attrs_, mbuf_addrs));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::UpdateFusionInputsMbufAddr(const std::vector<uint64_t> &unique_input_mbuf_addrs) {
  // update input_mbuf_addrs_
  const std::vector<uint32_t> &input_queue_ids = model_queue_param_.input_queues;
  std::unordered_map<uint32_t, uint64_t> qid_to_mbuf_addr;
  size_t unique_mbuf_addr_index = 0UL;
  for (auto queue_id : input_queue_ids) {
    const auto iter = qid_to_mbuf_addr.find(queue_id);
    if (iter != qid_to_mbuf_addr.end()) {
      input_mbuf_addrs_.emplace_back(iter->second);
      continue;
    }
    const uint64_t mbuf_addr = unique_input_mbuf_addrs[unique_mbuf_addr_index++];
    input_mbuf_addrs_.emplace_back(mbuf_addr);
    qid_to_mbuf_addr[queue_id] = mbuf_addr;
  }
  return SUCCESS;
}

Status NpuSchedModelLoader::CreatePrepareDynamicInputOutputTask(const aclrtStream stream) {
  const auto task = MakeShared<SchedTaskPrepareDynamicInputOutput>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(input_dynamic_flags_, input_mbuf_addrs_,
                                     model_queue_param_.input_fusion_offsets, output_tensor_sizes_,
                                     output_mbuf_addrs_, req_msg_mbuf_addr_, enable_post_process_v2_));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateEnqueueTask(const aclrtStream stream, const uint32_t queue_id,
                                              const uint64_t mbuf_addr) {
  const auto task = MakeShared<SchedTaskModelEnqueue>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(queue_id, mbuf_addr));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

uint32_t NpuSchedModelLoader::GenerateNotifyId() const {
  static std::atomic<uint32_t> notify_id_gen{1};
  return notify_id_gen++;
}

Status NpuSchedModelLoader::CreateNotifyRecordTask(const aclrtStream stream, const uint32_t notify_id) {
  const auto task = MakeShared<SchedTaskNotifyRecord>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(notify_id));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateNotifyWaitTask(const aclrtStream stream, const uint32_t notify_id) {
  const auto task = MakeShared<SchedTaskNotifyWait>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(notify_id));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateStreamRepeatTask(const aclrtStream stream) {
  const auto task = MakeShared<SchedTaskStreamRepeat>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(runtime_model_id_));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateZeroCopyTask(const aclrtStream stream, const std::vector<uint64_t> &src_addrs,
                                               std::vector<uint64_t> &dst_addrs) {
  const auto task = MakeShared<SchedTaskZeroCopy>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(src_addrs, dst_addrs));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateDequeueTask(const aclrtStream stream, const uint32_t queue_id,
                                              uint64_t &mbuf_addr) {
  const auto task = MakeShared<SchedTaskModelDequeue>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(queue_id, mbuf_addr));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreatePostprocessDynamicOutputTask(const aclrtStream stream) {
  const auto task = MakeShared<SchedTaskPostprocessDynamicOutput>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(resp_msg_mbuf_addr_, postproc_input_mbuf_addrs_, postproc_output_mbuf_addrs_,
                                     output_dynamic_flags_, output_static_tensor_descs_, output_static_tensor_num_,
                                     enable_post_process_v2_));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::BindOutputQueue(const aclrtStream stream) {
  for (const uint32_t queue_id : output_queue_ids_) {
    GE_CHK_RT_RET(rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_OUTPUT_QUEUE));
  }
  GE_CHK_STATUS_RET(CreateModelBatchEnqueueTask(stream, output_queue_ids_, postproc_output_mbuf_addrs_),
                    "Fail to add model batch dequeue task, model_id:%u.", model_id_);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateModelBatchEnqueueTask(const aclrtStream stream,
                                                        const std::vector<uint32_t> &queue_ids,
                                                        const std::vector<uint64_t> &mbuf_addrs) {
  const auto task = MakeShared<SchedTaskModelBatchEnqueue>(stream);
  GE_CHECK_NOTNULL(task);
  GE_CHK_STATUS_RET_NOLOG(task->Init(queue_ids, mbuf_addrs));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::CreateMarkStepTask(const aclrtStream stream) {
  const auto task = MakeShared<SchedTaskMarkStep>(stream);
  GE_CHECK_NOTNULL(task);
  const auto dump_step = DumpManager::GetInstance().GetDumpProperties(kInferSessionId).GetDumpStep();
  GE_CHK_STATUS_RET_NOLOG(task->Init(model_queue_param_.group_total_count, model_queue_param_.group_index,
                                     model_queue_param_.group_policy, dump_step, global_step_));
  sched_tasks_.push_back(task);
  return SUCCESS;
}

Status NpuSchedModelLoader::UnloadModel() {
  GELOGD("Begin to unload model, model_id = %u, runtime_model_id = %u.", model_id_, runtime_model_id_);
  // check rt ctx is exist
  aclrtContext current_ctx = nullptr;
  if (aclrtGetCurrentContext(&current_ctx) == ACL_ERROR_NONE) {
    (void) UnbindStreams();
    (void) ReleaseTasks();
    // destroy model
    if (rt_model_handle_ != nullptr) {
      DF_CHK_ACL(aclmdlRIDestroy(rt_model_handle_));
      rt_model_handle_ = nullptr;
    }
  }

  (void) DestroyQueue(device_id_, req_msg_queue_id_);
  (void) DestroyQueue(device_id_, resp_msg_queue_id_);
  GELOGD("Success to unload model, model_id = %u, runtime_model_id = %u.", model_id_, runtime_model_id_);
  return SUCCESS;
}

Status NpuSchedModelLoader::UnbindStreams() {
  // unbind and destroy entry stream
  if (rt_entry_stream_ != nullptr) {
    GE_LOGW_IF(aclmdlRIUnbindStream(rt_model_handle_, rt_entry_stream_) != ACL_ERROR_NONE, "Fail to unbind stream.");
    GE_LOGW_IF(aclrtDestroyStream(rt_entry_stream_) != ACL_ERROR_NONE, "Fail to destroy stream.");
    rt_entry_stream_ = nullptr;
  }
  // unbind and destroy next stream
  if (rt_next_stream_ != nullptr) {
    GE_LOGW_IF(aclmdlRIUnbindStream(rt_model_handle_, rt_next_stream_) != ACL_ERROR_NONE, "Fail to unbind stream.");
    GE_LOGW_IF(aclrtDestroyStream(rt_next_stream_) != ACL_ERROR_NONE, "Fail to destroy stream.");
    rt_next_stream_ = nullptr;
  }
  // unbind and destroy runtime fake stream
  if (rt_fake_stream_ != nullptr) {
    GE_LOGW_IF(aclmdlRIUnbindStream(rt_model_handle_, rt_fake_stream_) != ACL_ERROR_NONE, "Fail to unbind stream.");
    GE_LOGW_IF(aclrtDestroyStream(rt_fake_stream_) != ACL_ERROR_NONE, "Fail to destroy stream.");
    rt_fake_stream_ = nullptr;
  }
  return SUCCESS;
}

Status NpuSchedModelLoader::ReleaseTasks() {
  for (const auto &task : sched_tasks_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "Fail to release task, model_id:%u, runtime_model_id:%u.",
                    model_id_, runtime_model_id_);
    }
  }
  sched_tasks_.clear();
  return SUCCESS;
}
}
