/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/dynamic_model_executor.h"
#include <future>
#include "mmpa/mmpa_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_context.h"
#include "common/utils/heterogeneous_profiler.h"
#include "common/utils/rts_api_utils.h"
#include "common/dump/dump_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "aicpu/aicpu_schedule/aicpusd_interface.h"
#include "aicpu/aicpu_schedule/aicpusd_info.h"
#include "aicpu/queue_schedule/dgw_client.h"
#include "executor/cpu_sched_event_dispatcher.h"
#include "dflow/base/deploy/exchange_service.h"
#include "executor/cpu_sched_model_builder.h"
#include "graph/utils/op_type_utils.h"
#include "graph/manager/util/hcom_ome_util.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "proto/deployer.pb.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/helper/model_parser_base.h"
#include "graph/ge_tensor.h"
#include "framework/runtime/gert_api.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr uint32_t kDynamicModelMaxIdBase = 1023U;
constexpr uint32_t kDummyQId = UINT32_MAX;
constexpr int32_t kQueueAttachTime = 3 * 1000; // 3s
constexpr int32_t kReportStatusEnqueueTimeout = 0;
constexpr int32_t kClearTypeStop = 1;
constexpr int32_t kClearTypeClear = 2;
constexpr int32_t kDefaultStreamPriority = 0;
constexpr const char *kModelStopFunc = "AICPUModelStop";
constexpr const char *kModelClearFunc = "AICPUModelClearInputAndRestart";
constexpr const char *kModelProcessDataException = "AICPUModelProcessDataException";
constexpr const char *kParallelModeSerial = "1";
const std::string kGatherDequeue = "gatherDequeue";
}
std::mutex DynamicModelExecutor::exec_mutex_;

DynamicModelExecutor::DynamicModelExecutor(bool is_host) : is_host_(is_host) {
  model_execute_param_.req_mbuf = nullptr;
  model_execute_param_.resp_mbuf = nullptr;
}

DynamicModelExecutor::~DynamicModelExecutor() {
  FinalizeInternal();
}

Status DynamicModelExecutor::Initialize() {
  const std::string kAicpu = "libaicpu_scheduler.so";
  const std::string kHostAicpu = "libhost_aicpu_scheduler.so";
  const std::string aicpu_so_name = is_host_ ? kHostAicpu : kAicpu;
  aicpu_handle_ = mmDlopen(aicpu_so_name.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
      static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
  GE_CHECK_NOTNULL(aicpu_handle_);
  GELOGI("Executor dlopen[%s] success.", aicpu_so_name.c_str());
  if (!is_host_) {
    std::string parallel_mode;
    (void) GetContext().GetOption(OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE, parallel_mode);
    exec_with_mutex_ = (parallel_mode == kParallelModeSerial);
    GELOGI("%s = [%s]", OPTION_EXEC_DYNAMIC_GRAPH_PARALLEL_MODE, parallel_mode.c_str());
  }
  HeterogeneousProfiler::Instance().InitHeterogeneousPoriler();
  return SUCCESS;
}

void DynamicModelExecutor::Finalize() {
  FinalizeInternal();
  HeterogeneousProfiler::Instance().PrintHeterogeneousProfilerData();
  GELOGI("Executor finalize success.");
}

void DynamicModelExecutor::FinalizeInternal() {
  if (is_host_ && (new_allocated_global_step_ != nullptr)) {
    free(new_allocated_global_step_);
  } else if (new_allocated_global_step_ != nullptr) {
    (void)aclrtFree(new_allocated_global_step_);
  }
  new_allocated_global_step_ = nullptr;
  if (aicpu_handle_ != nullptr) {
    (void)mmDlclose(aicpu_handle_);
    aicpu_handle_ = nullptr;
  }
  if (stream_ != nullptr) {
    DF_CHK_ACL(aclrtDestroyStream(stream_));
    stream_ = nullptr;
  }
  if (aicpu_model_handle_ != nullptr) {
    DF_CHK_ACL(aclmdlRIDestroy(aicpu_model_handle_));
    aicpu_model_handle_ = nullptr;
  }
  if (aicpu_stream_ != nullptr) {
    DF_CHK_ACL(aclrtDestroyStream(aicpu_stream_));
    aicpu_stream_ = nullptr;
  }
  CpuSchedEventDispatcher::GetInstance().Deregister(aicpu_model_id_);
}

Status DynamicModelExecutor::FreeEventIOBuffer() {
  std::set<void *> buf_addresses(input_buf_addresses_.cbegin(), input_buf_addresses_.cend());
  buf_addresses.insert(output_buf_addresses_.cbegin(), output_buf_addresses_.cend());
  for (void *buf_address : buf_addresses) {
    GE_CHK_STATUS_RET(aclrtFreeHost(buf_address), "aclrtFreeHost Failed, buf_addresses size = %zu.", buf_addresses.size());
  }
  input_buf_addresses_.clear();
  output_buf_addresses_.clear();
  return SUCCESS;
}

Status DynamicModelExecutor::AllocEventIOBuffer(const ComputeGraphPtr &root_graph) const {
  // delete related parameter when delete event
  (void)root_graph;
  return SUCCESS;
}

Status DynamicModelExecutor::LoadModel(const ModelData &model_data,
                                       const ComputeGraphPtr &root_graph,
                                       const ModelQueueParam &model_queue_param) {
  DF_CHK_ACL_RET(aclrtGetDevice(&device_id_));
  DF_CHK_ACL_RET(aclrtGetCurrentContext(&rt_context_));
  if (!GetContext().GetHostExecFlag()) {
    DF_CHK_ACL_RET(aclrtCreateStream(&stream_));
  }
  GE_CHK_STATUS_RET_NOLOG(GetInputAndOutputNum(root_graph, model_queue_param));
  input_queue_attrs_ = model_queue_param.input_queues_attrs;
  output_queue_attrs_ = model_queue_param.output_queues_attrs;
  input_fusion_offsets_ = model_queue_param.input_fusion_offsets;
  status_output_queue_device_id_ = model_queue_param.status_output_queue.device_id;
  status_output_queue_id_ = model_queue_param.status_output_queue.queue_id;
  auto status_queue_device_type = model_queue_param.status_output_queue.device_type;
  need_report_status_ = (model_queue_param.is_dynamic_sched && model_queue_param.need_report_status);
  auto is_client = status_queue_device_type == static_cast<int32_t>(NPU);
  if (is_client && need_report_status_) {
    HeterogeneousExchangeService::GetInstance().AddClientQueue(status_output_queue_id_);
  }
  model_uuid_ = model_queue_param.model_uuid;
  input_align_attrs_ = model_queue_param.input_align_attrs;
  if (input_fusion_offsets_.empty()) {
    input_fusion_offsets_.resize(num_inputs_);
  }
  input_buf_addresses_.resize(input_events_num_);
  output_buf_addresses_.resize(output_events_num_);
  input_mbuf_addresses_.resize(input_queues_num_);
  output_mbuf_addresses_.resize(output_queues_num_);
  GE_CHK_STATUS_RET_NOLOG(ParseModelDesc(root_graph));
  GE_CHK_STATUS_RET_NOLOG(DoLoadModel(model_data, root_graph));
  GE_CHK_STATUS_RET_NOLOG(GetGlobalStepAddr());
  GE_CHK_STATUS_RET_NOLOG(AllocEventIOBuffer(root_graph));
  if ((num_inputs_ == 0U) && (num_outputs_ == 0U)) {
    return ExecuteDirectly();
  }
  // load with aicpu-sd
  GE_CHK_STATUS_RET_NOLOG(LoadWithAicpuSd(root_graph, model_queue_param));
  if (need_report_status_) {
    GE_CHK_STATUS_RET(
        RtsApiUtils::MemQueueAttach(status_output_queue_device_id_, status_output_queue_id_, kQueueAttachTime),
        "Status queue mem queue attach failed, device_id=%d, queue_id=%u, timeout=%d", status_output_queue_device_id_,
        status_output_queue_id_, kQueueAttachTime);
  }
  run_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_drun");
    Run();
  });
  return SUCCESS;
}

void DynamicModelExecutor::DestroyDatasetResource() {
  rtCtxSetCurrent(rt_context_);
  GEEVENT("Destroy dataset resource begin, inner model_id = %u.", model_id_);
  if (model_desc_ != nullptr) {
    (void) aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  for (aclTensorDesc* &desc : acl_tensor_desc_) {
    if (desc != nullptr) {
      (void) aclDestroyTensorDesc(desc);
      desc = nullptr;
    }
  }
  for (aclDataBuffer* &buffer : output_data_buffer_) {
    if (buffer != nullptr) {
      (void) aclDestroyDataBuffer(buffer);
      buffer = nullptr;
    }
  }
  for (aclDataBuffer* &buffer : input_data_buffer_) {
    if (buffer != nullptr) {
      (void) aclDestroyDataBuffer(buffer);
      buffer = nullptr;
    }
  }
  if (input_dataset_ != nullptr) {
    (void) aclmdlDestroyDataset(input_dataset_);
    input_dataset_ = nullptr;
  }
  if (output_dataset_ != nullptr) {
    (void) aclmdlDestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
  }
  GEEVENT("Destroy dataset resource success, inner model_id = %u.", model_id_);
}

void DynamicModelExecutor::UnloadModel() {
  Stop();
  (void) UnloadFromAicpuSd();
  GEEVENT("UnloadModel model begin, inner model_id = %u.", model_id_);
  if (!external_weight_mem_data_.empty()){
    for (auto &external_weight : external_weight_mem_data_) {
      if (external_weight.device_mem == nullptr) {
        continue;
      }
      void *data_addr = const_cast<void*>(external_weight.device_mem);
      (void)aclrtFree(data_addr);
      external_weight.device_mem = nullptr;
      data_addr = nullptr;
    }
    GEEVENT("UnloadModel model external weight success, inner model_id = %u.", model_id_);
  }
  rtCtxSetCurrent(rt_context_);
  if (handle_ != nullptr) {
    (void) aclmdlDestroyConfigHandle(handle_);
    handle_ = nullptr;
  }
  (void) aclmdlUnload(model_id_);
  GEEVENT("UnloadModel model success, inner model_id = %u.", model_id_);
  FreeEventIOBuffer();
}

Status DynamicModelExecutor::UnloadFromAicpuSd() {
  const std::string kDestroyFunc = "AICPUModelDestroy";
  const auto destroy_func =
      reinterpret_cast<int32_t (*)(uint32_t)>(mmDlsym(aicpu_handle_, kDestroyFunc.c_str()));
  if (destroy_func != nullptr) {
    (void) destroy_func(aicpu_model_id_);
  }
  (void) AicpuModelIdResourceManager::GetInstance().DeAllocate(aicpu_model_ids_);
  return SUCCESS;
}

Status DynamicModelExecutor::ExecuteAsync(const std::function<void(Status, void*, void *)> &callback,
                                          void *req_mbuf, void *resp_mbuf) {
  if (!stop_schedule_flag_) {
    const ModelExecuteParam param {.callback = callback, .req_mbuf = req_mbuf, .resp_mbuf = resp_mbuf};
    GE_CHK_BOOL_RET_STATUS(task_queue_.Push(param), FAILED, "Failed to enqueue task, model_id = %u", model_id_);
    GELOGD("Enqueue task successfully, model_id = %u", model_id_);
  }
  return SUCCESS;
}

Status DynamicModelExecutor::CheckInputs() {
  is_need_execute_model_ = true;
  data_ret_code_ = 0;
  for (size_t i = 0U; i < num_inputs_; ++i) {
    if (i >= input_queues_num_) {
      continue;
    }
    void *mbuf = input_mbuf_addresses_[i];
    void *buffer_data = nullptr;
    uint64_t buffer_size = 0U;
    GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, &buffer_data));
    GE_CHK_RT_RET(rtMbufGetBuffSize(mbuf, &buffer_size));
    GE_CHECK_GE(buffer_size, sizeof(RuntimeTensorDesc));
    void *head_buf = nullptr;
    uint64_t head_size = 0U;
    GE_CHK_RT_RET(rtMbufGetPrivInfo(reinterpret_cast<rtMbufPtr_t>(mbuf), &head_buf, &head_size));
    if ((head_buf == nullptr) || (head_size < sizeof(ExchangeService::MsgInfo))) {
      GELOGE(PARAM_INVALID, "The input[%zu] mbuf head is invalid.", i);
      return PARAM_INVALID;
    }
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char_t *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    // is invalid input data
    if (msg_info->ret_code != 0) {
      is_need_execute_model_ = false;
      data_ret_code_ = msg_info->ret_code;
      GELOGD("The input[%zu] is invalid, data ret code = %d", i, data_ret_code_);
      return SUCCESS;
    }

    // is null data
    const bool is_null_data_input = ((msg_info->data_flag & kNullDataFlagBit) != 0U);
    if (is_null_data_input && is_need_execute_model_) {
      GELOGI("input[%zu] data flag=%u is null data, no need execute model.", i, msg_info->data_flag);
      is_need_execute_model_ = false;
    }
    GELOGD("The input[%zu] is ok, null data flag = %d.", i, is_null_data_input);
  }
  return SUCCESS;
}

Status DynamicModelExecutor::PublishOutputWithoutExecute() {
  void *src_head_buf = nullptr;
  uint64_t src_head_size = 0U;
  if (!input_mbuf_addresses_.empty()) {
    GE_CHK_RT_RET(rtMbufGetPrivInfo(input_mbuf_addresses_.back(), &src_head_buf, &src_head_size));
    if ((src_head_buf == nullptr) || (src_head_size == 0U)) {
      GELOGE(FAILED, "Get mbuf priv data failed.");
      return FAILED;
    }
  }
  const uint64_t buffer_size = static_cast<uint64_t>(sizeof(RuntimeTensorDesc));
  for (size_t i = 0U; i < num_outputs_; ++i) {
    if (IsEventOutput(i)) {
      continue;
    }
    if (is_need_alloc_output_mbuf_) {
      // alloc empty tensor
      GE_CHK_RT_RET(rtMbufAlloc(&output_mbuf_addresses_[i], buffer_size));
    }
    GE_CHK_RT_RET(rtMbufSetDataLen(output_mbuf_addresses_[i], buffer_size));
    void *dst_head_buf = nullptr;
    uint64_t dst_head_size = 0U;
    GE_CHK_RT_RET(rtMbufGetPrivInfo(output_mbuf_addresses_[i], &dst_head_buf, &dst_head_size));
    GE_CHECK_NOTNULL(dst_head_buf);
    GE_CHK_BOOL_RET_STATUS(dst_head_size >= sizeof(ExchangeService::MsgInfo), FAILED,
                           "dst_head_size = %lu, size of ExchangeService::MsgInfo = %zu.",
                           dst_head_size, sizeof(ExchangeService::MsgInfo));
    if (!input_mbuf_addresses_.empty()) {
      GE_CHK_BOOL_RET_STATUS(dst_head_size == src_head_size, FAILED, "dst_head_size = %lu, src_head_size = %lu",
                             dst_head_size, src_head_size);
      if (memcpy_s(dst_head_buf, dst_head_size, src_head_buf, src_head_size) != EOK) {
        GELOGE(FAILED, "Failed to copy input mbuf head to output[%zu] mbuf head.", i);
        return FAILED;
      }
    }
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char_t *>(dst_head_buf) + dst_head_size - sizeof(ExchangeService::MsgInfo));
    msg_info->ret_code = data_ret_code_;
    GELOGD("The output[%zu] data ret code = %d.", i, msg_info->ret_code);
    void *buffer_data = nullptr;
    GE_CHK_RT_RET(rtMbufGetBuffAddr(output_mbuf_addresses_[i], &buffer_data));
    GE_CHECK_NOTNULL(buffer_data);
    auto *const runtime_tensor_desc = reinterpret_cast<RuntimeTensorDesc *>(buffer_data);
    // empty tensor shape: dim_num = 1; dim_value = 0;
    runtime_tensor_desc->shape[0] = 1L;
    runtime_tensor_desc->shape[1] = 0L;
    runtime_tensor_desc->original_shape[0] = 1L;
    runtime_tensor_desc->original_shape[1] = 0L;
    runtime_tensor_desc->data_size = 0;
  }
  GELOGD("Success to publish null data tensor without execute.");
  return SUCCESS;
}

void DynamicModelExecutor::UpdateFusionInputsAddr() {
  std::map<uint32_t, void *> qid_to_addr;
  for (size_t i = 0; i < input_queues_num_; ++i) {
    auto queue_id = input_queue_attrs_[i].queue_id;
    const auto &it = qid_to_addr.find(queue_id);
    if (it != qid_to_addr.cend()) {
      input_mbuf_addresses_[i] = it->second;
      continue;
    }
    qid_to_addr[queue_id] = input_mbuf_addresses_[i];
  }
}

Status DynamicModelExecutor::ExecuteInternal() {
  GELOGD("Execute model started, model_id = %u", model_id_);
  ClearOutputs();
  UpdateFusionInputsAddr();
  GE_CHK_STATUS_RET_NOLOG(CheckInputs());
  if (!is_need_execute_model_) {
    GELOGD("The current inputs does not need to be executed, model_id = %u", model_id_);
    return PublishOutputWithoutExecute();
  }
  // prepare inputs
  std::vector<DataBuffer> model_inputs;
  std::vector<DataBuffer> model_outputs;
  GE_CHK_STATUS_RET_NOLOG(PrepareInputs(model_inputs));
  GELOGD("Inputs prepared successfully, model_id = %u.", model_id_);
  GE_CHK_STATUS_RET_NOLOG(PrepareOutputs(model_outputs));
  GELOGD("Output buffers prepared successfully, model_id = %u.", model_id_);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kDynamicExecute, device_id_);
  if (exec_with_mutex_) {
    // 防止多线程执行时导致oom
    std::lock_guard<std::mutex> lk(exec_mutex_);
    GE_CHK_STATUS_RET(DoExecuteModel(model_inputs, model_outputs), "Failed to execute model.");
  } else {
    GE_CHK_STATUS_RET(DoExecuteModel(model_inputs, model_outputs), "Failed to execute model.");
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kDynamicExecute, device_id_);
  GELOGD("Model executed successfully, model_id = %u.", model_id_);
  GE_CHK_STATUS_RET_NOLOG(UpdateOutputs(model_outputs));
  GELOGD("Outputs post processes done successfully, model_id = %u.", model_id_);
  return SUCCESS;
}

Status DynamicModelExecutor::ExecuteDirectly() {
  GE_CHK_BOOL_RET_STATUS((num_inputs_ == 0UL) && (num_outputs_ == 0UL),
                         UNSUPPORTED, "Inputs or outputs num is invalid, num_inputs_ = %zu, num_outputs_ = %zu",
                         num_inputs_, num_outputs_);
  std::vector<DataBuffer> model_inputs;
  std::vector<DataBuffer> model_outputs;
  GE_CHK_STATUS_RET(DoExecuteModel(model_inputs, model_outputs), "Failed to execute model");
  return SUCCESS;
}

Status DynamicModelExecutor::UpdateBufferDataAddr(size_t index, void *&buffer_data, uint64_t buffer_size) const {
  uint64_t total_offset = 0UL;
  auto buffer_base = PtrToPtr<void, uint8_t>(buffer_data);
  for (int32_t i = 0; i < input_fusion_offsets_[index]; ++i) {
    auto input_base = PtrAdd(buffer_base, buffer_size, total_offset);
    GE_CHECK_NOTNULL(input_base);
    GE_CHECK_LE(total_offset, buffer_size - sizeof(RuntimeTensorDesc));
    auto tensor_desc = PtrToPtr<uint8_t, RuntimeTensorDesc>(buffer_base);
    total_offset += sizeof(RuntimeTensorDesc);
    GE_CHECK_LE(tensor_desc->data_size, buffer_size);
    GE_CHECK_LE(total_offset, buffer_size - tensor_desc->data_size);
    total_offset += tensor_desc->data_size;
  }
  GE_CHECK_LE(total_offset, buffer_size - sizeof(RuntimeTensorDesc));
  auto input_addr = PtrAdd(buffer_base, buffer_size, total_offset);
  buffer_data = PtrToPtr<uint8_t, void>(input_addr);
  GELOGI("Input[%zu] update addr success, fusion offset = %d, total offset = %lu, buffer size = %lu",
         index, input_fusion_offsets_[index], total_offset, buffer_size);
  return SUCCESS;
}

Status DynamicModelExecutor::PrepareInputs(std::vector<DataBuffer> &model_inputs) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kPrepareInputs, device_id_);
  for (size_t i = 0U; i < num_inputs_; ++i) {
    DataBuffer data_buffer;
    uint64_t buffer_size = 0UL;
    if (!IsEventInput(i)) {
      void *m_buf = input_mbuf_addresses_[i];
      void *buffer_data = nullptr;
      GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferAddr(m_buf, &buffer_data));
      GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferSize(m_buf, buffer_size));
      GE_CHECK_GE(buffer_size, sizeof(RuntimeTensorDesc));
      if (DumpManager::GetInstance().CheckDumpFlag()) {
        void *head_buf = nullptr;
        uint64_t head_size = 0U;
        GE_CHK_RT_RET(rtMbufGetPrivInfo(reinterpret_cast<rtMbufPtr_t>(m_buf), &head_buf, &head_size));
        ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
            static_cast<char_t *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
        int32_t worker_id = msg_info->worker_id;
        const auto session_id = GetContext().SessionId();
        std::string dump_worker_id = std::to_string(worker_id);
        (void) DumpManager::GetInstance().SetDumpWorkerId(session_id, dump_worker_id);
        GELOGD("Session id is %d, worker_id is %s", session_id, dump_worker_id.c_str());
      }
      GE_CHK_STATUS_RET(UpdateBufferDataAddr(i, buffer_data, buffer_size),
                        "Update input[%zu] buffer data addr failed.", i);
      GELOGD("Inputs[%zu] buffer size = %zu", i, buffer_size);
      if (is_input_dynamic_[i]) {
        auto &tensor_desc = input_tensor_descs_[i];
        auto *runtime_tensor_desc = reinterpret_cast<const RuntimeTensorDesc *>(buffer_data);
        GE_CHK_STATUS_RET(UpdateTensorDesc(*runtime_tensor_desc, tensor_desc),
                          "Failed to update tensor desc, input index = %zu", i);
        GELOGD("Inputs[%zu] is dynamic, shape = [%s], original shape = [%s]", i,
               tensor_desc.GetShape().ToString().c_str(), tensor_desc.GetOriginShape().ToString().c_str());
      }
      data_buffer.data = static_cast<uint8_t *>(buffer_data) + sizeof(RuntimeTensorDesc);
      data_buffer.length = buffer_size - sizeof(RuntimeTensorDesc);
    } else {
      int64_t input_size = 0L;
      const int32_t align_size = 512;
      GE_CHK_STATUS_RET(HcomOmeUtil::GetAlignedTensorSize(input_tensor_descs_[i], align_size, input_size),
                        "[Get][Size] from TensorDesc of hcom recv op failed, index[%zu].", i);
      data_buffer.length = input_size;
      data_buffer.data = input_buf_addresses_[i - input_queues_num_];
    }
    data_buffer.placement = is_host_ ? kPlacementHost : kPlacementDevice;
    model_inputs.emplace_back(data_buffer);
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kPrepareInputs, device_id_);
  return SUCCESS;
}

Status DynamicModelExecutor::ParseModelDesc(const ComputeGraphPtr &root_graph) {
  GE_CHECK_NOTNULL(root_graph);
  input_tensor_descs_.resize(num_inputs_);
  input_tensor_sizes_.resize(num_inputs_);
  is_input_dynamic_.resize(num_inputs_);
  output_tensor_descs_.resize(num_outputs_);
  output_tensor_sizes_.resize(num_outputs_);
  is_output_dynamic_.resize(num_outputs_);
  output_runtime_tensor_descs_.resize(num_outputs_);
  std::map<int64_t, std::string> data_indices;
  for (const auto &node : root_graph->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      uint32_t index = 0;
      GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index),
                             PARAM_INVALID,
                             "Failed to get attribute \"index\" from data node: %s", node->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(data_indices[index].empty(),
                             PARAM_INVALID,
                             "Duplicated data index [%u], node name = %s, prev node name = %s",
                             index, node->GetName().c_str(), data_indices[index].c_str());
      data_indices[index] = node->GetName();
      GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(index) < is_input_dynamic_.size(),
                             PARAM_INVALID,
                             "Data index of node %s out of range, index = %u num_inputs = %zu",
                             node->GetName().c_str(), index, is_input_dynamic_.size());
      const auto &tensor_desc = node->GetOpDesc()->MutableOutputDesc(0U);
      GE_CHECK_NOTNULL(tensor_desc);
      input_tensor_descs_[index] = *tensor_desc;
      int64_t tensor_size = -1;
      GE_CHK_STATUS_RET_NOLOG(GetTensorSize(*tensor_desc, tensor_size));
      input_tensor_sizes_[index] = tensor_size;
      // string input data is not fixed. it should be got by data size instead of calculation by shape
      is_input_dynamic_[index] = (tensor_desc->GetDataType() == DT_STRING) ? true :
          tensor_desc->MutableShape().IsUnknownShape();
      NamedAttrs align_attr;
      if (AttrUtils::GetNamedAttrs(node->GetOpDesc(), ATTR_NAME_INPUTS_ALIGN_ATTR, align_attr)) {
        GELOGD("Input[%u] has aligned attr", index);
        align_attrs_[index] = align_attr;
      }
      GELOGD("Input[%zu], shape = [%s]", index, tensor_desc->GetShape().ToString().c_str());
    } else if (node->GetType() == NETOUTPUT) {
      size_t output_index = 0UL;
      for (const auto &tensor_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
        const bool output_dynamic_flag = tensor_desc->GetShape().IsUnknownShape();
        is_output_dynamic_[output_index] = output_dynamic_flag;
        int64_t tensor_size = -1;
        GE_CHK_STATUS_RET_NOLOG(GetTensorSize(*tensor_desc, tensor_size));
        output_tensor_sizes_[output_index] = tensor_size;
        output_tensor_descs_[output_index] = *tensor_desc;
        if (!output_dynamic_flag) {
          RuntimeTensorDesc runtime_tensor_desc{};
          GE_CHK_STATUS_RET_NOLOG(UpdateRuntimeTensorDesc(*tensor_desc, runtime_tensor_desc));
          if (tensor_size > 0) {
            runtime_tensor_desc.data_size = static_cast<uint64_t>(tensor_size);
          }
          if ((output_queue_attrs_.size() > output_index) &&
              (output_queue_attrs_[output_index].queue_id != kDummyQId)) {
            output_static_runtime_tensor_descs_.emplace_back(runtime_tensor_desc);
          }
          output_runtime_tensor_descs_[output_index] = runtime_tensor_desc;
        }
        GELOGD("Output[%zu], shape = [%s], size = %ld, is_dynamic = %d", output_index,
               tensor_desc->GetShape().ToString().c_str(), tensor_size, static_cast<int32_t>(output_dynamic_flag));
        output_index++;
      }
    } else {
      // skip other nodes
    }
  }
  return SUCCESS;
}

bool DynamicModelExecutor::IsEventInput(const int64_t index) const {
  return (input_events_num_ > 0) &&
         (index >= static_cast<int64_t>(input_queues_num_));
}

bool DynamicModelExecutor::IsEventOutput(const int64_t index) const {
  return (output_events_num_ > 0) &&
         (index >= static_cast<int64_t>(output_queues_num_));
}

Status DynamicModelExecutor::GetInputAndOutputNum(const ComputeGraphPtr &root_graph,
                                                  const ModelQueueParam &model_queue_param) {
  GE_CHECK_NOTNULL(root_graph);
  input_events_num_ = model_queue_param.input_events.size();
  GE_ASSERT_TRUE(input_events_num_ == 0, "input_events_num_ not equal to 0.");  // delete when delete event situation
  output_events_num_ = model_queue_param.output_events.size();
  GE_ASSERT_TRUE(output_events_num_ == 0, "output_events_num_ not equal to 0.");  // delete when delete event situation
  input_queues_num_ = model_queue_param.input_queues.size();
  output_queues_num_ = model_queue_param.output_queues.size();
  num_inputs_ = input_queues_num_;
  num_outputs_ = output_queues_num_;

  (void)model_queue_param;              
  GELOGD("Load model[%u], input num = [all:%zu, events:%zu, queue:%zu], output num = [all:%zu, events:%zu, queue:%zu]",
         model_id_, num_inputs_, input_events_num_, input_queues_num_, num_outputs_, output_events_num_,
         output_queues_num_);
  return SUCCESS;
}

Status DynamicModelExecutor::UpdateTensorDesc(const RuntimeTensorDesc &runtime_tensor_desc,
                                              GeTensorDesc &tensor_desc) {
  auto num_dims = runtime_tensor_desc.shape[0];
  auto num_ori_dims = runtime_tensor_desc.original_shape[0];
  GE_CHK_BOOL_RET_STATUS((num_dims >= 0) && (num_dims <= kMaxDimSize),
                         UNSUPPORTED,
                         "shape dim number out of range, num_dims = %ld, max = %ld",
                         num_dims, kMaxDimSize);
  GE_CHK_BOOL_RET_STATUS((num_ori_dims >= 0) && (num_ori_dims <= kMaxDimSize),
                         UNSUPPORTED,
                         "original shape dim number out of range, num_dims = %ld, max = %ld",
                         num_dims, kMaxDimSize);
  GeShape shape(std::vector<int64_t>(&runtime_tensor_desc.shape[1], &runtime_tensor_desc.shape[1 + num_dims]));
  if (num_ori_dims == 0) {  // origin shape not set, or is scalar(in which case original format equals to format)
    tensor_desc.SetOriginShape(shape);
  } else {
    GeShape ori_shape(std::vector<int64_t>(&runtime_tensor_desc.original_shape[1],
        &runtime_tensor_desc.original_shape[1 + num_dims]));
    tensor_desc.SetOriginShape(ori_shape);
  }
  tensor_desc.SetShape(std::move(shape));
  return SUCCESS;
}

Status DynamicModelExecutor::UpdateRuntimeTensorDesc(const GeTensorDesc &tensor_desc,
                                                     RuntimeTensorDesc &runtime_tensor_desc) {
  GE_CHK_STATUS_RET_NOLOG(UpdateRuntimeShape(tensor_desc.GetShape(), runtime_tensor_desc.shape));
  GE_CHK_STATUS_RET_NOLOG(UpdateRuntimeShape(tensor_desc.GetOriginShape(), runtime_tensor_desc.original_shape));
  runtime_tensor_desc.dtype = static_cast<int64_t>(tensor_desc.GetDataType());
  return SUCCESS;
}

Status DynamicModelExecutor::UpdateRuntimeShape(const GeShape &shape, int64_t (&shape_buffer)[33]) {
  auto num_dims = static_cast<int64_t>(shape.GetDimNum());
  GE_CHK_BOOL_RET_STATUS(num_dims <= kMaxDimSize,
                         UNSUPPORTED,
                         "shape dim number out of range, num_dims = %ld, max = %ld",
                         num_dims, kMaxDimSize);
  shape_buffer[0] = num_dims;
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    shape_buffer[1 + i] = shape.GetDim(i);
  }
  return SUCCESS;
}

Status DynamicModelExecutor::CopyMbufHead(rtMbufPtr_t src, rtMbufPtr_t dst) {
  void *src_head_buf = nullptr;
  uint64_t src_head_size = 0U;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetPrivData(src, &src_head_buf, &src_head_size));
  void *dst_head_buf = nullptr;
  uint64_t dst_head_size = 0U;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetPrivData(dst, &dst_head_buf, &dst_head_size));
  if ((src_head_size == dst_head_size) && (src_head_size != 0) &&
      (src_head_buf != nullptr) && (dst_head_buf != nullptr)) {
    if (memcpy_s(dst_head_buf, dst_head_size, src_head_buf, src_head_size) == EOK) {
      return SUCCESS;
    }
  }
  GELOGE(FAILED, "Copy mbuf head failed, src head size = %lu, dst head size = %lu.",
         src_head_size, dst_head_size);
  return FAILED;
}

Status DynamicModelExecutor::PrepareOutputs(std::vector<DataBuffer> &model_outputs) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kPrepareOutputs, device_id_);
  for (size_t i = 0; i < num_outputs_; ++i) {
    auto tensor_size = output_tensor_sizes_[i];
    if (tensor_size < 0) { // no valid range
      GELOGD("Output[%zu] is dynamic and cannot get a valid size by range.", i);
      output_mbuf_addresses_[i] = nullptr;
      model_outputs.emplace_back(DataBuffer{});
      continue;
    }

    DataBuffer data_buffer;
    data_buffer.length = tensor_size;
    uint64_t buffer_size = tensor_size + sizeof(RuntimeTensorDesc);
    // recv mbuf has been alloc when load model, skip alloc
    if (!IsEventOutput(i)) {
      if ((i < output_queue_attrs_.size()) && (output_queue_attrs_[i].queue_id == kDummyQId)) {
        model_outputs.emplace_back(data_buffer);
        continue;
      }
      GE_CHK_RT_RET(rtMbufAlloc(&output_mbuf_addresses_[i], buffer_size));
      GE_CHK_RT_RET(rtMbufSetDataLen(output_mbuf_addresses_[i], buffer_size));
      GE_CHK_RT_RET(CopyMbufHead(input_mbuf_addresses_.back(), output_mbuf_addresses_[i]));
      GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferAddr(output_mbuf_addresses_[i], &data_buffer.data));
      data_buffer.data = static_cast<uint8_t *>(data_buffer.data) + sizeof(RuntimeTensorDesc);
    } else {
      data_buffer.data = output_buf_addresses_[i - output_queues_num_];
    }
    data_buffer.placement = is_host_ ? kPlacementHost : kPlacementDevice;
    GELOGD("Output[%zu] is dynamic = %d, data buffer size = %zu", i,
           static_cast<int32_t>(is_output_dynamic_[i]), data_buffer.length);
    model_outputs.emplace_back(data_buffer);
    // mbuf will be freed by consumer
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kPrepareOutputs, device_id_);
  return SUCCESS;
}

Status DynamicModelExecutor::GetTensorSize(const GeTensorDesc &tensor_desc, int64_t &tensor_size) {
  if (tensor_desc.GetShape().IsUnknownShape()) {
    int64_t output_max_size = 0L;
    if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_GRAPH_OUTPUT_MAX_SIZE, output_max_size) && output_max_size != 0L) {
      tensor_size = output_max_size;
      GELOGD("Success to get tensor_size:%ld from attr:%s.", tensor_size, ATTR_NAME_GRAPH_OUTPUT_MAX_SIZE.c_str());
      return SUCCESS;
    }
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    (void) tensor_desc.GetShapeRange(shape_range);
    if (shape_range.empty()) {
      GELOGD("dynamic shape tensor without range. shape = [%s].", tensor_desc.GetShape().ToString().c_str());
      tensor_size = -1;
      return SUCCESS;
    }
  }
  GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSizeForNoTiling(tensor_desc,
                                                              tensor_desc.GetFormat(),
                                                              tensor_desc.GetDataType(),
                                                              tensor_size),
                    "Failed to calc tensor size, shape = [%s]", tensor_desc.GetShape().ToString().c_str());
  return SUCCESS;
}

Status DynamicModelExecutor::UpdateOutputs(std::vector<DataBuffer> &model_outputs) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kUpdateOutputs, device_id_);
  for (size_t i = 0; i < num_outputs_; ++i) {
    // recv mbuf has been alloc when load model(only support static shape), skip alloc
    if (IsEventOutput(i)) {
      continue;
    }
    if ((i < output_queue_attrs_.size()) && (output_queue_attrs_[i].queue_id == kDummyQId)) {
      continue;
    }
    auto &tensor_desc = output_tensor_descs_[i];
    void *buffer_addr = nullptr;
    if (output_mbuf_addresses_[i] == nullptr) {
      auto &data_buffer = model_outputs[i];
      auto buffer_size = sizeof(RuntimeTensorDesc) + data_buffer.length;
      GE_CHK_RT_RET(rtMbufAlloc(&output_mbuf_addresses_[i], buffer_size));
      GE_CHK_RT_RET(rtMbufSetDataLen(output_mbuf_addresses_[i], buffer_size));
      GE_CHK_RT_RET(CopyMbufHead(input_mbuf_addresses_.back(), output_mbuf_addresses_[i]));
      GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferAddr(output_mbuf_addresses_[i], &buffer_addr));
      GELOGD("output[%zu] was allocated by executor, Mbuf allocated, size = %zu", i, buffer_size);
      if (data_buffer.length > 0) {
        GE_CHK_BOOL_RET_STATUS(memcpy_s(static_cast<uint8_t *>(buffer_addr) + sizeof(RuntimeTensorDesc),
                                        data_buffer.length,
                                        data_buffer.data,
                                        data_buffer.length) == EOK,
                               FAILED, "Failed to copy output[%zu]", i);
      }
      GELOGD("Copy output[%zu] succeeded, size = %zu", i, data_buffer.length);
    } else {
      GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferAddr(output_mbuf_addresses_[i], &buffer_addr));
    }

    if (is_output_dynamic_[i]) {
      // update mbuf len
      int64_t data_len = 0L;
      GE_CHK_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc.GetShape(), tensor_desc.GetFormat(),
                                                       tensor_desc.GetDataType(), data_len),
                        "Failed to calc output size, shape = [%s]",
                        tensor_desc.GetShape().ToString().c_str());
      GE_CHK_RT_RET(rtMbufSetDataLen(output_mbuf_addresses_[i],
                                     data_len + static_cast<uint64_t>(sizeof(RuntimeTensorDesc))));
      GE_CHK_STATUS_RET_NOLOG(UpdateRuntimeTensorDesc(tensor_desc, output_runtime_tensor_descs_[i]));
    }
    GE_CHK_BOOL_RET_STATUS(memcpy_s(buffer_addr,
                                    sizeof(RuntimeTensorDesc),
                                    &output_runtime_tensor_descs_[i],
                                    sizeof(output_runtime_tensor_descs_[i])) == EOK,
                           FAILED,
                           "Failed to copy runtime tensor desc");
    GELOGD("Output[%zu] is dynamic, shape = [%s], original shape = [%s], data_type = [%d]",
           i,
           tensor_desc.GetShape().ToString().c_str(),
           tensor_desc.GetOriginShape().ToString().c_str(),
           static_cast<int32_t>(tensor_desc.GetDataType()));
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kUpdateOutputs, device_id_);
  return SUCCESS;
}

Status DynamicModelExecutor::LoadWithAicpuSd(const ComputeGraphPtr &root_graph,
                                             const ModelQueueParam &model_queue_param) {
  GE_CHK_STATUS_RET(CreateFakeAicpuModelAndStream(), "Failed to create aicpu model and stream");
  CpuSchedEventDispatcher::GetInstance().Register(aicpu_model_id_, this);

  CpuSchedModelBuilder builder(model_);
  builder.SetModelId(aicpu_model_id_);
  builder.SetAicpuStreamId(aicpu_stream_id_);
  builder.SetIsHost(is_host_);
  if (!align_attrs_.empty()) {
    uint32_t align_interval;
    std::vector<uint32_t> align_offsets;
    GE_CHK_STATUS_RET_NOLOG(CheckAndGetAlignAttr(align_interval, align_offsets));
    builder.SetAlignAttributes(align_interval, align_offsets);
  }

  std::set<uint32_t> unique_qids;
  for (size_t i = 0; i < input_queues_num_; ++i) {
    auto queue_id = input_queue_attrs_[i].queue_id;
    const auto &it = unique_qids.find(queue_id);
    if (it != unique_qids.cend()) {
      GELOGD("Input[%zu] is fusion tensor, queue_id = %u.", i, queue_id);
      continue;
    }
    (void) unique_qids.emplace(queue_id);
    auto mbuf_addr = reinterpret_cast<uintptr_t>(&input_mbuf_addresses_[i]);
    if ((input_queue_attrs_[i].device_type == NPU) && is_host_) {
      GELOGD("Current input queue %u is client queue. device id is %d", queue_id, input_queue_attrs_[i].device_id);
      builder.AddInputClientQueue(input_queue_attrs_[i], mbuf_addr);
    } else {
      builder.AddInputQueue(input_queue_attrs_[i], mbuf_addr);
    }
    GELOGD("Add input[%zu] queue success, queue_id = %u.", i, queue_id);
  }

  for (size_t i = 0; i < output_queues_num_; ++i) {
    uint32_t queue_id = output_queue_attrs_[i].queue_id;
    if (queue_id == kDummyQId) {
      continue;
    }
    auto mbuf_addr = reinterpret_cast<uintptr_t>(&output_mbuf_addresses_[i]);
    if ((output_queue_attrs_[i].device_type == NPU) && is_host_) {
      GELOGD("Current output queue %u is client queue. device id is %d", queue_id, output_queue_attrs_[i].device_id);
      builder.AddOutputClientQueue(output_queue_attrs_[i], mbuf_addr);
    } else {
      builder.AddOutputQueue(output_queue_attrs_[i], mbuf_addr);
    }
    GELOGD("Add output[%zu] queue success, queue_id = %u.", i, queue_id);
  }

  builder.SetModelQueueParam(model_queue_param);
  builder.SetInputBufferAddrs(input_buf_addresses_);
  builder.SetOutputBufferAddrs(output_buf_addresses_);
  builder.SetInputTensor(input_tensor_descs_);
  builder.SetOutputTensor(output_tensor_descs_);
  builder.SetGlobalStep(global_step_);
  if (input_align_attrs_.align_max_cache_num > 0U) {
    bool is_gather_supported = false;
    GE_CHK_STATUS_RET(CheckAicpuKernelSupported(kGatherDequeue, is_gather_supported));
    GE_ASSERT_TRUE(is_gather_supported, "Gather dequeue is not supported in current version. "
                   "Please update software or unset input align attrs.");
  }
  builder.SetInputAlignAttrs(input_align_attrs_);
  GE_CHK_STATUS_RET(builder.Build(), "Failed to build CpuSchedModel");

  model_.LogModelDesc();
  const std::string kLoadFunc =
      ((input_events_num_ + output_events_num_) == 0UL) ? "AicpuLoadModelWithQ" : "AicpuLoadModel";
  const auto load_func = reinterpret_cast<int32_t (*)(void *)>(mmDlsym(aicpu_handle_, kLoadFunc.c_str()));
  GE_CHECK_NOTNULL(load_func);
  int32_t ret = load_func(&model_.model_info_);
  if (ret != 0) {
    GELOGE(FAILED, "Failed to invoke AicpuLoadModelWithQ, ret = %d", ret);
    return FAILED;
  }

  GEEVENT("[LoadWithAicpuSd] success, model_id = %u, model_name = %s, device_id = %d, aicpu model_id = %u", model_id_,
          root_graph->GetName().c_str(), device_id_, aicpu_model_id_);
  return SUCCESS;
}
Status DynamicModelExecutor::CheckAicpuKernelSupported(const std::string &kernel_name, bool &is_supported) const {
  const std::string kCheckFunc = "CheckKernelSupported";
  const auto check_func = reinterpret_cast<int32_t (*)(void *)>(mmDlsym(aicpu_handle_, kCheckFunc.c_str()));
  GE_ASSERT_TRUE(check_func != nullptr, "Interface[%s] is not supported in aicpu scheduler.", kernel_name.c_str());

  int32_t result = -1;
  CheckKernelSupportedConfig check_cfg{};
  check_cfg.kernelNameAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernel_name.c_str()));
  check_cfg.kernelNameLen = kernel_name.length();
  check_cfg.checkResultAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&result));
  check_cfg.checkResultLen = sizeof(result);

  int32_t ret = check_func(&check_cfg);
  if (ret != 0) {
    GELOGE(FAILED, "Failed to invoke check kernel supported function, ret = %d", ret);
    return FAILED;
  }
  is_supported = (result == 0);
  return SUCCESS;
}

Status DynamicModelExecutor::CheckAndGetAlignAttr(uint32_t &align_interval, std::vector<uint32_t> &align_offsets) {
  GE_CHK_BOOL_RET_STATUS(align_attrs_.size() == num_inputs_,
                         PARAM_INVALID,
                         "Number of align attr(%zu) mismatches that of inputs(%zu)",
                         align_attrs_.size(), num_inputs_);

  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(align_attrs_.begin()->second,
                                           ATTR_NAME_INPUTS_ALIGN_INTERVAL,
                                           align_interval),
                         FAILED, "Failed to get attr: %s", ATTR_NAME_INPUTS_ALIGN_INTERVAL.c_str());

  uint32_t data_index = 0U;
  for (const auto &it : align_attrs_) {
    const auto &align_attr = it.second;
    uint32_t align_offset = 0;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(align_attr, ATTR_NAME_INPUTS_ALIGN_OFFSET, align_offset),
                           PARAM_INVALID, "Failed to get attr: %s, input_index = %u",
                           ATTR_NAME_INPUTS_ALIGN_OFFSET.c_str(), data_index);
    GELOGD("Input index = %u, align_offset = %u.", data_index, align_offset);
    align_offsets.emplace_back(align_offset);
    ++data_index;
  }
  return SUCCESS;
}

void DynamicModelExecutor::PublishErrorOutput(Status ret) {
  data_ret_code_ = ret;
  if (is_need_alloc_output_mbuf_) {
    FreeOutputs();
  }
  if (PublishOutputWithoutExecute() != SUCCESS) {
    FreeOutputs();
  }
}

bool DynamicModelExecutor::StopAndWaitRestart() {
  if (stop_schedule_flag_) {
    has_stop_schedule_ = true;
    while (stop_schedule_flag_) {}
    return true;
  }
  return false;
}

void DynamicModelExecutor::Run() {
  GELOGD("Run thread started, model_id = %u", model_id_);
  aclrtSetCurrentContext(rt_context_);
  GELOGD("current rt_context_ is %p, stream is %p.", rt_context_, stream_);
  while (true) {
    task_queue_.Pop(model_execute_param_);
    if (StopAndWaitRestart()) {
      continue;
    }
    if (model_execute_param_.callback == nullptr) {
      GELOGI("Got EOF, model_id = %u", model_id_);
      break;
    }
    GELOGD("Start to execute model, model_id = %u", model_id_);
    auto ret = ExecuteInternal();
    if (ret == SUCCESS) {
      GELOGD("Execute model successfully, model_id = %u", model_id_);
    } else {
      aclrtContext rt_ctx = nullptr;
      aclrtGetCurrentContext(&rt_ctx);
      GELOGD("current rt_context is %p, old rt_context is %p, stream is %p.", rt_ctx, rt_context_, stream_);
      GELOGE(ret, "Failed to execute model, model_id = %u", model_id_);
      PublishErrorOutput(ret);
    }
    DestroyDatasetResource();
    model_execute_param_.callback(ret, model_execute_param_.req_mbuf, model_execute_param_.resp_mbuf);
    GELOGD("callback finished");
    if (need_report_status_) {
      ret = ReportStatus();
      if (ret != SUCCESS) {
        aclrtContext rt_ctx = nullptr;
        aclrtGetCurrentContext(&rt_ctx);
        GELOGD("current rt_context is %p, old rt_context is %p, stream is %p.", rt_ctx, rt_context_, stream_);
        GELOGE(ret, "Failed to report status, model_id = %u", model_id_);
        PublishErrorOutput(ret);
      }
    }
  }
  (void)FreeEventIOBuffer();
  GELOGD("Run thread exit");
}

Status DynamicModelExecutor::ReportStatus() {
  input_consume_num_++;
  // construct SubmodelStatus protobuf object
  deployer::SubmodelStatus submodel_status;
  submodel_status.set_model_uuid(model_uuid_);
  for (const auto input_queue : input_queue_attrs_) {
    const uint32_t input_queue_id = input_queue.queue_id;
    uint32_t queue_depth = UINT32_MAX;
    rtMemQueueInfo_t info;
    const auto ret = rtMemQueueQueryInfo(device_id_, input_queue_id, &info);
    if (ret != RT_ERROR_NONE) {
      GELOGI("query queue info failed, queue id is %u, device id is %d, ret is %d.",
        input_queue_id, device_id_, ret);
    } else {
      queue_depth = info.size;
    }
    auto queue_status = submodel_status.add_queue_statuses();
    queue_status->set_queue_depth(queue_depth);
    queue_status->set_input_consume_num(input_consume_num_);
    auto queue_attrs = queue_status->mutable_queue_attrs();
    queue_attrs->set_queue_id(input_queue_id);
    queue_attrs->set_device_type(input_queue.device_type);
    queue_attrs->set_device_id(input_queue.device_id);
    queue_attrs->set_logic_id(input_queue.logic_id);
  }
  // enqueue
  ExchangeService::MsgInfo msg_info{};
  ExchangeService::ControlInfo control_info;
  control_info.msg_info = &msg_info;
  control_info.timeout = kReportStatusEnqueueTimeout;
  control_info.print_error_flag = false;
  ExchangeService::FillFunc fill_func = [&submodel_status](void *buffer, size_t size) {
    GE_CHK_BOOL_RET_STATUS(submodel_status.SerializeToArray(buffer, static_cast<int32_t>(size)),
      FAILED, "dynamic sched serialize to array failed.");
    return SUCCESS;
  };
  const auto req_size = submodel_status.ByteSizeLong();
  const auto enqueue_ret = HeterogeneousExchangeService::GetInstance().Enqueue(
      status_output_queue_device_id_, status_output_queue_id_, req_size, fill_func, control_info);
  if (enqueue_ret == SUCCESS) {
    input_consume_num_ = 0U;
  } else if (enqueue_ret != RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_FULL)) {
      GELOGE(enqueue_ret, "Failed to enqueue, device id is %d, queue id is %u, ret is %u.",
             status_output_queue_device_id_, status_output_queue_id_, enqueue_ret);
      return enqueue_ret;
  }

  GELOGI("dynamic sched report status, ret is %u, status is %s, device id is %d, " \
    "queue id is %u.", enqueue_ret, submodel_status.DebugString().c_str(), status_output_queue_device_id_,
      status_output_queue_id_);
  return SUCCESS;
}

void DynamicModelExecutor::Stop() {
  const ModelExecuteParam eof_param {.callback= nullptr, .req_mbuf = nullptr, .resp_mbuf = nullptr};
  task_queue_.Push(eof_param);
  if (run_thread_.joinable()) {
    run_thread_.join();
  }
  if (is_host_ && (new_allocated_global_step_ != nullptr)) {
    free(new_allocated_global_step_);
  } else if (new_allocated_global_step_ != nullptr) {
    (void)aclrtFree(new_allocated_global_step_);
  }
  new_allocated_global_step_ = nullptr;
  GELOGI("Global step is allocated in dynamic model executor which need to be deallocated when executor stopping");
}

Status DynamicModelExecutor::CreateFakeAicpuModelAndStream() {
  if (is_host_) {
    if (aicpu_model_id_ == UINT32_MAX) {
      GE_CHK_STATUS_RET(AicpuModelIdResourceManager::GetInstance().GenerateAicpuModelId(aicpu_model_id_),
                        "Generate aicpu model id failed");
      aicpu_model_ids_.emplace_back(aicpu_model_id_);
      GE_CHECK_LE(aicpu_model_id_, kDynamicModelMaxIdBase);
    }
  } else {
    if (aicpu_model_handle_ == nullptr) {
      DF_CHK_ACL_RET(aclmdlRIBuildBegin(&aicpu_model_handle_, 0U));
      GE_CHK_RT_RET(rtModelGetId(aicpu_model_handle_, &aicpu_model_id_));
    }
    if (aicpu_stream_ == nullptr) {
      uint32_t stream_flags = ACL_STREAM_CPU_SCHEDULE | ACL_STREAM_PERSISTENT;
      DF_CHK_ACL_RET(aclrtCreateStreamWithConfig(&aicpu_stream_, kDefaultStreamPriority, stream_flags));
      DF_CHK_ACL_RET(aclrtStreamGetId(aicpu_stream_, &aicpu_stream_id_));
    }
  }
  GELOGI("[Create][Fake] aicpu model and stream success, model id:%u, stream id:%d", aicpu_model_id_, aicpu_stream_id_);
  return SUCCESS;
}

Status DynamicModelExecutor::DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph) {
  int32_t device_id = is_host_ ? GetContext().DeviceId() : device_id_;
  aclError ret = aclrtSetDevice(device_id);
  GE_ASSERT_TRUE(ret == ACL_SUCCESS, "ACL set device id failed.");
  rtCtxSetCurrent(rt_context_);
  GE_CHK_STATUS_RET(InitExternalWeightMem(root_graph, external_weight_mem_data_), "Failed to init external weright mem.");
  handle_ = aclmdlCreateConfigHandle();
  GE_CHECK_NOTNULL(handle_, "Create acl load config handle failed.");
  GE_CHK_STATUS_RET(GenerateLoadConfig(model_data, external_weight_mem_data_, handle_));
  ret = aclmdlLoadWithConfig(handle_, &model_id_);
  if (ret != ACL_SUCCESS) {
    GELOGE(FAILED, "Failed to load model");
    (void) aclmdlDestroyConfigHandle(handle_);
    handle_ = nullptr;
    return FAILED;
  }
  GELOGI("Load model[%u] on device[%u] success.", model_id_, device_id);
  return SUCCESS;
}

Status DynamicModelExecutor::GenerateLoadConfig(const ModelData &model_data, const std::vector<FileConstantMem> &external_weight_mem_data, aclmdlConfigHandle *handle) {
  GELOGD("[GenerateLoadConfig] Start to generate acl type load config.");
  aclError ret;
  GE_CHECK_NOTNULL(handle, "Failed to create acl config handle.");
  for (auto &external_weight : external_weight_mem_data) {
    ret = aclmdlSetExternalWeightAddress(handle, external_weight.file_name.c_str(), 
                                                    const_cast<void *>(external_weight.device_mem), external_weight.mem_size);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to set acl external weight address.");
  }
  size_t load_type = ACL_MDL_LOAD_FROM_MEM;
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &load_type, sizeof(load_type));
  GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to set acl load option ACL_MDL_LOAD_TYPE_SIZET.");
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &model_data.model_data, sizeof(void *));
  GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to set acl load option ACL_MDL_MEM_ADDR_PTR.");
  size_t model_len = static_cast<size_t>(model_data.model_len);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &model_len, sizeof(size_t));
  GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to set acl load option ACL_MDL_MEM_SIZET.");
  GELOGD("[GenerateLoadConfig] Succeed to generate acl type load config.");
  return SUCCESS;
}

Status DynamicModelExecutor::InitExternalWeightMem(const ComputeGraphPtr &root_graph, std::vector<FileConstantMem> &external_weight_mem_data) {
  GELOGD("[InitExternalWeightMem] Start to init extrnal weight mem.");
  // load external weight
  for (const auto &node : root_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetType() != FILECONSTANT) {
      continue;
    }

    FileConstantMem external_weight{};
    std::string fileconstant_name;
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_LOCATION, fileconstant_name);
    if (fileconstant_name.empty()) {
      GELOGE(PARAM_INVALID, "File constant name invalid.");
      return PARAM_INVALID;
    }
    int64_t attr_length = 0;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_LENGTH, attr_length);
    if ((attr_length < 0) || (attr_length >= INT64_MAX)) {
      GELOGE(PARAM_INVALID, "Data length out of range, data length = %ld", attr_length);
      return PARAM_INVALID;
    }

    auto file_name = RealPath(fileconstant_name.c_str());
    if (file_name.empty()) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "The path[%s]is invalid", fileconstant_name.c_str());
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    external_weight.file_name = file_name;
    external_weight.mem_size = static_cast<size_t>(attr_length);
    external_weight.device_mem = nullptr;

    auto host_buffer = ge::MakeUnique<char_t[]>(attr_length);
    GE_CHK_STATUS_RET(FileConstantUtils::ReadExternalWeightFromFile(file_name , 0, attr_length, host_buffer.get()));

    auto alloc_size = (attr_length + 32 - 1) / 32 * 32;
    void *data_addr = nullptr;
    aclError ret = aclrtMalloc(&data_addr, alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to malloc device mem.");

    // copy to device
    ret = aclrtMemcpy(data_addr, attr_length,
                    host_buffer.get(), attr_length, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      GELOGE(FAILED, "Failed to copy host buffer to device.");
      (void)aclrtFree(data_addr);
      return FAILED;
    }
    external_weight.device_mem = data_addr;
    external_weight_mem_data.emplace_back(external_weight);
    GELOGD("Success initialize external weight mem from file[%s], length[%lu]", file_name.c_str(), attr_length);
  }
  GELOGD("[InitExternalWeightMem] Succeed to init extrnal weight mem.");
  return SUCCESS;
}

Status DynamicModelExecutor::CreateInputDataset(const std::vector<DataBuffer> &inputs) {
  // acl type input dataset
  GELOGD("[CreateInputDataset] Start to acl type create input dataset.");
  aclError ret;
  input_data_buffer_.resize(num_inputs_);
  acl_tensor_desc_.resize(num_inputs_);
  input_dataset_ = aclmdlCreateDataset();
  GE_CHECK_NOTNULL(input_dataset_);
  for (size_t i = 0U; i < num_inputs_; ++i){
    input_data_buffer_[i] = aclCreateDataBuffer(inputs[i].data, static_cast<size_t>(inputs[i].length));
    GE_CHECK_NOTNULL(input_data_buffer_[i]);
    ret = aclmdlAddDatasetBuffer(input_dataset_, input_data_buffer_[i]);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to add acl input databuffer to dataset.");
  }
  // add input tensor desc
  for (size_t i = 0; i < num_inputs_; ++i) {
    acl_tensor_desc_[i] = aclCreateTensorDesc(
      static_cast<aclDataType>(input_tensor_descs_[i].GetDataType()),
      input_tensor_descs_[i].GetShape().GetDims().size(),
      input_tensor_descs_[i].GetShape().GetDims().data(),
      static_cast<aclFormat>(input_tensor_descs_[i].GetFormat())
    );
    ret = aclmdlSetDatasetTensorDesc(input_dataset_, acl_tensor_desc_[i], i);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to add input tensor desc to input dataset.");
  }
  GELOGD("[CreateInputDataset] Succeed to acl type create input dataset.");
  return SUCCESS;
}

Status DynamicModelExecutor::CreateOutputDataset(const std::vector<DataBuffer> &outputs) {
  GELOGD("[CreateOutputDataset] Start to acl type create output dataset.");
  // acl type output
  output_data_buffer_.resize(num_outputs_);
  model_desc_ = aclmdlCreateDesc();
  GE_CHECK_NOTNULL(model_desc_);
  auto ret = aclmdlGetDesc(model_desc_, model_id_);
  GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to get acl model description.");

  output_dataset_ = aclmdlCreateDataset();
  GE_CHECK_NOTNULL(output_dataset_);
  for (size_t i = 0; i < num_outputs_; ++i){
    output_data_buffer_[i] = aclCreateDataBuffer(outputs[i].data, static_cast<size_t>(outputs[i].length));
    GE_CHECK_NOTNULL(output_data_buffer_[i]);
    ret = aclmdlAddDatasetBuffer(output_dataset_, output_data_buffer_[i]);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to add acl output databuffer to dataset.");
  }
  GELOGD("[CreateOutputDataset] Succeed to acl type create output dataset.");
  return SUCCESS;
}

Status DynamicModelExecutor::DoExecuteModel(const std::vector<DataBuffer> &inputs, std::vector<DataBuffer> &outputs) {
  GE_CHK_STATUS_RET(CreateInputDataset(inputs), "Failed to prepare acl type input dataset.");
  GE_CHK_STATUS_RET(CreateOutputDataset(outputs), "Failed to prepare acl type output dataset.");
  rtCtxSetCurrent(rt_context_);
  auto ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
  GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to execute model.");

  // parse output tensor_desc
  for (size_t i = 0; i < num_outputs_; ++i) {
    aclTensorDesc *acl_tensor_desc = aclmdlGetDatasetTensorDesc(output_dataset_, i);
    GE_CHK_STATUS_RET(ParseModelOutputToTensorDesc(acl_tensor_desc, output_tensor_descs_[i]));
    aclDataBuffer *output_tensor_data = aclmdlGetDatasetBuffer(output_dataset_, i);
    outputs[i].data = aclGetDataBufferAddr(output_tensor_data);
    outputs[i].length = static_cast<uint64_t>(aclGetDataBufferSizeV2(output_tensor_data));
  }

  GELOGI("Execute model[%u] success.", model_id_);
  return SUCCESS;
}

Status DynamicModelExecutor::ParseModelOutputToTensorDesc(const aclTensorDesc *acl_tensor_desc, GeTensorDesc &tensor_desc) const {
  tensor_desc.SetFormat(static_cast<Format>(aclGetTensorDescFormat(acl_tensor_desc)));
  tensor_desc.SetDataType(static_cast<DataType>(aclGetTensorDescType(acl_tensor_desc)));
  std::vector<int64_t> tensor_shape;
  size_t num_dims = aclGetTensorDescNumDims(acl_tensor_desc);
  for (size_t i = 0; i < num_dims; ++i) {
    int64_t dim;
    aclError ret = aclGetTensorDescDimV2(acl_tensor_desc, i, &dim);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to get dim at index[%lu].", dim);
    tensor_shape.emplace_back(dim);
  }
  GeShape shape(tensor_shape);
  tensor_desc.SetShape(shape);
  GELOGI("Successfully parse model output tensor desc. shape = [%s]", shape.ToString().c_str());
  return SUCCESS;
}

Status DynamicModelExecutor::GetGlobalStepAddr() {
  int32_t device_id = -1;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  GEEVENT("Current process procedure maybe runtime 2.0. Create global_step memory now.");
  if (is_host_) {
    GELOGI("Alloc global step memory for host cpu model.");
    new_allocated_global_step_ = malloc(sizeof(int64_t));
  } else {
    aclError ret = aclrtMalloc(&new_allocated_global_step_, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    GE_ASSERT_TRUE(ret == ACL_SUCCESS, "Failed to malloc device mem.");
  }
  GE_CHECK_NOTNULL(new_allocated_global_step_);
  global_step_ = PtrToValue(new_allocated_global_step_);
  GELOGI("Create global step success.");
  return SUCCESS;
}

void DynamicModelExecutor::SetModelEschedPriority(int32_t esched_process_priority, int32_t esched_event_priority) {
  esched_process_priority_ = esched_process_priority;
  esched_event_priority_ = esched_event_priority;
}

void DynamicModelExecutor::SetModelExecuteTimes(int32_t execute_times) {
  execute_times_ = execute_times;
}

void DynamicModelExecutor::FreeOutputs() {
  for (size_t i = 0U; i < output_mbuf_addresses_.size(); ++i) {
    if (output_mbuf_addresses_[i] != nullptr) {
      GE_CHK_RT(rtMbufFree(output_mbuf_addresses_[i]));
      output_mbuf_addresses_[i] = nullptr;
    }
  }
}

void DynamicModelExecutor::ClearOutputs() {
  for (size_t i = 0U; i < output_mbuf_addresses_.size(); ++i) {
    output_mbuf_addresses_[i] = nullptr;
  }
}

Status DynamicModelExecutor::ClearModelInner(const int32_t clear_type) {
  if (clear_type == kClearTypeStop) {
    GE_CHK_STATUS_RET(StopSchedule(), "Fail to stop schedule.");
  } else if (clear_type == kClearTypeClear) {
    GE_CHK_STATUS_RET(ClearAndRestart(), "Fail to clear model and restart.");
  }
  return SUCCESS;
}

Status DynamicModelExecutor::ClearModel(const int32_t clear_type) {
  GE_CHK_STATUS_RET_NOLOG(ClearModelInner(clear_type));
  GE_CHK_STATUS_RET(AicpuClearModel(clear_type), "Fail to clear aicpu model.");
  return SUCCESS;
}

Status DynamicModelExecutor::StopSchedule() {
  stop_schedule_flag_ = true;
  const ModelExecuteParam clear_param {.callback= nullptr, .req_mbuf = nullptr, .resp_mbuf = nullptr};
  // just for wake, will not be execute
  task_queue_.Push(clear_param);
  while (!has_stop_schedule_) {}
  return SUCCESS;
}

Status DynamicModelExecutor::ClearAndRestart() {
  ModelExecuteParam param;
  while (task_queue_.Pop(param, 0)) {}
  has_stop_schedule_ = false;
  stop_schedule_flag_ = false;
  return SUCCESS;
}

Status DynamicModelExecutor::AicpuClearModel(const int32_t clear_type) {
  const char *aicpuModelClearFuncName = nullptr;
  if (clear_type == kClearTypeStop) {
    aicpuModelClearFuncName = kModelStopFunc;
  } else if (clear_type == kClearTypeClear) {
    aicpuModelClearFuncName = kModelClearFunc;
  }
  const auto clear_func =
    reinterpret_cast<int32_t (*)(const ReDeployConfig *const)>(mmDlsym(aicpu_handle_,
    aicpuModelClearFuncName));
  GE_CHECK_NOTNULL(clear_func);
  ReDeployConfig config;
  config.modelIdNum = 1U;
  config.modelIdsAddr = PtrToValue(&aicpu_model_id_);
  const int32_t clear_ret = clear_func(&config);
  GE_CHK_BOOL_RET_STATUS((clear_ret == 0), FAILED,
    "Failed to execute aicpu func: %s, ret: %d", aicpuModelClearFuncName, clear_ret);
  return SUCCESS;
}

Status DynamicModelExecutor::CheckLocalAicpuSupportExceptionNotify() const {
  const auto notify_func = reinterpret_cast<int32_t (*)(const DataFlowExceptionNotify *const)>(
      mmDlsym(aicpu_handle_, kModelProcessDataException));
  GE_CHECK_NOTNULL(notify_func);
  return SUCCESS;
}

Status DynamicModelExecutor::ExceptionNotify(uint32_t type, uint64_t trans_id) {
  const auto notify_func = reinterpret_cast<int32_t (*)(const DataFlowExceptionNotify *const)>(
      mmDlsym(aicpu_handle_, kModelProcessDataException));
  GE_CHECK_NOTNULL(notify_func);
  DataFlowExceptionNotify notify_info{};
  notify_info.transId = trans_id;
  notify_info.type = type;
  notify_info.modelIdNum = 1U;
  notify_info.modelIdsAddr = PtrToValue(&aicpu_model_id_);
  const int32_t notify_ret = notify_func(&notify_info);
  GE_CHK_BOOL_RET_STATUS((notify_ret == 0), FAILED, "Failed to execute aicpu func: %s, ret: %d",
                         kModelProcessDataException, notify_ret);
  GELOGI("notify exception to aicpu success, trans_id=%lu, type=%u, aicpu_model_id=%u", trans_id, type,
         aicpu_model_id_);
  return SUCCESS;
}
}  // namespace ge
