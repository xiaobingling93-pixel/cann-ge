/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include <thread>
#include <chrono>
#include "common/profiling/profiling_properties.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "framework/common/runtime_tensor_desc.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/utils/rts_api_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/mem_utils.h"
#include "graph/ge_context.h"
#include "common/utils/heterogeneous_profiler.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"

#include "acl/acl.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr int32_t kQueueOpTimeout = 10 * 60 * 1000;  // 10 min
constexpr uint32_t kBufferCount = 1U;
constexpr size_t kContextLen = 0U;
constexpr size_t kAlignmentVal64 = 64U;
constexpr uint32_t kMbufHeadMaxSize = 256U;
constexpr uint32_t kMbufHeadEndOfSequencePos = 128U;
constexpr uint8_t kEndOfSequenceFlag = 0x5A;
constexpr uint32_t kEventGroupId = 3U;
constexpr int32_t kWaitEventTimeout = 1000; // 1s
constexpr int32_t kRtQueueTypeSingle = 2;
constexpr int32_t kDequeueInterval = 1000; // 1000ms
constexpr int32_t kEnqueueInterval = 100; // 100ms
constexpr size_t kDefaultQueueBufNum = 2U;

constexpr size_t kCopyThreadNum = 8U;
}  // namespace

HeterogeneousExchangeService::HeterogeneousExchangeService()
    : copy_thread_pool_("ge_hete_cpy", kCopyThreadNum, false) {}

HeterogeneousExchangeService &HeterogeneousExchangeService::GetInstance() {
  static HeterogeneousExchangeService instance;
  return instance;
}

HeterogeneousExchangeService::~HeterogeneousExchangeService() {
  (void)Finalize();
}

Status HeterogeneousExchangeService::Initialize(int32_t device_id) {
  HeterogeneousProfiler::Instance().InitHeterogeneousPoriler();
  rtMemQueueSetInputPara para = {};
  (void) rtMemQueueSet(device_id, RT_MQ_QUEUE_ENABLE_LOCAL_QUEUE, &para);
  auto ret = rtMemQueueInit(device_id);
  // It is normal return ACL_ERROR_RT_FEATURE_NOT_SUPPORT, in offline compile scenario,
  // because driver so is stub, rtMemQueueInit get driver version is always 0.
  if (ret != ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
    GE_CHK_STATUS_RET(EnsureInitialized(device_id), "Init queue failed, device_id = %d", device_id);
    GE_CHK_STATUS_RET(InitializeEvents(device_id), "Init event failed, device id = %d", device_id);
  }
  return SUCCESS;
}

Status HeterogeneousExchangeService::Finalize() {
  HeterogeneousProfiler::Instance().PrintHeterogeneousProfilerData();
  waiting_.store(false);
  for (auto &th : events_threads_) {
    if (th.joinable()) {
      th.join();
    }
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    initialized_devices_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(dequeue_mu_);
    subscribed_dequeues_.clear();
  }
  {
    std::lock_guard<std::mutex> lk(enqueue_mu_);
    subscribed_enqueues_.clear();
  }
  
  std::lock_guard<std::mutex> lk(client_q_mu_);
  client_queue_ids_.clear();
  std::lock_guard<std::mutex> ctx_lk(ctx_mu_);
  if (rt_context_ != nullptr) {
    (void) aclrtDestroyContext(rt_context_);
    rt_context_ = nullptr;
  }
  return SUCCESS;
}

Status HeterogeneousExchangeService::EnsureInitialized(int32_t device_id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (initialized_devices_.find(device_id) == initialized_devices_.end()) {
    rtMemQueueSetInputPara para = {};
    (void) rtMemQueueSet(device_id, RT_MQ_QUEUE_ENABLE_LOCAL_QUEUE, &para);
    GELOGI("[InitQueue] start, device id = %d", device_id);
    auto ret = rtMemQueueInit(device_id);
    if (ret != RT_ERROR_NONE && ret != ACL_ERROR_RT_REPEATED_INIT) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtMemQueueInit fail, ret: 0x%X", static_cast<uint32_t>(ret));
      GELOGE(RT_FAILED, "[InitQueue] failed, rt_err = %d, device id = %d", ret, device_id);
      return RT_ERROR_TO_GE_STATUS(ret);
    }
    GELOGI("[InitQueue] ended successfully, device id = %d", device_id);
    initialized_devices_.emplace(device_id);
  }
  return SUCCESS;
}

void HeterogeneousExchangeService::ProcessEmptyToNotEmptyEvent(const uint32_t queue_id) {
  std::lock_guard<std::mutex> lk(dequeue_mu_);
  auto it = subscribed_dequeues_.find(queue_id);
  GELOGI("[ReceiveEvent] receive queue id[%u] event, find result[%d]",
         queue_id, static_cast<int32_t>(it != subscribed_dequeues_.end()));
  if (it != subscribed_dequeues_.end()) {
    GELOGI("[ReceiveEvent] received queue id[%u] enqueue, notify dequeue", queue_id);
    it->second = true;
    dequeue_cv_.notify_all();
  }
}

void HeterogeneousExchangeService::ProcessF2NFEvent(const uint32_t queue_id) {
  std::lock_guard<std::mutex> lk(enqueue_mu_);
  auto it = subscribed_enqueues_.find(queue_id);
  GELOGI("[ReceiveEvent] receive queue id[%u] event, find result[%d]",
         queue_id, static_cast<int32_t>(it != subscribed_enqueues_.end()));
  if (it != subscribed_enqueues_.end()) {
    GELOGI("[ReceiveEvent] received queue id[%u] not full, notify enqueue", queue_id);
    it->second = true;
    enqueue_cv_.notify_all();
  }
}

void HeterogeneousExchangeService::WaitEvents(const int32_t device_id) {
  uint64_t mask = (1ULL << static_cast<uint32_t>(RT_EVENT_QUEUE_EMPTY_TO_NOT_EMPTY)) |
                  (1ULL << static_cast<uint32_t>(RT_EVENT_QUEUE_FULL_TO_NOT_FULL));
  auto result = RtsApiUtils::EschedSubscribeEvent(device_id, kEventGroupId, 0, mask);
  if (result != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Failed to invoke EschedSubscribeEvent, ret = 0x%X", result);
    return;
  }
  GELOGI("Event thread start successfully, device id = %d", device_id);
  {
    std::unique_lock<std::mutex> wait_lk(wait_event_mu_);
    event_thread_starting_ = false;
    wait_events_cv_.notify_all();
  }
  while (waiting_) {
    rtEschedEventSummary_t in_event = {};
    auto ret = rtEschedWaitEvent(device_id, kEventGroupId, 0, kWaitEventTimeout, &in_event);
    if (ret == RT_ERROR_NONE) {
      if (in_event.eventId == RT_EVENT_QUEUE_EMPTY_TO_NOT_EMPTY) {
        ProcessEmptyToNotEmptyEvent(in_event.subeventId);
      } else if (in_event.eventId == RT_EVENT_QUEUE_FULL_TO_NOT_FULL) {
        ProcessF2NFEvent(in_event.subeventId);
      }
    } else if (ret != ACL_ERROR_RT_REPORT_TIMEOUT) {
      GELOGW("Invoke rtEschedWaitEvent ret is 0x%X", ret);
    } else {
      GELOGD("Invoke rtEschedWaitEvent time out");
    }
  }
  GELOGI("Event thread exist successfully, device id = %d", device_id);
}

Status HeterogeneousExchangeService::EnsureEnqueueSubscribed(const int32_t device_id, const uint32_t queue_id) {
  const auto &it = subscribed_enqueues_.find(queue_id);
  if (it == subscribed_enqueues_.cend()) {
    GELOGI("[EnqueueSubscribe] start, queue id = %u", queue_id);
    auto ret = rtQueueSubF2NFEvent(device_id, queue_id, kEventGroupId);
    if (ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtQueueSubF2NFEvent fail, ret: 0x%X", static_cast<uint32_t>(ret));
      GELOGE(RT_FAILED, "[EnqueueSubscribe] failed, rt_err = %d, queue id = %u", ret, queue_id);
      return RT_ERROR_TO_GE_STATUS(ret);
    }
    GELOGI("[EnqueueSubscribe] ended successfully, queue id = %u", queue_id);
  }
  subscribed_enqueues_[queue_id] = false;
  return SUCCESS;
}

Status HeterogeneousExchangeService::EnsureDequeueSubscribed(const int32_t device_id, const uint32_t queue_id) {
  const auto &it = subscribed_dequeues_.find(queue_id);
  if (it == subscribed_dequeues_.cend()) {
    GELOGI("[DequeueSubscribe] start, queue id = %u", queue_id);
    auto ret = rtQueueSubscribe(device_id, queue_id, kEventGroupId, kRtQueueTypeSingle);
    if (ret != RT_ERROR_NONE) {
      REPORT_INNER_ERR_MSG("E19999", "Call rtQueueSubscribe fail, ret: 0x%X", static_cast<uint32_t>(ret));
      GELOGE(RT_FAILED, "[DequeueSubscribe] failed, rt_err = %d, queue id = %d", ret, queue_id);
      return RT_ERROR_TO_GE_STATUS(ret);
    }
    GELOGI("[DequeueSubscribe] ended successfully, queue id = %u", queue_id);
  }
  subscribed_dequeues_[queue_id] = false;
  return SUCCESS;
}

Status HeterogeneousExchangeService::InitializeEvents(const int32_t device_id) {
  std::lock_guard<std::mutex> lk(events_mu_);
  if (events_devices_.find(device_id) == events_devices_.end()) {
    GELOGI("[InitEvents] start, device id = %d", device_id);
    GE_CHK_STATUS_RET(RtsApiUtils::EschedAttachDevice(device_id));
    GE_CHK_STATUS_RET(RtsApiUtils::EschedCreateGroup(device_id, kEventGroupId, RT_GRP_TYPE_BIND_CP_CPU));
    events_threads_.emplace_back(&HeterogeneousExchangeService::WaitEvents, this, device_id);
    std::unique_lock<std::mutex> wait_lk(wait_event_mu_);
    if (event_thread_starting_) {
      wait_events_cv_.wait(wait_lk, [this] { return !event_thread_starting_; });
    }
    event_thread_starting_ = true;
    GELOGI("[InitEvents] ended successfully, device id = %d", device_id);
    events_devices_.emplace(device_id);
  }
  return SUCCESS;
}

void HeterogeneousExchangeService::DestroyTransInfo(const int32_t device_id, const uint32_t queue_id) {
  std::lock_guard<std::mutex> lk(trans_mu_);
  TransInfoContext context{device_id, queue_id};
  const auto &iter = trans_ids_.find(context);
  if (iter != trans_ids_.cend()) {
    trans_ids_.erase(iter);
    GELOGI("Destroy trans info successfully, device id = %d, queue id = %u", device_id, queue_id);
  }
}

Status HeterogeneousExchangeService::SetTransId(const int32_t device_id, const uint32_t queue_id, rtMbufPtr_t mbuf) {
  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  auto ret = rtMbufGetPrivInfo(mbuf, &head_buf, &head_size);
  if (ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMbufGetPrivInfo fail, ret: 0x%X", static_cast<uint32_t>(ret));
    GELOGE(RT_FAILED, "Set trans id failed, rt_err = %d, device id = %d, queue id = %u", ret, device_id, queue_id);
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  if ((head_buf != nullptr) && (head_size >= sizeof(MsgInfo))) {
    MsgInfo *msg_info = reinterpret_cast<MsgInfo *>(static_cast<char_t *>(head_buf) + head_size - sizeof(MsgInfo));

    uint64_t custom_trans_id = 0;
    if ((msg_info->data_flag & kCustomTransIdFlagBit) != 0) {
      custom_trans_id = msg_info->trans_id;
    }
    GE_CHK_STATUS_RET(GenTransId(device_id, queue_id, msg_info->trans_id, custom_trans_id),
                      "gen trans id failed, device_id=%d, queue_id=%u, custom_trans_id=%lu.", device_id, queue_id,
                      custom_trans_id);
    GELOGI("Set trans id[%lu] successfully, device id = %d, queue id = %u", msg_info->trans_id, device_id, queue_id);
  }
  return SUCCESS;
}

void HeterogeneousExchangeService::AddClientQueue(const uint32_t queue_id) {
  std::lock_guard<std::mutex> lk(client_q_mu_);
  client_queue_ids_.emplace(queue_id);
}

bool HeterogeneousExchangeService::IsClientQueue(const uint32_t queue_id) {
  std::lock_guard<std::mutex> lk(client_q_mu_);
  const auto &it = client_queue_ids_.find(queue_id);
  return it != client_queue_ids_.cend();
}

Status HeterogeneousExchangeService::CreateQueue(const int32_t device_id,
                                                 const string &name,
                                                 const MemQueueAttr &mem_queue_attr,
                                                 uint32_t &queue_id) {
  if (name.size() > static_cast<size_t>(RT_MQ_MAX_NAME_LEN - 1)) {
    GELOGE(PARAM_INVALID,
           "[CreateQueue] [CheckParam] Length of queue name out of range, name = %s, length = %zu",
           name.c_str(),
           name.size());
    return PARAM_INVALID;
  }
  GELOGD("[CreateQueue] start, device id = %d, queue name = %s, depth = %u, work_mode = %u",
         device_id,
         name.c_str(),
         mem_queue_attr.depth,
         mem_queue_attr.work_mode);
  GE_CHK_STATUS_RET(EnsureInitialized(device_id), "[CreateQueue] [Init] failed, queue name = %s", name.c_str());
  rtMemQueueAttr_t attr;
  attr.depth = mem_queue_attr.depth;
  attr.workMode = mem_queue_attr.work_mode;
  attr.flowCtrlFlag = false;
  attr.flowCtrlDropTime = static_cast<uint32_t>(kQueueOpTimeout);
  attr.overWriteFlag = mem_queue_attr.overwrite;
  attr.deployType = RT_MQ_LOCAL_QUEUE_DEPLOY;
  if (mem_queue_attr.is_client) {
    attr.deployType = RT_MQ_CLIENT_QUEUE_DEPLOY;
  }
  // actually this won't fail, length was checked
  if (strcpy_s(attr.name, sizeof(attr.name), name.c_str()) != EOK) {
    GELOGE(FAILED, "[CreateQueue] [CopyName] Failed");
    return FAILED;
  }

  auto ret = rtMemQueueCreate(device_id, &attr, &queue_id);
  if (ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemQueueCreate fail, ret: 0x%X", static_cast<uint32_t>(ret));
    GELOGE(RT_FAILED, "[CreateQueue] failed, rt_err = %d, device id = %d, queue name = %s, depth = %u",
           ret,
           device_id,
           name.c_str(),
           mem_queue_attr.depth);
    return RT_ERROR_TO_GE_STATUS(ret);
  }

  if (mem_queue_attr.is_client) {
    AddClientQueue(queue_id);
  }
  GEEVENT("[CreateQueue] ended successfully, device id = %d, queue name = %s, depth = %u, queue_id = %u, "
          "deploy type = %u, is_client = %d.",
          device_id,
          name.c_str(),
          mem_queue_attr.depth,
          queue_id,
          attr.deployType,
          static_cast<int32_t>(mem_queue_attr.is_client));
  return SUCCESS;
}

Status HeterogeneousExchangeService::DestroyQueue(int32_t device_id, uint32_t queue_id) {
  GELOGD("[DestroyQueue] start, device id = %d, queue_id = %u", device_id, queue_id);
  // fix runtime context invalid
  (void)aclrtSetDevice(device_id);
  auto ret = rtMemQueueDestroy(device_id, queue_id);
  if (ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemQueueDestroy fail, ret: 0x%X", static_cast<uint32_t>(ret));
    GELOGE(RT_FAILED, "[DestroyQueue] failed, rt_err = %d, device id = %d, queue id = %u",
           ret,
           device_id,
           queue_id);
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  DestroyTransInfo(device_id, queue_id);
  GELOGD("[DestroyQueue] ended successfully, device id = %d, queue_id = %u", device_id, queue_id);
  return SUCCESS;
}

Status HeterogeneousExchangeService::Enqueue(int32_t device_id,
                                             uint32_t queue_id,
                                             const void *data,
                                             size_t size,
                                             const ControlInfo &control_info) {
  if (control_info.is_shared_input) {
    rtMbufPtr_t mbuf = nullptr;
    GE_CHK_RT_RET(rtMbufBuild(const_cast<void *>(data), size, &mbuf));
    GE_CHK_STATUS_RET_NOLOG(InitHeadInfo(control_info, mbuf));
    HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                       ProfilerEvent::kMbufEnqueue,
                                                                       device_id, queue_id);
    auto ret = EnqueueMbuf(device_id, queue_id, mbuf, control_info.timeout);
    HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                       ProfilerEvent::kMbufEnqueue,
                                                                       device_id, queue_id);
    GE_CHK_STATUS_RET(ret, "Enqueue mbuf failed");
    return SUCCESS;
  }

  FillFunc fill_func = [data, size](void *buffer, size_t buffer_size) {
    if (size == 0 || memcpy_s(buffer, buffer_size, data, size) == EOK) {
      return SUCCESS;
    }
    return FAILED;
  };
  return Enqueue(device_id, queue_id, size, fill_func, control_info);
}

Status HeterogeneousExchangeService::InitHeadInfo(const ControlInfo &control_info, rtMbufPtr_t mbuf) {
  // clear end of sequence flag
  void *priv_info = nullptr;
  uint64_t priv_size = 0UL;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(mbuf, &priv_info, &priv_size));
  if ((priv_info != nullptr) && (priv_size > kMbufHeadEndOfSequencePos)) {
    *(static_cast<uint8_t *>(priv_info) + kMbufHeadEndOfSequencePos) = 0U;
  }
  if ((priv_info != nullptr) && (priv_size >= sizeof(MsgInfo))) {
    MsgInfo *msg_info = reinterpret_cast<MsgInfo *>(static_cast<char_t *>(priv_info) + priv_size - sizeof(MsgInfo));
    if (control_info.msg_info == nullptr) {
      *msg_info = {};
    } else {
      *msg_info = *control_info.msg_info;
    }
  }
  if (priv_size < static_cast<uint64_t>(kMaxUserDataSize)) {
    GELOGE(FAILED, "Failed to set user data, the mbuf head size[%lu] < user data size[%zu].", priv_size,
           kMaxUserDataSize);
    return FAILED;
  }
  const auto cpy_ret = memcpy_s(priv_info, priv_size, control_info.user_data, kMaxUserDataSize);
  GE_ASSERT_EOK(cpy_ret, "Failed to set user data, memcpy_s error, dst size[%lu], src size[%zu], ret[%d].", priv_size,
                kMaxUserDataSize, cpy_ret);
  GELOGD("[Init][Mbuf] head info success");
  return SUCCESS;
}

Status HeterogeneousExchangeService::Enqueue(int32_t device_id,
                                             uint32_t queue_id,
                                             size_t size,
                                             const ExchangeService::FillFunc &fill_func,
                                             const ControlInfo &control_info) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufAlloc,
                                                                     device_id, queue_id);
  rtMbufPtr_t m_buf = nullptr;
  void *buffer = nullptr;
  GE_CHK_RT_RET(rtMbufAlloc(&m_buf, size));
  GE_DISMISSABLE_GUARD(m_buf, ([m_buf]() { GE_CHK_RT(rtMbufFree(m_buf)); }));
  GE_CHK_RT_RET(rtMbufSetDataLen(m_buf, size));
  GE_CHK_RT_RET(rtMbufGetBuffAddr(m_buf, &buffer));
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufAlloc,
                                                                     device_id, queue_id);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMemCopyToMbuf,
                                                                     device_id, queue_id);
  auto ret = fill_func(buffer, size);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMemCopyToMbuf,
                                                                     device_id, queue_id);
  GE_CHK_STATUS_RET(ret, "[CopyTo][Mbuf] failed, size = %zu", size);
  GE_CHK_STATUS_RET_NOLOG(InitHeadInfo(control_info, m_buf));
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufEnqueue,
                                                                     device_id, queue_id);
  const bool print_error_flag = control_info.print_error_flag;
  ret = EnqueueMbuf(device_id, queue_id, m_buf, control_info.timeout);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufEnqueue,
                                                                     device_id, queue_id);
  if ((ret != SUCCESS) && (!print_error_flag)) {
    return ret;
  }
  GE_CHK_STATUS_RET(ret, "Enqueue mbuf failed");
  // mbuf will be freed by consumer
  GE_DISMISS_GUARD(m_buf);
  GELOGD("[Enqueue][Mbuf] succeeded, device_id = %d, queue_id = %u, size = %zu", device_id, queue_id, size);
  return SUCCESS;
}

Status HeterogeneousExchangeService::Enqueue(const int32_t device_id, const uint32_t queue_id,
                                             const std::vector<BuffInfo> &buffs, const ControlInfo &control_info) {
  if (IsClientQueue(queue_id)) {
    return ProcessEnqueueBuff(device_id, queue_id, buffs, control_info);
  }
  size_t mbuf_size = 0U;
  for (const auto &item : buffs) {
    GE_CHK_BOOL_RET_STATUS(!ge::AddOverflow(mbuf_size, item.len, mbuf_size), FAILED, "Total buffs size too large.");
  }
  GE_CHK_BOOL_RET_STATUS(mbuf_size > 0, FAILED, "Input buff item size is 0.");
  GELOGD("Queue[%u] is not client queue, mbuf size is [%zu].", queue_id, mbuf_size);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufAlloc,
                                                                     device_id, queue_id);
  rtMbufPtr_t m_buf = nullptr;
  GE_CHK_RT_RET(rtMbufAlloc(&m_buf, mbuf_size));
  GE_DISMISSABLE_GUARD(m_buf, ([m_buf]() { GE_CHK_RT(rtMbufFree(m_buf)); }));
  GE_CHK_RT_RET(rtMbufSetDataLen(m_buf, mbuf_size));
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufAlloc,
                                                                     device_id, queue_id);
  GE_CHK_STATUS_RET(ProcessEnqueueMbuf(device_id, queue_id, buffs, m_buf, control_info),
                    " Process enqueue mbuf failed.");
  GE_DISMISS_GUARD(m_buf);
  GELOGI("Enqueue succeeded, device_id = %d, queue_id = %u, buffs size = %zu", device_id, queue_id, buffs.size());
  return SUCCESS;
}

Status HeterogeneousExchangeService::ProcessEnqueueBuff(const int32_t device_id, const uint32_t queue_id,
                                                        const std::vector<BuffInfo> &buffs,
                                                        const ControlInfo &control_info) {
  constexpr size_t kHeaderBuffSize = 256U;
  uint8_t header_buff[kHeaderBuffSize]{};
  auto *msg_info = reinterpret_cast<MsgInfo *>(header_buff + sizeof(header_buff) - sizeof(MsgInfo));
  if (control_info.msg_info != nullptr) {
    *msg_info = *control_info.msg_info;
  }
  const auto cpy_ret = memcpy_s(header_buff, sizeof(header_buff), control_info.user_data, kMaxUserDataSize);
  GE_ASSERT_EOK(cpy_ret, "Failed to get user data, memcpy_s error, dst size[%zu], src size[%zu], ret[%d].",
                kMaxUserDataSize, kMaxUserDataSize, cpy_ret);

  GE_CHK_STATUS_RET(GenTransId(device_id, queue_id, msg_info->trans_id, msg_info->trans_id),
                    "gen trans id failed, device_id=%d, queue_id=%u.", device_id, queue_id);
  GELOGI("Set trans id[%lu] successfully, device id = %d, queue id = %u", msg_info->trans_id, device_id, queue_id);
  SmallVector<rtMemQueueBuffInfo, kDefaultQueueBufNum> queue_buf_info(buffs.size());
  uint32_t num_buffs = 0U;
  for (size_t i = 0; i < buffs.size(); i++) {
    const auto &buff = buffs[i];
    if (buff.len == 0) {
      GELOGW("Length of buff[%zu] is 0, skip.", i);
      continue;
    }
    auto &buf_info = queue_buf_info[num_buffs];
    buf_info.len = buff.len;
    buf_info.addr = buff.addr;
    GE_CHECK_NOTNULL(buf_info.addr);
    ++num_buffs;
  }
  rtMemQueueBuff_t queue_buf = {.contextAddr = header_buff,
                                .contextLen = kHeaderBuffSize,
                                .buffInfo = queue_buf_info.data(),
                                .buffCount = num_buffs};
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufEnqueue,
                                                                     device_id, queue_id);
  const rtError_t ret = rtMemQueueEnQueueBuff(device_id, queue_id, &queue_buf, control_info.timeout);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufEnqueue,
                                                                     device_id, queue_id);
  if (ret == ACL_ERROR_RT_QUEUE_FULL) {
    GELOGE(RT_FAILED, "[Enqueue][MBuf] timeout, device_id = %d, queue_id = %u, timeout = %d ms, rt_error_code = %d",
           device_id, queue_id, control_info.timeout, ret);
  }
  GE_CHK_RT_RET(ret);
  GELOGI("Enqueue succeeded, device_id = %d, queue_id = %u, buffs size = %zu", device_id, queue_id, buffs.size());
  return SUCCESS;
}

Status HeterogeneousExchangeService::MultiThreadCopy(uint8_t *dst, size_t dst_size, const uint8_t *src,
                                                     size_t src_size) {
  constexpr size_t kMinBatchSize = 20 * 1024 * 1024UL;
  if (src_size <= kMinBatchSize) {
    GE_CHK_BOOL_RET_STATUS(memcpy_s(dst, dst_size, src, src_size) == EOK, FAILED,
                           "Failed to copy data, dst size=%zu, src size=%zu", dst_size, src_size);
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(dst_size >= src_size, PARAM_INVALID,
                         "Multi thread copy failed as dst_size=%zu is small than src_size=%zu", dst_size, src_size);

  size_t block_num = (src_size + kMinBatchSize - 1) / kMinBatchSize;
  block_num = std::min(block_num, kCopyThreadNum + 1);
  size_t batch_size = (src_size + block_num - 1) / block_num;
  // batch 2M align
  constexpr size_t kCopyBlockAlignSize = 2 * 1024 * 1024UL;
  batch_size = (batch_size + kCopyBlockAlignSize - 1) / kCopyBlockAlignSize * kCopyBlockAlignSize;
  block_num = (src_size + batch_size - 1) / batch_size;
  std::vector<std::future<Status>> vector_future;
  vector_future.reserve(block_num - 1);
  size_t offset = 0;
  // the last block copy by current thread
  for (size_t idx = 0; idx < (block_num - 1); ++idx) {
    size_t copy_size = std::min(batch_size, src_size - offset);
    auto fut = copy_thread_pool_.commit([dst, dst_size, src, src_size, offset, copy_size]() {
      GE_CHK_BOOL_RET_STATUS(memcpy_s(dst + offset, dst_size - offset, src + offset, copy_size) == EOK, FAILED,
                             "Failed to copy data, offset=%zu, copy size=%zu, dst size=%zu, src size=%zu", offset,
                             copy_size, dst_size, src_size);
      GELOGD("copy by thread end, offset=%zu, copy size=%zu, dst size=%zu, src size=%zu", offset, copy_size, dst_size,
             src_size);
      return SUCCESS;
    });
    GE_CHK_BOOL_RET_STATUS(
        fut.valid(), FAILED,
        "Failed to commit copy task, idx=%zu, offset=%zu, copy size=%zu, dst size=%zu, src size=%zu.", idx, offset,
        copy_size, dst_size, src_size);
    vector_future.emplace_back(std::move(fut));
    offset += copy_size;
  }
  if (offset < src_size) {
    GE_CHK_BOOL_RET_STATUS(memcpy_s(dst + offset, dst_size - offset, src + offset, src_size - offset) == EOK, FAILED,
                           "Failed to copy left data, offset=%zu, src total size=%zu, dst total size=%zu", offset,
                           src_size, dst_size);
    GELOGD("copy left data end, offset=%zu, copy size=%zu, dst size=%zu, src size=%zu", offset, src_size - offset,
           dst_size, src_size);
  }
  Status return_ret = SUCCESS;
  for (size_t i = 0; i < vector_future.size(); ++i) {
    auto ret = vector_future[i].get();
    if (ret != SUCCESS) {
      return_ret = ret;
    }
  }
  return return_ret;
}

Status HeterogeneousExchangeService::ProcessEnqueueMbuf(const int32_t device_id, const uint32_t queue_id,
                                                        const std::vector<BuffInfo> &buffs, rtMbufPtr_t mbuf,
                                                        const ControlInfo &control_info) {
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMemCopyToMbuf,
                                                                     device_id, queue_id);
  void *buffer = nullptr;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, &buffer));
  uint64_t data_len = 0U;
  GE_CHK_RT_RET(rtMbufGetDataLen(mbuf, &data_len));
  size_t remaining_size = static_cast<size_t>(data_len);
  uint8_t *remaining_buffer = PtrToPtr<void, uint8_t>(buffer);
  for (size_t i = 0; i < buffs.size(); i++) {
    if (buffs[i].len == 0) {
      GELOGW("Length of buff[%zu] is 0, skip.", i);
      continue;
    }
    GE_CHECK_NOTNULL(buffs[i].addr);
    GE_CHK_STATUS_RET(
        MultiThreadCopy(remaining_buffer, remaining_size, static_cast<uint8_t *>(buffs[i].addr), buffs[i].len),
        "Failed to copy data to remaining_buffer, dst size = %zu, copy size = %zu", remaining_size, buffs[i].len);
    remaining_buffer = PtrAdd<uint8_t>(remaining_buffer, remaining_size, buffs[i].len);
    remaining_size -= buffs[i].len;
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMemCopyToMbuf,
                                                                     device_id, queue_id);
  GE_CHK_STATUS_RET_NOLOG(InitHeadInfo(control_info, mbuf));
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufEnqueue,
                                                                     device_id, queue_id);
  GE_CHK_STATUS_RET(EnqueueMbuf(device_id, queue_id, mbuf, control_info.timeout), "Enqueue mbuf failed");
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufEnqueue,
                                                                     device_id, queue_id);
  return SUCCESS;
}

Status HeterogeneousExchangeService::Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                                             const ControlInfo &control_info) {
  // clear end of sequence flag
  GE_CHK_RT_RET(rtMbufSetDataLen(m_buf, size));
  GE_CHK_STATUS_RET_NOLOG(InitHeadInfo(control_info, m_buf));
  auto ret = EnqueueMbuf(device_id, queue_id, m_buf, control_info.timeout);
  if (ret != SUCCESS) {
    return ret;
  }
  GELOGD("[Enqueue][Mbuf] succeeded, device_id = %d, queue_id = %u", device_id, queue_id);
  return SUCCESS;
}

Status HeterogeneousExchangeService::CheckResult(void *head_buf, size_t head_size, ControlInfo &control_info) {
  GE_CHECK_NOTNULL(head_buf);
  if (head_size > kMbufHeadEndOfSequencePos) {
    uint64_t value = PtrToValue(head_buf);
    uint8_t end_of_sequence = *(reinterpret_cast<uint8_t *>(value + kMbufHeadEndOfSequencePos));
    if (end_of_sequence == kEndOfSequenceFlag) {
      control_info.end_of_sequence_flag = true;
      GELOGI("[Dequeue] End of sequence is coming.");
      return SUCCESS;
    }
  }

  if (head_size < kMaxUserDataSize) {
    GELOGE(FAILED, "Failed to get user data, the mbuf head size[%zu] < user data size[%zu].", head_size,
           kMaxUserDataSize);
    return FAILED;
  }
  const auto cpy_ret = memcpy_s(control_info.user_data, kMaxUserDataSize, head_buf, kMaxUserDataSize);
  GE_ASSERT_EOK(cpy_ret, "Failed to get user data, memcpy_s error, dst size[%zu], src size[%zu], ret[%d].",
                kMaxUserDataSize, kMaxUserDataSize, cpy_ret);

  if (head_size >= sizeof(MsgInfo)) {
    MsgInfo *msg_info = reinterpret_cast<MsgInfo *>(static_cast<char_t *>(head_buf) + head_size - sizeof(MsgInfo));
    if (control_info.msg_info != nullptr) {
      *control_info.msg_info = *msg_info;
    }
    GE_CHK_BOOL_RET_STATUS(msg_info->ret_code == 0, static_cast<Status>(msg_info->ret_code),
                           "Failed to execute model, error code = %d, trans id = %lu", msg_info->ret_code,
                           msg_info->trans_id);
  }
  return SUCCESS;
}

Status HeterogeneousExchangeService::CheckResult(rtMbufPtr_t m_buf, ControlInfo &control_info) {
  size_t head_size = 0;
  void *head_buf = nullptr;
  GE_CHK_STATUS_RET(CopyMbufHeadTo(m_buf, head_buf, head_size), "Failed to get mbuf head.");
  return CheckResult(head_buf, head_size, control_info);
}

Status HeterogeneousExchangeService::Dequeue(int32_t device_id, uint32_t queue_id, void *data, size_t size,
                                             ControlInfo &control_info) {
  rtMbufPtr_t m_buf = nullptr;
  GE_CHK_STATUS_RET_NOLOG(DequeueMbuf(device_id, queue_id, &m_buf, control_info.timeout));
  GE_MAKE_GUARD(m_buf, [m_buf]() { GE_CHK_RT(rtMbufFree(m_buf)); });
  GE_CHK_STATUS_RET_NOLOG(CheckResult(m_buf, control_info));
  if (control_info.end_of_sequence_flag) {
    return SUCCESS;
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufCopyToMem,
                                                                     device_id, queue_id);
  // size 0 means no need to copy
  if (size != 0U) {
    GE_CHK_STATUS_RET_NOLOG(CopyMbufTo(m_buf, data, size, control_info));
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufCopyToMem,
                                                                     device_id, queue_id);

  GELOGD("[Dequeue] succeeded, device_id = %d, queue_id = %u, size = %zu",
         device_id, queue_id, size);
  return SUCCESS;
}

Status HeterogeneousExchangeService::DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id,
                                                       std::shared_ptr<AlignedPtr> &aligned_ptr,
                                                       const size_t size, ControlInfo &control_info) {
  (void)size;
  rtMbufPtr_t m_buf = nullptr;
  GE_CHK_STATUS_RET_NOLOG(DequeueMbuf(device_id, queue_id, &m_buf, control_info.timeout));
  GE_DISMISSABLE_GUARD(m_buf, ([m_buf]() { GE_CHK_RT(rtMbufFree(m_buf)); }));
  GE_CHK_STATUS_RET_NOLOG(CheckResult(m_buf, control_info));
  if (control_info.end_of_sequence_flag) {
    return SUCCESS;
  }
  GE_CHK_STATUS_RET_NOLOG(MoveMbufTo(m_buf, control_info, aligned_ptr));
  // mbuf will be freed by consumer
  GE_DISMISS_GUARD(m_buf);
  return SUCCESS;
}

Status HeterogeneousExchangeService::UpdateTensorDesc(const RuntimeTensorDesc &runtime_tensor_desc,
                                                      GeTensorDesc &tensor_desc) {
  auto num_dims = runtime_tensor_desc.shape[0];
  auto num_ori_dims = runtime_tensor_desc.original_shape[0];
  GE_CHK_BOOL_RET_STATUS(num_dims <= kMaxDimSize,
                         UNSUPPORTED,
                         "shape dim number out of range, num_dims = %ld, max = %ld",
                         num_dims, kMaxDimSize);
  GE_CHK_BOOL_RET_STATUS(num_ori_dims <= kMaxDimSize,
                         UNSUPPORTED,
                         "original shape dim number out of range, num_dims = %ld, max = %ld",
                         num_ori_dims, kMaxDimSize);
  GeShape shape(std::vector<int64_t>(&runtime_tensor_desc.shape[1], &runtime_tensor_desc.shape[1] + num_dims));
  GeShape ori_shape(std::vector<int64_t>(&runtime_tensor_desc.shape[1], &runtime_tensor_desc.shape[1] + num_dims));
  tensor_desc.MutableShape() = std::move(shape);
  tensor_desc.MutableShape() = std::move(ori_shape);
  tensor_desc.SetDataType(static_cast<DataType>(runtime_tensor_desc.dtype));
  return SUCCESS;
}

Status HeterogeneousExchangeService::GetOrCreateRtCtx(aclrtContext &ctx, int32_t device_id) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  if (rt_context_ == nullptr) {
    DF_CHK_ACL(aclrtCreateContext(&rt_context_, device_id));
  }
  ctx = rt_context_;
  return SUCCESS;
}

Status HeterogeneousExchangeService::AllocAlignedBuffer(const size_t buffer_size, uint8_t *&aligned_ptr, int32_t device_id) {
  aclrtContext ctx = nullptr;
  (void)aclrtGetCurrentContext(&ctx);
  if (ctx == nullptr) {
    GE_CHK_STATUS_RET(GetOrCreateRtCtx(ctx, device_id), "Failed to get rt context.");
    DF_CHK_ACL_RET(aclrtSetCurrentContext(ctx));
  }
  DF_CHK_ACL_RET(aclrtMallocHost(reinterpret_cast<void **>(&aligned_ptr), buffer_size));
  return SUCCESS;
}

Status HeterogeneousExchangeService::ProcessDequeueBuffTensor(int32_t device_id,
                                                              uint32_t queue_id,
                                                              GeTensor &tensor,
                                                              ControlInfo &control_info) {
  size_t data_buffer_size = 0U;
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufDequeue, queue_id);
  auto ret = rtMemQueuePeek(device_id, queue_id, &data_buffer_size, control_info.timeout);
  if (ret != RT_ERROR_NONE) {
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  GE_MAKE_GUARD(profile, [queue_id]() {
    HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                       ProfilerEvent::kMbufDequeue,
                                                                       queue_id);
  });

  GE_CHK_BOOL_RET_STATUS(data_buffer_size >= sizeof(RuntimeTensorDesc), FAILED,
                          "Failed to check queue buffer size[%zu].", data_buffer_size);
  rtMemQueueBuffInfo queue_buf_info = {};
  uint8_t *aligned_ptr = nullptr;
  GE_CHK_STATUS_RET(AllocAlignedBuffer(data_buffer_size, aligned_ptr, device_id), "Failed to alloc buffer");
  auto deleter = [aligned_ptr](const uint8_t *ptr) {
    (void) ptr;
    DF_CHK_ACL(aclrtFreeHost(aligned_ptr));
  };
  RuntimeTensorDesc *const tensor_desc = PtrToPtr<uint8_t, RuntimeTensorDesc>(aligned_ptr);
  *tensor_desc = {};
  GE_DISMISSABLE_GUARD(aligned_ptr, ([deleter]() { deleter(nullptr); }));
  queue_buf_info.len = data_buffer_size;
  queue_buf_info.addr = aligned_ptr;
  constexpr size_t kHeaderBuffSize = 256U;
  uint8_t header_buff[kHeaderBuffSize]{};
  rtMemQueueBuff_t queue_buf = {.contextAddr = header_buff,
                                .contextLen = kHeaderBuffSize,
                                .buffInfo = &queue_buf_info,
                                .buffCount = 1U};
  ret = rtMemQueueDeQueueBuff(device_id, queue_id, &queue_buf, control_info.timeout);
  if (ret == RT_ERROR_NONE) {
    GE_CHK_STATUS_RET(CheckResult(header_buff, kHeaderBuffSize, control_info), "Failed to check result");
    if (control_info.end_of_sequence_flag ||
        ((control_info.msg_info != nullptr) && ((control_info.msg_info->data_flag & kNullDataFlagBit) != 0U))) {
      // end of sequence or null data
      return SUCCESS;
    }
    auto &output_tensor_desc = tensor.MutableTensorDesc();
    GE_CHK_STATUS_RET(UpdateTensorDesc(*tensor_desc, output_tensor_desc), "Failed to update output tensor desc");
    int64_t tensor_raw_size = -1;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(output_tensor_desc.GetShape(),
                                                           output_tensor_desc.GetFormat(),
                                                           output_tensor_desc.GetDataType(),
                                                           tensor_raw_size),
                            "Failed to calc tensor mem size");
    size_t queue_data_size = data_buffer_size - sizeof(RuntimeTensorDesc);
    GE_CHK_BOOL_RET_STATUS(tensor_raw_size >= 0 && queue_data_size >= static_cast<size_t>(tensor_raw_size),
                           FAILED,
                           "Failed to check queue buffer size[%zu], must >= %ld.",
                           queue_data_size, tensor_raw_size);
    if (queue_data_size > 0U) {
      auto output_aligned_ptr = AlignedPtr::BuildFromData(aligned_ptr + sizeof(RuntimeTensorDesc), deleter);
      GE_CHECK_NOTNULL(output_aligned_ptr);
      tensor.SetData(output_aligned_ptr, static_cast<uint64_t>(tensor_raw_size));
      // buffer will be freed by tensor
      GE_DISMISS_GUARD(aligned_ptr);
    }
    GELOGI("[Dequeue] succeeded, device_id = %d, queue_id = %u, size = %zu", device_id, queue_id, tensor_raw_size);
  }
  return RT_ERROR_TO_GE_STATUS(ret);
}

Status HeterogeneousExchangeService::DequeueTensor(int32_t device_id, uint32_t queue_id, GeTensor &tensor,
                                                   ControlInfo &control_info) {
  if (IsClientQueue(queue_id)) {
    return ProcessDequeueBuffTensor(device_id, queue_id, tensor, control_info);
  }
  rtMbufPtr_t m_buf = nullptr;
  GE_CHK_STATUS_RET_NOLOG(DequeueMbuf(device_id, queue_id, &m_buf, control_info.timeout));
  GE_MAKE_GUARD(m_buf, [m_buf]() { GE_CHK_RT(rtMbufFree(m_buf)); });
  GE_CHK_STATUS_RET_NOLOG(CheckResult(m_buf, control_info));
  if (control_info.end_of_sequence_flag ||
      ((control_info.msg_info != nullptr) && ((control_info.msg_info->data_flag & kNullDataFlagBit) != 0U))) {
    // end of sequence or null data
    return SUCCESS;
  }
  auto &output_tensor_desc = tensor.MutableTensorDesc();
  void *data_buffer = nullptr;
  uint64_t data_buffer_size = 0U;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferAddr(m_buf, &data_buffer));
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferSize(m_buf, data_buffer_size));
  GE_CHK_BOOL_RET_STATUS(static_cast<size_t>(data_buffer_size) >= sizeof(RuntimeTensorDesc), FAILED,
                         "Dequeue size[%lu] is less than [%zu]",
                         data_buffer_size, sizeof(RuntimeTensorDesc));
  RuntimeTensorDesc *const mbuf_tensor_desc = PtrToPtr<void, RuntimeTensorDesc>(data_buffer);
  GE_CHK_STATUS_RET_NOLOG(UpdateTensorDesc(*mbuf_tensor_desc, output_tensor_desc));
  int64_t tensor_raw_size = -1;
  GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(output_tensor_desc.GetShape(), output_tensor_desc.GetFormat(),
                                                         output_tensor_desc.GetDataType(), tensor_raw_size),
                          "Failed to DequeueTensor");
  GELOGD("Tensor size = %zu", tensor_raw_size);
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufCopyToMem,
                                                                     device_id, queue_id);
  if (tensor_raw_size > 0) {
    auto output_aligned_ptr = MakeShared<AlignedPtr>(tensor_raw_size, kAlignmentVal64);
    GELOGD("Tensor buffer allocated, size = %zu", tensor_raw_size);
    GE_CHECK_NOTNULL(output_aligned_ptr);
    auto check_buffer_size = static_cast<size_t>(data_buffer_size) >=
        sizeof(RuntimeTensorDesc) + static_cast<size_t>(tensor_raw_size);
    GE_CHK_BOOL_RET_STATUS(check_buffer_size, FAILED, "Dequeue size[%lu] is less than [%zu]",
                           data_buffer_size, sizeof(RuntimeTensorDesc) + static_cast<size_t>(tensor_raw_size));
    const uint8_t *data_addr = PtrAdd<uint8_t>(
        PtrToPtr<void, uint8_t>(data_buffer), (sizeof(RuntimeTensorDesc) + 1UL), sizeof(RuntimeTensorDesc));
    if (memcpy_s(output_aligned_ptr->MutableGet(), static_cast<size_t>(tensor_raw_size), data_addr,
                 static_cast<size_t>(tensor_raw_size)) != EOK) {
      GELOGE(FAILED, "Failed to copy output tensor data copy size = %ld", tensor_raw_size);
      return FAILED;
    }
    tensor.SetData(output_aligned_ptr, static_cast<uint64_t>(tensor_raw_size));
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufCopyToMem,
                                                                     device_id, queue_id);
  GELOGD("[Dequeue] succeeded, device_id = %d, queue_id = %u, size = %zu", device_id, queue_id, tensor_raw_size);
  return SUCCESS;
}

Status HeterogeneousExchangeService::EnqueueMbufToClientQueue(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf,
                                                              int32_t timeout) {
  void *control_data = nullptr;
  void *data_buffer = nullptr;
  uint64_t head_size = 0U;
  uint64_t data_buffer_size = 0U;
  GE_CHK_STATUS_RET(RtsApiUtils::MbufGetPrivData(m_buf, &control_data, &head_size), "get mbuf priv data failed");
  GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferAddr(m_buf, &data_buffer), "get mbuf addr failed");
  GE_CHK_STATUS_RET(RtsApiUtils::MbufGetBufferSize(m_buf, data_buffer_size), "get mbuf size failed");
  rtMemQueueBuffInfo queue_buf_info = {data_buffer, data_buffer_size};
  rtMemQueueBuff_t queue_buf = {control_data, head_size, &queue_buf_info, 1U};
  auto ret = rtMemQueueEnQueueBuff(device_id, queue_id, &queue_buf, timeout);
  return (ret == RT_ERROR_NONE) ? SUCCESS : RT_ERROR_TO_GE_STATUS(ret);
}

Status HeterogeneousExchangeService::EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf,
                                                 int32_t timeout) {
  GE_CHK_STATUS_RET(SetTransId(device_id, queue_id, m_buf), "Set mbuf trans id failed, device_id = %d, queue_id = %u",
                    device_id, queue_id);
  if (IsClientQueue(queue_id)) {
    Status enqueue_ret = EnqueueMbufToClientQueue(device_id, queue_id, m_buf, timeout);
    if (enqueue_ret == SUCCESS) {
      // enqueue success, take ownership of m_buf
      GE_CHK_RT(rtMbufFree(m_buf));
    }
    return enqueue_ret;
  }
  auto ret = RT_ERROR_NONE;
  GE_CHK_STATUS_RET(InitializeEvents(device_id), "[Enqueue] [Init] failed, device id = %d", device_id);
  std::unique_lock<std::mutex> lk(enqueue_mu_);
  GE_CHK_STATUS_RET(EnsureEnqueueSubscribed(device_id, queue_id), "[Enqueue] [Init] failed, queue id = %u", queue_id);
  int32_t left_wait_time = timeout;
  GELOGD("Enqueue timout = %d ms.", timeout);
  const uint64_t begin_time = MsprofSysCycleTime();
  while (true) {
    subscribed_enqueues_[queue_id] = false;
    ret = rtMemQueueEnQueue(device_id, queue_id, m_buf);
    if (ret == RT_ERROR_NONE) {
      GELOGD("[EnqueueMbuf] success, device_id = %d, queue_id = %u", device_id, queue_id);
      if (ProfilingProperties::Instance().ProfilingTrainingTraceOn()) {
        (void)gert::GlobalProfilingWrapper::ReportApiInfoModelLevel(
            begin_time, MsprofSysCycleTime(), GetCurrentTransId(device_id, queue_id),
            static_cast<uint32_t>(gert::GeProfInfoType::kInputCopy));
      }
      return SUCCESS;
    }
    if (ret == ACL_ERROR_RT_QUEUE_FULL) {
      GELOGD("[Enqueue][MBuf] failed, queue is full, device_id = %d, queue_id = %u, left_wait_time = %d",
             device_id, queue_id, left_wait_time);
      // -1 means always wait.
      if ((left_wait_time == -1) || (left_wait_time > 0)) {
        GE_CHK_STATUS_RET(WaitF2NFEvent(queue_id, lk, left_wait_time), "[Enqueue] Wait f2nf event failed.");
        continue;
      }

      if (timeout != 0) {
        GELOGE(RT_FAILED, "[Enqueue][MBuf] timeout, device_id = %d, "
               "queue_id = %u, timeout = %d ms, rt_error_code = %d", device_id, queue_id, timeout, ret);
      }
      return RT_ERROR_TO_GE_STATUS(ret);
    }
    GELOGE(RT_FAILED, "[Enqueue][Mbuf] failed, device_id = %d, queue_id = %u, rt_error_code = %d",
           device_id, queue_id, ret);
    ret = RT_ERROR_TO_GE_STATUS(ret);
    break;
  }
  return ret;
}

Status HeterogeneousExchangeService::WaitF2NFEvent(const uint32_t queue_id,
                                                   std::unique_lock<std::mutex> &lk,
                                                   int32_t &left_wait_time) {
  int32_t wait_time =
      ((left_wait_time <= 0) || (left_wait_time > kEnqueueInterval)) ? kEnqueueInterval : left_wait_time;
  if (enqueue_cv_.wait_for(lk, std::chrono::milliseconds(wait_time), [this, queue_id] {
        return subscribed_enqueues_[queue_id];
      })) {
    GELOGI("[Enqueue] receive f2nf event");
  } else {
    if (left_wait_time >= wait_time) {
      left_wait_time -= wait_time;
    }
  }
  return SUCCESS;
}

Status HeterogeneousExchangeService::ClientQueueDequeueMbuf(int32_t device_id,
                                                            uint32_t queue_id,
                                                            rtMbufPtr_t *m_buf,
                                                            int32_t timeout) const {
  uint64_t data_buffer_size = 0U;
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufDequeue,
                                                                     device_id, queue_id);
  auto ret = rtMemQueuePeek(device_id, queue_id, &data_buffer_size, timeout);
  if (ret == RT_ERROR_NONE) {
    GE_CHK_RT_RET(rtMbufAlloc(m_buf, data_buffer_size));
    uint64_t head_size = 0U;
    void *control_data = nullptr;
    void *data_buffer = nullptr;
    RtsApiUtils::MbufGetPrivData(*m_buf, &control_data, &head_size);
    RtsApiUtils::MbufGetBufferAddr(*m_buf, &data_buffer);
    rtMemQueueBuffInfo queue_buf_info = {data_buffer, data_buffer_size};
    rtMemQueueBuff_t queue_buf = {control_data, head_size, &queue_buf_info, 1U};
    ret = rtMemQueueDeQueueBuff(device_id, queue_id, &queue_buf, timeout);
    if (ret == RT_ERROR_NONE) {
      HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                         ProfilerEvent::kMbufDequeue,
                                                                         device_id, queue_id);
      return SUCCESS;
    }
  }
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                     ProfilerEvent::kMbufDequeue,
                                                                     device_id, queue_id);
  return RT_ERROR_TO_GE_STATUS(ret);
}

Status HeterogeneousExchangeService::DequeueMbuf(int32_t device_id,
                                                 uint32_t queue_id,
                                                 rtMbufPtr_t *m_buf,
                                                 int32_t timeout) {
  auto ret = RT_ERROR_NONE;
  if (IsClientQueue(queue_id)) {
    return ClientQueueDequeueMbuf(device_id, queue_id, m_buf, timeout);
  }
  GE_CHK_STATUS_RET(InitializeEvents(device_id), "[Dequeue] [Init] failed, device id = %d", device_id);
  std::unique_lock<std::mutex> lk(dequeue_mu_);
  GE_CHK_STATUS_RET(EnsureDequeueSubscribed(device_id, queue_id), "[Dequeue] [Init] failed, queue id = %u", queue_id);
  int32_t left_wait_time = timeout;
  HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                     ProfilerEvent::kMbufDequeue,
                                                                     device_id, queue_id);
  const uint64_t begin_time = MsprofSysCycleTime();
  while (true) {
    subscribed_dequeues_[queue_id] = false;
    ret = rtMemQueueDeQueue(device_id, queue_id, m_buf);
    if (ret == RT_ERROR_NONE) {
      GELOGD("[DequeueMbuf] success, device_id = %d, queue_id = %u", device_id, queue_id);
      HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                                         ProfilerEvent::kMbufDequeue,
                                                                         device_id, queue_id);
      if (!ProfilingProperties::Instance().ProfilingTrainingTraceOn()) {
        return SUCCESS;
      }
      void *head_buf = nullptr;
      uint64_t head_size = 0U;
      uint64_t trans_id = UINT64_MAX;
      GE_CHECK_NOTNULL(m_buf);
      const rtError_t get_ret = rtMbufGetPrivInfo(*m_buf, &head_buf, &head_size);
      if ((get_ret == RT_ERROR_NONE) && (head_buf != nullptr) && (head_size >= sizeof(MsgInfo))) {
        const MsgInfo *const msg_info =
            PtrToPtr<char_t, MsgInfo>(static_cast<char_t *>(head_buf) + head_size - sizeof(MsgInfo));
        trans_id = msg_info->trans_id;
      } else {
        GELOGW("Dequeue mbuf get trans id failed.");
      }
      (void)gert::GlobalProfilingWrapper::ReportApiInfoModelLevel(
          begin_time, MsprofSysCycleTime(), trans_id, static_cast<uint32_t>(gert::GeProfInfoType::kInputCopy));
      return SUCCESS;
    }
    if (ret == ACL_ERROR_RT_QUEUE_EMPTY) {
      GELOGD("[Dequeue][MBuf] failed, queue is empty, device_id = %d, queue_id = %u, left_wait_time = %dms",
             device_id, queue_id, left_wait_time);
      // -1 means always wait.
      if ((left_wait_time == -1) || (left_wait_time > 0)) {
        GE_CHK_STATUS_RET(WaitEnqueueEvent(queue_id, lk, left_wait_time), "[Dequeue] Wait enqueue event failed.");
        continue;
      }
    }
    GELOGW("[Dequeue][MBuf] timeout, device_id = %d, queue_id = %u, timeout = %d ms, rt_error_code = %d",
           device_id, queue_id, timeout, ret);
    ret = RT_ERROR_TO_GE_STATUS(ret);
    break;
  }
  return ret;
}

Status HeterogeneousExchangeService::WaitEnqueueEvent(const uint32_t queue_id,
                                                      std::unique_lock<std::mutex> &lk,
                                                      int32_t &left_wait_time) {
  int32_t wait_time =
      ((left_wait_time <= 0) || (left_wait_time > kDequeueInterval)) ? kDequeueInterval : left_wait_time;
  if (dequeue_cv_.wait_for(lk, std::chrono::milliseconds(wait_time),
                           [this, queue_id] { return subscribed_dequeues_[queue_id]; })) {
    GELOGI("[Dequeue] receive enqueue event");
    HeterogeneousProfiler::Instance().RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                                       ProfilerEvent::kMbufDequeue, queue_id);
  } else {
    if (left_wait_time >= wait_time) {
      left_wait_time -= wait_time;
    }
  }
  return SUCCESS;
}

Status HeterogeneousExchangeService::CopyMbufHeadTo(void *m_buf, void *&control_data, size_t &head_size) {
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetPrivData(m_buf, &control_data, &head_size));
  return SUCCESS;
}

Status HeterogeneousExchangeService::MoveMbufTo(void *m_buf, const ControlInfo &control_info,
                                                std::shared_ptr<AlignedPtr> &aligned_ptr) {
  uint64_t buffer_size = 0;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferSize(m_buf, buffer_size));
  GE_CHK_BOOL_RET_STATUS(buffer_size > control_info.skip_size, FAILED, "Mbuf size must > skip size:%zu, but got %lu.",
                         control_info.skip_size, buffer_size);
  void *mbuf_addr = nullptr;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferAddr(m_buf, &mbuf_addr));
  const auto skip_size = control_info.skip_size;
  auto deleter = [skip_size](uint8_t *ptr) {
    GE_CHK_RT(rtBuffFree(ptr - skip_size));
    ptr = nullptr;
  };
  void *buff_free = nullptr;
  uint64_t buff_len = 0L;
  GE_CHK_RT_RET(rtMbufUnBuild(m_buf, &buff_free, &buff_len));
  aligned_ptr = AlignedPtr::BuildFromData(
      PtrAdd<uint8_t>(PtrToPtr<void, uint8_t>(mbuf_addr), (control_info.skip_size + 1UL), control_info.skip_size),
      deleter);
  return SUCCESS;
}

Status HeterogeneousExchangeService::CopyMbufTo(void *m_buf, void *data,
                                                size_t size, const ControlInfo &control_info) {
  uint64_t buffer_size = 0;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferSize(m_buf, buffer_size));
  GE_CHK_BOOL_RET_STATUS(buffer_size > control_info.skip_size, FAILED, "Mbuf size must > skip size:%zu, but got %lu.",
                         control_info.skip_size, buffer_size);
  void *mbuf_addr = nullptr;
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufGetBufferAddr(m_buf, &mbuf_addr));

  const uint8_t *data_buffer =
      PtrAdd<uint8_t>(PtrToPtr<void, uint8_t>(mbuf_addr), (control_info.skip_size + 1UL), control_info.skip_size);
  uint64_t data_size = buffer_size - static_cast<uint64_t>(control_info.skip_size);
  if (size != data_size) {
    GELOGW("User data size = %zu not equal with buffer size = %lu", size, data_size);
    data_size = std::min(size, data_size);
  }
  GE_CHK_BOOL_RET_STATUS(memcpy_s(data, size, data_buffer, data_size) == EOK,
                         FAILED,
                         "Failed to copy buffer to user, dst size = %zu, copy size = %lu",
                         size, data_size);
  return SUCCESS;
}

Status HeterogeneousExchangeService::GenTransId(const int32_t device_id, const uint32_t queue_id, uint64_t &trans_id,
                                                const uint64_t user_assign_trans_id) {
  if (user_assign_trans_id == UINT64_MAX) {
    GELOGE(PARAM_INVALID, "user assign trans id[%lu] must be smaller than UINT64_MAX[%lu]", user_assign_trans_id,
           UINT64_MAX);
    return PARAM_INVALID;
  }
  std::lock_guard<std::mutex> lk(trans_mu_);
  const TransInfoContext context{device_id, queue_id};
  uint64_t &last_trans_id_ref = trans_ids_[context];
  if (user_assign_trans_id != 0) {
    if (user_assign_trans_id < last_trans_id_ref) {
      GELOGE(PARAM_INVALID, "user assign trans id[%lu] cannot be smaller than last trans id[%lu]", user_assign_trans_id,
             last_trans_id_ref);
      return PARAM_INVALID;
    }
    trans_id = user_assign_trans_id;
    last_trans_id_ref = trans_id;
  } else {
    if (last_trans_id_ref == (UINT64_MAX - 1)) {
      GELOGE(FAILED, "trans id will reach UINT64_MAX[%lu], cannot gen trans id.", UINT64_MAX);
      return FAILED;
    }
    trans_id = ++last_trans_id_ref;
  }
  GELOGD("queue[%u] in device[%d] trans id=%lu.", trans_id);
  return SUCCESS;
}

uint64_t HeterogeneousExchangeService::GetCurrentTransId(int32_t device_id, uint32_t queue_id) {
  std::lock_guard<std::mutex> lk(trans_mu_);
  TransInfoContext context{device_id, queue_id};
  auto iter = trans_ids_.find(context);
  return (iter == trans_ids_.end()) ? UINT64_MAX : iter->second;
}

void HeterogeneousExchangeService::ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) {
  DestroyTransInfo(device_id, queue_id);
  return;
}
}  // namespace ge
