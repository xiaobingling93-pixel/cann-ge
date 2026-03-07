/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_DEPLOY_HETEROGENEOUS_EXCHANGE_SERVICE_H_
#define RUNTIME_DEPLOY_HETEROGENEOUS_EXCHANGE_SERVICE_H_

#include <mutex>
#include <set>
#include <thread>
#include <condition_variable>
#include <atomic>
#include "runtime/rt.h"
#include "runtime/rt_mem_queue.h"
#include "common/thread_pool/thread_pool.h"
#include "framework/common/runtime_tensor_desc.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
class HeterogeneousExchangeService : public ExchangeService {
 public:
  static HeterogeneousExchangeService &GetInstance();

  HeterogeneousExchangeService();
  ~HeterogeneousExchangeService() override;

  struct TransInfoContext {
    int32_t device_id;
    uint32_t queue_id;
    bool operator < (const TransInfoContext &other) const {
      if (device_id != other.device_id) {
        return device_id < other.device_id;
      } else {
        return queue_id < other.queue_id;
      }
    }
  };

  Status Initialize(int32_t device_id);

  Status Finalize();

  Status CreateQueue(const int32_t device_id,
                     const std::string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override;
  Status DestroyQueue(int32_t device_id, uint32_t queue_id) override;
  Status Enqueue(int32_t device_id, uint32_t queue_id, const void *data, size_t size,
                 const ControlInfo &control_info) override;
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override;
  Status Enqueue(int32_t device_id,
                 uint32_t queue_id,
                 size_t size,
                 const FillFunc &fill_func,
                 const ControlInfo &control_info) override;
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override;
  Status Dequeue(int32_t device_id, uint32_t queue_id, void *data, size_t size, ControlInfo &control_info) override;
  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) override;
  Status DequeueTensor(int32_t device_id, uint32_t queue_id, GeTensor &tensor, ControlInfo &control_info) override;
  Status EnsureInitialized(int32_t device_id);
  void DestroyTransInfo(const int32_t device_id, const uint32_t queue_id);
  Status SetTransId(const int32_t device_id, const uint32_t queue_id, rtMbufPtr_t mbuf);
  Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) override;
  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override;
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override;
  using ExchangeService::CreateQueue;
  void AddClientQueue(const uint32_t queue_id);
  bool IsClientQueue(const uint32_t queue_id);
  static Status CheckResult(rtMbufPtr_t m_buf, ControlInfo &control_info);

  // put m_buf info to client queue, do not take ownership of m_buf
  static Status EnqueueMbufToClientQueue(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout);
 private:
  static Status MoveMbufTo(void *m_buf, const ControlInfo &control_info,
                           std::shared_ptr<AlignedPtr> &aligned_ptr);
  static Status CopyMbufTo(void *m_buf, void *data, size_t size, const ControlInfo &control_info);
  static Status CopyMbufHeadTo(void *m_buf, void *&control_data, size_t &head_size);
  void ProcessEmptyToNotEmptyEvent(const uint32_t queue_id);
  void ProcessF2NFEvent(const uint32_t queue_id);
  void WaitEvents(const int32_t device_id);
  Status InitializeEvents(const int32_t device_id);
  Status EnsureEnqueueSubscribed(const int32_t device_id, const uint32_t queue_id);
  Status EnsureDequeueSubscribed(const int32_t device_id, const uint32_t queue_id);
  static Status CheckResult(void *head_buf, size_t head_size, ControlInfo &control_info);
  static Status UpdateTensorDesc(const RuntimeTensorDesc &runtime_tensor_desc, GeTensorDesc &tensor_desc);
  Status WaitF2NFEvent(const uint32_t queue_id, std::unique_lock<std::mutex> &lk, int32_t &left_wait_time);
  Status WaitEnqueueEvent(const uint32_t queue_id, std::unique_lock<std::mutex> &lk, int32_t &left_wait_time);
  static Status InitHeadInfo(const ControlInfo &control_info, rtMbufPtr_t mbuf);
  Status ProcessEnqueueBuff(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                            const ControlInfo &control_info);
  Status ProcessEnqueueMbuf(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                           rtMbufPtr_t mbuf, const ControlInfo &control_info);
  Status GenTransId(const int32_t device_id, const uint32_t queue_id, uint64_t &trans_id,
                    const uint64_t user_assign_trans_id);
  uint64_t GetCurrentTransId(int32_t device_id, uint32_t queue_id);
  Status ClientQueueDequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) const;
  Status AllocAlignedBuffer(const size_t buffer_size, uint8_t *&aligned_ptr, int32_t device_id);
  Status GetOrCreateRtCtx(rtContext_t &ctx, int32_t device_id);
  Status ProcessDequeueBuffTensor(int32_t device_id, uint32_t queue_id, GeTensor &tensor, ControlInfo &control_info);

  Status MultiThreadCopy(uint8_t *dst, size_t dst_size, const uint8_t *src, size_t src_size);

  std::mutex mu_;
  std::set<int32_t> initialized_devices_;

  std::mutex trans_mu_;
  std::map<TransInfoContext, uint64_t> trans_ids_;

  std::mutex dequeue_mu_;
  std::condition_variable dequeue_cv_;
  std::map<uint32_t, bool> subscribed_dequeues_; // key is qid, true means not empty

  std::mutex enqueue_mu_;
  std::condition_variable enqueue_cv_;
  std::map<uint32_t, bool> subscribed_enqueues_; // key is qid, true means not full

  std::atomic<bool> waiting_{true};
  std::vector<std::thread> events_threads_;
  std::mutex events_mu_;
  std::set<int32_t> events_devices_;
  std::mutex wait_event_mu_;
  std::condition_variable wait_events_cv_;
  bool event_thread_starting_{true};
  std::mutex client_q_mu_;
  std::set<uint32_t> client_queue_ids_;
  std::mutex ctx_mu_;
  rtContext_t rt_context_ = nullptr;

  ThreadPool copy_thread_pool_;
};
}  // namespace ge

#endif  // RUNTIME_DEPLOY_HETEROGENEOUS_EXCHANGE_SERVICE_H_
