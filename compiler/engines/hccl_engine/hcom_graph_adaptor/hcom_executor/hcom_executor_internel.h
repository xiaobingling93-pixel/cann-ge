/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_EXECUTOR_INTERNEL_H
#define HCOM_EXECUTOR_INTERNEL_H

#include <vector>
#include <list>
#include <unordered_set>
#include <memory>
#include <thread>
#include <atomic>
#include <hcom_executor.h>
#include "executor_message.h"
#include "hcom_log.h"
#include "hccl/hcom.h"
#include "hcom_acl_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __cplusplus
}
#endif  // __cplusplus

namespace hccl {
struct AclHostMemDeleter {
  void operator()(void *ptr) const {
    if (ptr != nullptr) {
      (void)hrtFreeHost(ptr);
    }
  }
};

using AclHostMemPtr = std::unique_ptr<void, AclHostMemDeleter>;

constexpr s32 EVENT_QUERY_TIMEOUT_S = 120;     // event query等待时间
constexpr s32 THREAD_SLEEP_DURATION_US = 100;  // 线程等待消息休眠时间100us
constexpr s32 MAX_ENQUEUE_SIZE = 4;            // 单次最大的入栈数量
constexpr int32_t HCCL_STREAM_PRIORITY_HIGH = 0;

#define HCOM_EXECUTOR_ERR_BREAK(result, exeLog, callback, setFlag) \
  if (static_cast<u32>(result) != 0) {                             \
    exeLog;                                                        \
    callback(result);                                              \
    setFlag;                                                       \
    break;                                                         \
  }

enum class MsgQueueType { OPBASE_QUEUE = 0, REMOTE_ACCESS_QUEUE = 1 };

using EventInfo_t = struct TagEventInfo {
  HcclRtEvent event;
  StatusCallback callback;
  ~TagEventInfo() {
    event = nullptr;
    callback = nullptr;
  }
};

using ExcutorParallelPara_t = struct ExcutorParallelPara {
  HcclRtStream stream{nullptr};
  bool shutDown{false};
  std::queue<ExecutorMessage> messageQueue;
  std::queue<EventInfo_t> eventQueue;
  std::unique_ptr<std::thread> callbackNotifyThread{nullptr};
  std::unique_ptr<std::thread> messageProThread{nullptr};
  ExcutorParallelPara() {}
  ~ExcutorParallelPara() {
    HcclResult ret = HCCL_SUCCESS;
    shutDown = true;
    if (messageProThread) {
      if (messageProThread->joinable()) {
        HCCL_INFO("hcom executor wait messagePro thread");
        messageProThread->join();  // 等待线程执行后释放资源
      }
      messageProThread.reset(nullptr);
    }
    if (callbackNotifyThread) {
      if (callbackNotifyThread->joinable()) {
        HCCL_INFO("hcom executor wait callback notify thread");
        callbackNotifyThread->join();  // 等待线程执行后释放资源
      }
      callbackNotifyThread.reset(nullptr);
    }
    stream = nullptr;
    messageQueue = std::queue<ExecutorMessage>();
    while (!eventQueue.empty()) {
      EventInfo_t eventInfo = eventQueue.front();
      eventQueue.pop();
      ret = hrtEventDestroy(eventInfo.event);
      if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("rt_event_destroy failed, ret[%d]", ret);
      }
    }
  }
};

const std::string HCCL_TYPE_BROADCAST = "HcomBroadcast";
const std::string HCCL_TYPE_ALLREDUCE = "HcomAllReduce";
const std::string HCCL_TYPE_ALLGATHER = "HcomAllGather";
const std::string HCCL_TYPE_REDUCESCATTER = "HcomReduceScatter";
const std::string HCOM_SEND = "HcomSend";
const std::string HCOM_RECV = "HcomReceive";
const std::string HCOM_ALLTOALLV = "HcomAlltoAllV";

class HcomExecutor {
 public:
  ~HcomExecutor();

  HcclResult Initialize();

  HcclResult InitGroup();

  HcclResult InitHcclComm(const char *group, HcclComm comm);

  HcclResult Finalize();

  HcclResult HcomExecEnqueueOperation(HcomOperation_t opInfo, StatusCallback callback);

  HcclResult HcomExecEnqueueRemoteOperation(HcomRemoteOperation_t opInfo, StatusCallback callback);

  HcclResult HcomExecEnqueueRemoteAccess(const std::string &remoteAccessType,
                                         const std::vector<HcomRemoteAccessAddrInfo> &addrInfos,
                                         StatusCallback callback);

  HcclResult HcomExecEnqueueAllToAllV(HcomAllToAllVParams opInfo, StatusCallback callback);

  HcclResult HcomExecEnqueueAllToAllVC(HcomAllToAllVCParams opInfo, StatusCallback callback);

  HcomExecutor(HcomExecutor &) = delete;                   // 禁止拷贝
  HcomExecutor &operator=(const HcomExecutor &) = delete;  // 禁止赋值
  static HcomExecutor &GetInstance();

 private:
  HcomExecutor();  // 禁止用户自己声明并定义实例。把构造函数设为 private

  HcclResult MessageProcessThreadLoop(MsgQueueType queType);
  HcclResult CallbackNotifyThreadLoop(MsgQueueType queType);

  HcclResult EnqueueAllreduce(HcomOperation_t opInfo, StatusCallback callback);
  HcclResult EnqueueBroadcast(HcomOperation_t opInfo, StatusCallback callback);
  HcclResult EnqueueAllGather(HcomOperation_t opInfo, StatusCallback callback);
  HcclResult EnqueueReduceScatter(HcomOperation_t opInfo, StatusCallback callback);

  HcclResult EnqueueEvent(HcclRtStream stream, StatusCallback callback, MsgQueueType queType);

  HcclResult ExecuteOperation(const HcomOperation_t &opInfo);
  HcclResult ExecuteAllreduce(const HcomOperation_t &opInfo);
  HcclResult ExecuteBroadcast(const HcomOperation_t &opInfo);
  HcclResult ExecuteAllGather(const HcomOperation_t &opInfo);
  HcclResult ExecuteReduceScatter(const HcomOperation_t &opInfo);

  HcclResult ExecuteAlltoAll(const HcomAllToAllVParams &opInfo, bool isGatherAlltoAll);

  HcclResult ExecuteAlltoAllVC(const HcomAllToAllVCParams &opInfo);

  HcclResult RunGather(std::vector<u64> &addrInfo, std::vector<u64> &addrInfoCountPerRank, u32 rankSize,
                       u64 *sendCounts, u64 *sdispls, void *sendDevBuf, s32 addrLength);

  void PushGatherAlltoAllParaToQue(HcomGatherAllToAllVParams &gatherParams, void *sendCounts, void *sdispls,
                                   void *recvCounts, void *rdispls, std::function<void(HcclResult status)> callback);

  void GatherMemCopyThread(void *baseAddr, u64 offset, std::vector<u64> &addrInfo, u64 beginIndex, u64 count,
                           u64 tmpMemSize);

  HcclResult AllocGatherAlltoAllMem(u64 *&sendCounts, u64 *&sdispls, void *&recvCountsPtr, void *&rdisplsPtr,
                                    u32 rankSize);

  void DeleteGatherAlltoAllMem(HcomAllToAllVParams &params);

  void CleanQueueResources();

  ExecutorMessage PopMessagesFromQueue(MsgQueueType queueType);
  EventInfo_t PopEventFromQueue(MsgQueueType queueType);
  void PushMessageToQueue(ExecutorMessage &message, MsgQueueType queueType);
  HcclResult GetComm(const char *group, HcclComm *comm);
  HcclResult AtomicInitSet();
  void AtomicInitClear();

  std::unordered_set<HcclComm> comms_;
  s32 deviceLogicId_;
  mutable std::mutex msgMutex_;
  mutable std::mutex eventMutex_;
  std::map<MsgQueueType, ExcutorParallelPara_t> parralMap_;
  std::atomic_flag executorInitedFlag_;
  std::list<std::vector<HcomRemoteAccessAddrInfo>> listRemoteAccessInfos;
  HcclWorkflowMode workflowMode_{HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE};
  HcomAllToAllVParams gatherAlltoAllInfo_;
};
}  // namespace hccl

#endif