/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <nlohmann/json.hpp>
#include "hccl/hcom.h"
#include "op_hcom_comm.h"
#include "graph/ge_local_context.h"
#include "framework/common/ge_types.h"  // ge对外options
#include "hcom_executor_internel.h"
#include "adapter_dlhcclfunc.h"

HcclResult HcomExecInitialize() {
  HCCL_INFO("Hcom Excutor Initialize start.");
  HcclResult ret = hccl::HcomExecutor::GetInstance().Initialize();
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Initialize][HcomExecutor]errNo[0x%016llx]initialize hcom executor failed", HCCL_ERROR_CODE(ret)),
      ret);
  HCCL_RUN_INFO("[Initialize][HcomExecutor]Hcom Excutor Initialize end. ret[%d]", ret);
  return ret;
}

HcclResult HcomExecFinalize() {
  HCCL_INFO("Hcom Excutor Finalize start.");
  HcclResult ret = hccl::HcomExecutor::GetInstance().Finalize();
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Finalize][HcomExecutor]errNo[0x%016llx]finalize hcom executor failed", HCCL_ERROR_CODE(ret)),
              ret);
  HCCL_RUN_INFO("[Finalize][HcomExecutor]Hcom Excutor Finalize end. ret[%d]", ret);
  return ret;
}

HcclResult HcomExecEnqueueOperation(struct HcomOperation opInfo, std::function<void(HcclResult status)> callback) {
  HCCL_RUN_INFO(
      "Entry-HcomExecEnqueueOperation:hcclType[%s], inputPtr[%p], outputPtr[%p], count[%llu], "
      "dataType[%s], op[%s]",
      opInfo.hcclType.c_str(), opInfo.inputPtr, opInfo.outputPtr, opInfo.count,
      hccl::GetDataTypeEnumStr(opInfo.dataType).c_str(), hccl::GetReduceOpEnumStr(opInfo.opType).c_str());
  return hccl::HcomExecutor::GetInstance().HcomExecEnqueueOperation(opInfo, callback);
}

HcclResult HcomExecEnqueueRemoteOperation(struct HcomRemoteOperation opInfo,
                                          std::function<void(HcclResult status)> callback) {
  HCCL_ERROR("[HcomExec][EnqueueRemoteOperation]HcomExecEnqueueRemoteOperation is not support.");
  return HCCL_E_NOT_SUPPORT;
}

HcclResult HcomExecEnqueueRemoteAccess(const std::string &remoteAccessType,
                                       const std::vector<HcomRemoteAccessAddrInfo> &addrInfos,
                                       std::function<void(HcclResult status)> callback) {
  HCCL_ERROR("HcomExecEnqueueRemoteAccess is not support");
  return HCCL_E_NOT_SUPPORT;
}

HcclResult HcomExecEnqueueAllToAllV(HcomAllToAllVParams params, std::function<void(HcclResult status)> callback) {
  HCCL_RUN_INFO("Entry-HcomExecEnqueueAllToAllV:AlltoAllV");
  return hccl::HcomExecutor::GetInstance().HcomExecEnqueueAllToAllV(params, callback);
}

HcclResult HcomExecEnqueueAllToAllVC(HcomAllToAllVCParams params, std::function<void(HcclResult status)> callback) {
  HCCL_RUN_INFO("Entry-HcomExecEnqueueAllToAllVC:AlltoAllVC");
  return hccl::HcomExecutor::GetInstance().HcomExecEnqueueAllToAllVC(params, callback);
}

HcclResult HcomExecEnqueueGatherAllToAllV(HcomGatherAllToAllVParams params,
                                          std::function<void(HcclResult status)> callback) {
  HCCL_ERROR("HcomExecEnqueueGatherAllToAllV is not supported.");
  return HCCL_E_NOT_SUPPORT;
}

namespace hccl {
namespace {
const MsgQueueType MSGQUE_TYPE[] = {MsgQueueType::OPBASE_QUEUE, MsgQueueType::REMOTE_ACCESS_QUEUE};
const u32 GATHER_THREAD_NUM = 16;
const u32 NUM_TWO = 2;
}  // namespace

HcomExecutor::HcomExecutor()
    : deviceLogicId_(-1), msgMutex_(), eventMutex_(), parralMap_(), executorInitedFlag_(ATOMIC_FLAG_INIT) {}

HcomExecutor::~HcomExecutor() {
  CleanQueueResources();
}

HcclResult HcomExecutor::InitGroup() {
  std::string groupConf;
  nlohmann::json groupListConf;
  ge::graphStatus geRet = ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, groupConf);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_WARNING("[HcomOpsKernelInfoStore][InitHcom]OPTION_EXEC_HCOM_GROUPLIST is not found."), HCCL_SUCCESS);

  HCCL_DEBUG("groupList:%s", groupConf.c_str());
  CHK_RET(SalParseInformation(groupListConf, groupConf));
  std::vector<nlohmann::json> groupList = groupListConf.get<std::vector<nlohmann::json>>();
  for (auto &groupInfo : groupList) {
    HCCL_DEBUG("groupInfo:%s", groupInfo.dump().c_str());
    std::vector<u32> ranks = groupInfo["group_rank_list"];
    std::string groupName = groupInfo["group_name"];
    HCCL_DEBUG("groupName:%s", groupName.c_str());

    u32 curRank = 0;
    CHK_RET(HcomGetRankId(HCCL_WORLD_GROUP, &curRank));
    if (!HcomFindGroup(groupName.c_str()) && find(ranks.begin(), ranks.end(), curRank) != ranks.end()) {
      if (strncmp(groupName.c_str(), HCCL_WORLD_GROUP, sizeof(HCCL_WORLD_GROUP)) == 0) {
        HCCL_WARNING("[HcomOpsKernelInfoStore][InitHcom]cur groupname is HCCL_WORLD_GROUP.");
        continue;
      }
      CHK_RET(HcomCreateGroup(groupName.c_str(), ranks.size(), ranks.data()));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::Initialize() {
  HcclResult ret;
  CHK_RET(hrtGetDeviceRefresh(&deviceLogicId_));

  CHK_RET(HcomInitialize());

  CHK_RET(InitGroup());

  // 执行器创建stream
  rtStream_t rtStreamOpbase = nullptr;        // StreamType::STREAM_TYPE_ONLINE
  rtStream_t rtStreamRemoteAccess = nullptr;  // StreamType::STREAM_TYPE_ONLINE
  if (hrtStreamCreateWithFlags(&rtStreamOpbase, HCCL_STREAM_PRIORITY_HIGH,
                               RT_STREAM_FAST_LAUNCH | RT_STREAM_FAST_SYNC) == HCCL_SUCCESS &&
      hrtStreamCreateWithFlags(&rtStreamRemoteAccess, HCCL_STREAM_PRIORITY_HIGH,
                               RT_STREAM_FAST_LAUNCH | RT_STREAM_FAST_SYNC) == HCCL_SUCCESS) {
    parralMap_[MsgQueueType::OPBASE_QUEUE].stream = const_cast<void *>(rtStreamOpbase);
    CHK_PTR_NULL(parralMap_[MsgQueueType::OPBASE_QUEUE].stream);
    parralMap_[MsgQueueType::REMOTE_ACCESS_QUEUE].stream = const_cast<void *>(rtStreamRemoteAccess);
    CHK_PTR_NULL(parralMap_[MsgQueueType::REMOTE_ACCESS_QUEUE].stream);
    HCCL_INFO("Construct stream success, streamType STREAM_TYPE_ONLINE");
  } else {
    REPORT_PREDEFINED_ERR_MSG("EI0007", std::vector<const char *>({"resource_type", "resource_info"}),
                              std::vector<const char *>({"stream", "streamType: STREAM_TYPE_ONLINE"}));
    HCCL_ERROR("[Stream]Construct stream failed, errNo[0x%016llx] rtStreamCreate error",
               HCCL_ERROR_CODE(HCCL_E_RUNTIME));
  }

  bool errorFlag = false;
  do {
    workflowMode_ = HcomGetWorkflowMode();
    // 启动消息处理线
    parralMap_[MsgQueueType::OPBASE_QUEUE].messageProThread.reset(
        new (std::nothrow) std::thread(&HcomExecutor::MessageProcessThreadLoop, this, MsgQueueType::OPBASE_QUEUE));
    CHK_PRT_BREAK(!parralMap_[MsgQueueType::OPBASE_QUEUE].messageProThread,
                  HCCL_ERROR("[Initialize][HcomExecutor]OPBASED MessageProcess threadPtr is Null"), errorFlag = true);

    parralMap_[MsgQueueType::REMOTE_ACCESS_QUEUE].messageProThread.reset(new (std::nothrow) std::thread(
        &HcomExecutor::MessageProcessThreadLoop, this, MsgQueueType::REMOTE_ACCESS_QUEUE));
    CHK_PRT_BREAK(!parralMap_[MsgQueueType::REMOTE_ACCESS_QUEUE].messageProThread,
                  HCCL_ERROR("[Initialize][HcomExecutor]REMOTE_ACCESS MessageProcess threadPtr is Null"),
                  errorFlag = true);

    // 启动回调处理线程
    parralMap_[MsgQueueType::OPBASE_QUEUE].callbackNotifyThread.reset(
        new (std::nothrow) std::thread(&HcomExecutor::CallbackNotifyThreadLoop, this, MsgQueueType::OPBASE_QUEUE));
    CHK_PRT_BREAK(!parralMap_[MsgQueueType::OPBASE_QUEUE].callbackNotifyThread,
                  HCCL_ERROR("[Initialize][HcomExecutor]OPBASED callback Notify threadPtr is Null"), errorFlag = true);

    parralMap_[MsgQueueType::REMOTE_ACCESS_QUEUE].callbackNotifyThread.reset(new (std::nothrow) std::thread(
        &HcomExecutor::CallbackNotifyThreadLoop, this, MsgQueueType::REMOTE_ACCESS_QUEUE));
    CHK_PRT_BREAK(!parralMap_[MsgQueueType::REMOTE_ACCESS_QUEUE].callbackNotifyThread,
                  HCCL_ERROR("[Initialize][HcomExecutor]REMOTE_ACCESS callback Notify threadPtr is Null"),
                  errorFlag = true);

    /* 初始化竞争, 只允许被初始化一次 */
    ret = AtomicInitSet();
    CHK_PRT_BREAK(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Initialize][HcomExecutor]errNo[0x%016llx] repeat initialized", HCCL_ERROR_CODE(ret)),
                  errorFlag = true);
  } while (0);

  if (errorFlag) {
    HCCL_ERROR("[Initialize][HcomExecutor]hcom executor init failed");
    if (Finalize() != HCCL_SUCCESS) {
      HCCL_ERROR("[Initialize][HcomExecutor]hcom finalize failed");
    }
    return HCCL_E_INTERNAL;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::InitHcclComm(const char *group, HcclComm comm) {
  std::string strGroup = (group == nullptr) ? "" : group;
  HCCL_DEBUG("[InitHcclComm][HcomExecutor] group[%s]", strGroup.c_str());
  CHK_PRT_RET((comm == nullptr), HCCL_ERROR("comm is not initialized"), HCCL_E_PTR);
  if (comms_.find(comm) == comms_.end()) {
    // 申请device内存
    void *inCCLBufferPtr = nullptr;
    u64 inCCLBufferSize = 0;
    void *outCCLBufferPtr = nullptr;
    u64 outCCLBufferSize = 0;
    CHK_RET(GetInCCLbuffer(group, inCCLBufferPtr, inCCLBufferSize));
    CHK_RET(GetOutCCLbuffer(group, outCCLBufferPtr, outCCLBufferSize));
    if (inCCLBufferPtr == nullptr || outCCLBufferPtr == nullptr) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
    }
    comms_.insert(comm);
  }
  return HCCL_SUCCESS;
}

HcomExecutor &HcomExecutor::GetInstance() {
  static HcomExecutor executor;
  return executor;
}

HcclResult HcomExecutor::AtomicInitSet() {
  CHK_PRT_RET(executorInitedFlag_.test_and_set(),
              HCCL_ERROR("[AtomicInitSet][HcomExecutor]errNo[0x%016llx] executor already been initialized",
                         HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
              HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

void HcomExecutor::AtomicInitClear() {
  executorInitedFlag_.clear();
}

HcclResult HcomExecutor::Finalize() {
  // 清理初始化标志位
  AtomicInitClear();
  for (auto &key : MSGQUE_TYPE) {
    parralMap_[key].shutDown = true;
    if (parralMap_[key].messageProThread != nullptr) {
      parralMap_[key].messageProThread->join();
    }
    if (parralMap_[key].callbackNotifyThread != nullptr) {
      parralMap_[key].callbackNotifyThread->join();
    }
  }
  CleanQueueResources();
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::HcomExecEnqueueOperation(HcomOperation_t opInfo, StatusCallback callback) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(InitHcclComm(opInfo.group, comm));

  CHK_SMART_PTR_NULL(parralMap_[MsgQueueType::OPBASE_QUEUE].stream);

  if (parralMap_[MsgQueueType::OPBASE_QUEUE].shutDown) {
    HCCL_ERROR("[HcomExec][EnqueueOperation]thread has shut down, Enqueue hcom operation fail, op type[%s]",
               opInfo.hcclType.c_str());
    return HCCL_E_INTERNAL;
  }

  if (opInfo.hcclType == HCCL_TYPE_ALLREDUCE) {
    CHK_RET(EnqueueAllreduce(opInfo, callback));
  } else if (opInfo.hcclType == HCCL_TYPE_BROADCAST) {
    CHK_RET(EnqueueBroadcast(opInfo, callback));
  } else if (opInfo.hcclType == HCCL_TYPE_ALLGATHER) {
    CHK_RET(EnqueueAllGather(opInfo, callback));
  } else if (opInfo.hcclType == HCCL_TYPE_REDUCESCATTER) {
    CHK_RET(EnqueueReduceScatter(opInfo, callback));
  } else {
    HCCL_ERROR("[HcomExec][EnqueueOperation]Unsupported operation type, hcclType=%s", opInfo.hcclType.c_str());
    return HCCL_E_NOT_SUPPORT;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::ExecuteAlltoAll(const HcomAllToAllVParams &opInfo, bool isGatherAlltoAll) {
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  CHK_RET(HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));

  if (!isGatherAlltoAll) {
    HcclResult ret;
    u32 rankSize = 0;
    CHK_RET(HcomGetRankSize(opInfo.group, &rankSize));

    void *sendCounts = nullptr;
    ret = hrtMallocHost(&sendCounts, rankSize * sizeof(u64));
    CHK_PRT_RET(ret != HCCL_SUCCESS || sendCounts == nullptr,
                HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
    AclHostMemPtr sendCountsPtr(sendCounts);
    CHK_RET(hrtMemSyncCopy(sendCountsPtr.get(), rankSize * sizeof(u64), opInfo.sendcounts, rankSize * sizeof(u64),
                           HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    void *sdispls = nullptr;
    ret = hrtMallocHost(&sdispls, rankSize * sizeof(u64));
    CHK_PRT_RET(ret != HCCL_SUCCESS || sdispls == nullptr,
                HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
    AclHostMemPtr sdisplsPtr(sdispls);
    CHK_RET(hrtMemSyncCopy(sdisplsPtr.get(), rankSize * sizeof(u64), opInfo.sdispls, rankSize * sizeof(u64),
                           HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    void *recvCounts = nullptr;
    ret = hrtMallocHost(&recvCounts, rankSize * sizeof(u64));
    CHK_PRT_RET(ret != HCCL_SUCCESS || recvCounts == nullptr,
                HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
    AclHostMemPtr recvCountsPtr(recvCounts);
    CHK_RET(hrtMemSyncCopy(recvCountsPtr.get(), rankSize * sizeof(u64), opInfo.recvcounts, rankSize * sizeof(u64),
                           HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    void *rdispls = nullptr;
    ret = hrtMallocHost(&rdispls, rankSize * sizeof(u64));
    CHK_PRT_RET(ret != HCCL_SUCCESS || rdispls == nullptr,
                HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
    AclHostMemPtr rdisplsPtr(rdispls);
    CHK_RET(hrtMemSyncCopy(rdisplsPtr.get(), rankSize * sizeof(u64), opInfo.rdispls, rankSize * sizeof(u64),
                           HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    CHK_RET(HcceAlltoAllV(opInfo.sendbuf, sendCountsPtr.get(), sdisplsPtr.get(), opInfo.sendtype, opInfo.recvbuf,
                          recvCountsPtr.get(), rdisplsPtr.get(), opInfo.recvtype, comm,
                          parralMap_[MsgQueueType::OPBASE_QUEUE].stream));
  } else {
    CHK_RET(HcceAlltoAllV(opInfo.sendbuf, opInfo.sendcounts, opInfo.sdispls, opInfo.sendtype, opInfo.recvbuf,
                          opInfo.recvcounts, opInfo.rdispls, opInfo.recvtype, comm,
                          parralMap_[MsgQueueType::OPBASE_QUEUE].stream));
  }
  CHK_RET(HcomSetWorkflowMode(lastWorkflowMode));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::ExecuteAlltoAllVC(const HcomAllToAllVCParams &opInfo) {
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  CHK_RET(HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  u32 rankSize = 0;
  CHK_RET(HcomGetRankSize(opInfo.group, &rankSize));

  void *sendCountMatrix = nullptr;
  HcclResult ret = hrtMallocHost(&sendCountMatrix, rankSize * rankSize * sizeof(u64));
  CHK_PRT_RET(ret != HCCL_SUCCESS || sendCountMatrix == nullptr,
              HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
  AclHostMemPtr sendCountMatrixPtr(sendCountMatrix);
  CHK_RET(hrtMemSyncCopy(sendCountMatrixPtr.get(), rankSize * rankSize * sizeof(u64), opInfo.sendcountmatrix,
                         rankSize * rankSize * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

  CHK_RET(HcceAlltoAllVC(opInfo.sendbuf, sendCountMatrixPtr.get(), opInfo.sendtype, opInfo.recvbuf, opInfo.recvtype,
                         comm, parralMap_[MsgQueueType::OPBASE_QUEUE].stream));

  CHK_RET(HcomSetWorkflowMode(lastWorkflowMode));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::ExecuteOperation(const HcomOperation_t &opInfo) {
  HcclResult ret = HCCL_SUCCESS;
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  if (lastWorkflowMode >= HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED ||
      lastWorkflowMode < HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
    HCCL_ERROR("[Execute][Operation]WorkflowMode[%d] invalid", lastWorkflowMode);
    return HCCL_E_INTERNAL;
  }
  CHK_RET(HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
  if (opInfo.hcclType == HCCL_TYPE_ALLREDUCE) {
    ret = ExecuteAllreduce(opInfo);
  } else if (opInfo.hcclType == HCCL_TYPE_BROADCAST) {
    ret = ExecuteBroadcast(opInfo);
  } else if (opInfo.hcclType == HCCL_TYPE_ALLGATHER) {
    ret = ExecuteAllGather(opInfo);
  } else if (opInfo.hcclType == HCCL_TYPE_REDUCESCATTER) {
    ret = ExecuteReduceScatter(opInfo);
  } else {
    HCCL_ERROR("[Execute][Operation]Unsupported operation type, op_type=%s", opInfo.hcclType.c_str());
    ret = HCCL_E_NOT_SUPPORT;
  }
  CHK_RET(HcomSetWorkflowMode(lastWorkflowMode));
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("[Execute][Operation]execute operation fail, op_type=%s", opInfo.hcclType.c_str());
  }
  return ret;
}

HcclResult HcomExecutor::ExecuteBroadcast(const HcomOperation_t &opInfo) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(HcceBroadcast(opInfo.inputPtr, opInfo.count, opInfo.dataType, opInfo.root, comm,
                        parralMap_[MsgQueueType::OPBASE_QUEUE].stream));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::ExecuteAllreduce(const HcomOperation_t &opInfo) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(HcceAllReduce(opInfo.inputPtr, opInfo.outputPtr, opInfo.count, opInfo.dataType, opInfo.opType, comm,
                        parralMap_[MsgQueueType::OPBASE_QUEUE].stream));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::ExecuteAllGather(const HcomOperation_t &opInfo) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(HcceAllGather(opInfo.inputPtr, opInfo.outputPtr, opInfo.count, opInfo.dataType, comm,
                        parralMap_[MsgQueueType::OPBASE_QUEUE].stream));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::ExecuteReduceScatter(const HcomOperation_t &opInfo) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(HcceReduceScatter(opInfo.inputPtr, opInfo.outputPtr, opInfo.count, opInfo.dataType, opInfo.opType, comm,
                            parralMap_[MsgQueueType::OPBASE_QUEUE].stream));

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::MessageProcessThreadLoop(MsgQueueType queType) {
  // 给当前线程添加名字
  SetThreadName("Hccl_MessageProcess");

  HCCL_INFO("MessageProcessThreadLoop start queType:[%d]", queType);
  HcclResult rtRet = hrtSetDevice(deviceLogicId_);
  if (rtRet != HCCL_SUCCESS) {
    HCCL_ERROR("[Process][ThreadLoop]errNo[0x%016llx] set device error", HCOM_ERROR_CODE(rtRet));
    parralMap_[queType].shutDown = true;
    return rtRet;
  }
  HcomSetWorkflowMode(workflowMode_);

  HcclResult ret = HCCL_SUCCESS;

  while (!parralMap_[queType].shutDown) {
    if (parralMap_[queType].messageQueue.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(THREAD_SLEEP_DURATION_US));
      continue;
    }

    ExecutorMessage message = PopMessagesFromQueue(queType);
    StatusCallback statusCallback = message.GetStatusCallback();
    if (message.GetMessageType() == ExecutorMessageType::MSG_OPBASED) {
      ret = ExecuteOperation(message.GetOperationMessage());
      HCOM_EXECUTOR_ERR_BREAK(ret, HCCL_ERROR("[Process][ThreadLoop]execute operation error, ret[%d]", ret),
                              statusCallback, parralMap_[queType].shutDown = true);
    } else if (message.GetMessageType() == ExecutorMessageType::MSG_ALLTOALL) {
      ret = ExecuteAlltoAll(message.GetAlltoAllMessage(), message.isGatherAlltoAll_);
      if (message.isGatherAlltoAll_) {
        HcomAllToAllVParams opInfo = message.GetAlltoAllMessage();
        DeleteGatherAlltoAllMem(opInfo);
      }
      HCOM_EXECUTOR_ERR_BREAK(ret, HCCL_ERROR("[Process][ThreadLoop]execute AllToAll error, ret[%d]", ret),
                              statusCallback, parralMap_[queType].shutDown = true);
    } else if (message.GetMessageType() == ExecutorMessageType::MSG_ALLTOALLVC) {
      ret = ExecuteAlltoAllVC(message.GetAlltoAllVCMessage());
      HCOM_EXECUTOR_ERR_BREAK(ret, HCCL_ERROR("[Process][ThreadLoop]execute AllToAllVC error, ret[%d]", ret),
                              statusCallback, parralMap_[queType].shutDown = true);
    }

    // 每个Hcom算子需申请一个event同步
    ret = EnqueueEvent(parralMap_[queType].stream, statusCallback, queType);
    HCOM_EXECUTOR_ERR_BREAK(ret, HCCL_ERROR("[Process][ThreadLoop]enqueue event error, ret[%d]", ret), statusCallback,
                            parralMap_[queType].shutDown = true);

    ret = hcclStreamSynchronize(parralMap_[queType].stream);
    HCOM_EXECUTOR_ERR_BREAK(ret, HCCL_ERROR("[Process][ThreadLoop]hcclStreamSynchronize error, ret[%d]", ret),
                            statusCallback, parralMap_[queType].shutDown = true);
  }

  CHK_RET(hrtResetDevice(deviceLogicId_));

  return ret;
}

HcclResult HcomExecutor::CallbackNotifyThreadLoop(MsgQueueType queType) {
  // 给当前线程添加名字
  SetThreadName("Hccl_Notify");

  HcclResult rtRet = hrtSetDevice(deviceLogicId_);
  if (rtRet != HCCL_SUCCESS) {
    HCCL_ERROR("[Callback][NotifyThreadLoop]errNo[0x%016llx] set device error", HCOM_ERROR_CODE(rtRet));
    parralMap_[queType].shutDown = true;
    return rtRet;
  }
  HcomSetWorkflowMode(workflowMode_);

  HcclResult status = HCCL_SUCCESS;
  while (!parralMap_[queType].shutDown) {
    if (parralMap_[queType].eventQueue.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(THREAD_SLEEP_DURATION_US));
      continue;
    }
    EventInfo_t eventInfo = PopEventFromQueue(queType);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EVENT_QUERY_TIMEOUT_S);
    bool bTimeout = false;
    HcclResult ret;
    while (true) {
      ret = hrtEventQuery(eventInfo.event);
      if (ret == HCCL_SUCCESS) {
        break;
      }
      bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
      if (bTimeout) {
        HCCL_ERROR("[Callback][NotifyThreadLoop]Event query timeout");
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(THREAD_SLEEP_DURATION_US));
    }

    ret = hrtEventDestroy(eventInfo.event);
    status = (ret != HCCL_SUCCESS || bTimeout) ? HCCL_E_RUNTIME : HCCL_SUCCESS;
    HCOM_EXECUTOR_ERR_BREAK(
        status,
        HCCL_ERROR("[Callback][NotifyThreadLoop]event execute failed, event destroy ret[%d], is timeout[%d s]", ret,
                   bTimeout),
        eventInfo.callback, parralMap_[queType].shutDown = true);
    // 执行成功，回调返回 SUCCESS
    eventInfo.callback(HCCL_SUCCESS);
    HCCL_RUN_INFO("[Callback][NotifyThreadLoop]hcom executor callback HCCL_SUCCESS, message type[%d]", queType);
  }

  CHK_RET(hrtResetDevice(deviceLogicId_));

  return status;
}

HcclResult HcomExecutor::EnqueueBroadcast(HcomOperation_t opInfo, StatusCallback callback) {
  // op信息合法性校验
  CHK_PTR_NULL(callback);
  CHK_PTR_NULL(opInfo.inputPtr);

  ExecutorMessage message;
  message.SetOperationMessage(opInfo);
  message.SetStatusCallback(callback);
  PushMessageToQueue(message, MsgQueueType::OPBASE_QUEUE);

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::EnqueueAllreduce(HcomOperation_t opInfo, StatusCallback callback) {
  // op信息合法性校验
  CHK_PTR_NULL(callback);
  CHK_PTR_NULL(opInfo.inputPtr);
  CHK_PTR_NULL(opInfo.outputPtr);

  ExecutorMessage message;
  message.SetOperationMessage(opInfo);
  message.SetStatusCallback(callback);
  PushMessageToQueue(message, MsgQueueType::OPBASE_QUEUE);

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::EnqueueReduceScatter(HcomOperation_t opInfo, StatusCallback callback) {
  // op信息合法性校验
  CHK_PTR_NULL(callback);
  CHK_PTR_NULL(opInfo.inputPtr);
  CHK_PTR_NULL(opInfo.outputPtr);

  ExecutorMessage message;
  message.SetOperationMessage(opInfo);
  message.SetStatusCallback(callback);
  PushMessageToQueue(message, MsgQueueType::OPBASE_QUEUE);

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::EnqueueAllGather(HcomOperation_t opInfo, StatusCallback callback) {
  // op信息合法性校验
  CHK_PTR_NULL(callback);
  CHK_PTR_NULL(opInfo.inputPtr);
  CHK_PTR_NULL(opInfo.outputPtr);

  ExecutorMessage message;
  message.SetOperationMessage(opInfo);
  message.SetStatusCallback(callback);
  PushMessageToQueue(message, MsgQueueType::OPBASE_QUEUE);

  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::EnqueueEvent(HcclRtStream stream, StatusCallback callback, MsgQueueType queType) {
  HcclResult ret;
  HcclRtEvent event = nullptr;
  CHK_RET(hrtEventCreate(&event));

  ret = hrtEventRecord(event, stream);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("[Enqueue][Event]rt_event_record failed, ret[%d]", ret);
    ret = hrtEventDestroy(event);
    event = nullptr;
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Enqueue][Event]rt_event_destroy failed, ret[%d]", ret), ret);
  }

  auto startTime = std::chrono::steady_clock::now();
  auto timeout = std::chrono::seconds(EVENT_QUERY_TIMEOUT_S);
  bool bTimeout = false;
  // 此处查询不能加锁，避免死锁
  while (parralMap_[queType].eventQueue.size() >= MAX_ENQUEUE_SIZE) {
    bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
    CHK_PRT_RET(bTimeout,
                HCCL_ERROR("[Enqueue][Event]Event enqueue timeout, size[%d]", parralMap_[queType].eventQueue.size()),
                HCCL_E_TIMEOUT);
    std::this_thread::sleep_for(std::chrono::microseconds(THREAD_SLEEP_DURATION_US));
  }

  // event以及回调信息入栈, 多线程加锁
  std::lock_guard<std::mutex> guard(eventMutex_);
  parralMap_[queType].eventQueue.push({event, callback});

  return HCCL_SUCCESS;
}

ExecutorMessage HcomExecutor::PopMessagesFromQueue(MsgQueueType queueType) {
  std::lock_guard<std::mutex> guard(msgMutex_);
  ExecutorMessage message = parralMap_[queueType].messageQueue.front();
  parralMap_[queueType].messageQueue.pop();

  return message;
}

EventInfo_t HcomExecutor::PopEventFromQueue(MsgQueueType queueType) {
  std::lock_guard<std::mutex> guard(eventMutex_);  // 多线程访问加锁
  EventInfo_t eventInfo = parralMap_[queueType].eventQueue.front();
  parralMap_[queueType].eventQueue.pop();

  return eventInfo;
}

void HcomExecutor::PushMessageToQueue(ExecutorMessage &message, MsgQueueType queueType) {
  std::unique_lock<std::mutex> lock(msgMutex_);
  parralMap_[queueType].messageQueue.push(std::move(message));
  lock.unlock();
}

HcclResult HcomExecutor::GetComm(const char *group, HcclComm *comm) {
  CHK_RET(HcomGetCommHandleByGroup(group, comm));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::HcomExecEnqueueAllToAllV(HcomAllToAllVParams opInfo, StatusCallback callback) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(InitHcclComm(opInfo.group, comm));
  CHK_SMART_PTR_NULL(parralMap_[MsgQueueType::OPBASE_QUEUE].stream);
  if (parralMap_[MsgQueueType::OPBASE_QUEUE].shutDown) {
    HCCL_ERROR("[Exec][EnqueueAlltoAll]Thread has been shut down, Enqueue AllToAll failed");
    return HCCL_E_INTERNAL;
  }

  CHK_PTR_NULL(opInfo.sendcounts);
  CHK_PTR_NULL(opInfo.sdispls);
  CHK_PTR_NULL(opInfo.rdispls);
  CHK_PTR_NULL(opInfo.recvcounts);

  ExecutorMessage message;
  message.SetAlltoAllMessage(opInfo);
  message.SetStatusCallback(callback);
  std::unique_lock<std::mutex> lock(msgMutex_);
  parralMap_[MsgQueueType::OPBASE_QUEUE].messageQueue.push(std::move(message));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::HcomExecEnqueueAllToAllVC(HcomAllToAllVCParams opInfo, StatusCallback callback) {
  // 获取通信域
  HcclComm comm;
  CHK_RET(GetComm(opInfo.group, &comm));
  CHK_RET(InitHcclComm(opInfo.group, comm));
  CHK_SMART_PTR_NULL(parralMap_[MsgQueueType::OPBASE_QUEUE].stream);
  if (parralMap_[MsgQueueType::OPBASE_QUEUE].shutDown) {
    HCCL_ERROR("[Exec][EnqueueAlltoAllVC]Thread has been shut down, Enqueue AllToAllVC failed");
    return HCCL_E_INTERNAL;
  }

  CHK_PTR_NULL(opInfo.sendcountmatrix);

  ExecutorMessage message;
  message.SetAlltoAllVCMessage(opInfo);
  message.SetStatusCallback(callback);
  std::unique_lock<std::mutex> lock(msgMutex_);
  parralMap_[MsgQueueType::OPBASE_QUEUE].messageQueue.push(std::move(message));
  return HCCL_SUCCESS;
}

HcclResult HcomExecutor::RunGather(std::vector<u64> &addrInfo, std::vector<u64> &addrInfoCountPerRank, u32 rankSize,
                                   u64 *sendCounts, u64 *sdispls, void *sendDevBuf, s32 addrLength) {
  u64 memSize = 0;
  u64 perThreadCount = addrInfo.size() / NUM_TWO / GATHER_THREAD_NUM;
  std::vector<u64> perThreadCounts(GATHER_THREAD_NUM, perThreadCount);
  perThreadCounts[GATHER_THREAD_NUM - 1] = addrInfo.size() / NUM_TWO - perThreadCount * (GATHER_THREAD_NUM - 1);
  std::vector<u64> offset(GATHER_THREAD_NUM, 0);
  if (addrLength == -1) {  // 数据包长度不一样的情况
    u32 offsetIndex = 0;
    for (u32 index = 1; index < addrInfo.size(); index += NUM_TWO) {  // 由于是二元组，单数为数据包的长度，每个循环+2
      /* 如果数据包数量小于线程数量则offset全置为0 */
      if (perThreadCount != 0) {
        /* 条件1：当累加的数量达到perThreadCount时往offset中填入累加值，即可计算出前面thread产生的offset值 */
        /* 条件2：由于第0个thread的offset为0，后面的线程的offset为前面线程处理数据量的累加，因此对最后一个值弃之不用 */
        if (index / NUM_TWO % perThreadCount == 0 && offsetIndex < GATHER_THREAD_NUM) {
          offset[offsetIndex] = memSize;
          offsetIndex++;
        }
      }
      memSize += addrInfo[index];
    }
  } else {
    memSize = addrInfo.size() / NUM_TWO * addrInfo[1];
    for (u32 index = 0; index < GATHER_THREAD_NUM; index++) {
      offset[index] = index * perThreadCount * addrInfo[1];
    }
  }

  // 多线程拷贝
  void *tmpHostMem = nullptr;
  HcclResult ret = hrtMallocHost(&tmpHostMem, memSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS || tmpHostMem == nullptr,
              HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
  AclHostMemPtr tmpHostMemPtr(tmpHostMem);

  std::vector<std::unique_ptr<std::thread>> threads(GATHER_THREAD_NUM);
  for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
    threads[num].reset(new (std::nothrow) std::thread(&HcomExecutor::GatherMemCopyThread, this, tmpHostMemPtr.get(),
                                                      offset[num], std::ref(addrInfo), num * perThreadCount * NUM_TWO,
                                                      perThreadCounts[num], memSize));
    CHK_PRT_RET(!threads[num],
                HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV]threads[%u] reset "
                           "failed ",
                           num),
                HCCL_E_INTERNAL);
  }

  // 构造入参
  CHK_PRT_RET(memset_s(sendCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64)) != EOK,
              HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed,count[%lld]", rankSize * sizeof(u64)),
              HCCL_E_SYSCALL);
  u64 prevNum = 0;
  u64 nextNum = 0;
  for (u32 index = 0; index < addrInfoCountPerRank.size(); index++) {
    nextNum += addrInfoCountPerRank[index];
    for (u64 i = NUM_TWO * prevNum; i < NUM_TWO * nextNum; i += NUM_TWO) {
      *(sendCounts + index) += addrInfo[i + 1];
    }
    prevNum = nextNum;
  }

  CHK_PRT_RET(memset_s(sdispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64)) != EOK,
              HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]", rankSize * sizeof(u64)),
              HCCL_E_SYSCALL);
  u64 displ = 0;
  for (u32 i = 0; i < rankSize; i++) {
    *(sdispls + i) = displ;
    displ += *(sendCounts + i);
  }

  // 等待线程执行完毕
  for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
    threads[num]->join();
  }

  CHK_RET(hrtMemSyncCopy(sendDevBuf, memSize, tmpHostMemPtr.get(), memSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
  return HCCL_SUCCESS;
}

void HcomExecutor::PushGatherAlltoAllParaToQue(HcomGatherAllToAllVParams &gatherParams, void *sendCounts, void *sdispls,
                                               void *recvCounts, void *rdispls,
                                               std::function<void(HcclResult status)> callback) {
  HcomAllToAllVParams opInfo;
  opInfo.sendbuf = gatherParams.gatheredbuf;
  opInfo.sendcounts = sendCounts;
  opInfo.sdispls = sdispls;
  opInfo.sendtype = HCCL_DATA_TYPE_INT8;
  opInfo.recvbuf = gatherParams.recvbuf;
  opInfo.recvcounts = recvCounts;
  opInfo.rdispls = rdispls;
  opInfo.recvtype = gatherParams.recvtype;
  opInfo.group = nullptr;

  ExecutorMessage message;
  message.SetAlltoAllMessage(opInfo, true);
  message.SetStatusCallback(callback);
  parralMap_[MsgQueueType::OPBASE_QUEUE].messageQueue.push(std::move(message));
}

void HcomExecutor::GatherMemCopyThread(void *baseAddr, u64 offset, std::vector<u64> &addrInfo, u64 beginIndex,
                                       u64 count, u64 tmpMemSize) {
  // 给当前线程添加名字
  SetThreadName("Hccl_GatherMemCopy");

  void *addr = nullptr;
  u64 length = 0;
  auto destMax = [tmpMemSize, offset]() -> u64 { return tmpMemSize < offset ? 0 : tmpMemSize - offset; };

  for (u32 index = 0; index < count; index++) {
    addr = reinterpret_cast<void *>(addrInfo[beginIndex + NUM_TWO * index]);
    length = addrInfo[beginIndex + index * NUM_TWO + 1];
    if (memcpy_s(static_cast<char *>(baseAddr) + offset, destMax(), addr, length) != EOK) {
      HCCL_ERROR("[MemCopy][GatherAlltoAllV] mem copy failed, destMax[%llu], count[%llu]", tmpMemSize - offset, length);
      return;
    }
    offset += length;
  }
}

HcclResult HcomExecutor::AllocGatherAlltoAllMem(u64 *&sendCounts, u64 *&sdispls, void *&recvCountsPtr,
                                                void *&rdisplsPtr, u32 rankSize) {
  HcclResult ret;
  do {
    sendCounts = new (std::nothrow) u64[rankSize];
    CHK_PRT_BREAK(sendCounts == nullptr, HCCL_ERROR("sendCounts new failed"), ret = HCCL_E_PTR);

    sdispls = new (std::nothrow) u64[rankSize];
    CHK_PRT_BREAK(sdispls == nullptr, HCCL_ERROR("sdispls new failed"), ret = HCCL_E_PTR);

    recvCountsPtr = new (std::nothrow) u64[rankSize];
    CHK_PRT_BREAK(recvCountsPtr == nullptr, HCCL_ERROR("recvCountsPtr new failed"), ret = HCCL_E_PTR);

    rdisplsPtr = new (std::nothrow) u64[rankSize];
    CHK_PRT_BREAK(rdisplsPtr == nullptr, HCCL_ERROR("rdisplsPtr new failed"), ret = HCCL_E_PTR);

    HCCL_DEBUG("[AllocGatherAlltoAllMem] new sendCounts, sdispls, recvCounts, rdispls success!");
    return HCCL_SUCCESS;
  } while (0);

  if (sendCounts != nullptr) {
    delete[] sendCounts;
    sendCounts = nullptr;
  }
  if (sdispls != nullptr) {
    delete[] sdispls;
    sdispls = nullptr;
  }
  if (recvCountsPtr != nullptr) {
    delete[] static_cast<u64 *>(recvCountsPtr);
    recvCountsPtr = nullptr;
  }
  if (rdisplsPtr != nullptr) {
    delete[] static_cast<u64 *>(rdisplsPtr);
    rdisplsPtr = nullptr;
  }
  return ret;
}

void HcomExecutor::DeleteGatherAlltoAllMem(HcomAllToAllVParams &params) {
  if (params.sendcounts != nullptr) {
    delete[] static_cast<u64 *>(params.sendcounts);
    params.sendcounts = nullptr;
  }
  if (params.sdispls != nullptr) {
    delete[] static_cast<u64 *>(params.sdispls);
    params.sdispls = nullptr;
  }
  if (params.recvcounts != nullptr) {
    delete[] static_cast<u64 *>(params.recvcounts);
    params.recvcounts = nullptr;
  }
  if (params.rdispls != nullptr) {
    delete[] static_cast<u64 *>(params.rdispls);
    params.rdispls = nullptr;
  }
}

void HcomExecutor::CleanQueueResources() {
  if (!parralMap_[MsgQueueType::OPBASE_QUEUE].messageQueue.empty()) {
    for (u32 index = 0; index < parralMap_[MsgQueueType::OPBASE_QUEUE].messageQueue.size(); index++) {
      ExecutorMessage message = PopMessagesFromQueue(MsgQueueType::OPBASE_QUEUE);
      if (message.isGatherAlltoAll_) {
        HcomAllToAllVParams opInfo = message.GetAlltoAllMessage();
        DeleteGatherAlltoAllMem(opInfo);
      }
    }
  }
  parralMap_.clear();
}

HcclResult HcomRegRemoteAccessMem(const MemRegisterAddr *addrList, u32 count) {
  HCCL_ERROR("[Reg][RemoteAccessMem] HcomRegRemoteAccessMem is not support.");
  return HCCL_E_NOT_SUPPORT;
}
}  // namespace hccl
