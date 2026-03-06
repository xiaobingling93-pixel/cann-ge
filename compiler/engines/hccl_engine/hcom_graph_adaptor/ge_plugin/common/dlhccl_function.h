/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DLHCCL_FUNCTION_H
#define DLHCCL_FUNCTION_H

#include <mutex>
#include <dlfcn.h>
#include <functional>
#include "hccl/hccl_types.h"
#include "hccl/hcom.h"
#include "hcom_log.h"

using aclrtStream = void *;

class DlHcclFunction {
 public:
  static DlHcclFunction &get_instance();
  HcclResult init();
  void deinit();

  HcclResult dlHcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream);

  HcclResult dlHcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                             aclrtStream stream);

  HcclResult dlHcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                 HcclReduceOp op, HcclComm comm, aclrtStream stream);

  HcclResult dlHcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf,
                                  uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                                  aclrtStream stream);

  HcclResult dlHcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                             aclrtStream stream);

  HcclResult dlHcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
                              const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream);

  HcclResult dlHcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
                        aclrtStream stream);

  HcclResult dlHcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
                        aclrtStream stream);

  HcclResult dlHcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                              const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream);

  HcclResult dlHcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                             const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                             HcclComm comm, aclrtStream stream);

  HcclResult dlHcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
                            uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream);

  HcclResult dlHcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                          uint32_t root, HcclComm comm, aclrtStream stream);

  HcclResult dlHcomGetandClearOverFlowTasks(const char *group, hccl::HcclDumpInfo **hcclDumpInfoPtr, s32 *len);

 private:
  DlHcclFunction();
  ~DlHcclFunction();
  DlHcclFunction(const DlHcclFunction &) = delete;
  DlHcclFunction &operator=(const DlHcclFunction &) = delete;

  void *dl_hccl_handle;
  void *dl_hcomm_handle;
  std::mutex handleMutex_;

  std::function<HcclResult(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                           aclrtStream stream)>
      dlHcclAllGatherFunc;

  std::function<HcclResult(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
                           const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)>
      dlHcclAllGatherVFunc;

  std::function<HcclResult(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                           HcclComm comm, aclrtStream stream)>
      dlHcclAllReduceFunc;

  std::function<HcclResult(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                           aclrtStream stream)>
      dlHcclBroadcastFunc;

  std::function<HcclResult(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                           HcclComm comm, aclrtStream stream)>
      dlHcclReduceScatterFunc;

  std::function<HcclResult(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf,
                           uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                           aclrtStream stream)>
      dlHcclReduceScatterVFunc;

  std::function<HcclResult(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                           const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                           HcclComm comm, aclrtStream stream)>
      dlHcclAlltoAllVFunc;

  std::function<HcclResult(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf,
                           HcclDataType recvType, HcclComm comm, aclrtStream stream)>
      dlHcclAlltoAllVCFunc;

  std::function<HcclResult(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
                           uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)>
      dlHcclAlltoAllFunc;

  std::function<HcclResult(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                           uint32_t root, HcclComm comm, aclrtStream stream)>
      dlHcclReduceFunc;

  std::function<HcclResult(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
                           aclrtStream stream)>
      dlHcclSendFunc;

  std::function<HcclResult(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
                           aclrtStream stream)>
      dlHcclRecvFunc;

  std::function<HcclResult(const char *group, hccl::HcclDumpInfo **hcclDumpInfoPtr, s32 *len)>
      dlHcomGetandClearOverFlowTasksFunc;
};

#endif
