/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTER_DLHCCLFUNC_H
#define ADAPTER_DLHCCLFUNC_H

#include "dlhccl_function.h"
#include "hccl/hccl_types.h"
#include "hccl/hcom.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

HcclResult HcceAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                         HcclComm comm, aclrtStream stream);

HcclResult HcceBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream);

HcclResult HcceReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream);

HcclResult HcceReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf,
                              uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                              aclrtStream stream);

HcclResult HcceAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                         aclrtStream stream);

HcclResult HcceAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
                          const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream);

HcclResult HcceSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
                    aclrtStream stream);

HcclResult HcceRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
                    aclrtStream stream);

HcclResult HcceAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf,
                          HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult HcceAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream);

HcclResult HcceAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
                        uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult HcceReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult HcceGetandClearOverFlowTasks(const char *group, hccl::HcclDumpInfo **hcclDumpInfo, s32 *len);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ADAPTER_DLHCCL_FUNCTION_H
