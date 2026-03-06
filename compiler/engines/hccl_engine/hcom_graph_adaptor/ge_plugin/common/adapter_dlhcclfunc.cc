/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_dlhcclfunc.h"

HcclResult HcceAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                         HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);
}

HcclResult HcceBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclBroadcast(buf, count, dataType, root, comm, stream);
}

HcclResult HcceReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
                             HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult HcceReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf,
                              uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                              aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclReduceScatterV(sendBuf, sendCounts, sendDispls, recvBuf, recvCount,
                                                             dataType, op, comm, stream);
}

HcclResult HcceAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                         aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

HcclResult HcceAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
                          const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclAllGatherV(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType,
                                                         comm, stream);
}

HcclResult HcceSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
                    aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclSend(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult HcceRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
                    aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclRecv(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult HcceAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf,
                          HcclDataType recvType, HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclAlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm,
                                                         stream);
}

HcclResult HcceAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclAlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts,
                                                        rdispls, recvType, comm, stream);
}

HcclResult HcceAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
                        uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm,
                                                       stream);
}

HcclResult HcceReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcclReduce(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
}

HcclResult HcceGetandClearOverFlowTasks(const char *group, hccl::HcclDumpInfo **hcclDumpInfo, s32 *len) {
  CHK_PRT_RET(DlHcclFunction::get_instance().init() != HCCL_SUCCESS, HCCL_ERROR("DlHcclFunction::get_instance().init() fail \n"),
              HCCL_E_PARA);

  return DlHcclFunction::get_instance().dlHcomGetandClearOverFlowTasks(group, hcclDumpInfo, len);
}
