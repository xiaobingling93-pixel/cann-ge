/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_ops_kernel_info_store.h"
#include <securec.h>
#include <functional>
#include <nlohmann/json.hpp>
#include <algorithm>
#include "graph/tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/memory/memory_api.h"
#include "framework/common/ge_types.h"  // ge对外options
#include "graph/types.h"
#include "hcom.h"
#include "hcom_graph_optimizer.h"
#include "graph/ge_local_context.h"
#include "hccl/hccl_ex.h"
#include "hccl/base.h"
#include "adump_api.h"  // 工具dump开关
#include "hcom/hcom_topo_info.h"
#include "offline_build_config_parse.h"
#include "hcom_op_utils.h"
#include "adapter_dlhcclfunc.h"

using namespace std;

namespace hccl {
constexpr s32 ALIGNMENT = 32;
constexpr s32 MAX_BLOCK_DIM = 48;

HcomOpsKernelInfoStore::HcomOpsKernelInfoStore() : workSpaceMemInfo_(), initCrackMem_(false) {}

HcomOpsKernelInfoStore::~HcomOpsKernelInfoStore() {}

bool HcomOpsKernelInfoStore::UpdateGraphIdGroupMap(std::string group, u32 graphId) {
  if (graphIdByGroup_.find(group) == graphIdByGroup_.end() || graphIdByGroup_[group] != graphId) {
    graphIdByGroup_[group] = graphId;
    return true;
  }
  return false;
}

HcclResult HcomOpsKernelInfoStore::SetCustomKernelInfo(ge::OpInfo &opinfo, std::map<string, ge::OpInfo> &infos) const {
  opinfo.opKernelLib = HCCL_OPS_LIB_NAME;
  for (u32 index = 0; index < HCOM_SUPPORTED_OP_TYPE.size(); index++) {
    HCCL_INFO("op[%s]: engine[%s] opKernelLib[%s] computeCost[%d] flagPartial[%d] flagAsync[%d]",
              HCOM_SUPPORTED_OP_TYPE[index].c_str(), opinfo.engine.c_str(), opinfo.opKernelLib.c_str(),
              opinfo.computeCost, opinfo.flagPartial, opinfo.flagAsync);
    // Allreduce 需要设定atomic标志位，其余算子不需要
    if (HCOM_SUPPORTED_OP_TYPE[index] == HCCL_KERNEL_OP_TYPE_ALLREDUCE ||
        HCOM_SUPPORTED_OP_TYPE[index] == HCCL_KERNEL_OP_TYPE_REDUCE) {
      opinfo.isAtomic = true;
    } else {
      opinfo.isAtomic = false;
    }
    infos.insert(std::pair<string, ge::OpInfo>(HCOM_SUPPORTED_OP_TYPE[index], opinfo));
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetSupportedOP(std::vector<std::string> &hcclSupportOp) const {
  hcclSupportOp.assign(HCOM_SUPPORTED_OP_TYPE.begin(), HCOM_SUPPORTED_OP_TYPE.end());
  return HCCL_SUCCESS;
}

bool HcomOpsKernelInfoStore::IsOpTypeCCLTag(const std::string &opType) {
  return (opType == HCCL_KERNEL_OP_TYPE_BROADCAST || opType == HCCL_KERNEL_OP_TYPE_ALLREDUCE ||
          opType == HCCL_KERNEL_OP_TYPE_ALLGATHER || opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER ||
          opType == HCCL_KERNEL_OP_TYPE_REDUCE || opType == HCCL_KERNEL_OP_TYPE_ALLTOALLV ||
          opType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC || opType == HCCL_KERNEL_OP_TYPE_ALLTOALL ||
          opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV || opType == HCCL_KERNEL_OP_TYPE_ALLGATHERV);
}

// ATTENTION: this function is NOT reenterable program!!!
HcclResult HcomOpsKernelInfoStore::GenerateOpTagFromTaskInfo(const ge::GETaskInfo &task, const std::string &opType,
                                                             std::string &sTag, u32 &loopMaxTime) {
  uint32_t srcRank = 0;
  uint32_t destRank = 0;

  std::string group;
  std::string identifier;
  int64_t comm = 0;
  HcclResult ret;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    ret = GetGroupFromTaskInfo(task, group);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Generate][OpTag]op[%s]: get comm and group failed. ret[%d]", opType.c_str(), ret), ret);
  }

  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  size_t nodeNameHash = privateDefBuf->nodeNameHash;
  uint32_t graphId = privateDefBuf->graphId;

  if (IsOpTypeCCLTag(opType)) {
    // Broadcast / AllReduce / AllGather / ReduceScatter 算子的 tag: op type + group name hash + op index in group.
    char cTag[CCL_OP_TAG_MAX_LEN];
    CHK_RET(HcomGenerateCclOpTag(opType.c_str(), comm, group.c_str(), cTag));
    sTag = cTag;
  } else if (opType == HCCL_KERNEL_OP_TYPE_SEND) {
    // Send/Receive 算子的 tag 为 group + sr_tag + src_rank + dest_rank
    uint32_t srTag = privateDefBuf->srTag;
    std::string sSrTag = std::to_string(srTag);
    destRank = privateDefBuf->destRank;
    std::string sDestRank = std::to_string(destRank);
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      ret = HcomGetRankId(group.c_str(), &srcRank);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Generate][OpTag]op[%s]: get rank id failed. ret[%d]", opType.c_str(), ret), ret);
      std::string sSrcRank = std::to_string(srcRank);
      sTag = group + "_" + sSrTag + "_" + sSrcRank + "_" + sDestRank;
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      ret = HcomGetRankId(groupname, &srcRank);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Generate][OpTag]op[%s]: get rank id failed. ret[%d]", opType.c_str(), ret), ret);
      std::string sSrcRank = std::to_string(srcRank);
      identifier = std::string(groupname);
      sTag = identifier + "_" + sSrTag + "_" + sSrcRank + "_" + sDestRank;
    }
  } else if (opType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    uint32_t srTag = privateDefBuf->srTag;
    std::string sSrTag = std::to_string(srTag);
    srcRank = privateDefBuf->srcRank;
    std::string sSrcRank = std::to_string(srcRank);
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      ret = HcomGetRankId(group.c_str(), &destRank);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Generate][OpTag]op[%s]: get rank id failed. ret[%d]", opType.c_str(), ret), ret);
      std::string sDestRank = std::to_string(destRank);
      sTag = group + "_" + sSrTag + "_" + sSrcRank + "_" + sDestRank;
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      ret = HcomGetRankId(groupname, &destRank);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Generate][OpTag]op[%s]: get rank id failed. ret[%d]", opType.c_str(), ret), ret);
      std::string sDestRank = std::to_string(destRank);
      identifier = std::string(groupname);
      sTag = identifier + "_" + sSrTag + "_" + sSrcRank + "_" + sDestRank;
    }
  } else {
    HCCL_ERROR("[Generate][OpTag]errNo[0x%016llx] get tag name failed. op type[%s] is invalid.",
               HCOM_ERROR_CODE(HCCL_E_PARA), opType.c_str());
    return HCCL_E_PARA;
  }

  HCCL_RUN_INFO("GenerateOpTag: graph[%u] opType[%s], nodeNameHash[%zu], group[%s], generated op tag[%s]", graphId,
                opType.c_str(), nodeNameHash, group.c_str(), sTag.c_str());

  if (task.needRefresh) {
    CHK_RET(GetOpKernelLoopTime(task, opType, sTag, loopMaxTime));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetOpKernelLoopTime(const ge::GETaskInfo &task, const std::string &opType,
                                                       std::string &sTag, u32 &loopMaxTime) {
  std::string group;
  int64_t comm = 0;
  u64 count;
  HcclDataType dataType;
  u32 shapeType;

  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[GatherOp][Kernel]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  u32 unitSize = SIZE_TABLE[dataType];

  if ((opType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) || (opType == HCCL_KERNEL_OP_TYPE_BROADCAST) ||
      (opType == HCCL_KERNEL_OP_TYPE_REDUCE) || (opType == HCCL_KERNEL_OP_TYPE_SEND)) {
    // 获取 in ccl buf
    u64 commInputSize;
    CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

    loopMaxTime = ((count * unitSize) + commInputSize - 1) / commInputSize;
  }
  if (opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
    // 获取 in ccl buf
    u64 commInputSize;
    CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

    u32 rankSize = 0;
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomGetRankSize(groupname, &rankSize));
    }

    loopMaxTime = ((count * unitSize * rankSize) + commInputSize - 1) / commInputSize;
  }
  if (opType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    // 获取 out ccl buf
    u64 commOutputSize;
    GetHcomOutCCLbufferSize(commOutputSize, shapeType, comm, group);

    loopMaxTime = ((count * unitSize) + commOutputSize - 1) / commOutputSize;
  }
  if (opType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
    // 获取 out ccl buf
    u64 commOutputSize;
    GetHcomOutCCLbufferSize(commOutputSize, shapeType, comm, group);

    u32 rankSize = 0;
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomGetRankSize(groupname, &rankSize));
    }

    loopMaxTime = ((count * unitSize * rankSize) + commOutputSize - 1) / commOutputSize;
  }

  HCCL_INFO("GetOpKernelLoopTime: opType[%s], generated op tag[%s], loopMaxTime[%u]", opType.c_str(), sTag.c_str(),
            loopMaxTime);

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetOriginalGraphShapeTypeFromTaskInfo(const ge::GETaskInfo &task, u32 &shapeType) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  shapeType = privateDefBuf->originalGraphShapeType;
  HCCL_INFO("get shapeType[%u] from task info success.", shapeType);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetGroupFromTaskInfo(const ge::GETaskInfo &task, std::string &sGroup) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  sGroup = reinterpret_cast<const char *>(privateDefBuf->group);
  HCCL_INFO("get group[%s] from task info success.", sGroup.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetDestRankFromTaskInfo(const ge::GETaskInfo &task, u32 &destRank) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  destRank = privateDefBuf->destRank;
  HCCL_INFO("get dest rank[%u] from task info success.", destRank);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetSrcRankFromTaskInfo(const ge::GETaskInfo &task, u32 &srcRank) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  srcRank = privateDefBuf->srcRank;
  HCCL_INFO("get src rank[%u] from task info success.", srcRank);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetSrTagFromTaskInfo(const ge::GETaskInfo &task, u32 &srTag) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  srTag = privateDefBuf->srTag;
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetCommFromTaskInfo(const ge::GETaskInfo &task, int64_t &comm) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  comm = privateDefBuf->comm;
  HCCL_INFO("get COMM[%lld] from task info success.", comm);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CheckPrivateDef(const ge::GETaskInfo &task) {
  CHK_PTR_NULL(task.privateDef);
  if (task.privateDef == nullptr) {
    HCCL_ERROR("[Check][PrivateDef]errNo[0x%016llx] privateDefLen[%u] is not equal to [%zu]",
               HCOM_ERROR_CODE(HCCL_E_PARA), task.privateDefLen, sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    return HCCL_E_PARA;
  }
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  size_t crackNum = privateDefBuf->tensorNum;
  // 表示有tensor间脏数据需要清零
  if (crackNum != 0) {
    uint32_t privateDefLenwithCrackClear =
        sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + crackNum * sizeof(int64_t) + crackNum * sizeof(int64_t);
    if (task.privateDefLen != privateDefLenwithCrackClear) {
      HCCL_ERROR("[Check][PrivateDef]errNo[0x%016llx] privateDefLen[%u] is not equal to [%zu]",
                 HCOM_ERROR_CODE(HCCL_E_PARA), task.privateDefLen, privateDefLenwithCrackClear);
      return HCCL_E_PARA;
    }
  } else {  // 表示无tensor间脏数据需要清零
    uint32_t privateDefLen = 0;
    if (task.kernelHcclInfo[0].hccl_type == HCCL_KERNEL_OP_TYPE_ALLTOALLV ||
        task.kernelHcclInfo[0].hccl_type == HCCL_KERNEL_OP_TYPE_ALLTOALLVC ||
        task.kernelHcclInfo[0].hccl_type == HCCL_KERNEL_OP_TYPE_ALLTOALL) {
      privateDefLen = sizeof(HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF);
    } else if (task.kernelHcclInfo[0].hccl_type == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
      privateDefLen = sizeof(HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF);
    } else if (task.kernelHcclInfo[0].hccl_type == HCCL_KERNEL_OP_TYPE_ALLGATHERV) {
      privateDefLen = sizeof(HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF);
    } else {
      privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    }
    if (task.privateDefLen != privateDefLen) {
      HCCL_ERROR("[Check][PrivateDef]errNo[0x%016llx] privateDefLen[%u] is not equal to [%zu]",
                 HCOM_ERROR_CODE(HCCL_E_PARA), task.privateDefLen, privateDefLen);
      return HCCL_E_PARA;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CheckCommunicatorValidity(const char *group, const ge::GETaskInfo &task) {
  int64_t comm = static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT);
  CHK_RET(GetCommFromTaskInfo(task, comm));  // pytorch 图模式通信域是否可用
  bool isHcomInit = false;
  CHK_RET(HcomGetInitStatus(&isHcomInit));  // TF 图模式通信域是否可用
  bool isCommunicatorValid =
      (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT) && !isHcomInit) ? false : true;
  // 图模式Load Task时，如果非Pytorch图模式场景，并且pComm无效，则判定为未配置ranktable情况
  HcclComm commHandle;
  if (!isCommunicatorValid && (HcomGetCommHandleByGroup(group, &commHandle) != HCCL_SUCCESS)) {
    REPORT_PREDEFINED_ERR_MSG(
        "EI0004", std::vector<const char *>({"ranktable_path", "error_reason"}),
        std::vector<const char *>({ "The ranktable path configured in the training can be found in the plogs.",
        "The rankTable file path does not exist, the permission is insufficient, or the JSON format is incorrect."}));
    HCCL_ERROR(
        "[%s][%s]No valid communicator found, please check the RankTable Path or Master Info config or hccl "
        "initialization has been called before call this function",
        LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str());
    return HCCL_E_PARA;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HCCLOpsKernel(const ge::GETaskInfo &task, const std::string &sCollectiveType,
                                                 const std::vector<std::string> &tagVec) {
  CHK_RET(kernelFuncTable_.at(sCollectiveType)(task, tagVec));
  return HCCL_SUCCESS;
}

void HcomOpsKernelInfoStore::GetAlltoAllVCParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, void *&sendCountMatrix,
                                                 HcclDataType &sendType, uintptr_t &recvBuf, HcclDataType &recvType) {
  HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF *privateDefPtr =
      reinterpret_cast<HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  sendCountMatrix = privateDefPtr->cparamsInfo.sendCountMatrix;
  sendType = privateDefPtr->cparamsInfo.sendType;
  recvType = privateDefPtr->cparamsInfo.recvType;
  if (!task.kernelHcclInfo.empty()) {
    sendBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].inputDataAddr);
    recvBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].outputDataAddr);
  }
}

void HcomOpsKernelInfoStore::GetAlltoAllVParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, void *&sendCounts,
                                                void *&sendDispls, uintptr_t &recvBuf, void *&recvCounts,
                                                void *&recvDispls, HcclDataType &sendType, HcclDataType &recvType) {
  HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF *privateDefPtr =
      reinterpret_cast<HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  sendCounts = privateDefPtr->paramsInfo.sendCounts;
  sendDispls = privateDefPtr->paramsInfo.sendDispls;
  recvCounts = privateDefPtr->paramsInfo.recvCounts;
  recvDispls = privateDefPtr->paramsInfo.recvDispls;
  sendType = privateDefPtr->paramsInfo.sendType;
  recvType = privateDefPtr->paramsInfo.recvType;
  if (!task.kernelHcclInfo.empty()) {
    sendBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].inputDataAddr);
    recvBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].outputDataAddr);
  }
}

void HcomOpsKernelInfoStore::GetReduceScatterVParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, void *&sendCounts,
                                                     void *&sendDispls, uintptr_t &recvBuf, int64_t &recvCount) {
  HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF *privateDefPtr =
      reinterpret_cast<HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  sendCounts = privateDefPtr->paramsInfo.sendCounts;
  sendDispls = privateDefPtr->paramsInfo.sendDispls;
  recvCount = privateDefPtr->paramsInfo.recvCounts[0];
  if (!task.kernelHcclInfo.empty()) {
    sendBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].inputDataAddr);
    recvBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].outputDataAddr);
  }
}

void HcomOpsKernelInfoStore::GetAllGatherVParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, int64_t &sendCount,
                                                 uintptr_t &recvBuf, void *&recvCounts, void *&recvDispls) {
  HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF *privateDefPtr =
      reinterpret_cast<HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  sendCount = privateDefPtr->paramsInfo.sendCount[0];
  recvCounts = privateDefPtr->paramsInfo.recvCounts;
  recvDispls = privateDefPtr->paramsInfo.recvDispls;
  if (!task.kernelHcclInfo.empty()) {
    sendBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].inputDataAddr);
    recvBuf = reinterpret_cast<uintptr_t>(task.kernelHcclInfo[0].outputDataAddr);
  }
}

HcclResult HcomOpsKernelInfoStore::HcomAlltoAllVOpKernel(const ge::GETaskInfo &task,
                                                         const std::vector<std::string> &tagVec) {
  CHK_PRT_RET((task.kernelHcclInfo.size() != 1),
              HCCL_ERROR("[AlltoAllVOp][Kernel]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  rtStream_t stream;
  CHK_RET(GetStreamMainFromTaskInfo(task, stream));

  uintptr_t sendBuf = 0;
  void *sendCounts = nullptr;
  void *sendDispls = nullptr;
  uintptr_t recvBuf = 0;
  void *recvCounts = nullptr;
  void *recvDispls = nullptr;
  HcclDataType sendType;
  HcclDataType recvType;

  GetAlltoAllVParams(task, sendBuf, sendCounts, sendDispls, recvBuf, recvCounts, recvDispls, sendType, recvType);

  std::string group;
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  u32 shapeType;
  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  void *sendBufPtr = reinterpret_cast<void *>(sendBuf);
  void *recvBufPtr = reinterpret_cast<void *>(recvBuf);
  u64 inputMemSize = 0;
  u64 outputMemSize = 0;

  CHK_RET(GetHcomAlltoallVOpMemSize(shapeType, HCCL_KERNEL_OP_TYPE_ALLTOALLV, comm, group, sendType, recvType,
                                    sendCounts, sendDispls, recvCounts, recvDispls, inputMemSize, outputMemSize));

  CHK_RET(RefreshInputAddr(shapeType, comm, group, reinterpret_cast<void *>(sendBuf), inputMemSize, stream));

  if (shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE) {
    CHK_RET(GetCommCCLBuf(shapeType, comm, group, sendBufPtr, recvBufPtr));
  }

  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomAlltoAllV(sendBufPtr, sendCounts, sendDispls, sendType, recvBufPtr, recvCounts, recvDispls, recvType,
                          group.c_str(), stream, tagVec[0].c_str()));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomAlltoAllV(sendBufPtr, sendCounts, sendDispls, sendType, recvBufPtr, recvCounts, recvDispls, recvType,
                          groupname, stream, tagVec[0].c_str()));
    HCCL_DEBUG("[HcclCommGraph][Type]AlltoAllVOpKernel.");
  }

  CHK_RET(RefreshOutputAddr(shapeType, comm, group, reinterpret_cast<void *>(recvBuf), outputMemSize, stream));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAlltoAllOpKernel(const ge::GETaskInfo &task,
                                                        const std::vector<std::string> &tagVec) {
  CHK_PRT_RET((task.kernelHcclInfo.size() != 1),
              HCCL_ERROR("[AlltoAllOp][Kernel]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  std::string group;
  CHK_RET(GetGroupFromTaskInfo(task, group));

  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  u32 rankSize = 0;
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
  }

  void *sendBuf = task.kernelHcclInfo[0].inputDataAddr;
  void *recvBuf = task.kernelHcclInfo[0].outputDataAddr;
  u64 sendCount = task.kernelHcclInfo[0].count / rankSize;
  u64 recvCount = task.kernelHcclInfo[0].count / rankSize;
  HcclDataType sendType;
  CHK_RET(GetDataTypeFromTaskInfo(task, sendType));
  HcclDataType recvType;
  CHK_RET(GetDataTypeFromTaskInfo(task, recvType));
  rtStream_t stream;
  CHK_RET(GetStreamMainFromTaskInfo(task, stream));

  HCCL_DEBUG("[AlltoAllOp][Kernel] totalCount[%llu] rankSize[%u] sendCount[%llu] recvCount[%llu]",
             task.kernelHcclInfo[0].count, rankSize, sendCount, recvCount);
  CHK_RET(HcomAllToAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, group.c_str(), stream,
                       tagVec[0].c_str()));
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAlltoAllVCOpKernel(const ge::GETaskInfo &task,
                                                          const std::vector<std::string> &tagVec) {
  CHK_PRT_RET((task.kernelHcclInfo.size() != 1),
              HCCL_ERROR("[AlltoAllVCOp][Kernel]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  rtStream_t stream;
  CHK_RET(GetStreamMainFromTaskInfo(task, stream));

  uintptr_t sendBuf = 0;
  uintptr_t recvBuf = 0;
  void *sendCountMatrix = nullptr;
  HcclDataType sendType;
  HcclDataType recvType;
  GetAlltoAllVCParams(task, sendBuf, sendCountMatrix, sendType, recvBuf, recvType);

  std::string group;
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  u32 shapeType;
  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  void *sendBufPtr = reinterpret_cast<void *>(sendBuf);
  void *recvBufPtr = reinterpret_cast<void *>(recvBuf);
  u64 inputMemSize = 0;
  u64 outputMemSize = 0;

  CHK_RET(GetHcomAlltoallVCOpMemSize(shapeType, HCCL_KERNEL_OP_TYPE_ALLTOALLVC, comm, group, sendType, recvType,
                                     sendCountMatrix, inputMemSize, outputMemSize));

  CHK_RET(RefreshInputAddr(shapeType, comm, group, reinterpret_cast<void *>(sendBuf), inputMemSize, stream));

  if (shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE) {
    CHK_RET(GetCommCCLBuf(shapeType, comm, group, sendBufPtr, recvBufPtr));
  }

  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomAlltoAllVC(sendBufPtr, sendCountMatrix, sendType, recvBufPtr, recvType, group.c_str(), stream,
                           tagVec[0].c_str()));
  } else {
    char *sGroup = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &sGroup));
    CHK_RET(
        HcomAlltoAllVC(sendBufPtr, sendCountMatrix, sendType, recvBufPtr, recvType, sGroup, stream, tagVec[0].c_str()));
    HCCL_DEBUG("[HcclCommGraph][Type]AlltoAllVCOpKernel.");
  }

  CHK_RET(RefreshOutputAddr(shapeType, comm, group, reinterpret_cast<void *>(recvBuf), outputMemSize, stream));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CleanIntervalMemoryOpKernel(const ge::GETaskInfo &task, const std::string &tag,
                                                               uintptr_t inputAddr, u64 inputOffset, rtStream_t stream,
                                                               HcclCMDType opType) {
  // 在执行hcom算子前，执行请零task
  constexpr const char *kCleanSeparately = "1";
  std::string atomic_clean_policy;
  u64 count;
  std::string group;
  HcclDataType dataType;
  HcclReduceOp reduceType;
  bool ifAiv;
  char algName[ALG_NAME_MAX_LEN];
  int64_t comm;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[AllReduceOp][Kernel]errNo[0x%016llx] GETaskInfo size in HCOM"
                         "should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  bool need_clean_separately =
      (ge::GetThreadLocalContext().GetOption(ge::ATOMIC_CLEAN_POLICY, atomic_clean_policy) == ge::GRAPH_SUCCESS) &&
      (atomic_clean_policy == kCleanSeparately);
  if (need_clean_separately) {
#ifdef HCCD
    HCCL_ERROR("[CleanIntervalMemoryOpKernel] does not support this interface.");
    return HCCL_E_PARA;
#endif

    HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
    size_t crackNum = privateDefBuf->tensorNum;

    CHK_RET(GetCommFromTaskInfo(task, comm));
    CHK_RET(GetGroupFromTaskInfo(task, group));
    CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
    CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
    CHK_RET(GetReduceTypeFromTaskInfo(hcclInfo, reduceType));
    CHK_RET(HcomSelectAlg(comm, group.c_str(), count, nullptr, dataType, reduceType, opType, MAX_BLOCK_DIM, ifAiv, algName));

    if (crackNum == 0 || (crackNum == 1 && ifAiv)) {
      HCCL_WARNING("[CleanIntervalMemoryOpKernel]The number of tensors to be cleared is 0.");
      return HCCL_SUCCESS;
    }

    size_t privateDefBufSize = privateDefBuf->privateDefSize;
    std::vector<std::int64_t> crackAddr(crackNum);
    std::vector<std::int64_t> crackSize(crackNum);

    // 获取缝隙的地址和大小
    CHK_RET(GetCrackParamsInfoFromTaskInfo(task, inputAddr, crackAddr, crackSize, crackNum, privateDefBufSize,
                                           inputOffset));
    for (size_t i = 0; i < crackNum; i++) {
      HCCL_INFO("[GetCrackParams] crackAddr[%d] %lld crackSize[%d] %lld.", i, crackAddr[i], i, crackSize[i]);
    }

    // 下发缝隙清0任务
    CHK_RET(CleanIntervalMemory(tag.c_str(), crackAddr, crackSize, stream));
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetCrackParamsInfoFromTaskInfo(const ge::GETaskInfo &task, uintptr_t &inputAddr,
                                                                  std::vector<std::int64_t> &crackAddr,
                                                                  std::vector<std::int64_t> &crackSize, size_t crackNum,
                                                                  size_t privateDefBufSize, u64 inputOffset) {
  // 得到缝隙的offset和size
  std::vector<std::int64_t> crackOffset(crackNum);
  CHK_SAFETY_FUNC_RET(memcpy_s(crackOffset.data(), crackNum * sizeof(int64_t),
                               reinterpret_cast<int64_t *>(static_cast<s8 *>(task.privateDef) + privateDefBufSize),
                               crackNum * sizeof(int64_t)));

  CHK_SAFETY_FUNC_RET(memcpy_s(
      crackSize.data(), crackNum * sizeof(int64_t),
      reinterpret_cast<int64_t *>(static_cast<s8 *>(task.privateDef) + privateDefBufSize + crackNum * sizeof(int64_t)),
      crackNum * sizeof(int64_t)));

  // 根据offset+inputadddr得到缝隙的地址
  for (size_t i = 0; i < crackNum; i++) {
    crackAddr[i] = crackOffset[i] - inputOffset + inputAddr;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CleanIntervalMemory(const char *tag, std::vector<std::int64_t> &crackAddr,
                                                       std::vector<std::int64_t> &crackSize, rtStream_t stream) {
  std::string strTag = (tag == nullptr) ? "" : tag;
  HCCL_DEBUG("[CleanIntervalMemory] tag[%s]", strTag.c_str());
  DevType devType = HcomGetDeviceType();
  HCCL_DEBUG("[CleanIntervalMemory][HcomGetDeviceType]devType is %d", devType);

#ifndef OPEN_BUILD_PROJECT
  if (devType == DevType::DEV_TYPE_950) {
    // A5适配
    return CleanInterMemoryV2(crackSize, crackAddr, stream);
  }
#endif
  return CleanInterMemory(crackAddr, crackSize, stream);
}

HcclResult HcomOpsKernelInfoStore::TbeCleanIntervalMemory(std::vector<std::int64_t> &crackAddr,
                                                          std::vector<std::int64_t> &crackSize, rtStream_t stream) {
  HCCL_DEBUG("Enter--para: crackAddr[%p], crackSize[%p], crackAddr.size[%d], crackSize.size[%d].", crackAddr.data(),
             crackSize.data(), crackAddr.size(), crackSize.size());

  CHK_PTR_NULL(crackAddr.data());
  CHK_PTR_NULL(crackSize.data());
  CHK_PTR_NULL(stream);

  if (crackAddr.size() == 0 || crackSize.size() == 0) {
    HCCL_WARNING("[TbeCleanIntervalMemory] TensorNum is 0.");
    return HCCL_SUCCESS;
  }

#ifndef HCCD
  s32 deviceLogicId;  // 防止编译阶段和加载阶段deviceLogicId变更，此处重新刷一下
  CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
  HcclResult ret = HcomTbeMemClean(crackAddr.data(), crackSize.data(), crackAddr.size(), stream, deviceLogicId);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[TbeCrackClearedAsync]errNo[0x%016llx] tbe crack cleared fail,return[%d]. "
                         "para: crackAddr[%p], crackSize[%p], stream[%p].",
                         HCCL_ERROR_CODE(ret), ret, crackAddr.data(), crackSize.data(), stream),
              ret);
#else
  HCCL_ERROR("[CleanIntervalMemoryOpKernel] does not support this interface.");
  return HCCL_E_PARA;
#endif

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAllReduceOpKernel(const ge::GETaskInfo &task,
                                                         const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;
  HcclReduceOp reduceType;
  u32 shapeType;
  std::vector<void *> globalWorkSpaceAddr;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[AllReduceOp][Kernel]errNo[0x%016llx] GETaskInfo size in HCOM"
                         "should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));

  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));

  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));

  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));

  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));

  CHK_RET(GetReduceTypeFromTaskInfo(hcclInfo, reduceType));
  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  // 获取溢出检测内存地址
  CHK_RET(GetGlobalWorkSpaceAddrFromTaskInfo(hcclInfo, globalWorkSpaceAddr));

  CHK_RET(SetGlobalWorkSpace(comm, group, globalWorkSpaceAddr));
  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);
  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  if (task.needRefresh) {
    CHK_RET(HcomAllReduceLoop(task, tagVec, shapeType, comm, group, inputDataPtr, outputDataPtr, count, dataType,
                              reduceType, streamMain));
  } else {
    CHK_RET(CleanIntervalMemoryOpKernel(task, tagVec[0], reinterpret_cast<uintptr_t>(inputDataPtr), 0, streamMain,
                                        HcclCMDType::HCCL_CMD_ALLREDUCE));

    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomAllReduce(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType, group.c_str(),
                            streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomAllReduce(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType, groupname,
                            streamMain));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CheckTensorNumAndTensorSize(const ge::GETaskInfo &task, u64 count, u32 unitSize,
                                                               u64 commInputSize) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  size_t tensorNum = privateDefBuf->tensorNum;
  if (tensorNum > 1 && ((count * unitSize) > commInputSize)) {
    HCCL_WARNING("tensorNum is [%d] UserMemSize is [%llu]  cclbufferSize[%llu]", tensorNum, count * unitSize,
                 commInputSize);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CheckHcomOpMemSize(DevType deviceType, u64 countLeft, u32 unitSize,
                                                      u64 cclBufferSize) {
  if (deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910) {
    // 用户内存大于ccl buf时返回错误
    CHK_PRT_RET(
        ((countLeft * unitSize) > cclBufferSize),
        HCCL_ERROR("inputMemSize[0x%x] is greater than CCLbufferSize[0x%x]", (countLeft * unitSize), cclBufferSize),
        HCCL_E_MEMORY);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAllReduceLoop(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec,
                                                     u32 shapeType, const int64_t &comm, const std::string &group,
                                                     void *&inputDataPtr, void *&outputDataPtr, u64 count,
                                                     HcclDataType dataType, HcclReduceOp reduceType,
                                                     rtStream_t streamMain) {
  // 获取 in ccl buf
  u64 commInputSize;
  CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u64 maxCountPerLoop = commInputSize / unitSize;  // ccl buffer内存单次最多能够接受的input count
  u64 curCount = 0;
  u64 outputMaxSize = count * unitSize;

  CHK_RET(CheckTensorNumAndTensorSize(task, count, unitSize, commInputSize));

  DevType devType = HcomGetDeviceType();

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize <= commInputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, inputOffset = 0, outputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomAllReduceLoop]:inputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu]", inputOffset,
              countLeft, curCount, commInputSize);

    CHK_PRT_RET((loopTime >= tagVec.size()),
                HCCL_ERROR("[HcomAllReduceLoop]errNo[0x%016llx] Current tagVec access out-of-bounds",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize) > commInputSize) ? maxCountPerLoop : countLeft;
    CHK_RET(CheckHcomOpMemSize(devType, countLeft, unitSize, commInputSize));

    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位 byte
    // 把size大小的内存，通过二级指针偏移拷贝，从input拷贝到ccl buffer中
    CHK_RET(RefreshInputAddr(devType, shapeType, comm, group, inputDataPtr, inputOffset, curSize,
                             secAddrCopyWithoutOffset, streamMain));
    // 获取cclbuffer
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    CHK_RET(GetCommCCLBuf(shapeType, comm, group, commInputPtr, commOutputPtr));

    // reduce相关场景，会执行缝隙清零
    if (curCount == countLeft) {
      CHK_RET(CleanIntervalMemoryOpKernel(task, tagVec[loopTime], reinterpret_cast<uintptr_t>(commInputPtr),
                                          inputOffset, streamMain, HcclCMDType::HCCL_CMD_ALLREDUCE));
    }

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomAllReduce(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, reduceType,
                            group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomAllReduce(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, reduceType,
                            groupname, streamMain));
    }

    // 将结果拷回二级指针上
    CHK_RET(RefreshOutputAddr(devType, shapeType, comm, group, outputDataPtr, outputOffset, curSize, outputMaxSize,
                              secAddrCopyWithoutOffset, streamMain));

    // 更新偏移量
    inputOffset += curSize;
    outputOffset += curSize;
    loopTime++;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAllGatherOpKernel(const ge::GETaskInfo &task,
                                                         const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;
  u32 shapeType;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[AllGatherOp][Kernel]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));

  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));

  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));

  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));

  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);
  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  if (task.needRefresh) {
    CHK_RET(
        HcomAllGatherLoop(tagVec, shapeType, comm, group, inputDataPtr, outputDataPtr, count, dataType, streamMain));
  } else {
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(
          HcomAllGather(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomAllGather(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, groupname, streamMain));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAllGatherVOpKernel(const ge::GETaskInfo &task,
                                                          const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[AllGatherVOp][Kernel]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));

  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));

  int64_t sendCount = 0;
  void *recvCounts = nullptr;
  void *recvDispls = nullptr;

  GetAllGatherVParams(task, inputAddr, sendCount, outputAddr, recvCounts, recvDispls);
  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);
  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  CHK_RET(HcomAllGatherV(tagVec[0].c_str(), inputDataPtr, sendCount, outputDataPtr, recvCounts, recvDispls, dataType,
                         group.c_str(), streamMain));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAllGatherLoop(const std::vector<std::string> &tagVec, u32 shapeType,
                                                     const int64_t &comm, const std::string &group, void *&inputDataPtr,
                                                     void *&outputDataPtr, u64 count, HcclDataType dataType,
                                                     rtStream_t streamMain) {
  // 获取 out ccl buf
  u64 commOutputSize;
  GetHcomOutCCLbufferSize(commOutputSize, shapeType, comm, group);

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u32 rankSize = 0;
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
  }
  u64 maxCountPerLoop = commOutputSize / (rankSize * unitSize);  // ccl buffer内存单次最多能够接受的input count
  u64 curCount = 0;

  DevType devType = HcomGetDeviceType();

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize * rankSize <= commOutputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, inputOffset = 0, outputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomAllGatherLoop]:inputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu].", inputOffset,
              countLeft, curCount, commOutputSize);

    CHK_PRT_RET((loopTime >= tagVec.size()),
                HCCL_ERROR("[HcomAllGatherLoop]errNo[0x%016llx] Current tagVec access out-of-bounds",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize * rankSize) > commOutputSize) ? maxCountPerLoop : countLeft;
    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位：字节

    CHK_RET(CheckHcomOpMemSize(devType, countLeft, unitSize, commOutputSize));

    // 把size大小的内存，通过二级指针偏移拷贝，从input拷贝到ccl buffer中
    CHK_RET(RefreshInputAddr(devType, shapeType, comm, group, inputDataPtr, inputOffset, curSize,
                             secAddrCopyWithoutOffset, streamMain));
    // 获取cclbuffer
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    CHK_RET(GetCommCCLBuf(shapeType, comm, group, commInputPtr, commOutputPtr));

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomAllGather(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, group.c_str(),
                            streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomAllGather(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, groupname,
                            streamMain));
    }

    // 将结果拷回二级指针上
    CHK_RET(RefreshAllgatherOutputAddr(devType, shapeType, comm, group, outputDataPtr, outputOffset, curSize, count,
                                       unitSize, rankSize, secAddrCopyWithoutOffset, streamMain));

    // 更新偏移量
    inputOffset += curSize;
    outputOffset += curSize;
    loopTime++;
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CreateIndirectCCLbuf() {
  void *indirectInCCLbuffer = nullptr;
  CHK_RET(hrtMalloc(&indirectInCCLbuffer, sizeof(uintptr_t), true));
  CHK_PTR_NULL(indirectInCCLbuffer);
  indirectInCCLbufferPtr_.reset(indirectInCCLbuffer);
  void *indirectOutCCLbuffer = nullptr;
  CHK_RET(hrtMalloc(&indirectOutCCLbuffer, sizeof(uintptr_t), true));
  CHK_PTR_NULL(indirectOutCCLbuffer);
  indirectOutCCLbufferPtr_.reset(indirectOutCCLbuffer);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetIndirectInCCLbuf(void *&ptr, u64 &size) {
  if (indirectInCCLbufferPtr_ == nullptr) {
    ptr = nullptr;
    size = 0;
  } else {
    ptr = indirectInCCLbufferPtr_.get();
    size = sizeof(uintptr_t);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetIndirectOutCCLbuf(void *&ptr, u64 &size) {
  if (indirectOutCCLbufferPtr_ == nullptr) {
    ptr = nullptr;
    size = 0;
  } else {
    ptr = indirectOutCCLbufferPtr_.get();
    size = sizeof(uintptr_t);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetOutputCCLbufPtrAndIndirectOutCCLbufPtr(const int64_t &hcomComm,
                                                                             const std::string &sGroup,
                                                                             void *&commOutputPtr, u64 &commOutputSize,
                                                                             void *&indirectOutCCLbufPtr,
                                                                             u64 &indirectCommOutputSize) {
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
    if (commOutputPtr == nullptr || commOutputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(sGroup.c_str()));
      CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
    }
    CHK_RET(GetIndirectOutCCLbuf(indirectOutCCLbufPtr, indirectCommOutputSize));
    if (indirectOutCCLbufPtr == nullptr || indirectCommOutputSize == 0) {
      CHK_RET(CreateIndirectCCLbuf());
      CHK_RET(GetIndirectOutCCLbuf(indirectOutCCLbufPtr, indirectCommOutputSize));
    }
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
    if (commOutputPtr == nullptr || commOutputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
      CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
    }
    CHK_RET(GetIndirectOutCCLbuf(indirectOutCCLbufPtr, indirectCommOutputSize));
    if (indirectOutCCLbufPtr == nullptr || indirectCommOutputSize == 0) {
      CHK_RET(CreateIndirectCCLbuf());
      CHK_RET(GetIndirectOutCCLbuf(indirectOutCCLbufPtr, indirectCommOutputSize));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetInputCCLbufPtrAndIndirectInCCLbufPtr(const int64_t &hcomComm,
                                                                           const std::string &sGroup,
                                                                           void *&commInputPtr, u64 &commInputSize,
                                                                           void *&indirectInCCLbufPtr,
                                                                           u64 &indirectCommInputSize) {
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    if (commInputPtr == nullptr || commInputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(sGroup.c_str()));
      CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    }
    CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    if (indirectInCCLbufPtr == nullptr || indirectCommInputSize == 0) {
      CHK_RET(CreateIndirectCCLbuf());
      CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    }
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    if (commInputPtr == nullptr || commInputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
      CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    }
    CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    if (indirectInCCLbufPtr == nullptr || indirectCommInputSize == 0) {
      CHK_RET(CreateIndirectCCLbuf());
      CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshAllgatherOutputAddr(DevType deviceType, u32 shapeType,
                                                              const int64_t &hcomComm, const std::string &sGroup,
                                                              void *&outputAddr, u64 outputOffset, u64 curSize,
                                                              u64 count, u32 unitSize, u32 rankSize,
                                                              bool secAddrCopyWithoutOffset, rtStream_t stream) {
  HCCL_DEBUG("[RefreshAllgatherOutputAddr] shapeType[%u]", shapeType);
  void *commOutputPtr = nullptr;
  u64 commOutputSize = 0;
  void *indirectOutCCLbufPtr = nullptr;
  u64 indirectCommOutputSize = 0;

  CHK_RET(GetOutputCCLbufPtrAndIndirectOutCCLbufPtr(hcomComm, sGroup, commOutputPtr, commOutputSize,
                                                    indirectOutCCLbufPtr, indirectCommOutputSize));

  CHK_RET(hrtMemSyncCopy(indirectOutCCLbufPtr, indirectCommOutputSize, &commOutputPtr, indirectCommOutputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把中转buf一级指针指向的数据,偏移拷贝到用户输出一级指针指向的内存空间 rts提供新的拷贝task
  if ((deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910) || secAddrCopyWithoutOffset) {
    CHK_RET(hrtMemAsyncCopy(outputAddr, curSize * rankSize, indirectOutCCLbufPtr, curSize * rankSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

    HCCL_DEBUG("outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p, MemCopySize=%llu",
               outputAddr, indirectOutCCLbufPtr, &commOutputPtr, commOutputPtr, curSize * rankSize);
  } else {
    for (u32 i = 0; i < rankSize; i++) {
      u64 userOutMemOffset = outputOffset + (count * unitSize * i);
      u64 indirectOutCCLBufOffset = curSize * i;
      u64 outputMaxSize = count * unitSize * rankSize;
      CHK_RET(hrtMemcpyAddrAsync(outputAddr, outputMaxSize, userOutMemOffset, indirectOutCCLbufPtr, curSize,
                                 indirectOutCCLBufOffset, stream));

      HCCL_DEBUG(
          "outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p curSize=%llu "
          "userOutMemOffset=%llu indirectOutCCLBufOffset=%llu.",
          outputAddr, indirectOutCCLbufPtr, &commOutputPtr, commOutputPtr, curSize, userOutMemOffset,
          indirectOutCCLBufOffset);
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcomOutCCLbufferSize(u64 &commOutputSize, u32 shapeType, const int64_t &hcomComm,
                                                           const std::string &sGroup) {
  HCCL_DEBUG("[GetHcomOutCCLbufferSize] shapeType[%u]", shapeType);
  void *commOutputPtr = nullptr;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
    if (commOutputPtr == nullptr || commOutputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(sGroup.c_str()));
      CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
    }
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
    if (commOutputPtr == nullptr || commOutputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
      CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcomInCCLbufferSize(u64 &commInputSize, u32 shapeType, const int64_t &hcomComm,
                                                          const std::string &sGroup) {
  HCCL_DEBUG("[GetHcomInCCLbufferSize] shapeType[%u]", shapeType);
  void *commInputPtr = nullptr;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    if (commInputPtr == nullptr || commInputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(sGroup.c_str()));
      CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    }
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    if (commInputPtr == nullptr || commInputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
      CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    }
  }
  return HCCL_SUCCESS;
}
HcclResult HcomOpsKernelInfoStore::HcomReduceScatterOpKernel(const ge::GETaskInfo &task,
                                                             const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;
  u32 shapeType;
  HcclDataType dataType;
  rtStream_t streamMain;
  HcclReduceOp reduceType;
  std::vector<void *> globalWorkSpaceAddr;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[ReduceScatterOp][Kernel]errNo[0x%016llx] GETaskInfo size"
                         "in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  // 获取 hcom api 必须的参数
  std::string funStr = "HcomReduceScatterOpKernel";
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  CHK_RET(GetReduceTypeFromTaskInfo(hcclInfo, reduceType));

  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);
  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  // 获取溢出检测内存地址
  CHK_RET(GetGlobalWorkSpaceAddrFromTaskInfo(hcclInfo, globalWorkSpaceAddr));

  CHK_RET(SetGlobalWorkSpace(comm, group, globalWorkSpaceAddr));

  if (task.needRefresh) {
    CHK_RET(HcomReduceScatterLoop(task, tagVec, shapeType, comm, group, inputDataPtr, outputDataPtr, count, dataType,
                                  reduceType, streamMain));
  } else {
    CHK_RET(CleanIntervalMemoryOpKernel(task, tagVec[0], reinterpret_cast<uintptr_t>(inputDataPtr), 0, streamMain,
                                        HcclCMDType::HCCL_CMD_REDUCE_SCATTER));

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomReduceScatter(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType,
                                group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomReduceScatter(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType, groupname,
                                streamMain));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomReduceScatterVOpKernel(const ge::GETaskInfo &task,
                                                              const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  HcclDataType dataType;
  rtStream_t streamMain;
  HcclReduceOp reduceType;
  std::vector<void *> globalWorkSpaceAddr;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[ReduceScatterVOp][Kernel]errNo[0x%016llx] GETaskInfo size"
                         "in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  // 获取 hcom api 必须的参数
  std::string funStr = "HcomReduceScatterVOpKernel";
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetReduceTypeFromTaskInfo(hcclInfo, reduceType));

  // 获取溢出检测内存地址
  CHK_RET(GetGlobalWorkSpaceAddrFromTaskInfo(hcclInfo, globalWorkSpaceAddr));

  CHK_RET(SetGlobalWorkSpace(comm, group, globalWorkSpaceAddr));

  void *sendCounts = nullptr;
  void *sendDispls = nullptr;
  int64_t recvCount = 0;

  GetReduceScatterVParams(task, inputAddr, sendCounts, sendDispls, outputAddr, recvCount);
  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);
  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  // 执行 hcom 算子
  CHK_RET(HcomReduceScatterV(tagVec[0].c_str(), inputDataPtr, sendCounts, sendDispls, outputDataPtr, recvCount,
                             dataType, reduceType, group.c_str(), streamMain));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomReduceScatterLoop(const ge::GETaskInfo &task,
                                                         const std::vector<std::string> &tagVec, u32 shapeType,
                                                         const int64_t &comm, const std::string &group,
                                                         void *&inputDataPtr, void *&outputDataPtr, u64 count,
                                                         HcclDataType dataType, HcclReduceOp reduceType,
                                                         rtStream_t streamMain) {
  // 获取 in ccl buf
  u64 commInputSize;
  CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u32 rankSize = 0;
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
  }
  u64 maxCountPerLoop = commInputSize / (rankSize * unitSize);  // 中转内存单次最多能够接受的output count
  u64 curCount = 0;
  u64 outputMaxSize = count * unitSize;

  DevType devType = HcomGetDeviceType();

  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  CHK_PTR_NULL(privateDefBuf);
  size_t tensorNum = privateDefBuf->tensorNum;
  if (tensorNum > 1 && ((count * unitSize * rankSize) > commInputSize)) {
    HCCL_WARNING("tensorNum is [%d] UserMemSize is [%llu]  cclbufferSize[%llu]", tensorNum, count * unitSize * rankSize,
                 commInputSize);
  }

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize * rankSize <= commInputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, inputOffset = 0, outputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomReduceScatterLoop]:inputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu].", inputOffset,
              countLeft, curCount, commInputSize);

    CHK_PRT_RET((loopTime >= tagVec.size()),
                HCCL_ERROR("[HcomReduceScatterLoop]errNo[0x%016llx] Current tagVec access out-of-bounds",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize * rankSize) > commInputSize) ? maxCountPerLoop : countLeft;
    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位 byte

    CHK_RET(CheckHcomOpMemSize(devType, countLeft * rankSize, unitSize, commInputSize));

    // 把size大小的内存，通过二级指针偏移拷贝，从input拷贝到ccl buffer中
    CHK_RET(RefreshReduceScatterInputAddr(devType, shapeType, comm, group, inputDataPtr, inputOffset, curSize, count,
                                          unitSize, rankSize, secAddrCopyWithoutOffset, streamMain));

    // 获取cclbuffer
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    CHK_RET(GetCommCCLBuf(shapeType, comm, group, commInputPtr, commOutputPtr));

    // reduce相关场景，会执行缝隙清零
    if (tensorNum > 1 && curCount == countLeft) {
      CHK_RET(CleanIntervalMemoryOpKernel(task, tagVec[loopTime], reinterpret_cast<uintptr_t>(commInputPtr),
                                          inputOffset, streamMain, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    }

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomReduceScatter(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, reduceType,
                                group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomReduceScatter(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType, groupname,
                                streamMain));
    }

    // 将结果拷回二级指针上
    CHK_RET(RefreshOutputAddr(devType, shapeType, comm, group, outputDataPtr, outputOffset, curSize, outputMaxSize,
                              secAddrCopyWithoutOffset, streamMain));

    // 更新偏移量
    inputOffset += curSize;
    outputOffset += curSize;
    loopTime++;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshReduceScatterInputAddr(DevType deviceType, u32 shapeType,
                                                                 const int64_t &hcomComm, const std::string &sGroup,
                                                                 void *&inputAddr, u64 inputOffset, u64 curSize,
                                                                 u64 count, u32 unitSize, u32 rankSize,
                                                                 bool secAddrCopyWithoutOffset, rtStream_t stream) {
  HCCL_DEBUG("[RefreshReduceScatterInputAddr] shapeType[%u]", shapeType);
  void *commInputPtr = nullptr;
  u64 commInputSize = 0;
  void *indirectInCCLbufPtr = nullptr;
  u64 indirectCommInputSize = 0;

  CHK_RET(GetInputCCLbufPtrAndIndirectInCCLbufPtr(hcomComm, sGroup, commInputPtr, commInputSize, indirectInCCLbufPtr,
                                                  indirectCommInputSize));

  CHK_RET(hrtMemSyncCopy(indirectInCCLbufPtr, indirectCommInputSize, &commInputPtr, indirectCommInputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把中转buf一级指针指向的数据,偏移拷贝到用户输出一级指针指向的内存空间 rts提供新的拷贝task
  if ((deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910) || secAddrCopyWithoutOffset) {
    CHK_RET(hrtMemAsyncCopy(indirectInCCLbufPtr, curSize * rankSize, inputAddr, curSize * rankSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

    HCCL_DEBUG("indirectInCCLbufPtr=%p, inputAddr=%p, &commInputPtr=%p commInputPtr=%p MemCopySize=%llu",
               indirectInCCLbufPtr, inputAddr, &commInputPtr, commInputPtr, curSize * rankSize);
  } else {
    for (u32 i = 0; i < rankSize; i++) {
      u64 userInMemOffset = inputOffset + (count * unitSize * i);
      u64 indirectInCCLBufOffset = curSize * i;

      CHK_RET(hrtMemcpyAddrAsync(indirectInCCLbufPtr, commInputSize, indirectInCCLBufOffset, inputAddr, curSize,
                                 userInMemOffset, stream));

      HCCL_DEBUG(
          "indirectInCCLbufPtr=%p, inputAddr=%p, &commInputPtr=%p commInputPtr=%p curSize=%llu "
          "userInMemOffset=%llu indirectInCCLBufOffset=%llu.",
          indirectInCCLbufPtr, inputAddr, &commInputPtr, commInputPtr, curSize, userInMemOffset,
          indirectInCCLBufOffset);
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomBroadcastOpKernel(const ge::GETaskInfo &task,
                                                         const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  u64 count;
  u32 shapeType;
  u32 root;
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[BroadcastOp][Kernel]errNo[0x%016llx] GETaskInfo size"
                         "in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));

  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));

  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));

  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));

  CHK_RET(GetRootFromTaskInfo(hcclInfo, root));

  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);

  if (task.needRefresh) {
    CHK_RET(HcomBroadcastLoop(tagVec, shapeType, comm, group, inputDataPtr, count, dataType, root, streamMain));
  } else {
    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomBroadcast(tagVec[0].c_str(), inputDataPtr, count, dataType, root, group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomBroadcast(tagVec[0].c_str(), inputDataPtr, count, dataType, root, groupname, streamMain));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomBroadcastLoop(const std::vector<std::string> &tagVec, u32 shapeType,
                                                     const int64_t &comm, const std::string &group, void *&inputDataPtr,
                                                     u64 count, HcclDataType dataType, u32 root,
                                                     rtStream_t streamMain) {
  // 获取 in ccl buf
  u64 commInputSize;
  CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u64 maxCountPerLoop = commInputSize / unitSize;  // ccl buffer内存单次最多能够接受的input count
  u64 curCount = 0;
  u64 outputMaxSize = count * unitSize;

  DevType devType = HcomGetDeviceType();

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize <= commInputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, inputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomBroadcastLoop]:inputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu]", inputOffset,
              countLeft, curCount, commInputSize);

    CHK_PRT_RET((loopTime >= tagVec.size()),
                HCCL_ERROR("[HcomBroadcastLoop]errNo[0x%016llx] Current tagVec access out-of-bounds",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize) > commInputSize) ? maxCountPerLoop : countLeft;
    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位 byte

    CHK_RET(CheckHcomOpMemSize(devType, countLeft, unitSize, commInputSize));

    // 把size大小的内存，通过二级指针偏移拷贝，从input拷贝到ccl buffer中
    CHK_RET(RefreshInputAddr(devType, shapeType, comm, group, inputDataPtr, inputOffset, curSize,
                             secAddrCopyWithoutOffset, streamMain));
    // 获取cclbuffer
    void *commInputPtr = nullptr;
    CHK_RET(GetCommCCLBuf(shapeType, HCCL_KERNEL_OP_TYPE_BROADCAST, comm, group, commInputPtr));

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(
          HcomBroadcast(tagVec[loopTime].c_str(), commInputPtr, curCount, dataType, root, group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomBroadcast(tagVec[loopTime].c_str(), commInputPtr, curCount, dataType, root, groupname, streamMain));
    }

    // 将结果拷回二级指针上
    CHK_RET(RefreshOutputAddr(devType, shapeType, HCCL_KERNEL_OP_TYPE_BROADCAST, comm, group, inputDataPtr, inputOffset,
                              curSize, outputMaxSize, secAddrCopyWithoutOffset, streamMain));

    // 更新偏移量
    inputOffset += curSize;
    loopTime++;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomReduceOpKernel(const ge::GETaskInfo &task,
                                                      const std::vector<std::string> &tagVec) {
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[ReduceOp][Kernel]errNo[0x%016llx] GETaskInfo size in HCOM"
                         "should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  HcclDataType dataType;
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));

  std::string group;
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }

  u64 count;
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));

  u32 root;
  CHK_RET(GetRootFromTaskInfo(hcclInfo, root));

  rtStream_t streamMain;
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));

  uintptr_t inputAddr = 0;
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));

  uintptr_t outputAddr = 0;
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));

  u32 shapeType = 0;
  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  std::vector<void *> globalWorkSpaceAddr;
  // 获取溢出检测内存地址
  CHK_RET(GetGlobalWorkSpaceAddrFromTaskInfo(hcclInfo, globalWorkSpaceAddr));

  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);
  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  CHK_RET(SetGlobalWorkSpace(comm, group, globalWorkSpaceAddr));
  HcclReduceOp reduceType;
  CHK_RET(GetReduceTypeFromTaskInfo(hcclInfo, reduceType));

  if (task.needRefresh) {
    CHK_RET(HcomReduceLoop(task, tagVec, shapeType, comm, group, inputDataPtr, outputDataPtr, count, dataType,
                           reduceType, root, streamMain));
  } else {
    CHK_RET(CleanIntervalMemoryOpKernel(task, tagVec[0], reinterpret_cast<uintptr_t>(inputDataPtr), 0, streamMain,
                                        HcclCMDType::HCCL_CMD_REDUCE));
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomReduce(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType, root,
                         group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomReduce(tagVec[0].c_str(), inputDataPtr, outputDataPtr, count, dataType, reduceType, root, groupname,
                         streamMain));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomReduceLoop(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec,
                                                  u32 shapeType, const int64_t &comm, const std::string &group,
                                                  void *&inputDataPtr, void *&outputDataPtr, u64 count,
                                                  HcclDataType dataType, HcclReduceOp reduceType, u32 root,
                                                  rtStream_t streamMain) {
  // 获取 in ccl buf
  u64 commInputSize;
  CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u64 maxCountPerLoop = commInputSize / unitSize;  // ccl buffer内存单次最多能够接受的input count
  u64 curCount = 0;

  // 获取当前rankid
  u32 rankId = 0;
  u32 rankSize = 0;
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankId(group.c_str(), &rankId));
    CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomGetRankId(groupname, &rankId));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
  }

  u64 outputMaxSize = count * unitSize / rankSize;

  DevType devType = HcomGetDeviceType();

  CHK_RET(CheckTensorNumAndTensorSize(task, count, unitSize, commInputSize));

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize <= commInputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, inputOffset = 0, outputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomReduceLoop]:inputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu].", inputOffset,
              countLeft, curCount, commInputSize);

    CHK_PRT_RET((loopTime >= tagVec.size()),
                HCCL_ERROR("[HcomReduceLoop]errNo[0x%016llx] Current tagVec access out-of-bounds",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize) > commInputSize) ? maxCountPerLoop : countLeft;
    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位 byte

    CHK_RET(CheckHcomOpMemSize(devType, countLeft, unitSize, commInputSize));

    // 把size大小的内存，通过二级指针偏移拷贝，从input拷贝到ccl buffer中
    CHK_RET(RefreshInputAddr(devType, shapeType, comm, group, inputDataPtr, inputOffset, curSize,
                             secAddrCopyWithoutOffset, streamMain));
    // 获取cclbuffer
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    CHK_RET(GetCommCCLBuf(shapeType, comm, group, commInputPtr, commOutputPtr));

    // 7 reduce相关场景，会多一步缝隙清零
    if (curCount == countLeft) {
      CHK_RET(CleanIntervalMemoryOpKernel(task, tagVec[loopTime], reinterpret_cast<uintptr_t>(commInputPtr),
                                          inputOffset, streamMain, HcclCMDType::HCCL_CMD_REDUCE));
    }
    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomReduce(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, reduceType, root,
                         group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomReduce(tagVec[loopTime].c_str(), commInputPtr, commOutputPtr, curCount, dataType, reduceType, root,
                         groupname, streamMain));
    }

    // 只root rank将结果拷回二级指针上
    if (rankId == root) {
      CHK_RET(RefreshOutputAddr(devType, shapeType, comm, group, outputDataPtr, outputOffset, curSize, outputMaxSize,
                                secAddrCopyWithoutOffset, streamMain));
    }

    // 更新偏移量
    inputOffset += curSize;
    outputOffset += curSize;
    loopTime++;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomSendOpKernel(const ge::GETaskInfo &task,
                                                    const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  u64 count = 0;
  u32 destRank = 0;
  u32 srTag = 0;

  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[SendOp][Kernel]errNo[0x%016llx] GETaskInfo size in HCOM"
                         "should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个

  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));
  CHK_RET(GetDestRankFromTaskInfo(task, destRank));
  CHK_RET(GetSrTagFromTaskInfo(task, srTag));

  u32 shapeType;

  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  void *inputDataPtr = reinterpret_cast<void *>(inputAddr);

  if (task.needRefresh) {
    CHK_RET(HcomSendLoop(tagVec, srTag, shapeType, comm, group, inputDataPtr, count, dataType, destRank, streamMain));
  } else {
    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomSend(tagVec[0].c_str(), inputDataPtr, count, dataType, destRank, srTag, group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomSend(tagVec[0].c_str(), inputDataPtr, count, dataType, destRank, srTag, groupname, streamMain));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomSendLoop(const std::vector<std::string> &tagVec, u32 &srTag, u32 shapeType,
                                                const int64_t &comm, const std::string &group, void *&inputDataPtr,
                                                u64 count, HcclDataType dataType, u32 &destRank,
                                                rtStream_t streamMain) {
  // 获取 in ccl buf
  u64 commInputSize;
  CHK_RET(GetHcomInCCLbufferSize(commInputSize, shapeType, comm, group));

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u64 maxCountPerLoop = commInputSize / unitSize;  // ccl buffer内存单次最多能够接受的input count
  u64 curCount = 0;

  DevType devType = HcomGetDeviceType();

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize <= commInputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, inputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomSendLoop]:inputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu].", inputOffset,
              countLeft, curCount, commInputSize);

    CHK_PRT_RET(
        (loopTime >= tagVec.size()),
        HCCL_ERROR("[HcomSendLoop]errNo[0x%016llx] Current tagVec access out-of-bounds", HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize) > commInputSize) ? maxCountPerLoop : countLeft;
    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位 byte

    CHK_RET(CheckHcomOpMemSize(devType, countLeft, unitSize, commInputSize));

    // 把size大小的内存，通过二级指针偏移拷贝，从input拷贝到ccl buffer中
    CHK_RET(RefreshInputAddr(devType, shapeType, comm, group, inputDataPtr, inputOffset, curSize,
                             secAddrCopyWithoutOffset, streamMain));
    // 获取cclbuffer
    void *commInputPtr = nullptr;
    if (shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE) {
      if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
        CHK_RET(GetInCCLbuffer(group.c_str(), commInputPtr, commInputSize));
      } else {
        char *groupname = nullptr;
        CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
        CHK_RET(GetInCCLbuffer(groupname, commInputPtr, commInputSize));
      }
    }

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomSend(tagVec[loopTime].c_str(), commInputPtr, curCount, dataType, destRank, srTag, group.c_str(),
                       streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(
          HcomSend(tagVec[loopTime].c_str(), commInputPtr, curCount, dataType, destRank, srTag, groupname, streamMain));
    }

    // 更新偏移量
    inputOffset += curSize;
    loopTime++;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomReceiveOpKernel(const ge::GETaskInfo &task,
                                                       const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t outputAddr = 0;
  u64 count = 0;
  u32 srcRank = 0;
  u32 srTag = 0;

  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[ReceiveOp][Kernel]errNo[0x%016llx] GETaskInfo size in HCOM"
                         "should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  CHK_RET(GetSrcRankFromTaskInfo(task, srcRank));
  CHK_RET(GetSrTagFromTaskInfo(task, srTag));

  u32 shapeType;
  // 动态shap地址刷新
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  void *outputDataPtr = reinterpret_cast<void *>(outputAddr);

  if (task.needRefresh) {
    CHK_RET(
        HcomReceiveLoop(tagVec, srTag, shapeType, comm, group, outputDataPtr, count, dataType, srcRank, streamMain));
  } else {
    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(
          HcomReceive(tagVec[0].c_str(), outputDataPtr, count, dataType, srcRank, srTag, group.c_str(), streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomReceive(tagVec[0].c_str(), outputDataPtr, count, dataType, srcRank, srTag, groupname, streamMain));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomReceiveLoop(const std::vector<std::string> &tagVec, u32 &srTag, u32 shapeType,
                                                   const int64_t &comm, const std::string &group, void *&outputDataPtr,
                                                   u64 count, HcclDataType dataType, u32 &srcRank,
                                                   rtStream_t streamMain) {
  // 获取 in ccl buf
  u64 commOutputSize;
  GetHcomOutCCLbufferSize(commOutputSize, shapeType, comm, group);

  // 计算出cclbuffer支持最大的count数量
  u32 unitSize = SIZE_TABLE[dataType];
  u64 maxCountPerLoop = commOutputSize / unitSize;  // ccl buffer内存单次最多能够接受的input count
  u64 curCount = 0;
  u64 outputMaxSize = count * unitSize;

  DevType devType = HcomGetDeviceType();

  // 如果通信size小于CCL BUFF size，不走二级地址偏移拷贝
  bool secAddrCopyWithoutOffset = false;
  if (count * unitSize <= commOutputSize) {
    secAddrCopyWithoutOffset = true;
  }

  for (u64 countLeft = count, outputOffset = 0, loopTime = 0; countLeft > 0; countLeft -= curCount) {
    HCCL_INFO("[HcomReceiveLoop]:outputOffset[%llu] countLeft[%llu] curCount[%llu] cclbuffer[%llu].", outputOffset,
              countLeft, curCount, commOutputSize);

    CHK_PRT_RET((loopTime >= tagVec.size()),
                HCCL_ERROR("[HcomReceiveLoop]errNo[0x%016llx] Current tagVec access out-of-bounds",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    // 大于cclbuffer，count取cclbuffer最大支持count，否则走input支持的count
    curCount = ((countLeft * unitSize) > commOutputSize) ? maxCountPerLoop : countLeft;
    // 通过count得出size
    u64 curSize = curCount * unitSize;  // 单位 byte

    CHK_RET(CheckHcomOpMemSize(devType, countLeft, unitSize, commOutputSize));

    // 获取cclbuffer
    void *commOutputPtr = nullptr;
    if (shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE) {
      if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
        CHK_RET(GetOutCCLbuffer(group.c_str(), commOutputPtr, commOutputSize));
      } else {
        char *groupname = nullptr;
        CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
        CHK_RET(GetOutCCLbuffer(groupname, commOutputPtr, commOutputSize));
      }
    }

    // 执行 hcom 算子
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomReceive(tagVec[loopTime].c_str(), commOutputPtr, curCount, dataType, srcRank, srTag, group.c_str(),
                          streamMain));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomReceive(tagVec[loopTime].c_str(), commOutputPtr, curCount, dataType, srcRank, srTag, groupname,
                          streamMain));
    }

    // 将结果拷回二级指针上
    CHK_RET(RefreshOutputAddr(devType, shapeType, comm, group, outputDataPtr, outputOffset, curSize, outputMaxSize,
                              secAddrCopyWithoutOffset, streamMain));

    // 更新偏移量
    outputOffset += curSize;
    loopTime++;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcomAlltoallVOpMemSize(u32 shapeType, const std::string &sCollectiveType,
                                                             const int64_t &hcomComm, const std::string &sGroup,
                                                             HcclDataType sendType, HcclDataType recvType,
                                                             void *sendCounts, void *sendDispls, void *recvCounts,
                                                             void *recvDispls, u64 &inputMemSize, u64 &outputMemSize) {
  HCCL_DEBUG("[GetHcomAlltoallVOpMemSize] sCollectiveType[%s]", sCollectiveType.c_str());
  CHK_PRT_RET((shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE), HCCL_DEBUG("No need to get op MemSize"), HCCL_SUCCESS);

  u32 sendDataTypeSize, recvDataTypeSize;
  CHK_RET(SalGetDataTypeSize(sendType, sendDataTypeSize));
  CHK_RET(SalGetDataTypeSize(recvType, recvDataTypeSize));

  u32 rankSize = 0;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankSize(sGroup.c_str(), &rankSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &groupname));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
  }

  u64 sendCount = 0;
  u64 recvCount = 0;
  for (u32 i = 0; i < rankSize; i++) {
    u64 curSendCount = *(static_cast<const u64 *>(sendCounts) + i) + *(static_cast<const u64 *>(sendDispls) + i);
    sendCount = std::max(sendCount, curSendCount);
    u64 curRecvCount = *(static_cast<const u64 *>(recvCounts) + i) + *(static_cast<const u64 *>(recvDispls) + i);
    recvCount = std::max(recvCount, curRecvCount);
  }

  inputMemSize = sendCount * sendDataTypeSize;
  outputMemSize = recvCount * recvDataTypeSize;

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcomAlltoallVCOpMemSize(u32 shapeType, const std::string &sCollectiveType,
                                                              const int64_t &hcomComm, const std::string &sGroup,
                                                              HcclDataType sendType, HcclDataType recvType,
                                                              void *sendCountMatrix, u64 &inputMemSize,
                                                              u64 &outputMemSize) {
  HCCL_DEBUG("[GetHcomAlltoallVCOpMemSize] sCollectiveType[%s]", sCollectiveType.c_str());
  CHK_PRT_RET((shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE), HCCL_DEBUG("No need to get op MemSize"), HCCL_SUCCESS);

  u32 sendDataTypeSize, recvDataTypeSize;
  CHK_RET(SalGetDataTypeSize(sendType, sendDataTypeSize));
  CHK_RET(SalGetDataTypeSize(recvType, recvDataTypeSize));

  u32 rankSize = 0;
  u32 rankId = 0;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankSize(sGroup.c_str(), &rankSize));
    CHK_RET(HcomGetRankId(sGroup.c_str(), &rankId));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &groupname));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
    CHK_RET(HcomGetRankId(groupname, &rankId));
  }

  u64 sendCount = 0;
  u64 recvCount = 0;
  for (u32 i = 0; i < rankSize; i++) {
    sendCount += *(static_cast<const u64 *>(sendCountMatrix) + rankId * rankSize + i);
    recvCount += *(static_cast<const u64 *>(sendCountMatrix) + rankId + rankSize * i);
  }

  inputMemSize = sendCount * sendDataTypeSize;
  outputMemSize = recvCount * recvDataTypeSize;

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcomOpMemSize(u32 shapeType, const std::string &sCollectiveType,
                                                    const int64_t &hcomComm, const std::string &sGroup,
                                                    HcclDataType dataType, u64 count, u64 &inputMemSize,
                                                    u64 &outputMemSize) {
  CHK_PRT_RET((shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE), HCCL_DEBUG("No need to get op MemSize"), HCCL_SUCCESS);

  u32 dataTypeSize;
  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  u32 rankSize = 0;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomGetRankSize(sGroup.c_str(), &rankSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &groupname));
    CHK_RET(HcomGetRankSize(groupname, &rankSize));
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
    inputMemSize = count * dataTypeSize;
    outputMemSize = inputMemSize;
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
    inputMemSize = count * dataTypeSize;
    outputMemSize = inputMemSize * rankSize;
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
    inputMemSize = rankSize * count * dataTypeSize;
    outputMemSize = count * dataTypeSize;
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE) {
    inputMemSize = count * dataTypeSize;
    outputMemSize = inputMemSize / rankSize;
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_SEND) {
    inputMemSize = count * dataTypeSize;
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    outputMemSize = count * dataTypeSize;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcomOpMemSize(u32 shapeType, const std::string &sCollectiveType,
                                                    HcclDataType dataType, u64 count, u64 &inputMemSize) {
  CHK_PRT_RET(sCollectiveType != HCCL_KERNEL_OP_TYPE_BROADCAST,
              HCCL_ERROR("[Get][HcomOpMemSize]do not support the communication type[%s]", sCollectiveType.c_str()),
              HCCL_E_PARA);
  CHK_PRT_RET((shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE), HCCL_DEBUG("No need to get op MemSize"), HCCL_SUCCESS);

  u32 dataTypeSize;
  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  inputMemSize = count * dataTypeSize;
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetCommCCLBuf(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup,
                                                 void *&commInputPtr, void *&commOutputPtr) {
  HCCL_DEBUG("[GetCommCCLBuf] shapeType[%u]", shapeType);
  u64 commInputSize = 0;
  u64 commOutputSize = 0;

  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CheckOutputMemSize(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup,
                                                      u64 outputMemSize) {
  CHK_PRT_RET(shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE, HCCL_INFO("not need to check output mem size"),
              HCCL_SUCCESS);

  void *commOutputPtr = nullptr;
  u64 commOutputSize;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
    if (commOutputPtr == nullptr || commOutputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(sGroup.c_str()));
      CHK_RET(GetOutCCLbuffer(sGroup.c_str(), commOutputPtr, commOutputSize));
    }
  } else {
    char *group = nullptr;
    CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
    if (commOutputPtr == nullptr || commOutputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
      CHK_RET(GetOutCCLbuffer(group, commOutputPtr, commOutputSize));
    }
  }

  CHK_PRT_RET((outputMemSize > commOutputSize),
              HCCL_ERROR("[Check][OutputMemSize]outputMemSize[0x%x] is greater than CCLbufferSize[0x%x]", outputMemSize,
                         commOutputSize),
              HCCL_E_MEMORY);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetCommCCLBuf(u32 shapeType, const std::string &sCollectiveType,
                                                 const int64_t &hcomComm, const std::string &sGroup,
                                                 void *&commInputPtr) {
  HCCL_DEBUG("[Get][CommCCLBuf] shapeType[%u]", shapeType);
  CHK_PRT_RET(sCollectiveType != HCCL_KERNEL_OP_TYPE_BROADCAST,
              HCCL_ERROR("[Get][CommCCLBuf]do not support the communication type[%s]", sCollectiveType.c_str()),
              HCCL_E_PARA);

  u64 commInputSize = 0;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SetWorkspaceResourceFromtagVec(const ge::GETaskInfo &task, const char *group,
                                                                  const std::vector<std::string> &tagVec, void *memPtr,
                                                                  u64 maxSize) {
  for (u32 loopTime = 0; loopTime < tagVec.size(); loopTime++) {
    CHK_RET(SetWorkspaceResource(tagVec[loopTime], group, task.kernelHcclInfo[0].hcclStreamList, memPtr, maxSize));
    HCCL_INFO("load task: tag[%s] sub stream size is %u,size is %llu bytes.", (tagVec[loopTime]).c_str(),
              task.kernelHcclInfo[0].hcclStreamList.size(), maxSize);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SetUnknownShapeWorkspaceResource(const ge::GETaskInfo &task,
                                                                    const std::string &sCollectiveType,
                                                                    const std::vector<std::string> &tagVec) {
  std::string sGroup;
  int64_t comm = 0;
  CHK_PRT_RET(task.kernelHcclInfo.empty(),
              HCCL_ERROR("[Set][UnknownShapeWorkspaceResource]kernelHcclInfo"
                         "is empty"),
              HCCL_E_PARA);
  CHK_RET(GetCommFromTaskInfo(task, comm));
  std::unique_lock<std::mutex> workSpaceLock(workSpaceMemMutex_);
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, sGroup));
    auto workSpaceIter = workSpaceMemInfo_.find(sGroup);
    if (workSpaceIter != workSpaceMemInfo_.end()) {
      CHK_RET(SetWorkspaceResourceFromtagVec(task, sGroup.c_str(), tagVec, std::get<0>(workSpaceMemInfo_[sGroup]),
                                             std::get<1>(workSpaceMemInfo_[sGroup])));
    } else {
      bool isDeterministicOptim = false;
      CHK_RET(HcomSupportDeterministicOptim(sGroup.c_str(), &isDeterministicOptim));
      if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER || isDeterministicOptim) {
        // reduce scatter 和确定性计算 strach mem 用于缓冲接收对端数据
        u64 inCclSize = 0;
        void *ptr = nullptr;
        CHK_RET(GetInCCLbuffer(sGroup.c_str(), ptr, inCclSize));
        u64 workSpaceMemSize = HCCL_WORKSPACE_MEM_32_KB + inCclSize;

        // 增加内存对齐部分
        workSpaceMemSize = (workSpaceMemSize + HCCL_ALIGN_SIZE - 1) / HCCL_ALIGN_SIZE * HCCL_ALIGN_SIZE;
        HCCL_INFO("workspace memory size: group[%s],mem size[%llu]", sGroup.c_str(), workSpaceMemSize);

        void *workSpaceMemPtr = nullptr;
        CHK_RET(hrtMalloc(&workSpaceMemPtr, workSpaceMemSize));
        CHK_RET(SetWorkspaceResourceFromtagVec(task, sGroup.c_str(), tagVec, workSpaceMemPtr, workSpaceMemSize));

        std::tuple<void *, u64> workSpaceMemInfo = std::make_tuple(workSpaceMemPtr, workSpaceMemSize);
        workSpaceMemInfo_.insert(std::make_pair(sGroup, workSpaceMemInfo));
      } else {
        // 设定其他算子的 workspace stream 资源 */
        void *workSpaceMemPtr = nullptr;
        u64 workSpaceMemSize = 0;
        CHK_RET(SetWorkspaceResourceFromtagVec(task, sGroup.c_str(), tagVec, workSpaceMemPtr, workSpaceMemSize));
      }
    }
  } else {
    auto workSpaceIter = workSpaceMemInfo_.find(to_string(comm));
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &group));
    if (workSpaceIter != workSpaceMemInfo_.end()) {
      CHK_RET(SetWorkspaceResourceFromtagVec(task, group, tagVec, std::get<0>(workSpaceMemInfo_[to_string(comm)]),
                                             std::get<1>(workSpaceMemInfo_[to_string(comm)])));
    } else {
      bool isDeterministicOptim = false;
      CHK_RET(HcomSupportDeterministicOptim(group, &isDeterministicOptim));
      if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER || isDeterministicOptim) {
        // reduce scatter 和确定性计算 strach mem 用于缓冲接收对端数据
        u64 inCclSize = 0;
        void *ptr = nullptr;
        CHK_RET(GetInCCLbuffer(sGroup.c_str(), ptr, inCclSize));
        u64 workSpaceMemSize = HCCL_WORKSPACE_MEM_32_KB + inCclSize;

        // 增加内存对齐部分
        workSpaceMemSize = (workSpaceMemSize + HCCL_ALIGN_SIZE - 1) / HCCL_ALIGN_SIZE * HCCL_ALIGN_SIZE;
        HCCL_INFO("workspace memory size: comm[%lld], mem size[%llu]", comm, workSpaceMemSize);

        void *workSpaceMemPtr = nullptr;
        CHK_RET(hrtMalloc(&workSpaceMemPtr, workSpaceMemSize));

        CHK_RET(SetWorkspaceResourceFromtagVec(task, group, tagVec, workSpaceMemPtr, workSpaceMemSize));

        std::tuple<void *, u64> workSpaceMemInfo = std::make_tuple(workSpaceMemPtr, workSpaceMemSize);
        workSpaceMemInfo_.insert(std::make_pair(to_string(comm), workSpaceMemInfo));
      } else {
        // 设定其他算子的 workspace stream 资源 */
        void *workSpaceMemPtr = nullptr;
        u64 workSpaceMemSize = 0;
        CHK_RET(SetWorkspaceResourceFromtagVec(task, group, tagVec, workSpaceMemPtr, workSpaceMemSize));
      }
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SetKnownShapeWorkspaceResource(const ge::GETaskInfo &task,
                                                                  const std::string &sCollectiveType,
                                                                  const std::vector<std::string> &tagVec) {
  std::string group;
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[Set][KnownShapeWorkspaceResource]errNo[0x%016llx] GETaskInfo"
                         "size in HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  // 设定 workspace memory 全局资源
  if ((hcclInfo.workSpaceAddr == nullptr) || (hcclInfo.workSpaceMemSize == 0)) {
    HCCL_ERROR(
        "[Set][KnownShapeWorkspaceResource]errNo[0x%016llx] load task failed. "
        "workspace memory ptr is null or size is [%llu]",
        HCOM_ERROR_CODE(HCCL_E_PARA), hcclInfo.workSpaceMemSize);
    return HCCL_E_PARA;
  }

  // 设定 stream 全局资源
  for (auto stream : hcclInfo.hcclStreamList) {
    CHK_PRT_RET((stream == nullptr),
                HCCL_ERROR("[Set][KnownShapeWorkspaceResource]errNo[0x%016llx] load"
                           "task failed. (stream from taskinfo is null)",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);
  }

  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(
        SetWorkspaceResourceFromtagVec(task, group.c_str(), tagVec, hcclInfo.workSpaceAddr, hcclInfo.workSpaceMemSize));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(SetWorkspaceResourceFromtagVec(task, groupname, tagVec, hcclInfo.workSpaceAddr, hcclInfo.workSpaceMemSize));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SetAttachedStream(const ge::GETaskInfo &task) {
  if (task.rt_attached_streams.empty()) {
    HCCL_INFO("[HcomOpsKernelInfoStore][SetAttachedStream] attached stream is empty, so don't set");
    return HCCL_SUCCESS;
  }

  std::string group;
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  rtStream_t *streamPtr = const_cast<rtStream_t *>(task.rt_attached_streams.data());
  s32 streamNum = task.rt_attached_streams.size();
  uint32_t graphId = privateDefBuf->graphId;
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
    CHK_RET(HcomSetAttachedStream(group.c_str(), graphId, streamPtr, streamNum));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomSetAttachedStream(groupname, graphId, streamPtr, streamNum));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshInputAddr(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup,
                                                    const void *inputAddr, u64 inputMemSize, rtStream_t stream) {
  CHK_PRT_RET((shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE), HCCL_DEBUG("No need to refresh addr"), HCCL_SUCCESS);

  void *commInputPtr = nullptr;
  u64 commInputSize = 0;
  void *indirectInCCLbufPtr = nullptr;
  u64 indirectCommInputSize = 0;
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    if (commInputPtr == nullptr || commInputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(sGroup.c_str()));
      CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    }
    // 用户内存大于ccl buf时返回错误
    CHK_PRT_RET((inputMemSize > commInputSize),
                HCCL_ERROR("[Refresh][InputAddr]inputMemSize[0x%x] is greater than CCLbufferSize[0x%x]", inputMemSize,
                           commInputSize),
                HCCL_E_MEMORY);
    CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    if (indirectInCCLbufPtr == nullptr || indirectCommInputSize == 0) {
      CHK_RET(CreateIndirectCCLbuf());
      CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    }
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    if (commInputPtr == nullptr || commInputSize == 0) {
      CHK_RET(HcomCreateCommCCLbuffer(group));
      CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    }
    // 用户内存大于ccl buf时返回错误
    CHK_PRT_RET((inputMemSize > commInputSize),
                HCCL_ERROR("[Refresh][InputAddr]inputMemSize[0x%x] is greater than CCLbufferSize[0x%x]", inputMemSize,
                           commInputSize),
                HCCL_E_MEMORY);
    CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    if (indirectInCCLbufPtr == nullptr || indirectCommInputSize == 0) {
      CHK_RET(CreateIndirectCCLbuf());
      CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
    }
  }

  CHK_RET(hrtMemSyncCopy(indirectInCCLbufPtr, indirectCommInputSize, &commInputPtr, indirectCommInputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把用户输入buf一级指针指向的数据拷贝到中转buf一级指针指向的内存空间 rts提供新的拷贝task
  CHK_RET(hrtMemAsyncCopy(indirectInCCLbufPtr, inputMemSize, inputAddr, inputMemSize,
                          HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

  HCCL_DEBUG("indirectInCCLbufPtr=%p, inputAddr=%p, &commInputPtr=%p commInputPtr=%p", indirectInCCLbufPtr, inputAddr,
             &commInputPtr, commInputPtr);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshInputAddr(DevType deviceType, u32 shapeType, const int64_t &hcomComm,
                                                    const std::string &sGroup, const void *inputAddr, u64 inputOffset,
                                                    u64 curSize, bool secAddrCopyWithoutOffset, rtStream_t stream) {
  HCCL_DEBUG("[RefreshInputAddr] shapeType[%u]", shapeType);
  void *commInputPtr = nullptr;
  u64 commInputSize = 0;
  void *indirectInCCLbufPtr = nullptr;
  u64 indirectCommInputSize = 0;

  CHK_RET(GetInputCCLbufPtrAndIndirectInCCLbufPtr(hcomComm, sGroup, commInputPtr, commInputSize, indirectInCCLbufPtr,
                                                  indirectCommInputSize));

  CHK_RET(hrtMemSyncCopy(indirectInCCLbufPtr, indirectCommInputSize, &commInputPtr, indirectCommInputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把用户输入buf一级指针指向的数据，偏移拷贝到中转buf一级指针指向的内存空间 rts提供新的拷贝task
  if ((deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910) || secAddrCopyWithoutOffset) {
    CHK_RET(hrtMemAsyncCopy(indirectInCCLbufPtr, curSize, inputAddr, curSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

    HCCL_DEBUG("indirectInCCLbufPtr=%p, inputAddr=%p, &commInputPtr=%p commInputPtr=%p", indirectInCCLbufPtr, inputAddr,
               &commInputPtr, commInputPtr);
  } else {
    CHK_RET(hrtMemcpyAddrAsync(indirectInCCLbufPtr, commInputSize, 0, inputAddr, curSize, inputOffset, stream));

    HCCL_DEBUG(
        "indirectInCCLbufPtr=%p, inputAddr=%p, &commInputPtr=%p commInputPtr=%p curSize=%llu "
        "inputOffset=%llu",
        indirectInCCLbufPtr, inputAddr, &commInputPtr, commInputPtr, curSize, inputOffset);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshOutputAddr(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup,
                                                     void *outputAddr, u64 outputMemSize, rtStream_t stream) {
  CHK_PRT_RET((shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE), HCCL_INFO("No need to refresh addr"), HCCL_SUCCESS);

  void *commOutputPtr = nullptr;
  u64 commOutputSize = 0;
  void *indirectOutCCLbufPtr = nullptr;
  u64 indirectCommOutputSize = 0;

  CHK_RET(GetOutputCCLbufPtrAndIndirectOutCCLbufPtr(hcomComm, sGroup, commOutputPtr, commOutputSize,
                                                    indirectOutCCLbufPtr, indirectCommOutputSize));

  CHK_RET(hrtMemSyncCopy(indirectOutCCLbufPtr, indirectCommOutputSize, &commOutputPtr, indirectCommOutputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把中转buf一级指针指向的数据拷贝到用户输出一级指针指向的内存空间 rts提供新的拷贝task
  CHK_RET(hrtMemAsyncCopy(outputAddr, outputMemSize, indirectOutCCLbufPtr, outputMemSize,
                          HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

  HCCL_DEBUG("outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p", outputAddr,
             indirectOutCCLbufPtr, &commOutputPtr, commOutputPtr);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshOutputAddr(DevType deviceType, u32 shapeType, const int64_t &hcomComm,
                                                     const std::string &sGroup, void *outputAddr, u64 outputOffset,
                                                     u64 curSize, u64 outputMaxSize, bool secAddrCopyWithoutOffset,
                                                     rtStream_t stream) {
  HCCL_DEBUG("[RefreshOutputAddr] shapeType[%u]", shapeType);
  void *commOutputPtr = nullptr;
  u64 commOutputSize = 0;
  void *indirectOutCCLbufPtr = nullptr;
  u64 indirectCommOutputSize = 0;

  CHK_RET(GetOutputCCLbufPtrAndIndirectOutCCLbufPtr(hcomComm, sGroup, commOutputPtr, commOutputSize,
                                                    indirectOutCCLbufPtr, indirectCommOutputSize));

  CHK_RET(hrtMemSyncCopy(indirectOutCCLbufPtr, indirectCommOutputSize, &commOutputPtr, indirectCommOutputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把中转buf一级指针指向的数据,偏移拷贝到用户输出一级指针指向的内存空间 rts提供新的拷贝task
  if ((deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910) || secAddrCopyWithoutOffset) {
    CHK_RET(hrtMemAsyncCopy(outputAddr, curSize, indirectOutCCLbufPtr, curSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

    HCCL_DEBUG("outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p", outputAddr,
               indirectOutCCLbufPtr, &commOutputPtr, commOutputPtr);
  } else {
    CHK_RET(hrtMemcpyAddrAsync(outputAddr, outputMaxSize, outputOffset, indirectOutCCLbufPtr, curSize, 0, stream));

    HCCL_DEBUG(
        "outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p curSize=%llu "
        "outputOffset=%llu",
        outputAddr, indirectOutCCLbufPtr, &commOutputPtr, commOutputPtr, curSize, outputOffset);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::RefreshOutputAddr(DevType deviceType, u32 shapeType,
                                                     const std::string &sCollectiveType, const int64_t &hcomComm,
                                                     const std::string &sGroup, void *outputAddr, u64 outputOffset,
                                                     u64 curSize, u64 outputMaxSize, bool secAddrCopyWithoutOffset,
                                                     rtStream_t stream) {
  HCCL_DEBUG("[Refresh][OutputAddr] shapeType[%u]", shapeType);
  CHK_PRT_RET(sCollectiveType != HCCL_KERNEL_OP_TYPE_BROADCAST,
              HCCL_ERROR("[Refresh][OutputAddr]do not support the communication type[%s]", sCollectiveType.c_str()),
              HCCL_E_PARA);

  void *commInputPtr = nullptr;
  u64 commInputSize = 0;
  void *indirectInCCLbufPtr = nullptr;
  u64 indirectCommInputSize = 0;

  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetInCCLbuffer(sGroup.c_str(), commInputPtr, commInputSize));
    CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(GetInCCLbuffer(group, commInputPtr, commInputSize));
    CHK_RET(GetIndirectInCCLbuf(indirectInCCLbufPtr, indirectCommInputSize));
  }
  CHK_RET(hrtMemSyncCopy(indirectInCCLbufPtr, indirectCommInputSize, &commInputPtr, indirectCommInputSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // 把中转buf一级指针指向的数据,偏移拷贝到用户输出一级指针指向的内存空间 rts提供新的拷贝task
  if ((deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910) || secAddrCopyWithoutOffset) {
    CHK_RET(hrtMemAsyncCopy(outputAddr, curSize, indirectInCCLbufPtr, curSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream));

    HCCL_DEBUG("outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p", outputAddr,
               indirectInCCLbufPtr, &commInputPtr, commInputPtr);
  } else {
    CHK_RET(hrtMemcpyAddrAsync(outputAddr, outputMaxSize, outputOffset, indirectInCCLbufPtr, curSize, 0, stream));

    HCCL_DEBUG(
        "outputAddr=%p, indirectOutCCLbufPtr=%p, &commOutputPtr=%p commOutputPtr=%p curSize=%llu "
        "outputOffset=%llu",
        outputAddr, indirectInCCLbufPtr, &commInputPtr, commInputPtr, curSize, outputOffset);
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SaveReduceDumpTask(std::vector<ge::HcclDumpInfo> &geDumpInfo,
                                                      std::vector<hccl::HcclDumpInfo> &dumpInfo) {
  geDumpInfo.resize(dumpInfo.size());
  for (u32 index = 0; index < dumpInfo.size(); index++) {
    geDumpInfo[index].task_id = dumpInfo[index].task_id;
    geDumpInfo[index].stream_id = dumpInfo[index].stream_id;
    geDumpInfo[index].sub_task_type = dumpInfo[index].sub_task_type;
    geDumpInfo[index].output_addr = dumpInfo[index].output_addr;
    geDumpInfo[index].output_size = dumpInfo[index].output_size;
    geDumpInfo[index].input_addr = dumpInfo[index].input_addr;
    geDumpInfo[index].input_size = dumpInfo[index].input_size;
    HCCL_DEBUG("HCCLDumpInfo taskID[%u] stream[%u] subTaskType[%u]", geDumpInfo[index].task_id,
               geDumpInfo[index].stream_id, geDumpInfo[index].sub_task_type);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::ConfigHcclDumpDebugMode() {
  // 该接口通过key值，获取ge session级别option的value
  std::string dumpDebugValue;
  bool dumpDebug;
  s32 value = 0;
  ge::graphStatus geRet = ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_ENABLE_DUMP_DEBUG, dumpDebugValue);
  bool dumpFlag = Adx::AdumpIsDumpEnable(Adx::DumpType::OP_OVERFLOW);
  if (geRet == ge::GRAPH_SUCCESS || dumpFlag) {
    if (dumpFlag) {
      dumpDebug = dumpFlag;
    } else {
      CHK_RET(SalStrToInt(dumpDebugValue, HCCL_BASE_DECIMAL, value));  // 校验是否为有效值
      dumpDebug = value ? true : false;
    }
    HCCL_INFO("LoadTask: enable_dump_debug mode is [%d] (OPTION_EXEC_ENABLE_DUMP_DEBUG[%s]), value[%u].", dumpDebug,
              dumpDebugValue.c_str(), value);
    HcomSetDumpDebugMode(dumpDebug);
  } else {
    HCCL_WARNING("LoadTask: OPTION_EXEC_ENABLE_DUMP_DEBUG is false, hccl overflow detection mode is disabled");
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::InitHcom() {
  return HcomInitialize();
}

HcclResult HcomOpsKernelInfoStore::GetJsonProperty(const nlohmann::json &obj, const char *propName,
                                                   nlohmann::json &propValue) {
  /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
  if (obj.find(propName) == obj.end()) {
    HCCL_WARNING("json object has no property called %s", propName);
    return HCCL_E_NOT_FOUND;
  }
  propValue = obj[propName];
  CHK_PRT_RET(propValue.size() == 0, HCCL_ERROR("[Get][JsonProperty]get property[%s] size is zero", propName),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index,
                                                              const char *propName, u32 &propValue) {
  if (!obj.is_array() || index >= obj.size()) {
    HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] index[%u] is out of json object range",
               HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), index);
    return HCCL_E_PARA;
  }

  nlohmann::json subObj = obj.at(index);
  if (subObj.find(propName) == subObj.end()) {
    HCCL_WARNING("json object index[%u] has no property called %s", index, propName);
    return HCCL_E_NOT_FOUND;
  }
  if (subObj[propName].is_number_unsigned()) {
    propValue = subObj[propName];
    return HCCL_SUCCESS;
  } else {
    HCCL_ERROR(
        "[Get][JsonArrayMemberProperty]errNo[0x%016llx] json object property value of Name[%s] is "
        "not unsigned int!",
        HCOM_ERROR_CODE(HCCL_E_PARA), propName);
    return HCCL_E_PARA;
  }
}

HcclResult HcomOpsKernelInfoStore::GetRealRankIdFromMap(const u32 srcRankId, const std::string &rankMapJsonStr,
                                                        u32 &dstRankId) {
  u32 curLogicRankId;
  nlohmann::json fileContent;
  nlohmann::json rankMapList;

  if (rankMapJsonStr.empty()) {
    HCCL_ERROR("[Load][GetRealRankIdFromMap]errNo[0x%016llx] json string length is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
    return HCCL_E_PARA;
  }
  CHK_RET(SalParseInformation(fileContent, rankMapJsonStr));
  CHK_RET(GetJsonProperty(fileContent, "rank_map", rankMapList));
  for (u32 index = 0; index < rankMapList.size(); index++) {
    CHK_RET(GetJsonArrayMemberProperty(rankMapList, index, "model_rank_id", curLogicRankId));
    HCCL_DEBUG("[GetRealRankIdFromMap] srcRankId[%u] curLogicRankId[%u]", srcRankId, curLogicRankId);
    if (curLogicRankId == srcRankId) {
      CHK_RET(GetJsonArrayMemberProperty(rankMapList, index, "logic_rank_id", dstRankId));
      HCCL_DEBUG("[GetRealRankIdFromMap] dstRankId[%u]", dstRankId);
      return HCCL_SUCCESS;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::TransfromRealRankId(const ge::GETaskInfo &task) {
  // 根据是否需要need_map_rank，将算子的self_rank/peer_rank 转换为ge::OPTION_EXEC_RANK_MAP对应的world_rank
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  if (privateDefBuf->needMapRank) {
    std::string rankMapJsonStr;
    ge::graphStatus geRet = ge::GetThreadLocalContext().GetOption("ge.exec.rankMap", rankMapJsonStr);
    if (geRet == ge::GRAPH_SUCCESS) {
      CHK_RET(GetRealRankIdFromMap(privateDefBuf->srcRank, rankMapJsonStr, privateDefBuf->srcRank));
      HCCL_DEBUG("[TransfromRealRankId]srcRank:[%u]", privateDefBuf->srcRank);
      CHK_RET(GetRealRankIdFromMap(privateDefBuf->destRank, rankMapJsonStr, privateDefBuf->destRank));
      HCCL_DEBUG("[TransfromRealRankId]srcRank:[%u]", privateDefBuf->destRank);
      CHK_RET(GetRealRankIdFromMap(privateDefBuf->selfRank, rankMapJsonStr, privateDefBuf->selfRank));
      HCCL_DEBUG("[TransfromRealRankId]srcRank:[%u]", privateDefBuf->selfRank);
    } else {
      HCCL_ERROR("[Load][Task]errNo[0x%016llx] transfrom real rankid failed. rank map not exist.",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CheckOfflineDevTypeIsSame(const ge::GETaskInfo &task) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  HCCL_DEBUG("[CheckOfflineDevTypeIsSame] isOfflineComp[%u] devType[%u]", privateDefBuf->isOfflineComp,
             privateDefBuf->devType);
  if (privateDefBuf->isOfflineComp) {
    // 获取芯片类型
    DevType devType = HcomGetDeviceType();
    if (devType != privateDefBuf->devType) {
      HCCL_ERROR("[LoadTask]check offline device type failed. build dev type[%u] load dev type[%u]",
                 privateDefBuf->devType, devType);
    }
  }
  return HCCL_SUCCESS;
}

bool HcomOpsKernelInfoStore::IsRefresh(ge::GETaskInfo &task, const std::string &opType, u32 shapeType) {
  // gather broadcast alltoall alltoallv暂不支持按照needRefrsh变量刷新
  if (opType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || opType == HCCL_KERNEL_OP_TYPE_ALLGATHER ||
      opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER || opType == HCCL_KERNEL_OP_TYPE_REDUCE ||
      opType == HCCL_KERNEL_OP_TYPE_SEND || opType == HCCL_KERNEL_OP_TYPE_RECEIVE ||
      opType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    return task.needRefresh;
  } else {
    return (shapeType != ORIGINAL_GRAPH_KNOWNSHAPE_TYPE);
  }
}

HcclResult HcomOpsKernelInfoStore::GetTagVectorInfo(const ge::GETaskInfo &task, const std::string &sCollectiveType,
                                                    std::vector<std::string> &tagVec) {
  std::string tag;
  u32 loopMaxTime = 0;

  HcclResult ret = GenerateOpTagFromTaskInfo(task, sCollectiveType, tag, loopMaxTime);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Task]load task failed. (set generate op tag[%s] fail)", tag.c_str()), HCCL_E_INTERNAL);

  if (loopMaxTime == 0) {
    tagVec.push_back(tag);
  } else {
    for (u32 loopTime = 0; loopTime < loopMaxTime; loopTime++) {
      std::string tagTemp = tag + "_looptime_" + std::to_string(loopTime);
      tagVec.push_back(tagTemp);
    }
  }

  return HCCL_SUCCESS;
}

ge::Status HcomOpsKernelInfoStore::LoadTask(ge::GETaskInfo &task) {
  s32 deviceLogicId;  // 防止编译阶段和加载阶段deviceLogicId变更，此处重新刷一下
  CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

  // 离线编译场景下，在load task阶段初始化通信域、子group
  CHK_RET(CheckOfflineDevTypeIsSame(task));
  CHK_RET(InitHcom());
  CHK_RET(InitGroup());

  HcclResult ret;
  std::string sCollectiveType;
  std::string sTag;
  HCCL_INFO("LoadTask Start. taskID[%u].", task.id);

  std::string group;
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
  }
  CHK_RET(CheckCommunicatorValidity(group.c_str(), task));

  // 设定为算子信息库工作流程
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  CHK_RET(HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

  std::vector<ge::GETaskKernelHcclInfo> &hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] GETaskInfo size[%zu] in"
                         "HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA), hcclInfos.size()),
              HCCL_E_PARA);

  ge::GETaskKernelHcclInfo hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  CHK_PRT_RET((task.type != RT_MODEL_TASK_HCCL),
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] TaskType[%u] from"
                         "taskinfo is invalid.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), task.type),
              ge::INTERNAL_ERROR);
  ret = CheckPrivateDef(task);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. privateDef is invalid.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = TransfromRealRankId(task);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] transfrom real rank id failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = GetCollectiveTypeFromTaskInfo(hcclInfo, sCollectiveType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. (get collective type fail)", HCOM_ERROR_CODE(ret)),
      ge::INTERNAL_ERROR);

  std::vector<std::string> tagVec;
  ret = GetTagVectorInfo(task, sCollectiveType, tagVec);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. (get tag vector info fail)", HCOM_ERROR_CODE(ret)),
      ge::INTERNAL_ERROR);

  std::unique_lock<std::mutex> taskIdLock(taskIDtoTagMutex_);
  CHK_PRT_RET(taskIDtoTag_.find(task.id) != taskIDtoTag_.end(),
              HCCL_ERROR("[Load][Task]load task failed. (task id [%u] already exists)", task.id), ge::INTERNAL_ERROR);
  taskIDtoTag_[task.id] = tagVec;
  taskIdLock.unlock();
  HCCL_INFO("LoadTask Start. add taskID[%u].", task.id);

  u32 shapeType;
  ret = GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Load][Task]errNo[0x%016llx] get shapeType fail", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  if (!IsRefresh(task, sCollectiveType, shapeType)) {
    ret = SetKnownShapeWorkspaceResource(task, sCollectiveType, tagVec);
  } else {
    ret = SetUnknownShapeWorkspaceResource(task, sCollectiveType, tagVec);
  }
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. (set workspace/stream fail)", HCOM_ERROR_CODE(ret)),
      ge::INTERNAL_ERROR);

  ret = ConfigHcclDumpDebugMode();
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Load]errNo[0x%016llx] ConfigHcclDumpDebugMode error", HCOM_ERROR_CODE(ret)), ret);

  // 设置附属从流信息
  CHK_RET(SetAttachedStream(task));

  // 清空aiv buffer
  rtStream_t streamMain;
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    if (graphIdByGroup_.find(group) == graphIdByGroup_.end() || graphIdByGroup_[group] != privateDefBuf->graphId) {
      graphIdByGroup_[group] = privateDefBuf->graphId;
      CHK_RET(HcomClearAivSyncBuf(group.c_str(), true));
    }
  } else {
    if (graphIdByCommId_.find(comm) == graphIdByCommId_.end() || graphIdByCommId_[comm] != privateDefBuf->graphId) {
      graphIdByCommId_[comm] = privateDefBuf->graphId;
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomClearAivSyncBuf(groupname, true));
    }
  }
  CHK_RET(SetAivCoreLimit(task));
  ret = HCCLOpsKernel(task, sCollectiveType, tagVec);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. (load op[%s] fail)", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);

  // 把aicpu/mc2的kernel流注册给GE
  CHK_RET(HcomAicpuStreamRegister(task));

  // 记录 hcom 算子溢出的task信息
  hccl::HcclDumpInfo *hcclDumpInfoPtr = nullptr;
  s32 len = 0;
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcceGetandClearOverFlowTasks(group.c_str(), &hcclDumpInfoPtr, &len));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcceGetandClearOverFlowTasks(groupname, &hcclDumpInfoPtr, &len));
  }
  HCCL_DEBUG("hcclDumpInfo.size = %d", len);
  if (len > 0 && !task.kernelHcclInfo.empty()) {
    // HCOM场景下只会有一个kernelHcclInfo
    std::vector<hccl::HcclDumpInfo> hcclDumpInfo =
        std::vector<hccl::HcclDumpInfo>(hcclDumpInfoPtr, hcclDumpInfoPtr + len);
    SaveReduceDumpTask(task.kernelHcclInfo[0].hccl_dump_info, hcclDumpInfo);
    if (hcclDumpInfoPtr != nullptr) {
      free(hcclDumpInfoPtr);
      hcclDumpInfoPtr = nullptr;
    }
  }

  CHK_RET(HcomSetWorkflowMode(lastWorkflowMode));

  HCCL_INFO("LoadTask success taskID[%u].", task.id);

  return ge::SUCCESS;
}

ge::Status HcomOpsKernelInfoStore::UnloadTask(ge::GETaskInfo &task) {
  HCCL_INFO("UnloadTask Start taskID[%u].", task.id);
  s32 deviceLogicId;  // 防止编译阶段和加载阶段deviceLogicId变更，此处重新刷一下
  CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
  std::unique_lock<std::mutex> taskIdLock(taskIDtoTagMutex_);
  CHK_PRT_RET(taskIDtoTag_.count(task.id) == 0, HCCL_ERROR("UnloadTask taskID[%u] not found.", task.id),
              ge::INTERNAL_ERROR);
  std::vector<std::string> tagVec = taskIDtoTag_[task.id];
  taskIdLock.unlock();
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));

  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  CHK_RET(HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));

  for (std::vector<std::string>::iterator tagIter = tagVec.begin(); tagIter != tagVec.end(); tagIter++) {
    if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      std::string group;
      CHK_RET(GetGroupFromTaskInfo(task, group));
      CHK_RET(HcomUnloadTask(group.c_str(), (*tagIter).c_str()));
    } else {
      char *groupname = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
      CHK_RET(HcomUnloadTask(groupname, (*tagIter).c_str()));
    }
  }
  CHK_RET(HcomAicpuStreamUnRegister(task));
  CHK_RET(HcomSetWorkflowMode(lastWorkflowMode));

  taskIdLock.lock();
  taskIDtoTag_.erase(task.id);
  HCCL_INFO("UnloadTask success taskID[%u].", task.id);
  return ge::SUCCESS;
}

ge::Status HcomOpsKernelInfoStore::Finalize() {
  HCCL_INFO("finalize hccl kernel info store.");
  HcclResult ret = HcomReleaseSubComms();
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Release Sub comms failed."), ge::INTERNAL_ERROR);
  std::unique_lock<std::mutex> workSpaceLock(workSpaceMemMutex_);
  for (auto iter = workSpaceMemInfo_.begin(); iter != workSpaceMemInfo_.end();) {
    ret = hrtFree(std::get<0>(iter->second));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("free workSpace mem failed"), ge::INTERNAL_ERROR);
    workSpaceMemInfo_.erase(iter++);
  }
  crackMemPtr_.reset();
  crackMemPtrV2_.reset();
  indirectInCCLbufferPtr_.reset();
  indirectOutCCLbufferPtr_.reset();
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetHcclGroup(ge::GETaskInfo &task, std::string &sGroup) {
  int64_t comm = 0;
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, sGroup));
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &group));
    sGroup = group;
  }
  HCCL_INFO("%s success, comm:%llu, group:%s", __func__, comm, sGroup.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAicpuStreamRegister(ge::GETaskInfo &task) {
  // 获取通信域
  std::string group;
  CHK_RET(GetHcclGroup(task, group));

  // 检查该通信域注册的引用计数
  std::unique_lock<std::mutex> mapLock(orderedStreamMutex_);
  s32 devId = -1;
  CHK_RET(hrtGetDeviceRefresh(&devId));
  auto it = orderedStreamCount_[devId].find(group);
  if (it != orderedStreamCount_[devId].end()) {
    it->second++;
    HCCL_DEBUG("%s group:%s has been set, count:%llu", __func__, group.c_str(), it->second);
    return HCCL_SUCCESS;
  }
  orderedStreamCount_[devId].insert({group, 1});

  // 获取主流mode
  rtStream_t streamMain;
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  uint64_t streamMode = 0;
  CHK_RET(hrtStreamGetMode(streamMain, &streamMode));

  // 获取aicpuStream
  rtStream_t aicpuStream;
  HcclResult ret = HcomMc2AiCpuStreamAllocAndGet(group.c_str(), streamMode, &aicpuStream);
  // V2 NOT_FOUND 跳过该流程
  CHK_PRT_RET(ret == HCCL_E_NOT_FOUND, HCCL_WARNING("[HcomMc2AiCpuStreamAllocAndGet] group is not exist, skip."),
              HCCL_SUCCESS);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcomMc2AiCpuStreamAllocAndGet] error [%d]", ret), ret);
  CHK_PRT_RET(aicpuStream == nullptr, HCCL_ERROR("%s group:%s aicpuStream is null", __func__, group.c_str()),
              HCCL_E_PTR);

  // 将group对应的aicpu kernel流注册给GE
  u32 geRet = ge::HcomTopoInfo::Instance().SetGroupOrderedStream(devId, group.c_str(), aicpuStream);
  CHK_PRT_RET(
      geRet != ge::GRAPH_SUCCESS,
      HCCL_ERROR("%s SetGroupOrderedStream fail, group %s, aicpuStream %p", __func__, group.c_str(), aicpuStream),
      HCCL_E_INTERNAL);
  HCCL_INFO("%s success, group %s", __func__, group.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::HcomAicpuStreamUnRegister(ge::GETaskInfo &task) {
  // 获取通信域
  std::string group;
  CHK_RET(GetHcclGroup(task, group));

  std::unique_lock<std::mutex> mapLock(orderedStreamMutex_);
  s32 devId = -1;
  CHK_RET(hrtGetDeviceRefresh(&devId));
  auto it = orderedStreamCount_[devId].find(group);
  if (it == orderedStreamCount_[devId].end() || it->second == 0) {  // 未注册或已经解注册
    HCCL_DEBUG("%s group:%s has not been set", __func__, group.c_str());
  } else if (it->second > 1) {
    it->second--;
    HCCL_DEBUG("%s group:%s, count:%llu", __func__, group.c_str(), it->second);
  } else {
    it->second--;
    ge::HcomTopoInfo::Instance().UnsetGroupOrderedStream(devId, group.c_str());
    HCCL_INFO("%s success, group:%s count:%llu", __func__, group.c_str(), it->second);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::GetDataTypeFromTaskInfo(const ge::GETaskInfo &task, HcclDataType &dataType) const {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = static_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  if ((privateDefBuf->dataType >= HCCL_DATA_TYPE_INT8) && (privateDefBuf->dataType < HCCL_DATA_TYPE_RESERVED)) {
    dataType = HcclDataType(privateDefBuf->dataType);
  } else {
    HCCL_ERROR("[Get][DataType]errNo[0x%016llx] get date type from task info failed. dataType[%s] is invalid.",
               HCOM_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(privateDefBuf->dataType).c_str());
    return HCCL_E_PARA;
  }
  HCCL_INFO("get dataType[%s] from task info success. expect:[%d]-[%d]", GetDataTypeEnumStr(dataType).c_str(),
            HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SetGlobalWorkSpace(const int64_t &hcomComm, const string &sGroup,
                                                      std::vector<void *> globalWorkSpaceAddr) {
  void **globalWorkSpaceAddrPtr = static_cast<void **>(globalWorkSpaceAddr.data());
  if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(HcomSetGlobalWorkSpace(sGroup.c_str(), globalWorkSpaceAddrPtr, globalWorkSpaceAddr.size()));
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
    CHK_RET(HcomSetGlobalWorkSpace(group, globalWorkSpaceAddrPtr, globalWorkSpaceAddr.size()));
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::SetAivCoreLimit(const ge::GETaskInfo &task) {
  HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf = reinterpret_cast<HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  std::string group;
  int64_t comm = 0;

  CHK_PRT_RET(privateDefBuf->aivCoreLimit == 0,
              HCCL_ERROR("[HcomOpsKernelInfoStore][SetAivCoreLimit] aivCoreLimit shouledn't be 0"), HCCL_E_PARA);
  CHK_RET(GetCommFromTaskInfo(task, comm));
  if (comm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupFromTaskInfo(task, group));
    CHK_RET(HcomSetAivCoreLimit(group.c_str(), privateDefBuf->aivCoreLimit));
  } else {
    char *groupname = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(comm, &groupname));
    CHK_RET(HcomSetAivCoreLimit(groupname, privateDefBuf->aivCoreLimit));
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelInfoStore::CleanInterMemory(std::vector<std::int64_t> &crackAddr,
                                                    std::vector<std::int64_t> &crackSize, rtStream_t stream) {
  u64 crackMemSize = CRACK_MEMORY_SIZE;
  if (!initCrackMem_) {
    // 申请32B内存做清零操作
    char crackMemTemp[crackMemSize] = {0};
    void *crackMem = nullptr;
    HcclResult ret = hrtMalloc(&crackMem, crackMemSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS || crackMem == nullptr,
                HCCL_ERROR("[Malloc][Device]rt malloc device fail. return[%d]", ret), HCCL_E_INTERNAL);
    crackMemPtr_.reset(crackMem);
    CHK_RET(hrtMemSyncCopy(crackMemPtr_.get(), crackMemSize, crackMemTemp, crackMemSize,
                           HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    initCrackMem_ = true;
  }
  // 如果当前tensorlist只有一个tensor，TBE无法准确清理crackSize大小内存，用D2D的memcpy做清零
  // 待TBE处理完上述情况，此处删除
  if (crackSize.size() == 1) {
    if (crackSize[0] != 0) {
      CHK_RET(hrtMemAsyncCopy(reinterpret_cast<void *>(crackAddr[0]), crackSize[0], crackMemPtr_.get(), crackSize[0],
                              HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
    }
    crackAddr.clear();
    crackSize.clear();
  }

  // 非32B对齐的缝隙用D2D的memcpy做清零，并且从vector中剔除
  for (int i = 0; i < (int)crackSize.size(); i++) {
    if (crackSize[i] >= 0 && crackSize[i] % CRACK_MEMORY_SIZE != 0) {
      // 缝隙大小不为0时，D2Dmemcopy做清零
      if (crackSize[i] != 0) {
        CHK_RET(hrtMemAsyncCopy(reinterpret_cast<void *>(crackAddr[i]), crackSize[i], crackMemPtr_.get(), crackSize[i],
                                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
      }
      crackAddr.erase(crackAddr.begin() + i);
      crackSize.erase(crackSize.begin() + i);
      // 由于容器size-1，还按原来的i的话相当于自动右移一位而漏掉一个元素
      i--;
    }
  }
  // 当前的crackAddr和crackSize为缝隙size大于32B
  // 下发TBE清零算法
  if (crackAddr.size() != 0 && crackSize.size() != 0) {
    CHK_RET(TbeCleanIntervalMemory(crackAddr, crackSize, stream));
  }

  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomOpsKernelInfoStore::CleanInterMemoryV2(std::vector<std::int64_t> &crackSize,
                                                      std::vector<std::int64_t> &crackAddr, rtStream_t stream) {
  // 申请内存做清零操作
  auto maxIt = std::max_element(crackSize.begin(), crackSize.end());
  u64 maxSize = static_cast<u64>(*maxIt);
  if (maxCrackMemSizeV2_ < maxSize) {
    HCCL_INFO("[HcomOpsKernelInfoStore][%s] alloc mem with size[%llu]", __func__, maxSize);
    maxCrackMemSizeV2_ = maxSize;
    char crackMemTemp[maxCrackMemSizeV2_] = {0};
    void *crackMem = nullptr;
    HcclResult ret = hrtMalloc(&crackMem, maxCrackMemSizeV2_);
    CHK_PRT_RET(ret != HCCL_SUCCESS || crackMem == nullptr,
                HCCL_ERROR("[Malloc][Device]rt malloc device fail. return[%d]", ret), HCCL_E_INTERNAL);
    crackMemPtrV2_.reset(crackMem);
    CHK_RET(hrtMemSyncCopy(crackMemPtrV2_.get(), maxCrackMemSizeV2_, crackMemTemp, maxCrackMemSizeV2_,
                          HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
  }
  // 遍历内存块列表
  for (size_t i = 0; i < crackSize.size();) {
    int64_t currentSize = crackSize[i];
    int64_t currentAddr = crackAddr[i];
    if (currentSize > 0) {
      HCCL_INFO("[HcomOpsKernelInfoStore][%s] D2D memcpy async with size[%lld]", __func__, currentSize);
      void *dstPtr = reinterpret_cast<void *>(currentAddr);
      void *srcPtr = crackMemPtrV2_.get();

      // 执行整块异步内存拷贝
      CHK_RET(hrtMemAsyncCopy(dstPtr, currentSize, srcPtr, currentSize,
                              HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
      // 移除已处理的内存块
      crackSize.erase(crackSize.begin() + i);
      crackAddr.erase(crackAddr.begin() + i);
    } else {
      i++;
    }
  }
  return HCCL_SUCCESS;
}
#endif
}  // namespace hccl
