/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_tuning_hcom_ops_kernel_info_store.h"
#include <functional>
#include <securec.h>
#include "graph/types.h"
#include "auto_tuning_hcom_ops_kernel_builder.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "hccl/hcom.h"
#include "acl/acl_rt.h"
#include "hcom_graph_optimizer.h"
#include "framework/common/fmk_error_codes.h"
#include "hcom_acl_adapter.h"
#include "hcom_op_utils.h"

using namespace std;

namespace hccl {
AutoTuningHcomOpsKernelInfoStore::AutoTuningHcomOpsKernelInfoStore() {}

AutoTuningHcomOpsKernelInfoStore::~AutoTuningHcomOpsKernelInfoStore() {}

HcclResult AutoTuningHcomOpsKernelInfoStore::SetCustomKernelInfo(ge::OpInfo &opinfo,
                                                                 std::map<string, ge::OpInfo> &infos) const {
  opinfo.opKernelLib = AUTOTUNE_HCCL_OPS_LIB_NAME;
  for (u32 index = 0; index < AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.size(); index++) {
    HCCL_INFO("op[%s]: engine[%s] opKernelLib[%s] computeCost[%d] flagPartial[%d] flagAsync[%d]",
              AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE[index].c_str(), opinfo.engine.c_str(), opinfo.opKernelLib.c_str(),
              opinfo.computeCost, opinfo.flagPartial, opinfo.flagAsync);
    opinfo.isAtomic = false;
    infos.insert(std::pair<string, ge::OpInfo>(AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE[index], opinfo));
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::GetOriginalGraphShapeTypeFromTaskInfo(const ge::GETaskInfo &task,
                                                                                   u32 &shapeType) {
  AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf =
      reinterpret_cast<AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  shapeType = privateDefBuf->originalGraphShapeType;
  HCCL_INFO("get shapeType[%u] from task info success.", shapeType);
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::GetRankSizeFromTaskInfo(const ge::GETaskInfo &task, uint32_t &rankSize) {
  AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf =
      reinterpret_cast<AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  rankSize = privateDefBuf->rankSize;
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::CheckPrivateDef(const ge::GETaskInfo &task) {
  CHK_PTR_NULL(task.privateDef);
  if (task.privateDef == nullptr) {
    HCCL_ERROR("[Check][PrivateDef]errNo[0x%016llx] privateDefLen[%u] is not equal to [%zu]",
               HCOM_ERROR_CODE(HCCL_E_PARA), task.privateDefLen, sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF));
    return HCCL_E_PARA;
  }
  if (task.privateDefLen == sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF)) {
    return HCCL_SUCCESS;
  } else {
    HCCL_ERROR("[Check][PrivateDef]errNo[0x%016llx] privateDefLen[%u] is not equal to [%zu]",
               HCOM_ERROR_CODE(HCCL_E_PARA), task.privateDefLen, sizeof(AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF));
    return HCCL_E_PARA;
  }
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HCCLOpsKernel(const ge::GETaskInfo &task,
                                                           const std::string &sCollectiveType) {
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    CHK_RET(HcomBroadcastOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
    CHK_RET(HcomAllReduceOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
    CHK_RET(HcomAllGatherOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE) {
    CHK_RET(HcomReduceOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
    CHK_RET(HcomReduceScatterOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_SEND) {
    CHK_RET(HcomSendOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    CHK_RET(HcomReceiveOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV) {
    CHK_RET(HcomAlltoAllVOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
    CHK_RET(HcomAlltoAllVCOpKernel(task));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALL) {
    CHK_RET(HcomAlltoAllOpKernel(task));
  } else {
    HCCL_ERROR("[HCCL][OpsKernel]errNo[0x%016llx]Op type[%s] is invalid.", HCOM_ERROR_CODE(HCCL_E_PARA),
               sCollectiveType.c_str());
    return HCCL_E_PARA;
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomBroadcastOpKernel([[maybe_unused]] const ge::GETaskInfo &task) {
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomAlltoAllVOpKernel(const ge::GETaskInfo &task) {
  // 获取 hcom api 必须的参数
  CHK_RET(HcomAlltoAllOpKernel(task));
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::InitAlltoAllHostMem(HcclDataType recvType, u64 recvCount,
                                                                 void *hostMemEmpty) {
  switch (recvType) {
    case HCCL_DATA_TYPE_INT8:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<int8_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_INT16:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<int16_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_INT32:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<int32_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_FP16:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<uint16_t *>(hostMemEmpty) + i) = 0x3F80;  // 0x3f80 is 1.0f in float16
      }
      break;
    case HCCL_DATA_TYPE_FP32:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<float32_t *>(hostMemEmpty) + i) = 1.0f;
      }
      break;
    case HCCL_DATA_TYPE_INT64:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<int64_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_UINT64:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<uint64_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_UINT8:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<uint8_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_UINT16:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<uint16_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_UINT32:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<uint32_t *>(hostMemEmpty) + i) = 1;
      }
      break;
    case HCCL_DATA_TYPE_FP64:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<float64_t *>(hostMemEmpty) + i) = 1.0;
      }
      break;
    case HCCL_DATA_TYPE_BFP16:
      for (u64 i = 0; i < recvCount; i++) {
        *(reinterpret_cast<uint16_t *>(hostMemEmpty) + i) = 0x3F80;  // 0x3f80 is 1.0f in bfloat16
      }
      break;
    default:
      HCCL_ERROR("[AlltoAllVOp][Kernel] invalid recv data type[%s]", GetDataTypeEnumStr(recvType).c_str());
      return HCCL_E_NOT_SUPPORT;
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomAlltoAllVCOpKernel(const ge::GETaskInfo &task) {
  CHK_RET(HcomAlltoAllOpKernel(task));
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomAlltoAllOpKernel(const ge::GETaskInfo &task) {
  u32 shapeType;
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  CHK_PRT_RET((shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE),
              HCCL_WARNING("[AlltoAllVOp][Kernel]do not support unknownshape node."),  // runtime 2.0 下适配该情况
              HCCL_SUCCESS);
  rtStream_t streamMain;
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  ge::GETaskKernelHcclInfo hcclInfo;
  CHK_RET(GetHcclInfo(task, hcclInfo));
  CHK_PTR_NULL(hcclInfo.outputDataAddr);
  void *recvBuf = hcclInfo.outputDataAddr;

  HcclDataType recvType;
  CHK_RET(GetDataTypeFromTaskInfo(task, recvType));
  uint32_t unitSize;
  CHK_RET(SalGetDataTypeSize(recvType, unitSize));
  AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf =
      reinterpret_cast<AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
  u64 recvCount = privateDefBuf->outputBytes / unitSize;

  if (recvCount == 0) {
    HCCL_WARNING("[AutoTuning][OpsKernelInfoStore] Alltoall output count is 0, no task loaded.");
    return HCCL_SUCCESS;
  }

  AclHostMemPtr hostMemPtr;
  std::map<u64, std::map<HcclDataType, AclHostMemPtr>>::iterator itCount = mapCountTypeHostMem_.find(recvCount);
  if (itCount == mapCountTypeHostMem_.end()) {
    void *hostMem = nullptr;
    HcclResult ret = hrtMallocHost(&hostMem, unitSize * recvCount);
    CHK_PRT_RET(ret != HCCL_SUCCESS || hostMem == nullptr,
                HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
    hostMemPtr.reset(hostMem);
    mapCountTypeHostMem_[recvCount][recvType] = std::move(hostMemPtr);
  } else {
    std::map<HcclDataType, AclHostMemPtr>::iterator itType = mapCountTypeHostMem_[recvCount].find(recvType);
    if (itType == mapCountTypeHostMem_[recvCount].end()) {
      void *hostMem = nullptr;
      aclError ret = hrtMallocHost(&hostMem, unitSize * recvCount);
      CHK_PRT_RET(ret != ACL_SUCCESS || hostMem == nullptr,
                  HCCL_ERROR("[Malloc][Host]rt malloc host fail. return[%d]", ret), HCCL_E_INTERNAL);
      hostMemPtr.reset(hostMem);
      mapCountTypeHostMem_[recvCount][recvType] = std::move(hostMemPtr);
    }
  }

  CHK_RET(InitAlltoAllHostMem(recvType, recvCount, mapCountTypeHostMem_[recvCount][recvType].get()));

  CHK_RET(hrtMemAsyncCopy(reinterpret_cast<int8_t *>(recvBuf), unitSize * recvCount,
                          reinterpret_cast<int8_t *>(mapCountTypeHostMem_[recvCount][recvType].get()),
                          unitSize * recvCount, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, streamMain));
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomAllReduceOpKernel(const ge::GETaskInfo &task) {
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;
  u32 shapeType;

  ge::GETaskKernelHcclInfo hcclInfo;
  CHK_RET(GetHcclInfo(task, hcclInfo));
  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  /* 动态shap地址刷新 */
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));

  uint32_t unitSize;

  CHK_RET(SalGetDataTypeSize(dataType, unitSize));
  uint64_t dataSize = unitSize * count;
  if (shapeType != ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE) {
    CHK_RET(hrtMemAsyncCopy(reinterpret_cast<char *>(outputAddr), dataSize, reinterpret_cast<char *>(inputAddr),
                            dataSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, streamMain));
  } else {
    CHK_RET(hrtMemAsyncCopy(reinterpret_cast<char *>(outputAddr), dataSize, reinterpret_cast<char *>(inputAddr),
                            dataSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, streamMain));
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::GetHcclInfo(const ge::GETaskInfo &task,
                                                         ge::GETaskKernelHcclInfo &hcclInfo) {
  std::vector<ge::GETaskKernelHcclInfo> hcclInfos = task.kernelHcclInfo;
  CHK_PRT_RET((hcclInfos.size() != 1),
              HCCL_ERROR("[AllGatherOp][Kernel]errNo[0x%016llx] GETaskInfo size in"
                         "HCOM should be 1",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  hcclInfo = hcclInfos[0];  // HCOM场景下只会有一个
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomAllGatherOpKernel(const ge::GETaskInfo &task) {
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;
  u32 shapeType;
  uint32_t rankSize;
  ge::GETaskKernelHcclInfo hcclInfo;
  CHK_RET(GetHcclInfo(task, hcclInfo));

  // 获取 hcom api 必须的参数
  CHK_RET(GetRankSizeFromTaskInfo(task, rankSize));
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  /* 动态shap地址刷新 */
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  CHK_PRT_RET((shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE),
              HCCL_ERROR("[AllGatherOp][Kernel]not support"
                         "unknownshape node."),
              HCCL_E_NOT_SUPPORT);

  uint32_t unitSize;
  CHK_RET(SalGetDataTypeSize(dataType, unitSize));
  uint64_t dataSize = unitSize * count;
  for (uint32_t i = 0; i < rankSize; i++) {
    CHK_RET(hrtMemAsyncCopy(reinterpret_cast<char *>(outputAddr) + dataSize * i, dataSize,
                            reinterpret_cast<char *>(inputAddr), dataSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, streamMain));
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomReduceOpKernel(const ge::GETaskInfo &task) {
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;

  u32 shapeType;
  ge::GETaskKernelHcclInfo hcclInfo;
  CHK_RET(GetHcclInfo(task, hcclInfo));
  // 获取 hcom api 必须的参数
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  /* 动态shap地址刷新 */
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  CHK_PRT_RET((shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE),
              HCCL_ERROR("[ReduceOp][Kernel]not support"
                         "unknownshape graph."),
              HCCL_E_NOT_SUPPORT);

  uint32_t unitSize;
  CHK_RET(SalGetDataTypeSize(dataType, unitSize));
  uint64_t dataSize = unitSize * count;
  CHK_RET(hrtMemAsyncCopy(reinterpret_cast<char *>(outputAddr), dataSize, reinterpret_cast<char *>(inputAddr), dataSize,
                          HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, streamMain));
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomReduceScatterOpKernel(const ge::GETaskInfo &task) {
  rtStream_t streamMain;
  HcclDataType dataType;
  uintptr_t inputAddr = 0;
  uintptr_t outputAddr = 0;
  u64 count;

  u32 shapeType;
  uint32_t rankSize;
  ge::GETaskKernelHcclInfo hcclInfo;
  CHK_RET(GetHcclInfo(task, hcclInfo));

  // 获取 hcom api 必须的参数
  CHK_RET(GetRankSizeFromTaskInfo(task, rankSize));
  CHK_RET(GetDataTypeFromTaskInfo(task, dataType));
  CHK_RET(GetCountFromTaskInfo(hcclInfo, count));
  CHK_RET(GetStreamMainFromTaskInfo(task, streamMain));
  CHK_RET(GetInputAddrFromTaskInfo(hcclInfo, inputAddr));
  CHK_RET(GetOutputAddrFromTaskInfo(hcclInfo, outputAddr));
  /* 动态shap地址刷新 */
  CHK_RET(GetOriginalGraphShapeTypeFromTaskInfo(task, shapeType));
  CHK_PRT_RET((shapeType == ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE),
              HCCL_ERROR("[ReduceScatter][OpKernel]not support"
                         "unknownshape node."),
              HCCL_E_NOT_SUPPORT);

  uint32_t unitSize;
  CHK_RET(SalGetDataTypeSize(dataType, unitSize));
  uint64_t dataSize = unitSize * count;

  CHK_RET(hrtMemAsyncCopy(reinterpret_cast<char *>(outputAddr), dataSize, reinterpret_cast<char *>(inputAddr), dataSize,
                          HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, streamMain));

  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomReceiveOpKernel([[maybe_unused]] const ge::GETaskInfo &task) {
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::HcomSendOpKernel([[maybe_unused]] const ge::GETaskInfo &task) {
  return HCCL_SUCCESS;
}

ge::Status AutoTuningHcomOpsKernelInfoStore::LoadTask(ge::GETaskInfo &task) {
  HcclResult ret;
  std::string sCollectiveType;
  std::string sTag;
  HCCL_INFO("LoadTask Start.");

  // 设定为算子信息库工作流程
  ge::GETaskKernelHcclInfo hcclInfo;
  CHK_RET(GetHcclInfo(task, hcclInfo));
  CHK_PRT_RET((task.type != RT_MODEL_TASK_HCCL),
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] TaskType[%u] from"
                         "taskinfo is invalid.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), task.type),
              ge::INTERNAL_ERROR);
  ret = CheckPrivateDef(task);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. privateDef is invalid.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);
  ret = GetCollectiveTypeFromTaskInfo(hcclInfo, sCollectiveType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. (get collective type fail)", HCOM_ERROR_CODE(ret)),
      ge::INTERNAL_ERROR);
  // 执行 hcom 算子处理
  ret = HCCLOpsKernel(task, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Load][Task]errNo[0x%016llx] load task failed. (load op[%s] fail)", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);
  HCCL_INFO("LoadTask success.");
  return ge::SUCCESS;
}

ge::Status AutoTuningHcomOpsKernelInfoStore::UnloadTask([[maybe_unused]] ge::GETaskInfo &task) {
  HCCL_INFO("UnloadTask skip.");
  return ge::SUCCESS;
}

HcclResult AutoTuningHcomOpsKernelInfoStore::GetDataTypeFromTaskInfo(const ge::GETaskInfo &task,
                                                                     HcclDataType &dataType) const {
  AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *privateDefBuf =
      static_cast<AUTO_TUNING_HCCL_KERNEL_INFO_PRIVATE_DEF *>(task.privateDef);
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
}  // namespace hccl
