/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_launch_kernel.h"
#include "hccl/hcom.h"
#include "hcom_ops_kernel_info_store.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "acl/acl_rt.h"
#include "hcom_acl_adapter.h"
#include "adapter_dlhcclfunc.h"

namespace hccl {
enum class HcomOpLaunchInput {
  OP_ARGS,
  STREAM,
  INPUT_NUM,
  OUTPUT_NUM,
};

enum class RecvOpLaunchInput {
  OP_ARGS,
  STREAM,
  ALLOCATOR,
  DATATYPE,
};

uint64_t RoundUp(const uint64_t originValue, const uint64_t multiple) {
  if (multiple == 0) {
    return 0;
  }
  return (originValue + multiple - 1) / multiple * multiple;
}

HcclResult GetCountByShapeAlign(const gert::Shape &shape, HcclDataType dataType, uint64_t &count) {
  const uint64_t memAlignSize = 512;
  uint32_t unitSize = SIZE_TABLE[dataType];
  count = RoundUp(shape.GetShapeSize() * unitSize, memAlignSize) / unitSize;
  HCCL_DEBUG("[GetCountByShapeAlign]shape size:%ld, count:%llu", shape.GetShapeSize(), count);
  return HCCL_SUCCESS;
}

HcclResult GetSendCountByShapeAlign(const gert::Shape &shape, HcclDataType dataType, uint64_t &count) {
  const uint64_t memAlignSize = 32;
  uint32_t unitSize = SIZE_TABLE[dataType];
  count = RoundUp(shape.GetShapeSize() * unitSize, memAlignSize) / unitSize;
  HCCL_DEBUG("[GetSendCountByShapeAlign]shape size:%ld, count:%llu", shape.GetShapeSize(), count);
  return HCCL_SUCCESS;
}

HcclResult GetCountByShape(const gert::Shape &shape, HcclDataType dataType, uint64_t &count) {
  count = shape.GetShapeSize();
  HCCL_DEBUG("[GetCountByShape]dataType:%d, count:%llu", dataType, count);
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomAllGatherKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  CHK_PRT_RET((launchArgs.inputNum != 1), HCCL_ERROR("HcomAllGather input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShape(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchAllGatherKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchAllGatherKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  uint64_t count = inputStruct->count;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceAllGather(inputAddrs[0], outputAddrs[0], count, launchArgs.opAttr.dataType, hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomAllGatherKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  uint64_t count = 0;
  CHK_RET(GetCountByShape(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));
  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  inputStruct->count = count;
  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchAllGatherKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;
  uint64_t count = inputStruct->count;

  CHK_RET(
      HcceAllGather(inputAddrs[0], outputAddrs[0], count, launchArgs.opAttr.dataType, hcclCommPtr, launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomAllGatherVKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  uint64_t sendCount;
  CHK_RET(GetCountByShape(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, sendCount));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->sendCount = sendCount;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchAllGatherVKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                      std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  u32 rankSize = 0;
  CHK_RET(HcomGetRankSize(launchArgs.opAttr.group, &rankSize));
  uint64_t dimzero = static_cast<uint64_t>(launchArgs.inputShapes[0].GetDim(0));
  uint64_t multiplier = (dimzero == 0) ? 1 : (inputStruct->sendCount / dimzero);
  uint64_t *recvcounts = static_cast<uint64_t *>(const_cast<void *>(inputAddrs[2]));
  std::vector<uint64_t> recvDispls;
  uint64_t tempSum = 0;
  for (u32 i = 0; i < rankSize; i++) {
    recvcounts[i] *= multiplier;
    recvDispls.push_back(tempSum);
    tempSum += recvcounts[i];
  }
  const void *recvDisplsPtr = static_cast<const void *>(recvDispls.data());
  const void *recvcountsPtr = static_cast<const void *>(recvcounts);

  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));

  CHK_RET(HcceAllGatherV(inputAddrs[0], inputStruct->sendCount, outputAddrs[0], recvcountsPtr, recvDisplsPtr,
                         launchArgs.opAttr.dataType, hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

HcclResult HcomGetBatchAllReduceInfo(const HcomOpLaunchArgs &launchArgs, uint64_t maxCommCount,
                                     std::vector<uint32_t> &sliceIdxs, std::vector<uint64_t> &inputsCount) {
  uint32_t sliceIdx = 0;
  uint32_t sliceSum = 0;
  uint64_t commCount = 0;
  inputsCount.resize(launchArgs.inputNum);
  for (uint32_t i = 0; i < launchArgs.inputNum; i++) {
    uint64_t count;
    CHK_RET(GetCountByShape(launchArgs.inputShapes[i], launchArgs.opAttr.dataType, count));
    HCCL_DEBUG("HcomAllReduceKernel: input[%u] count: %llu", i, count);
    uint64_t outCount;
    CHK_RET(GetCountByShape(launchArgs.outputShapes[i], launchArgs.opAttr.dataType, outCount));
    CHK_PRT_RET((count != outCount),
                HCCL_ERROR("HcomAllReduceKernel: input[%u] count: %llu is different with output[%u] count: %llu", i,
                           count, i, outCount),
                HCCL_E_PARA);

    commCount += count;
    inputsCount[i] = count;
    if (commCount <= maxCommCount) {
      sliceIdx++;
    } else {
      if (sliceIdx == 0) {
        sliceIdx++;
        HCCL_DEBUG("HcomAllReduceKernel: slice[%zu] has %u tensors.", sliceIdxs.size(), sliceIdx);
        sliceIdxs.push_back(sliceIdx);
        sliceSum += sliceIdx;
        commCount = 0;
        sliceIdx = 0;
      } else {
        HCCL_DEBUG("HcomAllReduceKernel: slice[%zu] has %u tensors.", sliceIdxs.size(), sliceIdx);
        sliceIdxs.push_back(sliceIdx);
        sliceSum += sliceIdx;
        commCount = count;
        sliceIdx = 1;
      }
    }
  }

  if (sliceSum < launchArgs.inputNum) {
    HCCL_DEBUG("HcomAllReduceKernel: slice[%zu] has %u tensors.", sliceIdxs.size(), (launchArgs.inputNum - sliceSum));
    sliceIdxs.push_back(launchArgs.inputNum - sliceSum);
  }
  return HCCL_SUCCESS;
}

// 将从inputsOffset开始的连续inputsNum个通信算子输入tensor copy至CCL buffer，并输出copy完成的数据量
HcclResult HcomCopyInputsToCCLbuff(const HcomOpLaunchArgs &launchArgs, uint32_t inputsNum, uint32_t inputsOffset,
                                   std::vector<uint64_t> &inputsCount, void *cclBuff, uint64_t cclBuffSize,
                                   uint64_t &commCount, std::vector<void *> &inputAddrs) {
  uint32_t unitSize = SIZE_TABLE[launchArgs.opAttr.dataType];
  u8 *dstBuff = static_cast<u8 *>(cclBuff);
  uint64_t maxDstSize = cclBuffSize;
  for (uint32_t j = 0; j < inputsNum; j++) {
    uint32_t inputIdx = inputsOffset + j;
    CHK_RET(hrtMemAsyncCopy(dstBuff, maxDstSize, inputAddrs[inputIdx], inputsCount[inputIdx] * unitSize,
                            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, launchArgs.stream));
    HCCL_DEBUG("HcomAllReduceKernel: copy input[%u] addr[%p] len[%u] to cclbuffer[%p].", inputIdx, inputAddrs[inputIdx],
               inputsCount[inputIdx] * unitSize, dstBuff);
    dstBuff += inputsCount[inputIdx] * unitSize;
    maxDstSize -= inputsCount[inputIdx] * unitSize;
    commCount += inputsCount[inputIdx];
  }
  return HCCL_SUCCESS;
}

// 将CCL buffer中的数据copy至从outputsOffset开始的连续outputsNum个通信算子输出tensor
HcclResult HcomCopyCCLbuffToOutnputs(const HcomOpLaunchArgs &launchArgs, uint32_t outputsNum, uint32_t outputsOffset,
                                     std::vector<uint64_t> &outputsCount, void *cclBuff,
                                     std::vector<void *> &outputAddrs) {
  uint32_t unitSize = SIZE_TABLE[launchArgs.opAttr.dataType];
  u8 *srcBuff = static_cast<u8 *>(cclBuff);
  for (uint32_t j = 0; j < outputsNum; j++) {
    uint32_t outputIdx = outputsOffset + j;
    CHK_RET(hrtMemAsyncCopy(outputAddrs[outputIdx], outputsCount[outputIdx] * unitSize, srcBuff,
                            outputsCount[outputIdx] * unitSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
                            launchArgs.stream));
    HCCL_DEBUG("HcomAllReduceKernel: copy cclbuffer[%p] to output[%u] addr[%p] len[%u].", srcBuff, outputIdx,
               outputAddrs[outputIdx], outputsCount[outputIdx] * unitSize);
    srcBuff += outputsCount[outputIdx] * unitSize;
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomAllReduceKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomAllReduce input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);
  void *commInputPtr = nullptr;
  u64 commInputSize = 0;
  void *commOutputPtr = nullptr;
  u64 commOutputSize = 0;
  CHK_RET(GetInCCLbuffer(launchArgs.opAttr.group, commInputPtr, commInputSize));
  CHK_RET(GetOutCCLbuffer(launchArgs.opAttr.group, commOutputPtr, commOutputSize));
  uint64_t maxCommCount = commInputSize / SIZE_TABLE[launchArgs.opAttr.dataType];

  std::vector<uint32_t> sliceIdxs;  // 按照cclbuffer size将输入tensors分段，每一段的tensor个数
  std::vector<uint64_t> inputsCount;
  CHK_RET(HcomGetBatchAllReduceInfo(launchArgs, maxCommCount, sliceIdxs, inputsCount));
  inputStruct->launchArgs = launchArgs;
  inputStruct->commInputPtr = commInputPtr;
  inputStruct->commInputSize = commInputSize;
  inputStruct->commOutputPtr = commOutputPtr;
  inputStruct->inputsCount = inputsCount;
  inputStruct->sliceIdxs = sliceIdxs;
  inputStruct->hcclComm = commHandle;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchAllReduceKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchAllReduceKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  std::vector<uint32_t> sliceIdxs = inputStruct->sliceIdxs;
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  std::vector<uint64_t> inputsCount = inputStruct->inputsCount;
  HcclComm hcclComm = inputStruct->hcclComm;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  uint32_t inputOffset = 0;
  for (uint32_t i = 0; i < sliceIdxs.size(); i++) {
    if (sliceIdxs[i] == 1) {
      u8 *curInputPtr = static_cast<u8 *>(inputAddrs[inputOffset]);
      u8 *curOutputPtr = static_cast<u8 *>(outputAddrs[inputOffset]);

      CHK_RET(HcceAllReduce(curInputPtr, curOutputPtr, inputsCount[inputOffset], launchArgs.opAttr.dataType,
                            launchArgs.opAttr.op.allreduce.reduction, hcclComm, launchArgs.stream));
    } else {
      uint64_t commCount = 0;
      // step1 将allreduce的输入tensor copy到in cclbuffer
      CHK_RET(HcomCopyInputsToCCLbuff(launchArgs, sliceIdxs[i], inputOffset, inputsCount, inputStruct->commInputPtr,
                                      inputStruct->commInputSize, commCount, inputAddrs));

      // step2 执行allreduce
      CHK_RET(HcceAllReduce(inputStruct->commInputPtr, inputStruct->commOutputPtr, commCount,
                            launchArgs.opAttr.dataType, launchArgs.opAttr.op.allreduce.reduction, hcclComm,
                            launchArgs.stream));

      // step3 将out cclbuffer的内容copy到allreduce的输出tensor
      CHK_RET(HcomCopyCCLbuffToOutnputs(launchArgs, sliceIdxs[i], inputOffset, inputsCount, inputStruct->commOutputPtr,
                                        outputAddrs));
    }

    inputOffset += sliceIdxs[i];
  }
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomAllReduceKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomAllReduce input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  void *commInputPtr = nullptr;
  u64 commInputSize = 0;
  void *commOutputPtr = nullptr;
  u64 commOutputSize = 0;
  std::vector<uint32_t> sliceIdxs;  // 按照cclbuffer size将输入tensors分段，每一段的tensor个数
  std::vector<uint64_t> inputsCount;
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  CHK_RET(GetInCCLbuffer(launchArgs.opAttr.group, commInputPtr, commInputSize));
  CHK_RET(GetOutCCLbuffer(launchArgs.opAttr.group, commOutputPtr, commOutputSize));
  uint64_t maxCommCount = commInputSize / SIZE_TABLE[launchArgs.opAttr.dataType];

  CHK_RET(HcomGetBatchAllReduceInfo(launchArgs, maxCommCount, sliceIdxs, inputsCount));
  inputStruct->launchArgs = launchArgs;
  inputStruct->commInputPtr = commInputPtr;
  inputStruct->commInputSize = commInputSize;
  inputStruct->commOutputPtr = commOutputPtr;
  inputStruct->inputsCount = inputsCount;
  inputStruct->sliceIdxs = sliceIdxs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchAllReduceKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs) {
  std::vector<uint32_t> sliceIdxs = inputStruct->sliceIdxs;
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  std::vector<uint64_t> inputsCount = inputStruct->inputsCount;
  void *hcclCommPtr = inputStruct->hcclCommPtr;
  uint32_t inputOffset = 0;
  for (uint32_t i = 0; i < sliceIdxs.size(); i++) {
    if (sliceIdxs[i] == 1) {
      u8 *curInputPtr = static_cast<u8 *>(inputAddrs[inputOffset]);
      u8 *curOutputPtr = static_cast<u8 *>(outputAddrs[inputOffset]);

      CHK_RET(HcceAllReduce(curInputPtr, curOutputPtr, inputsCount[inputOffset], launchArgs.opAttr.dataType,
                            launchArgs.opAttr.op.allreduce.reduction, hcclCommPtr, launchArgs.stream));
    } else {
      uint64_t commCount = 0;
      // step1 将allreduce的输入tensor copy到in cclbuffer
      CHK_RET(HcomCopyInputsToCCLbuff(launchArgs, sliceIdxs[i], inputOffset, inputsCount, inputStruct->commInputPtr,
                                      inputStruct->commInputSize, commCount, inputAddrs));

      // step2 执行allreduce
      CHK_RET(HcceAllReduce(inputStruct->commInputPtr, inputStruct->commOutputPtr, commCount,
                            launchArgs.opAttr.dataType, launchArgs.opAttr.op.allreduce.reduction, hcclCommPtr,
                            launchArgs.stream));

      // step3 将out cclbuffer的内容copy到allreduce的输出tensor
      CHK_RET(HcomCopyCCLbuffToOutnputs(launchArgs, sliceIdxs[i], inputOffset, inputsCount, inputStruct->commOutputPtr,
                                        outputAddrs));
    }

    inputOffset += sliceIdxs[i];
  }
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomBroadcastKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomBroadcastKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomBroadcast input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShapeAlign(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchBroadcastKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchBroadcastKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  uint64_t count = inputStruct->count;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceBroadcast(inputAddrs[0], count, launchArgs.opAttr.dataType, launchArgs.opAttr.op.broadcast.root, hcclComm,
                        launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomBroadcastKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomBroadcast input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShapeAlign(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));
  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  inputStruct->count = count;
  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchBroadcastKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       [[maybe_unused]] std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;
  uint64_t count = inputStruct->count;

  CHK_RET(HcceBroadcast(inputAddrs[0], count, launchArgs.opAttr.dataType, launchArgs.opAttr.op.broadcast.root,
                        hcclCommPtr, launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomReduceScatterKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomReduceScatterKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  CHK_PRT_RET((launchArgs.inputNum != 1),
              HCCL_ERROR("HcomReduceScatter input num[%u] is invalid.", launchArgs.inputNum), HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShape(launchArgs.outputShapes[0], launchArgs.opAttr.dataType, count));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchReduceScatterKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                         std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchReduceScatterKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  uint64_t count = inputStruct->count;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceReduceScatter(inputAddrs[0], outputAddrs[0], count, launchArgs.opAttr.dataType,
                            launchArgs.opAttr.op.reducescatter.reduction, hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterVKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));

  uint64_t recvCount;
  CHK_RET(GetCountByShape(launchArgs.outputShapes[0], launchArgs.opAttr.dataType, recvCount));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->recvCount = recvCount;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchReduceScatterVKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                          std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  u32 rankSize = 0;
  CHK_RET(HcomGetRankSize(launchArgs.opAttr.group, &rankSize));
  // 计算第0维后的维度乘积
  uint64_t dimzero = static_cast<uint64_t>(launchArgs.outputShapes[0].GetDim(0));
  uint64_t multiplier = (dimzero == 0) ? 1 : (inputStruct->recvCount / dimzero);
  uint64_t *sendCounts = static_cast<uint64_t *>(const_cast<void *>(inputAddrs[INPUT_INDEX_2]));
  vector<int64_t> sendDispls;
  int64_t tmpCount = 0;
  for (size_t i = 0; i < rankSize; i++) {
    sendCounts[i] *= multiplier;
    sendDispls.push_back(tmpCount);
    tmpCount += sendCounts[i];
  }
  const void *sendDisplsPtr = static_cast<const void *>(sendDispls.data());
  const void *sendCountsPtr = static_cast<const void *>(sendCounts);
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceReduceScatterV(inputAddrs[0], sendCountsPtr, sendDisplsPtr, outputAddrs[0], inputStruct->recvCount,
                             launchArgs.opAttr.dataType, launchArgs.opAttr.op.reducescatterv.reduction, hcclComm,
                             launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomReduceScatterKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  CHK_PRT_RET((launchArgs.inputNum != 1),
              HCCL_ERROR("HcomReduceScatter input num[%u] is invalid.", launchArgs.inputNum), HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShape(launchArgs.outputShapes[0], launchArgs.opAttr.dataType, count));
  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  inputStruct->count = count;
  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchReduceScatterKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                           std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;
  uint64_t count = inputStruct->count;

  CHK_RET(HcceReduceScatter(inputAddrs[0], outputAddrs[0], count, launchArgs.opAttr.dataType,
                            launchArgs.opAttr.op.reducescatter.reduction, hcclCommPtr, launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomAllToAllVKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomAllToAllVKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchAllToAllVKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchAllToAllVKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceAlltoAllV(inputAddrs[0], inputAddrs[1], inputAddrs[INPUT_INDEX_2], launchArgs.opAttr.dataType,
                        outputAddrs[0], inputAddrs[INPUT_INDEX_3], inputAddrs[INPUT_INDEX_4],
                        launchArgs.opAttr.op.alltoallv.recvType, hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomAllToAllVKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchAllToAllVKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;

  CHK_RET(HcceAlltoAllV(inputAddrs[0], inputAddrs[1], inputAddrs[INPUT_INDEX_2], launchArgs.opAttr.dataType,
                        outputAddrs[0], inputAddrs[INPUT_INDEX_3], inputAddrs[INPUT_INDEX_4],
                        launchArgs.opAttr.op.alltoallv.recvType, hcclCommPtr, launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomAllToAllVCKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomAllToAllVCKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchAllToAllVCKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                      std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchAllToAllVCKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceAlltoAllVC(inputAddrs[0], inputAddrs[1], launchArgs.opAttr.dataType, outputAddrs[0],
                         launchArgs.opAttr.dataType, hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomAllToAllVCKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchAllToAllVCKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                        std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;

  CHK_RET(HcceAlltoAllVC(inputAddrs[0], inputAddrs[1], launchArgs.opAttr.dataType, outputAddrs[0],
                         launchArgs.opAttr.dataType, hcclCommPtr, launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomAllToAllKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  HcclComm commHandle = nullptr;
  uint64_t sendCount;
  uint64_t recvCount;
  CHK_RET(GetCountByShape(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, sendCount));
  CHK_RET(GetCountByShape(launchArgs.outputShapes[0], launchArgs.opAttr.dataType, recvCount));
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->sendCount = sendCount;
  inputStruct->recvCount = recvCount;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchAllToAllKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                    std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  u32 rankSize = 0;
  CHK_RET(HcomGetRankSize(launchArgs.opAttr.group, &rankSize));
  uint64_t sendCount = (rankSize != 0) ? (inputStruct->sendCount / rankSize) : inputStruct->sendCount;
  uint64_t recvCount = (rankSize != 0) ? (inputStruct->recvCount / rankSize) : inputStruct->recvCount;
  HCCL_INFO("HcomLaunchAllToAllKernel: rankSize[%u], sendCount[%llu], recvCount[%llu], input sendCount[%llu], input recvCount[%llu]",
    rankSize, sendCount, recvCount, inputStruct->sendCount, inputStruct->recvCount);
  HcclComm hcclComm = inputStruct->hcclComm;
  CHK_RET(HcomSetAivCoreLimit(launchArgs.opAttr.group, launchArgs.opAttr.aivCoreLimit));
  CHK_RET(HcceAlltoAll(inputAddrs[0], sendCount, launchArgs.opAttr.dataType, outputAddrs[0],
                       recvCount, launchArgs.opAttr.dataType, hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomSendKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomSend input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  // host内存
  u32 shapeCount = 1 + launchArgs.inputShapes[0].GetDimNum();
  std::vector<int64_t> sendInfo(shapeCount, 0);
  sendInfo[0] = shapeCount - 1;
  for (u32 i = 1; i < sendInfo.size(); i++) {
    sendInfo[i] = launchArgs.inputShapes[0].GetDim(i - 1);
  }

  // dev内存
  auto deleter = [](void *dst) {
    if (dst != nullptr) {
      CHK_PRT(hrtFree(dst));
      dst = nullptr;
    }
  };

  void *sendShapeMemPtr = nullptr;
  u32 shapeSize = sizeof(int64_t) * shapeCount;
  u32 totalSize = sizeof(int64_t) * SHAPE_INFO_COUNT;
  CHK_RET(hrtMalloc(&sendShapeMemPtr, totalSize));
  std::unique_ptr<void, decltype(deleter)> sendShapeMemPtrUnique(sendShapeMemPtr, deleter);

  // host2dev拷贝
  CHK_RET(hrtMemSyncCopy(sendShapeMemPtr, shapeSize, sendInfo.data(), shapeSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // send
  CHK_RET(HcceSend(sendShapeMemPtr, SHAPE_INFO_COUNT, HCCL_DATA_TYPE_INT64, launchArgs.opAttr.op.send.destRank,
                   hcclCommPtr, launchArgs.stream));

  CHK_RET(hcclStreamSynchronize(launchArgs.stream));

  uint64_t count = 0;
  CHK_RET(GetSendCountByShapeAlign(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}
#endif

HcclResult HcomSendKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomSendKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomSend input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  // host内存
  u32 shapeCount = 1 + launchArgs.inputShapes[0].GetDimNum();
  std::vector<int64_t> sendInfo(shapeCount, 0);
  sendInfo[0] = shapeCount - 1;
  for (u32 i = 1; i < sendInfo.size(); i++) {
    sendInfo[i] = launchArgs.inputShapes[0].GetDim(i - 1);
  }

  // dev内存
  auto deleter = [](void *dst) {
    if (dst != nullptr) {
      CHK_PRT(hrtFree(dst));
      dst = nullptr;
    }
  };

  void *sendShapeMemPtr = nullptr;
  u32 shapeSize = sizeof(int64_t) * shapeCount;
  u32 totalSize = sizeof(int64_t) * SHAPE_INFO_COUNT;
  CHK_RET(hrtMalloc(&sendShapeMemPtr, totalSize));
  std::unique_ptr<void, decltype(deleter)> sendShapeMemPtrUnique(sendShapeMemPtr, deleter);

  // host2dev拷贝
  CHK_RET(hrtMemSyncCopy(sendShapeMemPtr, shapeSize, sendInfo.data(), shapeSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

  // send
  CHK_RET(HcceSend(sendShapeMemPtr, SHAPE_INFO_COUNT, HCCL_DATA_TYPE_INT64, launchArgs.opAttr.op.send.destRank,
                   commHandle, launchArgs.stream));

  CHK_RET(hcclStreamSynchronize(launchArgs.stream));

  uint64_t count = 0;
  CHK_RET(GetSendCountByShapeAlign(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchSendKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                  [[maybe_unused]] std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;

  CHK_RET(HcceSend(inputAddrs[0], inputStruct->count, launchArgs.opAttr.dataType, launchArgs.opAttr.op.send.destRank,
                   hcclCommPtr, launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

HcclResult HcomLaunchSendKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchSendKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif

  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;

  CHK_RET(HcceSend(inputAddrs[0], inputStruct->count, launchArgs.opAttr.dataType, launchArgs.opAttr.op.send.destRank,
                   hcclComm, launchArgs.stream));
  return HCCL_SUCCESS;
}

HcclResult HcomReduceKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomReduceKernelV2(launchArgs, inputStruct);
  }
#endif
  HcclComm commHandle = nullptr;
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomReduce input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShape(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));

  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclComm = commHandle;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}

HcclResult HcomLaunchReduceKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                  std::vector<void *> &outputAddrs) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomLaunchReduceKernelV2(inputStruct, inputAddrs, outputAddrs);
  }
#endif
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  HcclComm hcclComm = inputStruct->hcclComm;

  CHK_RET(HcceReduce(inputAddrs[0], outputAddrs[0], inputStruct->count, launchArgs.opAttr.dataType,
                     launchArgs.opAttr.op.reduce.reduction, launchArgs.opAttr.op.reduce.root, hcclComm,
                     launchArgs.stream));
  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomReduceKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct) {
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));
  CHK_PRT_RET((launchArgs.inputNum < 1), HCCL_ERROR("HcomReduce input num[%u] is invalid.", launchArgs.inputNum),
              HCCL_E_PARA);

  uint64_t count;
  CHK_RET(GetCountByShape(launchArgs.inputShapes[0], launchArgs.opAttr.dataType, count));
  inputStruct->launchArgs = launchArgs;
  inputStruct->hcclCommPtr = hcclCommPtr;
  inputStruct->count = count;

  return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomLaunchReduceKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                    std::vector<void *> &outputAddrs) {
  HcomOpLaunchArgs launchArgs = inputStruct->launchArgs;
  void *hcclCommPtr = inputStruct->hcclCommPtr;

  CHK_RET(HcceReduce(inputAddrs[0], outputAddrs[0], inputStruct->count, launchArgs.opAttr.dataType,
                     launchArgs.opAttr.op.reduce.reduction, launchArgs.opAttr.op.reduce.root, hcclCommPtr,
                     launchArgs.stream));
  return HCCL_SUCCESS;
}
#endif

std::vector<std::function<HcclResult(HcomOpLaunchArgs &, HcomOpInputStruct *)>> HcomOpKernelFuncs = {
    HcomAllGatherKernel,     HcomAllGatherVKernel,     HcomAllReduceKernel, HcomBroadcastKernel,
    HcomReduceScatterKernel, HcomReduceScatterVKernel, HcomAllToAllVKernel, HcomAllToAllVCKernel,
    HcomAllToAllKernel,      HcomSendKernel,           HcomReduceKernel};

std::vector<std::function<HcclResult(const HcomOpInputStruct *, std::vector<void *> &, std::vector<void *> &)>>
    HcomOpLaunchKernelFuncs = {HcomLaunchAllGatherKernel, HcomLaunchAllGatherVKernel,    HcomLaunchAllReduceKernel,
                               HcomLaunchBroadcastKernel, HcomLaunchReduceScatterKernel, HcomLaunchReduceScatterVKernel,
                               HcomLaunchAllToAllVKernel, HcomLaunchAllToAllVCKernel,    HcomLaunchAllToAllKernel,
                               HcomLaunchSendKernel,      HcomLaunchReduceKernel};

HcclResult GetHcomOpLaunchArgs(gert::KernelContext *context, HcomOpLaunchArgs &args) {
  args.opAttr = *(context->GetInputPointer<struct HcomOpAttr>(static_cast<int32_t>(HcomOpLaunchInput::OP_ARGS)));
  args.stream = context->GetInputValue<void *>(static_cast<int32_t>(HcomOpLaunchInput::STREAM));
  args.inputNum = context->GetInputValue<uint32_t>(static_cast<int32_t>(HcomOpLaunchInput::INPUT_NUM));
  args.outputNum = context->GetInputValue<uint32_t>(static_cast<int32_t>(HcomOpLaunchInput::OUTPUT_NUM));

  args.inputAddrs.resize(args.inputNum);
  args.outputAddrs.resize(args.outputNum);
  args.inputShapes.resize(args.inputNum);
  args.outputShapes.resize(args.outputNum);

  // 为提高性能，维测信息搬迁到PrintLaunchArgs函数中
  uint32_t i = 0U;
  auto addrStart = static_cast<int32_t>(HcomOpLaunchInput::OUTPUT_NUM) + 1;
  for (; i < args.inputNum; i++) {
    // allreduce 和 broadcast会格式扩散，导致original shape和storage shape不一致
    if (args.opAttr.opType == HcomOpType::HCOM_ALL_REDUCE || args.opAttr.opType == HcomOpType::HCOM_BROADCAST) {
      args.inputShapes[i] = context->GetInputPointer<gert::StorageShape>(addrStart + i)->GetStorageShape();
    } else {
      args.inputShapes[i] = *(context->GetInputPointer<gert::Shape>(addrStart + i));
    }
  }

  uint32_t j = 0U;
  for (; j < args.outputNum; j++) {
    if (args.opAttr.opType == HcomOpType::HCOM_ALL_REDUCE || args.opAttr.opType == HcomOpType::HCOM_BROADCAST) {
      args.outputShapes[j] =
          context->GetInputPointer<gert::StorageShape>(addrStart + args.inputNum + j)->GetStorageShape();
    } else {
      args.outputShapes[j] = *(context->GetInputPointer<gert::Shape>(addrStart + args.inputNum + j));
    }
  }

  return HCCL_SUCCESS;
}

std::vector<std::string> PrintLaunchArgs(const gert::KernelContext *context) {
  HcomOpLaunchArgs args;
  args.opAttr = *(context->GetInputPointer<struct HcomOpAttr>(static_cast<int32_t>(HcomOpLaunchInput::OP_ARGS)));
  args.inputNum = context->GetInputValue<uint32_t>(static_cast<int32_t>(HcomOpLaunchInput::INPUT_NUM));
  args.outputNum = context->GetInputValue<uint32_t>(static_cast<int32_t>(HcomOpLaunchInput::OUTPUT_NUM));

  std::stringstream ss;
  ss << "GetHcomOpLaunchArgs:"
     << " opType:" << static_cast<int32_t>(args.opAttr.opType)
     << ", dataType:" << GetDataTypeEnumStr(args.opAttr.dataType) << ", group:" << args.opAttr.group
     << ", input_num:" << args.inputNum << ", output_num:" << args.outputNum;

  std::vector<std::string> msgs;
  msgs.emplace_back(ss.str());
  return msgs;
}

/*
 * **********************************************************************
 * 注册kernel函数，提供kernel函数的执行原型
 * **********************************************************************
 */
ge::graphStatus PrepareHcomKernel(gert::KernelContext *context) {
  // 获取算子输入输出资源
  HcomOpLaunchArgs launchArgs;
  if (GetHcomOpLaunchArgs(context, launchArgs) != HCCL_SUCCESS) {
    HCCL_ERROR("PrepareHcomKernel: get hcom op launch args failed.");
    return ge::GRAPH_FAILED;
  }

  // 获取通信域资源，
  if (HcomCreateCommCCLbuffer(launchArgs.opAttr.group) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchHcomKernel: create ccl buffer failed. group:%s", launchArgs.opAttr.group);
    return ge::GRAPH_FAILED;
  }

  HcomOpInputStruct *inputStruct = context->GetOutputPointer<struct HcomOpInputStruct>(0U);
  if (inputStruct == nullptr) {
    HCCL_ERROR("PrepareHcomKernel: inputStruct is nullptr!");
    return ge::GRAPH_FAILED;
  }
  // 设置单算子mode
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  if (HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchHcomKernel: config opbase mode failed");
    return ge::GRAPH_FAILED;
  }
  HcomSetLaunchKernelMode(true);
  // 调用算子kernel，实现算子下发
  if (HcomOpKernelFuncs[static_cast<int32_t>(launchArgs.opAttr.opType)](launchArgs, inputStruct) != HCCL_SUCCESS) {
    HcomSetLaunchKernelMode(false);
    return ge::GRAPH_FAILED;
  }
  inputStruct->lastWorkflowMode = lastWorkflowMode;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LaunchHcomKernel(gert::KernelContext *context) {
  const HcomOpInputStruct *inputStruct = context->GetInputPointer<struct HcomOpInputStruct>(0);
  if (inputStruct == nullptr) {
    HCCL_ERROR("LaunchHcomKernel: inputStruct is nullptr!");
    return ge::GRAPH_FAILED;
  }
  auto addrStart = 1;
  std::vector<void *> inputAddrs;
  std::vector<void *> outputAddrs;
  inputAddrs.resize(inputStruct->launchArgs.inputNum);
  outputAddrs.resize(inputStruct->launchArgs.outputNum);
  for (uint32_t i = 0; i < inputStruct->launchArgs.inputNum; i++) {
    auto tensorData = context->GetInputValue<gert::TensorData *>(addrStart + i);
    inputAddrs[i] = tensorData->GetAddr();
  }

  for (uint32_t i = 0; i < inputStruct->launchArgs.outputNum; i++) {
    auto tensorData = context->GetInputValue<gert::TensorData *>(addrStart + inputStruct->launchArgs.inputNum + i);
    outputAddrs[i] = tensorData->GetAddr();
  }
  if (HcomOpLaunchKernelFuncs[static_cast<int32_t>(inputStruct->launchArgs.opAttr.opType)](
          inputStruct, inputAddrs, outputAddrs) != HCCL_SUCCESS) {
    HcomSetLaunchKernelMode(false);
    return ge::GRAPH_FAILED;
  }
  HcomSetLaunchKernelMode(false);
  // workflowmode状态回置
  if (HcomSetWorkflowMode(inputStruct->lastWorkflowMode) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchHcomKernel: restore hccl mode failed");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildPrepareHcomKernelOutput(const ge::FastNode *node, gert::KernelContext *context) {
  (void)node;
  auto inputStructChain = context->GetOutput(0);
  HcomOpInputStruct *inputStruct = new (std::nothrow) HcomOpInputStruct{};
  inputStructChain->Set(inputStruct, [](void *const data) { delete reinterpret_cast<HcomOpInputStruct *>(data); });
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(PrepareHcomKernel)
    .RunFunc(PrepareHcomKernel)
    .TracePrinter(PrintLaunchArgs)
    .OutputsCreator(BuildPrepareHcomKernelOutput);
REGISTER_KERNEL(LaunchHcomKernel).RunFunc(LaunchHcomKernel);

ge::graphStatus LaunchHcomKernelInitComm([[maybe_unused]] gert::KernelContext *context) {
  // 获取ge option 初始化hccl world group和group list
  CHK_RET(HcomInitialize());
  CHK_RET(InitGroup());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(LaunchHcomKernelInitComm).RunFunc(LaunchHcomKernelInitComm);

ge::graphStatus BuildHcclOutputShapeOutputs(const ge::FastNode *node, gert::KernelContext *context) {
  (void)node;
  auto chainOfShape = context->GetOutput(0);
  if (chainOfShape == nullptr) {
    HCCL_ERROR("BuildHcclOutputShapeOutputs shape failed");
    return ge::GRAPH_FAILED;
  }
  chainOfShape->SetWithDefaultDeleter(new (std::nothrow) gert::StorageShape());
  auto chainOfTensorData = context->GetOutput(1);
  if (chainOfTensorData == nullptr) {
    HCCL_ERROR("BuildHcclOutputShapeOutputs tensordata failed");
    return ge::GRAPH_FAILED;
  }
  chainOfTensorData->SetWithDefaultDeleter(new (std::nothrow) gert::GertTensorData());

  return ge::GRAPH_SUCCESS;
}

HcclResult GetRecvOpLaunchArgs(gert::KernelContext *context, HcomOpLaunchArgs &args) {
  // args
  args.opAttr = *(context->GetInputPointer<struct HcomOpAttr>(static_cast<int32_t>(RecvOpLaunchInput::OP_ARGS)));
  HCCL_INFO("GetHcomOpLaunchArgs: opType:%u, dataType:%s, group:%s", args.opAttr.opType,
            GetDataTypeEnumStr(args.opAttr.dataType).c_str(), args.opAttr.group);

  // stream
  args.stream = context->GetInputValue<void *>(static_cast<int32_t>(RecvOpLaunchInput::STREAM));

  return HCCL_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
ge::graphStatus HcomGetRecvBeforeKernelV2(HcomOpLaunchArgs &args, std::vector<int64_t> &recvShape) {
  // 获取通信域资源
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  if (HcomGetCommHandleByGroup(args.opAttr.group, &hcclCommPtr) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchHcomKernel: get hcom comm handle failed. group:%s", args.opAttr.group);
    return ge::GRAPH_FAILED;
  }

  // 1、create device mem
  auto deleter = [](void *dst) {
    if (dst != nullptr) {
      CHK_PRT(hrtFree(dst));
      dst = nullptr;
    }
  };

  void *recvShapeMemPtr = nullptr;
  u32 recvShapeSize = sizeof(int64_t) * SHAPE_INFO_COUNT;
  CHK_RET(hrtMalloc(&recvShapeMemPtr, recvShapeSize));
  std::unique_ptr<void, decltype(deleter)> recvShapeMemPtrUnique(recvShapeMemPtr, deleter);

  // 2、recv
  CHK_RET(HcceRecv(recvShapeMemPtr, SHAPE_INFO_COUNT, HCCL_DATA_TYPE_INT64, args.opAttr.op.recv.srcRank, hcclCommPtr,
                   args.stream));
  CHK_RET(hcclStreamSynchronize(args.stream));

  // 3、DevtoHost
  CHK_RET(hrtMemSyncCopy(recvShape.data(), recvShape.size(), recvShapeMemPtr, recvShapeSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

  HCCL_INFO("HcomGetRecvBeforeKernelV2 get dynamic recv shape DONE and dim is [%lld]", recvShape[0]);

  return ge::GRAPH_SUCCESS;
}
#endif

ge::graphStatus HcomGetRecvBeforeKernel(HcomOpLaunchArgs &args, std::vector<int64_t> &recvShape) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return HcomGetRecvBeforeKernelV2(args, recvShape);
  }
#endif
  // 获取通信域资源
  HcclComm commHandle = nullptr;
  if (HcomGetCommHandleByGroup(args.opAttr.group, &commHandle) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchHcomKernel: get hcom comm handle failed. group:%s", args.opAttr.group);
    return ge::GRAPH_FAILED;
  }

  // 1、create device mem
  auto deleter = [](void *dst) {
    if (dst != nullptr) {
      CHK_PRT(hrtFree(dst));
      dst = nullptr;
    }
  };

  void *recvShapeMemPtr = nullptr;
  u32 recvShapeSize = sizeof(int64_t) * SHAPE_INFO_COUNT;
  CHK_RET(hrtMalloc(&recvShapeMemPtr, recvShapeSize));
  std::unique_ptr<void, decltype(deleter)> recvShapeMemPtrUnique(recvShapeMemPtr, deleter);

  // 2、recv
  CHK_RET(HcceRecv(recvShapeMemPtr, SHAPE_INFO_COUNT, HCCL_DATA_TYPE_INT64, args.opAttr.op.recv.srcRank, commHandle,
                   args.stream));

  CHK_RET(hcclStreamSynchronize(args.stream));

  // 3、DevtoHost
  CHK_RET(hrtMemSyncCopy(recvShape.data(), recvShape.size(), recvShapeMemPtr, recvShapeSize,
                         HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

  HCCL_INFO("HcomGetRecvBeforeKernel get dynamic recv shape DONE and dim is [%lld]", recvShape[0]);

  return ge::GRAPH_SUCCESS;
}

#ifndef OPEN_BUILD_PROJECT
ge::graphStatus LaunchRecvKernelV2(gert::KernelContext *context) {
  uint64_t tensorSize = 0;
  std::vector<int64_t> recvShape(sizeof(int64_t) * SHAPE_INFO_COUNT, 0);

  HcomSetLaunchKernelMode(true);

  // 获取算子输入输出资源
  HcomOpLaunchArgs launchArgs;
  if (GetRecvOpLaunchArgs(context, launchArgs) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernelV2: get hcom op launch launchArgs failed.");
    return ge::GRAPH_FAILED;
  }

  // 获取通信域资源
  std::shared_ptr<HcclComm> hcclComm;
  void *hcclCommPtr = hcclComm.get();
  CHK_RET(HcomGetCommHandleByGroup(launchArgs.opAttr.group, &hcclCommPtr));

  CHK_RET(HcomCreateCommCCLbuffer(launchArgs.opAttr.group));

  // 设置单算子mode
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  if (HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernelV2: config opbase mode failed");
    return ge::GRAPH_FAILED;
  }
  HcomSetLaunchKernelMode(true);

  if (HcomGetRecvBeforeKernel(launchArgs, recvShape) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernelV2: get recvShape fail");
    return ge::GRAPH_FAILED;
  }

  // 1. create && append outputshape
  auto outputShape = context->GetOutputPointer<gert::StorageShape>(0U);
  if (outputShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t dimNum = recvShape[0];  // recvshape n维的长度
  if (dimNum > gert::Shape::kMaxDimNum || dimNum <= 0) {
    HCCL_ERROR("dimNum=[%u] is greater than MaxDimNum[%zu] or less than or equal to 0", dimNum,
               gert::Shape::kMaxDimNum);
    return ge::GRAPH_FAILED;
  }
  gert::Shape &originShape = outputShape->MutableOriginShape();
  gert::Shape &storageShape = outputShape->MutableStorageShape();
  originShape.SetDimNum(0);
  storageShape.SetDimNum(0);
  for (size_t i = 1U; i <= dimNum; ++i) {
    originShape.AppendDim(static_cast<int64_t>(recvShape[i]));
    storageShape.AppendDim(static_cast<int64_t>(recvShape[i]));
  }

  // 2. create tensor_data
  ge::DataType geDataType = context->GetInputValue<ge::DataType>(static_cast<int32_t>(RecvOpLaunchInput::DATATYPE));
  ge::graphStatus getSizeRet = CalcAlignedSizeByShape(originShape, geDataType, tensorSize);
  if (getSizeRet != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernelV2: get tensor size failed");
    return ge::GRAPH_FAILED;
  }

  auto gertAllocator =
      context->GetInputValue<gert::GertAllocator *>(static_cast<int32_t>(RecvOpLaunchInput::ALLOCATOR));
  if (gertAllocator == nullptr) {
    HCCL_ERROR("LaunchRecvKernelV2: get gertAllocator failed");
    return ge::GRAPH_FAILED;
  }

  auto outputTensorData = context->GetOutputPointer<gert::GertTensorData>(1U);

  *outputTensorData = gertAllocator->MallocTensorData(tensorSize);

  HcomSetLaunchKernelMode(false);

  // 调用算子kernel，实现算子下发
  uint64_t recvCount = (tensorSize - 32) / SIZE_TABLE[launchArgs.opAttr.dataType];
  if (HcceRecv(outputTensorData->GetAddr(), recvCount, launchArgs.opAttr.dataType, launchArgs.opAttr.op.recv.srcRank,
               hcclCommPtr, launchArgs.stream) != HCCL_SUCCESS) {
    HcomSetLaunchKernelMode(false);
    return ge::GRAPH_FAILED;
  }

  HcomSetLaunchKernelMode(false);
  HCCL_INFO("LaunchRecvKernelV2 dispatch hcclrecv DONE");

  // workflowmode状态回置
  if (HcomSetWorkflowMode(lastWorkflowMode) != HCCL_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  HCCL_INFO("LaunchRecvKernelV2 dispatch DONE");
  return ge::GRAPH_SUCCESS;
}
#endif

ge::graphStatus LaunchRecvKernel(gert::KernelContext *context) {
#ifndef OPEN_BUILD_PROJECT
  DevType devType = HcomGetDeviceType();
#ifdef MACRO_DEV_TYPE_NEW
  if (devType == DevType::DEV_TYPE_950) {
#else
  if (devType == DevType::DEV_TYPE_910_95) {
#endif
    return LaunchRecvKernelV2(context);
  }
#endif
  uint64_t tensorSize = 0;
  std::vector<int64_t> recvShape(sizeof(int64_t) * SHAPE_INFO_COUNT, 0);

  HcomSetLaunchKernelMode(true);

  // 获取算子输入输出资源
  HcomOpLaunchArgs launchArgs;
  if (GetRecvOpLaunchArgs(context, launchArgs) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernel: get hcom op launch launchArgs failed.");
    return ge::GRAPH_FAILED;
  }

  // 获取通信域资源
  HcclComm commHandle = nullptr;
  if (HcomGetCommHandleByGroup(launchArgs.opAttr.group, &commHandle) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernel: get hcom comm handle failed. group:%s", launchArgs.opAttr.group);
    return ge::GRAPH_FAILED;
  }
  if (HcomCreateCommCCLbuffer(launchArgs.opAttr.group) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernel: create ccl buffer failed. group:%s", launchArgs.opAttr.group);
    return ge::GRAPH_FAILED;
  }

  // 设置单算子mode
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  if (HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) != HCCL_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernel: config opbase mode failed");
    return ge::GRAPH_FAILED;
  }
  HcomSetLaunchKernelMode(true);

  if (HcomGetRecvBeforeKernel(launchArgs, recvShape) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernel: get recvShape fail");
    return ge::GRAPH_FAILED;
  }

  // 1. create && append outputshape
  auto outputShape = context->GetOutputPointer<gert::StorageShape>(0U);
  if (outputShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t dimNum = recvShape[0];  // recvshape n维的长度
  if (dimNum > gert::Shape::kMaxDimNum || dimNum <= 0) {
    HCCL_ERROR("dimNum=[%u] is greater than MaxDimNum[%zu] or less than or equal to 0", dimNum,
               gert::Shape::kMaxDimNum);
    return ge::GRAPH_FAILED;
  }
  gert::Shape &originShape = outputShape->MutableOriginShape();
  gert::Shape &storageShape = outputShape->MutableStorageShape();
  originShape.SetDimNum(0);
  storageShape.SetDimNum(0);
  for (size_t i = 1U; i <= dimNum; ++i) {
    originShape.AppendDim(static_cast<int64_t>(recvShape[i]));
    storageShape.AppendDim(static_cast<int64_t>(recvShape[i]));
  }

  // 2. create tensor_data
  ge::DataType geDataType = context->GetInputValue<ge::DataType>(static_cast<int32_t>(RecvOpLaunchInput::DATATYPE));
  ge::graphStatus getSizeRet = CalcAlignedSizeByShape(originShape, geDataType, tensorSize);
  if (getSizeRet != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("LaunchRecvKernel: get tensor size failed");
    return ge::GRAPH_FAILED;
  }

  auto gertAllocator =
      context->GetInputValue<gert::GertAllocator *>(static_cast<int32_t>(RecvOpLaunchInput::ALLOCATOR));
  if (gertAllocator == nullptr) {
    HCCL_ERROR("LaunchRecvKernel: get gertAllocator failed");
    return ge::GRAPH_FAILED;
  }

  auto outputTensorData = context->GetOutputPointer<gert::GertTensorData>(1U);

  *outputTensorData = gertAllocator->MallocTensorData(tensorSize);

  HcomSetLaunchKernelMode(false);

  // 调用算子kernel，实现算子下发
  uint64_t recvCount = (tensorSize - 32) / SIZE_TABLE[launchArgs.opAttr.dataType];
  if (HcceRecv(outputTensorData->GetAddr(), recvCount, launchArgs.opAttr.dataType, launchArgs.opAttr.op.recv.srcRank,
               commHandle, launchArgs.stream) != HCCL_SUCCESS) {
    HcomSetLaunchKernelMode(false);
    return ge::GRAPH_FAILED;
  }

  HcomSetLaunchKernelMode(false);
  HCCL_INFO("LaunchRecvKernel dispatch hcclrecv DONE");

  // workflowmode状态回置
  if (HcomSetWorkflowMode(lastWorkflowMode) != HCCL_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  HCCL_INFO("LaunchRecvKernel dispatch DONE");
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(LaunchRecvKernel).RunFunc(LaunchRecvKernel).OutputsCreator(BuildHcclOutputShapeOutputs);
}  // namespace hccl