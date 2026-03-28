/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_OPS_KERNEL_INFO_STORE_H
#define HCOM_OPS_KERNEL_INFO_STORE_H

#include "ops_kernel_info_store_base.h"
#include "op_hcom_comm.h"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include "acl/acl_rt.h"
#include "hcom_acl_adapter.h"
#include "hccl/hcom.h"

namespace hccl {
constexpr u32 CRACK_MEMORY_SIZE = 32;        // 申请32B内存，当缝隙SIZE小于32B时，用D2D Memcopy进行清零操作
constexpr u32 CRACK_MEMORY_MAX_SIZE = 1024;  // 缝隙最大memory size

enum kernelHcclInfoIndex { KERNEL_HCCL_INFO_IDX_0 = 0, KERNEL_HCCL_INFO_IDX_NUM = 1 };

enum UpdatePairedInputIndex {
  UPDATE_PRD_IT_TABLEID_IDX = 0,
  UPDATE_PRD_IT_KEY_IDX = 1,
  UPDATE_PRD_IT_VALUE_IDX = 2,
  UPDATE_PRD_IT_INDICES_IDX = 3,
  UPDATE_PRD_IT_NUMUNIQUED_IDX = 4,
  UPDATE_PRD_IT_PSSEG_IDX = 5,
  UPDATE_PRD_IT_PSSEGNUM_IDX = 6,
  UPDATE_PRD_IT_IDX_NUM = 7
};

// Ge适配的类
class HcomOpsKernelInfoStore : public HCCLOpsKernelInfoStore {
 public:
  HcomOpsKernelInfoStore();
  ~HcomOpsKernelInfoStore() override;
  // load task for op
  ge::Status LoadTask(ge::GETaskInfo &task) override;
  ge::Status UnloadTask(ge::GETaskInfo &task) override;
  ge::Status Finalize() override;

  // get and set map
  bool UpdateGraphIdGroupMap(std::string group, u32 graphId);

 protected:
  virtual HcclResult CheckPrivateDef(const ge::GETaskInfo &task);
  virtual HcclResult GetOriginalGraphShapeTypeFromTaskInfo(const ge::GETaskInfo &task, u32 &shapeType);

 private:
  HcclResult SetCustomKernelInfo(ge::OpInfo &opinfo, std::map<string, ge::OpInfo> &infos) const override;
  HcclResult GetSupportedOP(std::vector<std::string> &hcclSupportOp) const override;
  HcclResult GetDataTypeFromTaskInfo(const ge::GETaskInfo &task, HcclDataType &dataType) const override;
  HcclResult CheckOutputMemSize(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup, u64 outputMemSize);
  HcclResult GenerateOpTagFromTaskInfo(const ge::GETaskInfo &task, const std::string &opType, std::string &sTag,
                                       u32 &loopMaxTime);
  HcclResult GetGroupFromTaskInfo(const ge::GETaskInfo &task, std::string &sGroup);
  HcclResult RefreshInputAddr(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup, const void *inputAddr,
                              u64 inputMemSize, rtStream_t stream);
  HcclResult RefreshOutputAddr(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup, void *outputAddr,
                               u64 outputMemSize, rtStream_t stream);
  HcclResult RefreshOutputAddr(DevType deviceType, u32 shapeType, const std::string &sCollectiveType,
                               const int64_t &hcomComm, const std::string &sGroup, void *outputAddr, u64 outputOffset,
                               u64 curSize, u64 outputMaxSize, bool secAddrCopyWithoutOffset,
                               rtStream_t stream);  // 输入地址等于输出地址
  HcclResult GetHcomOpMemSize(u32 shapeType, const std::string &sCollectiveType, const int64_t &hcomComm,
                              const std::string &sGroup, HcclDataType dataType, u64 count, u64 &inputMemSize,
                              u64 &outputMemSize);
  HcclResult GetHcomOpMemSize(u32 shapeType, const std::string &sCollectiveType, HcclDataType dataType, u64 count,
                              u64 &inputMemSize);  // 输入地址等于输出地址
  HcclResult GetHcomAlltoallVOpMemSize(u32 shapeType, const std::string &sCollectiveType, const int64_t &hcomComm,
                                       const std::string &sGroup, HcclDataType sendType, HcclDataType recvType,
                                       void *sendCounts, void *sendDispls, void *recvCounts, void *recvDispls,
                                       u64 &inputMemSize, u64 &outputMemSize);
  HcclResult GetHcomAlltoallVCOpMemSize(u32 shapeType, const std::string &sCollectiveType, const int64_t &hcomComm,
                                        const std::string &sGroup, HcclDataType sendType, HcclDataType recvType,
                                        void *sendCountMatrix, u64 &inputMemSize, u64 &outputMemSize);
  HcclResult GetCommCCLBuf(u32 shapeType, const int64_t &hcomComm, const std::string &sGroup, void *&commInputPtr,
                           void *&commOutputPtr);
  HcclResult GetCommCCLBuf(u32 shapeType, const std::string &sCollectiveType, const int64_t &hcomComm,
                           const std::string &sGroup, void *&commInputPtr);  // 输入地址等于输出地址
  HcclResult GetCrackParamsInfoFromTaskInfo(const ge::GETaskInfo &task, uintptr_t &inputAddr,
                                            std::vector<std::int64_t> &crackAddr, std::vector<std::int64_t> &crackSize,
                                            size_t crackNum, size_t privateDefBufSize, u64 inputOffset);
  HcclResult GetDestRankFromTaskInfo(const ge::GETaskInfo &task, u32 &destRank);
  HcclResult GetSrcRankFromTaskInfo(const ge::GETaskInfo &task, u32 &srcRank);
  HcclResult GetSrTagFromTaskInfo(const ge::GETaskInfo &task, u32 &srTag);
  HcclResult GetCommFromTaskInfo(const ge::GETaskInfo &task, int64_t &comm);
  HcclResult SetKnownShapeWorkspaceResource(const ge::GETaskInfo &task, const std::string &sCollectiveType,
                                            const std::vector<std::string> &tagVec);
  HcclResult SetUnknownShapeWorkspaceResource(const ge::GETaskInfo &task, const std::string &sCollectiveType,
                                              const std::vector<std::string> &tagVec);
  HcclResult SetAttachedStream(const ge::GETaskInfo &task);
  HcclResult HCCLOpsKernel(const ge::GETaskInfo &task, const std::string &sCollectiveType,
                           const std::vector<std::string> &tagVec);
  HcclResult HcomBroadcastOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomAllReduceOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomAllGatherOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomAllGatherVOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomReduceScatterOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomReduceScatterVOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomSendOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomReceiveOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomReduceOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomAlltoAllVOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomAlltoAllVCOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);
  HcclResult HcomAlltoAllOpKernel(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec);

  HcclResult SaveReduceDumpTask(std::vector<ge::HcclDumpInfo> &geDumpInfo, std::vector<hccl::HcclDumpInfo> &dumpInfo);
  HcclResult ConfigHcclDumpDebugMode();
  bool IsOpTypeCCLTag(const std::string &opType);
  bool IsRefresh(ge::GETaskInfo &task, const std::string &opType, u32 shapeType);
  void GetAlltoAllVParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, void *&sendCounts, void *&sendDispls,
                          uintptr_t &recvBuf, void *&recvCounts, void *&recvDispls, HcclDataType &sendType,
                          HcclDataType &recvType);
  void GetAlltoAllVCParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, void *&sendCountMatrix,
                           HcclDataType &sendType, uintptr_t &recvBuf, HcclDataType &recvType);
  void GetReduceScatterVParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, void *&sendCounts, void *&sendDispls,
                               uintptr_t &recvBuf, int64_t &recvCount);
  void GetAllGatherVParams(const ge::GETaskInfo &task, uintptr_t &sendBuf, int64_t &sendCount, uintptr_t &recvBuf,
                           void *&recvCounts, void *&recvDispls);
  HcclResult CheckCommunicatorValidity(const char *group, const ge::GETaskInfo &task);
  HcclResult InitHcom();
  HcclResult CleanIntervalMemoryOpKernel(const ge::GETaskInfo &task, const std::string &tag, uintptr_t inputAddr,
                                         u64 inputOffset, rtStream_t stream, HcclCMDType opType);
  HcclResult CleanIntervalMemory(const char *tag, std::vector<std::int64_t> &crackAddr,
                                 std::vector<std::int64_t> &crackSize, rtStream_t stream);
  HcclResult TbeCleanIntervalMemory(std::vector<std::int64_t> &crackAddr, std::vector<std::int64_t> &crackSize,
                                    rtStream_t stream);
  HcclResult GetRealRankIdFromMap(const u32 srcRankId, const std::string &rankMapJsonStr, u32 &dstRankId);
  HcclResult TransfromRealRankId(const ge::GETaskInfo &task);
  HcclResult GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index, const char *propName,
                                        u32 &propValue);
  HcclResult GetJsonProperty(const nlohmann::json &obj, const char *propName, nlohmann::json &propValue);
  HcclResult CheckOfflineDevTypeIsSame(const ge::GETaskInfo &task);

  HcclResult HcomAllGatherLoop(const std::vector<std::string> &tagVec, u32 shapeType, const int64_t &comm,
                               const std::string &group, void *&inputDataPtr, void *&outputDataPtr, u64 count,
                               HcclDataType dataType, rtStream_t streamMain);
  HcclResult HcomAllReduceLoop(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec, u32 shapeType,
                               const int64_t &comm, const std::string &group, void *&inputDataPtr, void *&outputDataPtr,
                               u64 count, HcclDataType dataType, HcclReduceOp reduceType, rtStream_t streamMain);
  HcclResult RefreshAllgatherOutputAddr(DevType deviceType, u32 shapeType, const int64_t &hcomComm,
                                        const std::string &sGroup, void *&outputAddr, u64 outputOffset, u64 curSize,
                                        u64 count, u32 unitSize, u32 rankSize, bool secAddrCopyWithoutOffset,
                                        rtStream_t stream);
  HcclResult RefreshInputAddr(DevType deviceType, u32 shapeType, const int64_t &hcomComm, const std::string &sGroup,
                              const void *inputAddr, u64 inputOffset, u64 curSize, bool secAddrCopyWithoutOffset,
                              rtStream_t stream);
  HcclResult RefreshOutputAddr(DevType deviceType, u32 shapeType, const int64_t &hcomComm, const std::string &sGroup,
                               void *outputAddr, u64 outputOffset, u64 curSize, u64 outputMaxSize,
                               bool secAddrCopyWithoutOffset, rtStream_t stream);
  HcclResult GetHcomOutCCLbufferSize(u64 &commOutputSize, u32 shapeType, const int64_t &hcomComm,
                                     const std::string &sGroup);
  HcclResult GetHcomInCCLbufferSize(u64 &commInputSize, u32 shapeType, const int64_t &hcomComm,
                                    const std::string &sGroup);
  HcclResult HcomReduceScatterLoop(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec, u32 shapeType,
                                   const int64_t &comm, const std::string &group, void *&inputDataPtr,
                                   void *&outputDataPtr, u64 count, HcclDataType dataType, HcclReduceOp reduceType,
                                   rtStream_t streamMain);
  HcclResult RefreshReduceScatterInputAddr(DevType deviceType, u32 shapeType, const int64_t &hcomComm,
                                           const std::string &sGroup, void *&inputAddr, u64 inputOffset, u64 curSize,
                                           u64 count, u32 unitSize, u32 rankSize, bool secAddrCopyWithoutOffset,
                                           rtStream_t stream);
  HcclResult HcomReduceLoop(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec, u32 shapeType,
                            const int64_t &comm, const std::string &group, void *&inputDataPtr, void *&outputDataPtr,
                            u64 count, HcclDataType dataType, HcclReduceOp reduceType, u32 root, rtStream_t streamMain);
  HcclResult HcomSendLoop(const std::vector<std::string> &tagVec, u32 &srTag, u32 shapeType, const int64_t &comm,
                          const std::string &group, void *&inputDataPtr, u64 count, HcclDataType dataType,
                          u32 &destRank, rtStream_t streamMain);
  HcclResult HcomReceiveLoop(const std::vector<std::string> &tagVec, u32 &srTag, u32 shapeType, const int64_t &comm,
                             const std::string &group, void *&outputDataPtr, u64 count, HcclDataType dataType,
                             u32 &srcRank, rtStream_t streamMain);
  HcclResult HcomBroadcastLoop(const std::vector<std::string> &tagVec, u32 shapeType, const int64_t &comm,
                               const std::string &group, void *&inputDataPtr, u64 count, HcclDataType dataType,
                               u32 root, rtStream_t streamMain);
  HcclResult CheckHcomOpMemSize(DevType deviceType, u64 countLeft, u32 unitSize, u64 cclBufferSize);
  HcclResult CheckTensorNumAndTensorSize(const ge::GETaskInfo &task, u64 count, u32 unitSize, u64 commInputSize);
  HcclResult CreateIndirectCCLbuf();
  HcclResult GetIndirectInCCLbuf(void *&ptr, u64 &size);
  HcclResult GetIndirectOutCCLbuf(void *&ptr, u64 &size);
  HcclResult GetOutputCCLbufPtrAndIndirectOutCCLbufPtr(const int64_t &hcomComm, const std::string &sGroup,
                                                       void *&commOutputPtr, u64 &commOutputSize,
                                                       void *&indirectOutCCLbufPtr, u64 &indirectCommOutputSize);
  HcclResult GetInputCCLbufPtrAndIndirectInCCLbufPtr(const int64_t &hcomComm, const std::string &sGroup,
                                                     void *&commInputPtr, u64 &commInputSize,
                                                     void *&indirectInCCLbufPtr, u64 &indirectCommInputSize);
  HcclResult GetOpKernelLoopTime(const ge::GETaskInfo &task, const std::string &opType, std::string &sTag,
                                 u32 &loopMaxTime);
  HcclResult GetTagVectorInfo(const ge::GETaskInfo &task, const std::string &sCollectiveType,
                              std::vector<std::string> &tagVec);
  HcclResult SetWorkspaceResourceFromtagVec(const ge::GETaskInfo &task, const char *group,
                                            const std::vector<std::string> &tagVec, void *memPtr, u64 maxSize);
  HcclResult GetHcclGroup(ge::GETaskInfo &task, std::string &sGroup);
  HcclResult HcomAicpuStreamRegister(ge::GETaskInfo &task);
  HcclResult HcomAicpuStreamUnRegister(ge::GETaskInfo &task);
  HcclResult SetAivCoreLimit(const ge::GETaskInfo &task);
  HcclResult SetGlobalWorkSpace(const int64_t &hcomComm, const string &sGroup, std::vector<void *> globalWorkSpaceAddr);
#ifndef OPEN_BUILD_PROJECT
  HcclResult CleanInterMemoryV2(std::vector<std::int64_t> &crackSize,
                                std::vector<std::int64_t> &crackAddr, rtStream_t stream);
#endif
  HcclResult CleanInterMemory(std::vector<std::int64_t> &crackAddr,
                              std::vector<std::int64_t> &crackSize, rtStream_t stream);

  std::mutex workSpaceMemMutex_;
  std::map<std::string, std::tuple<void *, u64>> workSpaceMemInfo_;  // key:group name,value:workSpace mem ptr and size
  std::mutex taskIDtoTagMutex_;
  std::unordered_map<u32, std::vector<std::string>> taskIDtoTag_;  // key:GE::taskID,value:hccl::tag
  std::unordered_map<std::string, u64> graphInfoMap_;

  // 管理向GE注册的kernel流的引用计数
  std::mutex orderedStreamMutex_;
  std::array<std::unordered_map<std::string, u64>, MAX_MODULE_DEVICE_NUM> orderedStreamCount_;

 private:
  const std::map<std::string,
                 std::function<HcclResult(const ge::GETaskInfo &task, const std::vector<std::string> &tagVec)>>
      kernelFuncTable_ = {
          {HCCL_KERNEL_OP_TYPE_BROADCAST, std::bind(&HcomOpsKernelInfoStore::HcomBroadcastOpKernel, this,
                                                    std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_ALLREDUCE, std::bind(&HcomOpsKernelInfoStore::HcomAllReduceOpKernel, this,
                                                    std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_ALLGATHER, std::bind(&HcomOpsKernelInfoStore::HcomAllGatherOpKernel, this,
                                                    std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_ALLGATHERV, std::bind(&HcomOpsKernelInfoStore::HcomAllGatherVOpKernel, this,
                                                     std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_REDUCESCATTER, std::bind(&HcomOpsKernelInfoStore::HcomReduceScatterOpKernel, this,
                                                        std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_REDUCESCATTERV, std::bind(&HcomOpsKernelInfoStore::HcomReduceScatterVOpKernel, this,
                                                         std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_REDUCE,
           std::bind(&HcomOpsKernelInfoStore::HcomReduceOpKernel, this, std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_SEND,
           std::bind(&HcomOpsKernelInfoStore::HcomSendOpKernel, this, std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_RECEIVE,
           std::bind(&HcomOpsKernelInfoStore::HcomReceiveOpKernel, this, std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_ALLTOALLV, std::bind(&HcomOpsKernelInfoStore::HcomAlltoAllVOpKernel, this,
                                                    std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_ALLTOALLVC, std::bind(&HcomOpsKernelInfoStore::HcomAlltoAllVCOpKernel, this,
                                                     std::placeholders::_1, std::placeholders::_2)},
          {HCCL_KERNEL_OP_TYPE_ALLTOALL, std::bind(&HcomOpsKernelInfoStore::HcomAlltoAllOpKernel, this,
                                                   std::placeholders::_1, std::placeholders::_2)}};

  struct Deleter {
    void operator()(void *ptr) const {
      if (ptr != nullptr) {
        (void)hrtFree(ptr);
      }
    }
  };

  using CrackMemPtr = std::unique_ptr<void, Deleter>;

  bool initCrackMem_;
  CrackMemPtr crackMemPtr_{};
  CrackMemPtr crackMemPtrV2_{};
  u64 maxCrackMemSizeV2_ = 0;
  std::unordered_map<std::string, u32> graphIdByGroup_;
  std::unordered_map<s64, u32> graphIdByCommId_;
  std::unique_ptr<void, Deleter> indirectInCCLbufferPtr_;
  std::unique_ptr<void, Deleter> indirectOutCCLbufferPtr_;
};
}  // namespace hccl
#endif  // GE_OPS_KERNEL_INFO_H
