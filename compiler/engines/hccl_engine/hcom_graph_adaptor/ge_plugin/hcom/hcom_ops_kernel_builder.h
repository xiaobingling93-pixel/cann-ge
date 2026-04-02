/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_OPS_KERNEL_BUILDER_H
#define HCOM_OPS_KERNEL_BUILDER_H

#include "ops_kernel_builder_base.h"
#include "op_hcom_comm.h"

namespace hccl {
class HcomOpsKernelBuilder : public HCCLOpsKernelBuilder {
 public:
  HcomOpsKernelBuilder();
  ~HcomOpsKernelBuilder() override;
  // memory allocation requirement
  ge::Status CalcOpRunningParam(ge::Node &node) override;
  // generate taskinfo for op
  ge::Status GenerateTask(const ge::Node &node, ge::RunContext &runContext,
                          std::vector<domi::TaskDef> &taskDefList) override;

 protected:
  HcclResult CheckSupportedOP(const std::string &sCollectiveType) const override;

 private:
  HcclResult HcomCalcOpRunningParam(ge::Node &node);
  HcclResult SetHcomOpParam(const ge::Node &node, HcomOpParam *hcomOpParam, std::string &sCollectiveType,
                            std::string &sGroup, std::string &socVersion, std::vector<int64_t> &sendCountMatrix,
                            std::vector<int64_t> &sendCounts, std::vector<int64_t> &sendDispls,
                            std::vector<int64_t> &recvCounts, std::vector<int64_t> &recvDispls,
                            std::vector<u32> &curRanks, std::string &rankTableStr, std::string &rankTableM);
  HcclResult SetOpWorkerSpaceForKnowShape(ge::Node &node, u64 &opMemSize);
  HcclResult GetSupportedOP(std::vector<std::string> &hcclSupportOp) const override;
  HcclResult SetOpMemAttr(ge::Node &node, const std::string &sCollectiveType, const u64 &opMemSize) override;
  HcclResult SetOpAtomicInputIndex(ge::Node &node, const std::string &sCollectiveType) override;
  HcclResult GetCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType, HcclDataType dataType,
                                u64 &count);
  HcclCMDType GetOpType(const std::string &sCollectiveType) const;
  HcclResult PrepareSuperKernelRuntimeParams(const ge::OpDescPtr &opDescPtr, const std::string &sCollectiveType,
                                            HcclCMDType opType, HcclDataType &dataType, u64 &count,
                                            int64_t &comm, std::string &group, u32 &rankSize, u32 &aivCoreLimit,
                                            HcclReduceOp &reduction);
  HcclResult CheckSuperKernelEligibility(ge::Node &node, const ge::OpDescPtr &opDescPtr,
                                         std::string &sCollectiveType, std::string &superKernelScope,
                                         HcclCMDType &opType, bool &needProcess) const;
  HcclResult SetAivSuperKernelBinaryAttrs(const ge::OpDescPtr &opDescPtr, HcclCMDType opType, HcclDataType dataType,
                                          const std::string &algName, std::string &funcName);
  HcclResult SetAivSuperKernelBinaryAttrFor950(const ge::OpDescPtr &opDescPtr, HcclCMDType opType,
                                               HcclDataType dataType, const std::string &algName,
                                               std::string &funcName, const std::string & binPath) const;
  HcclResult SetAivSuperKernelBinaryAttrForDeter(const ge::OpDescPtr &opDescPtr, HcclCMDType opType,
                                                 const std::string &algName, std::string &funcName,
                                                 const std::string & binPath) const;
  HcclResult SetSuperKernelBlockDim(const ge::OpDescPtr &opDescPtr, const std::string &group, HcclCMDType opType,
                                    u64 count, void *counts, HcclDataType dataType, u32 aivCoreLimit, char *algName,
                                    u32 rankSize) const;
  HcclResult SetSuperKernelScopeAttr(ge::Node &node);
  HcclResult SKGetAlgPath(HcclCMDType opType, std::string &binaryPath) const;
  HcclResult GetHcomReceiveOpOutputSize(const ge::OpDescPtr &op, u32 dataTypeSize, u64 &outputSize);
  HcclResult GetRootGraphID(const ge::Node &node, uint32_t &graphId);
  HcclResult GetCommFromOpDesc(const ge::OpDescPtr &op, int64_t &hcomComm, std::string &sGroup);
  HcclResult GetSuperKernelFromDesc(const ge::OpDescPtr &op, std::string &superKernel);
  HcclResult GetDestRankFromDesc(const ge::OpDescPtr &op, u32 &destRank, const std::string &sCollectiveType);
  HcclResult GetSrcRankFromDesc(const ge::OpDescPtr &op, u32 &srcRank, const std::string &sCollectiveType);
  HcclResult GetSrTagFromDesc(const ge::OpDescPtr &op, u32 &srTag, const std::string &sCollectiveType);
  HcclResult GetOriginalGraphShapeTypeFromDesc(const ge::OpDescPtr &op, u32 &shapeType);
  HcclResult GenerateTaskDef(const ge::Node &node, HCCL_KERNEL_INFO_PRIVATE_DEF &privateDefBuf, domi::TaskDef &taskDef);
  HcclResult SetAlltoAllVParams(const ge::Node &node, HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                const std::string &sCollectiveType);
  HcclResult SetReduceScatterVParams(const ge::Node &node, HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult SetAllGatherVParams(const ge::Node &node, HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult CopyAlltoAllVParamsToDef(const ge::Node &node, HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult CopyReduceScatterVParamsToDef(const ge::Node &node,
                                           HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult CopyAllGatherVParamsToDef(const ge::Node &node, HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult SetPrivateDefWithTensorInfo(const ge::Node &node, HCCL_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                         domi::TaskDef &taskDef);
  HcclResult GetCrackParamsInfo(const ge::Node &node, u32 tensorNum, int64_t *tensorOffset, int64_t *tensorSize,
                                int64_t *crackOffset, int64_t *crackSize);
  HcclResult GetTensorParamsInfo(const ge::Node &node, u32 tensorNum, int64_t *tensorOffset, int64_t *tensorSize);
  HcclResult SetAlltoAllVDataTypeToDef(const ge::OpDescPtr &op, HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult CopyAlltoAllVCParamsToDef(const ge::Node &node, HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult SetAlltoAllVCDataTypeToDef(const ge::OpDescPtr &op, HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf);
  HcclResult GetNeedMapRankFromDesc(const ge::OpDescPtr &op, bool &needMapRank);
  HcclResult GenerateTaskPrivateDef(const ge::Node &node, HCCL_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                    domi::TaskDef &taskDef, const std::string sCollectiveType);
  HcclResult PrepareSelectAivParam(ge::Node &node, const std::string& sCollectiveType,
                                   int64_t &hcomComm, std::string &sGroup, u32 &rankSize,
                                   u64 &count, std::vector<int64_t> &counts, HcclDataType &dataType, HcclCMDType &opType,
                                   HcclReduceOp &reduction, u32 &aivCoreLimit);
  HcclResult JudgeIsAivMode(ge::Node &node, const std::string& sCollectiveType, bool &ifAiv);
  HcclResult GetCountsFromOpDesc(const ge::Node &node, std::vector<int64_t> &counts, HcclCMDType opType);
  HcclResult SetAttachedStreamInfoList(ge::Node &node, const std::string &group);  // 设置附属从流信息
  HcclResult TaskDefSetNumBlocks(const ge::Node &node, domi::TaskDef &taskDef, const std::string sCollectiveType,
                                 const u32 aivCoreLimit);
  int32_t optionFeatureBaseRefreshable_;
};
}  // namespace hccl
#endif  // GE_OPS_KERNEL_INFO_H
