/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_info_store_base.h"
#include <securec.h>
#include <functional>
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/types.h"
#include "hccl/hcom.h"
#include "auto_tuning_hcom_ops_kernel_builder.h"
#include "hcom_op_utils.h"

using namespace std;

namespace hccl {
HCCLOpsKernelInfoStore::HCCLOpsKernelInfoStore() {}

HCCLOpsKernelInfoStore::~HCCLOpsKernelInfoStore() {}

// initialize opsKernelInfoStore
ge::Status HCCLOpsKernelInfoStore::Initialize([[maybe_unused]] const map<string, string> &options) {
  // 直接返回, 有单独的初始化接口
  return ge::SUCCESS;
}

// close opsKernelInfoStore
ge::Status HCCLOpsKernelInfoStore::Finalize() {
  // 直接返回, 有单独的销毁接口
  return ge::SUCCESS;
}

// 检查HCCL是否支持传入的算子类型
bool HCCLOpsKernelInfoStore::CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &unSupportedReason) const {
  CHK_PRT_RET(!opDescPtr,
              HCCL_ERROR("[Check][Supported]errNo[0x%016llx] opDescPtr failed. null ptr.", HCOM_ERROR_CODE(HCCL_E_PTR)),
              false);
  unSupportedReason.clear();
  std::string sCollectiveType = opDescPtr->GetType();
  if (CheckSupportedOP(sCollectiveType) == HCCL_SUCCESS) {
    HCCL_INFO("hccl kernel info store support op[%s]", sCollectiveType.c_str());
    return true;
  } else {
    unSupportedReason = "hccl kernel info store dose not support op[" + sCollectiveType + "].";
    HCCL_INFO("%s", unSupportedReason.c_str());
    return false;
  }
}

HcclResult HCCLOpsKernelInfoStore::CheckSupportedOP(const std::string &sCollectiveType) const {
  std::vector<std::string>::const_iterator it;
  std::vector<std::string> hcclSupportOp;
  if (GetSupportedOP(hcclSupportOp) == HCCL_SUCCESS) {
    it = std::find(hcclSupportOp.begin(), hcclSupportOp.end(), sCollectiveType);
    return (it != hcclSupportOp.end()) ? HCCL_SUCCESS : HCCL_E_PARA;
  } else {
    return HCCL_E_PARA;
  }
}

// 返回HCCL算子信息库支持的算子
void HCCLOpsKernelInfoStore::GetAllOpsKernelInfo(std::map<string, ge::OpInfo> &infos) const {
  ge::OpInfo opinfo;
  HCCL_INFO("get all hccl ops kernel info start.");
  opinfo.engine = HCCL_OPS_ENGIN;
  // computeCost , flagPartial, flagAsync暂时返回固定值
  opinfo.computeCost = COMPUTE_COST_NUM;
  opinfo.flagPartial = false;
  opinfo.flagAsync = false;
  HcclResult ret = SetCustomKernelInfo(opinfo, infos);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("[Get][AllOpsKernelInfo]SetCustomKernelInfo failed, ret[%d].", ret);
  }
  HCCL_INFO("get all hccl ops kernel info end.");
}

HcclResult HCCLOpsKernelInfoStore::GetCollectiveTypeFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo,
                                                                 std::string &sCollectiveType) {
  if (CheckSupportedOP(hcclInfo.hccl_type) == HCCL_SUCCESS) {
    sCollectiveType = hcclInfo.hccl_type;
  } else {
    HCCL_ERROR(
        "[Get][CollectiveType]errNo[0x%016llx] get collective type from task info failed. HcclType[%s] is"
        "invalid.",
        HCOM_ERROR_CODE(HCCL_E_PARA), hcclInfo.hccl_type.c_str());
    return HCCL_E_PARA;
  }
  HCCL_INFO("get collective type[%s] from task info success.", hcclInfo.hccl_type.c_str());
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetCountFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, u64 &count) {
  count = hcclInfo.count;
  HCCL_INFO("get count[%llu] from task info success.", count);
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetStreamMainFromTaskInfo(const ge::GETaskInfo &taskDef, rtStream_t &stream) {
  stream = taskDef.stream;
  CHK_PRT_RET((stream == nullptr),
              HCCL_ERROR("[Get][Stream]errNo[0x%016llx] get stream failed. stream from task"
                         "info is null.",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetGlobalWorkSpaceAddrFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo,
                                                                      std::vector<void *> &globalWorkSpaceAddr) {
  globalWorkSpaceAddr.assign(hcclInfo.global_workspace_addr.begin(), hcclInfo.global_workspace_addr.end());
  CHK_PRT_RET((globalWorkSpaceAddr.size() == 0), HCCL_WARNING("[Get][globaladdr]get global workspace addr failed."),
              HCCL_SUCCESS);

  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetInputAddrFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo,
                                                            uintptr_t &inputAddr) {
  CHK_PTR_NULL(hcclInfo.inputDataAddr);
  inputAddr = (uintptr_t)(hcclInfo.inputDataAddr);
  HCCL_INFO("get input address[0x%016llx] from task info success.", inputAddr);
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetOutputAddrFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo,
                                                             uintptr_t &outputAddr) {
  CHK_PTR_NULL(hcclInfo.outputDataAddr);
  outputAddr = (uintptr_t)(hcclInfo.outputDataAddr);
  HCCL_INFO("get output address[0x%016llx] from task info success.", outputAddr);
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetRootFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo, u32 &root) {
  root = hcclInfo.rootId;
  HCCL_INFO("get root id[%u] from task info success.", root);
  return HCCL_SUCCESS;
}

HcclResult HCCLOpsKernelInfoStore::GetReduceTypeFromTaskInfo(const ge::GETaskKernelHcclInfo &hcclInfo,
                                                             HcclReduceOp &opType) {
  HCCL_DEBUG("GetReduceTypeFromTaskInfo optype is [%d]", hcclInfo.opType);
  if ((hcclInfo.opType >= HCCL_REDUCE_SUM) && (hcclInfo.opType < HCCL_REDUCE_RESERVED)) {
    opType = HcclReduceOp(hcclInfo.opType);
  } else {
    HCCL_ERROR(
        "[Get][ReduceType]errNo[0x%016llx] get optype from task info failed. optype[%s] "
        "is invalid.",
        HCOM_ERROR_CODE(HCCL_E_PARA), GetReduceOpEnumStr(static_cast<HcclReduceOp>(hcclInfo.opType)).c_str());
    return HCCL_E_PARA;
  }
  HCCL_INFO("get optype[%s] from task info success.", GetReduceOpEnumStr(opType).c_str());
  return HCCL_SUCCESS;
}
}  // namespace hccl
