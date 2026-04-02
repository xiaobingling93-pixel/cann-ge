/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>
#include "hcom_ops_kernel_builder.h"
#include "hcom_graph_optimizer.h"
#include "hcom_op_utils.h"
#include <securec.h>
#include <functional>
#include <vector>
#include <algorithm>
#include <memory>
#include <mutex>
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_local_context.h"
#include "framework/memory/memory_api.h"
#include "framework/common/ge_types.h"  // ge对外options
#include "hccl/hcom.h"
#include "register/ops_kernel_builder_registry.h"
#include "offline_build_config_parse.h"
#include "acl/acl_rt.h"
#include "mmpa/mmpa_api.h"

using namespace std;

namespace {
// 使用hcom_op_utils.h中定义的CreateDir枚举
using hccl::CreateDir;

// 获取LD_LIBRARY_PATH环境变量路径
HcclResult GetLdLibraryPath(std::string &libPath) {
  char *getPath = nullptr;
  // 从系统环境变量中获取LD_LIBRARY_PATH
  MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, getPath);
  if (getPath == nullptr) {
    HCCL_ERROR("[AIV][SKGetAlgPath][%s] ENV:LD_LIBRARY_PATH is not set", __func__);
    return HCCL_E_PARA;
  }
  libPath = getPath;
  return HCCL_SUCCESS;
}

HcclResult GetBinaryPathByDlAddr(std::string &binaryPath) {
  // 通过dladdr获取动态库的加载路径信息
  mmDlInfo infos;
  // 使用HcomGetRankSize函数地址作为参考点，获取包含该函数的动态库路径
  mmDladdr(reinterpret_cast<void *>(HcomGetRankSize), &infos);
  CHK_PRT_RET(infos.dli_fname == nullptr, HCCL_ERROR("[AIV][SKGetAlgPath][%s]get path of libhccl_plf.so failed", __func__),
              HCCL_E_UNAVAIL);

  // 将路径转换为绝对路径，解析符号链接
  char resolvedPath[PATH_MAX];
  if (realpath(infos.dli_fname, resolvedPath) == nullptr) {
    HCCL_ERROR("[AIV][SKGetAlgPath][%s]path %s is not a valid real path", __func__, infos.dli_fname);
    return HCCL_E_INTERNAL;
  }
  
  // 从完整路径中提取上级目录路径（向上跳过一级目录）
  std::string linkPath = resolvedPath;
  uint32_t linkPathSize = linkPath.length();
  uint32_t escapeLinkNum = 0;
  std::string midLinkPath;
  
  // 反转字符串以便从后向前查找目录分隔符
  std::reverse(linkPath.begin(), linkPath.end());
  for (uint32_t i = 0; i < linkPathSize; i++) {
    if ('/' == linkPath[i]) {
      midLinkPath = linkPath.substr(0, i + 1);
      escapeLinkNum += 1;
    }
    // 找到第一个目录分隔符后停止（向上跳过一级）
    if (escapeLinkNum == static_cast<uint32_t>(CreateDir::HCCL_DIR_NUM_ONE)) {
      break;
    }
  }
  
  // 恢复字符串顺序并提取上级目录路径
  std::reverse(linkPath.begin(), linkPath.end());
  binaryPath = linkPath.substr(0, linkPath.size() - midLinkPath.size());
  HCCL_DEBUG("[AIV][SKGetAlgPath][%s]op binary file path[%s]", __func__, binaryPath.c_str());
  return HCCL_SUCCESS;
}

void GetBinaryPathByLdLibraryPath(const std::string &libPath, const size_t mid, std::string &binaryPath) {
  // 从LD_LIBRARY_PATH中提取包含fwkacllib/lib64的路径段 LD_LIBRARY_PATH格式：path1:path2:path3
  u32 diff;
  if (libPath.find(":", mid) == libPath.npos) {
    // 如果mid之后没有冒号，说明是最后一个路径段
    diff = libPath.length() - libPath.rfind(":", mid);
  } else {
    // 如果mid之后有冒号，计算两个冒号之间的距离
    diff = libPath.find(":", mid) - libPath.rfind(":", mid);
  }
  // 提取路径段（跳过冒号）
  binaryPath = libPath.substr(libPath.rfind(":", mid) + 1, diff - 1);
}
} // namespace

namespace hccl {
const u32 DEFAULT_TASK_NUM = 254;
const std::string NO_CALCULATION = "_NO_CALCULATION";
static std::mutex g_taskNumCalModeMutex;
REGISTER_OPS_KERNEL_BUILDER(HCCL_OPS_LIB_NAME, hccl::HcomOpsKernelBuilder);
HcomOpsKernelBuilder::HcomOpsKernelBuilder() : optionFeatureBaseRefreshable_(0) {}

HcomOpsKernelBuilder::~HcomOpsKernelBuilder() {}

std::map<std::string, std::pair<std::string, std::string>> AivAlltoAllSuperKernelMap = {
    {"AlltoAllMeshAivSmallCountExecutor", {"/hccl_a2a_superkernel", "sk_alltoall"}},
    {"AlltoAllMeshAivExecutor", {"/hccl_a2a_superkernel", "sk_alltoall"}},
    {"AlltoAllMeshAivFor91093Executor", {"/hccl_sk_a2a_crossnode", "sk_alltoall_crossnode"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivAllGatherSuperKernelMap = {
    {"AllGatherMeshAivSmallCountExecutor", {"/hccl_ag_superkernel", "sk_allgather"}},
    {"AllGatherMeshAivExecutor", {"/hccl_ag_superkernel", "sk_allgather"}},
    {"AllGatherMeshAivFor91093Executor", {"/hccl_sk_ag_crossnode", "sk_allgather_crossnode"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivReduceScatterSuperKernelMap = {
    {"ReduceScatterMeshAivSmallCountExecutor", {"/hccl_rs_superkernel", "sk_reducescatter"}},
    {"ReduceScatterMeshAivExecutor", {"/hccl_rs_superkernel", "sk_reducescatter"}},
    {"ReduceScatterMeshAivFor91093Executor", {"/hccl_sk_rs_crossnode", "sk_reducescatter_crossnode"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivReduceScatterSuperKernelDeterMap = {
    {"ReduceScatterMeshAivFor91093Executor", {"/hccl_sk_rs_deter", "sk_reducescatter_deter"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivAllReduceSuperKernelMap = {
    {"AllReduceMeshAivSmallCountExecutor", {"/hccl_ar_superkernel", "sk_allreduce"}},
    {"AllReduceMeshAivExecutor", {"/hccl_ar_superkernel", "sk_allreduce"}},
    {"AllReduceMeshAivFor91093Executor", {"/hccl_sk_ar_crossnode", "sk_all_reduce_crossnode"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivAllReduceSuperKernelDeterMap = {
    {"AllReduceMeshAivFor91093Executor", {"/hccl_sk_ar_deter", "sk_allreduce_deter"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivAlltoAllSuperKernelMapV2 = {
    {"AivAlltoAllMesh1D", {"/hccl_a2a_superkernel_mesh_1d", "sk_alltoall_mesh_1d"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivAllGatherSuperKernelMapV2 = {
    {"AivAllGatherMesh1D", {"/hccl_ag_superkernel_mesh_1d", "sk_allgather_mesh_1d"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivReduceScatterSuperKernelMapV2 = {
    {"AivReduceScatterMesh1D", {"/hccl_rs_superkernel_mesh_1d", "sk_reducescatter_mesh_1d"}},
};

std::map<std::string, std::pair<std::string, std::string>> AivAllReduceSuperKernelMapV2 = {
    {"AivAllReduceMesh1DOneShot", {"/hccl_ar_superkernel_mesh_1d_oneshot", "sk_allreduce_mesh_1d_oneshot"}},
    {"AivAllReduceMesh1DTwoShot", {"/hccl_ar_superkernel_mesh_1d_twoshot", "sk_allreduce_mesh_1d_twoshot"}},
};

std::map<HcclCMDType, std::map<std::string, std::pair<std::string, std::string>>> AivSuperKernelMap = {
    {HcclCMDType::HCCL_CMD_ALLTOALL, AivAlltoAllSuperKernelMap},
    {HcclCMDType::HCCL_CMD_ALLGATHER, AivAllGatherSuperKernelMap},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AivReduceScatterSuperKernelMap},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, AivAllReduceSuperKernelMap},
};

std::map<HcclCMDType, std::map<std::string, std::pair<std::string, std::string>>> AivSuperKernelDeterMap = {
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AivReduceScatterSuperKernelDeterMap},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, AivAllReduceSuperKernelDeterMap},
};

std::map<HcclCMDType, std::map<std::string, std::pair<std::string, std::string>>> AivSuperKernelMapV2 = {
    {HcclCMDType::HCCL_CMD_ALLTOALL, AivAlltoAllSuperKernelMapV2},
    {HcclCMDType::HCCL_CMD_ALLGATHER, AivAllGatherSuperKernelMapV2},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AivReduceScatterSuperKernelMapV2},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, AivAllReduceSuperKernelMapV2},
};

// 返回运行参数，包括workspace 、stream数量以及atomic标志位
ge::Status HcomOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  HcclWorkflowMode lastWorkflowMode = HcomGetWorkflowMode();
  HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
  CHK_PRT_RET(
      !node.GetOpDesc(),
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetOpDesc failed. null ptr.", HCOM_ERROR_CODE(HCCL_E_PTR)),
      ge::INTERNAL_ERROR);

  bool unknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(node, unknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Calc][OpRunningParam]node[%s] get node unknown status failed", node.GetName().c_str()),
              ge::INTERNAL_ERROR);
  if (unknownShapeNode) {
    HCCL_INFO("node:%s is unknown shape, does not need to Calc Op Running Param", node.GetName().c_str());
    HcomSetWorkflowMode(lastWorkflowMode);
    return ge::SUCCESS;
  }

  HcclResult ret;
  std::string superKernel;
  ret = GetSuperKernelFromDesc(node.GetOpDesc(), superKernel);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetSuperKernelFromDesc failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);
  HCCL_INFO("SPK, superkernel is %s", superKernel.c_str());
  if (superKernel != "") {
    uint32_t graphId;
    ret = GetRootGraphID(node, graphId);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetRootGraphID failed.", HCOM_ERROR_CODE(ret)),
                ge::INTERNAL_ERROR);
    ge::AttrUtils::SetInt(node.GetOpDesc(), "hcom_graph_id", graphId);
  }

  ret = HcomCalcOpRunningParam(node);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] Calc Op Running Params failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = SetSuperKernelScopeAttr(node);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] SetSuperKernelScopeAttr failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  HcomSetWorkflowMode(lastWorkflowMode);
  return ge::SUCCESS;
}

// 检查算子是否支持
HcclResult HcomOpsKernelBuilder::CheckSupportedOP(const std::string &sCollectiveType) const {
  auto iter = HCCL_OPTYPE_NAME_MAP.find(sCollectiveType);
  return (iter != HCCL_OPTYPE_NAME_MAP.end()) ? HCCL_SUCCESS : HCCL_E_NOT_SUPPORT;
}

// 获取算子操作类型
HcclCMDType HcomOpsKernelBuilder::GetOpType(const std::string &sCollectiveType) const {
  auto iter = HCCL_OPTYPE_NAME_MAP.find(sCollectiveType);
  return (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;
}

// 检查SuperKernel资格以验证算子是否满足SuperKernel处理条件
HcclResult HcomOpsKernelBuilder::CheckSuperKernelEligibility(ge::Node &node, const ge::OpDescPtr &opDescPtr,
                                                            std::string &sCollectiveType,
                                                            std::string &superKernelScope, HcclCMDType &opType,
                                                            bool &needProcess) const {
  needProcess = false;
  
  // 步骤1：检查算子描述是否有效
  if (!opDescPtr) {
    HCCL_WARNING("desc of node[%s] is null.", node.GetName().c_str());
    return HCCL_SUCCESS;
  }

  // 步骤2：检查算子类型是否支持
  sCollectiveType = opDescPtr->GetType();
  if (CheckSupportedOP(sCollectiveType) != HCCL_SUCCESS) {
    HCCL_WARNING("op type[%s] not support super kernel", sCollectiveType.c_str());
    return HCCL_SUCCESS;
  }

  // 步骤3：检查是否具有super_kernel_scope属性
  if (!ge::AttrUtils::HasAttr(opDescPtr, "_super_kernel_scope")) {
    HCCL_INFO("SPK, [HcomOpsKernelBuilder][SetSuperKernelScopeAttr]node [%s] op type [%s] has no superKernelScope attr",
              node.GetName().c_str(), sCollectiveType.c_str());
    return HCCL_SUCCESS;
  }

  // 步骤4：获取super_kernel_scope属性值
  bool bRet = ge::AttrUtils::GetStr(opDescPtr, "_super_kernel_scope", superKernelScope);
  HCCL_INFO("SPK, [HcomOpsKernelBuilder][SetSuperKernelScopeAttr]node [%s] op type [%s] has superKernelScope attr[%s]",
            node.GetName().c_str(), sCollectiveType.c_str(), superKernelScope.c_str());
  CHK_PRT_RET(
      !bRet,
      HCCL_ERROR("[HcomOpsKernelBuilder][SetSuperKernelScopeAttr]node [%s] GetStr superKernelScope failed, op type[%s]",
                node.GetName().c_str(), sCollectiveType.c_str()),
      HCCL_E_PARA);

  // 步骤5：检查算子是否在AIV SuperKernel支持列表中
  opType = GetOpType(sCollectiveType);
  bool isSupportOP = AivSuperKernelMap.find(opType) != AivSuperKernelMap.end();
  if (!isSupportOP || optionFeatureBaseRefreshable_ == 1) {
    HCCL_WARNING("super kernel not support opType[%d] optionFeatureBaseRefreshable_[%d]", opType,
                optionFeatureBaseRefreshable_);
    opDescPtr->DelAttr("_super_kernel_scope");
    return HCCL_SUCCESS;
  }

  needProcess = true;
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAivSuperKernelBinaryAttrFor950(const ge::OpDescPtr &opDescPtr, HcclCMDType opType,
                                                              HcclDataType dataType, const std::string &algName,
                                                              std::string &funcName, const std::string & binPath) const {
  auto itMap = AivSuperKernelMapV2.find(opType);
  auto it = (itMap->second).find(algName);
  if (it != (itMap->second).end()) {
    std::string binFilePath = binPath + it->second.first + "_" + GetDataTypeEnumStr(dataType) + ".o";
    ge::AttrUtils::SetStr(opDescPtr, "bin_file_path", binFilePath);
    ge::AttrUtils::SetStr(opDescPtr, "hcom_bin_file_path", binFilePath);
    funcName = it->second.second;
    ge::AttrUtils::SetStr(opDescPtr, "hcom_func_name", funcName + "_" + GetDataTypeEnumStr(dataType));
  } else {
    HCCL_WARNING("no support alg, del superKernelScope attr");
    opDescPtr->DelAttr("_super_kernel_scope");
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAivSuperKernelBinaryAttrForDeter(const ge::OpDescPtr &opDescPtr, HcclCMDType opType,
                                                              const std::string &algName, std::string &funcName,
                                                              const std::string & binPath) const {
  auto itMap = AivSuperKernelDeterMap.find(opType);
  auto it = (itMap->second).find(algName);
  if (it != (itMap->second).end()) {
    std::string binFilePath = binPath + it->second.first + ".o";
    ge::AttrUtils::SetStr(opDescPtr, "bin_file_path", binFilePath);
    ge::AttrUtils::SetStr(opDescPtr, "hcom_bin_file_path", binFilePath);
    funcName = it->second.second;
    ge::AttrUtils::SetStr(opDescPtr, "hcom_func_name", funcName);
  } else {
    HCCL_WARNING("no support aiv, del superKernelScope attr");
    opDescPtr->DelAttr("_super_kernel_scope");
  }
  return HCCL_SUCCESS;
}

// 设置AIV SuperKernel二进制属性
HcclResult HcomOpsKernelBuilder::SetAivSuperKernelBinaryAttrs(const ge::OpDescPtr &opDescPtr, HcclCMDType opType,
                                                              HcclDataType dataType, const std::string &algName,
                                                              std::string &funcName) {
  // 步骤1：标记算子使用HCCL superkernel路径
  ge::AttrUtils::SetBool(opDescPtr, "_hccl", true);
  std::string binPath;
  CHK_RET(SKGetAlgPath(opType, binPath));
  // 步骤2：获取SOC_VERSION
  std::string socVersion{};
  if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("[HcomOpsKernelBuilder][SetAivSuperKernelBinaryAttrs] get soc version failed");
    return HCCL_E_NOT_FOUND;
  }
  // 步骤3：根据SOC_VERSION和确定性配置选择不同的二进制文件路径
  if (socVersion.find("Ascend950") != std::string::npos) {
    // 950架构下，使用AIV SuperKernelV2 Map进行二进制文件路径设置
    CHK_RET(SetAivSuperKernelBinaryAttrFor950(opDescPtr, opType, dataType, algName, funcName, binPath));
  } else {
    u8 deterministic = DETERMINISTIC_DISABLE;
    CHK_RET(GetDeterministic(deterministic));
    bool isDeterOptype = (opType == HCCL_CMD_ALLREDUCE || opType == HCCL_CMD_REDUCE_SCATTER);
    // 需要先判断是否确定性使能，再判断是否为确定性算子
    if ((deterministic != DETERMINISTIC_DISABLE) && isDeterOptype) {
      // 确定性算子，使用AIV SuperKernelDeterMap进行二进制文件路径设置
      CHK_RET(SetAivSuperKernelBinaryAttrForDeter(opDescPtr, opType, algName, funcName, binPath));
    } else {
      // 非确定性算子，使用AIV SuperKernelMap进行二进制文件路径设置
      auto itMap = AivSuperKernelMap.find(opType);
      auto it = (itMap->second).find(algName);
      if (it != (itMap->second).end()) {
        if (std::string(algName).find("91093") == std::string::npos) {
          std::string binFilePath = binPath + it->second.first + "_" + GetDataTypeEnumStr(dataType) + ".o";
          ge::AttrUtils::SetStr(opDescPtr, "bin_file_path", binFilePath);
          ge::AttrUtils::SetStr(opDescPtr, "hcom_bin_file_path", binFilePath);
          funcName = it->second.second;
          ge::AttrUtils::SetStr(opDescPtr, "hcom_func_name", funcName + "_" + GetDataTypeEnumStr(dataType));
        } else {
          std::string binFilePath = binPath + it->second.first + ".o";
          ge::AttrUtils::SetStr(opDescPtr, "bin_file_path", binFilePath);
          ge::AttrUtils::SetStr(opDescPtr, "hcom_bin_file_path", binFilePath);
          funcName = it->second.second;
          ge::AttrUtils::SetStr(opDescPtr, "hcom_func_name", funcName);
        }
      } else {
        HCCL_WARNING("no support aiv, del superKernelScope attr");
        opDescPtr->DelAttr("_super_kernel_scope");
      }
    }
  }
  return HCCL_SUCCESS;
}

// 设置SuperKernel Block维度以计算并设置AIV核数（block维度）
HcclResult HcomOpsKernelBuilder::SetSuperKernelBlockDim(const ge::OpDescPtr &opDescPtr, const std::string &group,
                                                        HcclCMDType opType, u64 count, void *counts, HcclDataType dataType,
                                                        u32 aivCoreLimit, char *algName, u32 rankSize) const {
  // 计算AIV核数
  u32 blockDim;
  CHK_RET(HcomCalcAivCoreNum(group.c_str(), opType, count, counts, dataType, aivCoreLimit, algName, &blockDim));
  
  // 设置block维度属性
  ge::AttrUtils::SetInt(opDescPtr, "hcom_block_dim", blockDim);
  HCCL_INFO("[HcomOpsKernelBuilder][%s] rankSize[%u] aivCoreLimit[%u] blockDim[%u]", __func__, rankSize,
            aivCoreLimit, blockDim);
  return HCCL_SUCCESS;
}

// 设置SuperKernel Scope属性（主入口函数）为支持SuperKernel的算子设置相关属性，包括二进制文件路径、函数名和block维度
HcclResult HcomOpsKernelBuilder::SetSuperKernelScopeAttr(ge::Node &node) {
  HCCL_INFO("SPK, start set SuperKernelScopeAttr.");
  std::string superKernelScope;
  std::string sCollectiveType;
  HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
  bool needProcess = false;
  auto opDescPtr = node.GetOpDesc();

  // 步骤1：入口校验与superkernel条件判定
  CHK_RET(CheckSuperKernelEligibility(node, opDescPtr, sCollectiveType, superKernelScope, opType, needProcess));
  if (!needProcess) {
    return HCCL_SUCCESS;
  }
  
  // 步骤2：调用JudgeIsAivMode函数获后续SetAivSuperKernelBinaryAttrs和SetSuperKernelBlockDim接口所需参数
  bool ifAiv = false;
  int64_t hcomComm = 0;
  std::string sGroup;
  u32 rankSize = 0;
  u64 count = 0;
  std::vector<int64_t> counts;
  HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
  HcclReduceOp reduction = HcclReduceOp::HCCL_REDUCE_SUM;
  u32 aivCoreLimit = 0;
  char algName[ALG_NAME_MAX_LEN];
  
  // 用于判断是否走 Aiv 的参数准备
  CHK_RET(PrepareSelectAivParam(node, sCollectiveType, hcomComm, sGroup, rankSize,
          count, counts, dataType, opType, reduction, aivCoreLimit));
  void *countsPtr = counts.data();
  CHK_RET(HcomSelectAlg(hcomComm, sGroup.c_str(), count, countsPtr, dataType, reduction, opType, aivCoreLimit, ifAiv,
                          algName)); 
  
  if (!ifAiv) {
    HCCL_INFO("no support aiv, del superKernelScope attr");
    opDescPtr->DelAttr("_super_kernel_scope");
    return HCCL_SUCCESS;
  }

  // 步骤3：设置superkernel二进制属性并计算block维度
  std::string funcName;
  CHK_RET(SetAivSuperKernelBinaryAttrs(opDescPtr, opType, dataType, algName, funcName));
  CHK_RET(SetSuperKernelBlockDim(opDescPtr, sGroup, opType, count, countsPtr, dataType, aivCoreLimit, algName, rankSize));
  HCCL_INFO("[HcomOpsKernelBuilder][SetSuperKernelScopeAttr] Support SPK Optype[%s] funcName[%s]",
              sCollectiveType.c_str(), funcName.c_str());
  return HCCL_SUCCESS;
}

// 获取SuperKernel算法路径以AIV SuperKernel二进制文件的存放路径
HcclResult HcomOpsKernelBuilder::SKGetAlgPath(HcclCMDType opType, std::string &binaryPath) const {
  HCCL_DEBUG("[AIV][SKGetAlgPath] opType[%d] binaryPath[%s]", opType, binaryPath.c_str());
  std::string libPath;
  CHK_RET(GetLdLibraryPath(libPath));

  // 查找fwkacllib/lib64路径段
  size_t mid = libPath.find("fwkacllib/lib64");
  if (mid == libPath.npos) {
    // 如果LD_LIBRARY_PATH中没有fwkacllib/lib64，使用dladdr方式获取路径
    HCCL_WARNING("[AIV][SKGetAlgPath]ENV:LD_LIBRARY_PATH lack fwkacllib/lib64");
    return GetBinaryPathByDlAddr(binaryPath);
  }
  
  // 从LD_LIBRARY_PATH中提取包含fwkacllib/lib64的路径段
  GetBinaryPathByLdLibraryPath(libPath, mid, binaryPath);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetNeedMapRankFromDesc(const ge::OpDescPtr &op, bool &needMapRank) {
  // ATTR_NAME_NEED_MAP_RANK_ID
  if (ge::AttrUtils::HasAttr(op, ge::ATTR_NAME_NEED_MAP_RANK_ID)) {
    if (ge::AttrUtils::GetBool(op, ge::ATTR_NAME_NEED_MAP_RANK_ID, needMapRank) == false) {
      HCCL_ERROR(
          "[Get][needMapRank]errNo[0x%016llx]: get need map rank failed. get \"need map rank\" from"
          "opDesc failed",
          HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  }
  HCCL_INFO("[Get][needMapRank] needMapRank[%u] success.", needMapRank);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetSuperKernelFromDesc(const ge::OpDescPtr &op, std::string &superKernel) {
  // ATTR_NAME_NEED_MAP_RANK_ID
  if (ge::AttrUtils::HasAttr(op, "_super_kernel_scope")) {
    if (ge::AttrUtils::GetStr(op, "_super_kernel_scope", superKernel) == false) {
      HCCL_ERROR(
          "[Get][superKernel]errNo[0x%016llx]: get superKernel failed. get \"superKernel\" from"
          "opDesc failed",
          HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  }
  HCCL_INFO("[Get][superKernel] superKernel[%s] success.", superKernel.c_str());
  return HCCL_SUCCESS;
}

ge::Status HcomOpsKernelBuilder::GenerateTask([[maybe_unused]] const ge::Node &node, [[maybe_unused]] ge::RunContext &runContext,
                                              std::vector<domi::TaskDef> &taskDefList) {
  bool unknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(node, unknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Generate][Task]node[%s] get node unknown status failed", node.GetName().c_str()),
              HCCL_E_PARA);
  if (unknownShapeNode) {
    HCCL_INFO("op:%s is unknown shape, does not need to generate Task.", node.GetName().c_str());
    return HCCL_SUCCESS;
  }

  CHK_PRT_RET(!node.GetOpDesc(),
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] opDesc is null.", HCOM_ERROR_CODE(HCCL_E_PTR)),
              ge::INTERNAL_ERROR);
  std::string sCollectiveType = node.GetOpDesc()->GetType();
  HcclResult ret = CheckSupportedOP(sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] op type[%s] is not supported.", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);

  // 获取 hcom 必需的参数
  HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
  std::string nodeName = node.GetOpDesc()->GetName();
  CHK_PRT_RET(nodeName.empty(),
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] op[%s] get tag name failed. node name"
                         "is empty.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str()),
              ge::INTERNAL_ERROR);
  std::hash<std::string> hashString;
  // node name长度不确定，将node name转换为hash值
  privateDefBuf.nodeNameHash = hashString(nodeName);

  ret = GetRootGraphID(node, privateDefBuf.graphId);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] node[%s] find root graph failed", HCOM_ERROR_CODE(ret),
                         nodeName.c_str()),
              ge::INTERNAL_ERROR);

  std::string sGroup;
  int64_t hcomComm = 0;
  ret = GetCommFromOpDesc(node.GetOpDesc(), hcomComm, sGroup);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get comm and group failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  if (hcomComm != 0) {
    privateDefBuf.comm = hcomComm;
  } else {
    s32 sret = memcpy_s(&privateDefBuf.group[0], sizeof(privateDefBuf.group), sGroup.c_str(), (sGroup.length() + 1));
    CHK_PRT_RET(sret != EOK,
                HCCL_ERROR("[Generate][Task]errNo[0x%016llx] memcpy failed. ret[%d],"
                           "params:destMaxSize[%zu],count[%zu]",
                           HCOM_ERROR_CODE(HCCL_E_MEMORY), sret, sizeof(privateDefBuf.group), (sGroup.length() + 1)),
                ge::INTERNAL_ERROR);
  }

  // aicpu/mc2算子统一设置group属性，GE保证通信域内aicpu和mc2的kernel展开时序
  std::vector<std::string> groupList(1, sGroup);
  CHK_PRT_RET(!ge::AttrUtils::SetListStr(node.GetOpDesc(), "_hccl_group_id_list", groupList),
              HCCL_ERROR("[Generate][Task]Set group id list attr for current node failed, group:%s", sGroup.c_str()),
              ge::INTERNAL_ERROR);

  ret = GetSrcRankFromDesc(node.GetOpDesc(), privateDefBuf.srcRank, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get src_rank failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = GetDestRankFromDesc(node.GetOpDesc(), privateDefBuf.destRank, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get dest_rank failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = GetSrTagFromDesc(node.GetOpDesc(), privateDefBuf.srTag, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get sr_tag failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = GetOriginalGraphShapeTypeFromDesc(node.GetOpDesc(), privateDefBuf.originalGraphShapeType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get shapeType failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = HcomOpUtils::ConversionOpDataType(node.GetOpDesc(), sCollectiveType, privateDefBuf.dataType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] conversion op data type failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

  privateDefBuf.tensorNum = 0;
  constexpr const char *kCleanSeparately = "1";
  std::string atomic_clean_policy;
  bool needCleanSeparately =
      (ge::GetThreadLocalContext().GetOption(ge::ATOMIC_CLEAN_POLICY, atomic_clean_policy) == ge::GRAPH_SUCCESS) &&
      (atomic_clean_policy == kCleanSeparately);
  if (needCleanSeparately &&
      ((sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) || (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) ||
       (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE))) {
    // 获取Tensor的个数
    privateDefBuf.tensorNum = node.GetOpDesc()->GetInputsSize();
  }
  ret = GetNeedMapRankFromDesc(node.GetOpDesc(), privateDefBuf.needMapRank);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] get need map rank failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  if (IsOfflineCompilation()) {
    privateDefBuf.isOfflineComp = true;
    CHK_RET(GetOffDeviceTypeWithoutDev(privateDefBuf.devType));
    HCCL_DEBUG("GenerateTask: isOfflineComp[%u] devType[%u]", privateDefBuf.isOfflineComp, privateDefBuf.devType);
  }

  CHK_RET(HcomOpUtils::GetAivCoreLimit(node.GetOpDesc(), sCollectiveType, privateDefBuf.aivCoreLimit));

  HCCL_RUN_INFO(
      "GenerateTask: graph[%u], node[%s]-hash[%zu], opType[%s], opID[%d], comm[%lld], group[%s], "
      "srcRank[%u], dstRank[%u], srTag[%u], dataType[%s], aivCoreLimit[%u].",
      privateDefBuf.graphId, nodeName.c_str(), privateDefBuf.nodeNameHash, sCollectiveType.c_str(),
      node.GetOpDesc()->GetId(), hcomComm, sGroup.c_str(), privateDefBuf.srcRank, privateDefBuf.destRank,
      privateDefBuf.srTag, GetDataTypeEnumStr(privateDefBuf.dataType).c_str(), privateDefBuf.aivCoreLimit);

  domi::TaskDef taskDef;
  ret = GenerateTaskPrivateDef(node, privateDefBuf, taskDef, sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] generate taskprivatedef failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = GenerateTaskDef(node, privateDefBuf, taskDef);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] generate taskdef failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  ret = TaskDefSetNumBlocks(node, taskDef, sCollectiveType, privateDefBuf.aivCoreLimit);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Generate][Task]errNo[0x%016llx] taskdef set numBlocks failed.", HCOM_ERROR_CODE(ret)),
              ge::INTERNAL_ERROR);

  taskDefList.push_back(taskDef);
  return ge::SUCCESS;
}

// 获取算子的counts，为后续选择算法和设置核数做准备
HcclResult HcomOpsKernelBuilder::GetCountsFromOpDesc(const ge::Node &node, std::vector<int64_t> &counts, HcclCMDType opType) {
  if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
    std::vector<int64_t> sendCounts;
    std::vector<int64_t> recvDispls;
    // allgatherV的counts代表recvCount，因为算子是收集操作，是recv从所有ranks
    HcomOpUtils::GetAllGatherVCountsDispl(const_cast<ge::Node &>(node), sendCounts, counts, recvDispls);
    // 不能通过是否为空来判断是否成功获取到数据量，因为根据vector标准库的定义，
    // 就算vector没有元素，其指针可能为空也可能不为空，需要根据recvCounts的size来判断是否成功获取到数据量
    if (counts.empty()) {
      HCCL_ERROR("[TaskDefSetNumBlocks][GetCountsFromOpDesc], counts is empty");
      return HCCL_E_PTR;
    }
  } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
    std::vector<int64_t> sendDispls;
    std::vector<int64_t> recvCount;
    // reducescatterV的counts代表sendCount，因为算子是散射操作，是send给所有ranks
    HcomOpUtils::GetReduceScatterVCountsDispl(const_cast<ge::Node &>(node), counts, sendDispls, recvCount);
    if (counts.empty()) {
      HCCL_ERROR("[TaskDefSetNumBlocks][GetCountsFromOpDesc], counts is empty");
      return HCCL_E_PTR;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::TaskDefSetNumBlocks(const ge::Node &node, domi::TaskDef &taskDef,
                                                     const std::string sCollectiveType, const u32 aivCoreLimit) {
  // 离线模式不设置核数
  if (IsOfflineCompilation()) {
    HCCL_DEBUG("[TaskDefSetNumBlocks] IsOfflineCompilation, not set numBlocks");
    return HCCL_SUCCESS;
  }

  u64 count = 0;
  HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
  std::string group = "";
  int64_t comm = 0;
  bool ifAiv = false;
  char algName[ALG_NAME_MAX_LEN];
  HcclReduceOp reduction = HcclReduceOp::HCCL_REDUCE_SUM;

  auto opDescPtr = node.GetOpDesc();
  if (!opDescPtr) {
    HCCL_ERROR("desc of node[%s] is null.", node.GetName().c_str());
    return HCCL_E_PARA;
  }

  auto iter = HCCL_OPTYPE_NAME_MAP.find(sCollectiveType);
  HcclCMDType opType = (iter != HCCL_OPTYPE_NAME_MAP.end()) ? iter->second : HcclCMDType::HCCL_CMD_INVALID;

  auto ret = HcomOpUtils::ConversionOpDataType(opDescPtr, sCollectiveType, dataType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Get][TaskDefSetNumBlocks]op[%s]: get data type failed. ret[%d]", sCollectiveType.c_str(), ret), ret);

  CHK_RET(GetCommFromOpDesc(opDescPtr, comm, group));
  // 获取rankSize，用于计算数据量
  u32 rankSize = 0;
  CHK_RET(HcomGetRankSize(group.c_str(), &rankSize));
  // 计算数据量，后续根据数据量选择算法并设置核数
  ret = HcomOpUtils::GetAccuracyCountFromOpDesc(opDescPtr, sCollectiveType, dataType, count, rankSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][TaskDefSetNumBlocks]op[%s]: get count failed. ret[%d]", sCollectiveType.c_str(), ret),
              ret);

  if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
    CHK_RET(HcomOpUtils::GetReduction(opDescPtr, reduction));
  }

  std::vector<int64_t> counts;
  CHK_RET(GetCountsFromOpDesc(node, counts, opType));

  void *countsPtr = counts.data();
  CHK_RET(HcomSelectAlg(comm, group.c_str(), count, countsPtr, dataType, reduction, opType, aivCoreLimit, ifAiv, algName));

  // 非AIV算法不设置核数
  if (!ifAiv) {
    HCCL_DEBUG("[TaskDefSetNumBlocks] not Aiv, do not set numBlocks");
    return HCCL_SUCCESS;
  }

  u32 numBlocks = 0;
  CHK_RET(HcomCalcAivCoreNum(group.c_str(), opType, count, countsPtr, dataType, aivCoreLimit, algName, &numBlocks));

  domi::KernelHcclDef *kernelDefHccl = taskDef.mutable_kernel_hccl();
  CHK_PRT_RET((kernelDefHccl == nullptr),
              HCCL_ERROR("[Generate][Task]node[%s]: kernelDefHccl is null.", node.GetOpDesc()->GetName().c_str()),
              HCCL_E_PTR);

  kernelDefHccl->set_aiv_block_dim(numBlocks);
  HCCL_INFO("[TaskDefSetNumBlocks] %s set numBlocks %d success", sCollectiveType.c_str(), numBlocks);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GenerateTaskPrivateDef(const ge::Node &node,
                                                        HCCL_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                                        domi::TaskDef &taskDef, const std::string sCollectiveType) {
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV || sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC ||
      sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALL) {
    HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF alltoallvPrivateDefBuf(privateDefBuf);
    CHK_RET(SetAlltoAllVParams(node, alltoallvPrivateDefBuf, sCollectiveType));
    taskDef.set_private_def(&alltoallvPrivateDefBuf, sizeof(HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
    HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF reducescattervPrivateDefBuf(privateDefBuf);
    CHK_RET(SetReduceScatterVParams(node, reducescattervPrivateDefBuf));
    taskDef.set_private_def(&reducescattervPrivateDefBuf, sizeof(HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHERV) {
    HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF allgathervPrivateDefBuf(privateDefBuf);
    CHK_RET(SetAllGatherVParams(node, allgathervPrivateDefBuf));
    taskDef.set_private_def(&allgathervPrivateDefBuf, sizeof(HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF));
  } else {
    if (privateDefBuf.tensorNum == 0) {
      HCCL_DEBUG("[Generate][TaskPrivateDef]Not Delivering a Clearing Task.");
      taskDef.set_private_def(&privateDefBuf, sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    } else {
      HCCL_DEBUG("[Generate][TaskPrivateDef]Delivering Clearing Task.");
      CHK_RET(SetPrivateDefWithTensorInfo(node, privateDefBuf, taskDef));
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::HcomCalcOpRunningParam(ge::Node &node) {
  HCCL_INFO("calculate hccl runing parameters start.");

  HcclResult ret;
  HcomOpParam hcomOpParam;
  HcomResResponse hcomResResponse;
  std::string sCollectiveType;
  std::string sGroup;
  std::string socVersion;
  std::vector<int64_t> sendCountMatrix;
  std::vector<int64_t> sendCounts;
  std::vector<int64_t> sendDispls;
  std::vector<int64_t> recvCounts;
  std::vector<int64_t> recvDispls;
  std::vector<u32> curRanks;
  std::string rankTableStr;
  std::string rankTableM;

  CHK_RET(SetHcomOpParam(node, &hcomOpParam, sCollectiveType, sGroup, socVersion, sendCountMatrix, sendCounts,
                         sendDispls, recvCounts, recvDispls, curRanks, rankTableStr, rankTableM));

  if (IsOfflineCompilation() || hcomOpParam.groupListSize != 0) {
    CHK_RET(HcomCalcOpResOffline(&hcomOpParam, &hcomResResponse));
  } else {
    CHK_RET(HcomCalcOpOnline(&hcomOpParam, &hcomResResponse));
  }

  std::string nodeName = node.GetName();
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_SEND || sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE ||
      (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST && nodeName.find(NO_CALCULATION) != std::string::npos)) {
    // 重新刷新从流为0
    hcomResResponse.streamNum = 0;
  }

  if (ge::AttrUtils::SetInt(node.GetOpDesc(), "used_stream_num", hcomResResponse.streamNum) == false) {
    HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op[%s]: set stream number[%llu] to OpDesc failed.", HCCL_E_PARA,
               hcomOpParam.opType, hcomResResponse.streamNum);
    return HCCL_E_INTERNAL;
  }

  // 计算清零task数量，累加到hcomResResponse算出的taskNum
  u32 taskNum = static_cast<u32>(hcomResResponse.taskNum);
  u32 cleanTaskNum = 0;
  CHK_RET(HcomOpUtils::GetTensorCleanTaskNum(node, sCollectiveType, cleanTaskNum));
  taskNum += cleanTaskNum;
  if (ge::AttrUtils::SetInt(node.GetOpDesc(), "_hccl_task_num", taskNum) == false) {
    HCCL_ERROR("[HcomCalc][OpRunningParam]errNo[0x%016llx] op[%s]: set _hccl_task_num to OpDesc failed.", HCCL_E_PARA,
               hcomOpParam.opType);
    return HCCL_E_PARA;
  }

  CHK_RET(SetOpWorkerSpaceForKnowShape(node, hcomResResponse.opMemSize));
  ret = SetOpMemAttr(node, node.GetOpDesc()->GetType(), hcomResResponse.opMemSize);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set node[%s] mem attr failed.", ret, node.GetName().c_str()),
      HCCL_E_INTERNAL);

  HCCL_INFO(
      "[Calc][OpRunningParam] node[%s] calculate hccl runing parameters completed. stream num:[%llu], workspace "
      "size:[%llu]bytes",
      node.GetName().c_str(), hcomResResponse.streamNum, hcomResResponse.opMemSize);
  HCCL_INFO("GetAndSetTaskNum success. task num:[%llu]", taskNum);

  // 设置output size 大小
  ret = SetOpOutputMemSize(node, hcomOpParam.opType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set op[%s] output size failed.", ret, hcomOpParam.opType),
      HCCL_E_INTERNAL);

  // 设定atomic index参数
  ret = SetOpAtomicInputIndex(node, hcomOpParam.opType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set op[%s] atomic input index failed.", ret,
                         hcomOpParam.opType),
              HCCL_E_INTERNAL);
  CHK_RET(SetAttachedStreamInfoList(node, sGroup));
  return HCCL_SUCCESS;
}

// PrepareSelectAivParam函数，返回多个计算参数用于判断是否支持AIV展开模式
HcclResult HcomOpsKernelBuilder::PrepareSelectAivParam(ge::Node &node, const std::string& sCollectiveType,
                                                int64_t &hcomComm, std::string &sGroup, u32 &rankSize,
                                                u64 &count, std::vector<int64_t> &counts, HcclDataType &dataType, HcclCMDType &opType,
                                                HcclReduceOp &reduction, u32 &aivCoreLimit) {
  auto const opDescPtr = node.GetOpDesc();
  // 获取通信域标识符
  CHK_RET(GetCommFromOpDesc(opDescPtr, hcomComm, sGroup));
  // 获取通信域名称
  if (sGroup.empty()) {
    HCCL_INFO("[%s] group is empty, try to get from op desc.", __func__);
    CHK_RET(HcomOpUtils::GetGroupFromOpDesc(opDescPtr, sGroup));
  }
  HCCL_INFO("[%s] hcomComm[%d], group[%s]", __func__, hcomComm, sGroup.c_str());

  // 获取rankSize
  CHK_RET(HcomGetRankSize(sGroup.c_str(), &rankSize));
  // 获取准确的 datatype
  HcclResult ret = HcomOpUtils::ConversionOpDataType(opDescPtr, sCollectiveType, dataType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[%s][Get][OpWorkspaceMemSize]op[%s]: get data type failed. ret[%d]", __func__,
                         sCollectiveType.c_str(), ret),
              ret);

  // 获取 opType
  opType = GetOpType(sCollectiveType);
  // 获取reduction
  reduction = HcclReduceOp::HCCL_REDUCE_SUM;
  if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
    CHK_RET(HcomOpUtils::GetReduction(opDescPtr, reduction));
  }

  // 获取 aivCoreLimit
  CHK_RET(HcomOpUtils::GetAivCoreLimit(opDescPtr, sCollectiveType, aivCoreLimit));
  // 获取准确的count
  ret = HcomOpUtils::GetAccuracyCountFromOpDesc(opDescPtr, sCollectiveType, dataType, count, rankSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[%s][Get][Count]op[%s]: get count failed. ret[%d]", __func__, sCollectiveType.c_str(), ret),
              ret);

  // 获取 counts
  CHK_RET(GetCountsFromOpDesc(node, counts, opType));

  HCCL_INFO("[%s] hcomComm[%d], group[%s], count[%u], counts[%zu], dataType[%u], reduction[%u], opType[%u]",
            __func__, hcomComm, sGroup.c_str(), count, counts.size(), dataType, reduction, opType);
  return HCCL_SUCCESS;
}

// JudgeIsAivMode函数，返回是否走 Aiv 模式
HcclResult HcomOpsKernelBuilder::JudgeIsAivMode(ge::Node &node, const std::string& sCollectiveType,
                                                bool &ifAiv) {
  int64_t hcomComm = 0;
  std::string sGroup;
  u32 rankSize = 0;
  u64 count = 0;
  std::vector<int64_t> counts;
  HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
  HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
  HcclReduceOp reduction = HcclReduceOp::HCCL_REDUCE_SUM;
  u32 aivCoreLimit = 0;
  char algName[ALG_NAME_MAX_LEN];
  
  // 用于判断是否走 Aiv 的参数准备
  CHK_RET(PrepareSelectAivParam(node, sCollectiveType, hcomComm, sGroup, rankSize,
          count, counts, dataType, opType, reduction, aivCoreLimit));
  // 判断是否走 Aiv
  DevType devType = HcomGetDeviceType();
  if (devType != DevType::DEV_TYPE_950) {
    void *countsPtr = counts.data();
    CHK_RET(HcomSelectAlg(hcomComm, sGroup.c_str(), count, countsPtr, dataType, reduction, opType, aivCoreLimit, ifAiv,
                          algName));
  } else {
    // 950按照原先流程，如果是 SuperKernel 那就肯定是 Aiv 模式了
    std::string superKernel{};
    auto const opDesc = node.GetOpDesc();
    CHK_RET(GetSuperKernelFromDesc(opDesc, superKernel));
    ifAiv = superKernel != "" ? true : false;
    HCCL_INFO("[%s] SPK, superkernel is %s, ifAiv[%d]", __func__, superKernel.c_str(), ifAiv);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAttachedStreamInfoList(ge::Node &node, const string &group) {
  const uint32_t STREAM_CONFIG_NAME = 0;
  const uint32_t STREAM_CONFIG_REUSE_KEY = 1;
  const uint32_t STREAM_CONFIG_STREAM_NAMED_ATTRS = 2;
  const uint32_t STREAM_ATTACHED_TASK_NUM = 2;

  ge::GeAttrValue::NAMED_ATTRS attachedGraphStream;
  ge::GeAttrValue::NAMED_ATTRS attachedGroupStream;
  // 多个通信域都用同一条图粒度的流；每个通信域用一条流
  const std::vector<std::tuple<std::string, std::string, ge::GeAttrValue::NAMED_ATTRS *>> streamConfigs = {
      std::make_tuple(std::string("hccl_attached_graph_stream"), std::string("hccl_attached_graph_stream"),
                      &attachedGraphStream),
      std::make_tuple(std::string("hccl_attached_group_stream"), std::string("hccl_attached_group_stream") + group,
                      &attachedGroupStream)};

  // 目前HCCL是否必须要申请从流，真表示如果从流申请失败则失败
  bool required = true;
  std::vector<ge::GeAttrValue::NAMED_ATTRS> attachedStreamInfo;

  // 判断是否是 AIV 展开模式
  bool ifAiv = false;
  CHK_PRT(JudgeIsAivMode(node, node.GetOpDesc()->GetType(), ifAiv)); 
  HCCL_INFO("[%s] ifAiv[%d] should %s set attached stream info", __func__, ifAiv, ifAiv ? "not" : "");

  // AIV 模式不设置从流信息
  if (!ifAiv) {
    const auto opDesc = node.GetOpDesc();
    (void)ge::AttrUtils::SetInt(opDesc, ge::ATTR_NAME_HCCL_ATTACHED_TASK_NUM, STREAM_ATTACHED_TASK_NUM);
    HCCL_INFO("%s HcclOp set STREAM_ATTACHED_TASK_NUM[%u]", __func__, STREAM_ATTACHED_TASK_NUM);  // 从流设置 task_num
    for (auto &config : streamConfigs) {
      std::string name = std::get<STREAM_CONFIG_NAME>(config);
      std::string reuseKey = std::get<STREAM_CONFIG_REUSE_KEY>(config);
      ge::GeAttrValue::NAMED_ATTRS &streamAttr = *std::get<STREAM_CONFIG_STREAM_NAMED_ATTRS>(config);

      (void)ge::AttrUtils::SetStr(streamAttr, ge::ATTR_NAME_ATTACHED_RESOURCE_NAME, name);
      (void)ge::AttrUtils::SetStr(streamAttr, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, reuseKey);
      (void)ge::AttrUtils::SetBool(streamAttr, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, required);
      (void)ge::AttrUtils::SetBool(streamAttr, ge::ATTR_NAME_ATTACHED_RESOURCE_FORCE_REUSE,
                                   true);  // true 表示强制主流复用

      HCCL_INFO(
          "[HcomOpsKernelBuilder][SetAttachedStreamInfoList] name[%s], reuse_key[%s], required[%d], nodeName[%s], "
          "groupId[%s].",
          name.c_str(), reuseKey.c_str(), required, node.GetName().c_str(), group.c_str());

      attachedStreamInfo.emplace_back(streamAttr);
    }
    ge::AttrUtils::SetListNamedAttrs(opDesc, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST, attachedStreamInfo);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetOpWorkerSpaceForKnowShape(ge::Node &node, u64 &opMemSize) {
  std::vector<int64_t> workspaceBytes;
  workspaceBytes.push_back(opMemSize);
  node.GetOpDesc()->SetWorkspaceBytes(workspaceBytes);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetCrackParamsInfo([[maybe_unused]] const ge::Node &node, u32 tensorNum, int64_t *tensorOffset,
                                                    int64_t *tensorSize, int64_t *crackOffset, int64_t *crackSize) {
  // 获取缝隙的offset和size
  for (u32 i = 0; i < tensorNum; i++) {
    // crackOffset基于LoadTask的inputaddr偏移，而不是基于基地址偏移
    crackOffset[i] = tensorOffset[i] + tensorSize[i] - tensorOffset[0];
    int64_t tensorSizeTemp = 0;
    tensorSizeTemp =
        (tensorSize[i] + TENSOR_ALIGNMENT_32 - 1) / TENSOR_ALIGNMENT_32 * TENSOR_ALIGNMENT_32 + TENSOR_ALIGNMENT_32;
    tensorSizeTemp = (tensorSizeTemp + TENSOR_ALIGNMENT_512 - 1) / TENSOR_ALIGNMENT_512 * TENSOR_ALIGNMENT_512;
    tensorSizeTemp = tensorSizeTemp - tensorSize[i];
    crackSize[i] = tensorSizeTemp;
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetTensorParamsInfo(const ge::Node &node, u32 tensorNum, int64_t *tensorOffset,
                                                     int64_t *tensorSize) {
  std::vector<int64_t> tensorOffsetTemp;
  std::vector<int64_t> tensorSizeTemp;

  auto op = node.GetOpDesc();
  // 获取tensor的偏移
  tensorOffsetTemp = op->GetInputOffset();
  for (size_t i = 0; i < tensorOffsetTemp.size(); i++) {
    HCCL_DEBUG("[HcomOpsKernelBuilder] node[%s] has %u inputs, input[%u] addr %lld.", op->GetName().c_str(),
               op->GetInputsSize(), i, tensorOffsetTemp[i]);
  }

  // 获取tensor的大小
  CHK_RET(HcomOpUtils::GetAllTensorSize(op, tensorNum, tensorSizeTemp));

  if (tensorOffsetTemp.size() == 0 || tensorSizeTemp.size() == 0) {
    HCCL_WARNING("[HcomOpsKernelBuilder] The value of tensorOffset or tensorSize is 0.");
    return HCCL_SUCCESS;
  }

  CHK_SAFETY_FUNC_RET(memcpy_s(tensorOffset, tensorOffsetTemp.size() * sizeof(int64_t), tensorOffsetTemp.data(),
                               tensorOffsetTemp.size() * sizeof(int64_t)));
  CHK_SAFETY_FUNC_RET(memcpy_s(tensorSize, tensorSizeTemp.size() * sizeof(int64_t), tensorSizeTemp.data(),
                               tensorSizeTemp.size() * sizeof(int64_t)));

  std::string name = op->GetName();
  HCCL_DEBUG("GetTensorParamsInfo name [%s].", name.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetPrivateDefWithTensorInfo(const ge::Node &node,
                                                             HCCL_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                                             domi::TaskDef &taskDef) {
  // 在set_private_def之前，获取tensorInfo
  int64_t tensorOffset[privateDefBuf.tensorNum];
  int64_t tensorSize[privateDefBuf.tensorNum];

  CHK_RET(GetTensorParamsInfo(node, privateDefBuf.tensorNum, tensorOffset, tensorSize));
  for (u32 i = 0; i < privateDefBuf.tensorNum; i++) {
    HCCL_DEBUG("[Builder][GetTensorParamsInfo] tensorOffset[%u] %lld tensorSize[%u] %lld.", i, tensorOffset[i], i,
               tensorSize[i]);
  }

  // 获取tensor间缝隙的offset和size
  int64_t crackOffset[privateDefBuf.tensorNum];
  int64_t crackSize[privateDefBuf.tensorNum];
  CHK_RET(GetCrackParamsInfo(node, privateDefBuf.tensorNum, tensorOffset, tensorSize, crackOffset, crackSize));

  for (u32 i = 0; i < privateDefBuf.tensorNum; i++) {
    HCCL_DEBUG("[Builder][SetPrivateDefWithTensorInfo] crackOffset[%u] %lld crackSize[%u] %lld.", i, crackOffset[i], i,
               crackSize[i]);
  }

  // 将获取的tensorInfo，拼接到privateDefBuf数据后
  void *privateDefPtr = nullptr;
  size_t privateDefBufSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(crackOffset) + sizeof(crackSize);
  privateDefPtr = malloc(privateDefBufSize);
  CHK_PTR_NULL(privateDefPtr);

  s32 ret = EOK;
  ret = memcpy_s(privateDefPtr, sizeof(privateDefBuf), &privateDefBuf, sizeof(privateDefBuf));
  if (UNLIKELY(ret != EOK)) {
    HCCL_ERROR("[Builder][SetPrivateDefWithTensorInfo][memcpy_s] copy privateDefBuf failed, ret -> %d", ret);
    free(privateDefPtr);
    return HCCL_E_INTERNAL;
  }
  ret = memcpy_s(static_cast<int64_t *>(static_cast<void *>(static_cast<s8 *>(privateDefPtr) + sizeof(privateDefBuf))),
                 sizeof(crackOffset), crackOffset, sizeof(crackOffset));
  if (UNLIKELY(ret != EOK)) {
    HCCL_ERROR("[Builder][SetPrivateDefWithTensorInfo][memcpy_s] copy crackOffset failed, ret -> %d", ret);
    free(privateDefPtr);
    return HCCL_E_INTERNAL;
  }
  ret = memcpy_s(static_cast<int64_t *>(static_cast<void *>(static_cast<s8 *>(privateDefPtr) + sizeof(privateDefBuf) +
                                                            sizeof(crackOffset))),
                 sizeof(crackSize), crackSize, sizeof(crackSize));
  if (UNLIKELY(ret != EOK)) {
    HCCL_ERROR("[Builder][SetPrivateDefWithTensorInfo][memcpy_s] copy crackSize failed, ret -> %d", ret);
    free(privateDefPtr);
    return HCCL_E_INTERNAL;
  }

  taskDef.set_private_def(privateDefPtr, privateDefBufSize);

  free(privateDefPtr);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GenerateTaskDef(const ge::Node &node, HCCL_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                                 domi::TaskDef &taskDef) {
  HCCL_DEBUG("[Generate][Task] privateDefBuf_graphId[%u]", privateDefBuf.graphId);
  taskDef.clear_kernel_hccl();
  domi::KernelHcclDef *kernelDefHccl = taskDef.mutable_kernel_hccl();
  CHK_PRT_RET((kernelDefHccl == nullptr),
              HCCL_ERROR("[Generate][Task]node[%s]: kernelDefHccl is null.", node.GetOpDesc()->GetName().c_str()),
              HCCL_E_PTR);

  taskDef.set_type(RT_MODEL_TASK_HCCL);
  taskDef.set_stream_id(node.GetOpDesc()->GetStreamId());

  kernelDefHccl->set_hccl_type(node.GetOpDesc()->GetType());
  kernelDefHccl->set_op_index(node.GetOpDesc()->GetId());
  std::string sCollectiveType = node.GetOpDesc()->GetType();
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetSupportedOP(std::vector<std::string> &hcclSupportOp) const {
  hcclSupportOp.assign(HCOM_SUPPORTED_OP_TYPE.begin(), HCOM_SUPPORTED_OP_TYPE.end());
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetRootGraphID(const ge::Node &node, uint32_t &graphId) {
  std::string nodeName = node.GetOpDesc()->GetName();
  auto ownerGraph = node.GetOwnerComputeGraph();
  CHK_PRT_RET((!ownerGraph), HCCL_ERROR("[Get][RootGraphID]node[%s] get owner graph failed", nodeName.c_str()),
              HCCL_E_PARA);

  auto rootGraph = ge::GraphUtils::FindRootGraph(ownerGraph);
  CHK_PRT_RET((!rootGraph), HCCL_ERROR("[Get][RootGraphID]node[%s] get root graph failed", nodeName.c_str()),
              HCCL_E_PARA);

  graphId = rootGraph->GetGraphID();
  return HCCL_SUCCESS;
}

// 返回HCCL的入参：comm and group
HcclResult HcomOpsKernelBuilder::GetCommFromOpDesc(const ge::OpDescPtr &op, int64_t &hcomComm, std::string &sGroup) {
  if (ge::AttrUtils::HasAttr(op, "comm")) {
    if (ge::AttrUtils::GetInt(op, "comm", hcomComm) == false) {
      HCCL_ERROR("[GetComm][OpDesc]errNo[0x%016llx]: get comm failed. get \"comm\" from opDesc failed",
                 HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    } else if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      HCCL_INFO("[HcomOpsKernelBuilder]get comm equal to 0, should get group.");
      CHK_RET(HcomOpUtils::GetGroupFromOpDesc(op, sGroup));
    } else {
      HCCL_INFO("[HcclCommGraph][Type]get comm name[%lld] success.", hcomComm);
    }
  } else {
    CHK_RET(HcomOpUtils::GetGroupFromOpDesc(op, sGroup));
    HCCL_INFO("%s get group[%s] success", __func__, sGroup.c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetOpMemAttr(ge::Node &node, const std::string &sCollectiveType,
                                              const u64 &opMemSize) {
  bool bRet = false;

  // ATTENTION: 算子在IR定义时input/output同名场合（参考HcomRemoteRefRead算子）会隐式设置reference属性为TRUE,
  //   此处只对IR定义中input/output不同名且需要复用内存的算子，进行内存复用配置。
  //   后续有类似算子实现建议在IR定义时将input/output配置为相同name。
  // broadcast算子因为输入/输出为同一内存Ref属性为true
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    bRet = ge::AttrUtils::SetBool(node.GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx] op[%s]: set  reference attr[%d] to"
                           "OpDesc failed.",
                           HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), true),
                HCCL_E_PARA);
    bRet = node.GetOpDesc()->UpdateOutputName(node.GetOpDesc()->GetAllInputName());
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx] op[%s]: update output name failed.",
                           HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str()),
                HCCL_E_PARA);
    HCCL_INFO("node[%s] set attr [reference]: %u", node.GetName().c_str(), true);

    // 算子属性为reference时，为减少GE的内存分配，设置 ouput 复用 input 内存
    for (uint32_t i = 0; i < static_cast<uint32_t>(node.GetOpDesc()->GetOutputsSize()); i++) {
      auto outDescPtr = node.GetOpDesc()->MutableOutputDesc(i);
      CHK_SMART_PTR_NULL(outDescPtr);
      ge::TensorUtils::SetReuseInput(*outDescPtr, true);
      ge::TensorUtils::SetReuseInputIndex(*outDescPtr, i);
    }
  } else {
    HCCL_INFO("node[%s] set attr [reference]: skip", node.GetName().c_str());
  }

  bRet = ge::AttrUtils::SetBool(node.GetOpDesc(), ge::ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  CHK_PRT_RET(!bRet,
              HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx] op[%s]: set is_fixed_addr_prior[%d] to"
                         "OpDesc failed.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), true),
              HCCL_E_PARA);
  HCCL_INFO("node[%s] set attr [is_fixed_addr_prior]: %d", node.GetName().c_str(), true);

  string groupListString;

  std::string sGroup;
  CHK_RET(HcomOpUtils::GetGroupFromOpDesc(node.GetOpDesc(), sGroup));

  std::string socVersion;
  if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("[offline][compilation] get soc version failed.");
    return HCCL_E_INTERNAL;
  }

  bool withoutImplCompile =
      IsOfflineCompilation() ||
      (ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, groupListString) == ge::GRAPH_SUCCESS);

  // 针对310p duo卡 2p场景申请内存为普通内存，不需要单独设置，其余场景需要设置申请为p2p内存
  // 板卡推理不需要设置申请内存为p2p
  u32 memType = 0;
  u32 p2pMemType = RT_MEMORY_P2P_DDR;
  CHK_RET(HcomGetMemType(sGroup.c_str(), socVersion.c_str(), false, &memType, nullptr, withoutImplCompile));
  if (memType == p2pMemType) {
    vector<int64_t> memTypeInput(node.GetOpDesc()->GetInputsSize(), p2pMemType);
    vector<int64_t> memTypeOutput(node.GetOpDesc()->GetOutputsSize(), p2pMemType);
    vector<int64_t> memTypeWorkSpace(1, p2pMemType);
    bool ret = ge::AttrUtils::SetListInt(node.GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, memTypeInput);
    CHK_PRT_RET(!ret,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx]: Set input mem addr failed. op[%s]", HCCL_E_PARA,
                           sCollectiveType.c_str()),
                HCCL_E_PARA);

    ret = ge::AttrUtils::SetListInt(node.GetOpDesc(), ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memTypeOutput);
    CHK_PRT_RET(!ret,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx]: Set output mem addr failed. op[%s]", HCCL_E_PARA,
                           sCollectiveType.c_str()),
                HCCL_E_PARA);

    if (opMemSize != 0) {
      ret = ge::AttrUtils::SetListInt(node.GetOpDesc(), ge::ATTR_NAME_WORKSPACE_TYPE_LIST, memTypeWorkSpace);
      CHK_PRT_RET(!ret,
                  HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx]: Set workspace mem addr failed. op[%s]", HCCL_E_PARA,
                             sCollectiveType.c_str()),
                  HCCL_E_PARA);
    }
    HCCL_INFO("[Set][OpMemAttr] Set memType p2p mem type");
  }

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetOpAtomicInputIndex(ge::Node &node, const std::string &sCollectiveType) {
  // allreduce，reduce 算子设定atomic Input Index属性
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE) {
    vector<int64_t> atomicInputIndex(1, -1);  // 回传vector的值为-1，作为标志位
    if (!ge::AttrUtils::SetListInt(node.GetOpDesc(), "atomic_input_index", atomicInputIndex)) {
      HCCL_ERROR("[Set][OpAtomicInputIndex]errNo[0x%016llx]: set op[%s] atomic index failed.",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType,
                                                    HcclDataType dataType, u64 &count) {
  HcclResult ret;
  u64 totalSize = 0;
  u32 dataTypeSize = 0;
  CHK_RET(SalGetDataTypeSize(dataType, dataTypeSize));
  CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[Get][CountFromOpDesc]dataType size is zero."), HCCL_E_PARA);

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    ret = GetHcomReceiveOpOutputSize(op, dataTypeSize, totalSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][Count]get op[%s] output size failed. ret[%d]", sCollectiveType.c_str(), ret), ret);
  } else {
    for (u64 i = 0; i < op->GetInputsSize(); i++) {
      u64 blockSize;
      CHK_SMART_PTR_NULL(op->GetInputDescPtr(i));
      if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
        // ReduceScatterV 算子的 count 为总数据的个数，count = input的size / dataTypeSize
        u64 shapeSize = 0;
        if (op->GetInputDescPtr(i)->GetShape().IsScalar()) {
          shapeSize = 1;
        } else {
          shapeSize = op->GetInputDescPtr(i)->GetShape().GetShapeSize();
        }
        CHK_PRT_RET((shapeSize > INVALID_U64 / dataTypeSize),
                    HCCL_ERROR("[Get][Count]op[%s] "
                               "shape size[%llu] is overflow.",
                               sCollectiveType.c_str(), shapeSize),
                    HCCL_E_PARA);
        const u32 paddingLen = 1024;  // 每个输入额外多申请 1024 bytes 的workspace memory。
        blockSize = (shapeSize * dataTypeSize + paddingLen);
      } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        // ReduceScatter 算子的 count 为输出数据的个数，count = (input的size / rank_size) / dataTypeSize
        s32 rankSize = 0;
        CHK_PRT_RET((!ge::AttrUtils::GetInt(op, HCOM_ATTR_RANK_SIZE, rankSize)),
                    HCCL_ERROR("[Get][Count]op[%s] get "
                               "attr[%s] failed.",
                               sCollectiveType.c_str(), HCOM_ATTR_RANK_SIZE.c_str()),
                    HCCL_E_PARA);

        CHK_PRT_RET((rankSize <= 0),
                    HCCL_ERROR("[Get][Count]errNo[0x%016llx] in ReduceScatter op,"
                               "rank_size[%d] should be greater than 0.",
                               HCOM_ERROR_CODE(HCCL_E_PARA), rankSize),
                    HCCL_E_PARA);

        u64 shapeSize = 0;
        if (op->GetInputDescPtr(i)->GetShape().IsScalar()) {
          shapeSize = 1;
        } else {
          shapeSize = (u64)op->GetInputDescPtr(i)->GetShape().GetShapeSize();
        }
        CHK_PRT_RET((shapeSize > INVALID_U64 / dataTypeSize),
                    HCCL_ERROR("[Get][Count]op[%s] shape size[%llu]"
                               "is overflow.",
                               sCollectiveType.c_str(), shapeSize),
                    HCCL_E_PARA);
        // reduce-scatter 融合场景：reduce-scatter算子的每个输入tensor均有补齐处理。
        // mindspore 补齐规则：(size + 32  -1 + 512) / 512 * 512
        // 因此，此处每个输入额外多申请 1024 bytes 的workspace memory。
        const u32 paddingLen = 1024;  // 每个输入额外多申请 1024 bytes 的workspace memory。
        blockSize = (shapeSize * dataTypeSize + paddingLen) / rankSize;
      } else {
        const u32 alignSize = 512;  // 以512 Byte 对齐
        int64_t inputSize = 0;
        CHK_PRT_RET((ge::GRAPH_SUCCESS != ge::TensorUtils::GetSize(*op->GetInputDescPtr(i), inputSize)),
                    HCCL_ERROR("[Get][Count]errNo[0x%016llx] get workspace bytes failed. get size from TensorDesc"
                               "failed, op : %s"
                               ", input index : %llu",
                               HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), i),
                    HCCL_E_PARA);
        if (inputSize == -1) {
          blockSize = 0;
        } else {
          CHK_PRT_RET((static_cast<u64>(inputSize) > INVALID_U64 - alignSize),
                      HCCL_ERROR("op[%s] input size[%llu] is "
                                 "overflow.",
                                 sCollectiveType.c_str(), static_cast<u64>(inputSize)),
                      HCCL_E_PARA);
          blockSize = (static_cast<u64>(inputSize) + alignSize - 1) / alignSize * alignSize;
        }
      }
      totalSize = totalSize + blockSize;
    }
  }
  count = totalSize / dataTypeSize;
  HCCL_INFO("op[%s] get count[%llu] success.", sCollectiveType.c_str(), count);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetHcomReceiveOpOutputSize(const ge::OpDescPtr &op, u32 dataTypeSize,
                                                            u64 &outputSize) {
  CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[Get][ReceiveOpOutputSize]dataType size is zero."), HCCL_E_PARA);

  std::string sCollectiveType = op->GetType();
  CHK_PRT_RET(
      (!ge::AttrUtils::HasAttr(op, HCOM_ATTR_SHAPE)),
      HCCL_ERROR("[Get][ReceiveOpOutputSize]op[%s] has no attr[%s].", sCollectiveType.c_str(), HCOM_ATTR_SHAPE.c_str()),
      HCCL_E_PARA);

  vector<int64_t> shapeDims;
  CHK_PRT_RET((!ge::AttrUtils::GetListInt(op, HCOM_ATTR_SHAPE, shapeDims)),
              HCCL_ERROR("[Get][ReceiveOpOutputSize]op[%s] get attr[%s] failed.", sCollectiveType.c_str(),
                         HCOM_ATTR_SHAPE.c_str()),
              HCCL_E_PARA);

  u64 shapeSize = 0;
  if (shapeDims.empty()) {
    // HcomReceive算子标量的话将shapeSize设置为1
    shapeSize = 1;
  } else {
    shapeSize = static_cast<u64>(ge::Shape(shapeDims).GetShapeSize());
  }
  const u32 alignSize = 512;  // 以512 Byte 对齐
  CHK_PRT_RET(
      (shapeSize > (INVALID_U64 - alignSize) / dataTypeSize),
      HCCL_ERROR("[Get][ReceiveOpOutputSize]op[%s] shape size[%llu] is overflow.", sCollectiveType.c_str(), shapeSize),
      HCCL_E_PARA);
  outputSize = (static_cast<u64>(shapeSize * dataTypeSize) + alignSize - 1) / alignSize * alignSize;
  return HCCL_SUCCESS;
}

// 返回HCCL的入参:srTag
HcclResult HcomOpsKernelBuilder::GetSrTagFromDesc(const ge::OpDescPtr &op, u32 &srTag,
                                                  const std::string &sCollectiveType) {
  srTag = 0;
  if ((sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) || (sCollectiveType == HCCL_KERNEL_OP_TYPE_SEND)) {
    if (ge::AttrUtils::HasAttr(op, "sr_tag")) {
      if (ge::AttrUtils::GetInt(op, "sr_tag", srTag) == false) {
        HCCL_ERROR("[Get][SrTag]errNo[0x%016llx] op[%s]: get srTag failed. get \"sr_tag\" from opDesc failed",
                   HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
        return HCCL_E_PARA;
      }
    } else {
      HCCL_ERROR("[Get][SrTag]errNo[0x%016llx] op[%s]: get srTag failed. no \"sr_tag\" in opDesc",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }
  HCCL_INFO("get srTag[%u] success.", srTag);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::GetOriginalGraphShapeTypeFromDesc(const ge::OpDescPtr &op, u32 &shapeType) {
  if (ge::AttrUtils::HasAttr(op, ORIGINAL_GRAPH_SHAPE_TYPE)) {
    if (ge::AttrUtils::GetInt(op, ORIGINAL_GRAPH_SHAPE_TYPE, shapeType) == false) {
      HCCL_ERROR(
          "[Get][OriginalGraphShapeType]errNo[0x%016llx]: get shapeType failed. get \"shapeType\" from"
          "opDesc failed",
          HCOM_ERROR_CODE(HCCL_E_PARA));
      return HCCL_E_PARA;
    }
  } else {
    shapeType = (u32)ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;
  }
  HCCL_INFO("get shapeType [%u] success.", shapeType);
  return HCCL_SUCCESS;
}

// 返回HCCL的入参：destRank
HcclResult HcomOpsKernelBuilder::GetDestRankFromDesc(const ge::OpDescPtr &op, u32 &destRank,
                                                     const std::string &sCollectiveType) {
  destRank = 0;
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_SEND) {
    if (ge::AttrUtils::HasAttr(op, "dest_rank")) {
      if (ge::AttrUtils::GetInt(op, "dest_rank", destRank) == false) {
        HCCL_ERROR(
            "[Get][DestRank]errNo[0x%016llx] op[%s]: get dest rank failed. get \"dest_rank\" from"
            "opDesc failed",
            HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
        return HCCL_E_PARA;
      }
    } else {
      HCCL_ERROR("[Get][DestRank]errNo[0x%016llx] op[%s]: get dest rank failed. no \"dest_rank\" in opDesc",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }
  HCCL_INFO("get dest rank[%u] success.", destRank);
  return HCCL_SUCCESS;
}

// 返回HCCL的入参：srcRank
HcclResult HcomOpsKernelBuilder::GetSrcRankFromDesc(const ge::OpDescPtr &op, u32 &srcRank,
                                                    const std::string &sCollectiveType) {
  srcRank = 0;
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
    if (ge::AttrUtils::HasAttr(op, "src_rank")) {
      if (ge::AttrUtils::GetInt(op, "src_rank", srcRank) == false) {
        HCCL_ERROR("[Get][SrcRank]errNo[0x%016llx] op[%s]: get src rank failed. no \"src_rank\" in opDesc",
                   HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
        return HCCL_E_PARA;
      }
    } else {
      HCCL_ERROR("[Get][SrcRank]errNo[0x%016llx] op[%s]: get src rank failed. no \"src_rank\" in opDesc",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }
  HCCL_INFO("get src rank[%u] success.", srcRank);
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAlltoAllVParams(const ge::Node &node,
                                                    HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf,
                                                    const std::string &sCollectiveType) {
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV) {
    auto op = node.GetOpDesc();
    CHK_RET(SetAlltoAllVDataTypeToDef(op, privateDefBuf));
    CHK_RET(CopyAlltoAllVParamsToDef(node, privateDefBuf));
  } else if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
    auto op = node.GetOpDesc();
    CHK_RET(SetAlltoAllVCDataTypeToDef(op, privateDefBuf));
    CHK_RET(CopyAlltoAllVCParamsToDef(node, privateDefBuf));
  }
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetReduceScatterVParams(const ge::Node &node,
                                                         HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  CHK_RET(CopyReduceScatterVParamsToDef(node, privateDefBuf));
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::CopyReduceScatterVParamsToDef(
    const ge::Node &node, HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  std::vector<int64_t> sendCounts;
  std::vector<int64_t> sendDispls;
  std::vector<int64_t> recvCount;
  CHK_RET(HcomOpUtils::GetReduceScatterVCountsDispl(const_cast<ge::Node &>(node), sendCounts, sendDispls, recvCount));

  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.sendCounts, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               sendCounts.data(), sendCounts.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.sendDispls, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               sendDispls.data(), sendDispls.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.recvCounts, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               recvCount.data(), recvCount.size() * sizeof(u64)));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAllGatherVParams(const ge::Node &node,
                                                     HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  CHK_RET(CopyAllGatherVParamsToDef(node, privateDefBuf));
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::CopyAlltoAllVParamsToDef(const ge::Node &node,
                                                          HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  std::vector<int64_t> sendCounts;
  std::vector<int64_t> sendDispls;
  std::vector<int64_t> recvCounts;
  std::vector<int64_t> recvDispls;

  auto op = node.GetOpDesc();
  if (ge::AttrUtils::HasAttr(op, "send_counts")) {
    CHK_RET(HcomOpUtils::GetAlltoAllCountsDispl(op, sendCounts, sendDispls, recvCounts, recvDispls));
  } else {
    CHK_RET(HcomOpUtils::GetAlltoAllCountsDispl(const_cast<ge::Node &>(node), sendCounts, sendDispls, recvCounts,
                                                recvDispls));
  }

  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.sendCounts, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               sendCounts.data(), sendCounts.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.sendDispls, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               sendDispls.data(), sendDispls.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.recvCounts, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               recvCounts.data(), recvCounts.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.recvDispls, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               recvDispls.data(), recvDispls.size() * sizeof(u64)));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::CopyAllGatherVParamsToDef(const ge::Node &node,
                                                           HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  std::vector<int64_t> sendCount;
  std::vector<int64_t> recvCounts;
  std::vector<int64_t> recvDispls;
  CHK_RET(HcomOpUtils::GetAllGatherVCountsDispl(const_cast<ge::Node &>(node), sendCount, recvCounts, recvDispls));

  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.sendCount, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               sendCount.data(), sendCount.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.recvCounts, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               recvCounts.data(), recvCounts.size() * sizeof(u64)));
  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.paramsInfo.recvDispls, ALLTOALLV_RANK_MAX_NUM * sizeof(u64),
                               recvDispls.data(), recvDispls.size() * sizeof(u64)));

  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::CopyAlltoAllVCParamsToDef(const ge::Node &node,
                                                           HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  std::vector<int64_t> sendCountMatrix;
  auto op = node.GetOpDesc();
  if (ge::AttrUtils::HasAttr(op, "send_count_matrix")) {
    CHK_RET(HcomOpUtils::GetAlltoAllCountMatrix(op, sendCountMatrix));
  } else {
    CHK_RET(HcomOpUtils::GetAlltoAllCountMatrix(const_cast<ge::Node &>(node), sendCountMatrix));
  }

  CHK_SAFETY_FUNC_RET(memcpy_s(privateDefBuf.cparamsInfo.sendCountMatrix,
                               ALLTOALLVC_RANK_MAX_NUM * ALLTOALLVC_RANK_MAX_NUM * sizeof(u64), sendCountMatrix.data(),
                               sendCountMatrix.size() * sizeof(u64)));
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAlltoAllVDataTypeToDef(const ge::OpDescPtr &op,
                                                           HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  CHK_RET(HcomOpUtils::GetAlltoAllDataType(op, privateDefBuf.paramsInfo.sendType, privateDefBuf.paramsInfo.recvType));
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetAlltoAllVCDataTypeToDef(const ge::OpDescPtr &op,
                                                            HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF &privateDefBuf) {
  CHK_RET(HcomOpUtils::GetAlltoAllDataType(op, privateDefBuf.cparamsInfo.sendType, privateDefBuf.cparamsInfo.recvType));
  return HCCL_SUCCESS;
}

HcclResult HcomOpsKernelBuilder::SetHcomOpParam(const ge::Node &node, HcomOpParam *hcomOpParam,
                                                std::string &sCollectiveType, std::string &sGroup,
                                                std::string &socVersion, std::vector<int64_t> &sendCountMatrix,
                                                std::vector<int64_t> &sendCounts, std::vector<int64_t> &sendDispls,
                                                std::vector<int64_t> &recvCounts, std::vector<int64_t> &recvDispls,
                                                std::vector<u32> &curRanks, std::string &rankTableStr,
                                                std::string &rankTableM) {
  HcclResult ret;
  sCollectiveType = node.GetOpDesc()->GetType();
  ret = CheckSupportedOP(sCollectiveType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op type[%s] is not supported.", ret, sCollectiveType.c_str()),
      HCCL_E_NOT_SUPPORT);
  hcomOpParam->opType = const_cast<char *>(sCollectiveType.c_str());

  HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
  ret = HcomOpUtils::ConversionOpDataType(node.GetOpDesc(), sCollectiveType, dataType);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: get data type failed. ret[%d]", sCollectiveType.c_str(), ret), ret);
  hcomOpParam->dataType = dataType;

  CHK_PRT_RET(ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS,
              HCCL_ERROR("[offline][compilation] get soc version failed."), HCCL_E_INTERNAL);
  hcomOpParam->socVersion = const_cast<char *>(socVersion.c_str());

  int64_t hcomComm = 0;
  ret = GetCommFromOpDesc(node.GetOpDesc(), hcomComm, sGroup);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: GetGroupFromOpDesc failed. ret[%d]", sCollectiveType.c_str(), ret),
      ret);
  if (hcomComm != static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &(hcomOpParam->group)));
  } else {
    hcomOpParam->group = const_cast<char *>(sGroup.c_str());
  }

  u32 rankSize = 0;
  if (!IsOfflineCompilation()) {
    if (hcomComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      CHK_RET(HcomGetRankSize(sGroup.c_str(), &rankSize));
    } else {
      char *group = nullptr;
      CHK_RET(GetGroupNameByOpBaseHcom(hcomComm, &group));
      CHK_RET(HcomGetRankSize(group, &rankSize));
    }
  } else {
    // 离线编译ranksize在HcomCalcOpResOffline中计算
  }
  if ((sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) || (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHER)) {
    CHK_PRT_RET((!ge::AttrUtils::GetInt(node.GetOpDesc(), HCOM_ATTR_RANK_SIZE, rankSize)),
                HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s] get  attr[%s] failed.", sCollectiveType.c_str(),
                           HCOM_ATTR_RANK_SIZE.c_str()),
                HCCL_E_PARA);
    CHK_PRT_RET((rankSize <= 0),
                HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: rank_size[%d] should be "
                           "greater than 0.",
                           sCollectiveType.c_str(), rankSize),
                HCCL_E_PARA);
  }
  hcomOpParam->rankSize = rankSize;

  // 获取aivCoreLimit，提供给HCCL，用于和Optype，count等参数一起选择具体的算法及判断是否是AIV模式
  uint32_t aivCoreLimit;
  CHK_RET(HcomOpUtils::GetAivCoreLimit(node.GetOpDesc(), sCollectiveType, aivCoreLimit));
  hcomOpParam->aivCoreLimit = aivCoreLimit;

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER ||
      sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
    HcclReduceOp reduction = HcclReduceOp::HCCL_REDUCE_SUM;
    CHK_RET(HcomOpUtils::GetReduction(node.GetOpDesc(), reduction));
    hcomOpParam->reduceOp = reduction;

    u8 deterministic = DETERMINISTIC_DISABLE;
    CHK_RET(GetDeterministic(deterministic));
    hcomOpParam->geDeterministic = deterministic;
  }

  u64 count = 0;
  ret = HcomOpUtils::GetAccuracyCountFromOpDesc(node.GetOpDesc(), sCollectiveType, dataType, count, rankSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][OpWorkspaceMemSize]op[%s]: get count failed. ret[%d]", sCollectiveType.c_str(), ret),
              ret);
  hcomOpParam->count = count;

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
    // reducescatterv复用HcomOpParam的All2AllDataDes字段
    CHK_RET(
        HcomOpUtils::GetReduceScatterVCountsDispl(const_cast<ge::Node &>(node), sendCounts, sendDispls, recvCounts));
    hcomOpParam->All2AllDataDes.sendCounts = static_cast<void *>(sendCounts.data());
    hcomOpParam->All2AllDataDes.sendDispls = static_cast<void *>(sendDispls.data());
    hcomOpParam->All2AllDataDes.recvCounts = static_cast<void *>(recvCounts.data());
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLGATHERV) {
    // allgatherv复用HcomOpParam的All2AllDataDes字段
    CHK_RET(HcomOpUtils::GetAllGatherVCountsDispl(const_cast<ge::Node &>(node), sendCounts, recvCounts, recvDispls));
    hcomOpParam->All2AllDataDes.sendCounts = static_cast<void *>(sendCounts.data());
    hcomOpParam->All2AllDataDes.recvCounts = static_cast<void *>(recvCounts.data());
    hcomOpParam->All2AllDataDes.recvDispls = static_cast<void *>(recvDispls.data());
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLV) {
    HcclDataType sendType;
    HcclDataType recvType;
    CHK_RET(HcomOpUtils::GetAlltoAllDataType(node.GetOpDesc(), sendType, recvType));

    auto op = node.GetOpDesc();
    if (ge::AttrUtils::HasAttr(op, "send_counts")) {
      CHK_RET(HcomOpUtils::GetAlltoAllCountsDispl(op, sendCounts, sendDispls, recvCounts, recvDispls));
    } else {
      CHK_RET(HcomOpUtils::GetAlltoAllCountsDispl(const_cast<ge::Node &>(node), sendCounts, sendDispls, recvCounts,
                                                  recvDispls));
    }

    if (sendCounts.size() < rankSize) {
      HCCL_ERROR("[sendCounts] size[%u] is invalid, expect size: %llu", sendCounts.size(), rankSize);
      return HCCL_E_PARA;
    }

    hcomOpParam->All2AllDataDes.sendType = sendType;
    hcomOpParam->All2AllDataDes.recvType = recvType;
    hcomOpParam->All2AllDataDes.sendCounts = static_cast<void *>(sendCounts.data());
    hcomOpParam->All2AllDataDes.sendDispls = static_cast<void *>(sendDispls.data());
    hcomOpParam->All2AllDataDes.recvCounts = static_cast<void *>(recvCounts.data());
    hcomOpParam->All2AllDataDes.recvDispls = static_cast<void *>(recvDispls.data());
  }

  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
    HcclDataType sendType;
    HcclDataType recvType;
    CHK_RET(HcomOpUtils::GetAlltoAllDataType(node.GetOpDesc(), sendType, recvType));

    if (!IsOfflineCompilation()) {
      CHK_RET(HcomOpUtils::CheckAlltoAllvcRank(node, hcomComm, sGroup));
    }

    auto op = node.GetOpDesc();
    if (ge::AttrUtils::HasAttr(op, "send_count_matrix")) {
      CHK_RET(HcomOpUtils::GetAlltoAllCountMatrix(op, sendCountMatrix));
    } else {
      CHK_RET(HcomOpUtils::GetAlltoAllCountMatrix(const_cast<ge::Node &>(node), sendCountMatrix));
    }
    if (sendCountMatrix.size() < rankSize * rankSize) {
      HCCL_ERROR("[sendCountMatrix] size[%u] is invalid, expect size: %llu", sendCountMatrix.size(),
                 rankSize * rankSize);
      return HCCL_E_PARA;
    }

    hcomOpParam->All2AllDataDes.sendType = sendType;
    hcomOpParam->All2AllDataDes.recvType = recvType;
    hcomOpParam->All2AllDataDes.sendCountMatrix = static_cast<void *>(sendCountMatrix.data());
  }

  std::string groupListString;
  if (ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_HCOM_GROUPLIST, groupListString) == ge::GRAPH_SUCCESS) {
    try {
      nlohmann::json groupListConf;
      CHK_RET(SalParseInformation(groupListConf, groupListString));
      std::vector<nlohmann::json> groupList = groupListConf.get<std::vector<nlohmann::json>>();
      for (auto &groupInfo : groupList) {
        HCCL_DEBUG("groupInfo:%s", groupInfo.dump().c_str());
        std::string curGroupName = groupInfo["group_name"];
        HCCL_DEBUG("curGroupName:%s", curGroupName.c_str());
        if (curGroupName == sGroup) {
          curRanks = groupInfo["group_rank_list"].get<std::vector<u32>>();
          break;
        }
      }
    } catch (const std::exception &e) {
      HCCL_ERROR("[HcomCalcOpRunningParam] exception caught. err[%s]", e.what());
      return HCCL_E_INTERNAL;
    }
    hcomOpParam->groupList = static_cast<u32 *>(curRanks.data());
    hcomOpParam->groupListSize = curRanks.size();
  } else {
    HCCL_INFO("get groupListString failed");
  }

  if ((ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_TABLE, rankTableStr) == ge::GRAPH_SUCCESS) &&
      !rankTableStr.empty()) {
    hcomOpParam->rankTable = const_cast<char *>(rankTableStr.c_str());
  } else {
    HCCL_INFO("get rankTableStr failed");
  }

  std::string rankTablePath;
  if ((ge::GetThreadLocalContext().GetOption(ge::OPTION_EXEC_RANK_TABLE_FILE, rankTablePath) == ge::GRAPH_SUCCESS) &&
      !rankTablePath.empty()) {
    ret = HcomLoadRanktableFile(rankTablePath, rankTableM);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[GetRanktable] rankTablePath[%s]"
                           "load rankTable error.",
                           rankTablePath.c_str()),
                HCCL_E_INTERNAL);
    hcomOpParam->rankTable = const_cast<char *>(rankTableM.c_str());
  } else {
    HCCL_INFO("get rankTablePath failed");
  }

  HCCL_INFO(
      "[SetHcomOpParam]HcomOpParam opType:[%s] dataType:[%s] count:[%llu] group:[%s] reduceOp:[%s] "
      "deterministic:[%d] socVersion:[%s] groupList:[%p] groupListSize:[%llu] ranktable:[%p]",
      hcomOpParam->opType, GetDataTypeEnumStr(hcomOpParam->dataType).c_str(), hcomOpParam->count, hcomOpParam->group,
      GetReduceOpEnumStr(hcomOpParam->reduceOp).c_str(), hcomOpParam->geDeterministic, hcomOpParam->socVersion,
      hcomOpParam->groupList, hcomOpParam->groupListSize, hcomOpParam->rankTable);
  return HCCL_SUCCESS;
}
}  // namespace hccl