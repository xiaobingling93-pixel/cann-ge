/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_tuning_hcom_graph_optimizer.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include "mmpa/mmpa_api.h"
#include "auto_tuning_hcom_all_reduce_fusion.h"
#include "auto_tuning_hcom_ops_kernel_info_store.h"
#include "auto_tuning_hcom_ops_kernel_builder.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/ge_local_context.h"
#include "hcom_op_utils.h"

using namespace std;

static uint32_t g_gradientNodeNo;

namespace hccl {
AutoTuningHcomGraphOptimizer::AutoTuningHcomGraphOptimizer() : isGradientAutoTune_(true) {}

AutoTuningHcomGraphOptimizer::~AutoTuningHcomGraphOptimizer() {}

ge::Status AutoTuningHcomGraphOptimizer::Initialize(const std::map<std::string, std::string> &options,
                                                    ge::OptimizeUtility *const optimizeUtility) {
  auto iterTuneType = options.find("ge.jobType");
  if (iterTuneType == options.end()) {
    HCCL_ERROR("[Initialize][GraphOptimizer]AutoTuning jobType not set");
    return ge::INTERNAL_ERROR;
  }
  // 非gdat场景
  if (iterTuneType->second != "4") {
    isGradientAutoTune_ = false;
    HCCL_INFO("Other autoTuning mode");
    HcclResult ret = HcomGraphOptimizeInitialize(options, optimizeUtility);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Initialize][GraphOptimizer] auto tuning init failed"),
                ge::INTERNAL_ERROR);
    return ge::SUCCESS;
  }

  bool profilingMode = false;
  std::string profilingOption;
  HcclResult ret = ParseProfilingConfig(profilingMode, profilingOption);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Initialize][GraphOptimizer]Parse profiling config failed."),
              ge::INTERNAL_ERROR);

  if (profilingMode && !profilingOption.empty()) {
    HCCL_ERROR(
        "[Initialize][GraphOptimizer]The profiling switch is from the environment variable."
        "Do not enable profiling in GDAT.");
    return ge::INTERNAL_ERROR;
  }

  auto iterProfilingMode = options.find(ge::OPTION_EXEC_PROFILING_MODE);
  auto iterProfilingOptions = options.find(ge::OPTION_EXEC_PROFILING_OPTIONS);
  // gdat场景，不支持开启profiling，profilingMode为1且profilingOption不为空表示开启；为0表示未开
  if (iterProfilingMode != options.end() && iterProfilingOptions != options.end()) {
    if (iterProfilingMode->second == "1" && !iterProfilingOptions->second.empty()) {
      HCCL_ERROR(
          "[Initialize][GraphOptimizer]The profiling switch is from the ge options. \
            Do not enable profiling in GDAT.");
      return ge::INTERNAL_ERROR;
    }
  }

  g_gradientNodeNo = 0;
  auto iter = options.find(ge::TUNING_PATH);
  if (iter == options.end()) {
    HCCL_ERROR("[Initialize][GraphOptimizer]auto tuning mode has no work path. exit.");
    return ge::INTERNAL_ERROR;
  }

  std::string workPath = iter->second;
  char realFile[PATH_MAX] = {0};
  if (realpath(workPath.c_str(), realFile) == nullptr) {
    HCCL_ERROR("[Initialize][GraphOptimizer]path %s is not a valid real path", workPath.c_str());
    return ge::INTERNAL_ERROR;
  }
  workPath_ = std::string(realFile);
  CHK_PRT_RET(workPath_.length() < 1,
              HCCL_ERROR("[Initialize][GraphOptimizer]workPath length is"
                         "incorrect: workPath length is %zu",
                         workPath_.length()),
              ge::INTERNAL_ERROR);
  if (workPath_.c_str()[workPath_.length() - 1] != '/') {
    workPath_.append("/");
  }
  HCCL_INFO("auto tuning gradient split: work path is %s", workPath_.c_str());
  std::string gradientInfosFile = workPath_ + "gradient_summary.csv";
  const int FILE_AUTHORITY = 0600;
  int fd = open(gradientInfosFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, FILE_AUTHORITY);
  CHK_PRT_RET(fd < 0, HCCL_ERROR("[Initialize][GraphOptimizer]Fail to open the file: %s.", gradientInfosFile.c_str()),
              ge::INTERNAL_ERROR);
  CHK_PRT_RET(close(fd) != 0,
              HCCL_ERROR("[Initialize][GraphOptimizer]Fail to close the file: %s.", gradientInfosFile.c_str()),
              ge::INTERNAL_ERROR);
  std::ofstream fileStream(gradientInfosFile.c_str(), std::ios::out | std::ios::binary);
  if (fileStream.is_open()) {
    fileStream << "No" << "," << "gradient_size(byte)" << "," << "data_type" << "," << "graph_id" << "," << "group_name"
               << "," << "gradient_node_name" << "," << "trace_node_name" << "," << "allreduce_node_name" << std::endl;
    fileStream.close();
    HCCL_INFO("gradient info file name: %s", gradientInfosFile.c_str());
  } else {
    HCCL_ERROR("[Initialize][GraphOptimizer]file %s open failed!", gradientInfosFile.c_str());
    return ge::INTERNAL_ERROR;
  }
  return ge::SUCCESS;
}

HcclResult AutoTuningHcomGraphOptimizer::ParseProfilingConfig(bool profilingMode, std::string &profilingOption) {
  char *mmSysGetEnvValue = nullptr;
  MM_SYS_GET_ENV(MM_ENV_PROFILING_MODE, mmSysGetEnvValue);
  std::string profilingEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
  CHK_PRT_RET(profilingEnv == "EmptyString",
              HCCL_RUN_INFO("environmental variable PROFILING_MODE and GE profiling option is not set, default: false"),
              HCCL_SUCCESS);
  CHK_PRT_RET(profilingEnv.compare("true") != 0, HCCL_INFO("environmental variable PROFILING_MODE = false"),
              HCCL_SUCCESS);
  profilingMode = true;
  HCCL_RUN_INFO("PROFILING_MODE[%s] is set", profilingEnv.c_str());

  mmSysGetEnvValue = nullptr;
  MM_SYS_GET_ENV(MM_ENV_PROFILING_OPTIONS, mmSysGetEnvValue);
  profilingEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
  CHK_PRT_RET(profilingEnv == "EmptyString", HCCL_RUN_INFO("environmental variable PROFILING_OPTIONS is not set."),
              HCCL_SUCCESS);
  profilingOption = profilingEnv;
  HCCL_RUN_INFO("Set Env [PROFILING_MODE]: Value[%s]", profilingOption.c_str());
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomGraphOptimizer::CheckSupportedOP(const std::string &sCollectiveType) const {
  std::vector<std::string>::const_iterator it =
      std::find(AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.begin(), AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.end(), sCollectiveType);
  return (it != AUTO_TUNING_HCOM_SUPPORTED_OP_TYPE.end()) ? HCCL_SUCCESS : HCCL_E_PARA;
}

HcclResult AutoTuningHcomGraphOptimizer::CalcHCCLOutputMemSize(const std::string &sCollectiveType, int64_t &memSize) {
  HCCL_DEBUG("[HcomGraphOptimizer][CalcHCCLOutputMemSize] sCollectiveType[%s] memSize[%lld]", sCollectiveType.c_str(),
             memSize);
  const u32 MEMORY_ALIGN_RATIO = 2;  // GE要求内存需要32KB对齐后，再外加32KB. out = (in + 2 * 32 - 1) / 32 * 32
  const u32 MEMORY_ALIGN_SIZE = 32;  // GE要求内存需要32KB对齐后，再外加32KB. out = (in + 2 * 32 - 1) / 32 * 32
  // GE要求内存需要32KB对齐后，再外加32KB
  memSize = ((memSize + MEMORY_ALIGN_RATIO * MEMORY_ALIGN_SIZE - 1) / MEMORY_ALIGN_SIZE) * MEMORY_ALIGN_SIZE;
  return HCCL_SUCCESS;
}

ge::Status AutoTuningHcomGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph) {
  HcclResult ret;
  bool uknownShapeGraph = false;
  ret = OriginalGraphShapeTypeCfg(graph, uknownShapeGraph);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Optimize][Graph]graph[%s]: OriginalGraphShapeTypeCfg failed. "
                         "ret[%d]",
                         graph.GetName().c_str(), ret),
              ge::INTERNAL_ERROR);
  ret = SetUnknownShapeAttr(graph, uknownShapeGraph);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Optimize][Graph]graph[%s]: SetUnknownShapeAttr failed. ret[%d]", graph.GetName().c_str(), ret),
      ge::INTERNAL_ERROR);

  /* 其他调优模式返回success */
  if (!isGradientAutoTune_) {
    ret = HcomOptimizeOriginalGraph(graph, uknownShapeGraph);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Optimize][Graph]graph[%s]: autotuning Original Optimize failed. ret[%d]",
                           graph.GetName().c_str(), ret),
                ge::INTERNAL_ERROR);
    return ge::SUCCESS;
  }

  AutoTuningHcomAllReduceFusion fusionHcomAllReduceOp;
  HCCL_INFO("start fusion HcomAllReduce Op.");
  std::vector<GradientDataInfo> recordInfos;
  ret = fusionHcomAllReduceOp.Run(graph, recordInfos);

  std::vector<std::shared_ptr<ge::ComputeGraph>> subgraph;
  subgraph = graph.GetAllSubgraphs();
  for (u32 index = 0; index < subgraph.size(); index++) {
    AutoTuningHcomAllReduceFusion fusionSubGraphOp;
    ret = fusionSubGraphOp.Run(*subgraph[index], recordInfos);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Optimize][Graph]fuse HcomAllReduce op failed in subgraph[%s]. ret[%d]",
                           (*subgraph[index]).GetName().c_str(), ret),
                ge::INTERNAL_ERROR);
  }
  HCCL_INFO("graph[%s] with [%d]subgraphs: end fusion HcomAllReduce node.", graph.GetName().c_str(), subgraph.size());

  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Optimize][Graph]graph[%s]: fuse HcomAllReduce op failed. ret[%d]", graph.GetName().c_str(), ret),
      ge::INTERNAL_ERROR);
  if (recordInfos.size() != 0) {
    std::string gradientInfosFile = workPath_ + "gradient_summary.csv";
    std::ofstream fileStream(gradientInfosFile.c_str(), std::ios::out | std::ios::app | std::ios::binary);
    if (fileStream.is_open()) {
      for (uint32_t i = 0; i < recordInfos.size(); i++) {
        g_gradientNodeNo++;
        fileStream << g_gradientNodeNo << "," << recordInfos[i].dataSize << "," << recordInfos[i].dataType << ","
                   << recordInfos[i].graphId << "," << recordInfos[i].groupName << ","
                   << recordInfos[i].gradientNodeName << "," << recordInfos[i].traceNodeName << ","
                   << recordInfos[i].allReduceNodeName << std::endl;
      }
      fileStream.close();
      HCCL_INFO("gradient summary info has stored in %s, length:%zu", gradientInfosFile.c_str(), recordInfos.size());
    } else {
      HCCL_ERROR("[Optimize][Graph]file %s open failed!", gradientInfosFile.c_str());
    }
  }
  return ge::SUCCESS;
}

ge::Status AutoTuningHcomGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph &graph) {
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    if (CheckSupportedOP(opDescPtr->GetType()) != HCCL_SUCCESS) {
      continue;
    }
    HcclResult ret = CalcOpRunningParam(*nodePtr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Optimize][FusedGraph]errNo[0x%016llx] Calc Op Running Params failed.", HCOM_ERROR_CODE(ret)),
        ge::INTERNAL_ERROR);
  }
  return ge::SUCCESS;
}

HcclResult AutoTuningHcomGraphOptimizer::CalcOpRunningParam(ge::Node &node) {
  HCCL_INFO("calculate hccl running parameters start.");
  CHK_PRT_RET(
      !node.GetOpDesc(),
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetOpDesc failed. null ptr.", HCOM_ERROR_CODE(HCCL_E_PTR)),
      HCCL_E_PTR);

  bool unknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(node, unknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Calc][OpRunningParam]node[%s] get node unknown status failed", node.GetName().c_str()),
              HCCL_E_PARA);
  if (unknownShapeNode) {
    HCCL_INFO("op:%s is unknown shape, does not need to Calc Op Running Param", node.GetName().c_str());
    return HCCL_SUCCESS;
  }

  // 获取需回传的信息
  u64 streamNum = 0;
  u64 opMemSize = 0;
  CHK_PRT_RET(
      !node.GetOpDesc(),
      HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] GetOpDesc failed. null ptr.", HCOM_ERROR_CODE(HCCL_E_PTR)),
      HCCL_E_PTR);
  std::string sCollectiveType = node.GetOpDesc()->GetType();
  HcclResult ret = CheckSupportedOP(sCollectiveType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op type[%s] is not supported.", HCOM_ERROR_CODE(ret),
                         sCollectiveType.c_str()),
              ret);
  // 获取并设定stream 数量
  if (ge::AttrUtils::SetInt(node.GetOpDesc(), "used_stream_num", streamNum) == false) {
    HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] op[%s]: set stream number[%llu] to OpDesc failed.",
               HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str(), streamNum);
    return HCCL_E_INTERNAL;
  }

  std::vector<int64_t> workspaceBytes;
  workspaceBytes.push_back(opMemSize);
  node.GetOpDesc()->SetWorkspaceBytes(workspaceBytes);

  // 设置内存属性
  ret = SetOpMemAttr(node, node.GetOpDesc()->GetType(), opMemSize);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Calc][OpRunningParam]errNo[0x%016llx] set node[%s] mem attr failed.", HCOM_ERROR_CODE(ret),
                         node.GetName().c_str()),
              ret);

  // 设置output size 大小
  CHK_RET(SetOpOutputMemSize(node, sCollectiveType));

  // allreduce，reduce 算子设定atomic Input Index属性
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || sCollectiveType == HCCL_KERNEL_OP_TYPE_REDUCE) {
    vector<int64_t> atomicInputIndex(1, -1);  // 回传vector的值为-1，作为标志位
    if (!ge::AttrUtils::SetListInt(node.GetOpDesc(), "atomic_input_index", atomicInputIndex)) {
      HCCL_ERROR("[Set][OpAtomicInputIndex]errNo[0x%016llx]: set op[%s] atomic index failed.",
                 HCOM_ERROR_CODE(HCCL_E_PARA), sCollectiveType.c_str());
      return HCCL_E_PARA;
    }
  }

  HCCL_INFO("calcute hccl running parameters completed. stream num:[%llu], workspace size:[%llu]bytes", streamNum,
            opMemSize);
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomGraphOptimizer::SetOpOutputMemSize(ge::Node &node, const std::string &sCollectiveType) {
  ge::OpDescPtr op = node.GetOpDesc();
  for (u32 i = 0; i < op->GetOutputsSize(); i++) {
    int64_t memSize = 0;
    ge::GeTensorDesc outputTensor = op->GetOutputDesc(i);
    ge::GeShape outputShape = outputTensor.GetShape();
    ge::Format format = outputTensor.GetFormat();
    ge::DataType dataType = outputTensor.GetDataType();
    // 获取内存大小
    bool bErr = (ge::GRAPH_SUCCESS != ge::TensorUtils::CalcTensorMemSize(outputShape, format, dataType, memSize));
    CHK_PRT_RET(bErr,
                HCCL_ERROR("[Set][OpOutputMemSize]In get output mem size, error outputSize because no"
                           "know shape, Format[%d], dataType[%d], outputSize[%lld], index[%u]",
                           format, dataType, memSize, i),
                HCCL_E_PARA);

    if (memSize == -1) {  // memsize 为-1 时，表示输入的shape不正确
      HCCL_ERROR(
          "[Set][OpOutputMemSize]In get output mem size, error outputSize because unknow shape,"
          "Format[%d], dataType[%d], outputSize[%lld], index[%u]",
          format, dataType, memSize, i);
      return HCCL_E_PARA;
    }

    // 根据 规则重新计算内存大小
    HcclResult ret = CalcHCCLOutputMemSize(sCollectiveType, memSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Calc][OutputMemSize]In calc output mem size, cacl, memsize error, ret[%d],"
                           "sCollectiveType[%s], memSize[%lld]",
                           ret, sCollectiveType.c_str(), memSize),
                HCCL_E_PARA);

    // 将内存大小重新传给上层
    ge::TensorUtils::SetSize(outputTensor, memSize);

    // 更新output Tensor
    if (op->UpdateOutputDesc(i, outputTensor) != ge::GRAPH_SUCCESS) {
      HCCL_ERROR(
          "[Calc][OutputMemSize]In get output mem size, update output desc error,"
          "Format[%d], dataType[%d], outputSize[%lld], index[%u]",
          format, dataType, memSize, i);
      return HCCL_E_PARA;
    }
    HCCL_INFO("In set output MemSize, sCollectiveType[%s], opMemSize[%lld]", sCollectiveType.c_str(), memSize);
  }
  return HCCL_SUCCESS;
}

HcclResult AutoTuningHcomGraphOptimizer::SetOpMemAttr(ge::Node &node, const std::string &sCollectiveType,
                                                      const u64 &opMemSize) {
  bool bRet = false;

  // ATTENTION: 算子在IR定义时input/output同名场合（参考HcomRemoteRefRead算子）会隐式设置reference属性为TRUE,
  //   此处只对IR定义中input/output不同名且需要复用内存的算子，进行内存复用配置。
  //   后续有类似算子实现建议在IR定义时将input/output配置为相同name。
  // broadcast算子因为输入/输出为同一内存Ref属性为true
  if (sCollectiveType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
    bRet = ge::AttrUtils::SetBool(node.GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
    CHK_PRT_RET(!bRet,
                HCCL_ERROR("[Set][OpMemAttr]errNo[0x%016llx] op[%s]: set  reference attr[%d] to OpDesc"
                           "failed.",
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

  string sGroup;
  CHK_RET(HcomOpUtils::GetGroupFromOpDesc(node.GetOpDesc(), sGroup));
  std::string socVersion;
  if (ge::GetThreadLocalContext().GetOption(ge::SOC_VERSION, socVersion) != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("[offline][compilation] get soc version failed.");
    return HCCL_E_INTERNAL;
  }
  u32 memType = 0;
  u32 p2pMemType = RT_MEMORY_P2P_DDR;
  CHK_RET(HcomGetMemType(sGroup.c_str(), socVersion.c_str(), false, &memType, nullptr, true));
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
}  // namespace hccl
