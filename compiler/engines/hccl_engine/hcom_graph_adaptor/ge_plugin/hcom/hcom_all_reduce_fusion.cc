/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_all_reduce_fusion.h"
#include "hcom_ops_kernel_info_store.h"
#include "hcom_op_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "calculation_tool/evaluator.h"
#include "calculation_tool/model.h"
#include "calculation_tool/cluster.h"
#include "mmpa/mmpa_api.h"
#include "hccl/hcom.h"

using namespace hccl;
using namespace std;

namespace hccl {
HcomAllReduceFusion::HcomAllReduceFusion()
    : bHasUnknownShapeNodeGraph_(false),
      unknownShapeOriginalGraph_(false),
      fusionHash_("HCCL_WORLD_HASH"),
      modelGraphId(0),
      tensorFusionLimit_(std::numeric_limits<u64>::max()) {}

HcomAllReduceFusion::~HcomAllReduceFusion() {}

HcclResult HcomAllReduceFusion::Run(ge::ComputeGraph &graph) {
  CHK_RET(GetFusionInformation(graph, fusionHash_));
  FusionInfos fusionInfos;
  HcclResult ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][ReduceFusion]graph[%s]: get fusion HcomAllReduce ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);
  HCCL_INFO("there are %u group to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of HcomAllReduce operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0), HCCL_INFO("NOT_CHANGED: the graph has no HcomAllReduce op."), HCCL_SUCCESS);

  FusionInfos fusionInfosTemp;
  if (unknownShapeOriginalGraph_) {
    ret = GetFusionOpsSlices(fusionInfos, fusionInfosTemp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][ReduceFusion]graph[%s]: GetFusionOpsSlices failed. ret[%d]", graph.GetName().c_str(), ret),
        ret);
  } else {
    fusionInfosTemp = fusionInfos;
  }

  for (auto iterFusionInfos = fusionInfosTemp.begin(); iterFusionInfos != fusionInfosTemp.end(); iterFusionInfos++) {
    HCCL_INFO("graph[%s] fusionlabel[%s]: there are %zu HcomAllreduce ops to be fused.", graph.GetName().c_str(),
              iterFusionInfos->first.c_str(), iterFusionInfos->second.size());

    ret = FuseOps(graph, iterFusionInfos->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] fusionlabel[%s]: fusion HcomAllreduce ops failed. "
                           "ret[%d]",
                           graph.GetName().c_str(), iterFusionInfos->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("fuse HcomAllReduce op end");
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetFusionOpsSlices(FusionInfos &fusionInfos, FusionInfos &fusionInfosTemp) {
  // 获取cclbuffer size
  u64 cclBuffSize;
  CHK_RET(GetCCLBufferAvailableSize(cclBuffSize));

  for (auto fusionInfosIndex : fusionInfos) {
    // 获取当前map的value值
    std::vector<ge::NodePtr> fusionVector;
    for (auto index : fusionInfosIndex.second) {
      fusionVector.push_back(index);
    }
    size_t i = 0;
    uint64_t cursize = 0;
    int flag = 0;
    while (i < fusionVector.size()) {
      auto opDescPtr = fusionVector[i]->GetOpDesc();
      if (!opDescPtr) {
        HCCL_WARNING("desc of node[%s] is null.", fusionVector[i]->GetName().c_str());
        continue;
      }
      uint64_t memSize = 0;
      CHK_RET(HcomOpUtils::GetAllInputsTensorMemSize(opDescPtr, memSize));
      if (cursize + memSize > cclBuffSize) {
        // 新建一个vectorTemp保存当前fusionVector[0]到fusionVector[i]，vectorTemp inesert到map中
        // 删除fusionVector[0]到fusionVector[i]，cursize和i为0，然后continue跳出循环
        if (i == 0) {
          i++;
        }
        std::vector<ge::NodePtr> vectorTemp;
        vectorTemp.assign(fusionVector.begin(), fusionVector.begin() + i);
        std::string fusionLabelTemp = fusionInfosIndex.first + "_" + std::to_string(flag);
        fusionInfosTemp.insert(std::make_pair(fusionLabelTemp, vectorTemp));
        flag++;
        fusionVector.erase(fusionVector.begin(), fusionVector.begin() + i);
        cursize = 0;
        i = 0;
        continue;
      }
      if (i + 1 == fusionVector.size()) {
        std::string fusionLabelTemp = fusionInfosIndex.first + "_" + std::to_string(flag);
        fusionInfosTemp.insert(std::make_pair(fusionLabelTemp, fusionVector));
      }
      cursize = cursize + memSize;
      i++;
    }
  }

  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  HcclResult ret;
  int64_t hcomComm = 0;
  CHK_PRT_RET((fusionSection.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %u HcomAllReduce op.", fusionSection.size()), HCCL_SUCCESS);

  bool bRet = ge::AttrUtils::GetInt(fusionSection[0]->GetOpDesc(), "comm", hcomComm);
  if (bRet && hcomComm != static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    ret = RunFusionOpsReduce(graph, fusionSection);
    HCCL_INFO("[HcclCommGraph][Type]FuseOps with comm[%ld].", hcomComm);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);
  } else {
    u32 segmentNum = 0;
    std::vector<u32> segmentIndex;
    ret = GetFusionStrategy(graph, fusionSection, segmentNum, segmentIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Ops][Fuse]graph[%s]: get HcomAllReduce ops split strategy"
                           "failed. ret[%d]",
                           graph.GetName().c_str(), ret),
                ret);
    // The number of split segments must be not equal to the number of HcomAllReduce operator
    CHK_PRT_RET((segmentNum == fusionSection.size()),
                HCCL_INFO("NOT_CHANGED: split segments[%u] is equal to "
                          "num[%u] of HcomAllReduce op.",
                          segmentNum, fusionSection.size()),
                HCCL_SUCCESS);

    ret = RunFusionOps(graph, fusionSection, segmentNum, segmentIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);
  }

  HCCL_INFO("graph[%s] fuse HcomAllReduce op end.", graph.GetName().c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos) {
  HcclResult ret;
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("HcomAllReduceFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomAllReduceFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    bool unknownShapeNode = false;
    ret = GetNodeUnknownShapeInfo(nodePtr, unknownShapeNode);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][FusionOps]node[%s] get node unknown shape info failed.", nodePtr->GetName().c_str()),
                ret);
    if (unknownShapeNode) {
      unknownShapeOriginalGraph_ = true;
    }

    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
      ret = GetFusionOpInfo(nodePtr, fusionInfos);
      CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("get fusion ops by group failed. ret[%d]", ret), ret);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
  auto opDescPtr = nodePtr->GetOpDesc();
  string nodeName = nodePtr->GetName();
  bool bErr = false;
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION, fusionOption.fusionAttr)) {
    fusionOption.fusionAttr = HCOM_ATTR_FUSION_BY_SPLIT_STRATEGY;
    HCCL_WARNING("node[%s] has no attr[%s], use default value[%lld].", nodeName.c_str(), HCOM_ATTR_NAME_FUSION.c_str(),
                 fusionOption.fusionAttr);
  }
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION_ID, fusionOption.fusionId)) {
    fusionOption.fusionId = HCOM_ATTR_FUSION_ID_DEFAULT;
    HCCL_WARNING("node[%s] has no attr[%s], use default value[%lld].", nodeName.c_str(),
                 HCOM_ATTR_NAME_FUSION_ID.c_str(), fusionOption.fusionId);
  }
  switch (fusionOption.fusionAttr) {
    case HCOM_ATTR_FUSION_NO_FUSION:
      HCCL_DEBUG("node[%s] with attr fusion[%lld], no fusion", nodeName.c_str(), fusionOption.fusionAttr);
      break;
    case HCOM_ATTR_FUSION_BY_SPLIT_STRATEGY:
      CHK_PRT_RET((fusionOption.fusionId != HCOM_ATTR_FUSION_ID_DEFAULT),
                  HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] node[%s] fusion[%lld] fusion_id[%lld]: fusion_id is"
                             "incorrect, should be %lld.",
                             HCOM_ERROR_CODE(HCCL_E_PARA), nodeName.c_str(), fusionOption.fusionAttr,
                             fusionOption.fusionId, HCOM_ATTR_FUSION_ID_DEFAULT),
                  HCCL_E_PARA);
      break;
    case HCOM_ATTR_FUSION_BY_FUSION_ID:
      bErr = ((fusionOption.fusionId < HCOM_ATTR_FUSION_ID_MIN) || (fusionOption.fusionId > HCOM_ATTR_FUSION_ID_MAX));
      CHK_PRT_RET(
          bErr,
          HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] node[%s] fusion[%lld]"
                     "fusion_id[%lld]: fusion_id is incorrect",
                     HCOM_ERROR_CODE(HCCL_E_PARA), nodeName.c_str(), fusionOption.fusionAttr, fusionOption.fusionId),
          HCCL_E_PARA);
      break;
    default:
      string fusionValue = std::to_string(fusionOption.fusionAttr);
      REPORT_PREDEFINED_ERR_MSG("EI0003", std::vector<const char *>({"ccl_op", "parameter", "value", "tips"}),
                                std::vector<const char *>({"HcomAllReduceFusion", "fusion", fusionValue.c_str(),
                                                           "please check fusion setting"}));
      HCCL_ERROR("[%s][%s]errNo[0x%016llx] node[%s] fusion[%lld] is incorrect, should be %lld ~ %lld",
                 LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA),
                 nodeName.c_str(), fusionOption.fusionAttr, HCOM_ATTR_FUSION_MIN, HCOM_ATTR_FUSION_MAX);
      return HCCL_E_PARA;
  }

  // 获取comm和group
  CHK_RET(GetCommFromOpDescPtr(nodePtr, fusionOption));
  CHK_RET(HcomOpUtils::GetDataType(nodePtr->GetOpDesc(), fusionOption.dtype));

  bErr = ge::AttrUtils::GetStr(opDescPtr, "reduction", fusionOption.reduction);
  CHK_PRT_RET(!bErr, HCCL_ERROR("errNo[0x%016llx] get attr \"reduction\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  fusionOption.optype = nodePtr->GetOpDesc()->GetType();
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetNodeUnknownShapeInfo(ge::NodePtr &nodePtr, bool &bUnknownShapeNode) {
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, bUnknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Get][Node]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
              HCCL_E_PARA);
  if (bUnknownShapeNode) {
    if (!bHasUnknownShapeNodeGraph_) {
      bHasUnknownShapeNodeGraph_ = true;
      HCCL_INFO("node[%s] is unknown shape", nodePtr->GetName().c_str());
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel) {
  if (fusionOption.fusionComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    fusionLabel = fusionOption.optype + "-" + fusionOption.group + "-" + to_string(fusionOption.fusionId) + "-" +
                  fusionOption.dtype + "-" + fusionOption.reduction;
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(fusionOption.fusionComm, &group));
    std::string identifier = std::string(group);
    fusionLabel = fusionOption.optype + "-" + identifier + "-" + to_string(fusionOption.fusionId) + "-" +
                  fusionOption.dtype + "-" + fusionOption.reduction;
    HCCL_DEBUG("[HcclCommGraph][Type]GenerateFusionLabel.");
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos) {
  bool bUnknownShapeNode = false;
  HcclResult ret = GetNodeUnknownShapeInfo(nodePtr, bUnknownShapeNode);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node unknown shape info failed.", nodePtr->GetName().c_str()),
              ret);
  CHK_PRT_RET(bUnknownShapeNode, HCCL_INFO("node[%s] is unknown shape, no fusion", nodePtr->GetName().c_str()),
              HCCL_SUCCESS);
  std::string reduction;
  CHK_PRT_RET(
      !ge::AttrUtils::GetStr(nodePtr->GetOpDesc(), ge::HCOM_ATTR_REDUCE_TYPE, reduction),
      HCCL_ERROR("[Get][FusionOpInfo]errNo[0x%016llx] get attr \"reduction\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);
  CHK_PRT_RET(reduction != "sum",
              HCCL_INFO("node[%s] reduction is %s, no fusion", nodePtr->GetName().c_str(), reduction.c_str()),
              HCCL_SUCCESS);
  FusionOption fusionOption;
  ret = GetFusionOption(nodePtr, fusionOption);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node unknown shape info failed.", nodePtr->GetName().c_str()),
              ret);
  CHK_PRT_RET(
      fusionOption.fusionAttr == 0,
      HCCL_INFO("node[%s] with attr fusion[%lld], no fusion", nodePtr->GetName().c_str(), fusionOption.fusionAttr),
      HCCL_SUCCESS);

  HCCL_DEBUG(
      "get fusion op: node[%s]: comm[%ld], group[%s], fusion[%lld], fusion_id[%lld], dtype[%s], \
        reduction[%s]",
      nodePtr->GetName().c_str(), fusionOption.fusionComm, fusionOption.group.c_str(), fusionOption.fusionAttr,
      fusionOption.fusionId, fusionOption.dtype.c_str(), fusionOption.reduction.c_str());

  std::string fusionLabel;
  ret = GenerateFusionLabel(fusionOption, fusionLabel);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("node[%s] generate fusion label failed.", nodePtr->GetName().c_str()),
              ret);
  HCCL_DEBUG("node[%s] generate fusion label[%s]", nodePtr->GetName().c_str(), fusionLabel.c_str());

  auto opDescPtr = nodePtr->GetOpDesc();
  ret = AddHcclFusionFlag(opDescPtr);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]add hccl fusion flag to node[%s] failed. ret[%d]",
                         nodePtr->GetName().c_str(), ret),
              ret);

  auto iterFusionInfos = fusionInfos.find(fusionLabel);
  if (iterFusionInfos == fusionInfos.end()) {
    FusionSection fusionSection;
    fusionSection.push_back(nodePtr);
    fusionInfos.insert({fusionLabel, fusionSection});
  } else {
    iterFusionInfos->second.push_back(nodePtr);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetFusionStrategy(const ge::ComputeGraph &graph, const FusionSection &fusionSection,
                                                  u32 &segmentNum, std::vector<u32> &segmentIndex) {
  std::string nodeGroup;
  int64_t nodeFusionId;

  bool bRet = ge::AttrUtils::GetStr(fusionSection[0]->GetOpDesc(), "group", nodeGroup);
  CHK_PRT_RET(!bRet, HCCL_ERROR("errNo[0x%016llx] get attr \"group\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  bRet = ge::AttrUtils::GetInt(fusionSection[0]->GetOpDesc(), "fusion_id", nodeFusionId);
  CHK_PRT_RET(!bRet, HCCL_ERROR("errNo[0x%016llx] get attr \"fusion_id\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  FusionOption option(nodeGroup, nodeFusionId);

  if (option.fusionId == HCOM_ATTR_FUSION_ID_DEFAULT) {
    HcclResult ret = GetGradSplitStrategy(graph.GetName(), option.group, fusionSection, segmentNum, segmentIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][FusionStrategy]graph[%s]: group[%s] get HcomAllReduce ops split"
                           "strategy failed. ret[%d]",
                           graph.GetName().c_str(), option.group.c_str(), ret),
                ret);
  } else {
    segmentNum = 1;
    segmentIndex.clear();
    segmentIndex.push_back(fusionSection.size() - 1);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetGradSplitStrategy(const std::string &modelName, const std::string &sGroup,
                                                     const FusionSection &fusionSection, u32 &segmentNum,
                                                     std::vector<u32> &segmentIndex) {
  uint64_t tensorSize = 0;
  bool isUseFusionLib = false;
  u32 gradientNum = fusionSection.size();

  std::vector<float> inputGradientSize(gradientNum, 0.0);
  std::vector<float> inputGradientTime(gradientNum, 0.0);
  for (u32 inputTensorIdx = 0; inputTensorIdx < gradientNum; inputTensorIdx++) {
    HcclResult ret = HcomOpUtils::GetAllInputsTensorMemSize(fusionSection[inputTensorIdx]->GetOpDesc(), tensorSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][GradSplitStrategy]In GetGradSplitStrategy, GetAllInputsTensorMemSize"
                           "failed, node[%s], ret[%u].",
                           fusionSection[inputTensorIdx]->GetOpDesc()->GetName().c_str(), ret),
                ret);
    HCCL_DEBUG("GetGradSplitStrategy: fusionOps idx[%u] node[%s] Size[%llu]", inputTensorIdx,
               fusionSection[inputTensorIdx]->GetOpDesc()->GetName().c_str(), tensorSize);
    inputGradientSize[inputTensorIdx] = static_cast<float>(tensorSize);
  }

  model_feature modelFeature{modelName.c_str(), gradientNum, inputGradientSize.data(), inputGradientTime.data()};

  GradSplitForceMode forceMode =
      bHasUnknownShapeNodeGraph_ ? GradSplitForceMode::FORCE_SIZE : GradSplitForceMode::FORCE_NONE;
  OriginalGraphShapeType shapeType =
      unknownShapeOriginalGraph_ ? OriginalGraphShapeType::UNKNOWN_SHAPE : OriginalGraphShapeType::KNOWN_SHAPE;
  u32 *midSegmentIndexPtr = nullptr;
  u32 len;
  HcclResult ret = HcomGetSplitStrategy(sGroup.c_str(), &modelFeature, &midSegmentIndexPtr, &len, &isUseFusionLib,
                                        forceMode, shapeType);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][SplitStrategy]group[%s] get gradient segment failed. ret[%d]", sGroup.c_str(), ret),
              ret);

  std::vector<u32> midSegmentIndex;
  if (len > 0 && midSegmentIndexPtr != nullptr) {
    midSegmentIndex = std::vector<u32>(midSegmentIndexPtr, midSegmentIndexPtr + len);
  } else {
    midSegmentIndex = std::vector<u32>();
  }

  if (!isUseFusionLib) {
    HCCL_INFO("[Get][SplitStrategy]Get SegmentIndex from Shell Script.");
    segmentIndex = midSegmentIndex;
  } else {
    HCCL_INFO("[Get][SplitStrategy]Find SegmentIndex from fusion library.");
    if (bHasUnknownShapeNodeGraph_) {
      u64 size = 0;
      CHK_RET(GetCCLBufferAvailableSize(size));
      tensorFusionLimit_ = size;
    }
    ret = CalculateSegmentIndex(fusionHash_, tensorFusionLimit_, segmentIndex);
    // segmentIndex得到allreduce只有1个时，选用默认切分策略，防止由于知识库计算误差导致拖尾时间变长
    if ((ret != HCCL_SUCCESS) || (segmentIndex.size() == 1)) {
      segmentIndex = midSegmentIndex;
    }
  }
  segmentNum = segmentIndex.size();
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::AddHcclFusionFlag(ge::OpDescPtr &opDescPtr) {
  bool bRet = ge::AttrUtils::SetBool(opDescPtr, "_hccl_fused_node", true);
  CHK_PRT_RET(!bRet,
              HCCL_ERROR("[Add][FusionFlag]errNo[0x%016llx] op[%s]: set _hccl_fused_node attr[%d] failed.",
                         HCOM_ERROR_CODE(HCCL_E_PARA), opDescPtr->GetName().c_str(), true),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::GetFusionInformation(const ge::ComputeGraph &graph, std::string &fusionHash) {
  HcclResult ret;
  bool isAllReduce = false;
  for (auto nodePtr : graph.GetDirectNode()) {
    std::string midFusionHash;
    if (!nodePtr) {
      HCCL_WARNING("HcomAllReduceFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomAllReduceFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
      HCCL_WARNING("HcomAllReduceFusion: null AllReduce exists.");
      isAllReduce = true;
      break;
    }
  }
  if (isAllReduce) {
    ret = GetFusionhashFromGraph(const_cast<ge::ComputeGraph &>(graph), fusionHash);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] get fusionhash failed, ret[%d]", graph.GetName().c_str(), ret), ret);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllReduceFusion::CalculateSegmentIndex(std::string &fusionHash, u64 tensorLimit,
                                                      std::vector<u32> &segmentIndex) {
  HcclResult ret;
  std::string ConfigVersion;
  std::string fusionStartPath;
  std::string fusionEndPath;

  CHK_RET(HcomOpUtils::CreateFusionConfigVersion(ConfigVersion));

  char *configEnv = nullptr;
  MM_SYS_GET_ENV(MM_ENV_TUNE_BANK_PATH, configEnv);
  auto CalculateSegmentByConfig = [&]() -> HcclResult {
    CHK_RET(HcomOpUtils::GetPathFromEnv(configEnv, fusionStartPath));
    CHK_RET(HcomOpUtils::GetFileNameFromPath(fusionStartPath, fusionEndPath));
    CHK_RET(GetInformationFromLibrary(fusionEndPath, fusionHash, tensorLimit, segmentIndex));
    return HCCL_SUCCESS;
  };
  ret = CalculateSegmentByConfig();
  if (ret == HCCL_SUCCESS) {
    return HCCL_SUCCESS;
  }

  char *aoeEnv = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HOME, aoeEnv);
  auto CalculateSegmentByAoe = [&]() -> HcclResult {
    CHK_RET(HcomOpUtils::GetPathFromEnv(aoeEnv, fusionStartPath));
    fusionStartPath = fusionStartPath + "Ascend/latest/data/aoe/custom/graph/" + ConfigVersion + "/";
    CHK_RET(HcomOpUtils::GetFileNameFromPath(fusionStartPath, fusionEndPath));
    CHK_RET(GetInformationFromLibrary(fusionEndPath, fusionHash, tensorLimit, segmentIndex));
    return HCCL_SUCCESS;
  };
  ret = CalculateSegmentByAoe();
  if (ret == HCCL_SUCCESS) {
    return HCCL_SUCCESS;
  }

  auto CalculateSegmentByDefault = [&]() -> HcclResult {
    CHK_RET(GetPathFromDefault(fusionStartPath));
    fusionEndPath = fusionStartPath + ConfigVersion + "_gradient_fusion.json";
    CHK_RET(GetInformationFromLibrary(fusionEndPath, fusionHash, tensorLimit, segmentIndex));
    return HCCL_SUCCESS;
  };
  ret = CalculateSegmentByDefault();
  if (ret == HCCL_SUCCESS) {
    return HCCL_SUCCESS;
  }
  return HCCL_E_PARA;
}

HcclResult HcomAllReduceFusion::GetInformationFromLibrary(std::string &fusionPath, std::string &fusionHash,
                                                          u64 tensorLimit, std::vector<u32> &segmentIndex) {
  HcclResult ret;
  std::fstream jFile;
  jFile.open(fusionPath, std::ios::in);
  if (jFile.peek() == std::ifstream::traits_type::eof()) {
    jFile.close();
    HCCL_INFO("[Get][Info]The library is empty, no hash matched.");
    return HCCL_E_AGAIN;
  }
  jFile.close();
  ret = GetInfoFromContentedLibrary(fusionPath, fusionHash, tensorLimit, segmentIndex);
  return ret;
}

HcclResult HcomAllReduceFusion::GetInfoFromContentedLibrary(std::string fusionPath, std::string &fusionHash,
                                                            u64 tensorLimit, std::vector<u32> &segmentIndex) {
  HcclResult ret;
  std::vector<uint64_t> graInfoCost;
  std::vector<uint64_t> graInfoSize;
  bool hasModelHash = false;
  const int patchSize = 32;
  std::fstream File;
  File.open(fusionPath, std::ios::in);
  nlohmann::json root;
  File >> root;
  HCCL_DEBUG("[Get][Info]File is nonempty.");
  int32_t rootSize = root.size();
  for (auto i = 0; i < rootSize; i++) {
    if (root[i]["modelhash"] == fusionHash) {
      hasModelHash = true;
      int32_t valueSize = root[i]["modelvalue"]["gradientTime"].size();
      for (auto j = 0; j < valueSize; j++) {
        graInfoCost.push_back(root[i]["modelvalue"]["gradientTime"][j]);
        graInfoSize.push_back(root[i]["modelvalue"]["gradientSize"][j]);
      }
    }
  }
  if (hasModelHash) {
    // 计算segment_index;
    std::vector<int> result;
    std::string segmentvalue;
    Cluster cluster(fusionPath);
    Model model(graInfoCost, graInfoSize, patchSize, tensorLimit);
    EvaluatorDataParallel evaluatorData;
    result = evaluatorData.run(model, cluster, patchSize);
    for (size_t i = 0; i < result.size(); i++) {
      segmentIndex.push_back(static_cast<u32>(result[i] - 1));
      segmentvalue = segmentvalue + to_string(result[i] - 1) + ",";
    }
    HCCL_RUN_INFO("[Get][Info]Use fusion library value [%s].", segmentvalue.c_str());
    ret = HCCL_SUCCESS;
  } else {
    HCCL_WARNING("Can not match the same hash value in library, This may be due to the following reasons:");
    HCCL_WARNING("    1. It could be a Dynamic Shape Network;");
    HCCL_WARNING("    2. It may not contain value of the Network;");
    HCCL_WARNING("    3. The Optimized Graph is different from Training Graph.");
    ret = HCCL_E_AGAIN;
  }
  File.close();
  return ret;
}

HcclResult HcomAllReduceFusion::GetPathFromDefault(std::string &fusionPath) {
  mmDlInfo infos;
  mmDladdr(reinterpret_cast<void *>(HcomGetRankSize), &infos);
  std::string linkPath = infos.dli_fname;
  uint32_t linkPathSize = linkPath.length();
  uint32_t escapeLinkNum = 0;
  std::string midLinkPath;
  std::reverse(linkPath.begin(), linkPath.end());
  for (uint32_t i = 0; i < linkPathSize; i++) {
    if ('/' == linkPath[i]) {
      midLinkPath = linkPath.substr(0, i);
      escapeLinkNum += 1;
    }
    if (escapeLinkNum == static_cast<uint32_t>(CreateDir::HCCL_DIR_NUM_TWO)) {
      break;
    }
  }
  std::reverse(linkPath.begin(), linkPath.end());
  linkPath = linkPath.substr(0, linkPath.size() - midLinkPath.size());
  fusionPath = linkPath + "data/fusion_strategy/build_in/";
  HCCL_INFO("Get Fusion Library from Default Path %s.", fusionPath.c_str());
  return HCCL_SUCCESS;
}
}  // namespace hccl
