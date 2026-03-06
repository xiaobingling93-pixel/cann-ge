/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_reduce_fusion.h"
#include "hcom_ops_kernel_info_store.h"
#include "hccl/hcom.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"

using namespace hccl;
using namespace std;

namespace hccl {
HcomReduceFusion::HcomReduceFusion() {}

HcomReduceFusion::~HcomReduceFusion() {}

HcclResult HcomReduceFusion::Run(ge::ComputeGraph &graph) {
  HcclResult ret;

  FusionInfos fusionInfos;
  ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][HcomReduceFusion]graph[%s]: get fusion Reduce ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  HCCL_INFO("there are %llu group to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of HcomReduce operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0), HCCL_INFO("NOT_CHANGED: the graph has no HcomReduce op."), HCCL_SUCCESS);

  FusionInfos fusionInfosTemp;
  ret = GetFusionOpsSlices(fusionInfos, fusionInfosTemp);
  CHK_PRT_RET(
      ret != HCCL_SUCCESS,
      HCCL_ERROR("[Run][ReduceFusion]graph[%s]: GetFusionOpsSlices failed. ret[%d]", graph.GetName().c_str(), ret),
      ret);

  for (auto iterFusionInfos = fusionInfosTemp.begin(); iterFusionInfos != fusionInfosTemp.end(); iterFusionInfos++) {
    ret = FuseOps(graph, iterFusionInfos->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s]: fusion_lable[%s] fusion Reduce ops failed. "
                           "ret[%d]",
                           graph.GetName().c_str(), iterFusionInfos->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("graph[%s] fuse HcomReduce op end", graph.GetName().c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomReduceFusion::GetFusionOpsSlices(FusionInfos &fusionInfos, FusionInfos &fusionInfosTemp) {
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

HcclResult HcomReduceFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  HcclResult ret;
  CHK_PRT_RET((fusionSection.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %llu HcomReduce op.", fusionSection.size()), HCCL_SUCCESS);

  ret = RunFusionOpsReduce(graph, fusionSection);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);
  return HCCL_SUCCESS;
}

HcclResult HcomReduceFusion::GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos) {
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("HcomReduceFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomReduceFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCE) {
      CHK_RET(GetFusionOpInfo(nodePtr, fusionInfos));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceFusion::GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
  auto opDescPtr = nodePtr->GetOpDesc();
  string nodeName = nodePtr->GetName();
  bool bErr = false;
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION, fusionOption.fusionAttr)) {
    fusionOption.fusionAttr = HCOM_ATTR_FUSION_NO_FUSION;
    HCCL_WARNING("node[%s] has no attr[%s], use default value[%lld].", nodeName.c_str(), HCOM_ATTR_NAME_FUSION.c_str(),
                 fusionOption.fusionAttr);
  }
  // 如果没有设置fusionid，会将该值设置为-1,不进行融合
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION_ID, fusionOption.fusionId)) {
    fusionOption.fusionAttr = HCOM_ATTR_FUSION_NO_FUSION;
    fusionOption.fusionId = HCOM_ATTR_FUSION_ID_DEFAULT;
    HCCL_WARNING("node[%s] has no attr[%s], use default value[%lld] then no fusion.", nodeName.c_str(),
                 HCOM_ATTR_NAME_FUSION_ID.c_str(), fusionOption.fusionId);
  }
  switch (fusionOption.fusionAttr) {
    case HCOM_ATTR_FUSION_NO_FUSION:
      HCCL_DEBUG("node[%s] with attr fusion[%lld], no fusion", nodeName.c_str(), fusionOption.fusionAttr);
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
      HCCL_ERROR(
          "[Get][FusionOption]errNo[0x%016llx] node[%s] fusion[%lld] is incorrect, should"
          "be %lld or %lld",
          HCOM_ERROR_CODE(HCCL_E_PARA), nodeName.c_str(), fusionOption.fusionAttr, HCOM_ATTR_FUSION_NO_FUSION,
          HCOM_ATTR_FUSION_BY_FUSION_ID);
      return HCCL_E_PARA;
  }
  // 获取comm和group
  CHK_RET(GetCommFromOpDescPtr(nodePtr, fusionOption));
  CHK_RET(HcomOpUtils::GetDataType(nodePtr->GetOpDesc(), fusionOption.dtype));
  bErr = ge::AttrUtils::GetStr(opDescPtr, "reduction", fusionOption.reduction);
  CHK_PRT_RET(
      !bErr,
      HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] get attr \"reduction\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);
  // 判断root属性是否包含
  CHK_PRT_RET(GetFusionRootAndLabel(nodePtr, fusionOption) != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] get attr root and label\" failed. ",
                         HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  fusionOption.optype = nodePtr->GetOpDesc()->GetType();
  fusionOption.reduction = "sum";
  return HCCL_SUCCESS;
}

HcclResult HcomReduceFusion::GetFusionRootAndLabel(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
  bool bRet = ge::AttrUtils::GetInt(nodePtr->GetOpDesc(), "root_rank", fusionOption.root);
  CHK_PRT_RET(
      !bRet,
      HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] get attr \"root_rank\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);
  return HCCL_SUCCESS;
}
HcclResult HcomReduceFusion::GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel) {
  std::string operation = fusionOption.reduction.empty() ? "NA" : fusionOption.reduction;
  if (fusionOption.fusionComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    fusionLabel = fusionOption.optype + "-" + fusionOption.group + "-" + to_string(fusionOption.fusionId) + "-" +
                  fusionOption.dtype + "-" + to_string(fusionOption.root) + "-" + operation;
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(fusionOption.fusionComm, &group));
    std::string identifier = std::string(group);
    fusionLabel = fusionOption.optype + "-" + identifier + "-" + to_string(fusionOption.fusionId) + "-" +
                  fusionOption.dtype + "-" + to_string(fusionOption.root) + "-" + operation;
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceFusion::GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos) {
  bool bUnknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, bUnknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
              HCCL_E_PARA);

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
  HcclResult ret = GetFusionOption(nodePtr, fusionOption);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get fusion config failed", nodePtr->GetName().c_str()), ret);
  CHK_PRT_RET(
      fusionOption.fusionAttr == 0,
      HCCL_INFO("node[%s] with attr fusion[%lld], no fusion", nodePtr->GetName().c_str(), fusionOption.fusionAttr),
      HCCL_SUCCESS);

  HCCL_DEBUG("get fusion op: node[%s]: comm[%lld], group[%s], fusion[%lld], fusion_id[%lld], dtype[%s], root[%lld]",
             nodePtr->GetName().c_str(), fusionOption.fusionComm, fusionOption.group.c_str(), fusionOption.fusionAttr,
             fusionOption.fusionId, fusionOption.dtype.c_str(), fusionOption.root);
  std::string fusionLabel;
  ret = GenerateFusionLabel(fusionOption, fusionLabel);

  auto opDescPtr = nodePtr->GetOpDesc();
  auto iterFusionInfos = fusionInfos.find(fusionLabel);
  if (iterFusionInfos == fusionInfos.end()) {
    FusionSection fusionSection;
    fusionSection.push_back(nodePtr);
    fusionInfos.insert({fusionLabel, fusionSection});
  } else {
    // 判断root是否同第一个节点一致
    int selfRoot = 0;
    bool bRet = ge::AttrUtils::GetInt(opDescPtr, "root_rank", selfRoot);
    CHK_PRT_RET(
        !bRet,
        HCCL_ERROR("[Get][FusionOpInfo]errNo[0x%016llx] get attr \"root_rank\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    int checkRoot = 0;
    CHK_PRT_RET(iterFusionInfos->second.empty(),
                HCCL_ERROR("[Get][FusionOpInfo]errNo[0x%016llx] get first node with fusionid failed. ",
                           HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);

    bRet = ge::AttrUtils::GetInt(iterFusionInfos->second[0]->GetOpDesc(), "root_rank", checkRoot);
    CHK_PRT_RET(
        !bRet,
        HCCL_ERROR("[Get][FusionOpInfo]errNo[0x%016llx] get attr \"root_rank\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    // 比较两个rank值是否一致
    CHK_PRT_RET(checkRoot != selfRoot,
                HCCL_ERROR("[Get][FusionOpInfo]errNo[0x%016llx] node root not equal ", HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);
    iterFusionInfos->second.push_back(nodePtr);
  }
  return HCCL_SUCCESS;
}
}  // namespace hccl
