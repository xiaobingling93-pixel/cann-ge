/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_reducescatter_fusion.h"
#include <cmath>
#include "hcom_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/ge_tensor.h"
#include "graph/types.h"

using namespace hccl;
using namespace std;

namespace hccl {
HcomReduceScatterFusion::HcomReduceScatterFusion() {}

HcomReduceScatterFusion::~HcomReduceScatterFusion() {}

HcclResult HcomReduceScatterFusion::Run(ge::ComputeGraph &graph) {
  FusionInfos fusionInfos;
  HcclResult ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][ReduceScatterFusion]graph[%s]: get fusion HcomReduceScatter ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);
  HCCL_INFO("there are %u group to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of  HcomReduceScatter operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0), HCCL_INFO("NOT_CHANGED: the graph has no HcomReduceScatter op."),
              HCCL_SUCCESS);

  FusionInfos fusionInfosTemp;
  ret = GetFusionOpsSlices(fusionInfos, fusionInfosTemp);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][ReduceScatterFusion]graph[%s]: GetFusionOpsSlices failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  for (auto iterFusionInfos = fusionInfosTemp.begin(); iterFusionInfos != fusionInfosTemp.end(); iterFusionInfos++) {
    HCCL_INFO("graph[%s] fusionlabel[%s]: there are %zu HcomReduceScatter ops to be fused.", graph.GetName().c_str(),
              iterFusionInfos->first.c_str(), iterFusionInfos->second.size());

    ret = FuseOps(graph, iterFusionInfos->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] fusionlabel[%s]: fusion HcomReduceScatter ops failed. "
                           "ret[%d]",
                           graph.GetName().c_str(), iterFusionInfos->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("fuse HcomReduceScatter op end");
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetFusionOpsSlices(FusionInfos &fusionInfos, FusionInfos &fusionInfosTemp) {
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

HcclResult HcomReduceScatterFusion::GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos) {
  HcclResult ret;
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("HcomReduceScatterFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomReduceScatterFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
      ret = GetFusionOpInfo(nodePtr, fusionInfos);
      CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("get fusion ops by group failed. ret[%d]", ret), ret);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos) {
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
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node info failed.", nodePtr->GetName().c_str()), ret);
  CHK_PRT_RET(
      fusionOption.fusionAttr != HCOM_ATTR_FUSION_BY_FUSION_ID,
      HCCL_INFO("node[%s] with attr fusion[%lld], no fusion", nodePtr->GetName().c_str(), fusionOption.fusionAttr),
      HCCL_SUCCESS);

  HCCL_DEBUG("get fusion op: node[%s]: comm[%ld], group[%s], fusion[%lld], fusion_id[%lld], dtype[%s]",
             nodePtr->GetName().c_str(), fusionOption.fusionComm, fusionOption.group.c_str(), fusionOption.fusionAttr,
             fusionOption.fusionId, fusionOption.dtype.c_str());

  std::string fusionLabel;
  ret = GenerateFusionLabel(fusionOption, fusionLabel);
  CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("node[%s] generate fusion label failed.", nodePtr->GetName().c_str()),
              ret);
  HCCL_DEBUG("node[%s] generate fusion label[%s]", nodePtr->GetName().c_str(), fusionLabel.c_str());

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

HcclResult HcomReduceScatterFusion::GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
  auto opDescPtr = nodePtr->GetOpDesc();
  string nodeName = nodePtr->GetName();
  bool bErr = false;
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION, fusionOption.fusionAttr)) {
    fusionOption.fusionAttr = HCOM_ATTR_FUSION_NO_FUSION;
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
                                std::vector<const char *>({"HcomReduceScatterFusion", "fusion", fusionValue.c_str(),
                                                           "please check fusion setting"}));
      HCCL_ERROR("[%s][%s]errNo[0x%016llx] node[%s] fusion[%lld] is incorrect, should be %lld or %lld",
                 LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA),
                 nodeName.c_str(), fusionOption.fusionAttr, HCOM_ATTR_FUSION_NO_FUSION, HCOM_ATTR_FUSION_BY_FUSION_ID);
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

HcclResult HcomReduceScatterFusion::GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel) {
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

HcclResult HcomReduceScatterFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  CHK_PRT_RET((fusionSection.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %u HcomReduceScatter op.", fusionSection.size()), HCCL_SUCCESS);

  HcclResult ret = RunFusionOpsReduceScatter(graph, fusionSection);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);

  HCCL_INFO("graph[%s] fuse HcomReduceScatter op end.", graph.GetName().c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::RunFusionOpsReduceScatter(ge::ComputeGraph &graph,
                                                              std::vector<ge::NodePtr> &fusionOps) {
  std::vector<CommonNodeInfo> nodeInfos(0);
  ge::OpDescPtr fusedOp;
  ReduceScatterFusionNodesInfo fusionNodesInfo;
  CHK_RET(RemoveOpsEdges(graph, fusionOps, nodeInfos, fusedOp));
  CHK_RET(AddFusionNode(graph, nodeInfos, fusionNodesInfo, fusedOp));
  CHK_RET(RestoreOpsEdges(nodeInfos, fusionNodesInfo));
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
                                                   std::vector<CommonNodeInfo> &nodeInfos, ge::OpDescPtr &fusedOp) {
  ge::graphStatus gRet;
  ge::OpDescPtr originDescPtr = fusionOps[0]->GetOpDesc();
  CHK_SMART_PTR_NULL(originDescPtr);
  fusedOp = ge::AttrUtils::CopyOpDesc(originDescPtr);
  CHK_SMART_PTR_NULL(fusedOp);

  for (uint32_t idx = 0; idx < fusionOps.size(); idx++) {
    CommonNodeInfo nodeInfo;
    // get all anchors connected to the node, and remove the edges between them.
    CHK_RET(GetPeerOutDataToInData(nodeInfo.peerOutDataAnchor, fusionOps[idx]));
    CHK_RET(GetPeerOutDataToInControl(nodeInfo.peerOutDataToInControl, fusionOps[idx]));
    CHK_RET(GetPeerOutControlToInControl(nodeInfo.peerOutControlAnchor, fusionOps[idx]));
    CHK_RET(GetPeerAnchorFromOutData(nodeInfo.peerInDataAnchor, nodeInfo.peerInControlFromOutData, fusionOps[idx]));
    CHK_RET(GetPeerInControlFromOutControl(nodeInfo.peerInControlAnchor, fusionOps[idx]));
    CHK_RET(GetReduceScatterOpInfo(nodeInfo.rankSize, nodeInfo.group, nodeInfo.nodeName, fusionOps[idx]));
    // remove the node after keeping all anchors
    gRet = graph.RemoveNode(fusionOps[idx]);
    CHK_PRT_RET((gRet != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Remove][OpsEdges]remove node[%s] failed. ret[%u]", nodeInfo.nodeName.c_str(), gRet),
                HCCL_E_INTERNAL);

    HCCL_DEBUG("fusionOps idx[%u] name[%s]", idx, nodeInfo.nodeName.c_str());
    nodeInfos.push_back(nodeInfo);

    HCCL_DEBUG(
        "graph[%s]: peerOutDataAnchor[%u], peerOutDataToInControl[%u], peerOutControlAnchor[%u], "
        "peerInDataAnchor[%u], peerInControlFromOutData[%u], peerInControlAnchor[%u].",
        graph.GetName().c_str(), nodeInfo.peerOutDataAnchor.size(), nodeInfo.peerOutDataToInControl.size(),
        nodeInfo.peerOutControlAnchor.size(), nodeInfo.peerInDataAnchor.size(),
        nodeInfo.peerInControlFromOutData.size(), nodeInfo.peerInControlAnchor.size());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerOutDataToInData(std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
                                                           ge::NodePtr &srcNodePtr) {
  CHK_SMART_PTR_NULL(srcNodePtr);
  CHK_SMART_PTR_NULL(srcNodePtr->GetOpDesc());
  std::string srcOpName = srcNodePtr->GetOpDesc()->GetName();
  for (auto inDataAnchor : srcNodePtr->GetAllInDataAnchors()) {
    if (!inDataAnchor) {
      continue;
    }
    ge::OutDataAnchorPtr peerOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    if (!peerOutDataAnchor) {
      continue;
    }
    peerOutDataAnchorVec.push_back(peerOutDataAnchor);
    CHK_PRT_RET(
        (ge::GraphUtils::RemoveEdge(peerOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS),
        HCCL_ERROR("[Get][Peer]GetPeerOutDataToInData: remove edge between peer outDataAnchor[%d] of Op[%s]"
                   "and inDataAnchor[%d] of Op[%s] failed.",
                   peerOutDataAnchor->GetIdx(), peerOutDataAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
                   inDataAnchor->GetIdx(), srcOpName.c_str()),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[Get][Peer]GetPeerOutDataToInData: remove edge between peer outDataAnchor[%d] of Op[%s] and "
        "inDataAnchor[%d] of Op[%s] ok.",
        peerOutDataAnchor->GetIdx(), peerOutDataAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
        inDataAnchor->GetIdx(), srcOpName.c_str());
  }

  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerOutDataToInControl(vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
                                                              ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  ge::InControlAnchorPtr inControlAnchor = srcNodePtr->GetInControlAnchor();
  CHK_SMART_PTR_NULL(inControlAnchor);
  for (auto peerOutDataToInControl : inControlAnchor->GetPeerOutDataAnchors()) {
    if (!peerOutDataToInControl) {
      continue;
    }
    peerOutDataToInControlVec.push_back(peerOutDataToInControl);
    gRet = ge::GraphUtils::RemoveEdge(peerOutDataToInControl, inControlAnchor);
    if (gRet != ge::GRAPH_SUCCESS) {
      HCCL_ERROR(
          "[Get][PeerOutData]GetPeerOutDataToInControl: remove edge between peer outDataAnchor[%d] of"
          "Op[%s] and inControlAnchor[%d] of Op[%s] failed. ret[%u]",
          peerOutDataToInControl->GetIdx(), peerOutDataToInControl->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
          inControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), gRet);
      return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[Get][PeerOutData]GetPeerOutDataToInControl: remove edge between peer outDataAnchor[%d] \
            of Op[%s] and inControlAnchor[%d] of Op[%s] ok",
        peerOutDataToInControl->GetIdx(), peerOutDataToInControl->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
        inControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerOutControlToInControl(
    vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec, ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  ge::InControlAnchorPtr inControlAnchor = srcNodePtr->GetInControlAnchor();
  CHK_SMART_PTR_NULL(inControlAnchor);
  for (auto peerOutControlAnchor : inControlAnchor->GetPeerOutControlAnchors()) {
    if (!peerOutControlAnchor) {
      continue;
    }
    peerOutControlToInControlVec.push_back(peerOutControlAnchor);
    gRet = ge::GraphUtils::RemoveEdge(peerOutControlAnchor, inControlAnchor);
    if (gRet != ge::GRAPH_SUCCESS) {
      HCCL_ERROR(
          "[Get][PeerOutControlToInControl]GetPeerOutControlToInControl: remove edge between peer"
          "outControlAnchor[%d] of Op[%s] and inControlAnchor[%d] of Op[%s] failed. ret[%u]",
          peerOutControlAnchor->GetIdx(), peerOutControlAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
          inControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), gRet);
      return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[Get][PeerOutControlToInControl]GetPeerOutControlToInControl: remove edge between peer \
            outControlAnchor[%d] of Op[%s] and inControlAnchor[%d] of Op[%s] ok",
        peerOutControlAnchor->GetIdx(), peerOutControlAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
        inControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerAnchorFromOutData(
    std::vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
    std::vector<ge::InControlAnchorPtr> &peerInControlFromOutDataVec, ge::NodePtr &srcNodePtr) {
  HcclResult ret;
  CHK_SMART_PTR_NULL(srcNodePtr);

  for (auto outDataAnchor : srcNodePtr->GetAllOutDataAnchors()) {
    if (!outDataAnchor) {
      continue;
    }

    ret = GetPeerInDataAnchorFromOutData(peerInDataFromOutDataVec, outDataAnchor, srcNodePtr);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][PeerAnchor]GetPeerInDataAnchorFromOutData failed."), ret);
    ret = GetPeerInControlAnchorFromOutData(peerInControlFromOutDataVec, outDataAnchor, srcNodePtr);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][PeerAnchor]GetPeerInControlAnchorFromOutData failed."), ret);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerInDataAnchorFromOutData(
    vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec, ge::OutDataAnchorPtr outDataAnchor,
    ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(outDataAnchor);
  CHK_SMART_PTR_NULL(srcNodePtr);
  HCCL_INFO("GetPeerAnchorFromOutData: node[%s] has %zu PeerInDataAnchors in OutDataAnchor[%d].",
            srcNodePtr->GetOpDesc()->GetName().c_str(), outDataAnchor->GetPeerInDataAnchors().size(),
            outDataAnchor->GetIdx());
  for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
    if (!peerInDataAnchor) {
      continue;
    }
    peerInDataFromOutDataVec.push_back(peerInDataAnchor);
    gRet = ge::GraphUtils::RemoveEdge(outDataAnchor, peerInDataAnchor);
    if (gRet != ge::GRAPH_SUCCESS) {
      HCCL_ERROR(
          "[Get][PeerInDataAnchor]GetPeerAnchorFromOutData: remove edge between outDataAnchor[%d]"
          "of Op[%s] and peer inDataAnchor[%d] of Op[%s] failed. ret[%u]",
          outDataAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInDataAnchor->GetIdx(),
          peerInDataAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(), gRet);
      return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[Get][PeerInDataAnchor]GetPeerAnchorFromOutData: remove edge between outDataAnchor[%d] "
        "of Op[%s] and peer inDataAnchor[%d] of Op[%s] ok",
        outDataAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInDataAnchor->GetIdx(),
        peerInDataAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerInControlAnchorFromOutData(
    std::vector<ge::InControlAnchorPtr> &peerInControlFromOutDataVec, ge::OutDataAnchorPtr outDataAnchor,
    ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(outDataAnchor);
  CHK_SMART_PTR_NULL(srcNodePtr);
  for (auto peerInControlAnchorFromData : outDataAnchor->GetPeerInControlAnchors()) {
    if (!peerInControlAnchorFromData) {
      continue;
    }
    peerInControlFromOutDataVec.push_back(peerInControlAnchorFromData);
    gRet = ge::GraphUtils::RemoveEdge(outDataAnchor, peerInControlAnchorFromData);
    if (gRet != ge::GRAPH_SUCCESS) {
      HCCL_ERROR(
          "[Get][PeerInControlAnchor]GetPeerAnchorFromOutData: remove edge between outDataAnchor[%d]"
          "of Op[%s] and peer inControlAnchor[%d] of Op[%s] failed. ret[%u]",
          outDataAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInControlAnchorFromData->GetIdx(),
          peerInControlAnchorFromData->GetOwnerNode()->GetOpDesc()->GetName().c_str(), gRet);
      return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[Get][PeerInControlAnchor]GetPeerAnchorFromOutData: remove edge between outDataAnchor[%d] \
            of Op[%s] and peer inControlAnchor[%d] of Op[%s] ok",
        outDataAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInControlAnchorFromData->GetIdx(),
        peerInControlAnchorFromData->GetOwnerNode()->GetOpDesc()->GetName().c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetPeerInControlFromOutControl(
    vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec, ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  ge::OutControlAnchorPtr outControlAnchor = srcNodePtr->GetOutControlAnchor();
  CHK_SMART_PTR_NULL(outControlAnchor);
  for (auto peerInControlAnchor : outControlAnchor->GetPeerInControlAnchors()) {
    if (!peerInControlAnchor) {
      continue;
    }
    peerInControlFromOutControlVec.push_back(peerInControlAnchor);
    gRet = ge::GraphUtils::RemoveEdge(outControlAnchor, peerInControlAnchor);
    if (gRet != ge::GRAPH_SUCCESS) {
      HCCL_ERROR(
          "[Get][PeerInControl]GetPeerInControlFromOutControl: remove edge between"
          "outControlAnchor[%d] of Op[%s] and peer inControlAnchor[%d] of Op[%s] failed. ret[%u]",
          outControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInControlAnchor->GetIdx(),
          peerInControlAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(), gRet);
      return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[Get][PeerInControl]GetPeerInControlFromOutControl: remove edge between outControlAnchor[%d] "
        "of Op[%s] and peer inControlAnchor[%d] of Op[%s] ok",
        outControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInControlAnchor->GetIdx(),
        peerInControlAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::GetReduceScatterOpInfo(s32 &rankSize, string &group, std::string &nodeName,
                                                           ge::NodePtr &srcNodePtr) {
  CHK_PRT_RET((srcNodePtr == nullptr), HCCL_ERROR("[Get][GetReduceScatterOpInfo]ReduceScatterOpInfo is nullptr"),
              HCCL_E_INTERNAL);
  bool bErr = false;
  // get rankSize
  bErr = ge::AttrUtils::GetInt(srcNodePtr->GetOpDesc(), "rank_size", rankSize);
  CHK_PRT_RET(
      !bErr,
      HCCL_ERROR("[Get][ReduceScatterOpInfo]errNo[0x%016llx] get attr rankSize failed.", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);

  // get group
  bErr = ge::AttrUtils::GetStr(srcNodePtr->GetOpDesc(), "group", group);
  CHK_PRT_RET(
      !bErr,
      HCCL_ERROR("[Get][ReduceScatterOpInfo]errNo[0x%016llx] get attr group failed.", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);

  // get nodeName
  nodeName = srcNodePtr->GetName();
  HCCL_DEBUG("[Get][ReduceScatterOpInfo]node[%s], rankSize[%d], group[%s]", nodeName.c_str(), rankSize, group.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::AddFusionNode(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                                                  ReduceScatterFusionNodesInfo &fusionNodesInfo,
                                                  ge::OpDescPtr &fusedOp) {
  CHK_RET(AddSendDataSplitV(graph, nodeInfos, fusionNodesInfo));
  CHK_RET(AddReduceScatterNode(graph, nodeInfos, fusionNodesInfo, fusedOp));

  HCCL_DEBUG("Add ReduceScatter FusionNode success");
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::AddSendDataSplitV(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                                                      ReduceScatterFusionNodesInfo &fusionNodesInfo) {
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    SplitVNodeInfo splitvNodeInfo;
    // name
    splitvNodeInfo.nodeName = nodeInfos[idx].nodeName + "_sendData_splitv";

    // input0: x
    ge::GeShape outputShape;
    ge::OutDataAnchorPtr sendDataAnchor = nodeInfos[idx].peerOutDataAnchor[0];
    CHK_SMART_PTR_NULL(sendDataAnchor);
    ge::NodePtr sendDataNodePtr = sendDataAnchor->GetOwnerNode();
    CHK_SMART_PTR_NULL(sendDataNodePtr);
    splitvNodeInfo.inputX = sendDataNodePtr->GetOpDesc()->GetOutputDesc(sendDataAnchor->GetIdx());
    ge::GeTensorDesc SizeSplit = sendDataNodePtr->GetOpDesc()->GetOutputDesc(sendDataAnchor->GetIdx());
    ge::GeShape inputShape = SizeSplit.GetShape();
    int32_t firstDim = inputShape.GetDim(0) / nodeInfos[0].rankSize;
    outputShape = inputShape;
    outputShape.SetDim(0, firstDim);
    // input1: size_splits

    ge::NodePtr size_splitNodePtr;
    std::vector<int32_t> size_split(nodeInfos[0].rankSize, firstDim);
    std::vector<int64_t> size_inputDim = {((int32_t)nodeInfos[0].rankSize)};
    std::string size_splitName = splitvNodeInfo.nodeName + "size_const";
    HcclResult ret = CreateConstNode(size_splitNodePtr, size_splitName.c_str(), size_split, size_inputDim, graph);
    CHK_PRT_RET((ret != HCCL_SUCCESS),
                HCCL_ERROR("[Add][SendDataSplitV]create node[%s] failed.", size_splitName.c_str()), HCCL_E_INTERNAL);
    splitvNodeInfo.inputSizeSplit = size_splitNodePtr->GetOpDesc()->GetOutputDesc(0);

    // input2: split_dim
    ge::NodePtr splitDimNodePtr;
    std::vector<int32_t> splitDim = {0};
    std::vector<int64_t> inputDim = {};
    std::string splitDimName = splitvNodeInfo.nodeName + "_dim_const";
    ret = CreateConstNode(splitDimNodePtr, splitDimName.c_str(), splitDim, inputDim, graph);
    CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][SendDataSplitV]create node[%s] failed.", splitDimName.c_str()),
                HCCL_E_INTERNAL);
    splitvNodeInfo.inputSplitDim = splitDimNodePtr->GetOpDesc()->GetOutputDesc(0);

    // attr: num_split
    splitvNodeInfo.numSplit = nodeInfos[0].rankSize;

    // output: y
    for (s32 i = 0; i < splitvNodeInfo.numSplit; i++) {
      ge::GeTensorDesc outputTensor = splitvNodeInfo.inputX.Clone();
      outputTensor.SetShape(outputShape);
      outputTensor.SetOriginShape(outputShape);
      splitvNodeInfo.outputY.push_back(outputTensor);
    }

    // create SplitV
    ret = CreateSplitVNode(splitvNodeInfo, graph);
    CHK_PRT_RET((ret != HCCL_SUCCESS),
                HCCL_ERROR("[Add][SendDataSplitV]create node[%s] failed.", splitvNodeInfo.nodeName.c_str()),
                HCCL_E_INTERNAL);

    // add edge
    CHK_RET(AddOpsEdge(sendDataAnchor, splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_X_INDEX)));
    CHK_RET(AddOpsEdge(size_splitNodePtr->GetOutDataAnchor(0),
                       splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_SIZESPLIT_INDEX)));
    CHK_RET(AddOpsEdge(splitDimNodePtr->GetOutDataAnchor(0),
                       splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_SPLITDIM_INDEX)));

    fusionNodesInfo.sendDataSplitVs.push_back(splitvNodeInfo.splitvNodePtr);
    HCCL_INFO("[Add][SendDataSplitV]node[%s] success.", splitvNodeInfo.nodeName.c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::AddReduceScatterNode(ge::ComputeGraph &graph,
                                                         std::vector<CommonNodeInfo> &nodeInfos,
                                                         ReduceScatterFusionNodesInfo &fusionNodesInfo,
                                                         ge::OpDescPtr &fusedOp) {
  // name
  std::string nodeName = nodeInfos[0].nodeName + "_fusion";

  // create opdesc
  CHK_PRT_RET((fusedOp == nullptr), HCCL_ERROR("[Add][ReduceScatter]node[%s] alloc desc failed", nodeName.c_str()),
              HCCL_E_INTERNAL);
  fusedOp->SetName(nodeName.c_str());

  // input: send_data
  std::vector<ge::GeShape> outputShapes;
  ge::GeShape inputShape;
  ge::GeShape OutShape;
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    ge::OutDataAnchorPtr sendDataAnchor = nodeInfos[idx].peerOutDataAnchor[0];
    ge::NodePtr sendDataNodePtr = sendDataAnchor->GetOwnerNode();
    CHK_SMART_PTR_NULL(sendDataNodePtr);
    ge::GeTensorDesc sendDataDec = sendDataNodePtr->GetOpDesc()->GetOutputDesc(sendDataAnchor->GetIdx());
    inputShape = sendDataDec.GetShape();
    int32_t firstDim = inputShape.GetDim(0) / nodeInfos[0].rankSize;
    OutShape = inputShape;
    OutShape.SetDim(0, firstDim);
    outputShapes.push_back(OutShape);
  }

  std::vector<ge::GeShape> inputShapes;
  std::vector<ge::Format> inputFormat;
  std::vector<ge::DataType> inputDataType;
  std::vector<ge::OutDataAnchorPtr> OutDataAnchor;
  CHK_PRT_RET(fusionNodesInfo.sendDataSplitVs.empty(),
              HCCL_ERROR("[Add][AddReduceScatterNode]sendDataSplitVs is empty."), HCCL_E_INTERNAL);
  uint32_t outDataAnchorSize = fusionNodesInfo.sendDataSplitVs[0]->GetAllOutDataAnchorsSize();
  for (uint32_t i = 0; i < outDataAnchorSize; i++) {
    for (ge::NodePtr &sendDataSplitNodePtr : fusionNodesInfo.sendDataSplitVs) {
      CHK_PRT_RET(outDataAnchorSize != sendDataSplitNodePtr->GetAllOutDataAnchorsSize(),
                  HCCL_ERROR("[Add][AddReduceScatterNode]sendDataSplitVs size not equal."), HCCL_E_INTERNAL);
      OutDataAnchor.push_back(sendDataSplitNodePtr->GetOutDataAnchor(i));
      ge::GeTensorDesc sendDataDesc = sendDataSplitNodePtr->GetOpDesc()->GetOutputDesc(i);
      inputShapes.push_back(sendDataDesc.GetShape());
      inputFormat.push_back(sendDataDesc.GetFormat());
      inputDataType.push_back(sendDataDesc.GetDataType());
    }
  }

  // add node and link send_data anchor
  ge::NodePtr reducescatterNodePtr = graph.AddNode(fusedOp);
  ge::NodeUtils::AppendInputAnchor(reducescatterNodePtr,
                                   ((int32_t)nodeInfos.size()) * ((int32_t)nodeInfos[0].rankSize));
  ge::NodeUtils::AppendOutputAnchor(reducescatterNodePtr, ((int32_t)nodeInfos.size()));

  for (uint32_t idx = 0; idx < nodeInfos.size() * nodeInfos[0].rankSize; idx++) {
    ge::GeTensorDescPtr ReduceScatterInDescPtr = reducescatterNodePtr->GetOpDesc()->MutableInputDesc(idx);
    ReduceScatterInDescPtr->SetShape(inputShapes[idx]);
    ReduceScatterInDescPtr->SetOriginShape(inputShapes[idx]);
    ReduceScatterInDescPtr->SetFormat(inputFormat[idx]);
    ReduceScatterInDescPtr->SetDataType(inputDataType[idx]);
    CHK_RET(AddOpsEdge(OutDataAnchor[idx], reducescatterNodePtr->GetInDataAnchor(idx)));
  }
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    ge::GeTensorDescPtr ReduceScatterOutDescPtr = reducescatterNodePtr->GetOpDesc()->MutableOutputDesc(idx);
    ReduceScatterOutDescPtr->SetShape(outputShapes[idx]);
    ReduceScatterOutDescPtr->SetOriginShape(outputShapes[idx]);
    ReduceScatterOutDescPtr->SetFormat(inputFormat[idx * nodeInfos[0].rankSize]);
    ReduceScatterOutDescPtr->SetDataType(inputDataType[idx * nodeInfos[0].rankSize]);
  }

  CHK_PRT_RET(!reducescatterNodePtr,
              HCCL_ERROR("[Add][ReduceScatter]create ReduceScatter node[%s] failed", fusedOp->GetName().c_str()),
              HCCL_E_INTERNAL);

  fusionNodesInfo.reducescatterFusionNodePtr = reducescatterNodePtr;
  HCCL_INFO("[Add][ReduceScatter]node[%s] success.", nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::RestoreOpsEdges(std::vector<CommonNodeInfo> &nodeInfos,
                                                    ReduceScatterFusionNodesInfo &fusionNodesInfo) {
  ge::graphStatus gRet;
  for (u32 nodeIndex = 0; nodeIndex < nodeInfos.size(); nodeIndex++) {
    for (ge::InDataAnchorPtr &peerIndataAnchor : nodeInfos[nodeIndex].peerInDataAnchor) {
      CHK_RET(AddOpsEdge(fusionNodesInfo.reducescatterFusionNodePtr->GetOutDataAnchor(static_cast<s32>(nodeIndex)),
                         peerIndataAnchor));
    }
    // link peerOutControlAnchor and inControlAnchor
    for (auto peerOutControlAnchor : nodeInfos[nodeIndex].peerOutControlAnchor) {
      gRet = ge::GraphUtils::AddEdge(peerOutControlAnchor,
                                     fusionNodesInfo.sendDataSplitVs[nodeIndex]->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.sendDataSplitVs[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }
    // link peerOutDataToInControl and inControlAnchor
    for (auto peerOutDataToInControl : nodeInfos[nodeIndex].peerOutDataToInControl) {
      gRet = ge::GraphUtils::AddEdge(peerOutDataToInControl,
                                     fusionNodesInfo.sendDataSplitVs[nodeIndex]->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.sendDataSplitVs[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }
    // Link outControlAnchor and peerInControlAnchor
    for (auto peerInControlAnchor : nodeInfos[nodeIndex].peerInControlAnchor) {
      gRet = ge::GraphUtils::AddEdge(fusionNodesInfo.reducescatterFusionNodePtr->GetOutControlAnchor(),
                                     peerInControlAnchor);
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add outControl edge failed.",
                             fusionNodesInfo.reducescatterFusionNodePtr->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }

    // Link outDataAnchor and peerInControlFromOutData
    for (auto peerInControlFromOutData : nodeInfos[nodeIndex].peerInControlFromOutData) {
      gRet = ge::GraphUtils::AddEdge(
          fusionNodesInfo.reducescatterFusionNodePtr->GetOutDataAnchor(static_cast<s32>(nodeIndex)),
          peerInControlFromOutData);
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add outControl edge failed.",
                             fusionNodesInfo.reducescatterFusionNodePtr->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::AddOpsEdge(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst) {
  CHK_PRT_RET((src == nullptr || dst == nullptr), HCCL_ERROR("[Add][OpsEdge]src Anchor or dst Anchor is nullptr."),
              HCCL_E_INTERNAL);
  ge::graphStatus geRet = ge::GraphUtils::AddEdge(src, dst);
  CHK_PRT_RET(geRet != ge::GRAPH_SUCCESS,
              HCCL_ERROR("[Add][OpsEdge]Failed to add edge between node[%s] and node[%s].",
                         src->GetOwnerNode()->GetName().c_str(), dst->GetOwnerNode()->GetName().c_str()),
              HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::CreateConstNode(ge::NodePtr &nodePtr, std::string nodeName,
                                                    std::vector<int32_t> nodeValue, std::vector<int64_t> dim,
                                                    ge::ComputeGraph &graph) {
  ge::GeShape shape(dim);
  ge::GeTensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);

  ge::OpDescPtr opDescPtr = nullptr;
  EXECEPTION_CATCH((opDescPtr = std::make_shared<ge::OpDesc>(nodeName.c_str(), "Const")), return HCCL_E_INTERNAL);

  ge::GeTensorPtr tensorPtr = nullptr;
  EXECEPTION_CATCH((tensorPtr = std::make_shared<ge::GeTensor>(
                        tensorDesc, reinterpret_cast<uint8_t *>(nodeValue.data()), sizeof(int32_t) * nodeValue.size())),
                   return HCCL_E_INTERNAL);
  ge::AttrUtils::SetTensor(opDescPtr, "value", tensorPtr);
  opDescPtr->AddOutputDesc(tensorPtr->GetTensorDesc());
  nodePtr = graph.AddNode(opDescPtr);
  CHK_PRT_RET(
      !nodePtr,
      HCCL_ERROR("[Create][Const]graph[%s] add node[%s] failed", graph.GetName().c_str(), opDescPtr->GetName().c_str()),
      HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterFusion::CreateSplitVNode(SplitVNodeInfo &splitvNodeInfo, ge::ComputeGraph &graph) {
  ge::graphStatus geRet = ge::GRAPH_SUCCESS;
  ge::OpDescPtr splitVOpDescPtr = nullptr;
  EXECEPTION_CATCH((splitVOpDescPtr = std::make_shared<ge::OpDesc>(splitvNodeInfo.nodeName.c_str(), "SplitV")),
                   return HCCL_E_INTERNAL);

  geRet = splitVOpDescPtr->AddInputDesc("x", splitvNodeInfo.inputX);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][SplitV]node[%s] add input: x failed", splitVOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  geRet = splitVOpDescPtr->AddInputDesc("size_splits", splitvNodeInfo.inputSizeSplit);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][SplitV]node[%s] add input: size_splits failed", splitVOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  geRet = splitVOpDescPtr->AddInputDesc("split_dim", splitvNodeInfo.inputSplitDim);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][SplitV]node[%s] add input: split_dim failed", splitVOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  for (u32 i = 0; i < splitvNodeInfo.outputY.size(); i++) {
    geRet = splitVOpDescPtr->AddOutputDesc("y" + std::to_string(i), splitvNodeInfo.outputY[i]);
    CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Create][SplitV]node[%s] add output failed", splitVOpDescPtr->GetName().c_str()),
                HCCL_E_INTERNAL);
  }

  CHK_PRT_RET(splitvNodeInfo.numSplit > SPLITV_NUMSPLIT_MAX,
              HCCL_ERROR("[Create][SplitV]node[%s] num_split[%d] is not support, 61 is the maximum of num_split",
                         splitVOpDescPtr->GetName().c_str(), splitvNodeInfo.numSplit),
              HCCL_E_NOT_SUPPORT);

  bool bErr = ge::AttrUtils::SetInt(splitVOpDescPtr, "num_split", splitvNodeInfo.numSplit);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Create][SplitV]node[%s] set attr: num_split failed", splitVOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);
  std::string dynamicOutputName = "y";
  bErr = ge::AttrUtils::SetInt(splitVOpDescPtr, DYNAMIC_OUTPUT_TD_NUM(dynamicOutputName), splitvNodeInfo.numSplit);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Create][SplitV]node[%s] set attr: dynamicOutput[%s] failed",
                         splitVOpDescPtr->GetName().c_str(), dynamicOutputName.c_str()),
              HCCL_E_INTERNAL);

  splitvNodeInfo.splitvNodePtr = graph.AddNode(splitVOpDescPtr);
  CHK_PRT_RET(!splitvNodeInfo.splitvNodePtr,
              HCCL_ERROR("[Create][SplitV]graph[%s] add node[%s] failed", graph.GetName().c_str(),
                         splitVOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  return HCCL_SUCCESS;
}
}  // namespace hccl
