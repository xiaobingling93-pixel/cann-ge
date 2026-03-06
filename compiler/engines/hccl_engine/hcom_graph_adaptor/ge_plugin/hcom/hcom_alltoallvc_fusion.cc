/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_alltoallvc_fusion.h"
#include <cmath>
#include "hcom_op_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/ge_tensor.h"
#include "graph/types.h"

using namespace hccl;
using namespace std;

namespace hccl {
HcomAlltoAllVCFusion::HcomAlltoAllVCFusion() {}

HcomAlltoAllVCFusion::~HcomAlltoAllVCFusion() {}

HcclResult HcomAlltoAllVCFusion::Run(ge::ComputeGraph &graph) {
  FusionInfos fusionInfos;
  HcclResult ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][AlltoAllVCFusion]graph[%s]: get fusion HcomAlltoAllVC ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);
  HCCL_INFO("there are %u group to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of  HcomAlltoAllVC operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0), HCCL_INFO("NOT_CHANGED: the graph has no HcomAlltoAllVC op."), HCCL_SUCCESS);

  for (auto iterFusionInfos = fusionInfos.begin(); iterFusionInfos != fusionInfos.end(); iterFusionInfos++) {
    HCCL_INFO("graph[%s] fusionlabel[%s]: there are %zu HcomAlltoAllVC ops to be fused.", graph.GetName().c_str(),
              iterFusionInfos->first.c_str(), iterFusionInfos->second.size());

    ret = FuseOps(graph, iterFusionInfos->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] fusionlabel[%s]: fusion HcomAlltoAllVC ops failed. "
                           "ret[%d]",
                           graph.GetName().c_str(), iterFusionInfos->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("fuse HcomAlltoAllVC op end");
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos) {
  HcclResult ret;
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("HcomAlltoAllVCFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomAlltoAllVCFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }

    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
      ret = GetFusionOpInfo(nodePtr, fusionInfos);
      CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("get fusion ops by group failed. ret[%d]", ret), ret);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos) {
  bool bUnknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, bUnknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
              HCCL_E_PARA);
  CHK_PRT_RET(!bUnknownShapeNode, HCCL_INFO("node[%s] is known shape, no fusion", nodePtr->GetName().c_str()),
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

HcclResult HcomAlltoAllVCFusion::GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
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
                                std::vector<const char *>({"HcomAlltoAllVCFusion", "fusion", fusionValue.c_str(),
                                                           "please check fusion setting"}));
      HCCL_ERROR("[%s][%s]errNo[0x%016llx] node[%s] fusion[%lld] is incorrect, should be %lld or %lld",
                 LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA),
                 nodeName.c_str(), fusionOption.fusionAttr, HCOM_ATTR_FUSION_NO_FUSION, HCOM_ATTR_FUSION_BY_FUSION_ID);
      return HCCL_E_PARA;
  }

  // 获取comm和group
  CHK_RET(GetCommFromOpDescPtr(nodePtr, fusionOption));
  CHK_RET(HcomOpUtils::GetDataType(nodePtr->GetOpDesc(), fusionOption.dtype));

  fusionOption.optype = nodePtr->GetOpDesc()->GetType();
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel) {
  if (fusionOption.fusionComm == static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
    fusionLabel = fusionOption.optype + "-" + fusionOption.group + "-" + to_string(fusionOption.fusionId) + "-" +
                  fusionOption.dtype;
  } else {
    char *group = nullptr;
    CHK_RET(GetGroupNameByOpBaseHcom(fusionOption.fusionComm, &group));
    std::string identifier = std::string(group);
    fusionLabel =
        fusionOption.optype + "-" + identifier + "-" + to_string(fusionOption.fusionId) + "-" + fusionOption.dtype;
    HCCL_DEBUG("[HcclCommGraph][Type]GenerateFusionLabel.");
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  CHK_PRT_RET((fusionSection.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %u HcomAlltoAllVC op.", fusionSection.size()), HCCL_SUCCESS);

  HcclResult ret = RunFusionOpsAlltoAllVC(graph, fusionSection);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);

  HCCL_INFO("graph[%s] fuse HcomAlltoAllVC op end.", graph.GetName().c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::RunFusionOpsAlltoAllVC(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps) {
  std::vector<AlltoAllVCNodeInfo> nodeInfos(0);
  ge::OpDescPtr fusedOp;
  AlltoAllVCFusionNodesInfo fusionNodesInfo;
  CHK_RET(RemoveOpsEdges(graph, fusionOps, nodeInfos, fusedOp));
  CHK_RET(CheckAlltoAllVCNodeInfo(nodeInfos));
  CHK_RET(AddFusionNode(graph, nodeInfos, fusionNodesInfo, fusedOp));
  CHK_RET(RestoreOpsEdges(nodeInfos, fusionNodesInfo));
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
                                                std::vector<AlltoAllVCNodeInfo> &nodeInfos, ge::OpDescPtr &fusedOp) {
  ge::graphStatus gRet;
  ge::OpDescPtr originDescPtr = fusionOps[0]->GetOpDesc();
  CHK_SMART_PTR_NULL(originDescPtr);
  fusedOp = ge::AttrUtils::CopyOpDesc(originDescPtr);
  CHK_SMART_PTR_NULL(fusedOp);

  for (uint32_t idx = 0; idx < fusionOps.size(); idx++) {
    AlltoAllVCNodeInfo nodeInfo;
    // get all anchors connected to the node, and remove the edges between them.
    CHK_RET(GetPeerOutDataToInData(nodeInfo.peerOutDataAnchor, fusionOps[idx]));
    CHK_RET(GetPeerOutDataToInControl(nodeInfo.peerOutDataToInControl, fusionOps[idx]));
    CHK_RET(GetPeerOutControlToInControl(nodeInfo.peerOutControlAnchor, fusionOps[idx]));
    CHK_RET(GetPeerAnchorFromOutData(nodeInfo.peerInDataAnchor, nodeInfo.peerInControlFromOutData, fusionOps[idx]));
    CHK_RET(GetPeerInControlFromOutControl(nodeInfo.peerInControlAnchor, fusionOps[idx]));
    CHK_RET(GetAlltoAllVCOpInfo(nodeInfo.rank, nodeInfo.rankSize, nodeInfo.group, nodeInfo.nodeName, fusionOps[idx]));
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

HcclResult HcomAlltoAllVCFusion::GetPeerOutDataToInData(std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
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
  CHK_PRT_RET(peerOutDataAnchorVec.size() != ALLTOALLVC_INPUT_NUM,
              HCCL_ERROR("[Check][NodeInfo]node[%s] peerOutDataAnchor size is %u, expect: %u.", srcOpName.c_str(),
                         peerOutDataAnchorVec.size(), ALLTOALLVC_INPUT_NUM),
              HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::GetPeerOutDataToInControl(vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
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

HcclResult HcomAlltoAllVCFusion::GetPeerOutControlToInControl(
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

HcclResult HcomAlltoAllVCFusion::GetPeerAnchorFromOutData(
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

HcclResult HcomAlltoAllVCFusion::GetPeerInDataAnchorFromOutData(vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
                                                                ge::OutDataAnchorPtr outDataAnchor,
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

HcclResult HcomAlltoAllVCFusion::GetPeerInControlAnchorFromOutData(
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

HcclResult HcomAlltoAllVCFusion::GetPeerInControlFromOutControl(
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

HcclResult HcomAlltoAllVCFusion::GetAlltoAllVCOpInfo(s32 &rank, s32 &rankSize, string &group, std::string &nodeName,
                                                     ge::NodePtr &srcNodePtr) {
  bool bErr = false;
  // get rank
  bErr = ge::AttrUtils::GetInt(srcNodePtr->GetOpDesc(), "rank", rank);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Get][AlltoAllVCOpInfo]errNo[0x%016llx] get attr rank failed.", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  // get rankSize
  auto peerIndataTensor = srcNodePtr->GetOpDesc()->GetInputDesc(1);
  rankSize = peerIndataTensor.GetShape().GetDim(0);

  // get group
  bErr = ge::AttrUtils::GetStr(srcNodePtr->GetOpDesc(), "group", group);
  CHK_PRT_RET(
      !bErr, HCCL_ERROR("[Get][AlltoAllVCOpInfo]errNo[0x%016llx] get attr group failed.", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);

  // get nodeName
  nodeName = srcNodePtr->GetName();
  HCCL_DEBUG("[Get][AlltoAllVCOpInfo]node[%s], rank[%d], rankSize[%d], group[%s]", nodeName.c_str(), rank, rankSize,
             group.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddFusionNode(ge::ComputeGraph &graph, std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                                               AlltoAllVCFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp) {
  CHK_PRT_RET((nodeInfos.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %u HcomAlltoAllVC node.", nodeInfos.size()), HCCL_SUCCESS);
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    CHK_RET(AddSendCountSplit(graph, nodeInfos[idx], fusionNodesInfo));
    CHK_RET(AddSendDataSplitV(graph, nodeInfos[idx], fusionNodesInfo, fusionNodesInfo.sendCountSplits[idx]));
    CHK_RET(AddRecvCountSplit(graph, nodeInfos[idx], fusionNodesInfo));
  }
  CHK_RET(AddSendDataConCat(graph, nodeInfos[0], fusionNodesInfo));
  CHK_RET(AddAddN(graph, nodeInfos, fusionNodesInfo));
  CHK_RET(AddAlltoAllVCNode(graph, nodeInfos[0], fusionNodesInfo, fusedOp));
  CHK_RET(AddRecvCountConCat(graph, nodeInfos[0], fusionNodesInfo));
  CHK_RET(AddRecvDataSplitV(graph, nodeInfos[0], fusionNodesInfo));
  CHK_RET(AddRecvDataConCat(graph, nodeInfos, fusionNodesInfo));

  std::vector<string> depInputs = {"send_count_matrix"};
  fusionNodesInfo.alltoallvcFusionNodePtr->GetOpDesc()->SetOpInferDepends(depInputs);

  HCCL_DEBUG("Add AlltoAllVC FusionNode success");
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddSendCountSplit(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  SplitNodeInfo splitNodeInfo;
  // name
  splitNodeInfo.nodeName = nodeInfo.nodeName + "_sendCount_split";

  // input0: split_dim
  ge::NodePtr splitDimNodePtr;
  std::vector<int32_t> splitDim = {0};
  std::vector<int64_t> inputDim = {};
  std::string splitDimName = splitNodeInfo.nodeName + "_dim_const";
  HcclResult ret = CreateConstNode(splitDimNodePtr, splitDimName.c_str(), splitDim, inputDim, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][SendCountSplit]create node[%s] failed.", splitDimName.c_str()),
              HCCL_E_INTERNAL);
  splitNodeInfo.inputSplitDim = splitDimNodePtr->GetOpDesc()->GetOutputDesc(0);

  // input1: x
  CHK_PRT_RET((nodeInfo.peerOutDataAnchor.size() != ALLTOALLVC_INPUT_NUM),
              HCCL_ERROR("[Add][SendCountSplit]get sendCountMatrixAnchor failed."), HCCL_E_INTERNAL);
  ge::OutDataAnchorPtr sendCountMatrixAnchor = nodeInfo.peerOutDataAnchor[1];
  CHK_SMART_PTR_NULL(sendCountMatrixAnchor);
  ge::NodePtr sendCountMatrixNodePtr = sendCountMatrixAnchor->GetOwnerNode();
  CHK_SMART_PTR_NULL(sendCountMatrixNodePtr);
  splitNodeInfo.inputX = sendCountMatrixNodePtr->GetOpDesc()->GetOutputDesc(sendCountMatrixAnchor->GetIdx());

  // output: y
  for (s32 i = 0; i < nodeInfo.rankSize; i++) {
    ge::GeTensorDesc outputTensor = splitNodeInfo.inputX.Clone();
    std::vector<int64_t> dims = {nodeInfo.rankSize};
    outputTensor.SetShape(ge::GeShape(dims));
    outputTensor.SetOriginShape(ge::GeShape(dims));
    splitNodeInfo.outputY.push_back(outputTensor);
  }

  // attr: num_split
  splitNodeInfo.numSplit = nodeInfo.rankSize;

  // create Split
  ret = CreateSplitNode(splitNodeInfo, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][SendCountSplit]create node[%s] failed.", splitNodeInfo.nodeName.c_str()),
              HCCL_E_INTERNAL);

  // link peerSendCountMatrixOutDataAnchor to sendCountSplitAnchor
  CHK_RET(AddOpsEdge(splitDimNodePtr->GetOutDataAnchor(0), splitNodeInfo.splitNodePtr->GetInDataAnchor(0)));
  CHK_RET(AddOpsEdge(sendCountMatrixAnchor, splitNodeInfo.splitNodePtr->GetInDataAnchor(1)));

  fusionNodesInfo.sendCountSplits.push_back(splitNodeInfo.splitNodePtr);
  HCCL_INFO("[Add][SendCountSplit]node[%s] success.", splitNodeInfo.nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddSendDataSplitV(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo,
                                                   ge::NodePtr &peerOutDataToSizeSplitNodePtr) {
  SplitVNodeInfo splitvNodeInfo;
  // name
  splitvNodeInfo.nodeName = nodeInfo.nodeName + "_sendData_splitv";

  // input0: x
  ge::OutDataAnchorPtr sendDataAnchor = nodeInfo.peerOutDataAnchor[0];
  CHK_SMART_PTR_NULL(sendDataAnchor);
  ge::NodePtr sendDataNodePtr = sendDataAnchor->GetOwnerNode();
  CHK_SMART_PTR_NULL(sendDataNodePtr);
  splitvNodeInfo.inputX = sendDataNodePtr->GetOpDesc()->GetOutputDesc(sendDataAnchor->GetIdx());

  // input1: size_splits
  ge::OutDataAnchorPtr sendCountAnchor = peerOutDataToSizeSplitNodePtr->GetOutDataAnchor(nodeInfo.rank);
  CHK_SMART_PTR_NULL(sendCountAnchor);
  splitvNodeInfo.inputSizeSplit =
      peerOutDataToSizeSplitNodePtr->GetOpDesc()->GetOutputDesc(static_cast<u32>(nodeInfo.rank));

  // input2: split_dim
  ge::NodePtr splitDimNodePtr;
  std::vector<int32_t> splitDim = {0};
  std::vector<int64_t> inputDim = {};
  std::string splitDimName = splitvNodeInfo.nodeName + "_dim_const";
  HcclResult ret = CreateConstNode(splitDimNodePtr, splitDimName.c_str(), splitDim, inputDim, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][SendDataSplitV]create node[%s] failed.", splitDimName.c_str()),
              HCCL_E_INTERNAL);
  splitvNodeInfo.inputSplitDim = splitDimNodePtr->GetOpDesc()->GetOutputDesc(0);

  // attr: num_split
  splitvNodeInfo.numSplit = nodeInfo.rankSize;

  // output: y
  for (s32 i = 0; i < splitvNodeInfo.numSplit; i++) {
    ge::GeTensorDesc outputTensor = splitvNodeInfo.inputX.Clone();
    std::vector<int64_t> dims = {-1};
    outputTensor.SetShape(ge::GeShape(dims));
    outputTensor.SetOriginShape(ge::GeShape(dims));
    splitvNodeInfo.outputY.push_back(outputTensor);
  }

  // create SplitV
  ret = CreateSplitVNode(splitvNodeInfo, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][SendDataSplitV]create node[%s] failed.", splitvNodeInfo.nodeName.c_str()),
              HCCL_E_INTERNAL);

  // add edge
  CHK_RET(AddOpsEdge(sendDataAnchor, splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_X_INDEX)));
  CHK_RET(AddOpsEdge(sendCountAnchor, splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_SIZESPLIT_INDEX)));
  CHK_RET(AddOpsEdge(splitDimNodePtr->GetOutDataAnchor(0),
                     splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_SPLITDIM_INDEX)));

  fusionNodesInfo.sendDataSplitVs.push_back(splitvNodeInfo.splitvNodePtr);
  HCCL_INFO("[Add][SendDataSplitV]node[%s] success.", splitvNodeInfo.nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddRecvCountSplit(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  // 第一级 send_count_matrix split: n*n -> n*1
  SplitNodeInfo splitNodeInfo0;
  // name
  splitNodeInfo0.nodeName = nodeInfo.nodeName + "_recvCount0_split";

  // input0: split_dim
  ge::NodePtr splitDimNodePtr0;
  std::vector<int32_t> splitDim0 = {1};
  std::vector<int64_t> inputDim0 = {};
  std::string splitDimName0 = splitNodeInfo0.nodeName + "_dim_const";
  HcclResult ret = CreateConstNode(splitDimNodePtr0, splitDimName0.c_str(), splitDim0, inputDim0, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][RecvCountSplit]create node[%s] failed.", splitDimName0.c_str()),
              HCCL_E_INTERNAL);
  splitNodeInfo0.inputSplitDim = splitDimNodePtr0->GetOpDesc()->GetOutputDesc(0);

  // input1: x
  ge::OutDataAnchorPtr sendCountMatrixAnchor = nodeInfo.peerOutDataAnchor[1];
  CHK_SMART_PTR_NULL(sendCountMatrixAnchor);
  ge::NodePtr sendCountMatrixNodePtr = sendCountMatrixAnchor->GetOwnerNode();
  CHK_SMART_PTR_NULL(sendCountMatrixNodePtr);
  splitNodeInfo0.inputX = sendCountMatrixNodePtr->GetOpDesc()->GetOutputDesc(sendCountMatrixAnchor->GetIdx());

  // output: y
  for (s32 i = 0; i < nodeInfo.rankSize; i++) {
    ge::GeTensorDesc outputTensor = splitNodeInfo0.inputX.Clone();
    std::vector<int64_t> dims = {nodeInfo.rankSize, 1};
    outputTensor.SetShape(ge::GeShape(dims));
    outputTensor.SetOriginShape(ge::GeShape(dims));
    splitNodeInfo0.outputY.push_back(outputTensor);
  }

  // attr: num_split
  splitNodeInfo0.numSplit = nodeInfo.rankSize;

  // create Split level0
  ret = CreateSplitNode(splitNodeInfo0, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][RecvCountSplit]create node[%s] failed.", splitNodeInfo0.nodeName.c_str()),
              HCCL_E_INTERNAL);

  // link peerSendCountMatrixOutDataAnchor to recvCountSplitAnchor0
  CHK_RET(AddOpsEdge(splitDimNodePtr0->GetOutDataAnchor(0), splitNodeInfo0.splitNodePtr->GetInDataAnchor(0)));
  CHK_RET(AddOpsEdge(sendCountMatrixAnchor, splitNodeInfo0.splitNodePtr->GetInDataAnchor(1)));

  fusionNodesInfo.recvCountSplits0.push_back(splitNodeInfo0.splitNodePtr);
  HCCL_INFO("[Add][RecvCountSplit]node[%s] success.", splitNodeInfo0.nodeName.c_str());

  // 第二级 send_count_matrix split level1: n*1 -> 1*1
  SplitNodeInfo splitNodeInfo1;
  splitNodeInfo1.nodeName = nodeInfo.nodeName + "_recvCount1_split";

  // input0
  ge::NodePtr splitDimNodePtr1;
  std::vector<int32_t> splitDim1 = {0};
  std::vector<int64_t> inputDim1 = {};
  std::string splitDimName1 = splitNodeInfo1.nodeName + "_dim_const";
  ret = CreateConstNode(splitDimNodePtr1, splitDimName1.c_str(), splitDim1, inputDim1, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][RecvCountSplit]create node[%s] failed.", splitDimName1.c_str()),
              HCCL_E_INTERNAL);
  splitNodeInfo1.inputSplitDim = splitDimNodePtr1->GetOpDesc()->GetOutputDesc(0);

  // input1: x
  splitNodeInfo1.inputX = splitNodeInfo0.splitNodePtr->GetOpDesc()->GetOutputDesc(nodeInfo.rank);

  // output: y
  for (s32 i = 0; i < nodeInfo.rankSize; i++) {
    ge::GeTensorDesc outputTensor = splitNodeInfo1.inputX.Clone();
    std::vector<int64_t> dims = {1};
    outputTensor.SetShape(ge::GeShape(dims));
    outputTensor.SetOriginShape(ge::GeShape(dims));
    splitNodeInfo1.outputY.push_back(outputTensor);
  }

  // attr: num_split
  splitNodeInfo1.numSplit = nodeInfo.rankSize;

  // create Split level1
  ret = CreateSplitNode(splitNodeInfo1, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][RecvCountSplit]create node[%s] failed.", splitNodeInfo1.nodeName.c_str()),
              HCCL_E_INTERNAL);

  // link peerRecvCountOutDataAnchor to recvCountSplitAnchor1
  CHK_RET(AddOpsEdge(splitDimNodePtr1->GetOutDataAnchor(0), splitNodeInfo1.splitNodePtr->GetInDataAnchor(0)));
  CHK_RET(AddOpsEdge(splitNodeInfo0.splitNodePtr->GetOutDataAnchor(nodeInfo.rank),
                     splitNodeInfo1.splitNodePtr->GetInDataAnchor(1)));

  fusionNodesInfo.recvCountSplits1.push_back(splitNodeInfo1.splitNodePtr);
  HCCL_INFO("[Add][RecvCountSplit]node[%s] success.", splitNodeInfo1.nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddSendDataConCat(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  ConcatNodeInfo concatNodeInfo;
  // name
  concatNodeInfo.nodeName = nodeInfo.nodeName + "_sendData_concat";

  // input0: concat_dim
  ge::NodePtr concatDimNodePtr;
  std::vector<int32_t> concatDim = {0};
  std::vector<int64_t> inputDim = {1};
  std::string concatDimName = concatNodeInfo.nodeName + "_dim_const";
  HcclResult ret = CreateConstNode(concatDimNodePtr, concatDimName.c_str(), concatDim, inputDim, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][SendDataConCat]create node[%s] failed.", concatDimName.c_str()),
              HCCL_E_INTERNAL);
  concatNodeInfo.inputConcatDim = concatDimNodePtr->GetOpDesc()->GetOutputDesc(0);

  // input1: x
  std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
  CHK_PRT_RET(fusionNodesInfo.sendDataSplitVs.empty(), HCCL_ERROR("[Add][SendDataConCat]sendDataSplitVs is empty."),
              HCCL_E_INTERNAL);
  uint32_t outDataAnchorSize = fusionNodesInfo.sendDataSplitVs[0]->GetAllOutDataAnchorsSize();
  for (uint32_t i = 0; i < outDataAnchorSize; i++) {
    for (ge::NodePtr &sendDataSplitNodePtr : fusionNodesInfo.sendDataSplitVs) {
      CHK_PRT_RET(outDataAnchorSize != sendDataSplitNodePtr->GetAllOutDataAnchorsSize(),
                  HCCL_ERROR("[Add][SendDataConCat]sendDataSplitVs size not equal."), HCCL_E_INTERNAL);
      concatNodeInfo.inputX.push_back(sendDataSplitNodePtr->GetOpDesc()->GetOutputDesc(i));
      peerOutDataAnchor.push_back(sendDataSplitNodePtr->GetOutDataAnchor(i));
    }
  }

  // attr: N
  concatNodeInfo.N = static_cast<s32>(concatNodeInfo.inputX.size());

  // output: y
  ge::GeTensorDesc outputTensor = concatNodeInfo.inputX[0].Clone();
  std::vector<int64_t> dims = {-1};
  outputTensor.SetShape(ge::GeShape(dims));
  outputTensor.SetOriginShape(ge::GeShape(dims));
  concatNodeInfo.outputY = outputTensor;

  // create Concat
  ret = CreateConcatNode(concatNodeInfo, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][SendDataConCat]create node[%s] failed.", concatNodeInfo.nodeName.c_str()),
              HCCL_E_INTERNAL);

  // link peerSendDataOutDataAnchor to concatInputAnchor
  CHK_RET(AddOpsEdge(concatDimNodePtr->GetOutDataAnchor(0), concatNodeInfo.concatNodePtr->GetInDataAnchor(0)));
  for (uint32_t idx = 0; idx < peerOutDataAnchor.size(); idx++) {
    CHK_RET(AddOpsEdge(peerOutDataAnchor[idx], concatNodeInfo.concatNodePtr->GetInDataAnchor(idx + 1)));
  }

  fusionNodesInfo.sendDataConcatNodePtr = concatNodeInfo.concatNodePtr;
  HCCL_INFO("[Add][SendDataConCat]node[%s] success.", concatNodeInfo.nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddAddN(ge::ComputeGraph &graph, std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                                         AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  CHK_PRT_RET(nodeInfos.empty(), HCCL_ERROR("[Add][AddN]nodeInfos is empty."), HCCL_E_INTERNAL);
  // name
  std::string nodeName = nodeInfos[0].nodeName + "_sendCountMatrix_AddN";

  // create AddN opdesc
  ge::graphStatus geRet = ge::GRAPH_SUCCESS;
  ge::OpDescPtr addnOpDescPtr = nullptr;
  EXECEPTION_CATCH((addnOpDescPtr = std::make_shared<ge::OpDesc>(nodeName.c_str(), "AddN")), return HCCL_E_INTERNAL);

  // input0: x
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    ge::OutDataAnchorPtr sendCountMatrixAnchor = nodeInfos[idx].peerOutDataAnchor[1];
    ge::NodePtr sendCountMatrixNodePtr = sendCountMatrixAnchor->GetOwnerNode();
    ge::GeTensorDesc inputX = sendCountMatrixNodePtr->GetOpDesc()->GetOutputDesc(sendCountMatrixAnchor->GetIdx());

    geRet = addnOpDescPtr->AddInputDesc("x" + std::to_string(idx), inputX);
    CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Add][AddN]node[%s] add input x failed", addnOpDescPtr->GetName().c_str()),
                HCCL_E_INTERNAL);
  }

  bool bErr = ge::AttrUtils::SetInt(addnOpDescPtr, "N", static_cast<s32>(nodeInfos.size()));
  CHK_PRT_RET(!bErr, HCCL_ERROR("[Add][AddN]node[%s] set attr N failed", addnOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  std::string dynamicInputName = "x";
  bErr = ge::AttrUtils::SetInt(addnOpDescPtr, DYNAMIC_INPUT_TD_NUM(dynamicInputName),
                               static_cast<int64_t>(nodeInfos.size()));
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Add][AddN]node[%s] set attr: dynamicInput[%s] failed", addnOpDescPtr->GetName().c_str(),
                         dynamicInputName.c_str()),
              HCCL_E_INTERNAL);

  // output: y
  ge::GeTensorDesc outputY = addnOpDescPtr->GetInputDesc(0).Clone();
  geRet = addnOpDescPtr->AddOutputDesc("y", outputY);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Add][AddN]node[%s] add output failed", addnOpDescPtr->GetName().c_str()), HCCL_E_INTERNAL);

  // add Node
  ge::NodePtr addnNodePtr = graph.AddNode(addnOpDescPtr);
  CHK_PRT_RET(!addnNodePtr, HCCL_ERROR("[Add][AddN]create AddN node failed"), HCCL_E_INTERNAL);
  CHK_RET(SetUnknownShape(addnNodePtr, graph));

  // link peerOutDataAnchor to InDataAnchor
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    ge::OutDataAnchorPtr sendCountMatrixAnchor = nodeInfos[idx].peerOutDataAnchor[1];
    CHK_RET(AddOpsEdge(sendCountMatrixAnchor, addnNodePtr->GetInDataAnchor(idx)));
  }

  fusionNodesInfo.sendCountMatrixAddNNodePtr = addnNodePtr;
  HCCL_INFO("[Add][AddN]node[%s] success.", nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddAlltoAllVCNode(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp) {
  // name
  std::string nodeName = nodeInfo.nodeName + "_fusion";

  // create opdesc
  CHK_PRT_RET((fusedOp == nullptr), HCCL_ERROR("[Add][AlltoAllVC]node[%s] alloc desc failed", nodeName.c_str()),
              HCCL_E_INTERNAL);
  fusedOp->SetName(nodeName.c_str());

  // input0: send_data
  ge::OutDataAnchorPtr sendDataConcatOutDataAnchor = fusionNodesInfo.sendDataConcatNodePtr->GetOutDataAnchor(0);

  // input1: send_count_matrix
  ge::OutDataAnchorPtr sendCountAddNOutDataAnchor = fusionNodesInfo.sendCountMatrixAddNNodePtr->GetOutDataAnchor(0);

  // add node
  ge::NodePtr alltoallvcNodePtr = graph.AddNode(fusedOp);
  CHK_PRT_RET(!alltoallvcNodePtr,
              HCCL_ERROR("[Add][AlltoAllVC]create AllToAllVC node[%s] failed", fusedOp->GetName().c_str()),
              HCCL_E_INTERNAL);
  CHK_RET(SetUnknownShape(alltoallvcNodePtr, graph));

  // link send_data anchor and send_count_matrix anchor
  CHK_RET(AddOpsEdge(sendDataConcatOutDataAnchor, alltoallvcNodePtr->GetInDataAnchor(0)));
  CHK_RET(AddOpsEdge(sendCountAddNOutDataAnchor, alltoallvcNodePtr->GetInDataAnchor(1)));

  fusionNodesInfo.alltoallvcFusionNodePtr = alltoallvcNodePtr;
  HCCL_INFO("[Add][AlltoAllVC]node[%s] success.", nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddRecvCountConCat(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                    AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  ConcatNodeInfo concatNodeInfo;
  // name
  concatNodeInfo.nodeName = nodeInfo.nodeName + "_recvCount_concat";

  // input0: concat_dim
  ge::NodePtr concatDimNodePtr;
  std::vector<int32_t> concatDim = {0};
  std::vector<int64_t> inputDim = {1};
  std::string concatDimName = concatNodeInfo.nodeName + "_dim_const";
  HcclResult ret = CreateConstNode(concatDimNodePtr, concatDimName.c_str(), concatDim, inputDim, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][RecvCountConCat]create node[%s] failed.", concatDimName.c_str()),
              HCCL_E_INTERNAL);
  concatNodeInfo.inputConcatDim = concatDimNodePtr->GetOpDesc()->GetOutputDesc(0);

  // input1: x
  std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
  uint32_t rankSize = fusionNodesInfo.recvCountSplits1[0]->GetAllOutDataAnchorsSize();
  for (uint32_t i = 0; i < rankSize; i++) {
    for (ge::NodePtr &recvCountSplitNodePtr : fusionNodesInfo.recvCountSplits1) {
      CHK_SMART_PTR_NULL(recvCountSplitNodePtr);
      concatNodeInfo.inputX.push_back(recvCountSplitNodePtr->GetOpDesc()->GetOutputDesc(i));
      peerOutDataAnchor.push_back(recvCountSplitNodePtr->GetOutDataAnchor(i));
    }
  }

  // attr: N
  concatNodeInfo.N = static_cast<s32>(concatNodeInfo.inputX.size());

  // output: y
  ge::GeTensorDesc outputTensor = concatNodeInfo.inputX[0].Clone();
  std::vector<int64_t> dims = {concatNodeInfo.N};
  outputTensor.SetShape(ge::GeShape(dims));
  outputTensor.SetOriginShape(ge::GeShape(dims));
  concatNodeInfo.outputY = outputTensor;

  // create Concat
  ret = CreateConcatNode(concatNodeInfo, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][RecvCountConCat]create node[%s] failed.", concatNodeInfo.nodeName.c_str()),
              HCCL_E_INTERNAL);

  // link peerRecvCountOutDataAnchor to concatInputAnchor
  CHK_RET(AddOpsEdge(concatDimNodePtr->GetOutDataAnchor(0), concatNodeInfo.concatNodePtr->GetInDataAnchor(0)));
  for (uint32_t idx = 0; idx < peerOutDataAnchor.size(); idx++) {
    CHK_RET(AddOpsEdge(peerOutDataAnchor[idx], concatNodeInfo.concatNodePtr->GetInDataAnchor(idx + 1)));
  }

  fusionNodesInfo.recvCountConcatNodePtr = concatNodeInfo.concatNodePtr;
  HCCL_INFO("[Add][AlltoAllVC]node[%s] success.", concatNodeInfo.nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddRecvDataSplitV(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  SplitVNodeInfo splitvNodeInfo;
  // name
  splitvNodeInfo.nodeName = nodeInfo.nodeName + "_recvData_splitv";

  // input0: x
  ge::OutDataAnchorPtr recvDataAnchor = fusionNodesInfo.alltoallvcFusionNodePtr->GetOutDataAnchor(0);
  CHK_SMART_PTR_NULL(recvDataAnchor);
  splitvNodeInfo.inputX = fusionNodesInfo.alltoallvcFusionNodePtr->GetOpDesc()->GetOutputDesc(0);

  // input1: size_splits
  ge::OutDataAnchorPtr recvCountAnchor = fusionNodesInfo.recvCountConcatNodePtr->GetOutDataAnchor(0);
  CHK_SMART_PTR_NULL(recvCountAnchor);
  splitvNodeInfo.inputSizeSplit = fusionNodesInfo.recvCountConcatNodePtr->GetOpDesc()->GetOutputDesc(0);

  // input2: split_dim
  ge::NodePtr splitDimNodePtr;
  std::vector<int32_t> splitDim = {0};
  std::vector<int64_t> inputDim = {};
  std::string splitDimName = splitvNodeInfo.nodeName + "_dim_const";
  HcclResult ret = CreateConstNode(splitDimNodePtr, splitDimName.c_str(), splitDim, inputDim, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[Add][RecvDataSplitV]create node[%s] failed.", splitDimName.c_str()),
              HCCL_E_INTERNAL);
  splitvNodeInfo.inputSplitDim = splitDimNodePtr->GetOpDesc()->GetOutputDesc(0);

  // attr: num_split
  splitvNodeInfo.numSplit = splitvNodeInfo.inputSizeSplit.GetShape().GetDim(0);

  // output: y
  for (s32 i = 0; i < splitvNodeInfo.numSplit; i++) {
    auto outputTensor = splitvNodeInfo.inputX.Clone();
    std::vector<int64_t> dims = {-1};
    outputTensor.SetShape(ge::GeShape(dims));
    outputTensor.SetOriginShape(ge::GeShape(dims));
    splitvNodeInfo.outputY.push_back(outputTensor);
  }

  // create SplitV
  ret = CreateSplitVNode(splitvNodeInfo, graph);
  CHK_PRT_RET((ret != HCCL_SUCCESS),
              HCCL_ERROR("[Add][SendDataSplitV]create node[%s] failed.", splitvNodeInfo.nodeName.c_str()),
              HCCL_E_INTERNAL);

  CHK_RET(AddOpsEdge(recvDataAnchor, splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_X_INDEX)));
  CHK_RET(AddOpsEdge(recvCountAnchor, splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_SIZESPLIT_INDEX)));
  CHK_RET(AddOpsEdge(splitDimNodePtr->GetOutDataAnchor(0),
                     splitvNodeInfo.splitvNodePtr->GetInDataAnchor(SPLITV_INPUT_SPLITDIM_INDEX)));

  fusionNodesInfo.recvDataSplitVNodePtr = splitvNodeInfo.splitvNodePtr;
  HCCL_INFO("[Add][SendDataSplitV]node[%s] success.", splitvNodeInfo.nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::AddRecvDataConCat(ge::ComputeGraph &graph, std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                                                   AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  ge::NodePtr peerOutDataNodePtr = fusionNodesInfo.recvDataSplitVNodePtr;
  uint32_t peerOutDataAnchorsSize = peerOutDataNodePtr->GetAllOutDataAnchorsSize();

  for (uint32_t nodeIndex = 0; nodeIndex < nodeInfos.size(); nodeIndex++) {
    ConcatNodeInfo concatNodeInfo;
    // name
    concatNodeInfo.nodeName = nodeInfos[nodeIndex].nodeName + "_recvData_concat";

    // input0: concat_dim
    ge::NodePtr concatDimNodePtr;
    std::vector<int32_t> concatDim = {0};
    std::vector<int64_t> inputDim = {1};
    std::string concatDimName = concatNodeInfo.nodeName + "_dim_const";
    HcclResult ret = CreateConstNode(concatDimNodePtr, concatDimName.c_str(), concatDim, inputDim, graph);
    CHK_PRT_RET((ret != HCCL_SUCCESS),
                HCCL_ERROR("[Add][RecvDataConCat]create node[%s] failed.", concatDimName.c_str()), HCCL_E_INTERNAL);
    concatNodeInfo.inputConcatDim = concatDimNodePtr->GetOpDesc()->GetOutputDesc(0);

    // input1: x
    std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
    for (uint32_t idx = nodeIndex; idx < peerOutDataAnchorsSize; idx += nodeInfos.size()) {
      concatNodeInfo.inputX.push_back(peerOutDataNodePtr->GetOpDesc()->GetOutputDesc(idx));
      peerOutDataAnchor.push_back(peerOutDataNodePtr->GetOutDataAnchor(static_cast<s32>(idx)));
    }

    // attr: N
    concatNodeInfo.N = static_cast<s32>(concatNodeInfo.inputX.size());

    // output: y
    ge::InDataAnchorPtr peerIndataAnchor0 = nodeInfos[nodeIndex].peerInDataAnchor[0];
    CHK_SMART_PTR_NULL(peerIndataAnchor0);
    ge::GeTensorDesc outputTensor =
        peerIndataAnchor0->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peerIndataAnchor0->GetIdx());
    std::vector<int64_t> dims = {-1};
    outputTensor.SetShape(ge::GeShape(dims));
    outputTensor.SetOriginShape(ge::GeShape(dims));
    concatNodeInfo.outputY = outputTensor;

    // create Concat
    ret = CreateConcatNode(concatNodeInfo, graph);
    CHK_PRT_RET((ret != HCCL_SUCCESS),
                HCCL_ERROR("[Add][RecvDataConCat]create node[%s] failed.", concatNodeInfo.nodeName.c_str()),
                HCCL_E_INTERNAL);

    // link peerOutDataAnchor to concatInDataAnchor
    CHK_RET(AddOpsEdge(concatDimNodePtr->GetOutDataAnchor(0), concatNodeInfo.concatNodePtr->GetInDataAnchor(0)));
    for (uint32_t idx = 0; idx < peerOutDataAnchor.size(); idx++) {
      CHK_RET(AddOpsEdge(peerOutDataAnchor[idx], concatNodeInfo.concatNodePtr->GetInDataAnchor(idx + 1)));
    }

    // link concatOutDataAnchor to peerInDataAnchor
    for (ge::InDataAnchorPtr &peerIndataAnchor : nodeInfos[nodeIndex].peerInDataAnchor) {
      CHK_RET(AddOpsEdge(concatNodeInfo.concatNodePtr->GetOutDataAnchor(0), peerIndataAnchor));
    }
    fusionNodesInfo.recvDataConcats.push_back(concatNodeInfo.concatNodePtr);
    HCCL_INFO("[Add][RecvDataConCat]node[%s] success.", concatNodeInfo.nodeName.c_str());
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::RestoreOpsEdges(std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                                                 AlltoAllVCFusionNodesInfo &fusionNodesInfo) {
  ge::graphStatus gRet;
  for (u32 nodeIndex = 0; nodeIndex < nodeInfos.size(); nodeIndex++) {
    // link peerOutControlAnchor and inControlAnchor
    for (auto peerOutControlAnchor : nodeInfos[nodeIndex].peerOutControlAnchor) {
      gRet = ge::GraphUtils::AddEdge(peerOutControlAnchor,
                                     fusionNodesInfo.sendCountSplits[nodeIndex]->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.sendCountSplits[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);

      gRet = ge::GraphUtils::AddEdge(peerOutControlAnchor,
                                     fusionNodesInfo.recvCountSplits0[nodeIndex]->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.recvCountSplits0[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);

      gRet = ge::GraphUtils::AddEdge(peerOutControlAnchor,
                                     fusionNodesInfo.sendCountMatrixAddNNodePtr->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.sendCountMatrixAddNNodePtr->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }

    // link peerOutDataToInControl and inControlAnchor
    for (auto peerOutDataToInControl : nodeInfos[nodeIndex].peerOutDataToInControl) {
      gRet = ge::GraphUtils::AddEdge(peerOutDataToInControl,
                                     fusionNodesInfo.sendCountSplits[nodeIndex]->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.sendCountSplits[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);

      gRet = ge::GraphUtils::AddEdge(peerOutDataToInControl,
                                     fusionNodesInfo.recvCountSplits0[nodeIndex]->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.recvCountSplits0[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);

      gRet = ge::GraphUtils::AddEdge(peerOutDataToInControl,
                                     fusionNodesInfo.sendCountMatrixAddNNodePtr->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add inControl edge failed.",
                             fusionNodesInfo.sendCountMatrixAddNNodePtr->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }

    // Link outControlAnchor and peerInControlAnchor
    for (auto peerInControlAnchor : nodeInfos[nodeIndex].peerInControlAnchor) {
      gRet = ge::GraphUtils::AddEdge(fusionNodesInfo.recvDataConcats[nodeIndex]->GetOutControlAnchor(),
                                     peerInControlAnchor);
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add outControl edge failed.",
                             fusionNodesInfo.recvDataConcats[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }

    // Link outDataAnchor and peerInControlFromOutData
    for (auto peerInControlFromOutData : nodeInfos[nodeIndex].peerInControlFromOutData) {
      gRet = ge::GraphUtils::AddEdge(fusionNodesInfo.recvDataConcats[nodeIndex]->GetOutDataAnchor(0),
                                     peerInControlFromOutData);
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add outControl edge failed.",
                             fusionNodesInfo.recvDataConcats[nodeIndex]->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::CreateSplitNode(SplitNodeInfo &splitNodeInfo, ge::ComputeGraph &graph) {
  ge::graphStatus geRet = ge::GRAPH_SUCCESS;
  ge::OpDescPtr splitOpDescPtr = nullptr;
  EXECEPTION_CATCH((splitOpDescPtr = std::make_shared<ge::OpDesc>(splitNodeInfo.nodeName.c_str(), "Split")),
                   return HCCL_E_INTERNAL);

  geRet = splitOpDescPtr->AddInputDesc("split_dim", splitNodeInfo.inputSplitDim);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][Split]node[%s] add input: split_dim failed", splitOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  geRet = splitOpDescPtr->AddInputDesc("x", splitNodeInfo.inputX);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][Split]node[%s] add input: x failed", splitOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  for (u32 i = 0; i < splitNodeInfo.outputY.size(); i++) {
    geRet = splitOpDescPtr->AddOutputDesc("y" + std::to_string(i), splitNodeInfo.outputY[i]);
    CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Create][Split]node[%s] add output failed", splitOpDescPtr->GetName().c_str()),
                HCCL_E_INTERNAL);
  }

  bool bErr = ge::AttrUtils::SetInt(splitOpDescPtr, "num_split", splitNodeInfo.numSplit);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Create][Split]node[%s] set attr: num_split failed", splitOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);
  std::string dynamicOutputName = "y";
  bErr = ge::AttrUtils::SetInt(splitOpDescPtr, DYNAMIC_OUTPUT_TD_NUM(dynamicOutputName), splitNodeInfo.numSplit);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Create][Split]node[%s] set attr: dynamicOutput[%s] failed",
                         splitOpDescPtr->GetName().c_str(), dynamicOutputName.c_str()),
              HCCL_E_INTERNAL);

  splitNodeInfo.splitNodePtr = graph.AddNode(splitOpDescPtr);
  CHK_PRT_RET(!splitNodeInfo.splitNodePtr,
              HCCL_ERROR("[Create][Split]graph[%s] add node[%s] failed", graph.GetName().c_str(),
                         splitOpDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  CHK_RET(SetUnknownShape(splitNodeInfo.splitNodePtr, graph));
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::CreateSplitVNode(SplitVNodeInfo &splitvNodeInfo, ge::ComputeGraph &graph) {
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
  CHK_RET(SetUnknownShape(splitvNodeInfo.splitvNodePtr, graph));
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::CreateConcatNode(ConcatNodeInfo &concatNodeInfo, ge::ComputeGraph &graph) {
  ge::graphStatus geRet = ge::GRAPH_SUCCESS;
  ge::OpDescPtr opDescPtr = nullptr;
  EXECEPTION_CATCH((opDescPtr = std::make_shared<ge::OpDesc>(concatNodeInfo.nodeName.c_str(), "Concat")),
                   return HCCL_E_INTERNAL);

  geRet = opDescPtr->AddInputDesc("concat_dim", concatNodeInfo.inputConcatDim);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][Concat]node[%s] "
                         "add input: concat_dim failed",
                         opDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  for (u32 i = 0; i < concatNodeInfo.inputX.size(); i++) {
    geRet = opDescPtr->AddInputDesc("x" + std::to_string(i), concatNodeInfo.inputX[i]);
    CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Create][Concat]node[%s] add input: x failed", opDescPtr->GetName().c_str()),
                HCCL_E_INTERNAL);
  }

  geRet = opDescPtr->AddOutputDesc("y", concatNodeInfo.outputY);
  CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Create][Concat]node[%s] add output failed", opDescPtr->GetName().c_str()), HCCL_E_INTERNAL);

  bool bErr = ge::AttrUtils::SetInt(opDescPtr, "N", concatNodeInfo.N);
  CHK_PRT_RET(!bErr, HCCL_ERROR("[Create][Concat]node[%s] set attr: N failed", opDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  std::string dynamicInputName = "x";
  bErr = ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(dynamicInputName), concatNodeInfo.N);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Create][Concat]node[%s] set attr: dynamicInput[%s] failed", opDescPtr->GetName().c_str(),
                         dynamicInputName.c_str()),
              HCCL_E_INTERNAL);

  concatNodeInfo.concatNodePtr = graph.AddNode(opDescPtr);
  CHK_PRT_RET(!concatNodeInfo.concatNodePtr,
              HCCL_ERROR("[Create][SplitV]graph[%s] add node[%s] failed", graph.GetName().c_str(),
                         opDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);
  CHK_RET(SetUnknownShape(concatNodeInfo.concatNodePtr, graph));
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::CreateConstNode(ge::NodePtr &nodePtr, std::string nodeName,
                                                 std::vector<int32_t> nodeValue, std::vector<int64_t> dim,
                                                 ge::ComputeGraph &graph) {
  ge::GeShape shape(dim);
  ge::GeTensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);

  ge::OpDescPtr opDescPtr = nullptr;
  EXECEPTION_CATCH((opDescPtr = std::make_shared<ge::OpDesc>(nodeName.c_str(), "Const")), return HCCL_E_INTERNAL);

  ge::GeTensorPtr tensorPtr = nullptr;
  EXECEPTION_CATCH((tensorPtr = std::make_shared<ge::GeTensor>(
                        tensorDesc, reinterpret_cast<uint8_t *>(nodeValue.data()), sizeof(int32_t))),
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

HcclResult HcomAlltoAllVCFusion::AddOpsEdge(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst) {
  CHK_PRT_RET((src == nullptr || dst == nullptr), HCCL_ERROR("[Add][OpsEdge]src Anchor or dst Anchor is nullptr."),
              HCCL_E_INTERNAL);
  ge::graphStatus geRet = ge::GraphUtils::AddEdge(src, dst);
  CHK_PRT_RET(geRet != ge::GRAPH_SUCCESS,
              HCCL_ERROR("[Add][OpsEdge]Failed to add edge between node[%s] and node[%s].",
                         src->GetOwnerNode()->GetName().c_str(), dst->GetOwnerNode()->GetName().c_str()),
              HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::CheckAlltoAllVCNodeInfo(std::vector<AlltoAllVCNodeInfo> &nodeInfos) {
  // 校验rank和rankSize有效性
  for (u32 i = 1; i < nodeInfos.size(); ++i) {
    bool bErr = nodeInfos[i].rank >= 0 && nodeInfos[i].rank < nodeInfos[i].rankSize;
    CHK_PRT_RET(!bErr,
                HCCL_ERROR("[Check][AlltoAllVCNodeInfo] nodeInfos[%u]: "
                           "rank[%d] and rankSize[%d] is not supported, expect rank >= 0 and rank < rankSize",
                           i, nodeInfos[i].rank, nodeInfos[i].rankSize),
                HCCL_E_PARA);
  }

  // 校验所有alltoallvc的rankSize一致
  for (u32 i = 1; i < nodeInfos.size(); ++i) {
    CHK_PRT_RET((nodeInfos[i - 1].rankSize != nodeInfos[i].rankSize),
                HCCL_ERROR("RankSize of node[%u] is %d and that of node[%u] is %d. Expect to be equal.", i - 1,
                           nodeInfos[i - 1].rankSize, i, nodeInfos[i].rankSize),
                HCCL_E_INTERNAL);
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCFusion::SetUnknownShape(ge::NodePtr &nodePtr, ge::ComputeGraph &graph) {
  bool bRet = ge::AttrUtils::SetBool(nodePtr->GetOpDesc(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  CHK_PRT_RET(!bRet,
              HCCL_ERROR("[Set][UnknownShapeAttr]graph[%s]: node [%s] SetBool unknown shape failed",
                         graph.GetName().c_str(), nodePtr->GetName().c_str()),
              HCCL_E_PARA);
  HCCL_DEBUG("graph[%s]: node [%s] unknown shape value is set", graph.GetName().c_str(), nodePtr->GetName().c_str());
  return HCCL_SUCCESS;
}
}  // namespace hccl
