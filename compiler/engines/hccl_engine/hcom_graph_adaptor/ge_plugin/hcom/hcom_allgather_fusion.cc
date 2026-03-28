/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_allgather_fusion.h"
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
HcomAllGatherFusion::HcomAllGatherFusion() {}

HcomAllGatherFusion::~HcomAllGatherFusion() {}

HcclResult HcomAllGatherFusion::Run(ge::ComputeGraph &graph) {
  FusionInfos fusionInfos;
  HcclResult ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][AllGatherFusion]graph[%s]: get fusion HcomAllGather ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);
  HCCL_INFO("there are %u group to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of  HcomAllGather operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0), HCCL_INFO("NOT_CHANGED: the graph has no HcomAllGather op."), HCCL_SUCCESS);

  for (auto iterFusionInfos = fusionInfos.begin(); iterFusionInfos != fusionInfos.end(); iterFusionInfos++) {
    HCCL_INFO("graph[%s] fusionlabel[%s]: there are %zu HcomAllGather ops to be fused.", graph.GetName().c_str(),
              iterFusionInfos->first.c_str(), iterFusionInfos->second.size());

    ret = FuseOps(graph, iterFusionInfos->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("graph[%s] fusionlabel[%s]: fusion HcomAllGather ops failed. "
                           "ret[%d]",
                           graph.GetName().c_str(), iterFusionInfos->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("fuse HcomAllGather op end");
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos) {
  HcclResult ret;
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("HcomAllGatherFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomAllGatherFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
      ret = GetFusionOpInfo(nodePtr, fusionInfos);
      CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("get fusion ops by group failed. ret[%d]", ret), ret);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos) {
  bool bUnknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, bUnknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
              HCCL_E_PARA);

  CHK_PRT_RET(bUnknownShapeNode, HCCL_INFO("node[%s] is unknown shape, no fusion", nodePtr->GetName().c_str()),
              HCCL_SUCCESS);

  FusionOption fusionOption;
  HcclResult ret = GetFusionOption(nodePtr, fusionOption);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node info failed.", nodePtr->GetName().c_str()), ret);
  CHK_PRT_RET(
      fusionOption.fusionAttr != HCOM_ATTR_FUSION_BY_FUSION_ID,
      HCCL_INFO("node[%s] with attr fusion[%lld], no fusion", nodePtr->GetName().c_str(), fusionOption.fusionAttr),
      HCCL_SUCCESS);

  HCCL_DEBUG("get fusion op: node[%s]: comm[%lld], group[%s], fusion[%lld], fusion_id[%lld], dtype[%s]",
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

HcclResult HcomAllGatherFusion::GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
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
      REPORT_PREDEFINED_ERR_MSG("EI0003", std::vector<const char *>({"ccl_op", "value", "parameter", "expect"}),
                                std::vector<const char *>({"HcomAllGatherFusion", fusionValue.c_str(), "fusion",
                                                           "should be 0 ~ 2"}));
      HCCL_ERROR("[%s][%s]errNo[0x%016llx] node[%s] fusion[%lld] is incorrect, should be %lld or %lld",
                 LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA),
                 nodeName.c_str(), fusionOption.fusionAttr, HCOM_ATTR_FUSION_NO_FUSION, HCOM_ATTR_FUSION_BY_FUSION_ID);
      return HCCL_E_PARA;
  }

  // 获取comm和group
  HcclDataType dataType;
  CHK_RET(GetCommFromOpDescPtr(nodePtr, fusionOption));
  CHK_RET(HcomOpUtils::ConversionOpDataType(opDescPtr, nodePtr->GetOpDesc()->GetType(), dataType));
  auto iter = HCOM_DATA_TYPE_STR_MAP.find(dataType);
  CHK_PRT_RET((iter == HCOM_DATA_TYPE_STR_MAP.end()),
              HCCL_ERROR("[Get][Data]node[%s]: hccl data type[%s] transform failed.", opDescPtr->GetName().c_str(),
                         GetDataTypeEnumStr(dataType).c_str()),
              HCCL_E_INTERNAL);
  fusionOption.dtype = iter->second;

  fusionOption.optype = nodePtr->GetOpDesc()->GetType();
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel) {
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

HcclResult HcomAllGatherFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  CHK_PRT_RET((fusionSection.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %u HcomAllGather op.", fusionSection.size()), HCCL_SUCCESS);

  HcclResult ret = RunFusionOpsAllGather(graph, fusionSection);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);

  HCCL_INFO("graph[%s] fuse HcomAllGather op end.", graph.GetName().c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::RunFusionOpsAllGather(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps) {
  std::vector<CommonNodeInfo> nodeInfos(0);
  ge::OpDescPtr fusedOp;
  AllGatherFusionNodesInfo fusionNodesInfo;
  CHK_RET(RemoveOpsEdges(graph, fusionOps, nodeInfos, fusedOp));
  CHK_RET(AddFusionNode(graph, nodeInfos, fusionNodesInfo, fusedOp));
  CHK_RET(RestoreOpsEdges(nodeInfos, fusionNodesInfo));
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
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
    CHK_RET(GetAllGatherOpInfo(nodeInfo.rankSize, nodeInfo.group, nodeInfo.nodeName, fusionOps[idx]));
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

HcclResult HcomAllGatherFusion::GetPeerOutDataToInData(std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
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

HcclResult HcomAllGatherFusion::GetPeerOutDataToInControl(vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
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

HcclResult HcomAllGatherFusion::GetPeerOutControlToInControl(
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

HcclResult HcomAllGatherFusion::GetPeerAnchorFromOutData(
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

HcclResult HcomAllGatherFusion::GetPeerInDataAnchorFromOutData(vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
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

HcclResult HcomAllGatherFusion::GetPeerInControlAnchorFromOutData(
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

HcclResult HcomAllGatherFusion::GetPeerInControlFromOutControl(
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

HcclResult HcomAllGatherFusion::GetAllGatherOpInfo(s32 &rankSize, string &group, std::string &nodeName,
                                                   ge::NodePtr &srcNodePtr) {
  CHK_PRT_RET((srcNodePtr == nullptr), HCCL_ERROR("[Get][GetAllGatherOpInfo]AllGatherOpInfo is nullptr"),
              HCCL_E_INTERNAL);
  bool bErr = false;
  // get rankSize
  bErr = ge::AttrUtils::GetInt(srcNodePtr->GetOpDesc(), "rank_size", rankSize);
  CHK_PRT_RET(
      !bErr,
      HCCL_ERROR("[Get][AllGatherOpInfo]errNo[0x%016llx] get attr rankSize failed.", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);

  // get group
  bErr = ge::AttrUtils::GetStr(srcNodePtr->GetOpDesc(), "group", group);
  CHK_PRT_RET(!bErr,
              HCCL_ERROR("[Get][AllGatherOpInfo]errNo[0x%016llx] get attr group failed.", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);

  // get nodeName
  nodeName = srcNodePtr->GetName();
  HCCL_DEBUG("[Get][AllGatherOpInfo]node[%s], rankSize[%d], group[%s]", nodeName.c_str(), rankSize, group.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::AddFusionNode(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                                              AllGatherFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp) {
  CHK_RET(AddAllGatherNode(graph, nodeInfos, fusionNodesInfo, fusedOp));
  CHK_RET(AddRecvDataConCat(graph, nodeInfos, fusionNodesInfo));
  HCCL_DEBUG("Add AllGather FusionNode success");
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::AddAllGatherNode(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                                                 AllGatherFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp) {
  // name
  std::string nodeName = nodeInfos[0].nodeName + "_fusion";

  // create opdesc
  CHK_PRT_RET((fusedOp == nullptr), HCCL_ERROR("[Add][AllGather]node[%s] alloc desc failed", nodeName.c_str()),
              HCCL_E_INTERNAL);
  fusedOp->SetName(nodeName.c_str());

  // input: send_data
  std::vector<ge::OutDataAnchorPtr> outDataAnchor;
  std::vector<ge::GeShape> inputShape;
  std::vector<ge::Format> inputFormat;
  std::vector<ge::DataType> inputDataType;
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    ge::OutDataAnchorPtr sendDataAnchor = nodeInfos[idx].peerOutDataAnchor[0];
    CHK_SMART_PTR_NULL(sendDataAnchor);
    ge::NodePtr sendDataNodePtr = sendDataAnchor->GetOwnerNode();
    CHK_SMART_PTR_NULL(sendDataNodePtr);
    ge::GeTensorDesc sendDataDesc = sendDataNodePtr->GetOpDesc()->GetOutputDesc(sendDataAnchor->GetIdx());
    outDataAnchor.push_back(sendDataNodePtr->GetOutDataAnchor(sendDataAnchor->GetIdx()));
    inputShape.push_back(sendDataDesc.GetShape());
    inputFormat.push_back(sendDataDesc.GetFormat());
    inputDataType.push_back(sendDataDesc.GetDataType());
  }

  // add node and link send_data anchor
  ge::NodePtr allgatherNodePtr = graph.AddNode(fusedOp);
  CHK_PRT_RET(!allgatherNodePtr,
              HCCL_ERROR("[Add][AllGather]create AllGather node[%s] failed", fusedOp->GetName().c_str()),
              HCCL_E_INTERNAL);
  ge::NodeUtils::AppendInputAnchor(allgatherNodePtr, (static_cast<int32_t>(nodeInfos.size())));
  ge::NodeUtils::AppendOutputAnchor(allgatherNodePtr, (static_cast<int32_t>(nodeInfos.size()) * static_cast<int32_t>(nodeInfos[0].rankSize)));
  for (uint32_t idx = 0; idx < nodeInfos.size(); idx++) {
    ge::GeTensorDescPtr allgatherNodeInputPtr = allgatherNodePtr->GetOpDesc()->MutableInputDesc(idx);
    CHK_SMART_PTR_NULL(allgatherNodeInputPtr);
    allgatherNodeInputPtr->SetShape(inputShape[idx]);
    allgatherNodeInputPtr->SetOriginShape(inputShape[idx]);
    allgatherNodeInputPtr->SetFormat(inputFormat[idx]);
    allgatherNodeInputPtr->SetDataType(inputDataType[idx]);
    CHK_RET(AddOpsEdge(outDataAnchor[idx], allgatherNodePtr->GetInDataAnchor(idx)));
  }
  for (uint32_t idx = 0; idx < nodeInfos.size() * nodeInfos[0].rankSize; idx++) {
    ge::GeTensorDescPtr allgatherOutDescPtr = allgatherNodePtr->GetOpDesc()->MutableOutputDesc(idx);
    CHK_SMART_PTR_NULL(allgatherOutDescPtr);
    allgatherOutDescPtr->SetShape(inputShape[idx % nodeInfos.size()]);
    allgatherOutDescPtr->SetOriginShape(inputShape[idx % nodeInfos.size()]);
    allgatherOutDescPtr->SetFormat(inputFormat[idx % nodeInfos.size()]);
    allgatherOutDescPtr->SetDataType(inputDataType[idx % nodeInfos.size()]);
  }

  fusionNodesInfo.allgatherFusionNodePtr = allgatherNodePtr;
  HCCL_INFO("[Add][AllGather]node[%s] success.", nodeName.c_str());
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::AddRecvDataConCat(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                                                  AllGatherFusionNodesInfo &fusionNodesInfo) {
  ge::NodePtr peerOutDataNodePtr = fusionNodesInfo.allgatherFusionNodePtr;
  CHK_SMART_PTR_NULL(peerOutDataNodePtr);
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
    int64_t dim_first = 0;
    ge::GeShape outputShape;
    ge::GeShape inputShape;
    std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
    for (uint32_t idx = nodeIndex; idx < peerOutDataAnchorsSize; idx += nodeInfos.size()) {
      concatNodeInfo.inputX.push_back(peerOutDataNodePtr->GetOpDesc()->GetOutputDesc(idx));
      peerOutDataAnchor.push_back(peerOutDataNodePtr->GetOutDataAnchor(static_cast<s32>(idx)));
      ge::GeTensorDesc SizeSplit = peerOutDataNodePtr->GetOpDesc()->GetOutputDesc(idx);
      inputShape = SizeSplit.GetShape();
      dim_first = dim_first + inputShape.GetDim(0);
    }
    outputShape = inputShape;
    outputShape.SetDim(0, dim_first);

    // attr: N
    concatNodeInfo.N = static_cast<s32>(concatNodeInfo.inputX.size());

    // output: y
    ge::InDataAnchorPtr peerIndataAnchor0 = nodeInfos[nodeIndex].peerInDataAnchor[0];
    CHK_SMART_PTR_NULL(peerIndataAnchor0);
    ge::GeTensorDesc outputTensor =
        peerIndataAnchor0->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peerIndataAnchor0->GetIdx());
    outputTensor.SetShape(outputShape);
    outputTensor.SetOriginShape(outputShape);
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
HcclResult HcomAllGatherFusion::AddOpsEdge(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst) {
  CHK_PRT_RET((src == nullptr || dst == nullptr), HCCL_ERROR("[Add][OpsEdge]src Anchor or dst Anchor is nullptr."),
              HCCL_E_INTERNAL);
  ge::graphStatus geRet = ge::GraphUtils::AddEdge(src, dst);
  CHK_PRT_RET(geRet != ge::GRAPH_SUCCESS,
              HCCL_ERROR("[Add][OpsEdge]Failed to add edge between node[%s] and node[%s].",
                         src->GetOwnerNode()->GetName().c_str(), dst->GetOwnerNode()->GetName().c_str()),
              HCCL_E_INTERNAL);
  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::CreateConcatNode(ConcatNodeInfo &concatNodeInfo, ge::ComputeGraph &graph) {
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
              HCCL_ERROR("[Create][Concat]graph[%s] add node[%s] failed", graph.GetName().c_str(),
                         opDescPtr->GetName().c_str()),
              HCCL_E_INTERNAL);

  return HCCL_SUCCESS;
}

HcclResult HcomAllGatherFusion::CreateConstNode(ge::NodePtr &nodePtr, std::string nodeName,
                                                std::vector<int32_t> nodeValue, std::vector<int64_t> dim,
                                                ge::ComputeGraph &graph) {
  ge::GeTensorDesc tensorDesc(ge::GeShape(dim), ge::FORMAT_ND, ge::DT_INT32);

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

HcclResult HcomAllGatherFusion::RestoreOpsEdges(std::vector<CommonNodeInfo> &nodeInfos,
                                                AllGatherFusionNodesInfo &fusionNodesInfo) {
  ge::graphStatus gRet;

  for (u32 nodeIndex = 0; nodeIndex < nodeInfos.size(); nodeIndex++) {
    // link peerOutControlAnchor and inControlAnchor
    for (auto peerOutControlAnchor : nodeInfos[nodeIndex].peerOutControlAnchor) {
      gRet =
          ge::GraphUtils::AddEdge(peerOutControlAnchor, fusionNodesInfo.allgatherFusionNodePtr->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add outControl edge failed.",
                             fusionNodesInfo.allgatherFusionNodePtr->GetName().c_str()),
                  HCCL_E_INTERNAL);
    }
    // link peerOutDataToInControl and inControlAnchor
    for (auto peerOutDataToInControl : nodeInfos[nodeIndex].peerOutDataToInControl) {
      gRet =
          ge::GraphUtils::AddEdge(peerOutDataToInControl, fusionNodesInfo.allgatherFusionNodePtr->GetInControlAnchor());
      CHK_PRT_RET(gRet != ge::GRAPH_SUCCESS,
                  HCCL_ERROR("[Restore][OpsEdges]node[%s] add outControl edge failed.",
                             fusionNodesInfo.allgatherFusionNodePtr->GetName().c_str()),
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
}  // namespace hccl
