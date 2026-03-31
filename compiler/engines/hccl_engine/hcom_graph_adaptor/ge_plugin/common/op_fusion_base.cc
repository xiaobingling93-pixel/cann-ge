/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "hcom_op_utils.h"
#include "hccl/hcom.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "op_fusion_base.h"

namespace hccl {
OpFusionBase::OpFusionBase() {}

OpFusionBase::~OpFusionBase() {}

HcclResult OpFusionBase::Run([[maybe_unused]] ge::ComputeGraph &graph) {
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::RunFusionOps(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps, u32 segmentNum,
                                      std::vector<u32> &segmentIndex) {
  std::vector<FusionAnchorInfo_t> anchorInfos(0);
  std::vector<ge::OpDescPtr> fusedOps(0);
  std::vector<ge::NodePtr> fusedNodes(0);
  CHK_RET(RemoveOpsEdges(graph, fusionOps, segmentNum, segmentIndex, anchorInfos, fusedOps));
  CHK_RET(AddFusionNode(graph, fusedOps, fusedNodes));
  CHK_RET(RestoreOpsEdges(fusedNodes, anchorInfos));
  CHK_RET(AddNodesDependence(fusedNodes));
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::RunFusionOpsReduce(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps) {
  std::vector<FusionAnchorInfo_t> anchorInfos(0);
  std::vector<ge::OpDescPtr> fusedOps(0);
  std::vector<ge::NodePtr> fusedNodes(0);
  CHK_RET(RemoveOpsEdges(graph, fusionOps, anchorInfos, fusedOps));
  CHK_RET(AddFusionNode(graph, fusedOps, fusedNodes));
  CHK_RET(RestoreOpsEdges(fusedNodes, anchorInfos));
  CHK_RET(AddNodesDependence(fusedNodes));
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::AddNodesDependence(const std::vector<ge::NodePtr> &fusedNodes) {
  // for avoiding allreduce ops parallel, add dependence edge between ops.
  ge::graphStatus gRet;
  for (u32 nodeIndex = 0; nodeIndex < (fusedNodes.size() - 1); nodeIndex++) {
    CHK_SMART_PTR_NULL(fusedNodes[nodeIndex]->GetOutControlAnchor());
    CHK_SMART_PTR_NULL(fusedNodes[nodeIndex + 1]->GetInControlAnchor());
    gRet = ge::GraphUtils::AddEdge(fusedNodes[nodeIndex]->GetOutControlAnchor(),
                                   fusedNodes[nodeIndex + 1]->GetInControlAnchor());
    bool bErr = (gRet != ge::GRAPH_SUCCESS);
    CHK_PRT_RET(bErr, HCCL_ERROR("[Add][NodesDependence]add node dependence failed."), HCCL_E_INTERNAL);
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::RestoreOpsEdges(std::vector<ge::NodePtr> &fusedNodes,
                                         std::vector<FusionAnchorInfo_t> &anchorInfos) {
  ge::graphStatus gRet;
  bool bErr = false;
  for (u32 segIndex = 0; segIndex < fusedNodes.size(); segIndex++) {
    auto node = fusedNodes[segIndex];
    auto anchorInfosPerSeg = anchorInfos[segIndex];
    // Link the inputDataAnchor
    for (u32 anchorIndex = 0; anchorIndex < anchorInfosPerSeg.peerOutDataAnchor.size(); anchorIndex++) {
      gRet =
          ge::GraphUtils::AddEdge(anchorInfosPerSeg.peerOutDataAnchor[anchorIndex], node->GetInDataAnchor(anchorIndex));
      bErr = (gRet != ge::GRAPH_SUCCESS);
      CHK_PRT_RET(bErr, HCCL_ERROR("[Restore][OpsEdges]add input data edge failed."), HCCL_E_INTERNAL);
    }
    // Link the inputControlAnchor
    for (u32 anchorIndex = 0; anchorIndex < anchorInfosPerSeg.peerOutControlAnchor.size(); anchorIndex++) {
      gRet = ge::GraphUtils::AddEdge(anchorInfosPerSeg.peerOutControlAnchor[anchorIndex], node->GetInControlAnchor());
      bErr = (gRet != ge::GRAPH_SUCCESS);
      CHK_PRT_RET(bErr, HCCL_ERROR("[Restore][OpsEdges]add input control edge failed."), HCCL_E_INTERNAL);
    }
    for (u32 anchorIndex = 0; anchorIndex < anchorInfosPerSeg.peerOutDataToInControl.size(); anchorIndex++) {
      gRet = ge::GraphUtils::AddEdge(anchorInfosPerSeg.peerOutDataToInControl[anchorIndex], node->GetInControlAnchor());
      bErr = (gRet != ge::GRAPH_SUCCESS);
      CHK_PRT_RET(bErr, HCCL_ERROR("[Restore][OpsEdges]add edge from out data to incontrol failed."), HCCL_E_INTERNAL);
    }
    // Link the outputDataAnchor
    for (u32 anchorIndex = 0; anchorIndex < anchorInfosPerSeg.peerInDataAnchor.size(); anchorIndex++) {
      auto peerInDataAnchor = anchorInfosPerSeg.peerInDataAnchor[anchorIndex].second;
      auto outDataAnchor = node->GetOutDataAnchor(anchorInfosPerSeg.peerInDataAnchor[anchorIndex].first);
      gRet = ge::GraphUtils::AddEdge(outDataAnchor, peerInDataAnchor);
      bErr = (gRet != ge::GRAPH_SUCCESS);
      CHK_PRT_RET(bErr, HCCL_ERROR("[Restore][OpsEdges]add output data edge failed."), HCCL_E_INTERNAL);
    }
    for (u32 anchorIndex = 0; anchorIndex < anchorInfosPerSeg.peerInControlFromOutData.size(); anchorIndex++) {
      auto peerInControlAnchor = anchorInfosPerSeg.peerInControlFromOutData[anchorIndex].second;
      auto outDataAnchor = node->GetOutDataAnchor(anchorInfosPerSeg.peerInControlFromOutData[anchorIndex].first);
      gRet = ge::GraphUtils::AddEdge(outDataAnchor, peerInControlAnchor);
      bErr = (gRet != ge::GRAPH_SUCCESS);
      CHK_PRT_RET(bErr, HCCL_ERROR("[Restore][OpsEdges]add edge from out data to in control failed."), HCCL_E_INTERNAL);
    }
    // Link the outputControlAnchor
    for (u32 anchorIndex = 0; anchorIndex < anchorInfosPerSeg.peerInControlAnchor.size(); anchorIndex++) {
      gRet = ge::GraphUtils::AddEdge(node->GetOutControlAnchor(), anchorInfosPerSeg.peerInControlAnchor[anchorIndex]);
      bErr = (gRet != ge::GRAPH_SUCCESS);
      CHK_PRT_RET(bErr, HCCL_ERROR("[Restore][OpsEdges]add output control edge failed."), HCCL_E_INTERNAL);
    }
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::AddFusionNode(ge::ComputeGraph &graph, std::vector<ge::OpDescPtr> fusedOps,
                                       std::vector<ge::NodePtr> &fusedNodes) {
  for (uint32_t segmentIdx = 0; segmentIdx < fusedOps.size(); segmentIdx++) {
    ge::NodePtr newNodePtr = graph.AddNode(fusedOps[segmentIdx]);
    fusedNodes.push_back(newNodePtr);
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps, u32 segmentNum,
                                        const std::vector<u32> &segmentIndex,
                                        std::vector<FusionAnchorInfo_t> &anchorInfos,
                                        std::vector<ge::OpDescPtr> &fusedOps) {
  uint32_t start = 0;
  uint32_t end = 0;
  for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; segmentIdx++) {
    int outDataAnchorIndex = 0;
    std::unordered_set<uintptr_t> anchorPtrSet;
    FusionAnchorInfo_t anchorInfosPreSeg;
    ge::OpDescPtr originDescPtr = fusionOps[start]->GetOpDesc();
    CHK_SMART_PTR_NULL(originDescPtr);
    ge::OpDescPtr newOpDesc = ge::AttrUtils::CopyOpDesc(originDescPtr);
    CHK_SMART_PTR_NULL(newOpDesc);
    end = segmentIndex[segmentIdx];
    HCCL_INFO("[Remove][OpsEdges]graph[%s] fusion from segment from %u to %u", graph.GetName().c_str(), start, end);
    for (uint32_t idx = start; idx <= end; idx++) {
      std::vector<std::pair<int, int>> duplication;
      std::string opName = fusionOps[idx]->GetOpDesc()->GetName();
      // get all anchors connected to the node,and remove the edges between them.
      CHK_RET(GetPeerOutDataToInData(anchorPtrSet, anchorInfosPreSeg.peerOutDataAnchor, fusionOps[idx], newOpDesc,
                                     duplication));
      CHK_RET(GetPeerOutDataToInControl(anchorPtrSet, anchorInfosPreSeg.peerOutDataToInControl, fusionOps[idx]));
      CHK_RET(GetPeerOutControlToInControl(anchorPtrSet, anchorInfosPreSeg.peerOutControlAnchor, fusionOps[idx]));
      CHK_RET(GetPeerAnchorFromOutData(anchorPtrSet, anchorInfosPreSeg.peerInDataAnchor,
                                       anchorInfosPreSeg.peerInControlFromOutData, fusionOps[idx], newOpDesc,
                                       outDataAnchorIndex, duplication));
      CHK_RET(GetPeerInControlFromOutControl(anchorPtrSet, anchorInfosPreSeg.peerInControlAnchor, fusionOps[idx]));
      // remove the node after keeping all anchors
      ge::graphStatus geRet = graph.RemoveNode(fusionOps[idx]);
      CHK_PRT_RET((geRet != ge::GRAPH_SUCCESS),
                  HCCL_ERROR("[Remove][OpsEdges]remove node[%s] failed. ret[%u]", opName.c_str(), geRet),
                  HCCL_E_INTERNAL);
      HCCL_DEBUG("fusionOps idx[%u] name[%s]", idx, opName.c_str());
    }
    anchorInfos.push_back(anchorInfosPreSeg);
    fusedOps.push_back(newOpDesc);
    HCCL_INFO(
        "FusedOp[%s]:index[%u],peerOutDataAnchor[%u],peerOutDataToInControl[%u],peerOutControlAnchor[%u], "
        "peerInDataAnchor[%u],peerInControlFromOutData[%u],peerInControlAnchor[%u], "
        "inDescSize[%zu],outDescSize[%zu]",
        newOpDesc->GetName().c_str(), segmentIdx, anchorInfosPreSeg.peerOutDataAnchor.size(),
        anchorInfosPreSeg.peerOutDataToInControl.size(), anchorInfosPreSeg.peerOutControlAnchor.size(),
        anchorInfosPreSeg.peerInDataAnchor.size(), anchorInfosPreSeg.peerInControlFromOutData.size(),
        anchorInfosPreSeg.peerInControlAnchor.size(), newOpDesc->GetAllInputName().size(),
        newOpDesc->GetAllOutputsDescSize());
    start = end + 1;
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
                                        std::vector<FusionAnchorInfo_t> &anchorInfos,
                                        std::vector<ge::OpDescPtr> &fusedOps) {
  ge::graphStatus gRet;
  uint32_t start = 0;
  int outDataAnchorIndex = 0;
  std::unordered_set<uintptr_t> anchorPtrSet;
  FusionAnchorInfo_t anchorInfosPreSeg;
  ge::OpDescPtr originDescPtr = fusionOps[start]->GetOpDesc();
  CHK_SMART_PTR_NULL(originDescPtr);
  ge::OpDescPtr newOpDesc = ge::AttrUtils::CopyOpDesc(originDescPtr);
  CHK_SMART_PTR_NULL(newOpDesc);

  uint32_t end = fusionOps.size();
  for (uint32_t idx = start; idx < end; idx++) {
    std::vector<std::pair<int, int>> duplication;
    std::string opName = fusionOps[idx]->GetOpDesc()->GetName();
    // get all anchors connected to the node,and remove the edges between them.
    CHK_RET(GetPeerOutDataToInData(anchorPtrSet, anchorInfosPreSeg.peerOutDataAnchor, fusionOps[idx], newOpDesc,
                                   duplication));
    CHK_RET(GetPeerOutDataToInControl(anchorPtrSet, anchorInfosPreSeg.peerOutDataToInControl, fusionOps[idx]));
    CHK_RET(GetPeerOutControlToInControl(anchorPtrSet, anchorInfosPreSeg.peerOutControlAnchor, fusionOps[idx]));
    CHK_RET(GetPeerAnchorFromOutData(anchorPtrSet, anchorInfosPreSeg.peerInDataAnchor,
                                     anchorInfosPreSeg.peerInControlFromOutData, fusionOps[idx], newOpDesc,
                                     outDataAnchorIndex, duplication));
    CHK_RET(GetPeerInControlFromOutControl(anchorPtrSet, anchorInfosPreSeg.peerInControlAnchor, fusionOps[idx]));
    // remove the node after keeping all anchors
    gRet = graph.RemoveNode(fusionOps[idx]);
    CHK_PRT_RET((gRet != ge::GRAPH_SUCCESS),
                HCCL_ERROR("[Remove][OpsEdges]remove node[%s] failed. ret[%u]", opName.c_str(), gRet), HCCL_E_INTERNAL);
    HCCL_DEBUG("fusionOps idx[%u] name[%s]", idx, opName.c_str());
  }
  anchorInfos.push_back(anchorInfosPreSeg);
  fusedOps.push_back(newOpDesc);

  HCCL_INFO(
      "FusedOp[%s]:peerOutDataAnchor[%u],peerOutDataToInControl[%u],peerOutControlAnchor[%u], "
      "peerInDataAnchor[%u],peerInControlFromOutData[%u],peerInControlAnchor[%u],inDescSize[%zu], "
      "outDescSize[%zu]",
      newOpDesc->GetName().c_str(), anchorInfosPreSeg.peerOutDataAnchor.size(),
      anchorInfosPreSeg.peerOutDataToInControl.size(), anchorInfosPreSeg.peerOutControlAnchor.size(),
      anchorInfosPreSeg.peerInDataAnchor.size(), anchorInfosPreSeg.peerInControlFromOutData.size(),
      anchorInfosPreSeg.peerInControlAnchor.size(), newOpDesc->GetAllInputName().size(),
      newOpDesc->GetAllOutputsDescSize());
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetPeerOutDataToInData(std::unordered_set<uintptr_t> &anchorSet,
                                                std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
                                                ge::NodePtr &srcNodePtr, ge::OpDescPtr &dstOpDescPtr,
                                                std::vector<std::pair<int, int>> &duplication) {
  CHK_SMART_PTR_NULL(srcNodePtr);
  CHK_SMART_PTR_NULL(srcNodePtr->GetOpDesc());
  CHK_SMART_PTR_NULL(dstOpDescPtr);
  for (auto inDataAnchor : srcNodePtr->GetAllInDataAnchors()) {
    if (!inDataAnchor) {
      continue;
    }
    ge::OutDataAnchorPtr peerOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    if (!peerOutDataAnchor) {
      continue;
    }
    std::string dstOpName = dstOpDescPtr->GetName();
    std::string srcOpName = srcNodePtr->GetOpDesc()->GetName();
    if (anchorSet.count((uintptr_t)peerOutDataAnchor.get()) == 0) {
      peerOutDataAnchorVec.push_back(peerOutDataAnchor);
      anchorSet.insert((uintptr_t)peerOutDataAnchor.get());
      CHK_SMART_PTR_NULL(inDataAnchor->GetOwnerNode());
      CHK_SMART_PTR_NULL(inDataAnchor->GetOwnerNode()->GetOpDesc());
      auto inTensor = inDataAnchor->GetOwnerNode()->GetOpDesc()->GetInputDesc(inDataAnchor->GetIdx());
      uint64_t memSize = 0;
      HcclResult ret = HcomOpUtils::GetTensorMemSize(inTensor, memSize);
      CHK_PRT_RET(ret != HCCL_SUCCESS,
                  HCCL_ERROR("[Get][Peer]node[%s] input[%d] GetTensorMemSize failed.", srcOpName.c_str(),
                             inDataAnchor->GetIdx()),
                  ret);
      CHK_SMART_PTR_NULL(peerOutDataAnchor->GetOwnerNode());
      CHK_SMART_PTR_NULL(peerOutDataAnchor->GetOwnerNode()->GetOpDesc());
      HCCL_DEBUG("fusion op[%s]: peer out data op: %s, input[%d] size[%llu]", srcOpName.c_str(),
                 peerOutDataAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(), inDataAnchor->GetIdx(), memSize);
      if (dstOpName != srcOpName) {
        CHK_PRT_RET((dstOpDescPtr->AddInputDesc(inTensor) != ge::GRAPH_SUCCESS),
                    HCCL_ERROR("[Get][Peer]GetPeerOutDataToInData: add inputDesc[%d] of srcOp[%s] to destOp[%s] "
                               "failed.",
                               inDataAnchor->GetIdx(), srcOpName.c_str(), dstOpName.c_str()),
                    HCCL_E_INTERNAL);
      }
    } else {
      HCCL_INFO("add inputDesc[%d] of srcOp[%s] to destOp[%s] is skipped. this edge is existed.",
                inDataAnchor->GetIdx(), srcOpName.c_str(), dstOpName.c_str());
      for (int position = 0; position < static_cast<int>(peerOutDataAnchorVec.size()); position++) {
        if (((uintptr_t)peerOutDataAnchor.get()) == ((uintptr_t)peerOutDataAnchorVec[position].get())) {
          duplication.push_back({inDataAnchor->GetIdx(), position});
          break;
        }
      }
    }
    CHK_PRT_RET(
        (ge::GraphUtils::RemoveEdge(peerOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS),
        HCCL_ERROR("[Get][Peer]GetPeerOutDataToInData: remove edge between peer outDataAnchor[%d] of Op[%s] "
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

HcclResult OpFusionBase::GetPeerOutDataToInControl(std::unordered_set<uintptr_t> &anchorSet,
                                                   vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
                                                   ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  ge::InControlAnchorPtr inControlAnchor = srcNodePtr->GetInControlAnchor();
  CHK_SMART_PTR_NULL(inControlAnchor);
  for (auto peerOutDataToInControl : inControlAnchor->GetPeerOutDataAnchors()) {
    if (!peerOutDataToInControl) {
      continue;
    }
    if (anchorSet.count((uintptr_t)peerOutDataToInControl.get()) == 0) {
      peerOutDataToInControlVec.push_back(peerOutDataToInControl);
      anchorSet.insert((uintptr_t)peerOutDataToInControl.get());
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
          "[Get][PeerOutData]GetPeerOutDataToInControl: remove edge between peer outDataAnchor[%d] "
          "of Op[%s] and inControlAnchor[%d] of Op[%s] ok",
          peerOutDataToInControl->GetIdx(), peerOutDataToInControl->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
          inControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str());
    }
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetPeerOutControlToInControl(std::unordered_set<uintptr_t> &anchorSet,
                                                      vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec,
                                                      ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  ge::InControlAnchorPtr inControlAnchor = srcNodePtr->GetInControlAnchor();
  CHK_SMART_PTR_NULL(inControlAnchor);
  for (auto peerOutControlAnchor : inControlAnchor->GetPeerOutControlAnchors()) {
    if (!peerOutControlAnchor) {
      continue;
    }
    if (anchorSet.count((uintptr_t)peerOutControlAnchor.get()) == 0) {
      peerOutControlToInControlVec.push_back(peerOutControlAnchor);
      anchorSet.insert((uintptr_t)peerOutControlAnchor.get());
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
          "[Get][PeerOutControlToInControl]GetPeerOutControlToInControl: remove edge between peer "
          "outControlAnchor[%d] of Op[%s] and inControlAnchor[%d] of Op[%s] ok",
          peerOutControlAnchor->GetIdx(), peerOutControlAnchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(),
          inControlAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str());
    }
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetPeerAnchorFromOutData(
    std::unordered_set<uintptr_t> &anchorSet,
    std::vector<std::pair<u32, ge::InDataAnchorPtr>> &peerInDataFromOutDataVec,
    std::vector<std::pair<u32, ge::InControlAnchorPtr>> &peerInControlFromOutDataVec, ge::NodePtr &srcNodePtr,
    ge::OpDescPtr &dstOpDescPtr, int &index, std::vector<std::pair<int, int>> &duplication) {
  HcclResult ret;
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  CHK_SMART_PTR_NULL(dstOpDescPtr);

  for (auto outDataAnchor : srcNodePtr->GetAllOutDataAnchors()) {
    if (!outDataAnchor) {
      continue;
    }

    int duplicationPosition = -1;
    int outIdx = outDataAnchor->GetIdx();
    for (size_t i = 0; i < duplication.size(); i++) {
      if (outIdx == duplication[i].first) {
        duplicationPosition = i;
        break;
      }
    }
    std::string dstOpName = dstOpDescPtr->GetName();
    std::string srcOpName = srcNodePtr->GetOpDesc()->GetName();
    if (dstOpName != srcOpName) {
      // 算子的输入重复时, 不新增 OutputDesc，使用已有重复的输出
      if ((duplicationPosition < 0) || (outDataAnchor->GetPeerInControlAnchors().size() > 0)) {
        gRet = dstOpDescPtr->AddOutputDesc(
            outDataAnchor->GetOwnerNode()->GetOpDesc()->GetOutputDesc(outDataAnchor->GetIdx()));
        if (gRet != ge::GRAPH_SUCCESS) {
          HCCL_ERROR(
              "[Get][PeerAnchor]GetPeerAnchorFromOutData: add outputDesc[%d] of srcOp[%s] to"
              "destOp[%s] failed.",
              outDataAnchor->GetIdx(), srcOpName.c_str(), dstOpName.c_str());
          return HCCL_E_INTERNAL;
        }
        index++;
      }
    }

    if (duplicationPosition < 0) {
      ret = GetPeerInDataAnchorFromOutData(anchorSet, peerInDataFromOutDataVec, outDataAnchor, srcNodePtr, index);
    } else {
      ret = GetPeerInDataAnchorFromOutData(anchorSet, peerInDataFromOutDataVec, outDataAnchor, srcNodePtr,
                                           duplicationPosition);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][PeerAnchor]GetPeerInDataAnchorFromOutData failed."), ret);
    CHK_RET(
        GetPeerInControlAnchorFromOutData(anchorSet, peerInControlFromOutDataVec, outDataAnchor, srcNodePtr, index));
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetPeerInDataAnchorFromOutData(
    std::unordered_set<uintptr_t> &anchorSet,
    std::vector<std::pair<u32, ge::InDataAnchorPtr>> &peerInDataFromOutDataVec, ge::OutDataAnchorPtr outDataAnchor,
    ge::NodePtr &srcNodePtr, int index) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(outDataAnchor);
  CHK_SMART_PTR_NULL(srcNodePtr);
  for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
    HCCL_INFO("GetPeerAnchorFromOutData: node[%s] has %zu PeerInDataAnchors in OutDataAnchor[%d].",
              srcNodePtr->GetOpDesc()->GetName().c_str(), outDataAnchor->GetPeerInDataAnchors().size(),
              outDataAnchor->GetIdx());
    if (!peerInDataAnchor) {
      continue;
    }
    if (anchorSet.count((uintptr_t)peerInDataAnchor.get()) == 0) {
      std::pair<u32, ge::InDataAnchorPtr> pairPeerInDataAnchor;
      pairPeerInDataAnchor.first = index;
      pairPeerInDataAnchor.second = peerInDataAnchor;
      peerInDataFromOutDataVec.push_back(pairPeerInDataAnchor);
      anchorSet.insert((uintptr_t)peerInDataAnchor.get());
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
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetPeerInControlAnchorFromOutData(
    std::unordered_set<uintptr_t> &anchorSet,
    std::vector<std::pair<u32, ge::InControlAnchorPtr>> &peerInControlFromOutDataVec,
    ge::OutDataAnchorPtr outDataAnchor, ge::NodePtr &srcNodePtr, int index) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(outDataAnchor);
  CHK_SMART_PTR_NULL(srcNodePtr);
  for (auto peerInControlAnchorFromData : outDataAnchor->GetPeerInControlAnchors()) {
    if (!peerInControlAnchorFromData) {
      continue;
    }
    if (anchorSet.count((uintptr_t)peerInControlAnchorFromData.get()) == 0) {
      std::pair<u32, ge::InControlAnchorPtr> pairPeerInControlAnchorFromData;
      pairPeerInControlAnchorFromData.first = index;
      pairPeerInControlAnchorFromData.second = peerInControlAnchorFromData;
      peerInControlFromOutDataVec.push_back(pairPeerInControlAnchorFromData);
      anchorSet.insert((uintptr_t)peerInControlAnchorFromData.get());
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
          "[Get][PeerInControlAnchor]GetPeerAnchorFromOutData: remove edge between outDataAnchor[%d] "
          "of Op[%s] and peer inControlAnchor[%d] of Op[%s] ok",
          outDataAnchor->GetIdx(), srcNodePtr->GetOpDesc()->GetName().c_str(), peerInControlAnchorFromData->GetIdx(),
          peerInControlAnchorFromData->GetOwnerNode()->GetOpDesc()->GetName().c_str());
    }
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetPeerInControlFromOutControl(std::unordered_set<uintptr_t> &anchorSet,
                                                        vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec,
                                                        ge::NodePtr &srcNodePtr) {
  ge::graphStatus gRet;
  CHK_SMART_PTR_NULL(srcNodePtr);
  ge::OutControlAnchorPtr outControlAnchor = srcNodePtr->GetOutControlAnchor();
  CHK_SMART_PTR_NULL(outControlAnchor);
  for (auto peerInControlAnchor : outControlAnchor->GetPeerInControlAnchors()) {
    if (!peerInControlAnchor) {
      continue;
    }
    if (anchorSet.count((uintptr_t)peerInControlAnchor.get()) == 0) {
      peerInControlFromOutControlVec.push_back(peerInControlAnchor);
      anchorSet.insert((uintptr_t)peerInControlAnchor.get());
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
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetCommFromOpDescPtr(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
  auto opDescPtr = nodePtr->GetOpDesc();
  bool bErr = false;
  std::string sGroup;
  if (ge::AttrUtils::HasAttr(opDescPtr, "comm")) {
    bErr = ge::AttrUtils::GetInt(opDescPtr, "comm", fusionOption.fusionComm);
    CHK_PRT_RET(!bErr, HCCL_ERROR("errNo[0x%016llx] get attr \"comm\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
                HCCL_E_PARA);
    if (fusionOption.fusionComm != static_cast<int64_t>(CommNumHcom::COMM_VALUE_DEFAULT)) {
      return HCCL_SUCCESS;
    } else {
      HCCL_INFO("[OpFusionBase]get comm equal to 0, should get group.");
      CHK_RET(GetGroupFromOpDescPtr(nodePtr, sGroup));
      fusionOption.group = sGroup;
    }
  } else {
    CHK_RET(GetGroupFromOpDescPtr(nodePtr, sGroup));
    fusionOption.group = sGroup;
  }
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetGroupFromOpDescPtr(const ge::NodePtr &nodePtr, std::string &group) {
  auto opDescPtr = nodePtr->GetOpDesc();
  bool bErr = false;
  CHK_PRT_RET(!(ge::AttrUtils::HasAttr(opDescPtr, "group")),
              HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] \
        node[%s] has no attr \"group\".",
                         HCOM_ERROR_CODE(HCCL_E_PARA), nodePtr->GetName().c_str()),
              HCCL_E_PARA);
  bErr = ge::AttrUtils::GetStr(opDescPtr, "group", group);
  CHK_PRT_RET(!bErr, HCCL_ERROR("errNo[0x%016llx] get attr \"group\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  CHK_PRT_RET(group.empty(), HCCL_ERROR("errNo[0x%016llx] group from opDesc is empty.", HCOM_ERROR_CODE(HCCL_E_PARA)),
              HCCL_E_PARA);
  return HCCL_SUCCESS;
}

HcclResult OpFusionBase::GetFusionhashFromGraph(ge::ComputeGraph &graph, std::string &fusionHash) {
  std::stringstream ss;
  int64_t inputSize = 0;
  std::hash<std::string> hashString;
  if (graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
    HCCL_ERROR("Topological of Graph sort failed!");
    return HCCL_E_PARA;
  }
  for (auto nodePtr : graph.GetDirectNode()) {
    auto opDescPtr = nodePtr->GetOpDesc();
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
      for (u64 i = 0; i < opDescPtr->GetInputsSize(); i++) {
        auto inputTensor = opDescPtr->GetInputDesc(i);
        auto dataType = inputTensor.GetDataType();
        ss << dataType;
        CHK_PRT_RET((ge::GRAPH_SUCCESS != ge::TensorUtils::GetSize(inputTensor, inputSize)),
                    HCCL_ERROR("[HcomOpUtils] Get input Size failed"), HCCL_E_PARA);
        ss << inputSize;
      }
    }
  }
  size_t middleHash = hashString(ss.str());
  fusionHash = std::to_string(middleHash);
  HCCL_INFO("Get fusionHash[%s] success.", fusionHash.c_str());
  return HCCL_SUCCESS;
}
}  // namespace hccl
