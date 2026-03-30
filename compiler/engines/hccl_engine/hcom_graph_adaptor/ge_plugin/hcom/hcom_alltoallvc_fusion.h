/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_ALLTOALLVC_FUSION_H
#define HCOM_ALLTOALLVC_FUSION_H

#include "hccl/base.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "op_fusion_base_pub.h"

namespace hccl {
constexpr u32 ALLTOALLVC_INPUT_NUM = 2;
constexpr s32 SPLITV_INPUT_X_INDEX = 0;
constexpr s32 SPLITV_INPUT_SIZESPLIT_INDEX = 1;
constexpr s32 SPLITV_INPUT_SPLITDIM_INDEX = 2;
constexpr s32 SPLITV_NUMSPLIT_MAX = 61;

// 记录单个AlltoAllVC算子的数据边、控制边、属性
using AlltoAllVCNodeInfo = struct alltoallvcNodeInfo {
  std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
  std::vector<ge::OutDataAnchorPtr> peerOutDataToInControl;
  std::vector<ge::OutControlAnchorPtr> peerOutControlAnchor;
  std::vector<ge::InDataAnchorPtr> peerInDataAnchor;
  std::vector<ge::InControlAnchorPtr> peerInControlFromOutData;
  std::vector<ge::InControlAnchorPtr> peerInControlAnchor;
  s32 rank;
  s32 rankSize;
  std::string group;
  std::string nodeName;

  alltoallvcNodeInfo()
      : peerOutDataAnchor(0),
        peerOutDataToInControl(0),
        peerOutControlAnchor(0),
        peerInDataAnchor(0),
        peerInControlFromOutData(0),
        peerInControlAnchor(0),
        rank(-1),
        rankSize(-1),
        group(""),
        nodeName("") {}
};

// 记录alltoallvc融合后新增的所有节点
using AlltoAllVCFusionNodesInfo = struct alltoallvcFusionNodesInfo {
  std::vector<ge::NodePtr> sendCountSplits;
  std::vector<ge::NodePtr> sendDataSplitVs;
  std::vector<ge::NodePtr> recvCountSplits0;  // n*n -> 1*n
  std::vector<ge::NodePtr> recvCountSplits1;  // 1*n -> 1*1
  std::vector<ge::NodePtr> recvDataConcats;
  ge::NodePtr sendDataConcatNodePtr;
  ge::NodePtr recvCountConcatNodePtr;
  ge::NodePtr sendCountMatrixAddNNodePtr;
  ge::NodePtr alltoallvcFusionNodePtr;
  ge::NodePtr recvDataSplitVNodePtr;

  alltoallvcFusionNodesInfo()
      : sendCountSplits(0),
        sendDataSplitVs(0),
        recvCountSplits0(0),
        recvCountSplits1(0),
        recvDataConcats(0),
        sendDataConcatNodePtr(nullptr),
        recvCountConcatNodePtr(nullptr),
        sendCountMatrixAddNNodePtr(nullptr),
        alltoallvcFusionNodePtr(nullptr),
        recvDataSplitVNodePtr(nullptr) {}
};

// Split 算子信息
using SplitNodeInfo = struct splitNodeInfo {
  ge::NodePtr splitNodePtr;
  std::string nodeName;
  ge::GeTensorDesc inputSplitDim;
  ge::GeTensorDesc inputX;
  std::vector<ge::GeTensorDesc> outputY;
  s32 numSplit;
};

// SplitV 算子信息
using SplitVNodeInfo = struct splitvNodeInfo {
  ge::NodePtr splitvNodePtr;
  std::string nodeName;
  ge::GeTensorDesc inputX;
  ge::GeTensorDesc inputSizeSplit;
  ge::GeTensorDesc inputSplitDim;
  std::vector<ge::GeTensorDesc> outputY;
  s32 numSplit;
};

// Concat 算子信息
using ConcatNodeInfo = struct concatNodeInfo {
  ge::NodePtr concatNodePtr;
  std::string nodeName;
  ge::GeTensorDesc inputConcatDim;
  std::vector<ge::GeTensorDesc> inputX;
  ge::GeTensorDesc outputY;
  s32 N;
};

class HcomAlltoAllVCFusion : public OpFusionBase {
 public:
  HcomAlltoAllVCFusion();
  ~HcomAlltoAllVCFusion() override;
  HcclResult Run(ge::ComputeGraph &graph) override;

 private:
  HcclResult GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos);
  HcclResult GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos);
  HcclResult GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption);
  HcclResult GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel);
  HcclResult FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection);

  HcclResult RunFusionOpsAlltoAllVC(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps);
  using OpFusionBase::RemoveOpsEdges;
  // 记录每个alltoallvc算子的输入数据边、输入控制边、输出数据边、输出控制边, 保存后删除原alltoallvc节点
  HcclResult RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
                            std::vector<AlltoAllVCNodeInfo> &nodeInfos, ge::OpDescPtr &fusedOp);
  using OpFusionBase::GetPeerOutDataToInData;
  HcclResult GetPeerOutDataToInData(std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec, ge::NodePtr &srcNodePtr);
  using OpFusionBase::GetPeerOutDataToInControl;
  HcclResult GetPeerOutDataToInControl(vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
                                       ge::NodePtr &srcNodePtr);
  using OpFusionBase::GetPeerOutControlToInControl;
  HcclResult GetPeerOutControlToInControl(vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec,
                                          ge::NodePtr &srcNodePtr);
  using OpFusionBase::GetPeerAnchorFromOutData;
  HcclResult GetPeerAnchorFromOutData(std::vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
                                      std::vector<ge::InControlAnchorPtr> &peerInControlFromOutDataVec,
                                      ge::NodePtr &srcNodePtr);
  using OpFusionBase::GetPeerInDataAnchorFromOutData;
  HcclResult GetPeerInDataAnchorFromOutData(std::vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
                                            ge::OutDataAnchorPtr outDataAnchor, ge::NodePtr &srcNodePtr);
  using OpFusionBase::GetPeerInControlAnchorFromOutData;
  HcclResult GetPeerInControlAnchorFromOutData(std::vector<ge::InControlAnchorPtr> &peerInControlFromOutDataVec,
                                               ge::OutDataAnchorPtr outDataAnchor, ge::NodePtr &srcNodePtr);
  using OpFusionBase::GetPeerInControlFromOutControl;
  HcclResult GetPeerInControlFromOutControl(vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec,
                                            ge::NodePtr &srcNodePtr);
  HcclResult GetAlltoAllVCOpInfo(s32 &rank, s32 &rankSize, string &group, std::string &nodeName,
                                 ge::NodePtr &srcNodePtr);

  using OpFusionBase::AddFusionNode;
  // 创建节点, 将节点添加到graph中, 并添加数据边: peerOutDataAnchor, peerInDataAnchor
  HcclResult AddFusionNode(ge::ComputeGraph &graph, std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                           AlltoAllVCFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp);
  HcclResult AddSendCountSplit(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo);
  HcclResult AddSendDataSplitV(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo, ge::NodePtr &peerOutDataToSizeSplitNodePtr);
  HcclResult AddRecvCountSplit(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo);
  HcclResult AddSendDataConCat(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo);
  HcclResult AddAddN(ge::ComputeGraph &graph, std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                     AlltoAllVCFusionNodesInfo &fusionNodesInfo);
  HcclResult AddAlltoAllVCNode(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp);
  HcclResult AddRecvCountConCat(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                                AlltoAllVCFusionNodesInfo &fusionNodesInfo);
  HcclResult AddRecvDataSplitV(ge::ComputeGraph &graph, AlltoAllVCNodeInfo &nodeInfo,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo);
  HcclResult AddRecvDataConCat(ge::ComputeGraph &graph, std::vector<AlltoAllVCNodeInfo> &nodeInfos,
                               AlltoAllVCFusionNodesInfo &fusionNodesInfo);

  using OpFusionBase::RestoreOpsEdges;
  // 恢复控制边: peerOutDataToInControl, peerOutControlAnchor, peerInControlFromOutData, peerInControlAnchor
  HcclResult RestoreOpsEdges(std::vector<AlltoAllVCNodeInfo> &nodeInfos, AlltoAllVCFusionNodesInfo &fusionNodesInfo);

  HcclResult CreateSplitNode(SplitNodeInfo &splitNodeInfo, ge::ComputeGraph &graph);
  HcclResult CreateSplitVNode(SplitVNodeInfo &splitvNodeInfo, ge::ComputeGraph &graph);
  HcclResult CreateConcatNode(ConcatNodeInfo &concatNodeInfo, ge::ComputeGraph &graph);
  HcclResult CreateConstNode(ge::NodePtr &nodePtr, std::string nodeName, std::vector<int32_t> nodeValue,
                             std::vector<int64_t> dim, ge::ComputeGraph &graph);
  HcclResult AddOpsEdge(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst);
  HcclResult CheckAlltoAllVCNodeInfo(std::vector<AlltoAllVCNodeInfo> &nodeInfos);
  HcclResult SetUnknownShape(ge::NodePtr &nodePtr, ge::ComputeGraph &graph);
};
}  // namespace hccl
#endif
