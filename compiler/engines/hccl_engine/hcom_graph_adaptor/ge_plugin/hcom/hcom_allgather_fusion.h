/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNTITLED4_HCOM_ALLGATHER_FUSION_H
#define UNTITLED4_HCOM_ALLGATHER_FUSION_H

#include "hccl/base.h"
#include "hcom_alltoallvc_fusion.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "op_fusion_base_pub.h"

namespace hccl {
// 记录单个AllGather算子的数据边、控制边、属性
using CommonNodeInfo = struct commonNodeInfo {
  std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
  std::vector<ge::OutDataAnchorPtr> peerOutDataToInControl;
  std::vector<ge::OutControlAnchorPtr> peerOutControlAnchor;
  std::vector<ge::InDataAnchorPtr> peerInDataAnchor;
  std::vector<ge::InControlAnchorPtr> peerInControlFromOutData;
  std::vector<ge::InControlAnchorPtr> peerInControlAnchor;
  s32 rankSize;
  std::string group;
  std::string nodeName;

  commonNodeInfo()
      : peerOutDataAnchor(0),
        peerOutDataToInControl(0),
        peerOutControlAnchor(0),
        peerInDataAnchor(0),
        peerInControlFromOutData(0),
        peerInControlAnchor(0),
        rankSize(-1),
        group(""),
        nodeName("") {}
};

// 记录alltgather融合后新增的所有节点
using AllGatherFusionNodesInfo = struct allgatherFusionNodesInfo {
  ge::NodePtr allgatherFusionNodePtr;
  std::vector<ge::NodePtr> recvDataConcats;

  allgatherFusionNodesInfo() : allgatherFusionNodePtr(nullptr), recvDataConcats(0) {}
};

class HcomAllGatherFusion : public OpFusionBase {
 public:
  HcomAllGatherFusion();
  ~HcomAllGatherFusion() override;
  HcclResult Run(ge::ComputeGraph &graph) override;

 private:
  HcclResult GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos);
  HcclResult GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos);
  HcclResult GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption);
  HcclResult GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel);
  HcclResult FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection);

  HcclResult RunFusionOpsAllGather(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps);
  // 记录每个allgather算子的输入数据边、输入控制边、输出数据边、输出控制边, 保存后删除原allgather节点
  HcclResult RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
                            std::vector<CommonNodeInfo> &nodeInfos, ge::OpDescPtr &fusedOp);
  HcclResult GetPeerOutDataToInData(std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec, ge::NodePtr &srcNodePtr);
  HcclResult GetPeerOutDataToInControl(vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
                                       ge::NodePtr &srcNodePtr);
  HcclResult GetPeerOutControlToInControl(vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec,
                                          ge::NodePtr &srcNodePtr);
  HcclResult GetPeerAnchorFromOutData(std::vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
                                      std::vector<ge::InControlAnchorPtr> &peerInControlFromOutDataVec,
                                      ge::NodePtr &srcNodePtr);
  HcclResult GetPeerInDataAnchorFromOutData(std::vector<ge::InDataAnchorPtr> &peerInDataFromOutDataVec,
                                            ge::OutDataAnchorPtr outDataAnchor, ge::NodePtr &srcNodePtr);
  HcclResult GetPeerInControlAnchorFromOutData(std::vector<ge::InControlAnchorPtr> &peerInControlFromOutDataVec,
                                               ge::OutDataAnchorPtr outDataAnchor, ge::NodePtr &srcNodePtr);
  HcclResult GetPeerInControlFromOutControl(vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec,
                                            ge::NodePtr &srcNodePtr);
  HcclResult GetAllGatherOpInfo(s32 &rankSize, string &group, std::string &nodeName, ge::NodePtr &srcNodePtr);

  // 创建节点, 将节点添加到graph中, 并添加数据边: peerOutDataAnchor, peerInDataAnchor
  HcclResult AddFusionNode(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                           AllGatherFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp);
  HcclResult AddAllGatherNode(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfo,
                              AllGatherFusionNodesInfo &fusionNodesInfo, ge::OpDescPtr &fusedOp);
  HcclResult AddRecvDataConCat(ge::ComputeGraph &graph, std::vector<CommonNodeInfo> &nodeInfos,
                               AllGatherFusionNodesInfo &fusionNodesInfo);

  // 恢复控制边: peerOutDataToInControl, peerOutControlAnchor, peerInControlFromOutData, peerInControlAnchor
  HcclResult RestoreOpsEdges(std::vector<CommonNodeInfo> &nodeInfos, AllGatherFusionNodesInfo &fusionNodesInfo);
  HcclResult AddOpsEdge(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst);
  HcclResult CreateConcatNode(ConcatNodeInfo &concatNodeInfo, ge::ComputeGraph &graph);
  HcclResult CreateConstNode(ge::NodePtr &nodePtr, std::string nodeName, std::vector<int32_t> nodeValue,
                             std::vector<int64_t> dim, ge::ComputeGraph &graph);
};
}  // namespace hccl
#endif
