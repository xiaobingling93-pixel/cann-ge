/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_broadcast_fusion.h"
#include "hcom_ops_kernel_info_store.h"
#include "hcom_op_utils.h"
#include "hccl/hcom.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"

using namespace hccl;
using namespace std;

namespace hccl {
HcomBroadcastFusion::HcomBroadcastFusion() : fusionTensorSizeLimit_(0) {}

HcomBroadcastFusion::~HcomBroadcastFusion() {}

HcclResult HcomBroadcastFusion::Run(ge::ComputeGraph &graph, uint64_t fusionTensorSizeLimit) {
  HcclResult ret;
  FusionInfos fusionInfos;
  fusionTensorSizeLimit_ = fusionTensorSizeLimit;
  ret = GetFusionOps(graph, fusionInfos);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Run][BroadcastFusion]graph[%s]: get fusion Broadcast ops failed. ret[%d]",
                         graph.GetName().c_str(), ret),
              ret);

  HCCL_INFO("there are %u fusion section to be fused in graph[%s].", fusionInfos.size(), graph.GetName().c_str());
  // The number of HcomBroadcast operator must be more than 1
  CHK_PRT_RET((fusionInfos.size() == 0),
              HCCL_INFO("NOT_CHANGED: graph[%s] has no HcomBroadcast op to be fused.", graph.GetName().c_str()),
              HCCL_SUCCESS);
  for (auto iterFusionInfos = fusionInfos.begin(); iterFusionInfos != fusionInfos.end(); iterFusionInfos++) {
    HCCL_INFO("graph[%s] fusionlabel[%s]: there are %zu HcomBroadcast ops to be fused.", graph.GetName().c_str(),
              iterFusionInfos->first.c_str(), iterFusionInfos->second.size());
    ret = FuseOps(graph, iterFusionInfos->second);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][BroadcastFusion]graph[%s] fusionlabel[%s]: fusion HcomBroadcast ops"
                           "failed. ret[%d]",
                           graph.GetName().c_str(), iterFusionInfos->first.c_str(), ret),
                ret);
  }
  HCCL_INFO("fuse HcomBroadcast op end");
  return HCCL_SUCCESS;
}

HcclResult HcomBroadcastFusion::FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) {
  HcclResult ret;
  CHK_PRT_RET((fusionSection.size() <= 1),
              HCCL_INFO("NOT_CHANGED: the section has %u HcomBroadcast op.", fusionSection.size()), HCCL_SUCCESS);

  std::vector<uint32_t> segmentIndex;
  CHK_RET(GetFusionSegments(fusionSection, segmentIndex));

  HCCL_INFO("the section has %u HcomBroadcast Ops, will be fusion to %u ops.", fusionSection.size(),
            segmentIndex.size());

  ret = RunFusionOps(graph, fusionSection, segmentIndex.size(), segmentIndex);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Ops][Fuse]graph[%s]: RunFusionOps failed. ret[%d]", graph.GetName().c_str(), ret), ret);
  return HCCL_SUCCESS;
}

HcclResult HcomBroadcastFusion::GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos) {
  for (auto nodePtr : graph.GetDirectNode()) {
    if (!nodePtr) {
      HCCL_WARNING("HcomBroadcastFusion: null node exists.");
      continue;
    }
    auto opDescPtr = nodePtr->GetOpDesc();
    if (!opDescPtr) {
      HCCL_WARNING("HcomBroadcastFusion: desc of node[%s] is null.", nodePtr->GetName().c_str());
      continue;
    }
    if (opDescPtr->GetType() == HCCL_KERNEL_OP_TYPE_BROADCAST) {
      CHK_RET(GetFusionOpInfo(nodePtr, fusionInfos));
    }
  }
  return HCCL_SUCCESS;
}

HcclResult HcomBroadcastFusion::GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption) {
  auto opDescPtr = nodePtr->GetOpDesc();
  string nodeName = nodePtr->GetName();
  bool bErr = false;
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION, fusionOption.fusionAttr)) {
    fusionOption.fusionAttr = HCOM_ATTR_FUSION_NO_FUSION;
    HCCL_WARNING("node[%s] has no attr[%s], use default value[%lld], no fusion.", nodeName.c_str(),
                 HCOM_ATTR_NAME_FUSION.c_str(), fusionOption.fusionAttr);
  }
  // 如果没有设置fusionid，会将该值设置为-1,不进行融合
  if (!ge::AttrUtils::GetInt(opDescPtr, HCOM_ATTR_NAME_FUSION_ID, fusionOption.fusionId)) {
    fusionOption.fusionAttr = HCOM_ATTR_FUSION_NO_FUSION;
    fusionOption.fusionId = HCOM_ATTR_FUSION_ID_DEFAULT;
    HCCL_WARNING("node[%s] has no attr[%s], use default value[%lld], no fusion.", nodeName.c_str(),
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
          HCCL_ERROR("errNo[0x%016llx] node[%s] fusion[%lld] fusion_id[%lld]: fusion_id is "
                     "incorrect",
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

  // 判断root属性是否包含
  bErr = ge::AttrUtils::GetInt(opDescPtr, "root_rank", fusionOption.root);
  CHK_PRT_RET(
      !bErr,
      HCCL_ERROR("[Get][FusionOption]errNo[0x%016llx] get attr \"root_rank\" failed. ", HCOM_ERROR_CODE(HCCL_E_PARA)),
      HCCL_E_PARA);
  CHK_RET(HcomOpUtils::GetDataType(opDescPtr, fusionOption.dtype));

  fusionOption.optype = nodePtr->GetOpDesc()->GetType();
  fusionOption.reduction = "NA";
  return HCCL_SUCCESS;
}

HcclResult HcomBroadcastFusion::GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel) {
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
    HCCL_DEBUG("[HcclCommGraph][Type]GenerateFusionLabel.");
  }
  return HCCL_SUCCESS;
}

HcclResult HcomBroadcastFusion::GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos) {
  bool bUnknownShapeNode = false;
  CHK_PRT_RET((ge::NodeUtils::GetNodeUnknownShapeStatus(*nodePtr, bUnknownShapeNode) != ge::GRAPH_SUCCESS),
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get node unknown status failed", nodePtr->GetName().c_str()),
              HCCL_E_PARA);

  CHK_PRT_RET(bUnknownShapeNode, HCCL_INFO("node[%s] is unknown shape, no fusion", nodePtr->GetName().c_str()),
              HCCL_SUCCESS);
  FusionOption fusionOption;
  HcclResult ret = GetFusionOption(nodePtr, fusionOption);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] get fusion config failed", nodePtr->GetName().c_str()), ret);
  CHK_PRT_RET(
      fusionOption.fusionAttr == 0,
      HCCL_INFO("node[%s] with attr fusion[%lld], no fusion", nodePtr->GetName().c_str(), fusionOption.fusionAttr),
      HCCL_SUCCESS);

  HCCL_DEBUG("get fusion op: node[%s]: group[%s], fusion[%lld], fusion_id[%lld], dtype[%s], root[%lld]",
             nodePtr->GetName().c_str(), fusionOption.group.c_str(), fusionOption.fusionAttr, fusionOption.fusionId,
             fusionOption.dtype.c_str(), fusionOption.root);

  std::string fusionLabel;
  ret = GenerateFusionLabel(fusionOption, fusionLabel);
  CHK_PRT_RET(ret != HCCL_SUCCESS,
              HCCL_ERROR("[Get][FusionOpInfo]node[%s] generate fusion label failed.", nodePtr->GetName().c_str()), ret);
  HCCL_DEBUG("node[%s] generate fusion label[%s]", nodePtr->GetName().c_str(), fusionLabel.c_str());

  auto opDescPtr = nodePtr->GetOpDesc();
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

HcclResult HcomBroadcastFusion::GetFusionSegments(const FusionSection &fusionSection,
                                                  std::vector<uint32_t> &segmentIndex) {
  uint64_t currentSegmentTensorSize = 0;
  for (uint32_t i = 0; i < fusionSection.size(); i++) {
    uint64_t inputTensorSize = 0;
    auto opDescPtr = fusionSection[i]->GetOpDesc();
    HcclResult ret = HcomOpUtils::GetAllInputsTensorOriginSize(opDescPtr, inputTensorSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Get][FusionSegments]node[%s] get input tensor size failed.", opDescPtr->GetName().c_str()),
                ret);
    CHK_PRT_RET(((INVALID_U64 - inputTensorSize) < currentSegmentTensorSize),
                HCCL_ERROR("[Get][FusionSegments]node[%s] fusion failed, fusion tensor size is overflow.",
                           opDescPtr->GetName().c_str()),
                ret);
    currentSegmentTensorSize += inputTensorSize;
    HCCL_DEBUG("index[%u]: node[%s] input size:%llu, current segment tensor size %llu", i,
               fusionSection[i]->GetName().c_str(), inputTensorSize, currentSegmentTensorSize);
    if (currentSegmentTensorSize > fusionTensorSizeLimit_) {
      // 如果加上该算子导致该 segment 超过 limit 值，需要将该算子之前的算子进行融合
      if (currentSegmentTensorSize > inputTensorSize) {
        HCCL_DEBUG("fusion segment[%u]: end position[%u], fusion tensor size[%llu]", segmentIndex.size(), (i - 1),
                   (currentSegmentTensorSize - inputTensorSize));
        segmentIndex.push_back(i - 1);
        currentSegmentTensorSize = inputTensorSize;
      }
      if (currentSegmentTensorSize >= fusionTensorSizeLimit_) {
        // 如果该 node 的 tensor size 超过 limit 值，则需要将该算子之前的算子进行融合，该算子不融合
        HCCL_DEBUG("fusion segment[%u]: end position[%u], fusion tensor size[%llu]", segmentIndex.size(), i,
                   currentSegmentTensorSize);
        HCCL_WARNING(
            "fusion segment[%u]: input tensor size[%llu] is over fusion tensor size limit[%llu], "
            "because the node[%s] input tensor size is %llu. then the node will not be fused.",
            segmentIndex.size(), currentSegmentTensorSize, fusionTensorSizeLimit_, opDescPtr->GetName().c_str(),
            inputTensorSize);
        segmentIndex.push_back(i);
        currentSegmentTensorSize = 0;
      }
    } else if (currentSegmentTensorSize == fusionTensorSizeLimit_) {
      HCCL_DEBUG("fusion segment[%u]: end position[%u], fusion tensor size[%llu]", segmentIndex.size(), i,
                 currentSegmentTensorSize);
      segmentIndex.push_back(i);
      currentSegmentTensorSize = 0;
    } else {
      // do nothing.
    }
  }
  if ((segmentIndex.empty()) || (segmentIndex.back() != (fusionSection.size() - 1))) {
    HCCL_DEBUG("fusion segment[%u]: end position[%u], fusion tensor size[%llu]", segmentIndex.size(),
               (fusionSection.size() - 1), currentSegmentTensorSize);
    segmentIndex.push_back(fusionSection.size() - 1);
  }
  return HCCL_SUCCESS;
}
}  // namespace hccl
