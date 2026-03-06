/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_ALL_REDUCE_FUSION_H
#define HCOM_ALL_REDUCE_FUSION_H

#include "hccl/base.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "op_fusion_base_pub.h"
#include "platform/platform_info.h"
#include <nlohmann/json.hpp>

namespace hccl {

class HcomAllReduceFusion : public OpFusionBase {
 public:
  HcomAllReduceFusion();
  ~HcomAllReduceFusion() override;
  HcclResult Run(ge::ComputeGraph &graph) override;

 protected:
  virtual HcclResult FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection);
  virtual HcclResult GetGradSplitStrategy(const std::string &modelName, const std::string &sGroup,
                                          const FusionSection &fusionSection, u32 &segmentNum,
                                          std::vector<u32> &segmentIndex);
  HcclResult GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos);
  HcclResult GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos);
  HcclResult GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption);
  HcclResult GetFusionStrategy(const ge::ComputeGraph &graph, const FusionSection &fusionSection, u32 &segmentNum,
                               std::vector<u32> &segmentIndex);
  HcclResult GetNodeUnknownShapeInfo(ge::NodePtr &nodePtr, bool &bUnknownShapeNode);
  HcclResult AddHcclFusionFlag(ge::OpDescPtr &opDescPtr);
  HcclResult GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel);
  HcclResult GetFusionInformation(const ge::ComputeGraph &graph, std::string &fusionHash);
  HcclResult CalculateSegmentIndex(std::string &fusionHash, u64 tensorLimit, std::vector<u32> &segmentIndex);
  HcclResult GetPathFromDefault(std::string &fusionPath);
  HcclResult GetInformationFromLibrary(std::string &fusionPath, std::string &fusionHash, u64 tensorLimit,
                                       std::vector<u32> &segmentIndex);
  HcclResult GetInfoFromContentedLibrary(std::string fusionPath, std::string &fusionHash, u64 tensorLimit,
                                         std::vector<u32> &segmentIndex);
  HcclResult GetFusionOpsSlices(FusionInfos &fusionInfos, FusionInfos &fusionInfosTemp);

 private:
  bool bHasUnknownShapeNodeGraph_;
  bool unknownShapeOriginalGraph_;
  std::string fusionHash_;
  uint32_t modelGraphId;
  u64 tensorFusionLimit_;
};
}  // namespace hccl
#endif
