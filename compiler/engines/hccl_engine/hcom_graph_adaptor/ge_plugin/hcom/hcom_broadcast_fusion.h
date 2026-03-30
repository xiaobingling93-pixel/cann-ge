/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_BROADCAST_FUSION_H
#define HCOM_BROADCAST_FUSION_H

#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "op_fusion_base_pub.h"

namespace hccl {

class HcomBroadcastFusion : public OpFusionBase {
 public:
  HcomBroadcastFusion();
  ~HcomBroadcastFusion() override;
  using OpFusionBase::Run;
  HcclResult Run(ge::ComputeGraph &graph, uint64_t fusionTensorSizeLimit);

 private:
  HcclResult GetFusionOps(ge::ComputeGraph &graph, FusionInfos &fusionInfos);
  HcclResult GetFusionOpInfo(ge::NodePtr &nodePtr, FusionInfos &fusionInfos);
  HcclResult GetFusionOption(const ge::NodePtr &nodePtr, FusionOption &fusionOption);
  HcclResult GenerateFusionLabel(const FusionOption &fusionOption, std::string &fusionLabel);
  HcclResult FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection);
  HcclResult GetFusionSegments(const FusionSection &fusionSection, std::vector<uint32_t> &segmentIndex);
  uint64_t fusionTensorSizeLimit_;
};
}  // namespace hccl
#endif
