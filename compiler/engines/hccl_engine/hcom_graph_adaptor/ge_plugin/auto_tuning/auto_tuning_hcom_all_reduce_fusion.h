/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_HCOM_ALL_REDUCE_FUSION_H
#define AUTO_TUNING_HCOM_ALL_REDUCE_FUSION_H

#include "hccl/hccl_types.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "op_fusion_base_pub.h"
#include "hcom_all_reduce_fusion.h"
#include "platform/platform_info.h"
#include "nlohmann/json.hpp"

namespace hccl {
using GradientDataInfo = struct GradientDataInfoDef {
  int64_t dataSize;
  std::string dataType;
  uint32_t graphId;
  std::string groupName;
  std::string gradientNodeName;
  std::string traceNodeName;
  std::string allReduceNodeName;
};

class AutoTuningHcomAllReduceFusion : public HcomAllReduceFusion {
 public:
  AutoTuningHcomAllReduceFusion();
  ~AutoTuningHcomAllReduceFusion() override;
  HcclResult Run(ge::ComputeGraph &graph, std::vector<GradientDataInfo> &recordInfos);

 protected:
  HcclResult FuseOps(ge::ComputeGraph &graph, FusionSection &fusionSection) override;
  HcclResult GetGradSplitStrategy(const std::string &modelName, const std::string &sGroup,
                                  const std::vector<ge::NodePtr> &fusionOps, u32 &segmentNum,
                                  std::vector<u32> &segmentIndex) override;

 private:
  HcclResult RecordGradientDataInfo(ge::ComputeGraph &graph, std::vector<GradientDataInfo> &recordInfos);
  HcclResult GetGroupName(const ge::OpDescPtr &op, std::string &group);
  HcclResult GetDataTypeName(const ge::DataType dataType, std::string &dataTypeName);
  HcclResult GetGradientDataInfo(ge::ComputeGraph &graph, ge::NodePtr &nodePtr, GradientDataInfo &gradientNodeInfo);
  HcclResult AddTraceNode(ge::ComputeGraph &graph, ge::NodePtr &nodePtr, const GradientDataInfo gradientNodeInfo);
  HcclResult SetGradientInformation(ge::ComputeGraph &graph);
  HcclResult AddFusionMapInFusionJson(const std::string &fusionHash);
  HcclResult SetFusionModelInLibrary(std::string &workPath, const std::string &fusionHash);
  HcclResult GetFusionWorkPath(std::string &fusionPath);
  HcclResult CreateFusionDir(std::string &dir);
  bool AutoTuneTargetNode(ge::NodePtr nodePtr);
  bool isNotFoundHash_;
};
}  // namespace hccl
#endif
