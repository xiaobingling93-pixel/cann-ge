/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING_HCOM_GRAPH_OPTIMIZER_H
#define AUTO_TUNING_HCOM_GRAPH_OPTIMIZER_H

#include <string>
#include <map>
#include "hcom_graph_optimizer.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "hccl/base.h"
#include "acl/acl_rt.h"

namespace hccl {

class AutoTuningHcomGraphOptimizer : public HcomGraphOptimizer {
 public:
  AutoTuningHcomGraphOptimizer();
  ~AutoTuningHcomGraphOptimizer() override;
  ge::Status Initialize(const std::map<std::string, std::string> &options,
                        ge::OptimizeUtility *const optimizeUtility) override;
  // optimize original graph, using in graph preparation stage
  ge::Status OptimizeOriginalGraph(ge::ComputeGraph &graph) override;
  ge::Status OptimizeFusedGraph(ge::ComputeGraph &graph) override;

 protected:
  HcclResult CheckSupportedOP(const std::string &sCollectiveType) const override;
  HcclResult CalcOpRunningParam(ge::Node &node);
  HcclResult SetOpOutputMemSize(ge::Node &node, const std::string &sCollectiveType) override;
  HcclResult CalcHCCLOutputMemSize(const std::string &sCollectiveType, int64_t &memSize) override;
  HcclResult SetOpMemAttr(ge::Node &node, const std::string &sCollectiveType, const u64 &opMemSize) override;
  HcclResult ParseProfilingConfig(bool &profilingMode, std::string &profilingOption);

 private:
  std::string workPath_;
  bool isGradientAutoTune_;
};
}  // namespace hccl

#endif