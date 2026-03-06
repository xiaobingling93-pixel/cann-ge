/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_FUSION_OPTIMIZER_H
#define HCOM_FUSION_OPTIMIZER_H

#include <string>
#include <map>
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hcom_log.h"
#include "op_hcom_comm.h"

namespace hccl {
const std::string HCCL_FUSION_OPTIMIZER_NAME = "hccl_alltoallvc_fusion_optimizer";

class HcomFusionOptimizer : public ge::GraphOptimizer {
 public:
  HcomFusionOptimizer();
  ~HcomFusionOptimizer() override;
  virtual ge::Status Initialize(const std::map<std::string, std::string> &options,
                                ge::OptimizeUtility *const optimizeUtility) override;
  // close graphOptimizer
  ge::Status Finalize() override;
  // optimize original graph for FE quant optimize
  ge::Status OptimizeGraphPrepare(ge::ComputeGraph &graph) override;
  // optimize original graph, using in graph preparation stage
  ge::Status OptimizeOriginalGraph(ge::ComputeGraph &graph) override;
  // optimize fused graph
  ge::Status OptimizeFusedGraph(ge::ComputeGraph &graph) override;
  // optimize whole graph, using after graph merged stage
  ge::Status OptimizeWholeGraph(ge::ComputeGraph &graph) override;
  // get attribute of graph optimizer
  ge::Status GetAttributes(ge::GraphOptimizerAttribute &attrs) const override;

 protected:
  HcclResult HcomOptimizeOriginalGraph(ge::ComputeGraph &graph);

 private:
  HcclResult FuseHcomAlltoAllVCNode(ge::ComputeGraph &graph);
  HcclResult FuseHcomAllgatherNode(ge::ComputeGraph &graph);
  HcclResult FuseHcomReduceScatterNode(ge::ComputeGraph &graph);
  HcclResult HcomOptimizeSetAttr(ge::ComputeGraph &graph);
};
}  // namespace hccl
#endif