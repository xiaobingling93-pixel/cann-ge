/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_GRAPH_OPTIMIZER_H
#define HCOM_GRAPH_OPTIMIZER_H

#include <string>
#include <map>
#include <vector>
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl/hcom.h"

namespace hccl {
const string HCCL_GRAPH_OPTIMIZER_NAME = "hccl_graph_optimizer";
constexpr std::int64_t HCCL_FORMAT_PAIRED_INPUT_OUTPUT = 2;  // 输入输出地格式须相同

using GroupParaLabel = std::map<std::string, std::string>;

class HcomGraphOptimizer : public ge::GraphOptimizer {
 public:
  HcomGraphOptimizer();
  ~HcomGraphOptimizer() override;
  virtual ge::Status Initialize(const map<std::string, std::string> &options,
                                ge::OptimizeUtility *const optimizeUtility) override;
  // close graphOptimizer
  ge::Status Finalize() override;
  // optimize original graph for FE quant optimize
  ge::Status OptimizeGraphPrepare(ge::ComputeGraph &graph) override;
  // optimize original graph, using in graph preparation stage
  ge::Status OptimizeOriginalGraph(ge::ComputeGraph &graph) override;
  // optimize fused graph
  ge::Status OptimizeSubgraphPreProc(ge::ComputeGraph &graph) override;
  ge::Status OptimizeFusedGraph(ge::ComputeGraph &graph) override;
  // optimize whole graph, using after graph merged stage
  ge::Status OptimizeWholeGraph(ge::ComputeGraph &graph) override;
  // get attribute of graph optimizer
  ge::Status GetAttributes(ge::GraphOptimizerAttribute &attrs) const override;

 protected:
  virtual HcclResult CheckSupportedOP(const std::string &sCollectiveType) const;
  virtual HcclResult CalcOpRunningParam(ge::Node &node, bool uknownShapeGraph);
  virtual HcclResult SetOpOutputMemSize(ge::Node &node, const std::string &sCollectiveType);
  virtual HcclResult CalcHCCLOutputMemSize(const std::string &sCollectiveType, int64_t &memSize);
  virtual HcclResult SetOpMemAttr(ge::Node &node, const std::string &sCollectiveType, const u64 &opMemSize);
  HcclResult HcomGraphOptimizeInitialize(const map<std::string, std::string> &options,
                                         ge::OptimizeUtility *const optimizeUtility);
  HcclResult HcomOptimizeOriginalGraph(ge::ComputeGraph &graph, bool &uknownShapeGraph);
  HcclResult OriginalGraphShapeTypeCfg(ge::ComputeGraph &graph, bool &uknownShapeGraph);
  HcclResult SetUnknownShapeAttr(ge::ComputeGraph &graph, bool uknownShapeGraph);
  HcclResult UpdateFusionTensorSizeLimit(bool unknownShape, u64 &fusionTensorSize);

 private:
  HcclResult FuseHcomAllReduceNode(ge::ComputeGraph &graph);
  HcclResult FuseHcomBroadcastNode(ge::ComputeGraph &graph, u64 fusionTensorSize);
  HcclResult FuseHcomReduceNode(ge::ComputeGraph &graph);
  HcclResult SetHcomOpAttrs(ge::OpDescPtr &opDescPtr);
  HcclResult SetHcomOpFormat(ge::OpDescPtr &opDescPtr);
  HcclResult SetHcomOpParallelLabel(ge::Node &node, std::string groupLabel);
  HcclResult GetCountFromOpDesc(const ge::OpDescPtr &op, const std::string &sCollectiveType, HcclDataType dataType,
                                u64 &count);
  HcclResult GetCommFromOpDesc(const ge::OpDescPtr &op, int64_t &hcomComm, std::string &sGroup);
  HcclResult SetOpAtomicInputIndex(ge::Node &node, const std::string &sCollectiveType);
  HcclResult GetHcomReceiveOpOutputSize(const ge::OpDescPtr &op, u32 dataTypeSize, u64 &outputSize);
  HcclResult HcomCalcOpRunningParam(ge::Node &node, bool uknownShapeGraph);
  HcclResult SetHcomOpParam(const ge::Node &node, HcomOpParam *hcomOpParam, std::string &sCollectiveType,
                            std::string &sGroup, std::string &socVersion, std::vector<int64_t> &sendCountMatrix,
                            std::vector<int64_t> &sendCounts, std::vector<int64_t> &sendDispls,
                            std::vector<int64_t> &recvCounts, std::vector<int64_t> &recvDispls,
                            std::vector<u32> &curRanks, std::string &rankTableStr, std::string &rankTableM);
  HcclResult SetOpWorkerSpaceForKnowShape(ge::Node &node, u64 &opMemSize);
  HcclResult GetOriginalGraphShapeTypeFromDesc(const ge::OpDescPtr &op, u32 &shapeType);
  HcclResult CheckForceUnknown(const ge::Node &node, u32 &taskNum);

  bool IsSubgraphMultiBatch(ge::ComputeGraph &graph);

  uint64_t fusionTensorSizeLimit_;
  int32_t hcomMultiMode_;
  int32_t optionFeatureBaseRefreshable_;
};
}  // namespace hccl
#endif