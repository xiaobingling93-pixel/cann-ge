/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef REDUCE_SCHDULE_CASE_GENERATOR_H
#define REDUCE_SCHDULE_CASE_GENERATOR_H

#include "ascir.h"
#include "ascgen_log.h"
#include "ascir_ops.h"
#include "task_generator/schedule_case_generator.h"

namespace optimize {

using ReduceType = std::variant<
    ge::ascir_op::Max,
    ge::ascir_op::Sum,
    ge::ascir_op::Min,
    ge::ascir_op::Prod,
    ge::ascir_op::Any,
    ge::ascir_op::All,
    ge::ascir_op::ArgMaxMultiRPhase1,
    ge::ascir_op::ArgMaxMultiRPhase2
>;

class ReducePartitionCaseGenerator : public FusionCaseGenerator {
public:
  Status Generate(ascir::HintGraph &graph,
                  std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;
  Status GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                       const OptimizerOptions &options) override;
  Status GenerateGeneralCase(ascir::HintGraph &graph,
                            std::vector<ascir::ImplGraph> &graphs,
                            std::vector<std::string> &score_functions);
  Status GenerateAllLoadCase(ascir::HintGraph &graph,
                            std::vector<ascir::ImplGraph> &graphs,
                            const std::vector<std::string> &score_functions);

private:
  Status GeneratorGeneralTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks);
  Status GeneratorAllLoadTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks);
  Status GeneratorRCoreTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks) const;
  Status ReducePartitionPostFusion(ascir::ImplGraph &impl_graph);
  Status PartitionByNode(ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph);
  bool IsInputNodePartitioned(const std::shared_ptr<ge::Node>& start, const std::shared_ptr<ge::Node>& node);
  Status FindNormLoop(const ge::AscNodePtr &start, std::vector<ge::AscNodePtr> &ends);
  bool IsNorm(const ge::AscNodePtr &start, const ge::AscNodePtr &end);
  Status PartitionNorm(ascir::ImplGraph &impl_graph, std::vector<std::pair<ge::AscNodePtr, ge::AscNodePtr>> &loop_start_end);
  void FindAllPath(const ge::AscNodePtr& start, const ge::AscNodePtr& end,
                       std::vector<ge::AscNodePtr> &path, std::vector<std::vector<ge::AscNodePtr>> &all_paths);
  static Status PartitionLoad(ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph);
  static Status PartitionScalar(ge::AscNodePtr &src_node, ge::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph);
  static bool HasReduce(const ascir::ImplGraph &impl_graph);
  static bool IsGroupGraphLegal(const ascir::ImplGraph &impl_graph);
  static bool CanReduceFuse(const ascir::ImplGraph &impl_graph);
  Status ReducePartitionMultipleCitations(ascir::ImplGraph &impl_graph);
  bool FindOutputReduce(const ge::AscNodePtr &node, ge::AscNodePtr &reduce_node);
  Status PartitionReduce(ge::AscNodePtr &src_node, ascir::ImplGraph &impl_graph);

  bool partition_{false};
  std::vector<ge::AscNodePtr> node_order_{};
};

class RMulticorePhase2Graph {
public:
  RMulticorePhase2Graph(ascir::ImplGraph &phase2graph, ascir::ImplGraph &phase1graph,
                       ascir::ImplGraph &phase_graph, ge::AscNodePtr reduce_node)
     : phase2graph(phase2graph),
       phase1graph(phase1graph),
       phase_graph(phase_graph),
       reduce_node(std::move(reduce_node)) {};
  RMulticorePhase2Graph(const RMulticorePhase2Graph&) = delete;
  RMulticorePhase2Graph& operator=(const RMulticorePhase2Graph&) = delete;
  Status Construct();
private:
  ge::Expression Rm_org_size;
  ge::Expression A_org_size;
  ascir::ImplGraph &phase2graph;
  ascir::ImplGraph &phase1graph;
  ascir::ImplGraph &phase_graph;
  ge::AscNodePtr reduce_node;
  Status CreateVarAxis();
  Status CompleteNodeAttr(ge::AscNodePtr &node, bool before_reduce, const ge::AscTensorDataType& data_type);
  Status CompletePhaseGraph(ReduceType &phase2graph_reduce);
  Status PartitionByReduce(ascir::ImplGraph &impl_graph, ReduceType &phase2graph_reduce,
                           std::vector<ge::AscNodePtr> &node_order);
  Status SetNodeOrder(std::vector<ge::AscNodePtr> &node_order);
  Status SetupArgMaxIndexNodes(const ge::AscNodePtr &reduce_node, ascir::ImplGraph &phase2graph);
};
}

#endif //REDUCE_SCHDULE_CASE_GENERATOR_H
