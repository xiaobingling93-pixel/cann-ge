/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_CASE_GENERATOR_H_
#define ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_CASE_GENERATOR_H_

#include "ascir.h"
#include "ascgen_log.h"
#include "ascir_utils.h"
#include "schedule_group_partitioner.h"
#include "schedule_task_generator.h"
#include "schedule_utils.h"

namespace optimize {
class FusionCaseGenerator {
 public:
  FusionCaseGenerator() = default;
  virtual ~FusionCaseGenerator() = default;
  FusionCaseGenerator(const FusionCaseGenerator&) = delete;
  FusionCaseGenerator &operator=(const FusionCaseGenerator&) = delete;
  virtual Status Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                          std::vector<std::string> &score_functions) = 0;
  virtual Status GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                               const OptimizerOptions &options) {
    (void)options;
    bool need_update_axis = false;
    GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(optimize_graph, need_update_axis));
    std::vector<ascir::ImplGraph> optimize_graphs;
    std::vector<std::string> score_funcs;
    GE_CHK_STATUS_RET(Generate(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
    score_funcs.resize(optimize_graphs.size());
    for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
      const auto &graph = optimize_graphs[i];
      ScheduleTask task{graph, {}, score_funcs[i]};
      task.has_load_store_conversion = HasLoadStoreConversion();
      GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs),
                        "Failed to partition graph");
      if (need_update_axis && task.grouped_graphs.size() > 1) {
        GE_ASSERT_SUCCESS(UpdateAxisSizes(task.grouped_graphs));
      }
      tasks.emplace_back(std::move(task));
    }
    return ge::GRAPH_SUCCESS;
  }

  static Status GroupPartitionAndGenTasks(const ascir::ImplGraph &impl_graph, std::vector<ScheduleTask> &tasks) {
    ::ascir::utils::DumpGraph(impl_graph, "BeforeGroupPartition");
    ScheduleTask new_schedule_task{impl_graph, {}, ""};
    GE_CHK_STATUS_RET(
        ScheduleGroupGraphPartitioner::PartitionByConnectivity(impl_graph, new_schedule_task.grouped_graphs),
        "Failed to partition graph [%s].", impl_graph.GetName().c_str());
    tasks.emplace_back(std::move(new_schedule_task));
    return ge::SUCCESS;
  }

  virtual bool HasLoadStoreConversion() const {
    return false;
  }

 private:
  static Status UpdateAxisSizes(std::vector<::ascir::ImplGraph> &grouped_graphs) {
    for (const auto &subgraph : grouped_graphs) {
      if (ScheduleUtils::FindFirstNodeOfType<ge::ascir_op::Concat>(subgraph) == nullptr) {
        GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::RefreshAxisSize(subgraph));
      }
    }
    return ge::SUCCESS;
  }
};
}  // namespace optimize
#endif  // ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_CASE_GENERATOR_H_
