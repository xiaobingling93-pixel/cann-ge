/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPHENGINE_TESTS_UT_GE_RUNTIME_DEPLOY_STUB_MODELS_H_
#define GRAPHENGINE_TESTS_UT_GE_RUNTIME_DEPLOY_STUB_MODELS_H_

#include "ge/ge_ir_build.h"
#include "graph/compute_graph.h"
#include "common/model/ge_root_model.h"
#include "dflow/base/deploy/deploy_planner.h"
#include "dflow/inc/data_flow/model/flow_model.h"

using namespace std;

namespace ge {
class StubModels {
 public:

  static ComputeGraphPtr BuildSinglePartitionedCallGraph();

  /**
   *      NetOutput
   *         |
   *         |
   *        PC_2
   *        |  \.
   *       PC_1 |
   *     /     \.
   *    |      |
   *  data1  data2
   */
  static ComputeGraphPtr BuildGraphWithQueueBindings();

  /**
   *  NetOutput
   *     |
   *    PC_2
   *     |
   *    PC_1
   *     |
   *   data1
   */
  static ComputeGraphPtr BuildSeriesPartitionedCallGraph();

  /**
   *   NetOutput
   *    /   \(控制边)
   *  PC_1   PC_2
   *    \   /
   *   data1
   */
  static ComputeGraphPtr BuildParallelPartitionedCallGraph();
  
  /**
   *   NetOutput
   *    /   \
   *  PC_1   PC_2
   *    \   /
   *   data1
   */
  static ComputeGraphPtr BuildParallelPartitionedCallGraphWithMultiOutputs();

  /**
   *     NetOutput
   *         |
   *       PC_3
   *      /   \
   *    PC_1  PC2
   *    |      |
   *  data1  data2
   */
  static ComputeGraphPtr BuildGraphWithoutNeedForBindingQueues(const std::string &prefix = "");

  static GeRootModelPtr BuildGeRootModel(ComputeGraphPtr root_graph);
  static PneModelPtr BuildRootModel(ComputeGraphPtr root_graph, bool pipeline_partitioned = true);
  static FlowModelPtr BuildFlowModel(ComputeGraphPtr root_graph, bool pipeline_partitioned = true);
  static Status SaveGeRootModelToModelData(const GeRootModelPtr &ge_root_model, ModelData &model_data, ModelBufferData &model_buffer_data);

  static void BuildModel(Graph &graph, ModelBufferData &model_buffer_data);

  /**
   *    NetOutput (4)       8->4(input_group)
   *        |
   *      PC_2 (2)/(3)      3->7(output_group)
   *        |
   *      PC_1 (1)/(2)     6(input_group) ->1
   *        |
   *      data1(0)  0->5(output_group)
   */
  static DeployPlan BuildSimpleDeployPlan(int32_t remote_node_id = 1);

  static DeployPlan BuildSingleModelDeployPlan(int32_t remote_node_id = 1);
  static DeployPlan BuildSingleModelDeployPlanWithDummyQ(int32_t remote_node_id = 1);
  static DeployPlan BuildSingleModelWithFileConstDeployPlan(const std::string &location, int32_t remote_node_id = 1);
  static DeployPlan BuildSingleModelDeployPlanWithProxy(int32_t remote_node_id = 1);
 private:
  static void InitGeLib();
  static void FinalizeGeLib();
};
}  // namespace ge
#endif // GRAPHENGINE_TESTS_UT_GE_RUNTIME_DEPLOY_STUB_MODELS_H_
