/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "dflow/compiler/model/dflow_graph_manager.h"
#include "common/ge_common/ge_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge/graph/ops_stub.h"
#include "ge_common/ge_api_types.h"
#include "ge/ge_api.h"

namespace ge {
namespace {
void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}
}

class FlowGraphManagerTest : public testing::Test {
};

TEST_F(FlowGraphManagerTest, BasicInitializeFinalize) {
  std::map<std::string, std::string> options;
  DflowGraphManager dflow_graph_manager;
  // finalize without init
  dflow_graph_manager.Finalize();
  EXPECT_EQ(dflow_graph_manager.Initialize(options), SUCCESS);
  // init twice
  EXPECT_EQ(dflow_graph_manager.Initialize(options), SUCCESS);
}

TEST_F(FlowGraphManagerTest, OperateGraphWithoutInit) {
  DflowGraphManager dflow_graph_manager;
  std::map<std::string, std::string> options;
  Graph graph("test");
  EXPECT_EQ(dflow_graph_manager.AddGraph(1, graph, options), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(dflow_graph_manager.RemoveGraph(1), ACL_ERROR_GE_EXEC_NOT_INIT);
  const std::vector<GeTensor> inputs;
  EXPECT_EQ(dflow_graph_manager.CompileGraph(1, inputs), ACL_ERROR_GE_EXEC_NOT_INIT);
  uint32_t model_id = 0;
  EXPECT_EQ(dflow_graph_manager.GetGraphModelId(1, model_id), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(dflow_graph_manager.GetFlowModel(1), nullptr);
  EXPECT_EQ(dflow_graph_manager.GetOptionsRunGraphFlag(), false);
}

TEST_F(FlowGraphManagerTest, AddGraphTwice) {
  DflowGraphManager dflow_graph_manager;
  Graph graph("test");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  options.emplace(ge::OUTPUT_DATATYPE, "FP16");
  dflow_graph_manager.Initialize(options);
  EXPECT_EQ(dflow_graph_manager.AddGraph(1, graph, options), SUCCESS);
  EXPECT_EQ(dflow_graph_manager.AddGraph(1, graph, options), FAILED);
  dflow_graph_manager.Finalize();
}

TEST_F(FlowGraphManagerTest, CompileGraph) {
  DflowGraphManager dflow_graph_manager;
  Graph graph("test");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  options.emplace(ge::OUTPUT_DATATYPE, "FP16");
  dflow_graph_manager.Initialize(options);
  EXPECT_EQ(dflow_graph_manager.AddGraph(1, graph, options), SUCCESS);
  const std::vector<GeTensor> inputs;
  // now DflowGraphManager is without graph_manager, init
  EXPECT_NE(dflow_graph_manager.CompileGraph(1, inputs), SUCCESS);
  dflow_graph_manager.Finalize();
}
}