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

#include "common/profiling/profiling_properties.h"
#include "ge/ge_api.h"
#include "framework/common/types.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "graph/compute_graph.h"
#include "graph/execute/model_executor.h"
#include "graph/ge_local_context.h"
#include "graph/op_desc.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "runtime/subscriber/global_profiler.h"

namespace ge {
class ProfilingStartNodeTest : public testing::Test {
  void SetUp() { ProfilingProperties::Instance().SetTrainingTrace(true); }
  void TearDown() { ProfilingProperties::Instance().SetTrainingTrace(false); }
};
static void BuildAddGraph(ComputeGraphPtr &graph) {
  const auto SetUnknownOpKernel = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    static uint32_t index = 0U;
    const static std::set<std::string> kGeLocalTypes{DATA, CONSTANT, VARIABLE, FILECONSTANT, NETOUTPUT, AIPP_DATA_TYPE};

    GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_FLOAT);
    TensorUtils::SetSize(tensor, 64);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      (void)AttrUtils::SetBool(op_desc, "OwnerGraphIsUnknown", true);
      if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AIcoreEngine");
      } else {
        std::string op_kernel_name =
            (kGeLocalTypes.count(op_desc->GetType()) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE";
        op_desc->SetOpKernelLibName(op_kernel_name);
      }

      vector<int64_t> output_offset;
      for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
        op_desc->UpdateOutputDesc(i, tensor);
        output_offset.emplace_back(index * 64);
        ++index;
      }
      op_desc->SetOutputOffset(output_offset);
      op_desc->SetWorkspace({});
      op_desc->SetWorkspaceBytes({});
    }

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      vector<int64_t> input_offset;
      for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
        op_desc->UpdateInputDesc(i, tensor);
        if (node->GetType() == NETOUTPUT && node->GetName() != NODE_NAME_NET_OUTPUT) {
          AttrUtils::SetInt(op_desc->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, i);
        }
        if (node->GetType() == NETOUTPUT && node->GetName() == NODE_NAME_NET_OUTPUT) {
          op_desc->SetSrcName({"add"});
          op_desc->SetSrcIndex({0});
        }

        const auto in_anchor = node->GetInDataAnchor(i);
        if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
          input_offset.emplace_back(-1);
          continue;
        }

        const auto out_anchor = in_anchor->GetPeerOutAnchor();
        const auto peer_node = out_anchor->GetOwnerNode();
        const vector<int64_t> output_offset = peer_node->GetOpDesc()->GetOutputOffset();
        if (static_cast<size_t>(out_anchor->GetIdx()) >= output_offset.size()) {
          input_offset.emplace_back(-1);
          continue;
        }

        input_offset.emplace_back(output_offset.at(out_anchor->GetIdx()));
      }
      op_desc->SetInputOffset(input_offset);
    }
  };

  DEF_GRAPH(g1) {
    CHAIN(NODE("start", STARTOFSEQUENCE)->CTRL_EDGE()->NODE("data1", DATA));
    CHAIN(NODE("start", STARTOFSEQUENCE)->CTRL_EDGE()->NODE("data2", DATA));
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("add", ADD));
    CHAIN(NODE("add", ADD)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(graph->GetDirectNode());
}

static void BuildAddGraphModel(ComputeGraphPtr &graph, GeModelPtr &ge_model, TBEKernelStore &tbe_kernel_store) {
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef_TE(graph, *model_task_def, "add", tbe_kernel_store);

  InitEventTaskDef(graph, *model_task_def);
  InitFusionTaskDef(graph, *model_task_def);
  InitEndGraphDef(graph, *model_task_def, "Node_Output");

  InitProfilerTaskDef(graph, *model_task_def);

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  std::vector<uint64_t> weights_value(100, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
}

TEST_F(ProfilingStartNodeTest, test_build_graph_with_profiling_success) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", DATA)->EDGE(0, 0)->NODE("add_1", ADD));
    CHAIN(NODE("data_2", DATA)->EDGE(0, 1)->NODE("add_1", ADD));
  };

  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1073741824";
  options[VARIABLE_MEMORY_MAX_SIZE] = "1073741824";
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  Status ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto node : graph->GetAllNodes()) {
      if (node->GetType() == DATA) {
        EXPECT_NE(node->GetInControlAnchor(), nullptr);
        EXPECT_NE(node->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 0);
        for (const auto &peer_out_ctrl_anchor : node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
          EXPECT_NE(peer_out_ctrl_anchor, nullptr);
          auto in_node_before_dst_node = peer_out_ctrl_anchor->GetOwnerNode();
          EXPECT_NE(in_node_before_dst_node, nullptr);
        }
      }
    }
  };
}

TEST_F(ProfilingStartNodeTest, test_execute_graph_with_profiling_success) {
  char runtime2_env[MMPA_MAX_PATH] = {'0'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  ComputeGraphPtr graph;
  BuildAddGraph(graph);

  GeModelPtr ge_model;
  TBEKernelStore tbe_kernel_store;
  BuildAddGraphModel(graph, ge_model, tbe_kernel_store);
  EXPECT_NE(ge_model, nullptr);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetAsync(true);
  ProfilingProperties::Instance().SetLoadProfiling(true);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kTaskTime}));

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  // Test for Execute.
  std::vector<gert::Tensor> inputs(2);
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
  runtime2_env[1] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  ProfilingProperties::Instance().SetLoadProfiling(false);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0);
}
}  // namespace ge
