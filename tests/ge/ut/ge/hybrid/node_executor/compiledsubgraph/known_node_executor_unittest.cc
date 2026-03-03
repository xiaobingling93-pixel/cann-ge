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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "hybrid/executor/subgraph_context.h"
#include "common/model/ge_root_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "hybrid/executor/resource_manager.h"
#include "hybrid/node_executor/compiledsubgraph/known_node_executor.h"
#include "common/dump/dump_manager.h"
#include "graph/load/model_manager/task_info/fe/kernel_task_info.h"
#include "graph/passes/graph_builder_utils.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestKnownNodeExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

class KnownNodeTaskMock : public KnownNodeTask {
 public:
  KnownNodeTaskMock(std::shared_ptr<DavinciModel> davinci_model): KnownNodeTask(davinci_model) {};
  ~KnownNodeTaskMock() override = default;
  MOCK_METHOD2(DoInitDavinciModel, Status(const uintptr_t, const size_t));
};

static ComputeGraphPtr BuildDataDirectConnectGraph() {
  const char *kRefIndex = "_parent_node_index";
  ge::ut::GraphBuilder builder("subgraph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 1, 1);
  (void)AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), kRefIndex, 0);

  builder.AddDataEdge(data, 0, netoutput, 0);
  return builder.GetGraph();
}

TEST_F(UtestKnownNodeExecutor, test_init_davinci_model) {
  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  davinci_model->SetDeviceId(0);
  davinci_model->SetFeatureBaseRefreshable(true);

  auto ge_model = make_shared<GeModel>();
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  davinci_model->Assign(ge_model);

  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("default");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  HybridModel model(ge_root_model);
  KnownNodeTaskMock mock(davinci_model);
  DumpProperties dump_properties;
  dump_properties.enable_dump_ = "1";
  DumpManager::GetInstance().AddDumpProperties(model.GetSessionId(), dump_properties);
  EXPECT_CALL(mock, DoInitDavinciModel).WillRepeatedly(::testing::Return(SUCCESS));
  ASSERT_EQ(mock.InitDavinciModel(model, model.GetModelWeight("subgraph")), SUCCESS);

  int32_t buffer[8];
  model.weight_buffer_map_.emplace("subgraph", TensorBuffer::Create(buffer, sizeof(buffer)));
  ASSERT_EQ(mock.InitDavinciModel(model, model.GetModelWeight("subgraph")), SUCCESS);
}

TEST_F(UtestKnownNodeExecutor, TestParseAttrForAllocatingOutputs) {
  ut::GraphBuilder builder("test-graph");
  auto data_node = builder.AddNode("Data0", DATA, 1, 1);
  auto netoutput_node = builder.AddNode("NodeOutput", NETOUTPUT, 2, 2);
  builder.AddDataEdge(data_node, 0, netoutput_node, 0);
  auto const_node = builder.AddNode("Const0", CONSTANT, 0, 1);
  builder.AddDataEdge(const_node, 0, netoutput_node, 1);
  auto graph = builder.GetGraph();

  ut::GraphBuilder builder2("root-graph");
  auto partitioned_call = builder2.AddNode("Node0", PARTITIONEDCALL, 1, 2);
  NodeItem node_item(partitioned_call);
  ASSERT_EQ(KnownNodeExecutor::ParseAttrForAllocatingOutputs(node_item, *graph), SUCCESS);
  ASSERT_EQ(node_item.ref_outputs.size(), 1);
  ASSERT_EQ(node_item.ref_outputs[1], const_node);
  ASSERT_EQ(node_item.reuse_inputs.size(), 1);
  ASSERT_EQ(node_item.reuse_inputs[0], 0);
  auto davinci_model = std::make_shared<DavinciModel>(0, nullptr);
  node_item.kernel_task = std::make_shared<hybrid::KnownNodeTask>(davinci_model);
  auto executor = std::make_shared<KnownNodeExecutor>();
  ASSERT_EQ(executor->ReportProfilingData(node_item), SUCCESS);
}

TEST_F(UtestKnownNodeExecutor, TestSetGlobalStep) {
  OpDescPtr op_desc = CreateOpDesc("PartitionedCall", "PartitionedCall");
  auto root_graph = make_shared<ComputeGraph>("root_graph");
  auto node = root_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(root_graph);
  auto sub_graph = BuildDataDirectConnectGraph();
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(node);
  node->GetOpDesc()->AddSubgraphName("subgraph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  root_graph->AddSubgraph("subgraph", sub_graph);

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_ = ge_root_model->GetRootGraph();
  auto *step_id = new int64_t[1];
  step_id[0] = 520;
  std::unique_ptr<TensorBuffer> tensor_buf;
  tensor_buf = tensor_buf->Create((void *)step_id, sizeof(int64_t));
  hybrid_model.global_step_ = std::move(tensor_buf);
  KnownNodeExecutor known_node_executor;
  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, nullptr);
  known_node_executor.SetDaviciModel(hybrid_model, node, davinci_model);
  EXPECT_EQ(*(reinterpret_cast<int64_t *>(davinci_model->global_step_addr_)), 520);
  delete[] step_id;
}

TEST_F(UtestKnownNodeExecutor, test_KnownNodeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  ge_sub_model->graph_ = graph;
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_sub_model->SetModelTaskDef(model_def);

  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root");

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  uint8_t *test_buffer = new uint8_t[16];
  hybrid_model.global_step_ = TensorBuffer::Create(test_buffer, 16);

  NodePtr node = CreateNode(*graph, "mul", MATMUL, 2, 2);
  NodePtr data_node = CreateNode(*graph, "data", DATA, 1, 1);
  NodePtr const_node = CreateNode(*graph, "const", CONSTANT, 1, 1);
  NodePtr out_node = CreateNode(*graph, "netoutput", NETOUTPUT, 2, 1);

  out_node->GetOpDesc()->SetSrcName(std::vector<std::string>({"out", "out2"}));
  out_node->GetOpDesc()->SetSrcIndex(std::vector<int64_t>({0, 1}));

  ASSERT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), node->GetInDataAnchor(0)), SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(1)), SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0)), SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(1)), SUCCESS);

  NodePtr root_node = CreateNode(*root_graph, "root_node", DATA, 1, 1);
  root_node->GetOpDesc()->AddSubgraphName("sub");
  NodeUtils::SetSubgraph(*root_node, 0, graph);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  new_node->input_start = 2;
  new_node->output_start = 2;
  new_node->num_inputs = 2;
  new_node->num_outputs = 2;
  hybrid_model.node_items_[node] = std::move(new_node);

  std::unique_ptr<NodeItem> new_data_node;
  ASSERT_EQ(NodeItem::Create(data_node, new_data_node), SUCCESS);
  NodeItem *data_node_item = new_data_node.get();
  data_node_item->input_start = 1;
  data_node_item->output_start = 1;
  hybrid_model.node_items_[data_node] = std::move(new_data_node);

  std::unique_ptr<NodeItem> new_const_node;
  ASSERT_EQ(NodeItem::Create(const_node, new_const_node), SUCCESS);
  NodeItem *const_node_item = new_const_node.get();
  const_node_item->input_start = 1;
  const_node_item->output_start = 1;
  hybrid_model.node_items_[const_node] = std::move(new_const_node);

  std::unique_ptr<NodeItem> new_out_node;
  ASSERT_EQ(NodeItem::Create(out_node, new_out_node), SUCCESS);
  NodeItem *out_node_item = new_out_node.get();
  ASSERT_NE(out_node_item, nullptr);

  out_node_item->input_start = 2;
  out_node_item->output_start = 1;

  hybrid_model.node_items_[out_node] = std::move(new_out_node);

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.node_items_.emplace_back(data_node_item);
  graph_item.node_items_.emplace_back(const_node_item);
  graph_item.node_items_.emplace_back(out_node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 2;

  EXPECT_TRUE(AttrUtils::SetInt(ge_sub_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  hybrid_model.known_shape_sub_models_[node] = ge_sub_model;
  hybrid_model.root_graph_ = root_graph;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  ASSERT_EQ(graph_context.callback_manager->Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  KnownNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);
  void *test1 = new uint8_t[8];
  void *test2 = new uint8_t[8];
  node_state->GetTaskContext()->outputs_start_ = new TensorValue[2];
  node_state->GetTaskContext()->outputs_start_->ref_buffer_ = test1;
  node_state->GetTaskContext()->outputs_start_->ref_size_ = 8;

  (node_state->GetTaskContext()->outputs_start_ + 1)->ref_buffer_ = test2;
  (node_state->GetTaskContext()->outputs_start_ + 1)->ref_size_ = 8;

  void *test3 = new uint8_t[8];
  void *test4 = new uint8_t[8];
  node_state->GetTaskContext()->inputs_start_ = new TensorValue[2];
  node_state->GetTaskContext()->inputs_start_->ref_buffer_ = test3;
  node_state->GetTaskContext()->inputs_start_->ref_size_ = 8;

  (node_state->GetTaskContext()->inputs_start_ + 1)->ref_buffer_ = test4;
  (node_state->GetTaskContext()->inputs_start_ + 1)->ref_size_ = 8;
  node_state->GetTaskContext()->execution_context_->allocator = new NpuMemoryAllocator(0, nullptr);
  ASSERT_EQ(node_executor.PrepareTask(*task, *node_state->GetTaskContext()), SUCCESS);
  ASSERT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  std::function<void()> done = []() {};

  ASSERT_EQ(node_executor.ExecuteTask(*task, *node_state->GetTaskContext(), done), SUCCESS);

  dynamic_cast<KnownNodeTask *>(task.get())->davinci_model_->task_list_.push_back(std::make_shared<KernelTaskInfo>());
  ASSERT_EQ(node_executor.ExecuteTask(*task, *node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(graph_context.callback_manager->Destroy(), SUCCESS);
  delete node_state->GetTaskContext()->execution_context_->allocator;
  delete[] (uint8_t *)test4;
  delete[] (uint8_t *)test3;
  delete[] (uint8_t *)test2;
  delete[] (uint8_t *)test1;
  delete[] (TensorValue *)node_state->GetTaskContext()->inputs_start_;
  delete[] (TensorValue *)node_state->GetTaskContext()->outputs_start_;
  delete[] test_buffer;
}
} // namespace ge
