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
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "common/model/ge_root_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "hybrid/executor/resource_manager.h"

#include "macro_utils/dt_public_unscope.h"

#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

const char *Shape = "Shape";

class UtestGeLocalNodeExecutor : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

TEST_F(UtestGeLocalNodeExecutor, test_no_op_task) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "noop", NOOP, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 0;
  graph_item.total_outputs_ = 0;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  GeLocalNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
}

TEST_F(UtestGeLocalNodeExecutor, test_reshape_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "reshape", RESHAPE, 1, 1);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 1;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);
  node_state->SetUserAllocated(true);

  NodeTaskPtr task = nullptr;
  GeLocalNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), nullptr), GRAPH_PARAM_INVALID);

  ASSERT_EQ((dynamic_cast<RefInputTask *>(task.get()))->RefByOrder(std::vector<uint32_t>({}), *node_state->GetTaskContext()), INTERNAL_ERROR);
  ASSERT_EQ((dynamic_cast<RefInputTask *>(task.get()))->RefByOrder(std::vector<uint32_t>({0}), *node_state->GetTaskContext()), SUCCESS);

  // type not match
  //(dynamic_cast<RefInputTask *>(task.get()))->node_type_ = "MATMUL";
  //ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), nullptr), UNSUPPORTED);


  // const_cast<NodeItem *>(node_state->GetTaskContext()->node_item_)->num_outputs = 10;
  // node_state->GetTaskContext()->outputs_start_ = new TensorValue();
  // node_state->GetTaskContext()->outputs_start_->ref_buffer_ = new uint8_t[8];
  // node_state->GetTaskContext()->outputs_start_->ref_size_ = 8;
  // ASSERT_EQ((dynamic_cast<RefInputTask *>(task.get()))->RefOneByOne(*node_state->GetTaskContext()), INTERNAL_ERROR);
}

static void MakeHybridModel(std::unique_ptr<HybridModel> &hybrid_model,
                            GraphItem &graph_item,
                            std::unordered_map<std::string, NodePtr> &all_nodes,
                            std::unordered_map<std::string, NodeItem *> &all_node_items) {
  int32_t max_size = 1;
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor =
      std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&max_size), sizeof(int32_t));

  const auto const_op = OP_CFG(CONSTANT).OutCnt(1).Weight(const_tensor);
  const auto stack = OP_CFG(STACK).InCnt(1).OutCnt(1)
                                  .Attr(ATTR_NAME_DATA_FLOW_HANDLE, 1)
                                  .Attr(ATTR_NAME_DATA_FLOW_MAX_SIZE, 1);
  const auto stack_push = OP_CFG(STACKPUSH).InCnt(2).OutCnt(1).Attr(ATTR_NAME_DATA_FLOW_HANDLE, 1);
  const auto stack_pop = OP_CFG(STACKPOP).InCnt(1).OutCnt(1).Attr(ATTR_NAME_DATA_FLOW_HANDLE, 1);
  const auto stack_close = OP_CFG(STACKCLOSE).InCnt(1).Attr(ATTR_NAME_DATA_FLOW_HANDLE, 1);

  DEF_GRAPH(g1) {
    CHAIN(NODE("const", const_op)->EDGE(0, 0)->NODE("stack", stack));
    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackpush", stack_push));
    CHAIN(NODE("const", const_op)->EDGE(0, 1)->NODE("stackpush", stack_push));
    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackpop", stack_pop));
    CHAIN(NODE("stack", stack)->EDGE(0, 0)->NODE("stackclose", stack_close));
    CTRL_CHAIN(NODE("stackpush", stack_push)->NODE("stackpop", stack_pop));
    CTRL_CHAIN(NODE("stackpop", stack_pop)->NODE("stackclose", stack_close));
  };
  const auto graph = ToGeGraph(g1);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  (void)compute_graph->TopologicalSorting();

  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  std::unique_ptr<HybridModel> hybrid_model_instance(new(std::nothrow)HybridModel(ge_root_model));

  const auto const_node = compute_graph->FindNode("const");
  const auto stack_node = compute_graph->FindNode("stack");
  const auto stack_push_node = compute_graph->FindNode("stackpush");
  const auto stack_pop_node = compute_graph->FindNode("stackpop");
  const auto stack_close_node = compute_graph->FindNode("stackclose");
  const map<string, uint32_t> name_index = {{"max_size", 0}};
  if (stack_node != nullptr && stack_node->GetOpDesc() != nullptr) {
    stack_node->GetOpDesc()->UpdateInputName(name_index);
  }

  auto create_node_item = [&](const NodePtr &node, int32_t input_start, int32_t output_start) {
    std::unique_ptr<NodeItem> new_node;
    ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
    NodeItem *node_item = new_node.get();
    hybrid_model_instance->node_items_[node] = std::move(new_node);
    node_item->input_start = input_start;
    node_item->output_start = output_start;
    all_nodes[node->GetName()] = node;
    all_node_items[node->GetName()] = node_item;
  };
  create_node_item(const_node, 0, 0);
  create_node_item(stack_node, 0, 1);
  create_node_item(stack_push_node, 1, 2);
  create_node_item(stack_pop_node, 3, 3);
  create_node_item(stack_close_node, 4, 3);
  for (const auto &item : all_node_items) {
    graph_item.node_items_.emplace_back(item.second);
  }
  graph_item.total_inputs_ = 5;
  graph_item.total_outputs_ = 4;
  hybrid_model = std::move(hybrid_model_instance);
}

TEST_F(UtestGeLocalNodeExecutor, test_dynamic_stack_task) {
  std::unique_ptr<HybridModel> hybrid_model;
  auto graph_item = MakeUnique<GraphItem>();
  std::unordered_map<std::string, NodePtr> all_nodes;
  std::unordered_map<std::string, NodeItem *> all_node_items;
  MakeHybridModel(hybrid_model, *graph_item, all_nodes, all_node_items);
  hybrid_model->root_graph_item_ = std::move(graph_item);

  GraphExecutionContext graph_context;
  graph_context.model = hybrid_model.get();
  graph_context.allocator = NpuMemoryAllocator::GetAllocator(0);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  ASSERT_EQ(graph_context.res_manager.Init(graph_context.model->GetRootGraphItem()), SUCCESS);
  SubgraphContext subgraph_context(hybrid_model->root_graph_item_.get(), &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  EXPECT_EQ(graph_context.res_manager.data_flow_resources_.size(), 1);
  EXPECT_EQ(graph_context.res_manager.data_flow_kernels_.size(), 4);

  auto load_and_run_op = [&](const std::string &node_name,
      const TensorValue *const input = nullptr, TensorValue *const output = nullptr) -> Status {
    auto node_state = subgraph_context.GetNodeState(all_node_items[node_name]);
    if (node_state == nullptr) {
      return FAILED;
    }
    NodeTaskPtr task = nullptr;
    GeLocalNodeExecutor node_executor;
    EXPECT_EQ(node_executor.LoadTask(*hybrid_model, all_nodes[node_name], task), SUCCESS);
    if (task == nullptr) {
      return FAILED;
    }
    if (input != nullptr) {
      TensorValue *in = node_state->GetTaskContext()->MutableInput(1);
      *in = *input;
    }
    EXPECT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
    std::function<void()> done = []() {};
    auto ret = task->ExecuteAsync(*node_state->GetTaskContext(), done);
    if (ret == SUCCESS && output != nullptr) {
      *output = *(node_state->GetTaskContext()->MutableOutput(0));
    }
    return ret;
  };

  auto ret = load_and_run_op("stack");
  EXPECT_EQ(ret, SUCCESS);

  // pop when res is empty
  ret = load_and_run_op("stackpop");
  EXPECT_EQ(ret, INTERNAL_ERROR);

  TensorValue value_in;
  value_in.SetName("stack_push_in");
  ret = load_and_run_op("stackpush", &value_in);
  EXPECT_EQ(ret, SUCCESS);

  // push when capacity has reached the maximum
  ret = load_and_run_op("stackpush");
  EXPECT_EQ(ret, INTERNAL_ERROR);

  TensorValue value_out;
  ret = load_and_run_op("stackpop", nullptr, &value_out);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(value_in.DebugString(), value_out.DebugString());

  // pop when res is empty
  ret = load_and_run_op("stackpop");
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // pop when res is empty
  ret = load_and_run_op("stackclose");
  EXPECT_EQ(ret, SUCCESS);
}

class TestShapeKernel : public Kernel {
 public:
  Status Compute(const NodePtr& node, std::vector<GeTensorPtr>& v_output) const override {
    if (node->GetName() == "test_fail") {
      return FAILED;
    }

    auto output = std::make_shared<GeTensor>();
    std::vector<uint8_t> data{1, 2, 3};
    std::vector<int64_t> shape{3};
    output->MutableTensorDesc().SetShape(GeShape(shape));
    output->SetData(data);
    output->MutableTensorDesc().SetDataType(DT_UINT8);
    v_output.push_back(output);
    return SUCCESS;
  }
};
REGISTER_COMPUTE_NODE_KERNEL(Shape, TestShapeKernel);


TEST_F(UtestGeLocalNodeExecutor, test_DependInputShapeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "shape", SHAPE, 2, 1);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  graph_context.allocator = NpuMemoryAllocator::GetAllocator();
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  ASSERT_EQ(graph_context.callback_manager->Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  GeLocalNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  std::function<void()> done = []() {};

  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  void *test1 = new uint8_t[1];
  node_state->GetTaskContext()->outputs_start_ = new TensorValue[1];
  node_state->GetTaskContext()->outputs_start_->ref_buffer_ = test1;
  node_state->GetTaskContext()->outputs_start_->ref_size_ = 1;

  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  std::vector<GeTensorPtr> outputs = {std::make_shared<GeTensor>()};
  ASSERT_EQ(dynamic_cast<DependInputShapeTask *>(task.get())->CopyDataToOutput(
      1, outputs, SHAPE, *node_state->GetTaskContext()), SUCCESS);

  uint8_t *t1 = new uint8_t[100];
  outputs[0]->MutableData().SetData(t1, 100);
  ASSERT_EQ(dynamic_cast<DependInputShapeTask *>(task.get())->CopyDataToOutput(
      1, outputs, SHAPE, *node_state->GetTaskContext()), INTERNAL_ERROR);

  uint8_t *t2 = new uint8_t[1];
  outputs[0]->MutableData().SetData(t2, 1);
  ASSERT_EQ(dynamic_cast<DependInputShapeTask *>(task.get())->CopyDataToOutput(
     1, outputs, SHAPE, *node_state->GetTaskContext()), SUCCESS);    // ??

  auto op_desc = node->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, RESHAPE);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), UNSUPPORTED);

  ge::OpDescUtilsEx::SetType(op_desc, SHAPE);
  node->GetOpDesc()->SetName("test_fail");
  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  node->GetOpDesc()->SetName("shape");
  const_cast<NodeItem *>(node_state->GetTaskContext()->node_item_)->num_outputs = 10;
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), INTERNAL_ERROR);
  ASSERT_EQ(graph_context.callback_manager->Destroy(), SUCCESS);
  const_cast<NodeItem *>(node_state->GetTaskContext()->node_item_)->num_outputs = 1;
  delete[] (uint8_t*)t1;
  delete[] (uint8_t*)t2;
  delete[] (TensorValue*)(node_state->GetTaskContext()->outputs_start_);
  delete[] (uint8_t*)test1;
}

TEST_F(UtestGeLocalNodeExecutor, test_LoadTaskFail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "const", CONSTANT, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 0;
  graph_item.total_outputs_ = 0;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  GeLocalNodeExecutor node_executor;
  hybrid_model.constant_tensors_.clear();
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), INTERNAL_ERROR);

  auto op_desc = node->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, MATMUL);
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), UNSUPPORTED);
}

TEST_F(UtestGeLocalNodeExecutor, test_DataFlowNodeTask_Fail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "stack", STACK, 0, 0);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 0;
  graph_item.total_outputs_ = 0;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  NodeTaskPtr task = nullptr;
  GeLocalNodeExecutor node_executor;
  hybrid_model.constant_tensors_.clear();
  ASSERT_NE(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);   //??
  ASSERT_NE(task, nullptr);
  ASSERT_EQ(dynamic_cast<DataFlowNodeTask *>(task.get())->InitTaskBasicInfo(node), INTERNAL_ERROR);

  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), INTERNAL_ERROR);
}

} // namespace ge
