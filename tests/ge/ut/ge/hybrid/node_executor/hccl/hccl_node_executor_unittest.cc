/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/runtime_inference_context.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/hccl/hccl_node_executor.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/env_path.h"
#include "faker/space_registry_faker.h"

using namespace std;
using namespace testing;
namespace {
const std::string kHcclSoPath = "libhccl_stub.so";
}
namespace ge {
using namespace hybrid;

class UtestHcclNodeExecutor : public testing::Test {
 protected:
  void SetUp() {
    EnvPath ep;
    hccl_so_path_ = PathUtils::Join({ep.GetOrCreateCaseTmpPath("UtestHcclNodeExecutor"), kHcclSoPath});
    auto hccl_so_src_path = PathUtils::Join({ep.GetBinRootPath(), "tests", "depends", "hccl", kHcclSoPath});
    system(("cp -rf " + hccl_so_src_path + " " + hccl_so_path_).c_str());
  }
  void TearDown() {
    EnvPath().RemoveRfCaseTmpPath("UtestHcclNodeExecutor");
  }
  std::string hccl_so_path_;
};

namespace {
struct FakeGraphItem : GraphItem {
  FakeGraphItem(NodePtr node) {
    NodeItem::Create(node, node_item);
    node_item->input_start = 0;
    node_item->output_start = 0;
    node_items_.emplace_back(node_item.get());
    total_inputs_ = node->GetAllInAnchors().size();
    total_outputs_ = node->GetAllOutAnchors().size();
  }

  NodeItem *GetNodeItem() {
    return node_item.get();
  }

 private:
  std::unique_ptr<NodeItem> node_item;
};
}  // namespace

TEST_F(UtestHcclNodeExecutor, test_rdmatask_extract_tensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = CreateNode(*graph, "hcom", HCOMREMOTEREAD, 1, 1);
  FakeGraphItem graph_item(node);
  NodeItem *node_item = graph_item.GetNodeItem();
  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  GeShape s({1, 3});
  GeTensorDesc tensor_desc(s);
  GeTensorPtr tensor = make_shared<GeTensor>(tensor_desc);
  std::vector<uint8_t> data = {1, 2, 3, 4};
  tensor->SetData(data);
  graph_context.runtime_context_.SetTensor(1, 0, tensor);

  vector<HcomRemoteAccessAddrInfo> addr_infos;
  shared_ptr<RdmaNodeTask> task = MakeShared<RdmaNodeTask>();
  task->remote_index_ = {1, 0};
  ASSERT_EQ(task->ExtractTensor(*node_state->GetTaskContext(), addr_infos), PARAM_INVALID);

  auto op_desc = node->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, HCOMREMOTEREFREAD);
  ASSERT_EQ(task->ExtractTensor(*node_state->GetTaskContext(), addr_infos), PARAM_INVALID);

  GeShape s2({1});
  GeTensorDesc tensor_desc2(s2);
  GeTensorPtr tensor2 = make_shared<GeTensor>(tensor_desc2);
  graph_context.runtime_context_.SetTensor(1, 0, tensor2);
  task->ExtractTensor(*node_state->GetTaskContext(), addr_infos);
  ASSERT_EQ(task->ExtractTensor(*node_state->GetTaskContext(), addr_infos), PARAM_INVALID);
  graph_context.runtime_context_.Release();
}

TEST_F(UtestHcclNodeExecutor, gatheralltoallv_execute) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "gatheralltoallv", HCOMGATHERALLTOALLV, 4, 2);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 4;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 4; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1 = 512;
  TensorValue out_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  NodeTaskPtr task = nullptr;
  HcclNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  const auto so_path = hccl_so_path_;
  auto handle = mmDlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle, nullptr);
  node_state->GetTaskContext()->handle_ = handle;
  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  HcclNodeTask hccl_node_task;
  ASSERT_NE(hccl_node_task.ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  node_state->GetTaskContext()->handle_ = nullptr;
  ge::AttrUtils::SetDataType(node->GetOpDesc(), HCOM_ATTR_DATA_TYPE, DT_STRING);
  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  if (handle != nullptr) {
    dlclose(handle);
  }

  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
}

TEST_F(UtestHcclNodeExecutor, alltoallv_execute) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "alltoallv", HCOMALLTOALLV, 5, 1);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 5;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;
  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 5; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_1 = 512;
  TensorValue out_tensor0(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);
  NodeTaskPtr task = nullptr;
  HcclNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  const auto so_path = hccl_so_path_;
  auto handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle, nullptr);
  node_state->GetTaskContext()->handle_ = handle;
  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(node_executor.ExecuteTask(*task, *node_state->GetTaskContext(), done), SUCCESS);  //??

  HcclNodeTask hccl_node_task;
  ASSERT_NE(hccl_node_task.ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  ge::AttrUtils::SetDataType(node->GetOpDesc(), HCOM_ATTR_DATA_TYPE, DT_STRING);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
  ASSERT_EQ(task->Init(*node_state->GetTaskContext()), SUCCESS);
  ASSERT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);

  if (handle != nullptr) {
    dlclose(handle);
  }
}

TEST_F(UtestHcclNodeExecutor, alltoallvc_execute) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "alltoallv", HCOMALLTOALLVC, 2, 1);

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
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 2; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  NodeTaskPtr task = nullptr;
  HcclNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  const auto so_path = hccl_so_path_;
  auto handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle, nullptr);
  node_state->GetTaskContext()->handle_ = handle;
  std::function<void()> done = []() {};
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  HcclNodeTask hccl_node_task;
  ASSERT_NE(hccl_node_task.ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  node_state->GetTaskContext()->handle_ = nullptr;
  ge::AttrUtils::SetDataType(node->GetOpDesc(), HCOM_ATTR_DATA_TYPE, DT_STRING);
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  if (handle != nullptr) {
    dlclose(handle);
  }

  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);
}

TEST_F(UtestHcclNodeExecutor, test_FillHcomOpInfo) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "broadcast", HCOMBROADCAST, 2, 1);

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
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  HcclNodeTask hccl_node_task;
  std::vector<void *> inputs;
  std::vector<void *> outputs;
  HcomOperation hcom_op_info;

  std::shared_ptr<ge::hybrid::TaskContext> context = node_state->GetTaskContext();

  node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_BOOL);
  EXPECT_EQ(hccl_node_task.FillHcomOpInfo(*context, node->GetOpDesc(), inputs, outputs, hcom_op_info), PARAM_INVALID);

  node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_INT32);
  AttrUtils::SetStr(node->GetOpDesc(), HCOM_ATTR_REDUCE_TYPE, "min");
  AttrUtils::SetInt(node->GetOpDesc(), HCOM_ATTR_ROOT_RANK, 1);
  EXPECT_EQ(hccl_node_task.FillHcomOpInfo(*context, node->GetOpDesc(), inputs, outputs, hcom_op_info), SUCCESS);

  auto op_desc = node->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, "HCOMALLREDUCE");
  EXPECT_NE(hccl_node_task.FillHcomOpInfo(*context, node->GetOpDesc(), inputs, outputs, hcom_op_info), SUCCESS);  //??
}

TEST_F(UtestHcclNodeExecutor, test_GetInputsOutPuts) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "broadcast", HCOMBROADCAST, 2, 1);

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
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  HcclNodeTask hccl_node_task;
  std::vector<void *> inputs;
  std::vector<void *> outputs;
  HcomOperation hcom_op_info;

  std::shared_ptr<ge::hybrid::TaskContext> context = node_state->GetTaskContext();

  context->inputs_start_ = new TensorValue[2];
  context->outputs_start_ = new TensorValue[1];

  EXPECT_EQ(hccl_node_task.GetInputsOutPuts(*context, inputs, outputs), SUCCESS);
  EXPECT_EQ(inputs.size(), 2);
  EXPECT_EQ(outputs.size(), 1);
  delete[] (TensorValue*)(context->outputs_start_);
  delete[] (TensorValue*)(context->inputs_start_);
}

TEST_F(UtestHcclNodeExecutor, test_RdmaNodeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("rdma", HCOMREMOTEREAD);
  op_desc->AddInputDesc("remote", GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc("local_offset", GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc("local", GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::NodePtr node = graph->AddNode(op_desc);

  ge::OpDescPtr data_op_desc = std::make_shared<OpDesc>("data", DATA);
  data_op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  data_op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::NodePtr data_node = graph->AddNode(data_op_desc);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  ge::OpDescPtr data_op_desc2 = std::make_shared<OpDesc>("data", DATA);
  data_op_desc2->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  data_op_desc2->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::NodePtr data_node2 = graph->AddNode(data_op_desc2);

  GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), node->GetInDataAnchor(1));

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 4;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i[32]{};
  for (int i = 0; i < 4; ++i) {
    TensorValue in_tensor0(value_i, sizeof(uint64_t) * 32);
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0[32];
  TensorValue out_tensor0(value_0, sizeof(uint64_t) * 32);
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1[32];
  TensorValue out_tensor1(value_1, sizeof(uint64_t) * 32);
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  NodeTaskPtr task = nullptr;
  HcclNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(node_executor.PrepareTask(*task, *node_state->GetTaskContext()), SUCCESS);

  const auto so_path = hccl_so_path_;
  auto handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle, nullptr);

  ASSERT_EQ(task->Init(*node_state->GetTaskContext()), SUCCESS);

  node_state->GetTaskContext()->handle_ = handle;
  std::function<void()> done = []() {};
  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  RuntimeInferenceContext &rt_ctx =
      const_cast<RuntimeInferenceContext &>(node_state->GetTaskContext()->GetExecutionContext()->runtime_context_);
  std::vector<HcomRemoteAccessAddrInfo> addr_infos;
  uint64_t *data = new uint64_t[4];
  ASSERT_NE((dynamic_cast<RdmaNodeTask *>(task.get()))
                ->SetAddrInfo(*node_state->GetTaskContext(), rt_ctx, data, 4, addr_infos),
            FAILED);  // ??

  data[2] = 1000;
  ASSERT_EQ((dynamic_cast<RdmaNodeTask *>(task.get()))
                ->SetAddrInfo(*node_state->GetTaskContext(), rt_ctx, data, 4, addr_infos),
            SUCCESS);

  (dynamic_cast<RdmaNodeTask *>(task.get()))->skip_flag_ = true;
  ASSERT_NE((dynamic_cast<RdmaNodeTask *>(task.get()))
                ->SetAddrInfo(*node_state->GetTaskContext(), rt_ctx, data, 4, addr_infos),
            SUCCESS);  //??

  if (handle != nullptr) {
    dlclose(handle);
  }
  delete[] data;
}

TEST_F(UtestHcclNodeExecutor, test_HcclNodeTask) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeModelPtr ge_sub_model = std::make_shared<GeModel>();
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge_root_model->SetSubgraphInstanceNameToModel("sub", ge_sub_model);
  HybridModel hybrid_model(ge_root_model);

  NodePtr node = CreateNode(*graph, "rdma", HCOMBROADCAST, 4, 2);

  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  hybrid_model.node_items_[node] = std::move(new_node);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 4;
  graph_item.total_outputs_ = 2;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);

  uint64_t value_i = 512;
  for (int i = 0; i < 4; ++i) {
    TensorValue in_tensor0(&value_i, sizeof(value_i));
    subgraph_context.SetInput(*node_item, i, in_tensor0);
  }

  uint64_t value_0 = 512;
  TensorValue out_tensor0(&value_0, sizeof(value_0));
  subgraph_context.SetOutput(*node_item, 0, out_tensor0);

  uint64_t value_1 = 512;
  TensorValue out_tensor1(&value_1, sizeof(value_1));
  subgraph_context.SetOutput(*node_item, 1, out_tensor1);

  NodeTaskPtr task = nullptr;
  HcclNodeExecutor node_executor;
  ASSERT_EQ(node_executor.LoadTask(hybrid_model, node, task), SUCCESS);
  ASSERT_NE(task, nullptr);

  ASSERT_EQ(node_executor.PrepareTask(*task, *node_state->GetTaskContext()), SUCCESS);

  const auto so_path = hccl_so_path_;
  auto handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(handle, nullptr);

  ASSERT_EQ(task->Init(*node_state->GetTaskContext()), SUCCESS);

  node_state->GetTaskContext()->handle_ = handle;
  std::function<void()> done = []() {};
  ASSERT_NE(task->ExecuteAsync(*node_state->GetTaskContext(), done), SUCCESS);

  node_state->GetTaskContext()->handle_ = nullptr;
  ASSERT_EQ(task->ExecuteAsync(*node_state->GetTaskContext(), done), FAILED);

  if (handle != nullptr) {
    dlclose(handle);
  }
}

TEST_F(UtestHcclNodeExecutor, test_HcclNodeExecutor_Initialize_Finalize) {
  HcclNodeExecutor node_executor;
  ASSERT_EQ(node_executor.Initialize(), FAILED);

  const std::string file_name = "libhcom_graph_adaptor.so";
  std::string path = GetModelPath();
  (void)path.append(file_name);
  system(("touch " + path).c_str());
  ASSERT_EQ(node_executor.Initialize(), FAILED);
  unlink(path.c_str());

  ASSERT_EQ(node_executor.Finalize(), FAILED);
}

}  // namespace ge
