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
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/node_executor/partitioned_call/partitioned_call_node_executor.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/graph.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

namespace {
  bool finalized = false;
}

class NodeExecutorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

class FailureNodeExecutor : public NodeExecutor {
 public:
  Status Initialize() override {
    return INTERNAL_ERROR;
  }
};

class SuccessNodeExecutor : public NodeExecutor {
 public:
  Status Initialize() override {
    initialized = true;
    finalized = false;
    return SUCCESS;
  }

  Status Finalize() override {
    finalized = true;
    return SUCCESS;
  }

  bool initialized = false;
};

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, FailureNodeExecutor);
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_TF, SuccessNodeExecutor);

TEST_F(NodeExecutorTest, TestGetOrCreateExecutor) {
  auto &manager = NodeExecutorManager::GetInstance();
  const NodeExecutor *executor = nullptr;
  Status ret = SUCCESS;
  // no builder
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::RESERVED, executor);
  ASSERT_EQ(ret, INTERNAL_ERROR);
  // initialize failure
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICORE, executor);
  ASSERT_EQ(ret, INTERNAL_ERROR);
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICPU_TF, executor);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_TRUE(executor != nullptr);
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICPU_TF, executor);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_TRUE(executor != nullptr);
  ASSERT_TRUE(((SuccessNodeExecutor*)executor)->initialized);
}

TEST_F(NodeExecutorTest, TestInitAndFinalize) {
  auto &manager = NodeExecutorManager::GetInstance();
  manager.FinalizeExecutors();
  manager.FinalizeExecutors();
  manager.EnsureInitialized();
  manager.EnsureInitialized();
  const NodeExecutor *executor = nullptr;
  auto ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICPU_TF, executor);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_TRUE(executor != nullptr);
  ASSERT_TRUE(((SuccessNodeExecutor*)executor)->initialized);
  manager.FinalizeExecutors();
  ASSERT_FALSE(manager.executors_.empty());
  manager.FinalizeExecutors();
  ASSERT_TRUE(manager.executors_.empty());
  ASSERT_TRUE(finalized);
}

TEST_F(NodeExecutorTest, TestPartitionedCall) {
  OpDescPtr op_desc = CreateOpDesc("Partitioned", "PartitionedCall", 2, 1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_unknown_rank");
  HybridModel hybrid_model(ge_root_model);
  ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>("sub_graph");
  NodeUtils::AddSubgraph(*node, "sub_graph", subgraph);
  hybrid_model.subgraph_items_["sub_graph"]=std::make_unique<GraphItem>();
  // 3. load empty task when fuzz compile
  std::shared_ptr<NodeTask> node_task_after_load;
  PartitionedCallNodeExecutor executor;
  ASSERT_EQ(executor.LoadTask(hybrid_model, node, node_task_after_load), SUCCESS);
}
} // namespace ge
