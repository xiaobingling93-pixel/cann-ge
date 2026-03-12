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
#include "ge_graph_dsl/graph_dsl.h"
#include "es_ge_test_ops.h"
#include "graph/utils/graph_utils_ex.h"
#include "jit_execution/user_graph_ctrl.h"
#include "stub/gert_runtime_stub.h"
#include <vector>
#include "jit_share_graph.h"
#include "common/model/external_allocator_manager.h"
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"
#include "faker/space_registry_faker.h"
#include "common_setup.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/execute/model_executor.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace testing;

namespace ge {
class UserGraphControlUT : public testing::Test {
 protected:
  void SetUp() override {
    const auto env_ptr = getenv("LD_PRELOAD");
    if (env_ptr != nullptr) {
      env = env_ptr;
      unsetenv("LD_PRELOAD");
    }
    CommonSetupUtil::CommonSetup();
    gert_stub_.GetKernelStub().StubTiling();
    RuntimeStub::Install(nullptr); // gert的rts stub不能在多线程环境下工作，因此使用默认rts stub
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
    std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend310"}};
    GetThreadLocalContext().SetGlobalOption(options);
  }
  void TearDown() override {
    CommonSetupUtil::CommonTearDown();
    gert_stub_.Clear();
    if (!env.empty()) {
      setenv("LD_PRELOAD", env.c_str(), 1);
    }
  }
  gert::GertRuntimeStub gert_stub_;
  std::string env;
};

TEST_F(UserGraphControlUT, AddGraphInstance_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);

  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphControlUT, GetSetCompiledFlag_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  graph_manager.AddGraph(user_graph_id, *graph.get(), {}, OmgContext());
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);

  EXPECT_FALSE(ctrl->GetCompiledFlag());
  ctrl->SetCompiledFlag(true);
  EXPECT_TRUE(ctrl->GetCompiledFlag());
  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphControlUT, AddGraphInstance_MultiThread_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  ThreadPool thread_pool("tset", 8);
  std::vector<std::future<Status>> futs;
  for (size_t i = 0u; i< 10; ++i) {
    auto fut = thread_pool.commit([&ctrl, this]() -> Status {
      EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);
      return SUCCESS;
    });
    EXPECT_TRUE(fut.valid());
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    EXPECT_EQ(fut.get(), SUCCESS);
  }
  // todo check log

  thread_pool.Destroy();
  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphControlUT, RunGraphAsync_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<gert::Tensor> gert_inputs(1U);
  TensorCheckUtils::ConstructGertTensor(gert_inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
    return SUCCESS;
  };

  auto task = MakeUnique<UserGraphExecution>(user_graph_id, gert_inputs, callback, 0);

  ctrl->RunGraphAsync(task);

  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}
TEST_F(UserGraphControlUT, RunGraphAsync_Failed) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, -3, 3, 2}; // invalid shape
  TensorDesc td(Shape(shape_dim), FORMAT_NCHW, DT_FLOAT);
  td.SetOriginShape(Shape(shape_dim)); // todo check tfa set origin shape?
  Tensor tensor(td);
  std::vector<Tensor> inputs{std::move(tensor)};
  std::vector<Tensor> outputs;
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_NE(status, SUCCESS);
    return SUCCESS;
  };
  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  auto task = MakeUnique<UserGraphExecution>(user_graph_id, gert_inputs, callback, 0);
  ctrl->RunGraphAsync(task);

  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

/**
 * 测试有10个jit实例的时候，执行接口被多线程调用
 * 预期，多个执行请求被并发执行
 * 测试条件：(1)add graph调用10次。表示只有10个jit executor
 *         (2)RunGraphAsync设置10个并发调用
 * 预期： callback函数被执行了10次。表示执行了10次
 * */
TEST_F(UserGraphControlUT, RunGraphAsync_MultiThread_MultiJitInstance_Success) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  ThreadPool thread_pool("tset", 8);
  std::vector<std::future<Status>> futs;
  for (size_t i = 0u; i< 10; ++i) {
    auto fut = thread_pool.commit([&ctrl, this]() -> Status {
      EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);
      return SUCCESS;
    });
    EXPECT_TRUE(fut.valid());
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    EXPECT_EQ(fut.get(), SUCCESS);
  }

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<std::vector<gert::Tensor>> thread_2_gert_inputs;
  thread_2_gert_inputs.resize(10U);
  futs.clear();
  std::atomic_int32_t callback_times = 0;
  for (size_t i = 0u; i< 10; ++i) {
    std::vector<gert::Tensor> &gert_inputs = thread_2_gert_inputs[i];
    gert_inputs.resize(1U);
    TensorCheckUtils::ConstructGertTensor(gert_inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);

    auto fut = thread_pool.commit([&user_graph_id, &ctrl, this, &callback_times, &shape_dim, &gert_inputs]() -> Status {
      auto promise_ptr = std::make_shared<std::promise<Status>>();
      auto future = promise_ptr->get_future();

      const RunAsyncCallbackV2 callback = [&, promise_ptr](Status status, std::vector<gert::Tensor> &outputs) {
        EXPECT_EQ(status, SUCCESS);
        EXPECT_EQ(outputs.size(), 1);
        if (outputs.empty()) {
          return FAILED;
        }
        auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
        EXPECT_EQ(cur_dims, shape_dim);
        callback_times.fetch_add(1);
        promise_ptr->set_value(status);
        return SUCCESS;
      };
      auto task = MakeUnique<UserGraphExecution>(user_graph_id, gert_inputs, callback, 0);
      ctrl->RunGraphAsync(task);
      EXPECT_EQ(future.get(), SUCCESS);
      return SUCCESS;
    });
    EXPECT_TRUE(fut.valid());
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    EXPECT_EQ(fut.get(), SUCCESS);
  }

  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
  EXPECT_EQ(callback_times.load(), 10);
}
/**
 * 测试只有1个jit实例的时候，执行接口被多线程调用
 * 预期，多个执行请求被串行执行
 * 测试条件：(1)add graph只调用1次。表示只有1个jit executor
 *         (2)RunGraphAsync设置10个并发调用
 * 预期： callback函数被执行了10次。表示串行执行了10次（因为只有1个jit executor)
 * */
TEST_F(UserGraphControlUT, RunGraphAsync_MultiThread_OneJitInstance_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodes();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  // one jit instance
  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);

  ThreadPool thread_pool("tset", 8);
  std::vector<std::future<Status>> futs;

  std::mutex cv_mutex;
  std::condition_variable finish_condition;

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<std::vector<gert::Tensor>> thread_2_gert_inputs;
  thread_2_gert_inputs.resize(10U);
  futs.clear();
  std::atomic_int32_t callback_times = 0;

  for (size_t i = 0u; i< 10; ++i) {
    std::vector<gert::Tensor> &gert_inputs = thread_2_gert_inputs[i];
    gert_inputs.resize(1U);
    TensorCheckUtils::ConstructGertTensor(gert_inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);

    auto fut = thread_pool.commit([&user_graph_id, &ctrl, this, &shape_dim, &callback_times, &cv_mutex, &finish_condition, &gert_inputs]() -> Status {
      const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
        EXPECT_EQ(status, SUCCESS);
        EXPECT_EQ(outputs.size(), 1);
        if (outputs.empty()) {
          return FAILED;
        }
        auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
        EXPECT_EQ(cur_dims, shape_dim);
        callback_times.fetch_add(1);
        if (callback_times.load() == 10) {
          std::lock_guard<std::mutex> lk(cv_mutex);
          finish_condition.notify_all();
        }
        return SUCCESS;
      };

      auto task = MakeUnique<UserGraphExecution>(user_graph_id, gert_inputs, callback, 0);
      ctrl->RunGraphAsync(task);
      return SUCCESS;
    });
    EXPECT_TRUE(fut.valid());
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    EXPECT_EQ(fut.get(), SUCCESS);
  }
  std::unique_lock<std::mutex> cv_lock(cv_mutex);
  finish_condition.wait(cv_lock);
  cv_lock.unlock();
  EXPECT_EQ(callback_times.load(), 10);
  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphControlUT, RunGraphAsync_StaticShape_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodesStaticShape();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<gert::Tensor> gert_inputs(1U);
  TensorCheckUtils::ConstructGertTensor(gert_inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);
  const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(outputs.size(), 1);
    auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
    EXPECT_EQ(cur_dims, shape_dim);
  };

  auto task = MakeUnique<UserGraphExecution>(user_graph_id, gert_inputs, callback, 0);
  ctrl->RunGraphAsync(task);

  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

TEST_F(UserGraphControlUT, RunGraphAsync_StaticShape_MultiThread_Success) {
  ModelExecutor model_executor;
  model_executor.Initialize({}, 0);
  GraphManager graph_manager;
  EXPECT_EQ(graph_manager.Initialize({}, &model_executor), SUCCESS);
  uint32_t user_graph_id = 0u;
  auto graph = JitShareGraph::AllNormalNodesStaticShape();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  CompileContext compile_context(graph_manager);

  auto ctrl = MakeUnique<UserGraphControl>(user_graph_id, compute_graph, compile_context, graph_manager);
  EXPECT_NE(ctrl, nullptr);
  ThreadPool thread_pool("tset", 8);
  std::vector<std::future<Status>> futs;
  for (size_t i = 0u; i< 10; ++i) {
    auto fut = thread_pool.commit([&ctrl, this]() -> Status {
      EXPECT_EQ(ctrl->AddGraphInstance(), SUCCESS);
      return SUCCESS;
    });
    EXPECT_TRUE(fut.valid());
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    EXPECT_EQ(fut.get(), SUCCESS);
  }

  // prepare run task
  std::vector<int64_t> shape_dim = {2, 3, 3, 2};
  std::vector<std::vector<gert::Tensor>> thread_2_gert_inputs;
  thread_2_gert_inputs.resize(10U);
  futs.clear();
  for (size_t i = 0u; i < 10; ++i) {
    std::vector<gert::Tensor> &gert_inputs = thread_2_gert_inputs[i];
    gert_inputs.resize(1U);
    TensorCheckUtils::ConstructGertTensor(gert_inputs[0], {2, 3, 3, 2}, DT_FLOAT, FORMAT_NCHW);
    auto fut = thread_pool.commit([&user_graph_id, &ctrl, this, &shape_dim, &gert_inputs]() -> Status {
      const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
        EXPECT_EQ(status, SUCCESS);
        EXPECT_EQ(outputs.size(), 1);
        if (outputs.empty()) {
          return FAILED;
        }
        auto cur_dims = TensorTransUtils::GetDimsFromGertShape(outputs[0].GetStorageShape());
        EXPECT_EQ(cur_dims, shape_dim);
        return SUCCESS;
      };
      auto task = MakeUnique<UserGraphExecution>(user_graph_id, gert_inputs, callback, 0);
      ctrl->RunGraphAsync(task);
      return SUCCESS;
    });
    EXPECT_TRUE(fut.valid());
    futs.emplace_back(std::move(fut));
  }
  for (auto &fut : futs) {
    EXPECT_EQ(fut.get(), SUCCESS);
  }

  EXPECT_EQ(ctrl->Finalize(), SUCCESS);
  EXPECT_EQ(graph_manager.Finalize(), SUCCESS);
}

}  // namespace ge