/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <thread>
#include <gtest/gtest.h>
#include <dlfcn.h>
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "es_ge_test_ops_c.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "common/env_path.h"
#include "common/topo_checker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/attr_utils.h"
#include "jit_execution/exe_points/guard_cache.h"
#include "jit_execution/exe_points/execution_point.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"

namespace ge {
class ExecutionPointUT : public testing::Test {
protected:
 void SetUp() override {
     auto ascend_install_path = EnvPath().GetAscendInstallPath();
     setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
     setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
     graph_ = EsCreateGraphBuilder("Hello");
     const auto env_ptr = getenv("LD_PRELOAD");
     if (env_ptr != nullptr) {
       env = env_ptr;
       unsetenv("LD_PRELOAD");
     }
 }
 void TearDown() override {
     EsDestroyGraphBuilder(graph_);
     graph_ = nullptr;
     unsetenv("ASCEND_OPP_PATH");
     unsetenv("LD_LIBRARY_PATH");
     if (!env.empty()) {
       setenv("LD_PRELOAD", env.c_str(), 1);
     }
 }
    EsCGraphBuilder *graph_{nullptr};

    ExecutionPoint *ep;
    std::string env;
};

TEST_F(ExecutionPointUT, load_guard_check_func) {
  dlog_setlevel(0, 1, 0);

  GuardCodegen codegen;
  auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph_))));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  // 产生guard
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  gert::Tensor tensor0 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor> inputs;
  inputs.emplace_back(std::move(tensor0));
  inputs.emplace_back(std::move(tensor1));
  // char_t reason[1024];
  // 1 模拟断图结果
  ep = new ExecutionPoint(1, compute_graph, compute_graph);

  // 2 首次获取GuardedExecutionPoint新实例对象，该对象状态为未编译，设置编译结果
  auto gep = ep->FindOrCreateGuarded(inputs);
  EXPECT_NE(gep, nullptr);
  EXPECT_NE(gep->GetGraph(), nullptr);
  if (!gep->Compiled()) {
    EXPECT_EQ(gep->SetCompiled(0, compute_graph), true);
  }
  EXPECT_EQ(gep->Compiled(), true);
  EXPECT_NE(gep->GetGraph(), nullptr);
  EXPECT_NE(gep->GetSlicedGraph(), nullptr);

  // 3 第二次获取GuardedExecutionPoint实例对象，GuardMatch
  gep = ep->FindOrCreateGuarded(inputs);
  EXPECT_NE(gep, nullptr);
  EXPECT_EQ(gep->Compiled(), true);
  EXPECT_NE(gep->GetGraph(), nullptr);
  EXPECT_NE(gep->GetSlicedGraph(), nullptr);

  // 4 第三次构造GuardMiss输入，创建GuardedExecutionPoint新实例对象，该对象状态为未编译
  // 传入非法参数，校验返回值失败
  inputs[1].MutableOriginShape().SetDim(0, 10);
  gep = ep->FindOrCreateGuarded(inputs);
  EXPECT_NE(gep, nullptr);
  EXPECT_EQ(gep->Compiled(), false);
  EXPECT_NE(gep->GetGraph(), nullptr);
  EXPECT_NE(gep->GetSlicedGraph(), nullptr);

  delete ep;
  dlog_setlevel(0, 3, 0);
}


ComputeGraphPtr CompileGraph1(ComputeGraphPtr compute_graph) {
  GuardCodegen codegen;
  //auto compute_graph = graph_->BuildComputeGraph();
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  // 产生guard
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  return compute_graph;
}

ComputeGraphPtr CompileGraph2(ComputeGraphPtr compute_graph) {
  GuardCodegen codegen;
  //auto compute_graph = graph_->BuildComputeGraph();
  auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  EXPECT_NE(attr, nullptr);
  ShapeEnvGuarder guard(attr);
  auto symbol0 = attr->CreateSymbol(10, MakeShared<InputShapeSource>(0, 0));
  auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));
  auto symbol2 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(0, 2));
  // 产生guard
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  EXPECT_SYMBOL_NE(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  return compute_graph;
}

static void ThreadFunction(ExecutionPoint *ep, const std::vector<gert::Tensor> *inputs, size_t *compiled_count, std::mutex *mutex, int flag) {
  static size_t count = 0;
  static size_t graph_id = 0;
  ++count;
  size_t thread_id = count;
  GELOGI("Thread %d", thread_id);

  std::map<const GuardedExecutionPoint *, uint32_t> geps_to_inner_ge_graph_id_;
  uint32_t instance_id;
  for (int i = 0; i < 3; i++) {
    std::lock_guard<std::mutex> locker(*mutex);
    auto gep = ep->FindOrCreateGuarded(*inputs);
    {
      if (!gep->Compiled()) {
        graph_id++;
        instance_id = graph_id;
        GELOGI("SetCompiled Thread %d begin", thread_id);
        EXPECT_EQ(geps_to_inner_ge_graph_id_.emplace(gep, instance_id).second, true);
        if (flag == 1) {
          EXPECT_EQ(gep->SetCompiled(instance_id, CompileGraph1(gep->GetGraph())), true);
        } else {
          EXPECT_EQ(gep->SetCompiled(instance_id, CompileGraph2(gep->GetGraph())), true);
        }
        (*compiled_count)++;
        GELOGI("SetCompiled Thread %d end graph_id %d", thread_id, graph_id);
      } else {
        auto iter = geps_to_inner_ge_graph_id_.find(gep);
        if (iter == geps_to_inner_ge_graph_id_.end()) {
          graph_id++;
          instance_id = graph_id;
          GELOGI("SetForked Thread %d begin", thread_id);
          gep->SetForked(instance_id);
          EXPECT_EQ(geps_to_inner_ge_graph_id_.emplace(gep, instance_id).second, true);
          GELOGI("SetForked Thread %d end graph_id %d", thread_id, graph_id);
        } else {
          instance_id = iter->second;
          GELOGI("Reuse Thread %d end graph_id %d", thread_id, instance_id);
        }
      }
    }
    EXPECT_NE(gep, nullptr);
    EXPECT_NE(gep->GetGraph(), nullptr);
    EXPECT_EQ(gep->Compiled(), true);
    EXPECT_NE(gep->GetSlicedGraph(), nullptr);
  }
}

TEST_F(ExecutionPointUT, NoCheckFunc_Error) {
  dlog_setlevel(0, 1, 0);

  auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph_))));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);

  gert::Tensor tensor0 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor> inputs;
  inputs.emplace_back(std::move(tensor0));
  inputs.emplace_back(std::move(tensor1));
  // char_t reason[1024];
  // 1 模拟断图结果
  ep = new ExecutionPoint(1, compute_graph, compute_graph);

  // 2 首次获取GuardedExecutionPoint新实例对象，该对象状态为未编译，设置编译结果
  auto gep = ep->FindOrCreateGuarded(inputs);
  EXPECT_NE(gep, nullptr);
  EXPECT_NE(gep->GetGraph(), nullptr);
  if (!gep->Compiled()) {
    EXPECT_EQ(gep->SetCompiled(0, compute_graph), false);
  }
  EXPECT_EQ(gep->Compiled(), false);
  EXPECT_NE(gep->GetGraph(), nullptr);
  EXPECT_NE(gep->GetSlicedGraph(), nullptr);

  delete ep;
  dlog_setlevel(0, 3, 0);
}

TEST_F(ExecutionPointUT, OneGraphMultiThread_Success) {
  dlog_setlevel(0, 1, 0);

  auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph_))));
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);

  gert::Tensor tensor0 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  gert::Tensor tensor1 = {{{3, 2, 9}, {3, 2, 9}},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  std::vector<gert::Tensor> inputs;
  inputs.emplace_back(std::move(tensor0));
  inputs.emplace_back(std::move(tensor1));

  // 模拟断图结果
  ep = new ExecutionPoint(1, compute_graph, compute_graph);

  {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::mutex mutex_;
    size_t compiled_count = 0;

    // Create 10 threads
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(ThreadFunction, ep, &inputs, &compiled_count, &mutex_, 1);
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
      thread.join();
    }
    EXPECT_EQ(compiled_count, 1);
    EXPECT_EQ(ep->GetSavedCacheNum(), 1);
  }

  // 构造GuardMiss输入，创建GuardedExecutionPoint新实例对象，该对象状态为未编译
  inputs[1].MutableOriginShape().SetDim(0, 10);
  {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::mutex mutex_;
    size_t compiled_count = 0;

    // Create 10 threads
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(ThreadFunction, ep, &inputs, &compiled_count, &mutex_, 2);
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
      thread.join();
    }
    EXPECT_EQ(compiled_count, 1);
    EXPECT_EQ(ep->GetSavedCacheNum(), 2);
  }

  delete ep;
  dlog_setlevel(0, 3, 0);
}
}