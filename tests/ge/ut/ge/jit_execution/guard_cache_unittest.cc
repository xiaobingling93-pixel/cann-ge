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
#include <gtest/gtest.h>
#include <dlfcn.h>
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "es_ge_test_ops_c.h"
#include "faker/space_registry_faker.h"
#include "common/env_path.h"
#include "common/topo_checker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/symbolizer/symbolic.h"
#include "jit_execution/exe_points/guard_cache.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_symbolizer.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "stub/gert_runtime_stub.h"

namespace ge {
class GuardCacheUT : public testing::Test {
  public:

  // s0xs1 维度的张量  与  s2xs3的张量 且 s0>s2
  ComputeGraphPtr make_computer_graph1() {
    GuardCodegen codegen;
    auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph1_))));
    auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
    auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
    EXPECT_NE(attr, nullptr);
    ShapeEnvGuarder guard(attr);


    // s0xs1 维度的张量  与  s2xs3的张量 且 s0>s2
    auto symbol0 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(0, 0));
    auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

    auto symbol2 = attr->CreateSymbol(2, MakeShared<InputShapeSource>(1, 0));
    auto symbol3 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
    EXPECT_SYMBOL_GT(symbol0, symbol2);
    EXPECT_SYMBOL_EQ(symbol1, symbol3);

    // 根据guard生成代码
    EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
    return compute_graph;
  }

  // s0xs1 维度的张量  与  s2xs3的张量 且 s0=s2
  ComputeGraphPtr make_computer_graph2() {
    GuardCodegen codegen;
    auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph2_))));
    auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
    auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
    EXPECT_NE(attr, nullptr);
    ShapeEnvGuarder guard(attr);


    // s0xs1 维度的张量  与  s2xs3的张量 且 s0=s2
    auto symbol0 = attr->CreateSymbol(2, MakeShared<InputShapeSource>(0, 0));
    auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

    auto symbol2 = attr->CreateSymbol(2, MakeShared<InputShapeSource>(1, 0));
    auto symbol3 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
    EXPECT_SYMBOL_EQ(symbol0, symbol2);
    EXPECT_SYMBOL_EQ(symbol1, symbol3);

    // 根据guard生成代码
    EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
    return compute_graph;
  }

  // s0xs1 维度的张量  与  s2xs3的张量 且 s0<s2
  ComputeGraphPtr make_computer_graph3() {
    GuardCodegen codegen;
    auto ge_graph = std::unique_ptr<Graph>(static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph3_))));
    auto compute_graph = GraphUtilsEx::GetComputeGraph(*ge_graph);
    auto attr = compute_graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
    EXPECT_NE(attr, nullptr);
    ShapeEnvGuarder guard(attr);


    // s0xs1 维度的张量  与  s2xs3的张量 且 s0<s2
    auto symbol0 = attr->CreateSymbol(1, MakeShared<InputShapeSource>(0, 0));
    auto symbol1 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(0, 1));

    auto symbol2 = attr->CreateSymbol(2, MakeShared<InputShapeSource>(1, 0));
    auto symbol3 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
    EXPECT_SYMBOL_LT(symbol0, symbol2);
    EXPECT_SYMBOL_EQ(symbol1, symbol3);

    // 根据guard生成代码
    EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
    return compute_graph;
  }

  gert::Tensor make_tensor(const std::initializer_list<int64_t> &origin_shape) {
    return {{origin_shape, origin_shape},                      // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                          gert::kOnDeviceHbm,                          // placement
                          ge::DT_FLOAT16,                              // data type
                          (void *)0x0};
  }

protected:
 void SetUp() override {
     auto ascend_install_path = EnvPath().GetAscendInstallPath();
     setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
     setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
     graph_ = EsCreateGraphBuilder("Hello");
     guardCheckCache_ = new GuardCheckCache(2, nullptr);
     env = getenv("LD_PRELOAD");
     unsetenv("LD_PRELOAD");
 }
 void TearDown() override {
     EsDestroyGraphBuilder(graph_);
     graph_ = nullptr;
     unsetenv("ASCEND_OPP_PATH");
     unsetenv("LD_LIBRARY_PATH");
     delete guardCheckCache_;
     if (env != nullptr) {
       setenv("LD_PRELOAD", env, 1);
     }
 }
    EsCGraphBuilder *graph_{nullptr};
    EsCGraphBuilder *graph1_{nullptr};
    EsCGraphBuilder *graph2_{nullptr};
    EsCGraphBuilder *graph3_{nullptr};

    GuardCheckCache *guardCheckCache_{nullptr};
    const char *env;
};

TEST_F(GuardCacheUT, load_guard_check_func) {
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

  gert::Tensor tensor0 = make_tensor({3,2,9});
  gert::Tensor tensor1 = make_tensor({3,2,9});
  std::vector<gert::Tensor> inputs;
  inputs.emplace_back(std::move(tensor0));
  inputs.emplace_back(std::move(tensor1));
  // char_t reason[1024];

  auto epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  if (!epm->Compiled()) {
    epm->SetCompiled(0, compute_graph);
  }
  EXPECT_EQ(epm->Compiled(), true);

  epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  EXPECT_EQ(epm->Compiled(), true);

  // bool ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  // EXPECT_EQ(ret, true);
  // 传入非法参数，校验返回值失败
  inputs[1].MutableOriginShape().SetDim(0, 10);
  // ret = reinterpret_cast<GuardCheckFunc>(func)(inputs.data(), 2, reason, 1024);
  epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  EXPECT_EQ(epm->Compiled(), false);
  // EXPECT_EQ(ret, false);
  // std::string reason1(reason);
  // EXPECT_NE(reason1.find("Check Symbol Check Expression: (s0 == s3) failed"), std::string::npos);
  // close(so_fd);
  // dlclose(handle);
  dlog_setlevel(0, 3, 0);
}


TEST_F(GuardCacheUT, check_priority_feat) {
  dlog_setlevel(0, 1, 0);
  graph1_ = EsCreateGraphBuilder("Hello");
  graph2_ = EsCreateGraphBuilder("Hello");
  graph3_ = EsCreateGraphBuilder("Hello");
  auto computer_graph1 = make_computer_graph1();
  auto computer_graph2 = make_computer_graph2();
  auto computer_graph3 = make_computer_graph3();

  // 符合gep1的input
  auto input_for_gep1_0 = make_tensor({3,2,9});
  auto input_for_gep1_1 = make_tensor({2,2,9});
  std::vector<gert::Tensor> inputs_for_gep1;
  inputs_for_gep1.emplace_back(std::move(input_for_gep1_0));
  inputs_for_gep1.emplace_back(std::move(input_for_gep1_1));

  auto gep1 = new GuardedExecutionPoint(nullptr);
  gep1->SetCompiled(1, computer_graph1);
  guardCheckCache_->AddCompiledCompiledGraph(gep1);

  // 符合gep2的input
  auto input_for_gep2_0 = make_tensor({3,2,9});
  auto input_for_gep2_1 = make_tensor({3,2,9});
  std::vector<gert::Tensor> inputs_for_gep2;
  inputs_for_gep2.emplace_back(std::move(input_for_gep2_0));
  inputs_for_gep2.emplace_back(std::move(input_for_gep2_1));

  auto gep2 = new GuardedExecutionPoint(nullptr);
  gep2->SetCompiled(2, computer_graph2);
  guardCheckCache_->AddCompiledCompiledGraph(gep2);

  // 模拟被踩中多次
  guardCheckCache_->FindGuardedExecutionPoint(inputs_for_gep1);
  guardCheckCache_->FindGuardedExecutionPoint(inputs_for_gep1);
  auto gep_find =  guardCheckCache_->FindGuardedExecutionPoint(inputs_for_gep1);
  EXPECT_EQ(gep_find->GetCompiledGraphId(), gep1->GetCompiledGraphId());
  EXPECT_EQ(gep_find->GetPriority(), 3);
  EXPECT_EQ(gep2->GetPriority(), 0);

  EsDestroyGraphBuilder(graph1_);
  EsDestroyGraphBuilder(graph2_);
  EsDestroyGraphBuilder(graph3_);
  dlog_setlevel(0, 3, 0);
}

TEST_F(GuardCacheUT, check_aging_feat) {
  dlog_setlevel(0, 1, 0);
  graph1_ = EsCreateGraphBuilder("Hello");
  graph2_ = EsCreateGraphBuilder("Hello");
  graph3_ = EsCreateGraphBuilder("Hello");
  auto computer_graph1 = make_computer_graph1();
  auto computer_graph2 = make_computer_graph2();
  auto computer_graph3 = make_computer_graph3();

  // 符合gep1的input
  auto input_for_gep1_0 = make_tensor({3,2,9});
  auto input_for_gep1_1 = make_tensor({2,2,9});
  std::vector<gert::Tensor> inputs_for_gep1;
  inputs_for_gep1.emplace_back(std::move(input_for_gep1_0));
  inputs_for_gep1.emplace_back(std::move(input_for_gep1_1));

  auto gep1 = new GuardedExecutionPoint(nullptr);
  gep1->SetCompiled(1, computer_graph1);
  guardCheckCache_->AddCompiledCompiledGraph(gep1);

  // 符合gep2的input
  auto input_for_gep2_0 = make_tensor({3,2,9});
  auto input_for_gep2_1 = make_tensor({3,2,9});
  std::vector<gert::Tensor> inputs_for_gep2;
  inputs_for_gep2.emplace_back(std::move(input_for_gep2_0));
  inputs_for_gep2.emplace_back(std::move(input_for_gep2_1));

  auto gep2 = new GuardedExecutionPoint(nullptr);
  gep2->SetCompiled(2, computer_graph2);
  guardCheckCache_->AddCompiledCompiledGraph(gep2);
  // 模拟被踩中多次
  guardCheckCache_->FindGuardedExecutionPoint(inputs_for_gep1);
  guardCheckCache_->FindGuardedExecutionPoint(inputs_for_gep1);
  auto gep_find =  guardCheckCache_->FindGuardedExecutionPoint(inputs_for_gep1);
  EXPECT_EQ(gep_find->GetCompiledGraphId(), gep1->GetCompiledGraphId());
  EXPECT_EQ(gep_find->GetPriority(), 3);
  EXPECT_EQ(gep2->GetPriority(), 0);

  // 构造一个全新用例且不满足任何cache保存的用例
  // 符合gep2的input
  auto input_for_gep3_0 = make_tensor({3,2,9});
  auto input_for_gep3_1 = make_tensor({4,2,9});
  std::vector<gert::Tensor> inputs_for_gep3;
  inputs_for_gep3.emplace_back(std::move(input_for_gep3_0));
  inputs_for_gep3.emplace_back(std::move(input_for_gep3_1));

  // 尝试加入第三个编译结果，期望将compute_Gprah_id 被
  auto gep3 = new GuardedExecutionPoint(nullptr);
  gep3->SetCompiled(3, computer_graph3);
  guardCheckCache_->AddCompiledCompiledGraph(gep3);

  const std::vector<std::unique_ptr<GuardedExecutionPoint>> & cache_model = guardCheckCache_->GetCache();
  for (const auto & item : cache_model) {
    EXPECT_NE(item->GetCompiledGraphId(), 2);
  }


  EsDestroyGraphBuilder(graph1_);
  EsDestroyGraphBuilder(graph2_);
  EsDestroyGraphBuilder(graph3_);
  dlog_setlevel(0, 3, 0);
}

TEST_F(GuardCacheUT, load_guard_check_func_always_true_when_no_guard_symbol) {
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
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  gert::Tensor tensor0 = make_tensor({3,2,9});
  gert::Tensor tensor1 = make_tensor({3,2,9});
  std::vector<gert::Tensor> inputs;
  inputs.emplace_back(std::move(tensor0));
  inputs.emplace_back(std::move(tensor1));

  auto epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  if (!epm->Compiled()) {
    epm->SetCompiled(0, compute_graph);
  }
  EXPECT_EQ(epm->Compiled(), true);

  epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  EXPECT_EQ(epm->Compiled(), true);

  // 修改不同的input
  inputs[1].MutableOriginShape().SetDim(0, 10);
  epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  EXPECT_EQ(epm->Compiled(), true);
  dlog_setlevel(0, 3, 0);
}

TEST_F(GuardCacheUT, gen_guard_func_verify_guard_num) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
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
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  // 产生guard表达式
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  auto find_log = runtime_stub.GetSlogStub().FindInfoLogEndsWith("GenCheckInfos, symbol_check_infos size: 3");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(0, 3, 0);
}

TEST_F(GuardCacheUT, gen_guard_func_verify_compile_cost) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
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
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  // 产生guard表达式
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);
  
  auto find_log = runtime_stub.GetSlogStub().FindInfoLogEndsWith("The time cost of GuardCompile is");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(0, 3, 0);
}

TEST_F(GuardCacheUT, gen_guard_func_verify_run_cost) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
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
  auto symbol4 = attr->CreateSymbol(3, MakeShared<InputShapeSource>(1, 0));
  auto symbol5 = attr->CreateSymbol(4, MakeShared<InputShapeSource>(1, 1));
  auto symbol6 = attr->CreateSymbol(5, MakeShared<InputShapeSource>(1, 2));
  // 产生guard表达式
  EXPECT_SYMBOL_EQ(symbol0, symbol4);
  EXPECT_SYMBOL_EQ(symbol1, symbol5);
  EXPECT_SYMBOL_EQ(symbol2, symbol6);

  // 根据guard生成代码
  EXPECT_EQ(codegen.GuardFuncCodegenAndCompile(compute_graph), ge::GRAPH_SUCCESS);

  gert::Tensor tensor0 = make_tensor({3,2,9});
  gert::Tensor tensor1 = make_tensor({3,2,9});
  std::vector<gert::Tensor> inputs;
  inputs.emplace_back(std::move(tensor0));
  inputs.emplace_back(std::move(tensor1));

  auto epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  if (!epm->Compiled()) {
    epm->SetCompiled(0, compute_graph);
  }
  EXPECT_EQ(epm->Compiled(), true);
  epm = guardCheckCache_->FindOrCreateGuarded(inputs);
  EXPECT_NE(epm, nullptr);
  EXPECT_EQ(epm->Compiled(), true);

  auto find_log = runtime_stub.GetSlogStub().FindInfoLogEndsWith("The time cost of GuardMatch is");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(0, 3, 0);
}
}