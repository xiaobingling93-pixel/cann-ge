/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#include "parser/ascend_graph_parser.h"
#define private public
#include "expr_gen/pipe_perf_expr.h"
#undef private
#include "ascir_ops.h"
#include "expr_gen/arg_list_manager.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph_construct_utils.h"
#include "../../../../../ut/att/testcase/gen_model_info/expr_gen/runtime_stub.h"
#include "common/platform_context.h"

namespace ge {
namespace ascir {
namespace cg {}
}
}
namespace att {
class TestPipePerfExprV2 : public ::testing::Test {
 public:
  static ge::RuntimeStubV2 stub_v_2;
  static void TearDownTestCase() {
    ge::RuntimeStub::UnInstall(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    ge::RuntimeStub::Install(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // dlog_setlevel(GE, 0, 1);
  }
  void TearDown() override {}
};
ge::RuntimeStubV2 TestPipePerfExprV2::stub_v_2;

TEST_F(TestPipePerfExprV2, TestLoadPipeHead) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "LoadV2";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::PIPE_NONE] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::PIPE_NONE);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "775.0");
}

TEST_F(TestPipePerfExprV2, TestUpdatePipeHead) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "LoadV2";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(50), CreateExpr(50)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::PIPE_NONE] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::PIPE_NONE);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "775.0");
}

TEST_F(TestPipePerfExprV2, TestUpdatePipeHeadV1) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "LoadV2";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(50), CreateExpr(50)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "775.0");
}

TEST_F(TestPipePerfExprV2, TestUpdatePipeHeadV2) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "LoadV2";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr(1024), CreateExpr(1024)};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "1174.30004882812");
}

TEST_F(TestPipePerfExprV2, TestUpdatePipeHeadV3) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "StoreV2";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr("z0t_size")};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE3] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE3);
  EXPECT_TRUE(iter != pipe_costs.end());
  EXPECT_EQ(Str(iter->second), "571.0");
}

TEST_F(TestPipePerfExprV2, TestUpdatePipeHeadTernaryOp) {
  TuningSpacePtr test_tuning_space = std::make_shared<TuningSpace>();
  NodeInfo node;
  node.node_type = "LoadV2";
  node.inputs.push_back(std::make_shared<Tensor>());
  node.inputs[0]->repeat = {CreateExpr("z0t_size")};
  node.inputs[0]->data_type_size = 2U;
  test_tuning_space->node_infos.emplace_back(node);
  PipePerfExpr pipe_perf(test_tuning_space);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  pipe_costs[PipeType::AIV_MTE2] = ge::sym::kSymbolZero;
  EXPECT_EQ(pipe_perf.UpdatePipeHead(pipe_costs, ternary_ops), ge::SUCCESS);
  auto iter = pipe_costs.find(PipeType::AIV_MTE2);
  EXPECT_TRUE(iter != pipe_costs.end());
  auto iter2 = ternary_ops.find(iter->second);
  EXPECT_TRUE(iter2 != ternary_ops.end());
  EXPECT_EQ(iter2->second.GetTernaryOpStr(),
        "TernaryOp((2 * z0t_size) < 256, 775.0, 1174.30004882812)");
}
}
