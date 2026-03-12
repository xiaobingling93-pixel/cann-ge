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

namespace ge {
namespace ascir {
namespace cg {
Status BuildVectorFunctionSubgraph(ge::AscGraph &subgraph) {
  auto ND = ge::Symbol("ND");
  auto nd = subgraph.CreateAxis("nd", ND);
  auto [ndB, ndb] = subgraph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = subgraph.TileSplit(ndb->id);
  auto data1 = subgraph.CreateContiguousData("input1", DT_FLOAT, {*ndbt});
  auto load1 = Load("load1", data1);
  auto abs1 = Abs("abs1", load1);
  auto sub1 = Sub("sub1", abs1, abs1);
  auto store1 = Store("store1", sub1);
  auto output1 = Output("output1", store1);
  return ge::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0WithVectorFunc(ge::AscGraph &graph) {
  auto S0 = ge::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0});
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto vector_func = ascir_op::VectorFunc("vector_func");
      vector_func.SetAttr("sub_graph_name", "vector_func");
      vector_func.InstanceOutputy(1);
      vector_func.x = {load1};
      *vector_func.y[0].axis = {z0bT->id, z0bt->id};
      *(vector_func.y[0].repeats) = {z0bT->size, z0bt->size};
      *(vector_func.y[0].strides) = {z0bt->size, ge::Symbol(1)};
      *vector_func.y[0].vectorized_axis = {z0bt->id};
      auto store1 = Store("store1", vector_func.y[0]);
      GE_ASSERT_SUCCESS(att::GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0bt}, {load1, store1}, 1));
      auto output1 = Output("output1", store1);
    }
  }
  constexpr char_t vector_func_node_name[] = "vector_func";
  AscGraph subgraph(vector_func_node_name);
  GE_ASSERT_SUCCESS(BuildVectorFunctionSubgraph(subgraph));
  graph.AddSubGraph(subgraph);
  auto node = graph.FindNode(vector_func_node_name);
  GE_ASSERT_NOTNULL(node);
  node->attr.sched.axis = {z0bT->id};
  node->attr.sched.loop_axis = z0bT->id;
  ge::AttrUtils::SetStr(node->GetOpDescBarePtr(), "sub_graph_name", vector_func_node_name);
  return ge::SUCCESS;
}
}
}
}
namespace att {
static TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
class TestPipePerfExpr : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override
  {
  }
  void TearDown() override
  {
  }
};

TEST_F(TestPipePerfExpr, case_get_perf_for_loop_with_vf)
{
  ge::AscGraph graph("graph");
  ASSERT_EQ(ge::ascir::cg::BuildConcatGroupAscendGraphS0WithVectorFunc(graph), ge::SUCCESS);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr ts = std::make_shared<TuningSpace>();
  ASSERT_NE(ts, nullptr);
  att::AscendGraphParser ascend_graph_parser(ts);
  EXPECT_EQ(ascend_graph_parser.GraphParser(graph), ge::SUCCESS);
  EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(ts), ge::SUCCESS);
  PipePerfExpr pipe_perf(ts);
  std::map<PipeType, Expr> pipe_costs;
  std::map<Expr, TernaryOp, ExprCmp> exe_times;
  Expr head_cost;
  EXPECT_EQ(pipe_perf.GetPerfExpr(pipe_costs, exe_times, head_cost), ge::SUCCESS);
  ASSERT_EQ(pipe_costs.size(), 3);
  std::vector<std::pair<Expr, Expr>> replace_vars;
  std::string var_name;
  std::string exe_time = "exe_time";
  for (const auto &pair : exe_times) {
    var_name = Str(pair.first);
    GELOGD("var_name: %s=%s", var_name.c_str(), pair.second.GetTernaryOpStr().c_str());
    if (var_name.rfind(exe_time) == (var_name.length() - exe_time.length())) {
      EXPECT_EQ(pair.second.GetTernaryOpStr(), "Ceiling((z0b_size / (z0bt_size)))");
    }
  }
  for (const auto &pipe_cost : pipe_costs) {
    std::cout << "pipe_cost.first: " << static_cast<int32_t>(pipe_cost.first)
              << ", pipe_cost.second: " << pipe_cost.second << std::endl;
  }
}
}