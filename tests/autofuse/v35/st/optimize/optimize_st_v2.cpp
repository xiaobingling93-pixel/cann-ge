/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#include "autoschedule/autoschedule.h"
#include "autoschedule/alignment_handler.h"
#include "platform_context.h"
#undef private
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "attr_utils.h"
#include "graph/debug/ge_op_types.h"
#include "autoschedule/axis_group.h"
#include "schedule_utils.h"
#include "attribute_group/attr_group_shape_env.h"
#include "autofuse/utils/autofuse_attrs.h"
#include "fused_graph/fused_graph_unfolder.h"
#include "graph/debug/ge_attr_define.h"
#include "task_generator/concat_group_partitioner.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "platform_context.h"
#include "platformv2.h"
#include "tests/autofuse/depends/runtime/src/runtime_stub.h"
#include "../../st/optimize/runtime_stub.h"
#include "backend/backend_spec.h"
#include "codegen.h"
#include "ascgraph_info_complete.h"
#include "tests/autofuse/framework/easy_asc_graph/asc_graph_builder.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ge::testing;
using optimize::autoschedule::AxisGroup;

namespace {
class OptimizerStV2 : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }

  optimize::Optimizer optimizer;

  OptimizerStV2() : optimizer(optimize::OptimizerOptions{}) {}

  static std::string ExpressToStr(std::vector<ge::Expression> exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }

  static std::string AxisToStr(ge::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    std::stringstream ss;
    for (auto axis_id : node->outputs[0].attr.axis) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss.str();
  }
};

namespace optimize {
TEST_F(OptimizerStV2, ElewiseAndBrcCanMerge) {
  ge::AscGraph graph1("graph1");
  graph1.SetGraphType(ge::AscGraphType::kImplGraph);
  auto ONE = Symbol(1);
  const Expression s0 = graph1.CreateSizeVar("s0");
  const Expression s1 = graph1.CreateSizeVar("s1");
  auto z0 = graph1.CreateAxis("z0", s0);
  auto z1 = graph1.CreateAxis("z1", s1);
  ge::ascir_op::Data data0("data0", graph1);
  data0.ir_attr.SetIndex(0);
  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};
  ge::ascir_op::Output out0("out0");
  out0.x = load0.y;
  out0.y.dtype = ge::DT_FLOAT16;
  out0.ir_attr.SetIndex(0);

  ge::AscGraph graph2("graph2");
  graph2.SetGraphType(ge::AscGraphType::kImplGraph);
  const Expression s1_0 = graph1.CreateSizeVar("s0");
  auto z1_0 = graph1.CreateAxis("z0", s1_0);
  ge::ascir_op::Data data1_0("data1_0", graph2);
  data1_0.ir_attr.SetIndex(0);
  ge::ascir_op::Load load1_0("load1_0");
  load1_0.x = data1_0.y;
  load1_0.attr.sched.axis = {z0.id};
  *load1_0.y.axis = {z0.id};
  *load1_0.y.repeats = {s0};
  *load1_0.y.strides = {ONE};
  ge::ascir_op::Output out1_0("out1_0");
  out1_0.x = load1_0.y;
  out1_0.y.dtype = ge::DT_FLOAT16;
  out1_0.ir_attr.SetIndex(0);

  AxisGroup lhs;
  EXPECT_EQ(GenAscGraphAxisGroup(graph1, lhs), 0);

  AxisGroup rhs;
  EXPECT_EQ(GenAscGraphAxisGroup(graph2, rhs), 0);
  // CanFuse do axis-mapping
  rhs.y_group.emplace_back(1);

  AxisGroup res;
  EXPECT_TRUE(CanMergeAxisGroup(lhs, rhs, res));

  EXPECT_EQ(res, lhs);
}

// 使用 AscGraphBuilder 重写
TEST_F(OptimizerStV2, ReduceNeedAlignment) {
  auto graph = AscGraphBuilder("ReduceNeedAlignment")
    .Loops({7, 8, 9, 10})
    .Data("arg4_1", 0, ge::DT_FLOAT)
    .Load("b0_load", "arg4_1")
    .Abs("abs", "b0_load")
    .Max("b0_max", "abs", {0, 2})  // reduce on axis 0 and 2
    .Store("b3_store", "b0_max")
    .Output("buf3", "b3_store", 0, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 5UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);

  const auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  const auto &reduce_node = impl_graph.FindNode("b0_max");
  std::vector<Expression> golden_stride = {
      Zero,
      Symbol(16),
      One,
  };
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerStV2, NotRemovePad) {
  auto s0 = Symbol(2);
  auto s1 = Symbol(3);
  auto s2 = Symbol(3);
  auto graph = AscGraphBuilder("Autoschedule_autoschedule_removepad_broadcast")
    .Loops({s0, s1, s2})
    .Data("data0", 0, {ge::ops::One, s1, s2}, {ge::sym::kSymbolZero, s2, ge::sym::kSymbolOne}, ge::DT_FLOAT16)
    .Load("load0", "data0", {ge::ops::One, s1, s2}, {ge::sym::kSymbolZero, s2, ge::sym::kSymbolOne})
    .Broadcast("brc0", "load0", {s0, s1, s2})
    .Data("data1", 1, {s0, s1, s2}, {s1 * s2, s2, ge::sym::kSymbolOne}, ge::DT_FLOAT16)
    .Load("load1", "data1")
    .Add("add0", "brc0", "load1")
    .Store("store0", "add0")
    .Output("y0", "store0", 0, ge::DT_FLOAT16)
    .Data("data2", 2, {ge::ops::One, s1, s2}, {ge::sym::kSymbolZero, s2, ge::sym::kSymbolOne}, ge::DT_FLOAT16)
    .Load("load2", "data2", {ge::ops::One, s1, s2}, {ge::sym::kSymbolZero, s2, ge::sym::kSymbolOne})
    .Broadcast("brc2", "load2", {s0, s1, s2})
    .Mul("mul0", "load1", "brc2")
    .Store("store1", "mul0")
    .Output("y1", "store1", 1, ge::DT_FLOAT16)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].GetName(), "Autoschedule_autoschedule_removepad_broadcast_0_B0Y0_S0G0C0");
}

/**
 * load0
 *   \
 * brc0
 *   \
 * brc1
 *   \
 *  brc2   load1
 *     \    /
 *      add
 *       |
 *     store
 */
TEST_F(OptimizerStV2, ContinuesBroadcastOptimization_3Brc) {
  const Expression s0 = ge::Symbol("s0");
  const Expression s1 = ge::Symbol("s1");
  const Expression s2 = ge::Symbol("s2");
  const Expression s3 = ge::Symbol("s3");

  // Load with padding: shape {One, One, One, s3}, strides {Zero, Zero, Zero, One}
  std::vector<Expression> load0_shape = {ge::sym::kSymbolOne, ge::sym::kSymbolOne, ge::sym::kSymbolOne, s3};
  std::vector<Expression> load0_strides = {ge::sym::kSymbolZero, ge::sym::kSymbolZero, ge::sym::kSymbolZero, ge::sym::kSymbolOne};

  // Broadcast 0: shape {One, One, s2, s3}, strides {Zero, Zero, s3, One}
  std::vector<Expression> brc0_shape = {ge::sym::kSymbolOne, ge::sym::kSymbolOne, s2, s3};
  std::vector<Expression> brc0_strides = {ge::sym::kSymbolZero, ge::sym::kSymbolZero, s3, ge::sym::kSymbolOne};

  // Broadcast 1: shape {One, s1, s2, s3}, strides {Zero, s2*s3, s3, One}
  std::vector<Expression> brc1_shape = {ge::sym::kSymbolOne, s1, s2, s3};
  std::vector<Expression> brc1_strides = {ge::sym::kSymbolZero, s2 * s3, s3, ge::sym::kSymbolOne};

  // Broadcast 2: shape {s0, s1, s2, s3}, strides {s1*s2*s3, s2*s3, s3, One}
  std::vector<Expression> brc2_shape = {s0, s1, s2, s3};
  std::vector<Expression> brc2_strides = {s1 * s2 * s3, s2 * s3, s3, ge::sym::kSymbolOne};

  // Store strides
  std::vector<Expression> store_strides = {s1 * s2 * s3, s2 * s3, s3, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("Continues_3Broadcast_Optimization_graph")
    .Loops({s0, s1, s2, s3})
    .Data("data0", 0, ge::DT_FLOAT16)
    .Load("load0", "data0", load0_shape, load0_strides)
    .Broadcast("brc0", "load0", brc0_shape)
    .Broadcast("brc1", "brc0", brc1_shape)
    .Broadcast("brc2", "brc1", brc2_shape)
    .Data("data1", 0, ge::DT_FLOAT16)
    .Load("load1", "data1")
    .Add("add", "brc2", "load1")
    .Store("store", "add", {}, store_strides)
    .Output("y", "store", 0, ge::DT_FLOAT16)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;

  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
}

TEST_F(OptimizerStV2, ContinuesBroadcastOptimization_BABAB) {
  const Expression s0 = ge::Symbol(4);
  const Expression s1 = ge::Symbol(8);
  const Expression s2 = ge::Symbol(16);
  const Expression s3 = ge::Symbol(64);
  const Expression s4 = ge::Symbol(32);

  // Load 1 with padding: shape {One, s1, One, s3, One}, strides {Zero, s3, Zero, One, Zero}
  std::vector<Expression> load1_shape = {ge::sym::kSymbolOne, s1, ge::sym::kSymbolOne, s3, ge::sym::kSymbolOne};
  std::vector<Expression> load1_strides = {ge::sym::kSymbolZero, s3, ge::sym::kSymbolZero, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  // Broadcast 0: shape {s0, s1, One, s3, One}, strides {s1*s3, s3, Zero, One, Zero}
  std::vector<Expression> brc0_shape = {s0, s1, ge::sym::kSymbolOne, s3, ge::sym::kSymbolOne};
  std::vector<Expression> brc0_strides = {s1 * s3, s3, ge::sym::kSymbolZero, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  // Broadcast 1: shape {s0, s1, s2, s3, One}, strides {s1*s2*s3, s2*s3, s3, One, Zero}
  std::vector<Expression> brc1_shape = {s0, s1, s2, s3, ge::sym::kSymbolOne};
  std::vector<Expression> brc1_strides = {s1 * s2 * s3, s2 * s3, s3, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  // Broadcast 2: shape {s0, s1, s2, s3, s4}, strides {s1*s2*s3*s4, s2*s3*s4, s3*s4, s4, One}
  std::vector<Expression> brc2_shape = {s0, s1, s2, s3, s4};
  std::vector<Expression> brc2_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, ge::sym::kSymbolOne};

  // Store strides
  std::vector<Expression> store_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("Continues_3Broadcast_Optimization_graph")
    .Loops({s0, s1, s2, s3, s4})
    .Data("x0", 0, ge::DT_FLOAT)
    .Load("load0", "x0")
    .Data("x1", 1, ge::DT_FLOAT)
    .Load("load1", "x1", load1_shape, load1_strides)
    .Broadcast("brc0", "load1", brc0_shape)
    .Broadcast("brc1", "brc0", brc1_shape)
    .Broadcast("brc2", "brc1", brc2_shape)
    .Store("store1", "brc2", {}, store_strides)
    .Output("y1", "store1", 1, ge::DT_FLOAT)
    .Add("add0", "load0", "brc2")
    .Store("store", "add0", {}, store_strides)
    .Output("y", "store", 0, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);

  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;

  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 10UL);

  auto impl_graph = impl_graphs[0];
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 10);
  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
}

/**
 *           data0
 *             |
 *           load0
 *             |
 *         broadcast
 *          /     \
 *        Exp     abs
 *          \      /
 *             Mul
 *              |
 *            store
 *              |
 *           output
 */
TEST_F(OptimizerStV2, NddmaCaseBrcOutputWithMultiRef) {
  const Expression s0 = ge::Symbol(64);
  const Expression s1 = ge::Symbol(64);

  // Load scalar: shape {1, 1}, strides {1, 1}
  std::vector<Expression> load_shape = {ge::sym::kSymbolOne, ge::sym::kSymbolOne};
  std::vector<Expression> load_strides = {ge::sym::kSymbolOne, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1})
    .Data("data0", 0, ge::DT_FLOAT)
    .Load("load0", "data0", load_shape, load_strides)
    .Broadcast("broadcast", "load0", {0, 1})  // broadcast on both axes
    .Exp("exp0", "broadcast")
    .Abs("abs0", "broadcast")
    .Mul("mul0", "exp0", "abs0")
    .Store("store", "mul0")
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(OptimizerStV2, NddmaCaseAlignTailBrcScoreFunc) {
  ge::AscGraph graph("gen_nddma");
  const auto dtype = ge::DT_UINT8;

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.repeats = {s0, Symbol(1)};
  *load0.y.strides = {Symbol(1), Symbol(0)};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = dtype;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, ge::ops::One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = dtype;
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ge::ops::One};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = dtype;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = dtype;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];

  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 2);

  const auto score_func_iter = schedule_group.graph_name_to_score_funcs.find(schedule_group.impl_graphs[2].GetName());
  ASSERT_NE(score_func_iter, schedule_group.graph_name_to_score_funcs.end());
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  return -1;\n"
      "}\n";
  EXPECT_EQ(score_func_iter->second, res);
}

TEST_F(OptimizerStV2, NddmaCaseAlignTailBrcScoreFunc_Dynamic) {
  ge::AscGraph graph("gen_nddma");
  const auto dtype = ge::DT_FLOAT16;

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.repeats = {s0, Symbol(1)};
  *load0.y.strides = {Symbol(1), Symbol(0)};

  Broadcast broadcast0("broadcast");
  broadcast0.x = load0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = dtype;
  *broadcast0.y.repeats = {s0, s1};
  *broadcast0.y.strides = {s1, ge::ops::One};

  Exp exp0("exp0");
  exp0.attr.sched.axis = {z0.id, z1.id};
  exp0.x = broadcast0.y;
  *exp0.y.axis = {z0.id, z1.id};
  exp0.y.dtype = dtype;
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ge::ops::One};

  Abs abs0("abs0");
  abs0.x = broadcast0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  abs0.y.dtype = dtype;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.x1 = exp0.y;
  mul0.x2 = abs0.y;
  mul0.y.dtype = dtype;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];

  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 2);
  const auto score_func_iter = schedule_group.graph_name_to_score_funcs.find(schedule_group.impl_graphs[2].GetName());
  ASSERT_NE(score_func_iter, schedule_group.graph_name_to_score_funcs.end());
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  const auto tail_size = static_cast<int64_t>((2 * tiling_data.s1));\n"
      "  if (tail_size % 32 == 0) { return -1; }\n"
      "  if (tail_size > 4096) { return -1; }\n"
      "  return 0;\n"
      "}\n";
  EXPECT_EQ(score_func_iter->second, res);
}

TEST_F(OptimizerStV2, NddmaCaseLargeTailBrcScoreFunc) {
  const Expression s0 = ge::Symbol(8);
  const Expression s1 = ge::Symbol(2012);

  // Load with padding: shape {s0, 1}, strides {1, 0}
  std::vector<Expression> load0_shape = {s0, ge::sym::kSymbolOne};
  std::vector<Expression> load0_strides = {ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1})
    .Data("data0", 0, ge::DT_FLOAT)
    .Load("load0", "data0", load0_shape, load0_strides)
    .Broadcast("broadcast", "load0", {1})  // broadcast on axis 1
    .Exp("exp0", "broadcast")
    .Abs("abs0", "broadcast")
    .Mul("mul0", "exp0", "abs0")
    .Store("store", "mul0")
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 2);

  const auto score_func_iter = schedule_group.graph_name_to_score_funcs.find(schedule_group.impl_graphs[2].GetName());
  ASSERT_NE(score_func_iter, schedule_group.graph_name_to_score_funcs.end());
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  return -1;\n"
      "}\n";
  EXPECT_EQ(score_func_iter->second, res);
}


TEST_F(OptimizerStV2, NddmaCaseLargeTailBrc_Dynamic) {
  const Expression s0 = ge::Symbol("s0");
  const Expression s1 = ge::Symbol("s1");

  // Load with padding: shape {s0, 1}, strides {1, 0}
  std::vector<Expression> load0_shape = {s0, ge::sym::kSymbolOne};
  std::vector<Expression> load0_strides = {ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1})
    .Data("data0", 0, ge::DT_FLOAT)
    .Load("load0", "data0", load0_shape, load0_strides)
    .Broadcast("broadcast", "load0", {1})  // broadcast on axis 1
    .Exp("exp0", "broadcast")
    .Abs("abs0", "broadcast")
    .Mul("mul0", "exp0", "abs0")
    .Store("store", "mul0")
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  const auto schedule_group = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  ASSERT_EQ(schedule_group.graph_name_to_score_funcs.size(), 2);
}

/**
 *           data0         data1
 *             |             |
 *           load0         load1
 *             |             |
 *         broadcast0   broadcast1
 *             \            /
 *                   Mul
 *                    |
 *                  store
 *                    |
 *                 output
 */
TEST_F(OptimizerStV2, NddmaCaseWithMultiNddma) {
  const Expression s0 = ge::Symbol(64);
  const Expression s1 = ge::Symbol(64);

  // Load0 with padding: shape {One, s1}, strides {Zero, One}
  std::vector<Expression> load0_shape = {ge::sym::kSymbolOne, s1};
  std::vector<Expression> load0_strides = {ge::sym::kSymbolZero, ge::sym::kSymbolOne};

  // Load1 with padding: shape {s0, One}, strides {One, Zero}
  std::vector<Expression> load1_shape = {s0, ge::sym::kSymbolOne};
  std::vector<Expression> load1_strides = {ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1})
    .Data("data0", 0, ge::DT_FLOAT)
    .Data("data1", 1, ge::DT_FLOAT)
    .Load("load0", "data0", load0_shape, load0_strides)
    .Broadcast("broadcast0", "load0", {0})  // broadcast on axis 0
    .Load("load1", "data1", load1_shape, load1_strides)
    .Broadcast("broadcast1", "load1", {1})  // broadcast on axis 1
    .Mul("mul0", "broadcast0", "broadcast1")
    .Store("store", "mul0")
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, concat_last1dim) {
  ge::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = ge::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = ge::DT_INT64;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_INT64;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, One};

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_INT64;
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_INT64;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, One};

  ge::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_INT64;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_INT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id};
  output0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  output0.y.dtype = ge::DT_INT64;
  output0.ir_attr.SetIndex(0);

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_INT64;
  x2.ir_attr.SetIndex(2);

  ge::ascir_op::Load load3("load3");
  load3.x = x2.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_INT64;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};

  ge::ascir_op::Store store1("store1");
  store1.x = load3.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = ge::DT_INT64;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.x = store1.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_INT64;
  y.ir_attr.SetIndex(0);

  auto axis = graph.GetAllAxis();
  axis.erase(axis.begin());
  const auto graph_attr = ge::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  graph_attr->axis = axis;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto res_axis = impl_graph.GetAllAxis();
  for (size_t i = 0UL; i < res_axis.size(); i++) {
    EXPECT_EQ(res_axis[i]->id, i);
  }

  auto load_node = impl_graph.FindNode("load");
  ASSERT_NE(nullptr, load_node);
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");
  auto concat_node = impl_graph.FindNode("concat");
  ASSERT_NE(nullptr, concat_node);
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[0].Str().get()), "2");
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(OptimizerStV2, LoadToNddmaCase) {
  const Expression s0 = ge::Symbol(129);
  const Expression s1 = ge::Symbol(32);
  const Expression s2 = ge::Symbol(32);
  const Expression s3 = ge::Symbol(68);

  // Load1 with padding: shape {s0, s1, s2, One}, strides {s1*s2, s2, One, Zero}
  std::vector<Expression> load1_shape = {s0, s1, s2, ge::sym::kSymbolOne};
  std::vector<Expression> load1_strides = {s1 * s2, s2, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  // Sum output: shape {s0, One, One, s3}, strides {s3, Zero, Zero, One}
  std::vector<Expression> sum_shape = {s0, ge::sym::kSymbolOne, ge::sym::kSymbolOne, s3};
  std::vector<Expression> sum_strides = {s3, ge::sym::kSymbolZero, ge::sym::kSymbolZero, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("gen_load_to_nddma")
    .Loops({s0, s1, s2, s3})
    .Data("data0", 0, ge::DT_FLOAT)
    .Data("data1", 1, ge::DT_FLOAT)
    .Load("load0", "data0")
    .Load("load1", "data1", load1_shape, load1_strides)
    .Broadcast("broadcast1", "load1", {3})  // broadcast on axis 3
    .Mul("mul0", "load0", "broadcast1")
    .Sum("sum0", "mul0", {1, 2})  // reduce on axis 1 and 2
    .Store("store", "sum0", sum_shape, sum_strides)
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
  }
}

/**
 *           data0         data1
 *             |             |
 *           load0         load1
 *             |             |
 *             |           Cast1
 *             |             |
 *             |         broadcast1
 *             \            /
 *                   Mul
 *                    |
 *                  store
 *                    |
 *                 output
 */
TEST_F(OptimizerStV2, LoadCastBrcCase) {
  auto s0 = Sym(256);
  auto s1 = Sym(50);
  auto s2 = Sym(16);

  std::vector<Expression> load1_shape = {s0, s1, ge::sym::kSymbolOne};
  std::vector<Expression> load1_strides = {s1, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1, s2})
    .Data("data0", 0, ge::DT_FLOAT)
    .Data("data1", 1, ge::DT_UINT8)
    .Load("load0", "data0")
    .Load("load1", "data1", load1_shape, load1_strides)
    .Cast("cast1", "load1", ge::DT_FLOAT)
    .Broadcast("broadcast1", "cast1", {2})  // broadcast on axis 2
    .Mul("mul0", "load0", "broadcast1")
    .Store("store", "mul0")
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Cast");
    }
  }
}

TEST_F(OptimizerStV2, LoadCastAndTailAxisBrcCase) {
  auto s0 = Sym(256);
  auto s1 = Sym(50);
  auto s2 = Sym(16);

  std::vector<Expression> load0_shape = {s0, s1, ge::sym::kSymbolOne};
  std::vector<Expression> load0_strides = {s1, ge::sym::kSymbolOne, ge::sym::kSymbolZero};
  std::vector<Expression> load1_shape = {s0, ge::sym::kSymbolOne, ge::sym::kSymbolOne};
  std::vector<Expression> load1_strides = {ge::sym::kSymbolOne, ge::sym::kSymbolZero, ge::sym::kSymbolZero};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1, s2})
    .Data("data0", 0, ge::DT_FLOAT)
    .Data("data1", 1, ge::DT_UINT8)
    .Load("load0", "data0", load0_shape, load0_strides)
    .Load("load1", "data1", load1_shape, load1_strides)
    .Cast("cast1", "load1", ge::DT_FLOAT)
    .Broadcast("broadcast1", "cast1", {1})  // broadcast on axis 1
    .Mul("mul0", "load0", "broadcast1")
    .Broadcast("broadcast2", "mul0", {2})  // broadcast on axis 2
    .Store("store", "broadcast2")
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Cast");
    }
  }
}

/**
 *           data0         data1
 *             |             |
 *           load0         load1
 *             |             |
 *             |           Cast1
 *             |             |
 *             |         broadcast1
 *             \            /
 *                   Mul
 *                    |
 *                   Min
 *                    |
 *                  store
 *                    |
 *                 output
 */
TEST_F(OptimizerStV2, LoadCastBrcMulMinCase) {
  AscGraph graph("nddma_alignment");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1, One};
  *cast1.y.strides = {s1, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = cast1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2};
  *broadcast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  Min min0("min0");
  min0.attr.sched.axis = {z0.id, z1.id, z2.id};
  min0.x = mul0.y;
  min0.y.dtype = ge::DT_FLOAT;
  *min0.y.axis = {z0.id, z1.id, z2.id};
  *min0.y.repeats = {One, s1, s2};
  *min0.y.strides = {Zero, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = min0.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {One, s1, s2};
  *store_op.y.strides = {Zero, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Cast");
    }
  }
}

TEST_F(OptimizerStV2, LoadOpSequenceAdjustCase) {
  auto s0 = Symbol(64);
  auto s1 = Symbol(64);
  auto graph = AscGraphBuilder("reorder_load_op")
    .Loops({s0, s1})
    .Data("data0", 0, {ge::ops::One, ge::ops::One}, {ge::sym::kSymbolZero, ge::sym::kSymbolZero}, ge::DT_FLOAT)
    .Load("load0", "data0", {ge::ops::One, ge::ops::One}, {ge::sym::kSymbolZero, ge::sym::kSymbolZero})
    .Broadcast("broadcast0", "load0", {s0, s1})
    .Data("data1", 1)
    .Load("load1", "data1")
    .template Op<ascir_op::Abs>("abs", {"load1"}, {s0, s1}, {s1, ge::ops::One}, ge::DT_FLOAT16)
    .Data("data2", 2, {ge::ops::One, ge::ops::One}, {ge::sym::kSymbolZero, ge::sym::kSymbolZero}, ge::DT_FLOAT)
    .Load("load2", "data2", {ge::ops::One, ge::ops::One}, {ge::sym::kSymbolZero, ge::sym::kSymbolZero})
    .Broadcast("broadcast1", "load2", {s0, s1})
    .Add("add", "abs", "broadcast1")
    .Mul("mul", "broadcast0", "add")
    .Store("store", "mul")
    .Output("output", "store", 8)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(OptimizerStV2, BackendSpec) {
  auto spec = ::optimize::BackendSpec::GetInstance();
  ASSERT_TRUE(spec != nullptr);
  ASSERT_EQ(spec->concat_max_input_num, 512);
}

TEST_F(OptimizerStV2, ConcatTailDim_SplitConcat_LargeRowNum) {
  ge::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{64, 6, 28, 42};
  auto s0 = graph.CreateSizeVar(64 * 64);
  auto concat_size = ge::Expression(ge::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    ge::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = ge::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, ge::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, ge::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  ::optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<::optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), ge::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << ge::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 1);
}

TEST_F(OptimizerStV2, OneAxisSliceNoNeedAlign) {
  const Expression s0 = ge::Symbol(10);
  const Expression s1 = ge::Symbol(4);
  const Expression s2 = ge::Symbol(3);

  // Load 使用自定义 shape {s0, s1} 和 strides {s1 * s2, s2}
  std::vector<Expression> load_shape = {s0, s1};
  std::vector<Expression> load_strides = {s1 * s2, s2};
  std::vector<Expression> store_strides = {s1 * s2, s2};

  auto graph = ge::testing::AscGraphBuilder("shorten_load")
    .Loops({s0, s1})
    .Data("x0", 0)
    .Load("load0", "x0", load_shape, load_strides)
    .Store("store5", "load0", {}, store_strides)
    .Output("output5", "store5", 0)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  auto impl_graph = fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto load_node = impl_graph.FindNode("load0");
  std::vector<ge::Expression> golden_stride{ge::Symbol(4), ge::sym::kSymbolOne};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerStV2, TwoAxisSliceNeedAlign) {
  const Expression s0 = ge::Symbol(10);
  const Expression s1 = ge::Symbol(4);
  const Expression s2 = ge::Symbol(3);
  const Expression s3 = ge::Symbol(2);

  // Load 使用自定义 shape {s0, s1} 和 strides {s1 * s2 * s3, s2}
  std::vector<Expression> load_shape = {s0, s1};
  std::vector<Expression> load_strides = {s1 * s2 * s3, s2};
  std::vector<Expression> store_strides = {s1 * s2 * s3, s2};

  auto graph = ge::testing::AscGraphBuilder("shorten_load")
    .Loops({s0, s1})
    .Data("x0", 0)
    .Load("load0", "x0", load_shape, load_strides)
    .Store("store5", "load0", {}, store_strides)
    .Output("output5", "store5", 0)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  auto impl_graph = fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  auto load_node = impl_graph.FindNode("load0");
  std::vector<ge::Expression> golden_stride{ge::Symbol(4), ge::sym::kSymbolOne};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerStV2, NoNeedAlign_AABToARA) {
  const Expression s0 = ge::Symbol(3);

  // Load: shape {s0, s0, 1}, strides {s0, 1, 0}
  std::vector<Expression> load_shape = {s0, s0, ge::sym::kSymbolOne};
  std::vector<Expression> load_strides = {s0, ge::sym::kSymbolOne, ge::sym::kSymbolZero};

  // Max output: shape {s0, 1, s0}, strides {s0, 0, 1}
  std::vector<Expression> max_shape = {s0, ge::sym::kSymbolOne, s0};
  std::vector<Expression> max_strides = {s0, ge::sym::kSymbolZero, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("shorten_load")
    .Loops({s0, s0, s0})
    .Data("x0", 0)
    .Load("load0", "x0", load_shape, load_strides)
    .Broadcast("brc", "load0", {2})  // broadcast on axis 2
    .Max("max", "brc", {1})  // Max reduce on axis 1
    .Store("store5", "max", max_shape, max_strides)
    .Output("output5", "store5", 0)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);

  auto impl_graph = fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  // load no need to align
  auto load_node = impl_graph.FindNode("load0");
  size_t total_size = load_node->outputs[0].attr.vectorized_strides.size();
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides[total_size - 1], ge::sym::kSymbolZero);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides[total_size - 2], ge::sym::kSymbolOne);
}

TEST_F(OptimizerStV2, NddmaCaseTranspose021OutputWithSingleRef) {
  const Expression s0 = ge::Symbol(32);
  const Expression s1 = ge::Symbol(64);
  const Expression s2 = ge::Symbol(16);

  // Transpose output: shape {s0, s2, s1}, strides {s2 * s1, s1, One}
  std::vector<Expression> transpose_strides = {s2 * s1, s1, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("Transpose_gen_nddma")
    .Loops({s0, s1, s2})
    .Data("data0", 0, ge::DT_FLOAT)
    .Load("load0", "data0")
    .Transpose("transpose", "load0", {0, 2, 1})  // transpose axes: {0, 2, 1}
    .Store("store", "transpose", {}, transpose_strides)
    .Output("output", "store", 8, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, LoadCastTransposeCase) {
  const Expression s0 = ge::Symbol(64);
  const Expression s1 = ge::Symbol(64);

  // Transpose output: shape {s1, s0}, strides {s0, One}
  std::vector<Expression> transpose_strides = {s0, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1})
    .Data("data0", 0, ge::DT_FLOAT16)
    .Load("load0", "data0")
    .Cast("cast1", "load0", ge::DT_FLOAT)
    .Transpose("transpose", "cast1", {1, 0})  // transpose axes: {1, 0}
    .Store("store", "transpose", {}, transpose_strides)
    .Output("output", "store", 5, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, LoadGEWhereTransposeCase) {
  auto s0 = Symbol(41);
  auto s1 = Symbol(54);
  auto s2 = Symbol(38);
  auto s3 = Symbol(55);
  auto graph = AscGraphBuilder("gen_nddma")
    .Loops({s0, s1, s2, s3})
    .Data("data0", 0)
    .Load("load0", "data0")
    .Data("data1", 1)
    .Load("load1", "data1")
    .template Op<ascir_op::Ge>("ge", {"load0", "load1"}, {s0, s1, s2, s3}, {s1 * s2 * s3, s2 * s3, s3, ge::ops::One}, ge::DT_UINT8)
    .template Op<ascir_op::Where>("where", {"ge", "load1"}, {s0, s1, s2, s3}, {s1 * s2 * s3, s2 * s3, s3, ge::ops::One}, ge::DT_FLOAT)
    .Transpose("transpose", "where", {0, 3, 1, 2})
    .Store("store", "transpose")
    .Output("output", "store", 8)
    .Build();
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
       }
}

TEST_F(OptimizerStV2, SliceConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto s2_sliced = graph.CreateSizeVar(7);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2_sliced};
  *load0.y.strides = {s1 * s2, s2, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  concat_op.x = {load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y,
                 load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y,
                 load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id, z2.id};
  *concat_op.y.repeats = {s0, s1, s2_sliced + s2_sliced};
  *concat_op.y.strides = {s1 * (s2_sliced + s2_sliced), s2_sliced + s2_sliced, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2_sliced + s2_sliced};
  *store_op.y.strides = {s1 * (s2_sliced + s2_sliced), s2_sliced + s2_sliced, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());
  auto concat_node =
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  EXPECT_EQ(ToString(concat_node->outputs[0].attr.vectorized_strides), "[14, 1]");
}

TEST_F(OptimizerStV2, SplitAndFirstDimConcat) {
  using ge::testing::AscGraphBuilder;
  using ge::testing::Sym;
  using ge::ops::One;

  auto s0 = Sym(16);
  auto s1 = Sym(32);
  auto s1_0 = Sym(16);

  auto graph = AscGraphBuilder("slice_concat")
    .Loops({s0, s1})
    .Data("data0", 0)
    .Load("load0", "data0", {s0, s1}, {s1, One})
    .Split("split", "load0", {
      AscGraphBuilder::SplitOutput{ge::DT_FLOAT, {}, {s0, s1_0}, {s1_0, One}},
      AscGraphBuilder::SplitOutput{ge::DT_FLOAT, {}, {s0, s1_0}, {s1_0, One}},
    })
    .Concat("concat", {"split:0", "split:1"}, /* concat_dim */ 0)
    .Store("store", "concat")
    .Output("output", "store", 0)
    .Build();
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  EXPECT_TRUE(impl_graph.FindNode("concat") == nullptr);
  ;
  EXPECT_TRUE(impl_graph.FindNode("split") != nullptr);
  ;
}

TEST_F(OptimizerStV2, TransposeTwoAxisSplitCaseNeedInputAlign) {
  const Expression s0 = ge::Symbol(347);
  const Expression s2 = ge::Symbol(15);
  const Expression s3 = ge::Symbol(49);

  // Load load0 with transpose: axis order {z0, z3, z2}
  std::vector<Expression> load0_strides = {s2 * s3, s2, ge::sym::kSymbolOne};

  // Load load1 with padding on axis 1: shape {s0, 1, s3}, strides {s3, 0, One}
  std::vector<Expression> load1_shape = {s0, ge::sym::kSymbolOne, s3};
  std::vector<Expression> load1_strides = {s3, ge::sym::kSymbolZero, ge::sym::kSymbolOne};

  auto graph = AscGraphBuilder("test")
    .Loops({s0, s2, s3})
    .Data("data0", 0, ge::DT_FLOAT)
    .Load("load0", "data0", {}, load0_strides)  // loads with transpose axis {z0, z3, z2}
    .Transpose("transpose0", "load0", {0, 2, 1})  // transpose axes: {z0, z3, z2} -> {z0, z2, z3}
    .Data("data1", 1, ge::DT_FLOAT)
    .Load("load1", "data1", load1_shape, load1_strides)
    .Broadcast("brc0", "load1", {1})  // broadcast on axis 1
    .Mul("mul0", "brc0", "transpose0")
    .Store("store0", "mul0")
    .Output("output", "store0", 0, ge::DT_FLOAT)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
}

TEST_F(OptimizerStV2, FirstDimSplitAndConcat) {
  using ge::testing::AscGraphBuilder;
  using ge::testing::Sym;
  using ge::ops::One;

  auto s0 = Sym(16);
  auto s1 = Sym(32);
  auto s0_0 = Sym(8);

  auto graph = AscGraphBuilder("slice_concat")
    .Loops({s0, s1})
    .Data("data0", 0)
    .Load("load0", "data0", {s0, s1}, {s1, One})
    .Split("split", "load0", {
      AscGraphBuilder::SplitOutput{ge::DT_FLOAT, {}, {s0_0, s1}, {s1, One}},
      AscGraphBuilder::SplitOutput{ge::DT_FLOAT, {}, {s0_0, s1}, {s1, One}},
    })
    .Concat("concat", {"split:0", "split:1"}, /* concat_dim */ 1)
    .Store("store", "concat")
    .Output("output", "store", 0)
    .Build();
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  EXPECT_TRUE(impl_graph.FindNode("concat") != nullptr);
  ;
  EXPECT_TRUE(impl_graph.FindNode("split") == nullptr);
  ;
}
ge::AscGraph CreatNestingLoadGraph(const std::string &name) {
  auto s0 = Sym("s0");
  auto s1 = Sym("s1");
  return AscGraphBuilder(name)
    .Loops({s0, s1 + s1 + s1 + s1})
    .Data("data0", 0, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load0", "data0", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Data("data1", 1, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load1", "data1", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Add("add0", "load0", "load1")
    .Data("data2", 2, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load2", "data2", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Add("add1", "add0", "load2")
    .Data("data3", 3, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load3", "data3", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Add("add2", "add1", "load3")
    .Data("data4", 4, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load4", "data4", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Add("add3", "add2", "load4")
    .Data("data5", 5, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load5", "data5", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Add("add4", "add3", "load5")
    .Data("data6", 6, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load6", "data6", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Add("add5", "add4", "load6")
    .Add("add6", "add5", "load5")
    .Add("add7", "add6", "load4")
    .Add("add8", "add7", "load3")
    .Add("add9", "add8", "load2")
    .Add("add10", "add9", "load1")
    .Add("add11", "add10", "load0")
    .Store("store", "add11")
    .Output("output", "store", 0, ge::DT_FLOAT16)
    .Build();
}

TEST_F(OptimizerStV2, TestNoNeedToShortenLoadLifeTime) {
  auto graph = CreatNestingLoadGraph("NestingLoadGraph");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  for (const auto &node : graph.GetAllNodes()) {
    EXPECT_NE(node->GetType(), ascir_op::Ub2ub::Type);
  }
}

TEST_F(OptimizerStV2, PowerScalar) {
  auto s0 = Sym("s0");
  auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("PowRemove")
    .Loops({s0, s1})
    .Data("data0", 0)
    .Load("load0", "data0")
    .Scalar("pow_input", "2.00000000000000000000e+00")
    .Broadcast("brc0", "pow_input", {ge::ops::One, s1})
    .Broadcast("brc1", "brc0", {s0, s1})
    .template Op<ascir_op::Pow>("pow", {"load0", "brc1"})
    .Scalar("pow_input1", "1")
    .template Op<ascir_op::Pow>("pow1", {"pow", "pow_input1"})
    .Scalar("pow_input2", "0.00000000000000000")
    .template Op<ascir_op::Pow>("pow2", {"pow1", "pow_input2"})
    .Store("store", "pow2")
    .Output("y", "store", 0, ge::DT_FLOAT16)
    .Build();
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
}

TEST_F(OptimizerStV2, ConcatSingleDim) {
  AscGraph graph("slice_concat");
  auto s1 = graph.CreateSizeVar(2);
  auto s1_0 = graph.CreateSizeVar(1);
  auto s1_1 = graph.CreateSizeVar(1);
  auto stride_1_0 = ge::ops::Zero;
  auto stride_1_1 = ge::ops::Zero;
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z1_0.id};
  load0.x = data0.y;
  *load0.y.axis = {z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s1_0};
  *load0.y.strides = {stride_1_0};

  Load load1("load1");
  load1.attr.sched.axis = {z1_0.id};
  load1.x = data1.y;
  *load1.y.axis = {z1_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s1_1};
  *load1.y.strides = {stride_1_1};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z1.id};
  *concat_op.y.repeats = {s1_0 + s1_1};
  *concat_op.y.strides = {ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s1_0 + s1_1};
  *store_op.y.strides = {ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
}

TEST_F(OptimizerStV2, JustMutmul) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, MutmulAndAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, MutmulAndBroadcastAdd) {
  setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1 * s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1 * s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 * s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, JustMutmulBias) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  MatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, JustMutmulOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_INT8;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_INT8;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  MatMulOffset matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulOffset");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, JustMutmulBaisOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = ge::DT_INT8;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.strides = {ge::ops::Zero, ge::ops::One};
  *data3.y.repeats = {ge::ops::One, s1};
  data3.ir_attr.SetIndex(2);

  Load load3("load3");
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.x = data3.y;
  *load3.y.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_INT8;
  *load3.y.strides = {ge::ops::Zero, ge::ops::One};
  *load3.y.repeats = {ge::ops::One, s1};

  MatMulOffsetBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load3.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 8) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulOffsetBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, JustBatchMutmul) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMul");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, BatchMutmulAndAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMul");
    }
  }
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, BatchMutmulAndBroadcastAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1 * s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1 * s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 * s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMul");
    }
  }
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, JustBatchMutmulBias) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  BatchMatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMulBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, JustBatchMutmulOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  BatchMatMulOffset matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMulOffset");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, JustBatchMutmulBaisOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = ge::DT_INT8;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.strides = {ge::ops::Zero, ge::ops::One};
  *data3.y.repeats = {ge::ops::One, s1};
  data3.ir_attr.SetIndex(2);

  Load load3("load3");
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.x = data3.y;
  *load3.y.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_INT8;
  *load3.y.strides = {ge::ops::Zero, ge::ops::One};
  *load3.y.repeats = {ge::ops::One, s1};

  BatchMatMulOffsetBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load3.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 8) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMulOffsetBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, MutmulAndAbs) {
  ge::AscGraph graph("mutmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Abs add_op("abs");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x = matmul.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(static_cast<int32_t>(fused_scheduled_result.node_idx_to_scheduled_results[0][0].cube_type), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
  EXPECT_EQ(static_cast<int32_t>(fused_scheduled_result.node_idx_to_scheduled_results[0][1].cube_type), 1);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Abs");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndBroadcastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndCastBroadcastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = cast.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, SliceSliceConcatD) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(90);
  auto s2 = graph.CreateSizeVar(1);
  auto s1_0 = graph.CreateSizeVar(60);
  auto s1_1 = graph.CreateSizeVar(30);
  auto s3 = graph.CreateSizeVar(97);
  auto s4 = graph.CreateSizeVar(65);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1_1, One};
  *data0.y.strides = {s1_1, One, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1_0, One};
  *data1.y.strides = {s1_0, One, One};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1_0.id};
  load0.x = data1.y;
  *load0.y.axis = {z0.id, z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1_0};
  *load0.y.strides = {s3 * s1_0, s3};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.x = data0.y;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s4 * s1_1, s4};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1};
  *concat_op.y.strides = {s1, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_remove_pad_0 = impl_graph.FindNode("load0_remove_pad_0");
    EXPECT_EQ(load0_remove_pad_0, nullptr);
    auto load1_remove_pad_0 = impl_graph.FindNode("load1_remove_pad_0");
    EXPECT_EQ(load1_remove_pad_0, nullptr);
  }
}

TEST_F(OptimizerStV2, TailAxisSliceWithMultiAxisSlice) {
  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(7);
  auto s2 = ge::Symbol(15);
  auto s3 = ge::Symbol(72);
  auto graph = AscGraphBuilder("slice_graph")
    .Loops({s0, s1, s2, s3})
    .Data("data0", 0)
    .Load("load0", "data0", {ge::ops::One, s1, s2, ge::ops::One}, {ge::sym::kSymbolZero, ge::Symbol(4941), ge::Symbol(61), ge::sym::kSymbolZero})
    .Broadcast("brc0", "load0", {s0, s1, s2, s3})
    .Data("data1", 1)
    .Load("load1", "data1", {s0, ge::ops::One, s2, s3}, {s2 * s3, ge::sym::kSymbolZero, s3, ge::ops::One})
    .Broadcast("brc1", "load1", {s0, s1, s2, s3})
    .Add("add0", "brc0", "brc1")
    .Store("store0", "add0")
    .Output("out0", "store0", 0)
    .Build();
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 8UL);

  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_remove_pad_0 = impl_graph.FindNode("load0_remove_pad_0");
    EXPECT_EQ(load0_remove_pad_0, nullptr);
    auto load1_remove_pad_0 = impl_graph.FindNode("load1_remove_pad_0");
    EXPECT_EQ(load1_remove_pad_0, nullptr);
  }
}

TEST_F(OptimizerStV2, MatmulAddExpAddAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  Data data3("data3", graph);
  data3.y.dtype = ge::DT_FLOAT16;
  data3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data3.y.axis = {z0.id, z1.id, z2.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.repeats = {One, One, One};
  *data3.y.strides = {Zero, Zero, Zero};
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id, z2.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id, z2.id};
  *load3.y.strides = {s1*s2, s2, ge::ops::One};
  *load3.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT16;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  ascir_op::Exp exp1("exp");
  exp1.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp1.x = add_op.y;
  exp1.y.dtype = ge::DT_FLOAT16;
  *exp1.y.axis = {z0.id, z1.id, z2.id};
  *exp1.y.strides = {s1*s2, s2, ge::ops::One};
  *exp1.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op1("add1");
  add_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op1.x1 = matmul.y;
  add_op1.x2 = load3.y;
  add_op1.y.dtype = ge::DT_FLOAT16;
  *add_op1.y.axis = {z0.id, z1.id, z2.id};
  *add_op1.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op1.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op2("add2");
  add_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op2.x1 = exp1.y;
  add_op2.x2 = add_op1.y;
  add_op2.y.dtype = ge::DT_FLOAT16;
  *add_op2.y.axis = {z0.id, z1.id, z2.id};
  *add_op2.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op2.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op2.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C1");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetName(),
            "matmul_1_ub_Y3_S0G0C2");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetName(),
            "matmul_1_ub_Y2_S0G0C3");
}

TEST_F(OptimizerStV2, MatmulAndCastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = cast.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C1");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetName(),
            "matmul_1_ub_Y3_S0G0C2");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetName(),
            "matmul_1_ub_Y2_S0G0C3");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndCastMultiRefsAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = cast.y;
  add_op.x2 = cast.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C1");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetName(),
            "matmul_1_ub_Y3_S0G0C2");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetName(),
            "matmul_1_ub_Y2_S0G0C3");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndBrcLoadMultiRefsAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = load2.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op1("add1");
  add_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op1.x1 = add_op.y;
  add_op1.x2 = load2.y;
  add_op1.y.dtype = ge::DT_FLOAT;
  *add_op1.y.axis = {z0.id, z1.id, z2.id};
  *add_op1.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op1.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op1.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}


TEST_F(OptimizerStV2, MatmulAndCastBrcMultiRefsAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = cast.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = cast.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ("matmul_0_S0G1C0", fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName());
  EXPECT_EQ("matmul_1_Y0_S0G0C0", fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName());
  EXPECT_EQ("matmul_1_Y1_S0G0C1", fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName());
}

TEST_F(OptimizerStV2, MatmulStoreAddExpAddAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT16;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = matmul.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.strides = {s1, ge::ops::One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op1("output1");
  output_op1.x = store_op1.y;
  output_op1.y.dtype = ge::DT_FLOAT16;
  output_op1.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  ascir_op::Exp exp("exp");
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.x = add_op.y;
  exp.y.dtype = ge::DT_FLOAT16;
  *exp.y.axis = {z0.id, z1.id};
  *exp.y.strides = {s1, ge::ops::One};
  *exp.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = exp.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_S0G0C1");
}

TEST_F(OptimizerStV2, JustMutmulBiasAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(39);
  auto s1 = graph.CreateSizeVar(39);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  MatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = ge::DT_FLOAT;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.strides = {s1, ge::ops::One};
  *data3.y.repeats = {s0, s1};
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.x = data3.y;
  *load3.y.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT;
  *load3.y.strides = {s1, ge::ops::One};
  *load3.y.repeats = {s0, s1};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id};
  add.x1 = load3.y;
  add.x2 = matmul.y;
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {s1, ge::ops::One};
  *add.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, ge::ops::One};
  *store_op1.y.repeats = {s0, s1};
  store_op1.ir_attr.SetOffset(ge::ops::One);

  Output output_op1("output1");
  output_op1.x = store_op1.y;
  output_op1.y.dtype = ge::DT_FLOAT;
  output_op1.ir_attr.SetIndex(1);

  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(OptimizerStV2, MatmulAndBrcLoadScalarAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Scalar data2("scalar", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.ir_attr.SetIndex(2);

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = data2.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc1.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndLoadMultiBrcAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1, One, Zero};
  *load2.y.repeats = {s0, s1, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load2.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc1.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(OptimizerStV2, MatmulAndLoadBrcAndAbsBrcAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1, One, Zero};
  *load2.y.repeats = {s0, s1, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = load2.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1, One, Zero};
  *abs.y.repeats = {s0, s1, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = abs.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = brc1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_Y0_S0G0C0");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Broadcast");
    }
  }
}

TEST_F(OptimizerStV2, ConcatSameShape_MixTQueAndTBuf) {
  auto dtype = ge::DT_INT16;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol(7);
  auto s2 = s1 + s1;
  auto graph = ge::testing::AscGraphBuilder("test_graph")
                   .Loops({s0, s2})
                   .Data("data0", 0, dtype)
                   .Load("load0", "data0", {s0, s1}, {s1, ge::sym::kSymbolOne})
                   .Data("data1", 1, dtype)
                   .Load("load1", "data1", {s0, s1}, {s1, ge::sym::kSymbolOne})
                   .Relu("relu0", "load1")
                   .Concat("concat", {"load0", "relu0"})
                   .Store("store", "concat")
                   .Output("out", "store")
                   .Build();
  auto concat_node = graph.FindNode("concat");
  ::optimize::Optimizer optimizer(::optimize::OptimizerOptions{});
  std::vector<::ascir::ScheduledResult> schedule_results;
  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
}

}  // namespace optimize
}  // namespace