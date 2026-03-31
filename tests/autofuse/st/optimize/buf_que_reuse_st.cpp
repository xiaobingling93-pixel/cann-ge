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
#undef private
#include "ascir_utils.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "platform/v1/platformv1.h"
#include "codegen.h"

#include "tests/autofuse/framework/easy_asc_graph/asc_graph_builder.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using ge::testing::Sym;
using ge::testing::AscGraphBuilder;

namespace optimize {

ge::AscGraph CreatSomeInputFusedConcatGraph(const std::string &name) {
  auto s0 = Sym("s0");
  auto s1 = Sym("s1");
  return AscGraphBuilder(name)
    .Loops({s0, s1 + s1 + s1 + s1})
    .Data("data0", 0, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load0", "data0", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Abs("abs0", "load0")
    .Exp("exp0", "abs0")
    .Data("data1", 1, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load1", "data1", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Data("data2", 2, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load2", "data2", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Abs("abs2", "load2")
    .Exp("exp2", "abs2")
    .Data("data3", 3, {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Load("load3", "data3", {s0, s1}, {s1, ge::sym::kSymbolOne})
    .Concat("concat", {"exp0", "load1", "exp2", "load3"})
    .Store("store", "concat")
    .Output("output", "store", 0, ge::DT_FLOAT16)
    .Build();
}

ge::AscGraph CreatBrcCascadeGraph(const std::string &name) {
  auto s0 = Symbol(128);
  auto s1 = Symbol(10);
  return AscGraphBuilder(name)
    .Loops({s0, s1})
    .Data("data0", 0)
    .Load("load0", "data0")
    .Data("data1", 1)
    .Load("load1", "data1", {s0, ge::sym::kSymbolOne}, {ge::sym::kSymbolOne, ge::sym::kSymbolZero})
    .Broadcast("brc3", "load1", {s0, s1})
    .Op<ascir_op::Gt>("gt", {"load0", "brc3"})
    .template Op<ascir_op::Sigmoid>("sigmoid0", {"gt"})
    .Abs("abs0", "gt")
    .Scalar("scalar1", "")
    .Broadcast("brc4", "scalar1", {ge::sym::kSymbolOne, s1})
    .Broadcast("brc5", "brc4", {s0, s1})
    .Add("add", "abs0", "sigmoid0")
    .Mul("mul", "add", "brc5")
    .Abs("abs1", "mul")
    .Abs("abs2", "abs1")
    .Store("store", "abs2")
    .Output("output", "store", 0)
    .Build();
}

ge::AscGraph CreatBrcReduceGraph(const std::string &name) {
  auto s0 = Symbol(12);
  auto s1 = Symbol(16);
  return AscGraphBuilder(name)
    .Loops({s0, s1})
    .Data("data0", 0)
    .Load("load0", "data0", {s0, ge::sym::kSymbolOne}, {ge::sym::kSymbolOne, ge::sym::kSymbolZero})
    .Exp("exp0", "load0")
    .Broadcast("brc0", "exp0", {s0, s1})
    .Data("data1", 1)
    .Load("load1", "data1")
    .Relu("relu0", "load1")
    .Add("add0", "brc0", "relu0")
    .template Op<ascir_op::Sigmoid>("sigmoid", {"add0"})
    .Max("max0", "sigmoid", {1})
    .template Op<ascir_op::Sigmoid>("Sigmoid1", {"max0"})
    .Store("store", "Sigmoid1")
    .Output("output", "store", 0)
    .Build();
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

class BufQueReuseSt : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  optimize::Optimizer optimizer;

  BufQueReuseSt() : optimizer(optimize::OptimizerOptions{}) {}

  static std::string ExpressToStr(std::vector<ge::Expression> &exprs) {
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

TEST_F(BufQueReuseSt, TestTQueShareInConcatMultiInputsScene) {
  auto graph = CreatSomeInputFusedConcatGraph("Concat4InputsGraph");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 4UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto load0 = impl_graph.FindNode("load0");
  ASSERT_NE(load0, nullptr);
  auto load1 = impl_graph.FindNode("load1");
  ASSERT_NE(load1, nullptr);
  auto load2 = impl_graph.FindNode("load2");
  ASSERT_NE(load2, nullptr);
  auto load3 = impl_graph.FindNode("load3");
  ASSERT_NE(load3, nullptr);

  // used 2vecin
  EXPECT_EQ(load0->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load0->outputs[0].attr.mem.reuse_id, 0);
  EXPECT_EQ(load1->outputs[0].attr.que.id, 0);  // load1 reuse load0
  EXPECT_EQ(load1->outputs[0].attr.mem.reuse_id, 6);
  EXPECT_EQ(load2->outputs[0].attr.que.id, 1);  // load2 use new que
  EXPECT_EQ(load2->outputs[0].attr.mem.reuse_id, 3);
  EXPECT_EQ(load3->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load3->outputs[0].attr.mem.reuse_id, 6);  // load1 share with load0
}

TEST_F(BufQueReuseSt, TestShortenLoadLifeTime) {
  auto graph = CreatNestingLoadGraph("NestingLoadGraph");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto load0 = impl_graph.FindNode("load0");
  ASSERT_NE(load0, nullptr);
  auto load1 = impl_graph.FindNode("load1");
  ASSERT_NE(load1, nullptr);
  auto load2 = impl_graph.FindNode("load2");
  ASSERT_NE(load2, nullptr);
  auto load3 = impl_graph.FindNode("load3");
  ASSERT_NE(load3, nullptr);
  auto load4 = impl_graph.FindNode("load4");
  ASSERT_NE(load4, nullptr);
  auto load5 = impl_graph.FindNode("load5");
  ASSERT_NE(load5, nullptr);
  auto load6 = impl_graph.FindNode("load6");
  ASSERT_NE(load6, nullptr);

  // used 2vecin
  EXPECT_EQ(load0->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load0->outputs[0].attr.mem.reuse_id, 0);
  EXPECT_EQ(load1->outputs[0].attr.que.id, 1);  // load1 reuse load0
  EXPECT_EQ(load1->outputs[0].attr.mem.reuse_id, 2);
  EXPECT_EQ(load2->outputs[0].attr.que.id, 0);  // load2 use new que
  EXPECT_EQ(load2->outputs[0].attr.mem.reuse_id, 4);
  EXPECT_EQ(load3->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load3->outputs[0].attr.mem.reuse_id, 8);  // load1 share with load0
  EXPECT_EQ(load4->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load4->outputs[0].attr.mem.reuse_id, 10);  // load1 share with load0
  EXPECT_EQ(load5->outputs[0].attr.que.id, 2);
  EXPECT_EQ(load5->outputs[0].attr.mem.reuse_id, 12);  // load1 share with load0
  EXPECT_EQ(load6->outputs[0].attr.que.id, 3);
  EXPECT_EQ(load6->outputs[0].attr.mem.reuse_id, 14);  // load1 share with load0
}

TEST_F(BufQueReuseSt, TestVecCanReuseTque) {
  auto graph = CreatBrcReduceGraph("MultiBrcGraph");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto load1 = impl_graph.FindNode("load1");
  auto sigmoid = impl_graph.FindNode("sigmoid");
  auto max = impl_graph.FindNode("max0");
  ASSERT_NE(load1, nullptr);
  ASSERT_NE(sigmoid, nullptr);
  ASSERT_NE(max, nullptr);
  int64_t que_id = load1->outputs[0].attr.que.id;
  EXPECT_EQ(sigmoid->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);
  EXPECT_EQ(sigmoid->outputs[0].attr.que.id, que_id);                               // sigmoid can reuse load1
  EXPECT_EQ(max->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeBuffer);  // different loop cannot reuse que
}

TEST_F(BufQueReuseSt, TestInplaceChainVecCanReuseTque) {
  auto graph = CreatBrcCascadeGraph("MultiBrcGraph");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto load0 = impl_graph.FindNode("load0");
  ASSERT_NE(load0, nullptr);
  auto sigmoid0 = impl_graph.FindNode("sigmoid0");
  ASSERT_NE(sigmoid0, nullptr);
  auto add = impl_graph.FindNode("add");
  ASSERT_NE(add, nullptr);
  auto mul = impl_graph.FindNode("mul");
  ASSERT_NE(mul, nullptr);
  auto abs1 = impl_graph.FindNode("abs1");
  ASSERT_NE(abs1, nullptr);

  int64_t que_id = load0->outputs[0].attr.que.id;
  EXPECT_EQ(sigmoid0->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);
  EXPECT_EQ(sigmoid0->outputs[0].attr.que.id, que_id);
  EXPECT_NE(sigmoid0->outputs[0].attr.mem.reuse_id, ge::kIdNone);

  int64_t vecout_que_id = abs1->outputs[0].attr.que.id;
  EXPECT_EQ(add->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);  // inplace reuse
  EXPECT_EQ(add->outputs[0].attr.que.id, vecout_que_id);
  EXPECT_NE(add->outputs[0].attr.mem.reuse_id, ge::kIdNone);

  EXPECT_EQ(mul->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);  // inplace reuse
  EXPECT_EQ(mul->outputs[0].attr.que.id, vecout_que_id);
  EXPECT_NE(mul->outputs[0].attr.mem.reuse_id, ge::kIdNone);

  EXPECT_EQ(abs1->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);  // inplace reuse
  EXPECT_EQ(abs1->outputs[0].attr.que.id, vecout_que_id);
  EXPECT_NE(abs1->outputs[0].attr.mem.reuse_id, ge::kIdNone);
}

TEST_F(BufQueReuseSt, TestVecoutCanInplaceReuseCalc) {
  auto graph = AscGraphBuilder("InplaceGraph")
    .Loops({Symbol(128)})
    .Data("data0", 0)
    .Load("load0", "data0")
    .Load("load1", "data0")
    .Op<ascir_op::Pow>("pow0", {"load0", "load1"})
    .Load("load2", "data0")
    .Add("add0", "pow0", "load2")
    .Store("store", "add0")
    .Output("output", "store", 0)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_TRUE(!fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto load0_node = impl_graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = impl_graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);

  auto load2_node = impl_graph.FindNode("load2");
  ASSERT_NE(load2_node, nullptr);

  auto pow0_node = impl_graph.FindNode("pow0");
  ASSERT_NE(pow0_node, nullptr);
  auto add0_node = impl_graph.FindNode("add0");
  ASSERT_NE(add0_node, nullptr);

  // tque share
  EXPECT_EQ(load0_node->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);
  EXPECT_EQ(load0_node->outputs[0].attr.que.id, load1_node->outputs[0].attr.que.id);
  EXPECT_EQ(load0_node->outputs[0].attr.mem.reuse_id, load1_node->outputs[0].attr.mem.reuse_id);

  EXPECT_EQ(load2_node->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);
  EXPECT_NE(load0_node->outputs[0].attr.que.id, load2_node->outputs[0].attr.que.id);
  EXPECT_NE(load0_node->outputs[0].attr.mem.reuse_id, load2_node->outputs[0].attr.mem.reuse_id);
  // vecout inplace reuse calc
  EXPECT_EQ(add0_node->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);
  EXPECT_EQ(pow0_node->outputs[0].attr.mem.alloc_type, ge::AllocType::kAllocTypeQueue);
  EXPECT_EQ(add0_node->outputs[0].attr.que.id, pow0_node->outputs[0].attr.que.id);
  EXPECT_NE(add0_node->outputs[0].attr.mem.reuse_id, pow0_node->outputs[0].attr.mem.reuse_id);
}

TEST_F(BufQueReuseSt, TestTmpBuffReuse) {
  auto graph = AscGraphBuilder("tmp_buf_reuse_graph")
    .Loops({Symbol(128)})
    .Data("data0", 0)
    .Load("load0", "data0")
    .Load("load1", "data0")
    .Op<ascir_op::Pow>("pow0", {"load0", "load1"})
    .Abs("abs0", "pow0")
    .Add("add0", "pow0", "abs0")
    .template Op<ascir_op::Sigmoid>("sigmoid", {"add0"})
    .Store("store", "sigmoid")
    .Output("output", "store", 0)
    .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto pow0_node = impl_graph.FindNode("pow0");
  ASSERT_NE(pow0_node, nullptr);
  auto abs0_node = impl_graph.FindNode("abs0");
  ASSERT_NE(abs0_node, nullptr);
  auto sig_node = impl_graph.FindNode("sigmoid");
  ASSERT_NE(sig_node, nullptr);

  EXPECT_EQ(pow0_node->attr.tmp_buffers[0].id, 0);
  EXPECT_EQ(abs0_node->outputs[0].attr.buf.id, -1);
  EXPECT_EQ(pow0_node->attr.tmp_buffers[0].id, sig_node->attr.tmp_buffers[0].id);
}
}  // namespace optimize
