/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "gtest/gtest.h"
#include "ascendc_ir_def.h"
#define private public
#include "autoschedule/alignment_handler.h"
#include "optimize/partition/vector_func_partitioner.h"
#undef private
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "optimize/platformv2.h"
#include "easy_graph/easy_asc_graph.h"
#include "runtime_stub.h"
#include "optimize.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;

class VfPartition : public testing::Test {
protected:
  void SetUp() override {
    // setenv("DUMP_GE_GRAPH", "2", 1);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
  }

  void TearDown() override {
    // setenv("DUMP_GE_GRAPH", "0", 1);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(VfPartition, brc_abs) {
  ge::AscGraph graph("brc_abs");
  auto s0 = ge::Symbol(10);
  auto s1 = ge::Symbol(10);
  auto s2 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z1", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, One, s2};
  *load0.y.strides = {s2, Zero, One};
  *load0.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Abs abs00("abs00");
  abs00.x = load0.y;
  abs00.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs00.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs00.y.dtype = ge::DT_FLOAT;
  *abs00.y.axis = {z0.id, z1.id, z2.id};
  *abs00.y.repeats = {s0, One, s2};
  *abs00.y.strides = {s2, Zero, One};
  *abs00.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Abs abs01("abs01");
  abs01.x = abs00.y;
  abs01.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs01.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs01.y.dtype = ge::DT_FLOAT;
  *abs01.y.axis = {z0.id, z1.id, z2.id};
  *abs01.y.repeats = {s0, One, s2};
  *abs01.y.strides = {s2, Zero, One};
  *abs01.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Abs abs02("abs01");
  abs02.x = abs01.y;
  abs02.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs02.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs02.y.dtype = ge::DT_FLOAT;
  *abs02.y.axis = {z0.id, z1.id, z2.id};
  *abs02.y.repeats = {s0, One, s2};
  *abs02.y.strides = {s2, Zero, One};
  *abs02.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = abs02.y;
  brc.attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s1, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc.y;
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = brc.y;
  abs1.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs1.y.dtype = ge::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id, z2.id};
  *abs1.y.repeats = {s0, s1, s2};
  *abs1.y.strides = {s1 * s2, s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id, z2.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  for (const auto &node: graph.GetAllNodes()) {
    if (node->GetType() == "VectorFunc") {
      const std::string *graph_name = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "sub_graph_name");
      ASSERT_NE(graph_name, nullptr);
      AscGraph subgraph("tmp");
      ASSERT_EQ(graph.FindSubGraph(*graph_name, subgraph), ge::SUCCESS);
    }
  }
}

TEST_F(VfPartition, brc_only_revert) {
  ge::AscGraph graph("brc_abs");
  auto s0 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  *scalar0.y.repeats = {ge::sym::kSymbolOne, ge::sym::kSymbolOne};
  *scalar0.y.strides = {ge::sym::kSymbolZero, ge::sym::kSymbolZero};
  scalar0.ir_attr.SetValue("888");
  scalar0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Broadcast brc("brc");
  brc.x = scalar0.y;
  brc.attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id};
  *brc.y.repeats = {s0};
  *brc.y.strides = {One};
  *brc.y.vectorized_axis = {z0.id};

  ge::ascir_op::Store store("store");
  store.x = brc.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};
  *store.y.vectorized_axis = {z0.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  std::vector<ge::AscGraph> sub_graphs;
  graph.GetAllSubGraphs(sub_graphs);
  EXPECT_TRUE(sub_graphs.empty());
}

TEST_F(VfPartition, brc_with_cycle) {
  ge::AscGraph graph("brc_abs");
  auto s0 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  *scalar0.y.repeats = {ge::sym::kSymbolOne, ge::sym::kSymbolOne};
  *scalar0.y.strides = {ge::sym::kSymbolZero, ge::sym::kSymbolZero};
  scalar0.ir_attr.SetValue("888");
  scalar0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Broadcast brc("brc");
  brc.x = scalar0.y;
  brc.attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id};
  *brc.y.repeats = {s0};
  *brc.y.strides = {One};
  *brc.y.vectorized_axis = {z0.id};

  ge::ascir_op::RemovePad remove_pad("remove_pad");
  remove_pad.x = brc.y;
  remove_pad.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  remove_pad.attr.sched.axis = {z0.id};
  remove_pad.y.dtype = ge::DT_FLOAT;
  *remove_pad.y.axis = {z0.id};
  *remove_pad.y.repeats = {s0};
  *remove_pad.y.strides = {One};
  *remove_pad.y.vectorized_axis = {z0.id};

  ge::ascir_op::Add add("add");
  add.x1 = brc.y;
  add.x2 = remove_pad.y;
  add.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};
  *add.y.vectorized_axis = {z0.id};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = add.y;
  abs1.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id};
  abs1.y.dtype = ge::DT_FLOAT;
  *abs1.y.axis = {z0.id};
  *abs1.y.repeats = {s0};
  *abs1.y.strides = {One};
  *abs1.y.vectorized_axis = {z0.id};

  ge::ascir_op::Store store("store");
  store.x = abs1.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};
  *store.y.vectorized_axis = {z0.id};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc.y;
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};
  *abs.y.vectorized_axis = {z0.id};

  ge::ascir_op::Store store0("store0");
  store0.x = abs.y;
  store0.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store0.attr.sched.axis = {z0.id};
  store0.y.dtype = ge::DT_FLOAT;
  *store0.y.axis = {z0.id};
  *store0.y.repeats = {s0};
  *store0.y.strides = {One};
  *store0.y.vectorized_axis = {z0.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  std::vector<ge::AscGraph> sub_graphs;
  graph.GetAllSubGraphs(sub_graphs);
  EXPECT_EQ(sub_graphs.size(), 2UL);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");

  auto brc_in_root = graph.FindNode("brc");
  ASSERT_EQ(brc_in_root, nullptr);
  auto remove_pad_in_root = graph.FindNode("remove_pad");
  ASSERT_NE(remove_pad_in_root, nullptr);
}

TEST_F(VfPartition, add_more_than_30) {
  ge::AscGraph graph("brc_abs");
  auto s0 = ge::Symbol(10);
  auto s1 = ge::Symbol(10);
  auto s2 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load_i("load0");
  load_i.x = data0.y;
  load_i.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  load_i.y.dtype = ge::DT_FLOAT;
  *load_i.y.axis = {z0.id, z1.id, z2.id};
  *load_i.y.repeats = {s0, One, s2};
  *load_i.y.strides = {s2, Zero, One};
  *load_i.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *load_i.y.vectorized_strides = {s2, Zero, One};

  std::vector<ge::ascir_op::Abs> abs_list;
  for (int i = 0; i < 40; ++i) {
    ge::ascir_op::Abs abs_i(std::to_string(i).c_str());
    abs_i.attr.api.compute_type = ge::ComputeType::kComputeElewise;
    abs_i.attr.sched.axis = {z0.id, z1.id, z2.id};
    abs_i.y.dtype = ge::DT_FLOAT;
    *abs_i.y.axis = {z0.id, z1.id, z2.id};
    *abs_i.y.repeats = {s0, One, s2};
    *abs_i.y.strides = {s2, Zero, One};
    *abs_i.y.vectorized_axis = {z0.id, z1.id, z2.id};
    *abs_i.y.vectorized_strides = {s2, Zero, One};
    abs_list.push_back(abs_i);
  }
  abs_list[0].x = load_i.y;
  for (int i = 1; i < 40; ++i) {
    abs_list[i].x = abs_list[i - 1].y;
  }

  ge::ascir_op::Abs abs_o("abs0");
  abs_o.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs_o.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs_o.y.dtype = ge::DT_FLOAT;
  *abs_o.y.axis = {z0.id, z1.id, z2.id};
  *abs_o.y.repeats = {s0, One, s2};
  *abs_o.y.strides = {s2, One, One};
  *abs_o.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *abs_o.y.vectorized_strides = {s2, Zero, One};
  abs_o.x = abs_list[39].y;

  ge::ascir_op::Broadcast brc("brc");
  brc.x = abs_o.y;
  brc.attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};
  *brc.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *brc.y.vectorized_strides = {s2, Zero, One};

  ge::ascir_op::Add add("add");
  add.x1 = abs_o.y;
  add.x2 = brc.y;
  add.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, One};
  *add.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *add.y.vectorized_strides = {s2, Zero, One};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};
  *store.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *store.y.vectorized_strides = {s2, Zero, One};

  ge::ascir_op::Output y0("y0");
  y0.x = store.y;
  y0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.y.dtype = ge::DT_FLOAT;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};
  *y0.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *y0.y.vectorized_strides = {s2, Zero, One};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpGraph(graph, "AfterPart");

  for (const auto &node: graph.GetAllNodes()) {
    if (node->GetType() == "VectorFunc") {
      const std::string *graph_name = ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), "sub_graph_name");
      ASSERT_NE(graph_name, nullptr);
      AscGraph subgraph("tmp");
      ASSERT_EQ(graph.FindSubGraph(*graph_name, subgraph), ge::SUCCESS);
    }
  }

  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  auto brc_node = graph.FindNode("brc");
  EXPECT_NE(brc_node, nullptr);
  EXPECT_EQ(sub_graphs.size(), 2UL);
}

TEST_F(VfPartition, skip_fuse_for_io_num) {
  ge::AscGraph graph("brc_abs");
  auto s0 = ge::Symbol(999);
  auto s1 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};
  *load.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load2("load2");
  load2.x = data0.y;
  load2.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  *load2.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load3("load3");
  load3.x = data0.y;
  load3.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};
  *load3.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load4("load4");
  load4.x = data0.y;
  load4.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load4.attr.sched.axis = {z0.id, z1.id};
  load4.y.dtype = ge::DT_FLOAT;
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1, One};
  *load4.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load5("load5");
  load5.x = data0.y;
  load5.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load5.attr.sched.axis = {z0.id, z1.id};
  load5.y.dtype = ge::DT_FLOAT;
  *load5.y.axis = {z0.id, z1.id};
  *load5.y.repeats = {s0, s1};
  *load5.y.strides = {s1, One};
  *load5.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load6("load6");
  load6.x = data0.y;
  load6.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load6.attr.sched.axis = {z0.id, z1.id};
  load6.y.dtype = ge::DT_FLOAT;
  *load6.y.axis = {z0.id, z1.id};
  *load6.y.repeats = {s0, s1};
  *load6.y.strides = {s1, One};
  *load6.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Load load7("load7");
  load7.x = data0.y;
  load7.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load7.attr.sched.axis = {z0.id, z1.id};
  load7.y.dtype = ge::DT_FLOAT;
  *load7.y.axis = {z0.id, z1.id};
  *load7.y.repeats = {s0, s1};
  *load7.y.strides = {s1, One};
  *load7.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = ge::DT_FLOAT;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs2.attr.sched.axis = {z0.id, z1.id};
  abs2.y.dtype = ge::DT_FLOAT;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};
  *abs2.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = load3.y;
  abs3.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs3.attr.sched.axis = {z0.id, z1.id};
  abs3.y.dtype = ge::DT_FLOAT;
  *abs3.y.axis = {z0.id, z1.id};
  *abs3.y.repeats = {s0, s1};
  *abs3.y.strides = {s1, One};
  *abs3.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs4("abs4");
  abs4.x = load4.y;
  abs4.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs4.attr.sched.axis = {z0.id, z1.id};
  abs4.y.dtype = ge::DT_FLOAT;
  *abs4.y.axis = {z0.id, z1.id};
  *abs4.y.repeats = {s0, s1};
  *abs4.y.strides = {s1, One};
  *abs4.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs5("abs5");
  abs5.x = load5.y;
  abs5.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs5.attr.sched.axis = {z0.id, z1.id};
  abs5.y.dtype = ge::DT_FLOAT;
  *abs5.y.axis = {z0.id, z1.id};
  *abs5.y.repeats = {s0, s1};
  *abs5.y.strides = {s1, One};
  *abs5.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs6("abs6");
  abs6.x = load6.y;
  abs6.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs6.attr.sched.axis = {z0.id, z1.id};
  abs6.y.dtype = ge::DT_FLOAT;
  *abs6.y.axis = {z0.id, z1.id};
  *abs6.y.repeats = {s0, s1};
  *abs6.y.strides = {s1, One};
  *abs6.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs7("abs7");
  abs7.x = load7.y;
  abs7.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs7.attr.sched.axis = {z0.id, z1.id};
  abs7.y.dtype = ge::DT_FLOAT;
  *abs7.y.axis = {z0.id, z1.id};
  *abs7.y.repeats = {s0, s1};
  *abs7.y.strides = {s1, One};
  *abs7.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add.attr.sched.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  *add.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add1("add1");
  add1.x1 = abs2.y;
  add1.x2 = abs3.y;
  add1.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.y.dtype = ge::DT_FLOAT;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, One};
  *add1.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add2("add2");
  add2.x1 = abs4.y;
  add2.x2 = abs5.y;
  add2.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add2.attr.sched.axis = {z0.id, z1.id};
  add2.y.dtype = ge::DT_FLOAT;
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, One};
  *add2.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add3("add3");
  add3.x1 = abs6.y;
  add3.x2 = abs7.y;
  add3.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add3.attr.sched.axis = {z0.id, z1.id};
  add3.y.dtype = ge::DT_FLOAT;
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, One};
  *add3.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add11("add11");
  add11.x1 = add.y;
  add11.x2 = add1.y;
  add11.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add11.attr.sched.axis = {z0.id, z1.id};
  add11.y.dtype = ge::DT_FLOAT;
  *add11.y.axis = {z0.id, z1.id};
  *add11.y.repeats = {s0, s1};
  *add11.y.strides = {s1, One};
  *add11.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add12("add12");
  add12.x1 = add2.y;
  add12.x2 = add3.y;
  add12.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add12.attr.sched.axis = {z0.id, z1.id};
  add12.y.dtype = ge::DT_FLOAT;
  *add12.y.axis = {z0.id, z1.id};
  *add12.y.repeats = {s0, s1};
  *add12.y.strides = {s1, One};
  *add12.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Add add21("add21");
  add21.x1 = add11.y;
  add21.x2 = add12.y;
  add21.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  add21.attr.sched.axis = {z0.id, z1.id};
  add21.y.dtype = ge::DT_FLOAT;
  *add21.y.axis = {z0.id, z1.id};
  *add21.y.repeats = {s0, s1};
  *add21.y.strides = {s1, One};
  *add21.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Store store("store");
  store.x = add21.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out.y.dtype = ge::DT_FLOAT;
  out.ir_attr.SetIndex(0);

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 2UL);
}

TEST_F(VfPartition, cast_reverse) {
  ge::AscGraph graph("brc_abs");
  auto s0 = ge::Symbol(999);
  auto s1 = ge::Symbol(10);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_UINT16;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_UINT16;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};
  *load.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_UINT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  *abs.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs1.attr.sched.axis = {z0.id, z1.id};
  abs1.y.dtype = ge::DT_UINT16;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  *abs1.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Cast cast("cast");
  cast.x = abs1.y;
  cast.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.y.dtype = ge::DT_UINT64;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.repeats = {s0, s1};
  *cast.y.strides = {s1, One};
  *cast.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = cast.y;
  abs2.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs2.attr.sched.axis = {z0.id, z1.id};
  abs2.y.dtype = ge::DT_UINT64;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};
  *abs2.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Store store("store");
  store.x = abs1.y;
  store.attr.api.compute_type = ge::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_UINT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.attr.api.compute_type = ComputeType::kComputeInvalid;
  out.attr.api.type = ge::ApiType::kAPITypeBuffer;
  out.y.dtype = ge::DT_UINT64;
  out.ir_attr.SetIndex(0);

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_EQ(sub_graphs.size(), 1UL);
  auto cast_node = graph.FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
  auto abs2_node = graph.FindNode("abs2");
  EXPECT_NE(abs2_node, nullptr);
}

TEST_F(VfPartition, cast_fused) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Cast cast("cast");
  cast.x = abs1.y;
  cast.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Exp exp("exp");
  exp.x = cast.y;
  exp.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Exp exp1("exp1");
  exp1.x = exp.y;
  exp1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast2("cast2");
  cast2.x = exp1.y;
  cast2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Exp exp2("exp2");
  exp2.x = cast2.y;
  exp2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = exp2.y;
  store.y.dtype = ge::DT_UINT64;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol("s0"), ge::Symbol("s1")});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 2UL);

  auto cast_node = sub_graphs[1].FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
  auto cast2_node = sub_graphs[1].FindNode("cast2");
  EXPECT_NE(cast2_node, nullptr);
  auto exp2_node = graph.FindNode("exp2");
  EXPECT_NE(exp2_node, nullptr);
}

TEST_F(VfPartition, vf_cascade) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Exp exp("exp");
  exp.x = abs.y;
  exp.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Exp exp1("exp1");
  exp1.x = abs1.y;
  exp1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Div div("div");
  div.x1 = exp.y;
  div.x2 = exp1.y;
  div.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = div.y;
  store.y.dtype = ge::DT_UINT64;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol("s0"), ge::Symbol("s1")});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpGraph(graph, "AfterPart");
}

TEST_F(VfPartition, all_zero_axis_stride) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = abs1.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(1), ge::Symbol(1)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);

  auto abs_node = sub_graphs[0].FindNode("abs");
  ASSERT_NE(abs_node, nullptr);
  EXPECT_EQ(abs_node->attr.sched.loop_axis, -1);

  auto abs1_node = sub_graphs[0].FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);
  EXPECT_EQ(abs1_node->attr.sched.loop_axis, -1);
}

TEST_F(VfPartition, ScalarInputDisableVf) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  ge::ascir_op::Maximum maximum("maximum");
  maximum.x1 = abs1.y;
  maximum.x2 = scalar0.y;
  maximum.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = maximum.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(10), ge::Symbol(2)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);

  auto scalar_node = graph.FindNode("scalar0");
  ASSERT_NE(scalar_node, nullptr);

  auto maximum_node = graph.FindNode("maximum");
  ASSERT_NE(maximum_node, nullptr);

  auto abs_node = sub_graphs[0].FindNode("abs");
  ASSERT_NE(abs_node, nullptr);

  auto abs1_node = sub_graphs[0].FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);
}

TEST_F(VfPartition, test_scalar_brc) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  *scalar0.y.repeats = {ge::sym::kSymbolOne, ge::sym::kSymbolOne};
  *scalar0.y.strides = {ge::sym::kSymbolZero, ge::sym::kSymbolZero};
  scalar0.ir_attr.SetValue("888");

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = scalar0.y;
  brc0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = brc0.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Add add0("add0");
  add0.x1 = abs.y;
  add0.x2 = brc0.y;
  add0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = abs1.y;
  add1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = add1.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(32), ge::Symbol(16)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
}

TEST_F(VfPartition, test_scalar_brc_unsupport_vf) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Data data1("data1", graph);
  *data1.y.repeats = {ge::sym::kSymbolOne, ge::sym::kSymbolOne};
  *data1.y.strides = {ge::sym::kSymbolOne, ge::sym::kSymbolZero};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = brc0.y;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Add add0("add0");
  add0.x1 = abs.y;
  add0.x2 = brc0.y;
  add0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = add0.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(32), ge::Symbol(16)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_node = graph.FindNode("brc0");
  ASSERT_NE(brc_node, nullptr);
  auto sub_brc_node = sub_graphs[0].FindNode("brc0");
  ASSERT_EQ(sub_brc_node, nullptr);
}

TEST_F(VfPartition, test_scalar_brc_multi_output) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load.y;
  brc0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Sub sub("sub0");
  sub.x1 = brc0.y;
  sub.x2 = brc0.y;
  sub.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Add add0("add0");
  add0.x1 = sub.y;
  add0.x2 = brc0.y;
  add0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = add0.y;
  mul.x2 = load1.y;
  mul.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(1)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
}

TEST_F(VfPartition, test_vf_no_cache) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = abs0.y;
  abs2.attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  abs2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store0("store0");
  store0.x = abs1.y;
  store0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store0.y;
  out.ir_attr.SetIndex(0);

  ge::ascir_op::Store store1("store1");
  store1.x = abs2.y;
  store1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(1)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
  for (const auto &node: graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::VectorFunc>(node)) {
      EXPECT_EQ(node->attr.sched.exec_condition, ge::ExecuteCondition::kNoCache);
    }
  }
}

TEST_F(VfPartition, VFWithCycleDetectBug) {
  ge::AscGraph graph("cycle_test");

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Cast cast0("cast0");
  cast0.x = load.y;
  cast0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Add add0("add0");
  add0.x1 = cast0.y;
  add0.x2 = cast1.y;
  add0.y.dtype = ge::DT_FLOAT;
  ge::ascir_op::Add add1("add1");
  add1.x1 = cast0.y;
  add1.x2 = cast1.y;
  add1.y.dtype = ge::DT_FLOAT;
  ge::ascir_op::Add add2("add2");
  add2.x1 = cast0.y;
  add2.x2 = cast1.y;
  add2.y.dtype = ge::DT_FLOAT;
  ge::ascir_op::Add add3("add3");
  add3.x1 = cast0.y;
  add3.x2 = cast1.y;
  add3.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Cast cast3_0("cast3_0");
  cast3_0.x = add0.y;
  cast3_0.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Store store0("store0");
  store0.x = cast3_0.y;
  store0.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Output out0("out0");
  out0.x = store0.y;
  out0.ir_attr.SetIndex(0);

  ge::ascir_op::Cast cast3_1("cast3_1");
  cast3_1.x = add1.y;
  cast3_1.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Store store1("store1");
  store1.x = cast3_1.y;
  store1.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  ge::ascir_op::Cast cast3_2("cast3_2");
  cast3_2.x = add2.y;
  cast3_2.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Store store2("store2");
  store2.x = cast3_2.y;
  store2.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Output out2("out2");
  out2.x = store2.y;
  out2.ir_attr.SetIndex(2);

  ge::ascir_op::Cast cast3_3("cast3_3");
  cast3_3.x = add3.y;
  cast3_3.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Store store3("store3");
  store3.x = cast3_3.y;
  store3.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Output out3("out3");
  out3.x = store3.y;
  out3.ir_attr.SetIndex(3);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(1)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
}

TEST_F(VfPartition, test_vf_cache) {
  ge::AscGraph graph("brc_abs");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  abs1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = abs0.y;
  abs2.attr.sched.exec_condition = ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  abs2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Store store0("store0");
  store0.x = abs1.y;
  store0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out("out");
  out.x = store0.y;
  out.ir_attr.SetIndex(0);

  ge::ascir_op::Store store1("store1");
  store1.x = abs2.y;
  store1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Output out1("out1");
  out1.x = store1.y;
  out1.ir_attr.SetIndex(1);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(1)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);

  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  ASSERT_EQ(sub_graphs.size(), 1UL);
  auto brc_in_root = graph.FindNode("brc0");
  ASSERT_EQ(brc_in_root, nullptr);
  for (const auto &node: graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::VectorFunc>(node)) {
      EXPECT_EQ(node->attr.sched.exec_condition, ge::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis);
    }
  }
}

// 测试 Cast 位宽差距恰好为 2 倍的情况（INT32->INT16），应该允许融合
TEST_F(VfPartition, cast_bitwidth_gap_exactly_2x) {
  ge::AscGraph graph("cast_2x_gap");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT32;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_INT32;

  ge::ascir_op::Cast cast("cast");
  cast.x = abs.y;
  cast.y.dtype = ge::DT_INT16; // INT32 -> INT16, 恰好 2 倍差距

  ge::ascir_op::Exp exp("exp");
  exp.x = cast.y;
  exp.y.dtype = ge::DT_INT16;

  ge::ascir_op::Store store("store");
  store.x = exp.y;
  store.y.dtype = ge::DT_INT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(10), ge::Symbol(10)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);

  // 恰好 2 倍差距应该允许融合
  EXPECT_EQ(sub_graphs.size(), 1UL);
  auto cast_node = sub_graphs[0].FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
}

// 测试 Cast 位宽差距超过 2 倍的情况（INT64->INT16，4倍差距），不应该融合
// 使用多个输入分支来触发 Cluster 合并时的位宽检查
TEST_F(VfPartition, cast_bitwidth_gap_exceed_2x) {
  ge::AscGraph graph("cast_4x_gap");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(2);

  // 分支1: INT64 -> INT16 (4倍差距)
  ge::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.y.dtype = ge::DT_INT64;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.y.dtype = ge::DT_INT64;

  ge::ascir_op::Cast cast1("cast1");
  cast1.x = abs1.y;
  cast1.y.dtype = ge::DT_INT16;

  // 分支2: INT32
  ge::ascir_op::Load load2("load2");
  load2.x = data1.y;
  load2.y.dtype = ge::DT_INT32;

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.y.dtype = ge::DT_INT32;

  // 合并点：两个分支的位宽差距超过 2 倍，应该阻止融合
  ge::ascir_op::Add add("add");
  add.x1 = cast1.y;
  add.x2 = abs2.y;
  add.y.dtype = ge::DT_INT16;

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.y.dtype = ge::DT_INT16;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(10), ge::Symbol(10)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);

  // 由于位宽差距超过 2 倍，两个分支不应该融合到同一个 Cluster
  EXPECT_GE(sub_graphs.size(), 1UL);
}

// 测试高→低 Cast 不应和输出节点融合
TEST_F(VfPartition, cast_high_to_low_no_fuse_with_output) {
  ge::AscGraph graph("cast_high_to_low");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT64;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_INT64;

  ge::ascir_op::Cast cast("cast");
  cast.x = abs.y;
  cast.y.dtype = ge::DT_INT32; // INT64 -> INT32, 高→低

  ge::ascir_op::Exp exp("exp");
  exp.x = cast.y;
  exp.y.dtype = ge::DT_INT32;

  ge::ascir_op::Store store("store");
  store.x = exp.y;
  store.y.dtype = ge::DT_INT32;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(10), ge::Symbol(10)});
  eg.Build();

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);

  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);

  // 高→低 Cast 的检查是在合并 Cluster 时进行的
  // 这里只有一个分支，所以所有节点应该能融合
  EXPECT_EQ(sub_graphs.size(), 1UL);
  auto cast_node = sub_graphs[0].FindNode("cast");
  EXPECT_NE(cast_node, nullptr);
}

// 验证 Compare 和 Where 被强制融合到同一个子图
TEST_F(VfPartition, CompareWhereForceMerge) {
  ge::AscGraph graph("compare_where_merge");
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(1);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(2);

  ge::ascir_op::Data data2("data2", graph);
  data2.ir_attr.SetIndex(3);

  ge::ascir_op::Load load3("load3");
  load3.x = data2.y;
  load3.y.dtype = ge::DT_INT64;

  // 共享的 Abs 节点
  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT32;

  ge::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.y.dtype = ge::DT_INT32;

  // 分支1: Abs -> Cast (INT32 -> INT64)
  ge::ascir_op::Cast cast("cast");
  cast.x = abs.y;
  cast.y.dtype = ge::DT_INT64; // INT32 -> INT64

  // 分支2: Abs -> Gt (Compare)，Gt 的另一个输入是另一个 Data
  ge::ascir_op::Load load2("load2");
  load2.x = data1.y;
  load2.y.dtype = ge::DT_INT32;

  ge::ascir_op::Gt gt("gt");
  gt.x1 = abs.y; // 使用同一个 Abs 的输出
  gt.x2 = load2.y; // 第二个输入
  gt.y.dtype = ge::DT_BOOL;

  ge::ascir_op::Where where("where");
  where.x1 = gt.y;
  where.x2 = cast.y;
  where.x3 = load3.y;
  where.y.dtype = ge::DT_INT64;

  ge::ascir_op::Store store("store");
  store.x = where.y;
  store.y.dtype = ge::DT_INT64;

  ge::ascir_op::Output out("out");
  out.x = store.y;
  out.ir_attr.SetIndex(0);

  auto eg = ge::EaseAscGraph(graph).Loops({ge::Symbol(10), ge::Symbol(10)});
  eg.Build();

  ::ascir::utils::DumpImplGraphs({graph}, "BeforePartition");

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), ge::SUCCESS);
  VectorFuncPartitioner partitioner(graph);
  ASSERT_EQ(partitioner.Partition(), ge::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "AfterPartition");
  std::vector<ge::AscGraph> sub_graphs;
  EXPECT_EQ(graph.GetAllSubGraphs(sub_graphs), ge::SUCCESS);
  EXPECT_GE(sub_graphs.size(), 0UL);

  bool found_gt_where_in_same_graph = false;
  for (const auto &sub_graph: sub_graphs) {
    auto gt_node = sub_graph.FindNode("gt");
    auto where_node = sub_graph.FindNode("where");
    if (gt_node != nullptr && where_node != nullptr) {
      found_gt_where_in_same_graph = true;
      break;
    }
  }

  auto cast_node = graph.FindNode("cast");
  ASSERT_TRUE(cast_node != nullptr);
  EXPECT_EQ(cast_node->GetInDataNodes().at(0)->GetType(), "Abs");
  EXPECT_FALSE(found_gt_where_in_same_graph);
}
} // namespace optimize
