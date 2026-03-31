/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ascendc_ir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "runtime_stub.h"
#include "graph_utils_ex.h"

#define private public
#include "buffer_allocate/buf_que_allocator.h"
#include "asc_graph_builder.h"
#include "ascgraph_info_complete.h"
#undef private
#include "ascir_ops_utils.h"
#include "autoschedule/tiling_group.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace optimize;

namespace optimize {
class BufQueAllocatorUT : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace optimize

TEST_F(BufQueAllocatorUT, test_reuse_id_vecacc) {
  ge::AscGraph graph("test_reuse_id_vecacc");
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.attr.api.unit = ComputeUnit::kUnitMTE2;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = abs1.y;
  abs2.attr.api.compute_type = ComputeType::kComputeElewise;
  abs2.attr.api.unit = ComputeUnit::kUnitVector;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = abs2.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.unit = ComputeUnit::kUnitVector;
  *abs3.y.axis = {z0.id, z1.id};
  *abs3.y.repeats = {s0, s1};
  *abs3.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = abs3.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.unit = ComputeUnit::kUnitMTE2;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);

  auto load_result = graph.FindNode(load.GetName().c_str());  // vec in
  EXPECT_EQ(load_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load_result->outputs[0].attr.mem.reuse_id, 0);
  auto abs0_result = graph.FindNode(abs0.GetName().c_str());  // vecout
  EXPECT_EQ(abs0_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs0_result->outputs[0].attr.mem.reuse_id, 1);
  auto abs1_result = graph.FindNode(abs1.GetName().c_str());  // vecout
  EXPECT_EQ(abs1_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs1_result->outputs[0].attr.mem.reuse_id, 2);
  auto abs2_result = graph.FindNode(abs2.GetName().c_str());  // vecout
  EXPECT_EQ(abs2_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs2_result->outputs[0].attr.mem.reuse_id, 3);
  auto abs3_result = graph.FindNode(abs3.GetName().c_str());  // vecout
  EXPECT_EQ(abs3_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs3_result->outputs[0].attr.mem.reuse_id, 4);
}

TEST_F(BufQueAllocatorUT, test_reuse_id_no_reuse_input) {
  ge::AscGraph graph("LoadAbsStore");
  ge::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = abs1.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);

  auto load_result = graph.FindNode(load.GetName().c_str());
  EXPECT_EQ(load_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load_result->outputs[0].attr.mem.reuse_id, 0);
  auto abs0_result = graph.FindNode(abs0.GetName().c_str());
  EXPECT_EQ(abs0_result->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(abs0_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs1_result = graph.FindNode(abs1.GetName().c_str());
  EXPECT_EQ(abs1_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs1_result->outputs[0].attr.mem.reuse_id, 2);  // vecout should not reuse vecin
}

TEST_F(BufQueAllocatorUT, test_reuse_id_no_reduce_to_broadcast) {
  ge::AscGraph graph("test_reuse_id_no_reduce_to_broadcast");
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.attr.api.unit = ComputeUnit::kUnitMTE2;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};

  ge::ascir_op::Max reduce0("reduce0");
  reduce0.x = abs0.y;
  reduce0.attr.api.compute_type = ComputeType::kComputeReduce;
  reduce0.attr.api.unit = ComputeUnit::kUnitVector;
  *reduce0.y.axis = {z0.id, z1.id};
  *reduce0.y.repeats = {One, s1};
  *reduce0.y.strides = {One, One};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = reduce0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {One, s1};
  *abs1.y.strides = {One, One};

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = abs1.y;
  abs2.attr.api.compute_type = ComputeType::kComputeElewise;
  abs2.attr.api.unit = ComputeUnit::kUnitVector;
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {One, s1};
  *abs2.y.strides = {One, One};

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = abs2.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.unit = ComputeUnit::kUnitVector;
  *abs3.y.axis = {z0.id, z1.id};
  *abs3.y.repeats = {One, s1};
  *abs3.y.strides = {One, One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = abs3.y;
  brc.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc.attr.api.unit = ComputeUnit::kUnitVector;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {One, s1};
  *brc.y.strides = {One, One};

  ge::ascir_op::Abs abs4("abs4");
  abs4.x = brc.y;
  abs4.attr.api.compute_type = ComputeType::kComputeElewise;
  abs4.attr.api.unit = ComputeUnit::kUnitVector;
  *abs4.y.axis = {z0.id, z1.id};
  *abs4.y.repeats = {One, s1};
  *abs4.y.strides = {One, One};

  ge::ascir_op::Store store("store");
  store.x = abs4.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {One, s1};
  *store.y.strides = {One, One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);

  auto load_result = graph.FindNode(load.GetName().c_str());
  EXPECT_EQ(load_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load_result->outputs[0].attr.mem.reuse_id, 0);
  auto abs0_result = graph.FindNode(abs0.GetName().c_str());
  EXPECT_EQ(abs0_result->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(abs0_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs1_result = graph.FindNode(abs1.GetName().c_str());
  auto abs3_result = graph.FindNode(abs3.GetName().c_str());
  EXPECT_NE(abs1_result->outputs[0].attr.buf.id, abs3_result->outputs[0].attr.buf.id);  // not reuse after reduce
}

TEST_F(BufQueAllocatorUT, test_vecout_reduce_not_reuse_other) {
  ge::AscGraph graph("test_vecout_reduce_not_reuse_other");
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.attr.api.unit = ComputeUnit::kUnitMTE2;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = abs0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.unit = ComputeUnit::kUnitMTE2;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = abs1.y;
  add0.x2 = abs0.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.unit = ComputeUnit::kUnitVector;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  ge::ascir_op::Max reduce0("reduce0");
  reduce0.x = add0.y;
  reduce0.attr.api.compute_type = ComputeType::kComputeReduce;
  reduce0.attr.api.unit = ComputeUnit::kUnitVector;
  *reduce0.y.axis = {z0.id, z1.id};
  *reduce0.y.repeats = {One, s1};
  *reduce0.y.strides = {One, One};

  ge::ascir_op::Store store1("store1");
  store1.x = reduce0.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.unit = ComputeUnit::kUnitMTE2;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {One, s1};
  *store1.y.strides = {One, One};

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);

  auto load_result = graph.FindNode(load.GetName().c_str());  // vec in
  EXPECT_EQ(load_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load_result->outputs[0].attr.mem.reuse_id, 0);
  auto abs0_result = graph.FindNode(abs0.GetName().c_str());  // vec out
  EXPECT_EQ(abs0_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs0_result->outputs[0].attr.mem.reuse_id, 1);
  auto reduce_result = graph.FindNode(reduce0.GetName().c_str());  // vec out
  EXPECT_EQ(reduce_result->outputs[0].attr.que.id, 2);
  EXPECT_EQ(reduce_result->outputs[0].attr.mem.reuse_id, 4);
}

TEST_F(BufQueAllocatorUT, test_vecout_db_reuse) {
  ge::AscGraph graph("LoadAbsStore");
  ge::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Max reduce0("reduce0");
  reduce0.x = abs0.y;
  reduce0.attr.api.compute_type = ComputeType::kComputeReduce;
  reduce0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = reduce0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store1("store1");
  store1.x = abs1.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = reduce0.y;
  abs2.attr.api.compute_type = ComputeType::kComputeElewise;
  abs2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store2("store2");
  store2.x = abs2.y;
  store2.attr.api.compute_type = ComputeType::kComputeStore;
  store2.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = reduce0.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store3("store3");
  store3.x = abs3.y;
  store3.attr.api.compute_type = ComputeType::kComputeStore;
  store3.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Output y2("y2");
  y2.x = store2.y;
  y2.attr.api.compute_type = ComputeType::kComputeInvalid;
  y2.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Output y3("y3");
  y3.x = store3.y;
  y3.attr.api.compute_type = ComputeType::kComputeInvalid;
  y3.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);

  auto labs1_result = graph.FindNode(abs1.GetName().c_str());
  EXPECT_EQ(labs1_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(labs1_result->outputs[0].attr.mem.reuse_id, 3);
  auto abs2_result = graph.FindNode(abs2.GetName().c_str());
  EXPECT_EQ(abs2_result->outputs[0].attr.que.id, 2);
  EXPECT_EQ(abs2_result->outputs[0].attr.mem.reuse_id, 4);
  auto abs3_result = graph.FindNode(abs3.GetName().c_str());
  EXPECT_NE(abs3_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs3_result->outputs[0].attr.mem.reuse_id, 5);
}

TEST_F(BufQueAllocatorUT, test_shorten_load_lifetime) {
  ge::AscGraph graph("shorten_load");
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", Axis::kAxisTypeTileInner, s1, {}, -1);

  ge::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  ge::ascir_op::Data x3("x3", graph);
  x3.attr.api.compute_type = ComputeType::kComputeInvalid;
  x3.attr.api.type = ge::ApiType::kAPITypeBuffer;
  ge::ascir_op::Data x4("x4", graph);
  x4.attr.api.compute_type = ComputeType::kComputeInvalid;
  x4.attr.api.type = ge::ApiType::kAPITypeBuffer;
  ge::ascir_op::Data x5("x5", graph);
  x5.attr.api.compute_type = ComputeType::kComputeInvalid;
  x5.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.api.unit = ComputeUnit::kUnitVector;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  ge::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = load2.y;
  add1.attr.api.compute_type = ComputeType::kComputeElewise;
  add1.attr.api.type = ge::ApiType::kAPITypeCompute;
  add1.attr.api.unit = ComputeUnit::kUnitVector;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, ONE};

  ge::ascir_op::Load load3("load3");
  load3.x = x3.y;
  load3.attr.api.compute_type = ComputeType::kComputeLoad;
  load3.attr.api.type = ge::ApiType::kAPITypeCompute;
  load3.attr.api.unit = ComputeUnit::kUnitMTE2;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ONE};

  ge::ascir_op::Add add2("add2");
  add2.x1 = add1.y;
  add2.x2 = load3.y;
  add2.attr.api.compute_type = ComputeType::kComputeElewise;
  add2.attr.api.type = ge::ApiType::kAPITypeCompute;
  add2.attr.api.unit = ComputeUnit::kUnitVector;
  add2.attr.sched.axis = {z0.id, z1.id};
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, ONE};

  ge::ascir_op::Load load4("load4");
  load4.x = x4.y;
  load4.attr.api.compute_type = ComputeType::kComputeLoad;
  load4.attr.api.type = ge::ApiType::kAPITypeCompute;
  load4.attr.api.unit = ComputeUnit::kUnitMTE2;
  load4.attr.sched.axis = {z0.id, z1.id};
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1, ONE};

  ge::ascir_op::Add add3("add3");
  add3.x1 = add2.y;
  add3.x2 = load4.y;
  add3.attr.api.compute_type = ComputeType::kComputeElewise;
  add3.attr.api.type = ge::ApiType::kAPITypeCompute;
  add3.attr.api.unit = ComputeUnit::kUnitVector;
  add3.attr.sched.axis = {z0.id, z1.id};
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, ONE};

  ge::ascir_op::Load load5("load5");
  load5.x = x5.y;
  load5.attr.api.compute_type = ComputeType::kComputeLoad;
  load5.attr.api.type = ge::ApiType::kAPITypeCompute;
  load5.attr.api.unit = ComputeUnit::kUnitMTE2;
  load5.attr.sched.axis = {z0.id, z1.id};
  *load5.y.axis = {z0.id, z1.id};
  *load5.y.repeats = {s0, s1};
  *load5.y.strides = {s1, ONE};

  ge::ascir_op::Add add4("add4");
  add4.x1 = add3.y;
  add4.x2 = load5.y;
  add4.attr.api.compute_type = ComputeType::kComputeElewise;
  add4.attr.api.type = ge::ApiType::kAPITypeCompute;
  add4.attr.api.unit = ComputeUnit::kUnitVector;
  add4.attr.sched.axis = {z0.id, z1.id};
  *add4.y.axis = {z0.id, z1.id};
  *add4.y.repeats = {s0, s1};
  *add4.y.strides = {s1, ONE};

  ge::ascir_op::Add add5("add5");
  add5.x1 = add4.y;
  add5.x2 = load0.y;
  add5.attr.api.compute_type = ComputeType::kComputeElewise;
  add5.attr.api.type = ge::ApiType::kAPITypeCompute;
  add5.attr.api.unit = ComputeUnit::kUnitVector;
  add5.attr.sched.axis = {z0.id, z1.id};
  *add5.y.axis = {z0.id, z1.id};
  *add5.y.repeats = {s0, s1};
  *add5.y.strides = {s1, ONE};

  ge::ascir_op::Add add6("add6");
  add6.x1 = add5.y;
  add6.x2 = load2.y;
  add6.attr.api.compute_type = ComputeType::kComputeElewise;
  add6.attr.api.type = ge::ApiType::kAPITypeCompute;
  add6.attr.api.unit = ComputeUnit::kUnitVector;
  add6.attr.sched.axis = {z0.id, z1.id};
  *add6.y.axis = {z0.id, z1.id};
  *add6.y.repeats = {s0, s1};
  *add6.y.strides = {s1, ONE};

  ge::ascir_op::Add add7("add7");
  add7.x1 = add6.y;
  add7.x2 = load3.y;
  add7.attr.api.compute_type = ComputeType::kComputeElewise;
  add7.attr.api.type = ge::ApiType::kAPITypeCompute;
  add7.attr.api.unit = ComputeUnit::kUnitVector;
  add7.attr.sched.axis = {z0.id, z1.id};
  *add7.y.axis = {z0.id, z1.id};
  *add7.y.repeats = {s0, s1};
  *add7.y.strides = {s1, ONE};

  ge::ascir_op::Add add8("add8");
  add8.x1 = add7.y;
  add8.x2 = load4.y;
  add8.attr.api.compute_type = ComputeType::kComputeElewise;
  add8.attr.api.type = ge::ApiType::kAPITypeCompute;
  add8.attr.api.unit = ComputeUnit::kUnitVector;
  add8.attr.sched.axis = {z0.id, z1.id};
  *add8.y.axis = {z0.id, z1.id};
  *add8.y.repeats = {s0, s1};
  *add8.y.strides = {s1, ONE};

  ge::ascir_op::Add add9("add9");
  add9.x1 = add8.y;
  add9.x2 = load1.y;
  add9.attr.api.compute_type = ComputeType::kComputeElewise;
  add9.attr.api.type = ge::ApiType::kAPITypeCompute;
  add9.attr.api.unit = ComputeUnit::kUnitVector;
  add9.attr.sched.axis = {z0.id, z1.id};
  *add9.y.axis = {z0.id, z1.id};
  *add9.y.repeats = {s0, s1};
  *add9.y.strides = {s1, ONE};

  ge::ascir_op::Store store("store");
  store.x = add9.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.api.unit = ComputeUnit::kUnitVector;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ScheduleUtils::TopologicalSorting(graph);
  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);
  BufQueAllocator().SetOutputTensorAttr(graph);
  EXPECT_EQ(total_vecin_nums, 6UL);
  EXPECT_EQ(BufQueAllocator().ShortenVecinLifetime(graph, 4), ge::SUCCESS);

  ScheduleUtils::TopologicalSorting(graph);
  size_t new_vecin_nums = 0UL;
  size_t new_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, new_vecin_nums, new_vecout_nums);
  BufQueAllocator().SetOutputTensorAttr(graph);
  EXPECT_EQ(new_vecin_nums, 4UL);
}

TEST_F(BufQueAllocatorUT, test_shorten_vecin_lifecycle_with_sorting) {
  auto graph = ge::testing::AscGraphBuilder("shorten_load")
    .Loops({ge::testing::Sym(32), ge::testing::Sym(16)})
    .Data("data0", 0)
    .Load("load00", "data0")
    .Load("load01", "data0")
    .Data("data1", 1)
    .Load("load10", "data1")
    .Load("load11", "data1")
    .Data("data2", 2)
    .Load("load20", "data2")
    .Load("load21", "data2")
    .Data("data3", 3)
    .Load("load30", "data3")
    .Load("load31", "data3")
    .Data("data4", 4)
    .Load("load40", "data4")
    .Load("load41", "data4")
    .Data("data5", 5)
    .Load("load50", "data5")
    .Load("load51", "data5")
    .Data("data6", 6)
    .Load("load60", "data6")
    .Load("load61", "data6")
    .Data("data7", 7)
    .Load("load70", "data7")
    .Load("load71", "data7")
    .Data("data8", 8)
    .Load("load80", "data8")
    .Load("load81", "data8")
    .Data("data9", 9)
    .Load("load90", "data9")
    .Load("load91", "data9")
    .Mul("mul0", "load00", "load01")
    .Mul("mul1", "load01", "load10")
    .Mul("mul2", "load10", "load11")
    .Mul("mul3", "load11", "load20")
    .Mul("mul4", "load20", "load21")
    .Mul("mul5", "load21", "load30")
    .Mul("mul6", "load30", "load31")
    .Mul("mul7", "load31", "load40")
    .Mul("mul8", "load40", "load41")
    .Mul("mul9", "load41", "load50")
    .Mul("mul10", "load50", "load51")
    .Mul("mul11", "load51", "load60")
    .Mul("mul12", "load60", "load61")
    .Mul("mul13", "load61", "load70")
    .Mul("mul14", "load70", "load71")
    .Mul("mul15", "load71", "load80")
    .Mul("mul16", "load80", "load81")
    .Mul("mul17", "load81", "load90")
    .Mul("mul18", "load90", "load91")
    .Concat("cat", {"mul1", "mul2", "mul3", "mul4", "mul5", "mul6", "mul7", "mul8", "mul9",
                    "mul10", "mul11", "mul12", "mul13", "mul14", "mul15", "mul16", "mul17", "mul18"})
    .Store("store", "cat")
    .Output("out", "store", 0)
    .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ScheduleUtils::TopologicalSorting(graph);
  EXPECT_EQ(BufQueAllocator().AllocBufQueForSingleImplGraph(graph, 4), ge::SUCCESS);

  std::set<uint32_t> vecin_ids;
  std::set<uint32_t> vecout_ids;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (auto &tensor : node->outputs()) {
      if (tensor->attr.mem.position == Position::kPositionVecIn) {
        vecin_ids.emplace(tensor->attr.que.id);
      } else if (tensor->attr.mem.position == Position::kPositionVecOut) {
        vecout_ids.emplace(tensor->attr.que.id);
      }
    }
  }
  ASSERT_TRUE(vecin_ids.size() <= 4UL);
  ASSERT_TRUE(vecout_ids.size() <= 4UL);
}

TEST_F(BufQueAllocatorUT, test_shorten_vecout_lifetime) {
  ge::AscGraph graph("shorten_load");
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data0("x0", graph);
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;
  load0.attr.sched.axis = {z0.id};
  *load0.y.axis = {z0.id};
  *load0.y.repeats = {s0};
  *load0.y.strides = {ONE};

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;
  abs0.attr.sched.axis = {z0.id};
  *abs0.y.axis = {z0.id};
  *abs0.y.repeats = {s0};
  *abs0.y.strides = {ONE};

  ge::ascir_op::Store store0("store0");
  store0.x = abs0.y;
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.attr.api.type = ge::ApiType::kAPITypeCompute;
  store0.attr.api.unit = ComputeUnit::kUnitMTE3;
  store0.attr.sched.axis = {z0.id};
  *store0.y.axis = {z0.id};
  *store0.y.repeats = {s0};
  *store0.y.strides = {ONE};

  ge::ascir_op::Output output0("output0");
  output0.x = store0.y;
  output0.attr.api.compute_type = ComputeType::kComputeInvalid;
  output0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;
  load1.attr.sched.axis = {z0.id};
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {ONE};

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;
  abs1.attr.sched.axis = {z0.id};
  *abs1.y.axis = {z0.id};
  *abs1.y.repeats = {s0};
  *abs1.y.strides = {ONE};

  ge::ascir_op::Store store1("store1");
  store1.x = abs1.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = ge::ApiType::kAPITypeCompute;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;
  store1.attr.sched.axis = {z0.id};
  *store1.y.axis = {z0.id};
  *store1.y.repeats = {s0};
  *store1.y.strides = {ONE};

  ge::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.attr.api.compute_type = ComputeType::kComputeInvalid;
  output1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load2("load2");
  load2.x = data0.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;
  load2.attr.sched.axis = {z0.id};
  *load2.y.axis = {z0.id};
  *load2.y.repeats = {s0};
  *load2.y.strides = {ONE};

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.attr.api.compute_type = ComputeType::kComputeElewise;
  abs2.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs2.attr.api.unit = ComputeUnit::kUnitVector;
  abs2.attr.sched.axis = {z0.id};
  *abs2.y.axis = {z0.id};
  *abs2.y.repeats = {s0};
  *abs2.y.strides = {ONE};

  ge::ascir_op::Store store2("store2");
  store2.x = abs2.y;
  store2.attr.api.compute_type = ComputeType::kComputeStore;
  store2.attr.api.type = ge::ApiType::kAPITypeCompute;
  store2.attr.api.unit = ComputeUnit::kUnitMTE3;
  store2.attr.sched.axis = {z0.id};
  *store2.y.axis = {z0.id};
  *store2.y.repeats = {s0};
  *store2.y.strides = {ONE};

  ge::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.attr.api.compute_type = ComputeType::kComputeInvalid;
  output2.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load3("load3");
  load3.x = data0.y;
  load3.attr.api.compute_type = ComputeType::kComputeLoad;
  load3.attr.api.type = ge::ApiType::kAPITypeCompute;
  load3.attr.api.unit = ComputeUnit::kUnitMTE2;
  load3.attr.sched.axis = {z0.id};
  *load3.y.axis = {z0.id};
  *load3.y.repeats = {s0};
  *load3.y.strides = {ONE};

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = load3.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs3.attr.api.unit = ComputeUnit::kUnitVector;
  abs3.attr.sched.axis = {z0.id};
  *abs3.y.axis = {z0.id};
  *abs3.y.repeats = {s0};
  *abs3.y.strides = {ONE};

  ge::ascir_op::Store store3("store3");
  store3.x = abs3.y;
  store3.attr.api.compute_type = ComputeType::kComputeStore;
  store3.attr.api.type = ge::ApiType::kAPITypeCompute;
  store3.attr.api.unit = ComputeUnit::kUnitMTE3;
  store3.attr.sched.axis = {z0.id};
  *store3.y.axis = {z0.id};
  *store3.y.repeats = {s0};
  *store3.y.strides = {ONE};

  ge::ascir_op::Output output3("output3");
  output3.x = store3.y;
  output3.attr.api.compute_type = ComputeType::kComputeInvalid;
  output3.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load4("load4");
  load4.x = data0.y;
  load4.attr.api.compute_type = ComputeType::kComputeLoad;
  load4.attr.api.type = ge::ApiType::kAPITypeCompute;
  load4.attr.api.unit = ComputeUnit::kUnitMTE2;
  load4.attr.sched.axis = {z0.id};
  *load4.y.axis = {z0.id};
  *load4.y.repeats = {s0};
  *load4.y.strides = {ONE};

  ge::ascir_op::Abs abs4("abs4");
  abs4.x = load4.y;
  abs4.attr.api.compute_type = ComputeType::kComputeElewise;
  abs4.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs4.attr.api.unit = ComputeUnit::kUnitVector;
  abs4.attr.sched.axis = {z0.id};
  *abs4.y.axis = {z0.id};
  *abs4.y.repeats = {s0};
  *abs4.y.strides = {ONE};

  ge::ascir_op::Store store4("store4");
  store4.x = abs4.y;
  store4.attr.api.compute_type = ComputeType::kComputeStore;
  store4.attr.api.type = ge::ApiType::kAPITypeCompute;
  store4.attr.api.unit = ComputeUnit::kUnitMTE3;
  store4.attr.sched.axis = {z0.id};
  *store4.y.axis = {z0.id};
  *store4.y.repeats = {s0};
  *store4.y.strides = {ONE};

  ge::ascir_op::Output output4("output4");
  output4.x = store4.y;
  output4.attr.api.compute_type = ComputeType::kComputeInvalid;
  output4.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Concat concat("concat");
  concat.x = {abs0.y, abs1.y, abs2.y, abs3.y, abs4.y};
  concat.attr.api.compute_type = ComputeType::kComputeConcat;
  concat.attr.api.type = ge::ApiType::kAPITypeCompute;
  concat.attr.api.unit = ComputeUnit::kUnitVector;
  concat.attr.sched.axis = {z0.id};
  *concat.y.axis = {z0.id};
  *concat.y.repeats = {s0};
  *concat.y.strides = {ONE};

  ge::ascir_op::Store store("store5");
  store.x = concat.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {ONE};

  ge::ascir_op::Output output5("output5");
  output5.x = store.y;
  output5.attr.api.compute_type = ComputeType::kComputeInvalid;
  output5.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Store store6("store6");
  store6.x = load0.y;
  store6.attr.api.compute_type = ComputeType::kComputeStore;
  store6.attr.api.type = ge::ApiType::kAPITypeCompute;
  store6.attr.api.unit = ComputeUnit::kUnitMTE3;
  store6.attr.sched.axis = {z0.id};
  *store6.y.axis = {z0.id};
  *store6.y.repeats = {s0};
  *store6.y.strides = {ONE};

  ge::ascir_op::Output output6("output6");
  output6.x = store6.y;
  output6.attr.api.compute_type = ComputeType::kComputeInvalid;
  output6.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Store store7("store7");
  store7.x = load1.y;
  store7.attr.api.compute_type = ComputeType::kComputeStore;
  store7.attr.api.type = ge::ApiType::kAPITypeCompute;
  store7.attr.api.unit = ComputeUnit::kUnitMTE3;
  store7.attr.sched.axis = {z0.id};
  *store7.y.axis = {z0.id};
  *store7.y.repeats = {s0};
  *store7.y.strides = {ONE};

  ge::ascir_op::Output output7("output7");
  output7.x = store7.y;
  output7.attr.api.compute_type = ComputeType::kComputeInvalid;
  output7.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Store store8("store8");
  store8.x = load2.y;
  store8.attr.api.compute_type = ComputeType::kComputeStore;
  store8.attr.api.type = ge::ApiType::kAPITypeCompute;
  store8.attr.api.unit = ComputeUnit::kUnitMTE3;
  store8.attr.sched.axis = {z0.id};
  *store8.y.axis = {z0.id};
  *store8.y.repeats = {s0};
  *store8.y.strides = {ONE};

  ge::ascir_op::Output output8("output8");
  output8.x = store8.y;
  output8.attr.api.compute_type = ComputeType::kComputeInvalid;
  output8.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Store store9("store9");
  store9.x = load3.y;
  store9.attr.api.compute_type = ComputeType::kComputeStore;
  store9.attr.api.type = ge::ApiType::kAPITypeCompute;
  store9.attr.api.unit = ComputeUnit::kUnitMTE3;
  store9.attr.sched.axis = {z0.id};
  *store9.y.axis = {z0.id};
  *store9.y.repeats = {s0};
  *store9.y.strides = {ONE};

  ge::ascir_op::Output output9("output9");
  output9.x = store9.y;
  output9.attr.api.compute_type = ComputeType::kComputeInvalid;
  output9.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Store store10("store10");
  store10.x = load4.y;
  store10.attr.api.compute_type = ComputeType::kComputeStore;
  store10.attr.api.type = ge::ApiType::kAPITypeCompute;
  store10.attr.api.unit = ComputeUnit::kUnitMTE3;
  store10.attr.sched.axis = {z0.id};
  *store10.y.axis = {z0.id};
  *store10.y.repeats = {s0};
  *store10.y.strides = {ONE};

  ge::ascir_op::Output output10("output10");
  output10.x = store10.y;
  output10.attr.api.compute_type = ComputeType::kComputeInvalid;
  output10.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ScheduleUtils::TopologicalSorting(graph);
  EXPECT_EQ(BufQueAllocator().AllocBufQueForSingleImplGraph(graph, 4), ge::SUCCESS);

  std::set<uint32_t> vecin_ids;
  std::set<uint32_t> vecout_ids;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (auto &tensor : node->outputs()) {
      if (tensor->attr.mem.position == Position::kPositionVecIn) {
        vecin_ids.emplace(tensor->attr.que.id);
      } else if (tensor->attr.mem.position == Position::kPositionVecOut) {
        vecout_ids.emplace(tensor->attr.que.id);
      }
    }
  }
  EXPECT_LE(vecin_ids.size(), 4UL);
  EXPECT_LE(vecout_ids.size(), 4UL);
}

TEST_F(BufQueAllocatorUT, test_reuse_id_shared_and_db_reuse) {
  ge::AscGraph graph("LoadAbsStore");
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", Axis::kAxisTypeTileInner, s1, {}, -1);

  ge::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.api.unit = ComputeUnit::kUnitVector;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  ge::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = load2.y;
  add1.attr.api.compute_type = ComputeType::kComputeElewise;
  add1.attr.api.type = ge::ApiType::kAPITypeCompute;
  add1.attr.api.unit = ComputeUnit::kUnitVector;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, ONE};

  ge::ascir_op::Data x3("x3", graph);
  x3.attr.api.compute_type = ComputeType::kComputeInvalid;
  x3.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load3("load3");
  load3.x = x3.y;
  load3.attr.api.compute_type = ComputeType::kComputeLoad;
  load3.attr.api.type = ge::ApiType::kAPITypeCompute;
  load3.attr.api.unit = ComputeUnit::kUnitMTE2;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ONE};

  ge::ascir_op::Add add2("add2");
  add2.x1 = add1.y;
  add2.x2 = load3.y;
  add2.attr.api.compute_type = ComputeType::kComputeElewise;
  add2.attr.api.type = ge::ApiType::kAPITypeCompute;
  add2.attr.api.unit = ComputeUnit::kUnitVector;
  add2.attr.sched.axis = {z0.id, z1.id};
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, ONE};

  ge::ascir_op::Store store("store");
  store.x = add1.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);
  // load0 load1 共用, load2 load3 间隔复用, 效果是load2 不复用 load3 复用
  auto load0_result = graph.FindNode(load0.GetName().c_str());
  EXPECT_EQ(load0_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load0_result->outputs[0].attr.mem.reuse_id, 0);
  auto load1_result = graph.FindNode(load1.GetName().c_str());
  EXPECT_EQ(load1_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load1_result->outputs[0].attr.mem.reuse_id, 0);
  auto load2_result = graph.FindNode(load2.GetName().c_str());
  EXPECT_EQ(load2_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load2_result->outputs[0].attr.mem.reuse_id, 2);
  auto load3_result = graph.FindNode(load3.GetName().c_str());
  EXPECT_EQ(load3_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load3_result->outputs[0].attr.mem.reuse_id, 4);
}

TEST_F(BufQueAllocatorUT, test_broadcast_id_mem_unique) {
  ge::AscGraph graph("LoadAbsStore");
  auto ZERO = Symbol(0);
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", Axis::kAxisTypeTileInner, s1, {}, -1);

  ge::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {ZERO, ZERO};
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.api.compute_type = ComputeType::kComputeLoad;
  broadcast1.attr.api.type = ge::ApiType::kAPITypeCompute;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  *broadcast1.y.repeats = {One, s1};
  *broadcast1.y.strides = {ZERO, ONE};
  broadcast1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = broadcast1.y;
  broadcast2.attr.api.compute_type = ComputeType::kComputeLoad;
  broadcast2.attr.api.type = ge::ApiType::kAPITypeCompute;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  *broadcast2.y.repeats = {s0, s1};
  *broadcast2.y.strides = {s1, ONE};
  broadcast2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = broadcast2.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = load2.y;
  add1.attr.api.compute_type = ComputeType::kComputeElewise;
  add1.attr.api.type = ge::ApiType::kAPITypeCompute;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, ONE};
  add1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Data x3("x3", graph);
  x3.attr.api.compute_type = ComputeType::kComputeInvalid;
  x3.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load3("load3");
  load3.x = x3.y;
  load3.attr.api.compute_type = ComputeType::kComputeLoad;
  load3.attr.api.type = ge::ApiType::kAPITypeCompute;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ONE};
  load3.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Add add2("add2");
  add2.x1 = add1.y;
  add2.x2 = load3.y;
  add2.attr.api.compute_type = ComputeType::kComputeElewise;
  add2.attr.api.type = ge::ApiType::kAPITypeCompute;
  add2.attr.sched.axis = {z0.id, z1.id};
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, ONE};
  add2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = add2.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};
  store.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);
  auto load0_result = graph.FindNode(load0.GetName().c_str());
  EXPECT_EQ(load0_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load0_result->outputs[0].attr.mem.reuse_id, 0);
  auto load1_result = graph.FindNode(load1.GetName().c_str());
  EXPECT_EQ(load1_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load1_result->outputs[0].attr.mem.reuse_id, 1);
  auto load2_result = graph.FindNode(load2.GetName().c_str());
  EXPECT_EQ(load2_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load2_result->outputs[0].attr.mem.reuse_id, 5);
  auto load3_result = graph.FindNode(load3.GetName().c_str());
  EXPECT_EQ(load3_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load3_result->outputs[0].attr.mem.reuse_id, 7);
}

TEST_F(BufQueAllocatorUT, test_brc_inline_id_mem_unique) {
  ge::AscGraph graph("test_brc_inline_id_mem_unique");
  auto ZERO = Symbol(0);
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs0.attr.sched.axis = {z0.id, z1.id};
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, ONE};
  *abs0.y.vectorized_axis = {z0.id, z1.id};
  abs0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {ZERO, ONE};
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs1.attr.sched.axis = {z0.id, z1.id};
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {One, s1};
  *abs1.y.strides = {ZERO, ONE};
  *abs1.y.vectorized_axis = {z0.id, z1.id};
  abs1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Add add0("add0");
  add0.x1 = abs0.y;
  add0.x2 = abs1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};
  *add0.y.vectorized_axis = {z0.id, z1.id};
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = add0.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs3.attr.sched.axis = {z0.id, z1.id};
  *abs3.y.axis = {z0.id, z1.id};
  *abs3.y.repeats = {s0, s1};
  *abs3.y.strides = {s1, ONE};
  abs3.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs4("abs4");
  abs4.x = abs3.y;
  abs4.attr.api.compute_type = ComputeType::kComputeElewise;
  abs4.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs4.attr.sched.axis = {z0.id, z1.id};
  *abs4.y.axis = {z0.id, z1.id};
  *abs4.y.repeats = {s0, s1};
  *abs4.y.strides = {s1, ONE};
  abs4.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = abs4.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};
  store.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  auto abs0_node = graph.FindNode("abs0");  // vec
  abs0_node->outputs[0].attr.vectorized_strides = {s1, ONE};
  auto abs1_node = graph.FindNode("abs1");  // vec
  abs1_node->outputs[0].attr.vectorized_strides = {ZERO, ONE};

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);
  auto abs4_result = graph.FindNode("abs4");  // vecout
  auto add0_result = graph.FindNode("add0");  // vecout
  EXPECT_EQ(add0_result->outputs[0].attr.que.id, abs4_result->outputs[0].attr.que.id);
  EXPECT_EQ(add0_result->outputs[0].attr.mem.reuse_id, 4);
  auto abs3_result = graph.FindNode("abs3");  // vecout
  EXPECT_EQ(abs3_result->outputs[0].attr.que.id, abs4_result->outputs[0].attr.que.id);
  EXPECT_EQ(abs3_result->outputs[0].attr.mem.reuse_id, 5);
}

/**
 *                          store
 *                            |
 *                          abs9
 *                           |
 *                          add2
 *                         /   \
 *                       /      \
 *                     /         \
 *                 abs7(half)   load2
 *                    |
 *                  abs3
 *                   |
 *                  add1
 *                  /  \
 *                /    add0
 *                \     / \
 *                 \  /    \
 *                 abs2   abs5
 *                  |      |
 *                abs1   abs4
 *                 |      |
 *               abs0   load1
 *                |
 *              load0
 */
TEST_F(BufQueAllocatorUT, test_inplace_resue_multi_input_output) {
  ge::AscGraph graph("test_inplace_resue_multi_input_output");
  ge::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs2("abs2");
  abs2.x = abs1.y;
  abs2.attr.api.compute_type = ComputeType::kComputeElewise;
  abs2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs4("abs4");
  abs4.x = load1.y;
  abs4.attr.api.compute_type = ComputeType::kComputeElewise;
  abs4.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs5("abs5");
  abs5.x = abs4.y;
  abs5.attr.api.compute_type = ComputeType::kComputeElewise;
  abs5.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Add add0("add0");
  add0.x1 = abs2.y;
  add0.x2 = abs5.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Add add1("add1");
  add1.x1 = abs2.y;
  add1.x2 = add0.y;
  add1.attr.api.compute_type = ComputeType::kComputeElewise;
  add1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs3("abs3");
  abs3.x = add1.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs7("abs7");
  abs7.x = abs3.y;
  abs7.attr.api.compute_type = ComputeType::kComputeElewise;
  abs7.attr.api.unit = ComputeUnit::kUnitVector;
  abs7.y.dtype = DataType::DT_FLOAT16;

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Add add2("add2");
  add2.x1 = abs7.y;
  add2.x2 = load2.y;
  add2.attr.api.compute_type = ComputeType::kComputeElewise;
  add2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs9("abs9");
  abs9.x = add2.y;
  abs9.attr.api.compute_type = ComputeType::kComputeElewise;
  abs9.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = abs9.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);

  auto load_result = graph.FindNode("load0");  // vec in
  EXPECT_EQ(load_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load_result->outputs[0].attr.mem.reuse_id, 0);
  auto load1_result = graph.FindNode("load1");  // vec in
  EXPECT_EQ(load1_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load1_result->outputs[0].attr.mem.reuse_id, 4);
  auto load2_result = graph.FindNode("load2");  // vec in
  EXPECT_EQ(load2_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load2_result->outputs[0].attr.mem.reuse_id, 11);

  auto abs0_result = graph.FindNode("abs0");  // vec calc
  EXPECT_EQ(abs0_result->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(abs0_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs1_result = graph.FindNode("abs1");  // vec calc
  EXPECT_EQ(abs1_result->outputs[0].attr.buf.id, 2);
  EXPECT_EQ(abs1_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs2_result = graph.FindNode("abs2");  // vec calc
  EXPECT_EQ(abs2_result->outputs[0].attr.buf.id, 3);
  EXPECT_EQ(abs2_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);

  auto abs4_result = graph.FindNode("abs4");  // vec calc
  EXPECT_EQ(abs4_result->outputs[0].attr.buf.id, 4);
  EXPECT_EQ(abs4_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs5_result = graph.FindNode("abs5");  // vec calc reuse que
  EXPECT_EQ(abs5_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs5_result->outputs[0].attr.mem.reuse_id, 6);

  auto add0_result = graph.FindNode("add0");  // vec calc
  EXPECT_EQ(add0_result->outputs[0].attr.buf.id, 5);
  EXPECT_EQ(add0_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto add1_result = graph.FindNode("add1");  // vec calc
  EXPECT_EQ(add1_result->outputs[0].attr.buf.id, 6);
  EXPECT_EQ(add1_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs3_result = graph.FindNode("abs3");  // vec calc
  EXPECT_EQ(abs3_result->outputs[0].attr.buf.id, 7);
  EXPECT_EQ(abs3_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto abs7_result = graph.FindNode("abs7");  // vec calc
  EXPECT_EQ(abs7_result->outputs[0].attr.buf.id, 8);
  EXPECT_EQ(abs7_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);
  auto add2_result = graph.FindNode("add2");  // vec calc
  EXPECT_EQ(add2_result->outputs[0].attr.buf.id, 9);
  EXPECT_EQ(add2_result->outputs[0].attr.mem.reuse_id, ge::kIdNone);

  auto abs9_result = graph.FindNode("abs9");  // vecout
  EXPECT_EQ(abs9_result->outputs[0].attr.que.id, 2);
  EXPECT_EQ(abs9_result->outputs[0].attr.mem.reuse_id, 13);
}

TEST_F(BufQueAllocatorUT, TestTensorInfoToStr) {
  TensorInfo info;
  info.group_id = 2;
  info.life_start = 10;
  info.life_end = 20;
  info.mem_position = ge::Position::kPositionVecIn;
  info.loop_axes = {1, 2};
  std::string res = info.ToString();
  ASSERT_FALSE(res.empty());
}

TEST_F(BufQueAllocatorUT, test_tmp_buff_reuse) {
  ge::AscGraph graph("LoadAbsStore");
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  ge::ascir_op::Pow pow0("pow0");
  pow0.x1 = load0.y;
  pow0.x2 = load1.y;
  pow0.attr.api.compute_type = ComputeType::kComputeElewise;
  pow0.attr.api.type = ge::ApiType::kAPITypeCompute;
  pow0.attr.api.unit = ComputeUnit::kUnitVector;
  pow0.attr.sched.axis = {z0.id, z1.id};
  *pow0.y.axis = {z0.id, z1.id};
  *pow0.y.repeats = {s0, s1};
  *pow0.y.strides = {s1, ONE};

  ge::ascir_op::Abs abs("abs");
  abs.x = load2.y;
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs.attr.api.unit = ComputeUnit::kUnitVector;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, ONE};

  ge::ascir_op::Add add0("add0");
  add0.x1 = pow0.y;
  add0.x2 = abs.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.api.unit = ComputeUnit::kUnitVector;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};

  ge::ascir_op::Sigmoid Sigmoid("Sigmoid");
  Sigmoid.x = add0.y;
  Sigmoid.attr.api.compute_type = ComputeType::kComputeElewise;
  Sigmoid.attr.api.type = ge::ApiType::kAPITypeCompute;
  Sigmoid.attr.api.unit = ComputeUnit::kUnitVector;
  Sigmoid.attr.sched.axis = {z0.id, z1.id};
  *Sigmoid.y.axis = {z0.id, z1.id};
  *Sigmoid.y.repeats = {s0, s1};
  *Sigmoid.y.strides = {s1, ONE};

  ge::ascir_op::Store store("store");
  store.x = Sigmoid.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;

  size_t total_vecin_nums = 0UL;
  size_t total_vecout_nums = 0UL;
  BufQueAllocator().AllocateWithinGroup(graph, total_vecin_nums, total_vecout_nums);
  auto load0_result = graph.FindNode(load0.GetName().c_str());
  EXPECT_EQ(load0_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load0_result->outputs[0].attr.buf.id, -1);
  EXPECT_EQ(load0_result->outputs[0].attr.mem.reuse_id, 0);
  auto load1_result = graph.FindNode(load1.GetName().c_str());
  EXPECT_EQ(load1_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load1_result->outputs[0].attr.buf.id, -1);
  EXPECT_EQ(load1_result->outputs[0].attr.mem.reuse_id, 0);
  auto load2_result = graph.FindNode(load2.GetName().c_str());
  EXPECT_EQ(load2_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load2_result->outputs[0].attr.buf.id, -1);
  EXPECT_EQ(load2_result->outputs[0].attr.mem.reuse_id, 1);
  auto pow0_result = graph.FindNode(pow0.GetName().c_str());
  EXPECT_EQ(pow0_result->attr.tmp_buffers[0].id, 0);
  EXPECT_EQ(pow0_result->outputs[0].attr.que.id, -1);
  EXPECT_EQ(pow0_result->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(pow0_result->outputs[0].attr.mem.reuse_id, -1);
  auto abs_result = graph.FindNode(abs.GetName().c_str());
  EXPECT_EQ(abs_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(abs_result->outputs[0].attr.buf.id, -1);
  EXPECT_EQ(abs_result->outputs[0].attr.mem.reuse_id, 3);
  auto add0_result = graph.FindNode(add0.GetName().c_str());
  EXPECT_EQ(add0_result->outputs[0].attr.que.id, -1);
  EXPECT_EQ(add0_result->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(add0_result->outputs[0].attr.mem.reuse_id, -1);
  auto Sigmoid_result = graph.FindNode(Sigmoid.GetName().c_str());
  EXPECT_EQ(Sigmoid_result->attr.tmp_buffers[0].id, 0);
  EXPECT_EQ(Sigmoid_result->outputs[0].attr.que.id, 2);
  EXPECT_EQ(Sigmoid_result->outputs[0].attr.buf.id, -1);
  EXPECT_EQ(Sigmoid_result->outputs[0].attr.mem.reuse_id, 5);
}