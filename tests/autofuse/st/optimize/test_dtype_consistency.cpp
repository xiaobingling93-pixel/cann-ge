/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#define private public
#include "ascir_utils.h"
#include "optimize/graph_completeness/dtype_consistency.h"
#undef private

using namespace ge;
using namespace ge::ascir_op;
using ge::ops::IsOps;
using ge::ops::One;

namespace {

class TestDtypeConsistencyST : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ST Test 1: Complex FP16 -> FP32 -> FP16 conversion chain with multiple operators
TEST_F(TestDtypeConsistencyST, ComplexFp16ToFp32Chain) {
  AscGraph graph("test_complex_fp16_fp32_chain");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  // Create nodes: Data -> Load -> Add -> Mul -> Store -> Output
  ge::ascir_op::Data x1("x1", graph);
  ge::ascir_op::Load load("load");
  ge::ascir_op::Add add("add");
  ge::ascir_op::Mul mul("mul");
  ge::ascir_op::Store store("store");
  ge::ascir_op::Output y("y");

  // Setup data node with FP16
  x1.ir_attr.SetIndex(0);
  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  // Setup load with FP16
  load.x = x1.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Setup add with FP16 input/output (should be converted to FP32)
  add.x1 = load.y;
  add.x2 = load.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  // Setup mul with FP16 input/output (should be converted to FP32)
  mul.x1 = add.y;
  mul.x2 = add.y;
  mul.attr.sched.axis = {z0.id};
  mul.y.dtype = ge::DT_FLOAT16;
  *mul.y.axis = {z0.id};
  *mul.y.repeats = {s0};
  *mul.y.strides = {One};

  // Setup store with FP16
  store.x = mul.y;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  // Setup output with FP16
  y.x = store.y;
  y.attr.sched.axis = {z0.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id};
  *y.y.repeats = {s0};
  *y.y.strides = {One};

  // Mock requirements: Add and Mul need FP32 for computation
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify the result:
  // Expected: Data(FP16) -> Load(FP16) -> Cast(FP16->FP32) -> Add(FP32) -> Mul(FP32) -> Cast(FP32->FP16) -> Store(FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;
  bool mul_is_fp32 = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
    if (node->GetType() == "Mul") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        mul_is_fp32 = true;
      }
    }
  }

  // Should have 2 Cast nodes: one before Add (FP16->FP32), one after Mul (FP32->FP16)
  EXPECT_EQ(cast_count, 2U);
  EXPECT_TRUE(add_is_fp32);
  EXPECT_TRUE(mul_is_fp32);
}

// ST Test 2: Multi-input with different dtypes, all need FP32
TEST_F(TestDtypeConsistencyST, MultiInputDifferentDtypes) {
  AscGraph graph("test_multi_input_diff_dtypes");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  // Create two data inputs: one FP16, one FP32
  ge::ascir_op::Data data1("data1", graph);
  ge::ascir_op::Data data2("data2", graph);

  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id};
  *data1.y.repeats = {s0};
  *data1.y.strides = {One};

  data2.ir_attr.SetIndex(1);
  data2.attr.sched.axis = {z0.id};
  data2.y.dtype = ge::DT_FLOAT16;
  *data2.y.axis = {z0.id};
  *data2.y.repeats = {s0};
  *data2.y.strides = {One};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {One};

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id};
  *load2.y.repeats = {s0};
  *load2.y.strides = {One};

  // Add node needs FP32
  ge::ascir_op::Add add("add");
  add.x1 = load1.y;
  add.x2 = load2.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load1"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("load2"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify both inputs are cast to FP32
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->inputs[1].attr.dtype == ge::DT_FLOAT &&
          node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
  }

  EXPECT_EQ(cast_count, 3U);  // Two cast nodes before add, one after add
  EXPECT_TRUE(add_is_fp32);
}

// ST Test 3: Merge with upstream cast optimization
TEST_F(TestDtypeConsistencyST, MergeWithUpstreamCast) {
  AscGraph graph("test_merge_upstream_cast");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id};
  *data1.y.repeats = {s0};
  *data1.y.strides = {One};

  ge::ascir_op::Load load("load");
  load.x = data1.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Manually add a Cast FP16->FP32
  ge::ascir_op::Cast existing_cast("existing_cast");
  existing_cast.x = load.y;
  existing_cast.attr.sched.axis = {z0.id};
  existing_cast.y.dtype = ge::DT_FLOAT;
  *existing_cast.y.axis = {z0.id};
  *existing_cast.y.repeats = {s0};
  *existing_cast.y.strides = {One};

  // Add needs FP32, but upstream already has FP32 cast
  ge::ascir_op::Add add("add");
  add.x1 = existing_cast.y;
  add.x2 = existing_cast.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify upstream cast is reused, no new cast inserted before add
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
  }

  // Should have 2 Cast: existing upstream one and one after add (FP32->FP16)
  EXPECT_EQ(cast_count, 2U);
  EXPECT_TRUE(add_is_fp32);
}

// ST Test 4: Cast CSE - multiple identical casts from same upstream
TEST_F(TestDtypeConsistencyST, CastCSEComplex) {
  AscGraph graph("test_cast_cse_complex");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id};
  *data1.y.repeats = {s0};
  *data1.y.strides = {One};

  ge::ascir_op::Load load("load");
  load.x = data1.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Multiple operators need FP32 from same FP16 source
  ge::ascir_op::Add add("add");
  add.x1 = load.y;
  add.x2 = load.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  ge::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.attr.sched.axis = {z0.id};
  mul.y.dtype = ge::DT_FLOAT16;
  *mul.y.axis = {z0.id};
  *mul.y.repeats = {s0};
  *mul.y.strides = {One};

  ge::ascir_op::Sub sub("sub");
  sub.x1 = load.y;
  sub.x2 = load.y;
  sub.attr.sched.axis = {z0.id};
  sub.y.dtype = ge::DT_FLOAT16;
  *sub.y.axis = {z0.id};
  *sub.y.repeats = {s0};
  *sub.y.strides = {One};

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("sub"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify CSE merged identical casts
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  size_t cast_with_multiple_downstream = 0;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
      auto out_anchor = node->GetOutDataAnchor(0);
      if (out_anchor != nullptr && out_anchor->GetPeerInDataAnchors().size() >= 6) {
        cast_with_multiple_downstream++;
      }
    }
  }

  // Should have only 1 Cast node (merged) serving all 3 operators
  EXPECT_EQ(cast_count, 1U);
  EXPECT_EQ(cast_with_multiple_downstream, 1U);
}

// ST Test 5: Cancel identity cast (Cast FP32->FP32)
TEST_F(TestDtypeConsistencyST, CancelIdentityCastComplex) {
  AscGraph graph("test_identity_cast_complex");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id};
  data1.y.dtype = ge::DT_FLOAT;
  *data1.y.axis = {z0.id};
  *data1.y.repeats = {s0};
  *data1.y.strides = {One};

  ge::ascir_op::Load load("load");
  load.x = data1.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Add identity cast (FP32 -> FP32)
  ge::ascir_op::Cast identity_cast("identity_cast");
  identity_cast.x = load.y;
  identity_cast.attr.sched.axis = {z0.id};
  identity_cast.y.dtype = ge::DT_FLOAT;
  *identity_cast.y.axis = {z0.id};
  *identity_cast.y.repeats = {s0};
  *identity_cast.y.strides = {One};

  ge::ascir_op::Add add("add");
  add.x1 = identity_cast.y;
  add.x2 = identity_cast.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify identity cast is removed
  auto all_nodes = graph.GetAllNodes();
  size_t active_cast_count = 0;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        active_cast_count++;
      }
    }
  }

  EXPECT_EQ(active_cast_count, 0U);
}

// ST Test 6: Multiple data type conversions in a deep chain
TEST_F(TestDtypeConsistencyST, DeepChainWithMultipleConversions) {
  AscGraph graph("test_deep_chain");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id};
  *data1.y.repeats = {s0};
  *data1.y.strides = {One};

  ge::ascir_op::Load load("load");
  load.x = data1.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Chain: Add -> Mul -> Add -> Mul, all need FP32
  ge::ascir_op::Add add1("add1");
  add1.x1 = load.y;
  add1.x2 = load.y;
  add1.attr.sched.axis = {z0.id};
  add1.y.dtype = ge::DT_FLOAT16;
  *add1.y.axis = {z0.id};
  *add1.y.repeats = {s0};
  *add1.y.strides = {One};

  ge::ascir_op::Mul mul1("mul1");
  mul1.x1 = add1.y;
  mul1.x2 = add1.y;
  mul1.attr.sched.axis = {z0.id};
  mul1.y.dtype = ge::DT_FLOAT16;
  *mul1.y.axis = {z0.id};
  *mul1.y.repeats = {s0};
  *mul1.y.strides = {One};

  ge::ascir_op::Add add2("add2");
  add2.x1 = mul1.y;
  add2.x2 = mul1.y;
  add2.attr.sched.axis = {z0.id};
  add2.y.dtype = ge::DT_FLOAT16;
  *add2.y.axis = {z0.id};
  *add2.y.repeats = {s0};
  *add2.y.strides = {One};

  ge::ascir_op::Mul mul2("mul2");
  mul2.x1 = add2.y;
  mul2.x2 = add2.y;
  mul2.attr.sched.axis = {z0.id};
  mul2.y.dtype = ge::DT_FLOAT16;
  *mul2.y.axis = {z0.id};
  *mul2.y.repeats = {s0};
  *mul2.y.strides = {One};

  ge::ascir_op::Store store("store");
  store.x = mul2.y;
  store.attr.sched.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify only 2 casts: one at beginning, one at end
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
  }

  EXPECT_EQ(cast_count, 2U);
}

// ST Test 8: TryMergeWithUpstreamCast - multiple consumers case (create new merged cast)
// Safe scenario: FP16 -> FP32 (widening), multiple consumers need same dtype
// mul1 needs FP32 (same as cast output), abs needs FP32 (same as cast output)
TEST_F(TestDtypeConsistencyST, MergeUpstreamCastMultipleConsumers) {
  AscGraph graph("test_merge_multiple_consumers");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id};
  *data1.y.repeats = {s0};
  *data1.y.strides = {One};

  ge::ascir_op::Load load("load");
  load.x = data1.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Upstream cast: FP16 -> FP32 (safe widening)
  ge::ascir_op::Cast cast("cast");
  cast.x = load.y;
  cast.attr.sched.axis = {z0.id};
  cast.y.dtype = ge::DT_FLOAT;
  *cast.y.axis = {z0.id};
  *cast.y.repeats = {s0};
  *cast.y.strides = {One};

  // Mul1 uses upstream cast, needs FP32 (same as cast output, no merge needed, just reuse)
  ge::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast.y;
  mul1.x2 = cast.y;
  mul1.attr.sched.axis = {z0.id};
  mul1.y.dtype = ge::DT_FLOAT;
  *mul1.y.axis = {z0.id};
  *mul1.y.repeats = {s0};
  *mul1.y.strides = {One};

  // Abs uses upstream cast, needs FP32 (same as cast output, no merge needed, just reuse)
  ge::ascir_op::Abs abs("abs");
  abs.x = cast.y;
  abs.attr.sched.axis = {z0.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};

  // Add uses abs and mul1 outputs
  ge::ascir_op::Add add("add");
  add.x1 = abs.y;
  add.x2 = mul1.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  ge::ascir_op::Store store1("store1");
  store1.x = add.y;
  store1.attr.sched.axis = {z0.id};
  store1.y.dtype = ge::DT_FLOAT;
  *store1.y.axis = {z0.id};
  *store1.y.repeats = {s0};
  *store1.y.strides = {One};

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("abs"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), ge::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), ge::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), ge::SUCCESS);

  // Verify: single FP16->FP32 cast serves all consumers (mul1, abs)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool mul1_is_fp32 = false;
  bool abs_is_fp32 = false;
  bool has_fp16_to_fp32_cast = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_FLOAT16 && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
          has_fp16_to_fp32_cast = true;
        }
      }
    }
    if (node->GetName() == "mul1") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        mul1_is_fp32 = true;
      }
    }
    if (node->GetName() == "abs") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        abs_is_fp32 = true;
      }
    }
  }

  // Should have 1 Cast: FP16->FP32 serving all consumers
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(mul1_is_fp32);
  EXPECT_TRUE(abs_is_fp32);
  EXPECT_TRUE(has_fp16_to_fp32_cast);
}

// ST Test 9: TryMergeWithUpstreamCast - safe widening merge scenario with single consumer
// Safe scenario: FP16 -> FP32, downstream node needs FP32 (same as cast output)
TEST_F(TestDtypeConsistencyST, TryMergeWithUpstreamCast) {
  AscGraph graph("test_merge_upstream_multiple");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id};
  *data0.y.repeats = {s0};
  *data0.y.strides = {One};

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.attr.sched.axis = {z0.id};
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {One};

  // Upstream cast: FP16 -> FP32 (safe widening)
  ge::ascir_op::Cast cast0("cast0");
  cast0.x = load.y;
  cast0.attr.sched.axis = {z0.id};
  cast0.y.dtype = ge::DT_FLOAT;
  *cast0.y.axis = {z0.id};
  *cast0.y.repeats = {s0};
  *cast0.y.strides = {One};

  // Downstream node that receives cast output
  ge::ascir_op::Add add("add");
  add.x1 = cast0.y;
  add.x2 = cast0.y;
  add.attr.sched.axis = {z0.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  auto upstream_cast = graph.FindNode("cast0");
  auto downstream_node = graph.FindNode("add");

  // FP16->FP32 preserves values, target is FP32 (same as cast output), merge should succeed
  // This will modify cast0's output dtype to FP32 (no change needed) and update add's input dtype
  EXPECT_TRUE(optimize::DtypeConsistency::TryMergeWithUpstreamCast(graph, upstream_cast, downstream_node, 0, DT_FLOAT));
}
}  // namespace
