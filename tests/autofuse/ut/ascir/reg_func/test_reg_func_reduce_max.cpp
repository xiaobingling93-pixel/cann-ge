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

#include "graph/operator_reg.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "../test_util.h"

namespace ge {
namespace ascir {

extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcReduceMaxTmpSize(const ge::AscNode &node);

using namespace testing;
using namespace ge::ascir_op;

class CalcReduceMaxTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcReduceMaxTmpSize_EmptyOutputStrides
 * @tc.desc: Test when output vectorized_strides is empty, returns empty vector
 */
TEST_F(CalcReduceMaxTmpSizeTest, CalcReduceMaxTmpSize_EmptyOutputStrides) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1("x1", graph);
  Load load1("load1");
  Max max0("max0");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_INT32;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_INT32;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id};
  max0.attr.sched.loop_axis = {z1.id};  // reduce along last axis
  max0.y.dtype = ge::DT_INT32;
  *max0.y.axis = {z0.id, z1.id};
  *max0.y.repeats = {s0, Symbol(1)};
  *max0.y.strides = {Symbol(1), Symbol(0)};
  *max0.y.vectorized_axis = {z0.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_INT32;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_INT32;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  // Do NOT set output vectorized_strides (empty)

  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcReduceMaxTmpSize(*node);
  ASSERT_EQ(result.size(), 0);
}

/**
 * @tc.name: CalcReduceMaxTmpSize_NonInt32_Float
 * @tc.desc: Test non-int32 dtype (float), returns default tmp size 8192
 */
TEST_F(CalcReduceMaxTmpSizeTest, CalcReduceMaxTmpSize_NonInt32_Float) {
  ge::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1("x1", graph);
  Load load1("load1");
  Max max0("max0");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_FLOAT;  // Non-int32 type
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id};
  max0.attr.sched.loop_axis = {z1.id};
  max0.y.dtype = ge::DT_FLOAT;
  *max0.y.axis = {z0.id, z1.id};
  *max0.y.repeats = {s0, Symbol(1)};
  *max0.y.strides = {Symbol(1), Symbol(0)};
  *max0.y.vectorized_axis = {z0.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<ge::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  node->outputs[0].attr.vectorized_strides = {Symbol(1), Symbol(0)};

  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcReduceMaxTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  // Non-int32 returns default tmp size
  ASSERT_EQ(result[0]->size, Symbol(8192));
}

// Helper function to create AR mode test node
static std::shared_ptr<ge::AscNode> CreateARMTestNode(ge::AscGraph& graph, ge::Expression s0, ge::Expression s1,
                                                       ge::Expression One, ge::Expression Zero) {
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  Data x1("x1", graph);
  Load load1("load1");
  Max max0("max0");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_INT32;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_INT32;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id};
  max0.attr.sched.loop_axis = {z1.id};
  max0.y.dtype = ge::DT_INT32;
  *max0.y.axis = {z0.id, z1.id};
  *max0.y.repeats = {s0, One};
  *max0.y.strides = {One, Zero};
  *max0.y.vectorized_axis = {z0.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_INT32;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_INT32;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, One};

  auto node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s1, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  return node;
}

/**
 * @tc.name: CalcReduceMaxTmpSize_AR_Mode_Int32
 * @tc.desc: Test AR mode (reduce along last axis) with int32, tmp_size = first * 32 + 320
 */
TEST_F(CalcReduceMaxTmpSizeTest, CalcReduceMaxTmpSize_AR_Mode_Int32) {
  ge::AscGraph graph("test");
  ge::Expression One = ge::Symbol(1);
  ge::Expression Zero = ge::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto node = CreateARMTestNode(graph, s0, s1, One, Zero);

  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcReduceMaxTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  ASSERT_EQ(result[0]->size, Symbol(32) * s0 + Symbol(320));
}

// Helper function to create RA mode test node
static std::shared_ptr<ge::AscNode> CreateRAMTestNode(ge::AscGraph& graph, ge::Expression s0, ge::Expression s1,
                                                       ge::Expression One, ge::Expression Zero) {
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  Data x1("x1", graph);
  Load load1("load1");
  Max max0("max0");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = ge::DT_INT32;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_INT32;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id};
  max0.attr.sched.loop_axis = {z0.id};
  max0.y.dtype = ge::DT_INT32;
  *max0.y.axis = {z0.id, z1.id};
  *max0.y.repeats = {One, s1};
  *max0.y.strides = {Zero, One};
  *max0.y.vectorized_axis = {z1.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_INT32;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_INT32;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, One};

  auto node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s1, One};
  node->outputs[0].attr.vectorized_strides = {Zero, s1};
  return node;
}

/**
 * @tc.name: CalcReduceMaxTmpSize_RA_Mode_Int32
 * @tc.desc: Test RA mode (reduce along first axis) with int32, tmp_size = input_size * 4 + 288
 */
TEST_F(CalcReduceMaxTmpSizeTest, CalcReduceMaxTmpSize_RA_Mode_Int32) {
  ge::AscGraph graph("test");
  ge::Expression One = ge::Symbol(1);
  ge::Expression Zero = ge::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto node = CreateRAMTestNode(graph, s0, s1, One, Zero);

  std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcReduceMaxTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  ASSERT_EQ(result[0]->size, Symbol(4) * s0 * s1 + Symbol(288));
}

}  // namespace ascir
}  // namespace ge
