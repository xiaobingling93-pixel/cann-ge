/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
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
#include "default_reg_func_v2.h"

namespace ge {
namespace ascir {

using namespace testing;
using namespace ge::ascir_op;

class CalcXorTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcXorTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsINT16
 * @tc.number: CalcXorTmpSizeV2_Test_001
 * @tc.desc: Test CalcXorTmpSizeV2 returns correct size when input is DT_INT16
 */
TEST_F(CalcXorTmpSizeV2Test, CalcXorTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsINT16)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1 + s2);
    auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
    auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Xor xor_op("xor");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_INT16;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_INT16;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    xor_op.x1 = load1.y;
    xor_op.x2 = load1.y;
    xor_op.attr.sched.axis = {z0.id, zo_s_0.id};
    xor_op.y.dtype = ge::DT_INT16;
    *xor_op.y.axis = {z0.id, zo_s_0.id};
    *xor_op.y.repeats = {s0, s1};
    *xor_op.y.strides = {s1, Symbol(1)};

    store.x = xor_op.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_INT16;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_INT16;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("xor");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcXorTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(256));  // XOR_ONE_REPEAT_BYTE_SIZE * XOR_UINT16_CALC_PROC = 256 * 1
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name: CalcXorTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsUINT16
 * @tc.number: CalcXorTmpSizeV2_Test_002
 * @tc.desc: Test CalcXorTmpSizeV2 returns correct size when input is DT_UINT16
 */
TEST_F(CalcXorTmpSizeV2Test, CalcXorTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsUINT16)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1 + s2);
    auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
    auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Xor xor_op("xor");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_UINT16;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_UINT16;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    xor_op.x1 = load1.y;
    xor_op.x2 = load1.y;
    xor_op.attr.sched.axis = {z0.id, zo_s_0.id};
    xor_op.y.dtype = ge::DT_UINT16;
    *xor_op.y.axis = {z0.id, zo_s_0.id};
    *xor_op.y.repeats = {s0, s1};
    *xor_op.y.strides = {s1, Symbol(1)};

    store.x = xor_op.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_UINT16;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_UINT16;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("xor");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcXorTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(256));  // XOR_ONE_REPEAT_BYTE_SIZE * XOR_UINT16_CALC_PROC = 256 * 1
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

} // namespace ascir
} // namespace ge
