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

class CalcSinhTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

/**
 * @tc.name: CalcSinhTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT
 * @tc.number: CalcSinhTmpSizeV2_Test_001
 * @tc.desc: Test CalcSinhTmpSizeV2 returns correct size when input is DT_FLOAT
 */
TEST_F(CalcSinhTmpSizeV2Test, CalcSinhTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT)
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
    ge::ascir_op::Sinh sinh("sinh");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    sinh.x = load1.y;
    sinh.attr.sched.axis = {z0.id, zo_s_0.id};
    sinh.y.dtype = ge::DT_FLOAT;
    *sinh.y.axis = {z0.id, zo_s_0.id};
    *sinh.y.repeats = {s0, s1};
    *sinh.y.strides = {s1, Symbol(1)};

    store.x = sinh.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("sinh");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcSinhTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(256));  // SINH_ONE_REPEAT_BYTE_SIZE * SINH_FLOAT_CALC_PROC = 256 * 1
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name: CalcSinhahTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT16
 * @tc.number: CalcSinhTmpSizeV2_Test_002
 * @tc.desc: Test CalcSinhTmpSizeV2 returns correct size when input is DT_FLOAT16
 */
TEST_F(CalcSinhTmpSizeV2Test, CalcSinhTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT16)
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
    ge::ascir_op::Sinh sinh("sinh");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_FLOAT16;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_FLOAT16;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    sinh.x = load1.y;
    sinh.attr.sched.axis = {z0.id, zo_s_0.id};
    sinh.y.dtype = ge::DT_FLOAT16;
    *sinh.y.axis = {z0.id, zo_s_0.id};
    *sinh.y.repeats = {s0, s1};
    *sinh.y.strides = {s1, Symbol(1)};

    store.x = sinh.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_FLOAT16;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_FLOAT16;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("sinh");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcSinhTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(1024));  // SINH_ONE_REPEAT_BYTE_SIZE * SINH_HALF_CALC_PROC = 256 * 4
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name: CalcSinhTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsBF16
 * @tc.number: CalcSinhTmpSizeV2_Test_003
 * @tc.desc: Test CalcSinhTmpSizeV2 returns correct size when input is DT_BF16
 */
TEST_F(CalcSinhTmpSizeV2Test, CalcSinhTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsBF16)
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
    ge::ascir_op::Sinh sinh("sinh");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_BF16;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_BF16;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    sinh.x = load1.y;
    sinh.attr.sched.axis = {z0.id, zo_s_0.id};
    sinh.y.dtype = ge::DT_BF16;
    *sinh.y.axis = {z0.id, zo_s_0.id};
    *sinh.y.repeats = {s0, s1};
    *sinh.y.strides = {s1, Symbol(1)};

    store.x = sinh.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = ge::DT_BF16;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = ge::DT_BF16;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("sinh");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcSinhTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(256));  // SINH_ONE_REPEAT_BYTE_SIZE * SINH_FLOAT_CALC_PROC = 256 * 1
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

} // namespace ascir
} // namespace ge
