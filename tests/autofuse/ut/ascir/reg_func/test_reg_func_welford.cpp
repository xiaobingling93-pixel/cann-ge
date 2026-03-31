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
namespace ge{
namespace ascir{
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcWelfordUpdateTmpSize(const ge::AscNode &node);
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcWelfordFinalizeTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcWelfordUpdateTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

// 使用符号化size构建2D shape [rn_len, ab_len]，测试CalcWelfordUpdateTmpSize
// 公式: ceil(rn_len * ab_len / 64) * 512
TEST_F(CalcWelfordUpdateTmpSizeTest, CalcWelfordUpdateTmpSizeWithSymbolicShape)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id, zo.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id, zo.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id, zo.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id, zo.id};
    *erf.y.repeats = {s0, s1};
    *erf.y.strides = {s1, Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id, zo.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordUpdateTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
    // ceil(s0 * s1 / 64) * 512
    ASSERT_EQ(result[0]->size, sym::Min(Symbol(65312), (((s0 * s1) + Symbol(63)) * Symbol(8))));
}

// 使用具体数值: rn_len=10, ab_len=100 => ceil(1000/64)*512 = 16*512 = 8192
TEST_F(CalcWelfordUpdateTmpSizeTest, CalcWelfordUpdateTmpSizeWithConcreteValues)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar(10);
    auto s1 = graph.CreateSizeVar(100);

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id, zo.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id, zo.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id, zo.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id, zo.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id, zo.id};
    *erf.y.repeats = {s0, s1};
    *erf.y.strides = {s1, Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id, zo.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordUpdateTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
    ASSERT_EQ(result[0]->size, Symbol(8504));
}

// 测试repeats只有1维时走默认ab_len=1的分支: ceil(s0 / 64) * 512
TEST_F(CalcWelfordUpdateTmpSizeTest, CalcWelfordUpdateTmpSizeWithSingleDimShape)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");

    auto z0 = graph.CreateAxis("z0", s0);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id};
    *x1.y.repeats = {s0};
    *x1.y.strides = {Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id};
    *load1.y.repeats = {s0};
    *load1.y.strides = {Symbol(1)};
    *load1.y.vectorized_axis = {z0.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id};
    *erf.y.repeats = {s0};
    *erf.y.strides = {Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id};
    *store.y.repeats = {s0};
    *store.y.strides = {Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id};
    *y.y.repeats = {s0};
    *y.y.strides = {Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordUpdateTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    // repeats.size() == 1 < 2 => ab_len = 1 => ceil(s0 * 1 / 64) * 512
    ASSERT_EQ(result[0]->size, sym::Min(Symbol(65312), ((Symbol(63) + s0) * Symbol(8))));
}

class CalcWelfordFinalizeTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

// 使用符号化size构建1D shape [ab_len]，测试CalcWelfordFinalizeTmpSize
// 公式: max(1024, 512 + ab_len * 8)
TEST_F(CalcWelfordFinalizeTmpSizeTest, CalcWelfordFinalizeTmpSizeWithSymbolicShape)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar("s0");

    auto z0 = graph.CreateAxis("z0", s0);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id};
    *x1.y.repeats = {s0};
    *x1.y.strides = {Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id};
    *load1.y.repeats = {s0};
    *load1.y.strides = {Symbol(1)};
    *load1.y.vectorized_axis = {z0.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id};
    *erf.y.repeats = {s0};
    *erf.y.strides = {Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id};
    *store.y.repeats = {s0};
    *store.y.strides = {Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id};
    *y.y.repeats = {s0};
    *y.y.strides = {Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordFinalizeTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
    // max(1024, 512 + s0 * 8)
    ASSERT_EQ(result[0]->size, sym::Min(Symbol(65312), ge::sym::Max(Symbol(1024), Symbol(512) + Symbol(8) * s0)));
}

// ab_len=32 (<64): max(1024, 512+32*8) = max(1024, 768) = 1024
TEST_F(CalcWelfordFinalizeTmpSizeTest, CalcWelfordFinalizeTmpSizeWhenAbLenSmall)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar(32);

    auto z0 = graph.CreateAxis("z0", s0);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id};
    *x1.y.repeats = {s0};
    *x1.y.strides = {Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id};
    *load1.y.repeats = {s0};
    *load1.y.strides = {Symbol(1)};
    *load1.y.vectorized_axis = {z0.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id};
    *erf.y.repeats = {s0};
    *erf.y.strides = {Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id};
    *store.y.repeats = {s0};
    *store.y.strides = {Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id};
    *y.y.repeats = {s0};
    *y.y.strides = {Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordFinalizeTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(1024));
}

// ab_len=100 (>64): max(1024, 512+100*8) = max(1024, 1312) = 1312
TEST_F(CalcWelfordFinalizeTmpSizeTest, CalcWelfordFinalizeTmpSizeWhenAbLenLarge)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar(100);

    auto z0 = graph.CreateAxis("z0", s0);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id};
    *x1.y.repeats = {s0};
    *x1.y.strides = {Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id};
    *load1.y.repeats = {s0};
    *load1.y.strides = {Symbol(1)};
    *load1.y.vectorized_axis = {z0.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id};
    *erf.y.repeats = {s0};
    *erf.y.strides = {Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id};
    *store.y.repeats = {s0};
    *store.y.strides = {Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id};
    *y.y.repeats = {s0};
    *y.y.strides = {Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordFinalizeTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(1312));
}

// ab_len=64 (边界值): max(1024, 512+64*8) = max(1024, 1024) = 1024
TEST_F(CalcWelfordFinalizeTmpSizeTest, CalcWelfordFinalizeTmpSizeWhenAbLenEq64)
{
    ge::AscGraph graph("test");
    auto s0 = graph.CreateSizeVar(64);

    auto z0 = graph.CreateAxis("z0", s0);

    ge::ascir_op::Data x1("x1", graph);
    ge::ascir_op::Load load1("load1");
    ge::ascir_op::Erf erf("erf");
    ge::ascir_op::Store store("store");
    ge::ascir_op::Output y("y");

    x1.attr.sched.axis = {z0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id};
    *x1.y.repeats = {s0};
    *x1.y.strides = {Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id};
    *load1.y.repeats = {s0};
    *load1.y.strides = {Symbol(1)};
    *load1.y.vectorized_axis = {z0.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id};
    erf.y.dtype = ge::DT_FLOAT;
    *erf.y.axis = {z0.id};
    *erf.y.repeats = {s0};
    *erf.y.strides = {Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id};
    *store.y.repeats = {s0};
    *store.y.strides = {Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id};
    *y.y.repeats = {s0};
    *y.y.strides = {Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcWelfordFinalizeTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->size, Symbol(1024));
}

} // namespace ascir
} // namespace ge
