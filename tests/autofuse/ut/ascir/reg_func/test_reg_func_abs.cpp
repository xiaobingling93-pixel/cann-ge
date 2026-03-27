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
extern std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAbsTmpSize(const ge::AscNode &node);

using namespace testing;

class CalcAbsTmpSizeTest:public::testing::Test{
protected:
    void SetUp() override{}
    void TearDown() override{}
};

// int32类型: tmp_size = 4 * input_size
TEST_F(CalcAbsTmpSizeTest, CalcAbsTmpSizeWhenInt32)
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
    x1.y.dtype = ge::DT_INT32;
    *x1.y.axis = {z0.id, zo.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo.id};
    load1.y.dtype = ge::DT_INT32;
    *load1.y.axis = {z0.id, zo.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo.id};

    erf.x = load1.y;
    erf.attr.sched.axis = {z0.id, zo.id};
    erf.y.dtype = ge::DT_INT32;
    *erf.y.axis = {z0.id, zo.id};
    *erf.y.repeats = {s0, s1};
    *erf.y.strides = {s1, Symbol(1)};

    store.x = erf.y;
    store.attr.sched.axis = {z0.id, zo.id};
    store.y.dtype = ge::DT_INT32;
    *store.y.axis = {z0.id, zo.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo.id};
    y.y.dtype = ge::DT_INT32;
    *y.y.axis = {z0.id, zo.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    std::shared_ptr<ge::AscNode> node = graph.FindNode("erf");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcAbsTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
    // int32: sizeof(int32) * input_size = 4 * s0 * s1
    ASSERT_EQ(result[0]->size, sym::Min(Symbol(65312), Symbol(4) * s0 * s1));
}

// float类型(非int32): 返回默认tmp buffer大小
TEST_F(CalcAbsTmpSizeTest, CalcAbsTmpSizeWhenFloat)
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

    std::vector<std::unique_ptr<ge::TmpBufDesc>> result = CalcAbsTmpSize(*node);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0]->life_time_axis_id, -1);
    // 非int32走CalcDefaultTmpSize => 返回 DEFAULT_TEMP_BUFFER_SIZE
    // CalcDefaultTmpSize调用GetTmpBuffer(Symbol(DEFAULT_TEMP_BUFFER_SIZE))
    ASSERT_EQ(result[0]->size, Symbol(8192));
}

} // namespace ascir
} // namespace ge
