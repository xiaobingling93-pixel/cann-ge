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

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

namespace optimize {
void CreatSomeInputFusedConcatGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s1 + s1 + s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ONE};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, ONE};

  Exp exp0("exp0");
  exp0.x = abs0.y;
  exp0.attr.sched.axis = {z0.id, z1.id};
  *exp0.y.axis = {z0.id, z1.id};
  *exp0.y.repeats = {s0, s1};
  *exp0.y.strides = {s1, ONE};

  Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ONE};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, ONE};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  Abs abs2("abs2");
  abs2.x = load2.y;
  abs2.attr.sched.axis = {z0.id, z1.id};
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, ONE};

  Exp exp2("exp2");
  exp2.x = abs2.y;
  exp2.attr.sched.axis = {z0.id, z1.id};
  *exp2.y.axis = {z0.id, z1.id};
  *exp2.y.repeats = {s0, s1};
  *exp2.y.strides = {s1, ONE};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  *data3.y.repeats = {s0, s1};
  *data3.y.strides = {s1, ONE};
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ONE};

  Concat concat("concat");
  concat.x = {exp0.y, load1.y, exp2.y, load3.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + s1 + s1 + s1};
  *concat.y.strides = {s1 + s1 + s1 + s1, ONE};

  Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + s1 + s1 + s1};
  *store.y.strides = {s1 + s1 + s1 + s1, ONE};

  Output y("output");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
}

void CreatBrcCascadeGraph(ge::AscGraph &graph) {
  auto s0 = Symbol(128);
  auto s1 = Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, Zero};

  Broadcast brc3("brc3");
  brc3.x = load1.y;
  brc3.attr.sched.axis = {z0.id, z1.id};
  *brc3.y.axis = {z0.id, z1.id};
  *brc3.y.repeats = {s0, s1};
  *brc3.y.strides = {s1, One};

  Gt gt("gt");
  gt.x1 = load0.y;
  gt.x2 = brc3.y;
  gt.attr.sched.axis = {z0.id, z1.id};
  *gt.y.axis = {z0.id, z1.id};
  *gt.y.repeats = {s0, s1};
  *gt.y.strides = {s1, One};

  Sigmoid sigmoid0("sigmoid0");
  sigmoid0.x = gt.y;
  sigmoid0.attr.sched.axis = {z0.id, z1.id};
  *sigmoid0.y.axis = {z0.id, z1.id};
  *sigmoid0.y.repeats = {s0, s1};
  *sigmoid0.y.strides = {s1, One};

  Abs abs0("abs0");
  abs0.x = gt.y;
  abs0.attr.sched.axis = {z0.id, z1.id};
  *abs0.y.axis = {z0.id, z1.id};
  *abs0.y.repeats = {s0, s1};
  *abs0.y.strides = {s1, One};

  Scalar scalar1("scalar1", graph);

  Broadcast brc4("brc4");
  brc4.x = scalar1.y;
  brc4.attr.sched.axis = {z0.id, z1.id};
  *brc4.y.axis = {z0.id, z1.id};
  *brc4.y.repeats = {One, s1};
  *brc4.y.strides = {Zero, One};

  Broadcast brc5("brc5");
  brc5.x = brc4.y;
  brc5.attr.sched.axis = {z0.id, z1.id};
  *brc5.y.axis = {z0.id, z1.id};
  *brc5.y.repeats = {s0, s1};
  *brc5.y.strides = {s1, One};

  Add add("add");
  add.x1 = abs0.y;
  add.x2 = sigmoid0.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};

  Mul mul("mul");
  mul.x1 = add.y;
  mul.x2 = brc5.y;
  mul.attr.sched.axis = {z0.id, z1.id};
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, One};

  Abs abs1("abs1");
  abs1.x = mul.y;
  abs1.attr.sched.axis = {z0.id, z1.id};
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};

  Abs abs2("abs2");
  abs2.x = abs1.y;
  abs2.attr.sched.axis = {z0.id, z1.id};
  *abs2.y.axis = {z0.id, z1.id};
  *abs2.y.repeats = {s0, s1};
  *abs2.y.strides = {s1, One};

  Store store("store");
  store.x = abs2.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);
}

void CreatBrcReduceGraph(ge::AscGraph &graph) {
  auto s0 = Symbol(12);
  auto s1 = Symbol(16);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};

  Exp exp0("exp0");
  exp0.x = load0.y;
  exp0.attr.sched.axis = {z0.id, z1.id};
  *exp0.y.axis = {z0.id, z1.id};
  *exp0.y.repeats = {s0, One};
  *exp0.y.strides = {One, Zero};

  Broadcast brc0("brc0");
  brc0.x = exp0.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};

  Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};

  Relu relu0("relu0");
  relu0.x = load1.y;
  relu0.attr.sched.axis = {z0.id, z1.id};
  *relu0.y.axis = {z0.id, z1.id};
  *relu0.y.repeats = {s0, s1};
  *relu0.y.strides = {s1, One};

  Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = relu0.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  Sigmoid sigmoid("sigmoid");
  sigmoid.x = add0.y;
  sigmoid.attr.sched.axis = {z0.id, z1.id};
  *sigmoid.y.axis = {z0.id, z1.id};
  *sigmoid.y.repeats = {s0, s1};
  *sigmoid.y.strides = {s1, One};

  Max max0("max0");
  max0.x = sigmoid.y;
  max0.attr.sched.axis = {z0.id, z1.id};
  *max0.y.axis = {z0.id, z1.id};
  *max0.y.repeats = {s0, One};
  *max0.y.strides = {One, Zero};

  Sigmoid Sigmoid1("Sigmoid1");
  Sigmoid1.x = max0.y;
  Sigmoid1.attr.sched.axis = {z0.id, z1.id};
  *Sigmoid1.y.axis = {z0.id, z1.id};
  *Sigmoid1.y.repeats = {s0, One};
  *Sigmoid1.y.strides = {One, Zero};

  Store store("store");
  store.x = Sigmoid1.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, One};
  *store.y.strides = {One, Zero};

  Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);
}

void CreatNestingLoadGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s1 + s1 + s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ONE};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};

  Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ONE};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ONE};

  Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ONE};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, ONE};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ONE};

  Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = load2.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, ONE};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  *data3.y.repeats = {s0, s1};
  *data3.y.strides = {s1, ONE};
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ONE};

  Add add2("add2");
  add2.x1 = add1.y;
  add2.x2 = load3.y;
  add2.attr.sched.axis = {z0.id, z1.id};
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, ONE};

  Data data4("data4", graph);
  data4.attr.sched.axis = {z0.id, z1.id};
  *data4.y.axis = {z0.id, z1.id};
  *data4.y.repeats = {s0, s1};
  *data4.y.strides = {s1, ONE};
  data4.ir_attr.SetIndex(4);

  Load load4("load4");
  load4.x = data4.y;
  load4.attr.sched.axis = {z0.id, z1.id};
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {s0, s1};
  *load4.y.strides = {s1, ONE};

  Add add3("add3");
  add3.x1 = add2.y;
  add3.x2 = load4.y;
  add3.attr.sched.axis = {z0.id, z1.id};
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, ONE};

  Data data5("data5", graph);
  data5.attr.sched.axis = {z0.id, z1.id};
  *data5.y.axis = {z0.id, z1.id};
  *data5.y.repeats = {s0, s1};
  *data5.y.strides = {s1, ONE};
  data5.ir_attr.SetIndex(5);

  Load load5("load5");
  load5.x = data5.y;
  load5.attr.sched.axis = {z0.id, z1.id};
  *load5.y.axis = {z0.id, z1.id};
  *load5.y.repeats = {s0, s1};
  *load5.y.strides = {s1, ONE};

  Add add4("add4");
  add4.x1 = add3.y;
  add4.x2 = load5.y;
  add4.attr.sched.axis = {z0.id, z1.id};
  *add4.y.axis = {z0.id, z1.id};
  *add4.y.repeats = {s0, s1};
  *add4.y.strides = {s1, ONE};

  Data data6("data6", graph);
  data6.attr.sched.axis = {z0.id, z1.id};
  *data6.y.axis = {z0.id, z1.id};
  *data6.y.repeats = {s0, s1};
  *data6.y.strides = {s1, ONE};
  data6.ir_attr.SetIndex(6);

  Load load6("load6");
  load6.x = data6.y;
  load6.attr.sched.axis = {z0.id, z1.id};
  *load6.y.axis = {z0.id, z1.id};
  *load6.y.repeats = {s0, s1};
  *load6.y.strides = {s1, ONE};

  Add add5("add5");
  add5.x1 = add4.y;
  add5.x2 = load6.y;
  add5.attr.sched.axis = {z0.id, z1.id};
  *add5.y.axis = {z0.id, z1.id};
  *add5.y.repeats = {s0, s1};
  *add5.y.strides = {s1, ONE};

  Add add6("add6");
  add6.x1 = add5.y;
  add6.x2 = load5.y;
  add6.attr.sched.axis = {z0.id, z1.id};
  *add6.y.axis = {z0.id, z1.id};
  *add6.y.repeats = {s0, s1};
  *add6.y.strides = {s1, ONE};

  Add add7("add7");
  add7.x1 = add6.y;
  add7.x2 = load4.y;
  add7.attr.sched.axis = {z0.id, z1.id};
  *add7.y.axis = {z0.id, z1.id};
  *add7.y.repeats = {s0, s1};
  *add7.y.strides = {s1, ONE};

  Add add8("add8");
  add8.x1 = add7.y;
  add8.x2 = load3.y;
  add8.attr.sched.axis = {z0.id, z1.id};
  *add8.y.axis = {z0.id, z1.id};
  *add8.y.repeats = {s0, s1};
  *add8.y.strides = {s1, ONE};

  Add add9("add9");
  add9.x1 = add8.y;
  add9.x2 = load2.y;
  add9.attr.sched.axis = {z0.id, z1.id};
  *add9.y.axis = {z0.id, z1.id};
  *add9.y.repeats = {s0, s1};
  *add9.y.strides = {s1, ONE};

  Add add10("add10");
  add10.x1 = add9.y;
  add10.x2 = load1.y;
  add10.attr.sched.axis = {z0.id, z1.id};
  *add10.y.axis = {z0.id, z1.id};
  *add10.y.repeats = {s0, s1};
  *add10.y.strides = {s1, ONE};

  Add add11("add11");
  add11.x1 = add10.y;
  add11.x2 = load0.y;
  add11.attr.sched.axis = {z0.id, z1.id};
  *add11.y.axis = {z0.id, z1.id};
  *add11.y.repeats = {s0, s1};
  *add11.y.strides = {s1, ONE};

  Store store("store");
  store.x = add11.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, ONE};

  Output y("output");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);
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
  ge::AscGraph graph("Concat4InputsGraph");
  CreatSomeInputFusedConcatGraph(graph);
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
  ge::AscGraph graph("NestingLoadGraph");
  CreatNestingLoadGraph(graph);
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
  ge::AscGraph graph("MultiBrcGraph");
  CreatBrcReduceGraph(graph);
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
  ge::AscGraph graph("MultiBrcGraph");
  CreatBrcCascadeGraph(graph);
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
  ge::AscGraph graph("InplaceGraph");

  auto s0 = Symbol(128);
  auto z0 = graph.CreateAxis("z0", s0);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id};
  *load0.y.axis = {z0.id};
  *load0.y.repeats = {s0};
  *load0.y.strides = {One};

  Load load1("load1");
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id};
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {One};

  Pow pow0("pow0");
  pow0.x1 = load0.y;
  pow0.x2 = load1.y;
  pow0.attr.sched.axis = {z0.id};
  *pow0.y.axis = {z0.id};
  *pow0.y.repeats = {s0};
  *pow0.y.strides = {One};

  Load load2("load2");
  load2.x = data0.y;
  load2.attr.sched.axis = {z0.id};
  *load2.y.axis = {z0.id};
  *load2.y.repeats = {s0};
  *load2.y.strides = {One};

  Add add("add0");
  add.x1 = pow0.y;
  add.x2 = load2.y;
  add.attr.sched.axis = {z0.id};
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

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
  ge::AscGraph graph("tmp_buf_reuse_graph");

  auto s0 = Symbol(128);
  auto z0 = graph.CreateAxis("z0", s0);

  Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id};
  *load0.y.axis = {z0.id};
  *load0.y.repeats = {s0};
  *load0.y.strides = {One};

  Load load1("load1");
  load1.x = data0.y;
  load1.attr.sched.axis = {z0.id};
  *load1.y.axis = {z0.id};
  *load1.y.repeats = {s0};
  *load1.y.strides = {One};

  Pow pow0("pow0");
  pow0.x1 = load0.y;
  pow0.x2 = load1.y;
  pow0.attr.sched.axis = {z0.id};
  *pow0.y.axis = {z0.id};
  *pow0.y.repeats = {s0};
  *pow0.y.strides = {One};

  Abs abs0("abs0");
  abs0.x = pow0.y;
  abs0.attr.sched.axis = {z0.id};
  *abs0.y.axis = {z0.id};
  *abs0.y.repeats = {s0};
  *abs0.y.strides = {One};

  Add add("add0");
  add.x1 = pow0.y;
  add.x2 = abs0.y;
  add.attr.sched.axis = {z0.id};
  *add.y.axis = {z0.id};
  *add.y.repeats = {s0};
  *add.y.strides = {One};

  Sigmoid sig("sigmoid");
  sig.x = add.y;
  sig.attr.sched.axis = {z0.id};
  *sig.y.axis = {z0.id};
  *sig.y.repeats = {s0};
  *sig.y.strides = {One};

  Store store("store");
  store.x = sig.y;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

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