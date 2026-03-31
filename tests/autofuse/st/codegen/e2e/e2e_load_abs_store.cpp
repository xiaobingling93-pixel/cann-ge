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
#include "ascir_ops_utils.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadAbsStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  *(x.y.axis) = {z0.id, z1.id, z2.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1*s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1*s2, s2, One};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1*s2, s2, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadAbsStore_BeforeAutofuse_DiscreteStore(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  *(x.y.axis) = {z0.id, z1.id, z2.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1*s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1*s2, s2, One};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1*(s2+s2), (s2+s2), One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadAbsStore_BeforeAutofuse_DiscreteStoreMergeAxis(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  x.y.dtype = ge::DT_FLOAT16;
  *(x.y.axis) = {z0.id, z1.id, z2.id, z3.id, z4.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.repeats = {s0, s1, s2, s3, s4};
  *load.y.strides = {s1*s2*s3*s4, s2*s3*s4, s3*s4, s4, One};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs.y.repeats = {s0, s1, s2, s3, s4};
  *abs.y.strides = {s1*s2*s3*s4, s2*s3*s4, s3*s4, s4, One};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1*s2*(s3*s4+s3*s4), s2*(s3*s4+s3*s4), (s3*s4+s3*s4), s4, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
}

void LoadAbsStore_BeforeAutofuse_z0_SplitTo_z0TBz0Tbz0t(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = ge::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
}

void LoadAbsStore_BeforeAutofuse_StoreScalar(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  *(x.y.axis) = {z0.id, z1.id, z2.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {Zero, Zero, Zero};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {Zero, Zero, Zero};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {Zero, Zero, Zero};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadAbsStore_BeforeAutofuse_StoreEmptyTensor(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", Zero);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.y.dtype = ge::DT_FLOAT16;
  *(x.y.axis) = {z0.id, z1.id, z2.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {Zero, Zero, Zero};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {Zero, Zero, Zero};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {Zero, Zero, Zero};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
}

void LoadAbsStore_AfterInferOutput(ge::AscGraph &graph) {
  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->attr.api.compute_type = ComputeType::kComputeLoad;

  auto abs = graph.FindNode("abs");
  abs->outputs[0].attr.dtype =(ge::DataType)load->outputs[0].attr.dtype;
  abs->outputs[0].attr.axis = load->outputs[0].attr.axis;
  abs->outputs[0].attr.repeats = load->outputs[0].attr.repeats;
  abs->outputs[0].attr.strides = load->outputs[0].attr.strides;
  abs->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = (ge::DataType)abs->outputs[0].attr.dtype;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadAbsStore_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x = graph.FindNode("x");
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto abs = graph.FindNode("abs");
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadAbsStore_AfterScheduler(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T->id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0B, z0b] = graph.BlockSplit(block_axis->id);
  vector<AxisId> vectorized_axis{z1t->id, z2};
  vector<ge::Expression> vectorized_strides{Zero, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  // ApplySplit on load, abs, store
  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load, block_axis->id);
  graph.ApplySplit(load, z0B->id, z0b->id);
  load->attr.sched.loop_axis = z0b->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySplit(abs, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(abs, block_axis->id);
  graph.ApplySplit(abs, z0B->id, z0b->id);
  abs->attr.sched.loop_axis = z0b->id;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(store, block_axis->id);
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadAbsStore_AfterScheduler_z0z1z2_splitTo_z0z1TBz0z1Tbz1tz2(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z1T, z1t] = graph.TileSplit(z1);
  std::vector<AxisId> axes{z0, z1T->id};
  auto block_axis = graph.MergeAxis(axes);
  auto [z0B, z0b] = graph.BlockSplit(block_axis->id);
  vector<AxisId> vectorized_axis{z1t->id, z2};
  vector<ge::Expression> vectorized_strides{One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  all_axis = graph.GetAllAxis();
  auto m_axis = all_axis[block_axis->id];

  // ApplySplit on load, abs, store
  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(load, block_axis->id, m_axis->from);
  graph.ApplySplit(load, z0B->id, z0b->id);
  load->attr.sched.loop_axis = z0b->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySplit(abs, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(abs, block_axis->id, m_axis->from);
  graph.ApplySplit(abs, z0B->id, z0b->id);
  abs->attr.sched.loop_axis = z0b->id;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplySchedAxisMerge(store, block_axis->id, m_axis->from);
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadAbsStore_AfterScheduler_z0_SplitTo_z0TBz0Tbz0t(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  vector<AxisId> vectorized_axis{z0t->id, z1};
  vector<ge::Expression> vectorized_strides{One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  // ApplySplit on load, abs, store
  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z0T->id, z0t->id);
  graph.ApplySplit(load, z0TB->id, z0Tb->id);
  load->attr.sched.loop_axis = z0Tb->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySplit(abs, z0T->id, z0t->id);
  graph.ApplySplit(abs, z0TB->id, z0Tb->id);
  abs->attr.sched.loop_axis = z0Tb->id;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0T->id, z0t->id);
  graph.ApplySplit(store, z0TB->id, z0Tb->id);
  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadAbsStore_AfterScheduler_z0_SplitTo_z0Bz0b(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z0B, z0b] = graph.BlockSplit(z0);
  vector<AxisId> vectorized_axis{z1, z2};
  vector<ge::Expression> vectorized_strides{One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  // ApplySplit on load, abs, store
  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z0B->id, z0b->id);
  load->attr.sched.loop_axis = z0b->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySplit(abs, z0B->id, z0b->id);
  abs->attr.sched.loop_axis = z0b->id;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadAbsStore_AfterScheduler_store_scalar(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  auto [z0B, z0b] = graph.BlockSplit(z0);
  vector<AxisId> vectorized_axis{z1, z2};
  vector<ge::Expression> vectorized_strides{Zero, Zero};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);

  // ApplySplit on load, abs, store
  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z0B->id, z0b->id);
  load->attr.sched.loop_axis = z0b->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySplit(abs, z0B->id, z0b->id);
  abs->attr.sched.loop_axis = z0b->id;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadAbsStore_AfterScheduler_DiscreteStoreMergeAxis(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;
  auto z3 = all_axis[3]->id;
  auto z4 = all_axis[4]->id;

  auto [z0B, z0b] = graph.BlockSplit(z0);
  vector<AxisId> vectorized_axis{z1, z2, z3, z4};
  vector<ge::Expression> vectorized_strides{One, One, One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  vectorized_strides[2] =  ge::sym::Align(graph.FindAxis(vectorized_axis[3])->size, 32 / size);
  vectorized_strides[1] = graph.FindAxis(vectorized_axis[2])->size * vectorized_strides[2];
  vectorized_strides[0] = graph.FindAxis(vectorized_axis[1])->size * vectorized_strides[1];

  // ApplySplit on load, abs, store
  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z0B->id, z0b->id);
  load->attr.sched.loop_axis = z0b->id;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySplit(abs, z0B->id, z0b->id);
  abs->attr.sched.loop_axis = z0b->id;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0B->id, z0b->id);
  store->attr.sched.loop_axis = z0b->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadAbsStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  auto x = graph.FindNode("x");
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = Position::kPositionGM;
  x->outputs[0].attr.buf.id = ge::kIdNone;
  x->outputs[0].attr.que.id = ge::kIdNone;
  x->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = ge::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->outputs[0].attr.mem.tensor_id = 2;
  abs->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  abs->outputs[0].attr.mem.position = Position::kPositionVecOut;
  abs->outputs[0].attr.buf.id = ge::kIdNone;
  abs->outputs[0].attr.que.id = 1;
  abs->outputs[0].attr.mem.reuse_id = 1;
  abs->outputs[0].attr.que.depth = 2;
  abs->outputs[0].attr.que.buf_num = 2;
  abs->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  abs->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}
