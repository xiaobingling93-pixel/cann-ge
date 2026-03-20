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

void GatherAbsStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1");
  graph.AddNode(x1);
  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.repeats = {s0};
  *(x1.y.axis) = {z0.id};

  Data x2("x2");
  graph.AddNode(x2);
  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = ge::DT_INT64;
  *x2.y.repeats = {s1, s2};
  *(x2.y.axis) = {z1.id, z2.id};

  Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.ir_attr.SetAxis(0);
  gather.ir_attr.SetNegative_index_support(false);
  gather.y.dtype = ge::DT_FLOAT;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};
  gather.attr.tmp_buffers = {{{ge::Symbol(81920), -1}, MemAttr(), 0}};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = gather.y;
  abs.attr.sched.axis = {z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.repeats = {s1, s2};
  *abs.y.strides = {s2, One};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {s2, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z1.id, z2.id};
}

void GatherAbsStore_AfterInferOutput(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = ge::DT_FLOAT;
  gather->attr.api.compute_type = ComputeType::kComputeGather;

  auto abs = graph.FindNode("abs");
  abs->outputs[0].attr.dtype =(ge::DataType)gather->outputs[0].attr.dtype;
  abs->outputs[0].attr.axis = gather->outputs[0].attr.axis;
  abs->outputs[0].attr.repeats = gather->outputs[0].attr.repeats;
  abs->outputs[0].attr.strides = gather->outputs[0].attr.strides;
  abs->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = (ge::DataType)abs->outputs[0].attr.dtype;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void GatherAbsStore_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;
  auto x2 = graph.FindNode("x2");
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->attr.api.type = ApiType::kAPITypeCompute;
  gather->attr.api.unit = ComputeUnit::kUnitVector;

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

void GatherAbsStore_AfterScheduler_z1z2_splitTo_z1z2TBz1z2Tbz1z2t(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;

  
  std::vector<AxisId> axes{z1, z2};
  auto z1z2 = graph.MergeAxis(axes);
  auto [z1z2T, z1z2t] = graph.TileSplit(z1z2->id);
  auto [z1z2TB, z1z2Tb] = graph.BlockSplit(z1z2T->id);
  vector<AxisId> vectorized_axis{z1z2t->id};
  vector<AxisId> axis{z1z2TB->id, z1z2Tb->id, z1z2t->id};
  vector<ge::Expression> axis_repeats{z1z2TB->size,z1z2Tb->size,z1z2t->size};
  vector<ge::Expression> axis_strides{z1z2Tb->size * z1z2t->size, z1z2t->size, One};
  vector<ge::Expression> vectorized_strides{One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);

  all_axis = graph.GetAllAxis();
  auto m_axis = all_axis[z1z2->id];

  // ApplySplit on load, abs, store
  auto gather = graph.FindNode("gather");
  graph.ApplySchedAxisMerge(gather, z1z2->id, m_axis->from);
  graph.ApplySplit(gather, z1z2T->id, z1z2t->id);
  graph.ApplySplit(gather, z1z2TB->id, z1z2Tb->id);
  gather->attr.sched.loop_axis = z1z2Tb->id;
  gather->outputs[0].attr.axis = axis;
  gather->outputs[0].attr.repeats = axis_repeats;
  gather->outputs[0].attr.strides = axis_strides;
  gather->outputs[0].attr.vectorized_axis = vectorized_axis;
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto abs = graph.FindNode("abs");
  graph.ApplySchedAxisMerge(abs, z1z2->id, m_axis->from);
  graph.ApplySplit(abs, z1z2T->id, z1z2t->id);
  graph.ApplySplit(abs, z1z2TB->id, z1z2Tb->id);
  abs->attr.sched.loop_axis = z1z2Tb->id;
  abs->outputs[0].attr.axis = axis;
  abs->outputs[0].attr.repeats = axis_repeats;
  abs->outputs[0].attr.strides = axis_strides;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySchedAxisMerge(store, z1z2->id, m_axis->from);
  graph.ApplySplit(store, z1z2T->id, z1z2t->id);
  graph.ApplySplit(store, z1z2TB->id, z1z2Tb->id);
  store->attr.sched.loop_axis = z1z2Tb->id;
  store->outputs[0].attr.axis = axis;
  store->outputs[0].attr.repeats = axis_repeats;
  store->outputs[0].attr.strides = axis_strides;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void GatherAbsStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  gather->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  gather->outputs[0].attr.buf.id = 0;
  gather->outputs[0].attr.que.id = ge::kIdNone;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = ge::kIdNone;
  gather->outputs[0].attr.que.buf_num = ge::kIdNone;
  gather->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->outputs[0].attr.mem.tensor_id = 3;
  abs->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  abs->outputs[0].attr.mem.position = Position::kPositionVecOut;
  abs->outputs[0].attr.buf.id = ge::kIdNone;
  abs->outputs[0].attr.que.id = 0;
  abs->outputs[0].attr.mem.reuse_id = 1;
  abs->outputs[0].attr.que.depth = 2;
  abs->outputs[0].attr.que.buf_num = 2;
  abs->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  abs->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}