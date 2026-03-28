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
#include "ascir_ops_utils.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadGatherAbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");
  auto s6 = graph.CreateSizeVar("s6");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);

  Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;
  x1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1.y.repeats = {s0, s1, s2, s3, s4};
  *x1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = ge::DT_INT32;
  x2.attr.sched.axis = {z5.id, z6.id};
  *x2.y.axis = {z5.id, z6.id};
  *x2.y.repeats = {s5, s6};
  *x2.y.strides = {s6, One};

  Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.ir_attr.SetAxis(gather_axis);
  gather.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather.y.repeats = {s0, s1, s5, s6, s3, s4};
  *gather.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};

  Abs abs("abs");
  graph.AddNode(abs);
  abs.x = gather.y;
  abs.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *abs.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *abs.y.repeats = {s0, s1, s5, s6, s3, s4};
  *abs.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};  

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s5, s6, s3, s4};
  *store.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
}

void LoadGather_BT_T_AbsStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = data_type;
  gather->attr.api.compute_type = ComputeType::kComputeGather;
  gather->attr.api.type = ApiType::kAPITypeCompute;
  gather->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = ComputeType::kComputeElewise;
  abs->outputs[0].attr.dtype = data_type;
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0 = axes[0]->id;
  auto z1 = axes[1]->id;
  auto z2 = axes[2]->id;
  auto z3 = axes[3]->id;
  auto z4 = axes[4]->id;
  auto z5 = axes[5]->id;
  auto z6 = axes[6]->id;

  auto z0z1z5z6 = graph.MergeAxis({z0, z1, z5, z6});
  auto [z0z1z5z6T, z0z1z5z6t] = graph.TileSplit(z0z1z5z6->id);
  auto [z0z1z5z6TB, z0z1z5z6Tb] = graph.BlockSplit(z0z1z5z6T->id);

  auto z3z4 = graph.MergeAxis({z3, z4});
  auto [z3z4T, z3z4t] = graph.TileSplit(z3z4->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z5z6->id);
    graph.ApplySplit(node, z0z1z5z6T->id, z0z1z5z6t->id);
    graph.ApplySplit(node, z0z1z5z6TB->id, z0z1z5z6Tb->id);

    graph.ApplyMerge(node, z3z4->id);
    graph.ApplySplit(node, z3z4T->id, z3z4t->id);

    graph.ApplyReorder(node, {z0z1z5z6TB->id, z0z1z5z6Tb->id, z3z4T->id, z0z1z5z6t->id, z3z4t->id});
  }

  auto size = ge::GetSizeByDataType(data_type);
  vector<ge::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(z3z4t->id)->size, 32 / size);

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z3z4T->id;
  gather->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  abs->attr.sched.loop_axis = z3z4T->id;
  abs->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z3z4T->id;
  store->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = ge::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  abs->outputs[0].attr.mem.tensor_id = 3;
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

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware =  MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void LoadGather_T_BT_AbsStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = data_type;
  gather->attr.api.compute_type = ComputeType::kComputeGather;
  gather->attr.api.type = ApiType::kAPITypeCompute;
  gather->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = ComputeType::kComputeElewise;
  abs->outputs[0].attr.dtype = data_type;
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0 = axes[0]->id;
  auto z1 = axes[1]->id;
  auto z2 = axes[2]->id;
  auto z3 = axes[3]->id;
  auto z4 = axes[4]->id;
  auto z5 = axes[5]->id;
  auto z6 = axes[6]->id;

  auto z0z1z5z6 = graph.MergeAxis({z0, z1, z5, z6});
  auto [z0z1z5z6T, z0z1z5z6t] = graph.TileSplit(z0z1z5z6->id);

  auto z3z4 = graph.MergeAxis({z3, z4});
  auto [z3z4T, z3z4t] = graph.TileSplit(z3z4->id);
  auto [z3z4TB, z3z4Tb] = graph.BlockSplit(z3z4T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z5z6->id);
    graph.ApplySplit(node, z0z1z5z6T->id, z0z1z5z6t->id);

    graph.ApplyMerge(node, z3z4->id);
    graph.ApplySplit(node, z3z4T->id, z3z4t->id);
    graph.ApplySplit(node, z3z4TB->id, z3z4Tb->id);

    graph.ApplyReorder(node, {z0z1z5z6T->id, z3z4TB->id, z3z4Tb->id, z0z1z5z6t->id, z3z4t->id});
  }

  auto size = ge::GetSizeByDataType(data_type);
  vector<ge::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(z3z4t->id)->size, 32 / size);

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z3z4Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  abs->attr.sched.loop_axis = z3z4Tb->id;
  abs->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z3z4Tb->id;
  store->outputs[0].attr.vectorized_axis = {z0z1z5z6t->id, z3z4t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = ge::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  abs->outputs[0].attr.mem.tensor_id = 3;
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

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware =  MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void LoadGather_B_T_AbsStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = data_type;
  gather->attr.api.compute_type = ComputeType::kComputeGather;
  gather->attr.api.type = ApiType::kAPITypeCompute;
  gather->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = ComputeType::kComputeElewise;
  abs->outputs[0].attr.dtype = data_type;
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0 = axes[0]->id;
  auto z1 = axes[1]->id;
  auto z2 = axes[2]->id;
  auto z3 = axes[3]->id;
  auto z4 = axes[4]->id;
  auto z5 = axes[5]->id;
  auto z6 = axes[6]->id;

  auto z0z1z5z6 = graph.MergeAxis({z0, z1, z5, z6});
  auto [z0z1z5z6B, z0z1z5z6b] = graph.BlockSplit(z0z1z5z6->id);

  auto z3z4 = graph.MergeAxis({z3, z4});
  auto [z3z4T, z3z4t] = graph.TileSplit(z3z4->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z5z6->id);
    graph.ApplySplit(node, z0z1z5z6B->id, z0z1z5z6b->id);

    graph.ApplyMerge(node, z3z4->id);
    graph.ApplySplit(node, z3z4T->id, z3z4t->id);

    graph.ApplyReorder(node, {z0z1z5z6B->id, z0z1z5z6b->id, z3z4T->id, z3z4t->id});
  }

  auto size = ge::GetSizeByDataType(data_type);
  vector<ge::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z3z4T->id;
  gather->outputs[0].attr.vectorized_axis = {z3z4t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  abs->attr.sched.loop_axis = z3z4T->id;
  abs->outputs[0].attr.vectorized_axis = {z3z4t->id};
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z3z4T->id;
  store->outputs[0].attr.vectorized_axis = {z3z4t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = ge::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  abs->outputs[0].attr.mem.tensor_id = 3;
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

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware =  MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void LoadGather_BT_AbsStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = data_type;
  gather->attr.api.compute_type = ComputeType::kComputeGather;
  gather->attr.api.type = ApiType::kAPITypeCompute;
  gather->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = ComputeType::kComputeElewise;
  abs->outputs[0].attr.dtype = data_type;
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0 = axes[0]->id;
  auto z1 = axes[1]->id;
  auto z2 = axes[2]->id;
  auto z3 = axes[3]->id;
  auto z4 = axes[4]->id;
  auto z5 = axes[5]->id;
  auto z6 = axes[6]->id;

  auto z0z1z5z6 = graph.MergeAxis({z0, z1, z5, z6});

  auto z3z4 = graph.MergeAxis({z3, z4});
  auto [z3z4T, z3z4t] = graph.TileSplit(z3z4->id);
  auto [z3z4TB, z3z4Tb] = graph.BlockSplit(z3z4T->id);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z0z1z5z6->id);

    graph.ApplyMerge(node, z3z4->id);
    graph.ApplySplit(node, z3z4T->id, z3z4t->id);
    graph.ApplySplit(node, z3z4TB->id, z3z4Tb->id);

    graph.ApplyReorder(node, {z0z1z5z6->id, z3z4TB->id, z3z4Tb->id, z3z4t->id});
  }

  auto size = ge::GetSizeByDataType(data_type);
  vector<ge::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z3z4Tb->id;
  gather->outputs[0].attr.vectorized_axis = {z3z4t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  abs->attr.sched.loop_axis = z3z4Tb->id;
  abs->outputs[0].attr.vectorized_axis = {z3z4t->id};
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z3z4Tb->id;
  store->outputs[0].attr.vectorized_axis = {z3z4t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = ge::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  abs->outputs[0].attr.mem.tensor_id = 3;
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

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware =  MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void LoadGather_FirstAxis_B_T_AbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = ge::DT_INT32;
  x2.attr.sched.axis = {z2.id, z3.id};
  *x2.y.axis = {z2.id, z3.id};
  *x2.y.repeats = {s2, s3};
  *x2.y.strides = {s3, One};

  Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.ir_attr.SetAxis(gather_axis);
  gather.attr.sched.axis = {z2.id, z3.id, z1.id};
  *gather.y.axis = {z2.id, z3.id, z1.id};
  *gather.y.repeats = {s2, s3, s1};
  *gather.y.strides = {s3 * s1, s1, One};

  Abs abs("abs");
  graph.AddNode(abs);
  abs.x = gather.y;
  abs.attr.sched.axis = {z2.id, z3.id, z1.id};
  *abs.y.axis = {z2.id, z3.id, z1.id};
  *abs.y.repeats = {s2, s3, s1};
  *abs.y.strides = {s3 * s1, s1, One};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z2.id, z3.id, z1.id};
  *store.y.axis = {z2.id, z3.id, z1.id};
  *store.y.repeats = {s2, s3, s1};
  *store.y.strides = {s3 * s1, s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
}

void LoadGather_FirstAxis_B_T_AbsStore_AfterAutofuse(ge::AscGraph& graph, ge::DataType data_type) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto gather = graph.FindNode("gather");
  gather->outputs[0].attr.dtype = data_type;
  gather->attr.api.compute_type = ComputeType::kComputeGather;
  gather->attr.api.type = ApiType::kAPITypeCompute;
  gather->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = ComputeType::kComputeElewise;
  abs->outputs[0].attr.dtype = data_type;
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = data_type;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;

  // Scheduler
  auto axes = graph.GetAllAxis();
  auto z0 = axes[0]->id;
  auto z1 = axes[1]->id;
  auto z2 = axes[2]->id;
  auto z3 = axes[3]->id;

  auto z2z3 = graph.MergeAxis({z2, z3});
  auto [z2z3B, z2z3b] = graph.BlockSplit(z2z3->id);

  auto [z1T, z1t] = graph.TileSplit(z1);

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplyMerge(node, z2z3->id);
    graph.ApplySplit(node, z2z3B->id, z2z3b->id);

    graph.ApplySplit(node, z1T->id, z1t->id);

    graph.ApplyReorder(node, {z2z3B->id, z2z3b->id, z1T->id, z1t->id});
  }

  auto size = ge::GetSizeByDataType(data_type);
  vector<ge::Expression> vectorized_strides{One};

  // Vectorized/Loop axis
  gather->attr.sched.loop_axis = z1T->id;
  gather->outputs[0].attr.vectorized_axis = {z1t->id};
  gather->outputs[0].attr.vectorized_strides = vectorized_strides;

  abs->attr.sched.loop_axis = z1T->id;
  abs->outputs[0].attr.vectorized_axis = {z1t->id};
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;

  store->attr.sched.loop_axis = z1T->id;
  store->outputs[0].attr.vectorized_axis = {z1t->id};
  store->outputs[0].attr.vectorized_strides = vectorized_strides;

  // Que/Buf alloc
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  gather->outputs[0].attr.mem.tensor_id = 2;
  gather->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  gather->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  gather->outputs[0].attr.mem.position = Position::kPositionVecIn;
  gather->outputs[0].attr.buf.id = ge::kIdNone;
  gather->outputs[0].attr.que.id = 0;
  gather->outputs[0].attr.mem.reuse_id = 0;
  gather->outputs[0].attr.que.depth = 2;
  gather->outputs[0].attr.que.buf_num = 2;
  gather->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  gather->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  abs->outputs[0].attr.mem.tensor_id = 3;
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

  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware =  MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}
