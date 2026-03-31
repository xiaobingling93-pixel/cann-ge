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

void LoadScalarAbsBrcStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x("x");
  graph.AddNode(x);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = ge::DT_FLOAT;
  *(x.y.axis) = {z0.id, z1.id};

  Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, Zero};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, One};
  *abs.y.strides = {One, Zero};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}};

  ge::ascir_op::Broadcast broadcast("broadcast");
  graph.AddNode(broadcast);
  broadcast.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};
  broadcast.x = abs.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Store store("store");
  graph.AddNode(store);
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
}

void LoadScalarAbsBrcStore_AfterAutofuse(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;

  auto x = graph.FindNode("x");
  x->attr.api.compute_type = ComputeType::kComputeInvalid; // ComputeType::COMPUTE_DATA;
  x->attr.api.type = ApiType::kAPITypeBuffer;
  x->attr.api.unit = ComputeUnit::kUnitNone;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = Position::kPositionGM;
  x->outputs[0].attr.buf.id = ge::kIdNone;
  x->outputs[0].attr.que.id = ge::kIdNone;
  x->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  vector<AxisId> vectorized_axis{z1};
  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->attr.api.compute_type = ComputeType::kComputeLoad;
  load->attr.api.type = ApiType::kAPITypeCompute;
  load->attr.api.unit = ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0;
  load->outputs[0].attr.vectorized_axis = vectorized_axis;
  load->outputs[0].attr.vectorized_strides = {Zero};
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
  abs->outputs[0].attr.dtype = ge::DT_FLOAT;
  abs->attr.api.compute_type = ComputeType::kComputeElewise;
  abs->attr.api.type = ApiType::kAPITypeCompute;
  abs->attr.api.unit = ComputeUnit::kUnitVector;
  abs->attr.sched.loop_axis = z0;
  abs->outputs[0].attr.vectorized_axis = vectorized_axis;
  abs->outputs[0].attr.vectorized_strides = {Zero};
  abs->outputs[0].attr.mem.tensor_id = 2;
  abs->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  abs->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  abs->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  abs->outputs[0].attr.buf.id = 0;
  abs->outputs[0].attr.que.id = ge::kIdNone;
  abs->outputs[0].attr.mem.reuse_id = 1;
  abs->outputs[0].attr.que.depth = 2;
  abs->outputs[0].attr.que.buf_num = 2;
  abs->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  abs->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->outputs[0].attr.dtype = ge::DT_FLOAT;
  broadcast->attr.api.compute_type = ComputeType::kComputeElewise;
  broadcast->attr.api.type = ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0;
  broadcast->outputs[0].attr.vectorized_axis = vectorized_axis;
  broadcast->outputs[0].attr.vectorized_strides = {One};
  broadcast->outputs[0].attr.mem.tensor_id = 3;
  broadcast->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  broadcast->outputs[0].attr.mem.position = Position::kPositionVecOut;
  broadcast->outputs[0].attr.buf.id = ge::kIdNone;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.mem.reuse_id = 2;
  broadcast->outputs[0].attr.que.depth = 2;
  broadcast->outputs[0].attr.que.buf_num = 2;
  broadcast->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  broadcast->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = ge::DT_FLOAT;
  store->attr.api.compute_type = ComputeType::kComputeStore;
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.mem.tensor_id = 4;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}
