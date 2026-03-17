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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "reg_store_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace codegen;

TEST(CodegenKernel, StoreRegApiCall_TwoStoreOneOutput) {
  ge::AscGraph graph("test_graph");

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(3);
  auto s1_0 = ge::Symbol(1);
  auto s1_1 = graph.CreateSizeVar("z0_t_size");
  auto s2 = ge::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Store store_op("store");
  ge::ascir_op::Store store_op_1("store_1");
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(store_op_1);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1, s2, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1_0, s2};
  *store_op.y.strides = {s2, s2, One};

  store_op_1.x = load_op.y;
  store_op_1.ir_attr.SetOffset(ge::Symbol(1));
  *store_op_1.y.axis = {z0.id, z1.id, z2.id};
  *store_op_1.y.repeats = {s0, s1_1, s2};
  *store_op_1.y.strides = {s1_1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(16), One};
  store->outputs[0].attr.dtype = ge::DT_INT16;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store_1 = graph.FindNode("store_1");
  store_1->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store_1->attr.api.type = ge::ApiType::kAPITypeCompute;
  store_1->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store_1->attr.sched.loop_axis = z0.id;
  store_1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  store_1->outputs[0].attr.vectorized_strides = {ge::Symbol(16), One};
  store_1->outputs[0].attr.dtype = ge::DT_INT16;
  store_1->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store_1->outputs[0].attr.mem.tensor_id = 1;
  store_1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store_1->outputs[0].attr.que.id = 2;
  store_1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(store_1->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::StoreRegApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{
                "DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(local_1[0 + 0], local_0, 1, 1, 16 - 1, 0);\n"});

  codegen::StoreRegApiCall call_1("Store");
  EXPECT_EQ(call_1.Init(store_1), 0);
  call_1.inputs.push_back(&x1);
  call_1.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(local_1[0 + 1], local_0, "
                                "z0_t_size, 1, 16 - 1, 0);\n"});
}

TEST(CodegenKernel, StoreRegApiCall_NeetMte3SyncMte2) {
  ge::AscGraph graph("test_graph");

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(3);
  auto s1_0 = ge::Symbol(1);
  auto s1_1 = graph.CreateSizeVar("z0_t_size");
  auto s2 = ge::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Store store_op("store");
  ge::ascir_op::Abs abs_op("abs_1");
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(abs_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1, s2, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1_0, s2};
  *store_op.y.strides = {s2, s2, One};

  abs_op.x = load_op.y;
  *abs_op.y.axis = {z0.id, z1.id, z2.id};
  *abs_op.y.repeats = {s0, s1_1, s2};
  *abs_op.y.strides = {s1_1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(16), One};
  store->outputs[0].attr.dtype = ge::DT_INT16;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto abs_1 = graph.FindNode("abs_1");
  abs_1->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs_1->attr.api.type = ge::ApiType::kAPITypeCompute;
  abs_1->attr.api.unit = ge::ComputeUnit::kUnitVector;
  abs_1->attr.sched.loop_axis = z0.id;
  abs_1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  abs_1->outputs[0].attr.vectorized_strides = {ge::Symbol(16), One};
  abs_1->outputs[0].attr.dtype = ge::DT_INT16;
  abs_1->outputs[0].attr.mem.position = ge::Position::kPositionVecCalc;
  abs_1->outputs[0].attr.mem.tensor_id = 1;
  abs_1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  abs_1->outputs[0].attr.que.id = 2;
  abs_1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Kernel kernel("test");
  kernel.tpipe.CollectQues(graph);
  EXPECT_EQ(kernel.tpipe.AddTensor(load->outputs[0]), 0);
  auto tensor_ptr = kernel.tpipe.GetTensor(load->outputs[0].attr.mem.tensor_id);
  auto &t = *tensor_ptr;
  t.is_ub_scalar = true;
  EXPECT_EQ(kernel.ParseOptimizeInfo(load, load->outputs[0]), 0);
  EXPECT_EQ(kernel.tpipe.AddTensor(abs_1->outputs[0]), 0);

  kernel.tiler.AddAxis(z0);
  kernel.tiler.AddAxis(z1);
  kernel.tiler.AddAxis(z2);
  kernel.tiler.AddSizeVar(ge::SizeVar(s0));
  kernel.tiler.AddSizeVar(ge::SizeVar(s1));
  kernel.tiler.AddSizeVar(ge::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  // 输入是is_load_link_store_vec
  std::string result;
  codegen::StoreRegApiCall call_1("Store");
  EXPECT_EQ(call_1.Init(store), 0);
  call_1.inputs.push_back(&x1);
  x1.reads.push_back(&call_1);
  call_1.Generate(kernel.tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{"DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(local_1[0 + 0], local_0, 1, 1, 16 - 1, 0);\n"
                  "auto local_0_e_mte3_2_mte2_t_0 = tpipe.AllocEventID<HardEvent::MTE3_MTE2>();\n"
                  "TQueSync<PIPE_MTE3, PIPE_MTE2> local_0_s_mte3_2_mte2_t_0;\n"
                  "local_0_s_mte3_2_mte2_t_0.SetFlag(local_0_e_mte3_2_mte2_t_0);\n"
                  "local_0_s_mte3_2_mte2_t_0.WaitFlag(local_0_e_mte3_2_mte2_t_0);\n"
                  "tpipe.ReleaseEventID<HardEvent::MTE3_MTE2>(local_0_e_mte3_2_mte2_t_0);\n"});
}

TEST(CodegenKernel, StoreRegApiCall_ThreeDimStore) {
  ge::AscGraph graph("test_graph");

  ge::Expression Two = ge::Symbol(2);
  ge::Expression Three = ge::Symbol(3);
  ge::Expression Four = ge::Symbol(4);

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(4);
  auto s3 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(8), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_INT16;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddSizeVar(ge::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::StoreRegApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(
      result,
      std::string{
          "DataCopyPadExtend<float, AscendC::PaddingMode::Compact>(local_1[0 + 0], local_0[0], 4, 2, 2 - 2, 2, "
          "{static_cast<uint32_t>(8), static_cast<uint32_t>(1), static_cast<uint64_t>(8 * 4), static_cast<uint64_t>(24 "
          "* 2), static_cast<uint64_t>(0 * 4), static_cast<uint64_t>(0 * 2)});\n"});
}

TEST(CodegenKernel, StoreRegApiCall_FiveDimStore) {
  ge::AscGraph graph("test_graph");

  ge::Expression Two = ge::Symbol(2);
  ge::Expression Three = ge::Symbol(3);
  ge::Expression Four = ge::Symbol(4);
  ge::Expression Five = ge::Symbol(5);
  ge::Expression Six = ge::Symbol(6);

  auto s0 = ge::Symbol(16);
  auto s1 = ge::Symbol(8);
  auto s2 = ge::Symbol(4);
  auto s3 = ge::Symbol(2);
  auto s4 = ge::Symbol(2);
  auto s5 = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);

  Data x_op("x", graph);
  Load load_op("load");
  ge::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *load_op.y.repeats = {s0, s1, s2, s3, s4, s5};
  *load_op.y.strides = {
      s1 * s2 * s3 * s4 * s5 * Six, s2 * s3 * s4 * s5 * Five, s3 * s4 * s5 * Four, s4 * s5 * Three, s5 * Two, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(ge::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *store_op.y.repeats = {s0, s1, s2, s3, s4, s5};
  *store_op.y.strides = {
      s1 * s2 * s3 * s4 * s5 * Six, s2 * s3 * s4 * s5 * Five, s3 * s4 * s5 * Four, s4 * s5 * Three, s5 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  load->attr.api.type = ge::ApiType::kAPITypeCompute;
  load->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id, z5.id};
  load->outputs[0].attr.vectorized_strides = {ge::Symbol(32), ge::Symbol(8), ge::Symbol(4), ge::Symbol(2), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id, z4.id, z5.id};
  store->outputs[0].attr.vectorized_strides = {ge::Symbol(32), ge::Symbol(8), ge::Symbol(4), ge::Symbol(2), One};
  store->outputs[0].attr.dtype = ge::DT_INT16;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddAxis(z5);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  tiler.AddSizeVar(ge::SizeVar(s2));
  tiler.AddSizeVar(ge::SizeVar(s3));
  tiler.AddSizeVar(ge::SizeVar(s4));
  tiler.AddSizeVar(ge::SizeVar(s5));
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::StoreRegApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<ge::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"for(int outer_for_0 = 0; outer_for_0 < 8; outer_for_0++) {\nDataCopyPadExtend<float, "
                        "AscendC::PaddingMode::Compact>(local_1[0 + 0 + "
                        "outer_for_0 * 160], local_0[outer_for_0 * 32], 2, 2, 2 - 2, 2, {static_cast<uint32_t>(2), "
                        "static_cast<uint32_t>(4), static_cast<uint64_t>(4 * 4), static_cast<uint64_t>(12 * 2), "
                        "static_cast<uint64_t>(8 * 4), "
                        "static_cast<uint64_t>(32 * 2)});\n\n}\n"});
}