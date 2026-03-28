/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "ascendc_ir.h"
#include "schedule_result.h"
#include "codegen.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "autofuse_config/auto_fuse_config.h"

using namespace ge::ascir_op;
using namespace ge::ops;

std::vector<std::string> splitString(const std::string& input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
      result.push_back(token);
  }

  return result;
}

class ScheduleMultiGoupTest : public testing::Test {
};

void MultiGroupDoScheduler(ascir::FusedScheduledResult &fused_schedule_result, std::vector<ge::AscGraph> &impl_graphs_group) {
  // Scheduler
  // impl_graph1
  auto all_axis1 = impl_graphs_group[0].GetAllAxis();
  auto all_sizevar1 = impl_graphs_group[0].GetAllSizeVar();
  vector<ge::Expression> vectorized_strides{One};
  auto data0 = impl_graphs_group[0].FindNode("data0");
  data0->attr.api.unit = ge::ComputeUnit::kUnitNone;
  data0->outputs[0].attr.mem.tensor_id = 0;
  auto &attr0 = reinterpret_cast<ge::AscDataIrAttrDef &>(*data0->attr.ir_attr);
  attr0.SetIndex(0);

  auto load0 = impl_graphs_group[0].FindNode("load0");
  load0->attr.sched.loop_axis = all_axis1[0]->id;
  load0->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load0->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load0->outputs[0].attr.vectorized_axis = {all_axis1[1]->id};
  load0->outputs[0].attr.vectorized_strides = vectorized_strides;
  load0->outputs[0].attr.mem.tensor_id = 1;
  load0->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load0->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load0->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load0->outputs[0].attr.buf.id = ge::kIdNone;
  load0->outputs[0].attr.que.id = 0;
  load0->outputs[0].attr.mem.reuse_id = 0;
  load0->outputs[0].attr.que.depth = 2;
  load0->outputs[0].attr.que.buf_num = 2;
  load0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load0->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto abs = impl_graphs_group[0].FindNode("abs");
  abs->attr.sched.loop_axis = all_axis1[0]->id;
  abs->attr.api.unit = ge::ComputeUnit::kUnitVector;
  abs->outputs[0].attr.dtype = ge::DT_FLOAT16;
  abs->outputs[0].attr.mem.tensor_id = 2;
  abs->outputs[0].attr.vectorized_axis = {all_axis1[1]->id};
  abs->outputs[0].attr.vectorized_strides = vectorized_strides;
  abs->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  abs->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  abs->outputs[0].attr.buf.id = ge::kIdNone;
  abs->outputs[0].attr.que.id = 1;
  abs->outputs[0].attr.mem.reuse_id = 1;
  abs->outputs[0].attr.que.depth = 2;
  abs->outputs[0].attr.que.buf_num = 2;
  abs->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  abs->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store0 = impl_graphs_group[0].FindNode("store0");
  store0->attr.sched.loop_axis = all_axis1[0]->id;
  store0->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store0->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store0->outputs[0].attr.mem.tensor_id = 3;
  store0->outputs[0].attr.vectorized_axis = {all_axis1[1]->id};
  store0->outputs[0].attr.vectorized_strides = vectorized_strides;
  store0->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store0->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store0->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store0->outputs[0].attr.buf.id = ge::kIdNone;
  store0->outputs[0].attr.que.id = ge::kIdNone;
  store0->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store0->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto workspace0_1 = impl_graphs_group[0].FindNode("workspace0");
  workspace0_1->attr.api.unit = ge::ComputeUnit::kUnitNone;
  workspace0_1->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace0_1->outputs[0].attr.mem.tensor_id = 3;
  workspace0_1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  workspace0_1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  workspace0_1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  // impl_graph2
  auto all_axis2 = impl_graphs_group[1].GetAllAxis();
  auto all_sizevar2 = impl_graphs_group[1].GetAllSizeVar();
  vectorized_strides = {One};

  auto data1 = impl_graphs_group[1].FindNode("data1");
  data1->attr.api.unit = ge::ComputeUnit::kUnitNone;
  data1->outputs[0].attr.mem.tensor_id = 4;
  data1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  data1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  data1->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  auto &attr1 = reinterpret_cast<ge::AscDataIrAttrDef &>(*data1->attr.ir_attr);
  attr1.SetIndex(1);

  auto workspace0_2 = impl_graphs_group[1].FindNode("workspace0");
  workspace0_2->attr.api.unit = ge::ComputeUnit::kUnitNone;
  workspace0_2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace0_2->outputs[0].attr.mem.tensor_id = 3;
  workspace0_2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  workspace0_2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  workspace0_2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto load1 = impl_graphs_group[1].FindNode("load1");
  load1->attr.sched.loop_axis = all_axis2[0]->id;
  load1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load1->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load1->outputs[0].attr.vectorized_axis = {all_axis2[1]->id};
  load1->outputs[0].attr.vectorized_strides = vectorized_strides;
  load1->outputs[0].attr.mem.tensor_id = 5;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = impl_graphs_group[1].FindNode("load2");
  load2->attr.sched.loop_axis = all_axis2[0]->id;
  load2->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.vectorized_axis = {all_axis2[1]->id};
  load2->outputs[0].attr.vectorized_strides = vectorized_strides;
  load2->outputs[0].attr.mem.tensor_id = 6;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = ge::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto add = impl_graphs_group[1].FindNode("add");
  add->attr.sched.loop_axis = all_axis2[0]->id;
  add->attr.api.unit = ge::ComputeUnit::kUnitVector;
  add->outputs[0].attr.dtype = ge::DT_FLOAT16;
  add->outputs[0].attr.mem.tensor_id = 7;
  add->outputs[0].attr.vectorized_axis = {all_axis2[1]->id};
  add->outputs[0].attr.vectorized_strides = vectorized_strides;
  add->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  add->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  add->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  add->outputs[0].attr.buf.id = ge::kIdNone;
  add->outputs[0].attr.que.id = 2;
  add->outputs[0].attr.mem.reuse_id = 1;
  add->outputs[0].attr.que.depth = 2;
  add->outputs[0].attr.que.buf_num = 2;
  add->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  add->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store1 = impl_graphs_group[1].FindNode("store1");
  store1->attr.sched.loop_axis = all_axis2[0]->id;
  store1->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
  store1->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store1->outputs[0].attr.vectorized_axis = {all_axis2[1]->id};
  store1->outputs[0].attr.vectorized_strides = vectorized_strides;
  store1->outputs[0].attr.mem.tensor_id = 8;
  store1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store1->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store1->outputs[0].attr.buf.id = ge::kIdNone;
  store1->outputs[0].attr.que.id = ge::kIdNone;
  store1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto output0 = impl_graphs_group[1].FindNode("output0");
  output0->attr.api.unit = ge::ComputeUnit::kUnitNone;
  output0->attr.api.type = ge::ApiType::kAPITypeBuffer;
  output0->attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  auto &out_attr = reinterpret_cast<ge::AscDataIrAttrDef &>(*output0->attr.ir_attr);
  out_attr.SetIndex(0);

  ascir::ScheduleGroup sch_groups1;
  ascir::ScheduleGroup sch_groups2;
  sch_groups1.impl_graphs.push_back(impl_graphs_group[0]);
  sch_groups2.impl_graphs.push_back(impl_graphs_group[1]);

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.push_back(sch_groups1);
  schedule_result.schedule_groups.push_back(sch_groups2);

  std::vector<ascir::ScheduledResult> schedule_results{schedule_result};

  fused_schedule_result.input_nodes.push_back(data0);
  fused_schedule_result.input_nodes.push_back(data1);
  fused_schedule_result.output_nodes.push_back(output0);
  fused_schedule_result.workspace_nodes.push_back(workspace0_1);
  fused_schedule_result.workspace_nodes.push_back(workspace0_2);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
}

/**
 *         Output0
 *           |
 *        AscBc2
 *        /    \
 *     data1  AscBc1
 *              |
 *            data0
 */
void ConstructMultiGroupGraph(ge::AscGraph& graph, ascir::FusedScheduledResult &fused_schedule_result) {
  ge::AscGraph impl_graph1("AscBc1");
  ge::AscGraph impl_graph2("AscBc2");

  // impl_graph1
  auto s0 = impl_graph1.CreateSizeVar("s0");
  auto s1 = impl_graph1.CreateSizeVar("s1");

  auto z0 = impl_graph1.CreateAxis("z0", s0);
  auto z1 = impl_graph1.CreateAxis("z1", s1);

  Data data0("data0");
  impl_graph1.AddNode(data0);
  data0.y.dtype = ge::DT_FLOAT16;

  Load load0("load0");
  impl_graph1.AddNode(load0);
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  impl_graph1.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  abs.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, ge::MemAttr(), 0}};
  abs.y.dtype = ge::DT_FLOAT16;

  Store store0("store0");
  impl_graph1.AddNode(store0);
  store0.x = abs.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};
  store0.y.dtype = ge::DT_FLOAT16;

  Workspace workspace0_1("workspace0");
  impl_graph1.AddNode(workspace0_1);
  workspace0_1.x = store0.y;
  workspace0_1.y.dtype = ge::DT_FLOAT16;

  // impl_graph2
  s0 = impl_graph2.CreateSizeVar("s0");
  s1 = impl_graph2.CreateSizeVar("s1");

  z0 = impl_graph2.CreateAxis("z0", s0);
  z1 = impl_graph2.CreateAxis("z1", s1);

  Data data1("data1");
  impl_graph2.AddNode(data1);
  data1.y.dtype = ge::DT_FLOAT16;

  Workspace workspace0_2("workspace0");
  impl_graph2.AddNode(workspace0_2);
  workspace0_2.y.dtype = ge::DT_FLOAT16;

  Load load1("load1");
  impl_graph2.AddNode(load1);
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};
  load1.y.dtype = ge::DT_FLOAT16;

  Load load2("load2");
  impl_graph2.AddNode(load2);
  load2.x = workspace0_2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  load2.y.dtype = ge::DT_FLOAT16;

  Add add("add");
  impl_graph2.AddNode(add);
  add.x1 = load1.y;
  add.x2 = load2.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  add.y.dtype = ge::DT_FLOAT16;

  Store store1("store1");
  impl_graph2.AddNode(store1);
  store1.x = add.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};
  store1.y.dtype = ge::DT_FLOAT16;

  Output output0("output0");
  impl_graph2.AddNode(output0);
  output0.x = store1.y;
  output0.y.dtype = ge::DT_FLOAT16;

  std::vector<ge::AscGraph> impl_graphs_group;
  impl_graphs_group.push_back(impl_graph1);
  impl_graphs_group.push_back(impl_graph2);
  MultiGroupDoScheduler(fused_schedule_result, impl_graphs_group);
}

TEST_F(ScheduleMultiGoupTest, ScheduleMultiGoupCodegen) {
  bool gen_success = true;
  ge::AscGraph graph("schedule_multi_group");
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("schedule_multi_group");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<ge::AscGraph> impl_graphs;

  ConstructMultiGroupGraph(graph, fused_schedule_result);

  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // schedule_multi_group_kernel.cpp
  std::string tiling_src_file_name = parts[1];      // schedule_multi_group_tiling.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result),0);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
   gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(ScheduleMultiGoupTest, ScheduleMultiGoupCodegen_EnableGroupParallel) {
  bool gen_success = true;
  ge::AscGraph graph("schedule_multi_group");
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("schedule_multi_group");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<ge::AscGraph> impl_graphs;

  ConstructMultiGroupGraph(graph, fused_schedule_result);
  auto &schedule_result = fused_schedule_result.node_idx_to_scheduled_results.front().front();
  schedule_result.enable_group_parallel = true;
  schedule_result.schedule_groups.emplace_back(schedule_result.schedule_groups[0]);
  schedule_result.schedule_groups.emplace_back(schedule_result.schedule_groups[0]);
  schedule_result.schedule_groups.emplace_back(schedule_result.schedule_groups[0]);
  schedule_result.schedule_groups.emplace_back(schedule_result.schedule_groups[0]);

  try {
    auto codegen = codegen::Codegen(codegen::CodegenOptions{
        .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(fused_schedule_result, result),0);
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
