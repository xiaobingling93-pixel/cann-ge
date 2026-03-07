/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include "gtest/gtest.h"
#include "tiling_code_generator.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "gen_model_info/stub_graph.h"
#include "base/att_const_values.h"
#include "gen_tiling_impl.h"
#include "graph_construct_utils.h"
#include "common/test_common_utils.h"
#include "test_common_utils.h"

using namespace att;
using namespace ge::ascir_op;
class TestAutoFuse : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};

void AutoFuseBeforeAutoFuse(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  auto s0 = ge::Symbol("s0");
  auto s1 = ge::Symbol("s1");
  auto s2 = ge::Symbol("s2");
  auto z0 = graph.CreateAxis("z0", s0*s1);
  auto z1 = graph.CreateAxis("z1", s2);

  auto axis_list = {z0.id, z1.id};
  std::initializer_list<Expr> repeats = {s0*s1, s2};
  std::initializer_list<Expr> strides = {ZERO, ONE};
  
  int32_t exec_order = 0;
  Data data("data", graph);
  data.attr.sched.exec_order = exec_order++;
  data.attr.sched.axis = axis_list;
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = axis_list;
  *data.y.repeats = repeats;
  *data.y.strides = strides;

  Load load("load");
  load.x = data.y;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = axis_list;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.axis = axis_list;
  *load.y.repeats = repeats;
  *load.y.strides = strides;

  Data data1("data1", graph);
  data1.attr.sched.exec_order = exec_order++;
  data1.attr.sched.axis = axis_list;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = axis_list;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.exec_order = exec_order++;
  load1.attr.sched.axis = axis_list;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = axis_list;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;

  ge::ascir_op::Add add("add");
  add.x1 = load.y;
  add.x2 = load1.y;
  add.attr.sched.exec_order = exec_order++;
  add.attr.sched.axis = axis_list;
  add.y.dtype = ge::DT_FLOAT16;
  *add.y.axis = axis_list;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  Store store("store");
  store.x = add.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = axis_list;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = axis_list;
  *store.y.repeats = repeats;
  *store.y.strides = strides;

  Output data_out("out");
  data_out.x = store.y;
  data_out.attr.sched.exec_order = exec_order++;
  data_out.y.dtype = ge::DT_FLOAT16;
  *data_out.y.axis = axis_list;
  *data_out.y.repeats = repeats;
  *data_out.y.strides = strides;
  std::cout << graph.GetAllAxis()[0]->id << std::endl;
  std::cout << graph.GetAllAxis()[0]->id << std::endl;
  std::cout << graph.GetAllAxis()[0]->id << std::endl;
}

void AutoFuseAfterScheduler(ge::AscGraph &graph) {
  auto z0 = graph.GetAllAxis()[0]->id;
  auto z1 = graph.GetAllAxis()[1]->id;
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  auto axis_0_size = (graph.GetAllAxis()[0])->size;
  auto axis_1_size = (graph.GetAllAxis()[1])->size;

  auto [z1T, z1t] = graph.TileSplit(z1);
  auto z1t_size = (graph.GetAllAxis()[3])->size;
  auto z0z1T = *(graph.MergeAxis({z0, z1T->id}));
  auto [z0z1TB, z0z1Tb] = graph.BlockSplit(z0z1T.id);

  auto axis_list = {z0, z1T->id, z1t->id};
  std::initializer_list<Expr> repeats = {axis_0_size, axis_1_size/z1t_size, z1t_size};
  std::initializer_list<Expr> broad_strides = {axis_1_size, z1t_size, ONE};
  std::initializer_list<Expr> init_strides = {ZERO, z1t_size, ONE};

  auto data = graph.FindNode("data");
  graph.ApplySplit(data, z1T->id, z1t->id);
  graph.ApplyMerge(data, z0z1T.id);
  graph.ApplySplit(data, z0z1TB->id, z0z1Tb->id);
  data->attr.sched.loop_axis = z0z1Tb->id;
  data->outputs[0].attr.axis = axis_list;
  data->outputs[0].attr.repeats = repeats;
  data->outputs[0].attr.strides = init_strides;
  data->outputs[0].attr.vectorized_axis = {z1t->id};

  auto load = graph.FindNode("load");
  graph.ApplySplit(load, z1T->id, z1t->id);
  graph.ApplyMerge(load, z0z1T.id);
  graph.ApplySplit(load, z0z1TB->id, z0z1Tb->id);
  load->attr.sched.loop_axis = z0z1Tb->id;
  load->outputs[0].attr.axis = axis_list;
  load->outputs[0].attr.repeats = repeats;
  load->outputs[0].attr.strides = init_strides;
  load->outputs[0].attr.vectorized_axis = {z1t->id};

  auto data1 = graph.FindNode("data1");
  graph.ApplySplit(data1, z1T->id, z1t->id);
  graph.ApplyMerge(data1, z0z1T.id);
  graph.ApplySplit(data1, z0z1TB->id, z0z1Tb->id);
  data1->attr.sched.loop_axis = z0z1Tb->id;
  data1->outputs[0].attr.vectorized_axis = {z1t->id};

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z1T->id, z1t->id);
  graph.ApplyMerge(load1, z0z1T.id);
  graph.ApplySplit(load1, z0z1TB->id, z0z1Tb->id);
  load1->attr.sched.loop_axis = z0z1Tb->id;
  load1->outputs[0].attr.vectorized_axis = {z1t->id};

  auto add =  graph.FindNode("add");
  graph.ApplySplit(add, z1T->id, z1t->id);
  graph.ApplyMerge(add, z0z1T.id);
  graph.ApplySplit(add, z0z1TB->id, z0z1Tb->id);
  add->attr.sched.loop_axis = z0z1Tb->id;
  add->outputs[0].attr.vectorized_axis = {z1t->id};

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z1T->id, z1t->id);
  graph.ApplyMerge(store, z0z1T.id);
  graph.ApplySplit(store, z0z1TB->id, z0z1Tb->id);
  store->attr.sched.loop_axis = z0z1Tb->id;
  store->outputs[0].attr.vectorized_axis = {z1t->id};
}

void AutoFuseAfterQueBufAlloc(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t bufID = 0;
  int32_t loadBuf = bufID++;
  int32_t addBuf = bufID++;
  int32_t load1Buf = bufID++;
  auto data = graph.FindNode("data");
  data->outputs[0].attr.mem.tensor_id = tensorID++;
  data->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  data->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  data->outputs[0].attr.buf.id = ge::kIdNone;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.mem.tensor_id = tensorID++;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = loadBuf;

  auto data1 = graph.FindNode("data1");
  data1->outputs[0].attr.mem.tensor_id = tensorID++;
  data1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  data1->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  data1->outputs[0].attr.buf.id = ge::kIdNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = tensorID++;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = load1Buf;

  auto add =  graph.FindNode("add");
  add->outputs[0].attr.mem.tensor_id = tensorID++;
  add->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  add->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  add->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  add->outputs[0].attr.buf.id = addBuf;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensorID++;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  store->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  store->outputs[0].attr.buf.id = addBuf;
}

TEST_F(TestAutoFuse, casev2)
{
  ascir::ScheduleGroup schedule_group;
  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::vector<att::ModelInfo> model_info_list;

  ge::AscGraph graph("graph");
  AutoFuseBeforeAutoFuse(graph);
  AutoFuseAfterScheduler(graph);
  AutoFuseAfterQueBufAlloc(graph);

  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  schedule_group.impl_graphs.emplace_back(graph);
  schedule_result.schedule_groups.emplace_back(schedule_group);
  schedule_results.emplace_back(schedule_result);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kGenConfigType, "HighPerf");
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "OpTest4";
  ascir::FusedScheduledResult fused_scheduled_results;
  fused_scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_results, options, tiling_funcs, true);
  for (auto & [key, value] : tiling_funcs) {
    if (key == "TilingHead") {
      std::ofstream oss_head;
      oss_head.open("autofuse_tiling_func_common.h", std::ios::out);
      oss_head << "#include \"AddLayerNorm_tiling_data.h\"\n";
      oss_head << value;
      oss_head.close();
    } else {
      std::ofstream oss;
      oss.open("autofuse_tiling_func_" + key + "_1.cpp", std::ios::out);
      oss << value;
      oss.close();
    }
  }
  EXPECT_EQ(res, true);
  options.emplace(kGenTilingDataDef, "1");
  res = GenTilingImpl(op_name, {graph}, options);
  EXPECT_EQ(res, true);
  auto ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/tiling_func_main_auto_fuse.cpp ./ -f").c_str());
  ret = std::system(std::string("cp ").append(ST_DIR).append("/testcase/op_log.h ./ -f").c_str());
  ret = autofuse::test::CopyStubFiles(ST_DIR, "testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system(
      "g++ tiling_func_main_auto_fuse.cpp OpTest4_*_tiling_func.cpp -I ./ "
      "-o tiling_func_main_autofuse -g");
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_autofuse");
}

TEST_F(TestAutoFuse, casev3)
{
  ascir::ScheduleGroup schedule_group;
  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results;
  std::vector<att::ModelInfo> model_info_list;

  ge::AscGraph graph("graph");
  AutoFuseBeforeAutoFuse(graph);
  AutoFuseAfterScheduler(graph);
  AutoFuseAfterQueBufAlloc(graph);

  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  schedule_group.impl_graphs.emplace_back(graph);
  schedule_result.schedule_groups.emplace_back(schedule_group);
  schedule_results.emplace_back(schedule_result);
  std::map<std::string, std::string> options;
  options.emplace(kTilingDataTypeName, "NpuKernel0TilingData");
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kDurationLevelName, "1");
  options.emplace(kGenConfigType, "HighPerf");
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "OpTest";
  ascir::FusedScheduledResult fused_scheduled_results;
  fused_scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_results, options, tiling_funcs, true);
  for (auto & [key, value] : tiling_funcs) {
    if (key == "TilingHead") {
      std::ofstream oss_head;
      oss_head.open("autofuse_tiling_func_common.h", std::ios::out);
      oss_head << "#include \"AddLayerNorm_tiling_data.h\"\n";
      oss_head << value;
      oss_head.close();
    } else {
      std::ofstream oss;
      oss.open("autofuse_tiling_func_" + key + "_2.cpp", std::ios::out);
      oss << value;
      oss.close();
    }
  }
  EXPECT_EQ(res, true);
}