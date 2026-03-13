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
#include <string>

#include <ascendc_ir.h>
#include "ascir.h"
#include <ascir_ops.h>
#include <ascir_utils.h>
#include "tests/autofuse/framework/easy_asc_graph/asc_graph_builder.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "runtime_stub.h"

#define private public
#include "optimize.h"
#include "platform_context.h"
#undef private
#include "asc_tensor_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "codegen.h"
#include "autofuse/utils/autofuse_attrs.h"
#include "task_generator/transpose_schedule_case_generator.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"
#include "attribute_group/attr_group_shape_env.h"
#include "autoschedule/tiling_group.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "platform_context.h"
#include "platformv2.h"
#include "optimize/graph_pass/pass_runner_handler.h"
#include "autoschedule/autoschedule.h"
#include "base/att_const_values.h"
#include "graph_pass/pow_equiv_substitution_pass.h"
#include "template/nddma_template.h"
#define protected public
#include "un_alignment_strategy.h"
#undef protected

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace optimize;
using namespace ge::testing;

class TestOptimizerV2 : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
  optimize::Optimizer optimizer;

  TestOptimizerV2() : optimizer(optimize::OptimizerOptions{}) {}

  static std::stringstream &SizeExprListStr(std::stringstream &ss, const ge::AscGraph &graph,
                                            const std::vector<ge::Expression> &size_expr_list) {
    for (auto &size_expr : size_expr_list) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss;
  }

  static std::stringstream &AxisListStr(std::stringstream &ss, ge::AscGraph &graph,
                                        const std::vector<ge::AxisId> &axis_list) {
    for (auto axis_id : axis_list) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss;
  }
};

TEST_F(TestOptimizerV2, platform_reg_test) {
  ge::AscGraph graph("tmp");
  std::string platform_str;
  ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_str);
  EXPECT_EQ(platform_str, "3510");
  const auto platform_v2 = optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_NE(platform_v2, nullptr);
  EXPECT_EQ(platform_v2->PartitionSubFunctions(graph), ge::SUCCESS);
  EXPECT_NE(platform_v2->GetAlignmentStrategy(), nullptr);
  EXPECT_NE(platform_v2->GetTemplateGenerator(), nullptr);
}

TEST_F(TestOptimizerV2, NotRemovePad) {
  ge::AscGraph graph("Autoschedule_autoschedule_removepad_broadcast");
  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(3);
  auto s2 = graph.CreateSizeVar(3);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {One, s1, s2};
  *data0.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {One, s1, s2};
  *load0.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1, s2};
  *data1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = brc0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id, z2.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id, z2.id};
  *y0.y.repeats = {s0, s1, s2};
  *y0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data2("data2", graph);
  data2.ir_attr.SetIndex(2);
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  data2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data2.y.dtype = ge::DT_FLOAT16;
  *data2.y.axis = {z0.id, z1.id, z2.id};
  *data2.y.repeats = {One, s1, s2};
  *data2.y.strides = {Zero, s2, One};

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, s1, s2};
  *load2.y.strides = {Zero, s2, One};

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = load2.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {s0, s1, s2};
  *brc2.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = load1.y;
  mul0.x2 = brc2.y;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Store store1("store1");
  store1.x = mul0.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "Autoschedule_autoschedule_removepad_broadcast_B0Y0");
}

/**
 * load0
 *   \
 * brc0
 *   \
 * brc1
 *   \
 *  brc2   load1
 *     \    /
 *      add
 *       |
 *     store
 */
TEST_F(TestOptimizerV2, ContinuesBroadcastOptimization_3Brc) {
  ge::AscGraph graph("ContinuesBroadcastOptimization");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, One, One, s3};
  *load0.y.strides = {Zero, Zero, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc0.y.repeats = {One, One, s2, s3};
  *brc0.y.strides = {Zero, Zero, s3, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc1.y.repeats = {One, s1, s2, s3};
  *brc1.y.strides = {Zero, s2 * s3, s3, One};

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc2.y.repeats = {s0, s1, s2, s3};
  *brc2.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(0);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, s3};
  *load1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add.x1 = brc2.y;
  add.x2 = load1.y;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.repeats = {s0, s1, s2, s3};
  *add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, s3};
  *store.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;

  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);

  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);

  EXPECT_NE(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2")->GetInDataAnchor(0)->GetPeerOutAnchor(),
            compute_graph->FindNode("load0")->GetOutDataAnchor(0));
}

TEST_F(TestOptimizerV2, PowScalar0) {
  ge::AscGraph graph("PowBrc");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input("pow_input", graph);
  pow_input.ir_attr.SetValue("0.0");

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = pow_input.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {One, s1};
  *brc0.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Pow pow("pow");
  pow.x1 = load0.y;
  pow.x2 = brc1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, One};

  ge::ascir_op::Add add("add0");
  add.x1 = pow.y;
  add.x2 = load0.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input1("pow_input1", graph);
  pow_input1.ir_attr.SetValue("haha");

  ge::ascir_op::Pow pow1("pow1");
  pow1.x1 = pow.y;
  pow1.x2 = pow_input1.y;
  pow1.attr.sched.axis = {z0.id, z1.id};
  *pow1.y.axis = {z0.id, z1.id};
  *pow1.y.repeats = {s0, s1};
  *pow1.y.strides = {s1, One};

  ge::ascir_op::Add add1("add1");
  add1.x1 = add.y;
  add1.x2 = pow1.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input2("pow_input2", graph);
  pow_input2.ir_attr.SetValue("0.6");

  ge::ascir_op::Pow pow2("pow2");
  pow2.x1 = add1.y;
  pow2.x2 = pow_input2.y;
  pow2.attr.sched.axis = {z0.id, z1.id};
  *pow2.y.axis = {z0.id, z1.id};
  *pow2.y.repeats = {s0, s1};
  *pow2.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input3("pow_input3", graph);
  pow_input3.ir_attr.SetValue("0.000000001");

  ge::ascir_op::Pow pow3("pow3");
  pow3.x1 = pow2.y;
  pow3.x2 = pow_input3.y;
  pow3.attr.sched.axis = {z0.id, z1.id};
  *pow3.y.axis = {z0.id, z1.id};
  *pow3.y.repeats = {s0, s1};
  *pow3.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = pow3.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  ::ascir::utils::DumpGraph(graph, "BEFORE");
  PowEquivSubstitutionPass pass;
  EXPECT_EQ(pass.RunPass(graph), ge::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "AFTER");
  auto pow0_node = graph.FindNode("pow");
  EXPECT_EQ(pow0_node, nullptr);
  auto pow1_node = graph.FindNode("pow1");
  EXPECT_NE(pow1_node, nullptr);
  auto pow2_node = graph.FindNode("pow2");
  EXPECT_NE(pow2_node, nullptr);
  auto pow3_node = graph.FindNode("pow3");
  EXPECT_NE(pow3_node, nullptr);
}

TEST_F(TestOptimizerV2, PowScalar1) {
  ge::AscGraph graph("PowRemove");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input("pow_input", graph);
  pow_input.ir_attr.SetValue("1.00000000000000000000e+00");

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = pow_input.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {One, s1};
  *brc0.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Pow pow("pow");
  pow.x1 = load0.y;
  pow.x2 = brc1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = pow.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  ::ascir::utils::DumpGraph(graph, "BEFORE");
  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "AFTER");
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
}

TEST_F(TestOptimizerV2, PowScalarSqrt) {
  ge::AscGraph graph("PowSqrt");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  *data0.y.axis = {z0.id, z1.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input("pow_input", graph);
  pow_input.ir_attr.SetValue("0.5");

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = pow_input.y;
  brc0.attr.sched.axis = {z0.id, z1.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {One, s1};
  *brc0.y.strides = {Zero, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  ge::ascir_op::Pow pow("pow");
  pow.x1 = load0.y;
  pow.x2 = brc1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, One};

  ge::ascir_op::Scalar pow_input1("pow_input1", graph);
  pow_input1.ir_attr.SetValue("2.0000000");

  ge::ascir_op::Pow pow1("pow1");
  pow1.x1 = pow.y;
  pow1.x2 = pow_input1.y;
  pow1.attr.sched.axis = {z0.id, z1.id};
  *pow1.y.axis = {z0.id, z1.id};
  *pow1.y.repeats = {s0, s1};
  *pow1.y.strides = {s1, One};

  ge::ascir_op::Store store("store");
  store.x = pow1.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  ge::ascir_op::Output y("y");
  y.ir_attr.SetIndex(0);
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  ::ascir::utils::DumpGraph(graph, "BEFORE");
  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "AFTER");
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  auto pow1_node = graph.FindNode("pow1");
  EXPECT_EQ(pow1_node, nullptr);
}

TEST_F(TestOptimizerV2, GatherReduceFuse) {
  ge::AscGraph graph("gather_reduce");

  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");
  const Expression s2 = graph.CreateSizeVar("s2");
  const Expression s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1, s2};
  *data0.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z3.id};
  *data1.y.axis = {z3.id};
  *data1.y.repeats = {s3};
  *data1.y.strides = {One};

  ge::ascir_op::Gather gather("gather");
  gather.attr.api.compute_type = ComputeType::kComputeGather;
  gather.x1 = data0.y;
  gather.x2 = data1.y;
  gather.ir_attr.SetAxis(2);
  gather.attr.sched.axis = {z0.id, z1.id, z3.id};
  gather.y.dtype = ge::DT_FLOAT;
  *gather.y.axis = {z0.id, z1.id, z3.id};
  *gather.y.repeats = {s0, s1, s3};
  *gather.y.strides = {s1 * s3, s3, One};

  ge::ascir_op::Sum sum("sum");
  sum.attr.api.compute_type = ComputeType::kComputeReduce;
  sum.x = gather.y;
  sum.attr.sched.axis = {z0.id, z1.id, z3.id};
  sum.y.dtype = ge::DT_FLOAT16;
  *sum.y.axis = {z0.id, z1.id, z3.id};
  *sum.y.repeats = {s0, s1, One};
  *sum.y.strides = {s1, One, Zero};

  ge::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.x = sum.y;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);

  auto gather_node = graph.FindNode("gather");
  EXPECT_EQ(gather_node->attr.api.compute_type, ComputeType::kComputeLoad);
}

/**
 *       load0
 *         |
 *       brc0
 *         |
 *       brc1
 *      /    \
 *  store0  brc2   load1
 *            \    /
 *             add
 *              |
 *             store1
 */
TEST_F(TestOptimizerV2, ContinuesBroadcastOptimization_Brc1MultiOut) {
  ge::AscGraph graph("ContinuesBroadcastOptimization");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, One, One, s3};
  *load0.y.strides = {Zero, Zero, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc0.y.repeats = {One, One, s2, s3};
  *brc0.y.strides = {Zero, Zero, s3, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc1.y.repeats = {One, s1, s2, s3};
  *brc1.y.strides = {Zero, s2 * s3, s3, One};

  ge::ascir_op::Store store0("store0");
  store0.x = brc1.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store0.y.repeats = {One, s1, s2, s3};
  *store0.y.strides = {Zero, s2 * s3, s3, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc2.y.repeats = {s0, s1, s2, s3};
  *brc2.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, s3};
  *load1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add.x1 = brc2.y;
  add.x2 = load1.y;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.repeats = {s0, s1, s2, s3};
  *add.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Store store1("store1");
  store1.x = add.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store1.y.repeats = {s0, s1, s2, s3};
  *store1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;

  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc1")->GetInDataAnchor(0)->GetPeerOutAnchor(),
            compute_graph->FindNode("load0")->GetOutDataAnchor(0));
}

/**
 *       load0
 *         \
 *       brc0
 *         \
 *       brc1
 *      /    \
 *    abs   brc2   load1
 *     |      \    /
 *   brc3      add
 *     |         |
 *   brc4    store1
 *     |
 *  store0
 */
TEST_F(TestOptimizerV2, ContinuesBroadcastOptimization_MultiBrcPath_Brc1MultiOut) {
  ge::AscGraph graph("ContinuesBroadcastOptimization");
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

  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, One, One, One, s4};
  *load0.y.strides = {Zero, Zero, Zero, Zero, One};

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load0.y;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.y.dtype = ge::DT_FLOAT16;
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {One, One, One, s3, s4};
  *brc0.y.strides = {Zero, Zero, Zero, s4, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = ge::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {One, One, s2, s3, s4};
  *brc1.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc1.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *abs.y.repeats = {One, One, s2, s3, s4};
  *abs.y.strides = {Zero, Zero, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc3("brc3");
  brc3.x = abs.y;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.y.dtype = ge::DT_FLOAT16;
  *brc3.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc3.y.repeats = {One, s1, s2, s3, s4};
  *brc3.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Broadcast brc4("brc4");
  brc4.x = brc3.y;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc4.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc4.y.dtype = ge::DT_FLOAT16;
  *brc4.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc4.y.repeats = {s0, s1, s2, s3, s4};
  *brc4.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store0("store0");
  store0.x = brc4.y;
  store0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = ge::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store0.y.repeats = {s0, s1, s2, s3, s4};
  *store0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y0.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.y.dtype = ge::DT_FLOAT16;
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {One, s1, s2, s3, s4};
  *brc2.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {One, s1, s2, s3, s4};
  *load1.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add.x1 = brc2.y;
  add.x2 = load1.y;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.repeats = {One, s1, s2, s3, s4};
  *add.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Store store1("store1");
  store1.x = add.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = ge::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {One, s1, s2, s3, s4};
  *store1.y.strides = {Zero, s2 * s3 * s4, s3 * s4, s4, One};

  ge::ascir_op::Output y1("y1");
  y1.ir_attr.SetIndex(1);
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DT_FLOAT16;

  Status res = optimize::autoschedule::PassRunnerHandler().RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);

  EXPECT_EQ(compute_graph->FindNode("brc0"), nullptr);

  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("load0")->GetOutDataAnchor(0),
            compute_graph->FindNode("brc1")->GetInDataAnchor(0)->GetPeerOutAnchor());

  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("abs")->GetOutDataAnchor(0),
            compute_graph->FindNode("brc4")->GetInDataAnchor(0)->GetPeerOutAnchor());
}

TEST_F(TestOptimizerV2, ScalarBroadcastOptimization_Not_Support_VF) {
  ge::AscGraph graph("ScalarBroadcastOptimization_Not_Support_VF");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(5);
  auto s2 = graph.CreateSizeVar(6);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.repeats = {ge::ops::One, ge::ops::One, ge::ops::One};
  *load.y.strides = {ge::ops::Zero, ge::ops::Zero, ge::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.repeats = {ge::ops::One, ge::ops::One, s2};
  *brc1.y.strides = {ge::ops::Zero, ge::ops::Zero, ge::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = ge::DT_FLOAT;
  *brc2.y.repeats = {ge::ops::One, s1, s2};
  *brc2.y.strides = {ge::ops::Zero, s2, ge::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = ge::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, ge::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = ge::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, ge::ops::One};

  Add add("add");
  add.attr.sched.axis = {z0.id, z1.id, z2.id};
  add.x1 = brc3.y;
  add.x2 = load1.y;
  add.y.dtype = ge::DT_FLOAT;
  *add.y.axis = {z0.id, z1.id, z2.id};
  *add.y.repeats = {s0, s1, s2};
  *add.y.strides = {s1 * s2, s2, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, ge::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;

  Status res = optimize::autoschedule::PassRunnerHandler::RunPasses(graph);
  EXPECT_EQ(res, ge::SUCCESS);
  auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
}

/**
 *           data0
 *             |
 *           load0
 *             |
 *         broadcast
 *             |
 *            store
 *              |
 *           output
 */
TEST_F(TestOptimizerV2, NddmaCaseBrcOutputWithSingleRef) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {One, One};
  *load0.y.repeats = {One, One};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
}

TEST_F(TestOptimizerV2, LargeTailBrcToNddmaLowScoreFunc) {
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(2012);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {Symbol(1), Symbol(0)};
  *load0.y.repeats = {s0, Symbol(1)};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y1");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y1_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[2].scheduled_graph);
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  return -1;\n"
      "}\n";
  EXPECT_EQ(score_func, res);
}

TEST_F(TestOptimizerV2, LargeTailBrcToNddmaLowScoreFunc_Dynaminc) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {Symbol(1), Symbol(0)};
  *load0.y.repeats = {s0, Symbol(1)};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y1");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y1_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[2].scheduled_graph);
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  const auto tail_size = static_cast<int64_t>((4 * tiling_data.s1));\n"
      "  if (tail_size > 4096) { return -1; }\n"
      "  return 0;\n"
      "}\n";
  EXPECT_EQ(score_func, res);
}

TEST_F(TestOptimizerV2, LoadBrcToNddmaAlignLowScoreFunc) {
  const auto dtype = ge::DT_FLOAT16;
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.strides = {Symbol(1), Symbol(0)};
  *load0.y.repeats = {s0, Symbol(1)};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = dtype;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y1");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y1_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[2].scheduled_graph);
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  return -1;\n"
      "}\n";
  EXPECT_EQ(score_func, res);
}


TEST_F(TestOptimizerV2, LoadBrcToNddmaAlignLowScoreFunc_NotContinue) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {Symbol(0), Symbol(128), Symbol(1)};
  *load0.y.repeats = {Symbol(1), s1, s2};
  *load0.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *load0.y.vectorized_strides = {Symbol(0), Symbol(128), Symbol(1)};

  Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc.y.axis = {z0.id, z2.id, z1.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.strides = {s1 * Symbol(128), Symbol(128), One};
  *brc.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = brc.y;
  *store_op.y.axis = {z0.id, z2.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 * Symbol(128), Symbol(128), One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();

  EXPECT_EQ(impl_graphs.size(), 5);
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[3].scheduled_graph);
  const auto res = "";
  EXPECT_EQ(score_func, res);
}

TEST_F(TestOptimizerV2, LoadBrcToNddmaAlignScoreFunc_Dynamic) {
  const auto dtype = ge::DT_FLOAT16;
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.strides = {Symbol(1), Symbol(0)};
  *load0.y.repeats = {s0, Symbol(1)};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = dtype;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y1");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y1_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[2].scheduled_graph);
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  const auto tail_size = static_cast<int64_t>((2 * tiling_data.s1));\n"
      "  if (tail_size % 32 == 0) { return -1; }\n"
      "  if (tail_size > 4096) { return -1; }\n"
      "  return 0;\n"
      "}\n";
  EXPECT_EQ(score_func, res);
}

TEST_F(TestOptimizerV2, LoadBrcToNddmaNotAlignHighScoreFunc) {
  const auto dtype = ge::DT_FLOAT16;
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(31);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.strides = {Symbol(1), Symbol(0)};
  *load0.y.repeats = {s0, Symbol(1)};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = dtype;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y1");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y1_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[2].scheduled_graph);
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  return 0;\n"
      "}\n";
  EXPECT_EQ(score_func, res);
}

TEST_F(TestOptimizerV2, LoadBrcToNddmaHighScoreFunc_Dynamic) {
  const auto dtype = ge::DT_FLOAT;
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = dtype;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.strides = {Symbol(0), Symbol(1)};
  *load0.y.repeats = {Symbol(1), s1};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = dtype;
  *broadcast.y.strides = {s1, One};
  *broadcast.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = broadcast.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0Y1");
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");

  const auto nddma_template = ge::ComGraphMakeUnique<NddmaTemplate>();
  const auto score_func = nddma_template->GetScoreFunc(graph, impl_graphs[2].scheduled_graph);
  const auto res =
      "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      "  const auto tail_size = static_cast<int64_t>((4 * tiling_data.s1));\n"
      "  return 0;\n"
      "}\n";
  EXPECT_EQ(score_func, res);
}

TEST_F(TestOptimizerV2, NddmaCaseTranspose10OutputWithSingleRef) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,One};
  *load0.y.repeats = {s0, s1};

  Transpose transpose("transpose");
  transpose.x = load0.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.strides = {s0, One};
  *transpose.y.repeats = {s1, s0};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z1.id, z0.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s0, One};
  *store_op.y.repeats = {s1, s0};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0X1Y0_nddma");
}

TEST_F(TestOptimizerV2, NddmaCaseTranspose021OutputWithSingleRef) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 * s2 , s2 ,  One};
  *load0.y.repeats = {s0, s1, s2};

  Transpose transpose("transpose");
  transpose.x = load0.y;
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};
  *transpose.y.axis = {z0.id, z2.id, z1.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.strides = {s2 * s1, s1, One};
  *transpose.y.repeats = {s0, s2, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z0.id, z2.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s2 * s1, s1, One};
  *store_op.y.repeats = {s0, s2, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 2);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0X2Y0_nddma");
  EXPECT_EQ(impl_graphs[1].scheduled_graph.GetName(), "gen_nddma_B0X2Y1_nddma");
}

TEST_F(TestOptimizerV2, LoadBrcTransposeCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, One};
  *load0.y.strides = {Zero, Zero};

  Broadcast broadcast("broadcast");
  broadcast.x = load0.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Transpose transpose("transpose");
  transpose.x = broadcast.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.strides = {s0, One};
  *transpose.y.repeats = {s1, s0};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z1.id, z0.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s0, One};
  *store_op.y.repeats = {s1, s0};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0X1Y0_nddma");
}

TEST_F(TestOptimizerV2, LoadCastBrcCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1, One};
  *cast1.y.strides = {s1, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = cast1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2};
  *broadcast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  EXPECT_EQ(autoschedule.DoAutoSchedule(), ge::SUCCESS);
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "gen_nddma_B0Y0_nddma");
  EXPECT_EQ(impl_graphs[3].scheduled_graph.GetName(), "gen_nddma_B0Y1_nddma");
}

TEST_F(TestOptimizerV2, LoadCastAndTailAxisBrcCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, One};
  *load0.y.strides = {s1, One, Zero};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, One, One};
  *load1.y.strides = {One, Zero, Zero};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, One, One};
  *cast1.y.strides = {One, Zero, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = cast1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, One};
  *broadcast1.y.strides = {s1, One, Zero};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, One};
  *mul0.y.strides = {s1, One, Zero};

  Broadcast broadcast2("broadcast2");
  broadcast2.x = mul0.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast2.y.axis = {z0.id, z1.id, z2.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {s0, s1, s2};
  *broadcast2.y.strides = {s1 * s2, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = broadcast2.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Cast");
    }
  }
}

TEST_F(TestOptimizerV2, LoadCastBrcTransposeCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {One, One};
  *load0.y.strides = {Zero, Zero};

  Cast cast1("cast1");
  cast1.x = load0.y;
  cast1.attr.sched.axis = {z0.id, z1.id};
  *cast1.y.axis = {z0.id, z1.id};
  cast1.y.dtype = ge::DT_FLOAT16;
  *cast1.y.repeats = {One, One};
  *cast1.y.strides = {Zero, Zero};

  Broadcast broadcast("broadcast");
  broadcast.x = cast1.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT16;
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Transpose transpose("transpose");
  transpose.x = broadcast.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  transpose.y.dtype = ge::DT_FLOAT16;
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z1.id, z0.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s1, s0};
  *store_op.y.strides = {s0, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0X1Y0_nddma");
}

TEST_F(TestOptimizerV2, LoadCastTransposeCase) {
  AscGraph graph("gen_nddma");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Cast cast1("cast1");
  cast1.x = load0.y;
  cast1.attr.sched.axis = {z0.id, z1.id};
  *cast1.y.axis = {z0.id, z1.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1};
  *cast1.y.strides = {s1, One};

  Transpose transpose("transpose");
  transpose.x = cast1.y;
  transpose.attr.sched.axis = {z0.id, z1.id};
  *transpose.y.axis = {z1.id, z0.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z1.id, z0.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s1, s0};
  *store_op.y.strides = {s0, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(5);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0X1Y0_nddma");
  for (auto node: impl_graphs[0].scheduled_graph.GetAllNodes()) {
    if (node->GetType() == "Nddma") {
      EXPECT_EQ(att::Str(node->outputs[0].attr.vectorized_strides[0]), "(16 * Ceiling((Rational(1 , 16) * z0t_size)))");
    }
    if (node->GetType() == att::kCast) {
      EXPECT_EQ(att::Str(node->outputs[0].attr.vectorized_strides[0]), "(8 * Ceiling((Rational(1 , 8) * z0t_size)))");
    }
  }
}

TEST_F(TestOptimizerV2, LoadGEWhereTransposeCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(41);
  auto s1 = graph.CreateSizeVar(54);
  auto s2 = graph.CreateSizeVar(38);
  auto s3 = graph.CreateSizeVar(55);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2, s3};
  *load0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, s3};
  *load1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Ge ge("ge");
  ge.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  ge.x1 = load0.y;
  ge.x2 = load1.y;
  *ge.y.axis = {z0.id, z1.id, z2.id, z3.id};
  ge.y.dtype = ge::DT_UINT8;
  *ge.y.repeats = {s0, s1, s2, s3};
  *ge.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Where where("where");
  where.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  where.x1 = ge.y;
  where.x2 = load1.y;
  *where.y.axis = {z0.id, z1.id, z2.id, z3.id};
  where.y.dtype = ge::DT_FLOAT;
  *where.y.repeats = {s0, s1, s2, s3};
  *where.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Transpose transpose("transpose");
  transpose.x = where.y;
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *transpose.y.axis = {z0.id, z3.id, z1.id, z2.id};
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.repeats = {s0, s3, s1, s2};
  *transpose.y.strides = {s3 * s1 * s2, s1 * s2, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store_op.x = transpose.y;
  *store_op.y.axis = {z0.id, z3.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s3, s1, s2};
  *store_op.y.strides = {s3 * s1 * s2, s1 * s2, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 3);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0X3Y0_nddma");
  for (auto node: impl_graphs[0].scheduled_graph.GetAllNodes()) {
    if (node->GetType() == "Nddma") {
      EXPECT_EQ(att::Str(node->outputs[0].attr.vectorized_strides[0]), "(2592 * z3t_size)");
    }
    if (node->GetType() == att::kGe) {
      EXPECT_EQ(att::Str(node->outputs[0].attr.vectorized_strides[0]), "(3456 * z3t_size)");
    }
  }
}

TEST_F(TestOptimizerV2, LoadCastBrcMulMinCase) {
  AscGraph graph("nddma_alignment");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1, One};
  *cast1.y.strides = {s1, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = cast1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2};
  *broadcast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  Min min0("min0");
  min0.attr.sched.axis = {z0.id, z1.id, z2.id};
  min0.x = mul0.y;
  min0.y.dtype = ge::DT_FLOAT;
  *min0.y.axis = {z0.id, z1.id, z2.id};
  *min0.y.repeats = {One, s1, s2};
  *min0.y.strides = {Zero, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = min0.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {One, s1, s2};
  *store_op.y.strides = {Zero, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  EXPECT_EQ(autoschedule.DoAutoSchedule(), ge::SUCCESS);
  EXPECT_EQ(impl_graphs.size(), 4);
  EXPECT_EQ(impl_graphs[2].scheduled_graph.GetName(), "nddma_alignment_B0Y1R0_load_to_nddma");
}

TEST_F(TestOptimizerV2, LoadToNddmaCase) {
  AscGraph graph("gen_transpose_load");

  auto s0 = graph.CreateSizeVar(129);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(32);
  auto s3 = graph.CreateSizeVar(68);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2, s3};
  *load0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1, s2, One};
  *load1.y.strides = {s1 * s2, s2, One, Zero};

  Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1, s2, s3};
  *broadcast1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  mul0.x1 = load0.y;
  mul0.x2 = broadcast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *mul0.y.repeats = {s0, s1, s2, s3};
  *mul0.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Sum sum0("sum0");
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  sum0.x = mul0.y;
  sum0.y.dtype = ge::DT_FLOAT;
  *sum0.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *sum0.y.repeats = {s0, One, One, s3};
  *sum0.y.strides = {s3, Zero, Zero, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store_op.x = sum0.y;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_op.y.repeats = {s0, One, One, s3};
  *store_op.y.strides = {s3, Zero, Zero, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  autoschedule.DoAutoSchedule();
  bool ret = false;
  for (long unsigned int i = 0; i < impl_graphs.size(); i++) {
    size_t pos = impl_graphs[i].scheduled_graph.GetName().find("load_to_nddma");
    if (pos != std::string::npos) {
      ret = true;
    }
  }
  EXPECT_EQ(ret, true);
}

TEST_F(TestOptimizerV2, LoadCastStoreCase) {
  AscGraph graph("gen_nddma");

  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_UINT8;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};

  Cast cast1("cast1");
  cast1.x = load1.y;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  cast1.y.dtype = ge::DT_FLOAT;
  *cast1.y.repeats = {s0, s1, s2};
  *cast1.y.strides = {s1 * s2, s2, One};

  Mul mul0("mul0");
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul0.x1 = load0.y;
  mul0.x2 = cast1.y;
  mul0.y.dtype = ge::DT_FLOAT;
  *mul0.y.axis = {z0.id, z1.id, z2.id};
  *mul0.y.repeats = {s0, s1, s2};
  *mul0.y.strides = {s1 * s2, s2, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = mul0.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  EXPECT_EQ(Optimizer::MergeContinuousAxis(graph), ge::SUCCESS);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, impl_graphs);
  ASSERT_EQ(autoschedule.DoAutoSchedule(), ge::SUCCESS);
  EXPECT_EQ(impl_graphs.size(), 1);
  EXPECT_EQ(impl_graphs[0].scheduled_graph.GetName(), "gen_nddma_B0Y0");
}

TEST_F(TestOptimizerV2, LoadOpSequenceAdjustCase1) {
  ge::AscGraph graph("reorder_load_op");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, One};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(3);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {ge::ops::One ,ge::ops::One};
  *load1.y.repeats = {ge::ops::One, ge::ops::One};

  Broadcast broadcast("broadcast");
  broadcast.x = load1.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1 ,ge::ops::One};
  *broadcast.y.repeats = {s0, s1};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
  }

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  std::set<std::string> supported_types = {"Data", "Output", "VectorFunc", "Load", "Store", "Nddma"};
  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    EXPECT_TRUE(supported_types.count(node->GetType()) > 0UL);
  }
}

TEST_F(TestOptimizerV2, ReduceFuseWithBrc) {
  auto s0 = Sym(128);
  auto s1 = Sym(112);
  auto s2 = Sym(256);
  auto s3 = Sym(3);

  // data0 [128,112,256,3] FP16 -> load0 -> cast0 (FP16->FP32)
  // data1 [128,112,256,1] FP16 -> load1 -> cast1 (FP16->FP32) -> brc (broadcast 轴3 to 3)
  // div(cast0, brc) -> reduce_max(axes=[0,1,3]) -> cast2 (FP32->FP16) -> store -> out1
  auto graph = AscGraphBuilder("test")
      .Loops({s0, s1, s2, s3})
      // 第一个输入路径：data0(index=0, FP16)
      .Data("data0", 0, ge::DT_FLOAT16)
      .Load("load0", "data0")
      .Cast("cast0", "load0", ge::DT_FLOAT)
      // 第二个输入路径：data1(index=1, FP16)，需要broadcast
      .Data("data1", 1, ge::DT_FLOAT16)
      .Load("load1", "data1", {s0, s1, s2, ge::sym::kSymbolOne},
            {s1 * s2, s2, ge::sym::kSymbolOne, ge::sym::kSymbolZero})
      .Cast("cast1", "load1", ge::DT_FLOAT)
      .Broadcast("brc", "cast1", {3})
      .Op<ascir_op::TrueDiv>("div", {"cast0", "brc"})
      .Max("max", "div", {0, 1, 3})
      // Cast 回 FP16
      .Cast("cast2", "max", ge::DT_FLOAT16)
      // Store -> Output(index=0)
      .Store("store", "cast2")
      .Output("out1", "store", 0)
      .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
}

TEST_F(TestOptimizerV2, BackendSpec) {
  auto spec = optimize::BackendSpec::GetInstance();
  ASSERT_TRUE(spec != nullptr);
  ASSERT_EQ(spec->concat_max_input_num, 512);
}

TEST_F(TestOptimizerV2, TestNddmaReAlignVectorizedStrides) {
  AscGraph graph("test");
  auto s0 = graph.CreateSizeVar(128);

  auto z0 = graph.CreateAxis("z0", s0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id};
  *load0.y.axis = {z0.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.repeats = {s0};
  *load0.y.strides = {One};

  Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id};
  abs.y.dtype = ge::DT_FLOAT16;
  *abs.y.axis = {z0.id};
  *abs.y.repeats = {s0};
  *abs.y.strides = {One};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;

  Store store("store");
  store.attr.sched.axis = {z0.id};
  store.x = abs.y;
  *store.y.axis = {z0.id};
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.repeats = {s0};
  *store.y.strides = {One};

  Output out1("out1");
  out1.x = store.y;
  out1.ir_attr.SetIndex(0);

  for (const auto &node : graph.GetAllNodes()) {
    EXPECT_EQ(optimize::NddmaTemplate::ReAlignVectorizedStrides(node), SUCCESS);
  }
}
TEST_F(TestOptimizerV2, SliceConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(256);
  auto s1 = graph.CreateSizeVar(50);
  auto s2 = graph.CreateSizeVar(16);
  auto s2_sliced = graph.CreateSizeVar(7);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id, z2.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1, s2_sliced};
  *load0.y.strides = {s1 * s2, s2, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  concat_op.x = {load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y,
                 load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y,
                 load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y, load0.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id, z2.id};
  *concat_op.y.repeats = {s0, s1, s2_sliced + s2_sliced};
  *concat_op.y.strides = {s1 * (s2_sliced + s2_sliced), s2_sliced + s2_sliced, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2_sliced + s2_sliced};
  *store_op.y.strides = {s1 * (s2_sliced + s2_sliced), s2_sliced + s2_sliced, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());
  auto concat_node =
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  EXPECT_EQ(ToString(concat_node->outputs[0].attr.vectorized_strides), "[14, 1]");
}

TEST_F(TestOptimizerV2, SplitAndFirstDimConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s1_0 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Split split0("split");
  split0.InstanceOutputy(2U);  // 需要先指定输出个数
  split0.attr.sched.axis = {z0.id, z1.id};
  split0.x = load0.y;
  split0.y[0].dtype = ge::DT_FLOAT;
  *split0.y[0].axis = {z0.id, z1.id};
  *split0.y[0].repeats = {s0, s1_0};
  *split0.y[0].strides = {s1_0, One};
  split0.y[1].dtype = ge::DT_FLOAT;
  *split0.y[1].axis = {z0.id, z1.id};
  *split0.y[1].repeats = {s0, s1_0};
  *split0.y[1].strides = {s1_0, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {split0.y[0], split0.y[1]};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0 + s0, s1_0};
  *concat_op.y.strides = {s1_0, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0 + s0, s1_0};
  *store_op.y.strides = {s1_0 , ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  EXPECT_TRUE(impl_graph.FindNode("concat") == nullptr);;
  EXPECT_TRUE(impl_graph.FindNode("split") != nullptr);;
}

TEST_F(TestOptimizerV2, FirstDimSplitAndConcat) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s0_0 = graph.CreateSizeVar(8);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);

  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  Split split0("split");
  split0.InstanceOutputy(2U);  // 需要先指定输出个数
  split0.attr.sched.axis = {z0.id, z1.id};
  split0.x = load0.y;
  split0.y[0].dtype = ge::DT_FLOAT;
  *split0.y[0].axis = {z0.id, z1.id};
  *split0.y[0].repeats = {s0_0, s1};
  *split0.y[0].strides = {s1, One};
  split0.y[1].dtype = ge::DT_FLOAT;
  *split0.y[1].axis = {z0.id, z1.id};
  *split0.y[1].repeats = {s0_0, s1};
  *split0.y[1].strides = {s1, One};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {split0.y[0], split0.y[1]};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0_0, s1 + s1};
  *concat_op.y.strides = {s1 + s1, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0_0, s1 + s1};
  *store_op.y.strides = {s1 + s1 , ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  EXPECT_TRUE(impl_graph.FindNode("concat") != nullptr);;
  EXPECT_TRUE(impl_graph.FindNode("split") == nullptr);;
}

TEST_F(TestOptimizerV2, ConcatSingleDim) {
  AscGraph graph("slice_concat");
  auto s1 = graph.CreateSizeVar(2);
  auto s1_0 = graph.CreateSizeVar(1);
  auto s1_1 = graph.CreateSizeVar(1);
  auto stride_1_0 = ge::ops::Zero;
  auto stride_1_1 = ge::ops::Zero;
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load0("load0");
  load0.attr.sched.axis = {z1_0.id};
  load0.x = data0.y;
  *load0.y.axis = {z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s1_0};
  *load0.y.strides = {stride_1_0};

  Load load1("load1");
  load1.attr.sched.axis = {z1_0.id};
  load1.x = data1.y;
  *load1.y.axis = {z1_0.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s1_1};
  *load1.y.strides = {stride_1_1};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z1.id};
  *concat_op.y.repeats = {s1_0 + s1_1};
  *concat_op.y.strides = {ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s1_0 + s1_1};
  *store_op.y.strides = {ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
}

TEST_F(TestOptimizerV2, SliceSliceConcatD) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(90);
  auto s2 = graph.CreateSizeVar(1);
  auto s1_0 = graph.CreateSizeVar(60);
  auto s1_1 = graph.CreateSizeVar(30);
  auto s3 = graph.CreateSizeVar(97);
  auto s4 = graph.CreateSizeVar(65);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1_1, One};
  *data0.y.strides = {s1_1, One, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1_0, One};
  *data1.y.strides = {s1_0, One, One};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1_0.id};
  load0.x = data1.y;
  *load0.y.axis = {z0.id, z1_0.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1_0};
  *load0.y.strides = {s3 * s1_0, s3};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.x = data0.y;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s4 * s1_1, s4};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1};
  *concat_op.y.strides = {s1, ge::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, ge::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_remove_pad_0 = impl_graph.FindNode("load0_remove_pad_0");
    EXPECT_EQ(load0_remove_pad_0, nullptr);
    auto load1_remove_pad_0 = impl_graph.FindNode("load1_remove_pad_0");
    EXPECT_EQ(load1_remove_pad_0, nullptr);
  }
}

TEST_F(TestOptimizerV2, LoadAlignmentInferFunc_multiple_axis_discontine) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(90);
  auto s1_0 = graph.CreateSizeVar(45);
  auto s1_1 = graph.CreateSizeVar(45);
  auto s2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(32);
  auto s3_0 = graph.CreateSizeVar(16);
  auto s4 = graph.CreateSizeVar(32);
  auto s4_0 = graph.CreateSizeVar(16);
  auto s5 = graph.CreateSizeVar(8);
  static ge::Expression Two = ge::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z3_0 = graph.CreateAxis("z3_0", s3_0);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z4_0 = graph.CreateAxis("z4_0", s4_0);
  auto z5 = graph.CreateAxis("z5", s5);

  Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *data0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *data0.y.repeats = {s0, s1, s2, s3, s4, s5};
  *data0.y.strides = {s5 * s4 * s3 * s2 * s1, s5 * s4 * s3 * s2, s5 * s4 * s3, s5 * s4, s5, One};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *data1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *data1.y.repeats ={s0, s1, s2, s3, s4, s5};
  *data1.y.strides = {s5 * s4 * s3 * s2 * s1 , s5 * s4 * s3 * s2, s5 * s4 * s3, s5 * s4, s5, One};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1_0.id, z2.id, z3_0.id, z4_0.id, z5.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1_0.id, z2.id, z3_0.id, z4_0.id, z5.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.repeats = {s0, s1_0, s2, s3_0, s4_0, s5};
  *load0.y.strides = {s5 * s4_0 * s3_0 * s2 * s1_0 * Two * Two, s5 * s4_0 * s3_0 * s2* Two * Two, s5 * s4_0 * s3_0 * Two * Two, s5 * s4_0 * Two * Two, s5 * Two * Two, Two};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1_1.id, z2.id, z3_0.id, z4_0.id, z5.id};
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1_1.id, z2.id, z3_0.id, z4_0.id, z5.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.repeats = {s0, s1_1, s2, s3_0, s4_0, s5};
  *load1.y.strides = {s5 * s4_0 * s3_0 * s2 * s1_1* Two * Two, s5 * s4_0 * s3_0 * s2 * Two * Two, s5 * s4_0 * s3_0* Two * Two, s5 * s4_0* Two * Two, s5 * Two * Two, Two};

  ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = ge::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  *concat_op.y.repeats = {s0, s1, s2, s3, s4, s5};
  *concat_op.y.strides = {s5 * s4 * s3 * s2 * s1, s5 * s4 * s3 * s2, s5 * s4 * s3, s5 * s4, s5, One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1, s2, s3, s4, s5};
  *store_op.y.strides = {s5 * s4 * s3 * s2 * s1, s5 * s4 * s3 * s2, s5 * s4 * s3, s5 * s4, s5, One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  auto unalign = optimize::UnAlignmentStrategy();
  auto concat_node = graph.FindNode("concat");
  EXPECT_NE(concat_node, nullptr);
  EXPECT_EQ(unalign.LoadAlignmentInferFunc(concat_node), ge::SUCCESS);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), ge::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_nddma = impl_graph.FindNode("load1");
    EXPECT_NE(load0_nddma, nullptr);
    EXPECT_EQ(load0_nddma->GetType(), "Nddma");
  }
}


TEST_F(TestOptimizerV2, JustMutmul) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, MutmulAndAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(TestOptimizerV2, MutmulAndBroadcastAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(TestOptimizerV2, JustMutmulBias) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  MatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, JustMutmulOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_INT8;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_INT8;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  MatMulOffset matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulOffset");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, JustMutmulBaisOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = ge::DT_INT8;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.strides = {ge::ops::Zero, ge::ops::One};
  *data3.y.repeats = {ge::ops::One, s1};
  data3.ir_attr.SetIndex(2);

  Load load3("load3");
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.x = data3.y;
  *load3.y.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_INT8;
  *load3.y.strides = {ge::ops::Zero, ge::ops::One};
  *load3.y.repeats = {ge::ops::One, s1};

  MatMulOffsetBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load3.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 8) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMulOffsetBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, JustBatchMutmul) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMul");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, BatchMutmulAndAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(TestOptimizerV2, BatchMutmulAndBroadcastAdd) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMul");
    }
  }
}

TEST_F(TestOptimizerV2, MatmulAndCastBroadcastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = cast.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT16;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1*s2, s2, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(TestOptimizerV2, JustBatchMutmulBias) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  BatchMatMulBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMulBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, JustBatchMutmulOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_INT8;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_INT8;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  BatchMatMulOffset matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMulOffset");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, JustBatchMutmulBaisOffset) {
  ge::AscGraph graph("mutmul");

  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.attr.sched.axis = {z0.id, z1.id};
  data2.y.dtype = ge::DT_FLOAT;
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.strides = {ge::ops::Zero, ge::ops::One};
  *data2.y.repeats = {ge::ops::One, s1};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.x = data2.y;
  *load2.y.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.strides = {ge::ops::Zero, ge::ops::One};
  *load2.y.repeats = {ge::ops::One, s1};

  Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = ge::DT_INT8;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.strides = {ge::ops::Zero, ge::ops::One};
  *data3.y.repeats = {ge::ops::One, s1};
  data3.ir_attr.SetIndex(2);

  Load load3("load3");
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.x = data3.y;
  *load3.y.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_INT8;
  *load3.y.strides = {ge::ops::Zero, ge::ops::One};
  *load3.y.repeats = {ge::ops::One, s1};

  BatchMatMulOffsetBias matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.offset_w = load3.y;
  matmul.bias = load2.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 8) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "BatchMatMulOffsetBias");
    }
  }

  auto codegen = codegen::Codegen(
      codegen::CodegenOptions{.tiling_lib_path = "gen_tiling.so", .tiling_lib_codegen_symbol = "CodegenTiling"});
  codegen::CodegenResult result;
  codegen.Generate(fused_scheduled_result, result);

  std::fstream tiling_func("JustMutmul_tiling_func.cpp", std::ios::out);
  std::fstream tiling_data("JustMutmul_tiling_data.h", std::ios::out);
  std::fstream kernel_func("JustMutmul_kernel_func.cpp", std::ios::out);

  tiling_func << result.tiling;
  tiling_data << result.tiling_data;
  kernel_func << result.kernel;
}

TEST_F(TestOptimizerV2, MatmulAddExpAddAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  Data data3("data3", graph);
  data3.y.dtype = ge::DT_FLOAT16;
  data3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data3.y.axis = {z0.id, z1.id, z2.id};
  data3.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data3.y.repeats = {One, One, One};
  *data3.y.strides = {Zero, Zero, Zero};
  data3.ir_attr.SetIndex(3);

  Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id, z2.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id, z2.id};
  *load3.y.strides = {s1*s2, s2, ge::ops::One};
  *load3.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT16;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  ascir_op::Exp exp("exp");
  exp.attr.sched.axis = {z0.id, z1.id, z2.id};
  exp.x = add_op.y;
  exp.y.dtype = ge::DT_FLOAT16;
  *exp.y.axis = {z0.id, z1.id, z2.id};
  *exp.y.strides = {s1*s2, s2, ge::ops::One};
  *exp.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op1("add1");
  add_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op1.x1 = matmul.y;
  add_op1.x2 = load3.y;
  add_op1.y.dtype = ge::DT_FLOAT16;
  *add_op1.y.axis = {z0.id, z1.id, z2.id};
  *add_op1.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op1.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op2("add2");
  add_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op2.x1 = exp.y;
  add_op2.x2 = add_op1.y;
  add_op2.y.dtype = ge::DT_FLOAT16;
  *add_op2.y.axis = {z0.id, z1.id, z2.id};
  *add_op2.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op2.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op2.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C1");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetName(),
            "matmul_1_ub_Y3_S0G0C2");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetName(),
            "matmul_1_ub_Y2_S0G0C3");
}

TEST_F(TestOptimizerV2, MatmulAndCastAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(1);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = cast.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C1");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetName(),
            "matmul_1_ub_Y3_S0G0C2");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetName(),
            "matmul_1_ub_Y2_S0G0C3");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(TestOptimizerV2, MatmulAndCastMultiRefsAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = cast.y;
  add_op.x2 = cast.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C1");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2].GetName(),
            "matmul_1_ub_Y3_S0G0C2");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[3].GetName(),
            "matmul_1_ub_Y2_S0G0C3");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
  }
}

TEST_F(TestOptimizerV2, MatmulAndBrcLoadMultiRefsAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1*s2, s2, ge::ops::One};
  *load2.y.repeats = {s0, s1, s2};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = load2.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op1("add1");
  add_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op1.x1 = add_op.y;
  add_op1.x2 = load2.y;
  add_op1.y.dtype = ge::DT_FLOAT;
  *add_op1.y.axis = {z0.id, z1.id, z2.id};
  *add_op1.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op1.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op1.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "VectorFunc");
    }
  }
}

TEST_F(TestOptimizerV2, MatmulAndCastBrcMultiRefsAdd) {
  ge::AscGraph graph("matmul");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast("cast");
  cast.x = matmul.y;
  cast.attr.sched.axis = {z0.id, z1.id, z2.id};
  cast.y.dtype = ge::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id, z2.id};
  *cast.y.repeats = {s0, s1, One};
  *cast.y.strides = {s1, ge::ops::One, Zero};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = cast.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = cast.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ("matmul_0_S0G1C0", fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName());
  EXPECT_EQ("matmul_1_Y0_S0G0C0", fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName());
  EXPECT_EQ("matmul_1_Y1_S0G0C1", fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName());
}

TEST_F(TestOptimizerV2, MatmulStoreAddExpAddAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One};
  *data2.y.strides = {Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.strides = {s1, ge::ops::One};
  *load2.y.repeats = {s0, s1};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT16;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = matmul.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.strides = {s1, ge::ops::One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op1("output1");
  output_op1.x = store_op1.y;
  output_op1.y.dtype = ge::DT_FLOAT16;
  output_op1.ir_attr.SetIndex(0);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = load2.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.strides = {s1, ge::ops::One};
  *add_op.y.repeats = {s0, s1};

  ascir_op::Exp exp("exp");
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.x = add_op.y;
  exp.y.dtype = ge::DT_FLOAT16;
  *exp.y.axis = {z0.id, z1.id};
  *exp.y.strides = {s1, ge::ops::One};
  *exp.y.repeats = {s0, s1};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = exp.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.strides = {s1, ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(8);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y2_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y2_S0G0C1");
}

TEST_F(TestOptimizerV2, MatmulAndBrcLoadScalarAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);
  
  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Scalar data2("scalar", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.ir_attr.SetIndex(2);

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = data2.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc1.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(TestOptimizerV2, MatmulAndLoadMultiBrcAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1, One, Zero};
  *load2.y.repeats = {s0, s1, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load2.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = brc1.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1*s2, s2, ge::ops::One};
  *abs.y.repeats = {s0, s1, s2};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = abs.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_ub_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_ub_Y3_non_db_S0G0C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1].GetName(),
            "matmul_1_ub_Y3_S0G0C1");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 1) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Nddma");
    }
  }
}

TEST_F(TestOptimizerV2, MatmulAndLoadBrcAndAbsBrcAdd) {
  ge::AscGraph graph("matmul");
  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto s2 = graph.CreateSizeVar(64);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data1.y.repeats = {One, One};
  *data1.y.strides = {Zero, Zero};
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {Zero, Zero};
  *load1.y.repeats = {One, One};

  Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data2.y.axis = {z0.id, z1.id, z2.id};
  data2.attr.api.compute_type = ComputeType::kComputeInvalid;
  *data2.y.repeats = {One, One, One};
  *data2.y.strides = {Zero, Zero, Zero};
  data2.ir_attr.SetIndex(2);

  Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.strides = {s1, One, Zero};
  *load2.y.repeats = {s0, s1, One};

  MatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Broadcast brc("brc");
  brc.x = matmul.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc.y.dtype = ge::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id, z2.id};
  *brc.y.repeats = {s0, s1, s2};
  *brc.y.strides = {s1 * s2, s2, One};

  ge::ascir_op::Abs abs("abs");
  abs.x = load2.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.strides = {s1, One, Zero};
  *abs.y.repeats = {s0, s1, One};

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = abs.y;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = ge::DT_FLOAT;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s1 * s2, s2, One};

  ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  add_op.x1 = brc.y;
  add_op.x2 = brc1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id, z2.id};
  *add_op.y.strides = {s1*s2, s2, ge::ops::One};
  *add_op.y.repeats = {s0, s1, s2};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1*s2, s2, ge::ops::One};
  *store_op.y.repeats = {s0, s1, s2};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(8);
  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetName(),
            "matmul_0_S0G1C0");
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            "matmul_1_Y0_S0G0C0");
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "MatMul");
    }
  }
  for (const auto &node : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Broadcast");
    }
  }
}