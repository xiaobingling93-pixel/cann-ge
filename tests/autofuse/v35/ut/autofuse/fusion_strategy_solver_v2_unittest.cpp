/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "common/ge_common/ge_types.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/symbolizer/symbolic.h"
#include "common/util/mem_utils.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_ops.h"
#include "graph/utils/node_utils.h"
#include "utils/auto_fuse_config.h"
#include "ascgen_log.h"
#include "autofuser.h"
#include "attribute_group/attr_group_shape_env.h"
#include "post_process/asc_backend_post_processor.h"
#include "lowering/op_helper/lower_concat_helper.h"
#include "can_fuse/graph_manager.h"
#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "platform_context.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
class UtestFusionStrategySolverV2 : public testing::Test {
 public:
  static std::shared_ptr<ge::AscGraph> CreatMulAscGraph(ge::AscGraph &graph, size_t out_num = 1, size_t in_num = 1) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::string data_name = "data" + graph.GetName();
    ge::ascir_op::Data x1(data_name.c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("load");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string scalar_name = "scalar" + graph.GetName();
    ge::ascir_op::Scalar x2(scalar_name.c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("load");
    x2Local.x = x1.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul(graph.GetName().c_str());
    mul.x1 = x1Local.y;
    mul.x2 = x2Local.y;
    mul.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul.y.repeats = {A, B, C, D, E};
    *mul.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = mul.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatMul2InputsAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::string data_name = "data" + graph.GetName();
    ge::ascir_op::Data x1(data_name.c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    data_name = "data" + graph.GetName() + "_1";
    ge::ascir_op::Data x2(data_name.c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("load");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("load_1");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Mul mul(graph.GetName().c_str());
    mul.x1 = x1Local.y;
    mul.x2 = x2Local.y;
    mul.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *mul.y.repeats = {A, B, C, D, E};
    *mul.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = mul.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatSplitAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1((graph.GetName() + "_data1").c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local((graph.GetName()+ "_load1").c_str());
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2((graph.GetName() + "_data2").c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local((graph.GetName() + "_load2").c_str());
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x3((graph.GetName() + "_data3").c_str(), graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {A, B, C, D, E};
    *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local((graph.GetName() + "_load3").c_str());
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Concat split(graph.GetName().c_str());
    split.x = {x1Local.y, x2Local.y, x3Local.y};
    split.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *split.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *split.y.repeats = {A, B, C, D, E};
    *split.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store((graph.GetName() + "_store").c_str());
    x_store.x = split.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out((graph.GetName() + "_out").c_str());
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode((graph.GetName() + "_out").c_str());
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }
  static std::shared_ptr<ge::AscGraph> CreatSplitDoubleOutPutsAscGraph(ge::AscGraph &graph,
                                                                       const std::vector<int64_t> &split_dims,
                                                                       size_t split_dim) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::vector<std::vector<ge::Expression>> output_dim_sizes (split_dims.size(), {A, B, C, D, E});
    std::vector<ge::Expression> input_dim_sizes {A, B, C, D, E};
    int64_t total_size = 0;
    for (size_t i = 0U; i < split_dims.size(); ++i) {
      auto e_size = graph.CreateSizeVar(split_dims[i]);
      total_size = total_size + split_dims[i];
      output_dim_sizes[i][split_dim] = e_size;
    }
    input_dim_sizes[split_dim] = graph.CreateSizeVar(total_size);

    ge::ascir_op::Data x1((graph.GetName() + "_data1").c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = input_dim_sizes;
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local((graph.GetName()+ "_load1").c_str());
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = input_dim_sizes;
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Split split(graph.GetName().c_str());
    split.InstanceOutputy(2);
    split.x = x1Local.y;
    split.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    size_t idx = 0U;
    // split.DynamicOutputRegister("y",2);

    split.y[0].dtype = ge::DT_FLOAT16;
    *split.y[0].axis = {a.id, b.id, c.id, d.id, e.id};
    *split.y[0].repeats = output_dim_sizes[0];
    *split.y[0].strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    ge::ascir_op::Store x_store((graph.GetName() + "_store0" ).c_str());
    x_store.x = split.y[0];
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = output_dim_sizes[0];
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out((graph.GetName() + "_out0" ).c_str());
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = output_dim_sizes[0];
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    idx++;

    split.y[1].dtype = ge::DT_FLOAT16;
    *split.y[1].axis = {a.id, b.id, c.id, d.id, e.id};
    *split.y[1].repeats = output_dim_sizes[1];
    *split.y[1].strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store1((graph.GetName() + "_store1" ).c_str());
    x_store1.x = split.y[1];
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = output_dim_sizes[1];
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1((graph.GetName() + "_out1").c_str());
    x_out1.x = x_store1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = output_dim_sizes[1];
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    idx++;

    // for (auto y : split.y) {
    //   *y.axis = {a.id, b.id, c.id, d.id, e.id};
    //   *y.repeats = {A, B, C, D, E};
    //   *y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    //   ge::ascir_op::Store x_store((graph.GetName() + "_store" + std::to_string(idx)).c_str());
    //   x_store.x = y;
    //   x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    //   x_store.attr.sched.loop_axis = c.id;
    //   *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    //   *x_store.y.repeats = {A, B, C, D, E};
    //   *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    //
    //   ge::ascir_op::Output x_out((graph.GetName() + "_out" + std::to_string(idx)).c_str());
    //   x_out.x = x_store.y;
    //   x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    //   x_out.attr.sched.loop_axis = c.id;
    //   *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    //   *x_out.y.repeats = {A, B, C, D, E};
    //   *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    //   idx++;
    // }

    auto x_out_node = graph.FindNode((graph.GetName() + "_out0").c_str());
    auto x_out_node1= graph.FindNode((graph.GetName() + "_out1").c_str());
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0},{x_out_node1,1}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

    static std::shared_ptr<ge::AscGraph> CreatSplitAscGraphSame(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1((graph.GetName() + "_data1").c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, Symbol(2) * C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local((graph.GetName()+ "_load1").c_str());
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, Symbol(2) * C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store((graph.GetName() + "_store").c_str());
    x_store.x = x1Local.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out((graph.GetName() + "_out").c_str());
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode((graph.GetName() + "_out").c_str());
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatAddAscGraph(ge::AscGraph &graph, size_t out_num = 1, size_t in_num = 1) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::string data_name = "data" + graph.GetName();
    ge::ascir_op::Data x1(data_name.c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("load");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add(graph.GetName().c_str());
    if (in_num == 2) {
      ge::ascir_op::Data x2(data_name.c_str(), graph);
      x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      x2.attr.sched.loop_axis = c.id;
      *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
      *x2.y.repeats = {A, B, C, D, E};
      *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

      ge::ascir_op::Load x2Local("load");
      x2Local.x = x2.y;
      x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
      *x2Local.y.repeats = {A, B, C, D, E};
      *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

      add.x1 = x1Local.y;
      add.x2 = x2Local.y;
      add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
      *add.y.repeats = {A, B, C, D, E};
      *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    } else {
      add.x1 = x1Local.y;
      add.x2 = x1Local.y;
      add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
      *add.y.repeats = {A, B, C, D, E};
      *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    }

    ge::ascir_op::Store x_store("store");
    x_store.x = add.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    if (out_num == 2) {
      ge::ascir_op::Store x_store1("store1");
      x_store1.x = add.y;
      x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      x_store1.attr.sched.loop_axis = c.id;
      *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
      *x_store1.y.repeats = {A, B, C, D, E};
      *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

      ge::ascir_op::Output x_out1("out1");
      x_out1.x = x_store1.y;
      x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      x_out1.attr.sched.loop_axis = c.id;
      *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
      *x_out1.y.repeats = {A, B, C, D, E};
      *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
      auto x_out_node1 = graph.FindNode("out1");
      std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 1}};
      compute_graph->SetOutputSize(2U);
      compute_graph->SetGraphOutNodesInfo(output_nodes);
    } else {
      std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
      compute_graph->SetOutputSize(1U);
      compute_graph->SetGraphOutNodesInfo(output_nodes);
    }
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatAdd2InputsAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::string data_name1 = "data1_" + graph.GetName();
    ge::ascir_op::Data x1(data_name1.c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string load1 = "load1_" + graph.GetName();
    ge::ascir_op::Load x1Local(load1.c_str());
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string data_name2 = "data2_" + graph.GetName();
    ge::ascir_op::Data x2(data_name2.c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string load2 = "load2_" + graph.GetName();
    ge::ascir_op::Load x2Local(load2.c_str());
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Add add(graph.GetName().c_str());
    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string store = "store" + graph.GetName();
    ge::ascir_op::Store x_store(store.c_str());
    x_store.x = add.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string out = "out" + graph.GetName();
    ge::ascir_op::Output x_out(out.c_str());
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode(out.c_str());
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatAbsAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::string data_name = "data" + graph.GetName();
    ge::ascir_op::Data x1(data_name.c_str(), graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string load = "load" + graph.GetName();
    ge::ascir_op::Load x1Local(load.c_str());
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Abs add(graph.GetName().c_str());
    add.x = x1Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string store = "store" + graph.GetName();
    ge::ascir_op::Store x_store(store.c_str());
    x_store.x = add.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    std::string out = "out" + graph.GetName();
    ge::ascir_op::Output x_out(out.c_str());
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode(out.c_str());
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatConcatAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, E};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x1Local("load1");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, E};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("load2");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Data x3("data3", graph);
    x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x3.attr.sched.loop_axis = c.id;
    *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3.y.repeats = {A, B, C, D, E};
    *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x3Local("load3");
    x3Local.x = x3.y;
    x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x3Local.y.repeats = {A, B, C, D, E};
    *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Concat concat(graph.GetName().c_str());
    concat.x = {x1Local.y, x2Local.y, x3Local.y};
    concat.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *concat.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *concat.y.repeats = {A, B, C, D, E};
    *concat.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = concat.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatConcatAscGraph(ge::AscGraph &graph,
                                                           const std::vector<int64_t> &concat_dims,
                                                           size_t concat_dim) {
    auto ONE = Symbol(1);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    std::vector<std::vector<ge::Expression>> intput_dim_sizes (concat_dims.size(), {A, B, C, D, E});
    std::vector<ge::Expression> output_dim_sizes {A, B, C, D, E};
    int64_t total_size = 0;
    for (size_t i = 0U; i < concat_dims.size(); ++i) {
      auto e_size = graph.CreateSizeVar(concat_dims[i]);
      total_size = total_size + concat_dims[i];
      intput_dim_sizes[i][concat_dim] = e_size;
    }
    output_dim_sizes[concat_dim] = graph.CreateSizeVar(total_size);

    std::vector<std::shared_ptr<ge::ascir_op::Data>> data_ops;
    std::vector<AscOpOutput> concat_inputs;
    for (size_t i = 0U; i < concat_dims.size(); ++i) {
      auto e_size = ge::Symbol(concat_dims[i]);
      auto x1 = std::make_shared<ge::ascir_op::Data>(("data_" + std::to_string(i)).c_str(), graph);
      x1->attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
      x1->attr.sched.loop_axis = c.id;
      *x1->y.axis = {a.id, b.id, c.id, d.id, e.id};
      *x1->y.repeats = intput_dim_sizes[i];
      data_ops.emplace_back(x1);
      concat_inputs.emplace_back(x1->y);
    }

    ge::ascir_op::Concat concat(graph.GetName().c_str());
    concat.x = concat_inputs;
    concat.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *concat.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *concat.y.repeats = output_dim_sizes;

    ge::ascir_op::Store x_store("store");
    x_store.x = concat.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = output_dim_sizes;

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = output_dim_sizes;

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatBroadcastAbsAscGraphWith6Axis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");
    const Expression F = graph.CreateSizeVar("F");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);
    auto f = graph.CreateAxis("F", F);

    ge::ascir_op::Data x1("x1_broadcast_abs", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *x1.y.repeats = {A, B, C, D, E, ONE};
    *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

    ge::ascir_op::Load x1Local("x1Local_broadcast_abs");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *x1Local.y.repeats = {A, B, C, D, E, ONE};
    *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

    ge::ascir_op::Abs abs("add_broadcast_abs");
    abs.x = x1Local.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *abs.y.repeats = {A, B, C, D, E, F};
    *abs.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

    ge::ascir_op::Store x_store("x_store_broadcast_abs");
    x_store.x = abs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *x_store.y.repeats = {A, B, C, D, E, F};
    *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

    ge::ascir_op::Output x_out("x_out_broadcast_abs");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
    *x_out.y.repeats = {A, B, C, D, E, F};
    *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};
    auto x_out_node = graph.FindNode("x_out_broadcast_abs");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatBroadcastAbsAscGraphWith5Axis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_broadcast_abs", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, ONE};
    *x1.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Load x1Local("x1Local_broadcast_abs");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, ONE};
    *x1Local.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Abs abs("add_broadcast_abs");
    abs.x = x1Local.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("x_store_broadcast_abs");
    x_store.x = abs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("x_out_broadcast_abs");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_out_broadcast_abs");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatBroadcastAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("x1_broadcast_abs", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = c.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1.y.repeats = {A, B, C, D, ONE};
    *x1.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Load x1Local("x1Local_broadcast_abs");
    x1Local.x = x1.y;
    x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x1Local.y.repeats = {A, B, C, D, ONE};
    *x1Local.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Broadcast abs("add_broadcast_abs");
    abs.x = x1Local.y;
    abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *abs.y.repeats = {A, B, C, D, E};
    *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("x_store_broadcast_abs");
    x_store.x = abs.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("x_out_broadcast_abs");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node = graph.FindNode("x_out_broadcast_abs");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraph(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar("E");

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = b.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id};
    *x1.y.repeats = {A, B, D, E};
    *x1.y.strides = {B * D * E, D * E, E, ONE};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = e.id;
    *x2.y.axis = {a.id, b.id};
    *x2.y.repeats = {B, C};
    *x2.y.strides = {C, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = {x1.y};
    gather.x2 = {x2.y};
    gather.ir_attr.SetAxis(1);
  gather.ir_attr.SetNegative_index_support(false);
    gather.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *gather.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *gather.y.repeats = {A, B, C, D, E};
    *gather.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Store x_store("store");
    x_store.x = gather.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, E};
    *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, E};
    *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static std::shared_ptr<ge::AscGraph> CreatGatherAscGraphWithInvlaidAxis(ge::AscGraph &graph) {
    auto ONE = Symbol(1);
    auto ZERO = Symbol(0);
    const Expression A = graph.CreateSizeVar("A");
    const Expression B = graph.CreateSizeVar("B");
    const Expression C = graph.CreateSizeVar("C");
    const Expression D = graph.CreateSizeVar("D");
    const Expression E = graph.CreateSizeVar(1);

    auto a = graph.CreateAxis("A", A);
    auto b = graph.CreateAxis("B", B);
    auto c = graph.CreateAxis("C", C);
    auto d = graph.CreateAxis("D", D);
    auto e = graph.CreateAxis("E", E);

    ge::ascir_op::Data x1("data1", graph);
    x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x1.attr.sched.loop_axis = b.id;
    *x1.y.axis = {a.id, b.id, c.id, d.id};
    *x1.y.repeats = {A, B, D, ONE};
    *x1.y.strides = {B * D * E, D * E, ONE, ZERO};

    ge::ascir_op::Data x2("data2", graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = e.id;
    *x2.y.axis = {a.id, b.id};
    *x2.y.repeats = {B, C};
    *x2.y.strides = {C, ONE};

    ge::ascir_op::Gather gather(graph.GetName().c_str());
    gather.x1 = {x1.y};
    gather.x2 = {x2.y};
    gather.ir_attr.SetAxis(1);
  gather.ir_attr.SetNegative_index_support(false);
    gather.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *gather.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *gather.y.repeats = {A, B, C, D, ONE};
    *gather.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Store x_store("store");
    x_store.x = gather.y;
    x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store.attr.sched.loop_axis = c.id;
    *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store.y.repeats = {A, B, C, D, ONE};
    *x_store.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    ge::ascir_op::Output x_out("out");
    x_out.x = x_store.y;
    x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out.attr.sched.loop_axis = c.id;
    *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out.y.repeats = {A, B, C, D, ONE};
    *x_out.y.strides = {B * C * D, C * D, D, ONE, ZERO};

    auto x_out_node = graph.FindNode("out");
    auto compute_graph = x_out_node->GetOwnerComputeGraph();

    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
    return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
  }

  static Status SetAttrsGroup(const NodePtr &node) {
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    auto attr = GetOrCreateAutoFuseAttrs(op_desc);
    GE_ASSERT_NOTNULL(attr);

    ge::AscGraph add_graph(node->GetName().c_str());
    if (node->GetName() == "D" || node->GetName() == "A" || node->GetName() == "R" || node->GetName() == "R1" || node->GetName() == "E1") {
      attr->SetAscGraph(CreatAbsAscGraph(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName() == "E" || node->GetName() == "C") {
      attr->SetAscGraph(CreatAdd2InputsAscGraph(add_graph), loop::FuseType::kPointwise);
    } else if ((node->GetName() == "F2") || (node->GetName() == "G2") || (node->GetName() == "H2")) {
      attr->SetAscGraph(CreatAddAscGraph(add_graph, 1U, 2U));
    } else if (node->GetName().find("Concat") != std::string::npos) {
      if (node->GetName() == "Concat3Inputs") {
        attr->SetAscGraph(CreatConcatAscGraph(add_graph, {3, 3, 3}, 4), loop::FuseType::kConcat);
      } else if (node->GetName() == "Concat2InputsFirstDim") {
        attr->SetAscGraph(CreatConcatAscGraph(add_graph, {3, 3}, 0), loop::FuseType::kConcat);
      } else if (node->GetName() == "Concat2Inputs") {
        attr->SetAscGraph(CreatConcatAscGraph(add_graph, {3, 3}, 4), loop::FuseType::kConcat);
      } else if (node->GetName() == "Concat4Inputs") {
        attr->SetAscGraph(CreatConcatAscGraph(add_graph, {3, 3, 3, 3}, 4), loop::FuseType::kConcat);
      } else if (node->GetName() == "ConcatFirstDim") {
        attr->SetAscGraph(CreatConcatAscGraph(add_graph, {1, 1, 1}, 0), loop::FuseType::kConcat);
      } else {
        attr->SetAscGraph(CreatConcatAscGraph(add_graph), loop::FuseType::kConcat);
      }
    } else if (node->GetName().find("Gather") != std::string::npos) {
      attr->SetAscGraph(CreatGatherAscGraph(add_graph), loop::FuseType::kGather);
    } else if (node->GetName().find("GatherWithInvaildAxis") != std::string::npos) {
      attr->SetAscGraph(CreatGatherAscGraphWithInvlaidAxis(add_graph), loop::FuseType::kGather);
    } else if (node->GetName().find("BroadcastWith5Axis") != std::string::npos) {
      attr->SetAscGraph(CreatBroadcastAbsAscGraphWith5Axis(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName().find("BroadcastWith6Axis") != std::string::npos) {
      attr->SetAscGraph(CreatBroadcastAbsAscGraphWith6Axis(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName() == "Reduce") {
      attr->SetAscGraph(CreatAddAscGraph(add_graph), loop::FuseType::kReduction);
    } else if (node->GetName() == "EletAbs") {
      attr->SetAscGraph(CreatAddAscGraph(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName() == "EletAdd") {
      attr->SetAscGraph(CreatAddAscGraph(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName() == "A1") {
      attr->SetAscGraph(CreatAddAscGraph(add_graph, 2, 2));
    } else if (node->GetName().find("SliceNode") != std::string::npos) {
      attr->SetAscGraph(CreatSplitAscGraph(add_graph), loop::FuseType::kSliceSplit);
    } else if (node->GetName().find("Slice") != std::string::npos ) {
      attr->SetAscGraph(CreatAddAscGraph(add_graph), loop::FuseType::kSliceSplit);
    } else if (node->GetName().find("SplitFirstDim") != std::string::npos){
      attr->SetAscGraph(CreatSplitDoubleOutPutsAscGraph(add_graph, {3, 3}, 0), loop::FuseType::kSplit);
    } else if (node->GetName().find("Split") != std::string::npos){
      attr->SetAscGraph(CreatSplitDoubleOutPutsAscGraph(add_graph, {3, 3}, 4), loop::FuseType::kSplit);
    } else if (node->GetName().find("Broadcast") != std::string::npos) {
      attr->SetAscGraph(CreatBroadcastAscGraph(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName().find("NonPointwise") != std::string::npos) {
      attr->SetAscGraph(CreatBroadcastAscGraph(add_graph));
    } else if (node->GetName().find("Mul2Inputs") != std::string::npos) {
      attr->SetAscGraph(CreatMul2InputsAscGraph(add_graph), loop::FuseType::kPointwise);
    } else if (node->GetName().find("Mul") != std::string::npos) {
      attr->SetAscGraph(CreatMulAscGraph(add_graph), loop::FuseType::kPointwise);
    } else {
      attr->SetAscGraph(CreatAddAscGraph(add_graph), loop::FuseType::kPointwise);
    }

    for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(out_anchor);
      const auto node_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(node_desc);
      auto output_tensor_desc = node_desc->MutableOutputDesc(out_anchor->GetIdx());
      gert::SymbolShape symbol_shape1({Symbol(1), Symbol(2), Symbol(3), Symbol(4)});
      gert::SymbolShape symbol_shape2({Symbol(2), Symbol(2), Symbol(3), Symbol(4)});
      gert::SymbolShape symbol_shape_end_dim_is_one({Symbol(2), Symbol(2), Symbol(3), Symbol(1)});
      if (node->GetName().find("KSlice") != std::string::npos) {
        symbol_shape1 = symbol_shape_end_dim_is_one;
        symbol_shape2 = symbol_shape_end_dim_is_one;
      }
      gert::SymbolShape symbol_shape;
      if (node_desc->GetName() == "A") {
        symbol_shape = symbol_shape2;
      } else {
        symbol_shape = symbol_shape1;
      }
      output_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          symbol_shape;
    }
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      const auto node_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(node_desc);
      auto input_tensor_desc = node_desc->MutableInputDesc(in_anchor->GetIdx());

      const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        GELOGW("Node:%s in_anchor:%u peer_out_anchor is nullptr.", node->GetNamePtr(), in_anchor->GetIdx());
        continue;
      }
      const auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(peer_node);
      const auto peer_node_desc = peer_node->GetOpDesc();
      GE_ASSERT_NOTNULL(peer_node_desc);
      const auto output_tensor_desc = peer_node_desc->MutableOutputDesc(peer_out_anchor->GetIdx());
      GE_ASSERT_NOTNULL(output_tensor_desc);
      const auto attr_group = output_tensor_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
      GE_ASSERT_NOTNULL(attr_group);

      input_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          attr_group->symbolic_tensor.MutableOriginSymbolShape();
    }
    // 给节点添加origin_input_names和origin_output_names
    if (node->GetType() == kAscBackendType) {
      std::vector<std::pair<std::string, int32_t>> origin_input_names;
      for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
        origin_input_names.emplace_back("origin_input" + std::to_string(i), i);
      }
      std::vector<std::pair<std::string, int32_t>> origin_output_names;
      for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
        origin_output_names.emplace_back("origin_output" + std::to_string(i), i);
      }
      GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_input_names_ = origin_input_names;
      GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_output_names_ = origin_output_names;
    }
    return SUCCESS;
  }

  class MockFusionDecider : public AscBackendFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }

    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      if (!MockCanFuse(node1) || !MockCanFuse(node2)) {
        return false;
      }
      return true;
    }
    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      if (!MockCanFuse(node1) || !MockCanFuse(node2)) {
        return false;
      }
      return true;
    }
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      auto op_desc = node1->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      GetOrCreateAutoFuseAttrs(op_desc);
      op_desc = node2->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(op_desc);
      GetOrCreateAutoFuseAttrs(op_desc);
      NodeFuseInfo node_fuse_info;
      GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
      auto new_node = FuseNode(node1, node2, nullptr, node_fuse_info, counter);
      GE_ASSERT_NOTNULL(new_node);
      if (new_node != nullptr) {
        SetAttrsGroup(new_node);
      }
      return new_node;
    }

   protected:
    bool MockCanFuse(const NodePtr &node) {
      if ((node->GetType() == "Data") || (node->GetType() == "NetOutput") || (node->GetType() == "Const")) {
        return false;
      }
      return true;
    }
  };

  bool CheckFuseNodes(const ComputeGraphPtr &graph, const std::map<size_t, std::set<std::string>> &expect_nodes) {
    std::map<size_t, std::set<std::string>> check_nodes;
    size_t count = 0U;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() != kFusedAscBackendType) {
        continue;
      }

      ComputeGraphPtr fused_graph;
      EXPECT_EQ(BackendUtils::GetNodeFusedGraph(node, fused_graph), SUCCESS);
      for (const auto &fused_node : fused_graph->GetAllNodes()) {
        if (fused_node->GetType() != kAscBackendType) {
          continue;
        }

        ComputeGraphPtr asc_graph;
        EXPECT_EQ(BackendUtils::GetNodeFusedGraph(fused_node, asc_graph), SUCCESS);
        for (const auto &asc_node : asc_graph->GetAllNodes()) {
          if ((asc_node->GetType() != "Data") && (asc_node->GetType() != "Load") && (asc_node->GetType() != "Store")
              && (asc_node->GetType() != "Output")) {
            check_nodes[count].insert(asc_node->GetName());
          }
        }
      }
      count++;
    }

    EXPECT_EQ(expect_nodes, check_nodes);
    return expect_nodes == check_nodes;
  }

 protected:
  void SetUp() override {
    RegisterAllOpCreator();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    setenv("ENABLE_LOWER_MATMUL", "true", 1);

    PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    PlatformContext::GetInstance().Reset();
    RuntimeStub::Reset();
  }
};

/*
 *  data data
 *    |   /
 *  gather
 *     |
 *  reduce
 *     |
 *  netoutput
 */
TEST_F(UtestFusionStrategySolverV2, Fuse_gather_reduce) {
  class MockFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Gather");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(gather)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);
}

/*
 *  data data
 *    |   /
 *  gather
 *     |
 *  broadcast
 *     |
 *  reduce
 *     |
 *  netoutput
 */
TEST_F(UtestFusionStrategySolverV2, Fuse_gather_reduce_broadcast) {
  class MockFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Gather");
  auto broadcast = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("Broadcast");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,1}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(gather)->EDGE(0, 0)->NODE(broadcast));
    CHAIN(NODE(broadcast)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);
}

/*
 *     data
 *     / | \
 *    A  B  C
 *    |  |  |
 *    |  M  |
 *    |  |  |
 *    D  E  F
 *   /\  |  /
 *  H  concat
 *   \    |  |
 *    \   G  P
 *     \  | /
 *    netoutput
 */
TEST_F(UtestFusionStrategySolverV2, Fuse_concat_backward) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("B");
  auto m = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
    .OutNames({"y"}).Build("Gather");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("C");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Reduce");
  auto h = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("H");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto p = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("NonPointwise");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(h));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(data)->EDGE(1, 0)->NODE(b)->NODE(m)->NODE(e)->EDGE(0, 1)->NODE(concat));
      CHAIN(NODE(data)->EDGE(2, 0)->NODE(c)->EDGE(0, 0)->NODE(f)->EDGE(0, 2)->NODE(concat));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(h)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE(p)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 6U, post_nodes_size);

  // 预期A，B，D，E，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("A");
  expect_nodes[0].insert("Gather");
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("G");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

TEST_F(UtestFusionStrategySolverV2, Fuse_concat_backward_merge_split) {
  PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) override {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto data_1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data_1");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("B");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("SplitFirstDim");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
      .OutNames({"y"}).Build("Mul2Inputs");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
      .OutNames({"y"}).Build("Concat2Inputs");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(b)->EDGE(0, 1)->NODE(concat));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(split0)->EDGE(0, 1)->NODE(g));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  graph->TopologicalSorting();
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  const FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("B");
  // split canfuse新方案里, Mul2Inputs、SplitFirstDim两个节点先融合成新AscBackend了
  expect_nodes[0].insert("Concat2Inputs");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

TEST_F(UtestFusionStrategySolverV2, Fuse_concat_backward_no_merge_split) {
  PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) override {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto data_1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data_1");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("B");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("Split");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
      .OutNames({"y"}).Build("Mul2Inputs");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
      .OutNames({"y"}).Build("Concat2InputsFirstDim");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(b)->EDGE(0, 1)->NODE(concat));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(data_1)->EDGE(0, 0)->NODE(split0)->EDGE(0, 1)->NODE(g));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  graph->TopologicalSorting();
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  const FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("B");
  // 新方案里Mul2Inputs节点先和split融合成新AscBackend了
  expect_nodes[0].insert("Concat2InputsFirstDim");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}
} // namespace ge
