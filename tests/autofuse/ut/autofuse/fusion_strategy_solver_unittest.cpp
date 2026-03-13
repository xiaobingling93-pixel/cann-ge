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

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
    .OutNames({"y"}).Build("Data");
auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
    .Build("A");
auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
    .OutNames({"y"}).Build("E");
auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
    .OutNames({"y"}).Build("D");
auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"ref", "value"})
    .OutNames({"ref"}).Build("C");
vector<int32_t> data_value(1 * 2 * 3 * 4, 0);
GeTensorDesc data_tensor_desc(GeShape({1,2,3,4}), FORMAT_NCHW, DT_INT32);
GeTensorPtr tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value.data(), sizeof(int32_t));
auto b = OP_CFG("Data").InCnt(0).OutCnt(1).Weight(tensor).Build("B");

class UtestFusionStrategySolver : public testing::Test {
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
  void SetUp() {
    RegisterAllOpCreator();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    setenv("ENABLE_LOWER_MATMUL", "true", 1);
  }
  void TearDown() {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

/*
 *    data
 *     |
 *     A      B
 *    |  \  /
 *    |   C
 *    |   |
 *    |   D
 *    |  /c
 *    E
 *     |
 *   netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_none) {
  class NoneMockFusionDecider : public MockFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }
    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      return false;
    }
    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      return CanFuseVertical(node1, node2);
    }
  };

  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d)->Ctrl()->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver  fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new NoneMockFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Fuse_all) {
  class AllMockFusionDecider : public MockFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }
    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      if ((node1->GetName() == "A") && (node2->GetName() == "E")) {
        return true;
      } else if ((node1->GetName() == "C") && (node2->GetName() == "D")) {
        return true;
      } else if ((node1->GetType() == "FusedAscBackend") && (node2->GetType() == "FusedAscBackend")) {
        return true;
      } else {
        return false;
      }
    }
    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      return (CanFuseVertical(node1, node2));
    }
  };
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AllMockFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

/*
 *   data
 *     |
 *     A         B
 *    |  \     /
 *    |     C
 *    |     |
 *    |     G
 *    |     |
 *    |     H
 *    |     |
 *    |     K
 *    |     |
 *    |     D
 *    |    /
 *     E
 *      |
 *      F
 *     netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_cycle) {
  class CycleMockFusionDecider : public MockFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }
    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      if ((node1->GetName() == "A") && (node2->GetName() == "C")) {
        return true;
      } else if ((node1->GetName() == "E") && (node2->GetName() == "F")) {
        return true;
      } else if ((node1->GetName() == "H") && (node2->GetName() == "K")) {
        return true;
      } else if ((node1->GetType() == "FusedAscBackend") && (node2->GetType() == "FusedAscBackend")) {
        return true;
      } else {
        return false;
      }
    }
    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      return CanFuseVertical(node1, node2);
    }
  };
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("F");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto h = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("H");
  auto k = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("K");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->EDGE(0, 0)->NODE(f)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(g)->NODE(h)->NODE(k)->NODE(d)->EDGE(0, 1)->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new CycleMockFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // {A,C} {H,K} {E,F} can fuse, {AC, EF} will create cycle
  EXPECT_EQ(pre_nodes_size - 3 , post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Fuse_failed_dump_cache_graph) {
  setenv("DUMP_GE_GRAPH", "1", 1);
  setenv("DUMP_GRAPH_LEVEL", "1", 1);
  setenv("DUMP_GRAPH_PATH", "./", 1);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  class FuseMockFusionDecider : public MockFusionDecider {
    Status CacheGraphBeforeMerge(const NodePtr &node1, const NodePtr &node2) {
      if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
        return SUCCESS;
      }

      GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node1));
      GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(node2));
      GE_ASSERT_SUCCESS(BackendUtils::CacheCurrentGraphName(node1->GetName(), node2->GetName()));
      return SUCCESS;
    }
    Status CacheGraphAfterMerge(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2,
                                const ComputeGraphPtr &merged_graph) {
      if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
        return SUCCESS;
      }

      // 后处理的子图dump后需要dump融合过程，如果没有在融合后缓存new_node下一次融合就没法通过node1或者node2来缓存这一次new_node
      GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(new_node->GetName(), merged_graph));
      // FusedAscBackend的子图dump后需要dump FusedAscBackend的融合过程
      GE_ASSERT_SUCCESS(BackendUtils::CacheGraph(merged_graph->GetName(), merged_graph));
      // 缓存dump图映射,如果是FusedAscBackend，在子图融合的时候需要找到融合获得FusedAscBackend的过程dump图
      GE_ASSERT_SUCCESS(BackendUtils::AddMergeGraphMap(
          new_node->GetName(), node1->GetName(), node2->GetName(),
          merged_graph->GetName()));  // 不能直接使用node1的子图graph1来获取name，中间流程会更改graph1的名字
      return SUCCESS;
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
      // 异常时dump图的缓存融合前dump图流程
      GE_ASSERT_SUCCESS(CacheGraphBeforeMerge(node1, node2));
      auto new_node = FuseNode(node1, node2, nullptr, node_fuse_info, counter);
      // 异常时dump图的缓存融合后dump图流程
      ComputeGraphPtr merged_graph;
      GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(new_node, merged_graph));
      GE_ASSERT_SUCCESS(CacheGraphAfterMerge(new_node, node1, node2, merged_graph));
      GE_ASSERT_NOTNULL(new_node);
      if (new_node != nullptr) {
        SetAttrsGroup(new_node);
      }
      if (node1->GetName() == "G") {
        return nullptr;
      } else {
        return new_node;
      }
    }
  };
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("F");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto h = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("H");
  auto k = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("K");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->EDGE(0, 0)->NODE(f)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(g)->NODE(h)->NODE(k)->NODE(d)->EDGE(0, 1)->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new FuseMockFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  ASSERT_NE(fusion_strategy_solver.Fuse(graph), SUCCESS);

  auto &manager = FusionGraphManager::GetInstance();
  // FusedAscBackend子图融合时缓存正在融合图的名字
  manager.CacheCurrentGraphName("graph_name1", "graph_name2", "origin_graph_name");
  // dump CacheCurrentGraphName缓存的对应图
  (void)(manager.DumpCurrentGraphAndSubgraphs(kCanFuseDir));
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  unsetenv("DUMP_GE_GRAPH");
  unsetenv("DUMP_GRAPH_LEVEL");
  unsetenv("DUMP_GRAPH_PATH");
}

/*
 *    data
 *     |
 *     A         B
 *    |  \     /
 *    |     C
 *    |     |
 *    |     D
 *    |    /
 *     E
 *      |
 *     netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_exceed_max_fusion_size_16) {
auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
    .OutNames({"y"}).Build("Data");
auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
    .Build("A");
auto b = OP_CFG("Data").InCnt(0).OutCnt(1).Weight(tensor).Build("B");
auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
    .OutNames({"y"}).Build("D");
auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"ref", "value"})
    .OutNames({"ref"}).Build("C");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d)->EDGE(0, 1)->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 16U;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Fuse_exceed_max_fusion_size_22) {
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Data");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
      .Build("A");
  auto b = OP_CFG("Data").InCnt(0).OutCnt(1).Weight(tensor).Build("B");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1)
      .InNames({"ref", "value"}).OutNames({"ref"}).Build("C");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  DEF_GRAPH(g) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d)->EDGE(0, 1)->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 21U;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
}

/*
 *      data
 *       |
 *       A
 *    / | \ \
 *   B  C  D  E
 *   \  |  / /
 *       H
 *       |
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_increase_peak_emory) {
  class MockFusionDecider : public AscBackendFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }
    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      return false;
    }
    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      if ((node1->GetName() == "B") && (node2->GetName() == "E")) {
        return true;
      }
      return false;
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                  .OutNames({"y"}).Build("data");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("C");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  auto h = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(4).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("H");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE(h)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->EDGE(0, 1)->NODE(h));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(d)->EDGE(0, 2)->NODE(h));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->EDGE(0, 3)->NODE(h));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  int64_t old_max_proximity = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_proximity;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_proximity = 2U;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_proximity = old_max_proximity;
}

TEST_F(UtestFusionStrategySolver, Fuse_fail) {
  class FailMockFusionDecider : public MockFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return nullptr;
    }
  };

  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d)->EDGE(0, 1)->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new FailMockFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_NE(fusion_strategy_solver.Fuse(graph), SUCCESS);
}

TEST_F(UtestFusionStrategySolver, Fuse_graph_has_group_attr) {
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d)->EDGE(0, 1)->NODE(e));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto attr = GetOrCreateAutoFuseAttrs(graph);
  GetInterAttrs(attr).possible_fusion_nodes.insert(std::make_pair(graph->FindNode("A"), graph->FindNode("C")));
  GetInterAttrs(attr).decider = std::move(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
}

/*
 *     data
 *     / | \
 *    A  B  C
 *    |  |  |
 *    D  E  F
 *   /\  |  /
 *  H  concat
*    \    |  |
 *    \   G  P
 *     \  | /
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_concat) {
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
      .OutNames({"y"}).Build("P");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(h));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(data)->EDGE(1, 0)->NODE(b)->EDGE(0, 0)->NODE(e)->EDGE(0, 1)->NODE(concat));
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
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 6U, post_nodes_size);

  // 预期A，B，D，E，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("A");
  expect_nodes[0].insert("B");
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

/*
 *     data
 *     / | \
 *    D  E  F
 *   /\  |  /
 * H<--concat
 *   \    |
 *    \   G
 *     \  |
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_concat_and_other) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
      .OutNames({"y"}).Build("data");
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
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(h));
      CHAIN(NODE(concat)->CTRL_EDGE()->NODE(h));
      CHAIN(NODE(data)->EDGE(1, 0)->NODE(e)->EDGE(0, 1)->NODE(concat));
      CHAIN(NODE(data)->EDGE(2, 0)->NODE(f)->EDGE(0, 2)->NODE(concat));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(h)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 1U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);

  // 预期D，E，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

/*
 *      data
 *       |
 *       D
 *      / \
 *    E     G
 *    |     |
 * concat  concat
 *     \   /
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Not_Fuse_concat_and_concat) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  auto concat1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat1");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(e)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE(concat1));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(concat1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);

  // 预期D,E，Concat融合在一起, G，Concat1融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("G");
  expect_nodes[0].insert("Concat1");
  expect_nodes[1].insert("E");
  expect_nodes[1].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

/*
 *      data
 *       |
 *       D
 *      / \
 *    E     G
 *    |     |
 * concat  reduce
 *     \   /
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Not_Fuse_concat_and_reduce) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Reduce");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(e)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE(f));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(f)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // 预期D,E，Concat融合在一起, G，Reduce融合在一起
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

/*
 *      data
 *       |
 *       D
 *      / \
 *    E     G
 *    |     |
 * reduce--->concat
 *     \   /
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Not_Fuse_reduce_and_concat) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Reduce");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(e)->EDGE(0, 0)->NODE(f));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(f)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(f)->CTRL_EDGE()->NODE(concat));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();

  // 预期D,E，Reduce融合在一起, G，Concat融合在一起
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

/*
 *     data
 *     / | \
 *    D  E  F
 *    \  |  /
 *     concat
 *       |
 *       G
 *       |
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_concat_no_limit) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT16, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT16, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("F");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(data)->EDGE(1, 0)->NODE(e)->EDGE(0, 1)->NODE(concat));
      CHAIN(NODE(data)->EDGE(2, 0)->NODE(f)->EDGE(0, 2)->NODE(concat));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto &cfg = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver();
  const auto rounds = cfg.max_fuse_rounds;
  const auto max_fusion_size = cfg.max_fusion_size;
  const auto max_input_nums_after_fuse = cfg.max_input_nums_after_fuse;
  cfg.max_fuse_rounds = 1U;
  cfg.max_fusion_size = 2U;
  cfg.max_input_nums_after_fuse = 2U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);

  // 预期D，E，F，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("F");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  cfg.max_fuse_rounds = rounds;
  cfg.max_fusion_size = max_fusion_size;
  cfg.max_input_nums_after_fuse = max_input_nums_after_fuse;
}

/*        D  E
 *       | \/|
 *       | /\|
 *       F2  G2
 *       |  /
 *       H2
 *       |
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_change_order) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto h = OP_CFG("H").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("H2");
  auto e = OP_CFG("E1").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,1}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E1");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("F2");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G2");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(e));
      CHAIN(NODE(e)->EDGE(0, 0)->NODE(f));
      CHAIN(NODE(e)->EDGE(0, 0)->NODE(g));
      CHAIN(NODE(d)->EDGE(0, 1)->NODE(f)->EDGE(0, 0)->NODE(h));
      CHAIN(NODE(d)->EDGE(1, 1)->NODE(g)->EDGE(0, 1)->NODE(h));
      CHAIN(NODE(h)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
    for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
      const auto node_desc = node->GetOpDesc();
      auto output_tensor_desc = node_desc->MutableOutputDesc(out_anchor->GetIdx());
      gert::SymbolShape symbol_shape1({Symbol(1), Symbol(1), Symbol(3), Symbol(4)});
      gert::SymbolShape symbol_shape2({Symbol(2), Symbol(2), Symbol(3), Symbol(4)});
      gert::SymbolShape symbol_shape;
      if (((node_desc->GetName() == "D") && (out_anchor->GetIdx() == 0))) {
        symbol_shape = symbol_shape2;
      } else if (node_desc->GetName() == "E1") {
        symbol_shape = symbol_shape1;
      }
      output_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
          symbol_shape;
    }
  }
  auto pre_nodes_size = graph->GetAllNodesSize();

  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);

  // 预期D，F2，G2融合在一起
}

/*
 *          data1   data2
 *             \     /
 *                A1
 *                |
 *                B
 *                |
 *            netoutput
 *  节点A1有2个输出anchor，但是只有1个anchor连接了下游节点，另一个anchor没有连接其他节点
 */
TEST_F(UtestFusionStrategySolver, Fuse_OutputAnchorNoPeerIn) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto a1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(2).InNames({"x"})
               .OutNames({"y"}).Build("A1");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT16, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(a1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(a1));
    CHAIN(NODE(a1)->EDGE(1, 0)->NODE(b)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);

  // 预期A1, B融合在一起
}

/*
 *     data
 *      |
 * StridedSlice
 *      |
 * StridedSlice
 *      |
 *      D
 *      |
 *   netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_slice_split) {
  class SliceSplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto stridedslice0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedSlice0");
  auto stridedslice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedSlice1");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(stridedslice0)->EDGE(0, 0)->NODE(stridedslice1));
    CHAIN(NODE(stridedslice1)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SliceSplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Fuse_slice_split_2) {
  class SliceSplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto stridedslice0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,1}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedKSlice0");
  auto stridedslice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,1}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedKSlice1");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,1}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(stridedslice0)->EDGE(0, 0)->NODE(stridedslice1));
    CHAIN(NODE(stridedslice1)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SliceSplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
}


/*
 *     data
 *       |
 *     split
 *     /   \
 *   abs  abs
 *     \   /
 *   netoutput
 */

TEST_F(UtestFusionStrategySolver, Fuse_split) {
  class SplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  dlog_setlevel(0, 0, 1);
  setenv("DUMP_GE_GRAPH", "1", 1);
  setenv("DUMP_GRAPH_LEVEL", "1", 1);
  setenv("DUMP_GRAPH_PATH", "./", 1);
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("Split0");
  auto abs0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("A");
  auto abs1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("R");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(split0)->EDGE(0, 0)->NODE(abs0)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(split0)->EDGE(1, 0)->NODE(abs1)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
}

/*
 *     data
 *       |
 *      abs
 *       |
 *     split
 *     |   |
 *    concat
 *       |
 *   netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_split_nok) {
  class SplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  dlog_setlevel(0, 0, 1);
  setenv("DUMP_GE_GRAPH", "1", 1);
  setenv("DUMP_GRAPH_LEVEL", "1", 1);
  setenv("DUMP_GRAPH_PATH", "./", 1);
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto abs0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("A");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("Split0");
  auto concat0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
    .OutNames({"y"}).Build("Concat2Inputs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(abs0)->EDGE(0, 0)->NODE(split0)->EDGE(0, 0)->NODE(concat0)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(split0)->EDGE(1, 1)->NODE(concat0));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
}

/*
 *     data
 *       |
 *     split
 *     /   \
 *   abs  abs
 *     \   /
 *    concat
 *       |
 *   netoutput
 */

TEST_F(UtestFusionStrategySolver, Fuse_split_concat_nok) {
  class SplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("Split0");
  auto abs0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("A");
  auto abs1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("R");
  auto concat0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
    .OutNames({"y"}).Build("Concat2Inputs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(split0)->EDGE(0, 0)->NODE(abs0)->EDGE(0, 0)->NODE(concat0)->EDGE(0, 0)->NODE(
        "NetOutput", kNetOutputType));
    CHAIN(NODE(split0)->EDGE(1, 0)->NODE(abs1)->EDGE(0, 1)->NODE(concat0));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
}

/*
 *     data
 *       |
 *     split
 *     /   \
 *   abs  abs
 *     \   /
 *    concat
 *       |
 *   netoutput
 */

TEST_F(UtestFusionStrategySolver, Fuse_first_dim_split_concat_ok) {
  class SplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("SplitFirstDim");
  auto abs0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("A");
  auto abs1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("R");
  auto concat0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
    .OutNames({"y"}).Build("Concat2Inputs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(split0)->EDGE(0, 0)->NODE(abs0)->EDGE(0, 0)->NODE(concat0)->EDGE(0, 0)->NODE(
        "NetOutput", kNetOutputType));
    CHAIN(NODE(split0)->EDGE(1, 0)->NODE(abs1)->EDGE(0, 1)->NODE(concat0));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

/*
 *     data
 *       |
 *     split
 *     /   \
 *   abs  abs
 *     \   /
 *    concat
 *       |
 *   netoutput
 */

TEST_F(UtestFusionStrategySolver, Fuse_split_first_dim_concat_ok) {
  class SplitFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto split0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(2).InNames({"x"})
      .OutNames({"y0","y1"}).Build("Split0");
  auto abs0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("A");
  auto abs1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("R");
  auto concat0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(2).OutCnt(1).InNames({"x1", "x2"})
    .OutNames({"y"}).Build("Concat2InputsFirstDim");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(split0)->EDGE(0, 0)->NODE(abs0)->EDGE(0, 0)->NODE(concat0)->EDGE(0, 0)->NODE(
        "NetOutput", kNetOutputType));
    CHAIN(NODE(split0)->EDGE(1, 0)->NODE(abs1)->EDGE(0, 1)->NODE(concat0));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new SplitFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

/*
 *    data1  data2   data3  
 *      \\     |     //
 *       \   gather  /
 *        \    |    /
 *           concat
 *             |
 *          netoutput
 */
TEST_F(UtestFusionStrategySolver, Not_Fuse_gather_and_concat) {
  class GatherFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data1");\
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data3");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Gather");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
      CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
      CHAIN(NODE(data3)->EDGE(0, 2)->NODE(gather));
      CHAIN(NODE(data1)->EDGE(1, 0)->NODE(concat));
      CHAIN(NODE(data3)->EDGE(1, 2)->NODE(concat));
      CHAIN(NODE(gather)->EDGE(0, 1)->NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new GatherFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 0U, post_nodes_size);

  printf("预期融合结果\n");
  // 预期Gather, concat不融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

/*
 *    data1 data2 data3  
 *      \     |     /
 *          gather
 *            |
 *           Abs
 *            |
 *        netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_gather_and_other) {
  class GatherFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data1");\
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data3");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Gather");
  auto abs = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("EletAbs");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
      CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
      CHAIN(NODE(data3)->EDGE(0, 2)->NODE(gather));
      CHAIN(NODE(gather)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new GatherFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);

  printf("预期融合结果\n");
  // 预期Gather, abs融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

// node2 BroadcastWith5Axis(Pointwise with Broadcast) can not fuse
TEST_F(UtestFusionStrategySolver, Fuse_gather_and_Broadcast_Failed0) {
  class GatherFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data1");\
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data3");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("GatherWithInvaildAxis");
  auto abs = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("BroadcastWith5Axis");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
      CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
      CHAIN(NODE(data3)->EDGE(0, 2)->NODE(gather));
      CHAIN(NODE(gather)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new GatherFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);

  printf("预期融合结果\n");
  // 预期Gather, abs融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

// can not fuse cause node1 axis size(5) != node2 axis size(6)
TEST_F(UtestFusionStrategySolver, Fuse_gather_and_Broadcast_Failed1) {
  class GatherFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data1");\
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data3");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("GatherWithInvaildAxis");
  auto abs = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("BroadcastWith6Axis");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
      CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
      CHAIN(NODE(data3)->EDGE(0, 2)->NODE(gather));
      CHAIN(NODE(gather)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new GatherFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);

  printf("预期融合结果\n");
  // 预期Gather, abs融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

/*
 *    data1 data2 data3  
 *      \ \   |   /  /
 *       \  gather  /
 *        \   |    /
 *          gather1
 *            |
 *           abs
 *            |
 *        netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_gather_and_gather) {
  class GatherFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data1");\
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data3");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Gather");
  auto gather1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5,6}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Gather1");
  auto add = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4,5,6}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("EletAbs");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data1)->EDGE(1, 0)->NODE(gather));
      CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
      CHAIN(NODE(data3)->EDGE(0, 2)->NODE(gather));
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather1));
      CHAIN(NODE(data3)->EDGE(1, 2)->NODE(gather1));
      CHAIN(NODE(gather)->EDGE(0, 1)->NODE(gather1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new GatherFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);

  printf("预期融合结果\n");
  // gather1和abs融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

/*
 *    data1 data2 data3  data4
 *      \     |     /     |
 *       \   abs   /      |
 *        \   |   /       |
 *          gather        |
 *          /    \        /
 *       reduce  abs1    / 
 *        \        \    /
 *         \        add   
 *          \       / 
 *          netoutput
 */
TEST_F(UtestFusionStrategySolver, Not_Fuse_gather_and_reduce) {
  class GatherFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data1");\
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data3");
  auto abs = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("EletAbs");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Gather");
  auto abs1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("EletAbs1");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Reduce");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data4");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto log = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Log");
  auto add = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("EletAdd");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
      CHAIN(NODE(data2)->EDGE(0, 0)->NODE(abs)->EDGE(0, 1)->NODE(gather));
      CHAIN(NODE(data3)->EDGE(0, 2)->NODE(gather));
      CHAIN(NODE(gather)->EDGE(0, 0)->NODE(reduce)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(gather)->EDGE(0, 0)->NODE(abs1)->EDGE(0, 0)->NODE(add)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(data4)->EDGE(0, 1)->NODE(add));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto rounds = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new GatherFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);

  printf("预期融合结果\n");
  // gather、abs1和add融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = rounds;
}

TEST_F(UtestFusionStrategySolver, FuseTorch) {
  class AllMockFusionDecider : public MockFusionDecider {
    FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
      return FusionPriority::HIGH;
    }
    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      if ((node1->GetName() == "A") && (node2->GetName() == "E")) {
        return true;
      } else if ((node1->GetName() == "C") && (node2->GetName() == "D")) {
        return true;
      } else if ((node1->GetType() == "FusedAscBackend") && (node2->GetType() == "FusedAscBackend")) {
        return true;
      } else {
        return false;
      }
    }
    bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) {
      return (CanFuseVertical(node1, node2));
    }
  };

  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");

  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
      CHAIN(NODE(a)->NODE(e)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(a)->EDGE(0, 0)->NODE(c)->NODE(d));
      CHAIN(NODE(b)->EDGE(0, 1)->NODE(c));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kTorch);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kTorch;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(graph);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

/*
 *   data
 *    |
 *    D  Data Data
 *    \   |  /
 *      concat
 *       |
 *       G
 *       |
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_concat_and_lifting_above_threshold) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(3).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  std::vector<ge::OpDescPtr> data_ops;
  for (int i = 0; i < 2; ++i) {
    auto data_i = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
        .OutNames({"y"}).Build("Data-" + std::to_string(i));
    data_ops.emplace_back(data_i);
  }
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat3Inputs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data_ops[0])->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(data_ops[1])->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto &cfg = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver();
  const auto rounds = cfg.max_fuse_rounds;
  const auto max_fusion_size = cfg.max_fusion_size;
  const auto max_input_nums_after_fuse = cfg.max_input_nums_after_fuse;
  cfg.max_fuse_rounds = 1U;
  cfg.max_fusion_size = 2U;
  cfg.max_input_nums_after_fuse = 2U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);

  // 预期D，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("Concat3Inputs");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  cfg.max_fuse_rounds = rounds;
  cfg.max_fusion_size = max_fusion_size;
  cfg.max_input_nums_after_fuse = max_input_nums_after_fuse;

  // 非对齐，比例超过阈值
  LowerConcatHelper::LiftingPoorPerfFusedAscBackendOps(graph);
  EXPECT_TRUE(graph->FindFirstNodeMatchType(kFusedAscBackendType) != nullptr);
}

TEST_F(UtestFusionStrategySolver, Fuse_concat_and_lifting_below_threshold) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(3).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  std::vector<ge::OpDescPtr> data_ops;
  for (int i = 0; i < 3; ++i) {
    auto data_i = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
        .OutNames({"y"}).Build("Data-" + std::to_string(i));
    data_ops.emplace_back(data_i);
  }
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat4Inputs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data_ops[0])->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(data_ops[1])->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(data_ops[2])->EDGE(0, 3)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto &cfg = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver();
  const auto rounds = cfg.max_fuse_rounds;
  const auto max_fusion_size = cfg.max_fusion_size;
  const auto max_input_nums_after_fuse = cfg.max_input_nums_after_fuse;
  cfg.max_fuse_rounds = 1U;
  cfg.max_fusion_size = 2U;
  cfg.max_input_nums_after_fuse = 2U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);
  // 预期D，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("Concat4Inputs");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  cfg.max_fuse_rounds = rounds;
  cfg.max_fusion_size = max_fusion_size;
  cfg.max_input_nums_after_fuse = max_input_nums_after_fuse;

  // 非对齐，比例低于阈值
  LowerConcatHelper::LiftingPoorPerfFusedAscBackendOps(graph);
  EXPECT_TRUE(graph->FindFirstNodeMatchType(kFusedAscBackendType) == nullptr);
}

TEST_F(UtestFusionStrategySolver, Fuse_concat_and_no_lifting_first_dim_concat) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(3).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  std::vector<ge::OpDescPtr> data_ops;
  for (int i = 0; i < 3; ++i) {
    auto data_i = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
        .OutNames({"y"}).Build("Data-" + std::to_string(i));
    data_ops.emplace_back(data_i);
  }
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(3).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("ConcatFirstDim");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data_ops[0])->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(data_ops[1])->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(data_ops[2])->EDGE(0, 3)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  auto &cfg = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver();
  const auto rounds = cfg.max_fuse_rounds;
  const auto max_fusion_size = cfg.max_fusion_size;
  const auto max_input_nums_after_fuse = cfg.max_input_nums_after_fuse;
  cfg.max_fuse_rounds = 1U;
  cfg.max_fusion_size = 2U;
  cfg.max_input_nums_after_fuse = 2U;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1U, post_nodes_size);

  // 预期D，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("D");
  expect_nodes[0].insert("ConcatFirstDim");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  cfg.max_fuse_rounds = rounds;
  cfg.max_fusion_size = max_fusion_size;
  cfg.max_input_nums_after_fuse = max_input_nums_after_fuse;

  // 非对齐，比例超过阈值
  LowerConcatHelper::LiftingPoorPerfFusedAscBackendOps(graph);
  EXPECT_TRUE(graph->FindFirstNodeMatchType(kFusedAscBackendType) != nullptr);
}

/*
 *      data
 *       |
 *       D
 *      / \
 *    E     G
 *    |     |  \
 *    |     |   E1
 *    |     |    |
 * reduce--->concat
 *     \   /
 *    netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_reduce_after_concat_not_fuse) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("E");
  auto e1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedSlice0");
  auto g = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("G");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Concat");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Reduce");
  DEF_GRAPH(g1) {
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(e)->EDGE(0, 0)->NODE(f));
      CHAIN(NODE(d)->EDGE(0, 0)->NODE(g)->EDGE(0, 0)->NODE(concat));
      CHAIN(NODE(g)->EDGE(0, 0)->NODE(e1)->EDGE(0, 1)->NODE(concat));
      CHAIN(NODE(f)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
      CHAIN(NODE(f)->CTRL_EDGE()->NODE(concat));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver  fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();

  // 预期D,E，Reduce融合在一起, G，Concat成环不融合在一起
  EXPECT_EQ(pre_nodes_size - 4U, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Slice_Horizontal_Has_Multi_Same_Input_Fuse_ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto slice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNode1");
  auto slice2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNode2");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(slice1));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(slice1));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice2));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(slice2));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(slice2));
    CHAIN(NODE(slice1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(slice2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver solver;
  EXPECT_EQ(solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Slice_Horizontal_Has_Same_Load_Fuse_ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  // auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
  //                  .OutNames({"y"}).Build("data2");
  // auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
  //                  .OutNames({"y"}).Build("data3");
  auto slice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNodeSame1");
  auto slice2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNodeSame2");
  auto slice3 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNodeSame3");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("ConcatNode");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice1));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice2));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice3));
    CHAIN(NODE(slice1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(slice2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(slice3)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver solver;
  EXPECT_EQ(solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Fuse_concat_has_vertical_and_horizontal_fuse) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto f2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("F2");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(4).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat4Inputs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(f2));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(f2));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(a)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(b)->EDGE(0, 3)->NODE(concat));
    CHAIN(NODE(f2)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);

  // 预期A, B, F2, Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("A");
  expect_nodes[0].insert("B");
  expect_nodes[0].insert("F2");
  expect_nodes[0].insert("Concat4Inputs");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

TEST_F(UtestFusionStrategySolver, Concat_Can_Not_Fuse_By_Only_Horizontal) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("C");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("F");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(f));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(e));
    CHAIN(NODE(c)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(f)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(e)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(b));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(c));
    CHAIN(NODE(b)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(concat)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // C，F，E，Concat融合在一起；Reduce和b融合在一起
  EXPECT_EQ(pre_nodes_size - 4U, post_nodes_size);

  // 预期C，F，E，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("C");
  expect_nodes[0].insert("F");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

// FusedAscBackend融合过程测试debug日志级别的dump缓存能力
TEST_F(UtestFusionStrategySolver, Concat_Fuse_For_CacheGraph) {
  setenv("DUMP_GE_GRAPH", "1", 1);
  setenv("DUMP_GRAPH_LEVEL", "1", 1);
  setenv("DUMP_GRAPH_PATH", "./", 1);
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto r = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("R");
  auto r1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("R1");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  auto b1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B1");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("C");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("F");
  auto e1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E1");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(r1));
    CHAIN(NODE(r1)->EDGE(0, 0)->NODE(r));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(f));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(e1));
    CHAIN(NODE(e1)->EDGE(0, 0)->NODE(e));
    CHAIN(NODE(c)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(f)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(e)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(r)->EDGE(0, 0)->NODE(b));
    CHAIN(NODE(r)->EDGE(0, 0)->NODE(c));
    CHAIN(NODE(b)->EDGE(0, 0)->NODE(b1));
    CHAIN(NODE(b1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(concat)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Abs");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Rsqrt");
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.insert("Sum");
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  ASSERT_EQ(post_processor.Do(graph), SUCCESS);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().improve_precision_blacklist.clear();
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 7U, post_nodes_size);

  // 预期 "C", "Concat", "E", "E1", "F", "R", "R1" 融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("C");
  expect_nodes[0].insert("F");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("E1");
  expect_nodes[0].insert("R");
  expect_nodes[0].insert("R1");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  unsetenv("DUMP_GE_GRAPH");
  unsetenv("DUMP_GRAPH_LEVEL");
  unsetenv("DUMP_GRAPH_PATH");
}

TEST_F(UtestFusionStrategySolver, Concat_Can_Fuse_By_Same_Sched_Info_Both_Horizontal_And_Vertical) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  auto f2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("F2");
  auto f = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("F");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(f));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(e));
    CHAIN(NODE(f2)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(f)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(e)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(b)->EDGE(0, 1)->NODE(f2));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(b));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(f2));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // F2，F，E，B，Concat融合在一起
  EXPECT_EQ(pre_nodes_size - 4U, post_nodes_size);

  // 预期B，F2，F，E，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("B");
  expect_nodes[0].insert("F2");
  expect_nodes[0].insert("F");
  expect_nodes[0].insert("E");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

TEST_F(UtestFusionStrategySolver, Concat_Can_Fuse_By_Different_Sched_Info_Both_Horizontal_And_Vertical) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("BroadcastWith6Axis1");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(b));
    CHAIN(NODE(reduce)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(b)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto node1 = graph->FindNode("BroadcastWith6Axis1");
  EXPECT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("Concat");
  EXPECT_NE(node2, nullptr);
  AscGraphAxisMapping graph_axis_map;
  NodeFuseInfo node_fuse_info;
  EXPECT_EQ(node_fuse_info.UpdateNodeFuseInfo(node1, node2), SUCCESS);
  EXPECT_EQ(BackendUtils::CheckSameSchedAxis(node1, node2, graph_axis_map.GetNode1AxisMap(),
                                             graph_axis_map.GetNode2AxisMap(), node_fuse_info), true);
}

TEST_F(UtestFusionStrategySolver, Node_PeerIn_Anchor_Has_No_PeerOut_Anchor_Link) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto f2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("F2");
  auto g2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("G2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 1)->NODE(f2));
    CHAIN(NODE(reduce)->EDGE(0, 1)->NODE(g2));
    CHAIN(NODE(f2)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(g2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto node1 = graph->FindNode("F2");
  EXPECT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("G2");
  EXPECT_NE(node2, nullptr);
  EXPECT_EQ(BackendUtils::IsHorizontal(node1, node2), true);
}

// 两个节点没有共用的内存不能融合
TEST_F(UtestFusionStrategySolver, Can_Not_Fuse_By_No_Shared_Data) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce)->NODE(a));
    CHAIN(NODE(a)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto attr = GetOrCreateAutoFuseAttrs(graph);
  GetInterAttrs(attr).possible_fusion_nodes.insert(std::make_pair(graph->FindNode("Reduce"), graph->FindNode("B")));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
}

// Concat融合场景--调度轴个数不同不能融合Case
TEST_F(UtestFusionStrategySolver, Can_Not_Fuse_By_Sched_Axis_Size_Not_Equal) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("BroadcastWith6Axis1");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("C");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("BroadcastWith6Axis2");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(a));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(c));
    CHAIN(NODE(a)->EDGE(0, 0)->NODE(b));
    CHAIN(NODE(a)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(c)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(c)->EDGE(0, 0)->NODE(e));
    CHAIN(NODE(b)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(concat)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(e)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // A，C，Concat融合在一起
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);

  // 预期A，C，Concat融合在一起
  std::map<size_t, std::set<std::string>> expect_nodes;
  expect_nodes[0].insert("A");
  expect_nodes[0].insert("C");
  expect_nodes[0].insert("Concat");
  EXPECT_TRUE(CheckFuseNodes(graph, expect_nodes));
}

/*
 *      data
 *       |
 *       A
 *     /   \
 *    B     E
 *     \   /
 *   netoutput
 */
TEST_F(UtestFusionStrategySolver, Fuse_output_memory_size_exceed_threshold) {
  class MockFusionDecider : public AscBackendFusionDecider {
    bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) {
      return false;
    }
  };
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                  .OutNames({"y"}).Build("data");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("B");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(a));
    CHAIN(NODE(a)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(a)->EDGE(0, 0)->NODE(e)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  int64_t old_max_output_memory_size_after_fusion =
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_output_memory_size_after_fusion;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_output_memory_size_after_fusion = 100;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_output_memory_size_after_fusion =
      old_max_output_memory_size_after_fusion;
}

TEST_F(UtestFusionStrategySolver, Fuse_concat_and_mulreference_node_1) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("C");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  auto f2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("Broadcast");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Gather");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(f2));
    CHAIN(NODE(d)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(gather)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(f2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(c)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data4)->EDGE(0, 0)->NODE(e)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // E，Broadcast，Concat融合在一起
  EXPECT_EQ(pre_nodes_size - 2U, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Fuse_concat_and_mulreference_node_2) {
  class ConcatFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("D");
  auto e = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("E");
  auto f2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("Broadcast");
  auto g2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("G2");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(f2));
    CHAIN(NODE(d)->EDGE(0, 1)->NODE(g2));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(g2));
    CHAIN(NODE(data4)->EDGE(0, 0)->NODE(e));
    CHAIN(NODE(f2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(g2)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(e)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new ConcatFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  auto post_nodes_size = graph->GetAllNodesSize();
  // E，Broadcast，G2，Concat融合在一起
  EXPECT_EQ(pre_nodes_size - 3U, post_nodes_size);
}

std::string ReadableComputeGraph(const ComputeGraphPtr &graph) {
  std::stringstream ss;
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  ss << "ComputeGraph(" << graph->GetName() << ")" << std::endl;
  std::set<NodePtr> can_reached;
  std::stack<NodePtr> stack;
  auto sink = graph->FindFirstNodeMatchType("NetOutput");
  can_reached.insert(sink);
  if (sink != nullptr) {
    stack.push(sink);
    while (!stack.empty()) {
      auto current = stack.top();
      stack.pop();
      for (auto &in_node : current->GetInAllNodes()) {
        if (can_reached.insert(in_node).second) {
          stack.push(in_node);
        }
      }
    }
  }
  for (const auto &node : graph->GetAllNodes()) {
    if (can_reached.find(node) == can_reached.end()) {
      continue;
    }
    std::vector<std::string> input_names;
    std::vector<std::string> control_names;
    for (auto &anchor : node->GetAllInDataAnchors()) {
      auto peer = anchor->GetPeerOutAnchor();
      if (peer == nullptr) {
        continue;
      }
      input_names.emplace_back(anchor_name[peer]);
    }
    for (auto &in_control : node->GetInControlNodes()) {
      control_names.emplace_back(in_control->GetName());
    }
    std::vector<std::string> output_names;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
    }
    if (output_names.size() > 1U) {
      ss << loop::StrJoin(output_names) << " = ";
    } else if (!output_names.empty()) {
      ss << output_names[0] << " = ";
    }
    if (control_names.empty()) {
      ss << "ge." << node->GetType() << "(" << node->GetName() << ", " << loop::StrJoin(input_names) << ")"
         << std::endl;
    } else {
      ss << "ge." << node->GetType() << "(" << node->GetName() << ", " << loop::StrJoin(input_names) << ", "
         << loop::StrJoin(control_names) << ")" << std::endl;
    }
  }
  return ss.str();
}

TEST_F(UtestFusionStrategySolver, FuseMatmul) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::MatMulV3(abs, data1, nullptr, nullptr);
  mm.SetSymbolShape({"s1", "s2"});
  auto abs1 = es::Abs(mm);
  abs1.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(abs1, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("MatMulV3_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc4 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_FLOAT16);
  tmp_desc4->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseMatmulOffsetRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto offset_w = es_graph->CreateInput(2, "offset_w", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  offset_w.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::MatMulV2(abs, data1, nullptr, offset_w);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("MatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("offset_w");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc4 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_FLOAT16);
  tmp_desc4->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseMatmulBaisOffsetRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph1"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto bais = es_graph->CreateInput(2, "bais", nullptr);
  auto offset_w = es_graph->CreateInput(3, "offset_w", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s2", "s3"});
  offset_w.SetSymbolShape({"s3", "s4"});
  bais.SetSymbolShape({"s4", "s5"});
  auto mm = es::MatMulV3(data0, data1, bais, offset_w);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);
  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("MatMulV3_0");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("bais");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_FLOAT);
  tmp_desc3->SetOriginDataType(DT_FLOAT);
  auto data4_node = cg->FindNode("offset_w");
  auto tmp_desc4 = data4_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_INT8);
  tmp_desc4->SetOriginDataType(DT_INT8);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseMatmulBaisRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto bais = es_graph->CreateInput(2, "bais", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  bais.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::MatMulV2(abs, data1, bais, nullptr);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("MatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("bais");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_FLOAT);
  tmp_desc3->SetOriginDataType(DT_FLOAT);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc5 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc5->SetDataType(DT_FLOAT16);
  tmp_desc5->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseMatmulMulReferenceWithBaisRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto bais = es_graph->CreateInput(2, "bais", nullptr);
  data0.SetSymbolShape({"s1", "s1"});
  data1.SetSymbolShape({"s1", "s1"});
  bais.SetSymbolShape({"s1", "s1"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s1"});
  auto mm = es::MatMulV2(abs, data1, bais, nullptr);
  mm.SetSymbolShape({"s1", "s1"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s1"});
  auto mm2 = es::MatMulV2(relu, data1, bais, nullptr);
  mm2.SetSymbolShape({"s1", "s1"});
  es_graph->SetOutput(mm2, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("MatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("bais");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_FLOAT);
  tmp_desc3->SetOriginDataType(DT_FLOAT);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc5 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc5->SetDataType(DT_FLOAT16);
  tmp_desc5->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseBatchMatmul) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto offset_w = es_graph->CreateInput(2, "offset_w", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  offset_w.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::BatchMatMulV3(abs, data1, nullptr, offset_w);
  mm.SetSymbolShape({"s1", "s2"});
  auto abs1 = es::Abs(mm);
  abs1.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(abs1, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("BatchMatMulV3_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("offset_w");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc4 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_FLOAT16);
  tmp_desc4->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseBatchMatmulOffsetRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto offset_w = es_graph->CreateInput(2, "offset_w", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  offset_w.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::BatchMatMulV2(abs, data1, nullptr, offset_w);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("BatchMatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("offset_w");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc4 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_FLOAT16);
  tmp_desc4->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseBatchMatmulBaisOffsetRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto bais = es_graph->CreateInput(2, "bais", nullptr);
  auto offset_w = es_graph->CreateInput(3, "offset_w", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  offset_w.SetSymbolShape({"s1", "s2"});
  bais.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::BatchMatMulV2(abs, data1, bais, offset_w);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("BatchMatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("bais");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_FLOAT);
  tmp_desc3->SetOriginDataType(DT_FLOAT);
  auto data4_node = cg->FindNode("offset_w");
  auto tmp_desc4 = data4_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(DT_INT8);
  tmp_desc4->SetOriginDataType(DT_INT8);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc5 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc5->SetDataType(DT_FLOAT16);
  tmp_desc5->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseBatchMatmulBaisRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto bais = es_graph->CreateInput(2, "bais", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  bais.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::BatchMatMulV2(abs, data1, bais, nullptr);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("BatchMatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("bais");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_FLOAT);
  tmp_desc3->SetOriginDataType(DT_FLOAT);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc5 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc5->SetDataType(DT_FLOAT16);
  tmp_desc5->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestFusionStrategySolver, FuseBatchMatmulBaisMulRelu) {
  std::unique_ptr<es::Graph> es_graph = std::unique_ptr<es::Graph>(new es::Graph("graph"));;
  auto data0 = es_graph->CreateInput(0, "data0", nullptr);
  auto data1 = es_graph->CreateInput(1, "data1", nullptr);
  auto bais = es_graph->CreateInput(2, "bais", nullptr);
  data0.SetSymbolShape({"s1", "s2"});
  data1.SetSymbolShape({"s1", "s2"});
  bais.SetSymbolShape({"s1", "s2"});
  auto abs = es::Abs(data0);
  abs.SetSymbolShape({"s1", "s2"});
  auto mm = es::BatchMatMulV2(abs, data1, bais, nullptr);
  mm.SetSymbolShape({"s1", "s2"});
  auto relu = es::Relu(mm);
  relu.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu, 0);
  auto relu1 = es::Relu(mm);
  relu1.SetSymbolShape({"s1", "s2"});
  es_graph->SetOutput(relu1, 1);

  auto graph = es_graph->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto matmul = cg->FindNode("BatchMatMulV2_1");
  ASSERT_NE(matmul, nullptr);
  auto tmp_desc = matmul->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);
  auto data0_node = cg->FindNode("data0");
  auto tmp_desc0 = data0_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1_node = cg->FindNode("data1");
  auto tmp_desc1 = data1_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);
  auto data3_node = cg->FindNode("bais");
  auto tmp_desc3 = data3_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_FLOAT);
  tmp_desc3->SetOriginDataType(DT_FLOAT);
  auto abs_node = cg->FindNode("Abs_0");
  auto tmp_desc5 = abs_node->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc5->SetDataType(DT_FLOAT16);
  tmp_desc5->SetOriginDataType(DT_FLOAT16);

  auto pre_nodes_size = cg->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()),
      AutoFuseFwkType::kGe);

  AutofuserOptions options;
  options.fwk_type = AutoFuseFwkType::kGe;
  Autofuser autofuser(options);
  auto ret = autofuser.Fuse(cg);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().fwk_type = AutoFuseFwkType::kGe;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fuse_rounds = 10U;
  ASSERT_EQ(ret, SUCCESS);
}
TEST_F(UtestFusionStrategySolver, Reduce_Can_Only_Fuse_At_Most_3_Elementwise) {
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
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("D");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("A");
  auto r = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("R");
  auto r1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("R1");
  auto e1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                .OutNames({"y"}).Build("E1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(gather)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(r));
    CHAIN(NODE(r)->EDGE(0, 0)->NODE(r1)->EDGE(0, 0)->NODE(e1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  // Reduce, D, A, R, R1融合在一起
  EXPECT_EQ(pre_nodes_size - 4U, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Reduce_Can_Not_Fuse_With_Elementwise_Which_Has_More_Than_3_Compute_Nodes) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Gather");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(gather)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto pre_nodes_size = graph->GetAllNodesSize();
  size_t old_max_reduce_can_fuse_elementwise_nums =
      AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_reduce_can_fuse_elementwise_nums;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_reduce_can_fuse_elementwise_nums = 0U;
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size, post_nodes_size);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_reduce_can_fuse_elementwise_nums =
      old_max_reduce_can_fuse_elementwise_nums;
}

TEST_F(UtestFusionStrategySolver, Reduce_Can_Not_Fuse_With_Elementwise_Has_More_Than_One_Input) {
  class MockFusionDecider : public AscBackendFusionDecider {
    NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
      return AscBackendFusionDecider::Fuse(node1, node2, counter);
    }
  };
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto gather = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Gather");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("C");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(gather)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(c));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(c)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1, post_nodes_size);
}

TEST_F(UtestFusionStrategySolver, Reduce_Can_Not_Fuse_With_Elementwise_Has_Scalar) {
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
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
               .OutNames({"y"}).Build("Mul");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(gather));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(gather));
    CHAIN(NODE(gather)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(c)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  const auto pre_nodes_size = graph->GetAllNodesSize();
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new MockFusionDecider()));
  FusionStrategySolver fusion_strategy_solver;
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  const auto post_nodes_size = graph->GetAllNodesSize();
  EXPECT_EQ(pre_nodes_size - 1, post_nodes_size);
}

} // namespace ge
