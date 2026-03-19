/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "share_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

using namespace ge;

namespace {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ComputeGraph>(name);
    node_type_ = node_type;
  }

  NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                  const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    tensor_desc->SetShape(GeShape(std::move(shape)));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const NodePtr &src_node, const std::int32_t src_idx, const NodePtr &dst_node,
                   const std::int32_t dst_idx) {
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
  std::string node_type_;
};

static void ConstructVVAscGraphAxisInfo(ge::AscGraph &graph, size_t dims_size) {
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(sym_str.c_str());
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVConstAscGraphAxisInfo(ge::AscGraph &graph, size_t dims_size, vector<int> dims) {
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(dims[i], sym_str.c_str());
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVAscGraphAxisInfo(ge::AscGraph &graph, size_t dims_size, size_t last_axis_strides) {
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  auto ONE = Symbol(last_axis_strides);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(sym_str.c_str());
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVAscGraphAxisInfo(ge::AscGraph &graph, size_t dims_size, vector<size_t> perms) {
  // 原始轴信息
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  // transpose后的轴信息
  std::vector<int64_t> t_axis;
  std::vector<ge::Expression> t_repeats;
  std::vector<ge::Expression> t_strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(sym_str.c_str());
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  for (size_t i = 0; i < dims_size; i++) {
    t_axis.push_back(axis[perms[i]]);
    t_repeats.push_back(repeats[perms[i]]);
    t_strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    t_strides[i] = ge::sym::Mul(t_repeats[i + 1], t_strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Scalar>(node)) {
      continue;
    }
    // transpose及之后的节点需要设置成转置属性
    if (ge::ops::IsOps<ge::ascir_op::Transpose>(node)) {
      axis = t_axis;
      repeats = t_repeats;
      strides = t_strides;
    }
    node->attr.sched.axis = t_axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVAscGraphAxisInfo(ge::AscGraph &graph,
                                        const std::vector<std::string> &dim_sizes,
                                        bool align) {
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  auto ONE = ge::Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dim_sizes.size(); i++) {
    ge::Symbol sym_s;
    if (dim_sizes[i][0] == 's') {
      sym_s = ge::Symbol(dim_sizes[i].c_str());
      graph.CreateSizeVar(dim_sizes[i]);
    } else {
      sym_s = ge::Symbol(std::atoi(dim_sizes[i].c_str()));
    }
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  // s0, s1
  if (align) {
    strides[dim_sizes.size() - 2] = ge::sym::Align(repeats[dim_sizes.size() - 1], 32 / sizeof(int32_t));
  } else {
    strides[dim_sizes.size() - 2] = repeats[dim_sizes.size() - 1];
  }
  for (int32_t i = static_cast<int32_t>(dim_sizes.size()) - 3; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVConstAscGraphAxisInfo(ge::AscGraph &graph, size_t dims_size, vector<size_t> perms, vector<int> dims) {
  // 原始轴信息
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  // transpose后的轴信息
  std::vector<int64_t> t_axis;
  std::vector<ge::Expression> t_repeats;
  std::vector<ge::Expression> t_strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(dims[i], sym_str.c_str());
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  for (size_t i = 0; i < dims_size; i++) {
    t_axis.push_back(axis[perms[i]]);
    t_repeats.push_back(repeats[perms[i]]);
    t_strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    t_strides[i] = ge::sym::Mul(t_repeats[i + 1], t_strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Scalar>(node)) {
      continue;
    }
    // transpose及之后的节点需要设置成转置属性
    if (ge::ops::IsOps<ge::ascir_op::Transpose>(node)) {
      axis = t_axis;
      repeats = t_repeats;
      strides = t_strides;
    }
    node->attr.sched.axis = t_axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}
}

namespace ascir {
/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddAbsAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add(bf16)
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateBF16AddAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = ge::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = add.y;
  x_out.y.dtype = ge::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add(bf16)
 *       /   \
 *   nddma0  nddma1
 *     |       |
 *   data0   data1
 */
static void CreateBF16NddmaAddAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_BF16;

  ge::ascir_op::Nddma x1Local("nddma0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Nddma x2Local("nddma1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = ge::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = add.y;
  x_out.y.dtype = ge::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAddAbsConstAscGraph(ge::AscGraph &graph, size_t dims_size, std::vector<int> dims) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVConstAscGraphAxisInfo(graph, dims_size, dims);
}

/**
 *      output
 *         |
 *       store
 *         |
 *     logical_or
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadLogicalOrStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  x1.y.dtype = ge::DT_UINT8;
  x2.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  x1Local.y.dtype = ge::DT_UINT8;
  x2Local.y.dtype = ge::DT_UINT8;
  ge::ascir_op::LogicalOr logical_or("logical_or");
  logical_or.x1 = x1Local.y;
  logical_or.x2 = x2Local.y;

  logical_or.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Store x_out("store");
  x_out.x = logical_or.y;

  x_out.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *    logical_and
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadLogicalAndStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  x1.y.dtype = ge::DT_UINT8;
  x2.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  x1Local.y.dtype = ge::DT_UINT8;
  x2Local.y.dtype = ge::DT_UINT8;
  ge::ascir_op::LogicalAnd logical_and("logical_and");
  logical_and.x1 = x1Local.y;
  logical_and.x2 = x2Local.y;

  logical_and.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Store x_out("store");
  x_out.x = logical_and.y;

  x_out.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *      floordiv
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateFloordivAbsAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = DT_INT8;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = DT_INT8;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = DT_INT8;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = DT_INT8;

  ge::ascir_op::FloorDiv floordiv("floordiv");
  floordiv.x1 = x1Local.y;
  floordiv.x2 = x2Local.y;
  floordiv.y.dtype = DT_INT8;

  ge::ascir_op::Abs abs("abs");
  abs.x = floordiv.y;
  abs.y.dtype = DT_INT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = DT_INT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        exp2
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddExp2AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Exp2 exp2("exp2");
  exp2.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = exp2.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *       floor
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddFloorAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Floor floor("floor");
  floor.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = floor.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**     (BF16)
 *      output
 *         |
 *       store
 *         |
 *       floor
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddFloorBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = ge::DT_BF16;

  ge::ascir_op::Floor floor("floor");
  floor.x = add.y;
  floor.y.dtype = ge::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = floor.y;
  x_out.y.dtype = ge::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**     (BF16)
 *      output
 *         |
 *       store
 *         |
 *        exp
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddExpBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = ge::DT_BF16;

  ge::ascir_op::Exp exp("exp");
  exp.x = add.y;
  exp.y.dtype = ge::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = exp.y;
  x_out.y.dtype = ge::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddExp2FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_exp2_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_exp2");
  CreateAddExp2AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddFloorFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_floor_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_floor");
  CreateAddFloorAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddFloorBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_floor_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_floor_bf16");
  CreateAddFloorBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddExpBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_exp_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_exp_bf16");
  CreateAddExpBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        axpy
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAxpyAbsAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Axpy axpy("axpy");
  axpy.x1 = x1Local.y;
  axpy.x2 = x2Local.y;
  axpy.ir_attr.SetAlpha(0.8);

  ge::ascir_op::Abs abs("abs");
  abs.x = axpy.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        axpy
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAxpyAbsHalfAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x1.y.dtype = ge::DT_FLOAT16;
  x2.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  x1Local.y.dtype = ge::DT_FLOAT16;
  x2Local.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Axpy axpy("axpy");
  axpy.x1 = x1Local.y;
  axpy.x2 = x2Local.y;
  axpy.ir_attr.SetAlpha(0.8);
  axpy.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Abs abs("abs");
  abs.x = axpy.y;

  abs.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  x_out.y.dtype = ge::DT_FLOAT16;
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *          output
 *             |
 *           store
 *             |
 *            add
 *           / |
 *        axpy |
 *       /   \ |
 *   load0  load1
 *     |      |
 *   data0  data1
 */
static void CreateAxpyAddAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Axpy axpy("axpy");
  axpy.x1 = x1Local.y;
  axpy.x2 = x2Local.y;
  axpy.ir_attr.SetAlpha(0.8);

  ge::ascir_op::Add add("add");
  add.x1 = axpy.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = add.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAddAbsScalarAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType dtype) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = dtype;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = dtype;

  ge::ascir_op::Abs abs("abs");
  abs.x = add.y;
  abs.y.dtype = dtype;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = dtype;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size, 0U);
}

/**
 *         data0
 *           |
 *         load0
 *         |    \
 *       store  abs
 *         |     |
 *        ouput0 output1
 */
static void CreateLoadToStoreAndAbsAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  ge::ascir_op::Store abs_store("store");
  abs_store.x = abs.y;

  ge::ascir_op::Store load_2_store("store2");
  load_2_store.x = x1Local.y;

  ge::ascir_op::Output y1("output");
  y1.x = abs_store.y;
  y1.ir_attr.SetIndex(0);

  ge::ascir_op::Output y2("output2");
  y2.x = load_2_store.y;
  y2.ir_attr.SetIndex(1);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *         data0
 *           |
 *         load0
 *         |    \
 *       store  abs
 *         |     |
 *        ouput0 output1
 */
static void CreateLoadUnalignPadAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  ge::ascir_op::Store abs_store("store");
  abs_store.x = abs.y;

  ge::ascir_op::Store load_2_store("store2");
  load_2_store.x = x1Local.y;

  ge::ascir_op::Output y1("output");
  y1.x = abs_store.y;
  y1.ir_attr.SetIndex(0);

  ge::ascir_op::Output y2("output2");
  y2.x = load_2_store.y;
  y2.ir_attr.SetIndex(1);

  ConstructVVAscGraphAxisInfo(graph, dims_size, 2UL);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *        | \
 *       |  abs
 *      |    \
 *   load0   brc
 *     |      \
 *  data0   scalar0
 */
static void CreateAbsBrcAddAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x0("data", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Scalar x1("scalar", graph);
  x1.ir_attr.SetIndex(1);
  x1.ir_attr.SetValue("1.0");

  ge::ascir_op::Broadcast brc0("brc");
  brc0.x = x1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs1("abs");
  abs1.x = brc0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeLoad;
  abs1.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs1.attr.sched.axis = {z0.id, z1.id};
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  abs1.y.dtype = ge::DataType::DT_FLOAT;
  abs1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Add add("add");
  add.x1 = load0.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = ComputeType::kComputeLoad;
  add.attr.api.type = ge::ApiType::kAPITypeCompute;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  add.y.dtype = ge::DataType::DT_FLOAT;
  add.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *        | \
 *       |  abs
 *      |    \
 *   load0   brc
 *     |      \
 *  data0   scalar0
 */
static void CreateUbScalerBrcAbsAddAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x0("data", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("scalar", graph);
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc0("brc");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Abs abs1("abs");
  abs1.x = brc0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeLoad;
  abs1.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs1.attr.sched.axis = {z0.id, z1.id};
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  abs1.y.dtype = ge::DataType::DT_FLOAT;
  abs1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Add add("add");
  add.x1 = load0.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = ComputeType::kComputeLoad;
  add.attr.api.type = ge::ApiType::kAPITypeCompute;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  add.y.dtype = ge::DataType::DT_FLOAT;
  add.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_abs");
  CreateAddAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::FloordivAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("floordiv_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("floordiv_abs");
  CreateFloordivAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::BF16AddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("bf16_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("bf16_add");
  CreateBF16AddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::BF16NddmaAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("bf16_nddma_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("bf16_nddma_add");
  CreateBF16NddmaAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAbsUint8AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_UINT8;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Abs abs("abs");
  abs.x = xLocal.y;
  abs.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AbsUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_uint8_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("abs_uint8_test");
  CreateAbsUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAbsBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Abs abs("abs");
  abs.x = xLocal.y;
  abs.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AbsBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("abs_bf16_test");
  CreateAbsBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCeilBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Ceil ceil("ceil");
  ceil.x = xLocal.y;
  ceil.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = ceil.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::CeilBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("ceil_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("ceil_bf16_test");
  CreateCeilBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCosBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Cos cos("cos");
  cos.x = xLocal.y;
  cos.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = cos.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::CosBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("cos_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("cos_bf16_test");
  CreateCosBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateErfBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Erf erf("erf");
  erf.x = xLocal.y;
  erf.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = erf.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::ErfBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("erf_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("erf_bf16_test");
  CreateErfBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAbsClipAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  ge::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  ge::ascir_op::ClipByValue clipbyvalue("clipbyvalue");
  clipbyvalue.x1 = abs.y;
  clipbyvalue.x2 = x2Local.y;
  clipbyvalue.x3 = x3Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = clipbyvalue.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAbsFmaBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);
  x3.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;
  x3Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;
  abs.y.dtype = ge::DT_BF16;

  ge::ascir_op::Fma fma("fma");
  fma.x1 = abs.y;
  fma.x2 = x2Local.y;
  fma.x3 = x3Local.y;
  fma.y.dtype = ge::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = fma.y;
  x_out.y.dtype = ge::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAbsFmaAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  ge::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  ge::ascir_op::Fma fma("fma");
  fma.x1 = abs.y;
  fma.x2 = x2Local.y;
  fma.x3 = x3Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = fma.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AbsClipFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_clip_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data1, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("abs_clip");
  CreateAbsClipAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

ge::ComputeGraphPtr ShareGraph::AbsFmaFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_fma_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data1, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("abs_fma");
  CreateAbsFmaAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

ge::ComputeGraphPtr ShareGraph::AbsFmaBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_fma_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data1, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("abs_fma_bf16");
  CreateAbsFmaBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadLogicalOrStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_logicalor_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_logicalor_store");
  CreateLoadLogicalOrStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadLogicalAndStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_logicaland_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_logicaland_store");
  CreateLoadLogicalAndStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AxpyAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("axpy_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("axpy_abs");
  CreateAxpyAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AxpyAbsHalfFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("axpy_abs_half_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("axpy_abs_half");
  CreateAxpyAbsHalfAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AxpyAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("axpy_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("axpy_add");
  CreateAxpyAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddAbsScalarFusedGraph(size_t dims_size, ge::DataType dtype) {
  auto builder = GraphBuilder("add_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_abs");
  CreateAddAbsScalarAscGraph(sub_graph, dims_size, dtype);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        cast0
 *         |
 *        abs
 *         |
 *        cast1
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastAbsAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype,
                                  ge::DataType out_dtype) {
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  ge::ascir_op::Cast cast0("cast0");
  cast0.x = load.y;
  cast0.y.dtype = out_dtype;

  ge::ascir_op::Abs abs("abs");
  abs.x = cast0.y;
  abs.y.dtype = out_dtype;

  ge::ascir_op::Cast cast1("cast1");
  cast1.x = abs.y;
  cast1.y.dtype = in_dtype;

  ge::ascir_op::Store store("store");
  store.x = cast1.y;
  store.y.dtype = in_dtype;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = in_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
ge::ComputeGraphPtr ShareGraph::CastCastFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("cast_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("cast_abs");
  CreateCastAbsAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        abs
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastReciprocalAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype,
                                  ge::DataType out_dtype) {
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  ge::ascir_op::Reciprocal reciprocal("reciprocal");
  reciprocal.x = load.y;
  reciprocal.y.dtype = out_dtype;

  ge::ascir_op::Store store("store");
  store.x = reciprocal.y;
  store.y.dtype = out_dtype;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
ge::ComputeGraphPtr ShareGraph::CastCastReciprocalFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("cast_reciprocal_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("cast_reciprocal_test");
  CreateCastReciprocalAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        IsNan
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastIsNanAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype,
                                  ge::DataType out_dtype) {
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  ge::ascir_op::Isnan isnan("IsNan");
  isnan.x = load.y;
  isnan.y.dtype = out_dtype;

  ge::ascir_op::Store store("store");
  store.x = isnan.y;
  store.y.dtype = out_dtype;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
ge::ComputeGraphPtr ShareGraph::CastCastNanFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("cast_nan_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("cast_nan");
  CreateCastIsNanAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        IsFinite
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastIsFiniteAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype,
                                  ge::DataType out_dtype) {
  ge::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  ge::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  ge::ascir_op::IsFinite Isfinite("IsFinite");
  Isfinite.x = load.y;
  Isfinite.y.dtype = out_dtype;

  ge::ascir_op::Store store("store");
  store.x = Isfinite.y;
  store.y.dtype = out_dtype;

  ge::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
ge::ComputeGraphPtr ShareGraph::CastCastIsFiniteFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("cast_isfinite_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("cast_isfinite_test");
  CreateCastIsFiniteAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      data
 *         |
 *      AscBc
 *         /
 *      netoutput
 */
ge::ComputeGraphPtr ShareGraph::LoadToStoreAndAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_to_store_and_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 2);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_abs");
  CreateLoadToStoreAndAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *              output
 *                |
 *              store
 *                |
 *              add0
 *              /   \
 *            load1  load0
 *             /      \
 *           data1   data0
 */
static void CreateLoadLoopModeFusedGraph(ge::AscGraph& graph, size_t dims_size) { // 4, 8, 16, 64, 32
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(16);

  auto Three = Symbol(3);
  auto Two = Symbol(2);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2 * Three, s2 * Two, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};
  add0.y.dtype = ge::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = add0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadNeedLoopModeFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_loop_mode_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_need_loop_mode");
  CreateLoadLoopModeFusedGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}


/**
 *      data
 *         |
 *      AscBc
 *         /
 *      netoutput
 */
ge::ComputeGraphPtr ShareGraph::LoadUnalignPadFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_unalign_pad_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 2);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_abs");
  CreateLoadUnalignPadAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        add
 *       /   \
 *   load0   Scalar
 *     |
 *   data0
 */
static void CreateScalarInfAddAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("inf");

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = scalar0.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = add.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::ScalarInfAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("scalar_inf_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("scalar_inf_add");
  CreateScalarInfAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        div
 *       /   \
 *   Scalar  load0
 *             |
 *           data0
 */
static void CreateScalarInfDivAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("inf");

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Div div("div");
  div.x1 = scalar0.y;
  div.x2 = x1Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = div.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::ScalarDivInfFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("scalar_div_inf_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("scalar_inf_div");
  CreateScalarInfDivAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        gelu
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddGeluAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Gelu gelu("gelu");
  gelu.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = gelu.y;
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddGeluFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_gelu_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_gelu");
  CreateAddGeluAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *               output
 *                 |
 *               store
 *                 |
 *              compare
 *              /       \
 *         load0      scalar0/data1->load1
 *           |
 *         data0
 */
static void CreateCompareEqAscGraph(ge::AscGraph &graph, size_t dims_size, bool is_second_input_tensor, ge::DataType dtype) {
  ge::ascir_op::Data x1("compare_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  ge::ascir_op::Eq eq("eq");
  eq.x1 = x1Local.y;
  if (is_second_input_tensor) {
    ge::ascir_op::Data x2("compare_data1", graph);
    x2.ir_attr.SetIndex(1);
    x2.y.dtype = dtype;
    ge::ascir_op::Load x2Local("load2");
    x2Local.x = x2.y;
    x2Local.y.dtype = dtype;
    eq.x2 = x2Local.y;
  } else {
    ge::ascir_op::Scalar scalar0("scalar0", graph);
    scalar0.ir_attr.SetValue("1");
    eq.x2 = scalar0.y;
  }

  eq.y.dtype =  ge::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = eq.y;
  x_out.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_UINT8;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateCompareGtAscGraph(ge::AscGraph &graph, size_t dims_size, bool is_second_input_tensor, ge::DataType dtype) {
  ge::ascir_op::Data x1("compare_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  ge::ascir_op::Gt gt("gt");
  gt.x1 = x1Local.y;
  if (is_second_input_tensor) {
    ge::ascir_op::Data x2("compare_data1", graph);
    x2.ir_attr.SetIndex(1);
    x2.y.dtype = dtype;
    ge::ascir_op::Load x2Local("load2");
    x2Local.x = x2.y;
    x2Local.y.dtype = dtype;
    gt.x2 = x2Local.y;
  } else {
    ge::ascir_op::Scalar scalar0("scalar0", graph);
    scalar0.ir_attr.SetValue("1");
    gt.x2 = scalar0.y;
  }

  gt.y.dtype =  ge::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = gt.y;
  x_out.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_UINT8;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *               output
 *                 |
 *               store
 *                 |
 *              concat
 *               /   \
 *           load0  load1
 *             |      |
 *           data0   data1
 */
static void CreateConcatAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("concat_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Data x2("concat_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype =  ge::DT_UINT8;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype =  ge::DT_UINT8;

  ge::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = concat.y;
  x_out.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_UINT8;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
  auto concat_node = graph.FindNode("concat");
  auto size = concat_node->attr.sched.axis.size();
  auto repeats = concat_node->outputs()[0]->attr.repeats;
  repeats[size - 1] = repeats[size - 1] + repeats[size - 1];
  auto strides = concat_node->outputs()[0]->attr.strides;
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  concat_node->outputs()[0]->attr.strides = strides;
  concat_node->outputs()[0]->attr.repeats = repeats;
  auto store_node = graph.FindNode("store");
  store_node->outputs()[0]->attr.strides = strides;
  store_node->outputs()[0]->attr.repeats = repeats;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *       /     \
 *   load0      sub
 *     |       /    \
 *   data0  scalar  scalar
 */
 static void CreateSubAbsAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x0("data0", graph);
  x0.ir_attr.SetIndex(0);
  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x0.y;

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("2");
  ge::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("1");

  ge::ascir_op::Sub sub("subs");
  sub.x1 = scalar0.y;
  sub.x2 = scalar1.y;

  ge::ascir_op::Add add("add");
  add.x1 = sub.y;
  add.x2 = x1Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = add.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *         |
 *       data0
 */
 ge::ComputeGraphPtr ShareGraph::SubAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sub_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sub_abs");
  CreateSubAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       add
 *        |
 *    transpose
 *        |
 *       sub
 *      /   \
 *   data0  data1
 */
static void CreateSubTransposeAbsAscGraph(ge::AscGraph &graph, size_t dims_size, vector<size_t> perms) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Sub sub("subs");
  sub.x1 = x1Local.y;
  sub.x2 = x2Local.y;

  ge::ascir_op::Transpose transpose("transpose");
  transpose.x = sub.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = transpose.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  ConstructVVAscGraphAxisInfo(graph, dims_size, perms);
}

static void CreateSubTransposeAbsConstAscGraph(ge::AscGraph &graph, size_t dims_size, vector<size_t> perms, std::vector<int> dims) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Sub sub("subs");
  sub.x1 = x1Local.y;
  sub.x2 = x2Local.y;

  ge::ascir_op::Transpose transpose("transpose");
  transpose.x = sub.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = transpose.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  ConstructVVConstAscGraphAxisInfo(graph, dims_size, perms, dims);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *      |    |
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::SubTransposeAbsFusedGraph(size_t dims_size, vector<size_t> perms) {
  auto builder = GraphBuilder("sub_transpose_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sub_transpose_abs");
  CreateSubTransposeAbsAscGraph(sub_graph, dims_size, perms);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *      |    |
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::SubTransposeAbsFusedConstGraph(size_t dims_size, vector<size_t> perms, std::vector<int> dims) {
  auto builder = GraphBuilder("sub_transpose_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sub_transpose_abs_const");
  CreateSubTransposeAbsConstAscGraph(sub_graph, dims_size, perms, dims);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *       NetOutput
 *           |
 *         asbc1
 *         /   \
 *     asbc0   data1
 *        |
 *      data0
 */
ge::ComputeGraphPtr ShareGraph::CompareFusedGraph(size_t dims_size, bool is_second_input_tensor, ge::DataType dtype,
                                                  std::string mode) {
  auto builder = GraphBuilder("compare_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  NodePtr data2;
  if (is_second_input_tensor) {
    data2 = builder.AddNode("data2", "Data", 0, 1);
    ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  }

  int input_num = 1;
  if (is_second_input_tensor) {
    input_num = 2;
  }
  auto ascbc0 = builder.AddNode("ascbc0", "AscGraph", input_num, 1);
  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc0, 0);
  if (is_second_input_tensor) {
    builder.AddDataEdge(data2, 0, ascbc0, 1);
  }
  builder.AddDataEdge(ascbc0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(ascbc1, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node0 = compute_graph->FindNode("ascbc0");
  ge::AscGraph sub_graph0("compare");
  if (mode == "eq") {
    CreateCompareEqAscGraph(sub_graph0, dims_size, is_second_input_tensor, dtype);
  } else if (mode == "gt") {
    CreateCompareGtAscGraph(sub_graph0, dims_size, is_second_input_tensor, dtype);
  }

  std::string sub_graph_str0;
  ge::AscGraphUtils::SerializeToReadable(sub_graph0, sub_graph_str0);
  ge::AttrUtils::SetStr(ascbc_node0->GetOpDescBarePtr(), "ascgraph", sub_graph_str0);

  auto ascbc_node1 = compute_graph->FindNode("ascbc1");
  ge::AscGraph sub_graph1("concat");
  CreateConcatAscGraph(sub_graph1, dims_size);

  std::string sub_graph_str1;
  ge::AscGraphUtils::SerializeToReadable(sub_graph1, sub_graph_str1);
  ge::AttrUtils::SetStr(ascbc_node1->GetOpDescBarePtr(), "ascgraph", sub_graph_str1);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        neg
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddNegAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Neg neg("neg");
  neg.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = neg.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AddNegFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_neg_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_neg");
  CreateAddNegAscGraph(sub_graph, dims_size);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                   output
 *                     |
 *                   store
 *                     |
 *                    mul
 *                /          \
 *              add0        load2
 *            /       \       |
 *       broadcast1  load0  data2
 *            |        |
 *          load1    data0
 *            |
 *          data1
 */
static void CreateBrcInlineAscGraph(ge::AscGraph& graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(320);
  const Expression s1 = graph.CreateSizeVar(32);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast1.attr.api.type = ge::ApiType::kAPITypeCompute;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, One};
  broadcast1.y.dtype = ge::DataType::DT_FLOAT;
  broadcast1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = broadcast1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};
  add0.y.dtype = ge::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x2.y.dtype = ge::DataType::DT_FLOAT;
  x2.ir_attr.SetIndex(2);

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  load2.y.dtype = ge::DataType::DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = load2.y;
  mul0.x2 = add0.y;
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.attr.api.type = ge::ApiType::kAPITypeCompute;
  mul0.attr.sched.axis = {z0.id, z1.id};
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};
  mul0.y.dtype = ge::DataType::DT_FLOAT;
  mul0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = mul0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::BrcInlineFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("brc_inline_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("brc_inline");
  CreateBrcInlineAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 *  load0 load1  load2
 *    |     |     |
 *  data0 data1  data2
 */
static void CreateLoadWhereStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  ge::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);


  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  ge::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = x3Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateLoadWhereReduceStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);
  x3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x3.y.axis = {z0.id, z1.id, z2.id};
  *x3.y.repeats = {s0, s1, s2};
  *x3.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;
  x3Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x3Local.y.axis = {z0.id, z1.id, z2.id};
  *x3Local.y.repeats = {s0, s1, s2};
  *x3Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = x3Local.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, ge::ops::One};
  *max.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Store x_out("store");
  x_out.x = max.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, ge::ops::One};
  *x_out.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

static void CreateLoadWhereReduceX2IsUbscalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");

  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = x2Local.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, ge::ops::One};
  *max.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, ge::ops::One};
  *x_out.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

static void CreateLoadWhereReduceX3IsUbscalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("200");

  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = scalar0.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, ge::ops::One};
  *max.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, ge::ops::One};
  *x_out.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

static void CreateLoadWhereReduceX2X3IsUbscalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");
  ge::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("200");

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = scalar1.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, ge::ops::One};

  ge::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, ge::ops::One};
  *max.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, ge::ops::One};
  *x_out.y.strides = {s1, ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 *  load0 load1  load2
 *    |     |     |
 *  data0 data1  data2
 */
ge::ComputeGraphPtr ShareGraph::LoadWhereStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_where_store");
  CreateLoadWhereStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

ge::ComputeGraphPtr ShareGraph::LoadWhereReduceStoreFusedGraph(size_t dims_size, bool x2_scalar, bool x3_scalar) {
  auto builder = GraphBuilder("load_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_where_store");
  if (x2_scalar && x3_scalar) {
    CreateLoadWhereReduceX2X3IsUbscalarStoreAscGraph(sub_graph, dims_size);
  } else if (x2_scalar) {
    CreateLoadWhereReduceX2IsUbscalarStoreAscGraph(sub_graph, dims_size);
  } else if (x3_scalar) {
    CreateLoadWhereReduceX3IsUbscalarStoreAscGraph(sub_graph, dims_size);
  } else {
    CreateLoadWhereReduceStoreAscGraph(sub_graph, dims_size);
  }

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 * load0 scalar0 scalar1
 *    |
 * data0
 */
static void CreateLoadWhereX2X3IsUbscalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");
  ge::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("200");

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = scalar1.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 * load0 scalar0 scalar1
 *    |
 * data0
 */
ge::ComputeGraphPtr ShareGraph::LoadWhereX2X3IsUbscalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_x2_x3_is_ubscalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_where_x2_x3_is_ubscalar_store");
  CreateLoadWhereX2X3IsUbscalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 * load0 scalar0 load1
 *    |             |
 * data0          data1
 */
static void CreateLoadWhereX2IsUbscalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = x2Local.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 * load0 scalar0 load1
 *    |            |
 * data0          data1
 */
ge::ComputeGraphPtr ShareGraph::LoadWhereX2IsUbscalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_x2_is_ubscalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_where_x2_is_ubscalar_store");
  CreateLoadWhereX2IsUbscalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
* load0 load1  scalar0
 *    |    |
 * data0  data1
 */
static void CreateLoadWhereX3IsUbscalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("200");

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_UINT8;
  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = scalar0.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = where.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 * load0 load1  scalar0
 *    |    |
 * data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadWhereX3IsUbscalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_x3_is_ubscalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_where_x3_is_ubscalar_store");
  CreateLoadWhereX3IsUbscalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *       logical_not
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadLogicalNotStoreAscGraph(ge::AscGraph &graph, size_t dims_size, DataType dt_in = ge::DT_FLOAT,
                                              DataType dt_out = ge::DT_FLOAT) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dt_in;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dt_in;

  ge::ascir_op::LogicalNot logical_not("logical_not");
  logical_not.x = x1Local.y;
  logical_not.y.dtype = dt_out;

  ge::ascir_op::Store logical_not_store("store");
  logical_not_store.x = logical_not.y;
  logical_not_store.y.dtype = dt_out;

  ge::ascir_op::Output y("output");
  y.x = logical_not_store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      data
 *         |
 *      AscBc
 *         /
 *      netoutput
 */
ge::ComputeGraphPtr ShareGraph::LoadLogicalNotStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_logical_not_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_logical_not_store");
  CreateLoadLogicalNotStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

ge::ComputeGraphPtr ShareGraph::LoadLogicalNotStoreFusedGraph(size_t dims_size, DataType dt_in, DataType dt_out) {
  auto builder = GraphBuilder("load_logical_not_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_logical_not_store");
  CreateLoadLogicalNotStoreAscGraph(sub_graph, dims_size, dt_in, dt_out);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    bitwiseand
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadBitwiseAndStoreAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  ge::ascir_op::BitwiseAnd bitwiseand("bitwiseand");
  bitwiseand.x1 = x1Local.y;
  bitwiseand.x2 = x2Local.y;
  bitwiseand.y.dtype = out_dtype;

  ge::ascir_op::Store x_out("store");
  x_out.x = bitwiseand.y;
  x_out.y.dtype = out_dtype;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadBitwiseAndStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_and_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_bitwise_and_store");
  CreateLoadBitwiseAndStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    bitwiseor
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadBitwiseOrStoreAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  ge::ascir_op::BitwiseOr bitwiseor("bitwiseor");
  bitwiseor.x1 = x1Local.y;
  bitwiseor.x2 = x2Local.y;
  bitwiseor.y.dtype = out_dtype;

  ge::ascir_op::Store x_out("store");
  x_out.x = bitwiseor.y;
  x_out.y.dtype = out_dtype;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadBitwiseOrStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_or_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_bitwise_or_store");
  CreateLoadBitwiseOrStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    bitwisexor
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadBitwiseXorStoreAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  ge::ascir_op::BitwiseXor bitwisexor("bitwisexor");
  bitwisexor.x1 = x1Local.y;
  bitwisexor.x2 = x2Local.y;
  bitwisexor.y.dtype = out_dtype;

  ge::ascir_op::Store x_out("store");
  x_out.x = bitwisexor.y;
  x_out.y.dtype = out_dtype;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadBitwiseXorStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_xor_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_bitwise_xor_store");
  CreateLoadBitwiseXorStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *       output
 *         |
 *       store
 *         |
 *     bitwisenot
 *         |
 *       load0
 *         |
 *       data0
 */
static void CreateLoadBitwiseNotStoreAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  ge::ascir_op::Data x("data", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = in_dtype;

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = in_dtype;

  ge::ascir_op::BitwiseNot bitwisenot("bitwisenot");
  bitwisenot.x = xLocal.y;
  bitwisenot.y.dtype = out_dtype;

  ge::ascir_op::Store x_out("store");
  x_out.x = bitwisenot.y;
  x_out.y.dtype = out_dtype;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadBitwiseNotStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_not_store_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_bitwise_not_store");
  CreateLoadBitwiseNotStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        Rsqrt
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddRsqrtAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  ge::ascir_op::Rsqrt rsqrt("rsqrt");
  rsqrt.x = add.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = rsqrt.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::AddRsqrtFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_rsqrt_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_rsqrt");
  CreateAddRsqrtAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/** BABAB
 *              output
 *                |
 *              store
 *                |
 *              add0
 *              /   \
 *            brc2  load0
 *             /      \
 *           brc1   data0
 *            /
 *          brc0
 *            |
 *          load1
 *            |
 *          data1
 */
static void CreateContinuesBrcAscGraph(ge::AscGraph& graph, size_t dims_size) { // 4, 8, 16, 64, 32
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(16);
  const Expression s3 = graph.CreateSizeVar(64);
  const Expression s4 = graph.CreateSizeVar(32);

  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  ge::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, s1, s2, s3, s4};
  *load0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.y.dtype = ge::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.repeats = {One, s1, One, s3, One};
  *load1.y.strides = {Zero, s3, Zero, One, Zero};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, s1, One, s3, One};
  *brc0.y.strides = {s1 * s3, s3, Zero, One, Zero};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, One};
  *brc1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};
  brc1.y.dtype = ge::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {s0, s1, s2, s3, s4};
  *brc2.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  brc2.y.dtype = ge::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store1("store1");
  store1.x = brc2.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = ge::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {s0, s1, s2, s3, s4};
  *store1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  store1.y.dtype = ge::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(1);


  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc2.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ge::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  add0.y.dtype = ge::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = add0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::ContinuesBrcFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("continues_brc_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("continues_brc");
  CreateContinuesBrcAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                   output
 *                     |
 *                   store
 *                     |
 *                    mul
 *                /          \
 *              add0        load2
 *            /       \       |
 *       broadcast2  load0  data2
 *            |        |
 *       broadcast1    |
 *            |        |
 *       broadcast0    |
 *            |        |
 *          load1    data0
 *            |
 *          data1
 */
static void CreateLoadBrcAscGraphSevenDim(ge::AscGraph& graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(20);
  const Expression s1 = graph.CreateSizeVar(2);
  const Expression s2 = graph.CreateSizeVar(2);
  const Expression s3 = graph.CreateSizeVar(2);
  const Expression s4 = graph.CreateSizeVar(2);
  const Expression s5 = graph.CreateSizeVar(2);
  const Expression s6 = graph.CreateSizeVar(2);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);
  std::vector<Expression> all_size_var = {s0, s1 ,s2, s3, s4, s5, s6};
  Expression tmp_stride = One;
  std::vector<bool> is_broadcast_axis = {false, true, false, true, false, true, false};

  ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = DT_FLOAT;

  ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  for (auto it = all_size_var.rbegin(); it != all_size_var.rend(); ++it) {
    load0.y.repeats->insert(load0.y.repeats->begin(), *it);
    load0.y.strides->insert(load0.y.strides->begin(), tmp_stride);
    tmp_stride = tmp_stride * (*it);
  }
  load0.y.dtype = DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ApiType::kAPITypeBuffer;
  x1.y.dtype = DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  tmp_stride = One;
  for (int i = static_cast<int>(is_broadcast_axis.size()) - 1; i >= 0; --i) {
    if (is_broadcast_axis[i]) {
      load1.y.repeats->insert(load1.y.repeats->begin(), One);
      load1.y.strides->insert(load1.y.strides->begin(), Zero);
    } else {
      load1.y.repeats->insert(load1.y.repeats->begin(), all_size_var[i]);
      load1.y.strides->insert(load1.y.strides->begin(), tmp_stride);
      tmp_stride = tmp_stride * all_size_var[i];
    }
  }
  load1.y.dtype = DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = load1.y;
  broadcast0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast0.attr.api.type = ApiType::kAPITypeCompute;
  broadcast0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id,};
  *broadcast0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast0.y.repeats = {s0, One, s2, One, s4, s5, s6};
  *broadcast0.y.strides = {s2 * s4 * s5 * s6, Zero, s4 * s5 * s6, Zero, s5 * s6, s6, One};
  broadcast0.y.dtype = DT_FLOAT;
  broadcast0.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast1.attr.api.type = ApiType::kAPITypeCompute;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast1.y.repeats = {s0, One, s2, s3, s4, s5, s6};
  *broadcast1.y.strides = {s2 * s3 * s4 * s5 * s6, Zero, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  broadcast1.y.dtype = DT_FLOAT;
  broadcast1.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = broadcast1.y;
  broadcast2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast2.attr.api.type = ApiType::kAPITypeCompute;
  broadcast2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast2.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *broadcast2.y.strides = {s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  broadcast2.y.dtype = DT_FLOAT;
  broadcast2.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = broadcast2.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *add0.y.strides = {s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  add0.y.dtype = DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ApiType::kAPITypeBuffer;
  x2.y.dtype = DT_FLOAT;
  x2.ir_attr.SetIndex(2);

  ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load2.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *load2.y.strides = {s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  load2.y.dtype = DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Mul mul0("mul0");
  mul0.x1 = load2.y;
  mul0.x2 = add0.y;
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.attr.api.type = ApiType::kAPITypeCompute;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *mul0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *mul0.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *mul0.y.strides = {s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  mul0.y.dtype = DT_FLOAT;
  mul0.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Store store("store");
  store.x = mul0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *store.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *store.y.strides = {s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  store.y.dtype = DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ApiType::kAPITypeBuffer;
  y.y.dtype = DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadBrcFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_brc_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_brc");
  CreateLoadBrcAscGraphSevenDim(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateScalarBrcAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store1("store1");
  store1.x = brc0.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = ge::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id};
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};
  store1.y.dtype = ge::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(0);

}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::ScalarBrcFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("scalar_brc_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("continues_brc");
  CreateScalarBrcAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  scalar0
 */
ge::ComputeGraphPtr ShareGraph::AbsBrcAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_brc_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("abs_brc_add");
  CreateAbsBrcAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  scalar0
 */
ge::ComputeGraphPtr ShareGraph::UbScalarBrcAbsAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("ub_scalar_brc_abs_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("ub_scalar_brc_abs_add");
  CreateUbScalerBrcAbsAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *       LeakyRelu
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadLeakyReluStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::LeakyRelu leakyrelu0("leakyrelu0");
  leakyrelu0.x = x1Local.y;
  leakyrelu0.ir_attr.SetNegative_slope(0.8);

  ge::ascir_op::Store leakyrelu0_store("store");
  leakyrelu0_store.x = leakyrelu0.y;

  ge::ascir_op::Output y("output");
  y.x = leakyrelu0_store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadLeakyReluStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_leaky_relu_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_leaky_relu_store");
  CreateLoadLeakyReluStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}


/**
 *         data0
 *           |
 *         load0
 *           |
 *        Sigmoid
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadSigmoidStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Sigmoid sigmoid0("sigmoid0");
  sigmoid0.x = x1Local.y;

  ge::ascir_op::Store sigmoid0_store("store");
  sigmoid0_store.x = sigmoid0.y;

  ge::ascir_op::Output y("output");
  y.x = sigmoid0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadSigmoidStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_sigmoid_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_sigmoid_store");
  CreateLoadSigmoidStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *          GE
 *           |
 *         Cast
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadCompareStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x1.y.dtype = ge::DT_FLOAT;
  x2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Ge ge0("ge0");
  ge0.x1 = x1Local.y;
  ge0.x2 = x2Local.y;
  ge0.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Cast cast0("cast0");
  cast0.x = ge0.y;
  cast0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store store0("store0");
  store0.x = cast0.y;
  store0.y.dtype = ge::DT_FLOAT;
  ge::ascir_op::Output y("output");
  y.x = store0.y;
  y.y.dtype = ge::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadCompareStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_compare_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_compare_store");
  CreateLoadCompareStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *          Erf
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadErfStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Erf sigmoid0("sigmoid0");
  sigmoid0.x = x1Local.y;

  ge::ascir_op::Store sigmoid0_store("store");
  sigmoid0_store.x = sigmoid0.y;

  ge::ascir_op::Output y("output");
  y.x = sigmoid0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadErfStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_erf_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_erf_store");
  CreateLoadErfStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateBrcReduceAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(7);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, One, s2};
  *load1.y.strides = {s2, Zero, One};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs("abs");
  abs.x = load1.y;
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, One, s2};
  *abs.y.strides = {s2, Zero, One};
  abs.y.dtype = ge::DataType::DT_FLOAT;
  abs.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Max max("max");
  max.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}, {{ge::Symbol(8192), 0}, MemAttr(), 1}};
  max.x = brc0.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, One, s2};
  *max.y.strides = {s2, Zero, One};

  ge::ascir_op::Store store1("store1");
  store1.x = max.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = ge::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, One, s2};
  *store1.y.strides = {s2, Zero, One};
  store1.y.dtype = ge::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(0);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *          Tanh
 *           |
 *         store
 *           |
 *         ouput0
 */
static void CreateLoadTanhStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Tanh sigmoid0("sigmoid0");
  sigmoid0.x = x1Local.y;

  ge::ascir_op::Store sigmoid0_store("store");
  sigmoid0_store.x = sigmoid0.y;

  ge::ascir_op::Output y("output");
  y.x = sigmoid0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateLoadTanhBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Tanh tanh("tanh");
  tanh.x = xLocal.y;
  tanh.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = tanh.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadTanhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_tanh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_tanh_bf16_test");
  CreateLoadTanhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadTanhStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_tanh_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_tanh_store");
  CreateLoadTanhStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSinBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data0", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = ge::DataType::DT_BF16;
  x1Local.x = x.y;

  ge::ascir_op::Sin sin("sin");
  sin.x = x1Local.y;
  sin.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store sin_store("store");
  sin_store.x = sin.y;
  sin_store.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = sin_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::BF16SinFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sin_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sin_bf16_test");
  CreateSinBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSqrtBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data0", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = ge::DataType::DT_BF16;
  x1Local.x = x.y;

  ge::ascir_op::Sqrt sqrt("sqrt");
  sqrt.x = x1Local.y;
  sqrt.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store sqrt_store("store");
  sqrt_store.x = sqrt.y;
  sqrt_store.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = sqrt_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::BF16SqrtFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sqrt_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sqrt_bf16_test");
  CreateSqrtBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRsqrtBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data0", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = ge::DataType::DT_BF16;
  x1Local.x = x.y;

  ge::ascir_op::Rsqrt rsqrt("rsqrt");
  rsqrt.x = x1Local.y;
  rsqrt.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store rsqrt_store("store");
  rsqrt_store.x = rsqrt.y;
  rsqrt_store.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = rsqrt_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::BF16RsqrtFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("rsqrt_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("rsqrt_bf16_test");
  CreateRsqrtBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSigmoidBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data0", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = ge::DataType::DT_BF16;
  x1Local.x = x.y;

  ge::ascir_op::Sigmoid sigmoid("sigmoid");
  sigmoid.x = x1Local.y;
  sigmoid.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store sigmoid_store("store");
  sigmoid_store.x = sigmoid.y;
  sigmoid_store.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = sigmoid_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::BF16SigmoidFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sigmoid_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sigmoid_bf16_test");
  CreateSigmoidBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::BrcReduceFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("brc_reduce_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("brc_reduce");
  CreateBrcReduceAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTailBrcTailReduceAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(7);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Abs abs("abs");
  abs.x = load1.y;
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, One};
  *abs.y.strides = {s1, One, Zero};
  abs.y.dtype = ge::DataType::DT_FLOAT;
  abs.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Max max("max");
  graph.AddNode(max);
  max.x = brc0.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, One};
  *max.y.strides = {s1, One, Zero};
  max.attr.tmp_buffers = {{{ge::Symbol(8192), -1}, MemAttr(), 0}, {{ge::Symbol(8192), 0}, MemAttr(), 1}};

  ge::ascir_op::Store store1("store1");
  store1.x = max.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = ge::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, One};
  *store1.y.strides = {s1, One, Zero};
  store1.y.dtype = ge::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y1.y.dtype = ge::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(0);
}

/**
 *      output
 *         |
 *       store
 *         |
 *      matmul
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateMatMulAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::MatMul matmul("matmul");
  matmul.x1 = x1Local.y;
  matmul.x2 = x2Local.y;
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  ge::ascir_op::Store x_out("store");
  x_out.x = matmul.y;
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::TailBrcTailReduceFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("tail_brc_tail_reduce_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("tail_brc_tail_reduce");
  CreateTailBrcTailReduceAscGraph(sub_graph, dims_size);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateFloorDivMulLessEqualSelectAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x0("data0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ge::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};
  load0.y.dtype = ge::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x1("data1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x1.ir_attr.SetIndex(1);
  x1.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {One, One, s2};
  *load1.y.strides = {Zero, Zero, One};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {One, s1, s2};
  *brc0.y.strides = {Zero, s2, One};
  brc0.y.dtype = ge::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s0 * s1, s2, One};
  brc1.y.dtype = ge::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::FloorDiv floor_div0("floor_div0");
  floor_div0.x1 = load0.y;
  floor_div0.x2 = brc1.y;
  floor_div0.attr.api.compute_type = ComputeType::kComputeElewise;
  floor_div0.attr.api.type = ge::ApiType::kAPITypeCompute;
  floor_div0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *floor_div0.y.axis = {z0.id, z1.id, z2.id};
  *floor_div0.y.repeats = {s0, s1, s2};
  *floor_div0.y.strides = {s0 * s1, s2, One};
  floor_div0.y.dtype = ge::DataType::DT_FLOAT;
  floor_div0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Data x2("data2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ge::ApiType::kAPITypeBuffer;
  x2.ir_attr.SetIndex(2);
  x2.y.dtype = ge::DataType::DT_FLOAT;

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, One, One};
  *load2.y.strides = {Zero, Zero, Zero};
  load2.y.dtype = ge::DataType::DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Exp exp0("exp0");
  exp0.x = load2.y;
  exp0.attr.api.compute_type = ComputeType::kComputeLoad;
  exp0.attr.api.type = ge::ApiType::kAPITypeCompute;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *exp0.y.axis = {z0.id, z1.id, z2.id};
  *exp0.y.repeats = {One, One, One};
  *exp0.y.strides =  {Zero, Zero, Zero};
  exp0.y.dtype = ge::DataType::DT_FLOAT;
  exp0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = exp0.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {One, One, s2};
  *brc2.y.strides = {Zero, Zero, One};
  brc2.y.dtype = ge::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc3("brc3");
  brc3.x = brc2.y;
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  *brc3.y.repeats = {One, s1, s2};
  *brc3.y.strides = {Zero, s2, One};
  brc3.y.dtype = ge::DataType::DT_FLOAT;
  brc3.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc4("brc4");
  brc4.x = brc3.y;
  brc4.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc4.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc4.y.axis = {z0.id, z1.id, z2.id};
  *brc4.y.repeats = {s0, s1, s2};
  *brc4.y.strides = {s1 * s2, s2, One};
  brc4.y.dtype = ge::DataType::DT_FLOAT;
  brc4.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Mul mul("mul");
  mul.x1 = brc4.y;
  mul.x2 = floor_div0.y;
  mul.attr.api.compute_type = ComputeType::kComputeElewise;
  mul.attr.api.type = ge::ApiType::kAPITypeCompute;
  mul.attr.sched.axis = {z0.id, z1.id, z2.id};
  *mul.y.axis = {z0.id, z1.id, z2.id};
  *mul.y.repeats = {s0, s1, s2};
  *mul.y.strides = {s1 * s2, s1, One};
  mul.y.dtype = ge::DataType::DT_FLOAT;
  mul.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetIndex(3);
  scalar.ir_attr.SetValue("1.0");

  ge::ascir_op::Broadcast brc5("brc5");
  brc5.x = scalar.y;
  brc5.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc5.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc5.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc5.y.axis = {z0.id, z1.id, z2.id};
  *brc5.y.repeats = {One, One, s2};
  *brc5.y.strides = {Zero, Zero, One};
  brc5.y.dtype = ge::DataType::DT_FLOAT;
  brc5.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc6("brc6");
  brc6.x = brc5.y;
  brc6.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc6.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc6.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc6.y.axis = {z0.id, z1.id, z2.id};
  *brc6.y.repeats = {One, s1, s2};
  *brc6.y.strides = {Zero, s2, One};
  brc6.y.dtype = ge::DataType::DT_FLOAT;
  brc6.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Broadcast brc7("brc7");
  brc7.x = brc6.y;
  brc7.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc7.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc7.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc7.y.axis = {z0.id, z1.id, z2.id};
  *brc7.y.repeats = {s0, s1, s2};
  *brc7.y.strides = {s1 * s2, s2, One};
  brc7.y.dtype = ge::DataType::DT_FLOAT;
  brc7.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Le le0("le0");
  le0.x1 = mul.y;
  le0.x2 = brc7.y;
  le0.attr.api.compute_type = ComputeType::kComputeInvalid;
  le0.attr.api.type = ge::ApiType::kAPITypeInvalid;
  le0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *le0.y.axis = {z0.id, z1.id, z2.id};
  *le0.y.repeats = {s0, s1, s2};
  *le0.y.strides = {s1 * s2, s2, One};
  le0.y.dtype = ge::DataType::DT_UINT8;
  le0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Where where0("where0");
  where0.x1 = le0.y;
  where0.x2 = floor_div0.y;
  where0.x3 = load0.y;
  where0.attr.api.compute_type = ComputeType::kComputeInvalid;
  where0.attr.api.type = ge::ApiType::kAPITypeInvalid;
  where0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where0.y.axis = {z0.id, z1.id, z2.id};
  *where0.y.repeats = {s0, s1, s2};
  *where0.y.strides = {s1 * s2, s2, One};
  where0.y.dtype = ge::DataType::DT_FLOAT;
  where0.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = where0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s1, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

ge::ComputeGraphPtr ShareGraph::MatMulFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("matmul_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("matmul");
  CreateMatMulAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

ge::ComputeGraphPtr ShareGraph::FloorDivMulLessEqualSelectFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("floordiv_mul_le_select_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("floordiv_mul_le_select");
  CreateFloorDivMulLessEqualSelectAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        Pow
 *     /      \
 *   brc1    brc2
 *    |       |
 *  load1  load2
 *    |      |
 * scalar0  scalar1
 */
static void CreateLoadPowAllInputIsScalarStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x1("scalar1", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ge::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};
  load1.y.dtype = ge::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id};
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};
  brc1.y.dtype = ge::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Data x2("scalar2", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ge::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {One, One};
  *load2.y.strides = {Zero, Zero};
  load2.y.dtype = ge::DataType::DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ge::ascir_op::Broadcast brc2("brc2");
  brc2.x = load2.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = ge::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id};
  *brc2.y.axis = {z0.id, z1.id};
  *brc2.y.repeats = {s0, s1};
  *brc2.y.strides = {s1, One};
  brc2.y.dtype = ge::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Pow pow("pow");
  pow.x1 = brc1.y;
  pow.x2 = brc2.y;
  pow.attr.api.compute_type = ComputeType::kComputeElewise;
  pow.attr.api.type = ge::ApiType::kAPITypeCompute;
  pow.attr.sched.axis = {z0.id, z1.id};
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, One};
  pow.y.dtype = ge::DataType::DT_FLOAT;
  pow.attr.api.unit = ComputeUnit::kUnitVector;

  ge::ascir_op::Store store("store");
  store.x = pow.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ge::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = ge::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ge::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *      /     \
 *  data0    data1
 */
ge::ComputeGraphPtr ShareGraph::LoadPowAllInputIsScalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_pow_all_input_is_scalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_pow_all_input_is_scalar_store");
  CreateLoadPowAllInputIsScalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

ge::ComputeGraphPtr ShareGraph::AddAbsFusedConstGraph(size_t dims_size, std::vector<int> dims) {
  auto builder = GraphBuilder("add_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("add_abs");
  CreateAddAbsConstAscGraph(sub_graph, dims_size, dims);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

void ShareGraph::ConcatAscGraph(AscGraph &graph, const vector<std::string> &dim_sizes, bool align) {
  ge::ascir_op::Data x1("concat_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_INT32;

  ge::ascir_op::Data x2("concat_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_INT32;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype =  ge::DT_INT32;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype =  ge::DT_INT32;

  ge::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.y.dtype = ge::DT_INT32;

  ge::ascir_op::Store x_out("store");
  x_out.x = concat.y;
  x_out.y.dtype = ge::DT_INT32;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_INT32;

  ConstructVVAscGraphAxisInfo(graph, dim_sizes, align);
  auto concat_node = graph.FindNode("concat");
  auto size = concat_node->attr.sched.axis.size();
  auto repeats = concat_node->outputs()[0]->attr.repeats;
  repeats[size - 1] = repeats[size - 1] + repeats[size - 1];
  auto strides = concat_node->outputs()[0]->attr.strides;
  for (int i = dim_sizes.size() - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  concat_node->outputs()[0]->attr.strides = strides;
  concat_node->outputs()[0]->attr.repeats = repeats;
  auto store_node = graph.FindNode("store");
  store_node->outputs()[0]->attr.strides = strides;
  store_node->outputs()[0]->attr.repeats = repeats;
}
static void LoadGatherAbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type) {
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

   ge::ascir_op::Data x1("x1");
   graph.AddNode(x1);
   x1.y.dtype = data_type;
   x1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
   *x1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
   *x1.y.repeats = {s0, s1, s2, s3, s4};
   *x1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, ge::ops::One};
   x1.ir_attr.SetIndex(0);

   ge::ascir_op::Data x2("x2");
   graph.AddNode(x2);
   x2.y.dtype = ge::DT_INT32;
   x2.attr.sched.axis = {z5.id, z6.id};
   *x2.y.axis = {z5.id, z6.id};
   *x2.y.repeats = {s5, s6};
   *x2.y.strides = {s6, ge::ops::One};
   x2.ir_attr.SetIndex(1);

   ge::ascir_op::Gather gather("gather");
   graph.AddNode(gather);
   gather.x1 = x1.y;
   gather.x2 = x2.y;
   gather.ir_attr.SetAxis(gather_axis);
   gather.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
   *gather.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
   *gather.y.repeats = {s0, s1, s5, s6, s3, s4};
   *gather.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, ge::ops::One};

   ge::ascir_op::Abs abs("abs");
   graph.AddNode(abs);
   abs.x = gather.y;
   abs.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
   *abs.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
   *abs.y.repeats = {s0, s1, s5, s6, s3, s4};
   *abs.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, ge::ops::One};

   ge::ascir_op::Store store("store");
   graph.AddNode(store);
   store.x = abs.y;
   store.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
   *store.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
   *store.y.repeats = {s0, s1, s5, s6, s3, s4};
   *store.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, ge::ops::One};

   ge::ascir_op::Output y("y");
   graph.AddNode(y);
   y.x = store.y;
   y.y.dtype = data_type;
   y.ir_attr.SetIndex(0);
 }

 ge::ComputeGraphPtr ShareGraph::LoadGatherAbsStore(int64_t gather_axis, ge::DataType data_type) {
   auto builder = GraphBuilder("load_gather_abs_store_store_test");
   auto data0 = builder.AddNode("data0", "Data", 0, 1);
   ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
   auto data1 = builder.AddNode("data1", "Data", 0, 1);
   ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

   auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
   auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

   builder.AddDataEdge(data0, 0, ascbc, 0);
   builder.AddDataEdge(data1, 0, ascbc, 1);
   builder.AddDataEdge(ascbc, 0, netoutput, 0);
   ComputeGraphPtr compute_graph = builder.GetGraph();
   if (compute_graph == nullptr) {
     return nullptr;
   }
   auto ascbc_node = compute_graph->FindNode("ascbc");
   ge::AscGraph sub_graph("load_gather_abs_store_store_test");
   LoadGatherAbsStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

   std::string sub_graph_str;
   ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
   ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
   return compute_graph;
 }

static void LoadGatherTailAbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type) {
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

   ge::ascir_op::Data x1("x1");
   graph.AddNode(x1);
   x1.y.dtype = data_type;
   x1.attr.sched.axis = {z0.id, z1.id, z2.id};
   *x1.y.axis = {z0.id, z1.id, z2.id};
   *x1.y.repeats = {s0, s1, s2};
   *x1.y.strides = {s1 * s2, s2, ge::ops::One};
   x1.ir_attr.SetIndex(0);

   ge::ascir_op::Data x2("x2");
   graph.AddNode(x2);
   x2.y.dtype = ge::DT_INT32;
   x2.attr.sched.axis = {z3.id, z4.id};
   *x2.y.axis = {z3.id, z4.id};
   *x2.y.repeats = {s3, s4};
   *x2.y.strides = {s4, ge::ops::One};
   x2.ir_attr.SetIndex(1);

   ge::ascir_op::Gather gather("gather");
   graph.AddNode(gather);
   gather.x1 = x1.y;
   gather.x2 = x2.y;
   gather.ir_attr.SetAxis(gather_axis);
   gather.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
   *gather.y.axis = {z0.id, z1.id, z3.id, z4.id};
   *gather.y.repeats = {s0, s1, s3, s4};
   *gather.y.strides = {s1 * s3 * s4, s3 * s4, s4, ge::ops::One};

   ge::ascir_op::Abs abs("abs");
   graph.AddNode(abs);
   abs.x = gather.y;
   abs.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
   *abs.y.axis = {z0.id, z1.id, z3.id, z4.id};
   *abs.y.repeats = {s0, s1, s3, s4};
   *abs.y.strides = {s1 * s3 * s4, s3 * s4, s4, ge::ops::One};

   ge::ascir_op::Store store("store");
   graph.AddNode(store);
   store.x = abs.y;
   store.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
   *store.y.axis = {z0.id, z1.id, z3.id, z4.id};
   *store.y.repeats = {s0, s1, s3, s4};
   *store.y.strides = {s1 * s3 * s4, s3 * s4, s4, ge::ops::One};

   ge::ascir_op::Output y("y");
   graph.AddNode(y);
   y.x = store.y;
   y.y.dtype = data_type;
   y.ir_attr.SetIndex(0);
 }

 ge::ComputeGraphPtr ShareGraph::LoadGatherTailAbsStore(int64_t gather_axis, ge::DataType data_type) {
   auto builder = GraphBuilder("load_gather_tail_abs_store_store_test");
   auto data0 = builder.AddNode("data0", "Data", 0, 1);
   ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
   auto data1 = builder.AddNode("data1", "Data", 0, 1);
   ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

   auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
   auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

   builder.AddDataEdge(data0, 0, ascbc, 0);
   builder.AddDataEdge(data1, 0, ascbc, 1);
   builder.AddDataEdge(ascbc, 0, netoutput, 0);
   ComputeGraphPtr compute_graph = builder.GetGraph();
   if (compute_graph == nullptr) {
     return nullptr;
   }
   auto ascbc_node = compute_graph->FindNode("ascbc");
   ge::AscGraph sub_graph("load_gather_tail_abs_store_store_test");
   LoadGatherTailAbsStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

   std::string sub_graph_str;
   ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
   ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
   return compute_graph;
 }

static void ConstructVVAscGraphAxisInfoForOneAxisGather(ge::AscGraph &graph, size_t dims_size) {
  std::vector<int64_t> axis;
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(2);
    auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(aixs_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
    if (node->GetName() == "data0") {
      auto node_output= node->outputs()[0];
      node_output->attr.dtype = DT_FLOAT;
    }
    if (node->GetName() == "data1") {
      auto node_output= node->outputs()[0];
      node_output->attr.dtype = DT_INT32;
    }
    if (node->GetType() == "Gather") {
      auto gather_output= node->outputs()[0];
      gather_output->attr.dtype = DT_FLOAT;
      auto gather_input0= node->inputs()[0];
      gather_input0->attr.dtype = DT_FLOAT;
      auto gather_input1= node->inputs()[1];
      gather_input1->attr.dtype = DT_INT32;
      const auto &op = node->GetOpDesc();
      const auto &attr = op->GetAttrsGroup<AscNodeAttr>();
      auto gather_ir_attr = dynamic_cast<ascir_op::Gather::AscGatherIrAttrDef *>(attr->ir_attr.get());
      gather_ir_attr->SetAxis(0);
    }
  }
}

static void LoadGatherOneAxisAbsStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Gather gather("gather");
  gather.x1 = x1.y;
  gather.x2 = x2.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = gather.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = gather.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfoForOneAxisGather(graph, 1);
 }

 ge::ComputeGraphPtr ShareGraph::LoadGatherOneAxisAbsStore(int64_t gather_axis, ge::DataType data_type) {
   auto builder = GraphBuilder("load_gather_one_axis_abs_store_store_test");
   auto data0 = builder.AddNode("data0", "Data", 0, 1);
   ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
   auto data1 = builder.AddNode("data1", "Data", 0, 1);
   ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

   auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
   auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

   builder.AddDataEdge(data0, 0, ascbc, 0);
   builder.AddDataEdge(data1, 0, ascbc, 1);
   builder.AddDataEdge(ascbc, 0, netoutput, 0);
   ComputeGraphPtr compute_graph = builder.GetGraph();
   if (compute_graph == nullptr) {
     return nullptr;
   }
   auto ascbc_node = compute_graph->FindNode("ascbc");
   ge::AscGraph sub_graph("load_gather_one_axis_abs_store_store_test");
   LoadGatherOneAxisAbsStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

   std::string sub_graph_str;
   ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
   ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
   return compute_graph;
 }

static void GatherReduceStore_BeforeAutofuse(ge::AscGraph &graph, int64_t gather_axis, ge::DataType data_type) {
   auto s0 = graph.CreateSizeVar("s0");
   auto s1 = graph.CreateSizeVar("s1");
   auto s2 = graph.CreateSizeVar("s2");
   auto s3 = graph.CreateSizeVar("s3");

   auto z0 = graph.CreateAxis("z0", s0);
   auto z1 = graph.CreateAxis("z1", s1);
   auto z2 = graph.CreateAxis("z2", s2);
   auto z3 = graph.CreateAxis("z3", s3);
   ge::ascir_op::Data x1("x1");
   graph.AddNode(x1);
   x1.y.dtype = data_type;
   x1.attr.sched.axis = {z0.id, z1.id};
   *x1.y.axis = {z0.id, z1.id};
   *x1.y.repeats = {s0, s1};
   *x1.y.strides = {s1, ge::ops::One};
   x1.ir_attr.SetIndex(0);

   ge::ascir_op::Data x2("x2");
   graph.AddNode(x2);
   x2.y.dtype = ge::DT_INT32;
   x2.attr.sched.axis = {z2.id, z3.id};
   *x2.y.axis = {z2.id, z3.id};
   *x2.y.repeats = {s2, s3};
   *x2.y.strides = {s3, ge::ops::One};
   x2.ir_attr.SetIndex(1);

   ge::ascir_op::Gather gather("gather");
   graph.AddNode(gather);
   gather.x1 = x1.y;
   gather.x2 = x2.y;
   gather.ir_attr.SetAxis(0);
   gather.attr.sched.axis = {z2.id, z3.id, z1.id};
   *gather.y.axis = {z2.id, z3.id, z1.id};
   *gather.y.repeats = {s2, s3, s1};
   *gather.y.strides = {s3 * s1, s1,ge::ops::One};

    ge::ascir_op::Max max("max");
    graph.AddNode(max);
    max.x = gather.y;
    max.attr.sched.axis = {z2.id, z3.id, z1.id};
    *max.y.axis = {z2.id, z3.id, z1.id};
    *max.y.repeats = {ops::One, s3, ops::One};
    *max.y.strides = {ops::Zero, ge::ops::One, ge::ops::Zero};

   ge::ascir_op::Store store("store");
   graph.AddNode(store);
   store.x = max.y;
   store.attr.sched.axis = {z2.id, z3.id, z1.id};
   *store.y.axis = {z2.id, z3.id, z1.id};
   *store.y.repeats = {ops::One, s3, ops::One};
   *store.y.strides = {ops::Zero, ge::ops::One, ge::ops::Zero};

   ge::ascir_op::Output y("y");
   graph.AddNode(y);
   y.x = store.y;
   y.y.dtype = data_type;
   y.ir_attr.SetIndex(0);
 }

 ge::ComputeGraphPtr ShareGraph::GatherReduceStore(int64_t gather_axis, ge::DataType data_type) {
   auto builder = GraphBuilder("gather_reduce_store_store_test");
   auto data0 = builder.AddNode("data0", "Data", 0, 1);
   ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
   auto data1 = builder.AddNode("data1", "Data", 0, 1);
   ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

   auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
   auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

   builder.AddDataEdge(data0, 0, ascbc, 0);
   builder.AddDataEdge(data1, 0, ascbc, 1);
   builder.AddDataEdge(ascbc, 0, netoutput, 0);
   ComputeGraphPtr compute_graph = builder.GetGraph();
   if (compute_graph == nullptr) {
     return nullptr;
   }
   auto ascbc_node = compute_graph->FindNode("ascbc");
   ge::AscGraph sub_graph("gather_reduce_store_store_test");
   GatherReduceStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

   std::string sub_graph_str;
   ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
   ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
   return compute_graph;
 }

 /**
 *         data0  data1
 *           |      |
 *         load0  load1
 *           \     /
 *              GE
 *              |
 *             Cast
 *              |
 *             Sum
 *              |
 *            store
 *              |
 *            ouput0
 */
static void CreateLoadCompareCastSumStoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(3);
  const Expression s1 = graph.CreateSizeVar(77);
  const Expression s2 = graph.CreateSizeVar(21);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x1.y.dtype = ge::DT_FLOAT;
  x2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.attr.api.compute_type = ComputeType::kComputeLoad;
  x1Local.attr.api.type = ge::ApiType::kAPITypeCompute;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1*s2, s2, One};
  x1Local.attr.api.unit = ComputeUnit::kUnitMTE2;
  x1Local.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.api.compute_type = ComputeType::kComputeLoad;
  x2Local.attr.api.type = ge::ApiType::kAPITypeCompute;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1*s2, s2, One};
  x2Local.attr.api.unit = ComputeUnit::kUnitMTE2;
  x2Local.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Ge ge0("ge0");
  ge0.x1 = x1Local.y;
  ge0.x2 = x2Local.y;
  ge0.attr.api.compute_type = ComputeType::kComputeElewise;
  ge0.attr.api.type = ge::ApiType::kAPITypeCompute;
  ge0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *ge0.y.axis = {z0.id, z1.id, z2.id};
  *ge0.y.repeats = {s0, s1, s2};
  *ge0.y.strides = {s1*s2, s2, One};
  ge0.attr.api.unit = ComputeUnit::kUnitVector;
  ge0.y.dtype = ge::DT_UINT8;

  ge::ascir_op::Cast cast0("cast0");
  cast0.x = ge0.y;
  cast0.attr.api.compute_type = ComputeType::kComputeElewise;
  cast0.attr.api.type = ge::ApiType::kAPITypeCompute;
  cast0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast0.y.axis = {z0.id, z1.id, z2.id};
  *cast0.y.repeats = {s0, s1, s2};
  *cast0.y.strides = {s1*s2, s2, One};
  cast0.attr.api.unit = ComputeUnit::kUnitVector;
  cast0.y.dtype = ge::DT_INT8;

  ge::ascir_op::Cast cast1("cast1");
  cast1.x = cast0.y;
  cast1.attr.api.compute_type = ComputeType::kComputeElewise;
  cast1.attr.api.type = ge::ApiType::kAPITypeCompute;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  *cast1.y.repeats = {s0, s1, s2};
  *cast1.y.strides = {s1*s2, s2, One};
  cast1.attr.api.unit = ComputeUnit::kUnitVector;
  cast1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.attr.api.compute_type = ComputeType::kComputeElewise;
  cast2.attr.api.type = ge::ApiType::kAPITypeCompute;
  cast2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast2.y.axis = {z0.id, z1.id, z2.id};
  *cast2.y.repeats = {s0, s1, s2};
  *cast2.y.strides = {s1*s2, s2, One};
  cast2.attr.api.unit = ComputeUnit::kUnitVector;
  cast2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Sum sum0("sum0");
  sum0.x = cast2.y;
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *sum0.y.axis = {z0.id, z1.id, z2.id};
  *sum0.y.repeats = {s0, One, s2};
  *sum0.y.strides = {s2, Zero, One};
  sum0.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store sum0_store("store");
  sum0_store.x = sum0.y;
  sum0_store.attr.api.compute_type = ComputeType::kComputeStore;
  sum0_store.attr.api.type = ge::ApiType::kAPITypeCompute;
  sum0_store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *sum0_store.y.axis = {z0.id, z1.id, z2.id};
  *sum0_store.y.repeats = {s0, One, s2};
  *sum0_store.y.strides = {s2, Zero, One};
  sum0_store.attr.api.unit = ComputeUnit::kUnitMTE3;
  sum0_store.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output y("output");
  y.x = sum0_store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ge::ApiType::kAPITypeBuffer;
  y.y.dtype = ge::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::LoadCompareCastSumStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_compare_cast_sum_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_compare_cast_sum_store");
  CreateLoadCompareCastSumStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        log2
 *         |
 *       load0
 *         |
 *       data0
 */
static void CreateLoadLog2StoreAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Log2 log2("log2");
  log2.x = x1Local.y;

  ge::ascir_op::Store log2_store("store");
  log2_store.x = log2.y;

  ge::ascir_op::Output y("output");
  y.x = log2_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadLog2StoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_log2_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_log2_store");
  CreateLoadLog2StoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        mod
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateModAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_FLOAT;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Mod mod("mod");
  mod.x1 = x1Local.y;
  mod.x2 = x2Local.y;
  mod.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Store x_out("store");
  x_out.x = mod.y;
  x_out.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::ModFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("mod_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("mod");
  CreateModAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       LShift
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadLShiftStoreAscGraph(ge::AscGraph &graph, size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  ge::ascir_op::LShift lshift("lshift");
  lshift.x1 = x1Local.y;
  lshift.x2 = x2Local.y;
  lshift.y.dtype = out_dtype;

  ge::ascir_op::Store x_out("store");
  x_out.x = lshift.y;
  x_out.y.dtype = out_dtype;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::LoadLShiftStoreFusedGraph(size_t dims_size, ge::DataType in_dtype, ge::DataType out_dtype) {
  auto builder = GraphBuilder("load_lshift_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_lshift_store");
  CreateLoadLShiftStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                  sub
 *               mul    \
 *              /   \    \
 *           add sigmoid rsqrt
 *          /   \     \    \
 *       Matmul brc    brc  brc
 *      /     \   \     \    \
 *   data0 data1 data2 data3 data4
 */
static void CreateMatmulElewiseBrcGraph(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ge::ops::One};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ge::ops::One};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::MatMul matmul("matmul");
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

  ge::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data2.y.repeats = {ge::ops::One, ge::ops::One};
  *data2.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data2.ir_attr.SetIndex(2);

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {ge::ops::One, ge::ops::One};
  *load2.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = load2.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {ge::ops::One, s1};
  *broadcast0.y.strides = {ge::ops::Zero, ge::ops::One};

  ge::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = matmul.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data3("data3", graph);
  data3.y.dtype = ge::DT_FLOAT;
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data3.y.repeats = {s0, ge::ops::One};
  *data3.y.strides = {ge::ops::One, ge::ops::Zero};
  data3.ir_attr.SetIndex(3);

  ge::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, ge::ops::One};
  *load3.y.strides = {ge::ops::One, ge::ops::Zero};

  ge::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load3.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {s0, s1};
  *broadcast2.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Sigmoid sigmoid0("sigmoid0");
  sigmoid0.x = broadcast2.y;
  sigmoid0.attr.sched.axis = {z0.id, z1.id};
  *sigmoid0.y.axis = {z0.id, z1.id};
  sigmoid0.y.dtype = ge::DT_FLOAT;
  *sigmoid0.y.repeats = {s0, s1};
  *sigmoid0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = sigmoid0.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data4("data4", graph);
  data4.y.dtype = ge::DT_FLOAT;
  data4.attr.sched.axis = {z0.id, z1.id};
  *data4.y.axis = {z0.id, z1.id};
  data4.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data4.y.repeats = {ge::ops::One, s1};
  *data4.y.strides = {ge::ops::Zero, ge::ops::One};
  data4.ir_attr.SetIndex(4);

  ge::ascir_op::Load load4("load4");
  load4.x = data4.y;
  load4.attr.sched.axis = {z0.id, z1.id};
  load4.y.dtype = ge::DT_FLOAT;
  *load4.y.axis = {z0.id, z1.id};
  *load4.y.repeats = {ge::ops::One, s1};
  *load4.y.strides = {ge::ops::Zero, ge::ops::One};

  ge::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = load4.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = ge::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Rsqrt rsqrt0("rsqrt0");
  rsqrt0.x = broadcast3.y;
  rsqrt0.attr.sched.axis = {z0.id, z1.id};
  *rsqrt0.y.axis = {z0.id, z1.id};
  rsqrt0.y.dtype = ge::DT_FLOAT;
  *rsqrt0.y.repeats = {s0, s1};
  *rsqrt0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Sub sub("sub");
  sub.attr.sched.axis = {z0.id, z1.id};
  sub.x1 = mul.y;
  sub.x2 = rsqrt0.y;
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.axis = {z0.id, z1.id};
  *sub.y.repeats = {s0, s1};
  *sub.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = sub.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1 ,ge::ops::One};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

ge::ComputeGraphPtr ShareGraph::LoadMatmulElewiseBrcFusedGraph() {
  auto builder = GraphBuilder("load_matmul_elewise_brc_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  ge::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  ge::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_matmul_elewise_brc_store");
  CreateMatmulElewiseBrcGraph(sub_graph);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *               where
 *          /        \     \
 *       Matmul      eq     \
 *      /    \     /    \    \
 *   data0 data1 data2 data3 scalar0
 */
static void CreateMatmulCompareScalarGraph(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(32);
  auto s1 = graph.CreateSizeVar(32);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ge::ops::One};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ge::ops::One};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::MatMul matmul("matmul");
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

  ge::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, ge::ops::One};
  data2.ir_attr.SetIndex(2);

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data3("data3", graph);
  data3.y.dtype = ge::DT_FLOAT;
  data3.attr.sched.axis = {z0.id, z1.id};
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data3.y.repeats = {ge::ops::One, ge::ops::One};
  *data3.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data3.ir_attr.SetIndex(3);

  ge::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {ge::ops::One, ge::ops::One};
  *load3.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Eq eq0("eq0");
  eq0.x1 = load2.y;
  eq0.x2 = load3.y;
  eq0.attr.sched.axis = {z0.id, z1.id};
  eq0.y.dtype = ge::DT_UINT8;
  *eq0.y.axis = {z0.id, z1.id};
  *eq0.y.repeats = {s0, s1};
  *eq0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("1");

  ge::ascir_op::Where where0("where");
  where0.x1 = eq0.y;
  where0.x2 = matmul.y;
  where0.x3 = scalar0.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = ge::DT_FLOAT;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.x = where0.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1 ,ge::ops::One};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

ge::ComputeGraphPtr ShareGraph::LoadMatmulCompareScalarFusedGraph() {
  auto builder = GraphBuilder("load_matmul_compare_scalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  ge::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  ge::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("load_matmul_elewise_brc_store");
  CreateMatmulCompareScalarGraph(sub_graph);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        div
 *       /   \
 *   load0  load1
 *      |     |
 *   data0  data1
 */
static void CreateDivAbsAscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  ge::ascir_op::Div div("div");
  div.x1 = x1Local.y;
  div.x2 = x2Local.y;

  ge::ascir_op::Abs abs("abs");
  abs.x = div.y;

  ge::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::DivAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("div_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("div_abs");
  CreateDivAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTrueDivBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.y.dtype = ge::DataType::DT_BF16;
  x1.ir_attr.SetIndex(0);

  ge::ascir_op::Data x2("data1", graph);
  x2.y.dtype = ge::DataType::DT_BF16;
  x2.ir_attr.SetIndex(1);

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::TrueDiv trueDiv("trueDiv");
  trueDiv.x1 = x1Local.y;
  trueDiv.x2 = x2Local.y;
  trueDiv.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = trueDiv.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::TrueDivBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("truediv_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("truediv_bf16");
  CreateTrueDivBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                where
 *          /           \  \
 *        ge            add  \
 *      /     \      /      \  \
 *   data0 scalar0 scalar0  data1
 */
static void CreateCompareScalarWhereGraph(ge::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", Symbol("s0"));
  auto z1 = graph.CreateAxis("z1", Symbol("s1"));

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ge::ops::One};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ge::ops::One};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("0.5");

  ge::ascir_op::Ge ge0("ge0");
  ge0.x1 = load0.y;
  ge0.x2 = scalar0.y;
  ge0.attr.sched.axis = {z0.id, z1.id};
  ge0.y.dtype = ge::DT_UINT8;
  *ge0.y.axis = {z0.id, z1.id};
  *ge0.y.repeats = {s0, s1};
  *ge0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load1.y;
  add0.x2 = scalar0.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Where where0("where");
  where0.x1 = ge0.y;
  where0.x2 = add0.y;
  where0.x3 = load1.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = ge::DT_FLOAT16;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.x = where0.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1 ,ge::ops::One};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

ge::ComputeGraphPtr ShareGraph::LoadCompareScalarWhereFusedGraph() {
  auto builder = GraphBuilder("load_compare_scalar_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("compare_scalar_where");
  CreateCompareScalarWhereGraph(sub_graph);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *            where
 *          /    \    \
 *        eq      \    \
 *      /    \     \    \
 *   data0 data1 data2 data3
 */
static void CreateCompareWhereGraph(ge::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", Symbol("s0"));
  auto z1 = graph.CreateAxis("z1", Symbol("s1"));

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, ge::ops::One};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, ge::ops::One};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, ge::ops::One};
  data2.ir_attr.SetIndex(2);

  ge::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = ge::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = ge::DT_FLOAT16;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data3.y.repeats = {s0, s1};
  *data3.y.strides = {s1, ge::ops::One};
  data3.ir_attr.SetIndex(3);

  ge::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = ge::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Eq eq0("eq0");
  eq0.x1 = load0.y;
  eq0.x2 = load1.y;
  eq0.attr.sched.axis = {z0.id, z1.id};
  eq0.y.dtype = ge::DT_UINT8;
  *eq0.y.axis = {z0.id, z1.id};
  *eq0.y.repeats = {s0, s1};
  *eq0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Where where0("where");
  where0.x1 = eq0.y;
  where0.x2 = load2.y;
  where0.x3 = load3.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = ge::DT_FLOAT16;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.x = where0.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1 ,ge::ops::One};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

ge::ComputeGraphPtr ShareGraph::LoadCompareWhereFusedGraph() {
  auto builder = GraphBuilder("load_compare_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  ge::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  ge::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 4, 1);
  ge::AttrUtils::SetInt(ascbc->GetOpDescBarePtr(), "_parent_node_index", 4);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("compare_where");
  CreateCompareWhereGraph(sub_graph);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       acos
 *         |
 *      load0
 *         |
 *      data0
 */
static void CreateAcosFloatAscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DT_FLOAT;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DT_FLOAT;

  // Acos 操作
  ge::ascir_op::Acos acos("acos");
  acos.x = xLocal.y;
  acos.y.dtype = ge::DT_FLOAT;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = acos.y;
  x_out.y.dtype = ge::DT_FLOAT;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_FLOAT;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *         |
 *       data0
 */
ge::ComputeGraphPtr ShareGraph::AcosFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("acos_float_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("acos_float");
  CreateAcosFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);

  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       acos
 *         |
 *      load0
 *         |
 *      data0
 */
static void CreateAcosBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DT_BF16;

  // Acos 操作
  ge::ascir_op::Acos acos("acos");
  acos.x = xLocal.y;
  acos.y.dtype = ge::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = acos.y;
  x_out.y.dtype = ge::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *         |
 *       data0
 */
ge::ComputeGraphPtr ShareGraph::AcosBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("acos_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("acos_bf16");
  CreateAcosBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);

  return compute_graph;
}

/**
 *                                         add
 *                             /                         \
 *                          add                            \
 *                 /                   \                     \
 *              add                    add                    add
 *          /         \            /         \             /        \
 *        add        mul       maximum     minimum   logical_and   logical_and
 *      /    \     /     \     /     \     /     \     /     \      /     \
 *   data0 data1 data0 data1 data0 data1 data0 data1 data0  data1 data0  data1
 */
static void CreateBinaryApiScalarGraph(ge::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", Symbol("s0"));
  auto z1 = graph.CreateAxis("z1", Symbol("s1"));

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.repeats = {ge::ops::One, ge::ops::One};
  *data0.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {ge::ops::One, ge::ops::One};
  *load0.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = ge::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {ge::ops::One, ge::ops::One};
  *data1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {ge::ops::One, ge::ops::One};
  *load1.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = ge::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Mul mul0("mul0");
  mul0.x1 = load0.y;
  mul0.x2 = load1.y;
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.y.dtype = ge::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Maximum maximum0("maximum0");
  maximum0.x1 = load0.y;
  maximum0.x2 = load1.y;
  maximum0.attr.sched.axis = {z0.id, z1.id};
  maximum0.y.dtype = ge::DT_FLOAT16;
  *maximum0.y.axis = {z0.id, z1.id};
  *maximum0.y.repeats = {s0, s1};
  *maximum0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Minimum minimum0("minimum0");
  minimum0.x1 = load0.y;
  minimum0.x2 = load1.y;
  minimum0.attr.sched.axis = {z0.id, z1.id};
  minimum0.y.dtype = ge::DT_FLOAT16;
  *minimum0.y.axis = {z0.id, z1.id};
  *minimum0.y.repeats = {s0, s1};
  *minimum0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::LogicalAnd logical_and0("logical_and0");
  logical_and0.x1 = load0.y;
  logical_and0.x2 = load1.y;
  logical_and0.attr.sched.axis = {z0.id, z1.id};
  logical_and0.y.dtype = ge::DT_UINT8;
  *logical_and0.y.axis = {z0.id, z1.id};
  *logical_and0.y.repeats = {s0, s1};
  *logical_and0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast0("cast0");
  cast0.x = logical_and0.y;
  cast0.attr.sched.axis = {z0.id, z1.id};
  cast0.y.dtype = ge::DT_FLOAT16;
  *cast0.y.axis = {z0.id, z1.id};
  *cast0.y.repeats = {s0, s1};
  *cast0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::LogicalOr logical_or0("logical_or0");
  logical_or0.x1 = load0.y;
  logical_or0.x2 = load1.y;
  logical_or0.attr.sched.axis = {z0.id, z1.id};
  logical_or0.y.dtype = ge::DT_UINT8;
  *logical_or0.y.axis = {z0.id, z1.id};
  *logical_or0.y.repeats = {s0, s1};
  *logical_or0.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Cast cast1("cast1");
  cast1.x = logical_or0.y;
  cast1.attr.sched.axis = {z0.id, z1.id};
  cast1.y.dtype = ge::DT_FLOAT16;
  *cast1.y.axis = {z0.id, z1.id};
  *cast1.y.repeats = {s0, s1};
  *cast1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = mul0.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.y.dtype = ge::DT_FLOAT16;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add2("add2");
  add2.x1 = maximum0.y;
  add2.x2 = minimum0.y;
  add2.attr.sched.axis = {z0.id, z1.id};
  add2.y.dtype = ge::DT_FLOAT16;
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add3("add3");
  add3.x1 = cast0.y;
  add3.x2 = cast1.y;
  add3.attr.sched.axis = {z0.id, z1.id};
  add3.y.dtype = ge::DT_FLOAT16;
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add4("add4");
  add4.x1 = add1.y;
  add4.x2 = add2.y;
  add4.attr.sched.axis = {z0.id, z1.id};
  add4.y.dtype = ge::DT_FLOAT16;
  *add4.y.axis = {z0.id, z1.id};
  *add4.y.repeats = {s0, s1};
  *add4.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add5("add5");
  add5.x1 = add3.y;
  add5.x2 = add4.y;
  add5.attr.sched.axis = {z0.id, z1.id};
  add5.y.dtype = ge::DT_FLOAT16;
  *add5.y.axis = {z0.id, z1.id};
  *add5.y.repeats = {s0, s1};
  *add5.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.x = add5.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1 ,ge::ops::One};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

ge::ComputeGraphPtr ShareGraph::BinaryApiScalarFusedGraph() {
  auto builder = GraphBuilder("binary_api_scalar_test");

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 0, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("binary_api_scalar");
  CreateBinaryApiScalarGraph(sub_graph);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAcoshBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data0", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = ge::DataType::DT_BF16;
  x1Local.x = x.y;

  // Acosh 操作
  ge::ascir_op::Acosh acosh("acosh");
  acosh.x = x1Local.y;
  acosh.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store acosh_store("store");
  acosh_store.x = acosh.y;
  acosh_store.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = acosh_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AcoshBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("acosh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("acosh_bf16_test");
  CreateAcoshBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAsinBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  ge::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = ge::DataType::DT_FLOAT;

  // Asin 操作
  ge::ascir_op::Asin asin("asin");
  asin.x = cast.y;
  asin.y.dtype = ge::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  ge::ascir_op::Cast castBack("cast_back");
  castBack.x = asin.y;
  castBack.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AsinBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("asin_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("asin_bf16_test");
  CreateAsinBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAsinhBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  ge::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = ge::DataType::DT_FLOAT;

  // Asinh 操作
  ge::ascir_op::Asinh asinh("asinh");
  asinh.x = cast.y;
  asinh.y.dtype = ge::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  ge::ascir_op::Cast castBack("cast_back");
  castBack.x = asinh.y;
  castBack.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AsinhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("asinh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("asinh_bf16_test");
  CreateAsinhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAtanBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Atan 操作
  ge::ascir_op::Atan atan("atan");
  atan.x = xLocal.y;
  atan.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = atan.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AtanBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("atan_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("atan_bf16_test");
  CreateAtanBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAtanhBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  ge::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = ge::DataType::DT_FLOAT;

  // Atanh 操作
  ge::ascir_op::Atanh atanh("atanh");
  atanh.x = cast.y;
  atanh.y.dtype = ge::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  ge::ascir_op::Cast castBack("cast_back");
  castBack.x = atanh.y;
  castBack.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::AtanhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("atanh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("atanh_bf16_test");
  CreateAtanhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCoshBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  ge::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = ge::DataType::DT_FLOAT;

  // Cosh 操作
  ge::ascir_op::Cosh cosh("cosh");
  cosh.x = cast.y;
  cosh.y.dtype = ge::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  ge::ascir_op::Cast castBack("cast_back");
  castBack.x = cosh.y;
  castBack.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::CoshBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("cosh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("cosh_bf16_test");
  CreateCoshBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateDigammaBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  ge::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = ge::DataType::DT_FLOAT;

  // Digamma 操作
  ge::ascir_op::Digamma digamma("digamma");
  digamma.x = cast.y;
  digamma.y.dtype = ge::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  ge::ascir_op::Cast castBack("cast_back");
  castBack.x = digamma.y;
  castBack.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::DigammaBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("digamma_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("digamma_bf16_test");
  CreateDigammaBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateErfcBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  // Data 节点
  ge::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = ge::DataType::DT_BF16;

  // Load 节点
  ge::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  ge::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = ge::DataType::DT_FLOAT;

  // Erfc 操作
  ge::ascir_op::Erfc erfc("erfc");
  erfc.x = cast.y;
  erfc.y.dtype = ge::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  ge::ascir_op::Cast castBack("cast_back");
  castBack.x = erfc.y;
  castBack.y.dtype = ge::DataType::DT_BF16;

  // Store 节点
  ge::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  // Output 节点
  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = ge::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::ErfcBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("erfc_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("erfc_bf16_test");
  CreateErfcBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
/**
 *      output
 *         |
 *       store
 *         |
 *        pow(bf16)
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreatePowBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DT_BF16;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DT_BF16;

  ge::ascir_op::Pow pow("pow");
  pow.x1 = x1Local.y;
  pow.x2 = x2Local.y;
  pow.y.dtype = ge::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = pow.y;
  x_out.y.dtype = ge::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::PowBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("pow_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("pow_bf16");
  CreatePowBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data
 *         |
 *        load
 *         |
 *       reciprocal
 *         |
 *       store
 *         |
 *      output
 */
static void CreateReciprocalBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Reciprocal reciprocal("reciprocal");
  reciprocal.x = xLocal.y;
  reciprocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = reciprocal.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *         data
 *          |
 *        load
 *          |
 *        Round
 *          |
 *        store
 *          |
 *       output
 */
static void CreateRoundBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Round round("round");
  round.x = xLocal.y;
  round.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = round.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
ge::ComputeGraphPtr ShareGraph::ReciprocalBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("reciprocal_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("reciprocal_bf16_test");
  CreateReciprocalBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *         |
 *       data
 */
ge::ComputeGraphPtr ShareGraph::RoundBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("round_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("round_bf16_test");
  CreateRoundBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data
 *          |
 *        load
 *          |
 *        Relu
 *          |
 *        store
 *          |
 *       output
 */
static void CreateReluUint8AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_UINT8;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Relu relu("relu");
  relu.x = xLocal.y;
  relu.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = relu.y;
  x_out.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *         |
 *       data
 */
ge::ComputeGraphPtr ShareGraph::ReluUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("relu_uint8_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("relu_uint8_test");
  CreateReluUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRshiftUint8AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = ge::DataType::DT_UINT8;
  ge::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = ge::DataType::DT_INT8;

  ge::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = ge::DataType::DT_INT8;

  ge::ascir_op::RShift rshift("rshift");
  rshift.x1 = x1Local.y;
  rshift.x2 = x2Local.y;
  rshift.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = rshift.y;
  x_out.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::RshiftUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("rshift_uint8_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  ge::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  ge::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("rshift_uint8");
  CreateRshiftUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSignUint8AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_UINT8;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Sign sign("sign");
  sign.x = xLocal.y;
  sign.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Store x_out("store");
  x_out.x = sign.y;
  x_out.y.dtype = ge::DataType::DT_UINT8;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::SignUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sign_uint8_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sign_uint8_test");
  CreateSignUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSignBf16AscGraph(ge::AscGraph &graph, size_t dims_size) {
  ge::ascir_op::Data x("data", graph);
  x.y.dtype = ge::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Sign sign("sign");
  sign.x = xLocal.y;
  sign.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Store x_out("store");
  x_out.x = sign.y;
  x_out.y.dtype = ge::DataType::DT_BF16;

  ge::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

ge::ComputeGraphPtr ShareGraph::SignBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sign_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  ge::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", ge::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  ge::AscGraph sub_graph("sign_bf16_test");
  CreateSignBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  ge::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  ge::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
}  // namespace ascir
