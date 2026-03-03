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
#include <iostream>
#include "ascir_ops.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph/ascendc_ir/ascir_register.h"
#include "graph/utils/graph_utils.h"
#include "dlog_pub.h"
#include "expression/const_values.h"
#include "code_extractor.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "depends/runtime/src/runtime_stub.h"

using namespace ge::ascir_op;
namespace {
constexpr int64_t ID_NONE = -1;
}
class UtestAscendCIR : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
using namespace ge;
using ge::Expression;
using ge::Symbol;
namespace {
struct AscNodeInfo {
  std::string name;
  std::string type;
  size_t input_num;
  size_t output_num;
  std::vector<int64_t> axis_ids;
};
template<class T>
class OpDtypeInfer {
 public:
  OpDtypeInfer &Input(ge::DataType input_type) {
    input_dtypes_.push_back(input_type);
    return *this;
  }
  OpDtypeInfer &Expect(const DataType &type) {
    expected_dtypes_.emplace_back(type);
    return *this;
  }
  void AssertSucceed() {
    std::string npu_arch = "socv1";
    if (expected_dtypes_.empty()) {
      std::vector<DataType> dtypes;
      ASSERT_EQ(T::InferDataType(input_dtypes_, dtypes, npu_arch), GRAPH_SUCCESS);
      std::vector<DataType> dtypes2;
      ASSERT_EQ(ascir::CommonInferDtype(T::Type, input_dtypes_, dtypes2, npu_arch), GRAPH_SUCCESS);
    } else {
      ASSERT_EQ(T::InferDataType(input_dtypes_, expected_dtypes_, npu_arch), GRAPH_SUCCESS);
      ASSERT_EQ(ascir::CommonInferDtype(T::Type, input_dtypes_, expected_dtypes_, npu_arch), GRAPH_SUCCESS);
    }
  }

  void AssertFailed() {
    std::string npu_arch = "socv1";
    if (expected_dtypes_.empty()) {
      std::vector<DataType> dtypes;
      ASSERT_NE(T::InferDataType(input_dtypes_, dtypes, npu_arch), GRAPH_SUCCESS);
      std::vector<DataType> dtypes2;
      ASSERT_NE(ascir::CommonInferDtype(T::Type, input_dtypes_, dtypes, npu_arch), GRAPH_SUCCESS);
    } else {
      ASSERT_NE(T::InferDataType(input_dtypes_, expected_dtypes_, npu_arch), GRAPH_SUCCESS);
      ASSERT_NE(ascir::CommonInferDtype(T::Type, input_dtypes_, expected_dtypes_, npu_arch), GRAPH_SUCCESS);
    }
  }
 private:
  std::vector<ge::DataType> input_dtypes_;
  std::vector<ge::DataType> expected_dtypes_;
};
}
TEST_F(UtestAscendCIR, TilingKey_OK) {
  AscGraph graph("test_graph");
  graph.SetTilingKey(10);
  EXPECT_EQ(graph.GetTilingKey(), 10);
}

TEST_F(UtestAscendCIR, CreateSizeVar_OK) {
  AscGraph graph("test_graph");
  const auto &s0 = graph.CreateSizeVar("s0");
  const auto &s1 = graph.CreateSizeVar("s1");
  const auto &s2 = graph.CreateSizeVar("s2");
  const auto &const_10 = graph.CreateSizeVar(10);
  Symbol symbol1(1, "MyOne");
  graph.CreateSizeVar(symbol1);
  Symbol symbol2("s3");
  graph.CreateSizeVar(symbol2);
  auto all_size_var = graph.GetAllSizeVar();
  EXPECT_EQ(all_size_var.size(), 6u);
  EXPECT_EQ(all_size_var[4]->expr.IsConstExpr(), true);
  int64_t i_get(-1);
  EXPECT_EQ(all_size_var[4]->expr.GetConstValue<>(i_get), true);
  EXPECT_EQ(i_get, 1);
  EXPECT_EQ(all_size_var[5]->expr.IsConstExpr(), false);
  EXPECT_EQ(all_size_var[5]->expr.Str().get(), std::string("s3"));
}

TEST_F(UtestAscendCIR, CreateAxis) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  s0_axis.align = Symbol(10);
  auto s0_axis_find = graph.FindAxis(s0_axis.id);
  EXPECT_NE(s0_axis_find, nullptr);
  EXPECT_EQ(s0_axis_find->name, "S0");
  EXPECT_EQ(s0_axis_find->align, Symbol(10));

  auto axis_invalid = graph.FindAxis(-1);
  EXPECT_EQ(axis_invalid, nullptr);
}

TEST_F(UtestAscendCIR, BlockSplit) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  auto split_axis = graph.BlockSplit(s0_axis.id);
  EXPECT_NE(split_axis.first, nullptr);
  EXPECT_NE(split_axis.second, nullptr);
  auto &outer_axis = *split_axis.first;
  auto &inner_axis = *split_axis.second;
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeBlockInner);
  EXPECT_EQ(outer_axis.type, Axis::kAxisTypeBlockOuter);
}

TEST_F(UtestAscendCIR, TileSplit) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  s0_axis.align = Symbol(10);
  auto split_axis = graph.TileSplit(s0_axis.id);
  EXPECT_NE(split_axis.first, nullptr);
  EXPECT_NE(split_axis.second, nullptr);
  auto &outer_axis = *split_axis.first;
  auto &inner_axis = *split_axis.second;
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeTileInner);
  EXPECT_EQ(outer_axis.type, Axis::kAxisTypeTileOuter);
}

TEST_F(UtestAscendCIR, TileSplitSizeOneAxis) {
  AscGraph graph("test_graph");
  Axis &s0_axis = graph.CreateAxis("S0", ge::sym::kSymbolOne);
  auto split_axis = graph.TileSplit(s0_axis.id);
  EXPECT_NE(split_axis.first, nullptr);
  EXPECT_NE(split_axis.second, nullptr);
  auto &outer_axis = *split_axis.first;
  auto &inner_axis = *split_axis.second;
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeTileInner);
  EXPECT_EQ(inner_axis.size, ge::sym::kSymbolOne);
  EXPECT_EQ(outer_axis.type, Axis::kAxisTypeTileOuter);
  EXPECT_EQ(outer_axis.size, ge::sym::kSymbolOne);
}

TEST_F(UtestAscendCIR, MergeAxis) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  const Expression s1 = graph.CreateSizeVar("s1");
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
  EXPECT_NE(merge_axis, nullptr);
  EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);
}

TEST_F(UtestAscendCIR, BindBlock) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  const Expression s1 = graph.CreateSizeVar("s1");
  Axis &s1_axis = graph.CreateAxis("S1", s1);

  s0_axis.align = Symbol(10);
  auto split_axis = graph.TileSplit(s0_axis.id);
  EXPECT_NE(split_axis.first, nullptr);
  EXPECT_NE(split_axis.second, nullptr);
  auto &outer_axis = *split_axis.first;
  auto &inner_axis = *split_axis.second;
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeTileInner);
  EXPECT_EQ(outer_axis.type, Axis::kAxisTypeTileOuter);

  auto merge_axis = graph.MergeAxis({outer_axis.id, s1_axis.id});
  EXPECT_TRUE(graph.BindBlock(merge_axis->id, inner_axis.id));
  EXPECT_EQ(merge_axis->type, Axis::kAxisTypeBlockOuter);
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeBlockInner);
}

TEST_F(UtestAscendCIR, GetAllAxisTransInfo) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto [aBO, aBI] = graph.BlockSplit(a.id);
  EXPECT_EQ(graph.GetAllAxisTransInfo().size(), 1U);
  EXPECT_EQ(graph.GetAllAxisTransInfo().front().trans_type, TransType::kSplit);
  EXPECT_EQ(graph.GetAllAxisTransInfo().front().src_axis.size(), 1U);
  EXPECT_EQ(graph.GetAllAxisTransInfo().front().src_axis.front()->id, a.id);
  EXPECT_EQ(graph.GetAllAxisTransInfo().front().dst_axis.size(), 2U);
  EXPECT_EQ(graph.GetAllAxisTransInfo().front().dst_axis, std::vector<AxisPtr>({aBO, aBI}));
  auto aBIb = graph.MergeAxis({aBI->id, b.id});
  EXPECT_EQ(graph.GetAllAxisTransInfo().size(), 2U);
  EXPECT_EQ(graph.GetAllAxisTransInfo()[1U].trans_type, TransType::kMerge);
  EXPECT_EQ(graph.GetAllAxis().size(), 6U);
}

TEST_F(UtestAscendCIR, ApplySplit) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id};
  *data.y.repeats = {s0};
  *data.y.strides = {sym::kSymbolOne};

  auto split_axis = graph.TileSplit(s0_axis.id);
  EXPECT_NE(split_axis.first, nullptr);
  EXPECT_NE(split_axis.second, nullptr);
  auto &outer_axis = *split_axis.first;
  auto &inner_axis = *split_axis.second;
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeTileInner);
  EXPECT_EQ(outer_axis.type, Axis::kAxisTypeTileOuter);

  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  graph.ApplySplit(data_node_find, outer_axis.id, inner_axis.id);
}

TEST_F(UtestAscendCIR, ApplySplit_BroadCast) {
  AscGraph graph("test_graph");
  auto A = graph.CreateSizeVar("A");
  auto R = graph.CreateSizeVar("R");
  auto BL = graph.CreateSizeVar("BL");
  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {a.id, r.id, bl.id};
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = {a.id, r.id, bl.id};
  *data.y.repeats = {A, R, sym::kSymbolOne};
  *data.y.strides = {R, sym::kSymbolOne, sym::kSymbolZero};

  auto split_axis = graph.TileSplit(bl.id);
  EXPECT_NE(split_axis.first, nullptr);
  EXPECT_NE(split_axis.second, nullptr);
  auto &outer_axis = *split_axis.first;
  auto &inner_axis = *split_axis.second;
  EXPECT_EQ(inner_axis.type, Axis::kAxisTypeTileInner);
  EXPECT_EQ(outer_axis.type, Axis::kAxisTypeTileOuter);

  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  graph.ApplySplit(data_node_find, outer_axis.id, inner_axis.id);
}

TEST_F(UtestAscendCIR, ApplyMerge) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  const Expression s1 = graph.CreateSizeVar("s1");
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
  EXPECT_NE(merge_axis, nullptr);
  EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

  auto data_node_find = graph.FindNode("data");
  data_node_find->attr.sched.exec_order = 1;
  data_node_find->attr.sched.axis = {s0_axis.id, s1_axis.id};
  EXPECT_NE(data_node_find, nullptr);
  graph.ApplyMerge(data_node_find, merge_axis->id);
  EXPECT_EQ(data_node_find->attr.sched.axis.size(), 1U);
  EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 1U);
}

TEST_F(UtestAscendCIR, ApplyMerge_0_not_merge_tensor_but_merge_node) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  const Expression s1 = graph.CreateSizeVar("s1");
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {sym::kSymbolOne, sym::kSymbolOne};

  auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
  EXPECT_NE(merge_axis, nullptr);
  EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  data_node_find->attr.sched.exec_order = 1;
  data_node_find->attr.sched.axis = {s0_axis.id, s1_axis.id};
  graph.ApplyMerge(data_node_find, merge_axis->id);
  EXPECT_EQ(data_node_find->attr.sched.axis.size(), 1U);
  EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 2U);
}

TEST_F(UtestAscendCIR, ApplyMerge_1) {
  AscGraph graph("test_graph");
  const auto s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  const auto s1 = graph.CreateSizeVar("s1");
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
  EXPECT_NE(merge_axis, nullptr);
  EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  graph.ApplyMerge(data_node_find, merge_axis->id);
  EXPECT_EQ((data_node_find->outputs[0].attr.repeats[0U]), (s0*s1));
  EXPECT_EQ((data_node_find->outputs[0].attr.repeats[0U]), (s0*s1));
  EXPECT_EQ((data_node_find->outputs[0].attr.strides[0U]), 1UL);
}

TEST_F(UtestAscendCIR, ApplySchedAxisMerge) {
  {
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplySchedAxisMerge(data_node_find, merge_axis->id);
    EXPECT_EQ(data_node_find->attr.sched.axis.size(), 1U);
    auto sched_axis = data_node_find->attr.sched.axis[0];
    EXPECT_EQ(sched_axis, merge_axis->id);
  }

  {
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplySchedAxisMerge(data_node_find, merge_axis->id, {s0_axis.id, s1_axis.id});
    auto sched_axis = data_node_find->attr.sched.axis[0];
    EXPECT_EQ(sched_axis, merge_axis->id);
  }

  {
    // 非连续，正序场景
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s0_axis.id, s3_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplySchedAxisMerge(data_node_find, merge_axis->id, {s0_axis.id, s3_axis.id});
    EXPECT_EQ(data_node_find->attr.sched.axis[0], merge_axis->id);
    EXPECT_EQ(data_node_find->attr.sched.axis[1], s1_axis.id);
    EXPECT_EQ(data_node_find->attr.sched.axis[2], s2_axis.id);
  }

  {
    // 连续，倒序场景
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s1_axis.id, s0_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplySchedAxisMerge(data_node_find, merge_axis->id, {s1_axis.id, s0_axis.id});
    EXPECT_EQ(data_node_find->attr.sched.axis[0], merge_axis->id);
    EXPECT_EQ(data_node_find->attr.sched.axis[1], s2_axis.id);
    EXPECT_EQ(data_node_find->attr.sched.axis[2], s3_axis.id);
  }

  {
    // 非合并轴场景
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    const Expression s4 = graph.CreateSizeVar("s4");
    Axis &s4_axis = graph.CreateAxis("S4", s4);
    const Expression s5 = graph.CreateSizeVar("s5");
    Axis &s5_axis = graph.CreateAxis("S5", s5);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s4_axis.id, s5_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplySchedAxisMerge(data_node_find, merge_axis->id, {s4_axis.id, s5_axis.id});
    EXPECT_EQ(data_node_find->attr.sched.axis[0], s0_axis.id);
    EXPECT_EQ(data_node_find->attr.sched.axis[1], s1_axis.id);
    EXPECT_EQ(data_node_find->attr.sched.axis[2], s2_axis.id);
    EXPECT_EQ(data_node_find->attr.sched.axis[3], s3_axis.id);
  }
}

TEST_F(UtestAscendCIR, ApplyTensorAxisMerge) {
  {
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisMerge(data_node_find, merge_axis->id);
    auto tensor_axis = data_node_find->outputs[0].attr.axis[0];
    EXPECT_EQ(tensor_axis, merge_axis->id);
    auto tensor_stride = data_node_find->outputs[0].attr.strides[0];
    EXPECT_TRUE(tensor_stride == 1);
    auto tensor_repeat = data_node_find->outputs[0].attr.repeats[0];
    EXPECT_EQ(tensor_repeat, merge_axis->size);
  }

  {
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s0_axis.id, s1_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisMerge(data_node_find, merge_axis->id, {s0_axis.id, s1_axis.id});
    auto tensor_axis = data_node_find->outputs[0].attr.axis[0];
    EXPECT_EQ(tensor_axis, merge_axis->id);
    auto tensor_stride = data_node_find->outputs[0].attr.strides[0];
    EXPECT_TRUE(tensor_stride == 1);
    auto tensor_repeat = data_node_find->outputs[0].attr.repeats[0];
    EXPECT_EQ(tensor_repeat, merge_axis->size);
  }

  {
    // 连续，正序场景
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s1_axis.id, s2_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisMerge(data_node_find, merge_axis->id, {s1_axis.id, s2_axis.id});
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[0], s0_axis.id);
    EXPECT_TRUE(data_node_find->outputs[0].attr.strides[0] == s1*s2*s3);
    EXPECT_EQ(data_node_find->outputs[0].attr.repeats[0], s0_axis.size);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[1], merge_axis->id);
    EXPECT_TRUE(data_node_find->outputs[0].attr.strides[1] == s3);
    EXPECT_EQ(data_node_find->outputs[0].attr.repeats[1], merge_axis->size);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[2], s3_axis.id);
    EXPECT_TRUE(data_node_find->outputs[0].attr.strides[2] == sym::kSymbolOne);
    EXPECT_EQ(data_node_find->outputs[0].attr.repeats[2], s3_axis.size);
  }

  {
    // 连续，倒序场景
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s2_axis.id, s1_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisMerge(data_node_find, merge_axis->id, {s2_axis.id, s1_axis.id});
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[0], s0_axis.id);
    EXPECT_TRUE(data_node_find->outputs[0].attr.strides[0] == s1*s2*s3);
    EXPECT_EQ(data_node_find->outputs[0].attr.repeats[0], s0_axis.size);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[1], merge_axis->id);
    EXPECT_TRUE(data_node_find->outputs[0].attr.strides[1] == s3);
    EXPECT_EQ(data_node_find->outputs[0].attr.repeats[1], merge_axis->size);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[2], s3_axis.id);
    EXPECT_TRUE(data_node_find->outputs[0].attr.strides[2] == sym::kSymbolOne);
    EXPECT_EQ(data_node_find->outputs[0].attr.repeats[2], s3_axis.size);
  }

  {
    // 非连续，倒序场景
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s3_axis.id, s1_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisMerge(data_node_find, merge_axis->id, {s3_axis.id, s1_axis.id});
    EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 4);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[0], s0_axis.id);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[1], s1_axis.id);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[2], s2_axis.id);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[3], s3_axis.id);
  }

  {
    // 未触发合轴
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    const Expression s2 = graph.CreateSizeVar("s2");
    Axis &s2_axis = graph.CreateAxis("S2", s2);
    const Expression s3 = graph.CreateSizeVar("s3");
    Axis &s3_axis = graph.CreateAxis("S3", s3);
    const Expression s4 = graph.CreateSizeVar("s4");
    Axis &s4_axis = graph.CreateAxis("S4", s4);
    const Expression s5 = graph.CreateSizeVar("s5");
    Axis &s5_axis = graph.CreateAxis("S5", s5);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id, s2_axis.id, s3_axis.id};
    *data.y.repeats = {s0, s1, s2, s3};
    *data.y.strides = {s1*s2*s3, s2*s3, s3, sym::kSymbolOne};

    auto merge_axis = graph.MergeAxis({s4_axis.id, s5_axis.id});
    EXPECT_NE(merge_axis, nullptr);
    EXPECT_EQ(merge_axis->type, Axis::kAxisTypeMerged);

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisMerge(data_node_find, merge_axis->id, {s4_axis.id, s5_axis.id});
    EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 4);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[0], s0_axis.id);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[1], s1_axis.id);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[2], s2_axis.id);
    EXPECT_EQ(data_node_find->outputs[0].attr.axis[3], s3_axis.id);
  }
}

TEST_F(UtestAscendCIR, ApplyReorder) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  const Expression s1 = graph.CreateSizeVar("s1");
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  graph.ApplyReorder(data_node_find, {s1_axis.id, s0_axis.id});
}

TEST_F(UtestAscendCIR, ApplyReorder_Sched_Tensor) {
  {
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, sym::kSymbolOne};

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplySchedAxisReorder(data_node_find, {s1_axis.id, s0_axis.id});
  }

  {
    AscGraph graph("test_graph");
    const Expression s0 = graph.CreateSizeVar("s0");
    Axis &s0_axis = graph.CreateAxis("S0", s0);
    const Expression s1 = graph.CreateSizeVar("s1");
    Axis &s1_axis = graph.CreateAxis("S1", s1);
    ascir_op::Data data("data", graph);
    auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
    EXPECT_NE(data_node, nullptr);
    data.attr.sched.exec_order = 1;
    data.attr.sched.axis = {s0_axis.id, s1_axis.id};
    data.y.dtype = ge::DT_FLOAT16;
    data.y.format = ge::FORMAT_ND;
    *data.y.axis = {s0_axis.id, s1_axis.id};
    *data.y.repeats = {s0, s1};
    *data.y.strides = {s1, sym::kSymbolOne};

    auto data_node_find = graph.FindNode("data");
    EXPECT_NE(data_node_find, nullptr);
    graph.ApplyTensorAxisReorder(data_node_find, {s1_axis.id, s0_axis.id});
  }
}

TEST_F(UtestAscendCIR, TryApplyReplace) {
  AscGraph graph("test_graph");
  const Expression s0 = graph.CreateSizeVar("s0");
  const Expression s1 = graph.CreateSizeVar("s1");

  Axis &s0_axis = graph.CreateAxis("S0", s0);
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  Axis &s1_new_axis = graph.CreateAxis("s1_new", s1);

  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  auto data_node_find = graph.FindNode("data");
  ASSERT_NE(data_node_find, nullptr);
  EXPECT_EQ(graph.TryApplyAxisReplace(data_node_find, s1_new_axis, s1_axis), false);
  EXPECT_EQ(data.attr.sched.axis.size(), 2UL);
  EXPECT_EQ(data.attr.sched.axis[0], s0_axis.id);
  EXPECT_EQ(data.attr.sched.axis[1], s1_axis.id);
  EXPECT_EQ(data.y.axis->size(), 2UL);
  EXPECT_EQ((*data.y.axis)[0], s0_axis.id);
  EXPECT_EQ((*data.y.axis)[1], s1_axis.id);

  EXPECT_EQ(data.attr.sched.axis[0], s0_axis.id);
  EXPECT_EQ(graph.TryApplyAxisReplace(data_node_find, s1_axis, s1_new_axis), true);
  EXPECT_EQ(data.attr.sched.axis[0], s0_axis.id);
  EXPECT_EQ(data.attr.sched.axis[1], s1_new_axis.id);
  EXPECT_EQ(data.y.axis->size(), 2UL);
  EXPECT_EQ((*data.y.axis)[0], s0_axis.id);
  EXPECT_EQ((*data.y.axis)[1], s1_new_axis.id);
}

TEST_F(UtestAscendCIR, Operator_OK) {
  AscGraph graph("test_graph");
  Expression s0 = graph.CreateSizeVar("s0");
  Axis &s0_axis = graph.CreateAxis("S0", s0);

  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id};
  *data.y.repeats = {s0};
  *data.y.strides = {sym::kSymbolOne};

  ascir_op::Abs abs("abs");
  auto abs_node = ge::NodeUtilsEx::GetNodeFromOperator(abs);
  EXPECT_EQ(abs_node, nullptr);
  abs.x = data.y;
  // invalid case
  abs.x = AscOpOutput();
  abs_node = ge::NodeUtilsEx::GetNodeFromOperator(abs);
  EXPECT_NE(abs_node, nullptr);

  // find Node
  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  EXPECT_EQ(data_node_find->attr.sched.exec_order, 1);
  EXPECT_EQ(data_node_find->attr.sched.axis.size(), 1U);
  EXPECT_EQ(data_node_find->attr.sched.axis[0], s0_axis.id);
  EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 1);
  EXPECT_EQ(ge::DataType(data_node_find->outputs[0].attr.dtype), ge::DT_FLOAT16);
  data_node_find->outputs[0].attr.dtype = ge::DT_FLOAT;
  EXPECT_EQ(ge::DataType(data_node_find->outputs[0].attr.dtype), ge::DT_FLOAT);
  auto abs_node_find = graph.FindNode("abs");
  EXPECT_NE(abs_node_find, nullptr);

  // GetAllNodes
  int num = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (num == 0) {
      EXPECT_EQ(node->GetName(), "data");
      EXPECT_EQ(node->attr.sched.exec_order, 1);
      EXPECT_EQ(node->attr.sched.axis.size(), 1U);
      EXPECT_EQ(node->attr.sched.axis[0], s0_axis.id);
      EXPECT_EQ(node->outputs[0].attr.axis.size(), 1);
      const auto outputs = node->outputs();
      EXPECT_EQ(outputs.size(), 1U);
      EXPECT_NE(outputs[0], nullptr);
      EXPECT_EQ(outputs[0]->attr.axis.size(), 1);
    }
    if (num == 1) {
      EXPECT_EQ(node->inputs.Size(), 1U);
      EXPECT_EQ(node->inputs[0].attr.axis.size(), 1);
    }
    num++;
  }
  EXPECT_EQ(num, 2);

  // GetAllNodes
  int input_nodes_num = 0;
  for (auto node : graph.GetInputNodes()) {
    if (input_nodes_num == 0) {
      EXPECT_EQ(node->GetName(), "data");
      EXPECT_EQ(node->attr.sched.exec_order, 1);
      EXPECT_EQ(node->attr.sched.axis.size(), 1U);
      EXPECT_EQ(node->attr.sched.axis[0], s0_axis.id);
      EXPECT_EQ(node->outputs[0].attr.axis.size(), 1);
    }
    input_nodes_num++;
  }
  EXPECT_EQ(input_nodes_num, 1);
  EXPECT_EQ(graph.GetName(), "test_graph");

  // GetAllAxis
  const AscGraph &const_graph = graph;
  const auto all_axis = const_graph.GetAllAxis();
  EXPECT_EQ(all_axis.size(), 1U);
}

TEST_F(UtestAscendCIR, Operator_Fail) {
  ascir_op::Abs abs("abs");
  ascir_op::Output output("output");
  output.x = abs.y;
  EXPECT_TRUE(ge::NodeUtilsEx::GetNodeFromOperator(abs) == nullptr);
  EXPECT_TRUE(ge::NodeUtilsEx::GetNodeFromOperator(output) == nullptr);
}

void Add_Layer_Norm_Normal_BeforeAutofuse(AscGraph &graph) {
  auto ONE = sym::kSymbolOne;
  auto ZERO = sym::kSymbolZero;
  // 定义轴的大小
  auto A = Symbol("A");
  auto R = Symbol("R");
  auto BL = Symbol(8, "BL");

  // 定义轴
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  // 定义节点
  int exec_order = 0;
  Data x1("x1", graph);
  x1.attr.sched.exec_order = exec_order++;
  x1.attr.sched.axis = {a.id, r.id, bl.id};
  x1.y.dtype = ge::DT_FLOAT16;
  *x1.y.axis = {a.id, r.id, bl.id};
  *x1.y.repeats = {A, R, ONE};
  *x1.y.strides = {R, ONE, ZERO};

  Load x1Local("x1Local");
  x1Local.x = x1.y;
  x1Local.attr.sched.exec_order = exec_order++;
  x1Local.attr.sched.axis = {a.id, r.id, bl.id};
  x1Local.y.dtype = ge::DT_FLOAT16;
  *x1Local.y.axis = {a.id, r.id, bl.id};
  *x1Local.y.repeats = {A, R, ONE};
  *x1Local.y.strides = {R, ONE, ZERO};

  Data x2("x2", graph);
  x2.attr.sched.exec_order = exec_order++;
  x2.attr.sched.axis = {a.id, r.id, bl.id};
  x2.y.dtype = ge::DT_FLOAT16;
  *x2.y.axis = {a.id, r.id, bl.id};
  *x2.y.repeats = {A, R, ONE};
  *x2.y.strides = {R, ONE, ZERO};

  Load x2Local("x2Local");
  x2Local.x = x2.y;
  x2Local.attr.sched.exec_order = exec_order++;
  x2Local.attr.sched.axis = {a.id, r.id, bl.id};
  x2Local.y.dtype = ge::DT_FLOAT16;
  *x2Local.y.axis = {a.id, r.id, bl.id};
  *x2Local.y.repeats = {A, R, ONE};
  *x2Local.y.strides = {R, ONE, ZERO};

  Data bias("bias", graph);
  bias.attr.sched.exec_order = exec_order++;
  bias.attr.sched.axis = {a.id, r.id, bl.id};
  bias.y.dtype = ge::DT_FLOAT16;
  *bias.y.axis = {a.id, r.id, bl.id};
  *bias.y.repeats = {A, R, ONE};
  *bias.y.strides = {R, ONE, ZERO};

  Load biasLocal("biasLocal");
  biasLocal.x = bias.y;
  biasLocal.attr.sched.exec_order = exec_order++;
  biasLocal.attr.sched.axis = {a.id, r.id, bl.id};
  biasLocal.y.dtype = ge::DT_FLOAT16;
  *biasLocal.y.axis = {a.id, r.id, bl.id};
  *biasLocal.y.repeats = {A, R, ONE};
  *biasLocal.y.strides = {R, ONE, ZERO};

  CalcMean mean("mean");
  mean.x1 = x1Local.y;
  mean.x2 = x2Local.y;
  mean.x3 = biasLocal.y;
  mean.attr.sched.exec_order = exec_order++;
  mean.attr.sched.axis = {a.id, r.id, bl.id};
  mean.y1.dtype = ge::DT_FLOAT;        // mean
  *mean.y1.axis = {a.id, r.id, bl.id};
  *mean.y1.repeats = {A, ONE, ONE};
  *mean.y1.strides = {ONE, ZERO, ZERO};
  mean.y2.dtype = ge::DT_FLOAT16;        // x out
  *mean.y2.axis = {a.id, r.id, bl.id};
  *mean.y2.repeats = {A, R, ONE};
  *mean.y2.strides = {R, ONE, ZERO};
  mean.y3.dtype = ge::DT_FLOAT;        // x fp32
  *mean.y3.axis = {a.id, r.id, bl.id};
  *mean.y3.repeats = {A, R, ONE};
  *mean.y3.strides = {R, ONE, ZERO};
  Store x_out("x_out");
  x_out.attr.sched.exec_order = exec_order++;
  x_out.attr.sched.axis = {a.id, r.id, bl.id};
  x_out.x = mean.y2;
  x_out.y.dtype = ge::DT_FLOAT16;
  *x_out.y.axis = {a.id, r.id, bl.id};
  *x_out.y.repeats = {A, R, ONE};
  *x_out.y.strides = {R, ONE, ZERO};

  Store mean_out("mean_out");
  mean_out.attr.sched.exec_order = exec_order++;
  mean_out.attr.sched.axis = {a.id, r.id, bl.id};
  mean_out.x = mean.y1;
  mean_out.y.dtype = ge::DT_FLOAT;
  *mean_out.y.axis = {a.id, r.id, bl.id};
  *mean_out.y.repeats = {A, ONE, ONE};
  *mean_out.y.strides = {ONE, ZERO, ZERO};

  TbufData one("one", graph);
  one.attr.sched.exec_order = exec_order++;
  one.attr.sched.axis = {a.id, r.id, bl.id};
  one.y.dtype = ge::DT_FLOAT;
  *one.y.axis = {a.id, r.id, bl.id};
  *one.y.repeats = {ONE, ONE, BL};
  *one.y.strides = {ZERO, ZERO, ONE};

  CalcRstd rstd("rstd");
  rstd.attr.sched.exec_order = exec_order++;
  rstd.attr.sched.axis = {a.id, r.id, bl.id};
  rstd.x1 = mean.y3;
  rstd.x2 = mean.y1;
  rstd.x3 = one.y;
  rstd.y1.dtype = ge::DT_FLOAT;      // x-mean
  *rstd.y1.axis = {a.id, r.id, bl.id};
  *rstd.y1.repeats = {A, R, ONE};
  *rstd.y1.strides = {R, ONE, ZERO};
  rstd.y2.dtype = ge::DT_FLOAT;     // rstd
  *rstd.y2.axis = {a.id, r.id, bl.id};
  *rstd.y2.repeats = {A, ONE, ONE};
  *rstd.y2.strides = {ONE, ZERO, ZERO};

  Store rstd_out("rstd_out");
  rstd_out.attr.sched.exec_order = exec_order++;
  rstd_out.attr.sched.axis = {a.id, r.id, bl.id};
  rstd_out.x = rstd.y2;
  rstd_out.y.dtype = ge::DT_FLOAT;
  *rstd_out.y.axis = {a.id, r.id, bl.id};
  *rstd_out.y.repeats = {A, ONE, ONE};
  *rstd_out.y.strides = {ONE, ZERO, ZERO};

  Data beta("beta", graph);
  beta.attr.sched.exec_order = exec_order++;
  beta.attr.sched.axis = {a.id, r.id, bl.id};
  beta.y.dtype = ge::DT_FLOAT16;
  *beta.y.axis = {a.id, r.id, bl.id};
  *beta.y.repeats = {ONE, R, ONE};
  *beta.y.strides = {ZERO, ONE, ZERO};

  Load betaLocal("betaLocal");
  betaLocal.x = beta.y;
  betaLocal.attr.sched.exec_order = exec_order++;
  betaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  betaLocal.y.dtype = ge::DT_FLOAT16;
  *betaLocal.y.axis = {a.id, r.id, bl.id};
  *betaLocal.y.repeats = {ONE, R, ONE};
  *betaLocal.y.strides = {ZERO, ONE, ZERO};

  Data gamma("gamma", graph);
  gamma.attr.sched.exec_order = exec_order++;
  gamma.attr.sched.axis = {a.id, r.id, bl.id};
  gamma.y.dtype = ge::DT_FLOAT16;
  *gamma.y.axis = {a.id, r.id, bl.id};
  *gamma.y.repeats = {ONE, R, ONE};
  *gamma.y.strides = {ZERO, ONE, ZERO};

  Load gammaLocal("gammaLocal");
  gammaLocal.x = gamma.y;
  gammaLocal.attr.sched.exec_order = exec_order++;
  gammaLocal.attr.sched.axis = {a.id, r.id, bl.id};
  gammaLocal.y.dtype = ge::DT_FLOAT16;
  *gammaLocal.y.axis = {a.id, r.id, bl.id};
  *gammaLocal.y.repeats = {ONE, R, ONE};
  *gammaLocal.y.strides = {ZERO, ONE, ZERO};

  CalcY y("y");
  y.attr.sched.exec_order = exec_order++;
  y.attr.sched.axis = {a.id, r.id, bl.id};
  y.x1 = rstd.y1;                 // x-mean
  y.x2 = betaLocal.y;
  y.x3 = gammaLocal.y;
  y.x4 = rstd.y2;                 // rstd
  y.y1.dtype = ge::DT_FLOAT16;
  *y.y1.axis = {a.id, r.id, bl.id};
  *y.y1.repeats = {A, R, ONE};
  *y.y1.strides = {R, ONE, ZERO};

  Store y_out("y_out");
  y_out.attr.sched.exec_order = exec_order++;
  y_out.attr.sched.axis = {a.id, r.id, bl.id};
  y_out.x = y.y1;
  y_out.y.dtype = ge::DT_FLOAT16;
  *y_out.y.axis = {a.id, r.id, bl.id};
  *y_out.y.repeats = {A, R, ONE};
  *y_out.y.strides = {R, ONE, ZERO};

  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.attr.sched.exec_order = exec_order++;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {a.id, r.id, bl.id};
  *buf1.y.repeats = {A, R, ONE};
  *buf1.y.strides = {R, ONE, ZERO};

  Output buf2("buf2");
  buf2.x = mean_out.y;
  buf2.attr.sched.exec_order = exec_order++;
  buf2.y.dtype = ge::DT_FLOAT;
  *buf2.y.axis = {a.id, r.id, bl.id};
  *buf2.y.repeats = {A, ONE, ONE};
  *buf2.y.strides = {ONE, ZERO, ZERO};

  Output buf3("buf3");
  buf3.x = rstd_out.y;
  buf3.attr.sched.exec_order = exec_order++;
  buf3.y.dtype = ge::DT_FLOAT;
  *buf3.y.axis = {a.id, r.id, bl.id};
  *buf3.y.repeats = {A, ONE, ONE};
  *buf3.y.strides = {ONE, ZERO, ZERO};

  Output buf("buf");
  buf.x = y_out.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = {a.id, r.id, bl.id};
  *buf.y.repeats = {A, R, ONE};
  *buf.y.strides = {R, ONE, ZERO};
}

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/

void Add_Layer_Norm_Normal_AfterScheduler(AscGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;

  auto [aBO, aBI] = graph.BlockSplit(a, "nbi", "nbo");   // AB Ab
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");  // AbT Abt
  // graph.UpdateAxisAlign(aBI.id, 1u);
  // graph.UpdateAxisAlign(aBII.id, 8u);
  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1, aBO->id, aBI->id);
  graph.ApplySplit(x1, aBIO->id, aBII->id);
  x1->attr.sched.loop_axis = aBIO->id;
  x1->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2, aBO->id, aBI->id);
  graph.ApplySplit(x2, aBIO->id, aBII->id);
  x2->attr.sched.loop_axis = aBIO->id;
  x2->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto bias = graph.FindNode("bias");
  graph.ApplySplit(bias, aBO->id, aBI->id);
  graph.ApplySplit(bias, aBIO->id, aBII->id);
  bias->attr.sched.loop_axis = aBIO->id;
  bias->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x1Local = graph.FindNode("x1Local");
  graph.ApplySplit(x1Local, aBO->id, aBI->id);
  graph.ApplySplit(x1Local, aBIO->id, aBII->id);
  x1Local->attr.sched.loop_axis = aBIO->id;
  x1Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto x2Local = graph.FindNode("x2Local");
  graph.ApplySplit(x2Local, aBO->id, aBI->id);
  graph.ApplySplit(x2Local, aBIO->id, aBII->id);
  x2Local->attr.sched.loop_axis = aBIO->id;
  x2Local->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto biasLocal = graph.FindNode("biasLocal");
  graph.ApplySplit(biasLocal,aBO->id, aBI->id);
  graph.ApplySplit(biasLocal, aBIO->id, aBII->id);
  biasLocal->attr.sched.loop_axis = aBIO->id;
  biasLocal->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean = graph.FindNode("mean");
  graph.ApplySplit(mean,aBO->id, aBI->id);
  graph.ApplySplit(mean,aBIO->id, aBII->id);
  mean->attr.sched.loop_axis = aBIO->id;
  mean->outputs[0].attr.vectorized_axis = {aBII->id, r};
  mean->outputs[1].attr.vectorized_axis = {aBII->id, r};
  mean->outputs[2].attr.vectorized_axis = {aBII->id, r};

  auto x_out = graph.FindNode("x_out");
  graph.ApplySplit(x_out, aBO->id, aBI->id);
  graph.ApplySplit(x_out, aBIO->id, aBII->id);
  x_out->attr.sched.loop_axis = aBIO->id;
  x_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto mean_out = graph.FindNode("mean_out");
  graph.ApplySplit(mean_out, aBO->id, aBI->id);
  graph.ApplySplit(mean_out, aBIO->id, aBII->id);
  mean_out->attr.sched.loop_axis = aBIO->id;
  mean_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto rstd = graph.FindNode("rstd");
  graph.ApplySplit(rstd,aBO->id, aBI->id);
  graph.ApplySplit(rstd,aBIO->id, aBII->id);
  rstd->attr.sched.loop_axis = aBIO->id;
  rstd->outputs[0].attr.vectorized_axis = {aBII->id, r};
  rstd->outputs[1].attr.vectorized_axis = {aBII->id, r};

  auto rstd_out = graph.FindNode("rstd_out");
  graph.ApplySplit(rstd_out,aBO->id, aBI->id);
  graph.ApplySplit(rstd_out,aBIO->id, aBII->id);
  rstd_out->attr.sched.loop_axis = aBIO->id;
  rstd_out->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto betaLocal = graph.FindNode("betaLocal");
  graph.ApplySplit(betaLocal,aBO->id, aBI->id);
  graph.ApplySplit(betaLocal,aBIO->id, aBII->id);
  betaLocal->attr.sched.loop_axis = aBIO->id;
  betaLocal->outputs[0].attr.vectorized_axis = {r};

  auto gammaLocal = graph.FindNode("gammaLocal");
  graph.ApplySplit(gammaLocal,aBO->id, aBI->id);
  graph.ApplySplit(gammaLocal,aBIO->id, aBII->id);
  gammaLocal->attr.sched.loop_axis = aBIO->id;
  gammaLocal->outputs[0].attr.vectorized_axis = {r};

  auto y = graph.FindNode("y");
  graph.ApplySplit(y,aBO->id, aBI->id);
  graph.ApplySplit(y,aBIO->id, aBII->id);
  y->attr.sched.loop_axis = aBIO->id;
  y->outputs[0].attr.vectorized_axis = {aBII->id, r};

  auto y_out = graph.FindNode("y_out");
  graph.ApplySplit(y_out,aBO->id, aBI->id);
  graph.ApplySplit(y_out,aBIO->id, aBII->id);
  y_out->attr.sched.loop_axis = aBIO->id;
  y_out->outputs[0].attr.vectorized_axis = {aBII->id, r};
}

void Add_Layer_Norm_Normal_AfterQueBufAlloc(AscGraph &graph) {
  int tensorID = 0;
  int queID = 0;
  int bufID = 0;
  int x1Que = queID++;
  int x2Que = queID++;
  int biasQue = queID++;
  int gammaQue = queID++;
  int betaQue = queID++;
  int meanQue = queID++;
  int rstdQue = queID++;
  int yQue = queID++;
  int xQue = queID++;
  int x32Queue = queID++;
  int oneTBuf = bufID++;

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bias = graph.FindNode("bias");
  bias->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bias->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto x1Local = graph.FindNode("x1Local");
  x1Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x1Local->outputs[0].attr.buf.id = ID_NONE;
  x1Local->outputs[0].attr.que.id = x1Que;
  x1Local->outputs[0].attr.que.depth = 1;
  x1Local->outputs[0].attr.que.buf_num = 1;
  x1Local->outputs[0].attr.opt.ref_tensor = ID_NONE;
  x1Local->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto x2Local = graph.FindNode("x2Local");
  x2Local->outputs[0].attr.mem.tensor_id = tensorID++;
  x2Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x2Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x2Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  x2Local->outputs[0].attr.buf.id = ID_NONE;
  x2Local->outputs[0].attr.que.id = x2Que;
  x2Local->outputs[0].attr.que.depth = 1;
  x2Local->outputs[0].attr.que.buf_num = 1;
  x2Local->outputs[0].attr.opt.ref_tensor = ID_NONE;
  x2Local->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto biasLocal = graph.FindNode("biasLocal");
  biasLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  biasLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  biasLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  biasLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  biasLocal->outputs[0].attr.buf.id = ID_NONE;
  biasLocal->outputs[0].attr.que.id = biasQue;
  biasLocal->outputs[0].attr.que.depth = 1;
  biasLocal->outputs[0].attr.que.buf_num = 1;
  biasLocal->outputs[0].attr.opt.ref_tensor = ID_NONE;
  biasLocal->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto mean = graph.FindNode("mean");
  mean->outputs[0].attr.mem.tensor_id = tensorID++;
  mean->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[0].attr.buf.id = ID_NONE;
  mean->outputs[0].attr.que.id = meanQue;
  mean->outputs[0].attr.que.depth = 1;
  mean->outputs[0].attr.que.buf_num = 1;
  mean->outputs[0].attr.opt.ref_tensor = ID_NONE;
  mean->outputs[0].attr.opt.merge_scope = ID_NONE;
  mean->outputs[1].attr.mem.tensor_id = tensorID++;
  mean->outputs[1].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[1].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[1].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[1].attr.buf.id = ID_NONE;
  mean->outputs[1].attr.que.id = xQue;
  mean->outputs[1].attr.que.depth = 1;
  mean->outputs[1].attr.que.buf_num = 1;
  mean->outputs[1].attr.opt.ref_tensor = ID_NONE;
  mean->outputs[1].attr.opt.merge_scope = ID_NONE;
  mean->outputs[2].attr.mem.tensor_id = tensorID++;
  mean->outputs[2].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mean->outputs[2].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mean->outputs[2].attr.mem.position = ge::Position::kPositionVecOut;
  mean->outputs[2].attr.buf.id = ID_NONE;
  mean->outputs[2].attr.que.id = x32Queue;
  mean->outputs[2].attr.que.depth = 1;
  mean->outputs[2].attr.que.buf_num = 1;
  mean->outputs[2].attr.opt.ref_tensor = ID_NONE;
  mean->outputs[2].attr.opt.merge_scope = ID_NONE;

  auto x_out = graph.FindNode("x_out");
  x_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  x_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto mean_out = graph.FindNode("mean_out");
  mean_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  mean_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto one = graph.FindNode("one");
  one->outputs[0].attr.mem.tensor_id = tensorID++;
  one->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  one->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  one->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  one->outputs[0].attr.buf.id = oneTBuf;
  one->outputs[0].attr.que.id = ID_NONE;
  one->outputs[0].attr.que.depth = ID_NONE;
  one->outputs[0].attr.que.buf_num = ID_NONE;
  one->outputs[0].attr.opt.ref_tensor = ID_NONE;
  one->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto rstd = graph.FindNode("rstd");
  rstd->outputs[0].attr.mem.tensor_id = tensorID++;
  rstd->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  rstd->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  rstd->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  rstd->outputs[0].attr.buf.id =ID_NONE;
  rstd->outputs[0].attr.que.id = yQue;
  rstd->outputs[0].attr.que.depth = 1;
  rstd->outputs[0].attr.que.buf_num = 1;
  rstd->outputs[0].attr.opt.ref_tensor = ID_NONE;
  rstd->outputs[0].attr.opt.merge_scope = ID_NONE;
  rstd->outputs[1].attr.mem.tensor_id = tensorID++;
  rstd->outputs[1].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  rstd->outputs[1].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  rstd->outputs[1].attr.mem.position = ge::Position::kPositionVecOut;
  rstd->outputs[1].attr.buf.id = ID_NONE;
  rstd->outputs[1].attr.que.id = rstdQue;
  rstd->outputs[1].attr.que.depth = 1;
  rstd->outputs[1].attr.que.buf_num = 1;
  rstd->outputs[1].attr.opt.ref_tensor = ID_NONE;
  rstd->outputs[1].attr.opt.merge_scope = ID_NONE;

  auto rstd_out = graph.FindNode("rstd_out");
  rstd_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  rstd_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto beta = graph.FindNode("beta");
  beta->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  beta->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto betaLocal = graph.FindNode("betaLocal");
  betaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  betaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  betaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  betaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  betaLocal->outputs[0].attr.buf.id = ID_NONE;
  betaLocal->outputs[0].attr.que.id = betaQue;
  betaLocal->outputs[0].attr.que.depth = 1;
  betaLocal->outputs[0].attr.que.buf_num = 1;
  betaLocal->outputs[0].attr.opt.ref_tensor = ID_NONE;
  betaLocal->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto gamma = graph.FindNode("gamma");
  gamma->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  gamma->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto gammaLocal = graph.FindNode("gammaLocal");
  gammaLocal->outputs[0].attr.mem.tensor_id = tensorID++;
  gammaLocal->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  gammaLocal->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  gammaLocal->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  gammaLocal->outputs[0].attr.buf.id = ID_NONE;
  gammaLocal->outputs[0].attr.que.id = gammaQue;
  gammaLocal->outputs[0].attr.que.depth = 1;
  gammaLocal->outputs[0].attr.que.buf_num = 1;
  gammaLocal->outputs[0].attr.opt.ref_tensor = ID_NONE;
  gammaLocal->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto y = graph.FindNode("y");
  y->outputs[0].attr.mem.tensor_id = tensorID++;
  y->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  y->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  y->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  y->outputs[0].attr.buf.id = ID_NONE;
  y->outputs[0].attr.que.id = yQue;
  y->outputs[0].attr.que.depth = 1;
  y->outputs[0].attr.que.buf_num = 1;
  y->outputs[0].attr.opt.ref_tensor = ID_NONE;
  y->outputs[0].attr.opt.merge_scope = ID_NONE;

  auto y_out = graph.FindNode("y_out");
  y_out->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  y_out->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}

TEST_F(UtestAscendCIR, CheckValid) {
  AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Add_Layer_Norm_Normal_BeforeAutofuse(graph_normal);
  Add_Layer_Norm_Normal_AfterScheduler(graph_normal);
  Add_Layer_Norm_Normal_AfterQueBufAlloc(graph_normal);
  EXPECT_EQ(graph_normal.CheckValid(), true);
  auto betaLocal = graph_normal.FindNode("betaLocal");
  betaLocal->outputs[0].attr.que.id = ID_NONE;
  EXPECT_EQ(graph_normal.CheckValid(), false);
}
TEST_F(UtestAscendCIR, CreateContiguousData) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data = graph.CreateContiguousData("data0", ge::DT_INT8, {a});
  EXPECT_EQ(data.output_index, 0U);
  EXPECT_EQ(static_cast<ge::DataType>(data.dtype), ge::DT_INT8);
  EXPECT_EQ(data.format, ge::FORMAT_ND);
  EXPECT_EQ(*data.axis, std::vector<int64_t>({a.id}));
  const auto &attr = graph.FindNode("data0")->attr;
  EXPECT_EQ(attr.sched.exec_order, 0U);
  EXPECT_EQ(attr.sched.loop_axis, -1);
  EXPECT_TRUE(attr.ir_attr != nullptr);
  int64_t value_get{-1};
  EXPECT_EQ(attr.ir_attr->GetAttrValue("index", value_get), GRAPH_SUCCESS);
  EXPECT_EQ(value_get, 0U);
  auto data1 = graph.CreateContiguousData("data1", ge::DT_FLOAT, {a, b}, ge::FORMAT_DHWCN);
  EXPECT_EQ(data1.output_index, 0U);
  EXPECT_EQ(static_cast<ge::DataType>(data1.dtype), ge::DT_FLOAT);
  EXPECT_EQ(data1.format, ge::FORMAT_DHWCN);
  EXPECT_EQ(*data1.axis, std::vector<int64_t>({a.id, b.id}));
  const auto &attr1 = graph.FindNode("data1")->attr;
  EXPECT_EQ(attr1.sched.exec_order, 1U);
  EXPECT_EQ(attr1.sched.loop_axis, -1);
  EXPECT_TRUE(attr1.ir_attr != nullptr);
  const auto &ir_attr1 = dynamic_cast<AscDataIrAttrDef &>(*attr1.ir_attr);
  EXPECT_EQ(ir_attr1.GetIndex(value_get), GRAPH_SUCCESS);
  EXPECT_EQ(value_get, 1U);
}

TEST_F(UtestAscendCIR, CreateContiguousOut) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto output1 = graph.CreateContiguousOut("out1", ge::DT_FLOAT16, {a, b}, ge::FORMAT_ND);

  EXPECT_EQ(output1.output_index, 0U);
  EXPECT_EQ(static_cast<ge::DataType>(output1.dtype), ge::DT_FLOAT16);
  EXPECT_EQ(output1.format, ge::FORMAT_ND);
  const auto *attr1 = dynamic_cast< AscNodeAttr *>(&(graph.FindNode("out1")->attr));
  EXPECT_EQ(attr1->sched.exec_order, -1);
  EXPECT_EQ(attr1->sched.loop_axis, -1);
}

TEST_F(UtestAscendCIR, StoreToOut) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data1 = graph.CreateContiguousData("data1", ge::DT_FLOAT, {a, b}, ge::FORMAT_DHWCN);
  auto output1 = graph.CreateContiguousOut("out1", ge::DT_FLOAT16, {a, b, c}, ge::FORMAT_ND);
  auto load1 = ascir::cg::LoadStub("load1", data1);
  ascir::cg::Store("StoreLoad1ToOutput1", load1, output1);
  EXPECT_EQ(output1.GetOwnerOp().GetInputsSize(), 1U);
  EXPECT_EQ(output1.output_index, 0U);
  EXPECT_EQ(static_cast<ge::DataType>(output1.dtype), ge::DT_FLOAT16);
  EXPECT_EQ(output1.format, ge::FORMAT_ND);
  EXPECT_EQ(*output1.axis, std::vector<int64_t>({a.id, b.id, c.id}));
  const auto *attr1 = dynamic_cast< AscNodeAttr *>(&(graph.FindNode("out1")->attr));
  EXPECT_EQ(attr1->sched.exec_order, 3);
  EXPECT_EQ(attr1->sched.loop_axis, -1);
}

TEST_F(UtestAscendCIR, StoreWithOffset) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data1 = graph.CreateContiguousData("data1", ge::DT_FLOAT, {a, b}, ge::FORMAT_DHWCN);
  auto output1 = graph.CreateContiguousOut("out1", ge::DT_FLOAT16, {a, b, c}, ge::FORMAT_ND);
  auto load1 = ascir::cg::LoadStub("load1", data1);
  int64_t offset1 = 1024;
  ascir::cg::StoreStub("StoreLoad1ToOutput1", load1, offset1);
  EXPECT_EQ(output1.GetOwnerOp().GetInputsSize(), 1U);
  auto store_node = graph.FindNode("StoreLoad1ToOutput1");
  EXPECT_NE(store_node->attr.ir_attr, nullptr);
  int64_t offset_get{-1};
  EXPECT_EQ(store_node->attr.ir_attr->GetAttrValue("offset", offset_get), GRAPH_SUCCESS);
  EXPECT_EQ(offset_get, offset1);
  auto data2 = graph.CreateContiguousData("data1", ge::DT_FLOAT, {a, b}, ge::FORMAT_DHWCN);
  auto load2 = ascir::cg::LoadStub("load2", data2);
  auto store2 = ascir_op::StoreStub("Store2");
  store2.x = load2;
  int64_t offset2 = 256;
  // 设置属性
  store2.ir_attr.SetOffset(offset2);
  auto store2_node = graph.FindNode("Store2");
  EXPECT_NE(store2_node->attr.ir_attr, nullptr);
  // 获取的方式1，调用子类的函数
  EXPECT_EQ(dynamic_cast<ascir_op::StoreStub::AscStoreStubIrAttrDef *>(store2_node->attr.ir_attr.get())->GetOffset(offset_get),
            GRAPH_SUCCESS);
  EXPECT_EQ(offset_get, offset2);
  // 获取方式2，调用基类的函数
  EXPECT_EQ(store2_node->attr.ir_attr->GetAttrValue("offset", offset_get), GRAPH_SUCCESS);
  EXPECT_EQ(offset_get, offset2);
}

TEST_F(UtestAscendCIR, IrAttrTest) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("(1 + D)");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data1 = graph.CreateContiguousData("data1", ge::DT_FLOAT, {a, b}, ge::FORMAT_DHWCN);
  auto stub_op1 = ascir_op::StubOp1("stub_op1");
  // input
  stub_op1.x = data1;
  // 通过op的方式设置attr
  stub_op1.ir_attr.SetMy_float(0.1);
  stub_op1.ir_attr.SetMy_int(1);
  stub_op1.ir_attr.SetMy_string("stub_test");
  stub_op1.ir_attr.SetOffset(D);
  auto node = graph.FindNode("stub_op1");
  EXPECT_NE(node, nullptr);
  EXPECT_NE(node->attr.ir_attr, nullptr);
  // 通过node的方式获取属性
  auto my_ir_attrs = dynamic_cast<ascir_op::StubOp1::AscStubOp1IrAttrDef *>(node->attr.ir_attr.get());
  EXPECT_NE(my_ir_attrs, nullptr);
  int64_t get_valuei;
  float get_valuef;
  std::string get_values;
  Expression get_expression;
  EXPECT_EQ(my_ir_attrs->GetMy_int(get_valuei), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(my_ir_attrs->GetMy_float(get_valuef), GRAPH_SUCCESS);
  EXPECT_EQ(my_ir_attrs->GetMy_string(get_values), GRAPH_SUCCESS);
  EXPECT_EQ(my_ir_attrs->GetOffset(get_expression), GRAPH_SUCCESS);
  EXPECT_EQ(get_valuei, 1);
  EXPECT_FLOAT_EQ(get_valuef, 0.1);
  EXPECT_EQ(get_values, "stub_test");
  EXPECT_EQ(get_expression, D);
  // 成员函数测试
  ascendc_ir::proto::AscIrAttrDef asc_ir_attr_def;
  my_ir_attrs->Serialize(asc_ir_attr_def);
  const std::string kExpected = R"PROTO(attr {
  key: "my_float"
  value {
    f: 0.1
  }
}
attr {
  key: "my_int"
  value {
    i: 1
  }
}
attr {
  key: "my_string"
  value {
    s: "stub_test"
  }
}
attr {
  key: "offset"
  value {
    expression: "(1 + D)"
  }
}
)PROTO";
  EXPECT_EQ(asc_ir_attr_def.DebugString(), kExpected);
  ascir_op::StubOp1::AscStubOp1IrAttrDef ir_attr_obj2;
  EXPECT_EQ(ir_attr_obj2.Deserialize(asc_ir_attr_def), GRAPH_SUCCESS);
  EXPECT_EQ(ir_attr_obj2.GetMy_int(get_valuei), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(ir_attr_obj2.GetMy_float(get_valuef), GRAPH_SUCCESS);
  EXPECT_EQ(ir_attr_obj2.GetMy_string(get_values), GRAPH_SUCCESS);
  EXPECT_EQ(ir_attr_obj2.GetOffset(get_expression), GRAPH_SUCCESS);
  EXPECT_EQ(get_valuei, 1);
  EXPECT_FLOAT_EQ(get_valuef, 0.1);
  EXPECT_EQ(get_values, "stub_test");
  EXPECT_TRUE(get_expression.IsValid());
  EXPECT_TRUE(D.IsValid());
  EXPECT_EQ(std::string(get_expression.Str().get()), std::string(D.Str().get()));

  auto ir_attr_obj_base = ir_attr_obj2.Clone();
  EXPECT_NE(ir_attr_obj_base, nullptr);
  EXPECT_EQ(ir_attr_obj_base->GetAttrValue("my_int", get_valuei), GRAPH_SUCCESS);
  EXPECT_NE(ir_attr_obj_base->GetAttrValue("others_int", get_valuei), GRAPH_SUCCESS);
  EXPECT_NE(ir_attr_obj_base->GetAttrValue("my_int", get_valuef), GRAPH_SUCCESS);
  EXPECT_EQ(ir_attr_obj_base->GetAttrValue("my_float", get_valuef), GRAPH_SUCCESS);
  EXPECT_EQ(ir_attr_obj_base->GetAttrValue("my_string", get_values), GRAPH_SUCCESS);
  EXPECT_EQ(ir_attr_obj_base->GetAttrValue("offset", get_expression), GRAPH_SUCCESS);
  EXPECT_EQ(get_valuei, 1);
  EXPECT_FLOAT_EQ(get_valuef, 0.1);
  EXPECT_EQ(get_values, "stub_test");
  EXPECT_TRUE(get_expression.IsValid());
  EXPECT_TRUE(D.IsValid());
  EXPECT_EQ(std::string(get_expression.Str().get()), std::string(D.Str().get()));
}

//REG_ASC_IR(StubOp2)
//.Input("x1", "T")
//.Input("x2", "T")
//.Output("y", "T")
//.DataType("T", TensorType{DT_INT32, DT_INT64});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp2_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp2";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 1U);

    // 校验同sym的输入的dtype是否在注册范围内并且一致
    GE_WARN_ASSERT(input_dtypes[0] == input_dtypes[1]);
    const static std::set<ge::DataType> support_dtypes_of_sym_T = {DT_INT32, DT_INT64};
    GE_WARN_ASSERT(support_dtypes_of_sym_T.find(input_dtypes[0]) != support_dtypes_of_sym_T.end());

    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(input_dtypes[0]);
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(input_dtypes[0] == expect_output_dtypes[0]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  //REG_ASC_IR(StubOp2)
  //.Input("x1", "T")
  //.Input("x2", "T")
  //.Output("y", "T")
  //.DataType("T", TensorType{DT_INT32, DT_INT64});
  OpDtypeInfer<ascir_op::StubOp2>().Input(DT_INT32).Input(DT_INT32).Expect(DT_INT32).AssertSucceed();
  // check input and output num
  OpDtypeInfer<ascir_op::StubOp2>().Input(DT_INT32).Input(DT_INT32).Input(DT_INT32).Expect(DT_INT32).AssertFailed();
  OpDtypeInfer<ascir_op::StubOp2>().Input(DT_INT32).Input(DT_INT32).Expect(DT_INT32).Expect(DT_INT32).AssertFailed();
  // check input same dtype of same sym
  OpDtypeInfer<ascir_op::StubOp2>().Input(DT_INT32).Input(DT_INT64).Expect(DT_INT32).AssertFailed();
  // check output same dtype of same sym of input
  OpDtypeInfer<ascir_op::StubOp2>().Input(DT_INT32).Input(DT_INT32).Expect(DT_INT64).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp2_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp2";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty());

    // 校验同sym的输入的dtype是否一致
    GE_WARN_ASSERT(input_dtypes[0] == input_dtypes[1]);

    expect_output_dtypes.push_back(input_dtypes[0]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

//REG_ASC_IR(StubOp3)
//.Input("x1", "T1")
//.Input("x2", "T2")
//.Input("x3", "T1")
//.Output("y1", "T1")
//.Output("y2", "T2")
//.DataType("T1", TensorType{DT_INT32, DT_INT64})
//.DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp3_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp3";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 3U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 2U);

    // 校验同sym的输入的dtype是否在注册范围内并且一致
    GE_WARN_ASSERT(input_dtypes[0] == input_dtypes[2]);
    const static std::set<ge::DataType> support_dtypes_of_sym_T1 = {DT_INT32, DT_INT64};
    GE_WARN_ASSERT(support_dtypes_of_sym_T1.find(input_dtypes[0]) != support_dtypes_of_sym_T1.end());
    const static std::set<ge::DataType> support_dtypes_of_sym_T2 = {DT_FLOAT, DT_FLOAT16};
    GE_WARN_ASSERT(support_dtypes_of_sym_T2.find(input_dtypes[1]) != support_dtypes_of_sym_T2.end());

    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(input_dtypes[0]);
      expect_output_dtypes.push_back(input_dtypes[1]);
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(input_dtypes[0] == expect_output_dtypes[0]);
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[1]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  //REG_ASC_IR(StubOp3)
  //.Input("x1", "T1")
  //.Input("x2", "T2")
  //.Input("x3", "T1")
  //.Output("y1", "T1")
  //.Output("y2", "T2")
  //.DataType("T1", TensorType{DT_INT32, DT_INT64})
  //.DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT});
  OpDtypeInfer<ascir_op::StubOp3>().Input(DT_INT32).Input(DT_FLOAT).Input(DT_INT32).Expect(DT_INT32).Expect(DT_FLOAT).AssertSucceed();
  // check input and output num
  OpDtypeInfer<ascir_op::StubOp3>().Input(DT_INT32).Input(DT_FLOAT).Input(DT_INT32).Input(DT_INT32).Expect(DT_INT32).Expect(
      DT_FLOAT).AssertFailed();
  OpDtypeInfer<ascir_op::StubOp3>().Input(DT_INT32).Input(DT_FLOAT).Input(DT_INT32).Expect(DT_INT32).Expect(DT_INT32).Expect(
      DT_FLOAT).AssertFailed();
  // check input same dtype of same sym
  OpDtypeInfer<ascir_op::StubOp3>().Input(DT_INT32).Input(DT_FLOAT).Input(DT_INT64).Expect(DT_INT32).Expect(DT_FLOAT).AssertFailed();
  // check output same dtype of same sym of input
  OpDtypeInfer<ascir_op::StubOp3>().Input(DT_INT32).Input(DT_FLOAT).Input(DT_INT32).Expect(DT_INT32).Expect(DT_INT64).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp3_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp3";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 3U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty());

    // 校验同sym的输入的dtype是否一致
    GE_WARN_ASSERT(input_dtypes[0] == input_dtypes[2]);

    expect_output_dtypes.push_back(input_dtypes[0]);
    expect_output_dtypes.push_back(input_dtypes[1]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

//REG_ASC_IR(StubOp4)
//.Input("x1", "T1")
//.Input("x2", "T2")
//.Output("y1", "T3")
//.Output("y2", "T3")
//.Output("y3", "T2")
//.DataType("T1", TensorType{DT_INT32, DT_INT64})
//.DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT})
//.DataType("T3", TensorType{DT_DOUBLE, DT_BOOL});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp4_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp4";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 3U);

    // 校验同sym的输入的dtype是否在注册范围内并且一致
    const static std::set<ge::DataType> support_dtypes_of_sym_T1 = {DT_INT32, DT_INT64};
    GE_WARN_ASSERT(support_dtypes_of_sym_T1.find(input_dtypes[0]) != support_dtypes_of_sym_T1.end());
    const static std::set<ge::DataType> support_dtypes_of_sym_T2 = {DT_FLOAT, DT_FLOAT16};
    GE_WARN_ASSERT(support_dtypes_of_sym_T2.find(input_dtypes[1]) != support_dtypes_of_sym_T2.end());

    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      GELOGW("Output ir_index [0] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
      GELOGW("Output ir_index [1] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
      return FAILED;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(expect_output_dtypes[0] == expect_output_dtypes[1]);
    static std::set<ge::DataType> support_dtypes_of_sym_T3 = {DT_DOUBLE, DT_BOOL};
    GE_WARN_ASSERT(support_dtypes_of_sym_T3.find(expect_output_dtypes[0]) != support_dtypes_of_sym_T3.end());
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[2]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  //REG_ASC_IR(StubOp4)
  //.Input("x1", "T1")
  //.Input("x2", "T2")
  //.Output("y1", "T3")
  //.Output("y2", "T3")
  //.Output("y3", "T2")
  //.DataType("T1", TensorType{DT_INT32, DT_INT64})
  //.DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT})
  //.DataType("T3", TensorType{DT_DOUBLE, DT_BOOL});
  OpDtypeInfer<ascir_op::StubOp4>().Input(DT_INT32).Input(DT_FLOAT16).Expect(DT_DOUBLE).Expect(DT_DOUBLE).Expect(
      DT_FLOAT16).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp4>().Input(DT_INT32).Input(DT_FLOAT).Expect(DT_DOUBLE).Expect(DT_DOUBLE).Expect(
      DT_FLOAT).AssertSucceed();
  // check input and output num
  OpDtypeInfer<ascir_op::StubOp4>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).Expect(DT_DOUBLE).Expect(DT_DOUBLE).Expect(
      DT_FLOAT16).AssertFailed();
  OpDtypeInfer<ascir_op::StubOp4>().Input(DT_INT32).Input(DT_FLOAT16).Expect(DT_DOUBLE).Expect(DT_DOUBLE).Expect(
      DT_FLOAT16).Expect(DT_FLOAT16).AssertFailed();
  // check out dtype of same sym
  OpDtypeInfer<ascir_op::StubOp4>().Input(DT_INT32).Input(DT_FLOAT16).Expect(DT_DOUBLE).Expect(DT_FLOAT16).Expect(
      DT_FLOAT16).AssertFailed();
  // infer out failed
  OpDtypeInfer<ascir_op::StubOp4>().Input(DT_INT32).Input(DT_FLOAT16).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp4_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp4";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty());

    // 校验同sym的输入的dtype是否一致

    GELOGW("Output ir_index [0] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
    GELOGW("Output ir_index [1] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
    return FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

//REG_ASC_IR(StubOp4New)
//    .Input("x1", "T1")
//    .Input("x2", "T2")
//    .Output("y1", "T3")
//    .Output("y2", "T3")
//    .Output("y3", "T2")
//    .Impl({"socv1"},
//          {nullptr,
//           nullptr,
//           {{"T1", TensorType{DT_INT32, DT_INT64}},
//            {"T2", TensorType{DT_FLOAT16, DT_FLOAT}},
//            {"T3", TensorType{DT_DOUBLE, DT_BOOL}}}});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp4New_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  class RuntimeMock : public RuntimeStub {
   public:
    rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
      (void) strcpy(version, "socv1");
      return RT_ERROR_NONE;
    }
  };
  RuntimeMock mock_runtime;
  RuntimeStub::Install(&mock_runtime);

  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp4New";
  const std::string target_func = "InferDataType";

  auto [sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 3U);

    // 校验同sym的输入的dtype是否在注册范围内并且一致
    std::set<ge::DataType> support_dtypes_of_sym_T1;
    if (npu_arch == "socv1") {
      support_dtypes_of_sym_T1 = {DT_INT32, DT_INT64};
    } else if (npu_arch == "socv2") {
      support_dtypes_of_sym_T1 = {DT_INT32, DT_UINT16, DT_INT64};
    } else if (npu_arch == "socv3") {
      support_dtypes_of_sym_T1 = {DT_INT32, DT_UINT16, DT_INT64};
    } else {
      GELOGE(ge::FAILED, "Unknown npu arch: %s", npu_arch.c_str());
      return ge::FAILED;
    }
    GE_WARN_ASSERT(support_dtypes_of_sym_T1.find(input_dtypes[0]) != support_dtypes_of_sym_T1.end());
    std::set<ge::DataType> support_dtypes_of_sym_T2;
    if (npu_arch == "socv1") {
      support_dtypes_of_sym_T2 = {DT_FLOAT, DT_FLOAT16};
    } else if (npu_arch == "socv2") {
      support_dtypes_of_sym_T2 = {DT_FLOAT, DT_FLOAT16, DT_UINT16};
    } else if (npu_arch == "socv3") {
      support_dtypes_of_sym_T2 = {DT_FLOAT, DT_FLOAT16, DT_UINT16};
    } else {
      GELOGE(ge::FAILED, "Unknown npu arch: %s", npu_arch.c_str());
      return ge::FAILED;
    }
    GE_WARN_ASSERT(support_dtypes_of_sym_T2.find(input_dtypes[1]) != support_dtypes_of_sym_T2.end());

    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      GELOGW("Output ir_index [0] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
      GELOGW("Output ir_index [1] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
      return FAILED;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(expect_output_dtypes[0] == expect_output_dtypes[1]);
    static std::set<ge::DataType> support_dtypes_of_sym_T3 = {DT_DOUBLE, DT_BOOL};
    GE_WARN_ASSERT(support_dtypes_of_sym_T3.find(expect_output_dtypes[0]) != support_dtypes_of_sym_T3.end());
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[2]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code)) << "Actual code:\n"
                                                              << actual_code << "\nExpected:\n"
                                                              << expected_code;

  OpDtypeInfer<ascir_op::StubOp4New>()
      .Input(DT_INT32)
      .Input(DT_FLOAT16)
      .Expect(DT_DOUBLE)
      .Expect(DT_DOUBLE)
      .Expect(DT_FLOAT16)
      .AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp4New>()
      .Input(DT_INT32)
      .Input(DT_FLOAT)
      .Expect(DT_DOUBLE)
      .Expect(DT_DOUBLE)
      .Expect(DT_FLOAT)
      .AssertSucceed();
  // check input and output num
  OpDtypeInfer<ascir_op::StubOp4New>()
      .Input(DT_INT32)
      .Input(DT_FLOAT16)
      .Input(DT_INT32)
      .Expect(DT_DOUBLE)
      .Expect(DT_DOUBLE)
      .Expect(DT_FLOAT16)
      .AssertFailed();
  OpDtypeInfer<ascir_op::StubOp4New>()
      .Input(DT_INT32)
      .Input(DT_FLOAT16)
      .Expect(DT_DOUBLE)
      .Expect(DT_DOUBLE)
      .Expect(DT_FLOAT16)
      .Expect(DT_FLOAT16)
      .AssertFailed();
  // check out dtype of same sym
  OpDtypeInfer<ascir_op::StubOp4New>()
      .Input(DT_INT32)
      .Input(DT_FLOAT16)
      .Expect(DT_DOUBLE)
      .Expect(DT_FLOAT16)
      .Expect(DT_FLOAT16)
      .AssertFailed();
  // infer out failed
  OpDtypeInfer<ascir_op::StubOp4New>().Input(DT_INT32).Input(DT_FLOAT16).AssertFailed();
  RuntimeStub::UnInstall(&mock_runtime);
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp4New_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp4New";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto [sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty());

    // 校验同sym的输入的dtype是否一致

    GELOGW("Output ir_index [0] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
    GELOGW("Output ir_index [1] has multi result {DT_DOUBLE, DT_BOOL}, can not infer.");
    return FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code)) << "Actual code:\n"
                                                              << actual_code << "\nExpected:\n"
                                                              << expected_code;
}

//REG_ASC_IR(StubOp5)
//.Input("x1", "T1")
//.DynamicInput("x2", "T2")
//.Output("y1", "T1")
//.Output("y2", "T2")
//.DataType("T1", TensorType{DT_INT32, DT_INT64})
//.DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp5_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp5";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 2U);

    // 校验同sym的输入的dtype是否在注册范围内并且一致
    const static std::set<ge::DataType> support_dtypes_of_sym_T1 = {DT_INT32, DT_INT64};
    GE_WARN_ASSERT(support_dtypes_of_sym_T1.find(input_dtypes[0]) != support_dtypes_of_sym_T1.end());
    const static std::set<ge::DataType> support_dtypes_of_sym_T2 = {DT_FLOAT, DT_FLOAT16};
    GE_WARN_ASSERT(support_dtypes_of_sym_T2.find(input_dtypes[1]) != support_dtypes_of_sym_T2.end());

    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(input_dtypes[0]);
      expect_output_dtypes.push_back(input_dtypes[1]);
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(input_dtypes[0] == expect_output_dtypes[0]);
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[1]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  //REG_ASC_IR(StubOp5)
  //.Input("x1", "T1")
  //.DynamicInput("x2", "T2")
  //.Output("y1", "T1")
  //.Output("y2", "T2")
  //.DataType("T1", TensorType{DT_INT32, DT_INT64})
  //.DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT});
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT16).Expect(DT_INT32).Expect(DT_FLOAT16).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT).Expect(DT_INT32).Expect(DT_FLOAT).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT).AssertSucceed();
  // check input and output num
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).Expect(DT_DOUBLE).Expect(DT_DOUBLE).AssertFailed();
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_FLOAT16).Expect(DT_DOUBLE).Expect(
      DT_DOUBLE).AssertFailed();
  // check out dtype of same sym
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT16).Expect(DT_INT32).Expect(DT_INT32).AssertFailed();
  OpDtypeInfer<ascir_op::StubOp5>().Input(DT_INT32).Input(DT_FLOAT16).Expect(DT_FLOAT16).Expect(DT_FLOAT16).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp5_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp5";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 2U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty());

    // 校验同sym的输入的dtype是否一致

    expect_output_dtypes.push_back(input_dtypes[0]);
    expect_output_dtypes.push_back(input_dtypes[1]);
    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

//REG_ASC_IR(StubOp6)
//.Input("x1", "T1")
//.Input("x2", "T2")
//.Input("x3", "T1")
//.Output("y1", "T3")
//.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT64})
//.DataType("T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT})
//.DataType("T3", OrderedTensorTypeList{DT_BOOL, DT_INT8});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp6_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp6";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
    inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 3U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 1U);

    GE_WARN_ASSERT(input_dtypes[0] == input_dtypes[2]);
    const static std::map<std::vector<ge::DataType>, ge::DataType> results = {
        {{DT_INT32, DT_FLOAT16}, DT_BOOL},
        {{DT_INT64, DT_FLOAT}, DT_INT8}
    };
    auto iter = results.find(std::vector<ge::DataType>{input_dtypes[0], input_dtypes[1]});
    GE_WARN_ASSERT(iter != results.end());
    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(iter->second);
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(iter->second == expect_output_dtypes[0]);

    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

// REG_ASC_IR(StubOp6)
//.Input("x1", "T1")
//.Input("x2", "T2")
//.Input("x3", "T1")
//.Output("y1", "T3")
//.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT64})
//.DataType("T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT})
//.DataType("T3", OrderedTensorTypeList{DT_BOOL, DT_INT8});
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).Expect(DT_BOOL).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT64).Expect(DT_INT8).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT64).AssertSucceed();
  // check input dtype of same sym
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT32).AssertFailed();
  // check inputs indicies not match
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT64).Input(DT_FLOAT16).Input(DT_INT64).Expect(DT_BOOL).AssertFailed();
  // check output input indicies not match
  OpDtypeInfer<ascir_op::StubOp6>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT64).Expect(DT_BOOL).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp6_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp6";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    (void)input_dtypes;
    (void)expect_output_dtypes;
    // 输入输出存在关联, 无法进行推导
    (void)input_dtypes;
    (void)expect_output_dtypes;
    GELOGW("Node type %s is not supported to infernocheck for dtype.", Type);
    return ge::FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

// 输出存在多个解的情况
//REG_ASC_IR(StubOp7)
// .Input("x1", "T1")
// .Input("x2", "T2")
// .Input("x3", "T1")
// .Output("y1", "T3")
// .Output("y2", "T2")
//.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
//.DataType("T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT16, DT_FLOAT})
//.DataType("T3", OrderedTensorTypeList{DT_BOOL, DT_INT4, DT_INT8});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp7_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp7";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
      inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 3U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 2U);

    GE_WARN_ASSERT(input_dtypes[0] == input_dtypes[2]);
    const static std::map<std::vector<ge::DataType>, std::set<ge::DataType>> results = {
        {{DT_INT32, DT_FLOAT16}, {DT_BOOL, DT_INT4}},
        {{DT_INT64, DT_FLOAT}, {DT_INT8}}
    };
    auto iter = results.find(std::vector<ge::DataType>{input_dtypes[0], input_dtypes[1]});
    GE_WARN_ASSERT(iter != results.end());
    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      GE_WARN_ASSERT(iter->second.size() == 1U);
      expect_output_dtypes.push_back(*(iter->second.begin()));
      expect_output_dtypes.push_back(input_dtypes[1]);
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(iter->second.find(expect_output_dtypes[0]) != iter->second.end());
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[1]);

    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  //REG_ASC_IR(StubOp7)
  // .Input("x1", "T1")
  // .Input("x2", "T2")
  // .Input("x3", "T1")
  // .Output("y1", "T3")
  // .Output("y2", "T2")
  //.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
  //.DataType("T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT16, DT_FLOAT})
  //.DataType("T3", OrderedTensorTypeList{DT_BOOL, DT_INT4, DT_INT8});
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).Expect(DT_BOOL).Expect(DT_FLOAT16).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).Expect(DT_INT4).Expect(DT_FLOAT16).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT64).Expect(DT_INT8).Expect(DT_FLOAT).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT64).AssertSucceed();
  // infer out failed by multi result
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT32).AssertFailed();
  // check input dtype of same sym
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT32).Input(DT_FLOAT16).Input(DT_INT64).Expect(DT_BOOL).Expect(DT_FLOAT16).AssertFailed();
  // check output input indicies not match
  OpDtypeInfer<ascir_op::StubOp7>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_INT64).Expect(DT_INT4).Expect(DT_FLOAT).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp7_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp7";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    (void)input_dtypes;
    (void)expect_output_dtypes;
    // 输入输出存在关联, 无法进行推导
    (void)input_dtypes;
    (void)expect_output_dtypes;
    GELOGW("Node type %s is not supported to infernocheck for dtype.", Type);
    return ge::FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

// 单输入单输出完全一对一，带有重复解优化测试
//REG_ASC_IR(StubOp8)
//.Input("x", "T1")
//.Output("y", "T2")
//.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
//.DataType("T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp8_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp8";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
   inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 1U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 1U);

    const static std::map<ge::DataType, ge::DataType> results = {
        {DT_INT32, DT_BF16},
        {DT_INT64, DT_FLOAT}
    };
    auto iter = results.find(input_dtypes[0]);
    GE_WARN_ASSERT(iter != results.end());
    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(iter->second);
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(iter->second == expect_output_dtypes[0]);

    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  //REG_ASC_IR(StubOp8)
  //.Input("x", "T1")
  //.Output("y", "T2")
  //.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
  //.DataType("T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT});
  OpDtypeInfer<ascir_op::StubOp8>().Input(DT_INT32).Expect(DT_BF16).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp8>().Input(DT_INT64).Expect(DT_FLOAT).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp8>().Input(DT_INT32).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp8>().Input(DT_INT64).AssertSucceed();
  // check output input indicies not match
  OpDtypeInfer<ascir_op::StubOp8>().Input(DT_INT64).Expect(DT_BF16).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp8_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp8";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    (void)input_dtypes;
    (void)expect_output_dtypes;
    // 输入输出存在关联, 无法进行推导
    (void)input_dtypes;
    (void)expect_output_dtypes;
    GELOGW("Node type %s is not supported to infernocheck for dtype.", Type);
    return ge::FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

// 单输入单输出完全一对一，带有重复解优化测试
//REG_ASC_IR(StubOp8New)
//    .Input("x", "T1")
//    .Output("y", "T2")
//    .Impl({"socv1"},
//          {nullptr,
//           nullptr,
//           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}}, {"T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT}}}});

TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp8New_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  class RuntimeMock : public RuntimeStub {
   public:
    rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
      (void) strcpy(version, "socv1");
      return RT_ERROR_NONE;
    }
  };
  RuntimeMock mock_runtime;
  RuntimeStub::Install(&mock_runtime);
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp8New";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
    inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 1U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 1U);

    std::map<ge::DataType, std::set<ge::DataType>> results;
    if (npu_arch == "socv1") {
       results = {
        {DT_INT32, {DT_BF16}},
        {DT_INT64, {DT_FLOAT}}
      };
    } else if (npu_arch == "socv2") {
       results = {
        {DT_INT32, {DT_BF16}},
        {DT_INT64, {DT_FLOAT}}
      };
    } else if (npu_arch == "socv3") {
       results = {
        {DT_INT32, {DT_BF16}},
        {DT_INT64, {DT_FLOAT}}
      };
    } else {
      GELOGE(ge::FAILED, "Unknown npu arch: %s", npu_arch.c_str());
      return ge::FAILED;
    }

    auto iter = results.find(input_dtypes[0]);
    GE_WARN_ASSERT(iter != results.end());
    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      GE_WARN_ASSERT(iter->second.size() == 1U);
      expect_output_dtypes.push_back(*(iter->second.begin()));
      return ge::SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(iter->second.find(expect_output_dtypes[0]) != iter->second.end());

    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
      << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  OpDtypeInfer<ascir_op::StubOp8New>().Input(DT_INT32).Expect(DT_BF16).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp8New>().Input(DT_INT64).Expect(DT_FLOAT).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp8New>().Input(DT_INT32).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp8New>().Input(DT_INT64).AssertSucceed();
  // check output input indicies not match
  OpDtypeInfer<ascir_op::StubOp8New>().Input(DT_INT64).Expect(DT_BF16).AssertFailed();
  RuntimeStub::UnInstall(&mock_runtime);
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp8New_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp8New";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    (void)input_dtypes;
    (void)expect_output_dtypes;
    // 输入输出存在关联, 无法进行推导
    (void)input_dtypes;
    (void)expect_output_dtypes;
    GELOGW("Node type %s is not supported to infernocheck for dtype.", Type);
    return ge::FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
      << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

// 多输入，多输出， 输出唯一解和多个解混合的复杂场景
//REG_ASC_IR(StubOp9)
//.Input("x1", "T1")
//.Input("x2", "T2")
//.Input("x3", "T3")
//.Output("y1", "T2")
//.Output("y2", "T1")
//.Output("y3", "T4")
//.Output("y4", "T5")
//.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
//.DataType("T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT})
//.DataType("T3", OrderedTensorTypeList{DT_INT8, DT_INT8, DT_FLOAT})
//.DataType("T4", OrderedTensorTypeList{DT_BOOL, DT_DOUBLE, DT_FLOAT})
//.DataType("T5", OrderedTensorTypeList{DT_BOOL, DT_COMPLEX128, DT_DUAL});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp9_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp9";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
     inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 3U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 4U);

    const static std::map<std::vector<ge::DataType>, std::vector<std::set<ge::DataType>>> results = {
        {{DT_INT32, DT_BF16, DT_INT8}, {{DT_DOUBLE, DT_BOOL}, {DT_BOOL, DT_COMPLEX128}}},
        {{DT_INT64, DT_FLOAT, DT_FLOAT}, {{DT_FLOAT}, {DT_DUAL}}}
    };
    auto iter = results.find(std::vector<ge::DataType>{input_dtypes[0], input_dtypes[1], input_dtypes[2]});
    GE_WARN_ASSERT(iter != results.end());
    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(input_dtypes[1]);
      expect_output_dtypes.push_back(input_dtypes[0]);
      GE_WARN_ASSERT(iter->second[0].size() == 1U);
      expect_output_dtypes.push_back(*(iter->second[0].begin()));
      GE_WARN_ASSERT(iter->second[1].size() == 1U);
      expect_output_dtypes.push_back(*(iter->second[1].begin()));
      return SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[0]);
    GE_WARN_ASSERT(input_dtypes[0] == expect_output_dtypes[1]);
    GE_WARN_ASSERT(iter->second[0].find(expect_output_dtypes[2]) != iter->second[0].end());
    GE_WARN_ASSERT(iter->second[1].find(expect_output_dtypes[3]) != iter->second[1].end());

    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  // 多输入，多输出， 输出唯一解和多个解混合的复杂场景
  //REG_ASC_IR(StubOp9)
  //.Input("x1", "T1")
  //.Input("x2", "T2")
  //.Input("x3", "T3")
  //.Output("y1", "T2")
  //.Output("y2", "T1")
  //.Output("y3", "T4")
  //.Output("y4", "T5")
  //.DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
  //.DataType("T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT})
  //.DataType("T3", OrderedTensorTypeList{DT_INT8, DT_INT8, DT_FLOAT})
  //.DataType("T4", OrderedTensorTypeList{DT_BOOL, DT_DOUBLE, DT_FLOAT})
  //.DataType("T5", OrderedTensorTypeList{DT_BOOL, DT_COMPLEX128, DT_DUAL});
  OpDtypeInfer<ascir_op::StubOp9>().Input(DT_INT32).Input(DT_BF16).Input(DT_INT8).Expect(DT_BF16).Expect(DT_INT32).Expect(
      DT_BOOL).Expect(DT_BOOL).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp9>().Input(DT_INT32).Input(DT_BF16).Input(DT_INT8).Expect(DT_BF16).Expect(DT_INT32).Expect(
      DT_DOUBLE).Expect(DT_COMPLEX128).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp9>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_FLOAT).Expect(DT_FLOAT).Expect(DT_INT64).Expect(
      DT_FLOAT).Expect(DT_DUAL).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp9>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_FLOAT).AssertSucceed();
  // infer out failed of multi result
  OpDtypeInfer<ascir_op::StubOp9>().Input(DT_INT32).Input(DT_BF16).Input(DT_INT8).AssertFailed();
  // check failed of error indicies
  OpDtypeInfer<ascir_op::StubOp9>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_FLOAT).Expect(DT_FLOAT).Expect(DT_INT64).Expect(
      DT_DOUBLE).Expect(DT_DUAL).AssertFailed();
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp9_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp9";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    (void)input_dtypes;
    (void)expect_output_dtypes;
    // 输入输出存在关联, 无法进行推导
    (void)input_dtypes;
    (void)expect_output_dtypes;
    GELOGW("Node type %s is not supported to infernocheck for dtype.", Type);
    return ge::FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

// 多输入，多输出， 输出唯一解和多个解混合的复杂场景
REG_ASC_IR(StubOp9New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T3")
    .Output("y1", "T2")
    .Output("y2", "T1")
    .Output("y3", "T4")
    .Output("y4", "T5")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}},
            {"T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT}},
            {"T3", OrderedTensorTypeList{DT_INT8, DT_INT8, DT_FLOAT}},
            {"T4", OrderedTensorTypeList{DT_BOOL, DT_DOUBLE, DT_FLOAT}},
            {"T5", OrderedTensorTypeList{DT_BOOL, DT_COMPLEX128, DT_DUAL}}}});
TEST_F(UtestAscendCIR, CheckInferDtypeImplementation_StubOp9New_InferDataType) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  class RuntimeMock : public RuntimeStub {
   public:
    rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
      (void) strcpy(version, "socv1");
      return RT_ERROR_NONE;
    }
  };
  RuntimeMock mock_runtime;
  RuntimeStub::Install(&mock_runtime);
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp9New";
  const std::string target_func = "InferDataType";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataType(const std::vector<DataType>& input_dtypes,
                                     std::vector<DataType>& expect_output_dtypes,
                                     [[maybe_unused]]const std::string& npu_arch) {
    // 校验入参容器的元素个数是否合法
    GE_ASSERT_EQ(input_dtypes.size(), 3U);
    GE_ASSERT_TRUE(expect_output_dtypes.empty() || expect_output_dtypes.size() == 4U);

    std::map<std::vector<ge::DataType>, std::vector<std::set<ge::DataType>>> results;
    if (npu_arch == "socv1") {
       results = {
        {{DT_INT32, DT_BF16, DT_INT8}, {{DT_DOUBLE, DT_BOOL}, {DT_BOOL, DT_COMPLEX128}}},
        {{DT_INT64, DT_FLOAT, DT_FLOAT}, {{DT_FLOAT}, {DT_DUAL}}}
      };
    } else if (npu_arch == "socv2") {
       results = {
        {{DT_INT32, DT_BF16, DT_INT8}, {{DT_DOUBLE, DT_BOOL}, {DT_BOOL, DT_COMPLEX128}}},
        {{DT_INT64, DT_FLOAT, DT_FLOAT}, {{DT_FLOAT}, {DT_DUAL}}}
      };
    } else if (npu_arch == "socv3") {
       results = {
        {{DT_INT32, DT_BF16, DT_INT8}, {{DT_DOUBLE, DT_BOOL}, {DT_BOOL, DT_COMPLEX128}}},
        {{DT_INT64, DT_FLOAT, DT_FLOAT}, {{DT_FLOAT}, {DT_DUAL}}}
      };
    } else {
      GELOGE(ge::FAILED, "Unknown npu arch: %s", npu_arch.c_str());
      return ge::FAILED;
    }

    auto iter = results.find(std::vector<ge::DataType>{input_dtypes[0], input_dtypes[1], input_dtypes[2]});
    GE_WARN_ASSERT(iter != results.end());
    // 输出外部不指定的时候，生成推导的代码
    if (expect_output_dtypes.empty()) {
      expect_output_dtypes.push_back(input_dtypes[1]);
      expect_output_dtypes.push_back(input_dtypes[0]);
      GE_WARN_ASSERT(iter->second[0].size() == 1U);
      expect_output_dtypes.push_back(*(iter->second[0].begin()));
      GE_WARN_ASSERT(iter->second[1].size() == 1U);
      expect_output_dtypes.push_back(*(iter->second[1].begin()));
      return ge::SUCCESS;
    }
    // 输出外部指定，生成校验的代码
    GE_WARN_ASSERT(input_dtypes[1] == expect_output_dtypes[0]);
    GE_WARN_ASSERT(input_dtypes[0] == expect_output_dtypes[1]);
    GE_WARN_ASSERT(iter->second[0].find(expect_output_dtypes[2]) != iter->second[0].end());
    GE_WARN_ASSERT(iter->second[1].find(expect_output_dtypes[3]) != iter->second[1].end());

    return SUCCESS;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
      << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;

  OpDtypeInfer<ascir_op::StubOp9New>().Input(DT_INT32).Input(DT_BF16).Input(DT_INT8).Expect(DT_BF16).Expect(DT_INT32).Expect(
                                                                                                                      DT_BOOL).Expect(DT_BOOL).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp9New>().Input(DT_INT32).Input(DT_BF16).Input(DT_INT8).Expect(DT_BF16).Expect(DT_INT32).Expect(
                                                                                                                      DT_DOUBLE).Expect(DT_COMPLEX128).AssertSucceed();
  OpDtypeInfer<ascir_op::StubOp9New>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_FLOAT).Expect(DT_FLOAT).Expect(DT_INT64).Expect(
                                                                                                                         DT_FLOAT).Expect(DT_DUAL).AssertSucceed();
  // infer out successfully
  OpDtypeInfer<ascir_op::StubOp9New>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_FLOAT).AssertSucceed();
  // infer out failed of multi result
  OpDtypeInfer<ascir_op::StubOp9New>().Input(DT_INT32).Input(DT_BF16).Input(DT_INT8).AssertFailed();
  // check failed of error indicies
  OpDtypeInfer<ascir_op::StubOp9New>().Input(DT_INT64).Input(DT_FLOAT).Input(DT_FLOAT).Expect(DT_FLOAT).Expect(DT_INT64).Expect(
                                                                                                                         DT_DOUBLE).Expect(DT_DUAL).AssertFailed();
  RuntimeStub::UnInstall(&mock_runtime);
}

TEST_F(UtestAscendCIR, CheckInferDataTypeWithNoCheckImplementation_StubOp9New_InferDataTypeWithNoCheck) {
  GTEST_SKIP() << "线上二进制冲突，待下一次更新run包后打开.";
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class = "StubOp9New";
  const std::string target_func = "InferDataTypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
  inline static Status InferDataTypeWithNoCheck(const std::vector<DataType>& input_dtypes,
                                                std::vector<DataType>& expect_output_dtypes,
                                                [[maybe_unused]]const std::string& npu_arch = "") {
    (void)input_dtypes;
    (void)expect_output_dtypes;
    // 输入输出存在关联, 无法进行推导
    (void)input_dtypes;
    (void)expect_output_dtypes;
    GELOGW("Node type %s is not supported to infernocheck for dtype.", Type);
    return ge::FAILED;
  };
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
      << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

TEST_F(UtestAscendCIR, AscNodeAttr_copy_constrcut_for_ir_attr) {
  AscGraph graph("test_graph");
  auto A = Symbol("A");
  auto B = Symbol("B");
  auto C = Symbol("C");
  auto D = Symbol("d");
  auto a = graph.CreateAxis("a", A);
  auto b = graph.CreateAxis("b", B);
  auto c = graph.CreateAxis("c", C);
  auto d = graph.CreateAxis("d", D);
  auto data1 = graph.CreateContiguousData("data1", ge::DT_FLOAT, {a, b}, ge::FORMAT_DHWCN);
  auto stub_op1 = ascir_op::StubOp1("stub_op1");
  // input
  stub_op1.x = data1;
  // 通过op的方式设置attr
  stub_op1.ir_attr.SetMy_float(0.1);
  stub_op1.ir_attr.SetMy_int(1);
  stub_op1.ir_attr.SetMy_string("stub_test");
  auto node = graph.FindNode("stub_op1");
  EXPECT_NE(node, nullptr);

  auto node_attr2 = node->attr;
  // 测试拷贝构造时，如果有ir_attr，则ir_attr的clone方法会被调用，进行ir_attr的拷贝
  EXPECT_NE(node_attr2.ir_attr, nullptr);
  auto my_ir_attrs = node_attr2.ir_attr->DownCastTo<ascir_op::StubOp1::AscStubOp1IrAttrDef>();
  EXPECT_NE(my_ir_attrs, nullptr);
  int64_t get_valuei;
  float get_valuef;
  std::string get_values;
  EXPECT_EQ(my_ir_attrs->GetMy_int(get_valuei), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(my_ir_attrs->GetMy_float(get_valuef), GRAPH_SUCCESS);
  EXPECT_EQ(my_ir_attrs->GetMy_string(get_values), GRAPH_SUCCESS);
  EXPECT_EQ(get_valuei, 1);
  EXPECT_FLOAT_EQ(get_valuef, 0.1);
  EXPECT_EQ(get_values, "stub_test");
}

TEST_F(UtestAscendCIR, Concat_OK) {
  AscGraph graph("test_graph");
  Expression s0 = graph.CreateSizeVar("s0");
  Expression s1 = graph.CreateSizeVar("s1");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  Axis &s1_axis = graph.CreateAxis("S1", s1);

  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Abs abs("abs");
  auto abs_node = ge::NodeUtilsEx::GetNodeFromOperator(abs);
  EXPECT_EQ(abs_node, nullptr);
  abs.x = data.y;
  abs.attr.sched.exec_order = 2;
  abs.attr.sched.axis = {s0_axis.id, s1_axis.id};
  abs.y.dtype = ge::DT_FLOAT16;
  abs.y.format = ge::FORMAT_ND;
  *abs.y.axis = {s0_axis.id, s1_axis.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Exp exp("exp");
  auto exp_node = ge::NodeUtilsEx::GetNodeFromOperator(exp);
  EXPECT_EQ(exp_node, nullptr);
  exp.x = data.y;
  exp.attr.sched.exec_order = 3;
  exp.attr.sched.axis = {s0_axis.id, s1_axis.id};
  exp.y.dtype = ge::DT_FLOAT16;
  exp.y.format = ge::FORMAT_ND;
  *exp.y.axis = {s0_axis.id, s1_axis.id};
  *exp.y.repeats = {s0, s1};
  *exp.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Concat concat("concat");
  auto concat_node = ge::NodeUtilsEx::GetNodeFromOperator(concat);
  EXPECT_EQ(concat_node, nullptr);
  concat.x = {abs.y, exp.y};
  concat.attr.sched.exec_order = 4;
  concat.attr.sched.axis = {s0_axis.id, s1_axis.id};
  concat.y.dtype = ge::DT_FLOAT16;
  concat.y.format = ge::FORMAT_ND;
  *concat.y.axis = {s0_axis.id, s1_axis.id};
  *concat.y.repeats = {s0, s1 * ge::Symbol(2)};
  *concat.y.strides = {s1* ge::Symbol(2), sym::kSymbolOne};


  // find Node
  auto data_node_find = graph.FindNode("data");
  EXPECT_NE(data_node_find, nullptr);
  EXPECT_EQ(data_node_find->attr.sched.exec_order, 1);
  EXPECT_EQ(data_node_find->attr.sched.axis.size(), 2U);
  EXPECT_EQ(data_node_find->attr.sched.axis[0], s0_axis.id);
  EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 2U);
  EXPECT_EQ(ge::DataType(data_node_find->outputs[0].attr.dtype), ge::DT_FLOAT16);
  auto abs_node_find = graph.FindNode("abs");
  EXPECT_NE(abs_node_find, nullptr);

  // GetAllNodes
  int num = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (num == 0) {
      EXPECT_EQ(node->GetName(), "data");
      EXPECT_EQ(node->attr.sched.exec_order, 1);
      EXPECT_EQ(node->attr.sched.axis.size(), 2U);
      EXPECT_EQ(node->attr.sched.axis[0], s0_axis.id);
      const auto outputs = node->outputs();
      EXPECT_EQ(outputs.size(), 1U);
      EXPECT_NE(outputs[0], nullptr);
      EXPECT_EQ(outputs[0]->attr.axis.size(), 2);
    }
    if (node->GetName() == "concat") {
      EXPECT_EQ(node->attr.sched.axis.size(), 2U);
      EXPECT_EQ(node->attr.sched.axis[0], s0_axis.id);
      EXPECT_EQ(node->outputs[0].attr.axis.size(), 2);
      const auto outputs = node->outputs();
      EXPECT_EQ(outputs.size(), 1U);
      EXPECT_NE(outputs[0], nullptr);
      EXPECT_EQ(outputs[0]->attr.axis.size(), 2);
      EXPECT_EQ(outputs[0]->attr.axis[0], s0_axis.id);
      EXPECT_EQ(outputs[0]->attr.axis[1], s1_axis.id);
    }
    num++;
  }
  EXPECT_EQ(num, 4);

  // GetAllNodes
  int input_nodes_num = 0;
  for (auto node : graph.GetInputNodes()) {
    if (input_nodes_num == 0) {
      EXPECT_EQ(node->GetName(), "data");
      EXPECT_EQ(node->attr.sched.exec_order, 1);
      EXPECT_EQ(node->attr.sched.axis.size(), 2U);
      EXPECT_EQ(node->attr.sched.axis[0], s0_axis.id);
      EXPECT_EQ(node->attr.sched.axis[1], s1_axis.id);
      EXPECT_EQ(node->outputs[0].attr.axis.size(), 2);
    }
    input_nodes_num++;
  }
  EXPECT_EQ(input_nodes_num, 1);
  EXPECT_EQ(graph.GetName(), "test_graph");

  // GetAllAxis
  const AscGraph &const_graph = graph;
  const auto all_axis = const_graph.GetAllAxis();
  EXPECT_EQ(all_axis.size(), 2U);
}

TEST_F(UtestAscendCIR, CreateStartNodesWithoutGraph) {
  AscGraph graph("test_graph");

  ascir_op::Data data("data");
  ascir_op::Constant constant("constant");
  ascir_op::Workspace ws("workspace");
  ascir_op::TbufData t_buf("t_buf");
  graph.AddNode(data);
  graph.AddNode(constant);
  graph.AddNode(ws);
  graph.AddNode(t_buf);

  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  auto const_node = ge::NodeUtilsEx::GetNodeFromOperator(constant);
  EXPECT_NE(const_node, nullptr);
  auto ws_node = ge::NodeUtilsEx::GetNodeFromOperator(ws);
  EXPECT_NE(ws_node, nullptr);
  auto t_buf_node = ge::NodeUtilsEx::GetNodeFromOperator(t_buf);
  EXPECT_NE(t_buf_node, nullptr);
}

TEST_F(UtestAscendCIR, CopyFrom) {
  AscGraph sub_graph("Sub1");
  ascir_op::Data sub_data("sub_data", sub_graph);
  sub_data.attr.api.type = ApiType::kAPITypeBuffer;
  sub_data.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Abs sub_abs("sub_abs");
  sub_abs.x = sub_data.y;
  sub_abs.attr.sched.exec_order = 2;
  sub_abs.y.dtype = ge::DT_FLOAT16;
  sub_abs.y.format = ge::FORMAT_ND;

  ascir_op::Output sub_out("sub_out");
  sub_out.x = sub_abs.y;
  sub_out.attr.api.type = ApiType::kAPITypeBuffer;
  sub_out.attr.api.unit = ComputeUnit::kUnitMTE2;

  AscGraph sub_graph2("Sub2");
  ascir_op::Data sub_data2("sub_data", sub_graph2);
  sub_data2.attr.api.type = ApiType::kAPITypeBuffer;
  sub_data2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Abs sub_abs2("sub_abs");
  sub_abs2.x = sub_data2.y;
  sub_abs2.attr.sched.exec_order = 2;
  sub_abs2.y.dtype = ge::DT_FLOAT16;
  sub_abs2.y.format = ge::FORMAT_ND;

  ascir_op::Output sub_out2("sub_out");
  sub_out2.x = sub_abs2.y;
  sub_out2.attr.api.type = ApiType::kAPITypeBuffer;
  sub_out2.attr.api.unit = ComputeUnit::kUnitMTE2;

  AscGraph graph("test_graph");
  Expression s0 = graph.CreateSizeVar("s0");
  Expression s1 = graph.CreateSizeVar("s1");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  Axis &s1_axis = graph.CreateAxis("S1", s1);

  ascir_op::Data data("data", graph);
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  EXPECT_NE(data_node, nullptr);
  data.attr.api.type = ApiType::kAPITypeBuffer;
  data.attr.api.unit = ComputeUnit::kUnitMTE1;
  data.attr.api.compute_type = ComputeType::kComputeLoad; // fake to check
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Abs abs("abs");
  auto abs_node = ge::NodeUtilsEx::GetNodeFromOperator(abs);
  EXPECT_EQ(abs_node, nullptr);
  abs.x = data.y;
  abs.attr.sched.exec_order = 2;
  abs.attr.sched.axis = {s0_axis.id, s1_axis.id};
  abs.y.dtype = ge::DT_FLOAT16;
  abs.y.format = ge::FORMAT_ND;
  *abs.y.axis = {s0_axis.id, s1_axis.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Exp exp("exp");
  auto exp_node = ge::NodeUtilsEx::GetNodeFromOperator(exp);
  EXPECT_EQ(exp_node, nullptr);
  exp.x = data.y;
  exp.attr.sched.exec_order = 3;
  exp.attr.sched.axis = {s0_axis.id, s1_axis.id};
  exp.y.dtype = ge::DT_FLOAT16;
  exp.y.format = ge::FORMAT_ND;
  *exp.y.axis = {s0_axis.id, s1_axis.id};
  *exp.y.repeats = {s0, s1};
  *exp.y.strides = {s1, sym::kSymbolOne};

  ascir_op::Concat concat("concat");
  auto concat_node = ge::NodeUtilsEx::GetNodeFromOperator(concat);
  EXPECT_EQ(concat_node, nullptr);
  concat.x = {abs.y, exp.y};
  concat.x = {abs.y, exp.y};  // reinit
  concat.attr.sched.exec_order = 4;
  concat.attr.sched.axis = {s0_axis.id, s1_axis.id};
  concat.y.dtype = ge::DT_FLOAT16;
  concat.y.format = ge::FORMAT_ND;
  *concat.y.axis = {s0_axis.id, s1_axis.id};
  *concat.y.repeats = {s0, s1 * ge::Symbol(2)};
  *concat.y.strides = {s1* ge::Symbol(2), sym::kSymbolOne};
  EXPECT_EQ(graph.AddSubGraph(sub_graph), ge::SUCCESS);
  EXPECT_EQ(graph.AddSubGraph(sub_graph2), ge::SUCCESS);

  auto cg = ge::AscGraphUtils::GetComputeGraph(graph);
  auto attr = cg->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  ASSERT_NE(attr, nullptr);
  attr->type = ge::AscGraphType::kImplGraph;

  AscGraph copy_graph(graph.GetName().c_str());
  copy_graph.CopyFrom(graph);
  ge::AscGraph sub1("tmp");
  EXPECT_EQ(copy_graph.FindSubGraph("Sub1", sub1), ge::SUCCESS);
  EXPECT_NE(copy_graph.FindSubGraph("Sub3", sub1), ge::SUCCESS);

  auto new_cg = ge::AscGraphUtils::GetComputeGraph(copy_graph);
  auto new_attr = new_cg->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  ASSERT_NE(new_attr, nullptr);
  EXPECT_EQ(new_attr->type, ge::AscGraphType::kImplGraph);

  std::vector<ge::AscGraph> copied_subs;
  EXPECT_EQ(copy_graph.GetAllSubGraphs(copied_subs), ge::SUCCESS);
  ASSERT_EQ(copied_subs.size(), 2UL);

  auto tmp_node = graph.FindNode("concat");
  ASSERT_NE(tmp_node, nullptr);
  ge::AscGraph owner_graph("onwer");
  ASSERT_EQ(ge::AscGraphUtils::FromComputeGraph(tmp_node->GetOwnerComputeGraph(), owner_graph), ge::SUCCESS);

  std::vector<ge::AscGraph> copied_subs_new;
  EXPECT_EQ(owner_graph.GetAllSubGraphs(copied_subs_new), ge::SUCCESS);
  ASSERT_EQ(copied_subs_new.size(), 2UL);

  // check graph attr
  auto all_axis = copy_graph.GetAllAxis();
  EXPECT_EQ(all_axis.size(), 2);
  EXPECT_EQ(all_axis[0]->name, "S0");
  EXPECT_EQ(all_axis[0]->size, s0);
  EXPECT_EQ(all_axis[1]->name, "S1");
  EXPECT_EQ(all_axis[1]->size, s1);
  auto all_sizevar = copy_graph.GetAllSizeVar();
  EXPECT_EQ(all_sizevar.size(), 2);
  EXPECT_EQ(all_sizevar[0]->expr, Symbol("s0"));
  EXPECT_EQ(all_sizevar[1]->expr, Symbol("s1"));

  // check node tensor attr
  // data
  auto data_node_find = graph.FindNode("data");
  ASSERT_NE(data_node_find, nullptr);
  EXPECT_EQ(data_node_find->attr.sched.exec_order, 1);
  EXPECT_EQ(data_node_find->attr.sched.axis.size(), 2U);
  EXPECT_EQ(data_node_find->attr.sched.axis[0], s0_axis.id);
  EXPECT_EQ(data_node_find->attr.api.unit, ComputeUnit::kUnitMTE1);
  EXPECT_EQ(data_node_find->attr.api.type, ApiType::kAPITypeBuffer);
  EXPECT_EQ(data_node_find->attr.api.compute_type, ComputeType::kComputeLoad);
  EXPECT_EQ(data_node_find->outputs[0].attr.axis.size(), 2U);
  EXPECT_EQ(ge::DataType(data_node_find->outputs[0].attr.dtype), ge::DT_FLOAT16);

  auto data_node_copy = copy_graph.FindNode("data");
  ASSERT_NE(data_node_copy, nullptr);
  EXPECT_EQ(data_node_copy->attr.sched.exec_order, 1);
  ASSERT_EQ(data_node_copy->attr.sched.axis.size(), 2U);
  EXPECT_EQ(data_node_copy->attr.api.unit, data_node_find->attr.api.unit);
  EXPECT_EQ(data_node_copy->attr.api.type, data_node_find->attr.api.type);
  EXPECT_EQ(data_node_copy->attr.api.compute_type, data_node_find->attr.api.compute_type);
  EXPECT_EQ(data_node_copy->outputs[0].attr.axis.size(), 2U);
  EXPECT_EQ(ge::DataType(data_node_copy->outputs[0].attr.dtype), ge::DT_FLOAT16);

  // 测试深拷贝
  data_node_find->outputs[0].attr.dtype = ge::DT_INT8;
  EXPECT_EQ(ge::DataType(data_node_copy->outputs[0].attr.dtype), ge::DT_FLOAT16);
  auto data_type_copy = data_node_find->outputs[0].attr.dtype;
  EXPECT_TRUE(data_type_copy == ge::DT_INT8);
  AscTensorDataType data_type_assign;
  // 异常测试
  data_type_assign = ge::DT_INT8;
  EXPECT_EQ(data_type_assign, ge::DT_UNDEFINED);
  // 测试浅拷贝
  data_type_assign = data_node_find->outputs[0].attr.dtype;
  EXPECT_TRUE(data_type_assign == ge::DT_INT8);

  // concat
  auto concat_node_find = graph.FindNode("concat");
  EXPECT_NE(concat_node_find, nullptr);
  EXPECT_EQ(concat_node_find->attr.sched.exec_order, 4);
  EXPECT_EQ(concat_node_find->attr.sched.axis.size(), 2U);
  EXPECT_EQ(concat_node_find->attr.sched.axis[1], s1_axis.id);
  EXPECT_EQ(concat_node_find->outputs[0].attr.axis.size(), 2U);
  EXPECT_EQ(ge::DataType(concat_node_find->outputs[0].attr.dtype), ge::DT_FLOAT16);

  auto concat_node_copy = copy_graph.FindNode("concat");
  EXPECT_NE(concat_node_copy, nullptr);
  EXPECT_EQ(concat_node_copy->attr.sched.exec_order, 4);
  EXPECT_EQ(concat_node_copy->attr.sched.axis.size(), 2U);
  EXPECT_EQ(concat_node_copy->attr.sched.axis[1], s1_axis.id);
  EXPECT_EQ(concat_node_copy->outputs[0].attr.axis.size(), 2U);
  EXPECT_EQ(ge::DataType(concat_node_copy->outputs[0].attr.dtype), ge::DT_FLOAT16);

  // check link (concat)
  auto in_node = concat_node_find->GetInDataNodes();
  EXPECT_EQ(in_node.size(), 2);
  EXPECT_EQ(in_node.at(0)->GetName(), "abs");
  EXPECT_EQ(in_node.at(1)->GetName(), "exp");

  auto in_node_copy = concat_node_copy->GetInDataNodes();
  EXPECT_EQ(in_node_copy.size(), 2);
  EXPECT_EQ(in_node_copy.at(0)->GetName(), "abs");
  EXPECT_EQ(in_node_copy.at(1)->GetName(), "exp");
}

TEST_F(UtestAscendCIR, CopyAttrFrom) {
  AscGraph graph("graph");
  graph.SetTilingKey(160);
  graph.SetGraphType(ge::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto &z0 = graph.CreateAxis("z0", s0);
  z0.type = ge::Axis::Type::kAxisTypeBlockOuter;
  z0.bind_block = true;
  auto &z1 = graph.CreateAxis("z1", ge::sym::kSymbolOne);
  z1.type = ge::Axis::Type::kAxisTypeTileInner;
  z1.from = {z0.id};

  AscGraph target_graph("target");
  ASSERT_TRUE(target_graph.CopyAttrFrom(graph));

  EXPECT_EQ(target_graph.GetTilingKey(), 160);
  EXPECT_EQ(target_graph.GetGraphType(), ge::AscGraphType::kImplGraph);
  auto all_size_var = target_graph.GetAllSizeVar();
  ASSERT_EQ(all_size_var.size(), 1UL);
  EXPECT_EQ(all_size_var[0]->expr, s0);

  auto all_axes = target_graph.GetAllAxis();
  ASSERT_EQ(all_axes.size(), 2UL);
  EXPECT_EQ(all_axes[0]->name, z0.name);
  EXPECT_EQ(all_axes[0]->type, ge::Axis::Type::kAxisTypeBlockOuter);
  EXPECT_EQ(all_axes[0]->bind_block, true);
  EXPECT_EQ(all_axes[0]->size, s0);

  EXPECT_EQ(all_axes[1]->name, z1.name);
  EXPECT_EQ(all_axes[1]->size, ge::sym::kSymbolOne);
  EXPECT_EQ(all_axes[1]->type, ge::Axis::Type::kAxisTypeTileInner);
  EXPECT_EQ(all_axes[1]->from, std::vector<int64_t>{z0.id});
}

TEST_F(UtestAscendCIR, CopyNodeAttr) {
  AscGraph graph("graph");
  Expression s0 = graph.CreateSizeVar("s0");
  Expression s1 = graph.CreateSizeVar("s1");
  Axis &z0 = graph.CreateAxis("S0", s0);
  Axis &z1 = graph.CreateAxis("S1", s1);

  ascir_op::Data data("data", graph);
  ascir_op::Abs abs("abs");
  abs.x = data.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.sched.loop_axis = 1;
  abs.attr.api.type = ge::ApiType::kAPITypeCompute;
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;
  abs.y.dtype = ge::DT_FLOAT16;
  abs.y.format = ge::FORMAT_ND;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, sym::kSymbolOne};
  *abs.y.vectorized_axis = {z0.id, z1.id};
  *abs.y.vectorized_strides = {s1, sym::kSymbolOne};
  ascir_op::Abs abs1("abs1");
  abs1.x = data.y;

  auto abs_node = graph.FindNode("abs");
  auto abs1_node = graph.FindNode("abs1");
  ASSERT_NE(abs_node, nullptr);
  ASSERT_NE(abs1_node, nullptr);
  ASSERT_TRUE(AscGraph::CopyAscNodeTensorAttr(abs_node, abs1_node));

  std::vector<int64_t> golden_axis{z0.id, z1.id};
  std::vector<Expression> golden_repeats{s0, s1};
  std::vector<Expression> golden_strides{s1, sym::kSymbolOne};

  EXPECT_EQ(abs1_node->attr.sched.axis, golden_axis);
  EXPECT_EQ(abs1_node->attr.sched.loop_axis, 1);
  EXPECT_EQ(abs1_node->attr.api.type, ge::ApiType::kAPITypeCompute);
  EXPECT_EQ(abs1_node->attr.api.compute_type, ge::ComputeType::kComputeElewise);

  EXPECT_EQ(abs1_node->outputs[0].attr.dtype, ge::DT_FLOAT16);
  EXPECT_EQ(abs1_node->outputs[0].attr.axis, golden_axis);
  EXPECT_EQ(abs1_node->outputs[0].attr.repeats, golden_repeats);
  EXPECT_EQ(abs1_node->outputs[0].attr.strides, golden_strides);
  EXPECT_EQ(abs1_node->outputs[0].attr.vectorized_axis, golden_axis);
  EXPECT_EQ(abs1_node->outputs[0].attr.vectorized_strides, golden_strides);
}


TEST_F(UtestAscendCIR, AscGraphAttr_Clone_Success) {
  AscGraphAttr asc_graph_attr;
  constexpr uint32_t kMagicNum = 0x5a5a;
  asc_graph_attr.tiling_key = kMagicNum;
  EXPECT_EQ(asc_graph_attr.type, ge::AscGraphType::kHintGraph);
  asc_graph_attr.type = ge::AscGraphType::kImplGraph;
  auto clone_attr = asc_graph_attr.Clone();
  ASSERT_NE(clone_attr, nullptr);
  auto clone_graph_attr = dynamic_cast<AscGraphAttr *>(clone_attr.get());
  ASSERT_NE(clone_graph_attr, nullptr);
  EXPECT_EQ(clone_graph_attr->tiling_key, kMagicNum);
  EXPECT_EQ(clone_graph_attr->type, ge::AscGraphType::kImplGraph);
}

TEST_F(UtestAscendCIR, AscGraphAttr_Ser_And_Des_Success) {
  AscGraphAttr asc_graph_attr;
  constexpr uint32_t kMagicNum = 0x5a5a;
  asc_graph_attr.tiling_key = kMagicNum;
  EXPECT_EQ(asc_graph_attr.type, ge::AscGraphType::kHintGraph);
  asc_graph_attr.type = ge::AscGraphType::kImplGraph;
  ascendc_ir::proto::AscGraphAttrGroupsDef asc_graph_group;
  EXPECT_EQ(asc_graph_attr.SerializeAttr(asc_graph_group), GRAPH_SUCCESS);
  EXPECT_EQ(asc_graph_group.tiling_key(), asc_graph_attr.tiling_key);
  EXPECT_EQ(asc_graph_group.type(), static_cast<int64_t>(asc_graph_attr.type));
  AscGraphAttr asc_graph_attr2;
  asc_graph_attr2.DeserializeAttr(asc_graph_group);
  EXPECT_EQ(asc_graph_attr2.tiling_key, asc_graph_attr.tiling_key);
  EXPECT_EQ(asc_graph_attr2.type, asc_graph_attr.type);
}

TEST_F(UtestAscendCIR, AscNodeAttr_Clone_Success) {
  AscNodeAttr asc_node_attr;
  auto data_ir_attr = ComGraphMakeUnique<AscDataIrAttrDef>();
  data_ir_attr->SetIndex(10);
  asc_node_attr.ir_attr = std::move(data_ir_attr);
  asc_node_attr.api.type = ApiType::kAPITypeCompute;
  MemAttr mem_attr{1, AllocType::kAllocTypeGlobal, Position::kPositionGM, MemHardware::kMemHardwareGM, {1}, "mem_name", 2};
  asc_node_attr.tmp_buffers = {TmpBuffer{TmpBufDesc{Expression(), 1}, mem_attr}};
  auto clone_attr = asc_node_attr.Clone();
  ASSERT_NE(clone_attr, nullptr);
  auto clone_node_attr = dynamic_cast<AscNodeAttr *>(clone_attr.get());
  ASSERT_NE(clone_node_attr, nullptr);
  EXPECT_EQ(clone_node_attr->api.type, ApiType::kAPITypeCompute);
  EXPECT_NE(clone_node_attr->ir_attr, nullptr);
  int64_t value_get{-1};
  EXPECT_EQ(clone_node_attr->ir_attr->GetAttrValue("index", value_get), GRAPH_SUCCESS);
  EXPECT_EQ(value_get, 10);
  EXPECT_EQ(clone_node_attr->tmp_buffers[0].buf_desc.life_time_axis_id, 1);
  EXPECT_EQ(clone_node_attr->tmp_buffers[0].mem.name, "mem_name");
  EXPECT_EQ(clone_node_attr->tmp_buffers[0].mem.reuse_id, 2);
}

TEST_F(UtestAscendCIR, AscTensorAttr_Clone_Success) {
  AscTensorAttr asc_tensor_attr;
  asc_tensor_attr.mem.alloc_type = AllocType::kAllocTypeL1;
  auto clone_attr = asc_tensor_attr.Clone();
  ASSERT_NE(clone_attr, nullptr);
  auto clone_tensor_attr = dynamic_cast<AscTensorAttr *>(clone_attr.get());
  ASSERT_NE(clone_tensor_attr, nullptr);
  EXPECT_EQ(clone_tensor_attr->mem.alloc_type, AllocType::kAllocTypeL1);
}

TEST_F(UtestAscendCIR, AscTensorAttr_Create_Success) {
  ascir_op::Data data("data0");
  AscTensorAttr asc_tensor_attr = AscTensorAttr::GetTensorAttr(&data, 0);
  asc_tensor_attr.dtype = DT_INT8;
  EXPECT_EQ(data.y.dtype, DT_INT8);
}

namespace ge {
namespace ascir {
inline std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcTmpSizeForStubOp11(const ge::AscNode &node) {
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_descs;
  return tmp_buf_descs;
}
}
}

TEST_F(UtestAscendCIR, CalcAscNodeTmpSize) {
  AscGraph graph("test_graph");
  Expression s0 = graph.CreateSizeVar("s0");
  Expression s1 = graph.CreateSizeVar("s1");
  Axis &s0_axis = graph.CreateAxis("S0", s0);
  Axis &s1_axis = graph.CreateAxis("S1", s1);
  ascir_op::Data data("data", graph);
  data.attr.api.type = ApiType::kAPITypeBuffer;
  data.attr.api.unit = ComputeUnit::kUnitMTE1;
  data.attr.api.compute_type = ComputeType::kComputeLoad; // fake to check
  data.attr.sched.exec_order = 1;
  data.attr.sched.axis = {s0_axis.id, s1_axis.id};
  auto data_node = ge::NodeUtilsEx::GetNodeFromOperator(data);
  data.y.dtype = ge::DT_FLOAT16;
  data.y.format = ge::FORMAT_ND;
  *data.y.axis = {s0_axis.id, s1_axis.id};
  *data.y.repeats = {s0, s1};
  *data.y.strides = {s1, sym::kSymbolOne};

  ascir_op::StubOp10 stubOp10("StubOp10");
  stubOp10.x = data.y;
  auto stubOp10_node = std::static_pointer_cast<const AscNode>(::NodeUtilsEx::GetNodeFromOperator(stubOp10));
  auto tmp_buf_desc_stubOp10 = ge::ascir::CalcAscNodeTmpSize(*stubOp10_node);
  EXPECT_EQ(tmp_buf_desc_stubOp10.size(), 1);
  EXPECT_EQ(tmp_buf_desc_stubOp10[0]->life_time_axis_id, -1);
  EXPECT_EQ(tmp_buf_desc_stubOp10[0]->size, ge::sym::Mul(ge::sym::Mul(Expression(Symbol(2)), s0), s1));

  ascir_op::StubOp11 stubOp11("StubOp11");
  stubOp11.x = data.y;
  auto stubOp11_node = std::static_pointer_cast<const AscNode>(::NodeUtilsEx::GetNodeFromOperator(stubOp11));
  auto tmp_buf_desc_stubOp11 = ge::ascir::CalcAscNodeTmpSize(*stubOp11_node);
  EXPECT_EQ(tmp_buf_desc_stubOp11.size(), 0);
}

TEST_F(UtestAscendCIR, AscNodeAttr_TmpBuffer_Serialize) {
  AscNodeAttr asc_node_attr;
  const TmpBufDesc tmp_buf_desc{Expression(Symbol(1)), 1};
  const MemAttr mem_attr{1, AllocType::kAllocTypeGlobal, Position::kPositionGM, MemHardware::kMemHardwareGM, {1}, "mem_name", 2};
  asc_node_attr.tmp_buffers.emplace_back(TmpBuffer{tmp_buf_desc, mem_attr});
  ascendc_ir::proto::AscNodeAttrGroupsDef asc_node_attr_def;
  asc_node_attr.SerializeAttr(asc_node_attr_def);
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).buf_desc().life_time_axis_id(), 1);
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).buf_desc().size(), Expression(Symbol(1)).Serialize().get());
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().alloc_type(), static_cast<int64_t>(AllocType::kAllocTypeGlobal));
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().position(), static_cast<int64_t>(Position::kPositionGM));
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().hardware(), static_cast<int64_t>(MemHardware::kMemHardwareGM));
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().name(), "mem_name");
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().tensor_id(), 1);
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().buf_ids(0), 1);
  EXPECT_EQ(asc_node_attr_def.tmp_buffers(0).mem().reuse_id(), 2);
}

TEST_F(UtestAscendCIR, AscNodeAttr_TmpBuffer_Deserialize) {
  ascendc_ir::proto::AscNodeAttrGroupsDef asc_node_attr_def;
  auto tmp_buffer = asc_node_attr_def.add_tmp_buffers();
  auto buf_desc = tmp_buffer->mutable_buf_desc();
  buf_desc->set_life_time_axis_id(1);
  buf_desc->set_size(Expression(Symbol(1)).Serialize().get());
  auto mem_attr = tmp_buffer->mutable_mem();
  mem_attr->set_alloc_type(static_cast<int64_t>(AllocType::kAllocTypeGlobal));
  mem_attr->set_position(static_cast<int64_t>(Position::kPositionVecCalc));
  mem_attr->set_hardware(static_cast<int64_t>(MemHardware::kMemHardwareGM));
  mem_attr->set_name("mem_name");
  mem_attr->set_tensor_id(2);
  mem_attr->add_buf_ids(1);
  mem_attr->add_buf_ids(2);
  AscNodeAttr asc_node_attr;
  asc_node_attr.DeserializeAttr(asc_node_attr_def);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].buf_desc.life_time_axis_id, 1);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].buf_desc.size, Expression(Symbol(1)));
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.alloc_type, AllocType::kAllocTypeGlobal);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.position, Position::kPositionVecCalc);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.hardware, MemHardware::kMemHardwareGM);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.name, "mem_name");
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.tensor_id, 2);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.buf_ids[0], 1);
  EXPECT_EQ(asc_node_attr.tmp_buffers[0].mem.buf_ids[1], 2);
}

TEST_F(UtestAscendCIR, AscNodeAttr_Create_invalid) {
  auto invalid_op = Operator();
  auto attr = AscNodeAttr::Create<AscDataIrAttrDef>(invalid_op);
  EXPECT_TRUE(attr == nullptr);
}

TEST_F(UtestAscendCIR, AscTensorAttr_Create_invalid) {
  auto invalid_op = Operator();
  auto attr = AscTensorAttr::GetTensorAttr(&invalid_op, 0);
  EXPECT_EQ(attr.dtype, DT_UNDEFINED);
  OutDataAnchor out_data_anchor(nullptr, -1);
  auto attr2 = AscTensorAttr::GetTensorAttr(&invalid_op, 0);
  EXPECT_EQ(attr2.dtype, DT_UNDEFINED);
}

TEST_F(UtestAscendCIR, AscOutputAttrFormat_invalid) {
  AscOutputAttrFormat asc_output_attr_format(nullptr, UINT32_MAX);
  EXPECT_EQ(asc_output_attr_format, FORMAT_RESERVED);
  asc_output_attr_format = FORMAT_ND;
  EXPECT_EQ(asc_output_attr_format, FORMAT_RESERVED);
  auto op = Operator();
  AscOutputAttrFormat asc_output_attr_format1(&op, UINT32_MAX);
  EXPECT_EQ(asc_output_attr_format1, FORMAT_RESERVED);
  asc_output_attr_format1 = FORMAT_ND;
  EXPECT_EQ(asc_output_attr_format1, FORMAT_RESERVED);
  auto op2 = Operator("stub", "stub");
  AscOutputAttrFormat asc_output_attr_format2(&op2, 1);
  EXPECT_EQ(asc_output_attr_format2, FORMAT_RESERVED);
  asc_output_attr_format2 = FORMAT_ND;
  EXPECT_EQ(asc_output_attr_format2, FORMAT_RESERVED);
}

TEST_F(UtestAscendCIR, AscOutputAttrDataType_invalid) {
  AscOutputAttrDataType asc_output_attr_data_type(nullptr, UINT32_MAX);
  EXPECT_EQ(asc_output_attr_data_type, DT_UNDEFINED);
  asc_output_attr_data_type = DT_INT32;
  EXPECT_EQ(asc_output_attr_data_type, DT_UNDEFINED);
  auto op = Operator();
  AscOutputAttrDataType asc_output_attr_data_type1(&op, UINT32_MAX);
  EXPECT_EQ(asc_output_attr_data_type1, DT_UNDEFINED);
  asc_output_attr_data_type1 = DT_INT32;
  EXPECT_EQ(asc_output_attr_data_type1, DT_UNDEFINED);
  auto op2 = Operator("stub", "stub");
  AscOutputAttrDataType asc_output_attr_data_type2(&op2, 1);
  EXPECT_EQ(asc_output_attr_data_type2, DT_UNDEFINED);
  asc_output_attr_data_type2 = DT_INT32;
  EXPECT_EQ(asc_output_attr_data_type2, DT_UNDEFINED);
}

TEST_F(UtestAscendCIR, CalcAscNodeTmpSizeFunc) {
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class;
  const std::string target_func = "CalcAscNodeTmpSize";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
inline std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcAscNodeTmpSize(const ge::AscNode &node) {
  typedef std::vector<std::unique_ptr<ge::TmpBufDesc>> (*calc_func_ptr) (const AscNode &node);
  static const std::unordered_map<std::string, calc_func_ptr> node_calc_tmp_buff_map = {
    {"Broadcast", &CalcBroadCastTmpSize},
    {"Pad", &CalcPadTmpSize},
    {"Cast", &CalcCastTmpSize},
    {"Rsqrt", &CalcRsqrtTmpSize},
    {"Reciprocal", &CalcDefaultTmpSize},
    {"Erf", &CalcErfTmpSize},
    {"Sign", &CalcSignTmpSize},
    {"Tanh", &CalcTanhTmpSize},
    {"Isnan", &CalcIsnanTmpSize},
    {"IsFinite", &CalcIsFiniteTmpSize},
    {"LogicalNot", &CalcLogicalNotTmpSize},
    {"Max", &CalcReduceTmpSize},
    {"Sum", &CalcReduceTmpSize},
    {"Min", &CalcReduceTmpSize},
    {"Mean", &CalcReduceTmpSize},
    {"Prod", &CalcReduceTmpSize},
    {"Sigmoid", &CalcSigmoidTmpSize},
    {"Any", &CalcReduceTmpSize},
    {"All", &CalcReduceTmpSize},
    {"Sub", &CalcSubTmpSize},
    {"Div", &CalcDivTmpSize},
    {"TrueDiv", &CalcTrueDivTmpSize},
    {"LogicalOr", &CalcLogicalOrTmpSize},
    {"LogicalAnd", &CalcLogicalAndTmpSize},
    {"Pow", &CalcPowTmpSize},
    {"ClipByValue", &CalcClipByValueTmpSize},
    {"Ge", &CalcGeTmpSize},
    {"Eq", &CalcEqTmpSize},
    {"Ne", &CalcNeTmpSize},
    {"Gt", &CalcGtTmpSize},
    {"Le", &CalcLeTmpSize},
    {"Lt", &CalcLtTmpSize},
    {"Concat", &CalcConcatTmpSize},
    {"Select", &CalcSelectTmpSize},
    {"Where", &CalcWhereTmpSize},
    {"BitwiseAnd", &CalcDefaultTmpSize},
    {"Gather", &CalcGatherTmpSize},
    {"FloorDiv", &GetInputDataSizeTmpBuffer},
    {"Gelu", &GetInputDataSizeTmpBuffer},
    {"Axpy", &CalcAxpyTmpSize},
    {"StubOp10", &SameTmpBufSizeWithFirstInput},
    {"StubOp11", &CalcTmpSizeForStubOp11},
  };
  ge::AscNodeAttr attr = node.attr;
  if (node_calc_tmp_buff_map.find(attr.type) != node_calc_tmp_buff_map.end()) {
    return node_calc_tmp_buff_map.at(node.attr.type)(node);
  }
  return std::vector<std::unique_ptr<ge::TmpBufDesc>>();
}
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

TEST_F(UtestAscendCIR, CommonInferDtypeFuncGen) {
  // 分离v35会导致ascir变化，先skip
  GTEST_SKIP();
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class;
  const std::string target_func = "CommonInferDtype";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
inline ge::Status CommonInferDtype(const std::string &type, const std::vector<DataType> &input_dtypes,
                                   std::vector<DataType> &expect_output_dtypes,
                                   [[maybe_unused]]const std::string& npu_arch) {
  using func = ge::Status (*)(const std::vector<DataType> &input_dtypes,
                              std::vector<DataType> &expect_output_dtypes,
                              const std::string& npu_arch);
  static const std::unordered_map<std::string, func> func_table = {
    {"Data", ::ge::ascir_op::Data::InferDataType},
    {"VectorFunc", ::ge::ascir_op::VectorFunc::InferDataType},
    {"Scalar", ::ge::ascir_op::Scalar::InferDataType},
    {"IndexExpr", ::ge::ascir_op::IndexExpr::InferDataType},
    {"Output", ::ge::ascir_op::Output::InferDataType},
    {"Workspace", ::ge::ascir_op::Workspace::InferDataType},
    {"Load", ::ge::ascir_op::Load::InferDataType},
    {"Store", ::ge::ascir_op::Store::InferDataType},
    {"Broadcast", ::ge::ascir_op::Broadcast::InferDataType},
    {"RemovePad", ::ge::ascir_op::RemovePad::InferDataType},
    {"Pad", ::ge::ascir_op::Pad::InferDataType},
    {"Nop", ::ge::ascir_op::Nop::InferDataType},
    {"Cast", ::ge::ascir_op::Cast::InferDataType},
    {"Abs", ::ge::ascir_op::Abs::InferDataType},
    {"Exp", ::ge::ascir_op::Exp::InferDataType},
    {"Ln", ::ge::ascir_op::Ln::InferDataType},
    {"Sqrt", ::ge::ascir_op::Sqrt::InferDataType},
    {"Rsqrt", ::ge::ascir_op::Rsqrt::InferDataType},
    {"Reciprocal", ::ge::ascir_op::Reciprocal::InferDataType},
    {"Erf", ::ge::ascir_op::Erf::InferDataType},
    {"Sign", ::ge::ascir_op::Sign::InferDataType},
    {"Tanh", ::ge::ascir_op::Tanh::InferDataType},
    {"Isnan", ::ge::ascir_op::Isnan::InferDataType},
    {"IsFinite", ::ge::ascir_op::IsFinite::InferDataType},
    {"Relu", ::ge::ascir_op::Relu::InferDataType},
    {"Neg", ::ge::ascir_op::Neg::InferDataType},
    {"LogicalNot", ::ge::ascir_op::LogicalNot::InferDataType},
    {"Max", ::ge::ascir_op::Max::InferDataType},
    {"Sum", ::ge::ascir_op::Sum::InferDataType},
    {"Min", ::ge::ascir_op::Min::InferDataType},
    {"Mean", ::ge::ascir_op::Mean::InferDataType},
    {"Prod", ::ge::ascir_op::Prod::InferDataType},
    {"Sigmoid", ::ge::ascir_op::Sigmoid::InferDataType},
    {"Any", ::ge::ascir_op::Any::InferDataType},
    {"All", ::ge::ascir_op::All::InferDataType},
    {"Add", ::ge::ascir_op::Add::InferDataType},
    {"Sub", ::ge::ascir_op::Sub::InferDataType},
    {"Div", ::ge::ascir_op::Div::InferDataType},
    {"Mul", ::ge::ascir_op::Mul::InferDataType},
    {"Minimum", ::ge::ascir_op::Minimum::InferDataType},
    {"Maximum", ::ge::ascir_op::Maximum::InferDataType},
    {"TrueDiv", ::ge::ascir_op::TrueDiv::InferDataType},
    {"LogicalOr", ::ge::ascir_op::LogicalOr::InferDataType},
    {"LogicalAnd", ::ge::ascir_op::LogicalAnd::InferDataType},
    {"Pow", ::ge::ascir_op::Pow::InferDataType},
    {"ClipByValue", ::ge::ascir_op::ClipByValue::InferDataType},
    {"Ge", ::ge::ascir_op::Ge::InferDataType},
    {"Eq", ::ge::ascir_op::Eq::InferDataType},
    {"Ne", ::ge::ascir_op::Ne::InferDataType},
    {"Gt", ::ge::ascir_op::Gt::InferDataType},
    {"Le", ::ge::ascir_op::Le::InferDataType},
    {"Lt", ::ge::ascir_op::Lt::InferDataType},
    {"Concat", ::ge::ascir_op::Concat::InferDataType},
    {"Select", ::ge::ascir_op::Select::InferDataType},
    {"Where", ::ge::ascir_op::Where::InferDataType},
    {"Ub2ub", ::ge::ascir_op::Ub2ub::InferDataType},
    {"LeakyRelu", ::ge::ascir_op::LeakyRelu::InferDataType},
    {"BitwiseAnd", ::ge::ascir_op::BitwiseAnd::InferDataType},
    {"Gather", ::ge::ascir_op::Gather::InferDataType},
    {"Transpose", ::ge::ascir_op::Transpose::InferDataType},
    {"FlashSoftmax", ::ge::ascir_op::FlashSoftmax::InferDataType},
    {"FloorDiv", ::ge::ascir_op::FloorDiv::InferDataType},
    {"Gelu", ::ge::ascir_op::Gelu::InferDataType},
    {"Axpy", ::ge::ascir_op::Axpy::InferDataType},
    {"MatMul", ::ge::ascir_op::MatMul::InferDataType},
    {"MatMulBias", ::ge::ascir_op::MatMulBias::InferDataType},
    {"MatMulOffset", ::ge::ascir_op::MatMulOffset::InferDataType},
    {"MatMulOffsetBias", ::ge::ascir_op::MatMulOffsetBias::InferDataType},
    {"BatchMatMul", ::ge::ascir_op::BatchMatMul::InferDataType},
    {"BatchMatMulBias", ::ge::ascir_op::BatchMatMulBias::InferDataType},
    {"BatchMatMulOffset", ::ge::ascir_op::BatchMatMulOffset::InferDataType},
    {"BatchMatMulOffsetBias", ::ge::ascir_op::BatchMatMulOffsetBias::InferDataType},
    {"Split", ::ge::ascir_op::Split::InferDataType},
    {"Constant", ::ge::ascir_op::Constant::InferDataType},
    {"TbufData", ::ge::ascir_op::TbufData::InferDataType},
    {"LoadStub", ::ge::ascir_op::LoadStub::InferDataType},
    {"StoreStub", ::ge::ascir_op::StoreStub::InferDataType},
    {"WorkspaceWithInput", ::ge::ascir_op::WorkspaceWithInput::InferDataType},
    {"AbsStub", ::ge::ascir_op::AbsStub::InferDataType},
    {"GT", ::ge::ascir_op::GT::InferDataType},
    {"Muls", ::ge::ascir_op::Muls::InferDataType},
    {"Dropout", ::ge::ascir_op::Dropout::InferDataType},
    {"CalcMean", ::ge::ascir_op::CalcMean::InferDataType},
    {"CalcMeanSlice", ::ge::ascir_op::CalcMeanSlice::InferDataType},
    {"CalcRstd", ::ge::ascir_op::CalcRstd::InferDataType},
    {"CalcRstdSlice", ::ge::ascir_op::CalcRstdSlice::InferDataType},
    {"VFWelfordPart1Update", ::ge::ascir_op::VFWelfordPart1Update::InferDataType},
    {"VFWelfordPart1Finalize", ::ge::ascir_op::VFWelfordPart1Finalize::InferDataType},
    {"VFCalcYWelford", ::ge::ascir_op::VFCalcYWelford::InferDataType},
    {"VectorFunction", ::ge::ascir_op::VectorFunction::InferDataType},
    {"FakeOpA", ::ge::ascir_op::FakeOpA::InferDataType},
    {"CalcY", ::ge::ascir_op::CalcY::InferDataType},
    {"CalcMeanStub", ::ge::ascir_op::CalcMeanStub::InferDataType},
    {"StubOp1", ::ge::ascir_op::StubOp1::InferDataType},
    {"StubOp2", ::ge::ascir_op::StubOp2::InferDataType},
    {"StubOp2New", ::ge::ascir_op::StubOp2New::InferDataType},
    {"StubOp3", ::ge::ascir_op::StubOp3::InferDataType},
    {"StubOp3New", ::ge::ascir_op::StubOp3New::InferDataType},
    {"StubOp4", ::ge::ascir_op::StubOp4::InferDataType},
    {"StubOp4New", ::ge::ascir_op::StubOp4New::InferDataType},
    {"StubOp5", ::ge::ascir_op::StubOp5::InferDataType},
    {"StubOp5New", ::ge::ascir_op::StubOp5New::InferDataType},
    {"StubOp6", ::ge::ascir_op::StubOp6::InferDataType},
    {"StubOp6New", ::ge::ascir_op::StubOp6New::InferDataType},
    {"StubOp7", ::ge::ascir_op::StubOp7::InferDataType},
    {"StubOp7New", ::ge::ascir_op::StubOp7New::InferDataType},
    {"StubOp8", ::ge::ascir_op::StubOp8::InferDataType},
    {"StubOp8New", ::ge::ascir_op::StubOp8New::InferDataType},
    {"StubOp9", ::ge::ascir_op::StubOp9::InferDataType},
    {"StubOp9New", ::ge::ascir_op::StubOp9New::InferDataType},
    {"StubOp10", ::ge::ascir_op::StubOp10::InferDataType},
    {"StubOp11", ::ge::ascir_op::StubOp11::InferDataType},
    {"StubRemovePad", ::ge::ascir_op::StubRemovePad::InferDataType},
  };
  const auto &iter = func_table.find(type);
  if (iter != func_table.end()) {
    return iter->second(input_dtypes, expect_output_dtypes, npu_arch);
  }
  GELOGW("Node type %s is not supported to infer for now!", type.c_str());
  return ge::FAILED;
}
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

TEST_F(UtestAscendCIR, CommonInferDtypeWithNoCheckFuncGen) {
  // 分离v35会导致ascir变化，先skip
  GTEST_SKIP();
  const std::string file_path = std::string(CMAKE_BINARY_DIR) + "/tests/depends/aihacb_autofusion/ascir_ops.h";
  const std::string target_class;
  const std::string target_func = "CommonInferDtypeWithNoCheck";

  auto[sig, actual_code] = CodeExtractor::ExtractFunction(file_path, target_class, target_func);

  const std::string expected_code = R"EXPECT(
inline ge::Status CommonInferDtypeWithNoCheck(const std::string &type, const std::vector<DataType> &input_dtypes,
                                   std::vector<DataType> &expect_output_dtypes,
                                   [[maybe_unused]]const std::string& npu_arch) {
  using func = ge::Status (*)(const std::vector<DataType> &input_dtypes,
                              std::vector<DataType> &expect_output_dtypes,
                              const std::string& npu_arch);
  static const std::unordered_map<std::string, func> func_table = {
    {"Data", ::ge::ascir_op::Data::InferDataTypeWithNoCheck},
    {"VectorFunc", ::ge::ascir_op::VectorFunc::InferDataTypeWithNoCheck},
    {"Scalar", ::ge::ascir_op::Scalar::InferDataTypeWithNoCheck},
    {"IndexExpr", ::ge::ascir_op::IndexExpr::InferDataTypeWithNoCheck},
    {"Output", ::ge::ascir_op::Output::InferDataTypeWithNoCheck},
    {"Workspace", ::ge::ascir_op::Workspace::InferDataTypeWithNoCheck},
    {"Load", ::ge::ascir_op::Load::InferDataTypeWithNoCheck},
    {"Store", ::ge::ascir_op::Store::InferDataTypeWithNoCheck},
    {"Broadcast", ::ge::ascir_op::Broadcast::InferDataTypeWithNoCheck},
    {"RemovePad", ::ge::ascir_op::RemovePad::InferDataTypeWithNoCheck},
    {"Pad", ::ge::ascir_op::Pad::InferDataTypeWithNoCheck},
    {"Nop", ::ge::ascir_op::Nop::InferDataTypeWithNoCheck},
    {"Cast", ::ge::ascir_op::Cast::InferDataTypeWithNoCheck},
    {"Abs", ::ge::ascir_op::Abs::InferDataTypeWithNoCheck},
    {"Exp", ::ge::ascir_op::Exp::InferDataTypeWithNoCheck},
    {"Ln", ::ge::ascir_op::Ln::InferDataTypeWithNoCheck},
    {"Sqrt", ::ge::ascir_op::Sqrt::InferDataTypeWithNoCheck},
    {"Rsqrt", ::ge::ascir_op::Rsqrt::InferDataTypeWithNoCheck},
    {"Reciprocal", ::ge::ascir_op::Reciprocal::InferDataTypeWithNoCheck},
    {"Erf", ::ge::ascir_op::Erf::InferDataTypeWithNoCheck},
    {"Sign", ::ge::ascir_op::Sign::InferDataTypeWithNoCheck},
    {"Tanh", ::ge::ascir_op::Tanh::InferDataTypeWithNoCheck},
    {"Isnan", ::ge::ascir_op::Isnan::InferDataTypeWithNoCheck},
    {"IsFinite", ::ge::ascir_op::IsFinite::InferDataTypeWithNoCheck},
    {"Relu", ::ge::ascir_op::Relu::InferDataTypeWithNoCheck},
    {"Neg", ::ge::ascir_op::Neg::InferDataTypeWithNoCheck},
    {"LogicalNot", ::ge::ascir_op::LogicalNot::InferDataTypeWithNoCheck},
    {"Max", ::ge::ascir_op::Max::InferDataTypeWithNoCheck},
    {"Sum", ::ge::ascir_op::Sum::InferDataTypeWithNoCheck},
    {"Min", ::ge::ascir_op::Min::InferDataTypeWithNoCheck},
    {"Mean", ::ge::ascir_op::Mean::InferDataTypeWithNoCheck},
    {"Prod", ::ge::ascir_op::Prod::InferDataTypeWithNoCheck},
    {"Sigmoid", ::ge::ascir_op::Sigmoid::InferDataTypeWithNoCheck},
    {"Any", ::ge::ascir_op::Any::InferDataTypeWithNoCheck},
    {"All", ::ge::ascir_op::All::InferDataTypeWithNoCheck},
    {"Add", ::ge::ascir_op::Add::InferDataTypeWithNoCheck},
    {"Sub", ::ge::ascir_op::Sub::InferDataTypeWithNoCheck},
    {"Div", ::ge::ascir_op::Div::InferDataTypeWithNoCheck},
    {"Mul", ::ge::ascir_op::Mul::InferDataTypeWithNoCheck},
    {"Minimum", ::ge::ascir_op::Minimum::InferDataTypeWithNoCheck},
    {"Maximum", ::ge::ascir_op::Maximum::InferDataTypeWithNoCheck},
    {"TrueDiv", ::ge::ascir_op::TrueDiv::InferDataTypeWithNoCheck},
    {"LogicalOr", ::ge::ascir_op::LogicalOr::InferDataTypeWithNoCheck},
    {"LogicalAnd", ::ge::ascir_op::LogicalAnd::InferDataTypeWithNoCheck},
    {"Pow", ::ge::ascir_op::Pow::InferDataTypeWithNoCheck},
    {"ClipByValue", ::ge::ascir_op::ClipByValue::InferDataTypeWithNoCheck},
    {"Ge", ::ge::ascir_op::Ge::InferDataTypeWithNoCheck},
    {"Eq", ::ge::ascir_op::Eq::InferDataTypeWithNoCheck},
    {"Ne", ::ge::ascir_op::Ne::InferDataTypeWithNoCheck},
    {"Gt", ::ge::ascir_op::Gt::InferDataTypeWithNoCheck},
    {"Le", ::ge::ascir_op::Le::InferDataTypeWithNoCheck},
    {"Lt", ::ge::ascir_op::Lt::InferDataTypeWithNoCheck},
    {"Concat", ::ge::ascir_op::Concat::InferDataTypeWithNoCheck},
    {"Select", ::ge::ascir_op::Select::InferDataTypeWithNoCheck},
    {"Where", ::ge::ascir_op::Where::InferDataTypeWithNoCheck},
    {"Ub2ub", ::ge::ascir_op::Ub2ub::InferDataTypeWithNoCheck},
    {"LeakyRelu", ::ge::ascir_op::LeakyRelu::InferDataTypeWithNoCheck},
    {"BitwiseAnd", ::ge::ascir_op::BitwiseAnd::InferDataTypeWithNoCheck},
    {"Gather", ::ge::ascir_op::Gather::InferDataTypeWithNoCheck},
    {"Transpose", ::ge::ascir_op::Transpose::InferDataTypeWithNoCheck},
    {"FlashSoftmax", ::ge::ascir_op::FlashSoftmax::InferDataTypeWithNoCheck},
    {"FloorDiv", ::ge::ascir_op::FloorDiv::InferDataTypeWithNoCheck},
    {"Gelu", ::ge::ascir_op::Gelu::InferDataTypeWithNoCheck},
    {"Axpy", ::ge::ascir_op::Axpy::InferDataTypeWithNoCheck},
    {"MatMul", ::ge::ascir_op::MatMul::InferDataTypeWithNoCheck},
    {"MatMulBias", ::ge::ascir_op::MatMulBias::InferDataTypeWithNoCheck},
    {"MatMulOffset", ::ge::ascir_op::MatMulOffset::InferDataTypeWithNoCheck},
    {"MatMulOffsetBias", ::ge::ascir_op::MatMulOffsetBias::InferDataTypeWithNoCheck},
    {"BatchMatMul", ::ge::ascir_op::BatchMatMul::InferDataTypeWithNoCheck},
    {"BatchMatMulBias", ::ge::ascir_op::BatchMatMulBias::InferDataTypeWithNoCheck},
    {"BatchMatMulOffset", ::ge::ascir_op::BatchMatMulOffset::InferDataTypeWithNoCheck},
    {"BatchMatMulOffsetBias", ::ge::ascir_op::BatchMatMulOffsetBias::InferDataTypeWithNoCheck},
    {"Split", ::ge::ascir_op::Split::InferDataTypeWithNoCheck},
    {"Constant", ::ge::ascir_op::Constant::InferDataTypeWithNoCheck},
    {"TbufData", ::ge::ascir_op::TbufData::InferDataTypeWithNoCheck},
    {"LoadStub", ::ge::ascir_op::LoadStub::InferDataTypeWithNoCheck},
    {"StoreStub", ::ge::ascir_op::StoreStub::InferDataTypeWithNoCheck},
    {"WorkspaceWithInput", ::ge::ascir_op::WorkspaceWithInput::InferDataTypeWithNoCheck},
    {"AbsStub", ::ge::ascir_op::AbsStub::InferDataTypeWithNoCheck},
    {"GT", ::ge::ascir_op::GT::InferDataTypeWithNoCheck},
    {"Muls", ::ge::ascir_op::Muls::InferDataTypeWithNoCheck},
    {"Dropout", ::ge::ascir_op::Dropout::InferDataTypeWithNoCheck},
    {"CalcMean", ::ge::ascir_op::CalcMean::InferDataTypeWithNoCheck},
    {"CalcMeanSlice", ::ge::ascir_op::CalcMeanSlice::InferDataTypeWithNoCheck},
    {"CalcRstd", ::ge::ascir_op::CalcRstd::InferDataTypeWithNoCheck},
    {"CalcRstdSlice", ::ge::ascir_op::CalcRstdSlice::InferDataTypeWithNoCheck},
    {"VFWelfordPart1Update", ::ge::ascir_op::VFWelfordPart1Update::InferDataTypeWithNoCheck},
    {"VFWelfordPart1Finalize", ::ge::ascir_op::VFWelfordPart1Finalize::InferDataTypeWithNoCheck},
    {"VFCalcYWelford", ::ge::ascir_op::VFCalcYWelford::InferDataTypeWithNoCheck},
    {"VectorFunction", ::ge::ascir_op::VectorFunction::InferDataTypeWithNoCheck},
    {"FakeOpA", ::ge::ascir_op::FakeOpA::InferDataTypeWithNoCheck},
    {"CalcY", ::ge::ascir_op::CalcY::InferDataTypeWithNoCheck},
    {"CalcMeanStub", ::ge::ascir_op::CalcMeanStub::InferDataTypeWithNoCheck},
    {"StubOp1", ::ge::ascir_op::StubOp1::InferDataTypeWithNoCheck},
    {"StubOp2", ::ge::ascir_op::StubOp2::InferDataTypeWithNoCheck},
    {"StubOp2New", ::ge::ascir_op::StubOp2New::InferDataTypeWithNoCheck},
    {"StubOp3", ::ge::ascir_op::StubOp3::InferDataTypeWithNoCheck},
    {"StubOp3New", ::ge::ascir_op::StubOp3New::InferDataTypeWithNoCheck},
    {"StubOp4", ::ge::ascir_op::StubOp4::InferDataTypeWithNoCheck},
    {"StubOp4New", ::ge::ascir_op::StubOp4New::InferDataTypeWithNoCheck},
    {"StubOp5", ::ge::ascir_op::StubOp5::InferDataTypeWithNoCheck},
    {"StubOp5New", ::ge::ascir_op::StubOp5New::InferDataTypeWithNoCheck},
    {"StubOp6", ::ge::ascir_op::StubOp6::InferDataTypeWithNoCheck},
    {"StubOp6New", ::ge::ascir_op::StubOp6New::InferDataTypeWithNoCheck},
    {"StubOp7", ::ge::ascir_op::StubOp7::InferDataTypeWithNoCheck},
    {"StubOp7New", ::ge::ascir_op::StubOp7New::InferDataTypeWithNoCheck},
    {"StubOp8", ::ge::ascir_op::StubOp8::InferDataTypeWithNoCheck},
    {"StubOp8New", ::ge::ascir_op::StubOp8New::InferDataTypeWithNoCheck},
    {"StubOp9", ::ge::ascir_op::StubOp9::InferDataTypeWithNoCheck},
    {"StubOp9New", ::ge::ascir_op::StubOp9New::InferDataTypeWithNoCheck},
    {"StubOp10", ::ge::ascir_op::StubOp10::InferDataTypeWithNoCheck},
    {"StubOp11", ::ge::ascir_op::StubOp11::InferDataTypeWithNoCheck},
    {"StubRemovePad", ::ge::ascir_op::StubRemovePad::InferDataTypeWithNoCheck},
  };
  const auto &iter = func_table.find(type);
  if (iter != func_table.end()) {
    return iter->second(input_dtypes, expect_output_dtypes, npu_arch);
  }
  GELOGW("Node type %s is not supported to infer for now!", type.c_str());
  return ge::FAILED;
}
)EXPECT";

  auto Normalize = [](const std::string &code) {
    std::string str = code;
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  };

  EXPECT_EQ(Normalize(actual_code), Normalize(expected_code))
            << "Actual code:\n" << actual_code << "\nExpected:\n" << expected_code;
}

// 正常场景已经在OpDtypeInfer类中校验，这个用例校验异常场景
TEST_F(UtestAscendCIR, CommonInferDtypeFunc_invalid_case) {
  std::vector<ge::DataType> outputs;
  EXPECT_EQ(ascir::CommonInferDtype("not_support_op", {}, outputs, "socv1"), ge::FAILED);
}
TEST_F(UtestAscendCIR, DataCopyConstructor) {
  AscGraph graph("test_graph");
  std::vector<Data> data_ops;
  data_ops.reserve(2);
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < 2; ++i) {
    std::string name = "x" + std::to_string(i);
    auto x_op = Data(name.c_str(), graph);
    x_op.y.dtype = ge::DT_FLOAT;
    x_op.ir_attr.SetIndex(static_cast<int64_t>(i));
    data_ops.push_back(x_op);
    outputs.push_back(data_ops[i].y);
  }
  int64_t data_ops_0_index, data_ops_1_index;
  data_ops[0].ir_attr.GetIndex(data_ops_0_index);
  data_ops[1].ir_attr.GetIndex(data_ops_1_index);
  EXPECT_EQ(data_ops[0].y.dtype, ge::DT_FLOAT);
  EXPECT_EQ(data_ops[1].y.dtype, ge::DT_FLOAT);
  EXPECT_EQ(data_ops_0_index, 0);
  EXPECT_EQ(data_ops_1_index, 1);

  data_ops[0].y.dtype = ge::DT_FLOAT16;
  data_ops[0].ir_attr.SetIndex(1);
  data_ops[1].y.dtype = ge::DT_INT16;
  data_ops[1].ir_attr.SetIndex(0);

  data_ops[0].ir_attr.GetIndex(data_ops_0_index);
  data_ops[1].ir_attr.GetIndex(data_ops_1_index);
  EXPECT_EQ(data_ops[0].y.dtype, ge::DT_FLOAT16);
  EXPECT_EQ(data_ops[1].y.dtype, ge::DT_INT16);
  EXPECT_EQ(data_ops_0_index, 1);
  EXPECT_EQ(data_ops_1_index, 0);
}

TEST_F(UtestAscendCIR, OutputCopyConstructor) {
  AscGraph graph("test_graph");
  std::vector<Output> output_ops;
  output_ops.reserve(2);
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < 2; ++i) {
    std::string name = "y" + std::to_string(i);
    auto y_op = Output(name.c_str());
    y_op.y.dtype = ge::DT_FLOAT;
    output_ops.push_back(y_op);
    outputs.push_back(output_ops[i].y);
  }
  EXPECT_EQ(output_ops[0].y.dtype, ge::DT_FLOAT);
  EXPECT_EQ(output_ops[1].y.dtype, ge::DT_FLOAT);

  output_ops[0].y.dtype = ge::DT_FLOAT16;
  output_ops[1].y.dtype = ge::DT_INT16;

  EXPECT_EQ(output_ops[0].y.dtype, ge::DT_FLOAT16);
  EXPECT_EQ(output_ops[1].y.dtype, ge::DT_INT16);
}

TEST_F(UtestAscendCIR, AscOpDynamicInputVectorConstructor) {
  AscGraph graph("test_graph");
  std::vector<Data> data_ops;
  data_ops.reserve(2);
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < 2; ++i) {
    std::string name = "x" + std::to_string(i);
    auto x_op = Data(name.c_str(), graph);
    x_op.y.dtype = ge::DT_FLOAT;
    x_op.ir_attr.SetIndex(static_cast<int64_t>(i));
    data_ops.push_back(x_op);
    outputs.push_back(data_ops[i].y);
  }
  Concat concat_op("concat");
  concat_op.x = outputs;
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(concat_op);
  const std::string name = "x";
  std::vector<int32_t> indexes;
  op_desc->GetDynamicInputIndexesByName(name, indexes);
  EXPECT_EQ(indexes.size(), 2);
  EXPECT_EQ(indexes[0], static_cast<int64_t>(0));
  EXPECT_EQ(indexes[1], static_cast<int64_t>(1));
  auto concat_node = ge::NodeUtilsEx::GetNodeFromOperator(concat_op);
  auto input_nodes = concat_node->GetInNodesPtr();
  EXPECT_EQ(input_nodes.size(), 2);
  EXPECT_EQ(input_nodes[0]->GetName(), "x0");
  EXPECT_EQ(input_nodes[1]->GetName(), "x1");
}

TEST_F(UtestAscendCIR, AscOpDynamicInputAndOutputToDynamicInput) {
  AscGraph graph("test_graph");
  std::vector<Data> data_ops;
  data_ops.reserve(2);
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < 2; ++i) {
    std::string name = "x" + std::to_string(i);
    auto x_op = Data(name.c_str(), graph);
    x_op.y.dtype = ge::DT_FLOAT;
    x_op.ir_attr.SetIndex(static_cast<int64_t>(i));
    data_ops.push_back(x_op);
    outputs.push_back(data_ops[i].y);
  }

  VectorFunction vf_op("vf");

  // 指明有两个输出
  vf_op.InstanceOutputy(2);

  vf_op.x = outputs;
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(vf_op);
  const std::string name = "x";
  std::vector<int32_t> indexes;
  op_desc->GetDynamicInputIndexesByName(name, indexes);
  EXPECT_EQ(indexes.size(), 2);
  EXPECT_EQ(indexes[0], static_cast<int64_t>(0));
  EXPECT_EQ(indexes[1], static_cast<int64_t>(1));
  auto concat_node = ge::NodeUtilsEx::GetNodeFromOperator(vf_op);
  auto input_nodes = concat_node->GetInNodesPtr();
  EXPECT_EQ(input_nodes.size(), 2);
  EXPECT_EQ(input_nodes[0]->GetName(), "x0");
  EXPECT_EQ(input_nodes[1]->GetName(), "x1");

  {
    Concat concat_op("concat");
    concat_op.x = vf_op.y;
    const auto op_desc2 = ge::OpDescUtils::GetOpDescFromOperator(concat_op);
    const std::string name2 = "x";
    std::vector<int32_t> indexes;
    op_desc2->GetDynamicInputIndexesByName(name2, indexes);
    EXPECT_EQ(indexes.size(), 2);
    EXPECT_EQ(indexes[0], static_cast<int64_t>(0));
    EXPECT_EQ(indexes[1], static_cast<int64_t>(1));
    auto concat_node = ge::NodeUtilsEx::GetNodeFromOperator(concat_op);
    auto input_nodes = concat_node->GetInNodesPtr();
    EXPECT_EQ(input_nodes.size(), 2);
    EXPECT_EQ(input_nodes[0]->GetName(), "vf");
    EXPECT_EQ(input_nodes[1]->GetName(), "vf");
  }
}

TEST_F(UtestAscendCIR, AscOpDynamicInputAndOutputToNonDynamicInput) {
  AscGraph graph("test_graph");
  std::vector<Data> data_ops;
  data_ops.reserve(2);
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < 2; ++i) {
    std::string name = "x" + std::to_string(i);
    auto x_op = Data(name.c_str(), graph);
    x_op.y.dtype = ge::DT_FLOAT;
    x_op.ir_attr.SetIndex(static_cast<int64_t>(i));
    data_ops.push_back(x_op);
    outputs.push_back(data_ops[i].y);
  }

  VectorFunction vf_op("vf");
  // 指明有两个输出
  vf_op.InstanceOutputy(2);

  vf_op.x = outputs;
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(vf_op);
  const std::string name = "x";
  std::vector<int32_t> indexes;
  op_desc->GetDynamicInputIndexesByName(name, indexes);
  EXPECT_EQ(indexes.size(), 2);
  EXPECT_EQ(indexes[0], static_cast<int64_t>(0));
  EXPECT_EQ(indexes[1], static_cast<int64_t>(1));
  auto concat_node = ge::NodeUtilsEx::GetNodeFromOperator(vf_op);
  auto input_nodes = concat_node->GetInNodesPtr();
  EXPECT_EQ(input_nodes.size(), 2);
  EXPECT_EQ(input_nodes[0]->GetName(), "x0");
  EXPECT_EQ(input_nodes[1]->GetName(), "x1");

  {
    Add add1_op("add1");
    add1_op.x1 = vf_op.y[0];
    add1_op.x2 = vf_op.y[1];
    const auto op_desc2 = ge::OpDescUtils::GetOpDescFromOperator(add1_op);
    auto add_node = ge::NodeUtilsEx::GetNodeFromOperator(add1_op);
    auto input_nodes = add_node->GetInNodesPtr();
    EXPECT_EQ(input_nodes.size(), 2);
    EXPECT_EQ(input_nodes[0]->GetName(), "vf");
    EXPECT_EQ(input_nodes[1]->GetName(), "vf");
  }

  {
    Add add2_op("add2");
    add2_op.x1 = vf_op.y[0];
    const auto op_desc2 = ge::OpDescUtils::GetOpDescFromOperator(add2_op);
    auto add_node = ge::NodeUtilsEx::GetNodeFromOperator(add2_op);
    auto input_nodes = add_node->GetInNodesPtr();
    EXPECT_EQ(input_nodes.size(), 1);
    EXPECT_EQ(input_nodes[0]->GetName(), "vf");
  }
}

TEST_F(UtestAscendCIR, RegisterTilingData) {
  auto ir_defs = ge::ascir::AscirRegistry::GetInstance().GetAll();
  EXPECT_NE(ir_defs.find("StubTilingData"), ir_defs.end());
  EXPECT_EQ(ir_defs["StubTilingData"].GetApiTilingDataName(), "StubTilingData");
}

TEST_F(UtestAscendCIR, AscirRegisterImpTest) {
  class AscIrAttStub : public ge::ascir::AscIrAtt {
    virtual void *GetApiPerf() const {
      return nullptr;
    }

    virtual void *GetMicroApiPerf() const {
      return nullptr;
    }
    virtual void *GetAscendCApiPerfTable() const {
      return nullptr;
    }
  };
  class AscIrCodegenStub : public ge::ascir::AscIrCodegen {
    public:
    virtual bool IsVectorFunctionSupported(const ge::AscNode &node) const {
      return true;
    }
    bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
      return false;
    }
    bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
      return true;
    }

    bool IsInplaceSupported(const ge::AscNode &node) const override {
      return true;
    }

    bool IsBrcInlineSupported(const ge::AscNode &node) const override {
      return true;
    }
  };
  ge::ascir::AscirRegister reg_test;
  reg_test.Impl(
      {"v1", "v2", "v3"},
      {ge::ascir::AscIrImplCreator<AscIrAttStub>(),
       ge::ascir::AscIrImplCreator<ge::ascir::AscIrCodegen>(),
       {{"T1", OrderedTensorTypeList{DT_INT8, DT_INT16}}, {"T2", OrderedTensorTypeList{DT_UINT8, DT_INT16}}}});

  EXPECT_EQ(reg_test.GetSocImplSize(), 3);

  {
    REG_ASC_IR(StubAbs).Input("x", "T").Output("y", "T").Impl({"910v1"},
                                                              {ge::ascir::AscIrImplCreator<AscIrAttStub>(),
                                                               ge::ascir::AscIrImplCreator<ge::ascir::AscIrCodegen>(),
                                                               {{"T", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT}}}});

    auto codegen_impl = ge::ascir::AscirRegistry::GetInstance().GetIrCodegenImpl("910v1", "StubAbs");
    auto att_impl = ge::ascir::AscirRegistry::GetInstance().GetIrAttImpl("910v1", "StubAbs");
    EXPECT_NE(att_impl, nullptr);
    EXPECT_NE(codegen_impl, nullptr);
    AscGraph graph_normal("graph_normal");
    Data x1("x1", graph_normal);
    for (const auto &node : graph_normal.GetAllNodes()) {
      EXPECT_EQ(codegen_impl->IsVectorFunctionSupported(*node), false);
    }
  }

  {
    REG_ASC_IR(StubAbs2).Input("x", "T").Output("y", "T").Impl({"910v1"},
                                                               {ge::ascir::AscIrImplCreator<AscIrAttStub>(),
                                                                ge::ascir::AscIrImplCreator<AscIrCodegenStub>(),
                                                                {{"T", TensorType{DT_FLOAT16, DT_FLOAT}}}});
    auto codegen_impl = ge::ascir::AscirRegistry::GetInstance().GetIrCodegenImpl("910v1", "StubAbs2");
    auto att_impl = ge::ascir::AscirRegistry::GetInstance().GetIrAttImpl("910v1", "StubAbs2");
    EXPECT_NE(att_impl, nullptr);
    EXPECT_NE(codegen_impl, nullptr);
    AscGraph graph_normal("graph_normal");
    Data x1("x1", graph_normal);
    for (const auto &node : graph_normal.GetAllNodes()) {
      EXPECT_EQ(codegen_impl->IsVectorFunctionSupported(*node), true);
    }
  }
  {
    REG_ASC_IR(StubAbs3).Input("x", "T").Output("y", "T");
    auto codegen_impl = ge::ascir::AscirRegistry::GetInstance().GetIrCodegenImpl("910v1", "StubAbs3");
    auto att_impl = ge::ascir::AscirRegistry::GetInstance().GetIrAttImpl("910v1", "StubAbs3");
    EXPECT_EQ(att_impl, nullptr);
    EXPECT_EQ(codegen_impl, nullptr);
  }
  {
    REG_ASC_IR(StubAdd2).Input("x1", "T").Input("x2", "T").Output("y", "T").Impl({"910v1"},
                                                              {ge::ascir::AscIrImplCreator<AscIrAttStub>(),
                                                               ge::ascir::AscIrImplCreator<AscIrCodegenStub>(),
                                                               {{"T", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT}}}});
    auto codegen_impl = ge::ascir::AscirRegistry::GetInstance().GetIrCodegenImpl("910v1", "StubAdd2");
    EXPECT_NE(codegen_impl, nullptr);
    std::vector<bool> is_scalar_list = {false, true};
    EXPECT_EQ(codegen_impl->IsScalarInputSupported(is_scalar_list), false);
    EXPECT_EQ(codegen_impl->IsScalarInputSupportedIfExchangeInputs(is_scalar_list), true);
    AscGraph graph_normal("graph_normal");
    Data x1("x1", graph_normal);
    for (const auto &node : graph_normal.GetAllNodes()) {
      EXPECT_EQ(codegen_impl->IsInplaceSupported(*node), true);
      EXPECT_EQ(codegen_impl->IsBrcInlineSupported(*node), true);
    }
  }
}