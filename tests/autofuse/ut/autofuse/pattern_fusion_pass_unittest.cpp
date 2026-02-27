/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <gtest/gtest.h>

#include "ge_tensor.h"
#include "operator_reg.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"

#include "all_ops_cpp.h"
#include "graph/compute_graph.h"
#include "framework/omg/parser/parser_types.h"
#include "esb_graph.h"
#include "esb_funcs_cpp.h"
#include "graph_utils.h"
#include "graph_utils_ex.h"
#include "lowering/lowerings.h"
#include "autofuse_frame/autofuse_frames.h"
#include "op_creator_register.h"
#include "pattern_fusion/pattern_fusion.h"
#include "pattern_fusion/transpose_with_broadcast_eliminate_pass.h"
#include "pattern_fusion/slice_forward_fusion_pass.h"
#include "pattern_fusion/cast_remove_pass.h"
#include "pattern_fusion/pattern_fusion_utils.h"
#include "utils/autofuse_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge {
class PatternFusionPassTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(0, 3, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    dlog_setlevel(0, 3, 0);
  }
  std::unique_ptr<es::Graph> es_graph_;

  template <typename T>
  static es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
    auto result = es::FileConstant(graph, dims, dtype);
    GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
    GeTensorPtr tensor =
        std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
    AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
    result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
    return result;
  }
};

TEST_F(PatternFusionPassTest, CastRemoveSameDtype) {
  // 测试相同输入输出dtype的Cast消除
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("Data");
  auto cast = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("cast");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(cast)->NODE(relu)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  bool changed = false;
  EXPECT_EQ(CastRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(graph->FindNode("cast") == nullptr);
  EXPECT_EQ(graph->FindNode("relu")->GetInDataNodes().at(0)->GetName(), "Data");
}

TEST_F(PatternFusionPassTest, CastRemoveConsecutiveCast) {
  // 测试连续Cast消除: A -> Cast1 -> Cast2 -> B
  // 其中 Cast1: float->float16, Cast2: float16->float
  // 消除后直接连接: A -> B
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("Data");
  auto cast1 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(1).Build("cast1");
  auto cast2 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("cast2");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(cast1)->NODE(cast2)->NODE(relu)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  bool changed = false;
  EXPECT_EQ(CastRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(graph->FindNode("cast1") == nullptr);
  EXPECT_TRUE(graph->FindNode("cast2") == nullptr);
  EXPECT_EQ(graph->FindNode("relu")->GetInDataNodes().at(0)->GetName(), "Data");
}

TEST_F(PatternFusionPassTest, CastRemoveNotSameDtype) {
  // 测试不同dtype的Cast不消除
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("Data");
  auto cast = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(1).Build("cast");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(cast)->NODE(relu)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  auto cast_node = graph->FindNode("cast");
  ASSERT_NE(cast_node, nullptr);
  cast_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  cast_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT);
  cast_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  cast_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT16);

  bool changed = false;
  EXPECT_EQ(CastRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_TRUE(graph->FindNode("cast") != nullptr);
}

TEST_F(PatternFusionPassTest, CastRemoveMultipleOutputs) {
  // 测试Cast有多个输出时不消除连续Cast
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("Data");
  auto cast1 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(2).Build("cast1");
  auto cast2 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("cast2");
  auto relu1 = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("relu1");
  auto relu2 = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(1).Build("relu2");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(cast1)->NODE(cast2)->NODE(relu1)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(cast1)->NODE(relu2)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  // cast1: float -> float16 (有多个输出，不应该被消除)
  // cast2: float16 -> float (输入是cast1的输出float16，输出是float)
  auto cast1_node = graph->FindNode("cast1");
  ASSERT_NE(cast1_node, nullptr);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT16);

  auto cast2_node = graph->FindNode("cast2");
  ASSERT_NE(cast2_node, nullptr);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);

  bool changed = false;
  EXPECT_EQ(CastRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_TRUE(graph->FindNode("cast1") != nullptr);
  EXPECT_TRUE(graph->FindNode("cast2") != nullptr);
}

TEST_F(PatternFusionPassTest, CastRemoveConsecutiveCastDtypesNotMatch) {
  // A(float) -> Cast1(float16) -> Cast2(int32) -> B(int32)
  // Cast1输入是float，Cast2输出是int32，不匹配
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("Data");
  auto cast1 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(1).Build("cast1");
  auto cast2 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_INT32, {2, 3}).InCnt(1).OutCnt(1).Build("cast2");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_INT32, {2, 3}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(cast1)->NODE(cast2)->NODE(relu)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  // cast1: float -> float16
  // cast2: float16 -> int32
  // 这样cast1输入是float，cast2输出是int32，不匹配
  auto cast1_node = graph->FindNode("cast1");
  ASSERT_NE(cast1_node, nullptr);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT16);

  auto cast2_node = graph->FindNode("cast2");
  ASSERT_NE(cast2_node, nullptr);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_INT32);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_INT32);

  bool changed = false;
  EXPECT_EQ(CastRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_TRUE(graph->FindNode("cast1") != nullptr);
  EXPECT_TRUE(graph->FindNode("cast2") != nullptr);
}

TEST_F(PatternFusionPassTest, CastRemoveConsecutiveCastDtypesMatch) {
  // 测试连续 Cast 消除，首尾 dtype 匹配
  // A(float) -> Cast1(float16) -> Cast2(float) -> B(float)
  // Cast1输入是float，Cast2输出是float，匹配，两个都消除
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("Data");
  auto cast1 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT16, {2, 3}).InCnt(1).OutCnt(1).Build("cast1");
  auto cast2 = OP_CFG("Cast").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("cast2");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(cast1)->NODE(cast2)->NODE(relu)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  // cast1: float -> float16
  // cast2: float16 -> float
  // 这样cast1输入是float，cast2输出是float，匹配
  auto cast1_node = graph->FindNode("cast1");
  ASSERT_NE(cast1_node, nullptr);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  cast1_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT16);

  auto cast2_node = graph->FindNode("cast2");
  ASSERT_NE(cast2_node, nullptr);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  cast2_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT16);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  cast2_node->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);

  bool changed = false;
  EXPECT_EQ(CastRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(graph->FindNode("cast1") == nullptr);
  EXPECT_TRUE(graph->FindNode("cast2") == nullptr);
  EXPECT_EQ(graph->FindNode("relu")->GetInDataNodes().at(0)->GetName(), "Data");
}

TEST_F(PatternFusionPassTest, TransposeWithBroadcastMultipleTranspose) {
  // 测试多个transpose+broadcast的情况
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto zeros1 = OP_CFG("ZerosLike").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("zeros1");
  auto transpose1 = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("transpose1");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("relu");

  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(0).OutCnt(1).Build("data2");
  auto zeros2 = OP_CFG("OnesLike").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("ones2");
  auto transpose2 = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("transpose2");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->NODE(zeros1)->NODE(transpose1)->NODE(relu)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(data2)->NODE(zeros2)->NODE(transpose2)->NODE(add)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm1 = {1, 0};
  std::vector<int64_t> perm2 = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose1")->GetOpDesc(), "perm", perm1);
  AttrUtils::SetListInt(graph->FindNode("transpose2")->GetOpDesc(), "perm", perm2);

  bool changed = false;
  EXPECT_EQ(TransposeWithBroadcastEliminatePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(graph->FindNode("transpose1") == nullptr);
  EXPECT_TRUE(graph->FindNode("transpose2") == nullptr);

  // 验证zeros1和ones2的shape已更新为对应transpose的输出shape
  auto zeros1_node = graph->FindNode("zeros1");
  auto ones2_node = graph->FindNode("ones2");
  ASSERT_NE(zeros1_node, nullptr);
  ASSERT_NE(ones2_node, nullptr);

  const auto &zeros1_shape = zeros1_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(zeros1_shape.GetDims().size(), 2UL);
  EXPECT_EQ(zeros1_shape.GetDim(0), 3);
  EXPECT_EQ(zeros1_shape.GetDim(1), 2);

  const auto &ones2_shape = ones2_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(ones2_shape.GetDims().size(), 2UL);
  EXPECT_EQ(ones2_shape.GetDim(0), 2);
  EXPECT_EQ(ones2_shape.GetDim(1), 3);
}

TEST_F(PatternFusionPassTest, TransposeFillWithScalarValue) {
  // 测试 Fill + Transpose 消除，value是标量
  auto shape_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_INT32, {2}).InCnt(0).OutCnt(1).Build("shape_data");
  auto value_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1}).InCnt(0).OutCnt(1).Build("value_data");
  auto fill = OP_CFG("Fill").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 4}).InCnt(2).OutCnt(1).Build("fill");
  auto transpose = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {4, 3}).InCnt(1).OutCnt(1).Build("transpose");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(shape_data)->DATA_EDGE(0, 0)->NODE(fill)->NODE(transpose)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(value_data)->DATA_EDGE(0, 1)->NODE(fill));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose")->GetOpDesc(), "perm", perm);

  bool changed = false;
  EXPECT_EQ(TransposeWithBroadcastEliminatePass().Run(graph, changed), GRAPH_SUCCESS);
  // value是标量，应该能消除
  EXPECT_TRUE(changed);
  EXPECT_TRUE(graph->FindNode("transpose") == nullptr);

  // 验证fill的shape已更新为transpose的输出shape
  auto fill_node = graph->FindNode("fill");
  ASSERT_NE(fill_node, nullptr);
  const auto &fill_shape = fill_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(fill_shape.GetDims().size(), 2UL);
  EXPECT_EQ(fill_shape.GetDim(0), 4);
  EXPECT_EQ(fill_shape.GetDim(1), 3);
}

TEST_F(PatternFusionPassTest, TransposeFillWithConstTensorValue) {
  // 测试 Fill + Transpose 不消除，value是const张量（只支持scalar）
  auto shape_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_INT32, {2}).InCnt(0).OutCnt(1).Build("shape_data");
  // 使用Const节点（类型为Constant），value是张量 {3, 2}
  GeTensorDesc value_desc(GeShape({3, 2}), FORMAT_ND, DT_FLOAT);
  GeTensor value_tensor(value_desc);
  auto value_const = OP_CFG("Constant").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(0).OutCnt(1).Build("value_const");
  auto fill = OP_CFG("Fill").TensorDesc(FORMAT_ND, DT_FLOAT, {8, 3, 2}).InCnt(2).OutCnt(1).Build("fill");
  auto transpose = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {8, 2, 3}).InCnt(1).OutCnt(1).Build("transpose");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(shape_data)->DATA_EDGE(0, 0)->NODE(fill)->NODE(transpose)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(value_const)->DATA_EDGE(0, 1)->NODE(fill));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose")->GetOpDesc(), "perm", perm);

  bool changed = false;
  EXPECT_EQ(TransposeWithBroadcastEliminatePass().Run(graph, changed), GRAPH_SUCCESS);
  // value是const张量，但不是scalar，不应该消除
  EXPECT_FALSE(changed);
  EXPECT_TRUE(graph->FindNode("transpose") != nullptr);
}

TEST_F(PatternFusionPassTest, TransposeFillWithNonConstTensorValue) {
  // 测试 Fill + Transpose 不消除，value是非const张量
  auto shape_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_INT32, {2}).InCnt(0).OutCnt(1).Build("shape_data");
  auto value_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("value_data");
  auto fill = OP_CFG("Fill").TensorDesc(FORMAT_ND, DT_FLOAT, {4, 2, 3}).InCnt(2).OutCnt(1).Build("fill");
  auto transpose = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2, 4}).InCnt(1).OutCnt(1).Build("transpose");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(shape_data)->DATA_EDGE(0, 0)->NODE(fill)->NODE(transpose)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(value_data)->DATA_EDGE(0, 1)->NODE(fill));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm = {0, 2, 1};
  AttrUtils::SetListInt(graph->FindNode("transpose")->GetOpDesc(), "perm", perm);

  bool changed = false;
  EXPECT_EQ(TransposeWithBroadcastEliminatePass().Run(graph, changed), GRAPH_SUCCESS);
  // value是非const张量，不应该消除
  EXPECT_FALSE(changed);
  EXPECT_TRUE(graph->FindNode("transpose") != nullptr);
}

TEST_F(PatternFusionPassTest, SliceForwardSingleElemNoChain) {
  // 测试单个elemwise没有链的情况
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({4, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{2, 3, 4});
    auto slice = es::Slice(data, offset, size);
    slice.SetSymbolShape({"2", "s1", "s2"});
    slice.SetShape({2, 3, 4});

    auto relu = es::Relu(slice);
    relu.SetSymbolShape({"2", "s1", "s2"});
    relu.SetShape({2, 3, 4});

    es_graph_->SetOutput(relu, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);
}

TEST_F(PatternFusionPassTest, SliceForwardMultiElemChain) {
  // 测试多elemwise节点的slice前移
  // 图结构: Data -> Abs -> Abs -> Slice -> Output
  // 前移后:  Data -> Slice -> Abs -> Abs -> Output
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({8, 3, 4});

    auto abs1 = es::Abs(data);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetShape({8, 3, 4});

    auto abs2 = es::Abs(abs1);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    abs2.SetShape({8, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{4, 3, 4});
    auto slice = es::Slice(abs2, offset, size);
    slice.SetSymbolShape({"4", "s1", "s2"});
    slice.SetShape({4, 3, 4});

    auto abs3 = es::Abs(slice);
    abs3.SetSymbolShape({"4", "s1", "s2"});
    abs3.SetShape({4, 3, 4});

    es_graph_->SetOutput(abs3, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Abs") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(output_desc->GetShape());
      input_desc->SetOriginShape(output_desc->GetOriginShape());
      input_desc->CopyAttrsFrom(*output_desc);
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // 验证slice被前移到Data后面
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Data");

  // 验证abs1和abs2的输入连接
  auto abs1_node = cg->FindNode("Abs_0");
  auto abs2_node = cg->FindNode("Abs_1");
  ASSERT_NE(abs1_node, nullptr);
  ASSERT_NE(abs2_node, nullptr);
  EXPECT_EQ(abs1_node->GetInDataNodes().at(0)->GetType(), "Slice");
  EXPECT_EQ(abs2_node->GetInDataNodes().at(0)->GetName(), "Abs_0");

  // 验证abs1和abs2的shape已更新为slice的输出shape
  const auto &abs1_shape = abs1_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(abs1_shape.GetDims().size(), 3UL);
  EXPECT_EQ(abs1_shape.GetDim(0), 4);
  EXPECT_EQ(abs1_shape.GetDim(1), 3);
  EXPECT_EQ(abs1_shape.GetDim(2), 4);

  const auto &abs2_shape = abs2_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(abs2_shape.GetDims().size(), 3UL);
  EXPECT_EQ(abs2_shape.GetDim(0), 4);
  EXPECT_EQ(abs2_shape.GetDim(1), 3);
  EXPECT_EQ(abs2_shape.GetDim(2), 4);

  // 验证abs1和abs2的输入输出shape一致（elementwise特性）
  const auto &abs1_input_shape = abs1_node->GetOpDesc()->GetInputDesc(0).GetShape();
  const auto &abs1_output_shape = abs1_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(abs1_input_shape.GetDims().size(), abs1_output_shape.GetDims().size());
  EXPECT_EQ(abs1_input_shape.GetDim(0), abs1_output_shape.GetDim(0));
  EXPECT_EQ(abs1_input_shape.GetDim(1), abs1_output_shape.GetDim(1));
  EXPECT_EQ(abs1_input_shape.GetDim(2), abs1_output_shape.GetDim(2));

  const auto &abs2_input_shape = abs2_node->GetOpDesc()->GetInputDesc(0).GetShape();
  const auto &abs2_output_shape = abs2_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(abs2_input_shape.GetDims().size(), abs2_output_shape.GetDims().size());
  EXPECT_EQ(abs2_input_shape.GetDim(0), abs2_output_shape.GetDim(0));
  EXPECT_EQ(abs2_input_shape.GetDim(1), abs2_output_shape.GetDim(1));
  EXPECT_EQ(abs2_input_shape.GetDim(2), abs2_output_shape.GetDim(2));

  // 验证abs1和abs2的符号化shape已更新为slice的输出符号化shape
  auto abs1_output_attr = abs1_node->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<ge::SymbolicDescAttr>();
  ASSERT_NE(abs1_output_attr, nullptr);
  const auto &abs1_symbol_shape = abs1_output_attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(abs1_symbol_shape.GetDimNum(), 3UL);
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs1_symbol_shape.GetDim(0)), "4");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs1_symbol_shape.GetDim(1)), "s1");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs1_symbol_shape.GetDim(2)), "s2");

  auto abs2_output_attr = abs2_node->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<ge::SymbolicDescAttr>();
  ASSERT_NE(abs2_output_attr, nullptr);
  const auto &abs2_symbol_shape = abs2_output_attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(abs2_symbol_shape.GetDimNum(), 3UL);
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs2_symbol_shape.GetDim(0)), "4");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs2_symbol_shape.GetDim(1)), "s1");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs2_symbol_shape.GetDim(2)), "s2");

  // 验证abs1和abs2的输入符号化shape也一致
  auto abs1_input_attr = abs1_node->GetOpDesc()->MutableInputDesc(0)->GetAttrsGroup<ge::SymbolicDescAttr>();
  ASSERT_NE(abs1_input_attr, nullptr);
  const auto &abs1_input_symbol_shape = abs1_input_attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(abs1_input_symbol_shape.GetDimNum(), 3UL);
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs1_input_symbol_shape.GetDim(0)), "4");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs1_input_symbol_shape.GetDim(1)), "s1");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs1_input_symbol_shape.GetDim(2)), "s2");

  auto abs2_input_attr = abs2_node->GetOpDesc()->MutableInputDesc(0)->GetAttrsGroup<ge::SymbolicDescAttr>();
  ASSERT_NE(abs2_input_attr, nullptr);
  const auto &abs2_input_symbol_shape = abs2_input_attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(abs2_input_symbol_shape.GetDimNum(), 3UL);
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs2_input_symbol_shape.GetDim(0)), "4");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs2_input_symbol_shape.GetDim(1)), "s1");
  EXPECT_EQ(ge::SymbolicUtils::ToString(abs2_input_symbol_shape.GetDim(2)), "s2");
}

TEST_F(PatternFusionPassTest, SliceForwardStopAtDtypeChange) {
  // 测试dtype变化时停止收集elemwise链
  // 图结构: Data(float) -> Relu(float) -> Cast(float16) -> Relu(float16) -> Slice -> Output
  // 只有第二个Relu会被收集（因为它和Slice都是float16）
  // 前移后: Data -> Relu -> Cast -> Slice -> Relu -> Output
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({4, 3, 4});

    auto relu1 = es::Relu(data);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetShape({4, 3, 4});

    // Cast: float -> float16
    auto cast = es::Cast(relu1, ge::DT_FLOAT16);
    cast.SetSymbolShape({"s0", "s1", "s2"});
    cast.SetShape({4, 3, 4});

    auto relu2 = es::Relu(cast);
    relu2.SetSymbolShape({"s0", "s1", "s2"});
    relu2.SetShape({4, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{2, 3, 4});
    auto slice = es::Slice(relu2, offset, size);
    slice.SetSymbolShape({"2", "s1", "s2"});
    slice.SetShape({2, 3, 4});

    auto relu3 = es::Relu(slice);
    relu3.SetSymbolShape({"2", "s1", "s2"});
    relu3.SetShape({2, 3, 4});

    es_graph_->SetOutput(relu3, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的，并且需要正确设置dtype
  int relu_dtype_count = 0;
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Relu" || node->GetType() == "Cast" || node->GetType() == "Slice") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(output_desc->GetShape());
      input_desc->SetOriginShape(output_desc->GetOriginShape());
      input_desc->CopyAttrsFrom(*output_desc);

      // 修复dtype：
      // Data(float) -> Relu1(float) -> Cast(float->float16) -> Relu2(float16) -> Slice(float16) -> Relu3(float16)
      if (node->GetType() == "Relu") {
        relu_dtype_count++;
        if (relu_dtype_count == 1) {
          // 第一个Relu：float
          input_desc->SetDataType(DT_FLOAT);
          input_desc->SetOriginDataType(DT_FLOAT);
          output_desc->SetDataType(DT_FLOAT);
          output_desc->SetOriginDataType(DT_FLOAT);
        } else if (relu_dtype_count == 2) {
          // 第二个Relu（Cast之后）：float16
          input_desc->SetDataType(DT_FLOAT16);
          input_desc->SetOriginDataType(DT_FLOAT16);
          output_desc->SetDataType(DT_FLOAT16);
          output_desc->SetOriginDataType(DT_FLOAT16);
        } else {
          // 第三个Relu（Slice之后）：float16
          input_desc->SetDataType(DT_FLOAT16);
          input_desc->SetOriginDataType(DT_FLOAT16);
          output_desc->SetDataType(DT_FLOAT16);
          output_desc->SetOriginDataType(DT_FLOAT16);
        }
      } else if (node->GetType() == "Cast") {
        // Cast: float -> float16
        input_desc->SetDataType(DT_FLOAT);
        input_desc->SetOriginDataType(DT_FLOAT);
        output_desc->SetDataType(DT_FLOAT16);
        output_desc->SetOriginDataType(DT_FLOAT16);
      } else if (node->GetType() == "Slice") {
        // Slice: float16
        input_desc->SetDataType(DT_FLOAT16);
        input_desc->SetOriginDataType(DT_FLOAT16);
        output_desc->SetDataType(DT_FLOAT16);
        output_desc->SetOriginDataType(DT_FLOAT16);
      }
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // slice前移到cast之后（因为relu2之前的节点dtype不匹配）
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  auto cast_node = cg->FindFirstNodeMatchType("Cast");
  ASSERT_NE(cast_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Cast");
  EXPECT_EQ(cast_node->GetOutDataNodes().at(0)->GetType(), "Slice");

  // 验证relu2的shape已更新为slice的输出shape
  // 通过GetAllNodes找到第二个Relu
  NodePtr relu2_node = nullptr;
  int relu_count = 0;
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Relu") {
      relu_count++;
      if (relu_count == 2) {  // 第二个Relu是cast之后的
        relu2_node = node;
        break;
      }
    }
  }
  ASSERT_NE(relu2_node, nullptr);
  const auto &relu2_shape = relu2_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(relu2_shape.GetDims().size(), 3UL);
  EXPECT_EQ(relu2_shape.GetDim(0), 2);
  EXPECT_EQ(relu2_shape.GetDim(1), 3);
  EXPECT_EQ(relu2_shape.GetDim(2), 4);

  // 验证relu2的输入输出shape一致（elementwise特性）
  const auto &relu2_input_shape = relu2_node->GetOpDesc()->GetInputDesc(0).GetShape();
  const auto &relu2_output_shape = relu2_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(relu2_input_shape.GetDims().size(), relu2_output_shape.GetDims().size());
  EXPECT_EQ(relu2_input_shape.GetDim(0), relu2_output_shape.GetDim(0));
  EXPECT_EQ(relu2_input_shape.GetDim(1), relu2_output_shape.GetDim(1));
  EXPECT_EQ(relu2_input_shape.GetDim(2), relu2_output_shape.GetDim(2));

  // 验证relu2的符号化shape已更新为slice的输出符号化shape
  auto relu2_output_attr = relu2_node->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(relu2_output_attr, nullptr);
  const auto &relu2_symbol_shape = relu2_output_attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(relu2_symbol_shape.GetDimNum(), 3UL);
  EXPECT_EQ(ge::SymbolicUtils::ToString(relu2_symbol_shape.GetDim(0)), "2");
  EXPECT_EQ(ge::SymbolicUtils::ToString(relu2_symbol_shape.GetDim(1)), "s1");
  EXPECT_EQ(ge::SymbolicUtils::ToString(relu2_symbol_shape.GetDim(2)), "s2");

  // 验证relu2的输入符号化shape也一致
  auto relu2_input_attr = relu2_node->GetOpDesc()->MutableInputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(relu2_input_attr, nullptr);
  const auto &relu2_input_symbol_shape = relu2_input_attr->symbolic_tensor.GetOriginSymbolShape();
  EXPECT_EQ(relu2_input_symbol_shape.GetDimNum(), 3UL);
  EXPECT_EQ(ge::SymbolicUtils::ToString(relu2_input_symbol_shape.GetDim(0)), "2");
  EXPECT_EQ(ge::SymbolicUtils::ToString(relu2_input_symbol_shape.GetDim(1)), "s1");
  EXPECT_EQ(ge::SymbolicUtils::ToString(relu2_input_symbol_shape.GetDim(2)), "s2");
}

TEST_F(PatternFusionPassTest, SliceForwardMultiInputElemwise) {
  // 测试多输入 elementwise 的 slice 上提
  // 图结构: Data -> Relu -> mul(Relu, Relu) -> Slice -> Output
  // 前移后:  Data -> Slice -> Relu -> mul(Relu, Relu) -> Output
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({8, 3, 4});

    auto relu = es::Relu(data);
    relu.SetSymbolShape({"s0", "s1", "s2"});
    relu.SetShape({8, 3, 4});

    // Mul需要两个输入，这里都来自relu
    auto mul = es::Mul(relu, relu);
    mul.SetSymbolShape({"s0", "s1", "s2"});
    mul.SetShape({8, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{4, 3, 4});
    auto slice = es::Slice(mul, offset, size);
    slice.SetSymbolShape({"4", "s1", "s2"});
    slice.SetShape({4, 3, 4});

    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Relu" || node->GetType() == "Mul") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(output_desc->GetShape());
      input_desc->SetOriginShape(output_desc->GetOriginShape());
      input_desc->CopyAttrsFrom(*output_desc);
    }
  }

  // 修复：Mul节点需要设置SymbolicDescAttr
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      // 获取Relu节点的symbolic shape作为模板
      auto relu_node = cg->FindFirstNodeMatchType("Relu");
      if (relu_node != nullptr) {
        auto relu_output_attr = ge::pattern_fusion::GetNodeMutableOutputAttr(relu_node);
        if (relu_output_attr != nullptr) {
          // 为Mul的输入输出设置SymbolicDescAttr
          for (const auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
            auto input_attr = input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
            if (input_attr == nullptr) {
              // 如果没有SymbolicDescAttr，则复制一份
              auto relu_input_desc = relu_node->GetOpDesc()->MutableInputDesc(0);
              auto relu_input_attr = relu_input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
              if (relu_input_attr != nullptr) {
                input_desc->CopyAttrsFrom(*relu_input_desc);
              }
            } else {
              input_attr->symbolic_tensor.MutableOriginSymbolShape() =
                  relu_output_attr->symbolic_tensor.GetOriginSymbolShape();
            }
          }
          for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
            auto output_attr = output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
            if (output_attr == nullptr) {
              auto relu_output_desc = relu_node->GetOpDesc()->MutableOutputDesc(0);
              output_desc->CopyAttrsFrom(*relu_output_desc);
            } else {
              output_attr->symbolic_tensor.MutableOriginSymbolShape() =
                  relu_output_attr->symbolic_tensor.GetOriginSymbolShape();
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // 验证 slice 被前移到 Data 后面
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Data");

  // 验证 relu 的输入来自 slice
  auto relu_node = cg->FindFirstNodeMatchType("Relu");
  ASSERT_NE(relu_node, nullptr);
  EXPECT_EQ(relu_node->GetInDataNodes().at(0)->GetType(), "Slice");

  // 验证 mul 的两个输入都来自 relu
  auto mul_node = cg->FindFirstNodeMatchType("Mul");
  ASSERT_NE(mul_node, nullptr);
  EXPECT_EQ(mul_node->GetInDataNodes().at(0)->GetType(), "Relu");
  EXPECT_EQ(mul_node->GetInDataNodes().at(1)->GetType(), "Relu");

  // 验证 relu 和 mul 的 shape 已更新为 slice 的输出 shape
  const auto &relu_shape = relu_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(relu_shape.GetDims(), std::vector<int64_t>({4, 3, 4}));

  const auto &mul_shape = mul_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(mul_shape.GetDims(), std::vector<int64_t>({4, 3, 4}));
}

TEST_F(PatternFusionPassTest, SliceForwardMultiInputElemwiseWithChain) {
  // 测试多输入 elementwise 链的 slice 上提
  // 图结构: Data -> Relu -> mul(Relu, Relu) -> add(mul, mul) -> Slice -> Output
  // 前移后:  Data -> Slice -> Relu -> mul(Relu, Relu) -> add(mul, mul) -> Output
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({8, 3, 4});

    auto relu = es::Relu(data);
    relu.SetSymbolShape({"s0", "s1", "s2"});
    relu.SetShape({8, 3, 4});

    // Mul需要两个输入，这里都来自relu
    auto mul = es::Mul(relu, relu);
    mul.SetSymbolShape({"s0", "s1", "s2"});
    mul.SetShape({8, 3, 4});

    // Add需要两个输入，这里都来自mul
    auto add = es::Add(mul, mul);
    add.SetSymbolShape({"s0", "s1", "s2"});
    add.SetShape({8, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{4, 3, 4});
    auto slice = es::Slice(add, offset, size);
    slice.SetSymbolShape({"4", "s1", "s2"});
    slice.SetShape({4, 3, 4});

    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Relu" || node->GetType() == "Mul" || node->GetType() == "Add") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(output_desc->GetShape());
      input_desc->SetOriginShape(output_desc->GetOriginShape());
      input_desc->CopyAttrsFrom(*output_desc);
    }
  }

  // 修复：Mul和Add节点需要设置SymbolicDescAttr
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Mul" || node->GetType() == "Add") {
      // 获取Relu节点的symbolic shape作为模板
      auto relu_node = cg->FindFirstNodeMatchType("Relu");
      if (relu_node != nullptr) {
        auto relu_output_attr = ge::pattern_fusion::GetNodeMutableOutputAttr(relu_node);
        if (relu_output_attr != nullptr) {
          // 为节点设置SymbolicDescAttr
          for (size_t i = 0; i < node->GetOpDesc()->GetInputsSize(); ++i) {
            auto input_desc = node->GetOpDesc()->MutableInputDesc(i);
            auto input_attr = input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
            if (input_attr == nullptr) {
              auto relu_input_desc = relu_node->GetOpDesc()->MutableInputDesc(0);
              auto relu_input_attr = relu_input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
              if (relu_input_attr != nullptr) {
                input_desc->CopyAttrsFrom(*relu_input_desc);
              }
            } else {
              input_attr->symbolic_tensor.MutableOriginSymbolShape() =
                  relu_output_attr->symbolic_tensor.GetOriginSymbolShape();
            }
          }
          for (size_t i = 0; i < node->GetOpDesc()->GetOutputsSize(); ++i) {
            auto output_desc = node->GetOpDesc()->MutableOutputDesc(i);
            auto output_attr = output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
            if (output_attr == nullptr) {
              auto relu_output_desc = relu_node->GetOpDesc()->MutableOutputDesc(0);
              output_desc->CopyAttrsFrom(*relu_output_desc);
            } else {
              output_attr->symbolic_tensor.MutableOriginSymbolShape() =
                  relu_output_attr->symbolic_tensor.GetOriginSymbolShape();
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // 验证 slice 被前移到 Data 后面
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Data");

  // 验证各节点的连接关系
  auto relu_node = cg->FindFirstNodeMatchType("Relu");
  auto mul_node = cg->FindFirstNodeMatchType("Mul");
  auto add_node = cg->FindFirstNodeMatchType("Add");

  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(mul_node, nullptr);
  ASSERT_NE(add_node, nullptr);

  EXPECT_EQ(relu_node->GetInDataNodes().at(0)->GetType(), "Slice");
  EXPECT_EQ(mul_node->GetInDataNodes().at(0)->GetType(), "Relu");
  EXPECT_EQ(mul_node->GetInDataNodes().at(1)->GetType(), "Relu");
  EXPECT_EQ(add_node->GetInDataNodes().at(0)->GetType(), "Mul");
  EXPECT_EQ(add_node->GetInDataNodes().at(1)->GetType(), "Mul");
}

TEST_F(PatternFusionPassTest, SliceForwardMultiInputDifferentSource) {
  // 测试多输入来自不同源节点时，slice 不能上提
  // 图结构: Data1 -> Relu1 --\
  //                         -> mul -> Slice -> Output
  //        Data2 -> Relu2 --/
  [this]() {
    auto data1 = es_graph_->CreateInput(0, "data0", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetShape({8, 3, 4});

    auto data2 = es_graph_->CreateInput(1, "data1", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    data2.SetShape({8, 3, 4});

    auto relu1 = es::Relu(data1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetShape({8, 3, 4});

    auto relu2 = es::Relu(data2);
    relu2.SetSymbolShape({"s0", "s1", "s2"});
    relu2.SetShape({8, 3, 4});

    // Mul需要两个输入，这里分别来自relu1和relu2
    auto mul = es::Mul(relu1, relu2);
    mul.SetSymbolShape({"s0", "s1", "s2"});
    mul.SetShape({8, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{4, 3, 4});
    auto slice = es::Slice(mul, offset, size);
    slice.SetSymbolShape({"4", "s1", "s2"});
    slice.SetShape({4, 3, 4});

    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Relu" || node->GetType() == "Mul") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(output_desc->GetShape());
      input_desc->SetOriginShape(output_desc->GetOriginShape());
      input_desc->CopyAttrsFrom(*output_desc);
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // 验证 slice 没有被前移（因为 mul 的两个输入来自不同的源）
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Mul");
}

TEST_F(PatternFusionPassTest, SliceForwardMultiInputFromSameSource) {
  // 测试多输入 elementwise 的输入来自同一个源节点时，slice 应该上提
  // 图结构: Data -> Mul(Data, Data) -> Slice -> Output
  // 前移后:    Data -> Slice -> Mul(Slice输出, Slice输出) -> Output
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({8, 3, 4});

    // Mul 的两个输入都直接来自 data
    auto mul = es::Mul(data, data);
    mul.SetSymbolShape({"s0", "s1", "s2"});
    mul.SetShape({8, 3, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{4, 3, 4});
    auto slice = es::Slice(mul, offset, size);
    slice.SetSymbolShape({"4", "s1", "s2"});
    slice.SetShape({4, 3, 4});

    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      if (input_desc != nullptr && output_desc != nullptr) {
        input_desc->SetShape(output_desc->GetShape());
        input_desc->SetOriginShape(output_desc->GetOriginShape());
        input_desc->CopyAttrsFrom(*output_desc);
      }
      // 处理第二个输入
      auto input_desc1 = node->GetOpDesc()->MutableInputDesc(1);
      if (input_desc1 != nullptr) {
        input_desc1->SetShape(output_desc->GetShape());
        input_desc1->SetOriginShape(output_desc->GetOriginShape());
        input_desc1->CopyAttrsFrom(*output_desc);
      }
    }
  }

  // 修复：Mul节点需要设置SymbolicDescAttr，从Data节点的output复制
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Mul") {
      auto data_node = cg->FindFirstNodeMatchType("Data");
      if (data_node != nullptr) {
        auto data_output_desc = data_node->GetOpDesc()->MutableOutputDesc(0);
        if (data_output_desc != nullptr) {
          auto data_output_attr = data_output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
          // 为Mul的输入输出设置SymbolicDescAttr
          for (size_t i = 0; i < node->GetOpDesc()->GetInputsSize(); ++i) {
            auto input_desc = node->GetOpDesc()->MutableInputDesc(i);
            if (input_desc != nullptr) {
              input_desc->CopyAttrsFrom(*data_output_desc);
            }
          }
          for (size_t i = 0; i < node->GetOpDesc()->GetOutputsSize(); ++i) {
            auto output_desc = node->GetOpDesc()->MutableOutputDesc(i);
            if (output_desc != nullptr) {
              output_desc->CopyAttrsFrom(*data_output_desc);
            }
          }
        }
      }
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // 验证 slice 的输入来自 Data
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Data");

  // 验证 mul 的两个输入都来自 Slice
  auto mul_node = cg->FindFirstNodeMatchType("Mul");
  ASSERT_NE(mul_node, nullptr);
  EXPECT_EQ(mul_node->GetInDataNodes().at(0)->GetType(), "Slice");
  EXPECT_EQ(mul_node->GetInDataNodes().at(1)->GetType(), "Slice");

  // 验证 mul 的 shape 已更新为 slice 的输出 shape
  const auto &mul_shape = mul_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(mul_shape.GetDims(), std::vector<int64_t>({4, 3, 4}));
}

}  // namespace ge
