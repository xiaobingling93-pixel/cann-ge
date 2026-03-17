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
#include "pattern_fusion/redundant_control_edge_remove_pass.h"
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
  // 测试多个 ZerosLike + Transpose 的情况
  // ZerosLike + Transpose 会被替换为 Constant + BroadcastTo
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto zeros1 = OP_CFG("ZerosLike").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("zeros1");
  auto transpose1 = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("transpose1");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("relu");

  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data2");
  auto zeros2 = OP_CFG("ZerosLike").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("zeros2");
  auto transpose2 = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("transpose2");

  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->NODE(zeros1)->NODE(transpose1)->NODE(relu)->DATA_EDGE(0, 0)->NODE(add)->NODE("output_0",
      NETOUTPUT));
    CHAIN(NODE(data2)->NODE(zeros2)->NODE(transpose2)->DATA_EDGE(0, 1)->NODE(add)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm1 = {1, 0};
  std::vector<int64_t> perm2 = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose1")->GetOpDesc(), "perm", perm1);
  GraphUtils::DumpGEGraphToOnnx(*graph, "BEFORE");
  bool changed = false;
  EXPECT_EQ(TransposeWithBroadcastEliminatePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  GraphUtils::DumpGEGraphToOnnx(*graph, "AFTER");

  // 验证原来的 zeros1, zeros2, transpose1, transpose2 节点被删除
  EXPECT_TRUE(graph->FindNode("transpose1") == nullptr);
  EXPECT_TRUE(graph->FindNode("transpose2") == nullptr);
  EXPECT_TRUE(graph->FindNode("zeros1") == nullptr);
  EXPECT_TRUE(graph->FindNode("zeros2") == nullptr);

  // 验证新的 BroadcastTo 节点被创建
  bool found_broadcast_to = false;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "BroadcastTo") {
      found_broadcast_to = true;
      break;
    }
  }
  EXPECT_TRUE(found_broadcast_to);

  // 验证 Constant 节点被创建（用于存放 0 值和 shape）
  auto const_nodes = graph->GetDirectNode();
  int const_count = 0;
  for (const auto &node : const_nodes) {
    if (node->GetType() == "Constant") {
      const_count++;
    }
  }
  // 应该有4个 Constant 节点：2个值常量 + 2个shape常量
  EXPECT_GE(const_count, 4);
}

TEST_F(PatternFusionPassTest, TransposeFillWithScalarValue) {
  // 测试 Fill + Transpose 转换为 BroadcastTo，value是标量
  auto shape_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_INT32, {2}).InCnt(0).OutCnt(1).Build("shape_data");
  auto value_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {}).InCnt(0).OutCnt(1).Build("value_data");
  auto fill = OP_CFG("Fill").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 4}).InCnt(2).OutCnt(1).Build("fill");
  auto transpose = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {4, 3}).InCnt(1).OutCnt(1).Build("transpose");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(shape_data)->DATA_EDGE(0, 0)->NODE(fill)->NODE(transpose)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(value_data)->DATA_EDGE(0, 1)->NODE(fill));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose")->GetOpDesc(), "perm", perm);

  EXPECT_EQ(PatternFusion::RunEarlyPasses(graph), GRAPH_SUCCESS);
  // value是标量，应该能转换为 BroadcastTo
  EXPECT_TRUE(graph->FindNode("transpose") == nullptr);
  EXPECT_TRUE(graph->FindNode("fill") == nullptr);

  // 验证 BroadcastTo 节点被创建
  bool found_broadcast_to = false;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == "BroadcastTo") {
      found_broadcast_to = true;
      break;
    }
  }
  EXPECT_TRUE(found_broadcast_to);
}

TEST_F(PatternFusionPassTest, TransposeFillWithConstTensorValue) {
  // 测试 Fill + Transpose 不转换为 BroadcastTo，value是const张量（只支持scalar）
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

  EXPECT_EQ(PatternFusion::RunEarlyPasses(graph), GRAPH_SUCCESS);
  // value是const张量，但不是scalar，不应该转换为 BroadcastTo
  EXPECT_TRUE(graph->FindNode("transpose") != nullptr);
}

TEST_F(PatternFusionPassTest, TransposeFillWithNonConstTensorValue) {
  // 测试 Fill + Transpose 不转换为 BroadcastTo，value是非const张量
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

  EXPECT_EQ(PatternFusion::RunEarlyPasses(graph), GRAPH_SUCCESS);
  // value是非const张量，不应该转换为 BroadcastTo
  EXPECT_TRUE(graph->FindNode("transpose") != nullptr);
}

TEST_F(PatternFusionPassTest, TransposeZerosLikeWithMultipleConsumers) {
  // 测试 ZerosLike 输出有多个消费者时，不应该消除
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data");
  auto zeros = OP_CFG("ZerosLike").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("zeros");
  auto transpose = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 2}).InCnt(1).OutCnt(1).Build("transpose");
  auto mul = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("mul");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(zeros));
    // zeros 有两个消费者：transpose 和 mul
    CHAIN(NODE(zeros)->NODE(transpose)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(zeros)->NODE(mul)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose")->GetOpDesc(), "perm", perm);

  EXPECT_EQ(PatternFusion::RunEarlyPasses(graph), GRAPH_SUCCESS);
  // zeros 有多个消费者，不应该消除
  EXPECT_TRUE(graph->FindNode("transpose") != nullptr);
  EXPECT_TRUE(graph->FindNode("zeros") != nullptr);
}

TEST_F(PatternFusionPassTest, TransposeFillWithMultipleConsumers) {
  // 测试 Fill 输出有多个消费者时，不应该消除
  auto shape_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_INT32, {2}).InCnt(0).OutCnt(1).Build("shape_data");
  auto value_data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {}).InCnt(0).OutCnt(1).Build("value_data");
  auto fill = OP_CFG("Fill").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 4}).InCnt(2).OutCnt(1).Build("fill");
  auto transpose = OP_CFG("Transpose").TensorDesc(FORMAT_ND, DT_FLOAT, {4, 3}).InCnt(1).OutCnt(1).Build("transpose");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {3, 4}).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(shape_data)->DATA_EDGE(0, 0)->NODE(fill));
    CHAIN(NODE(value_data)->DATA_EDGE(0, 1)->NODE(fill));
    // fill 有两个消费者：transpose 和 add
    CHAIN(NODE(fill)->NODE(transpose)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(fill)->NODE(add)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);
  std::vector<int64_t> perm = {1, 0};
  AttrUtils::SetListInt(graph->FindNode("transpose")->GetOpDesc(), "perm", perm);

  EXPECT_EQ(PatternFusion::RunEarlyPasses(graph), GRAPH_SUCCESS);
  // fill 有多个消费者，不应该消除
  EXPECT_TRUE(graph->FindNode("transpose") != nullptr);
  EXPECT_TRUE(graph->FindNode("fill") != nullptr);
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
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    data.SetShape({4, 4, 4});

    // Transpose: perm={1,0,2}, {4,4,4} -> {4,4,4} (shape 相同，但数据布局改变)
    auto perm = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{1, 0, 2});
    auto transpose = es::Transpose(data, perm);
    transpose.SetSymbolShape({"s1", "s0", "s2"});
    transpose.SetShape({4, 4, 4});

    // Mul 的两个输入都来自 transpose
    auto mul = es::Mul(transpose, transpose);
    mul.SetSymbolShape({"s1", "s0", "s2"});
    mul.SetShape({4, 4, 4});

    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 0});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{2, 4, 4});
    auto slice = es::Slice(mul, offset, size);
    slice.SetSymbolShape({"2", "s0", "s2"});
    slice.SetShape({2, 4, 4});

    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  // 修复：es构图只刷了output_shape，input_shape是空的
  for (const auto &node : cg->GetAllNodes()) {
    if (node->GetType() == "Transpose") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(GeShape({4, 4, 4}));
      input_desc->SetOriginShape(GeShape({4, 4, 4}));
      output_desc->SetShape(GeShape({4, 4, 4}));
      output_desc->SetOriginShape(GeShape({4, 4, 4}));
      input_desc->CopyAttrsFrom(*output_desc);
    } else if (node->GetType() == "Mul") {
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
    } else if (node->GetType() == "Slice") {
      auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
      auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
      input_desc->SetShape(output_desc->GetShape());
      input_desc->SetOriginShape(output_desc->GetOriginShape());
      input_desc->CopyAttrsFrom(*output_desc);
    }
  }

  EXPECT_EQ(SliceForwardFusionPass().Run(cg), GRAPH_SUCCESS);

  // 验证 slice 被前移，输入是 Transpose
  auto slice_node = cg->FindFirstNodeMatchType("Slice");
  ASSERT_NE(slice_node, nullptr);
  EXPECT_EQ(slice_node->GetInDataNodes().at(0)->GetType(), "Transpose");

  // 验证 mul 的两个输入都来自 Slice
  auto mul_node = cg->FindFirstNodeMatchType("Mul");
  ASSERT_NE(mul_node, nullptr);
  EXPECT_EQ(mul_node->GetInDataNodes().at(0)->GetType(), "Slice");
  EXPECT_EQ(mul_node->GetInDataNodes().at(1)->GetType(), "Slice");

  // 验证 Transpose 仍然在图中
  EXPECT_NE(cg->FindFirstNodeMatchType("Transpose"), nullptr);
}

TEST_F(PatternFusionPassTest, RemoveConstControlEdgeForConcatFusion) {
  // 测试场景：形成"环"的冗余控制边应该被删除
  // 图结构：
  //   data1 -data------> concat1
  //   data1 -ctrl-> const1 -data-> concat1
  //   data2 -----------> concat1
  //
  // 形成环：data1 -> const1 -> concat1 且 data1 -> concat1
  // 预期：RedundantControlEdgeRemovePass 删除 data1->const1 的控制边

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {10, 20}).InCnt(0).OutCnt(1).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {5, 20}).InCnt(0).OutCnt(1).Build("data2");
  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_ND, DT_FLOAT, {5, 20}).InCnt(0).OutCnt(1).Build("const1");

  auto concat1 = OP_CFG("ConcatD").TensorDesc(FORMAT_ND, DT_FLOAT, {15, 20}).InCnt(3).OutCnt(1).Build("concat1");
  auto relu = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {15, 20}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->DATA_EDGE(0, 0)->NODE(concat1));
    CHAIN(NODE(const1)->DATA_EDGE(0, 1)->NODE(concat1));
    CHAIN(NODE(data2)->DATA_EDGE(0, 2)->NODE(concat1));
    CHAIN(NODE(concat1)->NODE(relu)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);

  // 设置 concat_dim = 0 (在第0维拼接)
  AttrUtils::SetInt(graph->FindNode("concat1")->GetOpDesc(), "concat_dim", 0);
  AttrUtils::SetInt(graph->FindNode("concat1")->GetOpDesc(), "N", 3);

  // 添加控制边：data1 -ctrl-> const1（形成环）
  auto data1_node = graph->FindNode("data1");
  auto const1_node = graph->FindNode("const1");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_NE(const1_node, nullptr);
  auto ret = GraphUtils::AddEdge(data1_node->GetOutControlAnchor(), const1_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  // 验证控制边已添加
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);

  // 运行 RedundantControlEdgeRemovePass
  bool changed = false;
  EXPECT_EQ(RedundantControlEdgeRemovePass().Run(graph, changed), GRAPH_SUCCESS);

  // 验证控制边已被删除（data1 是 const1 输出节点 concat1 的数据输入，形成环）
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 0);
}

TEST_F(PatternFusionPassTest, RemoveConstControlEdgeSingleRef) {
  // 测试场景：单引用 const 的控制边形成"环"时应该被删除
  // 图结构：
  //   data1 -data-> add
  //   const1 -data-> add
  //   data1 -ctrl-> const1
  //
  // 形成环：data1 -> const1 -> add 且 data1 -> add
  // 预期：删除 data1->const1 的控制边

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("const1");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->DATA_EDGE(0, 0)->NODE(add));
    CHAIN(NODE(const1)->DATA_EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->NODE("output_0", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);

  // 添加控制边：data1 -ctrl-> const1（形成环）
  auto data1_node = graph->FindNode("data1");
  auto const1_node = graph->FindNode("const1");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_NE(const1_node, nullptr);
  auto ret = GraphUtils::AddEdge(data1_node->GetOutControlAnchor(), const1_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  // 验证控制边已添加
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);

  // 运行 Pass
  bool changed = false;
  EXPECT_EQ(RedundantControlEdgeRemovePass().Run(graph, changed), GRAPH_SUCCESS);

  // 验证控制边已被删除（data1 是 const1 输出节点 add 的数据输入，形成环）
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 0);
}

TEST_F(PatternFusionPassTest, RemoveConstControlEdgeMultiRefAllOutputsControlled) {
  // 测试场景：多引用 const，但所有输出都以 ctrl_src 为数据输入，控制边应该被删除
  // 图结构：
  //   data1 -data-> add1
  //   data1 -data-> add2
  //   const1 -data-> add1
  //   const1 -data-> add2
  //   data1 -ctrl-> const1
  //
  // 形成环：data1 通过数据边控制 add1 和 add2，而 const1 也输出到 add1 和 add2
  // 预期：删除 data1->const1 的控制边

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("const1");
  auto add1 = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add1");
  auto add2 = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add2");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->NODE(add1));
    CHAIN(NODE(data1)->NODE(add2));
    CHAIN(NODE(const1)->NODE(add1));
    CHAIN(NODE(const1)->NODE(add2));
    CHAIN(NODE(add1)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(add2)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);

  // 添加控制边：data1 -ctrl-> const1
  auto data1_node = graph->FindNode("data1");
  auto const1_node = graph->FindNode("const1");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_NE(const1_node, nullptr);
  auto ret = GraphUtils::AddEdge(data1_node->GetOutControlAnchor(), const1_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  // 验证控制边已添加
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);

  // 运行 Pass
  bool changed = false;
  EXPECT_EQ(RedundantControlEdgeRemovePass().Run(graph, changed), GRAPH_SUCCESS);

  // 验证控制边已被删除（所有输出 add1/add2 都以 data1 为数据输入）
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 0);
}

TEST_F(PatternFusionPassTest, NotRemoveConstControlEdgeMultiRefPartialOutputsControlled) {
  // 测试场景：多引用 const，但只有部分输出以 ctrl_src 为数据输入，控制边不应该被删除
  // 图结构：
  //   data1 -data-> add1
  //   data2 -data-> add2  (data2 不是 data1)
  //   const1 -data-> add1
  //   const1 -data-> add2
  //   data1 -ctrl-> const1
  //
  // 不形成完整环：add2 不以 data1 为数据输入
  // 预期：保留 data1->const1 的控制边

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data2");
  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("const1");
  auto add1 = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add1");
  auto add2 = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add2");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->NODE(add1));
    CHAIN(NODE(data2)->NODE(add2));  // add2 的输入是 data2，不是 data1
    CHAIN(NODE(const1)->NODE(add1));
    CHAIN(NODE(const1)->NODE(add2));
    CHAIN(NODE(add1)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(add2)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);

  // 添加控制边：data1 -ctrl-> const1
  auto data1_node = graph->FindNode("data1");
  auto const1_node = graph->FindNode("const1");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_NE(const1_node, nullptr);
  auto ret = GraphUtils::AddEdge(data1_node->GetOutControlAnchor(), const1_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  // 验证控制边已添加
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);

  // 运行 Pass
  bool changed = false;
  EXPECT_EQ(RedundantControlEdgeRemovePass().Run(graph, changed), GRAPH_SUCCESS);

  // 验证控制边未被删除（add2 不以 data1 为数据输入）
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);
}

TEST_F(PatternFusionPassTest, NotRemoveControlEdgeWhenSrcNotDataInput) {
  // 测试场景：控制边来源不是输出节点的数据输入，不删除
  // 图结构：
  //   data1 -data-> add
  //   const1 -data-> add
  //   other -ctrl-> const1
  //
  // 不形成环：other 不是 add 的数据输入
  // 预期：保留 other->const1 的控制边

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto other = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("other");
  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("const1");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->DATA_EDGE(0, 0)->NODE(add));
    CHAIN(NODE(const1)->DATA_EDGE(0, 1)->NODE(add));
    CHAIN(NODE(other)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(add)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);

  // 添加控制边：other -ctrl-> const1（不形成环，因为 other 不是 add 的数据输入）
  auto other_node = graph->FindNode("other");
  auto const1_node = graph->FindNode("const1");
  ASSERT_NE(other_node, nullptr);
  ASSERT_NE(const1_node, nullptr);
  auto ret = GraphUtils::AddEdge(other_node->GetOutControlAnchor(), const1_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  // 验证控制边已添加
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);

  // 运行 Pass
  bool changed = false;
  EXPECT_EQ(RedundantControlEdgeRemovePass().Run(graph, changed), GRAPH_SUCCESS);

  // 验证控制边未被删除（other 不是 const1 输出节点 add 的数据输入，不形成环）
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);
}

TEST_F(PatternFusionPassTest, NotRemoveConstControlEdgeWhenConstHasOutControlEdge) {
  // 测试场景：Const 有外出控制边时，不删除其入控制边
  // 图结构：
  //   data1 -data-> add
  //   const1 -data-> add
  //   data1 -ctrl-> const1
  //   const1 -ctrl-> other  (const1 有外出控制边)
  //
  // 预期：保留 data1->const1 的控制边（因为删除后可能破坏 data1 与 other 之间的执行顺序）

  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("data1");
  auto const1 = OP_CFG("Const").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(0).OutCnt(1).Build("const1");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(2).OutCnt(1).Build("add");
  auto other = OP_CFG("ReLU").TensorDesc(FORMAT_ND, DT_FLOAT, {2, 3}).InCnt(1).OutCnt(1).Build("other");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data1)->DATA_EDGE(0, 0)->NODE(add));
    CHAIN(NODE(const1)->DATA_EDGE(0, 1)->NODE(add));
    CHAIN(NODE(other)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(add)->NODE("output_1", NETOUTPUT));
  };

  auto graph = ToComputeGraph(test_graph);

  // 添加控制边：data1 -ctrl-> const1，const1 -ctrl-> other
  auto data1_node = graph->FindNode("data1");
  auto const1_node = graph->FindNode("const1");
  auto other_node = graph->FindNode("other");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_NE(const1_node, nullptr);
  ASSERT_NE(other_node, nullptr);

  auto ret = GraphUtils::AddEdge(data1_node->GetOutControlAnchor(), const1_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ret = GraphUtils::AddEdge(const1_node->GetOutControlAnchor(), other_node->GetInControlAnchor());
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  // 验证控制边已添加
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);
  EXPECT_EQ(const1_node->GetOutControlNodesSize(), 1);

  // 运行 Pass
  bool changed = false;
  EXPECT_EQ(RedundantControlEdgeRemovePass().Run(graph, changed), GRAPH_SUCCESS);

  // 验证控制边未被删除（const1 有外出控制边）
  EXPECT_EQ(const1_node->GetInControlNodesSize(), 1);
}

}  // namespace ge
