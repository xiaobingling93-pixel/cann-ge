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
#include <gmock/gmock.h>
#include "common/share_graph.h"
#include "es_ge_test_ops_c.h"
#include "es_ge_test_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"

#include "compiler/graph/fusion/node_matcher.h"
namespace ge {
namespace fusion {
class UtestConstNodeMatcher : public testing::Test {
public:
  static void SetUpTestSuite() {
    graph_ = EsCreateGraphBuilder("target");

    int32_t int32_scaler_const_data = 2;
    case_2_tensor_["int32_scaler"] = EsCreateScalarInt32(graph_, int32_scaler_const_data);

    int64_t int64_scaler_const_data = 2;
    case_2_tensor_["int64_scaler"] = EsCreateScalarInt64(graph_, int64_scaler_const_data);

    float float_scaler_const_data = 2;
    case_2_tensor_["float_scaler"] = EsCreateScalarFloat(graph_, float_scaler_const_data);

    std::vector<int32_t> int32_tensor_const_data(6, 2);
    std::vector<int64_t> int32_tensor_const_shape = {6};
    case_2_tensor_["int32_tensor"] = EsCreateConstInt32(
        graph_, int32_tensor_const_data.data(), int32_tensor_const_shape.data(), int32_tensor_const_shape.size());

    std::vector<int64_t> int32_tensor_const_shape_3_2 = {3,2};
    case_2_tensor_["int32_tensor_3_2"] = EsCreateConstInt32(
        graph_, int32_tensor_const_data.data(), int32_tensor_const_shape_3_2.data(), int32_tensor_const_shape_3_2.size());

    std::vector<int64_t> int64_tensor_const_data(6, 2);
    std::vector<int64_t> int64_tensor_const_shape = {6};
    case_2_tensor_["int64_tensor"] = EsCreateConstInt64(
        graph_, int64_tensor_const_data.data(), int64_tensor_const_shape.data(), int64_tensor_const_shape.size());

    std::vector<float> float_tensor_const_data(6, 2);
    std::vector<int64_t> float_tensor_const_shape = {6};
    case_2_tensor_["float_tensor"] = EsCreateConstFloat(
        graph_, float_tensor_const_data.data(), float_tensor_const_shape.data(), float_tensor_const_shape.size());

    std::vector<uint32_t> uint32_tensor_const_data(6, 2);
    std::vector<int64_t> uint32_tensor_const_shape = {6};
    case_2_tensor_["uint32_tensor"] = EsCreateConstUInt32(
        graph_, uint32_tensor_const_data.data(), uint32_tensor_const_shape.data(), uint32_tensor_const_shape.size());

    case_2_tensor_["relu"] = EsRelu(case_2_tensor_["int32_scaler"]);
  }
  static void TearDownTestSuite() {}

  NodePtr GetTargetNode(const std::string &case_name) {
    const auto esb_tensor = case_2_tensor_[case_name];
    if (esb_tensor != nullptr) {
      return NodeAdapter::GNode2Node(esb_tensor->GetProducer());
    }
    return nullptr;
  }

 private:
  static EsCGraphBuilder *graph_;
  static std::unordered_map<std::string, EsCTensorHolder *> case_2_tensor_;
};
EsCGraphBuilder *UtestConstNodeMatcher::graph_;
std::unordered_map<std::string, EsCTensorHolder *> UtestConstNodeMatcher::case_2_tensor_;

/**
 * 目标图上const, datatype int32, 值为2
 * pattern图上const, datatype int32, 值为3
 *
 * 预期： 匹配
 */
TEST_F(UtestConstNodeMatcher, DisableValueMatch_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 3);
  ConstantMatcher matcher(false, false);
  const auto target_node = GetTargetNode("int32_scaler");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}

/**
 * 目标图上relu
 * pattern图上const, datatype int32, 值为3
 *
 * 预期： 不匹配
 */
TEST_F(UtestConstNodeMatcher, DisableValueMatch_Miss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 3);
  ConstantMatcher matcher(false, false);
  const auto target_node = GetTargetNode("relu");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}

/**
 * 在子图中匹配data节点，其在根图上对应为scaler的constant,且value为0
 * pattern 中constant的值为2
 * 预期：可跨子图匹配
 */
TEST_F(UtestConstNodeMatcher, DisableValueMatch_Scaler_EnableCrossSubgraph_Match) {
  auto target_main_graph = gert::ShareGraph::IfGraphWithConstInput();
  auto target_graph = target_main_graph->GetSubgraph("then");
  auto target_node = target_graph->FindNode("data");
  EXPECT_STREQ(target_node->GetTypePtr(), "Data");

  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  ConstantMatcher matcher(false, true);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}

/**
 * 目标图上const, datatype int32, 值为2
 * pattern图上const, datatype int32, 值为2
 * enable_value_match = true
 *
 * 预期： 匹配
 */
TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Scaler_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_scaler");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Scaler_DataTypeMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto int64_scaler_const = EsCreateScalarInt64(pattern_graph_ptr, 2);
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_scaler");
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int64_scaler_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Scaler_ValueMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 3);
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_scaler");
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Tensor_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<int32_t> int32_tensor_const_data(6, 2);
  std::vector<int64_t> int32_tensor_const_shape = {6};
  auto int32_const = EsCreateConstInt32(pattern_graph_ptr, int32_tensor_const_data.data(),
                                        int32_tensor_const_shape.data(), int32_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_tensor");
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Tensor_ValueMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<int32_t> int32_tensor_const_data(6, 5);
  std::vector<int64_t> int32_tensor_const_shape = {6};
  auto int32_const = EsCreateConstInt32(pattern_graph_ptr, int32_tensor_const_data.data(),
                                        int32_tensor_const_shape.data(), int32_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_tensor");
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Tensor_ShapeMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<int32_t> int32_tensor_const_data(6, 2);
  std::vector<int64_t> int32_tensor_const_shape = {3, 2};
  auto int32_const = EsCreateConstInt32(pattern_graph_ptr, int32_tensor_const_data.data(),
                                        int32_tensor_const_shape.data(), int32_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_tensor");
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int32_Tensor_HighRank_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<int32_t> int32_tensor_const_data(6, 2);
  std::vector<int64_t> int32_tensor_const_shape = {3, 2};
  auto int32_const = EsCreateConstInt32(pattern_graph_ptr, int32_tensor_const_data.data(),
                                        int32_tensor_const_shape.data(), int32_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int32_tensor_3_2");
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int64_Scaler_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int64_scaler_const = EsCreateScalarInt64(pattern_graph_ptr, 2);
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int64_scaler");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int64_scaler_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int64_Scaler_ValueMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int64_scaler_const = EsCreateScalarInt64(pattern_graph_ptr, 88);
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int64_scaler");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int64_scaler_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Int64_Tensor_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<int64_t> int64_tensor_const_data(6, 2);
  std::vector<int64_t> int64_tensor_const_shape = {6};
  auto int64_const = EsCreateConstInt64(pattern_graph_ptr, int64_tensor_const_data.data(),
                                        int64_tensor_const_shape.data(), int64_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("int64_tensor");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int64_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Float_Scaler_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto float_scaler_const = EsCreateScalarFloat(pattern_graph_ptr, 2);
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("float_scaler");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(float_scaler_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Float_Tensor_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<float> float_tensor_const_data(6, 2);
  std::vector<int64_t> float_tensor_const_shape = {6};
  auto float_const = EsCreateConstFloat(pattern_graph_ptr, float_tensor_const_data.data(),
                                        float_tensor_const_shape.data(), float_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("float_tensor");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(float_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Float_Tensor_ValueMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<float> float_tensor_const_data(6, 4);
  std::vector<int64_t> float_tensor_const_shape = {6};
  auto float_const = EsCreateConstFloat(pattern_graph_ptr, float_tensor_const_data.data(),
                                        float_tensor_const_shape.data(), float_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("float_tensor");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(float_const->GetProducer()), target_node));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_Uint32_Tensor_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  std::vector<uint32_t> uint32_tensor_const_data(6, 2);
  std::vector<int64_t> uint32_tensor_const_shape = {6};
  auto uint32_const = EsCreateConstUInt32(pattern_graph_ptr, uint32_tensor_const_data.data(),
                                          uint32_tensor_const_shape.data(), uint32_tensor_const_shape.size());
  ConstantMatcher matcher(true, false);
  const auto target_node = GetTargetNode("uint32_tensor");
  EXPECT_TRUE(target_node != nullptr);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(uint32_const->GetProducer()), target_node));
}


TEST_F(UtestConstNodeMatcher, EnableValueMatch_TargetConst_NoWeight_Invalid_Miss) {
  auto pattern_graph_ = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph_.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  auto int32_scaler_const_target = EsCreateScalarInt32(pattern_graph_ptr, 2);

  GeTensorPtr weight_backup;
  AttrUtils::MutableTensor(NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weight_backup);


  AttrUtils::ClearAllAttrs(NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())->GetOpDesc());
  ConstantMatcher matcher(true, false);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())));
}
// todo 预期行为 是否要改变？ 使能值匹配，但pattern const 没有值
TEST_F(UtestConstNodeMatcher, EnableValueMatch_PatternConst_NoWeight_Match) {
  auto pattern_graph_ = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph_.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  auto int32_scaler_const_target = EsCreateScalarInt32(pattern_graph_ptr, 2);

  GeTensorPtr weight_backup;
  AttrUtils::MutableTensor(NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weight_backup);


  ConstantMatcher matcher(true, false);
  AttrUtils::ClearAllAttrs(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer())->GetOpDesc());
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_ShapeNotEqual_Invalid_Miss) {
  auto pattern_graph_ = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph_.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  auto int32_scaler_const_target = EsCreateScalarInt32(pattern_graph_ptr, 2);

  GeTensorPtr weight_backup;
  AttrUtils::MutableTensor(NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weight_backup);
  weight_backup->MutableTensorDesc().SetShape(GeShape({2,3}));

  ConstantMatcher matcher(true, false);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())));
}

TEST_F(UtestConstNodeMatcher, EnableValueMatch_DataSizeNotEqual_Invalid_Miss) {
  auto pattern_graph_ = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph_.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  auto int32_scaler_const_target = EsCreateScalarInt32(pattern_graph_ptr, 2);

  GeTensorPtr weight_backup;
  AttrUtils::MutableTensor(NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weight_backup);
  std::vector<int32_t> tmp_weight = {2, 3};
  weight_backup->MutableData().SetData(reinterpret_cast<const uint8_t *const>(tmp_weight.data()), 8);

  ConstantMatcher matcher(true, false);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), NodeAdapter::GNode2Node(int32_scaler_const_target->GetProducer())));
}
/**
* 当前target节点为data,其真实引用节点非const，因此类型匹配失败
*/
TEST_F(UtestConstNodeMatcher, EnableValueMatch_EnableCrossSubgraph_NodeTypeMiss) {
  auto target_main_graph = gert::ShareGraph::IfGraph();
  auto target_graph = target_main_graph->GetSubgraph("then");
  auto target_node = target_graph->FindNode("data");
  EXPECT_STREQ(target_node->GetTypePtr(), "Data");

  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 2);
  ConstantMatcher matcher(true, true);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}
/**
* 在子图中匹配data节点，其在根图上对应为scaler的constant,且value为0
*/
TEST_F(UtestConstNodeMatcher, EnableValueMatch_Scaler_EnableCrossSubgraph_Match) {
  auto target_main_graph = gert::ShareGraph::IfGraphWithConstInput();
  auto target_graph = target_main_graph->GetSubgraph("then");
  auto target_node = target_graph->FindNode("data");
  EXPECT_STREQ(target_node->GetTypePtr(), "Data");

  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();

  auto int32_scaler_const = EsCreateScalarInt32(pattern_graph_ptr, 0);
  ConstantMatcher matcher(true, true);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(int32_scaler_const->GetProducer()), target_node));
}
} // namespace fusion
} // namespace ge