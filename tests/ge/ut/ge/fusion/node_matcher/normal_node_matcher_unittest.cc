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
#include "es_ge_test_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "fusion/utils/log_checker.h"
#include "graph/utils/node_adapter.h"

#include "compiler/graph/fusion/node_matcher.h"
namespace ge {
namespace fusion {
class UtestNormalNodeMatcher : public testing::Test {
public:
  void SetUp() override {
    dlog_setlevel(GE_MODULE_NAME, 2, 0);
  }
  void TearDown() override {
    runtime_stub_.Clear();
    dlog_setlevel(GE_MODULE_NAME, 3, 0);
  }
  static void SetUpTestSuite() {
    graph_ = EsCreateGraphBuilder("target");

    int32_t int32_scaler_const_data = 2;
    node_2_tensor_["const"] = EsCreateScalarInt32(graph_, int32_scaler_const_data);
    node_2_tensor_["input"] = EsCreateGraphInput(graph_, 0);
    node_2_tensor_["add"] = EsAdd(node_2_tensor_["input"], node_2_tensor_["const"]);
    node_2_tensor_["cast_to_fp16"] = EsCast(node_2_tensor_["input"], DT_FLOAT16);
    std::vector<int64_t> list_int_attr = {1,2,3};
    node_2_tensor_["node_with_list_int_attr"] =
        EsCompress(node_2_tensor_["input"], reinterpret_cast<int64_t *>(list_int_attr.data()), 3).weight_compress;

  }
  static void TearDownTestSuite() {
  }

  NodePtr GetTargetNode(const std::string &case_name) {
    const auto esb_tensor = node_2_tensor_[case_name];
    if (esb_tensor != nullptr) {
      return NodeAdapter::GNode2Node(esb_tensor->GetProducer());
    }
    return nullptr;
  }

 private:
  static EsCGraphBuilder *graph_;
  static std::unordered_map<std::string, EsCTensorHolder *> node_2_tensor_;
  gert::GertRuntimeStub runtime_stub_;
};
EsCGraphBuilder *UtestNormalNodeMatcher::graph_;
std::unordered_map<std::string, EsCTensorHolder *> UtestNormalNodeMatcher::node_2_tensor_;

TEST_F(UtestNormalNodeMatcher, DisableIrAttr_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  auto input_tensor1 = EsCreateGraphInput(pattern_graph_ptr, 1);
  auto add_tensor = EsAdd(input_tensor, input_tensor1);

  auto target_node = GetTargetNode("add");
  NormalNodeMatcher matcher(false);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(add_tensor->GetProducer()), target_node));
}

TEST_F(UtestNormalNodeMatcher, DisableIrAttr_Miss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  auto input_tensor1 = EsCreateGraphInput(pattern_graph_ptr, 1);
  auto add_tensor = EsAdd(input_tensor, input_tensor1);

  auto target_node = GetTargetNode("const");
  NormalNodeMatcher matcher(false);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(add_tensor->GetProducer()), target_node));
}

TEST_F(UtestNormalNodeMatcher, EnableIrAttr_SingleAttr_TypeAttr) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  auto cast_tensor_match = EsCast(input_tensor, DT_FLOAT16);
  auto cast_tensor_miss = EsCast(input_tensor, DT_INT32);

  auto target_node = GetTargetNode("cast_to_fp16");
  NormalNodeMatcher matcher(true);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(cast_tensor_match->GetProducer()), target_node));
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(cast_tensor_miss->GetProducer()), target_node));
}

TEST_F(UtestNormalNodeMatcher, EnableIrAttr_SingleAttr_FloatAttr) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  auto cast_tensor_match = EsCast(input_tensor, DT_FLOAT16);
  auto cast_tensor_miss = EsCast(input_tensor, DT_INT32);

  auto target_node = GetTargetNode("cast_to_fp16");
  NormalNodeMatcher matcher(true);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(cast_tensor_match->GetProducer()), target_node));
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(cast_tensor_miss->GetProducer()), target_node));
}

TEST_F(UtestNormalNodeMatcher, EnableIrAttr_SingleAttr_ListIntAttr) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  std::vector<int64_t > list_int_value = {4,5,6};
  std::vector<int64_t > list_int_value_right = {1,2,3};
  auto compress_tensor_miss =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value.data()), 3).weight_compress;
  auto compress_tensor_match =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value_right.data()), 3).weight_compress;

  auto target_node = GetTargetNode("node_with_list_int_attr");
  NormalNodeMatcher matcher(true);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer()), target_node));
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(compress_tensor_match->GetProducer()), target_node));
}

// compress找不到ir， ir recover自动返回成功了
// 为pattern中的compress构造一个不存在的IR attr name，构造pattern ir attr name和target ir attr name数量不一样的场景
TEST_F(UtestNormalNodeMatcher, EnableIrAttr_SingleAttr_ListIntAttr_IrNameNumMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  std::vector<int64_t > list_int_value_right = {1,2,3};
  auto compress_tensor_miss =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value_right.data()), 3).weight_compress;

  NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer())->GetOpDesc()->AppendIrAttrName("fake_attr_name");
  auto target_node = GetTargetNode("node_with_list_int_attr");
  NormalNodeMatcher matcher(true);

  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer()), target_node));
  EXPECT_TRUE(ut::WarnLogContain(runtime_stub_, "Ir attr num is not match"));
}

// compress找不到ir， ir recover自动返回成功了
// 为pattern中的compress构造一个不存在的IR attr name a
// 为target中的compress构造一个不存在的IR attr name b
// 构造pattern ir attr name和target ir attr name不一样的场景
TEST_F(UtestNormalNodeMatcher, EnableIrAttr_SingleAttr_ListIntAttr_IrNameWrongMiss) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  std::vector<int64_t > list_int_value_right = {1,2,3};
  auto compress_tensor_miss =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value_right.data()), 3).weight_compress;

  auto compress_tensor_target =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value_right.data()), 3).weight_compress;

  NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer())->GetOpDesc()->AppendIrAttrName("fake_attr_a");
  NodeAdapter::GNode2Node(compress_tensor_target->GetProducer())->GetOpDesc()->AppendIrAttrName("fake_attr_b");

  NormalNodeMatcher matcher(true);
  EXPECT_FALSE(matcher.IsMatch(NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer()), NodeAdapter::GNode2Node(compress_tensor_target->GetProducer())));
  EXPECT_TRUE(ut::WarnLogContain(runtime_stub_, "Ir attr names is not match"));
}

// compress找不到ir， ir recover自动返回成功了
// 为pattern中的compress构造一个不存在的IR attr name
// 为target中的compress构造一个不存在的IR attr name
// 构造pattern ir attr name和target ir attr name一样的场景
TEST_F(UtestNormalNodeMatcher, EnableIrAttr_MultiAttr_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);
  std::vector<int64_t > list_int_value_right = {1,2,3};
  auto compress_tensor_miss =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value_right.data()), 3).weight_compress;

  auto compress_tensor_target =
      EsCompress(input_tensor, reinterpret_cast<int64_t *>(list_int_value_right.data()), 3).weight_compress;

  const std::string fake_attr_name = "fake_attr";
  NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer())->GetOpDesc()->AppendIrAttrName(fake_attr_name);
  AttrUtils::SetFloat(NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer())->GetOpDesc(), fake_attr_name, 2);
  NodeAdapter::GNode2Node(compress_tensor_target->GetProducer())->GetOpDesc()->AppendIrAttrName(fake_attr_name);
  AttrUtils::SetFloat(NodeAdapter::GNode2Node(compress_tensor_target->GetProducer())->GetOpDesc(), fake_attr_name, 2);

  NormalNodeMatcher matcher(true);
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(compress_tensor_miss->GetProducer()), NodeAdapter::GNode2Node(compress_tensor_target->GetProducer())));
}
} // namespace fusion
} // namespace ge