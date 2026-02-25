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
class UtestDataNodeMatcher : public testing::Test {
 public:
  static void SetUpTestSuite() {
    graph_ = EsCreateGraphBuilder("target");

    int32_t int32_scaler_const_data = 2;
    node_2_tensor_["const"] = EsCreateScalarInt32(graph_, int32_scaler_const_data);
    node_2_tensor_["input"] = EsCreateGraphInput(graph_, 0);
    node_2_tensor_["add"] = EsAdd(node_2_tensor_["input"], node_2_tensor_["const"]);
  }
  static void TearDownTestSuite() {}

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
};
EsCGraphBuilder *UtestDataNodeMatcher::graph_;
std::unordered_map<std::string, EsCTensorHolder *> UtestDataNodeMatcher::node_2_tensor_;

TEST_F(UtestDataNodeMatcher, NormalNode_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);

  auto target_node = GetTargetNode("add");
  DataMatcher matcher;
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(input_tensor->GetProducer()), target_node));
}

TEST_F(UtestDataNodeMatcher, Constant_Match) {
  auto pattern_graph = std::unique_ptr<EsCGraphBuilder, void (*)(EsCGraphBuilder *)>(EsCreateGraphBuilder("pattern"), EsDestroyGraphBuilder);
  auto pattern_graph_ptr = pattern_graph.get();
  auto input_tensor = EsCreateGraphInput(pattern_graph_ptr, 0);

  auto target_node = GetTargetNode("const");
  DataMatcher matcher;
  EXPECT_TRUE(matcher.IsMatch(NodeAdapter::GNode2Node(input_tensor->GetProducer()), target_node));
}
} // namespace fusion
} // namespace ge