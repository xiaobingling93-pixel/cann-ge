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
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"

#include "stub/gert_runtime_stub.h"

#include "ge/fusion/pass/decompose_pass.h"
#include "ge/fusion/pattern.h"

#include "common/topo_checker.h"
#include "register/custom_pass_context_impl.h"

namespace ge {
namespace fusion {
using namespace ge::es;
class UtestDecomposePass : public testing::Test {
public:
  static void SetUpTestSuite() {
  }
  static void TearDownTestSuite() {}
};
/**
 * single node match
 *      data
 *        |
 *     transdata
 *        |
 *       out
 */
TEST_F(UtestDecomposePass, SingleNode_1Input_1Output) {
  // define pass
  class TransDataToReluPass : public DecomposePass {
  TransDataToReluPass(const std::vector<AscendString> &op_types): DecomposePass(op_types) {}
  protected:
    std::unique_ptr<Graph> Replacement(const GNode &matched_node) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);

  CustomPassContext context;
  TransDataToReluPass transdata_2_relu_pass({TRANSDATA});
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), SUCCESS);

  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {"Relu"}, {"Relu"}, {CONSTANT}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(1, {{"Relu"}}), "success");
      EXPECT_EQ(checker.StrictConnectTo(2, {{"Relu"}}), "success");
    }
    if (node->GetName() == "x_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectTo(0, {{"Relu"}}), "success");
    }
    if (node->GetName() == "y_reshape") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}}), "success");
    }
    if (node->GetType() == "Relu") {
      std::cout << node->GetName() << std::endl;
      // check origin dump op attr
      std::vector<std::string> origin_types;
      const bool has_origin_op_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_types);
      EXPECT_TRUE(has_origin_op_attr);
      EXPECT_TRUE(origin_types.size() == 1);
      EXPECT_STREQ(origin_types[0].c_str(), TRANSDATA);
      // check pass_name attr
      const std::string kPassName = "pass_name";
      std::vector<std::string> pass_names;
      const bool has_pass_name_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), kPassName, pass_names);
      EXPECT_TRUE(has_pass_name_attr);
      EXPECT_TRUE(pass_names.size() == 1);
      EXPECT_STREQ(pass_names[0].c_str(), "ge::fusion::UtestDecomposePass_SingleNode_1Input_1Output_Test::TestBody()::TransDataToReluPass");
    }
  }
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), NOT_CHANGED);
}

TEST_F(UtestDecomposePass, NotMeetRequirement_NOT_CHANGE) {
  // define pass
  class TransDataToReluPass : public DecomposePass {
   public:
    TransDataToReluPass(const std::vector<AscendString> &op_type) : DecomposePass(op_type) {}
   protected:
    bool MeetRequirements(const GNode &matched_node) override {
      return false;
    }
    std::unique_ptr<Graph> Replacement(const GNode &matched_node) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  std::vector<GNode> target_nodes;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == TRANSDATA) {
      target_nodes.emplace_back(NodeAdapter::Node2GNode(node));
    }
  }
  CustomPassContext context;
  TransDataToReluPass transdata_2_relu_pass({TRANSDATA});
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), NOT_CHANGED);
}
TEST_F(UtestDecomposePass, ReplacementInvalid_Failed) {
  // define pass
  class TransDataToReluPass : public DecomposePass {
    TransDataToReluPass(const std::vector<AscendString> &op_types): DecomposePass(op_types) {}
   protected:
    bool MeetRequirements(const GNode &matched_node) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const GNode &matched_node) override { // WRONG REPLACEMENT
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto data1 = EsCreateGraphInput(esb_graph, 1);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      esb_graph->SetGraphOutput(data1, 1);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  CustomPassContext context;
  TransDataToReluPass transdata_2_relu_pass({TRANSDATA});
  auto ret = transdata_2_relu_pass.Run(target_graph, context);
  EXPECT_NE(ret, NOT_CHANGED);
  EXPECT_NE(ret, SUCCESS);
}
} // namespace fusion
} // namespace ge