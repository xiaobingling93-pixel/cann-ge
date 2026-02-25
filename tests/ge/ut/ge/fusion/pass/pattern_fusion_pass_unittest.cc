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
#include "graph/utils/op_desc_utils.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "stub/gert_runtime_stub.h"

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/pattern.h"

#include "common/topo_checker.h"
#include "register/custom_pass_context_impl.h"
#include "ge/ge_utils.h"
#include "graph/utils/connection_matrix_impl.h"

namespace ge {
namespace fusion {
using namespace ge::es;
class UtestPatternFusionPass : public testing::Test {
 public:
  static void SetUpTestSuite() {}
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
TEST_F(UtestPatternFusionPass, SingleNode_1Input_1Output) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
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
  TransDataToReluPass transdata_2_relu_pass;
  CustomPassContext context;
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), SUCCESS);

  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {"Relu"}, {"Relu"}, {CONSTANT}}),
                "success");
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
      // check attr
      std::vector<std::string> origin_types;
      const bool has_origin_op_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_types);
      EXPECT_TRUE(has_origin_op_attr);
      EXPECT_TRUE(origin_types.size() == 1);
      EXPECT_STREQ(origin_types[0].c_str(), TRANSDATA);
    }
  }

  // check not changed
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), NOT_CHANGED);
}

TEST_F(UtestPatternFusionPass, SingleNode_1Input_1Output_AfterInfershape) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph_builder = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph_builder.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      auto repalcement_graph = replace_graph_builder.BuildAndReset();
      // mock infer func
      const auto stub_infer_func = [](Operator &op) {
        auto input_shape = op.GetInputDesc(0).GetShape();
        auto output_desc = op.GetOutputDesc(0);
        output_desc.SetShape(input_shape);
        op.UpdateOutputDesc((int)0, output_desc);
        return GRAPH_SUCCESS;
      };
      for (const auto &node : GraphUtilsEx::GetComputeGraph(*repalcement_graph)->GetDirectNode()) {
        node->GetOpDesc()->AddInferFunc(stub_infer_func);
      }

      std::vector<SubgraphInput> subgraph_inputs;
      match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
      std::vector<ge::Shape> input_shapes;
      for (const auto &subgraph_input : subgraph_inputs) {
        auto boundary_input = subgraph_input.GetAllInputs();

        auto node_output = boundary_input.at(0);
        TensorDesc tensor_desc;
        node_output.node.GetOutputDesc(node_output.index, tensor_desc);
        input_shapes.emplace_back(tensor_desc.GetShape());
      }
      GE_ASSERT_SUCCESS(GeUtils::InferShape(*repalcement_graph, input_shapes));
      return repalcement_graph;
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  TransDataToReluPass transdata_2_relu_pass;
  CustomPassContext context;
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), SUCCESS);

  GraphUtils::DumpGEGraphToOnnx(*target_compute_graph, "after_replace");
  bool find_transdata_4 = false;
  for (const auto &node : target_compute_graph->GetDirectNode()) {
    if (node->GetType() == "DynamicRNNV3") {
      auto checker = gert::NodeTopoChecker(node);
      EXPECT_EQ(checker.StrictConnectFrom({{"Relu"}, {CONSTANT}, {CONSTANT}, {"Relu"}, {"Relu"}, {CONSTANT}}),
                "success");
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
      // check attr
      std::vector<std::string> origin_types;
      const bool has_origin_op_attr =
          ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_types);
      EXPECT_TRUE(has_origin_op_attr);
      EXPECT_TRUE(origin_types.size() == 1);
      EXPECT_STREQ(origin_types[0].c_str(), TRANSDATA);

      std::vector<std::string> origin_op_names;
      ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, origin_op_names);

      if (origin_op_names[0] == "transdata_4") {
        std::vector<int64_t> expect_shape = {1, -1, 256};
        find_transdata_4 = true;
        // check shape
        const auto input_shape = node->GetOpDesc()->GetInputDesc(0).GetShape();
        EXPECT_EQ(input_shape.GetDimNum(), expect_shape.size());
        for (size_t i = 0u; i < expect_shape.size(); ++i) {
          EXPECT_EQ(input_shape.GetDim(i), expect_shape[i]);
        }
        const auto output_shape = node->GetOpDesc()->GetOutputDesc(0).GetShape();
        EXPECT_EQ(output_shape.GetDimNum(), expect_shape.size());
        for (size_t i = 0u; i < expect_shape.size(); ++i) {
          EXPECT_EQ(output_shape.GetDim(i), expect_shape[i]);
        }
      }
    }
  }
  EXPECT_TRUE(find_transdata_4);
}

TEST_F(UtestPatternFusionPass, NotMeetRequirement) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return false;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
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
  TransDataToReluPass transdata_2_relu_pass;
  CustomPassContext context;
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), NOT_CHANGED);
}

TEST_F(UtestPatternFusionPass, RepalceFailed) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto data1 = EsCreateGraphInput(esb_graph, 1);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      esb_graph->SetGraphOutput(data1, 0);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  TransDataToReluPass transdata_2_relu_pass;
  CustomPassContext context;
  EXPECT_NE(transdata_2_relu_pass.Run(target_graph, context), SUCCESS);
  EXPECT_NE(transdata_2_relu_pass.Run(target_graph, context), NOT_CHANGED);
}

/**
 * single node match
 *      data
 *        |
 *     transdata
 *        |
 *       out
 */
TEST_F(UtestPatternFusionPass, SingleNode_1Input_1Output_EnableIrAttrMatch_NotChange) {
  // define pass
  class TransDataToReluPass : public PatternFusionPass {
   public:
    TransDataToReluPass() : PatternFusionPass(PatternMatcherConfigBuilder().EnableIrAttrMatch().Build()) {}

   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      // build pattern graph
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
      esb_graph->SetGraphOutput(transdata, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
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
  TransDataToReluPass transdata_2_relu_pass;
  CustomPassContext context;
  EXPECT_EQ(transdata_2_relu_pass.Run(target_graph, context), NOT_CHANGED);
}

// target graph:                       pattern_graph:
//      data0                                data0
//        |                                    |
//       abs      data1                       abs
//      /    \c   /                            |
//    relu     relu1                          relu
//      \     /c  |                            |
//        abs1    |                           abs1
//            \   /                            |
//            add                           netoutput
//             |
//          netoutput
TEST_F(UtestPatternFusionPass, CycleMakerPass_NotChange) {
  // define target graph
  auto target_graph_builder = ge::es::EsGraphBuilder("target");
  auto esb_target_graph = target_graph_builder.GetCGraphBuilder();
  auto data0 = EsCreateGraphInput(esb_target_graph, 0);
  auto data1 = EsCreateGraphInput(esb_target_graph, 1);
  auto abs = EsAbs(data0);
  auto relu = EsRelu(abs);
  auto abs1 = EsAbs(relu);
  auto relu1 = EsRelu(data1);
  auto add = EsAdd(abs1, relu1);
  GraphUtils::AddEdge(NodeAdapter::GNode2Node(abs->GetProducer())->GetOutControlAnchor(), NodeAdapter::GNode2Node(relu1->GetProducer())->GetInControlAnchor());
  GraphUtils::AddEdge(NodeAdapter::GNode2Node(relu1->GetProducer())->GetOutControlAnchor(), NodeAdapter::GNode2Node(abs1->GetProducer())->GetInControlAnchor());
  esb_target_graph->SetGraphOutput(add, 0);
  auto target_graph = std::make_shared<Graph>(*target_graph_builder.BuildAndReset());

  class CycleMakerPass : public PatternFusionPass {
   public:
    //CycleMakerPass() : PatternFusionPass() {}
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto abs1 = EsAbs(EsRelu(EsAbs(data)));
      esb_graph->SetGraphOutput(abs1, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto relu = EsRelu(data);
      esb_graph->SetGraphOutput(relu, 0);
      return replace_graph.BuildAndReset();
    }
  };

  CycleMakerPass cycle_maker_pass;
  CustomPassContext context;
  EXPECT_EQ(cycle_maker_pass.Run(target_graph, context), NOT_CHANGED);
}

TEST_F(UtestPatternFusionPass, CaptureTensors) {
  // define pass
  class TestPassForCaptureTensors : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      // build pattern graph
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ::ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      auto data1 = EsCreateGraphInput(esb_graph, 1);
      auto add_tensor = EsAdd(data, data1);
      esb_graph->SetGraphOutput(add_tensor, 0);
      auto graph = pattern_graph.BuildAndReset();
      auto pattern = std::make_unique<Pattern>(std::move(*graph));

      pattern->CaptureTensor({NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(data->GetProducer())), 0})
          .CaptureTensor({NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(data1->GetProducer())), 0});
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      NodeIo capture_io_0;
      NodeIo capture_io_1;
      EXPECT_EQ(match_result->GetCapturedTensor(0, capture_io_0), SUCCESS);
      EXPECT_EQ(match_result->GetCapturedTensor(1, capture_io_1), SUCCESS);
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ::ge::es::EsGraphBuilder("pattern");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      esb_graph->SetGraphOutput(data, 0);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::AicoreGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  TestPassForCaptureTensors test_pass_for_capture_tensors;
  CustomPassContext context;
  test_pass_for_capture_tensors.Run(target_graph, context);
}

TEST_F(UtestPatternFusionPass, CaptureData) {
  // define pass
  class TestPassForCaptureTensors : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      // build pattern graph
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ::ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
      std::vector<int64_t> x_reshape_shape({3});
      auto shape_const =
          EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
      auto reshape = EsReshape(data, shape_const, 0, 0);
      esb_graph->SetGraphOutput(reshape, 0);
      auto graph = pattern_graph.BuildAndReset();
      auto pattern = std::make_unique<Pattern>(std::move(*graph));

      pattern->CaptureTensor({NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(shape_const->GetProducer())), 0});
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      NodeIo capture_io;
      EXPECT_EQ(match_result->GetCapturedTensor(0, capture_io), SUCCESS);
      auto capture_node = capture_io.node;
      TensorDesc tensor_desc;
      EXPECT_EQ(capture_node.GetOutputDesc(0, tensor_desc), GRAPH_SUCCESS);
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ::ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      esb_graph->SetGraphOutput(data, 0);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  TestPassForCaptureTensors test_pass_for_capture_tensors;
  CustomPassContext context;
  test_pass_for_capture_tensors.Run(target_graph, context);
}

TEST_F(UtestPatternFusionPass, GetPatternName) {
  // define pass
  class TestPassForCaptureTensors : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      // build pattern graph
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ::ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      std::vector<int64_t> x_reshape_const_data({-1, 1, 256});
      std::vector<int64_t> x_reshape_shape({3});
      auto shape_const =
          EsCreateConstInt64(esb_graph, x_reshape_const_data.data(), x_reshape_shape.data(), x_reshape_shape.size());
      auto reshape = EsReshape(data, shape_const, 0, 0);
      esb_graph->SetGraphOutput(reshape, 0);
      auto graph = pattern_graph.BuildAndReset();
      auto pattern = std::make_unique<Pattern>(std::move(*graph));

      pattern->CaptureTensor({NodeAdapter::Node2GNode(NodeAdapter::GNode2Node(shape_const->GetProducer())), 0});
      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      auto pattern_name = match_result->GetPatternGraph().GetName();
      EXPECT_EQ(pattern_name, "pattern");
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph = ::ge::es::EsGraphBuilder("replacement");
      auto esb_graph = replace_graph.GetCGraphBuilder();
      auto data = EsCreateGraphInput(esb_graph, 0);
      esb_graph->SetGraphOutput(data, 0);
      return replace_graph.BuildAndReset();
    }
  };

  auto target_compute_graph = gert::ShareGraph::LstmpGraph();
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  TestPassForCaptureTensors test_pass_for_capture_tensors;
  CustomPassContext context;
  test_pass_for_capture_tensors.Run(target_graph, context);
}

TEST_F(UtestPatternFusionPass, ReplaceOutput_AutoUpdateOutput) {
  // define pass
  class TestPassForReplaceOutput : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      // build pattern graph
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data_1 = EsCreateGraphInput(esb_graph, 0);
      auto data_2 = EsCreateGraphInput(esb_graph, 1);
      auto add = EsAdd(data_1, data_2);
      esb_graph->SetGraphOutput(add, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));

      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      auto pattern_name = match_result->GetPatternGraph().GetName();
      EXPECT_EQ(pattern_name, "pattern");
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph_builder = ge::es::EsGraphBuilder("replace");
      auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
      auto data_r_1 = EsCreateGraphInput(replace_esb_graph, 0);
      auto data_r_2 = EsCreateGraphInput(replace_esb_graph, 1);
      auto sub_r = EsSub(data_r_1, data_r_2);
      replace_esb_graph->SetGraphOutput(sub_r, 0);
      return replace_graph_builder.BuildAndReset();
    }
  };

  // build target graph
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("add"));
  };

  auto target_compute_graph = ToComputeGraph(g1);
  auto net_output = target_compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_EQ(net_output, nullptr);
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  auto compute_output_node = target_compute_graph->FindFirstNodeMatchType("Add");
  auto output_gnode = NodeAdapter::Node2GNode(compute_output_node);
  ASSERT_EQ(target_graph->SetOutputs({{output_gnode, {0}}}), SUCCESS);

  TestPassForReplaceOutput test_pass_for_capture_tensors;
  CustomPassContext context;
  test_pass_for_capture_tensors.Run(target_graph, context);

  net_output = target_compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(net_output, nullptr);
  ASSERT_EQ(net_output->GetInDataNodes().size(), 1U);
  ASSERT_EQ(net_output->GetInDataNodes().at(0)->GetType(), "Sub");
}

TEST_F(UtestPatternFusionPass, ReplceTarget_AutoUpdateTarget) {
  // define pass
  class TestPassForReplaceOutput : public PatternFusionPass {
   protected:
    std::vector<PatternUniqPtr> Patterns() override {
      // build pattern graph
      std::vector<PatternUniqPtr> patterns;
      auto pattern_graph = ge::es::EsGraphBuilder("pattern");
      auto esb_graph = pattern_graph.GetCGraphBuilder();
      auto data_1 = EsCreateGraphInput(esb_graph, 0);
      auto data_2 = EsCreateGraphInput(esb_graph, 1);
      auto add = EsAdd(data_1, data_2);
      esb_graph->SetGraphOutput(add, 0);
      auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));

      patterns.emplace_back(std::move(pattern));
      return patterns;
    }
    bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
      auto pattern_name = match_result->GetPatternGraph().GetName();
      EXPECT_EQ(pattern_name, "pattern");
      return true;
    }
    std::unique_ptr<Graph> Replacement(const unique_ptr<MatchResult> &match_result) override {
      auto replace_graph_builder = ge::es::EsGraphBuilder("replace");
      auto replace_esb_graph = replace_graph_builder.GetCGraphBuilder();
      auto data_r_1 = EsCreateGraphInput(replace_esb_graph, 0);
      auto data_r_2 = EsCreateGraphInput(replace_esb_graph, 1);
      auto sub_r = EsSub(data_r_1, data_r_2);
      replace_esb_graph->SetGraphOutput(sub_r, 0);
      return replace_graph_builder.BuildAndReset();
    }
  };

  // build target graph
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("add"));
  };

  auto target_compute_graph = ToComputeGraph(g1);
  auto net_output = target_compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_EQ(net_output, nullptr);
  auto compute_output_node = target_compute_graph->FindFirstNodeMatchType("Add");
  auto target_operator = OpDescUtils::CreateOperatorFromNode(compute_output_node);
  auto target_graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(target_compute_graph);
  target_graph->SetTargets({target_operator});
  net_output = target_compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  ASSERT_NE(net_output, nullptr);

  TestPassForReplaceOutput test_pass_for_capture_tensors;
  CustomPassContext context;
  test_pass_for_capture_tensors.Run(target_graph, context);

  auto sub = target_compute_graph->FindFirstNodeMatchType(SUB);
  ASSERT_NE(sub, nullptr);
  ASSERT_EQ(sub->GetOutControlNodes().size(), 1U);
  ASSERT_EQ(sub->GetOutControlNodes().at(0)->GetOutControlNodes().at(0)->GetType(), NETOUTPUT);
}
}  // namespace fusion
}  // namespace ge
