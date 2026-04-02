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
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "ge_graph_dsl/assert/check_utils.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "init_ge.h"
#include "common/memory/tensor_trans_utils.h"
#include "ge/ge_api_v2.h"
#include "graph/ge_global_options.h"

using namespace std;
using namespace ge;
namespace {
graphStatus InferFunctionStub(Operator &op) { return GRAPH_SUCCESS; }
/**
 * 
 *       Variable(2, 3, 4, 5)                      Relu3(1,2,3,4,5)
 *          /            \                             /        \
 *     TransData      TransData                    TransData    summary
 *         |              |                            |
 *   Relu(1,2,3,4,5)  Relu(1,2,3,4,5)            Variable(2, 3, 4, 5)
 *          \            /                             |
 *             NetOutput -------------------------------
 * 
 */
Graph BuildVariableGraph() {
  GeTensorDesc tensor_4_desc(ge::GeShape({2,3,4,5}), FORMAT_NCHW, DT_INT32);
  GeTensorDesc tensor_5_desc(ge::GeShape({1,2,3,4,5}), FORMAT_NC1HWC0, DT_INT32);

  auto var1 = std::make_shared<OpDesc>("var1", VARIABLE);
  var1->AddInputDesc(tensor_4_desc);
  var1->AddOutputDesc(tensor_4_desc);
  var1->AddInferFunc(InferFunctionStub);

  auto trans1 = std::make_shared<OpDesc>("transdata1", TRANSDATA);
  trans1->AddInputDesc(tensor_4_desc);
  trans1->AddOutputDesc(tensor_5_desc);
  trans1->AddInferFunc(InferFunctionStub);

  auto trans2 = std::make_shared<OpDesc>("transdata2", TRANSDATA);
  trans2->AddInputDesc(tensor_4_desc);
  trans2->AddOutputDesc(tensor_5_desc);
  trans2->AddInferFunc(InferFunctionStub);

  auto trans3 = std::make_shared<OpDesc>("transdata3", TRANSDATA);
  trans3->AddInputDesc(tensor_5_desc);
  trans3->AddOutputDesc(tensor_4_desc);
  trans3->AddInferFunc(InferFunctionStub);

  auto data1 = std::make_shared<OpDesc>("data1", DATA);
  (void)AttrUtils::SetInt(data1, ATTR_NAME_INDEX, 0);
  data1->AddInputDesc(tensor_5_desc);
  data1->AddOutputDesc(tensor_5_desc);
  data1->AddInferFunc(InferFunctionStub);

  auto relu1 = std::make_shared<OpDesc>("relu1", RELU);
  relu1->AddInputDesc(tensor_5_desc);
  relu1->AddOutputDesc(tensor_5_desc);
  relu1->AddInferFunc(InferFunctionStub);

  auto relu2 = std::make_shared<OpDesc>("relu2", RELU);
  relu2->AddInputDesc(tensor_5_desc);
  relu2->AddOutputDesc(tensor_5_desc);
  relu2->AddInferFunc(InferFunctionStub);

  auto relu3 = std::make_shared<OpDesc>("relu3", RELU);
  relu3->AddInputDesc(tensor_5_desc);
  relu3->AddOutputDesc(tensor_5_desc);
  relu3->AddInferFunc(InferFunctionStub);

  auto var_ref = std::make_shared<OpDesc>("var_ref", VARIABLE);
  AttrUtils::SetStr(var_ref, REF_VAR_SRC_VAR_NAME, "var1");
  var_ref->AddInputDesc(tensor_4_desc);
  var_ref->AddOutputDesc(tensor_4_desc);
  var_ref->AddInferFunc(InferFunctionStub);

  auto summary = std::make_shared<OpDesc>("summary", "Summary");
  summary->AddInputDesc(tensor_5_desc);
  summary->AddOutputDesc(tensor_5_desc);
  summary->AddInferFunc(InferFunctionStub);

  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE(trans1)->NODE(relu1)->NODE("output", NETOUTPUT));
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE(trans2)->NODE(relu2)->NODE("output"));
    CHAIN(NODE(data1)->NODE(relu3)->NODE(trans3)->NODE(var_ref)->NODE("output"));
    CHAIN(NODE(relu3)->NODE(summary));
    ADD_OUTPUT(relu1, 0);
    ADD_OUTPUT(relu2, 0);
    ADD_OUTPUT(var_ref, 0);
  };
  return ToGeGraph(g1);
}
}  // namespace
class SummaryAndCheckPointTest : public testing::Test {
 protected:
  void SetUp() {
    ReInitGe();
  }
  void TearDown() {}
};

// after run graph, call GetVariables to generate checkpoint graph
TEST_F(SummaryAndCheckPointTest, test_generate_and_run_checkpoint_graph) {
  auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(EVALUATE_GRAPH_RESOURCE_MODE);
  if (it != global_options.end()) {
    global_options.erase(it);
  }
  // build graph
  Graph graph = BuildVariableGraph();

  // new session & add graph
  map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = AscendString("1"); // train flag
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(graph_id, inputs);
  EXPECT_EQ(ret, SUCCESS);

  DUMP_GRAPH_WHEN("PreRunBegin");
  // register Save call back func
  static uint32_t called_count = 0;
  ge::session::pCallBackFunc callbackFuncSave = [](uint32_t graph_id, const std::map<AscendString, ge::Tensor> &params_list) {
    called_count++;
    return SUCCESS;
  };
  ret = session.RegisterCallBackFunc("Save", callbackFuncSave);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<AscendString> var_names = {"var1"};
  std::vector<Tensor> var_values;
  // test get variable, generate checkpoint graph
  ret = session.GetVariables(var_names, var_values);
  EXPECT_EQ(ret, SUCCESS);

  // check result
  CHECK_GRAPH(PreRunBegin) {
    ASSERT_EQ(graph->GetAllNodesSize(), 3);
    auto var_node = graph->FindNode("var1");
    ASSERT_EQ(var_node->GetOutAllNodes().at(0)->GetType(), "Save");
  };
  EXPECT_EQ(var_values.size(), 1);
  EXPECT_TRUE(called_count == 0); // checkpoint graph gen by GetVariables should ignore checkpoint graph check
}

TEST_F(SummaryAndCheckPointTest, test_optimize_summary_graph) {
  // build graph
  Graph graph = BuildVariableGraph();

  // new session & add graph
  map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = AscendString("1"); // train flag
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);

  // register Summary call back func
  static uint32_t called_count = 0;
  ge::session::pCallBackFunc callbackFuncSummary = [](uint32_t graph_id, const std::map<AscendString, ge::Tensor> &params_list) {
    called_count++;
    return SUCCESS;
  };
  ret = session.RegisterCallBackFunc("Summary", callbackFuncSummary);
  EXPECT_EQ(ret, SUCCESS);

  // build input tensor
  std::vector<Tensor> inputs;
  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<uint8_t> data(224 * 224 * sizeof(float));
  Tensor tensor(tensor_desc, data);
  inputs.emplace_back(tensor);

  std::vector<Tensor> outputs;
  // build_graph through session
  DUMP_GRAPH_WHEN("PreRunAfterHandleSummaryOp");
  ret = session.RunGraph(graph_id, inputs, outputs);
  EXPECT_EQ(ret, SUCCESS);
  // check result
  CHECK_GRAPH(PreRunAfterHandleSummaryOp) {
    auto summary = graph->FindNode("summary");
    ASSERT_EQ(summary, nullptr); // summary has been deleted
    auto netoutput = graph->FindNode("output");
    ASSERT_NE(netoutput, nullptr);
    ASSERT_EQ(netoutput->GetInDataNodes().size(), 4); // add relu3 as input of netoutput
    ASSERT_EQ(netoutput->GetInDataNodes().at(3)->GetName(), "relu3");
  };
  EXPECT_EQ(called_count, 1);
}

TEST_F(SummaryAndCheckPointTest, test_optimize_summary_graph_gesession) {
  // build graph
  Graph graph = BuildVariableGraph();

  // new session & add graph
  map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = AscendString("1"); // train flag
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  GeSession session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);

  // register Summary call back func
  static uint32_t called_count = 0;
  RunCallback callbackFuncSummary = [](uint32_t graph_id, const std::map<AscendString, gert::Tensor> &params_list) {
    called_count++;
    return SUCCESS;
  };
  ret = session.RegisterCallBackFunc("Summary", callbackFuncSummary);
  EXPECT_EQ(ret, SUCCESS);

  // build input tensor
  std::vector<Tensor> inputs;
  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<uint8_t> data(224 * 224 * sizeof(float));
  Tensor tensor(tensor_desc, data);
  inputs.emplace_back(tensor);

  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::Tensors2GertTensors(inputs, gert_inputs);
  std::vector<gert::Tensor> outputs;
  // build_graph through session
  DUMP_GRAPH_WHEN("PreRunAfterHandleSummaryOp");
  ret = session.RunGraph(graph_id, gert_inputs, outputs);
  EXPECT_EQ(ret, SUCCESS);
  // check result
  CHECK_GRAPH(PreRunAfterHandleSummaryOp) {
    auto summary = graph->FindNode("summary");
    ASSERT_EQ(summary, nullptr); // summary has been deleted
    auto netoutput = graph->FindNode("output");
    ASSERT_NE(netoutput, nullptr);
    ASSERT_EQ(netoutput->GetInDataNodes().size(), 4); // add relu3 as input of netoutput
    ASSERT_EQ(netoutput->GetInDataNodes().at(3)->GetName(), "relu3");
  };
  EXPECT_EQ(called_count, 1);
}
