/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "all_ops.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "flow_graph/data_flow.h"
#include "node_builder.h"

using namespace ge; 
using namespace dflow;

namespace {
constexpr int32_t kFeedTimeout = 3000;
constexpr int32_t kFetchTimeout = 30000;
/**
 * @brief
 * Build a computed graph by Graph API to be used construct GraphPp for dataflow graph
 *
 * @return ge:::Graph
 *
 */
ge::Graph BuildGraph() {
  auto data0 = op::Data("data0").set_attr_index(0);
  auto data1 = op::Data("data1").set_attr_index(1);
  auto add = op::Add("add").set_input_x1(data0).set_input_x2(data1);
  ge::Graph graph("Graph");
  graph.SetInputs({data0, data1}).SetOutputs({add});
  return graph;
}

/**
 * @brief
 * Build a dataflow graph by DataFlow API
 * The dataflow graph contains 3 flow nodes and DAG shows as following:
 * FlowData    FlowData
 *   |    \    /   |
 *   |     \  /    |
 *   |      \/     |
 *   |      /\     |
 * FlowNode0   FlowNode1
 *    \            /
 *     \          /
 *      \        /
 *       \      /
 *        \    /
 *         \  /
 *          \/
 *       FlowNode2
 *           |
 *           |
 *       FlowOutput
 *
 * @return DataFlow graph
 *
 */
dflow::FlowGraph BuildDataFlow() {
  dflow::FlowGraph flow_graph("flow_graph");
  auto data0 = FlowData("Data0", 0);
  auto data1 = FlowData("Data1", 1);
  BuildBasicConfig udf1_build_cfg = {
      .node_name = "node0", .input_num = 2, .output_num = 1, .compile_cfg = "../config/add_func.json"};
  auto node0 = BuildFunctionNodeSimple(udf1_build_cfg).SetInput(0, data0).SetInput(1, data1);

  BuildBasicConfig udf2_build_cfg = {
      .node_name = "node1", .input_num = 2, .output_num = 1, .compile_cfg = "../config/invoke_func.json"};
  auto node1 =
      BuildFunctionNode(udf2_build_cfg,
                        [](FunctionPp pp) {
                          auto invoke_graph_pp0 =
                              GraphPp("invoke_graph_pp0", BuildGraph).SetCompileConfig("../config/add_graph.json");
                          pp.AddInvokedClosure("invoke_graph", invoke_graph_pp0);
                          return pp;
                        })
          .SetInput(0, data0)
          .SetInput(1, data1);

  BuildBasicConfig graph_build_cfg = {
      .node_name = "node2", .input_num = 2, .output_num = 1, .compile_cfg = "../config/add_graph.json"};
  auto node2 = BuildGraphNode(graph_build_cfg, BuildGraph).SetInput(0, node0).SetInput(1, node1);

  std::vector<FlowOperator> inputs_operator{data0, data1};
  std::vector<FlowOperator> outputs_operator{node2};
  flow_graph.SetInputs(inputs_operator).SetOutputs(outputs_operator);
  return flow_graph;
}

bool CheckResult(std::vector<ge::Tensor> &result, const std::vector<int32_t> &expect_out) {
  if (result.size() != 1) {
    std::cout << "ERROR=======Fetch data size is expected containing 1 element=" << std::endl;
    return false;
  }
  if (result[0].GetSize() != expect_out.size() * sizeof(int32_t)) {
    std::cout << "ERROR=======Verify data size failed===========" << std::endl;
    std::cout << "Tensor size:" << result[0].GetSize() << std::endl;
    std::cout << "Expect size:" << expect_out.size() * sizeof(int32_t) << std::endl;
    return false;
  }
  int32_t *output_data = reinterpret_cast<int32_t *>(result[0].GetData());
  if (output_data != nullptr) {
    for (size_t k = 0; k < expect_out.size(); ++k) {
      if (expect_out[k] != output_data[k]) {
        std::cout << "ERROR=======Verify data failed===========" << std::endl;
        std::cout << "ERROR======expect:" << expect_out[k] << "  real:" << output_data[k] << std::endl;
        return false;
      }
    }
  }
  return true;
}
}  // namespace

int32_t main() {
  // Build dataflow graph
  auto flow_graph = BuildDataFlow();

  // Initialize
  std::map<ge::AscendString, AscendString> config = {
      {"ge.exec.deviceId", "0"},
      {"ge.experiment.data_flow_deploy_info_path", "../config/data_flow_deploy_info.json"},
      {"ge.graphRunMode", "0"}};
  auto ge_ret = ge::GEInitialize(config);
  if (ge_ret != ge::SUCCESS) {
    std::cout << "ERROR=====GeInitialize failed.=======" << std::endl;
    return ge_ret;
  }

  // Create Session
  std::map<ge::AscendString, ge::AscendString> options;
  std::shared_ptr<ge::Session> session = std::make_shared<ge::Session>(options);
  if (session == nullptr) {
    std::cout << "ERROR=======Create session failed===========" << std::endl;
    ge::GEFinalize();
    return ge_ret;
  }

  // Add graph
  ge_ret = session->AddGraph(0, flow_graph.ToGeGraph());
  if (ge_ret != ge::SUCCESS) {
    std::cout << "ERROR=======Add graph failed===========" << std::endl;
    ge::GEFinalize();
    return ge_ret;
  }

  // Prepare Inputs
  const int64_t element_num = 3;
  std::vector<int64_t> shape = {element_num};
  int32_t input_data[element_num] = {4, 7, 5};
  ge::Tensor input_tensor;
  ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT32);
  input_tensor.SetTensorDesc(desc);
  input_tensor.SetData((uint8_t *)input_data, sizeof(int32_t) * element_num);

  ge::DataFlowInfo data_flow_info;
  std::vector<ge::Tensor> inputs_data = {input_tensor, input_tensor};

  // FeedInput
  const size_t loop_num = 4;
  for (size_t i = 0; i < loop_num; ++i) {
    ge_ret = session->FeedDataFlowGraph(0, inputs_data, data_flow_info, kFeedTimeout);
    if (ge_ret != ge::SUCCESS) {
      std::cout << "ERROR=======Feed data failed===========" << std::endl;
      ge::GEFinalize();
      return ge_ret;
    }
  }

  // Verify outputs
  std::vector<int32_t> expect_out = {16, 28, 20};
  for (size_t i = 0; i < loop_num; ++i) {
    std::vector<ge::Tensor> outputs_data;
    ge_ret = session->FetchDataFlowGraph(0, outputs_data, data_flow_info, kFetchTimeout);
    if (ge_ret != ge::SUCCESS) {
      std::cout << "ERROR=======Fetch data failed===========" << std::endl;
      ge::GEFinalize();
      return ge_ret;
    }
    if (!CheckResult(outputs_data, expect_out)) {
      std::cout << "ERROR=======Check result data failed===========" << std::endl;
      ge::GEFinalize();
      return -1;
    }
  }
  std::cout << "TEST=======run case success===========" << std::endl;
  ge::GEFinalize();
  return 0;
}