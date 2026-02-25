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
#include "ge_graph_dsl/graph_dsl.h"
#include "es_ge_test_ops.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/common/types.h"
#include "jit_execution/exe_points/execution_order.h"
#include <vector>
using namespace std;
using namespace testing;
using namespace ge;

class ExecutionOrderUT : public testing::Test {
 protected:
  void SetUp() override {
    es_graph_ = std::unique_ptr<es::EsGraphBuilder>(new es::EsGraphBuilder("Hi Lowering graph"));
  }
  void TearDown() override {}
  std::unique_ptr<es::EsGraphBuilder> es_graph_;
};

/**
 *      data
 *        |
 *       relu
 *        |
 *       relu1
 *        |
 *    netoutput
 *    该图没有切分机会，进入EO以后，不发生切分，first ep就是last ep。
 */
TEST_F(ExecutionOrderUT, no_slice_tests) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data", DATA);
    data.SetShape({-1,-1,-1,-1});
    auto relu = es::Relu(data);
    auto relu1 = es::Relu(relu);
    es::EsGraphBuilder::SetOutput(relu1, 0);
  }();
  auto graph = es_graph_->BuildAndReset();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());

  compute_graph->SetOutputSize(2); // make a wrong output size

  UserGraph user_graph {0, compute_graph};
  ExecutionOrder eo(user_graph);

  std::vector<GeTensor> input_tensors(1);
  GeTensorDesc td;
  td.SetShape((GeShape({-1, -1, -1, -1})));
  td.SetOriginShape((GeShape({-1, -1, -1, -1})));
  td.SetFormat(FORMAT_ND);
  td.SetOriginFormat(FORMAT_ND);
  td.SetPlacement(Placement::kPlacementDevice);
  td.SetDataType(DT_FLOAT16);

  input_tensors[0] = GeTensor(td);
  ExecutionPoint *first_ep = nullptr;
  EXPECT_EQ(eo.FirstPoint(input_tensors, first_ep), SUCCESS);
  EXPECT_NE(first_ep, nullptr);
  EXPECT_TRUE(first_ep->IsLast());
  EXPECT_EQ(first_ep->GetEpOutNum(), 1);

  ExecutionPoint *next_ep = nullptr;
  EXPECT_EQ(eo.NextPoint(*first_ep, input_tensors, next_ep), SUCCESS);
  EXPECT_EQ(next_ep, nullptr);
}

TEST_F(ExecutionOrderUT, test_has_next_ep) {
  // prepare graph
  [this]() {
    auto data = es_graph_->CreateInput(0, "data", DATA);
    data.SetShape({-1,-1,-1,-1});
    auto relu = es::Relu(data);
    auto relu1 = es::Relu(relu);
    es::EsGraphBuilder::SetOutput(relu1, 0);
  }();
  auto graph = es_graph_->BuildAndReset();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph.get());
  compute_graph->SetOutputSize(1);

  // fake eo with 3 ep
  UserGraph user_graph {0, compute_graph};
  ExecutionOrder eo(user_graph);

  auto ep0 = MakeUnique<ExecutionPoint>(0, compute_graph, compute_graph);
  auto ep1 = MakeUnique<ExecutionPoint>(1, compute_graph, compute_graph);
  auto ep2 = MakeUnique<ExecutionPoint>(2, compute_graph, nullptr);

  eo.slice_graphs_.emplace_back(std::move(ep0));
  eo.slice_graphs_.emplace_back(std::move(ep1));
  eo.slice_graphs_.emplace_back(std::move(ep2));

  // fake input tensor
  std::vector<GeTensor> input_tensors(1);
  GeTensorDesc td;
  td.SetShape((GeShape({-1, -1, -1, -1})));
  td.SetOriginShape((GeShape({-1, -1, -1, -1})));
  td.SetFormat(FORMAT_ND);
  td.SetOriginFormat(FORMAT_ND);
  td.SetPlacement(Placement::kPlacementDevice);
  td.SetDataType(DT_FLOAT16);
  input_tensors[0] = GeTensor(td);

  // check get first,next,last
  ExecutionPoint *first_ep = nullptr;
  EXPECT_EQ(eo.FirstPoint(input_tensors, first_ep), SUCCESS);
  EXPECT_NE(first_ep, nullptr);
  EXPECT_FALSE(first_ep->IsLast());

  ExecutionPoint *next_ep = nullptr;
  EXPECT_EQ(eo.NextPoint(*first_ep, input_tensors, next_ep), SUCCESS);
  EXPECT_NE(next_ep, nullptr);
  EXPECT_FALSE(next_ep->IsLast());

  ExecutionPoint *last_ep = nullptr;
  EXPECT_EQ(eo.NextPoint(*next_ep, input_tensors, last_ep), SUCCESS);
  EXPECT_NE(last_ep, nullptr);
  EXPECT_TRUE(last_ep->IsLast());

  ExecutionPoint *try_last_ep = nullptr;
  EXPECT_EQ(eo.NextPoint(*last_ep, input_tensors, try_last_ep), SUCCESS);
  EXPECT_EQ(try_last_ep, nullptr);
}