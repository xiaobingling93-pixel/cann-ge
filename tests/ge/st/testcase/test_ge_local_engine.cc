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
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/graph.h"
#include "graph/tuning_utils.h"
#include "api/gelib/gelib.h"

#include "common/summary_checker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"

#include "host_kernels/kernel_utils.h"

using namespace std;
namespace ge {
namespace {
const auto StubInferShapeForFlattenV2 = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->GetInputDescPtr(0);
  auto y_desc = op_desc->MutableOutputDesc(0);
  auto x_shape_dim = vector<int64_t>(x_desc->GetShape().GetDims());

  if (KernelUtils::IsUnknownShape(x_desc->GetShape())) {
    y_desc->SetShape(x_desc->GetShape());
    y_desc->SetOriginShape(x_desc->GetOriginShape());
    y_desc->SetDataType(x_desc->GetDataType());
    y_desc->SetOriginDataType(x_desc->GetOriginDataType());
    return GRAPH_SUCCESS;
  }

  int64_t axis = 0;
  int64_t end_axis = 0;

  if (!AttrUtils::GetInt(op_desc, "axis", axis)) {
    axis = 1;
  }
  if (!AttrUtils::GetInt(op_desc, "end_axis", end_axis)) {
    end_axis = -1;
  }
  GeTensorDesc x_desc1 = op_desc->GetInputDesc("x");
  auto dim_count = static_cast<int64_t>(x_desc1.GetShape().GetDimNum());
  if (axis < 0) {
    axis += dim_count;
  }
  if (end_axis < 0) {
    end_axis += dim_count;
  }

  std::vector<int64_t> y_shape_dim;
  for (int64_t i = 0; i < axis; i++) {
    y_shape_dim.emplace_back(x_shape_dim[i]);
  }
  int64_t dim_val = 1;
  for (int64_t i = axis; i < (end_axis + 1); i++) {
    dim_val = dim_val * x_shape_dim[i];
  }
  y_shape_dim.emplace_back(dim_val);

  for (int64_t i = (end_axis + 1); i <static_cast<int64_t>(x_shape_dim.size()); i++) {
    y_shape_dim.emplace_back(x_shape_dim[i]);
  }

  y_desc->SetShape(GeShape(y_shape_dim));
  y_desc->SetOriginShape(GeShape(y_shape_dim));
  return GRAPH_SUCCESS;
};

/**
┌────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌───────────┐
│ data_0 │ ───────> │ flattenv2_0 │ ───────> │ netoutput │
└────────┘          └─────────────┘          └───────────┘
 */
Graph BuildFalttenV2Graph(const std::vector<int64_t> &input_shape) {
  int64_t axis = 1;
  int64_t end_axis = -1;
  auto data_0 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, input_shape);
  auto flattenv2_0 = OP_CFG(FLATTENV2).TensorDesc(FORMAT_NCHW, DT_FLOAT, input_shape).Attr("axis", axis).Attr("end_axis", end_axis);
  DEF_GRAPH(g) {
                 CHAIN(NODE("data_0", data_0)->EDGE(0, 0)->NODE("flattenv2_0", flattenv2_0));
                 CHAIN(NODE("flattenv2_0", flattenv2_0)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
               };

  auto graph = ToGeGraph(g);
  const auto &compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == FLATTENV2) {
      node->GetOpDesc()->AddInferFunc(StubInferShapeForFlattenV2);
    }
  }
  return graph;
}
}  // namespace

class GeLocalEngineSystemTest : public testing::Test {
 protected:
  void SetUp() {
    ge_env.InstallDefault();
  }
  void TearDown() {
    ge_env.Reset();
    ge_env.InstallDefault();
  }
  GeRunningEnvFaker ge_env;
};

/**
 * 预置条件: DUMP_GRAPH_WHEN("PreRunAfterBuild");
 * 输入: 静态shape，默认空option
 * 执行步骤：构图
 *          AddGraph
 *          BuildGraph
 * 预期结果：编译成功
 *          FlattenV2节点删除（命中DimensionAdjustPass）
 *
 *       constant            constant
 *        |                    |
 *     flattenv2     -->     netoutput
 *         |
 *       netoutput
 */
TEST_F(GeLocalEngineSystemTest, TestFlattenV2_CheckStaticGraph_FlattenV2RemoveSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  // build graph
  const std::vector<int64_t> input_shape = {1, 2, 3};
  Graph graph = BuildFalttenV2Graph(input_shape);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(gert::SummaryChecker(compute_graph).StrictDirectNodeTypes(
            {{"Data", 1},
            {"FlattenV2", 1},
            {"NetOutput", 1}}),
            "success");
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(graph_id, inputs); // we only care about compile stage
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto flattenv2_node = graph->FindNode("flattenv2_0");
    EXPECT_EQ(flattenv2_node, nullptr);
    EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(
              {{"Data", 1},
              {"NetOutput", 1}}),
              "success");
  };
}

/**
 * 预置条件: DUMP_GRAPH_WHEN("PreRunAfterBuild");
 * 输入: 静态shape，session级别option: ge.oo.constantFolding="false";
 * 执行步骤：构图
 *          AddGraph
 *          BuildGraph
 * 预期结果：编译成功
 *          FlattenV2节点不会删除
 *          输出shape、offset符合预期
 */
TEST_F(GeLocalEngineSystemTest, TestFlattenV2_CheckStaticGraph_FlattenV2ReserveSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  ge_env.InstallDefault()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(FLATTENV2).Inputs({"x"}).Outputs({"y"}).AttrsDef("axis", 1).
          AttrsDef("end_axis", -1).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferShapeForFlattenV2))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  // build graph
  const std::vector<int64_t> input_shape = {1, 2, 3};
  Graph graph = BuildFalttenV2Graph(input_shape);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(gert::SummaryChecker(compute_graph).StrictDirectNodeTypes(
            {{"Data", 1},
            {"FlattenV2", 1},
            {"NetOutput", 1}}),
            "success");
  // new session & add graph
  map<AscendString, AscendString> options;
  options[OO_CONSTANT_FOLDING] = "false";
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(graph_id, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(ret, SUCCESS);
    // check output shape and offset
    std::vector<int64_t> output_dims;
    std::vector<int64_t> flattenv2_0_input_offsets;
    std::vector<int64_t> flattenv2_0_output_offsets;
    NodePtr flattenv2_0 = nullptr;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetName() == "flattenv2_0") {
        flattenv2_0 = node;
        output_dims = node->GetOpDesc()->GetOutputDesc(0U).GetShape().GetDims();
        flattenv2_0_input_offsets = node->GetOpDesc()->GetInputOffset();
        flattenv2_0_output_offsets = node->GetOpDesc()->GetOutputOffset();
      }
    }
    EXPECT_NE(flattenv2_0, nullptr);
    std::vector<int64_t> expect_shape = {1,6};
    EXPECT_EQ(output_dims, expect_shape);
    EXPECT_EQ(flattenv2_0_input_offsets.size(), 1U);
    EXPECT_EQ(flattenv2_0_output_offsets.size(), 1U);
    EXPECT_EQ(flattenv2_0_input_offsets[0], flattenv2_0_output_offsets[0]);
  };
}

/**
 * 输入: 动态shape，默认空option
 * 执行步骤：构图
 *          AddGraph
 *          BuildGraph
 *          RunGraph
 * 预期结果：编译成功
 *         执行成功
 *         输出shape符合预期
 */
TEST_F(GeLocalEngineSystemTest, TestFlattenV2_CheckDynamicGraph_FlattenV2ReserveSuccess) {
  ge_env.InstallDefault()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(FLATTENV2).Inputs({"x"}).Outputs({"y"}).AttrsDef("axis", 1).
        AttrsDef("end_axis", -1).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferShapeForFlattenV2))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
  // build graph
  const std::vector<int64_t> input_shape = {-1, 2, 3};
  Graph graph = BuildFalttenV2Graph(input_shape);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(gert::SummaryChecker(compute_graph).StrictDirectNodeTypes(
            {{"Data", 1},
            {"FlattenV2", 1},
            {"NetOutput", 1}}),
            "success");
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  ASSERT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(graph_id, inputs);
  ASSERT_EQ(ret, SUCCESS);

  // runGraph and check output shape
  std::vector<Tensor> input_tensors, output_tensors;
  TensorDesc tensor_desc(Shape({1, 2, 3}), FORMAT_NCHW, DT_FLOAT);
  std::vector<uint8_t> data = {2,2,2,2,2,2};
  Tensor tensor(tensor_desc, data);
  input_tensors.emplace_back(tensor);
  ret = session.RunGraph(graph_id, input_tensors, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);
  std::vector<int64_t> expect_shape = {1,6};
  ASSERT_EQ(output_tensors.at(0).GetTensorDesc().GetShape().GetDims(), expect_shape);
}
}  // namespace ge
