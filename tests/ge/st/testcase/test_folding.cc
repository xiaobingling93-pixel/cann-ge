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
#include <map>
#include <vector>
#include <graph/operator_reg.h>
#include <ge_running_env/fake_op.h>
#include <ge_running_env/ge_running_env_faker.h>

#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.h"
#include "graph/manager/util/graph_optimize_utility.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "base/common/fp16_t.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

using namespace std;
using namespace ge;

const char *ClipByValue = "ClipByValue";
class TestClipByValue : public Kernel {
 public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override {
    auto output1 = std::make_shared<GeTensor>();
    float value = 65504;
    output1->MutableTensorDesc().SetShape(GeShape());
    output1->SetData((uint8_t *const)(&value), sizeof(value));
    output1->MutableTensorDesc().SetDataType(DT_FLOAT16);
    v_output.push_back(output1);
    return SUCCESS;
  }
};
REGISTER_COMPUTE_NODE_KERNEL(ClipByValue, TestClipByValue);

const char *kFakeWhere = "FakeWhere";
class TestWhereKernelAicpu : public Kernel {
public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override {
    auto output = std::make_shared<GeTensor>();
    std::vector<uint8_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int64_t> shape{-1};
    output->MutableTensorDesc().SetShape(GeShape(shape));
    output->SetData(data);
    output->MutableTensorDesc().SetDataType(DT_UINT8);
    v_output.push_back(output);
    return SUCCESS;
  }
};
REGISTER_COMPUTE_NODE_KERNEL(kFakeWhere, TestWhereKernelAicpu);

namespace {
  graphStatus StubInferFunction(Operator &op) { return GRAPH_SUCCESS; }
// Transpose infer
REG_OP(Transpose)
  .INPUT(x, TensorType::BasicType())
  .INPUT(perm, TensorType::IndexNumberType())
  .OUTPUT(y, TensorType::BasicType())
  .OP_END_FACTORY_REG(Transpose)

const auto TransposeInfer = [](Operator &op) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(0);
  auto &input_shape = input_desc->GetShape();
  auto input_dtype = input_desc->GetDataType();

  auto output_desc = op_info->MutableOutputDesc(0);
  output_desc->SetShape(input_shape);
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(Transpose, TransposeInfer);

// Cast infer
REG_OP(Cast)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(dst_type, Int)
  .OP_END_FACTORY_REG(Cast)

const auto CastInfer = [](Operator &op) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(0);
  auto &input_shape = input_desc->GetShape();

  auto output_desc = op_info->MutableOutputDesc(0);
  output_desc->SetShape(input_shape);
  
  DataType type;
  op.GetAttr("dst_type", type);
  output_desc->SetDataType(type);
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(Cast, CastInfer);

}  // namespace

class ConstantFoldingTest : public testing::Test {
 protected:
  void SetUp() {
    std::cout << "Enter constant folding st" << std::endl;
    GeRunningEnvFaker ge_env;
    ge_env.InstallDefault().Install(FakeOp(GATHERSHAPES).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction))
    .Install(FakeOp(FLATTENV2).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction))
    .Install(FakeOp(FLATTEN).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction))
    .Install(FakeOp(SQUEEZEV3).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction))
    .Install(FakeOp(ClipByValue).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction))
    .Install(FakeOp(CAST).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction))
    .Install(FakeOp("FakeWhere").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferFunction));
  }
  void TearDown() {
    std::cout << "End constant folding st" << std::endl;
    GeRunningEnvFaker ge_env;
    ge_env.InstallDefault();
  }
};

static void BuildGatherShapesGraph(std::vector<std::vector<int64_t>> &axes, Graph &graph) {
  auto gathershapes = OP_CFG(GATHERSHAPES)
                          .TensorDesc(FORMAT_NCHW, DT_INT32, {-1})
                          .Attr<std::vector<std::vector<int64_t>>>("axes", axes);
  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
  auto constant_1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3});
  auto constant_2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4});

  DEF_GRAPH(g1) {
    CHAIN(NODE("constant_1", constant_1)->EDGE(0, 0)->NODE("gathershapes", gathershapes));
    CHAIN(NODE("constant_2", constant_2)->EDGE(0, 1)->NODE("gathershapes", gathershapes));
    CHAIN(NODE("gathershapes", gathershapes)->EDGE(0, 0)->NODE("netoutput", netouput));
  };

  graph = ToGeGraph(g1);
}

static void BuildGatherShapesGraph1(std::vector<std::vector<int64_t>> &axes, Graph &graph) {
  auto gathershapes = OP_CFG(GATHERSHAPES)
                              .TensorDesc(FORMAT_NCHW, DT_INT32, {-1})
                              .Attr<std::vector<std::vector<int64_t>>>("axes", axes);
  auto netoutput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
  auto constant_1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3});
  auto constant_2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4});

  DEF_GRAPH(g1) {
    CHAIN(NODE("constant_1", constant_1)->EDGE(0, 0)->NODE("gathershapes1", gathershapes));
    CHAIN(NODE("constant_2", constant_2)->EDGE(0, 1)->NODE("gathershapes1", gathershapes));
    CHAIN(NODE("gathershapes1", gathershapes)->EDGE(0, 0)->NODE("netoutput", netoutput));
    CHAIN(NODE("constant_3", constant_1)->EDGE(0, 0)->NODE("gathershapes2", gathershapes));
    CHAIN(NODE("constant_4", constant_2)->EDGE(0, 1)->NODE("gathershapes2", gathershapes));
    CHAIN(NODE("gathershapes2", gathershapes)->EDGE(0, 1)->NODE("netoutput", netoutput));
  };

  graph = ToGeGraph(g1);
}

/**
 *    constant,constant
 *        \     /
 *     gathershapes
 *           |
 *       netoutput
 */
TEST_F(ConstantFoldingTest, test_gathershapes_folding) {
  std::vector<std::vector<int64_t>> axes = {{1, 0}, {0, 1}};
  Graph graph;
  BuildGatherShapesGraph(axes, graph);
  auto all_gnodes = graph.GetDirectNode();
  for (GNode &gnode : all_gnodes) {
    AscendString node_name;
    (void)gnode.GetName(node_name);
    if (node_name == "gathershapes") {
      TensorDesc out_desc;
      out_desc.SetDataType((DataType)DT_INT32);
      gnode.UpdateOutputDesc(0, out_desc);
    }
  }

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto gathershapes_node = graph->FindNode("gathershapes");
    EXPECT_EQ(gathershapes_node, nullptr);
  };
}

/**
 *    constant,constant  constant,constant
 *        \     /             \     /
 *     gathershapes1       gathershapes2
 *                 \       /
 *                 netoutput
 */
TEST_F(ConstantFoldingTest, test_gathershapes_folding_aoe_succ) {
  std::vector<std::vector<int64_t>> axes = {{1, 0}, {0, 1}};
  Graph graph;
  BuildGatherShapesGraph1(axes, graph);
  auto all_gnodes = graph.GetDirectNode();
  for (GNode &gnode : all_gnodes) {
    AscendString node_name;
    (void)gnode.GetName(node_name);
    if (node_name == "gathershapes1" || node_name == "gathershapes2") {
      TensorDesc out_desc;
      out_desc.SetDataType((DataType)DT_INT32);
      gnode.UpdateOutputDesc(0, out_desc);
    }
  }

  map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";
  options["ge.buildMode"] = "tuning";
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
}

/**
 *    constant,constant
 *        \     /
 *     gathershapes
 *           |
 *       netoutput
 */
TEST_F(ConstantFoldingTest, test_gathershapes_folding_fail) {
  std::vector<std::vector<int64_t>> axes = {{1, 0}, {0, 8}};
  Graph graph;
  BuildGatherShapesGraph(axes, graph);
  auto all_gnodes = graph.GetDirectNode();
  for (GNode &gnode : all_gnodes) {
    AscendString node_name;
    (void)gnode.GetName(node_name);
    if (node_name == "gathershapes") {
      TensorDesc out_desc;
      out_desc.SetDataType((DataType)DT_INT32);
      gnode.UpdateOutputDesc(0, out_desc);
    }
  }

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, FAILED);
}

/**
 *       constant            constant
 *        |                    |
 *     flattenv2     -->     netoutput
 *         |
 *       netoutput
 */
TEST_F(ConstantFoldingTest, test_flattenv2_folding) {
  int64_t axis = 1;
  int64_t end_axis = -1;
  GeTensor weight;
  std::vector<uint8_t> data = {1, 2};
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({2, 3, 4}));
  weight_desc.SetOriginShape(GeShape({2, 3, 4}));
  weight.SetTensorDesc(weight_desc);
  auto constant =
      OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto flattenv2 =
      OP_CFG(FLATTENV2).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3}).Attr("axis", axis).Attr("end_axis", end_axis);
  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("constant", constant)->EDGE(0, 0)->NODE("flattenv2", flattenv2));
    CHAIN(NODE("flattenv2", flattenv2)->EDGE(0, 0)->NODE("netoutput", netouput));
  };

  auto graph = ToGeGraph(g1);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_ptr = compute_graph->FindNode("flattenv2");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  op_desc_ptr->SetOpKernelLibName("aicpu_ascend_kernel");
  GE_DUMP(compute_graph, "test_flattenv2");
  std::map<string, uint32_t> name_idx_map = {{"x", 0}};
  op_desc_ptr->UpdateInputName(name_idx_map);
  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(2, graph_2, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto flattenv2_node = graph->FindNode("flattenv2");
    EXPECT_EQ(flattenv2_node, nullptr);
  };
}

/**
 *  data   constant
 *   \         /
 *    \       /
 *     squeezev3
 *       |
 *      netoutput
 */
TEST_F(ConstantFoldingTest, test_squeezev3_folding) {
  GeTensor weight;
  std::vector<uint8_t> data = {1, 2};
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({2, 3, 4}));
  weight_desc.SetOriginShape(GeShape({2, 3, 4}));
  weight.SetTensorDesc(weight_desc);

  auto data_x = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 2, 3, 4})
        .InCnt(1)
        .OutCnt(1);

  auto squeezev3 =
      OP_CFG(SQUEEZEV3).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3});
  std::vector<uint8_t> data_2 = {0};    
  weight.SetData(data_2);
  auto constant_axes =
      OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {1}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);

  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3});
  DEF_GRAPH(graph_squeezev3) {
    CHAIN(NODE("data_x", data_x)->EDGE(0, 0)->NODE("squeezev3", squeezev3));
    CHAIN(NODE("constant_axes",constant_axes)->EDGE(0,1)->NODE("squeezev3", squeezev3));
    CHAIN(NODE("squeezev3", squeezev3)->EDGE(0, 0)->NODE("netoutput", netouput));
  };

  auto graph = ToGeGraph(graph_squeezev3);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_ptr = compute_graph->FindNode("squeezev3");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  std::map<string, uint32_t> name_idx_map = {{"x", 0}};
  op_desc_ptr->UpdateInputName(name_idx_map);
  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph_2, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto squeezev3_node = graph->FindNode("squeezev3");
    EXPECT_EQ(squeezev3_node, nullptr);
  };
}

/**
 *  data   constant
 *   \         /
 *    \       /
 *     unsqueezev3
 *       |
 *     netoutput
 */
TEST_F(ConstantFoldingTest, test_unsqueezev3_folding) {
  GeTensor weight;
  std::vector<uint8_t> data = {1, 2};
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({2, 3, 4}));
  weight_desc.SetOriginShape(GeShape({2, 3, 4}));
  weight.SetTensorDesc(weight_desc);

  auto data_x = OP_CFG(DATA)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 2, 3, 4})
        .InCnt(1)
        .OutCnt(1);

  auto unsqueezev3 =
      OP_CFG(UNSQUEEZEV3).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3});
  std::vector<uint8_t> data_2 = {0};    
  weight.SetData(data_2);
  auto constant_axes =
      OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {1}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);

  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4});
  DEF_GRAPH(graph_unsqueezev3) {
    CHAIN(NODE("data_x", data_x)->EDGE(0, 0)->NODE("unsqueezev3", unsqueezev3));
    CHAIN(NODE("constant_axes",constant_axes)->EDGE(0,1)->NODE("unsqueezev3", unsqueezev3));
    CHAIN(NODE("unsqueezev3", unsqueezev3)->EDGE(0, 0)->NODE("netoutput", netouput));
  };

  auto graph = ToGeGraph(graph_unsqueezev3);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_ptr = compute_graph->FindNode("unsqueezev3");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  std::map<string, uint32_t> name_idx_map = {{"x", 0}};
  op_desc_ptr->UpdateInputName(name_idx_map);
  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph_2, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto unsqueezev3_node = graph->FindNode("unsqueezev3");
    EXPECT_EQ(unsqueezev3_node, nullptr);
  };
}

/**
 *    data      perm               data
 *      \        /                  |
 *       \      /                   |
 *      transpose    ----->    netoutput
 *          |
 *      netoutput
 */
TEST_F(ConstantFoldingTest, test_unchanged_transpose_remove_pass) {
  auto data_x = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 3, 4});
  vector<int64_t> perm{1, 0, 2, 3};
  GeTensorDesc tensor_desc(GeShape(vector<int64_t>{4}), FORMAT_NCHW, DT_INT64);
  GeTensorPtr const_tensor =
      std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(perm.data()) , sizeof(int64_t)*perm.size());
  auto constant = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT64, {4}).Weight(const_tensor);

  auto transpose = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 3, 4});

  DEF_GRAPH(g) {
                  CHAIN(NODE("data_x", data_x)->EDGE(0, 0)->NODE("transpose", transpose));
                  CHAIN(NODE("constant", constant)->EDGE(0, 1)->NODE("transpose", transpose));
                  CHAIN(NODE("transpose", transpose)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToGeGraph(g);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto transpose_node = compute_graph->FindNode("transpose");
  transpose_node->GetOpDesc()->AddInferFunc(TransposeInfer);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  auto node_ptr = compute_graph->FindNode("transpose");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  std::map<string, uint32_t> name_idx_map;
  name_idx_map.emplace("x", 0);
  name_idx_map.emplace("perm", 1);
  op_desc_ptr->UpdateInputName(name_idx_map);

  EXPECT_EQ(graph.GetDirectNode().size(), 4);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph.GetDirectNode().size(), 2);
}

TEST_F(ConstantFoldingTest, test_unchanged_transpose_remove_pass_invalid) {
  auto data_x = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 2, 3, 4});
  vector<int64_t> perm{0, 2, 1, 3};
  GeTensorDesc tensor_desc(GeShape(vector<int64_t>{4}), FORMAT_NCHW, DT_INT64);
  GeTensorPtr const_tensor =
      std::make_shared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(perm.data()) , sizeof(int64_t)*perm.size());
  auto constant = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT64, {4}).Weight(const_tensor);

  auto transpose = OP_CFG(TRANSPOSE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 2, 3, 4});

  DEF_GRAPH(g) {
                 CHAIN(NODE("data_x", data_x)->EDGE(0, 0)->NODE("transpose", transpose));
                 CHAIN(NODE("constant", constant)->EDGE(0, 1)->NODE("transpose", transpose));
                 CHAIN(NODE("transpose", transpose)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
               };

  auto graph = ToGeGraph(g);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto transpose_node = compute_graph->FindNode("transpose");
  transpose_node->GetOpDesc()->AddInferFunc(TransposeInfer);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  auto node_ptr = compute_graph->FindNode("transpose");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  std::map<string, uint32_t> name_idx_map;
  name_idx_map.emplace("x", 0);
  name_idx_map.emplace("perm", 1);
  op_desc_ptr->UpdateInputName(name_idx_map);

  EXPECT_EQ(graph.GetDirectNode().size(), 4);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph.GetDirectNode().size(), 4);
}

TEST_F(ConstantFoldingTest, test_cast_float32toint32) {
  int32_t src = 65535;
  fp16_t val1;
  val1 = src;

  GeTensor weight;
  //std::vector<uint8_t> data = {2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<uint8_t> data = {2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2};
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({2, 3}));
  weight_desc.SetOriginShape(GeShape({2, 3}));
  weight.SetTensorDesc(weight_desc);
  auto constant =
      OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_FLOAT16, {2, 3}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto cast =
      OP_CFG(CAST).TensorDesc(FORMAT_NCHW, DT_FLOAT16, {2, 3}).Attr("dst_type", DT_INT32);
  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_INT32, {-1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("constant", constant)->EDGE(0, 0)->NODE("cast", cast));
    CHAIN(NODE("cast", cast)->EDGE(0, 0)->NODE("netoutput", netouput));
  };

  auto graph = ToGeGraph(g1);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_ptr = compute_graph->FindNode("cast");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  //op_desc_ptr->SetOpKernelLibName("aicpu_ascend_kernel");
  GE_DUMP(compute_graph, "test_cast");
  std::map<string, uint32_t> name_idx_map = {{"x", 0}};
  op_desc_ptr->UpdateInputName(name_idx_map);

  auto input_desc = op_desc_ptr->MutableInputDesc(0);

  auto output_desc = op_desc_ptr->MutableOutputDesc(0);
  output_desc->SetShape(GeShape({2, 3}));
  output_desc->SetDataType(DT_INT32);

  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(2, graph_2, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto cast_node = graph->FindNode("cast");
    EXPECT_EQ(cast_node, nullptr);
  };
}

TEST_F(ConstantFoldingTest, test_cast_float32tofloat16_inf) {
  dlog_setlevel(0, 0, 0);
  float value = 65505; 
  GeTensor weight;
  weight.SetData((uint8_t *const)(&value), sizeof(value));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape());
  weight_desc.SetDataType(DT_FLOAT);
  weight_desc.SetOriginShape(GeShape());
  weight.SetTensorDesc(weight_desc);
  auto constant =
      OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_FLOAT, {}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto cast =
      OP_CFG(CAST).TensorDesc(FORMAT_ND, DT_FLOAT, {}).Attr("dst_type", DT_FLOAT16);
  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_ND, DT_FLOAT16, {-1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("constant", constant)->EDGE(0, 0)->NODE("cast", cast));
    CHAIN(NODE("cast", cast)->EDGE(0, 0)->NODE("netoutput", netouput));
  };

  auto graph = ToGeGraph(g1);

  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_ptr = compute_graph->FindNode("cast");
  auto op_desc_ptr = node_ptr->GetOpDesc();
  //op_desc_ptr->SetOpKernelLibName("aicpu_ascend_kernel");
  std::map<string, uint32_t> name_idx_map = {{"x", 0}};
  op_desc_ptr->UpdateInputName(name_idx_map);

  auto input_desc = op_desc_ptr->MutableInputDesc(0);

  auto output_desc = op_desc_ptr->MutableOutputDesc(0);
  output_desc->SetShape(GeShape());
  output_desc->SetDataType(DT_FLOAT16);

  auto graph_2 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  options["ge.is_weight_clip"] = "1";
  Session session(options);
  session.AddGraph(2, graph_2, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto cast_node = graph->FindNode("cast");
    EXPECT_EQ(cast_node, nullptr);
  };
  dlog_setlevel(3, 3, 0);
}

TEST_F(ConstantFoldingTest, test_ConstantFolding_Notchanged) {
  std::vector<std::vector<int64_t>> axes = {{1, 0}, {0, 1}};
  Graph graph;
  BuildGatherShapesGraph(axes, graph);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 4);
  GraphOptimizeUtility graph_optimize_utility;
  auto gathershapes_node = compute_graph->FindNode("constant_1");
  EXPECT_NE(gathershapes_node, nullptr);
  auto ret = graph_optimize_utility.ConstantFolding(gathershapes_node);
  EXPECT_EQ(ret, NOT_CHANGED);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 4);
}

static void BuildNotFoldGraph(Graph &graph) {
  GeTensor weight;
  std::vector<uint8_t> data(2 * 5);
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({2, 5}));
  weight_desc.SetOriginShape(GeShape({2, 5}));
  weight.SetTensorDesc(weight_desc);
  auto constant_1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_INT32, {2, 5}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto constant_2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_INT32, {2, 5}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
  auto add1 = OP_CFG(ADDN).TensorDesc(FORMAT_ND, DT_INT32, {2, 5});
  auto netouput = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_ND, DT_INT32, {2, 5});
  auto flatten1 = OP_CFG(FLATTEN).TensorDesc(FORMAT_ND, DT_INT32, {10});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("constant_1", constant_1)->EDGE(0, 0)->NODE("add1", add1));
                  CHAIN(NODE("constant_2", constant_2)->EDGE(0, 1)->NODE("add1", add1));
                  CHAIN(NODE("add1", add1)->EDGE(0, 0)->NODE("netoutput", netouput));
                  CHAIN(NODE("constant_1", constant_1)->EDGE(0, 0)->NODE("flatten1", flatten1));
                };
  graph = ToGeGraph(g1);
}

TEST_F(ConstantFoldingTest, test_Not_ConstantFolding) {
  Graph graph;
  BuildNotFoldGraph(graph);
  map<AscendString, AscendString> options;
  options["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    ASSERT_EQ(graph->GetDirectNode().size(), 3);
  };
}

TEST_F(ConstantFoldingTest, test_do_not_constant_folding) {
  std::vector<std::vector<int64_t>> axes = {{1, 0}, {0, 1}};
  Graph graph;
  BuildGatherShapesGraph(axes, graph);
  auto all_gnodes = graph.GetDirectNode();
  for (GNode &gnode : all_gnodes) {
    AscendString node_name;
    (void)gnode.GetName(node_name);
    if (node_name == "gathershapes") {
      TensorDesc out_desc;
      out_desc.SetDataType((DataType)DT_INT32);
      gnode.UpdateOutputDesc(0, out_desc);
      bool value = true;
      EXPECT_EQ(gnode.SetAttr(AscendString(ATTR_NAME_DO_NOT_CONSTANT_FOLDING.c_str()), value), GRAPH_SUCCESS);
    }
  }

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    auto node = graph->FindNode("gathershapes");
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->GetOpDesc()->HasAttr(ATTR_NAME_DO_NOT_CONSTANT_FOLDING), true);
  };
}

TEST_F(ConstantFoldingTest, TestFolding_Ok_IgnoreFoldingWhen) {
  GeTensor weight;
  std::vector<uint8_t> data(9);
  weight.SetData(data);
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({9}));
  weight_desc.SetOriginShape(GeShape({9}));
  weight.SetTensorDesc(weight_desc);

  DEF_GRAPH(test_graph) {
    const auto const0 =
        OP_CFG(CONSTANT).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {9}).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
    auto where = OP_CFG("FakeWhere").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1);
    CHAIN(NODE("const0", const0)->EDGE(0, 0)->NODE("where", where)->NODE("net_output", net_output));
  };
  Graph graph = ToGeGraph(test_graph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  const auto where_node = compute_graph->FindNode("where");
  const auto where_infer_func = [](Operator &op) {
    op.GetOutputDesc(0).SetShape(Shape({-1}));
    op.GetOutputDesc(0).SetOriginShape(Shape({-1}));
    return GRAPH_SUCCESS;
  };
  where_node->GetOpDesc()->AddInferFunc(where_infer_func);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  auto ret = session.CompileGraph(1);
  // where has no lowering func
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    const auto &node = GraphUtils::FindNodeFromAllNodes(const_cast<ComputeGraphPtr &>(graph), "where");
    EXPECT_NE(node, nullptr);
  };
}
