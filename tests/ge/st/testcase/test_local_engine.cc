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
#include "host_cpu_engine/host_cpu_engine.h"
#include "engines/local_engine/common/constant/constant.h"
#include "ge/ge_api.h"
#include "framework/common/types.h"

#include "framework/ge_running_env/src/env/ge_default_running_env.h"
#include "framework/ge_running_env/include/ge_running_env/fake_op.h"
#include "framework/ge_running_env/include/ge_running_env/ge_running_env_faker.h"

#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/debug/ge_log.h"
#include "ge_context.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "transformation_ops.h"
#include "register/node_converter_registry.h"
#include "graph_builder/bg_infer_shape.h"
#include "ge_local_context.h"

using namespace gert;
namespace ge {
namespace {
LowerResult LoweringAddStub(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto out_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  return {HyperStatus::Success(), {}, out_shapes, {lower_input.input_addrs[0]}};
}
REGISTER_NODE_CONVERTER(ADD, LoweringAddStub);


ge::graphStatus FakeInferDataTypeForBitCast(gert::InferDataTypeContext *context) {
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto target_dtype = attrs->GetAttrPointer<ge::DataType>(0);
  GE_ASSERT_NOTNULL(target_dtype);
  return context->SetOutputDataType(0, *target_dtype);
}
ge::graphStatus FakeInferShapeForBitcast(gert::InferShapeContext *context) {
  const auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(x_shape);
  GE_ASSERT_NOTNULL(y_shape);
  GE_ASSERT_NOTNULL(attrs);

  const auto target_dtype = attrs->GetAttrPointer<ge::DataType>(0);
  GE_ASSERT_NOTNULL(target_dtype);
  const auto x_desc = context->GetInputDesc(0);
  GE_ASSERT_NOTNULL(x_desc);
  const auto x_dtype = x_desc->GetDataType();

  auto x_dtype_size = ge::GetSizeByDataType(x_dtype);
  auto target_dtype_size = ge::GetSizeByDataType(*target_dtype);
  *y_shape = *x_shape;
  if (x_dtype_size == target_dtype_size) {
    return ge::GRAPH_SUCCESS;
  }
  bool is_use_bit = (x_dtype_size > kDataTypeSizeBitOffset) || (target_dtype_size > kDataTypeSizeBitOffset);
  if (is_use_bit) {
    x_dtype_size = x_dtype_size > kDataTypeSizeBitOffset ? (x_dtype_size - kDataTypeSizeBitOffset)
                                                         : x_dtype_size * kBitNumOfOneByte;
    target_dtype_size = target_dtype_size > kDataTypeSizeBitOffset ? (target_dtype_size - kDataTypeSizeBitOffset)
                                                                   : target_dtype_size * kBitNumOfOneByte;
  }
  if (x_dtype_size > target_dtype_size) {
    GE_ASSERT_TRUE(x_dtype_size % target_dtype_size ==0);
    y_shape->AppendDim(x_dtype_size / target_dtype_size);
    return ge::GRAPH_SUCCESS;
  }
  GE_ASSERT_TRUE(target_dtype_size % x_dtype_size == 0);
  GE_ASSERT_TRUE((x_shape->GetDim(x_shape->GetDimNum() - 1) == ge::UNKNOWN_DIM) ||
                 (x_shape->GetDim(x_shape->GetDimNum() - 1) == (target_dtype_size / x_dtype_size)));
  y_shape->SetDimNum(y_shape->GetDimNum() - 1);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(BITCAST).InferDataType(FakeInferDataTypeForBitCast).InferShape(FakeInferShapeForBitcast);

/*
 *    data0   data1
 *       \    /
 *       add_0
 *         |
 *      bitcast_0
 *         |
 *      netoutput
 */
Graph GetBitcastGraph(const std::vector<int64_t> &input_shape, const ge::DataType input_dtype,
                      const std::vector<int64_t> &bitcast_out_shape, const ge::DataType bitcast_out_dtype) {
  auto data_0 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, input_dtype, input_shape);
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, input_dtype, input_shape);
  auto add_0 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, input_dtype, input_shape);

  auto bitcast_0 =
      OP_CFG(BITCAST).TensorDesc(FORMAT_NCHW, bitcast_out_dtype, bitcast_out_shape).Attr("type", bitcast_out_dtype);

  DEF_GRAPH(g) {
                 CHAIN(NODE("data_0", data_0)->EDGE(0, 0)->NODE("add", add_0));
                 CHAIN(NODE("data_1", data_1)->EDGE(0, 1)->NODE("add", add_0));
                 CHAIN(NODE("add", add_0)->NODE("bitcast", bitcast_0)->NODE("netoutput", NETOUTPUT));
               };

  auto graph = ToGeGraph(g);
  return graph;
}

class RealDivHostCpuOp : public HostCpuOp{
public:
    RealDivHostCpuOp() {};
    virtual graphStatus Compute(Operator &op,
                                const std::map<std::string, const Tensor> &inputs,
                                std::map<std::string, Tensor> &outputs) {
      return GRAPH_SUCCESS;
    }
};
}

class HostCpuEngineTest : public testing::Test {
 protected:
  void SetUp() {
    REGISTER_HOST_CPU_OP_BUILDER(REALDIV, RealDivHostCpuOp);

    ge_env.InstallDefault()
          .Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("aicpu_ascend_kernel"))
          .Install(FakeOp(REALDIV).InfoStoreAndBuilder("aicpu_ascend_kernel"));
  }
  void TearDown() {}
  GeRunningEnvFaker ge_env;
};

TEST_F(HostCpuEngineTest, host_cpu_engine_run) {
  vector<int64_t> perm1{1};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{1}), FORMAT_ND, DT_INT64);
  GeTensorPtr const_tensor1 = 
    std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANT).Weight(const_tensor1);

  vector<int32_t> perm2{1};
  GeTensorDesc tensor_desc2(GeShape(vector<int64_t>{1}), FORMAT_ND, DT_INT32);
  GeTensorPtr const_tensor2 = 
    std::make_shared<GeTensor>(tensor_desc2, reinterpret_cast<uint8_t *>(perm2.data()), sizeof(int32_t)*perm2.size());
  auto const2 = OP_CFG(CONSTANT).Weight(const_tensor2);

  DEF_GRAPH(g1) {
    CHAIN(NODE("const1", const1)->EDGE(0, 0)->NODE("realdiv1", REALDIV));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("realdiv1", REALDIV));
    CHAIN(NODE("data", DATA)->NODE("realdiv2", REALDIV));
    CHAIN(NODE("realdiv1", REALDIV)->NODE("realdiv2", REALDIV));
    CHAIN(NODE("realdiv2", REALDIV)->NODE("netoutput", NETOUTPUT));
  };

  map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  session.AddGraph(1, ToGeGraph(g1), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

class GeLocalEngineTest : public testing::Test {
 protected:
  void SetUp() {
    global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});

    ge::GeRunningEnvFaker ge_env;
    ge_env.InstallDefault();

    mmSetEnv(kEnvValue, "", 1);
  }

  void TearDown() {
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
  }
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
};


TEST_F(GeLocalEngineTest, StaticGraph_BitcastInt32ToInt4_ExpandDim) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl(BITCAST);
  op_impl_func->infer_shape = FakeInferShapeForBitcast;
  op_impl_func->infer_datatype = FakeInferDataTypeForBitCast;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  ge::OpsKernelBuilderRegistry::GetInstance().Unregister(ge::ge_local::kGeLocalOpKernelLibName);
  REGISTER_OPS_KERNEL_BUILDER(ge::ge_local::kGeLocalOpKernelLibName, ge_local::GeLocalOpsKernelBuilder);

  const std::vector<int64_t> input_shape = {1, 5, 4, 4};
  const ge::DataType input_dtype = DT_INT32;
  const std::vector<int64_t> bitcast_out_shape = {1, 5, 4, 4, 8};
  const ge::DataType bitcast_out_dtype = DT_INT4;
  Graph graph = GetBitcastGraph(input_shape, input_dtype, bitcast_out_shape, bitcast_out_dtype);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  const map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(graph_id, inputs); // we only care about compile stage
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(gert::SummaryChecker(compute_graph)
                .StrictDirectNodeTypes({{"Data", 2},
                                        {"Add", 1},
                                        {"Bitcast", 1},
                                        {"NetOutput", 1}}),
            "success");
  auto bitcast_node = compute_graph->FindFirstNodeMatchType(BITCAST);
  ASSERT_NE(bitcast_node, nullptr);
  auto bitcast_opdesc = bitcast_node->GetOpDesc();
  ASSERT_NE(bitcast_opdesc, nullptr);
  EXPECT_EQ(bitcast_opdesc->MutableOutputDesc(0)->GetDataType(), bitcast_out_dtype);
  EXPECT_EQ(bitcast_opdesc->MutableOutputDesc(0)->GetShape().GetDims(), bitcast_out_shape);
  bool reuse_input_flag = false;
  TensorUtils::GetReuseInput(bitcast_opdesc->GetOutputDesc(0), reuse_input_flag);
  EXPECT_EQ(reuse_input_flag, true);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(GeLocalEngineTest, GenerateTask) {
  auto p = std::make_shared<ge_local::GeLocalOpsKernelBuilder>();
  RunContext runContext;
  std::vector<domi::TaskDef> tasks;
  OpDescPtr test_opdesc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr test_graph;
  NodePtr test_node = std::make_shared<Node>(test_opdesc, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node, runContext, tasks), FAILED);
  OpDescPtr test_opdesc1 = std::make_shared<OpDesc>("test", "StackPop");
  NodePtr test_node1 = std::make_shared<Node>(test_opdesc1, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node1, runContext, tasks), SUCCESS);

  GeTensorDesc te_desc(GeShape({-1, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  test_opdesc->AddInputDesc(te_desc);
  EXPECT_EQ(p->GenerateTask(*test_node, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc2 = std::make_shared<OpDesc>("test", "NoOp");
  NodePtr test_node2 = std::make_shared<Node>(test_opdesc2, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node2, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc3 = std::make_shared<OpDesc>("test", "Reshape");
  NodePtr test_node3 = std::make_shared<Node>(test_opdesc3, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node3, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc4 = std::make_shared<OpDesc>("test", "ExpandDims");
  NodePtr test_node4 = std::make_shared<Node>(test_opdesc4, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node4, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc5 = std::make_shared<OpDesc>("test", "ReFormat");
  NodePtr test_node5 = std::make_shared<Node>(test_opdesc5, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node5, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc6 = std::make_shared<OpDesc>("test", "Squeeze");
  NodePtr test_node6 = std::make_shared<Node>(test_opdesc6, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node6, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc7 = std::make_shared<OpDesc>("test", "Unsqueeze");
  NodePtr test_node7 = std::make_shared<Node>(test_opdesc7, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node7, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc8 = std::make_shared<OpDesc>("test", "SqueezeV2");
  NodePtr test_node8 = std::make_shared<Node>(test_opdesc8, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node8, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc9 = std::make_shared<OpDesc>("test", "UnsqueezeV2");
  NodePtr test_node9 = std::make_shared<Node>(test_opdesc9, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node9, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc10 = std::make_shared<OpDesc>("test", "SqueezeV3");
  NodePtr test_node10 = std::make_shared<Node>(test_opdesc10, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node10, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc11 = std::make_shared<OpDesc>("test", "UnsqueezeV3");
  NodePtr test_node11 = std::make_shared<Node>(test_opdesc11, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node11, runContext, tasks), SUCCESS);

  OpDescPtr test_opdesc12 = std::make_shared<OpDesc>("test", "FlattenV2");
  NodePtr test_node12 = std::make_shared<Node>(test_opdesc12, test_graph);
  EXPECT_EQ(p->GenerateTask(*test_node12, runContext, tasks), SUCCESS);
}

TEST_F(GeLocalEngineTest, DynamicGraph_BitcastInt4ToInt32_SqueezeDim) {
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto op_impl_func = space_registry->CreateOrGetOpImpl(BITCAST);
  op_impl_func->infer_shape = FakeInferShapeForBitcast;
  op_impl_func->infer_datatype = FakeInferDataTypeForBitCast;
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);

  ge::OpsKernelBuilderRegistry::GetInstance().Unregister("DNN_VM_GE_LOCAL_OP_STORE");
  REGISTER_OPS_KERNEL_BUILDER("DNN_VM_GE_LOCAL_OP_STORE", ge_local::GeLocalOpsKernelBuilder);

  const std::vector<int64_t> input_shape = {-1, 5, 4, -1};
  const ge::DataType input_dtype = DT_INT4;
  const std::vector<int64_t> bitcast_out_shape = {-1, 5, 4};
  const ge::DataType bitcast_out_dtype = DT_INT32;
  Graph graph = GetBitcastGraph(input_shape, input_dtype, bitcast_out_shape, bitcast_out_dtype);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  const map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  uint32_t graph_id = 1;
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(graph_id, inputs); // we only care about compile stage
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_EQ(gert::SummaryChecker(compute_graph)
                .StrictDirectNodeTypes({{"Data", 2},
                                        {"PartitionedCall", 1},
                                        {"NetOutput", 1}}),
            "success");
  bool is_bitcast_found = false;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == BITCAST) {
      is_bitcast_found = true;
      EXPECT_EQ(node->GetOpDesc()->MutableOutputDesc(0)->GetDataType(), bitcast_out_dtype);
      EXPECT_EQ(node->GetOpDesc()->MutableOutputDesc(0)->GetShape().GetDims(), bitcast_out_shape);
      bool reuse_input_flag = false;
      TensorUtils::GetReuseInput(node->GetOpDesc()->GetOutputDesc(0), reuse_input_flag);
      EXPECT_TRUE(reuse_input_flag == false);
    }
  }
  EXPECT_TRUE(is_bitcast_found == true);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

/*
 *    data1  data2
 *      |      |
 *    mul1   mul2 
 *       \    / 
 *     phony_concat
 *          |
 *         mul3
 *          |
 *      netoutput
 */
ComputeGraphPtr GetSimplePhonyConcatComputeGraph() {
  DEF_GRAPH(test1) {
    auto mul3 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 8, 8});
    auto mul1 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul2 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto phony_concat = OP_CFG("PhonyConcat").Attr("concat_dim", std::vector<int64_t>{3}).Attr("N", std::vector<int64_t>{2});

    CHAIN(NODE("data1", "Data")->NODE("mul1", mul1)->EDGE(0, 0)->NODE("phony_concat", phony_concat)->
          NODE("mul3", mul3)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->NODE("mul2", mul2)->EDGE(0, 1)->NODE("phony_concat", phony_concat));
  };

  auto graph = ToGeGraph(test1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");

  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));

  auto pc = compute_graph->FindNode("phony_concat");

  pc->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  pc->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  pc->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 8, 8}));
  return compute_graph;
}

TEST_F(GeLocalEngineTest, PhonyConcatGraphBuild) {
  DUMP_GRAPH_WHEN("AfterAssignResource");

  map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  ge::OpsKernelBuilderRegistry::GetInstance().Unregister("DNN_VM_GE_LOCAL_OP_STORE");
  REGISTER_OPS_KERNEL_BUILDER("DNN_VM_GE_LOCAL_OP_STORE", ge_local::GeLocalOpsKernelBuilder);

  auto compute_graph = GetSimplePhonyConcatComputeGraph();

  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(compute_graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterAssignResource) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phony_concat"], nullptr);
    bool bool_attr = false;
    int64_t int_attr = 10;
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_NOTASK, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetInt(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, int_attr), true);
    EXPECT_EQ(int_attr, 0);

    ASSERT_NE(name_to_node["mul1"], nullptr);
    vector<int64_t> list_int_attr;
    EXPECT_EQ(ge::AttrUtils::GetListInt(name_to_node["mul1"]->GetOpDesc(), "_output_offset_list_for_continuous", list_int_attr), true);
    EXPECT_EQ(list_int_attr[0], 0);

    ASSERT_NE(name_to_node["mul2"], nullptr);
    EXPECT_EQ(ge::AttrUtils::GetListInt(name_to_node["mul2"]->GetOpDesc(), "_output_offset_list_for_continuous", list_int_attr), true);
    EXPECT_EQ(list_int_attr[0], 64);
  };
}

TEST_F(GeLocalEngineTest, PhonyConcatGraphBuildNegetiveDim) {
  DUMP_GRAPH_WHEN("AfterAssignResource");

  map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  ge::OpsKernelBuilderRegistry::GetInstance().Unregister("DNN_VM_GE_LOCAL_OP_STORE");
  REGISTER_OPS_KERNEL_BUILDER("DNN_VM_GE_LOCAL_OP_STORE", ge_local::GeLocalOpsKernelBuilder);

  auto compute_graph = GetSimplePhonyConcatComputeGraph();

  auto node_phony_concat = compute_graph->FindNode("phony_concat");
  EXPECT_EQ(ge::AttrUtils::SetListInt(node_phony_concat->GetOpDesc(), "concat_dim", {-2}), true);

  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(compute_graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterAssignResource) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phony_concat"], nullptr);
    bool bool_attr = false;
    int64_t int_attr = 10;
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_NOTASK, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetInt(name_to_node["phony_concat"]->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, int_attr), true);
    EXPECT_EQ(int_attr, 0);

    ASSERT_NE(name_to_node["mul1"], nullptr);
    vector<int64_t> list_int_attr;
    EXPECT_EQ(ge::AttrUtils::GetListInt(name_to_node["mul1"]->GetOpDesc(), "_output_offset_list_for_continuous", list_int_attr), true);
    EXPECT_EQ(list_int_attr[0], 0);

    ASSERT_NE(name_to_node["mul2"], nullptr);
    EXPECT_EQ(ge::AttrUtils::GetListInt(name_to_node["mul2"]->GetOpDesc(), "_output_offset_list_for_continuous", list_int_attr), true);
    EXPECT_EQ(list_int_attr[0], 64);
  };
}

/*
 *       data1
 *         |
 *        mul3
 *         |
 *     phony_split
 *       /    \
 *    mul1   mul2 
 *       \    /
 *      netoutput
 */
ComputeGraphPtr GetSimplePhonySplitComputeGraph() {
  DEF_GRAPH(test1) {
    auto mul3 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 8, 8});
    auto mul1 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto mul2 = OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT16, {1, 2, 2, 4, 8});
    auto phony_split = OP_CFG("PhonySplit").Attr("split_dim", std::vector<int64_t>{3}).Attr("num_split", std::vector<int64_t>{2});

    CHAIN(NODE("data1", "Data")->NODE("mul3", mul3)->NODE("phony_split", phony_split)->
          NODE("mul1", mul1)->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("phony_split", phony_split)->NODE("mul2", mul2)->NODE("netoutput", "NetOutput"));
  };

  auto graph = ToGeGraph(test1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto node_mul1 = compute_graph->FindNode("mul1");
  auto node_mul2 = compute_graph->FindNode("mul2");
  auto node_mul3 = compute_graph->FindNode("mul3");
  auto node_phony_split = compute_graph->FindNode("phony_split");
  node_phony_split->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_phony_split->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_phony_split->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_phony_split->GetOpDesc()->MutableOutputDesc(1)->SetFormat(FORMAT_ND);
  node_phony_split->GetOpDesc()->MutableOutputDesc(1)->SetDataType(DT_FLOAT16);
  node_phony_split->GetOpDesc()->MutableOutputDesc(1)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_mul1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  node_mul2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1, 2, 2, 4, 8}));
  node_mul3->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
  node_mul3->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  node_mul3->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1, 2, 2, 8, 8}));

  return compute_graph;
}

TEST_F(GeLocalEngineTest, PhonySplitGraphBuild) {
  DUMP_GRAPH_WHEN("AfterAssignResource");

  map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  ge::OpsKernelBuilderRegistry::GetInstance().Unregister("DNN_VM_GE_LOCAL_OP_STORE");
  REGISTER_OPS_KERNEL_BUILDER("DNN_VM_GE_LOCAL_OP_STORE", ge_local::GeLocalOpsKernelBuilder);

  auto compute_graph = GetSimplePhonySplitComputeGraph();

  Session session(options);
  session.AddGraph(4, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(4, inputs);
  if (ret != SUCCESS) {
    GE_DUMP(compute_graph, "BuildGraphFailed");
  }
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(AfterAssignResource) {
    std::map<std::string, NodePtr> name_to_node;
    for (const auto &node : graph->GetAllNodes()) {
      name_to_node[node->GetName()] = node;
    }
    ASSERT_NE(name_to_node["phony_split"], nullptr);
    bool bool_attr = false;
    int64_t int_attr = 10;
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_split"]->GetOpDesc(), ATTR_NAME_NOTASK, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_split"]->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetBool(name_to_node["phony_split"]->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, bool_attr), true);
    EXPECT_EQ(bool_attr, true);
    EXPECT_EQ(ge::AttrUtils::GetInt(name_to_node["phony_split"]->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, int_attr), true);
    EXPECT_EQ(int_attr, 0);

    ASSERT_NE(name_to_node["mul1"], nullptr);
    vector<int64_t> list_int_attr;
    EXPECT_EQ(ge::AttrUtils::GetListInt(name_to_node["mul1"]->GetOpDesc(), "_input_offset_list_for_continuous", list_int_attr), true);
    EXPECT_EQ(list_int_attr[0], 0);

    ASSERT_NE(name_to_node["mul2"], nullptr);
    EXPECT_EQ(ge::AttrUtils::GetListInt(name_to_node["mul2"]->GetOpDesc(), "_input_offset_list_for_continuous", list_int_attr), true);
    EXPECT_EQ(list_int_attr[0], 64);
  };
}

}  // namespace ge
