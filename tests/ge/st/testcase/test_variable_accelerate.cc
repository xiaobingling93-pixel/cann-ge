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

#include <mutex>
#include <chrono>

#include "macro_utils/dt_public_scope.h"
#include "ge/ge_api.h"
#include "graph/preprocess/graph_prepare.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "utils/graph_factory.h"
#include "graph/manager/graph_var_manager.h"
#include "ge_running_env/tensor_utils.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils.h"
#include "utils/synchronizer.h"
#include "graph/ge_global_options.h"
#include "graph/passes/variable_optimize/assign_remove_pass.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "graph/manager/mem_manager.h"
#include "common/math/hif8_t.h"

using namespace std;
using namespace ge;
namespace {
/**
 *       Variable(2, 3, 4, 5)                   Relu(1,2,3,4,5)
 *          /            \                             |
 *     TransData      TransData                    TransData
 *         |              |                            |
 *   Relu(1,2,3,4,5)  Relu(1,2,3,4,5)            Variable(2, 3, 4, 5)
 *          \            /                             |
 *             NetOutput -------------------------------
 */
Graph BuildVariableGraph(bool has_copy_from_attr = false) {
  GeTensorDesc tensor_4_desc(ge::GeShape({2,3,4,5}), FORMAT_NCHW, DT_INT32);
  GeTensorDesc tensor_5_desc(ge::GeShape({1,2,3,4,5}), FORMAT_NC1HWC0, DT_INT32);

  auto var1 = std::make_shared<OpDesc>("var1", VARIABLE);
  auto var2 = std::make_shared<OpDesc>("var2", VARIABLE);
  var1->AddInputDesc(tensor_4_desc);
  var1->AddOutputDesc(tensor_4_desc);
  var2->AddInputDesc(tensor_4_desc);
  var2->AddOutputDesc(tensor_4_desc);
  if (has_copy_from_attr) {
    (void)AttrUtils::SetStr(var1, "_copy_from_var_node", "var2");
    (void)AttrUtils::SetBool(var1, "_copy_value", false);
  }

  auto trans1 = std::make_shared<OpDesc>("transdata1", TRANSDATA);
  trans1->AddInputDesc("x", tensor_4_desc);
  trans1->AddOutputDesc("x", tensor_5_desc);

  auto trans2 = std::make_shared<OpDesc>("transdata2", TRANSDATA);
  trans2->AddInputDesc("x", tensor_4_desc);
  trans2->AddOutputDesc("x", tensor_5_desc);

  auto trans3 = std::make_shared<OpDesc>("transdata3", TRANSDATA);
  trans3->AddInputDesc(tensor_5_desc);
  trans3->AddOutputDesc(tensor_4_desc);

  auto data1 = std::make_shared<OpDesc>("data1", DATA);
  data1->AddInputDesc(tensor_5_desc);
  data1->AddOutputDesc(tensor_5_desc);

  auto relu1 = std::make_shared<OpDesc>("relu1", RELU);
  relu1->AddInputDesc(tensor_5_desc);
  relu1->AddOutputDesc(tensor_5_desc);

  auto relu2 = std::make_shared<OpDesc>("relu2", RELU);
  relu2->AddInputDesc(tensor_5_desc);
  relu2->AddOutputDesc(tensor_5_desc);

  auto relu3 = std::make_shared<OpDesc>("relu3", RELU);
  relu3->AddInputDesc(tensor_5_desc);
  relu3->AddOutputDesc(tensor_5_desc);

  auto var_ref = std::make_shared<OpDesc>("var_ref", VARIABLE);
  AttrUtils::SetStr(var_ref, REF_VAR_SRC_VAR_NAME, "var1");
  var_ref->AddInputDesc(tensor_4_desc);
  var_ref->AddOutputDesc(tensor_4_desc);

  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE(trans1)->NODE(relu1)->NODE("output", NETOUTPUT));
    CHAIN(NODE(var1)->EDGE(0, 0)->NODE(trans2)->NODE(relu2)->NODE("output"));
    CHAIN(NODE(data1)->NODE(relu3)->NODE(trans3)->NODE(var_ref)->NODE("output"));
    CHAIN(NODE(var2)->NODE("output"));
  };
  return ToGeGraph(g1);
}

ComputeGraphPtr BuildVariableAcrossPartionedCallGraph() {
  GeTensorDesc tensor_4_desc(ge::GeShape({2,3,4,5}), FORMAT_NCHW, DT_INT32);
  auto var1 = std::make_shared<OpDesc>("var1", VARIABLE);
  var1->AddInputDesc(tensor_4_desc);
  var1->AddOutputDesc(tensor_4_desc);
  auto var_ref = std::make_shared<OpDesc>("var_ref", VARIABLE);
  AttrUtils::SetStr(var_ref, REF_VAR_SRC_VAR_NAME, "var1");
  var_ref->AddInputDesc(tensor_4_desc);
  var_ref->AddOutputDesc(tensor_4_desc);

  DEF_GRAPH(g1) {
    CHAIN(NODE(var1)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE(var_ref)->
          NODE("Output", NETOUTPUT));
    CHAIN(NODE("data", DATA)->NODE("PartitionedCall_0"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  DEF_GRAPH(g2) {
    CHAIN(NODE("sub/variable", VARIABLE)->NODE("sub/assign", ASSIGN)->NODE("sub/Output", NETOUTPUT));
    CHAIN(NODE("sub/data", DATA)->NODE("sub/assign"));
  };
  ComputeGraphPtr subGraph = ToComputeGraph(g2);
  subGraph->SetName("f");

  ge::AttrUtils::SetInt(graph->FindNode("data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(subGraph->FindNode("sub/data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto output_node = subGraph->FindNode("sub/Output");
  ge::AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  AddPartitionedCall(graph, "PartitionedCall_0", subGraph);
  return graph;
}

/**
 * 
 *    Variable       Const
 *   [2, 3, 4, 5]    [1,2,3,4,5]             
 *             \     /                               
 *              Reshape                 
 *                 |                                         
 *         Relu[1,2,3,4,5]         
 *                 |                                     
 *             NetOutput 
 * 
 */
Graph BuildVariableAndReshapeGraph() {
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec = {1, 2, 3, 4, 5};
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *) data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int32_t));

  auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4, 5}).InCnt(0).OutCnt(1);
  auto const1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {5}).InCnt(0).OutCnt(1).Weight(data_tensor);
  auto reshape = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});
  auto relu = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});
  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});

  DEF_GRAPH(g1) {
    CHAIN(NODE("var1", var1)
              ->EDGE(0, 0)
              ->NODE("reshape", reshape)
              ->NODE("relu", relu)
              ->NODE("netoutput", netoutput));
    CHAIN(NODE("const1", const1)->EDGE(0, 1)->NODE("reshape"));
  };
  auto ge_graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(ge_graph);
  compute_graph->FindNode("var1")->GetOpDesc()->MutableOutputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({2, 3, 4, 5})));
  compute_graph->FindNode("reshape")->GetOpDesc()->MutableInputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({2, 3, 4, 5})));
  compute_graph->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({1, 2, 3, 4, 5})));
  compute_graph->FindNode("relu")->GetOpDesc()->MutableInputDesc(0)
      ->SetOriginShape(GeShape(std::vector<int64_t>({1, 2, 3, 4, 5})));
  return ge_graph;
}

/*
 *          netoutput
 *              |
 *         transdata1
 *              |
 *           assign1
 *           /     \
 * const1   |     add1
 *   c|    |      |  \
*     ----var1  data1 const2
 */
Graph BuildSimpleVarAssignGraph() {
  std::vector<int64_t> shape = {2, 2, 3, 2};  // HWCN
  auto data_tensor = GenerateTensor(shape);
  DEF_GRAPH(var_init) {
    auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("var1");

    auto assign1 = OP_CFG(ASSIGN).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(2).Build("assign1");

    auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(2).OutCnt(1).Build("add1");

    auto const1 =
        OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Weight(data_tensor).InCnt(1).OutCnt(1).Build("const1");

    auto const2 =
        OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Weight(data_tensor).InCnt(1).OutCnt(1).Build("const2");

    auto data1 =
        OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("data1");

    auto transdata1 = OP_CFG(TRANSDATA).InCnt(1).OutCnt(1).Build("transdata1");
    auto netoutput = OP_CFG(NETOUTPUT).Build("netoutput");

    assign1->MutableInputDesc(0)->SetRefPortByIndex(std::vector<uint32_t>({0}));

    CHAIN(NODE(var1)->NODE(assign1));
    CHAIN(NODE(var1)->Ctrl()->NODE(const1));
    CHAIN(NODE(data1)->NODE(add1)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(const2)->EDGE(0, 1)->NODE(add1));
    CHAIN(NODE(assign1)->NODE(transdata1));
    CTRL_CHAIN(NODE(transdata1)->NODE(netoutput));
  };

  return ToGeGraph(var_init);
}

Graph BuildGraphForVariablePrepareOpPass() {
  std::vector<int64_t> shape = {1};
  auto data_tensor = GenerateTensor(shape);
  auto variable_switch = OP_CFG(VARIABLE)
    .TensorDesc(FORMAT_ND, DT_BOOL, shape)
    .InCnt(1)
    .OutCnt(1)
    .Build("variable_switch");
  auto const1 = OP_CFG(CONSTANT)
    .Weight(data_tensor)
    .InCnt(1)
    .OutCnt(1)
    .Build("const1");
  DEF_GRAPH(ControlOpGraph) {
    CHAIN(NODE("variable", VARIABLE)->EDGE(0U, 0U)->NODE("refswitch", FRAMEWORKOP));
    CHAIN(NODE(variable_switch)->EDGE(0U, 1U)->NODE("refswitch", FRAMEWORKOP));
    CHAIN(NODE("refswitch", FRAMEWORKOP)->EDGE(0U, 0U)->NODE("assign", ASSIGN));
    CHAIN(NODE(const1)->EDGE(0U, 1U)->NODE("assign", ASSIGN));
    CHAIN(NODE("assign", ASSIGN)->EDGE(0U, 0U)->NODE("refmerge", FRAMEWORKOP));
    CHAIN(NODE("refswitch", FRAMEWORKOP)->EDGE(1U, 1U)->NODE("refmerge", FRAMEWORKOP));
    CHAIN(NODE("refmerge", FRAMEWORKOP)->EDGE(0U, 0U)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("refmerge", FRAMEWORKOP)->EDGE(1U, 1U)->NODE("netoutput", NETOUTPUT));
  };

  const auto graph = ToGeGraph(ControlOpGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  // set the same name for in/output name for assign
  auto assign = compute_graph->FindNode("assign");
  auto assign_op_desc = assign->GetOpDesc();
  const map<string, uint32_t> name_index1 = {{"ref", 0}, {"value", 1}};
  assign_op_desc->UpdateInputName(name_index1);
  assign_op_desc->UpdateOutputName(name_index1);

  auto refswitch = compute_graph->FindNode("refswitch");
  (void)AttrUtils::SetStr(refswitch->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, REFSWITCH);

  auto refmerge = compute_graph->FindNode("refmerge");
  (void)AttrUtils::SetStr(refmerge->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, REFMERGE);

  return graph;
}

}  // namespace
class VariableAccSt : public testing::Test {
 protected:
  void SetUp() {
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  }
  void TearDown() {
    VarManagerPool::Instance().Destory();
    char runtime2_env[MMPA_MAX_PATH] = {'1'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  }
};

TEST_F(VariableAccSt, test_variable_ref) {
  // build graph
  auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(EVALUATE_GRAPH_RESOURCE_MODE);
  if (it != global_options.end()) {
    global_options.erase(it);
  }

  Graph graph = BuildVariableGraph();
  Graph graph2 = BuildVariableGraph(true);

  // new session & add graph
  map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
  auto ret = session.AddGraph(2, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.AddGraph(3, graph2, options);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);

  ret = session.BuildGraph(3, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(VariableAccSt, test_variable_ref_across_subGraph) {
  // build graph
  auto graph = BuildVariableAcrossPartionedCallGraph();

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(0, GraphUtilsEx::CreateGraphFromComputeGraph(graph), options);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(VariableAccSt, test_variable_and_reshape_fusion_check_origin_shape_success) {
  // build graph
  Graph graph = BuildVariableAndReshapeGraph();
  DUMP_GRAPH_WHEN("OptimizeStage1_1");

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(2, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // check result
  CHECK_GRAPH(OptimizeStage1_1) {
    auto var_node = graph->FindNode("var1");
    ASSERT_EQ(var_node->GetOutAllNodes().at(0)->GetName(), "relu");
    EXPECT_EQ(var_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims(),
              std::vector<int64_t>({1, 2, 3, 4, 5}));
  };
}

void Fake5DNodeEngine(GeRunningEnvFaker &ge_env) {
  auto ffo = MakeShared<FakeFormatsOptimizer>();
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_NHWC, 5));
  ffo->OpFormatByType(
      CONV2D, {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,1,16,16,16}))},
              {dst_format, GeShape(std::vector<int64_t>({4,1,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({8,1,16,16,16}))}
          }
      });
  ge_env.InstallDefault();
  ge_env.Install(FakeEngine("TestForVarAcc").GraphOptimizer("FormatOp", ffo));
}

void RunAndCheckInitVarGraph(Session &session, const std::map<AscendString, AscendString> &graph_options) {
  auto var_init_graph = GraphFactory::BuildVarInitGraph1();
  session.AddGraph(1, var_init_graph, graph_options);
  std::vector<ge::Tensor> g1_inputs;
  Synchronizer sync;
  auto ret = session.RunGraphAsync(1, g1_inputs, [&sync](Status run_ret, std::vector<ge::Tensor> &) {
    EXPECT_EQ(run_ret, SUCCESS);
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);

  sync.WaitFor(60);
  ASSERT_TRUE(VarManagerPool::Instance().GetVarManager(session.sessionId_)->IsVarExist("var1"));
  ASSERT_TRUE(VarManagerPool::Instance().GetVarManager(session.sessionId_)->IsVarExist("var2"));
  GeTensorDesc td1;
  EXPECT_EQ(VarManagerPool::Instance().GetVarManager(session.sessionId_)->GetCurVarDesc("var1", td1), SUCCESS);
  // var1由ND格式的const做初始化，因此其格式被推为ND
  EXPECT_EQ(td1.GetFormat(), FORMAT_ND);
  EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  GeTensorDesc td2;
  EXPECT_EQ(VarManagerPool::Instance().GetVarManager(session.sessionId_)->GetCurVarDesc("var2", td2), SUCCESS);
  // var2由HWCN格式的const做初始化，因此其格式被推为HWCN
  EXPECT_EQ(td2.GetFormat(), FORMAT_HWCN);
  EXPECT_EQ(td2.GetShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  EXPECT_EQ(td2.GetOriginShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
}

void RunAndCheckCheckpointGraph(Session &session, bool is_5d, const std::map<AscendString, AscendString> &graph_options) {
  std::vector<ge::Tensor> g1_inputs;
  Synchronizer sync;
  GeTensorDesc td1;
  GeTensorDesc td2;

  auto ckp_graph = GraphFactory::BuildCheckpointGraph1();
  auto ret = session.AddGraph(2, ckp_graph, graph_options);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.RunGraphAsync(2, g1_inputs, [&sync](Status run_ret, std::vector<ge::Tensor> &) {
    EXPECT_EQ(run_ret, SUCCESS);
    sync.OnDone();
  });
  ASSERT_EQ(ret, SUCCESS);

  sync.WaitFor(60);
  EXPECT_EQ(VarManagerPool::Instance().GetVarManager(session.sessionId_)->GetCurVarDesc("var1", td1), SUCCESS);
  const Format expect_format = static_cast<Format>(GetPrimaryFormat(td1.GetFormat()));
  if (is_5d) {
    EXPECT_EQ(expect_format, FORMAT_FRACTAL_Z);
    EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({4,1,16,16}));
  } else {
    EXPECT_EQ(expect_format, FORMAT_ND);
    EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  }

  EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  EXPECT_EQ(VarManagerPool::Instance().GetVarManager(GraphUtilsEx::GetComputeGraph(ckp_graph)->GetSessionID())->GetCurVarDesc("var2", td2), SUCCESS);
  EXPECT_EQ(td2.GetFormat(), FORMAT_ND);  // 这里行为有些奇怪，变量管理器的当前格式由原来的HWCN改成ND了
  EXPECT_EQ(td2.GetShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  EXPECT_EQ(td2.GetOriginShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
}

void RunAndCheckTrainGraph(Session &session, const std::map<AscendString, AscendString> &graph_options) {
  std::vector<ge::Tensor> g1_inputs;
  Synchronizer sync;
  GeTensorDesc td1;
  GeTensorDesc td2;

  auto train_graph = GraphFactory::BuildVarTrainGraph1();
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,16,16})));
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({})));
  auto ret = session.AddGraph(3, train_graph, graph_options);
  EXPECT_EQ(ret, SUCCESS);

  ret = session.RunGraphAsync(3, g1_inputs, [&sync](Status run_ret, std::vector<ge::Tensor> &) {
    EXPECT_EQ(run_ret, SUCCESS);
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);

  sync.WaitFor(60);
  EXPECT_EQ(VarManagerPool::Instance().GetVarManager(session.sessionId_)->GetCurVarDesc("var1", td1), SUCCESS);
  const Format expect_format = static_cast<Format>(GetPrimaryFormat(td1.GetFormat()));
  EXPECT_EQ(expect_format, FORMAT_FRACTAL_Z);
  EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({4,1,16,16}));
  EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  EXPECT_EQ(VarManagerPool::Instance().GetVarManager(session.sessionId_)->GetCurVarDesc("var2", td2), SUCCESS);
  EXPECT_EQ(td2.GetShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
  EXPECT_EQ(td2.GetOriginShape().GetDims(), std::vector<int64_t>({2,2,3,2}));
}

/*
 * 用例场景：变量格式发生变化时，包含该变量的、之前下发的图，需要做重编译
 * 步骤：
 * step 1. 下发一张变量初始化图，初始化变量var1和var2，变量格式为常规ND
 * 期望：变量在变量管理器中创建成功
 * step 2. 下发一张checkpoint图，将变量var1和var2作为返回值返回
 * step 3. 下发一张训练图，var1作为权重连接到Conv2D
 * 期望： var1格式变更为FZ；var1按照新格式分配新的dev地址；var1的数据从device上被拷贝上来，在host完成格式转换后，按照新地址拷贝回dev
 * step 4. 判断step 2中的checkpoint图是否需要重编译
 * 期望：checkpoint图需要重编译
 * step 5. 删除ckp图，重新下发一次ckp图
 * 期望：ckp图上var1的变量格式为FZ，并且插入了FZ->HWCN的转换算子后作为图的输出结果返回
 */
TEST_F(VariableAccSt, variable_modified_when_second_graph_then_run_first_graph) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine(ge_env);
  system("mkdir ./build_cache_dir");

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  uint64_t pre_session_id = 0;
  options["ge.graph_compiler_cache_dir"] = "./build_cache_dir";
  {
    Session session(options);
    pre_session_id = session.GetSessionId();
    std::map<AscendString, AscendString> graph_options;
    graph_options["ge.graph_key"] = "graph_key1";
    RunAndCheckInitVarGraph(session, graph_options);
    graph_options["ge.graph_key"] = "graph_key2";
    RunAndCheckCheckpointGraph(session, false, graph_options);
    graph_options["ge.graph_key"] = "graph_key3";
    RunAndCheckTrainGraph(session, graph_options);
    EXPECT_TRUE(session.IsGraphNeedRebuild(2));
    session.RemoveGraph(2);
    graph_options["ge.graph_key"] = "graph_key2";
    RunAndCheckCheckpointGraph(session, true, graph_options);
  }
  // test load model from cache file
  {
    auto check_ret = mmAccess("./build_cache_dir/graph_key1.idx");
    EXPECT_EQ(check_ret, 0);
    check_ret = mmAccess("./build_cache_dir/graph_key2.idx");
    EXPECT_EQ(check_ret, 0);
    check_ret = mmAccess("./build_cache_dir/graph_key3.idx");
    EXPECT_EQ(check_ret, 0);
    Session session(options);
    std::map<AscendString, AscendString> graph_options;
    graph_options["ge.graph_key"] = "graph_key1";
    RunAndCheckInitVarGraph(session, graph_options);
    graph_options["ge.graph_key"] = "graph_key2";
    RunAndCheckCheckpointGraph(session, false, graph_options);
    graph_options["ge.graph_key"] = "graph_key3";
    RunAndCheckTrainGraph(session, graph_options);
    EXPECT_TRUE(session.IsGraphNeedRebuild(2));
    session.RemoveGraph(2);
    graph_options["ge.graph_key"] = "graph_key2";
    RunAndCheckCheckpointGraph(session, true, graph_options);
  }

  system("rm -rf ./build_cache_dir");
  ge_env.Reset();
  ge_env.InstallDefault();
  VarManagerPool::Instance().GetVarManager(pre_session_id)->FreeVarMemory();
}

/*
 * 用例场景：本用例测试变量回边写入场景，在写入算子没有输出ref时，最终图是正确的
 * 步骤：
 * step 1. 下发一张变量初始化图，初始化变量var1和var2，变量格式为常规ND
 * 期望：变量在变量管理器中创建成功
 * step 2. 下发一张训练图，var1作为权重连接到Conv2D
 * 期望： var1格式变更为FZ；var1按照新格式分配新的dev地址；var1的数据从device上被拷贝上来，在host完成格式转换后，按照新地址拷贝回dev
 * step 3. 下发一张训练图2，var1、var2连接到一个不支持FZ的、会写入var的算子
 * 期望：var1应该有一个较为复杂的写入图结构（详见代码校验）
 */
TEST_F(VariableAccSt, variable_write_edge_does_not_have_output) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine(ge_env);

  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);

  std::map<AscendString, AscendString> graph_options;
  RunAndCheckInitVarGraph(session, graph_options);
  RunAndCheckTrainGraph(session, graph_options);

  // step 3
  auto var_write_graph = GraphFactory::BuildVarWriteNoOutputRefGraph1();
  session.AddGraph(4, var_write_graph);
  std::vector<ge::Tensor> g1_inputs;
  Synchronizer sync;

  auto ret = session.RunGraphAsync(4, g1_inputs, [&sync](Status run_ret, std::vector<ge::Tensor> &) {
    EXPECT_EQ(run_ret, SUCCESS);
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10); // ge_env重置会清空全局对象的注册引擎等，如果这里不等待会因为异步执行线程访问正在清空
  //打桩引擎等发生coredump
  // todo check topo
  ge_env.Reset();
  ge_env.InstallDefault();
}

TEST_F(VariableAccSt, test_variable_is_initialized) {
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec = {1, 2, 3, 4, 5};
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int32_t));


  auto var1 = OP_CFG(VARIABLE).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4, 5}).InCnt(0).OutCnt(1);
  auto var_init = OP_CFG(VARISINITIALIZEDOP).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4, 5}).InCnt(1).OutCnt(1);
  auto const1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {2, 3, 4, 5}).InCnt(0).OutCnt(1);
  auto const2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT32, {5}).InCnt(0).OutCnt(1).Weight(data_tensor);
  auto reshape = OP_CFG(RESHAPE).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});
  auto relu = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});
  auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_INT32, {1, 2, 3, 4, 5});

  DEF_GRAPH(g1) {
    CHAIN(NODE("var_init", var_init)
              ->EDGE(0, 0)
              ->NODE("reshape", reshape)
              ->NODE("relu", relu)
              ->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("reshape"));
  };

  DEF_GRAPH(g2) {
    CHAIN(NODE("const1", const1)
              ->NODE("var_init", var_init)
              ->EDGE(0, 0)
              ->NODE("reshape", reshape)
              ->NODE("relu", relu)
              ->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("reshape"));
  };

  DEF_GRAPH(g3) {
    CHAIN(NODE("var1", var1)
              ->NODE("var_init", var_init)
              ->EDGE(0, 0)
              ->NODE("reshape", reshape)
              ->NODE("relu", relu)
              ->NODE("netoutput", netoutput));
    CHAIN(NODE("const2", const2)->EDGE(0, 1)->NODE("reshape"));
  };

  auto graph = ToGeGraph(g1);
  DUMP_GRAPH_WHEN("OptimizeStage1_1");

  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(2, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(2, inputs);
  // VARISINITIALIZEDOP must have input
  EXPECT_NE(ret, SUCCESS);

  graph = ToGeGraph(g2);
  ret = session.AddGraph(3, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(3, inputs);
  // VARISINITIALIZEDOP input must be varibale
  EXPECT_NE(ret, SUCCESS);

  graph = ToGeGraph(g3);
  ret = session.AddGraph(4, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  ret = session.BuildGraph(4, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(VariableAccSt, test_variable_prepare_op_pass_on_v1_control) {
  // build graph
  Graph graph = BuildGraphForVariablePrepareOpPass();
  DUMP_GRAPH_WHEN("PrepareAfterUpdateInputOutputByUserOptions");

  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(1, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // check result
  CHECK_GRAPH(PrepareAfterUpdateInputOutputByUserOptions) {
    auto var_node = graph->FindNode("Identity_assign_TO_refmerge_0");
    ASSERT_NE(var_node, nullptr);
  };
}

TEST_F(VariableAccSt, test_variable_with_assign) {
  // build graph
  Graph graph = BuildSimpleVarAssignGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  auto ref_node = compute_graph->FindNode("var1");
  auto value_node = compute_graph->FindNode("add1");
  auto assign_node = compute_graph->FindNode("assign1");
  ASSERT_NE(ref_node, nullptr);
  ASSERT_NE(value_node, nullptr);
  ASSERT_NE(assign_node, nullptr);
  ref_node->GetOpDescBarePtr()->SetId(100);
  value_node->GetOpDescBarePtr()->SetId(102);
  AssignRemovePass pass;
  pass.Run(assign_node);
  EXPECT_EQ(assign_node->GetOutNodesSize(), 1U);
}

// it suppose as ut
TEST_F(VariableAccSt, test_variable_recover_transroad_for_transpose_reshape) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("var_transpose", VARIABLE));
  };
  auto compute_graph = ToComputeGraph(g1);
  auto var = compute_graph->FindFirstNodeMatchType(VARIABLE);
  compute_graph->SetSessionID(1);
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  TransNodeInfo trans_node_info = {.node_type = TRANSDATA, .input = input_desc, .output = output_desc};
  TransNodeInfo transpose_node_info = {.node_type = TRANSPOSED, .input = input_desc, .output = output_desc};
  TransNodeInfo cast_node_info = {.node_type = CAST, .input = input_desc, .output = output_desc};
  TransNodeInfo reshape_node_info = {.node_type = RESHAPE, .input = input_desc, .output = output_desc};
  TransNodeInfo squeeze_node_info = {.node_type = SQUEEZEV2, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  fusion_road.emplace_back(transpose_node_info);
  fusion_road.emplace_back(cast_node_info);
  fusion_road.emplace_back(reshape_node_info);
  fusion_road.emplace_back(squeeze_node_info);
  VarManager::Instance(compute_graph->GetSessionID())->Init(0, 1, 0, 1);
  VarManager::Instance(compute_graph->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(compute_graph), SUCCESS);
}

uint8_t *PeekVariableAddr(const std::string &name, const GeTensorDesc &desc, uint32_t session_id) {
  auto manager = VarManager::Instance(session_id);
  uint8_t *logic_addr, *device_addr;
  EXPECT_EQ(manager->GetVarAddr(name, desc, logic_addr), SUCCESS);
  device_addr = manager->GetVarMemoryAddr(logic_addr, RT_MEMORY_HBM, 0);
  EXPECT_NE(device_addr, nullptr);
  return device_addr;
}

Graph BuildVariableInitGraph(const char *var_name, const GeTensorDesc &desc) {
  auto format = desc.GetFormat();
  auto dtype = desc.GetDataType();
  auto shape = desc.GetShape().GetDims();
  auto tensor = GenerateTensor(dtype, shape);
  auto var = OP_CFG(VARIABLE).TensorDesc(format, dtype, shape).InCnt(1).OutCnt(1)
                             .Build(var_name);
  auto constant = OP_CFG(CONSTANT).TensorDesc(format, dtype, shape).InCnt(0).OutCnt(1)
                                  .Weight(tensor)
                                  .Build("const");
  auto assign = OP_CFG(ASSIGN).TensorDesc(format, dtype, shape)
                              .InNames({"ref", "value"}).OutNames({"ref"})
                              .Build("assign");
  DEF_GRAPH(var_init) {
    CHAIN(NODE(var)->EDGE(0, 0)->NODE(assign));
    CHAIN(NODE(constant)->EDGE(0, 1)->NODE(assign));
    ADD_OUTPUT(assign, 0);
  };
  return ToGeGraph(var_init);
}

Graph BuildVariableCastGraph(const char *var_name, const GeTensorDesc &from, const GeTensorDesc &to) {
  auto var = OP_CFG(VARIABLE).TensorDesc(from.GetFormat(), from.GetDataType(), from.GetShape().GetDims())
                             .InCnt(1).OutCnt(1)
                             .Build(var_name);
  auto cast = OP_CFG(CAST).TensorDesc(to.GetFormat(), to.GetDataType(), to.GetShape().GetDims())
                          .InCnt(1).OutCnt(1)
                          .Build("cast");
  DEF_GRAPH(var_cast) {
    CHAIN(NODE(var)->NODE(cast));
  };
  return ToGeGraph(var_cast);
}

using ValueSetter = std::function<void(void *addr, size_t idx)>;
using ValueChecker = std::function<bool(const void *addr, size_t idx)>;
void EXPECT_BuildAndCheckCastFusionGraph(DataType src_dtype, DataType dst_dtype,
                                         ValueSetter setter, ValueChecker checker) {
  constexpr size_t length_1d = 256;
  const GeTensorDesc src_desc(GeShape(std::vector<int64_t>({length_1d})), FORMAT_ND, src_dtype);
  const GeTensorDesc dst_desc(GeShape(std::vector<int64_t>({length_1d})), FORMAT_ND, dst_dtype);
  auto var_name = "the_variable";
  auto var_init = BuildVariableInitGraph(var_name, src_desc);
  auto var_cast = BuildVariableCastGraph(var_name, src_desc, dst_desc);
  std::map<AscendString, AscendString> options;
  Session session(options);
  std::vector<Tensor> inputs;

  EXPECT_EQ(session.AddGraph(0, var_init), SUCCESS);
  EXPECT_EQ(session.BuildGraph(0, inputs), SUCCESS);
  uint8_t *addr_0 = PeekVariableAddr(var_name, src_desc, session.GetSessionId());
  for (size_t n = 0; n < length_1d; n++) {
    setter(addr_0, n);
    addr_0 += GetSizeByDataType(src_dtype);
  }

  EXPECT_EQ(session.AddGraph(1, var_cast), SUCCESS);
  EXPECT_EQ(session.BuildGraph(1, inputs), SUCCESS);
  uint8_t *addr_1 = PeekVariableAddr(var_name, dst_desc, session.GetSessionId());
  for (size_t n = 0; n < length_1d; n++) {
    EXPECT_TRUE(checker(addr_1, n));
    addr_1 += GetSizeByDataType(dst_dtype);
  }

  CHECK_GRAPH(PreRunAfterBuild) {
    auto node = graph->FindNode(var_name);
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->GetOpDesc()->GetOutputDesc(0).GetDataType(), dst_dtype);
    EXPECT_EQ(graph->FindNode(CAST), nullptr);
  };
}

TEST_F(VariableAccSt, test_cast_fusion_with_data_copy_ok) {
  EXPECT_BuildAndCheckCastFusionGraph(DT_FLOAT16, DT_HIFLOAT8,
    [](void *addr, size_t idx) {
      *reinterpret_cast<fp16_t *>(addr) = static_cast<fp16_t>(HiF8::FromRawBits(idx));
    },
    [](const void *addr, size_t idx) {
      return *static_cast<const hif8_t *>(addr) == HiF8::FromRawBits(idx);
    }
  );
  EXPECT_BuildAndCheckCastFusionGraph(DT_FLOAT, DT_HIFLOAT8,
    [](void *addr, size_t idx) {
      *reinterpret_cast<float *>(addr) = static_cast<float>(HiF8::FromRawBits(idx));
    },
    [](const void *addr, size_t idx) {
      return *static_cast<const hif8_t *>(addr) == HiF8::FromRawBits(idx);
    }
  );
}
