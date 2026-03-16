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
#include <condition_variable>
#include <memory>
#include <gtest/gtest.h>
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_local_context.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "flow_graph/data_flow.h"
#include "common/env_path.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "framework/common/runtime_tensor_desc.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"

#include "macro_utils/dt_public_scope.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "dflow/compiler/pne/npu/npu_process_node_engine.h"
#include "graph/ge_global_options.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/manager/mem_manager.h"
#include "utils/mock_execution_runtime.h"

using namespace testing;
using namespace std;
using namespace ge;
namespace ge {
enum class BatchType {
  kTimeBatch = 0,
  kCountBatch = 1,
};

namespace {
static ComputeGraphPtr BuildBatchComputeGraph(bool is_ok_graph, BatchType batch_type) {
  auto shape = std::vector<int64_t>{1, 2, 3};
  std::string path = "";
  std::string name = "func_name";
  std::vector<std::vector<int64_t>> shapes = {{1, 2, 3}};
  std::vector<DataType> types = {DT_FLOAT};
  std::string engine_type = "UDF";
  DEF_GRAPH(sub_graph_def) {
    auto arg_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape);

    auto neg = OP_CFG(NEG).InCnt(1).OutCnt(1);

    CHAIN(NODE("sub_data", arg_0)->NODE("neg", neg));
  };

  auto sub_graph = ToComputeGraph(sub_graph_def);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes;
  output_nodes.emplace_back(sub_graph->FindNode("neg"), 0);
  sub_graph->SetGraphOutNodesInfo(output_nodes);

  DEF_GRAPH(root_graph_def) {
    auto arg_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, shape);
    if (!is_ok_graph) {
      arg_0.Attr("_time_batch_window", -1);
    }
    if (batch_type == BatchType::kTimeBatch) {
      auto partitioned_call = OP_CFG("PartitionedCall")
                                  .InCnt(1)
                                  .OutCnt(1)
                                  .Attr("_time_batch_window", -1)
                                  .TensorDesc(FORMAT_ND, DT_INT32, shape)
                                  .Build("partitioned_call");
      partitioned_call->RegisterSubgraphIrName("f", SubgraphType::kStatic);
      partitioned_call->AddSubgraphName(sub_graph->GetName());
      partitioned_call->SetSubgraphInstanceName(0, sub_graph->GetName());
      CHAIN(NODE("root_data", arg_0)->NODE(partitioned_call));
    } else if (batch_type == BatchType::kCountBatch) {
      auto partitioned_call = OP_CFG("PartitionedCall")
                                  .InCnt(1)
                                  .OutCnt(1)
                                  .Attr("_count_batch_batch_size", 5)
                                  .TensorDesc(FORMAT_ND, DT_INT32, shape)
                                  .Build("partitioned_call");
      partitioned_call->RegisterSubgraphIrName("f", SubgraphType::kStatic);
      partitioned_call->AddSubgraphName(sub_graph->GetName());
      partitioned_call->SetSubgraphInstanceName(0, sub_graph->GetName());
      CHAIN(NODE("root_data", arg_0)->NODE(partitioned_call));
    }
  };
  auto root_graph = ToComputeGraph(root_graph_def);
  sub_graph->SetParentNode(root_graph->FindNode("partitioned_call"));
  sub_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  return root_graph;
}

void *mock_handle = nullptr;
void *mock_method = nullptr;

Status DequeueNoTilingStub(int32_t device_id, uint32_t queue_id, void *data, size_t size,
                           ExchangeService::ControlInfo &control_info) {
  RuntimeTensorDesc mbuf_tensor_desc;
  mbuf_tensor_desc.shape[0] = 4;
  mbuf_tensor_desc.shape[1] = 1;
  mbuf_tensor_desc.shape[2] = 1;
  mbuf_tensor_desc.shape[3] = 224;
  mbuf_tensor_desc.shape[4] = 224;
  mbuf_tensor_desc.dtype = static_cast<int64_t>(DT_INT64);
  mbuf_tensor_desc.data_addr = static_cast<int64_t>(reinterpret_cast<intptr_t>(data));
  if (memcpy_s(data, sizeof(RuntimeTensorDesc), &mbuf_tensor_desc, sizeof(RuntimeTensorDesc)) != EOK) {
    printf("Failed to copy mbuf data, dst size:%zu, src size:%zu\n", size, sizeof(RuntimeTensorDesc));
    return FAILED;
  }
  return 0;
}

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<ExecutionRuntimeStub>());
  auto mock_deploy_model =
    [](const FlowModelPtr &flow_model, DeployResult &deploy_result) -> Status {
    deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    deploy_result.broadcast_input_queue_attrs.resize(3);
    deploy_result.output_queue_attrs = {{4, 0, 0}};
    return SUCCESS;
  };

  // mock deploy model
  auto execution_runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), DeployModel).WillRepeatedly(
      testing::Invoke(mock_deploy_model));

  // mock undeploy model
  EXPECT_CALL(execution_runtime->GetModelDeployerStub(), Undeploy).WillRepeatedly(Return(SUCCESS));
  return SUCCESS;
}

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    return mock_handle;
  }
  void *DlSym(void *handle, const char *func_name) override {
    return mock_method;
  }

  int32_t DlClose(void *handle) override {
    return 0;
  }
};
}  // namespace
class HeterogeneousRuntimeTest : public testing::Test {
  void SetUp() {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    std::string st_dir_path = ge::PathUtils::Join({ge::EnvPath().GetAirBasePath(), "/tests/dflow/runner/st/"});
    auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
    setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  }
  void TearDown() {
    ExecutionRuntime::FinalizeExecutionRuntime();
    MmpaStub::GetInstance().Reset();
    mock_handle = nullptr;
    mock_method = nullptr;
    RuntimeStub::Reset();
    unsetenv("RESOURCE_CONFIG_PATH");
  }
};

TEST_F(HeterogeneousRuntimeTest, TestInitHeterogeneousRuntime) {
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), FAILED); // error load so
  mock_handle = (void *) 0xffffffff;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), FAILED); // error find init func
}

TEST_F(HeterogeneousRuntimeTest, TestTimeBatchPass) {
  mock_handle = (void *)0xffffffff;
  mock_method = (void *)&InitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  auto g1 = BuildBatchComputeGraph(true, BatchType::kTimeBatch);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousRuntimeTest, TestCountBatchPass) {
  mock_handle = (void *)0xffffffff;
  mock_method = (void *)&InitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  auto g1 = BuildBatchComputeGraph(true, BatchType::kCountBatch);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(g1);
  ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousRuntimeTest, TestDeployModel) {
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &InitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD);
  auto add_2 = OP_CFG(ADD);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<Tensor> input_tensors;
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> data(224 * 224 * sizeof(float));
    Tensor tensor(tensor_desc, data);
    input_tensors.emplace_back(tensor);
  }
  auto runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(runtime->GetExchangeServiceStub(), Dequeue).WillRepeatedly(Return(SUCCESS));

  std::vector<Tensor> output_tensors;
  ret = session.RunGraph(1, input_tensors, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);
}

namespace {
ge::dflow::FlowGraph BuildFlowGraph() {
  std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
  (void) system(cmd.c_str());
  std::ofstream cmakefile("./temp/CMakeLists.txt");
  {
    cmakefile << "cmake_minimum_required(VERSION 3.5)\n";
    // Prevent cmake from testing the toolchain
    cmakefile << "set(CMAKE_C_COMPILER_FORCED TRUE)\n";
    cmakefile << "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n";
    cmakefile << "project(test)\n";
    cmakefile << "set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\n";
    cmakefile << "execute_process(\n";
    cmakefile << "\tCOMMAND cp ${BASE_DIR}/libtest.so ${RELEASE_DIR}\n";
    cmakefile << ")\n";
    cmakefile << "unset(CMAKE_C_COMPILER_FORCED)\n";
    cmakefile << "unset(CMAKE_CXX_COMPILER_FORCED)\n";
  }
  {
    auto &global_options_mutex = ge::GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    auto &global_options = ge::GetMutableGlobalOptions();
    global_options[ge::OPTION_NUMA_CONFIG] =
        R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
  }
  auto data0 = ge::dflow::FlowData("Data0", 0);
  auto node0 = ge::dflow::FlowNode("node0", 1, 1).SetInput(0, data0);
  // function pp
  auto pp0 = ge::dflow::FunctionPp("func_pp0").SetCompileConfig("./pp0_config.json");
  {
    nlohmann::json pp0_cfg_json = {
        {"workspace", "./temp"}, {"target_bin", "libxxx.so"},          {"input_num", 1},
        {"output_num", 1},       {"cmakelist_path", "CMakeLists.txt"}, {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file("./pp0_config.json");
    json_file << pp0_cfg_json << std::endl;
  }
  node0.AddPp(pp0);
  std::vector<ge::dflow::FlowOperator> inputsOperator{data0};
  std::vector<ge::dflow::FlowOperator> outputsOperator{node0};
  ge::dflow::FlowGraph flow_graph("flow_graph");
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph;
}
}

TEST_F(HeterogeneousRuntimeTest, TestDeployModelNoTiling) {
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &InitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto add_2 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor0_range;
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0.SetShapeRange(tensor0_range);
    TensorUtils::SetSize(tensor0, 501760);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor1_range;
    tensor1_range.push_back(std::make_pair(1, 1));
    tensor1_range.push_back(std::make_pair(1, 10));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1.SetShapeRange(tensor1_range);
    TensorUtils::SetSize(tensor1, 501760);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
        vector<std::string> tiling_inline;
        vector<std::string> export_shape;
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        tiling_inline.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
        export_shape.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
      } else {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph, nullptr);
  SetUnknownOpKernelForNoTiling(compute_graph->GetDirectNode());

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  options["ge.exec.enableFusion"] = "true";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<Tensor> input_tensors;
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> data(224 * 224 * sizeof(float));
    Tensor tensor(tensor_desc, data);
    input_tensors.emplace_back(tensor);
  }

  auto runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(runtime->GetExchangeServiceStub(), Dequeue).WillRepeatedly(testing::Invoke(DequeueNoTilingStub));

  std::vector<Tensor> output_tensors;
  ret = session.RunGraph(1, input_tensors, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);
  auto dflow_graph = BuildFlowGraph();
  session.AddGraph(11, dflow_graph.ToGeGraph(), options);
  DataFlowInfo data_flow_info;
  data_flow_info.SetTransactionId(100);
  ret = session.FeedDataFlowGraph(11, input_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  std::vector<Tensor> fetch_output_tensors;
  ret = session.FetchDataFlowGraph(11, fetch_output_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(fetch_output_tensors.size(), 1);
  
  // feed empty data with eos
  data_flow_info.SetFlowFlags(1U);
  ret = session.FeedDataFlowGraph(11, {}, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  fetch_output_tensors.clear();
  ret = session.FetchDataFlowGraph(11, fetch_output_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(fetch_output_tensors.size(), 1);
}

TEST_F(HeterogeneousRuntimeTest, TestFeedDataWithoutRun) {
  map<std::string, std::string> sess_options = ge::GetThreadLocalContext().GetAllSessionOptions();
  GE_MAKE_GUARD(recover_sess_cfg, [&sess_options](){
    GetThreadLocalContext().SetSessionOption(sess_options);
  });
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &InitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto add_2 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor0_range;
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0.SetShapeRange(tensor0_range);
    TensorUtils::SetSize(tensor0, 501760);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor1_range;
    tensor1_range.push_back(std::make_pair(1, 1));
    tensor1_range.push_back(std::make_pair(1, 10));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1.SetShapeRange(tensor1_range);
    TensorUtils::SetSize(tensor1, 501760);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
        vector<std::string> tiling_inline;
        vector<std::string> export_shape;
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        tiling_inline.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
        export_shape.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
      } else {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph, nullptr);
  SetUnknownOpKernelForNoTiling(compute_graph->GetDirectNode());

  auto graph = BuildFlowGraph().ToGeGraph();
  map<AscendString, AscendString> options;
  options["ge.exec.enableFusion"] = "true";
  options["ge.runFlag"] = "0";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<Tensor> input_tensors;
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> data(224 * 224 * sizeof(float));
    Tensor tensor(tensor_desc, data);
    input_tensors.emplace_back(tensor);
  }

  auto runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(runtime->GetExchangeServiceStub(), Dequeue).WillRepeatedly(testing::Invoke(DequeueNoTilingStub));
  DataFlowInfo data_flow_info;
  ret = session.FeedDataFlowGraph(1, input_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(HeterogeneousRuntimeTest, TestDeployDynamicSchedModelNoTiling) {
  mock_handle = (void *) 0xffffffff;
  mock_method = (void *) &InitializeHeterogeneousRuntime;
  std::map<std::string, std::string> options_runtime;
  ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);

  std::vector<std::string> engine_list = {"AIcoreEngine"};
  auto add_1 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto add_2 = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG(DATA);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add_1)->EDGE(0, 0)->NODE("add_2", add_2));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add_1));
    CHAIN(NODE("data_3", data3)->EDGE(0, 1)->NODE("add_2", add_2));
    CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("Node_Output", NETOUTPUT));
  };

  const auto SetUnknownOpKernelForNoTiling = [](const ComputeGraph::Vistor<NodePtr> &all_nodes) {
    GeTensorDesc tensor0(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor0_range;
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(1, 1));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0_range.push_back(std::make_pair(224, 224));
    tensor0.SetShapeRange(tensor0_range);
    TensorUtils::SetSize(tensor0, 501760);
    AttrUtils::SetBool(tensor0, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor0, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0);
    std::vector<int64_t> max_shape_list = {1, 10, 224, 224};
    AttrUtils::SetListInt(tensor0, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    GeTensorDesc tensor1(GeShape({1, -1, 224, 224}), FORMAT_NCHW, DT_INT64);
    std::vector<std::pair<int64_t, int64_t>> tensor1_range;
    tensor1_range.push_back(std::make_pair(1, 1));
    tensor1_range.push_back(std::make_pair(1, 10));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1_range.push_back(std::make_pair(224, 224));
    tensor1.SetShapeRange(tensor1_range);
    TensorUtils::SetSize(tensor1, 501760);
    AttrUtils::SetBool(tensor1, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
    AttrUtils::SetInt(tensor1, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
    AttrUtils::SetListInt(tensor1, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list);

    for (const auto &node : all_nodes) {
      const auto op_desc = node->GetOpDesc();
      if (op_desc->GetType() == DATA) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateOutputDesc(0, tensor0);
        op_desc->SetOutputOffset({2048});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
      } else if (op_desc->GetType() == ADD) {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor0);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2048});
        op_desc->SetOutputOffset({2112});
        op_desc->SetWorkspace({});
        op_desc->SetWorkspaceBytes({});
        vector<std::string> tiling_inline;
        vector<std::string> export_shape;
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        tiling_inline.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline);
        AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
        export_shape.push_back("AIcoreEngine");
        AttrUtils::SetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, export_shape);
      } else {
        op_desc->SetOpKernelLibName("AiCoreLib");
        op_desc->SetOpEngineName("AIcoreEngine");
        op_desc->UpdateInputDesc(0, tensor1);
        op_desc->UpdateOutputDesc(0, tensor1);
        op_desc->SetInputOffset({2112});
        op_desc->SetSrcName( { "add" } );
        op_desc->SetSrcIndex({ 0 });
      }
    }
  };
  auto compute_graph = ToComputeGraph(g1);
  EXPECT_NE(compute_graph, nullptr);
  SetUnknownOpKernelForNoTiling(compute_graph->GetDirectNode());
  (void)AttrUtils::SetBool(compute_graph, "dynamic_schedule_enable", true);

  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_EQ(ret, SUCCESS);

  Shape shape({1, 1, 224, 224});
  TensorDesc tensor_desc(shape, FORMAT_NCHW, DT_FLOAT);
  std::vector<Tensor> input_tensors;
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> data(224 * 224 * sizeof(float));
    Tensor tensor(tensor_desc, data);
    input_tensors.emplace_back(tensor);
  }

  auto runtime = (ExecutionRuntimeStub *)ExecutionRuntime::GetInstance();
  EXPECT_CALL(runtime->GetExchangeServiceStub(), Dequeue).WillRepeatedly(testing::Invoke(DequeueNoTilingStub));

  std::vector<Tensor> output_tensors;
  ret = session.RunGraph(1, input_tensors, output_tensors);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(output_tensors.size(), 1);
  auto dflow_graph = BuildFlowGraph();
  ASSERT_EQ(session.AddGraph(11, dflow_graph.ToGeGraph(), options), SUCCESS);
  DataFlowInfo data_flow_info;
  ret = session.FeedDataFlowGraph(11, input_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  std::vector<Tensor> fetch_output_tensors;
  ret = session.FetchDataFlowGraph(11, fetch_output_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(fetch_output_tensors.size(), 1);

  // feed empty data with eos
  data_flow_info.SetFlowFlags(1U);
  ret = session.FeedDataFlowGraph(11, {}, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  fetch_output_tensors.clear();
  ret = session.FetchDataFlowGraph(11, fetch_output_tensors, data_flow_info, 100);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(fetch_output_tensors.size(), 1);
}
}  // namespace ge
