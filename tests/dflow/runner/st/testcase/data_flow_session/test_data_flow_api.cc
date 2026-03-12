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
#include <fstream>
#include "nlohmann/json.hpp"
#include "depends/mmpa/src/mmpa_stub.h"
#include "ge/st/stubs/utils/mock_execution_runtime.h"

#include "dflow/compiler/session/dflow_api.h"
#include "depends/slog/src/slog_stub.h"
#include "common/ge_common/ge_types.h"
#include "graph/ge_local_context.h"
#include "register/optimization_option_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "api/gelib/gelib.h"
#include "init_ge.h"
#include "ge/ge_api.h" // 等GE提供真正的依赖接口
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "common/env_path.h"

using namespace testing;
namespace ge {
namespace dflow {
using ge::Operator;
using ge::GRAPH_SUCCESS;
using ge::SUCCESS;
using ge::FAILED;
using ge::PARAM_INVALID;
using ge::ComputeGraphPtr;
using ge::Node;
using ge::RunContext;
using ge::TensorDesc;
using ge::INVALID_SESSION_ID;

Status InitializeExecutionRuntimeStub(const std::map<std::string, std::string> &options) {
  auto stub = std::make_shared<ge::ExecutionRuntimeStub>();
  ge::ExecutionRuntime::SetExecutionRuntime(stub);
  return stub->Initialize(options);
}

static int32_t g_so_addr = 0;
static void* g_handle = &(g_so_addr);
class MockMmpaDeployer : public ge::MmpaStubApiGe {
  public:
  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libmodel_deployer.so") {
      return g_handle;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &(InitializeExecutionRuntimeStub);
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == g_handle) {
      return 0;
    }
    return dlclose(handle);
  }
};
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
class DataFlowApiTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string st_dir_path = ge::PathUtils::Join({ge::EnvPath().GetAirBasePath(), "/tests/dflow/runner/st/"});
    auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
    setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  }
  static void TearDownTestSuite() {
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  void SetUp() {
    ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDeployer>());
    ge::GEFinalize();
    dflow::DFlowFinalize();
    std::map<AscendString, AscendString> init_options = {};
    EXPECT_EQ(ge::GEInitialize(init_options), SUCCESS);
    EXPECT_EQ(dflow::DFlowInitialize(init_options), SUCCESS);

    ge::GeRunningEnvFaker ge_env;
    auto multi_dims = ge::MakeShared<ge::FakeMultiDimsOptimizer>();
    ge_env.Install(ge::FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(ge::PriorityEnum::COST_0));
    ge_env.Install(ge::FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(ge::PriorityEnum::COST_1));
    ge_env.Install(ge::FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(ge::PriorityEnum::COST_3));
    ge_env.Install(ge::FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(ge::PriorityEnum::COST_2));
    ge_env.Install(ge::FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(ge::PriorityEnum::COST_1));
    ge_env.Install(ge::FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(ge::PriorityEnum::COST_1));
    ge_env.Install(ge::FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(ge::PriorityEnum::COST_9));
    ge_env.Install(ge::FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(ge::PriorityEnum::COST_10));
    ge_env.Install(ge::FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(ge::PriorityEnum::COST_1));
    ge_env.Install(ge::FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
    ge_env.Install(ge::FakeOp("Data").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
    ge_env.Install(ge::FakeOp("Add").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
    ge_env.Install(ge::FakeOp("NetOutput").InfoStoreAndBuilder("AiCoreLib"));

    ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
        ge::PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });

    // mock deploy model
    auto execution_runtime = (ge::ExecutionRuntimeStub *)ge::ExecutionRuntime::GetInstance();
    EXPECT_CALL(execution_runtime->GetModelDeployerStub(), DeployModel).WillRepeatedly(Return(SUCCESS));
    EXPECT_CALL(execution_runtime->GetModelDeployerStub(), Undeploy).WillRepeatedly(Return(SUCCESS));
  }

  void TearDown() {
    ge::GEFinalize();
    dflow::DFlowFinalize();
    {
      auto &global_options_mutex = ge::GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = ge::GetMutableGlobalOptions();
      if (global_options.find(ge::OPTION_NUMA_CONFIG) == global_options.end()) {
        global_options.erase(ge::OPTION_NUMA_CONFIG);
      }
    }
    ge::MmpaStub::GetInstance().Reset();
  }
};

TEST_F(DataFlowApiTest, AddCompileRunGraph) {
  auto graph = BuildFlowGraph();
  std::map<AscendString, AscendString> graph_options = {};
  DFlowSession session(graph_options);
  EXPECT_NE(session.GetSessionId(), INVALID_SESSION_ID);
  ASSERT_EQ(session.AddGraph(1, graph, graph_options), SUCCESS);
  vector<Tensor> inputs;
  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);
}

TEST_F(DataFlowApiTest, AddFeedFetchGraph) {
  std::map<AscendString, AscendString> init_options = {};
  // init twice
  EXPECT_EQ(dflow::DFlowInitialize(init_options), SUCCESS);
  auto graph = BuildFlowGraph();
  std::map<AscendString, AscendString> graph_options = {};
  DFlowSession session(graph_options);
  EXPECT_NE(session.GetSessionId(), INVALID_SESSION_ID);
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  std::string invalid_key = "key";
  std::string invalid_value = "value";
  graph_options[invalid_key.c_str()] = invalid_value.c_str();
  ASSERT_EQ(session.AddGraph(1, graph, graph_options), SUCCESS);
  EXPECT_EQ(session.FeedDataFlowGraph(1, inputs, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(session.FetchDataFlowGraph(1, outputs, data_flow_info, 0), SUCCESS);
  std::vector<FlowMsgPtr> inputs_flow_msg;
  std::vector<FlowMsgPtr> output_flow_msg;
  EXPECT_EQ(session.FeedDataFlowGraph(1, inputs_flow_msg, 0), SUCCESS);
  EXPECT_EQ(session.FetchDataFlowGraph(1, outputs, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(session.FetchDataFlowGraph(1, output_flow_msg, 0), SUCCESS);
  ASSERT_EQ(session.RemoveGraph(1), SUCCESS);
  ASSERT_EQ(session.RemoveGraph(100), FAILED); // id not existed
}

TEST_F(DataFlowApiTest, OperateSessionWithoutDflowInit) {
  dflow::DFlowFinalize();
  std::map<AscendString, AscendString> options = {};
  DFlowSession session(options);
  DFlowSession session1(options);
  EXPECT_EQ(session1.GetSessionId(), INVALID_SESSION_ID);
  auto graph = BuildFlowGraph();
  ASSERT_EQ(session.AddGraph(1, graph, options), FAILED);
  ASSERT_EQ(session.RemoveGraph(1), FAILED);
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  ASSERT_EQ(session.BuildGraph(1, inputs), FAILED);
  DataFlowInfo data_flow_info;
  EXPECT_EQ(session.FeedDataFlowGraph(1, inputs, data_flow_info, 0), FAILED);
  EXPECT_EQ(session.FetchDataFlowGraph(1, outputs, data_flow_info, 0), FAILED);
  uint64_t sample_data = 100;
  ge::RawData raw_data = {.addr = reinterpret_cast<void *>(&sample_data), .len = sizeof(uint64_t)};
  EXPECT_EQ(session.FeedRawData(1, {raw_data}, 0, data_flow_info, 0), FAILED);
}

TEST_F(DataFlowApiTest, FeedRawData) {
  vector<Tensor> inputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(ge::Shape({3, 3, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(ge::Shape({3, 3, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  inputs.emplace_back(tensor2);

  std::map<AscendString, AscendString> empty_options;
  std::map<AscendString, AscendString> no_run_session_options;
  no_run_session_options["ge.runFlag"] = "0";
  DFlowSession session1(no_run_session_options);
  uint32_t graph_id = 1;
  DataFlowInfo data_flow_info;

  // not initialized
  uint64_t sample_data = 100;
  ge::RawData raw_data = {.addr = reinterpret_cast<void *>(&sample_data), .len = sizeof(uint64_t)};
  EXPECT_EQ(session1.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), FAILED);
  EXPECT_EQ(DFlowInitialize(empty_options), SUCCESS);
  auto graph = BuildFlowGraph();
  DFlowSession session2(no_run_session_options);

  EXPECT_EQ(session2.AddGraph(graph_id, graph, empty_options), SUCCESS);
  EXPECT_EQ(session2.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), FAILED); // not build
  EXPECT_EQ(session2.BuildGraph(graph_id, inputs), SUCCESS);
  // runFlag is false, no need feed
  EXPECT_EQ(session2.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), SUCCESS);

  DFlowSession session3(empty_options);
  EXPECT_EQ(session3.AddGraph(graph_id, graph, empty_options), SUCCESS);
  EXPECT_EQ(session3.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_NE(session3.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
}
}  // namespace dflow
}  // namespace ge
