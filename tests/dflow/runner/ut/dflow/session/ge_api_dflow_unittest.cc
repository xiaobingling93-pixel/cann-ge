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

#include <vector>
#include <string>
#include <map>
#include "common/share_graph.h"
#include "stub/gert_runtime_stub.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "ge/ge_api.h"
#include "graph/utils/graph_utils_ex.h"
#include "register/ops_kernel_builder_registry.h"
#include "register/optimization_option_registry.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/operator_factory_impl.h"
#include "ge_running_env/fake_op.h"
#include "api/gelib/gelib.h"
#include "graph/build/stream/stream_utils.h"
#include "ge_running_env/fake_engine.h"
#include "nlohmann/json.hpp"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "flow_graph/flow_graph.h"
#include "ge/ge_api_v2.h"
#include "ge_running_env/dir_env.h"

using namespace std;

namespace ge {
using Json = nlohmann::json;
namespace {
Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  return SUCCESS;
}
class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return dlsym(handle, func_name);
  }
};


int32_t g_so_addr = 0;
class MockMmpa1 : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libmodel_deployer.so") == file_name) {
      return (void *) &g_so_addr;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return dlsym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    return 0;
  }
};

class GeFakeOpsKernelBuilder : public OpsKernelBuilder {
 public:
  GeFakeOpsKernelBuilder() = default;

 private:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  };
  Status Finalize() override {
    return SUCCESS;
  };
  Status CalcOpRunningParam(Node &node) override {
    return SUCCESS;
  };
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    return SUCCESS;
  };
};
}  // namespace

class UtestGeApiDflow : public testing::Test {
 protected:

  static void SetUpTestSuite() {
    // Init running dir env
    ge::DirEnv::GetInstance().InitEngineConfJson();
    const std::map<AscendString, AscendString> options{};
    // now DFlowInitialize is not include GEInitialize, so add here.
    EXPECT_EQ(ge::GEInitializeV2(options), SUCCESS);
    // ge session depend g_session_manager init
    EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
  }
  static void TearDownTestSuite() {
    ge::GEFinalizeV2();
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  void SetUp() override {
    const auto env_ptr = getenv("LD_PRELOAD");
    if (env_ptr != nullptr) {
      env = env_ptr;
      unsetenv("LD_PRELOAD");
    }
    OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("Add", [](Operator &op) {return GRAPH_SUCCESS;});
    OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});
    GetThreadLocalContext().SetGlobalOption({});
    GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().SetGraphOption({});
    GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
  }

  void TearDown() override {
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
    OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
    RuntimeStub::Reset();
    if (!env.empty()) {
      setenv("LD_PRELOAD", env.c_str(), 1);
    }
  }

  std::string env;
};

TEST_F(UtestGeApiDflow, Feed_test_not_init) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<Tensor> inputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FeedDataFlowGraph(1, inputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiDflow, Feed_test) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<Tensor> inputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FeedDataFlowGraph(1, inputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiDflow, Fetch_test_not_init) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FetchDataFlowGraph(1, outputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiDflow, Fetch_test) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FetchDataFlowGraph(1, outputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiDflow, Fetch_flow_msg_graph_not_exist) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<FlowMsgPtr> outputs;
  auto ret = session.FetchDataFlowGraph(1, outputs, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApiDflow, InitializeExecutionRuntime_test) {
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ExecutionRuntime::instance_ = nullptr;
  ExecutionRuntime::handle_ = (void *)0xffffffff;
  std::map<std::string, std::string> options;
  ExecutionRuntime::handle_ = nullptr;
  ExecutionRuntime::instance_ = nullptr;
  MmpaStub::GetInstance().Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
}

namespace {

ge::dflow::FlowGraph BuildFlowGraph() {
  std::string cmd = "mkdir -p temp; cd temp; touch libtest.so";
  (void) system(cmd.c_str());
  {
    std::ofstream cmakefile("./temp/CMakeLists.txt");
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

TEST_F(UtestGeApiDflow, feed_graph_without_run) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  setenv("RESOURCE_CONFIG_PATH", "./stub_resource_config_path.json", 0);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa1>());
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();
  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<Tensor> inputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  inputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options["ge.runFlag"] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  Session session(options);
  auto graph = BuildFlowGraph();

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph.ToGeGraph(), options), SUCCESS);
  DataFlowInfo data_flow_info;
  EXPECT_EQ(session.FeedDataFlowGraph(graph_id, inputs, data_flow_info, 0), SUCCESS);
  std::vector<Tensor> outputs;
  DataFlowInfo info;
  EXPECT_NE(session.FetchDataFlowGraph(graph_id, {0}, outputs, info, 0), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestGeApiDflow, Run_graph) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  setenv("RESOURCE_CONFIG_PATH", "./stub_resource_config_path.json", 0);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa1>());
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();
  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<Tensor> inputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  inputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  ge::GetThreadLocalContext().SetGraphOption(options);
  Session session(options);
  auto graph = BuildFlowGraph();

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph.ToGeGraph(), options), SUCCESS);
  std::vector<Tensor> outputs;
  // without deployer
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);

  std::map<std::string, std::string> options_no_run;
  options_no_run[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options_no_run["ge.runFlag"] = "0";
  Session session_not_run(options_no_run);
  EXPECT_EQ(session_not_run.AddGraph(graph_id, graph.ToGeGraph(), options), SUCCESS);
  EXPECT_EQ(session_not_run.RunGraph(graph_id, inputs, outputs), SUCCESS);

  unsetenv("RESOURCE_CONFIG_PATH");
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestGeApiDflow, feed_graph_by_rawdata) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<Tensor> inputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  inputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options["ge.runFlag"] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);

  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);
  Session session2(options);

  GraphId graph_id = 1;
  EXPECT_EQ(session2.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session2.BuildGraph(graph_id, inputs), SUCCESS);
  // without executor
  DataFlowInfo data_flow_info;
  uint64_t sample_data = 100;
  RawData raw_data = {.addr = reinterpret_cast<void *>(&sample_data), .len = sizeof(uint64_t)};
  EXPECT_NE(session2.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), SUCCESS);
}

TEST_F(UtestGeApiDflow, feed_graph_by_flow_msg) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();
  setenv("RESOURCE_CONFIG_PATH", "./stub_resource_config_path.json", 0);
  ge::ProcessNodeEngineRegisterar ps_engine_register __attribute__((unused)) (
      "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa1>());
  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<Tensor> inputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  inputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options["ge.runFlag"] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  Session session1(options);
  GraphId graph_id = 1;

  // not initilized
  auto tensor_msg = FlowBufferFactory::AllocTensorMsg({3, 3, 3}, DT_FLOAT);
  EXPECT_EQ(session1.FeedDataFlowGraph(graph_id, {tensor_msg, tensor_msg}, 0), FAILED);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  auto graph = BuildFlowGraph();
  Session session2(options);

  EXPECT_EQ(session2.AddGraph(graph_id, graph.ToGeGraph(), options), SUCCESS);
  EXPECT_EQ(session2.BuildGraph(graph_id, inputs), SUCCESS);
  // without run
  EXPECT_EQ(session2.FeedDataFlowGraph(graph_id, {tensor_msg, tensor_msg}, 0), SUCCESS);
  std::vector<FlowMsgPtr> outs;
  EXPECT_NE(session2.FetchDataFlowGraph(graph_id, {0}, outs, 0), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
  MmpaStub::GetInstance().Reset();
}
}  // namespace ge
