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
#include <nlohmann/json.hpp>
#include "dflow/compiler/session/dflow_api.h"
#include "graph/operator_factory_impl.h"
#include "depends/slog/src/slog_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/share_graph.h"
#include "common/ge_common/ge_types.h"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "register/optimization_option_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "register/ops_kernel_builder_registry.h"
#include "register/optimization_option_registry.h"
#include "stub/gert_runtime_stub.h"
#include "ge/graph/ops_stub.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/dir_env.h"
#include "ge/ge_api.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "dflow/executor/heterogeneous_model_executor.h"
#include "dflow/base/deploy/model_deployer.h"
#include "dflow/base/deploy/exchange_service.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/manager/graph_manager.h"
#include "dflow/compiler/session/dflow_session_impl.h"
#include "dflow/compiler/pne/npu/npu_process_node_engine.h"

namespace ge {
class MockExchangeService : public ExchangeService {
 public:
  Status CreateQueue(int32_t device_id,
                     const string &name,
                     const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    queue_id = queue_id_gen_++;
    return SUCCESS;
  }

  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const void *const data,
                const size_t size, const ControlInfo &control_info) {
    return SUCCESS;
  }

  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const size_t size,
                       const FillFunc &fill_func, const ControlInfo &control_info) {
    return SUCCESS;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout) override {
    return SUCCESS;
  }
  Status Dequeue(const int32_t device_id, const uint32_t queue_id, void *const data, const size_t size,
                 ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id,
                                 std::shared_ptr<AlignedPtr> &aligned_ptr, const size_t size,
                                 ControlInfo &control_info) override {
    return SUCCESS;
  }

  Status DequeueTensor(const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                              ControlInfo &control_info) override {
    return SUCCESS;
  }
  Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) override {
    return SUCCESS;
  }
  Status DestroyQueue(int32_t device_id, uint32_t queue_id) override {
    return SUCCESS;
  }

  int queue_id_gen_ = 100;
};

class MockModelDeployer : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model,
                     DeployResult &deploy_result) override {
    return SUCCESS;
  }
  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
  }
  Status GetDeviceMeshIndex(const int32_t device_id, std::vector<int32_t> &node_mesh_index) override {
    node_mesh_index = {0, 0, device_id, 0};
    return SUCCESS;
  }
};

class MockExecutionRuntime : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }

 private:
  MockModelDeployer model_deployer_;
  MockExchangeService exchange_service_;
};

namespace dflow {
namespace {
class TestProcessNodeEngine : public ge::NPUProcessNodeEngine {
  Status Initialize(const std::map<std::string, std::string> &options) override {
    (void) options;
    return FAILED;
  }
};

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

class GeFakeOpsKernelBuilder : public ge::OpsKernelBuilder {
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

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  return SUCCESS;
}
int32_t g_so_addr = 0;
class MockMmpa : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libmodel_deployer.so") == file_name) {
      return (void *) &g_so_addr;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (handle == &g_so_addr) {
      if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
        return (void *)&InitializeHeterogeneousRuntime;
      }
      return nullptr;
    }
    return dlsym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    if (handle == &g_so_addr) {
      return 0;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
};

class UtestDflowApi : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Init running dir env
    ge::DirEnv::GetInstance().InitEngineConfJson();
    const std::map<AscendString, AscendString> options{};
    // now DFlowInitialize is not include GEInitialize, so add here.
    EXPECT_EQ(ge::GEInitialize(options), SUCCESS);
    setenv("RESOURCE_CONFIG_PATH", "./stub_resource_config_path.json", 0);
  }
  static void TearDownTestSuite() {
    ge::GEFinalize();
    unsetenv("RESOURCE_CONFIG_PATH");
  }

  void SetUp() override {
    ge::OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
    ge::OperatorFactoryImpl::RegisterInferShapeFunc("Add", [](Operator &op) {return GRAPH_SUCCESS;});
    ge::OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetSessionOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().GetOo().Initialize({}, ge::OptionRegistry::GetInstance().GetRegisteredOptTable());
    ge::GeRunningEnvFaker().Reset()
      .Install(ge::FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(ge::FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(ge::FakeOp("Data").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(ge::FakeOp("FlowNode").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(ge::FakeOp("NetOutput").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"));
    ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
        "UDF", []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });

    ge::ExecutionRuntime::SetExecutionRuntime(make_shared<ge::MockExecutionRuntime>());
  }

  void TearDown() override {
    ge::OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
    ge::OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
    ge::OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
    ge::RuntimeStub::Reset();
    EXPECT_EQ(DFlowFinalize(), SUCCESS);
    {
      auto &global_options_mutex = ge::GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = ge::GetMutableGlobalOptions();
      if (global_options.find(ge::OPTION_NUMA_CONFIG) == global_options.end()) {
        global_options.erase(ge::OPTION_NUMA_CONFIG);
      }
    }
    ge::ExecutionRuntime::SetExecutionRuntime(nullptr);
  }
};

TEST_F(UtestDflowApi, DFlowInitialize) {
  std::map<AscendString, AscendString> options = {};
  std::string empty_key = "";
  std::string fake_key = "fake_key";
  std::string value = "0";
  options[fake_key.c_str()] = value.c_str();
  options[empty_key.c_str()] = value.c_str();
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  auto ret = DFlowInitialize(options);
  ASSERT_NE(ret, SUCCESS);
  options.erase(AscendString(empty_key.c_str()));
  ret = DFlowInitialize(options);
  ASSERT_EQ(ret, SUCCESS);
  ret = DFlowInitialize(options);
  ASSERT_EQ(ret, SUCCESS);
  ret = DFlowFinalize();
  ASSERT_EQ(ret, SUCCESS);
  ret = DFlowFinalize();
  ASSERT_EQ(ret, SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, DFlowInitialize_pne_init_failed) {
  auto engine = std::make_shared<TestProcessNodeEngine>();
  EXPECT_NE(engine, nullptr);
  ge::ProcessNodeEngineManager::GetInstance().init_flag_ = false;
  EXPECT_EQ(ge::ProcessNodeEngineManager::GetInstance().RegisterEngine("TEST", engine, nullptr), SUCCESS);

  std::map<AscendString, AscendString> options = {};
  std::string fake_key = "fake_key";
  std::string value = "0";
  options[fake_key.c_str()] = value.c_str();
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  auto ret = DFlowInitialize(options);
  ASSERT_NE(ret, SUCCESS);
  ret = DFlowFinalize();
  ASSERT_EQ(ret, SUCCESS);

  ge::ProcessNodeEngineManager::GetInstance().init_flag_ = true;
  EXPECT_EQ(ge::ProcessNodeEngineManager::GetInstance().Finalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, GetSessionId) {
  std::map<AscendString, AscendString> options;
  std::string key_str = "ge.options_unknown";
  std::string value = "0";
  options[key_str.c_str()] = value.c_str();
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  EXPECT_EQ(session.GetSessionId(), 0);
  DFlowSession session2(options);
  // because we init ge_session in dflow session, temporarily
  EXPECT_EQ(session2.GetSessionId(), 2);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, RemoveGraph) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::string key_str = "ge.Fake";
  std::string value = "0";
  options[key_str.c_str()] = value.c_str();
  auto graph = BuildFlowGraph();
  {
    DFlowSession session(options);
    ASSERT_EQ(session.AddGraph(graph_id, graph, options), FAILED); // not init
    ASSERT_EQ(session.RemoveGraph(graph_id), FAILED); // not init
  }

  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  EXPECT_NE(session.RemoveGraph(graph_id), SUCCESS); // not add graph

  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), FAILED); // add twice
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, BuildGraphTest) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  vector<Tensor> inputs;
  {
    DFlowSession session(options);
    ASSERT_EQ(session.BuildGraph(graph_id, inputs), FAILED); // not init
  }

  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  ASSERT_EQ(session.BuildGraph(graph_id, inputs), FAILED); // not add graph

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, BuildGraphTestInGESession) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  vector<Tensor> inputs;
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  ge::Session session(options);

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph.ToGeGraph()), SUCCESS);
  EXPECT_EQ(session.BuildGraph(100, inputs), FAILED); // not add graph
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, CompileGraphTestInGESession) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  ge::Session session(options);

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph.ToGeGraph()), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, BuildGraphWithTensorInfoTestInGESession) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  vector<ge::InputTensorInfo> inputs;
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  ge::Session session(options);

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph.ToGeGraph()), SUCCESS);
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, FeedDataFlowGraphFetch) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::string key_str = "ge.Fake";
  std::string value = "0";
  options[key_str.c_str()] = value.c_str();
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  {
    DFlowSession session(options);
    ASSERT_EQ(session.FeedDataFlowGraph(graph_id, inputs, data_flow_info, 0), FAILED); // not init
  }

  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  ASSERT_EQ(session.FeedDataFlowGraph(graph_id, inputs, data_flow_info, 0), FAILED); // not add graph

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.FeedDataFlowGraph(graph_id, inputs, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(session.FetchDataFlowGraph(graph_id, outputs, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, FeedDataFlowGraphWithoutRun) {
  std::map<AscendString, AscendString> options;
  std::string key_str = "ge.runFlag";
  std::string value = "0";
  options[key_str.c_str()] = value.c_str();
  vector<Tensor> inputs1;
  vector<FlowMsgPtr> inputs2;
  DataFlowInfo data_flow_info;

  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  auto graph1 = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(1, graph1), SUCCESS);
  EXPECT_EQ(session.FeedDataFlowGraph(1, inputs1, data_flow_info, 0), SUCCESS);
  auto graph2 = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(2, graph2), SUCCESS);
  EXPECT_EQ(session.FeedDataFlowGraph(2, inputs2, 0), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}


TEST_F(UtestDflowApi, FeedDataFlowGraphFetchWithoutAddGraph) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  DFlowSession session(options);
  EXPECT_EQ(session.FetchDataFlowGraph(graph_id, outputs, data_flow_info, 0), FAILED); // not init
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, FeedDataFlowGraphWithIndex) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::string key_str = "ge.Fake";
  std::string value = "0";
  options[key_str.c_str()] = value.c_str();
  std::vector<uint32_t> indexes;
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  {
    DFlowSession session(options);
    ASSERT_EQ(session.FeedDataFlowGraph(graph_id, indexes, inputs, data_flow_info, 0), FAILED); // not init
  }

  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  ASSERT_EQ(session.FeedDataFlowGraph(graph_id, indexes, inputs, data_flow_info, 0), FAILED); // not add graph

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.FeedDataFlowGraph(graph_id, indexes, inputs, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(session.FetchDataFlowGraph(graph_id, indexes, outputs, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, FeedDataFlowGraphWithFlowMsgAndIndex) {
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, FeedDataFlowGraphWithFlowMsg) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::string key_str = "ge.Fake";
  std::string value = "0";
  options[key_str.c_str()] = value.c_str();
  vector<FlowMsgPtr> inputs;
  vector<FlowMsgPtr> outputs;
  DataFlowInfo data_flow_info;
  {
    DFlowSession session(options);
    ASSERT_EQ(session.FeedDataFlowGraph(graph_id, inputs, 0), FAILED); // not init
  }

  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
  ASSERT_EQ(session.FeedDataFlowGraph(graph_id, inputs, 0), FAILED); // not add graph

  auto graph = BuildFlowGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);

  EXPECT_EQ(session.FeedDataFlowGraph(graph_id, inputs, 0), SUCCESS);
  EXPECT_EQ(session.FetchDataFlowGraph(graph_id, outputs, 0), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, FeedRawData) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  ge::OpsKernelBuilderPtr builder = ge::MakeShared<GeFakeOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameAiCore, builder);
  ge::OpsKernelBuilderRegistry::GetInstance().Register(ge::kEngineNameGeLocal, builder);
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

  std::map<AscendString, AscendString> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options["ge.runFlag"] = "0";
  DFlowSession session1(options);
  uint32_t graph_id = 1;
  DataFlowInfo data_flow_info;

  // not initialized
  uint64_t sample_data = 100;
  ge::RawData raw_data = {.addr = reinterpret_cast<void *>(&sample_data), .len = sizeof(uint64_t)};
  EXPECT_EQ(session1.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), FAILED);
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  auto graph = BuildFlowGraph();
  DFlowSession session2(options);

  EXPECT_EQ(session2.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session2.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), FAILED); // not build

  EXPECT_EQ(session2.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_EQ(session2.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), SUCCESS);

  std::map<AscendString, AscendString> empty_options;
  DFlowSession session3(empty_options);
  EXPECT_EQ(session3.AddGraph(graph_id, graph, empty_options), SUCCESS);
  EXPECT_EQ(session3.BuildGraph(graph_id, inputs), SUCCESS);
  EXPECT_NE(session3.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(DFlowFinalize(), SUCCESS);
  ge::MmpaStub::GetInstance().Reset();
}

TEST_F(UtestDflowApi, DFlowInitializeWithoutFinalize) {
  std::map<AscendString, AscendString> options;
  gert::GertRuntimeStub runtime_stub;
  ge::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  runtime_stub.GetSlogStub().Clear();
  EXPECT_EQ(DFlowInitialize(options), SUCCESS);
  DFlowSession session(options);
}

TEST_F(UtestDflowApi, test_build) {
  std::map<std::string, std::string> options = {{"ge.buildMode", "tuning"}, {"ge.buildStep", "after_merge"}};
  ge::DFlowSessionImpl impl(0, {});
  impl.Initialize(options);
  ge::FlowModelBuilder &builer = impl.dflow_graph_manager_.flow_model_builder_;
  EXPECT_FALSE(builer.process_node_engines_.empty());
  auto pne_iter = builer.process_node_engines_.find(ge::PNE_ID_NPU);
  ASSERT_NE(pne_iter, builer.process_node_engines_.cend());
  ge::ProcessNodeEnginePtr pne = pne_iter->second;
  ComputeGraphPtr graph = ge::MakeShared<ge::ComputeGraph>("test");
  ge::PneModelPtr model;
  
  // graph invalid
  EXPECT_NE(pne->BuildGraph(0, graph, options, {}, model), ge::SUCCESS);
  impl.Finalize();
}
}
}
