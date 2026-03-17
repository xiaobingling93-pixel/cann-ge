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
#include "nlohmann/json.hpp"
#include "mmpa/mmpa_api.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "ge/ge_api.h"
#include "flow_graph/data_flow.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "dflow/inc/data_flow/flow_graph/model_pp.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "graph/ge_global_options.h"
#include "proto/dflow.pb.h"
#include "init_ge.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/env_path.h"

using namespace std;
using namespace ge;
namespace ge {
namespace {
class CustomInnerPp : public dflow::InnerPp {
 public:
  CustomInnerPp(const char_t *pp_name, const char_t *inner_type) : dflow::InnerPp(pp_name, inner_type) {}
  ~CustomInnerPp() override = default;

 protected:
  virtual void InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const {
    serialize_map["TestAttr"] = "TestAttrValue";
  }
};
void PrepareForUdf() {
  std::string cmd = "mkdir -p model_pp_udf; cd model_pp_udf; touch libtest.so";
  (void)system(cmd.c_str());
  std::ofstream cmakefile("./model_pp_udf/CMakeLists.txt");
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

  constexpr const char *compiler_config = "./model_pp_udf/compiler_config.json";
  {
    nlohmann::json cpu_compiler_json = {
        {"compiler",
         {
             {
                 {"resource_type", "X86"},
                 {"toolchain", "/usr/bin/g++"},
             },
             {
                 {"resource_type", "Ascend"},
                 {"toolchain", "/usr/local/Ascend/hcc"},
             },
         }},
    };
    std::ofstream json_file(compiler_config);
    json_file << cpu_compiler_json << std::endl;
  }
  std::string model_pp_udf_config_file = "./model_pp_udf/model_pp_udf_config_file.json";
  {
    nlohmann::json model_pp_udf_cfg_json = {{"workspace", "./model_pp_udf"},
                                            {"target_bin", "libxxx.so"},
                                            {"input_num", 1},
                                            {"output_num", 1},
                                            {"cmakelist_path", "CMakeLists.txt"},
                                            {"compiler", compiler_config},
                                            {"func_list", {{{"func_name", "func1"}}}}};
    std::ofstream json_file(model_pp_udf_config_file);
    json_file << model_pp_udf_cfg_json << std::endl;
  }
}
Graph BuildGraphWithInvokeModelPp(const std::string &name, const std::string &model_path) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);

  auto invoked_model_pp0 = dflow::ModelPp("invoked_graph_pp0", model_path.c_str());
  // function pp
  auto udf_pp = dflow::FunctionPp("udf_pp")
                    .SetCompileConfig("./model_pp_udf/model_pp_udf_config_file.json")
                    .AddInvokedClosure("invoke_model_pp0", invoked_model_pp0);
  node0.AddPp(udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Graph BuildGraphWithInvokeModelPpWithConfig(const std::string &name, const std::string &model_path, int32_t type = 0) {
  // type 0: correct ;1 invalid path ;2 invalid json
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);
  std::string pp_config_file = "./config.json";
  {
    nlohmann::json pp1_cfg_json = {{"invoke_model_fusion_inputs", "0~7"}};
    std::ofstream json_file(pp_config_file);
    json_file << pp1_cfg_json << std::endl;
  }
  std::string pp_config_file1 = "./config1.json";
  {
    nlohmann::json pp1_cfg_json = {{"invoke_model_fusion_inputs", 10}};
    std::ofstream json_file(pp_config_file1);
    json_file << pp1_cfg_json << std::endl;
  }
  auto invoked_model_pp0 = dflow::ModelPp("invoked_graph_pp0", model_path.c_str());
  if (type == 0) {
    invoked_model_pp0.SetCompileConfig(pp_config_file.c_str());
  } else if (type == 1) {
    invoked_model_pp0.SetCompileConfig(nullptr).SetCompileConfig("invlid_file");
  } else {
    invoked_model_pp0.SetCompileConfig(pp_config_file1.c_str());
  }
  // function pp
  auto udf_pp = dflow::FunctionPp("udf_pp")
                    .SetCompileConfig("./model_pp_udf/model_pp_udf_config_file.json")
                    .AddInvokedClosure("invoke_model_pp0", invoked_model_pp0);
  node0.AddPp(udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

Graph BuildGraphWithInvokeCustomInnerPp(const std::string &name, const std::string &inner_pp_type) {
  auto data0 = dflow::FlowData("Data0", 0);
  auto node0 = dflow::FlowNode("node0", 1, 1).SetInput(0, data0);

  auto invoked_inner_pp0 = CustomInnerPp("invoked_pp0", inner_pp_type.c_str());
  // function pp
  auto udf_pp = dflow::FunctionPp("udf_pp")
                    .SetCompileConfig("./model_pp_udf/model_pp_udf_config_file.json")
                    .AddInvokedClosure("invoke_custom_pp0", invoked_inner_pp0);
  node0.AddPp(udf_pp);

  std::vector<dflow::FlowOperator> inputsOperator{data0};
  std::vector<dflow::FlowOperator> outputsOperator{node0};

  dflow::FlowGraph flow_graph(name.c_str());
  flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
  return flow_graph.ToGeGraph();
}

void *mock_handle = nullptr;
void *mock_method = nullptr;

class ExchangeServiceMock : public ExchangeService {
 public:
  Status CreateQueue(int32_t device_id, const string &name, const MemQueueAttr &mem_queue_attr,
                     uint32_t &queue_id) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, rtMbufPtr_t m_buf,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, size_t size, const FillFunc &fill_func,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status DestroyQueue(int32_t device_id, uint32_t queue_id) override {
    return 0;
  }
  Status Enqueue(int32_t device_id, uint32_t queue_id, const void *data, size_t size,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status Enqueue(const int32_t device_id, const uint32_t queue_id, const std::vector<BuffInfo> &buffs,
                 const ControlInfo &control_info) override {
    return 0;
  }
  Status EnqueueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t m_buf, int32_t timeout){
    return 0;
  }

  Status DequeueTensor(const int32_t device_id, const uint32_t queue_id, GeTensor &tensor,
                       ControlInfo &control_info) override {
    return 0;
  }

  Status DequeueMbuf(int32_t device_id, uint32_t queue_id, rtMbufPtr_t *m_buf, int32_t timeout) override {
    return 0;
  }

  void ResetQueueInfo(const int32_t device_id, const uint32_t queue_id) override {
    return;
  }

  Status DequeueMbufTensor(const int32_t device_id, const uint32_t queue_id, std::shared_ptr<AlignedPtr> &aligned_ptr,
                           const size_t size, ControlInfo &control_info) {
    return SUCCESS;
  }

  MOCK_METHOD5(Dequeue, Status(int32_t, uint32_t, void *, size_t, ExchangeService::ControlInfo &));
};

class ModelDeployerMock : public ModelDeployer {
 public:
  Status DeployModel(const FlowModelPtr &flow_model, DeployResult &deploy_result) override {
    deploy_result.input_queue_attrs = {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    deploy_result.output_queue_attrs = {{4, 0, 0}};
    return SUCCESS;
  }

  Status Undeploy(uint32_t model_id) override {
    return SUCCESS;
  }
};

class ExecutionRuntimeMock : public ExecutionRuntime {
 public:
  Status Initialize(const map<std::string, std::string> &options) override {
    return 0;
  }
  Status Finalize() override {
    return 0;
  }
  ModelDeployer &GetModelDeployer() override {
    return model_deployer_;
  }
  ExchangeService &GetExchangeService() override {
    return exchange_service_;
  }
  const std::string &GetCompileHostResourceType() const override {
    if (set_host_) {
      return host_stub_;
    }
    return host_stub2_;
  }
  const std::map<std::string, std::string> &GetCompileDeviceInfo() const override {
    if (set_dev_) {
      return logic_dev_id_to_res_type_;
    }
    return logic_dev_id_to_res_type2_;
  }

 public:
  ExchangeServiceMock exchange_service_;
  ModelDeployerMock model_deployer_;
  bool set_host_ = false;
  bool set_dev_ = false;
  std::string host_stub_ = "stub_host_type";
  std::map<std::string, std::string> logic_dev_id_to_res_type_ = {{"0:0:0", "stub_dev_type"},
                                                                  {"0:0:1", "stub_dev_type"}};
  std::string host_stub2_ = "";
  std::map<std::string, std::string> logic_dev_id_to_res_type2_ = {};
};

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<ExecutionRuntimeMock>());
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
class InnerPpTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    ExecutionRuntime::instance_ = ge::MakeShared<ExecutionRuntimeMock>();
    PrepareForUdf();
    ReInitGe();
    std::string st_dir_path = ge::PathUtils::Join({ge::EnvPath().GetAirBasePath(), "/tests/dflow/runner/st/"});
    auto real_path = st_dir_path + "st_run_data/json/helper_runtime/host/numa_config.json";
    setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
  }
  static void TearDownTestSuite() {
    ExecutionRuntime::instance_ = nullptr;
    system("rm -fr model_pp_udf");
    dflow::DFlowFinalize();
    unsetenv("RESOURCE_CONFIG_PATH");
  }
  void SetUp() {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    mock_handle = (void *)0xffffffff;
    mock_method = (void *)&InitializeHeterogeneousRuntime;
    std::map<std::string, std::string> options_runtime;
    ASSERT_EQ(ExecutionRuntime::InitHeterogeneousRuntime(options_runtime), SUCCESS);
    ge::ProcessNodeEngineRegisterar udf_engine_register __attribute__((unused)) (
        PNE_ID_UDF, []() -> ::ge::ProcessNodeEngine * { return new (std::nothrow) ge::UdfProcessNodeEngine(); });
    {
      auto &global_options_mutex = GetGlobalOptionsMutex();
      const std::lock_guard<std::mutex> lock(global_options_mutex);
      auto &global_options = GetMutableGlobalOptions();
      global_options[OPTION_NUMA_CONFIG] =
          R"({"cluster":[{"cluster_nodes":[{"is_local":true, "item_list":[{"item_id":0}], "node_id":0, "node_type":"TestNodeType1"}]}],"item_def":[{"aic_type":"[DAVINCI_V100:10]","item_type":"","memory":"[DDR:80GB]","resource_type":"Ascend"}],"node_def":[{"item_type":"","links_mode":"TCP:128Gb","node_type":"TestNodeType1","resource_type":"X86","support_links":"[ROCE]"}]})";
    }
  }
  void TearDown() {
    ExecutionRuntime::FinalizeExecutionRuntime();
    MmpaStub::GetInstance().Reset();
    mock_handle = nullptr;
    mock_method = nullptr;
  }
};

TEST_F(InnerPpTest, custom_inner_pp_serialize_success) {
  CustomInnerPp pp_stub("stub_name", "test_type");
  AscendString str;
  pp_stub.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.name(), "stub_name");
  const auto &extend_attrs = process_point.pp_extend_attrs();
  const auto &find_ret = extend_attrs.find("TestAttr");
  ASSERT_FALSE(find_ret == extend_attrs.cend());
  EXPECT_EQ(find_ret->second, "TestAttrValue");
}

TEST_F(InnerPpTest, custom_inner_pp_param_invalid) {
  CustomInnerPp pp_stub("stub_name", nullptr);
  AscendString str;
  pp_stub.Serialize(str);
  ASSERT_EQ(str.GetLength(), 0);
}

TEST_F(InnerPpTest, custom_inner_pp_unkown_type) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;

  Graph g1 = BuildGraphWithInvokeCustomInnerPp("test_graph", "unkown_type");
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(InnerPpTest, model_pp_with_ge_root_model) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;
  Graph g1 = BuildGraphWithInvokeModelPp(
      "single_model_pp_ge_root_graph",
      PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/ge/st/"}) + "st_run_data/origin_model/root_model.om");
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(InnerPpTest, model_pp_with_ge_root_model_with_config_succ) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;
  Graph g1 = BuildGraphWithInvokeModelPpWithConfig(
      "single_model_pp_ge_root_graph",
      PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/ge/st/"}) + "st_run_data/origin_model/root_model.om", 0);
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(InnerPpTest, model_pp_with_ge_root_model_with_config_file_not_exist) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;
  Graph g1 = BuildGraphWithInvokeModelPpWithConfig(
      "single_model_pp_ge_root_graph",
      PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/ge/st/"}) + "st_run_data/origin_model/root_model.om", 1);
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(InnerPpTest, model_pp_with_ge_root_model_with_config_file_format_err) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;
  Graph g1 = BuildGraphWithInvokeModelPpWithConfig(
      "single_model_pp_ge_root_graph",
      PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/ge/st/"}) + "st_run_data/origin_model/root_model.om", 2);
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(InnerPpTest, model_pp_with_flow_model) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;

  Graph g1 = BuildGraphWithInvokeModelPp(
      "single_model_pp_ge_root_graph",
      PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/ge/st/"}) + "st_run_data/origin_model/flow_model.om");
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(InnerPpTest, model_pp_with_not_exist_model) {
  map<AscendString, AscendString> options = {};
  std::vector<InputTensorInfo> inputs;

  Graph g1 = BuildGraphWithInvokeModelPp("single_model_pp_ge_root_graph", "./test_not_exist_model.om");
  Session session(options);
  session.AddGraph(1, g1, options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(InnerPpTest, model_pp_with_deploy_info) {
  constexpr const char *file_name = "./model_pp_udf/st_data_flow_deploy_info.json";
  std::ofstream json_file(file_name);
  std::string content = R"(
      {
        "batch_deploy_info": [
          {
            "flow_node_list":["node0"],
            "logic_device_list":"0:0:0:0"
          }
        ]
      })";
  json_file << content << std::endl;
  json_file.close();

  map<AscendString, AscendString> options = {};
  map<AscendString, AscendString> graph_options = {{"ge.experiment.data_flow_deploy_info_path", file_name}};
  std::vector<InputTensorInfo> inputs;
  Graph g1 = BuildGraphWithInvokeModelPp(
      "single_model_pp_ge_root_graph",
      PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/ge/st/"}) + "st_run_data/origin_model/root_model.om");
  Session session(options);
  session.AddGraph(1, g1, graph_options);
  auto ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
}
}  // namespace ge
