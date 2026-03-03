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
#include "ge/ge_api_error_codes.h"
#include "stub/gert_runtime_stub.h"
#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/plugin/ge_make_unique_util.h"
#include "proto/ge_ir.pb.h"
#include "ge/ge_api.h"
#include "session/session_manager.h"
#include "session/session_utils.h"
#include "framework/memory/memory_api.h"
#include "graph/utils/graph_utils_ex.h"
#include "register/ops_kernel_builder_registry.h"
#include "register/node_converter_registry.h"
#include "register/optimization_option_registry.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "graph/load/graph_loader.h"
#include "ge/ge_api_types.h"
#include "graph/operator_factory_impl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/label/label_maker.h"
#include "api/gelib/gelib.h"
#include "graph/build/stream/stream_utils.h"
#include "register/register_custom_pass.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "common/env_path.h"
#include "ge_running_env/fake_engine.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "nlohmann/json.hpp"
#include "graph/ge_local_context.h"
#include "graph/ge_global_options.h"
#include "dflow/compiler/pne/udf/udf_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "common/memory/tensor_trans_utils.h"
#include "flow_graph/flow_graph.h"
using namespace std;

extern ge::SessionManager *GetSessionManager();

namespace ge {
using Json = nlohmann::json;
namespace {
bool test_callback_called = false;
class FakeLabelMaker : public LabelMaker {
 public:
  FakeLabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : LabelMaker(graph, owner) {}

  ~FakeLabelMaker() override {}

  virtual Status Run(uint32_t &label_index) { return ge::GRAPH_SUCCESS; }
};

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

class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    return nullptr;
  }
  void Free(MemBlock *block) override {
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
REGISTER_LABEL_MAKER(PARTITIONEDCALL, FakeLabelMaker);
REGISTER_LABEL_MAKER(CASE, FakeLabelMaker);
}  // namespace

class UtestGeApi : public testing::Test {
 protected:
  void SetUp() override {
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
  }

  void CreateSharedLibrary(const std::string &path) {
    std::ofstream ofs(path + ".cpp");
    ofs << R"(
      #include <iostream>
      extern "C" void hello() {
        std::cout << "Hello, world!" << std::endl;
      }
    )";
    ofs.close();
    std::string cmd = "g++ -shared -fPIC -o " + path + ".so " + path + ".cpp";
    system(cmd.c_str());
    std::remove((path + ".cpp").c_str());
  }
};

TEST_F(UtestGeApi, run_graph_with_stream) {
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  InnerSession inner_session(1, options);
  inner_session.is_initialized_ = true;
  ret = inner_session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, delete_api) {
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.ShardGraphsToFile("./shard_graphs/");
  ASSERT_EQ(ret, FAILED);
  ret = session.SaveGraphsToPb("./shard_graphs/");
  ASSERT_EQ(ret, FAILED);
  ret = session.ShardGraphs();
  ASSERT_EQ(ret, FAILED);
}

TEST_F(UtestGeApi, build_graph_success) {
  vector<Tensor> inputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, ge_initialize_modify_mixlist) {
  std::map<std::string, std::string> options = {
    {ge::MODIFY_MIXLIST, "/mixlist.json"}
  };
  Json option_name_map;
  option_name_map.emplace("ge.enableSmallChannel", "enable_small_channel");
  options.insert({ge::OPTION_NAME_MAP, option_name_map.dump()});
  auto ret = GEInitialize(options);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, InitError_EmptyString) {
  std::map<std::string, std::string> options = {
    {"", "/mixlist.json"}
  };
  auto ret = GEInitialize(options);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, ge_initialize_fail) {
  std::map<std::string, std::string> options = {
    {"ge.optionInvalid", "Invalid"}
  };

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);
  auto ret = GEInitialize(options);
  ASSERT_EQ(ret, SUCCESS);
  auto find_log = runtime_stub.GetSlogStub().FindWarnLogEndsWith("unsupported option(ge.optionInvalid) by global level, Please check!");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
  GEFinalize();
}

TEST_F(UtestGeApi, execute_graph_with_stream) {
  vector<gert::Tensor> inputs;
  vector<gert::Tensor> outputs;
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.graphExecTimeout", "600000"));
  Session session(options);
  auto ret = session.ExecuteGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  InnerSession inner_session(1, options);
  inner_session.is_initialized_ = true;
  ret = inner_session.ExecuteGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, ge_not_initialized) {
  EXPECT_EQ(GEFinalize(), SUCCESS);
  vector<gert::Tensor> gert_inputs;
  vector<gert::Tensor> gert_outputs;
  std::map<std::string, std::string> options;
  std::map<AscendString, AscendString> ascend_options;
  Session session(options);
  auto ret = session.ExecuteGraphWithStreamAsync(10, nullptr, gert_inputs,gert_outputs);
  ASSERT_NE(ret, SUCCESS);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), FAILED);

  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph, ascend_options), FAILED);

  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(session.BuildGraph(graph_id, tensors), FAILED);

  vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), FAILED);

  vector<string> var_inputs;
  EXPECT_EQ(session.GetVariables(var_inputs, outputs), FAILED);

  vector<AscendString> var_names;
  EXPECT_EQ(session.GetVariables(var_names, outputs), FAILED);

  std::string key;
  pCallBackFunc ge_callback = nullptr;
  EXPECT_EQ(session.RegisterCallBackFunc(key, ge_callback), FAILED);

  session::pCallBackFunc session_callback = nullptr;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), session_callback), FAILED);

  EXPECT_FALSE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), FAILED);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, RunGraphAsync) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  std::vector<ge::Tensor> inputs;

  // invalid graph id
  // RunGraphAsync submit failed
  test_callback_called = false;
  auto callback = [](Status status, std::vector<ge::Tensor> &outputs) {
    EXPECT_NE(status, SUCCESS);
    test_callback_called = true;
  };

  // get graph_node fail
  EXPECT_NE(session_ptr->RunGraphAsync(10, inputs, callback), SUCCESS);
  sleep(1);  // wait callback

  // after RunGraphAsync run failed before, RunGraphAsync submit success
  EXPECT_EQ(session_ptr->RunGraphAsync(graph_id, inputs, callback), SUCCESS);
  sleep(1);  // wait callback
  EXPECT_EQ(test_callback_called, true);
  delete session_ptr;
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, AddGraph_for_max_load_option) {
  std::map<string, string> options;
  options.emplace("ge.graphMaxParallelModelNum", "10");
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);
  delete session_ptr;
  EXPECT_EQ(GEFinalize(), SUCCESS);
  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestGeApi, AddGraph_for_max_load_option2) {
  std::map<string, string> options;
  options.emplace("ge.graphMaxParallelModelNum", "-1");
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  const auto session_ptr = new Session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session_ptr->AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);
  delete session_ptr;
  EXPECT_EQ(GEFinalize(), SUCCESS);
  options.clear();
  ge::GetThreadLocalContext().SetGraphOption(options);
}

TEST_F(UtestGeApi, ge_session_ascend_string) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  Session session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session.AddGraph(graph_id, GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);

  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, ge_session_test) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.graphExecTimeout", "600000"));
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  Session session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);

  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph, ascend_options), FAILED);

  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(session.BuildGraph(graph_id, tensors), FAILED);

  vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), SUCCESS); // Push to queue.

  vector<string> var_inputs;
  EXPECT_EQ(session.GetVariables(var_inputs, outputs), FAILED);

  vector<AscendString> var_names;
  EXPECT_EQ(session.GetVariables(var_names, outputs), FAILED);

  vector<AscendString> var_names2;
  var_names2.push_back(AscendString(nullptr));
  EXPECT_EQ(session.GetVariables(var_names2, outputs), FAILED);

  vector<AscendString> var_names1;
  var_names1.push_back(AscendString(ge::ir_option::OUT_NODES));
  EXPECT_EQ(session.GetVariables(var_names1, outputs), FAILED);

  VarManager::Instance(session.GetSessionId())->var_resource_ = MakeShared<VarResource>(session.GetSessionId());
  GeTensorDesc tensor_desc(GeShape({1}), ge::FORMAT_NCHW, ge::DT_UINT64);
  VarManager::Instance(session.GetSessionId())->var_resource_->cur_var_tensor_desc_map_["var"] = tensor_desc;
  EXPECT_EQ(session.GetVariables(var_names1, outputs), FAILED);

  std::string key;
  pCallBackFunc ge_callback = nullptr;
  EXPECT_EQ(session.RegisterCallBackFunc(key, ge_callback), SUCCESS);

  session::pCallBackFunc session_callback = nullptr;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), session_callback), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}


TEST_F(UtestGeApi, ge_session_test1) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.graphExecTimeout", "-1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_GRAPH_RUN_MODE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_DEVICE_ID, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_JOB_ID, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_IS_USEHCOM, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_IS_USEHVD, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_DEPLOY_MODE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_POD_NAME, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_RANK_ID, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_SESSION_ID, "1"));
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  Session session(options);
  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);
  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;

  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(session.BuildGraph(graph_id, tensors), FAILED);

  vector<Tensor> outputs;
  ge::Tensor tensor;

  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor.SetData(data);
  inputs.emplace_back(tensor);

  ge::Tensor tensor1;
  TensorDesc tensor_desc2(Shape({1, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor1.SetData(data2);
  outputs.emplace_back(tensor1);
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, ge_session_test_fail) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  options.insert(pair<std::string, std::string>("ge.optionInvalid", "invalid"));
  Session session1(options);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);
  Session session2(ascend_options);
  auto find_log = runtime_stub.GetSlogStub().FindWarnLogEndsWith("unsupported option(ge.optionInvalid) by session level, Please check!");
  EXPECT_TRUE(find_log >= -1);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(UtestGeApi, AddGraph_test) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>("ge.exec.opWaitTimeout", "1"));
  options.insert(pair<std::string, std::string>("ge.exec.opExecuteTimeout", "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_GRAPH_RUN_MODE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_DEVICE_ID, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_JOB_ID, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_IS_USEHCOM, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_IS_USEHVD, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_DEPLOY_MODE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_POD_NAME, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_RANK_ID, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_SESSION_ID, "1"));
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  uint32_t graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> option;
  Session session(options);
  Status ret = session.AddGraph(graph_id, graph, option);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, AddGraph_test_fail) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  uint32_t graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> option;
  Session session(options);
  options.insert(pair<std::string, std::string>("ge.optionInvalid", "invalid"));
  (void)session.AddGraph(graph_id, graph, option);
  std::map<AscendString, AscendString> ascend_options = {
    {AscendString("ge.optionInvalid"), AscendString("invalid")}};

  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  dlog_setlevel(GE_MODULE_NAME, 2, 0);
  (void)session.AddGraph(graph_id, graph, ascend_options);
  (void)session.AddGraphWithCopy(graph_id, graph, ascend_options);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  auto find_log = runtime_stub.GetSlogStub().FindWarnLogEndsWith("unsupported option(ge.optionInvalid) by graph level, Please check!");
  EXPECT_TRUE(find_log > -1);
  dlog_setlevel(GE_MODULE_NAME, 3, 0);
}

TEST_F(UtestGeApi, CheckOptionsValid_Invalid_JobId_test) {
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>(OPTION_EXEC_JOB_ID, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
  Status ret = ge::GEInitialize(options);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, CheckOptionsInvalid_test) {
  std::map<AscendString, AscendString> options = {
    {AscendString(""), AscendString("Placeholder:0;Placeholder_1:1")}};
  Status ret = ge::GEInitialize(options);
  EXPECT_EQ(ret, FAILED);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, GEInitialize_test) {
  std::map<AscendString, AscendString> options = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  Status ret = ge::GEInitialize(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);

  std::map<AscendString, AscendString> options1 = {
    {AscendString(ge::ir_option::OUT_NODES), AscendString(nullptr)}};
  ret = ge::GEInitialize(options1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);

  std::map<AscendString, AscendString> options2 = {
    {AscendString("ge.autoTuneMode"), AscendString("RA")}};
  ret = ge::GEInitialize(options2);
  EXPECT_NE(ret, SUCCESS);

  ge::GEGetErrorMsg();
  ge::GEGetWarningMsg();
  ge::GEGetErrorMsgV2();
  ge::GEGetWarningMsgV2();
  Session session(options);
  Session session1(options1);
  Session session2(options2);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, GEInitialize_load_custom_pass_failed) {
  std::map<AscendString, AscendString> options;
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv("ASCEND_OPP_PATH", path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes";
  system(("mkdir -p " + custom_path).c_str());
  system(("touch " + custom_path + "/concat_pass.so").c_str());
  Status ret = ge::GEInitialize(options);
  EXPECT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  system(("rm -rf " + path).c_str());
}

TEST_F(UtestGeApi, GEInitialize_load_custom_pass_success) {
  std::map<AscendString, AscendString> options;
  std::string path = __FILE__;
  path = path.substr(0, path.rfind("/") + 1) + "opp";
  mmSetEnv("ASCEND_OPP_PATH", path.c_str(), 1);
  system(("mkdir -p " + path).c_str());

  std::string custom_path = path + "/vendors/1/custom_fusion_passes/add";
  system(("mkdir -p " + custom_path).c_str());

  CreateSharedLibrary(custom_path);
  Status ret = ge::GEInitialize(options);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  system(("rm -rf " + path).c_str());
}

TEST_F(UtestGeApi, ge_session_info_test) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  uint64_t session_id = 0;
  {
    Session session(options);
    session_id = session.sessionId_;
    EXPECT_EQ(session_id, session.GetSessionId());
  }
  EXPECT_EQ(GEFinalize(), SUCCESS);

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  vector<Tensor> inputs;
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, Feed_test_not_init) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<Tensor> inputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FeedDataFlowGraph(1, inputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, Feed_test) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  vector<Tensor> inputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FeedDataFlowGraph(1, inputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, Fetch_test_not_init) {
  std::map<std::string, std::string> options;
  Session session(options);
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FetchDataFlowGraph(1, outputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, Fetch_test) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  vector<Tensor> outputs;
  DataFlowInfo data_flow_info;
  auto ret = session.FetchDataFlowGraph(1, outputs, data_flow_info, 0);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, Fetch_flow_msg_graph_not_exist) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  vector<FlowMsgPtr> outputs;
  auto ret = session.FetchDataFlowGraph(1, outputs, 0);
  ASSERT_NE(ret, SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, InitializeExecutionRuntime_test) {
  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ExecutionRuntime::instance_ = nullptr;
  ExecutionRuntime::handle_ = (void *)0xffffffff;
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  ExecutionRuntime::handle_ = nullptr;
  EXPECT_EQ(GEFinalize(), SUCCESS);
  ExecutionRuntime::instance_ = nullptr;
  MmpaStub::GetInstance().Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(UtestGeApi, GetCompileGraphSummary_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    Session session(options);
    EXPECT_EQ(session.GetCompiledGraphSummary(graph_id), nullptr); // not init
  }

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_EQ(session.GetCompiledGraphSummary(graph_id), nullptr); // not add graph
}

TEST_F(UtestGeApi, SetGraphConstMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    Session session(options);
    EXPECT_NE(session.SetGraphConstMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.SetGraphConstMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApi, UpdateGraphFeatureMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    Session session(options);
    EXPECT_NE(session.UpdateGraphFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.UpdateGraphFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApi, SetGraphFixedFeatureMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  {
    Session session(options);
    EXPECT_NE(session.SetGraphFixedFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.SetGraphFixedFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApi, SetGraphFixedFeatureMemoryBaseWithType_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApi, UpdateGraphRefreshableFeatureMemoryBase_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    Session session(options);
    EXPECT_NE(session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.UpdateGraphRefreshableFeatureMemoryBase(graph_id, nullptr, 0), SUCCESS); // not add graph
}

TEST_F(UtestGeApi, CompileGraph_test) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;

  {
    Session session(options);
    EXPECT_NE(session.CompileGraph(graph_id), SUCCESS); // not init
  }

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS); // not add graph

  Graph graph = gert::ShareGraph::BuildSwitchMergeGraph();
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, CompileGraph_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({-1, -1, -1, -1}, {-1, -1, -1, -1});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[4, 1, 3, 2];1:[4, 2, 1, 2]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 8);
  ShapeEnvGuarder guard(shape_env_attr);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[0].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[1].first, Symbol(1)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[2].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[3].first, Symbol(4)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[4].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[5].first, Symbol(3)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[6].first, Symbol(1)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[7].first, Symbol(4)), true);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(2).Serialize().get()), "s2");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(3).Serialize().get()), "s3");

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s6");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s7");
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApi, CompileGraph_unknown_rank_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({-2}, {-1, -1, -1, -1});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[3, 2];1:[4, 2, 3, 2]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 7);
  ShapeEnvGuarder guard(shape_env_attr);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[0].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[1].first, Symbol(3)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[2].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[3].first, Symbol(4)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[4].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[5].first, Symbol(3)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[6].first, Symbol(2)), true);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "s1");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "s2");

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 4);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s3");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s4");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(2).Serialize().get()), "s5");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(3).Serialize().get()), "s6");

  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApi, CompileGraph_scalar_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({}, {-1, -1});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[];1:[4, 2]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 2);
  ShapeEnvGuarder guard(shape_env_attr);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[0].first, Symbol(2)), true);
  EXPECT_EQ(EXPECT_SYMBOL_EQ(symbol_infos[1].first, Symbol(4)), true);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 0);

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "s0");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "s1");
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApi, CompileGraph_static_with_hint_option_test) {
  mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_opp_path_env[MMPA_MAX_PATH] = {'\0'};
  char old_ld_path_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("ASCEND_OPP_PATH", old_opp_path_env, MMPA_MAX_PATH);
  (void)mmGetEnv("LD_LIBRARY_PATH", old_ld_path_env, MMPA_MAX_PATH);
  setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  Graph graph = gert::ShareGraph::OnlyDataGraph({2, 3}, {2, 3});
  options.emplace(std::make_pair("ge.inputHintShape", "0:[2, 3];1:[2, 3]"));
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);

  auto cg = GraphUtilsEx::GetComputeGraph(graph);
  auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  const auto symbol_infos = shape_env_attr->GetAllSym2Src();
  EXPECT_EQ(symbol_infos.size(), 0);

  const auto data0_node = cg->FindNode("data0");
  ASSERT_NE(data0_node, nullptr);
  const auto data0_op_desc = data0_node->GetOpDesc();
  ASSERT_NE(data0_op_desc, nullptr);
  auto data_symbol_attr0 = data0_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr0, nullptr);
  auto symbol_shape0 = data_symbol_attr0->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape0.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape0.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape0.GetDim(1).Serialize().get()), "3");

  const auto data1_node = cg->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  const auto data1_op_desc = data1_node->GetOpDesc();
  ASSERT_NE(data1_op_desc, nullptr);
  auto data_symbol_attr1 = data1_op_desc->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(data_symbol_attr1, nullptr);
  auto symbol_shape1 = data_symbol_attr1->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(symbol_shape1.GetDimNum(), 2);
  EXPECT_EQ(std::string(symbol_shape1.GetDim(0).Serialize().get()), "2");
  EXPECT_EQ(std::string(symbol_shape1.GetDim(1).Serialize().get()), "3");
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("AUTOFUSE_FLAGS");
  mmSetEnv("ASCEND_OPP_PATH", old_opp_path_env, 1);
  mmSetEnv("LD_LIBRARY_PATH", old_ld_path_env, 1);
}

TEST_F(UtestGeApi, RegisterExternalAllocator_test) {
  std::map<std::string, std::string> options;
  uint32_t stream = 1;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();

  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_EQ(session.RegisterExternalAllocator(&stream, external_allocator), SUCCESS);
  EXPECT_EQ(session.UnregisterExternalAllocator(&stream), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, test_GeSession_Api) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::map<std::string, std::string> options_init;
  Session session(options_init);
  EXPECT_EQ(GeSessionLoadGraph(session, graph_id, options, nullptr), FAILED);

  vector<gert::Tensor> gert_inputs;
  vector<gert::Tensor> gert_outputs;
  EXPECT_NE(GeSessionExecuteGraphWithStreamAsync(session, graph_id, nullptr,
    gert_inputs, gert_inputs), SUCCESS);
}


TEST_F(UtestGeApi, LoadGraph_with_graph_id) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::map<std::string, std::string> options_init;
  Session session(options_init);
  EXPECT_EQ(session.LoadGraph(graph_id, options, nullptr), FAILED);
  EXPECT_EQ(GEInitialize(options_init), SUCCESS);

  Session session1(options_init);

  options.insert(pair<AscendString, AscendString>("ge.exec.frozenInputIndexes", "1,2"));
  EXPECT_NE(session1.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}


TEST_F(UtestGeApi, Test_LoadGraphApi) {
  uint32_t graph_id = 1;
  std::map<AscendString, AscendString> options;
  std::map<std::string, std::string> options_init;
  Session session(options_init);
  EXPECT_EQ(session.LoadGraph(graph_id, options, nullptr), FAILED);
  EXPECT_EQ(GEInitialize(options_init), SUCCESS);

  Session session1(options_init);

  options.insert(pair<AscendString, AscendString>("ge.exec.frozenInputIndexes", "1,2"));
  EXPECT_NE(session1.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, run_graph_with_stream_async) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data1({1, 2, 3, 4});
  tensor2.SetData(data1);
  inputs.emplace_back(tensor2);

  ge::Tensor tensor3;
  TensorDesc tensor_desc3(Shape({1, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor3.SetTensorDesc(tensor_desc3);
  std::vector<uint8_t> data3({1, 2, 3, 4});
  tensor3.SetData(data3);
  outputs.emplace_back(tensor3);

  std::map<std::string, std::string> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);
  auto compiled_summary = session.GetCompiledGraphSummary(graph_id);
  ASSERT_NE(compiled_summary, nullptr);
  std::string expect_stream_info_0 =
      "logic_stream_id: 0, user_stream_label: , is_assigned_by_user_stream_pass: false, attached_stream_ids: "
      ", physical_model_stream_num: 1, hccl_followed_stream_num: 0.\n";
  std::shared_ptr<StreamAllocationSummary> stream_summary;
  EXPECT_EQ(compiled_summary->GetStreamAllocationSummary(stream_summary), SUCCESS);
  auto graph_to_stream_infos = stream_summary->GetAllLogicalStreamInfos();
  ASSERT_EQ(graph_to_stream_infos.size(), 1U);
  AscendString graph_name;
  ASSERT_EQ(graph.GetName(graph_name), SUCCESS);
  auto iter = graph_to_stream_infos.find(graph_name);
  ASSERT_TRUE(iter != graph_to_stream_infos.end());
  ASSERT_EQ(iter->second.size(), 1U);
  const auto &logical_stream_0_info = iter->second[0];
  EXPECT_EQ(logical_stream_0_info.ToStringInfo().GetString(), expect_stream_info_0);
  EXPECT_EQ(logical_stream_0_info.GetLogicalStreamId(), 0);
  EXPECT_EQ(logical_stream_0_info.GetAttachedStreamIds().size(), 0U);
  EXPECT_EQ(logical_stream_0_info.GetPhysicalStreamNum(), 1U);
  EXPECT_EQ(logical_stream_0_info.GetHcclFollowedStreamNum(), 0U);
  EXPECT_EQ(logical_stream_0_info.IsAssignedByStreamPass(), false);
  EXPECT_EQ(logical_stream_0_info.GetLogicalStreamId(), 0U);

  EXPECT_EQ(session.SetGraphFixedFeatureMemoryBase(graph_id, (void *)0x3558, 4000000), SUCCESS);

  auto ret = session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, run_graph_with_stream_with_dynamic) {
  ge::OperatorFactoryImpl::RegisterInferShapeFunc("Data", [](Operator &op) {return GRAPH_SUCCESS;});
  OperatorFactoryImpl::RegisterInferShapeFunc("Add", [](Operator &op) {return GRAPH_SUCCESS;});
  OperatorFactoryImpl::RegisterInferShapeFunc("NetOutput", [](Operator &op) {return GRAPH_SUCCESS;});

  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);

  vector<Tensor> inputs;
  vector<Tensor> outputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({1, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  outputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  options[ge::OO_LEVEL] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);
  ASSERT_NE(session.GetCompiledGraphSummary(graph_id), nullptr);
  // dynamic shape graph
  EXPECT_EQ(session.GetCompiledGraphSummary(graph_id)->IsStatic(), false);

  rtStream_t stream = (void*)0x01;
  auto ret = session.RunGraphWithStreamAsync(1, stream, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_EQ(GEFinalize(), SUCCESS);
}

ge::graphStatus StubInferShape(ge::Operator &op) {
  auto x_input_desc = op.GetInputDesc(0);
  auto x_shape = x_input_desc.GetShape().GetDims();
  auto x_type = x_input_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  (void)x_input_desc.GetShapeRange(x_shape_range);
  TensorDesc op_output_desc = op.GetOutputDesc(0);
  op_output_desc.SetShape(ge::Shape(x_shape));
  op_output_desc.SetOriginShape(ge::Shape(x_shape));
  op_output_desc.SetDataType(x_type);
  if (!x_shape_range.empty()) {
    op_output_desc.SetShapeRange(x_shape_range);
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  return op_desc->UpdateOutputDesc(0, TensorAdapter::TensorDesc2GeTensorDesc(op_output_desc));
}

ge::graphStatus GetShapeInferShape(ge::Operator &op) {
  std::cout << "Enter infershape getshape" << std::endl;
  std::vector<std::string> tiling_inline_engine;
  tiling_inline_engine.push_back("AIcoreEngine");
  vector<std::string> export_shape_engine;
  export_shape_engine.push_back("AIcoreEngine");
  op.SetAttr("_op_tiling_inline_engine", tiling_inline_engine);
  op.SetAttr("_op_export_shape_engine", export_shape_engine);
  return ge::GRAPH_SUCCESS;
}

TEST_F(UtestGeApi, run_graph_with_stream_with_multi_batch) {
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  outputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  options["ge.inputShape"] = "data1:-1,-1,-1;data2:-1,-1,-1";
  options["ge.dynamicDims"] = "1,1,1,1,1,1;3,3,3,3,3,3;5,5,5,5,5,5";
  options["ge.dynamicNodeType"] = "1";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);

  GraphId graph_id = 4;
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Data");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("Add");
  OperatorFactoryImpl::operator_infershape_funcs_->erase("NetOutput");
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  //  SchedulerConf conf;
  SchedulerConf scheduler_conf;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->name = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->id = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->skip_assign_stream = true;

  scheduler_conf.cal_engines["AIcoreEngine"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["AIcoreEngine"]->name = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->id = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->independent = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->attach = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_AICPU"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->name = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->id = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->name = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->id = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->skip_assign_stream = true;

  instance_ptr->DNNEngineManagerObj().schedulers_["multi_batch"] = scheduler_conf;

  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp("GetShape").InferShape(GetShapeInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"));

  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.CompileGraph(graph_id), SUCCESS);
  auto summary = session.GetCompiledGraphSummary(graph_id);
  ASSERT_NE(summary, nullptr);
  // dynamic shape graph
  EXPECT_EQ(summary->IsStatic(), true);
  std::vector<ge::Shape> output_shape;
  EXPECT_EQ(summary->GetOutputShapes(output_shape), ge::SUCCESS);
  std::vector<int64_t> expect_dims{5, 5, 5};
  ASSERT_EQ(output_shape.size(), 1);
  EXPECT_EQ(output_shape[0].GetDims(), expect_dims);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  RuntimeStub::Reset();
  ge_env.Reset();
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

TEST_F(UtestGeApi, feed_graph_without_run) {
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
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  auto graph = BuildFlowGraph();

  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph.ToGeGraph(), options), SUCCESS);
  DataFlowInfo data_flow_info;
  EXPECT_EQ(session.FeedDataFlowGraph(graph_id, inputs, data_flow_info, 0), SUCCESS);
  std::vector<Tensor> outputs;
  DataFlowInfo info;
  EXPECT_NE(session.FetchDataFlowGraph(graph_id, {0}, outputs, info, 0), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestGeApi, Run_graph) {
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
  EXPECT_EQ(GEInitialize(options), SUCCESS);
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

  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestGeApi, feed_graph_by_rawdata) {
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
  Session session1(options);
  GraphId graph_id = 1;
  DataFlowInfo data_flow_info;

  // not initilized
  uint64_t sample_data = 100;
  RawData raw_data = {.addr = reinterpret_cast<void *>(&sample_data), .len = sizeof(uint64_t)};
  EXPECT_EQ(session1.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), FAILED);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);
  Session session2(options);

  EXPECT_EQ(session2.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session2.BuildGraph(graph_id, inputs), SUCCESS);
  // without executor
  EXPECT_NE(session2.FeedRawData(graph_id, {raw_data}, 0, data_flow_info, 0), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, feed_graph_by_flow_msg) {
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
  EXPECT_EQ(GEFinalize(), SUCCESS);
  unsetenv("RESOURCE_CONFIG_PATH");
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestGeApi, GetSessionManager_test) {
  EXPECT_NE(GetSessionManager(), nullptr);
}

TEST_F(UtestGeApi, add_graph_and_shard_graphs) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  GraphId graph_id0 = 0;
  const auto compute_graph0 = MakeShared<ComputeGraph>("test_graph1");
  Graph graph0 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph0);
  EXPECT_EQ(session.AddGraph(graph_id0, graph0), SUCCESS);
  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_NE(session.ShardGraphs(), SUCCESS);
  EXPECT_NE(session.ShardGraphsToFile("./"), SUCCESS);
}

TEST_F(UtestGeApi, add_graph_and_shard_graphs_unsupported) {
  gert::GertRuntimeStub rtstub;
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  OpsKernelBuilderPtr builder = MakeShared<GeFakeOpsKernelBuilder>();
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameAiCore, builder);
  OpsKernelBuilderRegistry::GetInstance().Register(kEngineNameGeLocal, builder);

  std::map<std::string, std::string> options;
  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  (void)ge::AttrUtils::SetBool(com_graph, ge::ATTR_SINGLE_OP_SCENE, true);
  GraphId graph_id = 1;
  EXPECT_EQ(session.AddGraph(graph_id, graph, options), SUCCESS);
  EXPECT_EQ(session.ShardGraphs(), FAILED);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, add_graph_and_save_graph_to_pb_unsupported) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options = {
      {AscendString(ge::ir_option::OUT_NODES), AscendString("Placeholder:0;Placeholder_1:1")}};
  Session session(options);

  GraphId graph_id0 = 0;
  const auto compute_graph0 = MakeShared<ComputeGraph>("test_graph1");
  Graph graph0 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph0);
  GraphId graph_id1 = 1;
  const auto compute_graph1 = MakeShared<ComputeGraph>("test_graph2");
  Graph graph1 = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph1);

  EXPECT_EQ(session.AddGraph(graph_id0, graph0), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id1, graph1), SUCCESS);

  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_NE(session.ShardGraphs(), SUCCESS);
  EXPECT_NE(session.ShardGraphsToFile("/tmp/test_graph/"), SUCCESS);
  EXPECT_EQ(session.SaveGraphsToPb("/tmp/test_graph/"), FAILED);
  system("rm -rf /tmp/test_graph/");
}

TEST_F(UtestGeApi, profiling_option_fail) {
  EXPECT_EQ(GEFinalize(), SUCCESS);
  std::map<std::string, std::string> options;
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_MODE, "1"));
  options.insert(pair<std::string, std::string>(ge::OPTION_EXEC_PROFILING_OPTIONS, "1"));
  EXPECT_NE(GEInitialize(options), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, CheckOptionsValid_featureBaseRefreshable) {
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> options;
  options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "2");
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "1");
  EXPECT_NE(GEInitialize(options), SUCCESS);
}
TEST_F(UtestGeApi, NumSessions) {
  EXPECT_EQ(GEFinalize(), SUCCESS);
  EXPECT_EQ(SessionUtils::NumSessions(), 0U);
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  EXPECT_EQ(SessionUtils::NumSessions(), 0U);
  {
    Session session_1(options);
    EXPECT_EQ(SessionUtils::NumSessions(), 1U);
    Session session_2(options);
    EXPECT_EQ(SessionUtils::NumSessions(), 2U);
  }
  EXPECT_EQ(SessionUtils::NumSessions(), 0U);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, Construct_session) {
  GEFinalize();
  std::map<std::string, std::string> options;
  Session sess1(options);  // ge not initialized

  std::map<AscendString, AscendString> ascend_options;
  Session sess2(ascend_options);  // ge not initialized

  EXPECT_EQ(GEInitialize(options), SUCCESS);

  ascend_options[AscendString()] = "";  // option key is empty
  Session sess3(ascend_options);

  options["ge.exec.precision_mode"] = "invalid";  // invalid option value
  Session sess4(options);

  std::map<AscendString, AscendString> ascend_options1;
  ascend_options1[AscendString("ge.exec.precision_mode")] = "invalid";  // invalid option value
  Session sess5(ascend_options1);

  // add graph test
  std::map<std::string, std::string> options1;
  Session sess6(options1);  // contruct session successfully
  Graph g("hello");
  std::map<AscendString, AscendString> graph_options;
  graph_options[AscendString()] = "";  // graph option key is empty
  EXPECT_EQ(sess6.AddGraph(1, g, graph_options), FAILED);
}

TEST_F(UtestGeApi, PaRemapped_test) {
  uint32_t graph_id = 1;
  std::map<std::string, std::string> options;

  {
    Session session(options);
    EXPECT_NE(session.PaRemapped(0UL, 0UL, 64UL), SUCCESS); // not init
  }

  options[ge::OPTION_GRAPH_RUN_MODE] = "0";
  ge::GetThreadLocalContext().SetGraphOption(options);
  EXPECT_EQ(GEInitialize(options), SUCCESS);
  Session session(options);
  EXPECT_NE(session.PaRemapped(0UL, 0UL, 64UL), SUCCESS); // not add graph

  Graph graph = gert::ShareGraph::BuildHcomGraphWithRefData();
  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_NE(session.PaRemapped(0UL, 0UL, 64UL), SUCCESS); // not compile

  EXPECT_NE(session.CompileGraph(graph_id), SUCCESS);
  EXPECT_NE(session.PaRemapped(0UL, 0UL, 64UL), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, ge_session_oo_init) {
  std::map<std::string, std::string> global_options;
  global_options.emplace(OO_LEVEL, "O3");
  global_options.emplace(OO_CONSTANT_FOLDING, "false");
  EXPECT_EQ(GEInitialize(global_options), SUCCESS);

  std::map<std::string, std::string> session_options;
  session_options.emplace(OO_LEVEL, "O3");
  session_options.emplace(OO_CONSTANT_FOLDING, "false");
  Session session(session_options);

  GraphId graph_id = 1;
  ComputeGraphPtr compute_graph = gert::ShareGraph::AicoreGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(OO_LEVEL, "O1");
  graph_options.emplace(OO_CONSTANT_FOLDING, "true");
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  vector<Tensor> inputs;
  vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  std::string opt_value;
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApi, GEInitialize_oo_init_param_invalid) {
  std::map<std::string, std::string> global_options;
  global_options.emplace(OO_LEVEL, "O4");
  global_options.emplace(OO_CONSTANT_FOLDING, "false");
  EXPECT_NE(GEInitialize(global_options), SUCCESS);

  global_options[OO_LEVEL] = "O1";
  global_options[OO_CONSTANT_FOLDING] = "False";
  EXPECT_NE(GEInitialize(global_options), SUCCESS);

  global_options[OO_LEVEL] = "O1";
  global_options[OO_CONSTANT_FOLDING] = "0";
  EXPECT_NE(GEInitialize(global_options), SUCCESS);

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApi, Session_oo_init_param_invalid) {
  std::map<std::string, std::string> global_options;
  EXPECT_EQ(GEInitialize(global_options), SUCCESS);

  std::map<std::string, std::string> session_options;
  Session session(session_options);


  GraphId graph_id = 1;
  ComputeGraphPtr compute_graph = gert::ShareGraph::AicoreGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  std::map<std::string, std::string> graph_options;
  graph_options[OO_LEVEL] = "O4";
  graph_options[OO_CONSTANT_FOLDING] = "false";
  EXPECT_NE(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  graph_options[OO_LEVEL] = "O1";
  graph_options[OO_CONSTANT_FOLDING] = "False";
  EXPECT_NE(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  graph_options[OO_LEVEL] = "O1";
  graph_options[OO_CONSTANT_FOLDING] = "0";
  EXPECT_NE(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  GEFinalize();
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApi, Session_export_compile_stat_valid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  std::map<std::string, std::string> global_options;
  std::string opt_value;
  global_options[OPTION_EXPORT_COMPILE_STAT] = "0";
  EXPECT_EQ(GEInitialize(global_options), SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "0");

  std::map<std::string, std::string> session_options;
  session_options[OPTION_EXPORT_COMPILE_STAT] = "1";
  Session session(session_options);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "1");

  GraphId graph_id = 1;
  ComputeGraphPtr compute_graph = gert::ShareGraph::AicoreGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> graph_options;
  graph_options[OPTION_EXPORT_COMPILE_STAT] = "2";
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "2");

  vector<Tensor> inputs;
  vector<Tensor> outputs;
  EXPECT_NE(session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "2");
  std::string oo_value;
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OPTION_EXPORT_COMPILE_STAT, oo_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(oo_value, "2");

  GEFinalize();
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST_F(UtestGeApi, Session_export_compile_stat_invalid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  std::map<std::string, std::string> global_options;
  std::string opt_value;
  global_options[OPTION_EXPORT_COMPILE_STAT] = "3";
  EXPECT_NE(GEInitialize(global_options), SUCCESS);
  EXPECT_NE(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

namespace {
  class AbnormalRtsStub : public RuntimeStub {
  public:
    rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) override {
      return 1; // failed
    }
  };
} // namespace
  /**
   * 若session创建失败，确保session manager没有残留的未成功创建的session
   */
TEST_F(UtestGeApi, CreateSessionFailed) {
  auto rts_stub = std::make_shared<AbnormalRtsStub>();
  RuntimeStub::Install(rts_stub.get());

  GEFinalize();
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  Session sess1(options); // rt create context failed
  EXPECT_EQ(sess1.GetSessionId(), 0);
  Graph tmp_graph;
  EXPECT_NE(sess1.AddGraph(1, tmp_graph), SUCCESS);
  EXPECT_EQ(SessionUtils::NumSessions(), 0);

  Session sess2(options);  // rt create context failed
  EXPECT_EQ(sess2.GetSessionId(), 0);
  EXPECT_NE(sess1.AddGraph(2, tmp_graph), SUCCESS);
  EXPECT_EQ(SessionUtils::NumSessions(), 0);

  RuntimeStub::UnInstall(rts_stub.get());
}

#define EXPECT_STR_EQ(x, y) EXPECT_EQ(std::string(x.GetString()), std::string(y))

REG_OP(QueryIrTestOp1)
  .INPUT(required_x1, TensorType::ALL())
  .OPTIONAL_INPUT(optional_x2, TensorType::ALL())
  .DYNAMIC_INPUT(dynamic_x3, TensorType::ALL())
  .OUTPUT(required_y1, TensorType::ALL())
  .DYNAMIC_OUTPUT(dynamic_y1, TensorType::ALL())
  .OP_END_FACTORY_REG(QueryIrTestOp1)

TEST_F(UtestGeApi, QueryIrInputOutput) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp1", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 3U);
  EXPECT_EQ(outputs.size(), 2U);
  EXPECT_EQ(inputs[0].first, "required_x1");
  EXPECT_EQ(inputs[0].second, "required");
  EXPECT_EQ(inputs[1].first, "optional_x2");
  EXPECT_EQ(inputs[1].second, "optional");
  EXPECT_EQ(inputs[2].first, "dynamic_x3");
  EXPECT_EQ(inputs[2].second, "dynamic");
  EXPECT_EQ(outputs[0].first, "required_y1");
  EXPECT_EQ(outputs[0].second, "required");
  EXPECT_EQ(outputs[1].first, "dynamic_y1");
  EXPECT_EQ(outputs[1].second, "dynamic");
  EXPECT_EQ(attrs.size(), 0U);
}

REG_OP(QueryIrTestOp2)
  .DYNAMIC_INPUT(dynamic_x3, TensorType::ALL())
  .INPUT(required_x1, TensorType::ALL())
  .OPTIONAL_INPUT(optional_x2, TensorType::ALL())
  .DYNAMIC_OUTPUT(dynamic_y1, TensorType::ALL())
  .OUTPUT(required_y1, TensorType::ALL())
  .OP_END_FACTORY_REG(QueryIrTestOp2)

TEST_F(UtestGeApi, QueryIrInputOutputKeepOrder) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp2", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 3U);
  EXPECT_EQ(outputs.size(), 2U);
  EXPECT_EQ(inputs[0].first, "dynamic_x3");
  EXPECT_EQ(inputs[0].second, "dynamic");
  EXPECT_EQ(inputs[1].first, "required_x1");
  EXPECT_EQ(inputs[1].second, "required");
  EXPECT_EQ(inputs[2].first, "optional_x2");
  EXPECT_EQ(inputs[2].second, "optional");
  EXPECT_EQ(outputs[0].first, "dynamic_y1");
  EXPECT_EQ(outputs[0].second, "dynamic");
  EXPECT_EQ(outputs[1].first, "required_y1");
  EXPECT_EQ(outputs[1].second, "required");
  EXPECT_EQ(attrs.size(), 0U);
}

REG_OP(QueryIrTestOp3)
  .REQUIRED_ATTR(attr1, Int)
  .REQUIRED_ATTR(attr2, Float)
  .REQUIRED_ATTR(attr3, String)
  .REQUIRED_ATTR(attr4, Bool)
  .REQUIRED_ATTR(attr5, Tensor)
  .REQUIRED_ATTR(attr6, Type)
  .REQUIRED_ATTR(attr7, NamedAttrs)
  .REQUIRED_ATTR(attr8, ListInt)
  .REQUIRED_ATTR(attr9, ListFloat)
  .REQUIRED_ATTR(attr10, ListString)
  .REQUIRED_ATTR(attr11, ListBool)
  .REQUIRED_ATTR(attr12, ListTensor)
  .REQUIRED_ATTR(attr13, Bytes)
  .REQUIRED_ATTR(attr14, ListListInt)
  .REQUIRED_ATTR(attr15, ListNamedAttrs)
  .OP_END_FACTORY_REG(QueryIrTestOp3)

TEST_F(UtestGeApi, QueryIrAttr) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp3", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 0U);
  EXPECT_EQ(outputs.size(), 0U);
  ASSERT_EQ(attrs.size(), 15U);
  EXPECT_STR_EQ(attrs[0].first, "attr1");
  EXPECT_STR_EQ(attrs[0].second, "VT_INT");
  EXPECT_STR_EQ(attrs[1].first, "attr2");
  EXPECT_STR_EQ(attrs[1].second, "VT_FLOAT");
  EXPECT_STR_EQ(attrs[2].first, "attr3");
  EXPECT_STR_EQ(attrs[2].second, "VT_STRING");
  EXPECT_STR_EQ(attrs[3].first, "attr4");
  EXPECT_STR_EQ(attrs[3].second, "VT_BOOL");
  EXPECT_STR_EQ(attrs[4].first, "attr5");
  EXPECT_STR_EQ(attrs[4].second, "VT_TENSOR");
  EXPECT_STR_EQ(attrs[5].first, "attr6");
  EXPECT_STR_EQ(attrs[5].second, "VT_DATA_TYPE");
  EXPECT_STR_EQ(attrs[6].first, "attr7");
  EXPECT_STR_EQ(attrs[6].second, "VT_NAMED_ATTRS");
  EXPECT_STR_EQ(attrs[7].first, "attr8");
  EXPECT_STR_EQ(attrs[7].second, "VT_LIST_INT");
  EXPECT_STR_EQ(attrs[8].first, "attr9");
  EXPECT_STR_EQ(attrs[8].second, "VT_LIST_FLOAT");
  EXPECT_STR_EQ(attrs[9].first, "attr10");
  EXPECT_STR_EQ(attrs[9].second, "VT_LIST_STRING");
  EXPECT_STR_EQ(attrs[10].first, "attr11");
  EXPECT_STR_EQ(attrs[10].second, "VT_LIST_BOOL");
  EXPECT_STR_EQ(attrs[11].first, "attr12");
  EXPECT_STR_EQ(attrs[11].second, "VT_LIST_TENSOR");
  EXPECT_STR_EQ(attrs[12].first, "attr13");
  EXPECT_STR_EQ(attrs[12].second, "VT_BYTES");
  EXPECT_STR_EQ(attrs[13].first, "attr14");
  EXPECT_STR_EQ(attrs[13].second, "VT_LIST_LIST_INT");
  EXPECT_STR_EQ(attrs[14].first, "attr15");
  EXPECT_STR_EQ(attrs[14].second, "VT_LIST_NAMED_ATTRS");
}

REG_OP(QueryIrTestOp4)
  .ATTR(attr1, Int, 3)
  .ATTR(attr2, Float, 2.0)
  .REQUIRED_ATTR(attr3, String)
  .ATTR(attr4, Bool, false)
  .REQUIRED_ATTR(attr5, Tensor)
  .ATTR(attr6, Type, DT_BF16)
  .REQUIRED_ATTR(attr7, NamedAttrs)
  .REQUIRED_ATTR(attr8, ListInt)
  .REQUIRED_ATTR(attr9, ListFloat)
  .REQUIRED_ATTR(attr10, ListString)
  .REQUIRED_ATTR(attr11, ListBool)
  .REQUIRED_ATTR(attr12, ListTensor)
  .REQUIRED_ATTR(attr13, Bytes)
  .REQUIRED_ATTR(attr14, ListListInt)
  .REQUIRED_ATTR(attr15, ListNamedAttrs)
  .OP_END_FACTORY_REG(QueryIrTestOp4)

TEST_F(UtestGeApi, QueryIrAttrKeepOrder) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_EQ(GetRegisteredIrDef("QueryIrTestOp4", inputs, outputs, attrs), SUCCESS);
  EXPECT_EQ(inputs.size(), 0U);
  EXPECT_EQ(outputs.size(), 0U);
  ASSERT_EQ(attrs.size(), 15U);
  EXPECT_STR_EQ(attrs[0].first, "attr1");
  EXPECT_STR_EQ(attrs[0].second, "VT_INT");
  EXPECT_STR_EQ(attrs[1].first, "attr2");
  EXPECT_STR_EQ(attrs[1].second, "VT_FLOAT");
  EXPECT_STR_EQ(attrs[2].first, "attr3");
  EXPECT_STR_EQ(attrs[2].second, "VT_STRING");
  EXPECT_STR_EQ(attrs[3].first, "attr4");
  EXPECT_STR_EQ(attrs[3].second, "VT_BOOL");
  EXPECT_STR_EQ(attrs[4].first, "attr5");
  EXPECT_STR_EQ(attrs[4].second, "VT_TENSOR");
  EXPECT_STR_EQ(attrs[5].first, "attr6");
  EXPECT_STR_EQ(attrs[5].second, "VT_DATA_TYPE");
  EXPECT_STR_EQ(attrs[6].first, "attr7");
  EXPECT_STR_EQ(attrs[6].second, "VT_NAMED_ATTRS");
  EXPECT_STR_EQ(attrs[7].first, "attr8");
  EXPECT_STR_EQ(attrs[7].second, "VT_LIST_INT");
  EXPECT_STR_EQ(attrs[8].first, "attr9");
  EXPECT_STR_EQ(attrs[8].second, "VT_LIST_FLOAT");
  EXPECT_STR_EQ(attrs[9].first, "attr10");
  EXPECT_STR_EQ(attrs[9].second, "VT_LIST_STRING");
  EXPECT_STR_EQ(attrs[10].first, "attr11");
  EXPECT_STR_EQ(attrs[10].second, "VT_LIST_BOOL");
  EXPECT_STR_EQ(attrs[11].first, "attr12");
  EXPECT_STR_EQ(attrs[11].second, "VT_LIST_TENSOR");
  EXPECT_STR_EQ(attrs[12].first, "attr13");
  EXPECT_STR_EQ(attrs[12].second, "VT_BYTES");
  EXPECT_STR_EQ(attrs[13].first, "attr14");
  EXPECT_STR_EQ(attrs[13].second, "VT_LIST_LIST_INT");
  EXPECT_STR_EQ(attrs[14].first, "attr15");
  EXPECT_STR_EQ(attrs[14].second, "VT_LIST_NAMED_ATTRS");
}

TEST_F(UtestGeApi, QueryUnregisteredIr) {
  using OutType = std::vector<std::pair<AscendString, AscendString>>;
  OutType inputs, outputs, attrs;
  EXPECT_NE(GetRegisteredIrDef("QueryIrTestOpNotRegistered", inputs, outputs, attrs), SUCCESS);
}
}  // namespace ge
