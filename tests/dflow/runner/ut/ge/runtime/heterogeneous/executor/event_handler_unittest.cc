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

#include "macro_utils/dt_public_scope.h"
#include "deploy/resource/heterogeneous_deploy_planner.h"
#include "deploy/deployer/master_model_deployer.h"
#include "deploy/model_send/flow_model_sender.h"
#include "executor/event_handler.h"
#include "graph/load/model_manager/model_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/build/graph_builder.h"
#include "graph/utils/graph_utils_ex.h"
#include "stub_models.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "generator/ge_generator.h"
#include "ge/ge_api_types.h"
#include "ge/ge_api.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "proto/deployer.pb.h"
#include "graph/utils/op_desc_utils.h"
#include "dflow/inc/data_flow/model/flow_model_helper.h"

using namespace std;

namespace ge {
namespace {
class RuntimeMock : public RuntimeStub {
 public:
  rtError_t rtMalloc(void **dev_ptr, uint64_t size, rtMemType_t type, uint16_t moduleId) override {
    return 10;
  }
};

void InitBatchLoadDynamicModelMessage(const std::vector<QueueAttrs> &input_queues,
                                      const std::vector<QueueAttrs> &output_queues,
                                      deployer::ExecutorRequest &request) {
  auto batch_load_model_messgae = request.mutable_batch_load_model_message();
  deployer::ExecutorRequest_LoadModelRequest model_request;
  for (auto &in_queue : input_queues) {
    auto *queue_def = model_request.mutable_model_queues_attrs()->add_input_queues_attrs();
    queue_def->set_queue_id(in_queue.queue_id);
    queue_def->set_device_type(in_queue.device_type);
    queue_def->set_device_id(in_queue.device_id);
  }
  for (auto &out_queue : output_queues) {
    auto *queue_def = model_request.mutable_model_queues_attrs()->add_output_queues_attrs();
    queue_def->set_queue_id(out_queue.queue_id);
    queue_def->set_device_type(out_queue.device_type);
    queue_def->set_device_id(out_queue.device_id);
  }
  model_request.set_root_model_id(0);
  model_request.set_model_id(0);
  model_request.set_replica_num(1);
  model_request.set_replica_idx(0);
  model_request.set_model_path("./");
  model_request.set_is_dynamic_proxy_controlled(true);
  model_request.mutable_input_align_attrs()->set_align_max_cache_num(100);
  model_request.mutable_input_align_attrs()->set_align_timeout(30000);
  model_request.mutable_input_align_attrs()->set_drop_when_not_align(false);
  auto model = batch_load_model_messgae->mutable_models();
  model->Add(std::move(model_request));
}

static int32_t AicpuLoadModel(void *param) {
  return 0;
}

static int32_t HcomDestroy() {
  return 0;
}

class MockDynamicModelExecutor : public DynamicModelExecutor {
 public:
  explicit MockDynamicModelExecutor(bool is_host) : DynamicModelExecutor(is_host) {};
 private:
  virtual Status DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph) {
    return SUCCESS;
  }
};

class MockProxyDynamicModelExecutor : public ProxyDynamicModelExecutor {
 public:
  explicit MockProxyDynamicModelExecutor() : ProxyDynamicModelExecutor() {};
 private:
  Status DoLoadModel(const ModelData &model_data, const ComputeGraphPtr &root_graph) override {
    return SUCCESS;
  }
  void Dispatcher() override {
    return;
  }
};

class MockModelHandle : public ExecutorContext::ModelHandle {
 public:
  MockModelHandle() : ModelHandle() {}
 protected:
  unique_ptr<DynamicModelExecutor> CreateDynamicModelExecutor(bool is_host) override {
    return MakeUnique<MockDynamicModelExecutor>(is_host);
  }

  unique_ptr<ProxyDynamicModelExecutor> CreateProxyDynamicModelExecutor() override {
    return MakeUnique<MockProxyDynamicModelExecutor>();
  }
};

class MockExecutorContext : public ExecutorContext {
 protected:
  ExecutorContext::ModelHandle *GetOrCreateModelHandle(uint32_t root_model_id, uint32_t model_id) override {
    auto &submodels = model_handles_[root_model_id];
    const auto &it = submodels.find(model_id);
    if (it != submodels.cend()) {
      return it->second.get();
    }
    mock_handle = MakeUnique<MockModelHandle>();
    if (!model_handles_[root_model_id].emplace(model_id, std::move(mock_handle)).second) {
      return nullptr;
    }
    ExecutorContext::ModelHandle *handle = model_handles_[root_model_id][model_id].get();
    return handle;
  }

  unique_ptr<ModelHandle> mock_handle;
};

class ModelHandleMock : public ExecutorContext::ModelHandle {
 public:
  explicit ModelHandleMock() : ModelHandle() {}

  // MOCK_METHOD2(DoLoadModel, Status(const shared_ptr<GeRootModel> &root_model, const LoadParam &params));
  MOCK_METHOD1(DoUnloadModel, Status(const uint32_t));
  MOCK_METHOD1(ParseModel, Status(const std::string &));
  MOCK_METHOD1(LoadModel, Status(const LoadParam &));
};

class ModelHandleMock2 : public ExecutorContext::ModelHandle {
 public:
  explicit ModelHandleMock2() : ModelHandle() {}

  MOCK_METHOD1(ClearModel, Status(const int32_t));
  MOCK_METHOD2(ExceptionNotify, Status(uint32_t, uint64_t));
  MOCK_METHOD2(GetModelRuntimeIdOrHandle, Status(std::vector<uint32_t> &,
    std::vector<ExecutorContext::ModelHandle *> &));
};

class MockMmpa : public MmpaStubApiGe {
 public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return EN_OK;
  }

  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libaicpu_scheduler.so" ||
        std::string(file_name) == "libhost_aicpu_scheduler.so" ||
        std::string(file_name) == "libhccl.so") {
      return (void *)0x12345678;
    }
    return dlopen(file_name, mode);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 0;
    }
    return dlclose(handle);
  }

  void *DlSym(void *handle, const char *func_name) {
    std::string name = func_name;
    if (name == "HcomDestroy") {
      return reinterpret_cast<void *>(HcomDestroy);
    }
    return reinterpret_cast<void *>(AicpuLoadModel);
  }
};

class ExecutionContextMock : public ExecutorContext {
 public:
  MOCK_CONST_METHOD1(CreateInputStream, unique_ptr<std::istream>(const string &));
  MOCK_METHOD2(GetOrCreateModelHandle, ModelHandle *(uint32_t, uint32_t));
};
}
class EventHandlerTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
  }

  void BuildModel(ModelBufferData &model_buffer_data) {
    vector<std::string> engine_list = {"AIcoreEngine"};
    auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
    auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224});
    auto data1 = OP_CFG(DATA);
    auto data2 = OP_CFG(DATA);
    DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));
      CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_2", add2));
    };

    auto graph = ToGeGraph(g1);
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    auto root_model = StubModels::BuildRootModel(compute_graph, false);
    EXPECT_EQ(FlowModelSender::SerializeModel(root_model, model_buffer_data), SUCCESS);
  }
};

TEST_F(EventHandlerTest, TestSharedVariables) {
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<ExecutionContextMock>();
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;

  auto sync_var_manager_message = request.mutable_sync_var_manager_message();
  auto var_manager_info = sync_var_manager_message->mutable_var_manager_info();
  var_manager_info->set_session_id(1);
  var_manager_info->set_device_id(0);
  var_manager_info->set_graph_mem_max_size(1024);
  var_manager_info->set_var_mem_max_size(1024);

  auto shared_content_desc = sync_var_manager_message->add_shared_content_descs();
  shared_content_desc->set_om_content("hello world");
  shared_content_desc->set_mem_type(2);
  shared_content_desc->set_total_length(5);

  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  VarManagerPool::Instance().RemoveVarManager(1);
}

TEST_F(EventHandlerTest, TestEventHandler) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ModelBufferData model_buffer_data{};
  BuildModel(model_buffer_data);

  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<ExecutionContextMock>();
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;

  uint32_t root_model_id = 0;
  uint32_t submodel_id = 1;

  auto batch_load_model_request = request.mutable_batch_load_model_message();
  auto var_manager_info = batch_load_model_request->mutable_var_manager_info();
  var_manager_info->set_device_id(0);
  var_manager_info->set_session_id(1);
  var_manager_info->set_graph_mem_max_size(1024);
  var_manager_info->set_var_mem_max_size(1024);

  auto options = batch_load_model_request->mutable_options();
  options->mutable_global_options()->insert({"TestGlobalOption", "TestGlobalOptionValue"});
  options->mutable_global_options()->insert({OP_WAIT_TIMEOUT, "10"});
  options->mutable_global_options()->insert({OP_EXECUTE_TIMEOUT, "20"});
  options->mutable_global_options()->insert({"ge.exec.float_overflow_mode", "saturation"});
  options->mutable_session_options()->insert({"TestSessionOption", "TestSessionOptionValue"});
  options->mutable_graph_options()->insert({"TestGraphOption", "TestGraphOptionValue"});

  auto load_model_request = batch_load_model_request->add_models();
  load_model_request->set_root_model_id(root_model_id);
  load_model_request->set_model_id(submodel_id);
  load_model_request->set_model_path("test_model.om");
  auto *input_queue_def = load_model_request->mutable_model_queues_attrs()->add_input_queues_attrs();
  input_queue_def->set_queue_id(0);
  input_queue_def->set_device_type(NPU);
  input_queue_def->set_device_id(0);
  auto *output_queue_def = load_model_request->mutable_model_queues_attrs()->add_output_queues_attrs();
  output_queue_def->set_queue_id(1);
  output_queue_def->set_device_type(NPU);
  output_queue_def->set_device_id(0);
  auto &mock_context = *reinterpret_cast<ExecutionContextMock *>(handler.context_.get());
  auto mock_create_input_stream = [&model_buffer_data](const std::string &path) -> std::unique_ptr<istream> {
    auto iss = MakeUnique<std::stringstream>();
    iss->write(reinterpret_cast<char *>(model_buffer_data.data.get()),
               static_cast<std::streamsize>(model_buffer_data.length));
    return std::move(iss);
  };
  EXPECT_CALL(mock_context, CreateInputStream).WillRepeatedly(testing::Invoke(mock_create_input_stream));

  auto mock_model_handle = MakeUnique<ModelHandleMock>();
  auto &ref_mock_handle = *mock_model_handle;
  auto mock_create_model_handle = [&mock_model_handle](uint32_t root_model_id, uint32_t model_id) -> ExecutorContext::ModelHandle * {
    return mock_model_handle.get();
  };
  EXPECT_CALL(mock_context, GetOrCreateModelHandle).WillRepeatedly(testing::Invoke(mock_create_model_handle));
  EXPECT_CALL(ref_mock_handle, ParseModel).WillRepeatedly(testing::Return(SUCCESS));

  EXPECT_CALL(ref_mock_handle, LoadModel).WillRepeatedly(testing::Return(SUCCESS));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  request.clear_load_model_message();

  auto unload_model_request = request.mutable_unload_model_message();
  unload_model_request->set_model_id(root_model_id);

  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);
  handler.Finalize();
  request.clear_unload_model_message();
  VarManagerPool::Instance().RemoveVarManager(1);
}

TEST_F(EventHandlerTest, TestEventHandlerClearModel) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());

  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  uint32_t rootModelId = 0U;
  uint32_t modelId = 0U;
  ModelHandleMock2 *modelHandleMockPtr = new ModelHandleMock2();
  handler.context_->model_handles_[rootModelId].emplace(modelId, modelHandleMockPtr);
  auto &modelHandle = *reinterpret_cast<ModelHandleMock2 *>(
    handler.context_->model_handles_[rootModelId][modelId].get());

  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;

  auto clear_model_request = request.mutable_clear_model_message();
  clear_model_request->set_model_id(rootModelId);
  clear_model_request->set_clear_msg_type(0);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(SUCCESS));
  auto mock_get_clear_model_handle =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    dynamic_model_handles.emplace_back(&modelHandle);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  clear_model_request->set_clear_msg_type(2);
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  auto mock_get_clear_model_handle2 =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    dynamic_model_handles.emplace_back(&modelHandle);
    dynamic_model_handles.emplace_back(&modelHandle);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle2));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  clear_model_request->set_clear_msg_type(2);
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(SUCCESS));
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle2));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle2));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  auto runtime_stub = std::make_shared<RuntimeMock>();
  RuntimeStub::SetInstance(runtime_stub);

  auto mock_get_clear_model_handle3 =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    davinci_model_runtime_ids.emplace_back(0U);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle3));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  auto mock_get_clear_model_handle4 =
    [&modelHandle](std::vector<uint32_t> &davinci_model_runtime_ids,
      std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
    davinci_model_runtime_ids.emplace_back(0U);
    dynamic_model_handles.emplace_back(&modelHandle);
    return SUCCESS;
  };
  EXPECT_CALL(modelHandle, GetModelRuntimeIdOrHandle).WillRepeatedly(
    testing::Invoke(mock_get_clear_model_handle4));
  EXPECT_CALL(modelHandle, ClearModel).WillRepeatedly(testing::Return(FAILED));
  clear_model_request->set_clear_msg_type(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  RuntimeStub::Reset();

  auto model_handle = MakeShared<ExecutorContext::ModelHandle>();
  std::vector<uint32_t> davinci_model_runtime_ids;
  std::vector<ExecutorContext::ModelHandle *> dynamic_model_handles;
  EXPECT_EQ(model_handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids,
    dynamic_model_handles), FAILED);
  
  EXPECT_NE(model_handle->DoUnloadModel(UINT32_MAX), SUCCESS);
  model_handle->dynamic_model_executor_ = model_handle->CreateProxyDynamicModelExecutor();
  EXPECT_NE(model_handle->dynamic_model_executor_.get(), nullptr);
  model_handle->is_dynamic_proxy_controlled_ = true;
  EXPECT_EQ(model_handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids,
    dynamic_model_handles), SUCCESS);
  EXPECT_EQ(dynamic_model_handles.size(), 1U);
  EXPECT_EQ(davinci_model_runtime_ids.size(), 1U);

  model_handle->dynamic_model_executor_.reset(nullptr);
  model_handle->is_dynamic_proxy_controlled_ = false;
  auto shared_model = MakeShared<DavinciModel>(0, nullptr);
  uint32_t davinci_model_id = 0U;
  model_handle->inner_model_id_ = davinci_model_id;
  ModelManager::GetInstance().InsertModel(davinci_model_id, shared_model);
  dynamic_model_handles.clear();
  davinci_model_runtime_ids.clear();
  EXPECT_EQ(model_handle->GetModelRuntimeIdOrHandle(davinci_model_runtime_ids,
    dynamic_model_handles), SUCCESS);
  EXPECT_EQ(dynamic_model_handles.size(), 0U);
  EXPECT_EQ(davinci_model_runtime_ids.size(), 1U);

  handler.Finalize();
}

TEST_F(EventHandlerTest, TestInvalidRequest) {
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), UNSUPPORTED);
}

TEST_F(EventHandlerTest, TestInitVarManager) {
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  auto batch_load_model_request = request.mutable_batch_load_model_message();
  auto var_manager_info = batch_load_model_request->mutable_var_manager_info();
  var_manager_info->set_device_id(0);
  var_manager_info->set_var_mem_max_size(1024 * 1024 * 20);
  var_manager_info->set_session_id(1);

  std::vector<char> buffer(1024 * 1024 * 20);
  auto mock_create_input_stream = [&buffer](const std::string &path) -> std::unique_ptr<istream> {
    auto iss = MakeUnique<std::stringstream>();
    iss->write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return std::move(iss);
  };

  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<ExecutionContextMock>();
  auto &mock_context = *reinterpret_cast<ExecutionContextMock *>(handler.context_.get());
  EXPECT_CALL(mock_context, CreateInputStream).WillRepeatedly(testing::Invoke(mock_create_input_stream));
  VarManagerPool::Instance().RemoveVarManager(1);
}

TEST_F(EventHandlerTest, TestInitVarManagerInfo) {
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  auto batch_load_model_request = request.mutable_batch_load_model_message();
  auto var_manager_info = batch_load_model_request->mutable_var_manager_info();
  var_manager_info->set_device_id(0);
  var_manager_info->set_session_id(1);

  std::vector<char> buffer(1024 * 1024 * 20);
  auto mock_create_input_stream = [&buffer](const std::string &path) -> std::unique_ptr<istream> {
    auto iss = MakeUnique<std::stringstream>();
    iss->write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return std::move(iss);
  };

  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<ExecutionContextMock>();
  auto &mock_context = *reinterpret_cast<ExecutionContextMock *>(handler.context_.get());
  EXPECT_CALL(mock_context, CreateInputStream).WillRepeatedly(testing::Invoke(mock_create_input_stream));
  VarManagerPool::Instance().RemoveVarManager(1);
}

TEST_F(EventHandlerTest, CreateProxyDynamicModelExecutor_Success) {
  ExecutorContext::ModelHandle handle;
  EXPECT_NE(handle.CreateProxyDynamicModelExecutor(), nullptr);
}

TEST_F(EventHandlerTest, LoadDynamicModelWithQ_Failed) {
  auto options = GetThreadLocalContext().GetAllOptions();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  GE_MAKE_GUARD(recover_cfg, [&options](){
    MmpaStub::GetInstance().Reset();
    GetThreadLocalContext().SetGraphOption(options);
  });
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto data_1 = OP_CFG(DATA).InCnt(2).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto add0 = OP_CFG(ADD).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});
    CHAIN(NODE("_arg_0", data_0)->NODE("add_0", add0)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", data_1)->NODE("add_1", add0));
  };
  auto root_graph = ToComputeGraph(graph);
  root_graph->TopologicalSorting();
  auto output_node = root_graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_0"});
  // init request
  deployer::ExecutorRequest request;
  QueueAttrs in_queue_0 = {.queue_id = 0, .device_type = CPU, .device_id = 0};
  QueueAttrs in_queue_1 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = CPU, .device_id = 0};
  InitBatchLoadDynamicModelMessage({in_queue_0, in_queue_1}, {out_queue_0}, request);
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  auto pne_model = StubModels::BuildRootModel(root_graph, false);
  handler.context_->LocalContext().AddLocalModel(0, 0, pne_model);
  ASSERT_FALSE(handler.context_.get() == nullptr);
  g_runtime_stub_mock = "rtCpuKernelLaunchWithFlag";
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, FAILED);
  g_runtime_stub_mock.clear();
}

TEST_F(EventHandlerTest, LoadDynamicModelWithQ_Success) {
  auto options = GetThreadLocalContext().GetAllOptions();
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  GE_MAKE_GUARD(recover_cfg, [&options](){
    MmpaStub::GetInstance().Reset();
    GetThreadLocalContext().SetGraphOption(options);
  });
  DEF_GRAPH(graph) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto data_1 = OP_CFG(DATA).InCnt(2).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto add0 = OP_CFG(ADD).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});
    CHAIN(NODE("_arg_0", data_0)->NODE("add_0", add0)->NODE("Node_Output", net_output));
    CHAIN(NODE("_arg_1", data_1)->NODE("add_1", add0));
  };
  auto root_graph = ToComputeGraph(graph);
  auto output_node = root_graph->FindNode("Node_Output");
  output_node->GetOpDesc()->SetSrcIndex({0});
  output_node->GetOpDesc()->SetSrcName({"add_0"});
  // init request
  deployer::ExecutorRequest request;
  QueueAttrs in_queue_0 = {.queue_id = 0, .device_type = CPU, .device_id = 0};
  QueueAttrs in_queue_1 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_0 = {.queue_id = 2, .device_type = CPU, .device_id = 0};
  InitBatchLoadDynamicModelMessage({in_queue_0, in_queue_1}, {out_queue_0}, request);
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<MockExecutorContext>();
  auto pne_model = StubModels::BuildRootModel(root_graph, false);;
  handler.context_->LocalContext().AddLocalModel(0, 0, pne_model);
  ASSERT_FALSE(handler.context_.get() == nullptr);
  auto ret = handler.BatchLoadModels(request);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(handler.context_->model_handles_.size(), 1);
  auto &handle = handler.context_->model_handles_[0U][0U];
  EXPECT_EQ(handle->dynamic_model_executor_->num_outputs_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_queues_num_, 1);
  EXPECT_EQ(handle->dynamic_model_executor_->output_events_num_, 0);
  EXPECT_EQ(handle->dynamic_model_executor_->num_inputs_, 2);
  EXPECT_EQ(handle->dynamic_model_executor_->input_queues_num_, 2);
  EXPECT_EQ(handle->dynamic_model_executor_->input_events_num_, 0);
}

TEST_F(EventHandlerTest, TestExceptionNotify) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  uint32_t root_model_id = 0U;
  uint32_t model_id = 0U;
  uint32_t invoke_model_id = 1U;
  ModelHandleMock2 *model_handle_mock_ptr = new ModelHandleMock2();
  ModelHandleMock2 *invoke_model_handle_mock_ptr = new ModelHandleMock2();
  handler.context_->model_handles_[root_model_id].emplace(model_id, model_handle_mock_ptr);
  handler.context_->model_handles_[root_model_id].emplace(invoke_model_id, invoke_model_handle_mock_ptr);
  auto &model_handle = *reinterpret_cast<ModelHandleMock2 *>(
      handler.context_->model_handles_[root_model_id][model_id].get());
  auto &invoke_model_handle = *reinterpret_cast<ModelHandleMock2 *>(
      handler.context_->model_handles_[root_model_id][invoke_model_id].get());
  invoke_model_handle.is_invoked_nn_ = true;

  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  request.set_type(deployer::kExecutorExceptionNotify);
  auto exception_notify_request = request.mutable_exception_notify_request();
  // not exist root model id.
  exception_notify_request->set_root_model_id(999);
  auto exception_notify = exception_notify_request->mutable_exception_notify();
  exception_notify->set_trans_id(100);
  exception_notify->set_type(0);
  exception_notify->set_scope("");
  exception_notify->set_user_context_id(111);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  exception_notify_request->set_root_model_id(root_model_id);
  EXPECT_CALL(model_handle, ExceptionNotify).WillRepeatedly(testing::Return(SUCCESS));
  auto mock_get_model_handle2 =
      [&model_handle](std::vector<uint32_t> &davinci_model_runtime_ids,
                     std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
        dynamic_model_handles.emplace_back(&model_handle);
        dynamic_model_handles.emplace_back(&model_handle);
        return SUCCESS;
      };
  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_model_handle2));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Return(FAILED));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_model_handle2));
  EXPECT_CALL(model_handle, ExceptionNotify).WillRepeatedly(testing::Return(FAILED));
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), FAILED);

  auto runtime_stub = std::make_shared<RuntimeMock>();
  RuntimeStub::SetInstance(runtime_stub);
  auto mock_get_clear_model_handle3 =
      [](std::vector<uint32_t> &davinci_model_runtime_ids,
                     std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
        davinci_model_runtime_ids.emplace_back(0U);
        return SUCCESS;
      };
  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_clear_model_handle3));
  handler.HandleEvent(request, response);
  EXPECT_NE(response.error_code(), SUCCESS);

  auto mock_get_clear_model_handle4 =
      [&model_handle](std::vector<uint32_t> &davinci_model_runtime_ids,
                     std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles) -> Status {
        davinci_model_runtime_ids.emplace_back(0U);
        dynamic_model_handles.emplace_back(&model_handle);
        return SUCCESS;
      };
  EXPECT_CALL(model_handle, GetModelRuntimeIdOrHandle).WillRepeatedly(
      testing::Invoke(mock_get_clear_model_handle4));
  EXPECT_CALL(model_handle, ExceptionNotify).WillRepeatedly(testing::Return(FAILED));
  handler.HandleEvent(request, response);
  EXPECT_NE(response.error_code(), SUCCESS);

  RuntimeStub::Reset();
  handler.Finalize();
}

TEST_F(EventHandlerTest, UpdateProfilingInfo) {
  EventHandler handler;
  EXPECT_EQ(handler.Initialize(), SUCCESS);
  handler.context_ = MakeUnique<ExecutionContextMock>();
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  auto update_prof_request = request.mutable_update_prof_message();
  update_prof_request->set_is_prof_start(0);
  update_prof_request->set_prof_data("test");

  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  update_prof_request->set_is_prof_start(1);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  update_prof_request->set_is_prof_start(0);
  handler.HandleEvent(request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);

  handler.Finalize();
  request.clear_update_prof_message();
  VarManagerPool::Instance().RemoveVarManager(1);
}
}  // namespace ge
