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
#include <vector>
#include <string>
#include <map>
#include "framework/common/debug/ge_log.h"
#include "hccl/hccl_types.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "macro_utils/dt_public_scope.h"
#include "deploy/flowrm/flow_route_manager.h"
#include "deploy/flowrm/tsd_client.h"
#include "deploy/deployer/deploy_context.h"
#include "deploy/execfwk/builtin_executor_client.h"
#include "deploy/execfwk/executor_manager.h"
#include "common/config/configurations.h"
#include "common/subprocess/subprocess_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "runtime/rt.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/ge_local_context.h"
#include "graph/utils/tensor_utils.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"

using namespace std;
using namespace ::testing;
namespace ge {
namespace {
class MockRuntime : public RuntimeStub {
 public:
  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    rtMbufPtr_t data;
    (void)rtMbufAlloc(&data, 1536);
    *mbuf = data;
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    *databuf = data_;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    *size = 4;
    return 0;
  }

 private:
  uint8_t data_[1536];
};

class MockRuntimeNoLeaks : public RuntimeStub {
 public:
  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    *databuf = data_;
    return 0;
  }
  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    *size = 4;
    return 0;
  }
  rtError_t rtMbufFree(rtMbufPtr_t mbuf) {
    // 由MockRuntimeNoLeaks统一释放
    return RT_ERROR_NONE;
  }
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) {
    // 此处打桩记录所有申请的Mbuf,此UT不会Dequeue和Free而造成泄漏,因此在MockRuntime析构时统一释放
    RuntimeStub::rtMbufAlloc(mbuf, size);
    std::lock_guard<std::mutex> lk(mu_);
    mem_bufs_.emplace_back(*mbuf);
    return 0;
  }
  ~MockRuntimeNoLeaks() {
    for (auto &mbuf : mem_bufs_) {
      RuntimeStub::rtMbufFree(mbuf);
    }
    mem_bufs_.clear();
  }

 private:
  std::mutex mu_;
  uint8_t data_[1536];
  std::vector<void *> mem_bufs_;
};

} // namespace
void *mock_handle = nullptr;

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "TsdCapabilityGet") {
      return (void *) &TsdCapabilityGet;
    } else if (std::string(func_name) == "ProcessCloseSubProcList") {
      return (void *) &ProcessCloseSubProcList;
    } else if (std::string(func_name) == "TsdProcessOpen") {
      return (void *) &TsdProcessOpen;
    } else if (std::string(func_name) == "TsdGetProcListStatus") {
      return (void *) &TsdGetProcListStatus;
    } else if (std::string(func_name) == "TsdFileLoad") {
      return (void *) &TsdFileLoad;
    } else if (std::string(func_name) == "TsdInitFlowGw") {
      return (void *) &TsdInitFlowGw;
    }
    return MmpaStubApiGe::DlSym(handle, func_name);
  }

  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return EN_OK;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    if (std::string(fileName) == "libtsdclient.so") {
      return (void *) mock_handle;
    }
    return MmpaStubApiGe::DlOpen(fileName, mode);
  }
  int32_t DlClose(void *handle) override {
    if (mock_handle == handle) {
      return 0L;
    }
    return MmpaStubApiGe::DlClose(handle);
  }
  int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }
};

class DeployContextTest : public testing::Test {
 protected:
  void SetUp() override {
    mock_handle = (void *)0xFFFFFFFF;
    RuntimeStub::SetInstance(std::make_shared<MockRuntimeNoLeaks>());
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    SubprocessManager::GetInstance().Initialize();
    TsdClient::GetInstance().Initialize();
  }
  void TearDown() override {
    SubprocessManager::GetInstance().Finalize();
    TsdClient::GetInstance().Finalize();
    HeterogeneousExchangeService::GetInstance().Finalize();
    Configurations::GetInstance().information_ = {};
    RuntimeStub::Reset();
    MmpaStub::GetInstance().Reset();
    mock_handle = nullptr;
  }
};

class MockExecutorMessageClient : public ExecutorMessageClient {
 public:
  MockExecutorMessageClient() : ExecutorMessageClient(0) {}
  Status SendRequest(const deployer::ExecutorRequest &request, deployer::ExecutorResponse &resp, int64_t timeout) override {
    if (request.has_clear_model_message()) {
      resp.set_error_code(FAILED);
    }
    return SUCCESS;
  }
};

class MockPneExecutorClient : public BuiltinExecutorClient {
 public:
  explicit MockPneExecutorClient(int32_t device_id) : BuiltinExecutorClient(device_id) {}

 protected:
  Status ForAndInit(int32_t device_id, unique_ptr<ExecutorMessageClient> &executor_process) override {
    executor_process = MakeUnique<MockExecutorMessageClient>();
    pid_ = 1;
    return SUCCESS;
  }
};

class MockDeployerVarManager : public DeployerVarManager {
 public:
  MockDeployerVarManager(const DeployerVarManager &var_manger) {
    this->var_mbuf_ = var_manger.var_mbuf_;
    this->var_mem_base_ = var_manger.var_mem_base_;
    this->var_mem_size_ = var_manger.var_mem_size_;
    this->share_var_mem_ = var_manger.share_var_mem_;
    this->var_manager_info_ = var_manger.var_manager_info_;

    this->base_path_ = var_manger.base_path_;
    this->receiving_node_name_ = var_manger.receiving_node_name_;
  }
};

TEST_F(DeployContextTest, TestProcessVarManagerOfAIServer) {
  DeployContext context;
  deployer::DeployerResponse response;
  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(0);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_id(0);
  init_process_resource_request.set_profiling_on(true);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  init_process_resource_request.set_device_id(1);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_id(1);
  std::vector<int32_t> res_ids_1 = {1};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_1.begin(), res_ids_1.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  int32_t device_id = 0;
  deployer::MultiVarManagerRequest info;
  info.add_device_ids(device_id);
  auto var_manager_info = info.mutable_multi_var_manager_info()->add_var_manager_info();
  var_manager_info->set_device_id(device_id);
  var_manager_info->set_session_id(1);
  var_manager_info->set_use_max_mem_size(128);
  var_manager_info->set_var_mem_logic_base(0);
  deployer::VarDevAddrMgr var_dev_addr_mgr;
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor_desc, 4);
  GeTensorSerializeUtils::GeTensorDescAsProto(tensor_desc, var_dev_addr_mgr.mutable_desc());
  var_dev_addr_mgr.set_address(0);
  var_dev_addr_mgr.set_dev_addr(0);
  deployer::VarResourceInfo *const var_resource_info = var_manager_info->mutable_var_resource();
  var_resource_info->mutable_var_dev_addr_mgr_map()->insert({0, var_dev_addr_mgr});
  auto ret = context.ProcessMultiVarManager(info);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_TRUE(context.var_managers_[device_id][1] != nullptr);

  auto mock_runtime = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(mock_runtime);
  auto mock_var_manger = std::make_unique<MockDeployerVarManager>(*context.var_managers_[0][1]);
  context.var_managers_[device_id][1] = std::move(mock_var_manger);

  deployer::SharedContentDescRequest shared_info;
  auto shared_content_desc = shared_info.mutable_shared_content_desc();
  shared_content_desc->set_session_id(1);
  shared_content_desc->set_node_name("node");
  shared_content_desc->set_om_content("hello");
  shared_content_desc->set_total_length(4);
  shared_content_desc->set_current_offset(0);
  shared_info.add_device_ids(device_id);
  context.tansfer_routes_.clear();

  auto mock_var_manger_device1 = std::make_unique<MockDeployerVarManager>(*context.var_managers_[0][1]);
  context.var_managers_[1][1] = std::move(mock_var_manger_device1);
  auto worker_var_req = shared_info;
  worker_var_req.add_device_ids(1);
  auto remote_plan = worker_var_req.mutable_flow_route();
  {
    auto tag_endpoint = remote_plan->add_endpoints();
    tag_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag));
    tag_endpoint->set_name("transfer_file_tag");
    auto tag_desc = tag_endpoint->mutable_tag_desc();
    tag_desc->set_name("transfer_file_tag");
    tag_desc->set_rank_id(0);
    tag_desc->set_peer_rank_id(2);

    auto queue_endpoint = remote_plan->add_endpoints();
    queue_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint->set_name("transfer_file_receive");
    queue_endpoint->set_device_id(0);
    auto queue_desc = queue_endpoint->mutable_queue_desc();
    queue_desc->set_name("transfer_file_receive");
    queue_desc->set_depth(16);
    queue_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto queue_endpoint_send = remote_plan->add_endpoints();
    queue_endpoint_send->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint_send->set_name("transfer_file_send");
    queue_endpoint_send->set_device_id(0);
    auto queue_desc_send = queue_endpoint_send->mutable_queue_desc();
    queue_desc_send->set_name("transfer_file_send");
    queue_desc_send->set_depth(16);
    queue_desc_send->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto queue_endpoint_send1 = remote_plan->add_endpoints();
    queue_endpoint_send1->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint_send1->set_name("transfer_file_send1");
    queue_endpoint_send1->set_device_id(0);
    auto queue_desc_send1 = queue_endpoint_send1->mutable_queue_desc();
    queue_desc_send1->set_name("transfer_file_send1");
    queue_desc_send1->set_depth(16);
    queue_desc_send1->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto binding = remote_plan->add_bindings();
    binding->set_src_index(0);
    binding->set_dst_index(1);
  }
  context.tansfer_routes_.clear();
  ret = context.ProcessSharedContent(worker_var_req, response);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(context.tansfer_routes_.size(), 1);
}

TEST_F(DeployContextTest, TestProcessVarManagerAndSharedInfo) {
  DeployContext context;
  deployer::DeployerResponse response;
  deployer::SharedContentDescRequest shared_info;
  auto ret = context.ProcessSharedContent(shared_info, response);
  EXPECT_EQ(ret, PARAM_INVALID);

  const int32_t device_id = 0;
  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(device_id);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  auto shared_content_desc = shared_info.mutable_shared_content_desc();
  shared_content_desc->set_session_id(1);
  shared_content_desc->set_node_name("node");
  shared_content_desc->set_om_content("hello");
  shared_content_desc->set_total_length(4);
  shared_content_desc->set_current_offset(0);
  shared_info.add_device_ids(device_id);
  auto copy_shared_info = shared_info;
  context.tansfer_routes_.clear();
  ret = context.ProcessSharedContent(copy_shared_info, response);
  EXPECT_NE(ret, SUCCESS);  // var manager not initialized
  response.set_error_code(SUCCESS);

  deployer::MultiVarManagerRequest info;
  info.add_device_ids(device_id);
  auto var_manager_info = info.mutable_multi_var_manager_info()->add_var_manager_info();
  var_manager_info->set_device_id(device_id);
  var_manager_info->set_session_id(1);
  var_manager_info->set_use_max_mem_size(128);
  var_manager_info->set_var_mem_logic_base(0);
  deployer::VarDevAddrMgr var_dev_addr_mgr;
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor_desc, 4);
  GeTensorSerializeUtils::GeTensorDescAsProto(tensor_desc, var_dev_addr_mgr.mutable_desc());
  var_dev_addr_mgr.set_address(0);
  var_dev_addr_mgr.set_dev_addr(0);
  deployer::VarResourceInfo *const var_resource_info = var_manager_info->mutable_var_resource();
  var_resource_info->mutable_var_dev_addr_mgr_map()->insert({0, var_dev_addr_mgr});
  ret = context.ProcessMultiVarManager(info);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_TRUE(context.var_managers_[device_id][1] != nullptr);

  auto remote_plan = shared_info.mutable_flow_route();
  {
    auto tag_endpoint = remote_plan->add_endpoints();
    tag_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag));
    tag_endpoint->set_name("transfer_file_tag");
    auto tag_desc = tag_endpoint->mutable_tag_desc();
    tag_desc->set_name("transfer_file_tag");
    tag_desc->set_rank_id(0);
    tag_desc->set_peer_rank_id(2);

    auto queue_endpoint = remote_plan->add_endpoints();
    queue_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint->set_name("transfer_file");
    auto queue_desc = queue_endpoint->mutable_queue_desc();
    queue_desc->set_name("transfer_file");
    queue_desc->set_depth(16);
    queue_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto binding = remote_plan->add_bindings();
    binding->set_src_index(0);
    binding->set_dst_index(1);
  }

  auto mock_var_manger = std::make_unique<MockDeployerVarManager>(*context.var_managers_[device_id][1]);
  context.var_managers_[device_id][1] = std::move(mock_var_manger);

  copy_shared_info = shared_info;
  context.tansfer_routes_.clear();
  ret = context.ProcessSharedContent(copy_shared_info, response);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(context.tansfer_routes_.size(), 1);

  // test reuse transfer routes
  ret = context.ProcessSharedContent(copy_shared_info, response);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(context.tansfer_routes_.size(), 1);
  deployer::VarManagerInfo var_info;
  var_info.set_use_max_mem_size(12800);
  var_info.set_session_id(1);
  auto mock_var = std::make_unique<MockDeployerVarManager>(*context.var_managers_[device_id][1]);
  context.var_managers_[device_id][1] = std::move(mock_var);
  context.var_managers_[device_id][1]->Initialize(var_info);
  ret = context.ProcessSharedContent(copy_shared_info, response);
  EXPECT_EQ(ret, SUCCESS);
  HeterogeneousExchangeService::GetInstance().Finalize();
}

TEST_F(DeployContextTest, TestProcessVarManagerAndSharedInfoAndSyncMem) {
  DeployContext context;
  deployer::DeployerResponse response;
  deployer::SharedContentDescRequest shared_info;
  auto ret = context.ProcessSharedContent(shared_info, response);

  const int32_t device_id = 0;
  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(device_id);
  init_process_resource_request.set_device_type(0);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);

  auto shared_content_desc = shared_info.mutable_shared_content_desc();
  shared_content_desc->set_session_id(1);
  shared_content_desc->set_node_name("node");
  shared_content_desc->set_om_content("hello");
  shared_content_desc->set_total_length(4);
  shared_content_desc->set_current_offset(0);
  shared_info.add_device_ids(device_id);
  auto copy_shared_info = shared_info;
  context.tansfer_routes_.clear();

  deployer::MultiVarManagerRequest info;
  info.add_device_ids(device_id);
  auto var_manager_info = info.mutable_multi_var_manager_info()->add_var_manager_info();
  var_manager_info->set_device_id(device_id);
  var_manager_info->set_session_id(1);
  var_manager_info->set_use_max_mem_size(128);
  var_manager_info->set_var_mem_logic_base(0);
  deployer::VarDevAddrMgr var_dev_addr_mgr;
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor_desc, 4);
  GeTensorSerializeUtils::GeTensorDescAsProto(tensor_desc, var_dev_addr_mgr.mutable_desc());
  var_dev_addr_mgr.set_address(0);
  var_dev_addr_mgr.set_dev_addr(0);
  deployer::VarResourceInfo *const var_resource_info = var_manager_info->mutable_var_resource();
  var_resource_info->mutable_var_dev_addr_mgr_map()->insert({0, var_dev_addr_mgr});
  ret = context.ProcessMultiVarManager(info);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_TRUE(context.var_managers_[device_id][1] != nullptr);

  auto remote_plan = shared_info.mutable_flow_route();
  {
    auto tag_endpoint = remote_plan->add_endpoints();
    tag_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag));
    tag_endpoint->set_name("transfer_file_tag");
    auto tag_desc = tag_endpoint->mutable_tag_desc();
    tag_desc->set_name("transfer_file_tag");
    tag_desc->set_rank_id(0);
    tag_desc->set_peer_rank_id(2);

    auto queue_endpoint = remote_plan->add_endpoints();
    queue_endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
    queue_endpoint->set_name("transfer_file");
    auto queue_desc = queue_endpoint->mutable_queue_desc();
    queue_desc->set_name("transfer_file");
    queue_desc->set_depth(16);
    queue_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));

    auto binding = remote_plan->add_bindings();
    binding->set_src_index(0);
    binding->set_dst_index(1);
  }

  auto mock_var_manger = std::make_unique<MockDeployerVarManager>(*context.var_managers_[device_id][1]);
  context.var_managers_[device_id][1] = std::move(mock_var_manger);

  copy_shared_info = shared_info;
  context.tansfer_routes_.clear();
  ret = context.ProcessSharedContent(copy_shared_info, response);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(context.tansfer_routes_.size(), 1);
  // test sync auto malloc
  deployer::UpdateDeployPlanRequest req;
  req.set_device_id(0);
  req.set_root_model_id(1);
  req.set_session_id(1);
  auto submodel_desc = req.add_submodel_descs();
  submodel_desc->set_model_path("model.om");
  submodel_desc->set_is_dynamic(false);
  submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
  context.flow_model_receiver_.UpdateDeployPlan(req);

  deployer::AddFlowRoutePlanRequest req2;
  req2.set_node_id(0);
  req2.set_root_model_id(1);
  context.flow_model_receiver_.AddFlowRoutePlan(req2);

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
  EXPECT_EQ(context.UnloadSubmodels(1), SUCCESS);
  HeterogeneousExchangeService::GetInstance().Finalize();
}

TEST_F(DeployContextTest, TestLoadModel) {
  DeployContext context;
  deployer::UpdateDeployPlanRequest req;
  req.set_device_id(0);
  req.set_root_model_id(1);
  auto options = req.mutable_options();
  options->mutable_global_options()->insert({"TestGlobalOption", "TestGlobalOptionValue"});
  options->mutable_session_options()->insert({"TestSessionOption", "TestSessionOptionValue"});
  options->mutable_graph_options()->insert({"TestGraphOption", "TestGraphOptionValue"});
  auto submodel_desc = req.add_submodel_descs();
  submodel_desc->set_model_path("model.om");
  submodel_desc->set_is_dynamic(false);
  submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
  context.flow_model_receiver_.UpdateDeployPlan(req);

  deployer::AddFlowRoutePlanRequest req2;
  req2.set_root_model_id(1);
  context.flow_model_receiver_.AddFlowRoutePlan(req2);
  
  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  deployer::SendProfInfoRequest req_body;
  req_body.set_is_prof_start(1);
  req_body.set_prof_data("test");
  req_body.set_model_id(1);

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
  EXPECT_EQ(context.UpdateProfilingInfoProcess(req_body), SUCCESS);
  EXPECT_EQ(context.UnloadSubmodels(1), SUCCESS);
}

TEST_F(DeployContextTest, TestLoadModelWithInvoke) {
  DeployContext context;
  deployer::UpdateDeployPlanRequest req;
  req.set_device_id(0);
  req.set_root_model_id(1);
  auto options = req.mutable_options();
  options->mutable_global_options()->insert({"TestGlobalOption", "TestGlobalOptionValue"});
  options->mutable_session_options()->insert({"TestSessionOption", "TestSessionOptionValue"});
  options->mutable_graph_options()->insert({"TestGraphOption", "TestGraphOptionValue"});
  auto submodel_desc = req.add_submodel_descs();
  submodel_desc->set_model_path("model.om");
  submodel_desc->set_is_dynamic(false);
  submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
  auto proto_invoked_model_queues = submodel_desc->mutable_invoked_model_queues();
  deployer::ModelQueueIndices proto_model_queue_indices;
  proto_model_queue_indices.add_input_queue_indices(0);
  proto_model_queue_indices.add_output_queue_indices(1);
  (*proto_invoked_model_queues)["invoke_stub"] = std::move(proto_model_queue_indices);
  context.flow_model_receiver_.UpdateDeployPlan(req);

  deployer::AddFlowRoutePlanRequest req2;
  req2.set_root_model_id(1);
  auto exchange_plan = req2.mutable_flow_route_plan();
  deployer::EndpointDesc endpoint_desc;
  endpoint_desc.set_name("test1");
  endpoint_desc.set_type(1);  // queue
  *exchange_plan->add_endpoints() = endpoint_desc;
  *exchange_plan->add_endpoints() = endpoint_desc;
  context.flow_model_receiver_.AddFlowRoutePlan(req2);
  
  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
}

TEST_F(DeployContextTest, TestLoadModelNpu) {
  DeployContext context;
  deployer::UpdateDeployPlanRequest req;
  req.set_device_id(0);
  req.set_root_model_id(1);
  auto submodel_desc = req.add_submodel_descs();
  submodel_desc->set_model_path("model.om");
  submodel_desc->set_is_dynamic(false);
  submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
  submodel_desc->set_saved_model_file_path("model.om");
  submodel_desc->set_is_remote_model(true);
  context.flow_model_receiver_.UpdateDeployPlan(req);

  deployer::AddFlowRoutePlanRequest req2;
  req2.set_root_model_id(1);
  context.flow_model_receiver_.AddFlowRoutePlan(req2);

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  deployer::SendProfInfoRequest req_body;
  req_body.set_is_prof_start(1);
  req_body.set_prof_data("test");
  req_body.set_model_id(1);

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
  EXPECT_EQ(context.UpdateProfilingInfoProcess(req_body), SUCCESS);
  EXPECT_EQ(context.UnloadSubmodels(1), SUCCESS);
}

TEST_F(DeployContextTest, TestBatchPreLoadLocalUdf) {
  DeployContext context;
  deployer::UpdateDeployPlanRequest req;
  req.set_device_id(0);
  req.set_root_model_id(1);
  auto submodel_desc = req.add_submodel_descs();
  submodel_desc->set_model_path(" ./ut_udf_models/batch_result/test1.om");
  submodel_desc->set_is_dynamic(false);
  submodel_desc->set_engine_name(PNE_ID_UDF);
  submodel_desc->set_saved_model_file_path("./ut_udf_models/batch_untar_succes/test1_release.tar.gz");
  context.flow_model_receiver_.UpdateDeployPlan(req);

  deployer::AddFlowRoutePlanRequest req2;
  req2.set_root_model_id(1);
  context.flow_model_receiver_.AddFlowRoutePlan(req2);

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");
  // not initialize
  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), FAILED);
}

TEST_F(DeployContextTest, TestLoadModelInParallel) {
  Configurations::GetInstance().information_.node_config.ipaddr = "192.168.1.101";
  DeployContext context;
  {
    deployer::UpdateDeployPlanRequest req;
    req.set_device_id(0);
    req.set_root_model_id(1);
    auto submodel_desc = req.add_submodel_descs();
    submodel_desc->set_model_path("model.om");
    submodel_desc->set_is_dynamic(false);
    submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
    context.flow_model_receiver_.UpdateDeployPlan(req);

    deployer::AddFlowRoutePlanRequest req2;
    req2.set_root_model_id(1);
    context.flow_model_receiver_.AddFlowRoutePlan(req2);
  }
  {
    deployer::UpdateDeployPlanRequest req;
    req.set_device_id(1);
    req.set_root_model_id(1);
    auto submodel_desc = req.add_submodel_descs();
    submodel_desc->set_model_path("model.om");
    submodel_desc->set_is_dynamic(false);
    submodel_desc->set_engine_name("MOCK_ENGINE_NAME");
    context.flow_model_receiver_.UpdateDeployPlan(req);

    deployer::AddFlowRoutePlanRequest req2;
    req2.set_root_model_id(1);
    context.flow_model_receiver_.AddFlowRoutePlan(req2);
  }

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
  EXPECT_EQ(context.UnloadSubmodels(1), SUCCESS);
}

TEST_F(DeployContextTest, TestInitializeAndFinalize) {
  DeployContext context;
  context.Initialize();
  context.submodel_routes_[1] = 0;
  context.Finalize();
  ASSERT_EQ(context.submodel_routes_.size(), 0);
}

TEST_F(DeployContextTest, DownloadDevMaintenanceCfg) {
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;

  request.set_type(deployer::kDownloadConf);
  auto download_config_request = request.mutable_download_config_request();
  download_config_request->set_sub_type(deployer::kLogConfig);
  download_config_request->set_device_id(0);
  std::string conf_data;
  DeviceMaintenanceMasterCfg device_debug_conf;
  // init log env
  int32_t env_val = 1;
  const std::string kLogEventEnableEnvName = "ASCEND_GLOBAL_EVENT_ENABLE";
  const std::string kLogHostFileNumEnvName = "ASCEND_HOST_LOG_FILE_NUM";
  const std::vector<std::string> kLogEnvNames = {kLogEventEnableEnvName, kLogHostFileNumEnvName};
  for (const auto &env_name : kLogEnvNames) {
    setenv(env_name.c_str(), std::to_string(env_val).c_str(), 1);
    env_val++;
  }
  DeviceMaintenanceMasterCfg::InitGlobalMaintenanceConfigs();
  auto ret = device_debug_conf.GetJsonDataByType(DeviceDebugConfig::ConfigType::kLogConfigType,
                                                 conf_data);
  EXPECT_EQ(ret, SUCCESS);
  download_config_request->set_config_data(&conf_data[0], conf_data.size());
  context.DownloadDevMaintenanceCfg(request, response);
  ret = device_debug_conf.GetJsonDataByType(DeviceDebugConfig::ConfigType::kConfigTypeEnd,
                                            conf_data);
  EXPECT_EQ(ret, FAILED);
  for (const auto &env_name : kLogEnvNames) {
    unsetenv(env_name.c_str());
    env_val++;
  }
}

TEST_F(DeployContextTest, ProcessHeartbeat01) {
  auto client = new BuiltinExecutorClient(0);
  client->sub_proc_stat_ = ProcStatus::NORMAL;
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  exe_client.reset((PneExecutorClient *) client);

  auto client1 = new BuiltinExecutorClient(0);
  client1->sub_proc_stat_ = ProcStatus::STOPPED;
  std::unique_ptr<PneExecutorClient> exe_client1 = nullptr;
  exe_client1.reset((PneExecutorClient *) client1);

  auto client2 = new BuiltinExecutorClient(0);
  client2->sub_proc_stat_ = ProcStatus::EXITED;
  std::unique_ptr<PneExecutorClient> exe_client2 = nullptr;
  exe_client2.reset((PneExecutorClient *) client2);

  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 0, 0, "NPU", "", 666};
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);
  ExecutorManager::ExecutorKey key1 = {1, 0, 0, "NPU", "", 777};
  context.executor_manager_.executor_clients_[key1] = std::move(exe_client1);
  ExecutorManager::ExecutorKey key2 = {1, 0, 1, "NPU", "", 888};
  context.executor_manager_.executor_clients_[key2] = std::move(exe_client2);

  std::map<std::string, bool> submodel_instance_name;
  deployer::SubmodelDesc submodel_desc1;
  submodel_desc1.set_model_name("model_1");
  submodel_desc1.set_model_instance_name("model_666_1");
  deployer::SubmodelDesc submodel_desc2;
  submodel_desc2.set_model_name("model_2");
  submodel_desc2.set_model_instance_name("model_666_2");
  deployer::SubmodelDesc submodel_desc3;
  submodel_desc3.set_model_name("model_1");
  submodel_desc3.set_model_instance_name("model_777_1");

  context.local_rootmodel_to_submodel_descs_[1][key].push_back(submodel_desc1);
  submodel_instance_name.emplace("model_666_1", false);
  context.local_rootmodel_to_submodel_descs_[1][key].push_back(submodel_desc2);
  submodel_instance_name.emplace("model_666_2", false);
  context.local_rootmodel_to_submodel_descs_[1][key2].push_back(submodel_desc3);
  submodel_instance_name.emplace("model_777_1", false);

  deployer::DeployerRequest req;
  deployer::DeployerResponse res;

  dlog_setlevel(0, 0, 0);
  context.ProcessHeartbeat(req, res);

  // EXPECT_EQ(res.error_code(), static_cast<int32_t>(ProcStatus::STOPPED));
  for (const auto &submodel_instances : context.abnormal_submodel_instances_name_) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second) {
      printf("local abnormal submodel_instance_name is %s, i=%u\n", submodel_instance.first.c_str(), i);
      auto iter = submodel_instance_name.find(submodel_instance.first);
      EXPECT_NE(iter, submodel_instance_name.end());
      i++;
    }
  }
  const auto abnormal_submodel_instance_name = res.heartbeat_response().abnormal_submodel_instance_name();
  for (const auto &submodel_instances : abnormal_submodel_instance_name) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second.submodel_instance_name()) {
      printf("remote abnormal submodel_instance_name is %s, i=%u\n", submodel_instance.first.c_str(), i);
      auto iter = submodel_instance_name.find(submodel_instance.first);
      EXPECT_NE(iter, submodel_instance_name.end());
      i++;
    }
  }

  res.set_error_code(FAILED);
  context.ProcessHeartbeat(req, res);
  dlog_setlevel(0, 3, 0);
  context.executor_manager_.Finalize();
}

TEST_F(DeployContextTest, ProcessHeartbeat03) {
  auto client = new BuiltinExecutorClient(0);
  client->sub_proc_stat_ = ProcStatus::NORMAL;
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  exe_client.reset((PneExecutorClient *) client);

  auto client1 = new BuiltinExecutorClient(0);
  client1->sub_proc_stat_ = ProcStatus::EXITED;
  std::unique_ptr<PneExecutorClient> exe_client1 = nullptr;
  exe_client1.reset((PneExecutorClient *) client1);

  auto client2 = new BuiltinExecutorClient(0);
  client2->sub_proc_stat_ = ProcStatus::STOPPED;
  std::unique_ptr<PneExecutorClient> exe_client2 = nullptr;
  exe_client2.reset((PneExecutorClient *) client2);

  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 0, 0, "NPU", "", 666};
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);
  ExecutorManager::ExecutorKey key1 = {1, 0, 0, "NPU", "", 777};
  context.executor_manager_.executor_clients_[key1] = std::move(exe_client1);
  ExecutorManager::ExecutorKey key2 = {1, 0, 1, "NPU", "", 888};
  context.executor_manager_.executor_clients_[key2] = std::move(exe_client2);

  deployer::DeployerRequest req;
  deployer::DeployerResponse res;
  ASSERT_EQ(context.ProcessHeartbeat(req, res), SUCCESS);

  res.set_error_code(FAILED);
  ASSERT_EQ(context.ProcessHeartbeat(req, res), SUCCESS);
  context.executor_manager_.Finalize();
}

TEST_F(DeployContextTest, ProcessHeartbeat04) {
  auto client = new BuiltinExecutorClient(0);
  client->sub_proc_stat_ = ProcStatus::NORMAL;
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  exe_client.reset((PneExecutorClient *) client);

  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 0, 0, "NPU", "", 666};
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);

  auto flowgw_client1 = new FlowGwClient(0, 0, {}, false);
  flowgw_client1->status_func_ = []() -> ProcStatus {
    return ProcStatus::NORMAL;
  };
  context.flowgw_client_manager_.clients_.emplace_back(std::move(std::unique_ptr<FlowGwClient>(flowgw_client1)));

  auto flowgw_client2 = new FlowGwClient(0, 0, {}, false);
  flowgw_client2->status_func_ = []() -> ProcStatus {
    return ProcStatus::STOPPED;
  };
  context.flowgw_client_manager_.clients_.emplace_back(std::move(std::unique_ptr<FlowGwClient>(flowgw_client2)));

  auto flowgw_client3 = new FlowGwClient(0, 0, {}, false);
  flowgw_client3->status_func_ = []() -> ProcStatus {
    return ProcStatus::EXITED;
  };
  context.flowgw_client_manager_.clients_.emplace_back(std::move(std::unique_ptr<FlowGwClient>(flowgw_client3)));

  deployer::DeployerRequest req;
  deployer::DeployerResponse res;
  ASSERT_EQ(context.ProcessHeartbeat(req, res), SUCCESS);

  res.set_error_code(FAILED);
  ASSERT_EQ(context.ProcessHeartbeat(req, res), SUCCESS);
  context.executor_manager_.Finalize();
  context.flowgw_client_manager_.Finalize();
}

TEST_F(DeployContextTest, AddAbnormalSubmodelInstance) {
  DeployContext context;
  deployer::DeployerResponse response;
  std::map<uint32_t, std::vector<std::string>> model_instance_name;
  model_instance_name[1].push_back("model1");
  model_instance_name[1].push_back("model1");
  context.AddAbnormalSubmodelInstance(response, model_instance_name);
  const auto abnormal_submodel_instance_name = response.heartbeat_response().abnormal_submodel_instance_name();
  for (const auto &submodel_instances : abnormal_submodel_instance_name) {
    EXPECT_EQ(submodel_instances.first, 1);
    uint32_t i = 0U;
    for (auto &submodel_instance : submodel_instances.second.submodel_instance_name()) {
      printf("add abnormal submodel instance abnormal submodel_instance_name is %s, i=%u\n",
          submodel_instance.first.c_str(), i);
      EXPECT_EQ(submodel_instance.first, "model1");
      i++;
    }
  }
  context.Finalize();
}

TEST_F(DeployContextTest, TestCreateTransferQueueSuccess) {
  DeployContext context;
  uint32_t queue_id = 0U;
  ASSERT_EQ(context.GetOrCreateTransferQueue(0, queue_id), SUCCESS);
  // get from cache
  ASSERT_EQ(context.GetOrCreateTransferQueue(0, queue_id), SUCCESS);
  context.Finalize();
}

TEST_F(DeployContextTest, TestDataGwSchedInfoInputOutput) {
  DeployContext context;
  deployer::DataGwSchedInfos request;
  request.set_root_model_id(1U);
  request.set_device_id(0);
  request.set_device_type(0);
  request.set_is_dynamic_sched(true);
  request.set_output_queue_indice(2);
  context.flow_model_receiver_.AddDataGwSchedInfos(request);

  PneExecutorClientCreatorRegistrar<MockPneExecutorClient> registrar("MOCK_ENGINE_NAME");

  EXPECT_EQ(context.PreDeployLocalFlowRoute(1), SUCCESS);
  EXPECT_EQ(context.LoadLocalModel(1), SUCCESS);
  EXPECT_EQ(context.UnloadSubmodels(1), SUCCESS);
}

TEST_F(DeployContextTest, ClearModelRunningData_Failed) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kClearModelData);
  auto model_data_clear_req = request.mutable_model_data_clear();
  std::vector<uint32_t> model_ids = {1};
  model_data_clear_req->mutable_root_model_ids()->Add(model_ids.begin(), model_ids.end());
  
  auto exception_devices = model_data_clear_req->add_exception_dev_info();
  exception_devices->set_device_id(0);
  exception_devices->set_device_type(1);
  model_data_clear_req->set_clear_type(0);

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {1, 1, 0, "NPU", ""};
  context.submodel_devices_[1].emplace(key);

  auto client = new MockPneExecutorClient(0);
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  client->Initialize();
  exe_client.reset((PneExecutorClient *) client);
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);

  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(1);
  init_process_resource_request.set_device_type(1);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);
  const auto &req_body = request.model_data_clear();
  EXPECT_EQ(context.ClearModelRunningData(req_body), FAILED);
  context.executor_manager_.Finalize();
}

TEST_F(DeployContextTest, ClearModelRunningData_Flowgw_Succ) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kClearModelData);
  auto model_data_clear_req = request.mutable_model_data_clear();
  std::vector<uint32_t> model_ids = {1};
  model_data_clear_req->mutable_root_model_ids()->Add(model_ids.begin(), model_ids.end());
  
  auto exception_devices = model_data_clear_req->add_exception_dev_info();
  exception_devices->set_device_id(0);
  exception_devices->set_device_type(1);
  model_data_clear_req->set_clear_type(0);

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {1, 1, 0, "NPU", ""};
  context.submodel_devices_[1].emplace(key);
  auto flowgw_client1 = new FlowGwClient(0, 1, {}, false);
  flowgw_client1->status_func_ = []() -> ProcStatus {
    return ProcStatus::NORMAL;
  };
  context.flowgw_client_manager_.clients_.emplace_back(std::move(std::unique_ptr<FlowGwClient>(flowgw_client1)));

  auto flowgw_client2 = new FlowGwClient(1, 1, {}, false);
  flowgw_client2->status_func_ = []() -> ProcStatus {
    return ProcStatus::NORMAL;
  };
  context.flowgw_client_manager_.clients_.emplace_back(std::move(std::unique_ptr<FlowGwClient>(flowgw_client2)));
  auto client = new MockPneExecutorClient(0);
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  client->Initialize();
  exe_client.reset((PneExecutorClient *) client);
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);

  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(1);
  init_process_resource_request.set_device_type(1);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);
  const auto &req_body = request.model_data_clear();

  std::map<uint32_t, std::set<ExecutorManager::ExecutorKey>> clear_submodel_devices;
  int32_t type = req_body.clear_type();
  std::set<uint32_t> root_model_ids = {1};

  context.submodel_routes_[1] = 1;
  context.GetModelClearInfo(req_body, clear_submodel_devices, root_model_ids);
  EXPECT_EQ(context.SyncSubmitClearFlowgwTasks(root_model_ids, type), SUCCESS);
  std::vector<FlowGwClient::ExceptionDeviceInfo> devices;
  EXPECT_EQ(context.SyncUpdateExceptionRoutes(root_model_ids, devices), SUCCESS);
  context.executor_manager_.Finalize();
}

TEST_F(DeployContextTest, ClearModelRunningData_Succ) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kClearModelData);
  auto model_data_clear_req = request.mutable_model_data_clear();
  std::vector<uint32_t> model_ids = {1};
  model_data_clear_req->mutable_root_model_ids()->Add(model_ids.begin(), model_ids.end());
  
  auto exception_devices = model_data_clear_req->add_exception_dev_info();
  exception_devices->set_device_id(0);
  exception_devices->set_device_type(1);
  model_data_clear_req->set_clear_type(0);

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 1, 0, "NPU", ""};
  context.submodel_devices_[1].emplace(key);

  auto client = new MockPneExecutorClient(0);
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  client->Initialize();
  exe_client.reset((PneExecutorClient *) client);
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);

  deployer::InitProcessResourceRequest init_process_resource_request;
  init_process_resource_request.set_device_id(0);
  init_process_resource_request.set_device_type(1);
  init_process_resource_request.set_rank_table("rank_table");
  init_process_resource_request.set_rank_id(0);
  std::vector<int32_t> res_ids_0 = {0};
  init_process_resource_request.mutable_res_ids()->Add(res_ids_0.begin(), res_ids_0.end());
  EXPECT_EQ(context.InitProcessResource(init_process_resource_request, response), SUCCESS);
  const auto &req_body = request.model_data_clear();
  EXPECT_EQ(context.ClearModelRunningData(req_body), SUCCESS);
  context.executor_manager_.Finalize();
}

TEST_F(DeployContextTest, ExceptionNotify) {
  uint32_t root_model_id = 1;
  deployer::DataFlowExceptionNotifyRequest req_body;
  req_body.set_root_model_id(root_model_id);
  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {0, 1, 0, "NPU", ""};
  context.submodel_devices_[root_model_id].emplace(key);

  auto client = new MockPneExecutorClient(0);
  std::unique_ptr<PneExecutorClient> exe_client = nullptr;
  client->Initialize();
  exe_client.reset((PneExecutorClient *)client);
  context.executor_manager_.executor_clients_[key] = std::move(exe_client);

  EXPECT_EQ(context.DataFlowExceptionNotifyProcess(req_body), SUCCESS);
  context.executor_manager_.Finalize();
}

TEST_F(DeployContextTest, VarManagerPreAllocSuccess) {
  DeployContext context;
  deployer::VarManagerInfo var_manager_info;
  var_manager_info.set_device_id(0);
  var_manager_info.set_session_id(1);
  var_manager_info.set_use_max_mem_size(128);
  var_manager_info.set_var_mem_logic_base(0);
  deployer::VarDevAddrMgr var_dev_addr_mgr;
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_INT32);
  TensorUtils::SetSize(tensor_desc, 4);
  GeTensorSerializeUtils::GeTensorDescAsProto(tensor_desc, var_dev_addr_mgr.mutable_desc());
  var_dev_addr_mgr.set_address(0);
  var_dev_addr_mgr.set_dev_addr(0);
  deployer::VarResourceInfo *const var_resource_info = var_manager_info.mutable_var_resource();
  var_resource_info->mutable_var_dev_addr_mgr_map()->insert({0, var_dev_addr_mgr});
  DeployerVarManager var_manager;
  EXPECT_EQ(var_manager.Initialize(var_manager_info), SUCCESS);
  EXPECT_EQ(context.VarManagerPreAlloc(var_manager), SUCCESS);
}
}  // namespace ge
