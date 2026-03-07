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
#include "macro_utils/dt_public_scope.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "common/config/configurations.h"
#include "macro_utils/dt_public_unscope.h"


using namespace std;
namespace ge {
class DeployerServiceImplTest : public testing::Test {
 protected:
  void SetUp() override {
    DeviceDebugConfig::global_configs_ = nlohmann::json{};
  }
  void TearDown() override {
    DeviceDebugConfig::global_configs_ = nlohmann::json{};
  }
};

TEST_F(DeployerServiceImplTest, RequestTypeNotRegistered) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  EXPECT_EQ(deployer_service.Process(context, request, response), SUCCESS);
  EXPECT_EQ(response.error_code(), FAILED);
}

TEST_F(DeployerServiceImplTest, ProcessFuncIsNull) {
  ge::DeployerServiceImpl deployer_service;
  deployer_service.RegisterReqProcessor(deployer::kHeartbeat, nullptr);
  deployer::DeployerRequest request;
  request.set_type(deployer::kHeartbeat);
  deployer::DeployerResponse response;
  DeployContext context;
  EXPECT_EQ(deployer_service.Process(context, request, response), SUCCESS);
  EXPECT_EQ(response.error_code(), FAILED);
}

TEST_F(DeployerServiceImplTest, Process) {
  auto process_func = [](DeployContext &context,
                         const deployer::DeployerRequest &request,
                         deployer::DeployerResponse &response) {
    response.set_error_code(666);
    response.set_error_message("ErrorMessage");
  };
  ge::DeployerServiceImpl deployer_service;
  deployer_service.RegisterReqProcessor(deployer::kHeartbeat, process_func);
  deployer::DeployerRequest request;
  request.set_type(deployer::kHeartbeat);
  deployer::DeployerResponse response;
  DeployContext context;
  EXPECT_EQ(deployer_service.Process(context, request, response), SUCCESS);
  EXPECT_EQ(response.error_code(), 666);
  EXPECT_EQ(response.error_message(), "ErrorMessage");
}

TEST_F(DeployerServiceImplTest, LoadModel_DeployPlanNotExist) {
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  request.set_type(deployer::kLoadModel);
  DeployContext context;
  DeployerServiceImpl::LoadModelProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
  EXPECT_EQ(response.error_message(), "deploy plan not found");
}

TEST_F(DeployerServiceImplTest, UpdateDeployPlanProcess_InvalidRequest) {
  DeployContext context;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  request.set_type(deployer::kUpdateDeployPlan);
  DeployerServiceImpl::UpdateDeployPlanProcess(context, request, response);
  EXPECT_EQ(response.error_code(), PARAM_INVALID);
  EXPECT_EQ(response.error_message(), "invalid request");
}

TEST_F(DeployerServiceImplTest, AddFlowRoutePlanProcess_InvalidRequest) {
  DeployContext context;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  request.set_type(deployer::kAddFlowRoutePlan);
  DeployerServiceImpl::FlowRoutePlanProcess(context, request, response);
  EXPECT_EQ(response.error_code(), PARAM_INVALID);
  EXPECT_EQ(response.error_message(), "invalid request");
}

TEST_F(DeployerServiceImplTest, AddFlowRoutePlanProcess_InvalidRequest2) {
  DeployContext context;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  request.set_type(deployer::kAddFlowRoutePlan);
  for (int32_t i = 0; i < 64; ++i) {
    auto &body = *request.mutable_add_flow_route_plan_request();
    body.set_root_model_id(i);
    DeployerServiceImpl::FlowRoutePlanProcess(context, request, response);
  }

  auto &body = *request.mutable_add_flow_route_plan_request();
  body.set_root_model_id(64);
  DeployerServiceImpl::FlowRoutePlanProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
}

TEST_F(DeployerServiceImplTest, TransferFileProcess_InvalidRequest) {
  DeployContext context;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  request.set_type(deployer::kTransferFile);
  DeployerServiceImpl::TransferFileProcess(context, request, response);
  EXPECT_EQ(response.error_code(), PARAM_INVALID);
  EXPECT_EQ(response.error_message(), "invalid request");
}

TEST_F(DeployerServiceImplTest, TransferFileProcess_InvalidPath) {
  DeployContext context;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  request.set_type(deployer::kTransferFile);
  auto download_request = request.mutable_transfer_file_request();
  download_request->set_path("/invalid_path");
  DeployerServiceImpl::TransferFileProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
}

TEST_F(DeployerServiceImplTest, LoadModel_Failed) {
  DeployContext context;
  {
    deployer::DeployerRequest request;
    deployer::DeployerResponse response;
    request.set_type(deployer::kUpdateDeployPlan);
    auto &body = *request.mutable_update_deploy_plan_request();
    body.set_device_id(0);
    body.set_root_model_id(1);
    auto options = body.mutable_options();
    options->mutable_global_options()->insert({"TestGlobalOption", "TestGlobalOptionValue"});
    options->mutable_session_options()->insert({"TestSessionOption", "TestSessionOptionValue"});
    options->mutable_graph_options()->insert({"TestGraphOption", "TestGraphOptionValue"});
    DeployerServiceImpl::UpdateDeployPlanProcess(context, request, response);
    EXPECT_EQ(response.error_code(), SUCCESS);
    DeployState *deploy_state = nullptr;
    context.flow_model_receiver_.GetDeployState(1, deploy_state);
    EXPECT_NE(deploy_state, nullptr);
    const auto &global_option = deploy_state->GetAllGlobalOptions();
    auto it = global_option.find("TestGlobalOption");
    EXPECT_NE(it, global_option.end());
    const auto &session_option = deploy_state->GetAllSessionOptions();
    it = session_option.find("TestSessionOption");
    EXPECT_NE(it, session_option.end());
    const auto &graph_option = deploy_state->GetAllGraphOptions();
    it = graph_option.find("TestGraphOption");
    EXPECT_NE(it, graph_option.end());
  }
  {
    deployer::DeployerRequest request;
    deployer::DeployerResponse response;
    request.set_type(deployer::kLoadModel);
    auto &body = *request.mutable_load_model_request();
    body.set_root_model_id(1);
    DeployerServiceImpl::LoadModelProcess(context, request, response);
    EXPECT_EQ(response.error_code(), FAILED);
  }
}

TEST_F(DeployerServiceImplTest, UnloadModel_ModelIdNotFound) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  request.set_type(deployer::kUnloadModel);
  auto &body = *request.mutable_unload_model_request();
  body.set_model_id(1);
  DeployerServiceImpl::UnloadModelProcess(context, request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
}

TEST_F(DeployerServiceImplTest, UnloadModel_Failed) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {};
  key.engine_name = PNE_ID_NPU;
  context.submodel_devices_[1].emplace(key);
  request.set_type(deployer::kUnloadModel);
  auto &body = *request.mutable_unload_model_request();
  body.set_model_id(1);
  DeployerServiceImpl::UnloadModelProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
  EXPECT_EQ(response.error_message(), "Failed to unload model");
}

TEST_F(DeployerServiceImplTest, DownloadDevMaintenanceCfgProcess) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  // invalid json
  DeployerServiceImpl::DownloadDevMaintenanceCfgProcess(context, request, response);
  EXPECT_NE(response.error_code(), SUCCESS);
}

TEST_F(DeployerServiceImplTest, MultiVarManagerInfoProcess) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  request.set_type(deployer::kDownloadVarManager);
  DeployerServiceImpl::MultiVarManagerInfoProcess(context, request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
}

TEST_F(DeployerServiceImplTest, SharedContentProcess) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  request.set_type(deployer::kDownloadSharedContent);
  // shared_content_desc is not set
  DeployerServiceImpl::SharedContentProcess(context, request, response);
  EXPECT_NE(response.error_code(), SUCCESS);
}

TEST_F(DeployerServiceImplTest, InitProcessResourceProcess) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  deployer::DeployerResponse response;
  DeployContext context;
  request.set_type(deployer::kInitProcessResource);
  auto &body = *request.mutable_init_process_resource_request();
  body.set_device_id(0);
  body.set_device_type(CPU);
  std::string rank_table_json =
      "{\"collective_id\":\"192.168.1.1:111-123\",\"master_ip\":\"10.2.3.4\",\"master_port\":\"111\",\"node_list\":[{"
      "\"node_addr\":\"192.168.1.101\",\"ranks\":[{\"device_id\":\"0\",\"port\":\"1966\",\"rank_id\":\"2\"},{\"device_"
      "id\":\"1\",\"port\":\"1777\",\"rank_id\":\"1\"}]},{\"node_addr\":\"192.168.1.100\",\"ranks\":[{\"device_id\":"
      "\"0\",\"port\":\"1888\",\"rank_id\":\"0\"}]}]}";
  body.set_rank_table(rank_table_json);

  Configurations::GetInstance().information_.node_config.ipaddr = "192.168.1.101";
  DeployerServiceImpl::InitProcessResourceProcess(context, request, response);
  EXPECT_EQ(response.error_code(), SUCCESS);
  Configurations::GetInstance().information_ = {};
}

TEST_F(DeployerServiceImplTest, UpdateDeployPlan) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kUpdateDeployPlan);
  auto pre_deploy_req = request.mutable_update_deploy_plan_request();
  pre_deploy_req->set_root_model_id(1);

  auto submodel_desc = pre_deploy_req->add_submodel_descs();
  submodel_desc->set_model_size(128);
  submodel_desc->set_model_name("any-model");
  submodel_desc->add_input_queue_indices(1);
  submodel_desc->add_output_queue_indices(2);
  submodel_desc->set_engine_name(PNE_ID_NPU);

  auto submodel_desc2 = pre_deploy_req->add_submodel_descs();
  submodel_desc2->set_model_size(128);
  submodel_desc2->set_model_name("any-model2");
  submodel_desc2->add_input_queue_indices(1);
  submodel_desc2->add_output_queue_indices(2);
  submodel_desc2->set_engine_name(PNE_ID_NPU);
  submodel_desc2->set_rank_id("0");

  DeployContext context;
  deployer::DeployerResponse response;
  DeployerServiceImpl::UpdateDeployPlanProcess(context, request, response);
  ASSERT_EQ(response.error_code(), ge::SUCCESS);
  DeployState *deploy_state = nullptr;
  context.flow_model_receiver_.GetDeployState(1, deploy_state);
  ASSERT_NE(deploy_state, nullptr);
  ASSERT_EQ(deploy_state->local_submodel_descs_.size(), 1);
  auto it = deploy_state->local_submodel_descs_.begin();
  ASSERT_EQ(it->first.rank_id, "0");
}

TEST_F(DeployerServiceImplTest, AddFlowRoutePlan) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kAddFlowRoutePlan);
  auto pre_deploy_req = request.mutable_add_flow_route_plan_request();
  pre_deploy_req->set_root_model_id(1);
  auto exchange_plan = pre_deploy_req->mutable_flow_route_plan();
  deployer::EndpointDesc endpoint_desc;
  endpoint_desc.set_name("data-1");
  endpoint_desc.set_type(2);  // tag
  *exchange_plan->add_endpoints() = endpoint_desc;
  endpoint_desc.set_type(1);  // queue
  *exchange_plan->add_endpoints() = endpoint_desc;

  endpoint_desc.set_name("output-1");
  endpoint_desc.set_type(1);  // queue
  *exchange_plan->add_endpoints() = endpoint_desc;
  endpoint_desc.set_type(2);  // tag
  *exchange_plan->add_endpoints() = endpoint_desc;
  auto input_binding = exchange_plan->add_bindings();
  input_binding->set_src_index(0);
  input_binding->set_dst_index(1);
  auto output_binding = exchange_plan->add_bindings();
  output_binding->set_src_index(2);
  output_binding->set_dst_index(3);

  DeployContext context;
  deployer::DeployerResponse response;
  DeployerServiceImpl::FlowRoutePlanProcess(context, request, response);
  ASSERT_EQ(response.error_code(), ge::SUCCESS);
}

TEST_F(DeployerServiceImplTest, DataGwSchedInfo) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kDatagwSchedInfo);
  auto datagw_sched_info = request.mutable_datagw_sched_info();
  datagw_sched_info->set_root_model_id(0U);
  datagw_sched_info->set_device_id(0);
  datagw_sched_info->set_device_type(0);
  datagw_sched_info->set_is_dynamic_sched(true);
  datagw_sched_info->set_input_queue_indice(1);
  datagw_sched_info->set_output_queue_indice(2);

  DeployContext context;
  deployer::DeployerResponse response;
  DeployerServiceImpl::DataGwSchedInfo(context, request, response);
  ASSERT_EQ(response.error_code(), ge::SUCCESS);
}

TEST_F(DeployerServiceImplTest, DataGwSchedInfoParamInvalid) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kDatagwSchedInfo);

  DeployContext context;
  deployer::DeployerResponse response;
  DeployerServiceImpl::DataGwSchedInfo(context, request, response);
  ASSERT_EQ(response.error_code(), ge::PARAM_INVALID);
}

TEST_F(DeployerServiceImplTest, DataGwSchedInfoDeviceIdError) {
  deployer::DeployerRequest request;
  request.set_type(deployer::kDatagwSchedInfo);
  auto datagw_sched_info = request.mutable_datagw_sched_info();
  datagw_sched_info->set_root_model_id(0U);
  datagw_sched_info->set_device_id(-1);
  datagw_sched_info->set_device_type(0);
  datagw_sched_info->set_is_dynamic_sched(true);
  datagw_sched_info->set_input_queue_indice(1);
  datagw_sched_info->set_output_queue_indice(2);

  DeployContext context;
  deployer::DeployerResponse response;
  DeployerServiceImpl::DataGwSchedInfo(context, request, response);
  ASSERT_EQ(response.error_code(), ge::FAILED);
}

TEST_F(DeployerServiceImplTest, ClearModelRunningData_Succ) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  request.set_type(deployer::kClearModelData);
  auto model_data_clear_req = request.mutable_model_data_clear();
  std::vector<uint32_t> model_ids = {2};
  model_data_clear_req->mutable_root_model_ids()->Add(model_ids.begin(), model_ids.end());
  
  auto exception_devices = model_data_clear_req->add_exception_dev_info();
  exception_devices->set_device_id(1);
  exception_devices->set_device_type(1);
  model_data_clear_req->set_clear_type(0);

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {};
  key.engine_name = PNE_ID_NPU;
  context.submodel_devices_[2].emplace(key);
  DeployerServiceImpl::ClearModelRunningData(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
  EXPECT_EQ(response.error_message(), "Failed to clear data");
}

TEST_F(DeployerServiceImplTest, ExceptionNotify) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  request.set_type(deployer::kDataFlowExceptionNotify);
  auto exception_notify_req = request.mutable_exception_notify_request();
  std::vector<uint32_t> model_ids = {2};
  exception_notify_req->set_root_model_id(2);
  auto exception_notify = exception_notify_req->mutable_exception_notify();
  exception_notify->set_type(0);
  exception_notify->set_trans_id(100);

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {};
  key.engine_name = PNE_ID_NPU;
  context.submodel_devices_[2].emplace(key);
  DeployerServiceImpl::DataFlowExceptionNotifyProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
}

TEST_F(DeployerServiceImplTest, UpdateProfInfo) {
  ge::DeployerServiceImpl deployer_service;
  deployer::DeployerRequest request;
  request.set_type(deployer::kUpdateProfilingInfo);
  auto prof_info = request.mutable_prof_info();
  prof_info->set_model_id(2);
  prof_info->set_is_prof_start(1);
  prof_info->set_prof_data("test");

  deployer::DeployerResponse response;
  DeployContext context;
  ExecutorManager::ExecutorKey key = {};
  key.engine_name = PNE_ID_CPU;
  context.submodel_devices_[2].emplace(key);
  DeployerServiceImpl::UpdateProfilingInfoProcess(context, request, response);
  EXPECT_EQ(response.error_code(), FAILED);
}
}
