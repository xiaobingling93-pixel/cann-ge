/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_deployer_daemon.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include "mmpa/mmpa_api.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "framework/common/debug/log.h"
#include "framework/common/scope_guard.h"
#include "common/mem_grp/memory_group_manager.h"
#include "common/subprocess/subprocess_manager.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/utils/rts_api_utils.h"
#include "dflow/base/utils/process_utils.h"
#include "common/utils/memory_statistic_manager.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "deploy/flowrm/tsd_client.h"
#include "deploy/deployer/deployer_service_impl.h"
#include "deploy/abnormal_status_handler/device_abnormal_status_handler.h"

namespace ge {
namespace {
std::atomic_bool g_is_shutdown(false);
constexpr int32_t kShutdownThreadWaitTimeInSec = 1;
constexpr int32_t kExpectedArgCount = 4;
constexpr int32_t kDefaultTimeout = 10 * 1000;  // 10s
}

ge::DeployerServer ModelDeployerDaemon::grpc_server_{};

ModelDeployerDaemon::ModelDeployerDaemon(bool is_sub_deployer) : is_sub_deployer_(is_sub_deployer) {}

ModelDeployerDaemon::~ModelDeployerDaemon() {
  g_is_shutdown.store(true);
  if (shutdown_thread_.joinable()) {
    shutdown_thread_.join();
  }
}

Status ModelDeployerDaemon::Start(int32_t argc, char_t **argv) {
  g_is_shutdown.store(false);
  (void)std::signal(SIGTERM, static_cast<sighandler_t>(&ModelDeployerDaemon::SignalHandler));

  rtMemQueueSetInputPara para = {};
  (void)rtMemQueueSet(0, RT_MQ_QUEUE_ENABLE_LOCAL_QUEUE, &para);
  GE_CHK_STATUS_RET_NOLOG(SubprocessManager::GetInstance().Initialize());
  GE_MAKE_GUARD(sub_proc_manager, []() { SubprocessManager::GetInstance().Finalize(); });
  if (is_sub_deployer_) {
    GE_CHK_STATUS_RET_NOLOG(StartSubDeployer(argc, argv));
  } else {
    GE_CHK_STATUS_RET_NOLOG(StartDeployerDaemon());
  }
  return SUCCESS;
}

Status ModelDeployerDaemon::StartDeployerDaemon() {
  GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().Initialize(Configurations::GetInstance().GetLocalNode()),
                    "[Initialize][MemoryGroup] failed");
  GE_CHK_STATUS_RET(RtsApiUtils::MbufInit(), "[Initialize][MBuf] failed");

  shutdown_thread_ = std::thread(&ModelDeployerDaemon::GrpcWaitForShutdown, this);
  GE_CHK_STATUS_RET(RunGrpcService());
  return SUCCESS;
}

Status ModelDeployerDaemon::StartSubDeployer(int32_t argc, char_t **argv) {
  GE_CHK_STATUS_RET_NOLOG(ParseCmdLineArgs(argc, argv));
  InitDeployContext();
  MemoryGroupManager::GetInstance().SetQsMemGroupName(parent_group_name_);
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MemGrpAttach(parent_group_name_, kDefaultTimeout));
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufInit());
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::EschedAttachDevice(0));
  deployer_message_server_ = MakeShared<MessageServer>(0, req_msg_queue_id_, rsp_msg_queue_id_);
  GE_CHECK_NOTNULL(deployer_message_server_);
  GE_CHK_STATUS_RET_NOLOG(deployer_message_server_->Initialize());

  std::map<std::string, std::string> args_option;
  args_option.emplace(OPTION_EXEC_IS_USEHCOM, "1");
  args_option.emplace(OPTION_GRAPH_RUN_MODE, "0");
  GE_CHK_STATUS_RET(ge_executor_.Initialize(args_option), "Fail to init ge executor.");

  GE_CHK_STATUS_RET_NOLOG(NotifyInitialized());
  DeviceAbnormalStatusHandler::Instance().Initialize();
  MemoryStatisticManager::Instance().Initialize(parent_group_name_);
  GE_CHK_STATUS_RET_NOLOG(WaitDeployRequest());
  GE_CHK_STATUS_RET_NOLOG(Finalize());
  return SUCCESS;
}

void ModelDeployerDaemon::SignalHandler(int32_t sig_num) {
  (void) sig_num;
  g_is_shutdown.store(true);
}

Status ModelDeployerDaemon::RunGrpcService() {
  grpc_server_.SetServiceProvider(MakeUnique<DeployerDaemonService>());
  GE_CHK_STATUS_RET(grpc_server_.Run(), "Failed to run grpc server");
  return SUCCESS;
}

void ModelDeployerDaemon::GrpcWaitForShutdown() const {
  SET_THREAD_NAME(pthread_self(), "ge_dpl_shut");
  while (!g_is_shutdown.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(kShutdownThreadWaitTimeInSec));
  }
  grpc_server_.Finalize();
}

Status ModelDeployerDaemon::ParseCmdLineArgs(int32_t argc, char_t **argv) {
  GE_CHK_BOOL_RET_STATUS(argc >= kExpectedArgCount, PARAM_INVALID, "Sub deployer arg count (%d) is invalid", argc);
  const char_t *parent_group_name = argv[1];
  GE_CHECK_NOTNULL(parent_group_name);
  parent_group_name_ = std::string(parent_group_name);
  const char_t *req_msg_queue_id = argv[2];
  GE_CHECK_NOTNULL(req_msg_queue_id);
  GE_CHK_STATUS_RET(ToNumber(req_msg_queue_id, req_msg_queue_id_), "Request queue id %s is invalid", req_msg_queue_id);
  const char_t *rsp_msg_queue_id = argv[3];
  GE_CHECK_NOTNULL(rsp_msg_queue_id);
  GE_CHK_STATUS_RET(ToNumber(rsp_msg_queue_id, rsp_msg_queue_id_), "Response queue id %s is invalid", rsp_msg_queue_id);
  GELOGI("[Parse][Args] success, parent_group_name:%s, req_msg_queue_id:%u, rsp_msg_queue_id:%u",
         parent_group_name_.c_str(), req_msg_queue_id_, rsp_msg_queue_id_);
  return SUCCESS;
}

void ModelDeployerDaemon::InitDeployContext() {
  std::string ctx_name = std::string("client_") + std::to_string(mmGetPid());
  context_.SetName(ctx_name);
  context_.SetDeployerPid(mmGetPid());
  context_.Initialize();
}

Status ModelDeployerDaemon::NotifyInitialized() {
  deployer::DeployerResponse response;
  response.set_error_code(SUCCESS);
  response.set_error_message("Executor initialized success.");
  GE_CHK_STATUS_RET(deployer_message_server_->SendResponse(response), "[Notify][Initialized] failed");
  GELOGD("[Notify][Initialized] success");
  return SUCCESS;
}

Status ModelDeployerDaemon::WaitDeployRequest() {
  GELOGI("Wait deploy request started");
  bool is_finalize = false;
  while (!g_is_shutdown.load()) {
    auto request = MakeShared<deployer::DeployerRequest>();
    const auto ret = deployer_message_server_->WaitRequest(*request, is_finalize);
    if (ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      continue;
    }
    GE_CHK_STATUS_RET(ret, "Failed to wait deploy request");
    GE_IF_BOOL_EXEC(request->type() == deployer::kDisconnect || is_finalize, GEEVENT("Wait disconnect request success");
                    return SUCCESS);
    (void)pool_.commit([this, request]() -> void { ProcessDeployRequest(*request); });
  }
  return SUCCESS;
}

void ModelDeployerDaemon::ProcessDeployRequest(const deployer::DeployerRequest &request) {
  GELOGD("On event: %s", deployer::DeployerRequestType_Name(request.type()).c_str());
  deployer::DeployerResponse response;
  response.set_message_id(request.message_id());
  (void)DeployerServiceImpl::GetInstance().Process(context_, request, response);
  if (response.error_code() == SUCCESS) {
    GELOGD("[Handle][Event] succeeded");
    response.set_error_message("[Handle][Event] succeeded");
  } else {
    GELOGD("[Handle][Event] failed, error_code = %u, error_msg = %s", response.error_code(),
           response.error_message().c_str());
  }
  (void)deployer_message_server_->SendResponse(response);
  GELOGD("End event: %s", deployer::DeployerRequestType_Name(request.type()).c_str());
}

Status ModelDeployerDaemon::Finalize() {
  MemoryStatisticManager::Instance().Finalize();
  DeviceAbnormalStatusHandler::Instance().Finalize();
  pool_.Destroy();
  GE_CHK_STATUS_RET(ge_executor_.Finalize(), "Failed to finalize ge_executor");
  context_.Finalize();
  TsdClient::GetInstance().Finalize();
  if (deployer_message_server_ != nullptr) {
    deployer_message_server_->Finalize();
    deployer_message_server_.reset();
  }
  g_is_shutdown.store(true);
  GEEVENT("Sub model deployer finalize.");
  return SUCCESS;
}
}  // namespace ge
