/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/engine_daemon.h"
#include <csignal>
#include "mmpa/mmpa_api.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "proto/deployer.pb.h"
#include "event_handler.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/profiling/profiling_init.h"
#include "dflow/base/utils/process_utils.h"
#include "common/utils/memory_statistic_manager.h"
#include "common/config/device_debug_config.h"
#include "common/profiling/command_handle.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/ge_inner_attrs.h"
#include "common/string_util.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "executor/cpu_sched_event_dispatcher.h"
#include "adx_datadump_server.h"
#include "acl/acl.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr int32_t kDefaultTimeout = 10 * 1000;  // 10s
constexpr int32_t kAdxErrorNone = 0;
constexpr uint32_t kModelExeErr = 507018U;
constexpr char_t kDumpoff[] = "off";
constexpr char_t kProfilingPath[] = "/var/log/npu/profiling/";
const char_t * const kArgsKeyBaseDir = "--base_dir";
const char_t * const kArgsKeyDeviceId = "--device_id";
const char_t * const kArgsKeyMsgQueueDeviceId = "--msg_queue_device_id";
std::atomic<bool> kLoopFlag(true);
std::atomic<bool> acl_initialized{false};
}

EngineDaemon::EngineDaemon(bool is_host_cpu) : is_host_cpu_(is_host_cpu) {}

void EngineDaemon::PrintLogLevel() {
  const char_t *env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_GLOBAL_LOG_LEVEL, env_value);
  GEEVENT("[PrintLogLevel] pid = %d, ASCEND_GLOBAL_LOG_LEVEL = [%s].", getpid(), env_value);
}

void EngineDaemon::SignalHandler(int32_t sig_num) {
  (void) sig_num;
  kLoopFlag.store(false);
}

Status EngineDaemon::InitializeWithArgs(int32_t argc, char_t **argv) {
  (void)std::signal(SIGTERM, static_cast<sighandler_t>(&EngineDaemon::SignalHandler));
  PrintLogLevel();
  GE_CHK_STATUS_RET_NOLOG(ParseCmdLineArgs(argc, argv));
  GE_CHK_STATUS_RET_NOLOG(InitializeMaintenanceFromOption());
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MemGrpAttach(mem_group_name_, kDefaultTimeout));
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufInit());
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::EschedAttachDevice(msg_queue_device_id_));
  GE_CHK_STATUS_RET_NOLOG(InitializeExecutor());
  rtMemQueueSetInputPara para = {};
  (void) rtMemQueueSet(msg_queue_device_id_, RT_MQ_QUEUE_ENABLE_LOCAL_QUEUE, &para);
  executor_message_server_ = MakeShared<MessageServer>(msg_queue_device_id_, req_msg_queue_id_, rsp_msg_queue_id_);
  GE_CHECK_NOTNULL(executor_message_server_);
  GE_CHK_STATUS_RET_NOLOG(executor_message_server_->Initialize());
  event_handler_.SetBaseDir(base_dir_);
  GE_CHK_STATUS_RET_NOLOG(event_handler_.Initialize());
  GE_CHK_STATUS_RET_NOLOG(NotifyInitialized());
  MemoryStatisticManager::Instance().Initialize(mem_group_name_);
    if (!acl_initialized) {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
      GELOGE(FAILED, "ACL init failed.");
      return FAILED;
    } else {
      GELOGI("ACL init success.");
      acl_initialized.store(true);
    }
  }
  return SUCCESS;
}

Status EngineDaemon::InitializeExecutor() {
  // executor run on host no need dispatcher
  if (is_host_cpu_) {
    GE_CHK_STATUS_RET(CpuSchedEventDispatcher::GetInstance().Initialize(device_id_, is_host_cpu_),
                      "Failed to init event dispatcher");
  }

  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::SetDevice(device_id_));
  DF_CHK_ACL(aclrtCreateContext(&rt_context_, device_id_));
  GE_CHK_STATUS_RET_NOLOG(InitializeGeExecutor());
  const auto reg_ret = rtRegTaskFailCallbackByModule("NpuExe", [](rtExceptionInfo *excpt_info) {
    if (excpt_info != nullptr) {
      uint32_t retcode = excpt_info->retcode;
      if (retcode == ACL_ERROR_RT_SOCKET_CLOSE) {
        GELOGI("Aicpu sd dosen't exist, npu exe will exit.");
        kLoopFlag.store(false);
      } else if (retcode == kModelExeErr) {
        GELOGW("Model execute failed.");
      } else {
        GELOGD("Callback function is called, retcode[%u]", retcode);
      }
    }
  });
  GELOGI("Register callback, ret = %d.", static_cast<int32_t>(reg_ret));
  return SUCCESS;
}

void EngineDaemon::Finalize() {
  if (acl_initialized) {
    aclFinalize();
    acl_initialized.store(false);
  }
  MemoryStatisticManager::Instance().Finalize();
  (void)FinalizeMaintenance();
  (void)ge_executor_.Finalize();
  event_handler_.Finalize();
  CpuSchedEventDispatcher::GetInstance().Finalize();
  if (executor_message_server_ != nullptr) {
    executor_message_server_->Finalize();
    executor_message_server_.reset();
  }
  if (rt_context_ != nullptr) {
    is_finish_.store(false);
    std::thread th(&EngineDaemon::FinalizeThread, this);
    (void)aclrtDestroyContext(rt_context_);
    is_finish_.store(true);
    cond_var_.notify_one();
    th.join();
  }
  (void)aclrtResetDevice(device_id_);
  kLoopFlag.store(false);
  GEEVENT("Engine daemon finalized, device id = %d", device_id_);
}

void EngineDaemon::FinalizeThread() {
  SET_THREAD_NAME(pthread_self(), "ge_dpl_edfin");
  std::unique_lock<std::mutex> lk(single_mutex_);
  constexpr int32_t defaultTimeout = 1;
  cond_var_.wait_for(lk, std::chrono::minutes(defaultTimeout), [this] { return is_finish_.load(); });
  if (!is_finish_.load()) {
    GELOGE(FAILED, "EngineDaemon finalize timeout.");
    _exit(-1);
  }
}

Status EngineDaemon::InitProfilingFromOption(const std::map<std::string, std::string> &options_all) const {
  auto options = options_all;
  if (!is_host_cpu_) {
    GE_CHK_STATUS_RET(ge::ProfilingInit::Instance().Init(options), "Failed to init profiling");
    return SUCCESS;
  }
  rtProfCtrlHandle callback = ProfCtrlHandle;
  const rtError_t ret = rtProfRegisterCtrlCallback(GE, callback);
  if (ret != RT_ERROR_NONE) {
    GELOGE(FAILED, "Register CtrlCallback failed.");
    return FAILED;
  }
  bool is_execute_profiling = true;
  const auto &cfg_data = options[kProfilingDeviceConfigData];
  const auto &is_execute_on = options[kProfilingIsExecuteOn];
  if (cfg_data.empty()) {
    GELOGI("kProfilingDeviceConfigData is not set.");
    is_execute_profiling = false;
  }
  if (is_execute_on.empty() || (is_execute_on == "0")) {
    is_execute_profiling = false;
    GELOGI("kProfilingIsExecuteOn is not enable.");
  }
  if (is_execute_profiling) {
    struct MsprofCommandHandleParams prof_conf = {};
    if (strncpy_s(prof_conf.profData, sizeof(prof_conf.profData), cfg_data.c_str(), cfg_data.length()) != EOK) {
      GELOGE(INTERNAL_ERROR, "[copy][ProfilingConfigOption]Failed, data %s.", cfg_data.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Copy profiling data %s failed.", cfg_data.c_str());
      return INTERNAL_ERROR;
    }
    prof_conf.profDataLen = strlen(prof_conf.profData);
    if (strncpy_s(prof_conf.path, sizeof(prof_conf.path), kProfilingPath, strlen(kProfilingPath) + 1) != EOK) {
      GELOGE(INTERNAL_ERROR, "[copy][ProfilingConfigOption]Failed, path %s.", kProfilingPath);
      REPORT_INNER_ERR_MSG("E19999", "Copy profiling_options %s failed.", kProfilingPath);
      return INTERNAL_ERROR;
    }
    prof_conf.pathLen = strlen(prof_conf.path);
    prof_conf.storageLimit = UINT32_MAX;
    uint32_t data_type = is_host_cpu_ ? static_cast<uint32_t>(MSPROF_CTRL_INIT_PURE_CPU) :
                                        static_cast<uint32_t>(MSPROF_CTRL_INIT_HELPER);
    const auto df_ret = MsprofInit(data_type, &prof_conf, sizeof(prof_conf));
     if (df_ret != 0) {
       GELOGE(INTERNAL_ERROR, "[Call][msprofCtrlCallback]Failed, type %u, return %d", data_type, df_ret);
       REPORT_INNER_ERR_MSG("E19999", "Call msprofCtrlCallback failed, type %u, return %d", data_type, df_ret);
       return INTERNAL_ERROR;
     }
    GELOGI("Profiling init in binary, return %d.", df_ret);
  } else {
    const auto df_ret = MsprofInit(MSPROF_CTRL_INIT_DYNA, nullptr, 0);
    GELOGI("Default profiling init, return %d.", df_ret);
  }
  return SUCCESS;
}

Status EngineDaemon::InitDumpFromOption(std::map<std::string, std::string> &options) {
  DumpConfig dump_cfg;
  args_option_[kExecutorDevId] = std::to_string(device_id_);
  const bool is_enable = DumpManager::GetInstance().GetCfgFromOption(options, dump_cfg);
  const Status ret = ge_executor_.SetDump(dump_cfg);
  if (!is_enable) {
    GELOGI("Dump is not open, do not need to init dump server, ret=%d.", ret);
    return ret;
  }
  const int32_t adx_ret = AdxDataDumpServerInit();
  if (adx_ret != kAdxErrorNone) {
    GELOGE(ge::INTERNAL_ERROR, "[AdxDataDumpServer][Init]dump server run failed, adx result[%d].", adx_ret);
    return ge::INTERNAL_ERROR;
  }
  is_dump_inited_ = true;
  return ret;
}

Status EngineDaemon::InitializeMaintenanceFromOption() {
  GE_CHK_STATUS_RET(InitProfilingFromOption(args_option_), "InitProfilingFromOption failed.");
  GE_CHK_STATUS_RET(InitDumpFromOption(args_option_), "InitDumpFromOption failed.");
  return SUCCESS;
}

Status EngineDaemon::LoopEvents() {
  GELOGD("Event loop started");
  bool is_finalize = false;
  while (kLoopFlag) {
    deployer::ExecutorRequest request;
    const auto ret = executor_message_server_->WaitRequest(request, is_finalize);
    if (ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      GELOGD("No event was received, continue");
      continue;
    }
    GE_CHK_STATUS_RET(ret, "Failed to wait request event");
    if (is_finalize) {
      break;
    }
    deployer::ExecutorResponse response;
    GE_CHK_STATUS_RET_NOLOG(HandleEvent(request, response));
  }
  GELOGI("Event loop end.");
  return SUCCESS;
}

Status EngineDaemon::FinalizeMaintenance() {
  const auto ret = MsprofFinalize();
  GE_CHK_STATUS_RET(ret, "MsprofFinalize failed, ret=%d.", ret);
  if (is_dump_inited_) {
    const int32_t adx_ret = AdxDataDumpServerUnInit();
    GE_CHK_STATUS_RET(adx_ret, "AdxDataDumpServerUnInit failed, ret=%d.", adx_ret);
    DumpConfig dump_cfg;
    dump_cfg.dump_status = kDumpoff;
    dump_cfg.dump_debug = kDumpoff;
    const Status dump_ret = ge_executor_.SetDump(dump_cfg);
    GE_CHK_STATUS_RET(ret, "Executor set dump failed, ret=%d.", dump_ret);
  }
  is_dump_inited_ = false;
  return SUCCESS;
}

Status EngineDaemon::HandleEvent(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response) {
  event_handler_.HandleEvent(request, response);
  if (response.error_code() == SUCCESS) {
    GELOGD("[Handle][Event] succeeded");
    response.set_error_message("[Handle][Event] succeeded");
  } else {
    GELOGD("[Handle][Event] failed, error_code = %u, error_msg = %s", response.error_code(),
           response.error_message().c_str());
  }
  GE_CHK_STATUS_RET(executor_message_server_->SendResponse(response), "[Handle][Event] send response failed.");
  return SUCCESS;
}

Status EngineDaemon::InitializeWithKVArgs() {
  const auto &dir_it = args_option_.find(kArgsKeyBaseDir);
  if (dir_it != args_option_.cend()) {
    base_dir_ = dir_it->second;
    GELOGI("Get arg base_dir succeeded, dir = %s.", base_dir_.c_str());
  }

  const auto &dev_it = args_option_.find(kArgsKeyDeviceId);
  if (dev_it != args_option_.cend()) {
    GE_CHK_STATUS_RET_NOLOG(ToNumber(dev_it->second.c_str(), device_id_));
    GELOGI("Get arg device_id succeeded, device_id_ = %d.", device_id_);
  }

  const auto &queue_dev_it = args_option_.find(kArgsKeyMsgQueueDeviceId);
  if (queue_dev_it != args_option_.cend()) {
    GE_CHK_STATUS_RET_NOLOG(ToNumber(queue_dev_it->second.c_str(), msg_queue_device_id_));
    GELOGI("Get arg msg_queue_device_id succeeded, msg_queue_device_id_ = %d.", msg_queue_device_id_);
  }
  return SUCCESS;
}

void EngineDaemon::TransArray2ArgsOption(const int32_t start, const int32_t end, char_t **argv) {
  std::map<std::string, std::string> args_option;
  GELOGI("TransArray2ArgsOption enter, start=%d, end=%d.", start, end);
  for (int32_t var_id = start; var_id < end; var_id++) {
    const std::string str(argv[var_id]);
    constexpr size_t kPairArgsNum = 2UL;
    constexpr size_t kKeyPos = 0UL;
    constexpr size_t kValuePos = 1UL;
    std::vector<std::string> pair = StringUtils::Split(str, '=');
    if (pair.size() != kPairArgsNum) {
      GELOGW("Can not parse args in %s.", argv[var_id]);
      continue;
    }
    args_option.emplace(pair[kKeyPos], pair[kValuePos]);
  }
  args_option_ = std::move(args_option);
}

void EngineDaemon::GetGlobalEnvOptions(std::map<std::string, std::string> &env_option) {
  const std::string kLogLevelEnvName = "ASCEND_GLOBAL_LOG_LEVEL";
  const std::string kLogEventEnableEnvName = "ASCEND_GLOBAL_EVENT_ENABLE";
  const std::string kLogHostFileNumEnvName = "ASCEND_HOST_LOG_FILE_NUM";
  const std::vector<std::string> kLogEnvNames = {kLogLevelEnvName, kLogEventEnableEnvName, kLogHostFileNumEnvName};
  for (const auto &log_name : kLogEnvNames) {
    constexpr size_t kMaxClusterEnvStrLen = 1024UL;
    char_t env_value[kMaxClusterEnvStrLen] = {};
    const int32_t ret = mmGetEnv(log_name.c_str(), env_value, kMaxClusterEnvStrLen);
    if (ret != EN_OK) {
      GELOGW("[Check][Env]Get env[%s] failed.", log_name.c_str());
      continue;
    }
    GELOGD("Get env, key=%s, val=%s.", log_name.c_str(), env_value);
    env_option.emplace(log_name, std::string(env_value));
  }
}

Status EngineDaemon::ParseCmdLineArgs(int32_t argc, char_t **argv) {
  const int32_t kExpectedArgCount = 5;
  if (argc < kExpectedArgCount) {
    GELOGE(PARAM_INVALID, "[Parse][Args] failed, arg count (%d) is invalid", argc);
    return PARAM_INVALID;
  }
  const char_t *memory_group_name = argv[1];
  GE_CHECK_NOTNULL(memory_group_name);
  mem_group_name_ = std::string(memory_group_name);
  const char_t *req_msg_queue_id = argv[2];
  GE_CHECK_NOTNULL(req_msg_queue_id);
  GE_CHK_STATUS_RET_NOLOG(ToNumber(req_msg_queue_id, req_msg_queue_id_));
  const char_t *rsp_msg_queue_id = argv[3];
  GE_CHECK_NOTNULL(rsp_msg_queue_id);
  GE_CHK_STATUS_RET_NOLOG(ToNumber(rsp_msg_queue_id, rsp_msg_queue_id_));
  const char_t *device_id = argv[4];
  GE_CHECK_NOTNULL(device_id);
  GE_CHK_STATUS_RET_NOLOG(ToNumber(device_id, device_id_));
  TransArray2ArgsOption(kExpectedArgCount - 1, argc, argv);
  GE_CHK_STATUS_RET(InitializeWithKVArgs(), "Failed to init with kv args.");
  // for debug
  std::map<std::string, std::string> env_option;
  GetGlobalEnvOptions(env_option);
  for (const auto &env_it : env_option) {
    const auto &key = env_it.first;
    const auto &val = env_it.second;
    GELOGI("Get global env options, env=%s, env_val=%s.", key.c_str(), val.c_str());
  }
  GELOGD("[Parse][Args] succeeded, get env_option size=%zu, mem_grp_name = %s, req_msg_queue_id = %s, "
         "rsp_msg_queue_id = %s, device_id = %d, msg_queue_device_id = %d",
         env_option.size(), mem_group_name_.c_str(),
         req_msg_queue_id, rsp_msg_queue_id, device_id_, msg_queue_device_id_);
  return SUCCESS;
}

Status EngineDaemon::NotifyInitialized() const {
  deployer::ExecutorResponse response;
  response.set_error_code(SUCCESS);
  response.set_error_message("Executor initialized success.");
  GE_CHK_STATUS_RET(executor_message_server_->SendResponse(response), "[Notify][Initialized] failed, device_id = %d.",
                    msg_queue_device_id_);
  GELOGD("[Notify][Initialized] success");
  return SUCCESS;
}

Status EngineDaemon::InitializeGeExecutor() {
  ExecutionRuntimeUtils::EnableGlobalInHeterogeneousExecutor();
  args_option_.emplace(OPTION_EXEC_IS_USEHCOM, "1");
  args_option_.emplace(OPTION_GRAPH_RUN_MODE, "0");
  std::string hccl_flag = "1";
  if (is_host_cpu_) {
    hccl_flag = "0";
  }
  args_option_.emplace(OPTION_EXEC_HCCL_FLAG, hccl_flag);
  GE_CHK_STATUS_RET(ge_executor_.Initialize(args_option_), "Failed to init ge executor");
  if (is_host_cpu_) {
    std::map<std::string, std::string> options {
      {"ge.exec.placement", "HOST"},
    };
    ge::GetThreadLocalContext().SetGlobalOption(options);
    GetContext().SetCtxDeviceId(static_cast<uint32_t>(-1));
  } else {
    GetContext().SetCtxDeviceId(static_cast<uint32_t>(device_id_));
  }
  return SUCCESS;
}
}  // namespace ge
