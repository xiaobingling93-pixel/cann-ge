/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api/gelib/gelib.h"

#include <cstdlib>
#include <mutex>
#include <string>

#include "analyzer/analyzer.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/plugin/ge_make_unique_util.h"
#include "platform/platform_info.h"
#include "common/platform_info_util.h"
#include "ge/ge_api_types.h"
#include "framework/omg/ge_init.h"
#include "framework/common/helper/model_helper.h"
#include "graph/bin_cache/op_binary_cache.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "register/core_num_utils.h"

namespace ge {
namespace {
const int32_t kDecimal = 10;
const int32_t kSocVersionLen = 50;
const int32_t kDefaultDeviceIdForTrain = 0;
const int32_t kDefaultDeviceIdForInfer = -1;
const char *const kGlobalOptionFpCeilingModeDefault = "2";
const char *const kFpCeilingMode = "ge.fpCeilingMode";
const char *const kVectorcoreNum = "ge.vectorcoreNum";

void SetJitCompileDefaultValue(std::map<std::string, std::string> &options) {
  if (options.find(JIT_COMPILE) != options.cend()) {
    GELOGI("jit_compile option exists, value: %s", options[JIT_COMPILE].c_str());
    return;
  }
  auto jit_compile_default_value = PlatformInfoUtil::GetJitCompileDefaultValue();
  GELOGI("jit_compile option does not exist, use default value %s", jit_compile_default_value.c_str());
  options[JIT_COMPILE] = std::move(jit_compile_default_value);
}

int32_t GetSyncTimeoutValue(const std::string option) {
  int32_t value = -1;
  if (!option.empty()) {
    try {
      value = static_cast<int32_t>(std::stoi(option.c_str()));
    } catch (std::invalid_argument &) {
      GELOGE(ge::FAILED, "Get SyncTimeout failed, transform %s to int failed, as catching invalid_argument exception",
             option.c_str());
    } catch (std::out_of_range &) {
      GELOGE(ge::FAILED, "Get SyncTimeout failed, transform %s to int failed, as catching out_of_range exception",
             option.c_str());
    }
  }
  return value;
}

void SetSyncTimeout(std::map<std::string, std::string> &options) {
  if (options.find(ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT) != options.end()) {
    auto stream_sync_timeout = GetSyncTimeoutValue(options[ge::OPTION_EXEC_STREAM_SYNC_TIMEOUT]);
    ge::GetContext().SetStreamSyncTimeout(stream_sync_timeout);
  }
  if (options.find(ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT) != options.end()) {
    auto event_sync_timeout = GetSyncTimeoutValue(options[ge::OPTION_EXEC_EVENT_SYNC_TIMEOUT]);
    ge::GetContext().SetEventSyncTimeout(event_sync_timeout);
  }
}

void SetOptionNameMap(std::map<std::string, std::string> &options) {
  const auto iter = options.find(ge::OPTION_NAME_MAP);
  if (iter != options.end()) {
    (void)ge::GetContext().SetOptionNameMap(iter->second);
  }
}
}  // namespace
static std::shared_ptr<GELib> instancePtr_ = nullptr;

// Initial each module of GE, if one failed, release all
Status GELib::Initialize(const std::map<std::string, std::string> &options) {
  GEEVENT("[GEPERFTRACE] GE Init Start");
  GE_TRACE_START(GELibInitializePrepare);
  // Multiple initializations are not allowed
  instancePtr_ = MakeShared<GELib>();
  if (instancePtr_ == nullptr) {
    GELOGE(GE_CLI_INIT_FAILED, "[Create][GELib]GeLib initialize failed, malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "GELib Init failed for new GeLib failed.");
    return GE_CLI_INIT_FAILED;
  }

  std::map<std::string, std::string> new_options;
  new_options.insert(options.cbegin(), options.cend());
  if ((options.find("ge.exec.rankId") != options.end()) && (options.find("ge.exec.rankTableFile") != options.end())) {
    new_options[ge::OPTION_EXEC_DEPLOY_MODE] = "0";
  }
  Status ret = instancePtr_->SetRTSocVersion(new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][RTSocVersion]GeLib initial: SetRTSocVersion failed.");
    REPORT_INNER_ERR_MSG("E19999", "SetRTSocVersion failed.");
    return ret;
  }
  SetJitCompileDefaultValue(new_options);
  SetSyncTimeout(new_options);
  SetOptionNameMap(new_options);
  const auto &wait_iter = options.find(OP_WAIT_TIMEOUT);
  if (wait_iter != options.end()) {
    std::string op_wait_timeout = wait_iter->second;
    if (op_wait_timeout != "") {
      int32_t wait_timeout;
      GE_ASSERT_SUCCESS(ge::ConvertToInt32(wait_iter->second.c_str(), wait_timeout), "convert [%s] to int failed.",
                        wait_iter->second.c_str());
      if (wait_timeout >= 0) {
        GE_CHK_RT_RET(rtSetOpWaitTimeOut(static_cast<uint32_t>(wait_timeout)));
        GELOGI("Succeeded in setting rtSetOpWaitTimeOut[%s] to runtime.", wait_iter->second.c_str());
      }
    }
  }
  const auto &exe_iter = options.find(OP_EXECUTE_TIMEOUT);
  if (exe_iter != options.end()) {
    std::string op_execute_timeout = exe_iter->second;
    if (op_execute_timeout != "") {
      int32_t execute_timeout;
      GE_ASSERT_SUCCESS(ge::ConvertToInt32(op_execute_timeout.c_str(), execute_timeout), "convert [%s] to int failed.",
                        op_execute_timeout.c_str());
      if (execute_timeout >= 0) {
        GE_CHK_RT_RET(rtSetOpExecuteTimeOut(static_cast<uint32_t>(execute_timeout)));
        GELOGI("Succeeded in setting rtSetOpExecuteTimeOut[%s] to runtime.", exe_iter->second.c_str());
      }
    }
  }

  if (new_options.find(kFpCeilingMode) == new_options.end()) {
    new_options[kFpCeilingMode] = kGlobalOptionFpCeilingModeDefault;
  }

  const auto &os_iter = options.find(OPTION_HOST_ENV_OS);
  const auto &cpu_iter = options.find(OPTION_HOST_ENV_CPU);
  if (os_iter != options.end()) {
    new_options[OPTION_HOST_ENV_OS] = os_iter->second;
  }
  if (cpu_iter != options.end()) {
    new_options[OPTION_HOST_ENV_CPU] = cpu_iter->second;
  }

  GE_INIT_TRACE_TIMESTAMP_END(GELibInitializePrepare, "GELib::InitializePrepare");
  GE_TRACE_START(Init);
  ret = instancePtr_->InnerInitialize(new_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GeLib]GeLib initial failed.");
    REPORT_INNER_ERR_MSG("E19999", "GELib::InnerInitialize failed.");
    instancePtr_ = nullptr;
    return ret;
  }
  // init ccm while ge init, ge serial init and finalize several times in a process in some cases
  OpBinaryCache::GetInstance().GetNodeCcm()->Initialize();
  GE_INIT_TRACE_TIMESTAMP_END(Init, "GELib::Initialize");
  ReportTracingRecordDuration(ge::TracingModule::kCANNInitialize);
  return SUCCESS;
}

Status GELib::InnerInitialize(std::map<std::string, std::string> &options) {
  // Multiple initializations are not allowed
  if (init_flag_) {
    GELOGW("multi initializations");
    return SUCCESS;
  }

  // parse ge.aicoreNum
  GE_CHK_STATUS_RET(CoreNumUtils::ParseAicoreNumFromOption(options));

  const std::string path_base = GetModelPath();
  GELOGI("GE System initial.");
  GE_TRACE_START(SystemInitialize);
  Status initSystemStatus = SystemInitialize(options);
  GE_INIT_TRACE_TIMESTAMP_END(SystemInitialize, "InnerInitialize::SystemInitialize");
  if (initSystemStatus != SUCCESS) {
    GELOGE(initSystemStatus, "[Init][GESystem]GE system initial failed.");
    RollbackInit();
    return initSystemStatus;
  }

  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.GetHardwareInfo(options), "[Get][Hardware]GeLib initial: Get hardware info failed.");

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    // The setting of global options1 should be head of engines' options2, and options1 should be the same with options2
    auto &global_options = GetMutableGlobalOptions();
    for (const auto &option : options) {
      global_options[option.first] = option.second;
    }
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  }

  GELOGI("engineManager initial.");
  GE_TRACE_START(EngineInitialize);
  Status initEmStatus = DNNEngineManager::GetInstance().Initialize(options);
  GE_INIT_TRACE_TIMESTAMP_END(EngineInitialize, "InnerInitialize::EngineInitialize");
  if (initEmStatus != SUCCESS) {
    GELOGE(initEmStatus, "[Init][EngineManager]GE engine manager initial failed.");
    RollbackInit();
    return initEmStatus;
  }

  GELOGI("opsManager initial.");
  GE_TRACE_START(OpsManagerInitialize);
  Status initOpsStatus = OpsKernelManager::GetInstance().Initialize(options);
  GE_INIT_TRACE_TIMESTAMP_END(OpsManagerInitialize, "InnerInitialize::OpsManagerInitialize");
  if (initOpsStatus != SUCCESS) {
    GELOGE(initOpsStatus, "[Init][OpsManager]GE ops manager initial failed.");
    REPORT_INNER_ERR_MSG("E19999", "OpsManager initialize failed.");
    RollbackInit();
    return initOpsStatus;
  }

  GELOGI("opsBuilderManager initial.");
  GE_TRACE_START(OpsKernelBuilderManagerInitialize);
  Status initOpsBuilderStatus = OpsKernelBuilderManager::Instance().Initialize(options, path_base);
  GE_INIT_TRACE_TIMESTAMP_END(OpsKernelBuilderManagerInitialize, "InnerInitialize::OpsKernelBuilderManager");
  if (initOpsBuilderStatus != SUCCESS) {
    GELOGE(initOpsBuilderStatus, "[Init][OpsKernelBuilderManager]GE ops builder manager initial failed.");
    REPORT_INNER_ERR_MSG("E19999", "OpsBuilderManager initialize failed.");
    RollbackInit();
    return initOpsBuilderStatus;
  }

  GELOGI("Start to initialize HostCpuEngine");
  GE_TRACE_START(HostCpuEngineInitialize);
  Status initHostCpuEngineStatus = HostCpuEngine::GetInstance().Initialize(path_base);
  GE_INIT_TRACE_TIMESTAMP_END(HostCpuEngineInitialize, "InnerInitialize::HostCpuEngineInitialize");
  if (initHostCpuEngineStatus != SUCCESS) {
    GELOGE(initHostCpuEngineStatus, "[Init][HostCpuEngine]Failed to initialize HostCpuEngine.");
    REPORT_INNER_ERR_MSG("E19999", "HostCpuEngine initialize failed.");
    RollbackInit();
    return initHostCpuEngineStatus;
  }

  GELOGI("Start to init Analyzer!");
  GE_TRACE_START(AnalyzerInitialize);
  Status init_analyzer_status = ge::Analyzer::GetInstance()->Initialize();
  GE_INIT_TRACE_TIMESTAMP_END(AnalyzerInitialize, "InnerInitialize::AnalyzerInitialize");
  if (init_analyzer_status != SUCCESS) {
    GELOGE(init_analyzer_status, "[Init][Analyzer]Failed to initialize Analyzer.");
    RollbackInit();
    return init_analyzer_status;
  }

  init_flag_ = true;
  return SUCCESS;
}

Status GELib::SystemInitialize(const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(is_system_inited,
                  GELOGW("System is already inited.");
                  return SUCCESS);

  std::string mode = "Infer";
  auto iter = options.find(OPTION_GRAPH_RUN_MODE);
  if (iter != options.end()) {
    int32_t run_mode = 0;
    GE_ASSERT_SUCCESS(ge::ConvertToInt32(iter->second.c_str(), run_mode), "convert [%s] to int failed.",
                      iter->second.c_str());
    if (GraphRunMode(run_mode) >= TRAIN) {
      is_train_mode_ = true;
    }
  }

  device_id_ = is_train_mode_ ? kDefaultDeviceIdForTrain : kDefaultDeviceIdForInfer;
  iter = options.find(OPTION_EXEC_DEVICE_ID);
  if (iter != options.end()) {
    GE_ASSERT_SUCCESS(ge::ConvertToInt32(iter->second.c_str(), device_id_), "convert [%s] to int failed.",
                      iter->second.c_str());
  }

  // In train and infer, profiling is always needed.
  // 1.`is_train_mode_` means case: train
  // 2.`(!is_train_mode_) && (device_id_ != kDefaultDeviceIdForInfer)` means case: online infer
  // these two case with logical device id
  if (is_train_mode_ || (device_id_ != kDefaultDeviceIdForInfer)) {
    mode = is_train_mode_ ? "Training" : "Online infer";
    GetContext().SetCtxDeviceId(static_cast<uint32_t>(device_id_));
    GE_CHK_STATUS_RET(rtSetDevice(device_id_));
  }

  is_system_inited = true;
  GEEVENT("%s init GELib success, device id :%d ", mode.c_str(), device_id_);
  return SUCCESS;
}

Status GELib::SystemFinalize() {
  std::lock_guard<std::mutex> lock(status_mutex_);
  GE_IF_BOOL_EXEC(!is_system_inited,
                  GELOGW("System is not inited.");
                  return SUCCESS);

  std::string mode = "Infer";
  if (is_train_mode_ || (device_id_ != kDefaultDeviceIdForInfer)) {
    mode = is_train_mode_ ? "Training" : "Online infer";
    GE_CHK_RT(rtDeviceReset(device_id_));
  }

  is_system_inited = false;
  GEEVENT("%s finalize GELib success.", mode.c_str());
  return SUCCESS;
}

Status GELib::SetRTSocVersion(std::map<std::string, std::string> &new_options) const {
  std::map<std::string, std::string>::const_iterator it = new_options.find(ge::SOC_VERSION);
  if (it != new_options.end()) {
    GE_CHK_RT_RET(rtSetSocVersion(it->second.c_str()));
    GELOGI("Succeeded in setting SOC_VERSION[%s] to runtime.", it->second.c_str());
  } else {
    char version[kSocVersionLen] = {0};
    rtError_t rt_ret = rtGetSocVersion(version, kSocVersionLen);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
        REPORT_INNER_ERR_MSG("E19999", "rtGetSocVersion failed.");
        GELOGE(rt_ret, "[Get][SocVersion]rtGetSocVersion failed");
        return FAILED;)
    GELOGI("Succeeded in getting SOC_VERSION[%s] from runtime.", version);
    new_options.insert(std::make_pair(ge::SOC_VERSION, version));
  }
  return SUCCESS;
}

std::string GELib::GetPath() {
  return GetModelPath();
}

// Finalize all modules
Status GELib::Finalize() {
  GELOGI("GELib finalization start");
  // Finalization is not allowed before initialization
  if (!init_flag_) {
    GELOGW("GELib not initialize");
    return SUCCESS;
  }

  Status final_state = SUCCESS;
  Status mid_state;
  GELOGI("engineManager finalization.");
  mid_state = DNNEngineManager::GetInstance().Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("engineManager finalize failed");
    final_state = mid_state;
  }

  GELOGI("opsBuilderManager finalization.");
  mid_state = OpsKernelBuilderManager::Instance().Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("opsBuilderManager finalize failed");
    final_state = mid_state;
  }
  GELOGI("opsManager finalization.");
  mid_state = OpsKernelManager::GetInstance().Finalize();
  if (mid_state != SUCCESS) {
    GELOGW("opsManager finalize failed");
    final_state = mid_state;
  }

  GELOGI("VarManagerPool finalization.");
  VarManagerPool::Instance().Destory();

  GELOGI("ExternalWeightManagerPool finalization.");
  ExternalWeightManagerPool::Instance().Destroy();

  GELOGI("HostCpuEngine finalization.");
  HostCpuEngine::GetInstance().Finalize();

  GELOGI("Analyzer finalization");
  Analyzer::GetInstance()->Finalize();

  (void)SystemFinalize();

  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetMutableGlobalOptions().erase(ENABLE_SINGLE_STREAM);
  }

  is_train_mode_ = false;
  instancePtr_ = nullptr;
  init_flag_ = false;
  if (final_state != SUCCESS) {
    GELOGE(FAILED, "[Check][State]finalization failed.");
    REPORT_INNER_ERR_MSG("E19999", "GELib::Finalize failed.");
    return final_state;
  }
  // clear ccm while ge finalize
  OpBinaryCache::GetInstance().GetNodeCcm()->Finalize();
  GELOGI("GELib finalization success.");
  return SUCCESS;
}

// Get Singleton Instance
std::shared_ptr<GELib> GELib::GetInstance() { return instancePtr_; }

void GELib::RollbackInit() {
  if (DNNEngineManager::GetInstance().init_flag_) {
    (void)DNNEngineManager::GetInstance().Finalize();
  }
  if (OpsKernelManager::GetInstance().init_flag_) {
    (void)OpsKernelManager::GetInstance().Finalize();
  }
  VarManagerPool::Instance().Destory();
  ExternalWeightManagerPool::Instance().Destroy();
}

Status GEInit::Initialize(const std::map<std::string, std::string> &options) {
  Status ret = SUCCESS;
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    ret = GELib::Initialize(options);
  }
  return ret;
}

Status GEInit::Finalize() {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr != nullptr) {
    return instance_ptr->Finalize();
  }
  return SUCCESS;
}

std::string GEInit::GetPath() {
  return GELib::GetPath();
}
}  // namespace ge
