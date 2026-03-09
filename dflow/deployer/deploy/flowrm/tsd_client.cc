/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/tsd_client.h"
#include <atomic>
#include "aicpu/queue_schedule/dgw_client.h"
#include "runtime/dev.h"
#include "mmpa/mmpa_api.h"
#include "plog.h"
#include "common/debug/log.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "common/data_flow/event/proxy_event_manager.h"
#include "mmpa/mmpa_api.h"
#include "common/thread_pool/thread_pool.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
#define LOG_SAVE_MODE_SEP (0xFE736570U)  // LOG_SAVE_MODE_SEP在unified_dlog.so中不再使用，为保持兼容临时宏定义，中间态过渡用，后续需移除
using TsdFileLoad = uint32_t(*)(const uint32_t device_id,
                                const char_t *file_path,
                                const uint64_t path_len,
                                const char_t *file_name,
                                const uint64_t name_len);
using TsdFileUnLoad = uint32_t(*)(const uint32_t device_id,
                                  const char_t *file_path,
                                  const uint64_t path_len);
using TsdGetProcListStatus = uint32_t(*)(const uint32_t device_id, ProcStatusParam *status, const uint32_t num);
using TsdProcessOpen = uint32_t(*)(const uint32_t device_id, ProcOpenArgs *args);
using ProcessCloseSubProcList = uint32_t(*)(const uint32_t device_id, const ProcStatusParam *status,
                                            const uint32_t num);
using TsdInitFlowGw = uint32_t(*)(const uint32_t device_id, const InitFlowGwInfo * const info);

constexpr const char_t *kTsdClientLibName = "libtsdclient.so";
constexpr const char_t *kFuncNameTsdFileLoad = "TsdFileLoad";
constexpr const char_t *kFuncNameTsdFileUnLoad = "TsdFileUnLoad";
constexpr const char_t *kFuncNameTsdGetProcListStatus = "TsdGetProcListStatus";
constexpr const char_t *kFuncNameTsdProcessOpen = "TsdProcessOpen";
constexpr const char_t *kFuncNameTsdProcessCloseSubProcList = "ProcessCloseSubProcList";
constexpr const char_t *kFuncNameTsdCapabilityGet = "TsdCapabilityGet";
constexpr const char_t *kFuncNameTsdInitFlowGw = "TsdInitFlowGw";
std::atomic<uint64_t> g_load_file_count{0UL};
}  // namespace

TsdClient &TsdClient::GetInstance() {
  static TsdClient instance;
  return instance;
}

Status TsdClient::Initialize() {
  if (inited_) {
    return SUCCESS;
  }

  std::lock_guard<std::mutex> lk(init_mutex_);
  if (!inited_) {
    GE_CHK_STATUS_RET(LoadTsdClientLib(), "Failed to load tsd client lib.");
    GEEVENT("Initialize tsd client successfully.");
    inited_ = true;
  }
  return SUCCESS;
}

uint64_t TsdClient::GetLoadFileCount() {
  return g_load_file_count.fetch_add(1UL);
}

Status TsdClient::SetDlogReportStart(int32_t device_id) {
  std::unique_lock<std::mutex> guard(map_mutex_);
  const auto &iter = set_log_save_mode_.find(device_id);
  if (iter != set_log_save_mode_.cend()) {
    return SUCCESS;
  }
  const int32_t dlog_ret = DlogReportStart(device_id, LOG_SAVE_MODE_SEP);
  GELOGI("Param device_id is %d, dlog_ret is %d.", device_id, dlog_ret);
  if (dlog_ret == EN_OK) {
    set_log_save_mode_.emplace(device_id);
    return SUCCESS;
  }
  return FAILED;
}

void TsdClient::SetDlogReportStop() {
  std::unique_lock<std::mutex> guard(map_mutex_);
  for (const auto &item : set_log_save_mode_) {
    (void)DlogReportStop(item);
  }
  set_log_save_mode_.clear();
}

std::mutex &TsdClient::GetDeviceMutex(int32_t device_id) {
  std::unique_lock<std::mutex> guard(map_mutex_);
  return device_mutexs_[device_id];
}

Status TsdClient::LoadTsdClientLib() {
  const auto open_flag =
      static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  handle_ = mmDlopen(kTsdClientLibName, open_flag);
  GE_CHK_BOOL_RET_STATUS(handle_ != nullptr, FAILED,
                         "[Dlopen][So] failed, so name = %s, error_msg = %s",
                         kTsdClientLibName, mmDlerror());
  GELOGI("Open %s succeeded", kTsdClientLibName);
  tsd_capability_get_ = reinterpret_cast<TsdCapabilityGet>(mmDlsym(handle_, kFuncNameTsdCapabilityGet));
  GE_CHK_BOOL_RET_STATUS(tsd_capability_get_ != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdCapabilityGet, mmDlerror());
  return SUCCESS;
}

Status TsdClient::SetDevice(int32_t device_id) {
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  {
    std::unique_lock<std::mutex> map_guard(map_mutex_);
    auto it = set_device_list_.find(device_id);
    if (it != set_device_list_.cend()) {
      return SUCCESS;
    }
  }
  GELOGI("Set device begin, device_id = %d.", device_id);
  GE_CHK_RT_RET(LoadPackages(device_id));
  GE_CHK_RT_RET(rtSetDevice(device_id));
  std::unique_lock<std::mutex> map_guard(map_mutex_);
  set_device_list_.emplace(device_id);
  GELOGI("Set device success, device_id = %d.", device_id);
  return SUCCESS;
}

Status TsdClient::LoadPackages(int32_t device_id) const {
  GELOGI("Load packages begin, device_id = %d.", device_id);
  constexpr const char *kRuntimePkgName = "Ascend-runtime_device-minios.tar.gz";
  // 老驱动tsd client会加载Ascend-runtime_device-minios.tar.gz，新驱动tsd client内部兼容加载udf和hccd子包
  GE_CHK_STATUS_RET(LoadFileByTsd(device_id, nullptr, 0, kRuntimePkgName), "load runtime pkg failed.");
  GELOGI("Load packages success, device_id = %d.", device_id);
  return SUCCESS;
}

Status TsdClient::GetProcStatus(int32_t device_id, pid_t pid, ProcStatus &proc_status, const std::string &proc_type) {
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  auto proc = reinterpret_cast<TsdGetProcListStatus>(mmDlsym(handle_, kFuncNameTsdGetProcListStatus));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdGetProcListStatus, mmDlerror());
  ProcStatusParam status = {pid, TransferProcType(proc_type), SUB_PROCESS_STATUS_NORMAL};
  status.pid = pid;
  GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id), &status, 1U),
                    "Failed to get proc status, device_id = %d, pid = %d", device_id, pid);
  static std::map<SubProcessStatus, ProcStatus> transfer = {
      {SUB_PROCESS_STATUS_NORMAL, ProcStatus::NORMAL},
      {SUB_PROCESS_STATUS_EXITED, ProcStatus::EXITED},
      {SUB_PROCESS_STATUS_STOPED, ProcStatus::STOPPED}};
  proc_status = transfer[status.curStat];
  return SUCCESS;
}

Status TsdClient::GetProcStatus(int32_t device_id, const std::vector<pid_t> &pids,
                                std::map<pid_t, ProcStatus> &proc_status, const std::string &proc_type) {
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  auto proc = reinterpret_cast<TsdGetProcListStatus>(mmDlsym(handle_, kFuncNameTsdGetProcListStatus));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdGetProcListStatus, mmDlerror());
  SubProcType query_proc_type = TransferProcType(proc_type);
  std::vector<ProcStatusParam> query_status;
  query_status.reserve(pids.size());
  for (const auto &pid : pids) {
    ProcStatusParam status = {pid, query_proc_type, SUB_PROCESS_STATUS_NORMAL};
    query_status.emplace_back(std::move(status));
  }
  {
    std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
    GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id), query_status.data(), query_status.size()),
                      "Failed to get proc status, device_id = %d, pid = %s", device_id, ToString(pids).c_str());
  }
  static std::map<SubProcessStatus, ProcStatus> transfer = {
      {SUB_PROCESS_STATUS_NORMAL, ProcStatus::NORMAL},
      {SUB_PROCESS_STATUS_EXITED, ProcStatus::EXITED},
      {SUB_PROCESS_STATUS_STOPED, ProcStatus::STOPPED}};
  for (const auto &proc_status_info : query_status) {
    proc_status[proc_status_info.pid] = transfer[proc_status_info.curStat];
  }
  return SUCCESS;
}

SubProcType TsdClient::TransferProcType(const std::string &proc_type) {
  static std::map<std::string, SubProcType> transfer = {{PNE_ID_UDF, TSD_SUB_PROC_UDF},
                                                        {PNE_ID_NPU, TSD_SUB_PROC_NPU},
                                                        {"BUILTIN_UDF", TSD_SUB_PROC_BUILTIN_UDF},
                                                        {"queue_schedule", TSD_SUB_PROC_QUEUE_SCHEDULE}};
  return transfer[proc_type];
}

Status TsdClient::StartFlowGw(int32_t device_id,
                              const std::string &group_name,
                              pid_t &pid) {
  (void) group_name;
  GELOGI("Start flowgw begin, device_id = %d.", device_id);
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  auto proc = reinterpret_cast<TsdInitFlowGw>(mmDlsym(handle_, kFuncNameTsdInitFlowGw));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdInitFlowGw, mmDlerror());
  InitFlowGwInfo info = {};
  info.schedPolicy = static_cast<uint64_t>(bqs::SchedPolicy::POLICY_SUB_BUF_EVENT);
  GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id), &info),
                    "Failed to start flowgw device_id = %d", device_id);
  rtBindHostpidInfo_t pid_info{};
  pid_info.cpType = RT_DEV_PROCESS_QS;
  pid_info.hostPid = mmGetPid();
  pid_info.chipId = device_id;
  GE_CHK_RT_RET(rtQueryDevPid(&pid_info, &pid));
  GELOGI("Start flowgw success, device_id = %d, pid = %d.", device_id, pid);
  return SUCCESS;
}

Status TsdClient::ForkSubprocess(int32_t device_id,
                                 const SubprocessManager::SubprocessConfig &subprocess_config,
                                 const std::string &file_path,
                                 pid_t &pid) {
  GELOGI("Fork process begin, process_type = %s, device_id = %d.",
         subprocess_config.process_type.c_str(), device_id);
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  auto proc = reinterpret_cast<TsdProcessOpen>(mmDlsym(handle_, kFuncNameTsdProcessOpen));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdProcessOpen, mmDlerror());

  SubprocessManager::SubprocessConfig config = subprocess_config; // copy
  // need send back log
  constexpr int32_t kLogSaveMode = 2;
  config.envs.emplace("ASCEND_LOG_SAVE_MODE", std::to_string(kLogSaveMode));
  uint32_t host_pid = 0U;
  GE_CHK_RT_RET(rtDeviceGetBareTgid(&host_pid));
  config.envs.emplace("ASCEND_HOSTPID", std::to_string(host_pid));

  std::vector<ProcEnvParam> env_list;
  std::vector<ProcExtParam> arg_list;
  ProcOpenArgs args = {};
  args.subPid = &pid;

  for (const auto &it : config.envs) {
    ProcEnvParam param = {};
    param.envName =  it.first.c_str();
    param.nameLen = it.first.length();
    param.envValue = it.second.c_str();
    param.valueLen = it.second.length();
    GELOGI("Param env:%s=%s", it.first.c_str(), it.second.c_str());
    env_list.emplace_back(param);
  }
  args.envParaList = &env_list[0];
  args.envCnt = env_list.size();

  std::vector<std::string> args_strings = SubprocessManager::FormatArgs(config);
  for (const auto &arg : args_strings) {
    ProcExtParam param = {};
    param.paramInfo = arg.c_str();
    param.paramLen = arg.length();
    GELOGI("Param arg:%s", arg.c_str());
    arg_list.emplace_back(param);
  }
  args.extParamList = &arg_list[0];
  args.extParamCnt = arg_list.size();
  args.procType = TransferProcType(config.process_type);
  if (!file_path.empty()) {
    args.filePath = file_path.c_str();
    args.pathLen = file_path.length();
  }
  GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id), &args),
                    "Failed to open subprocess, device_id = %d, type = %d",
                    device_id, config.process_type.c_str());
  GELOGI("Fork process success, process_type = %s, device_id = %d.",
         subprocess_config.process_type.c_str(), device_id);
  return SUCCESS;
}

Status TsdClient::ForkSubprocess(int32_t device_id,
                                 const SubprocessManager::SubprocessConfig &subprocess_config,
                                 pid_t &pid) {
  GE_CHK_STATUS_RET(SetDlogReportStart(device_id), "Failed to dlog report start, device_id = %d,", device_id);
  std::string empty_path;
  GE_CHK_STATUS_RET(ForkSubprocess(device_id, subprocess_config, empty_path, pid),
                    "Failed to open subprocess, device_id = %d, type = %s",
                    device_id, subprocess_config.process_type.c_str());
  return SUCCESS;
}

Status TsdClient::ShutdownSubprocess(int32_t device_id, pid_t pid, const std::string &proc_type) {
  GELOGI("Shutdown process begin, pid = %d, device_id = %d.", pid, device_id);
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  auto proc = reinterpret_cast<ProcessCloseSubProcList>(mmDlsym(handle_, kFuncNameTsdProcessCloseSubProcList));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdProcessCloseSubProcList, mmDlerror());
  ProcStatusParam status = {pid, TransferProcType(proc_type), SUB_PROCESS_STATUS_NORMAL};
  GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id), &status, 1U),
                    "Failed to shutdown subprocess, device_id = %d, pid = %d", device_id, pid);
  GELOGI("Shutdown process success, pid = %d, device_id = %d.", pid, device_id);
  return SUCCESS;
}

Status TsdClient::LoadFile(int32_t device_id, const std::string &file_path, const std::string &file_name) {
  GELOGI("Load file to device begin, path = %s, name = %s, device_id = %d.",
         file_path.c_str(), file_name.c_str(), device_id);
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  GE_CHK_STATUS_RET_NOLOG(LoadFileByTsd(device_id, file_path.c_str(), file_path.length(), file_name));
  GELOGI("Load file to device success, path = %s, name = %s, device_id = %d.",
         file_path.c_str(), file_name.c_str(), device_id);
  return SUCCESS;
}

Status TsdClient::LoadFileByTsd(int32_t device_id, const char_t *const file_path, const size_t path_len,
                                const std::string &file_name) const {
  auto proc = reinterpret_cast<TsdFileLoad>(mmDlsym(handle_, kFuncNameTsdFileLoad));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED, "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdFileLoad, mmDlerror());
  GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id), file_path, path_len, file_name.c_str(), file_name.length()),
                    "Failed to load file, device_id = %d, file path = %s, file name = %s.", device_id,
                    (file_path == nullptr ? "" : file_path), file_name.c_str());
  return SUCCESS;
}

Status TsdClient::UnloadFile(int32_t device_id, const std::string &file_path) {
  GELOGI("Unload file begin, path = %s, device_id = %d.", file_path.c_str(), device_id);
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  // prevent concurrent in the same device.
  GE_CHK_STATUS_RET(SetDevice(device_id), "Failed to set device");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  auto proc = reinterpret_cast<TsdFileUnLoad>(mmDlsym(handle_, kFuncNameTsdFileUnLoad));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED,
                         "[Dlsym][So] failed, so name = %s, func name = %s, error_msg = %s",
                         kTsdClientLibName, kFuncNameTsdFileUnLoad, mmDlerror());
  GE_CHK_STATUS_RET(proc(static_cast<uint32_t>(device_id),
                         file_path.c_str(),
                         file_path.length()),
                    "Failed to unload file, device_id = %d, file path = %s.",
                    device_id, file_path.c_str());
  GELOGI("Unload file success, path = %s, device_id = %d.", file_path.c_str(), device_id);
  return SUCCESS;
}

Status TsdClient::CheckSupportInnerPackUnpack(int32_t device_id, bool &is_support) {
  GE_CHK_STATUS_RET_NOLOG(
      CheckCapabilitySupport(device_id, static_cast<int32_t>(TSD_CAPABILITY_OM_INNER_DEC), is_support));
  GELOGI("[Check][Support] inner pack unpack, ret:%u", static_cast<uint32_t>(is_support));
  return SUCCESS;
}

Status TsdClient::CheckSupportBuiltinUdfLaunch(int32_t device_id, bool &is_support) {
  GE_CHK_STATUS_RET_NOLOG(
      CheckCapabilitySupport(device_id, static_cast<int32_t>(TSD_CAPABILITY_BUILTIN_UDF), is_support));
  GELOGI("[Check][Support] builtin udf launch, ret:%u", static_cast<uint32_t>(is_support));
  return SUCCESS;
}

Status TsdClient::CheckCapabilitySupport(int32_t device_id, int32_t capability, bool &is_support, uint64_t required) {
  GE_CHK_STATUS_RET(Initialize(), "Failed to init tsd client");
  std::unique_lock<std::mutex> guard(GetDeviceMutex(device_id));
  uint64_t value = 0UL;
  uint64_t ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&value));
  GE_CHK_STATUS_RET(tsd_capability_get_(device_id, capability, ptr), "tsd capability get faild, type=%d, ptr=%lu.",
                    capability, ptr);
  GELOGI("Tsd capability get success, type=%d, value=%lu, required value=%lu.", capability, value, required);
  is_support = (value >= required);
  return SUCCESS;
}

void TsdClient::Finalize() {
  GEEVENT("Finalize tsd client begin.");
  g_load_file_count.store(0UL);
  {
    std::unique_lock<std::mutex> guard(map_mutex_);
    ThreadPool pool("ge_dpl_rdev", set_device_list_.size(), false);
    std::vector<std::future<Status>> fut_rets;
    for (auto device_id : set_device_list_) {
      auto fut = pool.commit([device_id]() -> Status {
        GE_CHK_RT_RET(rtDeviceReset(device_id));
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    for (auto &fut : fut_rets) {
      GE_CHK_STATUS(fut.get(), "Failed to reset device");
    }
    set_device_list_.clear();
    device_mutexs_.clear();
  }
  if (handle_ != nullptr) {
    (void) mmDlclose(handle_);
    tsd_capability_get_ = nullptr;
    handle_ = nullptr;
  }
  inited_ = false;
  SetDlogReportStop();
  GEEVENT("Finalize tsd client successfully.");
}
}  // namespace ge
