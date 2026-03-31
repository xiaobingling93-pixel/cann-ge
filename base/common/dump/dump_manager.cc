/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump/dump_manager.h"
#include "common/ge_inner_attrs.h"
#include "common/global_variables/diagnose_switch.h"
#include "graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "mmpa/mmpa_api.h"
#include "adump_api.h"
#include "adump_pub.h"
#include "acl_dump.h"
#include "common/checker.h"
#include <regex>
#include "base/err_msg.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
const std::string kDumpOFF = "OFF";
const std::string kDumpoff = "off";
const std::string kDumpOn = "on";
const std::string kDumpEnable = "1";
const std::string kLiteExceptionDumpEnable = "2";
const std::string kDeviceDumpOn = "device_on";
constexpr uint32_t kAllOverFlow = 3U;
const std::string kReloadDumpFuncName = "Load";
const std::string kUnloadDumpFuncName = "Unload";

// exception dump options
const std::string kDumpExceptionLite = "lite_exception";         // former name of l0 exception dump
const std::string kDumpExceptionBrief = "aic_err_brief_dump";    // l0 exception dump
const std::string kDumpExceptionNorm = "aic_err_norm_dump";      // l1 exception dump
const std::string kDumpExceptionDetail = "aic_err_detail_dump";  // npu coredump

const std::set<std::string> kExceptionDumps = {kDumpExceptionLite,   kDumpExceptionBrief,      kDumpExceptionNorm,
                                               kDumpExceptionDetail, kLiteExceptionDumpEnable, kDumpEnable};

static bool IsExceptionDumpOpen(const ge::DumpConfig &dump_config) {
  return kExceptionDumps.find(dump_config.dump_exception) != kExceptionDumps.end();
}

static bool IsDebugDumpOpen(const ge::DumpConfig &dump_config) {
  return (dump_config.dump_debug == kDumpOn);
}

static bool IsDataDumpOpen(const ge::DumpConfig &dump_config) {
  return (((dump_config.dump_status == kDeviceDumpOn) || (dump_config.dump_status == kDumpOn)) &&
          !IsExceptionDumpOpen(dump_config));
}
}  // namespace

DumpManager &DumpManager::GetInstance() {
  static DumpManager instance;
  return instance;
}

Status DumpManager::Init(const std::map<std::string, std::string> &run_options) {
  const char_t *env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, env);
  if (env == nullptr) {
    MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH_EXE, env);
  }

  if (env != nullptr) {
    npu_collect_path_ = env;
    GELOGI("get npu collect path: %s", npu_collect_path_.c_str());
  }

  Adx::DumpConfig config;
  config.dumpStatus = "on";

  // enable norm exception dump by npu_collect_path
  if (!npu_collect_path_.empty()) {
    config.dumpPath = npu_collect_path_;
    GE_ASSERT_TRUE((Adx::AdumpSetDumpConfig(Adx::DumpType::EXCEPTION, config) == Adx::ADUMP_SUCCESS),
                   "call AdumpSetDumpConfig to enable norm exception dump failed");
    l1_exception_dump_flag_ = true;
    diagnoseSwitch::EnableExceptionDump();
    GELOGI("Enable norm exception dump by env, dump path is %s", npu_collect_path_.c_str());
    return SUCCESS;
  }

  MM_SYS_GET_ENV(MM_ENV_ASCEND_WORK_PATH, env);
  if (env != nullptr) {
    ascend_work_path_ = env;
    GELOGI("get ASCEND_WORK_PATH: %s", ascend_work_path_.c_str());
  }

  // enable exception from options
  std::string enable_exception_dump;
  const auto iter = run_options.find(OPTION_EXEC_ENABLE_EXCEPTION_DUMP);
  if (iter != run_options.end()) {
    enable_exception_dump = iter->second;
    GELOGI("get ge.exec.enable_exception_dump: %s", enable_exception_dump.c_str());
  }

  if (enable_exception_dump == "1") {
    config.dumpPath = ascend_work_path_;
    GE_ASSERT_TRUE((Adx::AdumpSetDumpConfig(Adx::DumpType::EXCEPTION, config) == Adx::ADUMP_SUCCESS),
                   "call AdumpSetDumpConfig to enable norm exception dump failed");
    l1_exception_dump_flag_ = true;
    diagnoseSwitch::EnableExceptionDump();
    GELOGI("Enable norm exception dump by ge option, dump path is %s", ascend_work_path_.c_str());
    return SUCCESS;
  }

  // l0 exception, adump get ASCEND_WORK_PATH from env or use ./
  if (enable_exception_dump == "2") {
    config.dumpPath = ascend_work_path_;
    GE_ASSERT_TRUE((Adx::AdumpSetDumpConfig(Adx::DumpType::ARGS_EXCEPTION, config) == Adx::ADUMP_SUCCESS),
                   "call AdumpSetDumpConfig to enable brief exception dump failed");
    l0_exception_dump_flag_ = true;
    diagnoseSwitch::EnableLiteExceptionDump();
    GELOGI("Enable brief exception dump by ge option, dump path is %s", ascend_work_path_.c_str());
    return SUCCESS;
  }

  GELOGI("exception dump is not enabled");
  return SUCCESS;
}

Status DumpManager::Finalize() {
  l0_exception_dump_flag_ = false;
  l1_exception_dump_flag_ = false;
  npu_collect_path_.clear();
  ascend_work_path_.clear();
  GELOGI("DumpManager Finalize.");
  return SUCCESS;
}

Status DumpManager::RegisterCallBackFunc(const std::string &func,
                                         const std::function<Status(const DumpProperties &)> &callback) {
  GELOGI("DumpManager RegisterCallBackFunc, func = %s.", func.c_str());
  callback_map_[func] = callback;
  return SUCCESS;
}

Status DumpManager::ReloadDumpInfo(const DumpProperties &dump_properties) {
  const auto iter = callback_map_.find(kReloadDumpFuncName);
  if (iter == callback_map_.end()) {
    GELOGW("Can not find Load func.");
    return SUCCESS;
  }
  return iter->second(dump_properties);
}

Status DumpManager::UnloadDumpInfo(const DumpProperties &dump_properties) {
  if ((!dump_properties.IsDumpOpen()) && (!dump_properties.IsOpDebugOpen())) {
    GELOGW("Dump not open before.");
    return SUCCESS;
  }
  const auto iter = callback_map_.find(kUnloadDumpFuncName);
  if (iter == callback_map_.end()) {
    GELOGW("Can not find Unload func.");
    return SUCCESS;
  }
  return iter->second(dump_properties);
}

bool DumpManager::NeedDoDump(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  if (dump_config.dump_status.empty() && dump_config.dump_debug.empty()) {
    GELOGI("Dump does not open");
    return false;
  }
  GELOGI("Dump status: %s, Overflow dump status: %s.",
         dump_config.dump_status.empty() ? "off" : dump_config.dump_status.c_str(),
         dump_config.dump_debug.empty() ? "off" : dump_config.dump_debug.c_str());
  if (((dump_config.dump_status == kDumpoff) || (dump_config.dump_status == kDumpOFF)) &&
      (dump_config.dump_debug == kDumpoff)) {
    // if acl dump open before and unenable this time, do unload task
    const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
    if ((if_acl_dump_set_) && (infer_iter != infer_dump_properties_map_.end())) {
      (void)UnloadDumpInfo(infer_iter->second);
    }
    dump_properties.ClearDumpPropertyValue();
    infer_dump_properties_map_[kInferSessionId] = dump_properties;
    return false;
  }
  if ((dump_config.dump_status == kDumpOn) && (dump_config.dump_debug == kDumpOn)) {
    GELOGW("Not support coexistence of dump debug and dump status.");
    return false;
  }
  return true;
}

void DumpManager::SetDumpDebugConf(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  if (IsDebugDumpOpen(dump_config)) {
    GELOGI("Only do overflow detection, dump debug is %s, dump step is %s.", dump_config.dump_debug.c_str(),
           dump_config.dump_step.c_str());
    dump_properties.InitInferOpDebug();
    dump_properties.SetOpDebugMode(kAllOverFlow);
    dump_properties.SetDumpStep(dump_config.dump_step);
    if_acl_dump_set_ = true;
  }
}

bool DumpManager::CheckHasNpuCollectPath() const {
  char_t *record_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH_EXE, record_path);
  if (record_path == nullptr) {
    MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, record_path);
  }
  if ((record_path != nullptr) && (record_path[0U] != '\0')) {
    return true;
  }
  return false;
}

bool DumpManager::SetExceptionDump(const DumpConfig &dump_config) {
  if ((CheckHasNpuCollectPath()) || (dump_config.dump_exception == kDumpEnable) ||
      (dump_config.dump_exception == kDumpExceptionNorm)) {
    GELOGI("Enable norm exception dump.");
    diagnoseSwitch::EnableExceptionDump();
    return true;
  }

  // for npu coredump mode, ge open l0 exception dump
  if (dump_config.dump_exception == kDumpExceptionDetail) {
    l0_exception_dump_flag_ = true;
    diagnoseSwitch::EnableLiteExceptionDump();
    GELOGI("Enable detail exception dump.");
    return true;
  }

  if ((dump_config.dump_exception == kDumpExceptionBrief) || (dump_config.dump_exception == kDumpExceptionLite) ||
      (dump_config.dump_exception == kLiteExceptionDumpEnable)) {
    l0_exception_dump_flag_ = true;
    GELOGI("Enable brief exception dump.");
    diagnoseSwitch::EnableLiteExceptionDump();
    return true;
  }
  GELOGI("exception dump is not enabled.");
  return false;
}

bool DumpManager::IsValidFormat(const std::string& s) const {
    static const std::regex pattern("^(input|output)\\d+$");
    return std::regex_match(s, pattern);
}

uint32_t DumpManager::ExtractNumber(const std::string& s) const {
    static const std::regex num_pattern("\\d+");
    uint32_t num = 0;
    std::smatch match;
    if (std::regex_search(s, match, num_pattern)) {
      const size_t value = std::stoul(match.str());
      if (value <= (std::numeric_limits<uint32_t>::max())) {
        num = static_cast<uint32_t>(value);
      }
      return num;
    }
    return num;
}

void DumpManager::ExtractBlacklist(const std::vector<DumpBlacklist>& blacklists,
    std::map<std::string, OpBlacklist>& op_blacklist) const {
  for (const auto &blacklist : blacklists) {
    OpBlacklist bl;

    for (const auto &pos : blacklist.pos) {
      if (!IsValidFormat(pos)) {
        GELOGW("Dump blacklist pos %s is invalid", pos.c_str());
        continue;
      }
      if (pos.find("input") == 0) {
        (void)bl.input_indices.insert(ExtractNumber(pos));
      } else if (pos.find("output") == 0) {
        (void)bl.output_indices.insert(ExtractNumber(pos));
      } else {
        // do nothing
      }
    }

    auto& existing = op_blacklist[blacklist.name];
    existing.input_indices.insert(bl.input_indices.begin(), bl.input_indices.end());
    existing.output_indices.insert(bl.output_indices.begin(), bl.output_indices.end());
  }
}

void DumpManager::SetModelDumpBlacklist(ModelOpBlacklist& model_blacklist,
    const std::vector<DumpBlacklist>& type_blacklists, const std::vector<DumpBlacklist>& name_blacklists) const{
    ExtractBlacklist(type_blacklists, model_blacklist.dump_optype_blacklist);
    ExtractBlacklist(name_blacklists, model_blacklist.dump_opname_blacklist);
}


void DumpManager::SetDumpList(const DumpConfig &dump_config, DumpProperties &dump_properties) const {
  for (const auto &model_dump : dump_config.dump_list) {
    const std::string model_name = model_dump.model_name.empty() ? DUMP_LAYER_OP_MODEL : model_dump.model_name;
    GELOGI("Dump model is %s", model_name.c_str());
    std::set<std::string> dump_layers;
    for (const auto &layer : model_dump.layers) {
      GELOGI("Dump layer is %s in model", layer.c_str());
      (void)dump_layers.insert(layer);
    }
    dump_properties.AddPropertyValue(model_name, dump_layers);

    std::set<std::string> watcher_nodes;
    for (const auto &node : model_dump.watcher_nodes) {
      GELOGI("watcher node is %s in model", node.c_str());
      (void)watcher_nodes.insert(node);
    }
    if (!watcher_nodes.empty()) {
      dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, watcher_nodes);
    }

    const std::vector<DumpBlacklist> optype_config = model_dump.optype_blacklist;
    const std::vector<DumpBlacklist> opname_config = model_dump.opname_blacklist;
    ModelOpBlacklist model_blacklist;
    SetModelDumpBlacklist(model_blacklist, optype_config, opname_config);
    dump_properties.SetModelBlacklist(model_name, model_blacklist);
    if (!model_dump.dump_op_ranges.empty()) {
      dump_properties.SetOpDumpRange(model_name, model_dump.dump_op_ranges);
    }
  }
}

Status DumpManager::SetNormalDumpConf(const DumpConfig &dump_config, DumpProperties &dump_properties) {
  // device dumper only support dump all model
  std::string dump_status = dump_config.dump_status;
  if (dump_status == kDeviceDumpOn) {
    dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
    dump_status = kDumpOn;
  }

  if ((dump_status == kDumpOn) && !IsExceptionDumpOpen(dump_config)) {
    GELOGI("Only do normal dump process, dump status is %s", dump_config.dump_status.c_str());
    dump_properties.SetDumpStatus(dump_status);
    const std::string dump_op_switch = dump_config.dump_op_switch;
    dump_properties.SetDumpOpSwitch(dump_op_switch);
    if ((dump_op_switch == kDumpoff) && (dump_config.dump_list.empty())) {
      (void)infer_dump_properties_map_.emplace(kInferSessionId, dump_properties);
      GELOGE(PARAM_INVALID, "[Check][DumpList]Invalid, dump_op_switch is %s", dump_op_switch.c_str());
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({"dump_list", "", "Dump list is empty."}));
      return PARAM_INVALID;
    }

    if (!dump_config.dump_list.empty()) {
      if (dump_op_switch == kDumpOn) {
        GELOGI("Start to dump model and single op, dump op switch is %s", dump_op_switch.c_str());
      } else {
        GELOGI("Only dump model, dump op switch is %s", dump_op_switch.c_str());
      }
      SetDumpList(dump_config, dump_properties);
    } else {
      GELOGI("Only dump single op, dump op switch is %s", dump_op_switch.c_str());
      dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
    }
    GELOGI("Dump mode is %s", dump_config.dump_mode.c_str());
    dump_properties.SetDumpMode(dump_config.dump_mode);
    GELOGI("Dump step is %s", dump_config.dump_step.c_str());
    dump_properties.SetDumpStep(dump_config.dump_step);

    if (!dump_config.dump_data.empty()) {
      if ((dump_config.dump_data == "tensor") || (dump_config.dump_data == "stats")) {
        GELOGI("Dump data is %s", dump_config.dump_data.c_str());
        dump_properties.SetDumpData(dump_config.dump_data);
      } else {
        GELOGE(PARAM_INVALID, "[Check][DumpData]Invalid, dump_data is %s", dump_config.dump_data.c_str());
        (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                  std::vector<const ge::char_t *>({"--dump_data", dump_config.dump_data.c_str(),
                                                                   "The value must be tensor or stats."}));
        return PARAM_INVALID;
      }
    }

    if_acl_dump_set_ = true;
  }
  return SUCCESS;
}

Status DumpManager::SetDumpPath(const DumpConfig &dump_config, DumpProperties &dump_properties) const {
  std::string dump_path = dump_config.dump_path;
  if (dump_path.empty()) {
    GELOGE(PARAM_INVALID, "[Check][DumpPath]It is empty.");
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"dump_path", "", "Dump path is empty."}));
    return PARAM_INVALID;
  }
  if (dump_path[dump_path.size() - 1U] != '/') {
    dump_path = dump_path + "/";
  }
  dump_path = dump_path + CurrentTimeInStr() + "/";
  const char *adump_dump_path = acldumpGetPath(DATA_DUMP);
  if (adump_dump_path != nullptr) {
    std::string adump_dump_path_str(adump_dump_path);
    if (!adump_dump_path_str.empty()) {
      GELOGI("acldumpGetPath is not empty, use adump_dump_path=%s instead of %s", adump_dump_path_str.c_str(),
             dump_path.c_str());
      dump_path = adump_dump_path_str;
    }
  }
  GELOGI("Dump path is %s", dump_path.c_str());
  dump_properties.SetDumpPath(dump_path);
  return SUCCESS;
}

Status DumpManager::SetDumpConf(const DumpConfig &dump_config) {
  DumpProperties dump_properties;
  if (!NeedDoDump(dump_config, dump_properties)) {
    diagnoseSwitch::DisableDumper();
    GELOGD("No need do dump process.");
    return SUCCESS;
  }
  const bool is_acl_open_before = if_acl_dump_set_;
  const bool is_exception_dump_open = SetExceptionDump(dump_config);
  SetDumpDebugConf(dump_config, dump_properties);
  GE_CHK_STATUS_RET(SetNormalDumpConf(dump_config, dump_properties), "[Init][DumpConf] failed when dump status is on.");
  if (!is_exception_dump_open) {
    GE_CHK_STATUS_RET(SetDumpPath(dump_config, dump_properties), "[Init][DumpPath] failed.");
    infer_dump_properties_map_[kInferSessionId] = dump_properties;
  } else if (CheckHasNpuCollectPath() && (!IsExceptionDumpOpen(dump_config))) {
    GELOGE(PARAM_INVALID, "It's not support to open Data dump and open L1 exception dump by env at once.");
    REPORT_INNER_ERR_MSG("E19999", "It's not support to open Data dump and open L1 exception dump by env at once.");
    return PARAM_INVALID;
  }
  if ((is_acl_open_before) && (dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen())) {
    (void)ReloadDumpInfo(dump_properties);
  }
  if ((!is_acl_open_before) && (if_acl_dump_set_) && (!dump_properties_map_.empty())) {
    // acl set dump this time and has dump setting by options
    GELOGW("Set dump by options and acl simultaneously, will use the acl setting.");
  }
  dump_flag_ = true;
  if (IsDataDumpOpen(dump_config)) {
    diagnoseSwitch::EnableDataDump();
  } else if (IsDebugDumpOpen(dump_config)) {
    diagnoseSwitch::EnableOverflowDump();
  }
  return SUCCESS;
}

DumpProperties DumpManager::GetDumpProperties(const uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  // offline infer set session id for different model, but dump properties just on kInferSessionId
  const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
  if (infer_iter != infer_dump_properties_map_.end()) {
    return infer_iter->second;
  }
  const auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    return iter->second;
  }
  static DumpProperties default_properties;
  return default_properties;
}

Status DumpManager::AddDumpProperties(uint64_t session_id, const DumpProperties &dump_properties) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (dump_properties.IsDumpOpen()) {
    Adx::DumpConfig config;
    config.dumpStatus = "on";
    config.dumpPath = dump_properties.GetDumpPath();
    config.dumpMode = dump_properties.GetDumpMode();
    config.dumpData = dump_properties.GetDumpData();
    config.dumpSwitch = Adx::OPERATOR_OP_DUMP;
    GELOGI("Add dump properties, session id:%lu, dumpPath:%s, dumpMode: %s, dumpData: %s", session_id,
           config.dumpPath.c_str(), config.dumpMode.c_str(), config.dumpData.c_str());
    GE_ASSERT_TRUE((Adx::AdumpSetDumpConfig(Adx::DumpType::OPERATOR, config) == Adx::ADUMP_SUCCESS),
                   "call AdumpSetDumpConfig to enable data dump failed");
  } else if (dump_properties.IsOpDebugOpen()) {
    Adx::DumpConfig config;
    config.dumpStatus = "on";
    config.dumpPath = dump_properties.GetDumpPath();
    config.dumpMode = dump_properties.GetDumpMode();
    config.dumpData = dump_properties.GetDumpData();
    config.dumpSwitch = Adx::OPERATOR_OP_DUMP;
    GELOGI("Add overflow dump properties, session id:%lu, dumpPath:%s, dumpMode: %s, dumpData: %s", session_id,
           config.dumpPath.c_str(), config.dumpMode.c_str(), config.dumpData.c_str());
    GE_ASSERT_TRUE((Adx::AdumpSetDumpConfig(Adx::DumpType::OP_OVERFLOW, config) == Adx::ADUMP_SUCCESS),
                   "call AdumpSetDumpConfig to enable overflow dump failed");
  }
  (void)dump_properties_map_.emplace(session_id, dump_properties);
  GELOGI("Add dump properties, session id:%lu.", session_id);
  return SUCCESS;
}

void DumpManager::RemoveDumpProperties(uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    (void)dump_properties_map_.erase(iter);
    GELOGI("Remove dump properties, session id:%lu.", session_id);
  }

  const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
  if (infer_iter != infer_dump_properties_map_.end()) {
    (void)infer_dump_properties_map_.erase(infer_iter);
    if_acl_dump_set_ = false;
    GELOGI("Remove dump properties set by acl, session id:%lu", session_id);
  }

  if (dump_properties_map_.empty() && (Adx::AdumpGetDumpSwitch(Adx::DumpType::OPERATOR)) != 0) {
    Adx::DumpConfig config;
    config.dumpStatus = "off";
    const auto adx_ret = Adx::AdumpSetDumpConfig(Adx::DumpType::OPERATOR, config);
    GELOGI("call AdumpSetDumpConfig to disable data dump, adx errorCode = %d", adx_ret);
  }

  if (dump_properties_map_.empty() && (Adx::AdumpGetDumpSwitch(Adx::DumpType::OP_OVERFLOW)) != 0) {
    Adx::DumpConfig config;
    config.dumpStatus = "off";
    config.dumpSwitch = 0;
    const auto adx_ret = Adx::AdumpSetDumpConfig(Adx::DumpType::OP_OVERFLOW, config);
    GELOGI("call AdumpSetDumpConfig to disable overflow dump, adx errorCode = %d", adx_ret);
  }
}

bool DumpManager::GetCfgFromOption(const std::map<std::string, std::string> &options_all, DumpConfig &dump_cfg) {
  auto options = options_all;
  dump_cfg.dump_mode = options[OPTION_EXEC_DUMP_MODE];
  const std::string enable_flag = options[OPTION_EXEC_ENABLE_DUMP];
  const std::string dump_path = options[OPTION_EXEC_DUMP_PATH];
  if (enable_flag != kDumpEnable) {
    dump_cfg.dump_status = kDumpoff;
    dump_cfg.dump_debug = kDumpoff;
    return false;
  }
  // transfer from enable dump to dump status
  dump_cfg.dump_status = kDeviceDumpOn;
  dump_cfg.dump_debug = kDumpoff;
  dump_cfg.dump_step = options[OPTION_EXEC_DUMP_STEP];
  std::string host_pid_name = "unknown_pid";
  std::string executor_dev_id = "0";
  // pid
  if (options.find(kHostMasterPidName) != options.end()) {
    host_pid_name = options[kHostMasterPidName];
  }
  // device id
  if (options.find(kExecutorDevId) != options.end()) {
    executor_dev_id = options[kExecutorDevId];
  }
  const auto file_path = dump_path.empty() ? "/var/log/npu/dump" : dump_path;
  dump_cfg.dump_path = file_path + "/pid" + host_pid_name + "/device" + executor_dev_id + "/";
  GELOGD("Get dump config: dump_mode[%s], dump_status[%s], dump_debug[%s], dump_path[%s], dump_step[%s]",
         dump_cfg.dump_mode.c_str(), dump_cfg.dump_status.c_str(), dump_cfg.dump_debug.c_str(),
         dump_cfg.dump_path.c_str(), dump_cfg.dump_step.c_str());
  return true;
}

void DumpManager::SetDumpWorkerId(const uint64_t session_id, const std::string &worker_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = dump_properties_map_.find(session_id);
  if (iter != dump_properties_map_.end()) {
    iter->second.SetDumpWorkerId(worker_id);
  } else {
    const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
    if (infer_iter != infer_dump_properties_map_.end()) {
      infer_iter->second.SetDumpWorkerId(worker_id);
    }
  }
}

void DumpManager::SetDumpOpSwitch(const uint64_t session_id, const std::string &op_switch) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
  if (infer_iter != infer_dump_properties_map_.end()) {
    infer_iter->second.SetDumpOpSwitch(op_switch);
  } else {
    const auto iter = dump_properties_map_.find(session_id);
    if (iter != dump_properties_map_.end()) {
      iter->second.SetDumpOpSwitch(op_switch);
    }
  }
}

void DumpManager::ClearAclDumpSet() {
  const std::lock_guard<std::mutex> lock(mutex_);
  GELOGI("Begin to clear acl dump set.");
  const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
  if (infer_iter != infer_dump_properties_map_.end()) {
    (void)infer_dump_properties_map_.erase(infer_iter);
    if_acl_dump_set_ = false;
  }
}

bool DumpManager::CheckIfAclDumpSet() {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto infer_iter = infer_dump_properties_map_.find(kInferSessionId);
  if (infer_iter != infer_dump_properties_map_.end()) {
    return true;
  }
  return false;
}
}  // namespace ge
