/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "command_handle.h"

#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/load/model_manager/model_manager.h"
#include "aprof_pub.h"
#include "graph/ge_context.h"

namespace ge {
namespace {
constexpr size_t kDeviceListIndex = 3U;
constexpr uint32_t kCommandNum = 6U;
constexpr uint32_t kMaxDevNum = 64U;
const std::string kDeviceNums = "devNums";
const std::string kDeviceIdList = "devIdList";
const std::string kProfilingInit = "prof_init";
const std::string kProfilingFinalize = "prof_finalize";
const std::string kProfilingStart = "prof_start";
const std::string kProfilingStop = "prof_stop";
const std::string kProfilingModelSubscribe = "prof_model_subscribe";
const std::string kProfilingModelUnsubscribe = "prof_model_cancel_subscribe";
const std::string kProfilingModelId = "modelId";
constexpr int32_t RT_ERROR = -1;

enum class ProfCommandHandleType : uint32_t {
  kProfCommandHandleInit = 0,
  kProfCommandHandleStart,
  kProfCommandHandleStop,
  kProfCommandHandleFinalize,
  kProfCommandHandleModelSubscribe,
  kProfCommandHandleModelUnsubscribe
};

const std::map<ProfCommandHandleType, std::string> kProfCommandTypeMap {
    {ProfCommandHandleType::kProfCommandHandleInit, kProfilingInit},
    {ProfCommandHandleType::kProfCommandHandleStart, kProfilingStart},
    {ProfCommandHandleType::kProfCommandHandleStop, kProfilingStop},
    {ProfCommandHandleType::kProfCommandHandleFinalize, kProfilingFinalize},
    {ProfCommandHandleType::kProfCommandHandleModelSubscribe, kProfilingModelSubscribe},
    {ProfCommandHandleType::kProfCommandHandleModelUnsubscribe, kProfilingModelUnsubscribe}
};

bool IsProfConfigValid(const uint32_t deviceid_list[], const uint32_t device_nums) {
  if ((device_nums == 0U) || (device_nums > kMaxDevNum)) {
    GELOGE(PARAM_INVALID, "[Check][DeviceNums]Invalid, device nums: %u", device_nums);
    REPORT_INNER_ERR_MSG("E19999", "DeviceNums %u check invalid", device_nums);
    return false;
  }

  // real device num
  int32_t dev_count = 0;
  const rtError_t rt_err = rtGetDeviceCount(&dev_count);
  if (rt_err != RT_ERROR_NONE) {
    GELOGE(INTERNAL_ERROR, "[Get][DeviceCount]Failed, error_code %d", rt_err);
    REPORT_INNER_ERR_MSG("E19999", "Get device count failed, error_code %d", rt_err);
    return false;
  }

  if (device_nums > static_cast<uint32_t>(dev_count)) {
    GELOGE(PARAM_INVALID, "[Check][Param]Device num %u is not in range [1,%d]", device_nums, dev_count);
    REPORT_INNER_ERR_MSG("E19999", "Device num %u check invalid, it is not in range [1,%d]", device_nums, dev_count);
    return false;
  }

  std::set<uint32_t> record;
  for (uint32_t i = 0U; i < device_nums; ++i) {
    const uint32_t dev_id = deviceid_list[i];
    if (!record.insert(dev_id).second) {
      GELOGE(PARAM_INVALID, "[Check][DeviceId]Device id %u is duplicatedly set", dev_id);
      REPORT_INNER_ERR_MSG("E19999", "Device id %u is not unique, duplicatedly set", dev_id);
      return false;
    }
  }
  return true;
}

bool TransProfConfigToParam(const MsprofCommandHandle &prof_command_handle,
                            std::vector<std::string> &prof_config_params) {
  prof_config_params.clear();
  prof_config_params.emplace_back(kDeviceNums);
  prof_config_params.emplace_back(std::to_string(prof_command_handle.devNums));
  prof_config_params.emplace_back(kDeviceIdList);
  std::string dev_id;
  if (prof_command_handle.devNums == 0U) {
    GELOGE(FAILED, "[Check][Param]The device num is invalid.");
    return false;
  }
  for (uint32_t i = 0U; i < prof_command_handle.devNums; i++) {
    (void)dev_id.append(std::to_string(prof_command_handle.devIdList[i]));
    if (i != (prof_command_handle.devNums - 1U)) {
      (void)dev_id.append(",");
    }
  }

  prof_config_params.push_back(dev_id);
  return true;
}

Status NeedUnsubscribe(const ProfCommandHandleType type, const uint32_t graph_id,
                       std::vector<std::string> &prof_params) {
  if (type == ProfCommandHandleType::kProfCommandHandleModelUnsubscribe) {
    prof_params.clear();
    prof_params.emplace_back(kProfilingModelId);
    auto &prof_mgr = ProfilingManager::Instance();
    if (ProfilingProperties::Instance().GetSubscribeInfo().is_subscribe) {
      uint32_t model_id = 0U;
      const auto ret = prof_mgr.GetModelIdFromGraph(graph_id, model_id);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Get][GraphId]graph_id:%u not not found", graph_id);
        return ret;
      }
      prof_params.emplace_back(std::to_string(model_id));
    } else {
      prof_params.emplace_back(std::to_string(graph_id));
    }
  }

  return SUCCESS;
}

Status NeedHandleStartEnd(const ProfCommandHandleType type, const MsprofCommandHandle &prof_command_handle,
                          std::vector<std::string> &prof_params) {
  if ((type == ProfCommandHandleType::kProfCommandHandleStart) ||
      (type == ProfCommandHandleType::kProfCommandHandleStop)) {
    if (!IsProfConfigValid(prof_command_handle.devIdList, prof_command_handle.devNums)) {
      return FAILED;
    }
    if (!TransProfConfigToParam(prof_command_handle, prof_params)) {
      GELOGE(PARAM_INVALID, "[Check][Param]Transfer profilerConfig to std::string vector failed");
      REPORT_INNER_ERR_MSG("E19999", "Transfer profilerConfig to std::string vector failed");
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

void SubscribeInfoToParam(const ProfCommandHandleType type, const MsprofCommandHandle &prof_command_handle,
                          std::vector<std::string> &prof_params) {
  if (type == ProfCommandHandleType::kProfCommandHandleModelSubscribe) {
    prof_params.clear();
    prof_params.push_back(kProfilingModelId);
    prof_params.push_back(std::to_string(prof_command_handle.modelId));
  }
}

rtError_t ExecuteCommand(const ProfCommandHandleType type,
                         const MsprofCommandHandle &prof_command_handle,
                         const std::vector<std::string> &prof_params) {
  const auto it = kProfCommandTypeMap.find(type);
  if (it == kProfCommandTypeMap.end()) {
    GELOGE(PARAM_INVALID, "[Check][Param]The prof comand type is invalid.");
    return RT_ERROR;
  }

  Command command;
  command.cmd_type = it->second;
  command.cmd_params = prof_params;
  command.cache_flag = prof_command_handle.cacheFlag;

  if (type != ProfCommandHandleType::kProfCommandHandleFinalize) {
    command.module_index = prof_command_handle.profSwitch;
  }
  GELOGI("Command Type: %s, data type config: 0x%" PRIx64, it->second.c_str(), command.module_index);
  if ((type == ProfCommandHandleType::kProfCommandHandleStart) ||
      (type == ProfCommandHandleType::kProfCommandHandleStop)) {
    if (prof_params.size() > kDeviceListIndex) {
      GELOGI("Profiling device nums:%s, deviceId:%s", prof_params[0U].c_str(), prof_params[kDeviceListIndex].c_str());
    } else {
      GELOGW("Profiling input param[size=%zu] may invalid", prof_params.size());
    }
  }

  const Status ret = ModelManager::GetInstance().HandleCommand(command);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Handle][Command]Handle profiling command failed, command type %s, error_code %u",
           it->second.c_str(), ret);
    REPORT_INNER_ERR_MSG("E19999", "Handle profiling command failed, command type %s, error_code %u",
                      it->second.c_str(), ret);
    return RT_ERROR;
  }

  GELOGI("Successfully execute profiling command type: %d, command 0x%" PRIx64 ".",
         static_cast<int32_t>(type), command.module_index);
  return RT_ERROR_NONE;
}

rtError_t HandleCtrlSwitch(const MsprofCommandHandle &prof_command_handle) {
  if (prof_command_handle.type >= kCommandNum) {
    GELOGE(PARAM_INVALID, "[Check][Type]Type %u is invalid", prof_command_handle.type);
    return RT_ERROR;
  }

  GELOGD("Type is %u", prof_command_handle.type);
  std::vector<std::string> prof_params;
  const auto type = static_cast<ProfCommandHandleType>(prof_command_handle.type);
  Status ret = NeedHandleStartEnd(type, prof_command_handle, prof_params);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Handle][Command]Handle command failed, the command type is %d.", static_cast<int32_t>(type));
    return RT_ERROR;
  }
  std::string run_mode;
  if ((type == ProfCommandHandleType::kProfCommandHandleModelSubscribe) &&
      (GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == SUCCESS) && (!run_mode.empty())) {
    GELOGD("Subscribe in training.");
    ProfilingProperties::Instance().SetSubscribeInfo(prof_command_handle.profSwitch, prof_command_handle.modelId, true);
    return RT_ERROR_NONE;
  }
  SubscribeInfoToParam(type, prof_command_handle, prof_params);

  // GraphId is actually stored in prof_command_handle
  const uint32_t graph_id = prof_command_handle.modelId;
  ret = NeedUnsubscribe(type, graph_id, prof_params);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Check][Param]graph_id:%u not not found", graph_id);
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char_t *>({"value", "parameter", "reason"}),
        std::vector<const char_t *>({std::to_string(graph_id).c_str(), "GraphToModelMap", "Graph_id does not exist."}));
    return RT_ERROR;
  }
  return ExecuteCommand(type, prof_command_handle, prof_params);
}

rtError_t HandleCtrlSetStepInfo(const ProfStepInfoCmd_t &prof_set_stepinfo) {
  int32_t device_id = 0;
  const rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(ge::RT_FAILED, "[Get][LogicDeviceId]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Get logic device id failed, ret %d", rt_ret);
    return RT_ERROR;
  }

  auto &prof_mgr = ge::ProfilingManager::Instance();
  const uint64_t step_id = prof_set_stepinfo.index_id;
  const uint16_t tag_id = prof_set_stepinfo.tag_id;
  GELOGI("[Cann Profiling] set step info, step id is %" PRIu64 ", tag id is %u", step_id, static_cast<uint32_t>(tag_id));
  prof_mgr.SetStepInfoIndex(static_cast<int64_t>(step_id));
  const Status ret =
      gert::GlobalProfilingWrapper::ProfileStepTrace(step_id, ge::kInvalidModelId, tag_id, prof_set_stepinfo.stream);
  return ret == SUCCESS ? RT_ERROR_NONE : RT_ERROR;
}
} // namespace

rtError_t ProfCtrlHandle(const uint32_t ctrl_type, void *const ctrl_data, const uint32_t data_len) {
  if ((ctrl_data == nullptr) || (data_len == 0U)) {
    GELOGE(PARAM_INVALID, "[Check][Param]The prof comand is invalid.");
    return RT_ERROR;
  }

  if (ctrl_type == RT_PROF_CTRL_SWITCH) {
    const MsprofCommandHandle *const prof_command_handle = PtrToPtr<void, MsprofCommandHandle>(ctrl_data);
    return HandleCtrlSwitch(*prof_command_handle);
  } else if (ctrl_type == PROF_CTRL_STEPINFO) {
    const ProfStepInfoCmd_t *const prof_command_handle = PtrToPtr<void, ProfStepInfoCmd_t>(ctrl_data);
    return HandleCtrlSetStepInfo(*prof_command_handle);
  } else {
    // nothing to do
  }
  return RT_ERROR;
}
}  // namespace ge
