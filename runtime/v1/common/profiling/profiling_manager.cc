/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/profiling/profiling_manager.h"

#include <securec.h>
#include <algorithm>
#include <thread>

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/profiling_definitions.h"
#include "graph/utils/type_utils.h"
#include "runtime/dev.h"
#include "runtime/rts/rts_stream.h"
#include "runtime/rts/rts_kernel.h"
#include "mmpa/mmpa_api.h"
#include "graph/load/model_manager/model_manager.h"
#include "common/global_variables/diagnose_switch.h"

namespace {
constexpr int32_t kMaxDeviceNum = 256;
constexpr uint16_t kStartTag = 0U;
constexpr uint16_t kEndTag = 1U;
const std::string kConfigNumsdev = "devNums";
const std::string kConfigDevIdList = "devIdList";
const std::string kProfStart = "prof_start";
const std::string kProfStop = "prof_stop";
const std::string kProfModelSubscribe = "prof_model_subscribe";
const std::string kProfModelUnsubscribe = "prof_model_cancel_subscribe";
const std::string kInput = "input";
const std::string kOutput = "output";
const std::string kDynShapeOpExecuteData = "dynamic_op_execute";
const std::string kTensorTagName = "tensor_data_info";
const std::string kSingleOpTensorTagName = "single_op_tensor_info";
const std::string kTaskTagName = "task_desc_info";
const std::string kSingleOpTaskTagName = "single_op_task_info";
const std::string kHeterogeneousHost = "heterogeneous_host";


template <typename T>
void SetAlternativeValue(const int32_t property_len, const std::string &value, T &property) {
  if (value.size() < static_cast<size_t>(property_len)) {
    property.type = static_cast<uint8_t>(MSPROF_MIX_DATA_STRING);
    const auto ret = strncpy_s(property.data.dataStr, static_cast<size_t>(property_len), value.c_str(), value.size());
    if (ret != 0) {
      GELOGW("[Profiling] strncpy_s value [%s] error!", value.c_str());
    }
  } else {
    property.type = static_cast<uint8_t>(MSPROF_MIX_DATA_HASH_ID);
    property.data.hashId = MsprofGetHashId(value.c_str(), value.length());
  }
}
}  // namespace


namespace ge {
ProfilingManager::ProfilingManager() = default;

ProfilingManager &ProfilingManager::Instance() {
  static ProfilingManager profiling_manager;
  return profiling_manager;
}

void ProfilingManager::RegisterElement(int64_t &idx, const std::string &element) {
  if (ProfilingManager::Instance().ProfilingModelLoadOn() &&
      ProfilingProperties::Instance().IsTaskEventProfiling()) {
    const uint64_t hash_id = MsprofGetHashId(element.c_str(), element.length());
    idx = profiling::ProfilingContext::GetInstance().RegisterStringHash(hash_id, element);
  } else {
    idx = profiling::ProfilingContext::GetInstance().RegisterString(element);
  }
}

Status ProfilingManager::CheckInitForSubscribe(const uint64_t module, const uint32_t device, const uint32_t model_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  diagnoseSwitch::EnableDeviceProfiling();
  ProfilingProperties::Instance().MutableSubscribeCount()++;
  ProfilingProperties::Instance().InsertSubscribeGraphId(model_id);
  ProfilingProperties::Instance().UpdateSubscribeDeviceModuleMap(kProfModelSubscribe, device, module);
  return SUCCESS;
}

Status ProfilingManager::ProfModelUnsubscribe(const uint32_t device, const uint32_t model_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!ProfilingProperties::Instance().ProfilingSubscribeOn()) {
    GELOGW("The profiler has not been subscribed, you do not need to cannel the subscription.");
    return SUCCESS;
  }

  const auto &subs_dev_module = ProfilingProperties::Instance().GetDeviceSubInfo();
  const std::map<uint32_t, DeviceSubsInfo>::const_iterator iter = subs_dev_module.find(device);
  if (iter != subs_dev_module.cend()) {
    ProfilingProperties::Instance().UpdateSubscribeDeviceModuleMap(kProfModelUnsubscribe, device,
                                                                   iter->second.module);
  } else {
    GELOGE(FAILED, "[Cancel][DeviceId]The device_id %u has not been subscribed, do not need to cancel", device);
    REPORT_INNER_ERR_MSG("E19999", "The device_id %u has not been subscribed, do not need to cancel", device);
    return FAILED;
  }

  ProfilingProperties::Instance().MutableSubscribeCount()--;
  ProfilingProperties::Instance().RemoveSubscribeGraphId(model_id);

  ProfilingProperties::Instance().CleanSubscribeInfo();
  // reported_graph_id_ needs to be cleared on unsubscribe
  reported_graph_id_.clear();
  return SUCCESS;
}

Status ProfilingManager::ProfInit(const uint64_t module) {
  const std::lock_guard<std::mutex> lock(mutex_);

  ProfilingProperties::Instance().SetIfInited(true);
  GELOGI("ProfilingProperties is inited.");

  const uint64_t training_trace_mask = module & static_cast<uint64_t>(PROF_TRAINING_TRACE_MASK);
  if (training_trace_mask == static_cast<uint64_t>(PROF_TRAINING_TRACE_MASK)) {
    ProfilingProperties::Instance().SetTrainingTrace(true);
  }

  GELOGI("Prof init success.");
  return SUCCESS;
}

Status ProfilingManager::ProfFinalize() {
  const std::lock_guard<std::mutex> lock(mutex_);
  index_id_ = std::numeric_limits<int64_t>::max();

  ProfilingProperties::Instance().ClearProperties();
  ProfilingProperties::Instance().CleanSubscribeInfo();
  ProfilingProperties::Instance().ClearDeviceIdMap();
  device_id_.clear();
  device_id_map_.clear();
  model_id_map_.clear();
  reported_graph_id_.clear();
  GELOGI("Prof finalize success.");
  return SUCCESS;
}

Status ProfilingManager::ProfParseDeviceId(const std::map<std::string, std::string> &config_para,
                                           std::vector<int32_t> &device_list) const {
  const auto iter = config_para.find(kConfigDevIdList);
  if (iter != config_para.end()) {
    const std::string device_id_list = iter->second;
    std::string temp;
    std::vector<std::string> decvice_id;
    for (size_t i = 0U; i < device_id_list.size(); i++) {
      if (std::isdigit(static_cast<unsigned char>(device_id_list[i])) != 0) {
        (void)temp.append(1U, device_id_list[i]);
      } else {
        if (!temp.empty()) {
          decvice_id.emplace_back(temp);
        }
        temp.clear();
      }
    }
    if (!temp.empty()) {
      decvice_id.emplace_back(temp);
    }

    for (size_t i = 0U; i < decvice_id.size(); i++) {
      int32_t dev_id = -1;
      GE_CHK_STATUS_RET_NOLOG(ConvertToInt32(decvice_id[i], dev_id));
      device_list.push_back(dev_id);
    }
  } else {
    GELOGE(FAILED, "[Parse][DeviceId]Config para not contain device id list");
    REPORT_INNER_ERR_MSG("E19999", "Parse device id failed, config para not contain device id list");
    return FAILED;
  }
  return SUCCESS;
}

Status ProfilingManager::ProfParseParam(const std::map<std::string, std::string> &config_para, int32_t &device_num,
                                        std::vector<int32_t> &device_list) const {
  const auto device_num_iter = config_para.find(kHeterogeneousHost);
  if ((device_num_iter != config_para.end()) && (device_num_iter->second == "1")) {
    GELOGI("[Parse][DeviceId]Config para explicitly dose not include device_num.");
    return SUCCESS;
  }
  // device num
  const auto iter = config_para.find(kConfigNumsdev);
  if (iter == config_para.end()) {
    GELOGE(FAILED, "[Parse][Param]Config para not contain device num");
    REPORT_INNER_ERR_MSG("E19999", "Parse param failed, config para not contain device num");
    return FAILED;
  }
  GE_CHK_STATUS_RET_NOLOG(ConvertToInt32(iter->second, device_num));

  // device id
  if (ProfParseDeviceId(config_para, device_list) != SUCCESS) {
    GELOGE(FAILED, "[Parse][DeviceId]Failed");
    REPORT_INNER_ERR_MSG("E19999", "Parse device id failed");
    return FAILED;
  }

  if ((device_num == 0) || (device_num > kMaxDeviceNum) || (device_num != static_cast<int32_t>(device_list.size()))) {
    GELOGE(FAILED, "[Parse][Param]Failed, config para device num %d not equal to "
           "device list size %zu", device_num, device_list.size());
    REPORT_INNER_ERR_MSG("E19999", "[Parse][Param]Failed, config para device num %d "
                       "not equal to device list size %zu", device_num, device_list.size());
    return FAILED;
  }
  return SUCCESS;
}

void ProfilingManager::ProfOpDetailProfiling(const uint64_t module, const uint32_t cache_flag) const {
  const uint64_t op_detail_mask = module & (static_cast<uint64_t>(PROF_OP_DETAIL_MASK));
  if (op_detail_mask == static_cast<uint64_t>(PROF_OP_DETAIL_MASK) &&
      !ProfilingProperties::Instance().IsOpDetailProfiling()) {
    ProfilingProperties::Instance().SetOpDetailProfiling(true);
    GELOGI("Prof init: op detail profiling on, cache_flag=%u.", cache_flag);
    for (const auto &model_id : loaded_model_id_) {
      const auto &davinci_model = ModelManager::GetInstance().GetModel(model_id);
      if (davinci_model == nullptr) {
        continue;
      }
      if (davinci_model->NeedClearDfxCacheFlagAfterInit()) {
        GELOGW("Can not report profiling data due to cache is clear, model id is %u", model_id);
        continue;
      }
      (void)davinci_model->ReportProfilingData();
      // cache_flag is enable
      if (cache_flag == 1U) {
        davinci_model->ClearProfilingDataCache();
      }
    }
  }
}

Status ProfilingManager::ProfStartProfiling(const uint64_t module,
                                            const std::map<std::string, std::string> &config_para,
                                            const uint32_t cache_flag) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto IsEnabled = [&module](uint64_t prof_type) { return (module & prof_type) == prof_type; };

  if (IsEnabled(PROF_MODEL_LOAD_MASK)) {
    ProfilingProperties::Instance().SetLoadProfiling(true);
    GELOGI("Prof init: model load profiling on.");
  }

  if (IsEnabled(PROF_TASK_TIME_MASK)) {
    diagnoseSwitch::EnableTaskTimeProfiling();
    GELOGI("Prof init: task time profiling on.");
  }

  if (IsEnabled(PROF_TASK_TIME_L1_MASK)) {
    diagnoseSwitch::EnableDeviceProfiling();
    GELOGI("Prof init: task time l1 profiling on.");
  }

  if (IsEnabled(PROF_TASK_MEMORY_MASK)) {
    diagnoseSwitch::EnableMemoryProfiling();
    GELOGI("Prof init: task memory profiling on.");
  }

  if (IsEnabled(PROF_TRAINING_TRACE_MASK)) {
    ProfilingProperties::Instance().SetTrainingTrace(true);
    diagnoseSwitch::EnableTrainingTrace();
    GELOGI("Prof init: training trace profiling on.");
  }

  // profiling op detail on
  ProfOpDetailProfiling(module, cache_flag);
  gert::GlobalProfilingWrapper::GetInstance()->IncreaseProfCount();

  if (IsEnabled(PROF_TASK_FRAMEWORK_MASK)) {
    ProfilingProperties::Instance().SetTaskEventProfiling(true);
    diagnoseSwitch::EnableCannHostProfiling();
    profiling::ProfilingContext::GetInstance().UpdateElementHashId();
    GELOGI("Prof init: fwk schedule l0 profiling on.");
  }

  if (IsEnabled(PROF_FWK_SCHEDULE_L0_MASK)) {
    diagnoseSwitch::EnableCannHostProfiling();
    GELOGI("Prof init: fwk schedule l0 profiling on.");
  }

  if (IsEnabled(PROF_FWK_SCHEDULE_L1_MASK)) {
    diagnoseSwitch::EnableProfiling({gert::ProfilingType::kCannHost, gert::ProfilingType::kCannHostL1});
    GELOGI("Prof init: fwk schedule l1 profiling on.");
  }

  int32_t device_num = 0;
  std::vector<int32_t> device_list;
  if (ProfParseParam(config_para, device_num, device_list) != SUCCESS) {
    GELOGE(FAILED, "[Parse][Param]Prof start parse param failed, device num %d, "
           "device list size %zu", device_num, device_list.size());
    REPORT_INNER_ERR_MSG("E19999", "Prof start parse param failed, device num %d, "
                      "device list size %zu", device_num, device_list.size());
    return FAILED;
  }

  if (IsEnabled(PROF_MODEL_EXECUTE_MASK)) {
    for (size_t i = 0U; i < static_cast<size_t>(device_num); i++) {
      if (std::find(device_id_.begin(), device_id_.end(), device_list[i]) == device_id_.end()) {
        device_id_.push_back(device_list[i]);
      }
    }
    GELOGI("Prof start: ge execute model start profiling.");
  }

  if (IsEnabled(PROF_MODEL_LOAD_MASK)) {
    GELOGW("Prof start: load model module is invalid.");
  }

  GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::GetInstance()->RegisterProfType());
  ProfilingProperties::Instance().UpdateDeviceIdModuleMap(kProfStart, module, device_list);
  GELOGI("Prof start success, module=%" PRIu64, module);
  return SUCCESS;
}

Status ProfilingManager::ProfStopProfiling(const uint64_t module,
                                           const std::map<std::string, std::string> &config_para) {
  const std::lock_guard<std::mutex> lock(mutex_);
  int32_t device_num = 0;
  std::vector<int32_t> device_list;
  if (ProfParseParam(config_para, device_num, device_list) != SUCCESS) {
    GELOGE(FAILED, "[Stop][Profiling]Prof stop parse param failed, device num %d, "
           "device list size %zu", device_num, device_list.size());
    REPORT_INNER_ERR_MSG("E19999", "Prof stop parse param failed, device num %d, device list size %zu",
                      device_num, device_list.size());
    return FAILED;
  }

  const uint64_t execute_model_mask = module & static_cast<uint64_t>(PROF_MODEL_EXECUTE_MASK);
  if (execute_model_mask == static_cast<uint64_t>(PROF_MODEL_EXECUTE_MASK)) {
    for (size_t i = 0U; i < static_cast<size_t>(device_num); i++) {
      const auto iter = std::find(device_id_.begin(), device_id_.end(), device_list[i]);
      if (iter != device_id_.end()) {
        (void)device_id_.erase(iter);
      }
    }
    GELOGI("Prof stop: ge execute model stop profiling.");
  }
  if ((module & static_cast<uint64_t>(PROF_MODEL_LOAD_MASK)) == static_cast<uint64_t>(PROF_MODEL_LOAD_MASK)) {
    GELOGW("Prof stop: load model module is invalid.");
  }

  // profiling op detail off
  const uint64_t op_detail_mask = module & (static_cast<uint64_t>(PROF_OP_DETAIL_MASK));
  if (op_detail_mask == static_cast<uint64_t>(PROF_OP_DETAIL_MASK)) {
    ProfilingProperties::Instance().SetOpDetailProfiling(false);
  }

  const uint64_t task_event_profiling = module & (static_cast<uint64_t>(PROF_TASK_FRAMEWORK_MASK));
  if (task_event_profiling == static_cast<uint64_t>(PROF_TASK_FRAMEWORK_MASK)) {
    ProfilingProperties::Instance().SetTaskEventProfiling(false);
  }

  ProfilingProperties::Instance().UpdateDeviceIdModuleMap(kProfStop, module, device_list);

  if (device_id_.empty()) {
    ProfilingProperties::Instance().ClearProperties();
    GELOGI("Prof stop profiling clear properties success.");
  }
  GELOGI("Prof stop success.");
  return SUCCESS;
}

bool ProfilingManager::ProfilingModelExecuteOn() const {
  int32_t logic_device_id = 0;
  const rtError_t rt_ret = rtGetDevice(&logic_device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Get][LogicDeviceId]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Get logic device id failed, ret %d", rt_ret);
  }
  GELOGI("Current logic_device_id:%d", logic_device_id);

  bool execute_model_prof_on = false;
  const auto iter = std::find(device_id_.begin(), device_id_.end(), logic_device_id);
  if (iter != device_id_.end()) {
    execute_model_prof_on = true;
  }

  GELOGI("Flag is_execute_profiling: %d, execute_model_prof_on: %d, op_detail_prof_on: %d.",
         static_cast<int32_t>(ProfilingProperties::Instance().IsExecuteProfiling()),
         static_cast<int32_t>(execute_model_prof_on),
         static_cast<int32_t>(ProfilingProperties::Instance().IsOpDetailProfiling()));
  return execute_model_prof_on ||
         gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime);
}

void ProfilingManager::GetOpInputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const {
  std::vector<Format> input_format;
  std::vector<std::vector<int64_t>> input_shape;
  std::vector<DataType> input_data_type;
  for (size_t i = 0U; i < op->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr input_tensor_desc = op->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_tensor_desc == nullptr) {
      continue;
    }
    input_format.emplace_back(input_tensor_desc->GetFormat());
    input_shape.emplace_back(input_tensor_desc->GetShape().GetDims());
    input_data_type.emplace_back(input_tensor_desc->GetDataType());
  }

  const std::vector<Format> format_default =  { FORMAT_NULL };
  const std::vector<std::vector<int64_t>> shape_default = { {0} };
  const std::vector<DataType> data_type_default = { DT_UNDEFINED };
  task_desc_info.input_format = input_format.empty() ? format_default : input_format;
  task_desc_info.input_shape = input_shape.empty() ? shape_default : input_shape;
  task_desc_info.input_data_type = input_data_type.empty() ? data_type_default : input_data_type;
}

void ProfilingManager::GetOpOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const {
  std::vector<Format> output_format;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<DataType> output_data_type;
  for (size_t j = 0U; j < op->GetOutputsSize(); ++j) {
    const GeTensorDescPtr output_tensor_desc = op->MutableOutputDesc(static_cast<uint32_t>(j));
    if (output_tensor_desc == nullptr) {
      continue;
    }
    output_format.emplace_back(output_tensor_desc->GetFormat());
    output_shape.emplace_back(output_tensor_desc->GetShape().GetDims());
    output_data_type.emplace_back(output_tensor_desc->GetDataType());
  }

  const std::vector<Format> format_default =  { FORMAT_NULL };
  const std::vector<std::vector<int64_t>> shape_default = { {0} };
  const std::vector<DataType> data_type_default = { DT_UNDEFINED };
  task_desc_info.output_format = output_format.empty() ? format_default : output_format;
  task_desc_info.output_shape = output_shape.empty() ? shape_default : output_shape;
  task_desc_info.output_data_type = output_data_type.empty() ? data_type_default : output_data_type;
}

void ProfilingManager::GetOpInputOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const {
  GetOpInputInfo(op, task_desc_info);
  GetOpOutputInfo(op, task_desc_info);
}

Status ProfilingManager::GetDeviceIdFromGraph(const uint32_t graph_id, uint32_t &device_id) {
  const std::map<uint32_t, uint32_t>::const_iterator iter = device_id_map_.find(graph_id);
  if (iter != device_id_map_.cend()) {
    device_id = iter->second;
    return SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E19999", "graph_id:%u does not exist!", graph_id);
  GELOGE(PARAM_INVALID, "[Check][GraphId]graph_id:%u does not exist!", graph_id);
  return FAILED;
}

Status ProfilingManager::GetModelIdFromGraph(const uint32_t graph_id, uint32_t &model_id) {
  const std::map<uint32_t, uint32_t>::const_iterator iter = model_id_map_.find(graph_id);
  if (iter != model_id_map_.cend()) {
    model_id = iter->second;
    return SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E19999", "graph_id:%u does not exist!", graph_id);
  GELOGE(PARAM_INVALID, "[Check][GraphId]graph_id:%u does not exist!", graph_id);
  return FAILED;
}

bool ProfilingManager::IsGraphProfReported(const uint32_t graph_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = reported_graph_id_.find(graph_id);
  return (iter != reported_graph_id_.end());
}

void ProfilingManager::InsertReportedGraphId(const uint32_t graph_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  (void)reported_graph_id_.emplace(graph_id);
}

void ProfilingManager::SetGraphIdToDeviceMap(const uint32_t graph_id, const uint32_t device_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GELOGD("Save graph id:%u and device id:%u to device_id_map.", graph_id, device_id);
  device_id_map_[graph_id] = device_id;
}

void ProfilingManager::SetGraphIdToModelMap(const uint32_t graph_id, const uint32_t model_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GELOGD("Save graph id:%u and model id:%u to model_id_map.", graph_id, model_id);
  model_id_map_[graph_id] = model_id;
}

void ProfilingManager::RemoveFromGraphIdMap(const uint32_t model_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  GELOGD("Remove model id:%u.", model_id);
  auto iter = model_id_map_.begin();
  while (iter != model_id_map_.end()) {
    if (iter->second == model_id) {
      iter = model_id_map_.erase(iter);
    } else {
      ++iter;
    }
  }
}
bool ProfilingManager::ProfilingSubscribeOn() const {
  return ProfilingProperties::Instance().ProfilingSubscribeOn();
}

ProfilerCollector::ProfilerCollector(const uint32_t model_id, const uint32_t graph_id)
    : model_id_(model_id), graph_id_(graph_id), step_id_{1U} {}

ge::Status ProfilerCollector::RecordStart(const rtStream_t stream) const {
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
    MsprofEvent model_execute_info{};
    gert::GlobalProfilingWrapper::GetInstance()->SetModelIdStepId(model_id_, step_id_);
    GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::ReportEvent(
        static_cast<uint64_t>(model_id_), step_id_, gert::GeProfInfoType::kModelExecute, model_execute_info));
  }
  if (!host_cpu_flag_) {
    GE_ASSERT_SUCCESS(
        gert::GlobalProfilingWrapper::ProfileStepTrace(static_cast<uint64_t>(step_id_),
                                                       model_id_, kStartTag, stream));
  }
  return ge::SUCCESS;
}

ge::Status ProfilerCollector::RecordEnd(const rtStream_t stream) {
  if (!host_cpu_flag_) {
    GE_ASSERT_SUCCESS(
        gert::GlobalProfilingWrapper::ProfileStepTrace(static_cast<uint64_t>(step_id_), model_id_, kEndTag, stream));
  }
  if (gert::GlobalProfilingWrapper::GetInstance()->IsEnabled(gert::ProfilingType::kTaskTime)) {
    MsprofEvent model_execute_info{};
    GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::ReportEvent(
        static_cast<uint64_t>(model_id_), step_id_, gert::GeProfInfoType::kModelExecute, model_execute_info));
    GE_ASSERT_SUCCESS(gert::GlobalProfilingWrapper::ReportGraphIdMap(
        model_execute_info.timeStamp, model_execute_info.threadId, {graph_id_, model_id_}, true));
  }
  ++step_id_;
  return ge::SUCCESS;
}
}  // namespace ge
