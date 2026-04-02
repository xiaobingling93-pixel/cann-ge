/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PROFILING_PROFILING_MANAGER_H_
#define GE_COMMON_PROFILING_PROFILING_MANAGER_H_

#include <mutex>
#include <map>
#include <string>
#include <vector>
#include <atomic>
#include "securec.h"
#include "mmpa/mmpa_api.h"
#include "graph/op_desc.h"
#include "framework/common/ge_inner_error_codes.h"
#include "register/register_types.h"
#include "runtime/stream.h"
#include "aprof_pub.h"
#include "common/profiling/profiling_properties.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "framework/runtime/model_v2_executor.h"

namespace ge {
class ProfilingManager {
 public:
  ProfilingManager();
  virtual ~ProfilingManager() = default;
  static ProfilingManager &Instance();
  static void RegisterElement(int64_t &idx, const std::string &element);

  // Ctrl from ModelManager.
  Status ProfInit(const uint64_t module);
  Status ProfFinalize();
  void ProfOpDetailProfiling(const uint64_t module, const uint32_t cache_flag) const;
  Status ProfStartProfiling(const uint64_t module, const std::map<std::string, std::string> &config_para,
                            const uint32_t cache_flag = 0U);
  Status ProfStopProfiling(const uint64_t module, const std::map<std::string, std::string> &config_para);
  Status CheckInitForSubscribe(const uint64_t module, const uint32_t device, const uint32_t model_id);
  Status ProfModelUnsubscribe(const uint32_t device, const uint32_t model_id);

  // report model load profiling data flag, data contain task desc info, step info, model load fusion op info
  bool ProfilingModelLoadOn() const { return ProfilingProperties::Instance().IsLoadProfiling(); }
  // report model execute profiling data flag, data contain model execute time info
  bool ProfilingModelExecuteOn() const;
  // profiling subscribe is set
  bool ProfilingSubscribeOn() const;
  // is_execute_profiling_ only used by ge option and env

  void GetOpInputOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const;
  void SetStepInfoIndex(const int64_t index_id) { index_id_ = index_id; }
  int64_t GetStepInfoIndex() const { return index_id_; }
  void RecordLoadedModelId(const uint32_t model_id_) {
    const std::lock_guard<std::mutex> lock(mutex_);
    loaded_model_id_.insert(model_id_);
  }

  void RemoveUnloadedModelId(const uint32_t model_id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    loaded_model_id_.erase(model_id);
  }

  void RemoveFromGraphIdMap(const uint32_t model_id);

  void SetGraphIdToDeviceMap(const uint32_t graph_id, const uint32_t device_id);
  Status GetDeviceIdFromGraph(const uint32_t graph_id, uint32_t &device_id);
  void SetGraphIdToModelMap(const uint32_t graph_id, const uint32_t model_id);
  Status GetModelIdFromGraph(const uint32_t graph_id, uint32_t &model_id);
  bool IsGraphProfReported(const uint32_t graph_id);
  void InsertReportedGraphId(const uint32_t graph_id);

 private:
  Status ProfParseParam(const std::map<std::string, std::string> &config_para, int32_t &device_num,
                        std::vector<int32_t> &device_list) const;
  Status ProfParseDeviceId(const std::map<std::string, std::string> &config_para,
                           std::vector<int32_t> &device_list) const;
  void GetOpInputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const;
  void GetOpOutputInfo(const OpDescPtr &op, TaskDescInfo &task_desc_info) const;

  std::vector<int32_t> device_id_;
  std::mutex mutex_;
  std::mutex mutex_report_;
  std::mutex mutex_hash_;
  int64_t index_id_{std::numeric_limits<int64_t>::max()};
  std::map<uint32_t, uint32_t> device_id_map_; // key: graph_id, value: device_id
  std::map<uint32_t, uint32_t> model_id_map_; // key: graph_id, value: model_id
  std::unordered_set<uint32_t> reported_graph_id_;
  std::set<uint32_t> loaded_model_id_;
};

class ProfilerCollector {
 public:
  ProfilerCollector(const uint32_t model_id, const uint32_t graph_id);
  ge::Status RecordStart(const aclrtStream stream) const;
  ge::Status RecordEnd(const aclrtStream stream);

  bool host_cpu_flag_ = false;
 private:
  uint32_t model_id_;
  uint32_t graph_id_;
  uint32_t step_id_;
};
}  // namespace ge
#endif  // GE_COMMON_PROFILING_PROFILING_MANAGER_H_
