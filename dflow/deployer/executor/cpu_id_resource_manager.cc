/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/debug/log.h"
#include "common/debug/ge_log.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "executor/cpu_id_resource_manager.h"

namespace ge {

CpuIdResourceManager &CpuIdResourceManager::GetInstance() {
  static CpuIdResourceManager instance;
  return instance;
}

Status CpuIdResourceManager::Allocate(uint32_t &id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto available_id = std::find(resources_.begin(), resources_.end(), false) -
                      resources_.begin();
  if (available_id >= kMaxResourceId) {
    GELOGE(FAILED, "Failed to generate available id.");
    return FAILED;
  }
  id = available_id;
  resources_[id] = true;
  return SUCCESS;
}

Status CpuIdResourceManager::DeAllocate(std::vector<uint32_t> &ids) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto id : ids) {
    resources_[id] = false;
  }
  return SUCCESS;
}

Status CpuIdResourceManager::GenerateAicpuStreamId(uint32_t &id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto iter = std::find(streams_.begin(), streams_.end(), false);
  if (iter == streams_.end()) {
    GELOGE(FAILED, "Failed to generate available stream id.");
    return FAILED;
  }
  id = std::distance(streams_.begin(), iter);
  streams_[id] = true;
  return SUCCESS;
}

Status CpuIdResourceManager::FreeAicpuStreamId(const std::vector<uint32_t> &ids) {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto id : ids) {
    streams_[id] = false;
  }
  return SUCCESS;
}

Status AicpuModelIdResourceManager::GenerateAicpuModelId(uint32_t &id) {
  GE_CHK_STATUS_RET(Allocate(id), "Fail to allocate stream id");
  return SUCCESS;
}

Status NotifyIdResourceManager::GenerateNotifyId(uint32_t &id) {
  GE_CHK_STATUS_RET(Allocate(id), "Fail to allocate notify id");
  return SUCCESS;
}
}
