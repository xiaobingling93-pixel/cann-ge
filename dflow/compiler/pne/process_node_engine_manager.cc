/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "process_node_engine_manager.h"

#include <cstdio>
#include <map>

#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"

namespace ge {
ProcessNodeEngineManager &ProcessNodeEngineManager::GetInstance() {
  static ProcessNodeEngineManager instance;
  return instance;
}

Status ProcessNodeEngineManager::Initialize(const std::map<std::string, std::string> &options) {
  // Multiple initializations are not supported
  if (init_flag_.load()) {
    GELOGW("ProcessNodeEngineManager has been initialized.");
    return SUCCESS;
  }

  GE_TIMESTAMP_START(ProcessNodeEngine);
  // Load process node engine so
  const std::string so_path = "plugin/pnecompiler/";
  std::string path = GetModelPath();
  (void)path.append(so_path);
  const std::vector<std::string> so_func;
  const Status status = plugin_mgr_.Load(path, so_func);
  if (status != SUCCESS) {
    GELOGE(status, "[Load][EngineSo]Failed, lib path %s", path.c_str());
    REPORT_INNER_ERR_MSG("E19999", "Load engine so failed, lib path %s", path.c_str());
    return status;
  }

  GELOGI("The number of ProcessNodeEngine is %zu.", engines_map_.size());

  const std::lock_guard<std::mutex> lock(mutex_);
  // Engines initialize
  for (auto iter = engines_map_.cbegin(); iter != engines_map_.cend(); ++iter) {
    if (iter->second == nullptr) {
      GELOGI("Engine: %s point to nullptr", (iter->first).c_str());
      continue;
    }

    GELOGI("ProcessNodeEngine id:%s.", (iter->first).c_str());

    const uint64_t start = ge::GetCurrentTimestamp();
    const Status init_status = iter->second->Initialize(options);
    const uint64_t end = ge::GetCurrentTimestamp();
    GEEVENT("[GEPERFTRACE] The time cost of ProcessNodeEngineManager::Initialize[%s] is [%lu] micro seconds.",
            (iter->first).c_str(), (end - start));
    if (init_status != SUCCESS) {
      GELOGE(init_status, "[Init][Engine]Failed, ProcessNodeEngine:%s", (iter->first).c_str());
      REPORT_INNER_ERR_MSG("E19999", "Initialize ProcessNodeEngine:%s failed", (iter->first).c_str());
      return init_status;
    }
  }

  GE_TIMESTAMP_EVENT_END(ProcessNodeEngine, "InnerInitialize::ProcessNodeEngine");
  init_flag_.store(true);
  return SUCCESS;
}

Status ProcessNodeEngineManager::Finalize() {
  // Finalize is not allowed, initialize first is necessary
  if (!init_flag_.load()) {
    GELOGW("ProcessNodeEngineManager has been finalized.");
    return SUCCESS;
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter = engines_map_.cbegin(); iter != engines_map_.cend(); ++iter) {
    if (iter->second != nullptr) {
      GELOGI("ProcessNodeEngine id:%s.", (iter->first).c_str());
      const Status status = iter->second->Finalize();
      if (status != SUCCESS) {
        GELOGE(status, "[Finalize][Engine]Failed, ProcessNodeEngine:%s", (iter->first).c_str());
        REPORT_INNER_ERR_MSG("E19999", "Finalize ProcessNodeEngine:%s failed", (iter->first).c_str());
        return status;
      }
    }
  }
  engines_map_.clear();
  engines_create_map_.clear();
  init_flag_.store(false);
  return SUCCESS;
}

ProcessNodeEnginePtr ProcessNodeEngineManager::GetEngine(const std::string &engine_id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = engines_map_.find(engine_id);
  if (iter != engines_map_.end()) {
    return iter->second;
  }

  GELOGW("Failed to get ProcessNodeEngine object by id:%s.", engine_id.c_str());
  return nullptr;
}

bool ProcessNodeEngineManager::IsEngineRegistered(const std::string &engine_id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = engines_map_.find(engine_id);
  if (iter != engines_map_.end()) {
    return true;
  }
  GELOGW("ProcessNodeEngine id:%s is not registered", engine_id.c_str());
  return false;
}

Status ProcessNodeEngineManager::RegisterEngine(const std::string &engine_id, const ProcessNodeEnginePtr &engine,
                                                CreateFn const fn) {
  const std::lock_guard<std::mutex> lock(mutex_);
  engines_map_[engine_id] = engine;
  engines_create_map_[engine_id] = fn;
  GELOGI("Register ProcessNodeEngine id:%s success.", engine_id.c_str());
  return SUCCESS;
}

ProcessNodeEnginePtr ProcessNodeEngineManager::CloneEngine(const std::string &engine_id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  ProcessNodeEnginePtr engine = nullptr;
  const auto it = engines_create_map_.find(engine_id);
  if (it != engines_create_map_.end()) {
    const auto fn = it->second;
    if (fn != nullptr) {
      engine.reset(fn());
    }
  }
  if (engine != nullptr) {
    GELOGI("Clone ProcessNodeEngine id:%s success.", engine_id.c_str());
  } else {
    GELOGE(INTERNAL_ERROR, "Clone ProcessNodeEngine id:%s failed.", engine_id.c_str());
  }
  return engine;
}

ProcessNodeEngineRegisterar::ProcessNodeEngineRegisterar(const std::string &engine_id, CreateFn const fn) noexcept {
  ProcessNodeEnginePtr engine = nullptr;
  if (fn != nullptr) {
    engine.reset(fn());
    if (engine == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Create][ProcessNodeEngine] id:%s", engine_id.c_str());
    } else {
      (void)ProcessNodeEngineManager::GetInstance().RegisterEngine(engine_id, engine, fn);
    }
  } else {
    GELOGE(INTERNAL_ERROR, "[Check][Param:fn]Creator is nullptr, ProcessNodeEngine id:%s", engine_id.c_str());
  }
}
}  // namespace ge
