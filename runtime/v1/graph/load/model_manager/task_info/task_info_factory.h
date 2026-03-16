/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_FACTORY_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
class TaskInfo;
using TaskInfoPtr = std::shared_ptr<TaskInfo>;

class TaskInfoFactory {
 public:
  // TaskManagerCreator function def
  using TaskInfoCreatorFun = std::function<TaskInfoPtr(void)>;

  static TaskInfoFactory &Instance();
  static void Replace(std::shared_ptr<TaskInfoFactory> ins);

  TaskInfoPtr Create(const ModelTaskType task_type) {
    const auto iter = creator_map_.find(task_type);
    if (iter == creator_map_.end()) {
      GELOGW("Cannot find task type %d in inner map.", static_cast<int32_t>(task_type));
      return nullptr;
    }

    return iter->second();
  }

  // TaskInfo registerar
  class Registerar {
   public:
    Registerar(const ModelTaskType type, const TaskInfoCreatorFun &func) noexcept {
      TaskInfoFactory::Instance().RegisterCreator(type, func);
    }

    ~Registerar() = default;
  };

 private:
  friend class TaskInfoRegistryStub;  // for test
  TaskInfoFactory() = default;

  ~TaskInfoFactory() = default;

  // register creator, this function will call in the constructor
  void RegisterCreator(const ModelTaskType type, const TaskInfoCreatorFun &func) {
    const auto iter = creator_map_.find(type);
    if (iter != creator_map_.end()) {
      GELOGD("TaskManagerFactory::RegisterCreator: %d creator already exist", static_cast<int32_t>(type));
      return;
    }
    creator_map_[type] = func;
  }

  std::map<ModelTaskType, TaskInfoCreatorFun> creator_map_;
};
}  // namespace ge

#define REGISTER_TASK_INFO(type, clazz)                                                                       \
namespace {                                                                                                   \
  TaskInfoPtr Creator_Task_Info_##type() {                                                                    \
    std::shared_ptr<clazz> ptr = nullptr;                                                                     \
    ptr = MakeShared<clazz>();                                                                                \
    return ptr;                                                                                               \
  }                                                                                                           \
  TaskInfoFactory::Registerar g_Task_Info_Creator_##type(ModelTaskType::type, &Creator_Task_Info_##type);     \
}  // namespace
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_TASK_INFO_FACTORY_H_
