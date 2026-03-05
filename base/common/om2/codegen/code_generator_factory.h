/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_CODE_GENERATOR_FACTORY_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_CODE_GENERATOR_FACTORY_H_
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
class TaskCodeGenerator;
using TaskCodeGeneratorPtr = std::shared_ptr<TaskCodeGenerator>;

class TaskCodeGeneratorFactory {
 public:
  // TaskCodeGenerator creator function def
  using TaskCodeGeneratorCreatorFun = std::function<TaskCodeGeneratorPtr(void)>;

  static TaskCodeGeneratorFactory &Instance() {
    if (g_user_defined_instance_ != nullptr) {
      return *g_user_defined_instance_;
    } else {
      static TaskCodeGeneratorFactory instance;
      return instance;
    }
  }
  static void Replace(std::shared_ptr<TaskCodeGeneratorFactory> ins) {
    g_user_defined_instance_ = std::move(ins);
  }

  TaskCodeGeneratorPtr Create(const ModelTaskType generator_type) {
    const auto iter = creator_map_.find(generator_type);
    if (iter == creator_map_.end()) {
      GELOGW("Cannot find generator type %d in inner map.", static_cast<int32_t>(generator_type));
      return nullptr;
    }

    return iter->second();
  }

  // TaskCodeGenerator registerar
  class Registerar {
   public:
    Registerar(const ModelTaskType type, const TaskCodeGeneratorCreatorFun &func) noexcept {
      TaskCodeGeneratorFactory::Instance().RegisterCreator(type, func);
    }

    ~Registerar() = default;
  };

 private:
  TaskCodeGeneratorFactory() = default;

  ~TaskCodeGeneratorFactory() = default;

  // register creator, this function will call in the constructor
  void RegisterCreator(const ModelTaskType type, const TaskCodeGeneratorCreatorFun &func) {
    const auto iter = creator_map_.find(type);
    if (iter != creator_map_.end()) {
      GELOGD("TaskCodeGeneratorFactory::RegisterCreator: %d creator already exist", static_cast<int32_t>(type));
      return;
    }

    creator_map_[type] = func;
  }

  std::map<ModelTaskType, TaskCodeGeneratorCreatorFun> creator_map_;

  inline static std::shared_ptr<TaskCodeGeneratorFactory> g_user_defined_instance_ = nullptr;
};

#define REGISTER_TASK_CODE_GENERATOR(type, clazz)                                                                 \
  namespace {                                                                                                     \
  TaskCodeGeneratorPtr Creator_Task_Code_Generator_##type() {                                                     \
    std::shared_ptr<clazz> ptr = nullptr;                                                                         \
    ptr = MakeShared<clazz>();                                                                                    \
    return ptr;                                                                                                   \
  }                                                                                                               \
  TaskCodeGeneratorFactory::Registerar g_Task_Code_Generator_Creator_##type(ModelTaskType::type,                  \
                                                                            &Creator_Task_Code_Generator_##type); \
  }  // namespace
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_CODE_GENERATOR_FACTORY_H_