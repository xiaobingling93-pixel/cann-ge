/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_INC_KERNEL_FACTORY_H_
#define GE_INC_KERNEL_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/graph.h"

namespace ge {
class Kernel;

///
/// @ingroup domi_omg
/// @brief kernel create factory
/// @author
///
class GE_OBJECT_VISIBILITY KernelFactory {
 public:
  // KernelCreator（function）, type definition
  using KERNEL_CREATOR_FUN = std::function<std::shared_ptr<Kernel>(void)>;

  ///
  /// Get singleton instance
  ///
  static KernelFactory &Instance() {
    static KernelFactory instance;
    return instance;
  }

  ///
  /// create Kernel
  /// @param [in] op_type operation type
  ///
  std::shared_ptr<Kernel> Create(const std::string &op_type) {
    const std::map<std::string, KERNEL_CREATOR_FUN>::iterator &iter = creator_map_.find(op_type);
    if (iter != creator_map_.end()) {
      return iter->second();
    }

    return nullptr;
  }

  // Kernel registration function to register different types of kernel to the factory
  class Registerar {
   public:
    ///
    /// @ingroup domi_omg
    /// @brief Constructor
    /// @param [in] type operation type
    /// @param [in| fun kernel function of the operation
    ///
    Registerar(const std::string &type, const KERNEL_CREATOR_FUN &fun) noexcept {
      KernelFactory::Instance().RegisterCreator(type, fun);
    }
    ~Registerar() {}
  };

 protected:
  KernelFactory() {}
  ~KernelFactory() {}

  // register creator, this function will call in the constructor
  void RegisterCreator(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
    const std::map<std::string, KERNEL_CREATOR_FUN>::iterator &iter = creator_map_.find(type);
    if (iter != creator_map_.end()) {
      GELOGD("KernelFactory::RegisterCreator: %s creator already exist", type.c_str());
      return;
    }

    creator_map_[type] = fun;
  }

 private:
  std::map<std::string, KERNEL_CREATOR_FUN> creator_map_;
};
}  // namespace ge

#define REGISTER_COMPUTE_NODE_KERNEL(type, clazz)                                                       \
namespace {                                                                                \
  std::shared_ptr<ge::Kernel> Creator_##type##_Kernel() {                                  \
    return ge::MakeShared<clazz>();                                                        \
  }                                                                                        \
  ge::KernelFactory::Registerar g_##type##_Kernel_Creator(type, &Creator_##type##_Kernel); \
}

#endif  // GE_INC_KERNEL_FACTORY_H_
