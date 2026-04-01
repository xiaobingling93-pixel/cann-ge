/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_EXECUTOR_HYBRID_EXECUTOR_CALLBACK_MANAGER_H_
#define AIR_EXECUTOR_HYBRID_EXECUTOR_CALLBACK_MANAGER_H_

#include <functional>
#include "common/plugin/ge_make_unique_util.h"
#include "ge/ge_api_error_codes.h"
#include "acl/acl_rt.h"

namespace ge {
namespace hybrid {
class CallbackManager {
 public:
  CallbackManager() = default;
  virtual ~CallbackManager() = default;

  GE_DELETE_ASSIGN_AND_COPY(CallbackManager);

  virtual Status Init() = 0;

  virtual Status Destroy() = 0;

  virtual Status RegisterCallbackFunc(const aclrtStream stream, const std::function<void()> &callback) = 0;
};
}  // namespace hybrid
}  // namespace ge

#endif  // AIR_EXECUTOR_HYBRID_EXECUTOR_CALLBACK_MANAGER_H_
