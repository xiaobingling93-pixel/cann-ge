/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/placeholder_with_default_pass.h"
#include <string>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/omg_util/omg_util.h"

namespace ge {
Status PlaceholderWithDefaultPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  std::string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "[Get][OriginalType] of node:%s failed.", node->GetName().c_str());
    return status_ret;
  }
  if (type == PLACEHOLDERWITHDEFAULT) {
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}

REG_PASS_OPTION("PlaceholderWithDefaultPass").LEVELS(OoLevel::kO0);
}  // namespace ge
