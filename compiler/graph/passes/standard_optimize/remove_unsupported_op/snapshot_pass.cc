/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_unsupported_op/snapshot_pass.h"
#include <string>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/omg_util.h"

namespace ge {
Status SnapshotPass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return FAILED;
  }
  std::string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "[Get][OriginalType] of op:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str());
    return status_ret;
  }
  if (type == SNAPSHOT) {
    return IsolateAndDeleteNode(node, {0});
  }
  return SUCCESS;
}

REG_PASS_OPTION("SnapshotPass").LEVELS(OoLevel::kO0);
}  // namespace ge
