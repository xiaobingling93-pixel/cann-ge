/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_TRANSPOSE_REMOVE_PASS_H
#define GE_GRAPH_PASSES_TRANSPOSE_REMOVE_PASS_H

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "common/omg_util/omg_util.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/passes/pass_utils.h"

namespace ge {
class UnchangedTransposeRemovePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  static bool IsUnchangedTranspose(const NodePtr &node, const OpDescPtr &op_desc);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_TRANSPOSE_REMOVE_PASS_H
