/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_STOP_GRADIENT_PASS_H_
#define GE_GRAPH_PASSES_STOP_GRADIENT_PASS_H_

#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/omg_util/omg_util.h"
#include "graph/passes/base_pass.h"

namespace ge {
class StopGradientPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_STOP_GRADIENT_PASS_H_
