/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_ASCIR_OPS_UTILS_H
#define AUTOFUSE_ASCIR_OPS_UTILS_H

#include <memory>

#include "graph/node.h"
#include "graph/symbolizer/symbolic.h"

namespace ge {
namespace ops {
static ge::Expression Zero = ge::Symbol(0);
static ge::Expression One = ge::Symbol(1);

template <typename T>
bool IsOps(const ge::NodePtr &node) {
  return node->GetType() == T::Type;
}

template <typename T>
bool IsOps(const ge::Node *node) {
  return node->GetType() == T::Type;
}
}
}
#endif  // AUTOFUSE_ASCIR_OPS_UTILS_H
