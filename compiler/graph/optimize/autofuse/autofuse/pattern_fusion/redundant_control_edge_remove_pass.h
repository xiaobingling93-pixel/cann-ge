/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_REDUNDANT_CONTROL_EDGE_REMOVE_PASS_H_
#define AUTOFUSE_REDUNDANT_CONTROL_EDGE_REMOVE_PASS_H_

#include <memory>
#include "graph/node.h"

namespace ge {

class RedundantControlEdgeRemovePass {
 public:
  graphStatus Run(const ComputeGraphPtr &graph, bool &changed) const;
};

}  // namespace ge

#endif  // AUTOFUSE_REDUNDANT_CONTROL_EDGE_REMOVE_PASS_H_
