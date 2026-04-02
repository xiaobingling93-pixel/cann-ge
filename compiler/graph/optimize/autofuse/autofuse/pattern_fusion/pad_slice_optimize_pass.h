/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef CANN_GRAPH_ENGINE_PAD_SLICE_OPTIMIZE_PASS_H
#define CANN_GRAPH_ENGINE_PAD_SLICE_OPTIMIZE_PASS_H

#include "graph/node.h"
#include "graph/ge_error_codes.h"
#include "gnode.h"

namespace ge {
class PadSliceOptimizePass {
 public:
  graphStatus Run(const ComputeGraphPtr &graph, bool &changed);
  graphStatus PostProcess(const ComputeGraphPtr &graph, const NodePtr &node) const;
};
}
#endif  // CANN_GRAPH_ENGINE_PAD_SLICE_OPTIMIZE_PASS_H
