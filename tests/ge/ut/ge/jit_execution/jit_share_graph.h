/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_JIT_SHARE_GRAPH_H
#define CANN_GRAPH_ENGINE_JIT_SHARE_GRAPH_H
#include "graph/graph.h"
#include "graph/node.h"
namespace ge {
using UniqueGraphPtr = std::unique_ptr<Graph>;
struct JitShareGraph {
  static void AddCompileResult(const ge::NodePtr &node, bool atomic, const char *compile_info_json);
  static UniqueGraphPtr AllNormalNodes(const std::vector<int64_t> &input_dims = {});
  static UniqueGraphPtr AllNormalNodesStaticShape();
  static UniqueGraphPtr OneUniqueNode();
  static UniqueGraphPtr OneReshapeNode(const std::vector<int64_t> &input1_dims = {},
                                       const std::vector<int64_t> &input2_dims = {});
  static UniqueGraphPtr OneReshapeNodeWithHostInput(const std::vector<int64_t> &input1_dims = {},
                                                    const std::vector<int64_t> &input2_dims = {},
                                                    const std::vector<int64_t> &input3_dims = {});
  static UniqueGraphPtr OneReshapeNodeTwoRelu();
  static UniqueGraphPtr TwoReshapeNodeTwoRelu();
  static UniqueGraphPtr AddUniqueNode();
  static UniqueGraphPtr OneAddNode();
  static UniqueGraphPtr OneConstTwoReshapeNodeTwoRelu();
};
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_JIT_SHARE_GRAPH_H
