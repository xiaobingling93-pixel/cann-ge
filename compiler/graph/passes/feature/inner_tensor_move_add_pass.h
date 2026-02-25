/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CANN_GRAPH_ENGINE_TENSOR_MOVE_ADD_PASS_H
#define CANN_GRAPH_ENGINE_TENSOR_MOVE_ADD_PASS_H

#include "graph/passes/graph_pass.h"

namespace ge {
/**
 * 对图上所有ref类算子，在其ref输入前插入内部的TensorMove节点，防止后续其他pass对ref输入进行修改时，其他输出受ref算子影响。
 * 如下场景不插TensorMove：
 * 1. 当ref类算子的输入原本就是TensorMove，且该TensorMove只连给这个ref算子
 * 2. ref输入是variable类算子或者特殊算子（如RefSwitch、RefMerge、ReadVariableOp等)
 * 其他情况都需要插入TensorMove
 */
class InnerTensorMoveAddPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  NodePtr AddTensorMove(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_anchor,
                        const InDataAnchorPtr &dst_anchor);

 private:
  std::unique_ptr<ConnectionMatrix> connectivity_{nullptr};
};
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_TENSOR_MOVE_ADD_PASS_H
