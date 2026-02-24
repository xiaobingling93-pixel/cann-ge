/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CANN_GRAPH_ENGINE_INNER_TENSOR_MOVE_DELETE_PASS_H
#define CANN_GRAPH_ENGINE_INNER_TENSOR_MOVE_DELETE_PASS_H

#include "graph/passes/graph_pass.h"
#include "graph/utils/connection_matrix.h"

namespace ge {
/**
 * 对所有内部的TensorMove节点，根据特定规则删除冗余的TensorMove。
 * 如下情况可以直接删除：
 * 1. TensorMove输入节点是const之类不可改写的类型
 * 2. topo排序方式是稳定拓扑序
 * 3. TensorMove后面连的不是ref算子
 * 其他场景需要根据相关图结构进行判断：
 * 1. TensorMove输入节点A是单引用，且TensorMove输出是单引用，则可以删除
 * 2. TensorMove输入节点A是单引用，TensorMove是多引用，TensorMove除Ref之外的其他所有输出都受Ref控制则可以删除
 * 3. TensorMove输入节点A是多引用，且TensorMove输出是单引用，如果A节点除TensorMove之外的其他所有输出都指向了ref，则可以删除
 * 4. TensorMove输入节点A是多引用，且TensorMove输出是多引用，且TensorMove的其他输出节点都依赖ref，且A节点除TensorMove之外的其他输出都指向ref，则可以删除
 */
class InnerTensorMoveDeletePass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  Status DeleteInnerTensorMove(const NodePtr &node);

  Status IsolateAndDeleteTensorMoveNode(const NodePtr &node);

 private:
  std::unique_ptr<ConnectionMatrix> connectivity_{nullptr};
};
}  // namespace ge

#endif //CANN_GRAPH_ENGINE_INNER_TENSOR_MOVE_DELETE_PASS_H