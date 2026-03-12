/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_expr_id.h"
#include "parser/tuning_space.h"
#include "base/base_types.h"

namespace att {

namespace {
// 获取张量形状摘要
std::string GetShapeSummary(const Tensor *tensor) {
  if (!tensor) {
    return "[]";
  }
  const auto &repeat = tensor->repeat;
  if (repeat.empty()) {
    return "scalar";
  }
  // 将所有维度连接成一个字符串
  std::string result = "[";
  for (size_t i = 0; i < repeat.size(); ++i) {
    if (i > 0) {
      result += ",";
    }
    result += Str(repeat[i]);
  }
  result += "]";
  return result;
}
} // namespace

// 从 NodeInfo 构建节点表达式标识符
NodeExprId BuildNodeExprId(const att::NodeInfo &node_info) {
  NodeExprId id;
  auto node = node_info.node_ptr;

  id.node_name = node->GetName();
  // 优先使用 node_info.node_type，因为它已经包含了正确的类型信息
  id.node_type = node_info.node_type.empty() ? node->GetType() : node_info.node_type;

  // 获取输入形状
  for (const auto &tensor : node_info.inputs) {
    id.input_shapes.push_back(GetShapeSummary(tensor.get()));
  }

  // 获取输出形状
  for (const auto &tensor : node_info.outputs) {
    id.output_shapes.push_back(GetShapeSummary(tensor.get()));
  }

  return id;
}

} // namespace att
