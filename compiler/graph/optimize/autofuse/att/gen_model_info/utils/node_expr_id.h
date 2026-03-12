/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GEN_MODEL_INFO_UTILS_NODE_EXPR_ID_H_
#define ATT_GEN_MODEL_INFO_UTILS_NODE_EXPR_ID_H_

#include <cctype>
#include <string>
#include <vector>

namespace att {

struct NodeInfo;  // 前置声明

// 消毒节点名称中的非法字符，将其转换为合法的C++变量名
// 将所有非字母数字字符（除了下划线）替换为下划线
inline std::string SanitizeNodeName(const std::string &name) {
  std::string result;
  result.reserve(name.length() + 1U);
  for (char c : name) {
    if (std::isalnum(c) || c == '_') {
      result += c;
    } else {
      result += '_';
    }
  }
  // C++ 标识符不能以数字开头，若首字符为数字则加前缀下划线
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = '_' + result;
  }
  return result;
}

// 节点表达式标识符，用于生成表达式变量名
// 表达式变量名格式：{node_name}_{node_type}
// 注释显示格式：{node_name}_{node_type}_in{input_shapes}_out{output_shapes}
// 示例：表达式=load1_Load，注释=load1_Load_in[32,64]_out[32,64]
struct NodeExprId {
  std::string node_name;                  // 节点名称
  std::string node_type;                  // 节点类型
  std::vector<std::string> input_shapes;  // 输入张量形状摘要
  std::vector<std::string> output_shapes; // 输出张量形状摘要

  // 生成表达式变量名前缀：sanitized(node_name) + "_" + node_type
  std::string GetExprVarPrefix() const {
    return SanitizeNodeName(node_name) + "_" + node_type;
  }

  // 生成注释显示的完整前缀：{node_name}_{node_type}_in{shapes}_out{shapes}
  std::string GetVarPrefix() const {
    std::string prefix = node_name + "_" + node_type;
    // 始终添加 _in 和 _out，即使形状为空
    prefix += "_in" + (input_shapes.empty() ? "[]" : GetShapesString(input_shapes));
    prefix += "_out" + (output_shapes.empty() ? "[]" : GetShapesString(output_shapes));
    return prefix;
  }

 private:
  static std::string GetShapesString(const std::vector<std::string> &shapes) {
    // shapes 每个元素已经是 "[x,y]" 格式，直接用逗号连接
    std::string result;
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (i > 0) {
        result += ",";
      }
      result += shapes[i];
    }
    return result;
  }
};

// 从 NodeInfo 构建节点表达式标识符
NodeExprId BuildNodeExprId(const NodeInfo &node_info);

} // namespace att

#endif // ATT_GEN_MODEL_INFO_UTILS_NODE_EXPR_ID_H_
