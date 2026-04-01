/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_LOWERING_IMPL_H_
#define AUTOFUSE_LOWERING_IMPL_H_

#include <functional>
#include <vector>

#include "graph/node.h"
#include "common/checker.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/lowerings.h"
#include "lowering/lowering_utils.h"

namespace ge {

graphStatus Broadcast(const std::vector<loop::Index> &indices, loop::Index &broadcasted);
graphStatus LowerPointwise(const NodePtr &node,
                           const std::function<loop::LoopVar(const std::vector<loop::LoopVar> &)> &kernel);
graphStatus LowerReduction(const NodePtr &node, loop::ReduceType reduce_type);

/**
 * @brief 新增融合原因记录断言宏
 * 功能：
 * 1. 当断言失败时，先存储 node_name + 不能融合原因
 * 2. 再调用原有GE_WARN_ASSERT执行断言逻辑
 * 
 * @param exp 断言表达式（同GE_WARN_ASSERT）
 * @param node 节点指针（必须是ge::Node*类型，用于获取node_name）
 * @param reason 不能融合原因（字符串常量/变量）
 */
#define LOWERING_WARN_RECORD_REASON(exp, node, fmt, ...)                                      \
  do {                                                                                        \
    if (!(exp)) {                                                                             \
      std::string node_name = (node != nullptr) ? node->GetName() : "unknown_node";           \
      char reason_buf[256] = {0};                                                             \
      auto ret = snprintf_s(reason_buf, sizeof(reason_buf), sizeof(reason_buf) - 1U, fmt, ##__VA_ARGS__);\
      GE_ASSERT_TRUE(ret >= 0, "snprintf_s failed, ret: %d", ret);                            \
      std::string reason = reason_buf;                                                        \
      GraphFusionReasonStore::CountNodeFuseFailReason(node_name, reason, GraphFusionReasonStore::FailReasonCategory::NODE_INFO_ERROR); \
      GE_WARN_ASSERT(exp);                                                                    \
    }                                                                                         \
  } while (false)

#define REGISTER_LOWERING(T)                           \
  static graphStatus Lowering##T(const NodePtr &node); \
  const static bool kLowering##T##Registered = []() {  \
    LoweringManager::Register(#T, Lowering##T);        \
    return true;                                       \
  }();                                                 \
  static graphStatus Lowering##T(const NodePtr &node)

#define REGISTER_LOWERING_WITH_EXISTED(OP, FUNC) \
  REGISTER_LOWERING(OP) {                        \
    return FUNC(node);                           \
  }

#define REGISTER_POINTWISE_LOWER(OP, LOOP_OP)                                                          \
  REGISTER_LOWERING(OP) {                                                                              \
    return LowerPointwise(node, [](const std::vector<loop::LoopVar> &vars) { return LOOP_OP(vars); }); \
  }

#define REGISTER_REDUCTION_LOWER(OP, REDUCE_TYPE) \
  REGISTER_LOWERING(OP) {                         \
    return LowerReduction(node, REDUCE_TYPE);     \
  }
}  // namespace ge

#endif  // AUTOFUSE_LOWERING_IMPL_H
