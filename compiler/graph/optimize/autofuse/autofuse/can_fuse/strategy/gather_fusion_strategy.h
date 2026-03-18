/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_CAN_FUSE_STRATEGY_GATHER_FUSION_STRATEGY_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_CAN_FUSE_STRATEGY_GATHER_FUSION_STRATEGY_H_
#include "can_fuse/strategy/fusion_strategy.h"

namespace ge {
class GatherFusionStrategy : public FusionStrategy {
 public:
  GatherFusionStrategy() = default;

  ~GatherFusionStrategy() override = default;

  GatherFusionStrategy(const GatherFusionStrategy &) = delete;
  GatherFusionStrategy &operator=(const GatherFusionStrategy &) = delete;

  // 检查两个节点是否可以融合
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) override;
  static bool CheckGatherReduceFuse(const NodePtr &node1, const NodePtr &node2, const bool enable_gather_reduce,
                                    const bool enable_gather_broadcast);
  static bool CheckGatherElemwiseFuse(const NodePtr &node1, const NodePtr &node2, const bool enable_gather_broadcast);
  static bool CheckGatherConcatFuse(const NodePtr &node1, const NodePtr &node2, const bool enable_gather_concat,
                                    bool enable_gather_broadcast);
  // 获取融合对的优先级
  FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) override;
  static bool CheckGatherFuseAxis(const NodePtr &node1, const NodePtr &node2);
  static bool CheckGatherWithView(const NodePtr &node1, const NodePtr &node2, const bool enable_gather_broadcast);

 private:
  static bool IsForwardFuseGather(const NodePtr &node1, const NodePtr &node2);
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_CAN_FUSE_STRATEGY_GATHER_FUSION_STRATEGY_H_
