/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_FUSION_DECIDER_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_FUSION_DECIDER_H_
#include "graph/compute_graph.h"
#include "autofuse_frame/autofuse_frames.h"

namespace ge {
enum class FusionPriority: uint32_t {
  HIGHEST = 0U, // CanFuse阶段需要保证Split之间完成融合后再进行Split与其它类型节点间的融合
  HIGHER = 1U,  // CanFuse阶段需要保证Split优先与其它类型节点完成融合，保证Split的CanFuse融合预判与最终融合结果一致
  HIGH = 2U,
  DEFAULT = 3U,
  LOW = 4U
};

class FusionDecider {
 public:
  FusionDecider() = default;

  virtual ~FusionDecider() = default;

  FusionDecider(const FusionDecider &) = delete;
  FusionDecider &operator=(const FusionDecider &) = delete;

  // 检查两个节点是否可以垂直融合
  virtual bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) = 0;

  // 检查两个节点是否可以水平融合
  virtual bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2)  = 0;

  // 获取融合对的优先级
  virtual FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) = 0;

  // todo: 融合两个节点, 有没有必要用户写，后续流程串起来再看
  virtual NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
    (void)node1;
    (void)node2;
    (void)counter;
    return nullptr;
  }

  virtual bool CanFuse(const NodePtr &node1, const NodePtr &node2) const {
    (void)node1;
    (void)node2;
    return true;
  }
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_FUSION_DECIDER_H_