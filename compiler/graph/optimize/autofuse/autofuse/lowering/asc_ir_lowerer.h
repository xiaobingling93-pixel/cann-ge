/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_LOWERING_ASC_IR_LOWERER_H_
#define AUTOFUSE_LOWERING_ASC_IR_LOWERER_H_


#include "graph/node.h"
#include "asc_lowerer/loop_api.h"
#include "autofuse_frame/autofuse_frames.h"

namespace ge {
class AscIrLowerer {
 public:
  explicit AscIrLowerer(CounterPtr counter = nullptr) : counter_(counter) {}
  graphStatus Lowering(const ComputeGraphPtr &graph);
  graphStatus Lifting(const ComputeGraphPtr &graph);

 private:
  bool do_lowered_ = false;
  graphStatus RemoveDirectNodeUnusedEdges(const ComputeGraphPtr &graph);
  graphStatus PruneUnusedNodesAfterLifting(const ge::ComputeGraphPtr &graph) const;
  graphStatus DfxForAscBackendOp(const ComputeGraphPtr &graph) const;

  graphStatus ProcessControlEdge(const ComputeGraphPtr &graph);
  graphStatus RecoverInitControlEdge(const ComputeGraphPtr &graph);
  std::set<NodePtr> replaced_nodes_;
  CounterPtr counter_ = nullptr;
  ComputeGraphPtr pre_lowering_graph_;  // Lowering 前的原始图深拷贝，供落盘子图使用
  std::map<std::string, std::vector<NodePtr>> node_in_control_to_const_;
  std::map<std::string, std::vector<NodePtr>> node_out_control_to_const_;
};
}  // namespace ge

#endif  // AUTOFUSE_LOWERING_ASC_IR_LOWERER_H_
