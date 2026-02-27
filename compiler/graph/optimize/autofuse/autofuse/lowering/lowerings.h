/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_LOWERING_LOWERINGS_H_
#define AUTOFUSE_LOWERING_LOWERINGS_H_

#include <functional>

#include "graph/node.h"
#include "asc_lowerer/loop_api.h"
#include "lowering_utils.h"

namespace ge {
constexpr LoweringConfig kLoweringConfig;

struct AscBackendFuseConfig {
  size_t min_ascend_ir_nodes = 1U;
};
constexpr AscBackendFuseConfig kAscBackendFuseConfig;

const std::string kAscBackend = "AscBackend";
const std::string kAscBackendNoKernelOp = "AscBackendNoKernelOp";

class LoweringManager {
 public:
  static graphStatus Lowering(const NodePtr &node);
  static graphStatus LoweringGraph(const ComputeGraphPtr &graph, const LoweringConfig &config = kLoweringConfig);

  static graphStatus FusedLoopToAscBackendOp(const ComputeGraphPtr &graph,
                                             const AscBackendFuseConfig &config = kAscBackendFuseConfig, CounterPtr counter = nullptr);

  static graphStatus GetFusedOriginComputeGraph(const AutoFuseAttrs &attrs, const NodePtr &node);

  [[nodiscard]] bool IsLoweringRegistered(const std::string &op_type) const;
  static void Register(const std::string &op_type, const std::function<graphStatus(const NodePtr &)> &lower);

 private:
  LoweringManager() = default;
  ~LoweringManager() = default;

  static LoweringManager &Instance();
  void RegisterImpl(const std::string &op_type, const std::function<graphStatus(const NodePtr &)> &lower);
  graphStatus LowerImpl(const NodePtr &node);

  static graphStatus PostPrecessAfterLoweringNode(const NodePtr &node, const LoweringConfig &config);

  static OpDescPtr BuildOpDescForKernelBox(loop::KernelBox &kernel_box, std::vector<const ge::OutDataAnchor *> &origin_inputs, 
                                           CounterPtr counter);

  static graphStatus FusedSubgraphLoopToAscBackendOp(
    const ComputeGraphPtr &graph, const AscBackendFuseConfig &config,
    std::map<const ge::OutDataAnchor *, ge::OutDataAnchor *> &ascend_out_to_asc_out, CounterPtr counter);

  std::map<std::string, std::function<graphStatus(const NodePtr &)>> lowerings_;
};

}  // namespace ge

#endif  // AUTOFUSE_LOWERING_LOWERINGS_H_
