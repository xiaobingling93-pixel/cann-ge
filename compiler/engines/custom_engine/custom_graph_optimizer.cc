/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_graph_optimizer.h"
#include "common/ge_common/ge_types.h"
#include "graph/utils/attr_utils.h"
#include "common/checker.h"

namespace ge {
CustomGraphOptimizer::~CustomGraphOptimizer() = default;

Status CustomGraphOptimizer::Initialize(const std::map<std::string, std::string> &options,
    ge::OptimizeUtility *const optimize_utility) {
  (void)options;
  (void)optimize_utility;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::Finalize() {
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph) {
  (void)graph;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph &graph) {
  (void)graph;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph &graph) {
  (void)graph;
  return SUCCESS;
}

ge::Status CustomGraphOptimizer::GetAttributes(ge::GraphOptimizerAttribute &attrs) const {
  attrs.engineName = kEngineNameCustom;
  return SUCCESS;
}

} // namespace ge
