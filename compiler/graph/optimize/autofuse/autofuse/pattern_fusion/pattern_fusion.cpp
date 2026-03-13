/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pattern_fusion.h"
#include "utils/auto_fuse_config.h"
#include "utils/autofuse_attrs.h"
#include "common/checker.h"
#include "decompose_large_const_pass.h"
#include "flatten_concat_pass.h"
#include "flatten_split_pass.h"
#include "cascade_reshape_remove_pass.h"
#include "redundant_slice_remove_pass.h"
#include "gather_forward_fusion_pass.h"
#include "pad_slice_optimize_pass.h"
#include "concat_slice_simplification_pass.h"
#include "slice_forward_fusion_pass.h"
#include "transpose_with_broadcast_eliminate_pass.h"
#include "cast_remove_pass.h"

namespace ge {
constexpr uint32_t kMaxIterations = 3U;

// 在符号化推导之前执行的 Pass，不需要符号化信息、不依赖 lowering
graphStatus PatternFusion::RunEarlyPasses(const ComputeGraphPtr &graph, const GraphPasses &graph_passes) {
  GE_ASSERT_NOTNULL(graph);

  bool changed = true;
  uint32_t iter = 0U;
  while (changed && iter < kMaxIterations) {
    changed = false;
    iter++;
    GE_ASSERT_GRAPH_SUCCESS(PadSliceOptimizePass().Run(graph, changed));
    GE_ASSERT_GRAPH_SUCCESS(RedundantSliceRemovePass().Run(graph, changed));
    GE_ASSERT_GRAPH_SUCCESS(CascadeReshapeRemovePass().Run(graph, changed));
    GE_ASSERT_GRAPH_SUCCESS(TransposeWithBroadcastEliminatePass().Run(graph, changed));
    GE_ASSERT_GRAPH_SUCCESS(CastRemovePass().Run(graph, changed));
    GE_ASSERT_GRAPH_SUCCESS(ConcatSliceSimplificationPass().Run(graph, graph_passes, changed));
  }
  GE_ASSERT_GRAPH_SUCCESS(graph->TopologicalSorting());

  return GRAPH_SUCCESS;
}

graphStatus PatternFusion::RunAllPatternFusion(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  // ============ 融合类pass，先统一只调用一次 ============
  GE_ASSERT_GRAPH_SUCCESS(SliceForwardFusionPass().Run(graph));
  GE_ASSERT_GRAPH_SUCCESS(FlattenConcatPass().Run(graph));
  GE_ASSERT_GRAPH_SUCCESS(FlattenSplitPass().Run(graph));
  GE_ASSERT_GRAPH_SUCCESS(GatherForwardFusionPass().Run(graph));
  GE_ASSERT_GRAPH_SUCCESS(DecomposeLargeConstPass::Run(graph));
  return graph->TopologicalSorting();
}
} // namespace ge
