/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimize/task_generator/concat_inputs_unification_pass.h"

#include "ascir_utils.h"
#include "graph_utils.h"
#include "schedule_utils.h"
#include "buffer_allocate/tensor_mem_defs.h"

namespace optimize {

Status ConcatInputUnificationPass::Run(std::vector<ascir::ImplGraph> &graphs) {
  for (auto &graph : graphs) {
    GE_ASSERT_SUCCESS(RunOneGraph(graph));
  }
  return ge::SUCCESS;
}

Status ConcatInputUnificationPass::RunOneGraph(ascir::ImplGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (ge::ops::IsOps<ge::ascir_op::Concat>(node)) {
      const auto need_optimize = NeedOptimize(node);
      GELOGD("graph: %s, node %s need optimize = %d", graph.GetName().c_str(), node->GetNamePtr(),
             static_cast<int32_t>(need_optimize));
      if (need_optimize) {
        GE_ASSERT_SUCCESS(DoOptimize(graph, node));
      }
    }
  }
  return ge::SUCCESS;
}

bool ConcatInputUnificationPass::NeedOptimize(const ge::AscNodePtr &concat_node) {
  GE_WARN_ASSERT(concat_node->inputs.Size() > 0);
  // 1. 输入shape相同
  if (ascir::utils::AreConcatInputShapesEqual(concat_node) == ge::TriBool::kFalse) {
    GELOGI("input shapes of Concat differ, no need for optimization");
    return false;
  }

  // 2. 首轴concat不需要
  size_t concat_dim;
  bool is_first_dim = false;
  GE_CHK_STATUS_RET(ScheduleUtils::ResolveDiffDim(concat_node, concat_dim, is_first_dim), "ResolveConcatDim failed");
  GE_CHK_BOOL_RET_SPECIAL_STATUS(is_first_dim, false, "concat on the first dim, no need for optimization");

  // 3. 输入对齐到4B不需要(Scatter在输入未对齐到4B时性能才会劣化)
  const auto dtype_size = ge::GetSizeByDataType(concat_node->outputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0, "unsupported output data type");
  GELOGI("input repeat = %s, output repeat = %s, concat_dim = %zu, dtype_size = %d",
         ge::ToString(concat_node->inputs[0].attr.repeats).c_str(),
         ge::ToString(concat_node->outputs[0].attr.repeats).c_str(), concat_dim, dtype_size);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(IsSrcColSizeAlignedToB4(concat_node, concat_dim, dtype_size), false,
                                 "src col size aligned to B32, no need for optimization");

  // 4. dst_col_size超过阈值不需要
  GE_CHK_BOOL_RET_SPECIAL_STATUS(IsDstColSizeOverLimit(concat_node, concat_dim, dtype_size), false,
                                 "dst col size over limit, no need for optimization");

  // 5. 输入不能共用
  GE_CHK_BOOL_RET_SPECIAL_STATUS((!ascir::utils::AreAllInputDistinct(concat_node)), false,
                                 "contain multi-ref input, do not optimize");

  // 6. 输入全来自于Load不需要
  uint32_t load_num = 0;
  GE_WARN_ASSERT(GetLoadNum(concat_node, load_num) == ge::SUCCESS);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(load_num == concat_node->inputs.Size(), false,
                                 "All inputs are of compute type Load, no need for optimization");
  // 7. 需要增加的Ub2Ub节点数不能超过阈值
  constexpr uint32_t kCopyNumLimit = 3U;
  GE_CHK_BOOL_RET_SPECIAL_STATUS(load_num > kCopyNumLimit, false, "Load num = %zu, over limit = %zu, do not optimize",
                                 load_num, kCopyNumLimit);
  return true;
}

Status ConcatInputUnificationPass::DoOptimize(ascir::ImplGraph &graph, const ge::AscNodePtr &concat_node) {
  for (const auto &in_anchor : concat_node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_anchor);
    const auto in_node = out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(in_node);
    const auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(in_node);
    GE_ASSERT_NOTNULL(asc_node);
    std::vector<int64_t> no_reuse_output_indices{out_anchor->GetIdx()};
    if (asc_node->attr.api.compute_type != ge::ComputeType::kComputeLoad) {
      (void)ge::AttrUtils::SetListInt(in_node->GetOpDesc(), kAttrNameNoReuseOutputIndices, no_reuse_output_indices);
      continue;
    }

    const std::string ub_name = "ub_cpy_" + asc_node->GetName();
    ge::ascir_op::Ub2ub ub2ub(ub_name.c_str());
    ge::AscNodePtr ub2ub_node = graph.AddNode(ub2ub);
    GE_ASSERT_NOTNULL(ub2ub_node);
    ub2ub_node->attr.sched = asc_node->attr.sched;
    ub2ub_node->attr.api.compute_type = ge::ComputeType::kComputeElewise;
    ub2ub_node->attr.api.type = ge::ApiType::kAPITypeCompute;
    ub2ub_node->attr.api.unit = ge::ComputeUnit::kUnitVector;
    ub2ub_node->outputs[0].attr = asc_node->outputs[0].attr;
    ub2ub_node->outputs[0].attr.buf = {};
    ub2ub_node->outputs[0].attr.que = {};

    GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(out_anchor, in_anchor));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(ub2ub_node->GetOutDataAnchor(0), in_anchor));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(out_anchor, ub2ub_node->GetInDataAnchor(0)));
    (void)ge::AttrUtils::SetListInt(ub2ub_node->GetOpDesc(), kAttrNameNoReuseOutputIndices, std::vector<int64_t>{0});
    GELOGD("Ub2ub node: %s added", ub2ub_node->GetNamePtr());
  }
  return ge::SUCCESS;
}

ge::Expression ConcatInputUnificationPass::GetColSize(const ge::AscTensor &tensor, size_t concat_dim) {
  const auto &output_repeats = tensor.attr.repeats;
  ge::Expression dst_col_size = output_repeats[concat_dim];
  for (size_t i = concat_dim + 1; i < output_repeats.size(); ++i) {
    dst_col_size = dst_col_size * output_repeats[i];
  }
  return dst_col_size;
}

ge::Status ConcatInputUnificationPass::GetLoadNum(const ge::AscNodePtr &concat_node, uint32_t &load_num) {
  for (const auto &in_node : concat_node->GetInDataNodes()) {
    const auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(in_node);
    GE_WARN_ASSERT(asc_node);
    if (asc_node->attr.api.compute_type == ge::ComputeType::kComputeLoad) {
      ++load_num;
    }
  }
  return ge::SUCCESS;
}

bool ConcatInputUnificationPass::IsSrcColSizeAlignedToB4(const ge::AscNodePtr &concat_node, size_t concat_dim,
                                                         int32_t dtype_size) {
  const auto src_col_size_expr = GetColSize(concat_node->inputs[0], concat_dim);
  const auto aligned =
      (ge::sym::Mod((src_col_size_expr * ge::Symbol(dtype_size)), ge::Symbol(sizeof(uint32_t))) == ge::ops::Zero);
  return aligned;
}

bool ConcatInputUnificationPass::IsDstColSizeOverLimit(const ge::AscNodePtr &concat_node, size_t concat_dim,
                                                       int32_t dtype_size) {
  const auto dst_col_size_expr = GetColSize(concat_node->outputs[0], concat_dim);
  if (!dst_col_size_expr.IsConstExpr()) {
    return false;
  }
  int64_t dst_col_size = -1;
  GE_WARN_ASSERT(dst_col_size_expr.GetConstValue(dst_col_size));
  constexpr int64_t kDstColSizeLimit = 256;
  GELOGI("dst_col_size = %ld", dst_col_size);
  return (dst_col_size * dtype_size) > kDstColSizeLimit;
}
}  // optimize