/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCHEDULE_CASE_GENERATOR_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCHEDULE_CASE_GENERATOR_H_

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"
#include "optimize/task_generator/schedule_case_generator.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
struct ExpressionStaticCheckEq {
  // cooncat子图表达式可能不一致, 依赖guard
  bool operator()(const ge::Expression &lhs, const ge::Expression &rhs) const {
    return ge::SymbolicUtils::StaticCheckEq(lhs, rhs) == ge::TriBool::kTrue;
  }
};
using ConcatDimAxisMap = std::unordered_map<ge::Expression, ge::AxisId, ge::ExpressionHash, ExpressionStaticCheckEq>;
class ConcatFusionCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate(ascir::HintGraph &graph,
                  std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;
  ConcatFusionCaseGenerator& SetConvertToStoreMode();
  [[nodiscard]] bool HasLoadStoreConversion() const override{
    return true;
  }

 private:
  Status AddTemplateForSplitConcat(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs);
  static Status AddTemplateForSmallTail(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs);
  bool NeedDynSmallTailTemplate(const ge::AscNodePtr &concat_node) const;
  Status GenerateScoreFunctions(const std::vector<ascir::ImplGraph> &graphs,
                                size_t concat_dim,
                                std::vector<std::string> &score_functions) const;
  static std::vector<ge::AscNodePtr> FindConcatNodes(const ascir::HintGraph &owner_graph);
  Status ConvertConcatToStores(ascir::HintGraph &owner_graph, const ge::AscNodePtr &concat_node);
  Status SplitConcats(ascir::HintGraph &owner_graph, const ge::AscNodePtr &concat_node, bool &split);
  Status Prepare(const ge::AscNodePtr &concat_node, size_t concat_dim);
  Status ReplaceWithStore(const ge::AscNodePtr &concat_node, const ge::InDataAnchorPtr &concat_in_anchor,
                          const ge::Axis &replace_axis);
  Status ConvertSingleInput(ascir::HintGraph &owner_graph, const ge::AscNodePtr &concat_node, size_t in_index,
                            size_t group_idx, ConcatDimAxisMap &repeat_to_axis_id);
  Status PropagateAxisChanges(ge::Node *start_node, const std::vector<ascir::AxisId> &new_axis_ids) const;
  Status ReplaceWithConcat(::ascir::ImplGraph &owner_graph,
                           const ge::AscNodePtr &concat_node,
                           size_t start,
                           size_t end);
  static Status RemoveUnusedNodes(const ge::AscNodePtr &concat_node, const std::vector<ge::AscNodePtr> &nodes = {});
  static Status SplitDataForDifferentConcatDim(ascir::ImplGraph &owner_graph);
  static ge::Status SetConcatOpAttr(ge::ascir_op::Concat &concat_op,
                                    const ge::AscNodePtr &concat_node,
                                    size_t concat_dim,
                                    size_t start,
                                    size_t end);
  static Status CollectBackwardNodes(const ge::NodePtr &concat_node, std::vector<ge::AscNodePtr> &nodes);
  static Status CollectReachableLoadNodes(const ge::NodePtr &concat_node, std::set<ge::AscNodePtr> &nodes);
  Status CloneNonConcatNodes(const ge::Axis &new_axis, size_t index,
                             std::vector<ge::InDataAnchorPtr> &in_anchors,
                             const std::vector<ascir::AxisId> &new_axis_ids,
                             std::unordered_map<std::string, ge::NodePtr> &name_to_new_node);
  static ge::Status ReplaceAxis(const ge::AscNodePtr &node, size_t axis_index, const ge::Axis &to_axis,
                                const std::vector<ascir::AxisId> &new_axis_ids);
  static ge::Status UpdateRepeatAndStrides(const ge::AscNodePtr &node, size_t axis_index,
                                           const ge::Expression &axis_size, ge::AscTensorAttr &tensor_attr);
  static Status InsertAxis(ascir::ImplGraph &optimized_graph);
  static Status AddTemplateIfCanFitInOneKernel(const ge::AscNodePtr &concat_node, ascir::HintGraph &graph,
                                               std::vector<ascir::ImplGraph> &graphs);
  static Status MarkNoMergeFirstAxis(std::vector<ascir::ImplGraph> &graphs);
  bool KeepOriginGraph(const ge::AscNodePtr &concat_node) const;
  static bool IsSmallBlock(const ge::AscNodePtr &concat_node, size_t concat_dim);
  static Status ReconnectIfShareSameAncestor(const std::unordered_map<std::string, ge::NodePtr> &name_to_node, const ge::InDataAnchorPtr &in_anchor);
  static Status AddExtraShapeEnv(const ge::AscNodePtr &concat_node, size_t concat_dim);
  Status PrepareForModifyingGraph(const ge::AscNodePtr &concat_node);

  std::vector<ge::AscNodePtr> post_concat_nodes_;
  std::set<ge::AscNodePtr> reachable_load_nodes_;;
  std::map<std::string, std::vector<int32_t>> out_node_name_to_indices_;
  std::vector<ge::Expression> concat_dim_offsets_;
  ascir::AxisId concat_axis_id_ = -1;
  size_t concat_dim_ = std::numeric_limits<size_t>::max();
  bool support_small_tail_ = false;
  bool convert_to_store_ = false;
  bool split_concat_ = false;
  bool has_recompute_ = false;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_SCHEDULE_CASE_GENERATOR_H_
