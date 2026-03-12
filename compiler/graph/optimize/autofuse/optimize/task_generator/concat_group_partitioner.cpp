/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimize/task_generator/concat_group_partitioner.h"

#include <queue>

#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"

#include "ascir_ops.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "platform/platform_factory.h"

namespace optimize {
namespace {
constexpr int32_t kAlignment = 32;
constexpr int32_t kConcatAlgTranspose = 0;
constexpr int64_t kVectorBlockSize = 256;
constexpr int64_t kMinGroupNum = 2L;
constexpr int64_t kMinGroupSizeByte = 1024U;

template <typename T1, typename T2>
int64_t CeilDiv(const T1 n1, const T2 n2) {
  GE_ASSERT_TRUE(n1 != 0);
  GE_ASSERT_TRUE(n2 != 0);
  return ((static_cast<int64_t>(n1) - 1) / static_cast<int64_t>(n2)) + 1;
}
}  // namespace

ConcatGroupPartitioner::ConcatGroupPartitioner(ge::AscNodePtr concat_node, size_t concat_dim)
    : concat_node_(std::move(concat_node)), concat_dim_(concat_dim) {
}

Status ConcatGroupPartitioner::Initialize() {
  const auto backend_spec = BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  concat_by_transpose_ = (backend_spec->concat_alg == kConcatAlgTranspose);
  default_cols_per_group_ = kMaxBlockSize / dtype_size_;
  max_input_num_per_group_ = MaxInputNumPerGroup();
  // row足够大时仅row就足够分核, 不需要group parallel，否则尝试分组
  if (known_rows_ <= kGroupParallelRowThreshold) {
    const auto new_size = static_cast<uint32_t>(CeilDiv(concat_dim_sizes_.size(), kMinGroupNum));
    max_input_num_per_group_ = std::min(max_input_num_per_group_, new_size);
  }
  const auto &output_attr = concat_node_->outputs[0].attr;
  if (concat_by_transpose_) {
    const auto is_tail_concat = (concat_dim_ == output_attr.repeats.size() - 1UL);
    can_use_small_tail_ = is_tail_concat && (dtype_size_ == sizeof(uint16_t) || dtype_size_ == sizeof(uint32_t));
    group_type_to_limit_[kGroupTypeSmallTail] = kMaxBlockSizeForSmallTail;
  } else {
    if (output_cols_ > 0) {
      GE_ASSERT_SUCCESS(TryOptimizeGroupSize());
    }
    is_concat_scalar_ = (total_rows_ == 1);
  }
  GELOGI("output_repeat = %s, concat_dim = %zu, input_num = %u, max_input_num_per_group = %u",
         ge::ToString(output_attr.repeats).c_str(), concat_dim_, concat_node_->inputs.Size(), max_input_num_per_group_);
  return ge::SUCCESS;
}

Status ConcatGroupPartitioner::PartitionGroups(std::vector<ConcatGroup> &groups) {
  GE_ASSERT_SUCCESS(ParseConcatNode());
  GE_CHK_BOOL_RET_SPECIAL_STATUS((stride_ == -1), ge::SUCCESS, "contains non-static dim after concat dim");
  GE_ASSERT_SUCCESS(Initialize());
  const auto &all_in_data_anchors = concat_node_->GetAllInDataAnchorsPtr();
  for (size_t i = 0UL; i < all_in_data_anchors.size(); ++i) {
    const int64_t size = concat_dim_sizes_[i];
    const auto new_group_type = GetGroupType(size);
    // 需要单独一组的case: 动态, 或存在输出多引用, 或大小超过阈值
    if ((size < 0) || ((size * dtype_size_) > kGroupEltSizeThreshold) || (new_group_type == kGroupTypeNone)) {
      if (index_start_ != -1) {
        GroupEnd(i);
      }
      groups_.emplace_back(ConcatGroup{i, i + 1, kGroupTypeNone, -1});
      continue;
    }
    if (index_start_ == -1) {
      GroupStart(static_cast<int64_t>(i), new_group_type, size);
      continue;
    }
    if (NeedSubmit(i, size, new_group_type)) {
      GroupEnd(i);
      GroupStart(static_cast<int64_t>(i), new_group_type, size);
    } else {
      UpdateStatus(size);
    }
  }
  if (index_start_ != -1) {
    GroupEnd(all_in_data_anchors.size());
  }
  MergeSmallGroups();
  ConvertToDefaultIfTooSmall();
  MergeSmallGroups();
  if ((groups_.size() > 1) && (groups_.size() != concat_node_->inputs.Size())) {
    GE_ASSERT_SUCCESS(RecomputeNodesCrossGroups(false, has_recompute_));
    GE_ASSERT_SUCCESS(RecomputeNodesCrossGroups(true, has_recompute_));
  }
  groups = std::move(groups_);
  return ge::SUCCESS;
}

bool ConcatGroupPartitioner::HasRecompute() const {
  return has_recompute_;
}

bool ConcatGroupPartitioner::NeedSubmit(size_t i, int64_t size, uint32_t new_group_type) {
  // 以下场景需要提交当前group: 1. 超过size上限, 2. 超过数量上限
  if (((cur_size_ + size) > GetSizeLimitByGroupType(group_type_)) ||
      (i - index_start_ >= max_input_num_per_group_)) {
    GELOGD("Size limit(%ld) or number limit(%u) reached, index = %zu, size = %ld, num = %zu",
           GetSizeLimitByGroupType(group_type_), max_input_num_per_group_,
           i, cur_size_ + size, i - index_start_);
    return true;
  }
  if (new_group_type == group_type_) {
    return false;
  }

  // 检查group type是否相容
  auto merged_group_type = group_type_ & new_group_type;
  if (merged_group_type == 0U) {
    // 防止过小的group
    if ((i - index_start_ == 1) && (group_type_ != kGroupTypeDefault) && (group_type_ != kGroupTypeScalar)) {
      GELOGD("group has only one element, convert to %s", GroupTypeToString(new_group_type).c_str());
      group_type_ = kGroupTypeDefault;
      return false;
    }
    GELOGD("Group type changed, index = %zu, group_type = [%s], new_group_type = [%s]",
           i, GroupTypeToString(group_type_).c_str(), GroupTypeToString(new_group_type).c_str());
    return true;
  }
  // small_and_algin & small -> small时，需要检查size是否会超过
  if ((merged_group_type == kGroupTypeSmallTail) && (cur_size_ + size) > GetSizeLimitByGroupType(kGroupTypeSmallTail)) {
    GELOGD("Size limit(%ld) reached, index = %zu, size = %ld",
           GetSizeLimitByGroupType(kGroupTypeSmallTail), i, cur_size_ + size);
    return true;
  }
  group_type_ = merged_group_type;
  return false;
}

void ConcatGroupPartitioner::UpdateStatus(int64_t size) {
  cur_size_ += size;
  if ((group_type_ == kGroupTypeSmallTailAndAligned) && (cur_size_ >= GetSizeLimitByGroupType(kGroupTypeSmallTail))) {
    GELOGD("size(%ld) >= size limit(%ld), concat type from [AlignAndSmallTail] to [Aligned]",
           cur_size_, GetSizeLimitByGroupType(kGroupTypeSmallTail));
    group_type_ = kGroupTypeAligned;
  }
}

void ConcatGroupPartitioner::GroupStart(int64_t index_start, uint32_t group_type, int64_t size) {
  index_start_ = index_start;
  group_type_ = group_type;
  cur_size_ = size;
  GELOGD("group start, index = %zu, type = %s", index_start, GroupTypeToString(group_type).c_str());
}

void ConcatGroupPartitioner::GroupEnd(size_t index_end) {
  GELOGD("group end, start_index = %zu, end_index = %zu, type = [%s], size = %ld",
         index_start_, index_end, GroupTypeToString(group_type_).c_str(), cur_size_);
  groups_.emplace_back(ConcatGroup{static_cast<size_t>(index_start_), index_end, group_type_, cur_size_});
  index_start_ = -1;
  group_type_ = -1;
}

int64_t ConcatGroupPartitioner::GetSizeLimitByGroupType(uint32_t group_type) const {
  const auto it = group_type_to_limit_.find(group_type);
  return (it != group_type_to_limit_.end()) ? it->second : default_cols_per_group_;
}

uint32_t ConcatGroupPartitioner::GetGroupType(int64_t size) const {
  if (is_concat_scalar_) {
    return (size <= (kVectorBlockSize / dtype_size_)) ? kGroupTypeScalar : kGroupTypeNone;
  }
  if (use_default_group_) {
    return kGroupTypeDefault;
  }
  bool aligned = ((size * dtype_size_) % kAlignment) == 0;
  bool is_small_tail = (can_use_small_tail_ && (size <= kSmallTailLimit));
  if (aligned && is_small_tail) {
    return kGroupTypeSmallTailAndAligned;
  }
  if (aligned) {
    return kGroupTypeAligned;
  }
  if (is_small_tail) {
    return kGroupTypeSmallTail;
  }
  return kGroupTypeDefault;
}

void ConcatGroupPartitioner::MergeGroups(std::vector<ConcatGroup>::value_type &lhs_group,
                                         std::vector<ConcatGroup>::value_type &rhs_group) {
  rhs_group.start = lhs_group.start;
  rhs_group.size += lhs_group.size;
  lhs_group.size = 0;
  lhs_group.end = lhs_group.start;
}

void ConcatGroupPartitioner::MergeSmallGroups() {
  std::vector<ConcatGroup> groups;
  // 将过小的group转为default, 供后续合并
  for (size_t index = 0UL; index < groups_.size() - 1UL; ++index) {
    auto &lhs_group = groups_[index];
    auto &rhs_group = groups_[index + 1];
    if (CanMerge(lhs_group, rhs_group)) {
      // same mergeable group type
      // align + default -> default
      if (((lhs_group.group_type != kGroupTypeNone) && (lhs_group.group_type == rhs_group.group_type)) ||
          (IsAligned(lhs_group.group_type) && (rhs_group.group_type == kGroupTypeDefault))) {
        MergeGroups(lhs_group, rhs_group);
      }
    }
  }
  for (const auto &group : groups_) {
    if (group.start != group.end) {
      groups.emplace_back(group);
    }
  }
  groups_ = std::move(groups);
}

bool ConcatGroupPartitioner::CanMerge(const ConcatGroupPartitioner::ConcatGroup &lhs,
                                      const ConcatGroupPartitioner::ConcatGroup &rhs) const {
  auto total_num = (lhs.end - lhs.start) + (rhs.end - rhs.start);
  auto any_group_has_single_item = (lhs.end - lhs.start == 1) || (rhs.end - rhs.start == 1);
  return (any_group_has_single_item || (total_num <= max_input_num_per_group_)) &&
         ((lhs.size + rhs.size) <= kMaxBlockSize / dtype_size_);
}

void ConcatGroupPartitioner::ConvertToDefaultIfTooSmall() {
  for (auto &group : groups_) {
    if ((group.end - group.start == 1) &&
        (IsSmallTail(group.group_type) || ((!concat_by_transpose_) && IsAligned(group.group_type)))) {
      GELOGD("group start with index = %zu, size = 1, concat type from [%s] to [Default]", group.start,
             GroupTypeToString(group.group_type).c_str());
      group.group_type = kGroupTypeDefault;
    }
  }
}

std::string ConcatGroupPartitioner::GroupTypeToString(uint32_t group_type) {
  static const std::map<uint32_t, std::string> kGroupTypeToStr {
      {ConcatGroupPartitioner::kGroupTypeDefault, "Default"},
      {ConcatGroupPartitioner::kGroupTypeAligned, "Aligned"},
      {ConcatGroupPartitioner::kGroupTypeSmallTailAndAligned, "AlignAndSmallTail"},
      {ConcatGroupPartitioner::kGroupTypeSmallTail, "SmallTail"},
      {ConcatGroupPartitioner::kGroupTypeScalar, "Scalar"},
  };
  return kGroupTypeToStr.at(group_type);
}

bool ConcatGroupPartitioner::IsAligned(uint32_t group_type) {
  return (group_type & kGroupTypeAligned) != 0U;
}

bool ConcatGroupPartitioner::IsSmallTail(uint32_t group_type) {
  return (group_type & kGroupTypeSmallTail) != 0U;
}

ge::Status ConcatGroupPartitioner::RecomputeNodesCrossGroups(bool search_backward, bool &has_recompute) const {
  for (const auto &group : groups_) {
    std::set<ge::InDataAnchorPtr> visited_in_anchors;
    std::map<std::string, ge::AscNodePtr> name_to_new_node;
    for (size_t i = group.start; i < group.end; ++i) {
      GELOGD("input[%zu] check recompute start", i);
      auto const in_anchor = concat_node_->GetInDataAnchor(static_cast<int32_t>(i));
      int32_t depth = 1024;
      while (--depth >= 0) {
        ge::InDataAnchorPtr to_split;
        GE_ASSERT_SUCCESS(
            FindFirstMultiOutputAnchors(in_anchor, group.end, search_backward, visited_in_anchors, to_split));
        if (to_split == nullptr) {
          break;
        }
        GE_ASSERT_SUCCESS(RecomputeInNodes(to_split, i, name_to_new_node));
        has_recompute = true;
      }
      GE_ASSERT_TRUE(depth >= 0);
    }
  }
  return ge::SUCCESS;
}

ge::Status ConcatGroupPartitioner::FindFirstMultiOutputAnchors(const ge::InDataAnchorPtr &in_anchor, int32_t end_index,
                                                               bool search_backward,
                                                               std::set<ge::InDataAnchorPtr> &visited_in_anchors,
                                                               ge::InDataAnchorPtr &to_split) const {
  std::vector<const ge::Node *> root_nodes;
  std::queue<ge::InDataAnchorPtr> in_anchors;
  in_anchors.push(in_anchor);
  const auto &concat_dim_size = concat_node_->inputs[in_anchor->GetIdx()].attr.repeats[concat_dim_];
  std::set<ge::NodePtr> visited;
  while (!in_anchors.empty()) {
    const auto cur_in_anchor = in_anchors.front();
    in_anchors.pop();
    auto out_anchor = cur_in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    auto owner_node = out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(owner_node);
    if (visited_in_anchors.emplace(cur_in_anchor).second) {
      std::set<ge::NodePtr> out_nodes;
      for (const auto &out_node : owner_node->GetOutDataNodes()) {
        out_nodes.emplace(out_node);
      }
      if (out_nodes.size() > 1UL ||
          (((*out_nodes.begin())->GetType() == "Concat") && (out_anchor->GetPeerAnchorsSize() > 1UL))) {
        bool need_split = false;
        GE_ASSERT_SUCCESS(CheckIsAncestorOfConcat(out_anchor, end_index, concat_dim_size, search_backward, need_split));
        GELOGD("%s has multi-ref output, end_index = %d, need_split = %d", out_anchor->GetOwnerNode()->GetNamePtr(),
               end_index, need_split);
        if (need_split) {
          to_split = cur_in_anchor;
          return ge::SUCCESS;
        }
      }
    }
    for (const auto &in_data_anchor : owner_node->GetAllInDataAnchors()) {
      if (in_data_anchor != nullptr) {
        in_anchors.push(in_data_anchor);
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status ConcatGroupPartitioner::CheckIsAncestorOfConcat(const ge::OutDataAnchorPtr &out_anchor, int32_t start_index,
                                                           const ge::Expression &concat_dim_size,
                                                           bool search_backward,
                                                           bool &need_split) const {
  std::vector<const ge::Node *> nodes;
  std::set<const ge::Node *> visited;
  visited.emplace(concat_node_.get());
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    const auto owner_node = peer_in_anchor->GetOwnerNode().get();
    GE_ASSERT_NOTNULL(owner_node);
    if ((owner_node == concat_node_.get()) && NeedSplit(peer_in_anchor, start_index, concat_dim_size)) {
      need_split = true;
      return ge::SUCCESS;
    }
    if (visited.emplace(owner_node).second) {
      nodes.emplace_back(owner_node);
    }
  }
  while (!nodes.empty()) {
    const auto cur_node = nodes.back();
    nodes.pop_back();
    for (const auto &[out_node, in_anchor] : cur_node->GetOutDataNodesAndAnchors()) {
      if (out_node == concat_node_) {
        if (NeedSplit(in_anchor, start_index, concat_dim_size)) {
          need_split = true;
          return ge::SUCCESS;
        }
      } else {
        if (visited.emplace(out_node.get()).second) {
          nodes.emplace_back(out_node.get());
        }
      }
    }
    if (search_backward) {
      for (const auto &in_node: cur_node->GetInDataNodes()) {
        if (visited.emplace(in_node.get()).second) {
          nodes.emplace_back(in_node.get());
        }
      }
    }
  }
  return ge::SUCCESS;
}

bool ConcatGroupPartitioner::NeedSplit(const ge::InDataAnchorPtr &in_anchor, int32_t start_index,
                                       const ge::Expression &cur_dim_size) const {
  const auto &size = concat_node_->inputs[in_anchor->GetIdx()].attr.repeats[concat_dim_];
  const auto cur_group_size = GetGroupSize(static_cast<size_t>(start_index - 1));
  const auto next_group_size = GetGroupSize(static_cast<size_t>(in_anchor->GetIdx()));
  const auto is_single_group = (cur_group_size == 1) && (next_group_size == 1);
  const auto need_split =
      ((in_anchor->GetIdx() >= start_index) &&
       ((!is_single_group) || ge::SymbolicUtils::StaticCheckEq(size, cur_dim_size) != ge::TriBool::kTrue));
  GELOGD("start_index = %d, next_index = %d, is_single_group = %d, cur_size = %s, next_size = %s, need_split = %d",
         start_index, in_anchor->GetIdx(), static_cast<int32_t>(is_single_group),
         ge::SymbolicUtils::ToString(cur_dim_size).c_str(), ge::SymbolicUtils::ToString(size).c_str(),
         static_cast<int32_t>(need_split));
  return need_split;
}

size_t ConcatGroupPartitioner::GetGroupSize(size_t index) const {
  for (const auto &group : groups_) {
    if ((group.start <= index) && (index < group.end)) {
      return (group.end - group.start);
    }
  }
  return 0UL;
}

ge::Status ConcatGroupPartitioner::RecomputeInNodes(const ge::InDataAnchorPtr &in_anchor, size_t index,
                                                    std::map<std::string, ge::AscNodePtr> &name_to_new_nodes) const {
  ascir::ImplGraph owner_graph("");
  GE_ASSERT_SUCCESS(ge::AscGraphUtils::FromComputeGraph(concat_node_->GetOwnerComputeGraph(), owner_graph));
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(asc_node);
  ge::AscNodePtr &dst_new_node = name_to_new_nodes[asc_node->GetName()];
  if (dst_new_node == nullptr) {
    GELOGD("concat input index = %zu, ancestor node %s multi-ref output, re-compute it", index, asc_node->GetNamePtr());
    GE_ASSERT_EQ(asc_node->GetAllOutDataAnchorsSize(), 1U);
    const auto &op_desc = ge::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetName(asc_node->GetName() + "_recompute_" + std::to_string(index));
    ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    dst_new_node = owner_graph.AddNode(op);
    GE_ASSERT_TRUE(ge::AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                 "DoCopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
    // restore input edges
    for (const auto &in_data_anchor : asc_node->GetAllInDataAnchorsPtr()) {
      if (in_data_anchor != nullptr) {
        const auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        if (peer_out_anchor != nullptr) {
          GE_ASSERT_GRAPH_SUCCESS(
              ge::GraphUtils::AddEdge(peer_out_anchor, dst_new_node->GetInDataAnchor(in_data_anchor->GetIdx())));
        }
      }
    }
  }
  // replace output edge
  in_anchor->UnlinkAll();
  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(dst_new_node->GetOutDataAnchor(0), in_anchor));
  return ge::SUCCESS;
}

Status ConcatGroupPartitioner::ParseConcatNode() {
  const auto &output_attr = concat_node_->outputs[0].attr;
  dtype_size_ = ge::GetSizeByDataType(output_attr.dtype);
  GE_ASSERT_TRUE(dtype_size_ > 0, "unsupported dtype: %d", static_cast<int32_t>(output_attr.dtype));
  int64_t stride = 1;
  for (size_t i = concat_dim_ + 1; i < output_attr.repeats.size(); ++i) {
    const auto &expr = output_attr.repeats[i];
    GE_CHK_BOOL_RET_SPECIAL_STATUS(!expr.IsConstExpr(), ge::SUCCESS, "contains non-static dim after concat dim");
    int64_t value = -1;
    GE_ASSERT_TRUE(expr.GetConstValue(value));
    GE_ASSERT_TRUE(value >= 0);
    stride *= value;
  }
  stride_ = stride;

  int64_t concat_dim_size = -1;
  if (output_attr.repeats[concat_dim_].IsConstExpr()) {
    GE_ASSERT_TRUE(output_attr.repeats[concat_dim_].GetConstValue(concat_dim_size));
    output_cols_ = concat_dim_size * stride_;
  }
  GELOGD("concat_dim_stride = %ld, concat_dim = %ld", stride_, concat_dim_size);

  bool all_known = true;
  for (size_t i = 0UL; i < concat_dim_; ++i) {
    const auto &expr = output_attr.repeats[i];
    if (expr.IsConstExpr()) {
      int64_t value = -1;
      GE_ASSERT_TRUE(expr.GetConstValue(value) && (value >= 0));
      known_rows_ *= value;
    }
    all_known = all_known && expr.IsConstExpr();
  }
  total_rows_ = all_known ? known_rows_ : -1;
  GELOGD("known rows = %ld, total_rows = %ld", known_rows_, total_rows_);

  const auto &all_in_data_anchors = concat_node_->GetAllInDataAnchorsPtr();
  for (size_t i = 0UL; i < all_in_data_anchors.size(); ++i) {
    const auto &repeats = concat_node_->inputs[i].attr.repeats;
    const auto &expr = repeats[concat_dim_];
    if (expr.IsConstExpr()) {
      int64_t size = -1;
      GE_ASSERT_TRUE(expr.GetConstValue(size));
      GE_ASSERT_TRUE(size >= 0);
      size *= stride_;
      concat_dim_sizes_.emplace_back(size);
    } else {
      concat_dim_sizes_.emplace_back(-1);
    }
  }
  return ge::SUCCESS;
}

Status ConcatGroupPartitioner::TryOptimizeGroupSize() {
  GELOGD("all input concat dim is known shape, try optimize group size");
  // prod(dims[concat_dim:])较小, 分组输出会使用跳写导致性能劣化，该场景尽量做不切分
  const auto kMinColsPerGroup = kMinGroupSizeByte / dtype_size_;
  use_default_group_ = true;
  const auto num_inputs = concat_dim_sizes_.size();
  auto num_groups = CeilDiv(concat_dim_sizes_.size(), max_input_num_per_group_);
  GE_ASSERT_TRUE(num_groups > 0);  // impossible, just assert
  int64_t avg_cols_per_group = output_cols_ / num_groups;
  avg_cols_per_group = std::min(avg_cols_per_group, (kMaxBlockSize / dtype_size_));
  GELOGD("num_inputs = %zu, max_input_num_per_group = %u, estimated num_groups = %ld, cols_per_group = %ld", num_inputs,
         max_input_num_per_group_, num_groups, avg_cols_per_group);
  // 防止分group导致size过小，跳写导致性能劣化
  if (avg_cols_per_group <= kMinColsPerGroup) {
    avg_cols_per_group = kMinColsPerGroup;
    max_input_num_per_group_ = std::max(max_input_num_per_group_, MaxInputNumPerGroup());
    GELOGD("group is too small, adjust cols_per_group = %ld, max_input_num_per_group = %u", kMinColsPerGroup,
           max_input_num_per_group_);
  }
  default_cols_per_group_ = avg_cols_per_group;
  return ge::SUCCESS;
}

uint32_t ConcatGroupPartitioner::MaxInputNumPerGroup() const {
  constexpr uint32_t kLargeInputNum = 512;
  constexpr uint32_t kMaxInputNum = 32U;
  const auto min_group_size = concat_by_transpose_ ? 32U : 16U;
  const uint32_t max_input_num = (concat_dim_sizes_.size() >= kLargeInputNum) ? kMaxInputNum : min_group_size;
  return max_input_num;
}

Status ConcatGroupPartitioner::RecomputeDiffAxes() {
  const auto num_inputs = concat_node_->inputs.Size();
  for (uint32_t i = 0U; i < num_inputs; ++i) {
    groups_.emplace_back(ConcatGroup{i, i + 1, kGroupTypeDefault, 0});
  }
  GE_ASSERT_SUCCESS(RecomputeNodesCrossGroups(false, has_recompute_));
  GE_ASSERT_SUCCESS(RecomputeNodesCrossGroups(true, has_recompute_));
  return ge::SUCCESS;
}
}  // namespace optimize
