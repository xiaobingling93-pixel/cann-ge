/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_GROUP_PARTITIONER_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_GROUP_PARTITIONER_H_

#include "ascgen_log.h"
#include "ascir.h"
#include "ascir_ops.h"

namespace optimize {
class ConcatGroupPartitioner {
 public:
  struct ConcatGroup {
    size_t start;
    size_t end;
    uint32_t group_type;
    int64_t size;  // cols
  };

  ConcatGroupPartitioner(ge::AscNodePtr concat_node, size_t concat_dim);

  Status PartitionGroups(std::vector<ConcatGroup> &groups);

  bool HasRecompute() const;

  Status RecomputeDiffAxes();

 private:
  Status Initialize();
  void GroupEnd(size_t index_end);
  void GroupStart(int64_t index_start, uint32_t group_type, int64_t size);
  [[nodiscard]] int64_t GetSizeLimitByGroupType(uint32_t group_type) const;
  [[nodiscard]] uint32_t GetGroupType(int64_t size) const;
  void MergeSmallGroups();
  [[nodiscard]] bool CanMerge(const ConcatGroup &lhs, const ConcatGroup &rhs) const;
  void ConvertToDefaultIfTooSmall();
  void UpdateStatus(int64_t size);
  [[nodiscard]] bool NeedSubmit(size_t i, int64_t size, uint32_t new_group_type);
  static std::string GroupTypeToString(uint32_t group_type);
  static bool IsAligned(uint32_t group_type);
  static bool IsSmallTail(uint32_t group_type);
  static void MergeGroups(std::vector<ConcatGroup>::value_type &lhs_group,
                          std::vector<ConcatGroup>::value_type &rhs_group);
  ge::Status RecomputeNodesCrossGroups(bool search_backward,
                                       bool &has_recompute) const;
  ge::Status FindFirstMultiOutputAnchors(const ge::InDataAnchorPtr &in_anchor, int32_t end_index, bool search_backward,
                                         std::set<ge::InDataAnchorPtr> &visited_in_anchors,
                                         ge::InDataAnchorPtr &to_split) const;
  ge::Status CheckIsAncestorOfConcat(const ge::OutDataAnchorPtr &out_anchor, int32_t start_index,
                                     const ge::Expression &concat_dim_size, bool search_backward,
                                     bool &need_split) const;
  ge::Status RecomputeInNodes(const ge::InDataAnchorPtr &in_anchor, size_t index,
                              std::map<std::string, ge::AscNodePtr> &name_to_new_nodes) const;
  Status ParseConcatNode();
  Status TryOptimizeGroupSize();
  uint32_t MaxInputNumPerGroup() const;
  bool NeedSplit(const ge::InDataAnchorPtr &in_anchor, int32_t start_index, const ge::Expression &cur_dim_size) const;
  size_t GetGroupSize(size_t index) const;

  static constexpr uint32_t kGroupTypeDefault = 0x1;
  static constexpr uint32_t kGroupTypeAligned = 0x10;
  static constexpr uint32_t kGroupTypeSmallTail = 0x100;
  static constexpr uint32_t kGroupTypeSmallTailAndAligned = kGroupTypeAligned | kGroupTypeSmallTail;
  static constexpr uint32_t kGroupTypeScalar = 0x1000;
  static constexpr uint32_t kGroupTypeNone = 0x10000;
  static constexpr int64_t kGroupEltSizeThreshold = 1024 * 4;  // 单个足够大, 不要重复搬运
  static constexpr int64_t kMaxBlockSize = 8192;
  static constexpr int64_t kMaxBlockSizeForSmallTail = 96;
  static constexpr int32_t kSmallTailLimit = 35;
  static constexpr int64_t kGroupParallelRowThreshold = 64 * 64;  // 行数超过一定值, att会用满核，没有group parallel的空间

  ge::AscNodePtr concat_node_;
  size_t concat_dim_;
  uint32_t max_input_num_per_group_ = std::numeric_limits<uint32_t>::max();
  std::vector<ConcatGroup> groups_;
  uint32_t group_type_ = 0U;
  int64_t index_start_ = -1;
  int64_t cur_size_ = 0;
  int64_t dtype_size_ = 0;
  int64_t stride_ = -1;
  bool can_use_small_tail_ = false;
  bool concat_by_transpose_ = false;
  bool is_concat_scalar_ = false;
  bool use_default_group_ = false;
  std::set<std::string> consumed_;
  std::map<uint32_t, int64_t> group_type_to_limit_;
  std::map<std::string, std::string> name_to_origin_name_;
  std::vector<int64_t> concat_dim_sizes_;
  int64_t output_cols_ = -1;
  int64_t known_rows_ = 1L;
  int64_t total_rows_ = 0L;
  int64_t default_cols_per_group_ = 0L;
  bool has_recompute_ = false;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CONCAT_GROUP_PARTITIONER_H_
