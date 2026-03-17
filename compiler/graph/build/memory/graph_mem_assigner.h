/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "runtime/mem.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/graph_mem_splitter.h"

namespace ge {
struct MemoryOffset {
  MemoryOffset(rtMemType_t mem_type, size_t mem_offset, size_t theory_min = 0UL, size_t zero_copy_size = 0UL)
      : mem_type_(mem_type), mem_offset_(mem_offset), theory_min_(theory_min), zero_copy_size_(zero_copy_size) {}

 public:
  rtMemType_t mem_type_;
  size_t mem_offset_;
  size_t theory_min_;
  size_t zero_copy_size_;
};

struct CleanDataTypeValue {
  bool operator==(const CleanDataTypeValue &other) const {
    return (data_type == other.data_type) &&
           (int_val == other.int_val) &&
           std::fabs(float_val - other.float_val) <= std::numeric_limits<float>::epsilon();
  }
  int32_t data_type; // atomic_node通过TBE_OP_ATOMIC_DTYPES属性指定的数据类型，默认float
  int64_t int_val; // atomic_node通过TBE_OP_ATOMIC_INT64_VALUES属性指定的初始值
  float32_t float_val; // atomic_node通过TBE_OP_ATOMIC_FLOAT_VALUES属性指定的初始值，默认0.0
};

struct CleanMemInfo {
  std::string ToStr() const{
    std::stringstream ss;
    ss << "offset: " << offset << ", size: " << size << ", mem_type: " << memory_type
       << ", data_type: " << type_val.data_type << ", int_value: "
       << type_val.int_val << ", float_value: " << type_val.float_val;
    return ss.str();
  }
  // 为了保存在std::set中，按照offset从小到大排序
  bool operator<(const CleanMemInfo &other) const {
    // 首先按memory_type排序
    if (memory_type != other.memory_type) {
      return memory_type < other.memory_type;
    }
    if (offset != other.offset) {
      return offset < other.offset;
    }
    return size < other.size;
  }

  bool CanMerge(const CleanMemInfo &other) const {
    if ((memory_type == other.memory_type) && (!is_zero_copy) && (type_val == other.type_val)){
      if ((offset <= other.offset) && ((offset + size) >= other.offset)) {
        return true;
      }
    }
    return false;
  }
  // this和other合并，保留this
  void Merge(const CleanMemInfo &other) {
    size = std::max(size, other.size + other.offset - offset);
  }
  bool Contain(const CleanMemInfo &other) const {
    if ((memory_type == other.memory_type) && (type_val == other.type_val)) {
      if ((offset <= other.offset) && ((offset + size) >= (other.offset + other.size))) {
        return true;
      }
    }
    return false;
  }
  int64_t offset = -1; // 要清理的逻辑地址
  int64_t size = 0; // 要清理的内存大小， 如果是零拷贝则32字节对齐，其他512字节对齐
  uint32_t memory_type = RT_MEMORY_HBM; // 内存类型，默认hbm，也有p2p
  CleanDataTypeValue type_val{static_cast<int32_t>(ge::DT_FLOAT), 0, 0.0};
  bool is_zero_copy = false; // 是否是零拷贝，零拷贝的不能合并，因为多个用户输入地址可能不连续
};


// 需要atomic清零的算子，可能会通过属性指定初始值和数据类型，目前仅对输出和workspace设置属性，不包含输入的
class AtomicNodeCleanTypeVals {
 public:
  ge::Status Init(const ge::Node *node);
  // 获取下一个属性，注意调用顺序，先输出，后workspace，输入不能调用
  ge::Status GetNextAttr(CleanDataTypeValue &type_value);
  std::string ToStr() const;
 private:
  std::vector<int32_t> data_types_;
  std::vector<int64_t> int_vals_;
  std::vector<float32_t> float_vals_;
  size_t data_type_index_ = 0U;
  size_t int_val_index_ = 0U;
  size_t float_val_index_ = 0U;
  const ge::Node *node_;
};

struct MemsetNodeAddrAndAttr {
  MemsetNodeAddrAndAttr(const size_t reserve_size) {
    offsets.reserve(reserve_size);
    sizes.reserve(reserve_size);
    memory_types.reserve(reserve_size);
    data_type_list.reserve(reserve_size);
    int_list.reserve(reserve_size);
    float_list.reserve(reserve_size);
  }
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> memory_types;
  std::vector<int32_t> data_type_list;
  std::vector<int64_t> int_list;
  std::vector<float32_t> float_list;
};

inline bool IsFloatType(const ge::DataType dt) {
  return (dt == ge::DT_FLOAT) || (dt == ge::DT_FLOAT16) || (dt == ge::DT_DOUBLE) || (dt == ge::DT_BF16) ||
         (dt == ge::DT_HIFLOAT8) || (dt == ge::DT_FLOAT8_E5M2) || (dt == ge::DT_FLOAT8_E4M3FN) ||
         (dt == ge::DT_FLOAT8_E8M0) || (dt == ge::DT_FLOAT6_E3M2) || (dt == ge::DT_FLOAT6_E2M3) ||
         (dt == ge::DT_FLOAT4_E2M1) || (dt == ge::DT_FLOAT4_E1M2);
}

using MemoryOffsetMap = std::map<int64_t, MemoryOffset>;  // key: MemoryOffset::mem_type_

class VariableMemoryAssigner {
 public:
  explicit VariableMemoryAssigner(ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}

  VariableMemoryAssigner(const VariableMemoryAssigner &) = delete;

  VariableMemoryAssigner &operator=(const VariableMemoryAssigner &) = delete;

  virtual ~VariableMemoryAssigner() = default;

  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  Status Assign();

  /// @ingroup ge_graph
  /// @brief assign variable attr to nodes
  /// @return Status result of function
  Status AssignVarAttr2Nodes();

  Status AssignMemory2HasRefAttrNode();

 private:
  ComputeGraphPtr compute_graph_;
};

using VariableMemoryAssignerPtr = std::shared_ptr<VariableMemoryAssigner>;
using BlockMemAssignerPtr = std::shared_ptr<BlockMemAssigner>;
using HybridMemAssignerPtr = std::shared_ptr<HybridMemAssigner>;
using GraphMemSplitterPtr = std::shared_ptr<GraphMemSplitter>;


class GraphMemoryAssigner {
 public:
  explicit GraphMemoryAssigner(ComputeGraphPtr compute_graph)
      : compute_graph_(std::move(compute_graph)),
        mem_assigner_(nullptr), graph_mem_splitter_(nullptr) {}

  GraphMemoryAssigner(const GraphMemoryAssigner &) = delete;

  GraphMemoryAssigner &operator=(const GraphMemoryAssigner &) = delete;

  virtual ~GraphMemoryAssigner() = default;

  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  Status AssignMemory(const bool has_assigned_var_mem = false);

  /// @ingroup ge_graph
  /// @brief assign variable memory offset
  /// @return Status result of function
  static Status AssignVarMemory(const ComputeGraphPtr &compute_graph);

  /// @ingroup ge_graph
  /// @brief assign variable attr to nodes,
  /// must be called after all memory assigned.
  /// @return Status result of function
  Status AssignVarAttr2Nodes();

  ge::Status AssignMemory2HasRefAttrNode() const;

  ge::Status ReAssignMemory(std::map<uint64_t, size_t> &mem_type_to_offset);

  Status AssignZeroCopyMemory(std::map<uint64_t, size_t> &mem_offset, size_t &zero_mem_copy_size);

  Status ReAssignContinuousMemory();

  Status SetMemReuseInfo() const;

  void RecordSubsequentReuseNodeInfo(const MemoryBlock *const memory_block,
                                     const std::vector<MemReuseInfo> &parent_mem_resue_info,
                                     std::vector<MemReuseInfo> &total_child_mem_resue_info,
                                     uint32_t depth = 0U) const;

  Status SetInputOffset() const;

  Status UpdateOpInputOffset(const NodePtr &node) const;
  Status UpdateRefOpOutputOffset(const NodePtr &node, const std::map<int32_t, int32_t> &out2ins, const int32_t ref_in,
                                 const int64_t input_offset) const;

  Status AtomicCleanCheck() const;
  Status ReuseCheck() const;
  Status CheckOffset() const;
  Status CheckRefNodeOffset(const NodePtr &node) const;

  Status AssignReferenceMemory() const;

  void MarkDistanceAttr();

  const GraphMemSplitterPtr GetGraphMemSplitter() const {
    return graph_mem_splitter_;
  }

  const BlockMemAssignerPtr GetMemAssignerPtr() const {
    if (mem_assigner_ != nullptr) {
      return mem_assigner_->GetPriorityAssinger();
    }
    return nullptr;
  }
  /// @brief check the input of node whether support atomic attr
  /// @param node
  /// @return true:supported; false:not supported
  static bool CheckInputIsSupportAtomic(const Node *node);
  Status CollectAtomicNodeCleanMemInfos(const NodePtr &memset_node,
                                        std::set<CleanMemInfo> &clean_mem_infos) const;
  Status GetMemType(const Node *const node, const IOType &io_type, const uint32_t index, uint32_t &mem_type) const;
  Status GetInputCleanMemInfos(const NodePtr &node, std::set<CleanMemInfo> &clean_mem_infos) const;
  Status GetOutputCleanMemInfos(const NodePtr &node, AtomicNodeCleanTypeVals &type_vals,
                                std::set<CleanMemInfo> &clean_mem_infos) const;
  Status GetWorkspaceCleanMemInfos(const NodePtr &node, AtomicNodeCleanTypeVals &type_vals,
                                   std::set<CleanMemInfo> &clean_mem_infos) const;
  Status GetFusionWorkspaceCleanMemInfos(const NodePtr &node, std::set<CleanMemInfo> &clean_mem_infos) const;
  MemsetNodeAddrAndAttr ConstructMemsetAddrAndAttr(const std::vector<CleanMemInfo> &clean_mem_infos) const;
  Status SetAtomicCleanOffset() const;
  std::map<int64_t, int64_t> GetSplitOffsetSize() const;
  std::vector<CleanMemInfo> MergeCleanMemInfos(const std::set<CleanMemInfo> &clean_mem_infos,
                                               const std::map<int64_t, int64_t> &split_offset_to_size) const;
 private:
  Status AssignReferenceMemory(const NodePtr &node) const;

  Status OffsetValidCheck() const;

  Status ReAssignAtomicMemory();

  Status TryGetNodeRefIndexes(const NodePtr &node, std::map<int32_t, int32_t> &out2ins) const;

  bool IsAssignContinuousInputMemoryDirectly(const NodePtr &input_continuous_node,
                                                            std::map<NodePtr, uint32_t> &node_2_continuous_type) const;

  Status FilterAtomicNodes(std::map<std::string, std::map<NodePtr, std::vector<NodePtr>>> &atomic_nodes);

  Status SetMemOffset(const NodePtr &node, const InDataAnchorPtr &in_data_anchor, bool reverse_refresh,
                      int64_t &mem_offset) const;

  Status AssignContinuousInputMemory(const NodePtr &node,
                                     uint32_t continuous_type,
                                     bool reverse_refresh = false);

  Status AssignContinuousOutputMemory(const NodePtr &node, int64_t memory_type, uint32_t continuous_type) const;

  bool CheckAtomicNodeIsSupportRef(const NodePtr &node) const;

  Status GetMemoryAssignmentStatus(const NodePtr &node, int64_t output_index, bool &is_mem_assigned) const;

  Status AssignAtomicOutputMemory(const NodePtr &node, std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
                                  std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes);
  Status UpdateParentNodeOutputOffset(const ge::NodePtr &node, int64_t output_index, int64_t offset) const;
  Status AssignOrdinaryAtomicWorkspaceMemory(const OpDescPtr &op_desc,
                                             std::map<std::string, std::map<int64_t, int64_t>> &workspace_info,
                                             std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
                                             std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes);

  Status AssignFusionAtomicWorkspaceMemory(const OpDescPtr &op_desc,
                                           std::map<std::string, std::map<int64_t, int64_t>> &workspace_info,
                                           std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
                                           std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes);

  Status AssignAtomicOutputAndWorkspaceMemory(const NodePtr &node,
                                              std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
                                              std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes);

  Status AppendAddrSizeToMemSetOp(const NodePtr &memset_node, const MemsetNodeAddrAndAttr &addr_type) const;
  Status AppendAttrsToMemSetOp(const NodePtr &memset_node, const MemsetNodeAddrAndAttr &addr_type) const;

  void AlignMemOffset(const int64_t &mem_align_size, int64_t memory_type);

  Status UpdateOpInputOffset(const NodePtr &node, std::vector<int64_t> &input_list) const;

  Status UpdateConstArgsOffset(const NodePtr &node, std::vector<int64_t> &input_list) const;

  Status UpdateOpInputDescOffset(const NodePtr &node) const;

  NodePtr GetKnownInputNode(const NodePtr &node) const;

  Status GetNodeMemoryType(const NodePtr &node, int64_t &memory_type, std::string input_or_output) const;

  bool CheckContinuousMemType(std::vector<int64_t> mem_type_list) const;

  Status AssignBufferPoolMemory();

  bool IsRefFromInputOpCascade(const NodePtr &node) const;

  Status UpdateRefOpOffsetReverse(const NodePtr &node) const;

  bool IsOutputVisitedByMultiStream(const NodePtr &peer_out_node, int64_t out_anchor_index) const;

  void UpdatePrevNodeInputDesc(const NodePtr &prev_node,
                               const std::vector<int64_t> &prev_node_input_index_vec,
                               int64_t distance) const;

  void UpdateCurNodeInputDesc(const NodePtr &cur_node, int64_t cur_node_input_index, int64_t distance) const;

  void CheckNeedCalcDistAndUpdateVisitInfo(const NodePtr &peer_out_node,
                                           const OutDataAnchorPtr &peer_out_anchor,
                                           size_t matched_mem_offset,
                                           std::map<size_t, std::pair<NodePtr,
                                                                      std::vector<int64_t>>> &mem_block_visit_info,
                                           bool &is_need_calc_distance) const;

  void CalcDistanceAndUpdateDesc(const std::map<std::string, int64_t> &node_index_in_stream,
                                 const InDataAnchorPtr &in_data_anchor,
                                 size_t matched_mem_offset,
                                 const NodePtr &node,
                                 std::map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
                                 bool &is_need_skip) const;

  void DeleteVisitInfoWhenLifecycleEnded(const NodePtr &node,
                                         const InDataAnchorPtr &in_data_anchor,
                                         size_t matched_mem_offset,
                                         std::map<size_t, std::pair<NodePtr,
                                         std::vector<int64_t>>> &mem_block_visit_info) const;

  void MarkNodeDistanceAttr(const NodePtr &node,
                            std::map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
                            const std::map<std::string, int64_t> &node_index_in_stream);

  MemoryOffsetMap memory_offset_;
  ComputeGraphPtr compute_graph_;
  HybridMemAssignerPtr mem_assigner_;
  GraphMemSplitterPtr graph_mem_splitter_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
