/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reuse_checker.h"
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/unfold/graph_unfolder.h"
#include "runtime/mem.h"
#include "node_checker_utils.h"

namespace ge {
namespace {
constexpr size_t kPrintErrorReuseMaxNum = 5U;
constexpr size_t kPrintMaxNameLen = 350U;
bool IsDataRefConstOrVariable(const Node *node) {
  if (!OpTypeUtils::IsDataNode(node->GetType())) {
    return false;
  }
  const auto &in_node = NodeUtils::GetParentInput(*node);
  if (in_node == nullptr) {
    return false;
  }
  const auto &owner = in_node->GetOwnerComputeGraph();
  if (owner != nullptr) {
    bool dynamic_shape_partition = false;
    (void)AttrUtils::GetBool(owner, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);
    if ((owner->GetGraphUnknownFlag()) || dynamic_shape_partition) {
      return false;
    }
  }

  std::string const_type;
  if ((!NodeUtils::GetConstOpType(in_node, const_type)) || OpTypeUtils::IsVariableNode(in_node->GetType())) {
    return false;
  }
  return true;
}

bool NoNeedToCheckByAttr(const ge::Node *node) {
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
  const bool buffer_pool = node->GetOpDescBarePtr()->HasAttr(ge::ATTR_NAME_BUFFER_POOL_ID) &&
                           node->GetOpDescBarePtr()->HasAttr(ge::ATTR_NAME_BUFFER_POOL_SIZE);

  return no_task || buffer_pool;
}

bool NoNeedToCheck(const Node *node) {
  const auto &type = node->GetType();
  if (OpTypeUtils::IsVarLikeNode(type)) {
    return true;
  }
  if (NodeUtils::IsConst(*node)) {
    return true;
  }
  if (OpTypeUtils::IsConstPlaceHolderNode(type)) {
    return true;
  }
  if ((type == FILECONSTANT) || (type == MEMSET)) {
    return true;
  }
  if (IsDataRefConstOrVariable(node)) {
    return true;
  }
  if (NoNeedToCheckByAttr(node)) {
    return true;
  }
  // 不分配流，建立的到达关系受限，会产生误判
  if (node->GetOpDescBarePtr()->GetStreamId() == ge::kInvalidStreamId) {
    return true;
  }
  return false;
}

bool IsSkip(const GeTensorDescPtr &output_tensor_desc, const int64_t memory_type, const int64_t mem_size) {
  if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(*output_tensor_desc)) {
    return true;
  }

  if ((memory_type == RT_MEMORY_L1) || (memory_type == kRtMemoryUB)) {
    return true;
  }
  if (mem_size <= 0) {
    return true;
  }
  int32_t tensor_type = 0;
  bool ret = ge::AttrUtils::GetInt(output_tensor_desc, ATTR_NAME_TENSOR_MEMORY_SCOPE, tensor_type);
  if (ret && tensor_type == kOutputMemoryGlobalType) {
    return true;
  }
  return false;
}


bool IsRef(const ReuseNodeMem &lh, const ReuseNodeMem &rh) {
  return lh.symbol_index == rh.symbol_index;
}

std::string NodeOutMemStr(const ReuseNodeMem &n) {
  std::stringstream ss;
  ss << "node: " << n.node->GetName().substr(0, kPrintMaxNameLen) << "(" << n.node->GetType() << ")"
     << ", stream: " << n.node->GetOpDescBarePtr()->GetStreamId() << ", topo_id: " << n.topo_id
     << ", out_index: " << n.out_index << ", offset: " << n.offset << ", size: " << n.mem_size
     << ", symbol_id: " << n.symbol_index;
  return ss.str();
}

int64_t GetMemoryType(const OpDesc *const op_desc, const int32_t out_index,
                      const std::vector<int64_t> &output_memory_types) {
  int64_t memory_type = RT_MEMORY_HBM;
  if (static_cast<size_t>(out_index) < output_memory_types.size()) {
    memory_type = output_memory_types[out_index];
  } else {
    if (ge::AttrUtils::GetInt(op_desc->MutableOutputDesc(out_index), ATTR_NAME_TENSOR_MEM_TYPE, memory_type)) {
      GELOGI("node: %s out_index: %d, memory_type: %" PRId64 "", op_desc->GetNamePtr(), out_index, memory_type);
    }
  }
  return memory_type;
}
}  // namespacekMaxLogLen

ReuseChecker::ReuseChecker(ComputeGraphPtr &graph, AnchorToSymbol anchor_to_symbol, SymbolToAnchors symbol_to_anchors)
    : graph_(graph),
      anchor_to_symbol_(std::move(anchor_to_symbol)),
      symbol_to_anchors_(std::move(symbol_to_anchors)),
      deps_analyzer_(graph_, anchor_to_symbol_, symbol_to_anchors_) {}

Status ReuseChecker::Init() {
  return deps_analyzer_.Init();
}

Status ReuseChecker::Check() {
  GELOGI("memory reuse check start, graph: %s", graph_->GetName().c_str());
  GE_ASSERT_TRUE(deps_analyzer_.IsInit());
  GE_ASSERT_SUCCESS(CollectNodeOffset());
  for (auto &batched_map : reuse_nodes_) {
    for (auto &mem_typed_map : batched_map.second) {
      ConstructReuseNodesMap(mem_typed_map.second);
    }
  }

  for (auto &batched_map : reuse_nodes_) {
    for (auto &mem_typed_map : batched_map.second) {
      GE_ASSERT_SUCCESS(CheckReuseNodes(mem_typed_map.second), "Check reuse failed. batch_lable: %s, memory_type: "
                        "%" PRId64 "", batched_map.first.c_str(), mem_typed_map.first);
    }
  }
  GELOGI("memory reuse check success, graph: %s", graph_->GetName().c_str());
  return SUCCESS;
}

void ReuseChecker::SetMaxOffset(const size_t offset) {
  max_offset_ = offset;
}

Status ReuseChecker::CheckReuseNodes(OffsetReuseNodes &offset_reuse_nodes_map) {
  bool check_success = true;
  size_t print_count = 0U;
  for (const auto &offset_reuse_nodes : offset_reuse_nodes_map) {
    auto iter = offset_reuse_nodes.second.begin();
    auto iter_next = offset_reuse_nodes.second.begin();
    ++iter_next;
    while (iter_next != offset_reuse_nodes.second.end()) {
      if (IsRef(*iter, *iter_next)) {
        ++iter_next;
        ++iter;
        continue;
      }
      if (!deps_analyzer_.CanAReuseB(iter_next->node, iter_next->out_index, iter->node, iter->out_index) &&
          !deps_analyzer_.CanAReuseB(iter->node, iter->out_index, iter_next->node, iter_next->out_index)) {
        check_success = false;
        ++print_count;
        auto reason =
            deps_analyzer_.WhyACannotReuseB(iter_next->node, iter_next->out_index, iter->node, iter->out_index);
        REPORT_INNER_ERR_MSG("E19999", "can not reuse memory, %s and %s", NodeOutMemStr(*iter).c_str(),
                           NodeOutMemStr(*iter_next).c_str());
        GELOGE(FAILED, "can not reuse memory, %s and %s", NodeOutMemStr(*iter).c_str(),
               NodeOutMemStr(*iter_next).c_str());
        REPORT_INNER_ERR_MSG("E19999", "reason: %s", reason.c_str());
        GELOGE(FAILED, "reason: %s", reason.c_str());

        reason = deps_analyzer_.WhyACannotReuseB(iter->node, iter->out_index, iter_next->node, iter_next->out_index);
        REPORT_INNER_ERR_MSG("E19999", "reason: %s", reason.c_str());
        GELOGE(FAILED, "reason: %s", reason.c_str());
        if (print_count > kPrintErrorReuseMaxNum) {
          break;
        }
      }
      ++iter_next;
      ++iter;
    }
    if (print_count > kPrintErrorReuseMaxNum) {
      break;
    }
  }
  if (!check_success) {
    deps_analyzer_.Debug();
    REPORT_INNER_ERR_MSG("E19999", "Memory reuse does not satisfy dependency relationships");
    GELOGE(FAILED, "Memory reuse does not satisfy dependency relationships");
    return FAILED;
  }
  return SUCCESS;
}

Status ReuseChecker::CollectNodeOffset() {
  for (const auto node_ptr : graph_->GetAllNodesPtr()) {
    if (NoNeedToCheck(node_ptr)) {
      continue;
    }
    auto op_desc = node_ptr->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    std::string batch_label;
    (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label);

    std::vector<int64_t> output_memory_types;
    if (ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types) &&
        (!output_memory_types.empty())) {
      GELOGI("node: %s get memory type: %" PRId64 "", op_desc->GetNamePtr(), output_memory_types[0]);
    }

    const auto outputs_offset = op_desc->GetOutputOffset();
    for (const auto out_anchor_ptr : node_ptr->GetAllOutDataAnchorsPtr()) {
      std::string ref_var_name;
      if (MemLayoutConflictUtil::HasRefVarName(out_anchor_ptr, ref_var_name)) {
        continue;
      }
      GE_ASSERT_TRUE(static_cast<size_t>(out_anchor_ptr->GetIdx()) < outputs_offset.size(),
                     "node: %s, index: %d, outputs_offset size: %zu", op_desc->GetNamePtr(), out_anchor_ptr->GetIdx(),
                     outputs_offset.size());
      const auto offset = outputs_offset[out_anchor_ptr->GetIdx()];
      if (static_cast<size_t>(offset) > max_offset_) {
        continue;
      }
      const auto memory_type = GetMemoryType(op_desc, out_anchor_ptr->GetIdx(), output_memory_types);
      const auto symbol_index = GetSymbolIndex(node_ptr, static_cast<uint32_t>(out_anchor_ptr->GetIdx()));
      const auto output_tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(out_anchor_ptr->GetIdx()));
      GE_ASSERT_NOTNULL(output_tensor_desc);
      int64_t mem_size = 0;
      GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputSize(node_ptr, out_anchor_ptr->GetIdx(), mem_size),
                        "get node: %s output[%d] size failed.", node_ptr->GetNamePtr(), out_anchor_ptr->GetIdx());
      if (IsSkip(output_tensor_desc, memory_type, mem_size)) {
        continue;
      }
      reuse_nodes_[batch_label][memory_type][offset].insert({node_ptr, IOType::kOut,
                                                             static_cast<size_t>(out_anchor_ptr->GetIdx()),
                                                             symbol_index, mem_size, op_desc->GetId(), offset});
    }
  }
  return SUCCESS;
}

void ReuseChecker::ConstructReuseNodesMap(OffsetReuseNodes &offset_reuse_nodes_map) const {
  auto iter = offset_reuse_nodes_map.begin();
  while (iter != offset_reuse_nodes_map.end()) {
    auto &reuse_nodes = iter->second;
    if ((++iter) == offset_reuse_nodes_map.end()) {
      return;
    }
    auto iter_next = iter;
    for (auto &node_out_mem : reuse_nodes) {
      if (node_out_mem.mem_size > iter_next->first - node_out_mem.offset) {
        iter_next->second.insert(node_out_mem);
      }
    }
  }
}

size_t ReuseChecker::GetSymbolIndex(const Node *node_ptr, const uint32_t out_index) {
  NodeIndexIO node_index_io(node_ptr, out_index, kOut);
  const auto &symbol_str = anchor_to_symbol_[node_index_io.ToString()];
  auto iter = symbol_str_to_index_.find(symbol_str);
  if (iter == symbol_str_to_index_.end()) {
    symbol_str_to_index_[symbol_str] = unique_index_;
    ++unique_index_;
    return unique_index_ - 1U;
  } else {
    return iter->second;
  }
}
}  // namespace ge