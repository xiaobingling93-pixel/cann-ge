/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/block_mem_assigner.h"
#include <algorithm>
#include <sstream>
#include <stack>

#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "common/context/local_context.h"
#include "common/math/math_util.h"
#include "common/checker.h"
#include "common/memory/mem_type_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/optimize/params.h"
#include "framework/common/runtime_tensor_desc.h"
#include "graph/build/memory/dynamic_batch_mem_assigner.h"
#include "runtime/subscriber/global_profiler.h"

using std::unordered_map;
using std::unordered_set;

namespace {
const char *const kAttrNameWorkspaceReuseFlag = "workspace_reuse_flag";
const char *const kL2FusionDynamicConvergeOp = "l2fusion_dynamic_converge_op";
const char *const kOpNoReuseMem = "no_reuse_mem_flag";
const int32_t kReuseMaxOpNum = 10;
const int32_t kReuseMaxCharNum = 2000;
const uint32_t kAutoMode = 1U;
const std::set<ge::DataType> kNotPostReuseDataType = {ge::DT_RESOURCE, ge::DT_VARIANT};
constexpr int64_t kParentNodeDefaultStreamId = -2;

bool IsNodeSupportZeroCopy(const ge::Node *const node) {
  const auto op_engine = node->GetOpDesc()->GetOpKernelLibName();
  if (op_engine == ge::kEngineNameDsa) {
    GELOGD("dsa engine op[%s] not support zero copy", node->GetName().c_str());
    return false;
  }
  if (op_engine == ge::kCustomOpKernelLibName) {
    GELOGD("custom engine op[%s] not support zero copy", node->GetName().c_str());
    return false;
  }
  if (ge::OpUtils::IsHcomNodeNotSupportAddrRefresh(node->GetOpDesc())) {
    GELOGD("hccl engine op[%s] not support zero copy", node->GetName().c_str());
    return false;
  }
  return true;
}

// condition op, dsa op and same hccl op can not reuse model io block which can zero copy
bool CanReuseZeroCopyBlock(const ge::Node *const node) {
  const static std::set<std::string> kConditionOps = {ge::STREAMSWITCH, ge::LABELSWITCHBYINDEX};
  if (kConditionOps.count(node->GetTypePtr()) > 0U) {
    GELOGD("Condition op[%s] can not reuse zero copy block", node->GetName().c_str());
    return false;
  }
  const auto op_engine = node->GetOpDesc()->GetOpKernelLibName();
  if (op_engine == ge::kEngineNameDsa) {
    GELOGD("dsa engine op[%s] can not reuse zero copy block", node->GetName().c_str());
    return false;
  }
  if (ge::OpUtils::IsHcomNodeNotSupportAddrRefresh(node->GetOpDesc())) {
    GELOGD("hccl engine op[%s] can not reuse zero copy block", node->GetName().c_str());
    return false;
  }
  return true;
}

bool IsContinuousOutput(const ge::NodePtr &n) {
  if (ge::MemLayoutConflictUtil::IsContinuousOutput(n)) {
    if (n->GetOwnerComputeGraphBarePtr() != nullptr) {
      GELOGI("%s name[%s] set continuous, output size[%u].", n->GetOwnerComputeGraphBarePtr()->GetName().c_str(),
          n->GetNamePtr(), n->GetAllOutDataAnchorsSize());
      return true;
    }
  }

  return false;
}

// 编译图的子图中连接netoutput的节点不进行零拷贝
bool IsOutNodeInCurComputeGraph(const ge::Node *const node, const ge::ComputeGraphPtr &graph) {
  return (node->GetType() == ge::NETOUTPUT) && (node->GetOwnerComputeGraphBarePtr() == graph.get());
}

const std::list<ge::NodeIndexIO> &FindNodeOutputSameAnchors(const ge::NodeIndexIO &node_index_io,
                                                            const ge::AnchorToSymbol &anchor_to_symbol,
                                                            const ge::SymbolToAnchors &symbol_to_anchors) {
  static const std::list<ge::NodeIndexIO> res = {};
  const auto &symbol_iter = anchor_to_symbol.find(node_index_io.ToString());
  if (symbol_iter != anchor_to_symbol.cend()) {
    const auto &anchors_iter = symbol_to_anchors.find(symbol_iter->second);
    if (anchors_iter != symbol_to_anchors.cend()) {
      return anchors_iter->second;
    }
  }
  return res;
}

size_t GetOutputFlowToNetoutputNum(const ge::NodePtr &node, uint32_t output_index, const ge::ComputeGraphPtr &graph,
                                   const ge::SymbolToAnchors &symbol_to_anchors,
                                   const ge::AnchorToSymbol &anchor_to_symbol) {
  auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(output_index));
  if (out_anchor == nullptr) {
    return 0U;
  }
  size_t num_anchors_to_netoutput = 0U;
  ge::NodeIndexIO out_node_index_io(node, output_index, ge::kOut);
  const auto &same_anchors = FindNodeOutputSameAnchors(out_node_index_io, anchor_to_symbol, symbol_to_anchors);
  bool include_not_support_zero_copy_node = false;
  if (!same_anchors.empty()) {
    for (const auto &node_index_io : same_anchors) {
      if ((node_index_io.io_type_ != ge::kIn) && (node_index_io.node_ptr_ != nullptr) &&
          (node_index_io.node_ptr_->GetOpDescBarePtr() != nullptr)) {
        if (!IsNodeSupportZeroCopy(node_index_io.node_ptr_)) {
          GELOGI("op[%s] not support zero copy", node_index_io.node_ptr_->GetNamePtr());
          include_not_support_zero_copy_node = true;
        }
        continue;
      }

      if (IsOutNodeInCurComputeGraph(node_index_io.node_.get(), graph)) {  // Combined with NET-OUTPUT of root graph
        num_anchors_to_netoutput++;
      }
    }
  }

  if (num_anchors_to_netoutput > 0U) {
    GELOGI("Node %s output %u flow to %zu root graph output", node->GetNamePtr(), output_index,
           num_anchors_to_netoutput);
    if (include_not_support_zero_copy_node) {
      GELOGI("Node %s output %u symbol is same as not support zero copy node", node->GetNamePtr(), output_index);
      num_anchors_to_netoutput = 0U;
    }
  }
  return num_anchors_to_netoutput;
}

bool IsStrictReuseZeroMemoryMode() {
  const static std::string kEnabled = "1";
  std::string reuse_zero_copy_memory;
  (void)ge::GetContext().GetOption(ge::OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, reuse_zero_copy_memory);
  return (reuse_zero_copy_memory == kEnabled);
}

std::string ToString(const ge::NodeTypeIndex &x) {
  std::stringstream ss;
  if (x.node_ != nullptr) {
    ss << "[" << x.node_->GetNamePtr() << "(" << x.node_->GetTypePtr() << "), ";
  } else {
    ss << "[ (Subgraph)";
  }
  switch (x.mem_type_) {
    case ge::OpMemoryType::kOutput:ss << "Output, ";
      break;
    case ge::OpMemoryType::kWorkspace:ss << "Workspace, ";
      break;
    case ge::OpMemoryType::kOutputDesc:ss << "OutputDesc, ";
      break;
    default:break;
  }
  ss << x.index_ << ", ref_input:" << x.ref_input_ << ", begin:" << x.life_time_begin_ << ", end:" << x.life_time_end_
     << ", symbol end:" << x.symbol_max_life_time_end_;
  return ss.str();
}

bool IsNoReleaseNodeOutBlock(const ge::Node *const node) {
  for (const auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
    if ((input_desc != nullptr)
        && (kNotPostReuseDataType.find(input_desc->GetDataType()) != kNotPostReuseDataType.cend())) {
      return true;
    }
  }
  for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    if ((output_desc != nullptr)
        && (kNotPostReuseDataType.find(output_desc->GetDataType()) != kNotPostReuseDataType.cend())) {
      return true;
    }
  }
  return false;
}

int64_t GetStreamId(const ge::OpDesc *const desc) {
  return ge::MemReuseUtils::GetStreamId(desc);
}

std::string GetStreamIdDesc(const ge::OpDesc *const desc) {
  std::string stream_id_str;
  if (desc != nullptr) {
    const int64_t stream_id = GetStreamId(desc);
    stream_id_str = std::to_string(stream_id);
    if (stream_id != desc->GetStreamId()) {
      stream_id_str.append("--");
      stream_id_str.append(std::to_string(desc->GetStreamId()));
    }
  }
  return stream_id_str;
}

std::string GetName(const ge::MemoryBlock &block, bool last_node = false) {
  if (!block.NodeTypeIndexList().empty()) {
    if (last_node) {
      return ToString(block.NodeTypeIndexList().back());
    } else {
      return ToString(block.NodeTypeIndexList().front());
    }
  }
  return "";
}

const ge::Node *GeParentNode(const ge::Node *const node, const ge::ComputeGraphPtr &compute_graph, uint32_t depth) {
  depth++;
  const ge::Node *parent_node = nullptr;
  if ((node != nullptr) && (depth < ge::kMaxDepthNum)) {
    auto owner_graph = node->GetOwnerComputeGraphBarePtr();
    if (owner_graph != nullptr) {
      parent_node = owner_graph->GetParentNodeBarePtr();
      const bool is_root_graph =
          (parent_node != nullptr) && (parent_node->GetOwnerComputeGraphBarePtr() == compute_graph.get());
      if (!is_root_graph) {
        parent_node = GeParentNode(parent_node, compute_graph, depth);
      }
    }
  }
  if (parent_node == nullptr) {
    return node;
  }
  return parent_node;
}

///              -sub graph1
///             |
/// root graph -|-sub graph2 -|- sub graph4
///             |
///              -sub graph3
/// when compile sub graph2 it's NETOUTPUT is not SubGraphNetOutNode
/// sub graph4's NETOUTPUT is SubGraphNetOutNode
bool IsSubGraphNetOutNode(const ge::Node *const node, const ge::ComputeGraphPtr &compute_graph) {
  if ((node != nullptr) && (node->GetType() == ge::NETOUTPUT) && (node->GetOpDescBarePtr() != nullptr)) {
    auto owner_graph = node->GetOwnerComputeGraphBarePtr();
    if (owner_graph == compute_graph.get()) {
      return false;
    }
    for (uint32_t index = 0U; index < node->GetOpDescBarePtr()->GetInputsSize(); ++index) {
      const auto input_desc = node->GetOpDescBarePtr()->GetInputDescPtr(index);
      if ((input_desc != nullptr) && input_desc->HasAttr(ge::ATTR_NAME_PARENT_NODE_INDEX)) {
        GELOGD("Node: %s is subgraph net out.", node->GetNamePtr());
        return true;
      }
    }
  }
  return false;
}

bool PeerIsSubGraphNetOutNode(const ge::NodePtr &node, const ge::OutDataAnchorPtr &out_data_anchor,
                              const ge::ComputeGraphPtr &compute_graph) {
  if (node != nullptr) {
    for (const auto in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
      const auto peer_node = in_anchor->GetOwnerNodeBarePtr();
      if (IsSubGraphNetOutNode(peer_node, compute_graph)) {
        GELOGD("Node: %s peer node:%s is subgraph net out.", node->GetNamePtr(), peer_node->GetNamePtr());
        return true;
      }
    }
  }
  return false;
}

bool CanNotLifeReuse(const ge::MemoryBlock &block, bool child_reuse = false) {
  if ((!block.reuse_mem_) || ((!child_reuse) && block.child_block_)) {
    return true;
  }
  return false;
}

bool CanReuseBlock(size_t life_begin, const ge::MemoryBlock &reusable_block, size_t block_size) {
  bool can_reuse = false;
  if (reusable_block.Size() == block_size) {
    // in some continuous input case, continuous first input node's is not same as topo first node.
    if (life_begin > 0) {
      if (life_begin > reusable_block.GetLifeEnd(reusable_block.stream_id_)) {
        can_reuse = true;
      }
    } else {
      can_reuse = true;
    }
  }
  return (can_reuse && (!CanNotLifeReuse(reusable_block)));
}

bool ReuseBlock(ge::MemoryBlock &block, const size_t block_size, const size_t life_begin,
                const std::string &batch_label, const ge::NodeTypeIndex &node_type_index) {
  if (block.IsNoAlignSizeReuseBlock() || block.IsRealSizeReuseBlock() || (block.batch_label_ != batch_label)) {
    return false;
  }

  if (block.IsBlockTypeConflictWithNode(node_type_index)) {
    return false;
  }

  if (block.diff_stream_prior_) {
    return false;
  }

  // A node can reuse blocks of the same stream and preorder streams
  if (CanReuseBlock(life_begin, block, block_size)) {
    return true;
  }
  return false;
}

bool IsSubGraphInOrOutNode(const ge::Node *const node, const ge::ComputeGraphPtr &compute_graph) {
  if (node != nullptr) {
    auto owner_graph = node->GetOwnerComputeGraphBarePtr();
    if (owner_graph != compute_graph.get()) {
      std::string op_type(node->GetTypePtr());
      if ((op_type == ge::DATA) || (op_type == ge::NETOUTPUT) || (op_type == ge::PARTITIONEDCALL)) {
        return true;
      }
    }
  }
  return false;
}

bool IsDirectOutputNode(const ge::Node *const node, const ge::ComputeGraphPtr &compute_graph) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)
      || (node->GetOpDescBarePtr()->GetType() != ge::NETOUTPUT)
      || IsSubGraphNetOutNode(node, compute_graph)) {
    return false;
  }
  GELOGD("This is model netoutput node:%s.", node->GetNamePtr());
  return true;
}

bool IsDirectInputNode(const ge::Node *const node, const ge::ComputeGraphPtr &compute_graph) {
  if (node != nullptr) {
    auto owner_graph = node->GetOwnerComputeGraphBarePtr();
    if ((owner_graph == compute_graph.get()) && ge::OpTypeUtils::IsDataNode(node->GetType())) {
      GELOGD("This is model input node:%s.", node->GetNamePtr());
      return true;
    }
  }
  return false;
}

bool IsAtomicWorkSpace(const int64_t index, const std::map<std::string, std::map<int64_t, int64_t>> &atomic_workspace) {
  for (auto const &it : atomic_workspace) {
    if (it.second.empty()) {
      continue;
    }
    for (const auto &workspae_info : it.second) {
      if (workspae_info.first == index) {
        GELOGD("Node:%s's workspace:%ld is atomic.", it.first.c_str(), index);
        return true;
      }
    }
  }
  return false;
}

static bool CompareNodeId(const ge::InDataAnchor *const left, const ge::InDataAnchor *const right) {
  bool invalid_para = ((left == nullptr) || (left->GetPeerOutAnchor() == nullptr)
      || (left->GetPeerOutAnchor()->GetOwnerNodeBarePtr() == nullptr)
      || (left->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr)
      || (right == nullptr) || (right->GetPeerOutAnchor() == nullptr)
      || (right->GetPeerOutAnchor()->GetOwnerNodeBarePtr() == nullptr)
      || (right->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr));
  if (invalid_para) {
    return false;
  }
  return (left->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr()->GetId()
    < right->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr()->GetId());
}

// Ensure that the memory release order is consistent with the topo order
std::vector<ge::InDataAnchor *> GetSortAllInDataAnchors(const ge::NodePtr &node, const bool memory_priority_mode) {
  std::vector<ge::InDataAnchor *> anchors;
  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    anchors.emplace_back(in_anchor);
  }
  if (memory_priority_mode) {
    std::sort(anchors.begin(), anchors.end(), CompareNodeId);
  }
  return anchors;
}

void HandleDependentStreamRedundantInfo(
    std::pair<const int64_t, std::map<int64_t, std::set<ge::EdgeLife, ge::CompareEdgeLife>>> &in_stream_edge) {
  for (auto &depend_stream_info : in_stream_edge.second) {
    size_t in_node_id = 0U;
    size_t out_node_id = 0U;
    if (depend_stream_info.second.size() <= 1U) {
      continue;
    }
    for (auto iter = depend_stream_info.second.begin(); iter != depend_stream_info.second.end();) {
      if ((iter->node_id >= in_node_id) && (iter->peer_node_id <= out_node_id)) {
        GELOGI("[StreamEdge]In depend Node: stream_id:[%ld<-%ld] life_time:[%zu<-%zu] delete", in_stream_edge.first,
               depend_stream_info.first, iter->node_id, iter->peer_node_id);
        iter = depend_stream_info.second.erase(iter);
        continue;
      }
      in_node_id = iter->node_id;
      out_node_id = iter->peer_node_id;
      ++iter;
    }
  }
}

void HandleInStreamRedundantDependence(ge::DiffStreamEdgeLife &in_stream_edges) {
  for (auto &in_stream_edge : in_stream_edges) {
    HandleDependentStreamRedundantInfo(in_stream_edge);
  }
}

bool NotMatchNoReuseType(const std::set<std::string> &no_reuse_types, const std::string &type) {
  // Match BaseType, BaseTypeV1~BaseTypeV4
  const auto type_length = type.length();
  const std::string::size_type version_length = 2U;
  if ((type_length > version_length)
      && ((type.at(type_length - version_length) == 'V'))
      && (type.at(type_length - 1U) >= '1')
      && (type.at(type_length - 1U) <= '4')) {
    return (no_reuse_types.count(type.substr(0, type_length - version_length)) == 0UL);
  }
  return (no_reuse_types.count(type) == 0UL);
}
}  // namespace

namespace ge {
bool CrossLifeTime(const NodeTypeIndex &left, const NodeTypeIndex &right) {
  if ((left.node_ == nullptr) || (right.node_ == nullptr)) {
    return true;
  }
  auto left_node_op_desc = left.node_->GetOpDescBarePtr();
  auto right_node_op_desc = right.node_->GetOpDescBarePtr();
  if ((left_node_op_desc != nullptr) && (right_node_op_desc != nullptr)) {
    if (left.GetLifeBegin() < right.GetLifeBegin()) {
      if (left.life_time_end_ >= right.GetLifeBegin()) {
        return true;
      }
    } else if (left.GetLifeBegin() == right.GetLifeBegin()) {
      return true;
    } else {
      if (right.life_time_end_ >= left.GetLifeBegin()) {
        return true;
      }
    }
  }
  return false;
}

/// When child block's life time are not cross with parent block, they can be reused(only same stream).
/// |-----------------------------parent block---------------------|
/// |------child block1--------------||------child block2------|
/// |--child block1-1-|
static bool CanIntervalLifeReuse(const MemoryBlock &parent_block, MemoryBlock &child_block,
                          std::vector<MemoryBlock *> &clone_blocks) {
  // judge by interval life time, only same stream can be judged by interval life time
  bool not_same_stream = ((parent_block.stream_id_ != child_block.stream_id_) || (!parent_block.same_stream_) ||
                          (!child_block.same_stream_) || parent_block.NodeTypeIndexList().empty() ||
                          child_block.NodeTypeIndexList().empty() ||
                          (parent_block.NodeTypeIndexList().back().diff_stream_life_time_.size() > 0U) ||
                          (child_block.NodeTypeIndexList().back().diff_stream_life_time_.size() > 0U));
  if (not_same_stream) {
    return false;
  }
  if (parent_block.IsBlockTypeConflict(child_block)) {
    GELOGD("block type conflict, parent_block: %s(%s), child_block: %s(%s).", GetName(parent_block).c_str(),
           parent_block.BlockTypeStr().c_str(), GetName(child_block).c_str(), child_block.BlockTypeStr().c_str());
    return false;
  }
  bool can_interval_life_reuse = false;
  auto clone_block = child_block.Clone();
  if (clone_block == nullptr) {
    return false;
  }

  bool same_size = ((child_block.NodeTypeIndexList().size() == child_block.RealSizeList().size())
      && (child_block.NodeTypeIndexList().size() == child_block.NoAlignSizeList().size()));
  // ref node must keep in same block
  bool pre_node_cross = false;
  if (same_size) {
    for (auto it = child_block.NodeTypeIndexList().cbegin(); it != child_block.NodeTypeIndexList().cend();) {
      bool cross_node = (((*it).ref_input_ && pre_node_cross)
          || ((!(*it).ref_input_) && parent_block.CrossLifeTimeNode(it, child_block)));
      if (cross_node) {
        size_t node_pos = it - child_block.NodeTypeIndexList().cbegin();
        clone_block->AddNodeTypeIndex(*it, child_block.RealSizeList()[node_pos],
                                      child_block.NoAlignSizeList()[node_pos], child_block.stream_id_);
        it = child_block.DelNode(it);
        pre_node_cross = true;
      } else {
        can_interval_life_reuse = true;
        pre_node_cross = false;
        ++it;
      }
    }
  }
  child_block.UpdateContinuousFlag();
  // all life times cross, keep this block
  if (child_block.NodeTypeIndexList().empty()) {
    child_block.Swap(*clone_block);
    delete clone_block;
  } else {
    // partial life times cross, clone a new cross block
    if (!clone_block->NodeTypeIndexList().empty()) {
      clone_blocks.emplace_back(clone_block);
    } else {
      // no life time cross
      delete clone_block;
    }
  }
  if (can_interval_life_reuse) {
    GELOGD("Block size[%zu, %zu] life time are not cross.", parent_block.Size(), child_block.Size());
  }
  return can_interval_life_reuse;
}

// Memory size is fixed and has nothing to do with different batches.
bool SizeIndependentOfBatch(const std::string &node_type) {
  static const std::unordered_set<std::string> kSizeIndependentOps = {
    HCOMBROADCAST, HVDCALLBACKBROADCAST, HCOMALLREDUCE, HVDCALLBACKALLREDUCE, HCOMALLGATHER, HVDCALLBACKALLGATHER
  };
  return (kSizeIndependentOps.count(node_type) != 0UL);
}

static void GetDiffStreamMinLifeTime(const Node *const node, const int64_t src_stream,
                              const DiffStreamEdgeLife &in_stream_edge, int64_t &min_life_time) {
  const auto node_op_desc = node->GetOpDescBarePtr();
  const auto dst_stream = GetStreamId(node_op_desc);
  if (dst_stream == src_stream) {
    min_life_time = node_op_desc->GetId();
    GELOGI("same stream, node[%s] id as min life[%ld]", node_op_desc->GetNamePtr(), min_life_time);
    return;
  }

  const auto it = in_stream_edge.find(dst_stream);
  if (it != in_stream_edge.cend()) {
    const auto edges_it = it->second.find(src_stream);
    if (edges_it != it->second.cend()) {
      auto edge_it = edges_it->second.lower_bound({static_cast<size_t>(node_op_desc->GetId()), 0UL});
      if (edge_it != edges_it->second.end()) {
        if ((edge_it->node_id > static_cast<size_t>(node_op_desc->GetId())) && (edge_it != edges_it->second.begin())) {
          --edge_it;
        }
        if (edge_it->node_id <= static_cast<size_t>(node_op_desc->GetId())) {
          min_life_time = (*edge_it).peer_node_id;
          GELOGI("diff stream, get min life[%ld], node[%s], id[%ld], stream_id:[%ld<-%ld] life_time:[%zu<-%zu]",
                 min_life_time, node_op_desc->GetNamePtr(), node_op_desc->GetId(), dst_stream, src_stream,
                 (*edge_it).node_id, min_life_time);
          return;
        }
      }
    }
  }
  min_life_time = kMinLifeTime;
  GELOGI("diff stream, get default min life[%ld], node[%s], id[%ld], stream_id[%ld<-%ld]",
         min_life_time, node_op_desc->GetNamePtr(), node_op_desc->GetId(), dst_stream, src_stream);
}

void GetDiffStreamMaxLifeTime(const Node *const node, const int64_t stream_id,
                              const DiffStreamEdgeLife &diff_stream_edge_life, int64_t &max_life_time) {
  max_life_time = kMaxLifeTime;
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);
  GELOGD("Out depend node:[%s] life begin:%ld stream_id:[%ld->%ld]", node_op_desc->GetNamePtr(),
         node_op_desc->GetId(), GetStreamId(node_op_desc), stream_id);
  if (GetStreamId(node_op_desc) == stream_id) {
    max_life_time = node_op_desc->GetId();
    return;
  }
  const auto it = diff_stream_edge_life.find(GetStreamId(node_op_desc));
  if (it == diff_stream_edge_life.cend()) {
    return;
  }
  const auto edges_it = it->second.find(stream_id);
  if (edges_it == it->second.cend()) {
    return;
  }
  const auto edge_it =
      edges_it->second.lower_bound({static_cast<size_t>(node_op_desc->GetId()), 0UL});
  if (edge_it == edges_it->second.end()) {
    return;
  }
  GELOGD("Node:[%s] life begin:%ld stream_id:[%ld->%ld] life_time:[%ld->%ld]", node_op_desc->GetNamePtr(),
         node_op_desc->GetId(), GetStreamId(node_op_desc), stream_id, (*edge_it).node_id, (*edge_it).peer_node_id);
  max_life_time = (*edge_it).peer_node_id;
}

int64_t GetNodeMaxLifeBySymbol(const SymbolToAnchors &symbol_to_anchors, const Node *const n, uint32_t out_index,
                               int64_t &max_node_life_time_by_symbol, std::set<int64_t> &streams,
                               const DiffStreamEdgeLife &diff_stream_edge_life, int64_t stream_id = kInvalidStreamId) {
  NodeIndexIO out_node_index_io(n, out_index, kOut);
  const int64_t n_stream_id = (stream_id == kInvalidStreamId) ? GetStreamId(n->GetOpDescBarePtr()) : stream_id;
  SymbolToAnchors::const_iterator iter =
    symbol_to_anchors.find(out_node_index_io.ToString());
  // 先初始化返回值的为该节点本身的起使生命周期
  int64_t max_node_life_time = n->GetOpDescBarePtr()->GetId();
  if (iter != symbol_to_anchors.cend()) {
    for (const auto &node_index_io : iter->second) {
      if ((node_index_io.io_type_ != kIn) || (node_index_io.node_ptr_ == nullptr) ||
          (node_index_io.node_ptr_->GetOpDescBarePtr() == nullptr)) {
        continue;
      }
      const int64_t in_anchor_stream_id = GetStreamId(node_index_io.node_ptr_->GetOpDescBarePtr());
      if (node_index_io.node_ptr_->GetOpDescBarePtr()->GetOpKernelLibName() != kEngineNameGeLocal) {
        streams.emplace(in_anchor_stream_id);
      }
      /* max_node_life_time_by_symbol 返回值有使用，在函数SetOutStreamLifeTime会使用，不能赋值错误 */
      if (node_index_io.node_ptr_->GetOpDescBarePtr()->GetId() > max_node_life_time_by_symbol) {
        max_node_life_time_by_symbol = node_index_io.node_ptr_->GetOpDescBarePtr()->GetId();
        max_node_life_time = (max_node_life_time_by_symbol > max_node_life_time)
            ? max_node_life_time_by_symbol : max_node_life_time;
        GELOGI("Node[%s] stream[%ld] output[%u]'s life time by symbol [%ld][%ld], node_io[%s], stream_id[%ld].",
            n->GetNamePtr(), n_stream_id, out_index, max_node_life_time_by_symbol, max_node_life_time,
            node_index_io.node_ptr_->GetNamePtr(), in_anchor_stream_id);
      }
      if (n_stream_id != in_anchor_stream_id) {
        int64_t diff_stream_life_time_end = kMaxLifeTime;
        GetDiffStreamMaxLifeTime(node_index_io.node_ptr_, n_stream_id, diff_stream_edge_life,
                                 diff_stream_life_time_end);
        GELOGI("Node[%s] stream[%ld] output[%u]'s life time is max of [%ld][%ld][%ld], node_io[%s], stream_id[%ld].",
            n->GetNamePtr(), n_stream_id, out_index, max_node_life_time_by_symbol, max_node_life_time,
            diff_stream_life_time_end, node_index_io.node_ptr_->GetNamePtr(), in_anchor_stream_id);
        /* max_node_life_time_by_symbol 在此分支中不能赋值,此分支只影响最大值 */
        max_node_life_time = std::max(max_node_life_time_by_symbol,
                                      std::max(diff_stream_life_time_end, max_node_life_time));
      }
    }
  }

  // info日志, 打印node的生命周期
  GELOGI("Node[%s] output[%u]'s max life time[%ld][%ld].", n->GetNamePtr(), out_index, max_node_life_time_by_symbol,
         max_node_life_time);

  return max_node_life_time;
}

int64_t GetNodeMaxLife(const SymbolToAnchors &symbol_to_anchors,
  const DiffStreamEdgeLife &diff_stream_edge_life, const Node *const n, uint32_t out_index,
  int64_t &max_node_life_time_by_symbol, std::set<int64_t> &streams, int64_t stream_id = kInvalidStreamId) {
  const int64_t max_node_life_time = GetNodeMaxLifeBySymbol(symbol_to_anchors, n, out_index,
    max_node_life_time_by_symbol, streams, diff_stream_edge_life, stream_id);
  GELOGD("Node[%s] output[%u]'s max life time[%ld].", n->GetNamePtr(), out_index, max_node_life_time);
  return max_node_life_time;
}

void GetContinuousOutputMaxLife(const NodePtr &node, const SymbolToAnchors &symbol_to_anchors,
                                const DiffStreamEdgeLife &out_stream_edges, int64_t &max_life_time,
                                std::set<int64_t> &streams) {
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);
  for (uint32_t index = 0U; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    const int64_t
        life_time = GetNodeMaxLife(symbol_to_anchors, out_stream_edges, node.get(), index, max_life_time, streams);
    if (life_time > max_life_time) {
      max_life_time = life_time;
    }
  }
  GELOGI("Continuous output node:%s max life time:%ld", node->GetNamePtr(), max_life_time);
}

void GetContinuousOutputMaxLifeBySymbol(const Node *const node,
                                        const SymbolToAnchors &symbol_to_anchors,
                                        int64_t &max_life_time,
                                        const DiffStreamEdgeLife &diff_stream_edge_life) {
  std::set<int64_t> streams;
  const auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);
  for (uint32_t index = 0U; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    /* max_life_time 在此函数中已经进行最大值的赋值处理 */
    (void)GetNodeMaxLifeBySymbol(symbol_to_anchors, node, index, max_life_time, streams, diff_stream_edge_life);
  }
  GELOGI("Continuous output node:%s max life time:%ld by symbol", node->GetNamePtr(), max_life_time);
}

Status SetChildHeadOffset(size_t offset, size_t max_offset, std::vector<MemoryBlock *> &blocks) {
  for (auto block : blocks) {
    if (block != nullptr) {
      GE_ASSERT_SUCCESS(block->SetHeadOffset(offset), "set head offset failed, offset: %zu, block head offset: %zu,"
                                                      " max_offset: %zu", offset, block->HeadOffset(), max_offset);
      offset += block->Size();
      GE_ASSERT_TRUE(offset <= max_offset, "offset: %zu, max_offset: %zu", offset, max_offset);
    }
  }
  return SUCCESS;
}

void SetChildTailOffset(size_t offset, std::vector<MemoryBlock *> &blocks) {
  for (auto block : blocks) {
    if (block != nullptr) {
      offset += block->Size();
      block->SetTailOffset(offset - 1UL);
    }
  }
}

Status MemoryBlock::SetHeadOffset(size_t offset) {
  head_offset_ = offset;
  GE_ASSERT_TRUE(head_offset_ < std::numeric_limits<size_t>::max() - block_size_,
                 "head_offset_: %zu, block_size_: %zu", head_offset_, block_size_);
  const auto max_offset = head_offset_ + block_size_;
  GE_ASSERT_SUCCESS(SetChildHeadOffset(head_offset_, max_offset,  child_blocks_),
                    "set child block failed, head_offset: %zu, max_offset: %zu", head_offset_, max_offset);
  GE_ASSERT_SUCCESS(SetChildHeadOffset(head_offset_, max_offset,  sub_graph_blocks_),
                    "set subgraph block failed, head_offset: %zu, max_offset: %zu", head_offset_, max_offset);
  for (auto &blocks : batch_to_blocks_) {
    GE_ASSERT_SUCCESS(SetChildHeadOffset(head_offset_, max_offset, blocks.second),
                      "set batch block failed, head_offset: %zu, max_offset: %zu", head_offset_, max_offset);
  }
  return SUCCESS;
}

void MemoryBlock::SetTailOffset(size_t offset) {
  tail_offset_ = offset;
  SetChildTailOffset(head_offset_, child_blocks_);
  SetChildTailOffset(head_offset_, sub_graph_blocks_);
  for (auto &blocks : batch_to_blocks_) {
    SetChildTailOffset(head_offset_, blocks.second);
  }
}

std::vector<MemoryBlock *> MemoryBlock::AllChildBlockList() const {
  std::vector<MemoryBlock *> return_child_blocks;
  return_child_blocks.insert(return_child_blocks.end(), sub_graph_blocks_.cbegin(), sub_graph_blocks_.cend());
  for (auto &batch_blocks : batch_to_blocks_) {
    return_child_blocks.insert(return_child_blocks.end(), batch_blocks.second.cbegin(), batch_blocks.second.cend());
  }
  return_child_blocks.insert(return_child_blocks.end(), child_blocks_.cbegin(), child_blocks_.cend());
  return return_child_blocks;
}

void MemoryBlock::Resize() {
  size_t child_block_size = 0;
  for (auto block : child_blocks_) {
    if (block != nullptr) {
      block->Resize();
      child_block_size += block->Size();
    }
  }
  auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if (iter == real_size_list_.end()) {
    GELOGW("real_size_list_ is empty");
    return;
  } else {
    size_t block_size = (child_block_size > *iter) ? child_block_size : *iter;
    if ((block_size > 0UL) && (block_size % MEM_ALIGN_SIZE != 0UL)) {
      MemReuseUtils::AlignMemOffset(block_size);
    }
    block_size_ = block_size;
  }
}

size_t MemoryBlock::AlignSize() {
  // Only one calculation, performance optimization
  if (max_real_size_ == 0UL) {
    auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
    if (iter == real_size_list_.end()) {
      GELOGW("real_size_list_ is empty");
    } else {
      max_real_size_ = *iter;
      if ((max_real_size_ > 0UL) && ((max_real_size_ % MEM_ALIGN_SIZE) != 0UL)) {
        MemReuseUtils::AlignMemOffset(max_real_size_);
      }
    }
  }
  return max_real_size_;
}

bool MemoryBlock::IsSameBatchLabel() const {
  // only same batch label can reuse
  if (batch_label_.empty() || node_type_index_list_.empty()) {
    return false;
  }

  bool all_same_label = true;
  for (size_t index = 1UL; index < node_type_index_list_.size(); ++index) {
    if (node_type_index_list_[index].node_ == nullptr) {
      continue;
    }
    std::string batch_label;
    const auto index_op_desc = node_type_index_list_[index].node_->GetOpDescBarePtr();
    GE_IF_BOOL_EXEC(index_op_desc == nullptr, continue);
    // not all op has ATTR_NAME_BATCH_LABEL, no need check return value, only check out parameter
    (void)ge::AttrUtils::GetStr(index_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    if (batch_label_ != batch_label) {
      all_same_label = false;
      break;
    }
  }
  return all_same_label;
}

bool MemoryBlock::IsGraphInputAndGetSize(const ComputeGraphPtr &compute_graph, size_t &size) const {
  for (const auto &node_type_index : node_type_index_list_) {
    const auto node = node_type_index.node_;
    if (IsDirectInputNode(node, compute_graph)) {
      size = node_type_index.no_align_size_;
      GELOGD("Node:%s is input of %s, size=%zu", node->GetNamePtr(), compute_graph->GetName().c_str(), size);
      return true;
    }
  }
  return false;
}

void MemoryBlock::AddContinuousLifeReuseBlock(MemoryBlock &block) {
  // continuous memory case:only real_size is maximum can be reused and only one continuous memory in one block
  auto it_block = std::max_element(std::begin(block.NoAlignSizeList()), std::end(block.NoAlignSizeList()));
  auto it_this = std::max_element(std::begin(NoAlignSizeList()), std::end(NoAlignSizeList()));
  if (it_block != std::end(block.NoAlignSizeList()) && it_this != std::end(NoAlignSizeList())) {
    if ((IsNoAlignSizeReuseBlock() && block.IsNoAlignSizeReuseBlock()) ||
        (IsNoAlignSizeReuseBlock() && (*it_this < *it_block)) ||
        (block.IsNoAlignSizeReuseBlock() && (*it_this > *it_block))) {
      GELOGD("Conflict current block size:%zu continuous:%d, reuse block max size:%zu continuous:%d.",
             *it_this, GetContinuousFlag(), *it_block, block.GetContinuousFlag());
      return;
    }
  }
  if (IsBlockTypeConflict(block)) {
    GELOGD("block type conflict, this: %s(%s), param block: %s(%s).", GetName(*this).c_str(), BlockTypeStr().c_str(),
           GetName(block).c_str(), block.BlockTypeStr().c_str());
    return;
  }
  // merge small block to large block
  MemoryBlock *parent = nullptr;
  MemoryBlock *child = nullptr;
  if (((child_offset_ + block.AlignSize()) <= *it_this) && (IsNoAlignSizeReuseBlock())) {
    parent = this;
    child  = &block;
   } else if (((block.child_offset_ + AlignSize()) <= *it_block) && (block.IsNoAlignSizeReuseBlock()) &&
              (AlignSize() == block.AlignSize()) && child_blocks_.empty()) {
    parent = &block;
    child  = this;
  } else {
    return;
  }

  parent->child_blocks_.emplace_back(child);
  parent->child_offset_ += child->AlignSize();
  child->child_block_ = true;
  GELOGI("[no_align_size_block_reuse]"
         "Add block[%s size:%zu, stream id:%ld, life time[begin:%zu, end:%zu], continuous:%d]"
         " to block[%s size:%zu, stream id:%ld, life time[begin:%zu, end:%zu], continuous:%d]",
         GetName(*child).c_str(), child->block_size_, child->stream_id_, child->GetLifeBegin(),
         child->GetLifeEnd(child->stream_id_), child->GetContinuousFlag(),
         GetName(*parent).c_str(), parent->block_size_, parent->stream_id_, parent->GetLifeBegin(),
         parent->GetLifeEnd(parent->stream_id_), parent->GetContinuousFlag());

  return;
}

void MemoryBlock::AddZeroCopyLifeReuseBlock(MemoryBlock &block) {
  auto it_block = std::max_element(block.real_size_list_.begin(), block.real_size_list_.end());
  auto it_this = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if ((it_block == block.real_size_list_.end()) || (it_this == real_size_list_.end())) {
    return;
  }
  if ((is_zero_copy_ && block.is_zero_copy_) ||
      (is_zero_copy_ && (*it_this < *it_block)) ||
      (block.is_zero_copy_ && (*it_this > *it_block))) {
    GELOGD("Conflict current block size:%zu is_reuse_zero_copy:%d is_zero_copy:%d, "
        "reuse block max size:%zu is_reuse_zero_copy:%d is_zero_copy:%d.", *it_this, is_reuse_zero_copy_,
        is_zero_copy_, *it_block, block.is_reuse_zero_copy_, block.is_zero_copy_);
    return;
  }
  if (IsBlockTypeConflict(block)) {
    GELOGD("block type conflict, this: %s(%s), param block: %s(%s).", GetName(*this).c_str(), BlockTypeStr().c_str(),
           GetName(block).c_str(), block.BlockTypeStr().c_str());
    return;
  }
  MemoryBlock *parent = nullptr;
  MemoryBlock *child = nullptr;
  // 如果child_offset_ 都为0，且 real_size 都相等，也是允许复用
  if ((((child_offset_ + block.AlignSize()) <= *it_this) || ((child_offset_ == 0UL) && (block.child_offset_ == 0UL) &&
      (*it_block == *it_this))) && is_zero_copy_) {
    parent = this;
    child  = &block;
   } else if ((((block.child_offset_ + AlignSize()) <= *it_block) || ((child_offset_ == 0UL) &&
              (block.child_offset_ == 0UL) && (*it_block == *it_this))) && block.is_zero_copy_ &&
              (AlignSize() == block.AlignSize()) && child_blocks_.empty()) {
    parent = &block;
    child  = this;
  } else {
    return;
  }

  if ((parent->is_zero_copy_) && (!child->is_reuse_zero_copy_)) {
    return;
  }

  parent->child_blocks_.emplace_back(child);
  parent->child_offset_ += child->AlignSize();
  child->child_block_ = true;
  parent->is_reuse_zero_copy_ = (child->is_reuse_zero_copy_ && parent->is_reuse_zero_copy_);
  GELOGI("[zero_copy_size_block_reuse]"
         "Add block[%s size:%zu, stream id:%ld, life time[begin:%zu, end:%zu], continuous:%d, is_zero_copy:%d]"
         " to block[%s size:%zu, stream id:%ld, life time[begin:%zu, end:%zu], continuous:%d, is_zero_copy:%d]",
         GetName(*child).c_str(), child->block_size_, child->stream_id_, child->GetLifeBegin(),
         child->GetLifeEnd(child->stream_id_), child->GetContinuousFlag(), child->is_zero_copy_,
         GetName(*parent).c_str(), parent->block_size_, parent->stream_id_, parent->GetLifeBegin(),
         parent->GetLifeEnd(parent->stream_id_), parent->GetContinuousFlag(), parent->is_zero_copy_);

  return;
}

bool CanBlockLifeReuse(const BlockMemAssigner *const mem_assigner,
                       const MemoryBlock &in_block, const MemoryBlock &out_block,
                       DiffStreamEdgeLife &diff_stream_edge_life) {
  const auto first_node = out_block.NodeTypeIndexList().front();
  if ((first_node.mem_type_ == kOutput)
      && mem_assigner->HasSameOutAnchorWithDiffStream(first_node.node_, first_node.index_)) {
    GELOGD("out_block first node %s(topoid: %lld) output %u use same memory with node on other stream, return false.",
           first_node.node_->GetNamePtr(), first_node.node_->GetOpDescBarePtr()->GetId(), first_node.index_);
    return false;
  }
  if (in_block.IsBlockTypeConflict(out_block)) {
    GELOGD("block type conflict, in_block: %s(%s), out_block: %s.", GetName(in_block).c_str(),
           in_block.BlockTypeStr().c_str(), GetName(out_block).c_str(), out_block.BlockTypeStr().c_str());
    return false;
  }
  GELOGD("in_block[%s] out_block[%s]", GetName(in_block).c_str(), GetName(out_block).c_str());
  if (in_block.stream_id_ == out_block.stream_id_) {
    return (out_block.GetLifeBegin() > in_block.GetLifeEnd(out_block.stream_id_));
  } else {
    auto depend_node_id = out_block.GetDependLifeBegin(in_block.stream_id_, diff_stream_edge_life);
    /// |-stream 1-|         |-stream 2-|
    /// |node1-node3|        |--block---|
    /// |node2-node4-node6|  |--block---|
    /// |--block4--|       \ |--block5---|
    /// |--block---|        \_
    ///                      |node11-node13-node15|
    ///                      |node17-node19|
    /// edge node(node6) is the last in the block, node17 can reuse node6,node3
    size_t tail_node_id = 0UL;
    const auto &node_type_index = in_block.NodeTypeIndexList().back();
    if (node_type_index.node_ != nullptr) {
      auto node_op_desc = node_type_index.node_->GetOpDescBarePtr();
      if (node_op_desc != nullptr) {
        tail_node_id = static_cast<size_t>(node_op_desc->GetId());
      }
    }
    if ((tail_node_id != 0UL) && (depend_node_id >= tail_node_id)) {
      if (!in_block.GetReuseStrategy().memory_priority_mode_) {
        return (depend_node_id > in_block.GetLifeEnd(out_block.stream_id_));
      }
      int64_t end_stream_id = kInvalidStreamId;
      auto in_block_life_end = in_block.GetLifeEnd(out_block.stream_id_, end_stream_id);
      if (end_stream_id == out_block.stream_id_) {
        return (out_block.GetLifeBegin() > in_block_life_end);
      }
      if ((end_stream_id != in_block.stream_id_) && (end_stream_id != kInvalidStreamId)) {
        return (out_block.GetDependLifeBegin(end_stream_id, diff_stream_edge_life) >= in_block_life_end);
      }
      return (depend_node_id > in_block_life_end);
    }
  }
  return false;
}

bool MemoryBlock::AddLifeReuseBlock(const BlockMemAssigner *const mem_assigner,
                                    MemoryBlock *block, std::vector<MemoryBlock *> &clone_blocks, uint32_t depth,
                                    DiffStreamEdgeLife &diff_stream_edge_life, bool child_reuse) {
  GELOGD("this[%s size:%zu, stream id:%ld life time[begin:%zu, end:%zu] childs:%zu] "
    "block[%s size:%zu, stream id:%ld, life time[begin:%zu, end:%zu] childs:%zu]", GetName(*this).c_str(),
    block_size_, stream_id_, GetLifeBegin(), GetLifeEnd(block->stream_id_), child_blocks_.size(),
    GetName(*block).c_str(), block->block_size_, block->stream_id_, block->GetLifeBegin(),
    block->GetLifeEnd(stream_id_), block->child_blocks_.size());
  ++depth;
  const bool can_not_life_reuse = (CanNotLifeReuse(*this, child_reuse) || CanNotLifeReuse(*block) ||
      (batch_label_ != block->batch_label_) || (memory_type_ != block->memory_type_) || (depth > kMaxDepthNum));
  if (can_not_life_reuse || (!block->child_blocks_.empty())) {
    return false;
  }

  // Different streams must use stream dependency to judge the life cycle
  // In case same stream if it has child block, can judge all the child block's life time in CanIntervalLifeReuse
  bool can_block_life_reuse = CanBlockLifeReuse(mem_assigner, *this, *block, diff_stream_edge_life) ||
                              CanBlockLifeReuse(mem_assigner, *block, *this, diff_stream_edge_life);
  const bool is_continue_reuse_zero_copy = (is_zero_copy_ && (block->GetFirstContinuousFlag() ||
      block->GetLastContinuousFlag() || block->GetContinuousFlag())) || (block->is_zero_copy_ &&
      (GetFirstContinuousFlag() || GetLastContinuousFlag() || GetContinuousFlag()));
  GELOGD("continuous can not reuse zero copy, is_continue_not_reuse_zero_copy:%d", is_continue_reuse_zero_copy);
  if (is_continue_reuse_zero_copy) {
    return false;
  }
  // continuous block reuse proc
  const bool is_no_align_size_reuse_block = IsNoAlignSizeReuseBlock() || block->IsNoAlignSizeReuseBlock();
  if (is_no_align_size_reuse_block) {
    if (can_block_life_reuse) {
      AddContinuousLifeReuseBlock(*block);
    }
    return true;
  }
  // zero copy block reuse proc
  const bool is_real_size_reuse_block = IsRealSizeReuseBlock() || block->IsRealSizeReuseBlock();
  if (is_real_size_reuse_block) {
    if (can_block_life_reuse) {
      AddZeroCopyLifeReuseBlock(*block);
    }
    return true;
  }

  if (!can_block_life_reuse && !CanIntervalLifeReuse(*this, *block, clone_blocks)) {
    return false;
  }

  // |-parent block---------------------------------------|
  // |-child block level 1----|-child block level 1----|
  // |-child block level 2-|
  for (auto child_block : child_blocks_) {
    if ((child_block != nullptr) &&
        child_block->AddLifeReuseBlock(mem_assigner, block, clone_blocks, depth, diff_stream_edge_life, true)) {
      return true;
    }
  }

  // merge small block to large block
  // noalign size         802816 + 802816 = 1605632       can reuse
  // after 32 align size  802848 + 802848 > 1605664       can't reuse
  // after 512 align size 803328 + 803328 > 1606144       can't reuse
  // so                   803328 + 803328 = 1606144 + 512 can reuse
  if (block->AlignSize() != MEM_ALIGN_SIZE) {
    if ((child_offset_ + block->AlignSize()) > (AlignSize() + MEM_ALIGN_SIZE)) {
      return false;
    }
  } else {
    if ((child_offset_ + block->AlignSize()) > AlignSize()) {
      return false;
    }
  }

  child_blocks_.emplace_back(block);
  is_reuse_zero_copy_ = (block->is_reuse_zero_copy_ && is_reuse_zero_copy_);
  child_offset_ += block->AlignSize();
  block->child_block_ = true;
  GELOGI("Add block[%s size:%zu, stream id:%ld life time[begin:%zu, end:%zu]] to"
    " block[%s size:%zu, stream id:%ld, life time[begin:%zu, end:%zu]]", GetName(*block).c_str(), block->block_size_,
    block->stream_id_, block->GetLifeBegin(), block->GetLifeEnd(stream_id_), GetName(*this).c_str(), block_size_,
    stream_id_, GetLifeBegin(), GetLifeEnd(block->stream_id_));
  return true;
}

size_t MemoryBlock::GetLifeBegin(bool for_sort) const {
  if (!node_type_index_list_.empty()) {
    return node_type_index_list_.front().GetLifeBegin(for_sort);
  }
  return 0UL;
}

/// |-stream 1-|   |-stream 2-|
/// |--block1--|   |--block---|
/// |--block2--|   |--block---|
/// |--block3--|\  |--block---|
/// |--block4--| \ |--block5---|
/// |--block---|  \|--block6---|
/// |--block---|   |--block7--|
/// |--block---|   |--block---|
/// block7's first node's input node's life begin > block2's life end, block7 can reuse block1~block2
size_t MemoryBlock::GetDependLifeBegin(int64_t stream_id, DiffStreamEdgeLife &diff_stream_edge_life) const {
  GELOGD("In depend node:[%s] stream_id:[%ld->%ld] self life time[%ld-%ld]",
         NodeTypeIndexList().front().node_->GetNamePtr(), stream_id_, stream_id, GetLifeBegin(),
         GetLifeEnd(stream_id));
  const auto it = diff_stream_edge_life.find(stream_id_);
  if (it == diff_stream_edge_life.cend()) {
    return 0UL;
  }
  const auto edges_it = it->second.find(stream_id);
  if (edges_it == it->second.cend()) {
    return 0UL;
  }

  /// |-stream 1-|         |-stream 2-|
  /// |node1-node3|        |--block---|
  /// |node2-node4-node6|  |--block---|
  /// |--block4--|       \ |node7-node9|
  /// |--block---|        \_
  ///                      |node11-node13-node15|
  ///                      |node17-node19|
  auto first_node_id = GetLifeBegin();
  auto edge_it = edges_it->second.lower_bound({first_node_id, 0UL});
  if (edges_it->second.empty()) {
    return 0UL;
  }
  // lower_bound find node17, not found, so use node11-->node6
  if ((edge_it == edges_it->second.end()) || ((*edge_it).node_id > first_node_id)) {
    // lower_bound find node7, get node11-->node6, because node11 > node7, so return no depend node
    if (edge_it == edges_it->second.begin()) {
      GELOGD("Depend lower node id:%ld > node id:%ld.", (*edge_it).node_id, GetLifeBegin());
      return 0UL;
    }
    // not found, use tail data
    --edge_it;
  }

  // lower_bound find node11, get node11-->node6
  GELOGD("Node:[%s] life begin:%ld stream_id:[%ld->%ld] depend life_time:[%ld->%ld]",
         NodeTypeIndexList().front().node_->GetNamePtr(), first_node_id, stream_id_, stream_id, (*edge_it).node_id,
         (*edge_it).peer_node_id);
  return (*edge_it).peer_node_id;
}

// 这里stream_id和self stream_id可能不同，最终会和stream_id block->GetDependLifeBegin(self stream_id)比较确保正确性
size_t MemoryBlock::GetLifeEnd(int64_t stream_id) const {
  if (!node_type_index_list_.empty()) {
    const bool only_to_one_stream = (node_type_index_list_.back().out_stream_count_ == 1U)
        && (node_type_index_list_.back().diff_stream_life_time_.size() == 1U);
    const auto it = node_type_index_list_.back().diff_stream_life_time_.find(stream_id);
    if (only_to_one_stream && (it != node_type_index_list_.back().diff_stream_life_time_.cend())) {
      GELOGD("block %s stream[%ld] [%ld] life[%ld]", GetName(*this).c_str(), stream_id_,
             stream_id, it->second);
      return it->second;
    }

    GELOGD("block %s stream[%ld] [%ld] life[%ld]", GetName(*this).c_str(), stream_id_, stream_id,
           node_type_index_list_.back().life_time_end_);
    return node_type_index_list_.back().life_time_end_;
  }
  return kMaxLifeTime;
}

/// |-stream 1-|         |-stream 2-|     |-stream 3-|
/// |node1-node3|        |--block---|     |--block---|
/// |node2-node4-node6|
/// |--block4--|       \                  |--block---|
/// |--block---|        \                 |--block---|
///                      |node11|
///                      |node17-node19|
///                      |--block---|  \  |--block---|
///                                     \ |--block---|
///                                      |node30-node32|
///                                       |--block---|
size_t MemoryBlock::GetLifeEnd(int64_t stream_id, int64_t &end_stream_id) const {
  end_stream_id = stream_id_;
  if (!node_type_index_list_.empty()) {
    const bool only_to_one_stream = (node_type_index_list_.back().out_stream_count_ == 1U)
        && (node_type_index_list_.back().diff_stream_life_time_.size() == 1U);
    if (!only_to_one_stream) {
      // stream_id is 1, return normal end life time in stream 1 or kMaxLifeTime
      GELOGD("block %s stream[%ld] [%ld] life[%ld]",
             GetName(*this).c_str(), stream_id_, stream_id,
             node_type_index_list_.back().life_time_end_);
      return node_type_index_list_.back().life_time_end_;
    }
    const auto it = node_type_index_list_.back().diff_stream_life_time_.find(stream_id);
    // out to only one diff stream, stream_id is 2, end_stream_id is 2, return node11
    if (it != node_type_index_list_.back().diff_stream_life_time_.cend()) {
      GELOGD("block %s stream[%ld] [%ld] life[%ld]",
             GetName(*this).c_str(), stream_id_, stream_id, it->second);
      end_stream_id = stream_id;
      return it->second;
    }
    // out to only one diff stream, stream_id is 3, end_stream_id is 2, return node11
    if (stream_id != stream_id_) {
      end_stream_id = node_type_index_list_.back().diff_stream_life_time_.begin()->first;
      GELOGD("block %s stream[%ld] [%ld] [%ld] life[%ld]",
             GetName(*this).c_str(), stream_id_, end_stream_id, stream_id,
             node_type_index_list_.back().diff_stream_life_time_.begin()->second);
      return node_type_index_list_.back().diff_stream_life_time_.begin()->second;
    }
  }
  return kMaxLifeTime;
}

size_t MemoryBlock::GetSymbolLifeEnd() const {
  if (!node_type_index_list_.empty()) {
    return node_type_index_list_.back().symbol_max_life_time_end_;
  }
  return kDefaultLifeTime;
}

void MemoryBlock::SetSymbolLifeEnd(size_t symbol_life_end) {
  if (!node_type_index_list_.empty()) {
    if ((node_type_index_list_.back().symbol_max_life_time_end_ == kDefaultLifeTime) ||
        symbol_life_end > node_type_index_list_.back().symbol_max_life_time_end_) {
      node_type_index_list_.back().symbol_max_life_time_end_ = symbol_life_end;
    }
  }
}

void MemoryBlock::SetLifeTimeEnd(size_t time, int64_t stream_id) {
  if (!node_type_index_list_.empty()) {
    if (stream_id != stream_id_) {
      auto it = node_type_index_list_.back().diff_stream_life_time_.find(stream_id);
      if (it == node_type_index_list_.back().diff_stream_life_time_.end()) {
        node_type_index_list_.back().diff_stream_life_time_[stream_id] = time;
      } else if (time > it->second) {
        it->second = time;
      } else {}

      if (node_type_index_list_.back().life_time_end_ == kDefaultLifeTime) {
        node_type_index_list_.back().life_time_end_ = kMaxLifeTime;
      }
    } else {
      if ((node_type_index_list_.back().life_time_end_ == kDefaultLifeTime) ||
          (time > node_type_index_list_.back().life_time_end_)) {
        node_type_index_list_.back().life_time_end_ = time;
      }
    }
  }
}

void MemoryBlock::SetOutStreamLifeTime(size_t out_time, size_t end_time, int64_t stream_id) {
  const size_t symbol_life_time = GetSymbolLifeEnd();
  if ((symbol_life_time != kDefaultLifeTime) && (end_time < symbol_life_time)) {
    end_time = symbol_life_time;
    GELOGI("Block %s has continuous input node, which include multiple ref, end time is: %ld",
           GetName(*this, true).c_str(), end_time);
  }
  end_time = (end_time < out_time) ? kMaxLifeTime : end_time;
  if (!node_type_index_list_.empty()) {
    auto iter = node_type_index_list_.back().out_stream_life_time_.find(stream_id);
    if (iter == node_type_index_list_.back().out_stream_life_time_.end()) {
      node_type_index_list_.back().out_stream_life_time_.emplace(stream_id, std::make_pair(out_time, end_time));
      node_type_index_list_.back().SetOutStreamCount(node_type_index_list_.back().out_stream_life_time_.size());
      return;
    }

    if (out_time > iter->second.first) {
      iter->second.first = out_time;
    }
    if (end_time > iter->second.second) {
      iter->second.second = end_time;
    }
  }
}

bool MemoryBlock::CrossLifeTimeNode(const std::vector<NodeTypeIndex>::const_iterator &it,
                                    const MemoryBlock &child_block) const {
  if (node_type_index_list_.empty()) {
    return false;
  }

  const NodeTypeIndex &node_type_index = *it;
  // quick judge life time by begin and end
  if (!((node_type_index.life_time_end_ < node_type_index_list_.front().GetLifeBegin()) ||
        (node_type_index.GetLifeBegin() > node_type_index_list_.back().life_time_end_))) {
    for (const auto &node : node_type_index_list_) {
      if (CrossLifeTime(node, node_type_index)) {
        return true;
      }
    }
  }

  if (node_type_index.next_is_ref_input_) {
    // all ref node must in same block, judge all the ref node and return same result
    auto ref_it  = it;
    ref_it++;
    for (; ref_it != child_block.NodeTypeIndexList().cend(); ++ref_it) {
      if (!(*ref_it).ref_input_) {
        break;
      }
      for (const auto &node : node_type_index_list_) {
        if (CrossLifeTime(node, *ref_it)) {
          return true;
        }
      }
    }
  }
  return false;
}

MemoryBlock *MemoryBlock::Clone() const {
  auto block = new (std::nothrow) MemoryBlock(reuse_strategy_, block_size_, stream_id_, reuse_mem_, memory_type_);
  if (block != nullptr) {
    // 复用中作为判断条件的字段都需要clone，其他字段不需要clone
    block->same_stream_ = same_stream_;
    block->is_zero_copy_ = is_zero_copy_;
    block->is_reuse_zero_copy_ = is_reuse_zero_copy_;
    block->memory_type_logic_base_ = memory_type_logic_base_;
    block->need_same_offset_in_batch_ = need_same_offset_in_batch_;
    block->ref_count_ = ref_count_;
    block->input_index_ = input_index_;
    block->batch_label_ = batch_label_;
    block->has_sub_graph_in_out_node_ = has_sub_graph_in_out_node_;
    block->post_reuse_flag_ = post_reuse_flag_;
    block->is_fixed_addr_prior_ = is_fixed_addr_prior_;
    block->block_type_list_ = block_type_list_;
  }
  return block;
}

void MemoryBlock::UpdateContinuousFlag () {
  first_continuous_block_ = false;
  last_continuous_block_ = false;
  continuous_block_ = false;
  for (const auto &node : node_type_index_list_) {
    if (node.GetFirstContinuousNodeFlag()) {
      first_continuous_block_ = true;
    }
    if (node.GetLastContinuousNodeFlag()) {
      last_continuous_block_ = true;
    }
    if (node.GetContinuousNodeFlag()) {
      continuous_block_ = true;
    }
  }
}

// call UpdateContinuousFlag after DelNode
std::vector<NodeTypeIndex>::const_iterator MemoryBlock::DelNode(std::vector<NodeTypeIndex>::const_iterator &it) {
  // vector sizes are same
  if ((node_type_index_list_.size() == real_size_list_.size())
      && (node_type_index_list_.size() == no_align_size_list_.size())) {
    const auto to_delete = *it;
    size_t node_pos = it - node_type_index_list_.begin();
    auto return_it = node_type_index_list_.erase(it);
    real_size_list_.erase(real_size_list_.cbegin() + node_pos);
    no_align_size_list_.erase(no_align_size_list_.cbegin() + node_pos);
    block_type_list_.WithDeleted(*this, to_delete);
    return return_it;
  }
  return ++it;
}

void MemoryBlock::Swap(MemoryBlock &block) {
  node_type_index_list_.swap(block.node_type_index_list_);
  real_size_list_.swap(block.real_size_list_);
  no_align_size_list_.swap(block.no_align_size_list_);
  block_type_list_.swap(block.block_type_list_);
}

void SetLastUsedInputMemAttr(const NodePtr &node, int32_t input_index, std::vector<TAttr<bool>> &bool_attr) {
  if (node == nullptr) {
    return;
  }
  auto node_op_desc = node->GetOpDescBarePtr();
  if (node_op_desc != nullptr) {
    auto input_desc = node_op_desc->MutableInputDesc(input_index);
    if (input_desc == nullptr) {
      return;
    }
    bool_attr.emplace_back(input_desc.get(), node_op_desc, input_index, ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  }
}

std::string MemoryBlock::String() const {
  std::stringstream ss;
  ss << "Block size: " << Size() << " from " << HeadOffset() << " to " << TailOffset() << " ";
  ss << "ref_count: " << ref_count_ << " ";
  ss << "stream_id: " << stream_id_ << " ";
  ss << "is_zero_copy: " << is_zero_copy_ << " ";
  ss << "reuse_mem_: " << reuse_mem_ << " ";
  ss << "no_align_size: " << ToString(no_align_size_list_) << " ";
  ss << "real_size_list: " << ToString(real_size_list_) << " ";
  ss << "members: ";
  for (auto x : NodeTypeIndexList()) {
    ss << "__node: " << ::ToString(x) << " ";
  }
  for (const auto& symbol : SymbolList()) {
    ss << "__symbol: " << symbol << " ";
  }
  ss << "memory_type: " << memory_type_ << " ";
  return ss.str();
}

BlockMemAssigner::BlockMemAssigner(const MemAssistInfo &mem_assist_info)
    : compute_graph_(mem_assist_info.compute_graph), symbol_to_anchors_(mem_assist_info.symbol_to_anchors),
      anchor_to_symbol_(mem_assist_info.anchor_to_symbol), life_time_(0),
      parent_nodes_to_stream_ids_(mem_assist_info.parent_nodes_to_stream_ids) {
  std::string memory_optimization_policy;
  ge::GetContext().GetOption(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  if (memory_optimization_policy == kMemoryPriority) {
    memory_priority_mode_ = true;
  }

  strict_reuse_zero_memory_mode_ = IsStrictReuseZeroMemoryMode();
  (void)InitIoReuseFlag();
  ParseGraphIoAllocMode();

  std::string refreshable;
  (void)ge::GetContext().GetOption(ge::OPTION_FEATURE_BASE_REFRESHABLE, refreshable);
  is_feature_map_refreshable_ = (refreshable == "1");

  input_fusion_size_ = ge::GetContext().GetInputFusionSize();
  GELOGI("feature map refreshable: %d, input_fusion_size: %" PRIu64, is_feature_map_refreshable_, input_fusion_size_);
}

BlockMemAssigner::~BlockMemAssigner() {
  GELOGD("[Destruct][BlockMemAssigner]blocks_store_ size : %lu", blocks_store_.size());
  for (MemoryBlock *memory_block : blocks_store_) {
    GE_DELETE_NEW_SINGLE(memory_block);
  }
}

void GetMaxBatchAllMemorySize(std::map<std::string, std::vector<int64_t>> &batch_all_memory_size,
                              std::map<std::string, int64_t> batch_total_size, std::vector<int64_t> &all_memory_size,
                              std::string &max_batch_label) {
  // use max batch all memory size for reuse range
  int64_t max_batch_size = 0;
  for (const auto &it : batch_total_size) {
    GELOGI("Batch[%s] total memory size[%ld]", it.first.c_str(), it.second);
    // no batch label
    if (it.first.empty()) {
      continue;
    }
    if (it.second > max_batch_size) {
      max_batch_size = it.second;
      max_batch_label = it.first;
    }
  }
  GELOGI("Max batch[%s] total memory size[%ld]", max_batch_label.c_str(), max_batch_size);

  for (const auto &it : batch_all_memory_size) {
    if (it.first.empty() || (it.first == max_batch_label)) {
      all_memory_size.insert(all_memory_size.cend(), it.second.cbegin(), it.second.cend());
    }
  }
  // all_memory_size can't be empty
  if (all_memory_size.empty()) {
    all_memory_size.emplace_back(MEM_ALIGN_SIZE);
  }
  sort(all_memory_size.begin(), all_memory_size.end());
  GELOGD("All memory size: %s", ToString(all_memory_size).c_str());

  for (auto iter = all_memory_size.begin(); iter != all_memory_size.end();) {
    if (*iter == 0) {
      iter = all_memory_size.erase(iter);
    } else {
      ++iter;
    }
  }
}

void BlockMemAssigner::InsertStreamOutEdge() {
  for (const auto &dst_stream_to_edges : in_stream_edges_) {
    const auto dst_stream_id = dst_stream_to_edges.first;
    for (const auto &src_stream_to_edges : dst_stream_to_edges.second) {
      const auto src_stream_id = src_stream_to_edges.first;
      auto &out_stream_edge_set = out_stream_edges_[src_stream_id][dst_stream_id];
      for (const auto &edge : src_stream_to_edges.second) {
        out_stream_edge_set.insert({edge.peer_node_id, edge.node_id});
        GELOGI("[StreamEdge]Out depend Node: stream_id:[%ld->%ld] life_time:[%zu->%zu], only insert.",
               src_stream_id, dst_stream_id, edge.peer_node_id, edge.node_id);
      }
    }
  }
}

/*
 * stream1 stream2
 *    1------+
 *           |
 *    2 ---- 3
 *    |
 *    +------4
 * 对于stream2 的入边来讲
 * stream2<-stream1的入边  in_edge 3<-1 删掉
 *                                3<-2 保留
 * 可以简单记为：id差越小越好
 */
void BlockMemAssigner::InsertStreamInEdge(const EdgeLife &new_in_edge, const int64_t src_stream_id,
                                          const int64_t dst_stream_id, const char *src_name, const char *dst_name) {
  auto &in_edge_set = in_stream_edges_[dst_stream_id][src_stream_id];
  const auto old_in_edge_iter = in_edge_set.find(new_in_edge);
  if (old_in_edge_iter != in_edge_set.end()) {
    if (old_in_edge_iter->peer_node_id < new_in_edge.peer_node_id) {
      const auto old_peer_node_id = old_in_edge_iter->peer_node_id;
      in_edge_set.erase(old_in_edge_iter); // after erase, can not use old_peer_node_id below
      in_edge_set.insert(new_in_edge);
      if ((src_name != nullptr) && (dst_name != nullptr)) {
        GELOGI("[StreamEdge]In depend Node: [%s<-%s] stream_id:[%ld<-%ld] life_time:[%zu<-%zu], erase and insert,"
            " old_peer_node_id[%zu].", dst_name, src_name,
            dst_stream_id, src_stream_id, new_in_edge.node_id, new_in_edge.peer_node_id,
            old_peer_node_id);
      } else {
        GELOGI("[StreamEdge]In depend Node: stream_id:[%ld<-%ld] life_time:[%zu<-%zu], erase and insert,"
            " old_peer_node_id[%zu].", dst_stream_id, src_stream_id, new_in_edge.node_id, new_in_edge.peer_node_id,
            old_peer_node_id);
      }
    } else {
      if ((src_name != nullptr) && (dst_name != nullptr)) {
        GELOGI("[StreamEdge]In depend Node: [%s<-%s] stream_id:[%ld<-%ld] life_time:[%zu<-%zu], not erase, not insert, "
            "old_peer_node_id[%zu] >= new_peer_node_id[%zu].", dst_name, src_name,
            dst_stream_id, src_stream_id, new_in_edge.node_id, new_in_edge.peer_node_id,
            old_in_edge_iter->peer_node_id, new_in_edge.peer_node_id);
      } else {
        GELOGI("[StreamEdge]In depend Node: stream_id:[%ld<-%ld] life_time:[%zu<-%zu], not erase, not insert, "
            "old_peer_node_id[%zu] >= new_peer_node_id[%zu].", dst_stream_id, src_stream_id, new_in_edge.node_id,
            new_in_edge.peer_node_id, old_in_edge_iter->peer_node_id, new_in_edge.peer_node_id);
      }
    }
  } else {
    in_edge_set.insert(new_in_edge);
    if ((src_name != nullptr) && (dst_name != nullptr)) {
      GELOGI("[StreamEdge]In depend Node: [%s<-%s] stream_id:[%ld<-%ld] life_time:[%zu<-%zu], only insert.",
             dst_name, src_name, dst_stream_id, src_stream_id,
             new_in_edge.node_id, new_in_edge.peer_node_id);
    } else {
      GELOGI("[StreamEdge]In depend Node: stream_id:[%ld<-%ld] life_time:[%zu<-%zu], only insert.",
             dst_stream_id, src_stream_id, new_in_edge.node_id, new_in_edge.peer_node_id);
    }
  }
}

/*
 * stream1  stream2
 *   10--+
 *   20   \   30
 *   40 ---\->50
 *   60     \ 70
 *            90
 * 10->90
 * 40->50
 * only keep edge 40->50, 可以简单记为缩短peer_node_id与node_id差值
 */
static void EraseIntersectedEdge(std::set<EdgeLife, CompareEdgeLife> &in_edge_set,
                          const EdgeLife &old_in_edge,
                          const EdgeLife &new_in_edge,
                          const int64_t src_node_stream_id,
                          const int64_t dst_node_stream_id) {
  if (((old_in_edge.node_id > new_in_edge.node_id) && (old_in_edge.peer_node_id < new_in_edge.peer_node_id)) ||
      ((old_in_edge.node_id < new_in_edge.node_id) && (old_in_edge.peer_node_id > new_in_edge.peer_node_id))) {
    GELOGI("[StreamEdge]In depend Node: stream_id:[%ld<-%ld] erase life_time:[%zu<-%zu], will insert new "
        "life_time:[%zu<-%zu].", dst_node_stream_id, src_node_stream_id, old_in_edge.node_id, old_in_edge.peer_node_id,
        new_in_edge.node_id, new_in_edge.peer_node_id);
    auto it = in_edge_set.find(old_in_edge);
    if ((it != in_edge_set.end()) && (it->peer_node_id == old_in_edge.peer_node_id)) {
      in_edge_set.erase(it);
    }
  }
}

/*
 * 函数作用：
 * 该函数用于建立跨流的边，比如已有stream1->stream2->stream3, 建立stream1->stream3的边
 * 主要逻辑：在建立完stream2->stream3的边后，遍历所有到stream2的流（比如stream1），并建立该流到stream3的边。
 *
 * in_stream_edges : 入边
 * out_stream_edges : 出边
 * node_desc: 当前节点
 * in_node_desc: 输入节点
 *
 * 返回值：
 * 无返回值，函数直接修改传入的in_stream_edges和out_stream_edges。
 *
 * in edge: stream_id:[2<-1] life_time:[1<-0]
 * in edge: stream_id:[3<-2] life_time:[3<-1]
 * in edge: stream_id:[3<-1] life_time:[3<-0] new edge
 * in edge: stream_id:[3<-0] life_time:[3<-2]
 * in edge: stream_id:[1<-3] life_time:[4<-3]
 * in edge: stream_id:[1<-0] life_time:[4<-2] new edge
 * in edge: stream_id:[1<-1] life_time:[4<-0] new edge
 * in edge: stream_id:[1<-2] life_time:[4<-1] new edge
 */
void BlockMemAssigner::AddInStreamEdge(const ge::OpDesc *const node_desc, const ge::OpDesc *const in_node_desc) {
  const auto stream_id = GetStreamId(node_desc);
  const auto node_id = static_cast<size_t>(node_desc->GetId());
  const auto in_stream_id = GetStreamId(in_node_desc);
  const auto in_node_id = static_cast<size_t>(in_node_desc->GetId());
  for (const auto &stream_to_in_edges : in_stream_edges_[in_stream_id]) {
    const auto &in_edges = stream_to_in_edges.second;
    const auto third_stream_id = stream_to_in_edges.first;
    if (in_edges.empty() || (stream_id == third_stream_id)) {
      continue;
    }

    /*
     * 这段逻辑是目的是找到一条边，其peer_node_id作为stream_id<-third_stream_id的peer_node_id
     * 比如stream_id=3, in_stream_id=2, node_id=5, in_node_id=4, third_stream_id=1
     * in edge: stream_id:[3<-2] life_time:[5<-4]
     *
     * in edge: stream_id:[2<-1] life_time:[1<-0]
     * in edge: stream_id:[2<-1] life_time:[4<-2] <---edge_it 小于等于4的，最大的, 2就作为new_in_edge.peer_node_id
     * in edge: stream_id:[2<-1] life_time:[7<-3]
     */
    auto edge_it = in_edges.lower_bound({in_node_id, 0UL}); // 0UL不参与比较
    if ((edge_it == in_edges.end()) || ((*edge_it).node_id > in_node_id)) {
      // only one data
      if (edge_it == in_edges.begin()) {
        continue;
      }
      --edge_it;
    }

    // 要给in edge: stream_id:[stream_id<-third_stream_id] 建立新的边
    const EdgeLife new_in_edge{node_id, (*edge_it).peer_node_id};
    auto &in_edge_set = in_stream_edges_[stream_id][third_stream_id];
    const auto old_edge_it = in_edge_set.lower_bound({node_id, 0UL}); // 0UL不参与比较
    // 删除冗余交叉边，缩短node_id与peer_node_id的距离
    if (old_edge_it != in_edge_set.end()) {
      EraseIntersectedEdge(in_edge_set, *old_edge_it, new_in_edge, third_stream_id, stream_id);
    }
    InsertStreamInEdge(new_in_edge, third_stream_id, stream_id);
  }
}

///   Data
///     |----------
///     |          |
///  D stream 0   E stream 1
///  Data不是实际执行节点，产生stream 1->stream 0的依赖会导致错误结果，因此ge local类型不处理
void BlockMemAssigner::GetDiffStreamEdgeLife(const NodePtr &node, const std::set<int64_t> &exclude_merge_streams) {
  auto node_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_desc);
  if (NodeUtils::IsLikeAtomicClean(node) || (node_desc->GetOpKernelLibName() == kEngineNameGeLocal)) {
    return;
  }
  for (const auto &out_anchor : node->GetAllOutAnchors()) {
    GE_CHECK_NOTNULL_JUST_RETURN(out_anchor);
    for (auto const peer_in_anchor : out_anchor->GetPeerAnchorsPtr()) {
      GE_CHECK_NOTNULL_JUST_RETURN(peer_in_anchor);
      const auto peer_node = peer_in_anchor->GetOwnerNodeBarePtr();
      GE_CHECK_NOTNULL_JUST_RETURN(peer_node);
      const auto peer_in_node_desc = peer_node->GetOpDescBarePtr();
      GE_CHECK_NOTNULL_JUST_RETURN(peer_in_node_desc);
      const auto stream_id = GetStreamId(node_desc);
      const auto peer_in_stream_id = GetStreamId(peer_in_node_desc);
      if (stream_id == peer_in_stream_id) {
        continue;
      }

      if (exclude_merge_streams.find(stream_id) != exclude_merge_streams.cend() ||
          exclude_merge_streams.find(peer_in_stream_id) != exclude_merge_streams.cend()) {
        GELOGI("Stream [%ld->%ld] will interrupt memory reuse among streams", stream_id,
               peer_in_stream_id);
        continue;
      }

      const auto node_id = static_cast<size_t>(node_desc->GetId());
      const auto peer_node_id = static_cast<size_t>(peer_in_node_desc->GetId());
      const EdgeLife new_in_edge{peer_node_id, node_id};  // 从peer_node看，由node连接进来的边称为入边
      InsertStreamInEdge(new_in_edge, stream_id, peer_in_stream_id, node_desc->GetNamePtr(),
                         peer_in_node_desc->GetNamePtr());
      AddInStreamEdge(peer_in_node_desc, node_desc);
    }
  }
}

///        a stream:1
///           /   |
///  b stream:0    c stream:1
///           \   |
///             d stream:1
/// b can be reused as stream 1
void BlockMemAssigner::OptimizeStreamIdForMemoryReuse(const NodePtr &node) {
  SetRealStreamIdForDataNode(node.get());
  MemReuseStrategy::OptimizeDiffStream(node.get());
}

void BlockMemAssigner::SetRealStreamIdForDataNode(const Node *const node) {
  auto node_op_desc = node->GetOpDescBarePtr();
  if (node_op_desc == nullptr) {
    return;
  }
  // data use out put node's stream,
  if (OpTypeUtils::IsDataNode(node->GetType()) && (GetStreamId(node->GetOpDescBarePtr()) == kInvalidStreamId)) {
    const auto &out_anchor = node->GetOutDataAnchor(0U);
    if (out_anchor != nullptr) {
      std::set<int64_t> peer_streams;
      for (auto const peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
        if ((peer_in_anchor == nullptr) || (peer_in_anchor->GetOwnerNodeBarePtr() == nullptr)) {
          continue;
        }
        peer_streams.insert(GetStreamId(peer_in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr()));
        if (peer_streams.size() > 1U) {
          break;
        }
      }
      // data输出都是相同stream才做处理
      if (peer_streams.size() == 1U) {
        MemReuseUtils::SetStreamId(node_op_desc, *peer_streams.begin());
      }
    }
  }
}

Status GetNetoutputInNodeStream(const Node *const netoutput, const Node *const parent_node,
                                std::unordered_map<const Node *, std::vector<int64_t>> &parent_nodes_to_stream_ids) {
  auto &parent_node_to_stream_ids = parent_nodes_to_stream_ids[parent_node];
  const auto &netoutput_op_desc = netoutput->GetOpDesc();
  const auto input_size = netoutput->GetAllInDataAnchorsSize();
  for (uint32_t i = 0U; i < input_size; ++i) {
    const auto input_desc = netoutput_op_desc->GetInputDesc(i);
    uint32_t parent_out_index = 0U;
    if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_out_index) ||
        parent_out_index >= parent_node_to_stream_ids.size() ||
        parent_node_to_stream_ids[parent_out_index] == kInvalidStreamId) {
      continue;
    }
    const auto in_data_anchor = netoutput->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_data_anchor);
    GE_ASSERT_NOTNULL(in_data_anchor->GetPeerOutAnchor());
    const auto input_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(input_node);
    const auto input_node_iter = parent_nodes_to_stream_ids.find(input_node);
    int64_t input_node_stream_id = kInvalidStreamId;
    // input_node is not a parent node
    if (input_node_iter == parent_nodes_to_stream_ids.end()) {
      input_node_stream_id = GetStreamId(input_node->GetOpDescBarePtr());
    } else {
      const auto input_node_out_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();
      GE_ASSERT_TRUE(static_cast<size_t>(input_node_out_index) <= input_node_iter->second.size(),
                     "input_node_out_index: %d, input node output size: %zu, input node: %s",
                     input_node_out_index, input_node_iter->second.size(), input_node->GetNamePtr());
      input_node_stream_id = input_node_iter->second[input_node_out_index];
    }
    if ((parent_node_to_stream_ids[parent_out_index] != kParentNodeDefaultStreamId) &&
        (parent_node_to_stream_ids[parent_out_index] != input_node_stream_id)) {
      GELOGI("subgraph node has multi streams, set no reuse. node:%s(%s) output: %u, new stream id: %lld,"
             ", original stream id: %lld, input_node: %s", parent_node->GetNamePtr(), parent_node->GetTypePtr(),
             parent_out_index, input_node_stream_id, parent_node_to_stream_ids[parent_out_index],
             input_node->GetNamePtr());
      input_node_stream_id = kInvalidStreamId; // means no reuse
    }
    parent_node_to_stream_ids[parent_out_index] = input_node_stream_id;
    GELOGI("get stream id from subgraph node. node:%s(%s) output: %u stream id: %lld, input_node: %s",
           parent_node->GetNamePtr(), parent_node->GetTypePtr(), parent_out_index, input_node_stream_id,
           input_node->GetNamePtr());
  }
  return SUCCESS;
}

// 对于父节点，stream应该使用对应真实节点的（子图内netoutput输入节点），如果有多个真实节点并且stream不同，则设置为不可复用
Status BlockMemAssigner::SetRealStreamIdForParentNode(MemAssistInfo &mem_assist_info) {
  const auto compute_graph = mem_assist_info.compute_graph;
  auto &parent_nodes_to_stream_ids = mem_assist_info.parent_nodes_to_stream_ids;
  const auto root_graph = GraphUtils::FindRootGraph(compute_graph);
  GE_ASSERT_NOTNULL(root_graph);
  // 父节点可能有多个输出，每个输出对应子图内netoutput的一个输入节点的输出，vector保存的是这些输入节点的stream id
  std::map<int64_t, const Node *> ids_to_parent_node;
  for (const NodePtr &n : compute_graph->GetAllNodes()) {
    const auto op_desc = n->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->GetSubgraphInstanceNames().empty()) {
      continue;
    }
    parent_nodes_to_stream_ids[n.get()].resize(op_desc->GetOutputsSize(), kParentNodeDefaultStreamId);
    ids_to_parent_node[op_desc->GetId()] = n.get();
  }

  // 先处理topoid最大的，对于子图嵌套场景，保证了先处理最内层的
  for (auto iter = ids_to_parent_node.rbegin(); iter != ids_to_parent_node.rend(); ++iter) {
    const auto parent_node = iter->second;
    for (const auto &subgraph_name : parent_node->GetOpDescBarePtr()->GetSubgraphInstanceNames()) {
      const auto sub_graph = root_graph->GetSubgraph(subgraph_name);
      if (sub_graph == nullptr) {
        continue;
      }
      const auto netoutput = sub_graph->FindFirstNodeMatchType(NETOUTPUT);
      GE_ASSERT_NOTNULL(netoutput);
      GE_ASSERT_NOTNULL(netoutput->GetOpDesc());
      GE_ASSERT_SUCCESS(GetNetoutputInNodeStream(netoutput.get(), parent_node, parent_nodes_to_stream_ids));
    }
  }
  return SUCCESS;
}

// 对于父节点，stream应该使用对应真实节点的（子图内netoutput输入节点），如果有多个真实节点并且stream不同，则设置为不可复用
Status BlockMemAssigner::GetRealStreamIdForParentNode(const NodePtr &node, const uint32_t out_index,
    int64_t &stream_id, bool &is_reuse) const {
  is_reuse = true;
  const auto iter = parent_nodes_to_stream_ids_.find(node.get());
  if ((iter == parent_nodes_to_stream_ids_.end()) ||
      (out_index >= iter->second.size()) ||
      (iter->second.at(out_index) == kParentNodeDefaultStreamId)) {
    return SUCCESS;
  }
  if (iter->second.at(out_index) == kInvalidStreamId) {
    is_reuse = false;
    GELOGI("node %s(%s) out_index: %u has multi streams, set no reuse",
           node->GetNamePtr(), node->GetTypePtr(), out_index);
    return SUCCESS;
  }
  stream_id = iter->second.at(out_index);
  GELOGI("node %s(%s) out_index: %u get stream %lld", node->GetNamePtr(), node->GetTypePtr(), out_index, stream_id);
  return SUCCESS;
}

std::set<int64_t> GetStreamMergeAndOutStreams(const ge::ComputeGraphPtr &graph) {
  std::set<int64_t> merge_and_out_streams;
  for (const NodePtr &node : graph->GetAllNodes()) {
    if (!MemReuseUtils::IsMergeNode(node)) {
      continue;
    }
    if (merge_and_out_streams.insert(GetStreamId(node->GetOpDescBarePtr())).second) {
      GELOGD("Stream %ld not reuse memory with other streams", GetStreamId(node->GetOpDescBarePtr()));
    }
    for (const auto &out_node : node->GetOutAllNodes()) {
      if (merge_and_out_streams.insert(GetStreamId(out_node->GetOpDescBarePtr())).second) {
        GELOGD("Stream %ld not reuse memory with other streams", GetStreamId(out_node->GetOpDescBarePtr()));
      }
    }
  }
  return merge_and_out_streams;
}

/*
 * 不能改为非static的，不能修改成员变量，因为在HybridMemAssigner中，
 * 只有一个binary_assigner对象会调用该函数，其他开启多线程创建的assigner对象并没有调用这个接口
 */
Status BlockMemAssigner::PreparationForAssign(MemAssistInfo &mem_assist_info) {
  for (const NodePtr &n : mem_assist_info.compute_graph->GetAllNodes()) {
    OptimizeStreamIdForMemoryReuse(n);
  }
  // call after PreparationForAssign
  GE_ASSERT_SUCCESS(BlockMemAssigner::SetRealStreamIdForParentNode(mem_assist_info));
  return SUCCESS;
}

Status BlockMemAssigner::GetOutAndWorkSpaceMem(std::vector<int64_t> &all_memory_size) {
  std::vector<int64_t> temp;
  std::map<std::string, std::vector<int64_t>> batch_all_memory_size;
  std::map<std::string, int64_t> batch_total_size;
  std::set<int64_t> exclude_merge_streams = GetStreamMergeAndOutStreams(compute_graph_);
  for (const NodePtr &n : compute_graph_->GetAllNodes()) {
    GetDiffStreamEdgeLife(n, exclude_merge_streams);
    GetContinuousNodeLifeTimeBegin(n.get(), n.get(), 0, 0U);

    auto node_op_desc = n->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(node_op_desc);

    if (CheckIsZeroMemNodeType(node_op_desc->GetTypePtr())) {
      continue;
    }

    std::string batch_label;
    (void)ge::AttrUtils::GetStr(node_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);

    if (NodeUtils::IsLikeAtomicClean(n)) {
      atomic_addr_clean_id_ = node_op_desc->GetId();
    }

    for (auto out_anchor : n->GetAllOutDataAnchorsPtr()) {
      auto output_desc = node_op_desc->MutableOutputDesc(out_anchor->GetIdx());
      if (output_desc == nullptr) {
        continue;
      }
      int64_t size = 0;
      (void)MemReuseUtils::GetTensorSize(*output_desc, size,
                                         MemReuseUtils::IsNeedSplitSize(n, out_anchor->GetIdx()));
      GE_ASSERT_TRUE(size >= 0, "[Check][TensorSize]tensor_size:%ld is invalid, "
                     "maybe it is unknown shape node, Node_name:%s",
                     size, node_op_desc->GetNamePtr());
      batch_all_memory_size[batch_label].emplace_back(size);
      if (batch_total_size.find(batch_label) == batch_total_size.end()) {
        batch_total_size[batch_label] = size;
      } else {
        batch_total_size[batch_label] += size;
      }

      if (!anchor_to_symbol_.empty()) {
        auto iter1 = anchor_to_symbol_.find(NodeIndexIO(n.get(), out_anchor->GetIdx(), kOut).ToString());
        if (iter1 == anchor_to_symbol_.end()) {
          continue;
        }
        const std::string &symbol = iter1->second;
        auto iter2 = symbol_mem_reuse_info_.find(symbol);
        if (iter2 == symbol_mem_reuse_info_.end()) {
          symbol_mem_reuse_info_[symbol].size_ = size;
        } else if (size > static_cast<int64_t>(iter2->second.size_)) {
          iter2->second.size_ = size;
        }
      }
    }
    temp.clear();
    GetNodeWorkSpaceSize(n, temp, batch_total_size[batch_label]);
    batch_all_memory_size[batch_label].insert(batch_all_memory_size[batch_label].cend(), temp.cbegin(), temp.cend());
  }
  HandleInStreamRedundantDependence(in_stream_edges_);
  InsertStreamOutEdge();

  GELOGI("The last atomic_addr_clean node id: %ld", atomic_addr_clean_id_);
  GetMaxBatchAllMemorySize(batch_all_memory_size, batch_total_size, all_memory_size, max_batch_label_);
  InitReuseFlag();
  GE_ASSERT_SUCCESS(continuous_mem_mng_.Init(compute_graph_), "continuous memory manager init failed, graph: %s",
                    compute_graph_->GetName().c_str());
  PrintSymbolMap();
  return SUCCESS;
}

/// @ingroup domi
/// @brief decide memory size based on actual input memory size
/// @param [in] size actual memory size in need
/// @param [in] ranges memory size provided
/// @return size_t memory size to apply
size_t GetBlockSize(size_t size, const std::vector<int64_t> &ranges, bool use_range) {
  // binary block use real size
  if (!use_range) {
    size_t align_size = size;
    MemReuseUtils::AlignMemOffset(align_size);
    return align_size;
  }

  for (int64_t x : ranges) {
    auto x_temp = static_cast<size_t>(x);
    if (size <= x_temp) {
      return x_temp;
    }
  }

  GELOGW("Memory needed size:%zu is beyond the biggest block in memory ranges.", size);
  return size;
}

///          a    b   c
///          |___|___|
///              |
///          d   e   f
///          |___|___|
///              |
///              g
/// e ref input b, g are nopading continuous input, no need to alloc b's memory
bool BlockMemAssigner::IsNoNeedAssignMemory(const NodePtr &n, const NodeIndexIO &out_node_index_io,
                                            const uint32_t index) const {
  // ptr has been checked
  const auto op_desc = n->GetOpDescBarePtr();
  const auto output_tensor_desc = op_desc->MutableOutputDesc(index);
  std::string var_name;
  if (ge::AttrUtils::GetStr(output_tensor_desc, ASSIGN_VAR_NAME, var_name) && !var_name.empty()) {
    GELOGI("Op[%s] output[%u] ref var[%s].", op_desc->GetNamePtr(), index, var_name.c_str());
    return true;
  }
  const auto iter = symbol_mem_reuse_info_.find(out_node_index_io.ToString());
  if (iter != symbol_mem_reuse_info_.end()) {
    return iter->second.no_assign_mem_;
  }
  return false;
}

void BlockMemAssigner::GetRefContinuousInputNodeAndFixedAddrPriorFlag(const std::string &symbol,
                                                                      const std::list<NodeIndexIO> &anchors) {
  uint32_t in_count = 0U;
  uint32_t out_count = 0U;
  NodeIndexIO tail_node(nullptr, 0U, kIn);
  bool is_fixed_addr_prior = false;
  for (const auto &node_index_io : anchors) {
    if (node_index_io.node_ptr_ == nullptr) {
      continue;
    }
    if (node_index_io.io_type_ == kIn) {
      in_count++;
    } else if (node_index_io.io_type_ == kOut) {
      out_count++;
    } else {
      // do nothing
    }
    tail_node.node_ptr_ = node_index_io.node_ptr_;
    tail_node.index_ = node_index_io.index_;
    tail_node.io_type_ = node_index_io.io_type_;

    if (is_fixed_addr_prior) {
      continue;
    }

    (void) ge::AttrUtils::GetBool(node_index_io.node_ptr_->GetOpDesc(), ATTR_NAME_IS_FIXED_ADDR_PRIOR,
                                  is_fixed_addr_prior);
    if (is_fixed_addr_prior) {
      symbol_mem_reuse_info_[symbol].is_fixed_addr_prior_ = true;
      GELOGI("Symbol=%s is fixed addr prior, peer node=%s.", symbol.c_str(), node_index_io.ToString().c_str());
    }
  }

  // one or more ref node, one continuous input node and not continuous input node's first input node
  if ((in_count >= 2U) && (in_count == out_count) && (tail_node.index_ != 0U) && (tail_node.io_type_ == kIn)
      && (tail_node.node_ptr_->GetOpDescBarePtr() != nullptr)) {
    // Get the continuous input type of the node, default is false
    bool is_input_continuous = false;
    // If GetBool fail, is_input_continuous is false.
    (void) ge::AttrUtils::GetBool(tail_node.node_ptr_->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                  is_input_continuous);
    if (is_input_continuous) {
      symbol_mem_reuse_info_[symbol].no_assign_mem_ = true;
      GELOGI("Symbol=%s, ref count:%d, tail node:%s is continuous input.", symbol.c_str(), in_count,
             tail_node.node_ptr_->GetNamePtr());
    }
  }
}

///          a   b   c
///          |   |   |
///          d   e   f
///          |___|___|
///              |
///          g   h   i
///          |___|___|
///              |
///              j
/// h and j are nopading continuous input, g can't reuse with a,b,c
/// because their(d,e,f) memory will be replaced by g's memory (cascade continuous input)
/// so g's real life time begin is min of d,e,f
void BlockMemAssigner::GetContinuousNodeLifeTimeBegin(const Node *const org_node, const Node *const node,
                                                      const int32_t index, uint32_t depth) {
  ++depth;
  GE_IF_BOOL_EXEC((depth > kMaxDepthNum), return);

  bool is_nopading_input_continuous = false;
  const auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_EXEC(node_op_desc, return);
  const auto &org_node_desc = org_node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_EXEC(org_node_desc, return);
  (void) ge::AttrUtils::GetBool(node_op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_nopading_input_continuous);
  if (is_nopading_input_continuous) {
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      const bool invalid_node = ((in_anchor == nullptr) || (in_anchor->GetPeerOutAnchor() == nullptr)
          || (in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr() == nullptr));
      GE_IF_BOOL_EXEC(invalid_node, continue);
      GetContinuousNodeLifeTimeBegin(org_node, in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr(),
                                     in_anchor->GetIdx(), depth);
    }

    if (org_node == node) {
      SetContinuousNodeLifeTimeBegin(node, node, 0U);
    }
  } else {
    // 2 means has continuous input
    GE_IF_BOOL_EXEC((depth < 2U), return);
    auto it = cascade_min_life_time_.find(org_node_desc->GetNamePtr());
    if (it == cascade_min_life_time_.end()) {
      cascade_min_life_time_[org_node_desc->GetNamePtr()] = node_op_desc->GetId();
    } else {
      if (static_cast<size_t>(node_op_desc->GetId()) < it->second) {
        it->second = node_op_desc->GetId();
      }
    }
    // only set first node, continuous first input need alloc memory
    if (index == 0) {
      cascade_min_life_time_[node_op_desc->GetNamePtr()] = node_op_desc->GetId();
    }
    GELOGD("Find node:%s life time begin:%ld by ref node:%s index:%d.", node_op_desc->GetNamePtr(),
           node_op_desc->GetId(), org_node_desc->GetNamePtr(), index);
  }
  return;
}

void BlockMemAssigner::SetContinuousNodeLifeTimeBegin(const Node *const org_node, const Node *const node,
                                                      uint32_t depth) {
  ++depth;
  if (depth > kMaxDepthNum) {
    return;
  }

  const auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_EXEC(node_op_desc, return);
  bool is_nopading_input_continuous = false;
  (void)ge::AttrUtils::GetBool(node_op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_nopading_input_continuous);
  if (is_nopading_input_continuous) {
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      const bool invalid_node = (in_anchor == nullptr) || (in_anchor->GetPeerOutAnchor() == nullptr);
      GE_IF_BOOL_EXEC(invalid_node, continue);
      const auto peer_in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
      GE_CHECK_NOTNULL_EXEC(peer_in_node, continue);
      SetContinuousNodeLifeTimeBegin(org_node, peer_in_node, depth);
    }
  } else {
    // set min life time, only set first node
    auto it = cascade_min_life_time_.find(node_op_desc->GetNamePtr());
    if (it != cascade_min_life_time_.end()) {
      const auto org_node_desc = org_node->GetOpDescBarePtr();
      GE_CHECK_NOTNULL_EXEC(org_node_desc, return);
      const auto it_org = cascade_min_life_time_.find(org_node_desc->GetNamePtr());
      if (it_org != cascade_min_life_time_.cend()) {
        GELOGI("Node:%s set min life time begin from %zu to %zu by ref node:%s.", node->GetNamePtr(),
               it->second, it_org->second, org_node_desc->GetNamePtr());
        it->second = it_org->second;
      }
    }
  }
  return;
}

bool IsSeparateCleanContinuousInputNode(const Node *const node) {
  GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
  bool is_input_continuous = MemLayoutConflictUtil::IsContinuousInput(node);
  if (!is_input_continuous) {
    return false;
  }
  bool need_gentask_atomic = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), "need_gentask_atomic", need_gentask_atomic);
  const bool has_atomic_input = node->GetOpDescBarePtr()->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
  if (has_atomic_input && need_gentask_atomic) {
    return true;
  }
  return false;
}

/*
 * 1. NoPadding连续输入，仅首个输入分配一个block，所有输入使用这一个block
 * 2. 带Padding连续输入，每个输入有自己的block，连续在一起。
 *   （如果ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION为true，或设置了单独清零，等同于NoPadding连续输入了）
 */
bool BlockMemAssigner::IsOutNodeSetContinuousInput(const NodePtr &n, uint32_t out_index,
                                                   InDataAnchor *&continuous_in_anchor, bool &is_reuse_zero_copy,
                                                   std::set<int64_t> &streams) {
  if (out_index >= n->GetAllOutDataAnchorsSize()) {
    return false;
  }
  auto node_desc = n->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_desc);
  std::vector<int64_t> offsets_for_fusion = {};
  bool has_lx_fusion_attr =
    AttrUtils::GetListInt(node_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);

  auto out_anchor = n->GetOutDataAnchor(out_index);
  GE_ASSERT_NOTNULL(out_anchor);
  bool is_out_node_continuous_input = false;

  for (auto const peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
    InDataAnchor *new_peer_in_anchor = nullptr;
    auto is_input_continuous = MemLayoutConflictUtil::IsContinuousInputThroughRefNode(peer_in_anchor, true,
                                                                                      new_peer_in_anchor);
    if (is_input_continuous) {
      has_lx_fusion_attr = true;
    } else {
      is_input_continuous = MemLayoutConflictUtil::IsContinuousInputThroughRefNode(peer_in_anchor, false,
                                                                                   new_peer_in_anchor);
    }
    if (!is_input_continuous) {
      continue;  // peer node does not need continuous memory
    }
    /*
     * 判断输出节点是不是连续内存节点，不能只看直连的输出，而且也要看经过RefNode连接的节点
     * new_peer不一定是n的直连输出，也可能是n经过一个或多个RefNod连接的输出节点
     */
    GE_ASSERT_NOTNULL(new_peer_in_anchor);
    const auto continuous_node = new_peer_in_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(continuous_node);

    // lx_fusion memory only assign first input, broadcast's input some are variable some are not, reassign later
    // In CleanSeparately policy, padding continuous input only allocate index 0 input
    const bool is_separate_clean_continuous_input = IsSeparateCleanContinuousInputNode(continuous_node);
    if (CheckIsZeroMemNodeType(continuous_node->GetTypePtr()) ||
        ((has_lx_fusion_attr || is_separate_clean_continuous_input) && (new_peer_in_anchor->GetIdx() != 0))) {
      GELOGI("Node[%s] output[%u] peer node[%s] type[%s] input[%u].", n->GetNamePtr(), out_index,
             continuous_node->GetNamePtr(), continuous_node->GetTypePtr(), new_peer_in_anchor->GetIdx());
      return false;
    }

    // 到这里continuous_node有两种，一种是noPadding连续输入节点，且n是第0个输入。另一种是带Padding连续输入节点，n不一定是第0个输入
    if (n->GetOwnerComputeGraphBarePtr() == nullptr) {
      continue;
    }
    GELOGI("%s name[%s] output[%u] peer[%s] input[%d] need continuous, input size[%u].",
            n->GetOwnerComputeGraphBarePtr()->GetName().c_str(), n->GetNamePtr(), out_index,
           continuous_node->GetNamePtr(), new_peer_in_anchor->GetIdx(),
           continuous_node->GetAllInDataAnchorsSize());

    // Only set attr one times.
    const auto continuous_op_desc = continuous_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(continuous_op_desc);
    if (node_continuous_input_blocks_[continuous_op_desc->GetId()].size() == 0U) {
      is_reuse_zero_copy = false;
      // lx fusion case assign max size for first block, so reuse as none continuous
      // In CleanSeparately policy, need to calculate the size of the input application memory of index 0 through
      // is_out_node_continuous_input
      if (has_lx_fusion_attr || is_separate_clean_continuous_input) {
        is_op_reuse_mem_ = IsContinuousMemoryReuse(n.get(), out_index, continuous_node, streams);
        is_out_node_continuous_input = is_separate_clean_continuous_input ? true : is_out_node_continuous_input;
        is_separate_clean_continuous_inputs_ = is_separate_clean_continuous_input;
        continue;
      }
      node_continuous_input_counts_[continuous_op_desc->GetId()] = std::make_pair(
          continuous_node->GetTypePtr(), continuous_node->GetAllInDataAnchorsSize());
    }
    continuous_in_anchor = new_peer_in_anchor;
    is_out_node_continuous_input = true;
  }
  return is_out_node_continuous_input;
}

/*
 * 直连或经过RefNode间接连接连续输入节点的，有些情况只给第0个输入分配内存，大小是所有输入的总大小，其他输入不分配内存。
 * 1. NoPadding连续输入只给第0个输入分配内存
 * 2. 带Padding连续输入，一般情况下每个输入都分配内存，以下两个特殊情况下，只给第0个输入分配内存
 * 2.1 输入上带有lx fusion（ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION）属性的
 * 2.2 连续输入需要对输入做单独清零的（有need_gentask_atomic/ATOMIC_ATTR_INPUT_INDEX属性）
 * 3. NoPadding连续输入级联场景，只给最后一个PhonyConcat的第0个输入分配内存
 * 4. 既作为第0个输入，又作为其他输入的，需要分配内存。
 */
Status BlockMemAssigner::GetNoNeedAssignMemoryFlag(const NodePtr &n, uint32_t out_index,
                                                   bool &no_need_assign_memory) const {
  no_need_assign_memory = false;
  auto node_desc = n->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_desc);
  auto out_anchor = n->GetOutDataAnchor(out_index);
  GE_ASSERT_NOTNULL(out_anchor);
  std::vector<int64_t> offsets_for_fusion = {};
  const auto has_lx_fusion_attr =
      AttrUtils::GetListInt(node_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);

  for (auto const peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
    InDataAnchor *new_peer_in_anchor = nullptr;
    const auto nopadding_continuous = MemLayoutConflictUtil::IsContinuousInputThroughRefNode(peer_in_anchor, true,
                                                                                             new_peer_in_anchor);
    bool continuous = false;
    if (!nopadding_continuous) {
      continuous = MemLayoutConflictUtil::IsContinuousInputThroughRefNode(peer_in_anchor, false,
                                                                          new_peer_in_anchor);
    }
    if (!(continuous || nopadding_continuous)) {
      continue;  // peer node does not need continuous memory
    }
    /*
     * 判断输出节点是不是连续内存节点，不能只看直连的输出，而且也要看经过RefNode连接的节点
     * new_peer不一定是n的直连输出，也可能是n经过一个或多个RefNod连接的输出节点
     */
    GE_ASSERT_NOTNULL(new_peer_in_anchor);
    const auto continuous_node = new_peer_in_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(continuous_node);

    if (new_peer_in_anchor->GetIdx() == 0) {
      no_need_assign_memory = false;
      break;
    }
    if (continuous) {
      // lx_fusion memory only assign first input, broadcast's input some are variable some are not, reassign later
      // In CleanSeparately policy, padding continuous input only allocate index 0 input
      const bool is_separate_clean_continuous_input = IsSeparateCleanContinuousInputNode(continuous_node);
      if (!(has_lx_fusion_attr || is_separate_clean_continuous_input)) {
        continue;
      }
    }
    no_need_assign_memory = true;
    GELOGI("%s name[%s] output[%u] peer[%s] input[%d] need continuous, input size[%u], nopadding_continuous[%d], "
        "continuous[%d], has_lx_fusion_attr[%d]",
        n->GetOwnerComputeGraphBarePtr()->GetName().c_str(), n->GetNamePtr(), out_index,
        continuous_node->GetNamePtr(), new_peer_in_anchor->GetIdx(),
        continuous_node->GetAllInDataAnchorsSize(), nopadding_continuous, continuous, has_lx_fusion_attr);
  }
  GELOGI("%s name[%s] output[%u] no_need_assign_memory:%d.",
         n->GetOwnerComputeGraphBarePtr()->GetName().c_str(), n->GetNamePtr(), out_index, no_need_assign_memory);
  return SUCCESS;
}

Status BlockMemAssigner::CalNodeAsContinuousInputMaxLife(const Node *const n, uint32_t out_index,
                                                         const Node *const continuous_node,
                                                         int64_t &first_node_max_life, std::set<int64_t> &streams) {
  // n,peer_node_desc have been checked
  auto node_desc = n->GetOpDescBarePtr();
  auto peer_node_desc = continuous_node->GetOpDescBarePtr();
  life_begin_ = static_cast<size_t>(node_desc->GetId());
  // lx fusion case check all continuous input node, firt input node's life time should be min
  for (const auto &in_anchor : continuous_node->GetAllInDataAnchorsPtr()) {
    GE_CHECK_NOTNULL(in_anchor);
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    if ((in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr)) {
      GELOGE(FAILED, "[Check][OpDesc]Node[%s] output[%u] peer input node desc is null.", n->GetNamePtr(), out_index);
      return FAILED;
    }
    auto peer_out_node_desc = in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr();
    Node *src_node = nullptr;
    int32_t src_out_index = 0;
    GE_ASSERT_SUCCESS(MemReuseUtils::GetSrcNodeThroughRefNode(continuous_node, in_anchor->GetIdx(), src_node,
                                                              src_out_index));
    (void)src_out_index;
    GE_ASSERT_NOTNULL(src_node->GetOpDescBarePtr());
    /*
     *  a(stream 0)
     *  |
     *  b(stream 0)--+
     *  |            |
     *  c(stream 0)  d(stream 1)
     *   \          /
     *    Phonyconcat
     *
     *  如果Phonyconcat的所有输入的流相同，则取id最小的作为life_begin_
     *  如果有的输入与首个输入流不同，例如c是首个输入，而d的流不同，则找stream1 <- stream0的入边，本例子会找到b
     */
    int64_t min_life_time = kMinLifeTime;
    GetDiffStreamMinLifeTime(src_node, GetStreamId(n->GetOpDescBarePtr()), in_stream_edges_, min_life_time);
    if (static_cast<size_t>(min_life_time) < life_begin_) {
      life_begin_ = static_cast<size_t>(min_life_time);
      GELOGI(
          "Node[%s] life[%ld] output[%u] is not continuous input node[%s] life[%ld]'s min life time, "
          "min is life[%zu], src_node[%s], life[%ld], stream_id[%ld]",
          n->GetNamePtr(), node_desc->GetId(), out_index, peer_node_desc->GetNamePtr(), peer_node_desc->GetId(),
          min_life_time, src_node->GetNamePtr(), src_node->GetOpDescBarePtr()->GetId(),
          GetStreamId(src_node->GetOpDescBarePtr()));
    }
    // when node5's first input node2's life time is not max(node6 > node5), set it to max
    int64_t max_node_life_time_by_symbol = 0;
    const int64_t node_max_life =
        GetNodeMaxLife(symbol_to_anchors_, out_stream_edges_, in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr(),
                       in_anchor->GetPeerOutAnchor()->GetIdx(), max_node_life_time_by_symbol, streams,
                       GetStreamId(n->GetOpDescBarePtr()));
    if (node_max_life > first_node_max_life) {
      first_node_max_life = node_max_life;
      GELOGI(
          "Node[%s] life[%ld] output[%u]'s continuous input node[%s] life[%ld]'s is not node[%s] output[%d]'s "
          "max life node",
          n->GetNamePtr(), node_desc->GetId(), out_index, peer_node_desc->GetNamePtr(), peer_node_desc->GetId(),
          peer_out_node_desc->GetNamePtr(), in_anchor->GetPeerOutAnchor()->GetIdx());
    }
  }
  return SUCCESS;
}

/// @ingroup GE
/// @brief Check continuous memory reuseable
/// @return void
bool BlockMemAssigner::IsContinuousMemoryReuse(const Node *const n, uint32_t out_index,
                                               const Node *const continuous_node, std::set<int64_t> &streams) {
  if (!is_op_reuse_mem_) {
    return false;
  }

  int64_t first_node_max_life = 0;
  if (CalNodeAsContinuousInputMaxLife(n, out_index, continuous_node, first_node_max_life, streams) != SUCCESS) {
    return false;
  }
  life_end_ = static_cast<size_t>(first_node_max_life);
  return true;
}

void IsSymbolNodePreReuse(const Node *const node, const bool has_subgraph_data,
                          bool &pre_reuse_flag, bool &post_reuse_flag) {
  static const std::string kFunctionOp = "FunctionOp";
  // node reference subgraph data, data output can not reuse
  bool is_ref = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, is_ref);
  if (has_subgraph_data && is_ref) {
    pre_reuse_flag = false;
  }
  // iteratorGetNext output can not reuse
  if (node->GetType() == kFunctionOp) {
    std::string original_type;
    (void)AttrUtils::GetStr(node->GetOpDescBarePtr(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type);
    if (original_type == ITERATORV2) {
      pre_reuse_flag = false;
      post_reuse_flag = false;
    }
  }
}

bool BlockMemAssigner::GetOutputNodeReuseMemFlagByIndex(const int32_t index) {
  if (output_index_to_reuse_mem_flag_.size() == 0U) {
    return false;
  }

  return ((index >= 0) && (static_cast<size_t>(index) < output_index_to_reuse_mem_flag_.size())) ?
    output_index_to_reuse_mem_flag_[index] : false;
}

bool BlockMemAssigner::GetInputNodeReuseMemFlag(const NodePtr &n) {
  if (input_index_to_reuse_mem_flag_.size() == 0U) {
    return false;
  }

  const auto op_desc = n->GetOpDescBarePtr();
  int32_t index = 0;
  if (!(ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index))) {
    GELOGW("Node[%s] Get index from data attr failed.", op_desc->GetName().c_str());
    return false;
  }

  return ((index >= 0) && (static_cast<size_t>(index) < input_index_to_reuse_mem_flag_.size())) ?
    input_index_to_reuse_mem_flag_[index] : false;
}

/// @ingroup GE
/// @brief Check pre_reuse flag & post_reuse glag for each symbol
/// @return void
void BlockMemAssigner::InitReuseFlag() {
  static const std::set<std::string> kNoPreReuseTypes = {
      ge::DATA_TYPE,  ge::AIPP_DATA_TYPE, ge::ANN_DATA_TYPE,  ge::QUEUE_DATA, ge::PROPOSAL,    ge::CONSTANT,
      ge::CONSTANTOP, ge::GETNEXT,        ge::DROPOUTGENMASK, ge::REFDATA, "AdpGetNext",   "DynamicGetNext"};
  static const std::set<std::string> kNoPostReuseTypes = {ge::DATA_TYPE,       ge::AIPP_DATA_TYPE, ge::QUEUE_DATA,
                                                          ge::ENTER,           ge::REFENTER,       ge::NEXTITERATION,
                                                          ge::REFNEXTITERATION, ge::REFDATA, ge::GETNEXT, "AdpGetNext",
                                                          "DynamicGetNext", ge::DROPOUTGENMASK};
  InitDiffStreamSameOutTable();
  for (const auto &pair : symbol_to_anchors_) {
    const std::string &symbol = pair.first;
    bool pre_reuse_flag = true;
    bool post_reuse_flag = true;
    // default memory type
    int64_t mem_type = RT_MEMORY_HBM;
    GetSymbolMemType(pair.second, mem_type);
    AddSymbolMemType(symbol, mem_type);
    GetRefContinuousInputNodeAndFixedAddrPriorFlag(symbol, pair.second);
    GELOGD("The memory type of symbol[%s] is [%ld].", symbol.c_str(), mem_type);
    if (mem_type == RT_MEMORY_P2P_DDR) {
      UpdateOpTensorMemType(pair.second, mem_type);
    }
    bool has_subgraph_data = false;
    bool diff_stream_prior = false;
    for (const auto &node_index_io : pair.second) {
      GE_CHECK_NOTNULL_EXEC(node_index_io.node_ptr_, continue);
      GE_CHECK_NOTNULL_EXEC(node_index_io.node_ptr_->GetOpDescBarePtr(), continue);
      bool in_flag = IsDirectInputNode(node_index_io.node_ptr_, compute_graph_);
      bool in_reuse_mem_flag = in_flag ? GetInputNodeReuseMemFlag(node_index_io.node_) : false;

      // unknown graph subgraph data can not reuse because zero copy.
      if (!in_flag && (node_index_io.node_ptr_->GetType() == DATA) &&
          node_index_io.node_ptr_->GetOpDescBarePtr()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
        GELOGD("Node: %s is subgraph data, continue", node_index_io.node_ptr_->GetNamePtr());
        has_subgraph_data = true;
        continue;
      }
      if (node_index_io.io_type_ == kIn) {
        continue;
      }
      IsSymbolNodePreReuse(node_index_io.node_ptr_, has_subgraph_data, pre_reuse_flag, post_reuse_flag);
      diff_stream_prior = MemReuseStrategy::GetDiffStreamPrior(node_index_io.node_ptr_);
      OutDataAnchorPtr out_anchor = node_index_io.node_ptr_->GetOutDataAnchor(node_index_io.index_);
      if (out_anchor == nullptr) {
        continue;
      }

      bool out_flg = false;
      bool out_reuse_mem_flag = false;
      for (const auto in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
        if (IsDirectOutputNode(in_anchor->GetOwnerNodeBarePtr(), compute_graph_)) {
          out_flg = true;
          out_reuse_mem_flag = GetOutputNodeReuseMemFlagByIndex(in_anchor->GetIdx());
          break;
        }
      }

      const auto type = out_anchor->GetOwnerNodeBarePtr()->GetTypePtr();
      if (in_reuse_mem_flag || out_reuse_mem_flag) {
        // model data    no pre reuse, post reuse
        // model net out pre reuse,    no post reuse
        pre_reuse_flag = pre_reuse_flag && (!in_flag) && (out_flg || NotMatchNoReuseType(kNoPreReuseTypes, type));
        post_reuse_flag = post_reuse_flag && (!out_flg) && (in_flag || NotMatchNoReuseType(kNoPostReuseTypes, type));
      } else {
        // model data    no pre reuse, no post reuse
        // model net out no pre reuse, no post reuse
        pre_reuse_flag = pre_reuse_flag && (!in_flag) && (!out_flg) && NotMatchNoReuseType(kNoPreReuseTypes, type);
        post_reuse_flag = post_reuse_flag && (!in_flag) && (!out_flg) && NotMatchNoReuseType(kNoPostReuseTypes, type);
      }
      if (!pre_reuse_flag && !post_reuse_flag) {
        break;
      }
    }
    MemoryReuseInfo &memory_reuse_info = symbol_mem_reuse_info_[symbol];
    memory_reuse_info.pre_reuse_flag_ = pre_reuse_flag;
    memory_reuse_info.post_reuse_flag_ = post_reuse_flag;
    memory_reuse_info.diff_stream_prior_ = diff_stream_prior;
    // 全局不复用的内存后面会单独累加，这里只处理pre_reuse_flag_为false的
    if ((!memory_reuse_info.pre_reuse_flag_) && memory_reuse_info.post_reuse_flag_) {
      auto &memory_stat = memory_stat_[mem_type];
      auto align_size = memory_reuse_info.size_;
      MemReuseUtils::AlignMemOffset(align_size);
      memory_stat.theory_memory_size_ += align_size;
    }
  }
}

/*
 * 输出使用同一个memory_block的当成一组，在进行复用时，要么同时复用某个节点，要么同时不复用某个节点，共同进退。这里只需要处理不同流的，
 * StreamMerge/Merge输入必然不同流，NoPaddingContinuousInput 会计算同流或不同流的所有输入最小life_begin_。
 */
void BlockMemAssigner::InitDiffStreamSameOutTable() {
  for (auto &node : compute_graph_->GetAllNodesPtr()) {
    if (!MemReuseUtils::IsMergeNode(node)) {
      continue;
    }
    std::list<OutDataAnchor *> same_out_anchor;
    std::set<int64_t> stream_id_set;
    for (auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      if ((in_anchor->GetPeerOutAnchor() != nullptr)
          && (in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr() != nullptr)) {
        same_out_anchor.push_back(in_anchor->GetPeerOutAnchor().get());
        stream_id_set.insert(GetStreamId(in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr()));
      }
    }
    if (stream_id_set.size() <= 1U) {
      continue;
    }
    same_out_group_holder_.push_back(std::move(same_out_anchor));
    for (auto out_anchor : same_out_group_holder_.back()) {
      same_out_group_[out_anchor] = &same_out_group_holder_.back();
    }
    if (IsLogEnable(GE, DLOG_INFO)) {
      std::stringstream ss;
      for (auto out_anchor : same_out_group_holder_.back()) {
        ss << "topoid_" << out_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr()->GetId() << "_out_"
           << out_anchor->GetIdx() << ", ";
      }
      GELOGI("diff stream same out: %s", ss.str().c_str());
    }
  }
}

// 是否存在和n的index输出使用同一块的节点，且不同流的
bool BlockMemAssigner::HasSameOutAnchorWithDiffStream(const Node *n, const uint32_t index) const {
  GE_ASSERT_NOTNULL(n);
  const auto out_data_anchor = n->GetOutDataAnchor(index);
  GE_ASSERT_NOTNULL(out_data_anchor);
  return same_out_group_.find(out_data_anchor.get()) != same_out_group_.end();
}

void BlockMemAssigner::AddSymbolMemType(const std::string &symbol, int64_t memory_type) {
  // Only the memory with special requirements is processed. The HBM uses the default processing mode.
  if ((memory_type == RT_MEMORY_P2P_DDR) || (memory_type == RT_MEMORY_HOST) || (memory_type == RT_MEMORY_HOST_SVM)) {
    symbol_mem_reuse_info_[symbol].mem_type_ = memory_type;
  } else {
    symbol_mem_reuse_info_[symbol].mem_type_ = RT_MEMORY_HBM;
  }
}

/// @ingroup GE
/// @brief get pre_reuse flag
/// @param [in] node
/// @param [in] out_index
/// @return bool
bool BlockMemAssigner::IsPreReuse(const NodeIndexIO &cur_node_index_io, std::string &symbol) const {
  auto iter1 = anchor_to_symbol_.find(cur_node_index_io.ToString());
  if (iter1 == anchor_to_symbol_.end()) {
    return false;
  }

  symbol = iter1->second;
  auto iter2 = symbol_mem_reuse_info_.find(symbol);
  if (iter2 == symbol_mem_reuse_info_.end()) {
    return false;
  }
  return iter2->second.pre_reuse_flag_;
}

bool BlockMemAssigner::IsPostReuse(const std::string &symbol, bool &diff_stream_prior) const {
  auto iter = symbol_mem_reuse_info_.find(symbol);
  if (iter == symbol_mem_reuse_info_.end()) {
    return true;
  }

  diff_stream_prior = iter->second.diff_stream_prior_;
  return iter->second.post_reuse_flag_;
}

/// @ingroup GE
/// @brief get post_reuse flag
/// @param [in] mem_block
/// @return bool
bool BlockMemAssigner::IsPostReuse(const ge::MemoryBlock *const mem_block) const {
  if (mem_block == nullptr) {
    return false;
  }
  return mem_block->post_reuse_flag_;
}
/// @ingroup GE
/// @brief check if symbol of cur node_index_io has block
/// @param [in] node_index_io
/// @param [out] symbol
/// @return bool
bool BlockMemAssigner::IsSymbolExist(const NodeIndexIO &node_index_io, std::string &symbol, MemoryBlock *&block) const {
  block = nullptr;
  const auto node_io = node_index_io.ToString();
  auto iter = anchor_to_symbol_.find(node_io);
  if (iter == anchor_to_symbol_.end()) {
    return false;
  }

  symbol = iter->second;
  auto it_block = symbol_blocks_.find(iter->second);
  auto symbol_exist = (it_block != symbol_blocks_.end());
  if (symbol_exist) {
    GELOGD("Node io:%s symbol:%s block:%s", node_io.c_str(), symbol.c_str(), GetName(*(it_block->second)).c_str());
    block = it_block->second;
  }
  return symbol_exist;
}

/// @ingroup GE
/// @brief check if symbol of cur node_index_io has output description block
/// @param [in] node_index_io
/// @param [out] symbol
/// @return bool
bool BlockMemAssigner::IsSymbolDescBlockExist(const NodeIndexIO &node_index_io, std::string &symbol,
                                              MemoryBlock *&block) const {
  const auto node_io = node_index_io.ToString();
  auto iter = anchor_to_symbol_.find(node_io);
  if (iter == anchor_to_symbol_.end()) {
    return false;
  }

  symbol = iter->second;
  GELOGD("Node io:%s symbol:%s", node_io.c_str(), symbol.c_str());
  const auto it_block = symbol_desc_blocks_.find(iter->second);
  if (it_block == symbol_desc_blocks_.cend()) {
    return false;
  }
  block = it_block->second;
  return true;
}

/// @ingroup GE
/// @brief Print symbol
/// @return void
void BlockMemAssigner::PrintSymbolMap() {
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }

  for (const auto &pair : symbol_to_anchors_) {
    GELOGD("symbol=%s, max_size=%zu, pre_reuse=%s, post_reuse=%s", pair.first.c_str(),
           symbol_mem_reuse_info_[pair.first].size_,
           symbol_mem_reuse_info_[pair.first].pre_reuse_flag_ ? "true" : "false",
           symbol_mem_reuse_info_[pair.first].post_reuse_flag_ ? "true" : "false");
    for (const auto &node_index_io : pair.second) {
      GELOGD("anchor:%s id:%ld", node_index_io.ToString().c_str(),
             ((node_index_io.node_ptr_ != nullptr) && (node_index_io.node_ptr_->GetOpDescBarePtr() != nullptr))
             ? node_index_io.node_ptr_->GetOpDescBarePtr()->GetId() : 0);
    }
  }
}

void BlockMemAssigner::GetSymbolMemType(const std::list<NodeIndexIO> &node_index_io_list, int64_t &memory_type) {
  memory_type = RT_MEMORY_HBM;
  std::vector<int64_t> memory_types;
  for (auto &node_index_io : node_index_io_list) {
    auto op_desc = node_index_io.node_ptr_->GetOpDescBarePtr();
    GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
    if (node_index_io.io_type_ == kIn) {
      std::vector<int64_t> input_memory_types;
      (void) ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_types);
      if (!input_memory_types.empty() && node_index_io.index_ < input_memory_types.size()) {
        int64_t input_memory_type = input_memory_types[node_index_io.index_];
        GELOGD("Node[%s]: the memory type of input index [%u] is [%ld]].", op_desc->GetNamePtr(),
               node_index_io.index_, input_memory_type);
        memory_types.emplace_back(input_memory_type);
      }
    }
    if (node_index_io.io_type_ == kOut) {
      std::vector<int64_t> output_memory_types;
      (void) ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types);
      if (!output_memory_types.empty() && node_index_io.index_ < output_memory_types.size()) {
        int64_t output_memory_type = output_memory_types[node_index_io.index_];
        GELOGD("Node[%s]: the memory type of output index [%u] is [%ld].", op_desc->GetNamePtr(),
               node_index_io.index_, output_memory_type);
        memory_types.emplace_back(output_memory_type);
      }
    }
  }

  // memory priority
  for (auto node_memory_type : memory_types) {
    if (node_memory_type > memory_type) {
      memory_type = node_memory_type;
    }
  }
}

void BlockMemAssigner::UpdateOpTensorMemType(const std::list<NodeIndexIO> &node_index_io_list, int64_t memory_type) {
  for (const auto &node_index_io : node_index_io_list) {
    auto op_desc = node_index_io.node_ptr_->GetOpDescBarePtr();
    GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
    if (node_index_io.io_type_ == kIn) {
      auto input_desc = op_desc->MutableInputDesc(node_index_io.index_);
      int_attr_.emplace_back(input_desc.get(), op_desc, node_index_io.index_, ATTR_NAME_TENSOR_MEM_TYPE, memory_type);
    }

    if (node_index_io.io_type_ == kOut) {
      auto output_desc = op_desc->MutableOutputDesc(node_index_io.index_);
      int_attr_.emplace_back(output_desc.get(), op_desc, node_index_io.index_, ATTR_NAME_TENSOR_MEM_TYPE, memory_type);
    }
  }
}

bool BlockMemAssigner::IsNodeAndPeerNodeTaskSupportZeroCopy(const ge::NodePtr &node, uint32_t output_index) const {
  GELOGD("Check node %s and peer node of output %u task zero copy supported", node->GetNamePtr(), output_index);
  if (is_feature_map_refreshable_) {
    return true;
  }

  if (!IsNodeSupportZeroCopy(node.get())) {
    return false;
  }

  const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(output_index));
  if (out_anchor == nullptr) {
    return false;
  }
  const auto peer_anchors = out_anchor->GetPeerInDataAnchorsPtr();
  const bool support = std::all_of(peer_anchors.begin(), peer_anchors.end(), [](const ge::InDataAnchor *anchor) {
    return ((anchor != nullptr) && (anchor->GetOwnerNodeBarePtr() != nullptr) &&
            IsNodeSupportZeroCopy(anchor->GetOwnerNodeBarePtr()));
  });

  GELOGD("Task of node %s and peer node of output %u %s zero copy", node->GetNamePtr(), output_index,
         (support ? "support" : "not support"));
  return support;
}

bool BlockMemAssigner::IsZeroCopyBlock(const NodePtr &node, uint32_t output_index, bool continuous,
                                       size_t output_size) const {
  std::string op_type(node->GetTypePtr());
  if (NodeUtils::IsDynamicShape(node)) {
    if (compute_graph_.get() != node->GetOwnerComputeGraphBarePtr()) {
      return false;
    }
    if (OpTypeUtils::IsDataNode(op_type)) {
      return (!continuous);
    }
    if (is_static_model_addr_fixed_ && (node->GetOpDesc()->GetOpKernelLibName() == ge::kEngineNameHccl)) {
      return false;
    }
    return (GetOutputFlowToNetoutputNum(node, output_index, compute_graph_, symbol_to_anchors_, anchor_to_symbol_) >
            0U);
  }

  if (is_io_alloc_by_ge_in_run_graph_ && (output_size > input_fusion_size_)) {
    return false;
  }

  if (continuous) {  // Never zero copy for data flow to require-continuous-input node
    GELOGD("Node %s output %u can not zero copy as require continuous output", node->GetNamePtr(), output_index);
    return false;
  }

  if (op_type == NETOUTPUT) {
    const auto owner = node->GetOwnerComputeGraphBarePtr();
    bool ret = (owner != nullptr) && (owner->GetParentGraph() == nullptr);
    GELOGD("Node %s output %u result %d", node->GetNamePtr(), output_index, ret);
    return ret;
  }

  if (OpTypeUtils::IsDataNode(op_type)) {
    bool is_multi_batch_shape_data = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), "_is_multi_batch_shape_data", is_multi_batch_shape_data);
    if (is_multi_batch_shape_data) {
      GELOGD("Multi batch shape data node[%s] output memory no need zero copy.",
          node->GetName().c_str());
      return false;
    }
    if (node->GetOpDescBarePtr()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {  // Never zero copy for subgrapgh data
      return false;
    }
    // Data flow to unsupported zero copy task type eg. memcpy, can never zero copied
    return IsNodeAndPeerNodeTaskSupportZeroCopy(node, output_index);
  }

  // Only node output that flow to sure one output maybe zero copied
  if (GetOutputFlowToNetoutputNum(node, output_index, compute_graph_, symbol_to_anchors_, anchor_to_symbol_) ==
      1U) {  // 1U means output to only one netoutput
    // Output from unsupported task type eg. memcpy, can never zero copied
    return IsNodeAndPeerNodeTaskSupportZeroCopy(node, output_index);
  }

  return false;
}

namespace {
void MarkZeroCopyBlockAttr(std::vector<TAttr<bool>> &bool_attr, const OpDesc *const op_desc, bool is_zero_copy,
                           bool mem_type, uint32_t out_index) {
  if (is_zero_copy && (mem_type == kOutput)) {
    auto output_desc = op_desc->MutableOutputDesc(out_index);
    if (output_desc != nullptr) {
      bool_attr.emplace_back(output_desc.get(), op_desc, out_index, ATTR_IS_ZERO_COPY_BLOCK, true);
    } else {
      GELOGE(PARAM_INVALID, "Node %s output %u is zero copy block but not marked as output desc is nullptr",
             op_desc->GetNamePtr(), out_index);
    }
  }
}
}  // namespace

void BlockMemAssigner::AddMemoryStat(uint64_t memory_type, size_t real_size, bool is_reuse_memory) {
  auto &memory_stat = memory_stat_[memory_type];
  size_t align_size = real_size;
  MemReuseUtils::AlignMemOffset(align_size);
  memory_stat.total_memory_size_ += align_size;
  if (is_reuse_memory) {
    memory_stat.theory_memory_size_ += align_size;
  } else {
    memory_stat.theory_no_reuse_memory_size_ += align_size;
  }

  if (memory_stat.theory_memory_size_ > memory_stat.theory_min_memory_size_) {
    memory_stat.theory_min_memory_size_ = memory_stat.theory_memory_size_;
  }
}

MemoryBlock *BlockMemAssigner::ApplyMemory(const NodePtr &n, const std::vector<bool> &workspace_reuse_flag,
                                           const ApplyMemoryParam &param) {
  auto node_op_desc = n->GetOpDescBarePtr();
  std::string batch_label;
  (void)ge::AttrUtils::GetStr(node_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
  MemoryBlock *reusable_block = nullptr;
  auto stream_id = GetStreamId(node_op_desc);
  bool is_reuse_memory = false;
  GetRealStreamIdForParentNode(n, param.out_index, stream_id, is_reuse_memory);
  bool no_reuse = false;
  std::string symbol;
  // model data output can't reuse other, but it can be reused
  const bool pre_reuse = (param.mem_type == kOutput) ?
    IsPreReuse(NodeIndexIO(n.get(), param.out_index, kOut), symbol) : true;
  const auto it_life_time_begin = cascade_min_life_time_.find(node_op_desc->GetNamePtr());
  (void)ge::AttrUtils::GetBool(node_op_desc, kOpNoReuseMem, no_reuse);
  if ((!no_reuse) && (param.mem_type == kWorkspace)) {
    no_reuse = ((workspace_reuse_flag.size() > param.out_index) && !workspace_reuse_flag[param.out_index]);
  }
  const bool mod_life_begin = ((param.mem_type != kWorkspace) && (it_life_time_begin != cascade_min_life_time_.end())
      && (it_life_time_begin->second < life_begin_)) || no_reuse || (!pre_reuse);
  if (mod_life_begin) {
    // no pre reuse set life time begin to 1
    life_begin_ = ((!pre_reuse) || no_reuse) ? kMinLifeTime : it_life_time_begin->second;
    GELOGD("Node %s output %u life_begin_ change to %zu", n->GetNamePtr(), param.out_index, life_begin_);
  }
  bool diff_stream_prior = false;
  bool post_reuse_flag = IsPostReuse(symbol, diff_stream_prior);
  if ((param.mem_type == kOutput) && (!post_reuse_flag)) {
    life_end_ = kMaxLifeTime;
  }
  const auto has_diff_stream_same_out = (param.mem_type == kOutput)
                                        && HasSameOutAnchorWithDiffStream(n.get(), param.out_index);
  is_reuse_memory = is_reuse_memory && is_ge_reuse_mem_ && (param.mem_type != kOutputDesc)
      && !node_op_desc->HasAttr(kL2FusionDynamicConvergeOp) && !no_reuse && param.is_op_reuse_mem;
  auto &reusable_blocks = reusable_blocks_[param.memory_type][stream_id];
  // continuous memory reuse in level2 reuse
  bool do_reuse = is_reuse_memory && pre_reuse && !param.continuous && (!param.is_zero_copy)
                  && !has_diff_stream_same_out;
  const NodeTypeIndex node_type_index{n.get(), param.mem_type, param.out_index, false, life_begin_, stream_id};
  if (do_reuse) {
    if (reuse_strategy_.reuse_first_release_) {
      reusable_block = GetFirstReleaseBlock(param.block_size, batch_label, reusable_blocks, node_type_index);
    } else {
      reusable_block = GetLastReleaseBlock(param.block_size, batch_label, reusable_blocks, node_type_index);
    }
  }

  auto block = reusable_block;
  if (block == nullptr) {
    block = new(std::nothrow) MemoryBlock(reuse_strategy_, param.block_size, stream_id, is_reuse_memory,
                                          param.memory_type);
    GE_CHECK_NOTNULL_EXEC(block, return nullptr);
    memory_blocks_.emplace_back(block);
    // cause memory_blocks_ may reduce when swap after,
    // create blocks_store_ to assure blocks deleted finally
    blocks_store_.emplace_back(block);
    GELOGD("Node %s create new block:%s", n->GetNamePtr(), GetName(*block).c_str());
  } else {
    GELOGD("Node %s reuse block:%s", n->GetNamePtr(), GetName(*block).c_str());
  }

  if (param.mem_type == kOutput) {
    block->AddSymbol(symbol);
    block->post_reuse_flag_ = block->post_reuse_flag_ && post_reuse_flag;
    block->diff_stream_prior_ = diff_stream_prior;
  }
  block->AddNodeTypeIndex(node_type_index, param.real_size, param.no_align_size, stream_id);
  if (has_diff_stream_same_out) {
    block->same_stream_ = false;
  }
  // model net output can reuse other, but it can't be reused
  const bool post_reuse = IsPostReuse(block);
  if (param.continuous) {
    block->SetContinuousBlock();
  }
  block->batch_label_ = batch_label;
  block->reuse_mem_ = block->reuse_mem_ && (post_reuse || pre_reuse);
  if (life_end_ != 0U) {
    block->SetLifeTimeEnd((life_end_ == kMaxLifeTime) ? kDefaultLifeTime : life_end_, stream_id);
    block->SetSymbolLifeEnd(life_end_);
  }
  const bool cal_theory_size = (batch_label.empty() || (batch_label == max_batch_label_))
      && (node_op_desc->GetType() != ge::PARTITIONEDCALL) && ((!block->reuse_mem_) || pre_reuse);
  if (cal_theory_size) {
    AddMemoryStat(param.memory_type, param.real_size, block->reuse_mem_);
  }
  return block;
}

bool BlockMemAssigner::IsNodeOutputUseSameMemWithNetOutput(const ge::NodePtr &node, uint32_t out_index) const {
  const auto cur_node_index_io = NodeIndexIO(node, out_index, kOut);
  const auto &symbol_iter = anchor_to_symbol_.find(cur_node_index_io.ToString());
  if (symbol_iter == anchor_to_symbol_.cend()) {
    return false;
  }
  const auto &anchors_iter = symbol_to_anchors_.find(symbol_iter->second);
  if (anchors_iter == symbol_to_anchors_.cend()) {
    return false;
  }
  for (const auto &anchor : anchors_iter->second) {
    if (anchor.node_->GetType() == NETOUTPUT) {
      return true;
    }
  }
  return false;
}

MemoryBlock *BlockMemAssigner::GetFirstReleaseBlock(const size_t block_size, const std::string &batch_label,
                                                    std::vector<MemoryBlock *> &reusable_blocks,
                                                    const NodeTypeIndex &node_type_index) const {
  for (auto it = reusable_blocks.begin(); it != reusable_blocks.end(); ++it) {
    MemoryBlock *reusable_block = *it;
    if ((reusable_block == nullptr) ||
        (!ReuseBlock(*reusable_block, block_size, life_begin_, batch_label, node_type_index))) {
      continue;
    }
    reusable_blocks.erase(it);
    return reusable_block;
  }
  return nullptr;
}

MemoryBlock *BlockMemAssigner::GetLastReleaseBlock(const size_t block_size, const std::string &batch_label,
                                                   std::vector<MemoryBlock *> &reusable_blocks,
                                                   const NodeTypeIndex &node_type_index) const {
  for (auto it = reusable_blocks.rbegin(); it != reusable_blocks.rend(); ++it) {
    MemoryBlock *reusable_block = *it;
    if ((reusable_block == nullptr) ||
        (!ReuseBlock(*reusable_block, block_size, life_begin_, batch_label, node_type_index))) {
      continue;
    }
    reusable_blocks.erase((++it).base());
    return reusable_block;
  }
  return nullptr;
}

bool IsOutputIndexRef(const OpDesc *const op_desc, uint32_t index) {
  auto output_tensor = op_desc->GetOutputDescPtr(index);
  if (output_tensor == nullptr) {
    return false;
  }
  bool dst_reuse_input = false;
  (void)ge::TensorUtils::GetReuseInput(*output_tensor, dst_reuse_input);
  if (dst_reuse_input) {
    return true;
  }

  bool is_ref = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (is_ref) {
    std::string output_name = op_desc->GetOutputNameByIndex(index);
    for (const auto &input_name : op_desc->GetAllInputNames()) {
      if (output_name == input_name) {
        return true;;
      }
    }
  }
  return false;
}

bool IsSubgraphDataRefConstInput(const NodePtr &node) {
  std::string op_type;
  const auto &in_node = ge::NodeUtils::GetParentInput(node);
  return ge::NodeUtils::GetConstOpType(in_node, op_type) ||
         ((in_node != nullptr) && ge::OpTypeUtils::IsVariableNode(in_node->GetType()));
}

void BlockMemAssigner::ContinuousOutRefCheck(bool &is_all_output_ref, bool &is_output_has_ref, const NodePtr &n) {
  const auto node_op_desc = n->GetOpDescBarePtr();
  for (uint32_t index = 0U; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    if (!IsOutputIndexRef(node_op_desc, index)) {
      is_all_output_ref = false;
      break;
    } else {
      zero_memory_list_.emplace_back(n.get(), kOutput, index);
      is_output_has_ref = true;
    }
  }
}

Status BlockMemAssigner::ApplyContinuousMemory(const NodePtr &n, const std::vector<int64_t> &ranges,
                                               const bool is_op_reuse_mem) {
  auto node_op_desc = n->GetOpDescBarePtr();
  GE_CHECK_NOTNULL(node_op_desc);
  life_begin_ = node_op_desc->GetId();

  // continuous output support ref only when all output ref input
  bool is_all_output_ref = true;
  bool is_output_has_ref = false;

  ContinuousOutRefCheck(is_all_output_ref, is_output_has_ref, n);

  if (is_all_output_ref) {
    GELOGI("continuous output node ref all input, skip continuous alloc, node_name:%s", n->GetNamePtr());
    return SUCCESS;
  }

  if (!is_all_output_ref && is_output_has_ref) {
    REPORT_INNER_ERR_MSG("E19999", "continuous output node ref part input, not support now. node_name:%s",
                       n->GetNamePtr());
    GELOGE(INTERNAL_ERROR, "[Check][OutRefStatus]continuous output node ref part input, not support, node_name:%s",
           n->GetNamePtr());
    return INTERNAL_ERROR;
  }
  MemoryBlock *block = nullptr;
  size_t total_size = 0U;
  uint64_t memory_type = RT_MEMORY_HBM;
  int64_t max_life_time = 0;
  std::string symbol;
  std::set<int64_t> streams;
  GetContinuousOutputMaxLife(n, symbol_to_anchors_, out_stream_edges_, max_life_time, streams);
  for (uint32_t index = 0U; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    auto output_op_desc = node_op_desc->MutableOutputDesc(index);
    if (CheckIsZeroMemNodeType(n->GetTypePtr()) || CheckIsZeroMemNodeOutputIndex(n, index)) {
      zero_memory_list_.emplace_back(n.get(), kOutput, index);
      continue;
    }

    int64_t size = 0;
    if (MemReuseUtils::GetTensorSize(*output_op_desc, size, MemReuseUtils::IsNeedSplitSize(n, index)) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "get tensor_size failed, node_name:%s, output_index:%u",
                        n->GetNamePtr(), index);
      GELOGE(INTERNAL_ERROR, "[Get][TensorSize]node_name:%s, output_index:%u", n->GetNamePtr(), index);
      return INTERNAL_ERROR;
    }
    size_t align_size = static_cast<size_t>(size);
    MemReuseUtils::AlignMemOffset(align_size);
    total_size += align_size;

    // only apply total size in first output
    if (index != 0U) {
      zero_memory_list_.emplace_back(n.get(), kOutput, index);
      continue;
    }
    NodeIndexIO node_index_io(n.get(), index, kOut);
    auto iter = anchor_to_symbol_.find(node_index_io.ToString());
    if (iter != anchor_to_symbol_.end()) {
      symbol = iter->second;
      std::map<std::string, MemoryReuseInfo>::const_iterator it = symbol_mem_reuse_info_.find(symbol);
      if (it != symbol_mem_reuse_info_.cend()) {
        memory_type = it->second.mem_type_;
        GELOGD("Continuous out memory symbol is [%s], memory type is [%ld]", symbol.c_str(), memory_type);
      }
    }
  }

  if (total_size == 0U) {
    return SUCCESS;
  }

  auto block_size = GetBlockSize(total_size, ranges, reuse_strategy_.use_range_);
  std::vector<bool> workspace_reuse_flag;
  ApplyMemoryParam param = {block_size, total_size, total_size, kOutput, 0U, is_op_reuse_mem, false, memory_type,
                            false};
  block = ApplyMemory(n, workspace_reuse_flag, param);
  if (block != nullptr) {
    // hccl task need align header and tail
    block->SetFirstContinuousBlock();
    block->SetLastContinuousBlock();
    block->need_same_offset_in_batch_ = SizeIndependentOfBatch(n->GetTypePtr());
    bool is_reuse_zero_copy = true;
    NodeIndexIO node_index_io(n.get(), 0, kOut);
    int32_t ref_count = GetAllRefCount(node_index_io, is_reuse_zero_copy);
    block->ref_count_ += ref_count;
    block->is_reuse_zero_copy_ = (block->is_reuse_zero_copy_) && (is_reuse_zero_copy);
    max_life_time = (max_life_time == kMaxLifeTime) ? kDefaultLifeTime : max_life_time;
    block->SetLifeTimeEnd(max_life_time, GetStreamId(node_op_desc));
    block->SetOutStreamCount(streams.size());
    GELOGI("Node[%s] continuous out memory size[%zu] block size[%zu] out stream count:%zu ref_count:%d",
        node_op_desc->GetNamePtr(), total_size, block_size, streams.size(), block->ref_count_);
    if (!symbol.empty()) {
      symbol_blocks_[symbol] = block;
      auto iter = symbol_mem_reuse_info_.find(symbol);
      if (iter != symbol_mem_reuse_info_.end()) {
        iter->second.size_ = total_size;
        block->is_fixed_addr_prior_ = (block->is_fixed_addr_prior_ || iter->second.is_fixed_addr_prior_);
      }

      GELOGD("Node io:%s add symbol:%s block:%s, fixed addr prior:%d",
             NodeIndexIO(n.get(), 0, kOut).ToString().c_str(), symbol.c_str(),
             GetName(*block).c_str(), block->is_fixed_addr_prior_);
    }
  } else {
    REPORT_INNER_ERR_MSG("E19999", "apply continuousMemory failed, node_name:%s, total_size:%ld",
                      n->GetNamePtr(), total_size);
    GELOGE(INTERNAL_ERROR, "[Apply][ContinuousMemory]node_name:%s, total_size:%ld", n->GetNamePtr(), total_size);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

/*
 * 根据continuous_mem_mng_中对节点输出的排布分配内存，要求IsFound必须返回true才调用该接口
 */
Status BlockMemAssigner::ApplyContinuousMemWithMng(const NodePtr &n, int32_t idx, const std::vector<int64_t> &ranges) {
  auto op_desc = n->GetOpDescBarePtr();
  GE_CHECK_NOTNULL(op_desc);
  life_begin_ = op_desc->GetId();

  GE_ASSERT_TRUE(continuous_mem_mng_.IsFound(n.get(), idx));
  if (!continuous_mem_mng_.IsNeedAssignMemory(n.get(), idx)) {
    zero_memory_list_.emplace_back(n.get(), kOutput, idx, false);
    GELOGI("[ContinuousMem]node[%s] output %u, no need assign memory.", op_desc->GetNamePtr(), idx);
    return SUCCESS;
  }
  size_t out_streams_cnt = 1U;
  const auto &continuous_mem = continuous_mem_mng_.GetContinuousMem(n.get(), idx);
  is_op_reuse_mem_ = is_op_reuse_mem_ && continuous_mem.IsReuse();
  if (is_op_reuse_mem_) {
    int64_t begin_time = 0;
    int64_t out_time = 0;
    int64_t end_time = 0;
    GE_ASSERT_SUCCESS(GetContinuousMemLifeTime(continuous_mem, begin_time, out_time, end_time,
                                               out_streams_cnt));
    (void)out_time;
    life_begin_ = begin_time;
    life_end_ = end_time;
  }
  uint64_t memory_type = RT_MEMORY_HBM;
  GE_ASSERT_SUCCESS(GetContinuousMemType(continuous_mem, memory_type));
  /*
   * 对于连续输出-连续输入这种场景，不应该出现带有该属性的情况，明确报错不支持
   */
  for (const auto &continuous_node_out : continuous_mem.GetContinuousNodeOut()) {
    std::vector<int64_t> offsets;
    (void)AttrUtils::GetListInt(continuous_node_out.node_ptr_->GetOpDescBarePtr(),
                                ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets);
    GE_ASSERT_TRUE(offsets.empty(), "[ContinuousMem] check buffer fusion offset failed. node: %s has attr: %s",
                   continuous_node_out.node_ptr_->GetNamePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION.c_str());
  }
  const auto max_size = continuous_mem.GetTotalSize();
  const auto no_align_size = max_size;

  MemoryBlock *block = nullptr;
  NodeIndexIO node_index_io(n.get(), idx, kOut);
  auto symbol_iter = anchor_to_symbol_.find(node_index_io.ToString());
  GE_ASSERT_TRUE(symbol_iter != anchor_to_symbol_.end());
  const auto symbol = symbol_iter->second;

  const auto iter = symbol_mem_reuse_info_.find(symbol);
  if (iter != symbol_mem_reuse_info_.end()) {
    iter->second.size_ = max_size;
  }

  const auto block_size = GetBlockSize(max_size, ranges, reuse_strategy_.use_range_);
  std::vector<bool> workspace_reuse_flag;
  const auto is_zeor_copy = IsZeroCopyBlock(n, idx, true, no_align_size);
  ApplyMemoryParam param = {block_size, max_size, no_align_size, kOutput, static_cast<uint32_t>(idx), is_op_reuse_mem_,
                            false, memory_type, is_zeor_copy};
  block = ApplyMemory(n, workspace_reuse_flag, param);
  GE_ASSERT_NOTNULL(block);
  GE_ASSERT_SUCCESS(continuous_mem_mng_.PushBackBlock(n.get(), idx, block));
  block->need_same_offset_in_batch_ = SizeIndependentOfBatch(n->GetTypePtr());
  if (continuous_mem.IsUseOneBlock()) {
    block->SetFirstContinuousBlock();
    block->SetLastContinuousBlock();
  }
  bool is_reuse_zero_copy = true;
  const auto ref_count = GetAllRefCount(node_index_io, is_reuse_zero_copy);
  block->is_reuse_zero_copy_ = (block->is_reuse_zero_copy_) && (is_reuse_zero_copy);
  block->ref_count_ = block->ref_count_ + ref_count;

  if (iter != symbol_mem_reuse_info_.cend()) {
    block->is_fixed_addr_prior_ = (block->is_fixed_addr_prior_ || iter->second.is_fixed_addr_prior_);
  }
  MarkReuseZeroCopyBlockFlag(n, block, idx);
  MarkZeroCopyBlockAttr(bool_attr_, op_desc, block->is_zero_copy_, kOutput, idx);
  GELOGI("[ContinuousMem]Node name: %s index:%u size:%zu ref count: %d, zero copy:%d, fixed addr prior: %d, "
      "out_streams_cnt: %zu, memory_type: %d", n->GetNamePtr(), idx, block_size, block->ref_count_,
      block->is_zero_copy_, block->is_fixed_addr_prior_, out_streams_cnt, memory_type);

  block->SetOutStreamCount(out_streams_cnt);
  block->is_reuse_zero_copy_ = (is_reuse_zero_copy && block->is_reuse_zero_copy_);
  symbol_blocks_[symbol] = block;
  // The output is suspended, and will be released in allocation of next node.
  CheckAndReleaseSuspendedBlock(n, idx, block);
  return SUCCESS;
}

Status BlockMemAssigner::GetContinuousMemType(const ContinuousMem &continuous_mem, uint64_t &memory_type) const {
  const auto &all_outs = continuous_mem.GetContinuousNodeOut();
  memory_type = RT_MEMORY_HBM;
  uint64_t first_special_type = RT_MEMORY_HBM;
  for (size_t i = 0U; i < all_outs.size(); ++i) {
    NodeIndexIO node_index_io(all_outs.at(i).node_ptr_, all_outs.at(i).index_, kOut);
    auto symbol_iter = anchor_to_symbol_.find(node_index_io.ToString());
    GE_ASSERT_TRUE(symbol_iter != anchor_to_symbol_.end());
    const auto symbol = symbol_iter->second;

    const auto iter = symbol_mem_reuse_info_.find(symbol);
    GE_ASSERT_TRUE(iter != symbol_mem_reuse_info_.end());
    if (iter->second.mem_type_ > iter->second.mem_type_) {
      memory_type = iter->second.mem_type_;
    }

    if (MemTypeUtils::IsMemoryTypeSpecial(static_cast<int64_t>(iter->second.mem_type_))) {
      // 记录第一个特殊类型内存
      if (first_special_type == RT_MEMORY_HBM) {
        first_special_type = memory_type;
      } else {
        GE_ASSERT_TRUE(first_special_type == memory_type, "memory type conflict,"
                       " there are two different special memory type in one continuous memory block[%llu, %llu]",
                       first_special_type, memory_type);
      }
    }
  }
  return SUCCESS;
}

/*
 * begin_time: 符号最小的id
 * out_time:   符号内最大的id
 * end_time:   首节点的流和输出节点的流如果不一样，要计算回到首节点流的id. end_time大于等于out_time
 */
Status BlockMemAssigner::GetContinuousMemLifeTime(const ContinuousMem &continuous_mem, int64_t &begin_time,
    int64_t &out_time, int64_t &end_time, size_t &out_streams_cnt) const {
  const auto &all_out = continuous_mem.GetContinuousNodeOut();
  GE_ASSERT_TRUE(!all_out.empty());
  const auto &first_node = all_out.front();
  begin_time = first_node.node_ptr_->GetOpDescBarePtr()->GetId();
  std::set<int64_t> streams;
  for (const auto &node_out : all_out) {
    begin_time = std::min(begin_time, node_out.node_ptr_->GetOpDescBarePtr()->GetId());
    const auto ret = GetNodeMaxLifeBySymbol(symbol_to_anchors_, node_out.node_ptr_, node_out.index_, out_time, streams,
                                            out_stream_edges_, GetStreamId(first_node.node_ptr_->GetOpDescBarePtr()));
    end_time = std::max(end_time, ret);
  }
  out_streams_cnt = streams.size();
  return SUCCESS;
}

int32_t BlockMemAssigner::GetAllRefCount(const NodeIndexIO &out_node_index_io, bool &is_reuse_zero_copy) const {
  int32_t ref_count = 0;
  auto iter_symbol = anchor_to_symbol_.find(out_node_index_io.ToString());
  if (iter_symbol == anchor_to_symbol_.end()) {
    return ref_count;
  }

  auto iter = symbol_to_anchors_.find(iter_symbol->second);
  if (iter != symbol_to_anchors_.end()) {
    for (const auto &node_index_io : iter->second) {
      if (node_index_io.node_ptr_ == nullptr) {
        continue;
      }
      is_reuse_zero_copy = (is_reuse_zero_copy && CanReuseZeroCopyBlock(node_index_io.node_ptr_));
      if (node_index_io.io_type_ != kIn) {
        continue;
      }
      if ((node_index_io.node_ptr_->GetInDataAnchor(node_index_io.index_) == nullptr) ||
          (node_index_io.node_ptr_->GetInDataAnchor(node_index_io.index_)->GetPeerOutAnchor() == nullptr)) {
        GELOGI("Node: %s has no input.", node_index_io.node_ptr_->GetNamePtr());
        continue;
      }
      ref_count++;
    }
    GELOGD("symbol=%s, ref count is %d", out_node_index_io.ToString().c_str(), ref_count);
  }
  return ref_count;
}

Status BlockMemAssigner::GetOutputTotalSizeAndOutCount(const NodePtr &n, uint32_t output_index, size_t &max_size,
                                                       size_t &no_align_size, int32_t &out_count,
                                                       bool is_separate_clean_continuous_inputs) const {
  const auto node_op_desc = n->GetOpDescBarePtr();
  GE_CHECK_NOTNULL(node_op_desc);
  const auto out_data_anchor = n->GetOutDataAnchor(static_cast<int32_t>(output_index));
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    auto owner_node = in_anchor->GetOwnerNodeBarePtr();
    auto op_desc = owner_node->GetOpDescBarePtr();
    if (op_desc == nullptr) {
      continue;
    }
    Params *instance = Params::Instance();
    GE_CHECK_NOTNULL(instance);
    if (!((instance->GetTarget() == TARGET_TYPE_TINY) && (op_desc->GetType() == NETOUTPUT))) {
      out_count++;
    }
  }

  if (!is_separate_clean_continuous_inputs) {
    const auto output_op_desc = node_op_desc->MutableOutputDesc(output_index);
    GE_CHECK_NOTNULL(output_op_desc);
    int64_t size = 0;
    GE_CHK_STATUS_RET(MemReuseUtils::GetTensorSize(*output_op_desc, size,
                                                   MemReuseUtils::IsNeedSplitSize(n, output_index)),
                      "Get node %s output %ld size failed", node_op_desc->GetNamePtr(), output_index);
    max_size = static_cast<size_t>(size);
    GE_CHK_STATUS_RET(MemReuseUtils::GetOutputNoAlignSize(*node_op_desc, output_index, no_align_size),
                      "Get node_name:%s, output_index:%u no align size failed", n->GetNamePtr(), output_index);
    return SUCCESS;
  }

  for (const auto out_node_in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    if (out_node_in_anchor == nullptr) {
      continue;
    }
    const auto out_node = out_node_in_anchor->GetOwnerNodeBarePtr();
    bool is_input_continuous = MemLayoutConflictUtil::IsContinuousInput(out_node);
    if (!is_input_continuous) {
      continue;
    }
    size_t total_size = 0U;
    for (const auto input_anchor : out_node->GetAllInDataAnchorsPtr()) {
      GE_CHECK_NOTNULL(input_anchor);
      auto in_node_out_anchor = input_anchor->GetPeerOutAnchor();
      if (in_node_out_anchor == nullptr) {
        continue;
      }
      const auto in_node = in_node_out_anchor->GetOwnerNodeBarePtr();
      const auto in_op_desc = in_node->GetOpDescBarePtr();
      GE_CHECK_NOTNULL(in_op_desc);
      const auto output_op_desc = in_op_desc->MutableOutputDesc(in_node_out_anchor->GetIdx());
      int64_t size = 0;
      GE_CHK_STATUS_RET(MemReuseUtils::GetTensorSize(*output_op_desc, size,
                                                     MemReuseUtils::IsNeedSplitSize(n, output_index)),
                        "Get node %s out %ld size failed", in_node->GetNamePtr(), in_node_out_anchor->GetIdx());
      size_t align_size = static_cast<size_t>(size);
      MemReuseUtils::AlignMemOffset(align_size);
      total_size += align_size;
    }
    max_size = max_size < total_size ? total_size : max_size;
  }
  no_align_size = max_size;

  return SUCCESS;
}

void BlockMemAssigner::CalExitSymbolNodeLifeTime(const Node *const n, uint32_t out_index, size_t &max_life_time) {
  max_life_time = life_time_;
  bool is_cur_node_input_continuous = false;
  (void)ge::AttrUtils::GetBool(n->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                               is_cur_node_input_continuous);
  if (!is_cur_node_input_continuous) {
    is_cur_node_input_continuous = IsSeparateCleanContinuousInputNode(n);
  }
  const auto &out_anchor = n->GetOutDataAnchor(out_index);
  if (is_cur_node_input_continuous || (out_anchor == nullptr)) {
    return;
  }
  std::set<int64_t> streams;
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    const auto &out_node = peer_in_anchor->GetOwnerNodeBarePtr();
    bool is_input_continuous = false;
    (void)ge::AttrUtils::GetBool(out_node->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                 is_input_continuous);
    if (!is_input_continuous) {
      is_input_continuous = IsSeparateCleanContinuousInputNode(out_node);
    }
    if (!is_input_continuous) {
      continue;
    }
    if (peer_in_anchor->GetIdx() != 0) {
      continue;
    }
    int64_t node_max_life_time = 0;
    CalNodeAsContinuousInputMaxLife(n, out_index, out_node, node_max_life_time, streams);
    if (max_life_time < static_cast<size_t>(node_max_life_time)) {
      max_life_time = static_cast<size_t>(node_max_life_time);
    }
  }
  GELOGI("Node[%s:%u] has exit symbol, it's max_life_time:%zu stream count:%zu", n->GetNamePtr(), out_index,
         max_life_time, streams.size());
}

void BlockMemAssigner::MarkReuseZeroCopyBlockFlag(const NodePtr &n, MemoryBlock *const block,
                                                  const uint32_t index) const {
  auto node_op_desc = n->GetOpDescBarePtr();

  // 输出连续内存不能做零拷贝，同时NoPadding且Reuse的也不能做零拷贝
  bool can_reuse_zero_copy = true;
  bool is_continuous = ge::MemLayoutConflictUtil::IsContinuousOutput(n);
  bool is_nopadding_continuous = false;

  if (!is_continuous) {
    (void)ge::AttrUtils::GetBool(*node_op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_nopadding_continuous);
    if (is_nopadding_continuous) {
      bool attr_reuse = false;
      (void)ge::AttrUtils::GetBool(*node_op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
      can_reuse_zero_copy = !attr_reuse;
    }
  } else {
    can_reuse_zero_copy = false;
  }

  if (!can_reuse_zero_copy) {
    block->is_reuse_zero_copy_ = false;
  }

  // data直连特殊的rts算子且不可刷新状态下，不能做零拷贝
  const static std::set<std::string> kTaskUnsupportZeroCopyOp{ge::STREAMSWITCH, ge::LABELSWITCHBYINDEX};
  auto out_anchor = n->GetOutDataAnchor(static_cast<int32_t>(index));
  bool data_connect_unsupport_zero_copy = false;
  if (OpTypeUtils::IsDataNode(node_op_desc->GetType()) && (!is_feature_map_refreshable_) && (out_anchor != nullptr)) {
    auto peer_anchors = out_anchor->GetPeerInDataAnchorsPtr();
    data_connect_unsupport_zero_copy =
        std::any_of(peer_anchors.begin(), peer_anchors.end(), [](const ge::InDataAnchor *anchor) {
      return ((anchor != nullptr) && (anchor->GetOwnerNodeBarePtr() != nullptr) &&
              (kTaskUnsupportZeroCopyOp.count(anchor->GetOwnerNodeBarePtr()->GetTypePtr()) > 0U));
    });
  }

  if (data_connect_unsupport_zero_copy) {
    block->is_reuse_zero_copy_ = false;
    block->is_zero_copy_ = false;
  }

  GELOGD("Node name: %s index: %d, can_reuse_zero_copy: %s, data_connect_unsupport_zero_copy: %s ", n->GetNamePtr(),
         index, can_reuse_zero_copy ? "true" : "false", data_connect_unsupport_zero_copy ? "true" : "false");
}

MemoryBlock *BlockMemAssigner::ApplyOutMemory(const NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges,
                                              const bool is_op_reuse_mem, const bool out_node_need_continuous_input) {
  if (index >= n->GetAllOutDataAnchorsSize()) {
    GELOGE(FAILED, "[Check][OutIndex]index:%u exceed out_size:%u, node_name:%s", index, n->GetAllOutDataAnchorsSize(),
           n->GetNamePtr());
    return nullptr;
  }
  const auto out_data_anchor = n->GetOutDataAnchor(index);
  auto node_op_desc = n->GetOpDescBarePtr();
  if ((out_data_anchor == nullptr) || (node_op_desc == nullptr)) {
    GELOGE(FAILED, "[Check][OutAnchor]is null, index:%u, node_name:%s", index, n->GetNamePtr());
    return nullptr;
  }

  size_t size = 0U;
  size_t no_align_size = 0U;
  int32_t out_count = 0;
  size_t block_size = 0U;
  if (GetOutputTotalSizeAndOutCount(n, index, size, no_align_size, out_count, is_separate_clean_continuous_inputs_) !=
      SUCCESS) {
    GELOGE(FAILED, "Get output total size failed");
    return nullptr;
  }

  std::string symbol;
  MemoryBlock *block = nullptr;
  NodeIndexIO node_index_io(n.get(), index, kOut);
  if (IsSymbolExist(node_index_io, symbol, block)) {
    GE_IF_BOOL_EXEC(block == nullptr,
      REPORT_INNER_ERR_MSG("E19999", "get ref block failed, node_name:%s, symbol:%s",
                         node_op_desc->GetNamePtr(), node_index_io.ToString().c_str());
      GELOGE(FAILED, "[Get][RefBlock]node_name:%s, symbol:%s",
             node_op_desc->GetNamePtr(), node_index_io.ToString().c_str());
      return nullptr);

    const bool
        cal_theory_size = (!block->RealSizeList().empty()) && (block->NodeTypeIndexList().back().node_ != nullptr)
        && (block->NodeTypeIndexList().back().node_->GetOpDescBarePtr() != nullptr)
        && (block->NodeTypeIndexList().back().node_->GetOpDescBarePtr()->GetType() == ge::PARTITIONEDCALL)
        && (node_op_desc->GetType() != ge::PARTITIONEDCALL);
    if (cal_theory_size) {
      AddMemoryStat(block->memory_type_, block->RealSizeList().back(), block->reuse_mem_);
    }

    block_size = GetBlockSize(size, ranges, reuse_strategy_.use_range_);
    block->SetSize(block_size);
    size_t symbol_life_time = life_time_;
    CalExitSymbolNodeLifeTime(node_index_io.node_ptr_, node_index_io.index_, symbol_life_time);
    block->SetSymbolLifeEnd(symbol_life_time);
    block->SetLifeTimeEnd(life_time_, block->stream_id_);
    block->AddNodeTypeIndex(
        {n.get(), kOutput, index, true, life_begin_,
         GetStreamId(node_op_desc), false, block->GetSymbolLifeEnd()},
        size, no_align_size, block->stream_id_);
    block->has_sub_graph_in_out_node_ = block->has_sub_graph_in_out_node_ ||
                                        PeerIsSubGraphNetOutNode(n, out_data_anchor, compute_graph_) ||
                                        IsSubGraphInOrOutNode(n.get(), compute_graph_);
    bool no_reuse_flag = false;
    (void)ge::AttrUtils::GetBool(node_op_desc, kOpNoReuseMem, no_reuse_flag);
    block->reuse_mem_ = block->reuse_mem_ && (!no_reuse_flag) && is_op_reuse_mem;
    if (out_count == 0) {
      block->ref_count_++;
    }
  } else {
    // if ref input is variable or const(not alloc memory in reuse), can not find ref block, must judge alone
    // after unfolding dynamic shape graph, const or varible may be in root graph and can not be ref.
    if (IsOutputIndexRef(node_op_desc, index) ||
        (IsSubgraphDataRefConstInput(n) && (!IsDirectInputNode(n.get(), compute_graph_)))) {
      zero_memory_list_.emplace_back(n.get(), kOutput, index, false);
      GELOGI("ref mode skip out block assign. node_name: %s, index:%d", n->GetNamePtr(), index);
      return nullptr;
    }

    size_t max_size = size;
    auto iter = symbol_mem_reuse_info_.find(symbol);
    // In separate clean policy, node as continuous input, its output takes the largest continuous input size of its
    // output nodes, which has been calculated above
    if (iter != symbol_mem_reuse_info_.end()) {
      if (!is_separate_clean_continuous_inputs_) {
        max_size = iter->second.size_;
      } else {
        iter->second.size_ = max_size;
      }
    }

    uint64_t memory_type = RT_MEMORY_HBM;
    if (iter != symbol_mem_reuse_info_.cend()) {
      memory_type = iter->second.mem_type_;
    }

    block_size = GetBlockSize(max_size, ranges, reuse_strategy_.use_range_);
    std::vector<bool> workspace_reuse_flag;
    bool as_input_continuous = is_separate_clean_continuous_inputs_ ? false : out_node_need_continuous_input;
    bool is_zeor_copy = IsZeroCopyBlock(n, index, out_node_need_continuous_input, no_align_size);
    ApplyMemoryParam param = {block_size, max_size, no_align_size, kOutput, index, is_op_reuse_mem, as_input_continuous,
                              memory_type, is_zeor_copy};
    block = ApplyMemory(n, workspace_reuse_flag, param);
    GE_CHECK_NOTNULL_EXEC(block, return nullptr);
    block->need_same_offset_in_batch_ = SizeIndependentOfBatch(n->GetTypePtr());
    if (is_separate_clean_continuous_inputs_) {
      // hccl task need align header and tail
      block->SetFirstContinuousBlock();
      block->SetLastContinuousBlock();
    }
    // Data and netoutput need zero copy block
    block->is_zero_copy_ =
        block->is_zero_copy_ ||
        IsZeroCopyBlock(n, index, (out_node_need_continuous_input || block->GetContinuousFlag()), no_align_size);

    bool is_reuse_zero_copy = true;
    int32_t ref_count = GetAllRefCount(node_index_io, is_reuse_zero_copy);
    block->is_reuse_zero_copy_ = (block->is_reuse_zero_copy_) && (is_reuse_zero_copy);
    // in case symbol ref ref_count is total input ref
    if (ref_count > out_count) {
      out_count = ref_count;
    }
    block->ref_count_ = block->ref_count_ + out_count;

    if (iter != symbol_mem_reuse_info_.cend()) {
      block->is_fixed_addr_prior_ = (block->is_fixed_addr_prior_ || iter->second.is_fixed_addr_prior_);
    }

    GELOGD("Node name: %s size:%zu ref count: %d, out count: %d zero copy:%d, fixed addr prior: %d", n->GetNamePtr(),
      block_size, block->ref_count_, out_count, block->is_zero_copy_, block->is_fixed_addr_prior_);
  }

  MarkReuseZeroCopyBlockFlag(n, block, index);
  MarkZeroCopyBlockAttr(bool_attr_, node_op_desc, block->is_zero_copy_, kOutput, index);
  GELOGI("Node name: %s index:%u size:%zu ref count: %d, out count: %d zero copy:%d, out node need continuous input %d",
         n->GetNamePtr(), index, block_size, block->ref_count_, out_count, block->is_zero_copy_,
         out_node_need_continuous_input);
  return block;
}

MemoryBlock *BlockMemAssigner::ApplyOutDescMemory(const NodePtr &n, uint32_t index,
                                                  const std::vector<int64_t> &ranges) {
  GELOGI("Node[%s] tensor[%u] apply output desc memory.", n->GetNamePtr(), index);
  size_t size = sizeof(RuntimeTensorDesc);
  auto block_size = GetBlockSize(size, ranges, reuse_strategy_.use_range_);
  std::vector<bool> workspace_reuse_flag;

  MemoryBlock *block = nullptr;
  NodeIndexIO node_index_io(n.get(), index, kOut);
  std::string symbol;
  if (IsSymbolDescBlockExist(node_index_io, symbol, block)) {
    GE_IF_BOOL_EXEC(block == nullptr,
      REPORT_INNER_ERR_MSG("E19999", "get ref block failed, node_name:%s, symbol:%s",
                         n->GetNamePtr(), node_index_io.ToString().c_str());
      GELOGE(FAILED, "[Get][RefBlock]node_name:%s, symbol:%s",
             n->GetNamePtr(), node_index_io.ToString().c_str());
      return nullptr);

    block->AddNodeTypeIndex({n.get(), kOutputDesc, index, true, 0}, size, size, GetStreamId(n->GetOpDescBarePtr()));
    GELOGD("Ref tensor desc, symbol[%s], anchor[%s].", symbol.c_str(), node_index_io.ToString().c_str());
  } else {
    ApplyMemoryParam param = {block_size, size, size, kOutputDesc, index, false, false, RT_MEMORY_HBM, false};
    block = ApplyMemory(n, workspace_reuse_flag, param);
    if (block == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "apply out desc Memory failed, node_name:%s, block_size:%ld, out_index:%u",
                        n->GetNamePtr(), block_size, index);
      GELOGE(FAILED, "[Apply][Memory]node_name:%s, block_size:%ld, out_index:%u", n->GetNamePtr(),
             block_size, index);
      return nullptr;
    }

    bool is_fixed_addr_prior = false;
    (void) ge::AttrUtils::GetBool(n->GetOpDesc(), ATTR_NAME_IS_FIXED_ADDR_PRIOR, is_fixed_addr_prior);
    block->is_fixed_addr_prior_ = (block->is_fixed_addr_prior_ || is_fixed_addr_prior);
    GELOGD("%s's output desc memory fixed addr prior:%d, index:%zu.",
      n->GetNamePtr(), block->is_fixed_addr_prior_, index);
  }
  auto iter = anchor_to_symbol_.find(node_index_io.ToString());
  if (iter != anchor_to_symbol_.end()) {
    GELOGD("Add to ref tensor, symbol[%s], anchor[%s] ", iter->second.c_str(), node_index_io.ToString().c_str());
    symbol_desc_blocks_[iter->second] = block;
  }

  return block;
}

bool IsOutputBlock(const ge::InDataAnchor *const in_data_anchor) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(peer_out_anchor == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "Peer out anchor is nullptr.");
                  GELOGE(FAILED, "[Check][Param] Peer out anchor is nullptr."); return false);
  auto src = peer_out_anchor->GetOwnerNodeBarePtr();
  int32_t index = peer_out_anchor->GetIdx();
  auto iter = GetLocalOmgContext().out_nodes_map.find(src->GetNamePtr());
  if (iter != GetLocalOmgContext().out_nodes_map.end()) {
    for (auto id : iter->second) {
      if (index == id) {
        return true;
      }
    }
  }
  return false;
}

// atomic out memory will be reassigned
bool BlockMemAssigner::IsAtomicOutputMemory(const ge::NodePtr &node, uint32_t output_index, bool is_atomic,
                                            bool out_node_set_continuous_input) const {
  auto op_desc = node->GetOpDescBarePtr();
  if (op_desc == nullptr) {
    return false;
  }

  // if node need continue output, need assign memory.
  bool is_output_continuous = ge::MemLayoutConflictUtil::IsContinuousOutput(node);
  if (!is_output_continuous) {
    (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_output_continuous);
  }

  if ((!out_node_set_continuous_input) && is_atomic && (!is_output_continuous)) {
    if (IsZeroCopyBlock(node, output_index, is_output_continuous)) {
      GELOGI("atomic clean and zero copy, need assign memory, node:%s(%s), output_index: %u",
             node->GetNamePtr(), node->GetTypePtr(), output_index);
      return false;
    }
    std::vector<int64_t> atomic_output_index;
    // If GetListInt fail, atomic_output_index is empty.
    (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    for (auto &index : atomic_output_index) {
      if (static_cast<uint32_t>(index) == output_index) {
        if (node->GetOwnerComputeGraphBarePtr() != nullptr) {
          GELOGD("Atomic no assign %s name[%s] output[%ld] streamid[%ld].",
                 node->GetOwnerComputeGraphBarePtr()->GetName().c_str(), op_desc->GetNamePtr(), index,
                 GetStreamId(op_desc));
        }
        return true;
      }
    }
  }
  return false;
}

bool IsKnownSubgraphData(const Node *node) {
  if ((node == nullptr) || NodeUtils::IsDynamicShape(*node)) {
    return false;
  }

  return node->GetOpDescBarePtr()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
}

void SetReleaseBlockLifeEnd(MemoryBlock *to_release, int64_t stream_id) {
  const auto to_release_out_stream_life_time = to_release->NodeTypeIndexList().back().out_stream_life_time_;
  if (to_release_out_stream_life_time.size() == 1) {
    for (const auto &item : to_release_out_stream_life_time) {
      size_t end_life_time = item.second.second;
      int64_t release_stream_id = to_release->stream_id_;
      if (to_release->stream_id_ != item.first) {
        // kMaxLifeTime means can not return self stream, so need to set kMaxLifeTime to self stream, set real end time
        // to out stream.
        if (end_life_time == kMaxLifeTime) {
          release_stream_id = stream_id;
          end_life_time = item.second.first;
        }
      }
      to_release->SetLifeTimeEnd(end_life_time, release_stream_id);
      // stream 0->1->2->0场景，增加一个0->1的结束点，2上面的内存有机会和0上的复用
      if (to_release->diff_stream_prior_ && (to_release->stream_id_ != stream_id)) {
        to_release->SetLifeTimeEnd(item.second.first, stream_id);
      }
    }
    return;
  }
  size_t max_end_life_time = 0U;
  for (const auto &item : to_release_out_stream_life_time) {
    if (max_end_life_time < item.second.second) {
      max_end_life_time = item.second.second;
    }
  }
  if (max_end_life_time == kMaxLifeTime) {
    to_release->same_stream_ = false;
  }
  to_release->SetLifeTimeEnd(max_end_life_time, to_release->stream_id_);
}

void BlockMemAssigner::ReleaseMemory(MemoryBlock *const to_release, std::vector<MemoryBlock *> &reusable_memory,
                                     int64_t stream_id, const std::string &symbol, bool no_release) {
  if ((to_release == nullptr) || to_release->NodeTypeIndexList().empty()) {
    GELOGE(FAILED, "[Check][Param] Input parameter to_release is null.");
    return;
  }
  if (to_release->ref_count_ <= 0) {
    GELOGI("[Check][Param] to_release->ref_count_ must greater than 0");
    return;
  }

  if (!to_release->reuse_mem_) {
    GELOGI("[Check][Param] doesn't reuse memory");
    return;
  }
  int64_t max_life_time = life_time_;
  if (!to_release->used_by_diff_streams_) {
    to_release->used_by_diff_streams_ = (to_release->stream_id_ != stream_id) &&
                                        (to_release->NodeTypeIndexList().back().mem_type_ != kWorkspace);
  }

  int64_t max_node_life_time_by_symbol = life_time_;
  if (to_release->used_by_diff_streams_) {
    const auto &node_type_index_list = to_release->NodeTypeIndexList();
    const auto &node_type_index_iter =
        std::find_if(node_type_index_list.rbegin(), node_type_index_list.rend(),
                     [](const NodeTypeIndex &node_type_index) { return !node_type_index.ref_input_; });
    const auto &node_type_index = *node_type_index_iter;
    if (node_type_index.node_ != nullptr) {
      std::set<int64_t> streams;
      max_life_time = GetNodeMaxLife(symbol_to_anchors_, out_stream_edges_, node_type_index.node_,
                                     node_type_index.index_, max_node_life_time_by_symbol, streams);
      if (node_type_index_list.back().ref_input_) {
        to_release->SetOutStreamCount(streams.size());
      }
      GELOGI("Diff stream output node:%s max life time:%ld, back is ref: %d, streams size: %zu",
             node_type_index.node_->GetNamePtr(), max_life_time, node_type_index_list.back().ref_input_,
             streams.size());
      if (to_release->GetFirstContinuousFlag() && to_release->GetLastContinuousFlag()) {
        GetContinuousOutputMaxLifeBySymbol(node_type_index.node_, symbol_to_anchors_, max_node_life_time_by_symbol,
            out_stream_edges_);
      }
      GELOGI("Diff stream output node:%s max life time:%ld by symbol", node_type_index.node_->GetNamePtr(),
             max_node_life_time_by_symbol);
    }
  }
  to_release->SetOutStreamLifeTime(max_node_life_time_by_symbol, max_life_time, stream_id);

  --to_release->ref_count_;
  if (to_release->ref_count_ > 0) {
    return;
  }
  if (to_release->reuse_mem_ && (!to_release->RealSizeList().empty())
      && (to_release->batch_label_.empty() || (to_release->batch_label_ == max_batch_label_))) {
    size_t align_size = to_release->RealSizeList().back();
    if (!symbol.empty()) {
      const auto it_size = symbol_mem_reuse_info_.find(symbol);
      if (it_size != symbol_mem_reuse_info_.cend()) {
        align_size = it_size->second.size_;
      }
    }
    MemReuseUtils::AlignMemOffset(align_size);
    if (memory_stat_[to_release->memory_type_].theory_memory_size_ >= align_size) {
      memory_stat_[to_release->memory_type_].theory_memory_size_ -= align_size;
    }
  }
  SetReleaseBlockLifeEnd(to_release, stream_id);
  // model net output can reuse other, but it can't be reused
  if (!IsPostReuse(to_release) || no_release) {
    max_life_time = ge::kMaxLifeTime;
    to_release->ClearDiffStreamLifeInfo();
    to_release->SetLifeTimeEnd(ge::kMaxLifeTime, to_release->stream_id_);
  }
  reusable_memory.emplace_back(to_release);
  to_release->ClearOutStreamLifeInfo();
  to_release->used_by_diff_streams_ = false;
  GELOGD("Put block:%s to pool stream:%ld max life time:%ld", GetName(*to_release, true).c_str(),
         to_release->stream_id_, max_life_time);
}

void BlockMemAssigner::ReleaseMemorys(StreamIdToBlocks &to_releases,
                                      StreamIdToBlocks &reusable_memory) {
  // [stream id][blocks]
  for (auto &stream_blocks : to_releases) {
    for (auto mem_block : stream_blocks.second) {
      if (mem_block != nullptr) { // mem_block 不可能为空
        const bool output = (!mem_block->NodeTypeIndexList().empty()) &&
                            (mem_block->NodeTypeIndexList().back().mem_type_ == kOutput) &&
                            (!mem_block->SymbolList().empty());
        ReleaseMemory(mem_block, reusable_memory[stream_blocks.first], mem_block->stream_id_,
                      output ? mem_block->SymbolList().back() : "", false);
      }
    }
    GELOGD("Clear stream:%ld workspace blocks", stream_blocks.first);
    stream_blocks.second.clear();
  }
}

void BlockMemAssigner::ReleaseInputNodeOutMemory(const NodePtr &node) {
  for (const auto &in_anchor : GetSortAllInDataAnchors(node, IsMemoryPriorityMode())) {
    if ((node->GetOpDescBarePtr() == nullptr) || (in_anchor->GetPeerOutAnchor() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr)) {
      continue;
    }
    GE_IF_BOOL_EXEC(IsOutputBlock(in_anchor), continue);

    std::string op_type(in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetTypePtr());
    GE_IF_BOOL_EXEC((op_type == CONSTANT) || (op_type == FASTRCNNPREDICTIONS) || (op_type == CONSTANTOP),
                    continue);
    const bool
        is_no_release_node_out_block = IsNoReleaseNodeOutBlock(in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr());

    const auto in_data_node = in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_CHECK_NOTNULL_JUST_RETURN(in_data_node);
    const int32_t in_data_node_out_index = in_anchor->GetPeerOutAnchor()->GetIdx();
    NodeIndexIO node_index_io(in_data_node, in_data_node_out_index, kOut);
    MemoryBlock *block = nullptr;
    std::string symbol;
    if (!IsSymbolExist(node_index_io, symbol, block)) {
      GELOGI(
          "Block of peer out not find. Peer node:%s, output index:%d, "
          "current node:%s, input index:%d",
          in_data_node->GetNamePtr(), in_data_node_out_index, node->GetNamePtr(), in_anchor->GetIdx());
      continue;
    }
    GE_CHECK_NOTNULL_JUST_RETURN(block);
    auto reusable_blocks_iter = reusable_blocks_.find(block->memory_type_);
    if (reusable_blocks_iter == reusable_blocks_.end() || block->NodeTypeIndexList().empty()) {
      continue;
    }

    GELOGI(
        "Block of peer out is matched. Peer node:%s, output index:%d, "
        "current node:%s, input index:%d, block ref_count:%d.",
        in_data_node->GetNamePtr(), in_data_node_out_index, node->GetNamePtr(), in_anchor->GetIdx(), block->ref_count_);

    auto stream_id = GetStreamId(node->GetOpDescBarePtr());
    StreamIdToBlocks &reusable_memory = reusable_blocks_iter->second;
    ReleaseMemory(block, reusable_memory[block->stream_id_], stream_id,
                  symbol, is_no_release_node_out_block);
    if (block->ref_count_ == 0 && (stream_id == block->stream_id_)) {
      SetLastUsedInputMemAttr(node, in_anchor->GetIdx(), bool_attr_);
    }
  }
}

void CheckAndGetOpReuseEnv(const std::string &env, std::vector<std::string> &env_vec, bool &op_reuse_env_valid) {
  std::string env_str = std::string(env);
  if (env_str.size() > kReuseMaxCharNum) {
    GELOGE(FAILED, "[Check][Param] The OP_NO_REUSE_MEM has more than %d characters.", kReuseMaxCharNum);
    return;
  }

  SplitStringByComma(env_str, env_vec);
  if (env_vec.size() > kReuseMaxOpNum) {
    GELOGE(FAILED, "[Check][Param] The OP_NO_REUSE_MEM has more than %d nodes.", kReuseMaxOpNum);
    return;
  }

  op_reuse_env_valid = true;
  return;
}

void BlockMemAssigner::CheckAndReleaseSuspendedBlock(const NodePtr &node, uint32_t idx, MemoryBlock *block) {
  if ((node == nullptr) || (block == nullptr)) {
    return;
  }
  if (block->ref_count_ == 0) {
    block->ref_count_ = 1;
    stream_workspace_blocks_[block->memory_type_][block->stream_id_].emplace_back(block);
    GELOGI("The output is suspended, and will be released in allocation of next node. Name:%s, index:%u, "
           "size:%zu, ref_count:%d, batch:%s, stream:%ld, reuse_mem:%d.", node->GetNamePtr(), idx, block->Size(),
           block->ref_count_, block->batch_label_.c_str(), block->stream_id_, block->reuse_mem_);
  }
}

Status BlockMemAssigner::AssignOutputMemoryWithReuse(const NodePtr &node, std::vector<int64_t> &ranges) {
  auto op_desc = node->GetOpDescBarePtr();
  std::vector<int64_t> memorys_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memorys_type);
  GELOGD("Assign memory node[%s], output size[%zu], output memory type size[%zu]", op_desc->GetNamePtr(),
         op_desc->GetOutputsSize(), memorys_type.size());
  if (has_mem_type_attr && (memorys_type.size() != op_desc->GetOutputsSize())) {
    REPORT_INNER_ERR_MSG("E19999", "Attr[%s] size:%zu not equal to node output size:%zu, node_name:%s",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), memorys_type.size(),
                       op_desc->GetOutputsSize(), op_desc->GetNamePtr());
    GELOGE(
        INTERNAL_ERROR,
        "[Check][MemTypeAttr]Attr %s size:%zu not equal to node output size:%zu, node_name:%s",
        ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), memorys_type.size(),
        op_desc->GetOutputsSize(), op_desc->GetNamePtr());
    return INTERNAL_ERROR;
  }

  // restore node-level flags
  is_op_reuse_mem_ = true;

  if (op_reuse_env_valid_) {
    std::vector<std::string>::iterator it_name =
      std::find(op_no_reuse_mem_vec_.begin(), op_no_reuse_mem_vec_.end(), op_desc->GetNamePtr());
    std::vector<std::string>::iterator it_type =
      std::find(op_no_reuse_mem_vec_.begin(), op_no_reuse_mem_vec_.end(), op_desc->GetTypePtr());
    GE_IF_BOOL_EXEC(it_name != op_no_reuse_mem_vec_.end() || it_type != op_no_reuse_mem_vec_.end(),
                    is_op_reuse_mem_ = false;);
  }

  bool need_gentask_atomic = false;
  (void)ge::AttrUtils::GetBool(op_desc, "need_gentask_atomic", need_gentask_atomic);

  bool is_atomic = false;
  if (!need_gentask_atomic) {
    // If GetBool fail, is_atomic is false.
    (void)ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic);
  }

  // Allocate memory for the current node and release node memory of the same size in the workspace
  GE_IF_BOOL_EXEC(is_ge_reuse_mem_,
      for (auto iter = stream_workspace_blocks_.begin(); iter != stream_workspace_blocks_.end();
           ++iter) { ReleaseMemorys(iter->second, reusable_blocks_[iter->first]);
      });
  // work space is same life time, so set life_time_ after ReleaseMemorys for workspace
  life_time_ = op_desc->GetId();

  bool is_zero_mem_node = CheckIsZeroMemNodeType(node->GetTypePtr());
  bool is_buffer_pool_mem_supported = (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID)) &&
                                      (op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE)) && (!root_unknown_shape_flag_);
  bool need_apply_continuous_memory = IsContinuousOutput(node);
  // begin to assign memory for every output
  for (uint32_t i = 0U; i < static_cast<uint32_t>(op_desc->GetOutputsSize()); i++) {
    int64_t size = 0;
    auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    if (output_tensor_desc == nullptr) {
      GELOGW("op[%s] null output_tensor_desc, index[%u].", op_desc->GetNamePtr(), i);
      continue;
    }

    // every out need get again
    life_begin_ = op_desc->GetId();
    life_end_ = 0U;
    is_separate_clean_continuous_inputs_ = false;

    GE_IF_BOOL_EXEC(MemReuseUtils::GetTensorSize(*output_tensor_desc, size,
                                                 MemReuseUtils::IsNeedSplitSize(node, i)) != SUCCESS,
                    GELOGI("Tensor has no size"));

    // fusion: other type's size not means malloc HBM memory
    if (has_mem_type_attr && ((memorys_type[i] == RT_MEMORY_L1) || (memorys_type[i] == kRtMemoryUB))) {
      GELOGI("fusion: node[%s], output[%s], output memory type [%ld]",
             op_desc->GetNamePtr(), op_desc->GetOutputNameByIndex(i).c_str(), memorys_type[i]);
      size = 0;  // no need assgin block memory
    }

    int32_t calc_type = 0;
    bool ret = ge::AttrUtils::GetInt(output_tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    GE_IF_BOOL_EXEC((ret && (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY))), size = 0;);

    InDataAnchor *continuous_in_anchor = nullptr;
    bool out_node_set_continuous_input = false;
    bool no_need_assign_memory = (is_zero_mem_node || is_buffer_pool_mem_supported || (size == 0));
    if ((!no_need_assign_memory) && continuous_mem_mng_.IsFound(node.get(), i)) {
      return ApplyContinuousMemWithMng(node, i, ranges);
    }
    NodeIndexIO node_index_io(node.get(), i, kOut);
    bool is_reuse_zero_copy = true;
    std::set<int64_t> streams;
    if (!no_need_assign_memory) {
      out_node_set_continuous_input =
          IsOutNodeSetContinuousInput(node, i, continuous_in_anchor, is_reuse_zero_copy, streams);
      GE_ASSERT_SUCCESS(GetNoNeedAssignMemoryFlag(node, i, no_need_assign_memory));
      no_need_assign_memory = (no_need_assign_memory ||
                               IsAtomicOutputMemory(node, i, is_atomic, out_node_set_continuous_input) ||
                               IsNoNeedAssignMemory(node, node_index_io, i));
    }
    if (no_need_assign_memory) {
      zero_memory_list_.emplace_back(node.get(), kOutput, i, false);
      GELOGI("node[%s] output %u, no need assign memory.", op_desc->GetNamePtr(), i);
      continue;
    }

    if (need_apply_continuous_memory && (!out_node_set_continuous_input)) {
      return ApplyContinuousMemory(node, ranges, is_op_reuse_mem_);
    }

    // atomic can't be reused
    bool need_change = is_op_reuse_mem_ && is_atomic && out_node_set_continuous_input;
    GE_IF_BOOL_EXEC(need_change, is_op_reuse_mem_ = false);

    MemoryBlock *mem_block = ApplyOutMemory(node, i, ranges, is_op_reuse_mem_, out_node_set_continuous_input);
    if (mem_block != nullptr) {
      mem_block->SetOutStreamCount(streams.size());
      mem_block->is_reuse_zero_copy_ = (is_reuse_zero_copy && mem_block->is_reuse_zero_copy_);
      if (continuous_in_anchor != nullptr) {
        const auto continuous_op_desc = continuous_in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr();
        node_continuous_input_blocks_[continuous_op_desc->GetId()][continuous_in_anchor->GetIdx()]
            = mem_block;
      }
      auto iter = anchor_to_symbol_.find(node_index_io.ToString());
      if (iter != anchor_to_symbol_.end()) {
        symbol_blocks_[iter->second] = mem_block;
        GELOGD("Node io:%s add symbol:%s block:%s", node_index_io.ToString().c_str(), iter->second.c_str(),
               GetName(*mem_block).c_str());
        // The output is suspended, and will be released in allocation of next node.
        CheckAndReleaseSuspendedBlock(node, i, mem_block);
      }
    }

    bool is_tensor_desc_mem = false;
    (void)AttrUtils::GetBool(output_tensor_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_tensor_desc_mem);
    if (is_tensor_desc_mem) {
      MemoryBlock *out_desc_mem_block = ApplyOutDescMemory(node, i, ranges);
      GE_CHECK_NOTNULL(out_desc_mem_block);
    }
  }
  return SUCCESS;
}

Status BlockMemAssigner::AssignWorkSpaceMemoryWithReuse(const NodePtr &node, std::vector<int64_t> &ranges) {
  auto node_op_desc = node->GetOpDescBarePtr();
  std::vector<int64_t> temp;
  int64_t tatal_size = 0;
  GetNodeWorkSpaceSize(node, temp, tatal_size);

  std::vector<int64_t> workspace_type_list;
  const bool has_workspace_type_list_attr =
      ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);

  std::vector<int64_t> tvm_workspace_types;
  const bool has_tvm_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(node_op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);

  std::vector<int32_t> workspace_no_reuse_scope;
  const bool has_workspace_no_reuse_scope =
      ge::AttrUtils::GetListInt(node_op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);

  std::vector<bool> workspace_reuse_flag;
  GE_IF_BOOL_EXEC(!ge::AttrUtils::GetListBool(node_op_desc, kAttrNameWorkspaceReuseFlag, workspace_reuse_flag),
                  GELOGD("OP %s does not have workspace_reuse_flag attr", node_op_desc->GetNamePtr()));
  GELOGD("Assign memory node[%s], size [temp:%zu, tvm:%zu, list:%zu, no_reuse_scopes:%zu, reuse_flags:%zu.]",
         node_op_desc->GetNamePtr(), temp.size(), tvm_workspace_types.size(), workspace_type_list.size(),
         workspace_no_reuse_scope.size(), workspace_reuse_flag.size());

  if (((has_tvm_workspace_mem_type_attr) && (temp.size() != tvm_workspace_types.size())) ||
      ((has_workspace_type_list_attr) && (temp.size() != workspace_type_list.size()))) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces "
                       "num:%zu should be same, node_name:%s, check invalid", TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(),
                       tvm_workspace_types.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
                       workspace_type_list.size(), temp.size(), node->GetNamePtr());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces "
           "num:%zu should be same, node_name:%s, check invalid", TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(),
           tvm_workspace_types.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
           workspace_type_list.size(), temp.size(), node->GetNamePtr());
    return INTERNAL_ERROR;
  }
  bool need_gentask_atomic = false;
  (void)ge::AttrUtils::GetBool(node_op_desc, "need_gentask_atomic", need_gentask_atomic);
  auto atomic_workspace_info = node_op_desc->TryGetExtAttr(
      EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
  for (size_t i = 0UL; i < temp.size(); i++) {
    // workspace's life time is self node
    life_begin_ = node_op_desc->GetId();
    life_end_ = 0U;
    // fusion: other type's size not means malloc HBM memory
    bool workspace_skip_flag = false;
    if (has_tvm_workspace_mem_type_attr && ((tvm_workspace_types[i] == RT_MEMORY_L1)
                                            || (tvm_workspace_types[i] == kRtMemoryUB))) {
      GELOGI(
          "fusion:node[%s]workspace index[%zu] is not hbm type, add to zero_memory_list, workspace memory type [%ld]",
          node_op_desc->GetNamePtr(), i, tvm_workspace_types[i]);
      workspace_skip_flag = true;
    }
    if (temp[i] == 0 || workspace_skip_flag
        || (!need_gentask_atomic && IsAtomicWorkSpace(static_cast<int64_t>(i), atomic_workspace_info))) {
      zero_memory_list_.emplace_back(node.get(), kWorkspace, static_cast<uint32_t>(i), false);
      continue;
    }

    const bool session_scope_memory = (has_workspace_no_reuse_scope) && (i < workspace_no_reuse_scope.size()) &&
        (workspace_no_reuse_scope[i] == kSessionNoReuse);
    const bool is_p2p_memory =
        (has_workspace_type_list_attr) && (static_cast<uint64_t>(workspace_type_list[i]) == RT_MEMORY_P2P_DDR);
    uint64_t memory_type = GetWorkSpaceMemoryType(workspace_no_reuse_scope.size(), i, is_p2p_memory,
                                                  session_scope_memory, workspace_reuse_flag);
    GELOGI("%s's workspace mem_type:%lu, index:%zu.", node->GetNamePtr(), memory_type, i);
    ApplyMemoryParam param = {GetBlockSize(static_cast<size_t>(temp[i]), ranges, reuse_strategy_.use_range_),
                              static_cast<size_t>(temp[i]), static_cast<size_t>(temp[i]), kWorkspace,
                              static_cast<uint32_t>(i), is_op_reuse_mem_, false, memory_type, false};
    MemoryBlock *mem_block = ApplyMemory(node, workspace_reuse_flag, param);
    GE_CHECK_NOTNULL_EXEC(mem_block, continue);
    mem_block->is_reuse_zero_copy_ = (mem_block->is_reuse_zero_copy_) && (CanReuseZeroCopyBlock(node.get()));

    bool is_fixed_addr_prior = false;
    (void) ge::AttrUtils::GetBool(node_op_desc, ATTR_NAME_IS_FIXED_ADDR_PRIOR, is_fixed_addr_prior);
    mem_block->is_fixed_addr_prior_ = (mem_block->is_fixed_addr_prior_ || is_fixed_addr_prior);
    GELOGI("%s's workspace fixed addr prior:%d, index:%zu.", node->GetNamePtr(), mem_block->is_fixed_addr_prior_, i);

    ++(mem_block->ref_count_);
    CheckWorkspaceReuse(workspace_reuse_flag, i, GetStreamId(node_op_desc), mem_block, memory_type);
  }
  return SUCCESS;
}

void BlockMemAssigner::ParseGraphIoAllocMode() {
  // todo:临时方案，增加option控制静态子图hccl地址不支持刷新，待HCCL 1230正式方案上库后删除
  constexpr const char_t *kStaticModelAddrFixed = "ge.exec.static_model_addr_fixed";
  std::string is_addr_fixed_opt;
  (void)ge::GetContext().GetOption(kStaticModelAddrFixed, is_addr_fixed_opt);
  is_static_model_addr_fixed_ = !is_addr_fixed_opt.empty();

  if ((compute_graph_ == nullptr) || (compute_graph_->GetParentGraph() != nullptr)) {
    return;
  }

  std::string alloc_mode;
  (void)ge::GetContext().GetOption(OPTION_GRAPH_IO_MEM_ALLOC_MODE, alloc_mode);
  is_io_alloc_by_ge_in_run_graph_ = (alloc_mode == "ByGE");
  GELOGI("io_alloc_mode:%s, graph:%s.", (is_io_alloc_by_ge_in_run_graph_ ? "ByGE" : "ByApp"),
         compute_graph_->GetName().c_str());
}

Status BlockMemAssigner::InitIoReuseFlag() {
  GE_ASSERT_NOTNULL(compute_graph_);
  const auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  GE_ASSERT_NOTNULL(root_graph);

  // 动态shape静态子图输出内存复用
  if (root_graph->GetGraphUnknownFlag()) {
    const auto netoutput_node = compute_graph_->FindFirstNodeMatchType(NETOUTPUT);
    GE_ASSERT_NOTNULL(netoutput_node);
    const auto &netoutput_op_desc = netoutput_node->GetOpDesc();
    GE_ASSERT_NOTNULL(netoutput_op_desc);
    const size_t inputs_size = netoutput_op_desc->GetAllInputsSize();
    output_index_to_reuse_mem_flag_.resize(inputs_size, true);
    return SUCCESS;
  }
  ParseIoReuseMemOption();
  return SUCCESS;
}

void BlockMemAssigner::ParseIoReuseMemOption() {
  if ((compute_graph_ == nullptr) || (compute_graph_->GetParentGraph() != nullptr)) {
    return;
  }

  const auto &graph_option = GetThreadLocalContext().GetAllGraphOptions();
  const auto it_input_indexes = graph_option.find(OPTION_INPUT_REUSE_MEM_INDEXES);
  if (it_input_indexes != graph_option.end()) {
    GELOGI("[io_reuse_mem_option] option_input_indexes:%s", it_input_indexes->second.c_str());
    const size_t inputs_size = compute_graph_->GetInputNodes().size();
    input_index_to_reuse_mem_flag_.resize(inputs_size, false);
    std::vector<std::string> in_index_vec;
    SplitStringByComma(it_input_indexes->second, in_index_vec);
    for (const auto &str : in_index_vec) {
      const int32_t in_index = std::stoi(str);
      if ((in_index < 0) || (static_cast<size_t>(in_index) >= inputs_size)) {
        GELOGW("[Check][Option]Check failed because option(%s=%s) is invalid. Input_index must be in the "
               "range of [0, %zu)", OPTION_INPUT_REUSE_MEM_INDEXES, it_input_indexes->second.c_str(), inputs_size);
        continue;
      }
      input_index_to_reuse_mem_flag_[in_index] = true;
    }
  }

  const auto it_output_indexes = graph_option.find(OPTION_OUTPUT_REUSE_MEM_INDEXES);
  if (it_output_indexes != graph_option.end()) {
    GELOGI("[io_reuse_mem_option] option_output_indexes:%s", it_output_indexes->second.c_str());
    const auto netoutput_node = compute_graph_->FindFirstNodeMatchType(NETOUTPUT);
    GE_CHECK_NOTNULL_EXEC(netoutput_node, return);
    const auto &netoutput_op_desc = netoutput_node->GetOpDesc();
    GE_CHECK_NOTNULL_EXEC(netoutput_op_desc, return);
    const size_t inputs_size = netoutput_op_desc->GetAllInputsSize();
    output_index_to_reuse_mem_flag_.resize(inputs_size, false);
    std::vector<std::string> out_index_vec;
    SplitStringByComma(it_output_indexes->second, out_index_vec);
    for (const auto &str : out_index_vec) {
      const int32_t out_index = std::stoi(str);
      if ((out_index < 0) || (static_cast<size_t>(out_index) >= inputs_size)) {
        GELOGW("[Check][Option]Check failed because option(%s=%s) is invalid. Output_index must be in the "
               "range of [0, %zu)", OPTION_OUTPUT_REUSE_MEM_INDEXES, it_output_indexes->second.c_str(), inputs_size);
        continue;
      }
      output_index_to_reuse_mem_flag_[out_index] = true;
    }
  }

  return;
}

/// @ingroup domi
/// @brief traverse all nodes outputs and workspace in need, apply memory block considering memory reuse
/// @param [in/out] ranges memory size provided
/// @return Status result
Status BlockMemAssigner::AssignMemoryWithReuse(std::vector<int64_t> &ranges) {
  // init global flags
  std::string ge_disable_reuse_mem;
  (void)ge::GetContext().GetOption(OPTION_EXEC_DISABLE_REUSED_MEMORY, ge_disable_reuse_mem);
  GEEVENT("Reuse memory %s, memory_priority_mode is %s.", ge_disable_reuse_mem == "1" ? "close" : "open",
          memory_priority_mode_ ? "true" : "false");
  is_ge_reuse_mem_ = (ge_disable_reuse_mem != "1");

  const char_t *op_no_reuse_mem = nullptr;
  MM_SYS_GET_ENV(MM_ENV_OP_NO_REUSE_MEM, op_no_reuse_mem);
  if (op_no_reuse_mem != nullptr) {
    std::string op_no_reuse_mem_str = op_no_reuse_mem;
    CheckAndGetOpReuseEnv(op_no_reuse_mem_str, op_no_reuse_mem_vec_, op_reuse_env_valid_);
  }

  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  GE_ASSERT_NOTNULL(root_graph, "[Check][RootGraph]Root graph is nullptr, graph:%s.",
                    compute_graph_->GetName().c_str());
  root_unknown_shape_flag_ = root_graph->GetGraphUnknownFlag();

  (void) AttrUtils::GetBool(compute_graph_, ATTR_NAME_MEM_RELEASE_FIRST_REUSE_FIRST,
                            reuse_strategy_.reuse_first_release_);
  if (reuse_strategy_.reuse_first_release_) {
    GELOGI("The block usage strategy: first release, first reuse.");
  } else {
    GELOGI("The block usage strategy: first release, last reuse.");
  }

  // assign memory for every op
  for (NodePtr &n : compute_graph_->GetAllNodes()) {
    auto node_op_desc = n->GetOpDescBarePtr();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
    GE_ASSERT_SUCCESS(AssignOutputMemoryWithReuse(n, ranges), "node: %s(%s) assign output memory failed.",
                      n->GetNamePtr(), n->GetTypePtr());

    GE_ASSERT_SUCCESS(AssignWorkSpaceMemoryWithReuse(n, ranges), "node: %s(%s) assign workspace memory failed.",
                      n->GetNamePtr(), n->GetTypePtr());
    ReleaseInputNodeOutMemory(n);
  }

  for (const auto &block_pair : reusable_blocks_) {
    memory_stat_[block_pair.first].stream_count_ = block_pair.second.size();
  }

  GELOGD("Assigned memory blocks:");
  PrintMemBlock();

  GE_IF_BOOL_EXEC(is_ge_reuse_mem_, ReuseBlocksByLifeTime());
  AssignContinuousBlocks();
  GE_ASSERT_SUCCESS(ResizeMemoryBlocks(), "resize memory block failed");

  GELOGD("Memory blocks after resize:");
  PrintMemBlock();
  return SUCCESS;
}

void BlockMemAssigner::PrintMemBlock() {
  for (auto mem_block : memory_blocks_) {
    if (mem_block == nullptr) {
      continue;
    }
    mem_block->SetRefLifeTimeEnd();
    GELOGD("%s", mem_block->String().c_str());
  }
}

void BlockMemAssigner::CheckWorkspaceReuse(const std::vector<bool> &workspace_reuse_flag,
    uint32_t index, int64_t stream_id, MemoryBlock *const mem_block, uint64_t memory_type) {
  bool reuse_mem_flag =
      ((workspace_reuse_flag.size() > index) && (!workspace_reuse_flag[index])) ? false : true;
  if (reuse_mem_flag) {
    stream_workspace_blocks_[memory_type][stream_id].emplace_back(mem_block);
  }
}

void BlockMemAssigner::GetNodeWorkSpaceSize(const NodePtr &node, std::vector<int64_t> &workspace_memory,
                                            int64_t &total_size) const {
  if (node->GetOpDescBarePtr() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "param node opdesc is nullptr, check invalid.");
    GELOGE(FAILED, "[Check][Param] Op desc is null.");
    return;
  }
  std::vector<int64_t> workspace_byte_nums = node->GetOpDescBarePtr()->GetWorkspaceBytes();
  for (int64_t byte_size : workspace_byte_nums) {
    if ((byte_size < 0) && (byte_size != -1)) {
      // 后面校验range时会返回流程失败，这里只是补充日志
      GELOGE(FAILED, "[Check][Workspace]workspace_size:%ld is invalid, "
                     "maybe it is unknown shape node, Node_name:%s",
             byte_size, node->GetOpDescBarePtr()->GetNamePtr());
      REPORT_INNER_ERR_MSG("E19999", "workspace_size:%ld is invalid, "
                                   "maybe it is unknown shape node, Node_name:%s",
                         byte_size, node->GetOpDescBarePtr()->GetNamePtr());
      workspace_memory.emplace_back(byte_size);
      return;
    }
    byte_size = (byte_size < 0) ? 0 : byte_size;
    workspace_memory.emplace_back(byte_size);
    total_size += byte_size;
    GELOGI("node[%s] workspace_byte_nums:%zu, push back size:%ld", node->GetOpDescBarePtr()->GetNamePtr(),
           workspace_byte_nums.size(), byte_size);
  }
}

// asending order
static bool CompareBlockIndex(const MemoryBlock *const left, const MemoryBlock *const right) {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  if (left->input_index_ < right->input_index_) {
    return true;
  }
  return false;
}

/// @ingroup domi
/// @brief order blocks by continuous input index
/// @param [in] blocks need be processed
/// @param [in] input blocks need continuous
/// @param [out] blocks after continuous order
/// @param [in/out] blocks ordered
/// @param [in] input or output
void ReAssignContinuousBlocks(const std::vector<MemoryBlock *> &org_blocks,
                              const std::map<MemoryBlock *, uint32_t> &block_map,
                              std::vector<MemoryBlock *> &dest_blocks, std::vector<MemoryBlock *> &continuous_blocks,
                              const std::string &type) {
  for (auto &memory_block : org_blocks) {
    if (memory_block == nullptr || memory_block->child_block_) {
      continue;
    }
    if (block_map.find(memory_block) != block_map.end()) {
      continue;
    }
    dest_blocks.emplace_back(memory_block);
  }

  // add continuous block
  std::sort(continuous_blocks.begin(), continuous_blocks.end(), CompareBlockIndex);
  size_t count = 0UL;
  for (auto &memory_block : continuous_blocks) {
    GE_IF_BOOL_EXEC(memory_block == nullptr, continue);

    GELOGI("Block continuous %s index:%d", type.c_str(), memory_block->input_index_);
    count++;
    if (count == 1U) {
      memory_block->SetFirstContinuousBlock();
    }
    if (count == continuous_blocks.size()) {
      memory_block->SetLastContinuousBlock();
    }
    dest_blocks.emplace_back(memory_block);
  }
}

void BlockMemAssigner::AssignContinuousBlocks() {
  for (const auto &block_map : node_continuous_input_blocks_) {
    std::vector<MemoryBlock *> dest_memory_blocks;
    std::map<MemoryBlock *, uint32_t> continuous_block_map;
    std::vector<MemoryBlock *> continuous_blocks;
    const auto it = node_continuous_input_counts_.find(block_map.first);
    GE_IF_BOOL_EXEC(it == node_continuous_input_counts_.end(), continue);
    const bool size_independent = SizeIndependentOfBatch(it->second.first);
    GELOGI("Node %ld continuous input block count:%zu input count:%u, size_independent:%d", block_map.first,
           block_map.second.size(), it->second.second, static_cast<int32_t>(size_independent));
    GE_IF_BOOL_EXEC(it->second.second != block_map.second.size(), continue);

    for (auto &iter : block_map.second) {
      if (iter.second != nullptr) {
        iter.second->need_same_offset_in_batch_ = size_independent;
        continuous_block_map[iter.second] = iter.first;
        iter.second->input_index_ = iter.first;
        continuous_blocks.emplace_back(iter.second);
      }
    }
    if (continuous_block_map.size() != continuous_blocks.size()) {
      GELOGW("Node %ld continuous input map size:%zu vector size:%zu", block_map.first,
             continuous_block_map.size(), continuous_blocks.size());
      continue;
    }
    ReAssignContinuousBlocks(memory_blocks_, continuous_block_map, dest_memory_blocks, continuous_blocks, "input");
    memory_blocks_.swap(dest_memory_blocks);
  }
}

struct CompareLifeInterval {
  explicit CompareLifeInterval(const ReuseStrategy &reuse_strategy) : reuse_strategy_(reuse_strategy) {}

  bool operator() (MemoryBlock *const left, MemoryBlock *const right) const {
    if ((left != nullptr) && (right != nullptr)) {
      auto left_size = left->AlignSize();
      auto right_size = right->AlignSize();
      if (left->GetContinuousFlag()) {
        auto it = std::max_element(std::begin(left->NoAlignSizeList()), std::end(left->NoAlignSizeList()));
        if (it != left->NoAlignSizeList().end()) {
          left_size = *it;
        }
      }

      if (right->GetContinuousFlag()) {
        auto it = std::max_element(std::begin(right->NoAlignSizeList()), std::end(right->NoAlignSizeList()));
        if (it != right->NoAlignSizeList().end()) {
          right_size = *it;
        }
      }

      if (left_size == right_size) {
        if (!reuse_strategy_.ascending_sort_) {
          return (left->GetLifeBegin(true) > right->GetLifeBegin(true));
        }
        if (left->NodeTypeIndexList().size() == right->NodeTypeIndexList().size()) {
          return (left->GetLifeBegin(true) < right->GetLifeBegin(true));
        } else {
          return (left->NodeTypeIndexList().size() < right->NodeTypeIndexList().size());
        }
      } else {
        return (left_size > right_size);
      }
    }
    return false;
  }
  ReuseStrategy reuse_strategy_;
};

void BlockMemAssigner::ReuseBlocksByLifeTime() {
  if (!NeedLevel2Reuse()) {
    return;
  }
  CompareLifeInterval cmp(reuse_strategy_);
  std::sort(memory_blocks_.begin(), memory_blocks_.end(), cmp);
  for (size_t i = 0UL; i < memory_blocks_.size(); ++i) {
    auto parent = memory_blocks_[i];
    GE_IF_BOOL_EXEC((parent == nullptr || parent->child_block_), continue);
    for (size_t j = i + 1; j < memory_blocks_.size(); ++j) {
      auto child = memory_blocks_[j];
      GE_IF_BOOL_EXEC((child == nullptr), continue);

      // If node is before atomic_addr_clean node, the continus memory can't be reused, its out put will be cleared.
      if (!child->NodeTypeIndexList().empty() && parent->GetContinuousFlag()) {
        auto node = child->NodeTypeIndexList()[0].node_;
        bool before_atomic_clean = ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)
            || (node->GetOpDescBarePtr()->GetId() < GetAtomicAddrCleanId()));
        GE_IF_BOOL_EXEC(before_atomic_clean, continue);
      }
      std::vector<MemoryBlock *> clone_blocks;
      parent->AddLifeReuseBlock(this, child, clone_blocks, 0, in_stream_edges_);
      GE_IF_BOOL_EXEC(clone_blocks.empty(), continue);

      // insert after this child block
      memory_blocks_.insert(memory_blocks_.cbegin() + j + 1, clone_blocks.cbegin(), clone_blocks.cend());
      blocks_store_.insert(blocks_store_.cend(), clone_blocks.cbegin(), clone_blocks.cend());
      // if clone block's align size is less than nex block's size need sort again
      size_t min_block_align_size = parent->AlignSize();
      for (const auto block : clone_blocks) {
        min_block_align_size =
            ((block != nullptr) && (block->AlignSize() < min_block_align_size)) ? block->AlignSize() :
            min_block_align_size;
      }
      const size_t next_index = j + 1 + clone_blocks.size();
      // Sorting will increase processing time, but it can obtain smaller memory
      const bool need_sort_again = ((next_index < memory_blocks_.size()) && (memory_blocks_[next_index] != nullptr)
          && (min_block_align_size < memory_blocks_[next_index]->AlignSize())) || memory_priority_mode_;
      if (need_sort_again) {
        std::sort(memory_blocks_.begin() + j + 1, memory_blocks_.end(), cmp);
      }
    }
  }
}

Status AddBlockMemOffset(std::map<uint64_t, size_t> &mem_offsets, MemoryBlock &block) {
  auto it = mem_offsets.find(block.memory_type_);
  if (it == mem_offsets.end()) {
    auto result = mem_offsets.insert(std::pair<int64_t, size_t>(block.memory_type_, block.memory_type_logic_base_));
    GE_ASSERT_TRUE(result.second);
    it = result.first;
  }
  GE_ASSERT_TRUE(it != mem_offsets.end());
  auto &mem_offset = it->second;
  block.Resize();
  GE_ASSERT_SUCCESS(block.SetHeadOffset(mem_offset));
  mem_offset += block.Size();
  block.SetTailOffset(mem_offset - 1);
  return SUCCESS;
}

/// @ingroup domi_omg
/// @brief traverse memory size, resize, calculate offset
/// @param [in&out] memory_blocks_ memory block, after calculating offset
/// |-not dynamic batch block-||-dynamic batch block batch1|    |-zero copy block-|
/// |-not dynamic batch block-||-dynamic batch block batch2----||-zero copy block-|
/// |-not dynamic batch block-||-dynamic batch block batch3--|  |-zero copy block-|
/// 和里理论值有偏差原因，batch内外没有复用，batch间有对齐策略
Status BlockMemAssigner::ResizeMemoryBlocks() {
  DynamicBatchMemAssigner dynamic_batch_mem_assigner(reuse_strategy_, memory_blocks_, blocks_store_);
  dynamic_batch_mem_assigner.ResizeDynamicBatchBlocks();
  for (auto &memory_block : memory_blocks_) {
    if (memory_block == nullptr || memory_block->child_block_ || memory_block->is_zero_copy_) {
      continue;
    }
    GE_ASSERT_SUCCESS(AddBlockMemOffset(mem_offsets_, *memory_block));
  }

  for (auto &it : memory_stat_) {
    it.second.theory_min_memory_size_ += it.second.theory_no_reuse_memory_size_;
  }

  for (const auto &it : mem_offsets_) {
    GELOGI("Reuse result:%s memory type:%lu mem_offset exclude zero_copy_memory:%zu.",
           (compute_graph_ != nullptr) ? compute_graph_->GetName().c_str() : "", it.first, it.second);
  }
  return SUCCESS;
}

/// @ingroup domi
/// @brief given NodeTypeIndex, set offset in Op's OpDef
/// @param [in&out] node_type_index <node, memory type, id>
/// @param [in] offset offset to be set
/// @param [in] size memory size
/// @param [in] real_size memory size in need
/// @return Status result
void BlockMemAssigner::SetOffsetSize(const NodeTypeIndex &node_type, const MemoryBlock &block,
                                     size_t real_size, size_t no_align_size, int32_t child_block_level) const {
  GE_CHECK_NOTNULL_EXEC(node_type.node_, return);
  auto op_desc = node_type.node_->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_EXEC(op_desc, return);
  std::string graph_name = compute_graph_->GetName();
  std::vector<int64_t> memorys_type;
  int64_t offset = block.HeadOffset();
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memorys_type);
  if (node_type.mem_type_ == kOutput) {
    std::vector<int64_t> output_list = op_desc->GetOutputOffset();
    for (auto i = static_cast<uint32_t>(output_list.size()); i < node_type.index_ + 1; i++) {
      output_list.emplace_back(kInvalidOffset);
    }
    if (output_list.empty()) {
      GELOGW("Empty output");
      return;
    }

    static const std::set<std::string> kSetOffsetTypes = { DATA_TYPE, REFDATA, AIPP_DATA_TYPE, MULTISHAPE, NETOUTPUT };
    if ((kSetOffsetTypes.count(op_desc->GetTypePtr()) > 0) && !IsKnownSubgraphData(node_type.node_)) {
      if ((output_list[node_type.index_] == kInvalidOffset) || (output_list[node_type.index_] < offset)) {
        output_list.at(node_type.index_) = offset;
      }
    } else {
      // fusion: keep the original other type offset value from op_desc
      bool set_out_offset = (!has_mem_type_attr) ||
        (memorys_type.size() > node_type.index_ && memorys_type[node_type.index_] != RT_MEMORY_L1 &&
                             (memorys_type[node_type.index_] != kRtMemoryUB));
      if (set_out_offset) {
        output_list.at(node_type.index_) = offset;
      }
    }
    op_desc->SetOutputOffset(output_list);
  } else if ((node_type.mem_type_ == kWorkspace) && (!node_type.is_subgraph_workspace_)) {
    std::vector<int64_t> workspace_list;
    workspace_list = op_desc->GetWorkspace();
    for (auto i = static_cast<uint32_t>(workspace_list.size()); i < node_type.index_ + 1; i++) {
      workspace_list.emplace_back(kInvalidOffset);
    }
    std::vector<int64_t> workspace_mem_type;
    bool has_workspace_mem_type = ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, workspace_mem_type);
    // fusion: keep the original other type offset value from op_desc
    bool set_workspace_offset = (!has_workspace_mem_type) ||
      (workspace_mem_type.size() > node_type.index_ && workspace_mem_type[node_type.index_] != RT_MEMORY_L1 &&
                                 (workspace_mem_type[node_type.index_] != kRtMemoryUB));
    if (set_workspace_offset) {
      workspace_list.at(node_type.index_) = offset;
    }
    op_desc->SetWorkspace(workspace_list);
  } else if (node_type.mem_type_ == kOutputDesc) {
    auto tensor = op_desc->MutableOutputDesc(node_type.index_);
    GE_IF_BOOL_EXEC(tensor != nullptr,
                    (void)AttrUtils::SetInt(tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, offset));
  }
  GELOGI(
      "[IMAS]Set %s name[%s] optype[%s] %s[%u] offset to [%ld] streamid[%s] memtype[%lu] size[%zu] realsize[%zu] "
      "noalignsize[%zu] life time begin[%s] life time end[%s] child[%d:%d:%d:%d:%d] isref[%d] batch[%s], "
      "block_type[%s]",
      MemReuseUtils::GetGraphNameId(compute_graph_.get()).c_str(), op_desc->GetName().substr(0, kMaxLogLen).c_str(),
      node_type.node_->GetTypePtr(), node_type.GetMemType().c_str(), node_type.index_, offset,
      GetStreamIdDesc(op_desc).c_str(), block.memory_type_, block.Size(), real_size, no_align_size,
      node_type.GetLifeBeginDesc().c_str(), node_type.GetLifeEndDesc().c_str(), child_block_level, block.reuse_mem_,
      block.GetContinuousFlag(), block.is_zero_copy_, block.same_stream_, node_type.ref_input_,
      block.batch_label_.c_str(),
      NodeMemAttrUtils::GetAttrStr(node_type).c_str());

  if ((!node_type.ref_input_) && (real_size != 0U)) {
    size_t life_end = node_type.GetLifeEnd().back();
    life_end = (life_end == kDefaultLifeTime) ? kMaxLifeTime : life_end;  // trans default life end to max life time
    CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(compute_graph_, node_type.node_->GetOpDesc(), real_size,
                                             node_type.GetLifeBegin(), life_end);
  }
}

void BlockMemAssigner::SetBlockOpMemOffset(const MemoryBlock *const block,
                                           int32_t child_block_level, bool &is_fixed_addr_prior) const {
  if (block == nullptr) {
    return;
  }
  size_t index = 0UL;
  size_t real_size = 0UL;
  size_t no_align_size = 0UL;
  auto real_size_list_size = block->RealSizeList().size();
  for (const NodeTypeIndex &node_type_index : block->NodeTypeIndexList()) {
    if (index < real_size_list_size) {
      real_size = block->RealSizeList()[index];
      no_align_size = block->NoAlignSizeList()[index];
    }
    SetOffsetSize(node_type_index, *block, real_size, no_align_size, child_block_level);
    index++;
  }

  is_fixed_addr_prior = (is_fixed_addr_prior || block->is_fixed_addr_prior_);

  child_block_level++;
  if (!block->ChildSubGraphBlockList().empty()) {
    for (MemoryBlock *child_block : block->ChildSubGraphBlockList()) {
      SetBlockOpMemOffset(child_block, child_block_level, is_fixed_addr_prior);
    }
  }

  for (auto &blocks : block->BatchBlockList()) {
    for (auto child_block : blocks.second) {
      SetBlockOpMemOffset(child_block, child_block_level, is_fixed_addr_prior);
    }
  }

  for (MemoryBlock *child_block : block->ChildBlockList()) {
    SetBlockOpMemOffset(child_block, child_block_level, is_fixed_addr_prior);
  }
}

void BlockMemAssigner::SetOpMemOffset(bool is_zero_copy) const {
  if (!is_zero_copy) {
    for (const auto &attr : bool_attr_) {
      if (!ge::AttrUtils::SetBool(attr.ptr_, attr.name_, attr.value_)) {
        GELOGW("Set %s input[%d] %s to %s failed.",
               attr.desc_->GetNamePtr(), attr.index_, attr.name_.c_str(), attr.value_ ? "true" : "false");
        continue;
      }
      GELOGD("Set %s input[%d] %s to %s success.",
             attr.desc_->GetNamePtr(), attr.index_, attr.name_.c_str(), attr.value_ ? "true" : "false");
    }

    for (const auto &attr : int_attr_) {
      if (!ge::AttrUtils::SetInt(attr.ptr_, attr.name_, attr.value_)) {
        GELOGW("Set %s attr %s to %ld failed.", attr.desc_->GetNamePtr(), attr.name_.c_str(), attr.value_);
        continue;
      }
      GELOGD("Set %s attr %s to %ld success.", attr.desc_->GetNamePtr(), attr.name_.c_str(), attr.value_);
    }
  }
  for (MemoryBlock *memory_block : memory_blocks_) {
    if (memory_block == nullptr || memory_block->child_block_) {
      continue;
    }

    if ((is_zero_copy && !memory_block->is_zero_copy_) || (!is_zero_copy && memory_block->is_zero_copy_)) {
      continue;
    }

    bool is_fixed_addr_prior = false;
    SetBlockOpMemOffset(memory_block, 0, is_fixed_addr_prior);
    memory_block->is_fixed_addr_prior_ = (memory_block->is_fixed_addr_prior_ || is_fixed_addr_prior);
  }

  const auto var_mng = VarManager::Instance(compute_graph_->GetSessionID());
  if (!is_zero_copy) {
    for (const NodeTypeIndex &node_type_index : zero_memory_list_) {
      if (var_mng->IsVarExist(VarMemAssignUtil::GetNameForVarManager(node_type_index.node_->GetOpDesc()))) {
        continue;
      }
      MemoryBlock block(reuse_strategy_, 0, 0);
      SetOffsetSize(node_type_index, block, 0UL, 0UL, 0);
    }
  }
  SetOffsetForContinuousMem();
}

void BlockMemAssigner::SetOffsetForContinuousMem() const {
  for (const auto &continuous_mem : continuous_mem_mng_.GetAllContinuousMem()) {
    const auto &blocks = continuous_mem.GetBlocks();
    if (!blocks.empty()) {
      const auto block = blocks.front();
      auto offset = block->HeadOffset();
      for (size_t i = 0U; i < continuous_mem.GetContinuousNodeOut().size(); ++i) {
        const auto &node_index = continuous_mem.GetContinuousNodeOut().at(i);
        auto op_desc = node_index.node_ptr_->GetOpDescBarePtr();
        auto out_offsets = op_desc->GetOutputOffset();
        while (out_offsets.size() < node_index.index_ + 1U) {
          out_offsets.emplace_back(kInvalidOffset);
        }
        out_offsets[node_index.index_] = offset;
        op_desc->SetOutputOffset(out_offsets);
        const auto align_size = continuous_mem.GetAlignedSizes().at(i);
        GELOGI("[ContinuousMem][IMAS]Continuous input : Set %s name[%s] optype[%s] output[%d] offset to [%ld] "
            "stream_id[%ld] memtype[%ld] size[%zu] realsize[%ld] nopadding[%d], block_type[%s]",
            MemReuseUtils::GetGraphNameId(compute_graph_.get()).c_str(),
            op_desc->GetName().substr(0, kMaxLogLen).c_str(),
            op_desc->GetType().c_str(), node_index.index_, offset,
            op_desc->GetStreamId(), block->memory_type_, 0UL, align_size, false,
            NodeMemAttrUtils::GetAttrStr({node_index.node_ptr_, kOutput, node_index.index_}).c_str());
        offset += align_size;
      }
    }
  }
}

void BlockMemAssigner::SetOpMemOffset(const std::vector<MemoryBlock *> &zero_copy_blocks) const {
  for (const auto memory_block : zero_copy_blocks) {
    if ((memory_block != nullptr) && (!memory_block->child_block_)) {
      bool is_fixed_addr_prior = false;
      SetBlockOpMemOffset(memory_block, 0, is_fixed_addr_prior);
      memory_block->is_fixed_addr_prior_ = (memory_block->is_fixed_addr_prior_ || is_fixed_addr_prior);
    }
  }
}

Status BlockMemAssigner::Assign() {
  return SUCCESS;
}

bool BlockMemAssigner::CheckIsZeroMemNodeType(const std::string &node_type) const {
  return (node_type == VARIABLE) || (node_type == CONSTANT) || (node_type == MULTISHAPE) || (node_type == CONSTANTOP)
         || (node_type == HVDWAIT) || (node_type == FILECONSTANT) || (node_type == CONSTPLACEHOLDER);
}

bool BlockMemAssigner::CheckIsZeroMemNodeOutputIndex(const NodePtr &n, uint32_t index) const {
  const auto op_desc = n->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto output_tensor_desc = op_desc->MutableOutputDesc(index);
  if (output_tensor_desc == nullptr) {
    GELOGW("op[%s] null output_tensor_desc, index[%u]", op_desc->GetName().c_str(), index);
    return false;
  }
  int32_t tensor_type = 0;
  const bool ret = ge::AttrUtils::GetInt(output_tensor_desc, ATTR_NAME_TENSOR_MEMORY_SCOPE, tensor_type);
  if (ret && tensor_type == kOutputMemoryGlobalType) {
    GELOGD("node[%s] output[%u] is zero memory", n->GetName().c_str(), index);
    return true;
  }
  return false;
}

uint64_t BlockMemAssigner::GetWorkSpaceMemoryType(const size_t no_reuse_scope_size, const size_t index,
                                                  const bool is_p2p_memory, const bool session_scope_memory,
                                                  std::vector<bool> &workspace_reuse_flag) const {
  if (is_p2p_memory) {
    return RT_MEMORY_P2P_DDR;
  }

  if (session_scope_memory) {
    if (workspace_reuse_flag.empty()) {
      workspace_reuse_flag.assign(no_reuse_scope_size, true);
    }
    workspace_reuse_flag[index] = false;
    return kSessionScopeMemory | RT_MEMORY_HBM;
  }

  return RT_MEMORY_HBM;
}
}  // namespace ge
