/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_layout_conflict_util.h"
#include <atomic>
#include <stack>
#include "runtime/rt.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_type_utils.h"
#include "checker/checker_log.h"
#include "exec_runtime/execution_runtime_utils.h"
#include "common/memory/mem_type_utils.h"
#include "graph/manager/graph_var_manager.h"

namespace ge {
namespace {
const char *const kRefreshable = "1";
const std::set<std::string> kMemcpyNodeTypes{IDENTITY, MEMCPYASYNC, MEMCPYADDRASYNC};
const std::string kOffline = "offline";
constexpr uint8_t kThenBranchIndex = 0U;
constexpr uint8_t kElseBranchIndex = 1U;
constexpr uint8_t kBodyBranchIndex = 1U;

std::vector<OutDataAnchorPtr> GetAllInAnchorsPeer(const NodePtr &node) {
  std::vector<OutDataAnchorPtr> peer_out_anchors;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    if ((in_anchor != nullptr) && (in_anchor->GetPeerOutAnchor() != nullptr)) {
      peer_out_anchors.emplace_back(in_anchor->GetPeerOutAnchor());
    }
  }
  return peer_out_anchors;
};

std::vector<OutDataAnchorPtr> GetAllOutAnchors(const NodePtr &node) {
  std::vector<OutDataAnchorPtr> out_anchors;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    out_anchors.emplace_back(out_anchor);
  }
  return out_anchors;
};

// 如果out_anchor引用某个in_anchor, 再获取in_anchor的对端
OutDataAnchorPtr GetRealOutAnchor(const OutDataAnchorPtr &out_anchor) {
  auto real_out_anchor = out_anchor;
  int32_t reuse_in_index = 0;
  while (GraphUtils::IsRefFromInput(real_out_anchor, reuse_in_index)) {
    const auto in_node = real_out_anchor->GetOwnerNodeBarePtr();
    if ((in_node == nullptr) || (in_node->GetInDataAnchor(reuse_in_index) == nullptr) ||
        (in_node->GetInDataAnchor(reuse_in_index)->GetPeerOutAnchor() == nullptr)) {
      break;
    }
    real_out_anchor = in_node->GetInDataAnchor(reuse_in_index)->GetPeerOutAnchor();
  }
  return real_out_anchor;
}

// 找到第一个相同的index，如果没有找到，返回size_t最大值
std::pair<size_t, size_t> GetFirstSameIndex(const std::vector<OutDataAnchorPtr> &a_in_peer_anchors,
                                            const std::vector<OutDataAnchorPtr> &b_in_peer_anchors) {
  for (size_t i = 0U; i < a_in_peer_anchors.size(); ++i) {
    for (size_t j = 0U; j < b_in_peer_anchors.size(); ++j) {
      if (b_in_peer_anchors[j] == a_in_peer_anchors[i]) {
        return {i, j};
      }
    }
  }
  return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
}

// 如果sub_vec是vec的子集，返回true.
bool IsSubVector(const std::vector<OutDataAnchorPtr> &vec, const std::vector<OutDataAnchorPtr> &sub_vec) {
  size_t i = 0U;
  size_t j = GetFirstSameIndex(vec, sub_vec).first;
  for (; i < sub_vec.size() && j < vec.size(); ++i, ++j) {
    if (sub_vec[i] != vec[j]) {
      return false;
    }
  }
  return i == sub_vec.size();
}

bool IsSameOrPartialSame(const std::vector<OutDataAnchorPtr> &lh, const std::vector<OutDataAnchorPtr> &rh) {
  const auto same_index_pair = GetFirstSameIndex(lh, rh);
  size_t lh_index = same_index_pair.first;
  size_t rh_index = same_index_pair.second;
  if ((lh_index != 0) && (rh_index != 0U)) {
    return false;
  }
  for (; rh_index < rh.size() && lh_index < lh.size(); ++rh_index, ++lh_index) {
    if (rh[rh_index] != lh[lh_index]) {
      return false;
    }
  }
  if ((lh_index != lh.size()) && (rh_index != rh.size())) {
    return false;
  }
  /*
   * 存在交叉
   * node_a所有输出anchor     ： anchor1 anchor2
   * node_b所有输入anchor的对端：         anchor2 anchor1
   */
  if (lh_index != lh.size()) {
    for (; lh_index < lh.size(); ++lh_index) {
      if (std::find(rh.begin(), rh.end(), lh[lh_index]) != rh.end()) {
        return false;
      }
    }
  }
  if (rh_index < rh.size()) {
    for (; rh_index < rh.size(); ++rh_index) {
      if (std::find(lh.begin(), lh.end(), rh[rh_index]) != lh.end()) {
        return false;
      }
    }
  }
  return true;
}

bool IsRefFromRefData(const Node &node) {
  return node.GetOpDesc()->HasAttr(REF_VAR_SRC_VAR_NAME);
}
}

using SubGraphSolveConflictCall =
  std::function<Status(const NodePtr &ctrl_node, std::vector<InDataAnchorPtr> &in_data_anchors)>;
std::map<std::string, SubGraphSolveConflictCall> MemLayoutConflictUtil::check_subgraph_conflict_call = {
  {IF, MemLayoutConflictUtil::CheckIfConflict},
  {STATELESSIF, MemLayoutConflictUtil::CheckIfConflict},
  {CASE, MemLayoutConflictUtil::CheckCaseConflict},
  {STATELESSCASE, MemLayoutConflictUtil::CheckCaseConflict},
  {WHILE, MemLayoutConflictUtil::CheckWhileConflict},
  {STATELESSWHILE, MemLayoutConflictUtil::CheckWhileConflict}
};

bool MemLayoutConflictUtil::IsSkipInsert(const InDataAnchorPtr &in_anchor) {
  return MemLayoutConflictUtil::IsUnknownShape(in_anchor)
         || MemLayoutConflictUtil::PeerIsIdentityOrMemcpyAsync(in_anchor);
}

bool MemLayoutConflictUtil::IsSkipInsert(const OutDataAnchorPtr &out_anchor) {
  return MemLayoutConflictUtil::IsUnknownShape(out_anchor)
         || MemLayoutConflictUtil::PeerIsIdentityOrMemcpyAsync(out_anchor);
}

bool MemLayoutConflictUtil::IsUnknownShape(const InDataAnchorPtr &in_anchor) {
  GE_ASSERT_NOTNULL(in_anchor);
  GE_ASSERT_NOTNULL(in_anchor->GetOwnerNodeBarePtr());
  const auto op_desc = in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const bool unknown_shape = op_desc->GetInputDesc(in_anchor->GetIdx()).GetShape().IsUnknownShape();
  if (unknown_shape) {
    GELOGI("[MemConflict] unknown shape, skip insert identity, node[%s], input[%d].",
           op_desc->GetType().c_str(), in_anchor->GetIdx());
  }
  return unknown_shape;
}

bool MemLayoutConflictUtil::IsUnknownShape(const OutDataAnchorPtr &out_anchor) {
  GE_ASSERT_NOTNULL(out_anchor);
  GE_ASSERT_NOTNULL(out_anchor->GetOwnerNodeBarePtr());
  const auto op_desc = out_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const bool unknown_shape = op_desc->GetOutputDesc(out_anchor->GetIdx()).GetShape().IsUnknownShape();
  if (unknown_shape) {
    GELOGI("[MemConflict] unknown shape, skip insert identity, node[%s], output[%d].",
           op_desc->GetType().c_str(), out_anchor->GetIdx());
  }
  return unknown_shape;
}

bool MemLayoutConflictUtil::PeerIsIdentityOrMemcpyAsync(const InDataAnchorPtr &in_anchor) {
  GE_ASSERT_NOTNULL(in_anchor);
  GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor());
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  const auto &peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(peer_node);
  if ((kMemcpyNodeTypes.count(peer_node->GetType()) != 0U) && (peer_out_anchor->GetPeerInDataNodesSize() == 1U)) {
    GELOGI("[MemConflict] peer out is already %s(%s), skip insert identity.",
           peer_node->GetType().c_str(), peer_node->GetNamePtr());
    return true;
  }
  return false;
}

bool MemLayoutConflictUtil::PeerIsIdentityOrMemcpyAsync(const OutDataAnchorPtr &out_anchor) {
  GE_ASSERT_NOTNULL(out_anchor);
  if (out_anchor->GetPeerInDataAnchors().size() == 1U) {
    const auto peer_node = out_anchor->GetPeerInDataAnchors().at(0U)->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    if (kMemcpyNodeTypes.count(peer_node->GetType()) != 0U) {
      GELOGI("[MemConflict] peer in node %s, node_type: %s, skip insert identity.", peer_node->GetNamePtr(),
             peer_node->GetType().c_str());
      return true;
    }
  }
  return false;
}

bool MemLayoutConflictUtil::HasRefVarName(const OutDataAnchorPtr &out_data_anchor, std::string &ref_var_src_var_name) {
  return HasRefVarName(out_data_anchor.get(), ref_var_src_var_name);
}

bool MemLayoutConflictUtil::HasRefVarName(const OutDataAnchor *out_data_anchor, std::string &ref_var_src_var_name) {
  const auto owner_node = out_data_anchor->GetOwnerNode();
  const auto out_desc = owner_node->GetOpDesc()->GetOutputDescPtr(static_cast<uint32_t>(out_data_anchor->GetIdx()));
  GE_ASSERT_NOTNULL(out_desc);
  return ge::AttrUtils::GetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
}

void CopyAttr(const NodePtr &src_node, const NodePtr &dest_node, const OpDescPtr &identity_op_desc) {
  std::string stream_label;
  if (AttrUtils::GetStr(dest_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    (void)AttrUtils::SetStr(identity_op_desc, ATTR_NAME_STREAM_LABEL, stream_label);
    GELOGD("[MemConflict] Node %s set stream label: %s", identity_op_desc->GetName().c_str(), stream_label.c_str());
  }

  bool rts_label_node = false;
  if (AttrUtils::GetBool(dest_node->GetOpDesc(), ATTR_NAME_RTS_LABEL_NODE, rts_label_node)) {
    (void)AttrUtils::SetBool(identity_op_desc, ATTR_NAME_RTS_LABEL_NODE, rts_label_node);
    GELOGD("[MemConflict] Node %s set rts label node attribute", identity_op_desc->GetName().c_str());
  }

  std::string batch_label;
  if ((AttrUtils::GetStr(src_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label)) && (!batch_label.empty())) {
    (void)AttrUtils::SetStr(identity_op_desc, ATTR_NAME_BATCH_LABEL, batch_label);
    GELOGD("[MemConflict] Node %s set batch label attribute", identity_op_desc->GetName().c_str());
  }

  bool labeled_input = false;
  (void)ge::AttrUtils::GetBool(dest_node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, labeled_input);
  if (labeled_input) {
    (void)ge::AttrUtils::SetBool(dest_node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, false);
    (void)ge::AttrUtils::SetBool(identity_op_desc, ATTR_NAME_NODE_CONNECT_INPUT, true);
    GELOGD("[MemConflict] Node %s set connect input attribute", identity_op_desc->GetName().c_str());
  }

  bool labeled_output = false;
  (void)ge::AttrUtils::GetBool(src_node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, labeled_output);
  if (labeled_output) {
    (void)ge::AttrUtils::SetBool(src_node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, false);
    (void)ge::AttrUtils::SetBool(identity_op_desc, ATTR_NAME_NODE_CONNECT_OUTPUT, true);
    GELOGD("[MemConflict] Node %s set connect output attribute", identity_op_desc->GetName().c_str());
  }
}

void SetNotDeleteAttr(const OpDescPtr &identity_op_desc) {
  (void)AttrUtils::SetBool(identity_op_desc, ATTR_NO_NEED_CONSTANT_FOLDING, false);
  (void)AttrUtils::SetBool(identity_op_desc, ATTR_NAME_CANNOT_BE_DELETED, true);
}

Status MemLayoutConflictUtil::CreateIdentityOpDesc(const std::vector<InDataAnchorPtr> &in_data_anchors,
                                                   OpDescPtr &identity_op) {
  GE_ASSERT_TRUE(!in_data_anchors.empty());
  GE_ASSERT_NOTNULL(in_data_anchors[0]);
  GE_ASSERT_NOTNULL(in_data_anchors[0]->GetOwnerNodeBarePtr());

  static std::atomic<size_t> id(0U);
  std::string node_name = "identity";
  node_name += "_" + std::to_string(in_data_anchors[0]->GetIdx()) + "_"
               + in_data_anchors[0]->GetOwnerNodeBarePtr()->GetName() + "_MemLayoutConflict_"
               + to_string(id.fetch_add(1U));

  OpDescBuilder op_desc_builder(node_name, IDENTITY);
  size_t index = 0U;
  for (const auto &in_data_anchor : in_data_anchors) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    const auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    const auto src_node = peer_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(src_node);
    GE_ASSERT_NOTNULL(src_node->GetOpDescBarePtr());

    auto tensor_desc = src_node->GetOpDescBarePtr()->GetOutputDesc(peer_out_anchor->GetIdx());
    TensorUtils::SetReuseInput(tensor_desc, false);
    tensor_desc.DelAttr(REF_VAR_SRC_VAR_NAME);

    if (in_data_anchors.size() == 1) {
      op_desc_builder.AddInput("x", tensor_desc).AddOutput("y", tensor_desc).Build();
    } else {
      op_desc_builder.AddInput("x" + std::to_string(index), tensor_desc)
          .AddOutput("y" + std::to_string(index), tensor_desc);
    }

    ++index;
  }

  identity_op = op_desc_builder.Build();
  GE_ASSERT_NOTNULL(identity_op);

  for (const auto &in_data_anchor : in_data_anchors) {
    const auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    CopyAttr(peer_out_anchor->GetOwnerNode(), in_data_anchor->GetOwnerNode(), identity_op);
  }
  SetNotDeleteAttr(identity_op);
  return SUCCESS;
}

Status MemLayoutConflictUtil::UpdateIsInputConstForNetoutput(const std::vector<InDataAnchorPtr> &in_data_anchors,
                                                    const NodePtr &identity_node) {
  for (const auto &in_data_anchor : in_data_anchors) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    const auto dst_node = in_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(dst_node);
    if (dst_node->GetType() != NETOUTPUT) {
      continue;
    }
    NodeUtils::UpdateIsInputConst(identity_node);
    auto netoutput_op_desc = dst_node->GetOpDesc();
    GE_CHECK_NOTNULL(netoutput_op_desc);
    auto output_tensor_desc = netoutput_op_desc->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    GE_CHECK_NOTNULL(output_tensor_desc);
    int64_t data_offset = 0;
    (void)TensorUtils::GetDataOffset(*output_tensor_desc, data_offset);
    GE_ASSERT_NOTNULL(in_data_anchor->GetPeerOutAnchor());
    auto input_tensor = identity_node->GetOpDesc()->MutableInputDesc(in_data_anchor->GetPeerOutAnchor()->GetIdx());
    GE_CHECK_NOTNULL(input_tensor);
    GELOGI("Need update const Offset %ld to op [%s]", data_offset, identity_node->GetName().c_str());
    TensorUtils::SetDataOffset(*input_tensor, data_offset);
    TensorUtils::SetDataOffset(*output_tensor_desc, 0);
    NodeUtils::UpdateIsInputConst(dst_node);
  }
  return SUCCESS;
}

std::string MemLayoutConflictUtil::GetAnchorName(std::string name, IOType io_type, int32_t index) {
  std::string value_ = name + ((io_type == kOut) ? "_out_" : "_in_") + std::to_string(index);
  return value_;
}

AnchorPtr MemLayoutConflictUtil::GetAnchorFromIndexIo(const ge::NodeIndexIO &node) {
  GE_ASSERT_NOTNULL(node.node_ptr_);
  if (node.io_type_ == kOut) {
    return node.node_ptr_->GetOutAnchor(node.index_);
  } else {
    return node.node_ptr_->GetInAnchor(node.index_);
  }
}

bool MemLayoutConflictUtil::IsGraphFeatureMapRefreshable(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  std::string refreshable;
  (void)GetContext().GetOption(OPTION_FEATURE_BASE_REFRESHABLE, refreshable);
  if (refreshable.compare(kRefreshable) == 0) {
    return true;
  }
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  if (root_graph != nullptr) {
    return root_graph->GetGraphUnknownFlag() && (!graph->GetGraphUnknownFlag());
  }
  return false;
}

// for load model withq scens(helper, mdc) and single op scene
// the op2's input address can't be set correct on model execute, return false on this scene
// and identity will be insert in UserInputAndNoPaddingContinuousOutputChecker
//                                                                   data
//   data                                                              |
//    |                                                             identity
//   split (split_dim=0, no_task, no_padding_continuous_output) =>     |
//   /  \                                                            split
//  op1  op2                                                         /   |
//                                                                 op1  op2
// ref: MemcpyAddrAsyncPass::NeedInsertMemAddrAsyncNodeAfterData
bool MemLayoutConflictUtil::IsSupportUserInputNopaddingContinuousOutput(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  std::string build_graph_mode;
  const bool is_build_graph_offline =
    ((GetContext().GetOption(OPTION_BUILD_GRAPH_MODE, build_graph_mode) == GRAPH_SUCCESS) &&
     (build_graph_mode.compare(kOffline) == 0));

  bool is_single_op = false;
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  if (root_graph != nullptr) {
    (void)AttrUtils::GetBool(graph, ATTR_SINGLE_OP_SCENE, is_single_op);
  }

  const bool is_hete = ExecutionRuntimeUtils::IsHeterogeneous();
  GELOGI("[MemConflict] %s is_build_graph_offline %d, is_single_op %d, is_hete %d", graph->GetName().c_str(),
         is_build_graph_offline, is_single_op, is_hete);
  return !(is_build_graph_offline || is_single_op || is_hete);
}

bool MemLayoutConflictUtil::IsAddressRefreshable(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto type = node->GetType();
  if ((type == STREAMSWITCH) || (type == LABELSWITCHBYINDEX)) {
    return false;
  }

  GE_ASSERT_NOTNULL(node->GetOpDesc());
  if (ge::OpUtils::IsHcomNodeNotSupportAddrRefresh(node->GetOpDesc()) &&
      !IsGraphFeatureMapRefreshable(node->GetOwnerComputeGraph())) {
    return false;
  }
  // 后续支持地址刷新特性的时候需要调整判断逻辑
  if (node->GetOpDesc()->GetOpKernelLibName() == ge::kCustomOpKernelLibName) {
    return false;
  }

  return true;
}

bool MemLayoutConflictUtil::IsPhysicalAddressRefreshable(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto type = node->GetType();
  if ((type == STREAMSWITCH) || (type == LABELSWITCHBYINDEX)) {
    return false;
  }
  return true;
}

/*
 * 开启动态图和静态图复用为true，纯静态图虚拟地址不变化，但是物理地址会变化。
 * 静态图中有2种rts算子STREAMSWITCH/LABELSWITCHBYINDEX会使用物理地址，因此不支持地址刷新。
 * 所以如果开启了动静态内存复用，需要将feature map分为2段，这3种算子的内存在feature map单独一段，其虚拟地址和物理地址不会变化。
 */
bool MemLayoutConflictUtil::HasNotSupportPhysicalMemoryRefreshNode(const CheckFuncContext &context) {
  if (!context.graph_info.is_physical_memory_refreshable) {
    return false;
  }
  return (!IsPhysicalAddressRefreshable(context.node_a.node_)) || (!IsPhysicalAddressRefreshable(context.node_b.node_));
}

bool MemLayoutConflictUtil::IsStaticGraph(const ge::ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  bool is_dynamic_shape = false;

  (void)AttrUtils::GetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if (is_dynamic_shape || graph->GetGraphUnknownFlag()) {
    return false;
  } else {
    return true;
  }
}

bool MemLayoutConflictUtil::IsConst(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  return NodeUtils::IsConst(*node) || (node->GetType() == FILECONSTANT)
	 || OpTypeUtils::IsConstPlaceHolderNode(node->GetType());
}

/*
 * 1 两个in_anchor符号相同大部分场景是
 *   a    a_out_0 connect to b and c
 *   /\
 *  b  c  b_in_0 and c_in_0 is the same symbol
 *
 * 2 不过也有输出引用输入导致的场景
 *     a
 *     |
 *    RefNode
 *     |
 *     c
 *   RefNode_in_0 and c_in_0 is the same symbol
 *   如果RefNode_in_0 and c_in_0存在类型冲突，那么插入到RefNode_in_0前面是无法解决问题的。
 *
 * 该函数是识别场景2
 */
bool MemLayoutConflictUtil::IsNodeOutRefFromInput(const ge::NodeIndexIO &in_anchor,
                                                  const NodeIndexIOVector &all_nodes) {
  // all_nodes保存了和in_anchor同符号的所有anchor，如果in_anchor自身节点的输出也在all_nodes中，说明自身节点是输出引用输入
  for (const auto &node_index : all_nodes) {
    if ((node_index.io_type_ == kOut) && (node_index.node_ == in_anchor.node_)) {
      return true;
    }
  }
  return false;
}

/*
 * 获取out_anchor引用的输入anchor
 */
ge::NodeIndexIO MemLayoutConflictUtil::GetRefInput(const ge::NodeIndexIO &out_anchor,
                                                   const NodeIndexIOVector &all_nodes) {
  // all_nodes保存了和out_anchor同符号的所有anchor，如果out_anchor自身节点的输入也在all_nodes中，说明自身节点是输出引用输入
  for (const auto &node_index : all_nodes) {
    if ((node_index.io_type_ == kIn) && (node_index.node_ == out_anchor.node_)) {
      return node_index;
    }
  }
  return NodeIndexIO(static_cast<Node *>(nullptr), 0, kOut);
}

bool MemLayoutConflictUtil::IsNoPaddingContinuousInput(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  return IsNoPaddingContinuousInput(node.get());
}

bool MemLayoutConflictUtil::IsNoPaddingContinuousInput(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  bool is_nopading_input_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_nopading_input_continuous);
  return is_nopading_input_continuous && (node->GetAllInDataAnchorsSize() > 1U);
}

// 逆向遍历输出anchor的引用链，对每个节点执行判断函数
// 参数：  node - 起始节点
//        out_index - 起始输出anchor索引
//        judge_func - 判断函数，返回true表示终止遍历
// 返回值： true - judge_func返回true
//         false - judge_func从未返回true
bool MemLayoutConflictUtil::TraverseRefChainReverse(const Node *const node, int32_t out_index,
                                                    const std::function<bool(const Node *const, int32_t)>& judge_func) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(judge_func);
  const auto out_data_anchor = node->GetOutDataAnchor(out_index).get();
  GE_ASSERT_NOTNULL(out_data_anchor);

  std::stack<OutDataAnchor *> out_anchor_stack;
  out_anchor_stack.push(out_data_anchor);

  while (!out_anchor_stack.empty()) {
    const auto out_anchor = out_anchor_stack.top();
    const auto cur_node = out_anchor->GetOwnerNodeBarePtr();
    const auto cur_out_index = out_anchor->GetIdx();
    out_anchor_stack.pop();

    // 判断当前节点
    if (judge_func(cur_node, cur_out_index)) {
      return true;
    }

    // 继续遍历引用链
    int32_t reuse_in_index;
    if (GraphUtils::IsRefFromInput(cur_node->GetOutDataAnchor(cur_out_index), reuse_in_index)) {
      auto in_anchor = cur_node->GetInDataAnchor(reuse_in_index).get();
      if ((in_anchor != nullptr) &&
          (in_anchor->GetPeerOutAnchor() != nullptr) &&
          (in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr() != nullptr)) {
        out_anchor_stack.push(in_anchor->GetPeerOutAnchor().get());
      }
    }
  }

  return false;
}

// 遍历输入anchor的引用链，对每个节点执行判断函数
// 参数：  node - 起始节点
//        in_index - 起始输入anchor索引
//        judge_func - 判断函数，返回true表示终止遍历
// 返回值： true - judge_func返回true
//         false - judge_func从未返回true
bool MemLayoutConflictUtil::TraverseRefChain(const Node *const node, int32_t in_index,
                                             const std::function<bool(const Node *const, int32_t)>& judge_func) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(judge_func);
  const auto in_data_anchor = node->GetInDataAnchor(in_index).get();
  GE_ASSERT_NOTNULL(in_data_anchor);

  std::stack<InDataAnchor *> in_anchor_stack;
  in_anchor_stack.push(in_data_anchor);

  while (!in_anchor_stack.empty()) {
    const auto in_anchor = in_anchor_stack.top();
    const auto cur_node = in_anchor->GetOwnerNodeBarePtr();
    in_anchor_stack.pop();

    // 判断当前节点
    if (judge_func(cur_node, in_anchor->GetIdx())) {
      return true;
    }
    // 继续遍历引用链
    for (const auto out_anchor : cur_node->GetAllOutDataAnchorsPtr()) {
      int32_t reuse_in_index;
      if ((out_anchor == nullptr) ||
          (!GraphUtils::IsRefFromInput(cur_node->GetOutDataAnchor(out_anchor->GetIdx()), reuse_in_index)) ||
          (reuse_in_index != in_anchor->GetIdx())) {
        continue;
      }
      for (const auto peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
        if ((peer_in_anchor != nullptr) && (peer_in_anchor->GetOwnerNodeBarePtr() != nullptr)) {
          in_anchor_stack.push(peer_in_anchor);
        }
      }
    }
  }

  return false;
}

/*
          a    b   c
          |___|___|
              |
          d   pc1 f
          |___|___|
              |
              pc2
 级联场景对于a的对端，找到的是pc2，对于b/c的对端，找到的是pc1
*/
bool MemLayoutConflictUtil::IsContinuousInputThroughRefNode(InDataAnchor *const in_data_anchor,
                                                            const bool no_padding,
                                                            InDataAnchor *&continuous_node_in_anchor) {
  GE_ASSERT_NOTNULL(in_data_anchor);
  std::stack<InDataAnchor *> in_data_anchor_stack;
  in_data_anchor_stack.push(in_data_anchor);
  continuous_node_in_anchor = nullptr;

  while (!in_data_anchor_stack.empty()) {
    const auto in_anchor = in_data_anchor_stack.top();
    in_data_anchor_stack.pop();
    GE_ASSERT_NOTNULL(in_anchor);
    const auto  is_continue = no_padding ?
        MemLayoutConflictUtil::IsNoPaddingContinuousInput(in_anchor->GetOwnerNodeBarePtr()) :
        MemLayoutConflictUtil::IsContinuousInput(in_anchor->GetOwnerNodeBarePtr());
    if (is_continue) {
      continuous_node_in_anchor = in_anchor;
    }
    const auto cur_node = in_anchor->GetOwnerNodeBarePtr();
    for (const auto &out_anchor : cur_node->GetAllOutDataAnchors()) {
      int32_t reuse_in_index;
      if (GraphUtils::IsRefFromInput(out_anchor, reuse_in_index) && (reuse_in_index == in_anchor->GetIdx())) {
        for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
          in_data_anchor_stack.push(peer_in_anchor);
        }
      }
    }
  }
  return (continuous_node_in_anchor != nullptr);
}

bool MemLayoutConflictUtil::IsContinuousOutputThroughRefNode(ge::OutDataAnchor *const out_data_anchor,
    const bool no_padding, ge::OutDataAnchor *&continuous_node_out_anchor) {
  GE_ASSERT_NOTNULL(out_data_anchor);
  std::stack<OutDataAnchor *> out_data_anchor_stack;
  out_data_anchor_stack.push(out_data_anchor);
  continuous_node_out_anchor = nullptr;

  while (!out_data_anchor_stack.empty()) {
    const auto out_anchor = out_data_anchor_stack.top();
    out_data_anchor_stack.pop();
    GE_ASSERT_NOTNULL(out_anchor);
    const auto cur_node = out_anchor->GetOwnerNodeBarePtr();
    const auto is_continuous = no_padding ? MemLayoutConflictUtil::IsNoPaddingContinuousOutput(cur_node) :
        MemLayoutConflictUtil::IsContinuousOutput(cur_node);
    if (is_continuous) {
      continuous_node_out_anchor = out_anchor;
    }
    int32_t reuse_in_index;
    if (GraphUtils::IsRefFromInput(cur_node->GetOutDataAnchor(out_anchor->GetIdx()), reuse_in_index)) {
      const auto in_anchor = cur_node->GetInDataAnchor(reuse_in_index);
      if ((in_anchor != nullptr) && (in_anchor->GetPeerOutAnchor() != nullptr)) {
        out_data_anchor_stack.push(in_anchor->GetPeerOutAnchor().get());
      }
    }
  }
  return (continuous_node_out_anchor != nullptr);
}

bool MemLayoutConflictUtil::IsNoPaddingContinuousOutput(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  bool is_nopading_output_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_nopading_output_continuous);
  return is_nopading_output_continuous && (node->GetAllOutDataAnchorsSize() > 1U);
}

bool MemLayoutConflictUtil::IsNoPaddingContinuousOutput(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  return IsNoPaddingContinuousOutput(node.get());
}

bool MemLayoutConflictUtil::IsContinuousInput(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  return IsContinuousInput(node.get());
}

bool MemLayoutConflictUtil::IsContinuousInput(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  bool is_input_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
  return is_input_continuous && (node->GetAllInDataAnchorsSize() > 1U);
}

bool MemLayoutConflictUtil::IsContinuousOutput(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  return IsContinuousOutput(node.get());
}

bool MemLayoutConflictUtil::IsContinuousOutput(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  bool is_output_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);
  return is_output_continuous && (node->GetAllOutDataAnchorsSize() > 1U);
}

bool MemLayoutConflictUtil::IsContainTargetType(const SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> &type,
                                                const AnchorAttribute &target_type) {
  for (const auto type_item : type) {
    if (type_item == target_type) {
      return true;
    }
  }
  return false;
}

int64_t MemLayoutConflictUtil::GetAnchorMemType(const NodePtr &node, const IOType io_type, const uint32_t index) {
  if (node == nullptr) {
    return RT_MEMORY_DEFAULT;
  }
  std::vector<int64_t> mem_type_list;
  if (io_type == IOType::kIn) {
    (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type_list);
  } else {
    (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type_list);
  }
  if (index < mem_type_list.size()) {
    return mem_type_list[index];
  }
  return RT_MEMORY_DEFAULT;
}

bool MemLayoutConflictUtil::IsSameMemType(const CheckFuncContext &context) {
  const int64_t mem_type_a = MemLayoutConflictUtil::GetAnchorMemType(context.node_a.node_, context.node_a.io_type_,
                                                                     context.node_a.index_);
  const int64_t mem_type_b = MemLayoutConflictUtil::GetAnchorMemType(context.node_b.node_, context.node_b.io_type_,
                                                                     context.node_b.index_);
  if (mem_type_a == mem_type_b) {
    GELOGI("[MemConflict] same mem type[%lld], node_a: %s, node_b: %s", mem_type_a, context.node_a.ToString().c_str(),
           context.node_b.ToString().c_str());
    return true;
  }
  return false;
}

/*
 *   var  const
 *     \    /
 *     assign (输出引用输入)
 *       |
 *      node
 *
 *  判断in_anchor是不是连着assign连着variable
 *  入参 in_anchor ：node的输入anchor
 *  出参 flag :  true表示是这种场景
 */
Status MemLayoutConflictUtil::IsLinkAssignLinkVar(const NodeIndexIO &in_anchor, bool &flag) {
  const auto in_data_anchor = in_anchor.node_->GetInDataAnchor(in_anchor.index_);
  GE_CHECK_NOTNULL(in_data_anchor);
  if (in_data_anchor->GetPeerOutAnchor() != nullptr) {
    GE_CHECK_NOTNULL(in_data_anchor->GetPeerOutAnchor()->GetOwnerNode());
    if (OpTypeUtils::IsAssignLikeNode(in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetType())) {
      flag = true;
      GELOGI("[MemConflict] variable-assign-in_anchor, in_anchor: %s", in_anchor.ToString().c_str());
      return SUCCESS;
    }
  }
  flag = false;
  return SUCCESS;
}

/*
 *   var  const
 *     \    /
 *     assign (输出引用输入)
 *       |
 *     ref_node
 *
 *  判断是不是variable-assign-ref_node这种场景
 *  特征：ref_node是输出引用输入，assign也是输出引用输入，且ref_node和assign相连的in_data_anchor是被输出引用的
 *  入参 ref_node_out_anchor ：ref_node的输出anchor
 *  出参 in_anchors :  ref_node上和assign相连的anchor
 */
Status MemLayoutConflictUtil::IsVarLinkAssignLinkRefNode(const CheckFuncContext &context,
                                                         const NodeIndexIO &ref_node_out_anchor,
                                                         std::vector<NodeIndexIO> &in_anchors) {
  for (const auto &node_index_io : context.all_nodes) {
    // 满足这个条件的in_data_anchor，一定是被ref_node_index_io的输出引用的
    if ((node_index_io.node_ == ref_node_out_anchor.node_) && (node_index_io.io_type_ == kIn)) {
      bool flag;
      GE_ASSERT_SUCCESS(IsLinkAssignLinkVar(node_index_io, flag));
      if (flag) {
        in_anchors.emplace_back(node_index_io);
      }
    }
  }
  return SUCCESS;
}

/*
 *  refdata  const （refdata还可以是variable）
 *     \    /
 *     assign  a
 *       \   /
 *        xxx (可以是PhonyConcat, 可以是Hcom，可以是需要特殊内存类型的节点)
 *
 *  done: true 表示已经处理完成，后续不需要继续处理
 *
 *  refdata被认为是用户输入，不认为是不可变地址输入。但是如果发生冲突，需要插在assign后面
 */
Status MemLayoutConflictUtil::AssignVarInsertIdentity(CheckFuncContext &context,
                                                      const AnchorAttribute &var_type, bool &done) {
  auto ref_node = context.node_b;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_b, var_type)) {
    ref_node = context.node_a;
  }
  std::vector<NodeIndexIO> in_anchors;
  MemLayoutConflictUtil::IsVarLinkAssignLinkRefNode(context, ref_node, in_anchors);
  for (const auto &node_index_io : in_anchors) {
    auto in_data_anchor = node_index_io.node_->GetInDataAnchor(node_index_io.index_);
    context.result.insert(in_data_anchor);
    GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_index_io);
    done = true;
  }
  return SUCCESS;
}

/*
 * 1 anchor1, anchor2, anchorN代表任意节点的anchor，结尾数字相同表示是同一个anchor。
 * 2 node_a和node_b都需要NoPadding连续输入内存
 * 3 node_a和node_b输入anchor的对端如果有重合（也就是有的节点一个输出同时给到node_a,node_b作为输入），
 *   需要将node_a/node_b所有输入对端列表做个校验，满足以下排列的才不冲突(node_a和node_b可以交换)。
 *
 * 不冲突的排列组合:
 * 不冲突场景1：子集
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2 anchor3
 *
 * 不冲突场景2：完全一样
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3
 *
 * 不冲突场景3：完全不一样
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3
 * node_b所有输入anchor的对端：                         anchor4 anchor5 anchor6
 *
 * 冲突的举例：
 * node_a所有输入anchor的对端： anchor1 anchor2
 * node_b所有输入anchor的对端：         anchor2 anchor1
 *
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2         anchor4
 *
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2                 anchor5
 *
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：                                 anchor5  anchor2
 *
 * node_a所有输入anchor的对端：                anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端： anchor5 anchor6                anchor3 anchor4
 *
 * 冲突场景 ：部分重合，没有间隙
 * 理论上并不冲突，但是当前内存分配模块只会给anchor1分配内存，长度为anchor1+anchor2, anchor3, anchor4, 并不包含anchor5。
 * 所以当前按照冲突处理
 * node_a所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2 anchor3 anchor4 anchor5
 */
Status MemLayoutConflictUtil::IsNoPaddingContinuousNodeConflict(const CheckFuncContext &context, bool &has_conflict) {
  GE_ASSERT_NOTNULL(context.node_a.node_);
  GE_ASSERT_NOTNULL(context.node_b.node_);
  const auto a_in_peer_anchors = GetAllInAnchorsPeer(context.node_a.node_);
  const auto b_in_peer_anchors = GetAllInAnchorsPeer(context.node_b.node_);
  const auto index_a_b = GetFirstSameIndex(a_in_peer_anchors, b_in_peer_anchors);
  // 不冲突场景3：完全不一样
  if (index_a_b.first == std::numeric_limits<size_t>::max()) {
    has_conflict = false;
    GELOGI("[MemConflict] total different input nodes, not conflict. node_a: %s, node_b:%s",
           CheckerLog::ToStr(context.node_a).c_str(), CheckerLog::ToStr(context.node_b).c_str());
    return SUCCESS;
  }
  /*
  * 不冲突场景1：子集
  * 不冲突场景2：完全一样
  */
  if (IsSubVector(a_in_peer_anchors, b_in_peer_anchors) || IsSubVector(b_in_peer_anchors, a_in_peer_anchors)) {
    has_conflict = false;
    GELOGI("[MemConflict] node_a inputs and node_b inputs are the same or partial same, not conflict. node_a: %s, "
           "node_b:%s", CheckerLog::ToStr(context.node_a).c_str(), CheckerLog::ToStr(context.node_b).c_str());
    return SUCCESS;
  }
  has_conflict = true;
  GELOGI("[MemConflict][Conflict] partial same inputs, node_a: %s, node_b:%s",
         CheckerLog::ToStr(context.node_a).c_str(), CheckerLog::ToStr(context.node_b).c_str());
  return SUCCESS;
}

// 如果输入是RefNode，则获取RefNode的输入
std::vector<OutDataAnchorPtr> MemLayoutConflictUtil::GetAllRealInPeer(const NodePtr &node) {
  std::vector<OutDataAnchorPtr> peer_out_anchors;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    if ((in_anchor != nullptr) && (in_anchor->GetPeerOutAnchor() != nullptr)) {
      peer_out_anchors.emplace_back(GetRealOutAnchor(in_anchor->GetPeerOutAnchor()));
    }
  }
  return peer_out_anchors;
};

/*
 *  PhonyConcat 的所有输入是同一个anchor
 *              data
 *               /\
 *  assign_slice0 assign_slice1 (inplace)
 *             \   /
 *          PhonyConcat
 */
bool MemLayoutConflictUtil::AllRealInputsAreTheSameOutAnchor(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto real_peer_out_anchors = MemLayoutConflictUtil::GetAllRealInPeer(node);
  if (real_peer_out_anchors.empty()) {
    return false;
  }
  OutDataAnchor *first_anchor = nullptr;
  bool only_one_out_anchor = true;
  for (const auto &real_peer_out_anchor : real_peer_out_anchors) {
    if (first_anchor == nullptr) {
      first_anchor = real_peer_out_anchor.get();
    } else if (first_anchor != real_peer_out_anchor.get()) {
      only_one_out_anchor = false;
      break;
    }
  }
  if (only_one_out_anchor) {
    GELOGI("[MemConflict] node[%s][%s] all real inputs are the same anchor[%s]", node->GetNamePtr(), node->GetTypePtr(),
      CheckerLog::ToStr(real_peer_out_anchors[0U]).c_str());
    return true;
  }
  return false;
}
/*
 * 1 anchor1, anchor2, anchorN代表任意节点的anchor，结尾数字相同表示是同一个anchor。
 * 2 node_a连续输出， 和node_b需要连续输入
 * 3 node_a的输出如果连接到node_b，
 *   需要将node_a的输出，node_b所有输入对端列表做个校验，满足以下排列的才不冲突。
 *
 * 不冲突的排列组合:
 * 不冲突场景1：子集
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2 anchor3
 *
 * or:
 * node_a所有输出anchor     ：         anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 *
 * 不冲突场景2：完全一样
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3
 *
 * 不冲突场景3：部分一样
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2 anchor3 anchor4 anchor5
 *
 * or:
 * node_a所有输出anchor     ：         anchor2 anchor3 anchor4 anchor5
 * node_b所有输入anchor的对端： anchor1 anchor2 anchor3 anchor4
 *
 * 冲突的举例：
 * node_a所有输出anchor     ： anchor1 anchor2
 * node_b所有输入anchor的对端：         anchor2 anchor1
 *
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2         anchor4
 *
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：         anchor2                 anchor5
 *
 * node_a所有输出anchor     ： anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端：                                 anchor5  anchor2
 *
 * node_a所有输出anchor     ：                anchor1 anchor2 anchor3 anchor4
 * node_b所有输入anchor的对端： anchor5 anchor6                anchor3 anchor4
 */
Status MemLayoutConflictUtil::IsContinuousOutAndInConflict(const CheckFuncContext &context, bool &has_conflict) {
  GE_ASSERT_NOTNULL(context.node_a.node_);
  GE_ASSERT_NOTNULL(context.node_b.node_);
  auto continuous_out_node = context.node_a;
  auto continuous_in_node = context.node_b;
  if (IsContainTargetType(context.type_a, ANCHOR_ATTR_CONTINUOUS_INPUT)) {
    continuous_out_node = context.node_b;
    continuous_in_node = context.node_a;
  }
  const auto out_anchors = GetAllOutAnchors(continuous_out_node.node_);
  const auto peer_anchors = GetAllRealInPeer(continuous_in_node.node_);
  /*
  * 不冲突场景1：子集
  * 不冲突场景2：完全一样
  * 不冲突场景3：部分一样
  */
  if (IsSameOrPartialSame(out_anchors, peer_anchors)) {
    has_conflict = false;
    GELOGI("[MemConflict] continuous subsets of continuous in/out are the same, not conflict. node_a: %s, "
           "node_b:%s", CheckerLog::ToStr(context.node_a).c_str(), CheckerLog::ToStr(context.node_b).c_str());
    return SUCCESS;
  }
  has_conflict = true;
  GELOGI("[MemConflict][Conflict] continuous in/out are partial same, node_a: %s, node_b:%s",
         CheckerLog::ToStr(context.node_a).c_str(), CheckerLog::ToStr(context.node_b).c_str());
  return SUCCESS;
}

// 如果out_anchor引用了其他anchor，后续不会判断这两个anchor是否冲突
bool MemLayoutConflictUtil::IsRefFromVar(const OutDataAnchorPtr &out_anchor, NodePtr &src_node,
                                         const SymbolToAnchors &symbol_to_anchors,
                                         const AnchorToSymbol &anchor_to_symbol) {
  const auto node = out_anchor->GetOwnerNode();
  NodeIndexIO out_anchor_node(node, out_anchor->GetIdx(), kOut);
  const auto out_anchor_symbol_iter = anchor_to_symbol.find(out_anchor_node.ToString());
  if (out_anchor_symbol_iter == anchor_to_symbol.end()) {
    return false;
  }
  std::string src_var_name;
  if (HasRefVarName(out_anchor, src_var_name)) {
    const auto src_var_symbol = src_var_name + "_out_0";
    const auto src_symbol_iter = anchor_to_symbol.find(src_var_symbol);
    if ((src_symbol_iter != anchor_to_symbol.end())
        && (src_symbol_iter->second == out_anchor_symbol_iter->second)) {
      const auto anchor_iter = symbol_to_anchors.find(src_symbol_iter->second);
      GE_ASSERT(anchor_iter != symbol_to_anchors.end());
      for (const auto &anchor : anchor_iter->second) {
        if (anchor.ToString() == src_var_symbol) {
          src_node = anchor.node_;
          GELOGD("[MemConflict] %s is ref from %s by ref_var_src_var_name attr",
                 out_anchor->GetOwnerNode()->GetNamePtr(), anchor.node_->GetNamePtr());
          return true;
        }
      }
    }
  }
  return false;
}

TypeVector GetNodeInputTypes(const NodePtr &node, const ComputeGraphPtr &graph) {
  TypeVector node_types;
  if ((node->GetType() == NETOUTPUT) && (node->GetOwnerComputeGraph() == graph)) {
    node_types.emplace_back(ANCHOR_ATTR_USER_MEMORY_OUTPUT);
  }
  if (!MemLayoutConflictUtil::IsAddressRefreshable(node)) {
    node_types.emplace_back(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_INPUT);
  }
  if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(node)) {
    node_types.emplace_back(ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT);
  }
  if (MemLayoutConflictUtil::IsContinuousInput(node)) {
    node_types.emplace_back(ANCHOR_ATTR_CONTINUOUS_INPUT);
  }
  return node_types;
}

void MemLayoutConflictUtil::MarkInTypes(const NodePtr &node, const ComputeGraphPtr &graph,
                                        const std::unique_ptr<Checker> &checker) {
  std::vector<int64_t> input_memory_types;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_types);

  TypeVector node_types = GetNodeInputTypes(node, graph);
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    TypeVector types = node_types;
    if (!input_memory_types.empty() && static_cast<size_t>(in_data_anchor->GetIdx()) < input_memory_types.size()) {
      if (MemTypeUtils::IsMemoryTypeSpecial(input_memory_types[in_data_anchor->GetIdx()])) {
        types.emplace_back(ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT);
      }
    }

    if (!types.empty()) {
      GELOGD("[MemConflict] anchor: %s_In_%d, types: %s", node->GetNamePtr(), in_data_anchor->GetIdx(),
             CheckerLog::ToStr(types).c_str());
    } else {
      types.emplace_back(ANCHOR_ATTR_NORMAL_INPUT);
      GELOGD("[MemConflict] anchor: %s_In_%d, types: %s", node->GetNamePtr(), in_data_anchor->GetIdx(),
             CheckerLog::ToStr(types).c_str());
    }
    checker->SaveTypes(types, in_data_anchor);
  }
}

TypeVector GetNodeOutputTypes(const NodePtr &node, const ComputeGraphPtr &graph) {
  TypeVector node_types;
  const auto node_type = node->GetType();
  if (OpTypeUtils::IsDataNode(node_type) && (node->GetOwnerComputeGraph() == graph)) {
    node_types.emplace_back(ANCHOR_ATTR_USER_MEMORY_INPUT);
  }
  if (MemLayoutConflictUtil::IsConst(node) || OpTypeUtils::IsVariableNode(node_type)) {
    node_types.emplace_back(ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT);
  }
  if (!MemLayoutConflictUtil::IsAddressRefreshable(node)) {
    node_types.emplace_back(ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT);
  }
  if (MemLayoutConflictUtil::IsNoPaddingContinuousOutput(node)) {
    node_types.emplace_back(ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT);
  }
  if (MemLayoutConflictUtil::IsContinuousOutput(node)) {
    node_types.emplace_back(ANCHOR_ATTR_CONTINUOUS_OUTPUT);
  }
  return node_types;
}

void MemLayoutConflictUtil::MarkOutTypes(const NodePtr &node, const ComputeGraphPtr &graph,
                                         const SymbolToAnchors &symbol_to_anchors,
                                         const AnchorToSymbol &anchor_to_symbol,
                                         const std::unique_ptr<Checker> &checker) {
  std::vector<int64_t> output_memory_types;
  (void)ge::AttrUtils::GetListInt(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_types);

  bool is_user_input = false;
  if (OpTypeUtils::IsDataNode(node->GetType()) && (node->GetOwnerComputeGraph() == graph)) {
    is_user_input = true;
  }

  TypeVector node_types = GetNodeOutputTypes(node, graph);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    TypeVector types = node_types;

    if (!is_user_input) {
      int32_t reuse_in_index;
      NodePtr src_node;
      if (IsRefFromVar(out_data_anchor, src_node, symbol_to_anchors, anchor_to_symbol)) {
        types.emplace_back(ANCHOR_ATTR_REFERENCE_OUTPUT);
        checker->SaveRefMap(out_data_anchor, src_node);
      } else if (GraphUtils::IsRefFromInput(out_data_anchor, reuse_in_index)) {
        types.emplace_back(ANCHOR_ATTR_REFERENCE_OUTPUT);
      }
    }

    if (!output_memory_types.empty() && (static_cast<size_t>(out_data_anchor->GetIdx()) < output_memory_types.size())) {
      if (MemTypeUtils::IsMemoryTypeSpecial(output_memory_types[out_data_anchor->GetIdx()])) {
        types.emplace_back(ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT);
      }
    }

    if (!types.empty()) {
      GELOGD("[MemConflict] anchor: %s_Out_%d, types: %s", node->GetNamePtr(), out_data_anchor->GetIdx(),
             CheckerLog::ToStr(types).c_str());
    } else {
      types.emplace_back(ANCHOR_ATTR_NORMAL_OUTPUT);
      GELOGD("[MemConflict] anchor: %s_Out_%d, types: %s", node->GetNamePtr(), out_data_anchor->GetIdx(),
             CheckerLog::ToStr(types).c_str());
    }
    checker->SaveTypes(types, out_data_anchor);
  }
}

Status MemLayoutConflictUtil::FindConflictNodes(const NodeIndexIOVector &all_nodes, AnchorSet &result,
                         const GraphInfo &graph_info, const std::unique_ptr<Checker> &checker) {
  std::vector<vector_bit_t> visit_flag(ATTR_BIT_MAX_LEN, vector_bit_t(ATTR_BIT_MAX_LEN, false));
  for (size_t i = 0U; i < all_nodes.size(); i++) {
    for (size_t j = i + 1U; j < all_nodes.size(); j++) {
      const NodeIndexIO &node_a = all_nodes[i];
      const NodeIndexIO &node_b = all_nodes[j];
      const auto &type_a = checker->GetTypes(node_a);
      const auto &type_b = checker->GetTypes(node_b);

      CheckFuncContext context = {node_a, node_b, 0U, 0U, type_a, type_b, all_nodes, graph_info, result};
      GE_ASSERT_SUCCESS(checker->CheckConflict(context, visit_flag),
                        "[Call][CheckConflict] failed. node_a: %s, node_b: %s,"
                        " type_a: %s, type_b: %s",
                        node_a.ToString().c_str(), node_b.ToString().c_str(), CheckerLog::ToStr(type_a).c_str(),
                        CheckerLog::ToStr(type_b).c_str());
    }
  }
  return SUCCESS;
}

Status MemLayoutConflictUtil::IsGraphExistMemConflictSymbol(const ComputeGraphPtr &graph,
                                                            const AnchorToSymbol &anchor_to_symbol,
                                                            const SymbolToAnchors &symbol_to_anchors,
                                                            bool &has_conflict) {
  GE_ASSERT_NOTNULL(graph);
  std::unique_ptr<Checker> checker = MakeUnique<Checker>();
  GE_ASSERT_NOTNULL(checker);

  GraphInfo graph_info;
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  graph_info.is_root_graph_static = IsStaticGraph(root_graph);
  graph_info.is_feature_map_refreshable = IsGraphFeatureMapRefreshable(graph);
  graph_info.is_support_user_input_nopadding_continuous_output = IsSupportUserInputNopaddingContinuousOutput(graph);
  graph_info.is_physical_memory_refreshable = VarManager::IsGeUseExtendSizeMemoryFull();

  for (auto &symbol_to_anchor : symbol_to_anchors) {
    for (auto &anchor : symbol_to_anchor.second) {
      MarkInTypes(anchor.node_, graph, checker);
      MarkOutTypes(anchor.node_, graph, symbol_to_anchors, anchor_to_symbol, checker);
    }
  }
  for (const auto &anchor_iter : symbol_to_anchors) {
    NodeIndexIOVector all_nodes(anchor_iter.second.cbegin(), anchor_iter.second.cend());
    for (const auto &node_index_io : all_nodes) {
      GE_ASSERT_NOTNULL(GetAnchorFromIndexIo(node_index_io), "node_index_io, %s",
                        node_index_io.ToString().c_str());
    }

    AnchorSet conflict_set;
    GE_ASSERT_SUCCESS(FindConflictNodes(all_nodes, conflict_set, graph_info, checker),
                      "[Call][FindConflictNodes] for graph:%s, symbol: %s, is_root_graph_static: %d, "
                      "is_feature_map_refreshable: %d, is_physical_memory_refreshable: %d",
                      graph->GetName().c_str(), anchor_iter.first.c_str(), graph_info.is_root_graph_static,
                      graph_info.is_feature_map_refreshable, graph_info.is_physical_memory_refreshable);
    if (!conflict_set.empty()) {
      has_conflict = true;
      GELOGI("[MemConflict] Graph %s has conflict symbol %s", graph->GetName().c_str(), anchor_iter.first.c_str());
      return SUCCESS;
    }
  }

  return SUCCESS;
}

Status MemLayoutConflictUtil::CheckOneSubGraphConflict(const ComputeGraphPtr &sub_graph,
                                                       std::vector<InDataAnchorPtr> &in_data_anchors) {
  const auto netoutput = sub_graph->GetOrUpdateNetOutputNode();
  GE_CHECK_NOTNULL(netoutput);

  SymbolToAnchors symbol_to_anchors_;
  AnchorToSymbol anchor_to_symbol_;
  GE_ASSERT_SUCCESS(GraphUtils::GetRefMapping(sub_graph, symbol_to_anchors_, anchor_to_symbol_),
                    "[Call][GetRefMapping] for sub graph:%s failed.", sub_graph->GetName().c_str());

  std::unordered_set<std::string> symbols;
  for (const auto &in_anchor : netoutput->GetAllInDataAnchors()) {
    const auto in_anchor_name = MemLayoutConflictUtil::GetAnchorName(netoutput->GetName(), kIn, in_anchor->GetIdx());
    // find all out anchors with same symbol with netoutput
    const auto &symbol_name = anchor_to_symbol_[in_anchor_name];
    const auto &node_indexes = symbol_to_anchors_[symbol_name];

    bool is_symbol_from_data = false;
    for (const auto &node_index : node_indexes) {
      if (node_index.io_type_ != kOut) {
        continue;
      }

      const auto output_node = node_index.node_ptr_;
      const auto out_node_type = output_node->GetType();
      /*
        这些类型的算子直连netoutput，需要加identity
          1、data，且与netoutput在同一层图中
          2、不可变地址类型，包括const、constant、variable
          3、引用到其他算子地址
      */
      if ((out_node_type == DATA && output_node->GetOwnerComputeGraph() == netoutput->GetOwnerComputeGraph())
          || MemLayoutConflictUtil::IsConst(node_index.node_)
          || OpTypeUtils::IsVarLikeNode(out_node_type)
          || IsRefFromRefData(*output_node)) {
        in_data_anchors.emplace_back(in_anchor);
        GELOGI("[MemConflict][Conflict] Type[%s] node directly connect to netoutput.",
               output_node->GetType().c_str());
        is_symbol_from_data = true;
        break;
      }
    }

    if (is_symbol_from_data) {
      continue;
    }
    // 如果存在单输出多引用都连netoutput则冲突
    if (!symbols.emplace(symbol_name).second) {
      in_data_anchors.emplace_back(in_anchor);
      GELOGI("[MemConflict][Conflict] Node single output multiple references connect to netoutput, insert identity node"
             " before netoutput[%s] [%d]th inanchor of subgraph.",
             netoutput->GetName().c_str(), in_anchor->GetIdx());
    }
  }

  return SUCCESS;
}

Status MemLayoutConflictUtil::CheckIfConflict(const NodePtr &ctrl_node,
                                              std::vector<InDataAnchorPtr> &in_data_anchors) {
  const auto then_graph = NodeUtils::GetSubgraph(*ctrl_node, kThenBranchIndex);
  const auto else_graph = NodeUtils::GetSubgraph(*ctrl_node, kElseBranchIndex);
  GE_CHECK_NOTNULL(then_graph, "node: %s", ctrl_node->GetNamePtr());
  GE_CHECK_NOTNULL(else_graph, "node: %s", ctrl_node->GetNamePtr());

  GE_ASSERT_SUCCESS(CheckOneSubGraphConflict(then_graph, in_data_anchors),
                    "[Call][SolveOneCtrlNodeConflict] for if node:%s then graph failed.", ctrl_node->GetName().c_str());
  GE_ASSERT_SUCCESS(CheckOneSubGraphConflict(else_graph, in_data_anchors),
                    "[Call][SolveOneCtrlNodeConflict] for if node:%s else graph failed.", ctrl_node->GetName().c_str());

  return SUCCESS;
}

Status MemLayoutConflictUtil::CheckCaseConflict(const NodePtr &ctrl_node,
                                                std::vector<InDataAnchorPtr> &in_data_anchors) {
  const size_t num_subgraphs = ctrl_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  for (size_t i = 0U; i < num_subgraphs; ++i) {
    const auto sub_graph = NodeUtils::GetSubgraph(*ctrl_node, i);
    GE_CHECK_NOTNULL(sub_graph, "node: %s", ctrl_node->GetNamePtr());
    GE_ASSERT_SUCCESS(CheckOneSubGraphConflict(sub_graph, in_data_anchors),
                      "[Call][SolveOneCtrlNodeConflict] for case node:%s graph idx %d failed.",
                      ctrl_node->GetName().c_str(), i);
  }
  return SUCCESS;
}

Status MemLayoutConflictUtil::GetWhileBodyDataToNetoutputNodes(const ComputeGraphPtr &while_body,
                                                          std::vector<NodePtr> &data_nodes_change,
                                                          std::set<uint32_t> &bypass_index_no_change) {
  for (const auto &node : while_body->GetDirectNode()) {
    if (node->GetType() == DATA) {
      uint32_t input_index = 0U;
      if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, input_index)) {
        continue;
      }

      // Data node has and only has one output
      const auto out_data_anchor = node->GetOutDataAnchor(0);
      if (out_data_anchor == nullptr || (out_data_anchor->GetPeerInDataAnchors().size() != 1U)) {
        continue;
      }

      const auto peer_in_anchor = out_data_anchor->GetPeerInDataAnchors().at(0U);
      GE_CHECK_NOTNULL(peer_in_anchor);
      const auto peer_node = peer_in_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(peer_node);
      if (peer_node->GetType() != NETOUTPUT) {
        continue;
      }

      const auto op_desc = peer_node->GetOpDesc();
      uint32_t output_index = 0U;
      if ((op_desc == nullptr) ||
          !AttrUtils::GetInt(op_desc->GetInputDesc(peer_in_anchor->GetIdx()),
                             ATTR_NAME_PARENT_NODE_INDEX, output_index)) {
        continue;
      }

      if (input_index != output_index) {
        data_nodes_change.emplace_back(node);
      } else {
        bypass_index_no_change.insert(peer_in_anchor->GetIdx());
      }
    }
  }

  return SUCCESS;
}

Status MemLayoutConflictUtil::IsWhileNeedInsertIdentityAtOutput(const ComputeGraphPtr &while_body,
                                                                bool &is_need_insert) {
  const auto netoutput = while_body->GetOrUpdateNetOutputNode();
  GE_CHECK_NOTNULL(netoutput);
  const auto output_size = netoutput->GetAllInDataAnchorsSize();
  if (output_size != 1U) {
    is_need_insert = true;
    return SUCCESS;
  }

  const auto data_node = while_body->FindFirstNodeMatchType(DATA);
  GE_CHECK_NOTNULL(data_node);

  const auto in_anchor = netoutput->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(in_anchor);
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  const auto out_node = peer_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(out_node);

  if (data_node->GetAllOutDataAnchorsSize() != 1U) {
    is_need_insert = true;
    return SUCCESS;
  }

  const auto out_data_anchor = data_node->GetOutDataAnchor(0);
  SymbolToAnchors symbol_to_anchors_;
  AnchorToSymbol anchor_to_symbol_;
  GE_ASSERT_SUCCESS(GraphUtils::GetRefMapping(while_body, symbol_to_anchors_, anchor_to_symbol_),
                    "[Call][GetRefMapping] for while_body:%s failed.", while_body->GetName().c_str());

  const auto symbol_cnt = symbol_to_anchors_.size();
  // one input/output, 1 symbol, then equal with (data--netoutput), no need to insert at output
  // one input/output, >2 symbols, then more than one node inside, no need to insert at output
  if (symbol_cnt != 2) {
    GELOGD("[MemConflict] No need to insert output identity node in while_body %s, symbol_to_anchors size %d.",
            while_body->GetName().c_str(), symbol_cnt);
    is_need_insert = false;
    return SUCCESS;
  }

  is_need_insert = true;
  return SUCCESS;
}

Status MemLayoutConflictUtil::CheckWhileConflict(const NodePtr &ctrl_node,
                                                 std::vector<InDataAnchorPtr> &conflit_anchors) {
  const auto while_body = NodeUtils::GetSubgraph(*ctrl_node, kBodyBranchIndex);
  GE_CHECK_NOTNULL(while_body, "node: %s", ctrl_node->GetNamePtr());

  std::vector<NodePtr> data_nodes_change;
  std::set<uint32_t> bypass_index_no_change;
  GE_ASSERT_SUCCESS(GetWhileBodyDataToNetoutputNodes(while_body, data_nodes_change, bypass_index_no_change),
                    "[Call][GetWhileBodyDataToNetoutputNodes] faild.");

  const auto netoutput = while_body->GetOrUpdateNetOutputNode();
  GE_CHECK_NOTNULL(netoutput);

  bool is_need_insert = true;
  GE_ASSERT_SUCCESS(IsWhileNeedInsertIdentityAtOutput(while_body, is_need_insert));

  if (is_need_insert) {
    for (size_t i = 0U; i < netoutput->GetAllInDataAnchorsSize(); i++) {
      if (bypass_index_no_change.count(i) != 0U) {
        continue;
      }
      InDataAnchorPtr in_data_anchor = netoutput->GetInDataAnchor(i);
      GE_CHECK_NOTNULL(in_data_anchor);
      if (MemLayoutConflictUtil::IsSkipInsert(in_data_anchor)) {
        continue;
      }
      conflit_anchors.emplace_back(in_data_anchor);
    }
  }

  for (const auto &data_node : data_nodes_change) {
    GE_CHECK_NOTNULL(data_node);
    // Data node has and only has one output
    OutDataAnchorPtr out_data_anchor = data_node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_data_anchor);
    InDataAnchorPtr in_data_anchor = out_data_anchor->GetPeerInDataAnchors().at(0U);
    GE_CHECK_NOTNULL(in_data_anchor);
    conflit_anchors.emplace_back(in_data_anchor);
  }

  return SUCCESS;
}

// 保存一下冲突的node信息，方便后续恢复
bool MemLayoutConflictUtil::IsCtrlNodeSubgraphExistMemConflictSymbol(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  std::vector<InDataAnchorPtr> conflict_anchors;
  for (const auto &node : graph->GetAllNodes()) {
    const auto node_type = node->GetType();
    if (check_subgraph_conflict_call.find(node_type) != check_subgraph_conflict_call.end()) {
      GELOGI("[MemConflict] Check subgraph conflict for node: %s", node->GetName().c_str());
      GE_ASSERT(check_subgraph_conflict_call[node_type](node, conflict_anchors) == SUCCESS);
    }
  }

  if (!conflict_anchors.empty()) {
    return true;
  }

  return false;
}

}  // namespace ge
