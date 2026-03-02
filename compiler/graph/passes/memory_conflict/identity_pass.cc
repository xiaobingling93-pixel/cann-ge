/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_conflict/identity_pass.h"

#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/omg_util.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const auto node_filter = [](const Node &ctrl_node) { return OpTypeUtils::IsAssignLikeNode(ctrl_node.GetType()); };
bool IsIdentityUsefulForVarCache(const NodePtr &node) {
  if (node->GetType() != READVARIABLEOP) {
    return false;
  }
  const auto &out_assign_nodes = NodeUtils::GetOutControlNodes(*node, node_filter);
  if (out_assign_nodes.empty()) {
    return false;
  }
  for (const auto &out_data_node : node->GetOutDataNodes()) {
    std::unordered_set<Node *> out_data_node_all_in_nodes;
    for (const auto &out_data_node_all_in_node : out_data_node->GetInAllNodes()) {
      (void)out_data_node_all_in_nodes.insert(out_data_node_all_in_node.get());
    }
    for (const auto &out_assign_node : out_assign_nodes) {
      if (out_data_node_all_in_nodes.find(out_assign_node.get()) != out_data_node_all_in_nodes.end()) {
        GELOGI("Node [%s %s]'s out ctrl assign node [%s %s] is also a input of it's data consumer node [%s %s]",
               node->GetName().c_str(), node->GetType().c_str(), out_assign_node->GetName().c_str(),
               out_assign_node->GetType().c_str(), out_data_node->GetName().c_str(), out_data_node->GetType().c_str());
        return true;
      }
    }
  }
  return false;
}
///
/// 1. A `Identity` node may after a `Switch` node and has control-dependency-out nodes.
/// Or a `Identity` node may before a `Merge` node and has control-dependency-in nodes.
/// The identity nodes are used to represent control dependencies in condition branch, and can not be deleted.
/// 2. Check identity is near subgraph.
///    Eg. As output of Data node in subgraph
///        or as input of Netoutput of subgraph
///        or as input of one node with subgraph
///        or as output of one node with subgraph
/// 3. identity with attr no_need_constant_folding should not be deleted too
/// 4. identity for var cache can not be deleted
Status CheckIdentityUsable(const NodePtr &node, bool &usable) {
  std::string node_type;
  if (node->GetOpDesc()->HasAttr(ATTR_NO_NEED_CONSTANT_FOLDING) ||
      node->GetOpDesc()->HasAttr(ATTR_NAME_IS_INSERTED_BY_CANN)) {
    usable = true;
    return SUCCESS;
  }
  for (auto &in_node : node->GetInDataNodes()) {
    auto in_node_opdesc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(in_node_opdesc);
    // near entrance of subgraph || near subgraph
    if ((in_node->GetType() == DATA && NodeUtils::IsSubgraphInput(in_node))
        || !in_node_opdesc->GetSubgraphInstanceNames().empty()) {
      usable = true;
      return SUCCESS;
    }

    GE_CHK_STATUS_RET(GetOriginalType(in_node, node_type),
                      "[Get][OriginalType] of node:%s failed", in_node->GetName().c_str());
    bool need_skip = (node_type != SWITCH) && (node_type != REFSWITCH);
    if (need_skip) {
      GELOGD("skip identity %s connected to switch", node->GetName().c_str());
      break;
    }
    GE_CHECK_NOTNULL(node->GetOutControlAnchor());
    if (!node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
      usable = true;
      return SUCCESS;
    }
  }
  for (auto &out_node : node->GetOutDataNodes()) {
    auto out_node_opdesc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(out_node_opdesc);
    // near output of subgraph || near subgraph
    const bool is_near_subgraph_out =
        (NodeUtils::IsSubgraphOutput(out_node)) || (!out_node_opdesc->GetSubgraphInstanceNames().empty());
    if (is_near_subgraph_out) {
      usable = true;
      return SUCCESS;
    }
    GE_CHK_STATUS_RET(GetOriginalType(out_node, node_type),
                      "[Get][OriginalType] of node:%s failed", out_node->GetName().c_str());
    if ((node_type != MERGE) && (node_type != REFMERGE)) {
      GELOGD("skip identity %s connected to merge", node->GetName().c_str());
      break;
    }
    GE_CHECK_NOTNULL(node->GetInControlAnchor());
    if (!node->GetInControlAnchor()->GetPeerOutControlAnchors().empty()) {
      usable = true;
      return SUCCESS;
    }
  }
  usable = IsIdentityUsefulForVarCache(node);
  return SUCCESS;
}
}  // namespace

Status IdentityPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "[Get][OriginalType] of node:%s failed.", node->GetName().c_str());
    return status_ret;
  }
  if ((type != IDENTITY) && (type != IDENTITYN) && (type != READVARIABLEOP)) {
    return SUCCESS;
  }

  if (!force_) {
    bool usable = false;
    auto ret = CheckIdentityUsable(node, usable);
    if (ret != SUCCESS) {
      return ret;
    }
    if (usable) {
      return SUCCESS;
    }
  }
  size_t n = node->GetOpDesc()->GetOutputsSize();
  if (node->GetOpDesc()->GetInputsSize() != n) {
    REPORT_INNER_ERR_MSG("E19999", "Num:%zu of input desc node:%s(%s) not equal to it's output desc num:%zu, "
                      "check invalid", node->GetOpDesc()->GetInputsSize(),
                      node->GetName().c_str(), node->GetType().c_str(), n);
    GELOGE(PARAM_INVALID, "[Check][Param] Num:%zu of input desc node:%s(%s) not equal to it's output desc num:%zu",
           node->GetOpDesc()->GetInputsSize(), node->GetName().c_str(), node->GetType().c_str(), n);
    return PARAM_INVALID;
  }
  std::vector<int32_t> io_map;
  for (size_t i = 0; i < n; i++) {
    io_map.push_back(i);
  }
  return SafelyRemoveIdentity(node, io_map);
}

Status IdentityPass::SafelyRemoveIdentity(NodePtr &node, const std::vector<int32_t> &io_map) {
  if (node->GetType() == READVARIABLEOP) {
    const bool read_first_then_write = NodeUtils::IsIdentityUsefulForRWControl(node);
    for (const auto &out_data_node : node->GetOutDataNodes()) {
      if (read_first_then_write) {
        // `change ctrl out trigger from read_var to it's out data node` is a dangerous move
        // we use `filter` to reduce risk of topo-loop
        GE_ASSERT_SUCCESS(GraphUtils::CopyOutCtrlEdges(node, out_data_node, node_filter));
        GELOGI("Node [%s %s]'s out ctrl relation is copyed to it's data consumer node [%s %s]", node->GetName().c_str(),
               node->GetType().c_str(), out_data_node->GetName().c_str(), out_data_node->GetType().c_str());
      } else {
        // try to `change ctrl in dependency from read_var to it's out data node` for `write first and then read` if
        // needed
        GE_ASSERT_SUCCESS(GraphUtils::CopyInCtrlEdges(node, out_data_node));
        GELOGI("Node [%s %s]'s in ctrl relation is copyed to it's data consumer node [%s %s]", node->GetName().c_str(),
               node->GetType().c_str(), out_data_node->GetName().c_str(), out_data_node->GetType().c_str());
      }
    }
  }
  return IsolateAndDeleteNode(node, io_map);
}

REG_PASS_OPTION("IdentityPass").LEVELS(OoLevel::kO3);
}  // namespace ge
