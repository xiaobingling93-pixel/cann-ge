/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/folding_pass.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "framework/common/debug/ge_log.h"
#include "graph/ge_local_context.h"
#include "graph/tuning_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"

namespace ge {
namespace folding_pass {
shared_ptr<Kernel> GetKernelByType(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return nullptr;
  }
  KernelFactory &factory = KernelFactory::Instance();
  std::string type = node->GetType();
  if (type == FRAMEWORKOP) {
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) failed",
                        ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
                        node->GetName().c_str(), node->GetType().c_str());
      return nullptr;
    }
  }

  return factory.Create(type);
}
bool IsNoNeedConstantFolding(const NodePtr &node) {
  auto node_desc = node->GetOpDesc();
  return node_desc == nullptr || node_desc->HasAttr(ATTR_NO_NEED_CONSTANT_FOLDING);
}

// 判断节点是否是用户指定的不做常量折叠，通过属性标识；
// 当前支持TF设置_grappler_do_not_remove来指定节点不做常量折叠，Parser会转换成_do_not_constant_folding属性
bool IsUserSpecifiedSkipConstantFold(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  bool value = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_DO_NOT_CONSTANT_FOLDING, value);
  return value;
}
}  // namespace folding_pass

namespace {
IndexsToAnchors GetIndexAndPeerInDataAnchors(NodePtr &node) {
  IndexsToAnchors indexes_to_anchors;
  for (auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }
      const auto &peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }
      indexes_to_anchors[out_anchor->GetIdx()].push_back(peer_in_anchor);
    }
  }

  return indexes_to_anchors;
}

NodePtr AddConstNodeToGraph(GeTensorPtr &tensor, ComputeGraphPtr &graph) {
  auto const_desc = OpDescUtils::CreateConstOpZeroCopy(tensor);
  if (const_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Create Const op failed");
    GELOGE(OUT_OF_MEMORY, "[Create][ConstOp] failed");
    return nullptr;
  }

  GE_IF_BOOL_EXEC(graph == nullptr, GELOGW("input param graph is null"); return nullptr);
  (void) AttrUtils::SetListStr(const_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::move(std::vector<std::string>()));
  (void) AttrUtils::SetBool(const_desc, "_is_from_constant_folding", true);
  return graph->AddNodeFront(const_desc);
}

NodePtr AddIdentityNodeToGraph(const std::string &name, const GeTensorDesc &tensor, ComputeGraphPtr &graph,
                               const NodePtr &in_node) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Compute graph ptr is null in creating identity node.");
    return nullptr;
  }

  OpDescPtr desc = MakeShared<OpDesc>("", "");
  if (desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(MEMALLOC_FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }

  desc->SetName(name);
  ge::OpDescUtilsEx::SetType(desc, IDENTITY);
  auto ret = desc->AddInputDesc(tensor);
  auto ret2 = desc->AddOutputDesc(tensor);
  if ((ret != GRAPH_SUCCESS) || (ret2 != GRAPH_SUCCESS)) {
    REPORT_INNER_ERR_MSG("E19999", "Add input or output desc to op:%s(%s) failed",
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][GeTensorDesc] to op:%s(%s) failed",
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  return graph->InsertNode(in_node, desc);
}
}  // namespace

Status FoldingPass::Folding(NodePtr &node, std::vector<GeTensorPtr> &outputs) {
  GE_CHECK_NOTNULL(node);
  GELOGD("begin folding node:%s", node->GetName().c_str());
  // Before processing nodes, collect the relations between the out anchor and the peer out data nodes
  // to prepare for const reconnection
  auto indexes_to_anchors = GetIndexAndPeerInDataAnchors(node);

  auto ret = DealWithInNodes(node);
  if (ret != SUCCESS) {
    return ret;
  }
  if (AddConstNode(node, indexes_to_anchors, outputs) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  auto in_data_nodes = node->GetInDataNodes();
  OrderedNodeSet in_data_nodes_set(in_data_nodes.begin(), in_data_nodes.end());
  if (IsolateAndDeleteNode(node, {}) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[IsolateAndDelete][Node] %s(%s) failed.",
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  for (auto iter = in_data_nodes_set.begin(); iter != in_data_nodes_set.end(); ++iter) {
    auto pre_node = *iter;
    if (pre_node->GetOutDataNodesSize() == 0) {
      if (OpTypeUtils::IsDataNode(pre_node->GetType()) || (pre_node->GetType() == ENTER)) {
        GELOGI("No need to remove data/enter, node name:%s.", pre_node->GetName().c_str());
        continue;
      }
      if (IsolateAndDeleteNode(pre_node, {}) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Isolate and delete node:%s(%s) failed",
                           pre_node->GetName().c_str(), pre_node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[IsolateAndDelete][Node] %s(%s) failed.",
               pre_node->GetName().c_str(), pre_node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status FoldingPass::DealWithInNodes(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  auto graph = node->GetOwnerComputeGraph();
  auto in_data_anchors = node->GetAllInDataAnchors();
  for (auto &in_data_anchor : in_data_anchors) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    auto in_node_anchor = in_data_anchor->GetPeerOutAnchor();
    if (in_node_anchor == nullptr) {
      continue;
    }
    auto in_node = in_node_anchor->GetOwnerNode();
    if ((in_node->GetType() == SWITCH) || (in_node->GetType() == REFSWITCH)) {
      GELOGI("The in_node name is %s, and node type is %s.", in_node->GetName().c_str(), in_node->GetType().c_str());
      auto ret = in_node_anchor->Unlink(in_data_anchor);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Op:%s(%s) out index:%d unlink from op:%s(%s) in index:%d failed",
                          in_node->GetName().c_str(), in_node->GetType().c_str(), in_node_anchor->GetIdx(),
                          node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Unlink][Anchor] between const node:%s and constant-folding-node:%s(%s) failed.",
               in_node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
      GELOGI("Unlink anchor between in_node %s and node %s success.", in_node->GetName().c_str(),
             node->GetName().c_str());
      auto identity_name = node->GetName() + "_ctrl_identity_" + std::to_string(in_data_anchor->GetIdx());
      auto identity =
          AddIdentityNodeToGraph(identity_name,
          node->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx()), graph, in_node);
      if (identity == nullptr) {
        GELOGE(INTERNAL_ERROR, "[Add][IdentityNode] %s to graph:%s failed.",
               identity_name.c_str(), graph->GetName().c_str());
        return INTERNAL_ERROR;
      }
      // to avoid identity with empty tensor to be replace by emtpy const again, here mark attribute
      (void)AttrUtils::SetBool(identity->GetOpDesc(), ATTR_NAME_IS_INSERTED_BY_GE, true);
      ret = GraphUtils::AddEdge(in_node_anchor, identity->GetInDataAnchor(0));
      if (ret != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                          in_node->GetName().c_str(), in_node->GetType().c_str(), in_node_anchor->GetIdx(),
                          identity->GetName().c_str(), identity->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
               in_node->GetName().c_str(), in_node->GetType().c_str(), in_node_anchor->GetIdx(),
               identity->GetName().c_str(), identity->GetType().c_str());
        return INTERNAL_ERROR;
      }
      GELOGI("Create new identity node success.");
      ret = GraphUtils::AddEdge(identity->GetOutControlAnchor(), node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                          identity->GetName().c_str(), identity->GetType().c_str(),
                          node->GetName().c_str(), node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
               identity->GetName().c_str(), identity->GetType().c_str(),
               node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

Status FoldingPass::AddConstNode(const NodePtr &node, IndexsToAnchors indexes_to_anchors,
                                 std::vector<GeTensorPtr> &v_weight) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] node is nullptr");
    return FAILED;
  }
  auto graph = node->GetOwnerComputeGraph();
  for (auto &index_to_anchors : indexes_to_anchors) {
    auto index = static_cast<size_t>(index_to_anchors.first);
    if (index >= v_weight.size()) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%lu in param index_to_anchors >= param v_weight.size:%zu, "
                         "check invalid", index, v_weight.size());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to constant fold on node %s type %s, "
             "the out nodes num %lu calculated is less than the node out anchor index %zu",
             node->GetName().c_str(), node->GetType().c_str(), v_weight.size(), index);
      return INTERNAL_ERROR;
    }
    GeTensorPtr weight = v_weight[index];
    if (weight == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Index:%lu in param v_weight is nullptr check invalid", index);
      GELOGE(INTERNAL_ERROR,
             "[Check][Param] Failed to constant fold on node %s type %s, the %lust node calculated is null",
             node->GetName().c_str(), node->GetType().c_str(), index);
      return INTERNAL_ERROR;
    }

    auto const_node = AddConstNodeToGraph(weight, graph);
    if (const_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Add][ConstNode] To Graph failed, node name:%s, index:%zu.",
             node->GetName().c_str(), index);
      return INTERNAL_ERROR;
    }
    GELOGI("add const_node:%s, replace node %s, type %s, index %zu.", const_node->GetName().c_str(),
           node->GetName().c_str(), node->GetType().c_str(), index);

    // add new const to re-pass node
    for (auto &in_anchor : index_to_anchors.second) {
      if (in_anchor == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Index:%lu in param index_to_anchors has nullptr member in_anchor, "
                           "check invalid", index);
        GELOGE(INTERNAL_ERROR,
               "[Check][Param] Index:%lu in param index_to_anchors has nullptr member in_anchor", index);
        return INTERNAL_ERROR;
      }
      auto ret = ConnectNodeToInAnchor(in_anchor, const_node, 0);
      if (ret != SUCCESS) {
        return ret;
      }
      NodeUtils::UpdateIsInputConst(*(in_anchor->GetOwnerNode()));
    }
    Status ret = GraphUtils::AddEdge(node->GetOutControlAnchor(), const_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str(),
                        const_node->GetName().c_str(), const_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] failed, from node %s to const node %s.", node->GetName().c_str(),
             const_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string stream_label;
    if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
      GE_CHECK_NOTNULL(const_node->GetOpDesc());
      if (!AttrUtils::SetStr(const_node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_STREAM_LABEL.c_str(),
                          const_node->GetName().c_str(), const_node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_STREAM_LABEL.c_str(),
               const_node->GetName().c_str(), const_node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
    GELOGD("Add control edge when insert dynamic const, from node %s to const node %s, with stream label:%s.",
           node->GetName().c_str(), const_node->GetName().c_str(), stream_label.c_str());
  }

  return SUCCESS;
}

Status FoldingPass::RemoveNodeKeepingCtrlEdges(const NodePtr &node) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(PARAM_INVALID, "node is null"); return PARAM_INVALID);
  auto ret = GraphUtils::IsolateNode(node, {});
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Isolate node:%s(%s) in graph failed",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Isolate][Node] %s type %s failed", node->GetName().c_str(),
           node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  auto graph = node->GetOwnerComputeGraph();
  ret = GraphUtils::RemoveNodeWithoutRelink(graph, node);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] %s(%s) without relink in graph:%s failed",
           node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    return INTERNAL_ERROR;
  }
  AddNodeDeleted(node);
  return SUCCESS;
}

Status FoldingPass::ConnectNodeToInAnchor(const InDataAnchorPtr &in_anchor, const NodePtr &node, int32_t node_index) {
  // the origin edge must be removed before add
  if (in_anchor == nullptr || node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node or in_anchor is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] in anchor or node is null");
    return PARAM_INVALID;
  }
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    if (ge::GraphUtils::RemoveEdge(peer_out_anchor, in_anchor) != GRAPH_SUCCESS) {
      GELOGW("RemoveEdge failed.");
    }
  }

  auto new_out_anchor = node->GetOutDataAnchor(node_index);
  if (new_out_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param out index:%d data anchor of node:%s(%s) is nullptr, check invalid",
                       node_index, node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to add node to in anchor,"
           " the index %d for node %s, type %s is invalid",
           node_index, node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (GraphUtils::AddEdge(new_out_anchor, in_anchor) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      node->GetName().c_str(), node->GetType().c_str(), node_index,
                      in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetType().c_str(),
                      in_anchor->GetIdx());
    GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           node->GetName().c_str(), node->GetType().c_str(), node_index,
           in_anchor->GetOwnerNode()->GetName().c_str(), in_anchor->GetOwnerNode()->GetType().c_str(),
           in_anchor->GetIdx());
    return INTERNAL_ERROR;
  }
  AddRePassNodesWithInOut(node);
  return SUCCESS;
}
bool FoldingPass::IsResultSuccess(const Status compute_ret) const {
  if (compute_ret != SUCCESS && compute_ret != NOT_CHANGED && compute_ret != UNSUPPORTED) {
    return false;
  }
  return true;
}
}  // namespace ge
