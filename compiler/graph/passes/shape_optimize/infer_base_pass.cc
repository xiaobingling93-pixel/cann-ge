/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_base_pass.h"
#include "common/plugin/ge_make_unique_util.h"
#include "base/err_msg.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
graphStatus FindValidSubgraphNetoutput(const ConstNodePtr &node, const ComputeGraphPtr &sub_graph, NodePtr &netoutput) {
  for (auto &sub_node : sub_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(sub_node);
    if (sub_node->GetType() == NETOUTPUT) {
      auto sub_node_opdesc = sub_node->GetOpDesc();
      if (sub_node_opdesc == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Invalid NetOutput node in subgraph %s, parent node %s, no OpDesc on it",
                           sub_graph->GetName().c_str(), node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Invalid NetOutput node on sub graph %s, parent node %s, no OpDesc on it",
               sub_graph->GetName().c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }

      netoutput = sub_node;
      return GRAPH_SUCCESS;
    }
  }

  REPORT_INNER_ERR_MSG("E19999", "Can not find the NetOutput node in subgraph %s, parent node %s",
                     sub_graph->GetName().c_str(), node->GetName().c_str());
  GELOGE(GRAPH_FAILED, "[Check][Param] Can not find the NetOutput node in subgraph %s, parent node %s",
         sub_graph->GetName().c_str(), node->GetName().c_str());
  return GRAPH_FAILED;
}
}  // namespace

Status InferBasePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  bool need_infer = NeedInfer(node);
  if (!need_infer) {
    GELOGD("Node %s does not need to infer.", node->GetName().c_str());
    return SUCCESS;
  }

  OrderedNodeSet changed_nodes;
  auto ret = InferAndUpdate(node, !OptionExists(kOptimizeAfterSubGraph), changed_nodes);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Call][InferAndUpdate] for node %s failed! ret: %u", node->GetName().c_str(), ret);
    return GRAPH_FAILED;
  }

  AddChangedNodesImmediateRepass(changed_nodes);
  return SUCCESS;
}

bool InferBasePass::NeedInfer(const NodePtr &node) const {
  (void)node;
  return true;
}
void InferBasePass::AddChangedNodesImmediateRepass(const OrderedNodeSet &changed_nodes) {
// need passed_nodes set to solve the problem that multi-input operators do repass in advance.
// when there is passed_nodes set, wo should call AddImmediateRePassNode for all nodes in changed_nodes.
  for (const auto &node_ele : changed_nodes) {
    AddImmediateRePassNode(node_ele);
  }
}

graphStatus InferBasePass::InferAndUpdate(NodePtr &node, bool before_subgraph, OrderedNodeSet &changed_nodes) {
  graphStatus ret;
  if (ContainsSubgraph(node)) {
    if (before_subgraph) {
      ret = UpdateTensorDescToSubgraphData(node);
    } else {
      ret = UpdateTensorDescToParentNodeOutput(node);
    }
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "[Update][TensorDesc] failed between parent node %s and subgraphs. ret: %u", node->GetName().c_str(),
             ret);
      return ret;
    }
  }

  PrintInOutTensors(node, "before_infer");
  ret = Infer(node);
  PrintInOutTensors(node, "after_infer");
  if (ret == GRAPH_NODE_NEED_REPASS) {
    // if a node need re_pass, it is not necessary to update peer node input.
    changed_nodes.insert(node);
    return GRAPH_SUCCESS;
  } else if (ret != GRAPH_SUCCESS && ret != GRAPH_NOT_CHANGED) {
    GELOGE(ret, "[Call][Infer] failed for node %s, ret: %u", node->GetName().c_str(), ret);
    return ret;
  }

  ret = UpdateTensorDescToPeerInputs(node, changed_nodes);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Update][TensorDesc] for Node %s peer input nodes failed! ret: %u", node->GetName().c_str(), ret);
  }
  GELOGD("Node %s infer and update succeeded .", node->GetName().c_str());
  return ret;
}

bool InferBasePass::ContainsSubgraph(const NodePtr &node) const {
  auto sub_graph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
  return !sub_graph_names.empty();
}

graphStatus InferBasePass::UpdateTensorDescToPeerInputs(const NodePtr &node, OrderedNodeSet &changed_nodes) {
  auto op_desc = node->GetOpDesc();
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
      auto peer_anchor_opdesc = peer_anchor->GetOwnerNode()->GetOpDesc();
      if (peer_anchor_opdesc == nullptr) {
        continue;
      }
      auto peer_input_desc = peer_anchor_opdesc->MutableInputDesc(peer_anchor->GetIdx());
      if (peer_input_desc == nullptr) {
        continue;
      }

      bool changed = false;
      auto ret = UpdateTensorDesc(output_tensor, peer_input_desc, changed);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(ret, "[Update][TensorDesc] failed, node %s.", node->GetName().c_str());
        return ret;
      }
      if (changed) {
        changed_nodes.insert(peer_anchor->GetOwnerNode());
        GELOGD("Node %s update peer node succeeded, peer node %s is changed.", node->GetName().c_str(),
               peer_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
  }
  return GRAPH_SUCCESS;
}

std::vector<ComputeGraphPtr> InferBasePass::GetCurNodeSubgraphs(const NodePtr &node) const {
  std::vector<ComputeGraphPtr> cur_node_subgraph;
  auto op_desc = node->GetOpDesc();
  auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return cur_node_subgraph;
  }

  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  for (const auto &name : sub_graph_names) {
    if (name.empty()) {
      GELOGW("The node %s contains empty subgraph instance name", node->GetName().c_str());
      continue;
    }
    auto sub_graph = root_graph->GetSubgraph(name);
    if (sub_graph == nullptr) {
      GELOGW("The subgrpah %s for node %s is null.", name.c_str(), node->GetName().c_str());
      continue;
    }
    cur_node_subgraph.emplace_back(sub_graph);
  }
  return cur_node_subgraph;
}

bool InferBasePass::CheckSkipUpdateForMultiDims(const OpDescPtr &op_desc) const {
  std::vector<int64_t> ori_dims;
  bool has_attr = AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, ori_dims);
  if (!has_attr) {
    return false;
  }
  for (const auto dim : ori_dims) {
    if (dim < 0) {
      GELOGD("While updating subgraph data node, ignore multi-dims node %s which has dynamic original shape.",
             op_desc->GetName().c_str());
      return true;
    }
  }
  return false;
}

graphStatus InferBasePass::UpdateTensorDescToSubgraphData(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  for (const auto &sub_graph : GetCurNodeSubgraphs(node)) {
    for (const auto &node_sub : sub_graph->GetDirectNode()) {
      if (node_sub->GetType() != DATA) {
        continue;
      }

      const auto &data_opdesc = node_sub->GetOpDesc();
      if (data_opdesc == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Invalid data node on the sub graph %s parent node %s, no OpDesc",
                           sub_graph->GetName().c_str(), node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][OpDesc] Invalid data node on the sub graph %s parent node %s, no OpDesc",
               sub_graph->GetName().c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }
      int32_t ref_i;
      if (!AttrUtils::GetInt(data_opdesc, ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        REPORT_INNER_ERR_MSG("E19999", "Invalid data node on the sub graph %s parent node %s, no ref-index attribute",
                           sub_graph->GetName().c_str(), node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][Int] Invalid data node on the sub graph %s parent node %s, no ref-index attribute",
               sub_graph->GetName().c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }
      GELOGD("Subgraph Data node ref_index is %d, parent node is %s.", ref_i, node->GetName().c_str());

      // In multi-batch, data shape of subgraph is different, no need to refresh.
      if (CheckSkipUpdateForMultiDims(data_opdesc)) {
        continue;
      }
      auto input_desc = op_desc->MutableInputDesc(ref_i);
      if (input_desc == nullptr) {
        REPORT_INNER_ERR_MSG("E19999",
                           "The ref index(%d) on the data %s on the sub graph %s "
                           "parent node %s are incompatible, inputs num %u",
                           ref_i, node_sub->GetName().c_str(), sub_graph->GetName().c_str(), node->GetName().c_str(),
                           node->GetAllInDataAnchorsSize());
        GELOGE(GRAPH_FAILED,
               "[Call][MutableInputDesc] The ref index(%d) on the data %s on the sub graph %s "
               "parent node %s are incompatible, inputs num %u",
               ref_i, node_sub->GetName().c_str(), sub_graph->GetName().c_str(), node->GetName().c_str(),
               node->GetAllInDataAnchorsSize());
        return GRAPH_FAILED;
      }
      GELOGI("Ref index is %d, input_desc dtype is %d, node name is %s", ref_i, input_desc->GetDataType(),
             node->GetName().c_str());

      bool has_tensor_desc_changed = false;
      auto data_input_td = data_opdesc->MutableInputDesc(0);
      auto ret = UpdateTensorDesc(input_desc, data_input_td, has_tensor_desc_changed);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Update][InputDesc] of data %s on the sub graph %s parent node %s failed",
               node_sub->GetName().c_str(), sub_graph->GetName().c_str(), node->GetName().c_str());
        return ret;
      }

      auto data_output_td = data_opdesc->MutableOutputDesc(0);
      ret = UpdateTensorDesc(input_desc, data_output_td, has_tensor_desc_changed);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Update][OutputDesc] of data %s on the sub graph %s parent node %s failed",
               node_sub->GetName().c_str(), sub_graph->GetName().c_str(), node->GetName().c_str());
        return ret;
      }
      GELOGD("Parent node %s update subgraph data %s input and output succeed.", node->GetName().c_str(),
             data_opdesc->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus InferBasePass::UpdateTensorDescToParentNodeOutput(const NodePtr &node) {
  std::vector<std::vector<GeTensorDescPtr>> ref_out_tensors(node->GetAllOutDataAnchorsSize());

  for (const auto &sub_graph : GetCurNodeSubgraphs(node)) {
    NodePtr netoutput;
    auto ret = FindValidSubgraphNetoutput(node, sub_graph, netoutput);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }

    auto netoutput_opdesc = netoutput->GetOpDesc();
    for (auto &netoutput_in_anchor : netoutput->GetAllInDataAnchors()) {
      auto netoutput_in_desc = netoutput_opdesc->MutableInputDesc(netoutput_in_anchor->GetIdx());
      if (netoutput_in_desc == nullptr) {
        REPORT_INNER_ERR_MSG("E19999",
                           "Invalid NetOutput node on sub graph %s, parent node %s, can not find input tensor %d",
                           sub_graph->GetName().c_str(), node->GetName().c_str(), netoutput_in_anchor->GetIdx());
        GELOGE(GRAPH_FAILED,
               "[Get][Tensor] Invalid NetOutput node on sub graph %s, parent node %s, can not find input tensor %d",
               sub_graph->GetName().c_str(), node->GetName().c_str(), netoutput_in_anchor->GetIdx());
        return GRAPH_FAILED;
      }
      GELOGI("Netoutput %s in anchor index is %d, input tensor dim is %zu", netoutput->GetNamePtr(),
             netoutput_in_anchor->GetIdx(),
             netoutput_in_desc->GetShape().GetDimNum());
      int32_t ref_i;
      if (!AttrUtils::GetInt(netoutput_in_desc, ATTR_NAME_PARENT_NODE_INDEX, ref_i)) {
        // if there is no ref index on the TensorDesc, it means the output data will be ignored outer.
        continue;
      }
      GELOGI("Parent node %s index of edge desc is %d", node->GetNamePtr(), ref_i);
      if (ref_i < 0 || static_cast<uint32_t>(ref_i) >= node->GetAllOutDataAnchorsSize()) {
        REPORT_INNER_ERR_MSG("E19999",
                           "Invalid ref_index %d of parent node %s, ref_index should less than %u.", ref_i,
                           node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
        GELOGE(GRAPH_FAILED,
               "[Get][Ref_index] Invalid ref_index %d of parent node %s, ref_index should less than %u.", ref_i,
               node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
        return GRAPH_FAILED;
      }
      ref_out_tensors[ref_i].emplace_back(netoutput_in_desc);
    }
  }

  return UpdateParentNodeContainsSubgraphs(node, ref_out_tensors);
}

graphStatus InferBasePass::UpdateParentNodeContainsSubgraphs(
    const NodePtr &node, const std::vector<std::vector<GeTensorDescPtr>> &ref_out_tensors) {
  for (size_t i = 0; i < ref_out_tensors.size(); i++) {
    if (ref_out_tensors[i].empty()) {
      REPORT_INNER_ERR_MSG("E19999", "Parent node %s ref_index %zu subgraph output tensor list is empty.",
                        node->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Param][check] Parent node %s ref_index %zu subgraph output tensor list is empty.",
             node->GetName().c_str(), i);
      return GRAPH_FAILED;
    }
    auto node_op_desc = node->GetOpDesc();
    auto node_output_td = node_op_desc->MutableOutputDesc(i);
    if (node_output_td == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Node %s output %zu tensor desc is null.", node->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Param][check] Node %s output %zu tensor desc is null.", node->GetName().c_str(), i);
      return GRAPH_FAILED;
    }

    graphStatus ret;
    if (node_op_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
      bool is_subgraph_dyn_dims = false;
      if (AttrUtils::GetBool(node->GetOwnerComputeGraph(), ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS, is_subgraph_dyn_dims) &&
          is_subgraph_dyn_dims) {
        ret = UpdateOutputFromSubgraphsForSubgraphMultiDims(ref_out_tensors[i], node_output_td);
      } else {
        ret = UpdateOutputFromSubgraphsForMultiDims(ref_out_tensors[i], node_output_td);
      }
    } else {
      ret = UpdateOutputFromSubgraphs(ref_out_tensors[i], node_output_td);
    }
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Node %s update output %zu tensor desc failed. ret: %u", node->GetName().c_str(), i,
                        ret);
      GELOGE(GRAPH_FAILED, "[Param][check] Node %s update output %zu tensor desc failed. ret: %u",
             node->GetName().c_str(), i, ret);
      return ret;
    }
    GELOGD("Parent node %s successfully updated the output tensors from subgraphs.", node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

void InferBasePass::PrintInOutTensors(const NodePtr &node, const std::string &phase) const {
  if (!IsLogEnable(GE, DLOG_INFO)) {
    return;
  }

  ge::OpDescPtr op_desc = node->GetOpDesc();
  std::stringstream ss;
  ss << "{";
  int32_t in_idx = 0;
  for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
    if (input_desc == nullptr) {
      in_idx++;
      continue;
    }
    ss << "input " << in_idx << ": ";
    ss << SerialTensorInfo(input_desc);
    ss << ";";
    in_idx++;
  }
  int32_t out_idx = 0;
  for (const auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
    if (output_desc == nullptr) {
      out_idx++;
      continue;
    }
    ss << "output " << out_idx << ": ";
    ss << SerialTensorInfo(output_desc);
    ss << ";";
    out_idx++;
  }
  ss << "}";
  GELOGI("[%s] Node [%s]: %s", phase.c_str(), node->GetName().c_str(), ss.str().c_str());
}
}  // namespace ge
