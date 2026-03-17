/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/control_flow_and_stream/switch_dead_branch_elimination.h"

#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/omg_util/omg_util.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
const std::vector<int32_t>::size_type kDataInputIndex = 0;
const std::vector<int32_t>::size_type kPredInputIndex = 1;
const int32_t kDefaultInputIndex = -1;

bool ParsePred(const ConstGeTensorPtr &tensor) {
  if (tensor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param tensor is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter tensor is nullptr.");
    return false;
  }
  const uint8_t *data_ptr = tensor->GetData().data();
  auto type = tensor->GetTensorDesc().GetDataType();
  switch (type) {
    case DT_BOOL:
      return *reinterpret_cast<const bool *>(data_ptr);
    case DT_FLOAT:
      return static_cast<bool>(fabs(*reinterpret_cast<const float *>(data_ptr)) >
                               std::numeric_limits<float>::epsilon());
    case DT_DOUBLE:
      return static_cast<bool>(fabs(*reinterpret_cast<const double *>(data_ptr)) >
                               std::numeric_limits<double>::epsilon());
    case DT_INT8:
    case DT_UINT8:
    case DT_HIFLOAT8:
    case DT_FLOAT8_E5M2:
    case DT_FLOAT8_E4M3FN:
      return static_cast<bool>(*data_ptr);
    case DT_FLOAT16:
    case DT_INT16:
    case DT_UINT16:
      return static_cast<bool>(*reinterpret_cast<const int16_t *>(data_ptr));
    case DT_INT32:
    case DT_UINT32:
      return static_cast<bool>(*reinterpret_cast<const int32_t *>(data_ptr));
    case DT_INT64:
    case DT_UINT64:
      return static_cast<bool>(*reinterpret_cast<const int64_t *>(data_ptr));
    default:
      return static_cast<bool>(*data_ptr);
  }
}

bool ParseOutDataAnchors(const NodePtr &node, const NodePtr &pred_node, OutDataAnchorPtr &active_out_data_anchor,
                         OutDataAnchorPtr &inactive_out_data_anchor) {
  auto tensors = OpDescUtils::MutableWeights(pred_node);
  if (tensors.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no weight, check invalid",
                       pred_node->GetName().c_str(), pred_node->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] Node:%s(%s) has no weight",
           pred_node->GetName().c_str(), pred_node->GetType().c_str());
    return false;
  }

  bool pred_value = ParsePred(tensors[0]);
  int32_t inactive_output_index = pred_value ? 0 : 1;

  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return false;
  }
  GELOGI("[%s] Inactive output index = %d", node->GetName().c_str(), inactive_output_index);
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor->GetIdx() == inactive_output_index) {
      inactive_out_data_anchor = out_anchor;
    } else {
      active_out_data_anchor = out_anchor;
    }
  }

  return true;
}
}  // namespace

Status SwitchDeadBranchElimination::DeleteSwitchNode(NodePtr &node, NodePtr &pred_node,
                                                     const OutDataAnchorPtr &active_out_data_anchor) {
  if (node == nullptr || active_out_data_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node or active_out_data_anchor is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] parameter node or active_out_data_anchor is nullptr.");
    return FAILED;
  }

  // If two nodes aren't in same graph, get node's direct in_node instead of pred_node.
  if (node->GetOwnerComputeGraph() != pred_node->GetOwnerComputeGraph()) {
    pred_node = PassUtils::GetInDataNode(node, kPredInputIndex);
  }

  GE_ASSERT_NOTNULL(pred_node);
  // link pred's in control nodes to switch
  if (GraphUtils::CopyInCtrlEdges(pred_node, node) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Copy in control edge from node:%s(%s) to node:%s(%s) failed",
                      pred_node->GetName().c_str(), pred_node->GetType().c_str(),
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Copy][InCtrlEdges] from node:%s(%s) to node:%s(%s) failed",
           pred_node->GetName().c_str(), pred_node->GetType().c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  // Remove link between pred and switch
  auto in_pred_anchor = node->GetInDataAnchor(kPredInputIndex);
  GE_CHECK_NOTNULL(in_pred_anchor);
  in_pred_anchor->UnlinkAll();

  /// If condition Const is isolate, it will be delete with pruning
  /// Isolate Switch and delete it
  std::vector<int32_t> switch_io_map = {kDefaultInputIndex, kDefaultInputIndex};
  size_t out_index = static_cast<size_t>(active_out_data_anchor->GetIdx());
  if (out_index >= switch_io_map.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Out index:%zu of node:%s(%s) >= %zu, check invalid", out_index,
                       node->GetName().c_str(), node->GetType().c_str(), switch_io_map.size());
    GELOGE(FAILED, "[Check][Param] Out index:%zu of node:%s(%s) >= %zu.", out_index,
           node->GetName().c_str(), node->GetType().c_str(), switch_io_map.size());
    return FAILED;
  }
  switch_io_map[out_index] = kDataInputIndex;
  return IsolateAndDeleteNode(node, switch_io_map);
}

Status SwitchDeadBranchElimination::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Param [node] must not be null.");
    return PARAM_INVALID;
  }

  std::string op_type;
  GE_CHK_STATUS_RET(GetOriginalType(node, op_type),
                    "[Get][OriginalType] of node:%s failed", node->GetName().c_str());
  if ((op_type != SWITCH) && (op_type != REFSWITCH)) {
    return SUCCESS;
  }

  if (node->GetOutAllNodes().empty()) {
    return SUCCESS;
  }

  auto pred_node = PassUtils::GetInNodeCrossSubgraphByIndex(node, kPredInputIndex);
  if (pred_node == nullptr) {
    GELOGD("[%s] Pred input is null.", node->GetName().c_str());
    return SUCCESS;
  }

  // Can be optimized when pred is constant
  if (!PassUtils::IsConstant(pred_node)) {
    GELOGD("[%s] Pred is not constant.", node->GetName().c_str());
    return SUCCESS;
  }

  auto input_node = PassUtils::GetInDataNode(node, kDataInputIndex);
  if (input_node == nullptr) {
    GELOGD("[%s] Data input is null.", node->GetName().c_str());
    return SUCCESS;
  }

  // Get active & inactive output anchors by the value of pred
  OutDataAnchorPtr active_out_data_anchor = nullptr;
  OutDataAnchorPtr inactive_out_data_anchor = nullptr;
  if (!ParseOutDataAnchors(node, pred_node, active_out_data_anchor, inactive_out_data_anchor)) {
    return PARAM_INVALID;
  }

  if (inactive_out_data_anchor != nullptr) {
    GELOGI("[%s] To unlink inactive output %d", node->GetName().c_str(), inactive_out_data_anchor->GetIdx());
    std::vector<NodePtr> del_nodes;
    std::vector<NodePtr> end_nodes;
    Status ret = PassUtils::RemoveInactiveBranchToMerge(inactive_out_data_anchor, del_nodes, end_nodes);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Remove][InactiveBranch] from node:%s(%s) to merge failed",
             node->GetName().c_str(), node->GetType().c_str());
      return ret;
    }

    for (auto &end_node : end_nodes) {
      AddRePassNode(end_node);
    }
    std::unordered_set<NodePtr> delete_nodes_set;
    for (auto &delete_node : del_nodes) {
      if (delete_nodes_set.insert(delete_node).second &&
          IsolateAndDeleteNode(delete_node, std::vector<int32_t>()) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Remove inactive branch node:%s failed", delete_node->GetName().c_str());
        GELOGE(FAILED, "[Remove][InactiveBranch] node:%s failed", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  AddImmediateRePassNode(input_node);
  return DeleteSwitchNode(node, pred_node, active_out_data_anchor);
}

REG_PASS_OPTION("SwitchDeadBranchElimination").LEVELS(OoLevel::kO1).SWITCH_OPT(ge::OO_DEAD_CODE_ELIMINATION);
}  // namespace ge
