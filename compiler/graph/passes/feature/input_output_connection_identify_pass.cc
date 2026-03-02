/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/input_output_connection_identify_pass.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
Status InputOutputConnectionIdentifyPass::Run(ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Input param graph is nullptr, "
           "skip identification of nodes that connect to input and output.");
    return PARAM_INVALID;
  }

  if (graph->GetParentGraph() != nullptr) {
    GELOGD("Current graph %s is a subgraph, skip identification of nodes that connect to input and output.",
           graph->GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Start to identify nodes that connect to input and output.");
  if (graph->TopologicalSorting() != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Topological Sorting graph:%s failed", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][TopologicalSorting] for graph:%s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (GraphUtils::GetRefMapping(graph, symbol_to_anchors_, anchor_to_symbol_) != GRAPH_SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][RefMapping] for graph:%s failed.", graph->GetName().c_str());
    return INTERNAL_ERROR;
  }

  Node2Indexs connect_input_node_idx_map;
  Node2Indexs connect_output_node_idx_map;
  Status status = SUCCESS;
  for (const NodePtr &node : graph->GetDirectNode()) {
    // Not only node type Data is determined.
    if (OpTypeUtils::IsDataNode(node->GetType())) {
      GELOGD("Find nodes that connect to root graph input node: %s.", node->GetName().c_str());
      status = ProcessInputNode(node, connect_input_node_idx_map, connect_output_node_idx_map);
      if (status != SUCCESS) {
        GELOGE(status, "[Process][Nodes] that connect to input node:%s failed.", node->GetName().c_str());
        return status;
      }
    }

    if (node->GetType() == NETOUTPUT) {
      GELOGD("Find nodes that connect to root graph output node: %s.", node->GetName().c_str());
      status = ProcessOutputNode(node, connect_input_node_idx_map, connect_output_node_idx_map);
      if (status != SUCCESS) {
        GELOGE(status, "[Process][Nodes] that connect to output node:%s failed.", node->GetName().c_str());
        return status;
      }
    }
  }

  status = SetNodeAttrOfConnectingInputOutput(connect_input_node_idx_map, connect_output_node_idx_map);
  if (status != SUCCESS) {
    GELOGE(status, "[Set][Attr] for nodes that connect to input and output failed.");
    return status;
  }

  GELOGD("Success to identify nodes that connect to input and output.");
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::ProcessInputNode(const NodePtr &node, Node2Indexs &connect_input_node_idx,
                                                           Node2Indexs &connect_output_node_idx) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    // The return ptr of GetAllOutDataAnchors is always valid.
    const auto anchor_iter = anchor_to_symbol_.find(NodeIndexIO(node, out_data_anchor->GetIdx(), kOut).ToString());
    if (anchor_iter == anchor_to_symbol_.end()) {
      GELOGW("Current node: %s out_data_anchor: %d is invalid, can not find related symbol.", node->GetName().c_str(),
             out_data_anchor->GetIdx());
      continue;
    }

    const std::string &symbol = anchor_iter->second;
    auto status = UpdateNodeIdxMap(symbol, connect_input_node_idx, connect_output_node_idx);
    if (status != SUCCESS) {
      GELOGE(status, "[Call][UpdateNodeIdxMap] Failed to update node anchor_index map.");
      return status;
    }
  }
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::UpdateNodeIdxMap(const std::string &symbol_string,
                                                           Node2Indexs &connect_input_node_idx,
                                                           Node2Indexs &connect_output_node_idx) {
  const auto symbol_iter = symbol_to_anchors_.find(symbol_string);
  if (symbol_iter == symbol_to_anchors_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Can't find symbol:%s in symbol_to_anchors map, check invalid",
                      symbol_string.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Input param symbol std::string:%s is invalid.", symbol_string.c_str());
    return PARAM_INVALID;
  }
  const auto &node_index_io_list = symbol_iter->second;
  for (const auto &node_index_io : node_index_io_list) {
    if (node_index_io.io_type_ == kOut) {
      // find node that has shared output memory.
      connect_output_node_idx[node_index_io.node_ptr_].emplace_back(node_index_io.index_);
    } else {
      // find node that has shared input memory.
      connect_input_node_idx[node_index_io.node_ptr_].emplace_back(node_index_io.index_);
    }
  }
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::ProcessOutputNode(const NodePtr &node, Node2Indexs &connect_input_node_idx,
                                                            Node2Indexs &connect_output_node_idx) {
  GE_CHECK_NOTNULL(node);
  for (const auto in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    // The return ptr of GetAllInDataAnchors is always valid.
    const auto anchor_iter = anchor_to_symbol_.find(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn).ToString());
    if (anchor_iter == anchor_to_symbol_.end()) {
      GELOGW("Current node: %s in_data_anchor: %d is invalid, can not find related symbol.", node->GetName().c_str(),
             in_data_anchor->GetIdx());
      continue;
    }

    const std::string &symbol = anchor_iter->second;
    auto status = UpdateNodeIdxMap(symbol, connect_input_node_idx, connect_output_node_idx);
    if (status != SUCCESS) {
      GELOGE(status, "[Call][UpdateNodeIdxMap] Failed to update node anchor_index map.");
      return status;
    }
  }
  return SUCCESS;
}

Status InputOutputConnectionIdentifyPass::SetNodeAttrOfConnectingInputOutput(
    const Node2Indexs &connect_input_node_idx, const Node2Indexs &connect_output_node_idx) const {
  for (const auto &iter : connect_input_node_idx) {
    GE_CHECK_NOTNULL(iter.first);
    if (iter.first->GetOpDesc() != nullptr) {
      if (!AttrUtils::SetListInt(iter.first->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, iter.second)) {
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
                          iter.first->GetName().c_str(), iter.first->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_INPUT.c_str(),
               iter.first->GetName().c_str(), iter.first->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  for (const auto &iter : connect_output_node_idx) {
    GE_CHECK_NOTNULL(iter.first);
    if (iter.first->GetOpDesc() != nullptr) {
      if (!AttrUtils::SetListInt(iter.first->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, iter.second)) {
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_OUTPUT.c_str(),
                          iter.first->GetName().c_str(), iter.first->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_NODE_CONNECT_OUTPUT.c_str(),
               iter.first->GetName().c_str(), iter.first->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("InputOutputConnectionIdentifyPass").LEVELS(OoLevel::kO0);
}  // namespace ge
