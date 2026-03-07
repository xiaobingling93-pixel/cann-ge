/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/iterator_op_pass.h"

#include <memory>
#include <sstream>
#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/anchor.h"
#include "common/omg_util/omg_util.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_context.h"
#include "graph/manager/util/rt_context_util.h"

namespace ge {
const int32_t kMaxIterationsPerLoop = INT32_MAX - 1;

Status IteratorOpPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  if (!PassUtils::IsNeedTrainIteFlowCtrl(graph)) {
    return SUCCESS;
  }

  GELOGD("GetNextOpPass begin");
  std::string type;
  for (ge::NodePtr &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(GetOriginalType(node, type));
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const std::string op_type = op_desc->GetType();
    if ((type == ITERATORV2) || (type == ITERATOR) || (op_type == GETNEXT)) {
      ge::NodePtr memcpy_async_node = InsertMemcpyAsyncNode(node, graph);
      GE_CHECK_NOTNULL(memcpy_async_node);
      auto status = SetCycleEvent(memcpy_async_node);
      if (status != ge::SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Set cycle event to op:%s(%s) failed",
                          memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str());
        GELOGE(status, "[Set][CycleEvent] to op:%s(%s) failed",
               memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str());
        return status;
      }

      status = SetStreamLabel(memcpy_async_node, memcpy_async_node->GetName());
      if (status != ge::SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Set stream label:%s to op:%s(%s) failed",
                          memcpy_async_node->GetName().c_str(), memcpy_async_node->GetName().c_str(),
                          memcpy_async_node->GetType().c_str());
        GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
               memcpy_async_node->GetName().c_str(), memcpy_async_node->GetName().c_str(),
               memcpy_async_node->GetType().c_str());
        return status;
      }

      status = SetStreamLabel(node, node->GetName());
      if (status != ge::SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Set stream label:%s to op:%s(%s) failed",
                          node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        GELOGE(status, "[Set][StreamLabel] %s to op:%s(%s) failed",
               node->GetName().c_str(), node->GetName().c_str(), node->GetType().c_str());
        return status;
      }

      GELOGI("Set independent loop for iterator node success");
    }
  }
  GELOGD("GetNextOpPass end");
  return SUCCESS;
}

///
/// @brief insert memcpy after GetNext
///
/// @param pre_node
/// @param graph
/// @return ge::NodePtr
///
ge::NodePtr IteratorOpPass::InsertMemcpyAsyncNode(const ge::NodePtr &pre_node, const ge::ComputeGraphPtr &graph) {
  GE_CHK_BOOL_EXEC(pre_node != nullptr, GELOGW("Pre node is null."); return nullptr);
  GE_CHK_BOOL_EXEC(graph != nullptr, GELOGW("graph is null."); return nullptr);
  ge::OpDescPtr memcpy_async_op_desc = CreateMemcpyAsyncOp(pre_node);
  GE_CHK_BOOL_EXEC(memcpy_async_op_desc != nullptr, GELOGW("Create memcpyAsync op fail."); return nullptr);
  auto memcpy_async_node = graph->InsertNode(pre_node, memcpy_async_op_desc);
  GE_CHK_BOOL_EXEC(memcpy_async_node != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s failed",
                                     memcpy_async_op_desc->GetName().c_str(), memcpy_async_op_desc->GetType().c_str(),
                                     graph->GetName().c_str());
                   return nullptr,
                   "[Add][Node] %s(%s) to graph:%s failed", memcpy_async_op_desc->GetName().c_str(),
                   memcpy_async_op_desc->GetType().c_str(), graph->GetName().c_str());

  // Data out
  for (auto &out_anchor : pre_node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    ge::graphStatus status;
    GELOGI("Graph add memcpyAsync op in edge, index:%d.", out_anchor->GetIdx());
    for (auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(peer_in_anchor == nullptr, GELOGW("peer_in_anchor is nullptr"); return nullptr);
      status = GraphUtils::RemoveEdge(out_anchor, peer_in_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E19999",
                                         "Remove edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                         pre_node->GetName().c_str(),
                                         pre_node->GetType().c_str(), out_anchor->GetIdx(),
                                         peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                                         peer_in_anchor->GetOwnerNode()->GetType().c_str(),
                                         peer_in_anchor->GetIdx());
                       return nullptr,
                       "[Remove][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                       pre_node->GetName().c_str(), pre_node->GetType().c_str(), out_anchor->GetIdx(),
                       peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                       peer_in_anchor->GetOwnerNode()->GetType().c_str(),
                       peer_in_anchor->GetIdx());
      status = GraphUtils::AddEdge(memcpy_async_node->GetOutDataAnchor(out_anchor->GetIdx()), peer_in_anchor);
      GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E19999",
                                         "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                         memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                                         out_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                                         peer_in_anchor->GetOwnerNode()->GetType().c_str(), peer_in_anchor->GetIdx());
                       return nullptr,
                       "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                       memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                       out_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                       peer_in_anchor->GetOwnerNode()->GetType().c_str(), peer_in_anchor->GetIdx());
      GELOGI("Graph add memcpyAsync op out edge, src index:%d, dst index:%d, dst node: %s.", out_anchor->GetIdx(),
             peer_in_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetName().c_str());
    }
    status = GraphUtils::AddEdge(out_anchor, memcpy_async_node->GetInDataAnchor(out_anchor->GetIdx()));
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999",
                                       "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                       pre_node->GetName().c_str(), pre_node->GetType().c_str(), out_anchor->GetIdx(),
                                       memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                                       out_anchor->GetIdx());
                     return nullptr,
                     "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                     pre_node->GetName().c_str(), pre_node->GetType().c_str(), out_anchor->GetIdx(),
                     memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                     out_anchor->GetIdx());
  }
  // Control out
  OutControlAnchorPtr out_ctrl_anchor = pre_node->GetOutControlAnchor();
  GE_IF_BOOL_EXEC(out_ctrl_anchor != nullptr,
      for (auto &peer_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
    ge::graphStatus status = GraphUtils::RemoveEdge(out_ctrl_anchor, peer_in_ctrl_anchor);
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999",
                                       "Remove control edge between op:%s(%s) and op:%s(%s) failed",
                                       pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
                     return nullptr,
                     "[Remove][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                     pre_node->GetName().c_str(), pre_node->GetType().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    status = GraphUtils::AddEdge(memcpy_async_node->GetOutControlAnchor(), peer_in_ctrl_anchor);
    GE_CHK_BOOL_EXEC(status == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                                       memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                       peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
                     return nullptr,
                     "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
                     memcpy_async_node->GetName().c_str(), memcpy_async_node->GetType().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_in_ctrl_anchor->GetOwnerNode()->GetType().c_str());
    GELOGI("Graph add memcpyAsync op out ctrl edge, dst node: %s.",
           peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
  });
  GELOGI("Insert memcpyAsync op success.");

  return memcpy_async_node;
}

///
/// @brief create memcpy
///
/// @param pre_node
/// @return ge::OpDescPtr
///
ge::OpDescPtr IteratorOpPass::CreateMemcpyAsyncOp(const ge::NodePtr &pre_node) const {
  GE_CHK_BOOL_EXEC(pre_node != nullptr, return nullptr, "Input param invalid.");

  std::string node_name = pre_node->GetName() + "_MemcpyAsync";
  ge::OpDescPtr op_desc = MakeShared<OpDesc>(node_name.c_str(), MEMCPYASYNC);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(FAILED, "[New][OpDesc] failed");
    return op_desc;
  }
  GELOGI("Create memcpyAsync op:%s.", op_desc->GetName().c_str());

  ge::OpDescPtr pre_node_op_desc = pre_node->GetOpDesc();
  GE_CHK_BOOL_EXEC(pre_node_op_desc != nullptr,
                   REPORT_INNER_ERR_MSG("E19999", "OpDesc in node is nullptr, check invalid");
                   return nullptr, "[Get][OpDesc] failed, OpDesc of pre_node is invalid.");

  size_t out_size = pre_node_op_desc->GetOutputsSize();
  GELOGI("Create memcpyAsync op, pre_node out_size: %zu.", out_size);
  for (size_t i = 0; i < out_size; i++) {
    GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(pre_node_op_desc->GetOutputDesc(i)) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999", "Add input desc to op:%s(%s) failed",
                                       pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
                     return nullptr,
                     "[Add][InputDesc] to op:%s(%s) failed",
                     pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
    GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(pre_node_op_desc->GetOutputDesc(i)) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E19999", "Add output desc to op:%s(%s) failed",
                                       pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
                     return nullptr,
                     "[Add][OutputDesc] to op:%s(%s) failed",
                     pre_node_op_desc->GetName().c_str(), pre_node_op_desc->GetType().c_str());
  }

  return op_desc;
}

REG_PASS_OPTION("IteratorOpPass").LEVELS(OoLevel::kO1);
}  // namespace ge
