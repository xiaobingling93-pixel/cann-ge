/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "dtype_consistency.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"
#include "autofuse/utils/autofuse_attrs.h"
#include "graph_pass/pass_utils.h"

using namespace ge::ascir_op;

namespace {
Status RemoveDuplicateCast(const ge::AscNodePtr &keep_cast, const ge::AscNodePtr &remove_cast,
                           const ge::OutDataAnchorPtr &src_out_anchor) {
  auto keep_out_anchor = keep_cast->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(keep_out_anchor);

  auto remove_out_anchor = remove_cast->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(remove_out_anchor);

  // Reconnect remove_cast's downstream to keep_cast
  auto dst_anchors = remove_out_anchor->GetPeerInDataAnchors();
  for (auto &dst_anchor : dst_anchors) {
    GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(remove_out_anchor, dst_anchor));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(keep_out_anchor, dst_anchor));
  }

  // Remove remove_cast
  auto remove_in_anchor = remove_cast->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(remove_in_anchor);
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(src_out_anchor, remove_in_anchor));
  auto owner_graph = remove_cast->GetOwnerComputeGraph();
  GE_ASSERT_SUCCESS(owner_graph->RemoveNode(remove_cast));

  return ge::SUCCESS;
}
}  // namespace

namespace optimize {
Status DtypeConsistency::EnsureDtypeConsistency(ge::AscGraph &graph) {
  std::vector<NodeDtypeRequirement> requirements;
  GE_ASSERT_SUCCESS(CollectDtypeRequirements(graph, requirements), "Failed to collect dtype requirements");

  GE_ASSERT_SUCCESS(ApplyDtypeConversions(graph, requirements), "Failed to apply dtype conversions");

  GE_ASSERT_SUCCESS(CancelRedundantCast(graph), "Failed to cancel redundant cast");
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  return ge::SUCCESS;
}

Status DtypeConsistency::CollectDtypeRequirements(ge::AscGraph &graph,
                                                  std::vector<NodeDtypeRequirement> &requirements) {
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (ScheduleUtils::IsBuffer(node) || ge::ops::IsOps<Cast>(node)) {
      continue;
    }
    GE_ASSERT_NOTNULL(node);
    auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
    GE_ASSERT_NOTNULL(codegen_impl, "Cannot find impl for ir type:[%s].", node->GetTypePtr());
    auto [input_dtypes, output_dtypes] = codegen_impl->GetConversionDtype(*node);
    requirements.push_back({node, input_dtypes, output_dtypes});
  }
  return ge::SUCCESS;
}

Status DtypeConsistency::ApplyDtypeConversions(ge::AscGraph &graph,
                                               const std::vector<NodeDtypeRequirement> &requirements) {
  for (const auto &req : requirements) {
    GE_CHK_STATUS_RET(ProcessOutputDtype(req), "Failed to process output dtype");
    GE_CHK_STATUS_RET(ProcessInputDtype(graph, req), "Failed to process input dtype for node %s.",
                      req.node->GetNamePtr());
  }
  return ge::SUCCESS;
}

Status DtypeConsistency::ProcessOutputDtype(const NodeDtypeRequirement &req) {
  size_t output_nums = req.node->outputs().size();
  for (size_t i = 0UL; i < output_nums; ++i) {
    GE_ASSERT_TRUE(i < req.output_dtypes.size());
    if (req.node->outputs[i].attr.dtype != req.output_dtypes[i]) {
      GELOGD("Node [%s]'s output[%zu] need to change dtype from [%s] to [%s].", req.node->GetNamePtr(), i,
             ge::TypeUtils::DataTypeToSerialString(req.node->outputs[i].attr.dtype).c_str(),
             ge::TypeUtils::DataTypeToSerialString(req.output_dtypes[i]).c_str());
      req.node->outputs[i].attr.dtype = req.output_dtypes[i];
    }
  }
  return ge::SUCCESS;
}

Status DtypeConsistency::ProcessInputDtype(ge::AscGraph &graph, const NodeDtypeRequirement &req) {
  size_t input_nums = req.node->inputs().size();
  for (size_t i = 0UL; i < input_nums; ++i) {
    GE_ASSERT_TRUE(i < req.input_dtypes.size());
    if (req.input_dtypes[i] == req.node->inputs[i].attr.dtype) {
      continue;
    }

    auto in_anchor = req.node->GetInDataAnchor(static_cast<int32_t>(i));
    GE_ASSERT_NOTNULL(in_anchor);
    auto src_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_out_anchor);

    auto src_node = std::dynamic_pointer_cast<ge::AscNode>(src_out_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(src_node);

    auto src_dtype = src_node->outputs[static_cast<int32_t>(src_out_anchor->GetIdx())].attr.dtype;
    auto dst_dtype = req.input_dtypes[i];

    // Check if cast conversion is supported
    GE_CHK_STATUS_RET(CheckCastSupported(src_dtype, dst_dtype, req.node, i), "Node:%s check cast supported failed.",
                      req.node->GetNamePtr());

    // If upstream is already a cast, try to merge two casts
    if (ge::ops::IsOps<Cast>(src_node)) {
      if (TryMergeWithUpstreamCast(graph, src_node, req.node, i, dst_dtype)) {
        continue;
      }
    }

    // Insert a new cast node
    GE_CHK_STATUS_RET(InsertCastNode(graph, src_node, req.node, i, dst_dtype), "Insert cast node failed");
  }
  return ge::SUCCESS;
}

Status DtypeConsistency::CheckCastSupported(ge::DataType src_dtype, ge::DataType dst_dtype, const ge::AscNodePtr &node,
                                            size_t input_idx) {
  std::vector<ge::DataType> cast_input_dtypes = {src_dtype};
  std::vector<ge::DataType> cast_output_dtypes = {dst_dtype};
  auto infer_ret = ScheduleUtils::CallAscirInferDataType<ge::ascir_op::Cast>(cast_input_dtypes, cast_output_dtypes);
  if (infer_ret != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Failed to insert cast for node [%s] input [%zu]: cast from [%s] to [%s] is not supported.",
           node->GetNamePtr(), input_idx, ge::TypeUtils::DataTypeToSerialString(src_dtype).c_str(),
           ge::TypeUtils::DataTypeToSerialString(dst_dtype).c_str());
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

bool DtypeConsistency::TryMergeWithUpstreamCast(ge::AscGraph &graph, const ge::AscNodePtr &upstream_cast,
                                                const ge::AscNodePtr &downstream_node, size_t input_idx,
                                                ge::DataType target_dtype) {
  auto orig_src_dtype = upstream_cast->inputs[0].attr.dtype;
  std::vector<ge::DataType> merge_input_dtypes = {orig_src_dtype};
  std::vector<ge::DataType> merge_output_dtypes = {target_dtype};
  if ((orig_src_dtype != target_dtype) &&
    ScheduleUtils::CallAscirInferDataType<ge::ascir_op::Cast>(merge_input_dtypes, merge_output_dtypes) != ge::SUCCESS) {
    return false;
  }

  // Check if the original cast has multiple downstream consumers
  auto orig_cast_out_anchor = upstream_cast->GetOutDataAnchor(0);
  auto orig_cast_peer_anchors = orig_cast_out_anchor->GetPeerInDataAnchorsPtr();
  bool has_multiple_consumers = orig_cast_peer_anchors.size() > 1U;

  auto in_anchor = downstream_node->GetInDataAnchor(static_cast<int32_t>(input_idx));
  GE_ASSERT_NOTNULL(in_anchor);

  if (!has_multiple_consumers) {
    // Only one downstream consumer, can directly modify the original cast's output dtype
    GELOGD("Merge cast (single consumer) for node [%s] input [%zu]: [%s] -> [%s] -> [%s] => [%s] -> [%s]",
           downstream_node->GetNamePtr(), input_idx, ge::TypeUtils::DataTypeToSerialString(orig_src_dtype).c_str(),
           ge::TypeUtils::DataTypeToSerialString(upstream_cast->outputs[0].attr.dtype).c_str(),
           ge::TypeUtils::DataTypeToSerialString(target_dtype).c_str(),
           ge::TypeUtils::DataTypeToSerialString(orig_src_dtype).c_str(),
           ge::TypeUtils::DataTypeToSerialString(target_dtype).c_str());
    upstream_cast->outputs[0].attr.dtype = target_dtype;
    downstream_node->inputs[input_idx].attr.dtype = target_dtype;
    return true;
  }

  // Multiple downstream consumers, create a new merged cast and disconnect the edge
  GELOGD("Merge cast (multiple consumers) for node [%s] input [%zu]: [%s] -> [%s] -> [%s] => [%s] -> [%s]",
         downstream_node->GetNamePtr(), input_idx, ge::TypeUtils::DataTypeToSerialString(orig_src_dtype).c_str(),
         ge::TypeUtils::DataTypeToSerialString(upstream_cast->outputs[0].attr.dtype).c_str(),
         ge::TypeUtils::DataTypeToSerialString(target_dtype).c_str(),
         ge::TypeUtils::DataTypeToSerialString(orig_src_dtype).c_str(),
         ge::TypeUtils::DataTypeToSerialString(target_dtype).c_str());

  // Get the original cast's input
  auto orig_cast_in_anchor = upstream_cast->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(orig_cast_in_anchor);
  auto orig_src_out_anchor = orig_cast_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(orig_src_out_anchor);

  // Create a new merged cast
  std::string merged_cast_name = std::string(upstream_cast->GetName()) + "_merged_to_" + downstream_node->GetName();
  Cast merged_cast_node(merged_cast_name.c_str());
  auto merged_cast_ptr = graph.AddNode(merged_cast_node);
  GE_ASSERT_NOTNULL(merged_cast_ptr);
  merged_cast_ptr->attr.sched = downstream_node->attr.sched;
  merged_cast_ptr->outputs[0].attr = downstream_node->inputs[input_idx].attr;
  merged_cast_ptr->outputs[0].attr.dtype = target_dtype;

  // Disconnect the edge from original cast to current node
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(orig_cast_out_anchor, in_anchor));
  // Connect the new cast to the original cast's input and the current node
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(orig_src_out_anchor, merged_cast_ptr->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(merged_cast_ptr->GetOutDataAnchor(0), in_anchor));

  return true;
}

Status DtypeConsistency::InsertCastNode(ge::AscGraph &graph, const ge::AscNodePtr &src_node,
                                        const ge::AscNodePtr &dst_node, size_t input_idx, ge::DataType target_dtype) {
  auto in_anchor = dst_node->GetInDataAnchor(static_cast<int32_t>(input_idx));
  GE_ASSERT_NOTNULL(in_anchor);
  auto src_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_out_anchor);

  // Create a new cast node
  std::string cast_name = std::string(src_node->GetName()) + "_cast_to_" + dst_node->GetName();
  Cast cast_node(cast_name.c_str());
  auto cast_node_ptr = graph.AddNode(cast_node);
  GE_ASSERT_NOTNULL(cast_node_ptr);
  cast_node_ptr->attr.sched = dst_node->attr.sched;
  cast_node_ptr->outputs[0].attr = dst_node->inputs[input_idx].attr;
  cast_node_ptr->outputs[0].attr.dtype = target_dtype;

  // Reconnect edges
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(src_out_anchor, in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(src_out_anchor, cast_node_ptr->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(cast_node_ptr->GetOutDataAnchor(0), in_anchor));

  return ge::SUCCESS;
}

Status DtypeConsistency::CancelRedundantCast(ge::AscGraph &graph) {
  // 1. Cast CSE: Merge multiple identical dtype Cast nodes from the same upstream into one
  GE_ASSERT_SUCCESS(DoCastCSE(graph), "Failed to do cast CSE");
  // 2. Remove redundant Cast(A->A)
  GE_ASSERT_SUCCESS(CancelIdentityCast(graph), "Failed to cancel identity cast");
  return ge::SUCCESS;
}

Status DtypeConsistency::DoCastCSE(ge::AscGraph &graph) {
  std::map<ge::OutDataAnchorPtr, std::map<ge::DataType, std::vector<ge::AscNodePtr>>> src_casts_map;
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!ge::ops::IsOps<Cast>(node)) {
      continue;
    }

    auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    auto src_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_out_anchor);

    auto out_dtype = node->outputs[0].attr.dtype;
    src_casts_map[src_out_anchor][out_dtype].push_back(node);
  }

  // Merge identical casts for each source and dtype
  for (auto &src_casts_iter : src_casts_map) {
    auto &src_out_anchor = src_casts_iter.first;
    auto &dtype_casts = src_casts_iter.second;
    for (auto &dtype_casts_iter : dtype_casts) {
      auto &cast_nodes = dtype_casts_iter.second;
      if (cast_nodes.size() <= 1UL) {
        continue;
      }

      // Keep the first Cast, remove the rest
      auto &keep_cast = cast_nodes[0];
      for (size_t i = 1UL; i < cast_nodes.size(); ++i) {
        auto &remove_cast = cast_nodes[i];
        GE_ASSERT_SUCCESS(RemoveDuplicateCast(keep_cast, remove_cast, src_out_anchor));
      }
    }
  }

  return ge::SUCCESS;
}

Status DtypeConsistency::CancelIdentityCast(ge::AscGraph &graph) {
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!ge::ops::IsOps<Cast>(node)) {
      continue;
    }

    auto in_dtype = node->inputs[0].attr.dtype;
    auto out_dtype = node->outputs[0].attr.dtype;
    if (in_dtype != out_dtype) {
      continue;
    }

    // Remove redundant Cast: src -> cast -> dst  =>  src -> dst
    auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    auto new_src = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(new_src);
    auto cast_out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(cast_out_anchor);
    GE_ASSERT_SUCCESS(PassUtils::RelinkAllOutNodeToSrc(cast_out_anchor, new_src));
    auto owner_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_SUCCESS(owner_graph->RemoveNode(node));
  }
  return ge::SUCCESS;
}
}  // namespace optimize
