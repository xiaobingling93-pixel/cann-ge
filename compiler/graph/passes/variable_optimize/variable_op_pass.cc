/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/variable_optimize/variable_op_pass.h"
#include <string>
#include <vector>

#include "formats/formats.h"
#include "formats/utils/formats_trans_utils.h"
#include "common/datatype_transfer/datatype_transfer.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
namespace {
const int32_t kTransOpOutIndex = 0;
const std::unordered_set<std::string> kDataUnChangedNodeType = {RESHAPE, REFORMAT, SQUEEZEV2, UNSQUEEZEV2};

std::string GetKey(Format format, DataType type, const std::vector<int64_t> &dims) {
  std::stringstream key;
  key << static_cast<int32_t>(format) << '-';
  key << static_cast<int32_t>(type) << '-';
  for (auto dim : dims) {
    key << dim << '-';
  }
  return key.str();
}

Status ByPassTransNode(NodePtr &trans_node, NodePtr &ref_node) {
  GE_CHECK_NOTNULL(trans_node);
  GE_CHECK_NOTNULL(ref_node);
  GELOGD("Begin to bypass trans node %s", trans_node->GetName().c_str());
  auto ret = GraphUtils::CopyInCtrlEdges(trans_node, ref_node);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Copy in control edge from node:%s(%s) to node:%s(%s) failed",
                      trans_node->GetName().c_str(), trans_node->GetType().c_str(),
                      ref_node->GetName().c_str(), ref_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Copy][InCtrlEdges] from node:%s(%s) to node:%s(%s) failed",
           trans_node->GetName().c_str(), trans_node->GetType().c_str(),
           ref_node->GetName().c_str(), ref_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  auto ref_in_anchor = ref_node->GetInDataAnchor(0);
  if (ref_in_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no input anchor, check invalid",
                       ref_node->GetName().c_str(), ref_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][InDataAnchor] failed, The variable ref node %s does not have an input anchor",
           ref_node->GetName().c_str());
    return INTERNAL_ERROR;
  }
  ref_in_anchor->UnlinkAll();
  auto trans_in_anchor = trans_node->GetInDataAnchor(0);
  if (trans_in_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no input anchor, check invalid",
                       trans_node->GetName().c_str(), trans_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][InDataAnchor] failed, Node:%s(%s) has no input anchor",
           trans_node->GetName().c_str(), trans_node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  auto prev_trans_node_out_anchor = trans_in_anchor->GetPeerOutAnchor();
  if (prev_trans_node_out_anchor == nullptr) {
    GELOGW(
        "The trans node %s does not have an input, so the ref node %s does"
        " not have any inputs after bypass",
        trans_node->GetName().c_str(), trans_node->GetName().c_str());
  } else {
    ret = GraphUtils::AddEdge(prev_trans_node_out_anchor, ref_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
                        prev_trans_node_out_anchor->GetOwnerNode()->GetName().c_str(),
                        prev_trans_node_out_anchor->GetOwnerNode()->GetType().c_str(),
                        prev_trans_node_out_anchor->GetIdx(),
                        ref_node->GetName().c_str(), ref_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:0) failed",
             prev_trans_node_out_anchor->GetOwnerNode()->GetName().c_str(),
             prev_trans_node_out_anchor->GetOwnerNode()->GetType().c_str(),
             prev_trans_node_out_anchor->GetIdx(), ref_node->GetName().c_str(), ref_node->GetType().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool IsTransSupport(const TransNodeInfo &trans_info) {
  if (trans_info.output.GetShape().IsUnknownShape()) {
    return false;
  }
  if (kDataUnChangedNodeType.count(trans_info.node_type) > 0U) {
    return true;
  } else if (trans_info.node_type == TRANSDATA || trans_info.node_type == TRANSPOSED) {
    const Format src_primary_format =
        static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(trans_info.input.GetFormat())));
    const Format dst_primary_format =
        static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(trans_info.output.GetFormat())));
    const Format src_sub_format =
        static_cast<Format>(GetSubFormat(static_cast<int32_t>(trans_info.input.GetFormat())));
    const Format dst_sub_format =
        static_cast<Format>(GetSubFormat(static_cast<int32_t>(trans_info.output.GetFormat())));
    const int64_t src_c0_format = GetC0Value(static_cast<int32_t>(trans_info.input.GetFormat()));
    const int64_t dts_c0_format = GetC0Value(static_cast<int32_t>(trans_info.output.GetFormat()));
    formats::TransArgs args{nullptr, trans_info.input.GetFormat(), trans_info.output.GetFormat(), src_primary_format,
                            dst_primary_format, src_sub_format, dst_sub_format, src_c0_format, dts_c0_format,
                            trans_info.input.GetShape().GetDims(), trans_info.output.GetShape().GetDims(),
                            trans_info.input.GetDataType()};
    return formats::IsTransFormatSupport(args);
  } else if (trans_info.node_type == CAST) {
    formats::CastArgs datatype_args{nullptr, static_cast<size_t>(trans_info.input.GetShape().GetShapeSize()),
                                    trans_info.input.GetDataType(), trans_info.output.GetDataType()};
    return formats::IsTransDataTypeSupport(datatype_args);
  } else {
    return false;
  }
}
}  // namespace

Status VariableOpPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to run variable op pass, null graph");
    return INTERNAL_ERROR;
  }
  GE_ASSERT_NOTNULL(VarManager::Instance(graph->GetSessionID()));
  // In the multi-batch training scenario, multiple branches use and update the same variable weight,
  // so it cannot fuse variables and conversion operators based on format.
  GE_CHECK_NOTNULL(VarManager::Instance(graph->GetSessionID()));
  bool no_need_fusion = (graph->GetParentGraph() != nullptr) ||
                        (VarManager::Instance(graph->GetSessionID())->HasSharedVarMemBetweenBatch());
  if (no_need_fusion) {
    return SUCCESS;
  }

  auto graph_id = graph->GetGraphID();
  GELOGD("Begin to run variable op pass on graph %s, session %lu, graph id %u", graph->GetName().c_str(),
         GetContext().SessionId(), graph_id);

  if (var_accelerate_ctrl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "The variable accelerate control is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to run var op pass, the variable accelerate control is null");
    return INTERNAL_ERROR;
  }

  GELOGD("Begin to generate ref map for variable and refs, graph name:%s.", graph->GetName().c_str());
  if (RenewVarDesc(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Renew][VarDesc] on graph:%s failed", graph->GetName().c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  if (GenerateVariableVariableRefMap(graph) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Generate][VariableMap] for graph:%s failed", graph->GetName().c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  GELOGD("Begin to fusion variables and trans nodes");
  for (auto &var_to_refs : var_and_var_ref_map_) {
    GE_CHECK_NOTNULL(var_accelerate_ctrl_);
    if (!var_accelerate_ctrl_->IsVarPermitToChangeFormats(var_to_refs.first->var_name)) {
      GELOGD("The var %s does not permit to change formats, skip it", var_to_refs.first->var_name.c_str());
      continue;
    }

    VarTransRoad fusion_road;
    auto ret = FusionIfNeed(var_to_refs.first, fusion_road);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Call][FusionIfNeed] for node:%s failed", var_to_refs.first->var_name.c_str());
      return ret;
    }

    if (fusion_road.empty()) {
      GELOGD("No need to fusion variable and trans op for var %s", var_to_refs.first->var_name.c_str());
      continue;
    }

    auto start_iter = fusion_road.begin();
    auto end_iter = fusion_road.rbegin();
    GELOGI(
        "Trans variable data for %s from format %s to %s, shape %s to %s "
        "data-type %s to %s, path len %zu success",
        var_to_refs.first->var_name.c_str(), TypeUtils::FormatToSerialString(start_iter->input.GetFormat()).c_str(),
        TypeUtils::FormatToSerialString(end_iter->output.GetFormat()).c_str(),
        formats::ShapeToString(start_iter->input.GetShape().GetDims()).c_str(),
        formats::ShapeToString(end_iter->output.GetShape().GetDims()).c_str(),
        TypeUtils::DataTypeToSerialString(start_iter->input.GetDataType()).c_str(),
        TypeUtils::DataTypeToSerialString(end_iter->output.GetDataType()).c_str(), fusion_road.size());

    ret = VarManager::Instance(graph->GetSessionID())->SetTransRoad(var_to_refs.first->var_name, fusion_road);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Set Trans road for node:%s(Variable) failed, session_id:%lu",
                        var_to_refs.first->var_name.c_str(), graph->GetSessionID());
      GELOGE(INTERNAL_ERROR, "[Set][TransRoad] for node:%s(Variable) failed, session_id:%lu",
             var_to_refs.first->var_name.c_str(), graph->GetSessionID());
      return INTERNAL_ERROR;
    }
    ret = VarManager::Instance(graph->GetSessionID())->SetChangedGraphId(var_to_refs.first->var_name, graph_id);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Update graph_id:%u for node:%s(Variable) failed, session_id:%lu",
                        graph_id, var_to_refs.first->var_name.c_str(), graph->GetSessionID());
      GELOGE(INTERNAL_ERROR, "[Update][GraphId] %u for node:%s(Variable) failed, session_id:%lu",
             graph_id, var_to_refs.first->var_name.c_str(), graph->GetSessionID());
      return INTERNAL_ERROR;
    }
    var_accelerate_ctrl_->SetStateChanged(var_to_refs.first->var_name);

    GELOGD("Begin to update format info for var %s.", var_to_refs.first->var_name.c_str());
    if (UpdateIOFormatInfo(end_iter->output, var_to_refs.first) != SUCCESS) {
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }

    for (const auto &node : var_to_refs.first->var_nodes) {
      // renew var desc if the trans_road is all reshape or reformat
      ret = RenewVarDesc(graph->GetSessionID(), node, fusion_road);
      if (ret != SUCCESS) {
        GELOGE(FAILED, "[Renew][VarDesc] for var[%s] failed!", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status VariableOpPass::DealFusion(const SameVarPtr &same_vars) {
  for (const auto &var_node : same_vars->var_nodes) {
    GE_CHECK_NOTNULL(var_node);
    GELOGD("Begin to fusion var %s with trans", var_node->GetName().c_str());
    auto graph = var_node->GetOwnerComputeGraph();
    for (auto &trans_node : var_node->GetOutDataNodes()) {
      GELOGD("Remove node %s type %s when fusion with variable %s", trans_node->GetName().c_str(),
             trans_node->GetType().c_str(), var_node->GetName().c_str());

      if (GraphUtils::IsolateNode(trans_node, {0}) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Isolate node:%s(%s) failed",
                          trans_node->GetName().c_str(), trans_node->GetType().c_str());
        GELOGE(GE_GRAPH_VARIABLE_OP_PASS_FAILED, "[Isolate][Node] %s(%s) failed",
               trans_node->GetName().c_str(), trans_node->GetType().c_str());
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }

      if (GraphUtils::RemoveNodeWithoutRelink(graph, trans_node) != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                          trans_node->GetName().c_str(), trans_node->GetType().c_str(), graph->GetName().c_str());
        GELOGE(GE_GRAPH_VARIABLE_OP_PASS_FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
               trans_node->GetName().c_str(), trans_node->GetType().c_str(), graph->GetName().c_str());
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }
    }
  }

  for (auto ref_node : GetRefVars(same_vars)) {
    GE_CHECK_NOTNULL(ref_node);
    for (auto &trans_node : ref_node->GetInDataNodes()) {
      GELOGD("Remove node %s type %s when fusion with variable %s", trans_node->GetName().c_str(),
             trans_node->GetType().c_str(), same_vars->var_name.c_str());
      auto graph = trans_node->GetOwnerComputeGraph();
      if (trans_node->GetOutDataNodes().size() > 1) {
        GELOGD(
            "The trans node %s type %s connecting with var-ref %s has more"
            " than one output data nodes, unlink the edge between them",
            trans_node->GetName().c_str(), trans_node->GetType().c_str(), ref_node->GetName().c_str());
        if (ByPassTransNode(trans_node, ref_node) != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[ByPass][TransNode] %s to ref %s failed", trans_node->GetName().c_str(),
                 ref_node->GetName().c_str());
          return INTERNAL_ERROR;
        }
      } else {
        GELOGD(
            "The trans node %s type %s connecting with var-ref %s has only"
            " one output data nodes, isolate and remove it.",
            trans_node->GetName().c_str(), trans_node->GetType().c_str(), ref_node->GetName().c_str());
        if (GraphUtils::IsolateNode(trans_node, {0}) != SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Isolate node:%s(%s) failed",
                            trans_node->GetName().c_str(), trans_node->GetType().c_str());
          GELOGE(GE_GRAPH_VARIABLE_OP_PASS_FAILED, "[Isolate][Node] %s(%s) failed",
                 trans_node->GetName().c_str(), trans_node->GetType().c_str());
          return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
        }
        if (GraphUtils::RemoveNodeWithoutRelink(graph, trans_node) != SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                            trans_node->GetName().c_str(), trans_node->GetType().c_str(), graph->GetName().c_str());
          GELOGE(GE_GRAPH_VARIABLE_OP_PASS_FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
                 trans_node->GetName().c_str(), trans_node->GetType().c_str(), graph->GetName().c_str());
          return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
        }
      }
    }
  }

  return SUCCESS;
}

Status VariableOpPass::CheckSameAndTransOp(const SameVarPtr &same_vars, bool &is_matched,
                                           VarTransRoad &fusion_road) const {
  std::set<std::string> data_type_and_formats;
  std::string trans_op_type;
  ge::NodePtr out_node;
  ge::GeTensorDesc output_desc;
  for (const auto &var_node : same_vars->var_nodes) {
    GE_CHECK_NOTNULL(var_node);
    for (const auto &out_node_and_anchor : var_node->GetOutDataNodesAndAnchors()) {
      auto in_anchor = out_node_and_anchor.second;
      GE_CHECK_NOTNULL(in_anchor);
      out_node = out_node_and_anchor.first;
      GE_CHECK_NOTNULL(out_node);
      auto trans_op_desc = out_node->GetOpDesc();
      GE_CHECK_NOTNULL(trans_op_desc);
      trans_op_type = trans_op_desc->GetType();

      GELOGD("current node type is %s.", trans_op_type.c_str());
      int32_t data_index = TransOpUtil::GetTransOpDataIndex(trans_op_type);
      if (data_index < 0) {
        GELOGD("Variables only can be fusion with trans_op, the next op is %s type %s", out_node->GetName().c_str(),
               out_node->GetType().c_str());
        return SUCCESS;
      }
      if (data_index != in_anchor->GetIdx()) {
        GELOGD(
            "Variables only can be fusion with trans nodes, the next node %s"
            " type %s index %d does not trans anything(correct index %d)",
            out_node->GetName().c_str(), out_node->GetType().c_str(), in_anchor->GetIdx(), data_index);
        return SUCCESS;
      }

      output_desc = trans_op_desc->GetOutputDesc(kTransOpOutIndex);

      auto trans_op_format = output_desc.GetFormat();
      auto trans_op_data_type = output_desc.GetDataType();
      auto shape = output_desc.GetShape().GetDims();
      auto datatype_and_format = GetKey(trans_op_format, trans_op_data_type, shape);
      data_type_and_formats.insert(datatype_and_format);
    }
  }

  if (data_type_and_formats.empty()) {
    return SUCCESS;
  }

  if (data_type_and_formats.size() > 1UL) {
    std::stringstream type_and_formats_stream;
    bool first_time = true;
    for (const auto &data_type_and_format : data_type_and_formats) {
      if (first_time) {
        first_time = false;
      } else {
        type_and_formats_stream << "|";
      }
      type_and_formats_stream << data_type_and_format;
    }

    GELOGW(
        "trans_op type size for var Node(%s) is over 1, Currently not"
        " supported, dataTypeAndFormats is %s.",
        same_vars->var_name.c_str(), type_and_formats_stream.str().c_str());
    return SUCCESS;
  }

  GE_ASSERT_NOTNULL(out_node);
  int32_t tran_in_index = TransOpUtil::GetTransOpDataIndex(out_node->GetType());
  auto out_op_desc = out_node->GetOpDesc();
  GE_CHECK_NOTNULL(out_op_desc);
  TransNodeInfo trans_node_info;
  trans_node_info.node_type = out_node->GetType();
  trans_node_info.input = out_op_desc->GetInputDesc(tran_in_index);
  trans_node_info.output = out_op_desc->GetOutputDesc(kTransOpOutIndex);

  if (!IsTransSupport(trans_node_info)) {
    GELOGD("The trans node %s does not support, skip the variable accelerating", trans_node_info.node_type.c_str());
    return SUCCESS;
  }

  is_matched = true;
  fusion_road.emplace_back(trans_node_info);

  return SUCCESS;
}

Status VariableOpPass::CheckVariableRefLegally(const SameVarPtr &same_vars, bool &is_var_ref_legally) {
  is_var_ref_legally = true;
  auto var_ref_nodes = GetRefVars(same_vars);
  GELOGD("var name %s, ref var count %zu.", same_vars->var_name.c_str(), var_ref_nodes.size());
  for (const auto &var_node : same_vars->var_nodes) {
    GE_CHECK_NOTNULL(var_node);
    for (const auto &var_ref_node : var_ref_nodes) {
      if (CheckVarAndVarRefAreAlike(var_node, var_ref_node, is_var_ref_legally) != SUCCESS) {
        GELOGE(FAILED, "[Call][CheckVarAndVarRefAreAlike] for node:%s failed", var_node->GetName().c_str());
        return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
      }

      GELOGD("var name %s, is_var_ref_legally is %d", same_vars->var_name.c_str(), is_var_ref_legally);

      if (!is_var_ref_legally) {
        return SUCCESS;
      }
    }
  }
  return SUCCESS;
}

Status VariableOpPass::UpdateVarAndRefOutputFormatInfo(const GeTensorDesc &final_output, const ge::NodePtr &node,
                                                       const SameVarPtr &same_vars) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node or its op_desc is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] node or its opdesc is nullptr");
    return FAILED;
  }
  const Format &format = final_output.GetFormat();
  const DataType &data_type = final_output.GetDataType();
  const GeShape &shape = final_output.GetShape();
  GELOGD("last ref is (%s, %s, %lu), var_ref_name is %s.", TypeUtils::DataTypeToSerialString(data_type).c_str(),
         TypeUtils::FormatToSerialString(format).c_str(), shape.GetDims().size(), node->GetName().c_str());

  auto node_desc = node->GetOpDesc()->GetOutputDesc(0);
  CopyVariableFormatDataTypeAndShape(final_output, node_desc);
  if (node->GetOpDesc()->UpdateOutputDesc(0, node_desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Update ouput:0 desc in op:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Update][OutputDesc] in op:%s(%s) failed, index:0",
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  GELOGD("node ref is (%s, %s, %lu), var_ref_name is %s.",
         TypeUtils::DataTypeToSerialString(node->GetOpDesc()->GetOutputDesc(0).GetDataType()).c_str(),
         TypeUtils::FormatToSerialString(node->GetOpDesc()->GetOutputDesc(0).GetFormat()).c_str(),
         node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims().size(), node->GetName().c_str());

  auto iterator = var_and_var_ref_map_.find(same_vars);
  if (iterator == var_and_var_ref_map_.end()) {
    auto graph = node->GetOwnerComputeGraph();
    if (GenerateVariableVariableRefMap(graph) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Generate][VariableMap] for graph:%s failed", graph->GetName().c_str());
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }
  }

  iterator = var_and_var_ref_map_.find(same_vars);
  if (iterator != var_and_var_ref_map_.end()) {
    for (const auto &var_ref_node : iterator->second) {
      const auto var_ref_node_description = var_ref_node->GetOpDesc();
      GE_CHECK_NOTNULL(var_ref_node_description);

      GELOGD("var_ref_node before is (%s, %s, %zu), var_ref_name is %s.",
             TypeUtils::DataTypeToSerialString(data_type).c_str(), TypeUtils::FormatToSerialString(format).c_str(),
             shape.GetDims().size(), var_ref_node->GetName().c_str());
      if (var_ref_node_description->UpdateOutputDesc(0U, node_desc) != GRAPH_SUCCESS) {
        GELOGW("UpdateOutputDesc fail.");
      }
      if (var_ref_node_description->UpdateInputDesc(0U, node_desc) != GRAPH_SUCCESS) {
        GELOGW("UpdateInputDesc fail.");
      }
      const auto &input_desc = var_ref_node_description->MutableInputDesc(0U);
      const auto &output_desc = var_ref_node_description->MutableOutputDesc(0U);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHECK_NOTNULL(output_desc);
      GELOGD("var_ref_node ref is (%s, %s, %zu), var_ref_name is %s.",
             TypeUtils::DataTypeToSerialString(input_desc->GetDataType()).c_str(),
             TypeUtils::FormatToSerialString(input_desc->GetFormat()).c_str(), output_desc->GetShape().GetDims().size(),
             var_ref_node->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status VariableOpPass::GenerateVariableVariableRefMap(const ComputeGraphPtr &compute_graph) {
  std::map<std::string, std::set<NodePtr>> names_to_var;
  std::map<std::string, std::set<NodePtr>> names_to_refs;
  GE_CHECK_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (!OpTypeUtils::IsVariableNode(node->GetType())) {
      continue;
    }
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string ref_var_name;
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_name)) {
      names_to_refs[ref_var_name].insert(node);
    } else {
      names_to_var[node->GetName()].insert(node);
    }
  }

  for (const auto &name_to_var : names_to_var) {
    SameVarPtr same_vars = MakeShared<SameVariable>();
    GE_CHECK_NOTNULL(same_vars);
    same_vars->var_name = name_to_var.first;
    same_vars->var_nodes = name_to_var.second;
    var_and_var_ref_map_[same_vars] = names_to_refs[name_to_var.first];
  }
  return SUCCESS;
}

Status VariableOpPass::CheckVarAndVarRefAreAlike(const NodePtr &var_node, const NodePtr &var_ref_node,
                                                 bool &is_var_and_variable_ref_are_alike) const {
  GE_CHECK_NOTNULL(var_node);
  GE_CHECK_NOTNULL(var_ref_node);
  GELOGD("var_node GetOutDataNodes. name is %s.", var_node->GetName().c_str());
  const auto &var_node_trans_nodes = var_node->GetOutDataNodes();
  GELOGD("var_node_trans_nodes size is %zu.", var_node_trans_nodes.size());
  GELOGD("var_ref_node GetOutDataNodes. name is %s.", var_ref_node->GetName().c_str());
  const auto &var_ref_node_trans_nodes = var_ref_node->GetInDataNodes();
  GELOGD("var_ref_node_trans_nodes size is %zu.", var_ref_node_trans_nodes.size());

  if (var_ref_node_trans_nodes.size() > 1) {
    REPORT_INNER_ERR_MSG("E19999", "In data node num:%zu of node:%s(%s) bigger than 1, check invalid",
                       var_ref_node_trans_nodes.size(),
                       var_ref_node->GetName().c_str(), var_ref_node->GetType().c_str());

    GELOGE(GE_GRAPH_VARIABLE_OP_PASS_FAILED, "[Check][Param] In data node num:%zu of node:%s(%s) bigger than 1.",
           var_ref_node_trans_nodes.size(), var_ref_node->GetName().c_str(), var_ref_node->GetType().c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  const auto &var_node_trans_node = var_node_trans_nodes.at(0);
  const auto &var_ref_node_trans_node = var_ref_node_trans_nodes.at(0);

  if (CheckTransNodeAreInverse(var_node_trans_node, var_ref_node_trans_node, is_var_and_variable_ref_are_alike) !=
      SUCCESS) {
    GELOGE(FAILED, "[Call][CheckTransNodeAreInverse] failed");
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }

  return SUCCESS;
}

Status VariableOpPass::CheckTransNodeAreInverse(const NodePtr &node_a, const NodePtr &node_b, bool &is_same) const {
  GELOGD("In CheckTransNodeAreInverse.");
  GE_CHECK_NOTNULL(node_a);
  GE_CHECK_NOTNULL(node_b);
  const auto &node_a_op_desc = node_a->GetOpDesc();
  const auto &node_b_op_desc = node_b->GetOpDesc();
  GE_CHECK_NOTNULL(node_a_op_desc);
  GE_CHECK_NOTNULL(node_b_op_desc);
  const auto &node_a_out_op_desc = node_a_op_desc->MutableOutputDesc(0);
  const auto &node_a_in_op_desc = node_a_op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(node_a_out_op_desc);
  GE_CHECK_NOTNULL(node_a_in_op_desc);

  const auto &node_b_out_op_desc = node_b_op_desc->MutableOutputDesc(0);
  const auto &node_b_in_op_desc = node_b_op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(node_b_out_op_desc);
  GE_CHECK_NOTNULL(node_b_in_op_desc);

  is_same = IsOpDescSame(node_a_out_op_desc, node_b_in_op_desc) && IsOpDescSame(node_b_out_op_desc, node_a_in_op_desc);

  return SUCCESS;
}

bool VariableOpPass::IsOpDescSame(const GeTensorDescPtr &op_desc_a, const GeTensorDescPtr &op_desc_b) const {
  const auto &format_a = op_desc_a->GetFormat();
  const auto &type_a = op_desc_a->GetDataType();
  const auto &shape_a = op_desc_a->GetShape();

  const auto &format_b = op_desc_b->GetFormat();
  const auto &type_b = op_desc_b->GetDataType();
  const auto &shape_b = op_desc_b->GetShape();

  const auto &dims_a = shape_a.GetDims();
  const auto &dims_b = shape_b.GetDims();
  GELOGD("(format, data type, shape) = (%s, %s, %zu) (%s, %s, %zu)", TypeUtils::FormatToSerialString(format_a).c_str(),
         TypeUtils::DataTypeToSerialString(type_a).c_str(), dims_a.size(),
         TypeUtils::FormatToSerialString(format_b).c_str(), TypeUtils::DataTypeToSerialString(type_b).c_str(),
         dims_b.size());
  return (format_a == format_b) && (type_a == type_b) && (dims_a == dims_b);
}

void VariableOpPass::CopyVariableFormatDataTypeAndShape(const GeTensorDesc &src_tensor_desc,
                                                        GeTensorDesc &dst_tensor_desc) const {
  dst_tensor_desc.SetOriginShape(src_tensor_desc.GetOriginShape());
  dst_tensor_desc.SetShape(src_tensor_desc.GetShape());
  dst_tensor_desc.SetFormat(src_tensor_desc.GetFormat());
  dst_tensor_desc.SetDataType(src_tensor_desc.GetDataType());
}

Status VariableOpPass::CheckIfCouldBeOptimized(const SameVarPtr &same_vars, bool &flag, VarTransRoad &fusion_road) {
  bool is_matched = false;
  auto ret = CheckSameAndTransOp(same_vars, is_matched, fusion_road);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckSameAndTransOp] failed, node:%s", same_vars->var_name.c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }
  if (!is_matched) {
    flag = false;
    return SUCCESS;
  }

  bool is_var_ref_legally = false;
  ret = CheckVariableRefLegally(same_vars, is_var_ref_legally);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckVariableRefLegally] failed, node:%s", same_vars->var_name.c_str());
    return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
  }
  GELOGD("is_var_ref_legally is %d.", is_var_ref_legally);
  if (!is_var_ref_legally) {
    GELOGI("variable ref connection are illegally");
    flag = false;
    fusion_road.clear();
    return SUCCESS;
  }

  flag = true;
  GELOGD("node %s, is_matched = %d is_var_ref_legally = %d, flag = %d", same_vars->var_name.c_str(), is_matched,
         is_var_ref_legally, flag);

  return SUCCESS;
}

Status VariableOpPass::FusionIfNeed(const SameVarPtr &same_vars, VarTransRoad &fusion_road) {
  bool can_fusion = false;
  bool first_time = true;
  while (true) {
    auto ret = CheckIfCouldBeOptimized(same_vars, can_fusion, fusion_road);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Call][CheckIfCouldBeOptimized] failed");
      return ret;
    }
    if (!can_fusion) {
      break;
    }
    // In original graph where the variable output format and the output trans node input format are not
    // continuous, you need to insert ReFormat on trans_road. And only need to make this judgment in the first loop,
    // the trans node will be deleted in each loop, and the output format of the variable will
    // not change, so there will be scenarios where variables and trans node formats are not continuous in next loop.
    if (first_time) {
      const NodePtr var_node = *(same_vars->var_nodes.cbegin());
      auto var_op_desc = var_node->GetOpDesc();
      GE_CHECK_NOTNULL(var_op_desc);
      ge::GeTensorDesc var_output_desc = var_op_desc->GetOutputDesc(0);
      if (var_output_desc.GetFormat() != (fusion_road.begin()->input).GetFormat()) {
        TransNodeInfo additional_trans_node_info;
        additional_trans_node_info.node_type = REFORMAT;
        additional_trans_node_info.input = var_output_desc;
        additional_trans_node_info.output = fusion_road.begin()->input;
        fusion_road.insert(fusion_road.cbegin(), additional_trans_node_info);
      }
      first_time = false;
    }
    ret = DealFusion(same_vars);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Call][DealFusion] failed");
      return ret;
    }
  }
  return SUCCESS;
}

Status VariableOpPass::UpdateIOFormatInfo(const GeTensorDesc &final_output, const SameVarPtr &same_vars) {
  for (const auto &need_set_node : same_vars->var_nodes) {
    auto ret = UpdateVarAndRefOutputFormatInfo(final_output, need_set_node, same_vars);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Call][UpdateVarAndRefOutputFormatInfo] failed");
      return GE_GRAPH_VARIABLE_OP_PASS_FAILED;
    }
  }
  return SUCCESS;
}

Status VariableOpPass::RenewVarDesc(const ge::ComputeGraphPtr &graph) const {
  GE_CHECK_NOTNULL(graph);
  GE_ASSERT_NOTNULL(ge::VarManager::Instance(graph->GetSessionID()));
  // renew var manager desc
  auto graph_id = graph->GetGraphID();
  Status ret = SUCCESS;
  for (auto &node : graph->GetAllNodes()) {
    if (OpTypeUtils::IsVariableNode(node->GetType())) {
      auto var_manager = ge::VarManager::Instance(graph->GetSessionID());
      GE_CHECK_NOTNULL(var_manager);
      if (!var_manager->IsVarExist(node->GetName())) {
        GELOGD("var manager does not exist var node[%s]", node->GetName().c_str());
        continue;
      }
      GELOGD("var manager exist var node[%s], graph name[%s]", node->GetName().c_str(), graph->GetName().c_str());
      GE_CHECK_NOTNULL(node->GetOpDesc());
      ret = var_manager->RenewCurVarDesc(node->GetName(), node->GetOpDesc());
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Renew descriptor for node:%s(%s) failed, session_id:%lu",
                          node->GetName().c_str(), node->GetType().c_str(), graph->GetSessionID());
        GELOGE(FAILED, "[Renew][Descriptor] for node:%s(%s) failed, session_id:%lu",
               node->GetName().c_str(), node->GetType().c_str(), graph->GetSessionID());
        return FAILED;
      }

      ret = var_manager->RecordStagedVarDesc(graph_id,
                                                                                 node->GetName(),
                                                                                 node->GetOpDesc()->GetOutputDesc(0U));
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Record staged descriptor for node:%s(%s) failed, session_id:%lu",
                          node->GetName().c_str(), node->GetType().c_str(), graph->GetSessionID());
        GELOGE(FAILED, "[Record][Descriptor] for node:%s(%s) failed, session_id:%lu",
               node->GetName().c_str(), node->GetType().c_str(), graph->GetSessionID());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status VariableOpPass::RenewVarDesc(uint64_t session_id, const NodePtr &node, const VarTransRoad &fusion_road) const {
  // renew var desc if the trans_road is all reshape or reformat
  for (const auto &road : fusion_road) {
    if (kDataUnChangedNodeType.count(road.node_type) == 0) {
      return SUCCESS;
    }
  }
  GE_ASSERT_NOTNULL(ge::VarManager::Instance(session_id));
  if (!ge::VarManager::Instance(session_id)->IsVarExist(node->GetName())) {
    GELOGD("var manager does not exist var node[%s]", node->GetName().c_str());
    return SUCCESS;
  }
  GELOGD("var manager exist var node[%s]", node->GetName().c_str());
  GE_CHECK_NOTNULL(node->GetOpDesc());
  Status ret = ge::VarManager::Instance(session_id)->RenewCurVarDesc(node->GetName(), node->GetOpDesc());
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Renew descriptor for node:%s(%s) failed, session_id:%lu",
                      node->GetName().c_str(), node->GetType().c_str(), session_id);
    GELOGE(FAILED, "[Renew][Descriptor] for node:%s(%s) failed, session_id:%lu",
           node->GetName().c_str(), node->GetType().c_str(), session_id);
    return FAILED;
  }

  return SUCCESS;
}

std::vector<NodePtr> VariableOpPass::GetRefVars(const SameVarPtr &same_vars) {
  std::vector<NodePtr> nodes;
  auto iter = var_and_var_ref_map_.find(same_vars);
  if (iter != var_and_var_ref_map_.end()) {
    for (const auto &node : iter->second) {
      if ((node->GetOpDesc() != nullptr) && AttrUtils::HasAttr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME)) {
        nodes.emplace_back(node);
      }
    }
  }
  return nodes;
}

REG_PASS_OPTION("VariableOpPass").LEVELS(OoLevel::kO3);
}  // namespace ge
