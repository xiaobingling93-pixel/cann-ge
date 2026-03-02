/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/infershape_pass.h"
#include <unordered_set>

#include "common/checker.h"
#include "base/err_msg.h"
#include "common/util/mem_utils.h"
#include "analyzer/analyzer.h"
#include "framework/common/util.h"
#include "graph/shape_refiner.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils_ex.h"

#include "graph/operator_factory.h"
#include "graph_metadef/graph/ge_context.h"

namespace ge {
namespace {
constexpr int32_t kSwitchExitAnchorIndex = 0;
constexpr int32_t kSwitchPredAnchorIndex = 1;
void SerialSingleShapeRange(const std::vector<std::pair<int64_t, int64_t>> &shape_range, std::stringstream &desc_str) {
  if (!shape_range.empty()) {
    desc_str << "{" << shape_range[0].first << "," << shape_range[0].second << "}";
    for (size_t i = 1U; i < shape_range.size(); ++i) {
      desc_str << ",{";
      desc_str << std::to_string(shape_range[i].first) << "," << std::to_string(shape_range[i].second);
      desc_str << "}";
    }
  }
}
void SerialShapeRange(const GeTensorDescPtr &desc, std::stringstream &desc_str) {
  desc_str << "[";
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)desc->GetShapeRange(shape_range);
  SerialSingleShapeRange(shape_range, desc_str);
  desc_str << "],origin_shape_range:[";
  shape_range.clear();

  (void)desc->GetOriginShapeRange(shape_range);
  SerialSingleShapeRange(shape_range, desc_str);
  desc_str << "]";
}
void UpdateShapeAndDType(const GeTensorDescPtr &src, const GeTensorDescPtr &dst) {
  dst->SetOriginShape(src->GetOriginShape());
  dst->SetShape(src->GetShape());
  dst->SetDataType(src->GetDataType());
  dst->SetOriginDataType(src->GetOriginDataType());
  std::vector<std::pair<int64_t, int64_t>> src_shape_range;
  src->GetShapeRange(src_shape_range);
  dst->SetShapeRange(src_shape_range);
  dst->SetOriginShapeRange(src_shape_range);
  ge::TensorUtils::SetRealDimCnt(*dst, static_cast<uint32_t>(src->GetShape().GetDims().size()));
}
std::string GetResourceOpName(const std::string &graph_name, const std::string &op_name) {
  return (graph_name + op_name).c_str();
}
}  // namespace

std::string InferShapePass::SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const {
  std::stringstream ss;
  ss << "shape:[" << tensor_desc->MutableShape().ToString() << "],";
  ss << "format:" << TypeUtils::FormatToSerialString(tensor_desc->GetFormat()) << ",";
  ss << "dtype:" << TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()) << ",";
  ss << "origin_shape:[" << tensor_desc->GetOriginShape().ToString() << "],";
  ss << "origin_format:" << TypeUtils::FormatToSerialString(tensor_desc->GetOriginFormat()) << ",";
  ss << "origin_dtype:" << TypeUtils::DataTypeToSerialString(tensor_desc->GetOriginDataType()) << ",";
  std::stringstream range_str;
  SerialShapeRange(tensor_desc, range_str);
  ss << "shape_range:" << range_str.str();
  return ss.str();
}
Status InferShapePass::SuspendV1LoopExitNodes(const NodePtr &node) {
  if (node->GetType() != SWITCH) {
    return SUCCESS;
  }
  auto pred_node = NodeUtils::GetInDataNodeByIndex(*node, kSwitchPredAnchorIndex);
  GE_CHECK_NOTNULL(pred_node);
  if (pred_node->GetType() != LOOPCOND) {
    return SUCCESS;
  }

  for (const auto &anchor_2_node : NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, kSwitchExitAnchorIndex)) {
    GELOGI("Found v1 loop when infershape, suspend Exit node %s, type %s.", anchor_2_node.second->GetName().c_str(),
           anchor_2_node.second->GetType().c_str());
    auto &suspend_nodes = graphs_2_suspend_nodes_[GetCurrentGraphName()];
    if (suspend_nodes.nodes_set.insert(anchor_2_node.second).second) {
      suspend_nodes.nodes.push(anchor_2_node.second);
      AddNodeSuspend(anchor_2_node.second);
    }
  }
  return SUCCESS;
}

Status InferShapePass::Infer(NodePtr &node) {
  auto ret = InferShapeAndType(node);
  if (ret != GRAPH_SUCCESS && ret != GRAPH_NODE_NEED_REPASS) {
    auto graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(graph);
    auto root_graph = ge::GraphUtils::FindRootGraph(graph);
    GE_CHECK_NOTNULL(root_graph);
    analyzer::DataInfo analyze_info{root_graph->GetSessionID(), root_graph->GetGraphID(),
                                    analyzer::INFER_SHAPE, node, "InferShapeFailed!"};
    (void)Analyzer::GetInstance()->DoAnalyze(analyze_info);
    (void)Analyzer::GetInstance()->SaveAnalyzerDataToFile(root_graph->GetSessionID(),
                                                          root_graph->GetGraphID());
    REPORT_INNER_ERR_MSG("EZ9999", "Call InferShapeAndType for node:%s(%s) failed", node->GetName().c_str(),
                      node->GetType().c_str());
    GELOGE(GE_GRAPH_INFERSHAPE_FAILED, "[Call][InferShapeAndType] for node:%s(%s) failed", node->GetName().c_str(),
           node->GetType().c_str());
    GE_DUMP(root_graph, "InferShapeBlackBox");
    return GE_GRAPH_INFERSHAPE_FAILED;
  }
  return ret;
}

graphStatus InferShapePass::InferShapeAndType(NodePtr &node) {
  GE_ASSERT_SUCCESS(SuspendV1LoopExitNodes(node), "[Call][SuspendV1LoopExitNodes] failed.");
  if (NodeUtilsEx::Verify(node) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Verifying %s failed.", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Call][Verify] Verifying %s failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  Operator op = OpDescUtils::CreateOperatorFromNode(node);

  bool is_unknown_graph = node->GetOwnerComputeGraph()->GetGraphUnknownFlag();
  if (!is_unknown_graph) {
    InferenceContextPtr inference_context;
    if (ShapeRefiner::CreateInferenceContext(node, resource_context_mgr_, inference_context) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "CreateInferenceContext of %s failed.", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Create][Context] CreateInferenceContext of %s failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    GE_CHECK_NOTNULL(inference_context);
    std::vector<AscendString> marks;
    inference_context->GetMarks(marks);
    GELOGD("Create context for node:%s, marks %zu.", node->GetName().c_str(), marks.size());
    op.SetInferenceContext(inference_context);
  }

  graphStatus status = CallInferShapeFunc(node, op);
  if ((status != GRAPH_NODE_NEED_REPASS) && (status != GRAPH_PARAM_INVALID) && (status != GRAPH_SUCCESS)) {
    // node like netoutput return param_invalid, but valid ?
    return GE_GRAPH_INFERSHAPE_FAILED;
  }
  GE_ASSERT_SUCCESS(UpdateNetOutputIODesc(node), "Update IO desc of %s failed.", node->GetName().c_str());
  UpdateCurNodeOutputDesc(node);
  if (!is_unknown_graph) {
    auto ctx_after_infer = op.GetInferenceContext();
    if (ctx_after_infer != nullptr) {
      std::vector<AscendString> marks;
      ctx_after_infer->GetMarks(marks);

      GELOGD("[%s] after infershape. mark:%zu.", node->GetName().c_str(), marks.size());
      if (!ctx_after_infer->GetOutputHandleShapesAndTypes().empty() || !marks.empty()) {
        GELOGD("[%s] set inference context after. mark:%zu", node->GetName().c_str(),
               marks.size());
        ShapeRefiner::PushToContextMap(node, ctx_after_infer);
      }
      if (resource_op_access_ctrl_ != nullptr && resource_context_mgr_ != nullptr) {
        // if resource_shapes changed, need add nodes which relied on this resource to repass
        GE_ASSERT_SUCCESS(RepassReliedNodeIfResourceChanged(ctx_after_infer, node));
        // if node relied on some resource ,register to mgr
        (void) RegisterNodesReliedOnResource(ctx_after_infer, node);
      }
    }
  }

  return (status == GRAPH_NODE_NEED_REPASS) ? GRAPH_NODE_NEED_REPASS : GRAPH_SUCCESS;
}

Status InferShapePass::UpdateNetOutputIODesc(const NodePtr &node) const {
  if (node->GetType() != NETOUTPUT) {
    return SUCCESS;
  }
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto inputs_desc = op_desc->GetAllInputsDesc();
  if (inputs_desc.size() != op_desc->GetOutputsSize()) {
    return SUCCESS;
  }
  for (size_t i = 0U; i < inputs_desc.size(); ++i) {
    GE_ASSERT_SUCCESS(op_desc->UpdateOutputDesc(i, inputs_desc.at(i)), "Node: %s update output desc %zu failed.",
                      node->GetName().c_str(), i);
  }
  return SUCCESS;
}

void InferShapePass::UpdateCurNodeOutputDesc(const NodePtr &node) const {
  auto op_desc = node->GetOpDesc();
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    GE_IF_BOOL_EXEC(output_tensor == nullptr, continue);
    GE_IF_BOOL_EXEC(output_tensor->MutableShape().GetDims().empty(),
                    output_tensor->SetOriginShape(output_tensor->GetShape()));

    ge::TensorUtils::SetRealDimCnt(*output_tensor, static_cast<uint32_t>(output_tensor->GetOriginShape().GetDims()
      .size()));
    output_tensor->SetOriginDataType(output_tensor->GetDataType());
    // set output origin shape range
    std::vector<std::pair<int64_t, int64_t>> range;
    (void)output_tensor->GetShapeRange(range);
    output_tensor->SetOriginShapeRange(range);
    GELOGD("Node %s, output %d, origin shape is %ld, origin format is %s, origin data type is %s",
           node->GetName().c_str(), out_anchor->GetIdx(), output_tensor->GetOriginShape().GetShapeSize(),
           TypeUtils::FormatToSerialString(output_tensor->GetOriginFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor->GetOriginDataType()).c_str());
  }
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto input_tensor = op_desc->MutableInputDesc(in_anchor->GetIdx());
    GE_IF_BOOL_EXEC(input_tensor == nullptr, continue);

    // set input origin shape range
    std::vector<std::pair<int64_t, int64_t>> range;
    (void)input_tensor->GetShapeRange(range);
    input_tensor->SetOriginShapeRange(range);
  }
}

bool InferShapePass::SameTensorDesc(const GeTensorDescPtr &src, const GeTensorDescPtr &dst) const {
  // check shape range
  std::vector<std::pair<int64_t, int64_t>> src_shape_range;
  std::vector<std::pair<int64_t, int64_t>> dst_shape_range;
  src->GetShapeRange(src_shape_range);
  dst->GetShapeRange(dst_shape_range);
  if (src_shape_range.size() != dst_shape_range.size()) {
    GELOGI("Src shape range size is %zu, dst shape range size is %zu, not same.", src_shape_range.size(),
           dst_shape_range.size());
    return false;
  }
  for (size_t i = 0; i < src_shape_range.size(); ++i) {
    if (src_shape_range[i].first != dst_shape_range[i].first ||
        src_shape_range[i].second != dst_shape_range[i].second) {
      GELOGI("Current dim %zu. Src shape range is [%lu-%lu], dst shape range is [%lu-%lu], not same.",
             i, src_shape_range[i].first, src_shape_range[i].second, dst_shape_range[i].first, dst_shape_range[i].second);
      return false;
    }
  }

  // check shape
  auto src_shape = src->GetShape();
  auto dst_shape = dst->GetShape();
  if (src_shape.GetDims() != dst_shape.GetDims() || src->GetOriginShape().GetDims() != dst->GetOriginShape().GetDims() ||
      src->GetDataType() != dst->GetDataType() || src->GetOriginDataType() != dst->GetOriginDataType()) {
    GELOGD(
        "Src shape is %s, origin_shape is %s, data_type is %s, origin data_type is %s; "
        "Dst shape is %s, origin_shape is %s, data_type is %s, original data_type is %s, not same.",
        src_shape.ToString().c_str(), src->GetOriginShape().ToString().c_str(),
        TypeUtils::DataTypeToSerialString(src->GetDataType()).c_str(),
        TypeUtils::DataTypeToSerialString(src->GetOriginDataType()).c_str(), dst_shape.ToString().c_str(),
        dst->GetOriginShape().ToString().c_str(), TypeUtils::DataTypeToSerialString(dst->GetDataType()).c_str(),
        TypeUtils::DataTypeToSerialString(dst->GetOriginDataType()).c_str());
    return false;
  }
  return true;
}

graphStatus InferShapePass::UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) {
  changed = false;
  if (SameTensorDesc(src, dst)) {
    GELOGD("Peer dst tensor_desc is same as src tensor_desc. No need update.");
    return SUCCESS;
  }

  changed = true;
  UpdateShapeAndDType(src, dst);
  GELOGD(
      "UpdatePeerInputDesc from src Node: shape: [%s], datatype: %s, original datatype is %s."
      "To dst Node: shape: [%s], datatype: %s, original datatype is %s.",
      src->GetShape().ToString().c_str(), TypeUtils::DataTypeToSerialString(src->GetDataType()).c_str(),
      TypeUtils::DataTypeToSerialString(src->GetOriginDataType()).c_str(), dst->GetShape().ToString().c_str(),
      TypeUtils::DataTypeToSerialString(dst->GetDataType()).c_str(),
      TypeUtils::DataTypeToSerialString(dst->GetOriginDataType()).c_str());
  return SUCCESS;
}

graphStatus InferShapePass::CallInferShapeFunc(NodePtr &node, Operator &op) const {
  auto op_desc = node->GetOpDesc();
  bool out_shape_locked = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_OUT_SHAPE_LOCKED, out_shape_locked);
  GELOGD("Get shape locked flag:%u from node:%s.", static_cast<uint32_t>(out_shape_locked), node->GetName().c_str());
  if (out_shape_locked) {
    GELOGD("skip infer output shape process result of getting attribute out_shape_locked is true.");
    return GRAPH_SUCCESS;
  }

  return OpDescUtilsEx::CallInferFunc(op_desc, op);
}

graphStatus InferShapePass::UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src,
                                                      const GeTensorDescPtr &dst) {
  GELOGD("Enter update parent node shape for class branch op process");
  // check sub_graph shape.If not same ,do unknown shape process
  const auto &ref_out_tensor = src.at(0);
  GeTensorDescPtr tmp_dst = MakeShared<GeTensorDesc>();
  GE_ASSERT_NOTNULL(tmp_dst);
  UpdateShapeAndDType(ref_out_tensor, tmp_dst);
  GeShape &ref_out_tensor_shape = tmp_dst->MutableShape();

  for (auto &tensor : src) {
    if (ref_out_tensor->GetDataType() != tensor->GetDataType()) {
      REPORT_INNER_ERR_MSG("E19999", "Does not support diff dtype among all ref output, shape:%s",
                         ref_out_tensor_shape.ToString().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node does not support diff dtype output");
      return GRAPH_FAILED;
    }
    auto shape = tensor->MutableShape();
    if (shape.GetDims().size() != ref_out_tensor_shape.GetDims().size()) {
      GELOGD("Shape from subgraph size: %lu, ref_out_tensor_shape size: %lu", shape.GetShapeSize(),
             ref_out_tensor_shape.GetShapeSize());
      ref_out_tensor_shape = GeShape(UNKNOWN_RANK);
      (void)tmp_dst->SetShapeRange({});
      break;
    }
    for (size_t j = 0; j < ref_out_tensor_shape.GetDims().size(); j++) {
      if (ref_out_tensor_shape.GetDim(j) == shape.GetDim(j)) {
        continue;
      }
      GELOGD("j: %zu ,shape from subgraph size: %lu, ref_out_tensor_shape size: %lu", j, shape.GetShapeSize(),
             ref_out_tensor_shape.GetShapeSize());
      (void)ref_out_tensor_shape.SetDim(j, UNKNOWN_DIM);
    }
  }
  UpdateShapeAndDType(tmp_dst, dst);
  return GRAPH_SUCCESS;
}
graphStatus InferShapePass::UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                                  const GeTensorDescPtr &dst) {
  // check sub_graph shape. Get max for update.
  if (src.empty()) {
    GELOGI("Src subgraph shape is empty.");
    return SUCCESS;
  }

  int64_t max_size = 0;
  size_t max_shape_index = 0;
  auto &ref_out_tensor = src.at(0);
  for (size_t j = 0; j < src.size(); ++j) {
    auto &tensor = src.at(j);
    if (ref_out_tensor->GetDataType() != tensor->GetDataType()) {
      REPORT_INNER_ERR_MSG("E19999", "node does not support diff dtype among all ref output");
      GELOGE(GRAPH_FAILED, "[Check][Param] node does not support diff dtype among all ref output");
      return GRAPH_FAILED;
    }

    auto shape = tensor->MutableShape();
    int64_t size = 1;
    for (auto dim : shape.GetDims()) {
      if (dim < 0) {
        REPORT_INNER_ERR_MSG("E19999",
                           "Multi-batch not support middle dynamic shape. CurrentShape: [%s]. Please "
                           "check nodes in graph which cause dynamic shape.",
                           shape.ToString().c_str());
        GELOGE(PARAM_INVALID,
               "[Check][NotSupport] DynamicDims with multi-batch not support middle dynamic shape. CurrentShape: [%s]. "
               "Please check nodes in graph which cause dynamic shape.",
               shape.ToString().c_str());
        return PARAM_INVALID;
      }
      if (dim != 0 && INT64_MAX / dim < size) {
        REPORT_INNER_ERR_MSG("E19999", "The shape:%s size overflow", shape.ToString().c_str());
        GELOGE(PARAM_INVALID, "[Check][Overflow] The shape size overflow");
        return PARAM_INVALID;
      }
      size *= dim;
    }

    if (size > max_size) {
      max_size = size;
      max_shape_index = j;
    }
  }
  UpdateShapeAndDType(src.at(max_shape_index), dst);
  return GRAPH_SUCCESS;
}

graphStatus InferShapePass::UpdateOutputFromSubgraphsForSubgraphMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                                          const GeTensorDescPtr &dst) {
  // check sub_graph shape. Get max for update.
  if (src.empty()) {
    GELOGI("Src subgraph shape is empty.");
    return SUCCESS;
  }

  const auto &ref_out_tensor = src.at(0U);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  const std::vector<int64_t> &first_dims = ref_out_tensor->GetShape().GetDims();
  std::vector<int64_t> final_dims;
  for (size_t j = 0U; j < src.size(); ++j) {
    const auto &tensor = src.at(j);
    if (ref_out_tensor->GetDataType() != tensor->GetDataType() ||
        ref_out_tensor->GetShape().GetDimNum() != tensor->GetShape().GetDimNum()) {
      REPORT_INNER_ERR_MSG("E19999", "node does not support diff dtype/dim num among all ref output");
      GELOGE(GRAPH_FAILED, "[Check][Param] node does not support diff dtype/dim num among all ref output");
      return GRAPH_FAILED;
    }

    for (size_t i = 0U; i < first_dims.size(); i++) {
      // collect shape range
      if (shape_range.size() <= i) {
        int64_t dim = tensor->GetShape().GetDim(i);
        shape_range.emplace_back(dim, dim);
      } else {
        shape_range[i].first = (shape_range[i].first < tensor->GetShape().GetDim(i)) ?
                               shape_range[i].first : tensor->GetShape().GetDim(i);
        shape_range[i].second = (shape_range[i].first > tensor->GetShape().GetDim(i)) ?
                                shape_range[i].first : tensor->GetShape().GetDim(i);
      }

      // collect shape
      const int64_t dim = (tensor->GetShape().GetDim(i) != first_dims[i]) ? ge::UNKNOWN_DIM : first_dims[i];
      if (final_dims.size() <= i) {
        final_dims.push_back(dim);
        continue;
      }
      final_dims[i] = dim;
    }
  }

  dst->SetOriginShape(GeShape(final_dims));
  dst->SetShape(GeShape(final_dims));
  dst->SetDataType(ref_out_tensor->GetDataType());
  dst->SetOriginDataType(ref_out_tensor->GetOriginDataType());
  dst->SetShapeRange(shape_range);
  dst->SetOriginShapeRange(shape_range);
  ge::TensorUtils::SetRealDimCnt(*dst, static_cast<uint32_t>(final_dims.size()));
  GELOGD("Update shape[%s] and shape_range by sungraphs for case node in multi dims scene.",
         GeShape(final_dims).ToString().c_str());

  return GRAPH_SUCCESS;
}

Status InferShapePass::OnSuspendNodesLeaked() {
  const auto iter = graphs_2_suspend_nodes_.find(GetCurrentGraphName());
  if (iter == graphs_2_suspend_nodes_.end()) {
    GELOGI("Current graph %s no suspend node.", GetCurrentGraphName().c_str());
    return SUCCESS;
  }
  if (!iter->second.nodes.empty()) {
    AddNodeResume(iter->second.PopSuspendedNode());
  }
  return SUCCESS;
}
Status InferShapePass::RepassReliedNodeIfResourceChanged(const InferenceContextPtr &inference_context,
                                                         const NodePtr &cur_node) {
  auto changed_resource_keys = inference_context->GetChangedResourceKeys();
  for (const auto &key : changed_resource_keys) {
    auto changed_nodes = resource_context_mgr_->MutableNodesReliedOnResource(key.GetString());
    if (changed_nodes.empty()) {
      continue;
    }
    auto cur_graph = cur_node->GetOwnerComputeGraph();
    for (const auto &node : changed_nodes) {
      auto owner_graph = node->GetOwnerComputeGraph();
      if (owner_graph == nullptr) {
        continue;
      }
      // 1. node in current graph, just repass read_node
      if (owner_graph->GetName() == cur_graph->GetName()) {
        GELOGI("Node %s relied on resource %s, it will repass immediately.", node->GetName().c_str(), key.GetString());
        AddImmediateRePassNode(node);
        continue;
      } else {
        auto cur_root_graph = GraphUtils::FindRootGraph(cur_graph);
        auto owner_root_graph = GraphUtils::FindRootGraph(owner_graph);
        // 2. node in current geop, need repass top parent_node of this read_node
        if (cur_root_graph->GetName() == owner_root_graph->GetName()) {
          auto parent_node = owner_graph->GetParentNode();
          while (parent_node != nullptr) {
            parent_node = parent_node->GetOwnerComputeGraph()->GetParentNode();
          }
          (parent_node == nullptr) ? AddGlobalImmediateRePassNode(node) : AddGlobalImmediateRePassNode(parent_node);
        } else {
          // 3. node in other geop, need mark geop need rebuild
          auto graph_name = owner_root_graph->GetName();
          std::string parallel_option;
          (void)GetContext().GetOption(OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE, parallel_option);
          if (parallel_option == "1") {
            REPORT_INNER_ERR_MSG("E19999", "Graph %s has node %s relied on resource %s. Resource changes trigger graph"
                                 " rebuild, which is disabled when ge.AllowMultiGraphParallelCompile is set to \"1\".",
                                 graph_name.c_str(), node->GetName().c_str(), key.GetString());
            GELOGE(UNSUPPORTED, "Graph %s has node %s relied on resource %s. Resource changes trigger graph rebuild,"
                   " which is disabled when ge.AllowMultiGraphParallelCompile is set to \"1\".",
                   graph_name.c_str(), node->GetName().c_str(), key.GetString());
            return UNSUPPORTED;
          }
          GELOGI("Graph %s has node %s relied on resource %s, it will rebuild before next run", graph_name.c_str(),
                 node->GetName().c_str(), key.GetString());
          resource_op_access_ctrl_->SetStateChanged(GetResourceOpName(graph_name, node->GetName()));
        }
      }
    }
  }
  return SUCCESS;
}
Status InferShapePass::RegisterNodesReliedOnResource(const InferenceContextPtr &inference_context,
                                                     NodePtr &node) const {
  auto relied_on_resource = inference_context->GetReliedOnResourceKeys();
  if (!relied_on_resource.empty()) {
    auto owner_root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
    auto graph_name = owner_root_graph->GetName();
    auto graph_id = owner_root_graph->GetGraphID();
    resource_op_access_ctrl_->AddResourceName(graph_id, GetResourceOpName(graph_name, node->GetName()));
  }
  for (const auto &resource_key : relied_on_resource) {
    auto ret = resource_context_mgr_->RegisterNodeReliedOnResource(resource_key.GetString(), node);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("Failed to register node %s relied on resource %s.", node->GetName().c_str(), resource_key.GetString());
      return ret;
    }
    GELOGI("Node %s is relied on resource %s, if resource changed, node will repass.", node->GetName().c_str(),
           resource_key.GetString());
  }
  return SUCCESS;
}

REG_PASS_OPTION("InferShapePass").LEVELS(OoLevel::kO1);
}  // namespace ge
