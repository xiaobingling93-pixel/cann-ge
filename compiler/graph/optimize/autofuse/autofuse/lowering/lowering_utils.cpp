/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering_utils.h"

#include <map>
#include "common/ge_common/ge_types.h"
#include "framework/common/debug/ge_log.h"
#include "utils/node_utils.h"
#include "utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "asc_lowerer/loop_common.h"
#include "utils/autofuse_attrs.h"


namespace ge{
const std::string aiv_cnt_key = "_op_vectorcore_num";

std::string GetAscTensorDescStr(const OutDataAnchorPtr &anchor) {
  GE_WARN_ASSERT(anchor != nullptr);
  GE_WARN_ASSERT(anchor->GetOwnerNode() != nullptr);
  GE_WARN_ASSERT(anchor->GetOwnerNode()->GetOpDesc() != nullptr);
  const auto desc = anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(anchor->GetIdx());
  GE_WARN_ASSERT(desc != nullptr);
  const auto attr = desc->GetAttrsGroup<AscTensorAttr>();
  if (attr == nullptr || (attr->axis.empty() && attr->repeats.empty() && attr->strides.empty())) {
    return "";
  }
  std::stringstream ss;
  const static auto kExpressionStr = [](const Expression &e) { return std::string(e.Str().get()); };
  ge::DataType dtype = attr->dtype;
  ss << "dtype = " << ge::TypeUtils::DataTypeToSerialString(dtype);
  ss << ", axis = " << loop::StrJoin(attr->axis, [](const int64_t &e) { return std::to_string(e); });
  ss << ", repeats = " << loop::StrJoin(attr->repeats, kExpressionStr);
  ss << ", strides = " << loop::StrJoin(attr->strides, kExpressionStr);
  return ss.str();
}

graphStatus TransCoreNumToInt(std::string core_num_ori, int &core_num) {
  try {
    core_num = stoi(core_num_ori);
  } catch (...) {
    GELOGW("Attr core num Value %s is not integer.", core_num_ori.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus AddDataNodeForConstructGraph(const Node *const &node, const ComputeGraphPtr &graph,
                                         std::map<const OutDataAnchor *, OutDataAnchorPtr> &origin_to_replaced) {
  for (uint32_t i = 0U; i < node->GetAllInDataAnchorsSize(); i++) {
    auto origin_input = node->GetInDataAnchor(static_cast<int32_t>(i));
    GE_ASSERT_NOTNULL(origin_input);
    auto peer_out = origin_input->GetPeerOutAnchor().get();
    if (peer_out == nullptr) {
      continue;
    }
    OutDataAnchorPtr peer_out_anchor = origin_input->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    NodePtr input_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(input_node);
    auto iter = origin_to_replaced.find(input_node->GetOutDataAnchor(peer_out->GetIdx()).get());
    if (iter == origin_to_replaced.end()) {
      GE_ASSERT_NOTNULL(peer_out->GetOwnerNode());
      GE_ASSERT_NOTNULL(peer_out->GetOwnerNode()->GetOpDesc());
      GELOGD("Graph %s add data for source %s", node->GetName().c_str(), peer_out->GetOwnerNode()->GetName().c_str());
      const auto op_desc = std::make_shared<OpDesc>(loop::BufferName(peer_out), DATA);
      op_desc->AddOutputDesc(peer_out->GetOwnerNode()->GetOpDesc()->GetOutputDesc(peer_out->GetIdx()));
      NodePtr data = graph->AddNode(op_desc);
      GE_ASSERT_NOTNULL(data);
      GE_ASSERT_NOTNULL(data->GetOutDataAnchor(0));
      origin_to_replaced[peer_out] = data->GetOutDataAnchor(0);
    }
  }
  return GRAPH_SUCCESS;
}

void LoweringUtils::PrintReadableAscGraph(const AscGraph &asc_graph) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return;
  }
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  const static auto kAxisStr = [](const vector<AxisPtr> &asc_axis) {
    return loop::StrJoin(asc_axis, [](const AxisPtr &axis) {
      const auto axis_size = axis->size.Str();
      return std::to_string(axis->id) + ":" + (axis_size == nullptr ? "nullptr" : axis_size.get());
    });
  };
  GELOGI("AscGraph(%s, axis=%s)", asc_graph.GetName().c_str(), kAxisStr(asc_graph.GetAllAxis()).c_str());
  for (const auto &node : asc_graph.GetAllNodes()) {
    std::vector<std::string> input_names;
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      const auto peer = anchor->GetPeerOutAnchor();
      if (peer != nullptr) {
        input_names.emplace_back(anchor_name[peer]);
      }
    }
    std::vector<std::string> output_names;
    std::map<std::string, std::string> output_loop;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
      const auto loop = GetAscTensorDescStr(anchor);
      if (!loop.empty()) {
        output_loop[output_names.back()] = loop;
      }
    }
    std::string output_name;
    if (output_names.size() > 1U) {
      output_name = loop::StrJoin(output_names) + " = ";
    } else if (!output_names.empty()) {
      output_name = output_names[0] + " = ";
    }
    GELOGI("%sascir.%s(%s, %s)", output_name.c_str(), node->GetType().c_str(), node->GetName().c_str(),
           loop::StrJoin(input_names).c_str());
    for (auto &loop : output_loop) {
      GELOGI("%s.attr = {%s}", loop.first.c_str(), loop.second.c_str());
    }
  }
}

bool LoweringUtils::IsAnyKernelBoxOversize(std::vector<loop::KernelBox> &kernel_boxes, const LoweringConfig &config) {
  for (auto &kernel_box : kernel_boxes) {
    if (kernel_box.NumOps() > config.max_loop_ops) {
      GELOGI("Kernel box %s num loop ops %zu > %zu", kernel_box.Name().c_str(), kernel_box.NumOps(),
             config.max_loop_ops);
      return true;
    }
    if (kernel_box.NumAscendNodes() > 1U && kernel_box.NumLoads() > config.max_loop_loads) {
      GELOGI("Kernel box %s num loads %zu > %zu", kernel_box.Name().c_str(), kernel_box.NumLoads(),
             config.max_loop_loads);
      return true;
    }
  }
  return false;
}

bool LoweringUtils::IsNodeCoreNumDif(const NodePtr &node) {
  int32_t cur_node_aiv_cnt = -1;
  if (ge::AttrUtils::HasAttr(node->GetOpDesc(), aiv_cnt_key)) {
    std::string aiv_cnt_value;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), aiv_cnt_key, aiv_cnt_value);
    GE_ASSERT_GRAPH_SUCCESS(TransCoreNumToInt(aiv_cnt_value, cur_node_aiv_cnt));
  }
  int32_t preorder_node_aiv_cnt = -1;
  for (auto &anchor : node->GetAllInDataAnchors()) {
    if ((anchor != nullptr) && (anchor->GetPeerOutAnchor() != nullptr)) {
      OutDataAnchorPtr peer_out_anchor = anchor->GetPeerOutAnchor();
      NodePtr input_node = peer_out_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(input_node);
      if (OpTypeUtils::IsConstNode(input_node->GetType()) || OpTypeUtils::IsDataNode(input_node->GetType())) {
        continue;
      }

      if (ge::AttrUtils::HasAttr(input_node->GetOpDesc(), aiv_cnt_key)) {
        std::string front_aiv_cnt_value;
        (void)ge::AttrUtils::GetStr(input_node->GetOpDesc(), aiv_cnt_key, front_aiv_cnt_value);
        GE_ASSERT_GRAPH_SUCCESS(TransCoreNumToInt(front_aiv_cnt_value, preorder_node_aiv_cnt));
      }
      if (preorder_node_aiv_cnt != cur_node_aiv_cnt) {
        GELOGD("Check core num scope dif, front node core num value is %d, cur node core num value is %d",
               preorder_node_aiv_cnt, cur_node_aiv_cnt);
        return true;
      }
    }
  }
  return false;
}

graphStatus LoweringUtils::SetAttrCoreNum(OpDescPtr &asc_desc, loop::KernelBox &kernel_box) {
  auto fuse_attrs = asc_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  auto one_ge_node = kernel_box.GetAscendIrNodes().at(0);
  int32_t cur_node_aiv_cnt = 0;
  if (ge::AttrUtils::HasAttr(one_ge_node->GetOpDesc(), aiv_cnt_key)) {
    std::string aiv_cnt_value;
    (void)ge::AttrUtils::GetStr(one_ge_node->GetOpDesc(), aiv_cnt_key, aiv_cnt_value);
    GE_ASSERT_GRAPH_SUCCESS(TransCoreNumToInt(aiv_cnt_value, cur_node_aiv_cnt));
  }
  fuse_attrs->SetVectorCoreNum(cur_node_aiv_cnt);
  return GRAPH_SUCCESS;
}

graphStatus LoweringUtils::GetOriginToReplaced(const Node *const &node, const ComputeGraphPtr &graph,
                                               std::map<const OutDataAnchor *, OutDataAnchorPtr> &origin_to_replaced) {
  GE_ASSERT_NOTNULL(node);
  GELOGD("Copy node %s to construct original compute graph for ascbackend", node->GetName().c_str());
  GE_ASSERT(AddDataNodeForConstructGraph(node, graph, origin_to_replaced) == GRAPH_SUCCESS);
  const auto copied = graph->AddNode(GraphUtils::CopyOpDesc(node->GetOpDesc()));
  GE_ASSERT_NOTNULL(copied);
  std::vector<std::string> input_node_names;
  for (uint32_t i = 0U; i < node->GetAllInDataAnchorsSize(); i++) {
    auto copied_input = copied->GetInDataAnchor(static_cast<int32_t>(i));
    auto origin_input = node->GetInDataAnchor(static_cast<int32_t>(i));
    auto peer_out = origin_input->GetPeerOutAnchor().get();
    if (peer_out != nullptr) {
      NodePtr input_node = origin_input->GetPeerOutAnchor()->GetOwnerNode();
      GE_CHECK_NOTNULL(input_node);
      if (std::find(input_node_names.begin(), input_node_names.end(),
                    input_node->GetName()) != input_node_names.end()) {
        GELOGD("Input node %s is same, skip add edge.", input_node->GetName().c_str());
        continue;
      }
      input_node_names.emplace_back(input_node->GetName());
      GELOGD("Graph %s add edge from %s to %s", node->GetName().c_str(), loop::BufferName(peer_out).c_str(),
             loop::BufferName(copied_input).c_str());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(origin_to_replaced[peer_out], copied_input));
    }
  }
  for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); i++) {
    origin_to_replaced[node->GetOutDataAnchor(static_cast<int32_t>(i)).get()] =
        copied->GetOutDataAnchor(static_cast<int32_t>(i));
  }
  return GRAPH_SUCCESS;
}

std::string LoweringUtils::GetConstructDumpGraphName(const NodePtr &node) {
  std::string node_name = node->GetName();
  uint32_t pos_size_one = 1U;
  size_t pos1 = node_name.find('_');
  if (pos1 == std::string::npos) {
    return node_name;
  }
  size_t pos2 = node_name.find('_', pos1 + pos_size_one);
  if (pos2 == std::string::npos) {
    return node_name;
  }
  std::string part_node_name = node_name.substr(pos1 + pos_size_one, pos2 - pos1 - pos_size_one);
  bool is_node_name_from_lowering = false;
  for (char each_cha : part_node_name) {
    if (!isdigit(each_cha)) {
      is_node_name_from_lowering = true;
      GELOGD("Node name[%s] is generated from lowering.", node_name.c_str());
      break;
    }
  }
  std::string node_name_fragment;
  if (is_node_name_from_lowering) {
    size_t pos3 = node_name.find('_', pos2 + pos_size_one);
    if (pos3 == std::string::npos) {
      return node_name;
    }
    node_name_fragment = node_name.substr(0, pos3);
  } else {
    node_name_fragment = node_name.substr(0, pos2);
  }
  return node_name_fragment;
}

graphStatus LoweringUtils::CheckSpecialFuseType(loop::KernelBox &kernel_box,
                                                std::shared_ptr<ge::loop::AscOverrides>&asc_graph) {
  if (!kernel_box.IsCube() && !(kernel_box.Type() == ge::loop::FuseType::kSliceSplit)) {
    GE_WARN_ASSERT(!asc_graph->IsScalarGraph(),
                   "Fall back lowering for node scope: %s. As unsupported scalar AscendC IR graph for kernel box %s",
                   kernel_box.DebugString().c_str(), kernel_box.Name().c_str());
  } else if (kernel_box.Type() == ge::loop::FuseType::kSliceSplit) {
    GE_WARN_ASSERT(!asc_graph->IsAscAxisEmpty(),
                   "Fall back lowering for node scope: %s. As unsupported scalar AscendC IR graph for kernel box %s",
                   kernel_box.DebugString().c_str(), kernel_box.Name().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringUtils::SetStreamLabelForOpDesc(loop::KernelBox &kernel_box, OpDescPtr &asc_desc) {
  if (!kernel_box.StreamLabel().empty()) {
    GELOGI("Set stream label %s and priority %s for kernel box %s", kernel_box.StreamLabel().c_str(),
           kernel_box.StreamPriority().c_str(), kernel_box.Name().c_str());
    GE_WARN_ASSERT(AttrUtils::SetStr(asc_desc, public_attr::USER_STREAM_LABEL, kernel_box.StreamLabel()),
                   "Fall back lowering for node scope: %s. As failed to set USER_STREAM_LABEL.",
                   kernel_box.DebugString().c_str());
    GE_WARN_ASSERT(AttrUtils::SetStr(asc_desc, public_attr::USER_STREAM_PRIORITY, kernel_box.StreamPriority()),
                   "Fall back lowering for node scope: %s. As failed to set USER_STREAM_PRIORITY.",
                   kernel_box.DebugString().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringUtils::AddDataEdgesForAscNode(const NodePtr &asc_node, const std::vector<const OutDataAnchor *> &inputs,
                                                  ge::OutDataAnchor *origin_output, std::set<const ge::Node *> &used_in_nodes) {
  const auto asc_out_anchor = asc_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(asc_out_anchor);
  for (const auto &dst_anchor : origin_output->GetPeerInDataAnchors()) {
    GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(origin_output).c_str(),
           loop::BufferName(dst_anchor).c_str(), loop::BufferName(asc_out_anchor).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(origin_output->shared_from_this(), dst_anchor));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(asc_out_anchor, dst_anchor));
  }

  for (size_t i = 0U; i < inputs.size(); i++) {
    const auto ascend_out = const_cast<ge::OutDataAnchor *>(inputs[i]);
    GE_ASSERT_NOTNULL(ascend_out);
    used_in_nodes.insert(ascend_out->GetOwnerNode().get());
    auto asc_input_anchor = asc_node->GetInDataAnchor(static_cast<int32_t>(i));
    GELOGD("Add new data edge %s->%s", loop::BufferName(ascend_out).c_str(),
           loop::BufferName(asc_input_anchor).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(ascend_out->shared_from_this(), asc_input_anchor));
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringUtils::GetUnusedInNodes(loop::KernelBox &kernel_box, const std::set<const ge::Node *> &used_in_nodes,
                                            std::set<NodePtr> &unused_in_nodes) {
  GELOGD("Start get unused in nodes for kernel box %s", kernel_box.Name().c_str());
  const std::vector<const ge::Node *> fused_nodes = kernel_box.GetAscendIrNodes();
  std::stack<NodePtr> stack;
  stack.push(kernel_box.TargetBuffer()->GetOwnerNode());
  while (!stack.empty()) {
    GELOGD("Start find unused in nodes of node %s", stack.top()->GetName().c_str());
    const auto current = stack.top();
    stack.pop();
    for (auto &in_node : current->GetInDataNodes()) {
      if (std::find(fused_nodes.begin(), fused_nodes.end(), in_node.get()) != fused_nodes.end()) {
        stack.push(in_node);
        continue;
      }
      if (used_in_nodes.find(in_node.get()) != used_in_nodes.end()) {
        continue;
      }
      GELOGD("Found unused in node %s", in_node->GetName().c_str());
      unused_in_nodes.insert(in_node);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringUtils::MoveControlEdges(const NodePtr &src, const NodePtr &dst) {
  for (auto &n : src->GetInControlNodes()) {
    // never change any control or data input edge of src node
    GELOGD("Add new control edge %s->%s", loop::BufferName(n->GetOutControlAnchor()).c_str(),
           loop::BufferName(dst->GetInControlAnchor()).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(n->GetOutControlAnchor(), dst->GetInControlAnchor()));
  }
  for (auto &n : src->GetOutControlNodes()) {
    GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(src->GetOutControlAnchor()).c_str(),
           loop::BufferName(n->GetInControlAnchor()).c_str(), loop::BufferName(dst->GetOutControlAnchor()).c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(dst->GetOutControlAnchor(), n->GetInControlAnchor()));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src->GetOutControlAnchor(), n->GetInControlAnchor()));
  }
  return GRAPH_SUCCESS;
}

graphStatus LoweringUtils::AssembleConcreteEdges(loop::KernelBox &kernel_box, AutoFuseAttrs &fuse_attrs,
                                                 const std::vector<const ge::OutDataAnchor *> &origin_inputs) {
  const auto &concrete_edges = kernel_box.GetConcreteEdges();
  std::map<const ge::OutDataAnchor *, size_t> input_index;
  for (size_t i = 0U; i < origin_inputs.size(); ++i) {
    input_index[origin_inputs[i]] = i;
  }
  for (const auto &edge : concrete_edges) {
    auto iter = input_index.find(edge.first);
    GE_WARN_ASSERT(iter != input_index.end(), "Edge %s->%s consumed by kernel box %s is not input",
                   loop::BufferName(edge.first).c_str(), loop::BufferName(edge.second).c_str(),
                   kernel_box.Name().c_str());
    fuse_attrs.AddConcreteEdges(iter->second, edge.second);
  }
  return GRAPH_SUCCESS;
}

}