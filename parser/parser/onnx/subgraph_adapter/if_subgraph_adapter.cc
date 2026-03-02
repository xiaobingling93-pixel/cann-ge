/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "if_subgraph_adapter.h"
#include <unordered_set>
#include "subgraph_adapter_factory.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace ge {
using parser::IF;
namespace {
const std::map<std::string, int> kAttrNameToIndex = {{"then_branch", 0}, {"else_branch", 1}};
const int kIfNodeAttrSize = 2;
const char *kIf = "If";
}  // namespace
domi::Status IfSubgraphAdapter::AdaptAndFindAllSubgraphs(
    ge::onnx::NodeProto *parent_node, std::vector<ge::onnx::GraphProto *> &onnx_graphs,
    std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph, const std::string &parent_graph_name) {
  GE_CHECK_NOTNULL(parent_node);
  GELOGI("Onnx parent node name=%s, op type=%s, adapt subgraph.", parent_node->name().c_str(),
         parent_node->op_type().c_str());

  auto ret = ParseIfNodeSubgraphs(*parent_node, onnx_graphs, name_to_onnx_graph, parent_graph_name);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Node] Parse if node failed.");
    return ret;
  }

  return SUCCESS;
}

domi::Status IfSubgraphAdapter::ParseIfNodeSubgraphs(
    ge::onnx::NodeProto &parent_node, std::vector<ge::onnx::GraphProto *> &onnx_graphs,
    std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph, const std::string &parent_graph_name) const {
  if (parent_node.attribute_size() != kIfNodeAttrSize) {
    GELOGE(FAILED, "[Parse][Node] Invalid graph, if node attribute size:%d must be 2.", parent_node.attribute_size());
    REPORT_INNER_ERR_MSG("E19999", "Invalid graph, if node attribute size:%d must be 2.", parent_node.attribute_size());
    return FAILED;
  }

  GELOGD("node attribute size:%d.", parent_node.attribute_size());
  std::set<std::string> all_inputs;
  // for onnx graph, the first attribute may be else branch and the second attribute may be then branch
  for (int i = 0; i < parent_node.attribute_size(); i++) {
    ge::onnx::AttributeProto *attribute = parent_node.mutable_attribute(i);
    GE_CHECK_NOTNULL(attribute);
    std::string attr_name = attribute->name();
    auto itr = kAttrNameToIndex.find(attr_name);
    if (itr == kAttrNameToIndex.end()) {
      GELOGE(FAILED, "[Parse][Attribute] Invalid attribute name:%s, it should be then_branch or else_branch.",
             attr_name.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Invalid attribute name:%s, it should be then_branch or else_branch.",
                         attr_name.c_str());
      return FAILED;
    }
    std::string unique_subgraph_name;
    std::string node_name = parent_node.name();
    if (!parent_graph_name.empty()) {
      node_name = OnnxUtil::GenUniqueNodeName(parent_graph_name, node_name);
    }
    OnnxUtil::GenUniqueSubgraphName(itr->second, itr->first, node_name, unique_subgraph_name);
    GELOGI("Adapt if node attribute:%s, subgraph name:%s.", attr_name.c_str(), unique_subgraph_name.c_str());
    ge::onnx::GraphProto *onnx_graph = attribute->mutable_g();
    name_to_onnx_graph[unique_subgraph_name] = onnx_graph;
    onnx_graphs.emplace_back(onnx_graph);

    auto ret = GetSubgraphsAllInputs(*onnx_graph, all_inputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][Inputs] Get subgraph all inputs failed, attr_name:%s.", attr_name.c_str());
      return ret;
    }
  }

  for (auto &onnx_graph : onnx_graphs) {
    AddInputNodeForGraph(all_inputs, *onnx_graph);
  }

  AddInputForParentNode(all_inputs, parent_node);
  return SUCCESS;
}

domi::Status IfSubgraphAdapter::GetSubgraphsAllInputs(ge::onnx::GraphProto &onnx_graph,
                                                      std::set<std::string> &all_inputs) const {
  std::unordered_set<std::string> graph_inputs;
  std::unordered_set<std::string> graph_outputs;
  for (int i = 0; i < onnx_graph.node_size(); i++) {
    ge::onnx::NodeProto *node_proto = onnx_graph.mutable_node(i);
    for (int j = 0; j < node_proto->input_size(); j++) {
      graph_inputs.emplace(node_proto->input(j));
    }
    for (int j = 0; j < node_proto->output_size(); j++) {
      graph_outputs.emplace(node_proto->output(j));
    }
  }
  std::unordered_set<std::string> graph_initializer_tensors;
  for (int32_t i = 0; i < onnx_graph.initializer_size(); i++) {
    graph_initializer_tensors.emplace(onnx_graph.initializer(i).name());
  }
  for (const auto &input : graph_inputs) {
    if (graph_outputs.count(input) == 0 && graph_initializer_tensors.count(input) == 0) {
      // Record input node need to be constructed
      all_inputs.emplace(input);
    }
  }

  return SUCCESS;
}

void IfSubgraphAdapter::AddInputNodeForGraph(const std::set<std::string> &all_inputs,
                                             ge::onnx::GraphProto &onnx_graph) const {
  for (const auto &input_name : all_inputs) {
    ge::onnx::ValueInfoProto *value_info = onnx_graph.add_input();
    value_info->set_name(input_name);
  }
}

void IfSubgraphAdapter::AddInputForParentNode(const std::set<std::string> &all_inputs,
                                              ge::onnx::NodeProto &parent_node) const {
  for (const auto &input_name : all_inputs) {
    parent_node.add_input(input_name);
  }
}
REGISTER_SUBGRAPH_ADAPTER_CREATOR(kIf, IfSubgraphAdapter);
}  // namespace ge
