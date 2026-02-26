/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "equivalent_graph_recognizer.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace att {
namespace {
std::vector<ge::AscNodePtr> GetAllNodes(const ge::AscGraph &graph) {
  std::vector<ge::AscNodePtr> nodes;
  for (const auto &node : graph.GetAllNodes()) {
    nodes.push_back(node);
  }
  return nodes;
}

std::map<int64_t, ge::AxisPtr> GetAllAxisInfo(const ge::AscGraph &graph) {
  std::map<int64_t, ge::AxisPtr> axes_info;
  for (auto &ax : graph.GetAllAxis()) {
    axes_info[ax->id] = ax;
  }
  return axes_info;
}

bool IsAxisEqual(const ge::AxisPtr &axis1, const ge::AxisPtr &axis2) {
  if (axis1->type != axis2->type) {
    GELOGD("Axis: [%d] and Axis: [%d] has different type [%d vs %d]", axis1->id, axis2->id, axis1->type, axis2->type);
    return false;
  }
  // 当前仅仅比较轴的切分方式是否一致
  return true;
}

bool IsMemAttrEqual(const ge::MemAttr &mem1, const ge::MemAttr &mem2) {
  return (mem1.alloc_type == mem2.alloc_type) && (mem1.position == mem2.position) &&
         (mem1.hardware == mem2.hardware) && (mem1.reuse_id == mem2.reuse_id);
}

bool IsCompareAxis(const ge::AxisPtr &axis) {
  return (axis->type == ge::Axis::Type::kAxisTypeOriginal) || (axis->type == ge::Axis::Type::kAxisTypeTileInner) ||
         (axis->type == ge::Axis::Type::kAxisTypeBlockInner) || (axis->type == ge::Axis::Type::kAxisTypeMerged);
}

std::vector<ge::AxisPtr> GetCompareAxes(const std::vector<ge::AxisPtr> &graph1_all_axes) {
  std::vector<ge::AxisPtr> graph1_axes;
  for (const auto &axis : graph1_all_axes) {
    if (IsCompareAxis(axis)) {
      graph1_axes.push_back(axis);
    }
  }
  return graph1_axes;
}
}

bool EquivalentGraphRecognizer::IsInputVar(const Expr &expr1, const Expr &expr2) const {
  const bool is_expr1_input_var = (expr1.GetExprType() == ge::ExprType::kExprVariable) &&
      (input_axes_name_to_.find(ge::SymbolicUtils::ToString(expr1)) != input_axes_name_to_.end());
  const bool is_expr2_input_var =
      (expr2.GetExprType() == ge::ExprType::kExprVariable) &&
      (input_axes_name_from_.find(ge::SymbolicUtils::ToString(expr2)) != input_axes_name_from_.end());
  return is_expr1_input_var && is_expr2_input_var;
}

EquivalentGraphRecognizer::EquivalentGraphRecognizer(const ge::AscGraph &graph_to, const ge::AscGraph &graph_from,
                                                     const ReuseScheduleGroupInfo &group_info_to,
                                                     const ReuseScheduleGroupInfo &group_info_from)
    : graph_to_(graph_to), graph_from_(graph_from), group_info_to_(group_info_to), group_info_from_(group_info_from) {
  axis_id_to_axis_map_to_ = GetAllAxisInfo(graph_to_);
  axis_id_to_axis_map_from_ = GetAllAxisInfo(graph_from_);
  search_axes_name_to_.insert(group_info_to_.reuse_search_axes.begin(), group_info_to_.reuse_search_axes.end());
  search_axes_name_from_.insert(group_info_from_.reuse_search_axes.begin(), group_info_from_.reuse_search_axes.end());
  input_axes_name_to_.insert(group_info_to_.reuse_input_axes.begin(), group_info_to_.reuse_input_axes.end());
  input_axes_name_from_.insert(group_info_from_.reuse_input_axes.begin(), group_info_from_.reuse_input_axes.end());
}

bool EquivalentGraphRecognizer::CompareAxis(const int64_t axis_id, const int64_t axis_id2) const {
  const auto iter1 = axis_id_to_axis_map_to_.find(axis_id);
  const auto iter2 = axis_id_to_axis_map_from_.find(axis_id2);
  if ((iter1 != axis_id_to_axis_map_to_.cend()) && (iter2!= axis_id_to_axis_map_from_.cend())) {
    auto axis1_from = iter1->second->from;
    auto axis2_from = iter2->second->from;
    if (axis1_from.size() != axis2_from.size()) {
      GELOGD("Axis: [%d] and Axis: [%d] has different parent axis size [%zu vs %zu]", axis_id, axis_id2,
             axis1_from.size(), axis2_from.size());
      return false;
    }
    if (axis1_from.empty() && axis2_from.empty()) {
      return IsAxisEqual(iter1->second, iter2->second);
    }
    for (size_t i = 0; i < axis1_from.size(); ++i) {
      bool is_equal = CompareAxis(axis1_from[i], axis2_from[i]);
      if (!is_equal) {
        GELOGD("Axis: [%d] and Axis: [%d] has different parent axis [%d vs %d]", axis_id, axis_id2,
               axis1_from[i], axis2_from[i]);
        return false;
      }
    }
  }
  return true;
}

bool EquivalentGraphRecognizer::IsMemEquivalent(const ge::AscTensorAttr &tensor1,
                                                const ge::AscTensorAttr &tensor2) const {
  const auto &mem1 = tensor1.mem;
  const auto &mem2 = tensor2.mem;
  // 构图连边一致、节点类型一致、数据类型一致，切分方式也一致，当前认为内存复用也是一致的，暂不做额外的比较
  return IsMemAttrEqual(mem1, mem2);
}

std::string EquivalentGraphRecognizer::ReplaceSearchVarStr(const std::string &str) const {
  std::string cp_str(str);
  for (size_t i = 0UL; i < group_info_to_.reuse_search_axes.size(); ++i) {
    const std::string &reuse_search_axes1 = group_info_to_.reuse_search_axes[i];
    const std::string &reuse_search_axes2 = group_info_from_.reuse_search_axes[i];
    size_t pos = 0UL;
    while ((pos = cp_str.find(reuse_search_axes1, pos)) != std::string::npos) {
      cp_str.replace(pos, reuse_search_axes1.length(), reuse_search_axes2);
      pos += reuse_search_axes2.length();
    }
  }
  GELOGD("Before replace search var %s, after replace search var %s", str.c_str(), cp_str.c_str());
  return cp_str;
}

bool EquivalentGraphRecognizer::CanExprEquivalentAfterReplace(const ge::Expression &replace_expr,
                                                              const ge::Expression &reuse_expr) {
  // 需要对expr1尝试进行原始轴替换，检查替换后是否可以和reuse_expr相等，相等的话就可以复用，先不考虑需要替换多个原始轴的场景
  for (const auto &input_var : input_axes_name_to_) {
    std::vector<std::pair<Expr, Expr>> var_replacement;
    var_replacement.emplace_back(std::make_pair(ge::Symbol(input_var.c_str()), reuse_expr));
    auto replace_expr2 = replace_expr.Replace(var_replacement);
    const auto &replace_expr3 = ReplaceSearchVarStr(ge::SymbolicUtils::ToString(replace_expr2));
    const auto &reuse_expr3 = ReplaceSearchVarStr(ge::SymbolicUtils::ToString(reuse_expr));
    GELOGD("After replace input var[%s] is %s, reuse expr %s, replace org expr %s, replace reuse_expr %s",
           input_var.c_str(), ge::SymbolicUtils::ToString(replace_expr2).c_str(),
           ge::SymbolicUtils::ToString(reuse_expr).c_str(), replace_expr3.c_str(),
           reuse_expr3.c_str());
    if (replace_expr3 == reuse_expr3) {
      const auto &iter = mapped_input_axes_names_.find(ge::SymbolicUtils::ToString(reuse_expr));
      if (iter == mapped_input_axes_names_.end()) {
        GELOGD("Update map input axes %s->%s of graph %s->%s", ge::SymbolicUtils::ToString(reuse_expr).c_str(),
            input_var.c_str(), graph_from_.GetName().c_str(), graph_to_.GetName().c_str());
        mapped_input_axes_names_[ge::SymbolicUtils::ToString(reuse_expr)] = input_var;
        return true;
      }
      // 当前输入轴已经被映射过，但是映射的结果不一致，认为不一致
      if (iter->second != input_var) {
        GELOGD("Current map is %s->%s, but already has map %s->%s of graph %s->%s",
            ge::SymbolicUtils::ToString(reuse_expr).c_str(),
            input_var.c_str(), ge::SymbolicUtils::ToString(reuse_expr).c_str(),
            iter->second.c_str(), graph_from_.GetName().c_str(),
            graph_to_.GetName().c_str());
        return false;
      }
      return true;
    }
    GELOGD("Expression: [%s] and Expression: [%s] has different free symbols", replace_expr3.c_str(),
           reuse_expr3.c_str());
  }
  // 尝试替换所有输入轴后，都不相等，返回false
  return false;
}

bool EquivalentGraphRecognizer::CompareExpression(const ge::Expression &expr1, const ge::Expression &expr2) {
  if (expr1.GetExprType() != expr2.GetExprType()) {
    GELOGD("Expression: [%s] and Expression: [%s] has different type [%d vs %d]",
        ge::SymbolicUtils::ToString(expr1).c_str(), ge::SymbolicUtils::ToString(expr2).c_str(),
        static_cast<int32_t>(expr1.GetExprType()), static_cast<int32_t>(expr2.GetExprType()));
    return false;
  }
  if (expr1.IsConstExpr() && expr2.IsConstExpr()) {
    if (expr1 != expr2) {
      GELOGD("Expression: [%s] and Expression: [%s] has different const value", expr1.Serialize().get(),
             expr2.Serialize().get());
      return false;
    }
    return true;
  }
  // 替换Search变量
  const auto &free_sym1 = expr1.FreeSymbols();
  const auto &free_sym2 = expr2.FreeSymbols();
  if (free_sym1.size() != free_sym2.size()) {
    GELOGD("Expression: [%s] and Expression: [%s] has different free symbols size [%zu vs %zu]",
        ge::SymbolicUtils::ToString(expr1).c_str(), ge::SymbolicUtils::ToString(expr2).c_str(),
        free_sym1.size(), free_sym2.size());
    return false;
  }
  std::vector<std::pair<Expr, Expr>> var_replacement;
  // 针对非输入轴的替换，让sym1使用sym2的非输入轴符号，得到新的replaced_expr1
  for (size_t i = 0UL; i < free_sym1.size(); i++) {
    const auto &replace_expr = free_sym1[i];
    const auto &reuse_expr = free_sym2[i];
    if (!IsInputVar( replace_expr, reuse_expr)) {
      var_replacement.emplace_back(std::make_pair( replace_expr, reuse_expr));
    }
  }
  auto replaced_expr1 = expr1.Replace(var_replacement);
  GELOGD("After replace var is %s, original expr %s, compare expr %s",
         ge::SymbolicUtils::ToString(replaced_expr1).c_str(), ge::SymbolicUtils::ToString(expr1).c_str(),
         ge::SymbolicUtils::ToString(expr2).c_str());
  const auto &replaced_free_sym1 = replaced_expr1.FreeSymbols();
  // 针对替换轴后的表达式进行输入轴的替换
  for (size_t i = 0UL; i < replaced_free_sym1.size(); i++) {
    const auto &replaced_expr = replaced_free_sym1[i];
    const auto &reuse_expr = free_sym2[i];
    GELOGD("Sub expression replace_expr vs reuse_expr: %s vs %s", ge::SymbolicUtils::ToString(replaced_expr).c_str(),
           ge::SymbolicUtils::ToString(reuse_expr).c_str());
    if (IsInputVar(replaced_expr, reuse_expr)) {
      if (CanExprEquivalentAfterReplace(replaced_expr, reuse_expr)) {
        return true;
      }
    }
  }
  return replaced_expr1 == expr2;
}

bool EquivalentGraphRecognizer::CompareExprs(const std::vector<ge::Expression> &exprs1,
                                             const std::vector<ge::Expression> &exprs2) {
  for (size_t i = 0; i < exprs1.size(); ++i) {
    if (!CompareExpression(exprs1[i], exprs2[i])) {
      return false;
    }
  }
  return true;
}

bool EquivalentGraphRecognizer::IsTensorViewEquivalent(const ge::AscTensorAttr &tensor1,
                                                       const ge::AscTensorAttr &tensor2) {
  bool is_view_size_same = (tensor1.repeats.size() == tensor2.repeats.size()) &&
                           (tensor1.strides.size() == tensor2.strides.size()) &&
                           (tensor1.vectorized_strides.size() == tensor2.vectorized_strides.size()) &&
                           (tensor1.vectorized_axis.size() == tensor2.vectorized_axis.size());
  if (!is_view_size_same) {
    GELOGD("Tensor has different view size [%zu,%zu,%zu vs %zu,%zu,%zu]", tensor1.repeats.size(),
           tensor1.strides.size(), tensor1.axis.size(), tensor2.repeats.size(), tensor2.strides.size(),
           tensor2.axis.size());
    return false;
  }
  if (!CompareExprs(tensor1.repeats, tensor2.repeats)) {
    GELOGD("Tensor repeats is not equivalent");
    return false;
  }
  GELOGD("Tensors repeats are same, [%s] vs [%s]", GetVecString(tensor1.repeats).c_str(),
         GetVecString(tensor2.repeats).c_str());
  if (!CompareExprs(tensor1.strides, tensor2.strides)) {
    GELOGD("Tensor strides is not equivalent");
    return false;
  }
  GELOGD("Tensors strides are same, [%s] vs [%s]", GetVecString(tensor1.strides).c_str(),
         GetVecString(tensor2.strides).c_str());
  if (!CompareExprs(tensor1.vectorized_strides, tensor2.vectorized_strides)) {
    GELOGD("Tensor vectorized_strides is not equivalent");
    return false;
  }
  GELOGD("Tensors vectorized_strides are same, [%s] vs [%s]", GetVecString(tensor1.vectorized_strides).c_str(),
         GetVecString(tensor2.vectorized_strides).c_str());
  return true;
}

bool EquivalentGraphRecognizer::IsAscTensorEquivalent(const ge::AscTensorAttr &tensor1,
                                                      const ge::AscTensorAttr &tensor2) {
  if ((tensor1.axis.size() != tensor2.axis.size()) || (tensor1.dtype != tensor2.dtype)) {
    GELOGD("Tensor has different dtype [%d vs %d] or different axis size [%zu vs %zu]",
           tensor1.dtype.operator ge::DataType(), tensor2.dtype.operator ge::DataType(), tensor1.axis.size(),
           tensor2.axis.size());
    return false;
  }
  for (size_t i = 0; i < tensor1.axis.size(); ++i) {
    if (!CompareAxis(tensor1.axis[i], tensor2.axis[i])) {
      GELOGD("Tensor has different axis [%d vs %d]", tensor1.axis[i], tensor2.axis[i]);
      return false;
    }
  }
  if (!IsTensorViewEquivalent(tensor1, tensor2)) {
    GELOGD("Tensor is not equivalent");
    return false;
  }
  GELOGD("Tensors [%ld,%ld] views are same, reuse_id [%ld,%ld]", tensor1.mem.tensor_id, tensor2.mem.tensor_id,
         tensor1.mem.reuse_id, tensor2.mem.reuse_id);
  return IsMemEquivalent(tensor1, tensor2);
}

bool EquivalentGraphRecognizer::IsAscNodeEquivalent(ge::AscNode &node1, ge::AscNode &node2) {
  if (node1.GetType() != node2.GetType()) {
    GELOGD("Node: [%s] and Node: [%s] has different type [%s vs %s]", node1.GetName().c_str(),
           node2.GetName().c_str(), node1.GetType().c_str(), node2.GetType().c_str());
    return false;
  }
  const auto &node1_outputs = node1.outputs();
  const auto &node2_outputs = node2.outputs();
  if (node1_outputs.size() != node2_outputs.size()) {
    GELOGD("Node: [%s] and Node: [%s] has different output tensor size [%zu vs %zu]", node1.GetName().c_str(),
           node2.GetName().c_str(), node1_outputs.size(), node2_outputs.size());
    return false;
  }
  for (size_t i = 0; i < node1_outputs.size(); ++i) {
    const auto &tensor_att1 = node1_outputs[i];
    const auto &tensor_att2 = node2_outputs[i];
    GE_ASSERT_NOTNULL(tensor_att1);
    GE_ASSERT_NOTNULL(tensor_att2);
    if (!IsAscTensorEquivalent(tensor_att1->attr, tensor_att2->attr)) {
      GELOGD("Node: [%s] and Node: [%s] has different output tensor [%zu]", node1.GetName().c_str(),
             node2.GetName().c_str(), i);
      return false;
    }
  }
  return true;
}

bool EquivalentGraphRecognizer::IsInputNodeSame(const ge::AscNodePtr &asc_node1,
                                                const ge::AscNodePtr &asc_node2) {
  std::vector<ge::Node *> input_nodes1;
  for (const auto &input : asc_node1->inputs()) {
    GE_ASSERT_NOTNULL(input);
    const auto node = ge::ascir::AscTensorUtils::GetOwner(*input);
    GE_ASSERT_NOTNULL(node);
    input_nodes1.emplace_back(node);
  }
  std::vector<ge::Node *> input_nodes2;
  for (const auto &input : asc_node2->inputs()) {
    GE_ASSERT_NOTNULL(input);
    const auto node = ge::ascir::AscTensorUtils::GetOwner(*input);
    GE_ASSERT_NOTNULL(node);
    input_nodes2.emplace_back(node);
  }
  if (input_nodes2.size() != input_nodes1.size()) {
    GELOGD("Node: [%s] and Node: [%s] has different input size [%zu vs %zu]", asc_node1->GetName().c_str(),
           asc_node2->GetName().c_str(), input_nodes1.size(), input_nodes2.size());
    return false;
  }
  for (size_t i = 0UL; i < input_nodes1.size(); ++i) {
    ge::AscNode node1(input_nodes1[i]->GetOpDesc(), input_nodes1[i]->GetOwnerComputeGraph());
    ge::AscNode node2(input_nodes2[i]->GetOpDesc(), input_nodes2[i]->GetOwnerComputeGraph());
    if (!IsAscNodeEquivalent(node1, node2)) {
      GELOGD("Node: [%s] and Node: [%s] has different input node name [%s vs %s]", asc_node1->GetName().c_str(),
             asc_node2->GetName().c_str(), input_nodes1[i]->GetName().c_str(), input_nodes2[i]->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool EquivalentGraphRecognizer::UpdateOrderedInputNames() {
  // example1:
  // from s1,s2,s3,s4 s1->s3, s2->s1, s3->s2, s4->s4(mapped)
  // to   s1,s2,s3,s4 s3,     s1,     s2,     s4
  // example2:
  // from s1,s2,s3,s4 s1->s3, s2->s1(mapped)
  // to   s1,s2,s3,s4 s3,     s1(mapped),     s2,     s4(not mapped)
  std::set<std::string> not_mapped_input_axes(group_info_to_.reuse_input_axes.begin(),
                                              group_info_to_.reuse_input_axes.end());
  std::vector<std::string> mapped_input_axes;
  mapped_input_axes.reserve(group_info_to_.reuse_input_axes.size());
  graph_to_ordered_input_names_.resize(group_info_to_.reuse_input_axes.size());
  size_t input_id = 0UL;
  std::set<size_t> mapped_input_axis_ids;
  for (const auto &input_var : group_info_from_.reuse_input_axes) {
    const auto &iter = mapped_input_axes_names_.find(input_var);
    if (iter != mapped_input_axes_names_.end()) {
      graph_to_ordered_input_names_[input_id] = (iter->second);
      not_mapped_input_axes.erase(iter->second);
      mapped_input_axis_ids.insert(input_id);
    }
    input_id++;
  }
  for (size_t i = 0UL; i < group_info_to_.reuse_input_axes.size(); ++i) {
    if (mapped_input_axis_ids.insert(i).second) {
      GE_ASSERT_TRUE(!not_mapped_input_axes.empty());
      graph_to_ordered_input_names_[i] = *not_mapped_input_axes.begin();
      not_mapped_input_axes.erase(graph_to_ordered_input_names_[i]);
    }
  }
  GELOGD("Got Equal group graphs, ordered input names %s, mapped size %zu of graph %s->%s",
         ge::ToString(graph_to_ordered_input_names_).c_str(), mapped_input_axes_names_.size(),
         graph_from_.GetName().c_str(), graph_to_.GetName().c_str());
  return true;
}

bool EquivalentGraphRecognizer::IsInputAxesFromDuplicityMapped() const {
  // 检查是否有重复映射的轴
  std::set<std::string> mapped_to_input_axes_names;
  for (const auto &mapped_input_axis_name : mapped_input_axes_names_) {
    if (!mapped_to_input_axes_names.insert(mapped_input_axis_name.second).second) {
      GELOGD("Graph: [%s] and Graph: [%s]: mapped input axis[%s] has duplicate map to input axis[%s]",
             graph_to_.GetName().c_str(), graph_from_.GetName().c_str(), mapped_input_axis_name.first.c_str(),
             mapped_input_axis_name.second.c_str());
      return false;
    }
  }
  return true;
}

bool EquivalentGraphRecognizer::IsAscNodeAttrEquivalent(const ge::AscNodeAttr &node_attr_to,
                                                        const ge::AscNodeAttr &node_attr_from) {
  if (node_attr_to.sched.axis.size() != node_attr_from.sched.axis.size()) {
    GELOGD("Graph: [%s] and Graph: [%s] has different schedule axis size [%zu vs %zu]", graph_to_.GetName().c_str(),
           graph_from_.GetName().c_str(), node_attr_to.sched.axis.size(), node_attr_from.sched.axis.size());
    return false;
  }
  return true;
}

bool EquivalentGraphRecognizer::IsEquivalent() {
  std::vector<ge::AscNodePtr> graph1_nodes = GetAllNodes(graph_to_);
  std::vector<ge::AscNodePtr> graph2_nodes = GetAllNodes(graph_from_);
  auto graph1_nodes_size = graph1_nodes.size();
  auto graph2_nodes_size = graph2_nodes.size();
  if (graph1_nodes_size != graph2_nodes_size) {
    GELOGD("Graph: [%s] and Graph: [%s] has different node size [%zu vs %zu]", graph_to_.GetName().c_str(),
           graph_from_.GetName().c_str(), graph1_nodes_size, graph2_nodes_size);
    return false;
  }
  if ((group_info_to_.reuse_input_axes.size() != group_info_from_.reuse_input_axes.size()) ||
      (group_info_to_.reuse_search_axes.size() != group_info_from_.reuse_search_axes.size())) {
    GELOGD("AscGraphs are not equivalent, input axes[%zu vs %zu], search axes[%zu vs %zu]",
           group_info_to_.reuse_input_axes.size(), group_info_from_.reuse_input_axes.size(),
           group_info_to_.reuse_search_axes.size(), group_info_from_.reuse_search_axes.size());
    return false;
  }
  const auto &graph1_axes = GetCompareAxes(graph_to_.GetAllAxis());
  const auto &graph2_axes = GetCompareAxes(graph_from_.GetAllAxis());
  if (graph1_axes.size()!= graph2_axes.size()) {
    GELOGD("Graph: [%s] and Graph: [%s] has different axis size [%zu vs %zu]", graph_to_.GetName().c_str(),
           graph_from_.GetName().c_str(), graph1_axes.size(), graph2_axes.size());
    return false;
  }
  for (size_t i = 0; i < graph1_nodes_size; ++i) {
    const auto &graph1_node = graph1_nodes[i];
    const auto &graph2_node = graph2_nodes[i];
    GE_ASSERT_NOTNULL(graph1_node);
    GE_ASSERT_NOTNULL(graph2_node);
    if (!IsInputNodeSame(graph1_node, graph2_node)) {
      return false;
    }
    GELOGD("Nodes input node [%s,%s] are same", graph1_node->GetNamePtr(), graph2_node->GetNamePtr());
    if (!IsAscNodeAttrEquivalent(graph1_node->attr, graph2_node->attr)) {
      GELOGD("Graph: [%s] and Graph: [%s]: node[%d] has different node attr", graph_to_.GetName().c_str(),
             graph_from_.GetName().c_str(), i);
      return false;
    }
    GELOGD("Nodes [%s,%s] atts are same", graph1_node->GetNamePtr(), graph2_node->GetNamePtr());
    if (!IsAscNodeEquivalent(*graph1_node, *graph2_node)) {
      GELOGD("Graph: [%s] and Graph: [%s]: node[%d] is not equivalent", graph_to_.GetName().c_str(),
             graph_from_.GetName().c_str(), i);
      return false;
    }
    GELOGD("Nodes [%s,%s] are same", graph1_node->GetNamePtr(), graph2_node->GetNamePtr());
  }
  if (!IsInputAxesFromDuplicityMapped()) {
    return false;
  }
  return UpdateOrderedInputNames();
}
}
