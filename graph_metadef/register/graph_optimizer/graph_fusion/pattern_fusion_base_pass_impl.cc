/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
namespace {
const std::string kAttrDumpAble = "_dump_able";
}
PatternFusionBasePassImpl::PatternFusionBasePassImpl() {}

PatternFusionBasePassImpl::~PatternFusionBasePassImpl() {
  for (auto pattern : patterns_) {
    if (pattern != nullptr) {
      delete pattern;
      pattern = nullptr;
    }
  }
  for (auto inner_pattern : inner_patterns_) {
    if (inner_pattern != nullptr) {
      delete inner_pattern;
      inner_pattern = nullptr;
    }
  }
}

const std::vector<FusionPattern *> &PatternFusionBasePassImpl::GetPatterns() { return patterns_; }

void PatternFusionBasePassImpl::GetPatterns(std::vector<FusionPattern *> &patterns) { patterns = patterns_; }

const std::vector<FusionPattern *> &PatternFusionBasePassImpl::GetInnerPatterns() { return inner_patterns_; }

void PatternFusionBasePassImpl::GetInnerPatterns(std::vector<FusionPattern *> &inner_patterns) {
  inner_patterns = inner_patterns_;
}

void PatternFusionBasePassImpl::SetPatterns(const std::vector<FusionPattern *> &patterns) { patterns_ = patterns; }

void PatternFusionBasePassImpl::SetInnerPatterns(const std::vector<FusionPattern *> &inner_patterns) {
  inner_patterns_ = inner_patterns;
}

void PatternFusionBasePassImpl::SetOpsKernelInfoStore(const OpsKernelInfoStorePtr &ops_kernel_info_store_ptr) {
  ops_kernel_info_store_ptr_ = ops_kernel_info_store_ptr;
}

bool PatternFusionBasePassImpl::CheckOpSupported(const ge::OpDescPtr &op_desc_ptr) const {
  std::string un_supported_reason;

  if (ops_kernel_info_store_ptr_ == nullptr) {
    un_supported_reason = "opsKernelInfoStorePtr in PatternFusionBasePass is nullptr.";
    return false;
  }

  return ops_kernel_info_store_ptr_->CheckSupported(op_desc_ptr, un_supported_reason);
}

bool PatternFusionBasePassImpl::CheckOpSupported(const ge::NodePtr &node) const {
  std::string un_supported_reason;

  if (ops_kernel_info_store_ptr_ == nullptr) {
    un_supported_reason = "opsKernelInfoStorePtr in PatternFusionBasePass is nullptr.";
    return false;
  }

  return ops_kernel_info_store_ptr_->CheckSupported(node, un_supported_reason);
}

bool PatternFusionBasePassImpl::CheckAccuracySupported(const ge::NodePtr &node) const {
  if (node == nullptr) {
    GELOGD("Node is null.");
    return false;
  }
  if (ops_kernel_info_store_ptr_ == nullptr) {
    GELOGD("Ops kernel info store is null.");
    return false;
  }
  std::string un_supported_reason;
  const bool ret = ops_kernel_info_store_ptr_->CheckAccuracySupported(node, un_supported_reason, true);
  GELOGD("Check result for op[%s, %s] is [%d], reason is [%s].",
         node->GetName().c_str(), node->GetType().c_str(), ret, un_supported_reason.c_str());
  return ret;
}

bool PatternFusionBasePassImpl::IsNodesExist(const ge::NodePtr &current_node, const std::vector<ge::NodePtr> &nodes) {
  return find(nodes.begin(), nodes.end(), current_node) != nodes.end();
}

bool PatternFusionBasePassImpl::IsMatched(const std::shared_ptr<OpDesc> op_desc, const ge::NodePtr node,
                                          const Mapping &mapping) {
  if ((op_desc == nullptr) || (node == nullptr)) {
    GELOGD("opDesc or node could not be null");
    return false;
  }

  const auto iter = mapping.find(op_desc);

  // check op_desc does not exist in mapping
  return (iter != mapping.end()) && (find(iter->second.begin(), iter->second.end(), node) != iter->second.end());
}

void PatternFusionBasePassImpl::DumpMappings(const FusionPattern &pattern, const Mappings &mappings) const {
  std::ostringstream oss;
  oss << std::endl << "Mappings of pattern ";
  oss << pattern.GetName() << ":" << std::endl;
  for (size_t i = 0; i < mappings.size(); i++) {
    const Mapping &mapping = mappings[i];
    oss << " Mapping " << (i + 1) << "/" << mappings.size() << ":" << std::endl;
    for (const auto &item : mapping) {
      const std::shared_ptr<OpDesc> op_desc = item.first;
      const ge::NodePtr node = item.second[0];
      if ((op_desc != nullptr) && (node != nullptr)) {
        oss << "    " << op_desc->id << " -> " << node->GetName() << std::endl;
      }
    }
  }
  GELOGD("%s", oss.str().c_str());
}

bool PatternFusionBasePassImpl::IsOpTypeExist(const std::string &type, const std::vector<std::string> &types) {
  return find(types.begin(), types.end(), type) != types.end();
}

bool PatternFusionBasePassImpl::GetSortedInAnchors(const ge::NodePtr &node, const std::string&op_id,
                                                   std::vector<ge::InDataAnchorPtr> &in_anchors) const {
  if (node->GetInDataNodes().empty()) {
    GELOGW("[Match][Output] in data nodes of op %s is empty, pattern matching failed.", op_id.c_str());
    return false;
  }

  /* Input anchors should have an order. */
  GetInDataAnchors(node, in_anchors);
  if (in_anchors.empty()) {
    GELOGW("[Match][Output] The data anchor for op %s is empty, leading to a failure in pattern matching.", op_id.c_str());
    return false;
  }

  std::sort(in_anchors.begin(), in_anchors.end(),
            [](const ge::InDataAnchorPtr &a, const ge::InDataAnchorPtr &b) { return a->GetIdx() < b->GetIdx(); });
  return true;
}

bool PatternFusionBasePassImpl::MatchFromOutput(const ge::NodePtr output_node,
                                                const std::shared_ptr<OpDesc> output_op_desc, Mapping &mapping) const {
  if ((output_node == nullptr) || (output_op_desc == nullptr)) {
    GELOGW("[Match][Output] Output node or op_desc is null, pattern matching failed.");
    return false;
  }
  CandidateAndMapping cand(mapping);
  cand.candidate_nodes = {output_node};
  cand.candidate_op_descs = {output_op_desc};

  // store the nodes matched
  cand.mapping[output_op_desc].push_back(output_node);

  // match candidate node one by one
  while ((!cand.candidate_nodes.empty()) && (!cand.candidate_op_descs.empty())) {
    // get the first candidate node
    bool result = MatchFromOutput(cand);
    if (!result) {
      return false;
    }

    result = MatchOutputs(cand);
    if (!result) {
      return false;
    }
    // current op is matched successfully, thus remove it from candidate list
    (void)cand.candidate_nodes.erase(cand.candidate_nodes.cbegin());
    (void)cand.candidate_op_descs.erase(cand.candidate_op_descs.cbegin());

    // the sizes of candidate_nodes and candidate_op_descs should always keep the same
    if (cand.candidate_nodes.size() != cand.candidate_op_descs.size()) {
      GELOGW("[Match][Output] candidate_nodes_num != candidate_op_descs_num, pattern matching failed.");
      return false;
    }
  }

  // if candidate_nodes(or candidate_op_descs) is empty, the matching is done
  // successfully
  return cand.candidate_op_descs.empty();
}

bool PatternFusionBasePassImpl::VerifyInputDescNodes(const ge::NodePtr &input_node,
                                                     const std::shared_ptr<OpDesc> &input_desc,
                                                     const Mapping &mapping) {
  if (input_node == nullptr) {
    return true;
  }
  if (!input_desc->check_unique) {
    return true;
  }

  // if this input desc has been matched before, the current nodes should among the matched nodes
  auto iter = mapping.find(input_desc);
  if (iter == mapping.cend() || iter->second.empty()) {
    return true;
  }
  return std::find(iter->second.begin(), iter->second.end(), input_node) != iter->second.end();
}

bool PatternFusionBasePassImpl::MatchFromOutput(CandidateAndMapping &cand) const {
  if (cand.candidate_nodes.empty() || cand.candidate_op_descs.empty()) {
    GELOGW("[Match][Output] Either candidate_nodes or candidate_op_descs is empty, resulting in pattern matching failure.");
    return false;
  }
  const ge::NodePtr node = cand.candidate_nodes.front();
  std::shared_ptr<OpDesc> op_desc = cand.candidate_op_descs.front();
  const std::string op_id = op_desc->id;
  // add the input nodes into candidate list
  const std::vector<std::shared_ptr<OpDesc>> * const inputs_desc = FusionPattern::GetInputs(op_desc);
  if (inputs_desc == nullptr) {
    GELOGW("[Match][Output] Failed to get input_desc for op %s, pattern matching failed.", op_id.c_str());
    return false;
  }

  if (inputs_desc->empty()) {
    return true;
  }
  std::vector<ge::InDataAnchorPtr> in_anchors;
  if (!GetSortedInAnchors(node, op_id, in_anchors)) {
    return false;
  }
  // set flag for edge using
  const std::unique_ptr<bool[]> usage_flags(new (std::nothrow) bool[inputs_desc->size()]{});
  for (const auto &in_anchor : in_anchors) {
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      GELOGE(ge::FAILED, "Peer anchor is null.");
      return false;
    }
    const ge::NodePtr input_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    for (uint32_t j = 0U; j < inputs_desc->size(); j++) {
      const std::shared_ptr<OpDesc> &input_desc = inputs_desc->at(static_cast<size_t>(j));
      if (input_desc == nullptr) {
        GELOGW("[Match][Output] input_desc %u for op %s is null, pattern matching failed.", j, op_id.c_str());
        return false;
      }

      const bool matching_result =
          (IsOpTypeExist(ge::NodeUtils::GetNodeType(*input_node), input_desc->types) || input_desc->types.empty()) &&
          ((!usage_flags[static_cast<size_t>(j)]) || input_desc->repeatable) &&
          IsOpFusible(input_node->GetOpDesc(), input_desc) &&
          VerifyInputDescNodes(input_node, input_desc, cand.mapping);
      if (!matching_result) {
        continue;
      }

      // some nodes might be the input of multiple nodes, we use
      // IsMatched() to avoid repeat
      AddCandidateQueue(input_desc, input_node, cand);
      usage_flags[static_cast<size_t>(j)] = true;
      break;
    }
  }

  // return false if not all edges are matched
  if (!MatchAllEdges(inputs_desc->size(), usage_flags)) {
    GELOGD("[Match][Output] Not all inputs of op %s were matched; pattern matching did not succeed.", op_id.c_str());
    return false;
  }

  return true;
}

void PatternFusionBasePassImpl::AddCandidateQueue(const FusionPattern::OpDescPtr &op_desc,
                                                  const ge::NodePtr &node,
                                                  CandidateAndMapping &cand) const {
  if (IsMatched(op_desc, node, cand.mapping)) {
    return;
  }
  (void)cand.candidate_nodes.emplace_back(node);
  (void)cand.candidate_op_descs.emplace_back(op_desc);
  (void)cand.mapping[op_desc].emplace_back(node);
}

void PatternFusionBasePassImpl::MatchOneOutputNode(const ge::NodePtr &output_node,
                                                   const std::vector<FusionPattern::OpDescPtr> &outputs_desc,
                                                   size_t &out_idx, const std::unique_ptr<bool[]> &usage_flags,
                                                   CandidateAndMapping &cand) const {
  if (output_node == nullptr) {
    return;
  }
  for (size_t i = 0; i < outputs_desc.size(); i++) {
    const FusionPattern::OpDescPtr &output_desc = outputs_desc.at(i);
    const bool is_matched =
        (IsOpTypeExist(ge::NodeUtils::GetNodeType(*output_node), output_desc->types) || output_desc->types.empty()) &&
        (!usage_flags[out_idx + i]) && IsOpFusible(output_node->GetOpDesc(), output_desc);
    if (!is_matched) {
      continue;
    }
    AddCandidateQueue(output_desc, output_node, cand);
    usage_flags[out_idx + i] = true;
    break;
  }
}

void PatternFusionBasePassImpl::MatchFuzzyOutputs(const ge::NodePtr &node, const FusionPattern::OpDescPtr &op_desc,
                                                  size_t &out_idx, const std::unique_ptr<bool[]> &usage_flags,
                                                  CandidateAndMapping &cand) const {
  const FusionPattern::OutputMapDesc &outputs_desc_map = FusionPattern::GetOutputs(op_desc);
  auto peer_in_nodes = node->GetOutDataNodes();
  for (const auto &outputs_desc_pair : outputs_desc_map) {
    if (outputs_desc_pair.first != kFuzzyOutIndex) {
      continue;
    }

    for (const auto &peer_in_node : peer_in_nodes) {
      MatchOneOutputNode(peer_in_node, outputs_desc_pair.second, out_idx, usage_flags, cand);
    }
    if (out_idx > (std::numeric_limits<size_t>::max() - outputs_desc_pair.second.size())) {
      GELOGE(ge::FAILED, "Out idx %zu is overflow.", out_idx);
      return;
    }
    out_idx += outputs_desc_pair.second.size();
  }
}

void PatternFusionBasePassImpl::UpdateCandidates(
    const CandidateAndMapping &temp_cand, CandidateAndMapping &cand) const {
  if (temp_cand.candidate_op_descs.size() != temp_cand.candidate_nodes.size()) {
    return;
  }

  for (size_t i = 0; i < temp_cand.candidate_nodes.size(); i++) {
    AddCandidateQueue(temp_cand.candidate_op_descs[i], temp_cand.candidate_nodes[i], cand);
  }
}

bool PatternFusionBasePassImpl::MatchOutputs(CandidateAndMapping &cand) const {
  const auto &node = cand.candidate_nodes.front();
  const FusionPattern::OpDescPtr &op_desc = cand.candidate_op_descs.front();
  const std::string op_id = op_desc->id;
  const FusionPattern::OutputMapDesc &outputs_desc_map = FusionPattern::GetOutputs(op_desc);
  if (outputs_desc_map.empty()) {
    return true;
  }
  const size_t outputs_desc_size = FusionPattern::GetOutputSize(op_desc);
  if (op_desc->is_output_fullmatch && node->GetOutDataNodesSize() != outputs_desc_size) {
    GELOGW("[Match][Input] Full match mode: op %s description size (%zu) does not match output data node size (%u)", op_id.c_str(),
           outputs_desc_size, node->GetOutDataNodesSize());
    return false;
  }

  const std::unique_ptr<bool[]> usage_flags(new (std::nothrow) bool[outputs_desc_size] {});
  std::vector<ge::OutDataAnchorPtr> out_anchors;
  GetOutDataAnchors(node, out_anchors);

  size_t out_idx = 0;
  MatchFuzzyOutputs(node, op_desc, out_idx, usage_flags, cand);
  for (const auto &out_anchor : out_anchors) {
    if (outputs_desc_map.find(out_anchor->GetIdx()) == outputs_desc_map.end()) {
      GELOGW("[Match][Input] op %s out anchor idx: %d not configured in pattern", op_id.c_str(), out_anchor->GetIdx());
      continue;
    }
    const std::vector<FusionPattern::OpDescPtr> &outputs_desc = outputs_desc_map.at(out_anchor->GetIdx());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      const ge::NodePtr output_node = peer_in_anchor->GetOwnerNode();
      MatchOneOutputNode(output_node, outputs_desc, out_idx, usage_flags, cand);
    }
    out_idx += outputs_desc.size();
  }

  if (!MatchAllEdges(outputs_desc_size, usage_flags)) {
    GELOGW("[Match][Input] Not all outputs of op %s are matched; pattern matching failed.", op_id.c_str());
    return false;
  }
  return true;
}

bool PatternFusionBasePassImpl::MatchAllEdges(const size_t &input_size, const std::unique_ptr<bool[]> &usage_flags) {
  for (size_t i = 0; i != input_size; i++) {
    if (!usage_flags[i]) {
      return false;
    }
  }
  return true;
}

void PatternFusionBasePassImpl::GetInDataAnchors(const ge::NodePtr &node,
                                                 std::vector<ge::InDataAnchorPtr> &in_anchor_vec) {
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    if ((in_anchor == nullptr) || (in_anchor->GetPeerOutAnchor() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr)) {
      continue;
    }
    in_anchor_vec.push_back(in_anchor);
  }
}

void PatternFusionBasePassImpl::GetOutDataAnchors(const ge::NodePtr &node,
                                                  std::vector<ge::OutDataAnchorPtr> &out_anchor_vec) {
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr || out_anchor->GetPeerInDataNodesSize() == 0) {
      continue;
    }
    (void)out_anchor_vec.emplace_back(out_anchor);
  }
}

bool PatternFusionBasePassImpl::GetMatchOutputNodes(const ge::ComputeGraph &graph, const FusionPattern &pattern,
                                                    std::vector<ge::NodePtr> &matched_output_nodes) const {
  const FusionPattern::OpDescPtr output_op_desc = pattern.GetOutput();
  if (output_op_desc == nullptr) {
    GELOGW("[Get][Output] output op_desc is null, pattern matching failed");
    return false;
  }

  NodeMapInfoPtr node_map_info = nullptr;
  // get nodes by type from node
  if (GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph) == SUCCESS) {
    for (auto &OutOpType : output_op_desc->types) {
      const auto iter = node_map_info->node_type_map->find(OutOpType);
      if (iter != node_map_info->node_type_map->end()) {
        for (auto iter_node = iter->second.cbegin(); iter_node != iter->second.cend(); iter_node++) {
          const ge::NodePtr node_ptr = iter_node->second;

          if (node_ptr->GetInDataNodes().empty() && node_ptr->GetOutAllNodes().empty()) {
            continue;
          }
          if (ge::NodeUtils::GetNodeType(*node_ptr) == OutOpType &&
              IsOpFusible(node_ptr->GetOpDesc(), output_op_desc)) {
            matched_output_nodes.push_back(node_ptr);
          }
        }
      }
    }
  } else {  // for each graph to find type
    for (ge::NodePtr &n : graph.GetDirectNode()) {
      if (IsOpTypeExist(ge::NodeUtils::GetNodeType(*n), output_op_desc->types) &&
          IsOpFusible(n->GetOpDesc(), output_op_desc)) {
        matched_output_nodes.push_back(n);
      }
    }
  }

  if (matched_output_nodes.empty()) {
    return false;
  }
  return true;
}

const std::vector<ge::NodePtr>& PatternFusionBasePassImpl::GetActualFusedNodes() const {
  return actual_fused_nodes_;
}

void PatternFusionBasePassImpl::SetActualFusedNodes(const std::vector<ge::NodePtr> &fused_nodes) {
  actual_fused_nodes_ = fused_nodes;
}

bool PatternFusionBasePassImpl::IsOpFusible(const ge::OpDescPtr &op_desc, const FusionPattern::OpDescPtr &pattern_desc)
{
  if (op_desc == nullptr || pattern_desc == nullptr) {
    return false;
  }
  if (pattern_desc->allow_dumpable) {
    return true;
  }
  bool is_dump_able = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrDumpAble, is_dump_able);
  return !is_dump_able;
}
}
