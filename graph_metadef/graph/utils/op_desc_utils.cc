/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/op_desc_utils.h"

#include <queue>

#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/recover_ir_utils.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph/type/sym_dtype.h"
#include "graph/detail/model_serialize_imp.h"
#include "mmpa/mmpa_api.h"
#include "graph/operator_factory_impl.h"

/*lint -e512 -e737 -e752*/
namespace ge {
const char_t OP_DESC_QUANT_PARAMS[] = "quantize_factor";

namespace {
const uint32_t CONST_OP_NORMAL_WEIGHT_SIZE = 1U;
const char* const kMultiThreadCompile = "MULTI_THREAD_COMPILE";
const char* const kDisEnableFlag = "0";
void GetConstantOpName(std::string &op_name) {
  thread_local int64_t const_count = 0;
  std::string compile_thread;
  if ((ge::GetContext().GetOption(kMultiThreadCompile, compile_thread) == GRAPH_SUCCESS)
      && (compile_thread.compare(kDisEnableFlag) == 0)) {
    op_name = "dynamic_const_" + std::to_string(const_count);
  } else {
    op_name = "dynamic_const_" + std::to_string(GeLog::GetTid()) + "_" + std::to_string(const_count);
  }
  ++const_count;
}

bool FindSubsequentMatches(const std::map<uint32_t, std::string> &valid_index_2_names, size_t start_index,
                           const std::string &ir_name) {
  for (size_t i = start_index; i < valid_index_2_names.size(); ++i) {
    const auto name = valid_index_2_names.at(i);
    if (name == ir_name) {
      GELOGI("ir_name:%s, node input index:%zu", ir_name.c_str(), i);
      return true;
    }
  }
  return false;
}

std::string InputsNamesStr(const OpDescPtr &op_desc) {
  std::stringstream ss;
  ss << "node: " << op_desc->GetName() << "(" << op_desc->GetType() << ") ir inputs names: [";
  for (const auto &ir_input : op_desc->GetIrInputs()) {
    ss << ir_input.first << ", ";
  }
  ss << "], actual inputs names: [";
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); i++) {
    if (op_desc->MutableInputDesc(static_cast<uint32_t>(i)) != nullptr) {
      const auto valid_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(i));
      ss << valid_name << ", ";
    }
  }
  ss << "]";
  return ss.str();
}
}

bool OpDescUtils::ClearInputDesc(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
                   return false, "[Check][Param] node is nullptr");
  GE_CHK_BOOL_EXEC(node->GetOpDesc() != nullptr, REPORT_INNER_ERR_MSG("E18888", "opdesc is nullptr.");
                   return false, "[Check][Param] opdesc is nullptr");
  std::vector<int32_t> index_list;
  for (const auto &in_anchor : node->GetAllInDataAnchorsPtr()) {
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      index_list.push_back(in_anchor->GetIdx());
    }
  }
  std::sort(index_list.begin(), index_list.end());
  // Node's in anchor index need shrink
  if (node->GetOpDesc()->impl_ == nullptr) {
    GELOGE(FAILED, "[Clear][InputDesc] Op desc impl is nullptr. ");
    return false;
  }
  for (size_t i = 0UL; i < index_list.size(); ++i) {
    const auto iter = node->GetOpDesc()->impl_->inputs_desc_.begin() + static_cast<int64_t>(index_list[i]);
    if (iter < node->GetOpDesc()->impl_->inputs_desc_.end()) {
      (void)node->GetOpDesc()->impl_->inputs_desc_.erase(iter);
    } else {
      GELOGW("[Clear][InputDesc] inputs_desc_ iterator out of range.");
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearInputDesc(const OpDescPtr op_desc,
                                                                                const uint32_t index) {
  return NodeUtils::ClearInputDesc(op_desc, index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::HasQuantizeFactorParams(const OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    GELOGI("op_desc is nullptr");
    return false;
  }
  return op_desc->HasAttr(OP_DESC_QUANT_PARAMS);
}

bool OpDescUtils::ClearOutputDesc(const NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERR_MSG("E18888", "node is nullptr, check invalid.");
                   return false, "[Check][Param] node is nullptr");
  GE_CHK_BOOL_EXEC(node->GetOpDesc() != nullptr, REPORT_INNER_ERR_MSG("E18888", "opdesc is nullptr.");
                   return false, "[Check][Param] opdesc is nullptr");
  std::vector<int32_t> index_list;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor->GetPeerInDataAnchorsPtr().empty()) {
      index_list.push_back(out_anchor->GetIdx());
    }
  }
  std::sort(index_list.begin(), index_list.end());
  // Node's out anchor index need shrink
  if (node->GetOpDesc()->impl_ == nullptr) {
    GELOGE(FAILED, "[Clear][OutputDesc] Op desc impl is nullptr. ");
    return false;
  }
  for (size_t i = 0UL; i < index_list.size(); ++i) {
    const auto iter = node->GetOpDesc()->impl_->outputs_desc_.begin() + static_cast<int64_t>(index_list[i]);
    if (iter < node->GetOpDesc()->impl_->outputs_desc_.end()) {
      (void)node->GetOpDesc()->impl_->outputs_desc_.erase(iter);
    } else {
      GELOGW("[Clear][OutputDesc] outputs_desc_ iterator out of range.");
    }
  }

  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearOutputDesc(const OpDescPtr &op_desc,
                                                                                 const uint32_t index) {
  return NodeUtils::ClearOutputDesc(op_desc, index);
}

bool OpDescUtils::HasQuantizeFactorParams(const OpDesc &op_desc) { return op_desc.HasAttr(OP_DESC_QUANT_PARAMS); }

GeTensorPtr OpDescUtils::MutableWeights(OpDesc &op_desc) {
  GeTensorPtr weight = nullptr;
  (void)AttrUtils::MutableTensor(&op_desc, ATTR_NAME_WEIGHTS, weight);
  return weight;
}

GE_FUNC_HOST_VISIBILITY GeTensorPtr OpDescUtils::MutableWeights(const OpDescPtr op_desc) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] op_desc is null");
    return nullptr;
  }
  return MutableWeights(*op_desc);
}

graphStatus OpDescUtils::SetWeights(OpDesc &op_desc, const GeTensorPtr weight) {
  if (weight == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "weight is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] weight is null");
    return GRAPH_FAILED;
  }
  return AttrUtils::SetTensor(&op_desc, ATTR_NAME_WEIGHTS, weight) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus OpDescUtils::SetWeights(OpDescPtr op_desc, const GeTensorPtr weight) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(weight);
  return SetWeights(*op_desc, weight);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<ConstGeTensorPtr> OpDescUtils::GetWeights(const ge::Node &node) {
  auto weights = MutableWeights(node);
  std::vector<ConstGeTensorPtr> ret(weights.size());
  (void)std::copy(weights.begin(), weights.end(), ret.begin());
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ConstGeTensorPtr> OpDescUtils::GetWeights(
    const ge::ConstNodePtr &node) {
  if (node == nullptr) {
    return std::vector<ge::ConstGeTensorPtr>();
  }
  return GetWeights(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ge::NodePtr> OpDescUtils::GetConstInputNode(
    const ge::Node &node) {
  std::vector<ge::NodePtr> ret;
  const auto in_anchors = node.GetAllInDataAnchorsPtr();
  for (const auto &in_anchor : in_anchors) {
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      // normally out_anchor could be null, this is ok
      GELOGD("node %s' peer_out_anchor is null", node.GetName().c_str());
      continue;
    }
    auto in_node = out_anchor->GetOwnerNode();
    while (true) {
      if (in_node == nullptr) {
        break;
      }
      if (ConstantUtils::IsConstant(in_node)) {
        ret.push_back(in_node);
        break;
      } else if (in_node->GetType() == DATA) {
        if (NodeUtils::IsWhileVaryingInput(in_node)) {
          break;
        }
        in_node = NodeUtils::GetParentInput(in_node);
      } else if ((in_node->GetType() == ENTER) || (in_node->GetType() == REFENTER)) {
        bool is_constant = false;
        (void)AttrUtils::GetBool(in_node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant);
        if (!is_constant) {
          break;
        }
        // Enter node has and only has one input
        if (in_node->GetInDataNodesSize() != 1U) {
          GELOGW("[Get][ConstInput] Check number of input_nodes for Enter node %s failed, input_node_num=%zu.",
                 in_node->GetName().c_str(), in_node->GetInDataNodes().size());
          break;
        }
        in_node = in_node->GetInDataNodes().at(0UL);
      } else {
        break;
      }
    }
  }
  return ret;
}

std::vector<NodeToOutAnchor> OpDescUtils::GetConstInputNodeAndAnchor(const ge::Node &node) {
  std::vector<std::pair<NodePtr, OutDataAnchorPtr>> ret;
  const auto in_nodes_and_anchors = node.GetInDataNodesAndAnchors();
  for (const auto &in_node_2_anchor : in_nodes_and_anchors) {
    auto in_node = in_node_2_anchor.first;
    auto in_node_2_out_anchor = in_node_2_anchor;
    while (true) {
      if (in_node == nullptr) {
        break;
      }
      if (ConstantUtils::IsConstant(in_node)) {
        ret.push_back(in_node_2_out_anchor);
        break;
      } else if (in_node->GetType() == DATA) {
        if (NodeUtils::IsWhileVaryingInput(in_node)) {
          break;
        }
        in_node_2_out_anchor = NodeUtils::GetParentInputAndAnchor(in_node);
        in_node = in_node_2_out_anchor.first;
      } else if ((in_node->GetType() == ENTER) || (in_node->GetType() == REFENTER)) {
        bool is_constant = false;
        (void)AttrUtils::GetBool(in_node->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, is_constant);
        if (!is_constant) {
          break;
        }
        // Enter node has and only has one input
        if (in_node->GetInDataNodesSize() != 1U) {
          GELOGW("[Get][ConstInput] Check number of input_nodes for Enter node %s failed, input_node_num=%zu.",
                 in_node->GetName().c_str(), in_node->GetInDataNodes().size());
          break;
        }
        if (in_node->GetInDataAnchor(0) == nullptr) {
          break;
        }
        auto peer_out_anchor = in_node->GetInDataAnchor(0)->GetPeerOutAnchor();
        if (peer_out_anchor == nullptr) {
          break;
        }
        in_node = peer_out_anchor->GetOwnerNode();
        in_node_2_out_anchor = std::make_pair(in_node, peer_out_anchor);
      } else {
        break;
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ConstGeTensorPtr> OpDescUtils::GetInputData(
    const std::vector<ge::NodePtr> &input_nodes) {
  std::vector<ConstGeTensorPtr> ret;

  for (const auto &input_node : input_nodes) {
    const auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<ConstGeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}

vector<ConstGeTensorPtr> OpDescUtils::GetWeightsFromNodes(
    const std::vector<NodeToOutAnchor> &input_nodes_2_out_anchors) {
  std::vector<ConstGeTensorPtr> ret;
  for (const auto &input_node_2_anchor : input_nodes_2_out_anchors) {
    const auto input_node = input_node_2_anchor.first;
    GeTensorPtr temp_weight ;
    (void)ConstantUtils::MutableWeight(input_node->GetOpDesc(),
                                       static_cast<uint32_t>(input_node_2_anchor.second->GetIdx()),
                                       temp_weight);
    if (temp_weight == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<ConstGeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}
size_t OpDescUtils::GetNonConstInputsSize(const ge::Node &node) {
  if (NodeUtils::IsAnchorStatusSet(node)) {
    size_t input_num = 0UL;
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        input_num++;
        continue;
      }
    }
    return input_num;  // lint !e712
  } else {
    GE_IF_BOOL_EXEC(
        node.GetInDataNodesSize() < GetConstInputs(node).size(),
        REPORT_INNER_ERR_MSG("E18888", "InDataNodes size:%zu is smaller than ConstInputs size:%zu",
                           node.GetInDataNodes().size(), GetConstInputs(node).size());
        GELOGE(GRAPH_FAILED, "[Check][Param] %zu is smaller than %zu",
               node.GetInDataNodes().size(), GetConstInputs(node).size());
        return 0UL);
    return node.GetInDataNodesSize() - GetConstInputs(node).size();
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDescUtils::GetNonConstInputsSize(const ge::ConstNodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return 0UL;
  }
  return GetNonConstInputsSize(*node);
}

GeTensorDesc OpDescUtils::GetNonConstInputTensorDesc(const ge::Node &node, const size_t index_non_const) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, REPORT_INNER_ERR_MSG("E18888", "node.GetOpDesc() is nullptr!");
                   return GeTensorDesc(), "[Check][Param] node.GetOpDesc() is nullptr!");
  size_t i = 0UL;
  if (NodeUtils::IsAnchorStatusSet(node)) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        if (index_non_const == i) {
          return node.GetOpDesc()->GetInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
        }
        ++i;
      }
    }
  } else {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      const auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto owner_node = peer_anchor->GetOwnerNodeBarePtr();
      if (owner_node == nullptr) {
        continue;
      }
      if (owner_node->GetType() == CONSTANT) {
        continue;
      }
      if (index_non_const == i) {
        return node.GetOpDesc()->GetInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
      }
      ++i;
    }
  }
  return GeTensorDesc();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc
OpDescUtils::GetNonConstInputTensorDesc(const ge::ConstNodePtr &node, const size_t index_non_const) {
  CHECK_FALSE_EXEC(node != nullptr, return GeTensorDesc());
  return GetNonConstInputTensorDesc(*node, index_non_const);
}

bool OpDescUtils::GetNonConstInputIndex(const ge::Node &node, const size_t index_non_const, size_t &index) {
  bool ret = false;
  size_t i = 0UL;
  if (NodeUtils::IsAnchorStatusSet(node)) {
    for (const auto &anchor : node.GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(anchor) == ANCHOR_DATA) {
        if (index_non_const == i) {
          index = static_cast<size_t>(anchor->GetIdx());
          ret = true;
        }
        ++i;
      }
    }
  } else {
    for (const auto &anchor : node.GetAllInDataAnchorsPtr()) {
      const auto peer_anchor = anchor->GetPeerOutAnchor();
      if (peer_anchor == nullptr) {
        continue;
      }
      const auto owner_node = peer_anchor->GetOwnerNodeBarePtr();
      if (owner_node == nullptr) {
        continue;
      }
      if (owner_node->GetType() == CONSTANT) {
        continue;
      }
      if (index_non_const == i) {
        index = static_cast<size_t>(anchor->GetIdx());
        ret = true;
      }
      ++i;
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::GetNonConstInputIndex(const ge::ConstNodePtr &node,
                                                                                       const size_t index_non_const,
                                                                                       size_t &index) {
  CHECK_FALSE_EXEC(node != nullptr, return false);
  return GetNonConstInputIndex(*node, index_non_const, index);
}

bool OpDescUtils::IsNonConstInput(const ge::Node &node, const size_t index) {
  bool ret = false;
  if (index < static_cast<size_t>(node.GetAllInDataAnchorsSize())) {
    if (NodeUtils::IsAnchorStatusSet(node)) {
      ret = (ge::AnchorUtils::GetStatus(node.GetInDataAnchor(static_cast<int32_t>(index))) ==
             ANCHOR_DATA); // lint !e712
    } else {
      for (const auto &anchor : node.GetAllInDataAnchorsPtr()) {
        if (anchor->GetIdx() != static_cast<int32_t>(index)) {
          continue;
        }
        const auto peer_anchor = anchor->GetPeerOutAnchor();
        if (peer_anchor == nullptr) {
          break;
        }
        const auto owner_node = peer_anchor->GetOwnerNodeBarePtr();
        if (owner_node == nullptr) {
          break;
        }
        ret = (owner_node->GetType() != CONSTANT);
      }
    }
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::IsNonConstInput(const ge::ConstNodePtr &node,
                                                                                 const size_t index) {
  CHECK_FALSE_EXEC(node != nullptr, return false);
  return IsNonConstInput(*node, index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ge::NodePtr> OpDescUtils::GetConstInputs(
    const ge::ConstNodePtr &node) {
  if (node == nullptr) {
    return std::vector<ge::NodePtr>();
  }
  return GetConstInputs(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<ge::GeTensorDesc> OpDescUtils::GetNonConstTensorDesc(
    const ge::ConstNodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    return std::vector<ge::GeTensorDesc>();
  }
  std::vector<ge::GeTensorDesc> ret;
  if (NodeUtils::IsAnchorStatusSet(*node)) {
    for (const auto &in_anchor : node->GetAllInDataAnchors()) {
      if (ge::AnchorUtils::GetStatus(in_anchor) == ANCHOR_DATA) {
        ret.push_back(node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(in_anchor->GetIdx())));
      }
    }
  } else {
    for (const auto &in_anchor : node->GetAllInDataAnchorsPtr()) {
      const auto out_anchor = in_anchor->GetPeerOutAnchor();
      if ((out_anchor == nullptr) || (out_anchor->GetOwnerNodeBarePtr()->GetOpDesc() == nullptr)) {
        continue;
      }
      if (out_anchor->GetOwnerNodeBarePtr()->GetOpDesc()->GetType() != CONSTANT) {
        ret.push_back(node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(in_anchor->GetIdx())));
      }
    }
  }
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<ge::NodePtr> OpDescUtils::GetConstInputs(const ge::Node &node, const uint32_t depth) {
  std::vector<ge::NodePtr> ret;
  if (depth == 0U) {
    return ret;
  }

  const auto in_anchors = node.GetAllInDataAnchorsPtr();
  for (const auto &in_anchor : in_anchors) {
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }

    const auto in_node = out_anchor->GetOwnerNode();
    if (in_node->GetType() == CONSTANT) {
      ret.push_back(in_node);
    } else if ((in_node->GetType() == SWITCH) && (node.GetType() == MATMUL)) {
      // const --> switch --> matmul
      auto switch_input = GetConstInputs(*in_node, depth - 1U);
      if (switch_input.size() > 0U) {
        (void)ret.insert(ret.end(), switch_input.begin(), switch_input.end());
      }
    } else if (in_node->GetType() == DATA) {
      const auto parent = NodeUtils::GetParentInput(in_node);
      if ((parent != nullptr) && (parent->GetType() == CONSTANT)) {
        ret.push_back(parent);
      }
    } else {
      // do nothing
    }
  }
  return ret;
}


graphStatus OpDescUtils::SetNoneConstNodeWeights(ge::Node &node, const std::vector<ge::GeTensorPtr> &weights) {
  const auto input_nodes = GetConstInputs(node);
  if (weights.size() < input_nodes.size()) {
    REPORT_INNER_ERR_MSG("E18888", "weights count:%zu can't be less than const input count:%zu, node:%s(%s)",
                         weights.size(), input_nodes.size(), node.GetName().c_str(), node.GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] weights count:%zu can't be less than const input count:%zu",
           weights.size(), input_nodes.size());
    return GRAPH_PARAM_INVALID;
  }

  ge::NamedAttrs named_attrs;
  (void)ge::AttrUtils::SetListTensor(named_attrs, "key", weights);
  std::vector<ge::GeTensorPtr> copy_weights;
  (void)ge::AttrUtils::MutableListTensor(named_attrs, "key", copy_weights);

  for (size_t i = 0UL; i < input_nodes.size(); ++i) {
    if (input_nodes[i]->GetOpDesc() != nullptr) {
      if (SetWeights(input_nodes[i]->GetOpDesc(), copy_weights[i]) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "set weights failed, node:%s(%s)", input_nodes[i]->GetName().c_str(),
                             input_nodes[i]->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Weights] failed, node:%s(%s)",
               input_nodes[i]->GetName().c_str(), input_nodes[i]->GetType().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  // If set more weights than constop, need to add constop
  for (size_t i = input_nodes.size(); i < copy_weights.size(); ++i) {
    // Use org weight before SetWeights Overwrite
    const auto const_opdesc = CreateConstOpZeroCopy(copy_weights[i]);
    GE_CHECK_NOTNULL(const_opdesc);

    const auto owner_graph = node.GetOwnerComputeGraph();
    if (owner_graph == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "node's graph is empty, node name: %s", node.GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][Graph] node's graph is empty, name: %s", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    const auto const_node = owner_graph->AddNodeFront(const_opdesc);
    GE_CHK_BOOL_EXEC(node.AddLinkFrom(const_node) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E18888", "node:%s add link failed.", node.GetName().c_str());
                     GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] graph add link failed! node:%s",
                            node.GetName().c_str());
                     return GRAPH_FAILED);
    const std::vector<ge::NodePtr> original_nodes;
    ge::GraphUtils::RecordOriginalNames(original_nodes, const_node);
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtils::SetNoneConstNodeWeights(ge::Node &node, const std::map<int, ge::GeTensorPtr> &weights_map) {
  for (const auto &pair:weights_map) {
    const auto idx = pair.first;
    // idx = in data anchor size is valid, it meant to add a new const node
    if ((idx < 0) || (static_cast<size_t>(idx) > node.GetAllInDataAnchorsSize())) {
      REPORT_INNER_ERR_MSG("E18888", "Invalid map key: %d of node[%s].", idx, node.GetName().c_str());
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Invalid map key: %d of node[%s].", idx, node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    const auto peer_node = NodeUtils::GetInDataNodeByIndex(node, idx);
    if (peer_node != nullptr) {
      // a. update const input node
      if (peer_node->GetType() != CONSTANT) {
        REPORT_INNER_ERR_MSG("E18888", "op %s [%d]'s input node should be const, but is %s type:%s ",
                             node.GetName().c_str(), pair.first, peer_node->GetName().c_str(),
                             peer_node->GetType().c_str());
        GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] op %s [%d]'s input node should be const, but is %s type:%s ",
               node.GetName().c_str(), pair.first, peer_node->GetName().c_str(), peer_node->GetType().c_str());
      }
      if (SetWeights(peer_node->GetOpDesc(), pair.second) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "set weights failed, node:%s(%s)", peer_node->GetName().c_str(),
                             peer_node->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Weights] failed, node:%s(%s)",
               peer_node->GetName().c_str(), peer_node->GetType().c_str());
        return GRAPH_FAILED;
      }
    } else {
      // b. create new const input node
      const auto const_opdesc = CreateConstOpZeroCopy(pair.second);
      GE_CHECK_NOTNULL(const_opdesc);
      const auto owner_graph = node.GetOwnerComputeGraph();
      if (owner_graph == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "node's graph is empty, node name: %s", node.GetName().c_str());
        GELOGE(GRAPH_PARAM_INVALID, "[Get][Graph] node's graph is empty, name: %s", node.GetName().c_str());
        return GRAPH_PARAM_INVALID;
      }
      const auto const_node = owner_graph->AddNodeFront(const_opdesc);
      if (node.AddLinkFrom(static_cast<uint32_t>(pair.first), const_node) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "op %s add const to input index[%d] failed", node.GetName().c_str(), pair.first);
        GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] op %s add const to input index[%d] failed",
               node.GetName().c_str(), pair.first);
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::Node &node) {
  std::vector<GeTensorPtr> ret;
  auto op_desc = node.GetOpDesc();
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node's op_desc is nullptr.");
                   return ret, "[Check][Param] op_desc is nullptr!");
  // Const operator, take the weight directly
  if ((op_desc->GetType() == CONSTANT) || (op_desc->GetType() == CONSTANTOP)) {
    const auto weight = MutableWeights(op_desc);
    if (weight == nullptr) {
      GELOGD("op type %s has no weight, op name:%s", node.GetType().c_str(), node.GetName().c_str());
      return ret;
    }
    ret.push_back(weight);
    return ret;
  }
  // Place holder operator, try to get the weight from parent node
  // when parent node is const operator
  if (node.GetType() == PLACEHOLDER) {
    ConstGeTensorPtr ge_tensor = nullptr;
    if (NodeUtils::TryGetWeightByPlaceHolderNode(std::const_pointer_cast<Node>(node.shared_from_this()), ge_tensor) ==
            GRAPH_SUCCESS &&
        ge_tensor != nullptr) {
      ret.push_back(std::const_pointer_cast<GeTensor>(ge_tensor));
    }
    return ret;
  }

  if (node.GetType() == DATA) {
    ConstGeTensorPtr ge_tensor = nullptr;
    if (NodeUtils::TryGetWeightByDataNode(std::const_pointer_cast<Node>(node.shared_from_this()), ge_tensor) ==
            GRAPH_SUCCESS &&
        ge_tensor != nullptr) {
      ret.push_back(std::const_pointer_cast<GeTensor>(ge_tensor));
    }
    return ret;
  }

  // Other operators, get weights from connected constop
  const auto input_nodes = GetConstInputs(node);
  for (const auto &input_node : input_nodes) {
    const auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<GeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::NodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return std::vector<ge::GeTensorPtr>();
  }
  return MutableWeights(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::Node &node, const std::vector<ge::GeTensorPtr> &weights) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, REPORT_INNER_ERR_MSG("E18888", "opdesc of node is nullptr.");
                   return GRAPH_PARAM_INVALID, "[Check][Param] node.GetOpDesc is nullptr!");
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights[0UL]);
    }
    GELOGI("const op weight size %zu should be 1", weights.size());
    return GRAPH_PARAM_INVALID;
  }

  return SetNoneConstNodeWeights(node, weights);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::Node &node, const std::map<int, ge::GeTensorPtr> &weights_map) {
  GE_CHECK_NOTNULL(node.GetOpDesc());
  // 1. node is const
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights_map.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights_map.begin()->second);
    }
    REPORT_INNER_ERR_MSG("E18888", "const op %s weight size %zu should be 1", node.GetName().c_str(),
                         weights_map.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] const op %s weight size %zu should be 1",
           node.GetName().c_str(), weights_map.size());
    return GRAPH_PARAM_INVALID;
  }
  // 2. node is not const
  auto const ret = SetNoneConstNodeWeights(node, weights_map);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  NodeUtils::UpdateIsInputConst(node);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc) {
  return GraphUtils::CloneOpDesc(org_op_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  return GraphUtils::CopyOpDesc(org_op_desc);
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr) {
  return CreateConstOp(tensor_ptr, true);
}

OpDescPtr OpDescUtils::CreateConstOpZeroCopy(const GeTensorPtr& tensor_ptr) {
  return CreateConstOp(tensor_ptr, false);
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr, const bool copy) {
  GE_ASSERT_NOTNULL(tensor_ptr);
  const shared_ptr<OpDesc> const_opdesc = ComGraphMakeShared<OpDesc>();
  GE_ASSERT_NOTNULL(const_opdesc, "[Create][OpDesc] failed.");
  if (copy) {
    GE_ASSERT_GRAPH_SUCCESS(SetWeights(const_opdesc, tensor_ptr), "[Set][Weights] failed, op[%s]",
                            const_opdesc->GetNamePtr());
  } else {
    GE_ASSERT_TRUE(AttrUtils::SetShareTensor(const_opdesc, ATTR_NAME_WEIGHTS, *tensor_ptr),
                   "[Set][ShardTensor] success for %s.", const_opdesc->GetNamePtr());
  }
  const_opdesc->SetType(CONSTANT);
  std::string op_name;
  GetConstantOpName(op_name);
  const_opdesc->SetName(op_name);
  GELOGI("add const op: %s", const_opdesc->GetNamePtr());
  (void)const_opdesc->AddOutputDesc("y", tensor_ptr->GetTensorDesc());
  GELOGI("after add const op: %s", const_opdesc->GetName().c_str());
  return const_opdesc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::AddConstOpToAnchor(const InDataAnchorPtr in_anchor, const GeTensorPtr &tensor_ptr) {
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(tensor_ptr);
  const auto const_opdesc = CreateConstOpZeroCopy(tensor_ptr);
  GE_CHECK_NOTNULL(const_opdesc);
  const auto in_node = in_anchor->GetOwnerNodeBarePtr();
  GE_CHECK_NOTNULL(in_node);
  const auto owner_graph = in_node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node's graph is empty, name: %s", in_node->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "[Get][Graph] node's graph is empty, name: %s", in_node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto const_node = in_node->GetOwnerComputeGraph()->AddNodeFront(const_opdesc);
  GE_CHECK_NOTNULL(const_node);
  if (GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), in_anchor) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "AddEdge const %s to node %s failed", const_node->GetName().c_str(),
                         in_node->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "[Add][Edge] const %s to node %s failed.", const_node->GetName().c_str(),
           in_node->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(ge::NodePtr node, const std::vector<ge::GeTensorPtr> &weights) {
  GE_CHECK_NOTNULL(node);
  return SetWeights(*node, weights);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDescUtils::ClearWeights(const ge::NodePtr node) {
  GE_CHECK_NOTNULL(node);
  const auto const_ops = GetConstInputs(node);
  const auto graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "GetOwnerComputeGraph failed, graph is nullptr, node:%s", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] Graph is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  for (const auto &const_op : const_ops) {
    GE_CHK_STATUS_RET(GraphUtils::IsolateNode(const_op, {}), "[Isolate][Node] %s, type:%s failed",
                      const_op->GetName().c_str(), const_op->GetType().c_str());
    GE_CHK_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph, const_op),
                      "[Remove][Node] %s, type: %s without relink failed", const_op->GetName().c_str(),
                      const_op->GetType().c_str());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus OpDescUtils::SetSubgraphInstanceName(const std::string &subgraph_name,
                                                 const std::string &subgraph_instance_name,
                                                 OpDescPtr &op_desc) {
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  const auto iter = subgraph_names_to_index.find(subgraph_name);
  if (iter == subgraph_names_to_index.end()) {
    REPORT_INNER_ERR_MSG(
        "E18888", "Failed to set subgraph instance %s for node %s type %s, the subgraph name %s does not exist",
        subgraph_instance_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), subgraph_name.c_str());
    GELOGE(GRAPH_PARAM_INVALID,
        "[Check][Param] Failed to set subgraph instance %s for node %s type %s, the subgraph name %s does not exist",
        subgraph_instance_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), subgraph_name.c_str());
    return GRAPH_PARAM_INVALID;
  }

  return op_desc->SetSubgraphInstanceName(iter->second, subgraph_instance_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ConstGeTensorBarePtr OpDescUtils::GetInputConstData(const Operator &op, const uint32_t idx) {
  if (op.operator_impl_ == nullptr) {
    AscendString op_name;
    (void)op.GetName(op_name);
    GELOGW("[Check][Param] Op(%s) operator_impl_ is nullptr.", op_name.GetString());
    return nullptr;
  }

  ConstGeTensorPtr ge_tensor = nullptr;
  if (op.operator_impl_->GetInputConstData(idx, ge_tensor) == GRAPH_SUCCESS) {
    return ge_tensor.get();
  }
  AscendString name;
  (void) op.GetName(name);
  AscendString type;
  (void) op.GetOpType(type);
  GELOGI("[Get][ConstInput] Op(%s %s) is unable to get const data with input index[%u] ",
         name.GetString(), type.GetString(), idx);
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
void OpDescUtils::SetRuntimeContextToOperator(const Operator &op, RuntimeInferenceContext *const context) {
  op.operator_impl_->runtime_context_ = context;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
void OpDescUtils::SetCallbackGetConstInputFuncToOperator(const Operator &op,
                                                         GetConstInputOnRuntimeFun get_const_input_func) {
  op.operator_impl_->get_const_input_runtime_ = get_const_input_func;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool OpDescUtils::HasCallbackGetConstInputFunc(const Operator &op) {
  return (op.operator_impl_->get_const_input_runtime_ != nullptr);
}

ge::graphStatus IrInputRequiredCall(const OpDescPtr &op_desc, size_t ir_index, size_t start_index, size_t all_ins_num,
                                    const std::string &ir_name,
                                    const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num);

ge::graphStatus IrInputRequiredCall(const OpDescPtr &op_desc, size_t ir_index, size_t start_index, size_t all_ins_num,
                                    const std::string &ir_name,
                                    const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num) {
  (void)all_ins_num;
  const auto max_index = valid_index_2_names.rbegin()->first;
  if (start_index > max_index) {
    GELOGW("Failed to get instance num for node %s, current name %s current index %zu out of range %u",
           op_desc->GetName().c_str(), ir_name.c_str(), start_index, max_index);
    instance_num = 1U;
    return ge::SUCCESS;
  }
  const auto name = valid_index_2_names.at(start_index);
  if (name != ir_name) {
    GELOGW("Failed to get instance num for node %s, can not find the input for ir name %s, current index %zu, "
           "current name %s",
           op_desc->GetName().c_str(), ir_name.c_str(), start_index, name.c_str());
    if (FindSubsequentMatches(valid_index_2_names, start_index + 1U, ir_name)) {
      GELOGE(ge::FAILED, "Find another input name that match ir name. ir_index:%zu, ir_name:%s, inputs names:%s",
             ir_index, ir_name.c_str(), InputsNamesStr(op_desc).c_str());
      return FAILED;
    }
  }
  instance_num = 1U;
  GELOGD("Get instance num %zu for node %s, current name %s current ir index %zu, start_index %zu", instance_num,
         op_desc->GetName().c_str(), ir_name.c_str(), ir_index, start_index);
  return ge::SUCCESS;
}

ge::graphStatus IrInputOptionalCall(const OpDescPtr &op_desc, size_t ir_index, size_t start_index, size_t all_ins_num,
                                    const std::string &ir_name,
                                    const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num);

ge::graphStatus IrInputOptionalCall(const OpDescPtr &op_desc, size_t ir_index, size_t start_index, size_t all_ins_num,
                                    const std::string &ir_name,
                                    const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num) {
  (void)all_ins_num;
  const auto max_index = valid_index_2_names.rbegin()->first;
  // ooooooxxx
  // o : required input
  // x : option input
  if (start_index > max_index) {
    instance_num = 0U;
    return ge::SUCCESS;
  }
  const auto name = valid_index_2_names.at(start_index);
  if (name == ir_name) {
    instance_num = 1U;
  } else {
    instance_num = 0U;
  }
  GELOGD("Get instance num %zu for node %s, current name %s current ir index %zu, start_index %zu", instance_num,
         op_desc->GetName().c_str(), ir_name.c_str(), ir_index, start_index);
  return ge::SUCCESS;
}

ge::graphStatus IrDynamicCall(const OpDescPtr &op_desc, size_t ir_index, size_t start_index, size_t all_ins_num,
                                   const std::string &ir_name,
                                   const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num);

ge::graphStatus IrDynamicCall(const OpDescPtr &op_desc, size_t ir_index, size_t start_index, size_t all_ins_num,
                                   const std::string &ir_name,
                                   const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num) {
  size_t dyn_i = 0;
  const auto max_index = valid_index_2_names.rbegin()->first;
  for (size_t i = start_index; i < all_ins_num; ++i, ++dyn_i) {
    if (i > max_index) {
      break;
    }
    const auto name = valid_index_2_names.at(i);
    if (name != ir_name + std::to_string(dyn_i)) {
      break;
    }
  }
  instance_num = dyn_i;
  GELOGD("Get instance num %zu for node %s, current name %s current ir index %zu, start_index %zu", instance_num,
         op_desc->GetName().c_str(), ir_name.c_str(), ir_index, start_index);
  return ge::SUCCESS;
}

ge::graphStatus GetOutputInstanceNum(const OpDescPtr &op_desc, size_t ir_index, size_t start_index,
                                     const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num);

ge::graphStatus GetOutputInstanceNum(const OpDescPtr &op_desc, size_t ir_index, size_t start_index,
                                     const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num) {
  GE_CHECK_NOTNULL(op_desc);
  if (valid_index_2_names.empty()) {
    GELOGD("Node %s has not any outputs, just return", op_desc->GetName().c_str());
    return ge::SUCCESS;
  }
  const auto &ir_outputs = op_desc->GetIrOutputs();
  const auto ir_type = ir_outputs[ir_index].second;
  const auto ir_name = ir_outputs[ir_index].first;
  using GetInstanceCall = std::function<Status(
      const OpDescPtr &op_desc, const size_t ir_index, const size_t start_index, const size_t all_ins_num,
      const std::string &ir_name, const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num)>;
  static std::map<IrOutputType, GetInstanceCall> get_instance_calls = {{kIrOutputRequired, &IrInputRequiredCall},
                                                                       {kIrOutputDynamic, &IrDynamicCall}};
  const auto it = get_instance_calls.find(ir_type);
  if (it != get_instance_calls.end()) {
    const size_t all_ins_num = op_desc->GetAllOutputsDescSize();
    return (it->second)(op_desc, ir_index, start_index, all_ins_num, ir_name, valid_index_2_names, instance_num);
  }
  GELOGE(ge::FAILED, "Failed to get instance num for node %s, unknown ir output type %d, ir name %s",
         op_desc->GetName().c_str(), ir_type, ir_name.c_str());
  return ge::FAILED;
}

ge::graphStatus OpDescUtils::GetInstanceNum(const OpDescPtr &op_desc, size_t ir_index, size_t start_index,
                                            const std::map<uint32_t, std::string> &valid_index_2_names,
                                            size_t &instance_num) {
  GE_CHECK_NOTNULL(op_desc);
  if (valid_index_2_names.empty()) {
    GELOGD("Node %s has not any inputs, just return", op_desc->GetName().c_str());
    return ge::SUCCESS;
  }
  const auto &ir_inputs = op_desc->GetIrInputs();
  const auto ir_type = ir_inputs[ir_index].second;
  const auto ir_name = ir_inputs[ir_index].first;
  using GetInstanceCall = std::function<Status(
      const OpDescPtr &op_desc, const size_t ir_index, const size_t start_index, const size_t all_ins_num,
      const std::string &ir_name, const std::map<uint32_t, std::string> &valid_index_2_names, size_t &instance_num)>;
  static std::map<IrInputType, GetInstanceCall> get_instance_calls = {{kIrInputRequired, &IrInputRequiredCall},
                                                                      {kIrInputOptional, &IrInputOptionalCall},
                                                                      {kIrInputDynamic, &IrDynamicCall}};
  const auto it = get_instance_calls.find(ir_type);
  if (it != get_instance_calls.end()) {
    const size_t all_ins_num = op_desc->GetAllInputsSize();
    return (it->second)(op_desc, ir_index, start_index, all_ins_num, ir_name, valid_index_2_names, instance_num);
  }
  GELOGE(ge::FAILED, "Failed to get instance num for node %s, unknown ir input type %d, ir name %s",
         op_desc->GetName().c_str(), ir_type, ir_name.c_str());
  return ge::FAILED;
}

std::map<size_t, std::pair<size_t, size_t>> OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(
    const OpDescPtr &op_desc) {
  std::map<size_t, std::pair<size_t, size_t>> ir_index_to_instance_index_pair_map;

  if (GetIrInputInstanceDescRange(op_desc, ir_index_to_instance_index_pair_map) == GRAPH_SUCCESS) {
    return ir_index_to_instance_index_pair_map;
  }

  return {};
}

std::map<size_t, std::pair<size_t, size_t>> OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(
    const OpDescPtr &op_desc) {
  std::map<size_t, std::pair<size_t, size_t>> ir_index_to_instance_index_pair_map;

  if (GetIrOutputDescRange(op_desc, ir_index_to_instance_index_pair_map) == GRAPH_SUCCESS) {
    return ir_index_to_instance_index_pair_map;
  }

  return {};
}

ge::graphStatus OpDescUtils::GetInputIrIndexByInstanceIndex(const OpDescPtr &op_desc,
                                                            size_t instance_index, size_t &ir_index) {
  GE_CHECK_NOTNULL(op_desc);
  auto ir_index_to_instance_index_pair_map = GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  if (ir_index_to_instance_index_pair_map.empty()) {
    GELOGE(ge::GRAPH_FAILED,
           "node [%s(%s)] get ir indexes to instance indexes list failed, instance_index[%zu], which is empty",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), instance_index);
    return ge::GRAPH_FAILED;
  }
  ir_index = std::numeric_limits<size_t>::max();
  for (size_t i = 0U; i < op_desc->GetIrInputs().size(); ++i) {
    const auto &index_pair = ir_index_to_instance_index_pair_map[i];
    size_t ir_index_end = 0U;
    GE_ASSERT_TRUE(!ge::AddOverflow(index_pair.first, index_pair.second, ir_index_end));
    if ((instance_index >= index_pair.first) && (instance_index < ir_index_end)) {
      ir_index = i;
      GELOGD("node [%s(%s)] get ir index [%zu] successfully!", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             ir_index);
      return ge::GRAPH_SUCCESS;
    }
  }
  ir_index = std::numeric_limits<size_t>::max();
  GELOGW("node [%s(%s)] failed to get ir index by instance index[%zu], set ir_index to %zu", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), instance_index, ir_index);
  return GRAPH_SUCCESS;
}

ge::graphStatus OpDescUtils::GetOutputIrIndexByInstanceIndex(const OpDescPtr &op_desc,
                                                             size_t instance_index, size_t &ir_index) {
  GE_CHECK_NOTNULL(op_desc);
  ir_index = std::numeric_limits<size_t>::max();

  auto ir_index_to_instance_index_pair_map = GetOutputIrIndexes2InstanceIndexesPairMap(op_desc);
  if (ir_index_to_instance_index_pair_map.empty()) {
    return ge::GRAPH_SUCCESS;
  }

  for (size_t i = 0U; i < op_desc->GetIrOutputs().size(); ++i) {
    const auto &index_pair = ir_index_to_instance_index_pair_map[i];
    size_t ir_index_end = 0U;
    GE_ASSERT_TRUE(!ge::AddOverflow(index_pair.first, index_pair.second, ir_index_end));
    if ((instance_index >= index_pair.first) && (instance_index < ir_index_end)) {
      ir_index = i;
      GELOGD("node [%s(%s)] get ir index [%zu] successfully!", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             ir_index);
      return ge::GRAPH_SUCCESS;
    }
  }

  GELOGW("node [%s(%s)] failed to get ir index by instance index[%zu], set ir_index to %zu", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), instance_index, ir_index);
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtils::GetIrInputInstanceDescRange(const OpDescPtr &op,
                                                     std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  return ge::GetIrInputInstanceDescRange(op, ir_input_2_range);
}

graphStatus OpDescUtils::GetIrInputRawDescRange(const OpDescPtr &op,
                                                std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  return ge::GetIrInputRawDescRange(op, ir_input_2_range);
}

graphStatus OpDescUtils::GetIrOutputDescRange(const OpDescPtr &op,
                                              std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range) {
  return ge::GetIrOutputDescRange(op, ir_output_2_range);
}

graphStatus OpDescUtils::GetIrInputDtypeSymIds(const OpDescPtr &op_desc, std::vector<std::string> &dtype_sym_ids) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(op_desc->impl_);
  const auto &sym_store = op_desc->impl_->GetIRMeta().GetIRDataTypeSymbolStore();

  dtype_sym_ids.clear();
  dtype_sym_ids.resize(op_desc->GetIrInputs().size());
  for (const auto &sym_holder : sym_store.GetSymbols()) {
    GE_CHECK_NOTNULL(sym_holder);
    const auto ir_input_indexes = sym_holder->GetDirectIrInputIndexes();
    if (ir_input_indexes.empty()) {
      continue;
    }
    for (const auto idx : ir_input_indexes) {
      dtype_sym_ids[idx] = sym_holder->Id();
    }
  }
  return ge::GRAPH_SUCCESS;
}

graphStatus OpDescUtils::GetIrOutputDtypeSymIds(const OpDescPtr &op_desc, std::vector<std::string> &dtype_sym_ids) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(op_desc->impl_);
  const auto &sym_store = op_desc->impl_->GetIRMeta().GetIRDataTypeSymbolStore();

  dtype_sym_ids.clear();
  dtype_sym_ids.resize(op_desc->GetIrOutputs().size());
  const auto &out_syms = sym_store.GetOutSymbols();
  GE_ASSERT_EQ(out_syms.size(), dtype_sym_ids.size());
  for (size_t i = 0U; i < out_syms.size(); ++i) {
    dtype_sym_ids[i] = out_syms[i]->Id();
  }

  return ge::GRAPH_SUCCESS;
}

graphStatus OpDescUtils::GetPromoteIrInputList(const OpDescPtr &op_desc,
                                               std::vector<std::vector<size_t>> &promote_index_list) {
  GE_ASSERT_NOTNULL(op_desc);
  const ge::Operator operator_ir = ge::OperatorFactory::CreateOperator("temp_operator", op_desc->GetType().c_str());
  const auto opdesc_ir = ge::OpDescUtils::GetOpDescFromOperator(operator_ir);
  GE_ASSERT_NOTNULL(opdesc_ir);
  return opdesc_ir->GetPromoteIrInputList(promote_index_list);
}

graphStatus OpDescUtils::GetPromoteInstanceInputList(const OpDescPtr &op_desc,
                                                     std::vector<std::vector<size_t>> &promote_index_list) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_SUCCESS(ge::RecoverIrUtils::RecoverOpDescIrDefinition(op_desc, op_desc->GetTypePtr()));
  auto ir_ranges = GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  std::vector<std::vector<size_t>> ir_promote_index_list;
  GE_ASSERT_SUCCESS(op_desc->GetPromoteIrInputList(ir_promote_index_list));
  for (const auto& ir_input_indexes : ir_promote_index_list) {
    std::vector<size_t> instance_input_indexes;
    for (const auto& ir_input_index : ir_input_indexes) {
      auto ir_range = ir_ranges.find(ir_input_index);
      if (ir_range == ir_ranges.end()) {
        continue;
      }
      for (size_t i = ir_range->second.first; i < ir_range->second.second + ir_range->second.first; i++) {
        instance_input_indexes.push_back(i);
      }
    }
    promote_index_list.push_back(instance_input_indexes);
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge
/*lint +e512 +e737 +e752*/
