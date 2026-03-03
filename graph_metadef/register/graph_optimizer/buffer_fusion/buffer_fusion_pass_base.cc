/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include <map>
#include <string>
#include <vector>
#include "register/graph_optimizer/fusion_common/fusion_turbo.h"

namespace fe {
namespace {
  const std::string kAttrNameIsOpDynamicImpl = "_is_op_dynamic_impl";
  constexpr uint32_t kNoNeedCompareSize = 2;
}
BufferFusionPassBase::BufferFusionPassBase() {}

BufferFusionPassBase::~BufferFusionPassBase() {}

Status BufferFusionPassBase::GetFusionNodes(const BufferFusionMapping &mapping,
                                            std::vector<ge::NodePtr> &fusion_nodes) {
  fusion_nodes = GetMatchedNodes(mapping);
  return SUCCESS;
}

Status BufferFusionPassBase::GetMixl2FusionNodes(const BufferFusionMapping &mapping,
                                                 std::vector<ge::NodePtr> &fusion_nodes) {
  return NOT_CHANGED;
}

Status BufferFusionPassBase::PostFusion(const ge::NodePtr &fused_node) {
  return SUCCESS;
}

Status BufferFusionPassBase::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info) {
  return SUCCESS;
}

Status BufferFusionPassBase::CheckNodeCanFusion(const BufferFusionNodeDescMap &fusion_nodes,
                                                const ge::NodePtr &next_node) {
  return SUCCESS;
}

std::vector<ge::NodePtr> BufferFusionPassBase::GetMatchedNodes(const BufferFusionMapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (const auto &item : mapping) {
    for (const auto &node : item.second) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

bool BufferFusionPassBase::CheckNodeIsDynamicImpl(const ge::NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  bool is_dynamic_impl = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kAttrNameIsOpDynamicImpl, is_dynamic_impl);
  return is_dynamic_impl;
}

bool BufferFusionPassBase::CheckTwoNodesImplConsistent(const ge::NodePtr &src_node, const ge::NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    return false;
  }
  bool src_dynamic_impl = false;
  bool dst_dynamic_impl = false;
  (void)ge::AttrUtils::GetBool(src_node->GetOpDesc(), kAttrNameIsOpDynamicImpl, src_dynamic_impl);
  (void)ge::AttrUtils::GetBool(dst_node->GetOpDesc(), kAttrNameIsOpDynamicImpl, dst_dynamic_impl);
  return src_dynamic_impl == dst_dynamic_impl;
}

bool BufferFusionPassBase::CheckNodesImplConsistent(const BufferFusionMapping &mapping) {
  const std::vector<ge::NodePtr> fusion_nodes = GetMatchedNodes(mapping);
  return CheckNodesImplConsistent(fusion_nodes);
}

bool BufferFusionPassBase::CheckNodesImplConsistent(const std::vector<ge::NodePtr> &fusion_nodes) {
  if (fusion_nodes.size() < kNoNeedCompareSize) {
    return true;
  }
  const ge::NodePtr first_node = fusion_nodes[0];
  for (size_t index = 1; index < fusion_nodes.size(); ++index) {
    if (!CheckTwoNodesImplConsistent(first_node, fusion_nodes[index])) {
      return false;
    }
  }
  return true;
}

bool BufferFusionPassBase::CheckNodeIsDynamicShape(const ge::NodePtr& node) {
  const ge::OpDescPtr op_desc = node->GetOpDesc();
  for (size_t index = 0; index < op_desc->GetAllInputsSize(); ++index) {
    if (FusionTurbo::IsUnknownShape(node, static_cast<int32_t>(index), true)) {
      return true;
    }
  }

  for (size_t index = 0; index < op_desc->GetAllOutputsDescSize(); ++index) {
    if (FusionTurbo::IsUnknownShape(node, static_cast<int32_t>(index), false)) {
      return true;
    }
  }
  return false;
}

bool BufferFusionPassBase::CheckNodesIncDynamicShape(const BufferFusionMapping &mapping) {
  const std::vector<ge::NodePtr> fusion_nodes = GetMatchedNodes(mapping);
  return CheckNodesIncDynamicShape(fusion_nodes);
}

bool BufferFusionPassBase::CheckNodesIncDynamicShape(const std::vector<ge::NodePtr> &fusion_nodes) {
  for (const auto &node : fusion_nodes) {
    if (CheckNodeIsDynamicShape(node)) {
      return true;
    }
  }
  return false;
}

std::vector<ge::NodePtr> BufferFusionPassBase::GetMatchedNodesByDescName(const std::string &desc_name,
                                                                         const BufferFusionMapping &mapping) {
  std::vector<ge::NodePtr> nodes;
  for (const auto &item : mapping) {
    const BufferFusionOpDesc *const op_desc = item.first;
    if ((op_desc != nullptr) && (op_desc->desc_name == desc_name)) {
      for (const auto &node : item.second) {
        nodes.push_back(node);
      }
    }
  }
  return nodes;
}

ge::NodePtr BufferFusionPassBase::GetMatchedHeadNode(const std::vector<ge::NodePtr> &matched_nodes) {
  for (const auto &node : matched_nodes) {
    const auto input_nodes = node->GetInDataNodes();
    bool find_flag = false;
    for (const auto &in_node : input_nodes) {
      // find the node from fuison sub graph
      if (std::find(matched_nodes.begin(), matched_nodes.end(), in_node) != matched_nodes.end()) {
        find_flag = true;
        break;
      }
    }
    if (!find_flag) {
      return node;
    }
  }
  return nullptr;
}

}  // namespace fe
