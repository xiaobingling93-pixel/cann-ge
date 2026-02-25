/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/user_semantic_inference.h"
#include <map>
#include <string>
#include <vector>
#include "common/fe_utils.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/op_info_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "common/math_util.h"
#include "common/fe_context_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/ge_context.h"

 /*
      Data      NZ
        |
      Bitcast   NZ
        |
      GMM       ND
 */

namespace fe {

  static const string USERSEMANTICINFERENCE_PASS_NAME = "UserSemanticInferencePass";
  static const char *BITCAST = "Bitcast";
  static const char *Data = "Data";
  static const std::string PATTERN_BITCAST = "Bitcast";
  static const std::string PATTERN_DATA = "Data";
  static const std::string FUSED_OP_TYPE = "GroupedMatmul";
  static const std::string PATTERN_PYPTO = "pypto";

  static const std::set<ge::Format> dataOutputFormatSet = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_8,
                                                           ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32};

vector<FusionPattern *> UserSemanticInferencePass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("UserSemanticInferencePass");
  FE_CHECK(pattern == nullptr,
           REPORT_FE_ERROR("[GraphOpt][NdOpti][DefPtn] Failed to create a new object."), return patterns);
  pattern->AddOpDesc(PATTERN_BITCAST, {BITCAST})
      .AddOpDesc(PATTERN_DATA, {Data})
      .SetInputs(PATTERN_BITCAST, {PATTERN_DATA})
      .SetOutput(PATTERN_BITCAST);

  patterns.push_back(pattern);

  FusionPattern *pattern2 = new (std::nothrow) FusionPattern("UserSemanticInferencePass2");
  FE_CHECK(pattern2 == nullptr,
           REPORT_FE_ERROR("[GraphOpt][NdOpti][DefPtn] Failed to create a new object."), return patterns);
  pattern2->AddOpDesc(PATTERN_PYPTO, {"LightningIndexerPrologPto", "DeekseekIndexerAttentionPto"})
      .SetOutput(PATTERN_PYPTO);

  patterns.push_back(pattern2);
  FE_LOGD("Define UserSemanticInferencePass pattern end.");
  return patterns;
}

Status UserSemanticInferencePass::FusionForPyPTO(const ge::NodePtr &pyptoNode) const {
  ge::OpDescPtr opDesc = pyptoNode->GetOpDesc();
  FE_CHECK_NOTNULL(opDesc);
  for (uint32_t i = 0; i < pyptoNode->GetAllInDataAnchorsSize(); ++i) {
    auto inAnchor = pyptoNode->GetInDataAnchor(i);
    auto peerOutAnchor = inAnchor->GetPeerOutAnchor();
    if (peerOutAnchor == nullptr) {
      FE_LOGD("Node[%s]: input %lu peerOutAnchor is nullptr.", pyptoNode->GetNamePtr(), i);
      continue;
    }
    auto peerNode = peerOutAnchor->GetOwnerNodeBarePtr();
    if (peerNode == nullptr) {
      FE_LOGD("Node[%s]: input %lu peerNode is nullptr.", pyptoNode->GetNamePtr(), i);
      continue;
    }
    ge::OpDescPtr peerNodeDesc = peerNode->GetOpDesc();
    if (peerNodeDesc == nullptr) {
      FE_LOGD("Node[%s]: input %lu peerNode is nullptr.", pyptoNode->GetNamePtr(), i);
      continue;
    }
    auto peerNodeOutputDesc = peerNodeDesc->GetOutputDescPtr(0);
    if (peerNodeOutputDesc == nullptr) {
      FE_LOGD("Node[%s]: input %lu peerNodeOutputDesc is nullptr.", pyptoNode->GetNamePtr(), i);
      continue;
    }
    if (ge::GetPrimaryFormat(peerNodeOutputDesc->GetFormat()) != ge::FORMAT_FRACTAL_NZ) {
      continue;
    }
    FE_LOGD("Node[%s]: input %lu peerNode[%s] format is nz.", pyptoNode->GetNamePtr(), i, peerNode->GetNamePtr());
    auto inputTensorDesc = opDesc->MutableInputDesc(i);
    if (inputTensorDesc == nullptr) {
      FE_LOGD("Node[%s]: input %lu inputTensorDesc is nullptr.", pyptoNode->GetNamePtr(), i);
      continue;
    }
    inputTensorDesc->SetFormat(ge::FORMAT_FRACTAL_NZ);
    auto peerNodeOutputShape = peerNodeOutputDesc->GetShape();
    inputTensorDesc->SetShape(peerNodeOutputShape);
  }
  FE_LOGD("Node[%s]: end to do UserSemanticInferencePass.", pyptoNode->GetNamePtr());
  return SUCCESS;
}

Status UserSemanticInferencePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr> &fusionNodes) {
  ge::NodePtr pyptoNode = GetNodeFromMapping(PATTERN_PYPTO, mapping);
  if (pyptoNode != nullptr) {
    return FusionForPyPTO(pyptoNode);
  }
  ge::NodePtr dataNode = GetNodeFromMapping(PATTERN_DATA, mapping);
  ge::ConstOpDescPtr dataOpDesc = dataNode->GetOpDesc();
  ge::NodePtr bitcastNode = GetNodeFromMapping(PATTERN_BITCAST, mapping);
  auto bitcastOutNodes = bitcastNode->GetOutDataNodes();
  if (bitcastOutNodes.size() == 1) {
    ge::GeTensorDescPtr bitcastInputTensor = bitcastNode->GetOpDesc()->MutableInputDesc(0);
    ge::GeTensorDescPtr dataOutputTensor = dataOpDesc->MutableOutputDesc(0);
    ge::Format dataOutputFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(bitcastInputTensor->GetFormat()));
    if (dataOutputFormatSet.find(dataOutputFormat) == dataOutputFormatSet.end() ||
        !ge::AttrUtils::HasAttr(dataOpDesc, "_enable_storage_format_spread")) {
        FE_LOGD("BitcastInput format is not NZ or data node do not with _enable_storage_format_spread attribute.");
        return NOT_CHANGED;
    }
    FE_LOGD("BitcastInput format is NZ.");
    auto bitcastOutNode = bitcastOutNodes.at(0);
    auto bitcastOutAnchor = bitcastNode->GetOutDataAnchor(0);
    FE_CHECK_NOTNULL(bitcastOutAnchor);
    for (auto inDataAnchor : bitcastOutAnchor->GetPeerInDataAnchors()) {
      auto inPeerOutAnchor = inDataAnchor->GetPeerOutAnchor();
      FE_CHECK_NOTNULL(inPeerOutAnchor);
      if (bitcastNode->GetName() == inPeerOutAnchor->GetOwnerNode()->GetName()) {
        auto idx = inDataAnchor->GetIdx();
        auto inTensor = bitcastOutNode->GetOpDesc()->MutableInputDesc(idx);
        auto bitcastOutIdx = bitcastOutAnchor->GetIdx();
        auto bitcastOutTensor = bitcastNode->GetOpDesc()->MutableOutputDesc(bitcastOutIdx);
        auto newFormat = bitcastOutTensor->GetFormat();
        auto newShape = bitcastOutTensor->GetShape();
        auto inTensorFormat = inTensor->GetFormat();
        auto inTensorShape = inTensor->GetShape();
        inTensor->SetFormat(newFormat);
        inTensor->SetShape(newShape);
        if (CheckOpSupported(bitcastOutNode)) {
          FE_LOGD("UserSemanticInferencePass effected.");
          return SUCCESS;
        }
        inTensor->SetFormat(inTensorFormat);
        inTensor->SetShape(inTensorShape);
        FE_LOGD("Op[%s] checksupport unsuccess.", bitcastOutNode->GetName().c_str());
      }
    }
  }
  FE_LOGD("Bitcast out node size is [%zu] more than one.", bitcastOutNodes.size());
  return NOT_CHANGED;
}

REG_PASS(USERSEMANTICINFERENCE_PASS_NAME, BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
         UserSemanticInferencePass, SINGLE_SCENE_OPEN | FE_PASS);
}