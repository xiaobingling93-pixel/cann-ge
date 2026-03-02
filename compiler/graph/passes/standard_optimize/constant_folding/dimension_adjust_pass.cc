/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.h"

#include <memory>
#include <string>
#include <vector>
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"
#include "folding_pass.h"

namespace ge {
namespace {
const int32_t kDataInputIndex = 0;
const int32_t kRemoveInputIndex = 1;
}  // namespace

Status DimensionAdjustPass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);

  std::string type;
  Status ret = GetOriginalType(node, type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OriginalType] of op:%s(%s) failed", node->GetName().c_str(), node->GetType().c_str());
    return ret;
  }

  if (folding_pass::IsUserSpecifiedSkipConstantFold(node)) {
    GELOGI("Attr[%s] of the node[%s] is set to true, will not be folded.",
           ATTR_NAME_DO_NOT_CONSTANT_FOLDING.c_str(), node->GetNamePtr());
    return SUCCESS;
  }

  KernelFactory &factory = KernelFactory::Instance();
  shared_ptr<Kernel> op_kernel = factory.Create(type);
  if (op_kernel == nullptr) {
    return SUCCESS;
  }
  bool is_unknown = false;
  auto ret_status = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
  if (ret_status != GRAPH_SUCCESS) {
    GELOGW("Get node unknown status failed, node name:%s, type:%s.", node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (is_unknown) {
    GELOGI("Current node %s, type %s is unknown shape which should be skip.",
           node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  // call compute function
  ret = op_kernel->Compute(node);
  if (ret != SUCCESS) {
    if (ret == NOT_CHANGED || ret == UNSUPPORTED) {
      return SUCCESS;
    }
    REPORT_INNER_ERR_MSG("E19999", "kernel compute for op:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(ret, "[Call][Compute] for op:%s(%s) failed", node->GetName().c_str(), node->GetType().c_str());
    return ret;
  }
  // Need to handle axis_input of node like ExpandDims
  if (node->GetInDataNodes().size() > static_cast<size_t>(kRemoveInputIndex)) {
    auto axis_node = node->GetInDataNodes().at(kRemoveInputIndex);
    GE_CHK_STATUS_RET(DeleteUselessConstAxisNode(axis_node),
                      "Remove const axis input of node:%s failed", node->GetName().c_str());
  }

  std::vector<int32_t> data_relink_io_map = {kDataInputIndex};
  return IsolateAndDeleteNode(node, data_relink_io_map);
}

NodePtr DimensionAdjustPass::AddIdentityNodeToGraph(const std::string &name, const GeTensorDesc &tensor,
                                                    ComputeGraphPtr &graph) const {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param graph is nullptr, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Comput graph ptr is nullptr in creating identity node.");
    return nullptr;
  }

  OpDescPtr desc = MakeShared<OpDesc>("", "");
  if (desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(MEMALLOC_FAILED, "[New][OpDesc] failed.");
    return nullptr;
  }

  desc->SetName(name);
  ge::OpDescUtilsEx::SetType(desc, IDENTITY);
  auto ret = desc->AddInputDesc(tensor);
  auto ret2 = desc->AddOutputDesc(tensor);
  if ((ret != GRAPH_SUCCESS) || (ret2 != GRAPH_SUCCESS)) {
    REPORT_INNER_ERR_MSG("E19999", "Add input or ouput desc to op:%s(%s) failed",
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][GeTensorDesc] to op:%s(%s) failed",
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  return graph->AddNodeFront(desc);
}

REG_PASS_OPTION("DimensionAdjustPass").LEVELS(OoLevel::kO1).SWITCH_OPT(ge::OO_CONSTANT_FOLDING);
}  // namespace ge
