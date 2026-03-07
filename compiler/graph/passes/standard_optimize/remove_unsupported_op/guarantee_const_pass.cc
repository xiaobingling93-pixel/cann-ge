/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_unsupported_op/guarantee_const_pass.h"

#include <string>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "common/omg_util/omg_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
const uint32_t kGuaranteeConstInputsSize = 1;
}
Status GuaranteeConstPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    GELOGE(status_ret, "[Get][OriginalType] for node:%s failed", node->GetName().c_str());
    return status_ret;
  }
  if (type != GUARANTEECONST) {
    return SUCCESS;
  }
  if (node->GetOpDesc()->GetAllInputsSize() != kGuaranteeConstInputsSize) {
    REPORT_INNER_ERR_MSG("E19999", "Num:%zu of input desc in node:%s(%s) not equal to %u, "
                      "check invalid", node->GetOpDesc()->GetAllInputsSize(),
                      node->GetName().c_str(), node->GetType().c_str(), kGuaranteeConstInputsSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Num:%zu of input desc in node:%s(%s) not equal to %u",
           node->GetOpDesc()->GetAllInputsSize(),
           node->GetName().c_str(), node->GetType().c_str(), kGuaranteeConstInputsSize);
    return PARAM_INVALID;
  }
  // [Cascade pointer]
  const auto &in_desc = node->GetOpDesc()->MutableInputDesc(0);
  GE_CHECK_NOTNULL(in_desc);
  // Input tensor cannot be a resource variable handle.
  const DataType &input_dtype = in_desc->GetDataType();
  if (input_dtype == DT_RESOURCE) {
    REPORT_INNER_ERR_MSG("E19999", "Data type:%s of op:%s(%s) input0 tensor not equal to %s, check invalid",
                      TypeUtils::DataTypeToSerialString(input_dtype).c_str(),
                      node->GetName().c_str(), node->GetType().c_str(),
                      TypeUtils::DataTypeToSerialString(DT_RESOURCE).c_str());
    GELOGE(FAILED, "[Check][Param] Data type:%s of op:%s(%s) input0 tensor not equal to %s",
           TypeUtils::DataTypeToSerialString(input_dtype).c_str(),
           node->GetName().c_str(), node->GetType().c_str(), TypeUtils::DataTypeToSerialString(DT_RESOURCE).c_str());
    return FAILED;
  }

  return IsolateAndDeleteNode(node, {0});
}

REG_PASS_OPTION("GuaranteeConstPass").LEVELS(OoLevel::kO0);
}  // namespace ge
