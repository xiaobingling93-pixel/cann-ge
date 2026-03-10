/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_leaky_relu_api_call.h"

namespace codegen {
Status MicroLeakyReluApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe, CallParam &param,
                                       string &result) {
  std::stringstream ss;

  std::string x_dtype;
  ss << "AscendC::MicroAPI::" << this->api_name_ << "(";
  for (auto out_arg : this->outputs_) {
    ss << *(tensor_mng.GetTensor(out_arg.second)) << ", ";
  }
  for (auto in_arg : this->inputs_) {
    GE_CHK_STATUS_RET(Tensor::DtypeName(tensor_mng.GetTensor(in_arg.second)->dtype_, x_dtype), "get name of dtype:%d failed", static_cast<int32_t>(tensor_mng.GetTensor(in_arg.second)->dtype_));
    ss << *(tensor_mng.GetTensor(in_arg.second)) << ", ";
  }
  ss << "(" << x_dtype << ")" << std::to_string(this->negative_slope_) << ", ";
  ss << param.p_reg << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status MicroLeakyReluApiCall::Init(const ascir::NodeView &node) {
  // 从节点属性中获取negative_slope参数
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("negative_slope", this->negative_slope_),
                          "Failed to get LeakyRelu negative_slope attr, node = %s", node->GetNamePtr());
  GELOGI("name:%s, negative_slope:%f", node->GetNamePtr(), this->negative_slope_);
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroLeakyReluApiCall> register_micro_leaky_relu_api_call("MicroLeakyReluApiCall");
}  // namespace codegen
