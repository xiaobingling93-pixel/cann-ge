/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_add_api_call.h"

namespace codegen {
Status MicroAddApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                     CallParam &param, string &result) {
  GE_ASSERT_TRUE(this->inputs_.size() == 2, "Add api call must have 2 inputs");
  GE_ASSERT_TRUE(this->outputs_.size() == 1, "Add api call must have 1 output");
  for (auto input : this->inputs_) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(input.second));
  }
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(this->outputs_[0].second));
  std::stringstream ss;
  ss << "AscendC::MicroAPI::" << this->api_name_ << (this->second_input_scalar_ ? "s" : "") << "(";
  ss << *(tensor_mng.GetTensor(this->outputs_[0].second)) << ", ";
  ss << *(tensor_mng.GetTensor(this->inputs_[0].second)) << ", ";
  if (inputs_[1].first == TensorType::REG_TENSOR) {
    ss << *tensor_mng.GetTensor(inputs_[1].second) << ", ";
  } else {
    ss << *tpipe.GetTensor(inputs_[1].second) << ", ";
  }
  ss << param.p_reg << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status MicroAddApiCall::Init(const ascir::NodeView &node) {
  // 判断第二个输入是否是scalar
  if (node->GetInDataNodes().at(1)->GetType() == "Scalar") {
    this->second_input_scalar_ = true;
  }
  GELOGI("name:%s, second input scalar:%d", node->GetNamePtr(), this->second_input_scalar_);
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroAddApiCall> register_micro_add_api_call("MicroAddApiCall");
}  // namespace codegen
