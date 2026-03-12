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

#include "micro_where_api_call.h"

namespace codegen {
Status MicroWhereApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                   CallParam &param, string &result) {
  (void)param;
  std::stringstream ss;
  ss << "AscendC::MicroAPI::" << this->api_name_ << "(";
  for (auto out_arg : this->outputs_) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(out_arg.second));
    ss << *(tensor_mng.GetTensor(out_arg.second)) << ", ";
  }
  for (auto in_arg : this->inputs_) {
    GE_ASSERT_NOTNULL(tensor_mng.GetTensor(in_arg.second));
    if (tensor_mng.GetTensor(in_arg.second)->init_as_mask_reg_ == false) {
      ss << *(tensor_mng.GetTensor(in_arg.second)) << ", ";
    }
  }
  bool first = true;
  for (auto in_arg : this->inputs_) {
    if (tensor_mng.GetTensor(in_arg.second)->init_as_mask_reg_ == true) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      ss << *(tensor_mng.GetTensor(in_arg.second));
    }
  }
  ss << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroWhereApiCall> register_micro_where_api_call("MicroWhereApiCall");
}  // namespace codegen
