/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_scalar_broadcast_api_call.h"

#include "micro_api_call_factory.h"
#include "ascir_ops.h"

namespace codegen {
Status MicroScalarBroadcastApiCall::Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param,
                                             string &result) {
  std::stringstream ss;
  auto out_tensor_id = GetOutputTensorIdByIndex(0);
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(out_tensor_id));
  auto dts = tensor_mng.GetTensor(out_tensor_id);
  auto in_tensor_id = GetInputTensorIdByIndex(0);
  auto in_tensor_type = inputs_[0].first;
  ss << "AscendC::MicroAPI::Duplicate" << "(" << *dts << ", ";
  if (in_tensor_type == TensorType::REG_TENSOR) {
    ss << *tensor_mng.GetTensor(in_tensor_id);
  } else {
    ss << *tpipe.GetTensor(in_tensor_id);
  }
  ss << ", " << param.p_reg << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroScalarBroadcastApiCall> register_micro_scalar_broadcast_api_call(
    "MicroScalarBroadcastApiCall");
}  // namespace codegen