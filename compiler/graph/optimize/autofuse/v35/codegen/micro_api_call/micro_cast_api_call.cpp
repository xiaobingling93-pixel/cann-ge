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

#include "micro_cast_api_call.h"

using namespace ascgen_utils;

namespace {
  const std::map<std::pair<ge::DataType, ge::DataType>, std::string> trait_map = {
    {{ge::DT_FLOAT,   ge::DT_FLOAT16}, "cast_trait_float_2_half"},
    {{ge::DT_FLOAT,   ge::DT_INT64},   "cast_trait_float_2_int64"},
    {{ge::DT_FLOAT,   ge::DT_INT16},   "cast_trait_float_2_int64"},
    {{ge::DT_FLOAT,   ge::DT_INT32},   "cast_trait_float_2_int32"},
    {{ge::DT_FLOAT,   ge::DT_BF16},    "cast_trait_float_2_bf16"},
    {{ge::DT_FLOAT16, ge::DT_UINT8},   "cast_trait_float_2_int64"},
    {{ge::DT_FLOAT16, ge::DT_INT8},    "cast_trait_half_2_int8"},
    {{ge::DT_FLOAT16, ge::DT_FLOAT},   "cast_trait_bf16_2_float"},
    {{ge::DT_INT32,   ge::DT_FLOAT},   "cast_trait_int32_2_float"},
    {{ge::DT_INT32,   ge::DT_INT16},   "cast_trait_int32_2_int16"},
    {{ge::DT_INT64,   ge::DT_INT32},   "cast_trait_int32_2_int16"},
    {{ge::DT_INT64,   ge::DT_FLOAT},   "cast_trait_int64_2_float"},
    {{ge::DT_BF16,    ge::DT_FLOAT},   "cast_trait_bf16_2_float"},
    {{ge::DT_UINT8,   ge::DT_FLOAT16}, "cast_trait_bf16_2_float"},
    {{ge::DT_INT8,    ge::DT_FLOAT16}, "cast_trait_bf16_2_float"},
    {{ge::DT_INT8,    ge::DT_INT16},   "cast_trait_bf16_2_float"},
    {{ge::DT_INT16,   ge::DT_FLOAT},   "cast_trait_bf16_2_float"},
    {{ge::DT_INT16,   ge::DT_UINT8},   "cast_trait_int16_2_uint8"},
  };
}

namespace codegen {
Status MicroCastApiCall::Generate(const TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe, CallParam &param,
                                  string &result) {
  std::stringstream ss;
  auto input_tensor_id = GetInputTensorIdByIndex(0);
  auto output_tensor_id = GetOutputTensorIdByIndex(0);
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(input_tensor_id));
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(output_tensor_id));

  auto input_dtype = tensor_mng.GetTensor(input_tensor_id)->dtype_;
  auto output_dtype = tensor_mng.GetTensor(output_tensor_id)->dtype_;

  string input_dtype_name;
  string output_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(input_dtype, input_dtype_name), "Get data type:%d failed",
                    static_cast<int32_t>(input_dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(output_dtype, output_dtype_name), "Get data type:%d failed",
                    static_cast<int32_t>(output_dtype));

  ss << "AscendC::MicroAPI::Cast<" << output_dtype_name << ", " << input_dtype_name << ", " << this->cast_trait_ << ">";

  ss << "(" << *(tensor_mng.GetTensor(this->GetOutputTensorIdByIndex(0))) << ", "
     << *(tensor_mng.GetTensor(this->GetInputTensorIdByIndex(0))) << ", " << param.p_reg << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status MicroCastApiCall::Init(const ascir::NodeView &node) {
  std::pair<ge::DataType, ge::DataType> dtype_pair = {node->inputs[0].attr.dtype, node->outputs[0].attr.dtype};
  auto iter = trait_map.find(dtype_pair);
  if (iter != trait_map.end()) {
    cast_trait_ = iter->second;
  } else {
    cast_trait_ = "cast_trait_none";
  }
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroCastApiCall> register_micro_cast_api_call("MicroCastApiCall");
}  // namespace codegen