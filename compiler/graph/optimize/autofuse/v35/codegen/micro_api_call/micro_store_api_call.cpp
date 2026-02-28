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

#include "micro_store_api_call.h"

namespace {
// 数据类型大小常量定义
constexpr int DTYPE_SIZE_1BYTE = 1;
constexpr int DTYPE_SIZE_2BYTE = 2;
constexpr int DTYPE_SIZE_4BYTE = 4;
constexpr int DTYPE_SIZE_8BYTE = 8;
}  // namespace

namespace codegen {
Status MicroStoreApiCall::Generate(const codegen::TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param,
                                   string &result) {
  std::stringstream ss;
  auto tensor_id = GetInputTensorIdByIndex(0);
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(tensor_id));
  ss << "AscendC::MicroAPI::StoreAlign";
  if (!dist_.empty()) {
    auto dtype = tensor_mng.GetTensor(tensor_id)->dtype_;
    string dtype_name;
    Tensor::DtypeName(dtype, dtype_name);
    ss << "<" << dtype_name << ", AscendC::MicroAPI::StoreDist::" << this->dist_ << ">";
  }
  ss << "(" << *(tpipe.GetTensor(this->GetOutputTensorIdByIndex(0))) << " + " << param.offset << ", "
     << *(tensor_mng.GetTensor(tensor_id))
     << (tensor_mng.GetTensor(tensor_id)->init_as_mask_reg_ == true ? "" : ", " + param.p_reg)
     << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status MicroStoreApiCall::Init(const ascir::NodeView &node) {
  this->dist_ = "";
  auto in_node = std::dynamic_pointer_cast<ge::AscNode>(node->inputs[0].anchor.GetOwnerNode());
  auto in_dtype_size = ge::GetSizeByDataType(in_node->inputs[0].attr.dtype);
  auto out_dtype_size = ge::GetSizeByDataType(in_node->outputs[0].attr.dtype);
  bool is_all_zero = std::all_of(node->outputs[0].attr.vectorized_strides.begin(),
                                 node->outputs[0].attr.vectorized_strides.end(), [](const ascir::SizeExpr &stride) {
                                   return ge::SymbolicUtils::StaticCheckEq(stride.Simplify(), ge::sym::kSymbolZero) ==
                                          ge::TriBool::kTrue;
                                 });
  if (is_all_zero) {
    if (in_dtype_size == DTYPE_SIZE_1BYTE) {
      this->dist_ = "DIST_FIRST_ELEMENT_B8";
    } else if (in_dtype_size == DTYPE_SIZE_2BYTE) {
      this->dist_ = "DIST_FIRST_ELEMENT_B16";
    } else if (in_dtype_size == DTYPE_SIZE_4BYTE) {
      this->dist_ = "DIST_FIRST_ELEMENT_B32";
    }
  } else if (ge::ops::IsOps<ge::ascir_op::Cast>(in_node)) {
    if (in_dtype_size == DTYPE_SIZE_2BYTE && out_dtype_size == DTYPE_SIZE_1BYTE) {
      this->dist_ = "DIST_PACK_B16";
    } else if (in_dtype_size == DTYPE_SIZE_4BYTE && out_dtype_size == DTYPE_SIZE_2BYTE) {
      this->dist_ = "DIST_PACK_B32";
    } else if (in_dtype_size == DTYPE_SIZE_8BYTE && out_dtype_size == DTYPE_SIZE_4BYTE) {
      this->dist_ = "DIST_PACK_B64";
    } else if (in_dtype_size == DTYPE_SIZE_4BYTE && out_dtype_size == DTYPE_SIZE_1BYTE) {
      this->dist_ = "DIST_PACK4_B32";
    }
  }
  return ge::SUCCESS;
}
static MicroApiCallRegister<MicroStoreApiCall> register_micro_store_api_call("MicroStoreApiCall");
}  // namespace codegen