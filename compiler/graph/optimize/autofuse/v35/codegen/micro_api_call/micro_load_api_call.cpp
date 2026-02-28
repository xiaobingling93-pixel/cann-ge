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

#include "micro_load_api_call.h"

namespace {
// 数据类型大小常量定义
constexpr int DTYPE_SIZE_1BYTE = 1;
constexpr int DTYPE_SIZE_2BYTE = 2;
constexpr int DTYPE_SIZE_4BYTE = 4;
constexpr int DTYPE_SIZE_8BYTE = 8;
}  // namespace

namespace codegen {
Status MicroLoadApiCall::Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param,
                                  string &result) {
  std::stringstream ss;
  auto tensor_id = GetOutputTensorIdByIndex(0);
  GE_ASSERT_NOTNULL(tensor_mng.GetTensor(tensor_id));
  ss << "AscendC::MicroAPI::LoadAlign";
  if (!dist_.empty()) {
    auto dtype = tensor_mng.GetTensor(tensor_id)->dtype_;
    string dtype_name;
    Tensor::DtypeName(dtype, dtype_name);
    ss << "<" << dtype_name << ", AscendC::MicroAPI::LoadDist::" << this->dist_ << ">";
  }
  ss << "(" << *(tensor_mng.GetTensor(tensor_id)) << ", " << *(tpipe.GetTensor(this->GetInputTensorIdByIndex(0)))
     << " + " << param.offset << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status MicroLoadApiCall::Init(const ascir::NodeView &node) {
  this->dist_ = "";
  for (auto outAnchor : node->GetAllOutAnchors()) {
    GE_ASSERT_NOTNULL(outAnchor);
    GE_ASSERT_NOTNULL(outAnchor->GetFirstPeerAnchor());
    auto out_node = std::dynamic_pointer_cast<ge::AscNode>(outAnchor->GetFirstPeerAnchor()->GetOwnerNode());
    if (ge::ops::IsOps<ge::ascir_op::Cast>(out_node)) {
      auto in_dtype_size = ge::GetSizeByDataType(out_node->inputs[0].attr.dtype);
      auto out_dtype_size = ge::GetSizeByDataType(out_node->outputs[0].attr.dtype);
      if (in_dtype_size == DTYPE_SIZE_1BYTE && out_dtype_size == DTYPE_SIZE_2BYTE) {
        this->dist_ = "DIST_UNPACK_B8";
      } else if (in_dtype_size == DTYPE_SIZE_2BYTE && out_dtype_size == DTYPE_SIZE_4BYTE) {
        this->dist_ = "DIST_UNPACK_B16";
      } else if (in_dtype_size == DTYPE_SIZE_1BYTE && out_dtype_size == DTYPE_SIZE_4BYTE) {
        this->dist_ = "DIST_UNPACK4_B8";
      } else if (in_dtype_size == DTYPE_SIZE_4BYTE && out_dtype_size == DTYPE_SIZE_8BYTE) {
        this->dist_ = "DIST_UNPACK_B32";
      }
    }
  }
  return ge::SUCCESS;
}

// 对于DIST_BRC_XXX属性，由于Init函数入参不包含TPipe字段，导致无法拿到输入tensor的stride信息，去判断是否需要使用DIST_BRC_XXX模式
// 因此需要在Generate函数中额外判断一次
// 典型场景: src0 (A0, B1) + src(A0, A1)
// src0在Load时，随路完成尾轴brc, 根据尾轴stride是否为0，判断是否采用尾轴brc
Status MicroLoadApiCall::UpdateDistModeByStrideInfo(const TPipe &tpipe) {
  auto tensor_id = GetInputTensorIdByIndex(0);
  const Tensor *tensor_ptr = tpipe.GetTensor(tensor_id);
  GE_ASSERT_NOTNULL(tensor_ptr);
  ascir::SizeExpr last_dim_stride = tensor_ptr->vectorized_strides.back();
  if (ge::SymbolicUtils::StaticCheckEq(last_dim_stride.Simplify(), ge::sym::kSymbolZero) != ge::TriBool::kTrue) {
    // 尾轴stride不为0，默认采用DIST_NORM加载
    return ge::SUCCESS;
  }

  bool is_all_zero = std::all_of(
      tensor_ptr->vectorized_strides.begin(), tensor_ptr->vectorized_strides.end(), [](const ascir::SizeExpr &stride) {
        return ge::SymbolicUtils::StaticCheckEq(stride.Simplify(), ge::sym::kSymbolZero) == ge::TriBool::kTrue;
      });
  if (is_all_zero) {
    // 如果stride全部为0，也是用DIST_NORM模式进行加载
    return ge::SUCCESS;
  }

  std::map<int, string> LOAD_BRC_DIST_MODE = {
      {DTYPE_SIZE_1BYTE, "DIST_BRC_B8"}, {DTYPE_SIZE_2BYTE, "DIST_BRC_B16"}, {DTYPE_SIZE_4BYTE, "DIST_BRC_B32"}};
  auto dtype_size = ge::GetSizeByDataType(tensor_ptr->dtype);
  this->dist_ = LOAD_BRC_DIST_MODE[dtype_size];
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroLoadApiCall> register_micro_load_api_call("MicroLoadApiCall");
}  // namespace codegen