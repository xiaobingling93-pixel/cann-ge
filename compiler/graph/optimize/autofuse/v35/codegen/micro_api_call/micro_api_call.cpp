/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "micro_api_call.h"

#include <sstream>
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "micro_api_call/micro_api_call_factory.h"

using namespace std;
namespace codegen {
  Status TensorManager::AddTensor(const MicroApiTensor &tensor) {
    auto is_insert = this->tensors_.emplace(tensor.id_, tensor).second;
    GE_CHK_BOOL_RET_STATUS(is_insert, ge::FAILED, "Micro api tensor[%u,%s] is already added", tensor.id_,
                           tensor.name.c_str());
    return ge::SUCCESS;
  }

  const MicroApiTensor *TensorManager::GetTensor(ascir::TensorId id) const {
    auto iter = tensors_.find(id);
    GE_CHK_BOOL_EXEC(iter != tensors_.end(), return nullptr, "Micro api tensor[%ld] not found", id);
    return &iter->second;
  }

  Status TensorManager::GenerateVreg(std::string &result) const {
    GE_ASSERT_TRUE(!this->tensors_.empty(), "Micro api tensor map is empty.");
    stringstream ss;
    for (const auto &[tensor_id, m_api_tensor] : this->tensors_) {
      (void)tensor_id;
      ss << m_api_tensor.Define() << std::endl;
    }
    result = ss.str();
    return ge::SUCCESS;
  }

  const Type MicroApiTensor::UBTensorTypes(std::string &dtype_name) {
    return Type("__local_mem__ " + dtype_name + "*");
  }

  const Type MicroApiTensor::RegTensorTypes(std::string &dtype_name) {
    return Type("AscendC::MicroAPI::RegTensor<" + dtype_name + ">");
  }

  const Type MicroApiTensor::MaskRegTypes() {
    return Type("AscendC::MicroAPI::MaskReg");
  }

  MicroApiTensor::MicroApiTensor(const ascir::TensorAttr &tensor, std::string &dtype_name, bool init_as_mask_reg)
      : Variable((init_as_mask_reg ? MaskRegTypes() : RegTensorTypes(dtype_name)),
                 (init_as_mask_reg ? "mask_reg_" : "vreg_") + to_string(tensor.attr.mem.tensor_id)),
        id_(tensor.attr.mem.tensor_id),
        dtype_(tensor.attr.dtype),
        position_(tensor.attr.mem.position),
        axis_(tensor.attr.axis),
        axis_size_(tensor.attr.repeats),
        axis_strides_(tensor.attr.strides),
        vectorized_axis_(tensor.attr.vectorized_axis),
        vectorized_strides_(tensor.attr.vectorized_strides),
        size_(this->name + "_size"),
        actual_size_(this->name + "_actual_size"),
        init_as_mask_reg_(init_as_mask_reg) {}

// 生成micro api的调用
  Status MicroApiCall::Generate(const TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe, CallParam &param,
                                std::string &result) {
    stringstream ss;
    ss << "AscendC::MicroAPI::" << this->api_name_ << "(";
    for (auto out_arg : this->outputs_) {
      ss << *(tensor_mng.GetTensor(out_arg.second)) << ", ";
    }
    for (auto in_arg : this->inputs_) {
      if (in_arg.first == TensorType::REG_TENSOR) {
        ss << *(tensor_mng.GetTensor(in_arg.second)) << ", ";
      } else {
        ss << *(tpipe.GetTensor(in_arg.second)) << ", ";
      }
    }
    ss << param.p_reg << ");" << endl;
    result = ss.str();
    return ge::SUCCESS;
  }

  static MicroApiCallRegister<MicroApiCall> register_micro_api_call("MicroApiCall");
}  // namespace codegen