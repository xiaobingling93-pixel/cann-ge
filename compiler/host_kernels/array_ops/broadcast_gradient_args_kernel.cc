/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_gradient_args_kernel.h"

#include <vector>

#include "framework/common/types.h"
#include "common/b_cast/b_cast.h"
#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kBCastGradArgsInputsSize = 2;
const size_t kBCastGradArgsOutputsSize = 2;
}  // namespace

Status BroadcastGradientArgsKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                            std::vector<GeTensorPtr> &v_output) {
  GELOGD("BroadcastGradientArgs kernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  // check input size
  bool size_check_fail =
    (op_desc_ptr->GetAllInputsSize() != kBCastGradArgsInputsSize || input.size() != kBCastGradArgsInputsSize ||
     op_desc_ptr->GetAllOutputsDescSize() != kBCastGradArgsOutputsSize);
  if (size_check_fail) {
    GELOGW("input/output size error. InDesc size:%zu, OutDesc size:%zu, in size:%zu ",
           op_desc_ptr->GetAllInputsSize(), op_desc_ptr->GetAllOutputsDescSize(), input.size());
    return NOT_CHANGED;
  }

  std::vector<int64_t> x1_dims;
  std::vector<int64_t> x2_dims;
  DataType x1_data_type = op_desc_ptr->GetInputDesc(0).GetDataType();
  DataType x2_data_type = op_desc_ptr->GetInputDesc(1).GetDataType();
  bool result = (OpUtils::GetShapeDataFromConstTensor(input[0], x1_data_type, x1_dims) == SUCCESS) &&
                (OpUtils::GetShapeDataFromConstTensor(input[1], x2_data_type, x2_dims) == SUCCESS);
  if (!result) {
    GELOGE(PARAM_INVALID, "Get shape data from const tensor fail.");
    return PARAM_INVALID;
  }

  BCast bcast;
  Status ret = bcast.GenerateBcastInfo(x1_dims, x2_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "Generate bcast info fail.");
    return ret;
  }

  std::vector<std::vector<int64_t>> grad_reduce_idx;
  grad_reduce_idx.push_back(bcast.GetGradXReduceIdx());
  grad_reduce_idx.push_back(bcast.GetGradYReduceIdx());

  for (size_t i = 0; i < grad_reduce_idx.size(); i++) {
    ret = KernelUtils::ConstructTensorDescWithData(op_desc_ptr->GetOutputDesc(i), grad_reduce_idx[i], v_output);
    if (ret != SUCCESS) {
      GELOGE(ret, "BroadcastGradientArgs kernel construct tensor desc fail");
      return ret;
    }
  }

  for (const auto &output_tensor : v_output) {
    GE_CHECK_NOTNULL(output_tensor);
    if (output_tensor->GetTensorDesc().GetShape().IsUnknownShape()) {
      GELOGW("Output is unknown shape, [%s] skip BroadcastGradientArgsKernel.", op_desc_ptr->GetName().c_str());
      return NOT_CHANGED;
    }
  }

  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(BROADCASTGRADIENTARGS, BroadcastGradientArgsKernel);
}  // namespace ge
