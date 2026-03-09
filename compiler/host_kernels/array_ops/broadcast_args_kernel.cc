/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_args_kernel.h"

#include <vector>

#include "framework/common/types.h"
#include "common/b_cast/b_cast.h"
#include "host_kernels/kernel_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kBCastArgsInputsSize = 2;
const size_t kBCastArgsOutputsSize = 1;
}  // namespace

Status BroadcastArgsKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                    std::vector<GeTensorPtr> &v_output) {
  GELOGD("BroadcastArgsKernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  // check input size
  bool size_check =
      (op_desc_ptr->GetAllInputsSize() != kBCastArgsInputsSize || input.size() != kBCastArgsInputsSize ||
       op_desc_ptr->GetAllOutputsDescSize() != kBCastArgsOutputsSize);
  if (size_check) {
    GELOGW("input/output size error. InDesc size:%zu,"
           "OutDesc size:%zu, in size:%zu ",
           op_desc_ptr->GetAllInputsSize(), op_desc_ptr->GetAllOutputsDescSize(), input.size());
    return NOT_CHANGED;
  }

  std::vector<int64_t> x1_dims;
  std::vector<int64_t> x2_dims;
  const auto &op_in_desc = op_desc_ptr->MutableInputDesc(0);
  GE_CHECK_NOTNULL(op_in_desc);
  DataType data_type = op_in_desc->GetDataType();
  bool result = (OpUtils::GetShapeDataFromConstTensor(input[0], data_type, x1_dims) == SUCCESS) &&
                (OpUtils::GetShapeDataFromConstTensor(input[1], data_type, x2_dims) == SUCCESS);
  if (!result) {
    GELOGE(PARAM_INVALID, "GetShapeDataFromConstTensor fail.");
    return PARAM_INVALID;
  }

  BCast bcast;
  Status ret = bcast.GenerateBcastInfo(x1_dims, x2_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "GenerateBcastInfo fail.");
    return ret;
  }

  std::vector<int64_t> bcast_dims = bcast.GetOutputShape();
  ret = KernelUtils::ConstructTensorDescWithData(op_desc_ptr->GetOutputDesc(0), bcast_dims, v_output);
  if (ret != SUCCESS) {
    GELOGE(ret, "BroadcastArgs kernel construct tensor desc fail");
    return ret;
  }

  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(BROADCASTARGS, BroadcastArgsKernel);
}  // namespace ge
