/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_MUL_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_MUL_KERNEL_H_

#include <vector>

#include "graph/ge_tensor.h"
#include "host_kernels/kernel.h"
#include "common/fp16_t/fp16_t.h"

namespace ge {
class MulKernel : public Kernel {
 public:
  Status Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                 std::vector<GeTensorPtr> &v_output) override;

 private:
  Status MulCheck(const std::vector<ConstGeTensorPtr> &input) const;
  std::vector<int8_t> y_data_int8_t_;
  std::vector<int16_t> y_data_int16_t_;
  std::vector<int32_t> y_data_int32_t_;
  std::vector<int64_t> y_data_int64_t_;
  std::vector<uint8_t> y_data_uint8_t_;
  std::vector<uint16_t> y_data_uint16_t_;
  std::vector<uint32_t> y_data_uint32_t_;
  std::vector<uint64_t> y_data_uint64_t_;
  std::vector<fp16_t> y_data_fp16_t_;
  std::vector<float> y_data_float_;
  std::vector<double> y_data_double_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_MUL_KERNEL_H_
