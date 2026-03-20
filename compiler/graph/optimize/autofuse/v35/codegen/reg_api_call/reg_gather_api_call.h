/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_GATHER_API_CALL_H__
#define __AUTOFUSE_REG_GATHER_API_CALL_H__
#include "codegen_kernel.h"

namespace codegen {
class GatherRegApiCall : public ApiCall {
public:
  using ApiCall::Generate;
  explicit GatherRegApiCall(const std::string &api_name) : ApiCall(api_name) {}
  Status ParseAttr(const ascir::NodeView &node) override;
  ~GatherRegApiCall() final = default;
  Status GenerateComputeTypeGather(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                  const int64_t tmp_buf_id, std::string &result) const;
  Status GenerateComputeTypeLoad(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                const int64_t tmp_buf_id, std::string &result) const;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
private:
  Status GetGatherCase(const Tensor &x1, std::string &result) const;
  ge::ComputeType compute_type;
  int64_t axis = 0;
  bool negative_index_support = false;
};
}
#endif // __AUTOFUSE_REG_GATHER_API_CALL_H__