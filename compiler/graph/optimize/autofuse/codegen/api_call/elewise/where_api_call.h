/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_WHERE_API_CALL_H__
#define __AUTOFUSE_WHERE_API_CALL_H__
#include "codegen_kernel.h"
#include "api_call/utils/api_call_utils.h"

namespace codegen {
class WhereApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit WhereApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~WhereApiCall() final = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                 const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                 const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
 private:
  // 准备输入输出tensor引用
  Status PrepareInputsAndOutputs(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                 const Tensor *&x1, const Tensor *&x2, const Tensor *&x3,
                                 const Tensor *&y) const;

  // 获取临时缓冲区ID
  Status GetTempBufferId(int64_t &id) const;

  // 生成循环参数
  Status GenerateLoopParams(const Tensor &x1, const Tensor &x2, const Tensor &x3, const Tensor &y,
                            const TPipe &tpipe, ApiLoopParams &param) const;

  // 生成无循环情况的处理代码
  Status GenerateNoLoopCase(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                           const Tensor &x1, const Tensor &x2, const Tensor &x3, const Tensor &y,
                           const std::string &x2_scalar, const std::string &x3_scalar,
                           const int64_t id, std::stringstream &ss) const;

  // 生成双标量情况的处理代码
  Status GenerateBothScalarCase(const TPipe &tpipe, const ApiLoopParams &param,
                                const Tensor &x1, const Tensor &y,
                                const std::string &scalar_local_blk_tensor_name_x2,
                                const std::string &scalar_local_blk_tensor_name_x3,
                                const int64_t id, std::stringstream &ss) const;

  // 生成x2标量情况的处理代码
  Status GenerateX2ScalarCase(const TPipe &tpipe, const ApiLoopParams &param,
                             const Tensor &x1, const Tensor &x3, const Tensor &y,
                             const std::string &scalar_local_blk_tensor_name_x2,
                             const int64_t id, std::stringstream &ss) const;

  // 生成x3标量情况的处理代码
  Status GenerateX3ScalarCase(const TPipe &tpipe, const ApiLoopParams &param,
                             const Tensor &x1, const Tensor &x2, const Tensor &y,
                             const std::string &scalar_local_blk_tensor_name_x3,
                             const int64_t id, std::stringstream &ss) const;

  // 生成正常情况的处理代码
  Status GenerateNormalCase(const TPipe &tpipe, const ApiLoopParams &param,
                           const Tensor &x1, const Tensor &x2, const Tensor &x3, const Tensor &y,
                           const int64_t id, std::stringstream &ss) const;
};
}
#endif // __AUTOFUSE_WHERE_API_CALL_H__