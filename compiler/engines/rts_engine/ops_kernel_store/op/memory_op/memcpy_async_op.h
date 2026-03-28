/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _RTS_ENGINE_OP_MEMCPY_ASYNC_OP_H_
#define _RTS_ENGINE_OP_MEMCPY_ASYNC_OP_H_

#include "../op.h"
#include "../acl_rt_memcpy_kind.h"

using namespace ge;
using namespace std;

namespace cce {
namespace runtime {
typedef struct ffts_plus_memcpy_async_op_sqe_header {
  uint32_t opcode : 8;
  uint32_t ie : 1;
  uint32_t sssv : 1;
  uint32_t dssv : 1;
  uint32_t sns : 1;
  uint32_t dns : 1;
  uint32_t qos : 4;
  uint32_t sro : 1;
  uint32_t dro : 1;
  uint32_t partid : 8;
  uint32_t mpamns : 1;
  uint32_t pmg : 2;
  uint32_t format : 1;
  uint32_t res1 : 1;
} rt_fftsplus_memcpy_async_op_sqe_header_t;

class MemcpyAsyncOp : public Op {
 public:
  MemcpyAsyncOp(const ge::Node &node, ge::RunContext &runContext);

  ~MemcpyAsyncOp() override = default;

  MemcpyAsyncOp &operator=(const MemcpyAsyncOp &op) = delete;

  MemcpyAsyncOp(const MemcpyAsyncOp &op) = delete;

  /**
   *  @brief init param.
   *  @return SUCCESS: init success
   *          other: init failed
   */
  ge::Status Init() override;

  /**
   *  @brief generate task
   *  @return SUCCESS: run success
   *          other: run failed
   */
  ge::Status Run(vector<TaskDef> &tasks) override;

  ge::Status GenerateCtxDef(const ge::Node &node) override;

 private:
  ge::Status CheckPara();

  ge::graphStatus GetTensorSizeInBytesWithNoErrorOutput(const ge::GeTensorDesc &desc, int64_t &size);

  ge::Status CheckInputSize(size_t index, int64_t &inputSize);

  void ConstructSdmaSqeHeader(rt_fftsplus_memcpy_async_op_sqe_header_t *sdma_header);

  ge::Status FillContextInfo(const ge::Node &node, size_t index);

  uint64_t CalculateMemcpyAsyncSingleMaxSize(const rtMemcpyKind_t kind) const;
};
}  // namespace runtime
}  // namespace cce

#endif
