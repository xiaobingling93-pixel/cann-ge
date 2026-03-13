/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_ENGINE_OP_MEMCPY_ADDR_ASYNC_OP_H
#define RTS_ENGINE_OP_MEMCPY_ADDR_ASYNC_OP_H

#include "../op.h"

namespace cce {
namespace runtime {
class MemcpyAddrAsyncOp : public Op {
 public:
  MemcpyAddrAsyncOp(const ge::Node &node, ge::RunContext &runContext);

  ~MemcpyAddrAsyncOp() override = default;

  MemcpyAddrAsyncOp &operator=(const MemcpyAddrAsyncOp &op) = delete;

  MemcpyAddrAsyncOp(const MemcpyAddrAsyncOp &op) = delete;

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

 private:
  rtError_t AddArgsFormatDescInfo(domi::MemcpyAsyncDef * const memcpyAsyncDef, const uint64_t count);
};
}  // namespace runtime
}  // namespace cce

#endif  // RTS_ENGINE_OP_MEMCPY_ADDR_ASYNC_OP_H