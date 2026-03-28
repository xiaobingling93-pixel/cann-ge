/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_ENGINE_OP_STREAM_SWITCH_OP_H
#define RTS_ENGINE_OP_STREAM_SWITCH_OP_H
#include "../acl_rt_compare_data_type.h"
#include "../op.h"
#include "../acl_rt_condition.h"

namespace cce {
namespace runtime {
class StreamSwitchOp : public Op {
 public:
  StreamSwitchOp(const ge::Node &node, ge::RunContext &runContext);

  ~StreamSwitchOp() override = default;

  StreamSwitchOp &operator=(const StreamSwitchOp &op) = delete;

  StreamSwitchOp(const StreamSwitchOp &op) = delete;

  /**
   *  @brief init param for generate task
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

  ge::Status UpdateTaskDef(vector<TaskDef> &tasks) override;

  ge::Status GenerateCtxDef(const ge::Node &node) override;

 private:
  aclrtCondition cond_;

  uint32_t trueStreamIndex_;

  aclrtCompareDataType data_type_;
};
}  // namespace runtime
}  // namespace cce

#endif  // RTS_ENGINE_OP_STREAM_SWITCH_OP_H