/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _RTS_ENGINE_OP_STREAM_SWITCHN_OP_H_
#define _RTS_ENGINE_OP_STREAM_SWITCHN_OP_H_
#include "../acl_rt_compare_data_type.h"
#include "../op.h"

namespace cce {
namespace runtime {
class StreamSwitchNOp : public Op {
 public:
  StreamSwitchNOp(const ge::Node &node, ge::RunContext &runContext);

  ~StreamSwitchNOp() override = default;

  StreamSwitchNOp &operator=(const StreamSwitchNOp &op) = delete;

  StreamSwitchNOp(const StreamSwitchNOp &op) = delete;

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

 private:
  uint32_t element_size_;
  uint32_t size_;
  std::vector<int32_t> target_value_;
  std::vector<int64_t> target_value64_;
  aclrtCompareDataType data_type_;
  vector<uint32_t> activeStreamList_;
};
}  // namespace runtime
}  // namespace cce
#endif
