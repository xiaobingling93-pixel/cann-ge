/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_ENGINE_OP_LABEL_SWITCH_BY_INDEX_OP_H
#define RTS_ENGINE_OP_LABEL_SWITCH_BY_INDEX_OP_H
#include "../op.h"

namespace cce {
namespace runtime {
class LabelSwitchByIndexOp : public Op {
 public:
  LabelSwitchByIndexOp(const ge::Node &node, ge::RunContext &runContext);

  ~LabelSwitchByIndexOp() override = default;

  LabelSwitchByIndexOp &operator=(const LabelSwitchByIndexOp &op) = delete;

  LabelSwitchByIndexOp(const LabelSwitchByIndexOp &op) = delete;

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
  uint32_t branch_max_;     // max branch count.
  ge::DataType data_type_;  // datatype for load arsize.
};
}  // namespace runtime
}  // namespace cce

#endif  // RTS_ENGINE_OP_LABEL_SWITCH_BY_INDEX_OP_H
