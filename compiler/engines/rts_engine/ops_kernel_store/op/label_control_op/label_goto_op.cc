/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "label_goto_op.h"

#include "op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
LabelGotoOp::LabelGotoOp(const Node &node, RunContext &runContext) : Op(node, runContext) {}

Status LabelGotoOp::Init() {
  RTS_LOGI("LabelGotoOp Init, node: %s.", name_.c_str());
  input_num_ = 0;
  output_num_ = 0;

  Status ret = Op::Init();
  if (ret != SUCCESS) {
    RTS_LOGE("Op::Init failed, node: %s, retCode: %#x", name_.c_str(), ret);
    return ret;
  }

  uint32_t label_index = 0;
  if (!AttrUtils::GetInt(op_desc_, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    RTS_REPORT_CALL_ERROR("LabelGotoOp: %s attr [%s] not exist.", name_.c_str(), ATTR_NAME_LABEL_SWITCH_INDEX.c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status LabelGotoOp::Run(vector<TaskDef> &tasks) {
  (void)tasks;
  RTS_LOGE("not supported LabelGotoOp.");
  return FAILED;
}

}  // namespace runtime
}  // namespace cce
