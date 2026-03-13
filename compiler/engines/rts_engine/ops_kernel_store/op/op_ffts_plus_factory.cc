/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_ffts_plus_factory.h"

#include "graph/op_desc.h"
#include "label_control_op/label_set_op.h"
#include "label_control_op/label_switch_by_index_op.h"
#include "label_control_op/label_switch_op.h"
#include "memory_op/memcpy_async_op.h"
#include "common/util.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
OpFftsPlusFactory &OpFftsPlusFactory::Instance() {
  static OpFftsPlusFactory instance;
  return instance;
}

std::shared_ptr<Op> OpFftsPlusFactory::CreateOp(const Node &node, RunContext &runContext) {
  if (node.GetOpDesc() == nullptr) {
    RTS_REPORT_CALL_ERROR("Create ffts plus op failed, param can not be NULL.");
    return nullptr;
  }

  auto opDesc = node.GetOpDesc();
  auto iter = opCreatorMap_.find(opDesc->GetType());
  RTS_LOGD("ffts plus op desc type:%s.", (opDesc->GetType()).c_str());
  if (iter != opCreatorMap_.end()) {
    return iter->second(node, runContext);
  }

  RTS_REPORT_CALL_ERROR("Not supported OP, type = %s, name = %s", opDesc->GetType().c_str(), opDesc->GetName().c_str());
  return nullptr;
}

void OpFftsPlusFactory::RegisterCreator(const std::string &type, const OP_CREATOR_FUNC &func) {
  if (func == nullptr) {
    RTS_REPORT_CALL_ERROR("Register ffts plus creator failed, func is NULL.");
    return;
  }

  auto iter = opCreatorMap_.find(type);
  if (iter != opCreatorMap_.end()) {
    RTS_LOGW("%s ffts plus creator already exist", type.c_str());
    return;
  }

  opCreatorMap_[type] = func;
  allOps_.push_back(type);
}

REGISTER_OP_FFTS_PLUS_CREATOR(LabelSet, LabelSetOp);
REGISTER_OP_FFTS_PLUS_CREATOR(LabelSwitchByIndex, LabelSwitchByIndexOp);
REGISTER_OP_FFTS_PLUS_CREATOR(LabelSwitch, LabelSwitchOp);
REGISTER_OP_FFTS_PLUS_CREATOR(MemcpyAsync, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(Enter, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(RefEnter, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(LoopCond, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(NextIteration, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(RefNextIteration, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(Exit, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(RefExit, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(Identity, MemcpyAsyncOp);
REGISTER_OP_FFTS_PLUS_CREATOR(ReadVariableOp, MemcpyAsyncOp);

}  // namespace runtime
}  // namespace cce
