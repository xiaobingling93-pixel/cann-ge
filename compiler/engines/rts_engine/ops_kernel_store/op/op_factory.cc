/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_factory.h"

#include "graph/op_desc.h"
#include "model_op/end_graph_op.h"
#include "label_control_op/label_goto_ex_op.h"
#include "label_control_op/label_goto_op.h"
#include "label_control_op/label_set_op.h"
#include "label_control_op/label_switch_by_index_op.h"
#include "label_control_op/label_switch_op.h"
#include "memory_op/memcpy_addr_async_op.h"
#include "memory_op/memcpy_async_op.h"
#include "model_op/model_exit_op.h"
#include "npu_op/npu_clear_float_status_op.hpp"
#include "npu_op/npu_get_float_status_op.hpp"
#include "npu_op/npu_get_float_debug_status_op.h"
#include "npu_op/npu_clear_float_debug_status_op.h"
#include "communication_op/recv_op.h"
#include "communication_op/send_op.h"
#include "stream_control_op/stream_active_op.h"
#include "stream_control_op/stream_merge_op.h"
#include "stream_control_op/stream_switchN_op.h"
#include "stream_control_op/stream_switch_op.h"
#include "communication_op/recv_notify_op.h"
#include "communication_op/send_notify_op.h"
#include "memory_op/cmo_addr_op.h"
#include "common/util/log.h"

using namespace ge;
namespace cce {
namespace runtime {
OpFactory &OpFactory::Instance() {
  static OpFactory instance;
  return instance;
}

std::shared_ptr<Op> OpFactory::CreateOp(const Node &node, RunContext &runContext) {
  if (node.GetOpDesc() == nullptr) {
    RTS_REPORT_CALL_ERROR("Create op failed, param can not be NULL.");
    return nullptr;
  }

  auto opDesc = node.GetOpDesc();
  auto iter = op_creator_map_.find(opDesc->GetType());
  RTS_LOGD("op desc type:%s.", (opDesc->GetType()).c_str());
  if (iter != op_creator_map_.end()) {
    return iter->second(node, runContext);
  }

  RTS_REPORT_CALL_ERROR("Not supported OP, type = %s, name = %s", opDesc->GetType().c_str(), opDesc->GetName().c_str());
  return nullptr;
}

void OpFactory::RegisterCreator(const std::string &type, const OP_CREATOR_FUNC &func) {
  if (func == nullptr) {
    RTS_REPORT_CALL_ERROR("Register creator failed, func is NULL.");
  }

  auto iter = op_creator_map_.find(type);
  if (iter != op_creator_map_.end()) {
    RTS_LOGW("%s creator already exist", type.c_str());
    return;
  }

  op_creator_map_[type] = func;
  all_ops_.push_back(type);
}

REGISTER_OP_CREATOR(NetOutput, EndGraphOp);
REGISTER_OP_CREATOR(LabelGotoEx, LabelGotoExOp);
REGISTER_OP_CREATOR(LabelGoto, LabelGotoOp);
REGISTER_OP_CREATOR(LabelSet, LabelSetOp);
REGISTER_OP_CREATOR(LabelSwitchByIndex, LabelSwitchByIndexOp);
REGISTER_OP_CREATOR(LabelSwitch, LabelSwitchOp);
REGISTER_OP_CREATOR(MemcpyAddrAsync, MemcpyAddrAsyncOp);
REGISTER_OP_CREATOR(MemcpyAsync, MemcpyAsyncOp);
REGISTER_OP_CREATOR(Enter, MemcpyAsyncOp);
REGISTER_OP_CREATOR(RefEnter, MemcpyAsyncOp);
REGISTER_OP_CREATOR(LoopCond, MemcpyAsyncOp);
REGISTER_OP_CREATOR(NextIteration, MemcpyAsyncOp);
REGISTER_OP_CREATOR(RefNextIteration, MemcpyAsyncOp);
REGISTER_OP_CREATOR(Exit, MemcpyAsyncOp);
REGISTER_OP_CREATOR(RefExit, MemcpyAsyncOp);
REGISTER_OP_CREATOR(Identity, MemcpyAsyncOp);
REGISTER_OP_CREATOR(ReadVariableOp, MemcpyAsyncOp);
REGISTER_OP_CREATOR(ModelExit, ModelExitOp);
REGISTER_OP_CREATOR(NPUClearFloatStatus, NpuClearFloatStatusOp);
REGISTER_OP_CREATOR(NPUClearFloatStatusV2, NpuClearFloatStatusOp);
REGISTER_OP_CREATOR(NPUGetFloatStatus, NpuGetFloatStatusOp);
REGISTER_OP_CREATOR(NPUGetFloatStatusV2, NpuGetFloatStatusOp);
REGISTER_OP_CREATOR(NPUGetFloatDebugStatus, NpuGetFloatDebugStatusOp);
REGISTER_OP_CREATOR(NPUClearFloatDebugStatus, NpuClearFloatDebugStatusOp);
REGISTER_OP_CREATOR(Recv, RecvOp);
REGISTER_OP_CREATOR(Send, SendOp);
REGISTER_OP_CREATOR(StreamActive, StreamActiveOp);
REGISTER_OP_CREATOR(StreamMerge, StreamMergeOp);
REGISTER_OP_CREATOR(StreamSwitchN, StreamSwitchNOp);
REGISTER_OP_CREATOR(StreamSwitch, StreamSwitchOp);
REGISTER_OP_CREATOR(RecvNotify, RecvNotifyOp);
REGISTER_OP_CREATOR(SendNotify, SendNotifyOp);
REGISTER_OP_CREATOR(Cmo, CmoAddrOp);
REGISTER_OP_CREATOR(RecvMem, RecvOpMem);
REGISTER_OP_CREATOR(SendMem, SendOpMem);
}  // namespace runtime
}  // namespace cce
