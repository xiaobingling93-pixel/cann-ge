/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils.h"
#include "ge/ge_api_types.h"
#include "rt_error_codes.h"
#include "runtime/rt.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "common/debug/ge_log.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/kernel_context.h"
#include "core/builder/node_types.h"
#include "mmpa_api.h"
#include "base/err_msg.h"

namespace gert {
namespace {
ge::Status GetStreamByIndex(const Node *node, rtStream_t &stream, size_t index) {
  auto kernel_context = reinterpret_cast<const KernelContext *>(&node->context);
  GE_ASSERT_NOTNULL(kernel_context);
  stream = kernel_context->GetInputValue<rtStream_t>(index);
  return ge::SUCCESS;
}
}  // namespace

ge::Status DoRtStreamSyncWithTimeout(rtStream_t stream) {
  auto timeout = ge::GetContext().StreamSyncTimeout();
  auto rt_ret = rtStreamSynchronizeWithTimeout(stream, timeout);
  if (rt_ret == ACL_ERROR_RT_STREAM_SYNC_TIMEOUT) {
    GELOGE(rt_ret, "[Invoke][rtStreamSynchronizeWithTimeout] failed, stream synchronize timeout:%d, ret:%d.", timeout,
           rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "rtStreamSynchronizeWithTimeout failed, stream synchronize timeout:%d, ret:%d.",
                      timeout, rt_ret);
    return ge::FAILED;
  } else if (rt_ret == ACL_ERROR_RT_END_OF_SEQUENCE) {
    GELOGD("SyncStream return END_OF_SEQUENCE");
    return ge::END_OF_SEQUENCE;
  }
  GE_ASSERT_RT_OK(rt_ret);
  return ge::SUCCESS;
}

ge::Status GetKernelStream(const Node *node, rtStream_t &stream) {
  if (node == nullptr) {
    return ge::SUCCESS;
  }
  const auto kernel_extend_info = reinterpret_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
  GE_ASSERT_NOTNULL(kernel_extend_info);
  const auto kernel_type = kernel_extend_info->GetKernelType();
  if (IsAiCoreLaunchNode(kernel_type) || IsLaunchFFTSPlusTaskNode(kernel_type)) {
    GE_ASSERT_SUCCESS(GetStreamByIndex(node, stream, 0U));
  }
  if (IsAiCpuLaunchNode(kernel_type) || IsHcomLaunchNode(kernel_type)) {
    GE_ASSERT_SUCCESS(GetStreamByIndex(node, stream, 1U));
  }
  return ge::SUCCESS;
}

bool IsInputPlacementOnDeviceHbm() {
  if (ge::GetContext().GetHostExecFlag()) {
    return false;
  }
  std::string input_placement;
  (void)ge::GetThreadLocalContext().GetOption("ge.inputPlacement", input_placement);
  return input_placement == "DeviceHbm";
}

bool IsEnableRmLaunchFreeEdge() {
  const char_t *max_runtime_core_num = nullptr;
  MM_SYS_GET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, max_runtime_core_num);
  int32_t max_core_num;
  if (max_runtime_core_num != nullptr) {
    GE_ASSERT_SUCCESS(ge::ConvertToInt32(std::string(max_runtime_core_num), max_core_num));
  } else {
    max_core_num = 1;
  }
  return max_core_num > 1;
}
}  // namespace gert
