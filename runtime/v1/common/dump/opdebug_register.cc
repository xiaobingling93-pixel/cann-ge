/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump/opdebug_register.h"
#include "graph/def_types.h"

namespace ge {
namespace {
constexpr size_t kOpDebugMemorySize = 2048UL;
constexpr size_t kDebugP2pSize = 8UL;
}  // namespace
std::mutex OpdebugRegister::mu_;
std::map<aclrtStream, std::unique_ptr<OpDebugTask>> OpdebugRegister::op_debug_tasks_;
std::map<aclrtStream, uint32_t>  OpdebugRegister::stream_ref_count_;

OpDebugTask::~OpDebugTask() {
  if (op_debug_addr_ != nullptr) {
    GE_CHK_RT(rtFree(op_debug_addr_));
    op_debug_addr_ = nullptr;
  }
}

Status OpdebugRegister::RegisterDebugForModel(rtModel_t const model_handle, const uint32_t op_debug_mode,
                                              DataDumper &data_dumper) {
  GELOGD("Start to register debug for model in overflow");
  const auto ret = MallocMemForOpdebug();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Malloc][MemForOpdebug]Failed when debug for model overflow, ret:0x%X", ret);
    return ret;
  }
  uint32_t debug_stream_id = 0U;
  uint32_t debug_task_id = 0U;
  const auto rt_ret = rtDebugRegister(model_handle, op_debug_mode, op_debug_addr_, &debug_stream_id, &debug_task_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtDebugRegister error, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGD("debug_task_id:%u, debug_stream_id:%u in model overflow", debug_task_id, debug_stream_id);
  GE_CHK_STATUS_RET(data_dumper.SaveOpDebugId(debug_task_id, debug_stream_id, p2p_debug_addr_, true));
  return SUCCESS;
}

void OpdebugRegister::UnregisterDebugForModel(rtModel_t const model_handle) {
  rtError_t rt_ret = RT_ERROR_NONE;
  if (model_handle != nullptr) {
    GELOGD("start to call rtDebugUnRegister in model overflow.");
    rt_ret = rtDebugUnRegister(model_handle);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtDebugUnRegister failed, ret: 0x%X", rt_ret);
    }
  }

  if (op_debug_addr_ != nullptr) {
    rt_ret = rtFree(op_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    op_debug_addr_ = nullptr;
  }

  if (p2p_debug_addr_ != nullptr) {
    rt_ret = rtFree(p2p_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    p2p_debug_addr_ = nullptr;
  }
  return;
}

Status OpdebugRegister::CreateOpDebugTaskByStream(aclrtStream const stream, const uint32_t op_debug_mode) {
  const std::lock_guard<std::mutex> lk(mu_);
  stream_ref_count_[stream] += 1U;
  if (op_debug_tasks_.find(stream) != op_debug_tasks_.end()) {
    return SUCCESS;
  }
  auto &op_debug_task = op_debug_tasks_[stream];
  op_debug_task = MakeUnique<OpDebugTask>();
  GE_CHECK_NOTNULL(op_debug_task);
  const auto memory_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, kOpDebugMemorySize);
  GE_CHK_RT_RET(rtMalloc(&op_debug_task->op_debug_addr_, kOpDebugMemorySize, memory_type, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtDebugRegisterForStream(stream, op_debug_mode, op_debug_task->op_debug_addr_,
                                         &op_debug_task->debug_stream_id_, &op_debug_task->debug_task_id_));
  return SUCCESS;
}

Status OpdebugRegister::MallocP2PDebugMem(const void * const op_debug_addr) {
  const uint64_t debug_addrs_tmp = PtrToValue(op_debug_addr);
  GE_CHK_RT_RET(rtMalloc(&p2p_debug_addr_, kDebugP2pSize, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  GE_CHK_RT_RET(rtMemcpy(p2p_debug_addr_, sizeof(uint64_t), &debug_addrs_tmp, sizeof(uint64_t),
                         RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status OpdebugRegister::RegisterDebugForStream(aclrtStream const stream, const uint32_t op_debug_mode,
                                               DataDumper &data_dumper) {
  GELOGD("Start to register debug for stream in stream overflow");
  GE_CHK_STATUS_RET(CreateOpDebugTaskByStream(stream, op_debug_mode));
  auto &op_debug_task = op_debug_tasks_[stream];

  GELOGD("debug_task_id:%u, debug_stream_id:%u in stream overflow.",
         op_debug_task->debug_task_id_, op_debug_task->debug_stream_id_);

  GE_CHK_STATUS_RET(MallocP2PDebugMem(op_debug_task->op_debug_addr_));
  GE_CHK_STATUS_RET(data_dumper.SaveOpDebugId(op_debug_task->debug_task_id_, op_debug_task->debug_stream_id_, p2p_debug_addr_, true));
  return SUCCESS;
}

void OpdebugRegister::UnregisterDebugForStream(aclrtStream const stream) {
  rtError_t rt_ret = RT_ERROR_NONE;
  if (stream != nullptr) {
    const std::lock_guard<std::mutex> lk(mu_);
    stream_ref_count_[stream] -= 1U;
    if (stream_ref_count_[stream] == 0U) {
      GELOGD("start call rtDebugUnRegisterForStream in unknown shape over flow.");
      GE_CHK_RT(rtDebugUnRegisterForStream(stream));
      const auto iter = op_debug_tasks_.find(stream);
      if (iter != op_debug_tasks_.end()) {
        (void)op_debug_tasks_.erase(iter);
      }
    }
  }

  if (p2p_debug_addr_ != nullptr) {
    rt_ret = rtFree(p2p_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    p2p_debug_addr_ = nullptr;
  }
  return;
}

Status OpdebugRegister::MallocMemForOpdebug() {
  const auto memory_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, kOpDebugMemorySize);
  GELOGI("memory_type: %u", memory_type);
  rtError_t rt_ret = rtMalloc(&op_debug_addr_, kOpDebugMemorySize, memory_type, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtMalloc]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtMalloc failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  const uint64_t debug_addrs_tmp = PtrToValue(op_debug_addr_);
  // For data dump, aicpu needs the pointer to pointer that save the real debug address.
  rt_ret = rtMalloc(&p2p_debug_addr_, kDebugP2pSize, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtMalloc]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtMalloc failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtMemcpy(p2p_debug_addr_, sizeof(uint64_t), &debug_addrs_tmp, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtMemcpy]To p2p_addr error %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpy to p2p_addr error %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

}  // namespace ge
