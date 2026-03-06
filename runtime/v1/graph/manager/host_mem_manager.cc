/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/host_mem_manager.h"

#include <sstream>

#include "graph/ge_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/def_types.h"
#include "runtime/rt.h"
#include "base/err_msg.h"
#include "acl/acl_rt.h"

namespace {
constexpr uint32_t kMallocHostMemFlag = 0U;
}  // namespace
namespace ge {
Status SharedMemAllocator::Allocate(SharedMemInfo &mem_info) {
  const auto device_id = GetContext().DeviceId();
  GELOGD("SharedMemAllocator::Malloc host mem size= %zu for devid:[%u].", mem_info.mem_size, device_id);

  const int32_t dev_id = static_cast<int32_t>(device_id);
  GE_CHK_RT_RET(aclrtSetDevice(dev_id));
  // DeviceReset before memory finished!
  GE_MAKE_GUARD(not_used_var, [&dev_id]() { GE_CHK_RT(aclrtResetDevice(dev_id)); });

  rtMallocHostSharedMemoryIn input_para = {mem_info.shm_name.c_str(), mem_info.mem_size, kMallocHostMemFlag};
  rtMallocHostSharedMemoryOut output_para;
  const rtError_t rt_ret = rtMallocHostSharedMemory(&input_para, &output_para);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMallocHostSharedMemory fail, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMallocHostSharedMemory] failed, devid:[%u].", device_id);
    return GE_GRAPH_MEMORY_ALLOC_FAILED;
  }
  mem_info.fd = output_para.fd;
  mem_info.host_aligned_ptr = AlignedPtr::BuildFromAllocFunc(
      [&output_para](std::unique_ptr<uint8_t[], AlignedPtr::Deleter> &ptr) {
        ptr.reset(PtrToPtr<void, uint8_t>(output_para.ptr));
      },
      [](const uint8_t *const ptr) { (void)ptr; });
  mem_info.device_address = PtrToPtr<void, uint8_t>(output_para.devPtr);
  return SUCCESS;
}

Status SharedMemAllocator::DeAllocate(const SharedMemInfo &mem_info) {
  GELOGD("SharedMemAllocator::DeAllocate");
  rtFreeHostSharedMemoryIn free_para = {mem_info.shm_name.c_str(), mem_info.mem_size, mem_info.fd,
                                        mem_info.host_aligned_ptr->MutableGet(), mem_info.device_address};
  const rtError_t rt_ret = rtFreeHostSharedMemory(&free_para);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtFreeHostSharedMemory fail, ret:%d", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtFreeHostSharedMemory] failed, ret:%d.", rt_ret);
    return RT_FAILED;
  }
  return ge::SUCCESS;
}

HostMemManager &HostMemManager::Instance() {
  static HostMemManager mem_manager;
  return mem_manager;
}

Status HostMemManager::Initialize() {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  allocator_ = std::unique_ptr<SharedMemAllocator>(new (std::nothrow) SharedMemAllocator());
  if (allocator_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New SharedMemAllocator fail");
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[New][SharedMemAllocator] failed!");
    return GE_GRAPH_MALLOC_FAILED;
  }
  return SUCCESS;
}

void HostMemManager::Finalize() noexcept {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &it : var_memory_base_map_) {
    if ((allocator_ == nullptr) || (allocator_->DeAllocate(it.second) != SUCCESS)) {
      GELOGW("Host %s mem release failed!", it.first.c_str());
    }
  }
  var_memory_base_map_.clear();
}

Status HostMemManager::MallocHostSharedMemory(SharedMemInfo &mem_info) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto iter = var_memory_base_map_.find(mem_info.op_name);
  if (iter != var_memory_base_map_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Host shared memory for op %s has been malloced", mem_info.op_name.c_str());
    GELOGE(FAILED, "[Check][Param] Host shared memory for op %s has been malloced", mem_info.op_name.c_str());
    return FAILED;
  }
  mem_info.shm_name = OpNameToShmName(mem_info.op_name);
  GE_CHECK_NOTNULL(allocator_);
  GE_CHK_STATUS_RET(allocator_->Allocate(mem_info));
  var_memory_base_map_[mem_info.op_name] = mem_info;
  return SUCCESS;
}

bool HostMemManager::QueryVarMemInfo(const std::string &op_name, SharedMemInfo &mem_info) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = var_memory_base_map_.find(op_name);
  if (it == var_memory_base_map_.end()) {
    GELOGW("Host memory for node [%s] not found.", op_name.c_str());
    return false;
  }
  mem_info = it->second;
  return true;
}

std::string HostMemManager::OpNameToShmName(const std::string &op_name) {
  std::string sh_name("Ascend_");
  return sh_name.append(std::to_string(std::hash<std::string>()(op_name)));
}
}  // namespace ge
