/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/deployer_var_manager.h"
#include <fstream>
#include <thread>
#include "common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/util.h"
#include "common/config/configurations.h"
#include "securec.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"
#include "common/data_flow/event/proxy_event_manager.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr int32_t kCoreNumPerDevice = 8;
constexpr size_t kMaxSharedContentSize = 1024 * 1024U;
constexpr size_t kAlignSize = 512U;
constexpr size_t kAlignUnit = 2U;
} // namespace

DeployerVarManager::~DeployerVarManager() {
}

Status DeployerVarManager::Initialize(deployer::VarManagerInfo var_manager_info) {
  var_mem_size_ = var_manager_info.use_max_mem_size();
  var_manager_info_ = std::move(var_manager_info);
  GELOGI("VarManager initialized, session_id = %lu, device_id = %d, size = %lu",
         var_manager_info_.session_id(), var_manager_info_.device_id(), var_mem_size_);
  return SUCCESS;
}

void DeployerVarManager::Finalize() {
  for (auto &it_mbuf : var_mbuf_vec_) {
    if (it_mbuf != nullptr) {
      (void) ProxyEventManager::FreeMbuf(var_manager_info_.device_id(), it_mbuf);
      it_mbuf = nullptr;
    }
  }
}

Status DeployerVarManager::ProcessSharedContent(const deployer::SharedContentDescription &shared_content_desc,
                                                const size_t size,
                                                const size_t offset,
                                                const uint32_t queue_id) {
  uint64_t total_length = shared_content_desc.total_length();
  GE_CHECK_LE(shared_content_desc.current_offset(), UINT64_MAX - offset);
  uint64_t current_offset = shared_content_desc.current_offset() + offset;
  const auto &node_name = shared_content_desc.node_name();
  GELOGD("Begin to copy shared content to memory, var is %s, offset = %lu", node_name.c_str(), current_offset);

  void *var_dev_addr = nullptr;
  GE_CHECK_LE(shared_content_desc.current_offset(), UINT64_MAX - var_manager_info_.var_mem_logic_base());
  uint64_t logic_addr = shared_content_desc.current_offset() + var_manager_info_.var_mem_logic_base();
  GE_CHK_STATUS_RET(GetVarMemAddr(logic_addr, total_length, &var_dev_addr),
      "[GetVarMemAddr]Get or Malloc shared memory failed");
  var_dev_addr = static_cast<void *>(static_cast<uint8_t *>(var_dev_addr) + offset + kAlignSize);
  GELOGD("copy from queue id[%d] to device[%d]", queue_id, var_manager_info_.device_id());
  GE_CHK_STATUS_RET(
      ProxyEventManager::CopyQMbuf(var_manager_info_.device_id(), PtrToValue(var_dev_addr), size, queue_id),
      "Failed to copy q mbuf to device.");

  auto current_size = offset + size;
  if (current_size >= total_length) {
    GE_CHECK_LE(shared_content_descs_.size() + 1U, kMaxSharedContentSize);
    shared_content_descs_[node_name] = shared_content_desc;
    GELOGI("shared content [%s] received complete, size is %lu.", node_name.c_str(), total_length);
  } else {
    GELOGD("shared content [%s] received. %lu/%lu", node_name.c_str(), current_size, total_length);
  }
  return SUCCESS;
}

void DeployerVarManager::SetVarManagerInfo(deployer::VarManagerInfo var_manager_info) {
  var_manager_info_ = std::move(var_manager_info);
}

const deployer::VarManagerInfo &DeployerVarManager::GetVarManagerInfo() const {
  return var_manager_info_;
}

deployer::VarManagerInfo &DeployerVarManager::MutableVarManagerInfo() {
  return var_manager_info_;
}

Status DeployerVarManager::GetVarMemAddr(const uint64_t &offset,
                                         const uint64_t &total_length,
                                         void **dev_addr,
                                         bool need_malloc) {
  auto iter = offset_and_var_map_.find(offset);
  if (iter != offset_and_var_map_.end()) {
    *dev_addr = iter->second;
    return SUCCESS;
  }

  if (!need_malloc) {
    return SUCCESS;
  }
  // 如果在Var内存缓存表里找不到logic_addr对应的内存，则分配新内存
  uint64_t malloc_length = 0UL;
  GE_CHECK_LE(total_length, UINT64_MAX - kAlignSize);
  // 分配内存512字节对齐
  malloc_length = (total_length + kAlignSize - 1U) / kAlignSize * kAlignSize;
  GE_CHECK_LE(malloc_length, UINT64_MAX - kAlignSize * kAlignUnit);
  // 分配内存前后各预留512K
  malloc_length += (kAlignSize * kAlignUnit);
  // do not set var max size, var addr individual malloc
  GE_CHK_STATUS_RET(MallocVarMem(malloc_length, var_manager_info_.device_id(), dev_addr),
                    "[Malloc] Var mem failed.");
  offset_and_var_map_.emplace(std::make_pair(offset, *dev_addr));
  return SUCCESS;
}

uint8_t *DeployerVarManager::GetVarMemBase() {
  if (var_mem_base_ == nullptr) {
    void *buffer_address = nullptr;
    if (MallocVarMem(var_manager_info_.var_mem_max_size(), var_manager_info_.device_id(),
                     &buffer_address) != SUCCESS) {
      return nullptr;
    }
    var_mem_base_ = static_cast<uint8_t *>(buffer_address);
  }
  return var_mem_base_;
}

const std::map<std::string, deployer::SharedContentDescription> &DeployerVarManager::GetSharedContentDescs() const {
  return shared_content_descs_;
}

uint64_t DeployerVarManager::GetVarMemSize() const {
  return var_mem_size_;
}

Status DeployerVarManager::MallocVarMem(const uint64_t var_size, const uint32_t device_id, void **dev_addr) {
  GE_CHK_STATUS_RET(ProxyEventManager::AllocMbuf(device_id, var_size, &var_mbuf_, dev_addr),
                    "Failed to alloc mbuf in device, device_id = %u.", device_id);
  var_mbuf_vec_.push_back(var_mbuf_);
  GEEVENT("Alloc var memory successfully, device_id = %u, size = %lu", device_id, var_size);
  return SUCCESS;
}

void DeployerVarManager::SetBasePath(const std::string &base_path) {
  base_path_ = base_path;
}

bool DeployerVarManager::IsShareVarMem() const {
  return share_var_mem_;
}

void DeployerVarManager::SetShareVarMem(bool share_var_mem) {
  share_var_mem_ = share_var_mem;
}
}  // namespace ge
