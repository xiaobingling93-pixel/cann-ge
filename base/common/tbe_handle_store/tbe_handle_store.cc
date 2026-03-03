/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_handle_store/tbe_handle_store.h"

#include <limits>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/log.h"
#include "runtime/kernel.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "base/err_msg.h"

namespace ge {
TbeHandleInfo::~TbeHandleInfo() {
  if (handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle_));
  }
  handle_ = nullptr;
}

void TbeHandleInfo::used_inc(const uint32_t num) {
  if (used_ > (std::numeric_limits<uint32_t>::max() - num)) {
    REPORT_INNER_ERR_MSG("E19999", "Used:%u reach numeric max", used_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Used[%u] reach numeric max.", used_);
    return;
  }

  used_ += num;
}

void TbeHandleInfo::used_dec(const uint32_t num) {
  if (used_ < (std::numeric_limits<uint32_t>::min() + num)) {
    REPORT_INNER_ERR_MSG("E19999", "Used:%u reach numeric min", used_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Used[%u] reach numeric min.", used_);
    return;
  }

  used_ -= num;
}

uint32_t TbeHandleInfo::used_num() const {
  return used_;
}

void *TbeHandleInfo::handle() const {
  return handle_;
}

std::recursive_mutex TBEHandleStore::mutex_;

TBEHandleStore &TBEHandleStore::GetInstance() {
  static TBEHandleStore instance;

  return instance;
}

///
/// @ingroup ge
/// @brief Find Registered TBE handle by name.
/// @param [in] name: TBE handle name to find.
/// @param [out] handle: handle names record.
/// @return true: found / false: not found.
///
bool TBEHandleStore::FindTBEHandle(const std::string &name, void *&handle) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = bin_key_to_handle_.find(name);
  if (it == bin_key_to_handle_.end()) {
    return false;
  } else {
    handle = it->second->handle();
    return true;
  }
}

///
/// @ingroup ge
/// @brief Store registered TBE handle info.
/// @param [in] name: TBE handle name to store.
/// @param [in] handle: TBE handle addr to store.
/// @param [in] kernel: TBE kernel bin to store.
/// @return NA
///
Status TBEHandleStore::StoreTBEHandle(const std::string &name, void *handle, const OpKernelBinPtr &kernel) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = bin_key_to_handle_.find(name);
  if (it == bin_key_to_handle_.end()) {
    std::unique_ptr<TbeHandleInfo> handle_info = ge::MakeUnique<TbeHandleInfo>(handle, kernel);
    GE_ASSERT_NOTNULL(handle_info);
    handle_info->used_inc();
    (void)bin_key_to_handle_.emplace(name, std::move(handle_info));
  } else {
    auto &handle_info = it->second;
    handle_info->used_inc();
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief get unique id in handle map
/// @param [in] handle: TBE handle addr to store.
/// @param [in] kernel: kernel name to store.
/// @param [inserted] kernel: inserted flag.
/// @return the addr of unique id
///
void *TBEHandleStore::GetUniqueIdPtr(void *const handle, const std::string &kernel, bool &inserted) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::unordered_map<std::string, std::list<uint8_t>>& inner = handle_to_kernel_to_unique_id_[handle];
  std::pair<std::unordered_map<std::string, std::list<uint8_t>>::iterator, bool> ret =
    inner.insert(std::make_pair(kernel, std::list<uint8_t>{0}));
  inserted = ret.second;
  return &(ret.first->second.back());
}

///
/// @ingroup ge
/// @brief Increase reference of registered TBE handle info.
/// @param [in] name: handle name increase reference.
/// @return NA
///
void TBEHandleStore::ReferTBEHandle(const std::string &name) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = bin_key_to_handle_.find(name);
  if (it == bin_key_to_handle_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Kernel:%s not found in stored check invalid", name.c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Kernel[%s] not found in stored.", name.c_str());
    return;
  }
  it->second->used_inc();
}

///
/// @ingroup ge
/// @brief Erase TBE registered handle record.
/// @param [in] names: handle names erase.
/// @return NA
///
void TBEHandleStore::EraseTBEHandle(const std::map<std::string, uint32_t> &names) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &item : names) {
    const auto it = bin_key_to_handle_.find(item.first);
    if (it == bin_key_to_handle_.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Kernel:%s not found in stored check invalid", item.first.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Kernel[%s] not found in stored.", item.first.c_str());
      continue;
    }

    auto &info = it->second;
    if (info->used_num() > item.second) {
      info->used_dec(item.second);
    } else {
      (void)handle_to_kernel_to_unique_id_.erase(info->handle());
      (void)bin_key_to_handle_.erase(it);
    }
  }
}

KernelHolder::KernelHolder(const char_t *const stub_func,
                           const std::shared_ptr<ge::OpKernelBin> &kernel_bin)
    : stub_func_(stub_func), kernel_bin_(kernel_bin) {}

KernelHolder::~KernelHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

HandleHolder::HandleHolder(void *const bin_handle)
    : bin_handle_(bin_handle) {}

HandleHolder::~HandleHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

const char_t *KernelBinRegistry::GetUnique(const std::string &stub_func) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto it = unique_stubs_.find(stub_func);
  if (it != unique_stubs_.end()) {
    return it->c_str();
  } else {
    it = unique_stubs_.insert(unique_stubs_.cend(), stub_func);
    return it->c_str();
  }
}

const char_t *KernelBinRegistry::GetStubFunc(const std::string &stub_name) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = registered_bins_.find(stub_name);
  if (iter != registered_bins_.end()) {
    return iter->second->stub_func_;
  }

  return nullptr;
}

bool KernelBinRegistry::AddKernel(const std::string &stub_name, std::unique_ptr<KernelHolder> &&holder) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto ret = registered_bins_.emplace(stub_name, std::move(holder));
  return ret.second;
}

bool HandleRegistry::AddHandle(std::unique_ptr<HandleHolder> &&holder) {
  const auto ret = registered_handles_.emplace(std::move(holder));
  return ret.second;
}
} // namespace ge
