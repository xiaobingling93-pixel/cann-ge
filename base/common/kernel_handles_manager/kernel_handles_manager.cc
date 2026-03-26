/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_handles_manager.h"
#include "common/checker.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
std::unordered_map<std::string, KernelBinInfo> KernelHandlesManager::global_bin_store_;
std::recursive_mutex KernelHandlesManager::mtx_;
aclrtBinHandle KernelHandlesManager::FindKernel(const std::string &bin_name) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  auto iter = global_bin_store_.find(bin_name);
  if (iter != global_bin_store_.end()) {
    GELOGI("Find kernel by bin_name: %s", bin_name.c_str());
    local_refer_cnt_[bin_name]++;
    iter->second.refer_cnt++;
    return iter->second.bin_handle;
  }
  return nullptr;
}

KernelHandlesManager::~KernelHandlesManager() {
  (void)ClearKernel();
}

graphStatus KernelHandlesManager::ClearKernel() {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  for (const auto &local_refer_iter : local_refer_cnt_) {
    auto bin_iter = global_bin_store_.find(local_refer_iter.first);
    GE_ASSERT_TRUE(bin_iter != global_bin_store_.end(),
        "Cannot find bin handle in global bin store, bin_handle_name: %s", local_refer_iter.first.c_str());
    GE_ASSERT_TRUE(bin_iter->second.refer_cnt >= local_refer_iter.second,
        "Global refer cnt: %ld should more than local refer cnt: %ld",
        bin_iter->second.refer_cnt, local_refer_iter.second);
    bin_iter->second.refer_cnt -= local_refer_iter.second;
    if (bin_iter->second.refer_cnt == 0) {
      GE_ASSERT_RT_OK(aclrtBinaryUnLoad(bin_iter->second.bin_handle));
      (void)global_bin_store_.erase(bin_iter);
    }
  }
  local_refer_cnt_.clear();
  return SUCCESS;
}

aclrtBinHandle KernelHandlesManager::GetOrRegisterKernel(const KernelRegisterInfo &register_info,
    const std::string &bin_name) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);
  auto bin_handle = FindKernel(bin_name);
  if (bin_handle != nullptr) {
    return bin_handle;
  }
  return RegisterKernel(register_info, bin_name);
}

void KernelHandlesManager::StoredKernelHandle(const aclrtBinHandle bin_handle, const std::string &bin_name) {
  KernelBinInfo kernel_bin_handle;
  kernel_bin_handle.bin_handle = bin_handle;
  kernel_bin_handle.refer_cnt = 1;
  local_refer_cnt_[bin_name] = 1;
  global_bin_store_.emplace(bin_name, kernel_bin_handle);
}
}