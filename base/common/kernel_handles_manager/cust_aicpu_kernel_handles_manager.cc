/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cust_aicpu_kernel_handles_manager.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "common/checker.h"
#include "ge/ge_api_error_codes.h"
#include "kernel_handle_utils.h"

namespace ge {
aclrtBinHandle CustAicpuKernelHandlesManager::RegisterKernel(const KernelRegisterInfo &register_info,
    const std::string &bin_name) {
  GE_ASSERT_TRUE(!bin_name.empty(), "Bin handle name is empty.");
  auto *cust_aicpu_register_info = std::get_if<CustAicpuRegisterInfo>(&register_info);
  GE_ASSERT_NOTNULL(cust_aicpu_register_info);
  const auto cust_aicpu_kernel_bin = cust_aicpu_register_info->cust_aicpu_kernel_bin;
  GE_ASSERT_NOTNULL(cust_aicpu_kernel_bin, "Cust aicpu kernel bin is nullptr.");

  aclrtBinaryLoadOptions load_options;
  aclrtBinaryLoadOption option;
  load_options.numOpt = 1;
  load_options.options = &option;
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  constexpr const int32_t cust_cpu_kernel_mode = 2;
  option.value.cpuKernelMode = cust_cpu_kernel_mode;
  aclrtBinHandle bin_handle;
  GE_ASSERT_RT_OK(aclrtBinaryLoadFromData(static_cast<const void *>(cust_aicpu_kernel_bin->GetBinData()),
      cust_aicpu_kernel_bin->GetBinDataSize(), &load_options, &bin_handle));

  StoredKernelHandle(bin_handle, bin_name);
  GELOGI("Cust aicpu kernel register success, kernel bin_name: %s", bin_name.c_str());
  return bin_handle;
}

std::string CustAicpuKernelHandlesManager::GenerateKey(const KernelRegisterInfo &register_info) {
  auto *cust_aicpu_register_info = std::get_if<CustAicpuRegisterInfo>(&register_info);
  GE_ASSERT_NOTNULL(cust_aicpu_register_info);
  const auto cust_aicpu_kernel_bin = cust_aicpu_register_info->cust_aicpu_kernel_bin;
  GE_ASSERT_NOTNULL(cust_aicpu_kernel_bin, "Cust aicpu kernel bin is nullptr.");
  // 使用二进制做hash
  const size_t hash_id =
      std::hash<std::string>{}(std::string(cust_aicpu_kernel_bin->GetBinData(),
      cust_aicpu_kernel_bin->GetBinData() + cust_aicpu_kernel_bin->GetBinDataSize()));
  const std::string bin_name = std::to_string(hash_id) + "_CustAicpuKernel";
  GELOGI("Cust aicpu kernel generate bin_name: %s", bin_name.c_str());
  return bin_name;
}
}