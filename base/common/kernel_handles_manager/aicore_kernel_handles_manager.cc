/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_kernel_handles_manager.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "ge/ge_api_error_codes.h"
#include "common/checker.h"
#include "kernel_handle_utils.h"

namespace ge {
aclrtBinHandle AicoreKernelHandlesManager::RegisterKernel(const KernelRegisterInfo &register_info,
    const std::string &bin_name) {
  GE_ASSERT_TRUE(!bin_name.empty(), "Bin handle name is empty.");
  auto *aicore_register_info = std::get_if<AicoreRegisterInfo>(&register_info);
  GE_ASSERT_NOTNULL(aicore_register_info);
  const int32_t magic = aicore_register_info->magic;
  const auto kernel_bin = aicore_register_info->kernel_bin;
  GE_ASSERT_NOTNULL(kernel_bin, "Aicore kernel bin is nullptr, bin_name: %s", bin_name.c_str());

  aclrtBinaryLoadOptions load_options;
  aclrtBinaryLoadOption option;
  load_options.numOpt = 1;
  load_options.options = &option;
  option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  option.value.magic = magic;
  aclrtBinHandle bin_handle;
  GE_ASSERT_RT_OK(aclrtBinaryLoadFromData(kernel_bin->GetBinData(), kernel_bin->GetBinDataSize(),
      &load_options, &bin_handle));

  StoredKernelHandle(bin_handle, bin_name);
  GELOGI("Aicore kernel register success, kernel bin_name: %s", bin_name.c_str());
  return bin_handle;
}

std::string AicoreKernelHandlesManager::GenerateKey(const KernelRegisterInfo &register_info) {
  auto *aicore_register_info = std::get_if<AicoreRegisterInfo>(&register_info);
  GE_ASSERT_NOTNULL(aicore_register_info);
  const std::string bin_name = aicore_register_info->kernel_bin_name + "_AicoreKernel";
  GELOGI("Aicore kernel generate bin_name: %s", bin_name.c_str());
  return bin_name;
}
}