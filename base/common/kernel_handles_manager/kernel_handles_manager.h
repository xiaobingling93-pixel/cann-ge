/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_KERNEL_HANDLES_MANAGER_KERBEL_HANDLES_MANAGER_H
#define BASE_COMMON_KERNEL_HANDLES_MANAGER_KERBEL_HANDLES_MANAGER_H

#include <mutex>
#include <variant>
#include <unordered_map>
#include "acl/acl_rt.h"
#include "graph/ge_error_codes.h"
#include "common/tbe_handle_store/kernel_store.h"

namespace ge {
struct KernelBinInfo {
  aclrtBinHandle bin_handle{nullptr};
  int64_t refer_cnt{0};
};

struct AicoreRegisterInfo {
  std::string kernel_bin_name;
  int32_t magic{0};
  KernelBinPtr kernel_bin{nullptr};
};

struct AicpuRegisterInfo {
  std::string op_type;
  std::string so_name;
  std::string kernel_name;
  std::string op_kernel_lib;
};

struct CustAicpuRegisterInfo {
  KernelBinPtr cust_aicpu_kernel_bin{nullptr};
};
using KernelRegisterInfo = std::variant<AicoreRegisterInfo, AicpuRegisterInfo, CustAicpuRegisterInfo>;
class KernelHandlesManager {
 public:
  KernelHandlesManager() = default;
  virtual ~KernelHandlesManager();
  virtual std::string GenerateKey(const KernelRegisterInfo &register_info) = 0;
  aclrtBinHandle GetOrRegisterKernel(const KernelRegisterInfo &register_info,
      const std::string &bin_name);
  graphStatus ClearKernel();
  aclrtBinHandle FindKernel(const std::string &bin_name);
 protected:
  virtual aclrtBinHandle RegisterKernel(const KernelRegisterInfo &register_info,
      const std::string &bin_name) = 0;
  void StoredKernelHandle(const aclrtBinHandle bin_handle, const std::string &bin_name);
  std::unordered_map<std::string, int64_t> local_refer_cnt_;
  static std::unordered_map<std::string, KernelBinInfo> global_bin_store_;
  static std::recursive_mutex mtx_;
};
} // namespace ge

#endif // BASE_COMMON_KERNEL_HANDLES_MANAGER_KERBEL_HANDLES_MANAGER_H