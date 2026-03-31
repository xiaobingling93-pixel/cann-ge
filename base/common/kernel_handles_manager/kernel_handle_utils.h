/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_KERBEL_HANDLE_UTILS_H
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_KERBEL_HANDLE_UTILS_H

#include <string>
#include "acl/acl_rt.h"
#include "kernel_handles_manager.h"

namespace ge {
struct LaunchKernelConfig {
  uint8_t schedule_mode{0U};
  uint32_t local_memory_size{0U};
  aclrtEngineType engine_type{ACL_RT_ENGINE_TYPE_AIC};
  uint32_t block_dim_offset{0U};
  bool is_block_task_prefetch{false};
  bool is_data_dump{false};
  int16_t time_out{-1};
};

using RefreshAddrInfo = aclrtPlaceHolderInfo;

struct LaunchKernelParam {
  uint32_t block_dim{0U};
  void *stream{nullptr};
  void *args{nullptr};
  uint32_t args_size{0U};
  LaunchKernelConfig launch_config;
  std::vector<RefreshAddrInfo> refresh_add_infos;
  bool is_host_args{false};
};

class KernelHandleUtils {
 public:
  static aclrtFuncHandle GetFuncHandle(const aclrtBinHandle &bin_handle, const std::string &kernel_name);
  static aclrtFuncHandle GetCustAicpuFuncHandle(const aclrtBinHandle &bin_handle,
      const std::string &op_type, const std::string &func_name);
  static aclrtFuncHandle GetFuncHandle(const aclrtBinHandle &bin_handle, const uint64_t &tiling_key);
  static graphStatus LaunchKernel(const aclrtFuncHandle func_handle, const LaunchKernelParam &launch_param);
};
}

#endif // #ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_KERBEL_HANDLE_UTILS_H