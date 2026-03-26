/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_handle_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
void LogLaunchKernelParam(const LaunchKernelParam &launch_param) {
  GELOGI("LaunchKernelParam info: block_dim = %u, args_size = %u, is_host_args = %d",
      launch_param.block_dim, launch_param.args_size, launch_param.is_host_args);
  GELOGI("LaunchKernelConfig info: schedule_mode = %u, local_memory_size = %u, " 
      "engine_type = %d, block_dim_offset = %u, is_block_task_prefetch = %d, " 
      "is_data_dump = %d, time_out = %u",
      static_cast<uint32_t>(launch_param.launch_config.schedule_mode),
      launch_param.launch_config.local_memory_size,
      static_cast<int32_t>(launch_param.launch_config.engine_type),
      launch_param.launch_config.block_dim_offset,
      static_cast<int32_t>(launch_param.launch_config.is_block_task_prefetch),
      static_cast<int32_t>(launch_param.launch_config.is_data_dump),
      launch_param.launch_config.time_out);
}
}
aclrtFuncHandle KernelHandleUtils::GetFuncHandle(const aclrtBinHandle &bin_handle, const std::string &kernel_name) {
  GE_ASSERT_NOTNULL(bin_handle);
  aclrtFuncHandle func_handle;
  GE_ASSERT_RT_OK(aclrtBinaryGetFunction(bin_handle, kernel_name.c_str(), &func_handle),
      "Get func handle of node: %s failed.", kernel_name.c_str());
  GELOGI("Get func from bin handle by kernel name: %s.", kernel_name.c_str());
  return func_handle;
}
aclrtFuncHandle KernelHandleUtils::GetFuncHandle(const aclrtBinHandle &bin_handle, const uint64_t &tiling_key) {
  GE_ASSERT_NOTNULL(bin_handle);
  aclrtFuncHandle func_handle;
  GE_ASSERT_RT_OK(aclrtBinaryGetFunctionByEntry(bin_handle, tiling_key, &func_handle),
      "Get func handle by entry: %lu failed.", tiling_key);
  GELOGI("Get func from bin handle by entry: %llu.", tiling_key);
  return func_handle;
}

aclrtFuncHandle KernelHandleUtils::GetCustAicpuFuncHandle(const aclrtBinHandle &bin_handle,
    const std::string &op_type, const std::string &func_name) {
  GE_ASSERT_NOTNULL(bin_handle);
  aclrtFuncHandle func_handle;
  GE_ASSERT_RT_OK(aclrtRegisterCpuFunc(bin_handle, func_name.c_str(),
      op_type.c_str(), &func_handle), "Get func handle by kernel name:%s and func name: %s failed",
      op_type.c_str(), func_name.c_str());
  GELOGI("Get func from bin handle by op type: %s, func name: %s.", op_type.c_str(), func_name.c_str());
  return func_handle;
}

graphStatus KernelHandleUtils::LaunchKernel(const aclrtFuncHandle func_handle, const LaunchKernelParam &launch_param) {
  LogLaunchKernelParam(launch_param);
  aclrtLaunchKernelCfg rt_launch_config;
  constexpr const size_t max_launch_cfg_num = 8UL;
  aclrtLaunchKernelAttr attrs[max_launch_cfg_num];
  size_t actual_cfg_num = 0UL;
  attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attrs[actual_cfg_num].value.schemMode = launch_param.launch_config.schedule_mode;
  actual_cfg_num++;
  attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attrs[actual_cfg_num].value.engineType = launch_param.launch_config.engine_type;
  actual_cfg_num++;
  attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET;
  attrs[actual_cfg_num].value.blockDimOffset = launch_param.launch_config.block_dim_offset;
  actual_cfg_num++;
  attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH;
  attrs[actual_cfg_num].value.isBlockTaskPrefetch =
      static_cast<uint8_t>(launch_param.launch_config.is_block_task_prefetch);
  actual_cfg_num++;
  attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP;
  attrs[actual_cfg_num].value.isDataDump = static_cast<uint8_t>(launch_param.launch_config.is_data_dump);
  actual_cfg_num++;
  attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attrs[actual_cfg_num].value.dynUBufSize = launch_param.launch_config.local_memory_size;
  actual_cfg_num++;
  if (launch_param.launch_config.time_out >= 0) {
    attrs[actual_cfg_num].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attrs[actual_cfg_num].value.timeout = static_cast<uint16_t>(launch_param.launch_config.time_out);
    actual_cfg_num++;
  }
  rt_launch_config.attrs = &attrs[0];
  rt_launch_config.numAttrs = actual_cfg_num;
  GE_ASSERT_NOTNULL(func_handle);
  if (launch_param.is_host_args) {
    GE_ASSERT_RT_OK(aclrtLaunchKernelWithHostArgs(func_handle, launch_param.block_dim, launch_param.stream,
        &rt_launch_config, launch_param.args, launch_param.args_size,
        const_cast<RefreshAddrInfo *>(launch_param.refresh_add_infos.data()),
        launch_param.refresh_add_infos.size()));
  } else {
    GE_ASSERT_RT_OK(aclrtLaunchKernelV2(func_handle, launch_param.block_dim, launch_param.args,
        launch_param.args_size, &rt_launch_config, launch_param.stream));
  }
  return SUCCESS;
}
}