/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kernel_handles_manager.h"
#include <fstream>
#include <cstdio>
#include "mmpa/mmpa_api.h"
#include "ge/ge_api_error_codes.h"
#include "common/checker.h"
#include "common/ge_common/scope_guard.h"
#include "kernel_handle_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr uint32_t kMaxJsonFileLen = 512U;
graphStatus GenerateJsonFile(const KernelRegisterInfo &register_info, std::string &json_path) {
  json_path = "/tmp/temp_aicpu_ops_info_" + std::to_string(mmGetPid()) + "_" + std::to_string(mmGetTid()) + "_" +
      std::to_string(GetCurrentTimestamp()) + ".json";
  std::string json_data_format = R"(
{
    "%s":{
        "opInfo":{
            "opKernelLib":"%s",
            "kernelSo":"%s",
            "functionName":"%s"
        }
    }
}
)";
  auto *aicpu_register_info = std::get_if<AicpuRegisterInfo>(&register_info);
  GE_ASSERT_NOTNULL(aicpu_register_info);
  char json_data[kMaxJsonFileLen];
  std::string op_kernel_lib = aicpu_register_info->op_kernel_lib;
  std::string so_name = aicpu_register_info->so_name;
  std::string kernel_name = aicpu_register_info->kernel_name;
  std::string op_type = aicpu_register_info->op_type;
  auto ret = snprintf_s(json_data, kMaxJsonFileLen, kMaxJsonFileLen - 1U,
      json_data_format.c_str(), op_type.c_str(), op_kernel_lib.c_str(), so_name.c_str(), kernel_name.c_str());
  GE_ASSERT_TRUE(ret >= 0, "snprintf_s failed, ret: %d", ret);
  std::ofstream ofs(json_path.c_str(), std::ios::trunc);
  GE_ASSERT_TRUE(ofs, "Cannot open kernel json file: %s", json_path.c_str());
  ofs << json_data;
  GELOGD("Generate aicpu json data: ");
  GELOGD("%s", json_data);
  GELOGI("Aicpu json file path: %s", json_path.c_str());
  return SUCCESS;
}
}
aclrtBinHandle AicpuKernelHandlesManager::RegisterKernel(const KernelRegisterInfo &register_info,
    const std::string &bin_name) {
  GE_ASSERT_TRUE(!bin_name.empty(), "Bin handle name is empty.");
  std::string json_path;
  GE_ASSERT_SUCCESS(GenerateJsonFile(register_info, json_path));
  GE_MAKE_GUARD(json_guard, [&json_path]() {
    (void)std::remove(json_path.c_str());
  });
  GE_ASSERT_TRUE(!json_path.empty());
  aclrtBinaryLoadOptions load_options;
  aclrtBinaryLoadOption option;
  load_options.numOpt = 1;
  load_options.options = &option;
  option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
  constexpr const int32_t cpu_kernel_mode = 0;
  option.value.cpuKernelMode = cpu_kernel_mode;
  aclrtBinHandle bin_handle;
  GE_ASSERT_RT_OK(aclrtBinaryLoadFromFile(json_path.c_str(), &load_options, &bin_handle));

  StoredKernelHandle(bin_handle, bin_name);
  GELOGI("Aicpu kernel register success, kernel bin_name: %s", bin_name.c_str());
  return bin_handle;
}

std::string AicpuKernelHandlesManager::GenerateKey(const KernelRegisterInfo &register_info) {
  auto *aicpu_register_info = std::get_if<AicpuRegisterInfo>(&register_info);
  GE_ASSERT_NOTNULL(aicpu_register_info);
  const std::string bin_name = aicpu_register_info->op_type + "_" +
      aicpu_register_info->so_name + "_AicpuKernel";
  GELOGI("Aicpu kernel generate bin_name: %s", bin_name.c_str());
  return bin_name;
}
}