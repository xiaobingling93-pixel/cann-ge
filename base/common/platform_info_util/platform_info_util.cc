/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform_info_util.h"

#include "runtime/rt.h"
#include "common/debug/log.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "graph/ge_context.h"
#include "framework/common/helper/model_helper.h"

namespace ge {
namespace {
const std::string kJitCompileDefaultAuto = "2";
const std::string kJitCompileEnable = "1";
const std::string kHardwareInfo = "ge.hardwareInfo";
const std::string kMemorySizeName = "memory_size";
constexpr size_t kNameValueLen = 2U;
constexpr int32_t kStrToIntBase = 10;

// A set of NPU architecture IDs for which JIT compilation is enabled by default.
const std::set<std::string> kJitCompileEnabledByArch = {
    NPUARCH_TO_STR(NpuArch::DAV_1001), NPUARCH_TO_STR(NpuArch::DAV_2002), NPUARCH_TO_STR(NpuArch::DAV_2102),
    NPUARCH_TO_STR(NpuArch::DAV_3002), NPUARCH_TO_STR(NpuArch::DAV_3004), NPUARCH_TO_STR(NpuArch::DAV_3505),
    NPUARCH_TO_STR(NpuArch::DAV_3102), NPUARCH_TO_STR(NpuArch::DAV_5102),
};
}  // namespace

std::string PlatformInfoUtil::GetJitCompileDefaultValue() {
  std::string default_value = kJitCompileDefaultAuto;

  std::string npu_arch;
  if (GetSocSpec("version", "NpuArch", npu_arch) != SUCCESS) {
    GELOGW("Cannot get NpuArch, using default jit_compile = %s", default_value.c_str());
    return default_value;
  }

  const auto iter = kJitCompileEnabledByArch.find(npu_arch);
  if (iter != kJitCompileEnabledByArch.end()) {
    default_value = kJitCompileEnable;
  }

  GELOGD("Current NpuArch = %s, jit_compile = %s", npu_arch.c_str(), default_value.c_str());
  return default_value;
}

Status PlatformInfoUtil::GetSocSpec(const std::string &label, const std::string &key, std::string &value_str) {
  constexpr uint32_t kMaxValueLen = 16UL;
  char_t value[kMaxValueLen] = {0};

  // rtGetSocSpec will query the current SoC version
  const auto ret = rtGetSocSpec(label.c_str(), key.c_str(), value, kMaxValueLen);
  if (ret != RT_ERROR_NONE) {
    GELOGW("Cannot get value with label [%s] and key [%s], ret = %d", label.c_str(), key.c_str(), ret);
    value_str.clear();
    return FAILED;
  }

  value_str = std::string(value);
  return SUCCESS;
}

size_t PlatformInfoUtil::GetMemorySize() {
  std::string option_value;
  if (GetContext().GetOption(EVALUATE_GRAPH_RESOURCE_MODE, option_value) == GRAPH_SUCCESS) {
    // 1: graph resource evaluation
    GELOGI("EvaluateGraphResourceMode is %s", option_value.c_str());
    if (option_value == "1") {
      return std::numeric_limits<size_t>::max();
    }
  }

  size_t total_mem_size = 0U;
  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);

  std::string hard_ware_info_str;
  (void)ge::GetContext().GetOption(kHardwareInfo, hard_ware_info_str);
  GELOGI("Get from %s is %s.", kHardwareInfo.c_str(), hard_ware_info_str.c_str());

  std::vector<std::string> hard_ware_infos = StringUtils::Split(hard_ware_info_str, ';');
  for (const auto &hard_ware_info : hard_ware_infos) {
    std::vector<std::string> name_value = StringUtils::Split(hard_ware_info, ':');
    if ((name_value.size() == kNameValueLen) && (name_value[0] == kMemorySizeName)) {
      errno = 0;
      total_mem_size = static_cast<size_t>(std::strtoll(name_value[1].c_str(), nullptr, kStrToIntBase));
      GE_ASSERT_TRUE(errno == 0, "strtoll failed, value: %s", name_value[1].c_str());
      GELOGI("Get from %s platform %s memory size is %zu.", kHardwareInfo.c_str(), soc_version.c_str(), total_mem_size);
      break;
    }
  }

  if (total_mem_size == 0U) {
    fe::PlatformInfo plat_form_info;
    fe::OptionalInfo optional_info;
    plat_form_info.soc_info.memory_size = 0U;
    if (fe::PlatformInfoManager::GeInstance().GetPlatformInfo(soc_version, plat_form_info, optional_info) == 0U) {
      total_mem_size = static_cast<size_t>(plat_form_info.soc_info.memory_size);
    }
    GELOGI("Get from PlatformInfo platform %s memory size is %zu.", soc_version.c_str(), total_mem_size);
  }

  if (total_mem_size == 0U) {
    total_mem_size = std::numeric_limits<size_t>::max();
  }
  GELOGI("Final platform %s memory size is %zu.", soc_version.c_str(), total_mem_size);
  return total_mem_size;
}
} // namespace ge
