/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/tensor_utils.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <mutex>

#include "framework/common/debug/ge_log.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

namespace ge {
namespace {
// SoC specification query key constants for padding
constexpr const char *kSocSpecModuleAICore = "AICoreSpec";
constexpr const char *kSocSpecKeyPaddingSize = "padding_size";
// Default alignment values
constexpr int64_t kDefaultPaddingSize = 32;
constexpr int64_t kDataMemAlignSize = 32;

typedef int32_t rtError_t;
constexpr rtError_t RT_ERROR_NONE = 0;

using RtGetSocSpecFunc = rtError_t (*)(const char*, const char*, char*, const uint32_t);

std::string GetRuntimeLibPath() {
  std::string lib_dir = ge::GetModelPath();
  if (lib_dir.empty()) {
    GELOGW("[Runtime][Path] Failed to get lib directory path");
    return "";
  }
  return lib_dir + "libruntime.so";
}

RtGetSocSpecFunc GetRtGetSocSpecFunc() {
  static std::once_flag load_flag;
  static RtGetSocSpecFunc func = nullptr;
  static void* runtime_handle = nullptr;

  std::call_once(load_flag, []() {
    std::string runtime_path = GetRuntimeLibPath();
    if (runtime_path.empty()) {
      GELOGW("[Runtime][Load] Runtime lib path is empty, %s", runtime_path.c_str());
      return;
    }

    GELOGI("[Runtime][Load] Loading runtime from: %s", runtime_path.c_str());
    runtime_handle = mmDlopen(runtime_path.c_str(), MMPA_RTLD_NOW);
    if (runtime_handle == nullptr) {
      const char_t* error = mmDlerror();
      GELOGW("[Runtime][Load] mmDlopen failed, path=%s, error=%s",
             runtime_path.c_str(), error ? error : "");
      return;
    }

    func = reinterpret_cast<RtGetSocSpecFunc>(mmDlsym(runtime_handle, "rtGetSocSpec"));
    if (func == nullptr) {
      const char_t* error = mmDlerror();
      GELOGW("[Runtime][Symbol] mmDlsym rtGetSocSpec failed, error=%s",
             error ? error : "");
      return;
    }

    GELOGI("[Runtime][Load] Successfully loaded rtGetSocSpec");
  });

  return func;
}

bool QueryPaddingSizeFromSocSpec(int64_t &padding_size) {
  auto rt_get_soc_spec = GetRtGetSocSpecFunc();
  if (rt_get_soc_spec == nullptr) {
    GELOGW("[Query][PaddingSize] rtGetSocSpec function not available");
    return false;
  }

  constexpr uint32_t kMaxValueLen = 32U;
  char padding_size_str[kMaxValueLen] = {0};
  const rtError_t ret = rt_get_soc_spec(kSocSpecModuleAICore, kSocSpecKeyPaddingSize,
                                         padding_size_str, kMaxValueLen);
  if (ret != RT_ERROR_NONE) {
    GELOGW("[Query][PaddingSize] rtGetSocSpec failed, label=%s, key=%s, ret=0x%X",
           kSocSpecModuleAICore, kSocSpecKeyPaddingSize, ret);
    return false;
  }

  GELOGI("[Query][PaddingSize] rtGetSocSpec success, label=%s, key=%s, value=%s",
         kSocSpecModuleAICore, kSocSpecKeyPaddingSize, padding_size_str);

  if (padding_size_str[0] == '-') {
    GELOGW("[Parse][PaddingSize] Got negative value: %s", padding_size_str);
    return false;
  }
  char *end_ptr = nullptr;
  errno = 0;
  padding_size = static_cast<int64_t>(strtoll(padding_size_str, &end_ptr, 10));
  if (errno == ERANGE || end_ptr == padding_size_str || *end_ptr != '\0') {
    GELOGW("[Parse][PaddingSize] Invalid value string: %s", padding_size_str);
    return false;
  }

  GELOGI("[Query][PaddingSize] Parsed padding_size=%" PRId64, padding_size);
  return true;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t TensorUtils::GetPaddingSize() {
  static std::once_flag g_padding_size_flag;
  static int64_t g_cached_padding_size = 0;

  std::call_once(g_padding_size_flag, []() {
    int64_t padding_size = kDefaultPaddingSize;
    if (!QueryPaddingSizeFromSocSpec(padding_size)) {
      GELOGW("[Query][PaddingSize] Use default value: %" PRId64, kDefaultPaddingSize);
      padding_size = kDefaultPaddingSize;
    }
    g_cached_padding_size = padding_size;
    GELOGI("[Query][PaddingSize] Final cached padding_size=%" PRId64, g_cached_padding_size);
  });
  return g_cached_padding_size;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetTensorMemorySizeInBytesWithAutoPadding(const GeTensorDesc &desc_temp, int64_t &size_temp) {
  const graphStatus graph_status = GetTensorSizeInBytes(desc_temp, size_temp);
  if (graph_status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  const int64_t padding_size = GetPaddingSize();
  const int64_t append_size = kDataMemAlignSize + padding_size;
  if (size_temp > (std::numeric_limits<int64_t>::max() - append_size)) {
    GELOGW("[Util][CalcBytesSize] Mem size %" PRId64 " after alignment is bigger than INT64_MAX", size_temp);
  } else {
    size_temp = ((size_temp + append_size - 1) / kDataMemAlignSize) * kDataMemAlignSize;
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge