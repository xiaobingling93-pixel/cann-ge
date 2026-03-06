/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/allocator/scalable_allocator.h"
#include "runtime/dev.h"
#include "common/debug/log.h"
#include "graph/ge_local_context.h"
#include "framework/common/util.h"

namespace gert {
namespace {
constexpr MemSize kDeviceTotalMemorySizeLevel[MEMORY_SPECIFICATION_LEVEL_MAX] = {32_GB, 64_GB};

ge::Status GetDeviceTotalMemorySize(size_t &total_mem_size) {
  size_t free_mem;
  GE_CHK_RT_RET(rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free_mem, &total_mem_size));
  if (total_mem_size == 0U) {
    GE_CHK_RT_RET(rtMemGetInfoEx(RT_MEMORYINFO_DDR, &free_mem, &total_mem_size));
  }
  (void)free_mem;
  return ge::SUCCESS;
}
}

ScalableConfig::ScalableConfig() {
  size_t total_mem_size = 0U;
  const auto ret = GetDeviceTotalMemorySize(total_mem_size);
  if (ret != ge::SUCCESS) {
    GELOGW("Get device total memory size failed.");
  }

  for (size_t i = 0U; i < MEMORY_SPECIFICATION_LEVEL_MAX - 1U; ++i) {
    if (total_mem_size > kDeviceTotalMemorySizeLevel[i]) {
      page_mem_size_total_threshold = PAGE_MEM_SIZE_THRESHOLD_DEFAULT[i + 1U];
      uncacheable_size_threshold = SPAN_UNCACHEABLE_MEM_SIZE_DEFAULT[i + 1U];
    }
  }
  if (total_mem_size > 0U) {
    page_mem_size_total_threshold =
        static_cast<size_t>(floor(static_cast<float64_t>(total_mem_size) * kMaxMemorySizeRatio));
  }

  constexpr const char *kOptionMemoryPoolThreshold = "ge.experiment.memory_pool_threshold";
  std::string option_value;
  if (ge::GetThreadLocalContext().GetOption(kOptionMemoryPoolThreshold, option_value) == ge::GRAPH_SUCCESS) {
    GELOGI("option[%s] value[%s]", kOptionMemoryPoolThreshold, option_value.c_str());
    if (!option_value.empty()) {
      int32_t int_value = 0;
      if (ge::ConvertToInt32(option_value, int_value) != ge::SUCCESS) {
        GELOGW("Convert option[%s]=[%s] to int32 failed.", kOptionMemoryPoolThreshold, option_value.c_str());
      } else {
        int64_t config_value = int_value * static_cast<int64_t>(MEM_SIZE_GB);
        if ((config_value <= 0) || (config_value > static_cast<int64_t>(page_mem_size_total_threshold))) {
          GELOGW("option[%s] config value=%s(GB) is invalid, range is (0, %lu]", kOptionMemoryPoolThreshold,
                 option_value.c_str(), page_mem_size_total_threshold / MEM_SIZE_GB);
        } else {
          page_mem_size_total_threshold = static_cast<uint64_t>(config_value);
          GELOGI("page_mem_size_total_threshold is set to %lu, as option[%s]=%s(GB)", page_mem_size_total_threshold,
                  kOptionMemoryPoolThreshold, option_value.c_str());
        }
      }
    }
  }
  static bool printed = false;
  if (!printed) {
    printed = true;
    GEEVENT("device total max size: %zu, page_mem_size_total_threshold: %lu, uncacheable_size_threshold: %lu",
            total_mem_size, page_mem_size_total_threshold, uncacheable_size_threshold);
  }
}
}
