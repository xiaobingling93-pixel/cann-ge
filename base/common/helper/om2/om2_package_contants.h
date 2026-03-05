/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_OM2_CONTANTS_H
#define BASE_COMMON_HELPER_OM2_CONTANTS_H

#include <string>
#include <vector>
#include "securec.h"

namespace ge {
#define FORALL_OM2_CONSTANTS(DO)                                   \
  DO(OM2_ARCHIVE_VERSION, "om2_version");                          \
  DO(OM2_ARCHIVE_VERSION_VALUE, "0");                              \
  DO(OM2_MODEL_NUM, "model_num");                                  \
  DO(OM2_ATC_COMMAND, "atc_command");                              \
  DO(OM2_MANIFEST_PATH, "manifest.json");                          \
  DO(OM2_DATA_DIR, "data/");                                       \
  DO(OM2_MODEL_DIR_FORMAT, "data/model_%s/");                      \
  DO(OM2_MODEL_META_PATH_FORMAT, "data/model_%s/model_meta.json"); \
  DO(OM2_RUNTIME_DIR_FORMAT, "data/model_%s/runtime/");            \
  DO(OM2_KERNELS_DIR_FORMAT, "data/kernels_%s/");                  \
  DO(OM2_CONSTANTS_DIR, "data/constants/");                        \
  DO(OM2_CONSTANTS_FILE_PREFIX, "constant_");                      \
  DO(OM2_CONSTANTS_CONFIG_PATH_FORMAT, "data/constants/model_%s_constants_config.json";)

#define DEFINE_OM2_CONST(name, value) inline constexpr const char *name = value
FORALL_OM2_CONSTANTS(DEFINE_OM2_CONST)
#undef DEFINE_OM2_CONST

template <typename... Args>
std::string FormatOm2Path(const char *fmt, Args... args) {
  // 先计算所需长度（不包括终止符）
  const int32_t len = snprintf(nullptr, 0, fmt, args...);
  if (len < 0) {
    return "";
  }

  std::vector<char> buf(len + 1, '\0');
  if (snprintf_s(buf.data(), buf.size(), buf.size() - 1, fmt, args...) < 0) {
    return "";
  }

  return {buf.data()};
}
}  // namespace ge
#endif  // BASE_COMMON_HELPER_OM2_CONTANTS_H
