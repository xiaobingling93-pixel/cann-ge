/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_UTILS_H_
#define COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_UTILS_H_
#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>
#include "common/checker.h"

namespace ge {
// 解析结果结构体
struct ForceTilingCaseResult {
  bool is_single_mode{true};              // true表示所有组选择相同case
  int32_t single_case{-1};                // 统一选择的case编号（is_single_mode=true时有效）
  std::string single_sub_tag;             // 统一选择的sub_tag
  std::map<size_t, std::pair<int32_t, std::string>> group_cases;  // 各组的独立选择（is_single_mode=false时有效）
  [[nodiscard]] std::string Debug() const;
  [[nodiscard]] std::pair<int32_t, std::string> GetCase(size_t group_id) const;
  std::string GetTag(size_t group_id) const;
  void Clear();
};

class AttStrategyConfigUtils {
 public:
  static ge::Status ParseForceTilingCase(const std::string &input, ForceTilingCaseResult &force_tiling_case);
};

}  // namespace ge

#endif  // COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_UTILS_H_
