/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PROFILING_TEST_UTIL_H_
#define GE_PROFILING_TEST_UTIL_H_

#include <functional>
#include <iostream>
#include <utility>
#include <map>
#include <vector>
#include <cstring>
#include "runtime/base.h"

namespace ge {
enum InfoType {
  kApi = static_cast<size_t>(MsprofReporterCallbackType::MSPROF_REPORTER_HASH + 1),
  kEvent,
  kCompactInfo,
  kInfo,
  kEnd
};
class ProfilingTestUtil {
 public:
  using ProfFunc = std::function<int32_t(uint32_t, uint32_t, void *, uint32_t)>;
  using HashFunc = std::function<uint64_t(const char *, size_t)>;
  static ProfilingTestUtil &Instance() {
    static ProfilingTestUtil profiling_test_util;
    return profiling_test_util;
  }

  void SetProfFunc(ProfFunc func) {
    hash_count_ = 0UL;
    func_ = func;
  }

  int32_t RunProfFunc(uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
    if (func_ == nullptr) {
      return 0;
    }
    auto ret = (func_)(moduleId, type, data, len);
    return ret;
  }

  void Clear() {
    func_ = nullptr;
    reg_types_.clear();
  }

  size_t &GetHashCount() {
    return hash_count_;
  }

  void RegisterType(uint16_t level, uint32_t typeId, const char *typeName) {
    (void) typeName;
    auto &reg_type = reg_types_[level];
    for (auto &ele : reg_type) {
      if (ele.first == typeId) {
        ele.second = 1;
        return;
      }
    }
    reg_type.emplace_back(std::pair<uint32_t, uint32_t>{typeId, 1});
  }

  uint32_t GetRegisterType(uint16_t level, uint32_t type_id) {
    auto &reg_type = reg_types_[level];
    for (auto &ele : reg_type) {
      if (ele.first == type_id) {
        return ele.second;
      }
    }
    return 0;
  }

  HashFunc hash_func_;
 private:
  ProfilingTestUtil() = default;
  ProfFunc func_;
  size_t hash_count_{0UL};
  std::map<uint16_t, std::vector<std::pair<uint32_t, uint32_t>>> reg_types_{};
};
}  // namespace ge

#endif
