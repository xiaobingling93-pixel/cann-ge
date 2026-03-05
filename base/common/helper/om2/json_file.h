/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_JSON_FILE_H
#define BASE_COMMON_HELPER_JSON_FILE_H

#include "nlohmann/json.hpp"
#include <string>
#include <fstream>
#include "framework/common/debug/log.h"

namespace ge {
class JsonFile {
 public:
  using json = nlohmann::json;

  // Create empty json object
  JsonFile() noexcept : valid_(true) {}
  // Create json object from file
  explicit JsonFile(const std::string &file_path) noexcept : valid_(false) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
      GELOGW("Cannot open file [%s]", file_path.c_str());
      return;
    }

    try {
      ifs >> data_;
      valid_ = true;
    } catch (const std::exception &e) {
      GELOGW("Cannot create json from [%s], msg: %s", file_path.c_str(), e.what());
    }
  }

  JsonFile(const uint8_t *data, size_t data_size) noexcept : valid_(false) {
    if (data == nullptr || data_size == 0) {
      GELOGW("Cannot create json from empty memory buffer");
      return;
    }

    try {
      std::string json_str(reinterpret_cast<const char *>(data), data_size);
      data_ = json::parse(json_str);
      valid_ = true;
    } catch (const std::exception &e) {
      GELOGW("Cannot create json from memory buffer, msg: %s", e.what());
    }
  }

  explicit JsonFile(json j) noexcept : data_(std::move(j)), valid_(true) {}

  bool IsValid() const {
    return valid_;
  }

  template <typename T>
  JsonFile &Set(const std::string &key, const T &value) {
    data_[key] = value;
    return *this;
  }

  JsonFile &Set(const std::string &key, const JsonFile &jsonfile) {
    data_[key] = jsonfile.data_;
    return *this;
  }

  template <typename T>
  bool Get(const std::string &key, T &out) const {
    if (!valid_ || !data_.contains(key)) {
      return false;
    }
    try {
      out = data_.at(key).get<T>();
      return true;
    } catch (const std::exception &e) {
      GELOGW("Cannot get value with key [%s], msg: %s", key.c_str(), e.what());
      return false;
    }
  }

  std::string Dump(const bool pretty = true) const {
    if (!valid_) {
      return "{}";
    }
    return pretty ? data_.dump(4) : data_.dump();
  }

  const json &Raw() const {
    return data_;
  }

  json &Raw() {
    return data_;
  }

 private:
  json data_;
  bool valid_;
};
}  // namespace ge
#endif  // BASE_COMMON_HELPER_JSON_FILE_H
