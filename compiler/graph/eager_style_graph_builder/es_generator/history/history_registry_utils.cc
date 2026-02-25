/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history_registry_utils.h"

#include <ctime>
#include <fstream>
#include <regex>
#include <string>

namespace ge {
namespace es {
namespace history {
bool GetCurrentDate(std::string &date) {
  const std::time_t now = std::time(nullptr);
  const std::tm *const ptm = std::localtime(&now);
  if (ptm == nullptr) {
    return false;
  }
  constexpr size_t kDateBufferLen = 16U;
  char buffer[kDateBufferLen] = {};
  if (std::strftime(buffer, sizeof(buffer), kReleaseDateFormat, ptm) == 0U) {
    return false;
  }
  date = buffer;
  return true;
}

bool ReadJsonFile(const std::string &path, nlohmann::json &out, std::string &error_msg) {
  std::ifstream file(path);
  if (!file.is_open()) {
    error_msg = "cannot open file: " + path;
    return false;
  }
  try {
    file >> out;
    return true;
  } catch (const nlohmann::json::parse_error &e) {
    error_msg = "JSON parse error in " + path + ": " + e.what();
    return false;
  } catch (const std::exception &e) {
    error_msg = "error reading " + path + ": " + e.what();
    return false;
  }
}

bool ValidateReleaseDateFormat(const std::string &date) {
  static const std::regex kDateFormatRegex(R"(\d{4}-\d{2}-\d{2})");
  if (!std::regex_match(date, kDateFormatRegex)) {
    return false;
  }
  int year = std::stoi(date.substr(0, 4));
  int month = std::stoi(date.substr(5, 2));
  int day = std::stoi(date.substr(8, 2));
  if (year < 1900 || year > 9999) return false;
  if (month < 1 || month > 12) return false;
  if (day < 1 || day > 31) return false;
  return true;
}

bool ValidateRequireString(const nlohmann::json &j, const std::string &field,
                           std::string &error_msg, const std::string &prefix) {
  if (!j.contains(field) || !j[field].is_string()) {
    error_msg = prefix + field + " is required and must be a string";
    return false;
  }
  return true;
}

bool ValidateRequireArray(const nlohmann::json &j, const std::string &field,
                          std::string &error_msg, const std::string &prefix) {
  if (!j.contains(field) || !j[field].is_array()) {
    error_msg = prefix + field + " is required and must be an array";
    return false;
  }
  return true;
}

bool ValidateOptionalString(const nlohmann::json &j, const std::string &field,
                            std::string &error_msg, const std::string &prefix) {
  if (j.contains(field) && !j[field].is_string()) {
    error_msg = prefix + field + " must be a string";
    return false;
  }
  return true;
}

bool ValidateIndexJson(const nlohmann::json &index_json, std::string &error_msg) {
  if (!index_json.is_object()) {
    error_msg = "index.json is not a JSON object";
    return false;
  }
  if (!ValidateRequireString(index_json, "version", error_msg)) {
    return false;
  }
  if (!ValidateRequireArray(index_json, "releases", error_msg)) {
    return false;
  }
  return true;
}

bool ValidateIndexReleaseEntryJson(const nlohmann::json &entry, const size_t index,
                                   std::set<std::string> &seen_versions, std::string &error_msg) {
  const std::string prefix = "releases[" + std::to_string(index) + "] ";
  if (!entry.is_object()) {
    error_msg = prefix + "is not an object";
    return false;
  }
  if (!ValidateRequireString(entry, "release_version", error_msg, prefix)) {
    return false;
  }
  std::string version = entry["release_version"].get<std::string>();
  if (seen_versions.count(version) > 0U) {
    error_msg = prefix + "release_version is duplicate: " + version;
    return false;
  }
  seen_versions.insert(version);
  if (!ValidateRequireString(entry, "release_date", error_msg, prefix)) {
    return false;
  }
  const std::string date = entry["release_date"].get<std::string>();
  if (!ValidateReleaseDateFormat(date)) {
    error_msg = prefix + "release_date format is invalid: " + date;
    return false;
  }
  return true;
}
}  // namespace history
}  // namespace es
}  // namespace ge
