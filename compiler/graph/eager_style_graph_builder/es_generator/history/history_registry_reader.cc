/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history_registry_reader.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "history_registry_interface.h"
#include "history_registry_utils.h"
#include "ir_proto_codec.h"

namespace ge {
namespace es {
namespace history {
namespace {
std::string NormalizePkgDir(const std::string &pkg_dir) {
  if (pkg_dir.empty()) {
    throw std::invalid_argument("pkg_dir is empty, please check history_registry");
  }
  return pkg_dir.back() != '/' && pkg_dir.back() != '\\' ? pkg_dir + "/" : pkg_dir;
}

std::string GetVersionDir(const std::string &pkg_dir, const std::string &version) {
  return NormalizePkgDir(pkg_dir) + "registry/" + version + "/";
}

bool ValidateMetadataJson(const nlohmann::json &j, const std::string &version, std::string &error_msg) {
  if (!j.is_object()) {
    error_msg = "metadata.json is not a JSON object";
    return false;
  }
  if (!ValidateRequireString(j, "release_version", error_msg)) {
    return false;
  }
  const std::string release_version = j["release_version"].get<std::string>();
  if (release_version != version) {
    error_msg = "release_version is not equal to version: " + release_version;
    return false;
  }
  if (!ValidateOptionalString(j, "branch_name", error_msg)) {
    return false;
  }
  return true;
}

// 校验 operators.json 根结构，op 校验在 IrProtoCodec::FromJson 中
bool ValidateOperatorsJson(const nlohmann::json &j, std::string &error_msg) {
  if (!j.is_object()) {
    error_msg = "operators.json is not a JSON object";
    return false;
  }
  return ValidateRequireArray(j, "operators", error_msg);
}

// 将 YYYY-MM-DD 转为自 1970-01-01 的天数
int ParseReleaseDateToDays(const std::string &date) {
  std::tm tm = {};
  std::istringstream ss(date);
  ss >> std::get_time(&tm, kReleaseDateFormat);
  std::time_t t = std::mktime(&tm);
  return static_cast<int>(t / (24 * 60 * 60));
}
}  // namespace

std::vector<VersionMeta> HistoryRegistryReader::LoadIndex(const std::string &pkg_dir) {
  std::string path = NormalizePkgDir(pkg_dir) + kIndexFilename;
  nlohmann::json j;
  std::string error_msg;
  if (!ReadJsonFile(path, j, error_msg) || !ValidateIndexJson(j, error_msg)) {
    std::cerr << "Failed to load index.json: " << error_msg << std::endl;
    return {};
  }

  std::vector<VersionMeta> version_metas;
  std::set<std::string> seen_versions;
  size_t index = 0U;
  for (const auto &entry : j["releases"]) {
    std::string entry_warning;
    if (!ValidateIndexReleaseEntryJson(entry, index, seen_versions, entry_warning)) {
      std::cerr << "Skip invalid release entry in index.json: " << entry_warning << std::endl;
      ++index;
      continue;
    }
    const std::string version = entry["release_version"].get<std::string>();
    try {
      VersionMeta meta = LoadMetadata(pkg_dir, version);
      meta.release_date = entry["release_date"].get<std::string>();
      version_metas.push_back(std::move(meta));
    } catch (const std::exception &e) {
      std::cerr << "Skip invalid release version " << version << " in index.json: " << e.what() << std::endl;
    }
    ++index;
  }
  std::sort(version_metas.begin(), version_metas.end(),
            [](const VersionMeta &a, const VersionMeta &b) { return a.release_date < b.release_date; });
  return version_metas;
}

VersionMeta HistoryRegistryReader::LoadMetadata(const std::string &pkg_dir, const std::string &version) {
  std::string version_dir = GetVersionDir(pkg_dir, version);
  std::string path = version_dir + kMetadataFilename;
  nlohmann::json j;
  std::string warning;
  if (!ReadJsonFile(path, j, warning) || !ValidateMetadataJson(j, version, warning)) {
    throw std::runtime_error("Failed to load metadata.json: " + warning);
  }

  VersionMeta meta;
  meta.release_version = j["release_version"].get<std::string>();
  if (j.contains("branch_name")) {
    meta.branch_name = j["branch_name"].get<std::string>();
  }
  return meta;
}

IrOpProto HistoryRegistryReader::LoadOpProto(const std::string &pkg_dir, const std::string &version,
                                             const std::string &op_type) {
  std::string version_dir = GetVersionDir(pkg_dir, version);
  std::string path = version_dir + kOperatorsFilename;
  nlohmann::json j;
  std::string warning;
  if (!ReadJsonFile(path, j, warning) || !ValidateOperatorsJson(j, warning)) {
    throw std::runtime_error("Failed to load operators.json: " + warning);
  }

  const nlohmann::json *matched = nullptr;
  for (const auto &op_json : j["operators"]) {
    if (!op_json.is_object() || !op_json.contains("op_type") || !op_json["op_type"].is_string()) {
      continue;
    }
    if (op_json["op_type"].get<std::string>() != op_type) {
      continue;
    }
    if (matched != nullptr) {
      throw std::runtime_error("Found duplicate op " + op_type + " in operators.json");
    }
    matched = &op_json;
  }
  if (matched == nullptr) {
    throw std::runtime_error("Op " + op_type + " not found in operators.json");
  }
  try {
    return IrProtoCodec::FromJson(*matched);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to parse json for op " + op_type + ": " + e.what());
  }
}

std::vector<VersionMeta> HistoryRegistryReader::SelectWindowVersions(const std::vector<VersionMeta> &all_versions,
                                                                    const std::string &current_version, int window_days) {
  if (all_versions.empty()) {
    std::cout << "Ops history registry versions is empty." << std::endl;
    return {};
  }

  std::string current_date;
  if (current_version.empty()) {
    std::cout << "release_version is empty, use current date as anchor to select history registry versions." << std::endl;
    if (!GetCurrentDate(current_date)) {
      std::cerr << "Failed to get current date from system for history registry versions selection." << std::endl;
      return {};
    }
  } else {
    auto current_it = std::find_if(all_versions.begin(), all_versions.end(),
                                   [&current_version](const VersionMeta &m) {
                                     return m.release_version == current_version;
                                   });
    if (current_it == all_versions.end()) {
      std::cerr << "Version " << current_version << " not found in all history registry versions." << std::endl;
      return {};
    }
    current_date = current_it->release_date;
  }

  std::vector<VersionMeta> window;
  const int current_days = ParseReleaseDateToDays(current_date);
  for (auto it = all_versions.rbegin(); it != all_versions.rend(); ++it) {
    const int version_days = ParseReleaseDateToDays(it->release_date);
    const int delta_days = current_days - version_days;
    if (delta_days < 0) {
      continue;
    }
    if (delta_days > window_days) {
      break;
    }
    window.push_back(*it);
  }
  std::cerr << "Selected history registry versions num: " << window.size()  << std::endl;
  std::reverse(window.begin(), window.end());
  return window;
}
}  // namespace history
}  // namespace es
}  // namespace ge
