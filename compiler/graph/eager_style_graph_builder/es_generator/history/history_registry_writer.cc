/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history_registry_writer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/utils/file_utils.h"
#include "history_registry_utils.h"
#include "ir_proto_codec.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace es {
namespace history {
namespace {
constexpr const char *kIndexSchemaVersion = "1.0.0";
constexpr int kJsonIndent = 2;

void EnsureVersionDirectory(const std::string &version_dir) {
  if (mmAccess(version_dir.c_str()) != EN_OK) {
    if (ge::CreateDir(version_dir) != 0) {
      throw std::runtime_error("Failed to create version directory: " + version_dir);
    }
    std::cout << "Created version directory: " << version_dir << std::endl;
  }
  std::cout << "Version directory: " << version_dir << std::endl;
}

void WriteJsonFile(const std::string &path, const nlohmann::json &content) {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for write: " + path);
  }
  file << content.dump(kJsonIndent);
  std::cout << "Wrote file successfully: " << path << std::endl;
}

nlohmann::json LoadIndexJsonIfExists(const std::string &path) {
  if (mmAccess(path.c_str()) != EN_OK) {
    return {};
  }
  nlohmann::json existing;
  std::string error_msg;
  if (!ReadJsonFile(path, existing, error_msg) || !ValidateIndexJson(existing, error_msg)) {
    std::cerr << "Failed to load existing index.json: " << error_msg << ", creating new file" << std::endl;
    return {};
  }
  return existing;
}

nlohmann::json BuildIndexJson(const std::string &index_path, const std::string &release_version,
                              const std::string &release_date) {
  nlohmann::json json = nlohmann::json::object();
  json["version"] = kIndexSchemaVersion;
  json["releases"] = nlohmann::json::array();
  const auto existing = LoadIndexJsonIfExists(index_path);
  if (!existing.empty()) {
    json["version"] = existing["version"];
    std::set<std::string> seen_versions;
    size_t idx = 0U;
    for (const auto &entry : existing["releases"]) {
      std::string entry_warning;
      if (!ValidateIndexReleaseEntryJson(entry, idx, seen_versions, entry_warning)) {
        std::cerr << "Invalid index release record found, will be removed: " << entry_warning << std::endl;
        ++idx;
        continue;
      }

      if (entry["release_version"].get<std::string>() == release_version) {
        throw std::invalid_argument(
          "Given release_version already exists in index, please check index.json or use another version: " + release_version);
      }
      json["releases"].emplace_back(entry);
      ++idx;
    }
  }

  nlohmann::json release_entry = nlohmann::json::object();
  release_entry["release_version"] = release_version;
  release_entry["release_date"] = release_date;
  json["releases"].emplace_back(std::move(release_entry));
  std::cout << "Added release record to index: release version " << release_version
            << ", release date " << release_date << std::endl;

  std::sort(json["releases"].begin(), json["releases"].end(),
            [](const nlohmann::json &a, const nlohmann::json &b) {
              return a["release_date"].get<std::string>() < b["release_date"].get<std::string>(); });
  return json;
}

nlohmann::json BuildOperatorsJson(const std::vector<OpDescPtr> &ops) {
  nlohmann::json operators = nlohmann::json::array();
  for (const auto &op : ops) {
    if (op == nullptr) {
      continue;
    }
    try {
      const auto proto = IrProtoCodec::FromOpDesc(op);
      const nlohmann::json op_json = IrProtoCodec::ToJson(proto);
      operators.emplace_back(op_json);
    } catch (const std::exception &e) {
      std::cerr << "Failed to build structured data for op " << op->GetType() << ", skip: " << e.what() << std::endl;
    }
  }
  nlohmann::json root = nlohmann::json::object();
  root["operators"] = std::move(operators);
  return root;
}

nlohmann::json BuildMetadataJson(const std::string &release_version, const std::string &branch_name) {
  nlohmann::json meta = nlohmann::json::object();
  meta["release_version"] = release_version;
  if (!branch_name.empty()) {
    meta["branch_name"] = branch_name;
  }
  return meta;
}
}  // namespace

void HistoryRegistryWriter::WriteIndexJson(const std::string &output_dir, const std::string &release_version,
                                           const std::string &release_date) {
  const std::string index_path = output_dir + "index.json";
  WriteJsonFile(index_path, BuildIndexJson(index_path, release_version, release_date));
}

void HistoryRegistryWriter::WriteMetadataJson(const std::string &version_dir, const std::string &release_version,
                                              const std::string &branch_name) {
  const std::string metadata_path = version_dir + "metadata.json";
  WriteJsonFile(metadata_path, BuildMetadataJson(release_version, branch_name));
}

void HistoryRegistryWriter::WriteOperatorsJson(const std::string &version_dir, const std::vector<OpDescPtr> &all_ops) {
  const std::string operators_path = version_dir + "operators.json";
  WriteJsonFile(operators_path, BuildOperatorsJson(all_ops));
}

void HistoryRegistryWriter::WriteRegistry(const std::string &output_dir, const std::string &release_version,
                                          const std::string &release_date, const std::string &branch_name,
                                          const std::vector<OpDescPtr> &all_ops) {
  std::string date = release_date;
  if (date.empty()) {
    if (!GetCurrentDate(date)) {
      throw std::runtime_error("Failed to get current date from system. Please provide release_date in YYYY-MM-DD format.");
    }
  } else if (!ValidateReleaseDateFormat(date)) {
    throw std::invalid_argument(
      "Given release_date parameter for history registry generator is not in the correct format (YYYY-MM-DD).");
  }
  if (release_version.empty()) {
    throw std::invalid_argument("The required parameter release_version for history registry generator is not set.");
  }
  const std::string version_dir = output_dir + "registry/" + release_version + "/";
  EnsureVersionDirectory(version_dir);

  WriteMetadataJson(version_dir, release_version, branch_name);
  WriteOperatorsJson(version_dir, all_ops);
  WriteIndexJson(output_dir, release_version, date);
  std::cout << "History registry generated at: " << output_dir << std::endl;
}
}  // namespace history
}  // namespace es
}  // namespace ge
