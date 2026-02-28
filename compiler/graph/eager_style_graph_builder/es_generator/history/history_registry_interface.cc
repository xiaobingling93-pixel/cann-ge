/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "history_registry_interface.h"

#include <string>
#include <vector>

#include "history_registry_reader.h"
namespace ge {
namespace es {
namespace history {
bool LoadHistoryWindowVersions(const std::string &pkg_dir,
                               const std::string &baseline_version,
                               std::vector<VersionMeta> &window_versions,
                               std::string &error_msg) {
  std::vector<VersionMeta> all_versions;
  try {
    all_versions = HistoryRegistryReader::LoadIndex(pkg_dir);
  } catch (const std::exception &e) {
    window_versions.clear();
    error_msg = e.what();
    return false;
  }
  window_versions = HistoryRegistryReader::SelectWindowVersions(all_versions, baseline_version);
  error_msg.clear();
  return true;
}

HistoryContext LoadHistoryChain(const std::string &pkg_dir, const std::vector<VersionMeta> &window_versions,
                                const std::string &op_type, std::vector<std::string> &warnings) {
  HistoryContext ctx;
  for (const auto &ver : window_versions) {
    IrOpProto proto;
    try {
      proto = HistoryRegistryReader::LoadOpProto(pkg_dir, ver.release_version, op_type);
    } catch (const std::exception &e) {
      warnings.push_back("op " + op_type + " skip version " + ver.release_version + ": " + e.what());
      continue;
    }
    ctx.versions.push_back(ver);
    ctx.proto_chain.push_back(std::move(proto));
  }
  return ctx;
}

}  // namespace history
}  // namespace es
}  // namespace ge
