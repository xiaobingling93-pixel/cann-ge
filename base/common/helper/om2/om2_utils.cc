/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/checker.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
namespace {
thread_local uint32_t om2_workdir_count = 0U;
Status GetWorkspaceBaseDir(std::string &base_dir) {
  base_dir.clear();
  GE_ASSERT_SUCCESS(ge::GetAscendWorkPath(base_dir));
  if (base_dir.empty()) {
    const ge::char_t *path_env = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HOME, path_env);
    GE_ASSERT_NOTNULL(path_env);
    GE_ASSERT_TRUE(strnlen(path_env, static_cast<size_t>(MMPA_MAX_PATH)) > 0U);
    const std::string file_path = ge::RealPath(path_env);
    GE_ASSERT_TRUE(!file_path.empty());
    base_dir = file_path;
  }
  if (base_dir.back() != '/') {
    base_dir += '/';
  }
  return ge::GRAPH_SUCCESS;
}

Status GetOm2WorkspaceDir(std::string &ws_dir) {
  GE_ASSERT_SUCCESS(GetWorkspaceBaseDir(ws_dir));
  constexpr uint32_t kReservedPathLength = 128U;
  ws_dir.reserve(ws_dir.size() + kReservedPathLength);

  ws_dir.append(".ascend_temp/.tmp_om2_workspace/")
      .append(std::to_string(mmGetPid()))
      .append("_")
      .append(std::to_string(mmGetTid()))
      .append("_")
      .append(std::to_string(om2_workdir_count++))
      .append("/");

  GELOGI("om2 workspace dir is %s", ws_dir.c_str());
  return ge::GRAPH_SUCCESS;
}

}  // namespace
Status Om2Utils::GetAscendHomePath(std::string &home_path) {
  const char_t *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, path_env);
  GE_ASSERT_TRUE(path_env != nullptr, "[Call][MM_SYS_GET_ENV] Failed to get env ASCEND_HOME_PATH.");
  home_path = path_env;
  std::string file_path = RealPath(home_path.c_str());
  GE_ASSERT_TRUE(!file_path.empty(), "[Call][RealPath] File path %s is invalid.", home_path.c_str());
  if (home_path.back() != '/') {
    home_path += '/';
  }
  GELOGI("Get ascend home path from env: %s", home_path.c_str());
  return SUCCESS;
}

Status Om2Utils::CreateOm2WorkspaceDir(std::string &ws_dir) {
  GE_ASSERT_SUCCESS(GetOm2WorkspaceDir(ws_dir));
  if (mmAccess2(ws_dir.c_str(), M_F_OK) == EN_OK) {
    GELOGI("[OM2] Temporary workspace %s is not empty, will be removed.", ws_dir.c_str());
    GE_ASSERT_TRUE(mmRmdir(ws_dir.c_str()) == 0, "[OM2] Failed to remove dir:%s.", ws_dir.c_str());
  }
  GE_ASSERT_TRUE(ge::CreateDir(ws_dir) == 0, "Failed to create om2 work dir: [%s].", ws_dir.c_str());
  return ge::GRAPH_SUCCESS;
}

Status Om2Utils::RmOm2WorkspaceDir(const std::string &ws_dir) {
  GELOGD("Start to remove workspace dir %s", ws_dir.c_str());
  if (!ws_dir.empty()) {
    GE_ASSERT_TRUE(mmRmdir(ws_dir.c_str()) == 0, "remove dir [%s] failed!", ws_dir.c_str());
  }
  GELOGI("Remove dir success, workspace dir is %s", ws_dir.c_str());
  return ge::GRAPH_SUCCESS;
}

Status Om2Utils::CompileGeneratedCppToSo(const std::vector<std::string> &cpp_file_paths,
                                         const std::string &so_output_path, const bool is_release) {
  GE_ASSERT_TRUE(!cpp_file_paths.empty(), "No cpp files provided");
  for (const auto &cpp_file_path : cpp_file_paths) {
    GE_ASSERT_TRUE(mmAccess2(cpp_file_path.c_str(), M_F_OK) == EN_OK, "Cpp file not exist: %s", cpp_file_path.c_str());
  }
  const auto pos = so_output_path.find_last_of('/');
  GE_ASSERT_TRUE(pos != std::string::npos, "Invalid so output path: %s", so_output_path.c_str());
  const std::string work_dir = so_output_path.substr(0, pos);
  const std::string makefile_path = work_dir + "/Makefile";
  GE_ASSERT_TRUE(mmAccess2(makefile_path.c_str(), M_F_OK) == EN_OK, "Makefile not exist: %s", makefile_path.c_str());

  const std::string build_log_path =
      "/tmp/om2_make_" + std::to_string(mmGetPid()) + "_" + std::to_string(mmGetTid()) + ".log";
  std::string command = "make -C " + work_dir;
  if (!is_release) {
    command += " USE_STUB_LIB=1";
  }
  command += " >" + build_log_path + " 2>&1";
  GELOGI("[OM2] Compile generated cpp to so, command: %s", command.c_str());
  GE_ASSERT_TRUE(system(command.c_str()) == 0, "Failed to compile so: %s, build log: %s", so_output_path.c_str(),
                 build_log_path.c_str());
  GE_ASSERT_TRUE(mmAccess2(so_output_path.c_str(), M_F_OK) == EN_OK, "Compiled so not exist: %s", so_output_path.c_str());
  return GRAPH_SUCCESS;
}

}  // namespace ge
