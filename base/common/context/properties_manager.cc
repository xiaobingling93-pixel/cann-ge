/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/context/properties_manager.h"

#include <fstream>

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "base/err_msg.h"
#include "mmpa/mmpa_api.h"

namespace {
constexpr size_t kMaxErrorStrLength = 128U;
}  // namespace

namespace ge {
PropertiesManager::PropertiesManager() {}

// singleton
PropertiesManager &PropertiesManager::Instance() {
  static PropertiesManager instance;
  return instance;
}

// Initialize property configuration
bool PropertiesManager::Init(const std::string &file_path) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (is_inited_) {
    GELOGW("Already inited, will be initialized again");
    properties_map_.clear();
    is_inited_ = false;
    return is_inited_;
  }

  if (!LoadFileContent(file_path)) {
    return false;
  }

  is_inited_ = true;
  return is_inited_;
}

// Load file contents
bool PropertiesManager::LoadFileContent(const std::string &file_path) {
  // Normalize the path
  const std::string resolved_file_path = RealPath(file_path.c_str());
  if (resolved_file_path.empty()) {
    DOMI_LOGE("Invalid input file path [%s], make sure that the file path is correct.", file_path.c_str());
    return false;
  }
  std::ifstream fs(resolved_file_path, std::ifstream::in);

  if (!fs.is_open()) {
    GELOGE(PARAM_INVALID, "[Open][File]Failed, file path %s invalid", file_path.c_str());
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const std::string errmsg = "[Errno " + std::to_string(mmGetErrorCode()) + "] " +
                               mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E13001",
        std::vector<const char *>({"file", "errmsg"}),
        std::vector<const char *>({resolved_file_path.c_str(), errmsg.c_str()})
    );
    return false;
  }

  std::string line;

  while (getline(fs, line)) {  // line not with \n
    if (!ParseLine(line)) {
      GELOGE(PARAM_INVALID, "[Parse][Line]Failed, content is %s", line.c_str());
      fs.close();
      return false;
    }
  }

  fs.close();  // close the file

  GELOGI("LoadFileContent success.");
  return true;
}

// Parsing the command line
bool PropertiesManager::ParseLine(const std::string &line) {
  const std::string temp = TrimStr(line);
  // Comment or newline returns true directly
  if ((temp.find_first_of('#') == 0U) || (*(temp.c_str()) == '\n')) {
    return true;
  }

  if (!temp.empty()) {
    const std::string::size_type pos = temp.find_first_of(delimiter);
    if (pos == std::string::npos) {
      GELOGE(PARAM_INVALID, "[Check][Param]Incorrect line %s, it must include %s",
             line.c_str(), delimiter.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Incorrect line %s, it must include %s",
                        line.c_str(), delimiter.c_str());
      return false;
    }

    const std::string map_key = TrimStr(temp.substr(0U, pos));
    const std::string value = TrimStr(temp.substr(pos + 1U));
    if (map_key.empty() || value.empty()) {
      GELOGE(PARAM_INVALID, "[Check][Param]Map_key or value empty, line %s", line.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Map_key or value empty, line %s", line.c_str());
      return false;
    }

    properties_map_[map_key] = value;
  }

  return true;
}

// Remove the space and tab before and after the string
std::string PropertiesManager::TrimStr(const std::string &str) const {
  if (str.empty()) {
    return str;
  }

  const std::string::size_type start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return str;
  }

  const std::string::size_type end = str.find_last_not_of(" \t\r\n") + 1U;
  return str.substr(start, end);
}

// Get property value, if not found, return ""
std::string PropertiesManager::GetPropertyValue(const std::string &map_key) {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = properties_map_.find(map_key);
  if (properties_map_.end() != iter) {
    return iter->second;
  }

  return "";
}

// Set property value
void PropertiesManager::SetPropertyValue(const std::string &map_key, const std::string &value) {
  const std::lock_guard<std::mutex> lock(mutex_);
  properties_map_[map_key] = value;
}

// return properties_map_
std::map<std::string, std::string> PropertiesManager::GetPropertyMap() {
  const std::lock_guard<std::mutex> lock(mutex_);
  return properties_map_;
}

// Set separator
void PropertiesManager::SetPropertyDelimiter(const std::string &de) {
  const std::lock_guard<std::mutex> lock(mutex_);
  delimiter = de;
}
}  // namespace ge
