/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_fuse_config_utils.h"
#include <algorithm>
namespace ge {
namespace {
bool IsForceTagValid(const std::string &tag) {
  if (tag.empty()) {
    return true;
  }
  const std::set<std::string> kValidTag{"R"};
  if (kValidTag.find(tag) == kValidTag.cend()) {
    return false;
  }
  return true;
}

bool IsInputAllDigit(const std::string &input) {
  return !(std::any_of(input.cbegin(), input.cend(),
                       [](const ge::char_t c) -> bool { return !std::isdigit(static_cast<uint8_t>(c)); }));
}

ge::Status ParseForceTilingCaseForSingleMode(const std::string &input, ForceTilingCaseResult &force_tiling_case,
                                             bool &has_parsed) {
  // 检查是否包含下划线
  size_t underscore_pos = input.find('_');
  // 数字加下划线加sub_tag格式
  std::string case_str = (underscore_pos == std::string::npos) ? input : input.substr(0UL, underscore_pos);
  std::string sub_tag = (underscore_pos == std::string::npos) ? "" : input.substr(underscore_pos + 1UL);
  // 验证case部分是否为有效数字
  if (IsInputAllDigit(case_str) && !case_str.empty()) {
    int32_t case_num = std::stoi(case_str);
    // 返回单一选择模式结果（有sub_tag）
    force_tiling_case = {true, case_num, sub_tag, {}};
    GE_ASSERT_TRUE(IsForceTagValid(sub_tag), "sub_tag is invalid, sub_tag=%s", sub_tag.c_str());
    GELOGD("Parse force tiling case success, force_tiling_case=%s", force_tiling_case.Debug().c_str());
    has_parsed = true;
    return ge::SUCCESS;
  }
  has_parsed = false;
  return ge::SUCCESS;
}
}
void ForceTilingCaseResult::Clear() {
  is_single_mode = true;
  single_case = -1;
  group_cases.clear();
}

[[nodiscard]] std::string ForceTilingCaseResult::Debug() const {
  std::stringstream ss;
  ss << "ForceTilingCaseResult[is_single_mode(" << is_single_mode << "), single_case(" << single_case
     << "), group_cases(";
  if (is_single_mode) {
    ss << ", single_case(" << single_case << ")";
  } else {
    ss << ", group_cases(";
    for (const auto &pair : group_cases) {
      ss << "[" << pair.first << "=" << pair.second.first << ", " << pair.second.second << "]";
    }
    ss << ")";
  }
  ss << "]";
  return ss.str();
}

[[nodiscard]] std::pair<int32_t, std::string> ForceTilingCaseResult::GetCase(size_t group_id) const {
  if (is_single_mode) {
    return {single_case, ""};
  }
  auto it = group_cases.find(group_id);
  if (it != group_cases.cend()) {
    return it->second;
  }
  return {-1, ""};
}

std::string ForceTilingCaseResult::GetTag(size_t group_id) const {
  if (is_single_mode) {
    return single_sub_tag;
  }
  auto it = group_cases.find(group_id);
  if (it != group_cases.cend()) {
    return it->second.second;
  }
  return "";
}

ge::Status AttStrategyConfigUtils::ParseForceTilingCase(const std::string &input,
                                                        ForceTilingCaseResult &force_tiling_case) {
  // 检查空输入
  GE_ASSERT_TRUE(!input.empty(), "input is empty");
  // 检查是否纯数字或数字加下划线加sub_tag（单一选择模式）
  bool has_parsed = false;
  if (input.find(',') == std::string::npos) {
    GE_ASSERT_SUCCESS(ParseForceTilingCaseForSingleMode(input, force_tiling_case, has_parsed));
  }
  if (has_parsed) {
    return ge::SUCCESS;
  }
  // 解析逗号分隔的"gX_Y[_subtag]"模式
  std::map<size_t, std::pair<int32_t, std::string>> result_map;
  std::istringstream ss(input);
  std::string token;
  while (std::getline(ss, token, ',')) {
    // 去除可能的空格
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    // 检查基本结构：长度至少4字符，以'g'开头
    constexpr size_t kMinTokenSize = 4UL;
    GE_ASSERT_TRUE(token.size() >= kMinTokenSize && token[0] == 'g', "Invalid token format: %s", token.c_str());
    // 查找第一个下划线位置（分隔组号和case号）
    size_t first_underscore_pos = token.find('_');
    // 1.无下划线, 2.下划线紧接'g'后（缺少组号）, 3.下划线在末尾（缺少case号）)
    GE_ASSERT_TRUE(
        (first_underscore_pos != std::string::npos) && (first_underscore_pos != 1UL) && (first_underscore_pos != token.size() - 1UL),
        "Invalid underscore position in: %s", token.c_str());
    // 提取组号部分（'g'之后到下划线前）
    std::string group_str = token.substr(1UL, first_underscore_pos - 1UL);
    // 查找第二个下划线位置（分隔case号和sub_tag）
    size_t second_underscore_pos = token.find('_', first_underscore_pos + 1UL);
    std::string case_str;
    std::string sub_tag;
    if (second_underscore_pos == std::string::npos) {
      // 没有第二个下划线，整个剩余部分是case号
      case_str = token.substr(first_underscore_pos + 1UL);
    } else {
      // 有第二个下划线，提取case号和sub_tag
      case_str = token.substr(first_underscore_pos + 1UL, second_underscore_pos - first_underscore_pos - 1UL);
      sub_tag = token.substr(second_underscore_pos + 1UL);
    }
    // 验证组号和case号是否为有效数字
    auto is_num = [](const std::string &s) {
      return !s.empty() &&
             std::all_of(s.begin(), s.end(), [](ge::char_t c) { return std::isdigit(static_cast<uint8_t>(c)); });
    };
    GE_ASSERT_TRUE(is_num(group_str), "group_str is not num, token=%s", token.c_str());
    GE_ASSERT_TRUE(is_num(case_str), "case_str is not num, token=%s", token.c_str());
    int32_t group_id = std::stoi(group_str);
    int32_t case_id = std::stoi(case_str);
    // 检查组号是否重复
    GE_ASSERT_TRUE(result_map.find(static_cast<size_t>(group_id)) == result_map.end(),
                   "Duplicate group ID: %d, token=%s", group_id, token.c_str());
    GE_ASSERT_TRUE(IsForceTagValid(sub_tag), "sub_tag is invalid, sub_tag=%s", sub_tag.c_str());
    result_map[group_id] = std::make_pair(case_id, sub_tag);
  }
  // 返回分组选择模式结果
  force_tiling_case = {false, -1, "", result_map};
  GELOGD("Got input string: %s, force_tiling_case: %s", input.c_str(), force_tiling_case.Debug().c_str());
  return ge::SUCCESS;
}
}  // namespace ge