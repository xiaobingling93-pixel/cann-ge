/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_UTILS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_UTILS_H_

#include <set>
#include <string>

#include <nlohmann/json.hpp>

namespace ge {
namespace es {
namespace history {
constexpr const char *kIndexFilename = "index.json";
constexpr const char *kMetadataFilename = "metadata.json";
constexpr const char *kOperatorsFilename = "operators.json";
constexpr const char *kReleaseDateFormat = "%Y-%m-%d";

// 获取当前本地日期，格式 YYYY-MM-DD
bool GetCurrentDate(std::string &date);

bool ReadJsonFile(const std::string &path, nlohmann::json &out, std::string &error_msg);

// 校验日期格式是否为 "YYYY-MM-DD"，并做基本范围检查
bool ValidateReleaseDateFormat(const std::string &date);

// ============== 通用字段校验函数 ==============
// 校验必填 string 字段
bool ValidateRequireString(const nlohmann::json &j, const std::string &field,
                           std::string &error_msg, const std::string &prefix = "");

// 校验必填 array 字段
bool ValidateRequireArray(const nlohmann::json &j, const std::string &field,
                          std::string &error_msg, const std::string &prefix = "");

// 校验可选 string 字段
bool ValidateOptionalString(const nlohmann::json &j, const std::string &field,
                            std::string &error_msg, const std::string &prefix = "");

// ============== JSON 结构校验函数 ==============
bool ValidateIndexJson(const nlohmann::json &index_json, std::string &error_msg);

bool ValidateIndexReleaseEntryJson(const nlohmann::json &entry, size_t index,
                                   std::set<std::string> &seen_versions, std::string &error_msg);
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_HISTORY_REGISTRY_UTILS_H_
