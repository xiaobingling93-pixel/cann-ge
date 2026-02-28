/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attr_type_traits.h"

#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

namespace ge {
namespace es {
namespace history {
namespace {
std::string JoinStrings(const std::vector<std::string> &items) {
  std::stringstream ss;
  bool first = true;
  for (const auto &item : items) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << item;
  }
  return ss.str();
}

template<typename T>
std::string JoinNumbers(const std::vector<T> &items) {
  std::stringstream ss;
  bool first = true;
  for (const auto &item : items) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << std::to_string(item);
  }
  return ss.str();
}

std::string JoinBools(const std::vector<bool> &items) {
  std::stringstream ss;
  bool first = true;
  for (const auto item : items) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << (item ? "true" : "false");
  }
  return ss.str();
}

template<typename T>
bool TryGetJsonValue(const nlohmann::json &j, T &value) {
  try {
    value = j.get<T>();
    return true;
  } catch (...) {
    return false;
  }
}

AttrDefaultParseResult ParseIntDefault(const nlohmann::json &j) {
  int64_t value = 0;
  if (!TryGetJsonValue(j, value)) {
    return {false, "", "default_value type mismatch for Int"};
  }
  return {true, "=" + std::to_string(value), ""};
}

AttrDefaultParseResult ParseFloatDefault(const nlohmann::json &j) {
  double value = 0.0;
  if (!TryGetJsonValue(j, value)) {
    return {false, "", "default_value type mismatch for Float"};
  }
  return {true, "=" + std::to_string(value), ""};
}

AttrDefaultParseResult ParseBoolDefault(const nlohmann::json &j) {
  bool value = false;
  if (!TryGetJsonValue(j, value)) {
    return {false, "", "default_value type mismatch for Bool"};
  }
  return {true, std::string("=") + (value ? "true" : "false"), ""};
}

AttrDefaultParseResult ParseStringDefault(const nlohmann::json &j) {
  std::string value;
  if (!TryGetJsonValue(j, value)) {
    return {false, "", "default_value type mismatch for String"};
  }
  return {true, "=\"" + value + "\"", ""};
}

AttrDefaultParseResult ParseTypeDefault(const nlohmann::json &j) {
  std::string value;
  if (!TryGetJsonValue(j, value)) {
    return {false, "", "default_value type mismatch for Type"};
  }
  return {true, "=ge::" + value, ""};
}

AttrDefaultParseResult ParseTensorDefault(const nlohmann::json &j) {
  std::string value;
  if (!TryGetJsonValue(j, value)) {
    return {false, "", "default_value type mismatch for Tensor"};
  }
  if (value != "Tensor()") {
    return {false, "", "only \"Tensor()\" is supported for Tensor default_value"};
  }
  return {true, "=EsMakeUnique<ge::Tensor>(ge::Tensor())", ""};
}

AttrDefaultParseResult ParseListIntDefault(const nlohmann::json &j) {
  std::vector<int64_t> values;
  if (!TryGetJsonValue(j, values)) {
    return {false, "", "default_value type mismatch for ListInt"};
  }
  return {true, "={" + JoinNumbers(values) + "}", ""};
}

AttrDefaultParseResult ParseListFloatDefault(const nlohmann::json &j) {
  std::vector<double> values;
  if (!TryGetJsonValue(j, values)) {
    return {false, "", "default_value type mismatch for ListFloat"};
  }
  return {true, "={" + JoinNumbers(values) + "}", ""};
}

AttrDefaultParseResult ParseListBoolDefault(const nlohmann::json &j) {
  std::vector<bool> values;
  if (!TryGetJsonValue(j, values)) {
    return {false, "", "default_value type mismatch for ListBool"};
  }
  return {true, "={" + JoinBools(values) + "}", ""};
}

AttrDefaultParseResult ParseListTypeDefault(const nlohmann::json &j) {
  std::vector<std::string> values;
  if (!TryGetJsonValue(j, values)) {
    return {false, "", "default_value type mismatch for ListType"};
  }
  std::vector<std::string> typed_values;
  typed_values.reserve(values.size());
  for (const auto &value : values) {
    typed_values.emplace_back("ge::" + value);
  }
  return {true, "={" + JoinStrings(typed_values) + "}", ""};
}

AttrDefaultParseResult ParseListListIntDefault(const nlohmann::json &j) {
  std::vector<std::vector<int64_t>> values;
  if (!TryGetJsonValue(j, values)) {
    return {false, "", "default_value type mismatch for ListListInt"};
  }
  std::vector<std::string> groups;
  groups.reserve(values.size());
  for (const auto &inner : values) {
    groups.emplace_back("{" + JoinNumbers(inner) + "}");
  }
  return {true, "={" + JoinStrings(groups) + "}", ""};
}

AttrDefaultParseResult ParseListStringDefault(const nlohmann::json &j) {
  std::vector<std::string> values;
  if (!TryGetJsonValue(j, values)) {
    return {false, "", "default_value type mismatch for ListString"};
  }
  std::vector<std::string> quoted_values;
  quoted_values.reserve(values.size());
  for (const auto &value : values) {
    quoted_values.emplace_back("\"" + value + "\"");
  }
  return {true, "={" + JoinStrings(quoted_values) + "}", ""};
}

using DefaultExprParser = AttrDefaultParseResult (*)(const nlohmann::json &);

const std::map<std::string, ParamCxxKind> &GetHistoryTypeParamKindTable() {
  static const std::map<std::string, ParamCxxKind> kTable = {
    {"Int", ParamCxxKind::kInt64},
    {"Float", ParamCxxKind::kFloat},
    {"Bool", ParamCxxKind::kBool},
    {"String", ParamCxxKind::kCString},
    {"Type", ParamCxxKind::kDataType},
    {"Tensor", ParamCxxKind::kTensorUniquePtr},
    {"ListInt", ParamCxxKind::kListIntRef},
    {"ListFloat", ParamCxxKind::kListFloatRef},
    {"ListBool", ParamCxxKind::kListBoolRef},
    {"ListType", ParamCxxKind::kListTypeRef},
    {"ListListInt", ParamCxxKind::kListListIntRef},
    {"ListString", ParamCxxKind::kListStringRef}
  };
  return kTable;
}

const std::map<std::string, DefaultExprParser> &GetDefaultExprParserTable() {
  static const std::map<std::string, DefaultExprParser> kTable = {
    {"Int", &ParseIntDefault},
    {"Float", &ParseFloatDefault},
    {"Bool", &ParseBoolDefault},
    {"String", &ParseStringDefault},
    {"Type", &ParseTypeDefault},
    {"Tensor", &ParseTensorDefault},
    {"ListInt", &ParseListIntDefault},
    {"ListFloat", &ParseListFloatDefault},
    {"ListBool", &ParseListBoolDefault},
    {"ListType", &ParseListTypeDefault},
    {"ListListInt", &ParseListListIntDefault},
    {"ListString", &ParseListStringDefault}
  };
  return kTable;
}

const std::map<std::string, ParamCxxKind> &GetIrScalarTypeParamKindTable() {
  static const std::map<std::string, ParamCxxKind> kTable = {
    {"VT_INT", ParamCxxKind::kInt64},
    {"VT_FLOAT", ParamCxxKind::kFloat},
    {"VT_BOOL", ParamCxxKind::kBool},
    {"VT_STRING", ParamCxxKind::kCString},
    {"VT_DATA_TYPE", ParamCxxKind::kDataType},
    {"VT_TENSOR", ParamCxxKind::kTensorUniquePtr}
  };
  return kTable;
}

const std::map<std::string, ParamCxxKind> &GetIrListTypeParamKindTable() {
  static const std::map<std::string, ParamCxxKind> kTable = {
    {"VT_LIST_INT", ParamCxxKind::kListIntRef},
    {"VT_LIST_FLOAT", ParamCxxKind::kListFloatRef},
    {"VT_LIST_BOOL", ParamCxxKind::kListBoolRef},
    {"VT_LIST_DATA_TYPE", ParamCxxKind::kListTypeRef},
    {"VT_LIST_LIST_INT", ParamCxxKind::kListListIntRef},
    {"VT_LIST_STRING", ParamCxxKind::kListStringRef}
  };
  return kTable;
}
}  // namespace

bool AttrTypeTraits::TryGetParamKindByHistoryType(const std::string &av_type, ParamCxxKind &kind) {
  const auto &table = GetHistoryTypeParamKindTable();
  const auto iter = table.find(av_type);
  if (iter == table.end()) {
    return false;
  }
  kind = iter->second;
  return true;
}

bool AttrTypeTraits::TryGetParamKindByIrTypeInfo(const char *av_type, bool is_list_type, ParamCxxKind &kind) {
  if (av_type == nullptr) {
    return false;
  }
  const auto &table = is_list_type ? GetIrListTypeParamKindTable() : GetIrScalarTypeParamKindTable();
  const auto iter = table.find(av_type);
  if (iter == table.end()) {
    return false;
  }
  kind = iter->second;
  return true;
}

AttrDefaultParseResult AttrTypeTraits::ParseDefaultExpr(const std::string &av_type, const std::string &json_value) {
  if (json_value.empty()) {
    return {false, "", "missing default_value"};
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(json_value);
  } catch (...) {
    return {false, "", "default_value is not valid json"};
  }
  const auto &parsers = GetDefaultExprParserTable();
  const auto iter = parsers.find(av_type);
  if (iter == parsers.end()) {
    return {false, "", "unsupported attr type '" + av_type + "' for default_value parsing"};
  }
  return iter->second(j);
}

AttrPassStrategy AttrTypeTraits::GetAttrPassStrategy(ParamCxxKind kind) {
  switch (kind) {
    case ParamCxxKind::kListIntRef:
    case ParamCxxKind::kListFloatRef:
      return AttrPassStrategy::kListDataAndSize;
    case ParamCxxKind::kListBoolRef:
      return AttrPassStrategy::kListBoolDataAndSize;
    case ParamCxxKind::kListListIntRef:
      return AttrPassStrategy::kListListIntDataSizeCounts;
    case ParamCxxKind::kListTypeRef:
      return AttrPassStrategy::kListTypeDataAndSize;
    case ParamCxxKind::kListStringRef:
      return AttrPassStrategy::kListStringDataAndSize;
    case ParamCxxKind::kDataType:
      return AttrPassStrategy::kDataTypeCast;
    case ParamCxxKind::kTensorUniquePtr:
      return AttrPassStrategy::kTensorRelease;
    default:
      return AttrPassStrategy::kDirect;
  }
}
}  // namespace history
}  // namespace es
}  // namespace ge
