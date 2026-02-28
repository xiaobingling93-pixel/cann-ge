/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "warning_formatter.h"

namespace ge {
namespace es {
namespace history {
namespace {
struct WarningDetailParts {
  std::string context;
  std::string reason;
};

WarningDetailParts SplitWarningDetail(const std::string &detail) {
  WarningDetailParts parts;
  if (detail.empty()) {
    return parts;
  }
  const auto pos = detail.rfind("; ");
  if (pos == std::string::npos) {
    parts.context = detail;
    return parts;
  }
  parts.context = detail.substr(0, pos);
  parts.reason = detail.substr(pos + 2);
  return parts;
}

std::string BuildWarningPrefix(const std::string &detail) {
  const auto parts = SplitWarningDetail(detail);
  std::string prefix;
  if (!parts.context.empty()) {
    prefix += "Context: " + parts.context + ". ";
  }
  if (!parts.reason.empty()) {
    prefix += "Cause: " + parts.reason + ". ";
  }
  return prefix;
}
} // namespace

std::string FormatWarning(const Warning &warning) {
  const auto prefix = BuildWarningPrefix(warning.detail);
  switch (warning.code) {
    case WarningCode::kFallbackToA0:
      return prefix + "Action: fall back to A0 single-signature plan to avoid ambiguous or incompatible overloads.";
    case WarningCode::kUpgradeToA1:
      return prefix +
          "Action: Try0 overloads (legacy + full signature) are ambiguous, so switch to A1: "
          "force new inputs required and add nullptr-guard overloads.";
    case WarningCode::kUpgradeToA2:
      return prefix +
          "Action: A1 overloads are still ambiguous, so switch to A2: "
          "force new inputs as TensorHolder and add nullptr-guard overloads.";
    case WarningCode::kUnsupportedAttrType:
      return prefix + "Action: unsupported attr type in C++ signature; emit std::nullptr_t placeholder for this attr.";
    case WarningCode::kInvalidAttrDefaultValue:
      return prefix + "Action: attr default_value cannot be parsed; emit this attr without default in C++ signature.";
    default:
      return "unknown overload planning warning";
  }
}
} // namespace history
} // namespace es
} // namespace ge
