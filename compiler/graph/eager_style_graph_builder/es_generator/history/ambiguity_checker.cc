/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ambiguity_checker.h"

#include <algorithm>
#include <map>

namespace ge {
namespace es {
namespace history {
namespace {
bool HasTokenIntersection(const std::set<std::string> &a, const std::set<std::string> &b) {
  for (const auto &token : a) {
    if (b.count(token) > 0U) {
      return true;
    }
  }
  return false;
}

const std::map<ParamCxxKind, std::set<std::string>> &GetTypicalTokenTable() {
  static const std::map<ParamCxxKind, std::set<std::string>> kTable = {
    {ParamCxxKind::kEsTensorLikeRef, {"tensor", "nullptr", "0", "0.0"}},
    {ParamCxxKind::kTensorHolderRef, {"tensor"}},
    {ParamCxxKind::kTensorHoldersVecRef, {"tensor_vec"}},
    {ParamCxxKind::kDataType, {"dtype"}},
    {ParamCxxKind::kTensorUniquePtr, {"tensor_ptr", "nullptr"}},
    {ParamCxxKind::kListIntRef, {"list_int"}},
    {ParamCxxKind::kListFloatRef, {"list_float"}},
    {ParamCxxKind::kListBoolRef, {"list_bool"}},
    {ParamCxxKind::kListTypeRef, {"list_type"}},
    {ParamCxxKind::kListListIntRef, {"list_list_int"}},
    {ParamCxxKind::kListStringRef, {"list_string"}},
    {ParamCxxKind::kGraphUniquePtr, {"graph_ptr", "nullptr"}},
    {ParamCxxKind::kGraphsVec, {"graph_vec"}},
    {ParamCxxKind::kGraphBuilderRef, {"graph_builder"}},
    {ParamCxxKind::kGraphBuilderPtr, {"graph_builder", "nullptr"}},
    {ParamCxxKind::kInt64, {"int", "number", "0"}},
    {ParamCxxKind::kFloat, {"float", "number", "0", "0.0"}},
    {ParamCxxKind::kBool, {"bool", "0"}},
    {ParamCxxKind::kCString, {"string", "\"xx\""}},
    {ParamCxxKind::kNullptrT, {"nullptr"}}
  };
  return kTable;
}
}  // namespace

std::pair<int, int> AmbiguityChecker::CallRange(const Signature &sig) {
  int required = 0;
  for (const auto &param : sig.params) {
    if (!param.has_default) {
      ++required;
    }
  }
  return {required, static_cast<int>(sig.params.size())};
}

bool AmbiguityChecker::HasCallRangeOverlap(const Signature &a, const Signature &b) {
  const auto range_a = CallRange(a);
  const auto range_b = CallRange(b);
  return !(range_a.second < range_b.first || range_b.second < range_a.first);
}

bool AmbiguityChecker::HasPotentialAmbiguityByTypicalArgs(const Signature &a, const Signature &b) {
  if (!HasCallRangeOverlap(a, b)) {
    return false;
  }

  const auto range_a = CallRange(a);
  const auto range_b = CallRange(b);
  const int min_arity = std::max(range_a.first, range_b.first);
  const int max_arity = std::min(range_a.second, range_b.second);
  for (int arity = min_arity; arity <= max_arity; ++arity) {
    bool all_positions_overlap = true;
    for (int i = 0; i < arity; ++i) {
      if (static_cast<size_t>(i) >= a.params.size() || static_cast<size_t>(i) >= b.params.size()) {
        all_positions_overlap = false;
        break;
      }
      const auto tokens_a = TypicalTokens(a.params[static_cast<size_t>(i)].kind);
      const auto tokens_b = TypicalTokens(b.params[static_cast<size_t>(i)].kind);
      if (tokens_a.empty() || tokens_b.empty() || !HasTokenIntersection(tokens_a, tokens_b)) {
        all_positions_overlap = false;
        break;
      }
    }
    if (all_positions_overlap) {
      return true;
    }
  }
  return false;
}

std::set<std::string> AmbiguityChecker::TypicalTokens(ParamCxxKind kind) {
  const auto &table = GetTypicalTokenTable();
  const auto iter = table.find(kind);
  return iter == table.end() ? std::set<std::string>{} : iter->second;
}
}  // namespace history
}  // namespace es
}  // namespace ge
