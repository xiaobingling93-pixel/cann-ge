/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_AMBIGUITY_CHECKER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_AMBIGUITY_CHECKER_H_

#include <set>
#include <string>
#include <utility>

#include "overload_planner_types.h"

namespace ge {
namespace es {
namespace history {
class AmbiguityChecker {
 public:
  static std::pair<int, int> CallRange(const Signature &sig);
  static bool HasCallRangeOverlap(const Signature &a, const Signature &b);
  static bool HasPotentialAmbiguityByTypicalArgs(const Signature &a, const Signature &b);
  static std::set<std::string> TypicalTokens(ParamCxxKind kind);
};
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_AMBIGUITY_CHECKER_H_
