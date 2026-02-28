/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_OVERLOAD_PLANNER_TYPES_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_OVERLOAD_PLANNER_TYPES_H_

#include <string>
#include <vector>

namespace ge {
namespace es {
namespace history {
enum class ParamCxxKind {
  kEsTensorLikeRef,
  kTensorHolderRef,
  kTensorHoldersVecRef,
  kDataType,
  kTensorUniquePtr,
  kListIntRef,
  kListFloatRef,
  kListBoolRef,
  kListTypeRef,
  kListListIntRef,
  kListStringRef,
  kGraphUniquePtr,
  kGraphsVec,
  kGraphBuilderRef,
  kGraphBuilderPtr,
  kInt64,
  kFloat,
  kBool,
  kCString,
  kNullptrT
};

enum class ParamRole {
  kUnknown,
  kInput,
  kDynamicOutputNum,
  kSubgraph,
  kOwnerBuilder,
  kAttr
};

struct Param {
  ParamCxxKind kind;
  std::string name;
  bool has_default = false;
  std::string default_expr;
  ParamRole role = ParamRole::kUnknown;
  std::string ir_name;
};

struct Signature {
  std::vector<Param> params;
  bool is_deleted = false;
  bool is_deprecated = false;
  std::string deprecate_msg;
};

enum class WarningCode {
  kFallbackToA0,
  kUpgradeToA1,
  kUpgradeToA2,
  kUnsupportedAttrType,
  kInvalidAttrDefaultValue
};

struct Warning {
  WarningCode code = WarningCode::kFallbackToA0;
  std::string detail;
};

struct OverloadPlan {
  std::vector<Signature> signatures;
  std::vector<Warning> warnings;
};
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_OVERLOAD_PLANNER_TYPES_H_
