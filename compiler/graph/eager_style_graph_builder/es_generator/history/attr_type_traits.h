/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_ATTR_TYPE_TRAITS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_ATTR_TYPE_TRAITS_H_

#include <string>

#include "overload_planner_types.h"

namespace ge {
namespace es {
namespace history {
struct AttrDefaultParseResult {
  bool success = false;
  std::string default_expr;
  std::string error;
};

enum class AttrPassStrategy {
  kDirect,
  kListDataAndSize,
  kListBoolDataAndSize,
  kListListIntDataSizeCounts,
  kListTypeDataAndSize,
  kListStringDataAndSize,
  kDataTypeCast,
  kTensorRelease
};

class AttrTypeTraits {
 public:
  static bool TryGetParamKindByHistoryType(const std::string &av_type, ParamCxxKind &kind);
  static bool TryGetParamKindByIrTypeInfo(const char *av_type, bool is_list_type, ParamCxxKind &kind);
  static AttrDefaultParseResult ParseDefaultExpr(const std::string &av_type, const std::string &json_value);
  static AttrPassStrategy GetAttrPassStrategy(ParamCxxKind kind);
};
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_ATTR_TYPE_TRAITS_H_
