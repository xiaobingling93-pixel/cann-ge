/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_DEFAULT_VALUE_POLICY_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_DEFAULT_VALUE_POLICY_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "history_registry_types.h"

namespace ge {
namespace es {
namespace history {
struct DefaultValuePolicy {
  std::unordered_set<std::string> defaultable_inputs;
  std::unordered_set<std::string> defaultable_attrs;
  std::unordered_set<std::string> force_required_inputs;

  bool HasInputDefault(const std::string &name) const {
    return defaultable_inputs.count(name) > 0;
  }
  bool HasAnyInputDefault() const {
    return !defaultable_inputs.empty();
  }
  bool IsInputForcedRequired(const std::string &name) const {
    return force_required_inputs.count(name) > 0;
  }
  bool IsInputEffectivelyOptional(const IrInput &input) const {
    if (input.type != kIrInputOptional) {
      return false;
    }
    return !IsInputForcedRequired(input.name);
  }
  bool HasAttrDefault(const std::string &name) const {
    return defaultable_attrs.count(name) > 0;
  }
};

inline DefaultValuePolicy BuildDefaultValuePolicy(const IrOpProto &proto,
                                                  const std::vector<std::string> *force_required_inputs = nullptr) {
  DefaultValuePolicy policy;
  if (force_required_inputs != nullptr) {
    policy.force_required_inputs.insert(force_required_inputs->begin(), force_required_inputs->end());
  }
  bool has_required_attr = false;
  for (const auto &attr : proto.attrs) {
    if (attr.required) {
      has_required_attr = true;
      policy.defaultable_attrs.clear();
    } else {
      policy.defaultable_attrs.insert(attr.name);
    }
  }
  if (has_required_attr) {
    return policy;
  }
  for (const auto &output : proto.outputs) {
    if (output.type == kIrOutputDynamic) {
      return policy;
    }
  }
  for (const auto &input : proto.inputs) {
    if (input.type == kIrInputOptional) {
      if (!policy.IsInputForcedRequired(input.name)) {
        policy.defaultable_inputs.insert(input.name);
      }
    } else {
      policy.defaultable_inputs.clear();
    }
  }
  return policy;
}
}  // namespace history
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GENERATOR_DEFAULT_VALUE_POLICY_H_
