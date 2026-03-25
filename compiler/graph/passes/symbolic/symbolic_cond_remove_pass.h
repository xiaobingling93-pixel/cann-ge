/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_COND_GUARD_PASS_H
#define GE_GRAPH_PASSES_COND_GUARD_PASS_H

#include "graph/passes/base_pass.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/graph_utils.h"
#include "attribute_group/attr_group_shape_env.h"

namespace ge {
class InputValueForCondSource : public Source {
 public:
  InputValueForCondSource(int32_t input_data_idx, int32_t cond_value)
      : input_data_idx_(input_data_idx), cond_value_(cond_value){}
  [[nodiscard]] std::string GetSourceStr() const override;
 private:
  int32_t input_data_idx_;
  int32_t cond_value_;
};

class SymbolicCondRemovePass : public BaseNodePass {
 public:
  explicit SymbolicCondRemovePass(std::vector<GeTensor> graph_inputs) : graph_inputs_(std::move(graph_inputs)) {}
  Status Run(NodePtr &node) override;
 private:
  Status GetCondIndexSymbol(const NodePtr &cond_input, Expression &cond_index_sym, const std::string &node_type);
  std::vector<GeTensor> graph_inputs_;
  // key: data_idx, value: new sym
  std::unordered_map<int32_t, Expression> created_sym{};
};
} // namespace ge

#endif // GE_GRAPH_PASSES_COND_GUARD_PASS_H