/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_SYMBOLIC_SHAPE_FRAME_TEST_H
#define AIR_CXX_SYMBOLIC_SHAPE_FRAME_TEST_H

#include <memory>
#include <utility>
#include "graph/utils/graph_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/optimize/symbolic/shape_env_guarder.h"

namespace ge {
struct DataInfo {
  Format format;
  DataType dt;
  std::vector<int64_t> shape;
};
class ExpectNodeInfoCheckBase {
 public:
  ExpectNodeInfoCheckBase(std::string node_name,
                          std::vector<Expression> expect_symbol_output_shape,
                          std::set<std::string> expect_guard_infos,
                          std::set<std::string> expect_assert_infos,
                          std::vector<Expression> expect_symbolic_value)
      : node_name_(std::move(node_name)),
        expect_symbol_output_shape_(std::move(expect_symbol_output_shape)),
        expect_guard_infos_(std::move(expect_guard_infos)),
        expect_assert_infos_(std::move(expect_assert_infos)),
        expect_symbolic_value_(std::move(expect_symbolic_value)) {}

  std::string GetNodeName() const {
    return node_name_;
  }
  std::vector<Expression> GetExpectSymbolOutputShape() const {
    return expect_symbol_output_shape_;
  }
  std::set<std::string> GetExpectGuardInfos() const {
    return expect_guard_infos_;
  }
  std::set<std::string> GetExpectAssertInfos() const {
    return expect_assert_infos_;
  }
  std::vector<Expression> GetExpectSymbolicValue() const {
    return expect_symbolic_value_;
  }
  virtual bool ExpectShapeCheck(const gert::SymbolShape &real_shape) const = 0;
  virtual bool ExpectGuardInfoCheck(std::vector<SymbolCheckInfo> real_guard) const = 0;
  virtual bool ExpectAssertInfoCheck(std::vector<SymbolCheckInfo> real_assert) const = 0;
  virtual bool ExpectSymbolValCheck(const std::vector<ge::Expression> * real_val) const = 0;
  virtual ~ExpectNodeInfoCheckBase() = default;

 private:
  std::string node_name_;
  std::vector<Expression> expect_symbol_output_shape_;
  std::set<std::string> expect_guard_infos_;
  std::set<std::string> expect_assert_infos_;
  std::vector<Expression> expect_symbolic_value_;
};
}  // namespace ge
#endif  // AIR_CXX_SYMBOLIC_SHAPE_FRAME_TEST_H
