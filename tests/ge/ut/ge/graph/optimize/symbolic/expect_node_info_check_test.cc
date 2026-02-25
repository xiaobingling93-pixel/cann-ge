/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expect_node_info_check_test.h"
#include "attribute_group/attr_group_symbolic_desc.h"
namespace ge {
bool ExpectNodeInfo::ExpectShapeCheck(const gert::SymbolShape &real_shape) const {
  if (GetExpectSymbolOutputShape().empty()) {
    GE_ASSERT_TRUE(real_shape.IsScalar());
  }
  GE_ASSERT_EQ(real_shape.GetDimNum(), GetExpectSymbolOutputShape().size());
  GE_ASSERT_TRUE(real_shape.GetDims() == GetExpectSymbolOutputShape());
  return true;
}
bool ExpectNodeInfo::ExpectGuardInfoCheck(const std::vector<SymbolCheckInfo> real_guard) const {
  if (GetExpectGuardInfos().empty()) {
    return true;
  }
  GE_ASSERT_EQ(real_guard.size(), GetExpectGuardInfos().size());
  for (auto &iter : real_guard) {
    GE_ASSERT_TRUE(GetExpectGuardInfos().find(std::string(iter.expr.Serialize().get())) !=
                   GetExpectGuardInfos().end());
  }
  return true;
}

bool ExpectNodeInfo::ExpectAssertInfoCheck(const std::vector<SymbolCheckInfo> real_assert) const {
  if (GetExpectAssertInfos().empty()) {
    return true;
  }
  GE_ASSERT_EQ(real_assert.size(), GetExpectAssertInfos().size());
  for (auto &iter : real_assert) {
    GE_ASSERT_TRUE(GetExpectAssertInfos().find(std::string(iter.expr.Serialize().get())) !=
                   GetExpectAssertInfos().end());
  }
  return true;
}

bool ExpectNodeInfo::ExpectSymbolValCheck(const std::vector<ge::Expression> *real_val) const {
  if (GetExpectSymbolicValue().empty()) {
    return true;
  }
  GE_ASSERT_NOTNULL(real_val);
  GE_ASSERT_EQ(real_val->size(), GetExpectSymbolicValue().size());
  GE_ASSERT_TRUE(*real_val == GetExpectSymbolicValue());
  return true;
}

Status RunSymbolInferenceTest(const ComputeGraphPtr &cg, const std::vector<ExpectNodeInfo> &node_info_vec,
                              const std::vector<ge::GeTensor> &input_vec) {
  if (!input_vec.empty()) {
    GE_ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(cg, input_vec), ge::SUCCESS);
  }

  SymbolicShapeInference ssi;
  GE_ASSERT_EQ(ssi.Infer(cg), ge::SUCCESS);
  auto attr1 = cg->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(attr1);
  ShapeEnvGuarder guard(attr1);
  for (const auto &node : node_info_vec) {
    auto node_ptr = cg->FindNode(node.GetNodeName());
    if (node_ptr == nullptr) {
      node_ptr = cg->FindFirstNodeMatchType(node.GetNodeName());
    }
    GE_ASSERT_NOTNULL(node_ptr);
    auto op_desc = node_ptr->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto attr = op_desc->GetOutputDesc(0).template GetAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr);
    auto symbol_shape = attr->symbolic_tensor.GetOriginSymbolShape();
    GE_ASSERT_TRUE(node.ExpectShapeCheck(symbol_shape));
    auto shape_env_attr = cg->GetAttrsGroup<ShapeEnvAttr>();
    if (shape_env_attr != nullptr) {
      const auto guard_infos = shape_env_attr->GetAllSymbolCheckInfos();
      GE_ASSERT_TRUE(node.ExpectGuardInfoCheck(guard_infos));
      const auto assert_infos = shape_env_attr->GetAllSymbolAssertInfos();
      GE_ASSERT_TRUE(node.ExpectAssertInfoCheck(assert_infos));
      const auto symbol_value = attr->symbolic_tensor.GetSymbolicValue();
      GE_ASSERT_TRUE(node.ExpectSymbolValCheck(symbol_value));
    }
  }
  return SUCCESS;
}
}
