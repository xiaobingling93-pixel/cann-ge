/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "symbolic_cond_remove_pass.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_util.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "attribute_group/attr_group_shape_env.h"
#include "framework/common/types.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/control_flow_and_stream/cond_remove_pass.h"
#include "graph/expression/const_values.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
const int kCondInputIndex = 0;  // If/Case节点的cond输入索引

void GetIfCondIndexValue(const Expression &cond_index_sym, uint32_t &cond_index_val) {
  if (EXPECT_SYMBOL_GT(cond_index_sym, sym::kSymbolZero)) {
    cond_index_val = 1U;
  } else {
    cond_index_val = 0U;
  }
}

Status GetCaseCondIndexValue(const NodePtr &node, const Expression &cond_index_sym, uint32_t &cond_index_val) {
  uint32_t subgraph_size = node->GetOpDesc()->GetSubgraphInstanceNames().size();
  GE_ASSERT_TRUE(subgraph_size > 0);
  if (EXPECT_SYMBOL_GE(cond_index_sym, Symbol(subgraph_size))) {
    cond_index_val = subgraph_size - 1;
    return SUCCESS;
  }
  for (uint32_t subgraph_idx = 0; subgraph_idx < subgraph_size; subgraph_idx++) {
    if (EXPECT_SYMBOL_EQ(cond_index_sym, Symbol(subgraph_idx))) {
      cond_index_val = subgraph_idx;
      return SUCCESS;
    }
  }
  return FAILED;
}
} // namespace

std::string InputValueForCondSource::GetSourceStr() const {
  return R"([&]() -> int64_t {
      const auto *tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
      if (tensor == nullptr) {
        return -1;
      }
      const uint8_t *data_ptr = tensor->GetData<uint8_t>();
      // 字节数
      size_t tensor_size = tensor->GetSize();
      const auto type = tensor->GetDataType();
      switch (type) {
        case ge::DT_STRING:
          return (tensor_size - (sizeof(ge::StringHead) + 1) > 0) ? 1 : 0;
        case ge::DT_BOOL:
          return static_cast<int64_t>(*reinterpret_cast<const bool *>(data_ptr));
        case ge::DT_FLOAT:
          return static_cast<int64_t>(*reinterpret_cast<const float *>(data_ptr));
        case ge::DT_DOUBLE:
          return static_cast<int64_t>(*reinterpret_cast<const double *>(data_ptr));
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_HIFLOAT8:
        case ge::DT_FLOAT8_E5M2:
        case ge::DT_FLOAT8_E4M3FN:
          return static_cast<int64_t>(*data_ptr);
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
          return static_cast<int64_t>(*reinterpret_cast<const int16_t *>(data_ptr));
        case ge::DT_INT32:
          return static_cast<int64_t>(*reinterpret_cast<const int32_t *>(data_ptr));
        case ge::DT_UINT32:
          return static_cast<int64_t>(*reinterpret_cast<const int32_t *>(data_ptr));
        case ge::DT_INT64:
        case ge::DT_UINT64:
          return *reinterpret_cast<const int64_t *>(data_ptr);
        default:
          return static_cast<int64_t>(*data_ptr);
      }
    }())";
}

Status SymbolicCondRemovePass::GetCondIndexSymbol(const NodePtr &cond_input, Expression &cond_index_sym, const std::string &node_type) {
  auto op_desc = cond_input->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  int32_t data_index = -1;
  GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, "index", data_index));
  GE_ASSERT_TRUE(data_index < static_cast<int32_t>(graph_inputs_.size()));
  const auto &tensor = graph_inputs_.at(data_index);
  const auto &ge_tensor_desc = tensor.GetTensorDesc();
  auto ori_shape = ge_tensor_desc.IsOriginShapeInitialized() ? tensor.GetTensorDesc().GetOriginShape() :
                   ge_tensor_desc.GetShape();
  if (ori_shape.IsScalar()) {
    // data连接到多个if/case, 可以复用
    if (created_sym.find(data_index) != created_sym.end()) {
      cond_index_sym = created_sym[data_index];
    } else {
      auto cond_value = CondRemovePass::GetCondIndex(&tensor);
      if (cond_value < 0) {
        if (kCaseOpTypes.find(node_type) != kCaseOpTypes.end()) {
          GELOGI("Get cond index[%d] from node[%s][%s] success, cond_index is negative.", cond_value, cond_input->GetNamePtr(), cond_input->GetTypePtr());
          return UNSUPPORTED;
        }
      }
      GELOGD("Get cond index[%d] from node[%s][%s] success, ", cond_value, cond_input->GetNamePtr(), cond_input->GetTypePtr());
      auto value_source = MakeShared<InputValueForCondSource>(data_index, cond_value);
      auto shape_env = GetCurShapeEnvContext();
      GE_ASSERT_NOTNULL(shape_env);
      GELOGD("Get shape env success.");
      cond_index_sym = GetCurShapeEnvContext()->CreateSymbol(cond_value, value_source);
      created_sym[data_index] = cond_index_sym;
    }
  } else {
    auto output_desc = op_desc->GetOutputDesc(0U);
    auto sym_attr = output_desc.GetAttrsGroup<SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(sym_attr);
    cond_index_sym = sym_attr->symbolic_tensor.GetOriginSymbolShape().GetSymbolShapeSize();
  }
  return SUCCESS;
}

Status SymbolicCondRemovePass::Run(NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  // 不是if/case节点直接返回
  if (!SymbolicInferUtil::IsSupportCondNode(node)) {
    return SUCCESS;
  }
  // 获取if/case条件输入的节点
  auto cond_input_node = SymbolicInferUtil::GetCondInput(node);
  GE_ASSERT_NOTNULL(cond_input_node);
  GELOGD("Get cond[%s] input[%s] success.", node->GetNamePtr(), cond_input_node->GetNamePtr());
  // 条件输入的节点不是DATA不需要处理
  if (!OpTypeUtils::IsDataNode(cond_input_node->GetType())) {
    GELOGI("CondNode[%s] input[%s] is not Data.", node->GetNamePtr(), cond_input_node->GetNamePtr());
    return SUCCESS;
  }
  const auto &node_type = node->GetType();

  Expression cond_index_sym;

  auto ret = GetCondIndexSymbol(cond_input_node, cond_index_sym, node_type);
  if (ret == UNSUPPORTED) {
    GELOGI("condIndex of cond[%s] is negative, skip.", node->GetNamePtr());
    return SUCCESS;
  } else if (ret != SUCCESS) {
    GELOGI("Get condIndex of cond[%s] failed.", node->GetNamePtr());
    return ret;
  }

  uint32_t cond_index_val = 0;
  ComputeGraphPtr chosen_graph = nullptr;
  if (kCaseOpTypes.find(node_type) != kCaseOpTypes.end()) {
    GE_ASSERT_SUCCESS(GetCaseCondIndexValue(node, cond_index_sym, cond_index_val));
    GE_ASSERT_SUCCESS(CondRemovePass::GetCaseChosenBranch(node, cond_index_val, chosen_graph));
  } else {
    GetIfCondIndexValue(cond_index_sym, cond_index_val);
    GE_ASSERT_SUCCESS(CondRemovePass::GetIfChosenBranch(node, cond_index_val, chosen_graph));
  }

  GE_ASSERT_SUCCESS(CondRemovePass::RemoveDeadCondLink(kCondInputIndex, node));
  GE_ASSERT_SUCCESS(CondRemovePass::ReplaceIfCaseNodeWithPartitioncall(node, chosen_graph));
  GE_ASSERT_SUCCESS(IsolateAndDeleteNode(node, std::vector<int32_t>()));

  return SUCCESS;
}

REG_PASS_OPTION("SymbolicCondRemovePass").LEVELS(OoLevel::kO1).SWITCH_OPT(ge::OO_DEAD_CODE_ELIMINATION);
} // namespace ge