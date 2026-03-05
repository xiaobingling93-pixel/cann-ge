/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/codegen/om2_model_utils.h"
#include <cinttypes>
#include "common/checker.h"
#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/math/math_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "runtime/mem.h"

namespace ge {
constexpr uint64_t kSessionScopeMemoryMask = 0x100000000UL;
constexpr int32_t kSessionNoReuse = 1;

uint64_t Om2ModelUtils::GetWorkspaceMemTypeByPriority(const bool is_p2p_memory, const bool is_l1_memory,
                                                      const bool is_ub_memory,
                                                      const bool session_scope_memory) {
  if (is_p2p_memory) {
    return RT_MEMORY_P2P_DDR;
  }
  if (is_l1_memory) {
    return RT_MEMORY_L1;
  }
  if (is_ub_memory) {
    return kRtMemoryUB;
  }
  if (session_scope_memory) {
    return kSessionScopeMemoryMask | RT_MEMORY_HBM;
  }
  return RT_MEMORY_HBM;
}

bool Om2ModelUtils::ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                                     const int64_t size) {
  if (CheckInt64AddOverflow(offset, size) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[OM2] Int64 %" PRId64 " and %" PRId64 " addition can result in overflow!", offset, size);
    return false;
  }
  const int64_t mem_range = offset + size;
  if (total_size < static_cast<uint64_t>(mem_range)) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) memory out of range, offset:%" PRId64 ", size:"
                       "%" PRId64 ", exceed total size:%" PRIu64 ".", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str(), offset, size, total_size);
    GELOGE(OUT_OF_MEMORY, "[OM2][Check][Param]Node:%s(%s) memory out of range, offset:%" PRId64
           ", size:%" PRId64 ", exceed total size:%" PRIu64 ".",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), offset, size, total_size);
    return false;
  }
  return true;
}

Status Om2ModelUtils::GetValidatedTensorMemType(const GeTensorDescPtr &tensor_desc,
                                                const std::vector<int64_t> &mem_types,
                                                size_t index,
                                                uint64_t &memory_type) {
  int64_t tensor_mem_type = -1;
  const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
  memory_type = RT_MEMORY_DEFAULT;
  if (tensor_has_mem_type) {
    memory_type = static_cast<uint64_t>(tensor_mem_type);
  } else if (mem_types.size() > index) {
    memory_type = static_cast<uint64_t>(mem_types[index]);
  }

  if (memory_type != RT_MEMORY_HBM && memory_type != RT_MEMORY_DEFAULT) {
    REPORT_INNER_ERR_MSG("E19999", "Workspace mem type[%" PRIu64 "] is invalid, must be RT_MEMORY_HBM.", memory_type);
    GELOGE(ge::FAILED,
           "[OM2] Workspace mem type[%" PRIu64 "] is invalid, must be RT_MEMORY_HBM.",
           memory_type);
    return FAILED;
  }
  return SUCCESS;
}

Status Om2ModelUtils::GetOrCreateInputVarName(TaskDistributionContext &context, size_t input_idx,
                                              size_t non_const_idx, const std::vector<int64_t> &input_offsets,
                                              std::string &input_ptr_name, std::vector<AstNode *> &input_nodes) {
  const int64_t current_op_id = context.op_desc->GetId();
  const int64_t op_index = context.op_index;
  GE_ASSERT_TRUE(context.op_id_to_input_edges.find(current_op_id) != context.op_id_to_input_edges.end(),
                 "[OM2] Current op_id %" PRId64 " not found in op_id_to_input_edges", current_op_id);
  const OpInputEdges &current_edges = context.op_id_to_input_edges.at(current_op_id);
  GE_ASSERT_TRUE(input_idx < current_edges.input_op_ids.size(),
                 "[OM2] Input index %zu out of range for op_id %" PRId64, input_idx, current_op_id);
  const int64_t src_op_id = current_edges.input_op_ids[input_idx];
  const int32_t src_anchor_idx = current_edges.input_anchor_indices[input_idx];
  if (src_op_id == kInvalidOpId) {
    GELOGE(FAILED, "[OM2] Input %zu of op %s(%" PRId64 ") is unconnected optional input, not supported",
           input_idx, context.op_desc->GetName().c_str(), current_op_id);
    return FAILED;
  }
  GE_ASSERT_TRUE(context.op_id_to_input_edges.find(src_op_id) != context.op_id_to_input_edges.end(),
                 "[OM2] Source op_id %" PRId64 " not found in op_id_to_input_edges", src_op_id);
  OpInputEdges &src_edges = context.op_id_to_input_edges.at(src_op_id);
  GE_ASSERT_TRUE(src_anchor_idx < static_cast<int32_t>(src_edges.output_var_names.size()),
                 "[OM2] Source anchor index %d out of range for src_op_id %" PRId64, src_anchor_idx, src_op_id);
  if (!src_edges.output_var_names[src_anchor_idx].empty()) {
    input_ptr_name = src_edges.output_var_names[src_anchor_idx];
  } else {
    input_ptr_name = "op" + std::to_string(op_index) + "_input" + std::to_string(non_const_idx);
    input_nodes.push_back(RAW_CODE_STMT(context.ast_ctx,
      "  auto " + input_ptr_name + " = GET_ADDR(total_dev_mem_ptr_, " +
      std::to_string(input_offsets[non_const_idx]) + ");"));
    src_edges.output_var_names[src_anchor_idx] = input_ptr_name;
  }
  return SUCCESS;
}

Status Om2ModelUtils::GenInputAddrCode(TaskDistributionContext &context, std::vector<AddrGenInfo> &input_addr_nodes) {
  GE_CHECK_NOTNULL_EXEC(context.op_desc, return FAILED);
  GELOGD("[OM2] Start GenInputAddrCode: op_name[%s]", context.op_desc->GetName().c_str());
  const size_t inputs_size = context.op_desc->GetInputsSize();
  const vector_bit_t &v_is_input_const = context.op_desc->GetIsInputConst();
  const auto &input_offsets = context.op_desc->GetInputOffset();
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(context.op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  const bool check_failed = has_mem_type_attr && (v_memory_type.size() != inputs_size);
  if (check_failed) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
                       context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[OM2][Check][Param] Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
           context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());
    return FAILED;
  }

  input_addr_nodes.reserve(inputs_size);
  size_t non_const_index = 0UL;
  for (size_t i = 0U; i < context.op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = context.op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_IF_BOOL_EXEC(tensor_desc == nullptr,
                    GELOGD("[OM2] Op: %s, Index: %zu, has no input", context.op_desc->GetName().c_str(), i);
                    continue);
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return FAILED);

    input_addr_nodes.emplace_back();
    input_addr_nodes.back().mem_type = Om2MemoryAppType::kMemoryTypeFix;
    std::string input_ptr_name;
    // fileconstant和var场景一阶段暂不支持
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      // Add weights address to input
      int64_t data_offset = 0;
      GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));
      int64_t weight_size = 0;
      // The reason why GetTensorSizeInBytes is used here is that the weight is allocated based on the size of
      // TensorData in function AdjustConstWeightSize. and the size is zero when the tensor is empty.
      GE_CHK_STATUS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weight_size));
      GE_IF_BOOL_EXEC(!ValidateMemRange(context.op_desc, context.runtime_param.weight_size, data_offset, weight_size),
                      return FAILED);
      GE_ASSERT_TRUE(context.weight_offset_to_varname.find(data_offset) != context.weight_offset_to_varname.end(),
                     "[OM2] Const input offset %" PRId64 " not found, op %s, index %zu", data_offset,
                     context.op_desc->GetName().c_str(), i);
      input_ptr_name = context.weight_offset_to_varname[data_offset];
    } else {
      uint64_t memory_type = RT_MEMORY_DEFAULT;
      GE_ASSERT_SUCCESS(GetValidatedTensorMemType(tensor_desc, v_memory_type, i, memory_type));
      GE_IF_BOOL_EXEC(non_const_index >= input_offsets.size(), return FAILED);
      const int64_t input_offset = input_offsets[non_const_index];
      if (!ValidateMemRange(context.op_desc, context.runtime_param.mem_size, input_offset, 0)) {
        return FAILED;
      }
      if (context.model_io_offsets.find(input_offset) != context.model_io_offsets.end()) {
        input_addr_nodes.back().mem_type = Om2MemoryAppType::kMemoryTypeModelIo;
      }
      GE_ASSERT_SUCCESS(GetOrCreateInputVarName(context, i, non_const_index, input_offsets,
                                                   input_ptr_name, input_addr_nodes.back().nodes));
    }
    input_addr_nodes.back().var_name = input_ptr_name;
    non_const_index++;
  }
  GELOGD("[OM2] Input addrs code is successfully generated: op_name[%s].", context.op_desc->GetName().c_str());
  return SUCCESS;
}

Status Om2ModelUtils::GenOutputAddrCode(TaskDistributionContext &context, std::vector<AddrGenInfo> &output_addr_nodes,
                                        const bool has_optional_addr) {
  GE_CHECK_NOTNULL_EXEC(context.op_desc, return FAILED);
  GELOGD("[OM2] Start GetOutputDataAddrs: op_name[%s]", context.op_desc->GetName().c_str());
  const int64_t op_index = context.op_index;
  const size_t outputs_size = context.op_desc->GetOutputsSize();
  const int64_t current_op_id = context.op_desc->GetId();
  const std::vector<int64_t> v_output_offset = context.op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(
      v_output_offset.size() != outputs_size,
      GELOGW("[OM2] Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
      return FAILED);
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(context.op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    REPORT_INNER_ERR_MSG("E19999", "[OM2] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
                       context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[OM2][Check][Param] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
           context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());
    return FAILED;
  }
  GE_ASSERT_TRUE(context.op_id_to_input_edges.find(current_op_id) != context.op_id_to_input_edges.end(),
                 "[OM2] Current op_id %" PRId64 " not found in op_id_to_input_edges", current_op_id);
  OpInputEdges &current_edges = context.op_id_to_input_edges.at(current_op_id);
  GE_ASSERT_TRUE(current_edges.output_var_names.size() == outputs_size,
                 "[OM2] output_var_names size %zu != outputs_size %zu for op_id %" PRId64,
                 current_edges.output_var_names.size(), outputs_size, current_op_id);

  output_addr_nodes.reserve(outputs_size);
  // fileconstant和var场景一阶段暂不支持
  for (size_t i = 0U; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = context.op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    // skip some addr
    if (tensor_desc == nullptr) {
      GELOGW("[OM2] Op: %s, Index: %zu, Tensor Desc is null", context.op_desc->GetName().c_str(), i);
      continue;
    }
    output_addr_nodes.emplace_back();
    output_addr_nodes.back().mem_type = Om2MemoryAppType::kMemoryTypeFix;
    std::string output_ptr_name = "op" + std::to_string(op_index) + "_output" + std::to_string(i);
    int32_t calc_type = 0;
    (void)AttrUtils::GetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (calc_type == static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY)) {
      if (has_optional_addr) {
        output_addr_nodes.back().nodes.push_back(RAW_CODE_STMT(context.ast_ctx, "  auto " + output_ptr_name + " = nullptr;"));
        output_addr_nodes.back().var_name = output_ptr_name;
        current_edges.output_var_names[i] = output_ptr_name;  // Record even for optional output
      }
      GELOGD("[OM2] %s is an optional output, has option addr:%d.",
        context.op_desc->GetName().c_str(), static_cast<int32_t>(has_optional_addr));
      continue;
    }

    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return FAILED);
    uint64_t memory_type = RT_MEMORY_DEFAULT;
    GE_ASSERT_SUCCESS(GetValidatedTensorMemType(tensor_desc, v_memory_type, i, memory_type));
    if (!ValidateMemRange(context.op_desc, context.runtime_param.mem_size, v_output_offset[i], 0)) {
      return FAILED;
    }
    if (context.model_io_offsets.find(v_output_offset[i]) != context.model_io_offsets.end()) {
      output_addr_nodes.back().mem_type = Om2MemoryAppType::kMemoryTypeModelIo;
    }
    output_addr_nodes.back().nodes.push_back(RAW_CODE_STMT(context.ast_ctx,
      "  auto " + output_ptr_name + " = GET_ADDR(total_dev_mem_ptr_, " +
      std::to_string(v_output_offset[i]) + ");"));
    current_edges.output_var_names[i] = output_ptr_name;  // Record output variable name
    output_addr_nodes.back().var_name = output_ptr_name;
  }
  GELOGD("[OM2] Output addrs code is successfully generated: op_name[%s].", context.op_desc->GetName().c_str());
  return SUCCESS;
}

Status Om2ModelUtils::GenWorkspaceAddrsCode(TaskDistributionContext &context,
                                            std::vector<AddrGenInfo> &workspace_addr_nodes) {
  GE_CHECK_NOTNULL_EXEC(context.op_desc, return FAILED);
  GELOGD("[OM2] Start GenWorkspaceAddrCode: op_name[%s].", context.op_desc->GetName().c_str());
  const std::vector<int64_t> v_workspace_offset = context.op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = context.op_desc->GetWorkspaceBytes();
  if (v_workspace_offset.size() != v_workspace_bytes.size()) {
    GELOGW("[OM2] v_workspace_offset.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_offset.size(),
           v_workspace_bytes.size());
    return FAILED;
  }

  vector_bit_t workspace_reuse_flag;
  const bool has_workspace_reuse = AttrUtils::GetListBool(context.op_desc, "workspace_reuse_flag", workspace_reuse_flag);
  std::vector<int64_t> v_memory_type;
  std::vector<int64_t> workspace_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(context.op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  const bool has_mem_type_workspace =
      AttrUtils::GetListInt(context.op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_memory_type);
  if ((has_mem_type_attr && (v_memory_type.size() != v_workspace_offset.size())) ||
      (has_mem_type_workspace && (workspace_memory_type.size() != v_workspace_offset.size()))) {
    REPORT_INNER_ERR_MSG("E19999",
                       "[OM2] Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
                       "same, op:%s(%s), check invalid",
                       TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(),
                       ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(), workspace_memory_type.size(), v_workspace_offset.size(),
                       context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID,
           "[OM2][Check][Param] Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
           "same, op:%s(%s), check invalid",
           TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
           workspace_memory_type.size(), v_workspace_offset.size(), context.op_desc->GetName().c_str(),
           context.op_desc->GetType().c_str());
    return FAILED;
  }
  std::vector<int32_t> workspace_no_reuse_scope;
  const bool has_workspace_no_reuse_scope =
      AttrUtils::GetListInt(context.op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
  workspace_addr_nodes.reserve(v_workspace_bytes.size());
  for (size_t i = 0U; i < v_workspace_bytes.size(); ++i) {
    // Temporary solution, the aicpu workspace of multiple images cannot be shared.
    const bool aicpu_work_space =
        (has_workspace_reuse && (i < workspace_reuse_flag.size()) && (!workspace_reuse_flag[i]));
    // 要确认一下aicpu的怎么弄，是否有个单独的内存池aicpu_mem_mall
    if (aicpu_work_space) {
      GELOGE(FAILED, "[OM2] Aicpu task not support append workspace addrs for now.");
      return FAILED;
    }
    const bool session_scope_memory = (has_workspace_no_reuse_scope) && (i < workspace_no_reuse_scope.size()) &&
        (workspace_no_reuse_scope[i] == kSessionNoReuse);
    const bool is_p2p_memory =
        has_mem_type_workspace && (static_cast<uint64_t>(workspace_memory_type[i]) == RT_MEMORY_P2P_DDR);
    const bool is_l1_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == RT_MEMORY_L1);
    const bool is_ub_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == kRtMemoryUB);
    const uint64_t memory_type = GetWorkspaceMemTypeByPriority(is_p2p_memory, is_l1_memory, is_ub_memory,
                                                               session_scope_memory);
    // 未来支持所有类型内存后移除
    if (memory_type != RT_MEMORY_HBM) {
      REPORT_INNER_ERR_MSG("E19999", "Workspace mem type[%" PRIu64 "] is invalid, must be RT_MEMORY_HBM.", memory_type);
      GELOGE(ge::FAILED,
             "[OM2] Workspace mem type[%" PRIu64 "] is invalid, must be RT_MEMORY_HBM.",
             memory_type);
      return FAILED;
    }
    if (!ValidateMemRange(context.op_desc, context.runtime_param.mem_size, v_workspace_offset[i], 0)) {
      return FAILED;
    }

    const auto op_index = context.op_index;
    std::string ws_ptr_name = "op" + std::to_string(op_index) + "_ws" + std::to_string(i);
    workspace_addr_nodes.emplace_back();
    workspace_addr_nodes.back().nodes.push_back(RAW_CODE_STMT(context.ast_ctx,
      "  auto " + ws_ptr_name + " = GET_ADDR(total_dev_mem_ptr_, " +
      std::to_string(v_workspace_offset[i]) + ");"));
    workspace_addr_nodes.back().var_name = ws_ptr_name;
    workspace_addr_nodes.back().mem_type = Om2MemoryAppType::kMemoryTypeFix;
  }
  GELOGD("[OM2] Workspace addrs code is successfully generated: op_name[%s].", context.op_desc->GetName().c_str());
  return SUCCESS;
}
} // namespace ge
