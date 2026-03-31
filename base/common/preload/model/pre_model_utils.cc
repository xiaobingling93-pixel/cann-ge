/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/model/pre_model_utils.h"
#include "framework/common/tlv/pre_model_desc.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "base/err_msg.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr uint64_t kSessionScopeMemoryMask = 0x100000000UL;
const std::string kWorkSpace = "workspace";
constexpr int32_t kMemoryGlobalType = 2;
uint64_t GetWorkspaceMemTypeByPriority(const bool is_p2p_memory, const bool is_l1_memory, const bool is_ub_memory,
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
}  // namespace
Status PreModelUtils::GetInputConstAddrOffset(const ConstOpDescPtr &op_desc, const PreRuntimeParam &model_param,
                                              const GeTensorDescPtr &tensor_desc, const int64_t input_offset,
                                              KernelArgsParam &arg_param) {
  int32_t tensor_type = 0;
  const bool ret = ge::AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEMORY_SCOPE, tensor_type);
  if (ret && tensor_type == kMemoryGlobalType) {
    // 当前针对Nano形态适配FIFO类型算子  Global内存区域优先级最高
    arg_param.type = static_cast<uint8_t>(KERNEL_ARG_UPDATE_TYPE_ADDR);
    arg_param.offset.need_refresh = false;
    arg_param.offset.offset = static_cast<uint64_t>(input_offset);
    arg_param.para = static_cast<uint64_t>(KERNEL_ARG_UPADTE_ADDR_TYPE_ARGS);
    GELOGI("[IMAS]GetInputConstAddrOffset type global memory");
    return SUCCESS;
  }
  // Add weights address to input
  int64_t data_offset = 0;
  GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));
  int64_t weight_size = 0;
  // The reason why GetTensorSizeInBytes is used here is that the weight is allocated based on the size of
  // TensorData in function AdjustConstWeightSize. and the size is zero when the tensor is empty.
  GE_CHK_STATUS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weight_size));
  GE_ASSERT_TRUE(ValidateMemRange(op_desc, model_param.weight_size, data_offset, weight_size));
  arg_param.type = static_cast<uint8_t>(KERNEL_ARG_UPDATE_TYPE_ADDR);
  arg_param.offset.need_refresh = true;
  arg_param.offset.offset = static_cast<uint64_t>(data_offset);
  arg_param.para = static_cast<uint64_t>(KERNEL_ARG_UPADTE_ADDR_TYPE_WEIGHT);
  GELOGI("[IMAS]GetInputConstAddrOffset type[C] size[%ld] ", weight_size);
  return SUCCESS;
}

std::vector<std::pair<uint64_t, uint32_t>> PreModelUtils::GetInputDataAddrOffset(
    const PreRuntimeParam &model_param, const ConstOpDescPtr &op_desc, std::vector<KernelArgsParam> &args_param,
    std::vector<uint64_t> &args_offset_values) {
  std::vector<uint32_t> index_to_valid_idx;
  return GetInputDataAddrOffset(model_param, op_desc, args_param, args_offset_values, index_to_valid_idx);
}

std::vector<std::pair<uint64_t, uint32_t>> PreModelUtils::GetInputDataAddrOffset(
    const PreRuntimeParam &model_param, const ConstOpDescPtr &op_desc, std::vector<KernelArgsParam> &args_param,
    std::vector<uint64_t> &args_offset_values, std::vector<uint32_t> &index_to_valid_idx) {
  std::vector<std::pair<uint64_t, uint32_t>> v_input_data_addr;
  KernelArgsParam arg_param;
  const size_t inputs_size = op_desc->GetInputsSize();
  const std::vector<int64_t> v_input_offset = op_desc->GetInputOffset();
  const vector_bit_t &v_is_input_const = op_desc->GetIsInputConst();

  size_t non_const_index = 0UL;
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  const bool check_failed = has_mem_type_attr && (v_memory_type.size() != inputs_size);
  if (check_failed) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size, op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_input_data_addr;
  }

  index_to_valid_idx.resize(op_desc->GetAllInputsSize());
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_IF_BOOL_EXEC(tensor_desc == nullptr, GELOGD("Op: %s, Index: %zu, has no input", op_desc->GetName().c_str(), i);
                    continue;)
    index_to_valid_idx[i] = static_cast<uint32_t>(non_const_index);
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      GE_CHK_STATUS_EXEC(
          GetInputConstAddrOffset(op_desc, model_param, tensor_desc, v_input_offset[non_const_index], arg_param),
          return {});
      GELOGI("[IMAS]GetInputDataAddrs type[C] name[%s] input[%zu] data_offset[%lu]",
              op_desc->GetName().c_str(), i, arg_param.offset.offset);
      RefreshData(arg_param, args_param, args_offset_values, v_input_data_addr);
      non_const_index++;
      continue;
    }

    GE_IF_BOOL_EXEC(non_const_index >= v_input_offset.size(), break);
    const int64_t input_offset = v_input_offset[non_const_index];
    non_const_index++;

    int64_t tensor_mem_type = -1;
    const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
    uint64_t mem_type(RT_MEMORY_DEFAULT);
    if (tensor_has_mem_type) {
      mem_type = static_cast<uint64_t>(tensor_mem_type);
    } else if (v_memory_type.size() > i) {
      mem_type = static_cast<uint64_t>(v_memory_type[i]);
    } else {
      GELOGI("use default memory type");
    }
    const NodeMemInfo node_mem_info{mem_type, op_desc, i, "input", tensor_size, input_offset};
    if (RefreshAddressByMemType(model_param, node_mem_info, arg_param) != SUCCESS) {
      GELOGE(FAILED, "failed refresh addr.");
      return {};
    }
    RefreshData(arg_param, args_param, args_offset_values, v_input_data_addr);
  }
  return v_input_data_addr;
}

std::vector<std::pair<uint64_t, uint32_t>> PreModelUtils::GetOutputDataAddrOffset(
    const PreRuntimeParam &model_param, const ConstOpDescPtr &op_desc, std::vector<KernelArgsParam> &args_param,
    std::vector<uint64_t> &args_offset_values) {
  std::vector<std::pair<uint64_t, uint32_t>> v_output_data_addr;
  KernelArgsParam arg_param;
  const size_t outputs_size = op_desc->GetOutputsSize();
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_data_addr);
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size, op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_output_data_addr;
  }

  for (size_t i = 0U; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    // skip some addr
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }
    if (TensorUtils::IsMemorySizeCalcTypeAlwaysEmpty(*tensor_desc)) {
      GELOGD("%s is an optional output, the address don't need to be saved.", tensor_desc->GetName().c_str());
      continue;
    }
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});

    int64_t tensor_mem_type = -1;
    const bool tensor_has_mem_type = AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, tensor_mem_type);
    uint64_t mem_type(RT_MEMORY_DEFAULT);
    if (tensor_has_mem_type) {
      mem_type = static_cast<uint64_t>(tensor_mem_type);
    } else if (has_mem_type_attr) {
      mem_type = static_cast<uint64_t>(v_memory_type[i]);
    } else {  // 暂无处理
    }
    const NodeMemInfo node_mem_info{mem_type, op_desc, i, "output", tensor_size, v_output_offset[i]};
    if (RefreshAddressByMemType(model_param, node_mem_info, arg_param) != SUCCESS) {
      GELOGE(FAILED, "failed refresh addr.");
      return {};
    }
    RefreshData(arg_param, args_param, args_offset_values, v_output_data_addr);
  }
  return v_output_data_addr;
}

std::vector<std::pair<uint64_t, uint32_t>> PreModelUtils::GetWorkspaceDataAddrOffset(
    const PreRuntimeParam &model_param, const ConstOpDescPtr &op_desc, std::vector<KernelArgsParam> &args_param,
    std::vector<uint64_t> &args_offset_values) {
  std::vector<std::pair<uint64_t, uint32_t>> v_workspace_data_addr;
  KernelArgsParam arg_param;
  const std::vector<int64_t> v_workspace_offset = op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_offset.size() != v_workspace_bytes.size()) {
    GELOGW("v_workspace_offset.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_offset.size(),
           v_workspace_bytes.size());
    return v_workspace_data_addr;
  }

  vector_bit_t workspace_reuse_flag;
  const bool has_workspace_reuse = AttrUtils::GetListBool(op_desc, "workspace_reuse_flag", workspace_reuse_flag);
  std::vector<int64_t> v_memory_type;
  std::vector<int64_t> workspace_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  const bool has_mem_type_workspace =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_memory_type);
  if ((has_mem_type_attr && (v_memory_type.size() != v_workspace_offset.size())) ||
      (has_mem_type_workspace && (workspace_memory_type.size() != v_workspace_offset.size()))) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
                       "same, op:%s(%s), check invalid",
                       TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(),
                       ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(), workspace_memory_type.size(), v_workspace_offset.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID,
           "[Check][Param] Attr:%s, memory_type.size:%zu and %s, memory_type.size:%zu and workspaces num:%zu should be "
           "same, op:%s(%s), check invalid",
           TVM_ATTR_NAME_WORKSPACE_TYPE.c_str(), v_memory_type.size(), ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
           workspace_memory_type.size(), v_workspace_offset.size(), op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return v_workspace_data_addr;
  }
  std::vector<int32_t> workspace_no_reuse_scope;
  const bool has_workspace_no_reuse_scope =
      AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
  for (size_t i = 0U; i < v_workspace_bytes.size(); ++i) {
    // Temporary solution, the aicpu workspace of multiple images cannot be shared.
    const bool aicpu_work_space =
        (has_workspace_reuse && (i < workspace_reuse_flag.size()) && (!workspace_reuse_flag[i]));
    if (aicpu_work_space) {
      GELOGW("not support aicpu work space, pls check.");
      continue;
    }
    const bool session_scope_memory = (has_workspace_no_reuse_scope) && (i < workspace_no_reuse_scope.size());
    const bool is_p2p_memory =
        has_mem_type_workspace && (static_cast<uint64_t>(workspace_memory_type[i]) == RT_MEMORY_P2P_DDR);
    const bool is_l1_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == RT_MEMORY_L1);
    const bool is_ub_memory = has_mem_type_attr && (static_cast<uint64_t>(v_memory_type[i]) == kRtMemoryUB);
    const uint64_t mem_type = GetWorkspaceMemTypeByPriority(is_p2p_memory, is_l1_memory, is_ub_memory,
                                                            session_scope_memory);
    const NodeMemInfo node_mem_info{mem_type, op_desc, i, kWorkSpace, v_workspace_bytes[i], v_workspace_offset[i]};
    if (RefreshAddressByMemType(model_param, node_mem_info, arg_param) != SUCCESS) {
      GELOGE(FAILED, "failed refresh addr.");
      return {};
    }
    RefreshData(arg_param, args_param, args_offset_values, v_workspace_data_addr);
  }

  return v_workspace_data_addr;
}

Status PreModelUtils::RefreshAddressByMemType(const PreRuntimeParam &model_param, const NodeMemInfo &node_mem_info,
                                              KernelArgsParam &arg_param) {
  arg_param.type = static_cast<uint8_t>(KERNEL_ARG_UPDATE_TYPE_ADDR);
  arg_param.offset.need_refresh = true;
  arg_param.offset.offset = static_cast<uint64_t>(node_mem_info.logical_offset_);
  switch (node_mem_info.mem_type_) {
    case RT_MEMORY_L1:  // fusion
    case kRtMemoryUB:
      arg_param.para = static_cast<uint64_t>(KERNEL_ARG_UPADTE_ADDR_TYPE_L1);
      break;
    case RT_MEMORY_TS:  // not support
    case kSessionScopeMemoryMask | RT_MEMORY_HBM:
    case RT_MEMORY_HOST:
    case RT_MEMORY_HOST_SVM:
    case RT_MEMORY_P2P_DDR: {
      arg_param.offset.need_refresh = false;
      arg_param.para = static_cast<uint64_t>(KERNEL_ARG_UPDATE_TYPE_P2P);
      break;
    }
    case RT_MEMORY_HBM:
    case RT_MEMORY_L2:  // l2 also malloc hbm for datadump
    case RT_MEMORY_DEFAULT:
      // size can be 0 and need update addr for input and output
      if ((node_mem_info.size_ <= 0) && (node_mem_info.io_type_ == kWorkSpace)) {
        arg_param.offset.offset = 0U;
        arg_param.para = static_cast<uint64_t>(KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE);
        return SUCCESS;
      }
      // The input node_mem_info.size_ and peer output size may be not consecutive, therefore, the tensor_size is not
      // been checked.
      if (!ValidateMemRange(node_mem_info.op_desc_, model_param.mem_size, node_mem_info.logical_offset_, 0)) {
        return FAILED;
      }
      arg_param.para = static_cast<uint64_t>(KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE);
      break;
    default:
      return FAILED;
  }
  return SUCCESS;
}

void PreModelUtils::RefreshData(const KernelArgsParam &arg_param, std::vector<KernelArgsParam> &args_param,
                                std::vector<uint64_t> &args_offset_values,
                                std::vector<std::pair<uint64_t, uint32_t>> &v_input_data_addr) {
  args_param.push_back(arg_param);
  args_offset_values.push_back(arg_param.offset.offset);
  v_input_data_addr.push_back(std::make_pair(arg_param.offset.offset, arg_param.para));
}

bool PreModelUtils::ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                                     const int64_t size) {
  if (CheckInt64AddOverflow(offset, size) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Int64 %ld and %ld addition can result in overflow!", offset, size);
    return false;
  }
  const int64_t mem_range = offset + size;
  if (total_size < static_cast<uint64_t>(mem_range)) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Node:%s(%s) memory out of range, offset:%" PRId64
                       ", size:"
                       "%" PRId64 ", exceed total size:%" PRIu64 ".",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), offset, size, total_size);
    GELOGE(OUT_OF_MEMORY, "[Check][Param]Node:%s(%s) memory out of range, offset:%ld, size:%ld, exceed total size:%lu.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), offset, size, total_size);
    return false;
  }
  return true;
}

void PreModelUtils::InitRuntimeParams(const GeModelPtr &ge_model, PreRuntimeParam &runtime_param) {
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, runtime_param.mem_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, runtime_param.logic_mem_base);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, runtime_param.weight_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, runtime_param.logic_weight_base);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, runtime_param.zero_copy_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_STREAM_NUM, runtime_param.stream_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_EVENT_NUM, runtime_param.event_num);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_LABEL_NUM, runtime_param.label_num);

  auto memory_info_vec = GetAllMemoryTypeSize(ge_model);
  for (auto &i : memory_info_vec) {
    if (i.memory_type == RT_MEMORY_HBM) {
      continue;
    }
    runtime_param.memory_infos[i.memory_type] = std::move(i);
  }
}

std::vector<PreMemInfo> PreModelUtils::GetAllMemoryTypeSize(const GeModelPtr &ge_model) {
  std::vector<PreMemInfo> all_mem_info;
  PreMemInfo default_mem_info{};
  int64_t zero_copy_size = 0;
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, default_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size);
  if (zero_copy_size <= default_mem_info.memory_size) {
    default_mem_info.memory_size -= zero_copy_size;
  }
  default_mem_info.memory_type = RT_MEMORY_HBM;
  (void)all_mem_info.emplace_back(std::move(default_mem_info));

  PreMemInfo p2p_mem_info{};
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, p2p_mem_info.memory_size);
  p2p_mem_info.memory_type = RT_MEMORY_P2P_DDR;
  p2p_mem_info.memory_key = "_p";
  (void)all_mem_info.emplace_back(std::move(p2p_mem_info));

  PreMemInfo session_scope_mem_info{};
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, session_scope_mem_info.memory_size);
  session_scope_mem_info.memory_type = (kSessionScopeMemoryMask | RT_MEMORY_HBM);
  (void)all_mem_info.emplace_back(std::move(session_scope_mem_info));

  PreMemInfo host_mem_info{};
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, host_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, host_mem_info.logic_memory_base);
  host_mem_info.memory_type = RT_MEMORY_HOST;
  host_mem_info.memory_key = "_h";
  (void)all_mem_info.emplace_back(std::move(host_mem_info));

  PreMemInfo host_svm_mem_info{};
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, host_svm_mem_info.memory_size);
  (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, host_svm_mem_info.logic_memory_base);
  host_svm_mem_info.memory_type = RT_MEMORY_HOST_SVM;
  host_svm_mem_info.memory_key = "_svm";
  (void)all_mem_info.emplace_back(std::move(host_svm_mem_info));
  return all_mem_info;
}
std::vector<int64_t> PreModelUtils::GetInputSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_input_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_size);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  for (size_t i = 0U; i < inputs_size; i++) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op:%s, Index:%zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
                    GELOGI("Get size from TensorDesc failed, op:%s, input index:%zu.", op_desc->GetName().c_str(), i);
                    continue);

    GELOGI("GetInputSize op:%s, index:%zu, size:%" PRId64 "", op_desc->GetName().c_str(), i, tensor_size);
    v_input_size.push_back(tensor_size);
  }
  GELOGI("v_input_size size : %zu", v_input_size.size());
  return v_input_size;
}

std::vector<int64_t> PreModelUtils::GetOutputSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_output_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_size);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const std::vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, output=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_size);

  for (size_t i = 0U; i < outputs_size; i++) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGW("Op:%s, Index:%zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
                    GELOGI("Get size from TensorDesc failed, op:%s, output index:%zu.", op_desc->GetName().c_str(), i);
                    continue);

    GELOGI("GetOutputSize op:%s, index:%zu, size:%" PRId64 "", op_desc->GetName().c_str(), i, tensor_size);
    v_output_size.push_back(tensor_size);
  }
  GELOGI("v_output_size size : %zu", v_output_size.size());
  return v_output_size;
}

std::vector<int64_t> PreModelUtils::GetWorkspaceSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_workspace_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_size);

  const std::vector<int64_t> v_workspace_num = op_desc->GetWorkspace();
  const std::vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_num.size() != v_workspace_bytes.size()) {
    GELOGW("workspace_num[%zu] != v_workspace_bytes[%zu]", v_workspace_num.size(), v_workspace_bytes.size());
    return v_workspace_size;
  }
  GELOGI("v_workspace_bytes size : %zu", v_workspace_bytes.size());
  return v_workspace_bytes;
}

std::vector<int64_t> PreModelUtils::GetWeightSize(const ConstOpDescPtr &op_desc) {
  std::vector<int64_t> v_weight_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weight_size);

  // const op, get weight directly
  const std::string type_name = op_desc->GetType();
  if ((type_name == CONSTANT) || (type_name == CONSTANTOP)) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weight_size.push_back(static_cast<int64_t>(TensorUtils::GetWeightSize(weight)));
    }
    GELOGI("v_weight_size : %zu", v_weight_size.size());
    return v_weight_size;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetAllInputsSize();
  const std::vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0U; i < inputs_size; i++) {
    if ((i < v_is_input_const.size()) && (v_is_input_const[i])) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      if (tensor_desc == nullptr) {
        GELOGW("Op:%s, Index:%zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      int64_t tensor_size = 0;
      (void)TensorUtils::GetSize(*tensor_desc, tensor_size);
      v_weight_size.push_back(tensor_size);
    }
  }
  GELOGI("v_weight_size : %zu", v_weight_size.size());
  return v_weight_size;
}
}  // namespace ge