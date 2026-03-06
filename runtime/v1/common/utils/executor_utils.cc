/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/utils/executor_utils.h"
#include "common/math/math_util.h"
#include "graph/utils/tensor_utils.h"

#include <string>
#include <ostream>

namespace ge {
namespace {
constexpr int64_t kMemtypeHostCompileDependent = 1;
constexpr int64_t kMemtypeHostCompileIndependent = 2;
constexpr size_t kMaxTilingDataSize = 16UL * 1024UL;
std::string GetDataBufferInfo(const std::vector<ge::DataBuffer> &inputs) {
  std::ostringstream buf;
  buf << "[";
  for (auto &in : inputs) {
    buf << " (len: " << in.length << ", place: " << in.placement << " )";
  }
  buf << " ]";
  return buf.str();
}

Status GetAlignedValue(const size_t input, const size_t align_bytes, size_t &output) {
  if (align_bytes == 0U) {
    GELOGE(ge::PARAM_INVALID, "align_bytes is zero");
    return ge::PARAM_INVALID;
  }
  if (ge::CheckUint32AddOverflow(static_cast<uint32_t>(input),
                                 static_cast<uint32_t>((align_bytes - 1U))) != ge::SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "Padding size is beyond the UINT32_MAX, input = %zu, align_bytes = %zu.",
           input, align_bytes);
    return ge::PARAM_INVALID;
  }
  output = ((input + (align_bytes - 1U)) / align_bytes) * align_bytes;
  return ge::SUCCESS;
}
} // namespace

bool ExecutorUtils::HasHostMemInput(const OpDescPtr &op_desc) {
  GE_RT_FALSE_CHECK_NOTNULL(op_desc);
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_desc == nullptr) {
      continue;
    }
    int64_t mem_type = 0;
    (void)ge::AttrUtils::GetInt(*input_desc, ge::ATTR_NAME_PLACEMENT, mem_type);
    if ((mem_type == kMemtypeHostCompileIndependent) || (mem_type == kMemtypeHostCompileDependent)) {
      GELOGD("node[%s] input[%zu] has host mem", op_desc->GetName().c_str(), i);
      return true;
    }
  }
  return false;
}

// for non hybrid (dynamic) single op
Status ExecutorUtils::UpdateHostMemInputArgs(const std::vector<DataBuffer> &inputs,
                                             const OpTask &op_task,
                                             void *const io_base,
                                             const size_t io_size,
                                             std::vector<rtHostInputInfo_t> &host_inputs) {
  GE_CHECK_NOTNULL(op_task.GetOpdesc());
  GE_CHECK_NOTNULL(io_base);
  const auto &op_desc = op_task.GetOpdesc();
  size_t host_mem_data_offset = op_task.GetHostMemInputDataOffsetInIoAddr();
  if ((host_mem_data_offset + kMaxHostMemInputLen) > io_size) {
    GELOGE(PARAM_INVALID, "[Check] memory reserved for host memory input is not enough, offset = %zu,"
                          " io_addrs_size = %zu", host_mem_data_offset, io_size);
    return PARAM_INVALID;
  }
  const size_t align_bytes = op_task.GetInputAddrAlignBytes();
  size_t dst_len_left = kMaxHostMemInputLen;
  const vector_bit_t &input_is_const = op_task.GetOpdesc()->GetIsInputConst();
  size_t input_index = 0UL;
  size_t io_index = 0UL;
  for (size_t i = 0UL; i < op_task.GetOpdesc()->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr &tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      continue;
    }
    if ((i < input_is_const.size()) && input_is_const[i]) {
      io_index++;
      continue;
    }
    if (input_index >= inputs.size()) {
      GELOGE(PARAM_INVALID, "input index(%zu) >= inputs size(%zu)", input_index, inputs.size());
      return PARAM_INVALID;
    }
    if (inputs[input_index].placement == kHostMemType) {
      size_t aligned_len = 0U;
      GE_CHK_STATUS_RET(GetAlignedValue(inputs[input_index].length, align_bytes, aligned_len),
                        "get align value failed.");
      GE_CHECK_LE(io_index, io_size / sizeof(void *));
      GE_CHECK_LE(host_mem_data_offset, io_size);
      uintptr_t *const host_mem_input_addr =
          PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(io_base) + static_cast<uint64_t>(sizeof(void *) * io_index)));
      void *const data_addr = ValueToPtr(PtrToValue(io_base) + host_mem_data_offset);
      *host_mem_input_addr = static_cast<uintptr_t>(PtrToValue(data_addr));
      if (memcpy_s(data_addr, dst_len_left, inputs[input_index].data, inputs[input_index].length) != EOK) {
        GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][HostMemInputArgs]failed, dst length is %zu,"
                                                   " src length is %zu.", dst_len_left, inputs[input_index].length);
        REPORT_INNER_ERR_MSG("E19999", "update kernel args failed");
        return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      }
      rtHostInputInfo_t host_in = {};
      GE_CHECK_LE(host_mem_data_offset, std::numeric_limits<uint32_t>::max());
      GE_CHECK_LE(io_index, std::numeric_limits<uint32_t>::max() / sizeof(void *));
      host_in.addrOffset = static_cast<uint32_t>(sizeof(void *) * io_index);
      host_in.dataOffset = static_cast<uint32_t>(host_mem_data_offset);
      host_mem_data_offset += aligned_len; // No integer overflow
      GE_CHECK_GE(dst_len_left, aligned_len);
      dst_len_left -= aligned_len;
      host_inputs.emplace_back(std::move(host_in));
      GELOGD("Finish to copy host mem input[%zu]. size = %" PRIu64 ", task arg index = %zu", input_index,
             inputs[input_index].length, io_index);
    }
    input_index++;
    io_index++;
  }
  if (host_inputs.empty()) {
    GELOGE(GRAPH_FAILED, "[%s(%s)] host memory input(s) should be copied to io_base, but it(they) did not!!!"
           "inputs size:%zu, io_size:%zu, input_is_const:%s, inputs:%s, input_index:%zu, io_index:%zu,"
           "op_desc_input_num:%zu, align_bytes=%zu, io_base=%p",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), inputs.size(), io_size,
           ToString(input_is_const).c_str(), GetDataBufferInfo(inputs).c_str(), input_index, io_index,
           op_task.GetOpdesc()->GetAllInputsSize(), align_bytes, io_base);
    return GRAPH_FAILED;
  }
  return SUCCESS;
}

// for hybrid
Status ExecutorUtils::UpdateHostMemInputArgs(const hybrid::TaskContext &context,
                                             void *const io_addrs,
                                             const size_t io_addrs_size,
                                             const size_t host_mem_input_data_offset_in_args,
                                             std::vector<rtHostInputInfo_t> &host_inputs,
                                             const bool need_64b_aligned) {
  GE_CHECK_NOTNULL(io_addrs);
  if ((host_mem_input_data_offset_in_args + kMaxHostMemInputLen) > io_addrs_size) {
    GELOGE(PARAM_INVALID, "[Check] memory reserved for host memory input is not enough, offset = %zu,"
           " io_addrs_size = %zu", host_mem_input_data_offset_in_args, io_addrs_size);
    return PARAM_INVALID;
  }
  size_t host_mem_input_data_offset = host_mem_input_data_offset_in_args;
  size_t dst_length_left = kMaxHostMemInputLen;
  const size_t aligned_bytes = need_64b_aligned ? kAlignBytes64 : kAlignBytes4;
  for (int32_t i = 0; i < context.NumInputs(); ++i) {
    const auto input_data = context.GetInput(i);
    GE_CHECK_NOTNULL(input_data);
    if ((input_data->GetMemType() == MemStorageType::HOST_DDR) && (input_data->GetData() != nullptr)) {
      size_t aligned_size = 0U;
      GE_CHK_STATUS_RET(GetAlignedValue(input_data->GetSize(), aligned_bytes, aligned_size),
                        "get align value failed.");
      GE_CHECK_LE(host_mem_input_data_offset, io_addrs_size);
      GE_CHECK_LE(static_cast<size_t>(i), io_addrs_size / sizeof(void *));
      uint64_t *const host_mem_input_index = PtrAdd(PtrToPtr<void, uint64_t>(io_addrs),
                                                    io_addrs_size / sizeof(uint64_t), static_cast<size_t>(i));
      *host_mem_input_index = PtrToValue(io_addrs) + host_mem_input_data_offset;
      void *const host_mem_input_data_ptr = ValueToPtr(PtrToValue(io_addrs) + host_mem_input_data_offset);
      if (memcpy_s(host_mem_input_data_ptr, dst_length_left, input_data->GetData(), input_data->GetSize()) != EOK) {
        GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][Args]failed, dst length is %zu, src length is %zu.",
               dst_length_left, input_data->GetSize());
        REPORT_INNER_ERR_MSG("E19999", "update kernel args failed of %s.", context.GetNodeName());
        return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      }
      rtHostInputInfo_t host_input = {};
      GE_CHECK_LE(host_mem_input_data_offset, std::numeric_limits<uint32_t>::max());
      GE_CHECK_LE(static_cast<size_t>(i),
                  std::numeric_limits<uint32_t>::max() / sizeof(uint64_t));
      host_input.dataOffset = static_cast<uint32_t>(host_mem_input_data_offset);
      host_input.addrOffset = static_cast<uint32_t>(static_cast<size_t>(i) * sizeof(uint64_t));
      host_inputs.emplace_back(std::move(host_input));
      host_mem_input_data_offset += aligned_size; // No integer overflow
      GE_CHECK_GE(dst_length_left, aligned_size);
      dst_length_left -= aligned_size;
      GELOGD("Finish to copy host mem input[%d]. size = %zu", i, input_data->GetSize());
    }
  }
  if (host_inputs.empty()) {
    GELOGE(GRAPH_FAILED, "host memory input(s) should be copied to io_base, but it(they) did not!!!");
    return GRAPH_FAILED;
  }
  return SUCCESS;
}

Status ExecutorUtils::LoadAtomicWorkspace(const OpDescPtr &op_desc) {
  GeAttrValue::NAMED_ATTRS workspaces;
  if (!AttrUtils::GetNamedAttrs(op_desc, EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspaces)) {
    return SUCCESS;
  }
  std::vector<int64_t> value;
  const std::string &op_name = op_desc->GetName();
  (void)AttrUtils::GetListInt(workspaces, op_name, value);
  if (value.empty()) {
    return SUCCESS;
  }
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info = { {op_name, std::map<int64_t, int64_t>()} };

  std::map<int64_t, int64_t> &index_offset = workspace_info[op_name];
  for (size_t i = 0U; i < (value.size() - 1U); i += 2U) { // two sets of vector, parsing the key value of the map
    index_offset[value[i]] = value[i + 1U];
  }

  GE_CHK_BOOL_RET_STATUS(op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info), INTERNAL_ERROR,
                         "[Set][Attr:%s]fail for node:%s.", EXT_ATTR_ATOMIC_WORKSPACE_INFO.c_str(),
                         op_desc->GetName().c_str());
  return SUCCESS;
}

Status ExecutorUtils::InitAtomicAddrCleanIndices(const OpDescPtr &op_desc, std::vector<int32_t> &atomic_output_indices,
                                                 std::vector<int32_t> &atomic_workspace_indices) {
  std::vector<int64_t> output_indices;
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, output_indices);
  const std::map<std::string, std::map<int64_t, int64_t>> workspace_info =
      op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>());
  GE_CHK_BOOL_RET_STATUS((!output_indices.empty()) || (!workspace_info.empty()), INTERNAL_ERROR,
                         "[Check][Size][%s] atomic_output_indices and atomic_workspace_info must not be both empty.",
                         op_desc->GetName().c_str());

  for (const auto &iter : workspace_info) {
    for (const auto &info_iter : iter.second) {
      const int64_t workspace_index = info_iter.first;
      GELOGD("[%s] Adding workspace index [%" PRId64 "]", op_desc->GetName().c_str(), workspace_index);
      GE_CHECK_GE(workspace_index, 0);
      GE_CHECK_LE(workspace_index, INT32_MAX);
      atomic_workspace_indices.emplace_back(static_cast<int32_t>(workspace_index));
    }
  }

  for (const int64_t output_index : output_indices) {
    GELOGD("[%s] Adding output index [%" PRId64 "]", op_desc->GetName().c_str(), output_index);
    GE_CHECK_GE(output_index, 0);
    GE_CHECK_LE(output_index, INT32_MAX);
    atomic_output_indices.emplace_back(static_cast<int32_t>(output_index));
  }

  return SUCCESS;
}

bool ExecutorUtils::GetOpIndex(const domi::TaskDef &task_def, uint32_t &op_index) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  if (task_type == ModelTaskType::MODEL_TASK_KERNEL) {
    op_index = task_def.kernel().context().op_index();
  } else if (task_type == ModelTaskType::MODEL_TASK_KERNEL_EX) {
    op_index = task_def.kernel_ex().op_index();
  } else if (task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    op_index = task_def.kernel_with_handle().context().op_index();
  } else if (task_type == ModelTaskType::MODEL_TASK_FFTS_PLUS) {
    op_index = task_def.ffts_plus_task().op_index();
  } else {
    GELOGD("Skip task type: %d", static_cast<int32_t>(task_type));
    return false;
  }
  GELOGD("op_index = %u, task_type = %d.", op_index, task_type);
  return true;
}

/*
 * 静态shape复用二进制args组成：
 *   |inputs-addr|
 *   |outputs-addr|
 *   |workspaces-addr|
 *   |tiling-data-addr|
 *   |opt -- 可选arg(eg:overflow-addr)|
 *   |tiling data|
 */
Status ExecutorUtils::AssembleReuseBinaryArgs(const OpDescPtr &op_desc, optiling::utils::OpRunInfo &run_info,
                                              rtArgsEx_t &args_ex) {
  GE_CHECK_NOTNULL(op_desc);
  // 这里不要调用GetAllInputsSize，GetAllInputsSize是包含了可选输入
  const size_t io_num = op_desc->GetInputsSize() + op_desc->GetAllOutputsDescSize();
  const size_t workspace_num = op_desc->GetWorkspaceBytes().size();
  GE_CHK_STATUS_RET(ge::CheckUint32MulOverflow(static_cast<uint32_t>((io_num + workspace_num)),
      static_cast<uint32_t>(sizeof(uintptr_t))));
  args_ex.tilingAddrOffset = static_cast<uint32_t>((io_num + workspace_num) * sizeof(uintptr_t));
  constexpr size_t tiling_data_addr_size = sizeof(uintptr_t);
  const size_t overflow_addr_size = op_desc->HasAttr("globalworkspace_type") ? sizeof(uintptr_t) : 0U;
  args_ex.tilingDataOffset = static_cast<uint32_t>(args_ex.tilingAddrOffset +
      tiling_data_addr_size + overflow_addr_size);

  int64_t max_size = -1;
  if ((!ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_MAX_TILING_SIZE, max_size)) || (max_size < 0)) {
    GELOGD("No max tiling size in op_desc.");
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }
  const size_t max_tiling_size =
      (static_cast<size_t>(max_size) + sizeof(uintptr_t) - 1U) / sizeof(uintptr_t) * sizeof(uintptr_t);
  GELOGD("Max tiling size of %s is %zu.", op_desc->GetName().c_str(), max_tiling_size);
  const uint32_t args_size = args_ex.tilingDataOffset + static_cast<uint32_t>(max_tiling_size);
  GELOGI("Change args size from %u to %u.", args_ex.argsSize, args_size);
  args_ex.argsSize = args_size;
  const size_t tiling_data_size = run_info.GetAllTilingData().str().size();
  if ((tiling_data_size == 0U) || (max_tiling_size == 0U)) {
    GELOGD("Node: %s has no tiling data.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  args_ex.hasTiling = true;

  GE_CHECK_GE(max_tiling_size, tiling_data_size);
  const aclrtMemcpyKind
      memcpy_kind = op_desc->HasAttr(ge::ATTR_SINGLE_OP_SCENE) ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_HOST_TO_DEVICE;
  void *const tiling_data_addr = ge::ValueToPtr(ge::PtrToValue(args_ex.args) + args_ex.tilingDataOffset);
  void *const tiling_addr_offset = ge::ValueToPtr(ge::PtrToValue(args_ex.args) + args_ex.tilingAddrOffset);
  GE_CHK_RT_RET(aclrtMemcpy(tiling_addr_offset, sizeof(uintptr_t), &tiling_data_addr,
      sizeof(uintptr_t), memcpy_kind));
  GE_CHK_RT_RET(aclrtMemcpy(tiling_data_addr, max_tiling_size, run_info.GetAllTilingData().str().data(),
      tiling_data_size, memcpy_kind));

  GELOGD("Update args of %s, block dim: %u, tiling key: %" PRIu64 ", tilingAddrOffset: %u,"
         "tilingDataOffset: %u, max_tiling_size: %zu, arg_size: %u.",
         op_desc->GetName().c_str(), run_info.GetBlockDim(), run_info.GetTilingKey(),
         args_ex.tilingAddrOffset, args_ex.tilingDataOffset, max_tiling_size, args_ex.argsSize);
  return SUCCESS;
}
}
