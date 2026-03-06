/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/known_subgraph/davinci_model_tracing.h"

#include <sstream>
#include "exe_graph/runtime/kernel_context.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/known_subgraph/davinci_model_kernel.h"
#include "runtime/stream.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
namespace kernel {
namespace {
constexpr int32_t kInvalidStream = -1;
}

std::vector<std::string> PrintModelCreate(const KernelContext * context) {
  auto davinci_model = context->GetOutputPointer<ge::DavinciModel>(0);
  if (davinci_model == nullptr) {
    return {"davinci_model is nullptr"};
  }
  std::stringstream ss;
  const auto rt_streams = davinci_model->GetStreamList();
  ss << "davinci model init finish. runtime_model_id: " << davinci_model->GetRuntimeModelId();
  ss << ", rt stream num: " << rt_streams.size() << ", list: [";
  for (const auto &stream : rt_streams) {
    int32_t rt_stream_id = kInvalidStream;
    (void)rtGetStreamId(stream, &rt_stream_id);
    ss << rt_stream_id << ", ";
  }
  ss << "]";
  const auto weight_tensor = context->GetInputPointer<GertTensorData>(2U);
  if (weight_tensor != nullptr) {
    ss << ", weight_base: " << ge::PtrToValue(weight_tensor->GetAddr())
       << ", weight_size: " << weight_tensor->GetSize();
  }
  ss << ".";
  return {ss.str()};
}

std::vector<std::string> PrintWorkspaces(const KernelContext * context) {
  auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsCommon::kDavinciModel));
  const auto workspace_num = context->GetInputPointer<size_t>(static_cast<int32_t>(UpdateWorkspaces::kWorkspacesNum));
  if ((davinci_model == nullptr) || (workspace_num == nullptr)) {
    return {"davinci_model or workspace_num is nullptr"};
  }

  std::vector<uint64_t> types;
  std::vector<void *> addresses;
  for (size_t i = 0U; i < *workspace_num;) {
    const auto memory_type = context->GetInputPointer<uint64_t>(
        static_cast<int32_t>(UpdateWorkspaces::kWorkspaceMemory) + (i++));
    const auto tensor_data = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(UpdateWorkspaces::kWorkspaceMemory) + (i++));
    if ((tensor_data == nullptr) || (memory_type == nullptr)) {
      continue;
    }
    types.emplace_back(*memory_type);
    addresses.emplace_back(tensor_data->GetAddr());
  }

  std::stringstream ss;
  ss << "davinci model update workspaces address, size: " << types.size() << ". ";
  for (size_t i = 0U; i < types.size(); ++i) {
    ss << "[type: " << types[i] << ", address: " << addresses[i] << "]" << ((i + 1U == types.size()) ? "." : ", ");
  }
  return {ss.str()};
}

std::vector<std::string> PrintModelExecute(const KernelContext * context) {
  auto davinci_model = context->MutableInputPointer<ge::DavinciModel>(
      static_cast<int32_t>(InputsCommon::kDavinciModel));
  const auto input_num = context->GetInputPointer<size_t>(static_cast<int32_t>(ModelExecute::kInputNum));
  const auto output_num = context->GetInputPointer<size_t>(static_cast<int32_t>(ModelExecute::kOutputNum));
  if ((davinci_model == nullptr) || (input_num == nullptr) || (output_num == nullptr)) {
    return {"davinci_model or input_num or output_num is nullptr"};
  }
  std::stringstream ss;
  ss <<  "model execute, runtime_model_id: " << davinci_model->GetRuntimeModelId();
  ss << ", inputs address and size: [" << std::hex;
  for (size_t i = 0U; i < *input_num; ++i) {
    const auto tensor_data = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(ModelExecute::kModelExecuteEnd) + i);
    if (tensor_data != nullptr) {
      ss << std::hex << tensor_data->GetAddr() << " (" << tensor_data->GetSize() << "), ";
    }
  }

  ss << "], outputs address and size: [";
  for (size_t i = 0U; i < *output_num; ++i) {
    const auto tensor_data = context->GetInputValue<gert::GertTensorData *>(
        static_cast<int32_t>(ModelExecute::kModelExecuteEnd) + *input_num + i);
    if (tensor_data != nullptr) {
      ss << std::hex << tensor_data->GetAddr() << " (" << tensor_data->GetSize() << "), ";
    }
  }
  ss << "].";
  return {ss.str()};
}

std::vector<std::string> PrintGetRunAddress(const KernelContext *context) {
  std::stringstream ss;
  auto inputs_begin = static_cast<size_t>(InputsSpecial::kInputsCommonEnd);
  auto num = context->GetInputNum() <= inputs_begin ? 0U : context->GetInputNum() - inputs_begin;
  ss << "GetRunAddress num: " << num << ", ";
  for (size_t i = inputs_begin; i < context->GetInputNum(); ++i) {
    auto type_offset_pair = context->GetInputPointer<MemoryBaseTypeOffset>(static_cast<size_t>(i));
    auto tensor_data = context->GetOutputPointer<GertTensorData>(i - inputs_begin);
    if ((type_offset_pair == nullptr) || (tensor_data == nullptr)) {
      continue;
    }
    ss << "[" << std::hex << "mem_base_type: " << static_cast<int32_t>(type_offset_pair->base_type) << ", offset: "
       << type_offset_pair->offset << ", size: " << type_offset_pair->size
       << ", run_address: " << tensor_data->GetAddr() << "], ";
  }
  return {ss.str()};
}
} // namespace kernel
} // namespace gert
