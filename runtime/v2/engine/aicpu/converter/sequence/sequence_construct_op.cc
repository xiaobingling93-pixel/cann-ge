/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tensor.h"
#include "register/kernel_registry_impl.h"
#include "tensor_sequence.h"
#include "resource_mgr.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
ge::graphStatus SequenceConstructCompute(KernelContext* context) {
  GELOGD("SequenceConstructCompute begin");
  constexpr int32_t session_id_idx = 0;
  constexpr int32_t container_id_idx = 1;
  constexpr int32_t input_num_idx = 2;
  constexpr int32_t first_input_idx = 3;
  auto input_num = context->GetInputValue<int32_t>(input_num_idx);
  if (input_num < 1) {
    GELOGE(ge::PARAM_INVALID, "input num must be at least 1, but now is %d", input_num);
    REPORT_INNER_ERR_MSG("E39999", "input num must be at least 1, but now is %d", input_num);
    return ge::PARAM_INVALID;
  }

  auto extend_ctx = reinterpret_cast<ExtendedKernelContext*>(context);
  auto input0_desc = extend_ctx->GetInputDesc(0);
  if (input0_desc == nullptr) {
    GELOGE(ge::PARAM_INVALID, "input0 desc is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "input0 desc is nullptr");
    return ge::PARAM_INVALID;
  }

  auto data0_type = input0_desc->GetDataType();
  TensorSeqPtr tensor_seq_ptr = std::make_shared<TensorSeq>(data0_type);
  if (tensor_seq_ptr == nullptr) {
    GELOGE(ge::PARAM_INVALID, "tensor_seq_ptr is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "tensor_seq_ptr is nullptr");
    return ge::PARAM_INVALID;
  }

  for (int32_t i = 0; i < input_num; i++) {
    auto input_desc = extend_ctx->GetInputDesc(i);
    if (input_desc == nullptr) {
      GELOGE(ge::PARAM_INVALID, "input_desc %u desc is nullptr", i);
      REPORT_INNER_ERR_MSG("E39999", "input_desc %u desc is nullptr", i);
      return ge::PARAM_INVALID;
    }

    auto data_type = input_desc->GetDataType();
    // in lowering func, we set tensor_data and shape together
    auto tensor_data_index = i * 2 + first_input_idx;
    auto tensor_data =
        context->GetInputPointer<TensorData>(tensor_data_index);
    if (tensor_data == nullptr) {
      GELOGE(ge::PARAM_INVALID, "tensor data is nullptr");
      REPORT_INNER_ERR_MSG("E39999", "tensor data is nullptr");
      return ge::PARAM_INVALID;
    }

    auto storage_shape =
        context->GetInputPointer<StorageShape>(tensor_data_index + 1);
    tensor_seq_ptr->Add(data_type, *tensor_data, *storage_shape);
  }
  auto session_id = context->GetInputValue<size_t>(session_id_idx);
  auto container_id = context->GetInputValue<size_t>(container_id_idx);
  GELOGD("session_id is %llu, container_id is %llu", session_id,
         container_id);
  ResourceMgrPtr out_rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  GELOGD("handle is %llu", handle);
  out_rm->StoreStepHandle(handle);
  out_rm->Create(handle, tensor_seq_ptr);
  int32_t output_idx = first_input_idx + input_num * 2;
  auto output_tensor = context->MutableInputPointer<Tensor>(output_idx);
  if ((output_tensor == nullptr) || (output_tensor->GetData<uint64_t>() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "output_tensor is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "output_tensor is nullptr");
    return ge::PARAM_INVALID;
  }

  uint64_t* data_ptr = output_tensor->GetData<uint64_t>();
  *data_ptr = handle;
  GELOGD("tensor sequence size is %zu.", tensor_seq_ptr->Size());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SequenceConstructCompute).RunFunc(SequenceConstructCompute);
}  // namespace gert