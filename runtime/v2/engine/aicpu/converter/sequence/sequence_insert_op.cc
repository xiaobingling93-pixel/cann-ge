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
ge::graphStatus SequenceInsertCompute(KernelContext* context) {
  constexpr int32_t session_id_idx = 0;
  constexpr int32_t container_id_idx = 1;
  constexpr int32_t input_num_idx = 2;
  constexpr int32_t input_handle_idx = 3;
  constexpr int32_t input_value_idx = 4;
  constexpr int32_t input_index_idx = 6;
  GELOGD("Enter SequenceInsertCompute");
  auto input_num = context->GetInputValue<uint32_t>(input_num_idx);
  constexpr int32_t least_input_number = 2;
  if (input_num < least_input_number) {
    GELOGE(ge::PARAM_INVALID, "input num must be at least 2, but now is %u", input_num);
    REPORT_INNER_ERR_MSG("E39999", "input num must be at least 2, but now is %u", input_num);
    return ge::PARAM_INVALID;
  }

  auto input_handle_data =
      context->GetInputPointer<TensorData>(input_handle_idx);
  if ((input_handle_data == nullptr) || (input_handle_data->GetAddr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "Get input handle tensor data failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get input handle tensor data failed.");
    return ge::PARAM_INVALID;
  }

  // handle's type is DT_RESOURCE(aka uint64_t)
  auto handle = static_cast<uint64_t*>(input_handle_data->GetAddr());
  GELOGD("handle is %llu", *handle);

  auto session_id = context->GetInputValue<size_t>(session_id_idx);
  auto container_id = context->GetInputValue<size_t>(container_id_idx);
  GELOGD("Session_id is %llu, Container_id is %llu", session_id, container_id);
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, rm);
  TensorSeqPtr tensor_seq_ptr;
  rm->Lookup(*handle, &tensor_seq_ptr);
  // Get input value data type from extended context
  auto extend_ctx = reinterpret_cast<ExtendedKernelContext*>(context);
  auto input_value_desc = extend_ctx->GetInputDesc(1);
  if (input_value_desc == nullptr) {
    GELOGE(ge::PARAM_INVALID, "input value desc is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "input value desc is nullptr");
    return ge::PARAM_INVALID;
  }

  auto input_value_data_type = input_value_desc->GetDataType();
  GELOGD("input value data_type is %u", input_value_data_type);
  auto input_value_tensor_data =
      context->GetInputPointer<TensorData>(input_value_idx);
  auto input_value_storage_shape =
      context->GetInputPointer<StorageShape>(input_value_idx + 1);

  int32_t output_idx = input_index_idx;
  ge::graphStatus ret = ge::GRAPH_SUCCESS;
  if (input_num == (least_input_number + 1)) {
    auto insert_index_desc = extend_ctx->GetInputDesc(2);
    if (insert_index_desc == nullptr) {
      GELOGE(ge::PARAM_INVALID, "insert_index_desc is nullptr");
      REPORT_INNER_ERR_MSG("E39999", "insert_index_desc is nullptr");
      return ge::PARAM_INVALID;
    }

    auto insert_index_type = insert_index_desc->GetDataType();
    auto insert_index_data =
        context->GetInputPointer<TensorData>(input_index_idx);
    if (insert_index_data == nullptr) {
      GELOGE(ge::PARAM_INVALID, "Get insert index failed.");
      REPORT_INNER_ERR_MSG("E39999", "Get insert index failed.");
      return ge::PARAM_INVALID;
    }
    if (insert_index_data->GetAddr() == nullptr) {
      GELOGE(ge::PARAM_INVALID, "TensorData is nullptr.");
      REPORT_INNER_ERR_MSG("E39999", "TensorData is nullptr.");
      return ge::PARAM_INVALID;
    }

    int64_t index = 0;
    switch (insert_index_type) {
      case ge::DT_INT32:
        index = static_cast<int64_t>(
            *static_cast<int32_t*>(insert_index_data->GetAddr()));
        break;
      case ge::DT_INT64:
        index = *(static_cast<int64_t*>(insert_index_data->GetAddr()));
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "Sequence Insert input index data type should be DT_INT32 "
              "or DT_INT64, [%u] not support.", insert_index_type);
        REPORT_INNER_ERR_MSG("E39999", "Sequence Insert input index data type should be DT_INT32 "
                           "or DT_INT64, [%u] not support.", insert_index_type);
        return ge::PARAM_INVALID;
    }
    GELOGD("SequenceInsert insert index is %lld.", index);
    ret = tensor_seq_ptr->Add(input_value_data_type, *input_value_tensor_data,
                              *input_value_storage_shape, index);
    output_idx++;
  } else {
    ret = tensor_seq_ptr->Add(input_value_data_type, *input_value_tensor_data,
                              *input_value_storage_shape);
  }
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "insert value to sequence tensor failed.");
    REPORT_INNER_ERR_MSG("E39999", "insert value to sequence tensor failed.");
    return ge::PARAM_INVALID;
  }
  auto output_tensor = context->MutableInputPointer<Tensor>(output_idx);
  if (output_tensor == nullptr) {
    GELOGE(ge::PARAM_INVALID, "output_tensor is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "output_tensor is nullptr");
    return ge::PARAM_INVALID;
  }

  uint64_t* data_ptr = output_tensor->GetData<uint64_t>();
  *data_ptr = *handle;
  GELOGD("Finish SequenceInsertCompute tensor sequence size is %zu.", tensor_seq_ptr->Size());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SequenceInsertCompute).RunFunc(SequenceInsertCompute);
}  // namespace gert