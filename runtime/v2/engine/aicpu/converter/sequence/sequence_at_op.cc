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
#include "common/util/mem_utils.h"
#include "framework/common/debug/ge_log.h"

namespace {
  constexpr size_t kSessionIdIndex = 0U;
  constexpr size_t kContainerIdIndex = 1U;
  constexpr size_t kHandleIndex = 2U;
  constexpr size_t kDataIndex = 3U;
  constexpr size_t kIndexDescIndex = 1U;
  constexpr size_t kDataTypeIndex = 2U;
  constexpr size_t kOutputTensorIndex = 3U;
}
namespace gert {
ge::graphStatus SequenceAtDoComputeExtend(KernelContext *context, const uint64_t handle,
                                          const int64_t index) {
  uint64_t session_id = context->GetInputValue<size_t>(kSessionIdIndex);
  uint64_t container_id = context->GetInputValue<size_t>(kContainerIdIndex);
  GELOGD("SequenceAt session = %llu, container = %llu", session_id, container_id);
  GELOGD("SequenceAt handle = %llu, index = %lld", handle, index);
  ResourceMgrPtr out_rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  TensorSeqPtr sequence;
  auto ret = out_rm->Lookup(handle, &sequence);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "SequenceAt lookup tensor sequence fail, handle = %llu", handle);
    REPORT_INNER_ERR_MSG("E39999", "SequenceAt lookup tensor sequence fail");
    return ret;
  }

  auto tensor_ref = sequence->Get(index);
  if (tensor_ref == nullptr) {
    GELOGE(ge::PARAM_INVALID, "input index is out of range.");
    REPORT_INNER_ERR_MSG("E39999", "input index is out of range.");
    return ge::PARAM_INVALID;
  }

  auto dtype = sequence->DataType();

  auto output_tensor = context->GetOutputPointer<Tensor>(0);
  if (output_tensor == nullptr) {
    GELOGE(ge::PARAM_INVALID, "output0 is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "output0 is nullptr");
    return ge::PARAM_INVALID;
  }

  output_tensor->SetDataType(dtype);

  output_tensor->MutableTensorData().ShareFrom(tensor_ref->tensor_addr_);
  output_tensor->MutableStorageShape() = tensor_ref->tensor_shape_;
  output_tensor->MutableOriginShape() = tensor_ref->tensor_shape_;

  auto output_tensor_data_type = output_tensor->GetDataType();
  GELOGD("SequenceAt output tensor data type is %u, tensor size is %u",
         output_tensor_data_type, output_tensor->GetSize());
  GELOGD("SequenceAtDoCompute end");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SequenceAtDoCompute(KernelContext *context) {
  GELOGD("SequenceAtDoCompute begin");

  auto input_handle = context->GetInputPointer<TensorData>(kHandleIndex);
  if ((input_handle == nullptr) || (input_handle->GetAddr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Failed to get input handle.");
    REPORT_INNER_ERR_MSG("E39999", "Failed to get input handle.");
    return ge::PARAM_INVALID;
  }
  auto handle = *(static_cast<uint64_t*>(input_handle->GetAddr()));
  
  auto index_data = context->GetInputPointer<TensorData>(kDataIndex);
  if ((index_data == nullptr) || (index_data->GetAddr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "Failed to get input index.");
    REPORT_INNER_ERR_MSG("E39999", "Failed to get input index.");
    return ge::PARAM_INVALID;
  }

  auto extend_ctx = reinterpret_cast<ExtendedKernelContext*>(context);
  auto index_desc = extend_ctx->GetInputDesc(kIndexDescIndex);
  if (index_desc == nullptr) {
    GELOGE(ge::PARAM_INVALID, "index_desc is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "index_desc is nullptr");
    return ge::PARAM_INVALID;
  }

  auto index_type = index_desc->GetDataType();
  int64_t index = 0;
  switch (index_type) {
    case ge::DT_INT32:
      index = static_cast<int64_t>(*(static_cast<int32_t*>(index_data->GetAddr())));
      break;
    case ge::DT_INT64:
      index = *(static_cast<int64_t*>(index_data->GetAddr()));;
      break;
    default:
      GELOGE(ge::PARAM_INVALID, 
            "Sequence SequenceAt input index data type should be DT_INT32 or "
            "DT_INT64, [%u] not support.", index_type);
      REPORT_INNER_ERR_MSG("E39999", "Sequence SequenceAt input index data type should be DT_INT32 "
                         "or DT_INT64, [%u] not support.", index_type);
      return ge::PARAM_INVALID;
  }

  return SequenceAtDoComputeExtend(context, handle, index);
}

ge::graphStatus CreateBuildTensorOutputs(const ge::FastNode *node, KernelContext *context) {
  GELOGD("CreateBuildTensorOutputs begin");
  (void)node;
  auto output0 = context->GetOutput(0);
  if (output0 == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "output0 is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "output0 is nullptr");
    return ge::INTERNAL_ERROR;
  }

  auto tensor = new (std::nothrow) Tensor();
  if (tensor == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "tensor is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "tensor is nullptr");
    return ge::INTERNAL_ERROR;
  }

  output0->SetWithDefaultDeleter(tensor);
  GELOGD("CreateBuildTensorOutputs end");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(SequenceAtCompute).RunFunc(SequenceAtDoCompute).OutputsCreator(CreateBuildTensorOutputs);

ge::graphStatus SequenceEmptyDoCompute(KernelContext *context) {
  GELOGD("SequenceEmptyDoCompute begin");
  uint64_t session_id = context->GetInputValue<size_t>(kSessionIdIndex);
  uint64_t container_id = context->GetInputValue<size_t>(kContainerIdIndex);
  auto dtype = context->GetInputValue<ge::DataType>(kDataTypeIndex);
  GELOGD("SequenceEmpty session_id = %llu, container_id = %llu", session_id, container_id);
  GELOGD("SequenceEmpty dtype = %u", dtype);
  ResourceMgrPtr out_rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  uint64_t handle = out_rm->GetHandle();
  out_rm->StoreStepHandle(handle);
  GELOGD("SequenceEmpty handle = %llu", handle);

  TensorSeqPtr sequence(new (std::nothrow) TensorSeq(dtype));
  if (sequence == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "sequence is nullptr");
    REPORT_INNER_ERR_MSG("E39999", "sequence is nullptr");
    return ge::INTERNAL_ERROR;
  }
  out_rm->Create(handle, sequence);

  auto output_tensor = context->MutableInputPointer<Tensor>(kOutputTensorIndex);
  if (output_tensor == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get output tensor failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get output tensor failed.");
    return ge::PARAM_INVALID;
  }

  uint64_t* data = output_tensor->GetData<uint64_t>();
  *data = handle;
  GELOGD("SequenceEmptyDoCompute end");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(SequenceEmptyCompute).RunFunc(SequenceEmptyDoCompute);

ge::graphStatus SequenceLengthDoCompute(KernelContext *context) {
  GELOGD("SequenceLengthDoCompute begin");
  uint64_t session_id = context->GetInputValue<size_t>(kSessionIdIndex);
  uint64_t container_id = context->GetInputValue<size_t>(kContainerIdIndex);
  auto input_handle = context->GetInputPointer<TensorData>(kHandleIndex);
  if ((input_handle == nullptr) || (input_handle->GetAddr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "Failed to get input handle.");
    REPORT_INNER_ERR_MSG("E39999", "Failed to get input handle.");
    return ge::PARAM_INVALID;
  }

  auto handle = *(static_cast<uint64_t*>(input_handle->GetAddr()));
  GELOGD("SequenceLength session = %llu, container = %llu", session_id, container_id);
  GELOGD("SequenceLength handle = %llu", handle);
  ResourceMgrPtr out_rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, out_rm);
  TensorSeqPtr sequence;
  auto ret = out_rm->Lookup(handle, &sequence);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "SequenceLength lookup tensor sequence fail, handle = %llu", handle);
    REPORT_INNER_ERR_MSG("E39999", "SequenceLength lookup tensor sequence fail");
    return ret;
  }

  int64_t length = static_cast<int64_t>(sequence->Size());
  GELOGD("SequenceLength length = %llu", length);

  auto output_tensor = context->MutableInputPointer<Tensor>(kOutputTensorIndex);
  if (output_tensor == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get output tensor failed.");
    REPORT_INNER_ERR_MSG("E39999", "Get output tensor failed.");
    return ge::PARAM_INVALID;
  }

  int64_t* data = output_tensor->GetData<int64_t>();
  *data = length;
  GELOGD("SequenceLengthDoCompute end");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(SequenceLengthCompute).RunFunc(SequenceLengthDoCompute);
}  // namespace gert