/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "graph/utils/op_desc_utils.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "common/checker.h"
#include "graph/operator_factory.h"
#include "graph/utils/math_util.h"
#include "graph/op_desc.h"

namespace gert {
namespace {
constexpr size_t kMemAlignment = 512U;
void SetTensorDesc(const StorageShape &shape, const StorageFormat &format, ge::DataType dtype, Tensor *dst) {
  auto &storage_shape = dst->MutableStorageShape();
  storage_shape = shape.GetStorageShape();
  auto &origin_shape = dst->MutableOriginShape();
  origin_shape = shape.GetOriginShape();
  dst->SetStorageFormat(format.GetStorageFormat());
  dst->SetOriginFormat(format.GetOriginFormat());
  dst->SetDataType(dtype);
}
ge::OpDescPtr GetOpDescPtr(const EagerOpExecutionContext &ctx) {
  const auto node_type = ctx.GetNodeType();
  auto const node_op = ge::OperatorFactory::CreateOperator("_", node_type);
  if (node_op.IsEmpty()) {
    GELOGE(ge::FAILED, "get op from OperatorFactory fail. opType: %s", node_type);
    return nullptr;
  }
  GELOGD("get op from OperatorFactory success. opType is %s", node_type);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
  return op_desc;
}
} // namespace

rtStream EagerOpExecutionContext::GetStream() const {
  auto start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(start_index >= 0);

  const auto av = GetInput(start_index + static_cast<int64_t>(AdditionalInputIndex::kStream));
  GE_ASSERT_NOTNULL(av);
  return av->GetValue<rtStream>();
}

Tensor *EagerOpExecutionContext::MallocOutputTensor(size_t index, const StorageShape &shape,
                                                    const StorageFormat &format, const ge::DataType dtype) {
  auto additional_start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_start_index >= 0);
  auto gert_allocator =
      GetInputValue<GertAllocator *>(additional_start_index + static_cast<int64_t>(AdditionalInputIndex::kDeviceAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);

  auto op_desc = GetOpDescPtr(*this);
  auto output_name = op_desc->GetOutputNameByIndex(index);
  GE_ASSERT_TRUE(op_desc->GetInputIndexByName(output_name) == -1, "[MallocOutputTensor] output name exists in input");

  auto output_tensor = GetOutputPointer<Tensor>(index);
  GE_ASSERT_NOTNULL(output_tensor);

  SetTensorDesc(shape, format, dtype, output_tensor);
  const size_t tensor_size = shape.GetStorageShape().GetShapeSize() * GetSizeByDataType(dtype);
  size_t aligned_tensor_size = tensor_size;
  GE_ASSERT_TRUE(!ge::RoundUpOverflow(tensor_size, kMemAlignment, aligned_tensor_size));
  const TensorData& tensor_data = output_tensor->GetTensorData();
  // 静态场景下内存地址已赋值 不需要去Malloc
  if (tensor_data.GetAddr() != nullptr && tensor_data.GetSize() > 0) {
    return output_tensor;
  }
  auto gert_tensor_data = gert_allocator->MallocTensorData(aligned_tensor_size);
  output_tensor->SetData(std::move(gert_tensor_data.MutableTensorData()));
  return output_tensor;
}

Tensor *EagerOpExecutionContext::MakeOutputRefInput(size_t output_index, size_t input_index) const {
  auto additional_start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_start_index >= 0);

  auto op_desc = GetOpDescPtr(*this);

  auto input_name = op_desc->GetInputNameByIndex(input_index);
  auto output_name = op_desc->GetOutputNameByIndex(output_index);
  GE_ASSERT_TRUE(input_name == output_name, "[MakeOutputRefInput] output name does not exist in input");

  auto *output_tensor = const_cast<Tensor *>(GetOutputPointer<Tensor>(output_index));
  GE_ASSERT_NOTNULL(output_tensor);

  auto input_tensor = GetInputPointer<Tensor>(input_index);
  GE_ASSERT_NOTNULL(input_tensor);
  SetTensorDesc(input_tensor->GetShape(), input_tensor->GetFormat(), input_tensor->GetDataType(), output_tensor);
  output_tensor->MutableTensorData().ShareFrom(input_tensor->GetTensorData());
  return output_tensor;
}

void *EagerOpExecutionContext::MallocWorkSpace(size_t size) {
  auto additional_input_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_input_index >= 0);
  auto gert_allocator = GetInputValue<GertAllocator *>(additional_input_index +
                                                       static_cast<int64_t>(AdditionalInputIndex::kDeviceAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);

  auto additional_output_start = GetAdditionalOutputStartIndex();
  auto memory_vec =
      GetOutputPointer<std::vector<GertMemBlock *>>(additional_output_start + static_cast<size_t>(AdditionalOutputIndex::kWorkSpace));
  GE_ASSERT_NOTNULL(memory_vec);

  size_t aligned_size = size;
  GE_ASSERT_TRUE(!ge::RoundUpOverflow(size, kMemAlignment, aligned_size));
  auto mem_block = gert_allocator->Malloc(aligned_size);
  GE_ASSERT_NOTNULL(mem_block);
  (void)memory_vec->emplace_back(mem_block);
  return mem_block->GetAddr();
}

}  // namespace gert