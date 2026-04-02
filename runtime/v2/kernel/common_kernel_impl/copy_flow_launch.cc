/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "copy_flow_launch.h"
#include "memory_copy.h"
#include "common/checker.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "common/table_driven.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "graph/utils/type_utils.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph_metadef/common/ge_common/util.h"

using namespace ge;
namespace gert {
namespace kernel {
namespace {
constexpr size_t kAlignBytes4 = 4U;
}  // namespace

ge::graphStatus CreateCopyFlowLaunchTensorData(const FastNode *node, KernelContext *context) {
  (void)node;
  auto out_num = context->GetOutputNum();
  for (size_t i = static_cast<size_t>(CopyFlowLaunchOutputs::kAddress); i < out_num; ++i) {
    auto chain = context->GetOutput(i);
    if (chain == nullptr) {
      return ge::GRAPH_FAILED;
    }
    auto tensor_data = new (std::nothrow) GertTensorData(0, kOnDeviceHbm, -1, nullptr);
    if (tensor_data == nullptr) {
      return ge::GRAPH_FAILED;
    }
    chain->SetWithDefaultDeleter(tensor_data);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyTensorToDevice(KernelContext *context, const size_t copy_index) {
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(CopyFlowLaunchInputs::kStream));
  auto gert_allocator =
      context->MutableInputPointer<GertAllocator>(
          static_cast<size_t>(CopyFlowLaunchInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  auto addr_index = copy_index * kSizeOfCopyToDevice + static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart);
  auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
  auto tensor_size = context->GetInputValue<size_t>(addr_index + 1U);
  auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
  auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
  auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(
      copy_index + static_cast<size_t>(CopyFlowLaunchOutputs::kAddress));
  GE_ASSERT_NOTNULL(tensor_data);
  GE_ASSERT_NOTNULL(out_tensor_data);
  GE_ASSERT_NOTNULL(src_storage_shape);
  auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(tensor_size));
  KERNEL_CHECK_NOTNULL(mem_block);
  KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", tensor_size);
  KERNEL_TRACE_ALLOC_MEM(gert_allocator->GetStreamId(), mem_block, mem_block->GetAddr(), mem_block->GetSize());
  *out_tensor_data = TensorUtils::ToGertTensorData(
      mem_block, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());

  auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
  GELOGD("StreamCopyH2D, host addr %p, host tensor size %zu, device addr %p, alloc device size %zu",
         tensor_data->GetAddr(), host_tensor_size, mem_block->GetAddr(), tensor_size);
  if (host_tensor_size > 0U) {
    GE_ASSERT_RT_OK(rtMemcpyAsync(mem_block->GetAddr(), tensor_size, tensor_data->GetAddr(), host_tensor_size,
                                  RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  }
  out_tensor_data->SetPlacement(kOnDeviceHbm);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyFlowLaunch(KernelContext *context) {
  auto output_num = context->GetOutputNum();
  if (static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart) + (output_num * kSizeOfCopyToDevice) !=
      context->GetInputNum()) {
    GELOGE(ge::GRAPH_FAILED, "input num is not matched, input start %zu, output num %zu, total input num %zu",
           static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart), output_num, context->GetInputNum());
    return ge::GRAPH_FAILED;
  }

  auto input_num = context->GetInputPointer<size_t>(static_cast<size_t>(CopyFlowLaunchInputs::kInputsNum));
  GE_CHECK_NOTNULL(input_num);
  GELOGD("host input num is %zu, output num is %zu.", *input_num, output_num);
  if (*input_num != output_num) {
    GELOGE(ge::GRAPH_FAILED, "host input num %zu, is not match output num %zu,", *input_num, output_num);
    return ge::GRAPH_FAILED;
  }

  auto args = context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(static_cast<size_t>(CopyFlowLaunchInputs::kRtArg));
  GE_CHECK_NOTNULL(args);
  // 更新host input data的offset，从图上保证先做tiling，然后 CopyFlowLaunch 进行随路拷贝
  GE_ASSERT_SUCCESS(args->UpdateMergedCopyInfo());

  auto inputs_index_cvv =
      context->GetInputValue<ContinuousVectorVector *>(static_cast<size_t>(CopyFlowLaunchInputs::kInputsIndex));
  GE_ASSERT_NOTNULL(inputs_index_cvv);
  for (size_t i = 0U; i < output_num; ++i) {
    auto addr_index = i * kSizeOfCopyToDevice + static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
    auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
    GE_CHECK_NOTNULL(src_storage_shape);
    auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
    auto out_tensor_data =
        context->GetOutputPointer<gert::GertTensorData>(i + static_cast<size_t>(CopyFlowLaunchOutputs::kAddress));
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(out_tensor_data);
    if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
      GELOGD("The [%zu]th tensor data placement is %d, no need to optimize", i,
             static_cast<int32_t>(tensor_data->GetPlacement()));
      out_tensor_data->ShareFrom(*tensor_data);
    } else if (TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      const auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
      if (host_tensor_size < 0) {
        GELOGE(ge::GRAPH_FAILED, "shape_size[%" PRId64 "], data_type[%s]",
               src_storage_shape->GetStorageShape().GetShapeSize(),
               ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
        return ge::GRAPH_FAILED;
      }
      size_t align_size = ge::RoundUp(static_cast<uint64_t>(host_tensor_size), kAlignBytes4);
      GELOGD("shape_size[%" PRId64 "], data_type[%s], host_tensor_size[%" PRId64 "], align_size[%zu]",
             src_storage_shape->GetStorageShape().GetShapeSize(),
             ge::TypeUtils::DataTypeToSerialString(data_type).c_str(), host_tensor_size, align_size);
      auto host_input_data_size = args->GetHostInputDataSize();
      host_input_data_size += align_size;
      auto max_host_input_data_len = kMaxHostInputDataLen + args->GetMergedCopySize();
      if (host_input_data_size > max_host_input_data_len) {
        GE_ASSERT_SUCCESS(CopyTensorToDevice(context, i));
      } else {
        auto inputs_index_cv = inputs_index_cvv->Get(i);
        GE_ASSERT_NOTNULL(inputs_index_cv);
        RtKernelLaunchArgsEx::HostInputInfo host_input{tensor_data->GetAddr(), inputs_index_cv,
                                                       static_cast<size_t>(host_tensor_size)};
        GE_ASSERT_SUCCESS(args->UpdateHostInputArgs(host_input));
      }
    } else {
      GELOGE(ge::GRAPH_FAILED, "unsupported copy form placement %d to device hbm",
             static_cast<int32_t>(tensor_data->GetPlacement()));
      return ge::GRAPH_FAILED;
    }
  }
  // copy flow launch之后，将字节进行对齐
  GE_ASSERT_GRAPH_SUCCESS(args->AlignHostInputSize());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CopyFlowLaunch)
    .RunFunc(CopyFlowLaunch)
    .OutputsCreator(CreateCopyFlowLaunchTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert
