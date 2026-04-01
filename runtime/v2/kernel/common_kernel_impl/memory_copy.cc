/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_copy.h"
#include "common/checker.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "common/debug/log.h"
#include "common/table_driven.h"
#include "runtime/mem.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/common_kernel_impl/build_tensor.h"
#include "kernel/common_kernel_impl/calc_tenorsize_from_shape.h"
#include "kernel/tensor_attr.h"
#include "graph/def_types.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "utils/utils.h"
#include "exe_graph/runtime/tensor_data_utils.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"

using namespace ge;
namespace gert {
namespace kernel {
/*
 * input: addr  tensor size
 * output: host addr
 */
constexpr int64_t kPaddingSize = 64;
struct HostMemHeader {
  size_t count;
};
struct HostMem {
  union {
    HostMemHeader header;
    uint8_t padding_size[kPaddingSize];
  } header;
  void *GetDataAddress() {
    return static_cast<void *>(this + 1);
  }
};

inline bool CalcSize(size_t size, size_t &padded_size) {
  if (ge::AddOverflow(size, kPaddingSize, padded_size)) {
    GELOGE(ge::PARAM_INVALID, "Invalid size %zu, failed to padding to %zu", size, kPaddingSize);
    return false;
  }
  padded_size = (padded_size - 1) & -kPaddingSize;
  // add memory head
  if (ge::AddOverflow(padded_size, sizeof(HostMem), padded_size)) {
    return false;
  }
  return true;
}

ge::graphStatus CopyD2H(KernelContext *context) {
  auto tensor_data_ptr =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MemoryCopyInputs::kSrcAddress));
  auto tensor_size = context->GetInputValue<size_t>(static_cast<size_t>(MemoryCopyInputs::kSrcLength));
  auto gert_allocator = context->GetInputValue<GertAllocator *>(static_cast<size_t>(MemoryCopyInputs::kAllocator));
  auto host_tensor_data = context->GetOutputPointer<gert::GertTensorData>(0);
  if (tensor_data_ptr == nullptr || host_tensor_data == nullptr || gert_allocator == nullptr) {
    return ge::GRAPH_FAILED;
  }

  size_t alloc_size;
  GE_ASSERT_TRUE(CalcSize(tensor_size, alloc_size));
  auto host_block = gert_allocator->Malloc(alloc_size);
  KERNEL_CHECK_NOTNULL(host_block);
  KERNEL_CHECK(host_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", alloc_size);
  if (tensor_size > 0U) {
    GE_CHK_RT_RET(rtMemcpyEx(host_block->GetAddr(), static_cast<uint64_t>(tensor_size), tensor_data_ptr->GetAddr(),
                             static_cast<uint64_t>(tensor_size), RT_MEMCPY_DEVICE_TO_HOST));
  }
  host_tensor_data->ShareFrom(
      {tensor_size, host_tensor_data->GetPlacement(), gert_allocator->GetStreamId(), host_block});

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus CreatePlacementTensorData(const ge::FastNode *node, KernelContext *context, TensorPlacement placement) {
  (void)node;
  auto output_num = context->GetOutputNum();
  for (size_t i = 0U; i < output_num; ++i) {
    auto chain = context->GetOutput(i);
    if (chain == nullptr) {
      return ge::GRAPH_FAILED;
    }
    auto tensor_data = new (std::nothrow) GertTensorData(0, placement, -1, nullptr);
    if (tensor_data == nullptr) {
      return ge::GRAPH_FAILED;
    }
    chain->SetWithDefaultDeleter(tensor_data);
  }

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus CreateTensorDataAtHost(const ge::FastNode *node, KernelContext *context) {
  return CreatePlacementTensorData(node, context, kOnHost);
}
ge::graphStatus CreateTensorDataAtDeviceHbm(const ge::FastNode *node, KernelContext *context) {
  return CreatePlacementTensorData(node, context, kOnDeviceHbm);
}

ge::graphStatus SinkWeightDataOutputCreator(const ge::FastNode *node, KernelContext *context) {
  return CreatePlacementTensorData(node, context, kOnDeviceHbm);
}
REGISTER_KERNEL(CopyD2H)
    .RunFunc(CopyD2H)
    .OutputsCreator(CreateTensorDataAtHost)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

size_t CalcStringSize(const uint8_t *addr, const int64_t &shape_size) {
  size_t string_size = 0U;
  for (int64_t i = 0; i < shape_size; i++) {
    auto head = reinterpret_cast<const ge::StringHead *>(addr);
    // 内存中包含\0，len不包含\0，所以计算内存长度需要额外+1
    string_size += head->len + 1U;
    addr += sizeof(ge::StringHead);
  }
  return string_size;
};
graphStatus CalcStringTensorSize(const gert::GertTensorData *const tensor_data, aclrtStream stream,
                                 const StorageShape *storage_shape, ge::DataType datatype, uint64_t &tensor_size) {
  if (datatype != ge::DT_STRING) {
    GELOGE(ge::GRAPH_FAILED, "[Calc][String] data_type[%d] is not DT_STRING", datatype);
    return ge::GRAPH_FAILED;
  }

  auto shape = storage_shape->GetOriginShape();
  auto shape_size = shape.GetShapeSize();
  tensor_size = shape_size * sizeof(ge::StringHead);

  if (tensor_size > tensor_data->GetSize()) {
    GELOGE(ge::PARAM_INVALID, "[Calc][String]src tensor size [%zu] is less than calc string head size [%zu]",
           tensor_data->GetSize(), tensor_size);
    return ge::GRAPH_FAILED;
  }

  if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
    size_t alloc_size;
    GE_ASSERT_TRUE(CalcSize(tensor_size, alloc_size));
    std::unique_ptr<uint8_t[]> host_addr_holder(new (std::nothrow) uint8_t[alloc_size]);
    GE_ASSERT_NOTNULL(host_addr_holder);

    auto host_mem = reinterpret_cast<HostMem *>(host_addr_holder.get());
    host_mem->header.header.count = 1U;

    GELOGD("[Calc][String]CopyD2H, device addr %p, host addr %p, tensor size %zu, alloc size %zu",
           tensor_data->GetAddr(), host_mem->GetDataAddress(), tensor_size, alloc_size);
    GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(stream));
    if (tensor_size > 0U) {
      GE_CHK_RT_RET(rtMemcpyEx(host_mem->GetDataAddress(), static_cast<uint64_t>(tensor_size), tensor_data->GetAddr(),
                               static_cast<uint64_t>(tensor_size), RT_MEMCPY_DEVICE_TO_HOST));
    }
    tensor_size += CalcStringSize(reinterpret_cast<uint8_t *>(host_mem->GetDataAddress()), shape_size);
  } else if (TensorPlacementUtils::IsOnHostNotFollowing(tensor_data->GetPlacement())) {
    tensor_size += CalcStringSize(reinterpret_cast<uint8_t *>(tensor_data->GetAddr()), shape_size);
  }

  if (tensor_size > tensor_data->GetSize()) {
    GELOGE(ge::PARAM_INVALID, "[Calc][String]src tensor size [%zu] is less than calc string real size [%zu]",
           tensor_data->GetSize(), tensor_size);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// context only for KERNEL_TRACE_ALLOC_MEM
ge::graphStatus CopyTensorDataToD(const StorageTensorDesc &tensor_desc, GertAllocator *gert_allocator,
                                  aclrtStream stream, gert::GertTensorData *out_tensor_data,
                                  KernelContext *const context) {
  size_t alloc_size;
  GE_ASSERT_TRUE(CalcSize(tensor_desc.tensor_size, alloc_size));
  auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(alloc_size));
  KERNEL_CHECK_NOTNULL(mem_block);
  KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", alloc_size);
  KERNEL_TRACE_ALLOC_MEM(gert_allocator->GetStreamId(), mem_block, mem_block->GetAddr(), mem_block->GetSize());
  *out_tensor_data =
      TensorUtils::ToGertTensorData(mem_block, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());

  uint64_t unaligned_tensor_size = 0UL;
  if (tensor_desc.data_type != ge::DT_STRING) {
    GE_ASSERT_GRAPH_SUCCESS(CalcUnalignedTensorSizeByShape(tensor_desc.storage_shape->GetStorageShape(),
                                                           tensor_desc.data_type, unaligned_tensor_size));
  } else {
    GE_ASSERT_GRAPH_SUCCESS(CalcStringTensorSize(tensor_desc.tensor_data, stream, tensor_desc.storage_shape,
                                                 tensor_desc.data_type, unaligned_tensor_size));
  }

  // RT_MEMCPY_HOST_TO_DEVICE_EX会备份一下host数据，会有浪费，后面考虑是否可以联系rts提供转移host指针所有权的方式
  // 异步拷贝+RT_MEMCPY_HOST_TO_DEVICE基本上代表bug，因为host内存不是基于流，并且有生命周期的，
  // 那么异步方式真正发生拷贝的时候，host内存可能已经被释放掉了
  if (unaligned_tensor_size > 0U) {
    const auto copy_direction = TensorPlacementUtils::IsOnHost(tensor_desc.tensor_data->GetPlacement())
                                    ? RT_MEMCPY_HOST_TO_DEVICE_EX
                                    : RT_MEMCPY_DEVICE_TO_DEVICE;
    GE_CHK_RT_RET(rtMemcpyAsync(mem_block->GetAddr(), alloc_size, tensor_desc.tensor_data->GetAddr(),
                                unaligned_tensor_size, copy_direction, stream));
    KERNEL_TRACE(
        "[MEM]StreamCopy, src addr %p, src tensor size %zu, src placement %s, dst addr %p,"
        " alloc device size %zu, dst placement %s",
        tensor_desc.tensor_data->GetAddr(), unaligned_tensor_size,
        GetPlacementStr(tensor_desc.tensor_data->GetPlacement()), mem_block->GetAddr(), alloc_size,
        GetPlacementStr(gert_allocator->GetPlacement()));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetStorageTensorDescByIndex(const KernelContext *const context, size_t i,
                                            StorageTensorDesc &storage_tensor_desc) {
  auto addr_index = i * kSizeOfCopyToDevice + static_cast<size_t>(MakeSureTensorAtDeviceInputs::kAddrAndLengthStart);
  storage_tensor_desc.tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
  GE_ASSERT_NOTNULL(storage_tensor_desc.tensor_data);
  storage_tensor_desc.tensor_size = context->GetInputValue<size_t>(addr_index + 1U);

  storage_tensor_desc.storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
  GE_ASSERT_NOTNULL(storage_tensor_desc.storage_shape);

  storage_tensor_desc.data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyH2D(KernelContext *context) {
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(MakeSureTensorAtDeviceInputs::kStream));
  auto gert_allocator =
      context->MutableInputPointer<GertAllocator>(static_cast<size_t>(MakeSureTensorAtDeviceInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  auto copy_num = context->GetOutputNum();
  GE_ASSERT_EQ(
      static_cast<size_t>(MakeSureTensorAtDeviceInputs::kAddrAndLengthStart) + (copy_num * kSizeOfCopyToDevice),
      context->GetInputNum());

  for (size_t i = 0U; i < copy_num; ++i) {
    StorageTensorDesc storage_tensor_desc;
    GE_ASSERT_GRAPH_SUCCESS(GetStorageTensorDescByIndex(context, i, storage_tensor_desc));
    auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(i);
    GE_ASSERT_NOTNULL(out_tensor_data);
    GE_ASSERT_GRAPH_SUCCESS(CopyTensorDataToD(storage_tensor_desc, gert_allocator, stream, out_tensor_data, context));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeSureTensorAtHost(KernelContext *context) {
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(MakeSureTensorAtHostInputs::kStream));
  auto copy_num = context->GetOutputNum();
  GE_ASSERT_EQ(static_cast<size_t>(MakeSureTensorAtHostInputs::kAddrAndLengthStart) + (copy_num * 2U),
               context->GetInputNum());
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(MakeSureTensorAtHostInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  for (size_t i = 0U; i < copy_num; ++i) {
    auto addr_index = i * 2U + static_cast<size_t>(MakeSureTensorAtHostInputs::kAddrAndLengthStart);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
    auto tensor_size = context->GetInputValue<size_t>(addr_index + 1U);
    auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(i);
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(out_tensor_data);
    if (TensorPlacementUtils::IsOnHostNotFollowing(tensor_data->GetPlacement())) {
      GELOGD("The [%zu]th tensor data placement is %d, no need to do D2H", i,
             static_cast<int32_t>(tensor_data->GetPlacement()));
      out_tensor_data->ShareFrom(*tensor_data);
    } else if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
      size_t alloc_size;
      GE_ASSERT_TRUE(CalcSize(tensor_size, alloc_size));
      auto host_block = gert_allocator->Malloc(alloc_size);
      KERNEL_CHECK_NOTNULL(host_block);
      KERNEL_CHECK(host_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", alloc_size);
      GELOGD("StreamCopyD2H, device addr %p, host addr %p, tensor size %zu, alloc size %zu", tensor_data->GetAddr(),
             host_block->GetAddr(), tensor_size, alloc_size);
      GE_ASSERT_SUCCESS(DoRtStreamSyncWithTimeout(stream));
      if (tensor_size > 0U) {
        GE_CHK_RT_RET(rtMemcpyEx(host_block->GetAddr(), static_cast<uint64_t>(tensor_size), tensor_data->GetAddr(),
                                 static_cast<uint64_t>(tensor_size), RT_MEMCPY_DEVICE_TO_HOST));
      }
      out_tensor_data->ShareFrom(
          {tensor_size, kOnHost, gert_allocator->GetStreamId(), host_block});
    } else {
      GELOGE(ge::GRAPH_FAILED, "unsupported copy form placement %d to host",
             static_cast<int32_t>(tensor_data->GetPlacement()));
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool IsNeedMallocWhenMakeSureAtDevice(const TensorPlacement src_placement, const TensorPlacement dst_placement) {
  if (TensorPlacementUtils::IsOnHost(src_placement)) {
    return true;
  }
  if (TensorPlacementUtils::IsOnDevice(src_placement) && IsPlacementSrcToDstNeedCopy(src_placement, dst_placement)) {
    return true;
  }
  return false;
}

ge::graphStatus MakeSureTensorAtDevice(KernelContext *context) {
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(MakeSureTensorAtDeviceInputs::kStream));
  auto gert_allocator = context->MutableInputPointer<GertAllocator>(
      static_cast<size_t>(MakeSureTensorAtDeviceInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  auto copy_num = context->GetOutputNum();
  GE_ASSERT_EQ(
      static_cast<size_t>(MakeSureTensorAtDeviceInputs::kAddrAndLengthStart) + (copy_num * kSizeOfCopyToDevice),
      context->GetInputNum());

  for (size_t i = 0U; i < copy_num; ++i) {
    auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(i);
    GE_ASSERT_NOTNULL(out_tensor_data);
    StorageTensorDesc storage_tensor_desc;
    GE_ASSERT_GRAPH_SUCCESS(GetStorageTensorDescByIndex(context, i, storage_tensor_desc));

    const auto src_placement = storage_tensor_desc.tensor_data->GetPlacement();
    const auto dst_placement = gert_allocator->GetPlacement();
    const bool need_malloc = IsNeedMallocWhenMakeSureAtDevice(src_placement, dst_placement);
    if (need_malloc) {
      GE_ASSERT_GRAPH_SUCCESS(CopyTensorDataToD(storage_tensor_desc, gert_allocator, stream, out_tensor_data, context));
    } else if (TensorPlacementUtils::IsOnDevice(src_placement)) {
      GELOGD("The [%zu]th tensor data placement is %s, dst placement is %s, no need to do copy", i,
             GetPlacementStr(src_placement), GetPlacementStr(dst_placement));
      out_tensor_data->ShareFrom(*storage_tensor_desc.tensor_data);
    } else {
      GELOGE(ge::GRAPH_FAILED, "unsupported copy form placement %s to %s", GetPlacementStr(src_placement),
             GetPlacementStr(dst_placement));
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CopyH2D)
    .RunFunc(CopyH2D)
    .OutputsCreator(CreateTensorDataAtDeviceHbm)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(MakeSureTensorAtHost)
    .RunFunc(MakeSureTensorAtHost)
    .OutputsCreator(CreateTensorDataAtHost)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(MakeSureTensorAtDevice)
    .RunFunc(MakeSureTensorAtDevice)
    .OutputsCreator(CreateTensorDataAtDeviceHbm)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus CopyTensorDataH2H(KernelContext *context) {
  auto src_gtd =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MemoryCopyH2HInputs::kSrcAddress));
  auto src_shape = context->GetInputPointer<StorageShape>(static_cast<size_t>(MemoryCopyH2HInputs::kSrcShape));
  auto data_type = context->GetInputValue<ge::DataType>(static_cast<size_t>(MemoryCopyH2HInputs::kSrcDataType));
  auto gert_allocator = context->GetInputValue<GertAllocator *>(static_cast<size_t>(MemoryCopyH2HInputs::kAllocator));
  auto host_gtd = context->GetOutputPointer<gert::GertTensorData>(0);
  if ((src_gtd == nullptr) || (src_shape == nullptr) ||
      (host_gtd == nullptr || gert_allocator == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  // calc tensor data size by shape
  auto tensor_data_size = ge::GetSizeInBytes(src_shape->GetStorageShape().GetShapeSize(), data_type);
  size_t alloc_size;
  GE_ASSERT_TRUE(CalcSize(tensor_data_size, alloc_size));

  auto host_block = gert_allocator->Malloc(alloc_size);
  KERNEL_CHECK_NOTNULL(host_block);
  KERNEL_CHECK(host_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", alloc_size);
  if (tensor_data_size > 0U) {
    GE_CHK_RT_RET(rtMemcpy(host_block->GetAddr(), static_cast<uint64_t>(tensor_data_size), src_gtd->GetAddr(),
                           tensor_data_size, RT_MEMCPY_HOST_TO_HOST));
  }
  host_gtd->ShareFrom(GertTensorData{static_cast<size_t>(tensor_data_size), host_gtd->GetPlacement(),
                                     gert_allocator->GetStreamId(), host_block});
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CopyTensorDataH2H)
    .RunFunc(CopyTensorDataH2H)
    .OutputsCreator(CreateTensorDataAtHost)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus CopyD2D(KernelContext *context) {
  auto src_tensor_data =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MemoryCopyD2DInputs::kSrcAddress));
  auto dst_tensor_data =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MemoryCopyD2DInputs::kDstAddress));
  auto tensor_size = context->GetInputValue<size_t>(static_cast<size_t>(MemoryCopyD2DInputs::kTensorSize));
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(MemoryCopyInputs::kStream));
  if (src_tensor_data == nullptr || dst_tensor_data == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (tensor_size > 0U) {
    GE_CHK_RT_RET(rtMemcpyAsyncWithoutCheckKind(dst_tensor_data->GetAddr(), tensor_size,
                                                src_tensor_data->GetAddr(), tensor_size,
                                                RT_MEMCPY_DEVICE_TO_DEVICE, stream));
  }

  dst_tensor_data->SetPlacement(kOnDeviceHbm);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CopyD2D).RunFunc(CopyD2D).OutputsCreator(CreateTensorDataAtDeviceHbm);

static auto g_copy_type = TableDriven2<kTensorPlacementEnd, kTensorPlacementEnd, tagRtMemcpyKind>(RT_MEMCPY_RESERVED)
    .Add(kOnHost, kOnDeviceHbm, RT_MEMCPY_HOST_TO_DEVICE_EX)
    .Add(kOnHost, kOnHost, RT_MEMCPY_HOST_TO_HOST)
    .Add(kOnHost, kFollowing, RT_MEMCPY_HOST_TO_HOST)
    .Add(kOnHost, kOnDeviceP2p, RT_MEMCPY_HOST_TO_DEVICE_EX)
    .Add(kFollowing, kOnDeviceHbm, RT_MEMCPY_HOST_TO_DEVICE_EX)
    .Add(kFollowing, kOnHost, RT_MEMCPY_HOST_TO_HOST)
    .Add(kFollowing, kFollowing, RT_MEMCPY_HOST_TO_HOST)
    .Add(kFollowing, kOnDeviceP2p, RT_MEMCPY_HOST_TO_DEVICE_EX)
    .Add(kOnDeviceHbm, kOnDeviceHbm, RT_MEMCPY_DEVICE_TO_DEVICE)
    .Add(kOnDeviceHbm, kOnHost, RT_MEMCPY_DEVICE_TO_HOST)
    .Add(kOnDeviceHbm, kFollowing, RT_MEMCPY_DEVICE_TO_HOST)
    .Add(kOnDeviceHbm, kOnDeviceP2p, RT_MEMCPY_DEVICE_TO_DEVICE)
    .Add(kOnDeviceP2p, kOnDeviceHbm, RT_MEMCPY_DEVICE_TO_DEVICE)
    .Add(kOnDeviceP2p, kOnHost, RT_MEMCPY_DEVICE_TO_HOST)
    .Add(kOnDeviceP2p, kFollowing, RT_MEMCPY_DEVICE_TO_HOST)
    .Add(kOnDeviceP2p, kOnDeviceP2p, RT_MEMCPY_DEVICE_TO_DEVICE);

ge::graphStatus TensorToOut(const StorageShape &shape, GertTensorData &tensor_data,
                            const BuildTensorAttr *tensor_attr, aclrtStream stream, Tensor &out_tensor) {
  // 外部没有分配内存，share out
  auto dst_address = out_tensor.GetAddr();
  if (dst_address == nullptr) {
    out_tensor.SetPlacement(tensor_attr->placement);
    auto msb = reinterpret_cast<memory::MultiStreamMemBlock *>(tensor_data.GetGertMemBlock());
    if (msb != nullptr) {
      auto l2_allocator = msb->GetBirthAllocator();
      GE_ASSERT_SUCCESS(l2_allocator->MoveL2ToL1(msb));
    }
    TensorUtils::ShareGtdToTd(tensor_data, out_tensor.MutableTensorData());
    return ge::GRAPH_SUCCESS;
  }

  auto shape_size = shape.GetStorageShape().GetShapeSize();
  const auto copy_size = ge::GetSizeInBytes(shape_size, tensor_attr->data_type);
  if (copy_size < 0) {
    GELOGE(ge::GRAPH_FAILED, "[Calc][TensorSizeByShape] shape_size[%" PRId64 "], data_type[%d]", shape_size,
           static_cast<int32_t>(tensor_attr->data_type));
    return ge::GRAPH_FAILED;
  }

  if (static_cast<size_t>(copy_size) > out_tensor.GetTensorData().GetSize()) {
    GELOGE(ge::PARAM_INVALID,
           "Failed to copy output tensor data to the given buffer, given buffer size %zu is less than output tensor "
           "size %" PRId64, out_tensor.GetTensorData().GetSize(), copy_size);
    return ge::GRAPH_FAILED;
  }

  // 零拷贝生效，不需要拷贝了
  auto src_address = tensor_data.GetAddr();
  if (src_address == dst_address) {
    GELOGI("Zero copy takes effect, output addr:%lx", src_address);
    return ge::GRAPH_SUCCESS;
  }

  // 外部申请了内存，且零拷贝没有生效，需要拷贝
  // todo: 不应该在这里校验，应该在AllocModeOutensor申请内存的时候，校验要申请的size不能小于storage shape 不加padding计算的size。
  // 需要将shape传到AllocModeOutensor中

  // 内部分配的内存大小可能是CalcTensorSizeFromShape计算的，和copy_size可能不一样。
  // 当信任out_tensor时，如果外部传入tensor的origin shape比storage shape小，可能会满足这个条件
  if (static_cast<size_t>(copy_size) > tensor_data.GetSize()) {
    GELOGE(ge::PARAM_INVALID,
           "Failed to copy output tensor data to the given buffer, output tensor data size %zu is less than copy size "
           "size %" PRId64 ", src_address: %p", tensor_data.GetSize(), copy_size, src_address);
    return ge::GRAPH_FAILED;
  }

  if (copy_size > 0U) {
    auto copy_type = g_copy_type.Find(tensor_data.GetPlacement(), out_tensor.GetPlacement());
    if (copy_type == RT_MEMCPY_RESERVED) {
      GELOGE(
          ge::PARAM_INVALID,
          "Failed to copy output tensor to the given buffer, do not support the copy direction, from %s(%d) to %s(%d)",
          GetPlacementStr(tensor_data.GetPlacement()), tensor_data.GetPlacement(),
          GetPlacementStr(out_tensor.GetPlacement()), out_tensor.GetPlacement());
      return ge::GRAPH_FAILED;
    }
    if (copy_type == RT_MEMCPY_HOST_TO_HOST) {
      GE_CHK_RT_RET(rtMemcpy(dst_address, out_tensor.GetTensorData().GetSize(), src_address, copy_size, copy_type));
      return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_RT_OK(rtMemcpyAsyncWithoutCheckKind(dst_address, out_tensor.GetTensorData().GetSize(),
                                                  src_address, copy_size, copy_type, stream));
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus EnsureTensorAtOutMemory(KernelContext *context) {
  auto shape = context->GetInputPointer<StorageShape>(static_cast<size_t>(BuildTensorInputs::kShape));
  auto gtd = context->MutableInputPointer<GertTensorData>(static_cast<size_t>(BuildTensorInputs::kTensorData));
  auto tensor_attr = context->GetInputPointer<BuildTensorAttr>(static_cast<size_t>(BuildTensorInputs::kTensorAttr));
  auto output_tensor_chain = context->MutableInput(static_cast<size_t>(EnsureTensorAtOutMemoryInputs::kOutputData));
  if (shape == nullptr || gtd == nullptr || tensor_attr == nullptr || output_tensor_chain == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto out_tensor = output_tensor_chain->GetPointer<Tensor>();
  // 外部未指定输出Tensor，创建一个，这种场景应该很罕见，应该是个异常场景，考虑直接返回失败得了
  if (out_tensor == nullptr) {
    out_tensor = new (std::nothrow) Tensor();
    GE_ASSERT_NOTNULL(out_tensor);
    output_tensor_chain->SetWithDefaultDeleter(out_tensor);
  }

  // 更新Tensor描述
  out_tensor->MutableStorageShape() = shape->GetStorageShape();
  out_tensor->MutableOriginShape() = shape->GetOriginShape();
  out_tensor->MutableFormat() = tensor_attr->storage_format;
  out_tensor->SetDataType(tensor_attr->data_type);

  return TensorToOut(*shape, *gtd, tensor_attr,
                     // stream 为空指针的时候，代表默认流，因此stream不需要校验空指针
                     context->GetInputValue<aclrtStream>(static_cast<size_t>(EnsureTensorAtOutMemoryInputs::kStream)),
                     *out_tensor);
}
REGISTER_KERNEL(EnsureTensorAtOutMemory)
    .RunFunc(EnsureTensorAtOutMemory)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

graphStatus CalcStringTensorSize(KernelContext *context) {
  auto datatype = context->GetInputValue<ge::DataType>(static_cast<size_t>(CalcStringTensorSizeInputs::kSrcDataType));
  auto storage_shape =
      context->GetInputPointer<StorageShape>(static_cast<size_t>(CalcStringTensorSizeInputs::kSrcShape));
  auto tensor_data =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(CalcStringTensorSizeInputs::kSrcAddress));
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(CalcStringTensorSizeInputs::kStream));
  auto string_tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  GE_ASSERT_NOTNULL(storage_shape);
  GE_ASSERT_NOTNULL(tensor_data);
  GE_ASSERT_NOTNULL(string_tensor_size_ptr);

  GE_ASSERT_GRAPH_SUCCESS(CalcStringTensorSize(tensor_data, stream, storage_shape, datatype, *string_tensor_size_ptr));
  GELOGD("[Calc][String], tensor_size is %zu, aline_size is %zu", tensor_data->GetSize(), *string_tensor_size_ptr);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CalcStringTensorSize).RunFunc(CalcStringTensorSize);

ge::graphStatus SinkWeightData(KernelContext *context) {
  auto src_info = context->GetInputPointer<GertTensorData>(static_cast<size_t>(SinkWeightDataInputs::kWeightData));
  auto dest_mem_info =
      context->GetInputPointer<gert::GertTensorData>(static_cast<size_t>(SinkWeightDataInputs::kDeviceBaseAddr));
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(SinkWeightDataInputs::kStream));
  GE_ASSERT_NOTNULL(src_info);
  GE_ASSERT_NOTNULL(dest_mem_info);

  auto dest_addr = dest_mem_info->GetAddr();
  auto dest_size = dest_mem_info->GetSize();
  auto src_addr = src_info->GetAddr();
  auto src_size = src_info->GetSize();
  // check where offset + size > total_mem_size
  if (dest_size < src_size) {
    GELOGE(ge::GRAPH_FAILED, "sink data is invalid. src_size[%zu], dest_size[%zu]", src_size, dest_size);
    return ge::GRAPH_FAILED;
  }
  if (src_size > 0U) {
    // rts will copy another host memory if use RT_MEMCPY_HOST_TO_DEVICE_EX,
    // here is weight mem, ge can ensure its lifetime so use RT_MEMCPY_HOST_TO_DEVICE to reduce host mem
    GE_CHK_RT_RET(rtMemcpyAsync(dest_addr, dest_size, src_addr, src_size, RT_MEMCPY_HOST_TO_DEVICE, stream));
  }
  auto device_gtd =
      context->GetOutputPointer<gert::GertTensorData>(static_cast<size_t>(SinkWeightDataOutputs::kTensorData));
  GE_ASSERT_NOTNULL(device_gtd);
  auto stream_id = context->GetInputPointer<int64_t>(static_cast<size_t>(SinkWeightDataInputs::kStreamId));
  GE_ASSERT_NOTNULL(stream_id);
  *device_gtd = GertTensorData{dest_addr, src_size, device_gtd->GetPlacement(), *stream_id};
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkWeightData).RunFunc(SinkWeightData).OutputsCreator(SinkWeightDataOutputCreator);
}  // namespace kernel
}  // namespace gert
