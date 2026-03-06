/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_kernel.h"

#include <cinttypes>
#include "runtime/rt.h"
#include "common/checker.h"
#include "kernel/memory.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "kernel/kernel_log.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "runtime/gert_api.h"
#include "exe_graph/runtime/tensor_data_utils.h"
#include "common/model/external_allocator_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace kernel {
namespace {
constexpr int64_t kMaxWorkspaceCount = 16;
}
/**
 * 创建Allocator
 * kernel输入
 *
 * @param context
 * @return
 */
ge::graphStatus CreateL1Allocator(KernelContext *context) {
  auto allocator_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(allocator_chain);

  auto placement = context->GetInputValue<TensorPlacement>(static_cast<size_t>(CreateAllocatorInputs::kPlacement));
  auto allocator = AllocatorFactory::Create(placement);
  GE_ASSERT_NOTNULL(allocator);
  allocator_chain->SetWithDefaultDeleter(allocator.release());
  KERNEL_TRACE("[MEM]create l1 allocators, placement: %s", GetPlacementStr(placement));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateL1Allocator).RunFunc(CreateL1Allocator);

ge::graphStatus SelectL1Allocator(KernelContext *context) {
  auto out_allocator = context->GetOutput(0);
  GE_ASSERT_NOTNULL(out_allocator);
  auto placement = context->GetInputValue<TensorPlacement>(static_cast<size_t>(SelectL1AllocatorInputs::kPlacement));
  GE_ASSERT_TRUE(static_cast<size_t>(placement) < static_cast<size_t>(TensorPlacement::kTensorPlacementEnd));
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(SelectL1AllocatorInputs::kStream));

  auto external_allocator =
      context->GetInputValue<Allocators *>(static_cast<size_t>(SelectL1AllocatorInputs::kExternalAllocator));
  if (external_allocator != nullptr) {
    auto allocator = external_allocator->GetAllocator(placement, 0);
    if (allocator != nullptr) {
      auto caching_mem_allocator = dynamic_cast<memory::CachingMemAllocator *>(allocator);
      if (caching_mem_allocator != nullptr) {
        caching_mem_allocator->SetStream(stream);
        KERNEL_TRACE("caching_mem_allocator: %p, stream: %p, placement: %s",
                     caching_mem_allocator, stream, GetPlacementStr(placement));
      }
      out_allocator->Set(allocator, nullptr);
      return ge::GRAPH_SUCCESS;
    }
  }
  auto created_allocator =
      context->GetInputValue<ge::Allocator *>(static_cast<size_t>(SelectL1AllocatorInputs::kCreatedAllocator));
  GE_ASSERT_NOTNULL(created_allocator);
  auto caching_mem_allocator = dynamic_cast<memory::CachingMemAllocator *>(created_allocator);
  if (caching_mem_allocator != nullptr) {
    caching_mem_allocator->SetStream(stream);
    KERNEL_TRACE("created_allocator: %p, stream: %p, placement: %s",
                 caching_mem_allocator, stream, GetPlacementStr(placement));
  }

  out_allocator->Set(created_allocator, nullptr);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SelectL1Allocator).RunFunc(SelectL1Allocator);

ge::graphStatus GetExternalL1Allocator(KernelContext *context) {
  auto out_allocator = context->GetOutput(0);
  GE_ASSERT_NOTNULL(out_allocator);
  auto placement = context->GetInputValue<TensorPlacement>(static_cast<size_t>(GetAllocatorInputs::kPlacement));
  if (placement >= kTensorPlacementEnd) {
    GELOGE(ge::PARAM_INVALID, "Invalid placement or memory type, placement: %zu",
           static_cast<size_t>(placement));
    return ge::GRAPH_FAILED;
  }
  auto external_allocator =
      context->GetInputValue<Allocators *>(static_cast<size_t>(GetAllocatorInputs::kExternalAllocator));
  GE_ASSERT_NOTNULL(external_allocator);
  auto allocator = external_allocator->GetAllocator(placement, 0);
  GE_ASSERT_NOTNULL(allocator);
  out_allocator->Set(allocator, nullptr);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetExternalL1Allocator).RunFunc(GetExternalL1Allocator);

ge::graphStatus CreateInitL2Allocator(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto chain = context->GetOutput(static_cast<size_t>(CreateInitL2AllocatorOutputs::kInitL2Allocator));
  GE_ASSERT_NOTNULL(chain);
  auto vec_chain =
      context->GetOutput(static_cast<size_t>(CreateInitL2AllocatorOutputs::kL2AllocatorsForInitL2Allocator));
  GE_ASSERT_NOTNULL(vec_chain);

  auto l2_allocators = context->GetInputPointer<TypedContinuousVector<GertAllocator *>>(
      static_cast<size_t>(CreateInitL2AllocatorInputs::kL2Allocators));
  GE_ASSERT_NOTNULL(l2_allocators);
  GE_ASSERT_NOTNULL(l2_allocators->GetData()[0U]); // init in CreateL2Allocators
  const auto placement = reinterpret_cast<GertAllocator *>(l2_allocators->GetData()[0U])->GetPlacement();
  const auto stream_num = l2_allocators->GetSize();
  KERNEL_TRACE("[MEM]create init l2 allocator, stream_num: %zu, placement: %s",
               stream_num, GetPlacementStr(placement));
  if (stream_num > 1U) {
    auto init_l2_allocators_holder = ContinuousVector::Create<memory::MultiStreamL2Allocator *>(stream_num);
    GE_ASSERT_NOTNULL(init_l2_allocators_holder);
    auto init_l2_allocators =
        reinterpret_cast<TypedContinuousVector<memory::MultiStreamL2Allocator *> *>(init_l2_allocators_holder.get());
    init_l2_allocators->SetSize(static_cast<size_t>(stream_num));
    auto init_l2_allocators_vec = init_l2_allocators->MutableData();

    auto init_l2_allocator =
        new (std::nothrow) memory::MultiStreamL2Allocator(0, placement, init_l2_allocators, nullptr);
    GE_ASSERT_NOTNULL(init_l2_allocator);
    // init图上的L2 Allocator里持有的数组的第1个元素一定是本身，其余的元素和main图上的保持一致
    init_l2_allocators_vec[0] = init_l2_allocator;
    for (size_t i = 1UL; i < static_cast<size_t>(stream_num); i++) {
      init_l2_allocators_vec[i] = reinterpret_cast<memory::MultiStreamL2Allocator *>(l2_allocators->GetData()[i]);
    }

    chain->SetWithDefaultDeleter(init_l2_allocator);
    auto deleter = [](void *ptr) { delete[] static_cast<uint8_t *>(ptr); };
    vec_chain->Set(init_l2_allocators_holder.release(), deleter);
    return ge::GRAPH_SUCCESS;
  }

  auto init_l2_allocator = new (std::nothrow) memory::SingleStreamL2Allocator(placement, nullptr);
  GE_ASSERT_NOTNULL(init_l2_allocator);
  chain->SetWithDefaultDeleter(init_l2_allocator);
  return ge::GRAPH_SUCCESS;
}

/**
 * 为Init图创建MultiStreamL2Allocator
 * kernel输入
 *
 * @param context
 * @return
 */
ge::graphStatus BindingL1Allocator(KernelContext *context) {
  auto l2_allocators = context->GetInputPointer<TypedContinuousVector<GertAllocator *>>(
      static_cast<size_t>(CreateInitL2AllocatorInputs::kL2Allocators));
  GE_ASSERT_NOTNULL(l2_allocators);
  auto stream_num = l2_allocators->GetSize();

  auto l2_allocator =
      context->GetOutputPointer<GertAllocator>(static_cast<size_t>(CreateInitL2AllocatorOutputs::kInitL2Allocator));
  GE_ASSERT_NOTNULL(l2_allocator);
  if (stream_num > 1U) {
    auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(CreateInitL2AllocatorInputs::kStream));
    reinterpret_cast<memory::MultiStreamL2Allocator *>(l2_allocator)->SetRtsStream(stream);
  }

  auto l1_allocator =
      context->GetInputValue<ge::Allocator *>(static_cast<size_t>(CreateInitL2AllocatorInputs::kL1Allocator));
  GE_ASSERT_NOTNULL(l1_allocator);
  l2_allocator->SetL1Allocator(l1_allocator);
  KERNEL_TRACE("l1 allocator's placement is Device, InitL2Allocator %p set l1 allocator %p", l2_allocator,
               l1_allocator);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateInitL2Allocator)
    .RunFunc(BindingL1Allocator)
    .OutputsCreator(CreateInitL2Allocator);

static void MultiL2AllocatorsDeleter(void *ptr) {
  auto l2_allocators = reinterpret_cast<TypedContinuousVector<GertAllocator *> *>(ptr);
  auto l2_allocators_data = l2_allocators->MutableData();
  for (size_t i = 0U; i < l2_allocators->GetSize(); ++i) {
    if (l2_allocators_data[i] != nullptr) {
      delete l2_allocators_data[i];
      l2_allocators_data[i] = nullptr;
    }
  }
  l2_allocators->SetSize(0U);
  delete[] static_cast<uint8_t *>(ptr);
}

/**
 * CreateL2Allocators by stream num for main graph
 * kernel输入
 *
 * @param context
 * @return
 */
ge::graphStatus CreateL2Allocators(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto stream_num =
      context->GetInputPointer<int64_t>(static_cast<size_t>(CreateL2AllocatorsInputs::kStreamNum));
  GE_ASSERT_NOTNULL(stream_num);

  TensorPlacement placement = kOnDeviceHbm;
  // todo: 兼容性处理，等待一次黄传蓝即可将if删除
  if (context->GetInputNum() > 1U) {
    placement = context->GetInputValue<TensorPlacement>(static_cast<size_t>(CreateL2AllocatorsInputs::kPlacement));
    GE_ASSERT_TRUE(placement < kTensorPlacementEnd, "Invalid placement or memory type, placement: %zu",
                   static_cast<size_t>(placement));
  }

  size_t l2_allocators_size = *stream_num;
  GE_ASSERT_TRUE(l2_allocators_size > 0U);

  auto l2_allocators_holder = ContinuousVector::Create<GertAllocator *>(l2_allocators_size);
  GE_ASSERT_NOTNULL(l2_allocators_holder);
  auto l2_allocators =
      reinterpret_cast<TypedContinuousVector<GertAllocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(static_cast<size_t>(l2_allocators_size));
  KERNEL_TRACE("Create l2 allocators, num %zu", l2_allocators_size);

  auto all_l2_mem_pool_holder = ContinuousVector::Create<memory::L2MemPool *>(*stream_num);
  GE_ASSERT_NOTNULL(all_l2_mem_pool_holder);
  auto all_l2_mem_pool = reinterpret_cast<TypedContinuousVector<memory::L2MemPool *> *>(all_l2_mem_pool_holder.get());
  all_l2_mem_pool->SetSize(l2_allocators_size);

  if (l2_allocators_size > 1U) {
    // 创建stream_num个MultiStreamL2Allocator,
    // 每个MultiStreamL2Allocator都需要持有所有的流对应的MultiStreamL2Allocator
    // 这样才能保证更新bitmap的时候，通过stream_id找到对应的MultiStreamL2Allocator
    for (size_t i = 0U; i < l2_allocators_size; ++i) {
      auto multi_stream_allocator = new (std::nothrow) memory::MultiStreamL2Allocator(
          i, placement, reinterpret_cast<TypedContinuousVector<memory::MultiStreamL2Allocator *> *>(l2_allocators),
          all_l2_mem_pool);
      GE_ASSERT_NOTNULL(multi_stream_allocator, "Failed to create l2 allocator at stream %" PRId64, i);
      l2_allocators->MutableData()[i] = multi_stream_allocator;
      all_l2_mem_pool->MutableData()[i] = multi_stream_allocator->GetL2MemPool();
    }
  } else {
    auto single_stream_allocator = new (std::nothrow) memory::SingleStreamL2Allocator(placement, nullptr);
    GE_ASSERT_NOTNULL(single_stream_allocator, "Failed to create l2 allocator at stream %" PRId64, 0);
    l2_allocators->MutableData()[0] = single_stream_allocator;
  }

  auto chain = context->GetOutput(static_cast<size_t>(CreateL2AllocatorsOutputs::kL2Allocators));
  GE_ASSERT_NOTNULL(chain);
  chain->Set(l2_allocators_holder.release(), MultiL2AllocatorsDeleter);
  auto l2_mem_pools_chain = context->GetOutput(static_cast<size_t>(CreateL2AllocatorsOutputs::kL2MemPools));
  GE_ASSERT_NOTNULL(l2_mem_pools_chain);
  auto deleter = [](void *ptr) { delete[] static_cast<uint8_t *>(ptr); };
  l2_mem_pools_chain->Set(all_l2_mem_pool_holder.release(), deleter);
  KERNEL_TRACE("[MEM]create l2 allocators, stream_num: %zu, placement: %s",
               *stream_num, GetPlacementStr(placement));
  return ge::GRAPH_SUCCESS;
}

/**
 * 创建MultiStreamL2Allocators
 * kernel输入
 *
 * @param context
 * @return
 */
ge::graphStatus EmptyRunFunc(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateL2Allocators).RunFunc(EmptyRunFunc).OutputsCreator(CreateL2Allocators);

ge::graphStatus CreateHostL2Allocator(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto gert_allocator = new (std::nothrow) memory::HostGertMemAllocator();
  GE_ASSERT_NOTNULL(gert_allocator);
  gert_allocator->SetPlacement(TensorPlacement::kOnHost);
  auto chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(gert_allocator);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus BindingHostL1Allocator(KernelContext *context) {
  auto l1_allocator =
      context->GetInputValue<ge::Allocator *>(static_cast<size_t>(CreateInitL2AllocatorInputs::kL1Allocator));
  GE_ASSERT_NOTNULL(l1_allocator);
  auto l2_allocator = context->GetOutputPointer<memory::HostGertMemAllocator>(0);
  l2_allocator->SetL1Allocator(l1_allocator);
  KERNEL_TRACE("l1 allocator's placement is Host, InitL2Allocator %p set l1 allocator %p", l2_allocator, l1_allocator);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(CreateHostL2Allocator).RunFunc(BindingHostL1Allocator).OutputsCreator(CreateHostL2Allocator);

ge::graphStatus SelectL2Allocator(KernelContext *context) {
  auto stream_id_ptr = context->GetInputPointer<int64_t>(static_cast<size_t>(SelectL2AllocatorInputs::kStreamId));
  GE_ASSERT_NOTNULL(stream_id_ptr);
  auto stream_id = *stream_id_ptr;

  auto l2_allocators = context->GetInputPointer<TypedContinuousVector<GertAllocator *>>(
      static_cast<size_t>(SelectL2AllocatorInputs::kL2Allocators));
  GE_ASSERT_NOTNULL(l2_allocators);
  if (l2_allocators->GetSize() <= static_cast<size_t>(stream_id)) {
    GELOGE(ge::PARAM_INVALID, "Failed to select allocator, invalid stream id %lld, should be less than %zu",
           stream_id, l2_allocators->GetSize());
    return ge::PARAM_INVALID;
  }

  // select l2 allocator according to stream id
  auto l2_allocator = l2_allocators->GetData()[stream_id];
  GE_ASSERT_NOTNULL(l2_allocator);
  const auto placement = l2_allocator->GetPlacement();
  GE_ASSERT_TRUE(TensorPlacementUtils::IsOnDevice(placement));

  // binding l1 allocator to l2 allocator
  auto l1_allocator =
      context->GetInputValue<ge::Allocator *>(static_cast<size_t>(SelectL2AllocatorInputs::kL1Allocator));
  GE_ASSERT_NOTNULL(l1_allocator);
  GE_ASSERT_GRAPH_SUCCESS(l2_allocator->SetL1Allocator(l1_allocator));

  // set rt streams to multi stream l2 allocator
  // todo set stream abstract to interface
  if (l2_allocators->GetSize() > 1U) {
    auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(SelectL2AllocatorInputs::kStream));
    (reinterpret_cast<memory::MultiStreamL2Allocator *>(l2_allocator))->SetRtsStream(stream);
    KERNEL_TRACE("[MEM]select l2 allocator, stream: %p, placement: %s",
                 stream, GetPlacementStr(placement));
  } else {
    KERNEL_TRACE("[MEM]select l2 allocator, placement: %s", GetPlacementStr(placement));
  }

  auto allocator_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(allocator_chain);
  allocator_chain->Set(l2_allocator, nullptr);  // CreateL2Allocator中已经设置了deleter,这里设置为nullptr
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SelectL2Allocator).RunFunc(SelectL2Allocator);

/**
 * 根据allocator申请hbm内存或者Host内存
 *
 * kernel输入
 *
 * |序号|数据类型|描述|
 * |--|---|---|
 * |0|基类MemAllocator|内存申请使用的allocator|
 * |1|size_t|输出Tensor的size，单位是字节|
 *
 * kernel输出
 *
 * |序号|数据类型|描述|
 * |--|---|---|
 * |0|GertTensorData|内存Block|
 *
 * @param context
 * @return
 */
// todo : rename AllocHbmMem to AllocMem
ge::graphStatus AllocHbmMem(KernelContext *context) {
  auto gert_allocator = context->GetInputValue<GertAllocator *>(static_cast<int32_t>(AllocHbmMemInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  const int64_t stream_id = gert_allocator->GetStreamId();
  const auto placement = gert_allocator->GetPlacement();
  for (auto i = static_cast<size_t>(AllocHbmMemInputs::kSizes); i < context->GetInputNum(); i++) {
    auto tensor_size = context->GetInputValue<size_t>(static_cast<size_t>(i));
    auto gert_tensor_data =
        context->GetOutputPointer<GertTensorData>(i - static_cast<size_t>(AllocHbmMemInputs::kSizes));
    KERNEL_CHECK_NOTNULL(gert_tensor_data);
    auto gert_mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(tensor_size));
    KERNEL_CHECK((gert_mem_block != nullptr) && (gert_mem_block->GetAddr() != nullptr),
                 "malloc failed, stream %" PRId64 ", tensor size=%zu, index=%zu",
                 stream_id, tensor_size, i);
    *gert_tensor_data = TensorUtils::ToGertTensorData(gert_mem_block, placement, stream_id);
    KERNEL_TRACE(TRACE_STR_ALLOC_MEM ", tensor size %zu, index %zu", stream_id, gert_mem_block,
                 gert_mem_block->GetAddr(), gert_mem_block->GetSize(), tensor_size, i);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllocOutputTensorData(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  for (size_t i = 0; i < context->GetOutputNum(); i++) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);
    auto gert_tensor_data = new (std::nothrow) GertTensorData();
    GE_ASSERT_NOTNULL(gert_tensor_data);
    chain->SetWithDefaultDeleter(gert_tensor_data);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FreeHbmMem(KernelContext *context) {
  for (size_t i = 0U; i < context->GetInputNum(); i++) {
    auto gert_tensor_data = context->GetInputValue<GertTensorData *>(i);
    if (gert_tensor_data != nullptr) {
      KERNEL_TRACE_FREE_MEM(gert_tensor_data->GetStreamId(), gert_tensor_data->GetAddr());
      gert_tensor_data->Free();
    } else {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus FreeHbmMemHoldAddr(KernelContext *context) {
  for (size_t i = 0U; i < context->GetInputNum(); i++) {
    auto gert_tensor_data = context->GetInputValue<GertTensorData *>(i);
    if (gert_tensor_data != nullptr) {
      KERNEL_TRACE_FREE_MEM(gert_tensor_data->GetStreamId(), gert_tensor_data->GetAddr());
      gert_tensor_data->FreeHoldAddr();
    } else {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FreeMemory(KernelContext *context) {
  auto gert_tensor_data = context->MutableInputPointer<GertTensorData>(0);
  if (gert_tensor_data == nullptr) {
    return ge::GRAPH_FAILED;
  }
  KERNEL_TRACE_FREE_MEM(gert_tensor_data->GetStreamId(), gert_tensor_data->GetAddr());
  return gert_tensor_data->Free();
}

ge::graphStatus FreeMemoryHoldAddr(KernelContext *context) {
  auto gert_tensor_data = context->MutableInputPointer<GertTensorData>(0);
  if (gert_tensor_data == nullptr) {
    return ge::GRAPH_FAILED;
  }
  KERNEL_TRACE_FREE_MEM(gert_tensor_data->GetStreamId(), gert_tensor_data->GetAddr());
  return gert_tensor_data->FreeHoldAddr();
}

ge::graphStatus FreeTensorMemory(KernelContext *context) {
  auto tensor = context->MutableInputPointer<Tensor>(0);
  if (tensor == nullptr) {
    return ge::GRAPH_FAILED;
  }
  KERNEL_TRACE(TRACE_STR_FREE_MEM ", tensor %p", static_cast<int64_t>(-1), tensor->GetTensorData().GetAddr(), tensor);
  return tensor->MutableTensorData().Free();
}

static void FreeBatchHbm(TypedContinuousVector<GertTensorData *> *addrs) {
  auto addrs_data = addrs->MutableData();
  for (size_t i = 0U; i < addrs->GetSize(); ++i) {
    if (addrs_data[i] != nullptr) {
      addrs_data[i]->Free();
    }
  }
  addrs->SetSize(0);
}
ge::graphStatus AllocBatchHbm(KernelContext *context) {
  auto gert_allocator = context->GetInputValue<GertAllocator *>(0);
  GE_ASSERT_NOTNULL(gert_allocator);
  auto sizes = context->GetInputPointer<TypedContinuousVector<size_t>>(1);
  auto addrs = context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0);
  KERNEL_CHECK_NOTNULL(sizes);
  KERNEL_CHECK_NOTNULL(addrs);
  if (sizes->GetSize() > addrs->GetCapacity()) {
    GELOGE(ge::GRAPH_FAILED, "[AllocBatchHbm] The alloc count %zu more than output capacity %zu", sizes->GetSize(),
           addrs->GetCapacity());
    return ge::GRAPH_FAILED;
  }
  FreeBatchHbm(addrs);

  auto sizes_data = sizes->GetData();
  auto addrs_data = static_cast<GertTensorData**>(addrs->MutableData());
  KERNEL_CHECK_NOTNULL(sizes_data);
  KERNEL_CHECK_NOTNULL(addrs_data);
  addrs->SetSize(sizes->GetSize());
  for (size_t i = 0; i < sizes->GetSize(); ++i) {
    const int64_t stream_id = gert_allocator->GetStreamId();
    const auto placement = gert_allocator->GetPlacement();
    if (sizes_data[i] == 0) {
      *(addrs_data[i]) = TensorUtils::ToGertTensorData(nullptr, placement, stream_id);
      continue;
    }
    auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(sizes_data[i]));
    if (mem_block == nullptr) {
      addrs->SetSize(i);
      GELOGE(ge::GRAPH_FAILED, "Get nullptr %s", mem_block);
      return ge::GRAPH_FAILED;
    }
    KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu, index=%zu", sizes_data[i], i);
    *(addrs_data[i]) = TensorUtils::ToGertTensorData(mem_block, placement, stream_id);
    KERNEL_TRACE(TRACE_STR_ALLOC_MEM ", index %zu", gert_allocator->GetStreamId(), mem_block, addrs_data[i]->GetAddr(),
                 sizes_data[i], i);
  }
  return ge::GRAPH_SUCCESS;
}

static void BatchHbmDeleter(void *memories) {
  auto tensor_data_vec = static_cast<TypedContinuousVector<GertTensorData *> *>(memories);
  FreeBatchHbm(tensor_data_vec);
  for (size_t i = 0U; i < static_cast<size_t>(kMaxWorkspaceCount); i++) {
    delete tensor_data_vec->MutableData()[i];
  }
  delete[] static_cast<uint8_t *>(memories);
}
ge::graphStatus CreateAllocBatchHbmOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  auto addrs = ContinuousVector::Create<GertTensorData *>(kMaxWorkspaceCount);
  GE_ASSERT_NOTNULL(addrs);
  auto tensor_data_vec = reinterpret_cast<ContinuousVector*>(addrs.get());
  tensor_data_vec->SetSize(kMaxWorkspaceCount);
  auto tensor_data_addr = static_cast<GertTensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < static_cast<size_t>(kMaxWorkspaceCount); i++) {
    tensor_data_addr[i] = new (std::nothrow) GertTensorData();
    GE_ASSERT_NOTNULL(tensor_data_addr[i]);
  }
  chain->Set(addrs.release(), BatchHbmDeleter);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FreeBatchHbm(KernelContext *context) {
  auto addrs = context->MutableInputPointer<TypedContinuousVector<GertTensorData *>>(0);
  if (addrs == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[FreeBatchHbm] input addrs para error");
    return ge::GRAPH_FAILED;
  }
  auto addrs_data = addrs->MutableData();
  KERNEL_CHECK_NOTNULL(addrs_data);
  for (size_t i = 0U; i < addrs->GetSize(); ++i) {
    KERNEL_TRACE(TRACE_STR_FREE_MEM ", index %zu", addrs_data[i]->GetStreamId(), addrs_data[i]->GetAddr(), i);
    addrs_data[i]->Free();
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FreeBatchHbmHoldAddr(KernelContext *context) {
  auto addrs = context->MutableInputPointer<TypedContinuousVector<GertTensorData *>>(0);
  if (addrs == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "[FreeBatchHbm] input addrs para error");
    return ge::GRAPH_FAILED;
  }
  auto addrs_data = addrs->MutableData();
  KERNEL_CHECK_NOTNULL(addrs_data);
  for (size_t i = 0U; i < addrs->GetSize(); ++i) {
    KERNEL_TRACE(TRACE_STR_FREE_MEM ", index %zu", addrs_data[i]->GetStreamId(), addrs_data[i]->GetAddr(), i);
    addrs_data[i]->FreeHoldAddr();
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateFftsMemAllocator(KernelContext *context) {
  auto gert_allocator = context->GetInputValue<GertAllocator *>(0UL);
  KERNEL_CHECK_NOTNULL(gert_allocator);
  auto single_stream_l2_allocator = reinterpret_cast<memory::SingleStreamL2Allocator *>(gert_allocator);
  auto allocator = reinterpret_cast<memory::CachingMemAllocator *>(single_stream_l2_allocator->GetL1Allocator());
  KERNEL_CHECK_NOTNULL(allocator);
  auto window_size = context->GetInputValue<uint32_t>(1UL);
  GE_ASSERT_TRUE(window_size > 0U, "windows_size should be greater than zero.");
  auto allocator_chain = context->GetOutput(0UL);
  KERNEL_CHECK_NOTNULL(allocator_chain);

  auto level2_allocator = allocator_chain->GetPointer<memory::FftsMemAllocator>();
  if (level2_allocator != nullptr) {
    KERNEL_TRACE("[MEM]Reuse FftsMemAllocator window_size:[%u].", window_size);
    level2_allocator->SetWidowSize(window_size);
  } else {
    auto ffts_allocator =
        new (std::nothrow) memory::FftsMemAllocator(*allocator, allocator->GetDeviceId(), window_size);
    GE_ASSERT_NOTNULL(ffts_allocator);
    KERNEL_TRACE("[MEM]Create FftsMemAllocator window_size:[%u].", window_size);
    allocator_chain->SetWithDefaultDeleter(ffts_allocator);
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateFftsMemAllocator).RunFunc(CreateFftsMemAllocator);

ge::graphStatus RecycleFftsMems(KernelContext *context) {
  auto allocator = context->GetInputValue<memory::FftsMemAllocator *>(0UL);
  KERNEL_CHECK_NOTNULL(allocator);
  allocator->Recycle();
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(RecycleFftsMems).RunFunc(RecycleFftsMems).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus AllocateFftsMem(KernelContext *context) {
  auto allocator = context->GetInputValue<memory::FftsMemAllocator *>(0UL);
  KERNEL_CHECK_NOTNULL(allocator);
  auto tensor_size = context->GetInputValue<size_t>(1UL);
  auto block_holder = context->GetOutputPointer<memory::FftsMemBlock *>(0UL);
  KERNEL_CHECK_NOTNULL(block_holder);

  auto mem_block = allocator->Malloc(tensor_size);
  KERNEL_CHECK_NOTNULL(mem_block);
  KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", tensor_size);
  KERNEL_TRACE("[MEM]AllocateFftsMem mem_info:[%s].", mem_block->DebugString().c_str());
  *block_holder = mem_block;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AllocateFftsMem).RunFunc(AllocateFftsMem).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus FreeFftsMem(KernelContext *context) {
  auto mem_block = context->GetInputValue<memory::FftsMemBlock *>(0UL);
  if (mem_block != nullptr) {
    KERNEL_TRACE("[MEM]FreeFftsMem mem_info:[%s].", mem_block->DebugString().c_str());
    mem_block->Free();
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FreeFftsMem).RunFunc(FreeFftsMem).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus CreateBatchFftsMems(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  constexpr int64_t kMaxBathWorkspaces = 8;
  auto addr_av = context->GetOutput(0UL);
  KERNEL_CHECK_NOTNULL(addr_av);
  auto addrs = ContinuousVector::Create<TensorAddress>(kMaxBathWorkspaces);
  KERNEL_CHECK_NOTNULL(addrs);
  auto deleter = [](void *mems) { delete[] static_cast<uint8_t *>(mems); };
  addr_av->Set(addrs.release(), deleter);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllocateBatchFftsMems(KernelContext *context) {
  auto allocator = context->GetInputValue<memory::FftsMemAllocator *>(0UL);
  KERNEL_CHECK_NOTNULL(allocator);
  auto sizes = context->GetInputPointer<TypedContinuousVector<size_t>>(1UL);
  KERNEL_CHECK_NOTNULL(sizes);
  auto blocks = context->GetOutputPointer<TypedContinuousVector<memory::FftsMemBlock *>>(0UL);
  KERNEL_CHECK_NOTNULL(blocks);

  if (sizes->GetSize() > blocks->GetCapacity()) {
    GELOGE(ge::GRAPH_FAILED, "[AllocateBatchFftsMems] The alloc count %zu is greater than output capacity %zu",
           sizes->GetSize(), blocks->GetCapacity());
    return ge::GRAPH_FAILED;
  }

  auto sizes_data = sizes->GetData();
  auto addrs_data = blocks->MutableData();
  KERNEL_CHECK_NOTNULL(sizes_data);
  KERNEL_CHECK_NOTNULL(addrs_data);
  GE_ASSERT_TRUE(blocks->GetCapacity() >= sizes->GetSize());
  for (size_t i = 0UL; i < sizes->GetSize(); ++i) {
    auto mem_block = reinterpret_cast<memory::FftsMemBlock *>(allocator->Malloc(sizes_data[i]));
    KERNEL_CHECK_NOTNULL(mem_block);
    KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", sizes_data[i]);
    KERNEL_TRACE("[MEM]Allocate ffts_mem, info:[%s]", mem_block->DebugString().c_str());
    addrs_data[i] = mem_block;
  }
  blocks->SetSize(sizes->GetSize());

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AllocateBatchFftsMems)
    .RunFunc(AllocateBatchFftsMems)
    .OutputsCreator(CreateBatchFftsMems)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus FreeBatchFftsMems(KernelContext *context) {
  auto blocks = context->MutableInputPointer<TypedContinuousVector<memory::FftsMemBlock *const>>(0UL);
  KERNEL_CHECK_NOTNULL(blocks);
  auto block_data = blocks->GetData();
  KERNEL_CHECK_NOTNULL(block_data);
  for (size_t i = 0UL; i < blocks->GetSize(); ++i) {
    if (block_data[i] != nullptr) {
      KERNEL_TRACE("[MEM]Free ffts_mem, info:[%s]", block_data[i]->DebugString().c_str());
      block_data[i]->Free();
    }
  }
  blocks->SetSize(0UL);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FreeBatchFftsMems).RunFunc(FreeBatchFftsMems).ConcurrentCriticalSectionKey(kKernelUseMemory);

REGISTER_KERNEL(AllocMemHbm)
    .RunFunc(AllocHbmMem)
    .OutputsCreator(AllocOutputTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(AllocMemHost)
    .RunFunc(AllocHbmMem)
    .OutputsCreator(AllocOutputTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(AllocBatchHbm)
    .RunFunc(AllocBatchHbm)
    .OutputsCreator(CreateAllocBatchHbmOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

REGISTER_KERNEL(FreeMemHbm).RunFunc(FreeHbmMem).ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(FreeMemHbmHoldAddr).RunFunc(FreeHbmMemHoldAddr).ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(FreeBatchHbm).RunFunc(FreeBatchHbm).ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(FreeBatchHbmHoldAddr).RunFunc(FreeBatchHbmHoldAddr).ConcurrentCriticalSectionKey(kKernelUseMemory);

REGISTER_KERNEL(AllocMemory)
    .RunFunc(AllocHbmMem)
    .OutputsCreator(AllocOutputTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(FreeMemory).RunFunc(FreeMemory).ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(FreeMemoryHoldAddr).RunFunc(FreeMemoryHoldAddr).ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(FreeTensorMemory).RunFunc(FreeTensorMemory).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus AccessMemCrossStream(KernelContext *context) {
  auto td = context->MutableInputPointer<GertTensorData>(0U);
  auto dst_td = context->GetOutputPointer<GertTensorData>(0U);
  if (SECUREC_UNLIKELY((td == nullptr) || (dst_td == nullptr))) {
    return ge::GRAPH_FAILED;
  }
  if (TensorPlacementUtils::IsOnHostNotFollowing(td->GetPlacement())) {
    return dst_td->ShareFrom(*td);
  } else {
    auto dst_stream_id = *(context->GetInputPointer<int64_t>(1));
    KERNEL_TRACE_WANDERING(td->GetStreamId(), dst_stream_id, td->GetAddr());
    return dst_td->WanderFrom(*td, dst_stream_id);
  }
}
ge::graphStatus AccessMemCrossStreamOutputCreator(const ge::FastNode *, KernelContext *context) {
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  auto td = new (std::nothrow) GertTensorData();
  GE_ASSERT_NOTNULL(td);
  chain->SetWithDefaultDeleter(td);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AccessMemCrossStream)
    .RunFunc(AccessMemCrossStream)
    .OutputsCreator(AccessMemCrossStreamOutputCreator);

ge::graphStatus EmptyTensorData(KernelContext *context) {
  (void) context;
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(EmptyTensorData).RunFunc(EmptyTensorData).OutputsCreator(AllocOutputTensorData);

ge::graphStatus GetUserAllocatorOrFixedBaseAllocator(KernelContext *context) {
  auto allocator_chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(allocator_chain);
  const auto placement = context->GetInputValue<TensorPlacement>(
      static_cast<size_t>(GerUserAllocatorOrFixedBaseAllocatorInputs::kPlacement));
  GE_ASSERT_TRUE((placement == kOnDeviceHbm) || (placement == kOnDeviceP2p),
                 "placement[%s] is invalid, only support %s and %s",
                 GetPlacementStr(placement), GetPlacementStr(kOnDeviceHbm), GetPlacementStr(kOnDeviceP2p));
  const rtMemType_t memory_type = placement == kOnDeviceHbm ? RT_MEMORY_HBM : RT_MEMORY_P2P_DDR;
  const auto session_id_ptr = context->GetInputPointer<uint64_t>(
      static_cast<size_t>(GerUserAllocatorOrFixedBaseAllocatorInputs::kSessionId));
  GE_CHECK_NOTNULL(session_id_ptr);

  /*
   * 用户注册进来的allocator，还有HybridModelRtV2Executor中创建的allocator，凡是通过gert::ModelExecuteArg::external_allocator
   * 传进来都会被认为是外置allocator。这里需要判断用户注册进来的allocator
   *
   * 优先使用用户注册的allocator
   */
  const auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(
      GerUserAllocatorOrFixedBaseAllocatorInputs::kStream));
  if (stream != nullptr) {
    const auto external_allocator = ge::ExternalAllocatorManager::GetExternalAllocator(stream);
    if ((placement == kOnDeviceHbm) && (external_allocator != nullptr)) {
      allocator_chain->Set(external_allocator.get(), nullptr);
      KERNEL_TRACE("[MEM]get external allocator, skip create fixed base expandable allocator, stream: %p", stream);
      return ge::GRAPH_SUCCESS;
    }
  }

  // 使用固定地址，内存大小可扩展的allocator
  int32_t device_id = 0;
  rtGetDevice(&device_id);
  ge::Allocator *allocator = ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().GetMemAllocator(
      *session_id_ptr, device_id, memory_type).get();
  GE_ASSERT_NOTNULL(allocator);
  allocator_chain->Set(allocator, nullptr);
  KERNEL_TRACE("[MEM]ger or create fixed base expandable allocator, device_id: %d, session_id: %llu, placement: %s",
               device_id, *session_id_ptr, GetPlacementStr(placement));
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(GetUserAllocatorOrFixedBaseAllocator).RunFunc(GetUserAllocatorOrFixedBaseAllocator);

ge::graphStatus AllocFixedFeatureMemory(KernelContext *context) {
  const auto tensor_size = context->GetInputValue<size_t>(static_cast<size_t>(AllocFixedFeatureMemoryInputs::kSize));
  const auto l2_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(AllocFixedFeatureMemoryInputs::kL2Allocator));
  auto gert_tensor_data = context->GetOutputPointer<GertTensorData>(0U);
  KERNEL_CHECK_NOTNULL(gert_tensor_data);

  auto allocator = context->GetInputValue<ge::Allocator *>(
      static_cast<int32_t>(AllocFixedFeatureMemoryInputs::kAllocator));
  GE_ASSERT_NOTNULL(allocator);
  auto mem_block = allocator->Malloc(tensor_size);
  KERNEL_CHECK((mem_block != nullptr) && (mem_block->GetAddr() != nullptr), "allocator malloc %zu failed.",
               tensor_size);
  auto td = TensorUtils::ToTensorData(mem_block, mem_block->GetSize(), kOnDeviceHbm);
  GE_ASSERT_SUCCESS(TensorUtils::ShareTdToGtd(td, *l2_allocator, *gert_tensor_data));
  KERNEL_TRACE("[MEM] alloc fixed_feature_memory, block: %p, addr: %p, size %zu",
               mem_block, mem_block->GetAddr(), tensor_size);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(AllocFixedFeatureMemory)
    .RunFunc(AllocFixedFeatureMemory)
    .OutputsCreator(AllocOutputTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus FreeFixedFeatureMemory(KernelContext *context) {
  auto gert_tensor_data = context->GetInputValue<GertTensorData *>(
      static_cast<int32_t>(FreeFixedFeatureMemoryInputs::kGertTensorData));
  GE_ASSERT_NOTNULL(gert_tensor_data);
  KERNEL_TRACE("[MEM] free fixed_feature_memory, addr: %p", gert_tensor_data->GetAddr());
  (void)gert_tensor_data->Free();
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FreeFixedFeatureMemory)
    .RunFunc(FreeFixedFeatureMemory)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert
