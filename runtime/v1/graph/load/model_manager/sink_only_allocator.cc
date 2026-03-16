/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "sink_only_allocator.h"
#include <checker.h>
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert::memory {
const std::string kMallocPurpose = "davinci_model_load";
SinkOnlyAllocator::SinkOnlyAllocator() : GertAllocator(-1, kTensorPlacementEnd) {}
SinkOnlyAllocator::~SinkOnlyAllocator() noexcept {
  std::lock_guard lock(blocks_mutex_);
  for (auto block : allocated_blocks_) {
    delete block;
  }
  allocated_blocks_.clear();
}

GertTensorData SinkOnlyAllocator::MallocTensorData(size_t size) {
  (void)size;
  GELOGE(ge::FAILED, "SinkOnlyAllocator::MallocTensorData is not supported temporarily");
  return {};
}

GertMemBlock *SinkOnlyAllocator::Malloc(size_t size) {
  GE_ASSERT_NOTNULL(allocator_);
  if (size == 0) {
    GELOGW("SinkOnlyAllocator Malloc size is 0, return nullptr");
    return nullptr;
  }
  void *ptr = allocator_->Malloc(kMallocPurpose, size);
  if (ptr == nullptr) {
    GELOGE(ge::FAILED, "SinkOnlyAllocator MemoryBlockManager Malloc failed, size: %zu", size);
    return nullptr;
  }
  auto sink_only_gert_mem_block = new SinkOnlyGertMemBlock(ptr);
  std::lock_guard lock(blocks_mutex_);
  allocated_blocks_.push_back(sink_only_gert_mem_block);
  return sink_only_gert_mem_block;
}

void SinkOnlyAllocator::Free(GertMemBlock *block) {
  (void)block;
  GELOGE(ge::FAILED, "SinkOnlyAllocator::Free is not supported temporarily");
}

ge::graphStatus SinkOnlyAllocator::FreeAt(int64_t stream_id, GertMemBlock *block) {
  (void)stream_id;
  (void)block;
  GELOGE(ge::FAILED, "SinkOnlyAllocator::FreeAt is not supported temporarily");
  return ge::FAILED;
}

int64_t SinkOnlyAllocator::GetStreamNum() {
  GELOGE(ge::FAILED, "SinkOnlyAllocator::GetStreamNum is not supported temporarily");
  return 0;
}

ge::graphStatus SinkOnlyAllocator::SetL1Allocator(ge::Allocator *allocator) {
  (void)allocator;
  GELOGE(ge::FAILED, "SinkOnlyAllocator::SetL1Allocator is not supported temporarily");
  return ge::FAILED;
}

TensorData SinkOnlyAllocator::MallocTensorDataFromL1(size_t size) {
  (void)size;
  GELOGE(ge::FAILED, "SinkOnlyAllocator::MallocTensorDataFromL1 is not supported temporarily");
  return TensorData();
}

ge::graphStatus SinkOnlyAllocator::ShareFromTensorData(const TensorData &td, GertTensorData &gtd) {
  (void)td;
  (void)gtd;
  GELOGE(ge::GRAPH_FAILED, "SinkOnlyAllocator::ShareFromTensorData is not supported temporarily");
  return ge::GRAPH_FAILED;
}
}  // namespace gert::memory