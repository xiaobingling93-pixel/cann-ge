/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_SINK_ONLY_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_SINK_ONLY_ALLOCATOR_H_
#include <vector>
#include <mutex>

#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_block.h"
#include "memory_block_manager.h"

namespace gert::memory {
class SinkOnlyGertMemBlock : public GertMemBlock {
 public:
  explicit SinkOnlyGertMemBlock(void *addr) : addr_(addr) {}

  ~SinkOnlyGertMemBlock() override {
    addr_ = nullptr;
  }

  void Free(int64_t stream_id) override {
    (void)stream_id;
  }
  void *GetAddr() override {
    return addr_;
  }

 private:
  void *addr_;
};

/**
 * 该类为静态内存分配包装器，只负责内存分配，分配的内存释放又davinci_model中的FreeDynamicWorkspaceMemory统一释放
 */
class SinkOnlyAllocator : public GertAllocator {
 public:
  SinkOnlyAllocator();
  ~SinkOnlyAllocator() noexcept override;
  GertTensorData MallocTensorData(size_t size) override;
  GertMemBlock *Malloc(size_t size) override;
  void Free(GertMemBlock *block) override;
  ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock *block) override;
  int64_t GetStreamNum() override;
  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override;
  TensorData MallocTensorDataFromL1(size_t size) override;
  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) override;

  /**
   * 获取内存块管理器的智能指针
   * @return std::shared_ptr<ge::MemoryBlockManager>
   */
  std::shared_ptr<ge::MemoryBlockManager> GetAllocator() const {
    return allocator_;
  }

  /**
   * 设置内存块管理器
   * @param allocator 待设置的MemoryBlockManager智能指针
   */
  void SetAllocator(const std::shared_ptr<ge::MemoryBlockManager> &allocator) {
    allocator_ = allocator;
  }

 private:
  std::shared_ptr<ge::MemoryBlockManager> allocator_{};
  std::vector<GertMemBlock *> allocated_blocks_;
  std::mutex blocks_mutex_;
};
}  // namespace gert::memory
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_SINK_ONLY_ALLOCATOR_H_
