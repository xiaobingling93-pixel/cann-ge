/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ast_context.h"
#include "framework/common/debug/log.h"

#include <checker.h>

namespace {
constexpr size_t kBlockSize = 16 * 1024;
constexpr int32_t kMemAlignSize = 8;
}

namespace ge {
AstNodePool::~AstNodePool() {
  GELOGI("Release AstNodePool, total memory size is %zu bytes", GetMemoryUsage());
  for (const auto &block : blocks_) {
    delete[] block.data;
  }
  blocks_.clear();
}

uint8_t *AstNodePool::Allocate(const size_t mem_size) {
  GE_ASSERT_TRUE(mem_size > 0, "The allocated memory size must be greater than zero.");
  const auto aligned_mem_size = AlignTo(mem_size, kMemAlignSize);
  /* Check if the current block has enough space.
   * AstNode is generally small, so the waste from tail fragmentation in blocks is minimal.
   * In the future, we can add a fallback allocation function for large blocks separately.
   */
  if (blocks_.empty() || aligned_mem_size > blocks_.back().length - blocks_.back().offset) {
    GE_ASSERT_TRUE(CreateNewBlock(aligned_mem_size) > 0);
  }

  Block &current_block = blocks_.back();
  uint8_t *allocated_memory = current_block.data + current_block.offset;
  current_block.offset += aligned_mem_size;
  return allocated_memory;
}

size_t AstNodePool::CreateNewBlock(const size_t min_size) {
  const size_t block_size = std::max(kBlockSize, min_size);
  // memory released upon destruction
  const auto new_block_data = new (std::nothrow) uint8_t[block_size];
  if (new_block_data == nullptr) {
    GELOGE(ge::FAILED, "Failed to create new block, block_size = %zu", block_size);
    return 0;
  }
  blocks_.push_back({new_block_data, block_size, 0});
  total_mem_usage_ += block_size;
  return block_size;
}
}  // namespace ge