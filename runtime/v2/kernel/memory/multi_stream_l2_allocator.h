/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MULTI_STREAM_L2_ALLOCATOR_H
#define AIR_CXX_MULTI_STREAM_L2_ALLOCATOR_H

#include <list>
#include <unordered_set>
#include "borrow_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "ge/ge_allocator.h"
#include "l2_mem_pool.h"
#include "multi_stream_mem_block.h"
#include "multi_stream_mem_block_pool.h"
#include "runtime/mem_allocator.h"
#include "version_blocks.h"
#include "ti_block_allocator.h"

namespace gert {
namespace memory {
using L2MemPoolPtr = std::unique_ptr<L2MemPool>;
class MultiStreamL2Allocator : public GertAllocator {
 public:
  MultiStreamL2Allocator(int64_t stream_id, TensorPlacement placement,
                         TypedContinuousVector<memory::MultiStreamL2Allocator *> *stream_ids_to_allocator,
                         TypedContinuousVector<L2MemPool *> *all_l2_mem_pool);
  explicit MultiStreamL2Allocator(
      ge::Allocator *allocator, TensorPlacement placement = kOnDeviceHbm, int64_t stream_id = 0,
      aclrtStream stream = nullptr,
      TypedContinuousVector<memory::MultiStreamL2Allocator *> *stream_ids_to_allocator = nullptr,
      TypedContinuousVector<L2MemPool *> *all_l2_mem_pool = nullptr);
  ~MultiStreamL2Allocator() override;
  GertMemBlock *Malloc(size_t size) override;
  void Free(GertMemBlock *block) override;
  ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock *block) override {
    if (stream_id == GetStreamId()) {
      Free(block);
    } else {
      stream_ids_to_allocator_->MutableData()[stream_id]->Free(block);
    }
    return ge::GRAPH_SUCCESS;
  }

  GertTensorData MallocTensorData(size_t size) override;
  TensorData MallocTensorDataFromL1(size_t size) override;

  ge::graphStatus BirthRecycle(MultiStreamMemBlock *block);
  std::list<MultiStreamMemBlock *> GetAndClearBorrowBlocks(int64_t dst_stream_id);
  void SetRtsStream(aclrtStream stream) {
    stream_ = stream;
    own_allocator_->SetStream(stream);
  }
  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override;
  int64_t GetStreamNum() override;

  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) override;

  /**
   * 读请接口，对于同一个dst stream和block，本接口仅会返回一次
   * @param dst_stream_id
   * @return
   */
  VersionBlocks<&BaseVersionBlocks::FindNext> GetClearLocalRecycleBlocks(int64_t dst_stream_id) {
    return VersionBlocks<&BaseVersionBlocks::FindNext>{GetStreamId(), BaseVersionBlocks::ToBit(dst_stream_id),
                                                       static_cast<int64_t>(stream_ids_to_allocator_->GetSize()),
                                                       local_recycle_blocks_};
  }

  VersionBlocks<&BaseVersionBlocks::FindNextForAll> GetClearLocalRecycleBlocks() {
    return VersionBlocks<&BaseVersionBlocks::FindNextForAll>{
        GetStreamId(),
        BaseVersionBlocks::ToAllBit(static_cast<int64_t>(stream_ids_to_allocator_->GetSize()), GetStreamId()),
        static_cast<int64_t>(stream_ids_to_allocator_->GetSize()), local_recycle_blocks_};
  }

  L2MemPool *GetL2MemPool() const {
    return own_allocator_.get();
  }
  ge::graphStatus MoveL2ToL1(GertMemBlock *block) override;
  ge::graphStatus RecycleFreeMem();

 private:
  void BorrowRecycle(MultiStreamMemBlock *block);
  void LocalRecycle(MultiStreamMemBlock *block);

 private:
  ge::Allocator *l1_allocator_;
  L2MemPoolPtr own_allocator_;
  std::list<StreamedVersionBlock> local_recycle_blocks_;
  TypedContinuousVector<memory::MultiStreamL2Allocator *> *stream_ids_to_allocator_;
  MultiStreamMemBlockPool ms_block_pool_;
  BorrowAllocator borrow_allocator_;
  TiGtdAllocator ti_allocator_;
  aclrtStream stream_;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_MULTI_STREAM_L2_ALLOCATOR_H
