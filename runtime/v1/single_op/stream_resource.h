/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_STREAM_RESOURCE_H_
#define GE_SINGLE_OP_STREAM_RESOURCE_H_

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "runtime/stream.h"
#include "single_op/single_op.h"
#include "hybrid/executor/node_done_manager.h"
#include "hybrid/executor/callback_manager.h"
#include "graph/utils/object_pool.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "common/thread_pool/thread_pool.h"
#include "framework/runtime/mem_allocator.h"
#include "common/checker.h"
#include "ge/ge_allocator.h"

namespace ge {
constexpr int64_t kOverflowSize = 512;

class InternalAllocator : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override;
  void Free(MemBlock *block) override {
    const auto rt_ret = rtFree(block->GetAddr());
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Free][Rt] failed."));
  };
  ~InternalAllocator() override;

 private:
  std::vector<std::unique_ptr<MemBlock>> memory_list_{};
  size_t max_memory_size_{0UL};
  rtStream_t stream_{nullptr};
};

class StreamResource {
 public:
  explicit StreamResource(const uintptr_t resource_id);
  ~StreamResource() noexcept;

  StreamResource(const StreamResource &) = delete;
  StreamResource(StreamResource &&) = delete;
  StreamResource &operator=(const StreamResource &) = delete;
  StreamResource &operator=(StreamResource &&) = delete;
  rtStream_t GetStream() const;
  void SetStream(const rtStream_t stream);

  Status Init();
  SingleOp *GetOperator(const uint64_t key);
  Status DeleteOperator(const uint64_t key);
  Status DeleteDynamicOperator(const uint64_t key);
  DynamicSingleOp *GetDynamicOperator(const uint64_t key);

  Status BuildOperator(const ModelData &model_data, SingleOp **const single_op, const uint64_t model_id);
  Status BuildDynamicOperator(const ModelData &model_data, DynamicSingleOp **const single_op, const uint64_t model_id);

  Status InitOverflowMemory();
  uint8_t *MallocMemory(const std::string &purpose, const size_t size, const bool holding_lock = true);
  uint8_t *MallocMemory(const std::string &purpose, const size_t size, const bool holding_lock, ge::MemBlock *&block);
  Status MallocExMem(const uint32_t device_id, RuntimeParam &runtime_param);
  void FreeExMem();
  uint8_t *MallocWeight(const std::string &purpose, const size_t size);
  void *GetDeviceBufferAddr() const {
    return static_cast<void *>(device_buffer_);
  }

  Status GetThreadPool(ThreadPool **const thread_pool);
  Status GetCallbackManager(hybrid::CallbackManager **const callback_manager);
  
  void *GetOverflowAddr() const {
    return overflow_addr_;
  }

  int64_t GetOverflowSize() const {
    return kOverflowSize;
  }

  void SetAllocator(ge::Allocator *const allocator) {
    allocator_= allocator;
  }

  InternalAllocator *GetInternalAllocator() {
    return &internal_allocator_;
  }

 private:
  uint8_t *DoMallocMemory(const std::string &purpose, const size_t size, ge::MemBlock *&block) const;

  uintptr_t resource_id_;
  std::vector<uint8_t *> weight_list_;
  std::map<uint32_t, std::vector<std::map<uint64_t, MemInfo>>> device_2_meminfos_;
  std::unordered_map<uint64_t, std::unique_ptr<SingleOp>> op_map_;
  std::unordered_map<uint64_t, std::unique_ptr<DynamicSingleOp>> dynamic_op_map_;
  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<hybrid::CallbackManager> callback_manager_;
  ObjectPool<GeTensor> tensor_pool_;
  rtStream_t stream_ = nullptr;
  std::mutex mu_;
  std::mutex stream_mu_;
  void *device_buffer_ = nullptr;
  InternalAllocator internal_allocator_{};
  ge::Allocator *allocator_{nullptr};
  void *overflow_addr_ = nullptr;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_STREAM_RESOURCE_H_
