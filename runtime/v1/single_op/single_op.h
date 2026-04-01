/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_SINGLE_OP_H_
#define GE_SINGLE_OP_SINGLE_OP_H_

#include <cstdint>
#include <memory>
#include <queue>
#include <mutex>

#include "framework/common/ge_types.h"
#include "graph/utils/object_pool.h"
#include "graph/ge_tensor.h"
#include "runtime/base.h"
#include "acl/acl_rt.h"

namespace ge {
constexpr uint64_t kFuzzDeviceBufferSize = 1U * 1024U * 1024U;

class SingleOpImpl;
class DynamicSingleOpImpl;
class StreamResource;
class SingleOp {
 public:
  SingleOp(StreamResource *const stream_resource, std::mutex *const stream_mutex, aclrtStream const stream);
  ~SingleOp();

  Status ExecuteAsync(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs);

  int64_t GetProfilingNodeIndex() const noexcept;

 private:
  friend class StreamResource;
  SingleOpImpl *impl_ = nullptr;
};

class DynamicSingleOp {
 public:
  DynamicSingleOp(ObjectPool<GeTensor> *const tensor_pool, const uintptr_t resource_id,
                  std::mutex *const stream_mutex, aclrtStream const stream);
  ~DynamicSingleOp();

  Status ExecuteAsync(const std::vector<GeTensorDesc> &input_desc,
                      const std::vector<DataBuffer> &input_buffers,
                      std::vector<GeTensorDesc> &output_desc,
                      std::vector<DataBuffer> &output_buffers);

  int64_t GetProfilingNodeIndex() const noexcept;

 private:
  friend class StreamResource;
  DynamicSingleOpImpl *impl_ = nullptr;
};
}  // namespace ge
#endif  // GE_SINGLE_OP_SINGLE_OP_H_
