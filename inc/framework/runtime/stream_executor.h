/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_STREAM_EXECUTOR_H
#define AIR_CXX_STREAM_EXECUTOR_H
#include <map>
#include <memory>
#include <mutex>
#include "acl/acl_rt.h"
#include "common/checker.h"
#include "model_v2_executor.h"

namespace gert {
// do not expose the Builder class definition to external api
class ModelV2ExecutorBuilder;
class VISIBILITY_EXPORT StreamExecutor {
 public:
  explicit StreamExecutor(ModelV2ExecutorBuilder *builder);
  StreamExecutor(const StreamExecutor &) = delete;
  StreamExecutor &operator=(const StreamExecutor &) = delete;
  StreamExecutor(StreamExecutor &&) = delete;
  StreamExecutor &operator=(StreamExecutor &&) = delete;
  ~StreamExecutor();
  ModelV2Executor *GetOrCreateLoaded(aclrtStream stream, const ModelExecuteArg &arg) {
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    const auto &iter = streams_to_executor_.find(stream);
    if (iter != streams_to_executor_.cend()) {
      return iter->second.get();
    }
    return CreateAndLoad(stream, arg);
  }
  ge::graphStatus Erase(aclrtStream stream);

 private:
  ModelV2Executor *CreateAndLoad(aclrtStream stream, const ModelExecuteArg &arg);

 private:
  std::recursive_mutex mutex_;
  ModelV2ExecutorBuilder *builder_;
  std::map<aclrtStream, std::unique_ptr<ModelV2Executor>> streams_to_executor_;
};
}  // namespace gert
#endif  // AIR_CXX_STREAM_EXECUTOR_H
