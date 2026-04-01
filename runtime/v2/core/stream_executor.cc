/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/stream_executor.h"
#include "builder/model_v2_executor_builder.h"
namespace gert {
StreamExecutor::StreamExecutor(ModelV2ExecutorBuilder *builder) : builder_(builder) {}
StreamExecutor::~StreamExecutor() {
  delete builder_;
}
ge::graphStatus StreamExecutor::Erase(aclrtStream stream) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto iter = streams_to_executor_.find(stream);
  if (iter != streams_to_executor_.end()) {
    GE_ASSERT_NOTNULL(iter->second);
    GELOGD("Unload executor on stream %p", stream);
    GE_ASSERT_SUCCESS(iter->second->UnLoad());
    streams_to_executor_.erase(iter);
  }
  return ge::GRAPH_SUCCESS;
}
ModelV2Executor *StreamExecutor::CreateAndLoad(aclrtStream stream, const ModelExecuteArg &arg) {
  GE_ASSERT_NOTNULL(builder_);
  GELOGD("Create a new executor for stream %p", stream);
  auto executor = builder_->Build();
  GE_ASSERT_NOTNULL(executor);
  GE_ASSERT_SUCCESS(executor->Load(arg));

  auto result = streams_to_executor_.emplace(stream, std::move(executor));
  GE_ASSERT_TRUE(result.second);
  return result.first->second.get();
}
}  // namespace gert
