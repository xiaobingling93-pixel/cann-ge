/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/util/rt_context_util.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/ge_context.h"

namespace ge {
namespace {
  const int64_t kDefaultGraphId = -1;
}

RtContextUtil &RtContextUtil::GetInstance() {
  static RtContextUtil instance;
  return instance;
}

Status RtContextUtil::SetRtContext(const uint64_t session_id, const uint32_t graph_id, const int32_t device_id,
                                   const rtCtxMode_t mode, rtContext_t rt_context) const {
  GELOGI("set rt_context, session id: %lu, graph id: %u, mode %d, device id:%u.", session_id,
         graph_id, static_cast<int32_t>(mode), ge::GetContext().DeviceId());

  GE_CHK_STATUS_RET(rtCtxCreate(&rt_context, mode, device_id));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddRtContext(session_id, graph_id, rt_context);

  return SUCCESS;
}

void RtContextUtil::AddRtContext(uint64_t session_id, rtContext_t context) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  rt_contexts_[session_id][kDefaultGraphId].emplace_back(context);
}

void RtContextUtil::AddRtContext(uint64_t session_id, uint32_t graph_id, rtContext_t context) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  rt_contexts_[session_id][static_cast<int64_t>(graph_id)].emplace_back(context);
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  auto &session_ctxs = rt_contexts_[session_id];
  for (auto &graph_ctx_pair : session_ctxs) {
    DestroyRtContexts(session_id, graph_ctx_pair.first, graph_ctx_pair.second);
  }

  const auto &iter = rt_contexts_.find(session_id);
  if (iter != rt_contexts_.end()) {
    (void)rt_contexts_.erase(iter);
  }
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id, uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  auto &session_ctxs = rt_contexts_[session_id];
  auto &graph_ctxs = session_ctxs[graph_id];
  DestroyRtContexts(session_id, static_cast<int64_t>(graph_id), graph_ctxs);

  const auto &iter = session_ctxs.find(graph_id);
  if (iter != session_ctxs.end()) {
    (void)session_ctxs.erase(iter);
  }
}

void RtContextUtil::DestroyAllRtContexts() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  for (auto &session_ctx_pair : rt_contexts_) {
    for (auto &graph_ctx_pair : session_ctx_pair.second) {
      DestroyRtContexts(session_ctx_pair.first, graph_ctx_pair.first, graph_ctx_pair.second);
    }
  }
  rt_contexts_.clear();
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id, int64_t graph_id,
                                      std::vector<rtContext_t> &contexts) const {
  GELOGI("Destroy %zu rts contexts for graph %ld of session %lu.", contexts.size(), graph_id, session_id);
  for (auto &rtContext : contexts) {
    (void)rtCtxDestroy(rtContext);
  }
  contexts.clear();
}
}  // namespace ge
