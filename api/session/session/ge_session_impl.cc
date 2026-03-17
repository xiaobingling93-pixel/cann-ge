/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <map>
#include <memory>
#include <vector>

#include "analyzer/analyzer.h"
#include "common/checker.h"
#include "common/dump/dump_properties.h"
#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "api/aclgrph/option_utils.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_init.h"
#include "common/model/external_allocator_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "common/memory/tensor_trans_utils.h"
#include "generator/ge_generator.h"
#include "session/ge_session_impl.h"

#include "graph/manager/session_id_manager.h"
#include "session/session_manager.h"
#include "session/ge_session_registry.h"
#include <utility>

namespace ge {
GeSession::Impl::Impl(const std::map<std::string, std::string> &options) {
  const auto next_session_id = SessionIdManager::GetNextSessionId();
  SessionPtr sessionPtr = MakeShared<InnerSession>(next_session_id, options);
  if (sessionPtr == nullptr) {
    GELOGE(GE_CLI_INIT_FAILED, "[Init][Create]GeSession failed");
    return;
  }
  Status ret = sessionPtr->Initialize();
  if (ret != SUCCESS) {
    GELOGE(ret, "Construct session failed, error code:%u.", ret);
    return;
  }
  session_id_ = next_session_id;
  inner_session_ = sessionPtr;
  // 注册到全局 Registry，用于 GEFinalizeV2 时清理
  auto finalize_func = [this]() {
    if (inner_session_ != nullptr) {
      (void)inner_session_->Finalize();
      inner_session_.reset();
      session_id_ = 0;
    }
  };
  GeSessionRegistry::Instance().Register(this, finalize_func);
}

GeSession::Impl::~Impl() {
  // 先从 Registry 注销，避免 GEFinalizeV2 重复清理
  GeSessionRegistry::Instance().Unregister(this);
  if (inner_session_ == nullptr) {
    return;
  }
  Status ret = inner_session_->Finalize();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Finalize] session failed, error code:%u.", ret);
    return;
  }
  session_id_ = 0;
}

void GeSession::Impl::SetSessionId(uint64_t session_id) {
  session_id_ = session_id;
}

uint64_t GeSession::Impl::GetSessionId() const {
  return session_id_;
}

std::shared_ptr<InnerSession> GeSession::Impl::GetInnerSession() {
  return inner_session_;
}

Status GeSession::Impl::AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->AddGraph(graph_id, graph, options);
}

Status GeSession::Impl::AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->AddGraphWithCopy(graph_id, graph, options);
}

Status GeSession::Impl::RemoveGraph(uint32_t graph_id) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->RemoveGraph(graph_id);
}

Status GeSession::Impl::CompileGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->CompileGraph(graph_id, inputs);
}

Status GeSession::Impl::LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options, void *stream) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->LoadGraph(graph_id, options, stream);
}

Status GeSession::Impl::RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->RunGraph(graph_id, inputs, outputs);
}

Status GeSession::Impl::RunGraphAsync(uint32_t graph_id, std::vector<gert::Tensor> &&inputs,
    std::function<void(Status status, std::vector<gert::Tensor> &outputs)> callback) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->RunGraphAsync(graph_id, std::move(inputs), callback);
}

Status GeSession::Impl::RunGraphWithStreamAsync(uint32_t graph_id, const rtStream_t stream,
                                               const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs);
}

Status GeSession::Impl::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->RegisterCallBackFunc(key, callback);
}

bool GeSession::Impl::IsGraphNeedRebuild(uint32_t graph_id) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->IsGraphNeedRebuild(graph_id);
}

Status GeSession::Impl::AddDumpProperties(const DumpProperties &dump_properties) const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->AddDumpProperties(dump_properties);
}

Status GeSession::Impl::RemoveDumpProperties() const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->RemoveDumpProperties();
}

Status GeSession::Impl::GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->GetCompiledGraphSummary(graph_id, summary);
}

Status GeSession::Impl::SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->SetGraphConstMemoryBase(graph_id, memory, size);
}

Status GeSession::Impl::UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->UpdateGraphFeatureMemoryBase(graph_id, memory, size);
}

Status GeSession::Impl::SetGraphFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory, size_t size) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->SetGraphFixedFeatureMemoryBase(graph_id, type, memory, size);
}

Status GeSession::Impl::UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->UpdateGraphRefreshableFeatureMemoryBase(graph_id, memory, size);
}

Status GeSession::Impl::RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->RegisterExternalAllocator(stream, allocator);
}

Status GeSession::Impl::UnregisterExternalAllocator(const void * const stream) const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->UnregisterExternalAllocator(stream);
}

Status GeSession::Impl::GetRunGraphMode(uint32_t graph_id, RunGraphMode &mode) const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->GetRunGraphMode(graph_id, mode);
}

Status GeSession::Impl::SetRunGraphMode(uint32_t graph_id, const RunGraphMode &mode) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->SetRunGraphMode(graph_id, mode);
}

Status GeSession::Impl::GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer) {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->GetCompiledModel(graph_id, model_buffer);
}

bool GeSession::Impl::GetBuildFlag(uint32_t graph_id) const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->GetBuildFlag(graph_id);
}

bool GeSession::Impl::GetLoadFlag(uint32_t graph_id) const {
  GE_CHK_BOOL_RET_STATUS(inner_session_ != nullptr, FAILED, "inner_session is null (null inner_session pointer)");
  return inner_session_->GetLoadFlag(graph_id);
}
}  // namespace ge