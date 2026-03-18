/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_SESSION_GE_SESSION_IMPL_H_
#define GE_SESSION_GE_SESSION_IMPL_H_

#include "ge/ge_api_v2.h"
#include <map>
#include <string>
#include <vector>
#include "graph/manager/graph_manager.h"
#include "ge/ge_allocator.h"
#include "jit_execution/user_graphs_manager.h"
#include "session/session_manager.h"

namespace ge {

class GeSession::Impl {
  public:
    Impl(const std::map<std::string, std::string> &options);

    ~Impl();

    void SetSessionId(uint64_t session_id);

    uint64_t GetSessionId() const;

    std::shared_ptr<InnerSession> GetInnerSession();

    Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

    Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

    Status RemoveGraph(uint32_t graph_id);

    Status CompileGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);

    Status LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options, void *stream);

    Status RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

    Status RunGraphAsync(uint32_t graph_id, std::vector<gert::Tensor> &&inputs,
        std::function<void(Status status, std::vector<gert::Tensor> &outputs)> callback);

    Status RunGraphWithStreamAsync(uint32_t graph_id, const rtStream_t stream,
                                   const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

    Status RegisterCallBackFunc(
      const std::string &key,
      const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback);

    bool IsGraphNeedRebuild(uint32_t graph_id);

    Status AddDumpProperties(const DumpProperties &dump_properties) const;

    Status RemoveDumpProperties() const;

    Status GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary);

    Status SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

    Status UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

    Status SetGraphFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory, size_t size);

    Status UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

    Status RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const;

    Status UnregisterExternalAllocator(const void * const stream) const;

    Status GetRunGraphMode(uint32_t graph_id, RunGraphMode &mode) const;

    Status SetRunGraphMode(uint32_t graph_id, const RunGraphMode &mode);

    Status GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer);

    bool GetBuildFlag(uint32_t graph_id) const;

    bool GetLoadFlag(uint32_t graph_id) const;

    void UpdateGlobalSessionContext() const;

  private:
    uint64_t session_id_{0};
    SessionPtr inner_session_;
};
}  // namespace ge

#endif  // GE_SESSION_GE_SESSION_IMPL_H_