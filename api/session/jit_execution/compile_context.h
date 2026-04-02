/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILE_CONTEXT_H
#define COMPILE_CONTEXT_H
#include <cstdint>
#include "ge/ge_api_types.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager.h"
#include "acl/acl_rt.h"

namespace ge {
class CompileContext {
public:
  explicit CompileContext(GraphManager &graph_manager) : graph_manager_(graph_manager) {}
  uint32_t GenNewGraphId() {
    return inner_ge_graph_id_generator_++;
  }
  Status Compile(uint32_t graph_id, const ComputeGraphPtr &graph, const std::vector<gert::Tensor> &inputs,
                 const std::map<std::string, std::string> &options, uint64_t session_id);
  Status Compile(uint32_t graph_id, const ComputeGraphPtr &graph, const std::vector<ge::Tensor> &inputs,
    uint64_t session_id);
  Status Fork(uint32_t origin_graph_id, uint32_t forked_graph_id);
  Status Load(uint32_t graph_id,  const aclrtStream stream) const;
  Status Load(uint32_t graph_id, const std::map<AscendString, AscendString> &options,
              const aclrtStream stream);
  bool IsGraphNeedRebuild(uint32_t graph_id);
  Status GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary) const;

 private:
  GraphManager &graph_manager_;
  uint32_t inner_ge_graph_id_generator_{0};
};

} // ge

#endif // COMPILE_CONTEXT_H
