/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile_context.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/utils/tensor_adapter.h"

#include <op_type_utils.h>

namespace ge {
// todo refactor func name to AddAndCompile
Status CompileContext::Compile(uint32_t graph_id, const ComputeGraphPtr &graph, const std::vector<gert::Tensor> &inputs,
                               const std::map<std::string, std::string> &options, uint64_t session_id) {
  Graph graph_to_add = GraphUtilsEx::CreateGraphFromComputeGraph(graph);  // todo check if need more info in graph
  GE_ASSERT_SUCCESS(graph_manager_.AddGraph(graph_id, graph_to_add, options, domi::GetContext()));
  GELOGI("[Session: ][AddGraph] success to add slice graph id: %ld, session_id: %llu", graph_id, session_id);
  std::vector<Tensor> inputs_to_ge;
  inputs_to_ge.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    const auto &gert_tensor = inputs[i];
    Tensor tensor;
    if (gert::TensorPlacementUtils::IsOnHost(gert_tensor.GetPlacement())) {
      GE_ASSERT_SUCCESS(TensorTransUtils::GertTensor2Tensor(gert_tensor, tensor));
      GELOGD("Trans gert_tensor[%u] to tensor with host data success.", i);
    } else {
      auto ge_tensor = TensorTransUtils::TransRtTensorToGeTensor(gert_tensor);
      ge_tensor.ClearData();
      ge_tensor.MutableTensorDesc().SetPlacement(kPlacementEnd);
      tensor = TensorAdapter::AsTensor(ge_tensor);
    }
    inputs_to_ge.emplace_back(tensor);
  }
  GE_ASSERT_TRUE(inputs_to_ge.size() == inputs.size());
  GE_ASSERT_SUCCESS(graph_manager_.CompileGraph(graph_id, session_id, inputs_to_ge), "graph id: %ld, session_id: %llu",
    graph_id, session_id);
  return SUCCESS;
}

Status CompileContext::Compile(uint32_t graph_id, const ComputeGraphPtr &graph, const std::vector<ge::Tensor> &inputs,
    uint64_t session_id) {
  Graph graph_to_add = GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  GE_ASSERT_SUCCESS(graph_manager_.AddGraph(graph_id, graph_to_add, {}, domi::GetContext()));
  GELOGI("[Session: ][AddGraph] success to add slice graph id: %ld, session_id: %llu", graph_id, session_id);
  GE_ASSERT_SUCCESS(graph_manager_.CompileGraph(graph_id, session_id, inputs), "graph id: %ld, session_id: %llu",
    graph_id, session_id);
  return SUCCESS;
}

Status CompileContext::Load(uint32_t graph_id, const aclrtStream stream) const {
  GE_ASSERT_SUCCESS(graph_manager_.LoadGraph(graph_id, {}, stream));
  GELOGI("[Session: ][AddGraph] success to load slice_graph_id: %ld", graph_id);
  return SUCCESS;
}

Status CompileContext::Load(uint32_t graph_id, const std::map<AscendString, AscendString> &options,
                            const aclrtStream stream) {
  GE_ASSERT_SUCCESS(graph_manager_.LoadGraph(graph_id, options, stream));
  GELOGI("[Session: ][LoadGraph] success to load slice_graph_id: %ld", graph_id);
  return SUCCESS;
}

Status CompileContext::Fork(uint32_t origin_graph_id, uint32_t forked_graph_id) {
  GE_ASSERT_SUCCESS(graph_manager_.ForkGraph(origin_graph_id, forked_graph_id));
  GELOGI("[Session: ][ForkGraph] success to fork graph: %u from graph:%u", forked_graph_id, origin_graph_id);
  return SUCCESS;
}
bool CompileContext::IsGraphNeedRebuild(uint32_t graph_id) {
  return graph_manager_.IsGraphNeedRebuild(graph_id);
}

Status CompileContext::GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary) const {
  GE_ASSERT_SUCCESS(graph_manager_.GetCompiledGraphSummary(graph_id, summary));
  GELOGI("[Session: ][GetCompiledGraphSummary] success to get compiled graph: %u", graph_id);
  return SUCCESS;
}
} // ge