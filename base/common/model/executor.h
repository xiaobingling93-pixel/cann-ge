/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_EXECUTOR_H
#define GE_COMMON_EXECUTOR_H

#include "ge/ge_api_types.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager_utils.h"
#include "exe_graph/runtime/tensor.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"
#include "acl/acl_rt.h"

namespace ge {
struct RunArgs {
  GraphNodePtr graph_node;
  GraphId graph_id;
  uint64_t session_id;
  error_message::ErrorManagerContext error_context;
  std::vector<gert::Tensor> input_tensor;
  GEThreadLocalContext context;
  RunAsyncCallbackV2 callback;
};

class Executor {
 public:
  Executor() = default;
  virtual ~Executor() = default;
  Executor &operator=(const Executor &)& = delete;
  Executor(const Executor &) = delete;

  /**
   * @ingroup ge
   * @brief Load mode from graph.
   * @param [in] GeRootModel: root model of graph compiled.
   * @param [in] GraphNode: node of graph.
   * @return Status result of function
   */
  virtual Status LoadGraph(const GeRootModelPtr &root_model, const GraphNodePtr &graph_node,
                           const aclrtStream stream = nullptr) = 0;

  /**
   * @ingroup ge
   * @brief Unload mode.
   * @param [in] root_model: root model of graph compiled.
   * @param [in] graph_id: graph identifier.
   * @return Status result of function
   */
  virtual Status UnloadGraph(const GeRootModelPtr &root_model, const uint32_t graph_id) = 0;

  /**
   * @ingroup ge
   * @brief Push model execution params to queue.
   * @param [in] RunArgs of for model execution.
   * @return Status result of function
   */
  virtual Status PushRunArgs(const std::shared_ptr<RunArgs> &args) = 0;

  /**
   * @ingroup ge
   * @brief Run graph for synchronize model.
   * @param [in] graph_node: node of graph.
   * @param [in] graph_id: graph identifier.
   * @param [in] inputs: input data for the graph running.
   * @param [out] outputs: output data of the graph running
   * @return Status result of function
   */
  virtual Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                          const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) = 0;

  /**
   * @ingroup ge
   * @brief Run graph for NN synchronize model.
   * @param [in] graph_node: node of graph.
   * @param [in] graph_id: graph identifier.
   * @param [in] stream: Stream for model running.
   * @param [in] inputs: input data for the graph running.
   * @param [out] outputs: output data of the graph running
   * @return Status result of function
   */
  virtual Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, const aclrtStream stream,
                                    const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) = 0;
  
  virtual Status ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                                         aclrtStream const stream, const std::vector<gert::Tensor> &inputs,
                                         std::vector<gert::Tensor> &outputs) = 0;

  /**
   * @ingroup ge
   * @brief Update graph feature memory base after load model
   * @param [in] graph_node: node of graph.
   * @param [in] mem_base: graph feature memory base(without input and output).
   * @param [in] size: memory size.
   * @return Status result of function
   */
  virtual Status UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base, const size_t size) {
    (void)graph_node;
    (void)mem_base;
    (void)size;
    return SUCCESS;
  }

  virtual Status PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                            const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) = 0;
};
}  // namespace ge
#endif // GE_COMMON_EXECUTOR_H
