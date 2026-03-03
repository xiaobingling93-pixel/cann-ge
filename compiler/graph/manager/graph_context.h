/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_MANAGER_GRAPH_CONTEXT_H_
#define GE_GRAPH_MANAGER_GRAPH_CONTEXT_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
class GraphContext;

using SessionId = uint64_t;

using GradOpList = std::vector<std::pair<GraphId, std::string>>;

using VariableRecord = std::tuple<std::string, GradOpList, uint8_t>;

using OutputOpNameIndex = std::pair<std::string, uint8_t>;

using VarNodeTensorTable = std::vector<std::pair<VariableRecord, GeTensor>>;

using SessionVarTableMap = std::map<ge::SessionId, VarNodeTensorTable>;

using GraphContextPtr = std::shared_ptr<GraphContext>;

struct OutputDescInfo {
  std::string op_name;
  uint8_t index = 0U;
  struct InputOutputDescInfo info;
};

///
/// @ingroup graph
/// @brief Global graph context sharing, provide variable sharing facility for
///        multiple graphs in the same session.
/// @author
///
class GraphContext {
 public:
  GraphContext() = default;

  ~GraphContext() = default;

  Status Initialize(const std::map<std::string, std::string> &options = {}) const;
  // Disable copy constructor and assignment operator
  GraphContext(const GraphContext &) = delete;

  GraphContext &operator=(const GraphContext &) = delete;

  Status Finalize() const;

  Status GetVariableTensor(const std::string &var_data_name, GeTensor &returned_tensor) const;

  const ComputeGraphPtr &GetComputeGraph() const { return compute_graph_; }

  Status SetComputeGraph(const GraphNodePtr &graph_node);

 private:
  explicit GraphContext(const GraphNodePtr &graph_node);

  ComputeGraphPtr compute_graph_ = nullptr;

  GraphId current_graph_id_ = 0;

  // Get the unique VarNode-Tensor table
  static VarNodeTensorTable &GetVarNodeTensorTable() {
    static VarNodeTensorTable _this;
    return _this;
  }
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_GRAPH_CONTEXT_H_
