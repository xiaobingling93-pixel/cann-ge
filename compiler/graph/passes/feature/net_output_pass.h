/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_NET_OUTPUT_PASS_H_
#define GE_GRAPH_PASSES_NET_OUTPUT_PASS_H_

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "graph/types.h"
#include "graph/passes/graph_pass.h"

namespace ge {


class NetOutputPass : public GraphPass {
 public:
  ///
  /// Entry of the NetOutputPass optimizer
  /// @param [in] graph: Input ComputeGraph
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status Run(ge::ComputeGraphPtr graph) override;

  ///
  /// @brief Clear Status, used for subgraph pass
  /// @return SUCCESS
  ///
  Status ClearStatus() override;

 private:
  ///
  /// Add user_def_dtype & format for netoutput node
  /// @param [in] output_node: The netOutput node
  /// @return SUCCESS: Execution succeed
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status SetUserDefDTypeAndFormatFromAtcParams(const ge::NodePtr &output_node) const;

  Status TryToSetOutputNodeName(const NodePtr &output_node) const;

  Status TryToSetOutputMaxSize(const NodePtr &output_node) const;

  Status AddCtrlEdgesBetweenLeafAndNetOutput(const ComputeGraphPtr &compute_graph,
                                             const ge::NodePtr &net_out_node) const;
  Status SetNetOutputFormat(const ge::NodePtr &net_output) const;

  friend class ReUpdateNetOutputPass;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_NET_OUTPUT_PASS_H_
